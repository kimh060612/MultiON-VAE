from typing import Dict, Tuple
from einops import rearrange
import torch.nn.functional as F
import torch_scatter
import torch
import numpy as np
import math

"""
Code adapted from https://github.com/saimwani/multiON
"""

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid

class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).floor()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).floor()
        return grid_x, grid_y
    
    def get_gps_coords(self, idx):
        # H, W indices to gps coordinates
        grid_x = idx[0].item()
        grid_y = idx[1].item()

        gps_x = self.coordinate_max - grid_x * self.grid_size
        gps_y = self.coordinate_min + grid_y * self.grid_size

        return gps_x, gps_y

class ComputeSpatialLocs():
    def __init__(self, egocentric_map_size, global_map_size, 
        device, coordinate_min, coordinate_max
    ):
        self.device = device
        self.cx, self.cy = 256./2., 256./2.     # Hard coded camera parameters
        self.fx = self.fy =  (256. / 2.) / np.tan(np.deg2rad(79 / 2.))
        self.egocentric_map_size = egocentric_map_size
        self.local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
        
    def forward(self, depth) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        x    = rearrange(torch.arange(0, imw), 'w -> () () () w').to(self.device)
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        xx   = (x - self.cx) / self.fx
        yy   = (y - self.cy) / self.fy
        # 3D real-world coordinates (in meters)
        Z = depth
        X = xx * Z
        Y = yy * Z
        
        # X ground projection and Y ground projection
        x_gp = ( (X / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, 1, imh, imw)
        y_gp = (-(Z / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, 1, imh, imw)
        
        return torch.cat([x_gp, y_gp], dim=1), Y

class ProjectToGroundPlane():
    def __init__(self, egocentric_map_size, device, 
            vaccant_bel, occupied_bel, 
            height_min, height_max
        ):
        self.egocentric_map_size = egocentric_map_size
        self.device = device
        self.vaccant_bel = vaccant_bel
        self.occupied_bel = occupied_bel
        self.height_min = height_min
        self.height_max = height_max

    def forward(self, img, spatial_locs):
        (outh, outw) = (self.egocentric_map_size, self.egocentric_map_size)
        bs, f, HbyK, WbyK = img.shape
        K = 1
        # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
        idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(self.device), \
                    (torch.arange(0, WbyK, 1)*K).long().to(self.device))

        spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
        
        # Filter out invalid spatial locations
        invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                            (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)
        
        # Set the idxes for all invalid locations to (0, 0)
        spatial_locs_ss[:, 0][invalid_spatial_locs] = 0
        spatial_locs_ss[:, 1][invalid_spatial_locs] = 0

        # Linearize ground-plane indices (linear idx = y * W + x)
        linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
        linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
        linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()
        img_target = rearrange(img, 'b e h w -> b e (h w)')
        
        proj_feats, _ = torch_scatter.scatter_min(
            img_target,
            linear_locs_ss,
            dim=2,
            dim_size=outh*outw
        )
        proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)
        # Valid inputs
        occupied_area = (proj_feats != 0) & ((proj_feats > self.height_min) & (proj_feats < self.height_max))
        vaccant_area = (proj_feats != 0) & (proj_feats < self.height_min)
        
        # The belief image for projection
        belief_map = torch.zeros_like(img)
        belief_map[occupied_area] = self.occupied_bel
        belief_map[vaccant_area] = self.vaccant_bel
        
        return belief_map

class Projection:
    def __init__(self, 
            egocentric_map_size, global_map_size, device, 
            coordinate_min, coordinate_max, vaccant_bel, occupied_bel, 
            height_min, height_max
        ):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputeSpatialLocs(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )
        self.project_to_ground_plane = ProjectToGroundPlane(egocentric_map_size, device, 
            vaccant_bel, occupied_bel,
            height_min, height_max
        )

    def forward(self, depth) -> torch.Tensor:
        spatial_locs, height_map = self.compute_spatial_locs.forward(depth)
        ego_local_map = self.project_to_ground_plane.forward(height_map, spatial_locs)
        return ego_local_map
    
class Registration():
    def __init__(self, egocentric_map_size, global_map_size, coordinate_min, coordinate_max, num_process, device, global_map_depth = 1):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth
        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.num_obs = num_process
        self.device = device
    
    def forward(
            self, 
            observations: Dict[str, torch.Tensor], 
            global_allocentric_map: torch.Tensor, 
            egocentric_map: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Register egocentric_map to full_global_map

        Args:
            observations: Dictionary containing habitat observations
                - Position: observations['gps']
                - Heading: observations['compass']
            global_allocentric_map: torch.tensor containing global map, (num_obs, global_map_depth=1, global_map_size, global_map_size) 
            egocentric_map: torch.tensor containing egocentrc map, (num_obs, global_map_depth=1, egocentric_map_size, egocentric_map_size) 

        Returns:
            registered_map: torch.tensor containing registered map, (num_obs, global_map_depth=1, global_map_size, global_map_size)
            egocentric_global_map: torch.tensor containing registered map, (num_obs, global_map_depth=1, global_map_size, global_map_size)
        """
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        global_allocentric_map = global_allocentric_map.to(self.device)
        bs = egocentric_map.shape[0]
        
        if bs != self.num_obs:
            global_allocentric_map[bs:, ...] = global_allocentric_map[bs:, ...] * 0.
        
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
        else:
            agent_view = torch.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)

        agent_view[:, :, 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ] = egocentric_map
        
        agent_egocentric_view = agent_view.clone().detach()
        global_allocentric_copy = rearrange(global_allocentric_map.clone().detach(), 'b h w -> b () h w')

        st_pose = torch.cat(
            [
                -(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                observations['compass']
            ], 
            dim=1
        ).to(self.device)
        st_pose_inverse = torch.cat(
            [
                (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                -observations['compass']
            ],
            dim=1
        ).to(self.device)
        
        # Generate warpping matrix for pytorch
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        rot_inverse_mat, trans_inverse_mat = get_grid(st_pose_inverse, global_allocentric_copy[:bs, ...].size(), self.device)
        
        # Warpping for global allocentric map
        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)
        registered_map = global_allocentric_map[:bs, ...].clone().detach().unsqueeze(dim=1) + translated # .permute(0, 2, 3, 1)
        
        # Warpping for global egocentric map
        egocentric_rotated = F.grid_sample(global_allocentric_copy[:bs, ...], rot_inverse_mat)
        egocentric_translated = F.grid_sample(egocentric_rotated, trans_inverse_mat)
        egocentric_global_map = agent_egocentric_view + egocentric_translated # .permute(0, 2, 3, 1)
        
        return registered_map.squeeze(dim=1), egocentric_global_map.squeeze(dim=1)
    
class OccupancyMap():
    
    def __init__(self, 
            global_map_size, egocentric_map_size, num_process,
            device, coordinate_min, coordinate_max, vaccant_bel, occupied_bel, 
            height_min=-0.7, height_max=3.0
        ):
        self.num_process = num_process
        self.global_map_size = global_map_size
        self.egocentric_map_size = egocentric_map_size
        self.BEL_VAC = vaccant_bel
        self.BEL_OCC = occupied_bel
        ## global egocentric/allocentric map: (!!!) This is belief map, not the probability
        self.global_allocentric_occupancy_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size
        )
        self.global_egocentric_occupancy_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size
        )
        self.occupancy_projection = Projection(self.egocentric_map_size, self.global_map_size, device, 
                    coordinate_min, coordinate_max, self.BEL_VAC, self.BEL_OCC, 
                    height_min, height_max
        )
        self.registration = Registration(self.egocentric_map_size, self.global_map_size, coordinate_min, coordinate_max, self.num_process, device)

    def _belief_to_prob(self, x: torch.Tensor) -> torch.Tensor:
        return 1. - 1./(1 + torch.exp(x))

    def get_current_global_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Transform the belief map toward probability map
            (???????: Is it better to train the belief map?)
            
            Return:
                - Global Allocentric Map \in (batch size, channel=1, global map size, global map size)
                - Global Egocentric Map \in (batch size, channel=1, global map size, global map size)
            Map contents 
            M_{i,j} = 0: Vaccant
            M_{i,j} = 1: Occupied
            Otherwise: uncertainty between them.
        '''
        gaom = self._belief_to_prob(self.global_allocentric_occupancy_map)
        geom = self._belief_to_prob(self.global_egocentric_occupancy_map)
        return gaom, geom

    def update_map(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = observations["depth"].clone().detach()
        
        # Project to 2D map & Generate 2D local egocentric map
        projection = self.occupancy_projection.forward(depth * 10.)
        # Update global allocentric map & Get global egocentric map
        (
            self.global_allocentric_occupancy_map, 
            self.global_egocentric_occupancy_map
        ) = self.registration.forward(observations, self.global_allocentric_occupancy_map, projection)
        
class OccupancyMapRollout():
    def __init__(self, global_map_size, egocentric_map_size, num_process, num_steps,
            device, coordinate_min, coordinate_max, vaccant_bel, occupied_bel, 
            height_min=-0.7, height_max=3.0
        ):
        self.num_process = num_process
        self.global_map_size = global_map_size
        self.egocentric_map_size = egocentric_map_size
        self.BEL_VAC = vaccant_bel
        self.BEL_OCC = occupied_bel
        self.num_steps = num_steps
        self.step = 0
        self.device = device
        ## global egocentric/allocentric map: (!!!) This is belief map, not the probability
        self.global_allocentric_occupancy_map = torch.zeros(
            self.num_steps + 1,
            self.num_process,
            self.global_map_size,
            self.global_map_size,
            device=device
        )
        self.global_egocentric_occupancy_map = torch.zeros(
            self.num_steps + 1,
            self.num_process,
            self.global_map_size,
            self.global_map_size,
            device=device
        )
        self.occupancy_projection = Projection(
            self.egocentric_map_size, self.global_map_size, device, 
            coordinate_min, coordinate_max, self.BEL_VAC, self.BEL_OCC, 
            height_min, height_max
        )
        self.registration = Registration(
            self.egocentric_map_size, self.global_map_size, coordinate_min, coordinate_max, self.num_process, device
        )
    
    def __getitem__(self, index):
        geom = self._belief_to_prob(self.global_egocentric_occupancy_map)
        return geom[index, ...]
    
    def _belief_to_prob(self, x: torch.Tensor) -> torch.Tensor:
        return 1. - 1./(1 + torch.exp(x))
    
    def get_current_global_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Transform the belief map toward probability map
            
            Return:
                - Global Allocentric Map \in (batch size, channel=1, global map size, global map size)
                - Global Egocentric Map \in (batch size, channel=1, global map size, global map size)
            Map contents 
            M_{i,j} = 0: Vaccant
            M_{i,j} = 1: Occupied
            Otherwise: uncertainty between them.
        '''
        gaom = self._belief_to_prob(self.global_allocentric_occupancy_map)
        geom = self._belief_to_prob(self.global_egocentric_occupancy_map)
        return gaom, geom
    
    def after_update(self):
        self.global_allocentric_occupancy_map[0].copy_(
            self.global_allocentric_occupancy_map[self.step]
        )
        self.global_egocentric_occupancy_map[0].copy_(
            self.global_egocentric_occupancy_map[self.step]
        )
        self.step = 0
    
    def update_map(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = observations["depth"].clone().detach().to(self.device)
        
        # Project to 2D map & Generate 2D local egocentric map
        projection = self.occupancy_projection.forward(depth * 10.)
        # Update global allocentric map & Get global egocentric map
        (
            self.global_allocentric_occupancy_map[self.step + 1], 
            self.global_egocentric_occupancy_map[self.step + 1]
        ) = self.registration.forward(observations, self.global_allocentric_occupancy_map[self.step], projection)
        self.step += 1
    
    