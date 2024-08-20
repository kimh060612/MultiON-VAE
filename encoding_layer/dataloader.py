import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from glob import glob

class MapDataset(Dataset):
    def __init__(self, data_dir, device):
        super(MapDataset, self).__init__()
        self.map_index = pd.read_csv(f"{data_dir}/map_pose_info.csv", header=None)
        self.map_index.columns = ["id", "pose"]
        self.data_dir = data_dir
        self.device = device
        self.coord_max = (275 // 2) + 128
        self.coord_min = (275 // 2) - 128
        
    def __len__(self):
        return len(self.map_index)
    
    def __getitem__(self, index):
        map_id = self.map_index.iloc[index]["id"]
        pose = np.array([ float(x) for x in self.map_index.iloc[index]["pose"][1:-1].split()])
        pose = torch.from_numpy(pose).float().to(self.device)
        ego_map = torch.from_numpy(np.load(f"{self.data_dir}/ego_#id{map_id}.npy")).float().to(self.device)[self.coord_min:self.coord_max, self.coord_min:self.coord_max]
        allo_map = torch.from_numpy(np.load(f"{self.data_dir}/allo_#id{map_id}.npy")).float().to(self.device)[self.coord_min:self.coord_max, self.coord_min:self.coord_max]
        return pose, ego_map.unsqueeze(0), allo_map.unsqueeze(0)