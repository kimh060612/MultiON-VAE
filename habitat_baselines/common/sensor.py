r"""
Code from FAIR's repository:
https://github.com/facebookresearch/OccupancyAnticipation
"""

from typing import Optional, Type, Any

import habitat
from habitat import Config, Sensor, Simulator, SensorTypes
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.registry import registry
import numpy as np
import cv2
from einops import asnumpy, rearrange
from gym import spaces

@registry.register_sensor(name="GTEgoMap")
class GTEgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_scale = self.config.MAP_SCALE
        if self.config.MAX_SENSOR_RANGE > 0:
            self.max_forward_range = self.config.MAX_SENSOR_RANGE
        else:
            self.max_forward_range = self.map_size * self.map_scale

        # Agent height for pointcloud tranforms
        self.camera_height = self._sim.habitat_config.DEPTH_SENSOR.POSITION[1]

        # Compute intrinsic matrix
        depth_H = self._sim.habitat_config.DEPTH_SENSOR.HEIGHT
        depth_W = self._sim.habitat_config.DEPTH_SENSOR.WIDTH
        hfov = float(self._sim.habitat_config.DEPTH_SENSOR.HFOV) * np.pi / 180
        vfov = 2 * np.arctan((depth_H / depth_W) * np.tan(hfov / 2.0))
        self.intrinsic_matrix = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1, 0],
                [0.0, 0.0, 0, 1],
            ]
        )
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.habitat_config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.habitat_config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.habitat_config.DEPTH_SENSOR.WIDTH
        H = self._sim.habitat_config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
            np.linspace(-1, 1, W), np.linspace(1, -1, H)
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_map_gt"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 2)
        return spaces.Box(low=0, high=1, shape=sensor_shape, dtype=np.uint8,)

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array

        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        valid_depths = (depth_float != self.min_depth) & (
            depth_float <= self.max_forward_range
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T  # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self._sim.habitat_config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the pointcloud
        XYZ_ego[:, 1] += self.camera_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2

        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_scale) + Vby2
        grid_y = (points[:, 2] / self.map_scale) + V

        # Filter out invalid points
        valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
        )
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        sim_depth = asnumpy(observations["depth"])
        ego_map_gt = self._get_depth_projection(sim_depth)

        return ego_map_gt