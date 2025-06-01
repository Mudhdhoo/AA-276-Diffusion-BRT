import os
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
from stats.train_stats import MEAN, STD

class BRTDataset(Dataset):
    """Dataset for BRT point clouds and environments"""
    def __init__(self, dataset_dir, split="train", return_value_function=False):
        """
        Args:
            dataset_dir: Directory containing the dataset
            split: One of "train", "val", or "test"
            return_value_function: If True, also returns the converged value function (last time slice)
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.return_value_function = return_value_function
        
        # Get all sample directories
        self.sample_dirs = sorted([d for d in os.listdir(dataset_dir) if d.startswith('sample_')])
        
        # Split into train/val/test (80/10/10)
        n_samples = len(self.sample_dirs)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
        
        if split == "train":
            self.sample_dirs = self.sample_dirs[:n_train]
        elif split == "val":
            self.sample_dirs = self.sample_dirs[n_train:n_train + n_val]
        elif split == "test":
            self.sample_dirs = self.sample_dirs[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        # Get all point cloud files
        self.point_cloud_files = []
        for sample_dir in self.sample_dirs:
            pc_files = [f for f in os.listdir(os.path.join(dataset_dir, sample_dir)) if f.startswith('point_cloud_')]
            for pc_file in pc_files:
                self.point_cloud_files.append((sample_dir, pc_file))
        
        # Load first point cloud and environment to get dimensions
        first_sample_dir, first_pc_file = self.point_cloud_files[0]
        first_point_cloud = np.load(os.path.join(dataset_dir, first_sample_dir, first_pc_file))
        first_env = np.load(os.path.join(dataset_dir, first_sample_dir, 'environment_grid.npy'))
        
        self.num_points = first_point_cloud.shape[0]  # N points
        self.state_dim = first_point_cloud.shape[1]   # 3D coordinates
        self.env_size = first_env.shape[0]            # Environment grid size
        
        # Check if value functions are available when requested
        if self.return_value_function:
            first_value_func_path = os.path.join(dataset_dir, first_sample_dir, 'value_function.npy')
            if os.path.exists(first_value_func_path):
                first_value_func = np.load(first_value_func_path)
                self.value_function_shape = first_value_func.shape
                logger.info(f"Value function shape: {self.value_function_shape}")
            else:
                logger.warning(f"Value function requested but not found in {first_sample_dir}. Will return None for missing value functions.")
                self.value_function_shape = None
        
        # Compute normalization statistics
        #self.compute_normalization_stats()
        self.compute_min_max_stats()
        
        # Train set mean and std
        self.points_mean = np.array(MEAN)
        self.points_std = np.array(STD)
        logger.info(f"Mean: {self.points_mean}")
        logger.info(f"Std: {self.points_std}")

        logger.info(f"Found {len(self.sample_dirs)} environments with {len(self.point_cloud_files)} total point clouds in {split} split")


    def compute_min_max_stats(self):
        """Compute min-max stats for point cloud"""
        all_points = []
        for sample_dir, pc_file in self.point_cloud_files:
            points = np.load(os.path.join(self.dataset_dir, sample_dir, pc_file))
            all_points.append(points)
        
        all_points = np.concatenate(all_points, axis=0)
        
        # Step 1: Min-max ranges for initial scaling
        self.points_min = np.array([0.0, 0.0, -np.pi])
        self.points_max = np.array([10.0, 10.0, np.pi])
        
        # For value (4th dimension if it exists), use min-max from dataset
        if all_points.shape[1] == 4:
            self.points_min = np.append(self.points_min, np.min(all_points[:, 3]))
            self.points_max = np.append(self.points_max, np.max(all_points[:, 3]))

        logger.info(f"Min-Max ranges:")
        logger.info(f"Min: {self.points_min}")
        logger.info(f"Max: {self.points_max}")
        
    def compute_normalization_stats(self):
        """Compute normalization stats for point cloud
        First step: min-max scaling per channel to [-1, 1]
        x: [0, 10] -> [-1, 1]
        y: [0, 10] -> [-1, 1]
        theta: [-pi, pi] -> [-1, 1]
        value: min-max from dataset -> [-1, 1]
        
        Second step: fixed mean-std normalization using stats from entire training set
        """
        all_points = []
        for sample_dir, pc_file in self.point_cloud_files:
            points = np.load(os.path.join(self.dataset_dir, sample_dir, pc_file))
            all_points.append(points)
        
        all_points = np.concatenate(all_points, axis=0)
        
        # Step 1: Min-max ranges for initial scaling
        self.points_min = np.array([0.0, 0.0, -np.pi])
        self.points_max = np.array([10.0, 10.0, np.pi])
        
        # For value (4th dimension if it exists), use min-max from dataset
        if all_points.shape[1] == 4:
            self.points_min = np.append(self.points_min, np.min(all_points[:, 3]))
            self.points_max = np.append(self.points_max, np.max(all_points[:, 3]))
        
        # Step 2: Compute fixed mean and std from entire training set
        # First normalize the points to [-1,1] range
        normalized_points = 2.0 * (all_points - self.points_min) / (self.points_max - self.points_min) - 1.0
        self.points_mean = np.mean(normalized_points, axis=0)  # Fixed mean per channel
        self.points_std = np.std(normalized_points, axis=0)    # Fixed std per channel
        # Ensure std is not zero
        self.points_std = np.maximum(self.points_std, 1e-6)
        
        print("Point cloud normalization stats:")
        print(f"Min-Max ranges:")
        print(f"Min: {self.points_min}")
        print(f"Max: {self.points_max}")
        print(f"\nFixed normalization stats (computed from entire training set):")
        print(f"Mean: {self.points_mean}")
        print(f"Std: {self.points_std}")

    
    def normalize_points(self, points):
        """Normalize point cloud coordinates using min-max scaling to [-1,1] followed by fixed mean-std normalization"""
        # First normalize to [-1,1] range
        normalized = 2.0 * (points - self.points_min) / (self.points_max - self.points_min) - 1.0
        # Then apply fixed mean-std normalization

        #return normalized
        return (normalized - self.points_mean) / self.points_std
    
    def denormalize_points(self, points):
        """Denormalize point cloud coordinates"""
        # First undo fixed mean-std normalization
        normalized = points * self.points_std + self.points_mean
        # Then undo min-max scaling from [-1,1] back to original range
        return (normalized + 1.0) / 2.0 * (self.points_max - self.points_min) + self.points_min
    
    def __len__(self):
        return len(self.point_cloud_files)
    
    def __getitem__(self, idx):
        sample_dir, pc_file = self.point_cloud_files[idx]
        
        # Load point cloud and environment
        point_cloud = np.load(os.path.join(self.dataset_dir, sample_dir, pc_file))
        env_grid = np.load(os.path.join(self.dataset_dir, sample_dir, 'environment_grid.npy'))

        point_cloud = self.normalize_points(point_cloud)

        point_cloud = torch.FloatTensor(point_cloud)
        env_grid = torch.FloatTensor(env_grid)
        
        # Optionally load value function
        if self.return_value_function:
            value_func_path = os.path.join(self.dataset_dir, sample_dir, 'value_function.npy')
            if os.path.exists(value_func_path):
                value_function = np.load(value_func_path)
                # Return the last time slice (converged value function)
                converged_value_func = torch.FloatTensor(value_function[-1])
                return point_cloud, env_grid, converged_value_func
            else:
                # Return None if value function is missing
                return point_cloud, env_grid, None
        
        return point_cloud, env_grid