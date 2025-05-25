import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BRTDataset(Dataset):
    """Dataset for BRT point clouds and environments"""
    def __init__(self, dataset_dir, split="train"):
        """
        Args:
            dataset_dir: Directory containing the dataset
            split: One of "train", "val", or "test"
        """
        self.dataset_dir = dataset_dir
        self.split = split
        
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
        
        # Compute normalization statistics
        self.compute_normalization_stats()
        
        print(f"Found {len(self.sample_dirs)} environments with {len(self.point_cloud_files)} total point clouds in {split} split")
        
    def compute_normalization_stats(self):
        """Compute mean and std for point cloud normalization"""
        all_points = []
        for sample_dir, pc_file in self.point_cloud_files:
            points = np.load(os.path.join(self.dataset_dir, sample_dir, pc_file))
            all_points.append(points)
        
        all_points = np.concatenate(all_points, axis=0)
        self.points_mean = np.mean(all_points, axis=0)
        self.points_std = np.std(all_points, axis=0)
        
        # Ensure std is not zero
        self.points_std = np.maximum(self.points_std, 1e-6)
        
        print("Point cloud normalization stats:")
        print(f"Mean: {self.points_mean}")
        print(f"Std: {self.points_std}")
    
    def normalize_points(self, points):
        """Normalize point cloud coordinates"""
        return (points - self.points_mean) / self.points_std
    
    def denormalize_points(self, points):
        """Denormalize point cloud coordinates"""
        return points * self.points_std + self.points_mean
    
    def __len__(self):
        return len(self.point_cloud_files)
    
    def __getitem__(self, idx):
        sample_dir, pc_file = self.point_cloud_files[idx]
        
        # Load point cloud and environment
        point_cloud = np.load(os.path.join(self.dataset_dir, sample_dir, pc_file))
        env_grid = np.load(os.path.join(self.dataset_dir, sample_dir, 'environment_grid.npy'))

        # Convert to torch tensors and normalize point cloud
        point_cloud = torch.FloatTensor(self.normalize_points(point_cloud))
        env_grid = torch.FloatTensor(env_grid)
        
        return point_cloud, env_grid