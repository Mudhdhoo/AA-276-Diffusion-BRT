import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BRTDataset(Dataset):
    """Dataset for BRT point clouds and environments"""
    def __init__(self, dataset_dir):
        """
        dataset_dir: Path to the dataset directory containing sample_* folders
        """
        self.dataset_dir = dataset_dir
        self.sample_dirs = []
        
        # Find all sample directories and verify they have required files
        for d in sorted(os.listdir(dataset_dir)):
            if d.startswith('sample_'):
                point_cloud_path = os.path.join(dataset_dir, d, 'point_cloud.npy')
                env_grid_path = os.path.join(dataset_dir, d, 'environment_grid.npy')
                
                if os.path.exists(point_cloud_path) and os.path.exists(env_grid_path):
                    self.sample_dirs.append(d)
                else:
                    print(f"Warning: Skipping {d} due to missing files")
        
        if not self.sample_dirs:
            raise ValueError("No valid samples found in dataset directory")
        
        # Load first sample to determine dimensions
        first_point_cloud = np.load(os.path.join(dataset_dir, self.sample_dirs[0], 'point_cloud.npy'))
        first_env = np.load(os.path.join(dataset_dir, self.sample_dirs[0], 'environment_grid.npy'))
        
        self.num_points = first_point_cloud.shape[0]  # N points
        self.state_dim = first_point_cloud.shape[1]   # 3D coordinates
        self.env_size = first_env.shape[0]            # Environment grid size
        
        # Compute normalization statistics
        self.compute_normalization_stats()
        
        print(f"Found {len(self.sample_dirs)} valid samples")
        
    def compute_normalization_stats(self):
        """Compute mean and std for point cloud normalization"""
        all_points = []
        for d in self.sample_dirs:
            points = np.load(os.path.join(self.dataset_dir, d, 'point_cloud.npy'))
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
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # Load point cloud and environment
        point_cloud = np.load(os.path.join(self.dataset_dir, sample_dir, 'point_cloud.npy'))
        env_grid = np.load(os.path.join(self.dataset_dir, sample_dir, 'environment_grid.npy'))
        
        # Convert to torch tensors and normalize point cloud
        point_cloud = torch.FloatTensor(self.normalize_points(point_cloud))
        env_grid = torch.FloatTensor(env_grid)
        
        return point_cloud, env_grid