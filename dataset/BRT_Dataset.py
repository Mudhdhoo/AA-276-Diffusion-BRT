import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import os
from loguru import logger
from dataset.statistics.stats import MEAN_VAL_FUNC, STD_VAL_FUNC, MEAN_ENV_GRID, STD_ENV_GRID
    
class BRT_Dataset(Dataset):
    def __init__(self, data_dir, split="train", padded_grid_size=None, device="cuda"):
        self.data_dir = data_dir
        self.sample_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f != ".DS_Store" and f != "results.csv"]
        self.padded_grid_size = padded_grid_size
        self.device = device

        # Split into train and test
        self.sample_paths = sorted(self.sample_paths)
        if split == "train":
            self.sample_paths = self.sample_paths[:int(0.8*len(self.sample_paths))]
        elif split == "test":
            self.sample_paths = self.sample_paths[int(0.8*len(self.sample_paths)):]

        # Normalize value function and environment grid
        self.transform_val_func = transforms.Normalize(mean=MEAN_VAL_FUNC,
                                                       std=STD_VAL_FUNC)
        self.transform_env_grid = transforms.Normalize(mean=MEAN_ENV_GRID,
                                                       std=STD_ENV_GRID)

        logger.info(f"Loaded {len(self.sample_paths)} BRT samples.")

    def pad_value_function(self, val_func):
        val_func_last = val_func[-1]  # Get the last time step
        h, w, c = val_func_last.shape  # height, width, channels

        padded_val_func = torch.zeros(self.padded_grid_size, self.padded_grid_size, c)
        
        # Copy the original data
        padded_val_func[:min(h, self.grid_size), :min(w, self.grid_size), :] = val_func_last[:min(h, self.grid_size), :min(w, self.grid_size), :]
        
        # Pad by copying border values
        if h < self.grid_size:
            # Copy last row to fill remaining rows
            padded_val_func[h:, :min(w, self.grid_size), :] = val_func_last[h-1:h, :min(w, self.grid_size), :].repeat(self.grid_size-h, 1, 1)
        
        if w < self.grid_size:
            # Copy last column to fill remaining columns
            padded_val_func[:, w:, :] = padded_val_func[:, w-1:w, :].repeat(1, self.grid_size-w, 1)
        
        return padded_val_func


    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        val_func = np.load(f"{sample_path}/value_function.npy")
        env_grid = np.load(f"{sample_path}/environment_grid.npy")

        val_func = torch.from_numpy(val_func).float()
        env_grid = torch.from_numpy(env_grid).float()

        # Get current dimensions
        val_func_last = val_func[-1]  # Get the last time step
        
        if self.padded_grid_size is not None:
            val_func_last = self.pad_value_function(val_func_last)
        
            # For env_grid (assuming 2D)
            if env_grid.shape[0] < self.padded_grid_size or env_grid.shape[1] < self.padded_grid_size:
                padded_env_grid = torch.zeros(self.padded_grid_size, self.padded_grid_size)
                padded_env_grid[:env_grid.shape[0], :env_grid.shape[1]] = env_grid
            env_grid = padded_env_grid

        # Set theta as channel dimension
        val_func_last = val_func_last.permute(2, 0, 1)
        env_grid = env_grid.unsqueeze(0)

        val_func_last = self.transform_val_func(val_func_last)
        env_grid = self.transform_env_grid(env_grid)

        return val_func_last, env_grid
    
    def sample_env_grid(self, n):
        """
        Sample n random environment grids from the dataset.
        
        Args:
            n (int): Number of environment grids to sample
            
        Returns:
            torch.Tensor: Tensor of shape (n, 1, H, W) containing n sampled environment grids
        """
        indices = torch.randint(0, len(self), (n,))
        env_grids = []
        
        for idx in indices:
            _, env_grid = self[idx]
            env_grids.append(env_grid)
            
        return torch.stack(env_grids)

if __name__ == "__main__":
    dataset = BRT_Dataset(data_dir="/Users/johncao/Documents/Programming/Stanford/AA276/project/dataset_64",
                          device="cpu")
    
    print(len(dataset))
    sample = dataset[0]
    print(sample[0].shape)
    print(sample[1].shape)