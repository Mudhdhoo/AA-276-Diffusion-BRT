import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from loguru import logger

class BRT_Dataset(Dataset):
    def __init__(self, data_dir, grid_size=128, num_samples=300, device="cuda"):
        self.data_dir = data_dir
        sample_paths = [os.path.join(data_dir, "10_May_2025_05_33", f) for f in os.listdir(os.path.join(data_dir, "10_May_2025_05_33"))]
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.device = device
        self.sample_paths = sample_paths[:num_samples]

        logger.info(f"Loaded {len(self.sample_paths)} BRT samples.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        val_func = np.load(f"{sample_path}/value_function.npy")
        env_grid = np.load(f"{sample_path}/environment_grid.npy")

        val_func = torch.from_numpy(val_func).float()#.to(self.device)
        env_grid = torch.from_numpy(env_grid).float()#.to(self.device)

        # Get current dimensions
        val_func_last = val_func[-1]  # Get the last time step
        
        # Pad the value function (3D tensor)
        h, w, c = val_func_last.shape  # height, width, channels
        
        if h < self.grid_size or w < self.grid_size:
            padded_val_func = torch.zeros(self.grid_size, self.grid_size, c)
            
            # Copy the original data
            padded_val_func[:min(h, self.grid_size), :min(w, self.grid_size), :] = val_func_last[:min(h, self.grid_size), :min(w, self.grid_size), :]
            
            # Pad by copying border values
            if h < self.grid_size:
                # Copy last row to fill remaining rows
                padded_val_func[h:, :min(w, self.grid_size), :] = val_func_last[h-1:h, :min(w, self.grid_size), :].repeat(self.grid_size-h, 1, 1)
            
            if w < self.grid_size:
                # Copy last column to fill remaining columns
                padded_val_func[:, w:, :] = padded_val_func[:, w-1:w, :].repeat(1, self.grid_size-w, 1)
            
            val_func_last = padded_val_func
        
        # For env_grid (assuming 2D)
        if env_grid.shape[0] < self.grid_size or env_grid.shape[1] < self.grid_size:
            padded_env_grid = torch.zeros(self.grid_size, self.grid_size)
            padded_env_grid[:env_grid.shape[0], :env_grid.shape[1]] = env_grid
        env_grid = padded_env_grid

        # Set theta as channel dimension
        val_func_last = val_func_last.permute(2, 0, 1)
        env_grid = env_grid.unsqueeze(0)

        return val_func_last, env_grid
    

if __name__ == "__main__":
    dataset = BRT_Dataset(data_dir="/Users/johncao/Documents/Programming/Stanford/AA276/project/outputs",
                          device="cpu")
    
    print(len(dataset))
    sample = dataset[0]
    print(sample["val_func"].shape)
    print(sample["env_grid"].shape)