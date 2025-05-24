import torch
import numpy as np
from dataset.BRT_Dataset import BRT_Dataset
from tqdm import tqdm
import os

def compute_dataset_statistics(data_dir, num_samples=None, device="cpu"):
    """
    Compute mean and variance of the BRT dataset.
    For value function, computes per-channel statistics.
    For environment grid, computes overall statistics (since it has only 1 channel).
    
    Args:
        data_dir (str): Directory containing the dataset
        num_samples (int, optional): Number of samples to use. If None, use all samples
        device (str): Device to use for computation
    
    Returns:
        tuple: (mean_val_func, std_val_func, mean_env_grid, std_env_grid)
            mean_val_func: shape (C,) - per-channel means
            std_val_func: shape (C,) - per-channel standard deviations
            mean_env_grid: shape (1,H,W) - overall mean
            std_env_grid: shape (1,H,W) - overall standard deviation
    """
    # Initialize dataset
    dataset = BRT_Dataset(data_dir=data_dir, num_samples=num_samples, device=device)
    
    C,H,W = dataset[0][0].shape
    N = len(dataset)

    mean_val_func = torch.zeros(C,H,W)
    std_val_func = torch.zeros(C,H,W)
    mean_env_grid = torch.zeros(1,H,W)
    std_env_grid = torch.zeros(1,H,W)
    
    # Compute mean
    for i in tqdm(range(N), desc="Computing mean"):
        val_func, env_grid = dataset[i]
        mean_val_func += val_func
        mean_env_grid += env_grid

    # Per channel mean
    mean_val_func /= N
    mean_env_grid /= N
    mean_val_func = mean_val_func.mean(dim=(1,2))
    mean_env_grid = mean_env_grid.mean(dim=(1,2))

    # Compute std
    for i in tqdm(range(N), desc="Computing std"):
        val_func, env_grid = dataset[i]
        std_val_func += (val_func - mean_val_func[:,None,None])**2
        std_env_grid += (env_grid - mean_env_grid[:,None,None])**2
        
    # Per channel std
    std_val_func /= N
    std_env_grid /= N
    std_val_func = torch.sqrt(std_val_func.mean(dim=(1,2)))
    std_env_grid = torch.sqrt(std_env_grid.mean(dim=(1,2)))

    
    return mean_val_func, std_val_func, mean_env_grid, std_env_grid

def save_statistics(mean_val_func, std_val_func, mean_env_grid, std_env_grid, output_dir):
    """Save the computed statistics to numpy files"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "mean_val_func.npy"), mean_val_func.numpy())
    np.save(os.path.join(output_dir, "std_val_func.npy"), std_val_func.numpy())
    np.save(os.path.join(output_dir, "mean_env_grid.npy"), mean_env_grid.numpy())
    np.save(os.path.join(output_dir, "std_env_grid.npy"), std_env_grid.numpy())

if __name__ == "__main__":
    # Set paths
    data_dir = "dataset_64"  # Directory containing the dataset
    output_dir = "dataset/statistics"  # Directory to save statistics
    
    # Compute statistics
    mean_val_func, std_val_func, mean_env_grid, std_env_grid = compute_dataset_statistics(
        data_dir=data_dir,
        num_samples=None,  # Use all samples
        device="cpu"
    )
    
    # Save statistics
    save_statistics(mean_val_func, std_val_func, mean_env_grid, std_env_grid, output_dir)
    
    print("Statistics computed and saved successfully!")
    print(f"Value function mean shape: {mean_val_func.shape}")
    print(f"Value function std shape: {std_val_func.shape}")
    print(f"Environment grid mean shape: {mean_env_grid.shape}")
    print(f"Environment grid std shape: {std_env_grid.shape}")
