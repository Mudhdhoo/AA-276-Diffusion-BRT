"""
Load a checkpoint and visualize the reverse process.

The reverse process is computed by sampling from the model and then denoising the point cloud.

The denoising process is visualized by plotting the point cloud at each step.

The true point cloud is also plotted for comparison.

"""

from utils.visualizations import visualize_comparison, visualize_denoising_with_true
from dataset.BRTDataset import BRTDataset
from models.diffusion_modules import BRTDiffusionModel
import torch
from tqdm import tqdm
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import random
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_dir = "../1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv"  # Update this path as needed
    dataset = BRTDataset(dataset_dir, split="val")
    
    # Create model
    model = BRTDiffusionModel(
        state_dim=dataset.state_dim,
        env_size=dataset.env_size,
        num_points=dataset.num_points,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.008,
        device=device
    )
    
    # Load checkpoint
    checkpoint_path = "checkpoints/lucky_moon_21/checkpoint_epoch_2000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Explicitly set weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    sample_idx = 23
    point_cloud, env_grid, *_ = dataset[sample_idx]  # Use wildcard to handle extra return values
    point_cloud = point_cloud.to(model.device)
    env_grid = env_grid.to(model.device)
    
    # Start from pure noise
    x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)
    
    # Save exactly 5 steps total
    num_steps = 5  # Total number of steps to visualize
    step_indices = np.linspace(0, model.num_timesteps-1, num_steps, dtype=int)

    points_sequence = []
    titles = []

    # Compute full reverse process
    for t in tqdm(reversed(range(model.num_timesteps)), desc="Denoising"):
        with torch.no_grad():
            t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
            x_t = model.p_sample(x_t, t_batch, env_grid, guidance_scale=1.7)

        if t in step_indices:
            points_sequence.append(x_t[0].cpu().numpy())
            titles.append(f't={t}')


    visualize_denoising_with_true(points_sequence, 
                                  point_cloud.cpu().numpy(), 
                                  titles, 
                                  save_path="reverse_process.png", 
                                  dataset=dataset)

if __name__ == "__main__":
    main()