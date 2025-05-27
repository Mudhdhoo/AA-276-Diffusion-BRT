import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion_modules import BRTDiffusionModel
from dataset.BRTDataset import BRTDataset

def visualize_forward_process(model, x_start, env, dataset, timesteps_to_show=None, save_path="plots/forward_process.png"):
    """Visualize the forward diffusion process at various timesteps using raw coordinate values"""
    if timesteps_to_show is None:
        # Show process at these specific timesteps
        timesteps_to_show = [0, 100, 300, 500, 700, 900, 999]
    
    num_timesteps = len(timesteps_to_show)
    fig = plt.figure(figsize=(20, 12))  # Increased height for 3 rows
    
    # x_start is normalized data from dataset, convert to torch tensor
    x_start_torch = torch.FloatTensor(x_start).unsqueeze(0).to(model.device)
    
    for i, t in enumerate(timesteps_to_show):
        if t == 0:
            # Original data - denormalize to get raw values
            x_t_raw = dataset.denormalize_points(x_start)
            x_t_norm = x_start  # Normalized version
            title = f"Original (t=0)"
        else:
            # Apply forward diffusion to normalized data
            t_tensor = torch.tensor([t]).to(model.device)
            x_t_torch = model.q_sample(x_start_torch, t_tensor)
            x_t_norm = x_t_torch.squeeze(0).cpu().numpy()  # Normalized version
            # Denormalize to get raw coordinate values
            x_t_raw = dataset.denormalize_points(x_t_norm)
            title = f"t={t}"
        
        # Row 1: Scaled coordinates (environment-aligned)
        ax1 = fig.add_subplot(3, num_timesteps, i + 1, projection='3d')
        
        # Scale coordinates to match environment dimensions
        x = x_t_raw[:, 0] * (10.0 / 64.0)  # Scale x from [0,64] to [0,10]
        y = x_t_raw[:, 1] * (10.0 / 64.0)  # Scale y from [0,64] to [0,10]
        theta = x_t_raw[:, 2] * (2 * np.pi / 64.0) - np.pi  # Scale theta from [0,64] to [-π,π]
        
        # Plot using scaled coordinate values
        ax1.scatter(x, y, theta, c=theta, cmap='viridis', s=8, alpha=0.7)
        ax1.set_title(f"{title} (Scaled)")
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('θ (rad)')
        # Set axis limits to match environment dimensions
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_zlim(-np.pi, np.pi)
        
        # Row 2: Normalized coordinates (what model sees)
        ax2 = fig.add_subplot(3, num_timesteps, i + 1 + num_timesteps, projection='3d')
        ax2.scatter(x_t_norm[:, 0], x_t_norm[:, 1], x_t_norm[:, 2], 
                   c=x_t_norm[:, 2], cmap='viridis', s=8, alpha=0.7)
        ax2.set_title(f"{title} (Normalized)")
        ax2.set_xlabel('X (normalized)')
        ax2.set_ylabel('Y (normalized)')
        ax2.set_zlabel('Z (normalized)')
        
        # Row 3: Environment plot
        ax3 = fig.add_subplot(3, num_timesteps, i + 1 + 2*num_timesteps)
        im = ax3.imshow(env, cmap='RdYlBu_r', origin='lower', extent=[0, 10, 0, 10])
        ax3.set_title('Environment')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        
        if i == num_timesteps - 1:  # Add colorbar to last subplot
            plt.colorbar(im, ax=ax3, shrink=0.8, label='Obstacle (1) / Free Space (0)')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the forward process visualization"""
    print("Creating simplified BRT Diffusion Forward Process Visualization...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_dir = "../1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv"
    dataset = BRTDataset(dataset_dir)
    print(f"Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    print(f"Point cloud shape: ({dataset.num_points}, {dataset.state_dim})")
    
    # Create model
    model = BRTDiffusionModel(
        state_dim=dataset.state_dim,
        env_size=dataset.env_size,
        num_points=dataset.num_points,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.005,
        device=device
    )
    model.to(device)
    
    # Get a sample from the dataset
    sample_idx = np.random.randint(0, len(dataset))
    point_cloud_normalized, env_grid = dataset[sample_idx]
    
    print(f"Using sample {sample_idx}")
    
    # Convert to numpy
    point_cloud_norm_np = point_cloud_normalized.cpu().numpy()
    env_grid_np = env_grid.cpu().numpy()
    
    # Show raw coordinate ranges
    point_cloud_raw = dataset.denormalize_points(point_cloud_norm_np)
    print(f"Raw point cloud coordinate ranges:")
    print(f"  X: [{point_cloud_raw[:, 0].min():.3f}, {point_cloud_raw[:, 0].max():.3f}]")
    print(f"  Y: [{point_cloud_raw[:, 1].min():.3f}, {point_cloud_raw[:, 1].max():.3f}]")
    print(f"  Z: [{point_cloud_raw[:, 2].min():.3f}, {point_cloud_raw[:, 2].max():.3f}]")
    
    # Show scaled coordinate ranges (what will be plotted)
    x_scaled = point_cloud_raw[:, 0] * (10.0 / 64.0)
    y_scaled = point_cloud_raw[:, 1] * (10.0 / 64.0)  
    theta_scaled = point_cloud_raw[:, 2] * (2 * np.pi / 64.0) - np.pi
    print(f"Scaled coordinate ranges (for plotting):")
    print(f"  X: [{x_scaled.min():.3f}, {x_scaled.max():.3f}] meters")
    print(f"  Y: [{y_scaled.min():.3f}, {y_scaled.max():.3f}] meters")
    print(f"  θ: [{theta_scaled.min():.3f}, {theta_scaled.max():.3f}] radians")
    
    # Visualize forward process with raw coordinates
    print("Visualizing forward diffusion process with raw coordinate values...")
    visualize_forward_process(model, point_cloud_norm_np, env_grid_np, dataset)
    
    print("Visualization complete! Check the plots/ directory for saved figures.")

if __name__ == "__main__":
    main()
