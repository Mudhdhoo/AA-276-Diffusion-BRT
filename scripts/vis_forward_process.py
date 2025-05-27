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
    """Visualize the forward diffusion process at various timesteps using both normalized and raw coordinate values"""
    if timesteps_to_show is None:
        # Show process at these specific timesteps
        timesteps_to_show = [0, 100, 300, 500, 700, 900, 999]
    
    num_timesteps = len(timesteps_to_show)
    fig = plt.figure(figsize=(25, 12))  # Increased width to accommodate moved colorbars
    
    # Adjust subplot spacing to ensure equal plot sizes
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    
    # x_start is normalized data from dataset, convert to torch tensor
    x_start_torch = torch.FloatTensor(x_start).unsqueeze(0).to(model.device)
    
    # Get value range for consistent coloring across all timesteps
    x_start_raw = dataset.denormalize_points(x_start)
    if x_start_raw.shape[1] >= 4:
        value_min = x_start_raw[:, 3].min()
        value_max = x_start_raw[:, 3].max()
    else:
        # Fallback if no 4th dimension
        value_min, value_max = -1, 1
    
    for i, t in enumerate(timesteps_to_show):
        if t == 0:
            # Original data
            x_t_norm = x_start  # Normalized version (what model sees)
            x_t_raw = dataset.denormalize_points(x_start)  # Denormalized version
            title = f"Original (t=0)"
        else:
            # Apply forward diffusion to normalized data
            t_tensor = torch.tensor([t]).to(model.device)
            x_t_torch = model.q_sample(x_start_torch, t_tensor)
            x_t_norm = x_t_torch.squeeze(0).cpu().numpy()  # Normalized version with noise
            # Denormalize the noisy normalized data to get raw coordinates
            x_t_raw = dataset.denormalize_points(x_t_norm)
            title = f"t={t}"
        
        # Row 1: Raw/Denormalized coordinates (after noise applied to normalized version)
        ax1 = fig.add_subplot(3, num_timesteps, i + 1, projection='3d')
        
        # Extract coordinates
        x = x_t_raw[:, 0] 
        y = x_t_raw[:, 1] 
        theta = x_t_raw[:, 2]
        
        # Use 4th dimension (value) for coloring if available
        if x_t_raw.shape[1] >= 4:
            colors = x_t_raw[:, 3]
            colormap = 'RdYlBu_r'  # Red-Yellow-Blue colormap (red=high value, blue=low value)
        else:
            colors = theta
            colormap = 'viridis'
            
        scatter1 = ax1.scatter(x, y, theta, c=colors, cmap=colormap, s=8, alpha=0.7, 
                              vmin=value_min, vmax=value_max)
        ax1.set_title(f"{title} (Raw)")
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('θ (rad)')
        # Set axis limits to match environment dimensions and force theta to [-pi, pi]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_zlim(-np.pi, np.pi)  # Force exactly [-pi, pi] range
        
        # Add colorbar to first plot in row
        if i == 0:
            if x_t_raw.shape[1] >= 4:
                cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.15)
                cbar1.set_label('BRT Value')
            else:
                cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.15)
                cbar1.set_label('θ (rad)')
        
        # Row 2: Normalized coordinates (what model sees during training)
        ax2 = fig.add_subplot(3, num_timesteps, i + 1 + num_timesteps, projection='3d')
        
        # Use 4th dimension for coloring in normalized space too
        if x_t_norm.shape[1] >= 4:
            colors_norm = x_t_norm[:, 3]
            # Get normalization range for consistent coloring
            norm_min = x_t_norm[:, 3].min() if t == 0 else None
            norm_max = x_t_norm[:, 3].max() if t == 0 else None
        else:
            colors_norm = x_t_norm[:, 2]
            norm_min, norm_max = None, None
            
        scatter2 = ax2.scatter(x_t_norm[:, 0], x_t_norm[:, 1], x_t_norm[:, 2], 
                              c=colors_norm, cmap=colormap, s=8, alpha=0.7,
                              vmin=norm_min, vmax=norm_max)
        ax2.set_title(f"{title} (Normalized)")
        ax2.set_xlabel('X (normalized)')
        ax2.set_ylabel('Y (normalized)')
        ax2.set_zlabel('θ (normalized)')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
        ax2.set_zlim(-2.5, 2.5)
        
        # Set normalized theta limits to correspond to [-pi, pi] range
        # Since theta is normalized from [-pi, pi] to the normalized range, we need to find what that maps to
        raw_theta_min, raw_theta_max = -np.pi, np.pi
        norm_theta_min = dataset.normalize_points(np.array([[0, 0, raw_theta_min, 0] if x_start.shape[1] >= 4 else [0, 0, raw_theta_min]]))[0, 2]
        norm_theta_max = dataset.normalize_points(np.array([[0, 0, raw_theta_max, 0] if x_start.shape[1] >= 4 else [0, 0, raw_theta_max]]))[0, 2]
        ax2.set_zlim(norm_theta_min, norm_theta_max)
        
        # Add colorbar to first plot in row
        if i == 0:
            if x_t_norm.shape[1] >= 4:
                cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.15)
                cbar2.set_label('BRT Value (normalized)')
            else:
                cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.15)
                cbar2.set_label('θ (normalized)')
        
        # Row 3: Environment plot
        ax3 = fig.add_subplot(3, num_timesteps, i + 1 + 2*num_timesteps)
        im = ax3.imshow(env, cmap='RdYlBu_r', origin='lower', extent=[0, 10, 0, 10])
        ax3.set_title('Environment')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        
        # Add colorbar to last environment subplot
        if i == num_timesteps - 1:
            cbar3 = plt.colorbar(im, ax=ax3, shrink=0.5, pad=0.15)
            cbar3.set_label('Obstacle (1) / Free Space (0)')
    
    # Use subplots_adjust instead of tight_layout for better control
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the forward process visualization"""
    print("Creating BRT Diffusion Forward Process Visualization...")
    
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
        beta_end=0.008,
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
    print(f"  θ: [{point_cloud_raw[:, 2].min():.3f}, {point_cloud_raw[:, 2].max():.3f}]")
    if point_cloud_raw.shape[1] >= 4:
        print(f"  Value: [{point_cloud_raw[:, 3].min():.3f}, {point_cloud_raw[:, 3].max():.3f}]")
    
    # Show normalized coordinate ranges
    print(f"Normalized point cloud coordinate ranges:")
    print(f"  X: [{point_cloud_norm_np[:, 0].min():.3f}, {point_cloud_norm_np[:, 0].max():.3f}]")
    print(f"  Y: [{point_cloud_norm_np[:, 1].min():.3f}, {point_cloud_norm_np[:, 1].max():.3f}]")
    print(f"  θ: [{point_cloud_norm_np[:, 2].min():.3f}, {point_cloud_norm_np[:, 2].max():.3f}]")
    if point_cloud_norm_np.shape[1] >= 4:
        print(f"  Value: [{point_cloud_norm_np[:, 3].min():.3f}, {point_cloud_norm_np[:, 3].max():.3f}]")
    
    # Visualize forward process
    print("Visualizing forward diffusion process...")
    print("- Row 1: Raw coordinates (denormalized after noise applied to normalized data)")
    print("- Row 2: Normalized coordinates (what model sees during training)")
    print("- Row 3: Environment with point overlay")
    print("- Colors represent BRT values (red=high, blue=low)")
    visualize_forward_process(model, point_cloud_norm_np, env_grid_np, dataset)
    
    print("Visualization complete! Check the plots/ directory for saved figures.")

if __name__ == "__main__":
    main()
