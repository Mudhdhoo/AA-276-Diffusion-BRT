"""
Generate a GIF showing the denoising process for a sample from the dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import sys
from loguru import logger
from argparse import ArgumentParser

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion_modules import BRTDiffusionModel
from dataset.BRTDataset import BRTDataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=70)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="plots/gifs")
    return parser.parse_args()


def generate_denoising_gif(model, dataset, sample_idx, num_frames=50, save_dir="plots/gifs"):
    """Generate a GIF showing the denoising process for a sample from the dataset."""
    # Get sample from dataset
    point_cloud, env_grid, *_ = dataset[sample_idx]  # Use wildcard to handle extra return values
    point_cloud = point_cloud.unsqueeze(0).to(model.device)  # Add batch dimension
    env_grid = env_grid.unsqueeze(0).to(model.device)  # Add batch dimension
    
    # Start from pure noise
    x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)
    
    # Compute full reverse process
    reverse_process = []
    for t in tqdm(reversed(range(model.num_timesteps)), desc="Denoising"):
        with torch.no_grad():
            t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
            x_t = model.p_sample(x_t, t_batch, env_grid)
        reverse_process.append(x_t.clone())
    
    # Select evenly spaced samples for the GIF
    timesteps = np.linspace(0, len(reverse_process)-1, num_frames, dtype=int)
    selected_samples = [reverse_process[i] for i in timesteps]
    
    # Add extra frames at the end to hold the final state
    hold_frames = 10  # Number of extra frames to hold the final state
    selected_samples.extend([selected_samples[-1]] * hold_frames)
    
    # Create figure with custom layout and professional styling
    plt.style.use('default')  # Ensure clean default style
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    
    # Main denoising process plot (left side, spans full height)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=4, projection='3d')
    ax_main.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax_main.set_zlabel('θ (rad)', fontsize=12, fontweight='bold')
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 10)
    ax_main.set_zlim(-np.pi, np.pi)
    ax_main.set_title('Real-time Denoising Process', fontsize=16, fontweight='bold', pad=25, color='black')
    ax_main.grid(True, alpha=0.3)
    ax_main.xaxis.pane.fill = False
    ax_main.yaxis.pane.fill = False
    ax_main.zaxis.pane.fill = False
    ax_main.xaxis.pane.set_edgecolor('gray')
    ax_main.yaxis.pane.set_edgecolor('gray')
    ax_main.zaxis.pane.set_edgecolor('gray')
    ax_main.xaxis.pane.set_alpha(0.1)
    ax_main.yaxis.pane.set_alpha(0.1)
    ax_main.zaxis.pane.set_alpha(0.1)
    
    # Ground truth plot (top right)
    ax_gt = plt.subplot2grid((4, 3), (0, 2), rowspan=2, projection='3d')
    ax_gt.set_xlabel('X (m)', fontsize=9, fontweight='bold')
    ax_gt.set_ylabel('Y (m)', fontsize=9, fontweight='bold')
    ax_gt.set_zlabel('θ (rad)', fontsize=9, fontweight='bold')
    ax_gt.set_xlim(0, 10)
    ax_gt.set_ylim(0, 10)
    ax_gt.set_zlim(-np.pi, np.pi)
    ax_gt.set_title('Ground Truth Target', fontsize=11, fontweight='bold', color='black')
    ax_gt.tick_params(labelsize=7)
    ax_gt.grid(True, alpha=0.2)
    
    # Environment grid plot (bottom right)
    ax_env = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
    ax_env.set_xlabel('X (m)', fontsize=9, fontweight='bold')
    ax_env.set_ylabel('Y (m)', fontsize=9, fontweight='bold')
    ax_env.set_title('Environment Constraints', fontsize=11, fontweight='bold', color='black')
    ax_env.tick_params(labelsize=7)
    
    # Initialize scatter plots with consistent blue colors
    scatter_main = ax_main.scatter([], [], [], s=8, alpha=0.8, c='#2E86AB', edgecolors='white', linewidth=0.1)
    
    # Plot ground truth with same blue color
    gt_points = dataset.denormalize_points(point_cloud[0].cpu().numpy())
    x_gt = gt_points[:, 0] * (10.0 / 64.0)
    y_gt = gt_points[:, 1] * (10.0 / 64.0)
    theta_gt = gt_points[:, 2] * (2 * np.pi / 64.0) - np.pi
    scatter_gt = ax_gt.scatter(x_gt, y_gt, theta_gt, s=4, alpha=0.8, c='#2E86AB', edgecolors='white', linewidth=0.1)
    
    # Plot environment grid - fix the visualization with better styling
    env_grid_np = env_grid[0].cpu().numpy()  # Get full environment grid
    
    # Handle different possible shapes of environment grid
    if len(env_grid_np.shape) == 3:  # If it's (channels, height, width)
        env_grid_2d = env_grid_np[0]  # Take first channel
    elif len(env_grid_np.shape) == 2:  # If it's already 2D
        env_grid_2d = env_grid_np
    else:  # If it's 1D, try to reshape to square
        size = int(np.sqrt(env_grid_np.size))
        if size * size == env_grid_np.size:
            env_grid_2d = env_grid_np.reshape(size, size)
        else:
            # If not square, create a simple visualization
            env_grid_2d = np.ones((64, 64)) * 0.5  # Gray background
    
    im = ax_env.imshow(env_grid_2d.T, origin='lower', extent=[0, 10, 0, 10], 
                       cmap='plasma', alpha=0.8, interpolation='bilinear')
    
    # Add colorbar for environment grid
    cbar = plt.colorbar(im, ax=ax_env, shrink=0.6, aspect=10)
    cbar.set_label('Constraint Level', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=6)
    
    # Add subtle borders around subplots
    for ax in [ax_main, ax_gt, ax_env]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.5)
    
    # Adjust layout with better spacing
    plt.tight_layout()
    
    def update(frame):
        # Get the point cloud for this frame
        x_t = selected_samples[frame]
        
        # Convert to numpy and denormalize
        points = x_t[0].cpu().numpy()
        points = dataset.denormalize_points(points)
        
        # Scale coordinates to match environment dimensions
        x = points[:, 0]
        y = points[:, 1]
        theta = points[:, 2]
        c = points[:, 3]  # Get the 4th dimension for coloring
        
        # Update main scatter plot
        scatter_main._offsets3d = (x, y, theta)
        scatter_main.set_array(c)  # Update the colors
        
        # Update title based on whether we're in the holding frames
        if frame < num_frames:
            current_timestep = model.num_timesteps - 1 - timesteps[frame]
            title = ax_main.set_title(f'Denoising Step t={current_timestep}', 
                            fontsize=16, fontweight='bold', pad=25, color='black')
        else:
            title = ax_main.set_title('Final Denoised State', 
                            fontsize=16, fontweight='bold', pad=25, color='black')
        
        return scatter_main, title  # Return all artists that are being updated
    
    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames + hold_frames,  # Add extra frames for holding
        interval=100,  # 100ms between frames
        blit=True
    )
    
    # Save animation
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, f'{sample_idx}.gif')
    anim.save(gif_path, writer=PillowWriter(fps=10))
    plt.close()
    logger.info(f"Saved GIF to {gif_path}")
    return gif_path

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = parse_args()
    
    # Load dataset
    dataset_dir = "point_cloud_dataset_4000"  # Update this path as needed
    dataset = BRTDataset(dataset_dir, split="val")
    
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
    
    # Load checkpoint
    checkpoint_path = "checkpoints/checkpoint_epoch_2000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Explicitly set weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    for i in range(args.num_samples):   
        sample_idx = np.random.randint(0, len(dataset))
        logger.info(f"Generating GIF for sample {sample_idx}")
        generate_denoising_gif(
            model,
            dataset,
            sample_idx,
            num_frames=args.num_frames,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
