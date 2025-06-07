import torch
from dataset.BRTDataset import BRTDataset
from models.diffusion_modules import BRTDiffusionModel
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

def main():
    num_samples = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_dir = "../point_cloud_dataset_4000"
    dataset = BRTDataset(dataset_dir, split="val")

    model = BRTDiffusionModel(
    state_dim=dataset.state_dim,
    env_size=dataset.env_size,
    num_points=dataset.num_points,
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.005,
    device=device)

    checkpoint_path = "checkpoints/checkpoint_epoch_2000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Explicitly set weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    sample_ind = np.random.randint(0, len(dataset))
    point_cloud, env_grid, *_ = dataset[sample_ind]  # Use wildcard to handle extra return values
    point_cloud = point_cloud.to(model.device)
    env_grid = env_grid.to(model.device)

    # Iterative reverse sampling
    reverse_process = []
    for i in range(num_samples):
        logger.info(f"Sampling {i+1}/{num_samples}")
        x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)

        for t in tqdm(reversed(range(model.num_timesteps)), desc="Denoising"):
            with torch.no_grad():
                t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
                x_t = model.p_sample(x_t, t_batch, env_grid)
        reverse_process.append(x_t.clone().squeeze(0))

    reverse_process = [dataset.denormalize_points(point_cloud) for point_cloud in reverse_process]
    accumulated_point_clouds = []

    for i in range(num_samples):
        combined_cloud = torch.stack(reverse_process[:i+1])
        accumulated_point_clouds.append(combined_cloud.view(-1, dataset.state_dim))

    # Get ground truth point cloud
    ground_truth = dataset.denormalize_points(point_cloud.squeeze(0))
    
    # Plot the accumulated point clouds and ground truth
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(4 * (num_samples + 1), 4))
    
    # If only one subplot, axes won't be an array
    if num_samples == 0:
        axes = [axes]
    elif num_samples == 1:
        axes = [axes[0], axes[1]]
    
    # Plot ground truth first
    gt_points = ground_truth.cpu().numpy()
    if dataset.state_dim == 2:
        axes[0].scatter(gt_points[:, 0], gt_points[:, 1], c='red', s=8, alpha=0.7)
        axes[0].set_title('Ground Truth')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
    elif dataset.state_dim == 3:
        ax = fig.add_subplot(1, num_samples + 1, 1, projection='3d')
        ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='red', s=8, alpha=0.7)
        ax.set_title('Ground Truth')
        axes[0] = ax
    
    # Plot accumulated point clouds
    for i, point_cloud in enumerate(accumulated_point_clouds):
        points = point_cloud.cpu().numpy()
        ax_idx = i + 1
        
        if dataset.state_dim == 2:
            axes[ax_idx].scatter(points[:, 0], points[:, 1], c='blue', s=8, alpha=0.7)
            axes[ax_idx].set_title(f'Accumulated Samples 1-{i+1}')
            axes[ax_idx].set_aspect('equal')
            axes[ax_idx].grid(True, alpha=0.3)
        elif dataset.state_dim == 3:
            # Remove the existing 2D axis and create a 3D one
            axes[ax_idx].remove()
            ax = fig.add_subplot(1, num_samples + 1, ax_idx + 1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=8, alpha=0.7)
            ax.set_title(f'Accumulated Samples 1-{i+1}')
            axes[ax_idx] = ax
    
    plt.tight_layout()
    plt.savefig('accumulated_point_clouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Plotted {num_samples} accumulated point clouds alongside ground truth")
    logger.info(f"Ground truth has {len(ground_truth)} points")
    for i, pc in enumerate(accumulated_point_clouds):
        logger.info(f"Accumulated sample {i+1} has {len(pc)} points")

if __name__ == "__main__":  
    main()
    
