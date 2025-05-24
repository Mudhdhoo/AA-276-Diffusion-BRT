import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import math
import os
import argparse
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_args():
    parser = argparse.ArgumentParser(description='Train BRT Diffusion Model')
    parser.add_argument('--dataset_dir', type=str, 
                      default='/Users/malte/AA-276-Diffusion-BRT/dataset/outputs/16_May_2025_20_36 (64x64x64)_pointcloud_2000',
                      help='Path to dataset directory containing sample_* folders')
    parser.add_argument('--num_epochs', type=int, default=2000,
                      help='Number of training epochs')
    parser.add_argument('--sample_every', type=int, default=100,
                      help='Generate samples every N epochs')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                      help='Save model checkpoint every N epochs')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                      help='Number of diffusion timesteps')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (cuda/cpu)')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                      help='Weights & Biases API key (optional)')
    parser.add_argument('--wandb_project', type=str, default='brt-diffusion',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity name')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EnvironmentEncoder(nn.Module):
    """Encoder for nxn environment matrix"""
    def __init__(self, env_size, hidden_dim=256, output_dim=128):
        super().__init__()
        self.env_size = env_size
        
        # Convolutional layers to process environment grid
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Global pooling and projection
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, env):
        # env shape: (batch_size, env_size, env_size)
        x = env.unsqueeze(1)  # Add channel dimension
        
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv3(x))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x


class PointDiffusionNetwork(nn.Module):
    """Network for denoising individual points with conditioning"""
    def __init__(self, state_dim, time_dim=128, env_dim=128, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Point processing network
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # Main processing blocks with conditioning
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + time_dim + env_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(4)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )
        
    def forward(self, x, t, env_embedding):
        """
        x: (batch_size, N, state_dim) - noisy points
        t: (batch_size,) - time steps
        env_embedding: (batch_size, env_dim) - environment embedding
        """
        batch_size, N, _ = x.shape
        
        # Get time embedding
        t_emb = self.time_mlp(t)  # (batch_size, time_dim)
        
        # Expand embeddings to match number of points
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)  # (batch_size, N, time_dim)
        env_emb_expanded = env_embedding.unsqueeze(1).expand(-1, N, -1)  # (batch_size, N, env_dim)
        
        # Process points
        x_flat = x.view(-1, self.state_dim)  # (batch_size * N, state_dim)
        h = self.input_proj(x_flat)  # (batch_size * N, hidden_dim)
        h = h.view(batch_size, N, -1)  # (batch_size, N, hidden_dim)
        
        # Apply blocks with conditioning
        for block in self.blocks:
            # Concatenate hidden state with embeddings
            h_cond = torch.cat([h, t_emb_expanded, env_emb_expanded], dim=-1)
            h_flat = h_cond.view(-1, h_cond.shape[-1])
            h_new = block(h_flat)
            h_new = h_new.view(batch_size, N, -1)
            h = h + h_new  # Residual connection
        
        # Output noise prediction
        h_flat = h.view(-1, h.shape[-1])
        noise_pred = self.output_proj(h_flat)
        noise_pred = noise_pred.view(batch_size, N, self.state_dim)
        
        return noise_pred


class BRTDiffusionModel(nn.Module):
    """Complete diffusion model for BRT generation"""
    def __init__(self, state_dim, env_size, num_points, num_timesteps=1000,
                 beta_start=0.0001, beta_end=0.02, device='cuda'):
        super().__init__()
        self.state_dim = state_dim
        self.env_size = env_size
        self.num_points = num_points
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Networks
        self.env_encoder = EnvironmentEncoder(env_size)
        self.denoiser = PointDiffusionNetwork(state_dim)
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-calculate useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process - add noise to data"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Broadcast to match dimensions
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_mean_variance(self, x_t, t, env):
        """Compute mean and variance for reverse process"""
        # Get environment embedding
        env_embedding = self.env_encoder(env)
        
        # Predict noise
        noise_pred = self.denoiser(x_t, t, env_embedding)
        
        # Calculate mean
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (
            x_t - beta_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1)
        
        return model_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, x_t, t, env):
        """Single reverse diffusion step"""
        model_mean, posterior_variance = self.p_mean_variance(x_t, t, env)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, 1, 1))
        
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, env, num_samples=1):
        """Generate BRT samples given environment"""
        batch_size = num_samples
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.num_points, self.state_dim).to(self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, env)
        
        return x_t
    
    def compute_loss(self, x_start, env):
        """Compute training loss"""
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Get environment embedding
        env_embedding = self.env_encoder(env)
        
        # Predict noise
        noise_pred = self.denoiser(x_noisy, t, env_embedding)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


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


def visualize_point_cloud(points, title=None, save_path=None, dataset=None):
    """Visualize a single point cloud and optionally save it."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Denormalize points if dataset is provided
    if dataset is not None:
        points = dataset.denormalize_points(points)
    
    # Scale coordinates to match environment dimensions
    x = points[:, 0] * (10.0 / 64.0)  # Scale x from [0,64] to [0,10]
    y = points[:, 1] * (10.0 / 64.0)  # Scale y from [0,64] to [0,10]
    theta = points[:, 2] * (2 * np.pi / 64.0) - np.pi  # Scale theta from [0,64] to [-π,π]
    
    ax.scatter(x, y, theta, s=2, alpha=0.5)
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('θ')
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(-np.pi, np.pi)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_environment_grid(grid, title=None, save_path=None):
    """Visualize a single environment grid and optionally save it."""
    plt.figure(figsize=(8, 8))
    # Use binary colormap and set vmin/vmax to ensure binary visualization
    plt.imshow(grid, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10])  # Set extent to match coordinate system
    if title:
        plt.title(title)
    plt.colorbar(label='Obstacle (1) / Free Space (0)')
    plt.axis('equal')  # Ensure equal aspect ratio
    # Set axis limits to match coordinate system
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_denoising_process(points_sequence, titles, save_path=None, dataset=None):
    """Visualize the denoising process in a single figure with subplots.
    
    Args:
        points_sequence: List of point clouds to visualize
        titles: List of titles for each subplot
        save_path: Optional path to save the figure
        dataset: Dataset object for denormalization
    """
    n_steps = len(points_sequence)
    n_cols = 5  # Fixed number of columns (initial + 3 steps + final)
    n_rows = 1  # Single row
    
    fig = plt.figure(figsize=(25, 5))  # Wider figure for 5 subplots
    
    for i, (points, title) in enumerate(zip(points_sequence, titles)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        
        # Denormalize points if dataset is provided
        if dataset is not None:
            points = dataset.denormalize_points(points)
        
        # Scale coordinates to match environment dimensions
        x = points[:, 0] * (10.0 / 64.0)  # Scale x from [0,64] to [0,10]
        y = points[:, 1] * (10.0 / 64.0)  # Scale y from [0,64] to [0,10]
        theta = points[:, 2] * (2 * np.pi / 64.0) - np.pi  # Scale theta from [0,64] to [-π,π]
        
        ax.scatter(x, y, theta, s=2, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('θ')
        # Set axis limits
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-np.pi, np.pi)
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_comparison(true_pc, generated_pc, env_grid, title=None, save_path=None, dataset=None):
    """Visualize true BRT, generated BRT, and environment side by side."""
    # Create figure with specific size ratio to ensure square environment plot
    fig = plt.figure(figsize=(30, 10))
    
    # Environment subplot - make it square
    ax1 = fig.add_subplot(131, aspect='equal')
    im = ax1.imshow(env_grid, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10])
    ax1.set_title('Environment Grid', fontsize=14, pad=20)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Obstacle (1) / Free Space (0)', fontsize=12)
    
    # True BRT subplot
    ax2 = fig.add_subplot(132, projection='3d')
    if dataset is not None:
        true_pc = dataset.denormalize_points(true_pc)
    x = true_pc[:, 0] * (10.0 / 64.0)
    y = true_pc[:, 1] * (10.0 / 64.0)
    theta = true_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    scatter2 = ax2.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    ax2.set_title('True BRT Point Cloud', fontsize=14, pad=20)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_zlabel('θ (rad)', fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_zlim(-np.pi, np.pi)
    
    # Generated BRT subplot
    ax3 = fig.add_subplot(133, projection='3d')
    if dataset is not None:
        generated_pc = dataset.denormalize_points(generated_pc)
    x = generated_pc[:, 0] * (10.0 / 64.0)
    y = generated_pc[:, 1] * (10.0 / 64.0)
    theta = generated_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    scatter3 = ax3.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    ax3.set_title('Generated BRT Point Cloud', fontsize=14, pad=20)
    ax3.set_xlabel('X Position (m)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.set_zlabel('θ (rad)', fontsize=12)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_zlim(-np.pi, np.pi)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_denoising_with_true(points_sequence, true_pc, titles, save_path=None, dataset=None):
    """Visualize the denoising process and true BRT side by side."""
    n_steps = len(points_sequence)
    fig = plt.figure(figsize=(30, 8))
    
    # Denoising process subplots
    for i, (points, title) in enumerate(zip(points_sequence, titles)):
        ax = fig.add_subplot(1, n_steps + 1, i + 1, projection='3d')
        
        if dataset is not None:
            points = dataset.denormalize_points(points)
        
        x = points[:, 0] * (10.0 / 64.0)
        y = points[:, 1] * (10.0 / 64.0)
        theta = points[:, 2] * (2 * np.pi / 64.0) - np.pi
        
        scatter = ax.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
        ax.set_title(f'Denoising Step {title}', fontsize=14, pad=20)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('θ (rad)', fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-np.pi, np.pi)
    
    # True BRT subplot
    ax_true = fig.add_subplot(1, n_steps + 1, n_steps + 1, projection='3d')
    if dataset is not None:
        true_pc = dataset.denormalize_points(true_pc)
    x = true_pc[:, 0] * (10.0 / 64.0)
    y = true_pc[:, 1] * (10.0 / 64.0)
    theta = true_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    scatter_true = ax_true.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    ax_true.set_title('True BRT Point Cloud', fontsize=14, pad=20)
    ax_true.set_xlabel('X Position (m)', fontsize=12)
    ax_true.set_ylabel('Y Position (m)', fontsize=12)
    ax_true.set_zlabel('θ (rad)', fontsize=12)
    ax_true.set_xlim(0, 10)
    ax_true.set_ylim(0, 10)
    ax_true.set_zlim(-np.pi, np.pi)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def train_model(model, dataset, num_epochs=1000, batch_size=32, lr=1e-4, sample_every=10, checkpoint_every=100, wandb_api_key=None, wandb_project='brt-diffusion', wandb_entity=None):
    """Training loop for the diffusion model"""
    # Initialize wandb
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.config.update({
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'sample_every': sample_every,
            'checkpoint_every': checkpoint_every,
            'num_timesteps': model.num_timesteps,
            'num_points': model.num_points,
            'env_size': model.env_size,
            'points_mean': dataset.points_mean.tolist(),
            'points_std': dataset.points_std.tolist()
        })

    # Create directories for checkpoints and samples
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create fixed training samples for consistent evaluation
    num_vis_samples = 4  # Number of different samples to visualize
    train_indices = torch.randint(0, len(train_dataset), (num_vis_samples,))
    vis_samples = [(train_dataset[i][0], train_dataset[i][1]) for i in train_indices]  # (point_cloud, env_grid) pairs
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for brt_batch, env_batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            brt_batch = brt_batch.to(model.device)
            env_batch = env_batch.to(model.device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(brt_batch, env_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
            if wandb_api_key:
                wandb.log({'loss': avg_loss}, step=epoch)
            
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vis_samples': vis_samples,
                'losses': losses
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Generate samples periodically using fixed training samples
        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\nGenerating samples at epoch {epoch+1}:")
                
                # Create epoch-specific directory
                epoch_dir = os.path.join('samples', f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                for i, (true_pc, env_grid) in enumerate(vis_samples):
                    # Move to device
                    env_grid = env_grid.to(model.device)
                    
                    # Generate sample
                    generated_brt = model.sample(env_grid.unsqueeze(0), num_samples=1)
                    generated_brt = generated_brt[0].cpu().numpy()  # Remove batch dimension
                    
                    # Create comparison visualization
                    comparison_save_path = os.path.join(epoch_dir, f'comparison_{i+1}.png')
                    visualize_comparison(
                        true_pc.cpu().numpy(),
                        generated_brt,
                        env_grid.squeeze(0).cpu().numpy(),
                        f'Sample {i+1} Comparison',
                        comparison_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/comparison_{i+1}': wandb.Image(comparison_save_path)})
                    
                    # Start from pure noise
                    x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)
                    
                    # Save exactly 5 steps total
                    num_steps = 5  # Total number of steps to visualize
                    step_indices = np.linspace(0, model.num_timesteps-1, num_steps, dtype=int)
                    
                    # Store points and titles for visualization
                    points_sequence = []  # Start empty
                    titles = []
                    
                    # Add all steps
                    for t in reversed(range(model.num_timesteps)):
                        t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
                        x_t = model.p_sample(x_t, t_batch, env_grid.unsqueeze(0))
                        
                        if t in step_indices:
                            points_sequence.append(x_t[0].cpu().numpy())
                            titles.append(f't={t}')
                    
                    # Create and save denoising process visualization with true BRT
                    denoising_save_path = os.path.join(epoch_dir, f'denoising_with_true_{i+1}.png')
                    visualize_denoising_with_true(
                        points_sequence,
                        true_pc.cpu().numpy(),
                        titles,
                        denoising_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/denoising_with_true_{i+1}': wandb.Image(denoising_save_path)})
                    
                    print(f"Training sample {i+1}, generated BRT shape: {generated_brt.shape}")
            
            model.train()
            print()  # Add newline for better readability
    
    if wandb_api_key:
        wandb.finish()
    
    return losses, vis_samples  # Return visualization samples for potential later use


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create dataset
    dataset = BRTDataset(args.dataset_dir)
    
    # Get dimensions from dataset
    STATE_DIM = dataset.state_dim
    NUM_POINTS = dataset.num_points
    ENV_SIZE = dataset.env_size
    
    # Initialize model
    model = BRTDiffusionModel(
        state_dim=STATE_DIM,
        env_size=ENV_SIZE,
        num_points=NUM_POINTS,
        num_timesteps=args.num_timesteps,
        device=args.device
    ).to(args.device)
    
    print(f"Model initialized on {args.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset dimensions: {NUM_POINTS} points, {STATE_DIM}D coordinates, {ENV_SIZE}x{ENV_SIZE} environment")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sampling every {args.sample_every} epochs")
    
    # Train model
    losses, vis_samples = train_model(
        model, 
        dataset, 
        num_epochs=args.num_epochs, 
        batch_size=args.batch_size,
        lr=args.lr,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    # Save the trained model and visualization samples
    torch.save({
        'model_state_dict': model.state_dict(),
        'vis_samples': vis_samples,
        'losses': losses
    }, 'brt_diffusion_model.pt')
    print("Model, visualization samples, and training losses saved to brt_diffusion_model.pt")