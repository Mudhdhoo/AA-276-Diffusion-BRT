import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


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
    def __init__(self, brt_data, env_data):
        """
        brt_data: (num_samples, N, state_dim) - BRT point clouds
        env_data: (num_samples, env_size, env_size) - environment matrices
        """
        self.brt_data = torch.FloatTensor(brt_data)
        self.env_data = torch.FloatTensor(env_data)
        
    def __len__(self):
        return len(self.brt_data)
    
    def __getitem__(self, idx):
        return self.brt_data[idx], self.env_data[idx]


def train_model(model, dataset, num_epochs=1000, batch_size=32, lr=1e-4):
    """Training loop for the diffusion model"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for brt_batch, env_batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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
    
    return losses


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    STATE_DIM = 3  # e.g., for unicycle model (x, y, theta)
    ENV_SIZE = 32  # 32x32 environment grid
    NUM_POINTS = 100  # Number of points in BRT point cloud
    NUM_TIMESTEPS = 1000  # Diffusion steps
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic data for demonstration
    num_samples = 1000
    brt_data = np.random.randn(num_samples, NUM_POINTS, STATE_DIM)
    env_data = np.random.rand(num_samples, ENV_SIZE, ENV_SIZE)
    
    # Create dataset
    dataset = BRTDataset(brt_data, env_data)
    
    # Initialize model
    model = BRTDiffusionModel(
        state_dim=STATE_DIM,
        env_size=ENV_SIZE,
        num_points=NUM_POINTS,
        num_timesteps=NUM_TIMESTEPS,
        device=DEVICE
    ).to(DEVICE)
    
    print(f"Model initialized on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    # losses = train_model(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    # Generate samples
    model.eval()
    test_env = torch.rand(1, ENV_SIZE, ENV_SIZE).to(DEVICE)
    generated_brt = model.sample(test_env, num_samples=1)
    print(f"Generated BRT shape: {generated_brt.shape}")