import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AttentionBlock(nn.Module):
    """Transformer-style block with self-attention and conditioning"""
    def __init__(self, dim, cond_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP with conditioning
        self.mlp = nn.Sequential(
            nn.Linear(dim + cond_dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, conditioning):
        """
        x: (batch_size, num_points, dim)
        conditioning: (batch_size, num_points, cond_dim) - time + env embeddings
        """
        # Self-attention with residual connection
        x_normed = self.norm1(x)
        attn_out, _ = self.attention(x_normed, x_normed, x_normed)
        x = x + attn_out
        
        # MLP with conditioning and residual connection
        x_normed = self.norm2(x)
        x_cond = torch.cat([x_normed, conditioning], dim=-1)
        mlp_out = self.mlp(x_cond)
        x = x + mlp_out
        
        return x


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
    """Network for denoising individual points with conditioning and self-attention"""
    def __init__(self, state_dim, time_dim=128, env_dim=128, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),  # Bigger intermediate
            nn.SiLU(),  # Better activation than ReLU
            nn.Linear(time_dim * 4, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Point processing network
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # Attention blocks with conditioning
        cond_dim = time_dim + env_dim
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, cond_dim, num_heads)
            for _ in range(num_layers)
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
        
        # Combine conditioning
        conditioning = torch.cat([t_emb_expanded, env_emb_expanded], dim=-1)  # (batch_size, N, time_dim + env_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x)  # (batch_size, N, hidden_dim)
        
        # Apply attention blocks with conditioning
        for block in self.blocks:
            h = block(h, conditioning)
        
        # Output noise prediction
        noise_pred = self.output_proj(h)
        
        return noise_pred


class BRTDiffusionModel(nn.Module):
    """Complete diffusion model for BRT generation"""
    def __init__(self, state_dim, env_size, num_points, num_timesteps=1000,
                 beta_start=0.0001, beta_end=0.02, device='cuda', 
                 null_conditioning_prob=0.15):
        super().__init__()
        self.state_dim = state_dim
        self.env_size = env_size
        self.num_points = num_points
        self.num_timesteps = num_timesteps
        self.device = device
        self.null_conditioning_prob = null_conditioning_prob
        
        # Networks
        self.env_encoder = EnvironmentEncoder(env_size)
        self.denoiser = PointDiffusionNetwork(state_dim)
        
        # Create null environment (all zeros) for unconditional training
        self.register_buffer('null_env', torch.zeros(1, env_size, env_size))
        
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
    
    def p_mean_variance(self, x_t, t, env, guidance_scale=1.0):
        """Compute mean and variance for reverse process with classifier-free guidance"""
        batch_size = x_t.shape[0]
        
        if guidance_scale != 1.0:
            # Classifier-free guidance: compute both conditional and unconditional predictions
            
            # Conditional prediction
            env_embedding_cond = self.env_encoder(env)
            noise_pred_cond = self.denoiser(x_t, t, env_embedding_cond)
            
            # Unconditional prediction (using null environment)
            null_env_batch = self.null_env.expand(batch_size, -1, -1)
            env_embedding_uncond = self.env_encoder(null_env_batch)
            noise_pred_uncond = self.denoiser(x_t, t, env_embedding_uncond)
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # Standard conditional prediction
            env_embedding = self.env_encoder(env)
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
    def p_sample(self, x_t, t, env, guidance_scale=1.0):
        """Single reverse diffusion step with classifier-free guidance"""
        model_mean, posterior_variance = self.p_mean_variance(x_t, t, env, guidance_scale)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, 1, 1))
        
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, env, num_samples=1, guidance_scale=1.0):
        """Generate BRT samples given environment with classifier-free guidance"""
        batch_size = num_samples
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.num_points, self.state_dim).to(self.device)
        
        # Reverse diffusion process with guidance
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, env, guidance_scale)
        
        return x_t
    
    def compute_loss(self, x_start, env):
        """Compute training loss with classifier-free guidance"""
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Apply null conditioning with probability null_conditioning_prob
        if self.training:
            # Create a mask for which samples should use null conditioning
            null_mask = torch.rand(batch_size, device=self.device) < self.null_conditioning_prob
            
            # Replace environments with null environment where mask is True
            env_conditioned = env.clone()
            if null_mask.any():
                null_env_batch = self.null_env.expand(null_mask.sum(), -1, -1)
                env_conditioned[null_mask] = null_env_batch
        else:
            env_conditioned = env
        
        # Get environment embedding
        env_embedding = self.env_encoder(env_conditioned)
        
        # Predict noise
        noise_pred = self.denoiser(x_noisy, t, env_embedding)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
