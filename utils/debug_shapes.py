import torch
from diffusion.ddpm_conditional_new import Diffusion
from diffusion.film_module import UNet_conditional_FiLM, EMA, GridProjection

def debug_shapes():
    """Function to debug shapes of tensors in the diffusion process"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    channels = 101
    grid_size = 128
    # Create a small diffusion model
    diffusion = Diffusion(noise_steps=1000, img_size=grid_size, device=device)
    model = UNet_conditional_FiLM(c_in=channels, c_out=channels, grid_size=grid_size, device=device)
    grid_projection = GridProjection(in_channels=1, out_channels=256, cond_dim=256).to(device)
    
    # Generate a small batch of images
    batch_size = 1
    images = torch.randn(batch_size, channels, grid_size, grid_size).to(device)
    labels = torch.randn(batch_size, 1, grid_size, grid_size).to(device)
    
    conditioning = grid_projection(labels)
    print(f"Conditioning shape: {conditioning.shape}")

    # Sample timesteps
    t = diffusion.sample_timesteps(batch_size).to(device)
    print(f"Timesteps shape: {t.shape}")
    
    # Apply noise
    x_t, noise = diffusion.noise_images(images, t)
    print(f"Noisy images shape: {x_t.shape}")
    print(f"Noise shape: {noise.shape}")
    
    # Get model prediction
    predicted_noise = model(x_t, t, conditioning)
    print(f"Predicted noise shape: {predicted_noise.shape}")
    
    # Check alpha shapes
    alpha = diffusion.alpha[t]
    alpha_hat = diffusion.alpha_hat[t]
    beta = diffusion.beta[t]
    
    print(f"Alpha shape: {alpha.shape}")
    print(f"Alpha expanded shape: {alpha[:, None, None, None].shape}")
    print(f"Alpha hat shape: {alpha_hat.shape}")
    print(f"Beta shape: {beta.shape}")