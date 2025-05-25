import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse
import wandb
from models.diffusion_modules import BRTDiffusionModel
from dataset.BRTDataset import BRTDataset
from utils.visualizations import visualize_comparison, visualize_denoising_with_true
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train BRT Diffusion Model')
    parser.add_argument('--dataset_dir', type=str, 
                      default='point_cloud_dataset',
                      help='Path to dataset directory containing sample_* folders')
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

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=2000,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--sample_every', type=int, default=100,
                      help='Generate samples every N epochs')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                      help='Save model checkpoint every N epochs')

    # Diffusion parameters
    parser.add_argument('--num_timesteps', type=int, default=1000,
                      help='Number of diffusion timesteps')
    parser.add_argument("--beta_start", type=float, default=1e-4,
                      help="Beta start for the beta schedule")
    parser.add_argument("--beta_end", type=float, default=0.02,
                      help="Beta end for the beta schedule")
    parser.add_argument('--null_conditioning_prob', type=float, default=0.15,
                      help='Probability of using null conditioning during training for CFG')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                      help='Guidance scale for classifier-free guidance during sampling')
    return parser.parse_args()


def train_model(model, dataset, num_epochs=1000, batch_size=32, lr=1e-4, sample_every=10, checkpoint_every=100, wandb_api_key=None, wandb_project='brt-diffusion', wandb_entity=None, guidance_scale=1.0):
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
            'points_std': dataset.points_std.tolist(),
            'null_conditioning_prob': model.null_conditioning_prob,
            'guidance_scale': guidance_scale
        })

    # Create directories for checkpoints and samples
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Create dataloaders for train and validation
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = BRTDataset(dataset.dataset_dir, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create fixed training samples for consistent evaluation
    num_vis_samples = 8  # Increased from 4 to 8 samples
    train_indices = torch.randint(0, len(dataset), (num_vis_samples,))
    vis_samples = [(dataset[i][0], dataset[i][1]) for i in train_indices]  # (point_cloud, env_grid) pairs
    
    # Create fixed validation samples for consistent evaluation
    num_val_vis_samples = 2  # 2 validation samples
    val_indices = torch.randint(0, len(val_dataset), (num_val_vis_samples,))
    val_vis_samples = [(val_dataset[i][0], val_dataset[i][1]) for i in val_indices]
    
    model.train()
    losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Training loop
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
        logger.info(f'Epoch {epoch+1}, Training Loss: {avg_loss:.6f}')
        
        # Validation loop
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for brt_batch, env_batch in val_dataloader:
                brt_batch = brt_batch.to(model.device)
                env_batch = env_batch.to(model.device)
                loss = model.compute_loss(brt_batch, env_batch)
                val_epoch_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        logger.info(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
            if wandb_api_key:
                wandb.log({
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss
                }, step=epoch)
            
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'vis_samples': vis_samples,
                'val_vis_samples': val_vis_samples,
                'train_losses': losses,
                'val_losses': val_losses
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Generate samples periodically using fixed training and validation samples
        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\nGenerating samples at epoch {epoch+1}:")
                
                # Create epoch-specific directory
                epoch_dir = os.path.join('samples', f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                # Generate and plot training samples
                for i, (true_pc, env_grid) in enumerate(vis_samples):
                    # Move to device
                    env_grid = env_grid.to(model.device)
                    
                    # Generate sample
                    generated_brt = model.sample(env_grid.unsqueeze(0), num_samples=1, guidance_scale=guidance_scale)
                    generated_brt = generated_brt[0].cpu().numpy()  # Remove batch dimension
                    
                    # Create comparison visualization
                    comparison_save_path = os.path.join(epoch_dir, f'train_comparison_{i+1}.png')
                    visualize_comparison(
                        true_pc.cpu().numpy(),
                        generated_brt,
                        env_grid.squeeze(0).cpu().numpy(),
                        f'Training Sample {i+1} Comparison',
                        comparison_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/train_comparison_{i+1}': wandb.Image(comparison_save_path)})
                    
                    # Start from pure noise
                    x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)
                    
                    # Save exactly 5 steps total
                    num_steps = 5  # Total number of steps to visualize
                    step_indices = np.linspace(0, model.num_timesteps-1, num_steps, dtype=int)
                    
                    # Store points and titles for visualization
                    points_sequence = []
                    titles = []
                    
                    # Add all steps
                    for t in reversed(range(model.num_timesteps)):
                        t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
                        x_t = model.p_sample(x_t, t_batch, env_grid.unsqueeze(0), guidance_scale=guidance_scale)
                        
                        if t in step_indices:
                            points_sequence.append(x_t[0].cpu().numpy())
                            titles.append(f't={t}')
                    
                    # Create and save denoising process visualization with true BRT
                    denoising_save_path = os.path.join(epoch_dir, f'train_denoising_{i+1}.png')
                    visualize_denoising_with_true(
                        points_sequence,
                        true_pc.cpu().numpy(),
                        titles,
                        denoising_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/train_denoising_{i+1}': wandb.Image(denoising_save_path)})
                
                # Generate and plot validation samples
                for i, (true_pc, env_grid) in enumerate(val_vis_samples):
                    # Move to device
                    env_grid = env_grid.to(model.device)
                    
                    # Generate sample
                    generated_brt = model.sample(env_grid.unsqueeze(0), num_samples=1, guidance_scale=guidance_scale)
                    generated_brt = generated_brt[0].cpu().numpy()  # Remove batch dimension
                    
                    # Create comparison visualization
                    comparison_save_path = os.path.join(epoch_dir, f'val_comparison_{i+1}.png')
                    visualize_comparison(
                        true_pc.cpu().numpy(),
                        generated_brt,
                        env_grid.squeeze(0).cpu().numpy(),
                        f'Validation Sample {i+1} Comparison',
                        comparison_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/val_comparison_{i+1}': wandb.Image(comparison_save_path)})
                    
                    # Start from pure noise
                    x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)
                    
                    # Save exactly 5 steps total
                    num_steps = 5  # Total number of steps to visualize
                    step_indices = np.linspace(0, model.num_timesteps-1, num_steps, dtype=int)
                    
                    # Store points and titles for visualization
                    points_sequence = []
                    titles = []
                    
                    # Add all steps
                    for t in reversed(range(model.num_timesteps)):
                        t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
                        x_t = model.p_sample(x_t, t_batch, env_grid.unsqueeze(0), guidance_scale=guidance_scale)
                        
                        if t in step_indices:
                            points_sequence.append(x_t[0].cpu().numpy())
                            titles.append(f't={t}')
                    
                    # Create and save denoising process visualization with true BRT
                    denoising_save_path = os.path.join(epoch_dir, f'val_denoising_{i+1}.png')
                    visualize_denoising_with_true(
                        points_sequence,
                        true_pc.cpu().numpy(),
                        titles,
                        denoising_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/val_denoising_{i+1}': wandb.Image(denoising_save_path)})
            
            model.train()
            print()  # Add newline for better readability
    
    if wandb_api_key:
        wandb.finish()
    
    return losses, val_losses, vis_samples, val_vis_samples  # Return both training and validation samples


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
    dataset = BRTDataset(args.dataset_dir, split="train")
    
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
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=args.device,
        null_conditioning_prob=args.null_conditioning_prob
    ).to(args.device)
    
    print(f"Model initialized on {args.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset dimensions: {NUM_POINTS} points, {STATE_DIM}D coordinates, {ENV_SIZE}x{ENV_SIZE} environment")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sampling every {args.sample_every} epochs")
    print(f"Classifier-free guidance: null_conditioning_prob={args.null_conditioning_prob}, guidance_scale={args.guidance_scale}")
    
    # Train model
    losses, val_losses, vis_samples, val_vis_samples = train_model(
        model, 
        dataset, 
        num_epochs=args.num_epochs, 
        batch_size=args.batch_size,
        lr=args.lr,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        guidance_scale=args.guidance_scale
    )
    
    # Save the trained model and visualization samples
    torch.save({
        'model_state_dict': model.state_dict(),
        'vis_samples': vis_samples,
        'val_vis_samples': val_vis_samples,
        'train_losses': losses,
        'val_losses': val_losses
    }, 'brt_diffusion_model.pt')
    print("Model, visualization samples, and training losses saved to brt_diffusion_model.pt")