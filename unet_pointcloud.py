import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
import argparse
import wandb
from models.unet_baseline import BRTUNet  # Your U-Net model
from dataset.BRTDataset import BRTDataset
from utils.visualizations import visualize_comparison
from loguru import logger
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train BRT U-Net Baseline Model')
    parser.add_argument('--dataset_dir', type=str, 
                      default='1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                      help='Path to dataset directory containing sample_* folders')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (cuda/cpu)')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                      help='Weights & Biases API key (optional)')
    parser.add_argument('--wandb_project', type=str, default='brt-unet-baseline',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity name')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1000,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                      help='Minimum learning rate for cosine annealing')
    parser.add_argument('--lr_restart_period', type=int, default=50,
                      help='Initial restart period (T_0) for cosine annealing warm restarts')
    parser.add_argument('--lr_restart_mult', type=int, default=2,
                      help='Restart period multiplier (T_mult) for cosine annealing warm restarts')
    parser.add_argument('--sample_every', type=int, default=30,
                      help='Generate samples every N epochs')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                      help='Save model checkpoint every N epochs')

    # Model parameters
    parser.add_argument('--env_encoding_dim', type=int, default=128,
                      help='Environment encoding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                      help='Dropout rate for regularization')
    parser.add_argument('--weight_decay_strength', type=float, default=0.005,
                      help='L2 regularization strength')
    
    
    return parser.parse_args()


def train_model(model, dataset, num_epochs=500, batch_size=16, lr=1e-3, lr_min=1e-6, 
                lr_restart_period=50, lr_restart_mult=2, sample_every=20, checkpoint_every=50, 
                wandb_api_key=None, wandb_project='brt-unet-baseline', wandb_entity=None, 
                weight_decay_strength=0.01):
    """Training loop for the U-Net baseline model"""
    
    # Initialize wandb
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.config.update({
            'model_type': 'U-Net Baseline',
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'lr_min': lr_min,
            'lr_restart_period': lr_restart_period,
            'lr_restart_mult': lr_restart_mult,
            'sample_every': sample_every,
            'checkpoint_every': checkpoint_every,
            'num_points': model.num_points,
            'env_size': model.env_size,
            'max_state_dim': model.max_state_dim,
            'points_mean': dataset.points_mean.tolist(),
            'points_std': dataset.points_std.tolist(),
            'loss_type': 'chamfer_only',
            'dropout_rate': getattr(model, 'dropout_rate', 0.2),
            'weight_decay_strength': weight_decay_strength
        })

    # Create directories for checkpoints and samples
    run_name = wandb.run.name if wandb_api_key else datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join('checkpoints', 'unet_baseline', run_name)
    samples_dir = os.path.join('samples', 'unet_baseline', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create dataloaders for train and validation
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = BRTDataset(dataset.dataset_dir, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=lr_restart_period, T_mult=lr_restart_mult, eta_min=lr_min)
    
    # Create fixed training samples for consistent evaluation
    num_vis_samples = 8
    train_indices = torch.randint(0, len(dataset), (num_vis_samples,))
    vis_samples = [(dataset[i][0], dataset[i][1]) for i in train_indices]  # (point_cloud, env_grid) pairs
    
    # Create fixed validation samples for consistent evaluation
    num_val_vis_samples = 2
    val_indices = torch.randint(0, len(val_dataset), (num_val_vis_samples,))
    val_vis_samples = [(val_dataset[i][0], val_dataset[i][1]) for i in val_indices]
    
    model.train()
    losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training samples: {len(dataset)}, Validation samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Training loop
        model.train()
        for brt_batch, env_batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            brt_batch = brt_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
            env_batch = env_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_brt = model(env_batch, target_state_dim=brt_batch.shape[-1])
            
            # Compute loss (simple Chamfer distance only)
            loss = model.compute_loss(pred_brt, brt_batch, include_l2_reg=True)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                brt_batch = brt_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                env_batch = env_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                
                # Forward pass
                pred_brt = model(env_batch, target_state_dim=brt_batch.shape[-1])
                
                # Compute loss (without L2 reg for validation)
                loss = model.compute_loss(pred_brt, brt_batch, include_l2_reg=False)
                val_epoch_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        logger.info(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
            if wandb_api_key:
                wandb.log({
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, step=epoch)
            
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'vis_samples': vis_samples,
                'val_vis_samples': val_vis_samples,
                'train_losses': losses,
                'val_losses': val_losses,
                'config': {
                    'model_type': 'U-Net Baseline',
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'lr_min': lr_min,
                    'lr_restart_period': lr_restart_period,
                    'lr_restart_mult': lr_restart_mult,
                    'num_points': model.num_points,
                    'env_size': model.env_size,
                    'max_state_dim': model.max_state_dim,
                    'loss_type': 'chamfer_only',
                    'dropout_rate': getattr(model, 'dropout_rate', 0.2),
                    'weight_decay_strength': weight_decay_strength,
                    'run_name': run_name
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save checkpoint to wandb
            if wandb_api_key:
                artifact = wandb.Artifact(
                    name=f'unet-checkpoint-{run_name}-epoch-{epoch+1}',
                    type='model',
                    description=f'U-Net model checkpoint at epoch {epoch+1}'
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
            
        # Generate samples periodically using fixed training and validation samples
        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\nGenerating samples at epoch {epoch+1}:")
                
                # Create epoch-specific directory
                epoch_dir = os.path.join(samples_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                # Generate and plot training samples
                for i, (true_pc, env_grid) in enumerate(vis_samples):
                    # Move to device
                    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
                    env_grid = env_grid.to(device)
                    
                    # Generate sample (single forward pass for U-Net)
                    generated_brt = model(env_grid.unsqueeze(0), target_state_dim=true_pc.shape[-1])
                    generated_brt = generated_brt[0].cpu().numpy()  # Remove batch dimension
                    
                    # Create comparison visualization
                    comparison_save_path = os.path.join(epoch_dir, f'train_comparison_{i+1}.png')
                    visualize_comparison(
                        true_pc.cpu().numpy(),
                        generated_brt,
                        env_grid.cpu().numpy(),
                        f'Training Sample {i+1} Comparison (Epoch {epoch+1})',
                        comparison_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/train_comparison_{i+1}': wandb.Image(comparison_save_path)})
                
                # Generate and plot validation samples
                for i, (true_pc, env_grid) in enumerate(val_vis_samples):
                    # Move to device
                    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
                    env_grid = env_grid.to(device)
                    
                    # Generate sample (single forward pass for U-Net)
                    generated_brt = model(env_grid.unsqueeze(0), target_state_dim=true_pc.shape[-1])
                    generated_brt = generated_brt[0].cpu().numpy()  # Remove batch dimension
                    
                    # Create comparison visualization
                    comparison_save_path = os.path.join(epoch_dir, f'val_comparison_{i+1}.png')
                    visualize_comparison(
                        true_pc.cpu().numpy(),
                        generated_brt,
                        env_grid.cpu().numpy(),
                        f'Validation Sample {i+1} Comparison (Epoch {epoch+1})',
                        comparison_save_path,
                        dataset
                    )
                    if wandb_api_key:
                        wandb.log({f'epoch_{epoch+1}/val_comparison_{i+1}': wandb.Image(comparison_save_path)})
            
            model.train()
            print()  # Add newline for better readability
        
        # Step the learning rate scheduler
        scheduler.step()
    
    if wandb_api_key:
        wandb.finish()
    
    # Save the trained model and visualization samples
    final_model_path = os.path.join('models', f'brt_unet_baseline_{run_name}.pt')
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vis_samples': vis_samples,
        'val_vis_samples': val_vis_samples,
        'train_losses': losses,
        'val_losses': val_losses,
        'config': {
            'model_type': 'U-Net Baseline',
            'batch_size': batch_size,
            'learning_rate': lr,
            'lr_min': lr_min,
            'lr_restart_period': lr_restart_period,
            'lr_restart_mult': lr_restart_mult,
            'num_points': model.num_points,
            'env_size': model.env_size,
            'max_state_dim': model.max_state_dim,
            'loss_type': 'chamfer_only',
            'dropout_rate': getattr(model, 'dropout_rate', 0.2),
            'weight_decay_strength': weight_decay_strength,
            'run_name': run_name
        }
    }, final_model_path)
    print(f"Model, visualization samples, and training losses saved to {final_model_path}")
    return losses, val_losses, vis_samples, val_vis_samples


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
    
    # Initialize U-Net model
    model = BRTUNet(
        env_size=ENV_SIZE,
        num_points=NUM_POINTS,
        max_state_dim=STATE_DIM,
        env_encoding_dim=args.env_encoding_dim,
        dropout_rate=args.dropout_rate,
        weight_decay_strength=args.weight_decay_strength
    ).to(args.device)
    
    # Set device attribute for convenience
    model.device = args.device
    
    print(f"U-Net Model initialized on {args.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset dimensions: {NUM_POINTS} points, {STATE_DIM}D coordinates, {ENV_SIZE}x{ENV_SIZE} environment")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    print(f"Learning rate: {args.lr} -> {args.lr_min} (cosine annealing warm restarts, T_0={args.lr_restart_period}, T_mult={args.lr_restart_mult})")
    print(f"Sampling every {args.sample_every} epochs")
    print(f"Regularization: dropout_rate={args.dropout_rate}, weight_decay_strength={args.weight_decay_strength}")
    
    # Train model
    losses, val_losses, vis_samples, val_vis_samples = train_model(
        model, 
        dataset, 
        num_epochs=args.num_epochs, 
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_restart_period=args.lr_restart_period,
        lr_restart_mult=args.lr_restart_mult,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        weight_decay_strength=args.weight_decay_strength
    )
    
    print("Training completed successfully!")