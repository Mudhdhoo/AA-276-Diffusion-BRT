#!/usr/bin/env python3
"""
Comprehensive model evaluation script for BRT generation models.
Supports UNet and Diffusion models with automatic 3D/4D detection.
Evaluates using Chamfer distance and value function L2 error.
Now includes wandb logging and artifact support.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json
from loguru import logger
from scipy.spatial.distance import cdist
from scipy.interpolate import interpn
import wandb

# Add project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.BRTDataset import BRTDataset
from models.unet_baseline import BRTUNet
from models.diffusion_modules import BRTDiffusionModel
from utils.visualizations import visualize_comparison, visualize_detailed_value_function_comparison

# Import the existing gif generation functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'visualization'))
try:
    from generate_gif import generate_denoising_gif as generate_gif_3d
    from generate_gif_4d import generate_denoising_gif as generate_gif_4d
    GIF_GENERATION_AVAILABLE = True
    logger.info("GIF generation functions imported successfully")
except ImportError as e:
    logger.warning(f"Could not import GIF generation functions: {e}")
    GIF_GENERATION_AVAILABLE = False

# Define default artifacts for models
DEFAULT_ARTIFACTS = {
    'unet': 'malteny-stanford/brt-unet-baseline/unet-checkpoint-deep-wave-6-epoch-50:v0',
    'diffusion': 'malteny-stanford/brt-diffusion/model-checkpoint-snowy-vortex-24-epoch-2000:v0'
}

def download_wandb_artifact(artifact_path, filename=None):
    """
    Download a wandb artifact and return the path to the checkpoint file.
    
    Args:
        artifact_path: Full wandb artifact path (e.g., 'entity/project/artifact_name:version')
        filename: Specific filename to look for in artifact (e.g., 'checkpoint_epoch_50.pt')
    
    Returns:
        Path to the downloaded checkpoint file
    """
    logger.info(f"Downloading wandb artifact: {artifact_path}")
    
    try:
        # For cross-project artifact access, we don't need a run context
        # Just initialize wandb without a run
        wandb.init(mode="disabled")  # Initialize without creating a run
        
        # Use the API directly for artifact access
        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        
        # Download the artifact
        artifact_dir = artifact.download()
        
        # Find checkpoint file
        if filename:
            checkpoint_path = os.path.join(artifact_dir, filename)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file {filename} not found in artifact")
        else:
            # Look for common checkpoint patterns
            checkpoint_patterns = ['*.pt', '*.pth', 'checkpoint*.pt', 'model*.pt']
            checkpoint_path = None
            for pattern in checkpoint_patterns:
                import glob
                matches = glob.glob(os.path.join(artifact_dir, pattern))
                if matches:
                    checkpoint_path = matches[0]
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint file found in artifact")
        
        logger.info(f"Successfully downloaded checkpoint: {checkpoint_path}")
        wandb.finish()
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Failed to download artifact {artifact_path}: {e}")
        # Try alternative approach using public artifact access
        try:
            logger.info(f"Trying alternative download method for {artifact_path}")
            
            # Parse artifact path
            parts = artifact_path.split('/')
            if len(parts) >= 3:
                entity = parts[0]
                project = parts[1] 
                artifact_name_version = '/'.join(parts[2:])
                
                # Try accessing with entity/project context
                api = wandb.Api()
                artifact = api.artifact(f"{entity}/{project}/{artifact_name_version}")
                artifact_dir = artifact.download()
                
                # Find checkpoint file
                if filename:
                    checkpoint_path = os.path.join(artifact_dir, filename)
                    if not os.path.exists(checkpoint_path):
                        raise FileNotFoundError(f"Checkpoint file {filename} not found in artifact")
                else:
                    # Look for common checkpoint patterns
                    checkpoint_patterns = ['*.pt', '*.pth', 'checkpoint*.pt', 'model*.pt']
                    checkpoint_path = None
                    for pattern in checkpoint_patterns:
                        import glob
                        matches = glob.glob(os.path.join(artifact_dir, pattern))
                        if matches:
                            checkpoint_path = matches[0]
                            break
                    
                    if checkpoint_path is None:
                        raise FileNotFoundError("No checkpoint file found in artifact")
                
                logger.info(f"Successfully downloaded checkpoint via alternative method: {checkpoint_path}")
                return checkpoint_path
                
        except Exception as e2:
            logger.error(f"Alternative download method also failed: {e2}")
            
        wandb.finish()
        raise

class ModelEvaluator:
    """Handles evaluation of different model types"""
    
    def __init__(self, device='cuda', save_visualizations=True, wandb_logging=False, wandb_project="model-evaluation"):
        self.device = device
        self.save_visualizations = save_visualizations
        self.max_vis_samples = 10  # Default value
        self.vis_sample_count = 0  # Track number of visualizations created
        self.wandb_logging = wandb_logging
        self.wandb_project = wandb_project
        self.wandb_run = None
        self.current_split = None  # Track current split being evaluated
        self.current_model_type = None  # Track current model type being evaluated
        
        # Initialize wandb if requested
        if self.wandb_logging:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                job_type="evaluation",
                tags=["model-evaluation"],
                settings=wandb.Settings(start_method="fork")  # Better for multiprocessing
            )
    
    def __del__(self):
        """Clean up wandb run on destruction"""
        if self.wandb_run is not None:
            wandb.finish()
        
    def load_model_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint and detect type automatically"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = checkpoint.get('config', {})
        
        # Detect model type
        model_type = config.get('model_type', 'Unknown')
        
        if 'U-Net' in model_type or 'unet' in model_type.lower():
            model = self._load_unet_model(checkpoint, config)
            model_info = {
                'type': 'UNet',
                'state_dim': config.get('max_state_dim', 4),
                'config': config
            }
        else:
            # Assume diffusion model
            model, state_dim = self._load_diffusion_model(checkpoint, config)
            model_info = {
                'type': 'Diffusion',
                'state_dim': state_dim,  # Use inferred state_dim
                'config': config
            }
            
        # Log model size and memory before moving to GPU
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Model device before moving: {next(model.parameters()).device}")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Log memory usage after moving to GPU
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        logger.info(f"Loaded {model_info['type']} model with {model_info['state_dim']}D output")
        logger.info(f"Model device after moving: {next(model.parameters()).device}")
        return model, model_info
    
    def _load_unet_model(self, checkpoint, config):
        """Load UNet model"""
        model = BRTUNet(
            env_size=config.get('env_size', 64),
            num_points=config.get('num_points', 4000),
            max_state_dim=config.get('max_state_dim', 4),
            env_encoding_dim=128,  # Default value
            dropout_rate=config.get('dropout_rate', 0.2),
            weight_decay_strength=config.get('weight_decay_strength', 0.01)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _load_diffusion_model(self, checkpoint, config):
        """Load Diffusion model and infer state_dim from checkpoint"""
        # Try to infer state_dim from model state dict
        state_dim = self._infer_diffusion_state_dim(checkpoint['model_state_dict'])
        
        model = BRTDiffusionModel(
            state_dim=state_dim,
            env_size=config.get('env_size', 64),
            num_points=config.get('num_points', 4000),
            num_timesteps=config.get('num_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            device=self.device,
            null_conditioning_prob=config.get('null_conditioning_prob', 0.1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, state_dim  # Return the inferred state_dim
    
    def _infer_diffusion_state_dim(self, state_dict):
        """Infer state_dim from diffusion model state dict"""
        # Look for the final output projection layer
        for key, tensor in state_dict.items():
            if 'denoiser.output_proj.2.weight' in key:  # Last layer of sequential
                return tensor.shape[0]
            elif 'output_proj' in key and 'weight' in key and tensor.dim() == 2:
                # Check if this looks like final layer (small output size)
                if tensor.shape[0] in [3, 4]:
                    return tensor.shape[0]
        
        # Fallback: assume 4D
        logger.warning("Could not infer state_dim from checkpoint, assuming 4D")
        return 4
    
    def generate_samples(self, model, model_info, env_batch, target_state_dim=None):
        """Generate samples from model"""
        model_type = model_info['type']
        batch_size = env_batch.shape[0]
        
        with torch.no_grad():
            if model_type == 'UNet':
                # UNet forward pass
                if target_state_dim is None:
                    target_state_dim = model_info['state_dim']
                predicted = model(env_batch, target_state_dim=target_state_dim)
                
            elif model_type == 'Diffusion':
                # Fixed: Batch diffusion sampling for much better GPU efficiency
                # Generate one sample per environment in the batch
                predicted = self.sample_batch_diffusion(model, env_batch, guidance_scale=1.5)
                
                if target_state_dim is not None and predicted.shape[-1] > target_state_dim:
                    predicted = predicted[:, :, :target_state_dim]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return predicted
    
    def sample_batch_diffusion(self, model, env_batch, guidance_scale=1.0):
        """
        Efficient batch sampling for diffusion models.
        Generate one sample per environment in the batch.
        
        The original model.sample() method takes a single environment and generates 
        num_samples for that environment. But we want to generate one sample per 
        different environment in env_batch.
        """
        batch_size = env_batch.shape[0]
        
        # Start from pure noise for all environments
        x_t = torch.randn(batch_size, model.num_points, model.state_dim).to(model.device)
        
        # Reverse diffusion process with guidance
        # The p_sample method properly handles batched environments and batched x_t
        for t in reversed(range(model.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=model.device, dtype=torch.long)
            x_t = model.p_sample(x_t, t_batch, env_batch, guidance_scale)
        
        return x_t
    
    def compute_value_function_l2_error(self, pred_points, target_points, value_function, dataset):
        """
        Compute L2 error between predicted values and interpolated true values.
        Returns per-pointcloud mean and std L2 error.
        """
        batch_size = pred_points.shape[0]
        l2_means = []
        l2_stds = []
        total_points = 0
        
        for b in range(batch_size):
            pred_denorm = dataset.denormalize_points(pred_points[b].cpu().numpy())  # (N, 4)
            value_func_3d = value_function[b].cpu().numpy()  # (64, 64, 64)
            coords = pred_denorm[:, :3]
            pred_values = pred_denorm[:, 3]
            x_indices = (coords[:, 0] / 10.0 * 64).clip(0, 63)
            y_indices = (coords[:, 1] / 10.0 * 64).clip(0, 63)
            theta_indices = ((coords[:, 2] + np.pi) / (2 * np.pi) * 64).clip(0, 63)
            grid_coords = np.stack([x_indices, y_indices, theta_indices], axis=1)
            x_coords = np.arange(64)
            y_coords = np.arange(64)
            theta_coords = np.arange(64)
            try:
                interpolated_values = interpn(
                    (x_coords, y_coords, theta_coords),
                    value_func_3d,
                    grid_coords,
                    method='linear',
                    bounds_error=False,
                    fill_value=1.0
                )
                errors = (pred_values - interpolated_values)
                l2 = (errors ** 2)
                l2_means.append(np.mean(l2))
                l2_stds.append(np.std(l2))
                total_points += len(pred_values)
            except Exception as e:
                l2_means.append(float('inf'))
                l2_stds.append(float('inf'))
        return l2_means, l2_stds, total_points
    
    def generate_denoising_gif_for_evaluation(self, model, model_info, dataset, sample_idx, output_dir):
        """Generate denoising GIF for diffusion models during evaluation"""
        if not GIF_GENERATION_AVAILABLE:
            logger.warning("GIF generation not available - skipping")
            return None
            
        if model_info['type'] != 'Diffusion':
            logger.info(f"Skipping GIF generation for {model_info['type']} model")
            return None
            
        gif_dir = os.path.join(output_dir, 'denoising_gifs')
        os.makedirs(gif_dir, exist_ok=True)
        
        try:
            logger.info(f"Generating denoising GIF for sample {sample_idx}")
            
            if model_info['state_dim'] == 4:
                # Use 4D GIF generation
                gif_path = generate_gif_4d(
                    model=model,
                    dataset=dataset,
                    sample_idx=sample_idx,
                    num_frames=30,  # Fewer frames for faster generation
                    save_dir=gif_dir
                )
            else:
                # Use 3D GIF generation  
                gif_path = generate_gif_3d(
                    model=model,
                    dataset=dataset,
                    sample_idx=sample_idx,
                    num_frames=30,
                    save_dir=gif_dir
                )
                
            logger.info(f"Successfully generated denoising GIF for sample {sample_idx}")
            
            # Log to wandb if enabled
            if self.wandb_logging and gif_path and os.path.exists(gif_path):
                try:
                    wandb.log({
                        f"denoising_gif/sample_{sample_idx}": wandb.Video(gif_path, format="gif"),
                        f"denoising_gif/model_type": model_info['type'],
                        f"denoising_gif/state_dim": model_info['state_dim']
                    })
                    logger.info(f"Logged denoising GIF to wandb: {gif_path}")
                except Exception as e:
                    logger.warning(f"Failed to log GIF to wandb: {e}")
            
            return gif_path
            
        except Exception as e:
            logger.error(f"Failed to generate denoising GIF: {e}")
            return None

    def save_evaluation_visualizations(self, predicted_points, target_points, env_batch, 
                                     value_functions, model_info, batch_idx, output_dir, dataset, l2_title=None):
        """Save visualization of predictions for sanity check"""
        if not self.save_visualizations:
            return
            
        # Check if we've reached the maximum number of visualizations
        if hasattr(self, 'vis_sample_count') and hasattr(self, 'max_vis_samples'):
            if self.vis_sample_count >= self.max_vis_samples:
                return
        
        os.makedirs(output_dir, exist_ok=True)
        model_name = f"{model_info['type']}_{model_info['state_dim']}D"
        split_info = f"_{self.current_split}" if self.current_split else ""
        
        # Visualize first sample in batch
        pred_sample = predicted_points[0].cpu().numpy()
        target_sample = target_points[0].cpu().numpy()
        env_sample = env_batch[0].cpu().numpy()
        value_func_sample = value_functions[0].cpu().numpy() if value_functions is not None else None
                
        # Create regular comparison visualization with proper denormalization
        save_path = os.path.join(output_dir, f"{model_name}{split_info}_sample_{self.vis_sample_count+1:03d}_comparison.png")
        visualize_comparison(
            target_sample, pred_sample, env_sample,
            title=f"{model_name} - {self.current_split.capitalize()} Split - Sample {self.vis_sample_count+1} Comparison",
            save_path=save_path,
            dataset=dataset  # Pass dataset for proper coordinate denormalization
        )
        
        # Create detailed comparison visualization with theta slices
        detailed_save_path = os.path.join(output_dir, f"{model_name}{split_info}_sample_{self.vis_sample_count+1:03d}_detailed.png")
        detailed_title = l2_title if l2_title is not None else f"{model_name} - {self.current_split.capitalize()} Split - Sample {self.vis_sample_count+1} Detailed Analysis"
        visualize_detailed_value_function_comparison(
            target_sample, pred_sample, env_sample,
            title=detailed_title,
            save_path=detailed_save_path,
            dataset=dataset,
        )
        
        # Log visualizations to wandb if enabled
        if self.wandb_logging:
            try:
                # Create hierarchical logging structure
                base_path = f"{model_info['type'].lower()}/{self.current_split}"
                
                # Log comparison plot
                wandb.log({
                    f"{base_path}/comparison_{self.vis_sample_count+1}": wandb.Image(save_path),
                    f"{base_path}/detailed_{self.vis_sample_count+1}": wandb.Image(detailed_save_path),
                    f"{base_path}/model_type": model_info['type'],
                    f"{base_path}/state_dim": model_info['state_dim'],
                    f"{base_path}/sample_idx": self.vis_sample_count+1,
                })
                logger.info(f"Logged visualization plots to wandb under {base_path}")
            except Exception as e:
                logger.warning(f"Failed to log visualization plots to wandb: {e}")
        
        # Generate denoising GIF for diffusion models (for ALL visualized samples)
        if model_info['type'] == 'Diffusion':
            # Use a deterministic sample from the dataset for GIF generation
            sample_idx = (self.vis_sample_count * 137) % min(100, len(dataset))  # Deterministic but varied
            logger.info(f"Generating denoising GIF for diffusion model - visualization {self.vis_sample_count+1}/{self.max_vis_samples}, dataset sample {sample_idx}")
            gif_path = self.generate_denoising_gif_for_evaluation(
                model_info.get('model_instance'),  # We'll need to pass the model instance
                model_info, 
                dataset, 
                sample_idx, 
                output_dir
            )
            if gif_path:
                logger.info(f"Successfully generated GIF for visualization {self.vis_sample_count+1}: {gif_path}")
                # Log GIF to wandb
                if self.wandb_logging:
                    try:
                        wandb.log({
                            f"{base_path}/denoising_gif_{self.vis_sample_count+1}": wandb.Video(gif_path, format="gif")
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log GIF to wandb: {e}")
            else:
                logger.warning(f"Failed to generate GIF for visualization {self.vis_sample_count+1}")
        
        # Increment visualization counter
        self.vis_sample_count += 1
        
        logger.info(f"Saved visualizations ({self.vis_sample_count}/{self.max_vis_samples}): {save_path} and {detailed_save_path}")
    
    def evaluate_model(self, model, model_info, test_loader, dataset_with_vf, output_dir):
        """Evaluate model on test set"""
        model_type = model_info['type']
        state_dim = model_info['state_dim']
        
        # Store model instance for GIF generation
        model_info['model_instance'] = model
        
        # Update current model type
        self.current_model_type = model_type
        
        logger.info(f"Evaluating {model_type} model with {state_dim}D output")
        logger.info(f"DataLoader info: batch_size={test_loader.batch_size}, num_workers={test_loader.num_workers}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Log model info to wandb if enabled
        if self.wandb_logging:
            base_path = f"{model_type.lower()}/{self.current_split}"
            wandb.log({
                f"{base_path}/model_info/type": model_type,
                f"{base_path}/model_info/state_dim": state_dim,
                f"{base_path}/model_info/config": model_info.get('config', {})
            })
        
        all_l2_means = []
        all_l2_stds = []
        total_points_evaluated = 0
        vis_samples = []
        batch_count = 0
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            logger.debug(f"Processing batch {batch_idx}")
            with logger.contextualize(batch=batch_idx):
                # Load and move data to GPU
                with logger.contextualize(operation="data_loading"):
                    if len(batch_data) == 3:
                        target_points, env_batch, value_functions = [d.to(self.device) for d in batch_data]
                    else:
                        target_points, env_batch = [d.to(self.device) for d in batch_data]
                        value_functions = None
                
                # Generate predictions
                with logger.contextualize(operation="model_forward"):
                    predicted_points = self.generate_samples(model, model_info, env_batch, target_state_dim=state_dim)
                
                # Compute metrics
                with logger.contextualize(operation="metrics"):
                    # Compute value function error if available
                    if value_functions is not None and state_dim == 4:
                        l2_means, l2_stds, batch_points = self.compute_value_function_l2_error(
                            predicted_points, target_points, value_functions, dataset_with_vf
                        )
                        all_l2_means.extend(l2_means)
                        all_l2_stds.extend(l2_stds)
                        total_points_evaluated += batch_points
                        
                        # Log batch metrics to wandb
                        if self.wandb_logging:
                            wandb.log({
                                f"{model_type.lower()}/{self.current_split}/batch_value_l2_error_mean": np.mean(l2_means),
                                f"{model_type.lower()}/{self.current_split}/batch_value_l2_error_std": np.std(l2_stds),
                                f"{model_type.lower()}/{self.current_split}/batch_idx": batch_idx,
                                f"{model_type.lower()}/{self.current_split}/total_points": total_points_evaluated
                            })
                
                # Store samples for visualization if needed
                if self.vis_sample_count < self.max_vis_samples:
                    vis_samples.append((predicted_points, target_points, env_batch, value_functions, l2_means, l2_stds))
                
                batch_count += 1
                
                # Log timing every 10 batches
                if batch_count % 10 == 0:
                    logger.info(f"Completed {batch_count} batches")
        
        # Generate visualizations after evaluation
        if self.save_visualizations and vis_samples:
            logger.info("Generating visualizations...")
            for batch_idx, (predicted_points, target_points, env_batch, value_functions, l2_means, l2_stds) in enumerate(vis_samples):
                if self.vis_sample_count >= self.max_vis_samples:
                    break
                # Use the first sample in the batch for visualization
                sample_l2_mean = l2_means[0] if l2_means else None
                sample_l2_std = l2_stds[0] if l2_stds else None
                l2_title = f"L2 Error: {sample_l2_mean:.6f} (std: {sample_l2_std:.6f})" if sample_l2_mean is not None else None
                self.save_evaluation_visualizations(
                    predicted_points, target_points, env_batch, 
                    value_functions, model_info, batch_idx, output_dir, dataset_with_vf,
                    l2_title=l2_title
                )
        
        # Compute final metrics
        results = {
            'model_type': model_type,
            'state_dim': state_dim,
            'config': model_info['config'],
        }
        
        if all_l2_means:
            results.update({
                'mean_l2_error_across_pointclouds': np.mean(all_l2_means),
                'std_l2_error_across_pointclouds': np.std(all_l2_means),
                'num_pointclouds': len(all_l2_means),
                'num_value_samples': total_points_evaluated,
                'per_pointcloud_l2_means': all_l2_means,
                'per_pointcloud_l2_stds': all_l2_stds,
            })
            
            # Log final metrics to wandb
            if self.wandb_logging:
                base_path = f"{model_type.lower()}/{self.current_split}"
                wandb.log({
                    f"{base_path}/final_metrics/mean_l2_error_across_pointclouds": results['mean_l2_error_across_pointclouds'],
                    f"{base_path}/final_metrics/std_l2_error_across_pointclouds": results['std_l2_error_across_pointclouds'],
                    f"{base_path}/final_metrics/num_pointclouds": results['num_pointclouds'],
                    f"{base_path}/final_metrics/num_value_samples": results['num_value_samples']
                })
                
                # Create and log summary table
                summary_data = [
                    [model_type, state_dim, self.current_split, 
                     results['mean_l2_error_across_pointclouds'], 
                     results['std_l2_error_across_pointclouds'], 
                     results['num_pointclouds']]
                ]
                summary_table = wandb.Table(
                    data=summary_data, 
                    columns=["Model Type", "State Dim", "Split", "Mean L2 Error", "Std L2 Error", "Pointclouds"]
                )
                wandb.log({f"{base_path}/evaluation_summary": summary_table})
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate BRT generation models')
    parser.add_argument('--dataset_dir', type=str, 
                      default='1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                      help='Path to dataset directory')
    parser.add_argument('--checkpoints', nargs='*', 
                      help='List of checkpoint files to evaluate (can be local paths or wandb artifacts)')
    parser.add_argument('--use_default_artifacts', action='store_true',
                      help='Use default wandb artifacts for UNet and Diffusion models')
    parser.add_argument('--unet_artifact', type=str, 
                      default=DEFAULT_ARTIFACTS['unet'],
                      help='Wandb artifact path for UNet model')
    parser.add_argument('--diffusion_artifact', type=str,
                      default=DEFAULT_ARTIFACTS['diffusion'], 
                      help='Wandb artifact path for Diffusion model')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for evaluation')
    parser.add_argument('--output_base_dir', type=str, default='evaluation_output',
                      help='Base output directory for results and visualizations')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    parser.add_argument('--no_visualizations', action='store_true',
                      help='Disable saving visualizations')
    parser.add_argument('--max_vis_samples', type=int, default=4,
                      help='Maximum number of samples to visualize per model')
    parser.add_argument('--splits', nargs='*', default=['test'],
                      help='Dataset splits to evaluate on')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Wandb arguments
    parser.add_argument('--wandb_logging', action='store_true', default=True,
                      help='Enable wandb logging for plots and metrics')
    parser.add_argument('--wandb_project', type=str, default='brt-model-evaluation',
                      help='Wandb project name for logging')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Wandb entity name')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                      help='Wandb API key (optional, can use environment variable)')
    
    args = parser.parse_args()
    
    # Set wandb API key if provided
    if args.wandb_api_key:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
        logger.info("Wandb API key set from command line argument")
    elif 'WANDB_API_KEY' in os.environ:
        logger.info("Wandb API key found in environment variables")
    elif args.wandb_logging or args.use_default_artifacts:
        logger.warning("Wandb functionality requested but no API key provided. This may cause authentication issues.")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine which checkpoints to evaluate
    checkpoints_to_evaluate = []
    
    if args.use_default_artifacts:
        logger.info("Using default wandb artifacts...")
        # Download UNet artifact
        try:
            unet_checkpoint = download_wandb_artifact(args.unet_artifact, 'checkpoint_epoch_50.pt')
            checkpoints_to_evaluate.append(('unet_default', unet_checkpoint))
        except Exception as e:
            logger.error(f"Failed to download UNet artifact: {e}")
        
        # Download Diffusion artifact
        try:
            diffusion_checkpoint = download_wandb_artifact(args.diffusion_artifact, 'checkpoint_epoch_2000.pt')
            checkpoints_to_evaluate.append(('diffusion_default', diffusion_checkpoint))
        except Exception as e:
            logger.error(f"Failed to download Diffusion artifact: {e}")
    
    # Add any manually specified checkpoints
    if args.checkpoints:
        for checkpoint in args.checkpoints:
            # Check if it's a wandb artifact path (entity/project/artifact:version format)
            if ('/' in checkpoint and ':' in checkpoint and not os.path.exists(checkpoint)) or checkpoint.count('/') >= 2:
                # Looks like a wandb artifact path
                try:
                    local_checkpoint = download_wandb_artifact(checkpoint)
                    checkpoints_to_evaluate.append((f'artifact_{os.path.basename(checkpoint)}', local_checkpoint))
                except Exception as e:
                    logger.error(f"Failed to download artifact {checkpoint}: {e}")
            else:
                # Local file path
                if os.path.exists(checkpoint):
                    checkpoints_to_evaluate.append((f'local_{os.path.basename(checkpoint)}', checkpoint))
                else:
                    logger.error(f"Checkpoint not found: {checkpoint}")
                    # Try as artifact path as fallback
                    try:
                        logger.info(f"Trying to interpret {checkpoint} as wandb artifact...")
                        local_checkpoint = download_wandb_artifact(checkpoint)
                        checkpoints_to_evaluate.append((f'artifact_{os.path.basename(checkpoint)}', local_checkpoint))
                    except Exception as e2:
                        logger.error(f"Also failed as artifact: {e2}")
    
    if not checkpoints_to_evaluate:
        logger.error("No valid checkpoints found to evaluate!")
        return
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_base_dir, f"eval_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = torch.device(args.device)
    
    evaluator = ModelEvaluator(
        device=device, 
        save_visualizations=not args.no_visualizations,
        wandb_logging=args.wandb_logging,
        wandb_project=args.wandb_project
    )
    
    # Initialize wandb for the overall evaluation run if requested
    if args.wandb_logging:
        # Configure wandb
        wandb_config = {
            'dataset_dir': args.dataset_dir,
            'splits': args.splits,
            'batch_size': args.batch_size,
            'device': args.device,
            'seed': args.seed,
            'max_vis_samples': args.max_vis_samples,
            'num_checkpoints': len(checkpoints_to_evaluate),
            'timestamp': timestamp
        }
        
        # Set up wandb entity if provided
        if args.wandb_entity:
            wandb_config['entity'] = args.wandb_entity
    
    # Evaluate each checkpoint across all splits
    all_results = []
    results_summary = []
    
    for checkpoint_idx, (checkpoint_name, checkpoint_path) in enumerate(checkpoints_to_evaluate):
        for split in args.splits:
            try:
                logger.info(f"Evaluating checkpoint {checkpoint_idx+1}/{len(checkpoints_to_evaluate)} on {split} split: {checkpoint_name} ({checkpoint_path})")
                
                # Load model
                model, model_info = evaluator.load_model_from_checkpoint(checkpoint_path)
                
                # Load dataset for current split
                logger.info(f"Loading {split} dataset...")
                dataset = BRTDataset(args.dataset_dir, split=split)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                       shuffle=False, num_workers=args.num_workers)
                
                # Load dataset with value functions for interpolation
                dataset_with_vf = BRTDataset(args.dataset_dir, split=split, return_value_function=True)
                data_loader_with_vf = DataLoader(dataset_with_vf, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers)
                
                logger.info(f"{split.capitalize()} dataset loaded: {len(dataset)} samples")
                
                # Choose appropriate data loader
                if model_info['state_dim'] == 4:
                    loader = data_loader_with_vf
                    dataset_for_denorm = dataset_with_vf
                else:
                    loader = data_loader
                    dataset_for_denorm = dataset
                
                # Update evaluator to limit visualizations and set split
                evaluator.max_vis_samples = args.max_vis_samples
                evaluator.vis_sample_count = 0
                evaluator.current_split = split  # Add this to ModelEvaluator class
                
                # Create split-specific output directory
                split_output_dir = os.path.join(args.output_dir, f"{checkpoint_name}_{split}")
                os.makedirs(split_output_dir, exist_ok=True)
                
                # Evaluate
                results = evaluator.evaluate_model(model, model_info, loader, dataset_for_denorm, split_output_dir)
                results['checkpoint_path'] = checkpoint_path
                results['checkpoint_name'] = checkpoint_name
                results['checkpoint_basename'] = os.path.basename(checkpoint_path)
                results['evaluation_timestamp'] = timestamp
                results['split'] = split
                
                # Log results
                logger.info(f"Results for {checkpoint_name} on {split} split:")
                logger.info(f"  Model Type: {results['model_type']}")
                logger.info(f"  State Dim: {results['state_dim']}")
                logger.info(f"  Mean L2 Error: {results['mean_l2_error_across_pointclouds']:.6f} ± {results['std_l2_error_across_pointclouds']:.6f}")
                
                all_results.append(results)
                
                # Create summary for this model and split
                summary = f"Model: {results['model_type']} ({results['state_dim']}D)\n"
                summary += f"Split: {split}\n"
                summary += f"Checkpoint: {checkpoint_name}\n"
                summary += f"Mean L2 Error: {results['mean_l2_error_across_pointclouds']:.6f} ± {results['std_l2_error_across_pointclouds']:.6f}\n"
                summary += f"Samples evaluated: {results['num_value_samples']}\n"
                summary += "\n"
                results_summary.append(summary)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {checkpoint_name} on {split} split: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Write detailed results to text file with both per-pointcloud and whole-split statistics
    results_text_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_text_file, 'w') as f:
        f.write(f"BRT Model Evaluation Results\n")
        f.write(f"============================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Max visualizations per model: {args.max_vis_samples}\n\n")
        
        # Write summary for each model and split
        f.write("SUMMARY\n")
        f.write("="*50 + "\n")
        for summary in results_summary:
            f.write(summary)
        
        # Write detailed statistics
        f.write("="*70 + "\n")
        f.write("DETAILED STATISTICS (PER-POINTCLOUD AND WHOLE-SPLIT)\n")
        f.write("="*70 + "\n\n")
        
        for result in all_results:
            f.write(f"Model: {result['model_type']} ({result['state_dim']}D)\n")
            f.write(f"Split: {result['split']}\n")
            f.write(f"Checkpoint: {result['checkpoint_name']}\n")
            f.write("-" * 50 + "\n")
            
            # Whole Split Statistics
            f.write(f"WHOLE SPLIT STATISTICS:\n")
            if 'mean_l2_error_across_pointclouds' in result:
                f.write(f"  Mean L2 Error (across all pointclouds): {result['mean_l2_error_across_pointclouds']:.8f}\n")
                f.write(f"  Std L2 Error (across all pointclouds):  {result['std_l2_error_across_pointclouds']:.8f}\n")
                f.write(f"  Number of pointclouds evaluated:       {result['num_pointclouds']}\n")
                f.write(f"  Total value function points sampled:   {result['num_value_samples']:,}\n")
            
            # Per-Pointcloud Statistics (show first 20)
            f.write(f"\nPER-POINTCLOUD STATISTICS (first 20 samples):\n")
            if 'per_pointcloud_l2_means' in result:
                per_pc_means = result['per_pointcloud_l2_means'][:20]
                per_pc_stds = result['per_pointcloud_l2_stds'][:20]
                f.write(f"  Pointcloud L2 means: {[f'{v:.6f}' for v in per_pc_means]}\n")
                f.write(f"  Pointcloud L2 stds:  {[f'{v:.6f}' for v in per_pc_stds]}\n")
                
                if len(result['per_pointcloud_l2_means']) > 20:
                    f.write(f"  ... (showing first 20 of {len(result['per_pointcloud_l2_means'])} total pointclouds)\n")
            
            f.write(f"\nModel Configuration:\n")
            for key, value in result.get('config', {}).items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")
    
    # Save all results and plots as wandb artifacts if logging enabled
    if args.wandb_logging and all_results:
        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append([
                result['checkpoint_name'],
                result['model_type'],
                result['state_dim'],
                result['split'],
                result['mean_l2_error_across_pointclouds'],
                result['std_l2_error_across_pointclouds'],
                result['num_value_samples']
            ])
        
        comparison_table = wandb.Table(
            data=comparison_data,
            columns=["Checkpoint", "Model Type", "State Dim", "Split", "Mean L2 Error", "Std L2 Error", "Samples"]
        )
        wandb.log({"evaluation_comparison": comparison_table})
        
        # Save result files as wandb artifacts
        artifact = wandb.Artifact(
            name=f"evaluation_results_{timestamp}",
            type="evaluation_results",
            description=f"Model evaluation results and statistics for {len(checkpoints_to_evaluate)} models"
        )
        
        # Add result files to artifact
        if os.path.exists(results_file):
            artifact.add_file(results_file, name="evaluation_results.json")
        if os.path.exists(results_text_file):
            artifact.add_file(results_text_file, name="evaluation_results.txt")
            
        # Add all visualization directories to artifact
        for checkpoint_name, _ in checkpoints_to_evaluate:
            for split in args.splits:
                split_output_dir = os.path.join(args.output_dir, f"{checkpoint_name}_{split}")
                if os.path.exists(split_output_dir):
                    artifact.add_dir(split_output_dir, name=f"visualizations/{checkpoint_name}_{split}")
        
        # Log the artifact
        wandb.log_artifact(artifact)
        logger.info(f"Saved evaluation results and visualizations as wandb artifact: evaluation_results_{timestamp}")
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")
    logger.info(f"  JSON results: {results_file}")
    logger.info(f"  Text results: {results_text_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluation completed at: {timestamp}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Results saved to: {args.output_dir}")
    if args.wandb_logging:
        print(f"Wandb project: {args.wandb_project}")
        print(f"Wandb artifact: evaluation_results_{timestamp}")
    print()
    
    for summary in results_summary:
        print(summary)

if __name__ == '__main__':
    main() 