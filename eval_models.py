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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import wandb
import tempfile
import shutil
import time

# Add project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.BRTDataset import BRTDataset
from models.unet_baseline import BRTUNet
from models.diffusion_modules import BRTDiffusionModel
from utils.visualizations import visualize_comparison, visualize_point_cloud, visualize_environment_grid, visualize_detailed_value_function_comparison, create_dual_colormap

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
                # Diffusion sampling
                predicted_list = []
                for i in range(batch_size):
                    # Sample one environment at a time for diffusion
                    env_single = env_batch[i:i+1]
                    sample = model.sample(env_single, num_samples=1, guidance_scale=1.5)
                    if target_state_dim is not None and sample.shape[-1] > target_state_dim:
                        sample = sample[:, :, :target_state_dim]
                    predicted_list.append(sample[0])  # Remove batch dimension
                predicted = torch.stack(predicted_list, dim=0)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return predicted
    
    def compute_chamfer_distance(self, pred_points, target_points, chunk_size=500):
        """Compute chunked Chamfer distance for large point clouds"""
        batch_size, N, D = pred_points.shape
        _, M, _ = target_points.shape
        
        total_chamfer = 0.0
        
        for b in range(batch_size):
            pred_b = pred_points[b].cpu().numpy()  # (N, D)
            target_b = target_points[b].cpu().numpy()  # (M, D)
            
            # Chunked computation for memory efficiency
            min_dists_pred_to_target = []
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                pred_chunk = pred_b[i:end_i]  # (chunk_size, D)
                
                # Compute distances from pred_chunk to all target points
                dists = cdist(pred_chunk, target_b, metric='euclidean')  # (chunk_size, M)
                min_dists = np.min(dists, axis=1)  # (chunk_size,)
                min_dists_pred_to_target.append(min_dists)
            
            min_dists_pred_to_target = np.concatenate(min_dists_pred_to_target)
            
            # Compute target to pred distances
            min_dists_target_to_pred = []
            for j in range(0, M, chunk_size):
                end_j = min(j + chunk_size, M)
                target_chunk = target_b[j:end_j]  # (chunk_size, D)
                
                # Compute distances from target_chunk to all pred points
                dists = cdist(target_chunk, pred_b, metric='euclidean')  # (chunk_size, N)
                min_dists = np.min(dists, axis=1)  # (chunk_size,)
                min_dists_target_to_pred.append(min_dists)
            
            min_dists_target_to_pred = np.concatenate(min_dists_target_to_pred)
            
            # Chamfer distance for this batch item
            chamfer_b = np.mean(min_dists_pred_to_target) + np.mean(min_dists_target_to_pred)
            total_chamfer += chamfer_b
        
        return total_chamfer / batch_size
    
    def compute_value_function_l2_error(self, pred_points, target_points, value_function, dataset):
        """
        Compute L2 error between predicted values and interpolated true values.
        
        pred_points: (batch_size, N, 4) - predicted points with values in 4th dimension
        target_points: (batch_size, M, 4) - target points with values in 4th dimension  
        value_function: (batch_size, 64, 64, 64) - true value function grid
        dataset: BRTDataset instance for denormalization
        """
        batch_size = pred_points.shape[0]
        total_l2_error = 0.0
        
        for b in range(batch_size):
            # Denormalize predicted points to physical coordinates
            pred_denorm = dataset.denormalize_points(pred_points[b].cpu().numpy())  # (N, 4)
            value_func_3d = value_function[b].cpu().numpy()  # (64, 64, 64)
            
            # COORDINATE SYSTEM CLARIFICATION:
            # Both environment and value function arrays follow the convention:
            # array[y_idx, x_idx, theta_idx] where:
            # - y_idx: 0 to 63 → y-coordinate 0 to 10
            # - x_idx: 0 to 63 → x-coordinate 0 to 10  
            # - theta_idx: 0 to 63 → theta-coordinate -π to π
            
            # When using scipy.interpn, we need to be careful about coordinate order
            # interpn expects: interpn((coord1_grid, coord2_grid, coord3_grid), values, query_points)
            # where values[i, j, k] corresponds to (coord1_grid[i], coord2_grid[j], coord3_grid[k])
            
            # Since value_func_3d[y_idx, x_idx, theta_idx], we need:
            y_coords = np.linspace(0, 10, 64)     # First dimension: y
            x_coords = np.linspace(0, 10, 64)     # Second dimension: x
            theta_coords = np.linspace(-np.pi, np.pi, 64)  # Third dimension: theta
            
            # Extract 3D coordinates from predicted points
            pred_coords = pred_denorm[:, :3]  # (N, 3) - [x, y, theta]
            pred_values = pred_denorm[:, 3]   # (N,) - predicted values
            
            # Clamp coordinates to valid range to avoid extrapolation issues
            pred_coords[:, 0] = np.clip(pred_coords[:, 0], 0, 10)  # x
            pred_coords[:, 1] = np.clip(pred_coords[:, 1], 0, 10)  # y
            pred_coords[:, 2] = np.clip(pred_coords[:, 2], -np.pi, np.pi)  # theta
            
            # Reorder query points to match array indexing: [x, y, theta] → [y, x, theta]
            pred_coords_reordered = pred_coords[:, [1, 0, 2]]  # [y, x, theta]
            
            # Interpolate true values at predicted coordinates
            try:
                true_values = interpn(
                    (y_coords, x_coords, theta_coords),  # Grid coordinates [y, x, theta]
                    value_func_3d,                       # Values array [y_idx, x_idx, theta_idx]
                    pred_coords_reordered,               # Query points [y, x, theta]
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                
                # Compute L2 error
                l2_error = np.mean((pred_values - true_values) ** 2)
                total_l2_error += l2_error
                
            except Exception as e:
                logger.warning(f"Interpolation failed for batch {b}: {e}")
                logger.warning(f"pred_coords_reordered shape: {pred_coords_reordered.shape}")
                logger.warning(f"value_func_3d shape: {value_func_3d.shape}")
                total_l2_error += float('inf')
        
        return total_l2_error / batch_size
    
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
                                     value_functions, model_info, batch_idx, output_dir, dataset):
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
        visualize_detailed_value_function_comparison(
            target_sample, pred_sample, env_sample,
            title=f"{model_name} - {self.current_split.capitalize()} Split - Sample {self.vis_sample_count+1} Detailed Analysis",
            save_path=detailed_save_path,
            dataset=dataset
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
                    f"{base_path}/sample_idx": self.vis_sample_count+1
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
        
        # chamfer_distances = []  # Commented out Chamfer distance
        value_l2_errors = []
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, f"visualizations_{model_type}_{state_dim}D")
        
        # Store samples for visualization
        vis_samples = []
        batch_count = 0
        
        logger.info("Starting evaluation loop...")
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
                        value_l2_error = self.compute_value_function_l2_error(
                            predicted_points, target_points, value_functions, dataset_with_vf
                        )
                        value_l2_errors.append(value_l2_error)
                        
                        # Log batch metrics to wandb
                        if self.wandb_logging:
                            wandb.log({
                                f"{model_type.lower()}/{self.current_split}/batch_value_l2_error": value_l2_error,
                                f"{model_type.lower()}/{self.current_split}/batch_idx": batch_idx
                            })
                
                # Store samples for visualization if needed
                if self.vis_sample_count < self.max_vis_samples:
                    vis_samples.append((predicted_points, target_points, env_batch, value_functions))
                
                batch_count += 1
                
                # Log timing every 10 batches
                if batch_count % 10 == 0:
                    logger.info(f"Completed {batch_count} batches")
        
        # Generate visualizations after evaluation
        if self.save_visualizations and vis_samples:
            logger.info("Generating visualizations...")
            for batch_idx, (predicted_points, target_points, env_batch, value_functions) in enumerate(vis_samples):
                if self.vis_sample_count >= self.max_vis_samples:
                    break
                self.save_evaluation_visualizations(
                    predicted_points, target_points, env_batch, 
                    value_functions, model_info, batch_idx, vis_dir, dataset_with_vf
                )
        
        # Compute final metrics
        results = {
            'model_type': model_type,
            'state_dim': state_dim,
            'config': model_info['config'],
            # 'chamfer_distance_mean': np.mean(chamfer_distances),  # Commented out Chamfer distance
            # 'chamfer_distance_std': np.std(chamfer_distances),   # Commented out Chamfer distance
            # 'num_chamfer_samples': len(chamfer_distances)        # Commented out Chamfer distance
        }
        
        if value_l2_errors:
            results.update({
                'value_l2_error_mean': np.mean(value_l2_errors),
                'value_l2_error_std': np.std(value_l2_errors),
                'num_value_samples': len(value_l2_errors)
            })
            
            # Log final metrics to wandb
            if self.wandb_logging:
                base_path = f"{model_type.lower()}/{self.current_split}"
                wandb.log({
                    f"{base_path}/final_metrics/value_l2_error_mean": results['value_l2_error_mean'],
                    f"{base_path}/final_metrics/value_l2_error_std": results['value_l2_error_std'],
                    f"{base_path}/final_metrics/num_value_samples": results['num_value_samples']
                })
                
                # Create and log summary table
                summary_data = [
                    [model_type, state_dim, self.current_split, 
                     results['value_l2_error_mean'], 
                     results['value_l2_error_std'], 
                     results['num_value_samples']]
                ]
                summary_table = wandb.Table(
                    data=summary_data, 
                    columns=["Model Type", "State Dim", "Split", "Value L2 Error Mean", "Value L2 Error Std", "Samples"]
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
    parser.add_argument('--max_vis_samples', type=int, default=10,
                      help='Maximum number of samples to visualize per model')
    parser.add_argument('--splits', nargs='*', default=['train', 'val', 'test'],
                      help='Dataset splits to evaluate on')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Wandb arguments
    parser.add_argument('--wandb_logging', action='store_true',
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
                logger.info(f"  Value L2 Error: {results['value_l2_error_mean']:.6f} ± {results['value_l2_error_std']:.6f}")
                
                all_results.append(results)
                
                # Create summary for this model and split
                summary = f"Model: {results['model_type']} ({results['state_dim']}D)\n"
                summary += f"Split: {split}\n"
                summary += f"Checkpoint: {checkpoint_name}\n"
                summary += f"Value L2 Error: {results['value_l2_error_mean']:.6f} ± {results['value_l2_error_std']:.6f}\n"
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
    
    # Write detailed results to text file
    results_text_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_text_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Total samples evaluated: {len(dataset)}\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        # Write summary for each model and split
        for summary in results_summary:
            f.write(summary)
        
        # Write detailed statistics
        f.write("="*50 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*50 + "\n\n")
        
        for result in all_results:
            f.write(f"Model: {result['model_type']} ({result['state_dim']}D)\n")
            f.write(f"Split: {result['split']}\n")
            f.write(f"Checkpoint: {result['checkpoint_name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Value Function L2 Error:\n")
            f.write(f"  Mean: {result['value_l2_error_mean']:.8f}\n")
            f.write(f"  Std:  {result['value_l2_error_std']:.8f}\n")
            f.write(f"  Samples: {result['num_value_samples']}\n")
            
            f.write(f"Configuration:\n")
            for key, value in result.get('config', {}).items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")
    
    # Log final comparison results to wandb if enabled
    if args.wandb_logging and all_results:
        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append([
                result['checkpoint_name'],
                result['model_type'],
                result['state_dim'],
                result['split'],
                result['value_l2_error_mean'],
                result['value_l2_error_std'],
                result['num_value_samples']
            ])
        
        comparison_table = wandb.Table(
            data=comparison_data,
            columns=["Checkpoint", "Model Type", "State Dim", "Split", "Value L2 Mean", "Value L2 Std", "Samples"]
        )
        wandb.log({"evaluation_comparison": comparison_table})
        
        # Log result files as artifacts
        if os.path.exists(results_file):
            wandb.save(results_file)
        if os.path.exists(results_text_file):
            wandb.save(results_text_file)
    
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
    print()
    
    for summary in results_summary:
        print(summary)

if __name__ == '__main__':
    main() 