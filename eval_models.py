#!/usr/bin/env python3
"""
Comprehensive model evaluation script for BRT generation models.
Supports UNet and Diffusion models with automatic 3D/4D detection.
Evaluates using Chamfer distance and value function L2 error.
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

class ModelEvaluator:
    """Handles evaluation of different model types"""
    
    def __init__(self, device='cuda', save_visualizations=True):
        self.device = device
        self.save_visualizations = save_visualizations
        self.max_vis_samples = 10  # Default value
        self.vis_sample_count = 0  # Track number of visualizations created
        
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
            
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded {model_info['type']} model with {model_info['state_dim']}D output")
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
            return
            
        if model_info['type'] != 'Diffusion':
            logger.info(f"Skipping GIF generation for {model_info['type']} model")
            return
            
        gif_dir = os.path.join(output_dir, 'denoising_gifs')
        os.makedirs(gif_dir, exist_ok=True)
        
        try:
            logger.info(f"Generating denoising GIF for sample {sample_idx}")
            
            if model_info['state_dim'] == 4:
                # Use 4D GIF generation
                generate_gif_4d(
                    model=model,
                    dataset=dataset,
                    sample_idx=sample_idx,
                    num_frames=30,  # Fewer frames for faster generation
                    save_dir=gif_dir
                )
            else:
                # Use 3D GIF generation  
                generate_gif_3d(
                    model=model,
                    dataset=dataset,
                    sample_idx=sample_idx,
                    num_frames=30,
                    save_dir=gif_dir
                )
                
            logger.info(f"Successfully generated denoising GIF for sample {sample_idx}")
            
        except Exception as e:
            logger.error(f"Failed to generate denoising GIF: {e}")

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
        
        # Visualize first sample in batch
        pred_sample = predicted_points[0].cpu().numpy()
        target_sample = target_points[0].cpu().numpy()
        env_sample = env_batch[0].cpu().numpy()
        
        # Create regular comparison visualization with proper denormalization
        save_path = os.path.join(output_dir, f"{model_name}_sample_{self.vis_sample_count+1:03d}_comparison.png")
        visualize_comparison(
            target_sample, pred_sample, env_sample,
            title=f"{model_name} - Sample {self.vis_sample_count+1} Comparison",
            save_path=save_path,
            dataset=dataset  # Pass dataset for proper coordinate denormalization
        )
        
        # Create detailed comparison visualization with theta slices
        detailed_save_path = os.path.join(output_dir, f"{model_name}_sample_{self.vis_sample_count+1:03d}_detailed.png")
        visualize_detailed_value_function_comparison(
            target_sample, pred_sample, env_sample,
            title=f"{model_name} - Sample {self.vis_sample_count+1} Detailed Analysis",
            save_path=detailed_save_path,
            dataset=dataset
        )
        
        # Generate denoising GIF for diffusion models (only for first few samples)
        if self.vis_sample_count < 3 and model_info['type'] == 'Diffusion':
            # Use a deterministic sample from the dataset for GIF generation
            sample_idx = (self.vis_sample_count * 137) % min(100, len(dataset))  # Deterministic but varied
            self.generate_denoising_gif_for_evaluation(
                model_info.get('model_instance'),  # We'll need to pass the model instance
                model_info, 
                dataset, 
                sample_idx, 
                output_dir
            )
        
        # Increment visualization counter
        self.vis_sample_count += 1
        
        logger.info(f"Saved visualizations ({self.vis_sample_count}/{self.max_vis_samples}): {save_path} and {detailed_save_path}")
    
    def evaluate_model(self, model, model_info, test_loader, dataset_with_vf, output_dir):
        """Evaluate model on test set"""
        model_type = model_info['type']
        state_dim = model_info['state_dim']
        
        # Store model instance for GIF generation
        model_info['model_instance'] = model
        
        logger.info(f"Evaluating {model_type} model with {state_dim}D output")
        
        chamfer_distances = []
        value_l2_errors = []
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, f"visualizations_{model_type}_{state_dim}D")
        
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            if len(batch_data) == 3:
                target_points, env_batch, value_functions = batch_data
            else:
                target_points, env_batch = batch_data
                value_functions = None
            
            env_batch = env_batch.to(self.device)
            target_points = target_points.to(self.device)
            
            # Generate predictions
            if state_dim == 3:
                # 3D model: generate 3D points, evaluate with Chamfer distance only
                predicted_points = self.generate_samples(model, model_info, env_batch, target_state_dim=3)
                
                # Compute Chamfer distance on 3D coordinates
                chamfer_dist = self.compute_chamfer_distance(
                    predicted_points, target_points[:, :, :3]
                )
                chamfer_distances.append(chamfer_dist)
                
                # Save visualization if we haven't reached the limit
                if self.vis_sample_count < self.max_vis_samples:
                    self.save_evaluation_visualizations(
                        predicted_points, target_points, env_batch, 
                        value_functions, model_info, batch_idx, vis_dir, dataset_with_vf
                    )
                
            elif state_dim == 4:
                # 4D model: generate 4D points, evaluate with both metrics
                predicted_points = self.generate_samples(model, model_info, env_batch, target_state_dim=4)
                
                # Compute Chamfer distance on 3D coordinates
                chamfer_dist = self.compute_chamfer_distance(
                    predicted_points[:, :, :3], target_points[:, :, :3]
                )
                chamfer_distances.append(chamfer_dist)
                
                # Compute value function L2 error if value functions available
                if value_functions is not None:
                    value_l2_error = self.compute_value_function_l2_error(
                        predicted_points, target_points, value_functions, dataset_with_vf
                    )
                    value_l2_errors.append(value_l2_error)
                
                # Save visualization if we haven't reached the limit
                if self.vis_sample_count < self.max_vis_samples:
                    self.save_evaluation_visualizations(
                        predicted_points, target_points, env_batch, 
                        value_functions, model_info, batch_idx, vis_dir, dataset_with_vf
                    )
        
        # Compute final metrics
        results = {
            'model_type': model_type,
            'state_dim': state_dim,
            'config': model_info['config'],
            'chamfer_distance_mean': np.mean(chamfer_distances),
            'chamfer_distance_std': np.std(chamfer_distances),
            'num_chamfer_samples': len(chamfer_distances)
        }
        
        if value_l2_errors:
            results.update({
                'value_l2_error_mean': np.mean(value_l2_errors),
                'value_l2_error_std': np.std(value_l2_errors),
                'num_value_samples': len(value_l2_errors)
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate BRT generation models')
    parser.add_argument('--dataset_dir', type=str, 
                      default='1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                      help='Path to dataset directory')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                      help='List of checkpoint files to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
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
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                      help='Dataset split to evaluate on')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_base_dir, f"eval_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = torch.device(args.device)
    
    evaluator = ModelEvaluator(device=device, save_visualizations=not args.no_visualizations)
    
    # Load dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = BRTDataset(args.dataset_dir, split=args.split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Load dataset with value functions for interpolation
    dataset_with_vf = BRTDataset(args.dataset_dir, split=args.split, return_value_function=True)
    data_loader_with_vf = DataLoader(dataset_with_vf, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    
    logger.info(f"{args.split.capitalize()} dataset loaded: {len(dataset)} samples")
    
    # Write evaluation info to file
    info_file = os.path.join(args.output_dir, 'evaluation_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Evaluation Information\n")
        f.write(f"=====================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Max visualization samples: {args.max_vis_samples}\n")
        f.write(f"Checkpoints to evaluate: {len(args.checkpoints)}\n")
        for i, checkpoint in enumerate(args.checkpoints):
            f.write(f"  {i+1}. {checkpoint}\n")
        f.write(f"\n")
    
    # Evaluate each checkpoint
    all_results = []
    results_summary = []
    
    for checkpoint_idx, checkpoint_path in enumerate(args.checkpoints):
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            continue
            
        try:
            logger.info(f"Evaluating checkpoint {checkpoint_idx+1}/{len(args.checkpoints)}: {checkpoint_path}")
            
            # Load model
            model, model_info = evaluator.load_model_from_checkpoint(checkpoint_path)
            
            # Choose appropriate data loader
            if model_info['state_dim'] == 4:
                loader = data_loader_with_vf
                dataset_for_denorm = dataset_with_vf
            else:
                loader = data_loader
                dataset_for_denorm = dataset
            
            # Update evaluator to limit visualizations
            evaluator.max_vis_samples = args.max_vis_samples
            evaluator.vis_sample_count = 0
            
            # Evaluate
            results = evaluator.evaluate_model(model, model_info, loader, dataset_for_denorm, args.output_dir)
            results['checkpoint_path'] = checkpoint_path
            results['checkpoint_basename'] = os.path.basename(checkpoint_path)
            results['evaluation_timestamp'] = timestamp
            
            # Log results
            logger.info(f"Results for {checkpoint_path}:")
            logger.info(f"  Model Type: {results['model_type']}")
            logger.info(f"  State Dim: {results['state_dim']}")
            logger.info(f"  Chamfer Distance: {results['chamfer_distance_mean']:.6f} ± {results['chamfer_distance_std']:.6f}")
            if 'value_l2_error_mean' in results:
                logger.info(f"  Value L2 Error: {results['value_l2_error_mean']:.6f} ± {results['value_l2_error_std']:.6f}")
            
            all_results.append(results)
            
            # Create summary for this model
            summary = f"Model: {results['model_type']} ({results['state_dim']}D)\n"
            summary += f"Checkpoint: {results['checkpoint_basename']}\n"
            summary += f"Chamfer Distance: {results['chamfer_distance_mean']:.6f} ± {results['chamfer_distance_std']:.6f}\n"
            if 'value_l2_error_mean' in results:
                summary += f"Value L2 Error: {results['value_l2_error_mean']:.6f} ± {results['value_l2_error_std']:.6f}\n"
            summary += f"Samples evaluated: {results['num_chamfer_samples']}\n"
            summary += "\n"
            results_summary.append(summary)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
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
        f.write(f"Dataset: {args.dataset_dir} ({args.split} split)\n")
        f.write(f"Total samples evaluated: {len(dataset)}\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        # Write summary for each model
        for summary in results_summary:
            f.write(summary)
        
        # Write detailed statistics
        f.write("="*50 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*50 + "\n\n")
        
        for result in all_results:
            f.write(f"Model: {result['model_type']} ({result['state_dim']}D)\n")
            f.write(f"Checkpoint: {result['checkpoint_basename']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Chamfer Distance:\n")
            f.write(f"  Mean: {result['chamfer_distance_mean']:.8f}\n")
            f.write(f"  Std:  {result['chamfer_distance_std']:.8f}\n")
            f.write(f"  Samples: {result['num_chamfer_samples']}\n")
            
            if 'value_l2_error_mean' in result:
                f.write(f"Value Function L2 Error:\n")
                f.write(f"  Mean: {result['value_l2_error_mean']:.8f}\n")
                f.write(f"  Std:  {result['value_l2_error_std']:.8f}\n")
                f.write(f"  Samples: {result['num_value_samples']}\n")
            
            f.write(f"Configuration:\n")
            for key, value in result.get('config', {}).items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")
    logger.info(f"  JSON results: {results_file}")
    logger.info(f"  Text results: {results_text_file}")
    logger.info(f"  Info file: {info_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluation completed at: {timestamp}")
    print(f"Dataset: {args.dataset_dir} ({args.split} split)")
    print(f"Total samples: {len(dataset)}")
    print(f"Results saved to: {args.output_dir}")
    print()
    
    for summary in results_summary:
        print(summary)

if __name__ == '__main__':
    main() 