#!/usr/bin/env python3
"""
Simplified model evaluation script for BRT generation models.
Computes mean squared error and standard deviation of squared errors across all points.
Includes wandb logging, artifact downloading, and essential visualizations.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from scipy.interpolate import interpn
import wandb

# Add project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.BRTDataset import BRTDataset
from models.unet_baseline import BRTUNet
from models.diffusion_modules import BRTDiffusionModel
from utils.visualizations import visualize_comparison, visualize_detailed_value_function_comparison

# Default artifacts and their corresponding datasets
DEFAULT_ARTIFACTS = [
    {
        'name': 'diffusion_lucky_moon_21',
        'path': 'malteny-stanford/brt-diffusion/model-checkpoint-lucky-moon-21-epoch-1500:v0',
        'dataset': '~/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
        'checkpoint_file': 'checkpoint_epoch_1500.pt'
    },
    {
        'name': 'unet_deep_wave_6', 
        'path': 'malteny-stanford/brt-unet-baseline/unet-checkpoint-deep-wave-6-epoch-100:v0',
        'dataset': '~/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
        'checkpoint_file': 'checkpoint_epoch_100.pt'
    },
    {
        'name': 'diffusion_comfy_dew_36',
        'path': 'malteny-stanford/brt-diffusion/model-checkpoint-comfy-dew-36-epoch-1300:v0', 
        'dataset': '~/5000_1000_newest',
        'checkpoint_file': 'checkpoint_epoch_1300.pt'
        
    },
    {
        'name': 'diffusion_hearty_sea_37',
        'path': 'malteny-stanford/brt-diffusion/model-checkpoint-hearty-sea-37-epoch-800:v0',
        'dataset': '~/5000_1000_newest',
        'checkpoint_file': 'checkpoint_epoch_800.pt'
    }
]

def download_wandb_artifact(artifact_path, filename=None):
    """Download a wandb artifact and return the path to the checkpoint file."""
    print(f"Downloading wandb artifact: {artifact_path}")
    
    try:
        # Initialize wandb API
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
            import glob
            checkpoint_patterns = ['*.pt', '*.pth', 'checkpoint*.pt', 'model*.pt']
            checkpoint_path = None
            for pattern in checkpoint_patterns:
                matches = glob.glob(os.path.join(artifact_dir, pattern))
                if matches:
                    checkpoint_path = matches[0]
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint file found in artifact")
        
        print(f"Successfully downloaded checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"Failed to download artifact {artifact_path}: {e}")
        raise

class SimpleModelEvaluator:
    """Simplified model evaluator with wandb logging and visualizations"""
    
    def __init__(self, device='cuda', wandb_project="brt-model-evaluation", max_vis_samples=5):
        self.device = device
        self.wandb_project = wandb_project
        self.max_vis_samples = max_vis_samples
        self.vis_sample_count = 0
        
        # Initialize wandb
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            job_type="evaluation",
            tags=["simplified-evaluation"],
            settings=wandb.Settings(start_method="fork")
        )
    
    def __del__(self):
        """Clean up wandb run"""
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            wandb.finish()
    
    def load_model_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint and detect type automatically"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        
        # Detect model type
        model_type = config.get('model_type', 'Unknown')
        
        if 'U-Net' in model_type or 'unet' in model_type.lower():
            model = self._load_unet_model(checkpoint, config)
            state_dim = config.get('max_state_dim', 4)
            model_info = {'type': 'UNet', 'state_dim': state_dim, 'config': config}
        else:
            # Assume diffusion model
            model, state_dim = self._load_diffusion_model(checkpoint, config)
            model_info = {'type': 'Diffusion', 'state_dim': state_dim, 'config': config}
        
        # Log model info to wandb
        wandb.log({
            "model_info/type": model_info['type'],
            "model_info/state_dim": model_info['state_dim'],
            "model_info/total_params": sum(p.numel() for p in model.parameters()),
            "model_info/config": config
        })
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded {model_info['type']} model with {model_info['state_dim']}D output")
        return model, model_info
    
    def _load_unet_model(self, checkpoint, config):
        """Load UNet model"""
        model = BRTUNet(
            env_size=config.get('env_size', 64),
            num_points=config.get('num_points', 4000),
            max_state_dim=config.get('max_state_dim', 4),
            env_encoding_dim=128,
            dropout_rate=config.get('dropout_rate', 0.2),
            weight_decay_strength=config.get('weight_decay_strength', 0.01)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _load_diffusion_model(self, checkpoint, config):
        """Load Diffusion model and infer state_dim from checkpoint"""
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
        return model, state_dim
    
    def _infer_diffusion_state_dim(self, state_dict):
        """Infer state_dim from diffusion model state dict"""
        for key, tensor in state_dict.items():
            if 'denoiser.output_proj.2.weight' in key:
                return tensor.shape[0]
            elif 'output_proj' in key and 'weight' in key and tensor.dim() == 2:
                if tensor.shape[0] in [3, 4]:
                    return tensor.shape[0]
        print("Warning: Could not infer state_dim from checkpoint, assuming 4D")
        return 4
    
    def generate_samples(self, model, model_info, env_batch):
        """Generate samples from model"""
        with torch.no_grad():
            if model_info['type'] == 'UNet':
                predicted = model(env_batch, target_state_dim=model_info['state_dim'])
            elif model_info['type'] == 'Diffusion':
                predicted = self.sample_batch_diffusion(model, env_batch)
                if predicted.shape[-1] > model_info['state_dim']:
                    predicted = predicted[:, :, :model_info['state_dim']]
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")
        return predicted
    
    def sample_batch_diffusion(self, model, env_batch):
        """Efficient batch sampling for diffusion models"""
        batch_size = env_batch.shape[0]
        x_t = torch.randn(batch_size, model.num_points, model.state_dim).to(model.device)
        
        for t in reversed(range(model.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=model.device, dtype=torch.long)
            x_t = model.p_sample(x_t, t_batch, env_batch, guidance_scale=1.5)
        
        return x_t
    
    def compute_all_point_errors(self, pred_points, value_functions, dataset):
        """Compute squared errors for ALL points across all batches"""
        all_squared_errors = []
        batch_size = pred_points.shape[0]
        
        for b in range(batch_size):
            # Denormalize predicted points
            pred_denorm = dataset.denormalize_points(pred_points[b].cpu().numpy())
            value_func_3d = value_functions[b].cpu().numpy()
            
            coords = pred_denorm[:, :3]  # x, y, theta
            pred_values = pred_denorm[:, 3]  # predicted values
            
            # Convert coordinates to grid indices
            x_indices = (coords[:, 0] / 10.0 * 64).clip(0, 63)
            y_indices = (coords[:, 1] / 10.0 * 64).clip(0, 63)
            theta_indices = ((coords[:, 2] + np.pi) / (2 * np.pi) * 64).clip(0, 63)
            grid_coords = np.stack([x_indices, y_indices, theta_indices], axis=1)
            
            # Interpolate true values from value function
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
                
                # Compute squared errors for all points
                errors = pred_values - interpolated_values
                squared_errors = errors ** 2
                all_squared_errors.extend(squared_errors.tolist())
                
            except Exception as e:
                print(f"Warning: Error interpolating batch {b}: {e}")
                continue
        
        return all_squared_errors
    
    def save_visualizations(self, predicted_points, target_points, env_batch, 
                           value_functions, model_info, dataset, l2_error=None, output_dir="./visualizations", artifact_name=None):
        """Save visualization plots and log to wandb"""
        if self.vis_sample_count >= self.max_vis_samples:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Use first sample in batch
        pred_sample = predicted_points[0].cpu().numpy()
        target_sample = target_points[0].cpu().numpy() if target_points is not None else None
        env_sample = env_batch[0].cpu().numpy()
        value_func_sample = value_functions[0].cpu().numpy() if value_functions is not None else None
        
        model_name = f"{model_info['type']}_{model_info['state_dim']}D"
        artifact_title = f" - {artifact_name}" if artifact_name else ""
        
        # Create comparison visualization
        save_path = os.path.join(output_dir, f"{model_name}_{artifact_name}_sample_{self.vis_sample_count+1:03d}_comparison.png")
        visualize_comparison(
            target_sample, pred_sample, env_sample,
            title=f"{model_name}{artifact_title} - Test Split - Sample {self.vis_sample_count+1} Comparison",
            save_path=save_path,
            dataset=dataset
        )
        
        # Create detailed visualization with L2 error info
        detailed_save_path = os.path.join(output_dir, f"{model_name}_{artifact_name}_sample_{self.vis_sample_count+1:03d}_detailed.png")
        detailed_title = f"{model_name}{artifact_title} - Test Split - Sample {self.vis_sample_count+1}"
        if l2_error is not None:
            detailed_title += f" (L2: {l2_error:.6f})"
        
        visualize_detailed_value_function_comparison(
            target_sample, pred_sample, env_sample,
            title=detailed_title,
            save_path=detailed_save_path,
            dataset=dataset
        )
        
        # Log to wandb with artifact name in the key
        wandb.log({
            f"visualizations/{artifact_name}/comparison_{self.vis_sample_count+1}": wandb.Image(save_path),
            f"visualizations/{artifact_name}/detailed_{self.vis_sample_count+1}": wandb.Image(detailed_save_path),
            f"visualizations/{artifact_name}/sample_idx": self.vis_sample_count+1,
            f"visualizations/{artifact_name}/l2_error": l2_error if l2_error is not None else 0.0
        })
        
        self.vis_sample_count += 1
        print(f"Saved visualization {self.vis_sample_count}/{self.max_vis_samples}: {save_path}")
    
    def evaluate_model(self, model, model_info, test_loader, dataset, output_dir, artifact_name=None):
        """Evaluate model and return MSE statistics"""
        print(f"Evaluating {model_info['type']} model with {model_info['state_dim']}D output")
        
        all_squared_errors = []
        vis_samples = []
        
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Load data
            if len(batch_data) == 3:
                target_points, env_batch, value_functions = [d.to(self.device) for d in batch_data]
            else:
                print("Warning: No value functions in dataset - cannot compute MSE")
                return None
            
            # Generate predictions
            predicted_points = self.generate_samples(model, model_info, env_batch)
            
            # Compute squared errors for all points in this batch
            batch_squared_errors = self.compute_all_point_errors(
                predicted_points, value_functions, dataset
            )
            all_squared_errors.extend(batch_squared_errors)
            
            # Store samples for visualization
            if len(vis_samples) < self.max_vis_samples:
                # Compute sample-specific L2 error for first sample in batch
                sample_errors = self.compute_all_point_errors(
                    predicted_points[:1], value_functions[:1], dataset
                )
                sample_l2 = np.mean(sample_errors) if sample_errors else None
                vis_samples.append((predicted_points, target_points, env_batch, value_functions, sample_l2))
            
            # Log batch progress to wandb
            if batch_idx % 10 == 0:
                wandb.log({
                    "evaluation/batch_idx": batch_idx,
                    "evaluation/points_processed": len(all_squared_errors)
                })
        
        # Generate visualizations
        print("Generating visualizations...")
        for sample_data in vis_samples:
            predicted_points, target_points, env_batch, value_functions, sample_l2 = sample_data
            self.save_visualizations(
                predicted_points, target_points, env_batch, 
                value_functions, model_info, dataset, sample_l2, output_dir,
                artifact_name=artifact_name
            )
        
        # Compute final statistics
        if not all_squared_errors:
            print("No valid squared errors computed")
            return None
        
        all_squared_errors = np.array(all_squared_errors)
        mean_squared_error = np.mean(all_squared_errors)
        std_squared_error = np.std(all_squared_errors)
        rmse = np.sqrt(mean_squared_error)
        
        results = {
            'model_type': model_info['type'],
            'state_dim': model_info['state_dim'],
            'mean_squared_error': float(mean_squared_error),
            'std_squared_error': float(std_squared_error),
            'total_points': len(all_squared_errors),
            'rmse': float(rmse),
            'config': model_info['config'],
            'artifact_name': artifact_name
        }
        
        # Log final results to wandb
        wandb.log({
            f"results/{artifact_name}/mean_squared_error": mean_squared_error,
            f"results/{artifact_name}/std_squared_error": std_squared_error,
            f"results/{artifact_name}/rmse": rmse,
            f"results/{artifact_name}/total_points": len(all_squared_errors),
            f"results/{artifact_name}/model_type": model_info['type'],
            f"results/{artifact_name}/state_dim": model_info['state_dim']
        })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate BRT models with wandb logging')
    parser.add_argument('--use_default_artifacts', action='store_true', default=True,
                      help='Use default wandb artifacts')
    parser.add_argument('--checkpoint', type=str, 
                      help='Path to specific checkpoint (overrides default artifacts)')
    parser.add_argument('--dataset_dir', type=str,
                      help='Dataset directory (auto-detected from artifact if not specified)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    parser.add_argument('--max_vis_samples', type=int, default=5,
                      help='Maximum number of visualization samples')
    parser.add_argument('--wandb_project', type=str, default='brt-model-evaluation',
                      help='Wandb project name')
    parser.add_argument('--output_dir', type=str, default='evaluation_output',
                      help='Output directory for results and visualizations')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup evaluator
    evaluator = SimpleModelEvaluator(
        device=torch.device(args.device),
        wandb_project=args.wandb_project,
        max_vis_samples=args.max_vis_samples
    )
    
    all_results = []
    
    # Determine which models to evaluate
    if args.checkpoint:
        # Single checkpoint specified
        models_to_eval = [{'name': 'custom', 'path': args.checkpoint, 'dataset': args.dataset_dir}]
    elif args.use_default_artifacts:
        # Use default artifacts
        models_to_eval = DEFAULT_ARTIFACTS
    else:
        print("No checkpoints specified. Use --checkpoint or --use_default_artifacts")
        return
    
    # Evaluate each model
    for model_config in models_to_eval:
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_config['name']}")
            print(f"{'='*60}")
            
            # Download checkpoint if it's a wandb artifact
            if model_config['path'].count('/') >= 2 and ':' in model_config['path']:
                print(f"Downloading artifact: {model_config['path']}")
                checkpoint_path = download_wandb_artifact(
                    model_config['path'], 
                    model_config.get('checkpoint_file')
                )
            else:
                checkpoint_path = model_config['path']
                
            # Determine dataset directory
            if args.dataset_dir:
                dataset_dir = os.path.expanduser(args.dataset_dir)
            else:
                dataset_dir = os.path.expanduser(model_config['dataset'])
            
            print(f"Using dataset: {dataset_dir}")
            
            # Load model
            model, model_info = evaluator.load_model_from_checkpoint(checkpoint_path)
            
            # Load dataset (test split only, with value functions)
            print("Loading test dataset with value functions...")
            dataset = BRTDataset(dataset_dir, split='test', return_value_function=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=args.num_workers)
            
            print(f"Test dataset loaded: {len(dataset)} samples")
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_config['name'])
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Reset visualization counter for each model
            evaluator.vis_sample_count = 0
            
            # Evaluate
            results = evaluator.evaluate_model(model, model_info, data_loader, dataset, model_output_dir, artifact_name=model_config['name'])
            
            if results:
                results['model_name'] = model_config['name']
                results['checkpoint_path'] = checkpoint_path
                results['dataset_path'] = dataset_dir
                results['timestamp'] = timestamp
                
                # Print results
                print(f"\nResults for {model_config['name']}:")
                print(f"  Model Type: {results['model_type']}")
                print(f"  State Dimension: {results['state_dim']}")
                print(f"  Total Points: {results['total_points']:,}")
                print(f"  Mean Squared Error: {results['mean_squared_error']:.8f}")
                print(f"  Std of Squared Errors: {results['std_squared_error']:.8f}")
                print(f"  RMSE: {results['rmse']:.8f}")
                
                all_results.append(results)
                
        except Exception as e:
            print(f"Failed to evaluate {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table for wandb
    if all_results:
        comparison_data = []
        for result in all_results:
            comparison_data.append([
                result['model_name'],
                result['model_type'],
                result['state_dim'],
                result['mean_squared_error'],
                result['std_squared_error'],
                result['rmse'],
                result['total_points']
            ])
        
        comparison_table = wandb.Table(
            data=comparison_data,
            columns=["Model Name", "Type", "State Dim", "MSE", "Std Squared Error", "RMSE", "Total Points"]
        )
        wandb.log({"evaluation_comparison": comparison_table})
        
        # Save results as wandb artifact
        artifact = wandb.Artifact(
            name=f"evaluation_results_{timestamp}",
            type="evaluation_results",
            description=f"Model evaluation results for {len(all_results)} models"
        )
        artifact.add_file(results_file, name="evaluation_results.json")
        artifact.add_dir(output_dir, name="evaluation_output")
        wandb.log_artifact(artifact)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Evaluation completed at: {timestamp}")
    print(f"Results saved to: {output_dir}")
    print(f"Wandb project: {args.wandb_project}")
    
    for result in all_results:
        print(f"\n{result['model_name']} ({result['model_type']}, {result['state_dim']}D):")
        print(f"  MSE: {result['mean_squared_error']:.8f} ± {result['std_squared_error']:.8f}")
        print(f"  RMSE: {result['rmse']:.8f}")
        print(f"  Points: {result['total_points']:,}")

if __name__ == '__main__':
    main()