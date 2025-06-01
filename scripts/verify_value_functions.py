#!/usr/bin/env python3
"""
Script to verify that value function copying was successful.
Visualizes value function (zero level set) and point clouds side by side for verification.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.BRTDataset import BRTDataset
from utils.visualizations import create_dual_colormap

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. Will use simplified 3D visualization.")

def extract_zero_level_set(value_function):
    """
    Extract the zero level set (isosurface) from the 3D value function.
    
    Value function has shape (64, 64, 64) representing:
    - First dimension: x-axis (0 to 10)
    - Second dimension: y-axis (0 to 10) 
    - Third dimension: theta-axis (-π to π)
    """
    if not HAS_SKIMAGE:
        return None, None
    
    try:
        # Use marching cubes to extract the zero level set
        verts, faces, normals, values = measure.marching_cubes(
            value_function, level=0.0
        )
        
        # Convert grid indices to physical coordinates
        # verts contains indices, we need to scale to physical coordinates
        x_coords = verts[:, 0] * (10.0 / 64.0)  # x: grid index → [0, 10]
        y_coords = verts[:, 1] * (10.0 / 64.0)  # y: grid index → [0, 10]
        theta_coords = verts[:, 2] * (2*np.pi / 64.0) - np.pi  # theta: grid index → [-π, π]
        
        # Reconstruct vertices with physical coordinates
        verts_physical = np.column_stack([x_coords, y_coords, theta_coords])
        
        return verts_physical, faces
    except Exception as e:
        print(f"Warning: Could not extract zero level set: {e}")
        return None, None

def visualize_value_function_3d(value_function, ax, title="Value Function Zero Level Set"):
    """
    Visualize the 3D zero level set of the value function.
    """
    if len(value_function.shape) == 4:
        value_func_3d = value_function[-1]  # Last time slice
    else:
        value_func_3d = value_function
    
    verts, faces = extract_zero_level_set(value_func_3d)
    
    if verts is not None and faces is not None and len(verts) > 0:
        # Create 3D mesh
        mesh = [[verts[j] for j in faces[i]] for i in range(len(faces))]
        poly3d = Poly3DCollection(mesh, alpha=0.7, facecolor='red', edgecolor='darkred')
        ax.add_collection3d(poly3d)
    else:
        # Fallback: show sample points where value function is close to zero
        print("Using fallback visualization...")
        
        # Create coordinate grids in physical space
        x_vals = np.linspace(0, 10, value_func_3d.shape[0])
        y_vals = np.linspace(0, 10, value_func_3d.shape[1])
        theta_vals = np.linspace(-np.pi, np.pi, value_func_3d.shape[2])
        
        x_grid, y_grid, theta_grid = np.meshgrid(x_vals, y_vals, theta_vals, indexing='ij')
        
        # Find points close to zero
        zero_mask = np.abs(value_func_3d) < 0.1
        if np.any(zero_mask):
            zero_points_x = x_grid[zero_mask]
            zero_points_y = y_grid[zero_mask]
            zero_points_theta = theta_grid[zero_mask]
            
            # Subsample for visualization
            n_points = min(1000, len(zero_points_x))
            indices = np.random.choice(len(zero_points_x), n_points, replace=False)
            
            ax.scatter(zero_points_x[indices], zero_points_y[indices], zero_points_theta[indices],
                      c='red', s=1, alpha=0.6, label='Zero Level Set')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('θ (rad)')

def visualize_value_function_vs_pointcloud(value_function, point_cloud, env_grid, 
                                         sample_id, save_dir=None, dataset=None):
    """
    Visualize value function zero level set and corresponding point cloud side by side.
    """
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Environment grid
    ax1 = fig.add_subplot(231)
    # Important: imshow expects array[row, col] but displays as (col, row) with extent
    # env_grid[i,j] corresponds to physical position (x=j*dx, y=i*dy)
    # Therefore we need to transpose the grid so it displays correctly
    ax1.imshow(env_grid.T, cmap='binary', vmin=0, vmax=1, 
              extent=[0, 10, 0, 10], origin='lower')
    ax1.set_title('Environment Grid')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # 2. Value function slice (middle theta)
    ax2 = fig.add_subplot(232)
    if len(value_function.shape) == 4:
        value_func_3d = value_function[-1]  # Last time slice
    else:
        value_func_3d = value_function
    
    # Show middle theta slice (theta ≈ 0)
    middle_theta_idx = value_func_3d.shape[2] // 2
    middle_slice = value_func_3d[:, :, middle_theta_idx]
    
    cmap = create_dual_colormap()
    vmin, vmax = value_func_3d.min(), value_func_3d.max()
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Display slice: array[i,j] → position (x=j*dx, y=i*dy)
    # Therefore we need to transpose the slice so it displays correctly
    im = ax2.imshow(middle_slice.T, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, 10, 0, 10], origin='lower')
    ax2.set_title(f'Value Function (θ≈0 slice)')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im, ax=ax2, label='Value Function')
    
    # 3. 3D Zero Level Set of Value Function
    ax3 = fig.add_subplot(233, projection='3d')
    visualize_value_function_3d(value_function, ax3, "Value Function Zero Level Set")
    
    # 4. Point cloud in 3D
    ax4 = fig.add_subplot(234, projection='3d')
    
    # Denormalize points if dataset is provided
    if dataset is not None:
        points_denorm = dataset.denormalize_points(point_cloud)
    else:
        points_denorm = point_cloud
    
    x = points_denorm[:, 0]
    y = points_denorm[:, 1]
    theta = points_denorm[:, 2]
    
    if points_denorm.shape[1] > 3:
        w = points_denorm[:, 3]
        # Use same color scaling as value function
        scatter = ax4.scatter(x, y, theta, c=w, cmap=cmap, s=1, alpha=0.6, vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, ax=ax4, label='Value Function', shrink=0.8)
    else:
        ax4.scatter(x, y, theta, s=1, alpha=0.6)
    
    ax4.set_title('Point Cloud (4D)')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('θ (rad)')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_zlim(-np.pi, np.pi)
    
    # 5. Combined view: Zero level set + point cloud colored by value
    ax5 = fig.add_subplot(235, projection='3d')
    
    # Add zero level set first
    visualize_value_function_3d(value_function, ax5, "")
    
    # Add point cloud (only points close to zero level set for clarity)
    if points_denorm.shape[1] > 3:
        # Only show points close to zero level set for clarity
        zero_mask = np.abs(points_denorm[:, 3]) < 0.5
        if np.any(zero_mask):
            scatter = ax5.scatter(x[zero_mask], y[zero_mask], theta[zero_mask], 
                                c=w[zero_mask], cmap=cmap, s=2, alpha=0.8, 
                                vmin=vmin, vmax=vmax)
    
    ax5.set_title('Zero Level Set + Near-Zero Points')
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_zlabel('θ (rad)')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.set_zlim(-np.pi, np.pi)
    
    # 6. Statistics comparison
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    # Calculate statistics
    vf_stats = f"""Value Function Stats:
    Shape: {value_function.shape}
    Min: {value_func_3d.min():.4f}
    Max: {value_func_3d.max():.4f}
    Mean: {value_func_3d.mean():.4f}
    Negative %: {(value_func_3d < 0).sum() / value_func_3d.size * 100:.1f}%
    Zero crossings: {np.sum(np.abs(value_func_3d) < 0.01)}
    """
    
    if points_denorm.shape[1] > 3:
        pc_stats = f"""Point Cloud Stats:
        Shape: {point_cloud.shape}
        Value Min: {points_denorm[:, 3].min():.4f}
        Value Max: {points_denorm[:, 3].max():.4f}
        Value Mean: {points_denorm[:, 3].mean():.4f}
        Negative %: {(points_denorm[:, 3] < 0).sum() / len(points_denorm) * 100:.1f}%
        Near-zero points: {np.sum(np.abs(points_denorm[:, 3]) < 0.1)}
        """
    else:
        pc_stats = f"""Point Cloud Stats:
        Shape: {point_cloud.shape}
        No value dimension found
        """
    
    ax6.text(0.05, 0.8, vf_stats, transform=ax6.transAxes, fontsize=9, 
             verticalalignment='top', family='monospace')
    ax6.text(0.05, 0.35, pc_stats, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', family='monospace')
    
    fig.suptitle(f'Sample {sample_id}: Zero Level Set vs Point Cloud Verification', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'verification_sample_{sample_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Verify value function copying was successful')
    parser.add_argument('--dataset_dir', type=str, 
                      default='1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                      help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='verification_plots',
                      help='Directory to save plots')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset with value functions
    print("Loading dataset with value functions...")
    try:
        dataset = BRTDataset(args.dataset_dir, split="train", return_value_function=True)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Select random samples
    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    print(f"Selected samples: {sample_indices}")
    
    # Process each sample
    for i, sample_idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i+1}/{len(sample_indices)} (index {sample_idx})...")
        
        try:
            # Load sample
            point_cloud, env_grid, value_function = dataset[sample_idx]
            
            if value_function is None:
                print(f"Warning: No value function found for sample {sample_idx}")
                continue
            
            # Convert to numpy arrays
            point_cloud = point_cloud.numpy()
            env_grid = env_grid.numpy()
            value_function = value_function.numpy()
            
            print(f"  Point cloud shape: {point_cloud.shape}")
            print(f"  Environment shape: {env_grid.shape}")
            print(f"  Value function shape: {value_function.shape}")
            
            # Create visualization
            sample_info = dataset.point_cloud_files[sample_idx]
            sample_name = f"{sample_info[0]}_{sample_info[1].replace('.npy', '')}"
            
            visualize_value_function_vs_pointcloud(
                value_function, point_cloud, env_grid, 
                sample_name, args.save_dir, dataset
            )
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"\nVerification complete! Check {args.save_dir} for saved plots.")
    
    # Summary statistics
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    # Count samples with/without value functions
    samples_with_vf = 0
    samples_without_vf = 0
    
    print("Checking all samples for value function availability...")
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        try:
            _, _, vf = dataset[i]
            if vf is not None:
                samples_with_vf += 1
            else:
                samples_without_vf += 1
        except:
            samples_without_vf += 1
    
    print(f"Samples with value functions: {samples_with_vf}")
    print(f"Samples without value functions: {samples_without_vf}")
    print(f"Success rate: {samples_with_vf / (samples_with_vf + samples_without_vf) * 100:.1f}%")

if __name__ == '__main__':
    main() 