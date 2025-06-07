import os
import numpy as np
import argparse
from tqdm import tqdm
from scipy.interpolate import interpn
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# add to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.BRTDataset import BRTDataset
from utils.visualizations import visualize_point_cloud


def compute_gradient_probability(value_function):
    """
    Compute probability distribution based on the rate of change in the value function.
    Args:
        value_function: 4D array (time_steps, x, y, theta)
    Returns:
        probability: 3D array (x, y, theta) with sampling probabilities
    """
    # Get the last timestep (converged value function)
    last_timestep = value_function
    
    # Compute gradients in all three dimensions
    dx = np.gradient(last_timestep, axis=0)  # x gradient
    dy = np.gradient(last_timestep, axis=1)  # y gradient
    dtheta = np.gradient(last_timestep, axis=2)  # theta gradient
    
    # Compute gradient magnitude at each point
    gradient_magnitude = np.sqrt(dx**2 + dy**2 + dtheta**2)
    
    # Normalize to create probability distribution
    probability = gradient_magnitude / np.sum(gradient_magnitude)
    
    return probability



def generate_point_cloud(value_function: np.ndarray, num_points: int = 6000) -> np.ndarray:
    """
    Generate a point cloud by sampling from the value function space according to gradient probabilities.
    
    Args:
        value_function: 4D array (time_steps, x, y, theta) containing the value function
        num_points: Number of points to sample (default: 6000)
        
    Returns:
        np.ndarray: Array of shape (num_points, 4) containing (x, y, theta, value) in physical space
    """
    # Get probability distribution from gradient
    prob_dist = compute_gradient_probability(value_function)
    
    # Flatten the probability distribution and get corresponding indices
    flat_prob = prob_dist.flatten()
    indices = np.arange(len(flat_prob))
    
    # Sample points according to probability distribution
    sampled_indices = np.random.choice(indices, size=num_points, p=flat_prob)
    
    # Convert flat indices back to 3D coordinates
    # For a 64x64x64 array, the flattened index is: idx = x * (64*64) + y * 64 + theta
    x_coords = sampled_indices // (64 * 64)
    y_coords = (sampled_indices % (64 * 64)) // 64
    theta_coords = sampled_indices % 64
    
    # Get value function at sampled points
    values = value_function[x_coords, y_coords, theta_coords]
    
    # Transform to physical space
    x_physical = x_coords * (10.0 / 64)  # Scale to [0, 10]
    y_physical = y_coords * (10.0 / 64)  # Scale to [0, 10]
    theta_physical = theta_coords * (2 * np.pi / 64) - np.pi  # Scale to [-pi, pi]
    
    # Combine into point cloud
    point_cloud = np.column_stack((x_physical, y_physical, theta_physical, values))
    
    return point_cloud


def main():
    parser = argparse.ArgumentParser(description='Generate point cloud dataset from value functions')
    parser.add_argument('--input', type=str, default='/Users/malte/AA-276-Diffusion-BRT/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                        help='Input dataset path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output dataset path')
    parser.add_argument('--num_points', type=int, default=6000,
                        help='Number of points to sample per point cloud')
    parser.add_argument('--num_clouds', type=int, default=4,
                        help='Number of point clouds per environment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get all sample directories
    sample_dirs = sorted([d for d in os.listdir(args.input) if d.startswith('sample_')])
    
    # Process each sample
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        # Check if required files exist
        env_grid_path = os.path.join(args.input, sample_dir, 'environment_grid.npy')
        value_func_path = os.path.join(args.input, sample_dir, 'value_function.npy')
        
        if not os.path.exists(env_grid_path):
            print(f"Skipping {sample_dir}: environment_grid.npy not found")
            continue
            
        if not os.path.exists(value_func_path):
            print(f"Skipping {sample_dir}: value_function.npy not found")
            continue
        
        # Create output sample directory
        out_sample_dir = os.path.join(args.output, sample_dir)
        os.makedirs(out_sample_dir, exist_ok=True)
        
        try:
            # Load and save environment grid
            env_grid = np.load(env_grid_path)
            out_env_grid_path = os.path.join(out_sample_dir, 'environment_grid.npy')
            np.save(out_env_grid_path, env_grid)
            
            # Load and save value function
            value_function = np.load(value_func_path)
            out_value_func_path = os.path.join(out_sample_dir, 'value_function.npy')
            np.save(out_value_func_path, value_function)
            
            # Generate point clouds
            for i in range(args.num_clouds):
                # Set seed for this point cloud
                np.random.seed(args.seed + i)
                
                # Generate point cloud
                point_cloud = generate_point_cloud(value_function[-1], num_points=args.num_points)
                
                # Save point cloud
                out_pc_path = os.path.join(out_sample_dir, f'point_cloud_{i}.npy')
                np.save(out_pc_path, point_cloud)
                
        except Exception as e:
            print(f"Error processing {sample_dir}: {e}")
            continue

if __name__ == "__main__":
    main()

