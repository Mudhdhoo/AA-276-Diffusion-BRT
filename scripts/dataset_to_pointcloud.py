import os
import numpy as np
import argparse
from tqdm import tqdm
import shutil
from scipy.interpolate import interpn
import csv

def generate_point_cloud(value_function, n_points_inside=1000, n_points_outside=3000, seed=None):
    """
    Generate a point cloud from the value function with specified number of points inside and outside the BRT.
    Args:
        value_function: 4D array (time_steps, x, y, z)
        n_points_inside: number of points to sample where value <= 0
        n_points_outside: number of points to sample where value > 0
        seed: random seed for reproducibility
    Returns:
        points: Nx4 array of points in physical space, where the 4th dimension is the value function value
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get the last timestep
    last_timestep = value_function[-1]
    
    # Create grid points for interpolation
    x = np.arange(last_timestep.shape[0])
    y = np.arange(last_timestep.shape[1])
    z = np.arange(last_timestep.shape[2])
    points = (x, y, z)
    
    def sample_points(n_points, target_condition):
        points_found = 0
        valid_points = np.zeros((n_points, 4))  # Now storing 4D points
        batch_size = n_points * 10  # Start with 10x the needed points
        
        while points_found < n_points:
            # Generate random points in the volume
            x_rand = np.random.uniform(0, last_timestep.shape[0]-1, batch_size)
            y_rand = np.random.uniform(0, last_timestep.shape[1]-1, batch_size)
            z_rand = np.random.uniform(0, last_timestep.shape[2]-1, batch_size)
            
            # Stack coordinates for interpolation
            xi = np.stack([x_rand, y_rand, z_rand], axis=1)
            
            # Interpolate values at random points
            values = interpn(points, last_timestep, xi, method='linear', bounds_error=True)
            
            # Keep only points that satisfy the condition
            valid_mask = target_condition(values)
            new_valid_points = np.column_stack([xi[valid_mask], values[valid_mask]])  # Include values in output
            
            # Add new valid points to our collection
            remaining = n_points - points_found
            if len(new_valid_points) > 0:
                if len(new_valid_points) >= remaining:
                    valid_points[points_found:] = new_valid_points[:remaining]
                    points_found = n_points
                else:
                    valid_points[points_found:points_found + len(new_valid_points)] = new_valid_points
                    points_found += len(new_valid_points)
            
            # If we're not finding enough points, increase the batch size
            if points_found < n_points:
                batch_size *= 2
        
        return valid_points
    
    # Sample points inside BRT (value <= 0)
    inside_points = sample_points(n_points_inside, lambda v: v < 0)
    
    # Sample points outside BRT (value > 0)
    outside_points = sample_points(n_points_outside, lambda v: v >= 0)
    
    # Combine both sets of points
    all_points = np.vstack([inside_points, outside_points])
    
    # Scale coordinates to physical space (but not the value function)
    all_points[:, 0] = all_points[:, 0] / 64 * 10  # x: [0,64] -> [0,10]
    all_points[:, 1] = all_points[:, 1] / 64 * 10  # y: [0,64] -> [0,10]
    all_points[:, 2] = all_points[:, 2] / 64 * 2 * np.pi - np.pi  # z: [0,64] -> [-π,π]
    # all_points[:, 3] remains unchanged as it's the value function value
    
    return all_points

def process_dataset(dataset_dir, output_dir, n_points_inside=1000, n_points_outside=3000, clouds_per_sample=1):
    """
    Process all samples in the dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sample_')]
    results = []
    missing_count = 0
    missing_samples = []
    
    # Set global seed
    np.random.seed(42)
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_output_dir = os.path.join(output_dir, sample_dir)
        os.makedirs(sample_output_dir, exist_ok=True)
        value_function_path = os.path.join(dataset_dir, sample_dir, 'value_function.npy')
        if not os.path.exists(value_function_path):
            missing_count += 1
            missing_samples.append(sample_dir)
            continue
        value_function = np.load(value_function_path)
        
        # Generate multiple point clouds with different seeds
        for i in range(clouds_per_sample):
            # Use global seed + i to get different but deterministic sampling
            points = generate_point_cloud(value_function, n_points_inside, n_points_outside, seed=42 + i)
            np.save(os.path.join(sample_output_dir, f'point_cloud_{i}.npy'), points)
            
        # Also copy the value function
        shutil.copy2(value_function_path, os.path.join(sample_output_dir, 'value_function.npy'))
        
        # Copy environment files
        for file in ['environment_grid.npy', 'environment.png']:
            src = os.path.join(dataset_dir, sample_dir, file)
            dst = os.path.join(sample_output_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        results.append({
            'sample_id': sample_dir,
            'num_points_inside': n_points_inside,
            'num_points_outside': n_points_outside,
            'num_clouds': clouds_per_sample,
            'value_function_shape': str(value_function.shape),
            'point_cloud_shape': f'(n_points_inside + n_points_outside, 4)'
        })
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'num_points_inside', 'num_points_outside', 'num_clouds', 'value_function_shape', 'point_cloud_shape'])
        writer.writeheader()
        writer.writerows(results)
        # Add a summary row for missing samples
        writer.writerow({
            'sample_id': f'MISSING ({missing_count})',
            'num_points_inside': '',
            'num_points_outside': '',
            'num_clouds': '',
            'value_function_shape': '',
            'point_cloud_shape': ''
        })
        if missing_samples:
            writer.writerow({
                'sample_id': 'Missing sample list:',
                'num_points_inside': '',
                'num_points_outside': '',
                'num_clouds': '',
                'value_function_shape': '',
                'point_cloud_shape': ''
            })

def main():
    parser = argparse.ArgumentParser(description='Convert value functions to point clouds')
    parser.add_argument('--dataset_dir', type=str, default='dataset_64', help='Path to input dataset directory')
    parser.add_argument('--output_dir', type=str, default='point_cloud_dataset', help='Path to output directory')
    parser.add_argument('--n_points_inside', type=int, default=1000, help='Number of points to sample inside BRT (value <= 0)')
    parser.add_argument('--n_points_outside', type=int, default=5000, help='Number of points to sample outside BRT (value > 0)')
    parser.add_argument('--clouds_per_sample', type=int, default=4, help='Number of point clouds to generate per sample')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_dir, args.output_dir, args.n_points_inside, args.n_points_outside, args.clouds_per_sample)

if __name__ == '__main__':
    main()
