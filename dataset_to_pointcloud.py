import os
import numpy as np
import argparse
from tqdm import tqdm
import shutil
from scipy.interpolate import interpn
import csv

def generate_point_cloud(value_function, n_points):
    """
    Generate a point cloud from the value function where value_function[-1] <= 0
    using interpolation for better sampling. Ensures exactly n_points are returned.
    Args:
        value_function: 4D array (time_steps, x, y, z)
        n_points: number of points to sample
    Returns:
        points: Nx3 array of points
    """
    # Get the last timestep
    last_timestep = value_function[-1]
    
    # Create grid points for interpolation
    x = np.arange(last_timestep.shape[0])
    y = np.arange(last_timestep.shape[1])
    z = np.arange(last_timestep.shape[2])
    points = (x, y, z)
    
    # Initialize arrays to store valid points
    valid_points = np.zeros((n_points, 3))
    points_found = 0
    
    # Keep generating points until we have enough
    batch_size = n_points * 10  # Start with 10x the needed points
    while points_found < n_points:
        # Generate random points in the volume
        x_rand = np.random.uniform(0, last_timestep.shape[0]-1, batch_size)
        y_rand = np.random.uniform(0, last_timestep.shape[1]-1, batch_size)
        z_rand = np.random.uniform(0, last_timestep.shape[2]-1, batch_size)
        
        # Stack coordinates for interpolation
        xi = np.stack([x_rand, y_rand, z_rand], axis=1)
        
        # Interpolate values at random points
        values = interpn(points, last_timestep, xi, method='linear', bounds_error=False, fill_value=1.0)
        
        # Keep only points where value <= 0
        valid_mask = values <= 0
        new_valid_points = xi[valid_mask]
        
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

def process_dataset(dataset_dir, output_dir, n_points):
    """
    Process all samples in the dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sample_')]
    results = []
    missing_count = 0
    missing_samples = []
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_output_dir = os.path.join(output_dir, sample_dir)
        os.makedirs(sample_output_dir, exist_ok=True)
        value_function_path = os.path.join(dataset_dir, sample_dir, 'value_function.npy')
        if not os.path.exists(value_function_path):
            missing_count += 1
            missing_samples.append(sample_dir)
            continue
        value_function = np.load(value_function_path)
        points = generate_point_cloud(value_function, n_points)
        np.save(os.path.join(sample_output_dir, 'point_cloud.npy'), points)
        for file in ['environment_grid.npy', 'environment.png']:
            src = os.path.join(dataset_dir, sample_dir, file)
            dst = os.path.join(sample_output_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        results.append({
            'sample_id': sample_dir,
            'num_points': len(points),
            'value_function_shape': str(value_function.shape),
            'point_cloud_shape': str(points.shape)
        })
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'num_points', 'value_function_shape', 'point_cloud_shape'])
        writer.writeheader()
        writer.writerows(results)
        # Add a summary row for missing samples
        writer.writerow({
            'sample_id': f'MISSING ({missing_count})',
            'num_points': '',
            'value_function_shape': '',
            'point_cloud_shape': ''
        })
        if missing_samples:
            writer.writerow({
                'sample_id': 'Missing sample list:',
                'num_points': ', '.join(missing_samples),
                'value_function_shape': '',
                'point_cloud_shape': ''
            })

def main():
    parser = argparse.ArgumentParser(description='Convert value functions to point clouds')
    parser.add_argument('--dataset_dir', type=str, default='dataset_64', help='Path to input dataset directory')
    parser.add_argument('--output_dir', type=str, default='point_cloud_dataset', help='Path to output directory')
    parser.add_argument('--n_points', type=int, default=2048, help='Number of points to sample per cloud')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_dir, args.output_dir, args.n_points)

if __name__ == '__main__':
    main()
