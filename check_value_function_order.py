#!/usr/bin/env python3
"""
Script to verify value function grid coordinate orderings by comparing point cloud values
with interpolated value function values.
"""

import torch
import numpy as np
from dataset.BRTDataset import BRTDataset
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from tqdm import tqdm

def check_value_function_order(dataset_dir, split='train'):
    # Load dataset with value functions
    dataset = BRTDataset(dataset_dir, split=split, return_value_function=True)
    
    print(f"\nChecking {len(dataset)} samples...")
    
    # Store statistics for all samples
    all_differences = []
    
    # Debug: Print first sample details
    target_points, env, value_function = dataset[0]
    value_func_3d = value_function.numpy()
    target_points = target_points.numpy()
    denorm_points = dataset.denormalize_points(target_points)
    coords = denorm_points[:, :3]
    point_values = denorm_points[:, 3]
    
    print("\nFirst sample debug info:")
    print("=" * 50)
    print("Original coordinates (first 5 points):")
    print(coords[:5])
    
    # Use perfect inverse scaling (matching dataset generation)
    x_indices = (coords[:, 0] / 10.0 * 64).clip(0, 63)
    y_indices = (coords[:, 1] / 10.0 * 64).clip(0, 63)
    theta_indices = ((coords[:, 2] + np.pi) / (2 * np.pi) * 64).clip(0, 63)
    
    print("\nGrid indices (first 5 points, perfect inverse):")
    print(np.stack([x_indices[:5], y_indices[:5], theta_indices[:5]], axis=1))
    
    # Interpolate at these floating-point indices
    x_coords = np.arange(64)
    y_coords = np.arange(64)
    theta_coords = np.arange(64)
    grid_coords = np.stack([x_indices, y_indices, theta_indices], axis=1)
    interpolated_values = interpn(
        (x_coords, y_coords, theta_coords),
        value_func_3d,
        grid_coords,
        method='linear',
        bounds_error=True
        )
    print("\nInterpolated values (first 5 points):")
    print(interpolated_values[:5])
    print("\nPoint cloud values (first 5 points):")
    print(point_values[:5])
    print("\nDiff (first 5 points):")
    print(point_values[:5] - interpolated_values[:5])
    
    # Now continue with the full dataset check
    for sample_idx in tqdm(range(len(dataset))):
        # Get a sample
        target_points, env, value_function = dataset[sample_idx]
        
        # Convert to numpy
        value_func_3d = value_function.numpy()  # (64, 64, 64)
        target_points = target_points.numpy()   # (N, 4) - [x, y, theta, value]
        
        # Denormalize points to physical coordinates
        denorm_points = dataset.denormalize_points(target_points)
        
        # Extract coordinates and values
        coords = denorm_points[:, :3]  # [x, y, theta]
        point_values = denorm_points[:, 3]  # values from point cloud
        
        # Use perfect inverse scaling
        x_indices = (coords[:, 0] / 10.0 * 64).clip(0, 63)
        y_indices = (coords[:, 1] / 10.0 * 64).clip(0, 63)
        theta_indices = ((coords[:, 2] + np.pi) / (2 * np.pi) * 64).clip(0, 63)
        grid_coords = np.stack([x_indices, y_indices, theta_indices], axis=1)
        
        # Interpolate value function at point coordinates
        interpolated_values = interpn(
            (x_coords, y_coords, theta_coords),
            value_func_3d,
            grid_coords,
            method='linear',
            bounds_error=False,
            fill_value=1.0
        )
        
        # Compute differences
        differences = point_values - interpolated_values
        all_differences.extend(differences)
    
    # Convert to numpy array for statistics
    all_differences = np.array(all_differences)
    abs_diffs = np.abs(all_differences)
    
    print("\nAggregate statistics across all samples:")
    print("=" * 50)
    print(f"Number of points checked: {len(all_differences)}")
    print(f"Mean absolute difference: {np.mean(abs_diffs):.6f}")
    print(f"Max absolute difference: {np.max(abs_diffs):.6f}")
    print(f"Min absolute difference: {np.min(abs_diffs):.6f}")
    print(f"Std of differences: {np.std(all_differences):.6f}")
    
    # Plot histogram of all differences
    plt.figure(figsize=(10, 6))
    plt.hist(all_differences, bins=50)
    plt.title('Distribution of Differences between Point Cloud and Grid Values')
    plt.xlabel('Difference (Point Cloud - Grid)')
    plt.ylabel('Count')
    plt.savefig('value_differences_histogram.png')
    print("\nSaved histogram to value_differences_histogram.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check value function grid coordinate orderings')
    parser.add_argument('--dataset_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train',
                      help='Dataset split to use')
    args = parser.parse_args()
    
    check_value_function_order(args.dataset_dir, args.split) 