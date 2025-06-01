#!/usr/bin/env python3
"""
Script to copy value_function.npy files from a source dataset to the 1070 dataset.
"""

import os
import shutil
import argparse
from tqdm import tqdm

def copy_value_functions(source_dir, target_dir):
    """
    Copy value_function.npy files from source dataset to target dataset.
    
    Args:
        source_dir: Path to source dataset containing sample folders with value_function.npy
        target_dir: Path to target dataset (1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv)
    """
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory does not exist: {target_dir}")
        return
    
    # Get all sample directories from source
    source_samples = [d for d in os.listdir(source_dir) if d.startswith('sample_') and os.path.isdir(os.path.join(source_dir, d))]
    
    # Get all sample directories from target
    target_samples = [d for d in os.listdir(target_dir) if d.startswith('sample_') and os.path.isdir(os.path.join(target_dir, d))]
    
    print(f"Found {len(source_samples)} sample folders in source directory")
    print(f"Found {len(target_samples)} sample folders in target directory")
    
    copied_count = 0
    missing_source = 0
    missing_target = 0
    
    # Copy value functions for matching samples
    for sample_dir in tqdm(target_samples, desc="Copying value functions"):
        source_sample_path = os.path.join(source_dir, sample_dir)
        target_sample_path = os.path.join(target_dir, sample_dir)
        
        # Check if source sample exists
        if not os.path.exists(source_sample_path):
            missing_source += 1
            continue
        
        source_value_func = os.path.join(source_sample_path, 'value_function.npy')
        target_value_func = os.path.join(target_sample_path, 'value_function.npy')
        
        # Check if source value function exists
        if not os.path.exists(source_value_func):
            print(f"Warning: value_function.npy not found in {source_sample_path}")
            continue
        
        # Copy the file
        try:
            shutil.copy2(source_value_func, target_value_func)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {source_value_func} to {target_value_func}: {e}")
    
    print(f"\nCopy complete!")
    print(f"Successfully copied: {copied_count} value function files")
    print(f"Missing source samples: {missing_source}")
    print(f"Total target samples: {len(target_samples)}")

def main():
    parser = argparse.ArgumentParser(description='Copy value_function.npy files from source dataset to 1070 dataset')
    parser.add_argument('source_dir', type=str, 
                       help='Path to source dataset directory containing sample folders with value_function.npy files')
    parser.add_argument('--target_dir', type=str, 
                       default='1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                       help='Path to target dataset directory (default: 1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv)')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute if needed
    if not os.path.isabs(args.target_dir):
        args.target_dir = os.path.join(os.getcwd(), args.target_dir)
    
    copy_value_functions(args.source_dir, args.target_dir)

if __name__ == '__main__':
    main() 