#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path

def get_max_sample_id(dataset_dir):
    """Get the highest sample ID in the dataset directory."""
    max_id = -1
    for item in os.listdir(dataset_dir):
        if item.startswith('sample_'):
            try:
                sample_id = int(item.split('_')[1])
                max_id = max(max_id, sample_id)
            except (ValueError, IndexError):
                continue
    return max_id

def merge_datasets(dataset1_dir, dataset2_dir, output_dir):
    """
    Merge two datasets by copying all samples from dataset1 and renumbering samples from dataset2.
    
    Args:
        dataset1_dir: Path to first dataset
        dataset2_dir: Path to second dataset
        output_dir: Path where merged dataset will be created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the highest sample ID from dataset1
    max_id = get_max_sample_id(dataset1_dir)
    if max_id == -1:
        print("Warning: No samples found in dataset1")
        max_id = -1
    
    # Copy all samples from dataset1
    print(f"Copying samples from {dataset1_dir}...")
    for item in os.listdir(dataset1_dir):
        if item.startswith('sample_'):
            src = os.path.join(dataset1_dir, item)
            dst = os.path.join(output_dir, item)
            shutil.copytree(src, dst)
    
    # Copy and renumber samples from dataset2
    print(f"Copying and renumbering samples from {dataset2_dir}...")
    for item in sorted(os.listdir(dataset2_dir)):
        if item.startswith('sample_'):
            try:
                old_id = int(item.split('_')[1])
                new_id = max_id + 1 + old_id
                new_name = f"sample_{new_id}"
                
                src = os.path.join(dataset2_dir, item)
                dst = os.path.join(output_dir, new_name)
                shutil.copytree(src, dst)
                print(f"Renamed {item} to {new_name}")
            except (ValueError, IndexError):
                print(f"Warning: Skipping invalid sample name {item}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Merge two datasets by renumbering sample folders')
    parser.add_argument('dataset1', help='Path to first dataset directory')
    parser.add_argument('dataset2', help='Path to second dataset directory')
    parser.add_argument('output', help='Path to output directory for merged dataset')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    dataset1_dir = os.path.abspath(args.dataset1)
    dataset2_dir = os.path.abspath(args.dataset2)
    output_dir = os.path.abspath(args.output)
    
    # Validate directories
    if not os.path.isdir(dataset1_dir):
        raise ValueError(f"Dataset1 directory does not exist: {dataset1_dir}")
    if not os.path.isdir(dataset2_dir):
        raise ValueError(f"Dataset2 directory does not exist: {dataset2_dir}")
    
    merge_datasets(dataset1_dir, dataset2_dir, output_dir)
    print("Dataset merging completed successfully!")

if __name__ == "__main__":
    main() 