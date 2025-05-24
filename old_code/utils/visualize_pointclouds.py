import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def visualize_point_clouds(input_dir, n_samples):
    # Find all sample directories
    sample_dirs = [d for d in os.listdir(input_dir) if d.startswith('sample_')]
    if not sample_dirs:
        print('No sample directories found.')
        return
    
    # Randomly select n_samples
    n_samples = min(n_samples, len(sample_dirs))
    selected_samples = random.sample(sample_dirs, n_samples)
    
    # Set up subplots
    ncols = min(n_samples, 3)
    nrows = (n_samples + ncols - 1) // ncols
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    for i, sample_dir in enumerate(selected_samples):
        pc_path = os.path.join(input_dir, sample_dir, 'point_cloud.npy')
        if not os.path.exists(pc_path):
            print(f'No point cloud found for {sample_dir}, skipping.')
            continue
        points = np.load(pc_path)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.5)
        ax.set_title(f'{sample_dir}\n{points.shape[0]} points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize point clouds from a generated folder')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder with sample_* subfolders')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of random samples to visualize')
    args = parser.parse_args()
    visualize_point_clouds(args.input_dir, args.n_samples)

if __name__ == '__main__':
    main() 