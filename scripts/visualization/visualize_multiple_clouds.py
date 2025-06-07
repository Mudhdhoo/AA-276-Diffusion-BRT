import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.BRTDataset import BRTDataset

def visualize_point_clouds(ax, points, title, color='b', alpha=0.6):
    """Visualize a single point cloud on the given axis"""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=color, alpha=alpha, s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    # Set the view to look at the first quadrant (positive x, y, z)
    ax.view_init(elev=20, azim=45)

def visualize_environment(ax, env_grid):
    """Visualize the environment grid as a heatmap with (0,0) at the bottom-left (Cartesian)"""
    # Transpose grid so that grid[i,j] maps to physical position (x=j, y=i)
    ax.imshow(env_grid.T, cmap='gray', origin='lower', extent=[0, 10, 0, 10])
    ax.set_title('Environment Grid')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

def main():
    # Initialize dataset
    dataset_dir = 'point_cloud_dataset_4000'
    dataset = BRTDataset(dataset_dir)
    
    # Get a random sample directory
    sample_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sample_')]
    random_sample = random.choice(sample_dirs)
    print(f"Visualizing sample: {random_sample}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Create 3D subplots for point clouds
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Load and visualize each point cloud
    for i in range(4):
        pc_file = f'point_cloud_{i}.npy'
        points = np.load(os.path.join(dataset_dir, random_sample, pc_file))
        
        # Denormalize points
        points = dataset.denormalize_points(points)
        
        # Visualize on corresponding subplot
        ax = [ax1, ax2, ax3, ax4][i]
        visualize_point_clouds(ax, points, f'Point Cloud {i}')
    
    # Add environment visualization
    env_fig = plt.figure(figsize=(6, 6))
    env_ax = env_fig.add_subplot(111)
    env_grid = np.load(os.path.join(dataset_dir, random_sample, 'environment_grid.npy'))
    visualize_environment(env_ax, env_grid)
    
    # Adjust layout and show
    fig.tight_layout()
    env_fig.tight_layout()
    
    # Save figures
    os.makedirs('visualizations', exist_ok=True)
    fig.savefig(f'visualizations/point_clouds_{random_sample}.png', dpi=300, bbox_inches='tight')
    env_fig.savefig(f'visualizations/environment_{random_sample}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    main() 