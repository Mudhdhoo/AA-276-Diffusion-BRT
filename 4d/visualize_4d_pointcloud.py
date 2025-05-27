import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

def create_dual_colormap():
    """
    Create a colormap with two distinct scales:
    - Negative values: Orange to Red (unsafe)
    - Positive values: Green to Blue (safe)
    """
    unsafe_colors = [(1.0, 0.6, 0.0),  # Orange
                    (0.8, 0.0, 0.0)]   # Red
    safe_colors = [(0.0, 0.8, 0.0),   # Green
                  (0.0, 0.4, 0.8)]    # Blue
    
    # Create the colormaps
    unsafe_cmap = LinearSegmentedColormap.from_list('unsafe', unsafe_colors, N=50)
    safe_cmap = LinearSegmentedColormap.from_list('safe', safe_colors, N=50)
    
    # Combine the colormaps
    colors = []
    # Add unsafe colors (reversed to go from orange to red)
    colors.extend(unsafe_cmap(np.linspace(1, 0, 50)))
    # Add safe colors
    colors.extend(safe_cmap(np.linspace(0, 1, 50)))
    
    return LinearSegmentedColormap.from_list('dual_scale', colors, N=100)

def visualize_point_cloud(points, title):
    """
    Visualize a 4D point cloud in 3D space with color mapping based on the 4th dimension.
    Args:
        points: Nx4 array where first 3 dimensions are spatial coordinates and 4th is value
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates and values
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    values = points[:, 3]
    
    # Ensure 0 is centered in the colormap
    vmin, vmax = values.min(), values.max()
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Create dual-scale colormap
    cmap = create_dual_colormap()
    
    # Plot points with continuous color mapping
    scatter = ax.scatter(x, y, z, c=values, cmap=cmap, s=4, alpha=0.6, vmin=vmin, vmax=vmax)
    
    # Add colorbar with explicit value range
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Value Function')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize 4D point clouds from dataset')
    parser.add_argument('--dataset_dir', type=str, default='4d/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv',
                      help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=3,
                      help='Number of sample environments to visualize')
    parser.add_argument('--output_dir', type=str, default='4d/visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of sample directories
    sample_dirs = [d for d in os.listdir(args.dataset_dir) if d.startswith('sample_')]
    sample_dirs = sample_dirs[:args.num_samples]  # Take only specified number of samples
    
    for sample_dir in tqdm(sample_dirs, desc="Visualizing samples"):
        sample_path = os.path.join(args.dataset_dir, sample_dir)
        
        # Get all point cloud files for this sample
        point_cloud_files = [f for f in os.listdir(sample_path) if f.startswith('point_cloud_') and f.endswith('.npy')]
        
        # Create a figure with subplots for each point cloud
        n_clouds = len(point_cloud_files)
        fig = plt.figure(figsize=(20, 5 * n_clouds))
        
        for i, cloud_file in enumerate(point_cloud_files):
            # Load point cloud
            points = np.load(os.path.join(sample_path, cloud_file))
            
            # Create subplot
            ax = fig.add_subplot(n_clouds, 1, i+1, projection='3d')
            
            # Extract coordinates and values
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            values = points[:, 3]
            
            # Ensure 0 is centered in the colormap
            vmin, vmax = values.min(), values.max()
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            
            # Create dual-scale colormap
            cmap = create_dual_colormap()
            
            # Plot points with continuous color mapping
            scatter = ax.scatter(x, y, z, c=values, cmap=cmap, s=4, alpha=0.6, vmin=vmin, vmax=vmax)
            
            # Add colorbar with explicit value range
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Value Function')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{sample_dir} - Point Cloud {i}')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{sample_dir}_visualization.png'))
        plt.show()
        plt.close()

if __name__ == '__main__':
    main() 