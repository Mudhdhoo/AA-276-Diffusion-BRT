import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import sys

# Add to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_to_gradient_pointcloud import compute_gradient_probability

def visualize_point_cloud_and_probability(sample_dir: str):
    """
    Visualize a point cloud, probability distribution, BRT, and environment grid.
    
    Args:
        sample_dir: Path to the sample directory containing point_cloud_0.npy and value_function.npy
    """
    # Load data
    point_cloud = np.load(os.path.join(sample_dir, 'point_cloud_0.npy'))
    value_function = np.load(os.path.join(sample_dir, 'value_function.npy'))
    env_grid = np.load(os.path.join(sample_dir, 'environment_grid.npy'))
    
    # Get probability distribution
    prob_dist = compute_gradient_probability(value_function[-1])
    
    # Get BRT (subzero level set)
    converged_value = value_function[-1]
    brt = np.min(converged_value, axis=2) < 0  # BRT is where value is negative in any theta
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D point cloud plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                         c=point_cloud[:, 3], cmap='viridis', alpha=0.6, s=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('θ')
    ax1.set_title('Point Cloud (colored by value)')
    plt.colorbar(scatter, ax=ax1, label='Value')
    
    # Probability distribution heatmap (averaged over theta)
    ax2 = fig.add_subplot(222)
    prob_2d = np.mean(prob_dist, axis=2)  # Average over theta dimension
    im = ax2.imshow(prob_2d.T, origin='lower', extent=[0, 10, 0, 10],
                    cmap='hot', aspect='auto')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Probability Distribution (averaged over θ)')
    plt.colorbar(im, ax=ax2, label='Probability')
    
    # BRT plot
    ax3 = fig.add_subplot(223)
    brt_plot = ax3.imshow(brt.T, origin='lower', extent=[0, 10, 0, 10],
                          cmap='binary', aspect='auto')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Backward Reachable Set (BRT)')
    
    # Environment grid plot
    ax4 = fig.add_subplot(224)
    env_plot = ax4.imshow(env_grid.T, origin='lower', extent=[0, 10, 0, 10],
                          cmap='gray', aspect='auto')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Environment Grid')
    
    plt.tight_layout()
    plt.show()

def main():
    # Use the first sample from the dataset
    dataset_path = '/Users/malte/AA-276-Diffusion-BRT/6000_density_weighted'
    sample_dir = os.path.join(dataset_path, 'sample_000')
    
    visualize_point_cloud_and_probability(sample_dir)

if __name__ == "__main__":
    main() 