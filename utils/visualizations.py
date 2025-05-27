import matplotlib.pyplot as plt
import numpy as np
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

def visualize_point_cloud(points, title=None, save_path=None, dataset=None):
    """Visualize a single point cloud and optionally save it."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Denormalize points if dataset is provided
    if dataset is not None:
        points = dataset.denormalize_points(points)
    
    # Scale coordinates to match environment dimensions
    x = points[:, 0]  # Scale x from [0,64] to [0,10]
    y = points[:, 1]  # Scale y from [0,64] to [0,10]
    theta = points[:, 2]  # Scale theta from [0,64] to [-π,π]
    
    # Check if we have a 4th dimension
    if points.shape[1] > 3:
        w = points[:, 3]  # No scaling needed for safety value
        
        # Ensure 0 is centered in the colormap
        vmin, vmax = w.min(), w.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create dual-scale colormap
        cmap = create_dual_colormap()
        
        scatter = ax.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, label='Value Function')
    else:
        ax.scatter(x, y, theta, s=2, alpha=0.5)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('θ')
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(-np.pi, np.pi)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_environment_grid(grid, title=None, save_path=None):
    """Visualize a single environment grid and optionally save it."""
    plt.figure(figsize=(8, 8))
    # Use binary colormap and set vmin/vmax to ensure binary visualization
    plt.imshow(grid, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10])  # Set extent to match coordinate system
    if title:
        plt.title(title)
    plt.colorbar(label='Obstacle (1) / Free Space (0)')
    plt.axis('equal')  # Ensure equal aspect ratio
    # Set axis limits to match coordinate system
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_denoising_process(points_sequence, titles, save_path=None, dataset=None):
    """Visualize the denoising process in a single figure with subplots.
    
    Args:
        points_sequence: List of point clouds to visualize
        titles: List of titles for each subplot
        save_path: Optional path to save the figure
        dataset: Dataset object for denormalization
    """
    n_steps = len(points_sequence)
    n_cols = 5  # Fixed number of columns (initial + 3 steps + final)
    n_rows = 1  # Single row
    
    fig = plt.figure(figsize=(25, 5))  # Wider figure for 5 subplots
    
    for i, (points, title) in enumerate(zip(points_sequence, titles)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        
        # Denormalize points if dataset is provided
        if dataset is not None:
            points = dataset.denormalize_points(points)
        
        # Scale coordinates to match environment dimensions
        x = points[:, 0] * (10.0 / 64.0)  # Scale x from [0,64] to [0,10]
        y = points[:, 1] * (10.0 / 64.0)  # Scale y from [0,64] to [0,10]
        theta = points[:, 2] * (2 * np.pi / 64.0) - np.pi  # Scale theta from [0,64] to [-π,π]
        
        ax.scatter(x, y, theta, s=2, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('θ')
        # Set axis limits
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-np.pi, np.pi)
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_comparison(true_pc, generated_pc, env_grid, title=None, save_path=None, dataset=None):
    """Visualize true BRT, generated BRT, and environment side by side."""
    # Create figure with specific size ratio to ensure square environment plot
    fig = plt.figure(figsize=(30, 10))
    
    # Environment subplot - make it square
    ax1 = fig.add_subplot(131, aspect='equal')
    im = ax1.imshow(env_grid, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10])
    ax1.set_title('Environment Grid', fontsize=14, pad=20)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Obstacle (1) / Free Space (0)', fontsize=12)
    
    # True BRT subplot
    ax2 = fig.add_subplot(132, projection='3d')
    if dataset is not None:
        true_pc = dataset.denormalize_points(true_pc)
    x = true_pc[:, 0] * (10.0 / 64.0)
    y = true_pc[:, 1] * (10.0 / 64.0)
    theta = true_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    
    # Check if we have a 4th dimension
    if true_pc.shape[1] > 3:
        w = true_pc[:, 3]  # No scaling needed for safety value
        
        # Ensure 0 is centered in the colormap
        vmin, vmax = w.min(), w.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create dual-scale colormap
        cmap = create_dual_colormap()
        
        scatter2 = ax2.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, label='BRT Points', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter2, ax=ax2, label='Value Function')
    else:
        scatter2 = ax2.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    
    ax2.set_title('True BRT Point Cloud', fontsize=14, pad=20)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_zlabel('θ (rad)', fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_zlim(-np.pi, np.pi)
    
    # Generated BRT subplot
    ax3 = fig.add_subplot(133, projection='3d')
    if dataset is not None:
        generated_pc = dataset.denormalize_points(generated_pc)
    x = generated_pc[:, 0] * (10.0 / 64.0)
    y = generated_pc[:, 1] * (10.0 / 64.0)
    theta = generated_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    
    # Check if we have a 4th dimension
    if generated_pc.shape[1] > 3:
        w = generated_pc[:, 3]  # No scaling needed for safety value
        
        # Ensure 0 is centered in the colormap
        vmin, vmax = w.min(), w.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create dual-scale colormap
        cmap = create_dual_colormap()
        
        scatter3 = ax3.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, label='BRT Points', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter3, ax=ax3, label='Value Function')
    else:
        scatter3 = ax3.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    
    ax3.set_title('Generated BRT Point Cloud', fontsize=14, pad=20)
    ax3.set_xlabel('X Position (m)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.set_zlabel('θ (rad)', fontsize=12)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_zlim(-np.pi, np.pi)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_denoising_with_true(points_sequence, true_pc, titles, save_path=None, dataset=None):
    """Visualize the denoising process and true BRT side by side."""
    n_steps = len(points_sequence)
    fig = plt.figure(figsize=(30, 8))
    
    # Denoising process subplots
    for i, (points, title) in enumerate(zip(points_sequence, titles)):
        ax = fig.add_subplot(1, n_steps + 1, i + 1, projection='3d')
        
        if dataset is not None:
            points = dataset.denormalize_points(points)
        
        x = points[:, 0] * (10.0 / 64.0)
        y = points[:, 1] * (10.0 / 64.0)
        theta = points[:, 2] * (2 * np.pi / 64.0) - np.pi
        
        # Check if we have a 4th dimension
        if points.shape[1] > 3:
            w = points[:, 3]  # No scaling needed for safety value
            
            # Ensure 0 is centered in the colormap
            vmin, vmax = w.min(), w.max()
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            
            # Create dual-scale colormap
            cmap = create_dual_colormap()
            
            scatter = ax.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, label='BRT Points', vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, ax=ax, label='Value Function')
        else:
            scatter = ax.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
        
        ax.set_title(f'Denoising Step {title}', fontsize=14, pad=20)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('θ (rad)', fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-np.pi, np.pi)
    
    # True BRT subplot
    ax_true = fig.add_subplot(1, n_steps + 1, n_steps + 1, projection='3d')
    if dataset is not None:
        true_pc = dataset.denormalize_points(true_pc)
    x = true_pc[:, 0] * (10.0 / 64.0)
    y = true_pc[:, 1] * (10.0 / 64.0)
    theta = true_pc[:, 2] * (2 * np.pi / 64.0) - np.pi
    
    # Check if we have a 4th dimension
    if true_pc.shape[1] > 3:
        w = true_pc[:, 3]  # No scaling needed for safety value
        
        # Ensure 0 is centered in the colormap
        vmin, vmax = w.min(), w.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create dual-scale colormap
        cmap = create_dual_colormap()
        
        scatter_true = ax_true.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, label='BRT Points', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter_true, ax=ax_true, label='Value Function')
    else:
        scatter_true = ax_true.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    
    ax_true.set_title('True BRT Point Cloud', fontsize=14, pad=20)
    ax_true.set_xlabel('X Position (m)', fontsize=12)
    ax_true.set_ylabel('Y Position (m)', fontsize=12)
    ax_true.set_zlabel('θ (rad)', fontsize=12)
    ax_true.set_xlim(0, 10)
    ax_true.set_ylim(0, 10)
    ax_true.set_zlim(-np.pi, np.pi)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()