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
    x = points[:, 0]
    y = points[:, 1]
    theta = points[:, 2]
    
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
    # Transpose grid so that grid[i,j] maps to physical position (x=j, y=i)
    plt.imshow(grid.T, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10], origin='lower')
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
        x = points[:, 0]
        y = points[:, 1]
        theta = points[:, 2]
        
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
    # Transpose grid so that grid[i,j] maps to physical position (x=j, y=i)
    im = ax1.imshow(env_grid.T, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10], origin='lower')
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
    x = true_pc[:, 0]
    y = true_pc[:, 1]
    theta = true_pc[:, 2]
    
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
    x = generated_pc[:, 0]
    y = generated_pc[:, 1]
    theta = generated_pc[:, 2]
    
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
        
        x = points[:, 0]
        y = points[:, 1]
        theta = points[:, 2]
        
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
         #   plt.colorbar(scatter, ax=ax, label='Value Function')
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
    # ax_true = fig.add_subplot(1, n_steps + 1, n_steps + 1, projection='3d')
    # if dataset is not None:
    #     true_pc = dataset.denormalize_points(true_pc)
    # x = true_pc[:, 0]
    # y = true_pc[:, 1]
    # theta = true_pc[:, 2]
    
    # Check if we have a 4th dimension
    if true_pc.shape[1] > 3:
        w = true_pc[:, 3]  # No scaling needed for safety value
        
        # Ensure 0 is centered in the colormap
        vmin, vmax = w.min(), w.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create dual-scale colormap
        cmap = create_dual_colormap()
        
       # scatter_true = ax_true.scatter(x, y, theta, c=w, cmap=cmap, s=2, alpha=0.5, label='BRT Points', vmin=vmin, vmax=vmax)
       # plt.colorbar(scatter_true, ax=ax_true, label='Value Function')
    else:
        pass
      #  scatter_true = ax_true.scatter(x, y, theta, s=2, alpha=0.5, label='BRT Points')
    
    #ax_true.set_title('True BRT Point Cloud', fontsize=14, pad=20)
    #ax_true.set_xlabel('X Position (m)', fontsize=12)
    #ax_true.set_ylabel('Y Position (m)', fontsize=12)
    #ax_true.set_zlabel('θ (rad)', fontsize=12)
    #ax_true.set_xlim(0, 10)
    #ax_true.set_ylim(0, 10)
    #ax_true.set_zlim(-np.pi, np.pi)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_detailed_value_function_comparison(true_pc, pred_pc, env_grid, title=None, save_path=None, dataset=None, sample_l2_error=None):
    """
    Create detailed comparison visualization with theta slices showing value function differences.
    
    Args:
        true_pc: True point cloud (N, 4) with [x, y, theta, value]
        pred_pc: Predicted point cloud (M, 4) with [x, y, theta, value]
        env_grid: Environment grid (64, 64)
        title: Plot title
        save_path: Path to save the figure
        dataset: Dataset for denormalization
        sample_l2_error: Optional, mean L2 error for this sample (float)
    """
    # Denormalize points if dataset provided
    if dataset is not None:
        true_pc = dataset.denormalize_points(true_pc)
        pred_pc = dataset.denormalize_points(pred_pc)
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(25, 15))
    
    # Define theta slice values for visualization
    theta_slices = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    theta_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    
    # Get value ranges for consistent coloring
    if true_pc.shape[1] > 3 and pred_pc.shape[1] > 3:
        all_values = np.concatenate([true_pc[:, 3], pred_pc[:, 3]])
        vmin, vmax = all_values.min(), all_values.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        cmap = create_dual_colormap()
    else:
        vmin, vmax = -1, 1
        cmap = 'viridis'
    
    # Row 1: Environment and 3D overview
    # Environment
    ax_env = fig.add_subplot(4, 6, 1)
    # Transpose grid so that grid[i,j] maps to physical position (x=j, y=i)
    ax_env.imshow(env_grid.T, cmap='binary', vmin=0, vmax=1, extent=[0, 10, 0, 10], origin='lower')
    ax_env.set_title('Environment', fontsize=12, fontweight='bold')
    ax_env.set_xlabel('X Position')
    ax_env.set_ylabel('Y Position')
    
    # True 3D
    ax_true_3d = fig.add_subplot(4, 6, 2, projection='3d')
    if true_pc.shape[1] > 3:
        scatter_true = ax_true_3d.scatter(true_pc[:, 0], true_pc[:, 1], true_pc[:, 2], 
                                         c=true_pc[:, 3], cmap=cmap, s=1, alpha=0.6, 
                                         vmin=vmin, vmax=vmax)
    else:
        scatter_true = ax_true_3d.scatter(true_pc[:, 0], true_pc[:, 1], true_pc[:, 2], 
                                         s=1, alpha=0.6)
    ax_true_3d.set_title('True BRT (3D)', fontsize=12, fontweight='bold')
    ax_true_3d.set_xlabel('X'); ax_true_3d.set_ylabel('Y'); ax_true_3d.set_zlabel('θ')
    ax_true_3d.set_xlim(0, 10); ax_true_3d.set_ylim(0, 10); ax_true_3d.set_zlim(-np.pi, np.pi)
    
    # Predicted 3D
    ax_pred_3d = fig.add_subplot(4, 6, 3, projection='3d')
    if pred_pc.shape[1] > 3:
        scatter_pred = ax_pred_3d.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], 
                                         c=pred_pc[:, 3], cmap=cmap, s=1, alpha=0.6, 
                                         vmin=vmin, vmax=vmax)
    else:
        scatter_pred = ax_pred_3d.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], 
                                         s=1, alpha=0.6)
    ax_pred_3d.set_title('Predicted BRT (3D)', fontsize=12, fontweight='bold')
    ax_pred_3d.set_xlabel('X'); ax_pred_3d.set_ylabel('Y'); ax_pred_3d.set_zlabel('θ')
    ax_pred_3d.set_xlim(0, 10); ax_pred_3d.set_ylim(0, 10); ax_pred_3d.set_zlim(-np.pi, np.pi)
    
    # Statistics comparison
    ax_stats = fig.add_subplot(4, 6, 4)
    ax_stats.axis('off')
    
    if true_pc.shape[1] > 3 and pred_pc.shape[1] > 3:
        stats_text = f"""Statistics Comparison:
        
True BRT:
  Points: {len(true_pc):,}
  Value range: [{true_pc[:, 3].min():.3f}, {true_pc[:, 3].max():.3f}]
  Value mean: {true_pc[:, 3].mean():.3f}
  Value std: {true_pc[:, 3].std():.3f}
  Negative %: {(true_pc[:, 3] < 0).sum() / len(true_pc) * 100:.1f}%

Predicted BRT:
  Points: {len(pred_pc):,}
  Value range: [{pred_pc[:, 3].min():.3f}, {pred_pc[:, 3].max():.3f}]
  Value mean: {pred_pc[:, 3].mean():.3f}
  Value std: {pred_pc[:, 3].std():.3f}
  Negative %: {(pred_pc[:, 3] < 0).sum() / len(pred_pc) * 100:.1f}%
        """
    else:
        stats_text = f"""Statistics Comparison:
        
True BRT: {len(true_pc):,} points
Predicted BRT: {len(pred_pc):,} points
(No value function available)
        """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=10,
                 verticalalignment='top', family='monospace')
    
    # Value distribution histograms
    if true_pc.shape[1] > 3 and pred_pc.shape[1] > 3:
        ax_hist = fig.add_subplot(4, 6, 5)
        ax_hist.hist(true_pc[:, 3], bins=50, alpha=0.7, label='True', color='blue', density=True)
        ax_hist.hist(pred_pc[:, 3], bins=50, alpha=0.7, label='Predicted', color='red', density=True)
        ax_hist.set_xlabel('Value Function')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Value Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    
    # Colorbar
    if true_pc.shape[1] > 3 or pred_pc.shape[1] > 3:
        ax_cbar = fig.add_subplot(4, 6, 6)
        ax_cbar.axis('off')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_cbar, fraction=0.8, aspect=20)
        cbar.set_label('Value Function', fontsize=12, fontweight='bold')
    
    # Rows 2-4: Theta slices
    for row in range(3):
        for col in range(5):
            slice_idx = row * 5 + col
            if slice_idx >= len(theta_slices):
                break
                
            theta_val = theta_slices[slice_idx]
            theta_label = theta_labels[slice_idx]
            
            # True BRT slice
            ax_slice = fig.add_subplot(4, 6, 7 + slice_idx)
            
            if true_pc.shape[1] > 3 and pred_pc.shape[1] > 3:
                # Find points close to this theta value
                theta_tolerance = 0.3  # ~17 degrees
                
                # True points
                true_mask = np.abs(true_pc[:, 2] - theta_val) < theta_tolerance
                true_slice_points = true_pc[true_mask]
                
                # Predicted points  
                pred_mask = np.abs(pred_pc[:, 2] - theta_val) < theta_tolerance
                pred_slice_points = pred_pc[pred_mask]
                
                # Plot true points
                if len(true_slice_points) > 0:
                    scatter1 = ax_slice.scatter(true_slice_points[:, 0], true_slice_points[:, 1], 
                                              c=true_slice_points[:, 3], cmap=cmap, s=20, alpha=0.8,
                                              vmin=vmin, vmax=vmax, marker='o', label='True')
                
                # Plot predicted points
                if len(pred_slice_points) > 0:
                    scatter2 = ax_slice.scatter(pred_slice_points[:, 0], pred_slice_points[:, 1], 
                                              c=pred_slice_points[:, 3], cmap=cmap, s=15, alpha=0.8,
                                              vmin=vmin, vmax=vmax, marker='x', label='Pred')
                
                # Add legend only for first slice
                if slice_idx == 0:
                    ax_slice.legend(loc='upper right', fontsize=8)
                    
                ax_slice.set_title(f'θ = {theta_label}\n({len(true_slice_points)} true, {len(pred_slice_points)} pred)', 
                                 fontsize=10, fontweight='bold')
            else:
                # Just show spatial points without values
                true_mask = np.abs(true_pc[:, 2] - theta_val) < 0.3
                true_slice_points = true_pc[true_mask]
                pred_mask = np.abs(pred_pc[:, 2] - theta_val) < 0.3  
                pred_slice_points = pred_pc[pred_mask]
                
                if len(true_slice_points) > 0:
                    ax_slice.scatter(true_slice_points[:, 0], true_slice_points[:, 1], 
                                   s=20, alpha=0.8, marker='o', label='True', color='blue')
                if len(pred_slice_points) > 0:
                    ax_slice.scatter(pred_slice_points[:, 0], pred_slice_points[:, 1], 
                                   s=15, alpha=0.8, marker='x', label='Pred', color='red')
                
                ax_slice.set_title(f'θ = {theta_label}', fontsize=10, fontweight='bold')
            
            # Show environment overlay
            # Transpose grid so that grid[i,j] maps to physical position (x=j, y=i)
            ax_slice.imshow(env_grid.T, cmap='gray', alpha=0.3, extent=[0, 10, 0, 10], origin='lower')
            ax_slice.set_xlim(0, 10)
            ax_slice.set_ylim(0, 10)
            ax_slice.set_xlabel('X Position')
            ax_slice.set_ylabel('Y Position')
            ax_slice.grid(True, alpha=0.3)
    
    # Set the figure title, including L2 error if provided
    if title is not None and sample_l2_error is not None:
        fig.suptitle(f"{title}\nSample L2 Error: {sample_l2_error:.6f}", fontsize=16, fontweight='bold', y=0.98)
    elif title is not None:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    elif sample_l2_error is not None:
        fig.suptitle(f"Sample L2 Error: {sample_l2_error:.6f}", fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()