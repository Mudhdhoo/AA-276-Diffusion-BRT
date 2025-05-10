import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import jax.numpy as jnp
import os
from matplotlib.colors import LinearSegmentedColormap
import jax

def plot_3d_value_evolution(values, grid, times, save_path='outputs/3d_value_evolution.gif',
                          level=0.0, opacity=0.7,
                          elev=20, azim=45, interval=200, max_frames=20):
    """
    Create a 3D visualization of the value function's evolution over time.
    The Backward Reachable Tube (BRT) is visualized as an isosurface.
    
    Args:
        values: ndarray with shape [
                len(times),
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1]),
                len(grid.coordinate_vectors[2])
            ]
        grid: Grid object containing coordinate_vectors
        times: ndarray with shape [len(times)]
        save_path: Path to save the output GIF
        level: Value level to visualize the isosurface at (default 0.0 for BRT boundary)
        opacity: Opacity of the isosurface (0-1)
        cmap: Colormap to use (default: custom dark RdYlGn)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        interval: Delay between frames in milliseconds
        max_frames: Maximum number of frames to use in the animation (will downsample if needed)
    """
    # Move values to CPU if they're on GPU
    values = jax.device_get(values)
    times = jax.device_get(times)
    
    # Downsample time series if needed
    if len(times) > max_frames:
        # Create evenly spaced indices for downsampling
        indices = np.linspace(0, len(times)-1, max_frames, dtype=int)
        values = values[indices]
        times = times[indices]
    
    # Create a darker version of RdYlGn
    colors = [
        (0.0, '#8B0000'),    # Dark red for unsafe states (BRT)
        (0.3, '#CD5C5C'),    # Indian red
        (0.499, '#FFB6C1'),  # Light pink
        (0.5, '#8B0000'),    # Dark red for boundary
        (0.501, '#E0FFF0'),  # Light cyan
        (0.7, '#32CD32'),    # Lime green
        (1.0, '#006400')     # Dark green for safe states
    ]
    custom_cmap = LinearSegmentedColormap.from_list('RdYlGn_dark', colors)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize values for consistent coloring
    abs_max = max(abs(np.min(values)), abs(np.max(values)))
    vmin, vmax = -abs_max, abs_max
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create mappable for colorbar (only once)
    mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=10)
    cbar.set_label('Value V(X,t)')
    
    # Create initial plot
    def plot_frame(i):
        ax.clear()
        
        # Add semi-transparent xy-plane at z=0
        x = np.linspace(grid.coordinate_vectors[0][0], grid.coordinate_vectors[0][-1], 2)
        y = np.linspace(grid.coordinate_vectors[1][0], grid.coordinate_vectors[1][-1], 2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        # Plot isosurface at the specified level
        if np.any(values[i] < level) and np.any(values[i] > level):
            # Use a lower resolution for marching cubes
            current_values = np.asarray(values[i])
            if current_values.ndim != 3:
                raise ValueError(f"Expected 3D array for marching_cubes, got shape {current_values.shape}")
            
            # Downsample the data for faster marching cubes
            downsample_factor = 2
            downsampled_values = current_values[::downsample_factor, ::downsample_factor, ::downsample_factor]
            
            verts, faces, _, _ = marching_cubes(
                downsampled_values,
                level=level,
                spacing=[np.ndim(g) for g in grid.coordinate_vectors]
            )
            
            # Scale vertices back to original resolution
            verts = verts * downsample_factor
            
            # Transform vertices from grid coordinates to data coordinates
            verts = np.array([
                np.interp(verts[:, 0], np.arange(len(grid.coordinate_vectors[0])), grid.coordinate_vectors[0]),
                np.interp(verts[:, 1], np.arange(len(grid.coordinate_vectors[1])), grid.coordinate_vectors[1]),
                np.interp(verts[:, 2], np.arange(len(grid.coordinate_vectors[2])), grid.coordinate_vectors[2])
            ]).T
            
            # Create mesh with simplified coloring
            mesh = Poly3DCollection(verts[faces], alpha=opacity)
            
            # Get the signed distance values at the vertices
            vertex_values = current_values[
                np.clip(np.floor(verts[:, 0]).astype(int), 0, len(grid.coordinate_vectors[0])-1),
                np.clip(np.floor(verts[:, 1]).astype(int), 0, len(grid.coordinate_vectors[1])-1),
                np.clip(np.floor(verts[:, 2]).astype(int), 0, len(grid.coordinate_vectors[2])-1)
            ]
            
            # Add subtle texture to the surface
            noise = np.random.normal(0, 0.1, len(vertex_values))
            vertex_values = vertex_values + noise
            
            # Color based on the value, with red for the boundary
            face_colors = np.array(['red' if abs(v - level) < 0.01 else ('darkred' if v < level else 'darkgreen') for v in vertex_values])
            mesh.set_facecolor(face_colors)
            
            # Add edge color for texture
            mesh.set_edgecolor('black')
            mesh.set_linewidth(0.1)
            
            ax.add_collection3d(mesh)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('θ (rad)')
        ax.set_title(f'Backward Reachable Tube (BRT) at t={times[i]:.2f}s')
        
        # Set axis limits
        ax.set_xlim([grid.coordinate_vectors[0][0], grid.coordinate_vectors[0][-1]])
        ax.set_ylim([grid.coordinate_vectors[1][0], grid.coordinate_vectors[1][-1]])
        ax.set_zlim([grid.coordinate_vectors[2][0], grid.coordinate_vectors[2][-1]])
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add grid lines for better depth perception
        ax.grid(True)
    
    # Create animation
    anim = FuncAnimation(
        fig,
        plot_frame,
        frames=len(times),
        interval=interval,
        blit=False
    )
    
    # Save animation
    writer = PillowWriter(fps=10)
    with tqdm(total=len(times)) as pbar:
        anim.save(
            save_path,
            writer=writer,
            dpi=100,
            progress_callback=lambda i, n: pbar.update(1)
        )
    
    plt.close()

def plot_value_and_zero_level(values, grid, ax=None, figsize=(10, 8), title='Value Function'):
    """
    Plot the value function and its zero level set.
    
    Args:
        values: ndarray with shape [
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: Grid object containing coordinate_vectors
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    # Move values to CPU if they're on GPU
    values = jax.device_get(values)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create a darker version of RdYlGn
    colors = [
        (0.0, '#8B0000'),    # Dark red for unsafe states (BRT)
        (0.3, '#CD5C5C'),    # Indian red
        (0.499, '#FFB6C1'),  # Light pink
        (0.5, '#8B0000'),    # Dark red for boundary
        (0.501, '#E0FFF0'),  # Light cyan
        (0.7, '#32CD32'),    # Lime green
        (1.0, '#006400')     # Dark green for safe states
    ]
    custom_cmap = LinearSegmentedColormap.from_list('RdYlGn_dark', colors)
    
    # Create interpolator for smooth visualization
    interpolator = RegularGridInterpolator(
        grid.coordinate_vectors,
        values,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    # Create fine grid for visualization
    fine_grid = [
        np.linspace(coord[0], coord[-1], num=201) 
        for coord in grid.coordinate_vectors
    ]
    mesh = np.meshgrid(*fine_grid, indexing='ij')
    points = np.stack(mesh, axis=-1)
    
    # Interpolate values on fine grid
    fine_values = interpolator(points)
    
    # Plot value function
    vmin, vmax = np.nanmin(fine_values), np.nanmax(fine_values)
    im = ax.pcolormesh(
        fine_grid[0],
        fine_grid[1],
        fine_values.T,
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Value V(X,t)')
    
    # Plot zero level set
    cs = ax.contour(
        fine_grid[0],
        fine_grid[1],
        fine_values.T,
        levels=[0],
        colors='red',
        linewidths=2
    )
    
    ax.set_title(title)
    ax.set_xlabel('State Dimension 1')
    ax.set_ylabel('State Dimension 2')
    
    return fig, ax

def plot_environment(ax, env):
    """Plot the environment with obstacles.
    
    Args:
        ax: Matplotlib axes
        env: Environment object
    """
    for obs in env.obstacles:
        obs.plot(ax)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

def plot_signed_distances(ax, env, n_points=101):
    """Plot signed distances in the environment.
    
    Args:
        ax: Matplotlib axes
        env: Environment object
        n_points: Number of points for the grid
    """
    # Convert to numpy arrays for plotting
    x = np.linspace(0, env.width, n_points)
    y = np.linspace(0, env.height, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Get signed distances from the environment
    signed_distances = env.get_signed_distances(X, Y)
    
    # Convert to numpy for plotting
    X_np = np.array(X)
    Y_np = np.array(Y)
    signed_distances_np = np.array(signed_distances)
    
    # Create a custom colormap that makes red start at 0
    colors = [
        (0.0, 'red'),      # Most negative (darkest red)
        (0.4, 'lightcoral'),  # Less negative (lighter red)
        (0.499, 'mistyrose'),  # Just negative (lightest red)
        (0.501, 'honeydew'),   # Just positive (lightest green)
        (0.6, 'lightgreen'),   # Less positive (lighter green)
        (1.0, 'darkgreen')     # Most positive (darkest green)
    ]
    cmap = LinearSegmentedColormap.from_list('custom_rdylgn', colors)
    
    # Set levels to have more resolution around 0
    levels = np.linspace(-2, 2, n_points)
    
    # Plot with the new colormap and normalization
    contour = ax.contourf(
        X_np, Y_np, signed_distances_np,
        levels=levels,
        cmap=cmap,
        norm=plt.Normalize(-2, 2),
        extend='both'
    )
    ax.contour(X_np, Y_np, signed_distances_np, levels=[0], colors='k', linewidths=2)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Signed Distance to Obstacle l(X)')
    cbar.ax.axhline(0, color='black', linewidth=2)

def save_environment_plot(env, save_path, n_points=101):
    """Save a plot of the environment with signed distances.
    
    Args:
        env: Environment object
        save_path: Path to save the plot
        n_points: Number of points for the grid
    """
    fig, ax = plt.subplots()
    plot_signed_distances(ax, env, n_points)
    plot_environment(ax, env)
    plt.savefig(save_path)
    plt.close()
