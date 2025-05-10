"""Module containing visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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