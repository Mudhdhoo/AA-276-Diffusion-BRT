import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualize_random_grids(directory_path, num_samples=15):
    """Visualize random environment grids from the dataset."""
    directory = Path(directory_path)
    
    # Find all environment grid files
    grid_files = list(directory.glob('**/environment_grid.npy'))
    
    if len(grid_files) < num_samples:
        print(f"Warning: Only found {len(grid_files)} files, but requested {num_samples} samples")
        num_samples = len(grid_files)
    
    # Randomly sample files
    selected_files = random.sample(grid_files, num_samples)
    
    # Create a figure with a grid of subplots
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()
    
    # Plot each grid
    for idx, grid_file in enumerate(selected_files):
        try:
            # Load and plot the grid
            grid = np.load(grid_file)
            im = axes[idx].imshow(grid, cmap='viridis')
            axes[idx].set_title(f'Sample {idx+1}')
            axes[idx].axis('off')
            
            # Add colorbar to the last subplot
            if idx == num_samples - 1:
                plt.colorbar(im, ax=axes[idx], label='Value')
                
        except Exception as e:
            print(f"Error processing {grid_file}: {str(e)}")
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Random Environment Grid Samples', y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    output_path = directory / 'environment_grid_samples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    target_dir = "/Users/malte/AA-276-Diffusion-BRT/dataset/outputs/10_May_2025_05_33"
    visualize_random_grids(target_dir) 