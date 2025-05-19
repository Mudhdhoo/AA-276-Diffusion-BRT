import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def plot_value_function_batch(batch_tensor, time=None, threshold=0, x_interval=None, y_interval=None, z_interval=None, 
                             grid_cols=6, figsize=(12, 6), save_path=None):
    """
    Plot a batch of 3D tensors in a grid layout.
    
    Parameters:
    -----------
    batch_tensor : torch.Tensor
        Batch of 3D tensors with shape (B, C, H, W) where B is batch size
    threshold : float
        Threshold value for displaying points (values < threshold are plotted)
    x_interval, y_interval, z_interval : tuple of (min, max) or None
        Custom intervals for mapping indices
    grid_cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path instead of being displayed
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(batch_tensor):
        batch_tensor_np = batch_tensor.detach().cpu().numpy()
    else:
        batch_tensor_np = batch_tensor
        
    batch_size = batch_tensor_np.shape[0]
    grid_rows = (batch_size + grid_cols - 1) // grid_cols  # Calculate needed rows
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Loop through each tensor in the batch
    for i in range(batch_size):
        tensor = batch_tensor_np[i]
        
        # Create 3D subplot
        ax = fig.add_subplot(grid_rows, grid_cols, i+1, projection='3d')
        
        # Get coordinates where values are negative
        x_indices, y_indices, z_indices = np.where(tensor < threshold)
        values = tensor[x_indices, y_indices, z_indices]
        
        # Map indices to custom intervals if provided
        x_coords = x_indices.copy()
        y_coords = y_indices.copy()
        z_coords = z_indices.copy()
        
        if x_interval is not None:
            # Transform x indices to the specified interval
            x_min, x_max = x_interval
            x_size = tensor.shape[0] - 1  # Max index value
            x_coords = x_min + (x_indices / x_size) * (x_max - x_min)
        
        if y_interval is not None:
            # Transform y indices to the specified interval
            y_min, y_max = y_interval
            y_size = tensor.shape[1] - 1  # Max index value
            y_coords = y_min + (y_indices / y_size) * (y_max - y_min)
        
        if z_interval is not None:
            # Transform z indices to the specified interval
            z_min, z_max = z_interval
            z_size = tensor.shape[2] - 1  # Max index value
            z_coords = z_min + (z_indices / z_size) * (z_max - z_min)
        
        # Normalize the values for coloring
        if len(values) > 0:
            norm = plt.Normalize(values.min(), threshold)
            
            # Plot scatter points for negative values using the mapped coordinates
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                       c=values, cmap='Spectral', alpha=0.8, 
                       s=25, edgecolor='none')  # Smaller point size for grid plots
            
            # Set labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('$\\theta$')  

            if time is not None:
                ax.set_title(f't = {time[i]+1}')
            else:
                ax.set_title(f'Sample{i+1}')
            
        else:
            ax.text(0.5, 0.5, 0.5, "No values < threshold", 
                   horizontalalignment='center', verticalalignment='center')
    
    # Add a single colorbar for the entire figure - using PyTorch compatible check
    has_values_below_threshold = (batch_tensor_np < threshold).any()
    if has_values_below_threshold:
        cbar_ax = fig.add_axes([0.97, 0.30, 0.02, 0.4])  # Position for colorbar
        # Get all values below threshold
        all_values = batch_tensor_np[batch_tensor_np < threshold]
        norm = plt.Normalize(all_values.min(), threshold)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Spectral'), cax=cbar_ax)
        cbar.set_label('Value')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, right=0.9)  # Adjust for colorbar
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
