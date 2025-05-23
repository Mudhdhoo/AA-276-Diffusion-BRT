import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_value_function_batch_3d(batch_tensor, time=None, threshold=0, x_interval=None, y_interval=None, z_interval=None, 
                                 grid_cols=4, figsize=(16, 8), save_path=None):
    """
    Plot a batch of 3D tensors in a grid layout.
    batch_tensor: (B, C, D, H, W) or (B, D, H, W)
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(batch_tensor):
        batch_tensor_np = batch_tensor.detach().cpu().numpy()
    else:
        batch_tensor_np = batch_tensor
    if batch_tensor_np.ndim == 5:
        # (B, C, D, H, W) -> (B, D, H, W) for C=1
        if batch_tensor_np.shape[1] == 1:
            batch_tensor_np = batch_tensor_np[:, 0]
        else:
            raise ValueError("Only single-channel 3D tensors are supported for plotting.")
    batch_size = batch_tensor_np.shape[0]
    grid_rows = (batch_size + grid_cols - 1) // grid_cols
    fig = plt.figure(figsize=figsize)
    for i in range(batch_size):
        tensor = batch_tensor_np[i]
        ax = fig.add_subplot(grid_rows, grid_cols, i+1, projection='3d')
        x_indices, y_indices, z_indices = np.where(tensor < threshold)
        values = tensor[x_indices, y_indices, z_indices]
        x_coords = x_indices.copy()
        y_coords = y_indices.copy()
        z_coords = z_indices.copy()
        if x_interval is not None:
            x_min, x_max = x_interval
            x_size = tensor.shape[0] - 1
            x_coords = x_min + (x_indices / x_size) * (x_max - x_min)
        if y_interval is not None:
            y_min, y_max = y_interval
            y_size = tensor.shape[1] - 1
            y_coords = y_min + (y_indices / y_size) * (y_max - y_min)
        if z_interval is not None:
            z_min, z_max = z_interval
            z_size = tensor.shape[2] - 1
            z_coords = z_min + (z_indices / z_size) * (z_max - z_min)
        if len(values) > 0:
            norm = plt.Normalize(values.min(), threshold)
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                                 c=values, cmap='Spectral', alpha=0.8, 
                                 s=10, edgecolor='none')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            if time is not None:
                ax.set_title(f't = {time[i]+1}')
            else:
                ax.set_title(f'Sample {i+1}')
        else:
            ax.text(0.5, 0.5, 0.5, "No values < threshold", 
                    horizontalalignment='center', verticalalignment='center')
    has_values_below_threshold = (batch_tensor_np < threshold).any()
    if has_values_below_threshold:
        cbar_ax = fig.add_axes([0.97, 0.30, 0.02, 0.4])
        all_values = batch_tensor_np[batch_tensor_np < threshold]
        norm = plt.Normalize(all_values.min(), threshold)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Spectral'), cax=cbar_ax)
        cbar.set_label('Value')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, right=0.9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show() 