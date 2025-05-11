import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ValueFunctionDataset(Dataset):
    def __init__(self, results_path):
        """
        Create a PyTorch dataset from the converged value function and environment grid.
        
        Args:
            results_path (str): Path to the results.csv file containing paths to value functions
                              and environment grids
        """
        # Get the directory containing the results file
        self.results_dir = os.path.dirname(os.path.abspath(results_path))
        
        # Read the results file
        results = np.genfromtxt(results_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        
        # Filter for converged samples
        converged_mask = results['converged'] == True
        self.value_function_paths = results['value_function_path'][converged_mask]
        self.environment_grid_paths = results['env_grid_path'][converged_mask]
        
        # Convert paths to be relative to results directory
        self.value_function_paths = [os.path.join(self.results_dir, os.path.basename(path)) for path in self.value_function_paths]
        self.environment_grid_paths = [os.path.join(self.results_dir, os.path.basename(path)) for path in self.environment_grid_paths]
        
        # Load the first sample to get dimensions
        first_value_function = np.load(self.value_function_paths[0])
        first_env_grid = np.load(self.environment_grid_paths[0])
        
        # Store dimensions
        self.time_steps = first_value_function.shape[0]
        self.grid_size = first_value_function.shape[1:]
        self.env_grid_size = first_env_grid.shape
        
    def __len__(self):
        return len(self.value_function_paths)
    
    def __getitem__(self, idx):
        # Load value function and environment grid
        value_function = np.load(self.value_function_paths[idx])
        environment_grid = np.load(self.environment_grid_paths[idx])
        
        # Convert to PyTorch tensors
        value_function = torch.from_numpy(value_function).float()
        environment_grid = torch.from_numpy(environment_grid).float()
        
        # Return the last time step of the value function (converged solution)
        # and the environment grid
        return {
            'value_function': value_function[-1],  # Shape: (101, 101, 101)
            'environment_grid': environment_grid,  # Shape: (101, 101)
            'value_function_path': self.value_function_paths[idx],
            'environment_grid_path': self.environment_grid_paths[idx]
        }

def create_value_function_dataset(results_path):
    """
    Helper function to create a ValueFunctionDataset instance.
    
    Args:
        results_path (str): Path to the results.csv file
        
    Returns:
        ValueFunctionDataset: A PyTorch dataset containing the value function and environment grid
    """
    return ValueFunctionDataset(results_path) 