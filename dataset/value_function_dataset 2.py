import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class ValueFunctionDataset(Dataset):
    def __init__(self, results_path):
        """
        Create a PyTorch dataset from the converged value function and environment grid.
        
        Args:
            results_path (str): Path to the results.csv file containing paths to value functions
                              and environment grids
        """
        # Get the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(results_path), "../.."))
        
        # Read the results file using pandas
        results = pd.read_csv(results_path)
        
        # Filter for converged samples
        converged_mask = results['converged'] == True
        self.environment_grid_paths = results['env_grid_path'][converged_mask].values
        
        # Convert absolute paths to relative paths from project root
        def make_relative(path):
            # Extract the part after 'outputs/'
            parts = path.split('outputs/')
            if len(parts) > 1:
                return os.path.join(self.project_root, 'outputs', parts[1])
            return path
            
        self.environment_grid_paths = [make_relative(path) for path in self.environment_grid_paths]
        
        # Create value function paths by replacing 'environment_grid.npy' with 'value_function.npy'
        self.value_function_paths = [
            path.replace('environment_grid.npy', 'value_function.npy')
            for path in self.environment_grid_paths
        ]
        
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