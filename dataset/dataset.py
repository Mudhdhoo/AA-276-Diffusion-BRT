"""Main module for dataset generation.

This module handles the generation of dataset samples for the airplane obstacle environment.
It manages result logging and sequential processing of samples.
"""

import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import jax
import jax.numpy as jnp
import numpy as np
import csv
import time
from datetime import datetime
import hj_reachability as hj
from utils.visualization import save_environment_plot
from tqdm import tqdm

from dataset.config import (
    N_POINTS, T, DEFAULT_NUM_OBSTACLES,
    OUTPUT_DIR, SAMPLE_DIR_PREFIX, ENVIRONMENT_PLOT_NAME,
    VALUE_FUNCTION_NAME, RESULTS_CSV_NAME, NUM_SAMPLES,
    GLOBAL_SEED, ENVIRONMENT_GRID_NAME
)

# Define CSV columns with clearer names
CSV_COLUMNS = [
    'sample_id',
    'seed',
    'converged',
    'computation_time_seconds',  # Time taken for script to run until convergence
    'simulation_time_horizon',   # Actual time horizon in HJ reachability simulation
    'num_obstacles',            # Number of obstacles in the environment
    'env_grid_path',           # Path to the environment grid file
    'value_function_path',     # Path to the value function file
    'env_plot_path',          # Path to the environment visualization
    'timestamp'
]

from dataset.environment import AirplaneObstacleEnvironment
from dataset.dynamics import AirplaneDynamics
from dataset.value_function import get_V

def get_timestamp_dir():
    """Create a directory name based on current timestamp.
    
    Returns:
        str: Directory name in format 'DD_Mon_YYYY_HH_MM'
    """
    return datetime.now().strftime("%d_%b_%Y_%H_%M")

def write_result_to_csv(result, csv_path):
    """Write a single result to a CSV file.
    
    Args:
        result: Dictionary containing the result data
        csv_path: Path to the CSV file
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([
            result['sample_id'],
            result['seed'],
            result['converged'],
            result['computation_time_seconds'],
            result.get('simulation_time_horizon', None),
            result['num_obstacles'],
            result['env_grid_path'],
            result.get('value_function_path', None),  # None if not converged
            result['env_plot_path'],
            result['timestamp']
        ])

def create_environment_grid(env, grid):
    """Create a binary grid representation of the environment.
    
    Args:
        env: Environment object
        grid: Grid object for discretization
        
    Returns:
        jax.numpy.ndarray: Binary grid where 1 represents obstacles and 0 represents free space
    """
    # Create meshgrid from coordinate vectors
    x, y = jnp.meshgrid(grid.coordinate_vectors[0], grid.coordinate_vectors[1], indexing='ij')
        
    signed_distances = env.get_signed_distances(x, y)
    
    # Convert signed distances to binary grid (1 for obstacles, 0 for free space)
    # Negative values indicate inside obstacles
    binary_grid = jnp.where(signed_distances <= 0, 1.0, 0.0)
        
    assert binary_grid.shape == (N_POINTS, N_POINTS)
        
    return binary_grid

def process_sample(sample_id, output_dir, key):
    """Process a single sample with given ID and key.
    
    Args:
        sample_id: ID of the sample to process
        output_dir: Directory to save outputs
        key: JAX random key for this sample
        
    Returns:
        Dictionary containing sample results
    """
    print(f"\nProcessing sample {sample_id + 1}/{NUM_SAMPLES}")
    
    # Create sample directory
    sample_dir = os.path.join(output_dir, f'{SAMPLE_DIR_PREFIX}{sample_id:03d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create and setup environment
    env = AirplaneObstacleEnvironment()
    # Randomly add -1 to 1 number of obstacles
    num_obstacles = DEFAULT_NUM_OBSTACLES + jax.random.randint(key, (1,), -1, 2)
    env.set_random_obstacles(num_obstacles, key=key)
        
    # Define file paths
    #env_plot_path = os.path.join(sample_dir, ENVIRONMENT_PLOT_NAME)
    env_grid_path = os.path.join(sample_dir, ENVIRONMENT_GRID_NAME)
    value_function_path = os.path.join(sample_dir, VALUE_FUNCTION_NAME)
    
    # Save environment plot
    #save_environment_plot(env, env_plot_path)
    
    # Setup grid and dynamics
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(
            jnp.array([0.0, 0.0, -jnp.pi]),
            jnp.array([env.width, env.height, jnp.pi])
        ),  
        shape=(N_POINTS, N_POINTS, N_POINTS)
    )
    
    # Create and save environment grid
    env_grid = create_environment_grid(env, grid)
    # Move to CPU only when saving
    np.save(env_grid_path, jax.device_get(env_grid))
    
    initial_times = jnp.linspace(0, T, N_POINTS)
    dynamics = AirplaneDynamics()
    
    # Compute value function and measure time
    start_time = time.time()
    V, converged, final_time = get_V(env, dynamics, grid, initial_times)
    computation_time = time.time() - start_time
    
    result = {
        'sample_id': sample_id,
        'seed': int(jax.random.fold_in(key, 0)[0]),
        'converged': converged,
        'computation_time_seconds': computation_time,
        'simulation_time_horizon': float(final_time) if converged else None,
        'num_obstacles': int(num_obstacles[0]),  # Convert from JAX array to int
        'env_grid_path': env_grid_path,
        'env_plot_path': None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Write result to CSV file
    write_result_to_csv(result, os.path.join(output_dir, RESULTS_CSV_NAME))
    
    if not converged or V is None:
        print(f"Sample {sample_id + 1} did not converge")
        return result
    
    # Save value function data (only contains last two timesteps when converged)
    # Move to CPU only when saving
    np.save(value_function_path, jax.device_get(V))
    result['value_function_path'] = value_function_path
    
    return result

def main():
    """Main function to generate the dataset."""
    # Create output directory with timestamp
    output_dir = os.path.join(OUTPUT_DIR, get_timestamp_dir())
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize global key
    global_key = jax.random.PRNGKey(GLOBAL_SEED)
    
    # Create a list of keys for each sample
    keys = []
    current_key = global_key
    for _ in range(NUM_SAMPLES):
        current_key, subkey = jax.random.split(current_key)
        keys.append(subkey)
    
    # Process samples sequentially
    results = []
    for sample_id in tqdm(range(NUM_SAMPLES), desc="Processing samples", unit="sample"):
        result = process_sample(sample_id, output_dir, keys[sample_id])
        results.append(result)

if __name__ == "__main__":
    main()