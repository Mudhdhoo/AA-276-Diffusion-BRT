"""Main module for dataset generation.

This module handles the generation of dataset samples for the airplane obstacle environment.
It manages result logging and sequential processing.
"""

import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
import csv
import time
from datetime import datetime
import hj_reachability as hj
from utils.visualization import save_environment_plot
from tqdm import tqdm
import tempfile
from pathlib import Path

from dataset.config import (
    N_POINTS, T, DEFAULT_NUM_OBSTACLES,
    OUTPUT_DIR, SAMPLE_DIR_PREFIX, ENVIRONMENT_PLOT_NAME,
    VALUE_FUNCTION_NAME, RESULTS_CSV_NAME, NUM_SAMPLES,
    GLOBAL_SEED, CSV_COLUMNS, ENVIRONMENT_GRID_NAME
)
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
            result['convergence_time'],
            result.get('final_time_horizon', None),
            result['timestamp']
        ])

def create_environment_grid(env, grid):
    """Create a binary grid representation of the environment.
    
    Args:
        env: Environment object
        grid: Grid object for discretization
        
    Returns:
        numpy.ndarray: Binary grid where 1 represents obstacles and 0 represents free space
    """
    # Extract x, y coordinates from the grid states
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    
    # Get signed distances for the x, y coordinates
    signed_distances = env.get_signed_distances(x, y)
    
    # Convert signed distances to binary grid (1 for obstacles, 0 for free space)
    # Negative values indicate inside obstacles
    binary_grid = (signed_distances <= 0).astype(np.float32)
    
    return binary_grid

def process_sample(args):
    """Process a single sample with given ID and key.
    
    Args:
        args: Tuple containing (sample_id, output_dir, global_key, temp_dir)
        
    Returns:
        Dictionary containing sample results
    """
    sample_id, output_dir, global_key, temp_dir = args
    
    # Create sample directory
    sample_dir = os.path.join(output_dir, f'{SAMPLE_DIR_PREFIX}{sample_id:03d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get new subkey for this sample
    global_key, subkey = jax.random.split(global_key)
    
    # Create and setup environment
    env = AirplaneObstacleEnvironment()
    env.set_random_obstacles(DEFAULT_NUM_OBSTACLES, key=subkey)
    
    # Save environment plot
    save_environment_plot(env, os.path.join(sample_dir, ENVIRONMENT_PLOT_NAME))
    
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
    np.save(os.path.join(sample_dir, ENVIRONMENT_GRID_NAME), env_grid)
    
    initial_times = jnp.linspace(0, T, N_POINTS)
    dynamics = AirplaneDynamics()
    
    # Compute value function and measure time
    start_time = time.time()
    V, converged, final_time = get_V(env, dynamics, grid, initial_times)
    convergence_time = time.time() - start_time
    
    result = {
        'sample_id': sample_id,
        'seed': int(jax.random.fold_in(subkey, 0)[0]),
        'converged': converged,
        'convergence_time': convergence_time,
        'final_time_horizon': float(final_time) if converged else None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Write result to temporary CSV file
    temp_csv = os.path.join(temp_dir, f'result_{sample_id}.csv')
    write_result_to_csv(result, temp_csv)
    
    if not converged or V is None:
        return result
    
    # Save value function data
    np.save(os.path.join(sample_dir, VALUE_FUNCTION_NAME), V)
    
    return result

def combine_csv_files(temp_dir, output_csv):
    """Combine all temporary CSV files into the final output file.
    
    Args:
        temp_dir: Directory containing temporary CSV files
        output_csv: Path to the final output CSV file
    """
    # Write header
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(CSV_COLUMNS)
    
    # Combine all temporary files
    for temp_file in sorted(os.listdir(temp_dir)):
        if temp_file.endswith('.csv'):
            with open(os.path.join(temp_dir, temp_file), 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                with open(output_csv, 'a', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(reader)

def main():
    """Main function to generate the dataset."""
    # Create output directory with timestamp
    output_dir = os.path.join(OUTPUT_DIR, get_timestamp_dir())
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for CSV files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize global key
        global_key = jax.random.PRNGKey(GLOBAL_SEED)
        
        # Create a list of keys for each sample
        keys = []
        current_key = global_key
        for _ in range(NUM_SAMPLES):
            current_key, subkey = jax.random.split(current_key)
            keys.append(subkey)
        
        # Process samples sequentially
        print("Processing samples sequentially")
        results = []
        pbar = tqdm(total=NUM_SAMPLES, desc="Processing samples", unit="sample")
        for i, key in enumerate(keys):
            result = process_sample((i, output_dir, key, temp_dir))
            if result is not None:
                results.append(result)
            pbar.update(1)
            pbar.set_postfix({
                'converged': result['converged'],
                'time_horizon': result.get('final_time_horizon', 'N/A')
            })
        pbar.close()
        
        # Combine all temporary CSV files into the final output
        combine_csv_files(temp_dir, os.path.join(output_dir, RESULTS_CSV_NAME))
    
    print(f"\nCompleted processing {len(results)} samples")

if __name__ == "__main__":
    main()