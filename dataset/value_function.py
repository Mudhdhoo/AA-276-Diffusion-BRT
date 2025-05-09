"""Module for computing the value function.

This module handles the computation of the value function for the airplane obstacle
environment using Hamilton-Jacobi reachability analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
from tqdm import tqdm
import contextlib
import io
import sys
import os
from .config import CONVERGENCE_THRESHOLD, MAX_TIME

@contextlib.contextmanager
def custom_progress_bar(desc):
    """Context manager to create a custom progress bar that works with multiprocessing.
    
    Args:
        desc: Description to display in the progress bar
        
    Yields:
        tqdm: Progress bar instance
    """
    pbar = tqdm(total=100, desc=desc, leave=False)
    try:
        yield pbar
    finally:
        pbar.close()

def get_V(env, dynamics, grid, times, convergence_threshold=CONVERGENCE_THRESHOLD, max_time=MAX_TIME):
    """Compute the value function with convergence checking.
    
    This function computes the value function for the given environment and dynamics,
    checking for convergence at each time step. If convergence is not achieved within
    the initial time horizon, it extends the horizon until convergence or max_time is reached.
    
    Args:
        env: Environment object containing obstacles
        dynamics: Dynamics object describing the system behavior
        grid: Grid object for discretization
        times: Initial time points for computation
        convergence_threshold: Maximum allowed change in value function between consecutive time steps
        max_time: Maximum time horizon to try before giving up
        
    Returns:
        tuple: (values, converged, final_time) where:
            - values: Value function array with shape [time, x, y, theta]
            - converged: Boolean indicating if convergence was achieved
            - final_time: The final time horizon used (None if not converged)
    """
    try:
        print("Starting value function computation...")
        
        # Extract x, y coordinates from the grid states
        print("Extracting grid coordinates...")
        x = grid.states[..., 0]
        y = grid.states[..., 1]
        print(f"Grid shape: {grid.shape}")
        
        # Get signed distances for the x, y coordinates
        print("Computing signed distances...")
        failure_set = env.get_signed_distances(x, y)
        print("Signed distances computed successfully")
        print(f"Failure set shape: {failure_set.shape}")

        # Create solver settings
        print("Creating solver settings...")
        solver_settings = hj.SolverSettings.with_accuracy(
            'very_high',
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )
        print("Solver settings created successfully")
        
        current_times = times
        values = None
        converged = False
        final_time = None
        
        iteration = 0
        while not converged and current_times[-1] > max_time:  # Check the last time point
            print(f"\nStarting iteration {iteration}")
            print(f"Current time horizon: {current_times[-1]}")
            print(f"Number of time points: {len(current_times)}")
            
            try:
                # Solve for current time horizon
                print("Solving value function...")
                try:
                    # Clear JAX cache before solve
                    jax.clear_caches()
                    
                    # Move failure set to device if not already there
                    failure_set_device = jax.device_put(failure_set)
                    
                    # Solve with explicit device placement and memory management
                    with jax.default_device(jax.devices()[0]):  # Explicitly use first GPU
                        values = hj.solve(
                            solver_settings, 
                            dynamics, 
                            grid, 
                            current_times, 
                            failure_set_device, 
                            progress_bar=False
                        )
                    
                    # Move result back to host immediately
                    values = jax.device_get(values)
                    print("Value function solved successfully")
                    print(f"Value function shape: {values.shape}")
                except Exception as solve_error:
                    print(f"Error during solve: {str(solve_error)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Keep values on CPU for convergence check
                values_device = values
                
                # Check convergence by comparing only the last two time steps
                print("Checking convergence...")
                # Move only the last two time steps to CPU for comparison
                last_two_steps = values_device[-2:]
                max_change = np.max(np.abs(last_two_steps[1] - last_two_steps[0]))
                print(f"Max change: {max_change:.2e}")
                
                if max_change < convergence_threshold:
                    converged = True
                    final_time = float(current_times[-1])
                    print(f"Converged with max change: {max_change:.2e} at time {final_time}")
                else:
                    # Extend time horizon by doubling it and double the number of points
                    print("Not converged, extending time horizon...")
                    new_end_time = current_times[-1] * 2  # Double the end time (more negative)
                    new_num_points = len(current_times) * 2  # Double the number of points
                    new_times = jnp.linspace(0, new_end_time, new_num_points)
                    current_times = new_times
                    print(f"New time horizon: {new_end_time}")
                    print(f"New number of time points: {len(new_times)}")
                
            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            iteration += 1
        
        return values, converged, final_time
        
    except Exception as e:
        print(f"Error in get_V: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 