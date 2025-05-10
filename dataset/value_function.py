"""Module for computing the value function.

This module handles the computation of the value function for the airplane obstacle
environment using Hamilton-Jacobi reachability analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
from tqdm import tqdm
from .config import CONVERGENCE_THRESHOLD, MAX_TIME

# Enable JIT compilation for better GPU performance
@jax.jit
def check_convergence(values):
    """Check convergence of value function computation.
    
    Args:
        values: Value function array with shape [time, x, y, theta]
        
    Returns:
        tuple: (max_change, converged) where:
            - max_change: Maximum change between consecutive time steps
            - converged: Boolean indicating if convergence was achieved
    """
    last_two_steps = values[-2:]
    max_change = jnp.max(jnp.abs(last_two_steps[1] - last_two_steps[0]))
    return max_change, max_change < CONVERGENCE_THRESHOLD

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
    # Print GPU memory info
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
        print(f"\nGPU Memory Info (before computation):\n{nvidia_smi.decode()}")
    except:
        print("Could not get GPU memory info")

    # Extract x, y coordinates from the grid states
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    
    # Get signed distances for the x, y coordinates
    failure_set = env.get_signed_distances(x, y)

    # Create solver settings
    solver_settings = hj.SolverSettings.with_accuracy(
        'very_high',
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )
    
    current_times = times
    values = None
    converged = False
    final_time = None
    
    try:
        while not converged and current_times[-1] > max_time:  # Check the last time point
            # Solve for current time horizon
            print(f"\nComputing value function with time horizon: {current_times[-1]:.2f}")
            print(f"Grid shape: {grid.states.shape}")
            print(f"Time points: {len(current_times)}")
            values = hj.solve(solver_settings, dynamics, grid, current_times, failure_set, progress_bar=True)
            
            # Check convergence using JIT-compiled function
            max_change, converged = check_convergence(values)
            
            if converged:
                final_time = float(current_times[-1])
                print(f"Converged with max change: {max_change:.2e} at time {final_time}")
                # Only keep the last two timesteps
                values = values[-2:]
            else:
                # Extend time horizon by doubling it and double the number of points
                new_end_time = current_times[-1] * 2  # Double the end time (more negative)
                new_num_points = len(current_times) * 2  # Double the number of points
                new_times = jnp.linspace(0, new_end_time, new_num_points)
                current_times = new_times
    finally:
        # Clean up JAX resources
        jax.clear_caches()
    
    return values, converged, final_time 