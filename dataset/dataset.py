import jax
import jax.numpy as jnp
import hj_reachability as hj
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np  # Only for plotting
import sys
import os
import csv
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.value_visualization import plot_3d_value_evolution

N_POINTS = 101
T = -5

class Obstacle(ABC):
    """ Abstract base class for obstacles. """
    @abstractmethod
    def get_distance(self, X, Y):
        """Get the signed distance to the closest point on the obstacle.
        
        Args:
            X: x-coordinate(s), can be scalar or array-like
            Y: y-coordinate(s), must match shape of X
            
        Returns:
            Signed distance(s) with same shape as X and Y
        """
        pass

    @abstractmethod
    def plot(self, ax):
        """Plot the obstacle."""
        pass


class RectangleObstacle(Obstacle):
    """ Axis aligned rectangle obstacle. """
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        
    
    def get_distance(self, X, Y):
        """Get the signed distance to the closest point on the rectangle.
        
        Args:
            X: x-coordinate(s), can be scalar or array-like
            Y: y-coordinate(s), must match shape of X
            
        Returns:
            Signed distance(s) to the rectangle. Negative values indicate points inside the rectangle.
        """
        # Convert inputs to JAX arrays
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        
        # Calculate distance to closest point on rectangle boundary
        dx = jnp.maximum(jnp.maximum(self.x1 - X, X - self.x2), 0.0)
        dy = jnp.maximum(jnp.maximum(self.y1 - Y, Y - self.y2), 0.0)
        outside_dist = jnp.sqrt(dx**2 + dy**2)
        
        # For points inside the rectangle, compute negative distance to nearest boundary
        inside_dx = jnp.minimum(X - self.x1, self.x2 - X)
        inside_dy = jnp.minimum(Y - self.y1, self.y2 - Y)
        inside_dist = -jnp.minimum(inside_dx, inside_dy)
        
        # Determine which points are inside the rectangle
        is_inside = ((self.x1 <= X) & (X <= self.x2) & 
                    (self.y1 <= Y) & (Y <= self.y2))
        
        return jnp.where(is_inside, inside_dist, outside_dist)
    
    def plot(self, ax):
        """Plot the obstacle."""
        rect = plt.Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1, 
                             edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        

class AirplaneDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    """ Airplane dynamics with control and disturbance. """
    def __init__(self, v0=0.4):
        self.v0 = v0
        self.control_space = hj.sets.Box(jnp.array([0.1, -0.5]), jnp.array([0.8, 0.5]))
        self.disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(
            control_space=self.control_space, 
            disturbance_space=self.disturbance_space, 
            control_mode='max', 
            disturbance_mode='min'
        )
    
    def open_loop_dynamics(self, state, time):
        """Open loop dynamics of the airplane."""
        x, y, theta = state
        return jnp.array([
            self.v0 * jnp.cos(theta),
            self.v0 * jnp.sin(theta),
            0.0
        ])
    
    def control_jacobian(self, state, time):
        """Control Jacobian of the airplane."""
        x, y, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],
            [jnp.sin(theta), 0.0],
            [0.0, 1.0]
        ])
    
    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian of the airplane."""
        return jnp.zeros((3, 1))
    
    def control_space(self):
        """Control space of the airplane."""
        return self.control_space

    def disturbance_space(self):
        """Disturbance space of the airplane."""
        return self.disturbance_space
            

class AirplaneObstacleEnvironment():
    def __init__(self, width=10, height=10, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else []

    def set_random_obstacles(self, num_obstacles=1, key=None):
        self.obstacles = []
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Define min and max sizes relative to environment
        min_size = 0.05 * min(self.width, self.height)
        max_size = 0.15 * min(self.width, self.height)
        
        while len(self.obstacles) < num_obstacles:
            # Generate new keys for this obstacle
            key, *subkeys = jax.random.split(key, 5)
            
            # Randomize width and height independently
            width = jax.random.uniform(subkeys[0], minval=min_size, maxval=max_size)
            height = jax.random.uniform(subkeys[1], minval=min_size, maxval=max_size)
            
            # Ensure the obstacle fits within the environment
            max_x = self.width - width
            max_y = self.height - height
            
            if max_x <= 0 or max_y <= 0:
                continue  # Skip if the obstacle is too large for the environment
                
            # Randomize position
            x1 = float(jax.random.uniform(subkeys[2], minval=0, maxval=max_x))
            y1 = float(jax.random.uniform(subkeys[3], minval=0, maxval=max_y))
            
            x2 = x1 + width
            y2 = y1 + height
            
            # Create and add the obstacle
            self.obstacles.append(RectangleObstacle(x1, y1, x2, y2))
    
    def get_signed_distances(self, X, Y):
        """
        Calculate signed distances for given grid points.
        
        Args:
            X: 2D array of x-coordinates
            Y: 2D array of y-coordinates (same shape as X)
            
        Returns:
            2D array of signed distances where:
            - Positive values: distance to nearest obstacle (outside)
            - Negative values: distance to nearest boundary (inside)
            - Zero: on obstacle boundary
        """
        if not self.obstacles:
            return jnp.full_like(X, jnp.inf)
            
        # Calculate distances to all obstacles
        all_distances = jnp.stack([obs.get_distance(X, Y) for obs in self.obstacles])
        
        # Find minimum distance at each point
        return jnp.min(all_distances, axis=0)


#########################################################################
def plot_environment(ax, env):
    for obs in env.obstacles:
        obs.plot(ax)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

def plot_signed_distances(ax, env):
    # Convert to numpy arrays for plotting
    x = jnp.linspace(0, env.width, N_POINTS)
    y = jnp.linspace(0, env.height, N_POINTS)
    X, Y = jnp.meshgrid(x, y)
    
    # Get signed distances from the environment
    signed_distances = env.get_signed_distances(X, Y)
    
    # Convert to numpy for plotting
    X_np = np.array(X)
    Y_np = np.array(Y)
    signed_distances_np = np.array(signed_distances)
    
    # Create a custom colormap that makes red start at 0
    from matplotlib.colors import LinearSegmentedColormap
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
    levels = np.linspace(-2, 2, N_POINTS)
    
    # Plot with the new colormap and normalization
    contour = ax.contourf(X_np, Y_np, signed_distances_np, levels=levels, cmap=cmap, norm=plt.Normalize(-2, 2), extend='both')
    ax.contour(X_np, Y_np, signed_distances_np, levels=[0], colors='k', linewidths=2)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Signed Distance to Obstacle l(X)')
    cbar.ax.axhline(0, color='black', linewidth=2)


def get_V(env, dynamics, grid, times, convergence_threshold=1e-3, max_time=-40):
    """
    Compute the value function with convergence checking.
    
    Args:
        env: Environment object
        dynamics: Dynamics object
        grid: Grid object
        times: Initial time points
        convergence_threshold: Maximum allowed change in value function between consecutive time steps
        max_time: Maximum time horizon to try before giving up
        
    Returns:
        values: Value function array with shape [time, x, y, theta]
        converged: Boolean indicating if convergence was achieved
    """
    # Extract x, y coordinates from the grid states
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    
    # Get signed distances for the x, y coordinates
    failure_set = env.get_signed_distances(x, y)

    solver_settings = hj.SolverSettings.with_accuracy(
        'very_high',
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )
    
    current_times = times
    values = None
    converged = False
    
    while not converged and current_times[-1] > max_time:  # Check the last time point
        # Solve for current time horizon
        values = hj.solve(solver_settings, dynamics, grid, current_times, failure_set)
        values = np.array(values)  # Convert to numpy array
        
        # Check convergence by comparing only the last two time steps
        max_change = np.max(np.abs(values[-1] - values[-2]))
        
        if max_change < convergence_threshold:
            converged = True
            print(f"Converged with max change: {max_change:.2e}")
        else:
            # Extend time horizon by doubling it and double the number of points
            new_end_time = current_times[-1] * 2  # Double the end time (more negative)
            new_num_points = len(current_times) * 2  # Double the number of points
            new_times = jnp.linspace(0, new_end_time, new_num_points)  # Go from 0 to negative time
            current_times = new_times
            print(f"Not converged. Max change: {max_change:.2e}. Extending time horizon to {current_times[-1]:.1f} with {len(current_times)} points")
    
    if not converged:
        print(f"Warning: Value function did not converge within time horizon {max_time}")
        return None, False
    
    # Ensure values has the correct shape [time, x, y, theta]
    if values.ndim != 4:
        raise ValueError(f"Expected 4D array from hj.solve, got shape {values.shape}")
    
    return values, True


def get_timestamp_dir():
    """Create a directory name based on current timestamp."""
    return datetime.now().strftime("%d_%b_%Y_%H_%M")

def save_environment_plot(env, save_path):
    """Save a plot of the environment with signed distances."""
    fig, ax = plt.subplots()
    plot_signed_distances(ax, env)
    plot_environment(ax, env)
    plt.savefig(save_path)
    plt.close()

def process_sample(sample_id, output_dir, global_key):
    """Process a single sample with given ID and key."""
    print(f"\nProcessing sample {sample_id + 1}/300")
    
    # Create sample directory
    sample_dir = os.path.join(output_dir, f'sample_{sample_id:03d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get new subkey for this sample
    global_key, subkey = jax.random.split(global_key)
    
    # Create and setup environment
    env = AirplaneObstacleEnvironment()
    env.set_random_obstacles(3, key=subkey)
    
    # Save environment plot
    save_environment_plot(env, os.path.join(sample_dir, 'environment.png'))
    
    # Setup grid and dynamics
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(
            jnp.array([0.0, 0.0, -jnp.pi]),
            jnp.array([env.width, env.height, jnp.pi])
        ),  
        shape=(N_POINTS, N_POINTS, N_POINTS)
    )
    initial_times = jnp.linspace(0, T, N_POINTS)
    dynamics = AirplaneDynamics()
    
    # Compute value function and measure time
    start_time = time.time()
    V, converged = get_V(env, dynamics, grid, initial_times)
    convergence_time = time.time() - start_time
    
    result = {
        'sample_id': sample_id,
        'seed': int(jax.random.fold_in(subkey, 0)[0]),
        'converged': converged,
        'convergence_time': convergence_time,
        'timestamp': datetime.now().isoformat()
    }
    
    if not converged or V is None:
        print(f"Sample {sample_id} did not converge. Skipping value function visualization.")
        return result
    
    # Create time points based on the shape of V
    # IS THIS WRONG?
    times = jnp.linspace(0, V.shape[0] * T / N_POINTS, V.shape[0])
    
    # Save value function visualization
    plot_3d_value_evolution(
        V, grid, times,
        save_path=os.path.join(sample_dir, 'value_evolution.gif'),
        level=0.0, opacity=0.3,
        elev=30, azim=45, interval=200, max_frames=20
    )
    
    # Save value function data
    np.save(os.path.join(sample_dir, 'value_function.npy'), V)
    
    print(f"Completed sample {sample_id + 1}")
    return result

def process_sample_wrapper(args):
    """Wrapper function to handle the arguments correctly for multiprocessing."""
    sample_id, key, output_dir = args
    return process_sample(sample_id, output_dir, key)

def main():
    # Create output directory with timestamp
    output_dir = os.path.join('outputs', get_timestamp_dir())
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV file for logging
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'seed', 'converged', 'convergence_time', 'timestamp'])
    
    # Initialize global key
    global_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Create a list of keys for each sample
    keys = []
    current_key = global_key
    for _ in range(300):
        current_key, subkey = jax.random.split(current_key)
        keys.append(subkey)
    
    # Process samples in parallel
    with mp.Pool(num_cores) as pool:
        args = [(i, key, output_dir) for i, key in enumerate(keys)]
        results = pool.map(process_sample_wrapper, args)
    
    # Write all results to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow([
                result['sample_id'],
                result['seed'],
                result['converged'],
                result['convergence_time'],
                result['timestamp']
            ])

if __name__ == "__main__":
    main()