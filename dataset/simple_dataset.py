import jax
import jax.numpy as jnp
import hj_reachability as hj
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np  # Only for plotting

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
        super().__init__(control_mode='max', disturbance_mode='min')
        self.v0 = v0
        self.control_space = hj.sets.Box(jnp.array([0.2, -0.5]), jnp.array([0.6, 0.5]))
        self.disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
    
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
        return jnp.zeros((3, 0))
    
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
        max_size = 0.2 * min(self.width, self.height)
        
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
    x = jnp.linspace(0, env.width, 1000)
    y = jnp.linspace(0, env.height, 1000)
    X, Y = jnp.meshgrid(x, y)
    
    # Get signed distances from the environment
    signed_distances = env.get_signed_distances(X, Y)
    
    # Convert to numpy for plotting
    X_np = np.array(X)
    Y_np = np.array(Y)
    signed_distances_np = np.array(signed_distances)
    
    # Create a custom colormap that makes red start at 0
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.0, 'red'), (0.5, 'white'), (1.0, 'green')]  # Red to white to green
    cmap = LinearSegmentedColormap.from_list('custom_rdylgn', colors)
    
    # Set levels to have more resolution around 0
    levels = np.linspace(-2, 2, 100)
    
    # Plot with the new colormap and normalization
    contour = ax.contourf(X_np, Y_np, signed_distances_np, levels=levels, cmap=cmap, norm=plt.Normalize(-2, 2), extend='both')
    ax.contour(X_np, Y_np, signed_distances_np, levels=[0], colors='k', linewidths=2)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Safety function l(x)')
    cbar.ax.axhline(0, color='black', linewidth=2)


def main():
    key = jax.random.PRNGKey(42)
    
    env = AirplaneObstacleEnvironment()
    env.set_random_obstacles(5, key=key)
    fig, ax = plt.subplots()
    plot_signed_distances(ax, env)
    plot_environment(ax, env)
    plt.show()



if __name__ == "__main__":
    main()