"""Module containing the environment class."""

import jax
import jax.numpy as jnp
from .obstacles import RectangleObstacle
from .config import (
    DEFAULT_ENV_WIDTH,
    DEFAULT_ENV_HEIGHT,
    MIN_OBSTACLE_SIZE_RATIO,
    MAX_OBSTACLE_SIZE_RATIO
)

class AirplaneObstacleEnvironment:
    """Environment containing obstacles for the airplane."""
    
    def __init__(self, width=DEFAULT_ENV_WIDTH, height=DEFAULT_ENV_HEIGHT, obstacles=None):
        """Initialize environment.
        
        Args:
            width: Width of the environment
            height: Height of the environment
            obstacles: List of obstacles (optional)
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else []

    def set_random_obstacles(self, num_obstacles=1, key=None):
        """Set random obstacles in the environment.
        
        Args:
            num_obstacles: Number of obstacles to generate
            key: JAX random key (optional)
        """
        self.obstacles = []
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Define min and max sizes relative to environment
        min_size = MIN_OBSTACLE_SIZE_RATIO * min(self.width, self.height)
        max_size = MAX_OBSTACLE_SIZE_RATIO * min(self.width, self.height)
        
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
        """Calculate signed distances for given grid points.
        
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