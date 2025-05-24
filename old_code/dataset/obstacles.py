"""Module containing obstacle classes for the environment."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Obstacle(ABC):
    """Abstract base class for obstacles."""
    
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
    """Axis aligned rectangle obstacle."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        """Initialize rectangle obstacle.
        
        Args:
            x1: x-coordinate of first corner
            y1: y-coordinate of first corner
            x2: x-coordinate of second corner
            y2: y-coordinate of second corner
        """
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
        rect = plt.Rectangle(
            (self.x1, self.y1),
            self.x2 - self.x1,
            self.y2 - self.y1,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect) 