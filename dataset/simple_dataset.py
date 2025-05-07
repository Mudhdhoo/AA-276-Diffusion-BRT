import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Obstacle(ABC):
    """ Abstract base class for obstacles. """
    @abstractmethod
    def get_distance(self, x: float, y: float) -> float:
        """Get the signed distance to the closest point on the obstacle."""
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
        
    
    def get_distance(self, x: float, y: float) -> float:
        """Get the signed distance to the closest point on the rectangle."""
        dx = jnp.maximum(jnp.maximum(self.x1 - x, 0.0), x - self.x2)
        dy = jnp.maximum(jnp.maximum(self.y1 - y, 0.0), y - self.y2)
        outside_dist = jnp.hypot(dx, dy)
        inside_dx = jnp.minimum(x - self.x1, self.x2 - x)
        inside_dy = jnp.minimum(y - self.y1, self.y2 - y)
        inside_dist = -jnp.minimum(inside_dx, inside_dy)
        is_inside = (self.x1 <= x) & (x <= self.x2) & (self.y1 <= y) & (y <= self.y2)
        return jnp.where(is_inside, inside_dist, outside_dist)
    
    def plot(self, ax):
        """Plot the obstacle."""
        rect = plt.Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1, facecolor='red', alpha=0.5)
        ax.add_patch(rect)
        

class AirplaneDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    """ Airplane dynamics with control and disturbance. """
    def __init__(self, v0=0.4):
        super().__init__(control_mode='max', disturbance_mode='min')
        self.v0 = v0
    
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
        return hj.sets.Box(jnp.array([0.2, -0.5]), jnp.array([0.6, 0.5]))

    def disturbance_space(self):
        """Disturbance space of the airplane."""
        return hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        

class AirplaneObstacleEnvironment():
    def __init__(self, width=10, height=10, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else []

    def set_random_obstacles(self, num_obstacles=1):
        self.obstacles = []
        for _ in range(num_obstacles):
            max_width = min(self.width, self.height) * 0.1
            max_height = max_width
            x1 = np.random.uniform(0, self.width - max_width)
            y1 = np.random.uniform(0, self.height - max_height)
            x2 = x1 + max_width
            y2 = y1 + max_height
            self.obstacles.append(RectangleObstacle(x1, y1, x2, y2))


#########################################################################
def plot_environment(env):
    fig, ax = plt.subplots()
    for obs in env.obstacles:
        obs.plot(ax)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    plt.show()


def main():
    env = AirplaneObstacleEnvironment()
    env.set_random_obstacles(5)
    plot_environment(env)



if __name__ == "__main__":
    main()