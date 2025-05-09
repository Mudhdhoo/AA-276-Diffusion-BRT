"""Module containing the airplane dynamics."""

import jax.numpy as jnp
import hj_reachability as hj

class AirplaneDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    """Airplane dynamics with control and disturbance."""
    
    def __init__(self, v0=0.4):
        """Initialize airplane dynamics.
        
        Args:
            v0: Nominal velocity of the airplane
        """
        self.v0 = v0
        self.control_space = hj.sets.Box(
            jnp.array([0.5, -0.5]),
            jnp.array([1.0, 0.5])
        )
        self.disturbance_space = hj.sets.Box(
            jnp.array([0.0]),
            jnp.array([0.0])
        )
        super().__init__(
            control_space=self.control_space,
            disturbance_space=self.disturbance_space,
            control_mode='max',
            disturbance_mode='min'
        )
    
    def open_loop_dynamics(self, state, time):
        """Open loop dynamics of the airplane.
        
        Args:
            state: State vector [x, y, theta]
            time: Current time
            
        Returns:
            State derivative vector
        """
        x, y, theta = state
        return jnp.array([
            self.v0 * jnp.cos(theta),
            self.v0 * jnp.sin(theta),
            0.0
        ])
    
    def control_jacobian(self, state, time):
        """Control Jacobian of the airplane.
        
        Args:
            state: State vector [x, y, theta]
            time: Current time
            
        Returns:
            Control Jacobian matrix
        """
        x, y, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],
            [jnp.sin(theta), 0.0],
            [0.0, 1.0]
        ])
    
    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian of the airplane.
        
        Args:
            state: State vector [x, y, theta]
            time: Current time
            
        Returns:
            Disturbance Jacobian matrix
        """
        return jnp.zeros((3, 1))
    
    def control_space(self):
        """Get the control space.
        
        Returns:
            Control space box
        """
        return self.control_space

    def disturbance_space(self):
        """Get the disturbance space.
        
        Returns:
            Disturbance space box
        """
        return self.disturbance_space 