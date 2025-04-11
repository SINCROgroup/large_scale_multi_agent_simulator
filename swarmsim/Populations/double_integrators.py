import numpy as np
from swarmsim.Populations import Populations
from typing import Optional
from swarmsim.Utils import broadcast_parameter


class DampedDoubleIntegrators(Populations):
    """
    A class that implements (noisy) second order integrators

    Arguments
    -------
    x (Nx2D double matrix) : state of the agents row=agent, column=state variable
    N (double) : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    D (NxD double matrix) : Diffusion matrix
    damping (double) : Damping coefficient
    params (Dictionary) : Dictionary of parameters

    Methods
    -------
    get_drift(self,u):
        Integrate the input forces
    get_diffusion(self,u):
        Brownian motion
    reset_state(self):
        Resets the state of the agent.
    """

    def __init__(self, config_path) -> None:

        self.damping: Optional[float] = None
        self.D: Optional[np.ndarray] = None

        super().__init__(config_path)

    def get_drift(self):
        d = self.state_dim // 2  # position and velocity dimension

        velocity = self.x[:, d:]  # current velocities
        acceleration = -self.damping * velocity + self.u + self.f

        return np.hstack((velocity, acceleration))

    def get_diffusion(self):
        return self.D

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.damping = broadcast_parameter(self.params['damping'], (1,))
        self.D = broadcast_parameter(self.params['D'], (self.state_dim,self.state_dim))

