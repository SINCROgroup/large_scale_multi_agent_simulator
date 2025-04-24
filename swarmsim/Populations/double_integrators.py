import numpy as np
from swarmsim.Populations import Population
from typing import Optional


class DampedDoubleIntegrators(Population):
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


    def __init__(self, config_path: str, name: str = None) -> None:
        super().__init__(config_path, name)

        self.damping: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        self.params_shapes = {
            'damping': (),
            'D': ()
        }

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.damping = self.params['damping']
        self.D = self.params['D'][:,np.newaxis] * np.array([0, 0, 1, 1])

    def get_drift(self):

        velocity = self.x[:, self.input_dim:]  # current velocities
        acceleration = -self.damping[:, np.newaxis] * velocity + self.u + self.f

        return np.hstack((velocity, acceleration))

    def get_diffusion(self):
        return self.D


