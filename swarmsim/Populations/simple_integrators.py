import numpy as np
from swarmsim.Populations import Populations


class SimpleIntegrators(Populations):
    """
    A class that implements (noisy) first order integrators

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable
    N (double) : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    D (NxD double matrix) : Diffusion matrix
    config (Dictionary) : Dictionary of parameters

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
        super().__init__(config_path)

        self.v_max = self.config.get('v_max', float('inf'))  # Action limit

    def get_drift(self):
        return np.clip(self.u + self.f, -self.v_max, self.v_max)

    def get_diffusion(self):
        return np.zeros((self.N, self.state_dim))
