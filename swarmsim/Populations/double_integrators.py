import numpy as np
import yaml
from swarmsim.Populations import Populations


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
        super().__init__(config_path)

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.config = config.get('DampedDoubleIntegrators', {})

        self.id = self.config.get('id', "Targets")  # Population ID

        self.D = np.array([0, 0, 1, 1]) * self.config.get('D', 0)  # Diffusion matrix

        self.damping = self.config.get('damping', 1)

        self.f = np.zeros((self.x.shape[0], self.x.shape[1] // 2))  # Initialization of the external forces
        self.u = np.zeros((self.x.shape[0], self.x.shape[1] // 2))  # Initialization of the control input

    def get_drift(self):
        # Combine the first two columns of x and u to form drift
        return np.hstack((self.x[:, 2:], - self.damping * self.x[:, 2:] + self.u + self.f))

    def get_diffusion(self):
        return self.D * np.ones((self.N, self.state_dim))

    def reset_state(self):
        N = self.N
        self.x = self.get_initial_conditions()
        self.f = np.zeros((self.x.shape[0], self.x.shape[1] // 2))  # Initialization of the external forces
        self.u = np.zeros((self.x.shape[0], self.x.shape[1] // 2))  # Initialization of the control input

    def reset_params(self) -> None:
        pass