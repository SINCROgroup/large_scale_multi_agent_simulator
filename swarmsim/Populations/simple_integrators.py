import numpy as np
import yaml
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

    Methods
    -------
    get_drift(self,u):
        Integrate the input forces
    get_diffusion(self,u):
        Brownian motion
    """

    def __init__(self, config_path) -> None:
        super().__init__()

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        params = config.get('SimpleIntegrators', {})

        self.N = params.get('N', 1)
        N = self.N
        self.x = eval(params.get('x0', 'np.random(-1, 1, size=(self.N, 2))'))  # Initial conditions
        self.id = params.get('id', "Targets")  # Population ID
        self.D = params.get('D', 0)  # Diffusion coefficient

        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

    def get_drift(self):
        return self.u + self.f

    def get_diffusion(self):
        return self.D
