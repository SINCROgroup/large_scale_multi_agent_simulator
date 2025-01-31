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

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.config = config.get('SimpleIntegrators', {})
        self.id = self.config.get('id', "Targets")  # Population ID

        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

    def get_drift(self):
        return self.u + self.f

    def get_diffusion(self):
        return np.array([self.params['D_x'].values,self.params['D_y'].values]).T

    def reset_state(self):
        N = self.N
        self.x = self.get_initial_conditions()  # Initial conditions
        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input
