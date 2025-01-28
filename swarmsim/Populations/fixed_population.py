import numpy as np
import yaml
from swarmsim.Populations import Populations


class FixedPopulation(Populations):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable 
    N (double) : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    params (Dictionary) : Dictionary of parameters
    
    Methods
    -------
    get_drift(self,u):
        No movement
    get_diffusion(self,u):
        No diffusion
    reset_state(self):
        Resets the state of the agent.
    """

    def __init__(self, config_path) -> None:

        super().__init__(config_path)

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            self.params = yaml.safe_load(file)
        
        self.id = self.params["FixedPopulation"]["id"]  # Population ID

        self.f = np.zeros(self.x.shape)          # Initialization of the external forces
        self.u = np.zeros(self.x.shape)          # Initialization of the control input

    def get_drift(self):

        return np.zeros(self.x.shape)+self.u

    def get_diffusion(self):
        
        return np.zeros(self.x.shape)

    def reset_state(self):
        self.x = eval(self.params["Fixed"]["x0"])     # Initial conditions
        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

        