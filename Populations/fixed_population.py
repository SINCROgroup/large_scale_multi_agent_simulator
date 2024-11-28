import numpy as np
import yaml
from Populations.populations import Populations


class FixedPopulation(Populations):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable 
    N (double) : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    
    Methods
    -------
    get_drift(self,u):
        No movement
    get_diffusion(self,u):
        No diffusion
    """

    def __init__(self, config_path) -> None:

        super().__init__()

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            pars = yaml.safe_load(file)
        
        N = pars["Fixed"]["N"]
        self.N = N
        self.x = eval(pars["Fixed"]["x0"])     # Initial conditions
        self.id = pars["Fixed"]["id"]  # Population ID

        self.f = np.zeros(self.x.shape)          # Initialization of the external forces
        self.u = np.zeros(self.x.shape)          # Initialization of the control input

    def get_drift(self):

        return np.zeros(self.x.shape)

    def get_diffusion(self):
        
        return np.zeros(self.x.shape)
        