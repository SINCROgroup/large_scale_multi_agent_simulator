import numpy as np
import yaml
from Models.agents import Agents


class Fixed_population(Agents):

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

        self.f = np.zeros(self.x.size)          # Initialization of the external forces 
        self.u = np.zeros(self.x.size)          # Initialization of the control input


    def get_drift(self, x, u):

        return np.zeros(self.x.size)

    def get_diffusion(self, x, u):
        
        return np.zeros(self.x.size)
        