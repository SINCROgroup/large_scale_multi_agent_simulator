import numpy as np
import yaml
from Models.agents import agents


class brownian_motion(agents):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable 
    mu (D dimensional vector) : Average velocity along the axes
    
    Methods
    -------
    get_drift(self,u):
        Constant diffusion in each dimension.
    get_diffusion(self,u):
        Standard weined process
    """

    def __init__(self,avg_v,config_file) -> None:

        # Load the YAML configuration file
        with open(config_file, "r") as file:
            pars = yaml.safe_load(file)
        
        self.x = np.array(pars["brownian_motion"]["x0"])           # Initial conditions
        self.mu = np.array(pars["brownian_motion"]["mu"])          # Average velocity
        self.D = np.array(pars["brownian_motion"]["D"])            # Diffusion coefficient



    def get_drift(self,x,u):
        
        return self.mu

    def get_diffusion(self,x,u):
        
        return self.D

        