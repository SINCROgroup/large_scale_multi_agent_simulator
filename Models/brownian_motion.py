import numpy as np
import yaml
from Models.agents import agents


class brownian_motion(agents):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable 
    
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
        
        self.x = np.array(pars["brownian_motion"]["x0"])

        print(self.x)


    def get_drift(self,u):
        pass

    def get_diffusion(self):
        pass

        