import numpy as np
import yaml
from Populations.populations import Populations


class BrownianMotion(Populations):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    x (NxD double matrix) : state of the agents row=agent, column=state variable 
    mu (D dimensional vector) : Average velocity along the axes
    D (D dimensional vecotr) : Diffusion coefficient
    N (double) : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    
    Methods
    -------
    get_drift(self,u):
        Constant diffusion in each dimension.
    get_diffusion(self,u):
        Standard Wiener process
    """

    def __init__(self, config_path) -> None:

        super().__init__()

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            pars = yaml.safe_load(file)
        
        N = pars["BrownianMotion"]["N"]
        self.N = N
        self.x = eval(pars["BrownianMotion"]["x0"])     # Initial conditions
        self.mu = eval(pars["BrownianMotion"]["mu"])    # Average velocity
        self.D = eval(pars["BrownianMotion"]["D"])      # Diffusion coefficient
        self.id = pars["BrownianMotion"]["id"]  # Population ID

        self.f = np.zeros(self.x.shape)          # Initialization of the external forces
        self.u = np.zeros(self.x.shape)          # Initialization of the control input

    def get_drift(self):
        drift = self.mu + self.f + self.u
        return drift

    def get_diffusion(self):
        
        return self.D
        