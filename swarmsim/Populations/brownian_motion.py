import numpy as np
import yaml
from swarmsim.Populations import Populations


class BrownianMotion(Populations):
    """
    A class that implements a biased Brownian motion with avreage speed \mu and diffusion coefficient D.

    Parameters
    -------
        config_path : str (Path of the configuration file)  

    Attributes
    -------
        x  : numpy array (num_agents x dim_state)   State of the agents row=agent, column=state variable 
        mu : numpy array (num_agents)               Average velocity along the axes
        N  : int                                    Number of agents in the population
        f  : numpy array (num_agents x dim_state)   External forces (interactions and environment)
        u  : numpy array (num_agents x dim_state)   Control input
        D  : numpy array (num_agents) :             Diffusion coefficient
        params : Dictionary                         Dictionary of parameters

    Configuration file requirements
    ------
        - x0_mode: str                  The mode in which the Initial conditions are generated. ("From file" or "Random")   
        - x0_file_path: str             (If x0_mode == "From file") Path of the .csv file where Initial conditions are stored
        - N : int                       (If x0_mode == "Random") Number of agents in the poulation
        - state_dim : int               (If x0_mode == "Random") Dimensions of the state
        - D: python command             command to generate the diffusion coefficient 
        - mu: python command            Command to generate the Average velocities
        - id: str                       Identifier of the population

    Notes
    ------
        For more details on how initial conditions can be generated see the method get_initial_conditions of the class Population

    """

    def __init__(self, config_path:str) -> None:

        super().__init__(config_path)

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            self.params = yaml.safe_load(file)
        
        N = self.N
        self.mu = eval(self.params["BrownianMotion"]["mu"])    # Average velocity
        self.D = eval(self.params["BrownianMotion"]["D"])      # Diffusion coefficient
        self.id = self.params["BrownianMotion"]["id"]  # Population ID

        self.f = np.zeros(self.x.shape)          # Initialization of the external forces
        self.u = np.zeros(self.x.shape)          # Initialization of the control input

    def get_drift(self):
        '''
        Movement at constant speed \mu (\mu_1, \mu_2,..., \mu_n).

        Returns:
        ------
            drift: numpy array (num_agents x dim_state)         Dirft of the population

        '''
        drift = self.mu + self.f + self.u
        return drift

    def get_diffusion(self):
        '''
        Stochastic diffusion in the environment (Standard Weiner Process).

        Returns:
        ------
            diffusion: numpy array (num_agents x dim_state)     Diffusion of the population

        '''
        return self.D

    def reset_state(self):
        '''
        
        Resets the state to the initial conditions.

        '''
        
        self.get_initial_conditions(self.config_path)
        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

        