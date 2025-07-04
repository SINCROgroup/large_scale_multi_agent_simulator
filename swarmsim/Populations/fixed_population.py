import numpy as np
from swarmsim.Populations import Population


class FixedPopulation(Population):
    """
    A class that implements a population that does not move.

    Parameters
    -------
        config_path : str (Path of the configuration file)  

    Attributes
    -------
        x  : numpy array (num_agents x dim_state)   State of the agents row=agent, column = x,y coordinate 
        mu : numpy array (num_agents)               Average velocity along the axes
        N  : int                                    Number of agents in the population
        f  : numpy array (num_agents x dim_state)   External forces (interactions and environment)
        u  : numpy array (num_agents x dim_state)   Control input
        D  : numpy array (num_agents) :             Diffusion coefficient
        config : Dictionary                         Dictionary of parameters

    Configuration file requirements
    ------
        - x0_mode: str                  The mode in which the Initial conditions are generated. ("From file" or "Random")   
        - x0_file_path: str             (If x0_mode == "From file") Path of the .csv file where Initial conditions are stored
        - N : int                       (If x0_mode == "Random") Number of agents in the poulation
        - state_dim : int               (If x0_mode == "Random") Dimensions of the state
        - id: str                       Identifier of the population

    Notes
    ------
        For more details on how initial conditions can be generated see the method get_initial_conditions of the class Population

    """

    def get_drift(self) -> np.array :
        '''
        No movement on average.

        Returns:
        ------
            drift: numpy array (num_agents x dim_state)         Dirft of the population

        '''
        return self.u

    def get_diffusion(self) -> np.array :
        '''
        No stochasticity.

        Returns:
        ------
            diffusion: numpy array (num_agents x dim_state)     Diffusion of the population

        '''
        
        return np.zeros(self.x.shape)
