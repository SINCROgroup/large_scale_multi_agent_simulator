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

        self.id = self.config.get('id', "Targets")  # Population ID
        
        #Load D
        self.D = np.empty([self.N,len(self.params['D'][0])])
        i=0
        for agent_D in self.params['D']:
            self.D[i,:] = agent_D
            i+=1

        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

        self.v_max = self.config.get('v_max', float('inf'))  # Action limit

    def get_drift(self):
        return self.u + self.f

    def get_diffusion(self):
        return self.D * np.ones((self.N, self.state_dim))

    def reset_state(self):
        N = self.N
        self.x = self.get_initial_conditions()  # Initial conditions
        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input
