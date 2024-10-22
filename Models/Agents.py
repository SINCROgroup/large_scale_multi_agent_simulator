from abc import ABC, abstractmethod


class agents(ABC):
    """
    An interface that defines all the methods that an agent should implement

    Methods
    -------
    get_drift(self,u):
        Returns the drift (deterministic part of the dynamics) of the agent.
    get_diffusion(self,u):
        Returns the diffusion (stochastic part of the dynamics) of the agent.
    """
    
    #This method 
    @abstractmethod
    def get_drift(self):
        pass

    @abstractmethod
    def get_diffusion(self):
        pass
