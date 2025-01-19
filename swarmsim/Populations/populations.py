from abc import ABC, abstractmethod
import numpy as np


class Populations(ABC):
    """
    An interface that defines all the methods that an agent should implement
    Arguments
    -------
    N : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    id (string) : defines the population name

    Methods
    -------
    get_drift(self,u):
        Returns the drift (deterministic part of the dynamics) of the agent.
    get_diffusion(self,u):
        Returns the diffusion (stochastic part of the dynamics) of the agent.
    """

    def __init__(self) -> None:
        super().__init__()

    # This method
    @abstractmethod
    def get_drift(self):
        pass

    @abstractmethod
    def get_diffusion(self):
        pass
