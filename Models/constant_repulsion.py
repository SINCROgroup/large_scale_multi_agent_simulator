import numpy as np
import yaml
from Models.agents import Interaction


class Repulsion_const(Interaction):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    ity (double): 
        Intensity of the repulsion force

    
    Methods
    -------
    get_interaction(self):
        Returns a vector (N1,D) that describes how population2 influences the dynamics of population 1. N1 is the number of agents of
        population 1 and D is the dimension of the state space of the agents of population 1

    """

    def __init__(self,pop1,pop2) -> None:
        super().__init__(pop1,pop2)

    def get_interaction(self):
        pass


    