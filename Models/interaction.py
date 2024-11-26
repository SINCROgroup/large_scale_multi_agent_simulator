from abc import ABC, abstractmethod


class Interaction(ABC):
    
    """
    An interface that defines all the methods that an interaction between populations should implement

    Methods
    -------
    get_interaction(self):
        Returns a vector (N1,D) that describes how population2 influences the dynamics of population 1. N1 is the number of agents of
        population 1 and D is the dimension of the state space of the agents of population 1
    """

    def __init__(self,pop1,pop2) -> None:
        super().__init__()
        self.pop1=pop1
        self.pop2=pop2

    # This method returns the forces that population 2 applies on population 1
    @abstractmethod
    def get_interaction(self):
        pass

