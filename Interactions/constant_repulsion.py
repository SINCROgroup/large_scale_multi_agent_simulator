import numpy as np
import yaml
from Interactions.interaction import Interaction


class RepulsionConst(Interaction):

    """
    A class that implements a biased Brownian motion 

    Arguments
    -------
    ity (double): 
        Intensity of the repulsion force
    dis (double):
        max distance at which the interaction takes place
    

    
    Methods
    -------
    get_interaction(self):
        Returns a vector (N1,D) that describes how population2 influences the dynamics of population 1. N1 is the number of agents of
        population 1 and D is the dimension of the state space of the agents of population 1

    """

    def __init__(self, pop1, pop2, config) -> None:
        super().__init__(pop1,pop2)
        # Load the YAML configuration file
        with open(config, "r") as file:
            pars = yaml.safe_load(file)

        self.ity = pars["repulsion"]["ity"]
        self.dis = pars["repulsion"]["dis"]

    def get_interaction(self):

        differences = self.pop2.x[:, np.newaxis, :] - self.pop1.x[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)
        nearby_agents = distances < self.dis
        nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)
        distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)
        nearby_unit_vector = nearby_differences / distances_with_min
        repulsion = -self.ity * np.sum((self.dis - distances[:, :, np.newaxis]) * nearby_unit_vector, axis=0)
        return repulsion
        #
        # # f_i = np.zeros(self.pop1.N,self.pop1.x.shape[1])
        # differences = self.pop1.x[:, np.newaxis, :] - self.pop2.x[np.newaxis, :, :]  # Element wise differences (N1xN2xD)
        # distances = np.linalg.norm(differences, axis=2)                              # Norm of differences (N1xN2)
        # norm_diff = differences/distances[:, :, np.newaxis]                            # Versor of the differences (N1xN2xD)
        # # range_diff = (distances<self.dis).astype(float)                              # Distances in range (N1xN2)
        # range_diff = norm_diff * distances[:, :, np.newaxis]                           # Differences in range (N1xN2xD)
        # f_i = self.ity * np.sum(range_diff, 1)                                        # Forces on each agent (N1xD)
        #
        # return f_i


# IN: then in utilities we should add a function that computes the distance matrix and call that here
