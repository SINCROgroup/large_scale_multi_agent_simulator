import numpy as np
import yaml
from swarmsim.Interactions import Interaction


class HarmonicRepulsion(Interaction):

    """
    A class that implements a finite range harmonic repulsion

    Arguments
    -------
    ity (double): 
        max intensity of the repulsion force
    dis (double):
        max distance at which the interaction takes place
    

    
    Methods
    -------
    get_interaction(self):
        Returns a vector (N1,D) that describes how herders influences the dynamics of population 1. N1 is the number of agents of
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


# IN: then in utilities we should add a function that computes the distance matrix and call that here
