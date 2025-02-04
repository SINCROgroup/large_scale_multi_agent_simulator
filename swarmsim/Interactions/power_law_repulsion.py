import numpy as np
import yaml
from swarmsim.Interactions import Interaction

from swarmsim.Utils import compute_distances


class PowerLawRepulsion(Interaction):

    """
    A class that implements a shifted and truncated power law repulsion

    Arguments
    -------
    strength (double):
        max intensity of the repulsion force
    max_distance (double):
        max distance at which the interaction takes place
    p (int):
        exponent of the power law repulsion
    

    
    Methods
    -------
    get_interaction(self):
        Returns a vector (N1,D) that describes how herders influences the dynamics of population 1. N1 is the number of agents of
        population 1 and D is the dimension of the state space of the agents of population 1

    """

    def __init__(self, pop1, pop2, config, repulsion_name) -> None:
        super().__init__(pop1, pop2)

        # Load the YAML configuration file
        with open(config, "r") as file:
            config = yaml.safe_load(file)
        self.params = config.get(repulsion_name, {})

        self.strength = self.params["strength"]
        self.max_distance = self.params["max_distance"]
        self.p = self.params["p"]

    def get_interaction(self):

        distances, relative_positions = compute_distances(self.pop1.x[:, :2], self.pop2.x[:, :2])

        distances = np.maximum(distances, 1e-6)

        y_f = 1 / (self.max_distance ** self.p)

        kernel = (1 / (distances ** self.p) - y_f)
        kernel = self.strength * np.maximum(kernel, 0)

        repulsion = np.sum(kernel[:, :, np.newaxis] * relative_positions, axis=1)

        return repulsion
