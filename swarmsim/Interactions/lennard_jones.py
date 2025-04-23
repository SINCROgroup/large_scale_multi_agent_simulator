import numpy as np
from swarmsim.Interactions import Interaction
from swarmsim.Utils import compute_distances
from swarmsim.Populations import Population
from typing import Optional


class LennardJones(Interaction):

    def __init__(self,
                 target_population: Population,
                 source_population: Population,
                 config_path: str,
                 name: str = None) -> None:

        super().__init__(target_population, source_population, config_path, name)

        self.epsilon: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

        self.params_shapes = {
            'epsilon': (),
            'sigma': ()
        }

    def reset(self):
        super().reset()
        
        self.epsilon = self.params.get('epsilon')
        self.sigma = self.params.get('sigma')

    def get_interaction(self):
        # Compute pairwise distances and relative positions between agents
        distances, relative_positions = compute_distances(self.target_population.x[:, :2],
                                                          self.source_population.x[:, :2])

        # Prevent division by zero
        distances = np.maximum(distances, 1e-6)

        # Get epsilon and sigma values with correct shape
        epsilon = self.epsilon.reshape(-1, 1)  # shape: (N_target, 1)
        sigma = self.sigma.reshape(-1, 1)  # shape: (N_target, 1)

        # Compute normalized sigma/distance
        sigma_over_r = sigma / distances

        # Compute Lennard-Jones scalar force magnitude
        lj_scalar = 24 * epsilon * (2 * (sigma_over_r ** 12) - (sigma_over_r ** 6)) / (distances ** 2)

        # Compute vector force
        forces = np.sum(lj_scalar[:, :, np.newaxis] * relative_positions, axis=1)

        return forces
