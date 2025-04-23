import numpy as np
import yaml
from swarmsim.Interactions import Interaction
from swarmsim.Populations import Population
from typing import Optional

class HarmonicRepulsion(Interaction):
    """
    Implements a finite-range harmonic repulsion force between two populations.

    This interaction models a repulsive force between agents in two populations,
    where the force magnitude decreases linearly as the distance increases, up to
    a maximum interaction distance.

    Parameters
    ----------
    target_population : Population
        The first population (typically the affected agents).
    source_population : Population
        The second population (agents exerting repulsion).
    config : str
        Path to the YAML configuration file containing interaction parameters.

    Attributes
    ----------
    strength : float
        Maximum intensity of the repulsion force.
    distance : float
        Maximum distance at which the interaction takes place.

    Config requirements
    -------------------
    repulsion:
        strength : float
            Maximum repulsion force intensity.
        max_distance : float
            Maximum interaction range.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required interaction parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        repulsion:
            strength: 1.5
            max_distance: 10.0

    This sets a repulsion force with a strength of `1.5` and an interaction
    range of `10.0` units.
    """

    def __init__(self,
                 target_population: Population,
                 source_population: Population,
                 config_path: str,
                 name: str = None) -> None:

        super().__init__(target_population, source_population, config_path, name)

        self.strength: Optional[np.ndarray] = None
        self.distance: Optional[np.ndarray] = None

        self.params_shapes = {
            "strength": (),
            "distance": ()
        }

    def reset(self):
        super().reset()
        self.strength = self.params['strength']
        self.distance = self.params['distance']

    def get_interaction(self):
        """
        Computes the repulsion force exerted by `source_population` on `target_population`.

        The repulsion force follows a **harmonic** behavior where:
        - The force is strongest when two agents are very close.
        - It decreases linearly as the distance increases.
        - It becomes zero when the distance exceeds `max_distance`.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the repulsion force applied to each
            agent in `target_population`, where `N1` is the number of agents in `target_population` and
            `D` is the dimensionality of the state space.

        Notes
        -----
        - The function computes **pairwise distances** between agents in `target_population`
          and `source_population` and applies the harmonic repulsion formula.
        - To avoid division by zero, a small epsilon (`1e-6`) is added to distances.

        """
        # Compute pairwise differences between agents in `target_population` and `source_population`
        differences = self.source_population.x[:, np.newaxis, :] - self.target_population.x[np.newaxis, :, :]

        # Compute Euclidean distances between agents
        distances = np.linalg.norm(differences, axis=2)

        # Identify agents within interaction range
        nearby_agents = distances < self.distance

        # Zero out interactions beyond max interaction range
        nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)

        # Prevent division by zero by setting a small minimum value for distances
        distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)

        # Compute unit vectors pointing from `target_population` to `source_population`
        nearby_unit_vector = nearby_differences / distances_with_min

        # Apply harmonic repulsion force formula
        repulsion = -self.strength[:, np.newaxis] * np.sum(
            (self.distance[np.newaxis, :, np.newaxis] - distances[:, :, np.newaxis]) * nearby_unit_vector,
            axis=0
        )

        return repulsion
