import numpy as np
import yaml
from swarmsim.Interactions import Interaction


class HarmonicRepulsion(Interaction):
    """
    Implements a finite-range harmonic repulsion force between two populations.

    This interaction models a repulsive force between agents in two populations,
    where the force magnitude decreases linearly as the distance increases, up to
    a maximum interaction distance.

    Parameters
    ----------
    pop1 : Population
        The first population (typically the affected agents).
    pop2 : Population
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

    def __init__(self, pop1, pop2, config) -> None:
        super().__init__(pop1, pop2)

        # Load repulsion parameters from YAML configuration file
        with open(config, "r") as file:
            pars = yaml.safe_load(file)

        # Extract strength and interaction range from the config
        self.strength_nom = pars["repulsion"]["strength"]  # Maximum repulsion intensity
        self.distance_nom = pars["repulsion"]["max_distance"]  # Maximum interaction distance

        if pars.get("repulsion").get("std", None) is not None:
            self.std = pars["repulsion"]["std"]
        else:
            self.std = 0

        self.strength = self.strength_nom
        self.distance = self.distance_nom

    def reset_params(self):
        self.strength = np.random.normal(self.strength_nom, self.std * self.strength_nom)
        self.distance = np.random.normal(self.distance_nom, self.std * self.distance_nom)

    def get_interaction(self):
        """
        Computes the repulsion force exerted by `pop2` on `pop1`.

        The repulsion force follows a **harmonic** behavior where:
        - The force is strongest when two agents are very close.
        - It decreases linearly as the distance increases.
        - It becomes zero when the distance exceeds `max_distance`.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the repulsion force applied to each
            agent in `pop1`, where `N1` is the number of agents in `pop1` and
            `D` is the dimensionality of the state space.

        Notes
        -----
        - The function computes **pairwise distances** between agents in `pop1`
          and `pop2` and applies the harmonic repulsion formula.
        - To avoid division by zero, a small epsilon (`1e-6`) is added to distances.

        """
        # Compute pairwise differences between agents in `pop1` and `pop2`
        differences = self.pop2.x[:, np.newaxis, :] - self.pop1.x[np.newaxis, :, :]

        # Compute Euclidean distances between agents
        distances = np.linalg.norm(differences, axis=2)

        # Identify agents within interaction range
        nearby_agents = distances < self.distance

        # Zero out interactions beyond max interaction range
        nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)

        # Prevent division by zero by setting a small minimum value for distances
        distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)

        # Compute unit vectors pointing from `pop1` to `pop2`
        nearby_unit_vector = nearby_differences / distances_with_min

        # Apply harmonic repulsion force formula
        repulsion = -self.strength * np.sum(
            (self.distance - distances[:, :, np.newaxis]) * nearby_unit_vector,
            axis=0
        )

        return repulsion
