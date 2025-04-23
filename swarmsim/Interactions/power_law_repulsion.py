import numpy as np
from swarmsim.Interactions import Interaction
from swarmsim.Populations import Population
from swarmsim.Utils import compute_distances
from typing import Optional


class PowerLawRepulsion(Interaction):
    """
    Implements a power-law repulsion interaction between two populations.

    This interaction models a repulsive force that decays according to a power-law function
    with respect to the distance between agents. The strength of the force is determined by
    the power exponent `p` and is active within a defined maximum interaction range.

    Parameters
    ----------
    pop1 : Population
        The first population that is influenced by the interaction.
    pop2 : Population
        The second population that applies the repulsion force.
    config : str
        Path to the YAML configuration file containing interaction parameters.
    repulsion_name : str
        The section name in the YAML file specifying the repulsion parameters.

    Attributes
    ----------
    params : dict
        Dictionary containing interaction parameters loaded from the configuration file.
    strength : float
        Maximum intensity of the repulsion force.
    max_distance : float
        Maximum distance at which the interaction takes place.
    p : float
        Power exponent controlling the decay rate of the repulsion force.

    Config requirements
    -------------------
    repulsion_name : str
        The section name in the YAML file specifying the repulsion parameters.
    strength : float
        The maximum repulsion force intensity.
    max_distance : float
        The maximum interaction range.
    p : float
        The power exponent determining the force decay.

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

        power_law_repulsion:
            strength: 2.0
            max_distance: 5.0
            p: 3.0

    This sets a repulsion force with `strength = 2.0`, active within `5.0` units,
    and decaying with a power exponent `p = 3.0`.
    """

    def __init__(self,
                 target_population: Population,
                 source_population: Population,
                 config_path: str,
                 name: str = None) -> None:

        super().__init__(target_population, source_population, config_path, name)

        self.strength: Optional[np.ndarray] = None
        self.max_distance: Optional[np.ndarray] = None

        self.params_shapes = {
            "strength": (),
            "max_distance": ()
        }

        self.p: int = self.config.get("p")

    def reset(self):
        super().reset()
        self.strength = self.params['strength']
        self.max_distance = self.params['max_distance']


    def get_interaction(self):
        """
        Computes the repulsion force exerted by `source_population` on `target_population` using a power-law function.

        The repulsion force is computed as:

            F_repulsion = strength * (1/distance^p - 1/max_distance^p)

        where:
            - `distance` is the Euclidean distance between agents in `target_population` and `source_population`.
            - `max_distance` defines the interaction cutoff beyond which no force is applied.
            - `p` controls how rapidly the force decays with distance.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the repulsion force applied to each
            agent in `target_population`, where `N1` is the number of agents in `target_population` and
            `D` is the dimensionality of the state space.

        Notes
        -----
        - The function prevents division by zero by setting a minimum distance (`1e-6`).
        - The force is capped at `10` to avoid numerical instabilities.
        """

        # Compute pairwise distances and relative positions between agents
        distances, relative_positions = compute_distances(self.target_population.x[:, :2], self.source_population.x[:, :2])

        # Prevent division by zero
        distances = np.maximum(distances, 1e-6)

        # Compute the force kernel using power-law repulsion
        y_f = 1 / (self.max_distance ** self.p)
        kernel = (1 / (distances ** self.p) - y_f[:, np.newaxis])
        kernel = self.strength[:, np.newaxis] * np.minimum(np.maximum(kernel, 0), 10)  # Cap forces to avoid instability

        # Compute final repulsion forces
        repulsion = np.sum(kernel[:, :, np.newaxis] * relative_positions, axis=1)

        return repulsion
