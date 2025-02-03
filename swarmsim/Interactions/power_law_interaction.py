import numpy as np
import yaml
from swarmsim.Interactions import Interaction
from swarmsim.Utils import compute_distances


class PowerLawInteraction(Interaction):
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

    def __init__(self, pop1, pop2, config, repulsion_name) -> None:
        super().__init__(pop1, pop2)

        # Load repulsion parameters from YAML configuration file
        with open(config, "r") as file:
            config = yaml.safe_load(file)
        self.params = config.get(repulsion_name, {})

        # Extract parameters from configuration
        self.strength_attr = self.params["strength_attr"]
        self.strength_rep = self.params["strength_rep"]
        self.max_distance = self.params["max_distance"]
        self.p_attr = self.params["p_attr"]
        self.p_rep = self.params["p_rep"]
        self.is_attractive = self.params["is_attractive"]

    def get_interaction(self):
        """
        Computes the repulsion force exerted by `pop2` on `pop1` using a power-law function.

        The repulsion force is computed as:

            F_repulsion = strength * (1/distance^p - 1/max_distance^p)

        where:
            - `distance` is the Euclidean distance between agents in `pop1` and `pop2`.
            - `max_distance` defines the interaction cutoff beyond which no force is applied.
            - `p` controls how rapidly the force decays with distance.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the repulsion force applied to each
            agent in `pop1`, where `N1` is the number of agents in `pop1` and
            `D` is the dimensionality of the state space.

        Notes
        -----
        - The function prevents division by zero by setting a minimum distance (`1e-6`).
        - The force is capped at `10` to avoid numerical instabilities.
        """

        # Compute pairwise distances and relative positions between agents
        distances, relative_positions = compute_distances(self.pop1.x[:, :2], self.pop2.x[:, :2])

        # Prevent division by zero
        distances = np.maximum(distances, 1e-6)

        # Attraction and repulsion kernel
        shift = (self.strength_rep / (self.max_distance ** self.p_rep) -
                 self.strength_attr / (self.max_distance ** self.p_attr))

        mask = distances <= self.max_distance

        kernel = (self.strength_rep / (distances ** self.p_rep) -
                  self.strength_attr / (distances ** self.p_attr)) - shift

        kernel = mask * kernel

        kernel = np.minimum(kernel, 10)
        if not self.is_attractive:
            kernel = np.maximum(kernel, 0)

        #
        #
        # # Compute the force kernel using power-law repulsion
        # y_f = 1 / (self.max_distance ** self.p)
        # kernel = (1 / (distances ** self.p) - y_f)
        # kernel = self.strength * np.minimum(np.maximum(kernel, 0), 10)  # Cap forces to avoid instability

        # Compute final repulsion forces
        repulsion = np.sum(kernel[:, :, np.newaxis] * relative_positions, axis=1)

        return repulsion
