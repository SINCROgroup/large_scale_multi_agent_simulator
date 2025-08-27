import numpy as np
from swarmsim.Interactions import Interaction
from swarmsim.Populations import Population
from swarmsim.Utils import compute_distances
from typing import Optional


class PowerLawRepulsion(Interaction):
    """
    Pure power-law repulsion interaction with finite range cutoff.

    This interaction implements a repulsive force that decays according to an inverse
    power law with respect to inter-agent distance. Unlike the more general PowerLawInteraction,
    this class focuses specifically on repulsion with a clean cutoff mechanism that ensures
    zero force at the maximum interaction distance.

    The repulsion force follows:

    .. math::

        F(r) = k \\left( \\frac{1}{r^p} - \\frac{1}{r_{max}^p} \\right)

    for :math:`r < r_{max}`, and :math:`F(r) = 0` for :math:`r \\geq r_{max}`, where:
    - :math:`k` is the repulsion strength parameter
    - :math:`r` is the inter-agent distance
    - :math:`p` is the power law exponent
    - :math:`r_{max}` is the maximum interaction distance

    The subtraction term ensures the force smoothly approaches zero at the cutoff distance,
    preventing discontinuous force jumps that can cause numerical instabilities.

    Parameters
    ----------
    target_population : Population
        The population that receives repulsion forces.
    source_population : Population
        The population that generates repulsion forces.
    config_path : str
        Path to the YAML configuration file containing interaction parameters.
    name : str, optional
        Name identifier for the interaction. Defaults to class name if None.

    Attributes
    ----------
    target_population : Population
        Population affected by power-law repulsion forces.
    source_population : Population
        Population generating power-law repulsion forces.
    strength : np.ndarray or None
        Repulsion strength parameter for each agent in the target population.
    max_distance : np.ndarray or None
        Maximum interaction distance for each agent in the target population.
    p : int
        Power law exponent controlling the decay rate of repulsion.
        Higher values create steeper decay (stronger short-range repulsion).
    params_shapes : dict
        Defines expected shapes for interaction parameters.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the interaction's section:

    parameters : dict
        Parameter configuration for the power-law repulsion:

    p : int
        Power law exponent for the repulsion force. Common values:


    Notes
    -----
    **Force Characteristics:**

    - **Monotonic Decay**: Force decreases monotonically with distance
    - **Finite Range**: Zero force beyond ``max_distance``
    - **Customizable Steepness**: Power exponent controls interaction range vs strength


    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required interaction parameters are missing in the configuration file.
    ValueError
        If power exponent p is not positive or if parameter shapes are incompatible.

    Examples
    --------
    **Soft Repulsion Configuration:**

    .. code-block:: yaml

        PowerLawRepulsion:
            id: "soft_repulsion"
            p: 2
            parameters:
                mode: "Fixed"
                names: ["strength", "max_distance"]
                values:
                    strength: 1.0
                    max_distance: 3.0

    
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
        Compute power-law repulsion forces between source and target populations.

        This method calculates repulsive forces using the power-law formula with a smooth
        cutoff mechanism. The computation handles distance calculations, applies the power-law
        kernel, and includes numerical stability measures to prevent instabilities.

        Returns
        -------
        np.ndarray
            Repulsion forces array of shape ``(N_target, 2)`` where:
            - ``N_target`` is the number of agents in the target population
            - Each row represents the total repulsion force on one target agent in 2D space
            Force vectors point away from source agents (repulsive direction).

        Notes
        -----
        **Algorithm Steps:**

        1. **Distance Computation**: Calculate pairwise distances between all source-target pairs
        2. **Cutoff Application**: Apply smooth cutoff using reference force at max_distance
        3. **Kernel Evaluation**: Compute power-law repulsion kernel with bounds
        4. **Force Assembly**: Convert scalar forces to vector forces using relative positions
        5. **Force Summation**: Sum contributions from all source agents

        **Mathematical Implementation:**

        The force kernel is computed as:

        .. math::

            K(r) = k \\cdot \\text{clip}\\left( \\frac{1}{r^p} - \\frac{1}{r_{max}^p}, 0, F_{max} \\right)

        where:
        - :math:`k` is the strength parameter
        - :math:`r` is the inter-agent distance (with minimum threshold)
        - :math:`p` is the power law exponent
        - :math:`r_{max}` is the maximum interaction distance
        - :math:`F_{max} = 1000` is the force magnitude cap

        
        """

        # Compute pairwise distances and relative positions between agents
        distances, relative_positions = compute_distances(self.target_population.x[:, :2], self.source_population.x[:, :2])

        # Prevent division by zero
        distances = np.maximum(distances, 1e-6)

        # Compute the force kernel using power-law repulsion
        y_f = 1 / (self.max_distance ** self.p)
        kernel = (1 / (distances ** self.p) - y_f[:, np.newaxis])
        kernel = self.strength[:, np.newaxis] * np.minimum(np.maximum(kernel, 0), 1000)  # Cap forces to avoid instability

        # Compute final repulsion forces
        repulsion = np.sum(kernel[:, :, np.newaxis] * relative_positions, axis=1)

        return repulsion
