import numpy as np
import yaml
from swarmsim.Interactions import Interaction
from swarmsim.Utils import compute_distances
from swarmsim.Populations import Population
from typing import Optional


class PowerLawInteraction(Interaction):
    """
    Power-law interaction with configurable attraction and repulsion components.

    This interaction models a repulsive force that decays according to a power-law function
    with respect to the distance between agents. The strength of the force is determined by
    the power exponent `p` and is active within a defined maximum interaction range.

    Parameters
    ----------
    target_population : Population
        The population that receives interaction forces.
    source_population : Population
        The population that generates interaction forces.
    config_path : str
        Path to the YAML configuration file containing interaction parameters.
    name : str, optional
        Name identifier for the interaction. Defaults to class name if None.

    Attributes
    ----------
    target_population : Population
        Population affected by power-law forces.
    source_population : Population
        Population generating power-law forces.
    strength_attr : np.ndarray or None
        Attraction strength parameter(s).
    strength_rep : np.ndarray or None
        Repulsion strength parameter(s).
    max_distance : np.ndarray or None
        Maximum interaction distance parameter(s).
    p_attr : int
        Power exponent for attraction term.
    p_rep : int
        Power exponent for repulsion term.
    is_attractive : bool
        Whether to allow attractive forces (negative values).
    params_shapes : dict
        Defines expected shapes for interaction parameters.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the interaction's section:

    parameters : dict
        Parameter configuration for the power-law interaction:
        
        - ``mode`` : str - Parameter assignment mode
        - ``names`` : list - Parameter names ["strength_attr", "strength_rep", "max_distance"]
        - ``limits`` or ``values`` : dict - Parameter ranges or fixed values

    p_attr : int
        Power exponent for the attraction term (typically 6-8).
    p_rep : int  
        Power exponent for the repulsion term (typically 12-16).
    is_attractive : bool
        Whether to allow attractive forces. If False, forces are clipped to non-negative values.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required interaction parameters are missing in the configuration file.
    ValueError
        If parameter shapes are incompatible with population sizes.

    Examples
    --------
    **Lennard-Jones-like Configuration:**

    .. code-block:: yaml

        PowerLawInteraction:
            id: "lj_like_interaction"
            p_attr: 6
            p_rep: 12
            is_attractive: true
            parameters:
                params_mode: "Fixed"
                params_names: ["strength_attr", "strength_rep", "max_distance"]
                params_values:
                    strength_attr: 1.0
                    strength_rep: 4.0
                    max_distance: 5.0

    **Pure Repulsion Configuration:**

    .. code-block:: yaml

        PowerLawInteraction:
            id: "repulsion_only"
            p_attr: 6
            p_rep: 3
            is_attractive: false
            parameters:
                params_mode: "Fixed"
                params_names: ["strength_attr", "strength_rep", "max_distance"]
                params_values:
                    strength_attr: 0.0
                    strength_rep: 2.0
                    max_distance: 3.0

    **Usage in Simulation:**

    .. code-block:: python

        from swarmsim.Interactions import PowerLawInteraction
        from swarmsim.Populations import BrownianMotion

        # Create populations
        agents = BrownianMotion('agent_config.yaml')
        
        # Create power-law self-interaction
        interaction = PowerLawInteraction(
            target_population=agents,
            source_population=agents,
            config_path='powerlaw_config.yaml'
        )
        
        # Initialize parameters
        interaction.reset()
        
        # Compute forces
        forces = interaction.get_interaction()

    **Force Profile Analysis:**

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        
        # Plot force vs distance
        r = np.linspace(0.5, 5.0, 100)
        k_rep, k_attr = 4.0, 1.0
        p_rep, p_attr = 12, 6
        
        force = k_rep / r**p_rep - k_attr / r**p_attr
        
        plt.plot(r, force)
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Distance')
        plt.ylabel('Force')
        plt.title('Power-Law Force Profile')

    **Equilibrium Distance Calculation:**

    The equilibrium distance where force is zero occurs when:

    .. math::

        \\frac{k_{rep}}{r_{eq}^{p_{rep}}} = \\frac{k_{attr}}{r_{eq}^{p_{attr}}}

    Solving for :math:`r_{eq}`:

    .. math::

        r_{eq} = \\left(\\frac{k_{rep}}{k_{attr}}\\right)^{\\frac{1}{p_{rep} - p_{attr}}}
    """


    def __init__(self,
                 target_population: Population,
                 source_population: Population,
                 config_path: str,
                 name: str = None) -> None:

        super().__init__(target_population, source_population, config_path, name)

        self.strength_attr: Optional[np.ndarray] = None
        self.strength_rep: Optional[np.ndarray] = None
        self.max_distance: Optional[np.ndarray] = None

        self.params_shapes = {
            "strength_attr": (),
            "strength_rep": (),
            "max_distance": ()
        }


        self.p_attr: int = self.config.get("p_attr")
        self.p_rep: int = self.config.get("p_rep")
        self.is_attractive: bool = self.config.get("is_attractive")

    def reset(self):
        super().reset()
        
        self.strength_attr = self.params['strength_attr']
        self.strength_rep = self.params['strength_rep']
        self.max_distance = self.params.get('max_distance', None)

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

        # Attraction and repulsion kernel
        if self.max_distance is not None:
            shift = (self.strength_rep / (self.max_distance ** self.p_rep) -
                     self.strength_attr / (self.max_distance ** self.p_attr))
        else:
            shift = 0

        kernel = (self.strength_rep[:, np.newaxis] / (distances ** self.p_rep) -
                  self.strength_attr[:, np.newaxis] / (distances ** self.p_attr)) - shift[:, np.newaxis]

        if self.max_distance is not None:
            mask = distances <= self.max_distance[:, np.newaxis]
            kernel = mask * kernel

        kernel = np.minimum(kernel, 1000)
        if not self.is_attractive:
            kernel = np.maximum(kernel, 0)

        # Compute final repulsion forces
        repulsion = np.sum(kernel[:, :, np.newaxis] * relative_positions, axis=1)

        return repulsion
