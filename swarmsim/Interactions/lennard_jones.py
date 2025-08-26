import numpy as np
from swarmsim.Interactions import Interaction
from swarmsim.Utils import compute_distances
from swarmsim.Populations import Population
from typing import Optional


class LennardJones(Interaction):
    """
    Lennard-Jones interaction potential for agent-agent interactions.

    This interaction implements the classic Lennard-Jones potential, which models
    both attractive and repulsive forces between agents. The potential combines
    a strong short-range repulsion (r^-12 term) with a weaker long-range attraction
    (r^-6 term), commonly used to model molecular interactions and agent clustering.

    The interaction force deriving from a Lennard-Jones potential is:

    .. math::

        F(r) = 24\\epsilon \\frac{1}{r^2} \\left[ 2\\left(\\frac{\\sigma}{r}\\right)^{12} - \\left(\\frac{\\sigma}{r}\\right)^6 \\right]

    where:
    - :math:`\\epsilon` is the depth of the potential well (energy scale)
    - :math:`\\sigma` is the distance at which the potential is zero (length scale)
    - :math:`r` is the distance between agents

    Parameters
    ----------
    target_population : Population
        The population that receives Lennard-Jones forces.
    source_population : Population
        The population that generates Lennard-Jones forces.
    config_path : str
        Path to the YAML configuration file containing interaction parameters.
    name : str, optional
        Name identifier for the interaction. Defaults to class name if None.

    Attributes
    ----------
    target_population : Population
        Population affected by Lennard-Jones forces.
    source_population : Population
        Population generating Lennard-Jones forces.
    epsilon : np.ndarray or None
        Energy scale parameter(s), shape depends on parameter configuration.
    sigma : np.ndarray or None
        Length scale parameter(s), shape depends on parameter configuration.
    params_shapes : dict
        Defines expected shapes for interaction parameters.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the interaction's section:

    parameters : dict
        Parameter configuration for the Lennard-Jones interaction. Typical structure:
        
        - ``mode`` : str - Parameter assignment mode (e.g., "Random", "Fixed")
        - ``names`` : list - List of parameter names ["epsilon", "sigma"]
        - ``limits`` or ``values`` : dict - Parameter ranges or fixed values

    Notes
    -----
    **Force Characteristics:**

    - **Equilibrium Distance**: Force is zero at :math:`r = 2^{1/6}\\sigma \\approx 1.122\\sigma`
    - **Attractive Region**: :math:`r > 2^{1/6}\\sigma` (negative force, particles attract)
    - **Repulsive Region**: :math:`r < 2^{1/6}\\sigma` (positive force, particles repel)
    - **Long-Range**: Force decays as :math:`r^{-7}` for large distances

    
    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required interaction parameters are missing in the configuration file.
    ValueError
        If parameter shapes are incompatible with population sizes.
    """

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
        """
        Compute Lennard-Jones interaction forces between source and target populations.

        This method calculates the attractive and repulsive forces between agents
        according to the Lennard-Jones potential. The computation handles both
        the attractive long-range and repulsive short-range components of the interaction.

        Returns
        -------
        np.ndarray
            Lennard-Jones forces array of shape ``(N_target, 2)`` where:
            - ``N_target`` is the number of agents in the target population
            - Each row represents the total LJ force on one target agent in 2D space
            Force vectors point away from source agents for repulsion, toward them for attraction.

        Notes
        -----
        **Algorithm Steps:**

        1. **Distance Computation**: Calculate pairwise distances between all source-target pairs
        2. **Force Magnitude**: Evaluate Lennard-Jones force equation for each pair
        3. **Force Direction**: Compute unit vectors for force direction
        4. **Vector Forces**: Combine magnitude and direction for vector forces
        5. **Force Summation**: Sum contributions from all source agents

        """
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
