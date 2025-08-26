import numpy as np
import yaml
from swarmsim.Interactions import Interaction
from swarmsim.Populations import Population
from typing import Optional

class HarmonicRepulsion(Interaction):
    """
    Harmonic repulsion interaction with finite interaction range.

    This interaction implements a linearly decaying repulsive force between agents
    from two different populations. The repulsion follows a harmonic potential that
    provides short-range repulsion up to a finite interaction range.

    The repulsion force magnitude follows:

    .. math::
        F_i = \\sum_{j \\in population} F_{ij}(r_{ij})

        F_{ij}(r_{ij}) = \\begin{cases}
        -k (r_{max} - r_{ij}) \\hat{r}_{ij} & \\text{if } r_{ij} < r_{max} \\\\
        0 & \\text{if } r_{ij} \\geq r_{max}
        \\end{cases}

    where:
    - :math:`F_i` is the total force on agent i
    - :math:`k` is the strength parameter
    - :math:`r_{max}` is the maximum interaction distance
    - :math:`r_{ij}` is the distance between agents i and j
    - :math:`\\hat{r}_{ij}` is the unit vector joining agents i and j

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
        Population affected by repulsion forces.
    source_population : Population
        Population generating repulsion forces.
    strength : np.ndarray or None
        Repulsion strength parameter(s), shape depends on parameter configuration.
    distance : np.ndarray or None
        Maximum interaction distance parameter(s), shape depends on parameter configuration.
    params_shapes : dict
        Defines expected shapes for interaction parameters.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the interaction's section:

    strength : float or list 
        Repulsion strength parameter(s)
    distance : float or list 
        Maximum interaction distance parameter(s)
    parameters : dict
        Parameter configuration for the harmonic repulsion. Typical structure:
        
        - ``mode`` : str - Parameter assignment mode
        
    Notes
    -----
    **Force Characteristics:**

    - **Linear Decay**: Force decreases linearly with distance until cutoff
    - **Finite Range**: No interaction beyond maximum distance
    - **Pairwise Computation**: All source-target agent pairs are evaluated
    - **Vectorized**: Efficient computation using NumPy broadcasting

    **Parameter Flexibility:**

    Parameters can be configured as:
    - **Fixed values**: Same for all interactions
    - **Random ranges**: Sampled for each agent or interaction
    - **Population-dependent**: Different values for different populations

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required interaction parameters are missing in the configuration file.
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
        Compute harmonic repulsion forces between source and target populations.

        This method calculates the repulsive forces that source population agents
        exert on target population agents using the harmonic repulsion model.
        Forces decay linearly with distance and have finite interaction range.

        Returns
        -------
        np.ndarray
            Repulsion forces array of shape ``(N_target, D)`` where:
            - ``N_target`` is the number of agents in the target population
            - ``D`` is the spatial dimension (typically 2 or 3)
            Each row represents the total repulsion force on one target agent.

        Notes
        -----
        **Algorithm Steps:**

        1. **Distance Computation**: Calculate pairwise distances between all source-target pairs
        2. **Range Filtering**: Identify agent pairs within interaction range
        3. **Force Calculation**: Apply harmonic repulsion formula for nearby pairs
        4. **Force Summation**: Sum contributions from all source agents for each target

        **Numerical Stability:**

        - Minimum distance threshold (``1e-6``) prevents division by zero
        - Vectorized operations ensure computational efficiency
        - Broadcasting handles different parameter shapes automatically

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
