from abc import ABC, abstractmethod
from swarmsim.Populations import Population
import numpy as np
import pandas as pd
from typing import Optional
from swarmsim.Utils import get_parameters, load_config


class Interaction(ABC):
    """
    Abstract base class for modeling interactions between agent populations.

    This class provides the framework for implementing various types of inter-agent
    interactions such as repulsion, attraction, alignment, and collision avoidance.
    It defines how one population (source) influences another population (target)
    through force computations.

    Parameters
    ----------
    target_population : Population
        The population that receives the interaction forces.
    source_population : Population
        The population that generates the interaction forces.
    config_path : str
        Path to the YAML configuration file containing interaction parameters.
    name : str, optional
        Name identifier for the interaction. If None, defaults to the class name.

    Attributes
    ----------
    target_population : Population
        The population affected by the interaction forces.
    source_population : Population
        The population generating the interaction forces.
    config : dict
        Configuration parameters specific to this interaction type.
    param_config : dict
        Parameter configuration for interaction-specific parameters.
    id : str
        Identifier for this interaction instance.
    params : dict of np.ndarray or None
        Dictionary containing interaction-specific parameters.
    params_shapes : dict of tuple or None
        Dictionary defining expected shapes for interaction parameters.

    Config Requirements
    -------------------
    The YAML configuration file should contain parameters under the interaction's section:

    id : str, optional
        Identifier for the interaction. Defaults to the class name.
    parameters : dict, optional
        Configuration for interaction-specific parameters.

    Notes
    -----
    - Subclasses must implement the abstract method `get_interaction()`.
    - Interactions are computed at each simulation timestep.
    

    Examples
    --------
    Example YAML configuration for an interaction:

    .. code-block:: yaml

        HarmonicRepulsion:
            id: "repulsion_interaction"
            parameters:
                params_mode: "Random"
                params_names: ["strength", "range"]
                params_limits:
                    strength: [1.0, 5.0]
                    range: [0.5, 2.0]

    Example of implementing a custom interaction:

    .. code-block:: python

        class CustomInteraction(Interaction):
            def get_interaction(self):
                # Compute interaction forces here
                
                return forces
    """

    def __init__(self,
                 target_population: Population,
                 source_population: Population,
                 config_path: str,
                 name: str = None) -> None:
        """
        Initialize the Interaction with target and source populations.

        Parameters
        ----------
        target_population : Population
            The population that will be affected by the interaction forces.
        source_population : Population
            The population that will generate the interaction forces.
        config_path : str
            Path to the YAML configuration file containing interaction parameters.
        name : str, optional
            Name identifier for the interaction. If None, defaults to the class name.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found.
        """

        super().__init__()
        self.target_population: Population = target_population  # The affected population
        self.source_population: Population = source_population  # The interacting population

        config_file = load_config(config_path)

        # Retrieve configuration for the specific population class
        if name is None:
            name = type(self).__name__
        self.config: dict = config_file.get(name)
        self.param_config: dict = self.config.get("parameters")

        self.id: str = self.config.get("id", name)  # Population ID

        # Initialize params, state and inputs
        self.params: Optional[dict[str, np.ndarray]] = None
        self.params_shapes: Optional[dict[str, tuple]] = None

        # self.reset()

    def reset(self) -> None:
        """
        Reset the interaction parameters to their initial values.

        This method reinitializes interaction-specific parameters based on the
        configuration. It should be called before starting a new simulation.
        """
        if self.param_config is not None:
            self.params = get_parameters(self.param_config, self.params_shapes, self.target_population.N)

    @abstractmethod
    def get_interaction(self) -> np.ndarray:
        """
        Compute the interaction forces between source and target populations.

        This abstract method must be implemented by subclasses to define the specific
        interaction. It calculates the forces that the source population exerts
        on the target population based on their current states and interaction parameters.

        Returns
        -------
        np.ndarray
            Array of shape (N_target, state_dim) representing the interaction forces
            applied to each agent in the target population, where N_target is the number
            of agents in the target population and state_dim is the spatial dimension.

        Notes
        -----
        The returned forces will be added to the target population's force accumulator
        and integrated by the numerical integrator.

        Raises
        ------
        NotImplementedError
            If called directly from the base class without implementation.
        """
        pass

