from abc import ABC, abstractmethod
from swarmsim.Populations import Populations
import numpy as np
import pandas as pd
from typing import Optional
from swarmsim.Utils import get_parameters, load_config


class Interaction(ABC):
    """
    Abstract base class that defines the structure for interactions between two populations.

    This class serves as an interface for all interaction models in a multi-agent system.
    It requires subclasses to implement the `get_interaction` method, which computes
    the effect that `pop2` (e.g., herders) has on `pop1` (e.g., targets).

    Parameters
    ----------
    target_population : Population
        The first population that is influenced by the interaction.
    source_population : Population
        The second population that applies the interaction force.

    Attributes
    ----------
    target_population : Population
        The population affected by the interaction.
    source_population : Population
        The population exerting the interaction force.

    Notes
    -----
    - The `get_interaction` method must be implemented in all subclasses.
    - This class is designed for interactions such as **repulsion, attraction,** and **alignment**.

    Examples
    --------
    Example of a subclass implementing a specific interaction:

    .. code-block:: python

        class HarmonicRepulsion(Interaction):
            def get_interaction(self):
                # Compute repulsion forces here
                return forces
    """

    def __init__(self,
                 target_population: Populations,
                 source_population: Populations,
                 config_path: str,
                 name: str = None) -> None:

        super().__init__()
        self.target_population: Populations = target_population  # The affected population
        self.source_population: Populations = source_population  # The interacting population

        config_file = load_config(config_path)

        # Retrieve configuration for the specific population class
        if name is None:
            name = type(self).__name__
        self.config: dict = config_file.get(name)
        self.param_config: dict = self.config.get("parameters")

        self.id: str = self.config.get("id", name)  # Population ID

        # Initialize params, state and inputs
        self.params: Optional[pd.DataFrame] = None

        self.reset()

    @abstractmethod
    def get_interaction(self) -> np.ndarray:
        """
        Computes the forces that `source_population` applies on `target_population`.

        This method must be implemented by subclasses to define the specific
        interaction between the two populations.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the forces exerted by `source_population` on `target_population`,
            where `N1` is the number of agents in `target_population` and `D` is the state space dimension.

        Raises
        ------
        NotImplementedError
            If called directly from the base class.
        """
        pass

    def reset(self) -> None:
        if self.params is None and self.param_config is not None:
            self.params = get_parameters(self.param_config, self.target_population.N)
