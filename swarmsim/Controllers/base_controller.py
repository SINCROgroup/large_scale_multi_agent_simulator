from abc import ABC, abstractmethod
from swarmsim.Utils import load_config

from swarmsim.Environments import Environment
from swarmsim.Populations import Population

import numpy as np


class Controller(ABC):
    """
    Abstract base class for implementing control strategies.

    This class provides the interface for controllers that compute control actions to influence
    agent behaviors. Controllers can access population states, environment information, and other
    populations to make informed control decisions.

    Parameters
    ----------
    population : Population
        The target population that this controller will influence.
    environment : Environment, optional
        The environment where the agents operate. Can be None if environmental information
        is not needed for control.
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters.
    other_populations : list of Population, optional
        List of other populations that may influence the control strategy.

    Attributes
    ----------
    population : Population
        The target population being controlled.
    environment : Environment or None
        The environment instance (None if environmental information are not relevant).
    other_populations : list of Population or None
        List of other populations required to compute the control action.
    config : dict
        Configuration parameters specific to this controller class.
    dt : float
        Sampling time interval between two consecutive control actions.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the controller's section:

    dt : float
        Sampling time of the controller (time interval between consecutive control actions).

    Notes
    -----
    - Subclasses must implement the abstract method `get_action()` to define the control strategy.
    - The `get_action_in_space()` method can be optionally overridden for spatially non-uniform control actions.
    - Controllers operate at discrete time intervals specified by `dt`.

    Examples
    --------
    Example YAML configuration for a controller:

    .. code-block:: yaml

        MyController:
            dt: 0.1

    This configuration sets the controller sampling time to 0.1 seconds.
    """

    def __init__(self, population: Population, environment: Environment =None, config_path: str =None, other_populations = None) -> None:
        """
        Initialize the Controller with the target population and optional components.

        Arguments
        ---------
        population : Population
            The population that this controller will influence.
        environment : Environment, optional
            The environment where agents operate. Default is None.
        config_path : str, optional
            Path to the YAML configuration file. Default is None.
        other_populations : list of Population, optional
            Other populations that may influence control decisions. Default is None.

        """
        
        super().__init__()
        self.population: Population = population
        self.environment: Environment = environment
        self.other_populations = other_populations

        config: dict = load_config(config_path)

        # Get configuration for the specific population class
        class_name = type(self).__name__
        self.config = config.get(class_name, {})
        
        self.dt: float = self.config.get('dt')


    @abstractmethod
    def get_action(self) -> np.ndarray:
        """
        Compute the control action for the target population.

        This abstract method must be implemented by subclasses to define the specific
        control strategy. The method should use the current state of the population,
        environment, and other available information to compute appropriate control inputs.

        Returns
        -------
        np.ndarray
            Control action array of shape (N, input_dim) where N is the number of agents
            in the target population and input_dim is the dimensionality of the control input.

        Notes
        -----
        This is the core method that defines the controller behavior and must be overridden in subclasses.

        """
        pass


    def get_action_in_space(self, positions: np.ndarray) -> np.ndarray:
        """
        Computes spatially varying control actions at specified positions.

        This method provides control actions at arbitrary spatial locations. It can be overridden by subclasses that
        implement spatially varying control strategies.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (num_positions, state_dim) specifying the positions where
            control actions should be evaluated.

        Returns
        -------
        np.ndarray
            Control action array of shape (num_positions, input_dim) representing the
            control field values at the specified positions.

        Notes
        -----
        The default implementation returns None. Subclasses should override this method
        if they implement spatially varying control fields.
        """
        pass
