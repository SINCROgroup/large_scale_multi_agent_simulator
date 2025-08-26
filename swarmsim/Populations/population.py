from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from swarmsim.Utils import get_parameters, get_states, load_config


class Population(ABC):
    """
    Abstract base class that defines the structure for agent populations in a multi-agent system.

    This class provides the foundational framework for implementing different types of agent populations
    with configurable initial conditions and parameters. It handles the loading of configuration files,
    initialization of agent states, and provides abstract methods for defining population dynamics.

    Parameters
    ----------
    config : str
        Path to a YAML configuration file.
    name : str, optional
        Name identifier for the population. If None, defaults to the class name.

    Attributes
    ----------
    id : str
        Identifier for the population instance.
    config : str or dict
        Either a path to a YAML configuration file containing population parameters,
        or a dictionary with configuration parameters.
    init_config : dict
        Configuration parameters specifically for initial conditions.
    param_config : dict or None
        Configuration parameters for population-specific parameters.
    N : int
        Number of agents in the population.
    state_dim : int
        Dimensionality of the state space for each agent.
    input_dim : int
        Dimensionality of the input space for each agent. Defaults to `state_dim` if not specified.
    lim_i : np.ndarray
        Lower limits for the state of each agent. Defaults to ``['-inf']``.
    lim_s : np.ndarray
        Upper limits for the state of each agent. Defaults to ``['inf']``.
    params : dict of np.ndarray or None
        Dictionary containing population-specific parameters for each agent.
    params_shapes : dict of tuple or None
        Dictionary defining the expected shapes for each parameter.
    x : np.ndarray or None
        Current state of all agents, shape (N, state_dim).
    u : np.ndarray or None
        Current control input for all agents, shape (N, input_dim).
    f : np.ndarray or None
        Current environmental forces acting on all agents, shape (N, input_dim).

    Config Requirements
    -------------------
    The configuration must contain the following required parameters:

    N : int
        Number of agents in the population.
    state_dim : int
        Dimensionality of the state space.

    input_dim : int, optional
        Dimensionality of the input space. Defaults to `state_dim`.
    lim_i : list of float, optional
        Lower limits for the state of each agent. Defaults to ``['-inf']``.
    lim_s : list of float, optional
        Upper limits for the state of each agent. Defaults to ``['inf']``.

    initial_conditions : dict, optional
        Configuration for initial agent states. See `get_states` utility function for details.
    parameters : dict, optional
        Configuration for population-specific parameters. See `get_parameters` utility function for details.

    Notes
    -----
    - Subclasses must implement the abstract methods `get_drift()` and `get_diffusion()`.
    - The initial_conditions dictionary handles specifying initial conditions using various modes:

      * ``"random"``: Generate random initial positions
      * ``"file"``: Load initial positions from a CSV file
      
    - Initial condition shapes can be:
      
      * ``"box"``: Uniform distribution within specified limits
      * ``"circle"``: Uniform distribution within a circular region

    - The parameters dictionary handles specifying population-specific parameters using various modes:

      * ``"random"``: Generate random parameter values
      * ``"file"``: Load parameter values from a CSV file

    Examples
    --------
    Example YAML configuration for a population:

    .. code-block:: yaml

        MyPopulation:
            N: 100
            state_dim: 2
            input_dim: 2
            initial_conditions:
                mode: "random"
                random:
                    shape: "box"
                    box:
                        lower_bounds: [0, 0]
                        upper_bounds: [10, 10]
            parameters:
                mode: "file"
                file:
                    file_path: "path/to/parameter/file.csv"

    This configuration creates a population of 100 agents in a 2D space with random initial
    positions in a 10x10 box and parameters defined in the path/to/parameter/file.csv file.
    """

    def __init__(self, config: str | dict, name: str = None) -> None:
        """
        Initialize the population using the parameters specified in the configuration dictionary or file.

        Parameters
        ----------
        config : str or dict
            Either a path to a YAML configuration file containing population parameters,
            or a dictionary with configuration parameters.
        name : str, optional
            Name identifier for the population. If None, defaults to the class name.

        Raises
        ------
        TypeError
            If config is neither a string filepath nor a dictionary.
        ValueError
            If required parameters 'N' or 'state_dim' are missing from the configuration.
        """
        super().__init__()

        if name is None:
            name = type(self).__name__

        self.id: str = name

        # If config is a filepath, load it as a dictionary
        if isinstance(config, str):
            config = load_config(config)

        # Validate config type early
        if not isinstance(config, dict):
            raise TypeError(f"Expected config to be dict or str filepath, got {type(config).__name__}")

        # Extract the specific configuration for this class instance
        self.config: dict = config.get(self.id, {})
        self.init_config: dict = self.config.get("initial_conditions", {})
        self.param_config: dict | None = self.config.get("parameters", None)

        # Load primary configuration parameters, with sensible defaults or clear errors
        self.N: int = self.config.get("N")
        self.state_dim: int = self.config.get("state_dim")

        if self.N is None or self.state_dim is None:
            raise ValueError(f"'N' and 'state_dim' must be specified in the config for '{self.id}'.")

        # input_dim defaults to state_dim if unspecified
        self.input_dim: int = self.config.get("input_dim", self.state_dim)

        # Limit configuration, with a clear default (no limit: inf)
        lim_values = self.config.get('lim_i', ['-inf'])
        self.lim_i: np.ndarray = np.array([float(value) for value in lim_values])
        lim_values = self.config.get('lim_s', ['inf'])
        self.lim_s: np.ndarray = np.array([float(value) for value in lim_values])

        # Initialize params, states, inputs, and dynamics explicitly
        self.params: Optional[dict[str, np.ndarray]] = None
        self.params_shapes: Optional[dict[str, tuple]] = None

        self.x: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None

    def reset(self) -> None:
        """
        Reset agent parameters, states, and control inputs.

        This method reinitializes all population parameters, agent states, control inputs,
        and external forces based on the configuration. It should be called before starting
        a new simulation or when resetting the simulation state.

        Notes
        -----
        - Regenerates parameters
        - Resets agent states to initial conditions
        - Resets to 0 control inputs and external forces
        """
        if self.param_config is not None:
            self.params = get_parameters(self.param_config, self.params_shapes, self.N)

        self.x = get_states(self.init_config, self.N, self.state_dim)
        self.u = np.zeros([self.N, self.input_dim])
        self.f = np.zeros([self.N, self.input_dim])

    @abstractmethod
    def get_drift(self) -> np.ndarray:
        """
        Compute the deterministic drift component of the population dynamics.

        This abstract method must be implemented by subclasses to define the deterministic
        part of the agent dynamics. The drift typically includes intrinsic motion patterns,
        external forces, and control inputs.

        Returns
        -------
        np.ndarray
            Array of shape (N, state_dim) representing the drift for each agent.

        Notes
        -----
        Subclasses should implement this method to return the drift term in the stochastic
        differential equation: dx = drift * dt + diffusion * dW, where dW is a Wiener process.
        """
        pass

    @abstractmethod
    def get_diffusion(self) -> np.ndarray:
        """
        Compute the stochastic diffusion component of the population dynamics.

        This abstract method must be implemented by subclasses to define the stochastic
        part of the agent dynamics.

        Returns
        -------
        np.ndarray
            Array of shape (N, state_dim) representing the diffusion component for each agent.

        Notes
        -----
        Subclasses should implement this method to return the diffusion term in the stochastic
        differential equation: dx = drift * dt + diffusion * dW, where dW is a Wiener process.
        """
        pass

