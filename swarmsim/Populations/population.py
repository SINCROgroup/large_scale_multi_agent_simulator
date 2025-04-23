from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from swarmsim.Utils import get_parameters, get_states, load_config


class Population(ABC):
    """
    Abstract base class that defines the structure for agent populations in a multi-agent system.

    This class provides methods for initializing agent states and parameters from either
    a configuration file or random values.

    Configuration Example (YAML):

    PopulationClassName:
        x0_mode: "Random"                 # "Random" or "From_File"
        x0_shape: "box"                   # "box" or "circle" (only for Random)
        x0_limits:
            - [0, 1]
            - [0, 1]
            - [-1, 1]
        max_initial_radius: 25            # Used for "circle" mode
        min_initial_radius: 0             # Used for "circle" mode
        extra_x0_limits:                  # For dimensions beyond the first two in "circle" mode
            - [0, 1]
        params_mode: "Random"             # "Random", "RandomNormal", or "From_File"
        params_names: ["speed", "size"]
        params_limits:                    # Uniform random limits (for Random mode)
            speed: [0.5, 2.0]
            size: [1.0, 5.0]
        params_values:                    # Mean and std (for RandomNormal mode)
            speed: [0.5, 0.2]
            size: [3.0, 1.0]
        N: 50
        state_dim: 3
        input_dim: 3
        lim: [inf]
    """

    def __init__(self, config: str | dict, name: str = None) -> None:
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
        lim_values = self.config.get('lim', ['inf'])
        self.lim: np.ndarray = np.array([float(value) for value in lim_values])

        # Initialize params, states, inputs, and dynamics explicitly
        self.params: Optional[dict[str, np.ndarray]] = None
        self.params_shapes: Optional[dict[str, tuple]] = None

        self.x: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None

    def reset(self) -> None:
        """
        Reset agent parameters, initial conditions, and control forces.
        """
        if self.param_config is not None:
            self.params = get_parameters(self.param_config, self.params_shapes, self.N)

        self.x = get_states(self.init_config, self.N, self.state_dim)
        self.u = np.zeros([self.N, self.input_dim])
        self.f = np.zeros([self.N, self.input_dim])

    @abstractmethod
    def get_drift(self) -> np.ndarray:
        """
        Abstract method to compute the drift term for the population.
        """
        pass

    @abstractmethod
    def get_diffusion(self) -> np.ndarray:
        """
        Abstract method to compute the diffusion term for the population.
        """
        pass

