from abc import ABC, abstractmethod
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

from swarmsim.Utils import get_parameters, get_states

# Configure logging for debugging and diagnostics.
logging.basicConfig(level=logging.DEBUG)


class Populations(ABC):
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

    def __init__(self, config_path: str, name: str = None) -> None:
        super().__init__()
        self.config_path: str = config_path

        # Verify that the configuration file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, "r") as file:
            config_file = yaml.safe_load(file)

        # Retrieve configuration for the specific population class
        if name is None:
            name = type(self).__name__
        self.config: dict = config_file.get(name)
        self.init_config: dict = self.config.get("initial_conditions")
        self.param_config: dict = self.config.get("parameters")

        self.id: str = self.config.get("id", name)  # Population ID

        # Load primary configuration settings
        self.N: int = self.config.get("N")
        self.state_dim: int = self.config.get("state_dim")
        self.input_dim: int = self.config.get("input_dim", self.state_dim)  # default if not provided
        self.lim: np.ndarray = np.asarray(list(map(float, self.config.get('lim', ['inf']))))

        # Initialize params, state and inputs
        self.params: Optional[pd.DataFrame] = None

        self.x: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None

        self.reset()

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

    def reset(self) -> None:
        """
        Reset agent parameters, initial conditions, and control forces.
        """
        if self.params is None and self.param_config is not None:
            self.params = get_parameters(self.param_config, self.N)
        self.x = self.get_initial_conditions()
        self.u = np.zeros([self.N, self.input_dim])
        self.f = np.zeros([self.N, self.input_dim])

    def get_initial_conditions(self) -> np.ndarray:
        """
        Loads or generates the initial conditions for the population.
        Depending on the mode setting, the conditions are either loaded from file or generated randomly.
        """

        states = get_states(self.init_config, self.N, self.state_dim)

        # Warning for inconsistent number of agents
        if states.shape[0] != self.N:
            logging.warning(
                f"CSV file has {states.shape[0]} agents; updating N from {self.N} to {states.shape[0]}.")
            self.N = states.shape[0]

        # Warning for inconsistent number of states
        if states.shape[1] != self.state_dim:
            logging.warning(
                f"CSV file state_dim {states.shape[1]} differs from configured state_dim {self.state_dim}; updating accordingly.")
            self.state_dim = states.shape[1]

        return states
