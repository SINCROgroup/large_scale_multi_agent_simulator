from abc import ABC, abstractmethod
import numpy as np
import yaml
import pandas as pd
from pathlib import Path


class Populations(ABC):
    """
    Abstract base class that defines the structure for agent populations in a multi-agent system.

    This class provides methods for initializing agent states and parameters from either
    a configuration file or random values.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population initialization settings.

    Attributes
    ----------
    config_path : str
        Path to the YAML configuration file.
    config : dict
        Dictionary containing the population's configuration parameters.
    N : int
        Number of agents in the population (set in `get_initial_conditions`).
    state_dim : int
        Dimensionality of the agent state space (set in `get_initial_conditions`).
    x : np.ndarray
        Initial conditions of the agents (set in `get_initial_conditions`).
    params : pd.DataFrame
        Parameters of the population (set in `get_parameters`).

    Config requirements
    -------------------
    x0_mode : str
        Mode for generating initial conditions. Options:
            - ``"From_File"`` : Load from a CSV file.
            - ``"Random"`` : Generate randomly.
    x0_path : str
        Path to the CSV file containing initial conditions.
    N : int
        Number of agents in the population.
    state_dim : int
        Dimensions of the agent state space.
    x0_limits : list of list
        Limits for uniform sampling of initial states, with shape ``(state_dim, 2)``.
    params_mode : str
        Mode for generating parameters. Options:
            - ``"From_File"`` : Load from a CSV file.
            - ``"Random"`` : Generate randomly.
    params_path : str
        Path to the CSV file containing parameter values.
    params_names : list of str
        Names of parameters.
    params_limits : dict
        Dictionary mapping parameter names to their uniform sampling limits.


    Raises
    ------
    FileNotFoundError
        If the specified configuration file or CSV files are not found.
    KeyError
        If required initialization parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        PopulationClassName:
            x0_mode: "Random"
            N: 50
            state_dim: 3
            x0_limits:
                - [0, 1]
                - [0, 1]
                - [-1, 1]
            params_mode: "Random"
            params_names: ["speed", "size"]
            params_limits:
                speed: [0.5, 2.0]
                size: [1.0, 5.0]

    This will create a population of 50 agents with 3D state space and two parameters
    (`speed` and `size`).
    """

    def __init__(self, config_path) -> None:
        super().__init__()
        self.config_path = config_path

        # Load configuration from YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Get configuration for the specific population class
        class_name = type(self).__name__
        self.config = config.get(class_name, {})

        # Initialize state and parameters
        self.N = None  # Number of agents (set in get_initial_conditions)
        self.state_dim = None  # State dimensionality (set in get_initial_conditions)
        self.lim = np.asarray(list(map(float, self.config.get('lim', ['inf']))))
        self.x = self.get_initial_conditions()
        self.params = self.get_parameters()

    @abstractmethod
    def get_drift(self):
        """
        Abstract method to compute the drift term for the population.

        This method should be implemented in subclasses to define how agents' states evolve
        over time in a deterministic manner.

        Returns
        -------
        np.ndarray
            Array of shape `(N, state_dim)` representing the drift term for each agent.
        """
        pass

    @abstractmethod
    def get_diffusion(self):
        """
        Abstract method to compute the diffusion term for the population.

        This method should be implemented in subclasses to define the stochastic component
        of the agents' state evolution.

        Returns
        -------
        np.ndarray
            Array of shape `(N, state_dim)` representing the diffusion term for each agent.
        """
        pass

    @abstractmethod
    def reset_state(self):
        """
        Abstract method to reset the state of the population.

        This method should be implemented in subclasses to reset agent states
        to their initial conditions or another predefined state.
        """
        pass

    def get_initial_conditions(self) -> np.ndarray:
        """
        Loads or generates the initial conditions for the population.

        Depending on the configuration, this method either:
        - Loads initial conditions from a CSV file (`From_File` mode).
        - Generates random initial conditions (`Random` mode).

        Returns
        -------
        np.ndarray
            Array of shape `(N, state_dim)`, where `N` is the number of agents and
            `state_dim` is the dimensionality of the agent's state space.

        Raises
        ------
        RuntimeError
            If an invalid initialization mode is specified in the configuration.

        Notes
        -----
        - If using `"From_File"`, the CSV file should have `N` rows (agents) and `state_dim` columns (state variables).
        - If using `"Random"`, values are drawn uniformly within the specified limits.
        """

        # Retrieve the initialization mode (default: "Random")
        x0_load_type = self.config.get('x0_mode', "Random")

        match x0_load_type:
            case "From_File":
                # Load initial conditions from a CSV file
                x0_path = self.config.get("x0_path", "")  # File path from YAML config
                self.x0 = pd.read_csv(x0_path, header=None).values  # Read CSV into NumPy array

                # Set population size and state dimensionality
                self.N = self.x0.shape[0]
                self.state_dim = self.x0.shape[1]

            case "Random":
                x0_shape = self.config.get("x0_shape", "box")

                self.N = self.config.get("N")  # Number of agents
                self.state_dim = self.config.get("state_dim")  # State space dimensionality

                # Initialize state array
                self.x0 = np.zeros([self.N, self.state_dim])

                if x0_shape == "box":
                    # Generate random initial conditions
                    limits = np.array(self.config.get("x0_limits", []))  # Get sampling limits

                    # Assign random values within specified limits for each state variable
                    for i in range(min(limits.shape[0], self.state_dim)):
                        self.x0[:, i] = np.random.uniform(limits[i, 0], limits[i, 1], self.N)

                if x0_shape == "circle":
                    max_radius = self.config.get("max_initial_radius", 25)
                    agent_radii = np.sqrt(np.random.uniform(0, 1, self.N)) * max_radius
                    agent_angles = np.random.uniform(0, 2 * np.pi, self.N)

                    self.x0[:, 0] = agent_radii * np.cos(agent_angles)
                    self.x0[:, 1] = agent_radii * np.sin(agent_angles)

            case _:
                raise RuntimeError("Invalid initialization type. Check the YAML config file.")

        return self.x0

    def get_parameters(self) -> pd.DataFrame:
        """
        Loads or generates the parameters of the population.

        Depending on the configuration, this method either:
        - Loads parameters from a CSV file (`From_File` mode).
        - Generates random parameters (`Random` mode).

        Returns
        -------
        pd.DataFrame
            DataFrame of shape `(N, num_params)`, where `N` is the number of agents and
            `num_params` is the number of parameters per agent.

        Raises
        ------
        RuntimeError
            If an invalid parameter mode is specified in the configuration.

        Notes
        -----
        - If using `"From_File"`, the CSV file should have `N` rows (agents) and
          `num_params` columns (parameters).
        - If using `"Random"`, parameters are sampled uniformly within the given limits.
        """

        # Retrieve parameter loading mode (default: "Random")
        param_load_type = self.config.get("params_mode", "Random")

        match param_load_type:
            case "From_File":
                # Load parameters from a CSV file
                params_path = self.config.get("params_path", "")  # File path from YAML config
                params = pd.read_csv(params_path)  # Read CSV into DataFrame

                # If the file has fewer agents than `N`, randomly repeat parameters
                if params.shape[0] < self.N:
                    rows_to_add = self.N - params.shape[0]
                    for i in range (0,rows_to_add):
                        params_to_add = params.sample(n=1)
                        params = pd.concat([params, params_to_add],ignore_index=True)                   # Add the new parameters
                if params.shape[0] > self.N:
                    rows_to_drop = params.shape[0] - self.N
                    params = params.drop(params.sample(n=rows_to_drop).index)

            case "Random":
                # Generate random parameters
                params_names = self.config.get("params_names", [])  # List of parameter names
                params_limits = self.config.get("params_limits", {})  # Parameter sampling limits

                params = pd.DataFrame()  # Initialize DataFrame

                for par_name in params_names:
                    # Determine dimensionality of the parameter
                    par_values = params_limits[par_name][0]

                    if isinstance(par_values, (float, int)):  # Scalar parameter
                        par_dim = 1
                    else:  # Multi-dimensional parameter
                        par_dim = len(par_values)

                    # Generate random parameter values
                    if par_dim > 1:
                        values = np.empty([self.N, par_dim])
                        for i in range(par_dim):
                            values[:, i] = np.random.uniform(
                                params_limits[par_name][i][0], params_limits[par_name][i][1], self.N
                            )
                        params[par_name] = list(values)  # Store as a list of arrays
                    else:
                        params[par_name] = np.random.uniform(
                            params_limits[par_name][0], params_limits[par_name][1], self.N
                        )

            case _:
                raise RuntimeError("Invalid parameter mode. Check the YAML config file.")

        return params
