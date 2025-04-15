import numpy as np
from typing import Optional
from swarmsim.Populations import Populations
from swarmsim.Utils import set_parameter


class BrownianMotion(Populations):
    """
    Implements a biased Brownian motion model with average velocity `mu` and diffusion coefficient `D`.

    This model simulates agent movement with a constant drift (`mu`) and stochastic diffusion (`D`),
    influenced by external forces (`f`) and control inputs (`u`).

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing population parameters.

    Attributes
    ----------
    x : np.ndarray, shape (N, state_dim)
        Current state of the agents, where `N` is the number of agents and `state_dim` is
        the dimensionality of the state space.
    mu : np.ndarray, shape (N, state_dim)
        Average velocity of each agent along the axes.
    D : np.ndarray, shape (N, state_dim)
        Diffusion coefficient determining the magnitude of stochastic motion.
    N : int
        Number of agents in the population.
    f : np.ndarray, shape (N, state_dim)
        External forces affecting each agent, such as environmental influences.
    u : np.ndarray, shape (N, state_dim)
        Control input applied to the agents.
    id : str
        Identifier for the population.
    config : dict
        Dictionary containing the parsed configuration parameters.

    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    x0_mode : str
        Mode for generating initial conditions. Options:
        - ``"From_File"`` : Load from a CSV file.
        - ``"Random"`` : Generate randomly.

    If ``x0_mode="From_File"``:
        x0_file_path : str
            Path to the CSV file containing initial conditions.

    If ``x0_mode="Random"``:
        N : int
            Number of agents in the population.
        state_dim : int
            Dimensions of the state space.

    D : list of list or callable
        Command to generate the diffusion coefficient, either as a list of predefined values or
        as a function that initializes `D`.

    mu : list of list or callable
        Command to generate the average velocity (`mu`), either as a list of values or
        as a function that initializes `mu`.

    id : str
        Identifier for the population.

    Notes
    -----
    - For more details on how initial conditions are generated, see the `get_initial_conditions`
      method in the `Populations` class.
    - The drift component (`mu`) and diffusion component (`D`) can be either predefined lists
      or dynamically generated using callable expressions from the YAML configuration.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        BrownianMotion:
            x0_mode: "Random"
            N: 100
            state_dim: 2
            mu: [[0.5, 0.0], [0.3, 0.2], [0.1, 0.5]]
            D: [[0.01, 0.01], [0.02, 0.02], [0.01, 0.03]]
            id: "BM_population"

    This defines a `BrownianMotion` population with 100 agents in a 2D space,
    assigned predefined drift velocities (`mu`) and diffusion coefficients (`D`).
    """

    def __init__(self, config_path: str, name: str = None) -> None:
        """
        Initializes the `BrownianMotion` population by loading parameters from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing initialization parameters.
        """

        self.mu: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        super().__init__(config_path, name)


    def get_drift(self) -> np.ndarray:
        """
        Computes the deterministic drift component of agent motion.

        The drift is given by:

            drift = mu + f + u

        where:
        - `mu` is the average velocity,
        - `f` represents external forces (e.g., interactions, environment),
        - `u` is the control input applied to the agents.

        Returns
        -------
        np.ndarray
            Array of shape `(N, state_dim)` representing the drift velocity of each agent.
        """
        return self.mu + self.f + self.u

    def get_diffusion(self) -> np.ndarray:
        """
        Computes the stochastic diffusion component of agent motion.

        The diffusion follows a standard Wiener process with coefficient `D`.

        Returns
        -------
        np.ndarray
            Array of shape `(N, state_dim, state_dim)` representing the diffusion coefficients for each agent.
        """
        return self.D

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.mu = set_parameter(self.params['mu'], (self.state_dim,))
        self.D = set_parameter(self.params['D'], (self.state_dim, self.state_dim))