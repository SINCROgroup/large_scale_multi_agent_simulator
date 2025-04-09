import numpy as np
from swarmsim.Integrators import Integrator


class EulerMaruyamaIntegrator(Integrator):
    """
    Euler-Maruyama Integrator for stochastic differential equations (SDEs).

    This class extends the `Integrator` base class and implements the `step` method
    to perform numerical integration using the Euler-Maruyama method.

    The method is used to integrate stochastic processes where the evolution of
    populations follows the general form:

        dx = drift * dt + diffusion * sqrt(dt) * dW

    where:
        - `drift` represents the deterministic part of the system.
        - `diffusion` represents the stochastic term.
        - `dW` is a Wiener process (Gaussian noise).

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing integration parameters.

    Attributes
    ----------
    dt : float
        The timestep value for integration, inherited from the `Integrator` base class.

    Config requirements
    -------------------
    dt : float, optional
        The timestep value for integration. Default is ``0.01``.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required integration parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        integrator:
            dt: 0.05

    This will set ``dt = 0.05`` as the timestep value for numerical integration.
    """

    def __init__(self, config_path):
        """
        Initializes the Euler-Maruyama Integrator with configuration parameters from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        """
        # Initialize the parent `Integrator` class, which loads configuration parameters
        super().__init__(config_path)

    # def step(self, populations):
    #     """
    #     Performs a single integration step using the Euler-Maruyama method.
    #
    #     The method updates the state of each population using:
    #
    #         x_new = x_old + drift * dt + diffusion * sqrt(dt) * noise
    #
    #     where `noise` is a random variable sampled from a standard normal distribution.
    #
    #     Parameters
    #     ----------
    #     populations : list
    #         List of population objects for which the integration step is performed.
    #         Each population object must have:
    #         - `x` (np.ndarray): The current state of the population.
    #         - `get_drift()` method returning the drift term.
    #         - `get_diffusion()` method returning the diffusion term.
    #         - `lim` (np.ndarray): limit of the state for saturation [Optional, Default inf]
    #
    #     Raises
    #     ------
    #     AttributeError
    #         If any population object does not have the required methods (`get_drift`, `get_diffusion`) or attributes (`x`).
    #     """
    #     for population in populations:
    #         # Compute the drift and diffusion terms
    #         drift = population.get_drift()
    #         diffusion = population.get_diffusion()
    #
    #         # Generate random noise (standard normal distribution)
    #         noise = np.random.normal(0, 1, size=population.x.shape)
    #
    #         # Update the state using the Euler-Maruyama method
    #         population.x = population.x + drift * self.dt + diffusion * np.sqrt(self.dt) * noise
    #         population.x = np.clip(population.x, -population.lim, population.lim)

    def step(self, populations):
        """
        Performs a single integration step using the Euler-Maruyama method.

        Parameters
        ----------
        populations : list
            List of population objects for which the integration step is performed.
        """
        for population in populations:
            # Compute the drift and diffusion terms
            drift = population.get_drift()
            diffusion = population.get_diffusion()

            # Call the Numba function for efficient computation
            step_numba(population.x, drift, diffusion, self.dt, population.lim)


import numpy as np
from numba import njit


@njit(fastmath=True)
def step_numba(x, drift, diffusion, dt, lim):
    """
    Numba-optimized function using explicit loops for the Euler-Maruyama update step.
    Handles scalars, 1D, and 2D arrays for drift, diffusion, and lim.

    Parameters
    ----------
    x : np.ndarray (2D)
        The state array of the population.
    drift : np.ndarray, scalar, or 1D array
        The drift term (can be scalar, 1D, or 2D).
    diffusion : np.ndarray, scalar, or 1D array
        The diffusion term (can be scalar, 1D, or 2D).
    dt : float
        Time step size.
    lim : np.ndarray, scalar, or 1D array
        The limit for state saturation (can be scalar, 1D, or 2D).

    Returns
    -------
    None (updates `x` in-place).
    """
    num_agents, num_dims = x.shape

    for i in range(num_agents):  # Loop over agents
        for j in range(num_dims):  # Loop over dimensions
            # Generate noise
            noise = np.random.normal(0, 1)

            # Compute Euler-Maruyama step
            x[i, j] += drift[i, j] * dt + diffusion[i, j] * np.sqrt(dt) * noise

            # Apply state limits (clipping)
            x[i, j] = max(-lim[j], min(x[i, j], lim[j]))
