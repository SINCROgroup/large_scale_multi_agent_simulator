import numpy as np
from swarmsim.Integrators import Integrator


class EulerMaruyamaIntegrator(Integrator):
    """
    Euler-Maruyama numerical integrator for stochastic differential equations.

    This integrator implements the Euler-Maruyama scheme for solving stochastic differential
    equations (SDEs) governing multi-agent dynamics. It handles both scalar and matrix-valued
    diffusion terms, making it suitable for complex stochastic systems with correlated noise.

    The integration scheme follows:

    .. math::

        x_{n+1} = x_n + f(x_n, t_n) \\Delta t + g(x_n, t_n) \\sqrt{\\Delta t} \\, \\xi_n

    where:
    - :math:`x_n` is the state at time step n
    - :math:`f(x_n, t_n)` is the drift term (deterministic dynamics)
    - :math:`g(x_n, t_n)` is the diffusion term (noise amplitude)
    - :math:`\\xi_n` is standard Gaussian white noise
    - :math:`\\Delta t` is the integration timestep

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing integration parameters.

    Attributes
    ----------
    dt : float
        Integration timestep, inherited from the Integrator base class.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the integrator section:

    dt : float, optional
        Integration timestep for the Euler-Maruyama scheme. Default is ``0.01``.
        Smaller timesteps improve accuracy but increase computational cost.

    Notes
    -----
    **Diffusion Term Handling:**

    The integrator automatically detects the dimensionality of the diffusion term:

    - **Scalar/Vector Diffusion** (shape ``(N, d)``): Element-wise multiplication with noise
    - **Matrix Diffusion** (shape ``(N, d, d)``): Matrix-vector multiplication for correlated noise

    **State Constraints:**

    Agent states are automatically clipped to respect population limits after each integration step.

    **Numerical Stability:**

    For strong convergence, the timestep should satisfy stability conditions specific to
    the drift and diffusion terms. Generally, smaller timesteps are required for systems
    with large diffusion coefficients.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required integration parameters are missing in the configuration file.
    AttributeError
        If population objects lack required methods or attributes.

    Examples
    --------
    

    **Configuration Example:**

    .. code-block:: yaml

        integrator:
            dt: 0.01  # Small timestep for good accuracy


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

    def step(self, populations):
        """
        Perform a single Euler-Maruyama integration step for all populations.

        This method updates the state of each population by applying the Euler-Maruyama
        scheme with automatic handling of scalar and matrix diffusion terms. The state
        update follows the discrete SDE formula with proper noise scaling.

        Parameters
        ----------
        populations : list of Population
            List of population objects to integrate. Each population must provide:
            
            - ``x`` (np.ndarray): Current state array of shape ``(N, d)``
            - ``get_drift()`` method: Returns drift term of shape ``(N, d)``
            - ``get_diffusion()`` method: Returns diffusion term of shape ``(N, d)`` or ``(N, d, d)``
            - ``lim_i`` (np.ndarray): Lower state bounds for clipping
            - ``lim_s`` (np.ndarray): Upper state bounds for clipping

        Notes
        -----
        **Integration Algorithm:**

        For each population, the state update is:

        1. **Drift Computation**: Get deterministic dynamics :math:`f(x,t)`
        2. **Diffusion Computation**: Get noise amplitude :math:`g(x,t)`
        3. **Noise Generation**: Sample :math:`\\xi \\sim \\mathcal{N}(0,I)`
        4. **State Update**: Apply Euler-Maruyama formula
        5. **Constraint Enforcement**: Clip states to population limits

        **Diffusion Term Handling:**

        - **Vector Diffusion** (shape ``(N, d)``): ``noise_term = diffusion * noise``
        - **Matrix Diffusion** (shape ``(N, d, d)``): ``noise_term = diffusion @ noise``

        **State Constraints:**

        After integration, all agent states are clipped to respect the population's
        spatial or behavioral limits defined by ``lim_i`` and ``lim_s``.

        Raises
        ------
        AttributeError
            If any population object lacks required methods (``get_drift``, ``get_diffusion``) 
            or attributes (``x``, ``lim_i``, ``lim_s``).
        ValueError
            If diffusion term shape is incompatible with noise or state dimensions.

        
        """
        for population in populations:
            # Compute the drift and diffusion terms
            drift = population.get_drift()
            diffusion = population.get_diffusion()

            # Generate random noise (standard normal distribution)
            noise = np.random.normal(0, 1, size=population.x.shape)

            # Update the state using the Euler-Maruyama method
            if np.ndim(diffusion) == 3:  # diffusion is a matrix (e.g. shape (N, d, d))
                noise_term = np.matmul(diffusion, noise[..., np.newaxis]).squeeze(-1)
            else:
                noise_term = diffusion * noise
            population.x = population.x + drift * self.dt + noise_term * np.sqrt(self.dt)
            population.x = np.clip(population.x, population.lim_i, population.lim_s)
