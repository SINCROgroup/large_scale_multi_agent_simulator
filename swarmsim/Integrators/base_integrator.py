from swarmsim.Utils import load_config
from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Abstract base class for numerical integration schemes in stochastic multi-agent systems.

    This class provides the interface for implementing various numerical integration methods
    for stochastic differential equations (SDEs) that govern agent dynamics. It handles the
    temporal evolution of agent states based on drift and diffusion components.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing integration parameters.

    Attributes
    ----------
    dt : float
        The timestep value for numerical integration.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the integrator section:

    dt : float, optional
        The timestep value for numerical integration. Default is ``0.01``.

    Notes
    -----
    - Subclasses must implement the abstract method `step()` to define the integration scheme.
    - Common integration schemes include Euler-Maruyama for SDEs.
    - The timestep should be chosen carefully to ensure numerical stability and accuracy.

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

    This sets the integration timestep to 0.05 time units.

    For stochastic systems, smaller timesteps generally improve accuracy but increase
    computational cost. The choice depends on the specific dynamics and required precision.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Integrator with configuration parameters.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing integration parameters.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found.
        """

        config = load_config(config_path)

        # Extract timestep value (default: 0.01 if not specified)
        self.dt = config.get('integrator', {}).get('dt', 0.01)

    @abstractmethod
    def step(self, populations):
        """
        Perform a single numerical integration step.

        This abstract method must be implemented by subclasses to define the specific
        numerical integration scheme. It advances the state of all agent populations
        by one timestep according to their dynamics.

        Parameters
        ----------
        populations : list of Population
            List of Population objects whose states will be updated by the integration step.
            Each population provides drift and diffusion terms through their respective methods.

        Notes
        -----
        The implementation should:
        
        - Call `get_drift()` and `get_diffusion()` methods for each population
        - Apply the numerical integration scheme (e.g., Euler-Maruyama, Runge-Kutta)
        - Update the state `x` attribute of each population
        - Handle stochastic terms appropriately for SDE integration
        - Respect any state or input limits defined in the populations


        Raises
        ------
        NotImplementedError
            If called from the base class without being implemented in a subclass.
        """
        pass
