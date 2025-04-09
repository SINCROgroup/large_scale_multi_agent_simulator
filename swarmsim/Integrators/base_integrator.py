import yaml
from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Abstract base class for an Integrator, defining the structure for numerical integration.

    This class reads configuration parameters from a YAML file and requires derived classes
    to implement the `step` method for numerical integration.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing integration parameters.

    Attributes
    ----------
    dt : float
        The timestep value for integration, loaded from the configuration file.

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

    def __init__(self, config_path: str):
        # Load configuration parameters from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Extract timestep value (default: 0.01 if not specified)
        self.dt = config.get('integrator', {}).get('dt', 0.01)

    @abstractmethod
    def step(self, populations):
        """
        Abstract method to perform a single integration step.

        This method must be implemented by any subclass to define how numerical
        integration is performed for a given set of populations.

        Parameters
        ----------
        populations : list or np.ndarray
            A list or array representing the populations for which the integration step is being performed.

        Raises
        ------
        NotImplementedError
            If called from the base class without being implemented in a subclass.
        """
        pass
