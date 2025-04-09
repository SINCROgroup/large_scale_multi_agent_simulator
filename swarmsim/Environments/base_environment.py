import yaml
from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract base class for an environment in which agents operate.

    This class reads configuration parameters from a YAML file and requires derived classes
    to implement the `get_forces`, `get_info`, and `update` methods.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing environment parameters.

    Attributes
    ----------
    dimensions : tuple of (int, int)
        The dimensions of the environment in 2D (width, height), loaded from the configuration file.

    Config requirements
    -------------------
    dimensions : tuple of (int, int), optional
        The dimensions of the environment in 2D. Default is ``(100, 100)``.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required environment parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        environment:
            dimensions: [200, 150]

    This will set the environment dimensions to ``(200, 150)``.
    """

    def __init__(self, config_path):
        """
        Initializes the Environment with configuration parameters from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.params = config['environment']

        # Set dimensions (default: 100x100)
        self.dimensions = self.params.get('dimensions', (100, 100))

    @abstractmethod
    def get_forces(self, agents):
        """
        Computes the forces exerted by the environment on the agents.

        This method must be implemented by subclasses to define the environmental forces
        acting on agents in the simulation.

        Parameters
        ----------
        agents : list
            A list of agent objects for which the environmental forces are being computed.

        Returns
        -------
        np.ndarray
            An array representing the forces exerted on each agent.
        """
        pass

    @abstractmethod
    def get_info(self):
        """
        Retrieves environment-specific information for logging.

        This method must be implemented by subclasses to return relevant environment data.

        Returns
        -------
        dict
            A dictionary containing information about the environment.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Updates the environment state.

        This method must be implemented by subclasses to define how the environment evolves over time.
        """
        pass
