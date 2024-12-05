import yaml
from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract base class for an environment in which agents operate.
    This class reads configuration parameters from a YAML file and requires derived classes to implement the `get_forces` method.

    Attributes:
        dimensions (tuple): The dimensions of the environment in 2D (width, height).
    """

    def __init__(self, config_path):
        """
        Initializes the Environment with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.params = config['environment']

        self.dimensions = self.params.get('dimensions', (100, 100))  # Default to 100x100 if not specified

    @abstractmethod
    def get_forces(self, agents):
        """
        Abstract method to compute the forces exerted by the environment on the agents.
        You need to override this model to implement environmental forces

        Args:
            agents: A list of agents for which the environmental forces are being computed.

        Returns:
            np.ndarray: An array representing the forces exerted on each agent.
        """
        pass

    @abstractmethod
    def get_info(self):
        """
        Returns information to log
        """
        return {''}
