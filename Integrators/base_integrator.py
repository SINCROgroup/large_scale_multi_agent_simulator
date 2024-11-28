import yaml
from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Abstract base class for an Integrator, which defines the basic structure for numerical integration.
    This class reads configuration parameters from a YAML file and requires derived classes to implement the `step` method.

    Attributes:
        dt (float): The timestep value for integration, loaded from the configuration file.
    """

    def __init__(self, config_path):
        """
        Initializes the Integrator with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.dt = config.get('integrator', {}).get('dt', 0.01)  # Default timestep value if not specified  # Default to 100x100 if not specified

    @abstractmethod
    def step(self, populations):
        """
        Abstract method to perform a single integration step.

        Args:
            populations: A list of the populations for which the integration step is being performed

        This method must be implemented by any subclass of Integrator.
        """
        pass
