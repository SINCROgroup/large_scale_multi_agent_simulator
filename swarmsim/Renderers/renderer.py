from abc import ABC, abstractmethod
import yaml


class Renderer(ABC):
    """
    Abstract base class that defines the interface for rendering environments and agents.
    """

    @abstractmethod
    def __init__(self, populations, environment, config_path):
        """
        Initializes the renderer with the configuration parameters.
        Args:
            populations: A list of the populations to render
            environment: An instance of the environment class
            config_path (str): The path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.config = config.get('renderer', {})
        self.populations = populations
        self.environment = environment

    @abstractmethod
    def render(self):
        """
        Renders the environment and the agents in it.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the rendering interface, freeing up any used resources.
        """
        pass
