from abc import ABC, abstractmethod
import yaml

from swarmsim.Environments import Environment


class Renderer(ABC):
    """
    Abstract base class that defines the interface for rendering environments and agents.

    This class provides an interface for visualizing multi-agent environments, requiring
    subclasses to implement `render` and `close` methods.

    Parameters
    ----------
    populations : list
        A list of population objects to render.
    environment : object
        The environment instance in which the populations exist.
    config_path : str
        Path to the YAML configuration file containing rendering settings.

    Attributes
    ----------
    populations : list
        A list of population objects to be rendered.
    environment : object
        The environment in which the agents operate.
    config : dict
        Dictionary containing rendering configuration parameters.

    Config requirements
    -------------------
    renderer:
        The YAML configuration file must contain a `renderer` section specifying visualization settings.

    Notes
    -----
    - This is an abstract base class (ABC) and must be subclassed.
    - The `render` method must be implemented to display the environment and agent states.
    - The `close` method should release any resources allocated for rendering.
    """

    @abstractmethod
    def __init__(self, populations: list, environment: Environment, config_path: str):
        """
        Initializes the renderer with the configuration parameters.

        Parameters
        ----------
        populations : list
            A list of population objects to render.
        environment : object
            An instance of the environment class.
        config_path : str
            Path to the YAML configuration file.
        """
        # Load config params from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.config: dict = config.get('renderer', {})
        self.populations: list = populations
        self.environment: Environment = environment

    @abstractmethod
    def render(self):
        """
        Renders the environment and the agents in it.

        This method should be implemented by subclasses to define how the environment
        and agents are visually represented.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the rendering interface and releases any allocated resources.

        This method should be implemented in subclasses to properly clean up
        rendering-related resources such as open windows, memory buffers, or GPU contexts.
        """
        pass
