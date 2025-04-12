from abc import ABC, abstractmethod
from swarmsim.Utils import load_config

from swarmsim.Environments import Environment
from swarmsim.Populations import Populations


class Controller(ABC):
    """
    An interface that defines all the methods that a controller should implement
    Arguments
    -------
    population : The controlled population
    environment : The environment
    other_populations : A list of other populations

    Methods
    -------
    get_action(self):
        Returns the control action for the agents

    """

    def __init__(self, population: Populations, environment: Environment =None, config_path: str =None) -> None:
        """ 
            Initializes a controller, providing the controlled population. 
            Optionally you can provide the controller with the simulated environment and a configuration file (YAML file)
        """
        super().__init__()
        self.population: Populations = population
        self.environment: Environment = environment

        config: dict = load_config(config_path)

        # Get configuration for the specific population class
        class_name = type(self).__name__
        self.config = config.get(class_name, {})
        
        self.dt: float = self.config.get('dt',0.1)

    # This method
    @abstractmethod
    def get_action(self):
        """ 
            The get_action method uses the information given to the Controller class to compute the control acting on "Population".
            This is the method you need to override to implement your control action
        """
        pass


    def get_action_in_space(self,positions):
        pass
