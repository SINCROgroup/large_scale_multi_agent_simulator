from abc import ABC, abstractmethod
import yaml


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

    def __init__(self, population, environment=None, config_path=None) -> None:
        """ 
            Initializes a controller, providing the controlled population. 
            Optionally you can provide the controller with the simulated environment and a configuration file (YAML file)
        """
        super().__init__()
        self.population = population
        self.environment = environment

        with open(config_path, "r") as file:
            self.params = yaml.safe_load(file)
        
        self.dt = self.params.get('dt',0.1)

    # This method
    @abstractmethod
    def get_action(self):
        """ 
            The get_action method uses the information given to the Controller class to compute the control acting on "Population".
            This is the method you need to override to implement your control action
        """
        pass
