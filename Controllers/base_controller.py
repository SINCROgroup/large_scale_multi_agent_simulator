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
        super().__init__()
        self.population = population
        self.environment = environment

        with open(config_path, "r") as file:
            self.params = yaml.safe_load(file)

    # This method
    @abstractmethod
    def get_action(self):
        pass
