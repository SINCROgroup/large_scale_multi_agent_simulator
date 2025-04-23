from abc import ABC, abstractmethod
from swarmsim.Utils import load_config

from swarmsim.Environments import Environment
from swarmsim.Populations import Population


class Controller(ABC):
    """
    An interface that defines all the methods that a controller should implement
    
    
    Arguments
    ---------
    population : Population
        The population where the control is exerted
    environment : Environment
        The environment where the agents live
    other_populations : list[Population]
        A list of other populations that can influence the control action

    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    dt: float
        Sampling time of the controller (time interval between two consecutive control actions)

    """

    def __init__(self, population: Population, environment: Environment =None, config_path: str =None, other_populations = None) -> None:
        
        super().__init__()
        self.population: Population = population
        self.environment: Environment = environment
        self.other_populations = other_populations

        config: dict = load_config(config_path)

        # Get configuration for the specific population class
        class_name = type(self).__name__
        self.config = config.get(class_name, {})
        
        self.dt: float = self.config.get('dt')


    # This method
    @abstractmethod
    def get_action(self):
        """ 
            The get_action method uses the information given to the Controller class to compute the control acting on "Population".
            NOTE: This is the method you need to OVERRIDE to implement your controller
        """
        pass


    def get_action_in_space(self,positions):
        """

            For spatially non uniform control actions, this method gives back the control action at the positions specified in the positions vector

            Arguments
            ---------
            positions : numpy.Array(num_positions, num_dimensions)
                The positions where you want to retrieve the value of the control action
        
        """
        pass
