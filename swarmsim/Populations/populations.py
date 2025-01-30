from abc import ABC, abstractmethod
import numpy as np
import yaml
import pandas as pd
from pathlib import Path


class Populations(ABC):
    """
    An interface that defines all the methods that an agent should implement.
    """

    def __init__(self,config_path) -> None:
        #Initialization steps from the Parent Class
        super().__init__()
        self.config_path = config_path
        self.x = self.get_initial_conditions(self.config_path)


    # This method
    @abstractmethod
    def get_drift(self):
        pass

    @abstractmethod
    def get_diffusion(self):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    def get_initial_conditions(self,config_path:str) -> None: 
        '''
        A function that loads the initial conditions for the population. 

        Parameters
        -------
        config_path: str (Absolute path of the configuration path)

        Returns
        -------
        x0: numpy array (num_agents x dim_state) Initial conditions of the population

        Notes
        -------
        In the configuration file (a yaml file) there should be a namespace with the name of the population you are creating.
        To load correctly the initial conditions the following parameter is required:
        - x0_mode: str (The mode in which the Initial conditions are generated).     
        Supported modes are "From File" (load the initial conditions from a .csv file, where rows are states of different agents and columns are different states of the same agent) and "Random" where the initial conditions are randomly selected.
        If the mode selected is "From File", in the yaml file it is required:
        - x0_file_path: str (Absolute path of the csv file)
        Instead, if the choicie is "Random", the configuration file needs:
        - N : int Number of agents in the poulation
        - state_dim : int Dimensions of the state
        Note that in the "random methods" only the first env_dim states are drawn at random, the others are set to 0. Env dim is the dimension of the environment specified in the environment namespace of the configuration file
        
        '''
        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        Class_name = type(self).__name__                                # Getting the name of the class
        self.params = config.get(Class_name, {})                        # Loading the parameters of the population
        
        x0_load_type = self.params.get('x0_mode',"From_File")
        match x0_load_type:
            case "From_File":
                x0_path = self.params.get("x0_path","")                 # Retrieving the file name
                self.x0 = pd.read_csv(x0_path,header=None).values       # Reading the initial conditions 
                self.N = self.x0.shape[0]                               # Getting the Number of Agents
                self.state_dim = self.x0.shape[1]                       # Getting the state dimension of the agent
            case "Random":
                env = config.get("environment",{})                                                      # Get the environment parameters
                env_dimension = env.get("dimensions",1)                                                   # Get the environment Dimensions
                self.N = self.params.get("N")                                                           # Get the number of Agents
                self.state_dim = self.params.get("state_dim")                                           # Get the state dimension
                self.x0 = np.zeros([self.N,self.state_dim])                                             # Initialize the vector of the initial conditions
                for i in range(0,len(env_dimension)):                                                   # For every dimension
                    self.x0[:,i] = np.random.uniform(-env_dimension[i]/2,env_dimension[i]/2,self.N)     # Generate uniformly distributed points in the domain
            case _ :
                raise RuntimeError("Invalid Initialization type, please check the YAML config file")
        return self.x0