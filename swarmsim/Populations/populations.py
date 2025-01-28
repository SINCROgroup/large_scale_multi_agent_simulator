from abc import ABC, abstractmethod
import numpy as np
import yaml
import pandas as pd
from pathlib import Path


class Populations(ABC):
    """
    An interface that defines all the methods that an agent should implement
    Arguments
    -------
    N : Number of agents in the population
    f (NxD double matrix) : External forces (interactions and environment)
    u (NxD double matrix) : Control input
    id (string) : defines the population name

    Methods
    -------
    get_drift(self,u):
        Returns the drift (deterministic part of the dynamics) of the agent.
    get_diffusion(self,u):
        Returns the diffusion (stochastic part of the dynamics) of the agent.
    reset_state(self):
        Resets the state of the agent.
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

    def get_initial_conditions(self,config_path):
        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        Class_name = type(self).__name__                                # Getting the name of the class
        self.params = config.get(Class_name, {})                        # Loading the parameters of the population
        
        x0_load_type = self.params.get('x0_mode',"From_File")
        match x0_load_type:
            case "From_File":
                root_folder = Path(__file__).resolve().parent.parent    # Getting the root folder
                x0_file_name = self.params.get("x0_file_name","")       # Retrieving the file name
                x0_path = str(root_folder / "Data" / x0_file_name)      # Constructing the path of the file
                self.x0 = pd.read_csv(x0_path,header=None).values       # Reading the initial conditions 
                self.N = self.x0.shape[0]                               # Getting the Number of Agents
                self.state_dim = self.x0.shape[1]                       # Getting the state dimension of the agent
            case "Random":
                env = config.get("environment",{})                                                      # Get the environment parameters
                env_dimension = env.get("dimensions")                                                   # Get the environment Dimensions
                self.N = self.params.get("N")                                                           # Get the number of Agents
                self.state_dim = self.params.get("state_dim")                                           # Get the state dimension
                self.x0 = np.zeros([self.N,self.state_dim])                                             # Initialize the vector of the initial conditions
                for i in range(0,len(env_dimension)):                                                   # For every dimension
                    self.x0[:,i] = np.random.uniform(-env_dimension[i]/2,env_dimension[i]/2,self.N)     # Generate uniformly distributed points in the domain
            case _ :
                raise RuntimeError("Invalid Initialization type, please check the YAML config file")
        return self.x0