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
        # Load the YAML configuration file
        self.N = None                                                   # N and state_dim are ste in get_initial_conditions
        self.state_dim = None

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        Class_name = type(self).__name__                                # Getting the name of the class
        self.config = config.get(Class_name, {})                        # Loading the configuration of the population
        self.x = self.get_initial_conditions()                          # Loading Initial conditions
        self.params = self.get_parameters()                             # Loading parameters


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

    def get_initial_conditions(self) -> np.array: 
        '''
        A function that loads the initial conditions for the population. 

        Returns
        -------
            x0: numpy.array (num_agents x dim_state)
                Initial conditions of the population

        Configuration Requirements
        -------
        
            x0_mode: str
                The mode in which the Initial conditions are generated. Supported modes are "From File" (load the initial conditions from a .csv file, where rows are states of different agents and columns are different states of the same agent) and "Random" where the initial conditions are randomly selected.
            
            If the mode selected is "From File", in the yaml file it is required

            x0_file_path: str 
                Absolute path of the csv file
            
            Instead, if the choicie is "Random", the configuration file needs

            N : int
                Number of agents in the poulation
            state_dim : int 
                Dimensions of the state
            x0_limits : list (state_dim x 2)
                Limits for the uniform distribution from which each state is drawn. Note that if limits dimensions is lower than state_dim, the other states will be set to 0
        
        Notes
        -------
        
            In the configuration file (a yaml file) there should be a namespace with the name of the population you are creating.

        '''
        
        x0_load_type = self.config.get('x0_mode',"Random")
        match x0_load_type:
            case "From_File":
                x0_path = self.config.get("x0_path","")                                                 # Retrieving the file name
                self.x0 = pd.read_csv(x0_path,header=None).values                                       # Reading the initial conditions 
                self.N = self.x0.shape[0]                                                               # Getting the Number of Agents
                self.state_dim = self.x0.shape[1]                                                       # Getting the state dimension of the agent
            case "Random":
                limits = np.array(self.config.get("x0_limits",[]))                                                  # Get the environment parameters
                self.N = self.config.get("N")                                                           # Get the number of Agents
                self.state_dim = self.config.get("state_dim")                                           # Get the state dimension
                self.x0 = np.zeros([self.N,self.state_dim])                                             # Initialize the vector of the initial conditions
                for i in range(0,limits.shape[1]):                                                    # For every dimension
                    self.x0[:,i] = np.random.uniform(limits[i,0],limits[i,1],self.N)                   # Generate uniformly distributed points in the domain
            case _ :
                raise RuntimeError("Invalid Initialization type, please check the YAML config file")
        return self.x0
    
    def get_parameters(self) -> pd.DataFrame :
        '''
        A function that initializes the parameters of the population. 

        Returns
        -------
            params: pandas.DataFrame (num_agents x param_names)
                Parameters of the populations

        Configuration Requirements
        -------
        
            params_mode: str
                The mode in which the Initial conditions are generated. Supported modes are "From File" (load the initial conditions from a .csv file, where rows are states of different agents and columns are different parameters of the same agent) and "Random" where the initial conditions are randomly selected.
                Note that in From File if the number of agents does not meet N (the rows in the file < N) the remaining parameters are drawn randomly from the ones in the file
            
            If the mode selected is "From File", in the yaml file it is required

            params_file_path: str 
                Absolute path of the .csv file
            
            Instead, if the choicie is "Random", the configuration file needs

            params_names : list (params_number) 
                Dimensions of the state
            params_limits : list (params_number x 2)
                Limits for the uniform distribution from which each parameter is drawn. 
        
        Notes
        -------
            In the configuration file (a yaml file) there should be a namespace with the name of the population you are creating.

        '''

        param_load_type = self.config.get("params_mode","Random")
        match param_load_type:
            case "From_File":
                params_path = self.config.get("params_path","")                                         # Retrieving the file name
                params = pd.read_csv(params_path)                                                       # Reading the parameters
                if params.shape[0] < self.N:
                    rows_to_add = self.N - params.shape[0]
                    params_to_add = params.sample(n=rows_to_add)
                    params = pd.concat([params, params_to_add],ignore_index=True)                   # Add the new parameters
                if params.shape[0] > self.N:
                    rows_to_drop = params.shape[0] - self.N
                    params = params.drop(params.sample(n=rows_to_drop).index)

            case "Random":
                params_names = self.config.get("params_names",[])                                       # Get parameter names
                params_limits = self.config.get("params_limits",[])                                     # Get parameter limits
                params = pd.DataFrame()                                                                 # Init
                for par_name in params_names: 
                    #Understanding if the parameter is mono or multi dimensional
                    par_dim = params_limits[par_name][0]
                    if type(par_dim) == float or type(par_dim) == int:
                        par_dim=1
                    else:
                        par_dim = len(params_limits[par_name][0])
                    
                    #Setting random parameters
                    if par_dim>1:
                        par_values=np.empty([self.N,par_dim]) 
                        for i in range(0,par_dim):
                            par_values[:,i] =  np.random.uniform(params_limits[par_name][i][0],params_limits[par_name][i][1],self.N)                                                               # Set random parameters values for every parameter
                        params[par_name] = list(par_values)
                    else:
                        params[par_name] = np.random.uniform(params_limits[par_name][0],params_limits[par_name][1],self.N)
            case _ :
                raise RuntimeError("Invalid Initialization type, please check the YAML config file")

        return params