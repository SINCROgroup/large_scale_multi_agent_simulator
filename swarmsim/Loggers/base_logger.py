import pathlib
from datetime import datetime
from swarmsim.Loggers import Logger
from swarmsim.Utils import add_entry, append_csv, append_txt, print_log, save_npz, save_mat
import yaml
import time
import os
import numpy as np
from swarmsim.Utils.sim_utils import load_config


class BaseLogger(Logger):
    """
    A class that implements the simplest version of a logger.

    Parameters
    ----------
        populations: list of instances of class populations.
        environment: Instance of class environment.
        config_path: str (Path of the configuration file).

    Attributes
    ----------
        - config: dict                                  
            Dictionary of parameters
        - activate: bool                                
            Flag to activate and save logger
        - date: date object                             
            Current date to log and name files
        - log_freq: int                                 
            Print information frequency
        - save_freq: int                                
            Save information frequency
        - log_path: str                                 
            Path where logger is saved
        - comment_enable: bool                          
            Flag to enable adding comment at the beginning and end of an experiment
        - populations: list of population objects       
            List of populations in the experiment
        - environment: environment object               
            Environment of the experiment
        - name: str                                     
            Name of the logger, appended to the date to name output files
        - log_name_csv: str                             
            Name of the .csv machine-readable file
        - log_name_txt: str                             
            Name of the .txt human-readable file
        - log_name_npz: str                             
            Name of the .npz file to store tensors
        - start: time object                             
            Starting time of the simulation
        - end: time object                               
            Final time of the simulation
        - step_count: int                                 
            Step counter to track time
        - done: bool                                    
            Flag to truncate experiment early
        - current_info: dict                            
            Information to log in a specific time step as a dict of {name_variable: value}

    Config requirements
    -------------------

        - activate: bool            
            True to have logger, False otherwise
        - log_freq: int             
            Print every log_freq steps information (0: never print)
        - save_freq: int            
            Save every save_freq steps information (0: never save)
        - save_data_freq: int
            TBD
        - save_global_data_freq: int
            TBD
        - comment_enable: bool      
            If true, add initial and final comments to the logger about the experiment
        - log_path: str             
            Path where logger output should be saved
        - log_name: str             
            String appended to date in the name of the file
        
        

    Notes
    -----

        If active, outputs two files: one .csv computer-readable and one .txt human-readable named DATEname.csv and DATEname.txt, respectively.
        Moreover, save_data stores a .npz of the data given in input.

    """

    
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__()

        # Load configuration file
        config:dict = load_config(config_path)
        class_name = type(self).__name__
        logger_config = config.get(class_name, {})
        self.config = config  
        self.logger_config = logger_config
         

        
        # Initialize parameters
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  # Get current date to init logger
        self.name = datetime.today().strftime('%Y%m%d_%H%M%S') + logger_config.get('log_name', '')
        self.activate = logger_config.get('activate', True)  # Activate
        self.log_freq = logger_config.get('log_freq', 0)  # Print frequency
        self.save_freq = logger_config.get('save_freq', 0)  # Save frequency
        self.save_data_freq = logger_config.get('save_data_freq', 0)
        self.save_global_data_freq = logger_config.get('save_global_data_freq', 0)
        self.log_path = logger_config.get('log_path', './logs')
        self.comment_enable = logger_config.get('comment_enable', False)
        self.populations = populations
        self.environment = environment

        log_folder = self.log_path + '/' + self.name
        # If the path does not exist, create it
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)



        #  Generate log file names
        self.log_name_csv = log_folder + '/' + self.name + '.csv'
        self.log_name_txt = log_folder + '/' + self.name + '.txt'
        self.log_name_npz = log_folder + '/' + self.name + '.npz'
        self.log_name_mat = log_folder + '/' + self.name + '.mat'



        # Initialize auxiliary variables
        self.start = None  # Time start
        self.end = None  # Time end
        self.step_count = None  # Count steps for frequency check and logging
        self.experiment_count = None
        self.done = None  # Episode truncation
        self.current_info = None
        self.global_info = None

        
        if self.activate:
            # If there are any comments to describe the experiment add them, otherwise empty
            if self.comment_enable:
                comment = input('Comment: ')
            else:
                comment = ''
            

            # Create file with current date, setting, and comment (if active)
            with open(self.log_name_txt, 'w') as file:
                file.write('Date:' + self.date)
                file.write('\nConfiguration settings: \n')
                for key, value in self.config.items():
                    file.write(str(key) + ': ' + str(value) + '\n')
                file.write('\nInitial comment: ' + comment)

    def reset(self) -> bool:
        """
        Reset logger at the beginning of the simulation. Verifies if active, reset step counter and start time counter

        Returns
        -------
            activate: bool  flag to check whether the logger is active
        """

        self.done = False
        if self.activate:
            # Initialize logger: create file with date, current config settings, and add eventual comments
            self.start = time.time()  # Start counter for elapsed time
            self.step_count = 0  # Keeps track of time
            self.experiment_count = 0
            self.global_info = {}
        return self.activate

    def log(self, data: dict | None = None):
        """
        A function that defines the information to log.

        Parameters
        ----------
            data: dict  composed of {name_variabile: value} to log

        Returns
        -------
            done: bool flag to truncate a simulation early. Default value=False.

        Notes
        -----
            In the configuration file (a yaml file) there should be a namespace with the name of the log you are creating.
            By default, it does not truncate episode early.
            See add_data from Utils/logger_utils.py to quickly add variables to log.

        """

        # Get log info
        self.current_info = {}
        self.done = False

        if self.activate:
            self.log_internal_data()
            self.log_external_data(data)
            self.output_data()

            # Update step counter
            self.step_count += 1

        return self.done

    def close(self, data: dict = None) -> bool:
        """
        Function to store final step information, end-of-the-experiment information and close logger

        Parameters
        ----------
            data: dict  composed of {name_variabile: value} to log

        Returns
        -------
            activate: bool flag to check whether the logger is active
        """

        # Log final step before closing
        if self.activate:
            self.experiment_count += 1
            self.done = self.log(data)  # Log last time step before closing
            self.end = time.time()  # Get end time for elapsed time

            # (Optional) get final comments on the simulation
            if self.comment_enable:
                comment = input('\nComment: ')
            else:
                comment = ''

            # Save final row with 'Done', elapsed time, and (optional) comment.
            with open(self.log_name_txt, 'a') as file:
                file.write('\nDone: ' + str(self.done) +
                           '\nSettling time [steps]:' + str(self.step_count) +
                           '\nElapsed time [s]:' + str(self.end - self.start) +
                           '\nComments: ' + comment + '\n')

        return self.activate

    def log_external_data(self, data, save_mode=['npz', 'mat']):
        if data is not None:
            for key, value in data.items():
                add_entry(self.current_info, save_mode, **{key: value})

    def log_internal_data(self, save_mode=['txt', 'print']):
        add_entry(self.current_info, save_mode, step=self.step_count)  # Get timestamp
        add_entry(self.current_info, save_mode, done=self.done)  # Add done flag

    def output_data(self):
        # Print line if wanted
        if self.log_freq > 0:
            if self.step_count % self.log_freq == 0:
                print_log(self.current_info)

        # Save line if wanted
        if self.save_freq > 0:
            if self.step_count % self.save_freq == 0:
                # Save to CSV
                append_csv(self.log_name_csv, self.current_info)

                # Save to TXT
                append_txt(self.log_name_txt, self.current_info)

        # Save to npz and mat if wanted
        if self.save_data_freq > 0:
            if self.step_count % self.save_data_freq == 0:
                save_npz(self.log_name_npz, self.current_info)
                save_mat(self.log_name_mat, self.current_info)

        if self.save_global_data_freq and self.experiment_count > 0:
            if self.experiment_count % self.save_global_data_freq == 0:
                save_npz(self.log_name_npz, self.global_info)
                save_mat(self.log_name_mat, self.global_info)
