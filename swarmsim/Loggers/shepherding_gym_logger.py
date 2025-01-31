from datetime import datetime
from swarmsim.Loggers import ShepherdingLogger
from swarmsim.Utils import add_entry, get_done_shepherding, xi_shepherding
import yaml
import time
import os
import csv
import numpy as np


class ShepherdingGymLogger(ShepherdingLogger):
    """
        A class that implements a logger.

        Parameters
        -------
            populations: list of instances of clas populations.
            environment: Instance of class environment.
            config_path: str (Path of the configuration file).

        Attributes
        -------
            - config: dict                                  Dictionary of parameters
            - activate: bool                                Flag to activate and save logger
            - date: date object                             Current date to log and name files
            - log_freq: int                                 Print information frequency
            - save_freq: int                                Save information frequency
            - log_path: str                                 Path where logger is saved
            - comment_enable: bool                          Flag to enable adding comment at the beginning and end of an experiment
            - populations: list of population objects       List of populations in the experiment
            - environment: environment object               Environment of the experiment
            - name: str                                     Name of the logger, appended to the date to name output files
            - log_name_csv: str                             Name of the .csv machine-readable file
            - log_name_txt: str                             Name of the .txt human-readable file
            - log_name_npz: str                             Name of the .npz file to store tensors
            - start: time object                            Starting time of the simulation
            - end: time object                              Final time of the simulation
            - step_count: int                               Step counter to track time
            - done: bool                                    Flag to truncate experiment early
            - current_info: dict                            Information to log in a specific time step as a dict of {name_variable: value}

        Configuration file requirements
            - activate: bool            True to have logger, False otherwise
            - log_freq: int             Print every log_freq steps information (0: never print)
            - save_freq: int            Save every save_freq steps information (0: never save)
            - comment_enable: bool      If true, add initial and final comments to the logger about the experiment
            - log_path: str             Path where logger output should be saved
            - log_name: str             String appended to date in the name of the file

        Notes
        -------
            If active, outputs two files: one .csv computer-readable and one .txt human-readable named DATEname.csv and DATEname.txt, respectively.
            Moreover, save_data stores a .npz of the data given in input.

        """
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)
        self.log_name_npz = self.log_path + '/' + self.name + '.npz'  # Name

    def log(self, data: dict = None):
        """
        A function that defines the information to log.

        Parameters
        ----------
            data: dict  composed of {name_variabile: value} to log

        Returns
        -------
            done: bool flag to truncate a simulation early. Default value=False.

        Notes
        -------
            In the configuration file (a yaml file) there should be a namespace with the name of the log you are creating.
            By default, it does not truncate episode early.
            See add_data from Utils/logger_utils.py to quickly add variables to log.
        """

        # Get log info
        self.current_info = {}
        self.done = self.get_event()  # Verify if episode is done

        if self.activate:
            # Get metrics
            xi = self.get_xi()

            # Include desired information
            add_entry(self.current_info, step=self.step_count)  # Get timestamp
            add_entry(self.current_info, xi=xi)
            add_entry(self.current_info, done=self.done)
            if data is not None:
                for key, value in data.items():
                    add_entry(self.current_info, **{key: value})

            # Print line if wanted
            if self.log_freq > 0:
                if self.step_count % self.log_freq == 0:
                    self.print_log()

            # Save line if wanted
            if self.save_freq > 0:
                if self.step_count % self.save_freq == 0:
                    self.save()

            self.step_count += 1  # Update step counter

        return self.done
