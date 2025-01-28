from datetime import datetime
from swarmsim.Loggers import Logger
from swarmsim.Utils import add_entry, append_csv, append_txt, get_done_shepherding, xi_shepherding
import yaml
import time
import os
import csv
import numpy as np


# Outputs two files: one csv computer-readable and one txt human-readable


class ShepherdingGymLogger(Logger):
    def __init__(self, populations, environment, config_path):
        super().__init__()
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  # Get current date to init logger
        logger_config = config.get('logger', {})
        self.activate = logger_config.get('activate', True)
        self.log_freq = logger_config.get('log_freq', 1)  # Print frequency
        self.save_freq = logger_config.get('save_freq', 1)  # Save frequency
        self.log_path = os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.comment_enable = logger_config.get('comment_enable', False)
        self.populations = populations
        self.environment = environment

        # If the path does not exist, create it
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Outputs DATEname.csv and DATEname.txt
        self.name = datetime.today().strftime('%Y%m%d_%H%M%S') + logger_config.get('log_name', '')

        #  Init key variables
        self.log_name_csv = self.log_path + '/' + self.name + '.csv'  # Check concat string
        self.log_name_txt = self.log_path + '/' + self.name + '.txt'  # Check concat string
        self.log_name_npz = self.log_path + '/' + self.name + '.npz'  # Check concat string
        self.start = None  # Time start
        self.end = None  # Time end
        self.step_count = None  # Count steps for frequency check and logging
        self.done = None  # Episode truncation
        self.config = config  # Get config to track experiments
        self.current_info = None

        if self.activate:
            # If there are any comments to describe the experiment
            if self.comment_enable:
                comment = input('Comment: ')
            else:
                comment = ''

            # Create file with current date, setting, and comment
            with open(self.log_name_txt, 'w') as file:
                file.write('Date:' + self.date)
                file.write('\nConfiguration settings: \n')
                for key, value in self.config.items():
                    file.write(str(key) + ': ' + str(value) + '\n')
                file.write('\nInitial comment: ' + comment)

    def reset(self):
        self.done = False
        if self.activate:
            # Initialize logger: create file with date, current config settings, and add eventual comments
            self.start = time.time()  # Start counter for elapsed time
            self.step_count = 0  # Keep track of time
        return self.activate

    def log(self, data=None):
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

    def print_log(self):
        for key, value in self.current_info.items():
            print(f"{key}: {value};", end=" ")
        print('\n')

    def save(self):
        if self.activate:
            """Save the current entry to both CSV and TXT files."""
            # Save to CSV
            append_csv(self.log_name_csv, self.current_info)

            # Save to TXT
            append_txt(self.log_name_txt, self.current_info)

        return self.activate

    def close(self):
        # Log final step before closing

        if self.activate:
            self.done = self.log()  # Log last time step
            self.end = time.time()  # Get end time for elapsed time

            # Eventually get final comments on the simulation
            if self.comment_enable:
                comment = input('\nComment: ')
            else:
                comment = ''

            #  Save final row with 'Done', elapsed time, and eventual comment.
            with open(self.log_name_txt, 'a') as file:
                file.write('\nDone: ' + str(self.done) +
                           '\nSettling time [steps]:' + str(self.step_count) +
                           '\nElapsed time [s]:' + str(self.end - self.start) +
                           '\nComments: ' + comment + '\n')
        return self.activate

    def get_xi(self):
        return xi_shepherding(self.populations[0], self.environment)

    def get_event(self):
        return get_done_shepherding(self.populations[0], self.environment)

    def save_data(self, data):
        for key, value in data.items():
            np.savez(self.log_name_npz, **{key: value})
