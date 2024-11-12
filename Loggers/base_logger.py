from datetime import datetime
import yaml
import time
import os
import csv
import numpy as np


# Outputs two files: one csv computer-readable and one txt human-readable


class Logger:
    def __init__(self, config_path):

        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  # Get current date to init logger
        logger_config = config.get('logger', {})
        self.log_freq = logger_config.get('log_freq', 1)  # Print frequency
        self.save_freq = logger_config.get('save_freq', 1)  # Save frequency
        self.log_path = os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.comment_enable = logger_config.get('comment_enable', False)

        # If the path does not exist, create it
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Outputs DATEname.csv and DATEname.txt
        self.name = datetime.today().strftime('%Y%m%d_%H%M%S') + logger_config.get('log_name', '')

        #  Init key variables
        self.log_name_csv = self.log_path + '/' + self.name + '.csv'  # Check concat string
        self.log_name_txt = self.log_path + '/' + self.name + '.txt'  # Check concat string
        self.start = None  # Time start
        self.end = None  # Time end
        self.step_count = None  # Count steps for frequency check and logging
        self.config = config  # Get config to track experiments

    def reset(self):
        # Initialize logger: create file with date, current config settings, and add eventual comments
        self.start = time.time()  # Start counter for elapsed time
        self.step_count = 0  # Keep track of time

        # If there are any comments to describe the experiment
        if self.comment_enable:
            comment = input('Comment: ')
        else:
            comment = ''

        # Create file with current date, setting, and comment
        with open(self.log_name_csv, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Date', self.date, 'Config settings', self.config, 'Comment', comment])

        with open(self.log_name_txt, 'w') as file:
            file.write('Date:' + self.date)
            file.write('\nConfiguration settings: \n')
            for key, value in self.config.items():
                file.write(str(key) + ': ' + str(value) + '\n')
            file.write('\nInitial comment on the experiment:' + comment)

        return True

    def log(self, x, u, f, env):
        # Get log info
        # metrics = self.get_metric()
        # events = self.get_event()

        # Create line to append with step count, inputs, metrics, events
        current_line = ['Step', self.step_count, 'State', np.array(x).flatten(), 'Control', np.array(u).flatten(), 'f', np.array(f).flatten(), 'Env', np.array(env).flatten()]

        # Print line
        if self.log_freq > 0:
            if self.step_count % self.log_freq == 0:
                print('\n', current_line)

        # Save line
        if self.save_freq > 0:
            if self.step_count % self.save_freq == 0:
                self.save(current_line)

        self.step_count += 1  # Update step counter

        return current_line

    def get_metric(self):
        pass

    def def_metric(self):
        pass

    def def_event(self):
        pass

    def get_event(self):
        pass

    def save(self, current_line):
        # Save line appending it to the csv file
        with open(self.log_name_csv, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(current_line)
        with open(self.log_name_txt, 'a') as file:
            file.write('\n')
            for el in current_line:
                file.write('\n' + str(el))
            file.write('\n')
        return True

    def close(self):
        self.end = time.time()  # Get end time for elapsed time

        # Eventually get final comments on the simulation
        if self.comment_enable:
            comment = input('\nComment: ')
        else:
            comment = ''

        #  Save final row
        with open(self.log_name_csv, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Done', 'Elapsed time [s]:', self.end-self.start, 'Comments: ', comment])
        with open(self.log_name_txt, 'a') as file:
            file.write('Done. \nElapsed time [s]:' + str(self.end-self.start) + '\nComments: ' + comment + '\n')
        return True

