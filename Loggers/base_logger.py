from datetime import datetime
import yaml
import time
import os
import csv
import numpy as np
import re


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
        # self.append_freq = logger_config.get('append_freq', 1)  # Append frequency in case of saving batch of lines

        # If the path does not exist, create it
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # If there is no name set in the config file, call it 'trial' + next number available
        file_names = [f for f in os.listdir(self.log_path) if os.path.isfile(os.path.join(self.log_path, f)) and f.startswith('trial')]
        numbers = [re.search(r'(?<=trial)\d+', f).group() for f in file_names if re.search(r'(?<=trial)\d+', f)]
        if not numbers:
            self.name = logger_config.get('log_name', 'trial0.csv')
        else:
            numbers_int = [int(numeric_string) for numeric_string in numbers]
            max_num = np.max(np.array(numbers_int)) + 1
            self.name = logger_config.get('log_name', 'trial' + str(max_num) + '.csv')

        #  Init key variables
        self.log_name = self.log_path + '/' + self.name  # Check concat string
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
        with open(self.log_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Date', self.date, 'Config settings', self.config, 'Comment', comment])

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
        with open(self.log_name, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(current_line)
        return True

    def close(self):
        self.end = time.time()  # Get end time for elapsed time

        # Eventually get final comments on the simulation
        if self.comment_enable:
            comment = input('Comment: ')
        else:
            comment = ''

        #  Save final row
        with open(self.log_name, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Done', 'Elapsed time [s]:', self.end-self.start, 'Comments: ', comment])

        return True

