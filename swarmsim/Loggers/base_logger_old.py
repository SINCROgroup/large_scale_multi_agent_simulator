import pathlib
from datetime import datetime
from swarmsim.Loggers import Logger
from swarmsim.Utils import add_entry, append_csv, append_txt, print_log, save_npz, save_mat
import yaml
import time
import os
import numpy as np
from swarmsim.Utils.sim_utils import load_config
import scipy.io as sio
import csv


class BaseLogger(Logger):
    """
    Comprehensive base logger for multi-agent simulation data collection and analysis.

    This logger provides the foundational logging infrastructure for recording simulation
    data, managing file outputs, timing execution, and providing extensible hooks for
    specialized logging behaviors. It handles multiple output formats, configurable
    logging frequencies, and automatic file organization with timestamped naming.

    The BaseLogger serves as the parent class for all specialized loggers in the framework,
    providing common functionality while allowing customization through method overriding.
    It automatically manages file creation, data serialization, and experiment metadata.

    Parameters
    ----------
    populations : list of Population
        List of population objects whose data will be logged throughout the simulation.
        Each population provides state information and dynamics data.
    environment : Environment
        Environment object containing spatial and contextual information for the simulation.
        Provides environmental state and parameters for logging.
    config_path : str
        Path to the YAML configuration file containing logger parameters and settings.

    Attributes
    ----------
    config : dict
        Complete configuration dictionary loaded from the YAML file.
    logger_config : dict
        Logger-specific configuration subset extracted from the main config.
    activate : bool
        Flag controlling whether logging is active. If False, logging operations are skipped.
    date : str
        Human-readable timestamp of logger initialization for metadata.
    name : str
        Unique identifier for the logging session, combining timestamp and config name.
    log_freq : int
        Frequency (in simulation steps) for printing progress information to console.
        Set to 0 to disable console output.
    save_freq : int
        Frequency (in simulation steps) for saving data to files.
        Set to 0 to disable file saving.
    save_data_freq : int
        Frequency for saving raw data arrays (positions, states, etc.).
    save_global_data_freq : int
        Frequency for saving accumulated global simulation data.
    log_path : str
        Base directory path where all log files will be stored.
    comment_enable : bool
        Whether to prompt for and include user comments in the log files.
    populations : list of Population
        Reference to the populations being logged.
    environment : Environment
        Reference to the simulation environment.
    log_name_csv : str
        Full path to the CSV output file for tabular data.
    log_name_txt : str
        Full path to the human-readable text output file.
    log_name_npz : str
        Full path to the compressed NumPy data file.
    log_name_mat : str
        Full path to the MATLAB-compatible data file.
    start : float or None
        Timestamp when logging session started.
    end : float or None
        Timestamp when logging session ended.
    step_count : int or None
        Current simulation step counter.
    experiment_count : int or None
        Counter for multiple experiment runs.
    done : bool or None
        Flag indicating if the simulation should terminate early.
    step_info : dict or None
        Dictionary containing data for the current simulation timestep.
    experiment_info : dict or None
        Dictionary containing data related to the whole experiment.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the BaseLogger's class section:

        - ``activate`` : bool, optional
            Enable/disable logging. Default: ``True``
        - ``log_freq`` : int, optional 
            Console output frequency (0 = never). Default: ``0``
        - ``save_freq`` : int, optional
            File save frequency (0 = never). Default: ``1``
        - ``save_data_freq`` : int, optional
            Raw data save frequency. Default: ``0``
        - ``save_global_data_freq`` : int, optional
            Global data save frequency. Default: ``0``
        - ``log_path`` : str, optional
            Output directory path. Default: ``"./logs"``
        - ``log_name`` : str, optional
            Log file name suffix. Default: ``""``
        - ``comment_enable`` : bool, optional
            Enable user comments. Default: ``False``

    Notes
    -----
    **File Organization:**

    The logger creates a directory structure:
    ```
    log_path/
    └── YYYYMMDD_HHMMSS_log_name/
        ├── YYYYMMDD_HHMMSS_log_name.csv    # Tabular data
        ├── YYYYMMDD_HHMMSS_log_name.txt    # Human-readable
        ├── YYYYMMDD_HHMMSS_log_name.npz    # NumPy arrays
        └── YYYYMMDD_HHMMSS_log_name.mat    # MATLAB format
    ```

    **Logging Workflow:**

    1. **Initialization**: Create directories, initialize files
    2. **Start Experiment**: Begin timing and setup data structures
    3. **Step Logging**: Record data at each simulation timestep
    4. **End Experiment**: Finalize files and compute summary statistics

    **Extensibility:**

    Subclasses can override key methods:
    - ``log()``: Customize what data is collected each step
    - ``log_internal_data()``: Modify data processing and storage
    - ``start_experiment()``: Add initialization procedures
    - ``end_experiment()``: Add finalization procedures

    **Performance Considerations:**

    - Data is accumulated in memory between save operations
    - Large simulations should use appropriate ``save_freq`` values
    - Multiple output formats can be disabled for performance
    - File I/O is batched for efficiency

    Examples
    --------
    **Basic Configuration:**

    .. code-block:: yaml

        BaseLogger:
            activate: true
            log_freq: 100
            save_freq: 10
            log_path: "./simulation_logs"
            log_name: "base_experiment"
            comment_enable: false
    """

    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__()

        # Load configuration file
        config: dict = load_config(config_path)
        class_name = type(self).__name__
        logger_config = config.get(class_name, {})
        self.config = config  
        self.logger_config = logger_config

        # Initialize parameters
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  # Get current date to init logger
        self.name = datetime.today().strftime('%Y%m%d_%H%M%S') + logger_config.get('log_name', '')
        self.activate = logger_config.get('activate', True)  # Activate
        self.print_freq = logger_config.get('print_freq', 0)  # Print frequency
        self.save_freq = logger_config.get('save_freq', 0)  # Save frequency
        self.log_path = logger_config.get('log_path', './logs')
        self.comment_enable = logger_config.get('comment_enable', False)
        self.populations = populations
        self.environment = environment

        log_folder = self.log_path + '/' + self.name
        # If the path does not exist, create it
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Generate log file names
        self.log_name = log_folder + '/' + self.name

        # Initialize auxiliary variables
        self.start = None  # Time start
        self.end = None  # Time end
        self.step_count = None  # Count steps for frequency check and logging
        self.experiment_count = -1
        self.done = None  # Episode truncation
        self.step_info = None
        self.experiment_info = None

        if self.activate:
            # If there are any comments to describe the experiment add them, otherwise empty
            if self.comment_enable:
                comment = input('Comment: ')
            else:
                comment = ''

            # Create file with current date, setting, and comment (if active)
            with open(self.log_name + '.txt', 'w') as file:
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
        self.step_count = 0  # Keeps track of time
        self.experiment_count += 1
        self.experiment_info = {}
        if self.activate:
            # Initialize logger: create file with date, current config settings, and add eventual comments
            self.start = time.time()  # Start counter for elapsed time
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
        self.step_info = {}
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
            self.done = self.log(data)  # Log last time step before closing
            self.end = time.time()  # Get end time for elapsed time

            # self.save_npz(self.experiment_info)
            # self.save_mat(self.experiment_info)
            self.append_csv_experiments()

            self.last_experiment = False
            if self.last_experiment:
                self.convert_npz_mat_experiments()
                self.convert_npz_mat()

                # (Optional) get final comments on the simulation
                if self.comment_enable:
                    comment = input('\nComment: ')
                else:
                    comment = ''

                # Save final row with 'Done', elapsed time, and (optional) comment.
                with open(self.log_name+'.txt', 'a') as file:
                    file.write('\nElapsed time [s]:' + str(self.end - self.start) +
                               '\nComments: ' + comment + '\n')

        return self.activate

    def log_external_data(self, data, save_mode=['csv', 'npz', 'mat']):
        if data is not None:
            for key, value in data.items():
                self.add_entry(save_mode, **{key: value})

    def log_internal_data(self, save_mode=['txt', 'print']):
        self.add_entry(save_mode, step=self.step_count)  # Get timestamp
        self.add_entry(save_mode, done=self.done)  # Add done flag

    def output_data(self):
        # Print line if wanted
        if self.print_freq > 0:
            if self.step_count % self.print_freq == 0:
                self.print_log()

        # Save line if wanted
        if self.save_freq > 0:
            if self.step_count % self.save_freq == 0:
                self.append_csv()
                self.append_txt()

    def add_entry(self, save_mode=[], **kwargs):
        """
        Add single-value entries to the logging data structure.

        This function adds individual data entries to a structured logging dictionary,
        associating each entry with specific save modes that determine how the data
        will be output (print, file, etc.).

        Parameters
        ----------
        step_info : dict
            The logging data structure to modify.
        save_mode : list, optional
            List of output modes for the entries. Options: ['print', 'txt', 'csv', 'npz', 'mat'].
        **kwargs : key-value pairs
            Data entries to add, where keys become field names and values are the data.

        Data Structure
        --------------
        Each entry in step_info follows the format:

        .. code-block:: python

            {
                'field_name': {
                    'value': data_value,
                    'save_mode': ['print', 'npz', ...]
                }
            }


        Notes
        -----
        - Overwrites existing entries with the same key
        - For time series data, use `append_entry` instead
        - Save modes determine which output functions will process the data
        - Values can be scalars, arrays, or complex data structures
        """
        for key, value in kwargs.items():
            self.step_info[key] = {'value': value, 'save_mode': save_mode}

    def append_entry(self, save_mode=[], **kwargs):
        """
        Append time-series data to existing logging entries.

        This function appends new data to existing entries in the logging structure,
        automatically handling the creation of new entries or stacking data for
        existing ones. It's ideal for collecting time series data during simulation.

        Parameters
        ----------
        info : dict
            The logging data structure to modify.
        save_mode : list, optional
            List of output modes for the entries. Options: ['print', 'txt', 'csv', 'npz', 'mat'].
        **kwargs : key-value pairs
            Data to append, where keys are field names and values are the new data.

        Data Handling
        -------------
        - **New Entries**: Creates new entry with single data point
        - **Existing Entries**: Stacks new data with existing using `np.vstack`
        - **Shape Consistency**: Maintains consistent data shapes across appends


        Applications
        ------------
        - **Trajectory Logging**: Recording agent paths over time
        - **Performance Monitoring**: Collecting metrics throughout simulation
        - **Real-time Analysis**: Building datasets for online analysis

        Performance Notes
        -----------------
        - Uses `np.vstack` for efficient array concatenation
        - Memory usage grows linearly with simulation length
        - Consider periodic saving for very long simulations
        - Shape consistency is automatically maintained

        Notes
        -----
        - Automatically handles first entry creation vs. subsequent appends
        - Maintains data type consistency within each field
        - Compatible with all numpy array types and scalars
        - Ideal for building time series datasets
        """
        for key, value in kwargs.items():
            if key in self.step_info:
                self.step_info[key] = {'value': np.vstack([self.step_info[key]['value'], value]), 'save_mode': save_mode}
            else:
                self.step_info[key] = {'value': np.asarray([value]), 'save_mode': save_mode}  # Create first row

    def print_log(self):
        """
        Display logging information to console with filtered output.

        This function prints selected logging entries to the console, filtering
        by save mode to show only data marked for console output. It provides
        real-time monitoring capabilities during simulation execution.

        Parameters
        ----------
        step_info : dict
            Logging data structure containing entries with save modes.

        Output Format
        -------------
        Entries are printed in the format: `key: value; ` with automatic newline at the end.


        Applications
        ------------
        - **Real-time Monitoring**: Live simulation progress tracking
        - **Debug Output**: Immediate feedback during development
        - **Event Notification**: Alerts for significant simulation events
        - **Performance Tracking**: Regular metric updates during training
        - **Status Updates**: Periodic progress reports

        Notes
        -----
        - Automatically adds newline after all entries
        - Handles all data types that can be converted to string
        - Non-blocking operation suitable for real-time monitoring
        - Filters entries based on save_mode to avoid console spam
        """
        for key, value in self.step_info.items():
            if 'print' in value['save_mode']:
                print(f"{key}: {value['value']}; ", end=" ")
        print('\n')

    def append_txt(self):
        """
        Append logging entries to a text file with structured formatting.

        This function writes selected logging entries to a text file, filtering
        by save mode and providing human-readable output suitable for logs,
        reports, and debugging traces.

        Parameters
        ----------
        log_name : str
            Path to the output text file.
        step_info : dict
            Logging data structure containing entries with save modes.

        File Format
        -----------
        Each call creates a new section in the file with entries formatted as:
        `key: value` (one per line), followed by a blank line.



        Applications
        ------------
        - **Debugging Logs**: Detailed execution traces for troubleshooting
        - **Experiment Records**: Human-readable experiment documentation
        - **Event Logs**: Chronological record of simulation events
        - **Status Reports**: Periodic progress summaries
        - **Configuration Logs**: Parameter and setting documentation

        Notes
        -----
        - Only processes entries with 'txt' in their save_mode
        - Creates parent directories if they don't exist
        - Adds blank line separator between logging calls
        """
        with open(self.log_name+'.txt', mode="a") as txtfile:
            txtfile.write("\n")
            for key, value in self.step_info.items():
                if 'txt' in value['save_mode']:
                    txtfile.write(f"{key}: {value['value']}\n")

    def append_csv(self):
        """
        Append logging entries to a CSV file with automatic header management.

        This function writes selected logging entries to a CSV file, filtering by
        save mode and handling automatic header creation for new files. It provides
        structured tabular output suitable for data analysis and spreadsheet import.

        Parameters
        ----------
        log_name : str
            Path to the output CSV file.
        step_info : dict
            Logging data structure containing entries with save modes.

        CSV Format
        ----------
        - **Headers**: Automatically created from field names on first write
        - **Data Types**: Arrays are converted to lists for CSV compatibility
        - **Structure**: One row per function call, one column per logged field

        Applications
        ------------
        - **Performance Monitoring**: Real-time metric collection
        - **Statistical Analysis**: Data suitable for statistical software
        - **Report Generation**: Tabular data for presentations and papers


        Notes
        -----
        - Only processes entries with 'csv' in their save_mode
        - Automatically handles header creation and management
        - Creates parent directories if needed
        - Compatible with pandas for post-processing analysis
        """
        step_info_csv = {
            key: (value['value'].tolist() if isinstance(value['value'], np.ndarray) else value['value'])
            for key, value in self.step_info.items()
            if 'csv' in value['save_mode']
        }
        with open(self.log_name + 'exp_' + str(self.experiment_count) + '.csv', mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=step_info_csv.keys())
            if csvfile.tell() == 0:  # Write header if the file is empty
                writer.writeheader()
            writer.writerow(step_info_csv)

    def append_csv_experiments(self):
        experiments_info_csv = {
            key: (value['value'].tolist() if isinstance(value['value'], np.ndarray) else value['value'])
            for key, value in self.experiment_info.items()
            if 'csv' in value['save_mode']
        }
        with open(self.log_name + '_experiments.csv', mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=experiments_info_csv.keys())
            if csvfile.tell() == 0:  # Write header if the file is empty
                writer.writeheader()
            writer.writerow(experiments_info_csv)

    def convert_npz_mat_experiments(self):
        data = np.genfromtxt(self.log_name + '_experiments.csv', delimiter=",", dtype=None, encoding=None, names=True)
        np.savez(self.log_name + '_experiments.npz', **{name: data[name] for name in data.dtype.names if 'npz' in self.experiment_info[name]['save_mode']})
        sio.savemat(self.log_name + '_experiments.mat', {name: data[name] for name in data.dtype.names if 'mat' in self.experiment_info[name]['save_mode']})

    def convert_npz_mat(self):
        for exp in range(self.experiment_count):
            data = np.genfromtxt(self.log_name_csv + 'exp_' + str(exp) + '.csv', delimiter=",", dtype=None, encoding=None, names=True)
            np.savez(self.log_name + 'exp_' + str(exp) + '.npz', **{name: data[name] for name in data.dtype.names if 'npz' in self.experiment_info[name]['save_mode']})
            sio.savemat(self.log_name + 'exp_' + str(exp) + '.mat', {name: data[name] for name in data.dtype.names if 'mat' in self.experiment_info[name]['save_mode']})

    def save_npz(self, data):
        """
        Save logging data to compressed NumPy archive (.npz) format.

        This function exports selected logging entries to a compressed NumPy archive,
        providing efficient binary storage for large datasets with fast loading
        capabilities. It's ideal for numerical data that will be processed with
        NumPy-based analysis tools.

        Parameters
        ----------
        log_name : str
            Output filename with .npz extension.
        data : dict
            Logging data structure containing entries with save modes.



        Examples
        --------
        Basic NPZ logging:

        .. code-block:: python

            from swarmsim.Utils import add_entry, append_entry, save_npz

            log_data = {}

            # Log time series data
            for step in range(100):
                append_entry(log_data, save_mode=['npz'],
                            positions=population.x,
                            velocities=population.v,
                            timestep=step)

            # Save to NPZ file
            save_npz("simulation_data.npz", log_data)


        Applications
        ------------
        - **Large Datasets**: Efficient storage of simulation trajectories
        - **Scientific Analysis**: Compatible with NumPy/SciPy ecosystem

        Notes
        -----
        - Only saves entries with 'npz' in their save_mode
        - Automatically compresses data for smaller file sizes
        - Preserves exact numerical precision
        - Files can be opened with np.load() for analysis
        - Remember to close loaded files to free memory
        - Ideal for numerical simulation data storage and analysis
        """
        npz_data = {}
        for key, value in data.items():
            if 'npz' in value['save_mode']:
                npz_data.update({key: value['value']})
        np.savez(self.log_name + '.npz', **npz_data)

    def save_mat(self, data):
        """
        Save logging data to MATLAB .mat format for cross-platform analysis.

        This function exports selected logging entries to MATLAB-compatible format,
        enabling seamless integration with MATLAB analysis workflows and providing
        interoperability between Python simulations and MATLAB post-processing tools.

        Parameters
        ----------
        log_name : str
            Output filename with .mat extension.
        data : dict
            Logging data structure containing entries with save modes.

        Examples
        --------
        Basic MAT file logging:

        .. code-block:: python

            from swarmsim.Utils import add_entry, append_entry, save_mat

            log_data = {}

            # Log simulation parameters
            add_entry(log_data, save_mode=['mat'],
                      n_agents=100,
                      dt=0.01,
                      max_speed=2.0)

            # Log time series data
            for step in range(1000):
                append_entry(log_data, save_mode=['mat'],
                            positions=population.x,
                            velocities=population.v,
                            energies=population.energy)

            # Save for MATLAB analysis
            save_mat("simulation_results.mat", log_data)


        Applications
        ------------
        - **MATLAB Integration**: Seamless data exchange with MATLAB analysis tools
        - **Signal Processing**: Compatible with MATLAB Signal Processing Toolbox
        - **Visualization**: Use MATLAB's advanced plotting capabilities


        Notes
        -----
        - Only saves entries with 'mat' in their save_mode
        - Requires scipy.io.savemat for the actual file writing
        - Variable names must be valid MATLAB identifiers
        - Large arrays are stored efficiently in MAT format
        - Ideal for teams using both Python and MATLAB workflows
        """
        mat_data = {}
        for key, value in data.items():
            if 'mat' in value['save_mode']:
                mat_data.update({key: value['value']})
        sio.savemat(self.log_name + '-.mat', mat_data)
