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
    current_info : dict or None
        Dictionary containing data for the current simulation timestep.
    global_info : dict or None
        Dictionary containing accumulated data across all timesteps.

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
