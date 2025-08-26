from abc import ABC, abstractmethod
import numpy as np


class Logger(ABC):
    """
    Abstract base class for data logging in multi-agent simulations.

    This class defines the interface for recording simulation data, including agent states,
    environment information, and custom metrics. Loggers can write data to various formats
    (CSV, NPZ, HDF5, etc.) and provide mechanisms for early simulation termination.

    Notes
    -----
    Subclasses must implement the abstract methods:
    
    - `reset()`: Initialize logging for a new simulation run
    - `log()`: Record data at each timestep and return termination flag
    - `close()`: Finalize logging and close files/connections

    The logger operates in the simulation loop and can influence simulation control
    by returning a termination flag from the `log()` method.

    """

    def __init__(self) -> None:
        """
        Initialize the Logger base class.

        Subclasses should call this constructor and then initialize their specific
        logging infrastructure (files, databases, network connections, etc.).
        """
        super().__init__()

    @abstractmethod
    def reset(self):
        """
        Initialize or reset the logger for a new simulation run.

        This method should prepare the logger for a new simulation by clearing previous
        data, creating new files, or resetting internal state. It is called before
        the simulation loop begins.

        Notes
        -----
        Implementations should:
        
        - Clear any accumulated data from previous runs
        - Create new output files or database entries
        - Initialize timestamps and counters
        - Set up any required data structures
        """
        pass

    @abstractmethod
    def log(self, data: dict | None =None) -> bool:
        """
        Record simulation data at the current timestep.

        This method is called at each simulation timestep to record relevant data.
        It can log agent positions, velocities, environment state, performance metrics,
        or any other simulation data. The method can also signal early termination.

        Parameters
        ----------
        data : dict or None, optional
            Dictionary containing custom data to log, with format {variable_name: value}.
            If None, the logger should record default simulation data. Default is None.

        Returns
        -------
        bool
            Flag indicating whether the simulation should terminate early. If True,
            the simulation loop will exit before reaching the specified end time.

        Notes
        -----
        Implementations should:
        
        - Record current simulation state (time, agent positions, etc.)
        - Process and store custom data if provided
        - Update any running calculations (averages, statistics, etc.)
        - Check termination conditions (convergence, time limits, etc.)
        - Return True only if early termination is desired

        The logger has access to all simulation components and can extract data
        from populations, environment, controllers, and interactions.
        """

    @abstractmethod
    def close(self):
        """
        Finalize logging and clean up resources.

        This method is called at the end of a simulation to properly close files,
        save final data, and clean up any resources used by the logger. It should
        ensure all data is safely stored and accessible for analysis.

        Notes
        -----
        Implementations should:
        
        - Close any open files or database connections
        - Save accumulated data to persistent storage
        - Write metadata (simulation parameters, timing info, etc.)
        - Compress or archive data if appropriate
        - Clean up temporary files or memory structures
        - Print summary information if desired

        This method is called even if the simulation terminates early due to
        the logger returning True from the `log()` method.
        """
        pass
