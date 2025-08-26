from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry, get_positions,append_entry
import numpy as np


class PositionLogger(BaseLogger):
    """
    Position-based data logger for tracking agent spatial dynamics.

    This logger extends the BaseLogger to capture and record detailed positional
    information about agent populations throughout the simulation. It automatically
    logs agent positions, computes position-based statistics, and tracks control
    inputs when available.

    The logger captures position data at regular intervals and saves it in multiple
    formats (CSV, NPZ, MAT) for analysis and visualization.

    Parameters
    ----------
    populations : list of Population
        List of population objects whose positions will be logged.
    environment : Environment
        Environment object containing spatial context and boundaries.
    config_path : str
        Path to the YAML configuration file containing logger parameters.

    Attributes
    ----------
    populations : list of Population
        Population objects being monitored for position data.
    environment : Environment
        Environment context for spatial logging, inherited from BaseLogger.
    global_info : dict
        Accumulated data structure for position information across timesteps.
    step_count : int
        Current simulation step counter, inherited from BaseLogger.
    save_freq : int
        Frequency (in steps) for saving position data, inherited from BaseLogger.

    Config Requirements
    -------------------
    The YAML configuration file must contain logger parameters under the class section:

    PositionLogger : dict
        Configuration section for the position logger:
        
        - ``activate`` : bool, optional 
          Enable/disable logging. Default: ``True``
        - ``log_freq`` : int, optional 
          Print frequency (0 = never). Default: ``0``
        - ``save_freq`` : int, optional 
          Save frequency (0 = never). Default: ``1``
        - ``save_data_freq`` : int, optional 
          Data save frequency. Default: ``0``
        - ``save_global_data_freq`` : int, optional 
          Global data save frequency. Default: ``0``
        - ``log_path`` : str, optional 
          Output directory path. Default: ``"./logs"``
        - ``log_name`` : str, optional 
          Log file name suffix. Default: ``""``
        - ``comment_enable`` : bool, optional 
          Enable experiment comments. Default: ``False``

    Notes
    -----
    **Data Capture:**

    The logger automatically captures:

    - **Agent Positions**: Full position arrays for all agents in all populations
    - **Control Inputs**: Mean control input values when available (e.g., ``u`` for first population)
    - **Temporal Information**: Timestep and timing data for synchronization

    **File Formats:**

    Position data is saved in multiple formats for flexibility:

    - **CSV**: Human-readable comma-separated values for spreadsheet analysis
    - **NPZ**: Compressed NumPy format for efficient Python data loading
    - **MAT**: MATLAB format for analysis in MATLAB/Octave

    **Performance Considerations:**

    - Data is accumulated in memory and saved at specified intervals
    - Memory usage scales with number of agents and save frequency
    - For large populations, consider increasing ``save_freq`` to reduce I/O overhead


    Examples
    --------
    **Basic Configuration:**

    .. code-block:: yaml

        PositionLogger:
            activate: true
            save_freq: 10
            log_path: "./simulation_logs"
            log_name: "position_data"

    """
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_internal_data(self, save_mode=['csv','npz','mat']):
        """
        Log position data and control for all populations.

        This method captures detailed position information for all agents in all populations
        and information about control inputs. Data is saved according to
        the specified save frequency and formats.

        Parameters
        ----------
        save_mode : list of str, optional
            List of output formats for data saving. Default: ``['csv','npz','mat']``
            
            - ``'csv'`` : Comma-separated values for spreadsheet analysis
            - ``'npz'`` : Compressed NumPy format for Python analysis
            - ``'mat'`` : MATLAB format for MATLAB/Octave analysis
            - ``'print'`` : Console output for debugging
            - ``'txt'`` : Human-readable text format

        Notes
        -----
        **Position Data Capture:**

        The method uses ``get_positions()`` utility to extract:

        - Full position arrays for all agents in all populations
        - Timestep information
        - Population identifiers for multi-population simulations

        **Control Input:**

        For the first population (index 0), computes and logs:

        - Mean control input value: ``np.mean(populations[0].u)``
        - Useful for monitoring control effort and system behavior

        **Save Frequency:**

        Data logging occurs only when ``step_count % save_freq == 0``, ensuring
        efficient memory usage and I/O performance for long simulations.

        **Data Format:**

        The logged data structure includes:

        - ``positions_Population_N`` : Position arrays for population N
        - ``u`` : Mean control input for first population
        - ``step`` : Current simulation step
        - ``time`` : Simulation time or timestamp
        
        """
        if self.step_count % self.save_freq == 0:
            get_positions(self.global_info, self.populations, save_mode)
            append_entry(self.global_info, save_mode, **{"u": np.mean(self.populations[0].u)})
            super().log_internal_data(save_mode)
        
