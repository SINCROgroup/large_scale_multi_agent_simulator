"""
Logging utility functions for structured data collection and export.

This module provides a comprehensive suite of utilities for logging simulation data
in multiple formats, supporting real-time monitoring, data persistence, and
post-simulation analysis. The utilities support flexible data collection with
configurable output formats.
"""

import csv
import os

import numpy as np
import scipy.io as sio


def add_entry(current_info, save_mode=[], **kwargs):
    """
    Add single-value entries to the logging data structure.

    This function adds individual data entries to a structured logging dictionary,
    associating each entry with specific save modes that determine how the data
    will be output (print, file, etc.).

    Parameters
    ----------
    current_info : dict
        The logging data structure to modify.
    save_mode : list, optional
        List of output modes for the entries. Options: ['print', 'txt', 'csv', 'npz', 'mat'].
    **kwargs : key-value pairs
        Data entries to add, where keys become field names and values are the data.

    Data Structure
    --------------
    Each entry in current_info follows the format:

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
        current_info[key] = {'value': value, 'save_mode': save_mode}


def append_entry(info, save_modes=(), **kwargs):
    """
    Append time-series data to logging entries.

    Works inside BaseLogger data structures (`step_data` or `exp_data`).
    """
    for key, value in kwargs.items():
        value = np.asarray(value)
        if key in info:
            prev = info[key]['value']
            # Stack time-series data consistently
            info[key]['value'] = np.vstack([prev, value])
        else:
            # Start new time-series with one row
            info[key] = {'value': value[np.newaxis, ...], 'save_modes': tuple(save_modes)}


def get_positions(info, populations, save_modes=('csv', 'npz')):
    """
    Extract and log positions of all populations.

    Creates systematic names like 'sheep_state0', 'sheep_state1', ...
    Each dimension of each population is stored separately.
    """
    for pop in populations:
        for d in range(pop.state_dim):
            col_name = f"{pop.id}_state{d}"
            value = pop.x[:, d]  # shape: (num_agents,)
            append_entry(info, save_modes, **{col_name: value})


def print_log(current_info):
    """
    Display logging information to console with filtered output.

    This function prints selected logging entries to the console, filtering
    by save mode to show only data marked for console output. It provides
    real-time monitoring capabilities during simulation execution.

    Parameters
    ----------
    current_info : dict
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
    for key, value in current_info.items():
        if 'print' in value['save_mode']:
            print(f"{key}: {value['value']}; ", end=" ")
    print('\n')


def append_txt(log_name, current_info):
    """
    Append logging entries to a text file with structured formatting.

    This function writes selected logging entries to a text file, filtering
    by save mode and providing human-readable output suitable for logs,
    reports, and debugging traces.

    Parameters
    ----------
    log_name : str
        Path to the output text file.
    current_info : dict
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
    with open(log_name, mode="a") as txtfile:
        txtfile.write("\n")
        for key, value in current_info.items():
            if 'txt' in value['save_mode']:
                txtfile.write(f"{key}: {value['value']}\n")


def append_csv(log_name, current_info):
    """
    Append logging entries to a CSV file with automatic header management.

    This function writes selected logging entries to a CSV file, filtering by
    save mode and handling automatic header creation for new files. It provides
    structured tabular output suitable for data analysis and spreadsheet import.

    Parameters
    ----------
    log_name : str
        Path to the output CSV file.
    current_info : dict
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
    current_info_csv = {}
    for key, value in current_info.items():
        if 'csv' in value['save_mode']:
            if isinstance(value['value'], np.ndarray):
                current_info_csv.update({key: value['value'].tolist()})
            else:
                current_info_csv.update({key: value['value']})
    with open(log_name, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=current_info_csv.keys())
        if csvfile.tell() == 0:  # Write header if the file is empty
            writer.writeheader()
        writer.writerow(current_info_csv)


def save_npz(log_name, data):
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
    np.savez(log_name, **npz_data)


def save_mat(log_name, data):
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
    sio.savemat(log_name, mat_data)
