Utilities Package
=================

The Utils package provides essential utility functions and helper modules for SwarmSim operations.
These utilities cover common tasks such as parameter management, data logging, control algorithms,
visualization, and simulation setup.

.. automodule:: swarmsim.Utils
   :members:
   :undoc-members:
   :show-inheritance:

Package Overview
----------------

The Utils package is organized into specialized modules:

* **Parameter Management**: Loading and generating agent parameters from files or distributions
* **Data Logging**: Comprehensive data collection and export in multiple formats
* **Control Utilities**: Mathematical functions for control system implementation
* **Initialization**: Helper functions for setting up simulations and populations
* **Visualization**: Plotting utilities for data analysis and visualization
* **Shepherding**: Specialized utilities for shepherding behavior analysis
* **Simulation**: General simulation management and utility functions

Core Modules
------------

Parameter Utilities
~~~~~~~~~~~~~~~~~~~

The parameter utilities module provides flexible parameter loading and generation capabilities.

.. automodule:: swarmsim.Utils.params_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **File-based Loading**: Load parameters from CSV or NPZ files
- **Statistical Generation**: Generate parameters using 30+ probability distributions
- **Population Matching**: Automatically adjust parameter count to match population size
- **Shape Broadcasting**: Intelligent parameter reshaping for multi-dimensional properties
- **Validation**: Parameter validation and statistical analysis

Example Usage:

.. code-block:: python

    from swarmsim.Utils import get_parameters
    
    # Load from file
    config = {
        'mode': 'file',
        'file': {'file_path': 'agent_params.csv'}
    }
    
    # Or generate with distributions
    config = {
        'mode': 'generate',
        'generate': {
            'mass': {'sampler': 'normal', 'args': {'loc': 1.0, 'scale': 0.1}},
            'position': {'sampler': 'uniform', 'args': {'low': -5, 'high': 5}, 'shape': [3]}
        }
    }
    
    params = get_parameters(config, shapes={'mass': (), 'position': (3,)}, num_samples=100)

Logger Utilities
~~~~~~~~~~~~~~~~

The logger utilities module handles comprehensive data collection and export functionality.

.. automodule:: swarmsim.Utils.logger_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Multi-format Export**: Support for CSV, NPZ, MAT, and text formats
- **Time Series Logging**: Efficient time series data collection
- **Real-time Monitoring**: Live data streaming and console output
- **Memory Management**: Optimized for large-scale simulations
- **Flexible Save Modes**: Configure output formats per logging entry

Example Usage:

.. code-block:: python

    from swarmsim.Utils import add_entry, append_entry, save_npz
    
    log_data = {}
    
    # Log single values
    add_entry(log_data, save_mode=['csv', 'npz'], 
              simulation_time=100.0, total_agents=200)
    
    # Log time series
    for step in range(1000):
        append_entry(log_data, save_mode=['npz'],
                    positions=population.x, velocities=population.v)
    
    # Export data
    save_npz("simulation_results.npz", log_data)

Control Utilities
~~~~~~~~~~~~~~~~~

Mathematical functions and utilities for implementing control algorithms.

.. automodule:: swarmsim.Utils.control_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Geometric Calculations**: Distance, angle, and spatial relationship functions
- **Control Algorithms**: PID controllers, feedback systems, and stability analysis
- **Coordinate Transformations**: Conversion between coordinate systems
- **Signal Processing**: Filtering and noise reduction utilities

Initialization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

Helper functions for setting up simulations, populations, and environments.

.. automodule:: swarmsim.Utils.init_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Population Setup**: Initialize agent populations with diverse properties
- **Environment Configuration**: Set up simulation environments and boundaries
- **Parameter Loading**: Load configuration files and simulation parameters
- **Validation**: Check configuration consistency and parameter validity

Supporting Modules
------------------

Plot Utilities
~~~~~~~~~~~~~~

Visualization and plotting utilities for data analysis and presentation.

.. automodule:: swarmsim.Utils.plot_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Trajectory Visualization**: Plot agent paths and movement patterns
- **Statistical Plots**: Distribution analysis and statistical summaries
- **Animation Support**: Create animated visualizations of simulation data
- **Export Options**: Save plots in various formats (PNG, PDF, SVG)

Shepherding Utilities
~~~~~~~~~~~~~~~~~~~~~

Specialized utilities for analyzing shepherding behaviors and multi-agent coordination.

.. automodule:: swarmsim.Utils.shepherding_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Herding Metrics**: Calculate shepherding effectiveness and performance
- **Group Analysis**: Analyze group cohesion and formation patterns
- **Control Strategies**: Implement shepherding control algorithms
- **Behavioral Assessment**: Evaluate shepherding behavior quality

Simulation Utilities
~~~~~~~~~~~~~~~~~~~~

General simulation management and utility functions.

.. automodule:: swarmsim.Utils.sim_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Simulation Management**: Start, stop, and monitor simulation execution
- **Performance Monitoring**: Track simulation performance and resource usage
- **Configuration Handling**: Load and validate simulation configurations
- **Error Handling**: Robust error management and recovery


