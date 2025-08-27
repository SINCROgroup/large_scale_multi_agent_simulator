"""
swarmsim.Utils
==============

This module provides essential utility functions for multi-agent simulation setup, execution, and analysis.

The Utils module contains a comprehensive collection of utility functions that support
all aspects of multi-agent simulation development, from initialization and parameter
management to data logging and visualization. These utilities streamline simulation
setup and provide robust tools for analysis and debugging.

Module Contents
---------------
**Core Utilities**:

    - `load_config` : YAML configuration file loading and validation
    - `set_global_seed` : Reproducible random number generation across frameworks
    - `get_states` : Agent initialization from files or random distributions
    - `get_parameters` : Parameter loading and generation for populations

**Control and Interaction Utilities**:

    - `compute_distances` : Efficient pairwise distance computation
    - `gaussian_input` : 2D Gaussian function evaluation for spatial fields

**Logging and Data Management**:

    - `add_entry`, `append_entry` : Structured data logging utilities
    - `get_positions` : Population state extraction for logging
    - `print_log` : Console output formatting
    - `append_txt`, `append_csv` : File output utilities
    - `save_npz`, `save_mat` : Binary data serialization

**Shepherding Analysis**:

    - `get_target_distance` : Distance computation to goal regions
    - `xi_shepherding` : Shepherding success metric calculation
    - `get_done_shepherding` : Termination condition evaluation

**Visualization Utilities**:

    - `get_snapshot` : Pygame surface screenshot capture

Key Features
------------
**Configuration Management**:

- YAML-based configuration loading with validation
- Nested configuration structure support
- Error handling for missing files and malformed configs

**Initialization Flexibility**:

- Multiple initialization modes (random, file-based)
- Support for box and circular spatial distributions
- Parameter generation with various statistical distributions
- File format support (CSV, NPZ) with validation

**Performance Optimization**:

- Vectorized distance computations for large populations
- Efficient memory management for parameter generation
- Optimized data structures for logging operations

**Data Analysis Tools**:

- Shepherding metrics and success criteria
- Statistical analysis utilities
- Flexible data export formats

Notes
-----
- All utilities support numpy arrays and standard Python data types
- Configuration validation prevents common setup errors
- Reproducible random number generation across multiple frameworks
- Flexible data serialization supports various analysis workflows
"""

from swarmsim.Utils.control_utils import *
from swarmsim.Utils.shepherding_utils import *
from swarmsim.Utils.logger_utils import *
from swarmsim.Utils.sim_utils import *
from swarmsim.Utils.init_utils import get_states
from swarmsim.Utils.plot_utils import *
from swarmsim.Utils.params_utils import get_parameters

