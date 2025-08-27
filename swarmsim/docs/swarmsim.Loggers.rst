Data Loggers
============

The loggers package provides comprehensive data collection, recording, and analysis capabilities for multi-agent simulations. These loggers capture simulation dynamics, performance metrics, and behavioral patterns with configurable formats and frequencies.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

SwarmSim's logging framework enables systematic data collection from complex multi-agent simulations, supporting real-time monitoring, post-simulation analysis, and reproducible research workflows. The modular design allows for specialized logging behaviors while maintaining consistent data formats.

Key Features
------------

* **Multi-Format Output**: Support for CSV, NumPy, MATLAB, and custom formats
* **Configurable Frequency**: Adaptive logging intervals and selective data capture
* **Real-Time Monitoring**: Live performance tracking and system diagnostics
* **Memory Efficient**: Streaming writes and chunked processing for large datasets
* **Extensible Architecture**: Custom logger development for specialized applications
* **Automated Organization**: Timestamped directories and structured file naming


Module Reference
----------------

.. automodule:: swarmsim.Loggers
   :members:
   :undoc-members:
   :show-inheritance:

Base Logger Interface
~~~~~~~~~~~~~~~~~~~~~

The abstract foundation for all logging implementations, defining the core interface and common functionality.

.. automodule:: swarmsim.Loggers.base_logger
   :members:
   :undoc-members:
   :show-inheritance:


Standard Logger Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary logger implementation providing comprehensive data collection capabilities.

.. automodule:: swarmsim.Loggers.logger
   :members:
   :undoc-members:
   :show-inheritance:

Features
--------

* **Multi-Population Support**: Handle multiple agent populations simultaneously
* **Flexible Output Formats**: CSV, NumPy, MATLAB, and HDF5 support
* **Real-Time Analytics**: Live computation of statistical measures
* **Memory Optimization**: Efficient handling of large-scale simulations

Position Logger
~~~~~~~~~~~~~~~

Specialized logger focused on trajectory tracking and spatial analysis.

.. automodule:: swarmsim.Loggers.position_logger
   :members:
   :undoc-members:
   :show-inheritance:

Capabilities
------------

* **High-Resolution Tracking**: Detailed trajectory recording
* **Spatial Statistics**: Automatic computation of spatial measures
* **Trajectory Analysis**: Built-in diffusion and mobility metrics
* **Compression**: Efficient storage of position time series

Shepherding Logger
~~~~~~~~~~~~~~~~~~

Domain-specific logger for shepherding simulations with predator-prey dynamics.

.. automodule:: swarmsim.Loggers.shepherding_logger
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Features
--------------------

* **Herding Metrics**: Success rates, containment measures, escape events
* **Predator-Prey Analysis**: Pursuit dynamics, capture statistics
* **Formation Tracking**: Herd cohesion, shape evolution, splitting events
* **Strategic Analysis**: Decision points, behavioral transitions


See Also
--------

* :doc:`swarmsim.Populations` - Population data structures
* :doc:`swarmsim.Simulators` - Simulation control and management
* :doc:`swarmsim.Utils` - Utility functions for data processing
* :doc:`examples.data_analysis` - Data analysis examples and workflows


