IntelliSwarm Package
====================

The IntelliSwarm package provides a comprehensive framework for multi-agent simulation and analysis. 
The package is organized into several key modules, each serving specific functions in the simulation pipeline.

.. automodule:: swarmsim
   :members:
   :undoc-members:
   :show-inheritance:

Package Overview
----------------

The SwarmSim framework consists of the following main components:

* **Controllers**: Implement control algorithms for agent guidance and coordination
* **Environments**: Define simulation environments and boundary conditions
* **Integrators**: Provide numerical integration methods for agent dynamics
* **Interactions**: Model inter-agent forces and communication
* **Loggers**: Handle data collection and export functionality
* **Populations**: Define agent populations and their behavioral models
* **Renderers**: Provide visualization and rendering capabilities
* **Simulators**: Coordinate simulation execution and component integration
* **Utils**: Supply utility functions for common simulation tasks

Architecture
------------

SwarmSim follows a modular architecture that allows for flexible composition of simulation components:

.. code-block:: python

    # Basic simulation setup
    simulator = Simulator(
        populations=[population1, population2],  # Multiple agent types
        environment=environment,                 # Simulation environment
        integrator=integrator,                  # Numerical integration
        interactions=[interaction1, interaction2], # Inter-agent forces
        controllers=[controller],               # Control algorithms
        loggers=[logger],                      # Data collection
        renderer=renderer                      # Visualization
    )

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   swarmsim.Controllers
   swarmsim.Environments
   swarmsim.Integrators
   swarmsim.Interactions
   swarmsim.Loggers
   swarmsim.Populations
   swarmsim.Renderers
   swarmsim.Simulators
   swarmsim.Utils



