"""
SwarmSim: A Large-Scale Multi-Agent Simulator
==============================================

SwarmSim is a comprehensive Python framework for simulating large-scale multi-agent systems
with stochastic dynamics. It provides a modular architecture for defining agent populations,
environments, interactions, controllers, and integrators.

Key Features
------------
- **Modular Design**: Easily extensible components for populations, environments, interactions, etc.
- **Stochastic Dynamics**: Support for stochastic differential equations with drift and diffusion
- **Flexible Configuration**: YAML-based configuration system for all simulation parameters
- **Scalable**: Designed to handle large numbers of agents efficiently
- **Visualization**: Built-in rendering capabilities for real-time visualization
- **Logging**: Comprehensive data logging for analysis and visualization

Main Components
---------------
- **Populations**: Define agent behavior and dynamics (e.g., Brownian motion, double integrators)
- **Environments**: Specify the physical or virtual space where agents operate
- **Interactions**: Model agent-to-agent and agent-to-environment interactions
- **Controllers**: Implement control strategies for guiding agent behavior
- **Integrators**: Numerical integration schemes for stochastic differential equations
- **Simulators**: Orchestrate the simulation loop and coordinate all components
- **Loggers**: Record simulation data for analysis
- **Renderers**: Provide real-time visualization of simulations

Examples
--------
Basic usage example:

.. code-block:: python

    from swarmsim import BrownianMotion, Simulator
    from swarmsim.Environments import EmptyEnvironment
    from swarmsim.Integrators import EulerMaruyama
    
    # Create population
    population = BrownianMotion('config.yaml')
    
    # Create environment
    environment = EmptyEnvironment('config.yaml')
    
    # Create integrator
    integrator = EulerMaruyama('config.yaml')
    
    # Create and run simulator
    simulator = Simulator('config.yaml', [population], environment, integrator)
    simulator.simulate()

See Also
--------
For detailed documentation and examples, visit the project documentation at [Documentation](https://sincrogroup.github.io/large_scale_multi_agent_simulator/).
"""
