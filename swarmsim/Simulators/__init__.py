"""
swarmsim.Simulators
===================

This module provides simulation engines for orchestrating multi-agent system simulations.

The Simulators module contains the core simulation engines that coordinate all aspects
of multi-agent simulations, from basic research simulations to specialized reinforcement
learning environments. These simulators manage the execution flow, component integration,
and timing control for complex multi-agent systems.

Module Contents
---------------
    - `Simulator` : Main simulation engine for research and analysis applications
    - `GymSimulator` : OpenAI Gym-compatible simulator for reinforcement learning

Key Features
------------
**Simulation Orchestration**:

- Component coordination (populations, environment, interactions, controllers)
- Execution flow management with proper sequencing
- Progress tracking and early termination handling
- Real-time visualization and data logging integration

**Performance Optimization**:

- Efficient timestep management and loop optimization
- Configurable sampling rates for different components
- Memory-efficient state management
- Parallel interaction computation support

**Flexibility and Extensibility**:

- Modular architecture supporting custom components
- Configuration-driven simulation parameters
- Support for multiple simulation paradigms
- Integration with visualization and analysis tools

Examples
--------
Basic simulation setup:

.. code-block:: python

    from swarmsim.Simulators import Simulator
    from swarmsim.Populations import BrownianMotion
    from swarmsim.Environments import EmptyEnvironment
    from swarmsim.Integrators import EulerMaruyama

    # Create simulation components
    population = BrownianMotion(config_path="config.yaml")
    environment = EmptyEnvironment(config_path="config.yaml")  
    integrator = EulerMaruyama(config_path="config.yaml")

    # Create and run simulation
    simulator = Simulator(
        config_path="config.yaml",
        populations=[population],
        environment=environment,
        integrator=integrator
    )
    simulator.simulate()

Reinforcement learning setup:

.. code-block:: python

    from swarmsim.Simulators import GymSimulator
    import numpy as np

    # Create RL-compatible simulator
    gym_sim = GymSimulator(
        populations=[sheep, herders],
        interactions=[repulsion, attraction],
        environment=shepherding_env,
        integrator=integrator,
        logger=logger,
        renderer=renderer,
        config_path="rl_config.yaml"
    )

    # RL training loop
    for episode in range(num_episodes):
        gym_sim.reset()
        for step in range(max_steps):
            action = agent.get_action(observation)
            gym_sim.step(action)
            gym_sim.render()

Configuration
-------------
Simulator configuration via YAML:

.. code-block:: yaml

    simulator:
        T: 100.0          # Total simulation time
        dt: 0.01          # Timestep (inherited from integrator)
        progress_bar: true # Show progress during simulation

Notes
-----
- All simulators coordinate multiple simulation components in proper sequence
- Timestep management ensures numerical stability and accuracy
- Component sampling rates can be configured independently
- Early termination is supported through logger done flags
- Memory management prevents accumulation of interaction forces between timesteps
"""

from swarmsim.Simulators.base_simulator import Simulator
from swarmsim.Simulators.gym_simulator import GymSimulator

