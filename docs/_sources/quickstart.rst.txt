Quick Start Guide
=================

This guide will get you up and running with SwarmSim in just a few minutes. We'll walk through
creating your first simulation, understanding the core concepts, and exploring different types
of multi-agent behaviors.

Your First Simulation
---------------------

Let's start with a simple Brownian motion simulation:

.. code-block:: python

    import sys
    import os
    import pathlib

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from swarmsim.Populations import BrownianMotion
    from swarmsim.Populations import FixedPopulation
    from swarmsim.Interactions import HarmonicRepulsion
    from swarmsim.Integrators import EulerMaruyamaIntegrator
    from swarmsim.Renderers import BaseRenderer
    from swarmsim.Loggers.position_logger import PositionLogger
    from swarmsim.Simulators import Simulator
    from swarmsim.Environments import EmptyEnvironment


    config_path = str(pathlib.Path(__file__).resolve().parent.parent/"Configuration"/"base_config.yaml")

    # Create populations
    population1 = BrownianMotion(config_path)
    population2 = FixedPopulation(config_path)
    populations = [population1, population2]


    # Create an empty environment (no boundaries or obstacles)
    environment = EmptyEnvironment(config_path)

    # Create interactions
    repulsion_12 = HarmonicRepulsion(population1, population2, config_path)
    interactions = [repulsion_12]

    # Setup Integrator
    integrator = EulerMaruyamaIntegrator(config_path)

    # Setup Renderer and Logger
    renderer = BaseRenderer(populations, environment, config_path)
    logger = PositionLogger(populations, environment, config_path)

    # Create the simulator
    simulator = BaseSimulator(
        populations=[population],
        environment=environment,
        dt=0.01                        # Time step size
    )

    # Setup Simulator
    simulator = Simulator(populations=populations, interactions=interactions, environment=environment,
                          integrator=integrator, logger=logger, renderer=renderer, config_path=config_path)

    # Run the simulation
    simulator.simulate()


Core Concepts
-------------

SwarmSim is built around several key components that work together:

**Populations**
    Define the agents and their behavioral models (Brownian motion, flocking, etc.)

**Environments** 
    Specify the simulation space, boundaries, and environmental forces

**Interactions**
    Model forces between agents (attraction, repulsion, alignment, etc.)

**Integrators**
    Handle the numerical integration of agent dynamics

**Simulators**
    Coordinate all components and manage the simulation execution

**Controllers**
    Implement control algorithms for guiding agent behavior

**Loggers**
    Handle data collection and export

**Renderers**
    Visualize the simulation results (e.g., 2D/3D plots, animations)


Configuration Files
-------------------

For all simulations, use YAML configuration files:

.. code-block:: yaml

    # BrownianMotion configuration
    BrownianMotion:
        N: 2000
        id: BrownianMotion
        initial_conditions:
            mode: Random
            random:
            box:
                lower_bounds:
                - -25
                - -25
                upper_bounds:
                - 25
                - 25
            shape: box
        parameters:
            generate:
            D:
                args:
                loc: 1
                scale: 0.1
                homogeneous: 0
                sampler: normal
            mu:
                positional_args:
                - -1
                - 1
                sampler: uniform
                shape:
                - 2
            mode: generate
        state_dim: 2

    # FixedPopulation configuration
    FixedPopulation:
        N: 100
        id: Fixed
        initial_conditions:
            mode: Random
            random:
            circle:
                max_radius: 10
                min_radius: 0
            shape: circle
        state_dim: 2

    # HarmonicRepulsion configuration
    HarmonicRepulsion:
        id: Interaction Force
        parameters:
            generate:
            distance: 10
            strength: 0.1
            mode: generate

    # PositionLogger configuration
    PositionLogger:
        activate: 1
        comment_enable: 0
        log_freq: 0
        log_name: Base_Simulation
        log_path: ./logs
        save_data_freq: 0
        save_freq: 500
        save_global_data_freq: 1

    # Environment configuration
    environment:
        dimensions:
        - 50
        - 50

    # Integrator configuration
    integrator:
        dt: 0.01

    # Renderer configuration
    renderer:
        agent_colors:
        - blue
        - red
        agent_shapes:
        - circle
        - diamond
        agent_sizes:
        - 1
        - 1
        background_color: white
        render_dt: 0.01
        render_mode: pygame

    # Simulator configuration
    simulator:
        T: 10
        dt: 0.001


Running Simulations using GUI
-----------------------------

To run simulations using the GUI, run the following command in your terminal:

```bash
streamlit run swarmsim/streamlit_gui.py
```

Next Steps
----------

Now that you understand the basics:

1. **Explore Examples**: Check the ``Examples/`` directory for more complex scenarios
2. **Read API Documentation**: Dive deep into specific modules
3. **Join the Community**: Contribute to the project on GitHub
4. **Build Custom Components**: Create your own populations, interactions, and controllers


Remember: IntelliSwarm is designed to be modular and extensible. Don't hesitate to create
custom components when the built-in options don't fit your specific needs!