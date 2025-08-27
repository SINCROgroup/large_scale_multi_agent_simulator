Simulators Package
==================

The Simulators package provides the core simulation engines that coordinate all components
of a multi-agent simulation. Simulators manage the integration of populations, environments,
interactions, controllers, and data logging to create complete simulation systems.

.. automodule:: swarmsim.Simulators
   :members:
   :undoc-members:
   :show-inheritance:

Package Overview
----------------

Simulators in SwarmSim provide:

* **Component Coordination**: Orchestrate interactions between all simulation elements
* **Time Management**: Handle simulation timing, stepping, and synchronization
* **Integration Support**: Work with various numerical integrators and time schemes
* **Performance Monitoring**: Track simulation performance and resource usage
* **Flexible Architecture**: Support different simulation paradigms and use cases

The package includes specialized simulators for different applications, from basic
research simulations to reinforcement learning environments.

Core Modules
------------

Base Simulator
~~~~~~~~~~~~~~

The foundational simulator class that provides core simulation functionality.

.. automodule:: swarmsim.Simulators.base_simulator
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Component Integration**: Seamlessly coordinate populations, environments, and interactions
- **Flexible Stepping**: Support both fixed and adaptive time stepping
- **State Management**: Maintain complete simulation state and history
- **Event Handling**: Process simulation events and state changes
- **Extensible Design**: Easy to extend for specialized simulation needs

Core Capabilities:

.. code-block:: python

    simulator = BaseSimulator(
        populations=[population1, population2],
        environment=environment,
        interactions=[interaction1, interaction2],
        integrator=integrator,
        dt=0.01
    )
    
    # Basic simulation loop
    for step in range(1000):
        simulator.step()

Gym Simulator
~~~~~~~~~~~~~

Specialized simulator implementing OpenAI Gym interface for reinforcement learning applications.

.. automodule:: swarmsim.Simulators.gym_simulator
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **OpenAI Gym Interface**: Standard RL environment interface (step, reset, render)
- **Action/Observation Spaces**: Configurable action and observation spaces
- **Reward Functions**: Flexible reward function definition
- **Episode Management**: Handle episode termination and reset conditions
- **Multi-Agent Support**: Extensions for multi-agent reinforcement learning

RL Integration:

.. code-block:: python

    gym_sim = GymSimulator(
        populations=[agents], 
        environment=env,
        action_space=action_space,
        observation_space=obs_space,
        reward_function=reward_fn
    )
    
    # Standard RL training loop
    obs = gym_sim.reset()
    for step in range(episode_length):
        action = agent.act(obs)
        obs, reward, done, info = gym_sim.step(action)
        if done:
            break

Usage Examples
--------------

Basic Simulation Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Simulators import BaseSimulator
    from swarmsim.Populations import BrownianMotion
    from swarmsim.Environments import EmptyEnvironment
    from swarmsim.Interactions import LennardJones
    from swarmsim.Integrators import EulerMaruyama
    
    # Create simulation components
    population = BrownianMotion(n=100, x_dim=2)
    environment = EmptyEnvironment(boundary_size=[20, 20])
    interaction = LennardJones(epsilon=1.0, sigma=1.0)
    integrator = EulerMaruyama(dt=0.01)
    
    # Create and configure simulator
    simulator = BaseSimulator(
        populations=[population],
        environment=environment,
        interactions=[interaction],
        integrator=integrator,
        dt=0.01
    )
    
    # Run simulation
    for step in range(1000):
        simulator.step()
        
        # Access simulation state
        positions = simulator.get_positions()
        velocities = simulator.get_velocities()

Multi-Population Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Simulators import BaseSimulator
    from swarmsim.Populations import BrownianMotion, SimpleIntegrators
    from swarmsim.Interactions import PowerLawRepulsion
    
    # Create multiple populations
    prey = BrownianMotion(n=200, x_dim=2, species_id=0)
    predators = SimpleIntegrators(n=10, x_dim=2, species_id=1)
    
    # Species-specific interactions
    prey_interaction = PowerLawRepulsion(
        species_pairs=[(0, 0)],  # Prey-prey repulsion
        strength=0.5, 
        power=2
    )
    
    predator_interaction = PowerLawRepulsion(
        species_pairs=[(1, 0)],  # Predator-prey attraction
        strength=2.0, 
        power=1
    )
    
    # Multi-species simulation
    simulator = BaseSimulator(
        populations=[prey, predators],
        environment=environment,
        interactions=[prey_interaction, predator_interaction]
    )

Controlled Simulation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Simulators import BaseSimulator
    from swarmsim.Controllers import ShepherdingController
    from swarmsim.Loggers import Logger
    
    # Setup with controller and logging
    simulator = BaseSimulator(
        populations=[sheep_pop, shepherd_pop],
        environment=environment,
        controllers=[shepherding_controller],
        loggers=[data_logger],
        dt=0.02
    )
    
    # Simulation with automatic logging
    simulator.run(
        total_steps=2000,
        log_interval=10,
        render_interval=50
    )
    
    # Access logged data
    logged_data = simulator.get_log_data()

Reinforcement Learning Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Simulators import GymSimulator
    from swarmsim.Populations import DoubleIntegrators
    from swarmsim.Environments import ShepherdingEnvironment
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    
    # Define action and observation spaces
    n_agents = 5
    action_space = spaces.Box(
        low=-1.0, high=1.0, 
        shape=(n_agents, 2), 
        dtype=np.float32
    )
    
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(n_agents, 4),  # x, y, vx, vy per agent
        dtype=np.float32
    )
    
    def reward_function(state, action, next_state):
        # Custom reward based on formation maintenance
        positions = next_state[:, :2]
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        formation_reward = -np.std(distances)  # Reward tight formation
        return formation_reward
    
    # Create RL environment
    gym_simulator = GymSimulator(
        populations=[DoubleIntegrators(n=n_agents, x_dim=2)],
        environment=ShepherdingEnvironment(),
        action_space=action_space,
        observation_space=observation_space,
        reward_function=reward_function,
        max_episode_steps=1000
    )
    
    # Use with RL algorithms
    obs = gym_simulator.reset()
    for step in range(1000):
        action = policy.predict(obs)  # Your RL policy
        obs, reward, done, info = gym_simulator.step(action)
        if done:
            obs = gym_simulator.reset()


Performance Optimization
------------------------

Simulators are optimized for various performance scenarios:

- **Large Populations**: Efficient algorithms that scale well with agent count
- **Long Simulations**: Memory management for extended simulation runs  
- **Real-time Applications**: Low-latency stepping for interactive use


