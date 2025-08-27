import numpy as np
from swarmsim.Simulators import Simulator


class GymSimulator(Simulator):
    """
    OpenAI Gym-compatible simulator for reinforcement learning applications.

    This simulator extends the base Simulator class to provide a Gym-style interface
    suitable for reinforcement learning algorithms. It implements the standard
    ``reset()``, ``step()``, ``render()``, and ``close()`` methods expected by
    RL frameworks while maintaining full compatibility with the swarmsim ecosystem.

    The GymSimulator is specifically designed for scenarios where one population
    (typically herders or controllers) receives actions from an RL agent, while
    other populations follow their programmed dynamics. This makes it ideal for
    training agents in multi-agent environments like shepherding, swarm control,
    or cooperative robotics.

    Key Features
    ------------
    **RL Compatibility**:
    
    - Standard Gym interface (reset, step, render, close)
    - Configurable render modes including "rgb_array" for headless training
    - Episode-based simulation management
    - Action space integration with swarmsim populations

    **Multi-Agent Integration**:
    
    - Seamless integration with existing swarmsim components
    - Support for mixed controlled/autonomous populations
    - Interaction computation between all agent types
    - Environment state management across episodes

    **Training Optimization**:
    
    - Efficient episode reset without full reinitialization
    - Configurable rendering for training vs evaluation
    - Memory management for long training sessions
    - Component state preservation between episodes

    Parameters
    ----------
    populations : list of Population
        List of agent populations, where populations[1] is typically controlled by RL actions.
    interactions : list of Interaction
        Inter-agent interaction models applied during simulation.
    environment : Environment
        The environment instance containing spatial and physical constraints.
    integrator : Integrator
        Numerical integration scheme for updating agent states.
    logger : Logger
        Data recording component for tracking training metrics.
    renderer : Renderer
        Visualization component with configurable render modes.
    render_mode : str, optional
        Rendering mode: "human" for display, "rgb_array" for numpy arrays. Default is None.
    config_path : str, optional
        Path to YAML configuration file with simulation parameters.

    Attributes
    ----------
    render_mode : str or None
        Current rendering mode configuration.

    Methods
    -------
    reset()
        Reset all simulation components to initial states for new episode.
    step(action)
        Execute one simulation timestep with the provided action.
    render()
        Render current simulation state according to render_mode.
    close()
        Clean up simulation resources and close rendering.

    Examples
    --------
    Basic RL setup for shepherding:

    .. code-block:: python

        from swarmsim.Simulators import GymSimulator
        from swarmsim.Populations import BrownianMotion, SimpleIntegrators
        from swarmsim.Environments import ShepherdingEnvironment

        # Create populations (sheep and herders)
        sheep = BrownianMotion(config_path="sheep_config.yaml")
        herders = SimpleIntegrators(config_path="herder_config.yaml")

        # Create RL-compatible simulator
        gym_sim = GymSimulator(
            populations=[sheep, herders],  # herders[1] will receive RL actions
            interactions=[repulsion, attraction],
            environment=shepherding_env,
            integrator=integrator,
            logger=logger,
            renderer=renderer,
            render_mode="rgb_array"  # For headless training
        )

        # RL training loop
        for episode in range(1000):
            gym_sim.reset()
            done = False
            while not done:
                action = rl_agent.get_action()  # Shape: (n_herders, action_dim)
                gym_sim.step(action)
                frame = gym_sim.render()  # Returns numpy array
                done = gym_sim.logger.check_termination()

    Integration with popular RL libraries:

    .. code-block:: python

        import gym
        from stable_baselines3 import PPO

        # Wrap GymSimulator in Gym environment
        class SwarmEnv(gym.Env):
            def __init__(self):
                self.simulator = GymSimulator(...)
                self.action_space = gym.spaces.Box(...)
                self.observation_space = gym.spaces.Box(...)
            
            def reset(self):
                self.simulator.reset()
                return self._get_observation()
            
            def step(self, action):
                self.simulator.step(action)
                obs = self._get_observation()
                reward = self._compute_reward()
                done = self._check_done()
                return obs, reward, done, {}

        # Train with stable-baselines3
        env = SwarmEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)

    Notes
    -----
    - The controlled population is assumed to be populations[1] by convention
    - Actions are directly assigned to the controlled population's input (u)
    - All populations and interactions are reset between episodes
    - Logger provides episode termination signals through done flags
    - Rendering behavior depends on the render_mode configuration
    """

    def __init__(self, populations, interactions, environment, integrator, logger, renderer, render_mode=None,
                 config_path=None) -> None:

        """        
        Initializes the Simulator class with configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """

        # Load config params from YAML file
        super().__init__(populations=populations,
                         interactions=interactions,
                         environment=environment,
                         controllers=None,
                         integrator=integrator,
                         logger=logger,
                         renderer=renderer,
                         config_path=config_path)

        self.render_mode = render_mode     # IMPLEMENT RENDER MODE

    def reset(self):
        """
        Reset all simulation components to initial states for new episode.

        This method prepares the simulator for a new episode by resetting all
        populations, interactions, and logger to their initial configurations.
        It ensures that each episode starts from a clean, reproducible state.

        Reset Operations
        ----------------
        1. **Population Reset**: All agent populations return to initial positions and states
        2. **Interaction Reset**: Interaction models clear any accumulated state
        3. **Logger Reset**: Logging system prepares for new episode data collection

        Performance Notes
        -----------------
        - Optimized to avoid full component reinitialization
        - Reuses existing object instances for memory efficiency
        - Faster than creating new simulator instance

        Notes
        -----
        - Called at the beginning of each RL episode
        - Does not reset renderer or environment (typically persistent)
        - Maintains component configurations while resetting states
        """

        # RESET INITIAL CONDITIONS OF THE POPULATIONS AND ENVIRONMENT AND LOGGER
        for population in self.populations:
            population.reset()

        for interaction in self.interactions:
            interaction.reset()

        self.logger.reset()

    def step(self, action):
        """
        Execute one simulation timestep with the provided RL action.

        This method advances the simulation by one timestep, applying the provided
        action to the controlled population and updating all system components
        according to the standard simulation pipeline.

        Parameters
        ----------
        action : np.ndarray
            Control action for the controlled population. Shape should match
            the controlled population's input dimension: (n_agents, input_dim).

        Simulation Pipeline
        -------------------
        1. **Action Application**: Assign action to controlled population's input (u)
        2. **Interaction Computation**: Calculate forces between all agent populations
        3. **State Integration**: Update agent positions and velocities using integrator
        4. **Force Reset**: Clear interaction forces for next timestep
        5. **Environment Update**: Update environmental conditions

        Action Space
        ------------
        The action space depends on the controlled population and application:

        - **Velocity Control**: Direct velocity commands (vx, vy)
        - **Acceleration Control**: Force/acceleration inputs (fx, fy) 

        Notes
        -----
        - Assumes populations[1] is the controlled population by convention
        - Action dimensions must match controlled population's input_dim
        - Forces are automatically reset after each timestep
        - Environment state can change dynamically during simulation
        """

        self.populations[1].u = action  # the first population is assumed to be controlled

        # Compute the interactions between the agents
        for interact in self.interactions:
            interact.target_population.f += interact.get_interaction()

        # Update the state of the agents
        self.integrator.step(self.populations)

        # Reset interaction forces
        for population in self.populations:
            population.f = np.zeros([population.N, population.input_dim])

        # Update the environment
        self.environment.update()

    def render(self):
        """
        Render the current simulation state according to configured render mode.

        This method provides flexible rendering output depending on the render_mode
        setting, supporting both human visualization and programmatic access to
        rendered frames for analysis or recording.

        Returns
        -------
        np.ndarray or None
            - If render_mode == "rgb_array": Returns numpy array of shape (height, width, 3)
            - Otherwise: Returns None, displays visualization

        Render Modes
        ------------
        **"rgb_array" Mode**:
        
        - Returns rendered frame as numpy array
        - Suitable for headless training and automated analysis
        - Efficient for batch processing and video generation
        - Compatible with gym.wrappers.RecordVideo

        **Other Mode**:
        
        - Displays visualization in real-time window
        - Interactive visualization with user controls
        - Suitable for debugging and demonstration
        - May block execution depending on renderer implementation

        Notes
        -----
        - Render mode can be changed dynamically during simulation
        - Frame dimensions depend on renderer configuration
        - Some renderers may not support all render modes
        - Rendering quality vs performance trade-offs are renderer-dependent
        """

        if self.render_mode == "rgb_array":
            return self.renderer.render()
        else:
            self.renderer.render()

    def close(self):
        # self.logger.close()
        self.renderer.close()
        """
        Clean up simulation resources and shut down rendering.

        This method properly terminates the simulation by closing rendering
        windows and releasing allocated resources. It should be called at
        the end of training or evaluation to ensure clean shutdown.

        Cleanup Operations
        ------------------
        - **Renderer Shutdown**: Close visualization windows and graphics contexts
        - **Resource Release**: Free allocated memory and system resources
        - **Logger Cleanup**: Finalize data logging and close output files

        
        Notes
        -----
        - Should be called after training completion
        - Safe to call multiple times (idempotent)
        - Essential for preventing resource leaks in long training sessions
        - May save final logs or visualizations depending on configuration
        """
