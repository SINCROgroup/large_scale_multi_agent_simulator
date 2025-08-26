import numpy as np
from swarmsim.Environments import EmptyEnvironment


class ShepherdingEnvironment(EmptyEnvironment):
    """
    Dynamic environment for shepherding tasks with moving goal positions.

    This environment extends EmptyEnvironment to support shepherding scenarios where
    the goal position moves along a predefined trajectory. The goal starts at
    an initial position and moves toward a final destination over a specified number
    of simulation steps, creating dynamic targets for shepherding controllers.

    The environment is designed for multi-agent shepherding tasks where herder agents
    must guide target agents to a moving goal region. The linear goal movement creates
    realistic scenarios where the target location changes predictably over time.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing environment parameters.

    Attributes
    ----------
    goal_radius : float
        Radius of the goal region where agents are considered to have reached the target.
    goal_pos : np.ndarray
        Current 2D position of the goal center.
    final_goal_pos : np.ndarray
        Final 2D coordinates where the goal will stop moving.
    num_steps : int
        Total number of simulation steps over which the goal moves.
    step_count : int
        Current simulation step counter for tracking goal movement progress.
    start_step : int
        Simulation step at which goal movement begins (allows for initial stationary period).
    direction : np.ndarray
        Unit direction vector from initial to final goal position.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the environment section:

    goal_radius : float, optional
        Radius of the goal region in environment units. Default is ``5.0``.
    goal_pos : list of float, optional
        Initial 2D coordinates [x, y] of the goal position. Default is ``[0, 0]``.
    final_goal_pos : list of float
        Final 2D coordinates [x, y] where the goal movement terminates. Required parameter.
    num_steps : int, optional
        Number of simulation steps for the complete goal trajectory. Default is ``2000``.
    start_step : int, optional
        Simulation step when goal movement begins. Default is ``0``.
    dimensions : list of int, optional
        Environment dimensions [width, height]. Inherited from EmptyEnvironment.

    Notes
    -----
    The goal movement follows a linear trajectory:

        goal_pos(t) = initial_pos + (t - start_step) / num_steps * (final_pos - initial_pos)

    where t is the current step count. The goal remains stationary before start_step
    and after reaching the final position.

    Key features:
    - **Linear Movement**: Goal moves in straight line from initial to final position
    - **Configurable Timing**: Movement start time and duration are adjustable
    - **Goal Region**: Circular region around goal position with specified radius
    - **Status Tracking**: Provides information about goal state and agent proximity

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        environment:
            dimensions: [100, 100]
            goal_radius: 8.0
            goal_pos: [0, 0]
            final_goal_pos: [30, -20]
            num_steps: 1500
            start_step: 100

    Advanced usage with shepherding simulation:

    .. code-block:: python

        from swarmsim.Environments import ShepherdingEnvironment
        from swarmsim.Populations import BrownianMotion
        from swarmsim.Controllers import ShepherdingLamaController

        # Create environment
        env = ShepherdingEnvironment('config.yaml')
        
        # Create populations
        targets = BrownianMotion('config.yaml', 'Targets')
        herders = BrownianMotion('config.yaml', 'Herders')
        
        # Create shepherding controller
        controller = ShepherdingLamaController(
            population=herders,
            targets=targets,
            environment=env,
            config_path='config.yaml'
        )
        
        # Environment automatically updates goal position each step
        env.update()  # Advances goal along trajectory
        
        # Check if targets are near current goal
        info = env.get_info()
        print(f"Goal position: {env.goal_pos}")
        print(f"Targets in goal: {info['targets_in_goal']}")

    Applications include:
    - Dynamic shepherding scenarios
    - Moving target tracking problems
    - Adaptive goal-seeking behaviors
    - Multi-phase shepherding tasks
    """

    def __init__(self, config_path):
        """
        Initialize the ShepherdingEnvironment with dynamic goal movement.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing environment parameters.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found.
        KeyError
            If required parameters are missing from the configuration.

        Notes
        -----
        The initialization computes the step-wise movement vector based on the
        linear trajectory from initial to final goal position over the specified
        number of steps.
        """
        super().__init__(config_path)

        # Load goal parameters from the configuration file
        self.goal_radius = self.params.get('goal_radius', 5)
        self.buffer = self.params.get('buffer_size', 0)
        self.goal_pos = np.array(self.params.get('goal_pos', (0, 0)), dtype=np.float32)
        self.final_goal_pos = np.array(self.params.get('final_goal_pos', (0, 0)), dtype=np.float32)
        self.num_steps = self.params.get('num_steps', 2000)  # Total steps over which goal moves
        self.start_step = self.params.get('start_step', 0)
        self.step_count = 0  # Counter for tracking time steps

        # Compute direction vector from (0,0) to final_goal_pos
        displacement = self.final_goal_pos - self.goal_pos
        self.direction = displacement / self.num_steps  # Step-wise movement vector

    def get_info(self):
        """
        Retrieve current environment state information for logging and monitoring.

        Returns
        -------
        dict
            Dictionary containing environment status with keys:
            
            - 'Goal region radius': float
                Current radius of the goal region
            - 'Goal region center': np.ndarray
                Current 2D coordinates of the goal center

        Notes
        -----
        This information can be used by:
        - Loggers to track goal movement over time
        - Controllers to access current goal state
        - Renderers to visualize the goal region
        - Analysis tools to compute performance metrics
        """
        return {'Goal region radius': self.goal_radius, 'Goal region center': self.goal_pos}

    def update(self):
        """
        Advance the goal position along its trajectory.

        This method is called at each simulation timestep to update the goal position.
        The goal moves linearly from its initial position toward the final position
        over the specified number of steps, starting at the configured start step.

        Notes
        -----
        The goal movement algorithm:

        1. **Step Counting**: Increment internal step counter
        2. **Movement Window**: Check if current step is within movement period
        3. **Position Update**: Add direction vector to current goal position
        4. **Boundary Conditions**: Goal stops at final position

        Movement occurs only when:
        - Current step > start_step (allows initial stationary period)
        - Current step < start_step + num_steps (prevents overshoot)

        The direction vector is pre-computed during initialization as:
            direction = (final_pos - initial_pos) / num_steps

        After num_steps, the goal remains stationary at the final position.
        """
        # Ensure the goal stops moving once it reaches the final position
        self.step_count += 1
        if (self.step_count > self.start_step) and (self.step_count < self.start_step + self.num_steps):
            self.goal_pos += self.direction
