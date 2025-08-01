import numpy as np
from swarmsim.Environments import EmptyEnvironment


class ShepherdingEnvironment(EmptyEnvironment):
    """
    A shepherding environment where the goal position moves along a linear path
    from (0, 0) to a final position specified in the configuration file.

    This environment extends `EmptyEnvironment` and updates the goal position
    at each time step, allowing dynamic goal movement for shepherding tasks.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing environment parameters.

    Attributes
    ----------
    goal_radius : float
        Radius of the goal region where agents need to reach.
    goal_pos : np.ndarray
        Current position of the goal.
    final_goal_pos : np.ndarray
        The final coordinates of the goal, loaded from the configuration.
    num_steps : int
        The total number of steps over which the goal moves.
    step_count : int
        Counter to track the current time step.
    start_step: int
        The step at which the goal region starts moving.
    direction : np.ndarray
        The unit vector representing the direction of movement from `(0,0)` to the final position.

    Config requirements
    -------------------
    goal_radius : float, optional
        The radius of the goal region. Default is ``5``.
    goal_pos : tuple of (float, float), optional
        The initial position of the goal. Default is ``(0, 0)``.
    final_goal_pos : tuple of (float, float), required
        The final position of the goal. This must be specified in the configuration.
    num_steps : int, optional
        The total number of time steps over which the goal moves. Default is ``2000``.
    start_step: int, optional
        The step at which the goal region starts moving. Default is ``0``

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required environment parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        environment:
            goal_radius: 5
            goal_pos: [0, 0]
            final_goal_pos: [20, -20]
            num_steps: 2000

    This will move the goal position from ``(0, 0)`` to ``(20, -20)`` over ``2000`` steps.
    """

    def __init__(self, config_path):
        """
        Initializes the ShepherdingEnvironment with the configuration parameters from a YAML file.

        The goal position moves from `(0, 0)` towards a final position specified in
        the configuration file over a number of time steps.

        Parameters
        ----------
        config_path : str
            The path to the YAML configuration file.
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
        Retrieves environment information including the goal region radius and center.

        Returns
        -------
        dict
            A dictionary containing:
            - ``Goal region radius``: The radius of the goal region.
            - ``Goal region center``: The current position of the goal.
        """
        return {'Goal region radius': self.goal_radius, 'Goal region center': self.goal_pos}

    def update(self):
        """
        Updates the goal position at every time step.

        The goal moves incrementally in a straight line from `(0,0)` to the final
        position specified in the configuration file. Once it reaches the final
        position, it stops moving.

        The movement follows the equation:

            goal_pos = goal_pos + direction

        where ``direction`` is the step-wise movement vector.

        Notes
        -----
        - The goal starts moving **after 1000 steps**.
        - The goal stops moving once it reaches the final position.
        """
        # Ensure the goal stops moving once it reaches the final position
        self.step_count += 1
        if (self.step_count > self.start_step) and (self.step_count < self.start_step + self.num_steps):
            self.goal_pos += self.direction
