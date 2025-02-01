import numpy as np
from swarmsim.Environments import EmptyEnvironment


class ShepherdingEnvironment(EmptyEnvironment):
    """
    A shepherding environment where the goal position moves along a linear path
    from (0, 0) to a final position specified in the configuration file.

    The environment extends the EmptyEnvironment and updates the goal position
    at each time step.

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
    direction : np.ndarray
        The unit vector representing the direction of movement from (0,0) to the final position.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing environment parameters.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required environment parameters are missing in the configuration file.
    """

    def __init__(self, config_path):
        """
        Initializes the ShepherdingEnvironment with the configuration parameters from a YAML file.

        The goal position moves from (0, 0) towards a final position specified in
        the configuration file over a number of time steps.

        Parameters
        ----------
        config_path : str
            The path to the YAML configuration file.
        """
        super().__init__(config_path)

        # Load goal parameters from the configuration file
        self.goal_radius = self.params.get('goal_radius', 5)
        self.goal_pos = np.array(self.params.get('goal_pos', (0, 0)), dtype=np.float32)
        self.final_goal_pos = np.array(self.params.get('final_goal_pos', (0, 0)), dtype=np.float32)
        self.num_steps = self.params.get('num_steps', 2000)  # Total steps over which goal moves
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
            - 'Goal region radius': The radius of the goal region.
            - 'Goal region center': The current position of the goal.
        """
        return {'Goal region radius': self.goal_radius, 'Goal region center': self.goal_pos}

    def update(self):
        """
        Updates the goal position at every time step.

        The goal moves incrementally in a straight line from (0,0) to the final
        position specified in the configuration file. Once it reaches the final
        position, it stops moving.

        The movement follows the equation:
        `goal_pos = goal_pos + direction`
        where `direction` is the step-wise movement vector.

        """
        # Ensure the goal stops moving once it reaches the final position
        self.step_count += 1
        if (self.step_count > 1000) and (self.step_count < 1000 + self.num_steps):
            self.goal_pos += self.direction

