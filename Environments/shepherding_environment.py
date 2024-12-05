import numpy as np
from Environments.empty_environment import EmptyEnvironment


class ShepherdingEnvironment(EmptyEnvironment):
    """
    An environment with no forces acting on the agents. The environment is a square with dimensions 50x50.
    """

    def __init__(self, config_path):
        """
        Initializes the EmptyEnvironment with the configuration parameters from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        super().__init__(config_path)
        self.goal_radius = self.params.get('goal_radius', 5)
        self.goal_pos = np.array(self.params.get('goal_pos', (0, 0)))

    def get_forces(self, agents):
        """
        Computes the forces exerted by the environment on the agents.
        Since this is an empty environment, no forces are exerted.

        Args:
            agents: A list of agents for which the environmental forces are being computed.

        Returns:
            np.ndarray: An array of zeros representing no force on each agent.
        """
        return np.zeros((agents.x.shape[0], 2))  # No forces, so return zero vectors for each agent
