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

    def get_info(self):
        return {'Goal region radius': self.goal_radius, 'Goal region center': self.goal_pos}

    def update(self):
        pass
