from gymnasium import Wrapper
import yaml
from swarmsim.Utils.shepherding_utils import get_done_shepherding


class TerminateWhenSuccessful(Wrapper):
    """
    A Gymnasium wrapper that terminates the episode when the shepherding task
    is deemed successful for a consecutive number of steps.

    The wrapper monitors whether all target agents are within the goal region
    and maintains a success buffer. If the environment remains in a successful
    state for a specified number of consecutive steps, the episode is terminated.

    The termination condition is based on the `get_done_shepherding` function,
    which checks if all targets are sufficiently close to the goal.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    config_path : str
        Path to the YAML configuration file containing termination parameters.

        The configuration file must contain the following key under the `Gym` namespace:

        - `terminate_when_successful.num_steps` : int, optional
            Number of consecutive steps in which the success condition must hold
            before terminating the episode. Default is `1`.

    Attributes
    ----------
    success_buffer : int
        Counter tracking consecutive successful steps.
    num_steps : int
        Required number of consecutive successful steps before termination.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required termination parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    ```yaml
    Gym:
      terminate_when_successful:
        num_steps: 5
    ```

    This will terminate the episode if the shepherding task remains successful for `5` consecutive steps.
    """

    def __init__(self, env, config_path):
        super().__init__(env)
        self.env = env

        # Load termination configuration from YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract the 'terminate_when_successful' configuration from the 'Gym' section
        gym_config = config.get('Gym', {})
        wrapper_config = gym_config.get('terminate_when_successful', {})

        # Initialize success tracking variables
        self.success_buffer = 0
        self.num_steps = wrapper_config.get('num_steps', 1)  # Default is 1

    def step(self, action) -> tuple:
        """
        Steps through the environment, updating the termination condition.

        If the `get_done_shepherding` function determines that the shepherding
        task is successful, the success buffer increases. If success persists
        for `num_steps` consecutive steps, the episode is terminated.

        Parameters
        ----------
        action : Any
            The action taken by the agent.

        Returns
        -------
        tuple
            (obs, reward, terminated, truncated, info), where:
            - `obs` : observation after taking the action.
            - `reward` : reward received from the environment.
            - `terminated` : bool, `True` if termination condition is met.
            - `truncated` : bool, whether the episode was truncated due to time limit.
            - `info` : dict, additional information.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if the shepherding task is successful
        if get_done_shepherding(self.env.unwrapped.targets, self.env.unwrapped.environment, threshold=1):
            self.success_buffer += 1
        else:
            self.success_buffer = 0

        # Terminate the episode if success persists for `num_steps`
        if self.success_buffer >= self.num_steps:
            terminated = True

        return obs, reward, terminated, truncated, info
