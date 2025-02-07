import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, spaces


class FlattenAction(ActionWrapper):
    """
    A Gymnasium ActionWrapper that flattens a multi-agent action space.

    This wrapper transforms the original action space, which has a shape of (num_agents, 2),
    into a 1D action space of shape (2 * num_agents,). When an action is passed to the environment,
    it reshapes it back into the original (num_agents, 2) format.

    Parameters
    ----------
    env : gym.Env
        The environment to be wrapped

    Attributes
    ----------
    action_space : spaces.Box
        The modified action space with shape (2 * num_agents)

    Raises
    ------
    AssertionError :
        If the original action space does not have shape (num_agents, 2).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # Get the original action space shape
        original_action_shape = self.env.action_space.shape

        # Ensure the action space is in the expected format (num_agents, 2)
        assert len(original_action_shape) == 2 and original_action_shape[1] == 2, \
            "Original action space shape must be (num_agents, 2)"

        num_agents = original_action_shape[0]

        # Define the new action space with shape (2 * num_agents,)
        low = self.env.action_space.low.flatten().reshape(2 * num_agents)
        high = self.env.action_space.high.flatten().reshape(2 * num_agents)

        # Update the action space to match the new flattened shape
        self.action_space = spaces.Box(low=low, high=high, dtype=self.env.action_space.dtype)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Reshapes the flattened action array back into the original multi-agent format.

        Parameters
        -------
            action : np.ndarray     A 1D action array of shape (2 * num_agents,).

        Returns
        -------
            np.ndarray: A reshaped array of shape (num_agents, 2).
        """
        num_agents = int(self.env.action_space.shape[0] / 2)  # Compute number of agents
        return np.reshape(action, (num_agents, 2))  # Convert to (num_agents, 2) format
