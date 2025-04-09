import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
import yaml

from swarmsim.Utils import compute_distances


class SingleAgentEnv(Wrapper):
    """
    A Gymnasium Wrapper that modifies the base shepherding environment for a single-herder
    single-target setting.

    This wrapper modifies the reward function based on:
        - Distance between targets and herders.
        - Distance between targets and the goal.
        - Penalty for herders entering the goal region.
        - Reward for targets reaching the goal.

    The observation consists of:
        - Relative position of targets with respect to herders.
        - Relative position of targets with respect to the goal.
        - Normalization by the smallest dimension of the environment.

    The reward function coefficients are loaded from a configuration file.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    config_path : str
        Path to the YAML configuration file containing reward coefficients.

        The configuration file must contain the following keys under the `Gym` namespace:

        - `single_agent_reward.obs_style` : str, optional
            Observation style, it can be 'DQN' or 'PPO'. Default is `PPO`.
        - `single_agent_reward.k_1` : float, optional
            Weight for the penalty based on target-herder distance. Default is `1.0`.
        - `single_agent_reward.k_2` : float, optional
            Weight for the penalty based on target-goal distance. Default is `1.0`.
        - `single_agent_reward.k_3` : float, optional
            Reward for targets reaching the goal. Default is `1.0`.
        - `single_agent_reward.k_4` : float, optional
            Penalty for herders entering the goal region. Default is `1.0`.

    Attributes
    ----------
    obs_style : str
        Observations style compliant to PPO or DQN policies
    k_1 : float
        Weight for the penalty based on target-herder distance.
    k_2 : float
        Weight for the penalty based on target-goal distance.
    k_3 : float
        Reward for targets reaching the goal.
    k_4 : float
        Penalty for herders entering the goal region.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required reward parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    ```yaml
    Gym:
      single_agent_reward:
        k_1: 0.5
        k_2: 1.2
        k_3: 2.0
        k_4: 0.8
    ```
    """

    def __init__(self, env: gym.Env, config_path: str) -> None:
        super().__init__(env)
        self.env = env

        # Load reward configuration from YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract the 'single_agent_reward' configuration from the 'Gym' section
        gym_config = config.get('Gym', {})
        wrapper_config = gym_config.get('single_agent_env', {})

        self.obs_style = gym_config.get('obs_style', 'PPO')

        # Reward function coefficients
        self.k_1 = wrapper_config.get('k_1', 1)  # Target-herder distance penalty
        self.k_2 = wrapper_config.get('k_2', 1)  # Target-goal distance penalty
        self.k_3 = wrapper_config.get('k_3', 1)  # Reward for reaching the goal
        self.k_4 = wrapper_config.get('k_4', 1)  # Penalty for herders in the goal region

    def _get_obs(self) -> np.ndarray:
        """
        Computes the observation for the agent.

        The observation for PPO consists of:
            - Relative position of targets with respect to herders.
            - Relative position of targets with respect to the goal.
            - Normalization by the smallest dimension of the environment.

        The observation for DQN consists of:
            - Relative position of herders with respect to the goal.
            - Relative position of targets with respect to the goal.

        Returns
        -------
        np.ndarray
            Normalized observation vector.
        """
        if self.obs_style == 'PPO':
            target_position = self.env.unwrapped.targets.x[:, :2] - self.env.unwrapped.environment.goal_pos
            relative_position = self.env.unwrapped.targets.x[:, :2] - self.env.unwrapped.herders.x[:, :2]

            # Concatenate and normalize by the smallest dimension of the environment
            obs = np.concatenate((relative_position, target_position)).astype(np.float32)
            obs = obs / min(self.env.unwrapped.environment.dimensions)
        else:
            target_position = self.env.unwrapped.targets.x[:, :2] - self.env.unwrapped.environment.goal_pos
            herder_position = self.env.unwrapped.herders.x[:, :2] - self.env.unwrapped.environment.goal_pos

            # Concatenate and normalize by the smallest dimension of the environment
            obs = np.concatenate((target_position, herder_position)).astype(np.float32)

        return obs

    def _get_reward(self) -> float:
        """
        Computes the reward function for the agent.

        The reward is computed based on:
            - Distance between targets and herders (penalty).
            - Distance between targets and the goal (penalty).
            - Additional penalty if herders enter the goal radius.
            - Additional reward if targets reach the goal.

        Returns
        -------
        float
            Computed reward value.
        """
        # Compute distances between key entities
        target_distance, _ = compute_distances(self.env.unwrapped.targets.x[:, :2],
                                               self.env.unwrapped.environment.goal_pos)
        herder_distance, _ = compute_distances(self.env.unwrapped.herders.x[:, :2],
                                               self.env.unwrapped.environment.goal_pos)
        ht_distance, _ = compute_distances(self.env.unwrapped.targets.x[:, :2],
                                           self.env.unwrapped.herders.x[:, :2])

        # Base reward computation
        reward = - self.k_1 * ht_distance - self.k_2 * target_distance

        # Apply additional penalties and rewards
        if herder_distance < self.env.unwrapped.environment.goal_radius:
            reward -= self.k_4  # Penalty for herders inside the goal region
        if target_distance < self.env.unwrapped.environment.goal_radius:
            reward += self.k_3  # Reward for targets reaching the goal

        return reward
