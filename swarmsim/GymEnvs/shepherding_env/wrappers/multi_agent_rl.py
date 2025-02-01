import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from gymnasium import Wrapper
import yaml

# Import your neural network
from swarmsim.Utils.actor_critic_continuous import ActorCriticContinuous
from swarmsim.Utils.ActorCritic import ActorCritic
from swarmsim.Utils.DQN_Agent import DeepQNetwork_LL, get_discrete_action

from swarmsim.GymEnvs.shepherding_env.wrappers import SingleAgentEnv


class MultiAgentRL(Wrapper):
    """
   A Gymnasium wrapper for multi-agent reinforcement learning in shepherding tasks.

   This wrapper transforms the original environment into a multi-agent RL setting
   where herders use a learned policy to chase the closest targets.

   The wrapper:
       - Defines a discrete high-level policy for selecting target indices.
       - Uses a low-level policy (e.g., PPO) to control herder movements.
       - Normalizes observations and actions based on the environment's dimensions.
       - Maintains cooperative metrics for analysis.

   The configuration parameters for the wrapper are loaded from a YAML file.

   Parameters
   ----------
   env : gym.Env
       The Gymnasium environment to wrap.
   config_path : str
       Path to the YAML configuration file containing RL parameters.

       The configuration file must contain the following keys under `Gym`:

    - `marl_wrapper.update_frequency` : int, optional
        Number of steps before updating target selection. Default is `1`.
    - `marl_wrapper.num_closest_targets` : int, optional
        Number of closest targets each herder considers. Default is `5`.
    - `marl_wrapper.num_closest_herders` : int, optional
        Number of closest herders each herder considers. Default is `1`.
    - `marl_wrapper.low_level_policy` : str, optional
        Type of low-level policy, e.g., `"PPO"` or `"DQN"`. Default is `"PPO"`.
    - `marl_wrapper.high_level_policy` : str, optional
        Type of high-level policy, e.g., `"PPO"` or `"DQN"`. Default is `"PPO"`.

    Attributes
    ----------
    update_frequency : int
        Number of steps before updating target selection.
    num_closest_targets : int
        Number of closest targets considered for selection.
    num_closest_herders : int
        Number of closest herders considered in observation space.
    low_level_policy : str
        Type of low-level policy controlling herder actions.
    obs_style : str
        Type of high-level policy selecting target indices.
    scaling_low_level : float
        Scaling factor for low-level policy inputs.
    scaling_high_level : float
        Scaling factor for high-level policy inputs.
    model : ActorCritic or None
        The neural network model for the low-level policy.
    cooperative_metric : float
        Metric tracking cooperative behavior in target selection.
    target_selection_steps : int
        Counter tracking how often targets have been selected.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    KeyError
        If required parameters are missing in the configuration file.

    Examples
    --------
    Example YAML configuration:

    ```yaml
    Gym:
      marl_wrapper:
        update_frequency: 3
        num_closest_targets: 5
        num_closest_herders: 1
        low_level_policy: "PPO"
        high_level_policy: "PPO"
    ```

    This configuration updates target selection every `3` steps,
    considers `5` closest targets per herder, and uses `"PPO"` for both policies.
    """

    def __init__(self, env, config_path):
        super().__init__(env)
        self.env = env

        # Load configuration parameters from the YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract the configuration under 'marl_wrapper' within the 'Gym' section
        gym_config = config.get('Gym', {})
        wrapper_config = gym_config.get('marl_wrapper', {})

        # Load hyperparameters from the configuration file
        self.update_frequency = wrapper_config.get('update_frequency', 1)  # Steps before target selection update
        self.num_closest_targets = wrapper_config.get('num_closest_targets', 5)  # Number of closest targets considered
        self.num_closest_herders = wrapper_config.get('num_closest_herders', 1)  # Number of closest herders considered
        self.low_level_policy = wrapper_config.get('low_level_policy', "PPO")  # Low-level movement policy
        self.obs_style = wrapper_config.get('obs_style', "PPO")  # High-level target selection policy

        # Buffers for storing closest target and herder indices
        self.closest_targets_indices = []
        self.closest_herders_indices = []

        # Define discrete action space: Each herder selects a target index from the closest targets
        self.action_space = spaces.MultiDiscrete([self.num_closest_targets] * env.unwrapped.herders.N)

        # Define the observation space structure:
        # - Own position (2)
        # - Position of closest herder (2 * self.num_closest_herders)
        # - Positions of closest targets (2 * self.num_closest_targets)
        # Total features per herder: 2 + 2*num_closest_herders + 2*num_closest_targets
        num_features = 2 + 2 * self.num_closest_herders + self.num_closest_targets * 2
        num_herders = self.env.unwrapped.herders.N

        # Compute scaling factors for normalizing observations based on environment size
        self.scaling_low_level = (
                1 / min(self.env.unwrapped.environment.dimensions)) if self.low_level_policy == "PPO" else 1
        self.scaling_high_level = (
                1 / min(self.env.unwrapped.environment.dimensions)) if self.obs_style == "PPO" else 1

        # Define the observation space with infinite bounds (to be clipped if needed)
        obs_low = np.full((num_herders, num_features), -np.inf, dtype=np.float32)
        obs_high = np.full((num_herders, num_features), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize the neural network model for low-level control if using PPO
        self.control_inputs = None
        self.num_acts = gym_config.get("num_acts", None)
        if self.low_level_policy == "PPO":
            # network_config = wrapper_config.get('network', {'hidden_sizes': [128, 64], 'activation': 'ReLU'})
            # env_ppo = SingleAgentEnv(env, config_path)
            # self.model = ActorCriticContinuous(env_ppo, network_config)
            self.model = ActorCritic()
        else:
            action_max = gym_config["action_bound"]
            self.model = DeepQNetwork_LL(lr=0, n_actions=self.num_acts, name='LLC_DQN', input_dims=[6], chkpt_dir='./models/')
            self.model.load_checkpoint()
            control_input = np.linspace(-action_max, action_max, self.num_acts)
            self.control_inputs = np.vstack((control_input, control_input))

        # Create an observation buffer to store a batch of observations for training
        self.obs_batch = np.zeros(shape=((self.update_frequency,) + env.unwrapped.observation_space.shape))

        # Initialize tracking metrics
        self.cooperative_metric = 0  # Tracks cooperation among herders
        self.target_selection_steps = 0  # Counts how often targets have been selected

    def step(self, action):
        """
        Processes the action and steps through the environment.

        This function executes a high-level decision-making step where each herder
        selects a target to chase. The selected target indices are then used as input
        for a low-level policy (e.g., PPO) to compute movement actions.

        The function runs for `update_frequency` steps, accumulating rewards and
        tracking episode termination conditions.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (num_herders,), where each element is an integer
            representing the index (0 to num_closest_targets-1) of the closest
            target that the corresponding herder should chase.

        Returns
        -------
        obs : np.ndarray
            Processed observations after executing the selected actions.
        reward : float
            Cumulative reward obtained over `update_frequency` steps.
        terminated : bool
            Whether the episode has reached a termination condition.
        truncated : bool
            Whether the episode was truncated due to a time limit.
        info : dict
            Additional information including cooperative metrics.
        """

        num_herders = self.env.unwrapped.herders.N

        # Map selected target indices to actual targets in the environment
        target_indices_to_chase = self.closest_targets_indices[np.arange(num_herders), action]

        # Increment counter tracking how often targets have been selected
        self.target_selection_steps += 1

        # Initialize cumulative reward across update steps
        cumulative_reward = 0

        # Reset the observation batch storage
        self.obs_batch = np.zeros(shape=((self.update_frequency,) + self.env.unwrapped.observation_space.shape))

        # Execute `update_frequency` steps in the environment
        for step in range(self.update_frequency):

            # Convert target indices to an array (ensuring correct format)
            target_indices_to_chase = np.array(target_indices_to_chase)

            # Prepare batched inputs for the neural network based on the low-level policy type
            if self.low_level_policy == "PPO":
                # Compute relative positions for PPO-based movement policy
                target_positions = self.env.unwrapped.targets.x[target_indices_to_chase,
                                   :2] - self.env.unwrapped.environment.goal_pos
                relative_positions = self.env.unwrapped.targets.x[target_indices_to_chase,
                                     :2] - self.env.unwrapped.herders.x[:, :2]

                # Stack positions into a single batch input and normalize
                batch_inputs = np.hstack((relative_positions, target_positions)).astype(
                    np.float32) * self.scaling_low_level

            elif self.low_level_policy == "DQN":
                # Compute positions for DQN-based movement policy
                target_positions = self.env.unwrapped.targets.x[target_indices_to_chase,
                                   :2] - self.env.unwrapped.environment.goal_pos
                target_velocities = self.env.unwrapped.targets.x[target_indices_to_chase, 2:4]
                herder_positions = self.env.unwrapped.herders.x[:, :2] - self.env.unwrapped.environment.goal_pos

                # Stack positions into a single batch input
                batch_inputs = np.hstack((target_positions, target_velocities, herder_positions)).astype(np.float32)

            # Convert batch input into a PyTorch tensor
            batch_tensor = torch.tensor(batch_inputs)

            # Compute actions using the neural network (only for PPO, DQN logic not implemented)
            if self.low_level_policy == "PPO":
                # batched_herder_actions = self.model.get_action(batch_tensor).cpu().numpy()
                batched_herder_actions = self.model.get_action_mean(batch_tensor).cpu().numpy()
            elif self.low_level_policy == "DQN":
                batched_herder_actions_idx = self.model.get_action(batch_tensor)
                batched_herder_actions = get_discrete_action(batched_herder_actions_idx, self.control_inputs)
                batched_herder_actions = batched_herder_actions.transpose()

            # Step the environment with the computed herder actions
            obs_unw, reward_unw, terminated, truncated, info = self.env.step(batched_herder_actions)

            # Accumulate the reward
            cumulative_reward += reward_unw

            # Store the new observations in the batch buffer
            self.obs_batch[step] = obs_unw

            # If the episode has ended, break out of the loop
            if terminated or truncated:
                break

        # Compute additional info metrics (e.g., cooperation)
        info = self._get_info(info, target_indices_to_chase)

        # Process observations before returning
        obs = self._process_observations()
        reward = cumulative_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and initializes relevant tracking metrics.

        This function resets the environment and processes the initial observation
        to align with the multi-agent RL setup. Additionally, it resets tracking
        variables such as the cooperative metric and target selection counter.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters that may be passed to the environment's reset function.

        Returns
        -------
        processed_obs : np.ndarray
            Processed initial observations formatted for the RL policy.
        info : dict
            Additional environment-specific information.
        """

        # Reset the environment and retrieve the initial observation and info
        obs, info = self.env.reset(**kwargs)

        # Process the raw observation into the structured format required for RL
        processed_obs = self._process_observations()

        # Reset cooperative metric tracking how well herders distribute target selection
        self.cooperative_metric = 0

        # Reset the counter tracking how many times targets have been selected
        self.target_selection_steps = 0

        return processed_obs, info

    def _process_observations(self):
        """
        Processes the observations to structure them for the RL policy.

        This function computes:
        - Each herder's own position.
        - The positions of the `num_closest_herders` nearest herders.
        - The positions of the `num_closest_targets` nearest targets.

        The processed observations are normalized based on the environment dimensions.

        Returns
        -------
        processed_obs : np.ndarray
            Processed observation array of shape (num_herders, observation_features).
        """

        # Retrieve the positions of all herders and targets
        herder_positions = self.env.unwrapped.herders.x[:, :2]  # Shape: (num_herders, 2)
        target_positions = self.env.unwrapped.targets.x[:, :2]  # Shape: (num_targets, 2)
        num_herders = self.env.unwrapped.herders.N

        # Normalize positions using the environment scaling factor
        herder_positions_normalized = herder_positions * self.scaling_high_level
        target_positions_normalized = target_positions * self.scaling_high_level

        # Compute pairwise distances between all herders (excluding self)
        distances_between_herders = np.linalg.norm(
            herder_positions_normalized[:, np.newaxis, :] - herder_positions_normalized[np.newaxis, :, :],
            axis=2
        )

        # Set diagonal to infinity to exclude self from closest-herder search
        np.fill_diagonal(distances_between_herders, np.inf)

        # Get indices of the closest herders for each herder
        closest_herders_indices = np.argsort(distances_between_herders, axis=1)[:, :self.num_closest_herders]

        # Store sorted closest herders for consistency
        self.closest_herders_indices = np.sort(closest_herders_indices, axis=1)

        # Retrieve positions of the closest herders
        closest_herders = herder_positions_normalized[self.closest_herders_indices]

        # Flatten the closest herder positions into a single feature vector
        closest_herders_flat = closest_herders.reshape(num_herders, -1)  # Shape: (num_herders, num_closest_herders * 2)

        # Compute distances from herders to all targets
        distances_to_targets = np.linalg.norm(
            herder_positions_normalized[:, np.newaxis, :] - target_positions_normalized[np.newaxis, :, :],
            axis=2
        )

        # Get indices of the closest targets for each herder
        closest_target_indices = np.argsort(distances_to_targets, axis=1)[:, :self.num_closest_targets]

        # Store sorted closest target indices for consistency
        self.closest_targets_indices = np.sort(closest_target_indices, axis=1)

        # Retrieve positions of the closest targets
        closest_targets = target_positions_normalized[self.closest_targets_indices]

        # Flatten the closest target positions into a single feature vector
        closest_targets_flat = closest_targets.reshape(num_herders, -1)  # Shape: (num_herders, num_closest_targets * 2)

        # Concatenate herder's own position, closest herders, and closest targets
        if self.obs_style == "PPO":
            processed_obs = np.concatenate(
                [herder_positions_normalized, closest_herders_flat, closest_targets_flat],
                axis=1
            )
        elif self.obs_style == "DQN":
            processed_obs = np.concatenate(
                [closest_targets_flat, herder_positions_normalized, closest_herders_flat],
                axis=1
            )

        return processed_obs

    def _get_info(self, info, target_indices):
        """
        Updates the info dictionary with additional metrics.

        This function computes and adds the cooperative metric to track how evenly
        targets are distributed among herders. It also stores the observation batch.

        Parameters
        ----------
        info : dict
            The original info dictionary returned by the environment.
        target_indices : np.ndarray
            Array of shape (num_herders,) containing the indices of the targets
            assigned to each herder.

        Returns
        -------
        dict
            Updated info dictionary including:
            - 'cooperative_metric': A measure of how evenly herders distribute target selection.
            - 'obs_batch': A batch of past observations.
        """

        # Compute cooperative behavior metric and store it in the info dictionary
        info['cooperative_metric'] = self._compute_cooperative_metric(target_indices)

        # Store the batch of observations for analysis/debugging
        info['obs_batch'] = self.obs_batch

        return info

    def _compute_cooperative_metric(self, target_indices):
        """
        Computes a metric to evaluate cooperative behavior among herders.

        This metric measures how evenly herders select different targets:
        - A higher value indicates more diverse target selection (better cooperation).
        - A lower value suggests that herders are converging on the same targets.

        The metric is computed as the normalized count of unique targets selected by herders.

        Parameters
        ----------
        target_indices : np.ndarray
            Array of shape (num_herders,) representing the targets assigned to each herder.

        Returns
        -------
        float
            The cooperative metric, a normalized value in the range [0, 1].
        """

        # Count the number of unique targets selected
        num_different_targets = len(np.unique(target_indices, axis=0))

        # Normalize the diversity of target selection
        if self.env.unwrapped.herders.N > 1:
            ratio_different_targets = (num_different_targets - 1) / (self.env.unwrapped.herders.N - 1)
        else:
            ratio_different_targets = 0

        # Update cooperative metric using an exponential moving average
        self.cooperative_metric = (self.cooperative_metric +
                                   (ratio_different_targets - self.cooperative_metric) / self.target_selection_steps)

        return self.cooperative_metric
