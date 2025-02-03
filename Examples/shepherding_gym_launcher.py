import gymnasium as gym
import os
import yaml
import torch

from swarmsim.GymEnvs.shepherding_env.envs import ShepherdingEnv
from swarmsim.GymEnvs.shepherding_env.wrappers.multi_agent_rl import MultiAgentRL
from gymnasium.wrappers import RecordVideo

from swarmsim.Controllers import ShepherdingLamaControllerHighLevel, ShepherdingLamaController
from swarmsim.Utils.actor_critic_mappo import ActorCriticMAPPO
from swarmsim.Utils.plot_utils import get_snapshot
import pathlib

config_path = str(pathlib.Path(__file__).resolve().parent.parent / "Configuration" / "shepherding_gym_config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
params = config.get('Gym', {})

num_episodes = params['num_episodes']

env = gym.make(id='ShepherdingSwarmsim-v0', config_path=config_path, render_mode="human")
env._max_episode_steps = 10000
# env = RecordVideo(env, video_folder="videos")
env = MultiAgentRL(env, config_path=config_path)

# Run the simulation for a certain number of steps
truncated = False
terminated = False

controller = ShepherdingLamaControllerHighLevel(population=env.unwrapped.herders,
                                                targets=env.unwrapped.targets,
                                                environment=env.unwrapped.simulator.environment,
                                                config_path=config_path)
#
# controller = ShepherdingLamaController(population=env.unwrapped.herders,
#                                                 targets=env.unwrapped.targets,
#                                                 environment=env.unwrapped.simulator.environment,
#                                                 config_path=config_path)

# PPO SPECIFIC CODE: COMMENT IF NOT NEEDED
# Network configuration
# network_config = params.get('high_level_network', {'hidden_sizes': [128, 64], 'activation': 'ReLU'})
# ppo_agent = ActorCriticMAPPO(env, network_config=network_config,)


for episode in range(1, num_episodes + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=episode)

    truncated = False
    terminated = False

    while not (terminated or truncated):
        # Choose a random action (here, randomly setting velocities for herders)
        # action = env.action_space.sample()

        action = controller.get_action()

        # observation = torch.as_tensor(observation, dtype=torch.float32)
        # action = ppo_agent.get_action(observation)

        # Take an episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

    print("episode: ", episode)

# Close the environment (optional)

env.close()
