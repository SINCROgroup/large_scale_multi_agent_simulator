import gymnasium as gym
import yaml
from swarmsim.GymEnvs.shepherding_env.envs import ShepherdingEnv
from swarmsim.Controllers import ShepherdingLamaController
from swarmsim.Utils import set_global_seed
import pathlib

config_path = str(pathlib.Path(__file__).resolve().parent.parent / "Configuration" / "shepherding_gym_config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
params = config.get('Gym', {})

seed = config.get('seed')
if seed is not None:
    set_global_seed(seed=seed)

num_episodes = params['num_episodes']

env = gym.make(id='ShepherdingSwarmsim-v0', config_path=config_path, render_mode="human")
env._max_episode_steps = 10000

# Run the simulation for a certain number of steps
truncated = False
terminated = False

controller = ShepherdingLamaController(population=env.unwrapped.herders,
                                       targets=env.unwrapped.targets,
                                       environment=env.unwrapped.simulator.environment,
                                       config_path=config_path)

for episode in range(1, num_episodes + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset()

    truncated = False
    terminated = False

    while not (terminated or truncated):
        action = controller.get_action()
        # Take an episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

    print("episode: ", episode)

# Close the environment
env.close()
