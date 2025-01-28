import gymnasium as gym
import numpy as np
from gymnasium import spaces
import yaml

from typing import Optional

from swarmsim.Populations import DampedDoubleIntegrators
from swarmsim.Populations import SimpleIntegrators
from swarmsim.Interactions import PowerLawRepulsion
from swarmsim.Integrators import EulerMaruyamaIntegrator
from swarmsim.Renderers import ShepherdingRenderer
from swarmsim.Simulators import GymSimulator
from swarmsim.Environments import ShepherdingEnvironment
from swarmsim.Loggers import ShepherdingGymLogger
from swarmsim.Utils import get_target_distance, xi_shepherding
from swarmsim.Utils.plot_utils import get_snapshot


class ShepherdingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 200}

    def __init__(self, config_path, render_mode: Optional[str] = None):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        params = config.get('Gym', {})

        integrator = EulerMaruyamaIntegrator(config_path)

        environment = ShepherdingEnvironment(config_path)

        targets = DampedDoubleIntegrators(config_path)
        herders = SimpleIntegrators(config_path)
        populations = [targets, herders]

        self.herders = herders
        self.targets = targets
        self.environment = environment

        repulsion_ht = PowerLawRepulsion(targets, herders, config_path, "RepulsionHerderTarget")
        repulsion_tt = PowerLawRepulsion(targets, targets, config_path, "RepulsionTargetTarget")
        repulsion_hh = PowerLawRepulsion(herders, herders, config_path, "RepulsionHerderHerder")
        interactions = [repulsion_ht, repulsion_tt]

        renderer = ShepherdingRenderer(populations, environment, config_path)
        logger = ShepherdingGymLogger(populations, environment, config_path)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.simulator = GymSimulator(populations=populations,
                                      interactions=interactions,
                                      environment=environment,
                                      integrator=integrator,
                                      logger=logger,
                                      renderer=renderer,
                                      config_path=config_path)

        obs_shape = (self.herders.N + self.targets.N, 2)
        obs_min = -np.inf
        obs_max = np.inf

        action_shape = herders.u.shape
        action_max = params["action_bound"]
        action_min = -action_max

        self.observation_space = spaces.Box(low=obs_min, high=obs_max, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=action_min, high=action_max, shape=action_shape, dtype=np.float32)

        self.reward_gain = params["reward_gain"]
        self.cum_rew = None  # Cumulative reward in the current episode
        self.cum_rews = []  # Array of cumulative rewards
        self.episode_number = 0
        self.xi = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.simulator.reset()
        self.episode_number += 1
        observation = self._get_obs()
        self.cum_rew = 0
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            # get_snapshot(self.simulator.renderer.render(),'x0.png')

        return observation, info

    def step(self, action):
        self.simulator.step(action)

        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()

        self.cum_rew += reward
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            self._render_frame()

    def _get_obs(self):
        targets_position = self.targets.x[:, :2]
        herders_position = self.herders.x
        obs = np.concatenate((herders_position, targets_position)).astype(np.float32)
        return obs

    def _get_info(self):
        if self.episode_number > 1:
            self.cum_rews.append(self.cum_rew)
        self.xi = xi_shepherding(self.targets, self.environment)
        info = {"num_herders": self.herders.N,
                "num_targets": self.targets.N,
                "episode_number": self.episode_number,
                "cumulative_reward": self.cum_rew,
                "settling_time": None,
                "fraction_captured_targets": self.xi}
        self.simulator.logger.log(info)
        return info

    def _render_frame(self):
        return self.simulator.render()

    def close(self):
        self.cum_rews.append(self.cum_rew)
        if self.simulator.logger.activate:
            data = {'cum_rews': np.asarray(self.cum_rews, dtype=np.float64)}
            self.simulator.logger.save_data(data)
        self.simulator.close()

    def _get_reward(self):
        target_radii = get_target_distance(self.targets, self.environment)
        distance_from_goal = target_radii - self.environment.goal_radius
        reward_vector = np.where(distance_from_goal < 0, self.reward_gain, distance_from_goal)
        reward = -np.sum(reward_vector)
        return reward
