import gymnasium as gym
import numpy as np
from gymnasium import spaces
import yaml

from typing import Optional

from swarmsim.Populations import BrownianMotion
from swarmsim.Populations import SimpleIntegrators
from swarmsim.Interactions import HarmonicRepulsion
from swarmsim.Integrators import EulerMaruyamaIntegrator
from swarmsim.Renderers import ShepherdingRenderer
from swarmsim.Simulators import GymSimulator
from swarmsim.Environments import ShepherdingEnvironment
from swarmsim.Loggers import ShepherdingLogger


class ShepherdingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 200}

    def __init__(self, config_path, render_mode: Optional[str] = None):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        params = config.get('Gym', {})

        integrator = EulerMaruyamaIntegrator(config_path)

        environment = ShepherdingEnvironment(config_path)

        targets = BrownianMotion(config_path)
        herders = SimpleIntegrators(config_path)
        populations = [targets, herders]

        self.herders = herders
        self.targets = targets
        self.environment = environment

        repulsion_ht = HarmonicRepulsion(targets, herders, config_path)
        interactions = [repulsion_ht]

        renderer = ShepherdingRenderer(populations, environment, config_path)
        logger = ShepherdingLogger(populations, environment, config_path)

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.simulator.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.simulator.step(action)

        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()

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
        targets_position = self.targets.x
        herders_position = self.herders.x
        obs = np.concatenate((herders_position, targets_position)).astype(np.float32)
        return obs

    def _get_info(self):
        return {"num_herders": self.herders.N,
                "num_targets": self.targets.N,
                "settling_time": None,
                "fraction_captured_targets": None}

    def _render_frame(self):
        self.simulator.render()

    def close(self):
        self.simulator.close()

    def _get_reward(self):
        target_radii = np.linalg.norm(self.targets.x, axis=1)
        distance_from_goal = target_radii - self.environment.goal_radius
        reward_vector = np.where(distance_from_goal < 0, self.reward_gain, distance_from_goal)
        reward = -np.sum(reward_vector)
        return reward
