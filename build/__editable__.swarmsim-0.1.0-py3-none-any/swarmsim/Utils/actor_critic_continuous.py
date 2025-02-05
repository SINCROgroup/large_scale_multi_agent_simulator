# ppo_agent/models/actor_critic_continuous.py

import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from swarmsim.Utils.actor_critic_base import ActorCriticBase, layer_init, get_activation


class ActorCriticContinuous(ActorCriticBase):
    def __init__(self, env, network_config):
        # obs_shape = env.observation_space.shape[0]
        # action_shape = env.action_space.shape[0]
        obs_shape = 4
        action_shape = 2
        super(ActorCriticContinuous, self).__init__(obs_shape, action_shape, network_config)

        # Build actor and critic heads
        last_hidden_size = network_config['hidden_sizes'][-1]

        # Critic head
        self.critic = nn.Sequential(
            self.critic_base_net,
            layer_init(nn.Linear(last_hidden_size, 1), std=1.0)
        )

        # Actor mean head
        self.actor_mean = nn.Sequential(
            self.actor_base_net,
            layer_init(nn.Linear(last_hidden_size, action_shape), std=0.01),
            nn.Tanh()
        )

        # Action log std
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_shape))

        # Action bounds
        self.action_min = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_max = torch.tensor(env.action_space.high, dtype=torch.float32)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'PPO_params.pt')
        self.device = torch.device('cpu')
        self.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.scale_action(self.actor_mean(x))
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def get_action(self, x):
        with torch.no_grad():
            action = self.scale_action(self.actor_mean(x))
        return action

    def scale_action(self, action):
        action_range = (self.action_max - self.action_min) / 2.0
        action_center = (self.action_max + self.action_min) / 2.0
        return action * action_range + action_center

    def to(self, device):
        self.action_min = self.action_min.to(device)
        self.action_max = self.action_max.to(device)
        return super(ActorCriticContinuous, self).to(device)
