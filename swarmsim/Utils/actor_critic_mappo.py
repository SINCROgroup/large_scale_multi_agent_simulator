# ppo_agent/models/actor_critic_mappo.py

import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from swarmsim.Utils.actor_critic_base import ActorCriticBase, layer_init, get_activation


class ActorCriticMAPPO(ActorCriticBase):
    def __init__(self, envs, network_config):
        # Assuming observations are per agent
        obs_shape = envs.observation_space.shape[-1]  # Observation per agent
        action_shape = envs.action_space.nvec[0]  # Assuming all agents have the same action space
        super(ActorCriticMAPPO, self).__init__(obs_shape, action_shape, network_config)

        # Build actor and critic heads
        last_hidden_size = network_config['hidden_sizes'][-1]

        # Critic head
        self.critic = nn.Sequential(
            self.critic_base_net,
            layer_init(nn.Linear(last_hidden_size, 1), std=1.0)
        )

        # Actor head
        self.actor = nn.Sequential(
            self.actor_base_net,
            layer_init(nn.Linear(last_hidden_size, action_shape), std=0.01)
        )

        # Number of discrete actions per agent
        self.nvec = envs.action_space.nvec  # Shape: (num_agents,)
        self.num_agents = len(self.nvec)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'MAPPO_params.pt')
        self.device = torch.device('cpu')
        self.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def get_value(self, x):
        # x shape: (batch_size, obs_shape)
        return self.critic(x).squeeze(-1)  # Output shape: (batch_size,)

    def get_action_and_value(self, x, action=None):
        # x shape: (batch_size, obs_shape)
        logits = self.actor(x)  # Output shape: (batch_size, action_shape)
        dist = Categorical(logits=logits)  # Assuming single categorical per agent
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(x)
        return action, logprob, entropy, value

    def get_action(self, x):
        # x shape: (batch_size, obs_shape)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1)
        return action
