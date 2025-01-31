import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork_LL(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork_LL, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.net = nn.Sequential(
            nn.Linear(*input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, int(n_actions**2)))

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        actions = self.forward(state)
        action = T.argmax(actions, dim=1)
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DeepQNetwork_HL(nn.Module):
    def __init__(self, n_actions, name, input_dims, chkpt_dir, lr=0):
        super(DeepQNetwork_HL, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)


        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

    def get_action(self, state):
        actions = self.forward(state)
        action = T.argmax(actions, dim=1)
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


def get_discrete_action(act: int, control_inputs: np.array) -> np.array:
    index = np.unravel_index(act, (control_inputs.shape[1], control_inputs.shape[1]))
    return np.array([control_inputs[0, index[0]], control_inputs[1, index[1]]])
