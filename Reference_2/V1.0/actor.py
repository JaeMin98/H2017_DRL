import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, config.HIDDEN1_UNITS)
        self.fc2 = nn.Linear(config.HIDDEN1_UNITS, config.HIDDEN2_UNITS)
        self.fc3 = nn.Linear(config.HIDDEN2_UNITS, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))