import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, config.HIDDEN1_UNITS)
        self.fc2 = nn.Linear(config.HIDDEN1_UNITS, config.HIDDEN2_UNITS)
        self.fc3 = nn.Linear(config.HIDDEN2_UNITS, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)