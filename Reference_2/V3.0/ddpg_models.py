import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        layers = [nn.Linear(state_size, CONFIG['ACTOR_LAYERS'][0])]
        for i in range(1, len(CONFIG['ACTOR_LAYERS'])):
            layers.append(nn.Linear(CONFIG['ACTOR_LAYERS'][i-1], CONFIG['ACTOR_LAYERS'][i]))
        layers.append(nn.Linear(CONFIG['ACTOR_LAYERS'][-1], action_size))
        
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return F.tanh(self.layers[-1](x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, CONFIG['CRITIC_LAYERS'][0])
        layers = []
        for i in range(1, len(CONFIG['CRITIC_LAYERS'])):
            if i == 1:
                layers.append(nn.Linear(CONFIG['CRITIC_LAYERS'][i-1] + action_size, CONFIG['CRITIC_LAYERS'][i]))
            else:
                layers.append(nn.Linear(CONFIG['CRITIC_LAYERS'][i-1], CONFIG['CRITIC_LAYERS'][i]))
        layers.append(nn.Linear(CONFIG['CRITIC_LAYERS'][-1], 1))
        
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)