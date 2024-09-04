import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_models import Actor, Critic
from ou_noise import OUNoise
from her_replay_buffer import HERReplayBuffer
from config import CONFIG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HERDDPGAgent:
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = CONFIG['GOAL_SIZE']
        self.seed = random.seed(random_seed)
        self.agent_critic_loss = None

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size + self.goal_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size + self.goal_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=CONFIG['LR_ACTOR'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size + self.goal_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size + self.goal_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CONFIG['LR_CRITIC'], weight_decay=CONFIG['WEIGHT_DECAY'])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = HERReplayBuffer(action_size, CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'], random_seed)
    
    def step(self, state, action, reward, next_state, done, goal, timestep):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, goal)

        # Learn, if enough samples are available in memory
        if len(self.memory) > CONFIG['BATCH_SIZE'] and timestep % CONFIG['UPDATE_INTERVAL'] == 0:
            experiences = self.memory.sample_her_batch()
            self.learn(experiences, CONFIG['GAMMA'])

    def act(self, state, goal, add_noise=True):
        state_goal = np.concatenate([state, goal])
        state_goal = torch.from_numpy(state_goal).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_goal).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, goals = experiences

        # Concatenate states and goals
        states_goals = torch.cat([states, goals], dim=1)
        next_states_goals = torch.cat([next_states, goals], dim=1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states_goals)
        Q_targets_next = self.critic_target(next_states_goals, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states_goals, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states_goals)
        actor_loss = -self.critic_local(states_goals, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, CONFIG['TAU'])
        self.soft_update(self.actor_local, self.actor_target, CONFIG['TAU']) 

        self.agent_critic_loss = critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)