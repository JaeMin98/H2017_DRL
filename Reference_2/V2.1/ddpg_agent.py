import numpy as np
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_models import Actor, Critic
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from config import CONFIG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """
    DDPG (Deep Deterministic Policy Gradient) agent.
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size: int, action_size: int, random_seed: int) -> None:
        """
        Initialize an Agent object.
        
        Args:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        random_seed (int): Random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_critic_loss = None

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=CONFIG['LR_ACTOR'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CONFIG['LR_CRITIC'], weight_decay=CONFIG['WEIGHT_DECAY'])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'], random_seed)
    
    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, timestep: int) -> None:
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        
        Args:
        states (np.ndarray): Current states
        actions (np.ndarray): Actions taken
        rewards (np.ndarray): Rewards received
        next_states (np.ndarray): Next states
        dones (np.ndarray): Done flags
        timestep (int): Current timestep
        """
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > CONFIG['BATCH_SIZE'] and timestep % CONFIG['UPDATE_INTERVAL'] == 0:
            experiences = self.memory.sample()
            self.learn(experiences, CONFIG['GAMMA'])

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Returns actions for given state as per current policy.
        
        Args:
        state (np.ndarray): Current state
        add_noise (bool): Whether to add noise for exploration
        
        Returns:
        np.ndarray: Chosen action
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self) -> None:
        """Reset the noise process."""
        self.noise.reset()

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], gamma: float) -> None:
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Args:
        experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples 
        gamma (float): Discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, CONFIG.TAU)
        self.soft_update(self.actor_local, self.actor_target, CONFIG.TAU) 

        self.agent_critic_loss = critic_loss.item()

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
        local_model (torch.nn.Module): Model to copy from
        target_model (torch.nn.Module): Model to copy to
        tau (float): Interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)