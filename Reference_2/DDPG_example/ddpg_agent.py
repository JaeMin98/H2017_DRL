import torch
import torch.nn.functional as F
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise
import config

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(config.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(config.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        self.noise = OUNoise(action_dim, config.NOISE_MU, config.NOISE_THETA, config.NOISE_SIGMA)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(config.DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        return (action + self.noise.sample() * self.max_action).clip(-self.max_action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.memory.sample(config.BATCH_SIZE)

        state = torch.FloatTensor(state).to(config.DEVICE)
        action = torch.FloatTensor(action).to(config.DEVICE)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(config.DEVICE)
        next_state = torch.FloatTensor(next_state).to(config.DEVICE)
        done = torch.FloatTensor(done).reshape(-1, 1).to(config.DEVICE)

        # Update Critic
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * config.GAMMA * target_Q
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)

    def save_models(self):
        torch.save(self.actor.state_dict(), config.ACTOR_PATH)
        torch.save(self.critic.state_dict(), config.CRITIC_PATH)

    def load_models(self):
        self.actor.load_state_dict(torch.load(config.ACTOR_PATH))
        self.critic.load_state_dict(torch.load(config.CRITIC_PATH))