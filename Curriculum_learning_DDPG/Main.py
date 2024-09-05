import Env
import numpy as np
from ddpg_agent import Agent
import wandb
import datetime
import Config
import os

class DDPG:
    def __init__(self, n_episodes=Config.n_episodes, max_t=Config.max_episode_steps):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.env = self.create_environment()
        self.agent = self.create_agent(state_size=9, action_size=3, random_seed=123456)
        self.episode_success = []
        self.success_rate_list = []
        self.folder_name = f'models/DDPG_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{Config.Current_Data_Selection_Ratio}_{Config.action_weight}'
        self.memory_threshold = self.agent.memory.batch_size * 10

    def create_environment(self):
        return Env.RobotArmControl()

    def create_agent(self, state_size, action_size, random_seed):
        return Agent(state_size, action_size, random_seed)

    def get_action(self, states):
        if len(self.agent.memory) < self.memory_threshold:
            return np.random.uniform(-1, 1, size=3)
        return self.agent.act(np.array(states), add_noise=True)

    def update_agent(self, states, actions, rewards, next_states, dones, timestep):
        return self.agent.step(np.array([states]), np.array([actions]), np.array([rewards]),
                               np.array([next_states]), np.array([dones]), timestep)

    def log_wandb(self, log_data, step):
        wandb.log(log_data, step=step)

    def should_increase_level(self):
        if len(self.success_rate_list) > 15 and np.mean(self.success_rate_list[-5:]) >= 0.9:
            return self.env.Level_Of_Point < self.env.MAX_Level_Of_Point
        return False

    def reset_for_next_level(self):
        self.episode_success = []
        self.success_rate_list = []
        self.agent.memory.reset()
        self.agent.noise.reset()
        self.env.Level_Of_Point += 1

    def run_episode(self):
        self.env.reset()
        states = self.env.get_state()
        self.agent.reset()
        scores = 0
        episode_critic_loss = None
        for timestep in range(self.max_t):
            actions = self.get_action(states)
            next_states, rewards, dones, success = self.env.step(actions)
            critic_loss = self.update_agent(states, actions, rewards, next_states, dones, timestep)
            if critic_loss is not None:
                episode_critic_loss = critic_loss
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        return scores, success, episode_critic_loss

    def save_checkpoint(self, episode):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        ckpt_path = os.path.join(self.folder_name, f'{episode}.tar')
        self.agent.save_checkpoint(ckpt_path)

    def train(self):
        for i_episode in range(1, self.n_episodes + 1):
            scores, success, episode_critic_loss = self.run_episode()
            self.episode_success.append(success)
            success_rate = np.mean(self.episode_success[-min(10, len(self.episode_success)):])
            self.success_rate_list.append(success_rate)

            log_data = {
                'episode_reward': scores,
                'success_rate': success_rate,
                f'level_{self.env.Level_Of_Point}': success_rate,
                'memory_size': len(self.agent.memory),
            }
            self.log_wandb(log_data, i_episode)

            if episode_critic_loss is not None:
                self.log_wandb({'critic_loss': episode_critic_loss}, i_episode)

            print(f"Episode: {i_episode}, Reward: {scores}, level: {self.env.Level_Of_Point}")

            if success:
                self.save_checkpoint(i_episode)

            if self.should_increase_level():
                self.reset_for_next_level()
            elif self.env.Level_Of_Point >= self.env.MAX_Level_Of_Point:
                break

if __name__ == "__main__":
    wandb.init(project='H2017_DDPG')
    wandb.run.name = 'DDPG'
    wandb.run.save()
    ddpg = DDPG()
    ddpg.train()