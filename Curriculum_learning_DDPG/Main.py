import Env
import numpy as np
from ddpg_agent import Agent
import torch
import wandb
import datetime
import Config
import os

def create_environment():
    return Env.RobotArmControl()

def create_agent(state_size, action_size, random_seed):
    return Agent(state_size, action_size, random_seed)

def get_action(agent, states, memory_threshold):
    if len(agent.memory) < memory_threshold:
        return np.random.uniform(-1, 1, size=3)
    return agent.act(np.array(states), add_noise=True)

def update_agent(agent, states, actions, rewards, next_states, dones, timestep):
    return agent.step(np.array([states]), np.array([actions]), np.array([rewards]),
                      np.array([next_states]), np.array([dones]), timestep)

def log_wandb(log_data, step):
    wandb.log(log_data, step=step)

def should_increase_level(success_rate_list, env):
    if len(success_rate_list) > 15 and np.mean(success_rate_list[-5:]) >= 0.9:
        return env.Level_Of_Point < env.MAX_Level_Of_Point
    return False

def reset_for_next_level(env, agent):
    episode_success = []
    success_rate_list = []
    agent.memory.reset()
    env.Level_Of_Point += 1
    return episode_success, success_rate_list

def run_episode(env, agent, max_t, memory_threshold):
    env.reset()
    states = env.get_state()
    agent.reset()
    scores = 0
    episode_critic_loss = None

    for timestep in range(max_t):
        actions = get_action(agent, states, memory_threshold)
        next_states, rewards, dones, success = env.step(actions)
        critic_loss = update_agent(agent, states, actions, rewards, next_states, dones, timestep)
        
        if critic_loss is not None:
            episode_critic_loss = critic_loss
        
        states = next_states
        scores += rewards
        
        if np.any(dones):
            break
    
    return scores, success, episode_critic_loss

def save_checkpoint(agent, folder_name, episode):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    ckpt_path = os.path.join(folder_name, f'{episode}.tar')
    agent.save_checkpoint(ckpt_path)

def ddpg(n_episodes=Config.n_episodes, max_t=Config.max_episode_steps):
    env = create_environment()
    agent = create_agent(state_size=9, action_size=3, random_seed=123456)
    episode_success, success_rate_list = [], []
    folder_name = f'models/DDPG_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{Config.Current_Data_Selection_Ratio}_{Config.action_weight}'
    memory_threshold = agent.memory.batch_size * 10

    for i_episode in range(1, n_episodes+1):
        scores, success, episode_critic_loss = run_episode(env, agent, max_t, memory_threshold)
        
        episode_success.append(success)
        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)
        
        log_data = {
            'episode_reward': scores,
            'success_rate': success_rate,
            f'level_{env.Level_Of_Point}': success_rate,
            'memory_size': len(agent.memory),
        }
        log_wandb(log_data, i_episode)
        
        if episode_critic_loss is not None:
            log_wandb({'critic_loss': episode_critic_loss}, i_episode)
        
        print(f"Episode: {i_episode}, Reward: {scores}, level: {env.Level_Of_Point}")
        
        if success:
            save_checkpoint(agent, folder_name, i_episode)
        
        if should_increase_level(success_rate_list, env):
            episode_success, success_rate_list = reset_for_next_level(env, agent)
        elif env.Level_Of_Point >= env.MAX_Level_Of_Point:
            break

if __name__ == "__main__":
    wandb.init(project='H2017_DDPG')
    wandb.run.name = 'DDPG'
    wandb.run.save()
    ddpg()