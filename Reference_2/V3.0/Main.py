import Env
import numpy as np
from her_ddpg_agent import HERDDPGAgent 
import torch
import wandb
from config import CONFIG
import os

def init_wandb():
    wandb.init(project=CONFIG['PROJECT_NAME'], config=CONFIG)
    wandb.run.name = CONFIG['RUN_NAME']
    for key, value in CONFIG.items():
        wandb.config[key] = value
    wandb.run.save()

def save_model(agent, episode, success):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    if success:
        torch.save({
            'actor': agent.actor_local.state_dict(),
            'critic': agent.critic_local.state_dict()
        }, f'./models/{episode}_model.pth')
    
    if len(success_rate_list) > 4 and np.mean(success_rate_list[-5:]) >= 0.9:
        torch.save({
            'actor': agent.actor_local.state_dict(),
            'critic': agent.critic_local.state_dict()
        }, 'trained_model.pth')
        return True
    return False

def her_ddpg(n_episodes = CONFIG['N_EPISODES'], max_t = CONFIG['MAX_T']):
    for i_episode in range(1, n_episodes+1):
        env.reset()
        state = env.get_state()
        goal = env.target
        agent.reset()

        episode_reward = 0
        for timestep in range(max_t):
            if len(agent.memory) < agent.memory.batch_size * 10:
                action = np.random.uniform(-1, 1, size=3)
            else:
                action = agent.act(np.array(state), goal, add_noise=True)

            next_state, reward, done, success = env.step(action)

            agent.step(state, action, reward, next_state, done, goal, timestep)

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_success.append(success)
        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)

        log_data = {
            'episode_reward': episode_reward,
            'success_rate': success_rate,
            'memory_size': len(agent.memory),
        }
        if agent.agent_critic_loss is not None:
            log_data['critic_loss'] = agent.agent_critic_loss

        wandb.log(log_data, step=i_episode)

        print(f"Episode: {i_episode}, Reward: {episode_reward}, Success Rate: {success_rate:.2f}")

        if save_model(agent, i_episode, success):
            print("Training completed successfully!")
            break

if __name__ == "__main__":
    init_wandb()

    env = Env.Ned2_control()
    agent = HERDDPGAgent(state_size=9, action_size=3, random_seed=CONFIG['RANDOM_SEED'])

    episode_success, success_rate_list = [], []

    her_ddpg()