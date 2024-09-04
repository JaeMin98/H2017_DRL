import Env
import numpy as np
from ddpg_agent import Agent 
import torch
import wandb

wandb.init(project='H2017_Refer2')
wandb.run.name = 'DDPG'
wandb.run.save()

env = Env.RobotArmControl()
agent = Agent(state_size=6, action_size=3, random_seed=123456)

episode_success, success_rate_list = [], []

def ddpg(n_episodes=100000, max_t=200):
    for i_episode in range(1, n_episodes+1):
        env.reset()
        states = env.get_state()
        agent.reset()

        scores = 0
        episode_ciritic_loss = None
        for timestep in range(max_t):
            if len(agent.memory) < agent.memory.batch_size * 10:
                actions = np.random.uniform(-1, 1, size=3)
            else:
                actions = agent.act(np.array(states), add_noise=True)
            next_states, rewards, dones, success = env.step(actions)
            ciritic_loss = agent.step(np.array([states]), np.array([actions]), np.array([rewards]), np.array([next_states]), np.array([dones]), timestep)
            if(ciritic_loss != None) : episode_ciritic_loss = ciritic_loss
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        episode_success.append(success)
        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)

        log_data = {
            'episode_reward': scores,
            'success_rate': success_rate,
            'memory_size': len(agent.memory),
        }
        if(episode_ciritic_loss != None):
            wandb.log({'ciritic_loss':episode_ciritic_loss}, step=i_episode)

        wandb.log(log_data, step=i_episode)

        print(f"Episode: {i_episode}, Reward: {scores}")

        if success:
            torch.save(agent.actor_local.state_dict(), f'./models/{i_episode}_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'./models/{i_episode}_critic.pth')

        if len(success_rate_list) > 4 and np.mean(success_rate_list[-5:]) >= 0.9:
            torch.save(agent.actor_local.state_dict(), 'actor_solved.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_solved.pth')
            break

if __name__ == "__main__":
    ddpg()