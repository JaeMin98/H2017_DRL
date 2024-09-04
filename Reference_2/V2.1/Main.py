import Env
import numpy as np
from ddpg_agent import Agent 
import torch
import wandb
from config import CONFIG

def init_wandb() -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.
    This function sets up the wandb project, run name, and logs all configuration parameters.
    """
    wandb.init(project=CONFIG['PROJECT_NAME'], config=CONFIG)
    wandb.run.name = CONFIG['RUN_NAME']
    # Log all CONFIG parameters
    for key, value in CONFIG.items():
        wandb.config[key] = value
    wandb.run.save()

def save_model(agent, episode, success) -> bool:
    """
    Save the trained model.
    
    Args:
    agent (Agent): The DDPG agent
    episode (int): Current episode number
    success (bool): Whether the episode was successful
    
    Returns:
    bool: True if training should be terminated, False otherwise
    """
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

def ddpg(n_episodes=CONFIG['N_EPISODES'], max_t=CONFIG['MAX_T']) -> None:
    """
    Implement the DDPG (Deep Deterministic Policy Gradient) algorithm.
    
    Args:
    n_episodes (int): Maximum number of training episodes
    max_t (int): Maximum number of timesteps per episode
    """
    for i_episode in range(1, n_episodes+1):
        env.reset()
        states = env.get_state()
        agent.reset()

        scores = 0
        for timestep in range(max_t):
            # Choose action
            actions = np.random.uniform(-1, 1, size=3) if len(agent.memory) < agent.memory.batch_size * 10 else agent.act(np.array(states), add_noise=True)
            # Take action and observe next state and reward
            next_states, rewards, dones, success = env.step(actions)
            # Store experience in replay memory and learn
            agent.step(states, actions, rewards, next_states, dones, timestep)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        # Update success rate
        episode_success.append(success)
        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)

        # Log data to wandb
        log_data = {
            'episode_reward': scores,
            'success_rate': success_rate,
            'memory_size': len(agent.memory),
        }
        if agent.agent_critic_loss is not None:
            log_data['critic_loss'] = agent.agent_critic_loss

        wandb.log(log_data, step=i_episode)

        print(f"Episode: {i_episode}, Reward: {scores}")

        # Check if training should be terminated
        if save_model(agent, i_episode, success):
            print("Training completed successfully!")
            break

if __name__ == "__main__":
    # Initialize wandb
    init_wandb()
    # Create environment and agent
    env = Env.Ned2_control()
    agent = Agent(state_size=6, action_size=3, random_seed=123456)
    # Initialize success tracking lists
    episode_success, success_rate_list = [], []
    # Start training
    ddpg()