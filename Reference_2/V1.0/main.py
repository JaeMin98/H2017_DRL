import gym
import numpy as np
from ddpg_agent import DDPGAgent
import config
import wandb
import Env

wandb.init(project='Refer2')
wandb.run.name = 'DDPG'
wandb.run.save()


episode_success, success_rate_list = [], []

def main():
    env = Env.Ned2_control()
    np.random.seed(config.SEED)

    state_dim = 6
    action_dim = 3
    max_action = 1.0

    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    for episode in range(config.EPISODES):
        env.reset()  # OpenAI Gym의 새로운 버전에 맞춰 수정
        state = env.get_state()  # 새로운 env.get_state() 메소드 사용
        episode_reward = 0
        
        for step in range(config.MAX_STEPS):
            if(step <= config.BATCH_SIZE): action = np.random.uniform(-1, 1, 3)
            else : action = agent.select_action(np.array(state))  # state를 flatten하여 전달

            next_state, reward, done, truncated = env.step(action)  # 새로운 env.step() 반환 값에 맞춰 수정

            agent.store_transition(state, action, reward, next_state, done)
            if len(agent.memory) > config.BATCH_SIZE:
                actor_loss, critic_loss = agent.update()
                wandb.log({'critic_loss':critic_loss}, step=episode+1)

            state = next_state
            episode_reward += reward

            if done or truncated:  # truncated 조건 추가
                break

        wandb.log({'memory size':len(agent.memory)}, step=episode+1)
        episode_success.append(done)

        print(f"Episode: {episode+1}, Reward: {episode_reward}")
        wandb.log({'episode_reward':episode_reward}, step=episode+1)

        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)
        wandb.log({'success_rate':success_rate}, step=episode+1)

        if(len(success_rate_list) > 4):
            if np.mean(success_rate_list[-min(5, len(success_rate_list)):]) >= 0.9:
                agent.save_models()
                break

        if (episode + 1) % config.SAVE_INTERVAL == 0:
            agent.save_models()

if __name__ == "__main__":
    main()