import gym
import numpy as np
from ddpg_agent import DDPGAgent
import config

def main():
    env = gym.make(config.ENV_NAME)
    # env.seed(config.SEED)
    np.random.seed(config.SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(max_action)

    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    for episode in range(config.EPISODES):
        state, _ = env.reset()  # OpenAI Gym의 새로운 버전에 맞춰 수정
        episode_reward = 0

        for step in range(config.MAX_STEPS):
            action = agent.select_action(state.flatten())  # state를 flatten하여 전달
            next_state, reward, done, truncated, _ = env.step(action)  # 새로운 env.step() 반환 값에 맞춰 수정
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.memory) > config.BATCH_SIZE:
                agent.update()

            state = next_state
            episode_reward += reward

            if done or truncated:  # truncated 조건 추가
                break

        print(f"Episode: {episode+1}, Reward: {episode_reward}")

        if (episode + 1) % config.SAVE_INTERVAL == 0:
            agent.save_models()

if __name__ == "__main__":
    main()