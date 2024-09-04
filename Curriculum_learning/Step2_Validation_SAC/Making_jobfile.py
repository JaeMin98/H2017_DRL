from sac import SAC
import Config
import Env
import time
import datetime

def Run_Validation(model_name = 'model_1', target = [0,1,1]):
    # Environment
    env = Env.RobotArmControl()
    env.reset()

    # Agent
    agent = SAC(9, 3, Config)
    model_name = 'model_1'
    model_path = f'models/{model_name}.tar'
    agent.load_checkpoint(model_path,evaluate=False)
    # agent.load_checkpoint(model_path,evaluate=False)

    # checkpoint = torch.load(model_path)
    # agent.policy.load_state_dict(checkpoint['model'])
    # agent.policy_optim.load_state_dict(checkpoint['optimizer'])


    done = False
    env.reset()
    env.target = target
    env.target_reset()

    state = env.get_state()

    while not done:
        action = agent.select_action(state)  # action from policy

        env.action(action)
        time.sleep(Config.time_sleep_interval)
        next_state, reward, done, success = env.observation() # Step

        state = next_state
    
    now = datetime.datetime.now()
    date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    env.make_job_file(f"XYZ:{str(target[0])},{str(target[1])},{str(target[2])}", date_time)


    
if __name__ == '__main__':
    Run_Validation()