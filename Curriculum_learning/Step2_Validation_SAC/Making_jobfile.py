from sac import SAC
import Config
import Env
import time
import datetime
import torch
import csv

def read_csv_file(file_path):
    csv_content = ""
    with open(file_path, 'r') as file:
        csv_content = file.read()

    csv_data = []
    color_data = []

    for row in csv.reader(csv_content.split('\n')):
        if row and len(row) >= 4:  # 최소 4개의 값이 있는지 확인
            csv_data.append([float(val) for val in row[:3]])  # 첫 3개 값은 위치
            color_data.append(float(row[3]))  # 4번째 값은 색상에 사용

    return csv_data


def Run_Validation(model_name, target, target_count):
    # Environment
    env = Env.RobotArmControl()
    env.reset()

    # Agent
    agent = SAC(9, 3, Config)
    # model_name = 'model_1'
    # model_path = f'models/{model_name}.tar'
    # agent.load_checkpoint(model_path,evaluate=False)
    # agent.load_checkpoint(model_path,evaluate=False)

    checkpoint = torch.load(model_name)
    agent.policy.load_state_dict(checkpoint['model'])
    agent.policy_optim.load_state_dict(checkpoint['optimizer'])


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
    env.make_job_file(f"{target_count+1}, XYZ:{str(target[0])},{str(target[1])},{str(target[2])}", f"nodes:{len(env.job_list)}, {date_time}")

    return len(env.job_list)


def write_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([item])  # 각 항목을 리스트로 감싸 1열로 작성
    
if __name__ == '__main__':
    csv_content = read_csv_file('matching_rows.csv')
    jobfile_node = []
    for i in range (len(csv_content)) :
        node = Run_Validation('trained_models/SAC_7113epi_weight20.tar', csv_content[i], i)
        print(node)
        jobfile_node.append(node)

    write_list_to_csv(jobfile_node, 'jobfile_nodes.csv')
    