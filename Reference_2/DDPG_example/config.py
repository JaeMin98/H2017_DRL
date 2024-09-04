import torch

# 환경 설정
ENV_NAME = 'Pendulum-v1'
SEED = 0

# 학습 설정
EPISODES = 1000
MAX_STEPS = 500

# DDPG 하이퍼파라미터
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.001
BATCH_SIZE = 64

# 리플레이 버퍼 설정
BUFFER_SIZE = 100000

# 노이즈 설정
NOISE_MU = 0
NOISE_THETA = 0.15
NOISE_SIGMA = 0.2

# 신경망 구조
HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

# 기기 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 저장 및 로드 설정
SAVE_INTERVAL = 100
ACTOR_PATH = 'actor.pth'
CRITIC_PATH = 'critic.pth'