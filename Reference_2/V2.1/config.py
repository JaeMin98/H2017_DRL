CONFIG = {
    # DDPG Agent hyperparameters
    'BUFFER_SIZE': int(1e6),
    'BATCH_SIZE': 2048,
    'GAMMA': 0.99,
    'TAU': 5e-3,
    'LR_ACTOR': 1.5e-4,
    'LR_CRITIC': 3e-4,
    'WEIGHT_DECAY': 0,
    'UPDATE_INTERVAL': 2,

    # OU Noise parameters
    'MU': 0.,
    'THETA': 0.15,
    'SIGMA': 0.2,

    # Environment parameters
    'MAX_TIME_STEP': 200,

    # Training parameters
    'N_EPISODES': 40000,
    'MAX_T': 200,

    # Wandb parameters
    'PROJECT_NAME': 'RobotArm',
    'RUN_NAME': 'DDPG',
}