#! /usr/bin/env python3
# -*- coding: utf-8 -*-

env_name = 'H2017'


#agent or model
BUFFER_SIZE = 1000000  # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 0.0003         # learning rate of the actor 
LR_CRITIC = 0.0006        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_INTERVER = 2
HIDDEN_SIZE = 64


#environment
Current_Data_Selection_Ratio = 0.8
n_episodes = 100000
max_episode_steps = 200
action_weight = 20.0


#