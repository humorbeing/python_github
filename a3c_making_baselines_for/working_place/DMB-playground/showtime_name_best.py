import argparse
import os
import gym
import numpy as np
import math
import cv2
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim


from this_utility import *
from this_models import *
Model = RNN_only

def ss(s=''):
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    # print('        >>>>>>>>>>>>>>>>>>>>                <<<<<<<<<<<<<<<<<<<<        ')
    print(s)
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    import sys
    sys.exit()


actions = 2
action_map = {
    0: 2,
    1: 3
}
mm = Model(actions, action_map)
env_name = 'Pong-ram-v0'
env = gym.make(env_name)
env._max_episode_steps = 1000000
# env.seed(args.seed + rank)
# torch.manual_seed(args.seed + rank)
is_test_render = True
# is_test_render = False
# model = Model(actions, action_map)
best_reward = -999
seed = 51

torch.manual_seed(seed)
state = env.reset()
env.seed(seed)
reward_sum = 0
done = True

start_time = time.time()
episode_length = 0
PATH = './model_save_1/'
filelist = os.listdir(PATH)
onefilepath = './model_save_1/gcp-cpu8-R500-a3c_model.pytorch'
mm.load_state_dict(torch.load(onefilepath, map_location=lambda storage, loc: storage))
mm.eval()
torch.manual_seed(seed)
while True:
    # onefilepath = os.path.join(PATH, f)
    # print(onefilepath)
    # ss('1')
    # print('loading:',onefilepath)
    onefilepath = './model_save_1/gcp-cpu8-R200-a3c_model.pytorch'
    # onefilepath = './model_save_1/gcp-cpu8-R500-a3c_model.pytorch'
    onefilepath = './model_save_1/note9-cl-so-noFreeze_model.pytorch'
    onefilepath = './model_save_1/gcp-cpu8-R500-a3c_model.pytorch'
    onefilepath = './model_save_1/gcp-pyt2-R500-dmb-soso_model.pytorch'


    while True:
        # print('hi')
    # ss('1')
        episode_length += 1
        # Sync with the shared model
        if is_test_render:
            env.render()
            # env.seed(seed)
        if done:
            # model.load_state_dict(shared_model.state_dict())
            h1 = torch.zeros(1, 16)
            c1 = torch.zeros(1, 16)

        action, h1, c1 = mm.test(state, h1, c1)

        state, reward, done, _ = env.step(action)

        reward_sum += reward
        # ss('1')
        if done:
            print('reward:', reward_sum)
            # if reward_sum > 15:

            # print('hi')
            log_string = "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                0, 0 / (time.time() - start_time),
                reward_sum, episode_length)
            # print(log_string)
            # log.log(log_string)
            if reward_sum > best_reward:
                best_reward = reward_sum
                print('seed:', seed, 'loading:', onefilepath)
                print('reward:', reward_sum)
            #     save_this_model(model, log_name)
            reward_sum = 0
            episode_length = 0
            # env_name = 'Pong-ram-v0'
            # env = gym.make(env_name)
            # env._max_episode_steps = 100000
            state = env.reset()
            # env.seed(seed)
            break
            # time.sleep(5)
