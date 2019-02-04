'''
description of this code:
run random policy on game_name
record and store them in certain path.
'''

import random
from this_utility import *
import os
import gym
import numpy as np

MAX_FRAMES = 10000  # max length of game
MAX_TRIALS = 1000  # just use this to extract one trial.
is_render = False

file_path = VAE_DATA_PATH
if not os.path.exists(file_path):
    os.makedirs(file_path)
game_name = 'Pong-ram-v0'
env = gym.make(game_name)

total_frames = 0

for trial in range(MAX_TRIALS):
    try:
        random_generated_int = random.randint(0, 2 ** 31 - 1)

        filename = file_path + "/" + str(random_generated_int) + ".npz"
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        obs = env.reset()  # obs shape: (210, 160, 3)
        for frame in range(MAX_FRAMES):
            if is_render:
                env.render()

            # obs = image_pre_process(obs)    # obs shape: (3, 42, 42)

            recording_obs.append(obs)

            action = np.random.choice([2, 3])
            recording_action.append(action)
            obs, reward, done, info = env.step(action)

            if done:
                break

        total_frames += (frame + 1)
        print("dead at", frame + 1, "total recorded frames for this worker", total_frames)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)

        np.savez_compressed(filename, obs=recording_obs, action=recording_action)
    except gym.error.Error:
        print("stupid gym error, life goes on")
        env.close()
        continue
env.close()