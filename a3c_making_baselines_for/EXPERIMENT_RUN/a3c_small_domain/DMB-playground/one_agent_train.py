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

# from ss import ss
from this_utility import *
from this_models import *
from encoder_model import encoder

action_map = {
    0: 2,
    1: 3
}
# vae = VAE()
# vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=lambda storage, loc: storage))
ENCODER_MODEL_PATH = '/mnt/36D4F815D4F7D559/workspace/python_github/a3c_making_baselines_for/working_place/encoder_decoder/model_save/WM_competing_loss_lambda_1_model.pytorch'
ENCODER_MODEL_PATH = '/mnt/36D4F815D4F7D559/workspace/python_github/a3c_making_baselines_for/working_place/encoder_decoder/model_save/enc_dec_1_model.pytorch'
en = encoder()
# for param in en.encoder.parameters():
#     param.requires_grad = False
en.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=lambda storage, loc: storage))
# en.load_state_dict(torch.load(ENCODER_MODEL_PATH))

# for param in en.encoder.parameters():
#     param.requires_grad = False
model = RNN_only(2, action_map)
for param in model.encoder.parameters():
    param.requires_grad = False
# print(en.encoder.state_dict())
# print(model.encoder.state_dict())
model.encoder.load_state_dict(en.encoder.state_dict())
# print(model.encoder.state_dict())
# for param in model.encoder.parameters():
#     param.requires_grad = False
for p in model.encoder.parameters():
    print(p)
ss('s')

def train():
    env = gym.make('Pong-ram-v0')

    # model = RNN_vae(2, action_map)
    model.train()
    # model.eval()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    state = env.reset()

    done = True
    episode_length = 0
    while True:

        # hx = torch.zeros(1, 16)
        # cx = torch.zeros(1, 16)




        values = []
        log_probs = []
        rewards = []
        entropies = []
        while True:
            episode_length += 1
            z = vae.get_z(state)
            # ss()
            action, hx, cx = model(z, hx, cx)
            entropies.append(model.entropy)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            values.append(model.v)
            log_probs.append(model.log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = 0.99 * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + 0.99 * values[i + 1].data - values[i].data
            gae = gae * 0.99 + delta_t

            policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        # ensure_shared_grads(model, shared_model)
        optimizer.step()
        print('loss:', loss)

train()
