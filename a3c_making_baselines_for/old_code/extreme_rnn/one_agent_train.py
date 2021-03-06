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
# from vae_model import VAE

action_map = {
    0: 2,
    1: 3
}

# vae = VAE()
# vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=lambda storage, loc: storage))


def train():
    env = gym.make('Pong-ram-v0')

    model = RNN_EXT(2, action_map)
    model.train()
    # model.eval()
    optimizer = optim.Adam(model.parameters())
    state = env.reset()

    done = True
    episode_length = 0
    while True:

        # hx = torch.zeros(1, 16)
        # cx = torch.zeros(1, 16)
        h1 = torch.zeros(1, 64)
        c1 = torch.zeros(1, 64)
        h2 = torch.zeros(1, 32)
        c2 = torch.zeros(1, 32)
        h3 = torch.zeros(1, 16)
        c3 = torch.zeros(1, 16)
        h12 = torch.zeros(1, 64)
        c12 = torch.zeros(1, 64)
        hc = torch.zeros(1, 1)
        cc = torch.zeros(1, 1)
        ha = torch.zeros(1, 2)
        ca = torch.zeros(1, 2)



        values = []
        log_probs = []
        rewards = []
        entropies = []
        while True:
            episode_length += 1
            # z = vae.get_z(state)
            # ss()
            action, h1, c1, h2, c2, h3, c3, h12, c12, hc, cc, ha, ca = model(state, h1, c1, h2, c2, h3, c3, h12, c12, hc, cc, ha, ca)
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
