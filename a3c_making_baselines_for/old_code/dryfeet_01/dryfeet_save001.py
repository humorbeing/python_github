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

from ss import ss
from this_utility import *


action_map = {
    0: 2,
    1: 3
}

class ActorCritic(torch.nn.Module):
    def __init__(self, action_space, num_inputs=3, lstm_input=64, lstm_output=64):
        super(ActorCritic, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_inputs, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, stride=2),
            nn.ReLU())
        self.lstm_output=lstm_output
        self.mid_linear = nn.Linear(16 * 3 * 3, lstm_input)

        self.lstm = nn.LSTMCell(lstm_input, lstm_output)
        self.critic_linear = nn.Linear(lstm_output, 1)
        self.actor_linear = nn.Linear(lstm_output, action_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        # self.mid_linear.weight.data = normalized_columns_initializer(
        #     self.mid_linear.weight.data, 1.0)
        # self.mid_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def reset_hidden(self):
        self.cx = torch.zeros(1, self.lstm_output)
        self.hx = torch.zeros(1, self.lstm_output)

    def forward(self, state):
        state = image_pre_process(state)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.cnn(state)

        x = x.view(-1, 16 * 3 * 3)
        x = self.mid_linear(x)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx

        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = action_map[selected.numpy()[0, 0]]
        if self.training:
            log_prob_all = torch.log(prob)
            self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
            self.log_prob = log_prob_all.gather(1, selected)
            self.v = self.critic_linear(x)
        return action


def train():
    env = gym.make('Pong-v0')
    # env.seed(args.seed + rank)
    # torch.manual_seed(args.seed + rank)

    model = ActorCritic(2)
    model.train()
    # model.eval()
    optimizer = optim.Adam(model.parameters())
    state = env.reset()
    # state = tensor_state(state)
    done = True
    episode_length = 0
    while True:

        if done:
            model.reset_hidden()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        while True:
            episode_length += 1
            action = model(state)
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
