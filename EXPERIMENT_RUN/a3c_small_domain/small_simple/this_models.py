import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from this_utility import *


class RNN_only(nn.Module):
    def __init__(self, action_space, action_map):
        super(RNN_only, self).__init__()
        self.action_map = action_map
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(32, 16)
        self.critic_linear = nn.Linear(16, 1)
        self.actor_linear = nn.Linear(16, action_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, state, hx, cx):
        # state = image_pre_process(state)
        state = state.astype(np.float32)
        state = state/255.0
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.layers(state)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = self.action_map[selected.numpy()[0, 0]]
        if self.training:
            log_prob_all = torch.log(prob)
            self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
            self.log_prob = log_prob_all.gather(1, selected)
            self.v = self.critic_linear(x)
        return action, hx, cx


class RNN_EXT(nn.Module):
    def __init__(self, action_space, action_map):
        super(RNN_EXT, self).__init__()
        self.action_map = action_map

        self.lstm1 = nn.LSTMCell(128, 64)
        self.lstm2 = nn.LSTMCell(64, 32)
        self.lstm3 = nn.LSTMCell(32, 16)
        self.lstm12 = nn.LSTMCell(64, 64)
        self.lstmc = nn.LSTMCell(16, 1)
        self.lstma = nn.LSTMCell(16, action_space)
        # self.critic_linear = nn.Linear(16, 1)
        # self.actor_linear = nn.Linear(16, action_space)

        self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)
        self.train()

    def forward(self, state, h1, c1, h2, c2, h3, c3, h12, c12, hc, cc, ha, ca):
        state = state.astype(np.float32)
        state = state / (255.0 * 2)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        h1, c1 = self.lstm1(state, (h1, c1))
        x = h1
        h12, c12 = self.lstm12(x, (h12, c12))
        x = x + h12
        h2, c2 = self.lstm2(x, (h2, c2))
        x = h2
        h3, c3 = self.lstm3(x, (h3, c3))
        x = h3
        hc, cc = self.lstmc(x, (hc, cc))
        ha, ca = self.lstma(x, (ha, ca))

        logit = ha
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = self.action_map[selected.numpy()[0, 0]]
        if self.training:
            log_prob_all = torch.log(prob)
            self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
            self.log_prob = log_prob_all.gather(1, selected)
            self.v = hc
        return action, h1, c1, h2, c2, h3, c3, h12, c12, hc, cc, ha, ca


class PixelPolicy(nn.Module):
    def __init__(self, action_space, action_map):
        super(PixelPolicy, self).__init__()
        self.action_map = action_map
        self.layers = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU()
        )

        self.critic_linear = nn.Linear(16, 1)
        self.actor_linear = nn.Linear(16, action_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        # self.layers.weight.data = normalized_columns_initializer(
        #     self.layers.weight.data, 1.0)
        # self.layers.bias.data.fill_(0)
        self.train()

    def forward(self, state1, state2):
        # state = image_pre_process(state)
        state = state1 + 255
        state = state - state2
        state = state.astype(np.float32)
        state = state/(255.0 * 2)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.layers(state)

        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = self.action_map[selected.numpy()[0, 0]]
        # if self.training:
        log_prob_all = torch.log(prob)
        self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
        self.log_prob = log_prob_all.gather(1, selected)
        self.v = self.critic_linear(x)
        return action

class Pixel_RNN(nn.Module):
    def __init__(self, action_space, action_map):
        super(Pixel_RNN, self).__init__()
        self.action_map = action_map
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(32, 16)
        self.critic_linear = nn.Linear(16, 1)
        self.actor_linear = nn.Linear(16, action_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, state1, state2, hx, cx):
        # state = image_pre_process(state)
        state = state1 + 255
        state = state - state2
        state = state.astype(np.float32)
        state = state/255.0
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.layers(state)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = self.action_map[selected.numpy()[0, 0]]
        if self.training:
            log_prob_all = torch.log(prob)
            self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
            self.log_prob = log_prob_all.gather(1, selected)
            self.v = self.critic_linear(x)
        return action, hx, cx