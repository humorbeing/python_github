import torch
import torch.nn as nn
import torch.nn.functional as F
from this_utility import *

class Policy(nn.Module):
    def __init__(self, action_space, action_map, num_inputs=3, lstm_input=64, lstm_output=64):
        super(Policy, self).__init__()
        self.action_map = action_map
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
        # self.reset_hidden()
        self.train()

    # def reset_hidden(self):
    #     self.cx = torch.zeros(1, self.lstm_output)
    #     self.hx = torch.zeros(1, self.lstm_output)

    def forward(self, state, hx, cx):
        state = image_pre_process(state)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.cnn(state)

        x = x.view(-1, 16 * 3 * 3)
        x = self.mid_linear(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        action = self.action_map[selected.numpy()[0, 0]]
        # if self.training:
        log_prob_all = torch.log(prob)
        self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
        self.log_prob = log_prob_all.gather(1, selected)
        self.v = self.critic_linear(x)
        return action, hx, cx