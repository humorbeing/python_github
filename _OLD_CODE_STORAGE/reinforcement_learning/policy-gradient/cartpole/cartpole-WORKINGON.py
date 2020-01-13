import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE_policy = 0.0001
LEARNING_RATE_value = 0.0001
# RENDER = True
RENDER = False
env = gym.make('CartPole-v1')
action_map = {
    0: 0,
    1: 1
}
input_size = 4
action_size = 2
hidden_size = 6

class Policy(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)
        dice = Categorical(out)
        sample_number = dice.sample()
        log_pi = dice.log_prob(sample_number)
        sample_number = sample_number.item()
        return sample_number, log_pi

policy = Policy(input_size, hidden_size, action_size)
policy_optimizer = optim.SGD(policy.parameters(), lr=LEARNING_RATE_policy)


class ValueNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

value = ValueNetwork(input_size, hidden_size, 1)
value_optim = optim.SGD(value.parameters(), lr=LEARNING_RATE_value)


def discounted_return(rewards):
    gs = []
    for t in range(len(rewards)):
        g = 0
        for i, reward in enumerate(rewards[t:]):
            g += GAMMA ** i * reward
        gs.append(g)
    return gs

def tensor_me(s):
    s = s[None, :]
    s = torch.Tensor(s)
    return s

while True:
    observe = env.reset()
    grads = []
    rewards = []
    score = 0
    state_now = observe
    state_now = tensor_me(state_now)
    I = 1
    while True:
        if RENDER:
            env.render()
        state_now = Variable(state_now)
        num, log_pi = policy(state_now)
        v_now = value(state_now)
        action = action_map[num]

        observe, reward, done, _ = env.step(action)
        state_next = observe
        state_next = tensor_me(state_next)
        v_next = value(state_next)
        # print()
        # print(v_next)
        # print(v_next.detach())


        target = reward + GAMMA * v_next.detach()
        delta = target - v_now.detach()
        value_loss = (v_now - target) ** 2 * I
        # print(value_loss)

        value_optim.zero_grad()
        # value_loss.backward()
        value_loss.backward(retain_graph=True)
        value_optim.step()
        # print(value_loss)
        policy_loss = delta * log_pi * I * (-1)
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        score += reward
        state_now = state_next
        # I = I * GAMMA
        if done:
            break
    print(score)
    # break
