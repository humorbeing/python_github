import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
# RENDER = True
RENDER = False
env = gym.make('CartPole-v1')
action_map = {
    0: 0,
    1: 1
}

class Policy(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self, x, a):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        out = torch.mm(out, a)
        return out

    def get_action(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)
        dice = Categorical(out)
        sample_number = dice.sample()
        sample_number = sample_number.data.numpy()[0]
        return sample_number

policy = Policy(4, 6, 2)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

def discounted_return(rewards):
    gs = []
    for t in range(len(rewards)):
        g = 0
        for i, reward in enumerate(rewards[t:]):
            g += GAMMA ** i * reward
        gs.append(g)
    return gs

def sample_number_mask(n):
    mask = np.zeros((2, 1))
    mask[n] = 1
    return mask

while True:
    state = env.reset()
    grads = []
    rewards = []
    states = []
    score = 0
    count = 0
    while True:
        if RENDER:
            env.render()
        state = state[None, :]
        state = torch.Tensor(state)
        num = policy.get_action(state)
        mask = sample_number_mask(num)
        mask = torch.Tensor(mask)
        mask = Variable(mask)
        state = Variable(state)
        log_prob = policy(state, mask)
        grads.append(log_prob)
        action = action_map[num]
        state, reward, done, _ = env.step(action)
        score += reward
        rewards.append(reward)
        if done:
            break
    dr = discounted_return(rewards)
    dr = torch.Tensor(dr)
    grads = torch.stack(grads)
    grads = grads.view(-1)
    loss = grads*dr*-1
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
