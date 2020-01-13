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
LR = 0.01
# RENDER = True
RENDER = False
theta = np.random.rand(4, 2)
w = np.random.rand(4,1)
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
        p = dice.log_prob(sample_number)
        # print('log_p', p)
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
        # g = [g]
        # g = torch.Tensor(g)
        # g = Variable(g)
        # print('g',g)
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
        # state = np.random.rand(2, 4)
        state = torch.Tensor(state)
        num = policy.get_action(state)
        # print('sample number', num)
        mask = sample_number_mask(num)
        # print('mask is', mask)
        # print('mask shape', mask.shape)
        state = Variable(state)
        mask = torch.Tensor(mask)
        mask = Variable(mask)
        log_prob = policy(state, mask)
        # print('out', log_prob)
        # print('out shape', log_prob.shape)

    #     log_softmax = policy(state)
    #     action_distribution = torch.exp(log_softmax)
    #     print(state)
    #     print(state.shape)
    #     print(log_softmax)
    #     print(log_softmax.shape)
    #     print(action_distribution)
    #     m = Categorical(action_distribution)
    #     print(m)
    #     sample_number = m.sample()
    #     g = m.log_prob(sample_number)
    #     g = g.item()
        grads.append(log_prob)
    #     print(g)
    #     # sample_number = sample_number
    #     sample_number = sample_number.item()
    #
    #     print(sample_number)
    #
    #
        action = action_map[num]
    #     print(action)
        state, reward, done, _ = env.step(action)
        score += reward
        # reward = reward
        rewards.append(reward)
        if done:
        #     count += 1
        # if count == 100:
        #     env.close()
            break
    dr = discounted_return(rewards)
    dr = torch.Tensor(dr)
    # print(dr.shape)
    grads = torch.stack(grads)
    grads = grads.view(-1)
    # print(dr)
    # print(grads.shape)
    loss = grads*dr*-1
    # print(loss.shape)
    # print(dr)
    # print(grads)
    # print(loss)
    loss = torch.sum(loss)
    # loss = torch.zeros(len(grads))
    # for i in range(len(loss)):
    #     loss[i] = dr[i] * grads[i] * -1
    # loss = torch.sum(loss)
    # loss = dr*grads

    # print(loss)
    # grads = np.array(grads)
    #
    # print(grads)
    # print(dr)
    optimizer.zero_grad()
    # grads = torch.Tensor(grads)
    # dr = torch.Tensor(dr)
    # grads = Variable(grads)
    # dr = Variable(dr)
    # print(grads.shape)
    # print(dr.shape)
    # loss = dr
    # loss = grads
    # loss = torch.sum(loss)
    # # loss = Variable(loss)
    loss.backward()
    optimizer.step()
    # print(loss)
    # print(loss.shape)
    print(score)
    # print(rewards)
    # break



