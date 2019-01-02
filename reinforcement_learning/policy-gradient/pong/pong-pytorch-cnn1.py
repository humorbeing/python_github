import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim

is_cuda = True
RENDER = False
# RENDER = True
H = 200
D = 160*160
GAMMA = 0.99
LEARNING_RATE = 1e-3
env = gym.make("Pong-v0")
action_map = {
    0: 1,  # stay still
    1: 2,  # up
    2: 3,  # down
}

class Policy(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 10, stride=2)
        self.conv2 = nn.Conv2d(4, 8, 10, stride=2)
        self.conv3 = nn.Conv2d(8, 10, 7, stride=3)
        self.fc1 = nn.Linear(1000, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = out.view(1, -1)
        out = F.relu(self.fc1(out))
        out = F.softmax(self.fc2(out), dim=1)
        dice = Categorical(out)
        sample_number = dice.sample()
        log_pi = dice.log_prob(sample_number)
        sample_number = sample_number.item()
        return sample_number, log_pi

policy = Policy(200, 3)
policy.load_state_dict(torch.load('/media/ray/SSD/workspace/python/reinforcement_learning/policy-gradient/m.m'))
if is_cuda:
    policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    # I = I[::2, ::2, 0]  # downsample by factor of 2
    I = I[:, :, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def discounted_return(rewards):
    gs = []
    for t in range(len(rewards)):
        g = 0
        for i, reward in enumerate(rewards[t:]):
            g += GAMMA ** i * reward
        gs.append(g)
    return gs
done = True
count = 0
when_save = 5000
while True:
    for cc in range(when_save):
        if done:
            state = env.reset()
        grads = []
        rewards = []
        score = 0
        while True:
            if RENDER:
                env.render()
            s = prepro(state)
            s = s[None, None, :, :]
            s = torch.Tensor(s)
            if is_cuda:
                s = s.cuda()
            s = Variable(s)
            num, log_pi = policy(s)
            grads.append(log_pi)
            action = action_map[num]
            state, reward, done, _ = env.step(action)
            score += reward
            rewards.append(reward)

            if reward == -1.0:
                break
            if reward == 1.0:
                break
        dr = discounted_return(rewards)
        dr = torch.Tensor(dr)

        grads = torch.stack(grads)
        if is_cuda:
            grads = grads.cuda()
            dr = dr.cuda()
        grads = grads.view(-1)
        loss = grads * dr * -1
        loss = torch.sum(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('cc:', cc, ' count:', count, ' score',score)
        if score == 1.0:
            count += 1
            # print('yeah !!!!!!', count)
        else:
            count = 0
        if count == 15:
            RENDER = True
    print('saving...')
    torch.save(policy.state_dict(), '/media/ray/SSD/workspace/python/reinforcement_learning/policy-gradient/m.m')


