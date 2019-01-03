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

class Phi(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Phi, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

phi = Phi(input_size, 5, 3)
phi_optim = optim.Adam(phi.parameters(), lr=LEARNING_RATE_value)

class State_action_to_next_state(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(State_action_to_next_state, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

forward = State_action_to_next_state(3+3, 5, 3)
# forward_optim = optim.Adam(forward.parameters(), lr=LEARNING_RATE_value)


class State_state_to_action(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(State_state_to_action, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out
inverse_dyn = State_state_to_action(3+3, 5, 3)
# inverse_dyn_optim = optim.Adam(inverse_dyn.parameters(), lr=LEARNING_RATE_value)

paras = list(phi.parameters()) + list(forward.parameters()) + list(inverse_dyn.parameters())
the_optim = optim.Adam(paras, lr=LEARNING_RATE_value)

mse_loss = torch.nn.MSELoss()
cross_loss = torch.nn.CrossEntropyLoss()

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



# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# print(input)
# print(target)
# print(output)
# print('dsasdfasdfasdf')

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
        action_vector = np.array([0, 0, 0])
        action_vector[num] = 1.0
        action_vector = tensor_me(action_vector)
        # action_vector = action_vector.type(torch.LongTensor)
        phi_S_t = phi(state_now)
        # v_now = value(state_now)
        action = action_map[num]
        observe, reward, done, _ = env.step(action)
        state_next = observe
        state_next = tensor_me(state_next)
        phi_S_t1 = phi(state_next)
        forward_vector = torch.cat((phi_S_t, action_vector), dim=1)
        hat_phi_s_t1 = forward(forward_vector)
        inv_vector = torch.cat((phi_S_t, phi_S_t1), dim=1)
        # print('inv:', inv_vector)
        hat_action = inverse_dyn(inv_vector)
        # hat_action = hat_action.type(torch.FloatTensor)
        # print('hat action:', hat_action)
        # print('---')
        # print(phi_S_t1)
        # print(hat_phi_s_t1)
        loss = phi_S_t1 - hat_phi_s_t1
        # print(loss)
        loss = loss ** 2
        # print(loss)
        m_loss = torch.sum(loss, dim=1)
        # print(loss)
        # print('oopipoi')
        # print(action_vector)
        action_vector = action_vector.type(torch.LongTensor)
        # print(action_vector)
        # print(hat_action)
        target_action = torch.Tensor([num]).type(torch.LongTensor)
        # print(num)
        # print(target_action)
        c_loss = cross_loss(hat_action, target_action)
        # print('222')
        # print(hat_action)
        # print(type(hat_action))
        # print(action_vector)
        # print(type(action_vector))
        # print(c_loss)
        beta = 0.5
        loss = (1 - beta) * m_loss + beta * c_loss
        the_optim.zero_grad()
        loss.backward()
        the_optim.step()
        # break
        print(loss)
        score += reward
        state_now = state_next
        # I = I * GAMMA
        if done:
            break
    # print(score)
    # break
