import numpy as np

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import wrap_deepmind


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def ss(s):
    print('-> '*10)
    print(s)
    print('<- '*10)
    assert False, s + ' ---- in UUUUU.py'
env_name = 'Pong-v0'
seed = 1
num_processes = 2


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        env = wrap_deepmind(env)
        return env
    return _f

envs = []
for seed in range(num_processes):
    env = make_env(env_name, seed)
    envs.append(env)

envs = SubprocVecEnv(envs)
envs = VecFrameStack(envs, 4)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def bas_i(x):
    x = nn.init.constant_(x, 0)
    return x
def init_(m):
    m = init(m, nn.init.orthogonal_, bas_i, nn.init.calculate_gain('relu'))
    return m
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
def init_no_relu(m):
    m = init(m, nn.init.orthogonal_, bas_i)
    return m
def init_pi(m):
    m = init(m, nn.init.orthogonal_, bas_i, gain=0.01)
    return m


class Policy(nn.Module):
    def __init__(self, input_channel, output, hidden_size=512):
        super(Policy, self).__init__()

        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channel, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        self.critic_linear = init_no_relu(nn.Linear(hidden_size, 1))
        self.actor_linear = init_pi(nn.Linear(hidden_size, output))
        self.train()

    def forward(self, inputs):
        x = self.main(inputs)
        critic = self.critic_linear(x)
        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        return critic, prob


def tensor_state(state):
    state = np.moveaxis(state, -1, 1)
    state = state.astype(np.float32)
    state *= (1.0 / 255.0)
    state = torch.from_numpy(state)
    return state

class A2C():
    def __init__(self,
                 model,
                 lr):
        self.model = model
        self.optimizer = optim.RMSprop(model.parameters(), lr, eps=1e-5, alpha=0.99)

policy = Policy(4, 6)
num_steps = 5
states = []
values = []
probs = []
state = envs.reset()
state = tensor_state(state)
states.append(state)
while True:
    for step in range(num_steps):
        with torch.no_grad():
            value, prob = policy(state)
            log_prob = torch.log(prob)
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)
            state, reward, done, _ = envs.step(action.numpy())
            print(action)
            print(state.shape)
            print(reward)
            print(done)
            state = tensor_state(state)
            states.append(state)
            values.append(value)
            probs.append(prob)
            ss('show me')
