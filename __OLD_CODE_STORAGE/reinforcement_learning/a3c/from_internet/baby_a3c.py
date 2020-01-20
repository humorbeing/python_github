# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

# from __future__ import print_function
import torch
import os
import gym
import time
import glob
import argparse
import sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize  # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
from matplotlib import pyplot as plt

def si(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=1, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=3, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()




def discount(x, gamma):
    gs = []
    for t in range(len(x)):
        g = 0
        for i, reward in enumerate(x[t:]):
            g += gamma ** i * reward
        gs.append(g)
    return np.array(gs)


def prepro(img):
    x = img
    x = x[35:195]
    x = x.mean(2)
    x = imresize(x, (80, 80))
    x = x.astype(np.float32)
    x = x / 255.
    x = x.reshape(1, 80, 80)
    return x


class NNPolicy(nn.Module):  # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        print('ahaha',inputs.shape)
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx


class SharedAdam(torch.optim.Adam):  # extend a pytorch_playground optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1  # a "step += 1"  comes later
            super.step(closure)


def cost_func(args, values, logps, actions, rewards):

    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    print(rewards)
    print(rewards[-1])
    print(np_values)
    print(np_values[-1])
    # input()
    rewards[-1] += args.gamma * np_values[-1]
    print(rewards)
    input()
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum()  # encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env)  # make a local (unshared) environment
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)  # seed everything
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)  # a local/unshared model
    state = env.reset()
    state = prepro(state)
    state = torch.tensor(state)  # get first state

    episode_length = 0
    epr = 0
    eploss = 0
    done = True  # bookkeeping


    while info['frames'][0] <= 8e7 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict())  # sync with shared model

        if done:
            hx = torch.zeros(1, 256)
        else:
            hx = hx.detach()

        values = []
        logps = []
        actions = []
        rewards = []

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render:
                env.render()

            state = prepro(state)
            state = torch.tensor(state)
            epr += reward
            reward = np.clip(reward, -1, 1)  # reward

            # done = done or episode_length >= 1e4  # don't playing one ep for too long

            if episode_length >= 1e4:
                done = True
            # info['frames'].add_(1)
            info['frames'] += 1

            if done:  # update shared data
                info['episodes'] += 1
                # interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                if info['episodes'][0] == 1:
                    interp = 1
                else:
                    interp = 1 - args.horizon

                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if done:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)
        if done:
            next_value = torch.zeros(1, 1)
        else:
            next_value = model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


if __name__ == "__main__":

    mp.set_start_method('spawn')  # this must not be in global scope

    args = get_args()
    if args.render:
        args.processes = 1
        args.test = True  # render mode -> test mode w one process
    if args.test:
        args.lr = 0  # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n  # get the action space of this game

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_model = shared_model.share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {
        'run_loss': torch.DoubleTensor([0]).share_memory_(),
        'run_epr': torch.DoubleTensor([0]).share_memory_(),
        'episodes': torch.DoubleTensor([0]).share_memory_(),
        'frames': torch.DoubleTensor([0]).share_memory_(),
    }

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()