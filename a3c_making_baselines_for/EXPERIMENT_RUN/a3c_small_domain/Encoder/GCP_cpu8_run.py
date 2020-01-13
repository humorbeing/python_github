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

# from ss import ss
from this_utility import *
from this_models import *
from encoder_model import ENCODER

log_name = 'encoder_rnn'

def get_args():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00,
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=10,
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in A3C (default: 20)')

    #
    parser.add_argument('--env-name', default='Pong-ram-v0',
                        help='environment to train on (default: PongDeterministic-v4)')
    return parser.parse_args()


action_map = {
    0: 2,
    1: 3
}
encoder = ENCODER()
encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=lambda storage, loc: storage))


def train(rank, args, shared_model, optimizer, counter, lock):
    env = gym.make(args.env_name)
    env._max_episode_steps = 100000
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    model = RNN_vae(2, action_map)
    model.train()
    state = env.reset()
    done = True
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            hx = torch.zeros(1, 16)
            cx = torch.zeros(1, 16)
        else:
            hx = hx.data
            cx = cx.data


        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(args.num_steps):
            episode_length += 1
            z = encoder.get_z(state)
            action, hx, cx = model(z, hx, cx)
            entropies.append(model.entropy)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            values.append(model.v)
            log_probs.append(model.log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            z = encoder.get_z(state)
            action, hx, cx = model(z, hx, cx)
            R = model.v.data
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae - args.entropy_coef * entropies[i]
        loss = policy_loss + args.value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

def test(rank, args, shared_model, counter):
    log = Log(log_name)
    env = gym.make(args.env_name)
    env._max_episode_steps = 100000
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    model = RNN_vae(2, action_map)

    model.eval()

    state = env.reset()

    reward_sum = 0
    done = True
    best_reward = -999
    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        # env.render()
        if done:
            model.load_state_dict(shared_model.state_dict())
            hx = torch.zeros(1, 16)
            cx = torch.zeros(1, 16)

        z = encoder.get_z(state)
        action, hx, cx = model(z, hx, cx)
        state, reward, done, _ = env.step(action)

        reward_sum += reward

        if done:
            log_string = "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length)
            log.log(log_string)
            if reward_sum > best_reward:
                best_reward = reward_sum
                save_this_model(model, 'rnn')
            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(5)


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    args = get_args()
    # env = gym.make(args.env_name)
    # env._max_episode_steps = 100000

    shared_model = RNN_vae(2, action_map)

    shared_model = shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args,
                                           shared_model,
                                           optimizer,
                                           counter,
                                           lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
