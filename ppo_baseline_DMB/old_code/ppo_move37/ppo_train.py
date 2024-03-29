# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import argparse
import math
import os
import random
import gym
import numpy as np

# Follow instructions here to install https://github.com/openai/roboschool
# import roboschool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
# from tensorboardX import SummaryWriter

from lib.common import mkdir
from lib.model import ActorCritic
from lib.multiprocessing_env import SubprocVecEnv

def ss(s=''):
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    # print('        >>>>>>>>>>>>>>>>>>>>                <<<<<<<<<<<<<<<<<<<<        ')
    print(s)
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    import sys
    sys.exit()

NUM_ENVS            = 3
ENV_ID              = "RoboschoolHalfCheetah-v1"
ENV_ID              = "Pong-ram-v0"
HIDDEN_SIZE         = 256
LEARNING_RATE       = 1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 10
MINI_BATCH_SIZE     = 4
PPO_EPOCHS          = 10
TEST_EPOCHS         = 10
NUM_TESTS           = 10
TARGET_REWARD       = 2500


def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = gym.make(ENV_ID)
        return env
    return _thunk

    
def test_env(env, model, device, deterministic=True):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    print('here '*20)
    print(values)

    values = values + [next_value]
    print(values)
    print(masks)
    print(rewards)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        print(step)
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    print(rewards)
    return returns


def ppo_iter(states, actions, log_probs, returns, advantage):

    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        print(' o '*50)
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        print(batch_size)
        print(MINI_BATCH_SIZE)
        print(rand_ids)
        # rand_ids = np.array([29, 28,27,26 ])
        # rand_ids = np.array([0, 0, 0, 0])
        print(rand_ids)
        # ss('yo')
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
    # print(aa)
    # ss('s')
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        a = 0
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            # print(action, a)
            # a += 1
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            
            count_steps += 1
    
    # writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    # writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    # writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    # writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    # writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    # writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)

action_map = {
    0: 2,
    1: 3
}

if __name__ == "__main__":
    aa = 5
    mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    args = parser.parse_args()
    # writer = SummaryWriter(comment="ppo_" + args.name)

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    print('Device:', device)
    # ss('here')
    # Prepare environments
    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_ID)
    num_inputs  = envs.observation_space.shape[0]
    # print(envs.observation_space)
    # print(envs.action_space.n)
    num_outputs = envs.action_space.n
    # num_outputs = 2
    # ss('here')
    model = ActorCritic(num_inputs, num_outputs).to(device)
    # ss('model define')
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    frame_idx  = 0
    train_epoch = 0
    best_reward = None

    state = envs.reset()
    early_stop = False

    while not early_stop:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in range(PPO_STEPS):
            # state = torch.FloatTensor(state).to(device)
            # print(state.shape)
            state = state.astype(np.float32)
            state = state / 255.0
            state = torch.from_numpy(state)
            action = model.trainer(state)
            # ss('here')

            # action = dist.sample()
            # each state, reward, done is a list of results from each parallel environment
            next_state, reward, done, _ = envs.step(action)
            # print('OO '*30)
            # print(next_state.shape)
            # print(reward.shape)
            # print(done)
            # ss('here')
            # log_prob = dist.log_prob(action)
            # print(done)
            # print(1-done)
            # ss('h')
            log_probs.append(model.log_prob)
            values.append(model.v)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            # action = action.astype(np.float32)
            # state = state / 255.0
            action = torch.from_numpy(action)
            actions.append(action)
            
            state = next_state
            frame_idx += 1

        # ss('out')
        # next_state = torch.FloatTensor(next_state).to(device)
        state = next_state
        state = state.astype(np.float32)
        state = state / 255.0
        state = torch.from_numpy(state)
        _ = model.trainer(state)
        next_value = model.v
        print(next_value)

        returns = compute_gae(next_value, rewards, masks, values)
        # ss('gae')
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        advantage = normalize(advantage)
        
        ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
        ss('update')
        train_epoch += 1

        if train_epoch % TEST_EPOCHS == 0:
            test_reward = np.mean([test_env(env, model, device) for _ in range(NUM_TESTS)])
            writer.add_scalar("test_rewards", test_reward, frame_idx)
            print('Frame %s. reward: %s' % (frame_idx, test_reward))
            # Save a checkpoint every time we achieve a best reward
            if best_reward is None or best_reward < test_reward:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
                    fname = os.path.join('.', 'checkpoints', name)
                    torch.save(model.state_dict(), fname)
                best_reward = test_reward
            if test_reward > TARGET_REWARD: early_stop = True
