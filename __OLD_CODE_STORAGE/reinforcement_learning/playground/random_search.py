import gym
import numpy as np

env = gym.make('CartPole-v1')
# env = env.unwrapped

def run_episode(env_in, parameters_in):
    observation = env_in.reset()
    totalreward = 0
    for _ in range(500):
        action = 0 if np.matmul(parameters_in, observation) < 0 else 1
        observation, reward, done, info = env_in.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


best_params = None


def train():
    global best_params
    best_reward = 0
    for _ in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > best_reward:
            best_reward = reward
            best_params = parameters
            if reward == 500:
                break


def random_policy_pi(state_in):
    if np.matmul(best_params, state_in) < 0:
        return 0
    else:
        return 1


def play(pi):
    total_reward = 0
    while True:
        observation = env.reset()
        while True:
            action = pi(observation)
            observation, reward, done, info = env.step(action)
            env.render()
            total_reward += reward


train()
play(random_policy_pi)