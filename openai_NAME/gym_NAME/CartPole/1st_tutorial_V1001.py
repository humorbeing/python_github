import gym
import numpy as np
env = gym.make('CartPole-v0')
parameters = np.random.rand(4) * 2 - 1  # [-1, 1]


def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


for _ in range(10):
    print(run_episode(env, parameters))
pass
