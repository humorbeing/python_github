import gym
import numpy as np
env = gym.make('CartPole-v2')


def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def show_run(env, parameters, epoch=10):
    for _ in range(epoch):
        observation = env.reset()
        score = 0
        while True:
            env.render()
            action = 0 if np.matmul(parameters, observation) < 0 else 1
            observation, reward, done, info = env.step(action)
            score += reward
            print(score)
            if done:
                break


for _ in range(10):
    bestparams = None
    bestreward = 0

    for _ in range(100000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 50000:
                break

    show_run(env, bestparams, epoch=3)
