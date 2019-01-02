import gym
import numpy as np

env = gym.make('CartPole-v1')
# env.reset()
# env.render()


def run_episode(env, parameters):
    observation = env.reset()
    env.render()
    totalreward = 0
    for _ in range(50000):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            print('end', totalreward)
            break
    return totalreward


bestparams = None
bestreward = 0
for _ in range(10000):
    # print('1')
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        # if reward == 200:
        #     break