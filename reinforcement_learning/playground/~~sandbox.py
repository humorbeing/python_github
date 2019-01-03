from itertools import count
import numpy as np
import gym
# import tensorflow_playground as tf
# print(np.random.rand(4)*2-1)
# env = gym.make('CartPole-v0')
# print(env)
# env = env.unwrapped
# print(env)
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
# print(5 % 2)
# print(np.random.choice(500, 2))
# for i in count(1):
#     print(i)


state = np.random.rand(4)
# state = state[None, :]
state = [state]
print(state)
first, *l, last = [1,2,3,4]
print(l)

