import gym
env_name = 'Pendulum-v0'

env = gym.make(env_name)
state = env.reset()
while True:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)


