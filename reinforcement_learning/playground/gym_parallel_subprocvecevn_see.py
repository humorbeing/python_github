import numpy as np
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

env_name = 'Pendulum-v0'
num_process = 8

def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

envs = []
for seed in range(num_process):
    env = make_env(env_name, seed)
    envs.append(env)
envs = SubprocVecEnv(envs)

states = envs.reset()
while True:
    actions = []
    for _ in range(num_process):
        action = envs.action_space.sample()
        actions.append(action)
    actions = np.stack(actions)
    states, rewards, dones, infos = envs.step(actions)
