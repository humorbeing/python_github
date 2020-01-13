import numpy as np
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

env_name = 'Pendulum-v0'
env_name = 'CartPole-v0'
nproc = 8
T = 1010

def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

envs = []
for seed in range(nproc):
    env = make_env(env_name, seed)
    envs.append(env)

envs = SubprocVecEnv(envs)

xt = envs.reset()
print(xt)

# actions = []
# for _ in range(nproc):
#     action = envs.action_space.sample()
#     # print(type(action))
#     actions.append(action)
# actions = np.stack(actions)
# xtp1, rt, done, info = envs.step(actions)
# print(done)
for t in range(T):
    actions = []
    for _ in range(nproc):
        action = envs.action_space.sample()
        # print(type(action))
        actions.append(action)
    actions = np.stack(actions)
    xtp1, rt, done, info = envs.step(actions)
    # print(done)