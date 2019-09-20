import gym
import torch
import numpy as np
try:
    from .in_here.vec_env import VecEnvWrapper
except Exception:
    from in_here.vec_env import VecEnvWrapper
try:
    from .shmem_vec_env import ShmemVecEnv
except Exception:
    from shmem_vec_env import ShmemVecEnv

from in_here.vec_normalize import VecNormalize as VecNormalize_

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


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env = env.env
        # print(seed + rank)
        env.seed(seed + rank)
        return env
    return _thunk


def make_vec_envs(env_name, seed, num_processes, is_evl=False):
    # print(num_processes)
    # ss('here')
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]
    # print('sdjklfsdkf'*2)
    if len(envs) > 1:
        # envs = ShmemVecEnv(envs, context='fork')
        envs = ShmemVecEnv(envs)  # , context='fork')
    if is_evl:
        envs = VecNormalize(envs, ret=False)
    else:
        envs = VecNormalize(envs, gamma=0.99)
    # ss('yo stopped at vec norm')
    # print('yoyoyo',envs.ob_rms)
    envs = VecPyTorch(envs)
    # e = get_vec_normalize(envs)
    # print(e.ob_rms)
    # ss('nono')
    return envs

# def get_vec_normalize(venv):
#     if isinstance(venv, VecNormalize):
#         return venv
#     elif hasattr(venv, 'venv'):
#         return get_vec_normalize(venv.venv)
#
#     return None
# Checks whether done was caused my timit limits or not


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float()
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        # print(obs)
        # ss('we are in here, before i am over it')

        if self.ob_rms:
            # print('in here')
            if self.training and update:
                # print('in here')
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            # print(obs)
            # print('>>  '*20)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False