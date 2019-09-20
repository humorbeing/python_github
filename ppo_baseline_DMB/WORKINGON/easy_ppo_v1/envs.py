import gym
import torch
try:
    from .from_init import VecEnvWrapper
except Exception:
    from from_init import VecEnvWrapper
try:
    from .shmem_vec_env import ShmemVecEnv
except Exception:
    from shmem_vec_env import ShmemVecEnv


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
        env.seed(seed + rank)
        return env
    return _thunk


def make_vec_envs(env_name, seed, num_processes):
    # print(num_processes)
    # ss('here')
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        # envs = ShmemVecEnv(envs, context='fork')
        envs = ShmemVecEnv(envs)  # , context='fork')
    envs = VecPyTorch(envs)
    return envs


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

