import gym

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
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        # envs = ShmemVecEnv(envs, context='fork')
        envs = ShmemVecEnv(envs)  # , context='fork')
    return envs
