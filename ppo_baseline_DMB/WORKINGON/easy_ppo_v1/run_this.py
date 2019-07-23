import torch
try:
    from .envs import make_vec_envs
except Exception: #ImportError
    from envs import make_vec_envs
try:
    from .model import Policy
except Exception:
    from model import Policy

args_env_name = 'Pong-ram-v0'
args_num_processes = 2
args_seed = 0
args_gamma = 0.99




def main():
    torch.set_num_threads(1)
    envs = make_vec_envs(args_env_name, args_seed, args_num_processes)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})





if __name__ == "__main__":
    main()