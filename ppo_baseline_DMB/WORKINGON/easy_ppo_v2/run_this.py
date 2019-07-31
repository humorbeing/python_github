from collections import deque
import time
import torch
import numpy as np
try:
    from .envs import make_vec_envs
except Exception: #ImportError
    from envs import make_vec_envs
try:
    from .model import Policy
except Exception:
    from model import Policy
try:
    from .ppo import PPO
except Exception:
    from ppo import PPO
try:
    from .storage import RolloutStorage
except Exception:
    from storage import RolloutStorage


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

args_env_name = 'Pong-ram-v0'
args_num_processes = 2  # how many envs running, default: 10
args_seed = 1
args_gamma = 0.99
args_num_mini_batch = 2  # how many batchs to train, default: 32
args_clip_param = 0.2
args_ppo_epoch = 4  # in training weight after collection, how many epoch to train agent, default: 4
args_value_loss_coef = 0.5
args_entropy_coef = 0.01
args_lr = 0.0007
args_eps = 1e-5
args_max_grad_norm = 0.5
args_num_steps = 4  # in gathering rollouts, how many steps forward, default: 4
args_num_env_steps = 1e8  # total training steps
args_log_interval = 100

def main():
    torch.set_num_threads(1)
    envs = make_vec_envs(
        args_env_name,
        args_seed,
        args_num_processes)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)

    agent = PPO(
        actor_critic,
        args_clip_param,
        args_ppo_epoch,
        args_num_mini_batch,
        args_value_loss_coef,
        args_entropy_coef,
        lr=args_lr,
        eps=args_eps,
        max_grad_norm=args_max_grad_norm)
    rollouts = RolloutStorage(
        args_num_steps,
        args_num_processes,
        envs.observation_space.shape)


    obs = envs.reset()
    np.copyto(rollouts.obs[0], obs)

    num_updates = int(
        args_num_env_steps) // args_num_steps // args_num_processes

    episode_rewards = deque(maxlen=10)
    start = time.time()
    sum_re = np.zeros(shape=(args_num_processes, 1))

    for j in range(num_updates):

        for step in range(args_num_steps):
            with torch.no_grad():
                value, action, action_log_prob\
                    = actor_critic.act(rollouts.obs[step])
            print(action)
            ss('hoho')
            obs, reward, done, infos = envs.step(action)
            sum_re += reward

            if any(done):

                for i in range(len(done)):
                    if done[i]:
                        episode_rewards.append(sum_re[i].item())
                        sum_re[i] *= 0
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, action,
                            action_log_prob,
                            value, reward,
                            masks, bad_masks)
        with torch.no_grad():

            next_value = actor_critic.get_value(
                rollouts.obs[-1])

        rollouts.compute_returns(
            next_value,
            args_gamma)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args_log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args_num_processes * args_num_steps
            end = time.time()
            print(
                "E {}, N_steps {}, FPS {}"
                " mean/median {:.1f}/{:.1f}, min/max {:.1f}/{:.1f} Ent {:.4f},V {:.4f},A {:.4f}"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards),
                            dist_entropy, value_loss,
                            action_loss))
if __name__ == "__main__":
    main()