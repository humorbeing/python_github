import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr import utils
# from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


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



def main():
    args = get_args()

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")

    # args.env_name = 'Pong-ramNoFrameskip-v4'
    args.env_name = 'Pong-ram-v0'

    args.num_processes = 2

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    # ss('here')
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    print(args.recurrent_policy)
    print(args.clip_param)
    print(args.ppo_epoch)
    print('ccccccccc')
    print(args.num_mini_batch)
    print(args.value_loss_coef)
    print(args.entropy_coef)
    print('dddddddddddd')
    print(args.lr)
    print(args.eps)
    print(args.max_grad_norm)
    ss('in main, after actor_critic')


    args.num_mini_batch = 2
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)



    # ss('out of define ppo')
    args.num_steps = 4
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    # ss('rollouts')
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # print(args.num_env_steps)
    # print()
    # ss('pp')
    sum_re = torch.zeros(args.num_processes, 1)
    # print(sum_re.shape)
    for j in range(num_updates):

        # ss('pp')
        is_any_done = False
        for step in range(args.num_steps):
        # for step in range(50000):
            # print(step)
            # ss('pp')
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # print(value)
            # print(action_log_prob)
            # print(action)
            # ss('runner')
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            sum_re += reward
            # print('- --'*20)
            # print(reward)
            # print(sum_re)
            # print()
            # print(reward.shape)
            if any(done):
                # print(sum_re)
                # print(done)
                # input('hi')
                # is_any_done = True
                for i in range(len(done)):
                    if done[i]:
                        # print(i)
                        # print(*sum_re[i])
                        # print(sum_re[i].item())
                        episode_rewards.append(sum_re[i].item())
                        # print(sum_re[i])
                        sum_re[i] *= 0
                # pass
            # episode_rewards.append(reward.item())

            # ss('make reward')
            # print(infos)
            # ss('runner')

            for info in infos:
                # print(info)
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    print('what env info with episode do?', info.keys())
                    # ss('break')

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            # ss('runner')

        with torch.no_grad():

            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, is_any_done, args.use_proper_time_limits)
        # ss('runner1')
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        # ss('runner1')
        rollouts.after_update()
        # ss('runner2')
        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))
        # print(args.log_interval)
        args.log_interval = 100
        if j % args.log_interval == 0 and len(episode_rewards) > 1:

        # if j % args.log_interval == 0:  # and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n Ent {},V {},A {}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards),
                        dist_entropy, value_loss,
                        action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)
        # is_any_done = False

if __name__ == "__main__":
    main()
