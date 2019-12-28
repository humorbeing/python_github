import numpy as np
import torch

# from a2c_ppo_acktr import utils
try:
    from .envs import make_vec_envs
except Exception:
    from envs import make_vec_envs

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



def evaluate(actor_critic, env_name, seed, num_processes):

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes)



    eval_episode_rewards = []

    obs = eval_envs.reset()
    sum_re = torch.zeros(num_processes, 1)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, = actor_critic.act(
                obs,
                deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        sum_re += reward
        if any(done):

            for i in range(len(done)):
                if done[i]:
                    eval_episode_rewards.append(sum_re[i].item())
                    sum_re[i] *= 0



    eval_envs.close()

    log = " Evaluation using {} episodes: mean reward {:.5f}".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards))
    return log