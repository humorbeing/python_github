import numpy as np
import torch

# from a2c_ppo_acktr import utils
try:
    from .envs import make_vec_envs
except Exception:
    from envs import make_vec_envs
try:
    from .utils_from_pytorch import get_vec_normalize
except Exception:
    from utils_from_pytorch import get_vec_normalize
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



def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, is_limit_action=False):
    # print('start making eva envs')
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, gamma=None)
    # print('end of making')
    norm_envs = get_vec_normalize(eval_envs)
    norm_envs.eval()
    norm_envs.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    # print(obs)
    # ss('haha')
    sum_re = torch.zeros(num_processes, 1)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, = actor_critic.act(
                obs,
                deterministic=True)
        # action = action + 1
        # print(action)
        # Obser reward and next obs
        if is_limit_action:
            obs, reward, done, infos = eval_envs.step(action+1)
        else:
            obs, reward, done, infos = eval_envs.step(action)
        sum_re += reward
        if any(done):
            # print(infos)
            for i in range(len(done)):
                if done[i]:
                    eval_episode_rewards.append(sum_re[i].item())
                    # print(done)
                    # print(sum_re[i])
                    sum_re[i] *= 0



    eval_envs.close()

    log = " Evaluation using {} episodes: mean reward {:.5f}".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards))
    return log
