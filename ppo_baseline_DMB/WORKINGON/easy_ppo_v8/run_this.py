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
try:
    from .util_this import Log
except Exception:
    from util_this import Log
try:
    from .evaluation import evaluate
except Exception:
    from evaluation import evaluate
try:
    from .utils_from_pytorch import get_vec_normalize
except Exception:
    from utils_from_pytorch import get_vec_normalize

try:
    from .utils_from_pytorch import update_linear_schedule
except Exception:
    from utils_from_pytorch import update_linear_schedule

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

log_name = 'ppo_PongD43act_64RNN_gpu'
args_env_name = 'Pong-ramDeterministic-v4'
args_num_processes = 16  # how many envs running, default: 10
args_seed = 1
args_gamma = 0.99
args_num_mini_batch = 16  # how many batchs to train, default: 32
args_clip_param = 0.2
args_ppo_epoch = 4  # in training weight after collection, how many epoch to train agent, default: 4
args_value_loss_coef = 0.5
args_entropy_coef = 0.01

args_lr = 0.0007
args_eps = 1e-5
args_max_grad_norm = 0.5
args_num_steps = 64  # in gathering rollouts, how many steps forward, default: 4
args_num_env_steps = 5e6  # total training steps
args_log_interval = 200
args_eval_interval = 200
args_use_gae = True
args_gae_lambda = 0.95
args_use_clipped_value_loss = True
args_recurrent_policy = True
# args_recurrent_policy = False
# new set
args_num_processes = 4
args_num_mini_batch = 2
args_num_steps = 2
args_use_linear_lr_decay = True

is_limit_action = True
# is_limit_action = False
# args_cuda = True
args_cuda = False

def main():



    torch.manual_seed(args_seed)
    torch.cuda.manual_seed_all(args_seed)

    device = torch.device("cuda:0" if args_cuda else "cpu")

    train_log = Log(log_name+'_train_log')
    evl_log = Log(log_name+'_evaluation_log')
    torch.set_num_threads(1)
    envs = make_vec_envs(
        args_env_name,
        args_seed,
        args_num_processes,
        device,
        gamma=args_gamma)

    # norm_envs = get_vec_normalize(envs)
    # norm_envs = envs
    # norm_envs.eval()
    # norm_envs.ob_rms = 1
    # print(envs.ob_rms)
    # ss('hi')
    if is_limit_action:
        envs.action_space.n = 3
    print('Number of Actions:', envs.action_space.n)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args_recurrent_policy})
    actor_critic.to(device)
    # print(actor_critic.is_recurrent)
    # print(actor_critic.gru)
    # ss('hi')

    agent = PPO(
        actor_critic,
        args_clip_param,
        args_ppo_epoch,
        args_num_mini_batch,
        args_value_loss_coef,
        args_entropy_coef,
        lr=args_lr,
        eps=args_eps,
        max_grad_norm=args_max_grad_norm,
        use_clipped_value_loss=args_use_clipped_value_loss)

    rollouts = RolloutStorage(
        args_num_steps,
        args_num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    # print(obs)
    # ss('i am over it')
    num_updates = int(
        args_num_env_steps) // args_num_steps // args_num_processes

    episode_rewards = deque(maxlen=10)
    start = time.time()
    sum_re = torch.zeros(args_num_processes, 1)

    for j in range(num_updates):

        if args_use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(
                agent.optimizer, j, num_updates,
                args_lr)

        for step in range(args_num_steps):

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # ss('dissecting actor critic. act')
            # print(action)
            # print()
            # action = action + 1
            # print(action)
            # ss('hoiohasdfhioas')
            if is_limit_action:
                obs, reward, done, infos = envs.step(action+1)
            else:
                obs, reward, done, infos = envs.step(action)
            sum_re += reward

            if any(done):

                for i in range(len(done)):
                    if done[i]:
                        episode_rewards.append(sum_re[i].item())
                        # print(done)
                        # print(sum_re[i])
                        sum_re[i] *= 0
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():

            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value,
                                 args_gamma,
                                 args_use_gae,
                                 args_gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args_log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args_num_processes * args_num_steps
            end = time.time()
            logstring = "E {}, N_steps {}, FPS {} mean/median" \
                        " {:.1f}/{:.1f}, min/max {:.1f}/{:.1f}" \
                        " Entropy {:.5f},V {:.5f},Action {:.5f}".format(
                j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards),
                            dist_entropy, value_loss,
                            action_loss)
            # print(logstring)
            train_log.log(logstring)
        # if True:
        if (args_eval_interval is not None and len(episode_rewards) > 1
                and j % args_eval_interval == 0):
            total_num_steps = (j + 1) * args_num_processes * args_num_steps
            ob_rms = get_vec_normalize(envs).ob_rms
            ev_result = evaluate(actor_critic, ob_rms, args_env_name, args_seed,
                     args_num_processes, device, is_limit_action=is_limit_action)
            ev_log_string = 'steps:'+str(total_num_steps)+'. '+ev_result
            evl_log.log(ev_log_string)


if __name__ == "__main__":
    main()