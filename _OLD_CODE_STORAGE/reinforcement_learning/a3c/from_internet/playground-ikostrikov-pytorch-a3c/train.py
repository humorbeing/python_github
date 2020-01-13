import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from envs import create_atari_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    # print(env.observation_space.shape)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            # env.render()
            # print('hi')
            episode_length += 1
            # print(state.shape)
            # print(state.unsqueeze(0).shape)
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),
                                            (hx, cx)))

            prob = F.softmax(logit)
            # print(prob)
            log_prob = F.log_softmax(logit)
            # print(log_prob)
            # print(torch.log(prob))
            # input('w')
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            # print(entropy)
            # input('w')
            entropies.append(entropy)

            # prob = torch.Tensor(
            #     [
            #         [0.05, 0.05, 0.9],
            #         [0.04, 0.06, 0.9]
            #     ]
            # )
            # print(prob)
            # a = prob.multinomial(num_samples=1)
            # print(a)
            # print(a.data)
            # log_prob = torch.log(prob)
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))
            # print(action.shape)
            # print(log_prob.shape)
            # dice = Categorical(prob)
            # sample_number = dice.sample()
            # print(sample_number.dtype())
            # sample_number += 1
            # sample_number = torch.Tensor([1, 0], dtype=torch.ShortTensor)
            # sample_number = sample_number.
            # print(sample_number.shape)
            # input('w')
            # log_pi = dice.log_prob(sample_number)
            # log_pi = dice.log_prob(action)
            # print(log_prob)
            # print(log_pi.shape)
            # input('w')
            # sample_number = sample_number.item()

            state, reward, done, _ = env.step(action.numpy())
            if episode_length >= args.max_episode_length:
                done = True


            reward = max(min(reward, 1), -1)

            # with lock:
            #     counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()
            # print(value.shape)
            # input('w')
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        print(rewards)
        input('w')
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
