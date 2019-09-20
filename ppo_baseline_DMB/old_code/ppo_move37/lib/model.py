import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

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

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        # print(num_inputs)
        # print(num_outputs)
        # ss('in model')
        # self.action_map = action_map
        self.critic = nn.Sequential(
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # self.lstm = nn.LSTMCell(32, 16)
        # self.critic_linear = nn.Linear(16, 1)
        # self.actor_linear = nn.Linear(16, action_space)

        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs)
        )
        # self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def trainer(self, state):
        # state = image_pre_process(state)
        # state = state.astype(np.float32)
        # state = state / 255.0
        # state = torch.from_numpy(state)
        # state = state.unsqueeze(0)
        print(state.shape)

        logit = self.actor(state)
        print(logit)
        prob = F.softmax(logit, dim=1)
        selected = prob.multinomial(num_samples=1).data
        print(selected)
        actions = selected.numpy()
        print(actions)
        log_prob_all = torch.log(prob)
        print(log_prob_all)
        self.entropy = -(log_prob_all * prob).sum(1, keepdim=True)
        print(self.entropy)
        self.log_prob = log_prob_all.gather(1, selected)
        print(self.log_prob)
        self.v = self.critic(state)
        print(self.v)
        return actions


