import numpy as np


SIZE = 4


def policy_pi(s):
    action_distribution = dict()
    action_distribution['up'] = 0.25
    action_distribution['down'] = 0.25
    action_distribution['right'] = 0.25
    action_distribution['left'] = 0.25
    return action_distribution


def grid_p(s, a):
    edge = {
        'up': [0, 1, 2, 3],
        'down': [12, 13, 14, 15],
        'right': [3, 7, 11, 15],
        'left': [0, 4, 8, 12],
    }
    s_prime = -1
    if s in edge[a]:
        s_prime = s
    else:
        if a == 'up':
            s_prime = s - SIZE
        if a == 'down':
            s_prime = s + SIZE
        if a == 'right':
            s_prime = s + 1
        if a == 'left':
            s_prime = s - 1
    probability = 1
    reward = -1
    return probability, s_prime, reward


def iterative_policy_evaluation(pi, p, gamma=1):
    S = [i for i in range(1, 15)]
    V = [0 for _ in range(16)]
    # V_next = []
    actions = ['up', 'down', 'right', 'left']
    for i in range(5):

        V_next = [i for i in V]

        for s in S:
            # update = 0
            action_distribution = pi(s)
            updates = []
            for a in actions:
                action_probability = action_distribution[a]
                transition_probability, s_prime, r = p(s, a)
                update = transition_probability * \
                          (r + gamma * V[s_prime])
                updates.append(update)
            V_next[s] = max(updates)

        V = V_next
        print('iteration:', i)
        grid_V = np.array(V).reshape((4, 4))
        print(grid_V)


iterative_policy_evaluation(policy_pi, grid_p)
