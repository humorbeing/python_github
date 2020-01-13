import numpy as np
import copy

move = {
    'up': (0, 1),
    'down': (0, -1),
    'right': (1, 0),
    'left': (-1, 0),
}

states = []
for x in range(1, 5):
    for y in range(1,4):
        if x == 2 and y ==2:
            pass
        else:
            states.append([x, y])

action_reward = -0.02
# action_reward = 0
gamma = 0.99
def init_pi():
    xs = dict()
    for x in range(1, 5):
        ys = dict()
        for y in range(1, 4):
            action_distribution = dict()
            action_distribution['up'] = 0.25
            action_distribution['down'] = 0.25
            action_distribution['right'] = 0.25
            action_distribution['left'] = 0.25
            ys[y] = action_distribution
        xs[x] = ys
    return xs


POLICY = init_pi()


def init_Q():
    xs = dict()
    for x in range(1, 5):
        ys = dict()
        for y in range(1, 4):
            action_distribution = dict()
            action_distribution['up'] = 0.0
            action_distribution['down'] = 0.0
            action_distribution['right'] = 0.0
            action_distribution['left'] = 0.0
            ys[y] = action_distribution
        xs[x] = ys
    return xs


Q = init_Q()


def init_V():
    xs = dict()
    for x in range(1, 5):
        ys = dict()
        for y in range(1, 4):
            ys[y] = 0
        xs[x] = ys
    return xs


V = init_V()


def pi(s):
    return POLICY[s[0]][s[1]]


def move_to(a, blow_to_prob=0.1):
    move_to_prob = 1 - 2 * blow_to_prob
    moves = dict()
    if a == 'up' or a == 'down':
        moves[a] = move_to_prob
        moves['left'] = blow_to_prob
        moves['right'] = blow_to_prob
    else:
        moves[a] = move_to_prob
        moves['up'] = blow_to_prob
        moves['down'] = blow_to_prob
    return moves


def correct_move(f, m):
    if m[0] < 1 or m[0] > 4:
        return f
    elif m[1] < 1 or m[1] > 3:
        return f
    elif m[0] == 2 and m[1] == 2:
        return f
    else:
        return m


def MDP_world_p(s, a):
    # state_and_prob = []
    states = []
    if s[0] == 4 and s[1] == 3:
        return [[[[5, 0], 1], 1]]
    elif s[0] == 4 and s[1] == 2:
        return [[[[5, 0], 1], -1]]
    else:
        moves_prob = move_to(a)
        for m in moves_prob:
            moving_to = np.add(s, move[m])
            # print('moving to:', a, '=:=', s, '->', m, ':', moving_to, ' |', moves_prob[m])
            end_up = correct_move(s, moving_to)
            # print('   end up:', a, '=:=', s, '->', m, ':', end_up, ' |', moves_prob[m])
            # print('- ' * 20)
            states.append([[np.array(end_up), moves_prob[m]], action_reward])
        return states


def policy_evaluation(pi_in, Prob, V_in, k, gamma=1):
    # v_out = copy.deepcopy(V_in)
    v_now = V_in
    for iter in range(k):
        v_next = copy.deepcopy(v_now)
        for s in states:
            actions = pi_in(s)
            updates = 0
            for a in actions:
                a_prob = actions[a]
                if a_prob == 0:
                    pass
                else:
                    # print(a_prob)
                    moves = Prob(s, a)
                    # print(moves)
                    add_s_prime = 0
                    for re in moves:
                        reward = re[1]
                        s_prime = re[0][0]
                        s_prob = re[0][1]
                        # print(reward)
                        # print(s_prime)
                        # print(s_prob)
                        if s_prime[0] == 5:
                            update = reward
                        else:
                            update = reward + gamma * v_now[s_prime[0]][s_prime[1]]
                        update = update * s_prob
                        # update = update * a_prob
                        add_s_prime += update
                    updates += add_s_prime * a_prob
            v_next[s[0]][s[1]] = updates

        v_now = v_next

    return v_next


def policy_iteration(pi_in, Prob, V_in, gamma=1):
    global POLICY
    for s in states:
        actions = pi_in(s)
        values = []
        acts = []
        # print()
        # print('on:', s)
        for a in actions:
            acts.append(a)
            # a_prob = actions[a]
            # print(a, ':', a_prob)

                # print(a_prob)
            moves = Prob(s, a)
            # print(moves)
            add_s_prime = 0
            for re in moves:
                reward = re[1]
                s_prime = re[0][0]
                s_prob = re[0][1]
                # print(reward)
                # print(s_prime)
                # print(s_prob)
                if s_prime[0] == 5:
                    update = reward
                else:
                    update = reward + gamma * V_in[s_prime[0]][s_prime[1]]
                update = update * s_prob
                # update = update * a_prob
                add_s_prime += update
                add_s_prime = round(add_s_prime, 3)
            values.append(add_s_prime)
        # print(values)
        max_v = max(values)
        num = 0
        for v in values:
            if max_v == v:
                num += 1
        # print(num)
        probability = 1 / num
        for a in POLICY[s[0]][s[1]]:
            POLICY[s[0]][s[1]][a] = 0.0
        for i, v in enumerate(values):
            # print('i:', i)
            # print('acts:', acts)
            # print('v:', v)
            # print('max_v:', max_v)
            # print()

            if v == max_v:
                POLICY[s[0]][s[1]][acts[i]] = probability
    pass

def show_v():

    for y in range(1, 4):
        rev_y = 4 - y
        for x in range(1, 5):
            print(' [{}]'.format(round(V[x][rev_y], 3)).rjust(10), end='')
        print()


def show_p():
    pr = ['U', 'D', 'R', 'L']

    for y in range(1, 4):
        rev_y = 4 - y
        for x in range(1, 5):
            val = []
            for a in POLICY[x][rev_y]:
                val.append(POLICY[x][rev_y][a])
            max_v = max(val)
            output = ''
            for i, v in enumerate(val):
                if max_v == v:
                    output += pr[i]
                else:
                    output += '-'
            print(' [  '+output+'  ]', end='')
        print()


# Q[1][3]['left'] = 5
# print(Q)
init_s = np.array([1, 1])
move_on = {
    0: 'up',
    1: 'down',
    2: 'right',
    3: 'left',
}
def get_an_action(s):
    choose = np.random.randint(0, 4)
    return move_on[choose]


def try_this(s, a):
    states = MDP_world_p(s, a)
    # print(states)
    if len(states) == 3:
        chance = np.random.random()
        # print(chance)
        if chance < 0.8:
            # print('0.8')
            # print(states[0])
            # print(states[0][0][0])
            # print(states[0][1])
            return states[0][0][0], states[0][1]
        elif chance < 0.9:
            # print('1st 0.1----------------------------------------------')
            # print(states[1])
            # print(states[1][0][0])
            # print(states[1][1])
            return states[1][0][0], states[1][1]
        else:
            # print('2nd 0.1 [ ] [] [] [ [ ][] [] [] [] ')
            # print(states[2])
            # print(states[2][0][0])
            # print(states[2][1])
            return states[2][0][0], states[2][1]
    else:
        return states[0][0][0], states[0][1]


def maxQ(s):
    # print(Q[s[0]][s[1]])
    m = max(Q[s[0]][s[1]], key=Q[s[0]][s[1]].get)
    ret = Q[s[0]][s[1]][m]
    # print(ret)
    return ret

def tabular_Q_learning(k, alpha=0.5):
    s = init_s
    for iter in range(k):
        a = get_an_action(s)
        # print(a)
        s_prime, reward = try_this(s, a)
        if s_prime[0] == 5:  # terminal
            target = reward
            # print()
            # print(' E N D ' * 10)
            # print()
            s_prime = init_s
        else:
            target = reward + gamma * maxQ(s_prime)

        Q[s[0]][s[1]][a] = Q[s[0]][s[1]][a] + alpha * (target - Q[s[0]][s[1]][a])
        # print('state:{}, action:{:>6}, s_:{}, reward:{}, target:{}'.format(s, a, s_prime, reward, target))
        s = s_prime


def show_q():
    for x in range(1, 5):
        print('ON Column: {}'.format(x))
        for y in range(1, 4):
            if x == 2 and y == 2:
                pass
            else:
                print('---ON State[{} {}]'.format(x,y))
                for a in Q[x][y]:
                    print('---|---Action[{:>7}]: {}'.format(a, Q[x][y][a]))

        print()


tabular_Q_learning(100)
show_q()
tabular_Q_learning(100)
show_q()
tabular_Q_learning(100)
show_q()