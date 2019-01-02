import numpy as np

# maximizing
# W=worst, G=good, B=best


def objective_function(point_in):
    return point_in[0]+point_in[1]


min_x = -30
max_x = 30
min_y = -30
max_y = 30

p = []
for _ in range(3):
    x_axis = np.random.random() * (max_x - min_x) + min_x
    y_axis = np.random.random() * (max_y - min_y) + min_y
    p.append([x_axis, y_axis])

p = [[1, 2], [2, 4], [5, 6]]


def sort_bgw(point_in):
    score = []
    bd = {}
    gd = {}
    wd = {}
    for i in range(3):
        score.append(objective_function(point_in[i]))
    max_n = np.argmax(score)
    bd['point'] = point_in[max_n]
    bd['score'] = score[max_n]
    score = np.delete(score, max_n, 0)
    point_in = np.delete(point_in, max_n, 0)
    max_n = np.argmax(score)
    if max_n == 0:
        gd['point'] = point_in[0]
        gd['score'] = score[0]
        wd['point'] = point_in[1]
        wd['score'] = score[1]
    else:
        gd['point'] = point_in[1]
        gd['score'] = score[1]
        wd['point'] = point_in[0]
        wd['score'] = score[0]
    return bd, gd, wd


def rolling(B, G, W):
    R = {}
    E = {}
    C = {}
    M = (G['point'] + B['point']) / 2
    R['point'] = M + (M - W['point'])
    R['score'] = objective_function(R['point'])

    if R['score'] > B['score']:
        E['point'] = M + (M - W['point']) * 2
        E['score'] = objective_function(E['point'])
        if E['score'] > B['score']:
            return E, B, G
        else:
            return R, B, G
    elif R['score'] > G['score']:
        return B, R, G
    else:
        C['point'] = M + (M - W['point']) * 0.5
        C['score'] = objective_function(C['point'])
        if C['score'] > W['score']:
            return B, G, C
        else:
            m = {}
            m['point'] = M
            m['score'] = objective_function(M)
            s = {}
            s['point'] = (W['point'] + B['point']) / 2
            s['score'] = objective_function(s['point'])
            return B, m, s

