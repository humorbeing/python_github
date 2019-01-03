import numpy as np


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


def sort_BGW(point_in):
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

