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

p = np.array(p)


def sort_BGW(point_in):
    # point_in = pp
    score = []
    for i in range(3):
        score.append(objective_function(point_in[i]))

    score = np.array(score)
    # print(score)
    max_n = np.argmax(score)
    b = point_in[max_n]
    score = np.delete(score, max_n, 0)
    point_in = np.delete(point_in, max_n, 0)
    max_n = np.argmax(score)

    if max_n == 0:
        g = point_in[0]
        w = point_in[1]
    else:
        g = point_in[1]
        w = point_in[0]

    return b, g, w


B, G, W = sort_BGW(p)
print(B, G, W)
