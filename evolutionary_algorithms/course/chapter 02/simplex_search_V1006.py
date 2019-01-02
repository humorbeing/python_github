from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from matplotlib import cm
# maximizing
# W=worst, G=good, B=best


def function_z(x_in, y_in):
    return (((x_in+50) ** 2 + y_in ** 2) / 4000) - (np.cos(x_in / 6.5) * np.cos(y_in / 7.777)) + 1


def objective_function(point_in):
    return function_z(point_in[0], point_in[1])


maxgen = 500
min_x = -30
max_x = 30
min_y = -30
max_y = 30

plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(min_x, max_x, 0.5)
Y = np.arange(min_y, max_y, 0.5)
X, Y = np.meshgrid(X, Y)
Z = function_z(X, Y)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
ax.set_zlabel('Z')
ax.set_zlim(0, 3.5)
ax.view_init(elev=90, azim=90)
plt.show()
fig.canvas.draw()
# back = fig.canvas.copy_from_bbox(ax.bbox)

p = []
for _ in range(3):
    x_axis = np.random.random() * (max_x - min_x) + min_x
    y_axis = np.random.random() * (max_y - min_y) + min_y
    p.append([x_axis, y_axis])


def sort_bgw(point_in):
    score = []
    bd = dict()
    gd = dict()
    wd = dict()
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
    R = dict()
    M = (G['point'] + B['point']) / 2
    R['point'] = M + (M - W['point'])
    R['score'] = objective_function(R['point'])
    R = check_point(R)
    if R['score'] > B['score']:
        E = dict()
        E['point'] = M + (M - W['point']) * 2
        E['score'] = objective_function(E['point'])
        E = check_point(E)
        if E['score'] > B['score']:
            return E, B, G
        else:
            return R, B, G
    elif R['score'] > G['score']:
        return B, R, G
    else:
        C = dict()
        C['point'] = M + (M - W['point']) * 0.5
        C['score'] = objective_function(C['point'])
        C = check_point(C)
        if C['score'] > W['score']:
            return B, G, C
        else:
            m = dict()
            m['point'] = M
            m['score'] = objective_function(M)
            s = dict()
            s['point'] = (W['point'] + B['point']) / 2
            s['score'] = objective_function(s['point'])
            return B, m, s


def check_point(point_in):
    is_outside = False
    if point_in['point'][0] > max_x:
        is_outside = True
    if point_in['point'][0] < min_x:
        is_outside = True
    if point_in['point'][1] > max_y:
        is_outside = True
    if point_in['point'][1] < min_x:
        is_outside = True
    if is_outside:
        point_in['score'] = -999
        return point_in
    else:
        return point_in


def draw_bgw(B_in, G_in, W_in):
    fig.canvas.flush_events()
    dot1 = ax.scatter(B_in['point'][0], B_in['point'][1], B_in['score'], c='r', marker='o')
    dot2 = ax.scatter(G_in['point'][0], G_in['point'][1], G_in['score'], c='b', marker='o')
    dot3 = ax.scatter(W_in['point'][0], W_in['point'][1], W_in['score'], c='g', marker='o')
    fig.canvas.blit(ax.bbox)
    sleep(1)
    dot1.remove()
    dot2.remove()
    dot3.remove()


B, G, W = sort_bgw(p)
draw_bgw(B, G, W)
for i in range(maxgen):
    print('B=[{}],G=[{}],W=[{}]'.format(round(B['score'], 2), round(G['score'], 2), round(W['score'], 2)))
    B, G, W = rolling(B, G, W)
    draw_bgw(B, G, W)





