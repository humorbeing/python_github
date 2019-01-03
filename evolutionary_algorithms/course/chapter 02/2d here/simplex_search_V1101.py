from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import threading


def objective_function(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    return (((x_in + 50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


min_x = -30
max_x = 30
min_y = -30
max_y = 30
w = 1/4
reset = w * 4

plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
X = np.arange(min_x-2, max_x+2, 0.1)
Y = np.arange(min_y-2, max_y+2, 0.1)
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
Z = objective_function([X, Y])
# ax.contourf(X, Y, Z, cmap=cm.jet)
ax.contourf(X, Y, Z, cmap=cm.coolwarm)

blue_dot, = ax.plot([], [], 'bo', ms=6)
blue_dot2, = ax.plot([], [], 'bo', ms=14)
red_dot, = ax.plot([], [], 'ro', ms=10)
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)


maxgen = 20
B = {'point': [max_x+99, max_y+99]}
G = {'point': [max_x+99, max_y+99]}
W = {'point': [max_x+99, max_y+99]}

def sort_bgw(a, b, c):
    if a['score'] >= b['score'] >= c['score']:
        return a, b, c
    elif a['score'] >= c['score'] >= b['score']:
        return a, c, b
    elif b['score'] >= a['score'] >= c['score']:
        return b, a, c
    elif b['score'] >= c['score'] >= a['score']:
        return b, c, a
    elif c['score'] >= a['score'] >= b['score']:
        return c, a, b
    else:
        return c, b, a


def initialization():
    p = []
    for _ in range(3):
        dp = dict()
        x_axis = np.random.random() * (max_x - min_x) + min_x
        y_axis = np.random.random() * (max_y - min_y) + min_y
        dp['point'] = np.array([x_axis, y_axis])
        dp['score'] = objective_function(dp['point'])
        p.append(dp)
    a, b, c = sort_bgw(p[0], p[1], p[2])
    return a, b, c


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


def survive():
    global B, G, W
    while True:
        B, G, W = initialization()
        for _ in range(maxgen):
            B, G, W = rolling(B, G, W)
            B, G, W = sort_bgw(B, G, W)
            time.sleep(w)

        B = {'point': [max_x + 99, max_y + 99]}
        G = {'point': [max_x + 99, max_y + 99]}
        W = {'point': [max_x + 99, max_y + 99]}
        time.sleep(reset)

t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    blue_dot2.set_data(B['point'][0], B['point'][1])
    blue_dot.set_data(G['point'][0], G['point'][1])
    red_dot.set_data(W['point'][0], W['point'][1])
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




