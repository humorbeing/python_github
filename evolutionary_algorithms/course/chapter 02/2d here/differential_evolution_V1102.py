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
w = 1/50
reset = w * 6

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
ax.contourf(X, Y, Z, cmap=cm.coolwarm)  # *****
# ax.contourf(X, Y, Z, cmap=cm.gray)  # ***
# ax.contourf(X, Y, Z, cmap=cm.Greys)
# ax.contourf(X, Y, Z, cmap=cm.BuPu)
# ax.contourf(X, Y, Z, cmap=cm.viridis)  # *****
# ax.contourf(X, Y, Z, cmap=cm.plasma)
# ax.contourf(X, Y, Z, cmap=cm.inferno)  # ***
# ax.contourf(X, Y, Z, cmap=cm.magma)
u_dot, = ax.plot([], [], 'r*', ms=25)
blue_dot, = ax.plot([], [], 'ko', ms=10)
red_dot, = ax.plot([], [], 'ro', ms=15)
green_dot, = ax.plot([], [], 'go', ms=10)

fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)


popsize = 20
maxgen = 20
F = 0.3
K = 0.5

generation = np.array([[max_x+99, 0], [max_x+99, 0]])
candidate = [max_x+99, 0]
Cr = 0.5
j_rand = [0]
x_best = 0
ev_x = [max_x+99, 0]
ev_v = [max_x+99, 0]
ev_u = [max_x+99, 0]


def initialization():
    global generation
    p = []
    for _ in range(popsize):
        x_axis = np.random.random() * (max_x - min_x) + min_x
        y_axis = np.random.random() * (max_y - min_y) + min_y
        p.append([x_axis, y_axis])
    generation = np.array(p)


def x_best_():
    score = []
    for i in generation:
        score.append(objective_function([i[0], i[1]]))
    return np.argmax(score)


def sampling(sample_number, haves, best=-1):

    check_samples = [haves]
    if best != -1:
        check_samples.append(best)
    samples = [0 for _ in range(sample_number)]
    for i in range(sample_number):
        samples[i] = np.random.randint(0, popsize)
        while samples[i] in check_samples:
            samples[i] = np.random.randint(0, popsize)
        check_samples.append(samples[0])
    return np.array(samples)


def r_1_v(haves):
    points = sampling(3, haves)
    v = generation[points[0]] + F * (generation[points[1]] - generation[points[2]])
    return v


def r_2_v(haves):
    points = sampling(5, haves)
    v = generation[points[0]] + F * (generation[points[1]] - generation[points[2]])
    v += K * (generation[points[3]] + generation[points[4]])
    return v


def b_1_v(haves):
    best = x_best_()
    points = sampling(2, haves, best)
    v = generation[best] + F * (generation[points[0]] - generation[points[1]])
    return v


def b_2_v(haves):
    best = x_best_()
    points = sampling(4, haves, best)
    v = generation[best] + F * (generation[points[0]] - generation[points[1]])
    v += K * (generation[points[2]] - generation[points[3]])
    return v


def c_2_v(haves):
    points = sampling(3, haves)
    v = generation[haves] + F * (generation[points[0]] - generation[points[1]])
    v += K * (generation[2] - generation[haves])
    return v


def c_b_1_v(haves):
    best = x_best_()
    points = sampling(2, haves, best)
    v = generation[haves] + F * (generation[points[0]] - generation[points[1]])
    v += K * (generation[best] - generation[haves])
    return v


def random_pick(haves):
    n = np.random.randint(0, 6)
    if n == 0:
        v = r_1_v(haves)
    elif n == 1:
        v = r_2_v(haves)
    elif n == 2:
        v = b_1_v(haves)
    elif n == 3:
        v = b_2_v(haves)
    elif n == 4:
        v = c_2_v(haves)
    elif n == 5:
        v = c_b_1_v(haves)
    return v


def get_u(haves):
    u = [0, 0]
    for i in range(2):
        if i in j_rand:
            u[i] = candidate[i]
        elif np.random.random() < Cr:
            u[i] = candidate[i]
        else:
            u[i] = generation[haves][i]
    return u


def survive():
    global candidate, ev_x, ev_u, ev_v, generation
    while True:
        initialization()
        for _ in range(maxgen):
            for i in range(popsize):
                ev_x = generation[i]
                candidate = random_pick(i)
                ev_v = candidate
                u = get_u(i)
                ev_u = u
                time.sleep(w*2)
                if min_x < u[0] < max_x and min_y < u[1] < max_y:
                    ob = objective_function([u[0], u[1]])
                else:
                    ob = -999

                if ob > objective_function([generation[i][0], generation[i][1]]):
                    generation[i] = u
                time.sleep(w)

        generation = np.array([[max_x + 99, 0], [max_x + 99, 0]])
        ev_x = [max_x + 99, 0]
        ev_v = [max_x + 99, 0]
        ev_u = [max_x + 99, 0]
        time.sleep(reset)

t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    red_dot.set_data(ev_x)
    green_dot.set_data(ev_v)
    u_dot.set_data(ev_u)
    blue_dot.set_data(generation.T)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




