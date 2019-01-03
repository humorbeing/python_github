from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from matplotlib import cm


def objective_function(x_in, y_in):
    return (((x_in+50) ** 2 + y_in ** 2) / 4000) - (np.cos(x_in / 6.5) * np.cos(y_in / 7.777)) + 1


popsize = 50
maxgen = 500
min_x = -30
max_x = 30
min_y = -30
max_y = 30
F = 0.7
K = 0.8
generation = []
candidate = []
Cr = 0.5
j_rand = [0]


plt.ion()
fig = plt.figure(figsize=(12, 9))

ax = fig.gca(projection='3d')
X = np.arange(min_x, max_x, 0.5)
Y = np.arange(min_y, max_y, 0.5)
X, Y = np.meshgrid(X, Y)
Z = objective_function(X, Y)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
ax.set_zlabel('Z')
ax.set_zlim(0, 3.5)
# ax.view_init(elev=90, azim=90)  # top down
# ax.view_init(elev=13, azim=98)  # side view
ax.view_init(elev=37, azim=125)  # side view
plt.show()
fig.canvas.draw()


def initialization():
    global generation
    p = []
    for _ in range(popsize):
        x_axis = np.random.random() * (max_x - min_x) + min_x
        y_axis = np.random.random() * (max_y - min_y) + min_y
        p.append([x_axis, y_axis])
    generation = np.array(p)


def draw_bgw(point_list, co='r'):
    fig.canvas.flush_events()
    for point in point_list:
        dot1 = ax.scatter(point[0], point[1], objective_function(point[0], point[1]), c=co, marker='o')
    fig.canvas.blit(ax.bbox)
    sleep(0.01)
    dot1.remove()


def sampling(sample_number):
    check_samples = []
    samples = [0 for _ in range(sample_number)]
    for i in range(sample_number):
        samples[i] = np.random.randint(0, popsize)
        while samples[i] in check_samples:
            samples[i] = np.random.randint(0, popsize)
        check_samples.append(samples[0])
    return np.array(samples)


def r_1_v():
    global candidate
    points = sampling(4)
    v = generation[points[0]] + F * (generation[points[1]] + generation[points[2]])
    candidate = [v, points[3]]


def r_2_v():
    global candidate
    points = sampling(6)
    v = generation[points[0]] + F * (generation[points[1]] + generation[points[2]])
    v += K * (generation[points[3]] + generation[points[4]])
    candidate = [v, points[5]]


def random_pick():
    n = np.random.randint(0, 2)
    if n == 0:
        r_1_v()
    elif n == 1:
        r_2_v()

def get_u():
    u = [0, 0]
    for i in range(2):
        if i in j_rand:
            u[i] = candidate[0][i]
        elif np.random.random() < Cr:
            u[i] = candidate[0][i]
        else:
            u[i] = generation[candidate[1]][i]
    return u


initialization()
draw_bgw(generation)

for i in range(50):
    random_pick()
    u = get_u()
    if min_x < u[0] < max_x and min_y < u[1] < max_y:
        print('here')
        ob = objective_function(u[0], u[1])
    else:
        ob = -999

    if ob > objective_function(generation[candidate[1]][0], generation[candidate[1]][1]):
        draw_bgw([u, generation[candidate[1]], candidate[0]], 'b')
        generation[candidate[1]] = u

    draw_bgw(generation)
