from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from matplotlib import cm


def objective_function(x_in, y_in):
    return (((x_in+50) ** 2 + y_in ** 2) / 4000) - (np.cos(x_in / 6.5) * np.cos(y_in / 7.777)) + 1


popsize = 10
maxgen = 500
min_x = -30
max_x = 30
min_y = -30
max_y = 30
F = 1.5
K = 1.5
generation = []
candidate = []
Cr = 0.5
j_rand = [0]
x_best = 0

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
ax.view_init(elev=90, azim=90)  # top down
# ax.view_init(elev=13, azim=98)  # side view
# ax.view_init(elev=37, azim=125)  # side view
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
    dots = []
    for point in point_list:
        dot = ax.scatter(point[0], point[1], objective_function(point[0], point[1]), c=co, marker='o')
        dots.append(dot)
    fig.canvas.blit(ax.bbox)
    sleep(0.05)
    for i in dots:
        i.remove()

def x_best_():
    score = []
    for i in generation:
        score.append(objective_function(i[0], i[1]))
    # print('max: ',np.max(score))
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


initialization()
draw_bgw(generation)

for _ in range(50):
    for i in range(popsize):
        candidate = random_pick(i)
        u = get_u(i)
        if min_x < u[0] < max_x and min_y < u[1] < max_y:
            print('here')
            ob = objective_function(u[0], u[1])
        else:
            ob = -999

        if ob > objective_function(generation[i][0], generation[i][1]):
            draw_bgw([u, generation[i], candidate], 'b')
            print(generation[i])
            generation[i] = u
            print(generation[i])

        draw_bgw(generation)
