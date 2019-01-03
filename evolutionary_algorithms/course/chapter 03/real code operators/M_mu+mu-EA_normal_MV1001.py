from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import threading

# bgrcmyk o+*>
def objective_function(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    return (((x_in + 50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


min_x = -30
max_x = 30
min_y = -30
max_y = 30
w = 1/6
reset = w * 5

plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
X = np.arange(min_x-2, max_x+2, 0.1)
Y = np.arange(min_y-2, max_y+2, 0.1)
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_xlim(min_x-1, max_x+1)
ax.set_ylabel('Y')
ax.set_ylim(min_y-1, max_y+1)
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
red_dot, = ax.plot([], [], 'ko', ms=12)
blue_dot, = ax.plot([], [], 'mo', ms=7)

fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)

mu = 20
# laaambda = 60  # 'lambda' is conflicting with python's function.
sigma_x = 2.0
sigma_y = 2.5
maxgen = 20

# range_x = abs()
# sigma = range_x


generation = np.array([[max_x+99, 0]])
fitness = []
representations = np.array([[max_x+99, 0]])
mating_pool = np.array([[max_x+99, 0]])
lambda_generation = np.array([[max_x+99, 0]])
mu_lambda_generation = np.array([[max_x+99, 0]])


def initialization():
    global generation
    x_new_generation = np.random.uniform(size=mu)
    x_new_generation = x_new_generation*(max_x-min_x)+min_x
    y_new_generation = np.random.uniform(size=mu)
    y_new_generation = y_new_generation * (max_y - min_y) + min_y
    generation = np.array([x_new_generation, y_new_generation])
    generation = generation.T


def representation():
    global representations
    representations = representations = np.concatenate((generation, lambda_generation))


def evaluate(rep):
    fit = []
    for individual in rep:
        fit.append(objective_function(individual))
    return np.array(fit)


def evaluation():
    global fitness
    fit = []
    for i in range(2*mu):
        fit.append((objective_function(representations[i]), i))

    fitness = np.array(fit).T


def selection():
    global mating_pool

    mating_pool = []


def variation():
    global lambda_generation
    lambda_generations = []

    for i in range(mu):
        xc = [0, 0]
        xc[0] = generation[i][0]
        xc[1] = generation[i][1]
        xc[0] += sigma_x * np.random.normal()
        xc[1] += sigma_y * np.random.normal()
        xc = fix_x(xc)
        # xc[1] = fix_x(xc)[1]
        # print(xc)
        lambda_generations.append(xc)
    lambda_generation = np.array(lambda_generations)
    # print(lambda_generation)


def next_generation():
    global generation
    gen_fit = fitness
    new_gen = []
    for i in range(mu):
        n = np.argmax(gen_fit[0])
        new_gen.append(int(gen_fit[1][n]))
        gen_fit = np.delete(gen_fit, n, 1)
    new_g = []
    # print(new_gen)
    for i in new_gen:
        new_g.append(representations[i])
    generation = np.array(new_g)


def fix_x(point_in):
    # aha = [0, 0]
    x_in = point_in[0]
    y_in = point_in[1]
    if x_in < min_x:
        x_in = min_x
    elif x_in > max_x:
        x_in = max_x
    else:
        pass
    if y_in < min_y:
        y_in = min_y
    elif y_in > max_y:
        y_in = max_y
    else:
        pass

    return np.array([x_in, y_in])


def survive():
    global generation, fitness, representations, mating_pool, lambda_generation
    while True:
        initialization()
        for _ in range(maxgen):
            time.sleep(w*1.7)
            # selection()
            variation()

            representation()
            # generation = np.array([[max_x + 99, 0]])
            time.sleep(w*2)
            evaluation()
            next_generation()
            representations = np.array([[max_x+99, 0]])

        representations = np.array([[max_x+99, 0]])
        generation = np.array([[max_x+99, 0]])
        lambda_generation = np.array([[max_x+99, 0]])
        time.sleep(reset)


t = threading.Thread(target=survive)
t.daemon = True
t.start()
# initialization()
while True:
    fig.canvas.restore_region(back)
    blue_dot.set_data(representations.T)
    red_dot.set_data(generation.T)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




