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
w = 1/5
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
red_dot, = ax.plot([], [], 'ko', ms=10)
blue_dot, = ax.plot([], [], 'mo', ms=6)
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)

mu = 4
laaambda = mu  # 'lambda' is conflicting with python's function.
sigma = 0
maxgen = 20
nn = 2  # SBX 2=spread, 5=focus
b = 2  # SA mutation, 5=converge fast, 2= slower
# range_x = abs()
# sigma = range_x


generation = np.array([[max_x+99, 0]])
fitness = []
representations = np.array([[max_x+99, 0]])
mating_pool = []
lambda_generation = np.array([[max_x+99, 0]])
mu_lambda_generation = []


def beta():
    u = np.random.random()
    if u > 0.5:
        c = (2 * u) ** (1 / (nn + 1))
    else:
        c = (2 * (1 - u)) ** (-(1 / (nn + 1)))
    return c

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
    representations = generation


def evaluate(rep):
    fit = []
    for individual in rep:
        fit.append(objective_function(individual))
    return np.array(fit)


def linear_scaling_max(fit_in):
    fit = []
    f_max = max(fit_in)
    f_avg = np.average(fit_in)
    a_in = f_avg / (f_max - f_avg)
    b_in = (f_avg * (f_max - 2 * f_avg)) / (f_max - f_avg)
    for i in fit_in:

        f = a_in * i + b_in
        print('f = ({})*({}) + ({}): {}'.format(a_in, i, b_in, f))
        fit.append(round(f, 5))
    return np.array(fit)


def linear_scaling_min(fit_in):
    fit = []
    f_min = min(fit_in)
    f_avg = np.average(fit_in)
    a_in = f_avg / (f_avg - f_min)
    b_in = (f_avg * f_min) / (f_avg - f_min)
    for i in fit_in:
        f = a_in * i - b_in
        print('f = ({})*({}) + ({}): {}'.format(a_in, i, b_in, f))
        fit.append(round(f, 5))
    return np.array(fit)

def evaluation():
    global fitness
    fit = []
    for i in range(laaambda):

        fit.append(objective_function(representations[i]))
    # print(fit)
    fitness = np.array(fit)
    # print('before', fitness)
    fitness = linear_scaling_max(fitness)
    # print('max', fitness)
    fitness = linear_scaling_min(fitness)
    # print('min', fitness)

    # print(fitness)


def selection():
    global mating_pool
    # print(fitness)
    pool = []
    P = []  # big 'P', not lower case p
    fitness_sum = sum(fitness)
    # print(fitness)
    # print(fitness_sum)
    for individual in range(mu):
        P.append(fitness[individual]/fitness_sum)

    P_sum = 0
    portion = 1 / mu
    winner = np.random.rand() * portion

    # print(P)
    P_sum += P[0]
    i = 0
    while len(pool) < mu:
        # print('winner:',winner)
        # print('P_sum:',P_sum)
        if winner < P_sum:
            pool.append(i)
            winner += portion
        else:
            i += 1
            P_sum += P[i]
            # print('pool',pool)

    np.random.shuffle(pool)
    # print(pool)
    mating_pool = np.array(pool)


def delta(g, y):
    rand = np.random.random()
    v = y * (1 - rand**((1-g / maxgen)**b))
    return v


def get_x(x, g, U, L):
    if np.random.random() < 0.5:
        xn = x + delta(g, (U - x))
    else:
        xn = x - delta(g, (x - L))
    return xn


def variation(gen):
    global lambda_generation
    lambda_generations = []
    for i in range(int(laaambda/2)):
        xc = [0, 0]
        bet = beta()
        # al = np.random.uniform(size=2)
        x1 = generation[mating_pool[i*2]]
        x2 = generation[mating_pool[(i*2)+1]]
        xcc = 0.5 * (x1 + x2) + 0.5 * bet * (x1 - x2)
        xc[0] = xcc[0]
        xc[1] = xcc[1]
        # xc = np.multiply(x1, al) + np.multiply(x2, np.subtract(1, al))
        # xc = (generation[mating_pool[i*2]]+generation[mating_pool[(i*2)+1]])/2.0
        xc[0] = get_x(xc[0], gen, max_x, min_x)
        xc[1] = get_x(xc[1], gen, max_y, min_y)
        xc = fix_x(xc)
        lambda_generations.append(xc)
        xc = np.array([0.0, 0.0])
        xcc = 0.5 * (x1 + x2) + 0.5 * bet * (x2 - x1)
        xc[0] = xcc[0]
        xc[1] = xcc[1]
        # xc = np.multiply(x1, al) + np.multiply(x2, np.subtract(1, al))
        # xc = (generation[mating_pool[i*2]]+generation[mating_pool[(i*2)+1]])/2.0
        xc[0] = get_x(xc[0], gen, max_x, min_x)
        xc[1] = get_x(xc[1], gen, max_y, min_y)
        xc = fix_x(xc)
        lambda_generations.append(xc)
    lambda_generation = np.array(lambda_generations)


def next_generation():
    global generation
    # gen_fit = fitness
    # new_gen = []
    # for i in range(mu):
    #     n = np.argmax(gen_fit[0])
    #     new_gen.append(int(gen_fit[1][n]))
    #     gen_fit = np.delete(gen_fit, n, 1)
    # new_g = []
    # # print(new_gen)
    # for i in new_gen:
    #     new_g.append(representations[i])
    generation = lambda_generation


def fix(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    if x_in < min_x:
        x_in = 2 * min_x - x_in
    elif x_in > max_x:
        x_in = 2 * max_x - x_in
    else:
        pass
    if y_in < min_y:
        y_in = 2 * min_y - y_in
    elif y_in > max_y:
        y_in = 2 * max_y - y_in
    else:
        pass

    return np.array([x_in, y_in])


def fix_x(point_in):
    old = point_in
    while True:
        new = fix(old)
        if all(new == old):
            break
        else:
            old = new
    return new


def survive():
    global generation, fitness, representations, mating_pool, lambda_generation
    while True:
        initialization()
        for i in range(maxgen):

            representation()
            time.sleep(w)
            evaluation()
            selection()
            variation(i)


            # evaluation()
            next_generation()
            # representations = np.array([[max_x+99, 0]])
            time.sleep(w * 2)

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




