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
w = 1/20
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

mu = 20
laaambda = 200  # 'lambda' is conflicting with python's function.
sigma = 0
maxgen = 20
sigma_xi = 0.2
sigma_eta = 0.15 / np.sqrt(2)
# range_x = abs()
# sigma = range_x


generation = np.array([[max_x+99, 0]])
fitness = []
representations = np.array([[max_x+99, 0]])
mating_pool = []
lambda_generation = np.array([[max_x+99, 0]])
mu_lambda_generation = []


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
    representations = lambda_generation


def evaluate(rep):
    fit = []
    for individual in rep:
        fit.append(objective_function(individual))
    return np.array(fit)


def evaluation():
    global fitness
    fit = []
    for i in range(laaambda):

        fit.append((objective_function(representations[i]), i))

    fitness = np.array(fit).T


def selection():
    global mating_pool
    check_pool = []
    pool = []
    for i in range(laaambda):
        x1 = np.random.randint(0, mu)
        check_pool.append(x1)
        x2 = np.random.randint(0, mu)
        while x2 in check_pool:
            x2 = np.random.randint(0, mu)
        check_pool.append(x2)
        x3 = np.random.randint(0, mu)
        while x3 in check_pool:
            x3 = np.random.randint(0, mu)
        check_pool = []
        pool.append(x1)
        pool.append(x2)
        pool.append(x3)

    mating_pool = np.array(pool)


def variation():
    global lambda_generation
    lambda_generations = []
    print(len(generation))
    sigma_x_v = np.var(generation)
    sigma_y_v = np.var(generation)
    # sigma_x_v *= (2/(max_x-min_x))
    # sigma_y_v *= (2/(max_y-min_y))
    print(sigma_x_v, sigma_y_v)
    for i in range(laaambda):
        # xc = np.array([0.0, 0.0])
        xc = [0, 0]
        e = [0.0, 0.0]
        # al = np.random.uniform(size=2)
        x1 = generation[mating_pool[i*3]]
        x2 = generation[mating_pool[(i*3)+1]]
        x3 = generation[mating_pool[(i*3)+2]]
        k = (x1 + x2) / 2
        d = (x2 - x1)
        e[0], e[1] = d[1], d[0]
        e[0] *= -1
        e = e / (np.linalg.norm(e) + 0.00001)
        D = np.dot(x3, e)
        o = k + sigma_xi * np.random.normal() * d
        o += D * sigma_eta * np.random.normal() * e
        # print(al1, al2, al3)
        # xc = np.multiply(x1, al) + np.multiply(x2, np.subtract(1, al))
        xc[0] = o[0]
        xc[1] = o[1]
        xc[0] = xc[0] + sigma_x_v * np.random.normal()
        xc[1] = xc[1] + sigma_y_v * np.random.normal()
        xc = fix_x(xc)
        lambda_generations.append(xc)
    lambda_generation = np.array(lambda_generations)


def next_generation():
    global generation
    gen_fit = fitness
    new_gen = []
    for i in range(mu):
        n = np.argmax(gen_fit[0])
        # print(n, gen_fit[1][n])
        new_gen.append(int(gen_fit[1][n]))
        gen_fit = np.delete(gen_fit, n, 1)
    new_g = []
    # print(new_gen)
    for i in new_gen:
        new_g.append(representations[i])
    generation = np.array(new_g)


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
        for _ in range(maxgen):
            time.sleep(w)
            selection()
            variation()

            representation()
            time.sleep(w * 2)
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




