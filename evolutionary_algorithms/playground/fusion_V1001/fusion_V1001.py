import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import threading


def objective_function(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    return (((x_in + 50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


is_back = True
x_min = -30
x_max = 30
y_min = -30
y_max = 30
w = 1/30
boundary = (x_min, x_max, y_min, y_max)
print(boundary)
plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
X = np.arange(x_min-2, x_max+2, 0.1)
Y = np.arange(y_min-2, y_max+2, 0.1)
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_xlim(x_min-1, x_max+1)
ax.set_ylabel('Y')
ax.set_ylim(y_min-1, y_max+1)
Z = objective_function([X, Y])
ax.contourf(X, Y, Z, cmap=cm.coolwarm)
red_dot, = ax.plot([], [], 'ko', ms=10)
blue_dot, = ax.plot([], [], 'mo', ms=6)
fig.canvas.draw()
if is_back:
    back = fig.canvas.copy_from_bbox(ax.bbox)

mu = 20
lamb_da = 60

generation = np.array([[x_max+99, 0]])
# mating_pool = []
# lambda_generation = np.array([[x_max+99, 0]])
candidate = np.array([[x_max+99, 0]])
# fitness = []


def initialize(mu_in, boundary_in):
    x_new_generation = np.random.uniform(size=mu_in)
    x_new_generation = x_new_generation * (boundary_in[1] - boundary_in[0]) + boundary_in[0]
    y_new_generation = np.random.uniform(size=mu_in)
    y_new_generation = y_new_generation * (boundary_in[3] - boundary_in[2]) + boundary_in[2]
    new_gen = np.array([x_new_generation, y_new_generation])
    return new_gen.T


def two_to_two(mu_in, lamb_da_in):
    pool = []
    for i in range(lamb_da_in):
        pool.append(np.random.randint(0, mu_in))
    return np.array(pool)


def UNDX_parents(mu_in, lamb_da_in):
    check_pool = []
    pool = []
    for i in range(lamb_da_in):
        x1 = np.random.randint(0, mu_in)
        check_pool.append(x1)
        x2 = np.random.randint(0, mu_in)
        while x2 in check_pool:
            x2 = np.random.randint(0, mu_in)
        check_pool.append(x2)
        x3 = np.random.randint(0, mu_in)
        while x3 in check_pool:
            x3 = np.random.randint(0, mu_in)
        check_pool = []
        pool.append(x1)
        pool.append(x2)
        pool.append(x3)

    return np.array(pool)


def crossover_UNDX(gen_in, mu_in, lamb_da_in, sigma_xi=0.8, sigma_eta=0.707):
    mating_pool = UNDX_parents(mu_in, lamb_da_in)
    lambda_gen = []

    for i in range(lamb_da_in):
        xc = [0, 0]
        e = [0.0, 0.0]
        x1 = gen_in[mating_pool[i*3]]
        x2 = gen_in[mating_pool[(i*3)+1]]
        x3 = gen_in[mating_pool[(i*3)+2]]
        k = (x1 + x2) / 2
        d = (x2 - x1)
        e[0], e[1] = d[1], d[0]
        e[0] *= -1
        e = e / (np.linalg.norm(e) + 0.00001)
        D = np.dot(x3, e)
        o = k + sigma_xi * np.random.normal() * d
        o += D * sigma_eta * np.random.normal() * e
        xc[0] = o[0]
        xc[1] = o[1]
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def mutation_normal(lambda_gen_in, sigma=0.5):
    lambda_gen = []
    for i in lambda_gen_in:
        x = [0, 0]
        for j in range(len(i)):
            x[j] = i[j] + sigma * np.random.normal()
        lambda_gen.append(x)
    return np.array(lambda_gen)


def is_fixed(point_in, boundary_in):
    x_in = point_in[0]
    y_in = point_in[1]
    if x_in < boundary_in[0]:
        x_in = 2 * boundary_in[0] - x_in
    elif x_in > boundary_in[1]:
        x_in = 2 * boundary_in[1] - x_in
    else:
        pass
    if y_in < boundary_in[2]:
        y_in = 2 * boundary_in[2] - y_in
    elif y_in > boundary_in[3]:
        y_in = 2 * boundary_in[3] - y_in
    else:
        pass

    return np.array([x_in, y_in])


def fix_one(point_in, boundary_in):
    old = point_in
    while True:
        new = is_fixed(old, boundary_in)
        if all(new == old):
            break
        else:
            old = new
    return new


def fix(lambda_gen_in, boundary_in):
    lambda_gen = []
    for i in lambda_gen_in:
        x = fix_one(i, boundary_in)
        lambda_gen.append(x)
    return np.array(lambda_gen)


def operate(gen_in, mu_in, lamb_da_in, boundary_in):
    lambda_gen = crossover_UNDX(gen_in, mu_in, lamb_da_in)
    lambda_gen = mutation_normal(lambda_gen)
    return fix(lambda_gen, boundary_in)


def nominate(gen_in, lambda_gen_in):
    cand = np.concatenate((gen_in, lambda_gen_in))
    return cand


def evaluate(cand_in):
    fit = []
    for i in cand_in:
        f = objective_function(i)
        fit.append(f)
    return np.array(fit)


def select(cand_in, fit_in, mu_in):
    ind = np.argpartition(fit_in, -1 * mu_in)[-1 * mu_in:]
    new_gen = []
    for i in ind:
        new_gen.append(cand_in[i])
    return np.array(new_gen)


def survive():
    global generation, candidate
    generation = initialize(mu, boundary)
    while True:
        lamb_da_gen = operate(generation, mu, lamb_da, boundary)
        candidate = nominate(generation, lamb_da_gen)
        fitness = evaluate(candidate)
        generation = select(candidate, fitness, mu)
        time.sleep(w)


t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    if is_back:
        fig.canvas.restore_region(back)
    blue_dot.set_data(candidate.T)
    red_dot.set_data(generation.T)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
