import numpy as np
import matplotlib.pyplot as plt
import time
import threading


def objective_function(x_in):
    return np.sin(10*np.pi*x_in)*x_in+2.0


min_x = -1
max_x = 2
min_y = 0
max_y = 4
w = 1 / 3

reset = w * 10

plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
x = np.arange(min_x, max_x+0.01, 0.005)
y = objective_function(x)
ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
ax.plot(x, y, lw=2, c='y')

blue_dot, = ax.plot([], [], 'bo', ms=6)
red_dot, = ax.plot([], [], 'ro', ms=10)
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)


mu = 1
laaambda = 200  # 'lambda' is conflicting with python's function.
sigma = 3
maxgen = 40

range_x = abs(max_x-min_x)
# sigma = range_x


generation = []
fitness = []
representations = []
mating_pool = []
lambda_generation = []
mu_lambda_generation = []



def initialization():
    global generation
    new_generation = np.random.uniform(size=mu)
    new_generation = new_generation*range_x+min_x
    generation = np.array(new_generation)


def representation():
    global representations
    representations = np.concatenate((generation, lambda_generation))


def evaluate(rep):
    fit = []
    for individual in rep:
        fit.append(objective_function(individual))
    return np.array(fit)


def evaluation():
    global fitness
    fit = []
    for individual in representations:
        fit.append((objective_function(individual), individual))

    fitness = np.array(fit).T


def selection():
    global mating_pool

    mating_pool = generation


def variation():
    global lambda_generation
    lambda_generations = []
    for i in range(laaambda):
        xc = generation[0] + sigma*np.random.normal()
        xc = fix_x(xc)
        lambda_generations.append(xc)
    lambda_generation = np.array(lambda_generations)


def next_generation():
    global generation
    gen_fit = fitness
    new_gen = []
    for i in range(mu):
        n = np.argmax(gen_fit[0])
        new_gen.append(gen_fit[1][n])
        gen_fit = np.delete(gen_fit, n, 1)

    generation = np.array(new_gen)


def fix_x(x_in):
    if x_in < min_x:
        return min_x
    elif x_in > max_x:
        return max_x
    else:
        return x_in


def survive():
    global generation, fitness, representations, mating_pool, old, lambda_generation
    while True:
        initialization()
        for _ in range(maxgen):
            time.sleep(w)
            selection()
            variation()
            representation()
            time.sleep(w*2)
            evaluation()
            next_generation()
            representations = []

        representations = []
        generation = []
        time.sleep(reset)


t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    blue_dot.set_data(representations, evaluate(representations))
    red_dot.set_data(generation, evaluate(generation))
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




