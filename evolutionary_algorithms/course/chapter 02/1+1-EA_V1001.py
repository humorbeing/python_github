import numpy as np
import matplotlib.pyplot as plt
from time import sleep

plt.ion()


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


mu = 20
laaambda = 1  # 'lambda' is conflicting with python's function.
sigma = 0.5
maxgen = 500
min_x = -1
max_x = 2


range_x = abs(max_x-min_x)
# sigma = range_x
generation = []
fitness = []
representations = []
mating_pool = []
lambda_generation = []
mu_lambda_generation = []


manual = False
sleeping = 0.1

x = np.arange(min_x, max_x, 0.001)
y = function_f(x)
figure, ax = plt.subplots()
ax.plot(x, y, lw=2)
bluelines, = ax.plot([], [], 'bo')
redlines, = ax.plot([], [], 'ro')
figure.canvas.draw()
figure.canvas.flush_events()


def initialization():
    global generation
    new_generation = np.random.uniform(size=mu)
    new_generation = new_generation*range_x+min_x
    generation = np.array(new_generation)


def representation():
    global representations
    representations = np.concatenate((generation, lambda_generation))


def evaluation():
    global fitness
    draw_reps(bluelines, figure.canvas, representations)
    fit = []
    for individual in representations:
        fit.append((function_f(individual), individual))

    fitness = np.array(fit).T


def selection():
    global mating_pool
    draw_reps(redlines, figure.canvas, generation)
    pool = []
    for i in range(laaambda):
        x1 = np.random.randint(0, mu)
        while True:
            x2 = np.random.randint(0, mu)
            if x2 != x1:
                break
        pool.append(x1)
        pool.append(x2)

    mating_pool = np.array(pool)


def variation():
    global lambda_generation
    lambda_generations = []
    for i in range(laaambda):
        x = (generation[mating_pool[i*2]]+generation[mating_pool[(i*2)+1]])/2.0
        x = x + sigma*np.random.normal()
        x = fix_x(x)
        lambda_generations.append(x)
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


def fix_x(x):
    if x < min_x:
        return min_x
    elif x > max_x:
        return max_x
    else:
        return x


def draw_reps(lins_in, canv_in, reps):
    if manual:
        input('Press any key to continue.')
    else:
        sleep(sleeping)
    lins_in.set_xdata(reps)
    fit = []
    for i in reps:
        fit.append(function_f(i))
    lins_in.set_ydata(fit)
    canv_in.draw()
    canv_in.flush_events()


initialization()
for _ in range(maxgen):
    selection()
    variation()
    representation()
    evaluation()
    next_generation()

plt.show()
