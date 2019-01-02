import numpy as np
import matplotlib.pyplot as plt
from time import sleep

plt.ion()
def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


mu = 20
laaambda = 20  # 'lambda' is conflicting with python's function.
sigma = 0.5
maxgen = 500
min_x = -1
max_x = 2


range_x = abs(max_x-min_x)
generation = []
fitness = []
representation = []
mating_pool = []


def generation_initialization():
    new_generation = np.random.uniform(size=mu)
    new_generation = new_generation*range_x+min_x
    return np.array(new_generation)


def generation_representation():
    representations = generation
    return np.array(representations)


def generation_evaluation():
    fit = []
    for individual in representation:
        fit.append((function_f(individual), individual))
    return np.array(fit)


def parent_selection():
    pool = []
    for i in range(laaambda):
        x1 = np.random.randint(0, mu)
        while True:
            x2 = np.random.randint(0, mu)
            if x2 != x1:
                break
        pool.append(x1)
        pool.append(x2)
    return np.array(pool)


def lambda_variation_operation():
    lambda_generation = []
    for i in range(laaambda):
        x = (generation[mating_pool[i*2]]+generation[mating_pool[(i*2)+1]])/2.0
        x = x + sigma*np.random.normal()
        x = fix_x(x)
        lambda_generation.append(x)
    return np.array(lambda_generation)


def pick_new_generation(gen_fit):
    new_gen = []
    for i in range(mu):
        n = np.argmax(gen_fit[0])
        new_gen.append(gen_fit[1][n])
        gen_fit = np.delete(gen_fit, n, 1)
    return np.array(new_gen)


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
    # lins_in, = ax_in.plot([], [], 'ro')
    lins_in.set_xdata(reps)
    fit = []
    for i in reps:
        fit.append(function_f(i))
    lins_in.set_ydata(fit)
    canv_in.draw()
    canv_in.flush_events()


manual = False
sleeping = 0.1

x = np.arange(min_x, max_x, 0.001)
y = function_f(x)
figure, ax = plt.subplots()
ax.plot(x, y, lw=2)
bluelines, = ax.plot([],[], 'bo')
redlines, = ax.plot([], [], 'ro')
figure.canvas.draw()
figure.canvas.flush_events()

generation = generation_initialization()
for _ in range(maxgen):
    draw_reps(redlines, figure.canvas, generation)
    mating_pool = parent_selection()
    new_gen = lambda_variation_operation()
    generation = np.concatenate((generation, new_gen))
    representation = generation_representation()
    draw_reps(bluelines, figure.canvas, generation)
    fitness = generation_evaluation().T
    generation = pick_new_generation(fitness)


# plt.ioff()
# plt.gcf().clear()
# plt.plot(best_fitness)
# plt.ylim(0, 4)
plt.show()
