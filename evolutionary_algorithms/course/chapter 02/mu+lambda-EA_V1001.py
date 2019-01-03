import numpy as np
import matplotlib.pyplot as plt


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


mu = 10
laaambda = 11  # 'lambda' is conflicting with python's function.
sigma = 1
maxgen = 10
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
        fit.append(function_f(individual))
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
        lambda_generation.append(x)
    return np.array(lambda_generation)


def main_loop():
    pass


generation = generation_initialization()
mating_pool = parent_selection()
print(generation.shape)
# print(generation)
# print(mating_pool.shape)
# print(mating_pool)
new_gen = lambda_variation_operation()
print(new_gen.shape)
generation = np.concatenate((generation, new_gen))
print(generation.shape)
# print(generation)

representation = generation_representation()
fitness = generation_evaluation()
print(fitness.shape)
print(fitness)
fitness = np.sort(fitness)[::-1]
print(fitness)