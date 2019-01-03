import numpy as np
import matplotlib.pyplot as plt


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


mu = 10
laaambda = 10  # 'lambda' is conflicting with python's function.
maxgen = 10
min_x = -1
max_x = 2
range_x = abs(max_x-min_x)
generation = []
fitness = []
representation = []
mating_pool = []


def generation_initialization():
    new_generation = np.random.uniform(size=(1, 10))
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
    for i in range(next_gen_size):
        x1 = np.random.randint()