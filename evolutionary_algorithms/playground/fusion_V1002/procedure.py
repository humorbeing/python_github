import numpy as np
from crossover import *
from mutation import mutate
from tools import *


def objective_function(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    return (((x_in + 50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


def initialize(mu_in, boundary_in):
    x_new_generation = np.random.uniform(size=mu_in)
    x_new_generation = x_new_generation * (boundary_in[1] - boundary_in[0]) + boundary_in[0]
    y_new_generation = np.random.uniform(size=mu_in)
    y_new_generation = y_new_generation * (boundary_in[3] - boundary_in[2]) + boundary_in[2]
    new_gen = np.array([x_new_generation, y_new_generation])
    return new_gen.T


def operate(gen_in, mu_in, lamb_da_in, boundary_in, gen, maxgen):
    lambda_gen = crossover_UNDX(gen_in, mu_in, lamb_da_in)
    mutant = np.random.randint(0, 6)
    lambda_gen = mutate(mutant, lambda_gen, boundary_in, gen,
                        normal_sigma=0.5, uniform_pm=0.1,
                        boundary_pm=0.1, maxgen=maxgen, b=5,
                        cauchy_sigma=0.5, delta_max=20, n=2)
    return reflect_fix(lambda_gen, boundary_in)


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
