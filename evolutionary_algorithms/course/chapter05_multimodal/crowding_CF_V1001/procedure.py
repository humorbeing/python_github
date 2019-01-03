import numpy as np
from crossover import crossover
from mutation import mutate
from tools import fix
from fitness import fitness


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
    crosser = np.random.randint(0, 7)
    crosser = 5
    lambda_gen = crossover(crosser, gen_in, mu_in, lamb_da_in, objective_function,
                           BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
                           UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707,
                           DE_K=0.5, DE_F=0.3)
    mutant = np.random.randint(0, 7)
    mutant = 3
    lambda_gen = mutate(mutant, lambda_gen, boundary_in, gen, objective_function,
                        normal_sigma=0.5, uniform_pm=0.1,
                        boundary_pm=0.1, maxgen=maxgen, b=5,
                        cauchy_sigma=0.5, delta_max=20, n=2,
                        DE_K=0.5, DE_F=0.3)
    fixme = np.random.randint(0, 2)
    return fix(fixme, lambda_gen, boundary_in)


def nominate(gen_in, lambda_gen_in):
    cand = np.concatenate((gen_in, lambda_gen_in))
    return cand


def evaluate(cand_in, gen, maxgen):
    fits = np.random.randint(0, 5)
    fits = 1
    return fitness(fits, cand_in, objective_function, gen,
                   reverse=False, c_sigma_truncation=1,
                   PLS_alpha=2, maxgen=maxgen)


def d(cand_in, i, j):
    vector_in = cand_in[i]-cand_in[j]
    return np.linalg.norm(vector_in)
def dis_matrix(cand_in, mu_in):
    # print(len(cand_in))
    # print(mu_in)
    d_matrix = []
    for i in range(mu_in, len(cand_in)):
        d_list = []
        for j in range(mu_in):
            d_list.append(d(cand_in, i, j))
        d_matrix.append(d_list)
    return np.array(d_matrix)
def two_to_two(mu_in, lamb_da_in):
    pool = []
    for i in range(lamb_da_in):
        pool.append(np.random.randint(0, mu_in))
    return np.array(pool)

def select(cand_in, fit_in, mu_in):
    # if min(fit_in)<0:
    #     print('it does')
    fit = fit_in
    d_matrix = dis_matrix(cand_in, mu_in)
    lll = len(cand_in) - mu_in
    CF = 3
    # print(lll)
    for i in range(lll):
        comp = []
        tournament_pool = two_to_two(mu_in, CF)
        for j in range(CF):
            comp.append(d_matrix[i][tournament_pool[j]])
        comp = np.array(comp)
        n = np.argmin(comp)
        fit[tournament_pool[n]] = -999


    ind = np.argpartition(fit, -1 * mu_in)[-1 * mu_in:]
    new_gen = []
    for i in ind:
        new_gen.append(cand_in[i])
    return np.array(new_gen)
