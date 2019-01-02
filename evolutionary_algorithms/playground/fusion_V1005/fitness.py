import numpy as np


# parameters
'''
cand_in, obj, g, reverse=False,
c_sigma_truncation=1, PLS_alpha=2, maxgen=50
'''


def get_fitness(cand_in, obj, reverse=False):
    fit = []
    for i in cand_in:
        f = obj(i)
        fit.append(f)
    fit = np.array(fit)
    # print(reverse)
    if reverse:
        return np.max(fit) - fit
    else:
        return fit


def fitness_no_change(cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    # print('n')
    fit = get_fitness(cand_in, obj, reverse=reverse)
    return fit


def linear_scaling_max(fit_in):
    fit = []
    f_max = max(fit_in)
    f_avg = np.average(fit_in)
    a_in = f_avg / (f_max - f_avg)
    b_in = (f_avg * (f_max - 2 * f_avg)) / (f_max - f_avg)
    for i in fit_in:
        f = a_in * i + b_in
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
        fit.append(round(f, 5))
    return np.array(fit)


def fitness_linear_scaling(cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    # print('l')
    fit = get_fitness(cand_in, obj, reverse=reverse)
    fit = linear_scaling_max(fit)
    fit = linear_scaling_min(fit)
    return fit


def sigma_truncation(fit_in, cand_in, c_sigma_truncation=1):
    fit = []
    sigma_in = np.std(cand_in)
    f_avg = np.average(fit_in)
    for i in fit_in:
        f = i - (f_avg - c_sigma_truncation * sigma_in)
        if f < 0:
            f = 0
        fit.append(f)
    # print(c_sigma_truncation)
    return np.array(fit)


def fitness_sigma_truncation(cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    # print('s')
    fit = get_fitness(cand_in, obj, reverse=reverse)
    fit = sigma_truncation(fit, cand_in, c_sigma_truncation)
    return fit


def power_law_scaling(fit_in, PLS_alpha=2):
    fit = []
    for i in fit_in:
        f = i**PLS_alpha
        fit.append(f)
    # print(PLS_alpha)
    return np.array(fit)


def fitness_power_law_scaling(cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    # print('p')
    fit = get_fitness(cand_in, obj, reverse=reverse)
    fit = power_law_scaling(fit, PLS_alpha)
    return fit


def blotzmann_scaling(fit_in, g, maxgen=50):
    fit = []
    alpha_in = 2
    t_in = alpha_in * (1.0000001 - (g / maxgen))
    for i in fit_in:
        f = np.exp(i / t_in)
        fit.append(f)
    # print(maxgen)
    return np.array(fit)


def fitness_boltzmann_scaling(cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    # print('b')
    fit = get_fitness(cand_in, obj, reverse=reverse)
    fit = blotzmann_scaling(fit, g, maxgen)
    return fit


fitnesses = {
    0: fitness_no_change,
    1: fitness_linear_scaling,
    2: fitness_sigma_truncation,
    3: fitness_power_law_scaling,
    4: fitness_boltzmann_scaling,
}


def fitness(choice, cand_in, obj, g,
            reverse=False, c_sigma_truncation=1,
            PLS_alpha=2, maxgen=50):
    return fitnesses[choice](cand_in, obj, g,
                             reverse=reverse, c_sigma_truncation=c_sigma_truncation,
                             PLS_alpha=PLS_alpha, maxgen=maxgen)
