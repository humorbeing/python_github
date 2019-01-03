import numpy as np


# parameters
'''
lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2
'''

def mutation_normal_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        x = [0, 0]
        for j in range(len(i)):
            x[j] = i[j] + normal_sigma * np.random.normal()
        lambda_gen.append(x)
    return np.array(lambda_gen)


def mutation_uniform_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        xc = [0, 0]
        for j in range(len(i)):
            xc[j] = i[j]
            if np.random.random() < uniform_pm:
                xc[j] = np.random.uniform(boundary_in[2*j], boundary_in[2*j+1])
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def mutation_boundary_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        xc = [0, 0]
        for j in range(len(i)):
            xc[j] = i[j]
            if np.random.random() < boundary_pm:
                if np.random.random() < 0.5:
                    xc[j] = boundary_in[2*j]
                else:
                    xc[j] = boundary_in[2*j+1]
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def delta(g, y, maxgen_in, b):
    rand = np.random.random()
    v = y * (1 - rand**((1-g / maxgen_in)**b))
    return v


def get_x(x, g, U, L, maxgen_in, b):
    if np.random.random() < 0.5:
        xn = x + delta(g, (U - x), maxgen_in, b)
    else:
        xn = x - delta(g, (x - L), maxgen_in, b)
    return xn


def mutation_simulated_annealing_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        xc = [0, 0]
        for j in range(len(i)):
            xc[j] = get_x(i[j], gen,
                          boundary_in[2 * j + 1], boundary_in[2 * j],
                          maxgen, b)
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def mutation_cauchy_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        x = [0, 0]
        for j in range(len(i)):
            x[j] = i[j] + cauchy_sigma * np.random.standard_cauchy()
        lambda_gen.append(x)
    return np.array(lambda_gen)


def poly(n):
    u = np.random.random()
    if u < 0.5:
        v = ((2 * u)**(1/(n+1))) - 1
    else:
        v = 1 - (2*(1-u)) ** (1/(n+1))
    return v


def mutation_polynomial_mutation(lambda_gen_in, boundary_in, gen, normal_sigma=0.5, uniform_pm=0.1,
boundary_pm=0.1, maxgen=50, b=5, cauchy_sigma=0.5,
delta_max=20, n=2):
    lambda_gen = []
    for i in lambda_gen_in:
        x = [0, 0]
        for j in range(len(i)):
            x[j] = i[j] + delta_max * poly(n)
        lambda_gen.append(x)
    return np.array(lambda_gen)


mutations = {
    0: mutation_normal_mutation,
    1: mutation_uniform_mutation,
    2: mutation_boundary_mutation,
    3: mutation_simulated_annealing_mutation,
    4: mutation_cauchy_mutation,
    5: mutation_polynomial_mutation,
}


def mutate(choice, lambda_gen_in, boundary_in, gen,
           normal_sigma=0.5, uniform_pm=0.1,
           boundary_pm=0.1, maxgen=50, b=5,
           cauchy_sigma=0.5,delta_max=20, n=2):

    return mutations[choice](lambda_gen_in, boundary_in, gen,
                             normal_sigma=0.5, uniform_pm=0.1,
                             boundary_pm=0.1, maxgen=50, b=5,
                             cauchy_sigma=0.5, delta_max=20, n=2)
