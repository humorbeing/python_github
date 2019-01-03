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


def x_best_(gen_in, obj):
    score = []
    for i in gen_in:
        score.append(obj([i[0], i[1]]))
    return np.argmax(score)


def sampling(sample_number, haves, mu_in, best=-1):

    check_samples = [haves]
    if best != -1:
        check_samples.append(best)
    samples = [0 for _ in range(sample_number)]
    for i in range(sample_number):
        samples[i] = np.random.randint(0, mu_in)
        while samples[i] in check_samples:
            samples[i] = np.random.randint(0, mu_in)
        check_samples.append(samples[0])
    return np.array(samples)


def r_1_v(haves, gen_in, mu_in, obj, F, K):
    points = sampling(3, haves, mu_in)
    v = gen_in[points[0]] + F * (gen_in[points[1]] - gen_in[points[2]])
    return v


def r_2_v(haves, gen_in, mu_in, obj, F, K):
    points = sampling(5, haves, mu_in)
    v = gen_in[points[0]] + F * (gen_in[points[1]] - gen_in[points[2]])
    v += K * (gen_in[points[3]] + gen_in[points[4]])
    return v


def b_1_v(haves, gen_in, mu_in, obj, F, K):
    best = x_best_(gen_in, obj)
    points = sampling(2, haves, mu_in, best)
    v = gen_in[best] + F * (gen_in[points[0]] - gen_in[points[1]])
    return v


def b_2_v(haves, gen_in, mu_in, obj, F, K):
    best = x_best_(gen_in, obj)
    points = sampling(4, haves, mu_in, best)
    v = gen_in[best] + F * (gen_in[points[0]] - gen_in[points[1]])
    v += K * (gen_in[points[2]] - gen_in[points[3]])
    return v


def c_2_v(haves, gen_in, mu_in, obj, F, K):
    points = sampling(3, haves, mu_in)
    v = gen_in[haves] + F * (gen_in[points[0]] - gen_in[points[1]])
    v += K * (gen_in[2] - gen_in[haves])
    return v


def c_b_1_v(haves, gen_in, mu_in, obj, F, K):
    best = x_best_(gen_in, obj)
    points = sampling(2, haves, mu_in, best)
    v = gen_in[haves] + F * (gen_in[points[0]] - gen_in[points[1]])
    v += K * (gen_in[best] - gen_in[haves])
    return v


des = {
    0: r_1_v,
    1: r_2_v,
    2: b_1_v,
    3: b_2_v,
    4: c_2_v,
    5: c_b_1_v,
}


def mutation_differential_evolution(gen_in, mu_in, lamb_da_in, obj, DE_K=0.5, DE_F=0.3):
    needed_children = int(lamb_da_in / mu_in)
    lambda_gen = []
    for i in range(mu_in):
        for j in range(needed_children):
            de = np.random.randint(0, 6)
            child = des[de](i, gen_in, mu_in, obj, DE_F, DE_K)
            lambda_gen.append(child)
    return np.array(lambda_gen)


mutations = {
    0: mutation_normal_mutation,
    1: mutation_uniform_mutation,
    2: mutation_boundary_mutation,
    3: mutation_simulated_annealing_mutation,
    4: mutation_cauchy_mutation,
    5: mutation_polynomial_mutation,
}


def mutate(choice, lambda_gen_in, boundary_in, gen, obj,
           normal_sigma=0.5, uniform_pm=0.1,
           boundary_pm=0.1, maxgen=50, b=5,
           cauchy_sigma=0.5, delta_max=20, n=2,
           DE_K=0.5, DE_F=0.3):
    if choice == 6:
        return mutation_differential_evolution(lambda_gen_in, len(lambda_gen_in),
                                               len(lambda_gen_in), obj, DE_K=DE_K, DE_F=DE_F)
    else:
        return mutations[choice](lambda_gen_in, boundary_in, gen,
                                 normal_sigma=normal_sigma, uniform_pm=uniform_pm,
                                 boundary_pm=boundary_pm, maxgen=maxgen, b=b,
                                 cauchy_sigma=cauchy_sigma, delta_max=delta_max, n=n)
