import numpy as np

# parameters
'''
gen_in, mu_in, lamb_da_in,BLX_alpha=0.5,
SPX_epsilon=1, SBX_n=2, UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707
'''


def parents(mu_in, lamb_da_in):
    check_pool = []
    pool = []
    for i in range(lamb_da_in):
        x1 = np.random.randint(0, mu_in)
        check_pool.append(x1)
        x2 = np.random.randint(0, mu_in)
        while x2 in check_pool:
            x2 = np.random.randint(0, mu_in)
        check_pool.append(x2)
        x3 = np.random.randint(0, mu_in)
        while x3 in check_pool:
            x3 = np.random.randint(0, mu_in)
        check_pool = []
        pool.append(x1)
        pool.append(x2)
        pool.append(x3)

    return np.array(pool)


def crossover_UNDX(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []

    for i in range(lamb_da_in):
        xc = [0, 0]
        e = [0.0, 0.0]
        x1 = gen_in[mating_pool[i*3]]
        x2 = gen_in[mating_pool[(i*3)+1]]
        x3 = gen_in[mating_pool[(i*3)+2]]
        k = (x1 + x2) / 2
        d = (x2 - x1)
        e[0], e[1] = d[1], d[0]
        e[0] *= -1
        e = e / (np.linalg.norm(e) + 0.00001)
        D = np.dot(x3, e)
        o = k + UNDX_sigma_xi * np.random.normal() * d
        o += D * UNDX_sigma_eta * np.random.normal() * e
        xc[0] = o[0]
        xc[1] = o[1]
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def two_to_two(mu_in, lamb_da_in):
    pool = []
    for i in range(lamb_da_in):
        pool.append(np.random.randint(0, mu_in))
    return np.array(pool)


def two_to_one(mu_in, lamb_da_in):
    pool = []
    for i in range(lamb_da_in*2):
        pool.append(np.random.randint(0, mu_in))
    return np.array(pool)


def crossover_whole_arithmetic_x(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []
    for i in range(int(lamb_da_in/2)):
        xc1 = [0, 0]
        xc2 = [0, 0]
        alpha = np.random.random()
        beta = 1 - alpha
        x1 = gen_in[mating_pool[i * 2]]
        x2 = gen_in[mating_pool[(i * 2) + 1]]
        xcc1 = x1*alpha + x2*beta
        xc1[0] = xcc1[0]
        xc1[1] = xcc1[1]
        xcc2 = x1*beta + x2*alpha
        xc2[0] = xcc2[0]
        xc2[1] = xcc2[1]
        lambda_gen.append(xc1)
        lambda_gen.append(xc2)
    return np.array(lambda_gen)


def crossover_deterministic_arithmetic_x(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []
    for i in range(lamb_da_in):
        xc1 = [0, 0]
        alpha = np.random.uniform(size=2)
        beta = np.array([1, 1])
        beta = beta - alpha
        x1 = gen_in[mating_pool[i * 2]]
        x2 = gen_in[mating_pool[(i * 2) + 1]]
        xcc1 = np.multiply(x1, alpha) + np.multiply(x2, beta)
        xc1[0] = xcc1[0]
        xc1[1] = xcc1[1]
        lambda_gen.append(xc1)
    return np.array(lambda_gen)


def crossover_BLX(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []
    for i in range(lamb_da_in):
        xc1 = [0, 0]
        x1 = gen_in[mating_pool[i * 2]]
        x2 = gen_in[mating_pool[(i * 2) + 1]]
        for j in range(2):
            big = max([x1[j], x2[j]])
            small = min([x1[j], x2[j]])
            xc1[j] = np.random.uniform((small - BLX_alpha * (big - small)),
                                       (big + BLX_alpha * (big - small)))
        lambda_gen.append(xc1)
    return np.array(lambda_gen)


def crossover_SPX(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []

    for i in range(lamb_da_in):
        xc = [0, 0]
        x1 = gen_in[mating_pool[i*3]]
        x2 = gen_in[mating_pool[(i*3)+1]]
        x3 = gen_in[mating_pool[(i*3)+2]]
        k = (x1 + x2 + x3) / 3
        y1 = x1 + SPX_epsilon * (x1 - k)
        y2 = x2 + SPX_epsilon * (x2 - k)
        y3 = x3 + SPX_epsilon * (x3 - k)
        al1 = np.random.random()
        al2 = np.random.uniform(0, (1 - al1))
        al3 = 1 - al1 - al2
        xcc = al1 * y1 + al2 * y2 + al3 * y3
        xc[0] = xcc[0]
        xc[1] = xcc[1]
        lambda_gen.append(xc)
    return np.array(lambda_gen)


def beta(SBX_n):
    u = np.random.random()
    if u > 0.5:
        b = (2 * u) ** (1 / (SBX_n + 1))
    else:
        b = (2 * (1 - u)) ** (-(1 / (SBX_n + 1)))
    return b


def crossover_SBX(gen_in, mu_in, lamb_da_in,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707):
    mating_pool = parents(mu_in, lamb_da_in)
    lambda_gen = []
    for i in range(int(lamb_da_in/2)):
        xc1 = [0, 0]
        xc2 = [0, 0]
        betaa = beta(SBX_n)
        x1 = gen_in[mating_pool[i * 2]]
        x2 = gen_in[mating_pool[(i * 2) + 1]]
        xcc1 = 0.5 * (x1 + x2) + 0.5 * betaa * (x1 - x2)
        xc1[0] = xcc1[0]
        xc1[1] = xcc1[1]
        xcc2 = 0.5 * (x1 + x2) + 0.5 * betaa * (x2 - x1)
        xc2[0] = xcc2[0]
        xc2[1] = xcc2[1]
        lambda_gen.append(xc1)
        lambda_gen.append(xc2)
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


def crossover_differential_evolution(gen_in, mu_in, lamb_da_in, obj, DE_K=0.5, DE_F=0.3):
    needed_children = int(lamb_da_in / mu_in)
    lambda_gen = []
    for i in range(mu_in):
        for j in range(needed_children):
            de = np.random.randint(0, 6)
            child = des[de](i, gen_in, mu_in, obj, DE_F, DE_K)
            lambda_gen.append(child)
    return np.array(lambda_gen)


crossovers = {
    0: crossover_whole_arithmetic_x,
    1: crossover_deterministic_arithmetic_x,
    2: crossover_BLX,
    3: crossover_SPX,
    4: crossover_SBX,
    5: crossover_UNDX,
}


def crossover(choice, gen_in, mu_in, lamb_da_in, obj,
              BLX_alpha=0.5, SPX_epsilon=1, SBX_n=2,
              UNDX_sigma_xi=0.8, UNDX_sigma_eta=0.707,
              DE_K=0.5, DE_F=0.3):
    if choice == 6:
        return crossover_differential_evolution(gen_in, mu_in, lamb_da_in, obj, DE_K=DE_K, DE_F=DE_F)
    else:
        return crossovers[choice](gen_in, mu_in, lamb_da_in,
                                  BLX_alpha=BLX_alpha, SPX_epsilon=SPX_epsilon,
                                  SBX_n=SBX_n, UNDX_sigma_xi=UNDX_sigma_xi,
                                  UNDX_sigma_eta=UNDX_sigma_eta)
