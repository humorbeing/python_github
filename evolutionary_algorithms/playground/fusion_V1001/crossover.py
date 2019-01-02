import numpy as np

def UNDX_parents(mu_in, lamb_da_in):
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


def crossover_UNDX(gen_in, mu_in, lamb_da_in, sigma_xi=0.8, sigma_eta=0.707):
    mating_pool = UNDX_parents(mu_in, lamb_da_in)
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
        o = k + sigma_xi * np.random.normal() * d
        o += D * sigma_eta * np.random.normal() * e
        xc[0] = o[0]
        xc[1] = o[1]
        lambda_gen.append(xc)
    return np.array(lambda_gen)