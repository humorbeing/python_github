import numpy as np


def mutation_normal(lambda_gen_in, sigma=0.5):
    lambda_gen = []
    for i in lambda_gen_in:
        x = [0, 0]
        for j in range(len(i)):
            x[j] = i[j] + sigma * np.random.normal()
        lambda_gen.append(x)
    return np.array(lambda_gen)
