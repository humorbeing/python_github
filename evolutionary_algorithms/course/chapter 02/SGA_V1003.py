import numpy as np
import matplotlib.pyplot as plt


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


def random_based(probability):
    if np.random.rand() < probability:
        return True
    else:
        return False


popsize = 10
maxgen = 10
Pc = 0.8
Pm = 0.01
l = 12  # chromosome's length
min_x = -1
max_x = 2
generation = []
fitness = []
representation = []


def initialization():
    new_generation = []
    for individual in range(popsize):
        chromosome = [gene for gene in range(l)]
        for locus in range(l):
            if random_based(0.5):
                allele = 1
            else:
                allele = 0
            chromosome[locus] = allele
        new_generation.append(chromosome)
    return np.array(new_generation)


def genotype_to_phenotype(chromosome):
    gene_sum = 0
    for locus in range(l):
        allele = chromosome[locus]
        gene_sum += allele * 2**locus
    return ((gene_sum / (2**l))*(max_x-min_x)) + min_x


def generation_representation():
    representations = []
    for individual_chromosome in generation:
        representations.append(genotype_to_phenotype(individual_chromosome))
    return np.array(representations)


def evaluation():
    fit = []
    for individual in representation:
        fit.append(function_f(individual))
    return fit


def selection():
    pool = []
    P = []  # big 'P', not lower case p
    fitness_sum = sum(fitness)
    # print(fitness)
    # print('sum of fitness: {}'.format(fitness_sum))
    for individual in range(popsize):
        P.append(fitness[individual]/fitness_sum)

    for new_individual in range(popsize):
        P_sum = 0
        winner = np.random.rand()
        for i in range(popsize):
            P_sum += P[i]
            if winner < P_sum:
                pool.append(i)
                break
            else:
                pass
    return np.array(pool)


generation = initialization()
# print(generation.shape)
representation = generation_representation()
# print(representation.shape)
fitness = evaluation()
mating_pool = selection()
print(mating_pool)
for i in mating_pool:
    print([representation[i], fitness[i]])

x = np.arange(min_x, max_x, 0.001)
y = function_f(x)
plt.plot(x, y, lw=2)
plt.plot(representation, fitness, 'ro')
plt.ylim(0, 4)
plt.show()

