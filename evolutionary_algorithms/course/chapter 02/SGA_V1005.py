import numpy as np
import matplotlib.pyplot as plt

plt.ion()

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
mating_pool = []


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


def variation_operation():
    new_generation = []
    for parent in range(int(popsize/2)):
        child_1 = generation[mating_pool[parent*2]]
        child_2 = generation[mating_pool[parent*2+1]]
        if random_based(Pc):
            random_cross_place = np.random.randint(1, l-1)
            for locus in range(l):
                if locus >= random_cross_place:
                    child_1[locus], child_2[locus] = child_2[locus], child_1[locus]
        else:
            pass
        for locus in range(l):
            if random_based(Pm):
                child_1[locus] = 1 if child_1[locus] == 0 else 0
            if random_based(Pm):
                child_2[locus] = 1 if child_2[locus] == 0 else 0
        new_generation.append(child_1)
        new_generation.append(child_2)
    return np.array(new_generation)

x = np.arange(min_x, max_x, 0.001)
y = function_f(x)
figure, ax = plt.subplots()
ax.plot(x, y, lw=2)
# ax.ylim(0, 4)
lines, = ax.plot([], [], 'ro')
figure.canvas.draw()
figure.canvas.flush_events()

generation = initialization()
# print(generation.shape)
for gens in range(maxgen):

    representation = generation_representation()
    fitness = evaluation()
    mating_pool = selection()
    input()
    lines.set_xdata(representation)
    lines.set_ydata(fitness)
    figure.canvas.draw()
    figure.canvas.flush_events()
    generation = variation_operation()


# plt.plot(x, y, lw=2)
# plt.plot(representation, fitness, 'ro')
# plt.ylim(0, 4)
# plt.show()

