import numpy as np
import matplotlib.pyplot as plt


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


def plotin(x):
    for i in x:
        plt.plot(i, function_f(i), 'ro')


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
    return new_generation


def genotype_to_phenotype(chromosome):
    gene_sum = 0
    for locus in range(l):
        allele = chromosome[locus]
        gene_sum += allele * 2**locus
    return ((gene_sum / (2**l))*(max_x-min_x)) + min_x


def generation_representation():
    representation = []
    for individual_chromosome in generation:
        representation.append(genotype_to_phenotype(individual_chromosome))
    return representation


def evaluation(representations):
    fit = []
    for individual in representations:
        fit.append(function_f(individual))
    return fit


generation = initialization()
representation = generation_representation()
fitness = evaluation(representation)
# test = [1,1,0,1,0,1,1,0,0,1,0,1]
# print(genotype_to_phenotype(test))


x = np.arange(-1.0, 2.0, 0.001)
y = function_f(x)
plt.plot(x, y, lw=2)
plt.plot(representation, fitness, 'ro')
plt.ylim(0, 4)
plt.show()

