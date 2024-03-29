import numpy as np
import matplotlib.pyplot as plt
import time
import threading


def objective_function(x_in):
    return np.sin(10*np.pi*x_in)*x_in+2.0


min_x = -1
max_x = 2
min_y = 0
max_y = 4
w = 1 / 3

reset = w * 7

plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
x = np.arange(min_x, max_x+0.01, 0.005)
y = objective_function(x)
ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
ax.plot(x, y, lw=2, c='y')
red_dot, = ax.plot([], [], 'ro', ms=10)
blue_dot, = ax.plot([], [], 'bo', ms=6)

fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)


def random_based(probability):
    if np.random.rand() < probability:
        return True
    else:
        return False


popsize = 30
maxgen = 20
# Pc = 0.5
l = 12  # chromosome's length
# Pm = 1 / l
Pm = 0.01

generation = []
fitness = []
representation = []
mating_pool = []
old = []


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


def evaluation(rep):
    fit = []
    for individual in rep:
        fit.append(objective_function(individual))
    return fit


def selection():
    pool = []
    P = []  # big 'P', not lower case p
    fitness_sum = sum(fitness)
    # print('fitness',fitness)
    # print('sum of fitness: {}'.format(fitness_sum))
    for individual in range(popsize):
        P.append(fitness[individual]/fitness_sum)

    P_sum = 0
    portion = 1 / popsize
    winner = np.random.rand() * portion

    # print(P)
    P_sum += P[0]
    i = 0
    while len(pool) < popsize:
        # print('winner:',winner)
        # print('P_sum:',P_sum)
        if winner < P_sum:
            pool.append(i)
            winner += portion
        else:
            i += 1
            P_sum += P[i]
        # print('pool',pool)


    np.random.shuffle(pool)

    return np.array(pool)


def variation_operation():
    new_generation = []
    for parent in range(int(popsize/2)):
        child_1 = generation[mating_pool[parent*2]]
        child_2 = generation[mating_pool[parent*2+1]]
        Pc = objective_function(representation[mating_pool[parent*2]])
        Pc = Pc / (objective_function(representation[mating_pool[parent*2]])
                   + objective_function(representation[mating_pool[parent*2+1]]))
        # print(Pc)
        for locus in range(l):
            if random_based(Pc):
                child_1[locus], child_2[locus] = child_2[locus], child_1[locus]
        for locus in range(l):
            if random_based(Pm):
                child_1[locus] = 1 if child_1[locus] == 0 else 0
            if random_based(Pm):
                child_2[locus] = 1 if child_2[locus] == 0 else 0
        new_generation.append(child_1)
        new_generation.append(child_2)
    return np.array(new_generation)


def survive():
    global generation, fitness, representation, mating_pool, old
    while True:
        generation = initialization()
        for _ in range(maxgen):
            # time.sleep(w)
            old = representation
            time.sleep(w)
            representation = generation_representation()
            fitness = evaluation(representation)
            # time.sleep(w)
            mating_pool = selection()
            generation = variation_operation()
            # old = [min_x-1]
            time.sleep(w*2)
        representation = [min_x-1]
        old = [min_x-1]
        time.sleep(reset)


t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    blue_dot.set_data(old, evaluation(old))
    red_dot.set_data(representation, fitness)
    fig.canvas.blit(ax.bbox)
    # fig.canvas.draw()
    fig.canvas.flush_events()




