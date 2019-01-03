import numpy as np

gen = 0
maxgen = 500

b = 5
U = 5
xt = 3
L = 1


def delta(g, y):
    rand = np.random.random()
    v = y * (1 - rand**((1-g / maxgen)**b))
    return v


def get_x(x, g):
    if np.random.random() < 0.5:
        xn = x + delta(g, (U - x))
    else:
        xn = x - delta(g, (x - L))
    return xn


for _ in range(500):
    print(get_x(xt, gen))
    gen += 1
    if gen == maxgen:
        gen = 0