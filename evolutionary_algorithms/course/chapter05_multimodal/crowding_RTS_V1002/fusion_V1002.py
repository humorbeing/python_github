import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import threading
from procedure import *


is_back = False
x_min = -30
x_max = 30
y_min = -30
y_max = 30
w = 1/100
boundary = (x_min, x_max, y_min, y_max)
# print(boundary)
plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
X = np.arange(x_min-2, x_max+2, 0.1)
Y = np.arange(y_min-2, y_max+2, 0.1)
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_xlim(x_min-1, x_max+1)
ax.set_ylabel('Y')
ax.set_ylim(y_min-1, y_max+1)
Z = objective_function([X, Y])
ax.contourf(X, Y, Z, cmap=cm.coolwarm)
red_dot, = ax.plot([], [], 'ko', ms=10)
blue_dot, = ax.plot([], [], 'mo', ms=6)
fig.canvas.draw()
if is_back:
    back = fig.canvas.copy_from_bbox(ax.bbox)

mu = 200
lamb_da = 5
maxgen = 50000

generation = np.array([[x_max+99, 0]])
candidate = np.array([[x_max+99, 0]])


def survive():
    global generation, candidate

    while True:
        generation = initialize(mu, boundary)
        for gen in range(maxgen):
            lamb_da_gen = operate(generation, mu, lamb_da,
                                  boundary, gen, maxgen)
            candidate = nominate(generation, lamb_da_gen)
            time.sleep(w*1)
            fitness = evaluate(candidate, gen, maxgen)
            generation = select(candidate, fitness, mu)
            time.sleep(w*1.4)


t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    if is_back:
        fig.canvas.restore_region(back)
    blue_dot.set_data(candidate.T)
    red_dot.set_data(generation.T)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
