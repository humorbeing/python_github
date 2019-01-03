from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import threading

# bgrcmyk o+*>
def objective_function(point_in):
    x_in = point_in[0]
    y_in = point_in[1]
    return (((x_in + 50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


min_x = -30
max_x = 30
min_y = -30
max_y = 30
w = 1/3
reset = w * 5

plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
X = np.arange(min_x-2, max_x+2, 0.1)
Y = np.arange(min_y-2, max_y+2, 0.1)
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
Z = objective_function([X, Y])
# ax.contourf(X, Y, Z, cmap=cm.jet)
ax.contourf(X, Y, Z, cmap=cm.coolwarm)  # *****
# ax.contourf(X, Y, Z, cmap=cm.gray)  # ***
# ax.contourf(X, Y, Z, cmap=cm.Greys)
# ax.contourf(X, Y, Z, cmap=cm.BuPu)
# ax.contourf(X, Y, Z, cmap=cm.viridis)  # *****
# ax.contourf(X, Y, Z, cmap=cm.plasma)
# ax.contourf(X, Y, Z, cmap=cm.inferno)  # ***
# ax.contourf(X, Y, Z, cmap=cm.magma)
red_dot, = ax.plot([], [], 'ro', ms=5)
blue_dot, = ax.plot([], [], 'bo', ms=5)
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)

sx = np.random.random(50) * (max_x - min_x) + min_x
sy = np.random.random(50) * (max_y - min_y) + min_y
ox = sx
oy = sy


def survive():
    global sx, sy, ox, oy
    for _ in range(5000):
        time.sleep(0.01)
        ox = sx
        oy = sy
        sx = np.random.random(50) * (max_x - min_x) + min_x
        sy = np.random.random(50) * (max_y - min_y) + min_y


t = threading.Thread(target=survive)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    blue_dot.set_data([ox, oy])
    red_dot.set_data([sx, sy])
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




