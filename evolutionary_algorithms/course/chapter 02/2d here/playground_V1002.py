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


plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
x = np.arange(min_x, max_x, 0.001)
y = objective_function(x)
ax.set_xlabel('X')
ax.set_xlim(min_x, max_x)
ax.set_ylabel('Y')
ax.set_ylim(min_y, max_y)
ax.plot(x, y, lw=2)
red_dot, = ax.plot([], [], 'ro')
blue_dot, = ax.plot([], [], 'bo')
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)

sx = np.random.random(50) * (max_x - min_x) + min_x
sy = objective_function(sx)
ox = sx
oy = sy


def dothis():
    global sx, sy, ox, oy
    w = 1/2
    for _ in range(5000):
        time.sleep(w*2)
        ox = sx
        oy = sy
        sx = []
        sy = []
        time.sleep(w)
        # ox = []
        # oy = []
        sx = np.random.random(50) * (max_x - min_x) + min_x
        sy = objective_function(sx)


t = threading.Thread(target=dothis)
t.daemon = True
t.start()


while True:
    fig.canvas.restore_region(back)
    blue_dot.set_data(ox, oy)
    red_dot.set_data(sx, sy)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()




