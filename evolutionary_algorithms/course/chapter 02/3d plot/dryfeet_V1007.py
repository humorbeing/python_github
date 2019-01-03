from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from time import sleep


def function_z(x_in, y_in):
    return (((x_in+50) ** 2 + (y_in + 15) ** 2) / 4000) - (np.cos(x_in / 4) * np.cos(y_in / 4)) + 1


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
def handle_close(evt):
    print('close me')
    plt.ioff()
    plt.close('all')
fig.canvas.mpl_connect('close_event', handle_close)
X = np.arange(-30, 30, 0.1)
Y = np.arange(-30, 30, 0.1)
X, Y = np.meshgrid(X, Y)
Z = function_z(X, Y)
cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
ax.set_xlabel('X')
ax.set_xlim(-30, 30)
ax.set_ylabel('Y')
ax.set_ylim(-30, 30)
dot, = ax.plot([], [], 'ro')
plt.show()
fig.canvas.draw()
back = fig.canvas.copy_from_bbox(ax.bbox)

for _ in range(500):
    fig.canvas.restore_region(back)
    # fig.canvas.flush_events()
    sx = np.random.random(50)*60-30
    sy = np.random.random(50)*60-30
    # dot1 = ax.scatter(sx, sy, c='r', marker='o')
    dot.set_data(sx, sy)
    # dot.set_ydata(sy)
    # fig.canvas.blit(ax.bbox)
    # fig.canvas.draw()

    # ax.draw_artist(dot)
    # plt.draw()
    # sleep(0.01)
    # sx = np.random.random(50) * 60 - 30
    # sy = np.random.random(50) * 60 - 30
    # dot2 = ax.scatter(sx, sy, c='b', marker='o')
    # # ax.scatter(sx, sy, 1)
    fig.canvas.blit(ax.bbox)
    # # fig.canvas.draw()
    # fig.canvas.update()
    fig.canvas.flush_events()
    # # plt.draw()
    # sleep(0.03)
    # dot2.remove()
    # dot1.remove()


# plt.close('all')