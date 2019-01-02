from mpl_toolkits.mplot3d import axes3d
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import cm
from time import sleep
# matplotlib.use('GTKAgg')

def function_z(x_in, y_in):
    return (((x_in+50) ** 2 + y_in ** 2) / 4000) - (np.cos(x_in / 6.5) * np.cos(y_in / 7.777)) + 1


plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-30, 30, 0.5)
Y = np.arange(-30, 30, 0.5)
X, Y = np.meshgrid(X, Y)
Z = function_z(X, Y)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

ax.set_xlabel('X')
ax.set_xlim(-30, 30)
ax.set_ylabel('Y')
ax.set_ylim(-30, 30)
ax.set_zlabel('Z')
ax.set_zlim(0, 3.5)
ax.view_init(elev=7, azim=-80)
plt.show()
fig.canvas.draw()
# back = fig.canvas.copy_from_bbox(ax.bbox)

for _ in range(500):
    # fig.canvas.restore_region(back)
    fig.canvas.flush_events()
    sx = np.random.random(50)*60-30
    sy = np.random.random(50)*60-30
    dot1 = ax.scatter(sx, sy, function_z(sx, sy), c='r', marker='o')
    fig.canvas.blit(ax.bbox)
    # fig.canvas.draw()

    # ax.draw_artist(line)
    # plt.draw()
    sleep(0.01)
    sx = np.random.random(50) * 60 - 30
    sy = np.random.random(50) * 60 - 30
    dot2 = ax.scatter(sx, sy, function_z(sx, sy), c='b', marker='o')
    # ax.scatter(sx, sy, 1)
    fig.canvas.blit(ax.bbox)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # plt.draw()
    sleep(0.01)
    dot2.remove()
    dot1.remove()


