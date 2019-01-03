from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def function_z(x_in, y_in):
    return ((x_in ** 2 + y_in ** 2) / 4000) - (np.cos(x_in / 6.5) * np.cos(y_in / 7.777)) + 1

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-30, 30, 0.25)
Y = np.arange(-30, 30, 0.25)
X, Y = np.meshgrid(X, Y)
# Z = ((X**2+Y**2)/4000)-(np.cos(X/6.5)*np.cos(Y/7.777))+1
Z = function_z(X, Y)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.2)
sx = np.random.random(50)*60-30
sy = np.random.random(50)*60-30
print(sx)
ax.scatter(sx, sy, function_z(sx, sy), c='r', marker='o')
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-30, 30)
ax.set_ylabel('Y')
ax.set_ylim(-30, 30)
ax.set_zlabel('Z')
ax.set_zlim(0, 2.5)

plt.show()
