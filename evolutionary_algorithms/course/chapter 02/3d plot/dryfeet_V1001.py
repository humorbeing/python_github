from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-30, 30, 0.25)
Y = np.arange(-30, 30, 0.25)
X, Y = np.meshgrid(X, Y)
Z = ((X**2+Y**2)/4000)-(np.cos(X)*np.cos(Y/np.sqrt(2)))+1
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
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
