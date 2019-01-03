import numpy as np
import matplotlib.pyplot as plt


def function_f(x):
    return np.sin(10*np.pi*x)*x+2.0


def plotin(x):
    for i in x:
        plt.plot(i, function_f(i), 'ro')

# ax = plt.subplot(111)
x = np.arange(-1.0, 2.0, 0.001)
y = function_f(x)
plt.plot(x, y, lw=2)
plotin([1, 2, 1.7, 1.65])
plt.ylim(0, 4)
plt.show()

