import matplotlib.pyplot as plt
import numpy as np


# np.random.seed(0)
x = np.random.randint(1, 100, 20)
# print(a)
# b = np.arange(0, 101, 20)
# print(b)
# a = sorted(a)
# print(a)
# fig, ax = plt.subplots()
# n, bins, patches = ax.hist(a, 5, normed=1)
plt.hist(x, 5, (0, 100))
# print(n)
# print(bins)
# print(patches)
plt.show()
