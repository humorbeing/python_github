import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(1, 100, 20)

plt.subplot(4, 1, 1)
plt.scatter([i for i in range(1, 21)], x)
plt.xlim(0, 21)

plt.subplot(4, 1, 2)
x = sorted(x)
plt.scatter([i for i in range(1, 21)], x)
plt.xlim(0, 21)

plt.subplot(4, 1, 3)
plt.hist(x, 5, (0, 100))

plt.subplot(4, 1, 4)
plt.scatter([i for i in range(1, 100, 5)], x)
plt.hist(x, 5, (0, 100))
plt.show()
