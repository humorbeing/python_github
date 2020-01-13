import matplotlib.pyplot as plt
import numpy as np
n = 100  # number of random samples
x = np.random.randint(1, 3, n)  # draw n random numbers, between 1 and 100.
# plt.subplot(2, 1, 1)  # (2, 1, 1) means: divide plt into 2x1 matrix blocks, and draw on 1st block
plt.scatter([i for i in range(1, n+1)], x)  # assign n numbers to n random samples.
plt.xlim(0, n+1)  # limit x axis from 0 to n+1.
# plt.subplot(2, 1, 2)  # (2, 1, 2) means: divide plt into 2x1 matrix blocks, and draw on 2nd block
# plt.hist(x, 5, (0, 100))  # x is random number list. 5 means 5 bars. (0, 100) means limit x axis from 0 to 100.
plt.show()

