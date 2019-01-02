import numpy as np
import matplotlib.pyplot as plt

x = np.arange(2,10,2)
y = x.copy()
x_ticks_labels = ['jan','feb','mar','apr','may']


# ax = plt.subplots(1,1)
ax = plt.subplot(2,1,1)
ax.plot(x,y)

# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)

ax = plt.subplot(2,1,2)
ax.plot(x,y)

# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)


plt.show()