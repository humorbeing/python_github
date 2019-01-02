import matplotlib.pyplot as plt

plt.plot(range(10))  # Creates the plot.  No need to save the current figure.
plt.draw()  # Draws, but does not block

plt.figure()  # New window, if needed.  No need to save it, as pyplot uses the concept of current figure
plt.plot(range(10, 20))
plt.draw()
