import matplotlib.pyplot as plt

plt.plot([1,2,3],[5,7,4])
plt.draw()

plt.figure()  # New window, if needed.  No need to save it, as pyplot uses the concept of current figure
plt.plot(range(10, 20))
plt.draw()
plt.show()
