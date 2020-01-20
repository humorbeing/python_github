from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

#xs = [1,2,3,4,5,6]
#ys = [5,4,6,5,6,7]
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)
#plt.plot(xs,ys)
#plt.scatter(xs,ys)
#plt.show()

def best_fit_slopes(xs,ys):
    m = (  ( (mean(xs) * mean(ys)) - mean(xs*ys) ) /
           ( (mean(xs)*mean(xs))   - mean(xs**2) )   )
    #x^2 doesn't works, x**2 works or x * x
    return m

m = best_fit_slopes(xs,ys)

print(m)
