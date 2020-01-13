import numpy as np

#print (False or True or False)
L = [[] for i in range(5)]

L[0] = [[5,5]]
L[2] = 3
L[4] = 7
L[3] = np.random.random((4,5))

print(len(L))
print(L)
print(L[3])
