import numpy as np

a = [2, 2]
a = a[0:500000]
print(a)
print(*a)
# print(np.random.randn(a)) # error
print(np.random.randn(*a))
a = [1, 2, 3]
b = np.array(a)
c = [2, 1, 0] # np.random.permutation(x)
# print(a[c]) # error
print(b[c])
print(b[np.array(c)])

a = np.zeros((2,3))
b = np.zeros((2,3))

print(a)
print(a[:])
a[:] = 6
a[:] = [6,2,1]
# a[:] = [6,2]  # error
b = 6
print(b)
print(a)