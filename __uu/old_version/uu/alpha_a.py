import numpy as np

# a = np.array([[[1], [0]], [[1], [1]]])
a = np.array([[1], [0]])
b = np.array([[2, 1]])
print(b.shape)
print(a.shape)
print(a)
c = np.dot(b, a)
print(c.shape)
print(c)
