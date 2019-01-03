import numpy as np

a = [1, 2]
b = [2, 3]
c = np.random.uniform(size=2)
print(np.multiply(a, b))
print(np.subtract(1, c))


d = np.multiply(a, c) + np.multiply(b, np.subtract(1, c))

print(d)
for _ in range(500):
    print(np.random.uniform(3,3))