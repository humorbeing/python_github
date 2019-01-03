import numpy as np


np.random.seed(0)
for i in range(1):
    print(np.random.random())
    print(np.random.randint(1,5))
np.random.seed(1)

for i in range(1):
    print(np.random.random())
np.random.seed()
for i in range(1):
    print(np.random.random())