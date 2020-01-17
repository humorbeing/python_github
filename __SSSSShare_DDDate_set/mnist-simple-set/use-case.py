import numpy as np


trainset = np.load('./simpleset_1per.npy')
# trainset = np.load('./simpleset_10per.npy')


print(trainset.shape)
for x, y in trainset:
    print(x.shape)
    print(y)
    break