import numpy as np
import os

print(os.getcwd())
trainset = np.load('./numpy_CNN/simpleset_1per.npy')  # 1 image per class, total 10 images
# trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images


for x, y in trainset:
    print(x)
    print(y)
    break