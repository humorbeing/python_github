import numpy as np

trainset = np.load('./simpleset_1per.npy')  # 1 image per class, total 10 images
# trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images


for x, y in trainset:
    print(x)
    print(y)
    break