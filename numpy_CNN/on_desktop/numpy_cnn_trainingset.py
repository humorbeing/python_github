import numpy as np
from os import chdir, getcwd
print(getcwd())
#chdir(wd)
import os
#abspath = os.path.abspath('') ## String which contains absolute path to the script file
#os.chdir(abspath)
#print(abspath)
trainset = np.load('./simpleset_1per.npy')  # 1 image per class, total 10 images
# trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images


for x, y in trainset:
    print(x)
    print(y)
    break