from torchvision import datasets
import numpy as np

mnist_train_set_data = datasets.MNIST('../storage/mnist/', train=True, download=False)
mnist_test_set_data = datasets.MNIST('../storage/mnist/', train=False, download=False)
print(len(mnist_train_set_data))
print(len(mnist_test_set_data))
for i, l in mnist_train_set_data:
    print(i)
    a = np.array(i)
    a = np.array([a])
    print(a/255)
    print(a.shape)
    print(l)
    break