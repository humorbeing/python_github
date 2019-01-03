from torchvision import datasets
import numpy as np
import sys
sys.path.insert(0, '../')
from package.mod import normalize_image
from package.mod import conv
from package.mod import relu
from package.mod import conv_bias

# mnist = datasets.MNIST('../storage/mnist/', download=True)
train_set = datasets.MNIST('../storage/mnist/', train=True)
# test_set = datasets.MNIST('../storage/mnist/', train=False)

adjust = 0.01

K1 = np.random.random(size=(5, 1, 5, 5)) * adjust
K2 = np.random.random(size=(10, 5, 5, 5)) * adjust
W1 = np.random.random(size=(1000, 100)) * adjust
W2 = np.random.random(size=(100, 10)) * adjust

B_K1 = np.zeros(shape=(K1.shape[0], 1))
B_K2 = np.zeros(shape=(K2.shape[0], 1))
B_W1 = np.zeros(shape=(1, W1.shape[1]))
B_W2 = np.zeros(shape=(1, W2.shape[1]))


EPOCH = 100

for epoch in range(EPOCH):
    for case, (image, label) in enumerate(train_set):
        Image = normalize_image(image)
        O = conv(Image, K1)  # U1
        O = conv_bias(O, B_K1)
        O, mf1 = relu(O)  # C1
        C1 = O
        O = conv(O, K2)  # U2
        O = conv_bias(O, B_K2)
        break
    break