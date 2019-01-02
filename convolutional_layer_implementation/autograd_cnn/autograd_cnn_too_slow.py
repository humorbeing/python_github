import torch
from torch.autograd import Variable
import numpy as np
from torchvision import datasets
# import sys
# sys.path.insert(0, './')
from package import mod

# Hyper Parameter
EPOCH = 100


# mnist = datasets.MNIST('../storage/mnist/', download=True)
train_set = datasets.MNIST('../storage/mnist/', train=True)
# test_set = datasets.MNIST('../storage/mnist/', train=False)

adjust = 0.01

K1 = np.random.random(size=(5, 1, 5, 5)) * adjust
K2 = np.random.random(size=(10, 5, 5, 5)) * adjust
W1 = np.random.random(size=(720, 10)) * adjust
W2 = np.random.random(size=(100, 10)) * adjust

B_K1 = np.zeros(shape=(K1.shape[0], 1))
B_K2 = np.zeros(shape=(K2.shape[0], 1))
B_W1 = np.zeros(shape=(1, W1.shape[1]))
B_W2 = np.zeros(shape=(1, W2.shape[1]))

K1 = Variable(torch.from_numpy(K1).type(torch.FloatTensor),
              requires_grad=True)
W1 = Variable(torch.from_numpy(W1).type(torch.FloatTensor),
              requires_grad=True)
B_K1 = Variable(torch.from_numpy(B_K1).type(torch.FloatTensor),
                requires_grad=True)
B_W1 = Variable(torch.from_numpy(B_W1).type(torch.FloatTensor),
                requires_grad=True)


def s(show_this):
    print(type(show_this.data))
    print(type(show_this))
    print(show_this.size())

learning_rate = 0.005
runs = 10
onr = 0
for epoch in range(EPOCH):
    for case, (image, label) in enumerate(train_set):

        if onr < runs:
            onr += 1
            O = mod.normalize_image(image)
            O = mod.convolution(O, K1)
            O = mod.conv_bias(O, B_K1)
            O = mod.relu(O)
            O = mod.max_pooling(O)
            O = mod.tensor_vector(O)
            O = torch.matmul(O, W1)
            O = mod.MLP_bias(O, B_W1)
            O = mod.softmax(O)
            O = mod.cross_entropy(O, label)
            print(O.data.numpy()[0])
            O.backward()
            K1t = K1 - learning_rate * K1.grad
            K1 = Variable(K1t.data, requires_grad=True)

            W1t = W1 - learning_rate * W1.grad
            W1 = Variable(W1t.data, requires_grad=True)

            B_K1t = B_K1 - learning_rate * B_K1.grad
            B_K1 = Variable(B_K1t.data, requires_grad=True)

            B_W1t = B_W1 - learning_rate * B_W1.grad
            B_W1 = Variable(B_W1t.data, requires_grad=True)

        else:
            onr = 0
            print()
            break




