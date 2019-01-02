import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import RecurrentAttention

# print(RecurrentAttention)
transf = transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
dst = '/media/ray/SSD/workspace/python/dataset/original/mnist'
mnist_trainset = datasets.MNIST(
    dst,
    train=True,
    download=True,
    transform=transf
)
mnist_testset = datasets.MNIST(
    dst,
    train=False,
    download=True,
    transform=transf
)


train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=2,
    # shuffle=True
)
for i, (x, y) in enumerate(train_loader):
    print(x.size())
    print(y)
    # print(type(x))
    x = Variable(x)
    y = Variable(y)
    # print(type(x))
    print(x.size())
    print(x.shape)
    print(x.shape[0])
    break