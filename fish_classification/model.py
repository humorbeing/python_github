import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.con1 = nn.Conv2d(3, 16, 5, stride=2)
        self.con2 = nn.Conv2d(16, 32, 5, stride=2)
        self.con3 = nn.Conv2d(32, 32, 5, stride=2)
        self.con4 = nn.Conv2d(32, 32, 5, stride=2)
        self.con5 = nn.Conv2d(32, 32, 5, stride=1)
        self.lin1 = nn.Linear(1568, 770)
        self.lin2 = nn.Linear(770, num_class)

    def forward(self, x):
        x = self.con1(x)
        x = F.relu(x)
        x = self.con2(x)
        x = F.relu(x)
        x = self.con3(x)
        x = F.relu(x)
        x = self.con4(x)
        x = F.relu(x)
        x = self.con5(x)
        x = F.relu(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.softmax(x, dim=1)
        print(x.shape)
        return x

if __name__ == "__main__":
    model = CNN(5)
    x = torch.randn((1,3,224,224))
    x1 = model(x)


    print(x1.shape)
