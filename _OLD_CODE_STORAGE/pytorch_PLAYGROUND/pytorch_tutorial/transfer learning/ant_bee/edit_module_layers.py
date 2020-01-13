import torchvision.models as models
import torch.nn as nn
net = models.resnet18(pretrained=True)
# net.cuda()
net.avgpool = None
# net.BasicBlock.conv1 = None
net.layer4 = None
net.layer3._modules['0']._modules['downsample']._modules['0'] = None


class Net(nn.Module):
    def __init__(self, n):
        # super(Net, self):__init__()
        super(Net, self).__init__()
        n.layer3._modules['0']._modules['downsample']._modules['1'] = None
        self.head = n

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.head(x)
        x = self.conv1(x)
        return x

nnn = Net(net)
nnn.add_module('fc4', nn.Dropout(0.5))
nnn.add_module('bbbbbb', net)

print(nnn)

'''(layer3): Sequential (
    (0): BasicBlock (
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
      (downsample): Sequential (
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
      )
    )
    (1): BasicBlock (
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    )
  )'''