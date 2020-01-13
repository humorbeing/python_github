import wandb
wandb.init()
from argparse import Namespace
args = Namespace()
args.hi = 'hi'
wandb.config.update(args)

import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
model = Net()  # define model, to(device), optimizer
# import torch.optim as optim
# op = optim.Adam(model.parameters())
wandb.watch(model)
# model.train()

# mod
import numpy as np
image1 = np.random.random((28, 28))
image2 = np.random.random((64, 64))
wi = wandb.Image(image1, caption="1")
wi2 = wandb.Image(image2, caption="2")
example_images = []
example_images.append(wi)
example_images.append(wi2)
import torch

im = torch.Tensor([[image1]])
# model.train()
# op.zero_grad()
z = model(im)
l = torch.mean(z)
l.backward()
# import torch.nn.functional as F
# loss = F.mse_loss(z, im)
# loss.backward()
# op.step()
# model.eval()
# z = model(im)
wandb.log({
        "image": image1,
        "wandb image": wi,
        "list of w i": example_images,
        "what i like": 99,
        "what is next": 5})