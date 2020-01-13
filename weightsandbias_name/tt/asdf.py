# n202 ubuntu: /mnt/D8442D91442D7382/Mystuff/Workspace/python_world/Venv/3.5/bin/wandb
# /mnt/D8442D91442D7382/Mystuff/Workspace/python_world/Venv/3.5/bin/wandb login a6f5079f5d5476735d22bac595bb76c5aa1cb369
# this will creat wandb folder inside this folder
import wandb
wandb.init()

import numpy as np
import torch
from argparse import Namespace
args = Namespace()
args.hi = 'hi'
wandb.config.update(args)

import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        return x
model = Net()  # define model, to(device), optimizer
wandb.watch(model)


image1 = np.random.random((28, 28))
x = torch.Tensor([[image1]])
z = model(x)
loss = torch.mean(z)
loss.backward()

image2 = torch.Tensor(np.random.random((3, 64, 64)))
# image3 = torch.Tensor(np.random.random((64, 64, 3)))  # wrong due to pytorch
image3 = np.random.random((64, 64, 3))
wandb_image1 = wandb.Image(image1, caption="1")
wandb_image2 = wandb.Image(image2, caption="2")
wandb_image3 = wandb.Image(image3, caption="2")
example_images = [wandb_image1, wandb_image2, wandb_image3]

wandb.log({
        "image1": image1,
        "wandb_image1": wandb_image1,
        "wandb_image2": wandb_image2,
        "wandb_image3": wandb_image3,
        "list of w i": example_images,
        "Some Number": 99,
        "trigger no loss": 5})

print('check output logs')