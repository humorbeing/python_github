
# pytorch_playground
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# torchvision
import torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
# more
import numpy as np
import copy
import time
# me
from package import me

# Hyper-parameter
test_set_size = 0.3
batch_size = 128
Epoch = 50
decay = 10
learning_rate = 0.01
#####################

since = time.time()
dataset_path = '/media/ray/SSD/workspace/python/dataset/fish/original'
gpu = torch.cuda.is_available()


dataset_loaders,test_loader,\
dataset_sizes, dataset_classes\
    = me.train_set_test_set(
    dataset_path,
    test_set_size,
    batch_size=batch_size,
    random_seed=169
)


model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
class_size = len(dataset_classes)
model_conv.fc = nn.Linear(num_ftrs, class_size)
if gpu:
    model_conv = model_conv.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)

model_conv = me.train_model(
    model_conv,
    dataset_loaders,
    dataset_sizes,
    criterion,
    optimizer_conv,
    learning_rate,
    me.exp_lr_scheduler,
    decay,
    gpu=gpu,
    num_epochs=Epoch
)
for param in model_conv.parameters():
    param.requires_grad = True

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

model_conv = me.train_model(
    model_conv,
    dataset_loaders,
    dataset_sizes,
    criterion,
    optimizer_conv,
    learning_rate,
    me.exp_lr_scheduler,
    decay,
    gpu=gpu,
    num_epochs=Epoch
)
me.test_model(
    model_conv,
    test_loader,
    dataset_sizes,
    batch_size,
    gpu
)
time_elapsed = time.time() - since
print(
    'Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60
    )
)









