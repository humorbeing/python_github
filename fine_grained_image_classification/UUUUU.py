
# pytorch_playground
import torch
# from torch.autograd_playground import Variable
import torch.nn as nn
import torch.optim as optim
# torchvision
import torchvision
# from torchvision import datasets
# from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import transforms
# more
# import numpy as np
# import copy
import time
# me
from package import me

# Hyper-parameter
valid_set_size = 0.5
batch_size = 128
Epoch = 50
decay = 10
learning_rate = 1
#####################

since = time.time()
# dataset_path = '/media/ray/SSD/workspace/python/dataset/trainset_testset/fish_dataset_train_test_split'
dataset_path = '/media/ray/SSD/workspace/python/dataset/trainset_testset/bird_dataset_train_test_split'
trainset_path = dataset_path + '/train'
testset_path = dataset_path + '/test'
gpu = torch.cuda.is_available()


dataset_loaders, dataset_sizes, dataset_classes\
    = me.train_set_test_set(
    trainset_path,
    batch_size=batch_size,
    val_set_ratio=valid_set_size
    # random_seed=169
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
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=0.9)

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

optimizer_conv = optim.SGD(model_conv.parameters(), lr=learning_rate, momentum=0.9)

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
del dataset_loaders, dataset_sizes
me.test_model(
    model_conv,
    testset_path,
    batch_size,
    gpu
)


time_elapsed = time.time() - since
print(
    'Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60
    )
)









