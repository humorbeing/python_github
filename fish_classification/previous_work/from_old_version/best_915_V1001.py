import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import time
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import copy


data_dir = 'for_learning'
test_dir = 'for_testing'
valid_size = 0.12
batch_size = 100
num_workers = 2
EPOCH = 150
# transformer + dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
valid_transform = transforms.Compose([
    transforms.Scale(224),
    torchvision.transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
train_transform = transforms.Compose([
    transforms.Scale(224),
    torchvision.transforms.CenterCrop(224),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_dataset = datasets.ImageFolder(data_dir, train_transform)
val_dataset = datasets.ImageFolder(data_dir, valid_transform)

# making train val data loader
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.seed(123)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)

dset_loaders = {'train': train_loader,
                'val': valid_loader}
dset_classes = train_dataset.classes
class_size = len(dset_classes)
dset_sizes = {'train': len(train_idx),
              'val': len(valid_idx)
              }
gpu = torch.cuda.is_available()
if gpu:
    gpu = True  # if model is too big for gpu, then false


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# model_conv = torchvision.models.resnet152(pretrained=True)
# print('---Training last layer.---')
#
# # gpu 9G memory
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, class_size)
# if gpu:
#     model_conv = model_conv.cuda()
#
# criterion = nn.CrossEntropyLoss()
#
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)
#
# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=EPOCH)
# print()
# print('---Fine tuning.---')
# # gpu 10G memory
# for param in model_conv.parameters():
#     param.requires_grad = True
#
# optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
#
# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=EPOCH)


model_conv = torchvision.models.resnet18(pretrained=True)

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, class_size)
if gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=EPOCH)
save_name = 'train.save'
torch.save(
    {'model': model_conv,
     'dset_classes': dset_classes
     },
    save_name)
print()
print('Model saved in "'+save_name+'".')

test_dataset = datasets.ImageFolder(test_dir, valid_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
running_loss = 0.0
running_corrects = 0
model_conv.train(False)
print()
print('---Testing---')
for data in test_loader:
    inputs, labels = data

    if gpu:
        inputs, labels = Variable(inputs.cuda()), \
            Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model_conv(inputs)
    _, preds = torch.max(outputs.data, 1)
    loss = criterion(outputs, labels)
    running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects / len(test_dataset)

print("Test accuracy: {:4f}".format(test_acc))
'''/usr/bin/python3.5 /home/visbic/python/pytorch/datasets/fish/train_val_dataloader_V1001.py
Epoch 0/149
----------
LR is set to 0.01
train Loss: 0.0112 Acc: 0.6451
val Loss: 0.0066 Acc: 0.7937

Epoch 1/149
----------
train Loss: 0.0036 Acc: 0.8746
val Loss: 0.0043 Acc: 0.8619

Epoch 2/149
----------
train Loss: 0.0019 Acc: 0.9363
val Loss: 0.0037 Acc: 0.8950

Epoch 3/149
----------
train Loss: 0.0011 Acc: 0.9619
val Loss: 0.0036 Acc: 0.8692

Epoch 4/149
----------
train Loss: 0.0007 Acc: 0.9777
val Loss: 0.0032 Acc: 0.8895

Epoch 5/149
----------
train Loss: 0.0005 Acc: 0.9787
val Loss: 0.0033 Acc: 0.8969

Epoch 6/149
----------
train Loss: 0.0005 Acc: 0.9832
val Loss: 0.0039 Acc: 0.9042

Epoch 7/149
----------
train Loss: 0.0004 Acc: 0.9799
val Loss: 0.0037 Acc: 0.8987

Epoch 8/149
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0035 Acc: 0.9061

Epoch 9/149
----------
train Loss: 0.0003 Acc: 0.9852
val Loss: 0.0036 Acc: 0.9134

Epoch 10/149
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9839
val Loss: 0.0040 Acc: 0.9061

Epoch 11/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0036 Acc: 0.9024

Epoch 12/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0038 Acc: 0.9042

Epoch 13/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0035 Acc: 0.9024

Epoch 14/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0036 Acc: 0.9042

Epoch 15/149
----------
train Loss: 0.0002 Acc: 0.9850
val Loss: 0.0038 Acc: 0.9042

Epoch 16/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0037 Acc: 0.9042

Epoch 17/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0040 Acc: 0.9061

Epoch 18/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0041 Acc: 0.9042

Epoch 19/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0037 Acc: 0.9061

Epoch 20/149
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0039 Acc: 0.9042

Epoch 21/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0042 Acc: 0.9042

Epoch 22/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0040 Acc: 0.9061

Epoch 23/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0038 Acc: 0.9061

Epoch 24/149
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0036 Acc: 0.9042

Epoch 25/149
----------
train Loss: 0.0002 Acc: 0.9855
val Loss: 0.0037 Acc: 0.9079

Epoch 26/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0038 Acc: 0.9079

Epoch 27/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0041 Acc: 0.9061

Epoch 28/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0041 Acc: 0.9042

Epoch 29/149
----------
train Loss: 0.0002 Acc: 0.9855
val Loss: 0.0037 Acc: 0.9042

Epoch 30/149
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9079

Epoch 31/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0037 Acc: 0.9042

Epoch 32/149
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0039 Acc: 0.9042

Epoch 33/149
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0040 Acc: 0.9042

Epoch 34/149
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0039 Acc: 0.9042

Epoch 35/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0036 Acc: 0.9061

Epoch 36/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0037 Acc: 0.9061

Epoch 37/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0038 Acc: 0.9061

Epoch 38/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9042

Epoch 39/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0041 Acc: 0.9042

Epoch 40/149
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9061

Epoch 41/149
----------
train Loss: 0.0002 Acc: 0.9837
val Loss: 0.0040 Acc: 0.9061

Epoch 42/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0040 Acc: 0.9061

Epoch 43/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0038 Acc: 0.9061

Epoch 44/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0038 Acc: 0.9079

Epoch 45/149
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0040 Acc: 0.9042

Epoch 46/149
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0038 Acc: 0.9061

Epoch 47/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0037 Acc: 0.9061

Epoch 48/149
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0040 Acc: 0.9061

Epoch 49/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0040 Acc: 0.9061

Epoch 50/149
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0037 Acc: 0.9061

Epoch 51/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0040 Acc: 0.9042

Epoch 52/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0036 Acc: 0.9061

Epoch 53/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0040 Acc: 0.9042

Epoch 54/149
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0037 Acc: 0.9079

Epoch 55/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0042 Acc: 0.9042

Epoch 56/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0040 Acc: 0.9061

Epoch 57/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0037 Acc: 0.9061

Epoch 58/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0038 Acc: 0.9061

Epoch 59/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0038 Acc: 0.9061

Epoch 60/149
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0042 Acc: 0.9061

Epoch 61/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0037 Acc: 0.9042

Epoch 62/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0042 Acc: 0.9042

Epoch 63/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9061

Epoch 64/149
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9042

Epoch 65/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0042 Acc: 0.9061

Epoch 66/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0042 Acc: 0.9042

Epoch 67/149
----------
train Loss: 0.0002 Acc: 0.9827
val Loss: 0.0041 Acc: 0.9042

Epoch 68/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0042 Acc: 0.9061

Epoch 69/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0040 Acc: 0.9061

Epoch 70/149
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9061

Epoch 71/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0042 Acc: 0.9061

Epoch 72/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0040 Acc: 0.9061

Epoch 73/149
----------
train Loss: 0.0002 Acc: 0.9842
val Loss: 0.0038 Acc: 0.9042

Epoch 74/149
----------
train Loss: 0.0002 Acc: 0.9855
val Loss: 0.0039 Acc: 0.9061

Epoch 75/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0038 Acc: 0.9061

Epoch 76/149
----------
train Loss: 0.0002 Acc: 0.9850
val Loss: 0.0038 Acc: 0.9061

Epoch 77/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9042

Epoch 78/149
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0039 Acc: 0.9061

Epoch 79/149
----------
train Loss: 0.0002 Acc: 0.9850
val Loss: 0.0037 Acc: 0.9061

Epoch 80/149
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9061

Epoch 81/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0040 Acc: 0.9061

Epoch 82/149
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0041 Acc: 0.9042

Epoch 83/149
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0037 Acc: 0.9061

Epoch 84/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0039 Acc: 0.9061

Epoch 85/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0038 Acc: 0.9042

Epoch 86/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0038 Acc: 0.9061

Epoch 87/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0038 Acc: 0.9061

Epoch 88/149
----------
train Loss: 0.0002 Acc: 0.9855
val Loss: 0.0037 Acc: 0.9042

Epoch 89/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9061

Epoch 90/149
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0039 Acc: 0.9042

Epoch 91/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0041 Acc: 0.9061

Epoch 92/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0045 Acc: 0.9061

Epoch 93/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0036 Acc: 0.9061

Epoch 94/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9061

Epoch 95/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0037 Acc: 0.9042

Epoch 96/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0040 Acc: 0.9061

Epoch 97/149
----------
train Loss: 0.0002 Acc: 0.9850
val Loss: 0.0042 Acc: 0.9061

Epoch 98/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0042 Acc: 0.9042

Epoch 99/149
----------
train Loss: 0.0002 Acc: 0.9852
val Loss: 0.0041 Acc: 0.9061

Epoch 100/149
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0038 Acc: 0.9061

Epoch 101/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0036 Acc: 0.9079

Epoch 102/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0038 Acc: 0.9042

Epoch 103/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0040 Acc: 0.9079

Epoch 104/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9061

Epoch 105/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0036 Acc: 0.9042

Epoch 106/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0038 Acc: 0.9079

Epoch 107/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9061

Epoch 108/149
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0037 Acc: 0.9061

Epoch 109/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0040 Acc: 0.9061

Epoch 110/149
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0042 Acc: 0.9024

Epoch 111/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9061

Epoch 112/149
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0038 Acc: 0.9061

Epoch 113/149
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0038 Acc: 0.9079

Epoch 114/149
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0044 Acc: 0.9079

Epoch 115/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0038 Acc: 0.9079

Epoch 116/149
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0041 Acc: 0.9042

Epoch 117/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0038 Acc: 0.9042

Epoch 118/149
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0040 Acc: 0.9061

Epoch 119/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0038 Acc: 0.9042

Epoch 120/149
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0036 Acc: 0.9079

Epoch 121/149
----------
train Loss: 0.0002 Acc: 0.9842
val Loss: 0.0039 Acc: 0.9061

Epoch 122/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0041 Acc: 0.9079

Epoch 123/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0038 Acc: 0.9061

Epoch 124/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0038 Acc: 0.9061

Epoch 125/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0037 Acc: 0.9042

Epoch 126/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0040 Acc: 0.9061

Epoch 127/149
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0039 Acc: 0.9042

Epoch 128/149
----------
train Loss: 0.0002 Acc: 0.9865
val Loss: 0.0042 Acc: 0.9042

Epoch 129/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0041 Acc: 0.9061

Epoch 130/149
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0039 Acc: 0.9061

Epoch 131/149
----------
train Loss: 0.0002 Acc: 0.9852
val Loss: 0.0041 Acc: 0.9042

Epoch 132/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0036 Acc: 0.9042

Epoch 133/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0044 Acc: 0.9042

Epoch 134/149
----------
train Loss: 0.0002 Acc: 0.9857
val Loss: 0.0040 Acc: 0.9061

Epoch 135/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0037 Acc: 0.9024

Epoch 136/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0042 Acc: 0.9061

Epoch 137/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0037 Acc: 0.9061

Epoch 138/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0039 Acc: 0.9042

Epoch 139/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9042

Epoch 140/149
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0002 Acc: 0.9847
val Loss: 0.0040 Acc: 0.9042

Epoch 141/149
----------
train Loss: 0.0002 Acc: 0.9890
val Loss: 0.0038 Acc: 0.9061

Epoch 142/149
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0038 Acc: 0.9042

Epoch 143/149
----------
train Loss: 0.0002 Acc: 0.9862
val Loss: 0.0040 Acc: 0.9079

Epoch 144/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0039 Acc: 0.9061

Epoch 145/149
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0040 Acc: 0.9061

Epoch 146/149
----------
train Loss: 0.0002 Acc: 0.9870
val Loss: 0.0040 Acc: 0.9042

Epoch 147/149
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0041 Acc: 0.9061

Epoch 148/149
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0039 Acc: 0.9061

Epoch 149/149
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0037 Acc: 0.9061

Training complete in 98m 5s
Best val Acc: 0.913444

Model saved in "train.save".

---Testing---
Test accuracy: 0.914980

Process finished with exit code 0
'''