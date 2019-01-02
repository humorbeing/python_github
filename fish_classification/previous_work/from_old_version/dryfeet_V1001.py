import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import time
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import copy

# Learning_Rate = 0.001
# NUM_epoch = 200

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(256),
    torchvision.transforms.CenterCrop(227),
    # torchvision.transforms.RandomCrop(227),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

data_dir = 'imagenet'
dsets = datasets.ImageFolder(data_dir, data_transforms)

dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=40,
                                          shuffle=True, num_workers=2)
dset_sizes = len(dsets)

dset_classes = dsets.classes
class_size = len(dset_classes)
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

        optimizer = lr_scheduler(optimizer, epoch)
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in dset_loaders:
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

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects / dset_sizes

        print('train Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

model_conv = torchvision.models.resnet50(pretrained=True)
print('training last layer.')
print('- '*10)
# gpu 9G memory
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, class_size)
if gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=100)
print('fine tuning.')
print('- '*10)
# gpu 10G memory
for param in model_conv.parameters():
    param.requires_grad = True

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=100)


model_conv = torchvision.models.resnet50(pretrained=True)
print('fine tuning from start.')
print('- '*10)
# gpu 10G memory


num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, class_size)
if gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=100)
