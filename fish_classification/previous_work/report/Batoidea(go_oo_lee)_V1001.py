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


train_this = 'Batoidea(ga_oo_lee)'
test_this = 'Batoidea(ga_oo_lee)'


def trainer(trainee, testee):
    dataset_dir = './datasets/'
    # trainee = 'Batoidea(ga_oo_lee)'
    # testee = 'Batoidea(ga_oo_lee)'
    data_dir = dataset_dir+trainee
    test_dir = dataset_dir+testee

    val_s = [0.1, 0.15, 0.2, 0.25, 0.3]
    scale_size = 224
    all_time_best = 0
    all_time_best_acc = 0.0
    performance = list()

    for runs in range(5):
        valid_size = val_s[runs]
        batch_size = 100
        num_workers = 2
        EPOCH = np.random.randint(50, 100)
        if np.random.random() < 0.5:
            randomcrop = True
        else:
            randomcrop = False
        decay = np.random.randint(3, 15)
        # transformer + dataset
        # EPOCH = 1
        print('-'*20)
        print()
        print('run info[val: {}, epoch: {}, randcrop: {}, decay: {}]'.format(
            valid_size, EPOCH, randomcrop, decay
        ))
        print()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valid_transform = transforms.Compose([
            transforms.Scale(scale_size),
            torchvision.transforms.CenterCrop(scale_size),
            transforms.ToTensor(),
            normalize
        ])
        if randomcrop:
            train_transform = transforms.Compose([
                transforms.Scale(scale_size),
                transforms.RandomCrop(scale_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Scale(scale_size),
                torchvision.transforms.CenterCrop(scale_size),
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
        np.random.seed()
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


        def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=decay):
            """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
            lr = init_lr * (0.1**(epoch // lr_decay_epoch))

            if epoch % lr_decay_epoch == 0:
                print('LR is set to {}'.format(lr))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            return optimizer

        model_conv = torchvision.models.resnet18(pretrained=True)
        print('---Training last layer.---')

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
                                 exp_lr_scheduler, num_epochs=EPOCH)
        print()
        print('---Fine tuning.---')
        # gpu 10G memory
        for param in model_conv.parameters():
            param.requires_grad = True

        optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

        model_conv = train_model(model_conv, criterion, optimizer_conv,
                                 exp_lr_scheduler, num_epochs=EPOCH)

        test_dataset = datasets.ImageFolder(test_dir, valid_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)

        running_loss = 0.0
        running_corrects = 0
        class_correct = list(0. for i in range(batch_size))
        class_total = list(0. for i in range(batch_size))
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

            aa = preds.cpu().numpy()
            bb = labels.cpu().data.numpy()
            c = (aa == bb)
            for i in range(len(bb)):
                label = bb[i]
                class_correct[label] += c[i]
                class_total[label] += 1

        test_acc = running_corrects / len(test_dataset)

        print("Test accuracy: {:4f}".format(test_acc))
        print('-'*20)
        perform = list()
        for i in range(len(dset_classes)):
            perform.append(class_correct[i] / class_total[i])
            print('Accuracy of %5s : %2d %%' % (
                dset_classes[i], 100 * perform[i]))

        avg_perform = np.mean(perform)
        std_perform = np.std(perform)
        print('mean: {}, std: {}'.format(avg_perform,std_perform))
        if test_acc > all_time_best_acc:
            all_time_best_acc = test_acc
            all_time_best = copy.deepcopy(model_conv)
            performance = perform

        pass  # end of runs

    save_dir = './weights/'
    save_name = save_dir+trainee
    save_name += '_['+str(round(all_time_best_acc, 2))+']'
    save_name += '_mean['+str(round(np.mean(performance), 2))+']'
    save_name += '_std['+str(round(np.std(performance), 2))+']'
    save_name += '.save'
    torch.save(
        {'model': all_time_best,
         'dset_classes': dset_classes,
         'performance': performance,
         'acc': all_time_best_acc
         },
        save_name)
    print()
    print('Model saved in "'+save_name+'".')


for _ in range(10):
    trainer(train_this, test_this)


'''/usr/bin/python3.5 "/home/visbic/python/pytorch_playground/datasets/new fish/git-test/weight_trainera.py"
--------------------

run info[val: 0.1, epoch: 93, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0156 Acc: 0.3724
val Loss: 0.0369 Acc: 0.5116

Epoch 1/92
----------
train Loss: 0.0124 Acc: 0.5561
val Loss: 0.0291 Acc: 0.5814

Epoch 2/92
----------
train Loss: 0.0090 Acc: 0.6837
val Loss: 0.0181 Acc: 0.6977

Epoch 3/92
----------
train Loss: 0.0073 Acc: 0.7628
val Loss: 0.0160 Acc: 0.7442

Epoch 4/92
----------
train Loss: 0.0056 Acc: 0.7985
val Loss: 0.0136 Acc: 0.8140

Epoch 5/92
----------
train Loss: 0.0050 Acc: 0.8265
val Loss: 0.0129 Acc: 0.8140

Epoch 6/92
----------
train Loss: 0.0042 Acc: 0.8622
val Loss: 0.0122 Acc: 0.8140

Epoch 7/92
----------
train Loss: 0.0041 Acc: 0.8648
val Loss: 0.0115 Acc: 0.8605

Epoch 8/92
----------
LR is set to 0.001
train Loss: 0.0033 Acc: 0.8903
val Loss: 0.0117 Acc: 0.8372

Epoch 9/92
----------
train Loss: 0.0034 Acc: 0.8929
val Loss: 0.0120 Acc: 0.8372

Epoch 10/92
----------
train Loss: 0.0033 Acc: 0.9107
val Loss: 0.0122 Acc: 0.8372

Epoch 11/92
----------
train Loss: 0.0030 Acc: 0.9107
val Loss: 0.0121 Acc: 0.8605

Epoch 12/92
----------
train Loss: 0.0035 Acc: 0.8929
val Loss: 0.0120 Acc: 0.8372

Epoch 13/92
----------
train Loss: 0.0033 Acc: 0.8929
val Loss: 0.0119 Acc: 0.8372

Epoch 14/92
----------
train Loss: 0.0034 Acc: 0.8878
val Loss: 0.0119 Acc: 0.8372

Epoch 15/92
----------
train Loss: 0.0031 Acc: 0.9107
val Loss: 0.0117 Acc: 0.8372

Epoch 16/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0031 Acc: 0.9082
val Loss: 0.0117 Acc: 0.8372

Epoch 17/92
----------
train Loss: 0.0032 Acc: 0.9005
val Loss: 0.0116 Acc: 0.8372

Epoch 18/92
----------
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0117 Acc: 0.8372

Epoch 19/92
----------
train Loss: 0.0031 Acc: 0.9056
val Loss: 0.0117 Acc: 0.8372

Epoch 20/92
----------
train Loss: 0.0033 Acc: 0.9056
val Loss: 0.0117 Acc: 0.8372

Epoch 21/92
----------
train Loss: 0.0031 Acc: 0.9184
val Loss: 0.0117 Acc: 0.8372

Epoch 22/92
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0117 Acc: 0.8372

Epoch 23/92
----------
train Loss: 0.0032 Acc: 0.9031
val Loss: 0.0117 Acc: 0.8372

Epoch 24/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0029 Acc: 0.9158
val Loss: 0.0118 Acc: 0.8372

Epoch 25/92
----------
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0117 Acc: 0.8372

Epoch 26/92
----------
train Loss: 0.0032 Acc: 0.9286
val Loss: 0.0117 Acc: 0.8372

Epoch 27/92
----------
train Loss: 0.0032 Acc: 0.9260
val Loss: 0.0117 Acc: 0.8372

Epoch 28/92
----------
train Loss: 0.0034 Acc: 0.9031
val Loss: 0.0118 Acc: 0.8372

Epoch 29/92
----------
train Loss: 0.0030 Acc: 0.9005
val Loss: 0.0118 Acc: 0.8372

Epoch 30/92
----------
train Loss: 0.0030 Acc: 0.9107
val Loss: 0.0118 Acc: 0.8372

Epoch 31/92
----------
train Loss: 0.0031 Acc: 0.9082
val Loss: 0.0118 Acc: 0.8372

Epoch 32/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0119 Acc: 0.8372

Epoch 33/92
----------
train Loss: 0.0033 Acc: 0.9056
val Loss: 0.0119 Acc: 0.8372

Epoch 34/92
----------
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0119 Acc: 0.8372

Epoch 35/92
----------
train Loss: 0.0032 Acc: 0.9158
val Loss: 0.0118 Acc: 0.8372

Epoch 36/92
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0118 Acc: 0.8372

Epoch 37/92
----------
train Loss: 0.0034 Acc: 0.8852
val Loss: 0.0118 Acc: 0.8372

Epoch 38/92
----------
train Loss: 0.0030 Acc: 0.9235
val Loss: 0.0118 Acc: 0.8372

Epoch 39/92
----------
train Loss: 0.0032 Acc: 0.9082
val Loss: 0.0118 Acc: 0.8372

Epoch 40/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0031 Acc: 0.9031
val Loss: 0.0118 Acc: 0.8372

Epoch 41/92
----------
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 42/92
----------
train Loss: 0.0032 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 43/92
----------
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0118 Acc: 0.8372

Epoch 44/92
----------
train Loss: 0.0033 Acc: 0.8980
val Loss: 0.0118 Acc: 0.8372

Epoch 45/92
----------
train Loss: 0.0029 Acc: 0.9388
val Loss: 0.0118 Acc: 0.8372

Epoch 46/92
----------
train Loss: 0.0029 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 47/92
----------
train Loss: 0.0032 Acc: 0.9056
val Loss: 0.0118 Acc: 0.8372

Epoch 48/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0029 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Epoch 49/92
----------
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0118 Acc: 0.8372

Epoch 50/92
----------
train Loss: 0.0030 Acc: 0.9362
val Loss: 0.0118 Acc: 0.8372

Epoch 51/92
----------
train Loss: 0.0028 Acc: 0.9184
val Loss: 0.0118 Acc: 0.8372

Epoch 52/92
----------
train Loss: 0.0031 Acc: 0.9107
val Loss: 0.0118 Acc: 0.8372

Epoch 53/92
----------
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0118 Acc: 0.8372

Epoch 54/92
----------
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 55/92
----------
train Loss: 0.0032 Acc: 0.9056
val Loss: 0.0118 Acc: 0.8372

Epoch 56/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0034 Acc: 0.8903
val Loss: 0.0118 Acc: 0.8372

Epoch 57/92
----------
train Loss: 0.0032 Acc: 0.9056
val Loss: 0.0117 Acc: 0.8372

Epoch 58/92
----------
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 59/92
----------
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 60/92
----------
train Loss: 0.0032 Acc: 0.8980
val Loss: 0.0118 Acc: 0.8372

Epoch 61/92
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0118 Acc: 0.8372

Epoch 62/92
----------
train Loss: 0.0030 Acc: 0.9056
val Loss: 0.0118 Acc: 0.8372

Epoch 63/92
----------
train Loss: 0.0031 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Epoch 64/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0118 Acc: 0.8372

Epoch 65/92
----------
train Loss: 0.0029 Acc: 0.9337
val Loss: 0.0118 Acc: 0.8372

Epoch 66/92
----------
train Loss: 0.0031 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Epoch 67/92
----------
train Loss: 0.0030 Acc: 0.9311
val Loss: 0.0118 Acc: 0.8372

Epoch 68/92
----------
train Loss: 0.0032 Acc: 0.9082
val Loss: 0.0118 Acc: 0.8372

Epoch 69/92
----------
train Loss: 0.0032 Acc: 0.9286
val Loss: 0.0119 Acc: 0.8372

Epoch 70/92
----------
train Loss: 0.0032 Acc: 0.9107
val Loss: 0.0118 Acc: 0.8372

Epoch 71/92
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Epoch 72/92
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 73/92
----------
train Loss: 0.0032 Acc: 0.9005
val Loss: 0.0118 Acc: 0.8372

Epoch 74/92
----------
train Loss: 0.0031 Acc: 0.9031
val Loss: 0.0118 Acc: 0.8372

Epoch 75/92
----------
train Loss: 0.0032 Acc: 0.9056
val Loss: 0.0119 Acc: 0.8372

Epoch 76/92
----------
train Loss: 0.0029 Acc: 0.9107
val Loss: 0.0118 Acc: 0.8372

Epoch 77/92
----------
train Loss: 0.0033 Acc: 0.8980
val Loss: 0.0118 Acc: 0.8372

Epoch 78/92
----------
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 79/92
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0118 Acc: 0.8372

Epoch 80/92
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9107
val Loss: 0.0118 Acc: 0.8372

Epoch 81/92
----------
train Loss: 0.0030 Acc: 0.9311
val Loss: 0.0118 Acc: 0.8372

Epoch 82/92
----------
train Loss: 0.0031 Acc: 0.9107
val Loss: 0.0119 Acc: 0.8372

Epoch 83/92
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0119 Acc: 0.8372

Epoch 84/92
----------
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0118 Acc: 0.8372

Epoch 85/92
----------
train Loss: 0.0032 Acc: 0.9056
val Loss: 0.0118 Acc: 0.8372

Epoch 86/92
----------
train Loss: 0.0031 Acc: 0.9031
val Loss: 0.0118 Acc: 0.8372

Epoch 87/92
----------
train Loss: 0.0033 Acc: 0.9082
val Loss: 0.0118 Acc: 0.8372

Epoch 88/92
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0030 Acc: 0.9235
val Loss: 0.0118 Acc: 0.8372

Epoch 89/92
----------
train Loss: 0.0031 Acc: 0.9209
val Loss: 0.0118 Acc: 0.8372

Epoch 90/92
----------
train Loss: 0.0033 Acc: 0.9133
val Loss: 0.0118 Acc: 0.8372

Epoch 91/92
----------
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0118 Acc: 0.8372

Epoch 92/92
----------
train Loss: 0.0030 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Training complete in 3m 14s
Best val Acc: 0.860465

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0033 Acc: 0.9082
val Loss: 0.0119 Acc: 0.8605

Epoch 1/92
----------
train Loss: 0.0020 Acc: 0.9617
val Loss: 0.0124 Acc: 0.8837

Epoch 2/92
----------
train Loss: 0.0011 Acc: 0.9847
val Loss: 0.0114 Acc: 0.8837

Epoch 3/92
----------
train Loss: 0.0007 Acc: 0.9949
val Loss: 0.0126 Acc: 0.8372

Epoch 4/92
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0122 Acc: 0.8372

Epoch 5/92
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8837

Epoch 6/92
----------
train Loss: 0.0002 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8837

Epoch 7/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0122 Acc: 0.8837

Epoch 8/92
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8837

Epoch 9/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 10/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 11/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 12/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 13/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 14/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 15/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 16/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 17/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 18/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 19/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 20/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 21/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 22/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 23/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 24/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 25/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 26/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 27/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 28/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 29/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 30/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 31/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 32/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 33/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 34/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 35/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8605

Epoch 36/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 37/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 38/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 39/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 40/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 41/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 42/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 43/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 44/92
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0116 Acc: 0.8605

Epoch 45/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 46/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 47/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 48/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 49/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 50/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 51/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 52/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 53/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 54/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 55/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 56/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 57/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 58/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 59/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 60/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 61/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 62/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 63/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 64/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 65/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 66/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 67/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 68/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 69/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 70/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 71/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 72/92
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 73/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 74/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 75/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 76/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 77/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 78/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 79/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 80/92
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 81/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 82/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 83/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 84/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 85/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 86/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 87/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 88/92
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 89/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 90/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Epoch 91/92
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0116 Acc: 0.8605

Epoch 92/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8605

Training complete in 3m 36s
Best val Acc: 0.883721

---Testing---
Test accuracy: 0.967816
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 94 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 98 %
mean: 0.9625636106367572, std: 0.014821045169577158
--------------------

run info[val: 0.15, epoch: 94, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0165 Acc: 0.3703
val Loss: 0.0238 Acc: 0.5692

Epoch 1/93
----------
train Loss: 0.0139 Acc: 0.5784
val Loss: 0.0182 Acc: 0.6615

Epoch 2/93
----------
train Loss: 0.0091 Acc: 0.7243
val Loss: 0.0104 Acc: 0.7692

Epoch 3/93
----------
train Loss: 0.0076 Acc: 0.7892
val Loss: 0.0091 Acc: 0.7538

Epoch 4/93
----------
train Loss: 0.0062 Acc: 0.7919
val Loss: 0.0084 Acc: 0.8000

Epoch 5/93
----------
train Loss: 0.0054 Acc: 0.8216
val Loss: 0.0069 Acc: 0.8923

Epoch 6/93
----------
LR is set to 0.001
train Loss: 0.0043 Acc: 0.8973
val Loss: 0.0070 Acc: 0.8462

Epoch 7/93
----------
train Loss: 0.0042 Acc: 0.8865
val Loss: 0.0070 Acc: 0.8769

Epoch 8/93
----------
train Loss: 0.0043 Acc: 0.8757
val Loss: 0.0069 Acc: 0.8308

Epoch 9/93
----------
train Loss: 0.0040 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8923

Epoch 10/93
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 11/93
----------
train Loss: 0.0039 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 12/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0039 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8769

Epoch 13/93
----------
train Loss: 0.0038 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8769

Epoch 14/93
----------
train Loss: 0.0038 Acc: 0.8838
val Loss: 0.0068 Acc: 0.8769

Epoch 15/93
----------
train Loss: 0.0039 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8769

Epoch 16/93
----------
train Loss: 0.0039 Acc: 0.8784
val Loss: 0.0068 Acc: 0.8769

Epoch 17/93
----------
train Loss: 0.0039 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 18/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8769

Epoch 19/93
----------
train Loss: 0.0038 Acc: 0.9162
val Loss: 0.0068 Acc: 0.8615

Epoch 20/93
----------
train Loss: 0.0039 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 21/93
----------
train Loss: 0.0037 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 22/93
----------
train Loss: 0.0038 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 23/93
----------
train Loss: 0.0039 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 24/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0036 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8615

Epoch 25/93
----------
train Loss: 0.0038 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 26/93
----------
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8769

Epoch 27/93
----------
train Loss: 0.0038 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8769

Epoch 28/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 29/93
----------
train Loss: 0.0038 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8615

Epoch 30/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0037 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8615

Epoch 31/93
----------
train Loss: 0.0038 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 32/93
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 33/93
----------
train Loss: 0.0039 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8615

Epoch 34/93
----------
train Loss: 0.0039 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8769

Epoch 35/93
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 36/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0037 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 37/93
----------
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8769

Epoch 38/93
----------
train Loss: 0.0036 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 39/93
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 40/93
----------
train Loss: 0.0036 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 41/93
----------
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8769

Epoch 42/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0038 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 43/93
----------
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8769

Epoch 44/93
----------
train Loss: 0.0038 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 45/93
----------
train Loss: 0.0038 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8615

Epoch 46/93
----------
train Loss: 0.0036 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8769

Epoch 47/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 48/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8615

Epoch 49/93
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8615

Epoch 50/93
----------
train Loss: 0.0039 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 51/93
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 52/93
----------
train Loss: 0.0038 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 53/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 54/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0039 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 55/93
----------
train Loss: 0.0039 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8769

Epoch 56/93
----------
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8769

Epoch 57/93
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8615

Epoch 58/93
----------
train Loss: 0.0037 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8615

Epoch 59/93
----------
train Loss: 0.0039 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 60/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0038 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8615

Epoch 61/93
----------
train Loss: 0.0037 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 62/93
----------
train Loss: 0.0036 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 63/93
----------
train Loss: 0.0037 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8769

Epoch 64/93
----------
train Loss: 0.0038 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8769

Epoch 65/93
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8769

Epoch 66/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0037 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8615

Epoch 67/93
----------
train Loss: 0.0036 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8615

Epoch 68/93
----------
train Loss: 0.0039 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 69/93
----------
train Loss: 0.0037 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8769

Epoch 70/93
----------
train Loss: 0.0036 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 71/93
----------
train Loss: 0.0039 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8769

Epoch 72/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8615

Epoch 73/93
----------
train Loss: 0.0039 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8769

Epoch 74/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 75/93
----------
train Loss: 0.0038 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 76/93
----------
train Loss: 0.0037 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 77/93
----------
train Loss: 0.0036 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 78/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8615

Epoch 79/93
----------
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8769

Epoch 80/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 81/93
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8615

Epoch 82/93
----------
train Loss: 0.0038 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 83/93
----------
train Loss: 0.0039 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 84/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8769

Epoch 85/93
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8769

Epoch 86/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 87/93
----------
train Loss: 0.0038 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 88/93
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 89/93
----------
train Loss: 0.0036 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Epoch 90/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0039 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8769

Epoch 91/93
----------
train Loss: 0.0039 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 92/93
----------
train Loss: 0.0041 Acc: 0.8676
val Loss: 0.0068 Acc: 0.8769

Epoch 93/93
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Training complete in 3m 19s
Best val Acc: 0.892308

---Fine tuning.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0041 Acc: 0.8973
val Loss: 0.0063 Acc: 0.9077

Epoch 1/93
----------
train Loss: 0.0023 Acc: 0.9622
val Loss: 0.0083 Acc: 0.8923

Epoch 2/93
----------
train Loss: 0.0011 Acc: 0.9865
val Loss: 0.0063 Acc: 0.8615

Epoch 3/93
----------
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0058 Acc: 0.9231

Epoch 4/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9231

Epoch 5/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9231

Epoch 6/93
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0053 Acc: 0.9231

Epoch 7/93
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0053 Acc: 0.9231

Epoch 8/93
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0053 Acc: 0.9231

Epoch 9/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 10/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.9231

Epoch 11/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 12/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 13/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 14/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 15/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 16/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 17/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 18/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 19/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 20/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 21/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 22/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 23/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 24/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 25/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 26/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 27/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 28/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 29/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 30/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 31/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 32/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 33/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 34/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 35/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 36/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 37/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 38/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 39/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 40/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 41/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 42/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 43/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 44/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 45/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 46/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 47/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 48/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 49/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 50/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 51/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 52/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 53/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 54/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 55/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 56/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 57/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 58/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 59/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 60/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 61/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 62/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 63/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 64/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 65/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 66/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 67/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 68/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 69/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 70/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 71/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 72/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 73/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 74/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 75/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 76/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 77/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 78/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 79/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 80/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 81/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 82/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 83/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 84/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 85/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 86/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 87/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 88/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 89/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 90/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 91/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Epoch 92/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9231

Epoch 93/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9231

Training complete in 3m 40s
Best val Acc: 0.923077

---Testing---
Test accuracy: 0.986207
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.9830968189505563, std: 0.00880558370277733
--------------------

run info[val: 0.2, epoch: 71, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0190 Acc: 0.3391
val Loss: 0.0172 Acc: 0.4253

Epoch 1/70
----------
train Loss: 0.0142 Acc: 0.4799
val Loss: 0.0162 Acc: 0.4713

Epoch 2/70
----------
train Loss: 0.0107 Acc: 0.6580
val Loss: 0.0090 Acc: 0.7356

Epoch 3/70
----------
LR is set to 0.001
train Loss: 0.0088 Acc: 0.7730
val Loss: 0.0086 Acc: 0.7356

Epoch 4/70
----------
train Loss: 0.0075 Acc: 0.7874
val Loss: 0.0086 Acc: 0.7471

Epoch 5/70
----------
train Loss: 0.0076 Acc: 0.7672
val Loss: 0.0087 Acc: 0.7126

Epoch 6/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0072 Acc: 0.7902
val Loss: 0.0086 Acc: 0.7356

Epoch 7/70
----------
train Loss: 0.0079 Acc: 0.7557
val Loss: 0.0085 Acc: 0.7356

Epoch 8/70
----------
train Loss: 0.0075 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7356

Epoch 9/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.7902
val Loss: 0.0084 Acc: 0.7471

Epoch 10/70
----------
train Loss: 0.0072 Acc: 0.7730
val Loss: 0.0084 Acc: 0.7586

Epoch 11/70
----------
train Loss: 0.0075 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7586

Epoch 12/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0073 Acc: 0.7931
val Loss: 0.0085 Acc: 0.7701

Epoch 13/70
----------
train Loss: 0.0075 Acc: 0.7586
val Loss: 0.0085 Acc: 0.7471

Epoch 14/70
----------
train Loss: 0.0073 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7471

Epoch 15/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0073 Acc: 0.7845
val Loss: 0.0085 Acc: 0.7471

Epoch 16/70
----------
train Loss: 0.0071 Acc: 0.7931
val Loss: 0.0085 Acc: 0.7471

Epoch 17/70
----------
train Loss: 0.0073 Acc: 0.8046
val Loss: 0.0085 Acc: 0.7586

Epoch 18/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0069 Acc: 0.8103
val Loss: 0.0085 Acc: 0.7471

Epoch 19/70
----------
train Loss: 0.0076 Acc: 0.7931
val Loss: 0.0085 Acc: 0.7586

Epoch 20/70
----------
train Loss: 0.0071 Acc: 0.8075
val Loss: 0.0085 Acc: 0.7471

Epoch 21/70
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0071 Acc: 0.7845
val Loss: 0.0085 Acc: 0.7471

Epoch 22/70
----------
train Loss: 0.0078 Acc: 0.7701
val Loss: 0.0085 Acc: 0.7471

Epoch 23/70
----------
train Loss: 0.0073 Acc: 0.7787
val Loss: 0.0084 Acc: 0.7586

Epoch 24/70
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0069 Acc: 0.7845
val Loss: 0.0084 Acc: 0.7816

Epoch 25/70
----------
train Loss: 0.0073 Acc: 0.7672
val Loss: 0.0084 Acc: 0.7816

Epoch 26/70
----------
train Loss: 0.0072 Acc: 0.7787
val Loss: 0.0084 Acc: 0.7816

Epoch 27/70
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0072 Acc: 0.7816
val Loss: 0.0084 Acc: 0.7931

Epoch 28/70
----------
train Loss: 0.0072 Acc: 0.7874
val Loss: 0.0083 Acc: 0.7931

Epoch 29/70
----------
train Loss: 0.0073 Acc: 0.7931
val Loss: 0.0084 Acc: 0.7816

Epoch 30/70
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0074 Acc: 0.7874
val Loss: 0.0084 Acc: 0.7816

Epoch 31/70
----------
train Loss: 0.0071 Acc: 0.8075
val Loss: 0.0084 Acc: 0.7701

Epoch 32/70
----------
train Loss: 0.0073 Acc: 0.7874
val Loss: 0.0084 Acc: 0.7816

Epoch 33/70
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0071 Acc: 0.8161
val Loss: 0.0084 Acc: 0.7701

Epoch 34/70
----------
train Loss: 0.0073 Acc: 0.7874
val Loss: 0.0084 Acc: 0.7931

Epoch 35/70
----------
train Loss: 0.0074 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7816

Epoch 36/70
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0073 Acc: 0.7989
val Loss: 0.0085 Acc: 0.7701

Epoch 37/70
----------
train Loss: 0.0074 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7701

Epoch 38/70
----------
train Loss: 0.0077 Acc: 0.7931
val Loss: 0.0084 Acc: 0.7816

Epoch 39/70
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0074 Acc: 0.7902
val Loss: 0.0084 Acc: 0.7816

Epoch 40/70
----------
train Loss: 0.0070 Acc: 0.7845
val Loss: 0.0084 Acc: 0.7816

Epoch 41/70
----------
train Loss: 0.0071 Acc: 0.7960
val Loss: 0.0084 Acc: 0.7816

Epoch 42/70
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0071 Acc: 0.8017
val Loss: 0.0084 Acc: 0.7586

Epoch 43/70
----------
train Loss: 0.0070 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7701

Epoch 44/70
----------
train Loss: 0.0075 Acc: 0.7787
val Loss: 0.0085 Acc: 0.7816

Epoch 45/70
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0077 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7816

Epoch 46/70
----------
train Loss: 0.0073 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7701

Epoch 47/70
----------
train Loss: 0.0070 Acc: 0.8161
val Loss: 0.0084 Acc: 0.7816

Epoch 48/70
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0069 Acc: 0.7931
val Loss: 0.0084 Acc: 0.7701

Epoch 49/70
----------
train Loss: 0.0071 Acc: 0.7902
val Loss: 0.0084 Acc: 0.7816

Epoch 50/70
----------
train Loss: 0.0073 Acc: 0.7730
val Loss: 0.0084 Acc: 0.7701

Epoch 51/70
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0073 Acc: 0.8017
val Loss: 0.0084 Acc: 0.7701

Epoch 52/70
----------
train Loss: 0.0070 Acc: 0.8075
val Loss: 0.0084 Acc: 0.7816

Epoch 53/70
----------
train Loss: 0.0071 Acc: 0.7902
val Loss: 0.0085 Acc: 0.7701

Epoch 54/70
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0074 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7816

Epoch 55/70
----------
train Loss: 0.0072 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7816

Epoch 56/70
----------
train Loss: 0.0071 Acc: 0.7902
val Loss: 0.0084 Acc: 0.7816

Epoch 57/70
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0073 Acc: 0.7759
val Loss: 0.0084 Acc: 0.7701

Epoch 58/70
----------
train Loss: 0.0073 Acc: 0.7874
val Loss: 0.0084 Acc: 0.7816

Epoch 59/70
----------
train Loss: 0.0074 Acc: 0.7701
val Loss: 0.0085 Acc: 0.7701

Epoch 60/70
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0073 Acc: 0.7672
val Loss: 0.0085 Acc: 0.7816

Epoch 61/70
----------
train Loss: 0.0070 Acc: 0.7586
val Loss: 0.0085 Acc: 0.7816

Epoch 62/70
----------
train Loss: 0.0075 Acc: 0.7644
val Loss: 0.0085 Acc: 0.7586

Epoch 63/70
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0076 Acc: 0.7931
val Loss: 0.0085 Acc: 0.7471

Epoch 64/70
----------
train Loss: 0.0073 Acc: 0.7816
val Loss: 0.0085 Acc: 0.7586

Epoch 65/70
----------
train Loss: 0.0075 Acc: 0.7529
val Loss: 0.0085 Acc: 0.7586

Epoch 66/70
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0072 Acc: 0.7730
val Loss: 0.0085 Acc: 0.7471

Epoch 67/70
----------
train Loss: 0.0072 Acc: 0.7874
val Loss: 0.0085 Acc: 0.7586

Epoch 68/70
----------
train Loss: 0.0074 Acc: 0.7902
val Loss: 0.0084 Acc: 0.7701

Epoch 69/70
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0073 Acc: 0.7874
val Loss: 0.0084 Acc: 0.7471

Epoch 70/70
----------
train Loss: 0.0076 Acc: 0.7759
val Loss: 0.0085 Acc: 0.7471

Training complete in 2m 36s
Best val Acc: 0.793103

---Fine tuning.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0074 Acc: 0.7845
val Loss: 0.0071 Acc: 0.8161

Epoch 1/70
----------
train Loss: 0.0048 Acc: 0.9138
val Loss: 0.0063 Acc: 0.8276

Epoch 2/70
----------
train Loss: 0.0029 Acc: 0.9454
val Loss: 0.0054 Acc: 0.8736

Epoch 3/70
----------
LR is set to 0.001
train Loss: 0.0016 Acc: 0.9828
val Loss: 0.0049 Acc: 0.8736

Epoch 4/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0047 Acc: 0.8966

Epoch 5/70
----------
train Loss: 0.0017 Acc: 0.9799
val Loss: 0.0045 Acc: 0.8966

Epoch 6/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.8966

Epoch 7/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.8966

Epoch 8/70
----------
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 9/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.8966

Epoch 10/70
----------
train Loss: 0.0015 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 11/70
----------
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 12/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 13/70
----------
train Loss: 0.0013 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 14/70
----------
train Loss: 0.0015 Acc: 0.9741
val Loss: 0.0044 Acc: 0.8966

Epoch 15/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0017 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 16/70
----------
train Loss: 0.0014 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 17/70
----------
train Loss: 0.0013 Acc: 0.9914
val Loss: 0.0044 Acc: 0.9080

Epoch 18/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0015 Acc: 0.9770
val Loss: 0.0044 Acc: 0.9080

Epoch 19/70
----------
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0043 Acc: 0.9080

Epoch 20/70
----------
train Loss: 0.0013 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 21/70
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0014 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 22/70
----------
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 23/70
----------
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 24/70
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0014 Acc: 0.9770
val Loss: 0.0044 Acc: 0.9080

Epoch 25/70
----------
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.8966

Epoch 26/70
----------
train Loss: 0.0016 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 27/70
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 28/70
----------
train Loss: 0.0013 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 29/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 30/70
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0013 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 31/70
----------
train Loss: 0.0016 Acc: 0.9741
val Loss: 0.0044 Acc: 0.9080

Epoch 32/70
----------
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.8966

Epoch 33/70
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 34/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9195

Epoch 35/70
----------
train Loss: 0.0015 Acc: 0.9770
val Loss: 0.0044 Acc: 0.9080

Epoch 36/70
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 37/70
----------
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 38/70
----------
train Loss: 0.0013 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9195

Epoch 39/70
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 40/70
----------
train Loss: 0.0015 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 41/70
----------
train Loss: 0.0014 Acc: 0.9713
val Loss: 0.0044 Acc: 0.9080

Epoch 42/70
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 43/70
----------
train Loss: 0.0013 Acc: 0.9914
val Loss: 0.0044 Acc: 0.9080

Epoch 44/70
----------
train Loss: 0.0014 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 45/70
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0015 Acc: 0.9770
val Loss: 0.0044 Acc: 0.8966

Epoch 46/70
----------
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.8966

Epoch 47/70
----------
train Loss: 0.0015 Acc: 0.9799
val Loss: 0.0044 Acc: 0.9080

Epoch 48/70
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0013 Acc: 0.9971
val Loss: 0.0044 Acc: 0.8966

Epoch 49/70
----------
train Loss: 0.0016 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 50/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9195

Epoch 51/70
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0016 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 52/70
----------
train Loss: 0.0013 Acc: 0.9943
val Loss: 0.0044 Acc: 0.9080

Epoch 53/70
----------
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.9080

Epoch 54/70
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 55/70
----------
train Loss: 0.0012 Acc: 0.9943
val Loss: 0.0044 Acc: 0.9080

Epoch 56/70
----------
train Loss: 0.0015 Acc: 0.9856
val Loss: 0.0044 Acc: 0.8966

Epoch 57/70
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0013 Acc: 0.9914
val Loss: 0.0044 Acc: 0.9080

Epoch 58/70
----------
train Loss: 0.0013 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 59/70
----------
train Loss: 0.0013 Acc: 0.9914
val Loss: 0.0044 Acc: 0.9080

Epoch 60/70
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 61/70
----------
train Loss: 0.0013 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 62/70
----------
train Loss: 0.0014 Acc: 0.9885
val Loss: 0.0044 Acc: 0.8966

Epoch 63/70
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0014 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 64/70
----------
train Loss: 0.0015 Acc: 0.9741
val Loss: 0.0044 Acc: 0.9080

Epoch 65/70
----------
train Loss: 0.0017 Acc: 0.9741
val Loss: 0.0044 Acc: 0.9080

Epoch 66/70
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0012 Acc: 0.9885
val Loss: 0.0044 Acc: 0.9080

Epoch 67/70
----------
train Loss: 0.0015 Acc: 0.9914
val Loss: 0.0044 Acc: 0.9080

Epoch 68/70
----------
train Loss: 0.0014 Acc: 0.9828
val Loss: 0.0044 Acc: 0.9080

Epoch 69/70
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0016 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8966

Epoch 70/70
----------
train Loss: 0.0013 Acc: 0.9856
val Loss: 0.0044 Acc: 0.8966

Training complete in 2m 50s
Best val Acc: 0.919540

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 99 %
mean: 0.9765989615550751, std: 0.0094542413618626
--------------------

run info[val: 0.25, epoch: 72, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0189 Acc: 0.3578
val Loss: 0.0244 Acc: 0.4815

Epoch 1/71
----------
train Loss: 0.0181 Acc: 0.4893
val Loss: 0.0167 Acc: 0.5833

Epoch 2/71
----------
train Loss: 0.0115 Acc: 0.6055
val Loss: 0.0126 Acc: 0.6389

Epoch 3/71
----------
train Loss: 0.0093 Acc: 0.7187
val Loss: 0.0092 Acc: 0.7963

Epoch 4/71
----------
train Loss: 0.0081 Acc: 0.7584
val Loss: 0.0152 Acc: 0.7222

Epoch 5/71
----------
train Loss: 0.0072 Acc: 0.7737
val Loss: 0.0103 Acc: 0.8241

Epoch 6/71
----------
train Loss: 0.0061 Acc: 0.8440
val Loss: 0.0082 Acc: 0.8519

Epoch 7/71
----------
train Loss: 0.0097 Acc: 0.7890
val Loss: 0.0124 Acc: 0.8148

Epoch 8/71
----------
LR is set to 0.001
train Loss: 0.0056 Acc: 0.8654
val Loss: 0.0100 Acc: 0.8056

Epoch 9/71
----------
train Loss: 0.0047 Acc: 0.8777
val Loss: 0.0101 Acc: 0.8426

Epoch 10/71
----------
train Loss: 0.0037 Acc: 0.9144
val Loss: 0.0075 Acc: 0.8519

Epoch 11/71
----------
train Loss: 0.0036 Acc: 0.9083
val Loss: 0.0073 Acc: 0.8889

Epoch 12/71
----------
train Loss: 0.0040 Acc: 0.9052
val Loss: 0.0078 Acc: 0.8796

Epoch 13/71
----------
train Loss: 0.0045 Acc: 0.8899
val Loss: 0.0109 Acc: 0.8796

Epoch 14/71
----------
train Loss: 0.0038 Acc: 0.9021
val Loss: 0.0067 Acc: 0.8796

Epoch 15/71
----------
train Loss: 0.0040 Acc: 0.9144
val Loss: 0.0065 Acc: 0.8796

Epoch 16/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0039 Acc: 0.9052
val Loss: 0.0096 Acc: 0.8889

Epoch 17/71
----------
train Loss: 0.0039 Acc: 0.8807
val Loss: 0.0058 Acc: 0.8796

Epoch 18/71
----------
train Loss: 0.0039 Acc: 0.9052
val Loss: 0.0135 Acc: 0.8889

Epoch 19/71
----------
train Loss: 0.0034 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8889

Epoch 20/71
----------
train Loss: 0.0039 Acc: 0.8991
val Loss: 0.0128 Acc: 0.8889

Epoch 21/71
----------
train Loss: 0.0033 Acc: 0.9266
val Loss: 0.0072 Acc: 0.8889

Epoch 22/71
----------
train Loss: 0.0032 Acc: 0.9174
val Loss: 0.0101 Acc: 0.8889

Epoch 23/71
----------
train Loss: 0.0034 Acc: 0.8960
val Loss: 0.0071 Acc: 0.8889

Epoch 24/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0035 Acc: 0.9083
val Loss: 0.0102 Acc: 0.8889

Epoch 25/71
----------
train Loss: 0.0041 Acc: 0.9021
val Loss: 0.0106 Acc: 0.8796

Epoch 26/71
----------
train Loss: 0.0032 Acc: 0.9083
val Loss: 0.0085 Acc: 0.8889

Epoch 27/71
----------
train Loss: 0.0046 Acc: 0.9083
val Loss: 0.0071 Acc: 0.8889

Epoch 28/71
----------
train Loss: 0.0037 Acc: 0.9021
val Loss: 0.0053 Acc: 0.8889

Epoch 29/71
----------
train Loss: 0.0039 Acc: 0.9174
val Loss: 0.0068 Acc: 0.8889

Epoch 30/71
----------
train Loss: 0.0038 Acc: 0.8960
val Loss: 0.0064 Acc: 0.8889

Epoch 31/71
----------
train Loss: 0.0040 Acc: 0.9083
val Loss: 0.0117 Acc: 0.8889

Epoch 32/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0039 Acc: 0.9113
val Loss: 0.0052 Acc: 0.8889

Epoch 33/71
----------
train Loss: 0.0033 Acc: 0.9021
val Loss: 0.0057 Acc: 0.8889

Epoch 34/71
----------
train Loss: 0.0036 Acc: 0.9297
val Loss: 0.0111 Acc: 0.8796

Epoch 35/71
----------
train Loss: 0.0037 Acc: 0.8960
val Loss: 0.0064 Acc: 0.8889

Epoch 36/71
----------
train Loss: 0.0037 Acc: 0.9205
val Loss: 0.0102 Acc: 0.8889

Epoch 37/71
----------
train Loss: 0.0037 Acc: 0.9144
val Loss: 0.0081 Acc: 0.8889

Epoch 38/71
----------
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0102 Acc: 0.8889

Epoch 39/71
----------
train Loss: 0.0033 Acc: 0.9266
val Loss: 0.0067 Acc: 0.8889

Epoch 40/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0064 Acc: 0.8889

Epoch 41/71
----------
train Loss: 0.0038 Acc: 0.8930
val Loss: 0.0088 Acc: 0.8889

Epoch 42/71
----------
train Loss: 0.0031 Acc: 0.9174
val Loss: 0.0079 Acc: 0.8889

Epoch 43/71
----------
train Loss: 0.0038 Acc: 0.9083
val Loss: 0.0110 Acc: 0.8889

Epoch 44/71
----------
train Loss: 0.0037 Acc: 0.9113
val Loss: 0.0090 Acc: 0.8889

Epoch 45/71
----------
train Loss: 0.0036 Acc: 0.9174
val Loss: 0.0077 Acc: 0.8889

Epoch 46/71
----------
train Loss: 0.0038 Acc: 0.8930
val Loss: 0.0119 Acc: 0.8889

Epoch 47/71
----------
train Loss: 0.0038 Acc: 0.9144
val Loss: 0.0062 Acc: 0.8889

Epoch 48/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0038 Acc: 0.8960
val Loss: 0.0091 Acc: 0.8889

Epoch 49/71
----------
train Loss: 0.0034 Acc: 0.9144
val Loss: 0.0071 Acc: 0.8889

Epoch 50/71
----------
train Loss: 0.0037 Acc: 0.9297
val Loss: 0.0134 Acc: 0.8889

Epoch 51/71
----------
train Loss: 0.0034 Acc: 0.9144
val Loss: 0.0122 Acc: 0.8889

Epoch 52/71
----------
train Loss: 0.0037 Acc: 0.9113
val Loss: 0.0063 Acc: 0.8889

Epoch 53/71
----------
train Loss: 0.0033 Acc: 0.9235
val Loss: 0.0061 Acc: 0.8889

Epoch 54/71
----------
train Loss: 0.0041 Acc: 0.8899
val Loss: 0.0071 Acc: 0.8796

Epoch 55/71
----------
train Loss: 0.0036 Acc: 0.9235
val Loss: 0.0067 Acc: 0.8889

Epoch 56/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0033 Acc: 0.9083
val Loss: 0.0083 Acc: 0.8889

Epoch 57/71
----------
train Loss: 0.0037 Acc: 0.9205
val Loss: 0.0052 Acc: 0.8889

Epoch 58/71
----------
train Loss: 0.0037 Acc: 0.8930
val Loss: 0.0072 Acc: 0.8889

Epoch 59/71
----------
train Loss: 0.0038 Acc: 0.9205
val Loss: 0.0059 Acc: 0.8889

Epoch 60/71
----------
train Loss: 0.0041 Acc: 0.9144
val Loss: 0.0075 Acc: 0.8889

Epoch 61/71
----------
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0115 Acc: 0.8796

Epoch 62/71
----------
train Loss: 0.0036 Acc: 0.9113
val Loss: 0.0097 Acc: 0.8889

Epoch 63/71
----------
train Loss: 0.0040 Acc: 0.8960
val Loss: 0.0106 Acc: 0.8889

Epoch 64/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0034 Acc: 0.9205
val Loss: 0.0128 Acc: 0.8889

Epoch 65/71
----------
train Loss: 0.0039 Acc: 0.9052
val Loss: 0.0070 Acc: 0.8889

Epoch 66/71
----------
train Loss: 0.0038 Acc: 0.9144
val Loss: 0.0100 Acc: 0.8889

Epoch 67/71
----------
train Loss: 0.0034 Acc: 0.9113
val Loss: 0.0053 Acc: 0.8889

Epoch 68/71
----------
train Loss: 0.0040 Acc: 0.9113
val Loss: 0.0058 Acc: 0.8889

Epoch 69/71
----------
train Loss: 0.0032 Acc: 0.9297
val Loss: 0.0089 Acc: 0.8889

Epoch 70/71
----------
train Loss: 0.0039 Acc: 0.9144
val Loss: 0.0080 Acc: 0.8889

Epoch 71/71
----------
train Loss: 0.0035 Acc: 0.9297
val Loss: 0.0084 Acc: 0.8796

Training complete in 2m 43s
Best val Acc: 0.888889

---Fine tuning.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0043 Acc: 0.8930
val Loss: 0.0058 Acc: 0.9074

Epoch 1/71
----------
train Loss: 0.0028 Acc: 0.9388
val Loss: 0.0092 Acc: 0.8148

Epoch 2/71
----------
train Loss: 0.0018 Acc: 0.9664
val Loss: 0.0065 Acc: 0.8426

Epoch 3/71
----------
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0065 Acc: 0.8981

Epoch 4/71
----------
train Loss: 0.0009 Acc: 0.9847
val Loss: 0.0072 Acc: 0.8611

Epoch 5/71
----------
train Loss: 0.0005 Acc: 0.9939
val Loss: 0.0127 Acc: 0.8333

Epoch 6/71
----------
train Loss: 0.0004 Acc: 0.9878
val Loss: 0.0124 Acc: 0.8796

Epoch 7/71
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8981

Epoch 8/71
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9939
val Loss: 0.0115 Acc: 0.9167

Epoch 9/71
----------
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0045 Acc: 0.9352

Epoch 10/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 11/71
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0070 Acc: 0.9167

Epoch 12/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9259

Epoch 13/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.9167

Epoch 14/71
----------
train Loss: 0.0001 Acc: 0.9969
val Loss: 0.0039 Acc: 0.9259

Epoch 15/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9259

Epoch 16/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9167

Epoch 17/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0103 Acc: 0.9259

Epoch 18/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9259

Epoch 19/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0106 Acc: 0.9259

Epoch 20/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0101 Acc: 0.9167

Epoch 21/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9167

Epoch 22/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9167

Epoch 23/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0114 Acc: 0.9167

Epoch 24/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9167

Epoch 25/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9167

Epoch 26/71
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0090 Acc: 0.9167

Epoch 27/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 28/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9167

Epoch 29/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9167

Epoch 30/71
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0037 Acc: 0.9167

Epoch 31/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9167

Epoch 32/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9167

Epoch 33/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9167

Epoch 34/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.9167

Epoch 35/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0130 Acc: 0.9167

Epoch 36/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9167

Epoch 37/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9167

Epoch 38/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 39/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9167

Epoch 40/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0103 Acc: 0.9167

Epoch 41/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9167

Epoch 42/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.9167

Epoch 43/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9167

Epoch 44/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9167

Epoch 45/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9167

Epoch 46/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.9167

Epoch 47/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9167

Epoch 48/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 49/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0133 Acc: 0.9167

Epoch 50/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 51/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.9167

Epoch 52/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9167

Epoch 53/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.9167

Epoch 54/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.9167

Epoch 55/71
----------
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0049 Acc: 0.9167

Epoch 56/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.9167

Epoch 57/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9167

Epoch 58/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.9167

Epoch 59/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9167

Epoch 60/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9167

Epoch 61/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0088 Acc: 0.9167

Epoch 62/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.9167

Epoch 63/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0081 Acc: 0.9167

Epoch 64/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0121 Acc: 0.9167

Epoch 65/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9167

Epoch 66/71
----------
train Loss: 0.0001 Acc: 0.9969
val Loss: 0.0039 Acc: 0.9167

Epoch 67/71
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9167

Epoch 68/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9167

Epoch 69/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9167

Epoch 70/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0101 Acc: 0.9167

Epoch 71/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0120 Acc: 0.9167

Training complete in 2m 56s
Best val Acc: 0.935185

---Testing---
Test accuracy: 0.979310
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.976863937168041, std: 0.01088789047522747
--------------------

run info[val: 0.3, epoch: 58, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0223 Acc: 0.2918
val Loss: 0.0213 Acc: 0.5692

Epoch 1/57
----------
train Loss: 0.0157 Acc: 0.5738
val Loss: 0.0235 Acc: 0.5000

Epoch 2/57
----------
train Loss: 0.0142 Acc: 0.5639
val Loss: 0.0201 Acc: 0.5231

Epoch 3/57
----------
train Loss: 0.0140 Acc: 0.6492
val Loss: 0.0167 Acc: 0.6308

Epoch 4/57
----------
train Loss: 0.0108 Acc: 0.6885
val Loss: 0.0141 Acc: 0.6538

Epoch 5/57
----------
LR is set to 0.001
train Loss: 0.0087 Acc: 0.7541
val Loss: 0.0128 Acc: 0.6692

Epoch 6/57
----------
train Loss: 0.0092 Acc: 0.7672
val Loss: 0.0129 Acc: 0.7462

Epoch 7/57
----------
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0091 Acc: 0.7385

Epoch 8/57
----------
train Loss: 0.0060 Acc: 0.8492
val Loss: 0.0115 Acc: 0.7462

Epoch 9/57
----------
train Loss: 0.0060 Acc: 0.8295
val Loss: 0.0108 Acc: 0.7538

Epoch 10/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0062 Acc: 0.8590
val Loss: 0.0108 Acc: 0.7385

Epoch 11/57
----------
train Loss: 0.0074 Acc: 0.8459
val Loss: 0.0104 Acc: 0.7385

Epoch 12/57
----------
train Loss: 0.0073 Acc: 0.8361
val Loss: 0.0089 Acc: 0.7385

Epoch 13/57
----------
train Loss: 0.0084 Acc: 0.8426
val Loss: 0.0107 Acc: 0.7308

Epoch 14/57
----------
train Loss: 0.0056 Acc: 0.8426
val Loss: 0.0109 Acc: 0.7385

Epoch 15/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0057 Acc: 0.8525
val Loss: 0.0104 Acc: 0.7385

Epoch 16/57
----------
train Loss: 0.0067 Acc: 0.8492
val Loss: 0.0110 Acc: 0.7462

Epoch 17/57
----------
train Loss: 0.0065 Acc: 0.8459
val Loss: 0.0106 Acc: 0.7462

Epoch 18/57
----------
train Loss: 0.0057 Acc: 0.8590
val Loss: 0.0103 Acc: 0.7538

Epoch 19/57
----------
train Loss: 0.0054 Acc: 0.8623
val Loss: 0.0121 Acc: 0.7308

Epoch 20/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0063 Acc: 0.8459
val Loss: 0.0110 Acc: 0.7462

Epoch 21/57
----------
train Loss: 0.0080 Acc: 0.8590
val Loss: 0.0109 Acc: 0.7538

Epoch 22/57
----------
train Loss: 0.0115 Acc: 0.8623
val Loss: 0.0112 Acc: 0.7385

Epoch 23/57
----------
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0112 Acc: 0.7308

Epoch 24/57
----------
train Loss: 0.0049 Acc: 0.8721
val Loss: 0.0093 Acc: 0.7308

Epoch 25/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0092 Acc: 0.8525
val Loss: 0.0103 Acc: 0.7462

Epoch 26/57
----------
train Loss: 0.0059 Acc: 0.8525
val Loss: 0.0127 Acc: 0.7231

Epoch 27/57
----------
train Loss: 0.0051 Acc: 0.8787
val Loss: 0.0109 Acc: 0.7308

Epoch 28/57
----------
train Loss: 0.0059 Acc: 0.8590
val Loss: 0.0103 Acc: 0.7385

Epoch 29/57
----------
train Loss: 0.0060 Acc: 0.8754
val Loss: 0.0106 Acc: 0.7538

Epoch 30/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0066 Acc: 0.8656
val Loss: 0.0105 Acc: 0.7538

Epoch 31/57
----------
train Loss: 0.0066 Acc: 0.8721
val Loss: 0.0102 Acc: 0.7615

Epoch 32/57
----------
train Loss: 0.0069 Acc: 0.8459
val Loss: 0.0105 Acc: 0.7615

Epoch 33/57
----------
train Loss: 0.0065 Acc: 0.8590
val Loss: 0.0107 Acc: 0.7462

Epoch 34/57
----------
train Loss: 0.0054 Acc: 0.8590
val Loss: 0.0099 Acc: 0.7538

Epoch 35/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0061 Acc: 0.8557
val Loss: 0.0113 Acc: 0.7385

Epoch 36/57
----------
train Loss: 0.0056 Acc: 0.8590
val Loss: 0.0100 Acc: 0.7615

Epoch 37/57
----------
train Loss: 0.0070 Acc: 0.8590
val Loss: 0.0097 Acc: 0.7538

Epoch 38/57
----------
train Loss: 0.0085 Acc: 0.8492
val Loss: 0.0102 Acc: 0.7615

Epoch 39/57
----------
train Loss: 0.0063 Acc: 0.8492
val Loss: 0.0109 Acc: 0.7385

Epoch 40/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0068 Acc: 0.8557
val Loss: 0.0105 Acc: 0.7462

Epoch 41/57
----------
train Loss: 0.0063 Acc: 0.8525
val Loss: 0.0106 Acc: 0.7462

Epoch 42/57
----------
train Loss: 0.0057 Acc: 0.8525
val Loss: 0.0099 Acc: 0.7615

Epoch 43/57
----------
train Loss: 0.0063 Acc: 0.8557
val Loss: 0.0099 Acc: 0.7692

Epoch 44/57
----------
train Loss: 0.0064 Acc: 0.8525
val Loss: 0.0101 Acc: 0.7538

Epoch 45/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0066 Acc: 0.8623
val Loss: 0.0102 Acc: 0.7308

Epoch 46/57
----------
train Loss: 0.0066 Acc: 0.8393
val Loss: 0.0101 Acc: 0.7308

Epoch 47/57
----------
train Loss: 0.0052 Acc: 0.8393
val Loss: 0.0102 Acc: 0.7231

Epoch 48/57
----------
train Loss: 0.0067 Acc: 0.8590
val Loss: 0.0111 Acc: 0.7154

Epoch 49/57
----------
train Loss: 0.0075 Acc: 0.8557
val Loss: 0.0108 Acc: 0.7231

Epoch 50/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0056 Acc: 0.8492
val Loss: 0.0091 Acc: 0.7308

Epoch 51/57
----------
train Loss: 0.0059 Acc: 0.8426
val Loss: 0.0102 Acc: 0.7385

Epoch 52/57
----------
train Loss: 0.0085 Acc: 0.8656
val Loss: 0.0113 Acc: 0.7308

Epoch 53/57
----------
train Loss: 0.0054 Acc: 0.8656
val Loss: 0.0109 Acc: 0.7308

Epoch 54/57
----------
train Loss: 0.0061 Acc: 0.8525
val Loss: 0.0100 Acc: 0.7615

Epoch 55/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0058 Acc: 0.8656
val Loss: 0.0103 Acc: 0.7538

Epoch 56/57
----------
train Loss: 0.0052 Acc: 0.8590
val Loss: 0.0105 Acc: 0.7538

Epoch 57/57
----------
train Loss: 0.0094 Acc: 0.8623
val Loss: 0.0095 Acc: 0.7615

Training complete in 2m 10s
Best val Acc: 0.769231

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8393
val Loss: 0.0082 Acc: 0.8308

Epoch 1/57
----------
train Loss: 0.0045 Acc: 0.9180
val Loss: 0.0076 Acc: 0.8462

Epoch 2/57
----------
train Loss: 0.0032 Acc: 0.9639
val Loss: 0.0110 Acc: 0.7769

Epoch 3/57
----------
train Loss: 0.0046 Acc: 0.9311
val Loss: 0.0169 Acc: 0.6385

Epoch 4/57
----------
train Loss: 0.0044 Acc: 0.8984
val Loss: 0.0375 Acc: 0.4615

Epoch 5/57
----------
LR is set to 0.001
train Loss: 0.0079 Acc: 0.8787
val Loss: 0.0216 Acc: 0.6538

Epoch 6/57
----------
train Loss: 0.0032 Acc: 0.9082
val Loss: 0.0117 Acc: 0.8000

Epoch 7/57
----------
train Loss: 0.0037 Acc: 0.9508
val Loss: 0.0084 Acc: 0.8154

Epoch 8/57
----------
train Loss: 0.0015 Acc: 0.9639
val Loss: 0.0110 Acc: 0.8231

Epoch 9/57
----------
train Loss: 0.0013 Acc: 0.9475
val Loss: 0.0089 Acc: 0.8231

Epoch 10/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0013 Acc: 0.9803
val Loss: 0.0112 Acc: 0.8538

Epoch 11/57
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0104 Acc: 0.8462

Epoch 12/57
----------
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0096 Acc: 0.8462

Epoch 13/57
----------
train Loss: 0.0018 Acc: 0.9770
val Loss: 0.0095 Acc: 0.8462

Epoch 14/57
----------
train Loss: 0.0037 Acc: 0.9902
val Loss: 0.0097 Acc: 0.8231

Epoch 15/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0009 Acc: 0.9738
val Loss: 0.0082 Acc: 0.8385

Epoch 16/57
----------
train Loss: 0.0077 Acc: 0.9672
val Loss: 0.0093 Acc: 0.8385

Epoch 17/57
----------
train Loss: 0.0055 Acc: 0.9869
val Loss: 0.0084 Acc: 0.8615

Epoch 18/57
----------
train Loss: 0.0073 Acc: 0.9836
val Loss: 0.0102 Acc: 0.8385

Epoch 19/57
----------
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0114 Acc: 0.8385

Epoch 20/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0036 Acc: 0.9770
val Loss: 0.0122 Acc: 0.8615

Epoch 21/57
----------
train Loss: 0.0082 Acc: 0.9770
val Loss: 0.0104 Acc: 0.8462

Epoch 22/57
----------
train Loss: 0.0012 Acc: 0.9770
val Loss: 0.0077 Acc: 0.8615

Epoch 23/57
----------
train Loss: 0.0012 Acc: 0.9902
val Loss: 0.0099 Acc: 0.8385

Epoch 24/57
----------
train Loss: 0.0031 Acc: 0.9836
val Loss: 0.0108 Acc: 0.8462

Epoch 25/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0008 Acc: 0.9869
val Loss: 0.0098 Acc: 0.8615

Epoch 26/57
----------
train Loss: 0.0015 Acc: 0.9869
val Loss: 0.0112 Acc: 0.8462

Epoch 27/57
----------
train Loss: 0.0032 Acc: 0.9770
val Loss: 0.0100 Acc: 0.8385

Epoch 28/57
----------
train Loss: 0.0012 Acc: 0.9934
val Loss: 0.0075 Acc: 0.8308

Epoch 29/57
----------
train Loss: 0.0018 Acc: 0.9934
val Loss: 0.0097 Acc: 0.8308

Epoch 30/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0072 Acc: 0.8385

Epoch 31/57
----------
train Loss: 0.0027 Acc: 0.9869
val Loss: 0.0091 Acc: 0.8385

Epoch 32/57
----------
train Loss: 0.0009 Acc: 0.9902
val Loss: 0.0082 Acc: 0.8308

Epoch 33/57
----------
train Loss: 0.0033 Acc: 0.9934
val Loss: 0.0089 Acc: 0.8538

Epoch 34/57
----------
train Loss: 0.0015 Acc: 0.9902
val Loss: 0.0075 Acc: 0.8385

Epoch 35/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0007 Acc: 0.9967
val Loss: 0.0093 Acc: 0.8462

Epoch 36/57
----------
train Loss: 0.0008 Acc: 0.9902
val Loss: 0.0116 Acc: 0.8385

Epoch 37/57
----------
train Loss: 0.0019 Acc: 0.9770
val Loss: 0.0083 Acc: 0.8615

Epoch 38/57
----------
train Loss: 0.0050 Acc: 0.9836
val Loss: 0.0124 Acc: 0.8385

Epoch 39/57
----------
train Loss: 0.0018 Acc: 0.9803
val Loss: 0.0104 Acc: 0.8462

Epoch 40/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0041 Acc: 0.9836
val Loss: 0.0073 Acc: 0.8538

Epoch 41/57
----------
train Loss: 0.0029 Acc: 0.9869
val Loss: 0.0106 Acc: 0.8615

Epoch 42/57
----------
train Loss: 0.0082 Acc: 0.9836
val Loss: 0.0115 Acc: 0.8538

Epoch 43/57
----------
train Loss: 0.0010 Acc: 0.9902
val Loss: 0.0086 Acc: 0.8538

Epoch 44/57
----------
train Loss: 0.0029 Acc: 0.9934
val Loss: 0.0108 Acc: 0.8538

Epoch 45/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0126 Acc: 0.8538

Epoch 46/57
----------
train Loss: 0.0020 Acc: 0.9902
val Loss: 0.0069 Acc: 0.8462

Epoch 47/57
----------
train Loss: 0.0011 Acc: 0.9934
val Loss: 0.0101 Acc: 0.8538

Epoch 48/57
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0083 Acc: 0.8615

Epoch 49/57
----------
train Loss: 0.0020 Acc: 0.9902
val Loss: 0.0092 Acc: 0.8615

Epoch 50/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0021 Acc: 0.9902
val Loss: 0.0090 Acc: 0.8538

Epoch 51/57
----------
train Loss: 0.0019 Acc: 0.9869
val Loss: 0.0091 Acc: 0.8538

Epoch 52/57
----------
train Loss: 0.0018 Acc: 0.9902
val Loss: 0.0098 Acc: 0.8462

Epoch 53/57
----------
train Loss: 0.0024 Acc: 0.9902
val Loss: 0.0085 Acc: 0.8385

Epoch 54/57
----------
train Loss: 0.0008 Acc: 0.9902
val Loss: 0.0089 Acc: 0.8462

Epoch 55/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0009 Acc: 0.9836
val Loss: 0.0112 Acc: 0.8308

Epoch 56/57
----------
train Loss: 0.0020 Acc: 0.9869
val Loss: 0.0107 Acc: 0.8231

Epoch 57/57
----------
train Loss: 0.0013 Acc: 0.9836
val Loss: 0.0099 Acc: 0.8462

Training complete in 2m 22s
Best val Acc: 0.861538

---Testing---
Test accuracy: 0.944828
--------------------
Accuracy of Dasyatiformes : 89 %
Accuracy of Myliobatiformes : 96 %
Accuracy of Rajiformes : 89 %
Accuracy of Rhinobatiformes : 90 %
Accuracy of Torpediniformes : 98 %
mean: 0.9289649125643213, std: 0.037994208014080814

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 88, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0161 Acc: 0.4184
val Loss: 0.0384 Acc: 0.3953

Epoch 1/87
----------
train Loss: 0.0121 Acc: 0.5612
val Loss: 0.0258 Acc: 0.6047

Epoch 2/87
----------
train Loss: 0.0089 Acc: 0.7168
val Loss: 0.0205 Acc: 0.6744

Epoch 3/87
----------
train Loss: 0.0071 Acc: 0.7781
val Loss: 0.0177 Acc: 0.6512

Epoch 4/87
----------
train Loss: 0.0057 Acc: 0.8112
val Loss: 0.0160 Acc: 0.7209

Epoch 5/87
----------
train Loss: 0.0051 Acc: 0.8291
val Loss: 0.0140 Acc: 0.8372

Epoch 6/87
----------
train Loss: 0.0048 Acc: 0.8418
val Loss: 0.0142 Acc: 0.8140

Epoch 7/87
----------
LR is set to 0.001
train Loss: 0.0038 Acc: 0.8673
val Loss: 0.0142 Acc: 0.8140

Epoch 8/87
----------
train Loss: 0.0038 Acc: 0.8852
val Loss: 0.0142 Acc: 0.8140

Epoch 9/87
----------
train Loss: 0.0037 Acc: 0.8852
val Loss: 0.0140 Acc: 0.8140

Epoch 10/87
----------
train Loss: 0.0033 Acc: 0.8954
val Loss: 0.0138 Acc: 0.8140

Epoch 11/87
----------
train Loss: 0.0036 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 12/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0136 Acc: 0.8140

Epoch 13/87
----------
train Loss: 0.0034 Acc: 0.9005
val Loss: 0.0138 Acc: 0.8140

Epoch 14/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0038 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 15/87
----------
train Loss: 0.0034 Acc: 0.9005
val Loss: 0.0137 Acc: 0.8140

Epoch 16/87
----------
train Loss: 0.0036 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 17/87
----------
train Loss: 0.0035 Acc: 0.8929
val Loss: 0.0137 Acc: 0.8140

Epoch 18/87
----------
train Loss: 0.0037 Acc: 0.8980
val Loss: 0.0137 Acc: 0.8140

Epoch 19/87
----------
train Loss: 0.0034 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 20/87
----------
train Loss: 0.0034 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 21/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0033 Acc: 0.9056
val Loss: 0.0137 Acc: 0.8140

Epoch 22/87
----------
train Loss: 0.0035 Acc: 0.8878
val Loss: 0.0138 Acc: 0.8140

Epoch 23/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 24/87
----------
train Loss: 0.0035 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 25/87
----------
train Loss: 0.0036 Acc: 0.8878
val Loss: 0.0137 Acc: 0.8140

Epoch 26/87
----------
train Loss: 0.0037 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 27/87
----------
train Loss: 0.0033 Acc: 0.9107
val Loss: 0.0137 Acc: 0.8140

Epoch 28/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.9056
val Loss: 0.0137 Acc: 0.8140

Epoch 29/87
----------
train Loss: 0.0036 Acc: 0.8776
val Loss: 0.0137 Acc: 0.8140

Epoch 30/87
----------
train Loss: 0.0036 Acc: 0.8980
val Loss: 0.0137 Acc: 0.8140

Epoch 31/87
----------
train Loss: 0.0035 Acc: 0.9005
val Loss: 0.0138 Acc: 0.8140

Epoch 32/87
----------
train Loss: 0.0035 Acc: 0.9082
val Loss: 0.0137 Acc: 0.8140

Epoch 33/87
----------
train Loss: 0.0037 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 34/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 35/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0037 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 36/87
----------
train Loss: 0.0033 Acc: 0.9158
val Loss: 0.0137 Acc: 0.8140

Epoch 37/87
----------
train Loss: 0.0035 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 38/87
----------
train Loss: 0.0034 Acc: 0.8878
val Loss: 0.0137 Acc: 0.8140

Epoch 39/87
----------
train Loss: 0.0035 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 40/87
----------
train Loss: 0.0036 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 41/87
----------
train Loss: 0.0036 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 42/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 43/87
----------
train Loss: 0.0036 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 44/87
----------
train Loss: 0.0035 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 45/87
----------
train Loss: 0.0033 Acc: 0.8878
val Loss: 0.0137 Acc: 0.8140

Epoch 46/87
----------
train Loss: 0.0035 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 47/87
----------
train Loss: 0.0034 Acc: 0.9005
val Loss: 0.0137 Acc: 0.8140

Epoch 48/87
----------
train Loss: 0.0035 Acc: 0.8852
val Loss: 0.0137 Acc: 0.8140

Epoch 49/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0034 Acc: 0.8929
val Loss: 0.0137 Acc: 0.8140

Epoch 50/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0138 Acc: 0.8140

Epoch 51/87
----------
train Loss: 0.0035 Acc: 0.9056
val Loss: 0.0137 Acc: 0.8140

Epoch 52/87
----------
train Loss: 0.0034 Acc: 0.9031
val Loss: 0.0136 Acc: 0.8140

Epoch 53/87
----------
train Loss: 0.0035 Acc: 0.8980
val Loss: 0.0136 Acc: 0.8140

Epoch 54/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 55/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 56/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0036 Acc: 0.8929
val Loss: 0.0137 Acc: 0.8140

Epoch 57/87
----------
train Loss: 0.0035 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 58/87
----------
train Loss: 0.0033 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 59/87
----------
train Loss: 0.0034 Acc: 0.9107
val Loss: 0.0137 Acc: 0.8140

Epoch 60/87
----------
train Loss: 0.0033 Acc: 0.9005
val Loss: 0.0137 Acc: 0.8140

Epoch 61/87
----------
train Loss: 0.0034 Acc: 0.9031
val Loss: 0.0137 Acc: 0.8140

Epoch 62/87
----------
train Loss: 0.0035 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 63/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0035 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 64/87
----------
train Loss: 0.0034 Acc: 0.9031
val Loss: 0.0137 Acc: 0.8140

Epoch 65/87
----------
train Loss: 0.0034 Acc: 0.9056
val Loss: 0.0137 Acc: 0.8140

Epoch 66/87
----------
train Loss: 0.0036 Acc: 0.9107
val Loss: 0.0137 Acc: 0.8140

Epoch 67/87
----------
train Loss: 0.0036 Acc: 0.8801
val Loss: 0.0137 Acc: 0.8140

Epoch 68/87
----------
train Loss: 0.0034 Acc: 0.9107
val Loss: 0.0136 Acc: 0.8140

Epoch 69/87
----------
train Loss: 0.0033 Acc: 0.8903
val Loss: 0.0136 Acc: 0.8140

Epoch 70/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9184
val Loss: 0.0137 Acc: 0.8140

Epoch 71/87
----------
train Loss: 0.0035 Acc: 0.8929
val Loss: 0.0137 Acc: 0.8140

Epoch 72/87
----------
train Loss: 0.0035 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 73/87
----------
train Loss: 0.0036 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 74/87
----------
train Loss: 0.0037 Acc: 0.8801
val Loss: 0.0136 Acc: 0.8140

Epoch 75/87
----------
train Loss: 0.0033 Acc: 0.9184
val Loss: 0.0137 Acc: 0.8140

Epoch 76/87
----------
train Loss: 0.0033 Acc: 0.8980
val Loss: 0.0136 Acc: 0.8140

Epoch 77/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0034 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 78/87
----------
train Loss: 0.0034 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 79/87
----------
train Loss: 0.0037 Acc: 0.8750
val Loss: 0.0137 Acc: 0.8140

Epoch 80/87
----------
train Loss: 0.0032 Acc: 0.9133
val Loss: 0.0137 Acc: 0.8140

Epoch 81/87
----------
train Loss: 0.0035 Acc: 0.8929
val Loss: 0.0137 Acc: 0.8140

Epoch 82/87
----------
train Loss: 0.0034 Acc: 0.8954
val Loss: 0.0136 Acc: 0.8140

Epoch 83/87
----------
train Loss: 0.0035 Acc: 0.9031
val Loss: 0.0136 Acc: 0.8140

Epoch 84/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0033 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 85/87
----------
train Loss: 0.0035 Acc: 0.8954
val Loss: 0.0137 Acc: 0.8140

Epoch 86/87
----------
train Loss: 0.0037 Acc: 0.8801
val Loss: 0.0136 Acc: 0.8140

Epoch 87/87
----------
train Loss: 0.0036 Acc: 0.8801
val Loss: 0.0137 Acc: 0.8140

Training complete in 3m 4s
Best val Acc: 0.837209

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0042 Acc: 0.8750
val Loss: 0.0125 Acc: 0.8605

Epoch 1/87
----------
train Loss: 0.0025 Acc: 0.9515
val Loss: 0.0119 Acc: 0.8372

Epoch 2/87
----------
train Loss: 0.0015 Acc: 0.9796
val Loss: 0.0114 Acc: 0.8605

Epoch 3/87
----------
train Loss: 0.0008 Acc: 0.9872
val Loss: 0.0117 Acc: 0.8605

Epoch 4/87
----------
train Loss: 0.0005 Acc: 0.9923
val Loss: 0.0116 Acc: 0.8605

Epoch 5/87
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0122 Acc: 0.8372

Epoch 6/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8372

Epoch 7/87
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8372

Epoch 8/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8372

Epoch 9/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 10/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 11/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 12/87
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 13/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 14/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 15/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 16/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 17/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 18/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 19/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 20/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 21/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 22/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 23/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8372

Epoch 24/87
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 25/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 26/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 27/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 28/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 29/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 30/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 31/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 32/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 33/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 34/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 35/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 36/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 37/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 38/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 39/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 40/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 41/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 42/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8372

Epoch 43/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 44/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 45/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 46/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 47/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 48/87
----------
train Loss: 0.0002 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 49/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 50/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 51/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8372

Epoch 52/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8605

Epoch 53/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 54/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 55/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 56/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 57/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 58/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 59/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 60/87
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 61/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 62/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 63/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 64/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 65/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 66/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 67/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 68/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 69/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 70/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 71/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 72/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 73/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 74/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 75/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8605

Epoch 76/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 77/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 78/87
----------
train Loss: 0.0002 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Epoch 79/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 80/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 81/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 82/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 83/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 84/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 85/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 86/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8605

Epoch 87/87
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8605

Training complete in 3m 23s
Best val Acc: 0.860465

---Testing---
Test accuracy: 0.931034
--------------------
Accuracy of Dasyatiformes : 82 %
Accuracy of Myliobatiformes : 86 %
Accuracy of Rajiformes : 91 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 96 %
mean: 0.9091732402607144, std: 0.05647815417226528
--------------------

run info[val: 0.15, epoch: 96, randcrop: False, decay: 8]

---Training last layer.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0179 Acc: 0.3162
val Loss: 0.0230 Acc: 0.4000

Epoch 1/95
----------
train Loss: 0.0134 Acc: 0.5216
val Loss: 0.0185 Acc: 0.4923

Epoch 2/95
----------
train Loss: 0.0098 Acc: 0.6486
val Loss: 0.0115 Acc: 0.8154

Epoch 3/95
----------
train Loss: 0.0077 Acc: 0.7541
val Loss: 0.0093 Acc: 0.7692

Epoch 4/95
----------
train Loss: 0.0061 Acc: 0.8027
val Loss: 0.0085 Acc: 0.7846

Epoch 5/95
----------
train Loss: 0.0050 Acc: 0.8270
val Loss: 0.0072 Acc: 0.8769

Epoch 6/95
----------
train Loss: 0.0043 Acc: 0.8568
val Loss: 0.0068 Acc: 0.8615

Epoch 7/95
----------
train Loss: 0.0038 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8769

Epoch 8/95
----------
LR is set to 0.001
train Loss: 0.0034 Acc: 0.8892
val Loss: 0.0066 Acc: 0.8769

Epoch 9/95
----------
train Loss: 0.0034 Acc: 0.8946
val Loss: 0.0064 Acc: 0.8923

Epoch 10/95
----------
train Loss: 0.0031 Acc: 0.9000
val Loss: 0.0063 Acc: 0.8769

Epoch 11/95
----------
train Loss: 0.0031 Acc: 0.9270
val Loss: 0.0063 Acc: 0.8923

Epoch 12/95
----------
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0063 Acc: 0.9077

Epoch 13/95
----------
train Loss: 0.0031 Acc: 0.9135
val Loss: 0.0062 Acc: 0.8923

Epoch 14/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0063 Acc: 0.8923

Epoch 15/95
----------
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0063 Acc: 0.8923

Epoch 16/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0028 Acc: 0.9189
val Loss: 0.0063 Acc: 0.8923

Epoch 17/95
----------
train Loss: 0.0030 Acc: 0.9081
val Loss: 0.0063 Acc: 0.8923

Epoch 18/95
----------
train Loss: 0.0030 Acc: 0.9162
val Loss: 0.0063 Acc: 0.8923

Epoch 19/95
----------
train Loss: 0.0030 Acc: 0.9189
val Loss: 0.0063 Acc: 0.8923

Epoch 20/95
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8923

Epoch 21/95
----------
train Loss: 0.0030 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8923

Epoch 22/95
----------
train Loss: 0.0028 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 23/95
----------
train Loss: 0.0031 Acc: 0.9135
val Loss: 0.0062 Acc: 0.8923

Epoch 24/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8923

Epoch 25/95
----------
train Loss: 0.0028 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 26/95
----------
train Loss: 0.0031 Acc: 0.9108
val Loss: 0.0062 Acc: 0.8923

Epoch 27/95
----------
train Loss: 0.0028 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8923

Epoch 28/95
----------
train Loss: 0.0030 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8923

Epoch 29/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 30/95
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 31/95
----------
train Loss: 0.0029 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 32/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0029 Acc: 0.9189
val Loss: 0.0063 Acc: 0.8923

Epoch 33/95
----------
train Loss: 0.0032 Acc: 0.9054
val Loss: 0.0062 Acc: 0.8923

Epoch 34/95
----------
train Loss: 0.0029 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8923

Epoch 35/95
----------
train Loss: 0.0028 Acc: 0.9162
val Loss: 0.0063 Acc: 0.8923

Epoch 36/95
----------
train Loss: 0.0031 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 37/95
----------
train Loss: 0.0029 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8923

Epoch 38/95
----------
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 39/95
----------
train Loss: 0.0029 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8923

Epoch 40/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 41/95
----------
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 42/95
----------
train Loss: 0.0032 Acc: 0.9108
val Loss: 0.0062 Acc: 0.8923

Epoch 43/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 44/95
----------
train Loss: 0.0028 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8923

Epoch 45/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 46/95
----------
train Loss: 0.0027 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8923

Epoch 47/95
----------
train Loss: 0.0029 Acc: 0.9378
val Loss: 0.0062 Acc: 0.8923

Epoch 48/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 49/95
----------
train Loss: 0.0029 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 50/95
----------
train Loss: 0.0029 Acc: 0.9054
val Loss: 0.0062 Acc: 0.8923

Epoch 51/95
----------
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 52/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 53/95
----------
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 54/95
----------
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0063 Acc: 0.8923

Epoch 55/95
----------
train Loss: 0.0029 Acc: 0.9081
val Loss: 0.0062 Acc: 0.8923

Epoch 56/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 57/95
----------
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 58/95
----------
train Loss: 0.0031 Acc: 0.9108
val Loss: 0.0063 Acc: 0.8923

Epoch 59/95
----------
train Loss: 0.0030 Acc: 0.9270
val Loss: 0.0063 Acc: 0.8923

Epoch 60/95
----------
train Loss: 0.0032 Acc: 0.9081
val Loss: 0.0062 Acc: 0.8923

Epoch 61/95
----------
train Loss: 0.0029 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8923

Epoch 62/95
----------
train Loss: 0.0030 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 63/95
----------
train Loss: 0.0029 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8923

Epoch 64/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0063 Acc: 0.8923

Epoch 65/95
----------
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 66/95
----------
train Loss: 0.0030 Acc: 0.9081
val Loss: 0.0062 Acc: 0.8923

Epoch 67/95
----------
train Loss: 0.0030 Acc: 0.9081
val Loss: 0.0063 Acc: 0.8923

Epoch 68/95
----------
train Loss: 0.0029 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 69/95
----------
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 70/95
----------
train Loss: 0.0028 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8923

Epoch 71/95
----------
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 72/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0031 Acc: 0.9081
val Loss: 0.0062 Acc: 0.8923

Epoch 73/95
----------
train Loss: 0.0028 Acc: 0.9216
val Loss: 0.0063 Acc: 0.8923

Epoch 74/95
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 75/95
----------
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 76/95
----------
train Loss: 0.0029 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 77/95
----------
train Loss: 0.0031 Acc: 0.9108
val Loss: 0.0062 Acc: 0.8923

Epoch 78/95
----------
train Loss: 0.0030 Acc: 0.9108
val Loss: 0.0062 Acc: 0.8923

Epoch 79/95
----------
train Loss: 0.0029 Acc: 0.9162
val Loss: 0.0062 Acc: 0.8923

Epoch 80/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0028 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 81/95
----------
train Loss: 0.0028 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8923

Epoch 82/95
----------
train Loss: 0.0029 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8923

Epoch 83/95
----------
train Loss: 0.0032 Acc: 0.9081
val Loss: 0.0062 Acc: 0.8923

Epoch 84/95
----------
train Loss: 0.0028 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 85/95
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 86/95
----------
train Loss: 0.0028 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8923

Epoch 87/95
----------
train Loss: 0.0030 Acc: 0.9162
val Loss: 0.0062 Acc: 0.8923

Epoch 88/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0030 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8923

Epoch 89/95
----------
train Loss: 0.0028 Acc: 0.9351
val Loss: 0.0063 Acc: 0.8923

Epoch 90/95
----------
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 91/95
----------
train Loss: 0.0030 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 92/95
----------
train Loss: 0.0028 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 93/95
----------
train Loss: 0.0030 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8923

Epoch 94/95
----------
train Loss: 0.0029 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8923

Epoch 95/95
----------
train Loss: 0.0029 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8923

Training complete in 3m 26s
Best val Acc: 0.907692

---Fine tuning.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0062 Acc: 0.8769

Epoch 1/95
----------
train Loss: 0.0016 Acc: 0.9703
val Loss: 0.0060 Acc: 0.8615

Epoch 2/95
----------
train Loss: 0.0009 Acc: 0.9919
val Loss: 0.0055 Acc: 0.8923

Epoch 3/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8615

Epoch 4/95
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0061 Acc: 0.8769

Epoch 5/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.9077

Epoch 6/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9077

Epoch 7/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9077

Epoch 8/95
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9077

Epoch 9/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9077

Epoch 10/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8923

Epoch 11/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9077

Epoch 12/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 13/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 14/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 15/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 16/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 17/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 18/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 19/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 20/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 21/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 22/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 23/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 24/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 25/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 26/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 27/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 28/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 29/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 30/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 31/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 32/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 33/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 34/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 35/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 36/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 37/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 38/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 39/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 40/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 41/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 42/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 43/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 44/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 45/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 46/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 47/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 48/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 49/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 50/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 51/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 52/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 53/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 54/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 55/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 56/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 57/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 58/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 59/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 60/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 61/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 62/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 63/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 64/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 65/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 66/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 67/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 68/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 69/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 70/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 71/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 72/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 73/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 74/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 75/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 76/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 77/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 78/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 79/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 80/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 81/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 82/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 83/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 84/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 85/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 86/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 87/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 88/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 89/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 90/95
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 91/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 92/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 93/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 94/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 95/95
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Training complete in 3m 44s
Best val Acc: 0.907692

---Testing---
Test accuracy: 0.986207
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 99 %
mean: 0.9815997429271646, std: 0.010527768337334459
--------------------

run info[val: 0.2, epoch: 98, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/97
----------
LR is set to 0.01
train Loss: 0.0190 Acc: 0.3190
val Loss: 0.0149 Acc: 0.5402

Epoch 1/97
----------
train Loss: 0.0143 Acc: 0.5402
val Loss: 0.0131 Acc: 0.5402

Epoch 2/97
----------
train Loss: 0.0103 Acc: 0.6523
val Loss: 0.0081 Acc: 0.7701

Epoch 3/97
----------
train Loss: 0.0077 Acc: 0.7989
val Loss: 0.0084 Acc: 0.6897

Epoch 4/97
----------
train Loss: 0.0074 Acc: 0.7586
val Loss: 0.0067 Acc: 0.8161

Epoch 5/97
----------
train Loss: 0.0058 Acc: 0.8420
val Loss: 0.0064 Acc: 0.8046

Epoch 6/97
----------
train Loss: 0.0051 Acc: 0.8621
val Loss: 0.0060 Acc: 0.8276

Epoch 7/97
----------
train Loss: 0.0046 Acc: 0.8333
val Loss: 0.0064 Acc: 0.8046

Epoch 8/97
----------
train Loss: 0.0035 Acc: 0.9023
val Loss: 0.0054 Acc: 0.8621

Epoch 9/97
----------
LR is set to 0.001
train Loss: 0.0039 Acc: 0.8822
val Loss: 0.0052 Acc: 0.8621

Epoch 10/97
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0051 Acc: 0.8621

Epoch 11/97
----------
train Loss: 0.0031 Acc: 0.9339
val Loss: 0.0051 Acc: 0.8736

Epoch 12/97
----------
train Loss: 0.0033 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 13/97
----------
train Loss: 0.0030 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8621

Epoch 14/97
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0054 Acc: 0.8621

Epoch 15/97
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8621

Epoch 16/97
----------
train Loss: 0.0033 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8621

Epoch 17/97
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8621

Epoch 18/97
----------
LR is set to 0.00010000000000000002
train Loss: 0.0035 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8621

Epoch 19/97
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8621

Epoch 20/97
----------
train Loss: 0.0034 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 21/97
----------
train Loss: 0.0029 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8621

Epoch 22/97
----------
train Loss: 0.0028 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Epoch 23/97
----------
train Loss: 0.0028 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Epoch 24/97
----------
train Loss: 0.0034 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 25/97
----------
train Loss: 0.0031 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 26/97
----------
train Loss: 0.0031 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8621

Epoch 27/97
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0052 Acc: 0.8621

Epoch 28/97
----------
train Loss: 0.0032 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 29/97
----------
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8621

Epoch 30/97
----------
train Loss: 0.0030 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8621

Epoch 31/97
----------
train Loss: 0.0028 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8621

Epoch 32/97
----------
train Loss: 0.0030 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 33/97
----------
train Loss: 0.0031 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 34/97
----------
train Loss: 0.0029 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8621

Epoch 35/97
----------
train Loss: 0.0033 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8621

Epoch 36/97
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0032 Acc: 0.9368
val Loss: 0.0052 Acc: 0.8621

Epoch 37/97
----------
train Loss: 0.0030 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Epoch 38/97
----------
train Loss: 0.0031 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 39/97
----------
train Loss: 0.0030 Acc: 0.9109
val Loss: 0.0052 Acc: 0.8621

Epoch 40/97
----------
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 41/97
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 42/97
----------
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 43/97
----------
train Loss: 0.0029 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8621

Epoch 44/97
----------
train Loss: 0.0030 Acc: 0.9109
val Loss: 0.0052 Acc: 0.8621

Epoch 45/97
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8621

Epoch 46/97
----------
train Loss: 0.0031 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 47/97
----------
train Loss: 0.0028 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 48/97
----------
train Loss: 0.0031 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 49/97
----------
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Epoch 50/97
----------
train Loss: 0.0032 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8621

Epoch 51/97
----------
train Loss: 0.0032 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 52/97
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 53/97
----------
train Loss: 0.0032 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Epoch 54/97
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0026 Acc: 0.9454
val Loss: 0.0052 Acc: 0.8621

Epoch 55/97
----------
train Loss: 0.0030 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 56/97
----------
train Loss: 0.0033 Acc: 0.9052
val Loss: 0.0052 Acc: 0.8621

Epoch 57/97
----------
train Loss: 0.0031 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8621

Epoch 58/97
----------
train Loss: 0.0034 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Epoch 59/97
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 60/97
----------
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 61/97
----------
train Loss: 0.0030 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Epoch 62/97
----------
train Loss: 0.0035 Acc: 0.8966
val Loss: 0.0052 Acc: 0.8621

Epoch 63/97
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0029 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 64/97
----------
train Loss: 0.0029 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 65/97
----------
train Loss: 0.0033 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8621

Epoch 66/97
----------
train Loss: 0.0031 Acc: 0.9109
val Loss: 0.0052 Acc: 0.8621

Epoch 67/97
----------
train Loss: 0.0033 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 68/97
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 69/97
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 70/97
----------
train Loss: 0.0034 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8621

Epoch 71/97
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8621

Epoch 72/97
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0032 Acc: 0.8966
val Loss: 0.0052 Acc: 0.8621

Epoch 73/97
----------
train Loss: 0.0033 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 74/97
----------
train Loss: 0.0031 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8621

Epoch 75/97
----------
train Loss: 0.0030 Acc: 0.9167
val Loss: 0.0051 Acc: 0.8736

Epoch 76/97
----------
train Loss: 0.0032 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8736

Epoch 77/97
----------
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 78/97
----------
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 79/97
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 80/97
----------
train Loss: 0.0036 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 81/97
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0032 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 82/97
----------
train Loss: 0.0032 Acc: 0.9023
val Loss: 0.0052 Acc: 0.8621

Epoch 83/97
----------
train Loss: 0.0032 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8621

Epoch 84/97
----------
train Loss: 0.0033 Acc: 0.8966
val Loss: 0.0052 Acc: 0.8621

Epoch 85/97
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 86/97
----------
train Loss: 0.0029 Acc: 0.9454
val Loss: 0.0052 Acc: 0.8621

Epoch 87/97
----------
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 88/97
----------
train Loss: 0.0031 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8621

Epoch 89/97
----------
train Loss: 0.0030 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8621

Epoch 90/97
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0030 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8621

Epoch 91/97
----------
train Loss: 0.0028 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8621

Epoch 92/97
----------
train Loss: 0.0034 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8621

Epoch 93/97
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8621

Epoch 94/97
----------
train Loss: 0.0031 Acc: 0.9109
val Loss: 0.0053 Acc: 0.8621

Epoch 95/97
----------
train Loss: 0.0032 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8621

Epoch 96/97
----------
train Loss: 0.0030 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8621

Epoch 97/97
----------
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Training complete in 3m 34s
Best val Acc: 0.873563

---Fine tuning.---
Epoch 0/97
----------
LR is set to 0.01
train Loss: 0.0040 Acc: 0.8937
val Loss: 0.0053 Acc: 0.8506

Epoch 1/97
----------
train Loss: 0.0022 Acc: 0.9425
val Loss: 0.0054 Acc: 0.8506

Epoch 2/97
----------
train Loss: 0.0010 Acc: 0.9914
val Loss: 0.0051 Acc: 0.8506

Epoch 3/97
----------
train Loss: 0.0007 Acc: 0.9943
val Loss: 0.0050 Acc: 0.8391

Epoch 4/97
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8621

Epoch 5/97
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0047 Acc: 0.8621

Epoch 6/97
----------
train Loss: 0.0003 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8736

Epoch 7/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 8/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 9/97
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9080

Epoch 10/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9080

Epoch 11/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 12/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 13/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 14/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 15/97
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0035 Acc: 0.8966

Epoch 16/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 17/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 18/97
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 19/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 20/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 21/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 22/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 23/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 24/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 25/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 26/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 27/97
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 28/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 29/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 30/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 31/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 32/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 33/97
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0036 Acc: 0.8966

Epoch 34/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 35/97
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0036 Acc: 0.8966

Epoch 36/97
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 37/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 38/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 39/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8851

Epoch 40/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 41/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 42/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 43/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 44/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 45/97
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 46/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 47/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 48/97
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0036 Acc: 0.8966

Epoch 49/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 50/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 51/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 52/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 53/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 54/97
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 55/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 56/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 57/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 58/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 59/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 60/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 61/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 62/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 63/97
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 64/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 65/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 66/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 67/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 68/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 69/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 70/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 71/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 72/97
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 73/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 74/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 75/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 76/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 77/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 78/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 79/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 80/97
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0036 Acc: 0.8966

Epoch 81/97
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 82/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 83/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 84/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 85/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 86/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 87/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 88/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 89/97
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0036 Acc: 0.8966

Epoch 90/97
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 91/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 92/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 93/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 94/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 95/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 96/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 97/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Training complete in 3m 56s
Best val Acc: 0.908046

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.978096037578467, std: 0.008429719905853828
--------------------

run info[val: 0.25, epoch: 70, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0202 Acc: 0.3180
val Loss: 0.0278 Acc: 0.4259

Epoch 1/69
----------
train Loss: 0.0161 Acc: 0.5443
val Loss: 0.0191 Acc: 0.5741

Epoch 2/69
----------
train Loss: 0.0114 Acc: 0.6391
val Loss: 0.0130 Acc: 0.6574

Epoch 3/69
----------
train Loss: 0.0097 Acc: 0.7187
val Loss: 0.0092 Acc: 0.7500

Epoch 4/69
----------
train Loss: 0.0077 Acc: 0.7768
val Loss: 0.0101 Acc: 0.7593

Epoch 5/69
----------
train Loss: 0.0063 Acc: 0.8410
val Loss: 0.0124 Acc: 0.8056

Epoch 6/69
----------
train Loss: 0.0064 Acc: 0.8287
val Loss: 0.0074 Acc: 0.8333

Epoch 7/69
----------
train Loss: 0.0059 Acc: 0.8043
val Loss: 0.0057 Acc: 0.8611

Epoch 8/69
----------
train Loss: 0.0053 Acc: 0.8716
val Loss: 0.0079 Acc: 0.8056

Epoch 9/69
----------
train Loss: 0.0038 Acc: 0.8930
val Loss: 0.0071 Acc: 0.7685

Epoch 10/69
----------
LR is set to 0.001
train Loss: 0.0043 Acc: 0.8838
val Loss: 0.0107 Acc: 0.7963

Epoch 11/69
----------
train Loss: 0.0044 Acc: 0.8685
val Loss: 0.0072 Acc: 0.8426

Epoch 12/69
----------
train Loss: 0.0035 Acc: 0.9083
val Loss: 0.0074 Acc: 0.8611

Epoch 13/69
----------
train Loss: 0.0034 Acc: 0.9235
val Loss: 0.0053 Acc: 0.8704

Epoch 14/69
----------
train Loss: 0.0037 Acc: 0.8991
val Loss: 0.0073 Acc: 0.8889

Epoch 15/69
----------
train Loss: 0.0038 Acc: 0.9083
val Loss: 0.0107 Acc: 0.8889

Epoch 16/69
----------
train Loss: 0.0044 Acc: 0.9052
val Loss: 0.0088 Acc: 0.8889

Epoch 17/69
----------
train Loss: 0.0033 Acc: 0.9052
val Loss: 0.0095 Acc: 0.8611

Epoch 18/69
----------
train Loss: 0.0028 Acc: 0.9266
val Loss: 0.0103 Acc: 0.8704

Epoch 19/69
----------
train Loss: 0.0033 Acc: 0.9297
val Loss: 0.0070 Acc: 0.8704

Epoch 20/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9327
val Loss: 0.0066 Acc: 0.8704

Epoch 21/69
----------
train Loss: 0.0030 Acc: 0.9358
val Loss: 0.0055 Acc: 0.8704

Epoch 22/69
----------
train Loss: 0.0033 Acc: 0.9297
val Loss: 0.0047 Acc: 0.8704

Epoch 23/69
----------
train Loss: 0.0031 Acc: 0.9266
val Loss: 0.0076 Acc: 0.8704

Epoch 24/69
----------
train Loss: 0.0033 Acc: 0.9388
val Loss: 0.0125 Acc: 0.8704

Epoch 25/69
----------
train Loss: 0.0027 Acc: 0.9572
val Loss: 0.0080 Acc: 0.8704

Epoch 26/69
----------
train Loss: 0.0028 Acc: 0.9572
val Loss: 0.0053 Acc: 0.8704

Epoch 27/69
----------
train Loss: 0.0031 Acc: 0.9144
val Loss: 0.0053 Acc: 0.8704

Epoch 28/69
----------
train Loss: 0.0035 Acc: 0.9419
val Loss: 0.0054 Acc: 0.8704

Epoch 29/69
----------
train Loss: 0.0030 Acc: 0.9388
val Loss: 0.0058 Acc: 0.8704

Epoch 30/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0034 Acc: 0.9174
val Loss: 0.0081 Acc: 0.8611

Epoch 31/69
----------
train Loss: 0.0035 Acc: 0.9205
val Loss: 0.0057 Acc: 0.8704

Epoch 32/69
----------
train Loss: 0.0033 Acc: 0.9266
val Loss: 0.0146 Acc: 0.8704

Epoch 33/69
----------
train Loss: 0.0042 Acc: 0.9144
val Loss: 0.0089 Acc: 0.8704

Epoch 34/69
----------
train Loss: 0.0030 Acc: 0.9480
val Loss: 0.0048 Acc: 0.8704

Epoch 35/69
----------
train Loss: 0.0032 Acc: 0.9358
val Loss: 0.0105 Acc: 0.8704

Epoch 36/69
----------
train Loss: 0.0031 Acc: 0.9511
val Loss: 0.0059 Acc: 0.8704

Epoch 37/69
----------
train Loss: 0.0029 Acc: 0.9174
val Loss: 0.0051 Acc: 0.8704

Epoch 38/69
----------
train Loss: 0.0032 Acc: 0.9297
val Loss: 0.0151 Acc: 0.8704

Epoch 39/69
----------
train Loss: 0.0030 Acc: 0.9266
val Loss: 0.0051 Acc: 0.8704

Epoch 40/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0035 Acc: 0.9052
val Loss: 0.0141 Acc: 0.8611

Epoch 41/69
----------
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0074 Acc: 0.8704

Epoch 42/69
----------
train Loss: 0.0032 Acc: 0.9388
val Loss: 0.0081 Acc: 0.8704

Epoch 43/69
----------
train Loss: 0.0027 Acc: 0.9358
val Loss: 0.0098 Acc: 0.8704

Epoch 44/69
----------
train Loss: 0.0030 Acc: 0.9388
val Loss: 0.0101 Acc: 0.8704

Epoch 45/69
----------
train Loss: 0.0031 Acc: 0.9144
val Loss: 0.0073 Acc: 0.8704

Epoch 46/69
----------
train Loss: 0.0032 Acc: 0.9235
val Loss: 0.0068 Acc: 0.8611

Epoch 47/69
----------
train Loss: 0.0035 Acc: 0.9266
val Loss: 0.0083 Acc: 0.8611

Epoch 48/69
----------
train Loss: 0.0033 Acc: 0.9235
val Loss: 0.0058 Acc: 0.8611

Epoch 49/69
----------
train Loss: 0.0034 Acc: 0.9358
val Loss: 0.0126 Acc: 0.8611

Epoch 50/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9205
val Loss: 0.0071 Acc: 0.8704

Epoch 51/69
----------
train Loss: 0.0030 Acc: 0.9327
val Loss: 0.0096 Acc: 0.8704

Epoch 52/69
----------
train Loss: 0.0032 Acc: 0.9266
val Loss: 0.0062 Acc: 0.8704

Epoch 53/69
----------
train Loss: 0.0040 Acc: 0.9144
val Loss: 0.0065 Acc: 0.8611

Epoch 54/69
----------
train Loss: 0.0030 Acc: 0.9235
val Loss: 0.0064 Acc: 0.8704

Epoch 55/69
----------
train Loss: 0.0034 Acc: 0.9235
val Loss: 0.0051 Acc: 0.8611

Epoch 56/69
----------
train Loss: 0.0027 Acc: 0.9297
val Loss: 0.0084 Acc: 0.8704

Epoch 57/69
----------
train Loss: 0.0031 Acc: 0.9205
val Loss: 0.0055 Acc: 0.8704

Epoch 58/69
----------
train Loss: 0.0040 Acc: 0.9083
val Loss: 0.0054 Acc: 0.8704

Epoch 59/69
----------
train Loss: 0.0034 Acc: 0.9358
val Loss: 0.0077 Acc: 0.8519

Epoch 60/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0029 Acc: 0.9419
val Loss: 0.0059 Acc: 0.8519

Epoch 61/69
----------
train Loss: 0.0033 Acc: 0.9327
val Loss: 0.0061 Acc: 0.8611

Epoch 62/69
----------
train Loss: 0.0035 Acc: 0.9052
val Loss: 0.0047 Acc: 0.8519

Epoch 63/69
----------
train Loss: 0.0026 Acc: 0.9297
val Loss: 0.0124 Acc: 0.8611

Epoch 64/69
----------
train Loss: 0.0037 Acc: 0.9144
val Loss: 0.0128 Acc: 0.8611

Epoch 65/69
----------
train Loss: 0.0033 Acc: 0.9297
val Loss: 0.0053 Acc: 0.8611

Epoch 66/69
----------
train Loss: 0.0034 Acc: 0.9297
val Loss: 0.0050 Acc: 0.8611

Epoch 67/69
----------
train Loss: 0.0030 Acc: 0.9450
val Loss: 0.0067 Acc: 0.8611

Epoch 68/69
----------
train Loss: 0.0032 Acc: 0.9297
val Loss: 0.0083 Acc: 0.8519

Epoch 69/69
----------
train Loss: 0.0037 Acc: 0.9297
val Loss: 0.0063 Acc: 0.8611

Training complete in 2m 36s
Best val Acc: 0.888889

---Fine tuning.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0032 Acc: 0.9174
val Loss: 0.0062 Acc: 0.8611

Epoch 1/69
----------
train Loss: 0.0031 Acc: 0.9358
val Loss: 0.0095 Acc: 0.8611

Epoch 2/69
----------
train Loss: 0.0014 Acc: 0.9786
val Loss: 0.0067 Acc: 0.8426

Epoch 3/69
----------
train Loss: 0.0012 Acc: 0.9694
val Loss: 0.0077 Acc: 0.8519

Epoch 4/69
----------
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0119 Acc: 0.8333

Epoch 5/69
----------
train Loss: 0.0004 Acc: 0.9878
val Loss: 0.0066 Acc: 0.8519

Epoch 6/69
----------
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0055 Acc: 0.8704

Epoch 7/69
----------
train Loss: 0.0004 Acc: 0.9939
val Loss: 0.0038 Acc: 0.8981

Epoch 8/69
----------
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0037 Acc: 0.9074

Epoch 9/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8704

Epoch 10/69
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0076 Acc: 0.8704

Epoch 11/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8704

Epoch 12/69
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0083 Acc: 0.8889

Epoch 13/69
----------
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0070 Acc: 0.8889

Epoch 14/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8981

Epoch 15/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 16/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8981

Epoch 17/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9074

Epoch 18/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8981

Epoch 19/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8889

Epoch 20/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8981

Epoch 21/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Epoch 22/69
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0122 Acc: 0.9074

Epoch 23/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 24/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9074

Epoch 25/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9074

Epoch 26/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 27/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 28/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8981

Epoch 29/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 30/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.9074

Epoch 31/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8981

Epoch 32/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 33/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8981

Epoch 34/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.9074

Epoch 35/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8981

Epoch 36/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8981

Epoch 37/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Epoch 38/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8889

Epoch 39/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 40/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8889

Epoch 41/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8981

Epoch 42/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8889

Epoch 43/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8981

Epoch 44/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8981

Epoch 45/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 46/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8981

Epoch 47/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8981

Epoch 48/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 49/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 50/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8889

Epoch 51/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 52/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8981

Epoch 53/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8981

Epoch 54/69
----------
train Loss: 0.0001 Acc: 0.9969
val Loss: 0.0033 Acc: 0.8889

Epoch 55/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0103 Acc: 0.8889

Epoch 56/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0121 Acc: 0.8981

Epoch 57/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8981

Epoch 58/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 59/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8981

Epoch 60/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8981

Epoch 61/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 62/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8889

Epoch 63/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8889

Epoch 64/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 65/69
----------
train Loss: 0.0001 Acc: 0.9969
val Loss: 0.0046 Acc: 0.8981

Epoch 66/69
----------
train Loss: 0.0002 Acc: 0.9939
val Loss: 0.0034 Acc: 0.8981

Epoch 67/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8981

Epoch 68/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8981

Epoch 69/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Training complete in 2m 52s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 96 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 99 %
mean: 0.9714631590859393, std: 0.012362587056455807
--------------------

run info[val: 0.3, epoch: 84, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0227 Acc: 0.3049
val Loss: 0.0248 Acc: 0.3769

Epoch 1/83
----------
train Loss: 0.0193 Acc: 0.2984
val Loss: 0.0328 Acc: 0.4077

Epoch 2/83
----------
train Loss: 0.0167 Acc: 0.4984
val Loss: 0.0220 Acc: 0.4154

Epoch 3/83
----------
train Loss: 0.0170 Acc: 0.5115
val Loss: 0.0111 Acc: 0.6692

Epoch 4/83
----------
train Loss: 0.0088 Acc: 0.7115
val Loss: 0.0154 Acc: 0.6308

Epoch 5/83
----------
train Loss: 0.0094 Acc: 0.7082
val Loss: 0.0110 Acc: 0.7308

Epoch 6/83
----------
train Loss: 0.0070 Acc: 0.8295
val Loss: 0.0082 Acc: 0.7769

Epoch 7/83
----------
train Loss: 0.0063 Acc: 0.8426
val Loss: 0.0132 Acc: 0.7231

Epoch 8/83
----------
train Loss: 0.0078 Acc: 0.7377
val Loss: 0.0112 Acc: 0.7615

Epoch 9/83
----------
LR is set to 0.001
train Loss: 0.0055 Acc: 0.8525
val Loss: 0.0082 Acc: 0.8077

Epoch 10/83
----------
train Loss: 0.0051 Acc: 0.8721
val Loss: 0.0093 Acc: 0.8154

Epoch 11/83
----------
train Loss: 0.0037 Acc: 0.8787
val Loss: 0.0081 Acc: 0.8077

Epoch 12/83
----------
train Loss: 0.0043 Acc: 0.8754
val Loss: 0.0087 Acc: 0.8077

Epoch 13/83
----------
train Loss: 0.0032 Acc: 0.9246
val Loss: 0.0088 Acc: 0.7923

Epoch 14/83
----------
train Loss: 0.0037 Acc: 0.9049
val Loss: 0.0067 Acc: 0.8231

Epoch 15/83
----------
train Loss: 0.0047 Acc: 0.8852
val Loss: 0.0083 Acc: 0.8385

Epoch 16/83
----------
train Loss: 0.0030 Acc: 0.9180
val Loss: 0.0097 Acc: 0.8385

Epoch 17/83
----------
train Loss: 0.0053 Acc: 0.9180
val Loss: 0.0091 Acc: 0.8385

Epoch 18/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0041 Acc: 0.9049
val Loss: 0.0074 Acc: 0.8308

Epoch 19/83
----------
train Loss: 0.0051 Acc: 0.8820
val Loss: 0.0080 Acc: 0.8154

Epoch 20/83
----------
train Loss: 0.0039 Acc: 0.8984
val Loss: 0.0082 Acc: 0.8231

Epoch 21/83
----------
train Loss: 0.0047 Acc: 0.9213
val Loss: 0.0088 Acc: 0.8154

Epoch 22/83
----------
train Loss: 0.0032 Acc: 0.8984
val Loss: 0.0069 Acc: 0.8462

Epoch 23/83
----------
train Loss: 0.0035 Acc: 0.8951
val Loss: 0.0088 Acc: 0.8462

Epoch 24/83
----------
train Loss: 0.0038 Acc: 0.9115
val Loss: 0.0090 Acc: 0.8308

Epoch 25/83
----------
train Loss: 0.0038 Acc: 0.9115
val Loss: 0.0078 Acc: 0.8154

Epoch 26/83
----------
train Loss: 0.0033 Acc: 0.9049
val Loss: 0.0074 Acc: 0.8077

Epoch 27/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0047 Acc: 0.9082
val Loss: 0.0072 Acc: 0.8154

Epoch 28/83
----------
train Loss: 0.0041 Acc: 0.9016
val Loss: 0.0082 Acc: 0.8308

Epoch 29/83
----------
train Loss: 0.0043 Acc: 0.8918
val Loss: 0.0114 Acc: 0.8385

Epoch 30/83
----------
train Loss: 0.0059 Acc: 0.9115
val Loss: 0.0070 Acc: 0.8385

Epoch 31/83
----------
train Loss: 0.0093 Acc: 0.9049
val Loss: 0.0079 Acc: 0.8154

Epoch 32/83
----------
train Loss: 0.0060 Acc: 0.9049
val Loss: 0.0085 Acc: 0.8308

Epoch 33/83
----------
train Loss: 0.0066 Acc: 0.9180
val Loss: 0.0096 Acc: 0.8308

Epoch 34/83
----------
train Loss: 0.0029 Acc: 0.9049
val Loss: 0.0073 Acc: 0.8231

Epoch 35/83
----------
train Loss: 0.0031 Acc: 0.9148
val Loss: 0.0079 Acc: 0.8231

Epoch 36/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0040 Acc: 0.9049
val Loss: 0.0091 Acc: 0.8231

Epoch 37/83
----------
train Loss: 0.0053 Acc: 0.9148
val Loss: 0.0087 Acc: 0.8231

Epoch 38/83
----------
train Loss: 0.0052 Acc: 0.9082
val Loss: 0.0081 Acc: 0.8231

Epoch 39/83
----------
train Loss: 0.0027 Acc: 0.9082
val Loss: 0.0072 Acc: 0.8231

Epoch 40/83
----------
train Loss: 0.0078 Acc: 0.9115
val Loss: 0.0099 Acc: 0.8154

Epoch 41/83
----------
train Loss: 0.0044 Acc: 0.9049
val Loss: 0.0101 Acc: 0.8077

Epoch 42/83
----------
train Loss: 0.0051 Acc: 0.9049
val Loss: 0.0088 Acc: 0.8154

Epoch 43/83
----------
train Loss: 0.0037 Acc: 0.9213
val Loss: 0.0072 Acc: 0.8154

Epoch 44/83
----------
train Loss: 0.0034 Acc: 0.9082
val Loss: 0.0096 Acc: 0.8231

Epoch 45/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0039 Acc: 0.9213
val Loss: 0.0084 Acc: 0.8308

Epoch 46/83
----------
train Loss: 0.0035 Acc: 0.9180
val Loss: 0.0085 Acc: 0.8231

Epoch 47/83
----------
train Loss: 0.0073 Acc: 0.9049
val Loss: 0.0090 Acc: 0.8308

Epoch 48/83
----------
train Loss: 0.0030 Acc: 0.9049
val Loss: 0.0095 Acc: 0.8231

Epoch 49/83
----------
train Loss: 0.0054 Acc: 0.8951
val Loss: 0.0076 Acc: 0.8308

Epoch 50/83
----------
train Loss: 0.0081 Acc: 0.9180
val Loss: 0.0078 Acc: 0.8231

Epoch 51/83
----------
train Loss: 0.0058 Acc: 0.9148
val Loss: 0.0079 Acc: 0.8154

Epoch 52/83
----------
train Loss: 0.0039 Acc: 0.9016
val Loss: 0.0087 Acc: 0.8154

Epoch 53/83
----------
train Loss: 0.0030 Acc: 0.9148
val Loss: 0.0078 Acc: 0.8154

Epoch 54/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0034 Acc: 0.9016
val Loss: 0.0067 Acc: 0.8154

Epoch 55/83
----------
train Loss: 0.0070 Acc: 0.9115
val Loss: 0.0090 Acc: 0.8000

Epoch 56/83
----------
train Loss: 0.0037 Acc: 0.9180
val Loss: 0.0086 Acc: 0.8077

Epoch 57/83
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0076 Acc: 0.8000

Epoch 58/83
----------
train Loss: 0.0044 Acc: 0.8918
val Loss: 0.0084 Acc: 0.8077

Epoch 59/83
----------
train Loss: 0.0035 Acc: 0.9082
val Loss: 0.0088 Acc: 0.8077

Epoch 60/83
----------
train Loss: 0.0042 Acc: 0.9082
val Loss: 0.0090 Acc: 0.7923

Epoch 61/83
----------
train Loss: 0.0054 Acc: 0.9180
val Loss: 0.0078 Acc: 0.8154

Epoch 62/83
----------
train Loss: 0.0056 Acc: 0.9016
val Loss: 0.0090 Acc: 0.8231

Epoch 63/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0051 Acc: 0.9016
val Loss: 0.0096 Acc: 0.8077

Epoch 64/83
----------
train Loss: 0.0083 Acc: 0.8984
val Loss: 0.0101 Acc: 0.8000

Epoch 65/83
----------
train Loss: 0.0051 Acc: 0.8918
val Loss: 0.0099 Acc: 0.8000

Epoch 66/83
----------
train Loss: 0.0053 Acc: 0.9016
val Loss: 0.0092 Acc: 0.8000

Epoch 67/83
----------
train Loss: 0.0045 Acc: 0.8918
val Loss: 0.0073 Acc: 0.8000

Epoch 68/83
----------
train Loss: 0.0040 Acc: 0.8885
val Loss: 0.0079 Acc: 0.8000

Epoch 69/83
----------
train Loss: 0.0073 Acc: 0.8951
val Loss: 0.0082 Acc: 0.8077

Epoch 70/83
----------
train Loss: 0.0076 Acc: 0.9049
val Loss: 0.0118 Acc: 0.8000

Epoch 71/83
----------
train Loss: 0.0029 Acc: 0.9148
val Loss: 0.0096 Acc: 0.8077

Epoch 72/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0041 Acc: 0.9049
val Loss: 0.0092 Acc: 0.8154

Epoch 73/83
----------
train Loss: 0.0091 Acc: 0.9082
val Loss: 0.0088 Acc: 0.8154

Epoch 74/83
----------
train Loss: 0.0048 Acc: 0.9115
val Loss: 0.0090 Acc: 0.8077

Epoch 75/83
----------
train Loss: 0.0042 Acc: 0.9082
val Loss: 0.0088 Acc: 0.8077

Epoch 76/83
----------
train Loss: 0.0035 Acc: 0.9082
val Loss: 0.0095 Acc: 0.8000

Epoch 77/83
----------
train Loss: 0.0035 Acc: 0.9115
val Loss: 0.0081 Acc: 0.8000

Epoch 78/83
----------
train Loss: 0.0027 Acc: 0.9148
val Loss: 0.0083 Acc: 0.8000

Epoch 79/83
----------
train Loss: 0.0049 Acc: 0.9279
val Loss: 0.0086 Acc: 0.8077

Epoch 80/83
----------
train Loss: 0.0098 Acc: 0.8852
val Loss: 0.0071 Acc: 0.8000

Epoch 81/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0030 Acc: 0.9246
val Loss: 0.0085 Acc: 0.8077

Epoch 82/83
----------
train Loss: 0.0032 Acc: 0.9279
val Loss: 0.0082 Acc: 0.8154

Epoch 83/83
----------
train Loss: 0.0049 Acc: 0.8918
val Loss: 0.0074 Acc: 0.8154

Training complete in 3m 7s
Best val Acc: 0.846154

---Fine tuning.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0097 Acc: 0.9246
val Loss: 0.0133 Acc: 0.6769

Epoch 1/83
----------
train Loss: 0.0092 Acc: 0.7803
val Loss: 0.0255 Acc: 0.5308

Epoch 2/83
----------
train Loss: 0.0161 Acc: 0.7967
val Loss: 0.0372 Acc: 0.4231

Epoch 3/83
----------
train Loss: 0.0193 Acc: 0.6590
val Loss: 0.0909 Acc: 0.2538

Epoch 4/83
----------
train Loss: 0.0173 Acc: 0.6656
val Loss: 0.0924 Acc: 0.3462

Epoch 5/83
----------
train Loss: 0.0283 Acc: 0.6984
val Loss: 0.0858 Acc: 0.3000

Epoch 6/83
----------
train Loss: 0.0282 Acc: 0.6656
val Loss: 0.1813 Acc: 0.1846

Epoch 7/83
----------
train Loss: 0.0286 Acc: 0.5738
val Loss: 0.2922 Acc: 0.1462

Epoch 8/83
----------
train Loss: 0.0269 Acc: 0.6262
val Loss: 1.4704 Acc: 0.1385

Epoch 9/83
----------
LR is set to 0.001
train Loss: 0.0223 Acc: 0.5803
val Loss: 0.3442 Acc: 0.3769

Epoch 10/83
----------
train Loss: 0.0201 Acc: 0.5770
val Loss: 0.0971 Acc: 0.4538

Epoch 11/83
----------
train Loss: 0.0218 Acc: 0.6295
val Loss: 0.0476 Acc: 0.5308

Epoch 12/83
----------
train Loss: 0.0225 Acc: 0.6689
val Loss: 0.0302 Acc: 0.5615

Epoch 13/83
----------
train Loss: 0.0095 Acc: 0.6984
val Loss: 0.0226 Acc: 0.6308

Epoch 14/83
----------
train Loss: 0.0129 Acc: 0.7377
val Loss: 0.0198 Acc: 0.6692

Epoch 15/83
----------
train Loss: 0.0096 Acc: 0.7639
val Loss: 0.0211 Acc: 0.6308

Epoch 16/83
----------
train Loss: 0.0088 Acc: 0.7738
val Loss: 0.0172 Acc: 0.6462

Epoch 17/83
----------
train Loss: 0.0081 Acc: 0.7574
val Loss: 0.0179 Acc: 0.6769

Epoch 18/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0088 Acc: 0.8033
val Loss: 0.0168 Acc: 0.6846

Epoch 19/83
----------
train Loss: 0.0068 Acc: 0.8197
val Loss: 0.0168 Acc: 0.6769

Epoch 20/83
----------
train Loss: 0.0084 Acc: 0.8066
val Loss: 0.0153 Acc: 0.6308

Epoch 21/83
----------
train Loss: 0.0090 Acc: 0.8033
val Loss: 0.0178 Acc: 0.6462

Epoch 22/83
----------
train Loss: 0.0052 Acc: 0.8361
val Loss: 0.0162 Acc: 0.6615

Epoch 23/83
----------
train Loss: 0.0082 Acc: 0.8197
val Loss: 0.0144 Acc: 0.6462

Epoch 24/83
----------
train Loss: 0.0064 Acc: 0.8525
val Loss: 0.0138 Acc: 0.6462

Epoch 25/83
----------
train Loss: 0.0120 Acc: 0.8361
val Loss: 0.0142 Acc: 0.6308

Epoch 26/83
----------
train Loss: 0.0050 Acc: 0.8393
val Loss: 0.0148 Acc: 0.6769

Epoch 27/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0052 Acc: 0.8557
val Loss: 0.0141 Acc: 0.6538

Epoch 28/83
----------
train Loss: 0.0086 Acc: 0.8656
val Loss: 0.0148 Acc: 0.6692

Epoch 29/83
----------
train Loss: 0.0083 Acc: 0.8459
val Loss: 0.0146 Acc: 0.6615

Epoch 30/83
----------
train Loss: 0.0045 Acc: 0.8492
val Loss: 0.0160 Acc: 0.6462

Epoch 31/83
----------
train Loss: 0.0070 Acc: 0.8459
val Loss: 0.0167 Acc: 0.6462

Epoch 32/83
----------
train Loss: 0.0048 Acc: 0.8656
val Loss: 0.0150 Acc: 0.6615

Epoch 33/83
----------
train Loss: 0.0080 Acc: 0.8656
val Loss: 0.0148 Acc: 0.6692

Epoch 34/83
----------
train Loss: 0.0091 Acc: 0.8557
val Loss: 0.0144 Acc: 0.6615

Epoch 35/83
----------
train Loss: 0.0072 Acc: 0.8623
val Loss: 0.0148 Acc: 0.6692

Epoch 36/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0064 Acc: 0.8721
val Loss: 0.0147 Acc: 0.6846

Epoch 37/83
----------
train Loss: 0.0151 Acc: 0.8590
val Loss: 0.0147 Acc: 0.6615

Epoch 38/83
----------
train Loss: 0.0056 Acc: 0.8557
val Loss: 0.0145 Acc: 0.6615

Epoch 39/83
----------
train Loss: 0.0069 Acc: 0.8689
val Loss: 0.0176 Acc: 0.6538

Epoch 40/83
----------
train Loss: 0.0053 Acc: 0.8787
val Loss: 0.0140 Acc: 0.6462

Epoch 41/83
----------
train Loss: 0.0058 Acc: 0.8492
val Loss: 0.0146 Acc: 0.6462

Epoch 42/83
----------
train Loss: 0.0069 Acc: 0.8721
val Loss: 0.0142 Acc: 0.6538

Epoch 43/83
----------
train Loss: 0.0073 Acc: 0.8459
val Loss: 0.0153 Acc: 0.6769

Epoch 44/83
----------
train Loss: 0.0074 Acc: 0.8426
val Loss: 0.0115 Acc: 0.6615

Epoch 45/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0052 Acc: 0.8459
val Loss: 0.0153 Acc: 0.6769

Epoch 46/83
----------
train Loss: 0.0080 Acc: 0.8623
val Loss: 0.0132 Acc: 0.6308

Epoch 47/83
----------
train Loss: 0.0080 Acc: 0.8393
val Loss: 0.0144 Acc: 0.6538

Epoch 48/83
----------
train Loss: 0.0061 Acc: 0.8590
val Loss: 0.0126 Acc: 0.6538

Epoch 49/83
----------
train Loss: 0.0049 Acc: 0.8590
val Loss: 0.0139 Acc: 0.6692

Epoch 50/83
----------
train Loss: 0.0079 Acc: 0.8492
val Loss: 0.0156 Acc: 0.6615

Epoch 51/83
----------
train Loss: 0.0067 Acc: 0.8525
val Loss: 0.0162 Acc: 0.6308

Epoch 52/83
----------
train Loss: 0.0101 Acc: 0.8492
val Loss: 0.0142 Acc: 0.6385

Epoch 53/83
----------
train Loss: 0.0129 Acc: 0.8557
val Loss: 0.0167 Acc: 0.6462

Epoch 54/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0069 Acc: 0.8820
val Loss: 0.0156 Acc: 0.6308

Epoch 55/83
----------
train Loss: 0.0058 Acc: 0.8787
val Loss: 0.0151 Acc: 0.6385

Epoch 56/83
----------
train Loss: 0.0088 Acc: 0.8361
val Loss: 0.0155 Acc: 0.6538

Epoch 57/83
----------
train Loss: 0.0049 Acc: 0.8852
val Loss: 0.0163 Acc: 0.6308

Epoch 58/83
----------
train Loss: 0.0060 Acc: 0.8689
val Loss: 0.0131 Acc: 0.6462

Epoch 59/83
----------
train Loss: 0.0097 Acc: 0.8689
val Loss: 0.0155 Acc: 0.6462

Epoch 60/83
----------
train Loss: 0.0051 Acc: 0.8689
val Loss: 0.0145 Acc: 0.6385

Epoch 61/83
----------
train Loss: 0.0058 Acc: 0.8656
val Loss: 0.0151 Acc: 0.6769

Epoch 62/83
----------
train Loss: 0.0082 Acc: 0.8557
val Loss: 0.0163 Acc: 0.6692

Epoch 63/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0094 Acc: 0.8656
val Loss: 0.0142 Acc: 0.6538

Epoch 64/83
----------
train Loss: 0.0063 Acc: 0.8754
val Loss: 0.0116 Acc: 0.6692

Epoch 65/83
----------
train Loss: 0.0068 Acc: 0.8852
val Loss: 0.0134 Acc: 0.6615

Epoch 66/83
----------
train Loss: 0.0063 Acc: 0.8393
val Loss: 0.0149 Acc: 0.6769

Epoch 67/83
----------
train Loss: 0.0093 Acc: 0.8459
val Loss: 0.0177 Acc: 0.6769

Epoch 68/83
----------
train Loss: 0.0058 Acc: 0.8623
val Loss: 0.0129 Acc: 0.6846

Epoch 69/83
----------
train Loss: 0.0055 Acc: 0.8328
val Loss: 0.0143 Acc: 0.6846

Epoch 70/83
----------
train Loss: 0.0086 Acc: 0.8426
val Loss: 0.0152 Acc: 0.6769

Epoch 71/83
----------
train Loss: 0.0078 Acc: 0.8557
val Loss: 0.0161 Acc: 0.6846

Epoch 72/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0070 Acc: 0.8689
val Loss: 0.0143 Acc: 0.6923

Epoch 73/83
----------
train Loss: 0.0049 Acc: 0.8852
val Loss: 0.0143 Acc: 0.6846

Epoch 74/83
----------
train Loss: 0.0084 Acc: 0.8262
val Loss: 0.0146 Acc: 0.6692

Epoch 75/83
----------
train Loss: 0.0058 Acc: 0.8557
val Loss: 0.0141 Acc: 0.7000

Epoch 76/83
----------
train Loss: 0.0091 Acc: 0.8557
val Loss: 0.0148 Acc: 0.6769

Epoch 77/83
----------
train Loss: 0.0074 Acc: 0.8557
val Loss: 0.0148 Acc: 0.6615

Epoch 78/83
----------
train Loss: 0.0078 Acc: 0.8525
val Loss: 0.0150 Acc: 0.6769

Epoch 79/83
----------
train Loss: 0.0092 Acc: 0.8852
val Loss: 0.0146 Acc: 0.6538

Epoch 80/83
----------
train Loss: 0.0060 Acc: 0.8689
val Loss: 0.0169 Acc: 0.6769

Epoch 81/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0086 Acc: 0.8459
val Loss: 0.0151 Acc: 0.6769

Epoch 82/83
----------
train Loss: 0.0064 Acc: 0.8623
val Loss: 0.0155 Acc: 0.6846

Epoch 83/83
----------
train Loss: 0.0096 Acc: 0.8230
val Loss: 0.0169 Acc: 0.6923

Training complete in 3m 28s
Best val Acc: 0.700000

---Testing---
Test accuracy: 0.804598
--------------------
Accuracy of Dasyatiformes : 51 %
Accuracy of Myliobatiformes : 76 %
Accuracy of Rajiformes : 68 %
Accuracy of Rhinobatiformes : 82 %
Accuracy of Torpediniformes : 91 %
mean: 0.7412149663811468, std: 0.13603531704275934

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 78, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/77
----------
LR is set to 0.01
train Loss: 0.0183 Acc: 0.2959
val Loss: 0.0340 Acc: 0.4884

Epoch 1/77
----------
train Loss: 0.0134 Acc: 0.4235
val Loss: 0.0334 Acc: 0.4419

Epoch 2/77
----------
train Loss: 0.0101 Acc: 0.6327
val Loss: 0.0176 Acc: 0.7674

Epoch 3/77
----------
train Loss: 0.0076 Acc: 0.7423
val Loss: 0.0187 Acc: 0.6279

Epoch 4/77
----------
train Loss: 0.0067 Acc: 0.7602
val Loss: 0.0133 Acc: 0.8140

Epoch 5/77
----------
LR is set to 0.001
train Loss: 0.0048 Acc: 0.8291
val Loss: 0.0131 Acc: 0.8140

Epoch 6/77
----------
train Loss: 0.0051 Acc: 0.8240
val Loss: 0.0132 Acc: 0.7907

Epoch 7/77
----------
train Loss: 0.0049 Acc: 0.8291
val Loss: 0.0134 Acc: 0.8140

Epoch 8/77
----------
train Loss: 0.0048 Acc: 0.8469
val Loss: 0.0137 Acc: 0.8372

Epoch 9/77
----------
train Loss: 0.0047 Acc: 0.8342
val Loss: 0.0139 Acc: 0.8140

Epoch 10/77
----------
LR is set to 0.00010000000000000002
train Loss: 0.0047 Acc: 0.8495
val Loss: 0.0139 Acc: 0.8140

Epoch 11/77
----------
train Loss: 0.0047 Acc: 0.8520
val Loss: 0.0139 Acc: 0.8140

Epoch 12/77
----------
train Loss: 0.0048 Acc: 0.8546
val Loss: 0.0139 Acc: 0.8140

Epoch 13/77
----------
train Loss: 0.0048 Acc: 0.8469
val Loss: 0.0139 Acc: 0.8140

Epoch 14/77
----------
train Loss: 0.0047 Acc: 0.8367
val Loss: 0.0138 Acc: 0.8140

Epoch 15/77
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0045 Acc: 0.8699
val Loss: 0.0138 Acc: 0.8140

Epoch 16/77
----------
train Loss: 0.0046 Acc: 0.8546
val Loss: 0.0138 Acc: 0.8140

Epoch 17/77
----------
train Loss: 0.0047 Acc: 0.8367
val Loss: 0.0138 Acc: 0.8140

Epoch 18/77
----------
train Loss: 0.0045 Acc: 0.8495
val Loss: 0.0138 Acc: 0.8140

Epoch 19/77
----------
train Loss: 0.0047 Acc: 0.8393
val Loss: 0.0138 Acc: 0.8140

Epoch 20/77
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0045 Acc: 0.8418
val Loss: 0.0138 Acc: 0.8140

Epoch 21/77
----------
train Loss: 0.0045 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 22/77
----------
train Loss: 0.0046 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 23/77
----------
train Loss: 0.0048 Acc: 0.8571
val Loss: 0.0139 Acc: 0.8140

Epoch 24/77
----------
train Loss: 0.0045 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 25/77
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0046 Acc: 0.8622
val Loss: 0.0138 Acc: 0.8140

Epoch 26/77
----------
train Loss: 0.0048 Acc: 0.8418
val Loss: 0.0138 Acc: 0.8140

Epoch 27/77
----------
train Loss: 0.0047 Acc: 0.8418
val Loss: 0.0138 Acc: 0.8140

Epoch 28/77
----------
train Loss: 0.0045 Acc: 0.8367
val Loss: 0.0138 Acc: 0.8140

Epoch 29/77
----------
train Loss: 0.0046 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 30/77
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0046 Acc: 0.8316
val Loss: 0.0138 Acc: 0.8140

Epoch 31/77
----------
train Loss: 0.0050 Acc: 0.8291
val Loss: 0.0138 Acc: 0.8140

Epoch 32/77
----------
train Loss: 0.0046 Acc: 0.8495
val Loss: 0.0138 Acc: 0.8140

Epoch 33/77
----------
train Loss: 0.0046 Acc: 0.8393
val Loss: 0.0138 Acc: 0.8140

Epoch 34/77
----------
train Loss: 0.0047 Acc: 0.8520
val Loss: 0.0138 Acc: 0.8140

Epoch 35/77
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0046 Acc: 0.8571
val Loss: 0.0137 Acc: 0.8140

Epoch 36/77
----------
train Loss: 0.0046 Acc: 0.8393
val Loss: 0.0137 Acc: 0.8140

Epoch 37/77
----------
train Loss: 0.0046 Acc: 0.8444
val Loss: 0.0137 Acc: 0.8140

Epoch 38/77
----------
train Loss: 0.0047 Acc: 0.8367
val Loss: 0.0137 Acc: 0.8140

Epoch 39/77
----------
train Loss: 0.0044 Acc: 0.8622
val Loss: 0.0138 Acc: 0.8140

Epoch 40/77
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0046 Acc: 0.8648
val Loss: 0.0138 Acc: 0.8140

Epoch 41/77
----------
train Loss: 0.0049 Acc: 0.8265
val Loss: 0.0138 Acc: 0.8140

Epoch 42/77
----------
train Loss: 0.0046 Acc: 0.8520
val Loss: 0.0138 Acc: 0.8140

Epoch 43/77
----------
train Loss: 0.0045 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 44/77
----------
train Loss: 0.0048 Acc: 0.8367
val Loss: 0.0138 Acc: 0.8140

Epoch 45/77
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0048 Acc: 0.8214
val Loss: 0.0138 Acc: 0.8140

Epoch 46/77
----------
train Loss: 0.0046 Acc: 0.8546
val Loss: 0.0138 Acc: 0.8140

Epoch 47/77
----------
train Loss: 0.0045 Acc: 0.8546
val Loss: 0.0138 Acc: 0.8140

Epoch 48/77
----------
train Loss: 0.0049 Acc: 0.8393
val Loss: 0.0138 Acc: 0.8140

Epoch 49/77
----------
train Loss: 0.0046 Acc: 0.8495
val Loss: 0.0138 Acc: 0.8140

Epoch 50/77
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0045 Acc: 0.8342
val Loss: 0.0138 Acc: 0.8140

Epoch 51/77
----------
train Loss: 0.0047 Acc: 0.8444
val Loss: 0.0137 Acc: 0.8140

Epoch 52/77
----------
train Loss: 0.0048 Acc: 0.8265
val Loss: 0.0138 Acc: 0.8140

Epoch 53/77
----------
train Loss: 0.0046 Acc: 0.8342
val Loss: 0.0138 Acc: 0.8140

Epoch 54/77
----------
train Loss: 0.0048 Acc: 0.8418
val Loss: 0.0138 Acc: 0.8140

Epoch 55/77
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0046 Acc: 0.8367
val Loss: 0.0138 Acc: 0.8140

Epoch 56/77
----------
train Loss: 0.0047 Acc: 0.8495
val Loss: 0.0138 Acc: 0.8140

Epoch 57/77
----------
train Loss: 0.0046 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 58/77
----------
train Loss: 0.0046 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 59/77
----------
train Loss: 0.0043 Acc: 0.8673
val Loss: 0.0138 Acc: 0.8140

Epoch 60/77
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0046 Acc: 0.8291
val Loss: 0.0138 Acc: 0.8140

Epoch 61/77
----------
train Loss: 0.0049 Acc: 0.8316
val Loss: 0.0138 Acc: 0.8140

Epoch 62/77
----------
train Loss: 0.0047 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 63/77
----------
train Loss: 0.0045 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 64/77
----------
train Loss: 0.0045 Acc: 0.8699
val Loss: 0.0138 Acc: 0.8140

Epoch 65/77
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0043 Acc: 0.8622
val Loss: 0.0138 Acc: 0.8140

Epoch 66/77
----------
train Loss: 0.0050 Acc: 0.8393
val Loss: 0.0137 Acc: 0.8140

Epoch 67/77
----------
train Loss: 0.0048 Acc: 0.8265
val Loss: 0.0138 Acc: 0.8140

Epoch 68/77
----------
train Loss: 0.0045 Acc: 0.8597
val Loss: 0.0138 Acc: 0.8140

Epoch 69/77
----------
train Loss: 0.0045 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 70/77
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0048 Acc: 0.8444
val Loss: 0.0138 Acc: 0.8140

Epoch 71/77
----------
train Loss: 0.0047 Acc: 0.8418
val Loss: 0.0138 Acc: 0.8140

Epoch 72/77
----------
train Loss: 0.0045 Acc: 0.8469
val Loss: 0.0138 Acc: 0.8140

Epoch 73/77
----------
train Loss: 0.0045 Acc: 0.8571
val Loss: 0.0138 Acc: 0.8140

Epoch 74/77
----------
train Loss: 0.0043 Acc: 0.8571
val Loss: 0.0138 Acc: 0.8140

Epoch 75/77
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0046 Acc: 0.8546
val Loss: 0.0139 Acc: 0.8140

Epoch 76/77
----------
train Loss: 0.0048 Acc: 0.8342
val Loss: 0.0138 Acc: 0.8140

Epoch 77/77
----------
train Loss: 0.0049 Acc: 0.8265
val Loss: 0.0139 Acc: 0.8140

Training complete in 2m 43s
Best val Acc: 0.837209

---Fine tuning.---
Epoch 0/77
----------
LR is set to 0.01
train Loss: 0.0045 Acc: 0.8648
val Loss: 0.0124 Acc: 0.8372

Epoch 1/77
----------
train Loss: 0.0029 Acc: 0.9362
val Loss: 0.0117 Acc: 0.8605

Epoch 2/77
----------
train Loss: 0.0018 Acc: 0.9592
val Loss: 0.0135 Acc: 0.8140

Epoch 3/77
----------
train Loss: 0.0009 Acc: 0.9898
val Loss: 0.0124 Acc: 0.8372

Epoch 4/77
----------
train Loss: 0.0007 Acc: 0.9847
val Loss: 0.0129 Acc: 0.8140

Epoch 5/77
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0127 Acc: 0.8372

Epoch 6/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0126 Acc: 0.8372

Epoch 7/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8372

Epoch 8/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 9/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 10/77
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 11/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 12/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 13/77
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 14/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 15/77
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 16/77
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 17/77
----------
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 18/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8372

Epoch 19/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8372

Epoch 20/77
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 21/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 22/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 23/77
----------
train Loss: 0.0003 Acc: 0.9923
val Loss: 0.0124 Acc: 0.8372

Epoch 24/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 25/77
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 26/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 27/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 28/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 29/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8372

Epoch 30/77
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 31/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0123 Acc: 0.8372

Epoch 32/77
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 33/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 34/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0123 Acc: 0.8372

Epoch 35/77
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 36/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 37/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 38/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 39/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 40/77
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 41/77
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 42/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 43/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 44/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 45/77
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0123 Acc: 0.8372

Epoch 46/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 47/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 48/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 49/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 50/77
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 51/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 52/77
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8372

Epoch 53/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 54/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 55/77
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 56/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 57/77
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 58/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 59/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 60/77
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 61/77
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8372

Epoch 62/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8372

Epoch 63/77
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 64/77
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 65/77
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 66/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0125 Acc: 0.8372

Epoch 67/77
----------
train Loss: 0.0002 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 68/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 69/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8372

Epoch 70/77
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 71/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 72/77
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8372

Epoch 73/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0123 Acc: 0.8372

Epoch 74/77
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0124 Acc: 0.8372

Epoch 75/77
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 76/77
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0124 Acc: 0.8372

Epoch 77/77
----------
train Loss: 0.0004 Acc: 0.9923
val Loss: 0.0124 Acc: 0.8372

Training complete in 2m 58s
Best val Acc: 0.860465

---Testing---
Test accuracy: 0.967816
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 96 %
Accuracy of Rajiformes : 93 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.9595707609713291, std: 0.023019977559542208
--------------------

run info[val: 0.15, epoch: 60, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0160 Acc: 0.3541
val Loss: 0.0199 Acc: 0.6154

Epoch 1/59
----------
train Loss: 0.0130 Acc: 0.5243
val Loss: 0.0164 Acc: 0.5846

Epoch 2/59
----------
train Loss: 0.0089 Acc: 0.7378
val Loss: 0.0104 Acc: 0.7846

Epoch 3/59
----------
train Loss: 0.0068 Acc: 0.7892
val Loss: 0.0106 Acc: 0.7846

Epoch 4/59
----------
train Loss: 0.0056 Acc: 0.8297
val Loss: 0.0077 Acc: 0.8769

Epoch 5/59
----------
train Loss: 0.0051 Acc: 0.8297
val Loss: 0.0071 Acc: 0.8615

Epoch 6/59
----------
train Loss: 0.0044 Acc: 0.8541
val Loss: 0.0074 Acc: 0.8615

Epoch 7/59
----------
train Loss: 0.0038 Acc: 0.8838
val Loss: 0.0063 Acc: 0.8462

Epoch 8/59
----------
train Loss: 0.0033 Acc: 0.9162
val Loss: 0.0067 Acc: 0.8923

Epoch 9/59
----------
LR is set to 0.001
train Loss: 0.0030 Acc: 0.9270
val Loss: 0.0066 Acc: 0.8923

Epoch 10/59
----------
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8923

Epoch 11/59
----------
train Loss: 0.0028 Acc: 0.9243
val Loss: 0.0063 Acc: 0.8615

Epoch 12/59
----------
train Loss: 0.0027 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8615

Epoch 13/59
----------
train Loss: 0.0027 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8462

Epoch 14/59
----------
train Loss: 0.0027 Acc: 0.9243
val Loss: 0.0063 Acc: 0.8615

Epoch 15/59
----------
train Loss: 0.0028 Acc: 0.9297
val Loss: 0.0064 Acc: 0.8615

Epoch 16/59
----------
train Loss: 0.0028 Acc: 0.9162
val Loss: 0.0063 Acc: 0.8615

Epoch 17/59
----------
train Loss: 0.0027 Acc: 0.9297
val Loss: 0.0063 Acc: 0.8615

Epoch 18/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0026 Acc: 0.9405
val Loss: 0.0063 Acc: 0.8769

Epoch 19/59
----------
train Loss: 0.0026 Acc: 0.9486
val Loss: 0.0063 Acc: 0.8769

Epoch 20/59
----------
train Loss: 0.0027 Acc: 0.9432
val Loss: 0.0063 Acc: 0.8769

Epoch 21/59
----------
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0063 Acc: 0.8769

Epoch 22/59
----------
train Loss: 0.0028 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8769

Epoch 23/59
----------
train Loss: 0.0027 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8615

Epoch 24/59
----------
train Loss: 0.0026 Acc: 0.9514
val Loss: 0.0063 Acc: 0.8769

Epoch 25/59
----------
train Loss: 0.0025 Acc: 0.9514
val Loss: 0.0062 Acc: 0.8769

Epoch 26/59
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8769

Epoch 27/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0026 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8769

Epoch 28/59
----------
train Loss: 0.0025 Acc: 0.9514
val Loss: 0.0062 Acc: 0.8769

Epoch 29/59
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0062 Acc: 0.8769

Epoch 30/59
----------
train Loss: 0.0027 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8769

Epoch 31/59
----------
train Loss: 0.0026 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8615

Epoch 32/59
----------
train Loss: 0.0026 Acc: 0.9324
val Loss: 0.0063 Acc: 0.8615

Epoch 33/59
----------
train Loss: 0.0025 Acc: 0.9351
val Loss: 0.0063 Acc: 0.8615

Epoch 34/59
----------
train Loss: 0.0026 Acc: 0.9351
val Loss: 0.0063 Acc: 0.8615

Epoch 35/59
----------
train Loss: 0.0027 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8615

Epoch 36/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0027 Acc: 0.9351
val Loss: 0.0063 Acc: 0.8615

Epoch 37/59
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8615

Epoch 38/59
----------
train Loss: 0.0026 Acc: 0.9486
val Loss: 0.0062 Acc: 0.8615

Epoch 39/59
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0062 Acc: 0.8769

Epoch 40/59
----------
train Loss: 0.0025 Acc: 0.9514
val Loss: 0.0062 Acc: 0.8615

Epoch 41/59
----------
train Loss: 0.0026 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8769

Epoch 42/59
----------
train Loss: 0.0026 Acc: 0.9378
val Loss: 0.0062 Acc: 0.8769

Epoch 43/59
----------
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0062 Acc: 0.8615

Epoch 44/59
----------
train Loss: 0.0027 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8615

Epoch 45/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0026 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8615

Epoch 46/59
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0062 Acc: 0.8615

Epoch 47/59
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0063 Acc: 0.8615

Epoch 48/59
----------
train Loss: 0.0026 Acc: 0.9378
val Loss: 0.0063 Acc: 0.8615

Epoch 49/59
----------
train Loss: 0.0025 Acc: 0.9405
val Loss: 0.0063 Acc: 0.8615

Epoch 50/59
----------
train Loss: 0.0027 Acc: 0.9351
val Loss: 0.0062 Acc: 0.8615

Epoch 51/59
----------
train Loss: 0.0025 Acc: 0.9568
val Loss: 0.0062 Acc: 0.8615

Epoch 52/59
----------
train Loss: 0.0027 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8615

Epoch 53/59
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0062 Acc: 0.8615

Epoch 54/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0025 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8769

Epoch 55/59
----------
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0062 Acc: 0.8769

Epoch 56/59
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0062 Acc: 0.8769

Epoch 57/59
----------
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0062 Acc: 0.8615

Epoch 58/59
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8615

Epoch 59/59
----------
train Loss: 0.0027 Acc: 0.9351
val Loss: 0.0063 Acc: 0.8615

Training complete in 2m 8s
Best val Acc: 0.892308

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0059 Acc: 0.9077

Epoch 1/59
----------
train Loss: 0.0014 Acc: 0.9811
val Loss: 0.0060 Acc: 0.8615

Epoch 2/59
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0057 Acc: 0.9077

Epoch 3/59
----------
train Loss: 0.0004 Acc: 0.9973
val Loss: 0.0053 Acc: 0.9077

Epoch 4/59
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0048 Acc: 0.9077

Epoch 5/59
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8923

Epoch 6/59
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.9077

Epoch 7/59
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9077

Epoch 8/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9077

Epoch 9/59
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9077

Epoch 10/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9077

Epoch 11/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 12/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 13/59
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 14/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 15/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 16/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 17/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 18/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 19/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 20/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 21/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 22/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 23/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 24/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 25/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 26/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 27/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 28/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 29/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 30/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 31/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 32/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 33/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 34/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 35/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 36/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 37/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 38/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 39/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 40/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 41/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 42/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 43/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 44/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 45/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 46/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 47/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 48/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 49/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 50/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 51/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 52/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 53/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 54/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 55/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 56/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 57/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 58/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 59/59
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Training complete in 2m 20s
Best val Acc: 0.907692

---Testing---
Test accuracy: 0.974713
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 99 %
mean: 0.9687964924192727, std: 0.015569677313009279
--------------------

run info[val: 0.2, epoch: 53, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0186 Acc: 0.3276
val Loss: 0.0178 Acc: 0.3563

Epoch 1/52
----------
train Loss: 0.0138 Acc: 0.5259
val Loss: 0.0133 Acc: 0.5747

Epoch 2/52
----------
train Loss: 0.0095 Acc: 0.7011
val Loss: 0.0089 Acc: 0.7586

Epoch 3/52
----------
train Loss: 0.0078 Acc: 0.7759
val Loss: 0.0074 Acc: 0.7701

Epoch 4/52
----------
train Loss: 0.0073 Acc: 0.7615
val Loss: 0.0071 Acc: 0.8391

Epoch 5/52
----------
train Loss: 0.0056 Acc: 0.8420
val Loss: 0.0068 Acc: 0.8161

Epoch 6/52
----------
train Loss: 0.0050 Acc: 0.8420
val Loss: 0.0058 Acc: 0.8621

Epoch 7/52
----------
train Loss: 0.0044 Acc: 0.8879
val Loss: 0.0064 Acc: 0.8391

Epoch 8/52
----------
train Loss: 0.0036 Acc: 0.8937
val Loss: 0.0059 Acc: 0.8276

Epoch 9/52
----------
train Loss: 0.0032 Acc: 0.8937
val Loss: 0.0059 Acc: 0.8736

Epoch 10/52
----------
train Loss: 0.0028 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8851

Epoch 11/52
----------
train Loss: 0.0027 Acc: 0.9253
val Loss: 0.0051 Acc: 0.8851

Epoch 12/52
----------
train Loss: 0.0022 Acc: 0.9655
val Loss: 0.0055 Acc: 0.8966

Epoch 13/52
----------
LR is set to 0.001
train Loss: 0.0022 Acc: 0.9684
val Loss: 0.0055 Acc: 0.8966

Epoch 14/52
----------
train Loss: 0.0022 Acc: 0.9569
val Loss: 0.0052 Acc: 0.9080

Epoch 15/52
----------
train Loss: 0.0023 Acc: 0.9569
val Loss: 0.0051 Acc: 0.8851

Epoch 16/52
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0051 Acc: 0.8851

Epoch 17/52
----------
train Loss: 0.0022 Acc: 0.9483
val Loss: 0.0051 Acc: 0.8851

Epoch 18/52
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0051 Acc: 0.8851

Epoch 19/52
----------
train Loss: 0.0020 Acc: 0.9626
val Loss: 0.0050 Acc: 0.8851

Epoch 20/52
----------
train Loss: 0.0020 Acc: 0.9655
val Loss: 0.0051 Acc: 0.8966

Epoch 21/52
----------
train Loss: 0.0021 Acc: 0.9713
val Loss: 0.0051 Acc: 0.8851

Epoch 22/52
----------
train Loss: 0.0022 Acc: 0.9770
val Loss: 0.0050 Acc: 0.8851

Epoch 23/52
----------
train Loss: 0.0021 Acc: 0.9713
val Loss: 0.0050 Acc: 0.8851

Epoch 24/52
----------
train Loss: 0.0020 Acc: 0.9684
val Loss: 0.0050 Acc: 0.8851

Epoch 25/52
----------
train Loss: 0.0019 Acc: 0.9684
val Loss: 0.0050 Acc: 0.8851

Epoch 26/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0020 Acc: 0.9713
val Loss: 0.0050 Acc: 0.8851

Epoch 27/52
----------
train Loss: 0.0021 Acc: 0.9655
val Loss: 0.0049 Acc: 0.8851

Epoch 28/52
----------
train Loss: 0.0020 Acc: 0.9713
val Loss: 0.0050 Acc: 0.8851

Epoch 29/52
----------
train Loss: 0.0020 Acc: 0.9655
val Loss: 0.0050 Acc: 0.8851

Epoch 30/52
----------
train Loss: 0.0018 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 31/52
----------
train Loss: 0.0018 Acc: 0.9828
val Loss: 0.0050 Acc: 0.8851

Epoch 32/52
----------
train Loss: 0.0020 Acc: 0.9684
val Loss: 0.0050 Acc: 0.8851

Epoch 33/52
----------
train Loss: 0.0018 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 34/52
----------
train Loss: 0.0018 Acc: 0.9799
val Loss: 0.0050 Acc: 0.8851

Epoch 35/52
----------
train Loss: 0.0018 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 36/52
----------
train Loss: 0.0018 Acc: 0.9828
val Loss: 0.0050 Acc: 0.8851

Epoch 37/52
----------
train Loss: 0.0018 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 38/52
----------
train Loss: 0.0019 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 39/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0019 Acc: 0.9655
val Loss: 0.0051 Acc: 0.8851

Epoch 40/52
----------
train Loss: 0.0019 Acc: 0.9684
val Loss: 0.0051 Acc: 0.8851

Epoch 41/52
----------
train Loss: 0.0019 Acc: 0.9828
val Loss: 0.0051 Acc: 0.8851

Epoch 42/52
----------
train Loss: 0.0020 Acc: 0.9684
val Loss: 0.0050 Acc: 0.8851

Epoch 43/52
----------
train Loss: 0.0018 Acc: 0.9799
val Loss: 0.0050 Acc: 0.8851

Epoch 44/52
----------
train Loss: 0.0019 Acc: 0.9655
val Loss: 0.0050 Acc: 0.8851

Epoch 45/52
----------
train Loss: 0.0019 Acc: 0.9655
val Loss: 0.0050 Acc: 0.8851

Epoch 46/52
----------
train Loss: 0.0021 Acc: 0.9626
val Loss: 0.0051 Acc: 0.8851

Epoch 47/52
----------
train Loss: 0.0019 Acc: 0.9713
val Loss: 0.0050 Acc: 0.8851

Epoch 48/52
----------
train Loss: 0.0020 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 49/52
----------
train Loss: 0.0018 Acc: 0.9741
val Loss: 0.0050 Acc: 0.8851

Epoch 50/52
----------
train Loss: 0.0018 Acc: 0.9770
val Loss: 0.0050 Acc: 0.8851

Epoch 51/52
----------
train Loss: 0.0018 Acc: 0.9713
val Loss: 0.0050 Acc: 0.8851

Epoch 52/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9655
val Loss: 0.0051 Acc: 0.8851

Training complete in 1m 56s
Best val Acc: 0.908046

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0021 Acc: 0.9626
val Loss: 0.0050 Acc: 0.8736

Epoch 1/52
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0051 Acc: 0.8851

Epoch 2/52
----------
train Loss: 0.0006 Acc: 0.9943
val Loss: 0.0057 Acc: 0.8506

Epoch 3/52
----------
train Loss: 0.0005 Acc: 0.9971
val Loss: 0.0053 Acc: 0.8391

Epoch 4/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8621

Epoch 5/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8851

Epoch 6/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8851

Epoch 7/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 8/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 9/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 10/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 11/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 12/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 13/52
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 14/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 15/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 16/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 17/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 18/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 19/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 20/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 21/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 22/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 23/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 24/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 25/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 26/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 27/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 28/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 29/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 30/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 31/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 32/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 33/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 34/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 35/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 36/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 37/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 38/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 39/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 40/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 41/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 42/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 43/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 44/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 45/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 46/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 47/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 48/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8966

Epoch 49/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8966

Epoch 50/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8966

Epoch 51/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 52/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Training complete in 2m 7s
Best val Acc: 0.896552

---Testing---
Test accuracy: 0.979310
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.9754293709118003, std: 0.007334480235971777
--------------------

run info[val: 0.25, epoch: 58, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0195 Acc: 0.3211
val Loss: 0.0272 Acc: 0.4167

Epoch 1/57
----------
train Loss: 0.0173 Acc: 0.5627
val Loss: 0.0227 Acc: 0.6481

Epoch 2/57
----------
train Loss: 0.0114 Acc: 0.6942
val Loss: 0.0161 Acc: 0.7130

Epoch 3/57
----------
LR is set to 0.001
train Loss: 0.0090 Acc: 0.7951
val Loss: 0.0178 Acc: 0.7500

Epoch 4/57
----------
train Loss: 0.0080 Acc: 0.8073
val Loss: 0.0169 Acc: 0.7778

Epoch 5/57
----------
train Loss: 0.0075 Acc: 0.8226
val Loss: 0.0116 Acc: 0.7500

Epoch 6/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0077 Acc: 0.8073
val Loss: 0.0127 Acc: 0.7407

Epoch 7/57
----------
train Loss: 0.0074 Acc: 0.7920
val Loss: 0.0153 Acc: 0.7593

Epoch 8/57
----------
train Loss: 0.0074 Acc: 0.8012
val Loss: 0.0130 Acc: 0.7685

Epoch 9/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.8104
val Loss: 0.0147 Acc: 0.7685

Epoch 10/57
----------
train Loss: 0.0073 Acc: 0.7982
val Loss: 0.0108 Acc: 0.7685

Epoch 11/57
----------
train Loss: 0.0074 Acc: 0.8104
val Loss: 0.0122 Acc: 0.7593

Epoch 12/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0077 Acc: 0.8165
val Loss: 0.0148 Acc: 0.7593

Epoch 13/57
----------
train Loss: 0.0070 Acc: 0.7859
val Loss: 0.0122 Acc: 0.7685

Epoch 14/57
----------
train Loss: 0.0070 Acc: 0.7951
val Loss: 0.0138 Acc: 0.7685

Epoch 15/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0070 Acc: 0.8073
val Loss: 0.0096 Acc: 0.7593

Epoch 16/57
----------
train Loss: 0.0074 Acc: 0.8043
val Loss: 0.0157 Acc: 0.7593

Epoch 17/57
----------
train Loss: 0.0072 Acc: 0.7982
val Loss: 0.0123 Acc: 0.7593

Epoch 18/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0074 Acc: 0.8196
val Loss: 0.0113 Acc: 0.7593

Epoch 19/57
----------
train Loss: 0.0074 Acc: 0.8012
val Loss: 0.0121 Acc: 0.7593

Epoch 20/57
----------
train Loss: 0.0071 Acc: 0.8165
val Loss: 0.0119 Acc: 0.7685

Epoch 21/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0072 Acc: 0.8012
val Loss: 0.0155 Acc: 0.7685

Epoch 22/57
----------
train Loss: 0.0074 Acc: 0.7920
val Loss: 0.0127 Acc: 0.7685

Epoch 23/57
----------
train Loss: 0.0072 Acc: 0.8257
val Loss: 0.0160 Acc: 0.7593

Epoch 24/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0073 Acc: 0.7798
val Loss: 0.0121 Acc: 0.7593

Epoch 25/57
----------
train Loss: 0.0076 Acc: 0.8226
val Loss: 0.0140 Acc: 0.7593

Epoch 26/57
----------
train Loss: 0.0074 Acc: 0.7982
val Loss: 0.0154 Acc: 0.7778

Epoch 27/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0070 Acc: 0.7982
val Loss: 0.0194 Acc: 0.7870

Epoch 28/57
----------
train Loss: 0.0072 Acc: 0.7982
val Loss: 0.0113 Acc: 0.7685

Epoch 29/57
----------
train Loss: 0.0077 Acc: 0.7859
val Loss: 0.0115 Acc: 0.7685

Epoch 30/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0073 Acc: 0.8196
val Loss: 0.0134 Acc: 0.7685

Epoch 31/57
----------
train Loss: 0.0069 Acc: 0.7859
val Loss: 0.0123 Acc: 0.7685

Epoch 32/57
----------
train Loss: 0.0070 Acc: 0.8226
val Loss: 0.0136 Acc: 0.7593

Epoch 33/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0076 Acc: 0.8104
val Loss: 0.0132 Acc: 0.7593

Epoch 34/57
----------
train Loss: 0.0076 Acc: 0.8104
val Loss: 0.0133 Acc: 0.7778

Epoch 35/57
----------
train Loss: 0.0072 Acc: 0.8073
val Loss: 0.0095 Acc: 0.7593

Epoch 36/57
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0071 Acc: 0.8073
val Loss: 0.0093 Acc: 0.7593

Epoch 37/57
----------
train Loss: 0.0074 Acc: 0.8104
val Loss: 0.0114 Acc: 0.7685

Epoch 38/57
----------
train Loss: 0.0067 Acc: 0.8043
val Loss: 0.0142 Acc: 0.7685

Epoch 39/57
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0075 Acc: 0.8165
val Loss: 0.0111 Acc: 0.7593

Epoch 40/57
----------
train Loss: 0.0074 Acc: 0.8165
val Loss: 0.0110 Acc: 0.7685

Epoch 41/57
----------
train Loss: 0.0072 Acc: 0.8073
val Loss: 0.0121 Acc: 0.7593

Epoch 42/57
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0071 Acc: 0.7982
val Loss: 0.0119 Acc: 0.7593

Epoch 43/57
----------
train Loss: 0.0079 Acc: 0.8073
val Loss: 0.0143 Acc: 0.7685

Epoch 44/57
----------
train Loss: 0.0078 Acc: 0.7951
val Loss: 0.0127 Acc: 0.7685

Epoch 45/57
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0074 Acc: 0.8073
val Loss: 0.0133 Acc: 0.7685

Epoch 46/57
----------
train Loss: 0.0077 Acc: 0.8043
val Loss: 0.0133 Acc: 0.7593

Epoch 47/57
----------
train Loss: 0.0079 Acc: 0.7859
val Loss: 0.0128 Acc: 0.7593

Epoch 48/57
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0074 Acc: 0.7951
val Loss: 0.0123 Acc: 0.7685

Epoch 49/57
----------
train Loss: 0.0077 Acc: 0.8073
val Loss: 0.0120 Acc: 0.7685

Epoch 50/57
----------
train Loss: 0.0070 Acc: 0.8012
val Loss: 0.0151 Acc: 0.7685

Epoch 51/57
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0070 Acc: 0.8226
val Loss: 0.0117 Acc: 0.7685

Epoch 52/57
----------
train Loss: 0.0074 Acc: 0.8135
val Loss: 0.0107 Acc: 0.7593

Epoch 53/57
----------
train Loss: 0.0080 Acc: 0.7829
val Loss: 0.0136 Acc: 0.7685

Epoch 54/57
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0068 Acc: 0.8226
val Loss: 0.0119 Acc: 0.7685

Epoch 55/57
----------
train Loss: 0.0074 Acc: 0.8135
val Loss: 0.0122 Acc: 0.7685

Epoch 56/57
----------
train Loss: 0.0070 Acc: 0.8257
val Loss: 0.0150 Acc: 0.7685

Epoch 57/57
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0075 Acc: 0.8043
val Loss: 0.0112 Acc: 0.7685

Training complete in 2m 11s
Best val Acc: 0.787037

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0069 Acc: 0.8165
val Loss: 0.0115 Acc: 0.8241

Epoch 1/57
----------
train Loss: 0.0047 Acc: 0.9297
val Loss: 0.0128 Acc: 0.8889

Epoch 2/57
----------
train Loss: 0.0026 Acc: 0.9602
val Loss: 0.0059 Acc: 0.9074

Epoch 3/57
----------
LR is set to 0.001
train Loss: 0.0012 Acc: 0.9939
val Loss: 0.0057 Acc: 0.8889

Epoch 4/57
----------
train Loss: 0.0011 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8796

Epoch 5/57
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8796

Epoch 6/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0048 Acc: 0.8796

Epoch 7/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0108 Acc: 0.8889

Epoch 8/57
----------
train Loss: 0.0011 Acc: 0.9939
val Loss: 0.0068 Acc: 0.8796

Epoch 9/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0053 Acc: 0.8796

Epoch 10/57
----------
train Loss: 0.0011 Acc: 0.9939
val Loss: 0.0067 Acc: 0.8889

Epoch 11/57
----------
train Loss: 0.0011 Acc: 0.9939
val Loss: 0.0042 Acc: 0.8796

Epoch 12/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0065 Acc: 0.8796

Epoch 13/57
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8796

Epoch 14/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0041 Acc: 0.8704

Epoch 15/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8796

Epoch 16/57
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8796

Epoch 17/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0061 Acc: 0.8796

Epoch 18/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0072 Acc: 0.8889

Epoch 19/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0064 Acc: 0.8796

Epoch 20/57
----------
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0059 Acc: 0.8796

Epoch 21/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0064 Acc: 0.8889

Epoch 22/57
----------
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0044 Acc: 0.8889

Epoch 23/57
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8796

Epoch 24/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8889

Epoch 25/57
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8889

Epoch 26/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0049 Acc: 0.8796

Epoch 27/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0053 Acc: 0.8889

Epoch 28/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0070 Acc: 0.8889

Epoch 29/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0071 Acc: 0.8796

Epoch 30/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0045 Acc: 0.8796

Epoch 31/57
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8796

Epoch 32/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0083 Acc: 0.8889

Epoch 33/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0062 Acc: 0.8796

Epoch 34/57
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8796

Epoch 35/57
----------
train Loss: 0.0012 Acc: 0.9969
val Loss: 0.0040 Acc: 0.8796

Epoch 36/57
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8796

Epoch 37/57
----------
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0050 Acc: 0.8796

Epoch 38/57
----------
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0059 Acc: 0.8704

Epoch 39/57
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0095 Acc: 0.8796

Epoch 40/57
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8796

Epoch 41/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0042 Acc: 0.8796

Epoch 42/57
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0056 Acc: 0.8796

Epoch 43/57
----------
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0067 Acc: 0.8796

Epoch 44/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0042 Acc: 0.8889

Epoch 45/57
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0012 Acc: 0.9908
val Loss: 0.0057 Acc: 0.8889

Epoch 46/57
----------
train Loss: 0.0011 Acc: 0.9939
val Loss: 0.0073 Acc: 0.8889

Epoch 47/57
----------
train Loss: 0.0012 Acc: 0.9939
val Loss: 0.0073 Acc: 0.8889

Epoch 48/57
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0060 Acc: 0.8796

Epoch 49/57
----------
train Loss: 0.0012 Acc: 0.9969
val Loss: 0.0056 Acc: 0.8796

Epoch 50/57
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8796

Epoch 51/57
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0078 Acc: 0.8796

Epoch 52/57
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0071 Acc: 0.8889

Epoch 53/57
----------
train Loss: 0.0010 Acc: 0.9969
val Loss: 0.0048 Acc: 0.8796

Epoch 54/57
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8796

Epoch 55/57
----------
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0049 Acc: 0.8796

Epoch 56/57
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8889

Epoch 57/57
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8889

Training complete in 2m 22s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.970115
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 94 %
Accuracy of Rhinobatiformes : 100 %
Accuracy of Torpediniformes : 95 %
mean: 0.9721230648277904, std: 0.018550697276318195
--------------------

run info[val: 0.3, epoch: 93, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0195 Acc: 0.3213
val Loss: 0.0347 Acc: 0.3923

Epoch 1/92
----------
train Loss: 0.0269 Acc: 0.4295
val Loss: 0.0242 Acc: 0.3385

Epoch 2/92
----------
train Loss: 0.0146 Acc: 0.4918
val Loss: 0.0216 Acc: 0.5000

Epoch 3/92
----------
train Loss: 0.0175 Acc: 0.5738
val Loss: 0.0159 Acc: 0.6615

Epoch 4/92
----------
train Loss: 0.0159 Acc: 0.4951
val Loss: 0.0147 Acc: 0.5923

Epoch 5/92
----------
train Loss: 0.0099 Acc: 0.7082
val Loss: 0.0166 Acc: 0.7462

Epoch 6/92
----------
train Loss: 0.0101 Acc: 0.7246
val Loss: 0.0104 Acc: 0.7462

Epoch 7/92
----------
train Loss: 0.0161 Acc: 0.7541
val Loss: 0.0137 Acc: 0.6615

Epoch 8/92
----------
train Loss: 0.0101 Acc: 0.8131
val Loss: 0.0101 Acc: 0.7385

Epoch 9/92
----------
train Loss: 0.0103 Acc: 0.7934
val Loss: 0.0082 Acc: 0.8231

Epoch 10/92
----------
train Loss: 0.0049 Acc: 0.8328
val Loss: 0.0122 Acc: 0.8000

Epoch 11/92
----------
train Loss: 0.0053 Acc: 0.8656
val Loss: 0.0078 Acc: 0.8308

Epoch 12/92
----------
train Loss: 0.0077 Acc: 0.8623
val Loss: 0.0088 Acc: 0.8077

Epoch 13/92
----------
LR is set to 0.001
train Loss: 0.0040 Acc: 0.8918
val Loss: 0.0083 Acc: 0.8385

Epoch 14/92
----------
train Loss: 0.0043 Acc: 0.9016
val Loss: 0.0067 Acc: 0.8538

Epoch 15/92
----------
train Loss: 0.0063 Acc: 0.8721
val Loss: 0.0104 Acc: 0.8308

Epoch 16/92
----------
train Loss: 0.0057 Acc: 0.8951
val Loss: 0.0062 Acc: 0.8385

Epoch 17/92
----------
train Loss: 0.0052 Acc: 0.8852
val Loss: 0.0061 Acc: 0.8692

Epoch 18/92
----------
train Loss: 0.0076 Acc: 0.9016
val Loss: 0.0087 Acc: 0.8385

Epoch 19/92
----------
train Loss: 0.0034 Acc: 0.9115
val Loss: 0.0090 Acc: 0.8615

Epoch 20/92
----------
train Loss: 0.0034 Acc: 0.9082
val Loss: 0.0077 Acc: 0.8615

Epoch 21/92
----------
train Loss: 0.0068 Acc: 0.8918
val Loss: 0.0074 Acc: 0.8462

Epoch 22/92
----------
train Loss: 0.0027 Acc: 0.9115
val Loss: 0.0076 Acc: 0.8692

Epoch 23/92
----------
train Loss: 0.0035 Acc: 0.9016
val Loss: 0.0071 Acc: 0.8615

Epoch 24/92
----------
train Loss: 0.0034 Acc: 0.8951
val Loss: 0.0060 Acc: 0.8538

Epoch 25/92
----------
train Loss: 0.0031 Acc: 0.9279
val Loss: 0.0081 Acc: 0.8769

Epoch 26/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0076 Acc: 0.8885
val Loss: 0.0074 Acc: 0.8615

Epoch 27/92
----------
train Loss: 0.0037 Acc: 0.9148
val Loss: 0.0084 Acc: 0.8692

Epoch 28/92
----------
train Loss: 0.0032 Acc: 0.9213
val Loss: 0.0076 Acc: 0.8615

Epoch 29/92
----------
train Loss: 0.0055 Acc: 0.9213
val Loss: 0.0074 Acc: 0.8692

Epoch 30/92
----------
train Loss: 0.0044 Acc: 0.9082
val Loss: 0.0070 Acc: 0.8692

Epoch 31/92
----------
train Loss: 0.0064 Acc: 0.9148
val Loss: 0.0080 Acc: 0.8615

Epoch 32/92
----------
train Loss: 0.0030 Acc: 0.9311
val Loss: 0.0065 Acc: 0.8538

Epoch 33/92
----------
train Loss: 0.0052 Acc: 0.9279
val Loss: 0.0064 Acc: 0.8538

Epoch 34/92
----------
train Loss: 0.0029 Acc: 0.9246
val Loss: 0.0080 Acc: 0.8385

Epoch 35/92
----------
train Loss: 0.0033 Acc: 0.9311
val Loss: 0.0075 Acc: 0.8462

Epoch 36/92
----------
train Loss: 0.0039 Acc: 0.9213
val Loss: 0.0072 Acc: 0.8538

Epoch 37/92
----------
train Loss: 0.0063 Acc: 0.9180
val Loss: 0.0093 Acc: 0.8692

Epoch 38/92
----------
train Loss: 0.0057 Acc: 0.9016
val Loss: 0.0075 Acc: 0.8615

Epoch 39/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0035 Acc: 0.9115
val Loss: 0.0062 Acc: 0.8538

Epoch 40/92
----------
train Loss: 0.0051 Acc: 0.9049
val Loss: 0.0074 Acc: 0.8615

Epoch 41/92
----------
train Loss: 0.0024 Acc: 0.9410
val Loss: 0.0076 Acc: 0.8692

Epoch 42/92
----------
train Loss: 0.0026 Acc: 0.9344
val Loss: 0.0075 Acc: 0.8462

Epoch 43/92
----------
train Loss: 0.0028 Acc: 0.9180
val Loss: 0.0077 Acc: 0.8462

Epoch 44/92
----------
train Loss: 0.0035 Acc: 0.9049
val Loss: 0.0060 Acc: 0.8538

Epoch 45/92
----------
train Loss: 0.0036 Acc: 0.9180
val Loss: 0.0077 Acc: 0.8462

Epoch 46/92
----------
train Loss: 0.0033 Acc: 0.9082
val Loss: 0.0079 Acc: 0.8538

Epoch 47/92
----------
train Loss: 0.0052 Acc: 0.9148
val Loss: 0.0063 Acc: 0.8538

Epoch 48/92
----------
train Loss: 0.0036 Acc: 0.9082
val Loss: 0.0078 Acc: 0.8538

Epoch 49/92
----------
train Loss: 0.0057 Acc: 0.9246
val Loss: 0.0063 Acc: 0.8462

Epoch 50/92
----------
train Loss: 0.0028 Acc: 0.9148
val Loss: 0.0077 Acc: 0.8538

Epoch 51/92
----------
train Loss: 0.0073 Acc: 0.8852
val Loss: 0.0071 Acc: 0.8692

Epoch 52/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0045 Acc: 0.9148
val Loss: 0.0084 Acc: 0.8615

Epoch 53/92
----------
train Loss: 0.0055 Acc: 0.9115
val Loss: 0.0073 Acc: 0.8538

Epoch 54/92
----------
train Loss: 0.0026 Acc: 0.9279
val Loss: 0.0072 Acc: 0.8462

Epoch 55/92
----------
train Loss: 0.0026 Acc: 0.9377
val Loss: 0.0078 Acc: 0.8538

Epoch 56/92
----------
train Loss: 0.0038 Acc: 0.9148
val Loss: 0.0070 Acc: 0.8538

Epoch 57/92
----------
train Loss: 0.0056 Acc: 0.9344
val Loss: 0.0066 Acc: 0.8385

Epoch 58/92
----------
train Loss: 0.0042 Acc: 0.9279
val Loss: 0.0070 Acc: 0.8538

Epoch 59/92
----------
train Loss: 0.0044 Acc: 0.9148
val Loss: 0.0064 Acc: 0.8615

Epoch 60/92
----------
train Loss: 0.0044 Acc: 0.9180
val Loss: 0.0063 Acc: 0.8692

Epoch 61/92
----------
train Loss: 0.0046 Acc: 0.9311
val Loss: 0.0065 Acc: 0.8615

Epoch 62/92
----------
train Loss: 0.0086 Acc: 0.9115
val Loss: 0.0059 Acc: 0.8462

Epoch 63/92
----------
train Loss: 0.0039 Acc: 0.8984
val Loss: 0.0069 Acc: 0.8462

Epoch 64/92
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0083 Acc: 0.8538

Epoch 65/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.8885
val Loss: 0.0056 Acc: 0.8462

Epoch 66/92
----------
train Loss: 0.0082 Acc: 0.8984
val Loss: 0.0080 Acc: 0.8462

Epoch 67/92
----------
train Loss: 0.0026 Acc: 0.9213
val Loss: 0.0066 Acc: 0.8462

Epoch 68/92
----------
train Loss: 0.0029 Acc: 0.9148
val Loss: 0.0076 Acc: 0.8538

Epoch 69/92
----------
train Loss: 0.0033 Acc: 0.9148
val Loss: 0.0080 Acc: 0.8615

Epoch 70/92
----------
train Loss: 0.0061 Acc: 0.8885
val Loss: 0.0075 Acc: 0.8615

Epoch 71/92
----------
train Loss: 0.0074 Acc: 0.9213
val Loss: 0.0065 Acc: 0.8615

Epoch 72/92
----------
train Loss: 0.0058 Acc: 0.9148
val Loss: 0.0076 Acc: 0.8538

Epoch 73/92
----------
train Loss: 0.0050 Acc: 0.9148
val Loss: 0.0086 Acc: 0.8385

Epoch 74/92
----------
train Loss: 0.0025 Acc: 0.9246
val Loss: 0.0078 Acc: 0.8462

Epoch 75/92
----------
train Loss: 0.0034 Acc: 0.9213
val Loss: 0.0064 Acc: 0.8462

Epoch 76/92
----------
train Loss: 0.0026 Acc: 0.9279
val Loss: 0.0069 Acc: 0.8462

Epoch 77/92
----------
train Loss: 0.0046 Acc: 0.9082
val Loss: 0.0087 Acc: 0.8538

Epoch 78/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0036 Acc: 0.9213
val Loss: 0.0077 Acc: 0.8615

Epoch 79/92
----------
train Loss: 0.0044 Acc: 0.9148
val Loss: 0.0065 Acc: 0.8538

Epoch 80/92
----------
train Loss: 0.0036 Acc: 0.9311
val Loss: 0.0069 Acc: 0.8538

Epoch 81/92
----------
train Loss: 0.0032 Acc: 0.9246
val Loss: 0.0060 Acc: 0.8462

Epoch 82/92
----------
train Loss: 0.0039 Acc: 0.9246
val Loss: 0.0087 Acc: 0.8538

Epoch 83/92
----------
train Loss: 0.0056 Acc: 0.9115
val Loss: 0.0061 Acc: 0.8385

Epoch 84/92
----------
train Loss: 0.0041 Acc: 0.9049
val Loss: 0.0064 Acc: 0.8308

Epoch 85/92
----------
train Loss: 0.0029 Acc: 0.9279
val Loss: 0.0086 Acc: 0.8462

Epoch 86/92
----------
train Loss: 0.0033 Acc: 0.9279
val Loss: 0.0069 Acc: 0.8462

Epoch 87/92
----------
train Loss: 0.0042 Acc: 0.9246
val Loss: 0.0076 Acc: 0.8615

Epoch 88/92
----------
train Loss: 0.0038 Acc: 0.8951
val Loss: 0.0062 Acc: 0.8462

Epoch 89/92
----------
train Loss: 0.0041 Acc: 0.9279
val Loss: 0.0070 Acc: 0.8615

Epoch 90/92
----------
train Loss: 0.0035 Acc: 0.9180
val Loss: 0.0062 Acc: 0.8462

Epoch 91/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0036 Acc: 0.9049
val Loss: 0.0064 Acc: 0.8692

Epoch 92/92
----------
train Loss: 0.0037 Acc: 0.9180
val Loss: 0.0089 Acc: 0.8615

Training complete in 3m 30s
Best val Acc: 0.876923

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0050 Acc: 0.9049
val Loss: 0.0065 Acc: 0.8692

Epoch 1/92
----------
train Loss: 0.0029 Acc: 0.9246
val Loss: 0.0269 Acc: 0.5231

Epoch 2/92
----------
train Loss: 0.0026 Acc: 0.9410
val Loss: 0.0146 Acc: 0.7077

Epoch 3/92
----------
train Loss: 0.0034 Acc: 0.9410
val Loss: 0.0172 Acc: 0.6308

Epoch 4/92
----------
train Loss: 0.0048 Acc: 0.9279
val Loss: 0.0452 Acc: 0.5077

Epoch 5/92
----------
train Loss: 0.0137 Acc: 0.8557
val Loss: 0.0608 Acc: 0.4769

Epoch 6/92
----------
train Loss: 0.0168 Acc: 0.7311
val Loss: 0.1035 Acc: 0.2231

Epoch 7/92
----------
train Loss: 0.0311 Acc: 0.5180
val Loss: 0.0990 Acc: 0.1615

Epoch 8/92
----------
train Loss: 0.0222 Acc: 0.5344
val Loss: 0.2222 Acc: 0.3923

Epoch 9/92
----------
train Loss: 0.0332 Acc: 0.5574
val Loss: 0.5861 Acc: 0.2231

Epoch 10/92
----------
train Loss: 0.0184 Acc: 0.6262
val Loss: 0.7442 Acc: 0.2308

Epoch 11/92
----------
train Loss: 0.0163 Acc: 0.6033
val Loss: 0.2590 Acc: 0.4231

Epoch 12/92
----------
train Loss: 0.0130 Acc: 0.6689
val Loss: 0.1295 Acc: 0.4846

Epoch 13/92
----------
LR is set to 0.001
train Loss: 0.0116 Acc: 0.6295
val Loss: 0.0457 Acc: 0.5923

Epoch 14/92
----------
train Loss: 0.0130 Acc: 0.6689
val Loss: 0.0328 Acc: 0.6000

Epoch 15/92
----------
train Loss: 0.0151 Acc: 0.7082
val Loss: 0.0353 Acc: 0.5615

Epoch 16/92
----------
train Loss: 0.0116 Acc: 0.6951
val Loss: 0.0331 Acc: 0.5923

Epoch 17/92
----------
train Loss: 0.0111 Acc: 0.7410
val Loss: 0.0268 Acc: 0.6077

Epoch 18/92
----------
train Loss: 0.0097 Acc: 0.7541
val Loss: 0.0249 Acc: 0.6000

Epoch 19/92
----------
train Loss: 0.0114 Acc: 0.7541
val Loss: 0.0194 Acc: 0.5923

Epoch 20/92
----------
train Loss: 0.0168 Acc: 0.7279
val Loss: 0.0157 Acc: 0.6538

Epoch 21/92
----------
train Loss: 0.0126 Acc: 0.7803
val Loss: 0.0175 Acc: 0.6308

Epoch 22/92
----------
train Loss: 0.0062 Acc: 0.7934
val Loss: 0.0146 Acc: 0.6231

Epoch 23/92
----------
train Loss: 0.0150 Acc: 0.7738
val Loss: 0.0151 Acc: 0.6538

Epoch 24/92
----------
train Loss: 0.0117 Acc: 0.8066
val Loss: 0.0160 Acc: 0.6615

Epoch 25/92
----------
train Loss: 0.0054 Acc: 0.8459
val Loss: 0.0170 Acc: 0.6385

Epoch 26/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0067 Acc: 0.8197
val Loss: 0.0214 Acc: 0.6615

Epoch 27/92
----------
train Loss: 0.0088 Acc: 0.7934
val Loss: 0.0169 Acc: 0.6615

Epoch 28/92
----------
train Loss: 0.0115 Acc: 0.8098
val Loss: 0.0171 Acc: 0.6538

Epoch 29/92
----------
train Loss: 0.0058 Acc: 0.8164
val Loss: 0.0181 Acc: 0.6538

Epoch 30/92
----------
train Loss: 0.0067 Acc: 0.8131
val Loss: 0.0198 Acc: 0.6462

Epoch 31/92
----------
train Loss: 0.0124 Acc: 0.8295
val Loss: 0.0168 Acc: 0.6538

Epoch 32/92
----------
train Loss: 0.0079 Acc: 0.8393
val Loss: 0.0168 Acc: 0.6538

Epoch 33/92
----------
train Loss: 0.0066 Acc: 0.8262
val Loss: 0.0152 Acc: 0.6615

Epoch 34/92
----------
train Loss: 0.0082 Acc: 0.8262
val Loss: 0.0144 Acc: 0.6538

Epoch 35/92
----------
train Loss: 0.0082 Acc: 0.8361
val Loss: 0.0174 Acc: 0.6385

Epoch 36/92
----------
train Loss: 0.0049 Acc: 0.8492
val Loss: 0.0178 Acc: 0.6538

Epoch 37/92
----------
train Loss: 0.0085 Acc: 0.8295
val Loss: 0.0139 Acc: 0.6538

Epoch 38/92
----------
train Loss: 0.0066 Acc: 0.8426
val Loss: 0.0156 Acc: 0.6462

Epoch 39/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0061 Acc: 0.8426
val Loss: 0.0175 Acc: 0.6462

Epoch 40/92
----------
train Loss: 0.0058 Acc: 0.8590
val Loss: 0.0160 Acc: 0.6462

Epoch 41/92
----------
train Loss: 0.0053 Acc: 0.8689
val Loss: 0.0168 Acc: 0.6538

Epoch 42/92
----------
train Loss: 0.0084 Acc: 0.8295
val Loss: 0.0148 Acc: 0.6615

Epoch 43/92
----------
train Loss: 0.0089 Acc: 0.8393
val Loss: 0.0187 Acc: 0.6615

Epoch 44/92
----------
train Loss: 0.0080 Acc: 0.8492
val Loss: 0.0169 Acc: 0.6308

Epoch 45/92
----------
train Loss: 0.0114 Acc: 0.8295
val Loss: 0.0138 Acc: 0.6462

Epoch 46/92
----------
train Loss: 0.0104 Acc: 0.8426
val Loss: 0.0142 Acc: 0.6308

Epoch 47/92
----------
train Loss: 0.0089 Acc: 0.8426
val Loss: 0.0155 Acc: 0.6538

Epoch 48/92
----------
train Loss: 0.0089 Acc: 0.8492
val Loss: 0.0176 Acc: 0.6385

Epoch 49/92
----------
train Loss: 0.0083 Acc: 0.8361
val Loss: 0.0161 Acc: 0.6615

Epoch 50/92
----------
train Loss: 0.0086 Acc: 0.8197
val Loss: 0.0161 Acc: 0.6538

Epoch 51/92
----------
train Loss: 0.0086 Acc: 0.8393
val Loss: 0.0135 Acc: 0.6769

Epoch 52/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0058 Acc: 0.8557
val Loss: 0.0150 Acc: 0.6538

Epoch 53/92
----------
train Loss: 0.0062 Acc: 0.8328
val Loss: 0.0147 Acc: 0.6538

Epoch 54/92
----------
train Loss: 0.0098 Acc: 0.8525
val Loss: 0.0157 Acc: 0.6538

Epoch 55/92
----------
train Loss: 0.0053 Acc: 0.8557
val Loss: 0.0165 Acc: 0.6538

Epoch 56/92
----------
train Loss: 0.0049 Acc: 0.8623
val Loss: 0.0141 Acc: 0.6615

Epoch 57/92
----------
train Loss: 0.0063 Acc: 0.8328
val Loss: 0.0162 Acc: 0.6846

Epoch 58/92
----------
train Loss: 0.0063 Acc: 0.8557
val Loss: 0.0181 Acc: 0.6615

Epoch 59/92
----------
train Loss: 0.0075 Acc: 0.8525
val Loss: 0.0153 Acc: 0.6385

Epoch 60/92
----------
train Loss: 0.0070 Acc: 0.8328
val Loss: 0.0147 Acc: 0.6462

Epoch 61/92
----------
train Loss: 0.0048 Acc: 0.8492
val Loss: 0.0158 Acc: 0.6538

Epoch 62/92
----------
train Loss: 0.0055 Acc: 0.8689
val Loss: 0.0154 Acc: 0.6615

Epoch 63/92
----------
train Loss: 0.0086 Acc: 0.8295
val Loss: 0.0146 Acc: 0.6923

Epoch 64/92
----------
train Loss: 0.0066 Acc: 0.8262
val Loss: 0.0152 Acc: 0.6846

Epoch 65/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0068 Acc: 0.8492
val Loss: 0.0143 Acc: 0.6923

Epoch 66/92
----------
train Loss: 0.0080 Acc: 0.8328
val Loss: 0.0154 Acc: 0.7000

Epoch 67/92
----------
train Loss: 0.0104 Acc: 0.8262
val Loss: 0.0184 Acc: 0.6538

Epoch 68/92
----------
train Loss: 0.0092 Acc: 0.8393
val Loss: 0.0188 Acc: 0.6692

Epoch 69/92
----------
train Loss: 0.0151 Acc: 0.8492
val Loss: 0.0139 Acc: 0.6923

Epoch 70/92
----------
train Loss: 0.0092 Acc: 0.8590
val Loss: 0.0158 Acc: 0.7000

Epoch 71/92
----------
train Loss: 0.0064 Acc: 0.8590
val Loss: 0.0150 Acc: 0.6769

Epoch 72/92
----------
train Loss: 0.0081 Acc: 0.8459
val Loss: 0.0142 Acc: 0.6615

Epoch 73/92
----------
train Loss: 0.0070 Acc: 0.8525
val Loss: 0.0144 Acc: 0.6769

Epoch 74/92
----------
train Loss: 0.0083 Acc: 0.8492
val Loss: 0.0147 Acc: 0.6615

Epoch 75/92
----------
train Loss: 0.0057 Acc: 0.8393
val Loss: 0.0131 Acc: 0.6769

Epoch 76/92
----------
train Loss: 0.0078 Acc: 0.8459
val Loss: 0.0138 Acc: 0.6385

Epoch 77/92
----------
train Loss: 0.0051 Acc: 0.8197
val Loss: 0.0185 Acc: 0.6615

Epoch 78/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0119 Acc: 0.8393
val Loss: 0.0161 Acc: 0.6846

Epoch 79/92
----------
train Loss: 0.0063 Acc: 0.8393
val Loss: 0.0167 Acc: 0.6846

Epoch 80/92
----------
train Loss: 0.0075 Acc: 0.8525
val Loss: 0.0148 Acc: 0.7000

Epoch 81/92
----------
train Loss: 0.0056 Acc: 0.8590
val Loss: 0.0169 Acc: 0.6769

Epoch 82/92
----------
train Loss: 0.0089 Acc: 0.8328
val Loss: 0.0167 Acc: 0.6615

Epoch 83/92
----------
train Loss: 0.0052 Acc: 0.8623
val Loss: 0.0185 Acc: 0.6846

Epoch 84/92
----------
train Loss: 0.0114 Acc: 0.8525
val Loss: 0.0156 Acc: 0.6923

Epoch 85/92
----------
train Loss: 0.0066 Acc: 0.8525
val Loss: 0.0196 Acc: 0.6923

Epoch 86/92
----------
train Loss: 0.0094 Acc: 0.8361
val Loss: 0.0179 Acc: 0.6769

Epoch 87/92
----------
train Loss: 0.0057 Acc: 0.8557
val Loss: 0.0147 Acc: 0.6769

Epoch 88/92
----------
train Loss: 0.0050 Acc: 0.8426
val Loss: 0.0160 Acc: 0.6615

Epoch 89/92
----------
train Loss: 0.0062 Acc: 0.8459
val Loss: 0.0186 Acc: 0.6769

Epoch 90/92
----------
train Loss: 0.0088 Acc: 0.8361
val Loss: 0.0144 Acc: 0.6692

Epoch 91/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0052 Acc: 0.8525
val Loss: 0.0137 Acc: 0.6385

Epoch 92/92
----------
train Loss: 0.0112 Acc: 0.8393
val Loss: 0.0164 Acc: 0.6538

Training complete in 3m 45s
Best val Acc: 0.869231

---Testing---
Test accuracy: 0.924138
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 89 %
Accuracy of Rajiformes : 84 %
Accuracy of Rhinobatiformes : 95 %
Accuracy of Torpediniformes : 95 %
mean: 0.9221570997510102, std: 0.044653865565237975

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.98]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 53, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0160 Acc: 0.3495
val Loss: 0.0422 Acc: 0.3953

Epoch 1/52
----------
train Loss: 0.0127 Acc: 0.5255
val Loss: 0.0266 Acc: 0.6512

Epoch 2/52
----------
train Loss: 0.0087 Acc: 0.7066
val Loss: 0.0190 Acc: 0.6744

Epoch 3/52
----------
train Loss: 0.0064 Acc: 0.7781
val Loss: 0.0172 Acc: 0.6977

Epoch 4/52
----------
train Loss: 0.0052 Acc: 0.8010
val Loss: 0.0138 Acc: 0.8372

Epoch 5/52
----------
train Loss: 0.0047 Acc: 0.8469
val Loss: 0.0135 Acc: 0.7907

Epoch 6/52
----------
train Loss: 0.0038 Acc: 0.8801
val Loss: 0.0127 Acc: 0.8605

Epoch 7/52
----------
train Loss: 0.0035 Acc: 0.9133
val Loss: 0.0125 Acc: 0.8372

Epoch 8/52
----------
train Loss: 0.0030 Acc: 0.9337
val Loss: 0.0128 Acc: 0.8372

Epoch 9/52
----------
train Loss: 0.0026 Acc: 0.9260
val Loss: 0.0119 Acc: 0.8372

Epoch 10/52
----------
LR is set to 0.001
train Loss: 0.0025 Acc: 0.9362
val Loss: 0.0121 Acc: 0.8372

Epoch 11/52
----------
train Loss: 0.0025 Acc: 0.9337
val Loss: 0.0121 Acc: 0.8372

Epoch 12/52
----------
train Loss: 0.0025 Acc: 0.9311
val Loss: 0.0123 Acc: 0.8372

Epoch 13/52
----------
train Loss: 0.0023 Acc: 0.9515
val Loss: 0.0123 Acc: 0.8372

Epoch 14/52
----------
train Loss: 0.0024 Acc: 0.9362
val Loss: 0.0123 Acc: 0.8372

Epoch 15/52
----------
train Loss: 0.0023 Acc: 0.9515
val Loss: 0.0123 Acc: 0.8372

Epoch 16/52
----------
train Loss: 0.0023 Acc: 0.9439
val Loss: 0.0123 Acc: 0.8372

Epoch 17/52
----------
train Loss: 0.0024 Acc: 0.9286
val Loss: 0.0122 Acc: 0.8372

Epoch 18/52
----------
train Loss: 0.0022 Acc: 0.9464
val Loss: 0.0122 Acc: 0.8372

Epoch 19/52
----------
train Loss: 0.0023 Acc: 0.9439
val Loss: 0.0122 Acc: 0.8372

Epoch 20/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9464
val Loss: 0.0123 Acc: 0.8372

Epoch 21/52
----------
train Loss: 0.0023 Acc: 0.9464
val Loss: 0.0123 Acc: 0.8372

Epoch 22/52
----------
train Loss: 0.0022 Acc: 0.9362
val Loss: 0.0123 Acc: 0.8372

Epoch 23/52
----------
train Loss: 0.0023 Acc: 0.9515
val Loss: 0.0123 Acc: 0.8372

Epoch 24/52
----------
train Loss: 0.0022 Acc: 0.9439
val Loss: 0.0122 Acc: 0.8372

Epoch 25/52
----------
train Loss: 0.0022 Acc: 0.9464
val Loss: 0.0122 Acc: 0.8372

Epoch 26/52
----------
train Loss: 0.0022 Acc: 0.9515
val Loss: 0.0122 Acc: 0.8372

Epoch 27/52
----------
train Loss: 0.0025 Acc: 0.9311
val Loss: 0.0122 Acc: 0.8372

Epoch 28/52
----------
train Loss: 0.0023 Acc: 0.9464
val Loss: 0.0123 Acc: 0.8372

Epoch 29/52
----------
train Loss: 0.0020 Acc: 0.9592
val Loss: 0.0123 Acc: 0.8372

Epoch 30/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0022 Acc: 0.9413
val Loss: 0.0123 Acc: 0.8372

Epoch 31/52
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0122 Acc: 0.8372

Epoch 32/52
----------
train Loss: 0.0022 Acc: 0.9566
val Loss: 0.0122 Acc: 0.8372

Epoch 33/52
----------
train Loss: 0.0022 Acc: 0.9490
val Loss: 0.0122 Acc: 0.8372

Epoch 34/52
----------
train Loss: 0.0023 Acc: 0.9413
val Loss: 0.0122 Acc: 0.8372

Epoch 35/52
----------
train Loss: 0.0022 Acc: 0.9464
val Loss: 0.0122 Acc: 0.8372

Epoch 36/52
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0122 Acc: 0.8372

Epoch 37/52
----------
train Loss: 0.0022 Acc: 0.9592
val Loss: 0.0122 Acc: 0.8372

Epoch 38/52
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0123 Acc: 0.8372

Epoch 39/52
----------
train Loss: 0.0021 Acc: 0.9643
val Loss: 0.0122 Acc: 0.8372

Epoch 40/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0123 Acc: 0.8372

Epoch 41/52
----------
train Loss: 0.0021 Acc: 0.9566
val Loss: 0.0122 Acc: 0.8372

Epoch 42/52
----------
train Loss: 0.0023 Acc: 0.9490
val Loss: 0.0122 Acc: 0.8372

Epoch 43/52
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0122 Acc: 0.8372

Epoch 44/52
----------
train Loss: 0.0022 Acc: 0.9617
val Loss: 0.0122 Acc: 0.8372

Epoch 45/52
----------
train Loss: 0.0023 Acc: 0.9515
val Loss: 0.0123 Acc: 0.8372

Epoch 46/52
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0122 Acc: 0.8372

Epoch 47/52
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0123 Acc: 0.8372

Epoch 48/52
----------
train Loss: 0.0023 Acc: 0.9490
val Loss: 0.0122 Acc: 0.8372

Epoch 49/52
----------
train Loss: 0.0021 Acc: 0.9668
val Loss: 0.0122 Acc: 0.8372

Epoch 50/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0021 Acc: 0.9668
val Loss: 0.0122 Acc: 0.8372

Epoch 51/52
----------
train Loss: 0.0023 Acc: 0.9515
val Loss: 0.0122 Acc: 0.8372

Epoch 52/52
----------
train Loss: 0.0024 Acc: 0.9439
val Loss: 0.0122 Acc: 0.8372

Training complete in 1m 51s
Best val Acc: 0.860465

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0034 Acc: 0.8903
val Loss: 0.0124 Acc: 0.8837

Epoch 1/52
----------
train Loss: 0.0015 Acc: 0.9898
val Loss: 0.0119 Acc: 0.8837

Epoch 2/52
----------
train Loss: 0.0009 Acc: 0.9872
val Loss: 0.0111 Acc: 0.8605

Epoch 3/52
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0125 Acc: 0.8605

Epoch 4/52
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0130 Acc: 0.8837

Epoch 5/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 6/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8837

Epoch 7/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8837

Epoch 8/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8605

Epoch 9/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 10/52
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 11/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 12/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 13/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 14/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 15/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 16/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 17/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 18/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 19/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 20/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 21/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 22/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 23/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 24/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 25/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 26/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 27/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 28/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 29/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 30/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 31/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 32/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 33/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 34/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 35/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 36/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 37/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 38/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 39/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 40/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 41/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 42/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 43/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 44/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 45/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 46/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 47/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 48/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8605

Epoch 49/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 50/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 51/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Epoch 52/52
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8605

Training complete in 2m 1s
Best val Acc: 0.883721

---Testing---
Test accuracy: 0.958621
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 92 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 97 %
mean: 0.9520332674484925, std: 0.023968045086884736
--------------------

run info[val: 0.15, epoch: 96, randcrop: False, decay: 4]

---Training last layer.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0158 Acc: 0.4108
val Loss: 0.0198 Acc: 0.5692

Epoch 1/95
----------
train Loss: 0.0123 Acc: 0.6270
val Loss: 0.0150 Acc: 0.6000

Epoch 2/95
----------
train Loss: 0.0091 Acc: 0.7000
val Loss: 0.0105 Acc: 0.7692

Epoch 3/95
----------
train Loss: 0.0074 Acc: 0.7757
val Loss: 0.0099 Acc: 0.7692

Epoch 4/95
----------
LR is set to 0.001
train Loss: 0.0065 Acc: 0.7865
val Loss: 0.0089 Acc: 0.8000

Epoch 5/95
----------
train Loss: 0.0058 Acc: 0.8189
val Loss: 0.0083 Acc: 0.8308

Epoch 6/95
----------
train Loss: 0.0055 Acc: 0.8514
val Loss: 0.0084 Acc: 0.8154

Epoch 7/95
----------
train Loss: 0.0058 Acc: 0.8243
val Loss: 0.0083 Acc: 0.8000

Epoch 8/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0083 Acc: 0.8000

Epoch 9/95
----------
train Loss: 0.0055 Acc: 0.8486
val Loss: 0.0083 Acc: 0.8308

Epoch 10/95
----------
train Loss: 0.0055 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 11/95
----------
train Loss: 0.0054 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 12/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0053 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 13/95
----------
train Loss: 0.0052 Acc: 0.8649
val Loss: 0.0082 Acc: 0.8308

Epoch 14/95
----------
train Loss: 0.0052 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 15/95
----------
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 16/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 17/95
----------
train Loss: 0.0053 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 18/95
----------
train Loss: 0.0052 Acc: 0.8649
val Loss: 0.0082 Acc: 0.8308

Epoch 19/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 20/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0053 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 21/95
----------
train Loss: 0.0055 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 22/95
----------
train Loss: 0.0052 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 23/95
----------
train Loss: 0.0055 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 24/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0054 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 25/95
----------
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0082 Acc: 0.8308

Epoch 26/95
----------
train Loss: 0.0053 Acc: 0.8622
val Loss: 0.0082 Acc: 0.8308

Epoch 27/95
----------
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 28/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0052 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 29/95
----------
train Loss: 0.0053 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 30/95
----------
train Loss: 0.0052 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 31/95
----------
train Loss: 0.0052 Acc: 0.8595
val Loss: 0.0082 Acc: 0.8308

Epoch 32/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0052 Acc: 0.8459
val Loss: 0.0082 Acc: 0.8308

Epoch 33/95
----------
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 34/95
----------
train Loss: 0.0052 Acc: 0.8676
val Loss: 0.0082 Acc: 0.8308

Epoch 35/95
----------
train Loss: 0.0051 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 36/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0051 Acc: 0.8595
val Loss: 0.0082 Acc: 0.8308

Epoch 37/95
----------
train Loss: 0.0054 Acc: 0.8676
val Loss: 0.0082 Acc: 0.8308

Epoch 38/95
----------
train Loss: 0.0052 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 39/95
----------
train Loss: 0.0054 Acc: 0.8622
val Loss: 0.0082 Acc: 0.8308

Epoch 40/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0051 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 41/95
----------
train Loss: 0.0054 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 42/95
----------
train Loss: 0.0052 Acc: 0.8595
val Loss: 0.0082 Acc: 0.8308

Epoch 43/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 44/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0053 Acc: 0.8378
val Loss: 0.0082 Acc: 0.8308

Epoch 45/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 46/95
----------
train Loss: 0.0052 Acc: 0.8649
val Loss: 0.0082 Acc: 0.8308

Epoch 47/95
----------
train Loss: 0.0054 Acc: 0.8378
val Loss: 0.0082 Acc: 0.8462

Epoch 48/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0054 Acc: 0.8324
val Loss: 0.0082 Acc: 0.8308

Epoch 49/95
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0082 Acc: 0.8308

Epoch 50/95
----------
train Loss: 0.0054 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 51/95
----------
train Loss: 0.0053 Acc: 0.8622
val Loss: 0.0082 Acc: 0.8308

Epoch 52/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0082 Acc: 0.8308

Epoch 53/95
----------
train Loss: 0.0053 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 54/95
----------
train Loss: 0.0054 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 55/95
----------
train Loss: 0.0053 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 56/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0053 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 57/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 58/95
----------
train Loss: 0.0054 Acc: 0.8378
val Loss: 0.0082 Acc: 0.8308

Epoch 59/95
----------
train Loss: 0.0053 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 60/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0054 Acc: 0.8649
val Loss: 0.0082 Acc: 0.8308

Epoch 61/95
----------
train Loss: 0.0053 Acc: 0.8622
val Loss: 0.0082 Acc: 0.8308

Epoch 62/95
----------
train Loss: 0.0052 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 63/95
----------
train Loss: 0.0054 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 64/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0052 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 65/95
----------
train Loss: 0.0053 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 66/95
----------
train Loss: 0.0054 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 67/95
----------
train Loss: 0.0052 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 68/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0054 Acc: 0.8324
val Loss: 0.0082 Acc: 0.8308

Epoch 69/95
----------
train Loss: 0.0054 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 70/95
----------
train Loss: 0.0055 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 71/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 72/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0053 Acc: 0.8297
val Loss: 0.0082 Acc: 0.8308

Epoch 73/95
----------
train Loss: 0.0053 Acc: 0.8324
val Loss: 0.0082 Acc: 0.8308

Epoch 74/95
----------
train Loss: 0.0054 Acc: 0.8459
val Loss: 0.0082 Acc: 0.8308

Epoch 75/95
----------
train Loss: 0.0052 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 76/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0051 Acc: 0.8622
val Loss: 0.0082 Acc: 0.8308

Epoch 77/95
----------
train Loss: 0.0053 Acc: 0.8649
val Loss: 0.0082 Acc: 0.8308

Epoch 78/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 79/95
----------
train Loss: 0.0055 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 80/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0054 Acc: 0.8405
val Loss: 0.0082 Acc: 0.8308

Epoch 81/95
----------
train Loss: 0.0053 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 82/95
----------
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 83/95
----------
train Loss: 0.0054 Acc: 0.8297
val Loss: 0.0082 Acc: 0.8308

Epoch 84/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0083 Acc: 0.8308

Epoch 85/95
----------
train Loss: 0.0051 Acc: 0.8568
val Loss: 0.0082 Acc: 0.8308

Epoch 86/95
----------
train Loss: 0.0053 Acc: 0.8459
val Loss: 0.0082 Acc: 0.8308

Epoch 87/95
----------
train Loss: 0.0054 Acc: 0.8297
val Loss: 0.0082 Acc: 0.8308

Epoch 88/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0053 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 89/95
----------
train Loss: 0.0054 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 90/95
----------
train Loss: 0.0053 Acc: 0.8514
val Loss: 0.0082 Acc: 0.8308

Epoch 91/95
----------
train Loss: 0.0052 Acc: 0.8541
val Loss: 0.0082 Acc: 0.8308

Epoch 92/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0054 Acc: 0.8432
val Loss: 0.0082 Acc: 0.8308

Epoch 93/95
----------
train Loss: 0.0055 Acc: 0.8324
val Loss: 0.0082 Acc: 0.8308

Epoch 94/95
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0082 Acc: 0.8308

Epoch 95/95
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0082 Acc: 0.8308

Training complete in 3m 24s
Best val Acc: 0.846154

---Fine tuning.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8378
val Loss: 0.0067 Acc: 0.8923

Epoch 1/95
----------
train Loss: 0.0032 Acc: 0.9270
val Loss: 0.0059 Acc: 0.9077

Epoch 2/95
----------
train Loss: 0.0016 Acc: 0.9865
val Loss: 0.0061 Acc: 0.8615

Epoch 3/95
----------
train Loss: 0.0007 Acc: 0.9973
val Loss: 0.0053 Acc: 0.9077

Epoch 4/95
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0052 Acc: 0.9077

Epoch 5/95
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 6/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 7/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 8/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 9/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 10/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 11/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 12/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 13/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 14/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 15/95
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 16/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 17/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 18/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 19/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 20/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 21/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 22/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 23/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 24/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 25/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 26/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 27/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 28/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 29/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 30/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 31/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 32/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 33/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 34/95
----------
train Loss: 0.0004 Acc: 0.9946
val Loss: 0.0051 Acc: 0.9077

Epoch 35/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 36/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0004 Acc: 0.9946
val Loss: 0.0051 Acc: 0.9077

Epoch 37/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 38/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 39/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 40/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 41/95
----------
train Loss: 0.0004 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 42/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 43/95
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 44/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 45/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 46/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 47/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 48/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 49/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 50/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 51/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 52/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 53/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 54/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 55/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 56/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 57/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 58/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 59/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 60/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 61/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 62/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 63/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 64/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 65/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 66/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 67/95
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 68/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 69/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 70/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 71/95
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 72/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 73/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 74/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 75/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 76/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 77/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 78/95
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 79/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 80/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 81/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 82/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 83/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 84/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 85/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 86/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 87/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 88/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 89/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 90/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 91/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 92/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0051 Acc: 0.9077

Epoch 93/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 94/95
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 95/95
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Training complete in 3m 44s
Best val Acc: 0.907692

---Testing---
Test accuracy: 0.967816
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 92 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.95916569768019, std: 0.02800059666980162
--------------------

run info[val: 0.2, epoch: 99, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0176 Acc: 0.3621
val Loss: 0.0180 Acc: 0.3793

Epoch 1/98
----------
train Loss: 0.0142 Acc: 0.5287
val Loss: 0.0129 Acc: 0.5632

Epoch 2/98
----------
train Loss: 0.0103 Acc: 0.6437
val Loss: 0.0098 Acc: 0.6437

Epoch 3/98
----------
train Loss: 0.0074 Acc: 0.7385
val Loss: 0.0079 Acc: 0.7586

Epoch 4/98
----------
train Loss: 0.0069 Acc: 0.8075
val Loss: 0.0069 Acc: 0.7586

Epoch 5/98
----------
train Loss: 0.0055 Acc: 0.8477
val Loss: 0.0062 Acc: 0.8621

Epoch 6/98
----------
train Loss: 0.0050 Acc: 0.8448
val Loss: 0.0059 Acc: 0.8736

Epoch 7/98
----------
train Loss: 0.0043 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8736

Epoch 8/98
----------
train Loss: 0.0039 Acc: 0.8822
val Loss: 0.0061 Acc: 0.8621

Epoch 9/98
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0053 Acc: 0.8736

Epoch 10/98
----------
LR is set to 0.001
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8506

Epoch 11/98
----------
train Loss: 0.0032 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 12/98
----------
train Loss: 0.0027 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8736

Epoch 13/98
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0056 Acc: 0.8621

Epoch 14/98
----------
train Loss: 0.0031 Acc: 0.9109
val Loss: 0.0056 Acc: 0.8621

Epoch 15/98
----------
train Loss: 0.0028 Acc: 0.9339
val Loss: 0.0055 Acc: 0.8736

Epoch 16/98
----------
train Loss: 0.0029 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8736

Epoch 17/98
----------
train Loss: 0.0031 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 18/98
----------
train Loss: 0.0029 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 19/98
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8621

Epoch 20/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0027 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8621

Epoch 21/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8621

Epoch 22/98
----------
train Loss: 0.0028 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8621

Epoch 23/98
----------
train Loss: 0.0030 Acc: 0.9282
val Loss: 0.0054 Acc: 0.8506

Epoch 24/98
----------
train Loss: 0.0030 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8506

Epoch 25/98
----------
train Loss: 0.0030 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 26/98
----------
train Loss: 0.0030 Acc: 0.9253
val Loss: 0.0054 Acc: 0.8736

Epoch 27/98
----------
train Loss: 0.0027 Acc: 0.9425
val Loss: 0.0054 Acc: 0.8621

Epoch 28/98
----------
train Loss: 0.0030 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 29/98
----------
train Loss: 0.0026 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8736

Epoch 30/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0028 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8736

Epoch 31/98
----------
train Loss: 0.0032 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8736

Epoch 32/98
----------
train Loss: 0.0029 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8736

Epoch 33/98
----------
train Loss: 0.0032 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Epoch 34/98
----------
train Loss: 0.0027 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8621

Epoch 35/98
----------
train Loss: 0.0027 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8736

Epoch 36/98
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 37/98
----------
train Loss: 0.0029 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 38/98
----------
train Loss: 0.0034 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8621

Epoch 39/98
----------
train Loss: 0.0028 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8736

Epoch 40/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0035 Acc: 0.9052
val Loss: 0.0053 Acc: 0.8736

Epoch 41/98
----------
train Loss: 0.0028 Acc: 0.9282
val Loss: 0.0054 Acc: 0.8621

Epoch 42/98
----------
train Loss: 0.0030 Acc: 0.9511
val Loss: 0.0053 Acc: 0.8736

Epoch 43/98
----------
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0054 Acc: 0.8736

Epoch 44/98
----------
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 45/98
----------
train Loss: 0.0031 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 46/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8736

Epoch 47/98
----------
train Loss: 0.0029 Acc: 0.9253
val Loss: 0.0054 Acc: 0.8736

Epoch 48/98
----------
train Loss: 0.0029 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 49/98
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 50/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8736

Epoch 51/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8621

Epoch 52/98
----------
train Loss: 0.0030 Acc: 0.9138
val Loss: 0.0054 Acc: 0.8621

Epoch 53/98
----------
train Loss: 0.0033 Acc: 0.9224
val Loss: 0.0054 Acc: 0.8621

Epoch 54/98
----------
train Loss: 0.0028 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8506

Epoch 55/98
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0054 Acc: 0.8621

Epoch 56/98
----------
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 57/98
----------
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 58/98
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8736

Epoch 59/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0053 Acc: 0.8736

Epoch 60/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8736

Epoch 61/98
----------
train Loss: 0.0026 Acc: 0.9483
val Loss: 0.0054 Acc: 0.8621

Epoch 62/98
----------
train Loss: 0.0027 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8736

Epoch 63/98
----------
train Loss: 0.0028 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 64/98
----------
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 65/98
----------
train Loss: 0.0032 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8736

Epoch 66/98
----------
train Loss: 0.0033 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 67/98
----------
train Loss: 0.0031 Acc: 0.9253
val Loss: 0.0054 Acc: 0.8621

Epoch 68/98
----------
train Loss: 0.0033 Acc: 0.9167
val Loss: 0.0054 Acc: 0.8621

Epoch 69/98
----------
train Loss: 0.0029 Acc: 0.9253
val Loss: 0.0054 Acc: 0.8621

Epoch 70/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 71/98
----------
train Loss: 0.0031 Acc: 0.9253
val Loss: 0.0054 Acc: 0.8621

Epoch 72/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8506

Epoch 73/98
----------
train Loss: 0.0032 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8621

Epoch 74/98
----------
train Loss: 0.0031 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8506

Epoch 75/98
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8506

Epoch 76/98
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8736

Epoch 77/98
----------
train Loss: 0.0031 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8736

Epoch 78/98
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8736

Epoch 79/98
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8736

Epoch 80/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0029 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8621

Epoch 81/98
----------
train Loss: 0.0033 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8621

Epoch 82/98
----------
train Loss: 0.0028 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8621

Epoch 83/98
----------
train Loss: 0.0032 Acc: 0.9167
val Loss: 0.0053 Acc: 0.8621

Epoch 84/98
----------
train Loss: 0.0033 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8621

Epoch 85/98
----------
train Loss: 0.0032 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 86/98
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0053 Acc: 0.8621

Epoch 87/98
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0054 Acc: 0.8621

Epoch 88/98
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0054 Acc: 0.8506

Epoch 89/98
----------
train Loss: 0.0028 Acc: 0.9368
val Loss: 0.0054 Acc: 0.8506

Epoch 90/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0032 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8621

Epoch 91/98
----------
train Loss: 0.0027 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 92/98
----------
train Loss: 0.0026 Acc: 0.9454
val Loss: 0.0054 Acc: 0.8621

Epoch 93/98
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8736

Epoch 94/98
----------
train Loss: 0.0030 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 95/98
----------
train Loss: 0.0029 Acc: 0.9195
val Loss: 0.0054 Acc: 0.8621

Epoch 96/98
----------
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0054 Acc: 0.8621

Epoch 97/98
----------
train Loss: 0.0033 Acc: 0.9310
val Loss: 0.0054 Acc: 0.8621

Epoch 98/98
----------
train Loss: 0.0028 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8621

Training complete in 3m 35s
Best val Acc: 0.873563

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0038 Acc: 0.8851
val Loss: 0.0052 Acc: 0.8621

Epoch 1/98
----------
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0057 Acc: 0.8506

Epoch 2/98
----------
train Loss: 0.0018 Acc: 0.9655
val Loss: 0.0050 Acc: 0.8851

Epoch 3/98
----------
train Loss: 0.0008 Acc: 0.9943
val Loss: 0.0046 Acc: 0.8851

Epoch 4/98
----------
train Loss: 0.0005 Acc: 0.9943
val Loss: 0.0044 Acc: 0.9195

Epoch 5/98
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0044 Acc: 0.9080

Epoch 6/98
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0044 Acc: 0.9195

Epoch 7/98
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9195

Epoch 8/98
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0044 Acc: 0.9080

Epoch 9/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8966

Epoch 10/98
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 11/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 12/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 13/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 14/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 15/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 16/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 17/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 18/98
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0042 Acc: 0.9195

Epoch 19/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 20/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 21/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 22/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 23/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 24/98
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0040 Acc: 0.9195

Epoch 25/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 26/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 27/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 28/98
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0040 Acc: 0.9195

Epoch 29/98
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0041 Acc: 0.9195

Epoch 30/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 31/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 32/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 33/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 34/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 35/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 36/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 37/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 38/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 39/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 40/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 41/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 42/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 43/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 44/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 45/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 46/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 47/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 48/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 49/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 50/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 51/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 52/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 53/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 54/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 55/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 56/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 57/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 58/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 59/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 60/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 61/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 62/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 63/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 64/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 65/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 66/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 67/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 68/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 69/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 70/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 71/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 72/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 73/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 74/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 75/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 76/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 77/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 78/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 79/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 80/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 81/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 82/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 83/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 84/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 85/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 86/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 87/98
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0041 Acc: 0.9195

Epoch 88/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 89/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 90/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 91/98
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0040 Acc: 0.9195

Epoch 92/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 93/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 94/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 95/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 96/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 97/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 98/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Training complete in 4m 0s
Best val Acc: 0.919540

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.978096037578467, std: 0.008429719905853828
--------------------

run info[val: 0.25, epoch: 94, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0191 Acc: 0.3394
val Loss: 0.0236 Acc: 0.4074

Epoch 1/93
----------
train Loss: 0.0145 Acc: 0.5443
val Loss: 0.0210 Acc: 0.6389

Epoch 2/93
----------
train Loss: 0.0117 Acc: 0.7003
val Loss: 0.0209 Acc: 0.6852

Epoch 3/93
----------
train Loss: 0.0096 Acc: 0.6728
val Loss: 0.0176 Acc: 0.6481

Epoch 4/93
----------
train Loss: 0.0080 Acc: 0.7309
val Loss: 0.0110 Acc: 0.8241

Epoch 5/93
----------
train Loss: 0.0064 Acc: 0.8410
val Loss: 0.0106 Acc: 0.7963

Epoch 6/93
----------
train Loss: 0.0046 Acc: 0.8960
val Loss: 0.0121 Acc: 0.8056

Epoch 7/93
----------
train Loss: 0.0040 Acc: 0.9174
val Loss: 0.0106 Acc: 0.8333

Epoch 8/93
----------
train Loss: 0.0038 Acc: 0.9144
val Loss: 0.0084 Acc: 0.7963

Epoch 9/93
----------
train Loss: 0.0031 Acc: 0.9388
val Loss: 0.0149 Acc: 0.8796

Epoch 10/93
----------
train Loss: 0.0033 Acc: 0.9174
val Loss: 0.0046 Acc: 0.8611

Epoch 11/93
----------
train Loss: 0.0028 Acc: 0.9664
val Loss: 0.0125 Acc: 0.8704

Epoch 12/93
----------
train Loss: 0.0031 Acc: 0.9235
val Loss: 0.0066 Acc: 0.8704

Epoch 13/93
----------
LR is set to 0.001
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0054 Acc: 0.8333

Epoch 14/93
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0095 Acc: 0.8426

Epoch 15/93
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0059 Acc: 0.8519

Epoch 16/93
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0056 Acc: 0.8611

Epoch 17/93
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0155 Acc: 0.8796

Epoch 18/93
----------
train Loss: 0.0025 Acc: 0.9664
val Loss: 0.0069 Acc: 0.8889

Epoch 19/93
----------
train Loss: 0.0021 Acc: 0.9633
val Loss: 0.0077 Acc: 0.8796

Epoch 20/93
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0060 Acc: 0.8796

Epoch 21/93
----------
train Loss: 0.0019 Acc: 0.9633
val Loss: 0.0090 Acc: 0.8796

Epoch 22/93
----------
train Loss: 0.0025 Acc: 0.9602
val Loss: 0.0080 Acc: 0.8704

Epoch 23/93
----------
train Loss: 0.0022 Acc: 0.9694
val Loss: 0.0065 Acc: 0.8796

Epoch 24/93
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0087 Acc: 0.8796

Epoch 25/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0069 Acc: 0.8611

Epoch 26/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0100 Acc: 0.8611

Epoch 27/93
----------
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0102 Acc: 0.8704

Epoch 28/93
----------
train Loss: 0.0020 Acc: 0.9786
val Loss: 0.0074 Acc: 0.8704

Epoch 29/93
----------
train Loss: 0.0019 Acc: 0.9725
val Loss: 0.0050 Acc: 0.8704

Epoch 30/93
----------
train Loss: 0.0021 Acc: 0.9755
val Loss: 0.0056 Acc: 0.8704

Epoch 31/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0051 Acc: 0.8704

Epoch 32/93
----------
train Loss: 0.0022 Acc: 0.9694
val Loss: 0.0069 Acc: 0.8704

Epoch 33/93
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0054 Acc: 0.8704

Epoch 34/93
----------
train Loss: 0.0018 Acc: 0.9817
val Loss: 0.0050 Acc: 0.8704

Epoch 35/93
----------
train Loss: 0.0019 Acc: 0.9786
val Loss: 0.0069 Acc: 0.8704

Epoch 36/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0060 Acc: 0.8704

Epoch 37/93
----------
train Loss: 0.0020 Acc: 0.9725
val Loss: 0.0058 Acc: 0.8704

Epoch 38/93
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0063 Acc: 0.8796

Epoch 39/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0061 Acc: 0.8796

Epoch 40/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0100 Acc: 0.8796

Epoch 41/93
----------
train Loss: 0.0024 Acc: 0.9664
val Loss: 0.0086 Acc: 0.8704

Epoch 42/93
----------
train Loss: 0.0026 Acc: 0.9664
val Loss: 0.0063 Acc: 0.8796

Epoch 43/93
----------
train Loss: 0.0020 Acc: 0.9725
val Loss: 0.0098 Acc: 0.8889

Epoch 44/93
----------
train Loss: 0.0024 Acc: 0.9694
val Loss: 0.0072 Acc: 0.8796

Epoch 45/93
----------
train Loss: 0.0018 Acc: 0.9817
val Loss: 0.0068 Acc: 0.8796

Epoch 46/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0097 Acc: 0.8796

Epoch 47/93
----------
train Loss: 0.0021 Acc: 0.9817
val Loss: 0.0053 Acc: 0.8796

Epoch 48/93
----------
train Loss: 0.0019 Acc: 0.9786
val Loss: 0.0052 Acc: 0.8796

Epoch 49/93
----------
train Loss: 0.0021 Acc: 0.9786
val Loss: 0.0117 Acc: 0.8796

Epoch 50/93
----------
train Loss: 0.0021 Acc: 0.9786
val Loss: 0.0112 Acc: 0.8796

Epoch 51/93
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0049 Acc: 0.8796

Epoch 52/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0021 Acc: 0.9755
val Loss: 0.0053 Acc: 0.8796

Epoch 53/93
----------
train Loss: 0.0019 Acc: 0.9817
val Loss: 0.0102 Acc: 0.8796

Epoch 54/93
----------
train Loss: 0.0029 Acc: 0.9511
val Loss: 0.0078 Acc: 0.8796

Epoch 55/93
----------
train Loss: 0.0023 Acc: 0.9786
val Loss: 0.0093 Acc: 0.8796

Epoch 56/93
----------
train Loss: 0.0020 Acc: 0.9755
val Loss: 0.0061 Acc: 0.8796

Epoch 57/93
----------
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0057 Acc: 0.8796

Epoch 58/93
----------
train Loss: 0.0020 Acc: 0.9755
val Loss: 0.0060 Acc: 0.8796

Epoch 59/93
----------
train Loss: 0.0022 Acc: 0.9633
val Loss: 0.0073 Acc: 0.8796

Epoch 60/93
----------
train Loss: 0.0020 Acc: 0.9633
val Loss: 0.0085 Acc: 0.8796

Epoch 61/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0054 Acc: 0.8889

Epoch 62/93
----------
train Loss: 0.0020 Acc: 0.9694
val Loss: 0.0109 Acc: 0.8889

Epoch 63/93
----------
train Loss: 0.0019 Acc: 0.9847
val Loss: 0.0079 Acc: 0.8889

Epoch 64/93
----------
train Loss: 0.0017 Acc: 0.9786
val Loss: 0.0110 Acc: 0.8889

Epoch 65/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0108 Acc: 0.8889

Epoch 66/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0094 Acc: 0.8889

Epoch 67/93
----------
train Loss: 0.0022 Acc: 0.9694
val Loss: 0.0115 Acc: 0.8889

Epoch 68/93
----------
train Loss: 0.0025 Acc: 0.9664
val Loss: 0.0133 Acc: 0.8796

Epoch 69/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0061 Acc: 0.8704

Epoch 70/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0069 Acc: 0.8796

Epoch 71/93
----------
train Loss: 0.0021 Acc: 0.9725
val Loss: 0.0057 Acc: 0.8796

Epoch 72/93
----------
train Loss: 0.0023 Acc: 0.9694
val Loss: 0.0072 Acc: 0.8611

Epoch 73/93
----------
train Loss: 0.0022 Acc: 0.9694
val Loss: 0.0137 Acc: 0.8704

Epoch 74/93
----------
train Loss: 0.0023 Acc: 0.9694
val Loss: 0.0062 Acc: 0.8889

Epoch 75/93
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0063 Acc: 0.8704

Epoch 76/93
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0110 Acc: 0.8704

Epoch 77/93
----------
train Loss: 0.0021 Acc: 0.9725
val Loss: 0.0060 Acc: 0.8796

Epoch 78/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0110 Acc: 0.8796

Epoch 79/93
----------
train Loss: 0.0021 Acc: 0.9817
val Loss: 0.0133 Acc: 0.8796

Epoch 80/93
----------
train Loss: 0.0020 Acc: 0.9633
val Loss: 0.0067 Acc: 0.8796

Epoch 81/93
----------
train Loss: 0.0021 Acc: 0.9755
val Loss: 0.0066 Acc: 0.8796

Epoch 82/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0117 Acc: 0.8796

Epoch 83/93
----------
train Loss: 0.0020 Acc: 0.9725
val Loss: 0.0062 Acc: 0.8796

Epoch 84/93
----------
train Loss: 0.0023 Acc: 0.9725
val Loss: 0.0063 Acc: 0.8796

Epoch 85/93
----------
train Loss: 0.0024 Acc: 0.9725
val Loss: 0.0062 Acc: 0.8796

Epoch 86/93
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0082 Acc: 0.8704

Epoch 87/93
----------
train Loss: 0.0022 Acc: 0.9694
val Loss: 0.0077 Acc: 0.8796

Epoch 88/93
----------
train Loss: 0.0018 Acc: 0.9817
val Loss: 0.0078 Acc: 0.8796

Epoch 89/93
----------
train Loss: 0.0019 Acc: 0.9694
val Loss: 0.0060 Acc: 0.8796

Epoch 90/93
----------
train Loss: 0.0020 Acc: 0.9694
val Loss: 0.0067 Acc: 0.8796

Epoch 91/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0025 Acc: 0.9664
val Loss: 0.0053 Acc: 0.8796

Epoch 92/93
----------
train Loss: 0.0020 Acc: 0.9786
val Loss: 0.0080 Acc: 0.8796

Epoch 93/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0095 Acc: 0.8796

Training complete in 3m 32s
Best val Acc: 0.888889

---Fine tuning.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0035 Acc: 0.9450
val Loss: 0.0073 Acc: 0.8704

Epoch 1/93
----------
train Loss: 0.0026 Acc: 0.9694
val Loss: 0.0294 Acc: 0.7130

Epoch 2/93
----------
train Loss: 0.0011 Acc: 0.9908
val Loss: 0.0114 Acc: 0.8148

Epoch 3/93
----------
train Loss: 0.0018 Acc: 0.9725
val Loss: 0.0069 Acc: 0.8611

Epoch 4/93
----------
train Loss: 0.0003 Acc: 0.9969
val Loss: 0.0071 Acc: 0.8426

Epoch 5/93
----------
train Loss: 0.0004 Acc: 0.9939
val Loss: 0.0124 Acc: 0.8426

Epoch 6/93
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0077 Acc: 0.8704

Epoch 7/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8889

Epoch 8/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 9/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 10/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 11/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Epoch 12/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 13/93
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0226 Acc: 0.8981

Epoch 14/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 15/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 16/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Epoch 17/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 18/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8981

Epoch 19/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8981

Epoch 20/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 21/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8981

Epoch 22/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 23/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8981

Epoch 24/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8981

Epoch 25/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8981

Epoch 26/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0084 Acc: 0.9074

Epoch 27/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.9074

Epoch 28/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9074

Epoch 29/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8981

Epoch 30/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 31/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 32/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8981

Epoch 33/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8981

Epoch 34/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 35/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0143 Acc: 0.8981

Epoch 36/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 37/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 38/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 39/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 40/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8981

Epoch 41/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 42/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8981

Epoch 43/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 44/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0085 Acc: 0.8981

Epoch 45/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.9074

Epoch 46/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8981

Epoch 47/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.9074

Epoch 48/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 49/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 50/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8981

Epoch 51/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 52/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 53/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 54/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8981

Epoch 55/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 56/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 57/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8981

Epoch 58/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 59/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 60/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.9074

Epoch 61/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8981

Epoch 62/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8981

Epoch 63/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8981

Epoch 64/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8981

Epoch 65/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8981

Epoch 66/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 67/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.9074

Epoch 68/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 69/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Epoch 70/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8981

Epoch 71/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8981

Epoch 72/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 73/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 74/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 75/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 76/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.9074

Epoch 77/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.9074

Epoch 78/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9074

Epoch 79/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.9074

Epoch 80/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.9074

Epoch 81/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.9074

Epoch 82/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 83/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9074

Epoch 84/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8981

Epoch 85/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 86/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8981

Epoch 87/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 88/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 89/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8981

Epoch 90/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 91/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.9074

Epoch 92/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9074

Epoch 93/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8981

Training complete in 3m 49s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 97 %
mean: 0.9755593254277196, std: 0.007126773658415128
--------------------

run info[val: 0.3, epoch: 84, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0225 Acc: 0.3115
val Loss: 0.0251 Acc: 0.2308

Epoch 1/83
----------
train Loss: 0.0215 Acc: 0.3475
val Loss: 0.0215 Acc: 0.4154

Epoch 2/83
----------
train Loss: 0.0163 Acc: 0.5377
val Loss: 0.0127 Acc: 0.7154

Epoch 3/83
----------
LR is set to 0.001
train Loss: 0.0114 Acc: 0.7311
val Loss: 0.0132 Acc: 0.6923

Epoch 4/83
----------
train Loss: 0.0108 Acc: 0.7344
val Loss: 0.0141 Acc: 0.6692

Epoch 5/83
----------
train Loss: 0.0089 Acc: 0.7672
val Loss: 0.0132 Acc: 0.6769

Epoch 6/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0088 Acc: 0.7574
val Loss: 0.0120 Acc: 0.6769

Epoch 7/83
----------
train Loss: 0.0085 Acc: 0.7836
val Loss: 0.0134 Acc: 0.6769

Epoch 8/83
----------
train Loss: 0.0109 Acc: 0.7639
val Loss: 0.0136 Acc: 0.6846

Epoch 9/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0087 Acc: 0.7508
val Loss: 0.0119 Acc: 0.6692

Epoch 10/83
----------
train Loss: 0.0084 Acc: 0.7541
val Loss: 0.0134 Acc: 0.6692

Epoch 11/83
----------
train Loss: 0.0096 Acc: 0.7475
val Loss: 0.0133 Acc: 0.6769

Epoch 12/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0085 Acc: 0.7475
val Loss: 0.0126 Acc: 0.6615

Epoch 13/83
----------
train Loss: 0.0083 Acc: 0.7803
val Loss: 0.0135 Acc: 0.6692

Epoch 14/83
----------
train Loss: 0.0086 Acc: 0.7607
val Loss: 0.0126 Acc: 0.7000

Epoch 15/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0076 Acc: 0.7672
val Loss: 0.0117 Acc: 0.6769

Epoch 16/83
----------
train Loss: 0.0123 Acc: 0.7410
val Loss: 0.0130 Acc: 0.6692

Epoch 17/83
----------
train Loss: 0.0098 Acc: 0.7639
val Loss: 0.0121 Acc: 0.6538

Epoch 18/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0115 Acc: 0.7672
val Loss: 0.0137 Acc: 0.6615

Epoch 19/83
----------
train Loss: 0.0081 Acc: 0.7672
val Loss: 0.0138 Acc: 0.6615

Epoch 20/83
----------
train Loss: 0.0092 Acc: 0.7410
val Loss: 0.0128 Acc: 0.6615

Epoch 21/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0091 Acc: 0.7803
val Loss: 0.0119 Acc: 0.6692

Epoch 22/83
----------
train Loss: 0.0105 Acc: 0.7607
val Loss: 0.0136 Acc: 0.6769

Epoch 23/83
----------
train Loss: 0.0084 Acc: 0.7869
val Loss: 0.0141 Acc: 0.6692

Epoch 24/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0094 Acc: 0.7574
val Loss: 0.0142 Acc: 0.6846

Epoch 25/83
----------
train Loss: 0.0095 Acc: 0.7344
val Loss: 0.0138 Acc: 0.6615

Epoch 26/83
----------
train Loss: 0.0089 Acc: 0.7639
val Loss: 0.0129 Acc: 0.6615

Epoch 27/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0083 Acc: 0.7705
val Loss: 0.0121 Acc: 0.6538

Epoch 28/83
----------
train Loss: 0.0088 Acc: 0.7541
val Loss: 0.0133 Acc: 0.6615

Epoch 29/83
----------
train Loss: 0.0085 Acc: 0.7475
val Loss: 0.0131 Acc: 0.6692

Epoch 30/83
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0082 Acc: 0.7541
val Loss: 0.0125 Acc: 0.6692

Epoch 31/83
----------
train Loss: 0.0098 Acc: 0.7574
val Loss: 0.0136 Acc: 0.7154

Epoch 32/83
----------
train Loss: 0.0093 Acc: 0.7705
val Loss: 0.0129 Acc: 0.7000

Epoch 33/83
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0091 Acc: 0.7541
val Loss: 0.0114 Acc: 0.6769

Epoch 34/83
----------
train Loss: 0.0088 Acc: 0.7672
val Loss: 0.0131 Acc: 0.6846

Epoch 35/83
----------
train Loss: 0.0107 Acc: 0.7639
val Loss: 0.0134 Acc: 0.7000

Epoch 36/83
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0099 Acc: 0.7705
val Loss: 0.0117 Acc: 0.6923

Epoch 37/83
----------
train Loss: 0.0104 Acc: 0.7508
val Loss: 0.0123 Acc: 0.7000

Epoch 38/83
----------
train Loss: 0.0106 Acc: 0.7475
val Loss: 0.0133 Acc: 0.6769

Epoch 39/83
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0088 Acc: 0.7541
val Loss: 0.0125 Acc: 0.6538

Epoch 40/83
----------
train Loss: 0.0101 Acc: 0.7574
val Loss: 0.0130 Acc: 0.6692

Epoch 41/83
----------
train Loss: 0.0094 Acc: 0.7443
val Loss: 0.0129 Acc: 0.6615

Epoch 42/83
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0096 Acc: 0.7508
val Loss: 0.0121 Acc: 0.6615

Epoch 43/83
----------
train Loss: 0.0082 Acc: 0.7705
val Loss: 0.0132 Acc: 0.6615

Epoch 44/83
----------
train Loss: 0.0106 Acc: 0.7672
val Loss: 0.0132 Acc: 0.6462

Epoch 45/83
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0085 Acc: 0.7508
val Loss: 0.0148 Acc: 0.6692

Epoch 46/83
----------
train Loss: 0.0081 Acc: 0.7607
val Loss: 0.0118 Acc: 0.6692

Epoch 47/83
----------
train Loss: 0.0094 Acc: 0.7574
val Loss: 0.0127 Acc: 0.6615

Epoch 48/83
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0086 Acc: 0.7607
val Loss: 0.0123 Acc: 0.6615

Epoch 49/83
----------
train Loss: 0.0092 Acc: 0.7738
val Loss: 0.0127 Acc: 0.6615

Epoch 50/83
----------
train Loss: 0.0091 Acc: 0.7508
val Loss: 0.0123 Acc: 0.6538

Epoch 51/83
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0076 Acc: 0.7377
val Loss: 0.0134 Acc: 0.6769

Epoch 52/83
----------
train Loss: 0.0086 Acc: 0.7541
val Loss: 0.0125 Acc: 0.6846

Epoch 53/83
----------
train Loss: 0.0110 Acc: 0.7443
val Loss: 0.0124 Acc: 0.6538

Epoch 54/83
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0105 Acc: 0.7705
val Loss: 0.0116 Acc: 0.6538

Epoch 55/83
----------
train Loss: 0.0077 Acc: 0.7672
val Loss: 0.0122 Acc: 0.6615

Epoch 56/83
----------
train Loss: 0.0106 Acc: 0.7475
val Loss: 0.0122 Acc: 0.6615

Epoch 57/83
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0075 Acc: 0.7770
val Loss: 0.0117 Acc: 0.6615

Epoch 58/83
----------
train Loss: 0.0086 Acc: 0.7541
val Loss: 0.0132 Acc: 0.6615

Epoch 59/83
----------
train Loss: 0.0094 Acc: 0.7246
val Loss: 0.0133 Acc: 0.6462

Epoch 60/83
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0104 Acc: 0.7475
val Loss: 0.0124 Acc: 0.6538

Epoch 61/83
----------
train Loss: 0.0088 Acc: 0.7574
val Loss: 0.0131 Acc: 0.6692

Epoch 62/83
----------
train Loss: 0.0082 Acc: 0.7475
val Loss: 0.0135 Acc: 0.6538

Epoch 63/83
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0081 Acc: 0.7607
val Loss: 0.0134 Acc: 0.6538

Epoch 64/83
----------
train Loss: 0.0076 Acc: 0.7836
val Loss: 0.0117 Acc: 0.6692

Epoch 65/83
----------
train Loss: 0.0090 Acc: 0.7672
val Loss: 0.0132 Acc: 0.6615

Epoch 66/83
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0102 Acc: 0.7672
val Loss: 0.0133 Acc: 0.6615

Epoch 67/83
----------
train Loss: 0.0085 Acc: 0.7672
val Loss: 0.0135 Acc: 0.6615

Epoch 68/83
----------
train Loss: 0.0096 Acc: 0.7377
val Loss: 0.0137 Acc: 0.6615

Epoch 69/83
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0091 Acc: 0.7443
val Loss: 0.0124 Acc: 0.6615

Epoch 70/83
----------
train Loss: 0.0087 Acc: 0.7639
val Loss: 0.0142 Acc: 0.6692

Epoch 71/83
----------
train Loss: 0.0083 Acc: 0.7541
val Loss: 0.0123 Acc: 0.6769

Epoch 72/83
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0087 Acc: 0.7377
val Loss: 0.0136 Acc: 0.6538

Epoch 73/83
----------
train Loss: 0.0104 Acc: 0.7475
val Loss: 0.0128 Acc: 0.6692

Epoch 74/83
----------
train Loss: 0.0086 Acc: 0.7574
val Loss: 0.0132 Acc: 0.6692

Epoch 75/83
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0073 Acc: 0.7541
val Loss: 0.0143 Acc: 0.6615

Epoch 76/83
----------
train Loss: 0.0075 Acc: 0.7541
val Loss: 0.0148 Acc: 0.6846

Epoch 77/83
----------
train Loss: 0.0091 Acc: 0.7672
val Loss: 0.0124 Acc: 0.6769

Epoch 78/83
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0104 Acc: 0.7607
val Loss: 0.0129 Acc: 0.6615

Epoch 79/83
----------
train Loss: 0.0091 Acc: 0.7377
val Loss: 0.0138 Acc: 0.6692

Epoch 80/83
----------
train Loss: 0.0107 Acc: 0.7705
val Loss: 0.0134 Acc: 0.6615

Epoch 81/83
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0088 Acc: 0.7443
val Loss: 0.0131 Acc: 0.6538

Epoch 82/83
----------
train Loss: 0.0084 Acc: 0.7541
val Loss: 0.0135 Acc: 0.6462

Epoch 83/83
----------
train Loss: 0.0095 Acc: 0.7770
val Loss: 0.0125 Acc: 0.6692

Training complete in 3m 9s
Best val Acc: 0.715385

---Fine tuning.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0115 Acc: 0.6525
val Loss: 0.0119 Acc: 0.6923

Epoch 1/83
----------
train Loss: 0.0108 Acc: 0.8164
val Loss: 0.0169 Acc: 0.5462

Epoch 2/83
----------
train Loss: 0.0133 Acc: 0.5967
val Loss: 0.0080 Acc: 0.7923

Epoch 3/83
----------
LR is set to 0.001
train Loss: 0.0032 Acc: 0.9180
val Loss: 0.0068 Acc: 0.7846

Epoch 4/83
----------
train Loss: 0.0047 Acc: 0.8885
val Loss: 0.0085 Acc: 0.8154

Epoch 5/83
----------
train Loss: 0.0036 Acc: 0.9148
val Loss: 0.0067 Acc: 0.8231

Epoch 6/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0027 Acc: 0.9475
val Loss: 0.0070 Acc: 0.8231

Epoch 7/83
----------
train Loss: 0.0029 Acc: 0.9246
val Loss: 0.0061 Acc: 0.8385

Epoch 8/83
----------
train Loss: 0.0032 Acc: 0.9508
val Loss: 0.0081 Acc: 0.8077

Epoch 9/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0077 Acc: 0.8231

Epoch 10/83
----------
train Loss: 0.0039 Acc: 0.9443
val Loss: 0.0097 Acc: 0.8000

Epoch 11/83
----------
train Loss: 0.0025 Acc: 0.9508
val Loss: 0.0086 Acc: 0.8077

Epoch 12/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9672
val Loss: 0.0066 Acc: 0.8000

Epoch 13/83
----------
train Loss: 0.0034 Acc: 0.9443
val Loss: 0.0077 Acc: 0.7923

Epoch 14/83
----------
train Loss: 0.0039 Acc: 0.9574
val Loss: 0.0086 Acc: 0.8154

Epoch 15/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0020 Acc: 0.9541
val Loss: 0.0072 Acc: 0.8231

Epoch 16/83
----------
train Loss: 0.0025 Acc: 0.9377
val Loss: 0.0078 Acc: 0.8231

Epoch 17/83
----------
train Loss: 0.0021 Acc: 0.9475
val Loss: 0.0070 Acc: 0.8077

Epoch 18/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0025 Acc: 0.9574
val Loss: 0.0085 Acc: 0.8154

Epoch 19/83
----------
train Loss: 0.0042 Acc: 0.9410
val Loss: 0.0089 Acc: 0.8077

Epoch 20/83
----------
train Loss: 0.0028 Acc: 0.9443
val Loss: 0.0075 Acc: 0.8077

Epoch 21/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0036 Acc: 0.9607
val Loss: 0.0093 Acc: 0.7923

Epoch 22/83
----------
train Loss: 0.0019 Acc: 0.9475
val Loss: 0.0109 Acc: 0.8000

Epoch 23/83
----------
train Loss: 0.0037 Acc: 0.9410
val Loss: 0.0074 Acc: 0.8231

Epoch 24/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0032 Acc: 0.9639
val Loss: 0.0077 Acc: 0.8231

Epoch 25/83
----------
train Loss: 0.0031 Acc: 0.9443
val Loss: 0.0087 Acc: 0.8308

Epoch 26/83
----------
train Loss: 0.0030 Acc: 0.9475
val Loss: 0.0072 Acc: 0.8308

Epoch 27/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0018 Acc: 0.9541
val Loss: 0.0082 Acc: 0.8308

Epoch 28/83
----------
train Loss: 0.0025 Acc: 0.9443
val Loss: 0.0063 Acc: 0.8154

Epoch 29/83
----------
train Loss: 0.0056 Acc: 0.9541
val Loss: 0.0075 Acc: 0.8154

Epoch 30/83
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0029 Acc: 0.9574
val Loss: 0.0067 Acc: 0.8154

Epoch 31/83
----------
train Loss: 0.0026 Acc: 0.9508
val Loss: 0.0075 Acc: 0.8154

Epoch 32/83
----------
train Loss: 0.0019 Acc: 0.9574
val Loss: 0.0063 Acc: 0.8154

Epoch 33/83
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0029 Acc: 0.9443
val Loss: 0.0077 Acc: 0.8000

Epoch 34/83
----------
train Loss: 0.0063 Acc: 0.9344
val Loss: 0.0078 Acc: 0.7923

Epoch 35/83
----------
train Loss: 0.0024 Acc: 0.9574
val Loss: 0.0068 Acc: 0.8000

Epoch 36/83
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0018 Acc: 0.9574
val Loss: 0.0076 Acc: 0.8077

Epoch 37/83
----------
train Loss: 0.0060 Acc: 0.9475
val Loss: 0.0091 Acc: 0.8154

Epoch 38/83
----------
train Loss: 0.0025 Acc: 0.9574
val Loss: 0.0078 Acc: 0.8154

Epoch 39/83
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0024 Acc: 0.9508
val Loss: 0.0078 Acc: 0.8000

Epoch 40/83
----------
train Loss: 0.0047 Acc: 0.9344
val Loss: 0.0071 Acc: 0.8077

Epoch 41/83
----------
train Loss: 0.0030 Acc: 0.9541
val Loss: 0.0082 Acc: 0.8077

Epoch 42/83
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0018 Acc: 0.9607
val Loss: 0.0085 Acc: 0.8077

Epoch 43/83
----------
train Loss: 0.0037 Acc: 0.9475
val Loss: 0.0060 Acc: 0.8154

Epoch 44/83
----------
train Loss: 0.0044 Acc: 0.9344
val Loss: 0.0078 Acc: 0.8077

Epoch 45/83
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0026 Acc: 0.9574
val Loss: 0.0093 Acc: 0.8000

Epoch 46/83
----------
train Loss: 0.0024 Acc: 0.9443
val Loss: 0.0065 Acc: 0.8077

Epoch 47/83
----------
train Loss: 0.0052 Acc: 0.9475
val Loss: 0.0076 Acc: 0.8308

Epoch 48/83
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0019 Acc: 0.9672
val Loss: 0.0074 Acc: 0.8308

Epoch 49/83
----------
train Loss: 0.0046 Acc: 0.9443
val Loss: 0.0076 Acc: 0.8077

Epoch 50/83
----------
train Loss: 0.0025 Acc: 0.9475
val Loss: 0.0073 Acc: 0.8154

Epoch 51/83
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0026 Acc: 0.9443
val Loss: 0.0085 Acc: 0.8077

Epoch 52/83
----------
train Loss: 0.0025 Acc: 0.9508
val Loss: 0.0100 Acc: 0.8077

Epoch 53/83
----------
train Loss: 0.0024 Acc: 0.9410
val Loss: 0.0094 Acc: 0.8000

Epoch 54/83
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0023 Acc: 0.9574
val Loss: 0.0089 Acc: 0.8077

Epoch 55/83
----------
train Loss: 0.0020 Acc: 0.9607
val Loss: 0.0094 Acc: 0.8077

Epoch 56/83
----------
train Loss: 0.0023 Acc: 0.9508
val Loss: 0.0078 Acc: 0.8154

Epoch 57/83
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0027 Acc: 0.9508
val Loss: 0.0081 Acc: 0.8308

Epoch 58/83
----------
train Loss: 0.0044 Acc: 0.9410
val Loss: 0.0069 Acc: 0.8308

Epoch 59/83
----------
train Loss: 0.0019 Acc: 0.9541
val Loss: 0.0069 Acc: 0.8077

Epoch 60/83
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0072 Acc: 0.9475
val Loss: 0.0080 Acc: 0.8154

Epoch 61/83
----------
train Loss: 0.0024 Acc: 0.9574
val Loss: 0.0098 Acc: 0.8077

Epoch 62/83
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0095 Acc: 0.8231

Epoch 63/83
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0020 Acc: 0.9607
val Loss: 0.0071 Acc: 0.8000

Epoch 64/83
----------
train Loss: 0.0027 Acc: 0.9475
val Loss: 0.0075 Acc: 0.8231

Epoch 65/83
----------
train Loss: 0.0025 Acc: 0.9541
val Loss: 0.0085 Acc: 0.8154

Epoch 66/83
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0022 Acc: 0.9377
val Loss: 0.0073 Acc: 0.8000

Epoch 67/83
----------
train Loss: 0.0023 Acc: 0.9410
val Loss: 0.0075 Acc: 0.8154

Epoch 68/83
----------
train Loss: 0.0027 Acc: 0.9574
val Loss: 0.0071 Acc: 0.8077

Epoch 69/83
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0076 Acc: 0.8077

Epoch 70/83
----------
train Loss: 0.0025 Acc: 0.9508
val Loss: 0.0087 Acc: 0.8000

Epoch 71/83
----------
train Loss: 0.0029 Acc: 0.9410
val Loss: 0.0090 Acc: 0.8077

Epoch 72/83
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0031 Acc: 0.9508
val Loss: 0.0087 Acc: 0.8231

Epoch 73/83
----------
train Loss: 0.0026 Acc: 0.9475
val Loss: 0.0078 Acc: 0.8231

Epoch 74/83
----------
train Loss: 0.0026 Acc: 0.9443
val Loss: 0.0085 Acc: 0.8077

Epoch 75/83
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0030 Acc: 0.9639
val Loss: 0.0074 Acc: 0.8231

Epoch 76/83
----------
train Loss: 0.0024 Acc: 0.9508
val Loss: 0.0091 Acc: 0.8231

Epoch 77/83
----------
train Loss: 0.0046 Acc: 0.9410
val Loss: 0.0079 Acc: 0.8077

Epoch 78/83
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0042 Acc: 0.9475
val Loss: 0.0075 Acc: 0.7923

Epoch 79/83
----------
train Loss: 0.0021 Acc: 0.9475
val Loss: 0.0085 Acc: 0.8077

Epoch 80/83
----------
train Loss: 0.0034 Acc: 0.9311
val Loss: 0.0083 Acc: 0.8154

Epoch 81/83
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0033 Acc: 0.9475
val Loss: 0.0074 Acc: 0.8231

Epoch 82/83
----------
train Loss: 0.0091 Acc: 0.9508
val Loss: 0.0097 Acc: 0.8000

Epoch 83/83
----------
train Loss: 0.0023 Acc: 0.9705
val Loss: 0.0094 Acc: 0.8154

Training complete in 3m 27s
Best val Acc: 0.838462

---Testing---
Test accuracy: 0.917241
--------------------
Accuracy of Dasyatiformes : 72 %
Accuracy of Myliobatiformes : 93 %
Accuracy of Rajiformes : 75 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 99 %
mean: 0.8748159890000962, std: 0.11084978800288152

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.98]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 85, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0163 Acc: 0.3163
val Loss: 0.0331 Acc: 0.3953

Epoch 1/84
----------
train Loss: 0.0119 Acc: 0.5306
val Loss: 0.0220 Acc: 0.5814

Epoch 2/84
----------
train Loss: 0.0082 Acc: 0.7321
val Loss: 0.0178 Acc: 0.6977

Epoch 3/84
----------
train Loss: 0.0064 Acc: 0.7985
val Loss: 0.0148 Acc: 0.8140

Epoch 4/84
----------
train Loss: 0.0056 Acc: 0.8112
val Loss: 0.0145 Acc: 0.8372

Epoch 5/84
----------
train Loss: 0.0050 Acc: 0.8163
val Loss: 0.0147 Acc: 0.7674

Epoch 6/84
----------
train Loss: 0.0044 Acc: 0.8367
val Loss: 0.0129 Acc: 0.8372

Epoch 7/84
----------
train Loss: 0.0037 Acc: 0.8903
val Loss: 0.0137 Acc: 0.8140

Epoch 8/84
----------
train Loss: 0.0034 Acc: 0.8827
val Loss: 0.0137 Acc: 0.8140

Epoch 9/84
----------
LR is set to 0.001
train Loss: 0.0034 Acc: 0.9107
val Loss: 0.0135 Acc: 0.8140

Epoch 10/84
----------
train Loss: 0.0034 Acc: 0.9031
val Loss: 0.0134 Acc: 0.8140

Epoch 11/84
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0133 Acc: 0.8140

Epoch 12/84
----------
train Loss: 0.0034 Acc: 0.8954
val Loss: 0.0134 Acc: 0.8140

Epoch 13/84
----------
train Loss: 0.0031 Acc: 0.9158
val Loss: 0.0133 Acc: 0.8140

Epoch 14/84
----------
train Loss: 0.0029 Acc: 0.9209
val Loss: 0.0131 Acc: 0.8140

Epoch 15/84
----------
train Loss: 0.0031 Acc: 0.9107
val Loss: 0.0130 Acc: 0.8140

Epoch 16/84
----------
train Loss: 0.0029 Acc: 0.9082
val Loss: 0.0130 Acc: 0.8140

Epoch 17/84
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 18/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 19/84
----------
train Loss: 0.0028 Acc: 0.9133
val Loss: 0.0132 Acc: 0.8140

Epoch 20/84
----------
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 21/84
----------
train Loss: 0.0029 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 22/84
----------
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0131 Acc: 0.8140

Epoch 23/84
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 24/84
----------
train Loss: 0.0030 Acc: 0.9107
val Loss: 0.0131 Acc: 0.8140

Epoch 25/84
----------
train Loss: 0.0029 Acc: 0.9286
val Loss: 0.0131 Acc: 0.8140

Epoch 26/84
----------
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0130 Acc: 0.8140

Epoch 27/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0028 Acc: 0.9311
val Loss: 0.0131 Acc: 0.8140

Epoch 28/84
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0131 Acc: 0.8140

Epoch 29/84
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0130 Acc: 0.8140

Epoch 30/84
----------
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0131 Acc: 0.8140

Epoch 31/84
----------
train Loss: 0.0030 Acc: 0.9031
val Loss: 0.0131 Acc: 0.8140

Epoch 32/84
----------
train Loss: 0.0028 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 33/84
----------
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 34/84
----------
train Loss: 0.0029 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 35/84
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 36/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0029 Acc: 0.9286
val Loss: 0.0131 Acc: 0.8140

Epoch 37/84
----------
train Loss: 0.0029 Acc: 0.9235
val Loss: 0.0131 Acc: 0.8140

Epoch 38/84
----------
train Loss: 0.0032 Acc: 0.8980
val Loss: 0.0131 Acc: 0.8140

Epoch 39/84
----------
train Loss: 0.0030 Acc: 0.9107
val Loss: 0.0131 Acc: 0.8140

Epoch 40/84
----------
train Loss: 0.0030 Acc: 0.9082
val Loss: 0.0131 Acc: 0.8140

Epoch 41/84
----------
train Loss: 0.0029 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 42/84
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0131 Acc: 0.8140

Epoch 43/84
----------
train Loss: 0.0029 Acc: 0.9286
val Loss: 0.0131 Acc: 0.8140

Epoch 44/84
----------
train Loss: 0.0030 Acc: 0.9311
val Loss: 0.0131 Acc: 0.8140

Epoch 45/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 46/84
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0131 Acc: 0.8140

Epoch 47/84
----------
train Loss: 0.0031 Acc: 0.8980
val Loss: 0.0131 Acc: 0.8140

Epoch 48/84
----------
train Loss: 0.0031 Acc: 0.9031
val Loss: 0.0131 Acc: 0.8140

Epoch 49/84
----------
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0131 Acc: 0.8140

Epoch 50/84
----------
train Loss: 0.0029 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 51/84
----------
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0131 Acc: 0.8140

Epoch 52/84
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0131 Acc: 0.8140

Epoch 53/84
----------
train Loss: 0.0029 Acc: 0.9260
val Loss: 0.0130 Acc: 0.8140

Epoch 54/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0130 Acc: 0.8140

Epoch 55/84
----------
train Loss: 0.0031 Acc: 0.9107
val Loss: 0.0131 Acc: 0.8140

Epoch 56/84
----------
train Loss: 0.0028 Acc: 0.9133
val Loss: 0.0131 Acc: 0.8140

Epoch 57/84
----------
train Loss: 0.0027 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 58/84
----------
train Loss: 0.0030 Acc: 0.9235
val Loss: 0.0131 Acc: 0.8140

Epoch 59/84
----------
train Loss: 0.0028 Acc: 0.9311
val Loss: 0.0132 Acc: 0.8140

Epoch 60/84
----------
train Loss: 0.0028 Acc: 0.9311
val Loss: 0.0131 Acc: 0.8140

Epoch 61/84
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0130 Acc: 0.8140

Epoch 62/84
----------
train Loss: 0.0032 Acc: 0.9082
val Loss: 0.0130 Acc: 0.8140

Epoch 63/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0030 Acc: 0.9209
val Loss: 0.0130 Acc: 0.8140

Epoch 64/84
----------
train Loss: 0.0029 Acc: 0.9260
val Loss: 0.0131 Acc: 0.8140

Epoch 65/84
----------
train Loss: 0.0031 Acc: 0.8954
val Loss: 0.0131 Acc: 0.8140

Epoch 66/84
----------
train Loss: 0.0029 Acc: 0.9107
val Loss: 0.0130 Acc: 0.8140

Epoch 67/84
----------
train Loss: 0.0029 Acc: 0.9362
val Loss: 0.0131 Acc: 0.8140

Epoch 68/84
----------
train Loss: 0.0028 Acc: 0.9209
val Loss: 0.0130 Acc: 0.8140

Epoch 69/84
----------
train Loss: 0.0031 Acc: 0.9133
val Loss: 0.0130 Acc: 0.8140

Epoch 70/84
----------
train Loss: 0.0027 Acc: 0.9209
val Loss: 0.0131 Acc: 0.8140

Epoch 71/84
----------
train Loss: 0.0031 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 72/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0030 Acc: 0.9082
val Loss: 0.0131 Acc: 0.8140

Epoch 73/84
----------
train Loss: 0.0027 Acc: 0.9413
val Loss: 0.0131 Acc: 0.8140

Epoch 74/84
----------
train Loss: 0.0029 Acc: 0.9260
val Loss: 0.0131 Acc: 0.8140

Epoch 75/84
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 76/84
----------
train Loss: 0.0029 Acc: 0.9158
val Loss: 0.0131 Acc: 0.8140

Epoch 77/84
----------
train Loss: 0.0029 Acc: 0.9235
val Loss: 0.0131 Acc: 0.8140

Epoch 78/84
----------
train Loss: 0.0031 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 79/84
----------
train Loss: 0.0029 Acc: 0.9260
val Loss: 0.0131 Acc: 0.8140

Epoch 80/84
----------
train Loss: 0.0029 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 81/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0028 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 82/84
----------
train Loss: 0.0030 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Epoch 83/84
----------
train Loss: 0.0029 Acc: 0.9235
val Loss: 0.0132 Acc: 0.8140

Epoch 84/84
----------
train Loss: 0.0029 Acc: 0.9184
val Loss: 0.0131 Acc: 0.8140

Training complete in 2m 58s
Best val Acc: 0.837209

---Fine tuning.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0048 Acc: 0.8469
val Loss: 0.0139 Acc: 0.8372

Epoch 1/84
----------
train Loss: 0.0032 Acc: 0.9311
val Loss: 0.0119 Acc: 0.8837

Epoch 2/84
----------
train Loss: 0.0017 Acc: 0.9643
val Loss: 0.0122 Acc: 0.8605

Epoch 3/84
----------
train Loss: 0.0010 Acc: 0.9821
val Loss: 0.0145 Acc: 0.8372

Epoch 4/84
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0107 Acc: 0.8605

Epoch 5/84
----------
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0104 Acc: 0.8605

Epoch 6/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8837

Epoch 7/84
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0117 Acc: 0.9070

Epoch 8/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 9/84
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 10/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 11/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 12/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 13/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 14/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.9070

Epoch 15/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.9070

Epoch 16/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.9070

Epoch 17/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 18/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 19/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 20/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 21/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 22/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 23/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 24/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 25/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 26/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 27/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 28/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 29/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 30/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 31/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 32/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 33/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 34/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 35/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 36/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 37/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 38/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 39/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 40/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 41/84
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0118 Acc: 0.9070

Epoch 42/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 43/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 44/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 45/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 46/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 47/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 48/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 49/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 50/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 51/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 52/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 53/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 54/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 55/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 56/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 57/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 58/84
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0117 Acc: 0.9070

Epoch 59/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 60/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 61/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 62/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 63/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 64/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 65/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 66/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 67/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 68/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 69/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 70/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 71/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 72/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 73/84
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0118 Acc: 0.9070

Epoch 74/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 75/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 76/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 77/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 78/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 79/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 80/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 81/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9070

Epoch 82/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 83/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Epoch 84/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.9070

Training complete in 3m 15s
Best val Acc: 0.906977

---Testing---
Test accuracy: 0.990805
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 100 %
mean: 0.985436000237106, std: 0.01112595611112107
--------------------

run info[val: 0.15, epoch: 76, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0183 Acc: 0.3135
val Loss: 0.0227 Acc: 0.4923

Epoch 1/75
----------
train Loss: 0.0137 Acc: 0.5622
val Loss: 0.0189 Acc: 0.5231

Epoch 2/75
----------
train Loss: 0.0107 Acc: 0.6108
val Loss: 0.0123 Acc: 0.7385

Epoch 3/75
----------
train Loss: 0.0081 Acc: 0.7432
val Loss: 0.0121 Acc: 0.7077

Epoch 4/75
----------
train Loss: 0.0066 Acc: 0.7730
val Loss: 0.0093 Acc: 0.8000

Epoch 5/75
----------
train Loss: 0.0054 Acc: 0.8351
val Loss: 0.0068 Acc: 0.8615

Epoch 6/75
----------
train Loss: 0.0052 Acc: 0.8405
val Loss: 0.0065 Acc: 0.8615

Epoch 7/75
----------
train Loss: 0.0044 Acc: 0.8676
val Loss: 0.0065 Acc: 0.8923

Epoch 8/75
----------
train Loss: 0.0037 Acc: 0.8784
val Loss: 0.0059 Acc: 0.9077

Epoch 9/75
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.8838
val Loss: 0.0059 Acc: 0.8923

Epoch 10/75
----------
train Loss: 0.0032 Acc: 0.9000
val Loss: 0.0060 Acc: 0.9077

Epoch 11/75
----------
train Loss: 0.0034 Acc: 0.9000
val Loss: 0.0062 Acc: 0.9077

Epoch 12/75
----------
train Loss: 0.0035 Acc: 0.9054
val Loss: 0.0063 Acc: 0.8923

Epoch 13/75
----------
train Loss: 0.0031 Acc: 0.9270
val Loss: 0.0062 Acc: 0.8923

Epoch 14/75
----------
train Loss: 0.0032 Acc: 0.9243
val Loss: 0.0062 Acc: 0.8923

Epoch 15/75
----------
train Loss: 0.0033 Acc: 0.9108
val Loss: 0.0060 Acc: 0.8923

Epoch 16/75
----------
train Loss: 0.0032 Acc: 0.8973
val Loss: 0.0059 Acc: 0.9077

Epoch 17/75
----------
train Loss: 0.0032 Acc: 0.9108
val Loss: 0.0059 Acc: 0.9077

Epoch 18/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0031 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 19/75
----------
train Loss: 0.0032 Acc: 0.9000
val Loss: 0.0058 Acc: 0.9077

Epoch 20/75
----------
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0059 Acc: 0.9077

Epoch 21/75
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 22/75
----------
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0059 Acc: 0.9077

Epoch 23/75
----------
train Loss: 0.0033 Acc: 0.9135
val Loss: 0.0059 Acc: 0.9077

Epoch 24/75
----------
train Loss: 0.0030 Acc: 0.9000
val Loss: 0.0060 Acc: 0.9077

Epoch 25/75
----------
train Loss: 0.0030 Acc: 0.9297
val Loss: 0.0059 Acc: 0.9077

Epoch 26/75
----------
train Loss: 0.0030 Acc: 0.9162
val Loss: 0.0059 Acc: 0.9077

Epoch 27/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0029 Acc: 0.9378
val Loss: 0.0059 Acc: 0.9077

Epoch 28/75
----------
train Loss: 0.0033 Acc: 0.9027
val Loss: 0.0059 Acc: 0.9077

Epoch 29/75
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 30/75
----------
train Loss: 0.0031 Acc: 0.9108
val Loss: 0.0059 Acc: 0.9077

Epoch 31/75
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 32/75
----------
train Loss: 0.0031 Acc: 0.9351
val Loss: 0.0059 Acc: 0.9077

Epoch 33/75
----------
train Loss: 0.0031 Acc: 0.9027
val Loss: 0.0059 Acc: 0.9077

Epoch 34/75
----------
train Loss: 0.0032 Acc: 0.9081
val Loss: 0.0059 Acc: 0.9077

Epoch 35/75
----------
train Loss: 0.0031 Acc: 0.9054
val Loss: 0.0060 Acc: 0.9077

Epoch 36/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0030 Acc: 0.9216
val Loss: 0.0060 Acc: 0.9077

Epoch 37/75
----------
train Loss: 0.0032 Acc: 0.8973
val Loss: 0.0059 Acc: 0.9077

Epoch 38/75
----------
train Loss: 0.0032 Acc: 0.9216
val Loss: 0.0059 Acc: 0.9077

Epoch 39/75
----------
train Loss: 0.0029 Acc: 0.9108
val Loss: 0.0059 Acc: 0.9077

Epoch 40/75
----------
train Loss: 0.0030 Acc: 0.9270
val Loss: 0.0059 Acc: 0.9077

Epoch 41/75
----------
train Loss: 0.0029 Acc: 0.9324
val Loss: 0.0059 Acc: 0.9077

Epoch 42/75
----------
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0060 Acc: 0.9077

Epoch 43/75
----------
train Loss: 0.0029 Acc: 0.9216
val Loss: 0.0060 Acc: 0.9077

Epoch 44/75
----------
train Loss: 0.0037 Acc: 0.9054
val Loss: 0.0060 Acc: 0.9077

Epoch 45/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 46/75
----------
train Loss: 0.0031 Acc: 0.9243
val Loss: 0.0060 Acc: 0.9077

Epoch 47/75
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0060 Acc: 0.9077

Epoch 48/75
----------
train Loss: 0.0031 Acc: 0.9216
val Loss: 0.0060 Acc: 0.9077

Epoch 49/75
----------
train Loss: 0.0032 Acc: 0.8973
val Loss: 0.0060 Acc: 0.9077

Epoch 50/75
----------
train Loss: 0.0028 Acc: 0.9324
val Loss: 0.0060 Acc: 0.9077

Epoch 51/75
----------
train Loss: 0.0031 Acc: 0.9297
val Loss: 0.0059 Acc: 0.9077

Epoch 52/75
----------
train Loss: 0.0030 Acc: 0.9081
val Loss: 0.0060 Acc: 0.9077

Epoch 53/75
----------
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0059 Acc: 0.9077

Epoch 54/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0032 Acc: 0.9027
val Loss: 0.0060 Acc: 0.9077

Epoch 55/75
----------
train Loss: 0.0028 Acc: 0.9243
val Loss: 0.0060 Acc: 0.9077

Epoch 56/75
----------
train Loss: 0.0032 Acc: 0.9081
val Loss: 0.0060 Acc: 0.9077

Epoch 57/75
----------
train Loss: 0.0030 Acc: 0.9135
val Loss: 0.0060 Acc: 0.9077

Epoch 58/75
----------
train Loss: 0.0032 Acc: 0.9135
val Loss: 0.0059 Acc: 0.9077

Epoch 59/75
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0059 Acc: 0.9077

Epoch 60/75
----------
train Loss: 0.0031 Acc: 0.9162
val Loss: 0.0059 Acc: 0.9077

Epoch 61/75
----------
train Loss: 0.0026 Acc: 0.9486
val Loss: 0.0059 Acc: 0.9077

Epoch 62/75
----------
train Loss: 0.0029 Acc: 0.9297
val Loss: 0.0059 Acc: 0.9077

Epoch 63/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0029 Acc: 0.9270
val Loss: 0.0060 Acc: 0.9077

Epoch 64/75
----------
train Loss: 0.0028 Acc: 0.9243
val Loss: 0.0060 Acc: 0.9077

Epoch 65/75
----------
train Loss: 0.0031 Acc: 0.9162
val Loss: 0.0060 Acc: 0.9077

Epoch 66/75
----------
train Loss: 0.0032 Acc: 0.9027
val Loss: 0.0060 Acc: 0.9077

Epoch 67/75
----------
train Loss: 0.0032 Acc: 0.9135
val Loss: 0.0060 Acc: 0.9077

Epoch 68/75
----------
train Loss: 0.0032 Acc: 0.9027
val Loss: 0.0060 Acc: 0.9077

Epoch 69/75
----------
train Loss: 0.0032 Acc: 0.9108
val Loss: 0.0060 Acc: 0.9077

Epoch 70/75
----------
train Loss: 0.0034 Acc: 0.9108
val Loss: 0.0060 Acc: 0.9077

Epoch 71/75
----------
train Loss: 0.0031 Acc: 0.9027
val Loss: 0.0059 Acc: 0.9077

Epoch 72/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0029 Acc: 0.9243
val Loss: 0.0060 Acc: 0.9077

Epoch 73/75
----------
train Loss: 0.0031 Acc: 0.9162
val Loss: 0.0059 Acc: 0.9077

Epoch 74/75
----------
train Loss: 0.0032 Acc: 0.9216
val Loss: 0.0059 Acc: 0.9077

Epoch 75/75
----------
train Loss: 0.0031 Acc: 0.9108
val Loss: 0.0059 Acc: 0.9077

Training complete in 2m 42s
Best val Acc: 0.907692

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0032 Acc: 0.9054
val Loss: 0.0073 Acc: 0.8615

Epoch 1/75
----------
train Loss: 0.0024 Acc: 0.9351
val Loss: 0.0069 Acc: 0.8615

Epoch 2/75
----------
train Loss: 0.0012 Acc: 0.9757
val Loss: 0.0056 Acc: 0.9077

Epoch 3/75
----------
train Loss: 0.0010 Acc: 0.9919
val Loss: 0.0069 Acc: 0.8923

Epoch 4/75
----------
train Loss: 0.0004 Acc: 0.9973
val Loss: 0.0074 Acc: 0.8923

Epoch 5/75
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8923

Epoch 6/75
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0060 Acc: 0.9077

Epoch 7/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 8/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 9/75
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 10/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 11/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 12/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 13/75
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0059 Acc: 0.9231

Epoch 14/75
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0059 Acc: 0.9231

Epoch 15/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 16/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 17/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 18/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 19/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 20/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 21/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 22/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 23/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 24/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 25/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 26/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 27/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 28/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 29/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 30/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 31/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 32/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 33/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 34/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 35/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 36/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 37/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 38/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 39/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 40/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 41/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 42/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 43/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 44/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 45/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 46/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 47/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 48/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 49/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 50/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 51/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 52/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 53/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 54/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 55/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 56/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 57/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 58/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 59/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 60/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 61/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 62/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 63/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 64/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 65/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 66/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 67/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 68/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 69/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 70/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 71/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 72/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 73/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 74/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 75/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Training complete in 2m 57s
Best val Acc: 0.923077

---Testing---
Test accuracy: 0.988506
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 100 %
mean: 0.9827693335704393, std: 0.012069291178028907
--------------------

run info[val: 0.2, epoch: 71, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0178 Acc: 0.3448
val Loss: 0.0162 Acc: 0.4253

Epoch 1/70
----------
train Loss: 0.0140 Acc: 0.5460
val Loss: 0.0118 Acc: 0.6322

Epoch 2/70
----------
train Loss: 0.0103 Acc: 0.6638
val Loss: 0.0115 Acc: 0.6322

Epoch 3/70
----------
train Loss: 0.0082 Acc: 0.7328
val Loss: 0.0078 Acc: 0.7701

Epoch 4/70
----------
train Loss: 0.0069 Acc: 0.7874
val Loss: 0.0074 Acc: 0.7816

Epoch 5/70
----------
train Loss: 0.0060 Acc: 0.8075
val Loss: 0.0068 Acc: 0.7816

Epoch 6/70
----------
train Loss: 0.0042 Acc: 0.8908
val Loss: 0.0060 Acc: 0.8276

Epoch 7/70
----------
train Loss: 0.0037 Acc: 0.8966
val Loss: 0.0063 Acc: 0.8391

Epoch 8/70
----------
train Loss: 0.0035 Acc: 0.8764
val Loss: 0.0059 Acc: 0.8161

Epoch 9/70
----------
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8046

Epoch 10/70
----------
train Loss: 0.0029 Acc: 0.9167
val Loss: 0.0056 Acc: 0.8391

Epoch 11/70
----------
LR is set to 0.001
train Loss: 0.0026 Acc: 0.9195
val Loss: 0.0055 Acc: 0.8391

Epoch 12/70
----------
train Loss: 0.0024 Acc: 0.9397
val Loss: 0.0054 Acc: 0.8391

Epoch 13/70
----------
train Loss: 0.0024 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8506

Epoch 14/70
----------
train Loss: 0.0026 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8621

Epoch 15/70
----------
train Loss: 0.0024 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8506

Epoch 16/70
----------
train Loss: 0.0023 Acc: 0.9483
val Loss: 0.0053 Acc: 0.8506

Epoch 17/70
----------
train Loss: 0.0023 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 18/70
----------
train Loss: 0.0022 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 19/70
----------
train Loss: 0.0023 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8506

Epoch 20/70
----------
train Loss: 0.0025 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 21/70
----------
train Loss: 0.0022 Acc: 0.9655
val Loss: 0.0052 Acc: 0.8506

Epoch 22/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8391

Epoch 23/70
----------
train Loss: 0.0026 Acc: 0.9483
val Loss: 0.0052 Acc: 0.8391

Epoch 24/70
----------
train Loss: 0.0021 Acc: 0.9598
val Loss: 0.0052 Acc: 0.8506

Epoch 25/70
----------
train Loss: 0.0026 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8506

Epoch 26/70
----------
train Loss: 0.0024 Acc: 0.9483
val Loss: 0.0052 Acc: 0.8506

Epoch 27/70
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0052 Acc: 0.8506

Epoch 28/70
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0052 Acc: 0.8506

Epoch 29/70
----------
train Loss: 0.0022 Acc: 0.9425
val Loss: 0.0052 Acc: 0.8506

Epoch 30/70
----------
train Loss: 0.0022 Acc: 0.9598
val Loss: 0.0053 Acc: 0.8506

Epoch 31/70
----------
train Loss: 0.0021 Acc: 0.9626
val Loss: 0.0052 Acc: 0.8506

Epoch 32/70
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0053 Acc: 0.8506

Epoch 33/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9483
val Loss: 0.0053 Acc: 0.8506

Epoch 34/70
----------
train Loss: 0.0023 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 35/70
----------
train Loss: 0.0022 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 36/70
----------
train Loss: 0.0022 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 37/70
----------
train Loss: 0.0022 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 38/70
----------
train Loss: 0.0024 Acc: 0.9425
val Loss: 0.0052 Acc: 0.8506

Epoch 39/70
----------
train Loss: 0.0022 Acc: 0.9655
val Loss: 0.0052 Acc: 0.8506

Epoch 40/70
----------
train Loss: 0.0022 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 41/70
----------
train Loss: 0.0026 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8506

Epoch 42/70
----------
train Loss: 0.0020 Acc: 0.9598
val Loss: 0.0052 Acc: 0.8506

Epoch 43/70
----------
train Loss: 0.0023 Acc: 0.9598
val Loss: 0.0052 Acc: 0.8506

Epoch 44/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0052 Acc: 0.8506

Epoch 45/70
----------
train Loss: 0.0023 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 46/70
----------
train Loss: 0.0021 Acc: 0.9626
val Loss: 0.0052 Acc: 0.8506

Epoch 47/70
----------
train Loss: 0.0022 Acc: 0.9655
val Loss: 0.0052 Acc: 0.8506

Epoch 48/70
----------
train Loss: 0.0022 Acc: 0.9626
val Loss: 0.0052 Acc: 0.8506

Epoch 49/70
----------
train Loss: 0.0020 Acc: 0.9626
val Loss: 0.0052 Acc: 0.8506

Epoch 50/70
----------
train Loss: 0.0021 Acc: 0.9569
val Loss: 0.0052 Acc: 0.8506

Epoch 51/70
----------
train Loss: 0.0022 Acc: 0.9454
val Loss: 0.0052 Acc: 0.8506

Epoch 52/70
----------
train Loss: 0.0022 Acc: 0.9684
val Loss: 0.0052 Acc: 0.8506

Epoch 53/70
----------
train Loss: 0.0021 Acc: 0.9511
val Loss: 0.0053 Acc: 0.8506

Epoch 54/70
----------
train Loss: 0.0024 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8506

Epoch 55/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0023 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8506

Epoch 56/70
----------
train Loss: 0.0024 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8506

Epoch 57/70
----------
train Loss: 0.0022 Acc: 0.9598
val Loss: 0.0053 Acc: 0.8506

Epoch 58/70
----------
train Loss: 0.0025 Acc: 0.9569
val Loss: 0.0053 Acc: 0.8506

Epoch 59/70
----------
train Loss: 0.0023 Acc: 0.9483
val Loss: 0.0053 Acc: 0.8391

Epoch 60/70
----------
train Loss: 0.0022 Acc: 0.9598
val Loss: 0.0053 Acc: 0.8506

Epoch 61/70
----------
train Loss: 0.0024 Acc: 0.9483
val Loss: 0.0052 Acc: 0.8506

Epoch 62/70
----------
train Loss: 0.0024 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8506

Epoch 63/70
----------
train Loss: 0.0023 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 64/70
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0053 Acc: 0.8506

Epoch 65/70
----------
train Loss: 0.0021 Acc: 0.9569
val Loss: 0.0053 Acc: 0.8506

Epoch 66/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0053 Acc: 0.8391

Epoch 67/70
----------
train Loss: 0.0023 Acc: 0.9454
val Loss: 0.0053 Acc: 0.8391

Epoch 68/70
----------
train Loss: 0.0023 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8506

Epoch 69/70
----------
train Loss: 0.0022 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8391

Epoch 70/70
----------
train Loss: 0.0023 Acc: 0.9598
val Loss: 0.0052 Acc: 0.8506

Training complete in 2m 35s
Best val Acc: 0.862069

---Fine tuning.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0022 Acc: 0.9483
val Loss: 0.0050 Acc: 0.8736

Epoch 1/70
----------
train Loss: 0.0011 Acc: 0.9799
val Loss: 0.0047 Acc: 0.8851

Epoch 2/70
----------
train Loss: 0.0005 Acc: 0.9914
val Loss: 0.0048 Acc: 0.8736

Epoch 3/70
----------
train Loss: 0.0003 Acc: 0.9971
val Loss: 0.0049 Acc: 0.8736

Epoch 4/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8851

Epoch 5/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 6/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8966

Epoch 7/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8966

Epoch 8/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 9/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8851

Epoch 10/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8851

Epoch 11/70
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8851

Epoch 12/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 13/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 14/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 15/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 16/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 17/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 18/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 19/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 20/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 21/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 22/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 23/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 24/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 25/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 26/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 27/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 28/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 29/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 30/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 31/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 32/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 33/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 34/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 35/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8736

Epoch 36/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 37/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 38/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 39/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 40/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 41/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 42/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 43/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 44/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 45/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 46/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 47/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 48/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 49/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 50/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 51/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 52/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 53/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 54/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 55/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 56/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 57/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 58/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 59/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 60/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8851

Epoch 61/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 62/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 63/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 64/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 65/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 66/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8851

Epoch 67/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8736

Epoch 68/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8966

Epoch 69/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8966

Epoch 70/70
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8966

Training complete in 2m 48s
Best val Acc: 0.896552

---Testing---
Test accuracy: 0.979310
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.976926446935192, std: 0.00725626992617254
--------------------

run info[val: 0.25, epoch: 68, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/67
----------
LR is set to 0.01
train Loss: 0.0197 Acc: 0.3303
val Loss: 0.0349 Acc: 0.4815

Epoch 1/67
----------
train Loss: 0.0165 Acc: 0.4618
val Loss: 0.0176 Acc: 0.5648

Epoch 2/67
----------
train Loss: 0.0110 Acc: 0.6911
val Loss: 0.0141 Acc: 0.6574

Epoch 3/67
----------
train Loss: 0.0086 Acc: 0.7248
val Loss: 0.0132 Acc: 0.8426

Epoch 4/67
----------
LR is set to 0.001
train Loss: 0.0070 Acc: 0.8073
val Loss: 0.0090 Acc: 0.8241

Epoch 5/67
----------
train Loss: 0.0071 Acc: 0.8165
val Loss: 0.0090 Acc: 0.8056

Epoch 6/67
----------
train Loss: 0.0066 Acc: 0.8043
val Loss: 0.0120 Acc: 0.8333

Epoch 7/67
----------
train Loss: 0.0062 Acc: 0.8073
val Loss: 0.0105 Acc: 0.8241

Epoch 8/67
----------
LR is set to 0.00010000000000000002
train Loss: 0.0068 Acc: 0.8165
val Loss: 0.0097 Acc: 0.8333

Epoch 9/67
----------
train Loss: 0.0061 Acc: 0.8165
val Loss: 0.0097 Acc: 0.8333

Epoch 10/67
----------
train Loss: 0.0058 Acc: 0.8287
val Loss: 0.0092 Acc: 0.8333

Epoch 11/67
----------
train Loss: 0.0061 Acc: 0.8196
val Loss: 0.0097 Acc: 0.8148

Epoch 12/67
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0062 Acc: 0.8349
val Loss: 0.0100 Acc: 0.8056

Epoch 13/67
----------
train Loss: 0.0063 Acc: 0.8257
val Loss: 0.0106 Acc: 0.8056

Epoch 14/67
----------
train Loss: 0.0064 Acc: 0.8043
val Loss: 0.0085 Acc: 0.7963

Epoch 15/67
----------
train Loss: 0.0063 Acc: 0.8196
val Loss: 0.0086 Acc: 0.7963

Epoch 16/67
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0059 Acc: 0.8257
val Loss: 0.0104 Acc: 0.7963

Epoch 17/67
----------
train Loss: 0.0067 Acc: 0.8257
val Loss: 0.0092 Acc: 0.7963

Epoch 18/67
----------
train Loss: 0.0062 Acc: 0.8165
val Loss: 0.0084 Acc: 0.7963

Epoch 19/67
----------
train Loss: 0.0065 Acc: 0.8349
val Loss: 0.0100 Acc: 0.8333

Epoch 20/67
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0067 Acc: 0.8104
val Loss: 0.0104 Acc: 0.8241

Epoch 21/67
----------
train Loss: 0.0066 Acc: 0.8226
val Loss: 0.0129 Acc: 0.8241

Epoch 22/67
----------
train Loss: 0.0066 Acc: 0.8135
val Loss: 0.0079 Acc: 0.8333

Epoch 23/67
----------
train Loss: 0.0065 Acc: 0.8135
val Loss: 0.0094 Acc: 0.8056

Epoch 24/67
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0061 Acc: 0.8165
val Loss: 0.0161 Acc: 0.8333

Epoch 25/67
----------
train Loss: 0.0057 Acc: 0.8104
val Loss: 0.0130 Acc: 0.8148

Epoch 26/67
----------
train Loss: 0.0064 Acc: 0.8104
val Loss: 0.0095 Acc: 0.8056

Epoch 27/67
----------
train Loss: 0.0062 Acc: 0.8226
val Loss: 0.0091 Acc: 0.8148

Epoch 28/67
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0073 Acc: 0.8043
val Loss: 0.0119 Acc: 0.8241

Epoch 29/67
----------
train Loss: 0.0060 Acc: 0.8257
val Loss: 0.0084 Acc: 0.8333

Epoch 30/67
----------
train Loss: 0.0067 Acc: 0.8379
val Loss: 0.0117 Acc: 0.8426

Epoch 31/67
----------
train Loss: 0.0060 Acc: 0.8440
val Loss: 0.0083 Acc: 0.8333

Epoch 32/67
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0059 Acc: 0.8287
val Loss: 0.0125 Acc: 0.8333

Epoch 33/67
----------
train Loss: 0.0067 Acc: 0.8349
val Loss: 0.0093 Acc: 0.8148

Epoch 34/67
----------
train Loss: 0.0061 Acc: 0.8226
val Loss: 0.0103 Acc: 0.8241

Epoch 35/67
----------
train Loss: 0.0066 Acc: 0.8196
val Loss: 0.0086 Acc: 0.8241

Epoch 36/67
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0059 Acc: 0.8410
val Loss: 0.0089 Acc: 0.8333

Epoch 37/67
----------
train Loss: 0.0058 Acc: 0.8287
val Loss: 0.0074 Acc: 0.8333

Epoch 38/67
----------
train Loss: 0.0063 Acc: 0.8165
val Loss: 0.0132 Acc: 0.8333

Epoch 39/67
----------
train Loss: 0.0063 Acc: 0.8226
val Loss: 0.0083 Acc: 0.8333

Epoch 40/67
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0060 Acc: 0.8318
val Loss: 0.0121 Acc: 0.8241

Epoch 41/67
----------
train Loss: 0.0059 Acc: 0.8440
val Loss: 0.0103 Acc: 0.7963

Epoch 42/67
----------
train Loss: 0.0061 Acc: 0.8318
val Loss: 0.0096 Acc: 0.8056

Epoch 43/67
----------
train Loss: 0.0059 Acc: 0.8135
val Loss: 0.0112 Acc: 0.8148

Epoch 44/67
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0062 Acc: 0.8318
val Loss: 0.0109 Acc: 0.8056

Epoch 45/67
----------
train Loss: 0.0060 Acc: 0.8257
val Loss: 0.0125 Acc: 0.7963

Epoch 46/67
----------
train Loss: 0.0059 Acc: 0.8410
val Loss: 0.0095 Acc: 0.7963

Epoch 47/67
----------
train Loss: 0.0065 Acc: 0.8379
val Loss: 0.0121 Acc: 0.7963

Epoch 48/67
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0066 Acc: 0.8257
val Loss: 0.0090 Acc: 0.7963

Epoch 49/67
----------
train Loss: 0.0063 Acc: 0.8379
val Loss: 0.0085 Acc: 0.7963

Epoch 50/67
----------
train Loss: 0.0064 Acc: 0.8532
val Loss: 0.0105 Acc: 0.7963

Epoch 51/67
----------
train Loss: 0.0063 Acc: 0.8287
val Loss: 0.0081 Acc: 0.8056

Epoch 52/67
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0068 Acc: 0.8104
val Loss: 0.0087 Acc: 0.8056

Epoch 53/67
----------
train Loss: 0.0067 Acc: 0.8165
val Loss: 0.0076 Acc: 0.8056

Epoch 54/67
----------
train Loss: 0.0060 Acc: 0.8379
val Loss: 0.0129 Acc: 0.8056

Epoch 55/67
----------
train Loss: 0.0063 Acc: 0.8440
val Loss: 0.0095 Acc: 0.8056

Epoch 56/67
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0058 Acc: 0.8502
val Loss: 0.0116 Acc: 0.8056

Epoch 57/67
----------
train Loss: 0.0060 Acc: 0.8196
val Loss: 0.0085 Acc: 0.8056

Epoch 58/67
----------
train Loss: 0.0062 Acc: 0.8012
val Loss: 0.0097 Acc: 0.8056

Epoch 59/67
----------
train Loss: 0.0063 Acc: 0.8257
val Loss: 0.0090 Acc: 0.8241

Epoch 60/67
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0061 Acc: 0.8226
val Loss: 0.0112 Acc: 0.8333

Epoch 61/67
----------
train Loss: 0.0059 Acc: 0.8471
val Loss: 0.0081 Acc: 0.8333

Epoch 62/67
----------
train Loss: 0.0056 Acc: 0.8410
val Loss: 0.0108 Acc: 0.8333

Epoch 63/67
----------
train Loss: 0.0065 Acc: 0.8287
val Loss: 0.0129 Acc: 0.8148

Epoch 64/67
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0060 Acc: 0.8318
val Loss: 0.0099 Acc: 0.7963

Epoch 65/67
----------
train Loss: 0.0061 Acc: 0.8349
val Loss: 0.0120 Acc: 0.8056

Epoch 66/67
----------
train Loss: 0.0063 Acc: 0.8165
val Loss: 0.0112 Acc: 0.8148

Epoch 67/67
----------
train Loss: 0.0066 Acc: 0.8226
val Loss: 0.0115 Acc: 0.7963

Training complete in 2m 34s
Best val Acc: 0.842593

---Fine tuning.---
Epoch 0/67
----------
LR is set to 0.01
train Loss: 0.0067 Acc: 0.8043
val Loss: 0.0085 Acc: 0.8333

Epoch 1/67
----------
train Loss: 0.0046 Acc: 0.8899
val Loss: 0.0085 Acc: 0.8611

Epoch 2/67
----------
train Loss: 0.0030 Acc: 0.9358
val Loss: 0.0074 Acc: 0.8981

Epoch 3/67
----------
train Loss: 0.0018 Acc: 0.9694
val Loss: 0.0071 Acc: 0.8519

Epoch 4/67
----------
LR is set to 0.001
train Loss: 0.0012 Acc: 0.9755
val Loss: 0.0051 Acc: 0.8796

Epoch 5/67
----------
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0058 Acc: 0.8889

Epoch 6/67
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 7/67
----------
train Loss: 0.0008 Acc: 0.9847
val Loss: 0.0036 Acc: 0.8981

Epoch 8/67
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 9/67
----------
train Loss: 0.0010 Acc: 0.9847
val Loss: 0.0079 Acc: 0.8981

Epoch 10/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0057 Acc: 0.8981

Epoch 11/67
----------
train Loss: 0.0006 Acc: 0.9908
val Loss: 0.0037 Acc: 0.8981

Epoch 12/67
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0091 Acc: 0.8981

Epoch 13/67
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 14/67
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 15/67
----------
train Loss: 0.0007 Acc: 0.9908
val Loss: 0.0061 Acc: 0.8981

Epoch 16/67
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0069 Acc: 0.8981

Epoch 17/67
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0035 Acc: 0.8981

Epoch 18/67
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0085 Acc: 0.8981

Epoch 19/67
----------
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0045 Acc: 0.8981

Epoch 20/67
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0040 Acc: 0.8981

Epoch 21/67
----------
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0079 Acc: 0.8981

Epoch 22/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0073 Acc: 0.8981

Epoch 23/67
----------
train Loss: 0.0011 Acc: 0.9969
val Loss: 0.0033 Acc: 0.8981

Epoch 24/67
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0049 Acc: 0.8981

Epoch 25/67
----------
train Loss: 0.0009 Acc: 0.9908
val Loss: 0.0060 Acc: 0.8981

Epoch 26/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0047 Acc: 0.8981

Epoch 27/67
----------
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0100 Acc: 0.8981

Epoch 28/67
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0035 Acc: 0.8981

Epoch 29/67
----------
train Loss: 0.0006 Acc: 0.9908
val Loss: 0.0077 Acc: 0.8981

Epoch 30/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0077 Acc: 0.8981

Epoch 31/67
----------
train Loss: 0.0010 Acc: 0.9878
val Loss: 0.0044 Acc: 0.8981

Epoch 32/67
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 33/67
----------
train Loss: 0.0006 Acc: 0.9908
val Loss: 0.0044 Acc: 0.8981

Epoch 34/67
----------
train Loss: 0.0006 Acc: 0.9908
val Loss: 0.0082 Acc: 0.8981

Epoch 35/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0146 Acc: 0.8981

Epoch 36/67
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0118 Acc: 0.8981

Epoch 37/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0033 Acc: 0.8981

Epoch 38/67
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0073 Acc: 0.8981

Epoch 39/67
----------
train Loss: 0.0007 Acc: 0.9878
val Loss: 0.0062 Acc: 0.8981

Epoch 40/67
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0007 Acc: 0.9878
val Loss: 0.0066 Acc: 0.8981

Epoch 41/67
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0050 Acc: 0.8981

Epoch 42/67
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0086 Acc: 0.8981

Epoch 43/67
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0078 Acc: 0.8981

Epoch 44/67
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0069 Acc: 0.8981

Epoch 45/67
----------
train Loss: 0.0006 Acc: 0.9939
val Loss: 0.0045 Acc: 0.8981

Epoch 46/67
----------
train Loss: 0.0007 Acc: 0.9908
val Loss: 0.0046 Acc: 0.8981

Epoch 47/67
----------
train Loss: 0.0009 Acc: 0.9908
val Loss: 0.0038 Acc: 0.9074

Epoch 48/67
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0009 Acc: 0.9878
val Loss: 0.0040 Acc: 0.8981

Epoch 49/67
----------
train Loss: 0.0011 Acc: 0.9817
val Loss: 0.0036 Acc: 0.8981

Epoch 50/67
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0037 Acc: 0.8981

Epoch 51/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8981

Epoch 52/67
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0010 Acc: 0.9908
val Loss: 0.0041 Acc: 0.8981

Epoch 53/67
----------
train Loss: 0.0007 Acc: 0.9878
val Loss: 0.0054 Acc: 0.8981

Epoch 54/67
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0035 Acc: 0.8981

Epoch 55/67
----------
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0044 Acc: 0.8981

Epoch 56/67
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8981

Epoch 57/67
----------
train Loss: 0.0009 Acc: 0.9847
val Loss: 0.0077 Acc: 0.8981

Epoch 58/67
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0046 Acc: 0.8981

Epoch 59/67
----------
train Loss: 0.0008 Acc: 0.9878
val Loss: 0.0037 Acc: 0.8981

Epoch 60/67
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0010 Acc: 0.9939
val Loss: 0.0056 Acc: 0.8981

Epoch 61/67
----------
train Loss: 0.0009 Acc: 0.9939
val Loss: 0.0034 Acc: 0.8981

Epoch 62/67
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0040 Acc: 0.8981

Epoch 63/67
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0038 Acc: 0.8981

Epoch 64/67
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0044 Acc: 0.8981

Epoch 65/67
----------
train Loss: 0.0007 Acc: 0.9878
val Loss: 0.0043 Acc: 0.8981

Epoch 66/67
----------
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0092 Acc: 0.8981

Epoch 67/67
----------
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0055 Acc: 0.8981

Training complete in 2m 46s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.97289772534218, std: 0.009121833957504507
--------------------

run info[val: 0.3, epoch: 55, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0194 Acc: 0.3770
val Loss: 0.0214 Acc: 0.4385

Epoch 1/54
----------
train Loss: 0.0180 Acc: 0.5574
val Loss: 0.0176 Acc: 0.5692

Epoch 2/54
----------
train Loss: 0.0125 Acc: 0.6426
val Loss: 0.0178 Acc: 0.5769

Epoch 3/54
----------
train Loss: 0.0105 Acc: 0.6557
val Loss: 0.0126 Acc: 0.6923

Epoch 4/54
----------
train Loss: 0.0116 Acc: 0.7016
val Loss: 0.0132 Acc: 0.7154

Epoch 5/54
----------
train Loss: 0.0094 Acc: 0.7246
val Loss: 0.0120 Acc: 0.7385

Epoch 6/54
----------
train Loss: 0.0086 Acc: 0.7672
val Loss: 0.0120 Acc: 0.7154

Epoch 7/54
----------
LR is set to 0.001
train Loss: 0.0095 Acc: 0.7410
val Loss: 0.0126 Acc: 0.7308

Epoch 8/54
----------
train Loss: 0.0086 Acc: 0.7770
val Loss: 0.0100 Acc: 0.8077

Epoch 9/54
----------
train Loss: 0.0068 Acc: 0.8426
val Loss: 0.0077 Acc: 0.8000

Epoch 10/54
----------
train Loss: 0.0061 Acc: 0.8361
val Loss: 0.0098 Acc: 0.7462

Epoch 11/54
----------
train Loss: 0.0082 Acc: 0.7770
val Loss: 0.0115 Acc: 0.7385

Epoch 12/54
----------
train Loss: 0.0060 Acc: 0.7934
val Loss: 0.0115 Acc: 0.7769

Epoch 13/54
----------
train Loss: 0.0065 Acc: 0.8525
val Loss: 0.0111 Acc: 0.8000

Epoch 14/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0075 Acc: 0.8492
val Loss: 0.0081 Acc: 0.7846

Epoch 15/54
----------
train Loss: 0.0061 Acc: 0.8459
val Loss: 0.0092 Acc: 0.8077

Epoch 16/54
----------
train Loss: 0.0069 Acc: 0.8656
val Loss: 0.0127 Acc: 0.7923

Epoch 17/54
----------
train Loss: 0.0095 Acc: 0.8623
val Loss: 0.0095 Acc: 0.8000

Epoch 18/54
----------
train Loss: 0.0081 Acc: 0.8557
val Loss: 0.0088 Acc: 0.8000

Epoch 19/54
----------
train Loss: 0.0078 Acc: 0.8492
val Loss: 0.0094 Acc: 0.8000

Epoch 20/54
----------
train Loss: 0.0056 Acc: 0.8557
val Loss: 0.0099 Acc: 0.7923

Epoch 21/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0063 Acc: 0.8426
val Loss: 0.0091 Acc: 0.7923

Epoch 22/54
----------
train Loss: 0.0065 Acc: 0.8590
val Loss: 0.0082 Acc: 0.8077

Epoch 23/54
----------
train Loss: 0.0056 Acc: 0.8721
val Loss: 0.0081 Acc: 0.8077

Epoch 24/54
----------
train Loss: 0.0073 Acc: 0.8492
val Loss: 0.0106 Acc: 0.8077

Epoch 25/54
----------
train Loss: 0.0055 Acc: 0.8262
val Loss: 0.0098 Acc: 0.8077

Epoch 26/54
----------
train Loss: 0.0070 Acc: 0.8492
val Loss: 0.0085 Acc: 0.8000

Epoch 27/54
----------
train Loss: 0.0046 Acc: 0.8557
val Loss: 0.0085 Acc: 0.8000

Epoch 28/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0043 Acc: 0.8590
val Loss: 0.0097 Acc: 0.7923

Epoch 29/54
----------
train Loss: 0.0059 Acc: 0.8459
val Loss: 0.0095 Acc: 0.7923

Epoch 30/54
----------
train Loss: 0.0043 Acc: 0.8590
val Loss: 0.0100 Acc: 0.8000

Epoch 31/54
----------
train Loss: 0.0041 Acc: 0.8426
val Loss: 0.0098 Acc: 0.8000

Epoch 32/54
----------
train Loss: 0.0059 Acc: 0.8459
val Loss: 0.0090 Acc: 0.7923

Epoch 33/54
----------
train Loss: 0.0085 Acc: 0.8459
val Loss: 0.0092 Acc: 0.8000

Epoch 34/54
----------
train Loss: 0.0063 Acc: 0.8525
val Loss: 0.0092 Acc: 0.7923

Epoch 35/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0067 Acc: 0.8328
val Loss: 0.0084 Acc: 0.7923

Epoch 36/54
----------
train Loss: 0.0046 Acc: 0.8492
val Loss: 0.0080 Acc: 0.8077

Epoch 37/54
----------
train Loss: 0.0049 Acc: 0.8459
val Loss: 0.0089 Acc: 0.8077

Epoch 38/54
----------
train Loss: 0.0062 Acc: 0.8721
val Loss: 0.0107 Acc: 0.7923

Epoch 39/54
----------
train Loss: 0.0043 Acc: 0.8590
val Loss: 0.0111 Acc: 0.7923

Epoch 40/54
----------
train Loss: 0.0060 Acc: 0.8492
val Loss: 0.0090 Acc: 0.7846

Epoch 41/54
----------
train Loss: 0.0046 Acc: 0.8459
val Loss: 0.0083 Acc: 0.7923

Epoch 42/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0068 Acc: 0.8557
val Loss: 0.0091 Acc: 0.8000

Epoch 43/54
----------
train Loss: 0.0068 Acc: 0.8689
val Loss: 0.0080 Acc: 0.7923

Epoch 44/54
----------
train Loss: 0.0061 Acc: 0.8393
val Loss: 0.0090 Acc: 0.7923

Epoch 45/54
----------
train Loss: 0.0070 Acc: 0.8787
val Loss: 0.0071 Acc: 0.8000

Epoch 46/54
----------
train Loss: 0.0055 Acc: 0.8459
val Loss: 0.0108 Acc: 0.8000

Epoch 47/54
----------
train Loss: 0.0065 Acc: 0.8361
val Loss: 0.0100 Acc: 0.8000

Epoch 48/54
----------
train Loss: 0.0074 Acc: 0.8492
val Loss: 0.0081 Acc: 0.7923

Epoch 49/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0079 Acc: 0.8590
val Loss: 0.0095 Acc: 0.7923

Epoch 50/54
----------
train Loss: 0.0053 Acc: 0.8525
val Loss: 0.0099 Acc: 0.7846

Epoch 51/54
----------
train Loss: 0.0085 Acc: 0.8393
val Loss: 0.0089 Acc: 0.8000

Epoch 52/54
----------
train Loss: 0.0045 Acc: 0.8525
val Loss: 0.0086 Acc: 0.8000

Epoch 53/54
----------
train Loss: 0.0061 Acc: 0.8557
val Loss: 0.0105 Acc: 0.7923

Epoch 54/54
----------
train Loss: 0.0043 Acc: 0.8590
val Loss: 0.0090 Acc: 0.7846

Training complete in 2m 3s
Best val Acc: 0.807692

---Fine tuning.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0066 Acc: 0.8262
val Loss: 0.0107 Acc: 0.7692

Epoch 1/54
----------
train Loss: 0.0047 Acc: 0.8689
val Loss: 0.0117 Acc: 0.7231

Epoch 2/54
----------
train Loss: 0.0044 Acc: 0.8984
val Loss: 0.0230 Acc: 0.5538

Epoch 3/54
----------
train Loss: 0.0050 Acc: 0.8656
val Loss: 0.0258 Acc: 0.5385

Epoch 4/54
----------
train Loss: 0.0059 Acc: 0.8984
val Loss: 0.0365 Acc: 0.4077

Epoch 5/54
----------
train Loss: 0.0172 Acc: 0.7639
val Loss: 0.0446 Acc: 0.3769

Epoch 6/54
----------
train Loss: 0.0207 Acc: 0.6295
val Loss: 0.0359 Acc: 0.6385

Epoch 7/54
----------
LR is set to 0.001
train Loss: 0.0118 Acc: 0.8033
val Loss: 0.0235 Acc: 0.7308

Epoch 8/54
----------
train Loss: 0.0049 Acc: 0.8623
val Loss: 0.0215 Acc: 0.7462

Epoch 9/54
----------
train Loss: 0.0083 Acc: 0.8984
val Loss: 0.0198 Acc: 0.8000

Epoch 10/54
----------
train Loss: 0.0052 Acc: 0.9443
val Loss: 0.0124 Acc: 0.8154

Epoch 11/54
----------
train Loss: 0.0038 Acc: 0.9475
val Loss: 0.0173 Acc: 0.8077

Epoch 12/54
----------
train Loss: 0.0046 Acc: 0.9213
val Loss: 0.0191 Acc: 0.7846

Epoch 13/54
----------
train Loss: 0.0027 Acc: 0.9246
val Loss: 0.0217 Acc: 0.7462

Epoch 14/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9115
val Loss: 0.0238 Acc: 0.7615

Epoch 15/54
----------
train Loss: 0.0047 Acc: 0.9213
val Loss: 0.0193 Acc: 0.7846

Epoch 16/54
----------
train Loss: 0.0138 Acc: 0.9180
val Loss: 0.0158 Acc: 0.7923

Epoch 17/54
----------
train Loss: 0.0031 Acc: 0.9213
val Loss: 0.0140 Acc: 0.8000

Epoch 18/54
----------
train Loss: 0.0050 Acc: 0.9246
val Loss: 0.0184 Acc: 0.8077

Epoch 19/54
----------
train Loss: 0.0046 Acc: 0.9475
val Loss: 0.0176 Acc: 0.8231

Epoch 20/54
----------
train Loss: 0.0053 Acc: 0.9574
val Loss: 0.0192 Acc: 0.8231

Epoch 21/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0017 Acc: 0.9377
val Loss: 0.0135 Acc: 0.8231

Epoch 22/54
----------
train Loss: 0.0051 Acc: 0.9541
val Loss: 0.0152 Acc: 0.8231

Epoch 23/54
----------
train Loss: 0.0018 Acc: 0.9541
val Loss: 0.0172 Acc: 0.8231

Epoch 24/54
----------
train Loss: 0.0023 Acc: 0.9607
val Loss: 0.0203 Acc: 0.8231

Epoch 25/54
----------
train Loss: 0.0125 Acc: 0.9574
val Loss: 0.0161 Acc: 0.8000

Epoch 26/54
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0160 Acc: 0.8000

Epoch 27/54
----------
train Loss: 0.0033 Acc: 0.9508
val Loss: 0.0171 Acc: 0.7923

Epoch 28/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0030 Acc: 0.9607
val Loss: 0.0141 Acc: 0.8154

Epoch 29/54
----------
train Loss: 0.0024 Acc: 0.9607
val Loss: 0.0144 Acc: 0.8231

Epoch 30/54
----------
train Loss: 0.0012 Acc: 0.9639
val Loss: 0.0108 Acc: 0.8077

Epoch 31/54
----------
train Loss: 0.0019 Acc: 0.9410
val Loss: 0.0132 Acc: 0.8154

Epoch 32/54
----------
train Loss: 0.0023 Acc: 0.9508
val Loss: 0.0151 Acc: 0.8231

Epoch 33/54
----------
train Loss: 0.0036 Acc: 0.9541
val Loss: 0.0173 Acc: 0.8231

Epoch 34/54
----------
train Loss: 0.0025 Acc: 0.9607
val Loss: 0.0106 Acc: 0.8231

Epoch 35/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0027 Acc: 0.9607
val Loss: 0.0127 Acc: 0.8231

Epoch 36/54
----------
train Loss: 0.0112 Acc: 0.9574
val Loss: 0.0174 Acc: 0.7846

Epoch 37/54
----------
train Loss: 0.0023 Acc: 0.9639
val Loss: 0.0147 Acc: 0.8077

Epoch 38/54
----------
train Loss: 0.0015 Acc: 0.9508
val Loss: 0.0170 Acc: 0.8077

Epoch 39/54
----------
train Loss: 0.0033 Acc: 0.9541
val Loss: 0.0139 Acc: 0.8154

Epoch 40/54
----------
train Loss: 0.0043 Acc: 0.9574
val Loss: 0.0152 Acc: 0.8154

Epoch 41/54
----------
train Loss: 0.0029 Acc: 0.9508
val Loss: 0.0148 Acc: 0.8077

Epoch 42/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0017 Acc: 0.9574
val Loss: 0.0153 Acc: 0.8231

Epoch 43/54
----------
train Loss: 0.0050 Acc: 0.9541
val Loss: 0.0171 Acc: 0.8154

Epoch 44/54
----------
train Loss: 0.0056 Acc: 0.9607
val Loss: 0.0166 Acc: 0.8154

Epoch 45/54
----------
train Loss: 0.0075 Acc: 0.9410
val Loss: 0.0152 Acc: 0.8154

Epoch 46/54
----------
train Loss: 0.0039 Acc: 0.9541
val Loss: 0.0146 Acc: 0.8154

Epoch 47/54
----------
train Loss: 0.0013 Acc: 0.9705
val Loss: 0.0180 Acc: 0.8154

Epoch 48/54
----------
train Loss: 0.0046 Acc: 0.9508
val Loss: 0.0152 Acc: 0.8154

Epoch 49/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0022 Acc: 0.9410
val Loss: 0.0143 Acc: 0.8231

Epoch 50/54
----------
train Loss: 0.0030 Acc: 0.9475
val Loss: 0.0132 Acc: 0.8231

Epoch 51/54
----------
train Loss: 0.0019 Acc: 0.9705
val Loss: 0.0112 Acc: 0.8308

Epoch 52/54
----------
train Loss: 0.0025 Acc: 0.9508
val Loss: 0.0162 Acc: 0.8231

Epoch 53/54
----------
train Loss: 0.0013 Acc: 0.9574
val Loss: 0.0141 Acc: 0.8231

Epoch 54/54
----------
train Loss: 0.0017 Acc: 0.9475
val Loss: 0.0165 Acc: 0.8154

Training complete in 2m 14s
Best val Acc: 0.830769

---Testing---
Test accuracy: 0.937931
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 87 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 93 %
mean: 0.9377533388412266, std: 0.03705369432130528

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.99]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 55, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0159 Acc: 0.3418
val Loss: 0.0435 Acc: 0.3953

Epoch 1/54
----------
train Loss: 0.0130 Acc: 0.6020
val Loss: 0.0363 Acc: 0.6047

Epoch 2/54
----------
train Loss: 0.0093 Acc: 0.7194
val Loss: 0.0201 Acc: 0.6512

Epoch 3/54
----------
train Loss: 0.0069 Acc: 0.7679
val Loss: 0.0156 Acc: 0.7674

Epoch 4/54
----------
train Loss: 0.0056 Acc: 0.8138
val Loss: 0.0179 Acc: 0.6512

Epoch 5/54
----------
LR is set to 0.001
train Loss: 0.0053 Acc: 0.8189
val Loss: 0.0173 Acc: 0.6512

Epoch 6/54
----------
train Loss: 0.0052 Acc: 0.8036
val Loss: 0.0162 Acc: 0.6744

Epoch 7/54
----------
train Loss: 0.0051 Acc: 0.8316
val Loss: 0.0151 Acc: 0.7442

Epoch 8/54
----------
train Loss: 0.0047 Acc: 0.8444
val Loss: 0.0143 Acc: 0.7907

Epoch 9/54
----------
train Loss: 0.0047 Acc: 0.8520
val Loss: 0.0139 Acc: 0.7674

Epoch 10/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0045 Acc: 0.8597
val Loss: 0.0139 Acc: 0.7907

Epoch 11/54
----------
train Loss: 0.0043 Acc: 0.8597
val Loss: 0.0139 Acc: 0.7907

Epoch 12/54
----------
train Loss: 0.0046 Acc: 0.8673
val Loss: 0.0139 Acc: 0.7907

Epoch 13/54
----------
train Loss: 0.0046 Acc: 0.8520
val Loss: 0.0139 Acc: 0.7907

Epoch 14/54
----------
train Loss: 0.0046 Acc: 0.8520
val Loss: 0.0140 Acc: 0.7907

Epoch 15/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0044 Acc: 0.8597
val Loss: 0.0139 Acc: 0.7907

Epoch 16/54
----------
train Loss: 0.0044 Acc: 0.8699
val Loss: 0.0140 Acc: 0.7907

Epoch 17/54
----------
train Loss: 0.0043 Acc: 0.8801
val Loss: 0.0140 Acc: 0.7907

Epoch 18/54
----------
train Loss: 0.0042 Acc: 0.8699
val Loss: 0.0140 Acc: 0.7907

Epoch 19/54
----------
train Loss: 0.0047 Acc: 0.8469
val Loss: 0.0140 Acc: 0.7907

Epoch 20/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0047 Acc: 0.8469
val Loss: 0.0140 Acc: 0.7907

Epoch 21/54
----------
train Loss: 0.0045 Acc: 0.8418
val Loss: 0.0140 Acc: 0.7907

Epoch 22/54
----------
train Loss: 0.0045 Acc: 0.8393
val Loss: 0.0140 Acc: 0.7907

Epoch 23/54
----------
train Loss: 0.0046 Acc: 0.8622
val Loss: 0.0140 Acc: 0.7907

Epoch 24/54
----------
train Loss: 0.0044 Acc: 0.8776
val Loss: 0.0140 Acc: 0.7907

Epoch 25/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0047 Acc: 0.8444
val Loss: 0.0139 Acc: 0.7907

Epoch 26/54
----------
train Loss: 0.0045 Acc: 0.8801
val Loss: 0.0140 Acc: 0.7907

Epoch 27/54
----------
train Loss: 0.0046 Acc: 0.8622
val Loss: 0.0140 Acc: 0.7907

Epoch 28/54
----------
train Loss: 0.0044 Acc: 0.8571
val Loss: 0.0139 Acc: 0.7907

Epoch 29/54
----------
train Loss: 0.0047 Acc: 0.8418
val Loss: 0.0139 Acc: 0.7907

Epoch 30/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0047 Acc: 0.8622
val Loss: 0.0140 Acc: 0.7907

Epoch 31/54
----------
train Loss: 0.0047 Acc: 0.8469
val Loss: 0.0140 Acc: 0.7907

Epoch 32/54
----------
train Loss: 0.0046 Acc: 0.8571
val Loss: 0.0140 Acc: 0.7907

Epoch 33/54
----------
train Loss: 0.0045 Acc: 0.8546
val Loss: 0.0140 Acc: 0.7907

Epoch 34/54
----------
train Loss: 0.0047 Acc: 0.8520
val Loss: 0.0140 Acc: 0.7907

Epoch 35/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0043 Acc: 0.8776
val Loss: 0.0140 Acc: 0.7907

Epoch 36/54
----------
train Loss: 0.0044 Acc: 0.8776
val Loss: 0.0140 Acc: 0.7907

Epoch 37/54
----------
train Loss: 0.0047 Acc: 0.8648
val Loss: 0.0140 Acc: 0.7907

Epoch 38/54
----------
train Loss: 0.0047 Acc: 0.8648
val Loss: 0.0140 Acc: 0.7907

Epoch 39/54
----------
train Loss: 0.0043 Acc: 0.8827
val Loss: 0.0140 Acc: 0.7907

Epoch 40/54
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0045 Acc: 0.8673
val Loss: 0.0140 Acc: 0.7907

Epoch 41/54
----------
train Loss: 0.0045 Acc: 0.8648
val Loss: 0.0140 Acc: 0.7907

Epoch 42/54
----------
train Loss: 0.0046 Acc: 0.8418
val Loss: 0.0139 Acc: 0.7907

Epoch 43/54
----------
train Loss: 0.0048 Acc: 0.8495
val Loss: 0.0140 Acc: 0.7907

Epoch 44/54
----------
train Loss: 0.0046 Acc: 0.8724
val Loss: 0.0140 Acc: 0.7907

Epoch 45/54
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0045 Acc: 0.8520
val Loss: 0.0140 Acc: 0.7907

Epoch 46/54
----------
train Loss: 0.0045 Acc: 0.8469
val Loss: 0.0140 Acc: 0.7907

Epoch 47/54
----------
train Loss: 0.0046 Acc: 0.8699
val Loss: 0.0140 Acc: 0.7907

Epoch 48/54
----------
train Loss: 0.0044 Acc: 0.8546
val Loss: 0.0140 Acc: 0.7907

Epoch 49/54
----------
train Loss: 0.0046 Acc: 0.8342
val Loss: 0.0139 Acc: 0.7907

Epoch 50/54
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0045 Acc: 0.8495
val Loss: 0.0139 Acc: 0.7907

Epoch 51/54
----------
train Loss: 0.0048 Acc: 0.8342
val Loss: 0.0140 Acc: 0.7907

Epoch 52/54
----------
train Loss: 0.0043 Acc: 0.8571
val Loss: 0.0140 Acc: 0.7907

Epoch 53/54
----------
train Loss: 0.0045 Acc: 0.8622
val Loss: 0.0140 Acc: 0.7907

Epoch 54/54
----------
train Loss: 0.0044 Acc: 0.8776
val Loss: 0.0140 Acc: 0.7907

Training complete in 1m 54s
Best val Acc: 0.790698

---Fine tuning.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0050 Acc: 0.8265
val Loss: 0.0124 Acc: 0.8605

Epoch 1/54
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0120 Acc: 0.8372

Epoch 2/54
----------
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0111 Acc: 0.8605

Epoch 3/54
----------
train Loss: 0.0009 Acc: 0.9898
val Loss: 0.0118 Acc: 0.8605

Epoch 4/54
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 5/54
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0114 Acc: 0.8605

Epoch 6/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0110 Acc: 0.8605

Epoch 7/54
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0106 Acc: 0.8605

Epoch 8/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0104 Acc: 0.8605

Epoch 9/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 10/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 11/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 12/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 13/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 14/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 15/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 16/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 17/54
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 18/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 19/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 20/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 21/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 22/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 23/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 24/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 25/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 26/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 27/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 28/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 29/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 30/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 31/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 32/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 33/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 34/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 35/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 36/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 37/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 38/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 39/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 40/54
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 41/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 42/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 43/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 44/54
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 45/54
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 46/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 47/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 48/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 49/54
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 50/54
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 51/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 52/54
----------
train Loss: 0.0003 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 53/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 54/54
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Training complete in 2m 6s
Best val Acc: 0.860465

---Testing---
Test accuracy: 0.931034
--------------------
Accuracy of Dasyatiformes : 89 %
Accuracy of Myliobatiformes : 89 %
Accuracy of Rajiformes : 88 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 95 %
mean: 0.9195976594573392, std: 0.03400824147832327
--------------------

run info[val: 0.15, epoch: 75, randcrop: True, decay: 12]

---Training last layer.---
Epoch 0/74
----------
LR is set to 0.01
train Loss: 0.0190 Acc: 0.3135
val Loss: 0.0231 Acc: 0.5538

Epoch 1/74
----------
train Loss: 0.0153 Acc: 0.4432
val Loss: 0.0203 Acc: 0.4615

Epoch 2/74
----------
train Loss: 0.0102 Acc: 0.6243
val Loss: 0.0106 Acc: 0.8154

Epoch 3/74
----------
train Loss: 0.0084 Acc: 0.7405
val Loss: 0.0094 Acc: 0.8000

Epoch 4/74
----------
train Loss: 0.0063 Acc: 0.7703
val Loss: 0.0082 Acc: 0.8154

Epoch 5/74
----------
train Loss: 0.0056 Acc: 0.8027
val Loss: 0.0070 Acc: 0.8615

Epoch 6/74
----------
train Loss: 0.0055 Acc: 0.8432
val Loss: 0.0066 Acc: 0.8615

Epoch 7/74
----------
train Loss: 0.0049 Acc: 0.8486
val Loss: 0.0066 Acc: 0.8923

Epoch 8/74
----------
train Loss: 0.0040 Acc: 0.8784
val Loss: 0.0073 Acc: 0.8923

Epoch 9/74
----------
train Loss: 0.0036 Acc: 0.8919
val Loss: 0.0059 Acc: 0.9077

Epoch 10/74
----------
train Loss: 0.0035 Acc: 0.8973
val Loss: 0.0062 Acc: 0.8923

Epoch 11/74
----------
train Loss: 0.0028 Acc: 0.9378
val Loss: 0.0063 Acc: 0.9077

Epoch 12/74
----------
LR is set to 0.001
train Loss: 0.0029 Acc: 0.9189
val Loss: 0.0061 Acc: 0.9077

Epoch 13/74
----------
train Loss: 0.0024 Acc: 0.9324
val Loss: 0.0058 Acc: 0.9077

Epoch 14/74
----------
train Loss: 0.0027 Acc: 0.9378
val Loss: 0.0057 Acc: 0.9077

Epoch 15/74
----------
train Loss: 0.0027 Acc: 0.9297
val Loss: 0.0057 Acc: 0.9077

Epoch 16/74
----------
train Loss: 0.0025 Acc: 0.9568
val Loss: 0.0057 Acc: 0.9077

Epoch 17/74
----------
train Loss: 0.0026 Acc: 0.9378
val Loss: 0.0058 Acc: 0.9077

Epoch 18/74
----------
train Loss: 0.0026 Acc: 0.9243
val Loss: 0.0058 Acc: 0.9077

Epoch 19/74
----------
train Loss: 0.0027 Acc: 0.9243
val Loss: 0.0057 Acc: 0.9077

Epoch 20/74
----------
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0058 Acc: 0.9077

Epoch 21/74
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0057 Acc: 0.9077

Epoch 22/74
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 23/74
----------
train Loss: 0.0026 Acc: 0.9324
val Loss: 0.0057 Acc: 0.9077

Epoch 24/74
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0057 Acc: 0.9077

Epoch 25/74
----------
train Loss: 0.0023 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 26/74
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 27/74
----------
train Loss: 0.0024 Acc: 0.9595
val Loss: 0.0057 Acc: 0.9077

Epoch 28/74
----------
train Loss: 0.0026 Acc: 0.9216
val Loss: 0.0057 Acc: 0.9077

Epoch 29/74
----------
train Loss: 0.0026 Acc: 0.9270
val Loss: 0.0057 Acc: 0.9077

Epoch 30/74
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 31/74
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0057 Acc: 0.9077

Epoch 32/74
----------
train Loss: 0.0026 Acc: 0.9405
val Loss: 0.0057 Acc: 0.9077

Epoch 33/74
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0057 Acc: 0.9077

Epoch 34/74
----------
train Loss: 0.0026 Acc: 0.9270
val Loss: 0.0057 Acc: 0.9077

Epoch 35/74
----------
train Loss: 0.0024 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 36/74
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0027 Acc: 0.9243
val Loss: 0.0057 Acc: 0.9077

Epoch 37/74
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 38/74
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 39/74
----------
train Loss: 0.0023 Acc: 0.9378
val Loss: 0.0057 Acc: 0.9077

Epoch 40/74
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0056 Acc: 0.9077

Epoch 41/74
----------
train Loss: 0.0023 Acc: 0.9378
val Loss: 0.0057 Acc: 0.9077

Epoch 42/74
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 43/74
----------
train Loss: 0.0027 Acc: 0.9486
val Loss: 0.0056 Acc: 0.9077

Epoch 44/74
----------
train Loss: 0.0024 Acc: 0.9351
val Loss: 0.0056 Acc: 0.9077

Epoch 45/74
----------
train Loss: 0.0025 Acc: 0.9514
val Loss: 0.0056 Acc: 0.9077

Epoch 46/74
----------
train Loss: 0.0023 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 47/74
----------
train Loss: 0.0023 Acc: 0.9378
val Loss: 0.0056 Acc: 0.9077

Epoch 48/74
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 49/74
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0057 Acc: 0.9077

Epoch 50/74
----------
train Loss: 0.0026 Acc: 0.9270
val Loss: 0.0057 Acc: 0.9077

Epoch 51/74
----------
train Loss: 0.0024 Acc: 0.9459
val Loss: 0.0057 Acc: 0.9077

Epoch 52/74
----------
train Loss: 0.0027 Acc: 0.9243
val Loss: 0.0057 Acc: 0.9077

Epoch 53/74
----------
train Loss: 0.0028 Acc: 0.9162
val Loss: 0.0057 Acc: 0.9077

Epoch 54/74
----------
train Loss: 0.0023 Acc: 0.9622
val Loss: 0.0057 Acc: 0.9077

Epoch 55/74
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 56/74
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0057 Acc: 0.9077

Epoch 57/74
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 58/74
----------
train Loss: 0.0025 Acc: 0.9405
val Loss: 0.0057 Acc: 0.9077

Epoch 59/74
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0057 Acc: 0.9077

Epoch 60/74
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0056 Acc: 0.9077

Epoch 61/74
----------
train Loss: 0.0026 Acc: 0.9378
val Loss: 0.0056 Acc: 0.9077

Epoch 62/74
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0057 Acc: 0.9077

Epoch 63/74
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0057 Acc: 0.9077

Epoch 64/74
----------
train Loss: 0.0024 Acc: 0.9270
val Loss: 0.0057 Acc: 0.9077

Epoch 65/74
----------
train Loss: 0.0023 Acc: 0.9514
val Loss: 0.0057 Acc: 0.9077

Epoch 66/74
----------
train Loss: 0.0023 Acc: 0.9514
val Loss: 0.0057 Acc: 0.9077

Epoch 67/74
----------
train Loss: 0.0023 Acc: 0.9432
val Loss: 0.0057 Acc: 0.9077

Epoch 68/74
----------
train Loss: 0.0027 Acc: 0.9297
val Loss: 0.0057 Acc: 0.9077

Epoch 69/74
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0057 Acc: 0.9077

Epoch 70/74
----------
train Loss: 0.0027 Acc: 0.9324
val Loss: 0.0056 Acc: 0.9077

Epoch 71/74
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0057 Acc: 0.9077

Epoch 72/74
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0026 Acc: 0.9459
val Loss: 0.0056 Acc: 0.9077

Epoch 73/74
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0057 Acc: 0.9077

Epoch 74/74
----------
train Loss: 0.0023 Acc: 0.9405
val Loss: 0.0056 Acc: 0.9077

Training complete in 2m 38s
Best val Acc: 0.907692

---Fine tuning.---
Epoch 0/74
----------
LR is set to 0.01
train Loss: 0.0035 Acc: 0.8892
val Loss: 0.0083 Acc: 0.8462

Epoch 1/74
----------
train Loss: 0.0021 Acc: 0.9622
val Loss: 0.0067 Acc: 0.9077

Epoch 2/74
----------
train Loss: 0.0009 Acc: 0.9919
val Loss: 0.0052 Acc: 0.9077

Epoch 3/74
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8923

Epoch 4/74
----------
train Loss: 0.0003 Acc: 0.9973
val Loss: 0.0054 Acc: 0.9077

Epoch 5/74
----------
train Loss: 0.0003 Acc: 0.9946
val Loss: 0.0055 Acc: 0.9077

Epoch 6/74
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 7/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.9077

Epoch 8/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.9077

Epoch 9/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.9077

Epoch 10/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 11/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9231

Epoch 12/74
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0050 Acc: 0.9231

Epoch 13/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 14/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 15/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 16/74
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0050 Acc: 0.9077

Epoch 17/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 18/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 19/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.9077

Epoch 20/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 21/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 22/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 23/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 24/74
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 25/74
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0050 Acc: 0.9077

Epoch 26/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 27/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 28/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 29/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 30/74
----------
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0049 Acc: 0.9077

Epoch 31/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 32/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 33/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 34/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 35/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 36/74
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 37/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 38/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 39/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 40/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 41/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 42/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 43/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 44/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 45/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 46/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 47/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 48/74
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 49/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 50/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 51/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 52/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 53/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 54/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 55/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 56/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 57/74
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 58/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 59/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 60/74
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 61/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 62/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 63/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 64/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 65/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 66/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 67/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 68/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 69/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 70/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9077

Epoch 71/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 72/74
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9973
val Loss: 0.0049 Acc: 0.9077

Epoch 73/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Epoch 74/74
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9077

Training complete in 2m 55s
Best val Acc: 0.923077

---Testing---
Test accuracy: 0.988506
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 99 %
mean: 0.9842664095938313, std: 0.009756578948343544
--------------------

run info[val: 0.2, epoch: 59, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/58
----------
LR is set to 0.01
train Loss: 0.0197 Acc: 0.3017
val Loss: 0.0171 Acc: 0.3793

Epoch 1/58
----------
train Loss: 0.0152 Acc: 0.4253
val Loss: 0.0170 Acc: 0.4368

Epoch 2/58
----------
train Loss: 0.0120 Acc: 0.5862
val Loss: 0.0092 Acc: 0.7586

Epoch 3/58
----------
train Loss: 0.0074 Acc: 0.7759
val Loss: 0.0099 Acc: 0.6667

Epoch 4/58
----------
train Loss: 0.0071 Acc: 0.7500
val Loss: 0.0075 Acc: 0.8046

Epoch 5/58
----------
train Loss: 0.0053 Acc: 0.8391
val Loss: 0.0070 Acc: 0.7816

Epoch 6/58
----------
train Loss: 0.0050 Acc: 0.8276
val Loss: 0.0060 Acc: 0.8161

Epoch 7/58
----------
train Loss: 0.0051 Acc: 0.8649
val Loss: 0.0057 Acc: 0.8621

Epoch 8/58
----------
LR is set to 0.001
train Loss: 0.0039 Acc: 0.8966
val Loss: 0.0056 Acc: 0.8506

Epoch 9/58
----------
train Loss: 0.0039 Acc: 0.8822
val Loss: 0.0056 Acc: 0.8506

Epoch 10/58
----------
train Loss: 0.0035 Acc: 0.9052
val Loss: 0.0056 Acc: 0.8276

Epoch 11/58
----------
train Loss: 0.0037 Acc: 0.9023
val Loss: 0.0056 Acc: 0.8391

Epoch 12/58
----------
train Loss: 0.0038 Acc: 0.8966
val Loss: 0.0056 Acc: 0.8276

Epoch 13/58
----------
train Loss: 0.0035 Acc: 0.8966
val Loss: 0.0057 Acc: 0.8276

Epoch 14/58
----------
train Loss: 0.0036 Acc: 0.9023
val Loss: 0.0056 Acc: 0.8276

Epoch 15/58
----------
train Loss: 0.0036 Acc: 0.9109
val Loss: 0.0056 Acc: 0.8276

Epoch 16/58
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9224
val Loss: 0.0056 Acc: 0.8276

Epoch 17/58
----------
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0056 Acc: 0.8276

Epoch 18/58
----------
train Loss: 0.0035 Acc: 0.8994
val Loss: 0.0056 Acc: 0.8276

Epoch 19/58
----------
train Loss: 0.0038 Acc: 0.9052
val Loss: 0.0056 Acc: 0.8276

Epoch 20/58
----------
train Loss: 0.0033 Acc: 0.9368
val Loss: 0.0056 Acc: 0.8276

Epoch 21/58
----------
train Loss: 0.0035 Acc: 0.9023
val Loss: 0.0057 Acc: 0.8276

Epoch 22/58
----------
train Loss: 0.0038 Acc: 0.9195
val Loss: 0.0056 Acc: 0.8276

Epoch 23/58
----------
train Loss: 0.0036 Acc: 0.9023
val Loss: 0.0056 Acc: 0.8276

Epoch 24/58
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0056 Acc: 0.8276

Epoch 25/58
----------
train Loss: 0.0037 Acc: 0.8966
val Loss: 0.0057 Acc: 0.8276

Epoch 26/58
----------
train Loss: 0.0035 Acc: 0.9167
val Loss: 0.0057 Acc: 0.8276

Epoch 27/58
----------
train Loss: 0.0038 Acc: 0.8994
val Loss: 0.0056 Acc: 0.8276

Epoch 28/58
----------
train Loss: 0.0036 Acc: 0.9138
val Loss: 0.0056 Acc: 0.8276

Epoch 29/58
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0056 Acc: 0.8276

Epoch 30/58
----------
train Loss: 0.0035 Acc: 0.9080
val Loss: 0.0056 Acc: 0.8276

Epoch 31/58
----------
train Loss: 0.0037 Acc: 0.9080
val Loss: 0.0056 Acc: 0.8276

Epoch 32/58
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0035 Acc: 0.8937
val Loss: 0.0056 Acc: 0.8276

Epoch 33/58
----------
train Loss: 0.0036 Acc: 0.9080
val Loss: 0.0056 Acc: 0.8276

Epoch 34/58
----------
train Loss: 0.0039 Acc: 0.8966
val Loss: 0.0056 Acc: 0.8276

Epoch 35/58
----------
train Loss: 0.0033 Acc: 0.9052
val Loss: 0.0056 Acc: 0.8276

Epoch 36/58
----------
train Loss: 0.0033 Acc: 0.9167
val Loss: 0.0056 Acc: 0.8276

Epoch 37/58
----------
train Loss: 0.0035 Acc: 0.9052
val Loss: 0.0057 Acc: 0.8276

Epoch 38/58
----------
train Loss: 0.0033 Acc: 0.9224
val Loss: 0.0056 Acc: 0.8276

Epoch 39/58
----------
train Loss: 0.0033 Acc: 0.9080
val Loss: 0.0056 Acc: 0.8276

Epoch 40/58
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0056 Acc: 0.8276

Epoch 41/58
----------
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0056 Acc: 0.8276

Epoch 42/58
----------
train Loss: 0.0036 Acc: 0.8937
val Loss: 0.0056 Acc: 0.8276

Epoch 43/58
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0055 Acc: 0.8276

Epoch 44/58
----------
train Loss: 0.0033 Acc: 0.9167
val Loss: 0.0055 Acc: 0.8276

Epoch 45/58
----------
train Loss: 0.0032 Acc: 0.9310
val Loss: 0.0056 Acc: 0.8276

Epoch 46/58
----------
train Loss: 0.0037 Acc: 0.9138
val Loss: 0.0056 Acc: 0.8276

Epoch 47/58
----------
train Loss: 0.0039 Acc: 0.8793
val Loss: 0.0056 Acc: 0.8391

Epoch 48/58
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0043 Acc: 0.9023
val Loss: 0.0056 Acc: 0.8276

Epoch 49/58
----------
train Loss: 0.0036 Acc: 0.9052
val Loss: 0.0056 Acc: 0.8276

Epoch 50/58
----------
train Loss: 0.0036 Acc: 0.9080
val Loss: 0.0055 Acc: 0.8276

Epoch 51/58
----------
train Loss: 0.0036 Acc: 0.9167
val Loss: 0.0055 Acc: 0.8276

Epoch 52/58
----------
train Loss: 0.0034 Acc: 0.9138
val Loss: 0.0055 Acc: 0.8276

Epoch 53/58
----------
train Loss: 0.0035 Acc: 0.9138
val Loss: 0.0055 Acc: 0.8276

Epoch 54/58
----------
train Loss: 0.0032 Acc: 0.9253
val Loss: 0.0055 Acc: 0.8276

Epoch 55/58
----------
train Loss: 0.0034 Acc: 0.9109
val Loss: 0.0055 Acc: 0.8276

Epoch 56/58
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0034 Acc: 0.8994
val Loss: 0.0056 Acc: 0.8276

Epoch 57/58
----------
train Loss: 0.0039 Acc: 0.8966
val Loss: 0.0056 Acc: 0.8276

Epoch 58/58
----------
train Loss: 0.0037 Acc: 0.9080
val Loss: 0.0055 Acc: 0.8391

Training complete in 2m 8s
Best val Acc: 0.862069

---Fine tuning.---
Epoch 0/58
----------
LR is set to 0.01
train Loss: 0.0047 Acc: 0.8563
val Loss: 0.0050 Acc: 0.8736

Epoch 1/58
----------
train Loss: 0.0025 Acc: 0.9454
val Loss: 0.0062 Acc: 0.8276

Epoch 2/58
----------
train Loss: 0.0015 Acc: 0.9828
val Loss: 0.0042 Acc: 0.9080

Epoch 3/58
----------
train Loss: 0.0006 Acc: 0.9971
val Loss: 0.0043 Acc: 0.9195

Epoch 4/58
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0039 Acc: 0.8966

Epoch 5/58
----------
train Loss: 0.0003 Acc: 0.9943
val Loss: 0.0037 Acc: 0.8966

Epoch 6/58
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8966

Epoch 7/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8851

Epoch 8/58
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 9/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 10/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 11/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 12/58
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0040 Acc: 0.8966

Epoch 13/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9080

Epoch 14/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 15/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9080

Epoch 16/58
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 17/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9080

Epoch 18/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 19/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 20/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 21/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 22/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 23/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 24/58
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 25/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 26/58
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0039 Acc: 0.9195

Epoch 27/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 28/58
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.9195

Epoch 29/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 30/58
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0039 Acc: 0.9195

Epoch 31/58
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.9195

Epoch 32/58
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 33/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 34/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 35/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 36/58
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0039 Acc: 0.9195

Epoch 37/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 38/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 39/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 40/58
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 41/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 42/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 43/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 44/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 45/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 46/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Epoch 47/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 48/58
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 49/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 50/58
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0039 Acc: 0.9195

Epoch 51/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 52/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9080

Epoch 53/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 54/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 55/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 56/58
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 57/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 58/58
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9195

Training complete in 2m 23s
Best val Acc: 0.919540

---Testing---
Test accuracy: 0.979310
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 98 %
mean: 0.9756269017759978, std: 0.010450436387734938
--------------------

run info[val: 0.25, epoch: 82, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0191 Acc: 0.3761
val Loss: 0.0257 Acc: 0.4444

Epoch 1/81
----------
train Loss: 0.0156 Acc: 0.4740
val Loss: 0.0208 Acc: 0.4259

Epoch 2/81
----------
train Loss: 0.0115 Acc: 0.6177
val Loss: 0.0115 Acc: 0.7500

Epoch 3/81
----------
train Loss: 0.0101 Acc: 0.7278
val Loss: 0.0118 Acc: 0.7685

Epoch 4/81
----------
train Loss: 0.0071 Acc: 0.7920
val Loss: 0.0140 Acc: 0.8056

Epoch 5/81
----------
train Loss: 0.0085 Acc: 0.8012
val Loss: 0.0106 Acc: 0.8426

Epoch 6/81
----------
train Loss: 0.0064 Acc: 0.8104
val Loss: 0.0121 Acc: 0.8056

Epoch 7/81
----------
train Loss: 0.0047 Acc: 0.8807
val Loss: 0.0081 Acc: 0.7500

Epoch 8/81
----------
LR is set to 0.001
train Loss: 0.0046 Acc: 0.8410
val Loss: 0.0120 Acc: 0.8241

Epoch 9/81
----------
train Loss: 0.0044 Acc: 0.8716
val Loss: 0.0096 Acc: 0.8519

Epoch 10/81
----------
train Loss: 0.0039 Acc: 0.9021
val Loss: 0.0154 Acc: 0.8519

Epoch 11/81
----------
train Loss: 0.0040 Acc: 0.9113
val Loss: 0.0089 Acc: 0.8704

Epoch 12/81
----------
train Loss: 0.0040 Acc: 0.9174
val Loss: 0.0065 Acc: 0.8796

Epoch 13/81
----------
train Loss: 0.0035 Acc: 0.9235
val Loss: 0.0065 Acc: 0.8519

Epoch 14/81
----------
train Loss: 0.0038 Acc: 0.9052
val Loss: 0.0080 Acc: 0.8796

Epoch 15/81
----------
train Loss: 0.0040 Acc: 0.9144
val Loss: 0.0112 Acc: 0.8889

Epoch 16/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0041 Acc: 0.9205
val Loss: 0.0092 Acc: 0.8796

Epoch 17/81
----------
train Loss: 0.0038 Acc: 0.9021
val Loss: 0.0075 Acc: 0.8796

Epoch 18/81
----------
train Loss: 0.0035 Acc: 0.9083
val Loss: 0.0055 Acc: 0.8889

Epoch 19/81
----------
train Loss: 0.0034 Acc: 0.9052
val Loss: 0.0080 Acc: 0.8796

Epoch 20/81
----------
train Loss: 0.0038 Acc: 0.9052
val Loss: 0.0064 Acc: 0.8796

Epoch 21/81
----------
train Loss: 0.0041 Acc: 0.9083
val Loss: 0.0065 Acc: 0.8796

Epoch 22/81
----------
train Loss: 0.0036 Acc: 0.9083
val Loss: 0.0068 Acc: 0.8796

Epoch 23/81
----------
train Loss: 0.0037 Acc: 0.9113
val Loss: 0.0102 Acc: 0.8796

Epoch 24/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0029 Acc: 0.9297
val Loss: 0.0125 Acc: 0.8796

Epoch 25/81
----------
train Loss: 0.0032 Acc: 0.9266
val Loss: 0.0072 Acc: 0.8796

Epoch 26/81
----------
train Loss: 0.0040 Acc: 0.9113
val Loss: 0.0062 Acc: 0.8796

Epoch 27/81
----------
train Loss: 0.0036 Acc: 0.9052
val Loss: 0.0053 Acc: 0.8889

Epoch 28/81
----------
train Loss: 0.0037 Acc: 0.9113
val Loss: 0.0091 Acc: 0.8889

Epoch 29/81
----------
train Loss: 0.0033 Acc: 0.9083
val Loss: 0.0070 Acc: 0.8889

Epoch 30/81
----------
train Loss: 0.0037 Acc: 0.9174
val Loss: 0.0070 Acc: 0.8889

Epoch 31/81
----------
train Loss: 0.0032 Acc: 0.9205
val Loss: 0.0105 Acc: 0.8796

Epoch 32/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0041 Acc: 0.9174
val Loss: 0.0078 Acc: 0.8889

Epoch 33/81
----------
train Loss: 0.0039 Acc: 0.9144
val Loss: 0.0078 Acc: 0.8889

Epoch 34/81
----------
train Loss: 0.0043 Acc: 0.8930
val Loss: 0.0051 Acc: 0.8889

Epoch 35/81
----------
train Loss: 0.0034 Acc: 0.9205
val Loss: 0.0067 Acc: 0.8889

Epoch 36/81
----------
train Loss: 0.0035 Acc: 0.9235
val Loss: 0.0087 Acc: 0.8889

Epoch 37/81
----------
train Loss: 0.0038 Acc: 0.9205
val Loss: 0.0074 Acc: 0.8889

Epoch 38/81
----------
train Loss: 0.0034 Acc: 0.9266
val Loss: 0.0104 Acc: 0.8889

Epoch 39/81
----------
train Loss: 0.0038 Acc: 0.9083
val Loss: 0.0089 Acc: 0.8889

Epoch 40/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0041 Acc: 0.9174
val Loss: 0.0051 Acc: 0.8889

Epoch 41/81
----------
train Loss: 0.0037 Acc: 0.9205
val Loss: 0.0063 Acc: 0.8889

Epoch 42/81
----------
train Loss: 0.0045 Acc: 0.9021
val Loss: 0.0092 Acc: 0.8889

Epoch 43/81
----------
train Loss: 0.0037 Acc: 0.9021
val Loss: 0.0065 Acc: 0.8889

Epoch 44/81
----------
train Loss: 0.0036 Acc: 0.9235
val Loss: 0.0062 Acc: 0.8889

Epoch 45/81
----------
train Loss: 0.0032 Acc: 0.9358
val Loss: 0.0063 Acc: 0.8889

Epoch 46/81
----------
train Loss: 0.0033 Acc: 0.9144
val Loss: 0.0076 Acc: 0.8889

Epoch 47/81
----------
train Loss: 0.0039 Acc: 0.9205
val Loss: 0.0075 Acc: 0.8704

Epoch 48/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0032 Acc: 0.9266
val Loss: 0.0062 Acc: 0.8889

Epoch 49/81
----------
train Loss: 0.0040 Acc: 0.9052
val Loss: 0.0093 Acc: 0.8889

Epoch 50/81
----------
train Loss: 0.0034 Acc: 0.9297
val Loss: 0.0119 Acc: 0.8889

Epoch 51/81
----------
train Loss: 0.0037 Acc: 0.9052
val Loss: 0.0116 Acc: 0.8889

Epoch 52/81
----------
train Loss: 0.0034 Acc: 0.9388
val Loss: 0.0102 Acc: 0.8889

Epoch 53/81
----------
train Loss: 0.0037 Acc: 0.9052
val Loss: 0.0107 Acc: 0.8889

Epoch 54/81
----------
train Loss: 0.0038 Acc: 0.9083
val Loss: 0.0127 Acc: 0.8889

Epoch 55/81
----------
train Loss: 0.0037 Acc: 0.8991
val Loss: 0.0064 Acc: 0.8889

Epoch 56/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0079 Acc: 0.8889

Epoch 57/81
----------
train Loss: 0.0038 Acc: 0.9297
val Loss: 0.0056 Acc: 0.8796

Epoch 58/81
----------
train Loss: 0.0041 Acc: 0.9083
val Loss: 0.0057 Acc: 0.8889

Epoch 59/81
----------
train Loss: 0.0039 Acc: 0.9083
val Loss: 0.0101 Acc: 0.8889

Epoch 60/81
----------
train Loss: 0.0033 Acc: 0.9266
val Loss: 0.0060 Acc: 0.8796

Epoch 61/81
----------
train Loss: 0.0034 Acc: 0.9235
val Loss: 0.0074 Acc: 0.8796

Epoch 62/81
----------
train Loss: 0.0035 Acc: 0.9174
val Loss: 0.0080 Acc: 0.8889

Epoch 63/81
----------
train Loss: 0.0036 Acc: 0.9266
val Loss: 0.0061 Acc: 0.8889

Epoch 64/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0033 Acc: 0.9235
val Loss: 0.0084 Acc: 0.8889

Epoch 65/81
----------
train Loss: 0.0032 Acc: 0.9266
val Loss: 0.0090 Acc: 0.8889

Epoch 66/81
----------
train Loss: 0.0037 Acc: 0.9174
val Loss: 0.0055 Acc: 0.8889

Epoch 67/81
----------
train Loss: 0.0034 Acc: 0.9021
val Loss: 0.0064 Acc: 0.8889

Epoch 68/81
----------
train Loss: 0.0038 Acc: 0.9235
val Loss: 0.0072 Acc: 0.8889

Epoch 69/81
----------
train Loss: 0.0040 Acc: 0.8930
val Loss: 0.0077 Acc: 0.8796

Epoch 70/81
----------
train Loss: 0.0040 Acc: 0.9113
val Loss: 0.0066 Acc: 0.8889

Epoch 71/81
----------
train Loss: 0.0037 Acc: 0.9021
val Loss: 0.0062 Acc: 0.8796

Epoch 72/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0036 Acc: 0.9083
val Loss: 0.0058 Acc: 0.8796

Epoch 73/81
----------
train Loss: 0.0031 Acc: 0.9174
val Loss: 0.0081 Acc: 0.8796

Epoch 74/81
----------
train Loss: 0.0035 Acc: 0.8838
val Loss: 0.0100 Acc: 0.8796

Epoch 75/81
----------
train Loss: 0.0039 Acc: 0.8960
val Loss: 0.0082 Acc: 0.8889

Epoch 76/81
----------
train Loss: 0.0037 Acc: 0.9052
val Loss: 0.0055 Acc: 0.8889

Epoch 77/81
----------
train Loss: 0.0041 Acc: 0.9052
val Loss: 0.0067 Acc: 0.8889

Epoch 78/81
----------
train Loss: 0.0036 Acc: 0.9235
val Loss: 0.0052 Acc: 0.8889

Epoch 79/81
----------
train Loss: 0.0036 Acc: 0.9113
val Loss: 0.0077 Acc: 0.8889

Epoch 80/81
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9297
val Loss: 0.0069 Acc: 0.8796

Epoch 81/81
----------
train Loss: 0.0035 Acc: 0.8930
val Loss: 0.0080 Acc: 0.8796

Training complete in 3m 3s
Best val Acc: 0.888889

---Fine tuning.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0037 Acc: 0.9174
val Loss: 0.0070 Acc: 0.8519

Epoch 1/81
----------
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0062 Acc: 0.8148

Epoch 2/81
----------
train Loss: 0.0015 Acc: 0.9817
val Loss: 0.0042 Acc: 0.8796

Epoch 3/81
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0046 Acc: 0.8704

Epoch 4/81
----------
train Loss: 0.0009 Acc: 0.9847
val Loss: 0.0076 Acc: 0.8981

Epoch 5/81
----------
train Loss: 0.0007 Acc: 0.9908
val Loss: 0.0046 Acc: 0.8796

Epoch 6/81
----------
train Loss: 0.0004 Acc: 0.9969
val Loss: 0.0088 Acc: 0.8704

Epoch 7/81
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8889

Epoch 8/81
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8889

Epoch 9/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0098 Acc: 0.9074

Epoch 10/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0099 Acc: 0.9074

Epoch 11/81
----------
train Loss: 0.0004 Acc: 0.9939
val Loss: 0.0033 Acc: 0.9074

Epoch 12/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8981

Epoch 13/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9074

Epoch 14/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0032 Acc: 0.9074

Epoch 15/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0069 Acc: 0.9074

Epoch 16/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9074

Epoch 17/81
----------
train Loss: 0.0005 Acc: 0.9969
val Loss: 0.0088 Acc: 0.9074

Epoch 18/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9074

Epoch 19/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9074

Epoch 20/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 21/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9074

Epoch 22/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9074

Epoch 23/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0105 Acc: 0.9074

Epoch 24/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 25/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9074

Epoch 26/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9074

Epoch 27/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.9074

Epoch 28/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0078 Acc: 0.9074

Epoch 29/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9074

Epoch 30/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9074

Epoch 31/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8981

Epoch 32/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9074

Epoch 33/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 34/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8981

Epoch 35/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8981

Epoch 36/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.9074

Epoch 37/81
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 38/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9074

Epoch 39/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0033 Acc: 0.9074

Epoch 40/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.9074

Epoch 41/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9074

Epoch 42/81
----------
train Loss: 0.0001 Acc: 0.9969
val Loss: 0.0049 Acc: 0.9074

Epoch 43/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 44/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 45/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9074

Epoch 46/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 47/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9074

Epoch 48/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 49/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.9074

Epoch 50/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 51/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8981

Epoch 52/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8981

Epoch 53/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0036 Acc: 0.9074

Epoch 54/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9074

Epoch 55/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9074

Epoch 56/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9074

Epoch 57/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.9074

Epoch 58/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.9074

Epoch 59/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0103 Acc: 0.9074

Epoch 60/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.9074

Epoch 61/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.9074

Epoch 62/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8981

Epoch 63/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0128 Acc: 0.9074

Epoch 64/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.9074

Epoch 65/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9074

Epoch 66/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9074

Epoch 67/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0111 Acc: 0.9074

Epoch 68/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 69/81
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9074

Epoch 70/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0050 Acc: 0.9074

Epoch 71/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9074

Epoch 72/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.9074

Epoch 73/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9074

Epoch 74/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0124 Acc: 0.9074

Epoch 75/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9074

Epoch 76/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9074

Epoch 77/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9074

Epoch 78/81
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0069 Acc: 0.9074

Epoch 79/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0053 Acc: 0.9074

Epoch 80/81
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.9074

Epoch 81/81
----------
train Loss: 0.0004 Acc: 0.9969
val Loss: 0.0032 Acc: 0.8981

Training complete in 3m 23s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 97 %
mean: 0.975694346524766, std: 0.010532651663091268
--------------------

run info[val: 0.3, epoch: 54, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0230 Acc: 0.2852
val Loss: 0.0271 Acc: 0.3154

Epoch 1/53
----------
train Loss: 0.0234 Acc: 0.3443
val Loss: 0.0180 Acc: 0.5538

Epoch 2/53
----------
train Loss: 0.0131 Acc: 0.6459
val Loss: 0.0122 Acc: 0.7231

Epoch 3/53
----------
train Loss: 0.0095 Acc: 0.7738
val Loss: 0.0124 Acc: 0.7154

Epoch 4/53
----------
train Loss: 0.0104 Acc: 0.7443
val Loss: 0.0101 Acc: 0.7385

Epoch 5/53
----------
train Loss: 0.0076 Acc: 0.7574
val Loss: 0.0124 Acc: 0.6923

Epoch 6/53
----------
train Loss: 0.0087 Acc: 0.8164
val Loss: 0.0080 Acc: 0.8154

Epoch 7/53
----------
train Loss: 0.0095 Acc: 0.8230
val Loss: 0.0082 Acc: 0.8231

Epoch 8/53
----------
train Loss: 0.0060 Acc: 0.8426
val Loss: 0.0122 Acc: 0.7385

Epoch 9/53
----------
train Loss: 0.0129 Acc: 0.7836
val Loss: 0.0120 Acc: 0.7000

Epoch 10/53
----------
train Loss: 0.0137 Acc: 0.6689
val Loss: 0.0094 Acc: 0.7615

Epoch 11/53
----------
train Loss: 0.0087 Acc: 0.7705
val Loss: 0.0191 Acc: 0.6769

Epoch 12/53
----------
LR is set to 0.001
train Loss: 0.0171 Acc: 0.7180
val Loss: 0.0123 Acc: 0.7000

Epoch 13/53
----------
train Loss: 0.0062 Acc: 0.7967
val Loss: 0.0099 Acc: 0.8154

Epoch 14/53
----------
train Loss: 0.0035 Acc: 0.8787
val Loss: 0.0090 Acc: 0.8077

Epoch 15/53
----------
train Loss: 0.0047 Acc: 0.9148
val Loss: 0.0092 Acc: 0.8154

Epoch 16/53
----------
train Loss: 0.0056 Acc: 0.9016
val Loss: 0.0085 Acc: 0.8154

Epoch 17/53
----------
train Loss: 0.0028 Acc: 0.9180
val Loss: 0.0086 Acc: 0.8077

Epoch 18/53
----------
train Loss: 0.0054 Acc: 0.9246
val Loss: 0.0077 Acc: 0.8231

Epoch 19/53
----------
train Loss: 0.0036 Acc: 0.9246
val Loss: 0.0086 Acc: 0.8308

Epoch 20/53
----------
train Loss: 0.0074 Acc: 0.9246
val Loss: 0.0084 Acc: 0.8462

Epoch 21/53
----------
train Loss: 0.0025 Acc: 0.9443
val Loss: 0.0066 Acc: 0.8462

Epoch 22/53
----------
train Loss: 0.0045 Acc: 0.9148
val Loss: 0.0084 Acc: 0.8231

Epoch 23/53
----------
train Loss: 0.0057 Acc: 0.9344
val Loss: 0.0073 Acc: 0.8231

Epoch 24/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9180
val Loss: 0.0089 Acc: 0.8231

Epoch 25/53
----------
train Loss: 0.0037 Acc: 0.9213
val Loss: 0.0092 Acc: 0.8231

Epoch 26/53
----------
train Loss: 0.0066 Acc: 0.9180
val Loss: 0.0084 Acc: 0.8231

Epoch 27/53
----------
train Loss: 0.0046 Acc: 0.9115
val Loss: 0.0088 Acc: 0.8231

Epoch 28/53
----------
train Loss: 0.0051 Acc: 0.9016
val Loss: 0.0074 Acc: 0.8231

Epoch 29/53
----------
train Loss: 0.0047 Acc: 0.9410
val Loss: 0.0065 Acc: 0.8308

Epoch 30/53
----------
train Loss: 0.0032 Acc: 0.9246
val Loss: 0.0083 Acc: 0.8308

Epoch 31/53
----------
train Loss: 0.0063 Acc: 0.9344
val Loss: 0.0080 Acc: 0.8231

Epoch 32/53
----------
train Loss: 0.0024 Acc: 0.9410
val Loss: 0.0076 Acc: 0.8231

Epoch 33/53
----------
train Loss: 0.0047 Acc: 0.9246
val Loss: 0.0078 Acc: 0.8231

Epoch 34/53
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0090 Acc: 0.8308

Epoch 35/53
----------
train Loss: 0.0042 Acc: 0.9410
val Loss: 0.0079 Acc: 0.8385

Epoch 36/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0044 Acc: 0.9377
val Loss: 0.0073 Acc: 0.8462

Epoch 37/53
----------
train Loss: 0.0032 Acc: 0.9213
val Loss: 0.0079 Acc: 0.8538

Epoch 38/53
----------
train Loss: 0.0043 Acc: 0.9246
val Loss: 0.0091 Acc: 0.8538

Epoch 39/53
----------
train Loss: 0.0028 Acc: 0.9180
val Loss: 0.0079 Acc: 0.8538

Epoch 40/53
----------
train Loss: 0.0024 Acc: 0.9574
val Loss: 0.0081 Acc: 0.8462

Epoch 41/53
----------
train Loss: 0.0048 Acc: 0.9377
val Loss: 0.0085 Acc: 0.8462

Epoch 42/53
----------
train Loss: 0.0032 Acc: 0.9443
val Loss: 0.0068 Acc: 0.8538

Epoch 43/53
----------
train Loss: 0.0036 Acc: 0.9344
val Loss: 0.0076 Acc: 0.8538

Epoch 44/53
----------
train Loss: 0.0052 Acc: 0.9279
val Loss: 0.0060 Acc: 0.8462

Epoch 45/53
----------
train Loss: 0.0029 Acc: 0.9213
val Loss: 0.0063 Acc: 0.8462

Epoch 46/53
----------
train Loss: 0.0025 Acc: 0.9475
val Loss: 0.0069 Acc: 0.8538

Epoch 47/53
----------
train Loss: 0.0037 Acc: 0.9475
val Loss: 0.0075 Acc: 0.8385

Epoch 48/53
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0037 Acc: 0.9311
val Loss: 0.0083 Acc: 0.8462

Epoch 49/53
----------
train Loss: 0.0024 Acc: 0.9344
val Loss: 0.0093 Acc: 0.8231

Epoch 50/53
----------
train Loss: 0.0054 Acc: 0.9279
val Loss: 0.0079 Acc: 0.8462

Epoch 51/53
----------
train Loss: 0.0035 Acc: 0.9410
val Loss: 0.0080 Acc: 0.8462

Epoch 52/53
----------
train Loss: 0.0051 Acc: 0.9115
val Loss: 0.0071 Acc: 0.8462

Epoch 53/53
----------
train Loss: 0.0031 Acc: 0.9508
val Loss: 0.0088 Acc: 0.8385

Training complete in 2m 2s
Best val Acc: 0.853846

---Fine tuning.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0039 Acc: 0.9410
val Loss: 0.0115 Acc: 0.8077

Epoch 1/53
----------
train Loss: 0.0060 Acc: 0.8852
val Loss: 0.0253 Acc: 0.4923

Epoch 2/53
----------
train Loss: 0.0055 Acc: 0.8918
val Loss: 0.0362 Acc: 0.4308

Epoch 3/53
----------
train Loss: 0.0093 Acc: 0.8393
val Loss: 0.0459 Acc: 0.4462

Epoch 4/53
----------
train Loss: 0.0088 Acc: 0.8098
val Loss: 0.0757 Acc: 0.3154

Epoch 5/53
----------
train Loss: 0.0150 Acc: 0.7803
val Loss: 0.0639 Acc: 0.3231

Epoch 6/53
----------
train Loss: 0.0108 Acc: 0.7115
val Loss: 0.0442 Acc: 0.4077

Epoch 7/53
----------
train Loss: 0.0109 Acc: 0.7574
val Loss: 0.0341 Acc: 0.5769

Epoch 8/53
----------
train Loss: 0.0077 Acc: 0.8426
val Loss: 0.0520 Acc: 0.5231

Epoch 9/53
----------
train Loss: 0.0111 Acc: 0.8295
val Loss: 0.0827 Acc: 0.4923

Epoch 10/53
----------
train Loss: 0.0099 Acc: 0.7967
val Loss: 0.0921 Acc: 0.3308

Epoch 11/53
----------
train Loss: 0.0193 Acc: 0.7180
val Loss: 0.1481 Acc: 0.4077

Epoch 12/53
----------
LR is set to 0.001
train Loss: 0.0041 Acc: 0.8623
val Loss: 0.0699 Acc: 0.5308

Epoch 13/53
----------
train Loss: 0.0069 Acc: 0.9016
val Loss: 0.0389 Acc: 0.6231

Epoch 14/53
----------
train Loss: 0.0028 Acc: 0.9082
val Loss: 0.0243 Acc: 0.7077

Epoch 15/53
----------
train Loss: 0.0084 Acc: 0.9410
val Loss: 0.0238 Acc: 0.7231

Epoch 16/53
----------
train Loss: 0.0033 Acc: 0.9541
val Loss: 0.0193 Acc: 0.7462

Epoch 17/53
----------
train Loss: 0.0045 Acc: 0.9639
val Loss: 0.0141 Acc: 0.7615

Epoch 18/53
----------
train Loss: 0.0018 Acc: 0.9639
val Loss: 0.0170 Acc: 0.7462

Epoch 19/53
----------
train Loss: 0.0016 Acc: 0.9607
val Loss: 0.0167 Acc: 0.7692

Epoch 20/53
----------
train Loss: 0.0032 Acc: 0.9344
val Loss: 0.0165 Acc: 0.7769

Epoch 21/53
----------
train Loss: 0.0022 Acc: 0.9705
val Loss: 0.0220 Acc: 0.8077

Epoch 22/53
----------
train Loss: 0.0066 Acc: 0.9541
val Loss: 0.0172 Acc: 0.7846

Epoch 23/53
----------
train Loss: 0.0012 Acc: 0.9705
val Loss: 0.0155 Acc: 0.7846

Epoch 24/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0074 Acc: 0.9639
val Loss: 0.0209 Acc: 0.7923

Epoch 25/53
----------
train Loss: 0.0047 Acc: 0.9639
val Loss: 0.0140 Acc: 0.7846

Epoch 26/53
----------
train Loss: 0.0015 Acc: 0.9672
val Loss: 0.0158 Acc: 0.7923

Epoch 27/53
----------
train Loss: 0.0072 Acc: 0.9639
val Loss: 0.0181 Acc: 0.7923

Epoch 28/53
----------
train Loss: 0.0007 Acc: 0.9738
val Loss: 0.0183 Acc: 0.7846

Epoch 29/53
----------
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0130 Acc: 0.7923

Epoch 30/53
----------
train Loss: 0.0096 Acc: 0.9738
val Loss: 0.0137 Acc: 0.7923

Epoch 31/53
----------
train Loss: 0.0022 Acc: 0.9803
val Loss: 0.0147 Acc: 0.8077

Epoch 32/53
----------
train Loss: 0.0017 Acc: 0.9770
val Loss: 0.0146 Acc: 0.8077

Epoch 33/53
----------
train Loss: 0.0034 Acc: 0.9803
val Loss: 0.0206 Acc: 0.8000

Epoch 34/53
----------
train Loss: 0.0018 Acc: 0.9836
val Loss: 0.0166 Acc: 0.8154

Epoch 35/53
----------
train Loss: 0.0022 Acc: 0.9836
val Loss: 0.0165 Acc: 0.8000

Epoch 36/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0007 Acc: 0.9836
val Loss: 0.0146 Acc: 0.8231

Epoch 37/53
----------
train Loss: 0.0015 Acc: 0.9869
val Loss: 0.0153 Acc: 0.8000

Epoch 38/53
----------
train Loss: 0.0007 Acc: 0.9934
val Loss: 0.0113 Acc: 0.8000

Epoch 39/53
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0133 Acc: 0.8308

Epoch 40/53
----------
train Loss: 0.0020 Acc: 0.9836
val Loss: 0.0156 Acc: 0.8000

Epoch 41/53
----------
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0154 Acc: 0.8231

Epoch 42/53
----------
train Loss: 0.0134 Acc: 0.9705
val Loss: 0.0138 Acc: 0.8154

Epoch 43/53
----------
train Loss: 0.0011 Acc: 0.9770
val Loss: 0.0168 Acc: 0.8077

Epoch 44/53
----------
train Loss: 0.0024 Acc: 0.9836
val Loss: 0.0127 Acc: 0.8154

Epoch 45/53
----------
train Loss: 0.0008 Acc: 0.9902
val Loss: 0.0172 Acc: 0.8231

Epoch 46/53
----------
train Loss: 0.0010 Acc: 0.9902
val Loss: 0.0171 Acc: 0.8000

Epoch 47/53
----------
train Loss: 0.0019 Acc: 0.9770
val Loss: 0.0163 Acc: 0.8077

Epoch 48/53
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0030 Acc: 0.9803
val Loss: 0.0180 Acc: 0.7923

Epoch 49/53
----------
train Loss: 0.0017 Acc: 0.9902
val Loss: 0.0193 Acc: 0.8000

Epoch 50/53
----------
train Loss: 0.0025 Acc: 0.9770
val Loss: 0.0151 Acc: 0.8000

Epoch 51/53
----------
train Loss: 0.0035 Acc: 0.9902
val Loss: 0.0239 Acc: 0.8231

Epoch 52/53
----------
train Loss: 0.0030 Acc: 0.9869
val Loss: 0.0159 Acc: 0.8231

Epoch 53/53
----------
train Loss: 0.0011 Acc: 0.9902
val Loss: 0.0185 Acc: 0.7923

Training complete in 2m 13s
Best val Acc: 0.830769

---Testing---
Test accuracy: 0.944828
--------------------
Accuracy of Dasyatiformes : 89 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 88 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 96 %
mean: 0.9339030525697499, std: 0.03611497873337058

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 56, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0165 Acc: 0.3061
val Loss: 0.0443 Acc: 0.3953

Epoch 1/55
----------
train Loss: 0.0127 Acc: 0.5740
val Loss: 0.0282 Acc: 0.6047

Epoch 2/55
----------
train Loss: 0.0090 Acc: 0.6811
val Loss: 0.0204 Acc: 0.6279

Epoch 3/55
----------
train Loss: 0.0077 Acc: 0.7474
val Loss: 0.0183 Acc: 0.6744

Epoch 4/55
----------
LR is set to 0.001
train Loss: 0.0062 Acc: 0.7704
val Loss: 0.0174 Acc: 0.6744

Epoch 5/55
----------
train Loss: 0.0055 Acc: 0.8010
val Loss: 0.0160 Acc: 0.7209

Epoch 6/55
----------
train Loss: 0.0054 Acc: 0.8189
val Loss: 0.0160 Acc: 0.7907

Epoch 7/55
----------
train Loss: 0.0054 Acc: 0.8163
val Loss: 0.0161 Acc: 0.7674

Epoch 8/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0055 Acc: 0.8291
val Loss: 0.0161 Acc: 0.7674

Epoch 9/55
----------
train Loss: 0.0056 Acc: 0.8265
val Loss: 0.0160 Acc: 0.7907

Epoch 10/55
----------
train Loss: 0.0056 Acc: 0.8112
val Loss: 0.0159 Acc: 0.7907

Epoch 11/55
----------
train Loss: 0.0052 Acc: 0.8393
val Loss: 0.0158 Acc: 0.7907

Epoch 12/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0051 Acc: 0.8673
val Loss: 0.0157 Acc: 0.7907

Epoch 13/55
----------
train Loss: 0.0055 Acc: 0.8291
val Loss: 0.0158 Acc: 0.7907

Epoch 14/55
----------
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0157 Acc: 0.7907

Epoch 15/55
----------
train Loss: 0.0053 Acc: 0.8367
val Loss: 0.0157 Acc: 0.7907

Epoch 16/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0054 Acc: 0.8112
val Loss: 0.0157 Acc: 0.7907

Epoch 17/55
----------
train Loss: 0.0054 Acc: 0.8112
val Loss: 0.0157 Acc: 0.7907

Epoch 18/55
----------
train Loss: 0.0053 Acc: 0.8469
val Loss: 0.0157 Acc: 0.7907

Epoch 19/55
----------
train Loss: 0.0054 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 20/55
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0055 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 21/55
----------
train Loss: 0.0052 Acc: 0.8265
val Loss: 0.0157 Acc: 0.7907

Epoch 22/55
----------
train Loss: 0.0053 Acc: 0.8240
val Loss: 0.0157 Acc: 0.7907

Epoch 23/55
----------
train Loss: 0.0053 Acc: 0.8291
val Loss: 0.0157 Acc: 0.7907

Epoch 24/55
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0052 Acc: 0.8240
val Loss: 0.0157 Acc: 0.7907

Epoch 25/55
----------
train Loss: 0.0050 Acc: 0.8597
val Loss: 0.0157 Acc: 0.7907

Epoch 26/55
----------
train Loss: 0.0053 Acc: 0.8265
val Loss: 0.0157 Acc: 0.7907

Epoch 27/55
----------
train Loss: 0.0052 Acc: 0.8418
val Loss: 0.0158 Acc: 0.7907

Epoch 28/55
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0054 Acc: 0.8138
val Loss: 0.0157 Acc: 0.7907

Epoch 29/55
----------
train Loss: 0.0051 Acc: 0.8546
val Loss: 0.0158 Acc: 0.7907

Epoch 30/55
----------
train Loss: 0.0053 Acc: 0.8291
val Loss: 0.0158 Acc: 0.7907

Epoch 31/55
----------
train Loss: 0.0053 Acc: 0.8520
val Loss: 0.0158 Acc: 0.7907

Epoch 32/55
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0051 Acc: 0.8469
val Loss: 0.0158 Acc: 0.7907

Epoch 33/55
----------
train Loss: 0.0054 Acc: 0.8240
val Loss: 0.0157 Acc: 0.7907

Epoch 34/55
----------
train Loss: 0.0053 Acc: 0.8316
val Loss: 0.0157 Acc: 0.8140

Epoch 35/55
----------
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0157 Acc: 0.7907

Epoch 36/55
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0052 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 37/55
----------
train Loss: 0.0052 Acc: 0.8444
val Loss: 0.0158 Acc: 0.7907

Epoch 38/55
----------
train Loss: 0.0053 Acc: 0.8342
val Loss: 0.0158 Acc: 0.7907

Epoch 39/55
----------
train Loss: 0.0054 Acc: 0.8367
val Loss: 0.0158 Acc: 0.7907

Epoch 40/55
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0051 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 41/55
----------
train Loss: 0.0054 Acc: 0.8342
val Loss: 0.0158 Acc: 0.7907

Epoch 42/55
----------
train Loss: 0.0054 Acc: 0.8061
val Loss: 0.0157 Acc: 0.7907

Epoch 43/55
----------
train Loss: 0.0053 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 44/55
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0052 Acc: 0.8291
val Loss: 0.0157 Acc: 0.8140

Epoch 45/55
----------
train Loss: 0.0052 Acc: 0.8495
val Loss: 0.0156 Acc: 0.8140

Epoch 46/55
----------
train Loss: 0.0054 Acc: 0.8316
val Loss: 0.0156 Acc: 0.7907

Epoch 47/55
----------
train Loss: 0.0050 Acc: 0.8622
val Loss: 0.0156 Acc: 0.7907

Epoch 48/55
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0054 Acc: 0.8265
val Loss: 0.0157 Acc: 0.7907

Epoch 49/55
----------
train Loss: 0.0051 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Epoch 50/55
----------
train Loss: 0.0055 Acc: 0.8240
val Loss: 0.0157 Acc: 0.8140

Epoch 51/55
----------
train Loss: 0.0054 Acc: 0.8495
val Loss: 0.0157 Acc: 0.7907

Epoch 52/55
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0055 Acc: 0.8265
val Loss: 0.0157 Acc: 0.7907

Epoch 53/55
----------
train Loss: 0.0054 Acc: 0.8444
val Loss: 0.0157 Acc: 0.7907

Epoch 54/55
----------
train Loss: 0.0051 Acc: 0.8316
val Loss: 0.0158 Acc: 0.7907

Epoch 55/55
----------
train Loss: 0.0052 Acc: 0.8342
val Loss: 0.0157 Acc: 0.7907

Training complete in 1m 57s
Best val Acc: 0.813953

---Fine tuning.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8214
val Loss: 0.0125 Acc: 0.8605

Epoch 1/55
----------
train Loss: 0.0036 Acc: 0.9107
val Loss: 0.0123 Acc: 0.8605

Epoch 2/55
----------
train Loss: 0.0019 Acc: 0.9694
val Loss: 0.0110 Acc: 0.8605

Epoch 3/55
----------
train Loss: 0.0011 Acc: 0.9949
val Loss: 0.0115 Acc: 0.8605

Epoch 4/55
----------
LR is set to 0.001
train Loss: 0.0007 Acc: 0.9923
val Loss: 0.0117 Acc: 0.8605

Epoch 5/55
----------
train Loss: 0.0007 Acc: 0.9898
val Loss: 0.0118 Acc: 0.8605

Epoch 6/55
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0119 Acc: 0.8605

Epoch 7/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 8/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 9/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 10/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 11/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 12/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 13/55
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 14/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0121 Acc: 0.8605

Epoch 15/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 16/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 17/55
----------
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 18/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 19/55
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0120 Acc: 0.8605

Epoch 20/55
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0120 Acc: 0.8605

Epoch 21/55
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 22/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 23/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 24/55
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9898
val Loss: 0.0120 Acc: 0.8605

Epoch 25/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 26/55
----------
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 27/55
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0120 Acc: 0.8605

Epoch 28/55
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 29/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 30/55
----------
train Loss: 0.0004 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 31/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 32/55
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 33/55
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 34/55
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 35/55
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 36/55
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0120 Acc: 0.8605

Epoch 37/55
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 38/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0119 Acc: 0.8605

Epoch 39/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 40/55
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0006 Acc: 0.9898
val Loss: 0.0120 Acc: 0.8605

Epoch 41/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0119 Acc: 0.8605

Epoch 42/55
----------
train Loss: 0.0005 Acc: 0.9923
val Loss: 0.0120 Acc: 0.8605

Epoch 43/55
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 44/55
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 45/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 46/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Epoch 47/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 48/55
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0121 Acc: 0.8605

Epoch 49/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 50/55
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0121 Acc: 0.8605

Epoch 51/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 52/55
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0120 Acc: 0.8605

Epoch 53/55
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0120 Acc: 0.8605

Epoch 54/55
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0121 Acc: 0.8605

Epoch 55/55
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8605

Training complete in 2m 10s
Best val Acc: 0.860465

---Testing---
Test accuracy: 0.905747
--------------------
Accuracy of Dasyatiformes : 86 %
Accuracy of Myliobatiformes : 84 %
Accuracy of Rajiformes : 82 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of Torpediniformes : 95 %
mean: 0.8882066225751585, std: 0.058483924909518016
--------------------

run info[val: 0.15, epoch: 73, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0186 Acc: 0.3000
val Loss: 0.0214 Acc: 0.4923

Epoch 1/72
----------
train Loss: 0.0140 Acc: 0.5054
val Loss: 0.0221 Acc: 0.4000

Epoch 2/72
----------
train Loss: 0.0108 Acc: 0.6027
val Loss: 0.0101 Acc: 0.8000

Epoch 3/72
----------
train Loss: 0.0073 Acc: 0.8027
val Loss: 0.0099 Acc: 0.7231

Epoch 4/72
----------
train Loss: 0.0058 Acc: 0.8054
val Loss: 0.0078 Acc: 0.8154

Epoch 5/72
----------
train Loss: 0.0050 Acc: 0.8135
val Loss: 0.0068 Acc: 0.9077

Epoch 6/72
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.8649
val Loss: 0.0067 Acc: 0.8923

Epoch 7/72
----------
train Loss: 0.0039 Acc: 0.8730
val Loss: 0.0068 Acc: 0.8769

Epoch 8/72
----------
train Loss: 0.0040 Acc: 0.8811
val Loss: 0.0070 Acc: 0.8615

Epoch 9/72
----------
train Loss: 0.0040 Acc: 0.8865
val Loss: 0.0069 Acc: 0.8615

Epoch 10/72
----------
train Loss: 0.0039 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8769

Epoch 11/72
----------
train Loss: 0.0040 Acc: 0.8892
val Loss: 0.0067 Acc: 0.8923

Epoch 12/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0036 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8923

Epoch 13/72
----------
train Loss: 0.0040 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8923

Epoch 14/72
----------
train Loss: 0.0039 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8923

Epoch 15/72
----------
train Loss: 0.0038 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 16/72
----------
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8923

Epoch 17/72
----------
train Loss: 0.0038 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 18/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0039 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 19/72
----------
train Loss: 0.0038 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 20/72
----------
train Loss: 0.0039 Acc: 0.8757
val Loss: 0.0068 Acc: 0.8923

Epoch 21/72
----------
train Loss: 0.0039 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 22/72
----------
train Loss: 0.0037 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 23/72
----------
train Loss: 0.0039 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 24/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0038 Acc: 0.8838
val Loss: 0.0068 Acc: 0.8769

Epoch 25/72
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8923

Epoch 26/72
----------
train Loss: 0.0038 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8923

Epoch 27/72
----------
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8923

Epoch 28/72
----------
train Loss: 0.0039 Acc: 0.8838
val Loss: 0.0068 Acc: 0.8923

Epoch 29/72
----------
train Loss: 0.0038 Acc: 0.9216
val Loss: 0.0068 Acc: 0.8923

Epoch 30/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0039 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 31/72
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8923

Epoch 32/72
----------
train Loss: 0.0038 Acc: 0.8784
val Loss: 0.0068 Acc: 0.8769

Epoch 33/72
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 34/72
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 35/72
----------
train Loss: 0.0037 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8923

Epoch 36/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0037 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 37/72
----------
train Loss: 0.0038 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8923

Epoch 38/72
----------
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8923

Epoch 39/72
----------
train Loss: 0.0037 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8923

Epoch 40/72
----------
train Loss: 0.0039 Acc: 0.8838
val Loss: 0.0068 Acc: 0.8923

Epoch 41/72
----------
train Loss: 0.0036 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8923

Epoch 42/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0037 Acc: 0.8811
val Loss: 0.0068 Acc: 0.8923

Epoch 43/72
----------
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8923

Epoch 44/72
----------
train Loss: 0.0037 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8923

Epoch 45/72
----------
train Loss: 0.0037 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 46/72
----------
train Loss: 0.0037 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 47/72
----------
train Loss: 0.0036 Acc: 0.9189
val Loss: 0.0068 Acc: 0.8923

Epoch 48/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0038 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8923

Epoch 49/72
----------
train Loss: 0.0037 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8923

Epoch 50/72
----------
train Loss: 0.0038 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8923

Epoch 51/72
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8923

Epoch 52/72
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 53/72
----------
train Loss: 0.0039 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 54/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0040 Acc: 0.8730
val Loss: 0.0068 Acc: 0.8769

Epoch 55/72
----------
train Loss: 0.0037 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8769

Epoch 56/72
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 57/72
----------
train Loss: 0.0038 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 58/72
----------
train Loss: 0.0036 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 59/72
----------
train Loss: 0.0036 Acc: 0.9027
val Loss: 0.0068 Acc: 0.8923

Epoch 60/72
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0037 Acc: 0.9135
val Loss: 0.0068 Acc: 0.8923

Epoch 61/72
----------
train Loss: 0.0036 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8923

Epoch 62/72
----------
train Loss: 0.0038 Acc: 0.9000
val Loss: 0.0068 Acc: 0.8923

Epoch 63/72
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0068 Acc: 0.8923

Epoch 64/72
----------
train Loss: 0.0041 Acc: 0.8946
val Loss: 0.0068 Acc: 0.8923

Epoch 65/72
----------
train Loss: 0.0037 Acc: 0.9054
val Loss: 0.0068 Acc: 0.8923

Epoch 66/72
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0039 Acc: 0.8892
val Loss: 0.0068 Acc: 0.8923

Epoch 67/72
----------
train Loss: 0.0038 Acc: 0.8892
val Loss: 0.0067 Acc: 0.8923

Epoch 68/72
----------
train Loss: 0.0038 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8923

Epoch 69/72
----------
train Loss: 0.0039 Acc: 0.8865
val Loss: 0.0068 Acc: 0.8923

Epoch 70/72
----------
train Loss: 0.0035 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8923

Epoch 71/72
----------
train Loss: 0.0037 Acc: 0.9108
val Loss: 0.0068 Acc: 0.8769

Epoch 72/72
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0036 Acc: 0.9081
val Loss: 0.0068 Acc: 0.8769

Training complete in 2m 36s
Best val Acc: 0.907692

---Fine tuning.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0043 Acc: 0.8649
val Loss: 0.0062 Acc: 0.9231

Epoch 1/72
----------
train Loss: 0.0023 Acc: 0.9676
val Loss: 0.0050 Acc: 0.9385

Epoch 2/72
----------
train Loss: 0.0010 Acc: 0.9946
val Loss: 0.0052 Acc: 0.9077

Epoch 3/72
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0061 Acc: 0.9077

Epoch 4/72
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0063 Acc: 0.9231

Epoch 5/72
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0057 Acc: 0.9077

Epoch 6/72
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8923

Epoch 7/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 8/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8923

Epoch 9/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 10/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 11/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 12/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 13/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 14/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 15/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9077

Epoch 16/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 17/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 18/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 19/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 20/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 21/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 22/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 23/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 24/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 25/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 26/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 27/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 28/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 29/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 30/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 31/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 32/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 33/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 34/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 35/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 36/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 37/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 38/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 39/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 40/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 41/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 42/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 43/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 44/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 45/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 46/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 47/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 48/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 49/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 50/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 51/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 52/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 53/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 54/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 55/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 56/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 57/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 58/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 59/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 60/72
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 61/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 62/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 63/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 64/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 65/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 66/72
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 67/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 68/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 69/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 70/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 71/72
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Epoch 72/72
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9077

Training complete in 2m 50s
Best val Acc: 0.938462

---Testing---
Test accuracy: 0.986207
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 96 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 100 %
Accuracy of Torpediniformes : 99 %
mean: 0.981402212062967, std: 0.01583730627609277
--------------------

run info[val: 0.2, epoch: 57, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0183 Acc: 0.3161
val Loss: 0.0158 Acc: 0.4368

Epoch 1/56
----------
train Loss: 0.0145 Acc: 0.4943
val Loss: 0.0141 Acc: 0.4483

Epoch 2/56
----------
train Loss: 0.0123 Acc: 0.5805
val Loss: 0.0097 Acc: 0.7586

Epoch 3/56
----------
train Loss: 0.0094 Acc: 0.6868
val Loss: 0.0092 Acc: 0.7011

Epoch 4/56
----------
train Loss: 0.0075 Acc: 0.7730
val Loss: 0.0066 Acc: 0.8161

Epoch 5/56
----------
train Loss: 0.0062 Acc: 0.8132
val Loss: 0.0066 Acc: 0.8276

Epoch 6/56
----------
train Loss: 0.0051 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7931

Epoch 7/56
----------
train Loss: 0.0038 Acc: 0.8879
val Loss: 0.0055 Acc: 0.8621

Epoch 8/56
----------
train Loss: 0.0046 Acc: 0.8420
val Loss: 0.0057 Acc: 0.8391

Epoch 9/56
----------
train Loss: 0.0046 Acc: 0.8592
val Loss: 0.0057 Acc: 0.8276

Epoch 10/56
----------
LR is set to 0.001
train Loss: 0.0036 Acc: 0.9080
val Loss: 0.0055 Acc: 0.8506

Epoch 11/56
----------
train Loss: 0.0032 Acc: 0.9080
val Loss: 0.0054 Acc: 0.8506

Epoch 12/56
----------
train Loss: 0.0031 Acc: 0.9109
val Loss: 0.0053 Acc: 0.8506

Epoch 13/56
----------
train Loss: 0.0033 Acc: 0.9109
val Loss: 0.0052 Acc: 0.8621

Epoch 14/56
----------
train Loss: 0.0027 Acc: 0.9253
val Loss: 0.0052 Acc: 0.8736

Epoch 15/56
----------
train Loss: 0.0026 Acc: 0.9540
val Loss: 0.0053 Acc: 0.8736

Epoch 16/56
----------
train Loss: 0.0029 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8736

Epoch 17/56
----------
train Loss: 0.0032 Acc: 0.9167
val Loss: 0.0053 Acc: 0.8736

Epoch 18/56
----------
train Loss: 0.0029 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 19/56
----------
train Loss: 0.0030 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8736

Epoch 20/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0030 Acc: 0.9167
val Loss: 0.0053 Acc: 0.8736

Epoch 21/56
----------
train Loss: 0.0027 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8736

Epoch 22/56
----------
train Loss: 0.0030 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 23/56
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 24/56
----------
train Loss: 0.0031 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8736

Epoch 25/56
----------
train Loss: 0.0035 Acc: 0.9339
val Loss: 0.0053 Acc: 0.8736

Epoch 26/56
----------
train Loss: 0.0030 Acc: 0.9368
val Loss: 0.0052 Acc: 0.8736

Epoch 27/56
----------
train Loss: 0.0027 Acc: 0.9339
val Loss: 0.0052 Acc: 0.8736

Epoch 28/56
----------
train Loss: 0.0031 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8736

Epoch 29/56
----------
train Loss: 0.0033 Acc: 0.9138
val Loss: 0.0053 Acc: 0.8736

Epoch 30/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0031 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 31/56
----------
train Loss: 0.0033 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 32/56
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8621

Epoch 33/56
----------
train Loss: 0.0028 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 34/56
----------
train Loss: 0.0033 Acc: 0.9310
val Loss: 0.0052 Acc: 0.8736

Epoch 35/56
----------
train Loss: 0.0027 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 36/56
----------
train Loss: 0.0029 Acc: 0.9224
val Loss: 0.0052 Acc: 0.8736

Epoch 37/56
----------
train Loss: 0.0028 Acc: 0.9425
val Loss: 0.0052 Acc: 0.8736

Epoch 38/56
----------
train Loss: 0.0026 Acc: 0.9425
val Loss: 0.0053 Acc: 0.8736

Epoch 39/56
----------
train Loss: 0.0027 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 40/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0029 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 41/56
----------
train Loss: 0.0030 Acc: 0.9138
val Loss: 0.0052 Acc: 0.8736

Epoch 42/56
----------
train Loss: 0.0027 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8621

Epoch 43/56
----------
train Loss: 0.0028 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Epoch 44/56
----------
train Loss: 0.0032 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 45/56
----------
train Loss: 0.0028 Acc: 0.9282
val Loss: 0.0053 Acc: 0.8736

Epoch 46/56
----------
train Loss: 0.0028 Acc: 0.9397
val Loss: 0.0053 Acc: 0.8736

Epoch 47/56
----------
train Loss: 0.0029 Acc: 0.9195
val Loss: 0.0052 Acc: 0.8736

Epoch 48/56
----------
train Loss: 0.0030 Acc: 0.9195
val Loss: 0.0053 Acc: 0.8736

Epoch 49/56
----------
train Loss: 0.0031 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 50/56
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0027 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 51/56
----------
train Loss: 0.0029 Acc: 0.9224
val Loss: 0.0053 Acc: 0.8736

Epoch 52/56
----------
train Loss: 0.0028 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8736

Epoch 53/56
----------
train Loss: 0.0031 Acc: 0.9167
val Loss: 0.0052 Acc: 0.8736

Epoch 54/56
----------
train Loss: 0.0028 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8736

Epoch 55/56
----------
train Loss: 0.0029 Acc: 0.9253
val Loss: 0.0053 Acc: 0.8736

Epoch 56/56
----------
train Loss: 0.0030 Acc: 0.9310
val Loss: 0.0053 Acc: 0.8621

Training complete in 2m 3s
Best val Acc: 0.873563

---Fine tuning.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0032 Acc: 0.9109
val Loss: 0.0049 Acc: 0.8621

Epoch 1/56
----------
train Loss: 0.0018 Acc: 0.9626
val Loss: 0.0054 Acc: 0.8506

Epoch 2/56
----------
train Loss: 0.0010 Acc: 0.9971
val Loss: 0.0048 Acc: 0.8736

Epoch 3/56
----------
train Loss: 0.0007 Acc: 0.9971
val Loss: 0.0044 Acc: 0.8736

Epoch 4/56
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 5/56
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 6/56
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0040 Acc: 0.8966

Epoch 7/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9080

Epoch 8/56
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0039 Acc: 0.9195

Epoch 9/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 10/56
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9195

Epoch 11/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 12/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 13/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 14/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9195

Epoch 15/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 16/56
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0041 Acc: 0.9195

Epoch 17/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 18/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 19/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 20/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 21/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 22/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 23/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 24/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 25/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 26/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 27/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 28/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 29/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 30/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 31/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 32/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 33/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 34/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 35/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 36/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 37/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 38/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9195

Epoch 39/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 40/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 41/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 42/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 43/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 44/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 45/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 46/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 47/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 48/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 49/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 50/56
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 51/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 52/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 53/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 54/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 55/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Epoch 56/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9195

Training complete in 2m 17s
Best val Acc: 0.919540

---Testing---
Test accuracy: 0.983908
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 99 %
mean: 0.9792656282217418, std: 0.010020501001622061
--------------------

run info[val: 0.25, epoch: 94, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0202 Acc: 0.3180
val Loss: 0.0265 Acc: 0.5463

Epoch 1/93
----------
train Loss: 0.0165 Acc: 0.5291
val Loss: 0.0214 Acc: 0.6204

Epoch 2/93
----------
train Loss: 0.0114 Acc: 0.6422
val Loss: 0.0140 Acc: 0.7222

Epoch 3/93
----------
train Loss: 0.0089 Acc: 0.7554
val Loss: 0.0128 Acc: 0.8148

Epoch 4/93
----------
train Loss: 0.0066 Acc: 0.8226
val Loss: 0.0109 Acc: 0.7870

Epoch 5/93
----------
train Loss: 0.0063 Acc: 0.8043
val Loss: 0.0084 Acc: 0.8333

Epoch 6/93
----------
train Loss: 0.0049 Acc: 0.8593
val Loss: 0.0118 Acc: 0.7778

Epoch 7/93
----------
train Loss: 0.0051 Acc: 0.8287
val Loss: 0.0078 Acc: 0.8519

Epoch 8/93
----------
train Loss: 0.0050 Acc: 0.8593
val Loss: 0.0093 Acc: 0.8611

Epoch 9/93
----------
train Loss: 0.0040 Acc: 0.8746
val Loss: 0.0088 Acc: 0.8148

Epoch 10/93
----------
train Loss: 0.0037 Acc: 0.8930
val Loss: 0.0086 Acc: 0.8426

Epoch 11/93
----------
train Loss: 0.0027 Acc: 0.9205
val Loss: 0.0096 Acc: 0.7963

Epoch 12/93
----------
LR is set to 0.001
train Loss: 0.0027 Acc: 0.9388
val Loss: 0.0088 Acc: 0.7963

Epoch 13/93
----------
train Loss: 0.0024 Acc: 0.9419
val Loss: 0.0135 Acc: 0.8241

Epoch 14/93
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0124 Acc: 0.8519

Epoch 15/93
----------
train Loss: 0.0024 Acc: 0.9388
val Loss: 0.0053 Acc: 0.8704

Epoch 16/93
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0095 Acc: 0.8704

Epoch 17/93
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0069 Acc: 0.8796

Epoch 18/93
----------
train Loss: 0.0025 Acc: 0.9602
val Loss: 0.0056 Acc: 0.8611

Epoch 19/93
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0054 Acc: 0.8611

Epoch 20/93
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0059 Acc: 0.8611

Epoch 21/93
----------
train Loss: 0.0023 Acc: 0.9725
val Loss: 0.0080 Acc: 0.8704

Epoch 22/93
----------
train Loss: 0.0026 Acc: 0.9572
val Loss: 0.0095 Acc: 0.8611

Epoch 23/93
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0054 Acc: 0.8611

Epoch 24/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0022 Acc: 0.9633
val Loss: 0.0103 Acc: 0.8611

Epoch 25/93
----------
train Loss: 0.0024 Acc: 0.9511
val Loss: 0.0081 Acc: 0.8704

Epoch 26/93
----------
train Loss: 0.0028 Acc: 0.9511
val Loss: 0.0078 Acc: 0.8519

Epoch 27/93
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0075 Acc: 0.8519

Epoch 28/93
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0076 Acc: 0.8519

Epoch 29/93
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0068 Acc: 0.8519

Epoch 30/93
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0096 Acc: 0.8519

Epoch 31/93
----------
train Loss: 0.0032 Acc: 0.9480
val Loss: 0.0069 Acc: 0.8704

Epoch 32/93
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0116 Acc: 0.8704

Epoch 33/93
----------
train Loss: 0.0022 Acc: 0.9633
val Loss: 0.0106 Acc: 0.8611

Epoch 34/93
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0062 Acc: 0.8519

Epoch 35/93
----------
train Loss: 0.0025 Acc: 0.9327
val Loss: 0.0059 Acc: 0.8519

Epoch 36/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0084 Acc: 0.8519

Epoch 37/93
----------
train Loss: 0.0025 Acc: 0.9572
val Loss: 0.0131 Acc: 0.8519

Epoch 38/93
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0061 Acc: 0.8519

Epoch 39/93
----------
train Loss: 0.0021 Acc: 0.9694
val Loss: 0.0095 Acc: 0.8519

Epoch 40/93
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0086 Acc: 0.8611

Epoch 41/93
----------
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0051 Acc: 0.8519

Epoch 42/93
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0047 Acc: 0.8519

Epoch 43/93
----------
train Loss: 0.0021 Acc: 0.9541
val Loss: 0.0114 Acc: 0.8519

Epoch 44/93
----------
train Loss: 0.0021 Acc: 0.9541
val Loss: 0.0067 Acc: 0.8519

Epoch 45/93
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0060 Acc: 0.8519

Epoch 46/93
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0096 Acc: 0.8519

Epoch 47/93
----------
train Loss: 0.0027 Acc: 0.9480
val Loss: 0.0150 Acc: 0.8519

Epoch 48/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0068 Acc: 0.8519

Epoch 49/93
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0122 Acc: 0.8519

Epoch 50/93
----------
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0050 Acc: 0.8519

Epoch 51/93
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0072 Acc: 0.8519

Epoch 52/93
----------
train Loss: 0.0023 Acc: 0.9725
val Loss: 0.0058 Acc: 0.8519

Epoch 53/93
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0064 Acc: 0.8519

Epoch 54/93
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0055 Acc: 0.8519

Epoch 55/93
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0048 Acc: 0.8519

Epoch 56/93
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0142 Acc: 0.8519

Epoch 57/93
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0125 Acc: 0.8519

Epoch 58/93
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0089 Acc: 0.8519

Epoch 59/93
----------
train Loss: 0.0026 Acc: 0.9388
val Loss: 0.0073 Acc: 0.8519

Epoch 60/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0052 Acc: 0.8519

Epoch 61/93
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0102 Acc: 0.8519

Epoch 62/93
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0057 Acc: 0.8519

Epoch 63/93
----------
train Loss: 0.0025 Acc: 0.9725
val Loss: 0.0065 Acc: 0.8519

Epoch 64/93
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0150 Acc: 0.8519

Epoch 65/93
----------
train Loss: 0.0025 Acc: 0.9602
val Loss: 0.0060 Acc: 0.8611

Epoch 66/93
----------
train Loss: 0.0020 Acc: 0.9541
val Loss: 0.0085 Acc: 0.8519

Epoch 67/93
----------
train Loss: 0.0022 Acc: 0.9633
val Loss: 0.0080 Acc: 0.8519

Epoch 68/93
----------
train Loss: 0.0029 Acc: 0.9664
val Loss: 0.0093 Acc: 0.8611

Epoch 69/93
----------
train Loss: 0.0021 Acc: 0.9541
val Loss: 0.0047 Acc: 0.8611

Epoch 70/93
----------
train Loss: 0.0026 Acc: 0.9572
val Loss: 0.0064 Acc: 0.8519

Epoch 71/93
----------
train Loss: 0.0024 Acc: 0.9511
val Loss: 0.0131 Acc: 0.8519

Epoch 72/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0079 Acc: 0.8704

Epoch 73/93
----------
train Loss: 0.0025 Acc: 0.9541
val Loss: 0.0054 Acc: 0.8519

Epoch 74/93
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0089 Acc: 0.8519

Epoch 75/93
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0087 Acc: 0.8519

Epoch 76/93
----------
train Loss: 0.0023 Acc: 0.9664
val Loss: 0.0070 Acc: 0.8519

Epoch 77/93
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0067 Acc: 0.8519

Epoch 78/93
----------
train Loss: 0.0025 Acc: 0.9541
val Loss: 0.0076 Acc: 0.8611

Epoch 79/93
----------
train Loss: 0.0028 Acc: 0.9388
val Loss: 0.0114 Acc: 0.8519

Epoch 80/93
----------
train Loss: 0.0020 Acc: 0.9664
val Loss: 0.0076 Acc: 0.8611

Epoch 81/93
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0136 Acc: 0.8611

Epoch 82/93
----------
train Loss: 0.0021 Acc: 0.9633
val Loss: 0.0043 Acc: 0.8611

Epoch 83/93
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0052 Acc: 0.8519

Epoch 84/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0094 Acc: 0.8519

Epoch 85/93
----------
train Loss: 0.0026 Acc: 0.9450
val Loss: 0.0050 Acc: 0.8519

Epoch 86/93
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0139 Acc: 0.8519

Epoch 87/93
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0088 Acc: 0.8519

Epoch 88/93
----------
train Loss: 0.0021 Acc: 0.9633
val Loss: 0.0062 Acc: 0.8519

Epoch 89/93
----------
train Loss: 0.0024 Acc: 0.9664
val Loss: 0.0062 Acc: 0.8519

Epoch 90/93
----------
train Loss: 0.0023 Acc: 0.9694
val Loss: 0.0081 Acc: 0.8519

Epoch 91/93
----------
train Loss: 0.0025 Acc: 0.9541
val Loss: 0.0106 Acc: 0.8519

Epoch 92/93
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0056 Acc: 0.8519

Epoch 93/93
----------
train Loss: 0.0024 Acc: 0.9358
val Loss: 0.0062 Acc: 0.8519

Training complete in 3m 30s
Best val Acc: 0.879630

---Fine tuning.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0055 Acc: 0.8889

Epoch 1/93
----------
train Loss: 0.0012 Acc: 0.9878
val Loss: 0.0074 Acc: 0.8704

Epoch 2/93
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0059 Acc: 0.8704

Epoch 3/93
----------
train Loss: 0.0004 Acc: 0.9969
val Loss: 0.0081 Acc: 0.8796

Epoch 4/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 5/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0133 Acc: 0.8704

Epoch 6/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8889

Epoch 7/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8889

Epoch 8/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0142 Acc: 0.8889

Epoch 9/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8889

Epoch 10/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8981

Epoch 11/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 12/93
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8981

Epoch 13/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0152 Acc: 0.8981

Epoch 14/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 15/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8981

Epoch 16/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8981

Epoch 17/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 18/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Epoch 19/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8981

Epoch 20/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 21/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 22/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 23/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8981

Epoch 24/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 25/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8981

Epoch 26/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8981

Epoch 27/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Epoch 28/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8981

Epoch 29/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 30/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8889

Epoch 31/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8981

Epoch 32/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 33/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 34/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8889

Epoch 35/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8981

Epoch 36/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8981

Epoch 37/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8981

Epoch 38/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0142 Acc: 0.8981

Epoch 39/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 40/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 41/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 42/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 43/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 44/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8981

Epoch 45/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8981

Epoch 46/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8981

Epoch 47/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 48/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8981

Epoch 49/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8981

Epoch 50/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 51/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 52/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8981

Epoch 53/93
----------
train Loss: 0.0002 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8981

Epoch 54/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8889

Epoch 55/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0114 Acc: 0.8981

Epoch 56/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 57/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8981

Epoch 58/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8981

Epoch 59/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 60/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 61/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 62/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8981

Epoch 63/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 64/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8981

Epoch 65/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8981

Epoch 66/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 67/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0198 Acc: 0.8981

Epoch 68/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8981

Epoch 69/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 70/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 71/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 72/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 73/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 74/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8981

Epoch 75/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8981

Epoch 76/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0136 Acc: 0.8981

Epoch 77/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8981

Epoch 78/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8981

Epoch 79/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 80/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 81/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8981

Epoch 82/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8981

Epoch 83/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8981

Epoch 84/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Epoch 85/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8981

Epoch 86/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 87/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Epoch 88/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8981

Epoch 89/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8981

Epoch 90/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 91/93
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8981

Epoch 92/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8981

Epoch 93/93
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Training complete in 3m 53s
Best val Acc: 0.898148

---Testing---
Test accuracy: 0.974713
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 97 %
mean: 0.9730901896252504, std: 0.00393052872867506
--------------------

run info[val: 0.3, epoch: 61, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0220 Acc: 0.3311
val Loss: 0.0244 Acc: 0.1923

Epoch 1/60
----------
train Loss: 0.0209 Acc: 0.3410
val Loss: 0.0180 Acc: 0.5000

Epoch 2/60
----------
train Loss: 0.0161 Acc: 0.6098
val Loss: 0.0261 Acc: 0.5077

Epoch 3/60
----------
train Loss: 0.0226 Acc: 0.5180
val Loss: 0.0208 Acc: 0.5000

Epoch 4/60
----------
train Loss: 0.0224 Acc: 0.4656
val Loss: 0.0095 Acc: 0.7692

Epoch 5/60
----------
train Loss: 0.0165 Acc: 0.6197
val Loss: 0.0146 Acc: 0.6923

Epoch 6/60
----------
train Loss: 0.0182 Acc: 0.5672
val Loss: 0.0159 Acc: 0.6154

Epoch 7/60
----------
train Loss: 0.0102 Acc: 0.7443
val Loss: 0.0128 Acc: 0.7615

Epoch 8/60
----------
train Loss: 0.0118 Acc: 0.7508
val Loss: 0.0099 Acc: 0.7538

Epoch 9/60
----------
train Loss: 0.0102 Acc: 0.7967
val Loss: 0.0103 Acc: 0.7692

Epoch 10/60
----------
train Loss: 0.0065 Acc: 0.8426
val Loss: 0.0097 Acc: 0.8231

Epoch 11/60
----------
train Loss: 0.0099 Acc: 0.8098
val Loss: 0.0075 Acc: 0.8154

Epoch 12/60
----------
train Loss: 0.0057 Acc: 0.8361
val Loss: 0.0101 Acc: 0.8077

Epoch 13/60
----------
LR is set to 0.001
train Loss: 0.0113 Acc: 0.8525
val Loss: 0.0109 Acc: 0.8154

Epoch 14/60
----------
train Loss: 0.0051 Acc: 0.8885
val Loss: 0.0091 Acc: 0.8385

Epoch 15/60
----------
train Loss: 0.0034 Acc: 0.9016
val Loss: 0.0084 Acc: 0.8615

Epoch 16/60
----------
train Loss: 0.0047 Acc: 0.9115
val Loss: 0.0093 Acc: 0.8308

Epoch 17/60
----------
train Loss: 0.0059 Acc: 0.8918
val Loss: 0.0099 Acc: 0.8000

Epoch 18/60
----------
train Loss: 0.0083 Acc: 0.8590
val Loss: 0.0107 Acc: 0.8000

Epoch 19/60
----------
train Loss: 0.0033 Acc: 0.8984
val Loss: 0.0070 Acc: 0.8308

Epoch 20/60
----------
train Loss: 0.0028 Acc: 0.9115
val Loss: 0.0100 Acc: 0.8462

Epoch 21/60
----------
train Loss: 0.0037 Acc: 0.8984
val Loss: 0.0083 Acc: 0.8231

Epoch 22/60
----------
train Loss: 0.0027 Acc: 0.9213
val Loss: 0.0083 Acc: 0.8308

Epoch 23/60
----------
train Loss: 0.0040 Acc: 0.9148
val Loss: 0.0072 Acc: 0.8308

Epoch 24/60
----------
train Loss: 0.0033 Acc: 0.9213
val Loss: 0.0075 Acc: 0.8615

Epoch 25/60
----------
train Loss: 0.0029 Acc: 0.9148
val Loss: 0.0091 Acc: 0.8538

Epoch 26/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0036 Acc: 0.9049
val Loss: 0.0071 Acc: 0.8538

Epoch 27/60
----------
train Loss: 0.0083 Acc: 0.8787
val Loss: 0.0084 Acc: 0.8385

Epoch 28/60
----------
train Loss: 0.0039 Acc: 0.9082
val Loss: 0.0076 Acc: 0.8308

Epoch 29/60
----------
train Loss: 0.0024 Acc: 0.9246
val Loss: 0.0080 Acc: 0.8231

Epoch 30/60
----------
train Loss: 0.0030 Acc: 0.9082
val Loss: 0.0078 Acc: 0.8538

Epoch 31/60
----------
train Loss: 0.0056 Acc: 0.9180
val Loss: 0.0087 Acc: 0.8462

Epoch 32/60
----------
train Loss: 0.0034 Acc: 0.9246
val Loss: 0.0079 Acc: 0.8462

Epoch 33/60
----------
train Loss: 0.0050 Acc: 0.9049
val Loss: 0.0097 Acc: 0.8385

Epoch 34/60
----------
train Loss: 0.0074 Acc: 0.9115
val Loss: 0.0083 Acc: 0.8308

Epoch 35/60
----------
train Loss: 0.0049 Acc: 0.9115
val Loss: 0.0075 Acc: 0.8538

Epoch 36/60
----------
train Loss: 0.0107 Acc: 0.8885
val Loss: 0.0074 Acc: 0.8462

Epoch 37/60
----------
train Loss: 0.0054 Acc: 0.8984
val Loss: 0.0092 Acc: 0.8615

Epoch 38/60
----------
train Loss: 0.0032 Acc: 0.9180
val Loss: 0.0090 Acc: 0.8615

Epoch 39/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.9246
val Loss: 0.0063 Acc: 0.8462

Epoch 40/60
----------
train Loss: 0.0052 Acc: 0.9180
val Loss: 0.0114 Acc: 0.8538

Epoch 41/60
----------
train Loss: 0.0026 Acc: 0.9443
val Loss: 0.0074 Acc: 0.8385

Epoch 42/60
----------
train Loss: 0.0029 Acc: 0.9016
val Loss: 0.0085 Acc: 0.8462

Epoch 43/60
----------
train Loss: 0.0042 Acc: 0.9279
val Loss: 0.0099 Acc: 0.8385

Epoch 44/60
----------
train Loss: 0.0034 Acc: 0.9016
val Loss: 0.0105 Acc: 0.8462

Epoch 45/60
----------
train Loss: 0.0032 Acc: 0.9148
val Loss: 0.0083 Acc: 0.8462

Epoch 46/60
----------
train Loss: 0.0049 Acc: 0.9016
val Loss: 0.0075 Acc: 0.8385

Epoch 47/60
----------
train Loss: 0.0039 Acc: 0.9049
val Loss: 0.0083 Acc: 0.8538

Epoch 48/60
----------
train Loss: 0.0039 Acc: 0.9213
val Loss: 0.0084 Acc: 0.8462

Epoch 49/60
----------
train Loss: 0.0045 Acc: 0.9180
val Loss: 0.0083 Acc: 0.8538

Epoch 50/60
----------
train Loss: 0.0035 Acc: 0.9082
val Loss: 0.0088 Acc: 0.8615

Epoch 51/60
----------
train Loss: 0.0064 Acc: 0.9148
val Loss: 0.0098 Acc: 0.8538

Epoch 52/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0024 Acc: 0.9115
val Loss: 0.0061 Acc: 0.8615

Epoch 53/60
----------
train Loss: 0.0037 Acc: 0.9213
val Loss: 0.0056 Acc: 0.8538

Epoch 54/60
----------
train Loss: 0.0049 Acc: 0.8951
val Loss: 0.0076 Acc: 0.8692

Epoch 55/60
----------
train Loss: 0.0051 Acc: 0.9115
val Loss: 0.0084 Acc: 0.8462

Epoch 56/60
----------
train Loss: 0.0029 Acc: 0.9016
val Loss: 0.0104 Acc: 0.8538

Epoch 57/60
----------
train Loss: 0.0071 Acc: 0.9082
val Loss: 0.0093 Acc: 0.8385

Epoch 58/60
----------
train Loss: 0.0035 Acc: 0.9213
val Loss: 0.0095 Acc: 0.8538

Epoch 59/60
----------
train Loss: 0.0059 Acc: 0.9016
val Loss: 0.0073 Acc: 0.8538

Epoch 60/60
----------
train Loss: 0.0062 Acc: 0.9082
val Loss: 0.0081 Acc: 0.8692

Training complete in 2m 17s
Best val Acc: 0.869231

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0028 Acc: 0.8984
val Loss: 0.0117 Acc: 0.8077

Epoch 1/60
----------
train Loss: 0.0012 Acc: 0.9705
val Loss: 0.0094 Acc: 0.7846

Epoch 2/60
----------
train Loss: 0.0012 Acc: 0.9770
val Loss: 0.0066 Acc: 0.8692

Epoch 3/60
----------
train Loss: 0.0008 Acc: 0.9902
val Loss: 0.0087 Acc: 0.8308

Epoch 4/60
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0075 Acc: 0.8538

Epoch 5/60
----------
train Loss: 0.0018 Acc: 0.9803
val Loss: 0.0057 Acc: 0.8769

Epoch 6/60
----------
train Loss: 0.0032 Acc: 0.9639
val Loss: 0.0129 Acc: 0.8154

Epoch 7/60
----------
train Loss: 0.0035 Acc: 0.9016
val Loss: 0.0244 Acc: 0.6846

Epoch 8/60
----------
train Loss: 0.0130 Acc: 0.8918
val Loss: 0.0301 Acc: 0.6154

Epoch 9/60
----------
train Loss: 0.0097 Acc: 0.8164
val Loss: 0.0479 Acc: 0.4231

Epoch 10/60
----------
train Loss: 0.0258 Acc: 0.7541
val Loss: 0.1117 Acc: 0.1077

Epoch 11/60
----------
train Loss: 0.0290 Acc: 0.5934
val Loss: 0.1203 Acc: 0.1846

Epoch 12/60
----------
train Loss: 0.0383 Acc: 0.6230
val Loss: 0.0894 Acc: 0.3308

Epoch 13/60
----------
LR is set to 0.001
train Loss: 0.0238 Acc: 0.5967
val Loss: 0.0661 Acc: 0.4846

Epoch 14/60
----------
train Loss: 0.0219 Acc: 0.6525
val Loss: 0.0413 Acc: 0.5923

Epoch 15/60
----------
train Loss: 0.0189 Acc: 0.7016
val Loss: 0.0326 Acc: 0.6692

Epoch 16/60
----------
train Loss: 0.0115 Acc: 0.7705
val Loss: 0.0240 Acc: 0.6769

Epoch 17/60
----------
train Loss: 0.0126 Acc: 0.7607
val Loss: 0.0242 Acc: 0.6923

Epoch 18/60
----------
train Loss: 0.0106 Acc: 0.7770
val Loss: 0.0213 Acc: 0.6769

Epoch 19/60
----------
train Loss: 0.0100 Acc: 0.8131
val Loss: 0.0228 Acc: 0.6385

Epoch 20/60
----------
train Loss: 0.0050 Acc: 0.8361
val Loss: 0.0259 Acc: 0.6000

Epoch 21/60
----------
train Loss: 0.0047 Acc: 0.8623
val Loss: 0.0285 Acc: 0.6231

Epoch 22/60
----------
train Loss: 0.0079 Acc: 0.8492
val Loss: 0.0190 Acc: 0.6692

Epoch 23/60
----------
train Loss: 0.0105 Acc: 0.8820
val Loss: 0.0186 Acc: 0.6846

Epoch 24/60
----------
train Loss: 0.0055 Acc: 0.8754
val Loss: 0.0173 Acc: 0.7154

Epoch 25/60
----------
train Loss: 0.0036 Acc: 0.8820
val Loss: 0.0140 Acc: 0.7308

Epoch 26/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0035 Acc: 0.8984
val Loss: 0.0210 Acc: 0.7077

Epoch 27/60
----------
train Loss: 0.0046 Acc: 0.8754
val Loss: 0.0126 Acc: 0.7231

Epoch 28/60
----------
train Loss: 0.0088 Acc: 0.8754
val Loss: 0.0116 Acc: 0.7385

Epoch 29/60
----------
train Loss: 0.0053 Acc: 0.9180
val Loss: 0.0165 Acc: 0.7385

Epoch 30/60
----------
train Loss: 0.0033 Acc: 0.9180
val Loss: 0.0142 Acc: 0.7615

Epoch 31/60
----------
train Loss: 0.0059 Acc: 0.9311
val Loss: 0.0144 Acc: 0.7692

Epoch 32/60
----------
train Loss: 0.0036 Acc: 0.9344
val Loss: 0.0152 Acc: 0.7615

Epoch 33/60
----------
train Loss: 0.0020 Acc: 0.9508
val Loss: 0.0159 Acc: 0.7538

Epoch 34/60
----------
train Loss: 0.0055 Acc: 0.9213
val Loss: 0.0111 Acc: 0.7538

Epoch 35/60
----------
train Loss: 0.0025 Acc: 0.9377
val Loss: 0.0182 Acc: 0.7538

Epoch 36/60
----------
train Loss: 0.0039 Acc: 0.9443
val Loss: 0.0145 Acc: 0.7538

Epoch 37/60
----------
train Loss: 0.0031 Acc: 0.9410
val Loss: 0.0156 Acc: 0.7615

Epoch 38/60
----------
train Loss: 0.0030 Acc: 0.9508
val Loss: 0.0179 Acc: 0.7538

Epoch 39/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0085 Acc: 0.9344
val Loss: 0.0138 Acc: 0.7615

Epoch 40/60
----------
train Loss: 0.0057 Acc: 0.9443
val Loss: 0.0146 Acc: 0.7692

Epoch 41/60
----------
train Loss: 0.0024 Acc: 0.9443
val Loss: 0.0144 Acc: 0.7692

Epoch 42/60
----------
train Loss: 0.0021 Acc: 0.9705
val Loss: 0.0145 Acc: 0.7692

Epoch 43/60
----------
train Loss: 0.0071 Acc: 0.9508
val Loss: 0.0140 Acc: 0.7692

Epoch 44/60
----------
train Loss: 0.0028 Acc: 0.9279
val Loss: 0.0144 Acc: 0.7615

Epoch 45/60
----------
train Loss: 0.0025 Acc: 0.9508
val Loss: 0.0124 Acc: 0.7615

Epoch 46/60
----------
train Loss: 0.0043 Acc: 0.9344
val Loss: 0.0119 Acc: 0.7769

Epoch 47/60
----------
train Loss: 0.0020 Acc: 0.9607
val Loss: 0.0147 Acc: 0.7538

Epoch 48/60
----------
train Loss: 0.0021 Acc: 0.9508
val Loss: 0.0146 Acc: 0.7615

Epoch 49/60
----------
train Loss: 0.0017 Acc: 0.9475
val Loss: 0.0149 Acc: 0.7538

Epoch 50/60
----------
train Loss: 0.0033 Acc: 0.9574
val Loss: 0.0165 Acc: 0.7615

Epoch 51/60
----------
train Loss: 0.0049 Acc: 0.9574
val Loss: 0.0120 Acc: 0.7538

Epoch 52/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0040 Acc: 0.9311
val Loss: 0.0177 Acc: 0.7769

Epoch 53/60
----------
train Loss: 0.0029 Acc: 0.9475
val Loss: 0.0152 Acc: 0.7692

Epoch 54/60
----------
train Loss: 0.0022 Acc: 0.9475
val Loss: 0.0117 Acc: 0.7692

Epoch 55/60
----------
train Loss: 0.0051 Acc: 0.9279
val Loss: 0.0114 Acc: 0.7769

Epoch 56/60
----------
train Loss: 0.0028 Acc: 0.9344
val Loss: 0.0147 Acc: 0.7692

Epoch 57/60
----------
train Loss: 0.0044 Acc: 0.9344
val Loss: 0.0138 Acc: 0.7538

Epoch 58/60
----------
train Loss: 0.0045 Acc: 0.9607
val Loss: 0.0159 Acc: 0.7462

Epoch 59/60
----------
train Loss: 0.0047 Acc: 0.9541
val Loss: 0.0141 Acc: 0.7692

Epoch 60/60
----------
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0159 Acc: 0.7692

Training complete in 2m 28s
Best val Acc: 0.876923

---Testing---
Test accuracy: 0.956322
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 93 %
Accuracy of Rajiformes : 93 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 97 %
mean: 0.949429110548977, std: 0.019411378629374507

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.98]_std[0.02].save".
--------------------

run info[val: 0.1, epoch: 86, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0147 Acc: 0.3597
val Loss: 0.0318 Acc: 0.4884

Epoch 1/85
----------
train Loss: 0.0115 Acc: 0.5816
val Loss: 0.0237 Acc: 0.5814

Epoch 2/85
----------
train Loss: 0.0084 Acc: 0.7194
val Loss: 0.0182 Acc: 0.6977

Epoch 3/85
----------
train Loss: 0.0071 Acc: 0.7551
val Loss: 0.0141 Acc: 0.7674

Epoch 4/85
----------
LR is set to 0.001
train Loss: 0.0053 Acc: 0.8495
val Loss: 0.0143 Acc: 0.7674

Epoch 5/85
----------
train Loss: 0.0055 Acc: 0.8189
val Loss: 0.0145 Acc: 0.7442

Epoch 6/85
----------
train Loss: 0.0052 Acc: 0.8367
val Loss: 0.0145 Acc: 0.7674

Epoch 7/85
----------
train Loss: 0.0053 Acc: 0.8342
val Loss: 0.0144 Acc: 0.7907

Epoch 8/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0053 Acc: 0.8367
val Loss: 0.0143 Acc: 0.7907

Epoch 9/85
----------
train Loss: 0.0052 Acc: 0.8189
val Loss: 0.0143 Acc: 0.7907

Epoch 10/85
----------
train Loss: 0.0051 Acc: 0.8393
val Loss: 0.0143 Acc: 0.7907

Epoch 11/85
----------
train Loss: 0.0050 Acc: 0.8520
val Loss: 0.0143 Acc: 0.7907

Epoch 12/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0051 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 13/85
----------
train Loss: 0.0051 Acc: 0.8342
val Loss: 0.0143 Acc: 0.7907

Epoch 14/85
----------
train Loss: 0.0053 Acc: 0.8342
val Loss: 0.0143 Acc: 0.7907

Epoch 15/85
----------
train Loss: 0.0055 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 16/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0054 Acc: 0.8367
val Loss: 0.0143 Acc: 0.7907

Epoch 17/85
----------
train Loss: 0.0053 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 18/85
----------
train Loss: 0.0050 Acc: 0.8495
val Loss: 0.0143 Acc: 0.7907

Epoch 19/85
----------
train Loss: 0.0052 Acc: 0.8138
val Loss: 0.0143 Acc: 0.7907

Epoch 20/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0052 Acc: 0.8444
val Loss: 0.0143 Acc: 0.7907

Epoch 21/85
----------
train Loss: 0.0052 Acc: 0.8240
val Loss: 0.0143 Acc: 0.7907

Epoch 22/85
----------
train Loss: 0.0052 Acc: 0.8367
val Loss: 0.0143 Acc: 0.7907

Epoch 23/85
----------
train Loss: 0.0052 Acc: 0.8342
val Loss: 0.0143 Acc: 0.7907

Epoch 24/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0053 Acc: 0.8189
val Loss: 0.0143 Acc: 0.8140

Epoch 25/85
----------
train Loss: 0.0051 Acc: 0.8520
val Loss: 0.0143 Acc: 0.8140

Epoch 26/85
----------
train Loss: 0.0051 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 27/85
----------
train Loss: 0.0053 Acc: 0.8240
val Loss: 0.0143 Acc: 0.7907

Epoch 28/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0052 Acc: 0.8342
val Loss: 0.0143 Acc: 0.7907

Epoch 29/85
----------
train Loss: 0.0052 Acc: 0.8112
val Loss: 0.0143 Acc: 0.7907

Epoch 30/85
----------
train Loss: 0.0051 Acc: 0.8189
val Loss: 0.0143 Acc: 0.7907

Epoch 31/85
----------
train Loss: 0.0050 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 32/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0052 Acc: 0.8495
val Loss: 0.0144 Acc: 0.7907

Epoch 33/85
----------
train Loss: 0.0054 Acc: 0.8316
val Loss: 0.0144 Acc: 0.7907

Epoch 34/85
----------
train Loss: 0.0054 Acc: 0.8240
val Loss: 0.0144 Acc: 0.7907

Epoch 35/85
----------
train Loss: 0.0053 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 36/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0052 Acc: 0.8444
val Loss: 0.0142 Acc: 0.7907

Epoch 37/85
----------
train Loss: 0.0051 Acc: 0.8393
val Loss: 0.0143 Acc: 0.7907

Epoch 38/85
----------
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0142 Acc: 0.7907

Epoch 39/85
----------
train Loss: 0.0054 Acc: 0.8061
val Loss: 0.0143 Acc: 0.7907

Epoch 40/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0054 Acc: 0.8240
val Loss: 0.0143 Acc: 0.8140

Epoch 41/85
----------
train Loss: 0.0050 Acc: 0.8444
val Loss: 0.0143 Acc: 0.8140

Epoch 42/85
----------
train Loss: 0.0056 Acc: 0.8087
val Loss: 0.0143 Acc: 0.8140

Epoch 43/85
----------
train Loss: 0.0053 Acc: 0.8469
val Loss: 0.0143 Acc: 0.7907

Epoch 44/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0054 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 45/85
----------
train Loss: 0.0053 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 46/85
----------
train Loss: 0.0055 Acc: 0.8163
val Loss: 0.0143 Acc: 0.7907

Epoch 47/85
----------
train Loss: 0.0052 Acc: 0.8546
val Loss: 0.0143 Acc: 0.7907

Epoch 48/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 49/85
----------
train Loss: 0.0050 Acc: 0.8495
val Loss: 0.0143 Acc: 0.7907

Epoch 50/85
----------
train Loss: 0.0051 Acc: 0.8418
val Loss: 0.0143 Acc: 0.7907

Epoch 51/85
----------
train Loss: 0.0052 Acc: 0.8393
val Loss: 0.0143 Acc: 0.7907

Epoch 52/85
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0051 Acc: 0.8393
val Loss: 0.0143 Acc: 0.7907

Epoch 53/85
----------
train Loss: 0.0051 Acc: 0.8495
val Loss: 0.0143 Acc: 0.7907

Epoch 54/85
----------
train Loss: 0.0051 Acc: 0.8189
val Loss: 0.0143 Acc: 0.7907

Epoch 55/85
----------
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 56/85
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0054 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 57/85
----------
train Loss: 0.0056 Acc: 0.8036
val Loss: 0.0143 Acc: 0.7907

Epoch 58/85
----------
train Loss: 0.0052 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 59/85
----------
train Loss: 0.0051 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 60/85
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0054 Acc: 0.8087
val Loss: 0.0143 Acc: 0.7907

Epoch 61/85
----------
train Loss: 0.0051 Acc: 0.8316
val Loss: 0.0143 Acc: 0.7907

Epoch 62/85
----------
train Loss: 0.0053 Acc: 0.8189
val Loss: 0.0143 Acc: 0.7907

Epoch 63/85
----------
train Loss: 0.0053 Acc: 0.8240
val Loss: 0.0143 Acc: 0.7907

Epoch 64/85
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0051 Acc: 0.8393
val Loss: 0.0143 Acc: 0.8140

Epoch 65/85
----------
train Loss: 0.0053 Acc: 0.8265
val Loss: 0.0143 Acc: 0.8140

Epoch 66/85
----------
train Loss: 0.0054 Acc: 0.8291
val Loss: 0.0143 Acc: 0.8140

Epoch 67/85
----------
train Loss: 0.0053 Acc: 0.8265
val Loss: 0.0143 Acc: 0.8140

Epoch 68/85
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0051 Acc: 0.8444
val Loss: 0.0143 Acc: 0.8140

Epoch 69/85
----------
train Loss: 0.0051 Acc: 0.8367
val Loss: 0.0143 Acc: 0.7907

Epoch 70/85
----------
train Loss: 0.0052 Acc: 0.8418
val Loss: 0.0143 Acc: 0.7907

Epoch 71/85
----------
train Loss: 0.0050 Acc: 0.8418
val Loss: 0.0143 Acc: 0.7907

Epoch 72/85
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0053 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 73/85
----------
train Loss: 0.0051 Acc: 0.8367
val Loss: 0.0143 Acc: 0.7907

Epoch 74/85
----------
train Loss: 0.0051 Acc: 0.8240
val Loss: 0.0143 Acc: 0.8140

Epoch 75/85
----------
train Loss: 0.0051 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 76/85
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0052 Acc: 0.8265
val Loss: 0.0143 Acc: 0.7907

Epoch 77/85
----------
train Loss: 0.0053 Acc: 0.8163
val Loss: 0.0143 Acc: 0.7907

Epoch 78/85
----------
train Loss: 0.0054 Acc: 0.8138
val Loss: 0.0143 Acc: 0.7907

Epoch 79/85
----------
train Loss: 0.0051 Acc: 0.8546
val Loss: 0.0143 Acc: 0.7907

Epoch 80/85
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0051 Acc: 0.8214
val Loss: 0.0143 Acc: 0.7907

Epoch 81/85
----------
train Loss: 0.0051 Acc: 0.8189
val Loss: 0.0143 Acc: 0.7907

Epoch 82/85
----------
train Loss: 0.0053 Acc: 0.8265
val Loss: 0.0143 Acc: 0.8140

Epoch 83/85
----------
train Loss: 0.0050 Acc: 0.8367
val Loss: 0.0143 Acc: 0.8140

Epoch 84/85
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0053 Acc: 0.8291
val Loss: 0.0143 Acc: 0.7907

Epoch 85/85
----------
train Loss: 0.0052 Acc: 0.8342
val Loss: 0.0143 Acc: 0.7907

Training complete in 2m 59s
Best val Acc: 0.813953

---Fine tuning.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0050 Acc: 0.8495
val Loss: 0.0131 Acc: 0.8372

Epoch 1/85
----------
train Loss: 0.0035 Acc: 0.9082
val Loss: 0.0113 Acc: 0.8140

Epoch 2/85
----------
train Loss: 0.0020 Acc: 0.9566
val Loss: 0.0102 Acc: 0.8605

Epoch 3/85
----------
train Loss: 0.0011 Acc: 0.9872
val Loss: 0.0105 Acc: 0.8605

Epoch 4/85
----------
LR is set to 0.001
train Loss: 0.0007 Acc: 0.9923
val Loss: 0.0103 Acc: 0.8605

Epoch 5/85
----------
train Loss: 0.0007 Acc: 0.9923
val Loss: 0.0103 Acc: 0.8605

Epoch 6/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 7/85
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8605

Epoch 8/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 9/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 10/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 11/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 12/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 13/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 14/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 15/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 16/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 17/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 18/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 19/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 20/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 21/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 22/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 23/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 24/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 25/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 26/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 27/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 28/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 29/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 30/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 31/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 32/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 33/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 34/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 35/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 36/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 37/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 38/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 39/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 40/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 41/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0101 Acc: 0.8605

Epoch 42/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 43/85
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0102 Acc: 0.8605

Epoch 44/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 45/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 46/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 47/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 48/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 49/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 50/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 51/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 52/85
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 53/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 54/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 55/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 56/85
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 57/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 58/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 59/85
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0102 Acc: 0.8605

Epoch 60/85
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 61/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 62/85
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0102 Acc: 0.8605

Epoch 63/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8605

Epoch 64/85
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 65/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 66/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 67/85
----------
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 68/85
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 69/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0101 Acc: 0.8605

Epoch 70/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 71/85
----------
train Loss: 0.0006 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 72/85
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 73/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 74/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 75/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 76/85
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0006 Acc: 0.9898
val Loss: 0.0102 Acc: 0.8605

Epoch 77/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 78/85
----------
train Loss: 0.0005 Acc: 0.9949
val Loss: 0.0102 Acc: 0.8605

Epoch 79/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 80/85
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0006 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 81/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 82/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Epoch 83/85
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 84/85
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8605

Epoch 85/85
----------
train Loss: 0.0005 Acc: 0.9974
val Loss: 0.0102 Acc: 0.8605

Training complete in 3m 17s
Best val Acc: 0.860465

---Testing---
Test accuracy: 0.979310
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.976863937168041, std: 0.01088789047522747
--------------------

run info[val: 0.15, epoch: 65, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0165 Acc: 0.3811
val Loss: 0.0234 Acc: 0.4308

Epoch 1/64
----------
train Loss: 0.0129 Acc: 0.5703
val Loss: 0.0157 Acc: 0.7077

Epoch 2/64
----------
train Loss: 0.0094 Acc: 0.6919
val Loss: 0.0118 Acc: 0.7692

Epoch 3/64
----------
train Loss: 0.0071 Acc: 0.8081
val Loss: 0.0094 Acc: 0.7846

Epoch 4/64
----------
train Loss: 0.0060 Acc: 0.7892
val Loss: 0.0083 Acc: 0.8308

Epoch 5/64
----------
train Loss: 0.0049 Acc: 0.8514
val Loss: 0.0076 Acc: 0.8923

Epoch 6/64
----------
train Loss: 0.0042 Acc: 0.8919
val Loss: 0.0071 Acc: 0.8308

Epoch 7/64
----------
train Loss: 0.0039 Acc: 0.8892
val Loss: 0.0064 Acc: 0.9077

Epoch 8/64
----------
train Loss: 0.0033 Acc: 0.9027
val Loss: 0.0066 Acc: 0.8923

Epoch 9/64
----------
train Loss: 0.0031 Acc: 0.9135
val Loss: 0.0062 Acc: 0.8769

Epoch 10/64
----------
LR is set to 0.001
train Loss: 0.0026 Acc: 0.9378
val Loss: 0.0062 Acc: 0.8769

Epoch 11/64
----------
train Loss: 0.0026 Acc: 0.9270
val Loss: 0.0062 Acc: 0.9077

Epoch 12/64
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0063 Acc: 0.9077

Epoch 13/64
----------
train Loss: 0.0025 Acc: 0.9405
val Loss: 0.0063 Acc: 0.9077

Epoch 14/64
----------
train Loss: 0.0026 Acc: 0.9324
val Loss: 0.0063 Acc: 0.9077

Epoch 15/64
----------
train Loss: 0.0026 Acc: 0.9405
val Loss: 0.0062 Acc: 0.9077

Epoch 16/64
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0062 Acc: 0.8923

Epoch 17/64
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 18/64
----------
train Loss: 0.0024 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 19/64
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0062 Acc: 0.9077

Epoch 20/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0062 Acc: 0.9077

Epoch 21/64
----------
train Loss: 0.0025 Acc: 0.9568
val Loss: 0.0061 Acc: 0.9077

Epoch 22/64
----------
train Loss: 0.0023 Acc: 0.9568
val Loss: 0.0061 Acc: 0.9077

Epoch 23/64
----------
train Loss: 0.0024 Acc: 0.9595
val Loss: 0.0061 Acc: 0.8923

Epoch 24/64
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0062 Acc: 0.8923

Epoch 25/64
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0061 Acc: 0.8923

Epoch 26/64
----------
train Loss: 0.0024 Acc: 0.9595
val Loss: 0.0062 Acc: 0.9077

Epoch 27/64
----------
train Loss: 0.0021 Acc: 0.9595
val Loss: 0.0062 Acc: 0.9077

Epoch 28/64
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0062 Acc: 0.8923

Epoch 29/64
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 30/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0025 Acc: 0.9351
val Loss: 0.0062 Acc: 0.9077

Epoch 31/64
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 32/64
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0062 Acc: 0.9077

Epoch 33/64
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0062 Acc: 0.9077

Epoch 34/64
----------
train Loss: 0.0026 Acc: 0.9405
val Loss: 0.0062 Acc: 0.9077

Epoch 35/64
----------
train Loss: 0.0023 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 36/64
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 37/64
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 38/64
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 39/64
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 40/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0061 Acc: 0.9077

Epoch 41/64
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 42/64
----------
train Loss: 0.0022 Acc: 0.9595
val Loss: 0.0061 Acc: 0.9077

Epoch 43/64
----------
train Loss: 0.0023 Acc: 0.9595
val Loss: 0.0061 Acc: 0.9077

Epoch 44/64
----------
train Loss: 0.0025 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 45/64
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0062 Acc: 0.9077

Epoch 46/64
----------
train Loss: 0.0024 Acc: 0.9405
val Loss: 0.0061 Acc: 0.9077

Epoch 47/64
----------
train Loss: 0.0025 Acc: 0.9378
val Loss: 0.0061 Acc: 0.9077

Epoch 48/64
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 49/64
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 50/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0061 Acc: 0.9077

Epoch 51/64
----------
train Loss: 0.0024 Acc: 0.9432
val Loss: 0.0061 Acc: 0.9077

Epoch 52/64
----------
train Loss: 0.0024 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 53/64
----------
train Loss: 0.0023 Acc: 0.9514
val Loss: 0.0061 Acc: 0.9077

Epoch 54/64
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0062 Acc: 0.9077

Epoch 55/64
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 56/64
----------
train Loss: 0.0023 Acc: 0.9649
val Loss: 0.0062 Acc: 0.8923

Epoch 57/64
----------
train Loss: 0.0023 Acc: 0.9568
val Loss: 0.0062 Acc: 0.8923

Epoch 58/64
----------
train Loss: 0.0023 Acc: 0.9568
val Loss: 0.0062 Acc: 0.8769

Epoch 59/64
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0061 Acc: 0.9077

Epoch 60/64
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0061 Acc: 0.9077

Epoch 61/64
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0062 Acc: 0.9077

Epoch 62/64
----------
train Loss: 0.0023 Acc: 0.9514
val Loss: 0.0062 Acc: 0.9077

Epoch 63/64
----------
train Loss: 0.0025 Acc: 0.9568
val Loss: 0.0062 Acc: 0.9077

Epoch 64/64
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0062 Acc: 0.9077

Training complete in 2m 18s
Best val Acc: 0.907692

---Fine tuning.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0033 Acc: 0.9000
val Loss: 0.0065 Acc: 0.9231

Epoch 1/64
----------
train Loss: 0.0016 Acc: 0.9757
val Loss: 0.0064 Acc: 0.8923

Epoch 2/64
----------
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0058 Acc: 0.9231

Epoch 3/64
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9231

Epoch 4/64
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0061 Acc: 0.9231

Epoch 5/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9231

Epoch 6/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 7/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9231

Epoch 8/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9231

Epoch 9/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9231

Epoch 10/64
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 11/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 12/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 13/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 14/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 15/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 16/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 17/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 18/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 19/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 20/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 21/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 22/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 23/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 24/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 25/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 26/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 27/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 28/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 29/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 30/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 31/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 32/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 33/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 34/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 35/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 36/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 37/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 38/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 39/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 40/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 41/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 42/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 43/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 44/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 45/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 46/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 47/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 48/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 49/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 50/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 51/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 52/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 53/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 54/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 55/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 56/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 57/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 58/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 59/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 60/64
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Epoch 61/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 62/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 63/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9231

Epoch 64/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9231

Training complete in 2m 32s
Best val Acc: 0.923077

---Testing---
Test accuracy: 0.960920
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 92 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 96 %
mean: 0.9606244260602198, std: 0.01874625405559423
--------------------

run info[val: 0.2, epoch: 83, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0193 Acc: 0.2586
val Loss: 0.0177 Acc: 0.4828

Epoch 1/82
----------
train Loss: 0.0150 Acc: 0.5259
val Loss: 0.0152 Acc: 0.4943

Epoch 2/82
----------
train Loss: 0.0101 Acc: 0.6782
val Loss: 0.0089 Acc: 0.7241

Epoch 3/82
----------
train Loss: 0.0087 Acc: 0.7615
val Loss: 0.0082 Acc: 0.7126

Epoch 4/82
----------
train Loss: 0.0077 Acc: 0.7385
val Loss: 0.0073 Acc: 0.7471

Epoch 5/82
----------
LR is set to 0.001
train Loss: 0.0057 Acc: 0.8247
val Loss: 0.0069 Acc: 0.7471

Epoch 6/82
----------
train Loss: 0.0052 Acc: 0.8305
val Loss: 0.0065 Acc: 0.7816

Epoch 7/82
----------
train Loss: 0.0051 Acc: 0.8362
val Loss: 0.0063 Acc: 0.8161

Epoch 8/82
----------
train Loss: 0.0051 Acc: 0.8534
val Loss: 0.0064 Acc: 0.8046

Epoch 9/82
----------
train Loss: 0.0049 Acc: 0.8649
val Loss: 0.0064 Acc: 0.8046

Epoch 10/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0046 Acc: 0.8621
val Loss: 0.0064 Acc: 0.7931

Epoch 11/82
----------
train Loss: 0.0048 Acc: 0.8649
val Loss: 0.0064 Acc: 0.7931

Epoch 12/82
----------
train Loss: 0.0050 Acc: 0.8649
val Loss: 0.0064 Acc: 0.8046

Epoch 13/82
----------
train Loss: 0.0047 Acc: 0.8736
val Loss: 0.0064 Acc: 0.8046

Epoch 14/82
----------
train Loss: 0.0048 Acc: 0.8506
val Loss: 0.0064 Acc: 0.7931

Epoch 15/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0049 Acc: 0.8736
val Loss: 0.0063 Acc: 0.8046

Epoch 16/82
----------
train Loss: 0.0048 Acc: 0.8707
val Loss: 0.0063 Acc: 0.7931

Epoch 17/82
----------
train Loss: 0.0047 Acc: 0.8851
val Loss: 0.0063 Acc: 0.7931

Epoch 18/82
----------
train Loss: 0.0046 Acc: 0.8764
val Loss: 0.0063 Acc: 0.8046

Epoch 19/82
----------
train Loss: 0.0046 Acc: 0.8678
val Loss: 0.0063 Acc: 0.8046

Epoch 20/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0047 Acc: 0.8764
val Loss: 0.0063 Acc: 0.7931

Epoch 21/82
----------
train Loss: 0.0047 Acc: 0.8822
val Loss: 0.0063 Acc: 0.8046

Epoch 22/82
----------
train Loss: 0.0049 Acc: 0.8707
val Loss: 0.0063 Acc: 0.8046

Epoch 23/82
----------
train Loss: 0.0047 Acc: 0.8908
val Loss: 0.0063 Acc: 0.7931

Epoch 24/82
----------
train Loss: 0.0046 Acc: 0.8879
val Loss: 0.0064 Acc: 0.7816

Epoch 25/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0047 Acc: 0.8764
val Loss: 0.0064 Acc: 0.7816

Epoch 26/82
----------
train Loss: 0.0050 Acc: 0.8592
val Loss: 0.0064 Acc: 0.7816

Epoch 27/82
----------
train Loss: 0.0045 Acc: 0.8707
val Loss: 0.0064 Acc: 0.7816

Epoch 28/82
----------
train Loss: 0.0045 Acc: 0.8678
val Loss: 0.0064 Acc: 0.7816

Epoch 29/82
----------
train Loss: 0.0050 Acc: 0.8707
val Loss: 0.0063 Acc: 0.8046

Epoch 30/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0048 Acc: 0.8649
val Loss: 0.0064 Acc: 0.7931

Epoch 31/82
----------
train Loss: 0.0049 Acc: 0.8736
val Loss: 0.0064 Acc: 0.7816

Epoch 32/82
----------
train Loss: 0.0048 Acc: 0.8764
val Loss: 0.0064 Acc: 0.7816

Epoch 33/82
----------
train Loss: 0.0046 Acc: 0.8822
val Loss: 0.0064 Acc: 0.7816

Epoch 34/82
----------
train Loss: 0.0047 Acc: 0.8678
val Loss: 0.0064 Acc: 0.7931

Epoch 35/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0044 Acc: 0.8879
val Loss: 0.0064 Acc: 0.7931

Epoch 36/82
----------
train Loss: 0.0051 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7931

Epoch 37/82
----------
train Loss: 0.0045 Acc: 0.8822
val Loss: 0.0063 Acc: 0.7931

Epoch 38/82
----------
train Loss: 0.0050 Acc: 0.8534
val Loss: 0.0064 Acc: 0.7816

Epoch 39/82
----------
train Loss: 0.0048 Acc: 0.8822
val Loss: 0.0064 Acc: 0.7931

Epoch 40/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0045 Acc: 0.8793
val Loss: 0.0064 Acc: 0.7931

Epoch 41/82
----------
train Loss: 0.0047 Acc: 0.8649
val Loss: 0.0064 Acc: 0.7931

Epoch 42/82
----------
train Loss: 0.0048 Acc: 0.8736
val Loss: 0.0064 Acc: 0.7931

Epoch 43/82
----------
train Loss: 0.0048 Acc: 0.8793
val Loss: 0.0064 Acc: 0.7816

Epoch 44/82
----------
train Loss: 0.0048 Acc: 0.8707
val Loss: 0.0064 Acc: 0.7816

Epoch 45/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0046 Acc: 0.8592
val Loss: 0.0064 Acc: 0.7816

Epoch 46/82
----------
train Loss: 0.0047 Acc: 0.8621
val Loss: 0.0064 Acc: 0.7931

Epoch 47/82
----------
train Loss: 0.0048 Acc: 0.8534
val Loss: 0.0064 Acc: 0.7816

Epoch 48/82
----------
train Loss: 0.0049 Acc: 0.8822
val Loss: 0.0064 Acc: 0.7931

Epoch 49/82
----------
train Loss: 0.0047 Acc: 0.8678
val Loss: 0.0064 Acc: 0.7931

Epoch 50/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0050 Acc: 0.8908
val Loss: 0.0064 Acc: 0.7931

Epoch 51/82
----------
train Loss: 0.0047 Acc: 0.8534
val Loss: 0.0063 Acc: 0.7931

Epoch 52/82
----------
train Loss: 0.0047 Acc: 0.8707
val Loss: 0.0063 Acc: 0.7931

Epoch 53/82
----------
train Loss: 0.0048 Acc: 0.8678
val Loss: 0.0063 Acc: 0.7931

Epoch 54/82
----------
train Loss: 0.0048 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7931

Epoch 55/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0048 Acc: 0.8707
val Loss: 0.0064 Acc: 0.7931

Epoch 56/82
----------
train Loss: 0.0046 Acc: 0.8764
val Loss: 0.0064 Acc: 0.7816

Epoch 57/82
----------
train Loss: 0.0048 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7816

Epoch 58/82
----------
train Loss: 0.0046 Acc: 0.8621
val Loss: 0.0064 Acc: 0.7816

Epoch 59/82
----------
train Loss: 0.0046 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7931

Epoch 60/82
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0049 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7931

Epoch 61/82
----------
train Loss: 0.0047 Acc: 0.8707
val Loss: 0.0063 Acc: 0.7816

Epoch 62/82
----------
train Loss: 0.0046 Acc: 0.8678
val Loss: 0.0064 Acc: 0.7931

Epoch 63/82
----------
train Loss: 0.0048 Acc: 0.8793
val Loss: 0.0063 Acc: 0.7816

Epoch 64/82
----------
train Loss: 0.0047 Acc: 0.8764
val Loss: 0.0063 Acc: 0.7931

Epoch 65/82
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0045 Acc: 0.8822
val Loss: 0.0064 Acc: 0.7931

Epoch 66/82
----------
train Loss: 0.0047 Acc: 0.8649
val Loss: 0.0064 Acc: 0.7931

Epoch 67/82
----------
train Loss: 0.0047 Acc: 0.8736
val Loss: 0.0064 Acc: 0.7931

Epoch 68/82
----------
train Loss: 0.0047 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7931

Epoch 69/82
----------
train Loss: 0.0049 Acc: 0.8678
val Loss: 0.0063 Acc: 0.7931

Epoch 70/82
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0048 Acc: 0.8879
val Loss: 0.0063 Acc: 0.7931

Epoch 71/82
----------
train Loss: 0.0047 Acc: 0.8649
val Loss: 0.0063 Acc: 0.7931

Epoch 72/82
----------
train Loss: 0.0048 Acc: 0.8621
val Loss: 0.0063 Acc: 0.7816

Epoch 73/82
----------
train Loss: 0.0047 Acc: 0.8793
val Loss: 0.0063 Acc: 0.7931

Epoch 74/82
----------
train Loss: 0.0048 Acc: 0.8707
val Loss: 0.0063 Acc: 0.7931

Epoch 75/82
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0047 Acc: 0.8736
val Loss: 0.0063 Acc: 0.7816

Epoch 76/82
----------
train Loss: 0.0047 Acc: 0.8649
val Loss: 0.0063 Acc: 0.7931

Epoch 77/82
----------
train Loss: 0.0049 Acc: 0.8506
val Loss: 0.0063 Acc: 0.7931

Epoch 78/82
----------
train Loss: 0.0050 Acc: 0.8707
val Loss: 0.0063 Acc: 0.7931

Epoch 79/82
----------
train Loss: 0.0049 Acc: 0.8707
val Loss: 0.0064 Acc: 0.7931

Epoch 80/82
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0047 Acc: 0.8822
val Loss: 0.0064 Acc: 0.7816

Epoch 81/82
----------
train Loss: 0.0046 Acc: 0.8793
val Loss: 0.0064 Acc: 0.7816

Epoch 82/82
----------
train Loss: 0.0050 Acc: 0.8793
val Loss: 0.0064 Acc: 0.7816

Training complete in 3m 3s
Best val Acc: 0.816092

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0047 Acc: 0.8534
val Loss: 0.0061 Acc: 0.8161

Epoch 1/82
----------
train Loss: 0.0026 Acc: 0.9626
val Loss: 0.0052 Acc: 0.8621

Epoch 2/82
----------
train Loss: 0.0011 Acc: 0.9971
val Loss: 0.0044 Acc: 0.8621

Epoch 3/82
----------
train Loss: 0.0008 Acc: 0.9943
val Loss: 0.0048 Acc: 0.8621

Epoch 4/82
----------
train Loss: 0.0003 Acc: 0.9971
val Loss: 0.0047 Acc: 0.8736

Epoch 5/82
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8736

Epoch 6/82
----------
train Loss: 0.0003 Acc: 0.9971
val Loss: 0.0043 Acc: 0.8736

Epoch 7/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8736

Epoch 8/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 9/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 10/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 11/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 12/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 13/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 14/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 15/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 16/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 17/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 18/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 19/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 20/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 21/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 22/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 23/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 24/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 25/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 26/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 27/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 28/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 29/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 30/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 31/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9080

Epoch 32/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9080

Epoch 33/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 34/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 35/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 36/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 37/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 38/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 39/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 40/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 41/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 42/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 43/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 44/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 45/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 46/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 47/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 48/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 49/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 50/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 51/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 52/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 53/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 54/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 55/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 56/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 57/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 58/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 59/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 60/82
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 61/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 62/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 63/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 64/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 65/82
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 66/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 67/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 68/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 69/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 70/82
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Epoch 71/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 72/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 73/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 74/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 75/82
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 76/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 77/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 78/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 79/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8966

Epoch 80/82
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 81/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8851

Epoch 82/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8851

Training complete in 3m 21s
Best val Acc: 0.908046

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.978096037578467, std: 0.008429719905853828
--------------------

run info[val: 0.25, epoch: 81, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0200 Acc: 0.3517
val Loss: 0.0310 Acc: 0.3148

Epoch 1/80
----------
train Loss: 0.0172 Acc: 0.3976
val Loss: 0.0209 Acc: 0.4074

Epoch 2/80
----------
train Loss: 0.0112 Acc: 0.6728
val Loss: 0.0170 Acc: 0.6574

Epoch 3/80
----------
train Loss: 0.0088 Acc: 0.7401
val Loss: 0.0194 Acc: 0.6944

Epoch 4/80
----------
train Loss: 0.0067 Acc: 0.7706
val Loss: 0.0079 Acc: 0.7870

Epoch 5/80
----------
train Loss: 0.0066 Acc: 0.8226
val Loss: 0.0098 Acc: 0.8241

Epoch 6/80
----------
train Loss: 0.0053 Acc: 0.8440
val Loss: 0.0117 Acc: 0.8148

Epoch 7/80
----------
train Loss: 0.0045 Acc: 0.8532
val Loss: 0.0084 Acc: 0.8333

Epoch 8/80
----------
train Loss: 0.0039 Acc: 0.8960
val Loss: 0.0061 Acc: 0.8704

Epoch 9/80
----------
train Loss: 0.0032 Acc: 0.9174
val Loss: 0.0067 Acc: 0.8519

Epoch 10/80
----------
train Loss: 0.0035 Acc: 0.9297
val Loss: 0.0073 Acc: 0.8426

Epoch 11/80
----------
LR is set to 0.001
train Loss: 0.0028 Acc: 0.9388
val Loss: 0.0063 Acc: 0.8426

Epoch 12/80
----------
train Loss: 0.0028 Acc: 0.9327
val Loss: 0.0082 Acc: 0.8704

Epoch 13/80
----------
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0113 Acc: 0.8796

Epoch 14/80
----------
train Loss: 0.0031 Acc: 0.9419
val Loss: 0.0130 Acc: 0.8796

Epoch 15/80
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0137 Acc: 0.8889

Epoch 16/80
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0097 Acc: 0.8796

Epoch 17/80
----------
train Loss: 0.0029 Acc: 0.9541
val Loss: 0.0069 Acc: 0.8519

Epoch 18/80
----------
train Loss: 0.0027 Acc: 0.9572
val Loss: 0.0097 Acc: 0.8611

Epoch 19/80
----------
train Loss: 0.0025 Acc: 0.9572
val Loss: 0.0072 Acc: 0.8519

Epoch 20/80
----------
train Loss: 0.0025 Acc: 0.9694
val Loss: 0.0136 Acc: 0.8426

Epoch 21/80
----------
train Loss: 0.0024 Acc: 0.9511
val Loss: 0.0107 Acc: 0.8519

Epoch 22/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0052 Acc: 0.8611

Epoch 23/80
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0069 Acc: 0.8796

Epoch 24/80
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0112 Acc: 0.8611

Epoch 25/80
----------
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0052 Acc: 0.8426

Epoch 26/80
----------
train Loss: 0.0026 Acc: 0.9633
val Loss: 0.0080 Acc: 0.8519

Epoch 27/80
----------
train Loss: 0.0028 Acc: 0.9266
val Loss: 0.0102 Acc: 0.8704

Epoch 28/80
----------
train Loss: 0.0024 Acc: 0.9633
val Loss: 0.0086 Acc: 0.8611

Epoch 29/80
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0089 Acc: 0.8611

Epoch 30/80
----------
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0057 Acc: 0.8704

Epoch 31/80
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0053 Acc: 0.8704

Epoch 32/80
----------
train Loss: 0.0026 Acc: 0.9541
val Loss: 0.0100 Acc: 0.8611

Epoch 33/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0086 Acc: 0.8611

Epoch 34/80
----------
train Loss: 0.0023 Acc: 0.9664
val Loss: 0.0064 Acc: 0.8704

Epoch 35/80
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0054 Acc: 0.8611

Epoch 36/80
----------
train Loss: 0.0023 Acc: 0.9694
val Loss: 0.0088 Acc: 0.8611

Epoch 37/80
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0081 Acc: 0.8611

Epoch 38/80
----------
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0051 Acc: 0.8796

Epoch 39/80
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0052 Acc: 0.8704

Epoch 40/80
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0120 Acc: 0.8704

Epoch 41/80
----------
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0097 Acc: 0.8796

Epoch 42/80
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0057 Acc: 0.8704

Epoch 43/80
----------
train Loss: 0.0027 Acc: 0.9633
val Loss: 0.0064 Acc: 0.8519

Epoch 44/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0023 Acc: 0.9664
val Loss: 0.0114 Acc: 0.8519

Epoch 45/80
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0091 Acc: 0.8519

Epoch 46/80
----------
train Loss: 0.0024 Acc: 0.9419
val Loss: 0.0069 Acc: 0.8519

Epoch 47/80
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0096 Acc: 0.8519

Epoch 48/80
----------
train Loss: 0.0026 Acc: 0.9572
val Loss: 0.0065 Acc: 0.8611

Epoch 49/80
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0090 Acc: 0.8704

Epoch 50/80
----------
train Loss: 0.0025 Acc: 0.9572
val Loss: 0.0127 Acc: 0.8519

Epoch 51/80
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0050 Acc: 0.8426

Epoch 52/80
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0057 Acc: 0.8519

Epoch 53/80
----------
train Loss: 0.0025 Acc: 0.9725
val Loss: 0.0050 Acc: 0.8519

Epoch 54/80
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0066 Acc: 0.8611

Epoch 55/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0121 Acc: 0.8611

Epoch 56/80
----------
train Loss: 0.0024 Acc: 0.9511
val Loss: 0.0048 Acc: 0.8611

Epoch 57/80
----------
train Loss: 0.0026 Acc: 0.9541
val Loss: 0.0134 Acc: 0.8611

Epoch 58/80
----------
train Loss: 0.0025 Acc: 0.9450
val Loss: 0.0064 Acc: 0.8519

Epoch 59/80
----------
train Loss: 0.0024 Acc: 0.9694
val Loss: 0.0071 Acc: 0.8611

Epoch 60/80
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0092 Acc: 0.8704

Epoch 61/80
----------
train Loss: 0.0023 Acc: 0.9480
val Loss: 0.0047 Acc: 0.8704

Epoch 62/80
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0056 Acc: 0.8704

Epoch 63/80
----------
train Loss: 0.0022 Acc: 0.9664
val Loss: 0.0091 Acc: 0.8704

Epoch 64/80
----------
train Loss: 0.0024 Acc: 0.9419
val Loss: 0.0145 Acc: 0.8611

Epoch 65/80
----------
train Loss: 0.0021 Acc: 0.9664
val Loss: 0.0118 Acc: 0.8796

Epoch 66/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0078 Acc: 0.8796

Epoch 67/80
----------
train Loss: 0.0026 Acc: 0.9602
val Loss: 0.0139 Acc: 0.8704

Epoch 68/80
----------
train Loss: 0.0025 Acc: 0.9450
val Loss: 0.0149 Acc: 0.8796

Epoch 69/80
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0047 Acc: 0.8796

Epoch 70/80
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0121 Acc: 0.8704

Epoch 71/80
----------
train Loss: 0.0025 Acc: 0.9633
val Loss: 0.0050 Acc: 0.8611

Epoch 72/80
----------
train Loss: 0.0025 Acc: 0.9572
val Loss: 0.0070 Acc: 0.8704

Epoch 73/80
----------
train Loss: 0.0025 Acc: 0.9602
val Loss: 0.0065 Acc: 0.8519

Epoch 74/80
----------
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0085 Acc: 0.8704

Epoch 75/80
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0072 Acc: 0.8704

Epoch 76/80
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0051 Acc: 0.8704

Epoch 77/80
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0068 Acc: 0.8704

Epoch 78/80
----------
train Loss: 0.0023 Acc: 0.9541
val Loss: 0.0057 Acc: 0.8611

Epoch 79/80
----------
train Loss: 0.0024 Acc: 0.9664
val Loss: 0.0054 Acc: 0.8611

Epoch 80/80
----------
train Loss: 0.0022 Acc: 0.9419
val Loss: 0.0116 Acc: 0.8611

Training complete in 3m 1s
Best val Acc: 0.888889

---Fine tuning.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0066 Acc: 0.8796

Epoch 1/80
----------
train Loss: 0.0012 Acc: 0.9939
val Loss: 0.0071 Acc: 0.8704

Epoch 2/80
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0057 Acc: 0.8796

Epoch 3/80
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0098 Acc: 0.9167

Epoch 4/80
----------
train Loss: 0.0004 Acc: 0.9939
val Loss: 0.0053 Acc: 0.9074

Epoch 5/80
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8981

Epoch 6/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 7/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 8/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8981

Epoch 9/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0145 Acc: 0.8981

Epoch 10/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 11/80
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8796

Epoch 12/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8796

Epoch 13/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8796

Epoch 14/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0211 Acc: 0.8796

Epoch 15/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8796

Epoch 16/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8796

Epoch 17/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8796

Epoch 18/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8796

Epoch 19/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8889

Epoch 20/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8889

Epoch 21/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8889

Epoch 22/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8889

Epoch 23/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8889

Epoch 24/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8889

Epoch 25/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8796

Epoch 26/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8889

Epoch 27/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8889

Epoch 28/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8889

Epoch 29/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8889

Epoch 30/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8889

Epoch 31/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8889

Epoch 32/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8889

Epoch 33/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8889

Epoch 34/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8889

Epoch 35/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8889

Epoch 36/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8889

Epoch 37/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0155 Acc: 0.8889

Epoch 38/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8889

Epoch 39/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8889

Epoch 40/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0122 Acc: 0.8889

Epoch 41/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8889

Epoch 42/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 43/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8889

Epoch 44/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8889

Epoch 45/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8889

Epoch 46/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0105 Acc: 0.8889

Epoch 47/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8889

Epoch 48/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8889

Epoch 49/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0152 Acc: 0.8889

Epoch 50/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8889

Epoch 51/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8889

Epoch 52/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 53/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8796

Epoch 54/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8889

Epoch 55/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8889

Epoch 56/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0106 Acc: 0.8889

Epoch 57/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8889

Epoch 58/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8889

Epoch 59/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8889

Epoch 60/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8889

Epoch 61/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0139 Acc: 0.8889

Epoch 62/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8796

Epoch 63/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8889

Epoch 64/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8889

Epoch 65/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8889

Epoch 66/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8889

Epoch 67/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8889

Epoch 68/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8889

Epoch 69/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8889

Epoch 70/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0103 Acc: 0.8889

Epoch 71/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8889

Epoch 72/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8889

Epoch 73/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8889

Epoch 74/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0176 Acc: 0.8889

Epoch 75/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8889

Epoch 76/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8889

Epoch 77/80
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8889

Epoch 78/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8889

Epoch 79/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8889

Epoch 80/80
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8889

Training complete in 3m 20s
Best val Acc: 0.916667

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 97 %
mean: 0.9755593254277196, std: 0.007126773658415128
--------------------

run info[val: 0.3, epoch: 81, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0213 Acc: 0.2820
val Loss: 0.0294 Acc: 0.2615

Epoch 1/80
----------
train Loss: 0.0211 Acc: 0.3639
val Loss: 0.0257 Acc: 0.5846

Epoch 2/80
----------
train Loss: 0.0227 Acc: 0.5344
val Loss: 0.0138 Acc: 0.7000

Epoch 3/80
----------
train Loss: 0.0144 Acc: 0.6393
val Loss: 0.0116 Acc: 0.7077

Epoch 4/80
----------
train Loss: 0.0148 Acc: 0.7705
val Loss: 0.0115 Acc: 0.8000

Epoch 5/80
----------
train Loss: 0.0098 Acc: 0.7213
val Loss: 0.0131 Acc: 0.7000

Epoch 6/80
----------
train Loss: 0.0108 Acc: 0.7377
val Loss: 0.0162 Acc: 0.6462

Epoch 7/80
----------
train Loss: 0.0100 Acc: 0.7344
val Loss: 0.0128 Acc: 0.6615

Epoch 8/80
----------
train Loss: 0.0096 Acc: 0.7443
val Loss: 0.0111 Acc: 0.7308

Epoch 9/80
----------
train Loss: 0.0077 Acc: 0.8033
val Loss: 0.0147 Acc: 0.6846

Epoch 10/80
----------
train Loss: 0.0114 Acc: 0.7541
val Loss: 0.0088 Acc: 0.8077

Epoch 11/80
----------
train Loss: 0.0089 Acc: 0.8033
val Loss: 0.0156 Acc: 0.6462

Epoch 12/80
----------
LR is set to 0.001
train Loss: 0.0100 Acc: 0.7672
val Loss: 0.0151 Acc: 0.7077

Epoch 13/80
----------
train Loss: 0.0047 Acc: 0.8328
val Loss: 0.0120 Acc: 0.8000

Epoch 14/80
----------
train Loss: 0.0088 Acc: 0.8820
val Loss: 0.0111 Acc: 0.7923

Epoch 15/80
----------
train Loss: 0.0043 Acc: 0.9049
val Loss: 0.0101 Acc: 0.8000

Epoch 16/80
----------
train Loss: 0.0048 Acc: 0.8951
val Loss: 0.0097 Acc: 0.8000

Epoch 17/80
----------
train Loss: 0.0053 Acc: 0.9016
val Loss: 0.0081 Acc: 0.8077

Epoch 18/80
----------
train Loss: 0.0040 Acc: 0.9115
val Loss: 0.0073 Acc: 0.8077

Epoch 19/80
----------
train Loss: 0.0069 Acc: 0.9082
val Loss: 0.0064 Acc: 0.8308

Epoch 20/80
----------
train Loss: 0.0036 Acc: 0.9049
val Loss: 0.0109 Acc: 0.7923

Epoch 21/80
----------
train Loss: 0.0064 Acc: 0.9016
val Loss: 0.0108 Acc: 0.8077

Epoch 22/80
----------
train Loss: 0.0049 Acc: 0.9049
val Loss: 0.0089 Acc: 0.8308

Epoch 23/80
----------
train Loss: 0.0039 Acc: 0.9279
val Loss: 0.0098 Acc: 0.8308

Epoch 24/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0056 Acc: 0.9279
val Loss: 0.0076 Acc: 0.8308

Epoch 25/80
----------
train Loss: 0.0027 Acc: 0.9344
val Loss: 0.0068 Acc: 0.8308

Epoch 26/80
----------
train Loss: 0.0041 Acc: 0.9311
val Loss: 0.0075 Acc: 0.8231

Epoch 27/80
----------
train Loss: 0.0043 Acc: 0.9311
val Loss: 0.0074 Acc: 0.8385

Epoch 28/80
----------
train Loss: 0.0026 Acc: 0.9279
val Loss: 0.0093 Acc: 0.8308

Epoch 29/80
----------
train Loss: 0.0063 Acc: 0.9148
val Loss: 0.0078 Acc: 0.8231

Epoch 30/80
----------
train Loss: 0.0026 Acc: 0.9279
val Loss: 0.0083 Acc: 0.8231

Epoch 31/80
----------
train Loss: 0.0037 Acc: 0.9246
val Loss: 0.0079 Acc: 0.8308

Epoch 32/80
----------
train Loss: 0.0033 Acc: 0.9049
val Loss: 0.0080 Acc: 0.8385

Epoch 33/80
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0075 Acc: 0.8308

Epoch 34/80
----------
train Loss: 0.0055 Acc: 0.9082
val Loss: 0.0076 Acc: 0.8385

Epoch 35/80
----------
train Loss: 0.0043 Acc: 0.9180
val Loss: 0.0072 Acc: 0.8308

Epoch 36/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0063 Acc: 0.9246
val Loss: 0.0104 Acc: 0.8385

Epoch 37/80
----------
train Loss: 0.0032 Acc: 0.9344
val Loss: 0.0080 Acc: 0.8308

Epoch 38/80
----------
train Loss: 0.0054 Acc: 0.9410
val Loss: 0.0086 Acc: 0.8231

Epoch 39/80
----------
train Loss: 0.0027 Acc: 0.9246
val Loss: 0.0078 Acc: 0.8231

Epoch 40/80
----------
train Loss: 0.0060 Acc: 0.9213
val Loss: 0.0076 Acc: 0.8231

Epoch 41/80
----------
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0075 Acc: 0.8231

Epoch 42/80
----------
train Loss: 0.0041 Acc: 0.9049
val Loss: 0.0100 Acc: 0.8154

Epoch 43/80
----------
train Loss: 0.0045 Acc: 0.9246
val Loss: 0.0077 Acc: 0.8385

Epoch 44/80
----------
train Loss: 0.0068 Acc: 0.9180
val Loss: 0.0088 Acc: 0.8385

Epoch 45/80
----------
train Loss: 0.0057 Acc: 0.9148
val Loss: 0.0082 Acc: 0.8385

Epoch 46/80
----------
train Loss: 0.0026 Acc: 0.9311
val Loss: 0.0090 Acc: 0.8308

Epoch 47/80
----------
train Loss: 0.0035 Acc: 0.9180
val Loss: 0.0081 Acc: 0.8308

Epoch 48/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0038 Acc: 0.9279
val Loss: 0.0087 Acc: 0.8154

Epoch 49/80
----------
train Loss: 0.0028 Acc: 0.9180
val Loss: 0.0081 Acc: 0.8154

Epoch 50/80
----------
train Loss: 0.0049 Acc: 0.9213
val Loss: 0.0087 Acc: 0.8308

Epoch 51/80
----------
train Loss: 0.0033 Acc: 0.9377
val Loss: 0.0098 Acc: 0.8231

Epoch 52/80
----------
train Loss: 0.0042 Acc: 0.9311
val Loss: 0.0094 Acc: 0.8462

Epoch 53/80
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0092 Acc: 0.8308

Epoch 54/80
----------
train Loss: 0.0044 Acc: 0.9213
val Loss: 0.0074 Acc: 0.8308

Epoch 55/80
----------
train Loss: 0.0052 Acc: 0.9213
val Loss: 0.0083 Acc: 0.8462

Epoch 56/80
----------
train Loss: 0.0047 Acc: 0.9180
val Loss: 0.0068 Acc: 0.8308

Epoch 57/80
----------
train Loss: 0.0027 Acc: 0.9279
val Loss: 0.0079 Acc: 0.8308

Epoch 58/80
----------
train Loss: 0.0076 Acc: 0.9180
val Loss: 0.0090 Acc: 0.8154

Epoch 59/80
----------
train Loss: 0.0024 Acc: 0.9279
val Loss: 0.0082 Acc: 0.8000

Epoch 60/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0043 Acc: 0.9344
val Loss: 0.0077 Acc: 0.8154

Epoch 61/80
----------
train Loss: 0.0026 Acc: 0.9279
val Loss: 0.0096 Acc: 0.8154

Epoch 62/80
----------
train Loss: 0.0032 Acc: 0.9115
val Loss: 0.0070 Acc: 0.8615

Epoch 63/80
----------
train Loss: 0.0033 Acc: 0.9082
val Loss: 0.0074 Acc: 0.8462

Epoch 64/80
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0108 Acc: 0.8308

Epoch 65/80
----------
train Loss: 0.0029 Acc: 0.9180
val Loss: 0.0090 Acc: 0.8462

Epoch 66/80
----------
train Loss: 0.0042 Acc: 0.9213
val Loss: 0.0078 Acc: 0.8385

Epoch 67/80
----------
train Loss: 0.0033 Acc: 0.9213
val Loss: 0.0068 Acc: 0.8462

Epoch 68/80
----------
train Loss: 0.0039 Acc: 0.9311
val Loss: 0.0085 Acc: 0.8385

Epoch 69/80
----------
train Loss: 0.0051 Acc: 0.9377
val Loss: 0.0090 Acc: 0.8538

Epoch 70/80
----------
train Loss: 0.0045 Acc: 0.9115
val Loss: 0.0082 Acc: 0.8385

Epoch 71/80
----------
train Loss: 0.0047 Acc: 0.9246
val Loss: 0.0079 Acc: 0.8462

Epoch 72/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0056 Acc: 0.9148
val Loss: 0.0086 Acc: 0.8385

Epoch 73/80
----------
train Loss: 0.0036 Acc: 0.9279
val Loss: 0.0089 Acc: 0.8462

Epoch 74/80
----------
train Loss: 0.0053 Acc: 0.9082
val Loss: 0.0067 Acc: 0.8538

Epoch 75/80
----------
train Loss: 0.0035 Acc: 0.9246
val Loss: 0.0101 Acc: 0.8308

Epoch 76/80
----------
train Loss: 0.0050 Acc: 0.9148
val Loss: 0.0063 Acc: 0.8385

Epoch 77/80
----------
train Loss: 0.0032 Acc: 0.9082
val Loss: 0.0066 Acc: 0.8462

Epoch 78/80
----------
train Loss: 0.0030 Acc: 0.9213
val Loss: 0.0082 Acc: 0.8538

Epoch 79/80
----------
train Loss: 0.0069 Acc: 0.9115
val Loss: 0.0084 Acc: 0.8308

Epoch 80/80
----------
train Loss: 0.0078 Acc: 0.9213
val Loss: 0.0078 Acc: 0.8231

Training complete in 3m 1s
Best val Acc: 0.861538

---Fine tuning.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0027 Acc: 0.9115
val Loss: 0.0067 Acc: 0.8615

Epoch 1/80
----------
train Loss: 0.0040 Acc: 0.9836
val Loss: 0.0083 Acc: 0.8615

Epoch 2/80
----------
train Loss: 0.0051 Acc: 0.9082
val Loss: 0.0344 Acc: 0.5308

Epoch 3/80
----------
train Loss: 0.0037 Acc: 0.8852
val Loss: 0.0410 Acc: 0.4923

Epoch 4/80
----------
train Loss: 0.0047 Acc: 0.9082
val Loss: 0.0221 Acc: 0.5769

Epoch 5/80
----------
train Loss: 0.0054 Acc: 0.9049
val Loss: 0.0452 Acc: 0.4077

Epoch 6/80
----------
train Loss: 0.0043 Acc: 0.8590
val Loss: 0.0693 Acc: 0.3769

Epoch 7/80
----------
train Loss: 0.0061 Acc: 0.9115
val Loss: 0.0419 Acc: 0.4769

Epoch 8/80
----------
train Loss: 0.0077 Acc: 0.8754
val Loss: 0.0488 Acc: 0.5077

Epoch 9/80
----------
train Loss: 0.0147 Acc: 0.7344
val Loss: 0.0863 Acc: 0.3308

Epoch 10/80
----------
train Loss: 0.0091 Acc: 0.7934
val Loss: 0.1069 Acc: 0.2385

Epoch 11/80
----------
train Loss: 0.0106 Acc: 0.7803
val Loss: 0.0536 Acc: 0.4154

Epoch 12/80
----------
LR is set to 0.001
train Loss: 0.0173 Acc: 0.8197
val Loss: 0.0317 Acc: 0.5923

Epoch 13/80
----------
train Loss: 0.0077 Acc: 0.8787
val Loss: 0.0151 Acc: 0.7538

Epoch 14/80
----------
train Loss: 0.0053 Acc: 0.8951
val Loss: 0.0152 Acc: 0.8154

Epoch 15/80
----------
train Loss: 0.0075 Acc: 0.8918
val Loss: 0.0092 Acc: 0.8231

Epoch 16/80
----------
train Loss: 0.0064 Acc: 0.9475
val Loss: 0.0088 Acc: 0.8385

Epoch 17/80
----------
train Loss: 0.0144 Acc: 0.9508
val Loss: 0.0088 Acc: 0.8385

Epoch 18/80
----------
train Loss: 0.0054 Acc: 0.9475
val Loss: 0.0090 Acc: 0.8000

Epoch 19/80
----------
train Loss: 0.0014 Acc: 0.9344
val Loss: 0.0126 Acc: 0.7923

Epoch 20/80
----------
train Loss: 0.0021 Acc: 0.9508
val Loss: 0.0127 Acc: 0.7769

Epoch 21/80
----------
train Loss: 0.0012 Acc: 0.9607
val Loss: 0.0116 Acc: 0.7462

Epoch 22/80
----------
train Loss: 0.0011 Acc: 0.9738
val Loss: 0.0088 Acc: 0.7692

Epoch 23/80
----------
train Loss: 0.0011 Acc: 0.9803
val Loss: 0.0084 Acc: 0.7923

Epoch 24/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9967
val Loss: 0.0094 Acc: 0.8077

Epoch 25/80
----------
train Loss: 0.0010 Acc: 0.9967
val Loss: 0.0096 Acc: 0.8000

Epoch 26/80
----------
train Loss: 0.0059 Acc: 0.9869
val Loss: 0.0075 Acc: 0.8000

Epoch 27/80
----------
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0082 Acc: 0.8077

Epoch 28/80
----------
train Loss: 0.0004 Acc: 0.9967
val Loss: 0.0080 Acc: 0.8154

Epoch 29/80
----------
train Loss: 0.0024 Acc: 0.9902
val Loss: 0.0086 Acc: 0.8231

Epoch 30/80
----------
train Loss: 0.0012 Acc: 0.9934
val Loss: 0.0078 Acc: 0.8538

Epoch 31/80
----------
train Loss: 0.0006 Acc: 0.9902
val Loss: 0.0080 Acc: 0.8462

Epoch 32/80
----------
train Loss: 0.0041 Acc: 0.9934
val Loss: 0.0105 Acc: 0.8462

Epoch 33/80
----------
train Loss: 0.0012 Acc: 0.9934
val Loss: 0.0076 Acc: 0.8385

Epoch 34/80
----------
train Loss: 0.0011 Acc: 0.9902
val Loss: 0.0100 Acc: 0.8538

Epoch 35/80
----------
train Loss: 0.0047 Acc: 0.9902
val Loss: 0.0069 Acc: 0.8538

Epoch 36/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0010 Acc: 0.9934
val Loss: 0.0071 Acc: 0.8692

Epoch 37/80
----------
train Loss: 0.0029 Acc: 0.9902
val Loss: 0.0083 Acc: 0.8615

Epoch 38/80
----------
train Loss: 0.0020 Acc: 0.9902
val Loss: 0.0067 Acc: 0.8385

Epoch 39/80
----------
train Loss: 0.0012 Acc: 0.9967
val Loss: 0.0080 Acc: 0.8462

Epoch 40/80
----------
train Loss: 0.0009 Acc: 0.9934
val Loss: 0.0075 Acc: 0.8615

Epoch 41/80
----------
train Loss: 0.0023 Acc: 0.9934
val Loss: 0.0102 Acc: 0.8154

Epoch 42/80
----------
train Loss: 0.0012 Acc: 0.9902
val Loss: 0.0073 Acc: 0.8077

Epoch 43/80
----------
train Loss: 0.0041 Acc: 0.9934
val Loss: 0.0075 Acc: 0.8154

Epoch 44/80
----------
train Loss: 0.0017 Acc: 0.9934
val Loss: 0.0071 Acc: 0.8077

Epoch 45/80
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8308

Epoch 46/80
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0088 Acc: 0.8385

Epoch 47/80
----------
train Loss: 0.0053 Acc: 0.9934
val Loss: 0.0071 Acc: 0.8077

Epoch 48/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0054 Acc: 0.9902
val Loss: 0.0076 Acc: 0.8385

Epoch 49/80
----------
train Loss: 0.0017 Acc: 0.9902
val Loss: 0.0086 Acc: 0.8385

Epoch 50/80
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0060 Acc: 0.8538

Epoch 51/80
----------
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0095 Acc: 0.8462

Epoch 52/80
----------
train Loss: 0.0031 Acc: 0.9934
val Loss: 0.0070 Acc: 0.8385

Epoch 53/80
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8308

Epoch 54/80
----------
train Loss: 0.0007 Acc: 0.9869
val Loss: 0.0074 Acc: 0.8385

Epoch 55/80
----------
train Loss: 0.0017 Acc: 0.9967
val Loss: 0.0068 Acc: 0.8692

Epoch 56/80
----------
train Loss: 0.0016 Acc: 0.9902
val Loss: 0.0093 Acc: 0.8615

Epoch 57/80
----------
train Loss: 0.0007 Acc: 0.9934
val Loss: 0.0064 Acc: 0.8615

Epoch 58/80
----------
train Loss: 0.0047 Acc: 0.9934
val Loss: 0.0071 Acc: 0.8692

Epoch 59/80
----------
train Loss: 0.0011 Acc: 0.9967
val Loss: 0.0100 Acc: 0.8462

Epoch 60/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8538

Epoch 61/80
----------
train Loss: 0.0027 Acc: 0.9967
val Loss: 0.0093 Acc: 0.8308

Epoch 62/80
----------
train Loss: 0.0013 Acc: 0.9967
val Loss: 0.0100 Acc: 0.8077

Epoch 63/80
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0097 Acc: 0.8385

Epoch 64/80
----------
train Loss: 0.0022 Acc: 0.9836
val Loss: 0.0079 Acc: 0.8154

Epoch 65/80
----------
train Loss: 0.0042 Acc: 0.9902
val Loss: 0.0085 Acc: 0.8385

Epoch 66/80
----------
train Loss: 0.0076 Acc: 0.9836
val Loss: 0.0095 Acc: 0.8308

Epoch 67/80
----------
train Loss: 0.0019 Acc: 0.9902
val Loss: 0.0105 Acc: 0.8538

Epoch 68/80
----------
train Loss: 0.0021 Acc: 0.9934
val Loss: 0.0081 Acc: 0.8385

Epoch 69/80
----------
train Loss: 0.0036 Acc: 0.9902
val Loss: 0.0088 Acc: 0.8462

Epoch 70/80
----------
train Loss: 0.0022 Acc: 0.9836
val Loss: 0.0095 Acc: 0.8692

Epoch 71/80
----------
train Loss: 0.0010 Acc: 0.9967
val Loss: 0.0097 Acc: 0.8615

Epoch 72/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0057 Acc: 0.9869
val Loss: 0.0098 Acc: 0.8308

Epoch 73/80
----------
train Loss: 0.0012 Acc: 0.9934
val Loss: 0.0085 Acc: 0.8538

Epoch 74/80
----------
train Loss: 0.0018 Acc: 0.9967
val Loss: 0.0073 Acc: 0.8462

Epoch 75/80
----------
train Loss: 0.0006 Acc: 0.9967
val Loss: 0.0104 Acc: 0.8615

Epoch 76/80
----------
train Loss: 0.0007 Acc: 0.9934
val Loss: 0.0098 Acc: 0.8692

Epoch 77/80
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8462

Epoch 78/80
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8615

Epoch 79/80
----------
train Loss: 0.0027 Acc: 0.9967
val Loss: 0.0085 Acc: 0.8538

Epoch 80/80
----------
train Loss: 0.0015 Acc: 0.9934
val Loss: 0.0076 Acc: 0.8692

Training complete in 3m 15s
Best val Acc: 0.869231

---Testing---
Test accuracy: 0.960920
--------------------
Accuracy of Dasyatiformes : 93 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 93 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 97 %
mean: 0.9532653678589185, std: 0.019197219499726966

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.98]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 70, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0155 Acc: 0.3393
val Loss: 0.0369 Acc: 0.5116

Epoch 1/69
----------
train Loss: 0.0123 Acc: 0.5893
val Loss: 0.0295 Acc: 0.5814

Epoch 2/69
----------
train Loss: 0.0085 Acc: 0.7168
val Loss: 0.0164 Acc: 0.7907

Epoch 3/69
----------
train Loss: 0.0068 Acc: 0.7985
val Loss: 0.0163 Acc: 0.7209

Epoch 4/69
----------
train Loss: 0.0057 Acc: 0.7908
val Loss: 0.0160 Acc: 0.7907

Epoch 5/69
----------
train Loss: 0.0050 Acc: 0.8393
val Loss: 0.0123 Acc: 0.8837

Epoch 6/69
----------
train Loss: 0.0039 Acc: 0.8699
val Loss: 0.0128 Acc: 0.8605

Epoch 7/69
----------
train Loss: 0.0034 Acc: 0.9107
val Loss: 0.0130 Acc: 0.8372

Epoch 8/69
----------
train Loss: 0.0030 Acc: 0.9260
val Loss: 0.0118 Acc: 0.8372

Epoch 9/69
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0119 Acc: 0.8372

Epoch 10/69
----------
train Loss: 0.0026 Acc: 0.9184
val Loss: 0.0123 Acc: 0.8372

Epoch 11/69
----------
train Loss: 0.0024 Acc: 0.9388
val Loss: 0.0129 Acc: 0.8140

Epoch 12/69
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0114 Acc: 0.8605

Epoch 13/69
----------
train Loss: 0.0020 Acc: 0.9464
val Loss: 0.0120 Acc: 0.8605

Epoch 14/69
----------
LR is set to 0.001
train Loss: 0.0019 Acc: 0.9694
val Loss: 0.0120 Acc: 0.8605

Epoch 15/69
----------
train Loss: 0.0019 Acc: 0.9592
val Loss: 0.0121 Acc: 0.8605

Epoch 16/69
----------
train Loss: 0.0018 Acc: 0.9592
val Loss: 0.0120 Acc: 0.8605

Epoch 17/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0119 Acc: 0.8605

Epoch 18/69
----------
train Loss: 0.0018 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8372

Epoch 19/69
----------
train Loss: 0.0018 Acc: 0.9643
val Loss: 0.0118 Acc: 0.8372

Epoch 20/69
----------
train Loss: 0.0018 Acc: 0.9694
val Loss: 0.0119 Acc: 0.8605

Epoch 21/69
----------
train Loss: 0.0016 Acc: 0.9872
val Loss: 0.0119 Acc: 0.8605

Epoch 22/69
----------
train Loss: 0.0019 Acc: 0.9668
val Loss: 0.0119 Acc: 0.8605

Epoch 23/69
----------
train Loss: 0.0018 Acc: 0.9719
val Loss: 0.0119 Acc: 0.8605

Epoch 24/69
----------
train Loss: 0.0018 Acc: 0.9617
val Loss: 0.0119 Acc: 0.8605

Epoch 25/69
----------
train Loss: 0.0017 Acc: 0.9770
val Loss: 0.0119 Acc: 0.8605

Epoch 26/69
----------
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Epoch 27/69
----------
train Loss: 0.0018 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8605

Epoch 28/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Epoch 29/69
----------
train Loss: 0.0017 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8372

Epoch 30/69
----------
train Loss: 0.0016 Acc: 0.9770
val Loss: 0.0118 Acc: 0.8605

Epoch 31/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 32/69
----------
train Loss: 0.0017 Acc: 0.9770
val Loss: 0.0118 Acc: 0.8605

Epoch 33/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 34/69
----------
train Loss: 0.0016 Acc: 0.9770
val Loss: 0.0118 Acc: 0.8605

Epoch 35/69
----------
train Loss: 0.0019 Acc: 0.9668
val Loss: 0.0118 Acc: 0.8605

Epoch 36/69
----------
train Loss: 0.0017 Acc: 0.9668
val Loss: 0.0118 Acc: 0.8605

Epoch 37/69
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0119 Acc: 0.8605

Epoch 38/69
----------
train Loss: 0.0017 Acc: 0.9770
val Loss: 0.0119 Acc: 0.8605

Epoch 39/69
----------
train Loss: 0.0016 Acc: 0.9821
val Loss: 0.0118 Acc: 0.8605

Epoch 40/69
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0118 Acc: 0.8605

Epoch 41/69
----------
train Loss: 0.0018 Acc: 0.9770
val Loss: 0.0118 Acc: 0.8605

Epoch 42/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 43/69
----------
train Loss: 0.0017 Acc: 0.9643
val Loss: 0.0118 Acc: 0.8605

Epoch 44/69
----------
train Loss: 0.0017 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 45/69
----------
train Loss: 0.0018 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8605

Epoch 46/69
----------
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Epoch 47/69
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0118 Acc: 0.8605

Epoch 48/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 49/69
----------
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Epoch 50/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 51/69
----------
train Loss: 0.0017 Acc: 0.9668
val Loss: 0.0118 Acc: 0.8605

Epoch 52/69
----------
train Loss: 0.0017 Acc: 0.9770
val Loss: 0.0118 Acc: 0.8605

Epoch 53/69
----------
train Loss: 0.0017 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Epoch 54/69
----------
train Loss: 0.0017 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8605

Epoch 55/69
----------
train Loss: 0.0017 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 56/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0119 Acc: 0.8605

Epoch 57/69
----------
train Loss: 0.0016 Acc: 0.9821
val Loss: 0.0118 Acc: 0.8605

Epoch 58/69
----------
train Loss: 0.0016 Acc: 0.9694
val Loss: 0.0119 Acc: 0.8605

Epoch 59/69
----------
train Loss: 0.0016 Acc: 0.9847
val Loss: 0.0118 Acc: 0.8605

Epoch 60/69
----------
train Loss: 0.0016 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8605

Epoch 61/69
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0118 Acc: 0.8605

Epoch 62/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 63/69
----------
train Loss: 0.0017 Acc: 0.9592
val Loss: 0.0118 Acc: 0.8605

Epoch 64/69
----------
train Loss: 0.0016 Acc: 0.9796
val Loss: 0.0118 Acc: 0.8605

Epoch 65/69
----------
train Loss: 0.0016 Acc: 0.9745
val Loss: 0.0118 Acc: 0.8605

Epoch 66/69
----------
train Loss: 0.0018 Acc: 0.9643
val Loss: 0.0118 Acc: 0.8605

Epoch 67/69
----------
train Loss: 0.0018 Acc: 0.9694
val Loss: 0.0118 Acc: 0.8605

Epoch 68/69
----------
train Loss: 0.0017 Acc: 0.9643
val Loss: 0.0118 Acc: 0.8605

Epoch 69/69
----------
train Loss: 0.0016 Acc: 0.9719
val Loss: 0.0118 Acc: 0.8605

Training complete in 2m 26s
Best val Acc: 0.883721

---Fine tuning.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0040 Acc: 0.8776
val Loss: 0.0135 Acc: 0.8605

Epoch 1/69
----------
train Loss: 0.0023 Acc: 0.9439
val Loss: 0.0117 Acc: 0.8837

Epoch 2/69
----------
train Loss: 0.0008 Acc: 0.9974
val Loss: 0.0113 Acc: 0.8605

Epoch 3/69
----------
train Loss: 0.0006 Acc: 0.9923
val Loss: 0.0121 Acc: 0.8837

Epoch 4/69
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 5/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8837

Epoch 6/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 7/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8837

Epoch 8/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8837

Epoch 9/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8837

Epoch 10/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8837

Epoch 11/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 12/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 13/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 14/69
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 15/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 16/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 17/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 18/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 19/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 20/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 21/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 22/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 23/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 24/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 25/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 26/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 27/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 28/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 29/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 30/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 31/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 32/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 33/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 34/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 35/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 36/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 37/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 38/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 39/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 40/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 41/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 42/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 43/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 44/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 45/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 46/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 47/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 48/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 49/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 50/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 51/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 52/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 53/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 54/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 55/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 56/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 57/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 58/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 59/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 60/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 61/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 62/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 63/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 64/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 65/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8837

Epoch 66/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 67/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 68/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Epoch 69/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0128 Acc: 0.8837

Training complete in 2m 41s
Best val Acc: 0.883721

---Testing---
Test accuracy: 0.972414
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 97 %
Accuracy of Rajiformes : 92 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 98 %
mean: 0.9677719244357883, std: 0.023513586961842855
--------------------

run info[val: 0.15, epoch: 96, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0168 Acc: 0.3622
val Loss: 0.0249 Acc: 0.4000

Epoch 1/95
----------
train Loss: 0.0130 Acc: 0.5892
val Loss: 0.0191 Acc: 0.5385

Epoch 2/95
----------
train Loss: 0.0089 Acc: 0.7108
val Loss: 0.0116 Acc: 0.8154

Epoch 3/95
----------
LR is set to 0.001
train Loss: 0.0079 Acc: 0.8135
val Loss: 0.0112 Acc: 0.8154

Epoch 4/95
----------
train Loss: 0.0070 Acc: 0.8081
val Loss: 0.0111 Acc: 0.7385

Epoch 5/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0113 Acc: 0.7385

Epoch 6/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0069 Acc: 0.7784
val Loss: 0.0112 Acc: 0.7385

Epoch 7/95
----------
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0111 Acc: 0.7385

Epoch 8/95
----------
train Loss: 0.0069 Acc: 0.7865
val Loss: 0.0111 Acc: 0.7385

Epoch 9/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0067 Acc: 0.7973
val Loss: 0.0111 Acc: 0.7385

Epoch 10/95
----------
train Loss: 0.0068 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7385

Epoch 11/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7385

Epoch 12/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0067 Acc: 0.7784
val Loss: 0.0110 Acc: 0.7538

Epoch 13/95
----------
train Loss: 0.0068 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 14/95
----------
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 15/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0066 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 16/95
----------
train Loss: 0.0065 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 17/95
----------
train Loss: 0.0068 Acc: 0.8000
val Loss: 0.0111 Acc: 0.7385

Epoch 18/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0111 Acc: 0.7385

Epoch 19/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7385

Epoch 20/95
----------
train Loss: 0.0068 Acc: 0.7811
val Loss: 0.0111 Acc: 0.7385

Epoch 21/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0066 Acc: 0.7919
val Loss: 0.0111 Acc: 0.7538

Epoch 22/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0111 Acc: 0.7538

Epoch 23/95
----------
train Loss: 0.0068 Acc: 0.7865
val Loss: 0.0111 Acc: 0.7538

Epoch 24/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0066 Acc: 0.8081
val Loss: 0.0111 Acc: 0.7538

Epoch 25/95
----------
train Loss: 0.0069 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 26/95
----------
train Loss: 0.0066 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 27/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0067 Acc: 0.7784
val Loss: 0.0110 Acc: 0.7538

Epoch 28/95
----------
train Loss: 0.0066 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 29/95
----------
train Loss: 0.0067 Acc: 0.7865
val Loss: 0.0110 Acc: 0.7385

Epoch 30/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0066 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7385

Epoch 31/95
----------
train Loss: 0.0065 Acc: 0.8027
val Loss: 0.0110 Acc: 0.7538

Epoch 32/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 33/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0064 Acc: 0.8081
val Loss: 0.0110 Acc: 0.7538

Epoch 34/95
----------
train Loss: 0.0065 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 35/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 36/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0067 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 37/95
----------
train Loss: 0.0066 Acc: 0.8000
val Loss: 0.0110 Acc: 0.7538

Epoch 38/95
----------
train Loss: 0.0066 Acc: 0.7946
val Loss: 0.0110 Acc: 0.7538

Epoch 39/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 40/95
----------
train Loss: 0.0067 Acc: 0.8000
val Loss: 0.0110 Acc: 0.7538

Epoch 41/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 42/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0066 Acc: 0.7784
val Loss: 0.0110 Acc: 0.7538

Epoch 43/95
----------
train Loss: 0.0066 Acc: 0.7811
val Loss: 0.0110 Acc: 0.7538

Epoch 44/95
----------
train Loss: 0.0066 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 45/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0065 Acc: 0.8135
val Loss: 0.0110 Acc: 0.7538

Epoch 46/95
----------
train Loss: 0.0067 Acc: 0.7946
val Loss: 0.0110 Acc: 0.7538

Epoch 47/95
----------
train Loss: 0.0067 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 48/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0066 Acc: 0.7919
val Loss: 0.0111 Acc: 0.7538

Epoch 49/95
----------
train Loss: 0.0066 Acc: 0.7946
val Loss: 0.0111 Acc: 0.7538

Epoch 50/95
----------
train Loss: 0.0067 Acc: 0.7946
val Loss: 0.0111 Acc: 0.7538

Epoch 51/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0068 Acc: 0.7703
val Loss: 0.0111 Acc: 0.7538

Epoch 52/95
----------
train Loss: 0.0066 Acc: 0.8027
val Loss: 0.0110 Acc: 0.7538

Epoch 53/95
----------
train Loss: 0.0066 Acc: 0.8054
val Loss: 0.0111 Acc: 0.7538

Epoch 54/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0067 Acc: 0.7811
val Loss: 0.0110 Acc: 0.7538

Epoch 55/95
----------
train Loss: 0.0066 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 56/95
----------
train Loss: 0.0066 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 57/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 58/95
----------
train Loss: 0.0066 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 59/95
----------
train Loss: 0.0068 Acc: 0.7973
val Loss: 0.0111 Acc: 0.7385

Epoch 60/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0065 Acc: 0.7892
val Loss: 0.0111 Acc: 0.7538

Epoch 61/95
----------
train Loss: 0.0067 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 62/95
----------
train Loss: 0.0066 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 63/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7385

Epoch 64/95
----------
train Loss: 0.0067 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 65/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7385

Epoch 66/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0069 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 67/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 68/95
----------
train Loss: 0.0065 Acc: 0.8027
val Loss: 0.0111 Acc: 0.7538

Epoch 69/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0067 Acc: 0.8054
val Loss: 0.0110 Acc: 0.7538

Epoch 70/95
----------
train Loss: 0.0066 Acc: 0.7946
val Loss: 0.0110 Acc: 0.7538

Epoch 71/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 72/95
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0110 Acc: 0.7538

Epoch 73/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 74/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0111 Acc: 0.7538

Epoch 75/95
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0068 Acc: 0.7838
val Loss: 0.0111 Acc: 0.7538

Epoch 76/95
----------
train Loss: 0.0066 Acc: 0.7865
val Loss: 0.0110 Acc: 0.7538

Epoch 77/95
----------
train Loss: 0.0065 Acc: 0.7946
val Loss: 0.0111 Acc: 0.7385

Epoch 78/95
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0067 Acc: 0.7973
val Loss: 0.0111 Acc: 0.7385

Epoch 79/95
----------
train Loss: 0.0068 Acc: 0.7892
val Loss: 0.0111 Acc: 0.7538

Epoch 80/95
----------
train Loss: 0.0066 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 81/95
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0067 Acc: 0.7946
val Loss: 0.0111 Acc: 0.7538

Epoch 82/95
----------
train Loss: 0.0067 Acc: 0.7892
val Loss: 0.0111 Acc: 0.7385

Epoch 83/95
----------
train Loss: 0.0066 Acc: 0.7892
val Loss: 0.0110 Acc: 0.7538

Epoch 84/95
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0066 Acc: 0.7946
val Loss: 0.0110 Acc: 0.7538

Epoch 85/95
----------
train Loss: 0.0067 Acc: 0.7865
val Loss: 0.0110 Acc: 0.7538

Epoch 86/95
----------
train Loss: 0.0066 Acc: 0.8000
val Loss: 0.0110 Acc: 0.7538

Epoch 87/95
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0068 Acc: 0.7811
val Loss: 0.0110 Acc: 0.7538

Epoch 88/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 89/95
----------
train Loss: 0.0067 Acc: 0.7865
val Loss: 0.0110 Acc: 0.7538

Epoch 90/95
----------
LR is set to 1.0000000000000017e-32
train Loss: 0.0068 Acc: 0.7919
val Loss: 0.0110 Acc: 0.7538

Epoch 91/95
----------
train Loss: 0.0066 Acc: 0.7973
val Loss: 0.0110 Acc: 0.7538

Epoch 92/95
----------
train Loss: 0.0067 Acc: 0.7865
val Loss: 0.0110 Acc: 0.7385

Epoch 93/95
----------
LR is set to 1.0000000000000016e-33
train Loss: 0.0066 Acc: 0.8000
val Loss: 0.0110 Acc: 0.7385

Epoch 94/95
----------
train Loss: 0.0069 Acc: 0.7865
val Loss: 0.0111 Acc: 0.7385

Epoch 95/95
----------
train Loss: 0.0067 Acc: 0.7919
val Loss: 0.0111 Acc: 0.7385

Training complete in 3m 25s
Best val Acc: 0.815385

---Fine tuning.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0074 Acc: 0.7892
val Loss: 0.0106 Acc: 0.7846

Epoch 1/95
----------
train Loss: 0.0048 Acc: 0.8486
val Loss: 0.0076 Acc: 0.8769

Epoch 2/95
----------
train Loss: 0.0025 Acc: 0.9703
val Loss: 0.0054 Acc: 0.8923

Epoch 3/95
----------
LR is set to 0.001
train Loss: 0.0016 Acc: 0.9730
val Loss: 0.0052 Acc: 0.9077

Epoch 4/95
----------
train Loss: 0.0012 Acc: 0.9919
val Loss: 0.0051 Acc: 0.8923

Epoch 5/95
----------
train Loss: 0.0011 Acc: 0.9973
val Loss: 0.0051 Acc: 0.8923

Epoch 6/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9946
val Loss: 0.0051 Acc: 0.8923

Epoch 7/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 8/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 9/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 10/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 11/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 12/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 13/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 14/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 15/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 16/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 17/95
----------
train Loss: 0.0010 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 18/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0010 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 19/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 20/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 21/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 22/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 23/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 24/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 25/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 26/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 27/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 28/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 29/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 30/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 31/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 32/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 33/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 34/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 35/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 36/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 37/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 38/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 39/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 40/95
----------
train Loss: 0.0010 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 41/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 42/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 43/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 44/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 45/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 46/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 47/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 48/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 49/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 50/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 51/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 52/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 53/95
----------
train Loss: 0.0010 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 54/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0009 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8923

Epoch 55/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 56/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 57/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 58/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 59/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 60/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 61/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 62/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 63/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 64/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 65/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 66/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 67/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 68/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 69/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 70/95
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 71/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 72/95
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 73/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 74/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 75/95
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0010 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 76/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 77/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 78/95
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 79/95
----------
train Loss: 0.0009 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8923

Epoch 80/95
----------
train Loss: 0.0010 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8923

Epoch 81/95
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 82/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 83/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 84/95
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 85/95
----------
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 86/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 87/95
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 88/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 89/95
----------
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 90/95
----------
LR is set to 1.0000000000000017e-32
train Loss: 0.0009 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 91/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 92/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 93/95
----------
LR is set to 1.0000000000000016e-33
train Loss: 0.0010 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 94/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Epoch 95/95
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8923

Training complete in 3m 47s
Best val Acc: 0.907692

---Testing---
Test accuracy: 0.972414
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 89 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 100 %
mean: 0.9643001977679703, std: 0.03732118142877652
--------------------

run info[val: 0.2, epoch: 83, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0176 Acc: 0.3534
val Loss: 0.0169 Acc: 0.3218

Epoch 1/82
----------
train Loss: 0.0146 Acc: 0.4799
val Loss: 0.0133 Acc: 0.5287

Epoch 2/82
----------
train Loss: 0.0098 Acc: 0.6724
val Loss: 0.0085 Acc: 0.7356

Epoch 3/82
----------
train Loss: 0.0080 Acc: 0.7213
val Loss: 0.0086 Acc: 0.7126

Epoch 4/82
----------
train Loss: 0.0065 Acc: 0.7931
val Loss: 0.0061 Acc: 0.8276

Epoch 5/82
----------
train Loss: 0.0058 Acc: 0.8161
val Loss: 0.0058 Acc: 0.8276

Epoch 6/82
----------
train Loss: 0.0047 Acc: 0.8592
val Loss: 0.0059 Acc: 0.8391

Epoch 7/82
----------
LR is set to 0.001
train Loss: 0.0041 Acc: 0.8707
val Loss: 0.0058 Acc: 0.8506

Epoch 8/82
----------
train Loss: 0.0040 Acc: 0.8851
val Loss: 0.0055 Acc: 0.8506

Epoch 9/82
----------
train Loss: 0.0045 Acc: 0.8822
val Loss: 0.0054 Acc: 0.8391

Epoch 10/82
----------
train Loss: 0.0040 Acc: 0.8851
val Loss: 0.0053 Acc: 0.8391

Epoch 11/82
----------
train Loss: 0.0041 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 12/82
----------
train Loss: 0.0040 Acc: 0.8908
val Loss: 0.0054 Acc: 0.8506

Epoch 13/82
----------
train Loss: 0.0041 Acc: 0.8822
val Loss: 0.0054 Acc: 0.8506

Epoch 14/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0043 Acc: 0.8822
val Loss: 0.0054 Acc: 0.8506

Epoch 15/82
----------
train Loss: 0.0036 Acc: 0.9138
val Loss: 0.0054 Acc: 0.8506

Epoch 16/82
----------
train Loss: 0.0037 Acc: 0.8908
val Loss: 0.0054 Acc: 0.8506

Epoch 17/82
----------
train Loss: 0.0041 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 18/82
----------
train Loss: 0.0038 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 19/82
----------
train Loss: 0.0041 Acc: 0.8937
val Loss: 0.0054 Acc: 0.8506

Epoch 20/82
----------
train Loss: 0.0036 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 21/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0040 Acc: 0.9080
val Loss: 0.0055 Acc: 0.8506

Epoch 22/82
----------
train Loss: 0.0041 Acc: 0.8908
val Loss: 0.0054 Acc: 0.8506

Epoch 23/82
----------
train Loss: 0.0037 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 24/82
----------
train Loss: 0.0042 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8506

Epoch 25/82
----------
train Loss: 0.0040 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 26/82
----------
train Loss: 0.0039 Acc: 0.9138
val Loss: 0.0054 Acc: 0.8506

Epoch 27/82
----------
train Loss: 0.0037 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 28/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0044 Acc: 0.9023
val Loss: 0.0054 Acc: 0.8506

Epoch 29/82
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0054 Acc: 0.8506

Epoch 30/82
----------
train Loss: 0.0038 Acc: 0.8966
val Loss: 0.0055 Acc: 0.8506

Epoch 31/82
----------
train Loss: 0.0038 Acc: 0.8994
val Loss: 0.0055 Acc: 0.8506

Epoch 32/82
----------
train Loss: 0.0036 Acc: 0.9167
val Loss: 0.0055 Acc: 0.8506

Epoch 33/82
----------
train Loss: 0.0038 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8506

Epoch 34/82
----------
train Loss: 0.0039 Acc: 0.9023
val Loss: 0.0054 Acc: 0.8506

Epoch 35/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0042 Acc: 0.8908
val Loss: 0.0054 Acc: 0.8506

Epoch 36/82
----------
train Loss: 0.0037 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 37/82
----------
train Loss: 0.0038 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 38/82
----------
train Loss: 0.0037 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 39/82
----------
train Loss: 0.0036 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 40/82
----------
train Loss: 0.0043 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 41/82
----------
train Loss: 0.0038 Acc: 0.9080
val Loss: 0.0054 Acc: 0.8506

Epoch 42/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 43/82
----------
train Loss: 0.0042 Acc: 0.9023
val Loss: 0.0054 Acc: 0.8506

Epoch 44/82
----------
train Loss: 0.0037 Acc: 0.9138
val Loss: 0.0054 Acc: 0.8506

Epoch 45/82
----------
train Loss: 0.0039 Acc: 0.9023
val Loss: 0.0055 Acc: 0.8506

Epoch 46/82
----------
train Loss: 0.0042 Acc: 0.8822
val Loss: 0.0055 Acc: 0.8506

Epoch 47/82
----------
train Loss: 0.0041 Acc: 0.8764
val Loss: 0.0054 Acc: 0.8506

Epoch 48/82
----------
train Loss: 0.0038 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 49/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0040 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 50/82
----------
train Loss: 0.0041 Acc: 0.8736
val Loss: 0.0054 Acc: 0.8506

Epoch 51/82
----------
train Loss: 0.0044 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 52/82
----------
train Loss: 0.0040 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 53/82
----------
train Loss: 0.0039 Acc: 0.9282
val Loss: 0.0054 Acc: 0.8506

Epoch 54/82
----------
train Loss: 0.0043 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 55/82
----------
train Loss: 0.0040 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 56/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0042 Acc: 0.8764
val Loss: 0.0053 Acc: 0.8621

Epoch 57/82
----------
train Loss: 0.0042 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8621

Epoch 58/82
----------
train Loss: 0.0044 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8621

Epoch 59/82
----------
train Loss: 0.0041 Acc: 0.8966
val Loss: 0.0054 Acc: 0.8506

Epoch 60/82
----------
train Loss: 0.0039 Acc: 0.8966
val Loss: 0.0054 Acc: 0.8506

Epoch 61/82
----------
train Loss: 0.0036 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8621

Epoch 62/82
----------
train Loss: 0.0038 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8621

Epoch 63/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0039 Acc: 0.9109
val Loss: 0.0054 Acc: 0.8506

Epoch 64/82
----------
train Loss: 0.0037 Acc: 0.9138
val Loss: 0.0055 Acc: 0.8506

Epoch 65/82
----------
train Loss: 0.0039 Acc: 0.8937
val Loss: 0.0055 Acc: 0.8506

Epoch 66/82
----------
train Loss: 0.0036 Acc: 0.9253
val Loss: 0.0055 Acc: 0.8506

Epoch 67/82
----------
train Loss: 0.0038 Acc: 0.8994
val Loss: 0.0055 Acc: 0.8506

Epoch 68/82
----------
train Loss: 0.0040 Acc: 0.8879
val Loss: 0.0055 Acc: 0.8506

Epoch 69/82
----------
train Loss: 0.0037 Acc: 0.9224
val Loss: 0.0055 Acc: 0.8506

Epoch 70/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0042 Acc: 0.8994
val Loss: 0.0055 Acc: 0.8506

Epoch 71/82
----------
train Loss: 0.0039 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 72/82
----------
train Loss: 0.0039 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 73/82
----------
train Loss: 0.0039 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8506

Epoch 74/82
----------
train Loss: 0.0040 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 75/82
----------
train Loss: 0.0037 Acc: 0.9138
val Loss: 0.0054 Acc: 0.8506

Epoch 76/82
----------
train Loss: 0.0037 Acc: 0.9052
val Loss: 0.0054 Acc: 0.8506

Epoch 77/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0044 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8506

Epoch 78/82
----------
train Loss: 0.0041 Acc: 0.8937
val Loss: 0.0054 Acc: 0.8621

Epoch 79/82
----------
train Loss: 0.0039 Acc: 0.8937
val Loss: 0.0054 Acc: 0.8621

Epoch 80/82
----------
train Loss: 0.0038 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 81/82
----------
train Loss: 0.0039 Acc: 0.8994
val Loss: 0.0054 Acc: 0.8506

Epoch 82/82
----------
train Loss: 0.0042 Acc: 0.8879
val Loss: 0.0054 Acc: 0.8506

Training complete in 2m 60s
Best val Acc: 0.862069

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0040 Acc: 0.8908
val Loss: 0.0053 Acc: 0.8506

Epoch 1/82
----------
train Loss: 0.0026 Acc: 0.9483
val Loss: 0.0055 Acc: 0.8736

Epoch 2/82
----------
train Loss: 0.0015 Acc: 0.9828
val Loss: 0.0042 Acc: 0.8736

Epoch 3/82
----------
train Loss: 0.0007 Acc: 0.9971
val Loss: 0.0045 Acc: 0.8851

Epoch 4/82
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 5/82
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8851

Epoch 6/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8966

Epoch 7/82
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8851

Epoch 8/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8966

Epoch 9/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 10/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 11/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8736

Epoch 12/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8736

Epoch 13/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8736

Epoch 14/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 15/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 16/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 17/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 18/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 19/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 20/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 21/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.8851

Epoch 22/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8736

Epoch 23/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 24/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 25/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 26/82
----------
train Loss: 0.0002 Acc: 0.9971
val Loss: 0.0038 Acc: 0.8851

Epoch 27/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 28/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 29/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 30/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 31/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 32/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 33/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 34/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 35/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 36/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 37/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 38/82
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.8966

Epoch 39/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 40/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 41/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 42/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 43/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 44/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 45/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 46/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 47/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 48/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 49/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 50/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8966

Epoch 51/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 52/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 53/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 54/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 55/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 56/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 57/82
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.8851

Epoch 58/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 59/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 60/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 61/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 62/82
----------
train Loss: 0.0001 Acc: 0.9971
val Loss: 0.0038 Acc: 0.8851

Epoch 63/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 64/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 65/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 66/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 67/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 68/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 69/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 70/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8736

Epoch 71/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 72/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 73/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 74/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9080

Epoch 75/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9080

Epoch 76/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 77/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8966

Epoch 78/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 79/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 80/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8851

Epoch 81/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8851

Epoch 82/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8736

Training complete in 3m 20s
Best val Acc: 0.908046

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.978096037578467, std: 0.008429719905853828
--------------------

run info[val: 0.25, epoch: 76, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0200 Acc: 0.3028
val Loss: 0.0265 Acc: 0.5000

Epoch 1/75
----------
train Loss: 0.0151 Acc: 0.5321
val Loss: 0.0243 Acc: 0.5000

Epoch 2/75
----------
train Loss: 0.0109 Acc: 0.6606
val Loss: 0.0136 Acc: 0.7222

Epoch 3/75
----------
train Loss: 0.0088 Acc: 0.7615
val Loss: 0.0117 Acc: 0.7315

Epoch 4/75
----------
train Loss: 0.0087 Acc: 0.7554
val Loss: 0.0135 Acc: 0.7315

Epoch 5/75
----------
train Loss: 0.0063 Acc: 0.7982
val Loss: 0.0082 Acc: 0.8056

Epoch 6/75
----------
train Loss: 0.0057 Acc: 0.8685
val Loss: 0.0101 Acc: 0.8148

Epoch 7/75
----------
train Loss: 0.0041 Acc: 0.8991
val Loss: 0.0069 Acc: 0.7963

Epoch 8/75
----------
train Loss: 0.0044 Acc: 0.8930
val Loss: 0.0103 Acc: 0.8333

Epoch 9/75
----------
train Loss: 0.0040 Acc: 0.9113
val Loss: 0.0077 Acc: 0.8056

Epoch 10/75
----------
train Loss: 0.0033 Acc: 0.9052
val Loss: 0.0113 Acc: 0.8704

Epoch 11/75
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.9205
val Loss: 0.0065 Acc: 0.8704

Epoch 12/75
----------
train Loss: 0.0032 Acc: 0.9113
val Loss: 0.0060 Acc: 0.8981

Epoch 13/75
----------
train Loss: 0.0027 Acc: 0.9419
val Loss: 0.0067 Acc: 0.8611

Epoch 14/75
----------
train Loss: 0.0027 Acc: 0.9358
val Loss: 0.0057 Acc: 0.8333

Epoch 15/75
----------
train Loss: 0.0025 Acc: 0.9450
val Loss: 0.0125 Acc: 0.8241

Epoch 16/75
----------
train Loss: 0.0025 Acc: 0.9419
val Loss: 0.0104 Acc: 0.8241

Epoch 17/75
----------
train Loss: 0.0027 Acc: 0.9419
val Loss: 0.0067 Acc: 0.8241

Epoch 18/75
----------
train Loss: 0.0030 Acc: 0.9388
val Loss: 0.0053 Acc: 0.8519

Epoch 19/75
----------
train Loss: 0.0027 Acc: 0.9450
val Loss: 0.0070 Acc: 0.8611

Epoch 20/75
----------
train Loss: 0.0027 Acc: 0.9358
val Loss: 0.0111 Acc: 0.8704

Epoch 21/75
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0057 Acc: 0.8519

Epoch 22/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0064 Acc: 0.8519

Epoch 23/75
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0081 Acc: 0.8611

Epoch 24/75
----------
train Loss: 0.0022 Acc: 0.9480
val Loss: 0.0075 Acc: 0.8426

Epoch 25/75
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0073 Acc: 0.8241

Epoch 26/75
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0056 Acc: 0.8426

Epoch 27/75
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0070 Acc: 0.8241

Epoch 28/75
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0083 Acc: 0.8426

Epoch 29/75
----------
train Loss: 0.0028 Acc: 0.9572
val Loss: 0.0064 Acc: 0.8519

Epoch 30/75
----------
train Loss: 0.0027 Acc: 0.9450
val Loss: 0.0068 Acc: 0.8519

Epoch 31/75
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0066 Acc: 0.8519

Epoch 32/75
----------
train Loss: 0.0021 Acc: 0.9541
val Loss: 0.0080 Acc: 0.8519

Epoch 33/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0121 Acc: 0.8519

Epoch 34/75
----------
train Loss: 0.0027 Acc: 0.9388
val Loss: 0.0077 Acc: 0.8519

Epoch 35/75
----------
train Loss: 0.0024 Acc: 0.9572
val Loss: 0.0084 Acc: 0.8611

Epoch 36/75
----------
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0093 Acc: 0.8611

Epoch 37/75
----------
train Loss: 0.0027 Acc: 0.9572
val Loss: 0.0062 Acc: 0.8426

Epoch 38/75
----------
train Loss: 0.0024 Acc: 0.9694
val Loss: 0.0095 Acc: 0.8519

Epoch 39/75
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0097 Acc: 0.8333

Epoch 40/75
----------
train Loss: 0.0026 Acc: 0.9480
val Loss: 0.0074 Acc: 0.8519

Epoch 41/75
----------
train Loss: 0.0025 Acc: 0.9572
val Loss: 0.0067 Acc: 0.8333

Epoch 42/75
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0094 Acc: 0.8426

Epoch 43/75
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0061 Acc: 0.8519

Epoch 44/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9511
val Loss: 0.0067 Acc: 0.8519

Epoch 45/75
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0086 Acc: 0.8519

Epoch 46/75
----------
train Loss: 0.0029 Acc: 0.9419
val Loss: 0.0065 Acc: 0.8426

Epoch 47/75
----------
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0069 Acc: 0.8426

Epoch 48/75
----------
train Loss: 0.0022 Acc: 0.9450
val Loss: 0.0088 Acc: 0.8426

Epoch 49/75
----------
train Loss: 0.0028 Acc: 0.9572
val Loss: 0.0065 Acc: 0.8426

Epoch 50/75
----------
train Loss: 0.0025 Acc: 0.9511
val Loss: 0.0100 Acc: 0.8426

Epoch 51/75
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0067 Acc: 0.8426

Epoch 52/75
----------
train Loss: 0.0024 Acc: 0.9602
val Loss: 0.0122 Acc: 0.8611

Epoch 53/75
----------
train Loss: 0.0026 Acc: 0.9480
val Loss: 0.0064 Acc: 0.8519

Epoch 54/75
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0106 Acc: 0.8611

Epoch 55/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9358
val Loss: 0.0092 Acc: 0.8333

Epoch 56/75
----------
train Loss: 0.0027 Acc: 0.9602
val Loss: 0.0062 Acc: 0.8333

Epoch 57/75
----------
train Loss: 0.0022 Acc: 0.9541
val Loss: 0.0087 Acc: 0.8611

Epoch 58/75
----------
train Loss: 0.0034 Acc: 0.9235
val Loss: 0.0078 Acc: 0.8519

Epoch 59/75
----------
train Loss: 0.0026 Acc: 0.9419
val Loss: 0.0072 Acc: 0.8519

Epoch 60/75
----------
train Loss: 0.0027 Acc: 0.9541
val Loss: 0.0089 Acc: 0.8519

Epoch 61/75
----------
train Loss: 0.0026 Acc: 0.9511
val Loss: 0.0100 Acc: 0.8333

Epoch 62/75
----------
train Loss: 0.0027 Acc: 0.9511
val Loss: 0.0075 Acc: 0.8519

Epoch 63/75
----------
train Loss: 0.0026 Acc: 0.9541
val Loss: 0.0072 Acc: 0.8519

Epoch 64/75
----------
train Loss: 0.0028 Acc: 0.9388
val Loss: 0.0060 Acc: 0.8519

Epoch 65/75
----------
train Loss: 0.0022 Acc: 0.9602
val Loss: 0.0081 Acc: 0.8519

Epoch 66/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0023 Acc: 0.9511
val Loss: 0.0098 Acc: 0.8519

Epoch 67/75
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0097 Acc: 0.8519

Epoch 68/75
----------
train Loss: 0.0031 Acc: 0.9419
val Loss: 0.0079 Acc: 0.8519

Epoch 69/75
----------
train Loss: 0.0024 Acc: 0.9541
val Loss: 0.0084 Acc: 0.8426

Epoch 70/75
----------
train Loss: 0.0023 Acc: 0.9602
val Loss: 0.0098 Acc: 0.8611

Epoch 71/75
----------
train Loss: 0.0023 Acc: 0.9633
val Loss: 0.0109 Acc: 0.8426

Epoch 72/75
----------
train Loss: 0.0026 Acc: 0.9297
val Loss: 0.0062 Acc: 0.8426

Epoch 73/75
----------
train Loss: 0.0025 Acc: 0.9541
val Loss: 0.0072 Acc: 0.8333

Epoch 74/75
----------
train Loss: 0.0027 Acc: 0.9480
val Loss: 0.0060 Acc: 0.8519

Epoch 75/75
----------
train Loss: 0.0024 Acc: 0.9511
val Loss: 0.0074 Acc: 0.8519

Training complete in 2m 49s
Best val Acc: 0.898148

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0029 Acc: 0.9388
val Loss: 0.0079 Acc: 0.8519

Epoch 1/75
----------
train Loss: 0.0016 Acc: 0.9664
val Loss: 0.0144 Acc: 0.8241

Epoch 2/75
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8796

Epoch 3/75
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0076 Acc: 0.8981

Epoch 4/75
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8796

Epoch 5/75
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9074

Epoch 6/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8889

Epoch 7/75
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8889

Epoch 8/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 9/75
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8889

Epoch 10/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8889

Epoch 11/75
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8981

Epoch 12/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0139 Acc: 0.8889

Epoch 13/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8889

Epoch 14/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8889

Epoch 15/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8889

Epoch 16/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 17/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8889

Epoch 18/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8981

Epoch 19/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8981

Epoch 20/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8981

Epoch 21/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8981

Epoch 22/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 23/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0140 Acc: 0.8981

Epoch 24/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8981

Epoch 25/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8981

Epoch 26/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 27/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0150 Acc: 0.8981

Epoch 28/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0155 Acc: 0.8981

Epoch 29/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0215 Acc: 0.9074

Epoch 30/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9074

Epoch 31/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 32/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8889

Epoch 33/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9074

Epoch 34/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0161 Acc: 0.9074

Epoch 35/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9074

Epoch 36/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.9074

Epoch 37/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0107 Acc: 0.9074

Epoch 38/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 39/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.9074

Epoch 40/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8981

Epoch 41/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 42/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8981

Epoch 43/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8981

Epoch 44/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8889

Epoch 45/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 46/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 47/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8981

Epoch 48/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8981

Epoch 49/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0218 Acc: 0.8981

Epoch 50/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9074

Epoch 51/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.9074

Epoch 52/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8981

Epoch 53/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8981

Epoch 54/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8981

Epoch 55/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 56/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8981

Epoch 57/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8889

Epoch 58/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8889

Epoch 59/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8889

Epoch 60/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8981

Epoch 61/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8981

Epoch 62/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8889

Epoch 63/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8889

Epoch 64/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8889

Epoch 65/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8981

Epoch 66/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0137 Acc: 0.8981

Epoch 67/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9074

Epoch 68/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.9074

Epoch 69/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8981

Epoch 70/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8889

Epoch 71/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8981

Epoch 72/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0126 Acc: 0.9074

Epoch 73/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0173 Acc: 0.9074

Epoch 74/75
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8981

Epoch 75/75
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9074

Training complete in 3m 8s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 97 %
mean: 0.9757568562919172, std: 0.006722370636501139
--------------------

run info[val: 0.3, epoch: 88, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0208 Acc: 0.3148
val Loss: 0.0258 Acc: 0.2538

Epoch 1/87
----------
train Loss: 0.0188 Acc: 0.3279
val Loss: 0.0253 Acc: 0.5000

Epoch 2/87
----------
train Loss: 0.0172 Acc: 0.4787
val Loss: 0.0172 Acc: 0.6308

Epoch 3/87
----------
train Loss: 0.0146 Acc: 0.6426
val Loss: 0.0113 Acc: 0.7692

Epoch 4/87
----------
LR is set to 0.001
train Loss: 0.0100 Acc: 0.7443
val Loss: 0.0106 Acc: 0.7923

Epoch 5/87
----------
train Loss: 0.0087 Acc: 0.8230
val Loss: 0.0105 Acc: 0.7538

Epoch 6/87
----------
train Loss: 0.0085 Acc: 0.8230
val Loss: 0.0126 Acc: 0.7308

Epoch 7/87
----------
train Loss: 0.0076 Acc: 0.7836
val Loss: 0.0100 Acc: 0.7462

Epoch 8/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0076 Acc: 0.7836
val Loss: 0.0121 Acc: 0.7462

Epoch 9/87
----------
train Loss: 0.0092 Acc: 0.8033
val Loss: 0.0123 Acc: 0.7385

Epoch 10/87
----------
train Loss: 0.0086 Acc: 0.7803
val Loss: 0.0107 Acc: 0.7615

Epoch 11/87
----------
train Loss: 0.0095 Acc: 0.7869
val Loss: 0.0113 Acc: 0.7692

Epoch 12/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0091 Acc: 0.8131
val Loss: 0.0118 Acc: 0.7615

Epoch 13/87
----------
train Loss: 0.0087 Acc: 0.7672
val Loss: 0.0107 Acc: 0.7538

Epoch 14/87
----------
train Loss: 0.0063 Acc: 0.8000
val Loss: 0.0096 Acc: 0.7538

Epoch 15/87
----------
train Loss: 0.0127 Acc: 0.7803
val Loss: 0.0107 Acc: 0.7615

Epoch 16/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0089 Acc: 0.7836
val Loss: 0.0104 Acc: 0.7615

Epoch 17/87
----------
train Loss: 0.0077 Acc: 0.7967
val Loss: 0.0123 Acc: 0.7692

Epoch 18/87
----------
train Loss: 0.0088 Acc: 0.8131
val Loss: 0.0094 Acc: 0.7692

Epoch 19/87
----------
train Loss: 0.0080 Acc: 0.7902
val Loss: 0.0112 Acc: 0.7615

Epoch 20/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0088 Acc: 0.7902
val Loss: 0.0096 Acc: 0.7615

Epoch 21/87
----------
train Loss: 0.0092 Acc: 0.8066
val Loss: 0.0106 Acc: 0.7538

Epoch 22/87
----------
train Loss: 0.0071 Acc: 0.7770
val Loss: 0.0099 Acc: 0.7615

Epoch 23/87
----------
train Loss: 0.0067 Acc: 0.7967
val Loss: 0.0098 Acc: 0.7615

Epoch 24/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0076 Acc: 0.7869
val Loss: 0.0116 Acc: 0.7538

Epoch 25/87
----------
train Loss: 0.0084 Acc: 0.7869
val Loss: 0.0108 Acc: 0.7538

Epoch 26/87
----------
train Loss: 0.0084 Acc: 0.8000
val Loss: 0.0121 Acc: 0.7615

Epoch 27/87
----------
train Loss: 0.0067 Acc: 0.7902
val Loss: 0.0103 Acc: 0.7615

Epoch 28/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0073 Acc: 0.8098
val Loss: 0.0112 Acc: 0.7615

Epoch 29/87
----------
train Loss: 0.0099 Acc: 0.7836
val Loss: 0.0108 Acc: 0.7692

Epoch 30/87
----------
train Loss: 0.0065 Acc: 0.7770
val Loss: 0.0105 Acc: 0.7692

Epoch 31/87
----------
train Loss: 0.0082 Acc: 0.7639
val Loss: 0.0110 Acc: 0.7692

Epoch 32/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0076 Acc: 0.7869
val Loss: 0.0114 Acc: 0.7615

Epoch 33/87
----------
train Loss: 0.0069 Acc: 0.7803
val Loss: 0.0110 Acc: 0.7615

Epoch 34/87
----------
train Loss: 0.0071 Acc: 0.7869
val Loss: 0.0107 Acc: 0.7692

Epoch 35/87
----------
train Loss: 0.0074 Acc: 0.7803
val Loss: 0.0101 Acc: 0.7615

Epoch 36/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0082 Acc: 0.7902
val Loss: 0.0111 Acc: 0.7615

Epoch 37/87
----------
train Loss: 0.0091 Acc: 0.8098
val Loss: 0.0102 Acc: 0.7538

Epoch 38/87
----------
train Loss: 0.0064 Acc: 0.7869
val Loss: 0.0113 Acc: 0.7615

Epoch 39/87
----------
train Loss: 0.0075 Acc: 0.8230
val Loss: 0.0105 Acc: 0.7538

Epoch 40/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0093 Acc: 0.8000
val Loss: 0.0114 Acc: 0.7538

Epoch 41/87
----------
train Loss: 0.0072 Acc: 0.7803
val Loss: 0.0102 Acc: 0.7692

Epoch 42/87
----------
train Loss: 0.0080 Acc: 0.7869
val Loss: 0.0109 Acc: 0.7538

Epoch 43/87
----------
train Loss: 0.0094 Acc: 0.7902
val Loss: 0.0112 Acc: 0.7462

Epoch 44/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0077 Acc: 0.7902
val Loss: 0.0106 Acc: 0.7615

Epoch 45/87
----------
train Loss: 0.0088 Acc: 0.8033
val Loss: 0.0088 Acc: 0.7615

Epoch 46/87
----------
train Loss: 0.0064 Acc: 0.7672
val Loss: 0.0118 Acc: 0.7615

Epoch 47/87
----------
train Loss: 0.0084 Acc: 0.7770
val Loss: 0.0111 Acc: 0.7615

Epoch 48/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0069 Acc: 0.8000
val Loss: 0.0104 Acc: 0.7615

Epoch 49/87
----------
train Loss: 0.0086 Acc: 0.7934
val Loss: 0.0120 Acc: 0.7538

Epoch 50/87
----------
train Loss: 0.0092 Acc: 0.7836
val Loss: 0.0114 Acc: 0.7615

Epoch 51/87
----------
train Loss: 0.0068 Acc: 0.7934
val Loss: 0.0110 Acc: 0.7692

Epoch 52/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0081 Acc: 0.7770
val Loss: 0.0112 Acc: 0.7692

Epoch 53/87
----------
train Loss: 0.0095 Acc: 0.7967
val Loss: 0.0112 Acc: 0.7692

Epoch 54/87
----------
train Loss: 0.0086 Acc: 0.8098
val Loss: 0.0104 Acc: 0.7692

Epoch 55/87
----------
train Loss: 0.0088 Acc: 0.7869
val Loss: 0.0100 Acc: 0.7615

Epoch 56/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0067 Acc: 0.8098
val Loss: 0.0110 Acc: 0.7692

Epoch 57/87
----------
train Loss: 0.0086 Acc: 0.8000
val Loss: 0.0103 Acc: 0.7692

Epoch 58/87
----------
train Loss: 0.0111 Acc: 0.7902
val Loss: 0.0109 Acc: 0.7615

Epoch 59/87
----------
train Loss: 0.0069 Acc: 0.7967
val Loss: 0.0124 Acc: 0.7615

Epoch 60/87
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0094 Acc: 0.7934
val Loss: 0.0110 Acc: 0.7615

Epoch 61/87
----------
train Loss: 0.0063 Acc: 0.8000
val Loss: 0.0108 Acc: 0.7615

Epoch 62/87
----------
train Loss: 0.0124 Acc: 0.7869
val Loss: 0.0127 Acc: 0.7615

Epoch 63/87
----------
train Loss: 0.0085 Acc: 0.7770
val Loss: 0.0101 Acc: 0.7692

Epoch 64/87
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0063 Acc: 0.7770
val Loss: 0.0101 Acc: 0.7538

Epoch 65/87
----------
train Loss: 0.0080 Acc: 0.8098
val Loss: 0.0093 Acc: 0.7538

Epoch 66/87
----------
train Loss: 0.0066 Acc: 0.7934
val Loss: 0.0107 Acc: 0.7615

Epoch 67/87
----------
train Loss: 0.0063 Acc: 0.7902
val Loss: 0.0116 Acc: 0.7615

Epoch 68/87
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0062 Acc: 0.8131
val Loss: 0.0095 Acc: 0.7692

Epoch 69/87
----------
train Loss: 0.0064 Acc: 0.7934
val Loss: 0.0102 Acc: 0.7615

Epoch 70/87
----------
train Loss: 0.0101 Acc: 0.8066
val Loss: 0.0113 Acc: 0.7692

Epoch 71/87
----------
train Loss: 0.0110 Acc: 0.7902
val Loss: 0.0102 Acc: 0.7692

Epoch 72/87
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0075 Acc: 0.8098
val Loss: 0.0099 Acc: 0.7615

Epoch 73/87
----------
train Loss: 0.0109 Acc: 0.7934
val Loss: 0.0111 Acc: 0.7615

Epoch 74/87
----------
train Loss: 0.0072 Acc: 0.8066
val Loss: 0.0118 Acc: 0.7615

Epoch 75/87
----------
train Loss: 0.0067 Acc: 0.7967
val Loss: 0.0108 Acc: 0.7615

Epoch 76/87
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0096 Acc: 0.7934
val Loss: 0.0120 Acc: 0.7462

Epoch 77/87
----------
train Loss: 0.0076 Acc: 0.7967
val Loss: 0.0099 Acc: 0.7769

Epoch 78/87
----------
train Loss: 0.0087 Acc: 0.8000
val Loss: 0.0110 Acc: 0.7769

Epoch 79/87
----------
train Loss: 0.0063 Acc: 0.7934
val Loss: 0.0130 Acc: 0.7769

Epoch 80/87
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0068 Acc: 0.8131
val Loss: 0.0101 Acc: 0.7615

Epoch 81/87
----------
train Loss: 0.0093 Acc: 0.7869
val Loss: 0.0096 Acc: 0.7538

Epoch 82/87
----------
train Loss: 0.0070 Acc: 0.8000
val Loss: 0.0103 Acc: 0.7615

Epoch 83/87
----------
train Loss: 0.0068 Acc: 0.7869
val Loss: 0.0103 Acc: 0.7692

Epoch 84/87
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0085 Acc: 0.8066
val Loss: 0.0122 Acc: 0.7615

Epoch 85/87
----------
train Loss: 0.0097 Acc: 0.8033
val Loss: 0.0109 Acc: 0.7462

Epoch 86/87
----------
train Loss: 0.0091 Acc: 0.7803
val Loss: 0.0125 Acc: 0.7615

Epoch 87/87
----------
train Loss: 0.0097 Acc: 0.8033
val Loss: 0.0098 Acc: 0.7615

Training complete in 3m 21s
Best val Acc: 0.792308

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0069 Acc: 0.8098
val Loss: 0.0122 Acc: 0.7000

Epoch 1/87
----------
train Loss: 0.0064 Acc: 0.7967
val Loss: 0.0090 Acc: 0.8000

Epoch 2/87
----------
train Loss: 0.0057 Acc: 0.9016
val Loss: 0.0108 Acc: 0.8000

Epoch 3/87
----------
train Loss: 0.0067 Acc: 0.8951
val Loss: 0.0267 Acc: 0.6308

Epoch 4/87
----------
LR is set to 0.001
train Loss: 0.0051 Acc: 0.9180
val Loss: 0.0170 Acc: 0.7000

Epoch 5/87
----------
train Loss: 0.0025 Acc: 0.9410
val Loss: 0.0105 Acc: 0.7846

Epoch 6/87
----------
train Loss: 0.0043 Acc: 0.9443
val Loss: 0.0055 Acc: 0.8692

Epoch 7/87
----------
train Loss: 0.0017 Acc: 0.9607
val Loss: 0.0052 Acc: 0.8769

Epoch 8/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0068 Acc: 0.9705
val Loss: 0.0063 Acc: 0.8692

Epoch 9/87
----------
train Loss: 0.0024 Acc: 0.9738
val Loss: 0.0073 Acc: 0.8846

Epoch 10/87
----------
train Loss: 0.0033 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8692

Epoch 11/87
----------
train Loss: 0.0019 Acc: 0.9607
val Loss: 0.0060 Acc: 0.8692

Epoch 12/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0067 Acc: 0.9738
val Loss: 0.0072 Acc: 0.8769

Epoch 13/87
----------
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0065 Acc: 0.8692

Epoch 14/87
----------
train Loss: 0.0042 Acc: 0.9803
val Loss: 0.0049 Acc: 0.8769

Epoch 15/87
----------
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0066 Acc: 0.8923

Epoch 16/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0020 Acc: 0.9770
val Loss: 0.0073 Acc: 0.8692

Epoch 17/87
----------
train Loss: 0.0014 Acc: 0.9738
val Loss: 0.0061 Acc: 0.8846

Epoch 18/87
----------
train Loss: 0.0025 Acc: 0.9672
val Loss: 0.0066 Acc: 0.8846

Epoch 19/87
----------
train Loss: 0.0012 Acc: 0.9738
val Loss: 0.0064 Acc: 0.8769

Epoch 20/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0062 Acc: 0.8769

Epoch 21/87
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0058 Acc: 0.8692

Epoch 22/87
----------
train Loss: 0.0023 Acc: 0.9836
val Loss: 0.0062 Acc: 0.8846

Epoch 23/87
----------
train Loss: 0.0012 Acc: 0.9803
val Loss: 0.0053 Acc: 0.8846

Epoch 24/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0037 Acc: 0.9639
val Loss: 0.0066 Acc: 0.8769

Epoch 25/87
----------
train Loss: 0.0019 Acc: 0.9902
val Loss: 0.0053 Acc: 0.8769

Epoch 26/87
----------
train Loss: 0.0016 Acc: 0.9803
val Loss: 0.0055 Acc: 0.8692

Epoch 27/87
----------
train Loss: 0.0049 Acc: 0.9705
val Loss: 0.0074 Acc: 0.8769

Epoch 28/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0031 Acc: 0.9770
val Loss: 0.0069 Acc: 0.8538

Epoch 29/87
----------
train Loss: 0.0013 Acc: 0.9836
val Loss: 0.0076 Acc: 0.8769

Epoch 30/87
----------
train Loss: 0.0020 Acc: 0.9770
val Loss: 0.0050 Acc: 0.8692

Epoch 31/87
----------
train Loss: 0.0020 Acc: 0.9672
val Loss: 0.0060 Acc: 0.8769

Epoch 32/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0009 Acc: 0.9934
val Loss: 0.0079 Acc: 0.8846

Epoch 33/87
----------
train Loss: 0.0033 Acc: 0.9869
val Loss: 0.0058 Acc: 0.8692

Epoch 34/87
----------
train Loss: 0.0040 Acc: 0.9803
val Loss: 0.0050 Acc: 0.8692

Epoch 35/87
----------
train Loss: 0.0025 Acc: 0.9803
val Loss: 0.0057 Acc: 0.8538

Epoch 36/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0013 Acc: 0.9770
val Loss: 0.0061 Acc: 0.8615

Epoch 37/87
----------
train Loss: 0.0019 Acc: 0.9836
val Loss: 0.0066 Acc: 0.8615

Epoch 38/87
----------
train Loss: 0.0033 Acc: 0.9869
val Loss: 0.0061 Acc: 0.8923

Epoch 39/87
----------
train Loss: 0.0023 Acc: 0.9770
val Loss: 0.0081 Acc: 0.8846

Epoch 40/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0023 Acc: 0.9607
val Loss: 0.0075 Acc: 0.8769

Epoch 41/87
----------
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0053 Acc: 0.8769

Epoch 42/87
----------
train Loss: 0.0068 Acc: 0.9639
val Loss: 0.0047 Acc: 0.8692

Epoch 43/87
----------
train Loss: 0.0043 Acc: 0.9770
val Loss: 0.0082 Acc: 0.8769

Epoch 44/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0020 Acc: 0.9738
val Loss: 0.0057 Acc: 0.8769

Epoch 45/87
----------
train Loss: 0.0050 Acc: 0.9738
val Loss: 0.0065 Acc: 0.8846

Epoch 46/87
----------
train Loss: 0.0019 Acc: 0.9705
val Loss: 0.0066 Acc: 0.8846

Epoch 47/87
----------
train Loss: 0.0040 Acc: 0.9803
val Loss: 0.0061 Acc: 0.8846

Epoch 48/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0064 Acc: 0.8692

Epoch 49/87
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8615

Epoch 50/87
----------
train Loss: 0.0014 Acc: 0.9738
val Loss: 0.0053 Acc: 0.8769

Epoch 51/87
----------
train Loss: 0.0020 Acc: 0.9770
val Loss: 0.0058 Acc: 0.8846

Epoch 52/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0023 Acc: 0.9770
val Loss: 0.0055 Acc: 0.8923

Epoch 53/87
----------
train Loss: 0.0025 Acc: 0.9836
val Loss: 0.0050 Acc: 0.8846

Epoch 54/87
----------
train Loss: 0.0025 Acc: 0.9869
val Loss: 0.0048 Acc: 0.8692

Epoch 55/87
----------
train Loss: 0.0019 Acc: 0.9738
val Loss: 0.0063 Acc: 0.8846

Epoch 56/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0056 Acc: 0.8769

Epoch 57/87
----------
train Loss: 0.0023 Acc: 0.9705
val Loss: 0.0076 Acc: 0.8846

Epoch 58/87
----------
train Loss: 0.0053 Acc: 0.9705
val Loss: 0.0059 Acc: 0.8923

Epoch 59/87
----------
train Loss: 0.0015 Acc: 0.9770
val Loss: 0.0060 Acc: 0.8769

Epoch 60/87
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0023 Acc: 0.9836
val Loss: 0.0061 Acc: 0.8692

Epoch 61/87
----------
train Loss: 0.0033 Acc: 0.9902
val Loss: 0.0058 Acc: 0.8769

Epoch 62/87
----------
train Loss: 0.0013 Acc: 0.9869
val Loss: 0.0056 Acc: 0.8846

Epoch 63/87
----------
train Loss: 0.0014 Acc: 0.9869
val Loss: 0.0058 Acc: 0.8846

Epoch 64/87
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0098 Acc: 0.9738
val Loss: 0.0048 Acc: 0.8846

Epoch 65/87
----------
train Loss: 0.0019 Acc: 0.9902
val Loss: 0.0066 Acc: 0.8769

Epoch 66/87
----------
train Loss: 0.0034 Acc: 0.9705
val Loss: 0.0060 Acc: 0.8769

Epoch 67/87
----------
train Loss: 0.0014 Acc: 0.9770
val Loss: 0.0062 Acc: 0.8769

Epoch 68/87
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0073 Acc: 0.8769

Epoch 69/87
----------
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0054 Acc: 0.8846

Epoch 70/87
----------
train Loss: 0.0011 Acc: 0.9869
val Loss: 0.0068 Acc: 0.8692

Epoch 71/87
----------
train Loss: 0.0013 Acc: 0.9738
val Loss: 0.0072 Acc: 0.8846

Epoch 72/87
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0065 Acc: 0.8846

Epoch 73/87
----------
train Loss: 0.0011 Acc: 0.9836
val Loss: 0.0082 Acc: 0.8846

Epoch 74/87
----------
train Loss: 0.0009 Acc: 0.9967
val Loss: 0.0064 Acc: 0.8769

Epoch 75/87
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0053 Acc: 0.8846

Epoch 76/87
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0037 Acc: 0.9705
val Loss: 0.0073 Acc: 0.8769

Epoch 77/87
----------
train Loss: 0.0016 Acc: 0.9803
val Loss: 0.0056 Acc: 0.8769

Epoch 78/87
----------
train Loss: 0.0033 Acc: 0.9803
val Loss: 0.0059 Acc: 0.8769

Epoch 79/87
----------
train Loss: 0.0055 Acc: 0.9738
val Loss: 0.0055 Acc: 0.8846

Epoch 80/87
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0015 Acc: 0.9705
val Loss: 0.0070 Acc: 0.8692

Epoch 81/87
----------
train Loss: 0.0014 Acc: 0.9770
val Loss: 0.0061 Acc: 0.8846

Epoch 82/87
----------
train Loss: 0.0012 Acc: 0.9902
val Loss: 0.0055 Acc: 0.8923

Epoch 83/87
----------
train Loss: 0.0026 Acc: 0.9770
val Loss: 0.0060 Acc: 0.8769

Epoch 84/87
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0008 Acc: 0.9967
val Loss: 0.0051 Acc: 0.8692

Epoch 85/87
----------
train Loss: 0.0011 Acc: 0.9869
val Loss: 0.0065 Acc: 0.8692

Epoch 86/87
----------
train Loss: 0.0017 Acc: 0.9934
val Loss: 0.0046 Acc: 0.8769

Epoch 87/87
----------
train Loss: 0.0033 Acc: 0.9803
val Loss: 0.0046 Acc: 0.8769

Training complete in 3m 33s
Best val Acc: 0.892308

---Testing---
Test accuracy: 0.960920
--------------------
Accuracy of Dasyatiformes : 89 %
Accuracy of Myliobatiformes : 94 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 97 %
mean: 0.9502625166307462, std: 0.02857834017454916

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.98]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 98, randcrop: False, decay: 8]

---Training last layer.---
Epoch 0/97
----------
LR is set to 0.01
train Loss: 0.0154 Acc: 0.3699
val Loss: 0.0372 Acc: 0.4419

Epoch 1/97
----------
train Loss: 0.0118 Acc: 0.6224
val Loss: 0.0288 Acc: 0.6279

Epoch 2/97
----------
train Loss: 0.0085 Acc: 0.7092
val Loss: 0.0184 Acc: 0.6744

Epoch 3/97
----------
train Loss: 0.0073 Acc: 0.7449
val Loss: 0.0156 Acc: 0.7674

Epoch 4/97
----------
train Loss: 0.0054 Acc: 0.8189
val Loss: 0.0136 Acc: 0.8372

Epoch 5/97
----------
train Loss: 0.0044 Acc: 0.8495
val Loss: 0.0129 Acc: 0.8140

Epoch 6/97
----------
train Loss: 0.0039 Acc: 0.8750
val Loss: 0.0127 Acc: 0.8140

Epoch 7/97
----------
train Loss: 0.0035 Acc: 0.9056
val Loss: 0.0113 Acc: 0.8605

Epoch 8/97
----------
LR is set to 0.001
train Loss: 0.0030 Acc: 0.8980
val Loss: 0.0113 Acc: 0.8837

Epoch 9/97
----------
train Loss: 0.0030 Acc: 0.9158
val Loss: 0.0114 Acc: 0.8605

Epoch 10/97
----------
train Loss: 0.0029 Acc: 0.9133
val Loss: 0.0115 Acc: 0.8605

Epoch 11/97
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0117 Acc: 0.8372

Epoch 12/97
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0118 Acc: 0.8372

Epoch 13/97
----------
train Loss: 0.0029 Acc: 0.9133
val Loss: 0.0117 Acc: 0.8372

Epoch 14/97
----------
train Loss: 0.0029 Acc: 0.9362
val Loss: 0.0116 Acc: 0.8372

Epoch 15/97
----------
train Loss: 0.0027 Acc: 0.9464
val Loss: 0.0115 Acc: 0.8372

Epoch 16/97
----------
LR is set to 0.00010000000000000002
train Loss: 0.0026 Acc: 0.9362
val Loss: 0.0114 Acc: 0.8605

Epoch 17/97
----------
train Loss: 0.0026 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 18/97
----------
train Loss: 0.0028 Acc: 0.9184
val Loss: 0.0114 Acc: 0.8605

Epoch 19/97
----------
train Loss: 0.0027 Acc: 0.9286
val Loss: 0.0114 Acc: 0.8605

Epoch 20/97
----------
train Loss: 0.0026 Acc: 0.9464
val Loss: 0.0114 Acc: 0.8605

Epoch 21/97
----------
train Loss: 0.0026 Acc: 0.9286
val Loss: 0.0114 Acc: 0.8605

Epoch 22/97
----------
train Loss: 0.0026 Acc: 0.9286
val Loss: 0.0114 Acc: 0.8605

Epoch 23/97
----------
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 24/97
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8605

Epoch 25/97
----------
train Loss: 0.0028 Acc: 0.9209
val Loss: 0.0115 Acc: 0.8605

Epoch 26/97
----------
train Loss: 0.0027 Acc: 0.9133
val Loss: 0.0115 Acc: 0.8372

Epoch 27/97
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8372

Epoch 28/97
----------
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0115 Acc: 0.8372

Epoch 29/97
----------
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 30/97
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8372

Epoch 31/97
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0115 Acc: 0.8372

Epoch 32/97
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 33/97
----------
train Loss: 0.0026 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8605

Epoch 34/97
----------
train Loss: 0.0028 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8605

Epoch 35/97
----------
train Loss: 0.0027 Acc: 0.9184
val Loss: 0.0115 Acc: 0.8372

Epoch 36/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8605

Epoch 37/97
----------
train Loss: 0.0027 Acc: 0.9439
val Loss: 0.0114 Acc: 0.8372

Epoch 38/97
----------
train Loss: 0.0026 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 39/97
----------
train Loss: 0.0028 Acc: 0.9235
val Loss: 0.0114 Acc: 0.8605

Epoch 40/97
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0026 Acc: 0.9260
val Loss: 0.0114 Acc: 0.8605

Epoch 41/97
----------
train Loss: 0.0026 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 42/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8605

Epoch 43/97
----------
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 44/97
----------
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0115 Acc: 0.8372

Epoch 45/97
----------
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0115 Acc: 0.8605

Epoch 46/97
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8372

Epoch 47/97
----------
train Loss: 0.0027 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8605

Epoch 48/97
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0028 Acc: 0.9133
val Loss: 0.0114 Acc: 0.8605

Epoch 49/97
----------
train Loss: 0.0027 Acc: 0.9286
val Loss: 0.0114 Acc: 0.8605

Epoch 50/97
----------
train Loss: 0.0026 Acc: 0.9413
val Loss: 0.0115 Acc: 0.8605

Epoch 51/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8372

Epoch 52/97
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8605

Epoch 53/97
----------
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8605

Epoch 54/97
----------
train Loss: 0.0028 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 55/97
----------
train Loss: 0.0026 Acc: 0.9260
val Loss: 0.0114 Acc: 0.8605

Epoch 56/97
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0026 Acc: 0.9490
val Loss: 0.0115 Acc: 0.8605

Epoch 57/97
----------
train Loss: 0.0027 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8605

Epoch 58/97
----------
train Loss: 0.0026 Acc: 0.9388
val Loss: 0.0114 Acc: 0.8605

Epoch 59/97
----------
train Loss: 0.0026 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 60/97
----------
train Loss: 0.0026 Acc: 0.9388
val Loss: 0.0115 Acc: 0.8605

Epoch 61/97
----------
train Loss: 0.0027 Acc: 0.9439
val Loss: 0.0115 Acc: 0.8372

Epoch 62/97
----------
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 63/97
----------
train Loss: 0.0027 Acc: 0.9184
val Loss: 0.0115 Acc: 0.8605

Epoch 64/97
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8605

Epoch 65/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8605

Epoch 66/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8605

Epoch 67/97
----------
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0114 Acc: 0.8605

Epoch 68/97
----------
train Loss: 0.0026 Acc: 0.9439
val Loss: 0.0114 Acc: 0.8605

Epoch 69/97
----------
train Loss: 0.0027 Acc: 0.9413
val Loss: 0.0114 Acc: 0.8605

Epoch 70/97
----------
train Loss: 0.0027 Acc: 0.9413
val Loss: 0.0114 Acc: 0.8605

Epoch 71/97
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8605

Epoch 72/97
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0026 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 73/97
----------
train Loss: 0.0030 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8605

Epoch 74/97
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8605

Epoch 75/97
----------
train Loss: 0.0027 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8605

Epoch 76/97
----------
train Loss: 0.0027 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8372

Epoch 77/97
----------
train Loss: 0.0026 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8372

Epoch 78/97
----------
train Loss: 0.0028 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8372

Epoch 79/97
----------
train Loss: 0.0027 Acc: 0.9184
val Loss: 0.0114 Acc: 0.8605

Epoch 80/97
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8372

Epoch 81/97
----------
train Loss: 0.0028 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8372

Epoch 82/97
----------
train Loss: 0.0028 Acc: 0.9260
val Loss: 0.0115 Acc: 0.8372

Epoch 83/97
----------
train Loss: 0.0028 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8372

Epoch 84/97
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8372

Epoch 85/97
----------
train Loss: 0.0026 Acc: 0.9311
val Loss: 0.0114 Acc: 0.8605

Epoch 86/97
----------
train Loss: 0.0030 Acc: 0.9337
val Loss: 0.0114 Acc: 0.8605

Epoch 87/97
----------
train Loss: 0.0028 Acc: 0.9362
val Loss: 0.0114 Acc: 0.8605

Epoch 88/97
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0027 Acc: 0.9286
val Loss: 0.0115 Acc: 0.8605

Epoch 89/97
----------
train Loss: 0.0026 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8372

Epoch 90/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8372

Epoch 91/97
----------
train Loss: 0.0027 Acc: 0.9235
val Loss: 0.0115 Acc: 0.8372

Epoch 92/97
----------
train Loss: 0.0026 Acc: 0.9439
val Loss: 0.0115 Acc: 0.8372

Epoch 93/97
----------
train Loss: 0.0027 Acc: 0.9209
val Loss: 0.0115 Acc: 0.8372

Epoch 94/97
----------
train Loss: 0.0026 Acc: 0.9362
val Loss: 0.0115 Acc: 0.8372

Epoch 95/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8372

Epoch 96/97
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0027 Acc: 0.9337
val Loss: 0.0115 Acc: 0.8605

Epoch 97/97
----------
train Loss: 0.0027 Acc: 0.9311
val Loss: 0.0115 Acc: 0.8605

Training complete in 3m 25s
Best val Acc: 0.883721

---Fine tuning.---
Epoch 0/97
----------
LR is set to 0.01
train Loss: 0.0029 Acc: 0.9235
val Loss: 0.0126 Acc: 0.8605

Epoch 1/97
----------
train Loss: 0.0016 Acc: 0.9745
val Loss: 0.0125 Acc: 0.8605

Epoch 2/97
----------
train Loss: 0.0008 Acc: 0.9923
val Loss: 0.0099 Acc: 0.8605

Epoch 3/97
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8837

Epoch 4/97
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8837

Epoch 5/97
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0121 Acc: 0.8837

Epoch 6/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 7/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8837

Epoch 8/97
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8837

Epoch 9/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8837

Epoch 10/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8837

Epoch 11/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8837

Epoch 12/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8837

Epoch 13/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 14/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 15/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 16/97
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 17/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 18/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8837

Epoch 19/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 20/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 21/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 22/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 23/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 24/97
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 25/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 26/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 27/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 28/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 29/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 30/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 31/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 32/97
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 33/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 34/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 35/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 36/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 37/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 38/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 39/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 40/97
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 41/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 42/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 43/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 44/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 45/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8837

Epoch 46/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 47/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 48/97
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 49/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 50/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 51/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 52/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 53/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 54/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 55/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 56/97
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 57/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 58/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 59/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 60/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 61/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 62/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 63/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 64/97
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 65/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 66/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 67/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 68/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 69/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 70/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 71/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 72/97
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 73/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 74/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 75/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 76/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 77/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 78/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 79/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 80/97
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8605

Epoch 81/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 82/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 83/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 84/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 85/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 86/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 87/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 88/97
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 89/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 90/97
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 91/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8837

Epoch 92/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 93/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 94/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 95/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8605

Epoch 96/97
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Epoch 97/97
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8605

Training complete in 3m 48s
Best val Acc: 0.883721

---Testing---
Test accuracy: 0.988506
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 99 %
mean: 0.9842664095938313, std: 0.009756578948343544
--------------------

run info[val: 0.15, epoch: 94, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0161 Acc: 0.3568
val Loss: 0.0211 Acc: 0.5077

Epoch 1/93
----------
train Loss: 0.0125 Acc: 0.5703
val Loss: 0.0145 Acc: 0.6923

Epoch 2/93
----------
train Loss: 0.0094 Acc: 0.7081
val Loss: 0.0109 Acc: 0.7538

Epoch 3/93
----------
train Loss: 0.0071 Acc: 0.7784
val Loss: 0.0098 Acc: 0.7538

Epoch 4/93
----------
LR is set to 0.001
train Loss: 0.0061 Acc: 0.7757
val Loss: 0.0096 Acc: 0.7846

Epoch 5/93
----------
train Loss: 0.0058 Acc: 0.8189
val Loss: 0.0091 Acc: 0.7692

Epoch 6/93
----------
train Loss: 0.0060 Acc: 0.8108
val Loss: 0.0088 Acc: 0.8154

Epoch 7/93
----------
train Loss: 0.0057 Acc: 0.8081
val Loss: 0.0087 Acc: 0.8615

Epoch 8/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0054 Acc: 0.8459
val Loss: 0.0087 Acc: 0.8615

Epoch 9/93
----------
train Loss: 0.0055 Acc: 0.8486
val Loss: 0.0087 Acc: 0.8615

Epoch 10/93
----------
train Loss: 0.0054 Acc: 0.8432
val Loss: 0.0088 Acc: 0.8615

Epoch 11/93
----------
train Loss: 0.0052 Acc: 0.8486
val Loss: 0.0088 Acc: 0.8615

Epoch 12/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0055 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 13/93
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 14/93
----------
train Loss: 0.0057 Acc: 0.8297
val Loss: 0.0088 Acc: 0.8615

Epoch 15/93
----------
train Loss: 0.0054 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 16/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0056 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 17/93
----------
train Loss: 0.0052 Acc: 0.8514
val Loss: 0.0088 Acc: 0.8615

Epoch 18/93
----------
train Loss: 0.0057 Acc: 0.8216
val Loss: 0.0088 Acc: 0.8615

Epoch 19/93
----------
train Loss: 0.0054 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 20/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0056 Acc: 0.8216
val Loss: 0.0088 Acc: 0.8615

Epoch 21/93
----------
train Loss: 0.0056 Acc: 0.8297
val Loss: 0.0088 Acc: 0.8615

Epoch 22/93
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 23/93
----------
train Loss: 0.0056 Acc: 0.8297
val Loss: 0.0088 Acc: 0.8462

Epoch 24/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0057 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 25/93
----------
train Loss: 0.0055 Acc: 0.8432
val Loss: 0.0088 Acc: 0.8615

Epoch 26/93
----------
train Loss: 0.0057 Acc: 0.8189
val Loss: 0.0088 Acc: 0.8615

Epoch 27/93
----------
train Loss: 0.0056 Acc: 0.8459
val Loss: 0.0088 Acc: 0.8615

Epoch 28/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0054 Acc: 0.8459
val Loss: 0.0088 Acc: 0.8615

Epoch 29/93
----------
train Loss: 0.0057 Acc: 0.8514
val Loss: 0.0088 Acc: 0.8615

Epoch 30/93
----------
train Loss: 0.0058 Acc: 0.8189
val Loss: 0.0088 Acc: 0.8615

Epoch 31/93
----------
train Loss: 0.0054 Acc: 0.8297
val Loss: 0.0088 Acc: 0.8615

Epoch 32/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0055 Acc: 0.8432
val Loss: 0.0088 Acc: 0.8615

Epoch 33/93
----------
train Loss: 0.0056 Acc: 0.8216
val Loss: 0.0088 Acc: 0.8615

Epoch 34/93
----------
train Loss: 0.0058 Acc: 0.8054
val Loss: 0.0088 Acc: 0.8615

Epoch 35/93
----------
train Loss: 0.0058 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 36/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0057 Acc: 0.8243
val Loss: 0.0088 Acc: 0.8615

Epoch 37/93
----------
train Loss: 0.0056 Acc: 0.8216
val Loss: 0.0088 Acc: 0.8615

Epoch 38/93
----------
train Loss: 0.0056 Acc: 0.8459
val Loss: 0.0088 Acc: 0.8615

Epoch 39/93
----------
train Loss: 0.0056 Acc: 0.8243
val Loss: 0.0087 Acc: 0.8615

Epoch 40/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0056 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 41/93
----------
train Loss: 0.0057 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8615

Epoch 42/93
----------
train Loss: 0.0054 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 43/93
----------
train Loss: 0.0054 Acc: 0.8297
val Loss: 0.0088 Acc: 0.8615

Epoch 44/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0057 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 45/93
----------
train Loss: 0.0054 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 46/93
----------
train Loss: 0.0053 Acc: 0.8541
val Loss: 0.0088 Acc: 0.8615

Epoch 47/93
----------
train Loss: 0.0056 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8462

Epoch 48/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0058 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8462

Epoch 49/93
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8462

Epoch 50/93
----------
train Loss: 0.0056 Acc: 0.8514
val Loss: 0.0088 Acc: 0.8615

Epoch 51/93
----------
train Loss: 0.0055 Acc: 0.8216
val Loss: 0.0088 Acc: 0.8615

Epoch 52/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0056 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8462

Epoch 53/93
----------
train Loss: 0.0057 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8615

Epoch 54/93
----------
train Loss: 0.0057 Acc: 0.8541
val Loss: 0.0088 Acc: 0.8615

Epoch 55/93
----------
train Loss: 0.0055 Acc: 0.8189
val Loss: 0.0088 Acc: 0.8615

Epoch 56/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0058 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 57/93
----------
train Loss: 0.0053 Acc: 0.8459
val Loss: 0.0088 Acc: 0.8615

Epoch 58/93
----------
train Loss: 0.0055 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8615

Epoch 59/93
----------
train Loss: 0.0057 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8615

Epoch 60/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0056 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8615

Epoch 61/93
----------
train Loss: 0.0054 Acc: 0.8514
val Loss: 0.0088 Acc: 0.8615

Epoch 62/93
----------
train Loss: 0.0057 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8462

Epoch 63/93
----------
train Loss: 0.0055 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8462

Epoch 64/93
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0058 Acc: 0.8243
val Loss: 0.0088 Acc: 0.8615

Epoch 65/93
----------
train Loss: 0.0056 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 66/93
----------
train Loss: 0.0056 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 67/93
----------
train Loss: 0.0055 Acc: 0.8514
val Loss: 0.0087 Acc: 0.8615

Epoch 68/93
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0058 Acc: 0.8351
val Loss: 0.0087 Acc: 0.8615

Epoch 69/93
----------
train Loss: 0.0056 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 70/93
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8462

Epoch 71/93
----------
train Loss: 0.0055 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 72/93
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0055 Acc: 0.8432
val Loss: 0.0088 Acc: 0.8615

Epoch 73/93
----------
train Loss: 0.0054 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8462

Epoch 74/93
----------
train Loss: 0.0055 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8462

Epoch 75/93
----------
train Loss: 0.0057 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8462

Epoch 76/93
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0056 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8462

Epoch 77/93
----------
train Loss: 0.0055 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8462

Epoch 78/93
----------
train Loss: 0.0054 Acc: 0.8378
val Loss: 0.0087 Acc: 0.8615

Epoch 79/93
----------
train Loss: 0.0055 Acc: 0.8351
val Loss: 0.0087 Acc: 0.8615

Epoch 80/93
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0055 Acc: 0.8405
val Loss: 0.0088 Acc: 0.8615

Epoch 81/93
----------
train Loss: 0.0055 Acc: 0.8324
val Loss: 0.0088 Acc: 0.8615

Epoch 82/93
----------
train Loss: 0.0057 Acc: 0.8162
val Loss: 0.0088 Acc: 0.8615

Epoch 83/93
----------
train Loss: 0.0056 Acc: 0.8378
val Loss: 0.0088 Acc: 0.8615

Epoch 84/93
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0056 Acc: 0.8081
val Loss: 0.0088 Acc: 0.8615

Epoch 85/93
----------
train Loss: 0.0056 Acc: 0.8243
val Loss: 0.0088 Acc: 0.8615

Epoch 86/93
----------
train Loss: 0.0053 Acc: 0.8486
val Loss: 0.0088 Acc: 0.8615

Epoch 87/93
----------
train Loss: 0.0055 Acc: 0.8162
val Loss: 0.0088 Acc: 0.8615

Epoch 88/93
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0053 Acc: 0.8351
val Loss: 0.0088 Acc: 0.8615

Epoch 89/93
----------
train Loss: 0.0056 Acc: 0.8243
val Loss: 0.0088 Acc: 0.8615

Epoch 90/93
----------
train Loss: 0.0059 Acc: 0.8081
val Loss: 0.0088 Acc: 0.8462

Epoch 91/93
----------
train Loss: 0.0056 Acc: 0.8243
val Loss: 0.0088 Acc: 0.8615

Epoch 92/93
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0056 Acc: 0.8270
val Loss: 0.0088 Acc: 0.8615

Epoch 93/93
----------
train Loss: 0.0056 Acc: 0.8459
val Loss: 0.0088 Acc: 0.8615

Training complete in 3m 21s
Best val Acc: 0.861538

---Fine tuning.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8378
val Loss: 0.0078 Acc: 0.8769

Epoch 1/93
----------
train Loss: 0.0037 Acc: 0.9027
val Loss: 0.0063 Acc: 0.8923

Epoch 2/93
----------
train Loss: 0.0023 Acc: 0.9595
val Loss: 0.0066 Acc: 0.8769

Epoch 3/93
----------
train Loss: 0.0011 Acc: 0.9811
val Loss: 0.0051 Acc: 0.9231

Epoch 4/93
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 5/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 6/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 7/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 8/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 9/93
----------
train Loss: 0.0008 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 10/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 11/93
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8769

Epoch 12/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0007 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 13/93
----------
train Loss: 0.0007 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 14/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 15/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 16/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 17/93
----------
train Loss: 0.0007 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 18/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0053 Acc: 0.8769

Epoch 19/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8769

Epoch 20/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 21/93
----------
train Loss: 0.0006 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8923

Epoch 22/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 23/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 24/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9865
val Loss: 0.0052 Acc: 0.8923

Epoch 25/93
----------
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 26/93
----------
train Loss: 0.0005 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 27/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 28/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 29/93
----------
train Loss: 0.0007 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 30/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0053 Acc: 0.8923

Epoch 31/93
----------
train Loss: 0.0005 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 32/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 33/93
----------
train Loss: 0.0007 Acc: 0.9865
val Loss: 0.0052 Acc: 0.8923

Epoch 34/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8769

Epoch 35/93
----------
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8923

Epoch 36/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 37/93
----------
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 38/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 39/93
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 40/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 41/93
----------
train Loss: 0.0006 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8923

Epoch 42/93
----------
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0052 Acc: 0.8923

Epoch 43/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 44/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 45/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 46/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 47/93
----------
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8769

Epoch 48/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0007 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 49/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 50/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 51/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 52/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0053 Acc: 0.8923

Epoch 53/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 54/93
----------
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8769

Epoch 55/93
----------
train Loss: 0.0005 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 56/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0053 Acc: 0.8769

Epoch 57/93
----------
train Loss: 0.0007 Acc: 0.9865
val Loss: 0.0053 Acc: 0.8769

Epoch 58/93
----------
train Loss: 0.0005 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8769

Epoch 59/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0053 Acc: 0.8923

Epoch 60/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 61/93
----------
train Loss: 0.0008 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8923

Epoch 62/93
----------
train Loss: 0.0005 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 63/93
----------
train Loss: 0.0007 Acc: 0.9865
val Loss: 0.0053 Acc: 0.9077

Epoch 64/93
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 65/93
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9077

Epoch 66/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8923

Epoch 67/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 68/93
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 69/93
----------
train Loss: 0.0005 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 70/93
----------
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8923

Epoch 71/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.9077

Epoch 72/93
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 73/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 74/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0053 Acc: 0.8769

Epoch 75/93
----------
train Loss: 0.0006 Acc: 0.9892
val Loss: 0.0053 Acc: 0.8769

Epoch 76/93
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 77/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 78/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 79/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 80/93
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 81/93
----------
train Loss: 0.0006 Acc: 0.9919
val Loss: 0.0052 Acc: 0.8923

Epoch 82/93
----------
train Loss: 0.0007 Acc: 0.9946
val Loss: 0.0053 Acc: 0.8923

Epoch 83/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 84/93
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8769

Epoch 85/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8769

Epoch 86/93
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8923

Epoch 87/93
----------
train Loss: 0.0006 Acc: 0.9973
val Loss: 0.0052 Acc: 0.8923

Epoch 88/93
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0052 Acc: 0.8769

Epoch 89/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Epoch 90/93
----------
train Loss: 0.0007 Acc: 0.9892
val Loss: 0.0053 Acc: 0.8923

Epoch 91/93
----------
train Loss: 0.0006 Acc: 0.9838
val Loss: 0.0053 Acc: 0.8923

Epoch 92/93
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0007 Acc: 0.9946
val Loss: 0.0052 Acc: 0.9077

Epoch 93/93
----------
train Loss: 0.0006 Acc: 0.9946
val Loss: 0.0052 Acc: 0.8923

Training complete in 3m 41s
Best val Acc: 0.923077

---Testing---
Test accuracy: 0.986207
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of Torpediniformes : 99 %
mean: 0.981734764024211, std: 0.010259449007902727
--------------------

run info[val: 0.2, epoch: 80, randcrop: False, decay: 4]

---Training last layer.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0185 Acc: 0.3218
val Loss: 0.0178 Acc: 0.4368

Epoch 1/79
----------
train Loss: 0.0135 Acc: 0.5460
val Loss: 0.0127 Acc: 0.5862

Epoch 2/79
----------
train Loss: 0.0100 Acc: 0.6810
val Loss: 0.0092 Acc: 0.7126

Epoch 3/79
----------
train Loss: 0.0071 Acc: 0.7701
val Loss: 0.0074 Acc: 0.8046

Epoch 4/79
----------
LR is set to 0.001
train Loss: 0.0059 Acc: 0.8276
val Loss: 0.0071 Acc: 0.8046

Epoch 5/79
----------
train Loss: 0.0056 Acc: 0.8420
val Loss: 0.0070 Acc: 0.7931

Epoch 6/79
----------
train Loss: 0.0054 Acc: 0.8247
val Loss: 0.0069 Acc: 0.7701

Epoch 7/79
----------
train Loss: 0.0061 Acc: 0.8362
val Loss: 0.0069 Acc: 0.8046

Epoch 8/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0054 Acc: 0.8305
val Loss: 0.0069 Acc: 0.7931

Epoch 9/79
----------
train Loss: 0.0055 Acc: 0.8132
val Loss: 0.0069 Acc: 0.7816

Epoch 10/79
----------
train Loss: 0.0051 Acc: 0.8477
val Loss: 0.0069 Acc: 0.7816

Epoch 11/79
----------
train Loss: 0.0055 Acc: 0.8362
val Loss: 0.0069 Acc: 0.7701

Epoch 12/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0056 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7701

Epoch 13/79
----------
train Loss: 0.0052 Acc: 0.8448
val Loss: 0.0069 Acc: 0.7701

Epoch 14/79
----------
train Loss: 0.0057 Acc: 0.8333
val Loss: 0.0068 Acc: 0.7701

Epoch 15/79
----------
train Loss: 0.0055 Acc: 0.8621
val Loss: 0.0069 Acc: 0.7701

Epoch 16/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0053 Acc: 0.8420
val Loss: 0.0068 Acc: 0.7701

Epoch 17/79
----------
train Loss: 0.0054 Acc: 0.8362
val Loss: 0.0068 Acc: 0.7701

Epoch 18/79
----------
train Loss: 0.0054 Acc: 0.8477
val Loss: 0.0068 Acc: 0.7701

Epoch 19/79
----------
train Loss: 0.0055 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7701

Epoch 20/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0056 Acc: 0.8391
val Loss: 0.0069 Acc: 0.7816

Epoch 21/79
----------
train Loss: 0.0052 Acc: 0.8448
val Loss: 0.0069 Acc: 0.7701

Epoch 22/79
----------
train Loss: 0.0056 Acc: 0.8448
val Loss: 0.0069 Acc: 0.7931

Epoch 23/79
----------
train Loss: 0.0054 Acc: 0.8448
val Loss: 0.0068 Acc: 0.7816

Epoch 24/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0054 Acc: 0.8391
val Loss: 0.0069 Acc: 0.7816

Epoch 25/79
----------
train Loss: 0.0055 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 26/79
----------
train Loss: 0.0054 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 27/79
----------
train Loss: 0.0058 Acc: 0.8592
val Loss: 0.0068 Acc: 0.7701

Epoch 28/79
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0055 Acc: 0.8563
val Loss: 0.0068 Acc: 0.7701

Epoch 29/79
----------
train Loss: 0.0055 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 30/79
----------
train Loss: 0.0054 Acc: 0.8534
val Loss: 0.0068 Acc: 0.7816

Epoch 31/79
----------
train Loss: 0.0053 Acc: 0.8362
val Loss: 0.0068 Acc: 0.7701

Epoch 32/79
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0055 Acc: 0.8420
val Loss: 0.0068 Acc: 0.7701

Epoch 33/79
----------
train Loss: 0.0055 Acc: 0.8391
val Loss: 0.0069 Acc: 0.7701

Epoch 34/79
----------
train Loss: 0.0054 Acc: 0.8506
val Loss: 0.0069 Acc: 0.7701

Epoch 35/79
----------
train Loss: 0.0051 Acc: 0.8592
val Loss: 0.0069 Acc: 0.7701

Epoch 36/79
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0058 Acc: 0.8448
val Loss: 0.0068 Acc: 0.7701

Epoch 37/79
----------
train Loss: 0.0054 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7701

Epoch 38/79
----------
train Loss: 0.0054 Acc: 0.8563
val Loss: 0.0069 Acc: 0.7701

Epoch 39/79
----------
train Loss: 0.0055 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7701

Epoch 40/79
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0056 Acc: 0.8333
val Loss: 0.0069 Acc: 0.7701

Epoch 41/79
----------
train Loss: 0.0056 Acc: 0.8477
val Loss: 0.0069 Acc: 0.7701

Epoch 42/79
----------
train Loss: 0.0056 Acc: 0.8362
val Loss: 0.0069 Acc: 0.7931

Epoch 43/79
----------
train Loss: 0.0053 Acc: 0.8506
val Loss: 0.0069 Acc: 0.7816

Epoch 44/79
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0052 Acc: 0.8592
val Loss: 0.0069 Acc: 0.7816

Epoch 45/79
----------
train Loss: 0.0051 Acc: 0.8592
val Loss: 0.0069 Acc: 0.7816

Epoch 46/79
----------
train Loss: 0.0053 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 47/79
----------
train Loss: 0.0057 Acc: 0.8420
val Loss: 0.0069 Acc: 0.7816

Epoch 48/79
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0057 Acc: 0.8333
val Loss: 0.0069 Acc: 0.7816

Epoch 49/79
----------
train Loss: 0.0053 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 50/79
----------
train Loss: 0.0055 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7816

Epoch 51/79
----------
train Loss: 0.0053 Acc: 0.8563
val Loss: 0.0069 Acc: 0.7816

Epoch 52/79
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0056 Acc: 0.8506
val Loss: 0.0068 Acc: 0.7816

Epoch 53/79
----------
train Loss: 0.0058 Acc: 0.8247
val Loss: 0.0068 Acc: 0.7701

Epoch 54/79
----------
train Loss: 0.0057 Acc: 0.8333
val Loss: 0.0068 Acc: 0.7816

Epoch 55/79
----------
train Loss: 0.0058 Acc: 0.8420
val Loss: 0.0068 Acc: 0.7701

Epoch 56/79
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0051 Acc: 0.8563
val Loss: 0.0068 Acc: 0.7701

Epoch 57/79
----------
train Loss: 0.0052 Acc: 0.8506
val Loss: 0.0069 Acc: 0.7931

Epoch 58/79
----------
train Loss: 0.0056 Acc: 0.8477
val Loss: 0.0068 Acc: 0.7701

Epoch 59/79
----------
train Loss: 0.0059 Acc: 0.8448
val Loss: 0.0069 Acc: 0.7816

Epoch 60/79
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0054 Acc: 0.8506
val Loss: 0.0069 Acc: 0.7816

Epoch 61/79
----------
train Loss: 0.0056 Acc: 0.8420
val Loss: 0.0068 Acc: 0.7816

Epoch 62/79
----------
train Loss: 0.0057 Acc: 0.8276
val Loss: 0.0068 Acc: 0.7701

Epoch 63/79
----------
train Loss: 0.0054 Acc: 0.8477
val Loss: 0.0068 Acc: 0.7701

Epoch 64/79
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0058 Acc: 0.8190
val Loss: 0.0068 Acc: 0.7701

Epoch 65/79
----------
train Loss: 0.0059 Acc: 0.8477
val Loss: 0.0068 Acc: 0.7701

Epoch 66/79
----------
train Loss: 0.0055 Acc: 0.8362
val Loss: 0.0068 Acc: 0.7701

Epoch 67/79
----------
train Loss: 0.0053 Acc: 0.8534
val Loss: 0.0069 Acc: 0.7701

Epoch 68/79
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0055 Acc: 0.8305
val Loss: 0.0069 Acc: 0.7701

Epoch 69/79
----------
train Loss: 0.0055 Acc: 0.8333
val Loss: 0.0068 Acc: 0.7701

Epoch 70/79
----------
train Loss: 0.0054 Acc: 0.8420
val Loss: 0.0069 Acc: 0.7816

Epoch 71/79
----------
train Loss: 0.0055 Acc: 0.8247
val Loss: 0.0069 Acc: 0.7701

Epoch 72/79
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0054 Acc: 0.8563
val Loss: 0.0069 Acc: 0.7701

Epoch 73/79
----------
train Loss: 0.0055 Acc: 0.8420
val Loss: 0.0069 Acc: 0.7701

Epoch 74/79
----------
train Loss: 0.0055 Acc: 0.8477
val Loss: 0.0068 Acc: 0.7816

Epoch 75/79
----------
train Loss: 0.0054 Acc: 0.8477
val Loss: 0.0069 Acc: 0.7701

Epoch 76/79
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0058 Acc: 0.8391
val Loss: 0.0069 Acc: 0.7816

Epoch 77/79
----------
train Loss: 0.0057 Acc: 0.8333
val Loss: 0.0068 Acc: 0.7816

Epoch 78/79
----------
train Loss: 0.0052 Acc: 0.8563
val Loss: 0.0068 Acc: 0.7701

Epoch 79/79
----------
train Loss: 0.0053 Acc: 0.8333
val Loss: 0.0068 Acc: 0.7931

Training complete in 2m 56s
Best val Acc: 0.804598

---Fine tuning.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0058 Acc: 0.8506
val Loss: 0.0062 Acc: 0.8506

Epoch 1/79
----------
train Loss: 0.0035 Acc: 0.9282
val Loss: 0.0052 Acc: 0.8621

Epoch 2/79
----------
train Loss: 0.0017 Acc: 0.9828
val Loss: 0.0044 Acc: 0.8851

Epoch 3/79
----------
train Loss: 0.0009 Acc: 0.9943
val Loss: 0.0044 Acc: 0.8966

Epoch 4/79
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 5/79
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9080

Epoch 6/79
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9080

Epoch 7/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 8/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0041 Acc: 0.8966

Epoch 9/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 10/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 11/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 12/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 13/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 14/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 15/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 16/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 17/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 18/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 19/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 20/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 21/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 22/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0041 Acc: 0.8966

Epoch 23/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 24/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 25/79
----------
train Loss: 0.0003 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 26/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 27/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 28/79
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 29/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 30/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 31/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 32/79
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 33/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 34/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 35/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 36/79
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 37/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 38/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 39/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 40/79
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 41/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 42/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 43/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 44/79
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 45/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 46/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 47/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 48/79
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 49/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 50/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 51/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 52/79
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 53/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 54/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 55/79
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 56/79
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 57/79
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 58/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 59/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 60/79
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 61/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 62/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 63/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 64/79
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 65/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 66/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 67/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 68/79
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 69/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 70/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 71/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 72/79
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8966

Epoch 73/79
----------
train Loss: 0.0004 Acc: 0.9971
val Loss: 0.0042 Acc: 0.8966

Epoch 74/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 75/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 76/79
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 77/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 78/79
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Epoch 79/79
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8966

Training complete in 3m 15s
Best val Acc: 0.908046

---Testing---
Test accuracy: 0.981609
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 98 %
mean: 0.978096037578467, std: 0.008429719905853828
--------------------

run info[val: 0.25, epoch: 50, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0176 Acc: 0.3976
val Loss: 0.0273 Acc: 0.4444

Epoch 1/49
----------
train Loss: 0.0144 Acc: 0.5535
val Loss: 0.0151 Acc: 0.6389

Epoch 2/49
----------
train Loss: 0.0120 Acc: 0.6208
val Loss: 0.0153 Acc: 0.6389

Epoch 3/49
----------
train Loss: 0.0093 Acc: 0.7156
val Loss: 0.0146 Acc: 0.6944

Epoch 4/49
----------
LR is set to 0.001
train Loss: 0.0076 Acc: 0.7615
val Loss: 0.0164 Acc: 0.7130

Epoch 5/49
----------
train Loss: 0.0066 Acc: 0.7859
val Loss: 0.0083 Acc: 0.7593

Epoch 6/49
----------
train Loss: 0.0064 Acc: 0.8043
val Loss: 0.0106 Acc: 0.8148

Epoch 7/49
----------
train Loss: 0.0066 Acc: 0.8165
val Loss: 0.0130 Acc: 0.8056

Epoch 8/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0068 Acc: 0.8196
val Loss: 0.0118 Acc: 0.8056

Epoch 9/49
----------
train Loss: 0.0066 Acc: 0.8257
val Loss: 0.0132 Acc: 0.8056

Epoch 10/49
----------
train Loss: 0.0067 Acc: 0.8257
val Loss: 0.0112 Acc: 0.8056

Epoch 11/49
----------
train Loss: 0.0068 Acc: 0.8135
val Loss: 0.0089 Acc: 0.8056

Epoch 12/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0065 Acc: 0.8532
val Loss: 0.0109 Acc: 0.7870

Epoch 13/49
----------
train Loss: 0.0064 Acc: 0.8471
val Loss: 0.0109 Acc: 0.7778

Epoch 14/49
----------
train Loss: 0.0063 Acc: 0.8165
val Loss: 0.0068 Acc: 0.8056

Epoch 15/49
----------
train Loss: 0.0067 Acc: 0.8135
val Loss: 0.0116 Acc: 0.7963

Epoch 16/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0067 Acc: 0.8226
val Loss: 0.0084 Acc: 0.7778

Epoch 17/49
----------
train Loss: 0.0067 Acc: 0.8349
val Loss: 0.0096 Acc: 0.7778

Epoch 18/49
----------
train Loss: 0.0067 Acc: 0.8135
val Loss: 0.0097 Acc: 0.7870

Epoch 19/49
----------
train Loss: 0.0071 Acc: 0.8287
val Loss: 0.0109 Acc: 0.7778

Epoch 20/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0065 Acc: 0.8104
val Loss: 0.0106 Acc: 0.7778

Epoch 21/49
----------
train Loss: 0.0067 Acc: 0.8349
val Loss: 0.0107 Acc: 0.8056

Epoch 22/49
----------
train Loss: 0.0065 Acc: 0.8257
val Loss: 0.0091 Acc: 0.8056

Epoch 23/49
----------
train Loss: 0.0070 Acc: 0.8257
val Loss: 0.0102 Acc: 0.8056

Epoch 24/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0069 Acc: 0.8012
val Loss: 0.0102 Acc: 0.7870

Epoch 25/49
----------
train Loss: 0.0065 Acc: 0.8379
val Loss: 0.0114 Acc: 0.8056

Epoch 26/49
----------
train Loss: 0.0066 Acc: 0.8073
val Loss: 0.0115 Acc: 0.8148

Epoch 27/49
----------
train Loss: 0.0062 Acc: 0.8287
val Loss: 0.0108 Acc: 0.8056

Epoch 28/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0064 Acc: 0.8440
val Loss: 0.0115 Acc: 0.8056

Epoch 29/49
----------
train Loss: 0.0058 Acc: 0.8226
val Loss: 0.0110 Acc: 0.7870

Epoch 30/49
----------
train Loss: 0.0060 Acc: 0.8287
val Loss: 0.0089 Acc: 0.8056

Epoch 31/49
----------
train Loss: 0.0074 Acc: 0.8287
val Loss: 0.0084 Acc: 0.7778

Epoch 32/49
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0061 Acc: 0.8410
val Loss: 0.0080 Acc: 0.7963

Epoch 33/49
----------
train Loss: 0.0064 Acc: 0.8104
val Loss: 0.0133 Acc: 0.7963

Epoch 34/49
----------
train Loss: 0.0075 Acc: 0.8379
val Loss: 0.0130 Acc: 0.7963

Epoch 35/49
----------
train Loss: 0.0063 Acc: 0.8226
val Loss: 0.0099 Acc: 0.7963

Epoch 36/49
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0065 Acc: 0.8196
val Loss: 0.0116 Acc: 0.7963

Epoch 37/49
----------
train Loss: 0.0065 Acc: 0.8104
val Loss: 0.0098 Acc: 0.7870

Epoch 38/49
----------
train Loss: 0.0065 Acc: 0.7982
val Loss: 0.0087 Acc: 0.7870

Epoch 39/49
----------
train Loss: 0.0064 Acc: 0.8257
val Loss: 0.0101 Acc: 0.7870

Epoch 40/49
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0066 Acc: 0.8135
val Loss: 0.0122 Acc: 0.7870

Epoch 41/49
----------
train Loss: 0.0064 Acc: 0.8104
val Loss: 0.0116 Acc: 0.7778

Epoch 42/49
----------
train Loss: 0.0068 Acc: 0.8287
val Loss: 0.0086 Acc: 0.7778

Epoch 43/49
----------
train Loss: 0.0062 Acc: 0.8196
val Loss: 0.0118 Acc: 0.7778

Epoch 44/49
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0063 Acc: 0.8104
val Loss: 0.0107 Acc: 0.7778

Epoch 45/49
----------
train Loss: 0.0067 Acc: 0.8226
val Loss: 0.0076 Acc: 0.7778

Epoch 46/49
----------
train Loss: 0.0061 Acc: 0.8043
val Loss: 0.0147 Acc: 0.7778

Epoch 47/49
----------
train Loss: 0.0068 Acc: 0.8226
val Loss: 0.0106 Acc: 0.7778

Epoch 48/49
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0061 Acc: 0.8502
val Loss: 0.0119 Acc: 0.7778

Epoch 49/49
----------
train Loss: 0.0059 Acc: 0.8532
val Loss: 0.0096 Acc: 0.7778

Training complete in 1m 53s
Best val Acc: 0.814815

---Fine tuning.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0063 Acc: 0.8287
val Loss: 0.0070 Acc: 0.8519

Epoch 1/49
----------
train Loss: 0.0044 Acc: 0.8960
val Loss: 0.0064 Acc: 0.8426

Epoch 2/49
----------
train Loss: 0.0032 Acc: 0.9450
val Loss: 0.0119 Acc: 0.8519

Epoch 3/49
----------
train Loss: 0.0015 Acc: 0.9847
val Loss: 0.0047 Acc: 0.8704

Epoch 4/49
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9908
val Loss: 0.0066 Acc: 0.8889

Epoch 5/49
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0040 Acc: 0.8981

Epoch 6/49
----------
train Loss: 0.0009 Acc: 0.9939
val Loss: 0.0038 Acc: 0.8981

Epoch 7/49
----------
train Loss: 0.0006 Acc: 0.9939
val Loss: 0.0060 Acc: 0.8889

Epoch 8/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8889

Epoch 9/49
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8889

Epoch 10/49
----------
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0055 Acc: 0.8981

Epoch 11/49
----------
train Loss: 0.0009 Acc: 0.9939
val Loss: 0.0087 Acc: 0.8981

Epoch 12/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9969
val Loss: 0.0037 Acc: 0.8981

Epoch 13/49
----------
train Loss: 0.0008 Acc: 0.9969
val Loss: 0.0033 Acc: 0.8889

Epoch 14/49
----------
train Loss: 0.0005 Acc: 0.9969
val Loss: 0.0081 Acc: 0.8796

Epoch 15/49
----------
train Loss: 0.0009 Acc: 0.9969
val Loss: 0.0052 Acc: 0.9074

Epoch 16/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8981

Epoch 17/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8981

Epoch 18/49
----------
train Loss: 0.0008 Acc: 0.9878
val Loss: 0.0035 Acc: 0.8796

Epoch 19/49
----------
train Loss: 0.0006 Acc: 0.9939
val Loss: 0.0050 Acc: 0.8889

Epoch 20/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0033 Acc: 0.8889

Epoch 21/49
----------
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0042 Acc: 0.8889

Epoch 22/49
----------
train Loss: 0.0008 Acc: 0.9939
val Loss: 0.0033 Acc: 0.8796

Epoch 23/49
----------
train Loss: 0.0007 Acc: 0.9908
val Loss: 0.0060 Acc: 0.8889

Epoch 24/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8889

Epoch 25/49
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8889

Epoch 26/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8889

Epoch 27/49
----------
train Loss: 0.0006 Acc: 0.9939
val Loss: 0.0048 Acc: 0.8889

Epoch 28/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9908
val Loss: 0.0123 Acc: 0.8889

Epoch 29/49
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0062 Acc: 0.8889

Epoch 30/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8889

Epoch 31/49
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8889

Epoch 32/49
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0032 Acc: 0.8889

Epoch 33/49
----------
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0041 Acc: 0.8889

Epoch 34/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8889

Epoch 35/49
----------
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0073 Acc: 0.8889

Epoch 36/49
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0008 Acc: 0.9908
val Loss: 0.0060 Acc: 0.8889

Epoch 37/49
----------
train Loss: 0.0008 Acc: 0.9878
val Loss: 0.0038 Acc: 0.8889

Epoch 38/49
----------
train Loss: 0.0006 Acc: 0.9969
val Loss: 0.0083 Acc: 0.8889

Epoch 39/49
----------
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8889

Epoch 40/49
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0125 Acc: 0.8889

Epoch 41/49
----------
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0070 Acc: 0.8889

Epoch 42/49
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8889

Epoch 43/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8889

Epoch 44/49
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0007 Acc: 0.9939
val Loss: 0.0055 Acc: 0.8981

Epoch 45/49
----------
train Loss: 0.0007 Acc: 0.9908
val Loss: 0.0035 Acc: 0.8889

Epoch 46/49
----------
train Loss: 0.0007 Acc: 0.9969
val Loss: 0.0034 Acc: 0.8889

Epoch 47/49
----------
train Loss: 0.0009 Acc: 0.9939
val Loss: 0.0047 Acc: 0.8889

Epoch 48/49
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8889

Epoch 49/49
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8981

Training complete in 2m 4s
Best val Acc: 0.907407

---Testing---
Test accuracy: 0.977011
--------------------
Accuracy of Dasyatiformes : 96 %
Accuracy of Myliobatiformes : 96 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 99 %
mean: 0.9714006493187881, std: 0.012533033578288224
--------------------

run info[val: 0.3, epoch: 99, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0184 Acc: 0.3279
val Loss: 0.0310 Acc: 0.4231

Epoch 1/98
----------
train Loss: 0.0182 Acc: 0.4361
val Loss: 0.0199 Acc: 0.5846

Epoch 2/98
----------
train Loss: 0.0147 Acc: 0.6000
val Loss: 0.0154 Acc: 0.6154

Epoch 3/98
----------
train Loss: 0.0120 Acc: 0.6295
val Loss: 0.0122 Acc: 0.7308

Epoch 4/98
----------
LR is set to 0.001
train Loss: 0.0103 Acc: 0.7574
val Loss: 0.0092 Acc: 0.7538

Epoch 5/98
----------
train Loss: 0.0076 Acc: 0.7934
val Loss: 0.0090 Acc: 0.7538

Epoch 6/98
----------
train Loss: 0.0065 Acc: 0.8098
val Loss: 0.0110 Acc: 0.7385

Epoch 7/98
----------
train Loss: 0.0072 Acc: 0.7902
val Loss: 0.0103 Acc: 0.7308

Epoch 8/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0072 Acc: 0.7967
val Loss: 0.0120 Acc: 0.7231

Epoch 9/98
----------
train Loss: 0.0083 Acc: 0.7803
val Loss: 0.0105 Acc: 0.7308

Epoch 10/98
----------
train Loss: 0.0067 Acc: 0.8033
val Loss: 0.0104 Acc: 0.7231

Epoch 11/98
----------
train Loss: 0.0066 Acc: 0.7967
val Loss: 0.0113 Acc: 0.7077

Epoch 12/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0097 Acc: 0.7902
val Loss: 0.0104 Acc: 0.7154

Epoch 13/98
----------
train Loss: 0.0097 Acc: 0.7967
val Loss: 0.0103 Acc: 0.7385

Epoch 14/98
----------
train Loss: 0.0088 Acc: 0.7902
val Loss: 0.0120 Acc: 0.7385

Epoch 15/98
----------
train Loss: 0.0075 Acc: 0.8033
val Loss: 0.0099 Acc: 0.7462

Epoch 16/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0079 Acc: 0.7836
val Loss: 0.0104 Acc: 0.7462

Epoch 17/98
----------
train Loss: 0.0072 Acc: 0.7967
val Loss: 0.0108 Acc: 0.7308

Epoch 18/98
----------
train Loss: 0.0115 Acc: 0.7902
val Loss: 0.0109 Acc: 0.7308

Epoch 19/98
----------
train Loss: 0.0077 Acc: 0.8295
val Loss: 0.0117 Acc: 0.7308

Epoch 20/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0090 Acc: 0.8000
val Loss: 0.0106 Acc: 0.7308

Epoch 21/98
----------
train Loss: 0.0107 Acc: 0.8066
val Loss: 0.0099 Acc: 0.7231

Epoch 22/98
----------
train Loss: 0.0086 Acc: 0.7770
val Loss: 0.0116 Acc: 0.7154

Epoch 23/98
----------
train Loss: 0.0062 Acc: 0.7770
val Loss: 0.0103 Acc: 0.7154

Epoch 24/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0094 Acc: 0.7934
val Loss: 0.0115 Acc: 0.7154

Epoch 25/98
----------
train Loss: 0.0093 Acc: 0.7738
val Loss: 0.0108 Acc: 0.7077

Epoch 26/98
----------
train Loss: 0.0067 Acc: 0.8033
val Loss: 0.0122 Acc: 0.7154

Epoch 27/98
----------
train Loss: 0.0070 Acc: 0.8000
val Loss: 0.0112 Acc: 0.7308

Epoch 28/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0086 Acc: 0.7803
val Loss: 0.0114 Acc: 0.7308

Epoch 29/98
----------
train Loss: 0.0067 Acc: 0.8066
val Loss: 0.0104 Acc: 0.7231

Epoch 30/98
----------
train Loss: 0.0080 Acc: 0.7967
val Loss: 0.0101 Acc: 0.7308

Epoch 31/98
----------
train Loss: 0.0089 Acc: 0.8230
val Loss: 0.0111 Acc: 0.7231

Epoch 32/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0101 Acc: 0.7934
val Loss: 0.0120 Acc: 0.7308

Epoch 33/98
----------
train Loss: 0.0081 Acc: 0.8066
val Loss: 0.0106 Acc: 0.7154

Epoch 34/98
----------
train Loss: 0.0080 Acc: 0.7967
val Loss: 0.0105 Acc: 0.7154

Epoch 35/98
----------
train Loss: 0.0080 Acc: 0.8131
val Loss: 0.0106 Acc: 0.7154

Epoch 36/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0078 Acc: 0.8033
val Loss: 0.0111 Acc: 0.7385

Epoch 37/98
----------
train Loss: 0.0080 Acc: 0.7967
val Loss: 0.0088 Acc: 0.7385

Epoch 38/98
----------
train Loss: 0.0076 Acc: 0.7967
val Loss: 0.0096 Acc: 0.7385

Epoch 39/98
----------
train Loss: 0.0086 Acc: 0.7869
val Loss: 0.0106 Acc: 0.7385

Epoch 40/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0067 Acc: 0.8131
val Loss: 0.0110 Acc: 0.7308

Epoch 41/98
----------
train Loss: 0.0080 Acc: 0.8197
val Loss: 0.0107 Acc: 0.7308

Epoch 42/98
----------
train Loss: 0.0090 Acc: 0.7869
val Loss: 0.0099 Acc: 0.7231

Epoch 43/98
----------
train Loss: 0.0116 Acc: 0.8131
val Loss: 0.0110 Acc: 0.7385

Epoch 44/98
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0071 Acc: 0.7934
val Loss: 0.0109 Acc: 0.7231

Epoch 45/98
----------
train Loss: 0.0091 Acc: 0.7836
val Loss: 0.0100 Acc: 0.7308

Epoch 46/98
----------
train Loss: 0.0088 Acc: 0.7869
val Loss: 0.0109 Acc: 0.7231

Epoch 47/98
----------
train Loss: 0.0092 Acc: 0.7869
val Loss: 0.0100 Acc: 0.7308

Epoch 48/98
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0076 Acc: 0.7869
val Loss: 0.0119 Acc: 0.7154

Epoch 49/98
----------
train Loss: 0.0075 Acc: 0.7770
val Loss: 0.0118 Acc: 0.7154

Epoch 50/98
----------
train Loss: 0.0077 Acc: 0.8033
val Loss: 0.0123 Acc: 0.7154

Epoch 51/98
----------
train Loss: 0.0097 Acc: 0.7869
val Loss: 0.0098 Acc: 0.7308

Epoch 52/98
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0116 Acc: 0.8000
val Loss: 0.0099 Acc: 0.7385

Epoch 53/98
----------
train Loss: 0.0074 Acc: 0.7836
val Loss: 0.0117 Acc: 0.7231

Epoch 54/98
----------
train Loss: 0.0076 Acc: 0.8000
val Loss: 0.0124 Acc: 0.7231

Epoch 55/98
----------
train Loss: 0.0079 Acc: 0.8098
val Loss: 0.0125 Acc: 0.7231

Epoch 56/98
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0070 Acc: 0.7902
val Loss: 0.0103 Acc: 0.7077

Epoch 57/98
----------
train Loss: 0.0072 Acc: 0.7836
val Loss: 0.0113 Acc: 0.7154

Epoch 58/98
----------
train Loss: 0.0071 Acc: 0.8164
val Loss: 0.0106 Acc: 0.7077

Epoch 59/98
----------
train Loss: 0.0071 Acc: 0.7934
val Loss: 0.0119 Acc: 0.7077

Epoch 60/98
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0075 Acc: 0.7869
val Loss: 0.0102 Acc: 0.7077

Epoch 61/98
----------
train Loss: 0.0094 Acc: 0.8033
val Loss: 0.0114 Acc: 0.7154

Epoch 62/98
----------
train Loss: 0.0102 Acc: 0.7902
val Loss: 0.0110 Acc: 0.7154

Epoch 63/98
----------
train Loss: 0.0088 Acc: 0.8131
val Loss: 0.0110 Acc: 0.7154

Epoch 64/98
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0071 Acc: 0.8098
val Loss: 0.0104 Acc: 0.7231

Epoch 65/98
----------
train Loss: 0.0066 Acc: 0.8066
val Loss: 0.0112 Acc: 0.7077

Epoch 66/98
----------
train Loss: 0.0097 Acc: 0.7639
val Loss: 0.0112 Acc: 0.7308

Epoch 67/98
----------
train Loss: 0.0084 Acc: 0.7967
val Loss: 0.0098 Acc: 0.7308

Epoch 68/98
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0074 Acc: 0.8033
val Loss: 0.0101 Acc: 0.7308

Epoch 69/98
----------
train Loss: 0.0090 Acc: 0.8000
val Loss: 0.0113 Acc: 0.7231

Epoch 70/98
----------
train Loss: 0.0071 Acc: 0.7803
val Loss: 0.0104 Acc: 0.7385

Epoch 71/98
----------
train Loss: 0.0072 Acc: 0.7705
val Loss: 0.0101 Acc: 0.7308

Epoch 72/98
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0066 Acc: 0.7836
val Loss: 0.0116 Acc: 0.7308

Epoch 73/98
----------
train Loss: 0.0081 Acc: 0.7705
val Loss: 0.0104 Acc: 0.7308

Epoch 74/98
----------
train Loss: 0.0067 Acc: 0.8000
val Loss: 0.0100 Acc: 0.7231

Epoch 75/98
----------
train Loss: 0.0069 Acc: 0.8164
val Loss: 0.0108 Acc: 0.7231

Epoch 76/98
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0074 Acc: 0.7967
val Loss: 0.0096 Acc: 0.7308

Epoch 77/98
----------
train Loss: 0.0112 Acc: 0.7770
val Loss: 0.0101 Acc: 0.7385

Epoch 78/98
----------
train Loss: 0.0077 Acc: 0.7869
val Loss: 0.0112 Acc: 0.7308

Epoch 79/98
----------
train Loss: 0.0091 Acc: 0.7902
val Loss: 0.0117 Acc: 0.7308

Epoch 80/98
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0097 Acc: 0.7967
val Loss: 0.0108 Acc: 0.7308

Epoch 81/98
----------
train Loss: 0.0092 Acc: 0.7508
val Loss: 0.0115 Acc: 0.7308

Epoch 82/98
----------
train Loss: 0.0077 Acc: 0.7967
val Loss: 0.0095 Acc: 0.7308

Epoch 83/98
----------
train Loss: 0.0066 Acc: 0.7836
val Loss: 0.0105 Acc: 0.7385

Epoch 84/98
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0070 Acc: 0.7967
val Loss: 0.0107 Acc: 0.7385

Epoch 85/98
----------
train Loss: 0.0075 Acc: 0.7967
val Loss: 0.0089 Acc: 0.7385

Epoch 86/98
----------
train Loss: 0.0103 Acc: 0.8328
val Loss: 0.0106 Acc: 0.7154

Epoch 87/98
----------
train Loss: 0.0072 Acc: 0.7869
val Loss: 0.0107 Acc: 0.7077

Epoch 88/98
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0081 Acc: 0.7705
val Loss: 0.0105 Acc: 0.7154

Epoch 89/98
----------
train Loss: 0.0076 Acc: 0.7934
val Loss: 0.0105 Acc: 0.7154

Epoch 90/98
----------
train Loss: 0.0070 Acc: 0.8000
val Loss: 0.0108 Acc: 0.7385

Epoch 91/98
----------
train Loss: 0.0073 Acc: 0.8033
val Loss: 0.0094 Acc: 0.7385

Epoch 92/98
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0070 Acc: 0.7967
val Loss: 0.0099 Acc: 0.7385

Epoch 93/98
----------
train Loss: 0.0104 Acc: 0.7770
val Loss: 0.0106 Acc: 0.7231

Epoch 94/98
----------
train Loss: 0.0075 Acc: 0.7770
val Loss: 0.0111 Acc: 0.7308

Epoch 95/98
----------
train Loss: 0.0070 Acc: 0.7934
val Loss: 0.0104 Acc: 0.7385

Epoch 96/98
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0099 Acc: 0.7770
val Loss: 0.0114 Acc: 0.7462

Epoch 97/98
----------
train Loss: 0.0104 Acc: 0.8033
val Loss: 0.0112 Acc: 0.7462

Epoch 98/98
----------
train Loss: 0.0072 Acc: 0.8000
val Loss: 0.0109 Acc: 0.7385

Training complete in 3m 42s
Best val Acc: 0.753846

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0074 Acc: 0.7607
val Loss: 0.0114 Acc: 0.7846

Epoch 1/98
----------
train Loss: 0.0043 Acc: 0.8492
val Loss: 0.0085 Acc: 0.8077

Epoch 2/98
----------
train Loss: 0.0047 Acc: 0.9279
val Loss: 0.0090 Acc: 0.8231

Epoch 3/98
----------
train Loss: 0.0062 Acc: 0.8721
val Loss: 0.0108 Acc: 0.8077

Epoch 4/98
----------
LR is set to 0.001
train Loss: 0.0046 Acc: 0.9410
val Loss: 0.0080 Acc: 0.8308

Epoch 5/98
----------
train Loss: 0.0025 Acc: 0.9607
val Loss: 0.0096 Acc: 0.8308

Epoch 6/98
----------
train Loss: 0.0033 Acc: 0.9803
val Loss: 0.0084 Acc: 0.8308

Epoch 7/98
----------
train Loss: 0.0019 Acc: 0.9869
val Loss: 0.0055 Acc: 0.8462

Epoch 8/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0019 Acc: 0.9738
val Loss: 0.0069 Acc: 0.8462

Epoch 9/98
----------
train Loss: 0.0021 Acc: 0.9836
val Loss: 0.0071 Acc: 0.8385

Epoch 10/98
----------
train Loss: 0.0013 Acc: 0.9869
val Loss: 0.0062 Acc: 0.8308

Epoch 11/98
----------
train Loss: 0.0012 Acc: 0.9836
val Loss: 0.0051 Acc: 0.8462

Epoch 12/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0080 Acc: 0.9705
val Loss: 0.0067 Acc: 0.8538

Epoch 13/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0053 Acc: 0.8538

Epoch 14/98
----------
train Loss: 0.0014 Acc: 0.9836
val Loss: 0.0066 Acc: 0.8462

Epoch 15/98
----------
train Loss: 0.0014 Acc: 0.9869
val Loss: 0.0070 Acc: 0.8308

Epoch 16/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0020 Acc: 0.9869
val Loss: 0.0071 Acc: 0.8462

Epoch 17/98
----------
train Loss: 0.0016 Acc: 0.9902
val Loss: 0.0079 Acc: 0.8462

Epoch 18/98
----------
train Loss: 0.0050 Acc: 0.9770
val Loss: 0.0055 Acc: 0.8538

Epoch 19/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0061 Acc: 0.8538

Epoch 20/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 0.9902
val Loss: 0.0059 Acc: 0.8462

Epoch 21/98
----------
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0077 Acc: 0.8385

Epoch 22/98
----------
train Loss: 0.0023 Acc: 0.9770
val Loss: 0.0070 Acc: 0.8538

Epoch 23/98
----------
train Loss: 0.0016 Acc: 0.9902
val Loss: 0.0070 Acc: 0.8462

Epoch 24/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0026 Acc: 0.9836
val Loss: 0.0061 Acc: 0.8615

Epoch 25/98
----------
train Loss: 0.0009 Acc: 0.9836
val Loss: 0.0066 Acc: 0.8538

Epoch 26/98
----------
train Loss: 0.0047 Acc: 0.9803
val Loss: 0.0078 Acc: 0.8385

Epoch 27/98
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0072 Acc: 0.8462

Epoch 28/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0065 Acc: 0.9770
val Loss: 0.0089 Acc: 0.8308

Epoch 29/98
----------
train Loss: 0.0011 Acc: 0.9902
val Loss: 0.0071 Acc: 0.8385

Epoch 30/98
----------
train Loss: 0.0013 Acc: 0.9738
val Loss: 0.0057 Acc: 0.8385

Epoch 31/98
----------
train Loss: 0.0018 Acc: 0.9738
val Loss: 0.0069 Acc: 0.8615

Epoch 32/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0039 Acc: 0.9705
val Loss: 0.0057 Acc: 0.8615

Epoch 33/98
----------
train Loss: 0.0016 Acc: 0.9869
val Loss: 0.0059 Acc: 0.8462

Epoch 34/98
----------
train Loss: 0.0026 Acc: 0.9770
val Loss: 0.0081 Acc: 0.8538

Epoch 35/98
----------
train Loss: 0.0024 Acc: 0.9770
val Loss: 0.0056 Acc: 0.8462

Epoch 36/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0049 Acc: 0.9770
val Loss: 0.0069 Acc: 0.8538

Epoch 37/98
----------
train Loss: 0.0036 Acc: 0.9738
val Loss: 0.0061 Acc: 0.8385

Epoch 38/98
----------
train Loss: 0.0035 Acc: 0.9869
val Loss: 0.0083 Acc: 0.8462

Epoch 39/98
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0066 Acc: 0.8231

Epoch 40/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0015 Acc: 0.9836
val Loss: 0.0090 Acc: 0.8462

Epoch 41/98
----------
train Loss: 0.0026 Acc: 0.9738
val Loss: 0.0076 Acc: 0.8308

Epoch 42/98
----------
train Loss: 0.0010 Acc: 0.9934
val Loss: 0.0064 Acc: 0.8462

Epoch 43/98
----------
train Loss: 0.0041 Acc: 0.9770
val Loss: 0.0060 Acc: 0.8538

Epoch 44/98
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0008 Acc: 0.9902
val Loss: 0.0068 Acc: 0.8462

Epoch 45/98
----------
train Loss: 0.0056 Acc: 0.9738
val Loss: 0.0066 Acc: 0.8154

Epoch 46/98
----------
train Loss: 0.0011 Acc: 0.9869
val Loss: 0.0060 Acc: 0.8462

Epoch 47/98
----------
train Loss: 0.0032 Acc: 0.9705
val Loss: 0.0072 Acc: 0.8154

Epoch 48/98
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0080 Acc: 0.8308

Epoch 49/98
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0063 Acc: 0.8462

Epoch 50/98
----------
train Loss: 0.0012 Acc: 0.9770
val Loss: 0.0066 Acc: 0.8538

Epoch 51/98
----------
train Loss: 0.0022 Acc: 0.9770
val Loss: 0.0090 Acc: 0.8462

Epoch 52/98
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0028 Acc: 0.9803
val Loss: 0.0089 Acc: 0.8462

Epoch 53/98
----------
train Loss: 0.0045 Acc: 0.9738
val Loss: 0.0082 Acc: 0.8462

Epoch 54/98
----------
train Loss: 0.0012 Acc: 0.9902
val Loss: 0.0084 Acc: 0.8538

Epoch 55/98
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0066 Acc: 0.8462

Epoch 56/98
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0009 Acc: 0.9869
val Loss: 0.0077 Acc: 0.8462

Epoch 57/98
----------
train Loss: 0.0023 Acc: 0.9803
val Loss: 0.0059 Acc: 0.8462

Epoch 58/98
----------
train Loss: 0.0031 Acc: 0.9836
val Loss: 0.0064 Acc: 0.8462

Epoch 59/98
----------
train Loss: 0.0041 Acc: 0.9869
val Loss: 0.0066 Acc: 0.8462

Epoch 60/98
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0021 Acc: 0.9770
val Loss: 0.0079 Acc: 0.8462

Epoch 61/98
----------
train Loss: 0.0022 Acc: 0.9803
val Loss: 0.0069 Acc: 0.8385

Epoch 62/98
----------
train Loss: 0.0013 Acc: 0.9836
val Loss: 0.0062 Acc: 0.8462

Epoch 63/98
----------
train Loss: 0.0039 Acc: 0.9869
val Loss: 0.0072 Acc: 0.8462

Epoch 64/98
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0014 Acc: 0.9770
val Loss: 0.0061 Acc: 0.8538

Epoch 65/98
----------
train Loss: 0.0010 Acc: 0.9869
val Loss: 0.0064 Acc: 0.8308

Epoch 66/98
----------
train Loss: 0.0011 Acc: 0.9770
val Loss: 0.0060 Acc: 0.8308

Epoch 67/98
----------
train Loss: 0.0013 Acc: 0.9869
val Loss: 0.0072 Acc: 0.8538

Epoch 68/98
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0008 Acc: 0.9869
val Loss: 0.0075 Acc: 0.8462

Epoch 69/98
----------
train Loss: 0.0025 Acc: 0.9738
val Loss: 0.0069 Acc: 0.8154

Epoch 70/98
----------
train Loss: 0.0024 Acc: 0.9836
val Loss: 0.0083 Acc: 0.8231

Epoch 71/98
----------
train Loss: 0.0037 Acc: 0.9770
val Loss: 0.0083 Acc: 0.8231

Epoch 72/98
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0084 Acc: 0.8154

Epoch 73/98
----------
train Loss: 0.0013 Acc: 0.9836
val Loss: 0.0058 Acc: 0.8385

Epoch 74/98
----------
train Loss: 0.0021 Acc: 0.9738
val Loss: 0.0067 Acc: 0.8462

Epoch 75/98
----------
train Loss: 0.0014 Acc: 0.9934
val Loss: 0.0076 Acc: 0.8538

Epoch 76/98
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0043 Acc: 0.9836
val Loss: 0.0077 Acc: 0.8385

Epoch 77/98
----------
train Loss: 0.0040 Acc: 0.9836
val Loss: 0.0072 Acc: 0.8077

Epoch 78/98
----------
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0085 Acc: 0.8308

Epoch 79/98
----------
train Loss: 0.0018 Acc: 0.9770
val Loss: 0.0085 Acc: 0.8385

Epoch 80/98
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0072 Acc: 0.8154

Epoch 81/98
----------
train Loss: 0.0015 Acc: 0.9934
val Loss: 0.0069 Acc: 0.8308

Epoch 82/98
----------
train Loss: 0.0016 Acc: 0.9869
val Loss: 0.0081 Acc: 0.8462

Epoch 83/98
----------
train Loss: 0.0010 Acc: 0.9934
val Loss: 0.0061 Acc: 0.8538

Epoch 84/98
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0010 Acc: 0.9902
val Loss: 0.0068 Acc: 0.8385

Epoch 85/98
----------
train Loss: 0.0012 Acc: 0.9869
val Loss: 0.0067 Acc: 0.8538

Epoch 86/98
----------
train Loss: 0.0033 Acc: 0.9836
val Loss: 0.0075 Acc: 0.8538

Epoch 87/98
----------
train Loss: 0.0018 Acc: 0.9803
val Loss: 0.0060 Acc: 0.8538

Epoch 88/98
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0019 Acc: 0.9836
val Loss: 0.0056 Acc: 0.8462

Epoch 89/98
----------
train Loss: 0.0016 Acc: 0.9803
val Loss: 0.0066 Acc: 0.8462

Epoch 90/98
----------
train Loss: 0.0016 Acc: 0.9902
val Loss: 0.0085 Acc: 0.8462

Epoch 91/98
----------
train Loss: 0.0020 Acc: 0.9803
val Loss: 0.0069 Acc: 0.8231

Epoch 92/98
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0024 Acc: 0.9836
val Loss: 0.0059 Acc: 0.8462

Epoch 93/98
----------
train Loss: 0.0018 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8462

Epoch 94/98
----------
train Loss: 0.0014 Acc: 0.9902
val Loss: 0.0064 Acc: 0.8385

Epoch 95/98
----------
train Loss: 0.0028 Acc: 0.9770
val Loss: 0.0068 Acc: 0.8385

Epoch 96/98
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0020 Acc: 0.9869
val Loss: 0.0065 Acc: 0.8385

Epoch 97/98
----------
train Loss: 0.0027 Acc: 0.9836
val Loss: 0.0072 Acc: 0.8308

Epoch 98/98
----------
train Loss: 0.0040 Acc: 0.9672
val Loss: 0.0074 Acc: 0.8385

Training complete in 4m 2s
Best val Acc: 0.861538

---Testing---
Test accuracy: 0.951724
--------------------
Accuracy of Dasyatiformes : 89 %
Accuracy of Myliobatiformes : 93 %
Accuracy of Rajiformes : 93 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of Torpediniformes : 96 %
mean: 0.9413629681815643, std: 0.027571950362333956

Model saved in "./weights/Batoidea(ga_oo_lee)_[0.99]_mean[0.98]_std[0.01].save".

Process finished with exit code 0
'''