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
since = time.time()

train_this = 'shark'
test_this = 'shark'


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

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/media/ray/PNU@myPC@DDDDD/workspace/python/git test/weight_trainera.py"
--------------------

run info[val: 0.1, epoch: 71, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0226 Acc: 0.2472
val Loss: 0.0246 Acc: 0.3529

Epoch 1/70
----------
train Loss: 0.0156 Acc: 0.4927
val Loss: 0.0171 Acc: 0.6029

Epoch 2/70
----------
train Loss: 0.0080 Acc: 0.7754
val Loss: 0.0119 Acc: 0.7206

Epoch 3/70
----------
train Loss: 0.0062 Acc: 0.8352
val Loss: 0.0102 Acc: 0.7353

Epoch 4/70
----------
train Loss: 0.0048 Acc: 0.8708
val Loss: 0.0124 Acc: 0.6765

Epoch 5/70
----------
LR is set to 0.001
train Loss: 0.0048 Acc: 0.8578
val Loss: 0.0123 Acc: 0.6765

Epoch 6/70
----------
train Loss: 0.0039 Acc: 0.8918
val Loss: 0.0107 Acc: 0.7500

Epoch 7/70
----------
train Loss: 0.0037 Acc: 0.9176
val Loss: 0.0099 Acc: 0.7647

Epoch 8/70
----------
train Loss: 0.0037 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7794

Epoch 9/70
----------
train Loss: 0.0034 Acc: 0.9305
val Loss: 0.0098 Acc: 0.7500

Epoch 10/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0036 Acc: 0.9208
val Loss: 0.0098 Acc: 0.7500

Epoch 11/70
----------
train Loss: 0.0039 Acc: 0.9144
val Loss: 0.0098 Acc: 0.7500

Epoch 12/70
----------
train Loss: 0.0034 Acc: 0.9176
val Loss: 0.0099 Acc: 0.7500

Epoch 13/70
----------
train Loss: 0.0035 Acc: 0.9225
val Loss: 0.0098 Acc: 0.7500

Epoch 14/70
----------
train Loss: 0.0036 Acc: 0.9225
val Loss: 0.0097 Acc: 0.7500

Epoch 15/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0098 Acc: 0.7500

Epoch 16/70
----------
train Loss: 0.0035 Acc: 0.9273
val Loss: 0.0101 Acc: 0.7500

Epoch 17/70
----------
train Loss: 0.0033 Acc: 0.9160
val Loss: 0.0100 Acc: 0.7500

Epoch 18/70
----------
train Loss: 0.0035 Acc: 0.9338
val Loss: 0.0101 Acc: 0.7500

Epoch 19/70
----------
train Loss: 0.0037 Acc: 0.9160
val Loss: 0.0100 Acc: 0.7500

Epoch 20/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.9321
val Loss: 0.0100 Acc: 0.7500

Epoch 21/70
----------
train Loss: 0.0034 Acc: 0.9192
val Loss: 0.0100 Acc: 0.7500

Epoch 22/70
----------
train Loss: 0.0038 Acc: 0.9192
val Loss: 0.0100 Acc: 0.7500

Epoch 23/70
----------
train Loss: 0.0036 Acc: 0.9176
val Loss: 0.0098 Acc: 0.7500

Epoch 24/70
----------
train Loss: 0.0039 Acc: 0.9208
val Loss: 0.0100 Acc: 0.7500

Epoch 25/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0035 Acc: 0.9241
val Loss: 0.0099 Acc: 0.7500

Epoch 26/70
----------
train Loss: 0.0034 Acc: 0.9305
val Loss: 0.0099 Acc: 0.7500

Epoch 27/70
----------
train Loss: 0.0035 Acc: 0.9241
val Loss: 0.0099 Acc: 0.7500

Epoch 28/70
----------
train Loss: 0.0036 Acc: 0.9305
val Loss: 0.0099 Acc: 0.7500

Epoch 29/70
----------
train Loss: 0.0037 Acc: 0.9176
val Loss: 0.0100 Acc: 0.7500

Epoch 30/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0036 Acc: 0.9225
val Loss: 0.0101 Acc: 0.7500

Epoch 31/70
----------
train Loss: 0.0034 Acc: 0.9128
val Loss: 0.0100 Acc: 0.7500

Epoch 32/70
----------
train Loss: 0.0035 Acc: 0.9208
val Loss: 0.0100 Acc: 0.7500

Epoch 33/70
----------
train Loss: 0.0035 Acc: 0.9128
val Loss: 0.0101 Acc: 0.7500

Epoch 34/70
----------
train Loss: 0.0039 Acc: 0.9111
val Loss: 0.0099 Acc: 0.7500

Epoch 35/70
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0037 Acc: 0.9225
val Loss: 0.0098 Acc: 0.7500

Epoch 36/70
----------
train Loss: 0.0036 Acc: 0.9241
val Loss: 0.0101 Acc: 0.7500

Epoch 37/70
----------
train Loss: 0.0037 Acc: 0.9305
val Loss: 0.0099 Acc: 0.7500

Epoch 38/70
----------
train Loss: 0.0036 Acc: 0.9192
val Loss: 0.0099 Acc: 0.7500

Epoch 39/70
----------
train Loss: 0.0035 Acc: 0.9289
val Loss: 0.0100 Acc: 0.7500

Epoch 40/70
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0036 Acc: 0.9192
val Loss: 0.0099 Acc: 0.7500

Epoch 41/70
----------
train Loss: 0.0034 Acc: 0.9289
val Loss: 0.0099 Acc: 0.7500

Epoch 42/70
----------
train Loss: 0.0038 Acc: 0.9047
val Loss: 0.0099 Acc: 0.7500

Epoch 43/70
----------
train Loss: 0.0035 Acc: 0.9160
val Loss: 0.0099 Acc: 0.7500

Epoch 44/70
----------
train Loss: 0.0035 Acc: 0.9176
val Loss: 0.0100 Acc: 0.7500

Epoch 45/70
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0039 Acc: 0.9079
val Loss: 0.0100 Acc: 0.7500

Epoch 46/70
----------
train Loss: 0.0037 Acc: 0.9111
val Loss: 0.0099 Acc: 0.7500

Epoch 47/70
----------
train Loss: 0.0036 Acc: 0.9289
val Loss: 0.0099 Acc: 0.7500

Epoch 48/70
----------
train Loss: 0.0041 Acc: 0.9111
val Loss: 0.0097 Acc: 0.7500

Epoch 49/70
----------
train Loss: 0.0041 Acc: 0.9111
val Loss: 0.0098 Acc: 0.7500

Epoch 50/70
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0035 Acc: 0.9289
val Loss: 0.0099 Acc: 0.7500

Epoch 51/70
----------
train Loss: 0.0035 Acc: 0.9305
val Loss: 0.0101 Acc: 0.7500

Epoch 52/70
----------
train Loss: 0.0036 Acc: 0.9176
val Loss: 0.0099 Acc: 0.7500

Epoch 53/70
----------
train Loss: 0.0037 Acc: 0.9160
val Loss: 0.0099 Acc: 0.7500

Epoch 54/70
----------
train Loss: 0.0035 Acc: 0.9160
val Loss: 0.0100 Acc: 0.7500

Epoch 55/70
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0034 Acc: 0.9144
val Loss: 0.0099 Acc: 0.7500

Epoch 56/70
----------
train Loss: 0.0036 Acc: 0.9208
val Loss: 0.0099 Acc: 0.7500

Epoch 57/70
----------
train Loss: 0.0035 Acc: 0.9257
val Loss: 0.0099 Acc: 0.7500

Epoch 58/70
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0098 Acc: 0.7500

Epoch 59/70
----------
train Loss: 0.0034 Acc: 0.9257
val Loss: 0.0099 Acc: 0.7500

Epoch 60/70
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0034 Acc: 0.9208
val Loss: 0.0099 Acc: 0.7500

Epoch 61/70
----------
train Loss: 0.0037 Acc: 0.9160
val Loss: 0.0099 Acc: 0.7500

Epoch 62/70
----------
train Loss: 0.0033 Acc: 0.9241
val Loss: 0.0099 Acc: 0.7500

Epoch 63/70
----------
train Loss: 0.0034 Acc: 0.9225
val Loss: 0.0100 Acc: 0.7500

Epoch 64/70
----------
train Loss: 0.0037 Acc: 0.9192
val Loss: 0.0098 Acc: 0.7500

Epoch 65/70
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0039 Acc: 0.9079
val Loss: 0.0099 Acc: 0.7500

Epoch 66/70
----------
train Loss: 0.0035 Acc: 0.9176
val Loss: 0.0100 Acc: 0.7500

Epoch 67/70
----------
train Loss: 0.0040 Acc: 0.9192
val Loss: 0.0099 Acc: 0.7500

Epoch 68/70
----------
train Loss: 0.0035 Acc: 0.9160
val Loss: 0.0099 Acc: 0.7500

Epoch 69/70
----------
train Loss: 0.0040 Acc: 0.9208
val Loss: 0.0100 Acc: 0.7500

Epoch 70/70
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0037 Acc: 0.9257
val Loss: 0.0098 Acc: 0.7500

Training complete in 3m 27s
Best val Acc: 0.779412

---Fine tuning.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0038 Acc: 0.9208
val Loss: 0.0094 Acc: 0.7794

Epoch 1/70
----------
train Loss: 0.0021 Acc: 0.9515
val Loss: 0.0095 Acc: 0.8088

Epoch 2/70
----------
train Loss: 0.0009 Acc: 0.9855
val Loss: 0.0081 Acc: 0.8088

Epoch 3/70
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8235

Epoch 4/70
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0059 Acc: 0.8676

Epoch 5/70
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8235

Epoch 6/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8235

Epoch 7/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8382

Epoch 8/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8235

Epoch 9/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8235

Epoch 10/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 11/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 12/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 13/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 14/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 15/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8088

Epoch 16/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 17/70
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0061 Acc: 0.8088

Epoch 18/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8088

Epoch 19/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8088

Epoch 20/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 21/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 22/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 23/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 24/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8382

Epoch 25/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8088

Epoch 26/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8382

Epoch 27/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8529

Epoch 28/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8382

Epoch 29/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8235

Epoch 30/70
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8088

Epoch 31/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8382

Epoch 32/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8235

Epoch 33/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 34/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8235

Epoch 35/70
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8088

Epoch 36/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 37/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8235

Epoch 38/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8235

Epoch 39/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8235

Epoch 40/70
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8382

Epoch 41/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8382

Epoch 42/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8235

Epoch 43/70
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 44/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8382

Epoch 45/70
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8382

Epoch 46/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 47/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 48/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 49/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 50/70
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8235

Epoch 51/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8088

Epoch 52/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8235

Epoch 53/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 54/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 55/70
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8088

Epoch 56/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8088

Epoch 57/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 58/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8088

Epoch 59/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8382

Epoch 60/70
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8235

Epoch 61/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8235

Epoch 62/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 63/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8382

Epoch 64/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8382

Epoch 65/70
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8235

Epoch 66/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8235

Epoch 67/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8529

Epoch 68/70
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Epoch 69/70
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8382

Epoch 70/70
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8088

Training complete in 3m 39s
Best val Acc: 0.867647

---Testing---
Test accuracy: 0.986900
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 96 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 97 %
mean: 0.982838630667578, std: 0.013604158553995464
--------------------

run info[val: 0.15, epoch: 57, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0202 Acc: 0.2517
val Loss: 0.0349 Acc: 0.3495

Epoch 1/56
----------
train Loss: 0.0134 Acc: 0.5702
val Loss: 0.0197 Acc: 0.5922

Epoch 2/56
----------
train Loss: 0.0087 Acc: 0.7654
val Loss: 0.0153 Acc: 0.7282

Epoch 3/56
----------
train Loss: 0.0067 Acc: 0.7877
val Loss: 0.0114 Acc: 0.7670

Epoch 4/56
----------
train Loss: 0.0054 Acc: 0.8493
val Loss: 0.0143 Acc: 0.8058

Epoch 5/56
----------
train Loss: 0.0043 Acc: 0.8750
val Loss: 0.0118 Acc: 0.7767

Epoch 6/56
----------
train Loss: 0.0036 Acc: 0.9041
val Loss: 0.0127 Acc: 0.8155

Epoch 7/56
----------
train Loss: 0.0036 Acc: 0.8955
val Loss: 0.0082 Acc: 0.7864

Epoch 8/56
----------
train Loss: 0.0033 Acc: 0.9058
val Loss: 0.0079 Acc: 0.8155

Epoch 9/56
----------
train Loss: 0.0031 Acc: 0.9298
val Loss: 0.0095 Acc: 0.7961

Epoch 10/56
----------
LR is set to 0.001
train Loss: 0.0027 Acc: 0.9298
val Loss: 0.0068 Acc: 0.7961

Epoch 11/56
----------
train Loss: 0.0029 Acc: 0.9315
val Loss: 0.0086 Acc: 0.7961

Epoch 12/56
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0108 Acc: 0.8058

Epoch 13/56
----------
train Loss: 0.0028 Acc: 0.9349
val Loss: 0.0067 Acc: 0.8155

Epoch 14/56
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0137 Acc: 0.8058

Epoch 15/56
----------
train Loss: 0.0026 Acc: 0.9452
val Loss: 0.0086 Acc: 0.8155

Epoch 16/56
----------
train Loss: 0.0026 Acc: 0.9555
val Loss: 0.0062 Acc: 0.7961

Epoch 17/56
----------
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0119 Acc: 0.7864

Epoch 18/56
----------
train Loss: 0.0025 Acc: 0.9401
val Loss: 0.0082 Acc: 0.7864

Epoch 19/56
----------
train Loss: 0.0025 Acc: 0.9401
val Loss: 0.0172 Acc: 0.7864

Epoch 20/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0090 Acc: 0.7864

Epoch 21/56
----------
train Loss: 0.0026 Acc: 0.9503
val Loss: 0.0095 Acc: 0.7961

Epoch 22/56
----------
train Loss: 0.0027 Acc: 0.9452
val Loss: 0.0212 Acc: 0.7864

Epoch 23/56
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0083 Acc: 0.7961

Epoch 24/56
----------
train Loss: 0.0025 Acc: 0.9418
val Loss: 0.0118 Acc: 0.7864

Epoch 25/56
----------
train Loss: 0.0025 Acc: 0.9469
val Loss: 0.0079 Acc: 0.7961

Epoch 26/56
----------
train Loss: 0.0027 Acc: 0.9384
val Loss: 0.0073 Acc: 0.7864

Epoch 27/56
----------
train Loss: 0.0025 Acc: 0.9521
val Loss: 0.0195 Acc: 0.7961

Epoch 28/56
----------
train Loss: 0.0025 Acc: 0.9332
val Loss: 0.0061 Acc: 0.7961

Epoch 29/56
----------
train Loss: 0.0027 Acc: 0.9435
val Loss: 0.0181 Acc: 0.7961

Epoch 30/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0026 Acc: 0.9435
val Loss: 0.0150 Acc: 0.7961

Epoch 31/56
----------
train Loss: 0.0024 Acc: 0.9503
val Loss: 0.0264 Acc: 0.7961

Epoch 32/56
----------
train Loss: 0.0028 Acc: 0.9281
val Loss: 0.0067 Acc: 0.7864

Epoch 33/56
----------
train Loss: 0.0025 Acc: 0.9452
val Loss: 0.0081 Acc: 0.7864

Epoch 34/56
----------
train Loss: 0.0026 Acc: 0.9435
val Loss: 0.0064 Acc: 0.7961

Epoch 35/56
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0098 Acc: 0.7961

Epoch 36/56
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0088 Acc: 0.7961

Epoch 37/56
----------
train Loss: 0.0025 Acc: 0.9384
val Loss: 0.0075 Acc: 0.7961

Epoch 38/56
----------
train Loss: 0.0026 Acc: 0.9452
val Loss: 0.0081 Acc: 0.7961

Epoch 39/56
----------
train Loss: 0.0026 Acc: 0.9435
val Loss: 0.0092 Acc: 0.7961

Epoch 40/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0025 Acc: 0.9435
val Loss: 0.0082 Acc: 0.7961

Epoch 41/56
----------
train Loss: 0.0025 Acc: 0.9418
val Loss: 0.0243 Acc: 0.7961

Epoch 42/56
----------
train Loss: 0.0027 Acc: 0.9315
val Loss: 0.0088 Acc: 0.7961

Epoch 43/56
----------
train Loss: 0.0027 Acc: 0.9469
val Loss: 0.0104 Acc: 0.7961

Epoch 44/56
----------
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0130 Acc: 0.7961

Epoch 45/56
----------
train Loss: 0.0025 Acc: 0.9349
val Loss: 0.0066 Acc: 0.7961

Epoch 46/56
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0075 Acc: 0.7961

Epoch 47/56
----------
train Loss: 0.0026 Acc: 0.9332
val Loss: 0.0069 Acc: 0.7961

Epoch 48/56
----------
train Loss: 0.0027 Acc: 0.9315
val Loss: 0.0087 Acc: 0.7961

Epoch 49/56
----------
train Loss: 0.0026 Acc: 0.9435
val Loss: 0.0096 Acc: 0.7961

Epoch 50/56
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0120 Acc: 0.7961

Epoch 51/56
----------
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0126 Acc: 0.7961

Epoch 52/56
----------
train Loss: 0.0026 Acc: 0.9384
val Loss: 0.0142 Acc: 0.7864

Epoch 53/56
----------
train Loss: 0.0026 Acc: 0.9452
val Loss: 0.0170 Acc: 0.7961

Epoch 54/56
----------
train Loss: 0.0026 Acc: 0.9384
val Loss: 0.0070 Acc: 0.7961

Epoch 55/56
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0125 Acc: 0.7961

Epoch 56/56
----------
train Loss: 0.0024 Acc: 0.9503
val Loss: 0.0068 Acc: 0.7961

Training complete in 2m 52s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0036 Acc: 0.9041
val Loss: 0.0171 Acc: 0.7961

Epoch 1/56
----------
train Loss: 0.0022 Acc: 0.9538
val Loss: 0.0077 Acc: 0.8447

Epoch 2/56
----------
train Loss: 0.0012 Acc: 0.9795
val Loss: 0.0072 Acc: 0.8252

Epoch 3/56
----------
train Loss: 0.0005 Acc: 0.9966
val Loss: 0.0053 Acc: 0.8447

Epoch 4/56
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0151 Acc: 0.8447

Epoch 5/56
----------
train Loss: 0.0003 Acc: 0.9966
val Loss: 0.0162 Acc: 0.8447

Epoch 6/56
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0057 Acc: 0.8350

Epoch 7/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8252

Epoch 8/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0153 Acc: 0.8252

Epoch 9/56
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0049 Acc: 0.8544

Epoch 10/56
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0156 Acc: 0.8544

Epoch 11/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0204 Acc: 0.8544

Epoch 12/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8544

Epoch 13/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8544

Epoch 14/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8544

Epoch 15/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8544

Epoch 16/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8544

Epoch 17/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 18/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 19/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0237 Acc: 0.8544

Epoch 20/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 21/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0322 Acc: 0.8544

Epoch 22/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0110 Acc: 0.8544

Epoch 23/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8544

Epoch 24/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 25/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 26/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0206 Acc: 0.8544

Epoch 27/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8544

Epoch 28/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 29/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8544

Epoch 30/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0175 Acc: 0.8544

Epoch 31/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8544

Epoch 32/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 33/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 34/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 35/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 36/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 37/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 38/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8544

Epoch 39/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8544

Epoch 40/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0149 Acc: 0.8544

Epoch 41/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8544

Epoch 42/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8544

Epoch 43/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0239 Acc: 0.8544

Epoch 44/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8544

Epoch 45/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8544

Epoch 46/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8544

Epoch 47/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8544

Epoch 48/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 49/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8544

Epoch 50/56
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8544

Epoch 51/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8544

Epoch 52/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 53/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 54/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8544

Epoch 55/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0159 Acc: 0.8544

Epoch 56/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8544

Training complete in 3m 2s
Best val Acc: 0.854369

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 97 %
mean: 0.9699894622920939, std: 0.026100939572756294
--------------------

run info[val: 0.2, epoch: 65, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0211 Acc: 0.2636
val Loss: 0.0229 Acc: 0.4526

Epoch 1/64
----------
train Loss: 0.0139 Acc: 0.6055
val Loss: 0.0174 Acc: 0.5328

Epoch 2/64
----------
train Loss: 0.0091 Acc: 0.7109
val Loss: 0.0120 Acc: 0.7664

Epoch 3/64
----------
train Loss: 0.0069 Acc: 0.8200
val Loss: 0.0110 Acc: 0.7591

Epoch 4/64
----------
train Loss: 0.0056 Acc: 0.8418
val Loss: 0.0089 Acc: 0.8175

Epoch 5/64
----------
train Loss: 0.0048 Acc: 0.8600
val Loss: 0.0088 Acc: 0.8175

Epoch 6/64
----------
train Loss: 0.0040 Acc: 0.9018
val Loss: 0.0072 Acc: 0.8175

Epoch 7/64
----------
train Loss: 0.0037 Acc: 0.8927
val Loss: 0.0078 Acc: 0.8248

Epoch 8/64
----------
train Loss: 0.0036 Acc: 0.9109
val Loss: 0.0074 Acc: 0.8248

Epoch 9/64
----------
train Loss: 0.0033 Acc: 0.9182
val Loss: 0.0067 Acc: 0.8175

Epoch 10/64
----------
train Loss: 0.0028 Acc: 0.9309
val Loss: 0.0083 Acc: 0.8394

Epoch 11/64
----------
LR is set to 0.001
train Loss: 0.0028 Acc: 0.9273
val Loss: 0.0089 Acc: 0.8321

Epoch 12/64
----------
train Loss: 0.0027 Acc: 0.9364
val Loss: 0.0077 Acc: 0.8394

Epoch 13/64
----------
train Loss: 0.0027 Acc: 0.9400
val Loss: 0.0077 Acc: 0.8467

Epoch 14/64
----------
train Loss: 0.0028 Acc: 0.9382
val Loss: 0.0070 Acc: 0.8467

Epoch 15/64
----------
train Loss: 0.0028 Acc: 0.9491
val Loss: 0.0071 Acc: 0.8467

Epoch 16/64
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0071 Acc: 0.8467

Epoch 17/64
----------
train Loss: 0.0026 Acc: 0.9436
val Loss: 0.0073 Acc: 0.8467

Epoch 18/64
----------
train Loss: 0.0027 Acc: 0.9545
val Loss: 0.0066 Acc: 0.8467

Epoch 19/64
----------
train Loss: 0.0026 Acc: 0.9455
val Loss: 0.0071 Acc: 0.8467

Epoch 20/64
----------
train Loss: 0.0026 Acc: 0.9473
val Loss: 0.0067 Acc: 0.8467

Epoch 21/64
----------
train Loss: 0.0023 Acc: 0.9527
val Loss: 0.0074 Acc: 0.8467

Epoch 22/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0026 Acc: 0.9564
val Loss: 0.0077 Acc: 0.8467

Epoch 23/64
----------
train Loss: 0.0027 Acc: 0.9436
val Loss: 0.0065 Acc: 0.8467

Epoch 24/64
----------
train Loss: 0.0027 Acc: 0.9509
val Loss: 0.0066 Acc: 0.8467

Epoch 25/64
----------
train Loss: 0.0027 Acc: 0.9309
val Loss: 0.0081 Acc: 0.8467

Epoch 26/64
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0066 Acc: 0.8394

Epoch 27/64
----------
train Loss: 0.0027 Acc: 0.9364
val Loss: 0.0081 Acc: 0.8394

Epoch 28/64
----------
train Loss: 0.0026 Acc: 0.9545
val Loss: 0.0079 Acc: 0.8394

Epoch 29/64
----------
train Loss: 0.0026 Acc: 0.9436
val Loss: 0.0076 Acc: 0.8394

Epoch 30/64
----------
train Loss: 0.0026 Acc: 0.9400
val Loss: 0.0075 Acc: 0.8321

Epoch 31/64
----------
train Loss: 0.0027 Acc: 0.9545
val Loss: 0.0071 Acc: 0.8467

Epoch 32/64
----------
train Loss: 0.0025 Acc: 0.9600
val Loss: 0.0089 Acc: 0.8467

Epoch 33/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0030 Acc: 0.9309
val Loss: 0.0085 Acc: 0.8467

Epoch 34/64
----------
train Loss: 0.0028 Acc: 0.9527
val Loss: 0.0079 Acc: 0.8467

Epoch 35/64
----------
train Loss: 0.0024 Acc: 0.9509
val Loss: 0.0078 Acc: 0.8467

Epoch 36/64
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0070 Acc: 0.8467

Epoch 37/64
----------
train Loss: 0.0028 Acc: 0.9364
val Loss: 0.0072 Acc: 0.8467

Epoch 38/64
----------
train Loss: 0.0026 Acc: 0.9509
val Loss: 0.0079 Acc: 0.8467

Epoch 39/64
----------
train Loss: 0.0027 Acc: 0.9400
val Loss: 0.0066 Acc: 0.8467

Epoch 40/64
----------
train Loss: 0.0027 Acc: 0.9436
val Loss: 0.0073 Acc: 0.8467

Epoch 41/64
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0080 Acc: 0.8467

Epoch 42/64
----------
train Loss: 0.0027 Acc: 0.9473
val Loss: 0.0075 Acc: 0.8394

Epoch 43/64
----------
train Loss: 0.0028 Acc: 0.9400
val Loss: 0.0062 Acc: 0.8467

Epoch 44/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0084 Acc: 0.8467

Epoch 45/64
----------
train Loss: 0.0028 Acc: 0.9382
val Loss: 0.0067 Acc: 0.8394

Epoch 46/64
----------
train Loss: 0.0026 Acc: 0.9436
val Loss: 0.0073 Acc: 0.8394

Epoch 47/64
----------
train Loss: 0.0028 Acc: 0.9382
val Loss: 0.0073 Acc: 0.8394

Epoch 48/64
----------
train Loss: 0.0026 Acc: 0.9436
val Loss: 0.0062 Acc: 0.8467

Epoch 49/64
----------
train Loss: 0.0024 Acc: 0.9582
val Loss: 0.0077 Acc: 0.8394

Epoch 50/64
----------
train Loss: 0.0024 Acc: 0.9527
val Loss: 0.0068 Acc: 0.8394

Epoch 51/64
----------
train Loss: 0.0025 Acc: 0.9582
val Loss: 0.0067 Acc: 0.8394

Epoch 52/64
----------
train Loss: 0.0024 Acc: 0.9509
val Loss: 0.0077 Acc: 0.8467

Epoch 53/64
----------
train Loss: 0.0026 Acc: 0.9509
val Loss: 0.0071 Acc: 0.8467

Epoch 54/64
----------
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0063 Acc: 0.8467

Epoch 55/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0028 Acc: 0.9327
val Loss: 0.0082 Acc: 0.8467

Epoch 56/64
----------
train Loss: 0.0027 Acc: 0.9364
val Loss: 0.0078 Acc: 0.8394

Epoch 57/64
----------
train Loss: 0.0027 Acc: 0.9455
val Loss: 0.0078 Acc: 0.8467

Epoch 58/64
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0068 Acc: 0.8394

Epoch 59/64
----------
train Loss: 0.0025 Acc: 0.9491
val Loss: 0.0065 Acc: 0.8394

Epoch 60/64
----------
train Loss: 0.0026 Acc: 0.9527
val Loss: 0.0065 Acc: 0.8467

Epoch 61/64
----------
train Loss: 0.0027 Acc: 0.9473
val Loss: 0.0067 Acc: 0.8467

Epoch 62/64
----------
train Loss: 0.0025 Acc: 0.9509
val Loss: 0.0067 Acc: 0.8467

Epoch 63/64
----------
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0078 Acc: 0.8321

Epoch 64/64
----------
train Loss: 0.0024 Acc: 0.9545
val Loss: 0.0079 Acc: 0.8321

Training complete in 3m 11s
Best val Acc: 0.846715

---Fine tuning.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0027 Acc: 0.9327
val Loss: 0.0067 Acc: 0.8467

Epoch 1/64
----------
train Loss: 0.0014 Acc: 0.9782
val Loss: 0.0055 Acc: 0.8832

Epoch 2/64
----------
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0069 Acc: 0.8613

Epoch 3/64
----------
train Loss: 0.0008 Acc: 0.9909
val Loss: 0.0071 Acc: 0.8394

Epoch 4/64
----------
train Loss: 0.0005 Acc: 0.9909
val Loss: 0.0066 Acc: 0.8540

Epoch 5/64
----------
train Loss: 0.0003 Acc: 0.9927
val Loss: 0.0061 Acc: 0.8540

Epoch 6/64
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8613

Epoch 7/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8613

Epoch 8/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8613

Epoch 9/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8540

Epoch 10/64
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0088 Acc: 0.8540

Epoch 11/64
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8540

Epoch 12/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8540

Epoch 13/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8613

Epoch 14/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8540

Epoch 15/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8540

Epoch 16/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8613

Epoch 17/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8540

Epoch 18/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8540

Epoch 19/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8540

Epoch 20/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8540

Epoch 21/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8540

Epoch 22/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0068 Acc: 0.8540

Epoch 23/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8540

Epoch 24/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8686

Epoch 25/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8613

Epoch 26/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8613

Epoch 27/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8686

Epoch 28/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8613

Epoch 29/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8613

Epoch 30/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8613

Epoch 31/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8613

Epoch 32/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8613

Epoch 33/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8613

Epoch 34/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8613

Epoch 35/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8613

Epoch 36/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8613

Epoch 37/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8613

Epoch 38/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8613

Epoch 39/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8613

Epoch 40/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8540

Epoch 41/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8613

Epoch 42/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8540

Epoch 43/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8613

Epoch 44/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8613

Epoch 45/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8613

Epoch 46/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8540

Epoch 47/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8613

Epoch 48/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8613

Epoch 49/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8613

Epoch 50/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8613

Epoch 51/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8613

Epoch 52/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8613

Epoch 53/64
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0078 Acc: 0.8613

Epoch 54/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8540

Epoch 55/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8540

Epoch 56/64
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8613

Epoch 57/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8613

Epoch 58/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8613

Epoch 59/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8613

Epoch 60/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8613

Epoch 61/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8540

Epoch 62/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8613

Epoch 63/64
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8540

Epoch 64/64
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8613

Training complete in 3m 23s
Best val Acc: 0.883212

---Testing---
Test accuracy: 0.960699
--------------------
Accuracy of Carcharhiniformes : 94 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 96 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 85 %
Accuracy of Squatiniformes : 97 %
mean: 0.9516559473730085, std: 0.04297273632436073
--------------------

run info[val: 0.25, epoch: 64, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0234 Acc: 0.2112
val Loss: 0.0192 Acc: 0.4327

Epoch 1/63
----------
train Loss: 0.0155 Acc: 0.5911
val Loss: 0.0140 Acc: 0.5322

Epoch 2/63
----------
train Loss: 0.0095 Acc: 0.7248
val Loss: 0.0109 Acc: 0.6667

Epoch 3/63
----------
train Loss: 0.0097 Acc: 0.7364
val Loss: 0.0073 Acc: 0.7719

Epoch 4/63
----------
train Loss: 0.0062 Acc: 0.8004
val Loss: 0.0083 Acc: 0.7719

Epoch 5/63
----------
train Loss: 0.0048 Acc: 0.8702
val Loss: 0.0075 Acc: 0.7544

Epoch 6/63
----------
train Loss: 0.0045 Acc: 0.8992
val Loss: 0.0068 Acc: 0.8187

Epoch 7/63
----------
train Loss: 0.0035 Acc: 0.9205
val Loss: 0.0071 Acc: 0.7778

Epoch 8/63
----------
train Loss: 0.0038 Acc: 0.9089
val Loss: 0.0061 Acc: 0.8421

Epoch 9/63
----------
train Loss: 0.0036 Acc: 0.9186
val Loss: 0.0060 Acc: 0.8070

Epoch 10/63
----------
LR is set to 0.001
train Loss: 0.0029 Acc: 0.9244
val Loss: 0.0063 Acc: 0.7953

Epoch 11/63
----------
train Loss: 0.0026 Acc: 0.9360
val Loss: 0.0060 Acc: 0.8012

Epoch 12/63
----------
train Loss: 0.0028 Acc: 0.9380
val Loss: 0.0059 Acc: 0.8187

Epoch 13/63
----------
train Loss: 0.0024 Acc: 0.9632
val Loss: 0.0060 Acc: 0.8363

Epoch 14/63
----------
train Loss: 0.0024 Acc: 0.9574
val Loss: 0.0060 Acc: 0.8363

Epoch 15/63
----------
train Loss: 0.0023 Acc: 0.9651
val Loss: 0.0060 Acc: 0.8363

Epoch 16/63
----------
train Loss: 0.0021 Acc: 0.9671
val Loss: 0.0057 Acc: 0.8304

Epoch 17/63
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0055 Acc: 0.8187

Epoch 18/63
----------
train Loss: 0.0024 Acc: 0.9651
val Loss: 0.0064 Acc: 0.8246

Epoch 19/63
----------
train Loss: 0.0023 Acc: 0.9593
val Loss: 0.0060 Acc: 0.8246

Epoch 20/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0023 Acc: 0.9671
val Loss: 0.0057 Acc: 0.8246

Epoch 21/63
----------
train Loss: 0.0022 Acc: 0.9632
val Loss: 0.0060 Acc: 0.8246

Epoch 22/63
----------
train Loss: 0.0024 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8246

Epoch 23/63
----------
train Loss: 0.0022 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8187

Epoch 24/63
----------
train Loss: 0.0025 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8304

Epoch 25/63
----------
train Loss: 0.0024 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8187

Epoch 26/63
----------
train Loss: 0.0027 Acc: 0.9690
val Loss: 0.0058 Acc: 0.8246

Epoch 27/63
----------
train Loss: 0.0022 Acc: 0.9651
val Loss: 0.0061 Acc: 0.8363

Epoch 28/63
----------
train Loss: 0.0023 Acc: 0.9535
val Loss: 0.0059 Acc: 0.8246

Epoch 29/63
----------
train Loss: 0.0022 Acc: 0.9729
val Loss: 0.0060 Acc: 0.8304

Epoch 30/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0058 Acc: 0.8187

Epoch 31/63
----------
train Loss: 0.0020 Acc: 0.9690
val Loss: 0.0056 Acc: 0.8187

Epoch 32/63
----------
train Loss: 0.0029 Acc: 0.9574
val Loss: 0.0059 Acc: 0.8129

Epoch 33/63
----------
train Loss: 0.0024 Acc: 0.9709
val Loss: 0.0061 Acc: 0.8187

Epoch 34/63
----------
train Loss: 0.0022 Acc: 0.9671
val Loss: 0.0057 Acc: 0.8187

Epoch 35/63
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0059 Acc: 0.8246

Epoch 36/63
----------
train Loss: 0.0022 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8246

Epoch 37/63
----------
train Loss: 0.0021 Acc: 0.9593
val Loss: 0.0059 Acc: 0.8246

Epoch 38/63
----------
train Loss: 0.0022 Acc: 0.9671
val Loss: 0.0060 Acc: 0.8187

Epoch 39/63
----------
train Loss: 0.0025 Acc: 0.9593
val Loss: 0.0060 Acc: 0.8187

Epoch 40/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0020 Acc: 0.9671
val Loss: 0.0059 Acc: 0.8187

Epoch 41/63
----------
train Loss: 0.0024 Acc: 0.9709
val Loss: 0.0060 Acc: 0.8187

Epoch 42/63
----------
train Loss: 0.0022 Acc: 0.9671
val Loss: 0.0057 Acc: 0.8246

Epoch 43/63
----------
train Loss: 0.0020 Acc: 0.9593
val Loss: 0.0058 Acc: 0.8246

Epoch 44/63
----------
train Loss: 0.0026 Acc: 0.9574
val Loss: 0.0058 Acc: 0.8304

Epoch 45/63
----------
train Loss: 0.0023 Acc: 0.9593
val Loss: 0.0058 Acc: 0.8363

Epoch 46/63
----------
train Loss: 0.0022 Acc: 0.9593
val Loss: 0.0058 Acc: 0.8246

Epoch 47/63
----------
train Loss: 0.0027 Acc: 0.9612
val Loss: 0.0058 Acc: 0.8187

Epoch 48/63
----------
train Loss: 0.0021 Acc: 0.9671
val Loss: 0.0059 Acc: 0.8304

Epoch 49/63
----------
train Loss: 0.0024 Acc: 0.9632
val Loss: 0.0058 Acc: 0.8304

Epoch 50/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0056 Acc: 0.8304

Epoch 51/63
----------
train Loss: 0.0027 Acc: 0.9535
val Loss: 0.0059 Acc: 0.8304

Epoch 52/63
----------
train Loss: 0.0021 Acc: 0.9651
val Loss: 0.0060 Acc: 0.8304

Epoch 53/63
----------
train Loss: 0.0020 Acc: 0.9632
val Loss: 0.0057 Acc: 0.8246

Epoch 54/63
----------
train Loss: 0.0028 Acc: 0.9651
val Loss: 0.0058 Acc: 0.8246

Epoch 55/63
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0056 Acc: 0.8187

Epoch 56/63
----------
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0057 Acc: 0.8187

Epoch 57/63
----------
train Loss: 0.0019 Acc: 0.9651
val Loss: 0.0057 Acc: 0.8187

Epoch 58/63
----------
train Loss: 0.0021 Acc: 0.9632
val Loss: 0.0059 Acc: 0.8187

Epoch 59/63
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0059 Acc: 0.8187

Epoch 60/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0025 Acc: 0.9574
val Loss: 0.0059 Acc: 0.8187

Epoch 61/63
----------
train Loss: 0.0021 Acc: 0.9612
val Loss: 0.0057 Acc: 0.8187

Epoch 62/63
----------
train Loss: 0.0021 Acc: 0.9690
val Loss: 0.0056 Acc: 0.8246

Epoch 63/63
----------
train Loss: 0.0021 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8246

Training complete in 3m 10s
Best val Acc: 0.842105

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0028 Acc: 0.9438
val Loss: 0.0058 Acc: 0.8363

Epoch 1/63
----------
train Loss: 0.0016 Acc: 0.9767
val Loss: 0.0056 Acc: 0.8363

Epoch 2/63
----------
train Loss: 0.0011 Acc: 0.9787
val Loss: 0.0064 Acc: 0.8304

Epoch 3/63
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8772

Epoch 4/63
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8713

Epoch 5/63
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0041 Acc: 0.8713

Epoch 6/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8772

Epoch 7/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8655

Epoch 8/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8772

Epoch 9/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8772

Epoch 10/63
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8772

Epoch 11/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8889

Epoch 12/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8889

Epoch 13/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8889

Epoch 14/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8889

Epoch 15/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8830

Epoch 16/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8830

Epoch 17/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 18/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8947

Epoch 19/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 20/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 21/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 22/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 23/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 24/63
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8947

Epoch 25/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 26/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8830

Epoch 27/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8889

Epoch 28/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8889

Epoch 29/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 30/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 31/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 32/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 33/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 34/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8889

Epoch 35/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8889

Epoch 36/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 37/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 38/63
----------
train Loss: 0.0004 Acc: 0.9981
val Loss: 0.0044 Acc: 0.8947

Epoch 39/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 40/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8947

Epoch 41/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 42/63
----------
train Loss: 0.0004 Acc: 0.9981
val Loss: 0.0046 Acc: 0.8947

Epoch 43/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 44/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8947

Epoch 45/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 46/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 47/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8889

Epoch 48/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 49/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 50/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8889

Epoch 51/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9006

Epoch 52/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 53/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 54/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 55/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9006

Epoch 56/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 57/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 58/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 59/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 60/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 61/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8947

Epoch 62/63
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 63/63
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9006

Training complete in 3m 19s
Best val Acc: 0.900585

---Testing---
Test accuracy: 0.975255
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 100 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9668544550057266, std: 0.030167894475109196
--------------------

run info[val: 0.3, epoch: 60, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0221 Acc: 0.2100
val Loss: 0.0257 Acc: 0.2767

Epoch 1/59
----------
train Loss: 0.0158 Acc: 0.4740
val Loss: 0.0175 Acc: 0.6553

Epoch 2/59
----------
train Loss: 0.0106 Acc: 0.6757
val Loss: 0.0128 Acc: 0.7087

Epoch 3/59
----------
train Loss: 0.0074 Acc: 0.8046
val Loss: 0.0097 Acc: 0.7767

Epoch 4/59
----------
train Loss: 0.0063 Acc: 0.8004
val Loss: 0.0109 Acc: 0.8155

Epoch 5/59
----------
LR is set to 0.001
train Loss: 0.0050 Acc: 0.8649
val Loss: 0.0088 Acc: 0.8204

Epoch 6/59
----------
train Loss: 0.0049 Acc: 0.8711
val Loss: 0.0069 Acc: 0.8155

Epoch 7/59
----------
train Loss: 0.0048 Acc: 0.8898
val Loss: 0.0085 Acc: 0.8107

Epoch 8/59
----------
train Loss: 0.0051 Acc: 0.8545
val Loss: 0.0062 Acc: 0.8155

Epoch 9/59
----------
train Loss: 0.0049 Acc: 0.8565
val Loss: 0.0093 Acc: 0.8107

Epoch 10/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0047 Acc: 0.8794
val Loss: 0.0073 Acc: 0.8107

Epoch 11/59
----------
train Loss: 0.0046 Acc: 0.8940
val Loss: 0.0069 Acc: 0.8107

Epoch 12/59
----------
train Loss: 0.0048 Acc: 0.8794
val Loss: 0.0065 Acc: 0.8107

Epoch 13/59
----------
train Loss: 0.0050 Acc: 0.8565
val Loss: 0.0080 Acc: 0.8107

Epoch 14/59
----------
train Loss: 0.0049 Acc: 0.8794
val Loss: 0.0088 Acc: 0.8107

Epoch 15/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0048 Acc: 0.8836
val Loss: 0.0077 Acc: 0.8107

Epoch 16/59
----------
train Loss: 0.0047 Acc: 0.8960
val Loss: 0.0107 Acc: 0.8107

Epoch 17/59
----------
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0121 Acc: 0.8107

Epoch 18/59
----------
train Loss: 0.0045 Acc: 0.8981
val Loss: 0.0064 Acc: 0.8107

Epoch 19/59
----------
train Loss: 0.0045 Acc: 0.8877
val Loss: 0.0060 Acc: 0.8107

Epoch 20/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0047 Acc: 0.8857
val Loss: 0.0089 Acc: 0.8107

Epoch 21/59
----------
train Loss: 0.0046 Acc: 0.8877
val Loss: 0.0082 Acc: 0.8107

Epoch 22/59
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0084 Acc: 0.8107

Epoch 23/59
----------
train Loss: 0.0046 Acc: 0.8857
val Loss: 0.0075 Acc: 0.8107

Epoch 24/59
----------
train Loss: 0.0045 Acc: 0.9023
val Loss: 0.0065 Acc: 0.8107

Epoch 25/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0044 Acc: 0.8898
val Loss: 0.0063 Acc: 0.8107

Epoch 26/59
----------
train Loss: 0.0046 Acc: 0.8836
val Loss: 0.0104 Acc: 0.8107

Epoch 27/59
----------
train Loss: 0.0046 Acc: 0.8898
val Loss: 0.0125 Acc: 0.8107

Epoch 28/59
----------
train Loss: 0.0047 Acc: 0.8836
val Loss: 0.0099 Acc: 0.8107

Epoch 29/59
----------
train Loss: 0.0046 Acc: 0.8753
val Loss: 0.0068 Acc: 0.8107

Epoch 30/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0065 Acc: 0.8107

Epoch 31/59
----------
train Loss: 0.0047 Acc: 0.8732
val Loss: 0.0068 Acc: 0.8107

Epoch 32/59
----------
train Loss: 0.0049 Acc: 0.8711
val Loss: 0.0076 Acc: 0.8107

Epoch 33/59
----------
train Loss: 0.0044 Acc: 0.8877
val Loss: 0.0067 Acc: 0.8107

Epoch 34/59
----------
train Loss: 0.0046 Acc: 0.8690
val Loss: 0.0111 Acc: 0.8107

Epoch 35/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0046 Acc: 0.8857
val Loss: 0.0085 Acc: 0.8107

Epoch 36/59
----------
train Loss: 0.0046 Acc: 0.8857
val Loss: 0.0062 Acc: 0.8107

Epoch 37/59
----------
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0074 Acc: 0.8107

Epoch 38/59
----------
train Loss: 0.0048 Acc: 0.8669
val Loss: 0.0090 Acc: 0.8155

Epoch 39/59
----------
train Loss: 0.0049 Acc: 0.8815
val Loss: 0.0068 Acc: 0.8107

Epoch 40/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0047 Acc: 0.8794
val Loss: 0.0082 Acc: 0.8107

Epoch 41/59
----------
train Loss: 0.0046 Acc: 0.8836
val Loss: 0.0080 Acc: 0.8107

Epoch 42/59
----------
train Loss: 0.0045 Acc: 0.8773
val Loss: 0.0081 Acc: 0.8107

Epoch 43/59
----------
train Loss: 0.0046 Acc: 0.8794
val Loss: 0.0089 Acc: 0.8107

Epoch 44/59
----------
train Loss: 0.0046 Acc: 0.8877
val Loss: 0.0078 Acc: 0.8107

Epoch 45/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0045 Acc: 0.8753
val Loss: 0.0067 Acc: 0.8107

Epoch 46/59
----------
train Loss: 0.0045 Acc: 0.8981
val Loss: 0.0075 Acc: 0.8107

Epoch 47/59
----------
train Loss: 0.0046 Acc: 0.8836
val Loss: 0.0104 Acc: 0.8107

Epoch 48/59
----------
train Loss: 0.0046 Acc: 0.8940
val Loss: 0.0074 Acc: 0.8107

Epoch 49/59
----------
train Loss: 0.0044 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8107

Epoch 50/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0047 Acc: 0.8711
val Loss: 0.0098 Acc: 0.8107

Epoch 51/59
----------
train Loss: 0.0045 Acc: 0.8815
val Loss: 0.0082 Acc: 0.8107

Epoch 52/59
----------
train Loss: 0.0046 Acc: 0.8981
val Loss: 0.0072 Acc: 0.8107

Epoch 53/59
----------
train Loss: 0.0045 Acc: 0.8753
val Loss: 0.0070 Acc: 0.8155

Epoch 54/59
----------
train Loss: 0.0046 Acc: 0.8960
val Loss: 0.0070 Acc: 0.8107

Epoch 55/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0046 Acc: 0.8877
val Loss: 0.0084 Acc: 0.8107

Epoch 56/59
----------
train Loss: 0.0047 Acc: 0.8794
val Loss: 0.0083 Acc: 0.8107

Epoch 57/59
----------
train Loss: 0.0046 Acc: 0.8815
val Loss: 0.0074 Acc: 0.8107

Epoch 58/59
----------
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0074 Acc: 0.8107

Epoch 59/59
----------
train Loss: 0.0048 Acc: 0.8794
val Loss: 0.0074 Acc: 0.8107

Training complete in 2m 54s
Best val Acc: 0.820388

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0050 Acc: 0.8524
val Loss: 0.0125 Acc: 0.8398

Epoch 1/59
----------
train Loss: 0.0030 Acc: 0.9356
val Loss: 0.0056 Acc: 0.8398

Epoch 2/59
----------
train Loss: 0.0017 Acc: 0.9605
val Loss: 0.0061 Acc: 0.8495

Epoch 3/59
----------
train Loss: 0.0009 Acc: 0.9938
val Loss: 0.0055 Acc: 0.8835

Epoch 4/59
----------
train Loss: 0.0008 Acc: 0.9834
val Loss: 0.0054 Acc: 0.8932

Epoch 5/59
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9917
val Loss: 0.0039 Acc: 0.8883

Epoch 6/59
----------
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0039 Acc: 0.8932

Epoch 7/59
----------
train Loss: 0.0005 Acc: 0.9917
val Loss: 0.0038 Acc: 0.8932

Epoch 8/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0055 Acc: 0.8981

Epoch 9/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9029

Epoch 10/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9029

Epoch 11/59
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0036 Acc: 0.9029

Epoch 12/59
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0037 Acc: 0.9029

Epoch 13/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9029

Epoch 14/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0057 Acc: 0.9029

Epoch 15/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0043 Acc: 0.9029

Epoch 16/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0078 Acc: 0.9029

Epoch 17/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0049 Acc: 0.9029

Epoch 18/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9029

Epoch 19/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0056 Acc: 0.9029

Epoch 20/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0042 Acc: 0.9029

Epoch 21/59
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0082 Acc: 0.9029

Epoch 22/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9029

Epoch 23/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0069 Acc: 0.9029

Epoch 24/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0036 Acc: 0.9029

Epoch 25/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0046 Acc: 0.9029

Epoch 26/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9029

Epoch 27/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0053 Acc: 0.9029

Epoch 28/59
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0053 Acc: 0.9029

Epoch 29/59
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0041 Acc: 0.9029

Epoch 30/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0084 Acc: 0.9029

Epoch 31/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9029

Epoch 32/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0038 Acc: 0.9029

Epoch 33/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0038 Acc: 0.9029

Epoch 34/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0058 Acc: 0.9029

Epoch 35/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0139 Acc: 0.9029

Epoch 36/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0048 Acc: 0.9029

Epoch 37/59
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0053 Acc: 0.9029

Epoch 38/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0063 Acc: 0.9029

Epoch 39/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9029

Epoch 40/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0052 Acc: 0.9029

Epoch 41/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9029

Epoch 42/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0053 Acc: 0.9029

Epoch 43/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0045 Acc: 0.9029

Epoch 44/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0037 Acc: 0.9029

Epoch 45/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0004 Acc: 0.9979
val Loss: 0.0038 Acc: 0.9029

Epoch 46/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9029

Epoch 47/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9029

Epoch 48/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0037 Acc: 0.9029

Epoch 49/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0060 Acc: 0.9029

Epoch 50/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0045 Acc: 0.9029

Epoch 51/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9029

Epoch 52/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9029

Epoch 53/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0047 Acc: 0.9029

Epoch 54/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0036 Acc: 0.9029

Epoch 55/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0079 Acc: 0.9029

Epoch 56/59
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0087 Acc: 0.9029

Epoch 57/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0066 Acc: 0.9029

Epoch 58/59
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0101 Acc: 0.9029

Epoch 59/59
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0053 Acc: 0.9029

Training complete in 3m 2s
Best val Acc: 0.902913

---Testing---
Test accuracy: 0.969432
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 87 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 98 %
mean: 0.960183904315395, std: 0.04423605250520585

Model saved in "./weights/shark_[0.99]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 88, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0224 Acc: 0.2310
val Loss: 0.0230 Acc: 0.4265

Epoch 1/87
----------
train Loss: 0.0133 Acc: 0.6494
val Loss: 0.0166 Acc: 0.6324

Epoch 2/87
----------
train Loss: 0.0090 Acc: 0.7447
val Loss: 0.0117 Acc: 0.6912

Epoch 3/87
----------
train Loss: 0.0061 Acc: 0.8191
val Loss: 0.0113 Acc: 0.6765

Epoch 4/87
----------
train Loss: 0.0047 Acc: 0.8578
val Loss: 0.0104 Acc: 0.7500

Epoch 5/87
----------
train Loss: 0.0041 Acc: 0.8869
val Loss: 0.0101 Acc: 0.7647

Epoch 6/87
----------
train Loss: 0.0033 Acc: 0.9128
val Loss: 0.0112 Acc: 0.7206

Epoch 7/87
----------
train Loss: 0.0037 Acc: 0.9095
val Loss: 0.0098 Acc: 0.7794

Epoch 8/87
----------
train Loss: 0.0035 Acc: 0.9192
val Loss: 0.0111 Acc: 0.7353

Epoch 9/87
----------
train Loss: 0.0031 Acc: 0.9386
val Loss: 0.0097 Acc: 0.7794

Epoch 10/87
----------
train Loss: 0.0028 Acc: 0.9467
val Loss: 0.0108 Acc: 0.7500

Epoch 11/87
----------
LR is set to 0.001
train Loss: 0.0023 Acc: 0.9628
val Loss: 0.0104 Acc: 0.7353

Epoch 12/87
----------
train Loss: 0.0021 Acc: 0.9596
val Loss: 0.0096 Acc: 0.7794

Epoch 13/87
----------
train Loss: 0.0021 Acc: 0.9515
val Loss: 0.0094 Acc: 0.7794

Epoch 14/87
----------
train Loss: 0.0021 Acc: 0.9596
val Loss: 0.0097 Acc: 0.7941

Epoch 15/87
----------
train Loss: 0.0020 Acc: 0.9677
val Loss: 0.0096 Acc: 0.7794

Epoch 16/87
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0096 Acc: 0.7794

Epoch 17/87
----------
train Loss: 0.0020 Acc: 0.9612
val Loss: 0.0094 Acc: 0.7647

Epoch 18/87
----------
train Loss: 0.0019 Acc: 0.9725
val Loss: 0.0095 Acc: 0.7647

Epoch 19/87
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0096 Acc: 0.7647

Epoch 20/87
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0095 Acc: 0.7794

Epoch 21/87
----------
train Loss: 0.0022 Acc: 0.9725
val Loss: 0.0094 Acc: 0.7647

Epoch 22/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0022 Acc: 0.9725
val Loss: 0.0096 Acc: 0.7647

Epoch 23/87
----------
train Loss: 0.0023 Acc: 0.9661
val Loss: 0.0096 Acc: 0.7647

Epoch 24/87
----------
train Loss: 0.0020 Acc: 0.9742
val Loss: 0.0096 Acc: 0.7647

Epoch 25/87
----------
train Loss: 0.0020 Acc: 0.9628
val Loss: 0.0096 Acc: 0.7794

Epoch 26/87
----------
train Loss: 0.0026 Acc: 0.9580
val Loss: 0.0097 Acc: 0.7647

Epoch 27/87
----------
train Loss: 0.0019 Acc: 0.9645
val Loss: 0.0096 Acc: 0.7794

Epoch 28/87
----------
train Loss: 0.0019 Acc: 0.9774
val Loss: 0.0095 Acc: 0.7647

Epoch 29/87
----------
train Loss: 0.0020 Acc: 0.9725
val Loss: 0.0095 Acc: 0.7794

Epoch 30/87
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0096 Acc: 0.7500

Epoch 31/87
----------
train Loss: 0.0019 Acc: 0.9758
val Loss: 0.0095 Acc: 0.7794

Epoch 32/87
----------
train Loss: 0.0019 Acc: 0.9758
val Loss: 0.0095 Acc: 0.7647

Epoch 33/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0018 Acc: 0.9742
val Loss: 0.0094 Acc: 0.7647

Epoch 34/87
----------
train Loss: 0.0022 Acc: 0.9661
val Loss: 0.0093 Acc: 0.7794

Epoch 35/87
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0094 Acc: 0.7794

Epoch 36/87
----------
train Loss: 0.0018 Acc: 0.9725
val Loss: 0.0093 Acc: 0.7647

Epoch 37/87
----------
train Loss: 0.0018 Acc: 0.9790
val Loss: 0.0094 Acc: 0.7647

Epoch 38/87
----------
train Loss: 0.0020 Acc: 0.9742
val Loss: 0.0094 Acc: 0.7647

Epoch 39/87
----------
train Loss: 0.0021 Acc: 0.9725
val Loss: 0.0093 Acc: 0.7647

Epoch 40/87
----------
train Loss: 0.0020 Acc: 0.9693
val Loss: 0.0094 Acc: 0.7647

Epoch 41/87
----------
train Loss: 0.0022 Acc: 0.9677
val Loss: 0.0094 Acc: 0.7647

Epoch 42/87
----------
train Loss: 0.0020 Acc: 0.9790
val Loss: 0.0094 Acc: 0.7647

Epoch 43/87
----------
train Loss: 0.0024 Acc: 0.9661
val Loss: 0.0096 Acc: 0.7794

Epoch 44/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0018 Acc: 0.9758
val Loss: 0.0095 Acc: 0.7794

Epoch 45/87
----------
train Loss: 0.0018 Acc: 0.9822
val Loss: 0.0094 Acc: 0.7794

Epoch 46/87
----------
train Loss: 0.0019 Acc: 0.9693
val Loss: 0.0095 Acc: 0.7500

Epoch 47/87
----------
train Loss: 0.0020 Acc: 0.9774
val Loss: 0.0096 Acc: 0.7500

Epoch 48/87
----------
train Loss: 0.0018 Acc: 0.9758
val Loss: 0.0094 Acc: 0.7647

Epoch 49/87
----------
train Loss: 0.0019 Acc: 0.9725
val Loss: 0.0095 Acc: 0.7647

Epoch 50/87
----------
train Loss: 0.0020 Acc: 0.9758
val Loss: 0.0094 Acc: 0.7647

Epoch 51/87
----------
train Loss: 0.0022 Acc: 0.9709
val Loss: 0.0094 Acc: 0.7647

Epoch 52/87
----------
train Loss: 0.0019 Acc: 0.9645
val Loss: 0.0095 Acc: 0.7647

Epoch 53/87
----------
train Loss: 0.0019 Acc: 0.9725
val Loss: 0.0093 Acc: 0.7647

Epoch 54/87
----------
train Loss: 0.0021 Acc: 0.9693
val Loss: 0.0094 Acc: 0.7647

Epoch 55/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0019 Acc: 0.9725
val Loss: 0.0095 Acc: 0.7647

Epoch 56/87
----------
train Loss: 0.0020 Acc: 0.9693
val Loss: 0.0096 Acc: 0.7647

Epoch 57/87
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0095 Acc: 0.7647

Epoch 58/87
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0095 Acc: 0.7500

Epoch 59/87
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0097 Acc: 0.7500

Epoch 60/87
----------
train Loss: 0.0021 Acc: 0.9742
val Loss: 0.0097 Acc: 0.7647

Epoch 61/87
----------
train Loss: 0.0022 Acc: 0.9742
val Loss: 0.0095 Acc: 0.7500

Epoch 62/87
----------
train Loss: 0.0019 Acc: 0.9742
val Loss: 0.0095 Acc: 0.7794

Epoch 63/87
----------
train Loss: 0.0019 Acc: 0.9774
val Loss: 0.0094 Acc: 0.7794

Epoch 64/87
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0094 Acc: 0.7794

Epoch 65/87
----------
train Loss: 0.0020 Acc: 0.9693
val Loss: 0.0093 Acc: 0.7794

Epoch 66/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0018 Acc: 0.9790
val Loss: 0.0095 Acc: 0.7794

Epoch 67/87
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0097 Acc: 0.7500

Epoch 68/87
----------
train Loss: 0.0020 Acc: 0.9693
val Loss: 0.0097 Acc: 0.7500

Epoch 69/87
----------
train Loss: 0.0018 Acc: 0.9742
val Loss: 0.0094 Acc: 0.7794

Epoch 70/87
----------
train Loss: 0.0020 Acc: 0.9693
val Loss: 0.0093 Acc: 0.7794

Epoch 71/87
----------
train Loss: 0.0022 Acc: 0.9661
val Loss: 0.0094 Acc: 0.7647

Epoch 72/87
----------
train Loss: 0.0022 Acc: 0.9645
val Loss: 0.0093 Acc: 0.7647

Epoch 73/87
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0094 Acc: 0.7794

Epoch 74/87
----------
train Loss: 0.0021 Acc: 0.9693
val Loss: 0.0094 Acc: 0.7647

Epoch 75/87
----------
train Loss: 0.0023 Acc: 0.9645
val Loss: 0.0095 Acc: 0.7500

Epoch 76/87
----------
train Loss: 0.0022 Acc: 0.9628
val Loss: 0.0095 Acc: 0.7794

Epoch 77/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0095 Acc: 0.7794

Epoch 78/87
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0095 Acc: 0.7647

Epoch 79/87
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0095 Acc: 0.7647

Epoch 80/87
----------
train Loss: 0.0018 Acc: 0.9725
val Loss: 0.0095 Acc: 0.7794

Epoch 81/87
----------
train Loss: 0.0020 Acc: 0.9661
val Loss: 0.0095 Acc: 0.7647

Epoch 82/87
----------
train Loss: 0.0021 Acc: 0.9725
val Loss: 0.0096 Acc: 0.7500

Epoch 83/87
----------
train Loss: 0.0022 Acc: 0.9758
val Loss: 0.0096 Acc: 0.7647

Epoch 84/87
----------
train Loss: 0.0021 Acc: 0.9790
val Loss: 0.0096 Acc: 0.7647

Epoch 85/87
----------
train Loss: 0.0020 Acc: 0.9725
val Loss: 0.0096 Acc: 0.7647

Epoch 86/87
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0096 Acc: 0.7500

Epoch 87/87
----------
train Loss: 0.0020 Acc: 0.9758
val Loss: 0.0097 Acc: 0.7647

Training complete in 4m 16s
Best val Acc: 0.794118

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0020 Acc: 0.9548
val Loss: 0.0098 Acc: 0.7794

Epoch 1/87
----------
train Loss: 0.0011 Acc: 0.9887
val Loss: 0.0098 Acc: 0.8088

Epoch 2/87
----------
train Loss: 0.0006 Acc: 0.9952
val Loss: 0.0089 Acc: 0.7941

Epoch 3/87
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8088

Epoch 4/87
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8088

Epoch 5/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8235

Epoch 6/87
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0071 Acc: 0.8529

Epoch 7/87
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0089 Acc: 0.8235

Epoch 8/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8235

Epoch 9/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 10/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8235

Epoch 11/87
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8235

Epoch 12/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 13/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 14/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 15/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 16/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 17/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 18/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 19/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 20/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 21/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 22/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 23/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 24/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 25/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8088

Epoch 26/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8088

Epoch 27/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 28/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 29/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 30/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 31/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 32/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 33/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 34/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8088

Epoch 35/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8088

Epoch 36/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 37/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 38/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 39/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 40/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 41/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 42/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 43/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 44/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 45/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8088

Epoch 46/87
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0087 Acc: 0.8088

Epoch 47/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 48/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8088

Epoch 49/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 50/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8088

Epoch 51/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8088

Epoch 52/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 53/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 54/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 55/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 56/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 57/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 58/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 59/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 60/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 61/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 62/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 63/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 64/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8088

Epoch 65/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 66/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 67/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 68/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 69/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 70/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 71/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8088

Epoch 72/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 73/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 74/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 75/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 76/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 77/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 78/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 79/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 80/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 81/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 82/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 83/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Epoch 84/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 85/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 86/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8088

Epoch 87/87
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8088

Training complete in 4m 34s
Best val Acc: 0.852941

---Testing---
Test accuracy: 0.985444
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 98 %
Accuracy of Squatiniformes : 97 %
mean: 0.9804183754841649, std: 0.01773543483018401
--------------------

run info[val: 0.15, epoch: 56, randcrop: True, decay: 14]

---Training last layer.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0198 Acc: 0.2603
val Loss: 0.0366 Acc: 0.3689

Epoch 1/55
----------
train Loss: 0.0139 Acc: 0.5291
val Loss: 0.0319 Acc: 0.6214

Epoch 2/55
----------
train Loss: 0.0089 Acc: 0.7243
val Loss: 0.0127 Acc: 0.6602

Epoch 3/55
----------
train Loss: 0.0069 Acc: 0.7928
val Loss: 0.0152 Acc: 0.7670

Epoch 4/55
----------
train Loss: 0.0050 Acc: 0.8493
val Loss: 0.0154 Acc: 0.7961

Epoch 5/55
----------
train Loss: 0.0046 Acc: 0.8767
val Loss: 0.0102 Acc: 0.7864

Epoch 6/55
----------
train Loss: 0.0042 Acc: 0.8647
val Loss: 0.0082 Acc: 0.8058

Epoch 7/55
----------
train Loss: 0.0037 Acc: 0.8938
val Loss: 0.0179 Acc: 0.8058

Epoch 8/55
----------
train Loss: 0.0032 Acc: 0.9161
val Loss: 0.0142 Acc: 0.8155

Epoch 9/55
----------
train Loss: 0.0030 Acc: 0.9298
val Loss: 0.0076 Acc: 0.8155

Epoch 10/55
----------
train Loss: 0.0030 Acc: 0.9247
val Loss: 0.0068 Acc: 0.8155

Epoch 11/55
----------
train Loss: 0.0027 Acc: 0.9452
val Loss: 0.0204 Acc: 0.7961

Epoch 12/55
----------
train Loss: 0.0026 Acc: 0.9298
val Loss: 0.0282 Acc: 0.8155

Epoch 13/55
----------
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0142 Acc: 0.7961

Epoch 14/55
----------
LR is set to 0.001
train Loss: 0.0025 Acc: 0.9418
val Loss: 0.0065 Acc: 0.7961

Epoch 15/55
----------
train Loss: 0.0020 Acc: 0.9640
val Loss: 0.0065 Acc: 0.7961

Epoch 16/55
----------
train Loss: 0.0022 Acc: 0.9521
val Loss: 0.0080 Acc: 0.7961

Epoch 17/55
----------
train Loss: 0.0022 Acc: 0.9503
val Loss: 0.0103 Acc: 0.7961

Epoch 18/55
----------
train Loss: 0.0022 Acc: 0.9521
val Loss: 0.0065 Acc: 0.7961

Epoch 19/55
----------
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0160 Acc: 0.7961

Epoch 20/55
----------
train Loss: 0.0021 Acc: 0.9521
val Loss: 0.0078 Acc: 0.7961

Epoch 21/55
----------
train Loss: 0.0022 Acc: 0.9452
val Loss: 0.0063 Acc: 0.8058

Epoch 22/55
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0086 Acc: 0.8058

Epoch 23/55
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0123 Acc: 0.8058

Epoch 24/55
----------
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0079 Acc: 0.7961

Epoch 25/55
----------
train Loss: 0.0022 Acc: 0.9521
val Loss: 0.0077 Acc: 0.7961

Epoch 26/55
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0105 Acc: 0.7961

Epoch 27/55
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0133 Acc: 0.7961

Epoch 28/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0226 Acc: 0.7961

Epoch 29/55
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0147 Acc: 0.7961

Epoch 30/55
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0067 Acc: 0.7961

Epoch 31/55
----------
train Loss: 0.0020 Acc: 0.9640
val Loss: 0.0188 Acc: 0.7961

Epoch 32/55
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0180 Acc: 0.7961

Epoch 33/55
----------
train Loss: 0.0021 Acc: 0.9486
val Loss: 0.0132 Acc: 0.8058

Epoch 34/55
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0070 Acc: 0.8058

Epoch 35/55
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0081 Acc: 0.8058

Epoch 36/55
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0240 Acc: 0.8058

Epoch 37/55
----------
train Loss: 0.0021 Acc: 0.9658
val Loss: 0.0157 Acc: 0.8058

Epoch 38/55
----------
train Loss: 0.0021 Acc: 0.9555
val Loss: 0.0129 Acc: 0.8058

Epoch 39/55
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0070 Acc: 0.8058

Epoch 40/55
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0071 Acc: 0.8058

Epoch 41/55
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0062 Acc: 0.8058

Epoch 42/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0021 Acc: 0.9521
val Loss: 0.0066 Acc: 0.8058

Epoch 43/55
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0073 Acc: 0.8058

Epoch 44/55
----------
train Loss: 0.0022 Acc: 0.9469
val Loss: 0.0150 Acc: 0.7961

Epoch 45/55
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0203 Acc: 0.8058

Epoch 46/55
----------
train Loss: 0.0022 Acc: 0.9538
val Loss: 0.0065 Acc: 0.8058

Epoch 47/55
----------
train Loss: 0.0021 Acc: 0.9538
val Loss: 0.0140 Acc: 0.8058

Epoch 48/55
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0067 Acc: 0.8058

Epoch 49/55
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0322 Acc: 0.7961

Epoch 50/55
----------
train Loss: 0.0023 Acc: 0.9503
val Loss: 0.0063 Acc: 0.8058

Epoch 51/55
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0062 Acc: 0.8058

Epoch 52/55
----------
train Loss: 0.0023 Acc: 0.9469
val Loss: 0.0095 Acc: 0.8058

Epoch 53/55
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0088 Acc: 0.8058

Epoch 54/55
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0172 Acc: 0.8058

Epoch 55/55
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0070 Acc: 0.7961

Training complete in 2m 50s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0027 Acc: 0.9315
val Loss: 0.0070 Acc: 0.8155

Epoch 1/55
----------
train Loss: 0.0016 Acc: 0.9675
val Loss: 0.0072 Acc: 0.8350

Epoch 2/55
----------
train Loss: 0.0011 Acc: 0.9777
val Loss: 0.0087 Acc: 0.8350

Epoch 3/55
----------
train Loss: 0.0006 Acc: 0.9863
val Loss: 0.0063 Acc: 0.8155

Epoch 4/55
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0234 Acc: 0.8252

Epoch 5/55
----------
train Loss: 0.0003 Acc: 0.9983
val Loss: 0.0068 Acc: 0.8350

Epoch 6/55
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0050 Acc: 0.8447

Epoch 7/55
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0193 Acc: 0.8447

Epoch 8/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8447

Epoch 9/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0172 Acc: 0.8544

Epoch 10/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 11/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0150 Acc: 0.8447

Epoch 12/55
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0055 Acc: 0.8544

Epoch 13/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0270 Acc: 0.8447

Epoch 14/55
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8350

Epoch 15/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0135 Acc: 0.8350

Epoch 16/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8350

Epoch 17/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8350

Epoch 18/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8350

Epoch 19/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 20/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8447

Epoch 21/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0266 Acc: 0.8447

Epoch 22/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8350

Epoch 23/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 24/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 25/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0152 Acc: 0.8350

Epoch 26/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8350

Epoch 27/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8350

Epoch 28/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0108 Acc: 0.8350

Epoch 29/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0244 Acc: 0.8350

Epoch 30/55
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8447

Epoch 31/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 32/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 33/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0108 Acc: 0.8447

Epoch 34/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8350

Epoch 35/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0254 Acc: 0.8350

Epoch 36/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 37/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 38/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 39/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 40/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8350

Epoch 41/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8350

Epoch 42/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0106 Acc: 0.8447

Epoch 43/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8350

Epoch 44/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 45/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 46/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8350

Epoch 47/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8447

Epoch 48/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 49/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0262 Acc: 0.8447

Epoch 50/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8447

Epoch 51/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8447

Epoch 52/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0323 Acc: 0.8447

Epoch 53/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8447

Epoch 54/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8447

Epoch 55/55
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8447

Training complete in 2m 59s
Best val Acc: 0.854369

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 96 %
mean: 0.9696827295511505, std: 0.029644403994808156
--------------------

run info[val: 0.2, epoch: 99, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0219 Acc: 0.2400
val Loss: 0.0282 Acc: 0.2482

Epoch 1/98
----------
train Loss: 0.0147 Acc: 0.5509
val Loss: 0.0173 Acc: 0.5328

Epoch 2/98
----------
train Loss: 0.0090 Acc: 0.7345
val Loss: 0.0122 Acc: 0.7007

Epoch 3/98
----------
train Loss: 0.0065 Acc: 0.8200
val Loss: 0.0104 Acc: 0.7372

Epoch 4/98
----------
train Loss: 0.0052 Acc: 0.8345
val Loss: 0.0088 Acc: 0.7737

Epoch 5/98
----------
train Loss: 0.0044 Acc: 0.8982
val Loss: 0.0088 Acc: 0.7664

Epoch 6/98
----------
train Loss: 0.0038 Acc: 0.8782
val Loss: 0.0089 Acc: 0.8102

Epoch 7/98
----------
train Loss: 0.0035 Acc: 0.9182
val Loss: 0.0082 Acc: 0.8102

Epoch 8/98
----------
train Loss: 0.0031 Acc: 0.9073
val Loss: 0.0092 Acc: 0.8102

Epoch 9/98
----------
train Loss: 0.0027 Acc: 0.9436
val Loss: 0.0082 Acc: 0.8029

Epoch 10/98
----------
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0086 Acc: 0.8029

Epoch 11/98
----------
train Loss: 0.0021 Acc: 0.9636
val Loss: 0.0086 Acc: 0.8175

Epoch 12/98
----------
train Loss: 0.0020 Acc: 0.9673
val Loss: 0.0075 Acc: 0.8029

Epoch 13/98
----------
LR is set to 0.001
train Loss: 0.0017 Acc: 0.9709
val Loss: 0.0075 Acc: 0.8029

Epoch 14/98
----------
train Loss: 0.0019 Acc: 0.9691
val Loss: 0.0083 Acc: 0.8029

Epoch 15/98
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0078 Acc: 0.8029

Epoch 16/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0072 Acc: 0.8102

Epoch 17/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0085 Acc: 0.8029

Epoch 18/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0077 Acc: 0.8102

Epoch 19/98
----------
train Loss: 0.0018 Acc: 0.9691
val Loss: 0.0072 Acc: 0.8029

Epoch 20/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0070 Acc: 0.8029

Epoch 21/98
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0078 Acc: 0.8029

Epoch 22/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0071 Acc: 0.7956

Epoch 23/98
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0068 Acc: 0.8029

Epoch 24/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0075 Acc: 0.8029

Epoch 25/98
----------
train Loss: 0.0018 Acc: 0.9836
val Loss: 0.0067 Acc: 0.8029

Epoch 26/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0073 Acc: 0.8029

Epoch 27/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0071 Acc: 0.8029

Epoch 28/98
----------
train Loss: 0.0016 Acc: 0.9800
val Loss: 0.0066 Acc: 0.8029

Epoch 29/98
----------
train Loss: 0.0018 Acc: 0.9764
val Loss: 0.0079 Acc: 0.7956

Epoch 30/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0061 Acc: 0.7956

Epoch 31/98
----------
train Loss: 0.0017 Acc: 0.9764
val Loss: 0.0066 Acc: 0.8102

Epoch 32/98
----------
train Loss: 0.0017 Acc: 0.9709
val Loss: 0.0062 Acc: 0.8102

Epoch 33/98
----------
train Loss: 0.0018 Acc: 0.9782
val Loss: 0.0074 Acc: 0.8102

Epoch 34/98
----------
train Loss: 0.0017 Acc: 0.9800
val Loss: 0.0068 Acc: 0.8029

Epoch 35/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0076 Acc: 0.8029

Epoch 36/98
----------
train Loss: 0.0017 Acc: 0.9727
val Loss: 0.0070 Acc: 0.8029

Epoch 37/98
----------
train Loss: 0.0018 Acc: 0.9691
val Loss: 0.0070 Acc: 0.8029

Epoch 38/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0080 Acc: 0.7956

Epoch 39/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0078 Acc: 0.7956

Epoch 40/98
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0074 Acc: 0.8029

Epoch 41/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0085 Acc: 0.8029

Epoch 42/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0081 Acc: 0.8029

Epoch 43/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0076 Acc: 0.8029

Epoch 44/98
----------
train Loss: 0.0016 Acc: 0.9782
val Loss: 0.0069 Acc: 0.8029

Epoch 45/98
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0065 Acc: 0.8029

Epoch 46/98
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0069 Acc: 0.8029

Epoch 47/98
----------
train Loss: 0.0018 Acc: 0.9800
val Loss: 0.0068 Acc: 0.8029

Epoch 48/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0068 Acc: 0.8029

Epoch 49/98
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0088 Acc: 0.8029

Epoch 50/98
----------
train Loss: 0.0018 Acc: 0.9691
val Loss: 0.0070 Acc: 0.8029

Epoch 51/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0076 Acc: 0.8029

Epoch 52/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0079 Acc: 0.8029

Epoch 53/98
----------
train Loss: 0.0016 Acc: 0.9727
val Loss: 0.0075 Acc: 0.8029

Epoch 54/98
----------
train Loss: 0.0016 Acc: 0.9745
val Loss: 0.0065 Acc: 0.8029

Epoch 55/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0077 Acc: 0.8029

Epoch 56/98
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0077 Acc: 0.8029

Epoch 57/98
----------
train Loss: 0.0017 Acc: 0.9764
val Loss: 0.0074 Acc: 0.8029

Epoch 58/98
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0064 Acc: 0.8029

Epoch 59/98
----------
train Loss: 0.0017 Acc: 0.9855
val Loss: 0.0081 Acc: 0.8029

Epoch 60/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0080 Acc: 0.8029

Epoch 61/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0071 Acc: 0.8029

Epoch 62/98
----------
train Loss: 0.0017 Acc: 0.9764
val Loss: 0.0073 Acc: 0.8029

Epoch 63/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0081 Acc: 0.8029

Epoch 64/98
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0070 Acc: 0.8029

Epoch 65/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0018 Acc: 0.9818
val Loss: 0.0065 Acc: 0.8029

Epoch 66/98
----------
train Loss: 0.0018 Acc: 0.9764
val Loss: 0.0072 Acc: 0.8029

Epoch 67/98
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0089 Acc: 0.8029

Epoch 68/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0072 Acc: 0.8102

Epoch 69/98
----------
train Loss: 0.0016 Acc: 0.9764
val Loss: 0.0088 Acc: 0.8102

Epoch 70/98
----------
train Loss: 0.0018 Acc: 0.9764
val Loss: 0.0079 Acc: 0.8102

Epoch 71/98
----------
train Loss: 0.0017 Acc: 0.9800
val Loss: 0.0068 Acc: 0.8102

Epoch 72/98
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0071 Acc: 0.8029

Epoch 73/98
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0077 Acc: 0.8029

Epoch 74/98
----------
train Loss: 0.0019 Acc: 0.9727
val Loss: 0.0077 Acc: 0.8029

Epoch 75/98
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0070 Acc: 0.8029

Epoch 76/98
----------
train Loss: 0.0018 Acc: 0.9800
val Loss: 0.0081 Acc: 0.8029

Epoch 77/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0068 Acc: 0.8029

Epoch 78/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0082 Acc: 0.8029

Epoch 79/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0074 Acc: 0.8029

Epoch 80/98
----------
train Loss: 0.0018 Acc: 0.9745
val Loss: 0.0077 Acc: 0.8029

Epoch 81/98
----------
train Loss: 0.0019 Acc: 0.9764
val Loss: 0.0069 Acc: 0.8029

Epoch 82/98
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0074 Acc: 0.8029

Epoch 83/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0067 Acc: 0.8029

Epoch 84/98
----------
train Loss: 0.0016 Acc: 0.9800
val Loss: 0.0067 Acc: 0.8029

Epoch 85/98
----------
train Loss: 0.0019 Acc: 0.9745
val Loss: 0.0072 Acc: 0.8029

Epoch 86/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0081 Acc: 0.8029

Epoch 87/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0073 Acc: 0.8029

Epoch 88/98
----------
train Loss: 0.0017 Acc: 0.9727
val Loss: 0.0069 Acc: 0.8029

Epoch 89/98
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0071 Acc: 0.8029

Epoch 90/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0070 Acc: 0.8029

Epoch 91/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0018 Acc: 0.9800
val Loss: 0.0068 Acc: 0.8029

Epoch 92/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0075 Acc: 0.8029

Epoch 93/98
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0065 Acc: 0.8029

Epoch 94/98
----------
train Loss: 0.0018 Acc: 0.9818
val Loss: 0.0075 Acc: 0.8029

Epoch 95/98
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0064 Acc: 0.8029

Epoch 96/98
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0078 Acc: 0.8029

Epoch 97/98
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0075 Acc: 0.8029

Epoch 98/98
----------
train Loss: 0.0019 Acc: 0.9764
val Loss: 0.0078 Acc: 0.8029

Training complete in 4m 54s
Best val Acc: 0.817518

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0020 Acc: 0.9655
val Loss: 0.0080 Acc: 0.8321

Epoch 1/98
----------
train Loss: 0.0009 Acc: 0.9909
val Loss: 0.0068 Acc: 0.8686

Epoch 2/98
----------
train Loss: 0.0004 Acc: 0.9982
val Loss: 0.0053 Acc: 0.8540

Epoch 3/98
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8467

Epoch 4/98
----------
train Loss: 0.0002 Acc: 0.9964
val Loss: 0.0042 Acc: 0.8540

Epoch 5/98
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0054 Acc: 0.8613

Epoch 6/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8686

Epoch 7/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8686

Epoch 8/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 9/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 10/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8832

Epoch 11/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8832

Epoch 12/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8832

Epoch 13/98
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 14/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8759

Epoch 15/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 16/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8759

Epoch 17/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 18/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 19/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 20/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8759

Epoch 21/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 22/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8759

Epoch 23/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 24/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 25/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 26/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8759

Epoch 27/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 28/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 29/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 30/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 31/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 32/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 33/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8759

Epoch 34/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 35/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 36/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8759

Epoch 37/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 38/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 39/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 40/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 41/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 42/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8686

Epoch 43/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8686

Epoch 44/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8686

Epoch 45/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 46/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 47/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 48/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 49/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8759

Epoch 50/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 51/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8759

Epoch 52/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 53/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 54/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 55/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 56/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 57/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 58/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8759

Epoch 59/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8759

Epoch 60/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8759

Epoch 61/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 62/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8759

Epoch 63/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 64/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 65/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 66/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8759

Epoch 67/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 68/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 69/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 70/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 71/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8686

Epoch 72/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 73/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 74/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8759

Epoch 75/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 76/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8759

Epoch 77/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 78/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 79/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 80/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 81/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8759

Epoch 82/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 83/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 84/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 85/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 86/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 87/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 88/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 89/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8759

Epoch 90/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 91/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8759

Epoch 92/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8759

Epoch 93/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 94/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 95/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8759

Epoch 96/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 97/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 98/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8759

Training complete in 5m 37s
Best val Acc: 0.883212

---Testing---
Test accuracy: 0.976710
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9669415014809752, std: 0.03481997850270461
--------------------

run info[val: 0.25, epoch: 51, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/50
----------
LR is set to 0.01
train Loss: 0.0233 Acc: 0.2539
val Loss: 0.0198 Acc: 0.3509

Epoch 1/50
----------
train Loss: 0.0160 Acc: 0.5930
val Loss: 0.0129 Acc: 0.6199

Epoch 2/50
----------
train Loss: 0.0095 Acc: 0.7907
val Loss: 0.0090 Acc: 0.7778

Epoch 3/50
----------
train Loss: 0.0071 Acc: 0.8333
val Loss: 0.0084 Acc: 0.8012

Epoch 4/50
----------
train Loss: 0.0057 Acc: 0.8663
val Loss: 0.0077 Acc: 0.8012

Epoch 5/50
----------
train Loss: 0.0061 Acc: 0.8488
val Loss: 0.0069 Acc: 0.8304

Epoch 6/50
----------
LR is set to 0.001
train Loss: 0.0047 Acc: 0.9031
val Loss: 0.0067 Acc: 0.8655

Epoch 7/50
----------
train Loss: 0.0042 Acc: 0.9050
val Loss: 0.0065 Acc: 0.8363

Epoch 8/50
----------
train Loss: 0.0037 Acc: 0.9089
val Loss: 0.0066 Acc: 0.8304

Epoch 9/50
----------
train Loss: 0.0040 Acc: 0.9089
val Loss: 0.0066 Acc: 0.8304

Epoch 10/50
----------
train Loss: 0.0039 Acc: 0.9050
val Loss: 0.0065 Acc: 0.8246

Epoch 11/50
----------
train Loss: 0.0044 Acc: 0.9109
val Loss: 0.0067 Acc: 0.8304

Epoch 12/50
----------
LR is set to 0.00010000000000000002
train Loss: 0.0045 Acc: 0.9147
val Loss: 0.0069 Acc: 0.8363

Epoch 13/50
----------
train Loss: 0.0038 Acc: 0.9070
val Loss: 0.0065 Acc: 0.8304

Epoch 14/50
----------
train Loss: 0.0037 Acc: 0.9089
val Loss: 0.0067 Acc: 0.8246

Epoch 15/50
----------
train Loss: 0.0041 Acc: 0.9089
val Loss: 0.0069 Acc: 0.8421

Epoch 16/50
----------
train Loss: 0.0038 Acc: 0.9109
val Loss: 0.0064 Acc: 0.8480

Epoch 17/50
----------
train Loss: 0.0037 Acc: 0.9147
val Loss: 0.0064 Acc: 0.8538

Epoch 18/50
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0040 Acc: 0.9070
val Loss: 0.0065 Acc: 0.8538

Epoch 19/50
----------
train Loss: 0.0038 Acc: 0.9167
val Loss: 0.0064 Acc: 0.8480

Epoch 20/50
----------
train Loss: 0.0038 Acc: 0.9070
val Loss: 0.0067 Acc: 0.8538

Epoch 21/50
----------
train Loss: 0.0043 Acc: 0.9031
val Loss: 0.0068 Acc: 0.8421

Epoch 22/50
----------
train Loss: 0.0037 Acc: 0.9147
val Loss: 0.0067 Acc: 0.8304

Epoch 23/50
----------
train Loss: 0.0043 Acc: 0.9128
val Loss: 0.0067 Acc: 0.8421

Epoch 24/50
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0040 Acc: 0.9050
val Loss: 0.0068 Acc: 0.8421

Epoch 25/50
----------
train Loss: 0.0039 Acc: 0.9205
val Loss: 0.0068 Acc: 0.8480

Epoch 26/50
----------
train Loss: 0.0043 Acc: 0.9109
val Loss: 0.0066 Acc: 0.8480

Epoch 27/50
----------
train Loss: 0.0041 Acc: 0.9205
val Loss: 0.0068 Acc: 0.8538

Epoch 28/50
----------
train Loss: 0.0037 Acc: 0.9341
val Loss: 0.0066 Acc: 0.8480

Epoch 29/50
----------
train Loss: 0.0040 Acc: 0.9128
val Loss: 0.0066 Acc: 0.8538

Epoch 30/50
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0044 Acc: 0.9167
val Loss: 0.0067 Acc: 0.8538

Epoch 31/50
----------
train Loss: 0.0038 Acc: 0.9109
val Loss: 0.0066 Acc: 0.8480

Epoch 32/50
----------
train Loss: 0.0043 Acc: 0.8953
val Loss: 0.0066 Acc: 0.8363

Epoch 33/50
----------
train Loss: 0.0042 Acc: 0.9050
val Loss: 0.0066 Acc: 0.8480

Epoch 34/50
----------
train Loss: 0.0042 Acc: 0.9128
val Loss: 0.0068 Acc: 0.8480

Epoch 35/50
----------
train Loss: 0.0038 Acc: 0.9225
val Loss: 0.0066 Acc: 0.8480

Epoch 36/50
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0039 Acc: 0.9089
val Loss: 0.0064 Acc: 0.8421

Epoch 37/50
----------
train Loss: 0.0039 Acc: 0.8953
val Loss: 0.0066 Acc: 0.8421

Epoch 38/50
----------
train Loss: 0.0046 Acc: 0.9070
val Loss: 0.0065 Acc: 0.8421

Epoch 39/50
----------
train Loss: 0.0041 Acc: 0.9050
val Loss: 0.0065 Acc: 0.8421

Epoch 40/50
----------
train Loss: 0.0039 Acc: 0.9089
val Loss: 0.0064 Acc: 0.8363

Epoch 41/50
----------
train Loss: 0.0042 Acc: 0.9089
val Loss: 0.0065 Acc: 0.8480

Epoch 42/50
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0040 Acc: 0.9012
val Loss: 0.0066 Acc: 0.8421

Epoch 43/50
----------
train Loss: 0.0043 Acc: 0.9031
val Loss: 0.0069 Acc: 0.8421

Epoch 44/50
----------
train Loss: 0.0046 Acc: 0.9070
val Loss: 0.0067 Acc: 0.8363

Epoch 45/50
----------
train Loss: 0.0043 Acc: 0.9167
val Loss: 0.0064 Acc: 0.8304

Epoch 46/50
----------
train Loss: 0.0045 Acc: 0.9012
val Loss: 0.0065 Acc: 0.8421

Epoch 47/50
----------
train Loss: 0.0045 Acc: 0.9186
val Loss: 0.0066 Acc: 0.8363

Epoch 48/50
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0041 Acc: 0.9186
val Loss: 0.0066 Acc: 0.8480

Epoch 49/50
----------
train Loss: 0.0039 Acc: 0.9128
val Loss: 0.0064 Acc: 0.8421

Epoch 50/50
----------
train Loss: 0.0041 Acc: 0.9070
val Loss: 0.0067 Acc: 0.8480

Training complete in 2m 38s
Best val Acc: 0.865497

---Fine tuning.---
Epoch 0/50
----------
LR is set to 0.01
train Loss: 0.0043 Acc: 0.9070
val Loss: 0.0065 Acc: 0.8480

Epoch 1/50
----------
train Loss: 0.0026 Acc: 0.9496
val Loss: 0.0055 Acc: 0.8538

Epoch 2/50
----------
train Loss: 0.0018 Acc: 0.9690
val Loss: 0.0060 Acc: 0.8538

Epoch 3/50
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0058 Acc: 0.8421

Epoch 4/50
----------
train Loss: 0.0013 Acc: 0.9884
val Loss: 0.0065 Acc: 0.8480

Epoch 5/50
----------
train Loss: 0.0009 Acc: 0.9826
val Loss: 0.0075 Acc: 0.8480

Epoch 6/50
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9903
val Loss: 0.0061 Acc: 0.8655

Epoch 7/50
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0057 Acc: 0.8772

Epoch 8/50
----------
train Loss: 0.0010 Acc: 0.9961
val Loss: 0.0061 Acc: 0.8772

Epoch 9/50
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0064 Acc: 0.8655

Epoch 10/50
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0059 Acc: 0.8596

Epoch 11/50
----------
train Loss: 0.0004 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8655

Epoch 12/50
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8655

Epoch 13/50
----------
train Loss: 0.0005 Acc: 0.9922
val Loss: 0.0062 Acc: 0.8713

Epoch 14/50
----------
train Loss: 0.0004 Acc: 0.9942
val Loss: 0.0056 Acc: 0.8713

Epoch 15/50
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0060 Acc: 0.8655

Epoch 16/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8713

Epoch 17/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8713

Epoch 18/50
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0054 Acc: 0.8713

Epoch 19/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8713

Epoch 20/50
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8713

Epoch 21/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8713

Epoch 22/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8713

Epoch 23/50
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0057 Acc: 0.8713

Epoch 24/50
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0057 Acc: 0.8713

Epoch 25/50
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0060 Acc: 0.8713

Epoch 26/50
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8713

Epoch 27/50
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0058 Acc: 0.8772

Epoch 28/50
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0059 Acc: 0.8772

Epoch 29/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8772

Epoch 30/50
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8772

Epoch 31/50
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0054 Acc: 0.8713

Epoch 32/50
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0059 Acc: 0.8772

Epoch 33/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 34/50
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0053 Acc: 0.8772

Epoch 35/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8772

Epoch 36/50
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8772

Epoch 37/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8772

Epoch 38/50
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8772

Epoch 39/50
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8713

Epoch 40/50
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0057 Acc: 0.8713

Epoch 41/50
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0057 Acc: 0.8772

Epoch 42/50
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8772

Epoch 43/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0060 Acc: 0.8772

Epoch 44/50
----------
train Loss: 0.0002 Acc: 0.9961
val Loss: 0.0057 Acc: 0.8772

Epoch 45/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0058 Acc: 0.8713

Epoch 46/50
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0053 Acc: 0.8772

Epoch 47/50
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0054 Acc: 0.8772

Epoch 48/50
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0054 Acc: 0.8772

Epoch 49/50
----------
train Loss: 0.0009 Acc: 0.9961
val Loss: 0.0058 Acc: 0.8772

Epoch 50/50
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8713

Training complete in 2m 48s
Best val Acc: 0.877193

---Testing---
Test accuracy: 0.966521
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 98 %
mean: 0.957480943053179, std: 0.03607728459914648
--------------------

run info[val: 0.3, epoch: 80, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0214 Acc: 0.2183
val Loss: 0.0272 Acc: 0.3252

Epoch 1/79
----------
train Loss: 0.0156 Acc: 0.4990
val Loss: 0.0156 Acc: 0.7039

Epoch 2/79
----------
train Loss: 0.0106 Acc: 0.7027
val Loss: 0.0141 Acc: 0.6845

Epoch 3/79
----------
train Loss: 0.0080 Acc: 0.7838
val Loss: 0.0077 Acc: 0.7961

Epoch 4/79
----------
train Loss: 0.0062 Acc: 0.8067
val Loss: 0.0081 Acc: 0.8058

Epoch 5/79
----------
train Loss: 0.0049 Acc: 0.8669
val Loss: 0.0057 Acc: 0.8252

Epoch 6/79
----------
train Loss: 0.0043 Acc: 0.8607
val Loss: 0.0093 Acc: 0.8350

Epoch 7/79
----------
train Loss: 0.0038 Acc: 0.8960
val Loss: 0.0061 Acc: 0.8398

Epoch 8/79
----------
LR is set to 0.001
train Loss: 0.0036 Acc: 0.9023
val Loss: 0.0072 Acc: 0.8447

Epoch 9/79
----------
train Loss: 0.0036 Acc: 0.9023
val Loss: 0.0064 Acc: 0.8447

Epoch 10/79
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0057 Acc: 0.8447

Epoch 11/79
----------
train Loss: 0.0034 Acc: 0.9064
val Loss: 0.0055 Acc: 0.8447

Epoch 12/79
----------
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0083 Acc: 0.8447

Epoch 13/79
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0055 Acc: 0.8447

Epoch 14/79
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0054 Acc: 0.8447

Epoch 15/79
----------
train Loss: 0.0032 Acc: 0.9272
val Loss: 0.0082 Acc: 0.8398

Epoch 16/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0032 Acc: 0.9397
val Loss: 0.0073 Acc: 0.8350

Epoch 17/79
----------
train Loss: 0.0033 Acc: 0.9064
val Loss: 0.0091 Acc: 0.8350

Epoch 18/79
----------
train Loss: 0.0034 Acc: 0.9210
val Loss: 0.0065 Acc: 0.8350

Epoch 19/79
----------
train Loss: 0.0032 Acc: 0.9314
val Loss: 0.0088 Acc: 0.8398

Epoch 20/79
----------
train Loss: 0.0032 Acc: 0.9231
val Loss: 0.0053 Acc: 0.8447

Epoch 21/79
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0066 Acc: 0.8447

Epoch 22/79
----------
train Loss: 0.0032 Acc: 0.9272
val Loss: 0.0051 Acc: 0.8447

Epoch 23/79
----------
train Loss: 0.0034 Acc: 0.9106
val Loss: 0.0071 Acc: 0.8447

Epoch 24/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0066 Acc: 0.8447

Epoch 25/79
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0101 Acc: 0.8447

Epoch 26/79
----------
train Loss: 0.0032 Acc: 0.9168
val Loss: 0.0067 Acc: 0.8398

Epoch 27/79
----------
train Loss: 0.0031 Acc: 0.9356
val Loss: 0.0068 Acc: 0.8447

Epoch 28/79
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0066 Acc: 0.8447

Epoch 29/79
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0070 Acc: 0.8447

Epoch 30/79
----------
train Loss: 0.0032 Acc: 0.9314
val Loss: 0.0080 Acc: 0.8495

Epoch 31/79
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0066 Acc: 0.8447

Epoch 32/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0070 Acc: 0.8447

Epoch 33/79
----------
train Loss: 0.0033 Acc: 0.9168
val Loss: 0.0062 Acc: 0.8447

Epoch 34/79
----------
train Loss: 0.0030 Acc: 0.9376
val Loss: 0.0051 Acc: 0.8447

Epoch 35/79
----------
train Loss: 0.0035 Acc: 0.8960
val Loss: 0.0082 Acc: 0.8398

Epoch 36/79
----------
train Loss: 0.0032 Acc: 0.9231
val Loss: 0.0110 Acc: 0.8398

Epoch 37/79
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0096 Acc: 0.8398

Epoch 38/79
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0062 Acc: 0.8398

Epoch 39/79
----------
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0061 Acc: 0.8447

Epoch 40/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0035 Acc: 0.9106
val Loss: 0.0062 Acc: 0.8398

Epoch 41/79
----------
train Loss: 0.0031 Acc: 0.9272
val Loss: 0.0080 Acc: 0.8398

Epoch 42/79
----------
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0093 Acc: 0.8447

Epoch 43/79
----------
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0085 Acc: 0.8398

Epoch 44/79
----------
train Loss: 0.0033 Acc: 0.9085
val Loss: 0.0109 Acc: 0.8350

Epoch 45/79
----------
train Loss: 0.0033 Acc: 0.9335
val Loss: 0.0070 Acc: 0.8398

Epoch 46/79
----------
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0068 Acc: 0.8447

Epoch 47/79
----------
train Loss: 0.0033 Acc: 0.9168
val Loss: 0.0077 Acc: 0.8447

Epoch 48/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0057 Acc: 0.8447

Epoch 49/79
----------
train Loss: 0.0035 Acc: 0.9085
val Loss: 0.0069 Acc: 0.8398

Epoch 50/79
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0059 Acc: 0.8447

Epoch 51/79
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0061 Acc: 0.8447

Epoch 52/79
----------
train Loss: 0.0032 Acc: 0.9272
val Loss: 0.0051 Acc: 0.8398

Epoch 53/79
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0127 Acc: 0.8447

Epoch 54/79
----------
train Loss: 0.0034 Acc: 0.9168
val Loss: 0.0058 Acc: 0.8447

Epoch 55/79
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0055 Acc: 0.8398

Epoch 56/79
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9231
val Loss: 0.0057 Acc: 0.8447

Epoch 57/79
----------
train Loss: 0.0031 Acc: 0.9272
val Loss: 0.0057 Acc: 0.8447

Epoch 58/79
----------
train Loss: 0.0035 Acc: 0.9064
val Loss: 0.0087 Acc: 0.8447

Epoch 59/79
----------
train Loss: 0.0034 Acc: 0.9210
val Loss: 0.0077 Acc: 0.8447

Epoch 60/79
----------
train Loss: 0.0033 Acc: 0.9148
val Loss: 0.0065 Acc: 0.8447

Epoch 61/79
----------
train Loss: 0.0034 Acc: 0.9044
val Loss: 0.0086 Acc: 0.8398

Epoch 62/79
----------
train Loss: 0.0033 Acc: 0.9064
val Loss: 0.0062 Acc: 0.8495

Epoch 63/79
----------
train Loss: 0.0033 Acc: 0.9314
val Loss: 0.0110 Acc: 0.8544

Epoch 64/79
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0036 Acc: 0.9002
val Loss: 0.0094 Acc: 0.8544

Epoch 65/79
----------
train Loss: 0.0034 Acc: 0.9044
val Loss: 0.0070 Acc: 0.8495

Epoch 66/79
----------
train Loss: 0.0033 Acc: 0.9335
val Loss: 0.0064 Acc: 0.8447

Epoch 67/79
----------
train Loss: 0.0034 Acc: 0.9231
val Loss: 0.0064 Acc: 0.8495

Epoch 68/79
----------
train Loss: 0.0031 Acc: 0.9272
val Loss: 0.0082 Acc: 0.8495

Epoch 69/79
----------
train Loss: 0.0034 Acc: 0.9085
val Loss: 0.0057 Acc: 0.8447

Epoch 70/79
----------
train Loss: 0.0031 Acc: 0.9168
val Loss: 0.0089 Acc: 0.8447

Epoch 71/79
----------
train Loss: 0.0032 Acc: 0.9314
val Loss: 0.0078 Acc: 0.8398

Epoch 72/79
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0032 Acc: 0.9335
val Loss: 0.0098 Acc: 0.8447

Epoch 73/79
----------
train Loss: 0.0035 Acc: 0.9002
val Loss: 0.0078 Acc: 0.8398

Epoch 74/79
----------
train Loss: 0.0036 Acc: 0.9148
val Loss: 0.0057 Acc: 0.8398

Epoch 75/79
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0056 Acc: 0.8398

Epoch 76/79
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0069 Acc: 0.8398

Epoch 77/79
----------
train Loss: 0.0031 Acc: 0.9439
val Loss: 0.0057 Acc: 0.8398

Epoch 78/79
----------
train Loss: 0.0034 Acc: 0.9252
val Loss: 0.0079 Acc: 0.8447

Epoch 79/79
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0055 Acc: 0.8447

Training complete in 4m 6s
Best val Acc: 0.854369

---Fine tuning.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0051 Acc: 0.8301

Epoch 1/79
----------
train Loss: 0.0019 Acc: 0.9584
val Loss: 0.0054 Acc: 0.8738

Epoch 2/79
----------
train Loss: 0.0011 Acc: 0.9709
val Loss: 0.0053 Acc: 0.8592

Epoch 3/79
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0093 Acc: 0.8592

Epoch 4/79
----------
train Loss: 0.0005 Acc: 0.9979
val Loss: 0.0047 Acc: 0.8689

Epoch 5/79
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0044 Acc: 0.8689

Epoch 6/79
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8738

Epoch 7/79
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0053 Acc: 0.8689

Epoch 8/79
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0075 Acc: 0.8738

Epoch 9/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8786

Epoch 10/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8883

Epoch 11/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 12/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 13/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8883

Epoch 14/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 15/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 16/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 17/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8883

Epoch 18/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8883

Epoch 19/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 20/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8883

Epoch 21/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8883

Epoch 22/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8883

Epoch 23/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 24/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8883

Epoch 25/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8883

Epoch 26/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8883

Epoch 27/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0037 Acc: 0.8883

Epoch 28/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8835

Epoch 29/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8883

Epoch 30/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8883

Epoch 31/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8835

Epoch 32/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8835

Epoch 33/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8835

Epoch 34/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 35/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8883

Epoch 36/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8883

Epoch 37/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8835

Epoch 38/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8786

Epoch 39/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0065 Acc: 0.8883

Epoch 40/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0043 Acc: 0.8883

Epoch 41/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8883

Epoch 42/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8835

Epoch 43/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8883

Epoch 44/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8883

Epoch 45/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 46/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8883

Epoch 47/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 48/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8835

Epoch 49/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8883

Epoch 50/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 51/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 52/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8883

Epoch 53/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8883

Epoch 54/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8883

Epoch 55/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 56/79
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8883

Epoch 57/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0054 Acc: 0.8883

Epoch 58/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 59/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8883

Epoch 60/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0043 Acc: 0.8883

Epoch 61/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 62/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 63/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8883

Epoch 64/79
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8883

Epoch 65/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8883

Epoch 66/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 67/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 68/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 69/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8883

Epoch 70/79
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8883

Epoch 71/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8883

Epoch 72/79
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0035 Acc: 0.8883

Epoch 73/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 74/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 75/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8883

Epoch 76/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8883

Epoch 77/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8883

Epoch 78/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 79/79
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Training complete in 4m 17s
Best val Acc: 0.888350

---Testing---
Test accuracy: 0.966521
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 92 %
Accuracy of Lamniformes : 87 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.9531280936477863, std: 0.04588798212389398

Model saved in "./weights/shark_[0.99]_mean[0.98]_std[0.02].save".
--------------------

run info[val: 0.1, epoch: 83, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0226 Acc: 0.2359
val Loss: 0.0235 Acc: 0.4412

Epoch 1/82
----------
train Loss: 0.0133 Acc: 0.6123
val Loss: 0.0136 Acc: 0.7353

Epoch 2/82
----------
train Loss: 0.0089 Acc: 0.7544
val Loss: 0.0113 Acc: 0.7794

Epoch 3/82
----------
train Loss: 0.0061 Acc: 0.8417
val Loss: 0.0115 Acc: 0.7206

Epoch 4/82
----------
train Loss: 0.0055 Acc: 0.8724
val Loss: 0.0092 Acc: 0.8088

Epoch 5/82
----------
train Loss: 0.0055 Acc: 0.8498
val Loss: 0.0127 Acc: 0.6912

Epoch 6/82
----------
train Loss: 0.0057 Acc: 0.8223
val Loss: 0.0097 Acc: 0.7647

Epoch 7/82
----------
LR is set to 0.001
train Loss: 0.0049 Acc: 0.8853
val Loss: 0.0095 Acc: 0.7941

Epoch 8/82
----------
train Loss: 0.0040 Acc: 0.9015
val Loss: 0.0093 Acc: 0.7794

Epoch 9/82
----------
train Loss: 0.0035 Acc: 0.9192
val Loss: 0.0094 Acc: 0.7794

Epoch 10/82
----------
train Loss: 0.0035 Acc: 0.9225
val Loss: 0.0096 Acc: 0.7794

Epoch 11/82
----------
train Loss: 0.0037 Acc: 0.9047
val Loss: 0.0098 Acc: 0.7941

Epoch 12/82
----------
train Loss: 0.0035 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7941

Epoch 13/82
----------
train Loss: 0.0035 Acc: 0.9047
val Loss: 0.0092 Acc: 0.7941

Epoch 14/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0035 Acc: 0.9095
val Loss: 0.0093 Acc: 0.7794

Epoch 15/82
----------
train Loss: 0.0041 Acc: 0.9128
val Loss: 0.0095 Acc: 0.7794

Epoch 16/82
----------
train Loss: 0.0036 Acc: 0.9128
val Loss: 0.0096 Acc: 0.7794

Epoch 17/82
----------
train Loss: 0.0033 Acc: 0.9063
val Loss: 0.0096 Acc: 0.7794

Epoch 18/82
----------
train Loss: 0.0034 Acc: 0.9257
val Loss: 0.0097 Acc: 0.7794

Epoch 19/82
----------
train Loss: 0.0038 Acc: 0.9257
val Loss: 0.0096 Acc: 0.7941

Epoch 20/82
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0095 Acc: 0.7794

Epoch 21/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0041 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7941

Epoch 22/82
----------
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0095 Acc: 0.7941

Epoch 23/82
----------
train Loss: 0.0032 Acc: 0.9192
val Loss: 0.0095 Acc: 0.7794

Epoch 24/82
----------
train Loss: 0.0031 Acc: 0.9225
val Loss: 0.0096 Acc: 0.7794

Epoch 25/82
----------
train Loss: 0.0035 Acc: 0.9192
val Loss: 0.0096 Acc: 0.7941

Epoch 26/82
----------
train Loss: 0.0035 Acc: 0.9241
val Loss: 0.0094 Acc: 0.7941

Epoch 27/82
----------
train Loss: 0.0033 Acc: 0.9176
val Loss: 0.0094 Acc: 0.7941

Epoch 28/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0033 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7794

Epoch 29/82
----------
train Loss: 0.0034 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7941

Epoch 30/82
----------
train Loss: 0.0033 Acc: 0.9063
val Loss: 0.0095 Acc: 0.7794

Epoch 31/82
----------
train Loss: 0.0034 Acc: 0.9208
val Loss: 0.0095 Acc: 0.7941

Epoch 32/82
----------
train Loss: 0.0036 Acc: 0.9111
val Loss: 0.0094 Acc: 0.7941

Epoch 33/82
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0094 Acc: 0.7941

Epoch 34/82
----------
train Loss: 0.0031 Acc: 0.9144
val Loss: 0.0094 Acc: 0.7941

Epoch 35/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0034 Acc: 0.9160
val Loss: 0.0093 Acc: 0.7941

Epoch 36/82
----------
train Loss: 0.0038 Acc: 0.9160
val Loss: 0.0095 Acc: 0.7941

Epoch 37/82
----------
train Loss: 0.0036 Acc: 0.9095
val Loss: 0.0094 Acc: 0.7941

Epoch 38/82
----------
train Loss: 0.0037 Acc: 0.9176
val Loss: 0.0094 Acc: 0.7941

Epoch 39/82
----------
train Loss: 0.0038 Acc: 0.9208
val Loss: 0.0096 Acc: 0.7941

Epoch 40/82
----------
train Loss: 0.0033 Acc: 0.9225
val Loss: 0.0097 Acc: 0.7941

Epoch 41/82
----------
train Loss: 0.0035 Acc: 0.9225
val Loss: 0.0098 Acc: 0.7794

Epoch 42/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0098 Acc: 0.7941

Epoch 43/82
----------
train Loss: 0.0042 Acc: 0.9208
val Loss: 0.0097 Acc: 0.7794

Epoch 44/82
----------
train Loss: 0.0039 Acc: 0.9225
val Loss: 0.0096 Acc: 0.7794

Epoch 45/82
----------
train Loss: 0.0032 Acc: 0.9241
val Loss: 0.0096 Acc: 0.7794

Epoch 46/82
----------
train Loss: 0.0037 Acc: 0.9257
val Loss: 0.0096 Acc: 0.7794

Epoch 47/82
----------
train Loss: 0.0033 Acc: 0.9257
val Loss: 0.0096 Acc: 0.7794

Epoch 48/82
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0095 Acc: 0.7794

Epoch 49/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0096 Acc: 0.7941

Epoch 50/82
----------
train Loss: 0.0033 Acc: 0.9289
val Loss: 0.0096 Acc: 0.7941

Epoch 51/82
----------
train Loss: 0.0034 Acc: 0.9225
val Loss: 0.0096 Acc: 0.7794

Epoch 52/82
----------
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0094 Acc: 0.7794

Epoch 53/82
----------
train Loss: 0.0036 Acc: 0.9160
val Loss: 0.0095 Acc: 0.7794

Epoch 54/82
----------
train Loss: 0.0042 Acc: 0.9160
val Loss: 0.0096 Acc: 0.7794

Epoch 55/82
----------
train Loss: 0.0033 Acc: 0.9386
val Loss: 0.0097 Acc: 0.7941

Epoch 56/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0035 Acc: 0.9225
val Loss: 0.0095 Acc: 0.7941

Epoch 57/82
----------
train Loss: 0.0033 Acc: 0.9128
val Loss: 0.0095 Acc: 0.7794

Epoch 58/82
----------
train Loss: 0.0032 Acc: 0.9192
val Loss: 0.0095 Acc: 0.7794

Epoch 59/82
----------
train Loss: 0.0035 Acc: 0.9160
val Loss: 0.0095 Acc: 0.7941

Epoch 60/82
----------
train Loss: 0.0035 Acc: 0.9192
val Loss: 0.0095 Acc: 0.7941

Epoch 61/82
----------
train Loss: 0.0036 Acc: 0.9160
val Loss: 0.0096 Acc: 0.7794

Epoch 62/82
----------
train Loss: 0.0038 Acc: 0.9144
val Loss: 0.0096 Acc: 0.7794

Epoch 63/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0039 Acc: 0.9241
val Loss: 0.0096 Acc: 0.7794

Epoch 64/82
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0096 Acc: 0.7794

Epoch 65/82
----------
train Loss: 0.0045 Acc: 0.9111
val Loss: 0.0095 Acc: 0.7794

Epoch 66/82
----------
train Loss: 0.0038 Acc: 0.9241
val Loss: 0.0095 Acc: 0.7941

Epoch 67/82
----------
train Loss: 0.0039 Acc: 0.9160
val Loss: 0.0095 Acc: 0.7794

Epoch 68/82
----------
train Loss: 0.0041 Acc: 0.9160
val Loss: 0.0094 Acc: 0.7941

Epoch 69/82
----------
train Loss: 0.0034 Acc: 0.9241
val Loss: 0.0095 Acc: 0.7794

Epoch 70/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0038 Acc: 0.9079
val Loss: 0.0096 Acc: 0.7794

Epoch 71/82
----------
train Loss: 0.0034 Acc: 0.9111
val Loss: 0.0097 Acc: 0.7794

Epoch 72/82
----------
train Loss: 0.0034 Acc: 0.9192
val Loss: 0.0096 Acc: 0.7794

Epoch 73/82
----------
train Loss: 0.0037 Acc: 0.9160
val Loss: 0.0097 Acc: 0.7794

Epoch 74/82
----------
train Loss: 0.0034 Acc: 0.9257
val Loss: 0.0097 Acc: 0.7794

Epoch 75/82
----------
train Loss: 0.0037 Acc: 0.9241
val Loss: 0.0095 Acc: 0.7794

Epoch 76/82
----------
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0096 Acc: 0.7794

Epoch 77/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0034 Acc: 0.9144
val Loss: 0.0096 Acc: 0.7794

Epoch 78/82
----------
train Loss: 0.0038 Acc: 0.9095
val Loss: 0.0095 Acc: 0.7794

Epoch 79/82
----------
train Loss: 0.0037 Acc: 0.9257
val Loss: 0.0095 Acc: 0.7794

Epoch 80/82
----------
train Loss: 0.0035 Acc: 0.9241
val Loss: 0.0095 Acc: 0.7794

Epoch 81/82
----------
train Loss: 0.0035 Acc: 0.9111
val Loss: 0.0095 Acc: 0.7941

Epoch 82/82
----------
train Loss: 0.0034 Acc: 0.9208
val Loss: 0.0096 Acc: 0.7941

Training complete in 4m 10s
Best val Acc: 0.808824

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0046 Acc: 0.8885
val Loss: 0.0110 Acc: 0.7059

Epoch 1/82
----------
train Loss: 0.0028 Acc: 0.9386
val Loss: 0.0103 Acc: 0.8088

Epoch 2/82
----------
train Loss: 0.0018 Acc: 0.9515
val Loss: 0.0093 Acc: 0.7941

Epoch 3/82
----------
train Loss: 0.0011 Acc: 0.9806
val Loss: 0.0081 Acc: 0.8235

Epoch 4/82
----------
train Loss: 0.0011 Acc: 0.9677
val Loss: 0.0100 Acc: 0.8235

Epoch 5/82
----------
train Loss: 0.0006 Acc: 0.9903
val Loss: 0.0106 Acc: 0.8382

Epoch 6/82
----------
train Loss: 0.0004 Acc: 0.9919
val Loss: 0.0112 Acc: 0.7941

Epoch 7/82
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9903
val Loss: 0.0107 Acc: 0.8088

Epoch 8/82
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0108 Acc: 0.7941

Epoch 9/82
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0108 Acc: 0.7941

Epoch 10/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0105 Acc: 0.7941

Epoch 11/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 12/82
----------
train Loss: 0.0003 Acc: 0.9935
val Loss: 0.0111 Acc: 0.7941

Epoch 13/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0113 Acc: 0.7941

Epoch 14/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0108 Acc: 0.8235

Epoch 15/82
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0109 Acc: 0.8235

Epoch 16/82
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0109 Acc: 0.8088

Epoch 17/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8088

Epoch 18/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8088

Epoch 19/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0110 Acc: 0.8088

Epoch 20/82
----------
train Loss: 0.0005 Acc: 0.9968
val Loss: 0.0111 Acc: 0.7941

Epoch 21/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0109 Acc: 0.8088

Epoch 22/82
----------
train Loss: 0.0005 Acc: 0.9984
val Loss: 0.0108 Acc: 0.8235

Epoch 23/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0108 Acc: 0.8235

Epoch 24/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0108 Acc: 0.8235

Epoch 25/82
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0109 Acc: 0.8235

Epoch 26/82
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0110 Acc: 0.8088

Epoch 27/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0110 Acc: 0.7941

Epoch 28/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0109 Acc: 0.7941

Epoch 29/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0108 Acc: 0.8088

Epoch 30/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0108 Acc: 0.8088

Epoch 31/82
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0110 Acc: 0.8088

Epoch 32/82
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0111 Acc: 0.8088

Epoch 33/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8088

Epoch 34/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0111 Acc: 0.8088

Epoch 35/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8088

Epoch 36/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0110 Acc: 0.7941

Epoch 37/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 38/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.7941

Epoch 39/82
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0109 Acc: 0.7941

Epoch 40/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 41/82
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0111 Acc: 0.7941

Epoch 42/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0110 Acc: 0.7941

Epoch 43/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0113 Acc: 0.7941

Epoch 44/82
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0115 Acc: 0.7941

Epoch 45/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0112 Acc: 0.8088

Epoch 46/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0112 Acc: 0.7941

Epoch 47/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8235

Epoch 48/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0106 Acc: 0.8235

Epoch 49/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8088

Epoch 50/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0108 Acc: 0.8088

Epoch 51/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 52/82
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0110 Acc: 0.7941

Epoch 53/82
----------
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0110 Acc: 0.8088

Epoch 54/82
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0110 Acc: 0.8088

Epoch 55/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8235

Epoch 56/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8088

Epoch 57/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0112 Acc: 0.7941

Epoch 58/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0109 Acc: 0.7941

Epoch 59/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0109 Acc: 0.8088

Epoch 60/82
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0111 Acc: 0.7941

Epoch 61/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8235

Epoch 62/82
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0110 Acc: 0.8088

Epoch 63/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0110 Acc: 0.8088

Epoch 64/82
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0110 Acc: 0.7941

Epoch 65/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.8088

Epoch 66/82
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0112 Acc: 0.7941

Epoch 67/82
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0109 Acc: 0.7941

Epoch 68/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0109 Acc: 0.7941

Epoch 69/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0108 Acc: 0.7941

Epoch 70/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 71/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 72/82
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0109 Acc: 0.7941

Epoch 73/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.8088

Epoch 74/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8088

Epoch 75/82
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8088

Epoch 76/82
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0114 Acc: 0.7941

Epoch 77/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0112 Acc: 0.7941

Epoch 78/82
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0111 Acc: 0.7941

Epoch 79/82
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8088

Epoch 80/82
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0112 Acc: 0.8088

Epoch 81/82
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0109 Acc: 0.7941

Epoch 82/82
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0111 Acc: 0.8088

Training complete in 4m 42s
Best val Acc: 0.838235

---Testing---
Test accuracy: 0.976710
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 100 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 97 %
mean: 0.9690454438414522, std: 0.02783855257172969
--------------------

run info[val: 0.15, epoch: 70, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0203 Acc: 0.2209
val Loss: 0.0344 Acc: 0.3786

Epoch 1/69
----------
train Loss: 0.0131 Acc: 0.5993
val Loss: 0.0272 Acc: 0.4563

Epoch 2/69
----------
train Loss: 0.0085 Acc: 0.7432
val Loss: 0.0210 Acc: 0.7864

Epoch 3/69
----------
train Loss: 0.0060 Acc: 0.8356
val Loss: 0.0095 Acc: 0.7282

Epoch 4/69
----------
train Loss: 0.0048 Acc: 0.8562
val Loss: 0.0071 Acc: 0.7961

Epoch 5/69
----------
train Loss: 0.0041 Acc: 0.8887
val Loss: 0.0087 Acc: 0.7864

Epoch 6/69
----------
train Loss: 0.0035 Acc: 0.8973
val Loss: 0.0080 Acc: 0.8058

Epoch 7/69
----------
train Loss: 0.0028 Acc: 0.9281
val Loss: 0.0231 Acc: 0.7961

Epoch 8/69
----------
train Loss: 0.0026 Acc: 0.9384
val Loss: 0.0222 Acc: 0.8058

Epoch 9/69
----------
train Loss: 0.0025 Acc: 0.9486
val Loss: 0.0071 Acc: 0.7864

Epoch 10/69
----------
LR is set to 0.001
train Loss: 0.0023 Acc: 0.9521
val Loss: 0.0098 Acc: 0.7767

Epoch 11/69
----------
train Loss: 0.0022 Acc: 0.9538
val Loss: 0.0239 Acc: 0.8058

Epoch 12/69
----------
train Loss: 0.0023 Acc: 0.9469
val Loss: 0.0104 Acc: 0.8058

Epoch 13/69
----------
train Loss: 0.0021 Acc: 0.9503
val Loss: 0.0073 Acc: 0.8155

Epoch 14/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0329 Acc: 0.8155

Epoch 15/69
----------
train Loss: 0.0022 Acc: 0.9658
val Loss: 0.0067 Acc: 0.8155

Epoch 16/69
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0091 Acc: 0.8155

Epoch 17/69
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0062 Acc: 0.8058

Epoch 18/69
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0174 Acc: 0.7961

Epoch 19/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0277 Acc: 0.8155

Epoch 20/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0105 Acc: 0.8155

Epoch 21/69
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0086 Acc: 0.8155

Epoch 22/69
----------
train Loss: 0.0021 Acc: 0.9658
val Loss: 0.0155 Acc: 0.8155

Epoch 23/69
----------
train Loss: 0.0021 Acc: 0.9726
val Loss: 0.0125 Acc: 0.8155

Epoch 24/69
----------
train Loss: 0.0021 Acc: 0.9658
val Loss: 0.0227 Acc: 0.8155

Epoch 25/69
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0162 Acc: 0.8155

Epoch 26/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0069 Acc: 0.8155

Epoch 27/69
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0081 Acc: 0.8155

Epoch 28/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0130 Acc: 0.8155

Epoch 29/69
----------
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0084 Acc: 0.8155

Epoch 30/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0124 Acc: 0.8058

Epoch 31/69
----------
train Loss: 0.0020 Acc: 0.9640
val Loss: 0.0083 Acc: 0.8155

Epoch 32/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0191 Acc: 0.7961

Epoch 33/69
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0064 Acc: 0.8155

Epoch 34/69
----------
train Loss: 0.0019 Acc: 0.9623
val Loss: 0.0131 Acc: 0.8058

Epoch 35/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0063 Acc: 0.8058

Epoch 36/69
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0094 Acc: 0.8058

Epoch 37/69
----------
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0108 Acc: 0.8058

Epoch 38/69
----------
train Loss: 0.0021 Acc: 0.9692
val Loss: 0.0070 Acc: 0.8058

Epoch 39/69
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0129 Acc: 0.7961

Epoch 40/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0021 Acc: 0.9538
val Loss: 0.0127 Acc: 0.7961

Epoch 41/69
----------
train Loss: 0.0021 Acc: 0.9555
val Loss: 0.0116 Acc: 0.8155

Epoch 42/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0093 Acc: 0.8155

Epoch 43/69
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0165 Acc: 0.8155

Epoch 44/69
----------
train Loss: 0.0021 Acc: 0.9503
val Loss: 0.0094 Acc: 0.8155

Epoch 45/69
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0070 Acc: 0.8155

Epoch 46/69
----------
train Loss: 0.0021 Acc: 0.9555
val Loss: 0.0181 Acc: 0.8155

Epoch 47/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0183 Acc: 0.8058

Epoch 48/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0062 Acc: 0.8155

Epoch 49/69
----------
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0064 Acc: 0.8155

Epoch 50/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0021 Acc: 0.9572
val Loss: 0.0069 Acc: 0.8155

Epoch 51/69
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0097 Acc: 0.8155

Epoch 52/69
----------
train Loss: 0.0020 Acc: 0.9640
val Loss: 0.0190 Acc: 0.8155

Epoch 53/69
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0103 Acc: 0.8058

Epoch 54/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0061 Acc: 0.8058

Epoch 55/69
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0062 Acc: 0.8058

Epoch 56/69
----------
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0089 Acc: 0.8155

Epoch 57/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0062 Acc: 0.8155

Epoch 58/69
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0233 Acc: 0.8155

Epoch 59/69
----------
train Loss: 0.0021 Acc: 0.9658
val Loss: 0.0113 Acc: 0.8155

Epoch 60/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0234 Acc: 0.8155

Epoch 61/69
----------
train Loss: 0.0021 Acc: 0.9623
val Loss: 0.0236 Acc: 0.8155

Epoch 62/69
----------
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0214 Acc: 0.8155

Epoch 63/69
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0080 Acc: 0.8155

Epoch 64/69
----------
train Loss: 0.0021 Acc: 0.9743
val Loss: 0.0148 Acc: 0.8155

Epoch 65/69
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0060 Acc: 0.8155

Epoch 66/69
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0069 Acc: 0.8155

Epoch 67/69
----------
train Loss: 0.0021 Acc: 0.9623
val Loss: 0.0096 Acc: 0.8155

Epoch 68/69
----------
train Loss: 0.0022 Acc: 0.9623
val Loss: 0.0130 Acc: 0.8155

Epoch 69/69
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0193 Acc: 0.8155

Training complete in 3m 32s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0022 Acc: 0.9486
val Loss: 0.0174 Acc: 0.8350

Epoch 1/69
----------
train Loss: 0.0009 Acc: 0.9932
val Loss: 0.0051 Acc: 0.8738

Epoch 2/69
----------
train Loss: 0.0005 Acc: 0.9966
val Loss: 0.0052 Acc: 0.8738

Epoch 3/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0151 Acc: 0.8544

Epoch 4/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 5/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8447

Epoch 6/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 7/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0251 Acc: 0.8447

Epoch 8/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8544

Epoch 9/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 10/69
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0103 Acc: 0.8544

Epoch 11/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 12/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8544

Epoch 13/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8544

Epoch 14/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8544

Epoch 15/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 16/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 17/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 18/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8544

Epoch 19/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8544

Epoch 20/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 21/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 22/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8447

Epoch 23/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8447

Epoch 24/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 25/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0105 Acc: 0.8544

Epoch 26/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8544

Epoch 27/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0177 Acc: 0.8544

Epoch 28/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8544

Epoch 29/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0177 Acc: 0.8544

Epoch 30/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8544

Epoch 31/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 32/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 33/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 34/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 35/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 36/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 37/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 38/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0176 Acc: 0.8544

Epoch 39/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8544

Epoch 40/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8447

Epoch 41/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8447

Epoch 42/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8544

Epoch 43/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 44/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 45/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8544

Epoch 46/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0178 Acc: 0.8544

Epoch 47/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8544

Epoch 48/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8447

Epoch 49/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 50/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 51/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8544

Epoch 52/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0176 Acc: 0.8544

Epoch 53/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8544

Epoch 54/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 55/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0320 Acc: 0.8544

Epoch 56/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0243 Acc: 0.8544

Epoch 57/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8544

Epoch 58/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8544

Epoch 59/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8447

Epoch 60/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 61/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0228 Acc: 0.8544

Epoch 62/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0227 Acc: 0.8544

Epoch 63/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0284 Acc: 0.8544

Epoch 64/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 65/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8544

Epoch 66/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0180 Acc: 0.8544

Epoch 67/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 68/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8544

Epoch 69/69
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8544

Training complete in 3m 60s
Best val Acc: 0.873786

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.9707158129460319, std: 0.03379842340448217
--------------------

run info[val: 0.2, epoch: 82, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0219 Acc: 0.2418
val Loss: 0.0258 Acc: 0.2920

Epoch 1/81
----------
train Loss: 0.0143 Acc: 0.5964
val Loss: 0.0158 Acc: 0.6496

Epoch 2/81
----------
train Loss: 0.0091 Acc: 0.7345
val Loss: 0.0115 Acc: 0.7518

Epoch 3/81
----------
train Loss: 0.0069 Acc: 0.8109
val Loss: 0.0102 Acc: 0.7372

Epoch 4/81
----------
train Loss: 0.0058 Acc: 0.8473
val Loss: 0.0081 Acc: 0.8102

Epoch 5/81
----------
train Loss: 0.0050 Acc: 0.8745
val Loss: 0.0077 Acc: 0.8102

Epoch 6/81
----------
train Loss: 0.0038 Acc: 0.9036
val Loss: 0.0083 Acc: 0.8029

Epoch 7/81
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0078 Acc: 0.8248

Epoch 8/81
----------
LR is set to 0.001
train Loss: 0.0034 Acc: 0.9236
val Loss: 0.0081 Acc: 0.8321

Epoch 9/81
----------
train Loss: 0.0035 Acc: 0.9200
val Loss: 0.0072 Acc: 0.8467

Epoch 10/81
----------
train Loss: 0.0033 Acc: 0.9236
val Loss: 0.0090 Acc: 0.8175

Epoch 11/81
----------
train Loss: 0.0032 Acc: 0.9164
val Loss: 0.0070 Acc: 0.8175

Epoch 12/81
----------
train Loss: 0.0035 Acc: 0.9055
val Loss: 0.0069 Acc: 0.8175

Epoch 13/81
----------
train Loss: 0.0032 Acc: 0.9182
val Loss: 0.0078 Acc: 0.8248

Epoch 14/81
----------
train Loss: 0.0034 Acc: 0.9182
val Loss: 0.0078 Acc: 0.8321

Epoch 15/81
----------
train Loss: 0.0031 Acc: 0.9382
val Loss: 0.0068 Acc: 0.8248

Epoch 16/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0030 Acc: 0.9218
val Loss: 0.0082 Acc: 0.8321

Epoch 17/81
----------
train Loss: 0.0031 Acc: 0.9255
val Loss: 0.0072 Acc: 0.8248

Epoch 18/81
----------
train Loss: 0.0030 Acc: 0.9291
val Loss: 0.0075 Acc: 0.8321

Epoch 19/81
----------
train Loss: 0.0031 Acc: 0.9236
val Loss: 0.0075 Acc: 0.8321

Epoch 20/81
----------
train Loss: 0.0035 Acc: 0.9236
val Loss: 0.0080 Acc: 0.8175

Epoch 21/81
----------
train Loss: 0.0030 Acc: 0.9273
val Loss: 0.0071 Acc: 0.8248

Epoch 22/81
----------
train Loss: 0.0031 Acc: 0.9273
val Loss: 0.0078 Acc: 0.8248

Epoch 23/81
----------
train Loss: 0.0031 Acc: 0.9182
val Loss: 0.0076 Acc: 0.8175

Epoch 24/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0032 Acc: 0.9109
val Loss: 0.0075 Acc: 0.8248

Epoch 25/81
----------
train Loss: 0.0031 Acc: 0.9273
val Loss: 0.0075 Acc: 0.8321

Epoch 26/81
----------
train Loss: 0.0032 Acc: 0.9200
val Loss: 0.0078 Acc: 0.8321

Epoch 27/81
----------
train Loss: 0.0032 Acc: 0.9382
val Loss: 0.0071 Acc: 0.8321

Epoch 28/81
----------
train Loss: 0.0031 Acc: 0.9255
val Loss: 0.0074 Acc: 0.8248

Epoch 29/81
----------
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0081 Acc: 0.8248

Epoch 30/81
----------
train Loss: 0.0033 Acc: 0.9164
val Loss: 0.0075 Acc: 0.8175

Epoch 31/81
----------
train Loss: 0.0031 Acc: 0.9218
val Loss: 0.0077 Acc: 0.8175

Epoch 32/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0032 Acc: 0.9109
val Loss: 0.0077 Acc: 0.8175

Epoch 33/81
----------
train Loss: 0.0030 Acc: 0.9327
val Loss: 0.0069 Acc: 0.8175

Epoch 34/81
----------
train Loss: 0.0031 Acc: 0.9200
val Loss: 0.0068 Acc: 0.8248

Epoch 35/81
----------
train Loss: 0.0034 Acc: 0.9218
val Loss: 0.0078 Acc: 0.8175

Epoch 36/81
----------
train Loss: 0.0029 Acc: 0.9309
val Loss: 0.0074 Acc: 0.8175

Epoch 37/81
----------
train Loss: 0.0031 Acc: 0.9273
val Loss: 0.0070 Acc: 0.8248

Epoch 38/81
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0075 Acc: 0.8175

Epoch 39/81
----------
train Loss: 0.0034 Acc: 0.9091
val Loss: 0.0083 Acc: 0.8248

Epoch 40/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9236
val Loss: 0.0074 Acc: 0.8175

Epoch 41/81
----------
train Loss: 0.0032 Acc: 0.9255
val Loss: 0.0071 Acc: 0.8175

Epoch 42/81
----------
train Loss: 0.0031 Acc: 0.9309
val Loss: 0.0071 Acc: 0.8175

Epoch 43/81
----------
train Loss: 0.0033 Acc: 0.9309
val Loss: 0.0080 Acc: 0.8248

Epoch 44/81
----------
train Loss: 0.0031 Acc: 0.9364
val Loss: 0.0073 Acc: 0.8321

Epoch 45/81
----------
train Loss: 0.0031 Acc: 0.9236
val Loss: 0.0082 Acc: 0.8248

Epoch 46/81
----------
train Loss: 0.0034 Acc: 0.9236
val Loss: 0.0067 Acc: 0.8248

Epoch 47/81
----------
train Loss: 0.0034 Acc: 0.9255
val Loss: 0.0083 Acc: 0.8175

Epoch 48/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0030 Acc: 0.9291
val Loss: 0.0077 Acc: 0.8175

Epoch 49/81
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0074 Acc: 0.8248

Epoch 50/81
----------
train Loss: 0.0034 Acc: 0.9291
val Loss: 0.0067 Acc: 0.8175

Epoch 51/81
----------
train Loss: 0.0031 Acc: 0.9400
val Loss: 0.0075 Acc: 0.8248

Epoch 52/81
----------
train Loss: 0.0031 Acc: 0.9109
val Loss: 0.0067 Acc: 0.8321

Epoch 53/81
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0080 Acc: 0.8248

Epoch 54/81
----------
train Loss: 0.0032 Acc: 0.9255
val Loss: 0.0075 Acc: 0.8175

Epoch 55/81
----------
train Loss: 0.0036 Acc: 0.9055
val Loss: 0.0073 Acc: 0.8248

Epoch 56/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9109
val Loss: 0.0075 Acc: 0.8248

Epoch 57/81
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0071 Acc: 0.8175

Epoch 58/81
----------
train Loss: 0.0029 Acc: 0.9327
val Loss: 0.0084 Acc: 0.8248

Epoch 59/81
----------
train Loss: 0.0032 Acc: 0.9309
val Loss: 0.0073 Acc: 0.8175

Epoch 60/81
----------
train Loss: 0.0031 Acc: 0.9218
val Loss: 0.0072 Acc: 0.8175

Epoch 61/81
----------
train Loss: 0.0031 Acc: 0.9200
val Loss: 0.0073 Acc: 0.8248

Epoch 62/81
----------
train Loss: 0.0032 Acc: 0.9200
val Loss: 0.0072 Acc: 0.8248

Epoch 63/81
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0069 Acc: 0.8248

Epoch 64/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0030 Acc: 0.9218
val Loss: 0.0071 Acc: 0.8321

Epoch 65/81
----------
train Loss: 0.0033 Acc: 0.9218
val Loss: 0.0075 Acc: 0.8321

Epoch 66/81
----------
train Loss: 0.0031 Acc: 0.9327
val Loss: 0.0073 Acc: 0.8394

Epoch 67/81
----------
train Loss: 0.0031 Acc: 0.9309
val Loss: 0.0077 Acc: 0.8321

Epoch 68/81
----------
train Loss: 0.0034 Acc: 0.9109
val Loss: 0.0086 Acc: 0.8248

Epoch 69/81
----------
train Loss: 0.0030 Acc: 0.9345
val Loss: 0.0089 Acc: 0.8394

Epoch 70/81
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0082 Acc: 0.8394

Epoch 71/81
----------
train Loss: 0.0030 Acc: 0.9236
val Loss: 0.0073 Acc: 0.8248

Epoch 72/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0030 Acc: 0.9400
val Loss: 0.0078 Acc: 0.8248

Epoch 73/81
----------
train Loss: 0.0033 Acc: 0.9164
val Loss: 0.0079 Acc: 0.8248

Epoch 74/81
----------
train Loss: 0.0032 Acc: 0.9255
val Loss: 0.0065 Acc: 0.8248

Epoch 75/81
----------
train Loss: 0.0030 Acc: 0.9418
val Loss: 0.0077 Acc: 0.8248

Epoch 76/81
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0074 Acc: 0.8321

Epoch 77/81
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0066 Acc: 0.8321

Epoch 78/81
----------
train Loss: 0.0033 Acc: 0.9164
val Loss: 0.0071 Acc: 0.8321

Epoch 79/81
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0076 Acc: 0.8248

Epoch 80/81
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9345
val Loss: 0.0079 Acc: 0.8248

Epoch 81/81
----------
train Loss: 0.0031 Acc: 0.9364
val Loss: 0.0070 Acc: 0.8248

Training complete in 4m 6s
Best val Acc: 0.846715

---Fine tuning.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0076 Acc: 0.8321

Epoch 1/81
----------
train Loss: 0.0020 Acc: 0.9564
val Loss: 0.0056 Acc: 0.8467

Epoch 2/81
----------
train Loss: 0.0013 Acc: 0.9818
val Loss: 0.0066 Acc: 0.8540

Epoch 3/81
----------
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8540

Epoch 4/81
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0077 Acc: 0.8613

Epoch 5/81
----------
train Loss: 0.0004 Acc: 0.9964
val Loss: 0.0070 Acc: 0.8540

Epoch 6/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8686

Epoch 7/81
----------
train Loss: 0.0003 Acc: 0.9945
val Loss: 0.0052 Acc: 0.8613

Epoch 8/81
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0052 Acc: 0.8759

Epoch 9/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8759

Epoch 10/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0074 Acc: 0.8759

Epoch 11/81
----------
train Loss: 0.0002 Acc: 0.9945
val Loss: 0.0063 Acc: 0.8686

Epoch 12/81
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8759

Epoch 13/81
----------
train Loss: 0.0002 Acc: 0.9964
val Loss: 0.0053 Acc: 0.8759

Epoch 14/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0064 Acc: 0.8759

Epoch 15/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8686

Epoch 16/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 17/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 18/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8759

Epoch 19/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8686

Epoch 20/81
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0063 Acc: 0.8759

Epoch 21/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0059 Acc: 0.8686

Epoch 22/81
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0061 Acc: 0.8613

Epoch 23/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8540

Epoch 24/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0061 Acc: 0.8686

Epoch 25/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0061 Acc: 0.8686

Epoch 26/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8686

Epoch 27/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8686

Epoch 28/81
----------
train Loss: 0.0001 Acc: 0.9945
val Loss: 0.0065 Acc: 0.8686

Epoch 29/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0054 Acc: 0.8686

Epoch 30/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0062 Acc: 0.8613

Epoch 31/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0058 Acc: 0.8540

Epoch 32/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0066 Acc: 0.8613

Epoch 33/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8613

Epoch 34/81
----------
train Loss: 0.0002 Acc: 0.9945
val Loss: 0.0060 Acc: 0.8686

Epoch 35/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8686

Epoch 36/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8759

Epoch 37/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0061 Acc: 0.8759

Epoch 38/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 39/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0069 Acc: 0.8686

Epoch 40/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8686

Epoch 41/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0062 Acc: 0.8686

Epoch 42/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8686

Epoch 43/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8686

Epoch 44/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0072 Acc: 0.8686

Epoch 45/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 46/81
----------
train Loss: 0.0002 Acc: 0.9964
val Loss: 0.0059 Acc: 0.8686

Epoch 47/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8686

Epoch 48/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8686

Epoch 49/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8686

Epoch 50/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8613

Epoch 51/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0072 Acc: 0.8613

Epoch 52/81
----------
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0066 Acc: 0.8686

Epoch 53/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 54/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8686

Epoch 55/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8686

Epoch 56/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0058 Acc: 0.8686

Epoch 57/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8686

Epoch 58/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8686

Epoch 59/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8686

Epoch 60/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8686

Epoch 61/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8686

Epoch 62/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8686

Epoch 63/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 64/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0069 Acc: 0.8540

Epoch 65/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8613

Epoch 66/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0058 Acc: 0.8686

Epoch 67/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8686

Epoch 68/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 69/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 70/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8686

Epoch 71/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0051 Acc: 0.8686

Epoch 72/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8686

Epoch 73/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0065 Acc: 0.8686

Epoch 74/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8686

Epoch 75/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0066 Acc: 0.8686

Epoch 76/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8613

Epoch 77/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8686

Epoch 78/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 79/81
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8759

Epoch 80/81
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9982
val Loss: 0.0045 Acc: 0.8686

Epoch 81/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8759

Training complete in 4m 15s
Best val Acc: 0.875912

---Testing---
Test accuracy: 0.975255
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 96 %
mean: 0.966173957621326, std: 0.03060866679797239
--------------------

run info[val: 0.25, epoch: 64, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0231 Acc: 0.2209
val Loss: 0.0195 Acc: 0.3509

Epoch 1/63
----------
train Loss: 0.0158 Acc: 0.5504
val Loss: 0.0124 Acc: 0.6901

Epoch 2/63
----------
train Loss: 0.0101 Acc: 0.6899
val Loss: 0.0089 Acc: 0.7427

Epoch 3/63
----------
LR is set to 0.001
train Loss: 0.0078 Acc: 0.8275
val Loss: 0.0085 Acc: 0.7427

Epoch 4/63
----------
train Loss: 0.0072 Acc: 0.8411
val Loss: 0.0081 Acc: 0.7719

Epoch 5/63
----------
train Loss: 0.0074 Acc: 0.8236
val Loss: 0.0080 Acc: 0.7953

Epoch 6/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0069 Acc: 0.8508
val Loss: 0.0081 Acc: 0.7953

Epoch 7/63
----------
train Loss: 0.0069 Acc: 0.8314
val Loss: 0.0082 Acc: 0.8012

Epoch 8/63
----------
train Loss: 0.0078 Acc: 0.8256
val Loss: 0.0078 Acc: 0.7953

Epoch 9/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0064 Acc: 0.8547
val Loss: 0.0081 Acc: 0.7953

Epoch 10/63
----------
train Loss: 0.0069 Acc: 0.8624
val Loss: 0.0081 Acc: 0.7836

Epoch 11/63
----------
train Loss: 0.0065 Acc: 0.8450
val Loss: 0.0081 Acc: 0.7895

Epoch 12/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0069 Acc: 0.8450
val Loss: 0.0080 Acc: 0.7953

Epoch 13/63
----------
train Loss: 0.0069 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7953

Epoch 14/63
----------
train Loss: 0.0068 Acc: 0.8488
val Loss: 0.0081 Acc: 0.7895

Epoch 15/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0066 Acc: 0.8547
val Loss: 0.0079 Acc: 0.7895

Epoch 16/63
----------
train Loss: 0.0070 Acc: 0.8547
val Loss: 0.0079 Acc: 0.7953

Epoch 17/63
----------
train Loss: 0.0067 Acc: 0.8450
val Loss: 0.0080 Acc: 0.7836

Epoch 18/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0065 Acc: 0.8527
val Loss: 0.0078 Acc: 0.7895

Epoch 19/63
----------
train Loss: 0.0066 Acc: 0.8508
val Loss: 0.0080 Acc: 0.7836

Epoch 20/63
----------
train Loss: 0.0072 Acc: 0.8605
val Loss: 0.0080 Acc: 0.7836

Epoch 21/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0066 Acc: 0.8450
val Loss: 0.0081 Acc: 0.7719

Epoch 22/63
----------
train Loss: 0.0075 Acc: 0.8372
val Loss: 0.0080 Acc: 0.7895

Epoch 23/63
----------
train Loss: 0.0063 Acc: 0.8547
val Loss: 0.0082 Acc: 0.7836

Epoch 24/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0063 Acc: 0.8527
val Loss: 0.0080 Acc: 0.7895

Epoch 25/63
----------
train Loss: 0.0068 Acc: 0.8508
val Loss: 0.0082 Acc: 0.7836

Epoch 26/63
----------
train Loss: 0.0065 Acc: 0.8585
val Loss: 0.0079 Acc: 0.7895

Epoch 27/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0068 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7953

Epoch 28/63
----------
train Loss: 0.0068 Acc: 0.8411
val Loss: 0.0079 Acc: 0.7836

Epoch 29/63
----------
train Loss: 0.0071 Acc: 0.8430
val Loss: 0.0079 Acc: 0.7778

Epoch 30/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0070 Acc: 0.8508
val Loss: 0.0080 Acc: 0.7895

Epoch 31/63
----------
train Loss: 0.0069 Acc: 0.8469
val Loss: 0.0081 Acc: 0.7953

Epoch 32/63
----------
train Loss: 0.0071 Acc: 0.8295
val Loss: 0.0081 Acc: 0.7895

Epoch 33/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0065 Acc: 0.8430
val Loss: 0.0079 Acc: 0.7895

Epoch 34/63
----------
train Loss: 0.0072 Acc: 0.8508
val Loss: 0.0081 Acc: 0.7836

Epoch 35/63
----------
train Loss: 0.0067 Acc: 0.8566
val Loss: 0.0081 Acc: 0.7836

Epoch 36/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0068 Acc: 0.8702
val Loss: 0.0080 Acc: 0.7836

Epoch 37/63
----------
train Loss: 0.0070 Acc: 0.8527
val Loss: 0.0079 Acc: 0.7778

Epoch 38/63
----------
train Loss: 0.0068 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7836

Epoch 39/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0067 Acc: 0.8508
val Loss: 0.0081 Acc: 0.7895

Epoch 40/63
----------
train Loss: 0.0067 Acc: 0.8721
val Loss: 0.0079 Acc: 0.7895

Epoch 41/63
----------
train Loss: 0.0067 Acc: 0.8547
val Loss: 0.0080 Acc: 0.7953

Epoch 42/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0065 Acc: 0.8488
val Loss: 0.0080 Acc: 0.7895

Epoch 43/63
----------
train Loss: 0.0071 Acc: 0.8372
val Loss: 0.0080 Acc: 0.7895

Epoch 44/63
----------
train Loss: 0.0069 Acc: 0.8411
val Loss: 0.0079 Acc: 0.7836

Epoch 45/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0067 Acc: 0.8411
val Loss: 0.0084 Acc: 0.7836

Epoch 46/63
----------
train Loss: 0.0067 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7895

Epoch 47/63
----------
train Loss: 0.0070 Acc: 0.8585
val Loss: 0.0081 Acc: 0.7778

Epoch 48/63
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0070 Acc: 0.8624
val Loss: 0.0079 Acc: 0.7778

Epoch 49/63
----------
train Loss: 0.0063 Acc: 0.8682
val Loss: 0.0080 Acc: 0.8070

Epoch 50/63
----------
train Loss: 0.0068 Acc: 0.8624
val Loss: 0.0080 Acc: 0.7895

Epoch 51/63
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0064 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7953

Epoch 52/63
----------
train Loss: 0.0071 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7895

Epoch 53/63
----------
train Loss: 0.0076 Acc: 0.8314
val Loss: 0.0081 Acc: 0.7836

Epoch 54/63
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0067 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7719

Epoch 55/63
----------
train Loss: 0.0066 Acc: 0.8391
val Loss: 0.0079 Acc: 0.7953

Epoch 56/63
----------
train Loss: 0.0069 Acc: 0.8295
val Loss: 0.0079 Acc: 0.7895

Epoch 57/63
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0065 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7953

Epoch 58/63
----------
train Loss: 0.0068 Acc: 0.8721
val Loss: 0.0080 Acc: 0.7895

Epoch 59/63
----------
train Loss: 0.0066 Acc: 0.8450
val Loss: 0.0080 Acc: 0.7895

Epoch 60/63
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0065 Acc: 0.8566
val Loss: 0.0081 Acc: 0.7953

Epoch 61/63
----------
train Loss: 0.0066 Acc: 0.8430
val Loss: 0.0080 Acc: 0.8012

Epoch 62/63
----------
train Loss: 0.0071 Acc: 0.8605
val Loss: 0.0079 Acc: 0.7895

Epoch 63/63
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0070 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7836

Training complete in 3m 11s
Best val Acc: 0.807018

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0060 Acc: 0.8760
val Loss: 0.0064 Acc: 0.8363

Epoch 1/63
----------
train Loss: 0.0039 Acc: 0.9147
val Loss: 0.0064 Acc: 0.8363

Epoch 2/63
----------
train Loss: 0.0026 Acc: 0.9457
val Loss: 0.0053 Acc: 0.8187

Epoch 3/63
----------
LR is set to 0.001
train Loss: 0.0012 Acc: 0.9806
val Loss: 0.0049 Acc: 0.8713

Epoch 4/63
----------
train Loss: 0.0012 Acc: 0.9864
val Loss: 0.0045 Acc: 0.8713

Epoch 5/63
----------
train Loss: 0.0013 Acc: 0.9826
val Loss: 0.0049 Acc: 0.8772

Epoch 6/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9806
val Loss: 0.0044 Acc: 0.8772

Epoch 7/63
----------
train Loss: 0.0010 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8713

Epoch 8/63
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0047 Acc: 0.8655

Epoch 9/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0013 Acc: 0.9767
val Loss: 0.0046 Acc: 0.8713

Epoch 10/63
----------
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8655

Epoch 11/63
----------
train Loss: 0.0008 Acc: 0.9826
val Loss: 0.0048 Acc: 0.8713

Epoch 12/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9942
val Loss: 0.0046 Acc: 0.8596

Epoch 13/63
----------
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0046 Acc: 0.8655

Epoch 14/63
----------
train Loss: 0.0010 Acc: 0.9922
val Loss: 0.0044 Acc: 0.8596

Epoch 15/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0010 Acc: 0.9981
val Loss: 0.0046 Acc: 0.8596

Epoch 16/63
----------
train Loss: 0.0012 Acc: 0.9884
val Loss: 0.0049 Acc: 0.8655

Epoch 17/63
----------
train Loss: 0.0013 Acc: 0.9903
val Loss: 0.0048 Acc: 0.8772

Epoch 18/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9884
val Loss: 0.0048 Acc: 0.8713

Epoch 19/63
----------
train Loss: 0.0013 Acc: 0.9922
val Loss: 0.0047 Acc: 0.8655

Epoch 20/63
----------
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0049 Acc: 0.8772

Epoch 21/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0046 Acc: 0.8713

Epoch 22/63
----------
train Loss: 0.0009 Acc: 0.9942
val Loss: 0.0048 Acc: 0.8772

Epoch 23/63
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0049 Acc: 0.8772

Epoch 24/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0051 Acc: 0.8596

Epoch 25/63
----------
train Loss: 0.0011 Acc: 0.9922
val Loss: 0.0048 Acc: 0.8596

Epoch 26/63
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8538

Epoch 27/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0048 Acc: 0.8713

Epoch 28/63
----------
train Loss: 0.0011 Acc: 0.9922
val Loss: 0.0047 Acc: 0.8772

Epoch 29/63
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0049 Acc: 0.8655

Epoch 30/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0011 Acc: 0.9903
val Loss: 0.0046 Acc: 0.8596

Epoch 31/63
----------
train Loss: 0.0011 Acc: 0.9903
val Loss: 0.0046 Acc: 0.8596

Epoch 32/63
----------
train Loss: 0.0012 Acc: 0.9845
val Loss: 0.0046 Acc: 0.8596

Epoch 33/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0044 Acc: 0.8596

Epoch 34/63
----------
train Loss: 0.0017 Acc: 0.9826
val Loss: 0.0046 Acc: 0.8713

Epoch 35/63
----------
train Loss: 0.0015 Acc: 0.9767
val Loss: 0.0048 Acc: 0.8772

Epoch 36/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0015 Acc: 0.9903
val Loss: 0.0049 Acc: 0.8655

Epoch 37/63
----------
train Loss: 0.0012 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8655

Epoch 38/63
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0045 Acc: 0.8596

Epoch 39/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0014 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8655

Epoch 40/63
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8596

Epoch 41/63
----------
train Loss: 0.0012 Acc: 0.9845
val Loss: 0.0046 Acc: 0.8772

Epoch 42/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0012 Acc: 0.9864
val Loss: 0.0050 Acc: 0.8655

Epoch 43/63
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0048 Acc: 0.8655

Epoch 44/63
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0048 Acc: 0.8713

Epoch 45/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0015 Acc: 0.9787
val Loss: 0.0047 Acc: 0.8713

Epoch 46/63
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0047 Acc: 0.8655

Epoch 47/63
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8538

Epoch 48/63
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0012 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8655

Epoch 49/63
----------
train Loss: 0.0011 Acc: 0.9903
val Loss: 0.0046 Acc: 0.8772

Epoch 50/63
----------
train Loss: 0.0016 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8772

Epoch 51/63
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0013 Acc: 0.9767
val Loss: 0.0048 Acc: 0.8713

Epoch 52/63
----------
train Loss: 0.0011 Acc: 0.9826
val Loss: 0.0048 Acc: 0.8713

Epoch 53/63
----------
train Loss: 0.0011 Acc: 0.9903
val Loss: 0.0046 Acc: 0.8655

Epoch 54/63
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0049 Acc: 0.8655

Epoch 55/63
----------
train Loss: 0.0012 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8655

Epoch 56/63
----------
train Loss: 0.0011 Acc: 0.9884
val Loss: 0.0045 Acc: 0.8655

Epoch 57/63
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0014 Acc: 0.9787
val Loss: 0.0046 Acc: 0.8596

Epoch 58/63
----------
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0047 Acc: 0.8713

Epoch 59/63
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0044 Acc: 0.8772

Epoch 60/63
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0049 Acc: 0.8655

Epoch 61/63
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0049 Acc: 0.8655

Epoch 62/63
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8655

Epoch 63/63
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0012 Acc: 0.9826
val Loss: 0.0046 Acc: 0.8772

Training complete in 3m 19s
Best val Acc: 0.877193

---Testing---
Test accuracy: 0.965066
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 90 %
Accuracy of Lamniformes : 89 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9530048772021573, std: 0.040544207042291155
--------------------

run info[val: 0.3, epoch: 85, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0233 Acc: 0.1455
val Loss: 0.0271 Acc: 0.3495

Epoch 1/84
----------
train Loss: 0.0166 Acc: 0.4428
val Loss: 0.0197 Acc: 0.5485

Epoch 2/84
----------
train Loss: 0.0110 Acc: 0.7069
val Loss: 0.0126 Acc: 0.7087

Epoch 3/84
----------
train Loss: 0.0086 Acc: 0.7318
val Loss: 0.0106 Acc: 0.7767

Epoch 4/84
----------
train Loss: 0.0062 Acc: 0.8316
val Loss: 0.0091 Acc: 0.7670

Epoch 5/84
----------
LR is set to 0.001
train Loss: 0.0055 Acc: 0.8170
val Loss: 0.0076 Acc: 0.7913

Epoch 6/84
----------
train Loss: 0.0050 Acc: 0.8545
val Loss: 0.0084 Acc: 0.8010

Epoch 7/84
----------
train Loss: 0.0050 Acc: 0.8773
val Loss: 0.0088 Acc: 0.8107

Epoch 8/84
----------
train Loss: 0.0049 Acc: 0.8773
val Loss: 0.0088 Acc: 0.8155

Epoch 9/84
----------
train Loss: 0.0048 Acc: 0.8773
val Loss: 0.0079 Acc: 0.8204

Epoch 10/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0048 Acc: 0.8753
val Loss: 0.0109 Acc: 0.8204

Epoch 11/84
----------
train Loss: 0.0048 Acc: 0.8815
val Loss: 0.0075 Acc: 0.8204

Epoch 12/84
----------
train Loss: 0.0047 Acc: 0.8898
val Loss: 0.0083 Acc: 0.8204

Epoch 13/84
----------
train Loss: 0.0049 Acc: 0.8711
val Loss: 0.0088 Acc: 0.8155

Epoch 14/84
----------
train Loss: 0.0048 Acc: 0.8628
val Loss: 0.0064 Acc: 0.8155

Epoch 15/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0047 Acc: 0.8898
val Loss: 0.0087 Acc: 0.8204

Epoch 16/84
----------
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0072 Acc: 0.8155

Epoch 17/84
----------
train Loss: 0.0048 Acc: 0.8753
val Loss: 0.0073 Acc: 0.8204

Epoch 18/84
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0090 Acc: 0.8155

Epoch 19/84
----------
train Loss: 0.0047 Acc: 0.8628
val Loss: 0.0094 Acc: 0.8155

Epoch 20/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0047 Acc: 0.8836
val Loss: 0.0071 Acc: 0.8204

Epoch 21/84
----------
train Loss: 0.0049 Acc: 0.8690
val Loss: 0.0075 Acc: 0.8204

Epoch 22/84
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0072 Acc: 0.8204

Epoch 23/84
----------
train Loss: 0.0047 Acc: 0.8711
val Loss: 0.0073 Acc: 0.8155

Epoch 24/84
----------
train Loss: 0.0047 Acc: 0.8669
val Loss: 0.0092 Acc: 0.8204

Epoch 25/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0048 Acc: 0.8711
val Loss: 0.0097 Acc: 0.8155

Epoch 26/84
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0069 Acc: 0.8155

Epoch 27/84
----------
train Loss: 0.0047 Acc: 0.8732
val Loss: 0.0102 Acc: 0.8204

Epoch 28/84
----------
train Loss: 0.0048 Acc: 0.8649
val Loss: 0.0075 Acc: 0.8204

Epoch 29/84
----------
train Loss: 0.0045 Acc: 0.8753
val Loss: 0.0079 Acc: 0.8204

Epoch 30/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0046 Acc: 0.8794
val Loss: 0.0075 Acc: 0.8204

Epoch 31/84
----------
train Loss: 0.0044 Acc: 0.8815
val Loss: 0.0067 Acc: 0.8204

Epoch 32/84
----------
train Loss: 0.0047 Acc: 0.8732
val Loss: 0.0095 Acc: 0.8155

Epoch 33/84
----------
train Loss: 0.0047 Acc: 0.8711
val Loss: 0.0082 Acc: 0.8155

Epoch 34/84
----------
train Loss: 0.0048 Acc: 0.8711
val Loss: 0.0079 Acc: 0.8155

Epoch 35/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0094 Acc: 0.8155

Epoch 36/84
----------
train Loss: 0.0046 Acc: 0.8753
val Loss: 0.0081 Acc: 0.8155

Epoch 37/84
----------
train Loss: 0.0048 Acc: 0.8711
val Loss: 0.0072 Acc: 0.8155

Epoch 38/84
----------
train Loss: 0.0048 Acc: 0.8794
val Loss: 0.0073 Acc: 0.8155

Epoch 39/84
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0072 Acc: 0.8155

Epoch 40/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0077 Acc: 0.8155

Epoch 41/84
----------
train Loss: 0.0047 Acc: 0.8877
val Loss: 0.0080 Acc: 0.8155

Epoch 42/84
----------
train Loss: 0.0047 Acc: 0.8732
val Loss: 0.0108 Acc: 0.8155

Epoch 43/84
----------
train Loss: 0.0048 Acc: 0.8628
val Loss: 0.0089 Acc: 0.8155

Epoch 44/84
----------
train Loss: 0.0047 Acc: 0.8732
val Loss: 0.0102 Acc: 0.8155

Epoch 45/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0065 Acc: 0.8155

Epoch 46/84
----------
train Loss: 0.0045 Acc: 0.8711
val Loss: 0.0081 Acc: 0.8155

Epoch 47/84
----------
train Loss: 0.0048 Acc: 0.8753
val Loss: 0.0096 Acc: 0.8155

Epoch 48/84
----------
train Loss: 0.0046 Acc: 0.8690
val Loss: 0.0081 Acc: 0.8204

Epoch 49/84
----------
train Loss: 0.0045 Acc: 0.8753
val Loss: 0.0070 Acc: 0.8155

Epoch 50/84
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0073 Acc: 0.8155

Epoch 51/84
----------
train Loss: 0.0045 Acc: 0.8732
val Loss: 0.0079 Acc: 0.8155

Epoch 52/84
----------
train Loss: 0.0047 Acc: 0.8607
val Loss: 0.0079 Acc: 0.8155

Epoch 53/84
----------
train Loss: 0.0046 Acc: 0.8753
val Loss: 0.0064 Acc: 0.8155

Epoch 54/84
----------
train Loss: 0.0047 Acc: 0.8836
val Loss: 0.0069 Acc: 0.8204

Epoch 55/84
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0047 Acc: 0.8857
val Loss: 0.0100 Acc: 0.8204

Epoch 56/84
----------
train Loss: 0.0047 Acc: 0.8794
val Loss: 0.0093 Acc: 0.8204

Epoch 57/84
----------
train Loss: 0.0046 Acc: 0.8732
val Loss: 0.0063 Acc: 0.8204

Epoch 58/84
----------
train Loss: 0.0050 Acc: 0.8649
val Loss: 0.0087 Acc: 0.8155

Epoch 59/84
----------
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0095 Acc: 0.8155

Epoch 60/84
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0045 Acc: 0.8898
val Loss: 0.0074 Acc: 0.8204

Epoch 61/84
----------
train Loss: 0.0046 Acc: 0.8815
val Loss: 0.0077 Acc: 0.8155

Epoch 62/84
----------
train Loss: 0.0047 Acc: 0.8815
val Loss: 0.0087 Acc: 0.8155

Epoch 63/84
----------
train Loss: 0.0045 Acc: 0.8815
val Loss: 0.0076 Acc: 0.8155

Epoch 64/84
----------
train Loss: 0.0045 Acc: 0.8981
val Loss: 0.0077 Acc: 0.8155

Epoch 65/84
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0046 Acc: 0.8794
val Loss: 0.0078 Acc: 0.8155

Epoch 66/84
----------
train Loss: 0.0049 Acc: 0.8753
val Loss: 0.0085 Acc: 0.8155

Epoch 67/84
----------
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0088 Acc: 0.8155

Epoch 68/84
----------
train Loss: 0.0048 Acc: 0.8628
val Loss: 0.0087 Acc: 0.8155

Epoch 69/84
----------
train Loss: 0.0049 Acc: 0.8586
val Loss: 0.0096 Acc: 0.8204

Epoch 70/84
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0048 Acc: 0.8815
val Loss: 0.0080 Acc: 0.8155

Epoch 71/84
----------
train Loss: 0.0049 Acc: 0.8649
val Loss: 0.0066 Acc: 0.8155

Epoch 72/84
----------
train Loss: 0.0046 Acc: 0.8669
val Loss: 0.0085 Acc: 0.8204

Epoch 73/84
----------
train Loss: 0.0047 Acc: 0.8857
val Loss: 0.0091 Acc: 0.8204

Epoch 74/84
----------
train Loss: 0.0048 Acc: 0.8628
val Loss: 0.0069 Acc: 0.8204

Epoch 75/84
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0048 Acc: 0.8773
val Loss: 0.0082 Acc: 0.8155

Epoch 76/84
----------
train Loss: 0.0046 Acc: 0.8857
val Loss: 0.0095 Acc: 0.8155

Epoch 77/84
----------
train Loss: 0.0049 Acc: 0.8711
val Loss: 0.0097 Acc: 0.8155

Epoch 78/84
----------
train Loss: 0.0048 Acc: 0.8711
val Loss: 0.0097 Acc: 0.8204

Epoch 79/84
----------
train Loss: 0.0048 Acc: 0.8669
val Loss: 0.0078 Acc: 0.8204

Epoch 80/84
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0048 Acc: 0.8794
val Loss: 0.0102 Acc: 0.8155

Epoch 81/84
----------
train Loss: 0.0048 Acc: 0.8690
val Loss: 0.0085 Acc: 0.8155

Epoch 82/84
----------
train Loss: 0.0044 Acc: 0.8940
val Loss: 0.0073 Acc: 0.8204

Epoch 83/84
----------
train Loss: 0.0050 Acc: 0.8524
val Loss: 0.0088 Acc: 0.8204

Epoch 84/84
----------
train Loss: 0.0047 Acc: 0.8711
val Loss: 0.0070 Acc: 0.8204

Training complete in 4m 8s
Best val Acc: 0.820388

---Fine tuning.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0047 Acc: 0.8753
val Loss: 0.0069 Acc: 0.8252

Epoch 1/84
----------
train Loss: 0.0027 Acc: 0.9501
val Loss: 0.0077 Acc: 0.8689

Epoch 2/84
----------
train Loss: 0.0016 Acc: 0.9709
val Loss: 0.0059 Acc: 0.8786

Epoch 3/84
----------
train Loss: 0.0010 Acc: 0.9792
val Loss: 0.0045 Acc: 0.8689

Epoch 4/84
----------
train Loss: 0.0006 Acc: 0.9896
val Loss: 0.0041 Acc: 0.8738

Epoch 5/84
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0079 Acc: 0.8786

Epoch 6/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8786

Epoch 7/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0037 Acc: 0.8883

Epoch 8/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0054 Acc: 0.8883

Epoch 9/84
----------
train Loss: 0.0004 Acc: 0.9979
val Loss: 0.0035 Acc: 0.8932

Epoch 10/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9938
val Loss: 0.0043 Acc: 0.8883

Epoch 11/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0047 Acc: 0.8883

Epoch 12/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0038 Acc: 0.8883

Epoch 13/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0040 Acc: 0.8883

Epoch 14/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0049 Acc: 0.8883

Epoch 15/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0041 Acc: 0.8883

Epoch 16/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8883

Epoch 17/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 18/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 19/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0037 Acc: 0.8883

Epoch 20/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 21/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8883

Epoch 22/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0065 Acc: 0.8883

Epoch 23/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0035 Acc: 0.8883

Epoch 24/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8883

Epoch 25/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 26/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0049 Acc: 0.8883

Epoch 27/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8883

Epoch 28/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 29/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8883

Epoch 30/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8883

Epoch 31/84
----------
train Loss: 0.0004 Acc: 0.9979
val Loss: 0.0058 Acc: 0.8883

Epoch 32/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0068 Acc: 0.8883

Epoch 33/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8883

Epoch 34/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0040 Acc: 0.8883

Epoch 35/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0054 Acc: 0.8883

Epoch 36/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0064 Acc: 0.8883

Epoch 37/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8883

Epoch 38/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 39/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 40/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0047 Acc: 0.8883

Epoch 41/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8883

Epoch 42/84
----------
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0038 Acc: 0.8883

Epoch 43/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8883

Epoch 44/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0081 Acc: 0.8883

Epoch 45/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 46/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8883

Epoch 47/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0106 Acc: 0.8883

Epoch 48/84
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0087 Acc: 0.8883

Epoch 49/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0053 Acc: 0.8883

Epoch 50/84
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0078 Acc: 0.8883

Epoch 51/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8883

Epoch 52/84
----------
train Loss: 0.0004 Acc: 0.9938
val Loss: 0.0034 Acc: 0.8883

Epoch 53/84
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0114 Acc: 0.8883

Epoch 54/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0047 Acc: 0.8883

Epoch 55/84
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0043 Acc: 0.8883

Epoch 56/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0035 Acc: 0.8883

Epoch 57/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8883

Epoch 58/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0070 Acc: 0.8883

Epoch 59/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0076 Acc: 0.8883

Epoch 60/84
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0046 Acc: 0.8883

Epoch 61/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8883

Epoch 62/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0040 Acc: 0.8883

Epoch 63/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8883

Epoch 64/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8835

Epoch 65/84
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0060 Acc: 0.8883

Epoch 66/84
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0111 Acc: 0.8883

Epoch 67/84
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0057 Acc: 0.8883

Epoch 68/84
----------
train Loss: 0.0004 Acc: 0.9979
val Loss: 0.0046 Acc: 0.8883

Epoch 69/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8883

Epoch 70/84
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0035 Acc: 0.8883

Epoch 71/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8883

Epoch 72/84
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 73/84
----------
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0043 Acc: 0.8883

Epoch 74/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8883

Epoch 75/84
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0068 Acc: 0.8883

Epoch 76/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 77/84
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 78/84
----------
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0058 Acc: 0.8883

Epoch 79/84
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0063 Acc: 0.8883

Epoch 80/84
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8883

Epoch 81/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8835

Epoch 82/84
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0038 Acc: 0.8883

Epoch 83/84
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 84/84
----------
train Loss: 0.0004 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8883

Training complete in 4m 19s
Best val Acc: 0.893204

---Testing---
Test accuracy: 0.967977
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 98 %
mean: 0.9574120991160022, std: 0.044923247671145215

Model saved in "./weights/shark_[0.98]_mean[0.97]_std[0.03].save".
--------------------

run info[val: 0.1, epoch: 93, randcrop: False, decay: 7]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0223 Acc: 0.2423
val Loss: 0.0233 Acc: 0.5294

Epoch 1/92
----------
train Loss: 0.0147 Acc: 0.5105
val Loss: 0.0193 Acc: 0.5294

Epoch 2/92
----------
train Loss: 0.0091 Acc: 0.7318
val Loss: 0.0107 Acc: 0.7794

Epoch 3/92
----------
train Loss: 0.0059 Acc: 0.8417
val Loss: 0.0135 Acc: 0.6618

Epoch 4/92
----------
train Loss: 0.0050 Acc: 0.8546
val Loss: 0.0093 Acc: 0.7794

Epoch 5/92
----------
train Loss: 0.0039 Acc: 0.9079
val Loss: 0.0116 Acc: 0.7059

Epoch 6/92
----------
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0091 Acc: 0.7647

Epoch 7/92
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.9225
val Loss: 0.0091 Acc: 0.7794

Epoch 8/92
----------
train Loss: 0.0030 Acc: 0.9483
val Loss: 0.0094 Acc: 0.7794

Epoch 9/92
----------
train Loss: 0.0029 Acc: 0.9418
val Loss: 0.0100 Acc: 0.7500

Epoch 10/92
----------
train Loss: 0.0033 Acc: 0.9257
val Loss: 0.0102 Acc: 0.7500

Epoch 11/92
----------
train Loss: 0.0028 Acc: 0.9321
val Loss: 0.0097 Acc: 0.7794

Epoch 12/92
----------
train Loss: 0.0029 Acc: 0.9499
val Loss: 0.0097 Acc: 0.7941

Epoch 13/92
----------
train Loss: 0.0033 Acc: 0.9386
val Loss: 0.0095 Acc: 0.7941

Epoch 14/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9515
val Loss: 0.0095 Acc: 0.7941

Epoch 15/92
----------
train Loss: 0.0027 Acc: 0.9499
val Loss: 0.0095 Acc: 0.7941

Epoch 16/92
----------
train Loss: 0.0031 Acc: 0.9483
val Loss: 0.0095 Acc: 0.7794

Epoch 17/92
----------
train Loss: 0.0027 Acc: 0.9370
val Loss: 0.0095 Acc: 0.7794

Epoch 18/92
----------
train Loss: 0.0028 Acc: 0.9354
val Loss: 0.0096 Acc: 0.7941

Epoch 19/92
----------
train Loss: 0.0026 Acc: 0.9499
val Loss: 0.0097 Acc: 0.7647

Epoch 20/92
----------
train Loss: 0.0028 Acc: 0.9483
val Loss: 0.0096 Acc: 0.7647

Epoch 21/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0028 Acc: 0.9499
val Loss: 0.0097 Acc: 0.7647

Epoch 22/92
----------
train Loss: 0.0027 Acc: 0.9532
val Loss: 0.0096 Acc: 0.7794

Epoch 23/92
----------
train Loss: 0.0026 Acc: 0.9515
val Loss: 0.0096 Acc: 0.7794

Epoch 24/92
----------
train Loss: 0.0028 Acc: 0.9451
val Loss: 0.0096 Acc: 0.7794

Epoch 25/92
----------
train Loss: 0.0027 Acc: 0.9515
val Loss: 0.0096 Acc: 0.7794

Epoch 26/92
----------
train Loss: 0.0028 Acc: 0.9435
val Loss: 0.0096 Acc: 0.7794

Epoch 27/92
----------
train Loss: 0.0028 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 28/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0026 Acc: 0.9499
val Loss: 0.0096 Acc: 0.7941

Epoch 29/92
----------
train Loss: 0.0027 Acc: 0.9386
val Loss: 0.0096 Acc: 0.7794

Epoch 30/92
----------
train Loss: 0.0035 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7794

Epoch 31/92
----------
train Loss: 0.0031 Acc: 0.9402
val Loss: 0.0096 Acc: 0.7794

Epoch 32/92
----------
train Loss: 0.0027 Acc: 0.9451
val Loss: 0.0095 Acc: 0.7941

Epoch 33/92
----------
train Loss: 0.0028 Acc: 0.9532
val Loss: 0.0096 Acc: 0.7941

Epoch 34/92
----------
train Loss: 0.0029 Acc: 0.9451
val Loss: 0.0096 Acc: 0.7941

Epoch 35/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9418
val Loss: 0.0096 Acc: 0.7941

Epoch 36/92
----------
train Loss: 0.0028 Acc: 0.9532
val Loss: 0.0096 Acc: 0.7794

Epoch 37/92
----------
train Loss: 0.0027 Acc: 0.9564
val Loss: 0.0096 Acc: 0.7794

Epoch 38/92
----------
train Loss: 0.0035 Acc: 0.9386
val Loss: 0.0096 Acc: 0.7794

Epoch 39/92
----------
train Loss: 0.0029 Acc: 0.9515
val Loss: 0.0096 Acc: 0.7941

Epoch 40/92
----------
train Loss: 0.0030 Acc: 0.9451
val Loss: 0.0096 Acc: 0.7794

Epoch 41/92
----------
train Loss: 0.0033 Acc: 0.9354
val Loss: 0.0098 Acc: 0.7794

Epoch 42/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0028 Acc: 0.9418
val Loss: 0.0098 Acc: 0.7647

Epoch 43/92
----------
train Loss: 0.0028 Acc: 0.9402
val Loss: 0.0096 Acc: 0.7794

Epoch 44/92
----------
train Loss: 0.0029 Acc: 0.9515
val Loss: 0.0096 Acc: 0.7794

Epoch 45/92
----------
train Loss: 0.0035 Acc: 0.9418
val Loss: 0.0095 Acc: 0.7941

Epoch 46/92
----------
train Loss: 0.0029 Acc: 0.9435
val Loss: 0.0096 Acc: 0.7941

Epoch 47/92
----------
train Loss: 0.0027 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 48/92
----------
train Loss: 0.0035 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 49/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0031 Acc: 0.9354
val Loss: 0.0095 Acc: 0.7647

Epoch 50/92
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0095 Acc: 0.7794

Epoch 51/92
----------
train Loss: 0.0027 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7941

Epoch 52/92
----------
train Loss: 0.0028 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7941

Epoch 53/92
----------
train Loss: 0.0028 Acc: 0.9370
val Loss: 0.0095 Acc: 0.7794

Epoch 54/92
----------
train Loss: 0.0028 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7794

Epoch 55/92
----------
train Loss: 0.0032 Acc: 0.9402
val Loss: 0.0095 Acc: 0.7794

Epoch 56/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0032 Acc: 0.9467
val Loss: 0.0095 Acc: 0.7794

Epoch 57/92
----------
train Loss: 0.0027 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7941

Epoch 58/92
----------
train Loss: 0.0029 Acc: 0.9435
val Loss: 0.0096 Acc: 0.7794

Epoch 59/92
----------
train Loss: 0.0032 Acc: 0.9499
val Loss: 0.0096 Acc: 0.7794

Epoch 60/92
----------
train Loss: 0.0027 Acc: 0.9451
val Loss: 0.0096 Acc: 0.7794

Epoch 61/92
----------
train Loss: 0.0028 Acc: 0.9483
val Loss: 0.0097 Acc: 0.7794

Epoch 62/92
----------
train Loss: 0.0030 Acc: 0.9354
val Loss: 0.0096 Acc: 0.7794

Epoch 63/92
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0028 Acc: 0.9499
val Loss: 0.0096 Acc: 0.7941

Epoch 64/92
----------
train Loss: 0.0031 Acc: 0.9435
val Loss: 0.0097 Acc: 0.7794

Epoch 65/92
----------
train Loss: 0.0028 Acc: 0.9402
val Loss: 0.0096 Acc: 0.7794

Epoch 66/92
----------
train Loss: 0.0028 Acc: 0.9435
val Loss: 0.0097 Acc: 0.7794

Epoch 67/92
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0096 Acc: 0.7794

Epoch 68/92
----------
train Loss: 0.0027 Acc: 0.9515
val Loss: 0.0096 Acc: 0.7794

Epoch 69/92
----------
train Loss: 0.0025 Acc: 0.9580
val Loss: 0.0097 Acc: 0.7794

Epoch 70/92
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0029 Acc: 0.9515
val Loss: 0.0097 Acc: 0.7794

Epoch 71/92
----------
train Loss: 0.0030 Acc: 0.9564
val Loss: 0.0097 Acc: 0.7794

Epoch 72/92
----------
train Loss: 0.0028 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 73/92
----------
train Loss: 0.0027 Acc: 0.9499
val Loss: 0.0096 Acc: 0.7794

Epoch 74/92
----------
train Loss: 0.0031 Acc: 0.9305
val Loss: 0.0096 Acc: 0.7794

Epoch 75/92
----------
train Loss: 0.0030 Acc: 0.9451
val Loss: 0.0099 Acc: 0.7500

Epoch 76/92
----------
train Loss: 0.0032 Acc: 0.9321
val Loss: 0.0098 Acc: 0.7500

Epoch 77/92
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0028 Acc: 0.9515
val Loss: 0.0099 Acc: 0.7647

Epoch 78/92
----------
train Loss: 0.0030 Acc: 0.9370
val Loss: 0.0097 Acc: 0.7794

Epoch 79/92
----------
train Loss: 0.0028 Acc: 0.9483
val Loss: 0.0097 Acc: 0.7794

Epoch 80/92
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0096 Acc: 0.7794

Epoch 81/92
----------
train Loss: 0.0028 Acc: 0.9370
val Loss: 0.0096 Acc: 0.7794

Epoch 82/92
----------
train Loss: 0.0029 Acc: 0.9386
val Loss: 0.0096 Acc: 0.7794

Epoch 83/92
----------
train Loss: 0.0031 Acc: 0.9386
val Loss: 0.0095 Acc: 0.7794

Epoch 84/92
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0029 Acc: 0.9483
val Loss: 0.0096 Acc: 0.7794

Epoch 85/92
----------
train Loss: 0.0028 Acc: 0.9483
val Loss: 0.0097 Acc: 0.7794

Epoch 86/92
----------
train Loss: 0.0030 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 87/92
----------
train Loss: 0.0028 Acc: 0.9467
val Loss: 0.0096 Acc: 0.7794

Epoch 88/92
----------
train Loss: 0.0028 Acc: 0.9451
val Loss: 0.0095 Acc: 0.7941

Epoch 89/92
----------
train Loss: 0.0033 Acc: 0.9435
val Loss: 0.0095 Acc: 0.7941

Epoch 90/92
----------
train Loss: 0.0029 Acc: 0.9402
val Loss: 0.0096 Acc: 0.7647

Epoch 91/92
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0028 Acc: 0.9532
val Loss: 0.0096 Acc: 0.7794

Epoch 92/92
----------
train Loss: 0.0028 Acc: 0.9499
val Loss: 0.0097 Acc: 0.7647

Training complete in 4m 34s
Best val Acc: 0.794118

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0030 Acc: 0.9321
val Loss: 0.0079 Acc: 0.8235

Epoch 1/92
----------
train Loss: 0.0013 Acc: 0.9822
val Loss: 0.0071 Acc: 0.8088

Epoch 2/92
----------
train Loss: 0.0008 Acc: 0.9887
val Loss: 0.0069 Acc: 0.8088

Epoch 3/92
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0069 Acc: 0.8382

Epoch 4/92
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0083 Acc: 0.7941

Epoch 5/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.7794

Epoch 6/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.7941

Epoch 7/92
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0083 Acc: 0.7941

Epoch 8/92
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.7941

Epoch 9/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 10/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 11/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 12/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8088

Epoch 13/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8088

Epoch 14/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 15/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 16/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 17/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 18/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.7941

Epoch 19/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 20/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 21/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 22/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.7941

Epoch 23/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 24/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 25/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 26/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 27/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 28/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 29/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 30/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 31/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 32/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 33/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 34/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 35/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 36/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 37/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 38/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 39/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 40/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 41/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 42/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 43/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 44/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 45/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 46/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 47/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 48/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 49/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 50/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 51/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 52/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 53/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 54/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 55/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 56/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 57/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 58/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 59/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 60/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 61/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 62/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 63/92
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 64/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 65/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 66/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 67/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 68/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 69/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 70/92
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8088

Epoch 71/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8088

Epoch 72/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8088

Epoch 73/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 74/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 75/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 76/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 77/92
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 78/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 79/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8088

Epoch 80/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 81/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8088

Epoch 82/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 83/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 84/92
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.7941

Epoch 85/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.7941

Epoch 86/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 87/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 88/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 89/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 90/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 91/92
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8088

Epoch 92/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Training complete in 4m 51s
Best val Acc: 0.838235

---Testing---
Test accuracy: 0.982533
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 96 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 97 %
mean: 0.9778981544771018, std: 0.014581075311861084
--------------------

run info[val: 0.15, epoch: 99, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0212 Acc: 0.1815
val Loss: 0.0291 Acc: 0.3107

Epoch 1/98
----------
train Loss: 0.0134 Acc: 0.5942
val Loss: 0.0231 Acc: 0.5728

Epoch 2/98
----------
train Loss: 0.0083 Acc: 0.7517
val Loss: 0.0218 Acc: 0.7864

Epoch 3/98
----------
train Loss: 0.0058 Acc: 0.8305
val Loss: 0.0211 Acc: 0.7767

Epoch 4/98
----------
train Loss: 0.0046 Acc: 0.8613
val Loss: 0.0155 Acc: 0.7864

Epoch 5/98
----------
train Loss: 0.0039 Acc: 0.8938
val Loss: 0.0065 Acc: 0.7961

Epoch 6/98
----------
train Loss: 0.0033 Acc: 0.9161
val Loss: 0.0105 Acc: 0.8155

Epoch 7/98
----------
train Loss: 0.0031 Acc: 0.9195
val Loss: 0.0074 Acc: 0.7670

Epoch 8/98
----------
train Loss: 0.0026 Acc: 0.9572
val Loss: 0.0082 Acc: 0.8058

Epoch 9/98
----------
LR is set to 0.001
train Loss: 0.0024 Acc: 0.9623
val Loss: 0.0061 Acc: 0.8058

Epoch 10/98
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0067 Acc: 0.7864

Epoch 11/98
----------
train Loss: 0.0023 Acc: 0.9469
val Loss: 0.0184 Acc: 0.7767

Epoch 12/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0101 Acc: 0.7767

Epoch 13/98
----------
train Loss: 0.0024 Acc: 0.9555
val Loss: 0.0157 Acc: 0.7767

Epoch 14/98
----------
train Loss: 0.0025 Acc: 0.9332
val Loss: 0.0204 Acc: 0.7864

Epoch 15/98
----------
train Loss: 0.0025 Acc: 0.9469
val Loss: 0.0179 Acc: 0.7864

Epoch 16/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0137 Acc: 0.7864

Epoch 17/98
----------
train Loss: 0.0022 Acc: 0.9469
val Loss: 0.0115 Acc: 0.8058

Epoch 18/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0099 Acc: 0.7961

Epoch 19/98
----------
train Loss: 0.0024 Acc: 0.9452
val Loss: 0.0067 Acc: 0.8058

Epoch 20/98
----------
train Loss: 0.0023 Acc: 0.9521
val Loss: 0.0090 Acc: 0.8058

Epoch 21/98
----------
train Loss: 0.0023 Acc: 0.9469
val Loss: 0.0080 Acc: 0.7961

Epoch 22/98
----------
train Loss: 0.0024 Acc: 0.9469
val Loss: 0.0062 Acc: 0.7961

Epoch 23/98
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0118 Acc: 0.7961

Epoch 24/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0063 Acc: 0.7961

Epoch 25/98
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0100 Acc: 0.7961

Epoch 26/98
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0060 Acc: 0.7961

Epoch 27/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9606
val Loss: 0.0057 Acc: 0.7864

Epoch 28/98
----------
train Loss: 0.0022 Acc: 0.9486
val Loss: 0.0138 Acc: 0.7864

Epoch 29/98
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0088 Acc: 0.7864

Epoch 30/98
----------
train Loss: 0.0022 Acc: 0.9623
val Loss: 0.0073 Acc: 0.7864

Epoch 31/98
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0116 Acc: 0.7961

Epoch 32/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0079 Acc: 0.7961

Epoch 33/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0058 Acc: 0.7961

Epoch 34/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0061 Acc: 0.7961

Epoch 35/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0072 Acc: 0.7864

Epoch 36/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9658
val Loss: 0.0187 Acc: 0.7864

Epoch 37/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0067 Acc: 0.7864

Epoch 38/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0071 Acc: 0.7864

Epoch 39/98
----------
train Loss: 0.0022 Acc: 0.9623
val Loss: 0.0061 Acc: 0.7864

Epoch 40/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0068 Acc: 0.7864

Epoch 41/98
----------
train Loss: 0.0023 Acc: 0.9469
val Loss: 0.0152 Acc: 0.7864

Epoch 42/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0060 Acc: 0.7864

Epoch 43/98
----------
train Loss: 0.0024 Acc: 0.9623
val Loss: 0.0059 Acc: 0.7864

Epoch 44/98
----------
train Loss: 0.0023 Acc: 0.9606
val Loss: 0.0076 Acc: 0.7864

Epoch 45/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0124 Acc: 0.7864

Epoch 46/98
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0147 Acc: 0.7864

Epoch 47/98
----------
train Loss: 0.0022 Acc: 0.9623
val Loss: 0.0065 Acc: 0.7864

Epoch 48/98
----------
train Loss: 0.0023 Acc: 0.9606
val Loss: 0.0060 Acc: 0.7864

Epoch 49/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0068 Acc: 0.7864

Epoch 50/98
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0195 Acc: 0.7961

Epoch 51/98
----------
train Loss: 0.0022 Acc: 0.9640
val Loss: 0.0061 Acc: 0.7961

Epoch 52/98
----------
train Loss: 0.0022 Acc: 0.9469
val Loss: 0.0161 Acc: 0.7961

Epoch 53/98
----------
train Loss: 0.0022 Acc: 0.9503
val Loss: 0.0061 Acc: 0.7961

Epoch 54/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0090 Acc: 0.7864

Epoch 55/98
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0058 Acc: 0.7864

Epoch 56/98
----------
train Loss: 0.0022 Acc: 0.9658
val Loss: 0.0061 Acc: 0.7864

Epoch 57/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0180 Acc: 0.7864

Epoch 58/98
----------
train Loss: 0.0023 Acc: 0.9521
val Loss: 0.0125 Acc: 0.7864

Epoch 59/98
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0121 Acc: 0.7864

Epoch 60/98
----------
train Loss: 0.0022 Acc: 0.9658
val Loss: 0.0134 Acc: 0.7961

Epoch 61/98
----------
train Loss: 0.0022 Acc: 0.9606
val Loss: 0.0083 Acc: 0.7864

Epoch 62/98
----------
train Loss: 0.0024 Acc: 0.9538
val Loss: 0.0072 Acc: 0.7961

Epoch 63/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0023 Acc: 0.9435
val Loss: 0.0070 Acc: 0.7961

Epoch 64/98
----------
train Loss: 0.0022 Acc: 0.9486
val Loss: 0.0101 Acc: 0.7961

Epoch 65/98
----------
train Loss: 0.0023 Acc: 0.9521
val Loss: 0.0066 Acc: 0.7961

Epoch 66/98
----------
train Loss: 0.0022 Acc: 0.9521
val Loss: 0.0071 Acc: 0.7961

Epoch 67/98
----------
train Loss: 0.0023 Acc: 0.9521
val Loss: 0.0074 Acc: 0.7864

Epoch 68/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0251 Acc: 0.7864

Epoch 69/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0064 Acc: 0.7864

Epoch 70/98
----------
train Loss: 0.0022 Acc: 0.9538
val Loss: 0.0059 Acc: 0.7864

Epoch 71/98
----------
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0086 Acc: 0.7864

Epoch 72/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0105 Acc: 0.7864

Epoch 73/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0078 Acc: 0.7961

Epoch 74/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0059 Acc: 0.7864

Epoch 75/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0076 Acc: 0.7864

Epoch 76/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0099 Acc: 0.7864

Epoch 77/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0065 Acc: 0.7864

Epoch 78/98
----------
train Loss: 0.0024 Acc: 0.9452
val Loss: 0.0075 Acc: 0.7961

Epoch 79/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0090 Acc: 0.7961

Epoch 80/98
----------
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0092 Acc: 0.7864

Epoch 81/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0023 Acc: 0.9589
val Loss: 0.0170 Acc: 0.7961

Epoch 82/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0078 Acc: 0.7961

Epoch 83/98
----------
train Loss: 0.0022 Acc: 0.9589
val Loss: 0.0061 Acc: 0.7864

Epoch 84/98
----------
train Loss: 0.0024 Acc: 0.9486
val Loss: 0.0075 Acc: 0.7864

Epoch 85/98
----------
train Loss: 0.0024 Acc: 0.9503
val Loss: 0.0068 Acc: 0.7864

Epoch 86/98
----------
train Loss: 0.0024 Acc: 0.9623
val Loss: 0.0071 Acc: 0.7864

Epoch 87/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0102 Acc: 0.7864

Epoch 88/98
----------
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0084 Acc: 0.7864

Epoch 89/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0083 Acc: 0.7864

Epoch 90/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0023 Acc: 0.9640
val Loss: 0.0105 Acc: 0.7864

Epoch 91/98
----------
train Loss: 0.0023 Acc: 0.9623
val Loss: 0.0218 Acc: 0.7864

Epoch 92/98
----------
train Loss: 0.0022 Acc: 0.9555
val Loss: 0.0068 Acc: 0.7864

Epoch 93/98
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0098 Acc: 0.7961

Epoch 94/98
----------
train Loss: 0.0023 Acc: 0.9503
val Loss: 0.0059 Acc: 0.7961

Epoch 95/98
----------
train Loss: 0.0023 Acc: 0.9640
val Loss: 0.0059 Acc: 0.7864

Epoch 96/98
----------
train Loss: 0.0023 Acc: 0.9555
val Loss: 0.0090 Acc: 0.7864

Epoch 97/98
----------
train Loss: 0.0024 Acc: 0.9555
val Loss: 0.0079 Acc: 0.7864

Epoch 98/98
----------
train Loss: 0.0023 Acc: 0.9572
val Loss: 0.0078 Acc: 0.7961

Training complete in 4m 60s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0029 Acc: 0.9178
val Loss: 0.0078 Acc: 0.8058

Epoch 1/98
----------
train Loss: 0.0015 Acc: 0.9726
val Loss: 0.0097 Acc: 0.8447

Epoch 2/98
----------
train Loss: 0.0007 Acc: 0.9949
val Loss: 0.0052 Acc: 0.8641

Epoch 3/98
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8447

Epoch 4/98
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 5/98
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0054 Acc: 0.8447

Epoch 6/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8447

Epoch 7/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8350

Epoch 8/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0174 Acc: 0.8350

Epoch 9/98
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0135 Acc: 0.8350

Epoch 10/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0122 Acc: 0.8350

Epoch 11/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 12/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 13/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 14/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0110 Acc: 0.8350

Epoch 15/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 16/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 17/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8350

Epoch 18/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8350

Epoch 19/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8350

Epoch 20/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8350

Epoch 21/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8350

Epoch 22/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8350

Epoch 23/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 24/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 25/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8350

Epoch 26/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8350

Epoch 27/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 28/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 29/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 30/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8350

Epoch 31/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 32/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0219 Acc: 0.8350

Epoch 33/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8350

Epoch 34/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 35/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8350

Epoch 36/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 37/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 38/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 39/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0155 Acc: 0.8350

Epoch 40/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8350

Epoch 41/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0106 Acc: 0.8350

Epoch 42/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 43/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8350

Epoch 44/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8350

Epoch 45/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0156 Acc: 0.8350

Epoch 46/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8350

Epoch 47/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 48/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 49/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8350

Epoch 50/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 51/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8350

Epoch 52/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8350

Epoch 53/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 54/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0199 Acc: 0.8350

Epoch 55/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8350

Epoch 56/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 57/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 58/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 59/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 60/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8350

Epoch 61/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8350

Epoch 62/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8350

Epoch 63/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 64/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8350

Epoch 65/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8350

Epoch 66/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8350

Epoch 67/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8350

Epoch 68/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 69/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0171 Acc: 0.8350

Epoch 70/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 71/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 72/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0145 Acc: 0.8350

Epoch 73/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0227 Acc: 0.8350

Epoch 74/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8350

Epoch 75/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 76/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8350

Epoch 77/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0130 Acc: 0.8350

Epoch 78/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 79/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 80/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 81/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 82/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0143 Acc: 0.8350

Epoch 83/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0212 Acc: 0.8350

Epoch 84/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0176 Acc: 0.8350

Epoch 85/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 86/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 87/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0142 Acc: 0.8350

Epoch 88/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0210 Acc: 0.8350

Epoch 89/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8350

Epoch 90/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8350

Epoch 91/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8350

Epoch 92/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 93/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 94/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8350

Epoch 95/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 96/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8350

Epoch 97/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8350

Epoch 98/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8350

Training complete in 5m 16s
Best val Acc: 0.864078

---Testing---
Test accuracy: 0.976710
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 97 %
mean: 0.9696848263887297, std: 0.03006706889914035
--------------------

run info[val: 0.2, epoch: 82, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0218 Acc: 0.2509
val Loss: 0.0266 Acc: 0.2628

Epoch 1/81
----------
train Loss: 0.0147 Acc: 0.5582
val Loss: 0.0156 Acc: 0.6131

Epoch 2/81
----------
train Loss: 0.0091 Acc: 0.7236
val Loss: 0.0106 Acc: 0.7445

Epoch 3/81
----------
train Loss: 0.0068 Acc: 0.7964
val Loss: 0.0098 Acc: 0.7591

Epoch 4/81
----------
train Loss: 0.0050 Acc: 0.8582
val Loss: 0.0094 Acc: 0.7372

Epoch 5/81
----------
train Loss: 0.0047 Acc: 0.8673
val Loss: 0.0081 Acc: 0.8102

Epoch 6/81
----------
train Loss: 0.0035 Acc: 0.9109
val Loss: 0.0078 Acc: 0.7883

Epoch 7/81
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0079 Acc: 0.8248

Epoch 8/81
----------
train Loss: 0.0027 Acc: 0.9509
val Loss: 0.0073 Acc: 0.7956

Epoch 9/81
----------
LR is set to 0.001
train Loss: 0.0024 Acc: 0.9545
val Loss: 0.0074 Acc: 0.7883

Epoch 10/81
----------
train Loss: 0.0024 Acc: 0.9618
val Loss: 0.0073 Acc: 0.7956

Epoch 11/81
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0072 Acc: 0.8029

Epoch 12/81
----------
train Loss: 0.0026 Acc: 0.9564
val Loss: 0.0078 Acc: 0.8029

Epoch 13/81
----------
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0082 Acc: 0.8102

Epoch 14/81
----------
train Loss: 0.0023 Acc: 0.9691
val Loss: 0.0059 Acc: 0.8175

Epoch 15/81
----------
train Loss: 0.0023 Acc: 0.9491
val Loss: 0.0086 Acc: 0.8029

Epoch 16/81
----------
train Loss: 0.0024 Acc: 0.9545
val Loss: 0.0086 Acc: 0.8029

Epoch 17/81
----------
train Loss: 0.0024 Acc: 0.9527
val Loss: 0.0076 Acc: 0.7956

Epoch 18/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0023 Acc: 0.9655
val Loss: 0.0082 Acc: 0.7956

Epoch 19/81
----------
train Loss: 0.0023 Acc: 0.9600
val Loss: 0.0071 Acc: 0.7956

Epoch 20/81
----------
train Loss: 0.0022 Acc: 0.9509
val Loss: 0.0071 Acc: 0.7956

Epoch 21/81
----------
train Loss: 0.0022 Acc: 0.9636
val Loss: 0.0076 Acc: 0.7956

Epoch 22/81
----------
train Loss: 0.0023 Acc: 0.9636
val Loss: 0.0082 Acc: 0.7956

Epoch 23/81
----------
train Loss: 0.0022 Acc: 0.9636
val Loss: 0.0075 Acc: 0.7956

Epoch 24/81
----------
train Loss: 0.0022 Acc: 0.9709
val Loss: 0.0092 Acc: 0.7956

Epoch 25/81
----------
train Loss: 0.0022 Acc: 0.9618
val Loss: 0.0073 Acc: 0.7956

Epoch 26/81
----------
train Loss: 0.0024 Acc: 0.9636
val Loss: 0.0077 Acc: 0.7956

Epoch 27/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0074 Acc: 0.7956

Epoch 28/81
----------
train Loss: 0.0023 Acc: 0.9600
val Loss: 0.0087 Acc: 0.7956

Epoch 29/81
----------
train Loss: 0.0021 Acc: 0.9636
val Loss: 0.0071 Acc: 0.7956

Epoch 30/81
----------
train Loss: 0.0025 Acc: 0.9618
val Loss: 0.0081 Acc: 0.7956

Epoch 31/81
----------
train Loss: 0.0023 Acc: 0.9655
val Loss: 0.0068 Acc: 0.7956

Epoch 32/81
----------
train Loss: 0.0025 Acc: 0.9618
val Loss: 0.0072 Acc: 0.7956

Epoch 33/81
----------
train Loss: 0.0023 Acc: 0.9618
val Loss: 0.0070 Acc: 0.7956

Epoch 34/81
----------
train Loss: 0.0023 Acc: 0.9564
val Loss: 0.0087 Acc: 0.7956

Epoch 35/81
----------
train Loss: 0.0023 Acc: 0.9691
val Loss: 0.0083 Acc: 0.8029

Epoch 36/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0023 Acc: 0.9636
val Loss: 0.0073 Acc: 0.8029

Epoch 37/81
----------
train Loss: 0.0024 Acc: 0.9600
val Loss: 0.0077 Acc: 0.8029

Epoch 38/81
----------
train Loss: 0.0022 Acc: 0.9673
val Loss: 0.0074 Acc: 0.7956

Epoch 39/81
----------
train Loss: 0.0022 Acc: 0.9691
val Loss: 0.0075 Acc: 0.8029

Epoch 40/81
----------
train Loss: 0.0022 Acc: 0.9618
val Loss: 0.0077 Acc: 0.8029

Epoch 41/81
----------
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0089 Acc: 0.8029

Epoch 42/81
----------
train Loss: 0.0023 Acc: 0.9600
val Loss: 0.0083 Acc: 0.8029

Epoch 43/81
----------
train Loss: 0.0023 Acc: 0.9618
val Loss: 0.0084 Acc: 0.8029

Epoch 44/81
----------
train Loss: 0.0024 Acc: 0.9600
val Loss: 0.0075 Acc: 0.7956

Epoch 45/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0022 Acc: 0.9564
val Loss: 0.0071 Acc: 0.8029

Epoch 46/81
----------
train Loss: 0.0022 Acc: 0.9673
val Loss: 0.0064 Acc: 0.8029

Epoch 47/81
----------
train Loss: 0.0022 Acc: 0.9673
val Loss: 0.0080 Acc: 0.7956

Epoch 48/81
----------
train Loss: 0.0022 Acc: 0.9600
val Loss: 0.0081 Acc: 0.8029

Epoch 49/81
----------
train Loss: 0.0023 Acc: 0.9636
val Loss: 0.0093 Acc: 0.8029

Epoch 50/81
----------
train Loss: 0.0022 Acc: 0.9618
val Loss: 0.0071 Acc: 0.8029

Epoch 51/81
----------
train Loss: 0.0023 Acc: 0.9673
val Loss: 0.0081 Acc: 0.8029

Epoch 52/81
----------
train Loss: 0.0023 Acc: 0.9636
val Loss: 0.0077 Acc: 0.8102

Epoch 53/81
----------
train Loss: 0.0022 Acc: 0.9655
val Loss: 0.0067 Acc: 0.8029

Epoch 54/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0023 Acc: 0.9600
val Loss: 0.0079 Acc: 0.8029

Epoch 55/81
----------
train Loss: 0.0021 Acc: 0.9636
val Loss: 0.0068 Acc: 0.8029

Epoch 56/81
----------
train Loss: 0.0024 Acc: 0.9600
val Loss: 0.0075 Acc: 0.8029

Epoch 57/81
----------
train Loss: 0.0023 Acc: 0.9582
val Loss: 0.0069 Acc: 0.7956

Epoch 58/81
----------
train Loss: 0.0022 Acc: 0.9673
val Loss: 0.0073 Acc: 0.7956

Epoch 59/81
----------
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0076 Acc: 0.7956

Epoch 60/81
----------
train Loss: 0.0022 Acc: 0.9673
val Loss: 0.0076 Acc: 0.7956

Epoch 61/81
----------
train Loss: 0.0021 Acc: 0.9673
val Loss: 0.0081 Acc: 0.7956

Epoch 62/81
----------
train Loss: 0.0023 Acc: 0.9600
val Loss: 0.0069 Acc: 0.7956

Epoch 63/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0022 Acc: 0.9636
val Loss: 0.0081 Acc: 0.8029

Epoch 64/81
----------
train Loss: 0.0024 Acc: 0.9473
val Loss: 0.0067 Acc: 0.8029

Epoch 65/81
----------
train Loss: 0.0022 Acc: 0.9600
val Loss: 0.0077 Acc: 0.8029

Epoch 66/81
----------
train Loss: 0.0023 Acc: 0.9655
val Loss: 0.0071 Acc: 0.8029

Epoch 67/81
----------
train Loss: 0.0024 Acc: 0.9582
val Loss: 0.0074 Acc: 0.7956

Epoch 68/81
----------
train Loss: 0.0024 Acc: 0.9655
val Loss: 0.0078 Acc: 0.8029

Epoch 69/81
----------
train Loss: 0.0022 Acc: 0.9600
val Loss: 0.0082 Acc: 0.7956

Epoch 70/81
----------
train Loss: 0.0022 Acc: 0.9636
val Loss: 0.0075 Acc: 0.7956

Epoch 71/81
----------
train Loss: 0.0021 Acc: 0.9673
val Loss: 0.0076 Acc: 0.8029

Epoch 72/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0024 Acc: 0.9582
val Loss: 0.0082 Acc: 0.8029

Epoch 73/81
----------
train Loss: 0.0023 Acc: 0.9618
val Loss: 0.0073 Acc: 0.7956

Epoch 74/81
----------
train Loss: 0.0021 Acc: 0.9673
val Loss: 0.0071 Acc: 0.7956

Epoch 75/81
----------
train Loss: 0.0021 Acc: 0.9727
val Loss: 0.0086 Acc: 0.7956

Epoch 76/81
----------
train Loss: 0.0022 Acc: 0.9709
val Loss: 0.0081 Acc: 0.8029

Epoch 77/81
----------
train Loss: 0.0024 Acc: 0.9673
val Loss: 0.0070 Acc: 0.7956

Epoch 78/81
----------
train Loss: 0.0023 Acc: 0.9582
val Loss: 0.0072 Acc: 0.8029

Epoch 79/81
----------
train Loss: 0.0022 Acc: 0.9709
val Loss: 0.0071 Acc: 0.7956

Epoch 80/81
----------
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0084 Acc: 0.8029

Epoch 81/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0024 Acc: 0.9673
val Loss: 0.0078 Acc: 0.7956

Training complete in 4m 4s
Best val Acc: 0.824818

---Fine tuning.---
Epoch 0/81
----------
LR is set to 0.01
train Loss: 0.0027 Acc: 0.9436
val Loss: 0.0067 Acc: 0.8394

Epoch 1/81
----------
train Loss: 0.0015 Acc: 0.9727
val Loss: 0.0059 Acc: 0.8686

Epoch 2/81
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0081 Acc: 0.8467

Epoch 3/81
----------
train Loss: 0.0004 Acc: 0.9964
val Loss: 0.0060 Acc: 0.8613

Epoch 4/81
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8540

Epoch 5/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 6/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 7/81
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8832

Epoch 8/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8905

Epoch 9/81
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8905

Epoch 10/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8905

Epoch 11/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8905

Epoch 12/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8832

Epoch 13/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8832

Epoch 14/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8832

Epoch 15/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8832

Epoch 16/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8832

Epoch 17/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8832

Epoch 18/81
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Epoch 19/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8905

Epoch 20/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8832

Epoch 21/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8832

Epoch 22/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8832

Epoch 23/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Epoch 24/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8832

Epoch 25/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8832

Epoch 26/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8832

Epoch 27/81
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8832

Epoch 28/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8905

Epoch 29/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Epoch 30/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8832

Epoch 31/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8905

Epoch 32/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8905

Epoch 33/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8832

Epoch 34/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8832

Epoch 35/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8832

Epoch 36/81
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8832

Epoch 37/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8832

Epoch 38/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8832

Epoch 39/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8832

Epoch 40/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8832

Epoch 41/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8832

Epoch 42/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8832

Epoch 43/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8832

Epoch 44/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8832

Epoch 45/81
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8905

Epoch 46/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8832

Epoch 47/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8832

Epoch 48/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8832

Epoch 49/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8832

Epoch 50/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8832

Epoch 51/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8832

Epoch 52/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8832

Epoch 53/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8832

Epoch 54/81
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8832

Epoch 55/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8832

Epoch 56/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8832

Epoch 57/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8905

Epoch 58/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8905

Epoch 59/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8832

Epoch 60/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8905

Epoch 61/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8832

Epoch 62/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8832

Epoch 63/81
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8905

Epoch 64/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8905

Epoch 65/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8832

Epoch 66/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8832

Epoch 67/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Epoch 68/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8832

Epoch 69/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8832

Epoch 70/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8832

Epoch 71/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8832

Epoch 72/81
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8832

Epoch 73/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8832

Epoch 74/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8832

Epoch 75/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8832

Epoch 76/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8832

Epoch 77/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8832

Epoch 78/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8832

Epoch 79/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8832

Epoch 80/81
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8832

Epoch 81/81
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Training complete in 4m 14s
Best val Acc: 0.890511

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 98 %
mean: 0.9696684894053316, std: 0.03134459520536045
--------------------

run info[val: 0.25, epoch: 61, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0230 Acc: 0.2616
val Loss: 0.0217 Acc: 0.3977

Epoch 1/60
----------
train Loss: 0.0167 Acc: 0.5174
val Loss: 0.0141 Acc: 0.5673

Epoch 2/60
----------
train Loss: 0.0113 Acc: 0.6570
val Loss: 0.0098 Acc: 0.7661

Epoch 3/60
----------
LR is set to 0.001
train Loss: 0.0083 Acc: 0.7791
val Loss: 0.0088 Acc: 0.7953

Epoch 4/60
----------
train Loss: 0.0068 Acc: 0.8314
val Loss: 0.0086 Acc: 0.8012

Epoch 5/60
----------
train Loss: 0.0069 Acc: 0.8663
val Loss: 0.0082 Acc: 0.7719

Epoch 6/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0064 Acc: 0.8663
val Loss: 0.0084 Acc: 0.7778

Epoch 7/60
----------
train Loss: 0.0061 Acc: 0.8682
val Loss: 0.0084 Acc: 0.7778

Epoch 8/60
----------
train Loss: 0.0063 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7778

Epoch 9/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0063 Acc: 0.8605
val Loss: 0.0082 Acc: 0.7719

Epoch 10/60
----------
train Loss: 0.0062 Acc: 0.8779
val Loss: 0.0082 Acc: 0.7778

Epoch 11/60
----------
train Loss: 0.0063 Acc: 0.8566
val Loss: 0.0080 Acc: 0.7895

Epoch 12/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0063 Acc: 0.8469
val Loss: 0.0080 Acc: 0.7836

Epoch 13/60
----------
train Loss: 0.0066 Acc: 0.8585
val Loss: 0.0079 Acc: 0.7836

Epoch 14/60
----------
train Loss: 0.0062 Acc: 0.8605
val Loss: 0.0079 Acc: 0.7836

Epoch 15/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0063 Acc: 0.8663
val Loss: 0.0081 Acc: 0.7836

Epoch 16/60
----------
train Loss: 0.0060 Acc: 0.8624
val Loss: 0.0082 Acc: 0.7836

Epoch 17/60
----------
train Loss: 0.0059 Acc: 0.8740
val Loss: 0.0080 Acc: 0.7895

Epoch 18/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0064 Acc: 0.8527
val Loss: 0.0080 Acc: 0.7895

Epoch 19/60
----------
train Loss: 0.0066 Acc: 0.8682
val Loss: 0.0082 Acc: 0.7836

Epoch 20/60
----------
train Loss: 0.0062 Acc: 0.8721
val Loss: 0.0080 Acc: 0.7836

Epoch 21/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0064 Acc: 0.8624
val Loss: 0.0082 Acc: 0.7836

Epoch 22/60
----------
train Loss: 0.0061 Acc: 0.8566
val Loss: 0.0081 Acc: 0.7719

Epoch 23/60
----------
train Loss: 0.0073 Acc: 0.8585
val Loss: 0.0081 Acc: 0.7719

Epoch 24/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0067 Acc: 0.8585
val Loss: 0.0080 Acc: 0.7836

Epoch 25/60
----------
train Loss: 0.0058 Acc: 0.8760
val Loss: 0.0080 Acc: 0.7778

Epoch 26/60
----------
train Loss: 0.0064 Acc: 0.8605
val Loss: 0.0083 Acc: 0.7895

Epoch 27/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0067 Acc: 0.8605
val Loss: 0.0080 Acc: 0.7953

Epoch 28/60
----------
train Loss: 0.0061 Acc: 0.8663
val Loss: 0.0082 Acc: 0.7953

Epoch 29/60
----------
train Loss: 0.0059 Acc: 0.8663
val Loss: 0.0081 Acc: 0.7778

Epoch 30/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0062 Acc: 0.8585
val Loss: 0.0081 Acc: 0.7778

Epoch 31/60
----------
train Loss: 0.0060 Acc: 0.8624
val Loss: 0.0081 Acc: 0.7778

Epoch 32/60
----------
train Loss: 0.0058 Acc: 0.8721
val Loss: 0.0081 Acc: 0.7778

Epoch 33/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0062 Acc: 0.8585
val Loss: 0.0081 Acc: 0.7836

Epoch 34/60
----------
train Loss: 0.0058 Acc: 0.8721
val Loss: 0.0081 Acc: 0.7895

Epoch 35/60
----------
train Loss: 0.0069 Acc: 0.8566
val Loss: 0.0084 Acc: 0.7661

Epoch 36/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0064 Acc: 0.8682
val Loss: 0.0080 Acc: 0.7895

Epoch 37/60
----------
train Loss: 0.0067 Acc: 0.8547
val Loss: 0.0081 Acc: 0.7836

Epoch 38/60
----------
train Loss: 0.0062 Acc: 0.8663
val Loss: 0.0080 Acc: 0.7719

Epoch 39/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0065 Acc: 0.8643
val Loss: 0.0083 Acc: 0.7895

Epoch 40/60
----------
train Loss: 0.0064 Acc: 0.8624
val Loss: 0.0082 Acc: 0.7661

Epoch 41/60
----------
train Loss: 0.0061 Acc: 0.8547
val Loss: 0.0082 Acc: 0.7719

Epoch 42/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0063 Acc: 0.8682
val Loss: 0.0082 Acc: 0.7719

Epoch 43/60
----------
train Loss: 0.0061 Acc: 0.8605
val Loss: 0.0083 Acc: 0.7836

Epoch 44/60
----------
train Loss: 0.0063 Acc: 0.8702
val Loss: 0.0080 Acc: 0.7895

Epoch 45/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0062 Acc: 0.8682
val Loss: 0.0081 Acc: 0.7895

Epoch 46/60
----------
train Loss: 0.0064 Acc: 0.8605
val Loss: 0.0082 Acc: 0.7895

Epoch 47/60
----------
train Loss: 0.0062 Acc: 0.8643
val Loss: 0.0081 Acc: 0.7895

Epoch 48/60
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0069 Acc: 0.8663
val Loss: 0.0080 Acc: 0.7836

Epoch 49/60
----------
train Loss: 0.0072 Acc: 0.8527
val Loss: 0.0080 Acc: 0.7778

Epoch 50/60
----------
train Loss: 0.0065 Acc: 0.8605
val Loss: 0.0080 Acc: 0.7836

Epoch 51/60
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0063 Acc: 0.8624
val Loss: 0.0083 Acc: 0.7836

Epoch 52/60
----------
train Loss: 0.0063 Acc: 0.8740
val Loss: 0.0081 Acc: 0.7778

Epoch 53/60
----------
train Loss: 0.0062 Acc: 0.8663
val Loss: 0.0080 Acc: 0.7895

Epoch 54/60
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0064 Acc: 0.8527
val Loss: 0.0080 Acc: 0.7836

Epoch 55/60
----------
train Loss: 0.0063 Acc: 0.8643
val Loss: 0.0079 Acc: 0.7836

Epoch 56/60
----------
train Loss: 0.0063 Acc: 0.8682
val Loss: 0.0080 Acc: 0.7778

Epoch 57/60
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0067 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7895

Epoch 58/60
----------
train Loss: 0.0063 Acc: 0.8682
val Loss: 0.0082 Acc: 0.7778

Epoch 59/60
----------
train Loss: 0.0064 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7778

Epoch 60/60
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0060 Acc: 0.8624
val Loss: 0.0081 Acc: 0.7836

Training complete in 3m 2s
Best val Acc: 0.801170

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0059 Acc: 0.8624
val Loss: 0.0072 Acc: 0.7895

Epoch 1/60
----------
train Loss: 0.0033 Acc: 0.9360
val Loss: 0.0074 Acc: 0.8129

Epoch 2/60
----------
train Loss: 0.0019 Acc: 0.9729
val Loss: 0.0053 Acc: 0.8538

Epoch 3/60
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8538

Epoch 4/60
----------
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0042 Acc: 0.8713

Epoch 5/60
----------
train Loss: 0.0008 Acc: 0.9942
val Loss: 0.0044 Acc: 0.8713

Epoch 6/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 7/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 8/60
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0043 Acc: 0.8772

Epoch 9/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 10/60
----------
train Loss: 0.0006 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8772

Epoch 11/60
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0044 Acc: 0.8772

Epoch 12/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0043 Acc: 0.8772

Epoch 13/60
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8713

Epoch 14/60
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0041 Acc: 0.8713

Epoch 15/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 0.9942
val Loss: 0.0042 Acc: 0.8713

Epoch 16/60
----------
train Loss: 0.0008 Acc: 0.9903
val Loss: 0.0047 Acc: 0.8772

Epoch 17/60
----------
train Loss: 0.0009 Acc: 0.9961
val Loss: 0.0044 Acc: 0.8830

Epoch 18/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9903
val Loss: 0.0042 Acc: 0.8830

Epoch 19/60
----------
train Loss: 0.0006 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8830

Epoch 20/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8772

Epoch 21/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0007 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8713

Epoch 22/60
----------
train Loss: 0.0008 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8713

Epoch 23/60
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8772

Epoch 24/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0040 Acc: 0.8772

Epoch 25/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0044 Acc: 0.8713

Epoch 26/60
----------
train Loss: 0.0008 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8772

Epoch 27/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0009 Acc: 0.9942
val Loss: 0.0040 Acc: 0.8772

Epoch 28/60
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0040 Acc: 0.8713

Epoch 29/60
----------
train Loss: 0.0008 Acc: 0.9961
val Loss: 0.0042 Acc: 0.8830

Epoch 30/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8772

Epoch 31/60
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0043 Acc: 0.8713

Epoch 32/60
----------
train Loss: 0.0010 Acc: 0.9942
val Loss: 0.0043 Acc: 0.8713

Epoch 33/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0008 Acc: 0.9922
val Loss: 0.0040 Acc: 0.8713

Epoch 34/60
----------
train Loss: 0.0008 Acc: 0.9903
val Loss: 0.0044 Acc: 0.8713

Epoch 35/60
----------
train Loss: 0.0008 Acc: 0.9922
val Loss: 0.0041 Acc: 0.8713

Epoch 36/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0010 Acc: 0.9922
val Loss: 0.0041 Acc: 0.8713

Epoch 37/60
----------
train Loss: 0.0009 Acc: 0.9961
val Loss: 0.0042 Acc: 0.8713

Epoch 38/60
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0042 Acc: 0.8713

Epoch 39/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 40/60
----------
train Loss: 0.0008 Acc: 0.9961
val Loss: 0.0044 Acc: 0.8713

Epoch 41/60
----------
train Loss: 0.0009 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8713

Epoch 42/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 43/60
----------
train Loss: 0.0010 Acc: 0.9942
val Loss: 0.0044 Acc: 0.8772

Epoch 44/60
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0041 Acc: 0.8772

Epoch 45/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0044 Acc: 0.8772

Epoch 46/60
----------
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8772

Epoch 47/60
----------
train Loss: 0.0006 Acc: 0.9981
val Loss: 0.0041 Acc: 0.8772

Epoch 48/60
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0011 Acc: 0.9922
val Loss: 0.0041 Acc: 0.8772

Epoch 49/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0041 Acc: 0.8655

Epoch 50/60
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0043 Acc: 0.8713

Epoch 51/60
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8830

Epoch 52/60
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0040 Acc: 0.8830

Epoch 53/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0040 Acc: 0.8772

Epoch 54/60
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0041 Acc: 0.8713

Epoch 55/60
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0043 Acc: 0.8772

Epoch 56/60
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0042 Acc: 0.8713

Epoch 57/60
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8772

Epoch 58/60
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8713

Epoch 59/60
----------
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0044 Acc: 0.8772

Epoch 60/60
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0042 Acc: 0.8772

Training complete in 3m 11s
Best val Acc: 0.883041

---Testing---
Test accuracy: 0.967977
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexanchiformes : 92 %
Accuracy of Lamniformes : 87 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 97 %
mean: 0.9574400098018078, std: 0.03809451555623211
--------------------

run info[val: 0.3, epoch: 88, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0217 Acc: 0.1830
val Loss: 0.0253 Acc: 0.3107

Epoch 1/87
----------
train Loss: 0.0160 Acc: 0.4782
val Loss: 0.0193 Acc: 0.5388

Epoch 2/87
----------
train Loss: 0.0112 Acc: 0.6632
val Loss: 0.0151 Acc: 0.6796

Epoch 3/87
----------
LR is set to 0.001
train Loss: 0.0085 Acc: 0.7422
val Loss: 0.0122 Acc: 0.7233

Epoch 4/87
----------
train Loss: 0.0076 Acc: 0.7900
val Loss: 0.0097 Acc: 0.7379

Epoch 5/87
----------
train Loss: 0.0072 Acc: 0.8087
val Loss: 0.0126 Acc: 0.7476

Epoch 6/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0070 Acc: 0.8274
val Loss: 0.0105 Acc: 0.7476

Epoch 7/87
----------
train Loss: 0.0073 Acc: 0.8170
val Loss: 0.0111 Acc: 0.7524

Epoch 8/87
----------
train Loss: 0.0071 Acc: 0.8358
val Loss: 0.0105 Acc: 0.7524

Epoch 9/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0072 Acc: 0.8254
val Loss: 0.0123 Acc: 0.7524

Epoch 10/87
----------
train Loss: 0.0074 Acc: 0.8150
val Loss: 0.0116 Acc: 0.7524

Epoch 11/87
----------
train Loss: 0.0071 Acc: 0.8378
val Loss: 0.0100 Acc: 0.7524

Epoch 12/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0071 Acc: 0.8295
val Loss: 0.0112 Acc: 0.7524

Epoch 13/87
----------
train Loss: 0.0071 Acc: 0.8233
val Loss: 0.0118 Acc: 0.7524

Epoch 14/87
----------
train Loss: 0.0070 Acc: 0.8358
val Loss: 0.0146 Acc: 0.7524

Epoch 15/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0072 Acc: 0.8191
val Loss: 0.0117 Acc: 0.7524

Epoch 16/87
----------
train Loss: 0.0073 Acc: 0.8087
val Loss: 0.0138 Acc: 0.7573

Epoch 17/87
----------
train Loss: 0.0072 Acc: 0.8274
val Loss: 0.0110 Acc: 0.7524

Epoch 18/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0072 Acc: 0.8274
val Loss: 0.0107 Acc: 0.7524

Epoch 19/87
----------
train Loss: 0.0073 Acc: 0.8212
val Loss: 0.0144 Acc: 0.7524

Epoch 20/87
----------
train Loss: 0.0071 Acc: 0.8358
val Loss: 0.0120 Acc: 0.7573

Epoch 21/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0072 Acc: 0.8191
val Loss: 0.0096 Acc: 0.7573

Epoch 22/87
----------
train Loss: 0.0070 Acc: 0.8358
val Loss: 0.0111 Acc: 0.7621

Epoch 23/87
----------
train Loss: 0.0073 Acc: 0.8170
val Loss: 0.0109 Acc: 0.7524

Epoch 24/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0072 Acc: 0.8378
val Loss: 0.0127 Acc: 0.7524

Epoch 25/87
----------
train Loss: 0.0073 Acc: 0.8004
val Loss: 0.0102 Acc: 0.7524

Epoch 26/87
----------
train Loss: 0.0071 Acc: 0.8025
val Loss: 0.0104 Acc: 0.7524

Epoch 27/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0073 Acc: 0.8274
val Loss: 0.0111 Acc: 0.7524

Epoch 28/87
----------
train Loss: 0.0072 Acc: 0.8233
val Loss: 0.0117 Acc: 0.7524

Epoch 29/87
----------
train Loss: 0.0072 Acc: 0.8358
val Loss: 0.0106 Acc: 0.7573

Epoch 30/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0071 Acc: 0.8046
val Loss: 0.0106 Acc: 0.7524

Epoch 31/87
----------
train Loss: 0.0070 Acc: 0.8274
val Loss: 0.0137 Acc: 0.7621

Epoch 32/87
----------
train Loss: 0.0072 Acc: 0.8254
val Loss: 0.0120 Acc: 0.7573

Epoch 33/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0072 Acc: 0.8295
val Loss: 0.0106 Acc: 0.7524

Epoch 34/87
----------
train Loss: 0.0071 Acc: 0.8420
val Loss: 0.0104 Acc: 0.7524

Epoch 35/87
----------
train Loss: 0.0072 Acc: 0.8254
val Loss: 0.0115 Acc: 0.7524

Epoch 36/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0072 Acc: 0.8212
val Loss: 0.0122 Acc: 0.7524

Epoch 37/87
----------
train Loss: 0.0072 Acc: 0.8108
val Loss: 0.0159 Acc: 0.7524

Epoch 38/87
----------
train Loss: 0.0074 Acc: 0.8046
val Loss: 0.0102 Acc: 0.7573

Epoch 39/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0073 Acc: 0.8046
val Loss: 0.0102 Acc: 0.7621

Epoch 40/87
----------
train Loss: 0.0070 Acc: 0.8378
val Loss: 0.0119 Acc: 0.7573

Epoch 41/87
----------
train Loss: 0.0071 Acc: 0.8191
val Loss: 0.0116 Acc: 0.7573

Epoch 42/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0070 Acc: 0.8316
val Loss: 0.0112 Acc: 0.7621

Epoch 43/87
----------
train Loss: 0.0071 Acc: 0.8295
val Loss: 0.0121 Acc: 0.7621

Epoch 44/87
----------
train Loss: 0.0073 Acc: 0.8233
val Loss: 0.0124 Acc: 0.7573

Epoch 45/87
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0072 Acc: 0.8212
val Loss: 0.0135 Acc: 0.7524

Epoch 46/87
----------
train Loss: 0.0072 Acc: 0.8170
val Loss: 0.0110 Acc: 0.7524

Epoch 47/87
----------
train Loss: 0.0074 Acc: 0.8295
val Loss: 0.0131 Acc: 0.7524

Epoch 48/87
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0072 Acc: 0.8191
val Loss: 0.0110 Acc: 0.7573

Epoch 49/87
----------
train Loss: 0.0071 Acc: 0.8212
val Loss: 0.0122 Acc: 0.7524

Epoch 50/87
----------
train Loss: 0.0071 Acc: 0.8295
val Loss: 0.0112 Acc: 0.7573

Epoch 51/87
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0072 Acc: 0.8004
val Loss: 0.0112 Acc: 0.7573

Epoch 52/87
----------
train Loss: 0.0072 Acc: 0.8358
val Loss: 0.0117 Acc: 0.7524

Epoch 53/87
----------
train Loss: 0.0071 Acc: 0.8337
val Loss: 0.0098 Acc: 0.7524

Epoch 54/87
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0073 Acc: 0.8025
val Loss: 0.0142 Acc: 0.7524

Epoch 55/87
----------
train Loss: 0.0069 Acc: 0.8274
val Loss: 0.0103 Acc: 0.7524

Epoch 56/87
----------
train Loss: 0.0073 Acc: 0.8004
val Loss: 0.0124 Acc: 0.7524

Epoch 57/87
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0072 Acc: 0.8191
val Loss: 0.0150 Acc: 0.7524

Epoch 58/87
----------
train Loss: 0.0072 Acc: 0.8316
val Loss: 0.0116 Acc: 0.7573

Epoch 59/87
----------
train Loss: 0.0069 Acc: 0.8378
val Loss: 0.0123 Acc: 0.7573

Epoch 60/87
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0074 Acc: 0.8067
val Loss: 0.0108 Acc: 0.7573

Epoch 61/87
----------
train Loss: 0.0071 Acc: 0.8254
val Loss: 0.0131 Acc: 0.7621

Epoch 62/87
----------
train Loss: 0.0072 Acc: 0.8233
val Loss: 0.0133 Acc: 0.7573

Epoch 63/87
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0072 Acc: 0.8337
val Loss: 0.0141 Acc: 0.7573

Epoch 64/87
----------
train Loss: 0.0070 Acc: 0.8191
val Loss: 0.0111 Acc: 0.7524

Epoch 65/87
----------
train Loss: 0.0071 Acc: 0.8316
val Loss: 0.0110 Acc: 0.7524

Epoch 66/87
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0071 Acc: 0.8378
val Loss: 0.0105 Acc: 0.7524

Epoch 67/87
----------
train Loss: 0.0069 Acc: 0.8378
val Loss: 0.0096 Acc: 0.7573

Epoch 68/87
----------
train Loss: 0.0073 Acc: 0.8212
val Loss: 0.0105 Acc: 0.7573

Epoch 69/87
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0070 Acc: 0.8295
val Loss: 0.0101 Acc: 0.7573

Epoch 70/87
----------
train Loss: 0.0073 Acc: 0.8170
val Loss: 0.0130 Acc: 0.7621

Epoch 71/87
----------
train Loss: 0.0073 Acc: 0.8254
val Loss: 0.0122 Acc: 0.7524

Epoch 72/87
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0072 Acc: 0.8420
val Loss: 0.0120 Acc: 0.7476

Epoch 73/87
----------
train Loss: 0.0072 Acc: 0.8337
val Loss: 0.0126 Acc: 0.7476

Epoch 74/87
----------
train Loss: 0.0069 Acc: 0.8358
val Loss: 0.0125 Acc: 0.7621

Epoch 75/87
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0072 Acc: 0.8274
val Loss: 0.0103 Acc: 0.7573

Epoch 76/87
----------
train Loss: 0.0072 Acc: 0.8233
val Loss: 0.0113 Acc: 0.7524

Epoch 77/87
----------
train Loss: 0.0070 Acc: 0.8254
val Loss: 0.0115 Acc: 0.7524

Epoch 78/87
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0073 Acc: 0.8295
val Loss: 0.0117 Acc: 0.7573

Epoch 79/87
----------
train Loss: 0.0072 Acc: 0.8233
val Loss: 0.0139 Acc: 0.7524

Epoch 80/87
----------
train Loss: 0.0071 Acc: 0.8233
val Loss: 0.0108 Acc: 0.7524

Epoch 81/87
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0071 Acc: 0.8191
val Loss: 0.0104 Acc: 0.7476

Epoch 82/87
----------
train Loss: 0.0073 Acc: 0.8150
val Loss: 0.0102 Acc: 0.7524

Epoch 83/87
----------
train Loss: 0.0073 Acc: 0.8212
val Loss: 0.0116 Acc: 0.7524

Epoch 84/87
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0071 Acc: 0.8212
val Loss: 0.0128 Acc: 0.7524

Epoch 85/87
----------
train Loss: 0.0070 Acc: 0.8295
val Loss: 0.0140 Acc: 0.7524

Epoch 86/87
----------
train Loss: 0.0073 Acc: 0.8191
val Loss: 0.0111 Acc: 0.7524

Epoch 87/87
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0072 Acc: 0.8212
val Loss: 0.0114 Acc: 0.7524

Training complete in 4m 16s
Best val Acc: 0.762136

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0070 Acc: 0.8482
val Loss: 0.0079 Acc: 0.7913

Epoch 1/87
----------
train Loss: 0.0045 Acc: 0.8857
val Loss: 0.0088 Acc: 0.8204

Epoch 2/87
----------
train Loss: 0.0024 Acc: 0.9522
val Loss: 0.0051 Acc: 0.8592

Epoch 3/87
----------
LR is set to 0.001
train Loss: 0.0015 Acc: 0.9813
val Loss: 0.0127 Acc: 0.8932

Epoch 4/87
----------
train Loss: 0.0014 Acc: 0.9792
val Loss: 0.0039 Acc: 0.8883

Epoch 5/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0057 Acc: 0.8883

Epoch 6/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9938
val Loss: 0.0039 Acc: 0.8883

Epoch 7/87
----------
train Loss: 0.0012 Acc: 0.9792
val Loss: 0.0088 Acc: 0.8883

Epoch 8/87
----------
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0043 Acc: 0.8883

Epoch 9/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0049 Acc: 0.8883

Epoch 10/87
----------
train Loss: 0.0011 Acc: 0.9771
val Loss: 0.0070 Acc: 0.8883

Epoch 11/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0066 Acc: 0.8883

Epoch 12/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0012 Acc: 0.9834
val Loss: 0.0044 Acc: 0.8883

Epoch 13/87
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0049 Acc: 0.8883

Epoch 14/87
----------
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0045 Acc: 0.8883

Epoch 15/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0071 Acc: 0.8883

Epoch 16/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0081 Acc: 0.8883

Epoch 17/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0057 Acc: 0.8883

Epoch 18/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0011 Acc: 0.9958
val Loss: 0.0051 Acc: 0.8883

Epoch 19/87
----------
train Loss: 0.0010 Acc: 0.9896
val Loss: 0.0073 Acc: 0.8883

Epoch 20/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0063 Acc: 0.8883

Epoch 21/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0049 Acc: 0.8883

Epoch 22/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0085 Acc: 0.8883

Epoch 23/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0039 Acc: 0.8883

Epoch 24/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0041 Acc: 0.8883

Epoch 25/87
----------
train Loss: 0.0012 Acc: 0.9875
val Loss: 0.0039 Acc: 0.8883

Epoch 26/87
----------
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0042 Acc: 0.8883

Epoch 27/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0012 Acc: 0.9834
val Loss: 0.0041 Acc: 0.8883

Epoch 28/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0071 Acc: 0.8883

Epoch 29/87
----------
train Loss: 0.0009 Acc: 0.9938
val Loss: 0.0042 Acc: 0.8883

Epoch 30/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0041 Acc: 0.8883

Epoch 31/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0042 Acc: 0.8883

Epoch 32/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0039 Acc: 0.8883

Epoch 33/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0065 Acc: 0.8883

Epoch 34/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0039 Acc: 0.8883

Epoch 35/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0045 Acc: 0.8883

Epoch 36/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0012 Acc: 0.9854
val Loss: 0.0058 Acc: 0.8883

Epoch 37/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0073 Acc: 0.8883

Epoch 38/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0073 Acc: 0.8883

Epoch 39/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0062 Acc: 0.8883

Epoch 40/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0040 Acc: 0.8883

Epoch 41/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0038 Acc: 0.8883

Epoch 42/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0011 Acc: 0.9938
val Loss: 0.0041 Acc: 0.8883

Epoch 43/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0052 Acc: 0.8883

Epoch 44/87
----------
train Loss: 0.0011 Acc: 0.9813
val Loss: 0.0074 Acc: 0.8883

Epoch 45/87
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0011 Acc: 0.9938
val Loss: 0.0041 Acc: 0.8883

Epoch 46/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0069 Acc: 0.8883

Epoch 47/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0047 Acc: 0.8883

Epoch 48/87
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0012 Acc: 0.9834
val Loss: 0.0043 Acc: 0.8883

Epoch 49/87
----------
train Loss: 0.0010 Acc: 0.9896
val Loss: 0.0064 Acc: 0.8883

Epoch 50/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0059 Acc: 0.8883

Epoch 51/87
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0048 Acc: 0.8883

Epoch 52/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0041 Acc: 0.8883

Epoch 53/87
----------
train Loss: 0.0011 Acc: 0.9792
val Loss: 0.0061 Acc: 0.8883

Epoch 54/87
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0010 Acc: 0.9834
val Loss: 0.0052 Acc: 0.8883

Epoch 55/87
----------
train Loss: 0.0010 Acc: 0.9896
val Loss: 0.0047 Acc: 0.8883

Epoch 56/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0043 Acc: 0.8883

Epoch 57/87
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0010 Acc: 0.9896
val Loss: 0.0051 Acc: 0.8883

Epoch 58/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0039 Acc: 0.8883

Epoch 59/87
----------
train Loss: 0.0011 Acc: 0.9792
val Loss: 0.0074 Acc: 0.8883

Epoch 60/87
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0011 Acc: 0.9917
val Loss: 0.0055 Acc: 0.8883

Epoch 61/87
----------
train Loss: 0.0012 Acc: 0.9792
val Loss: 0.0058 Acc: 0.8883

Epoch 62/87
----------
train Loss: 0.0012 Acc: 0.9834
val Loss: 0.0062 Acc: 0.8883

Epoch 63/87
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0058 Acc: 0.8883

Epoch 64/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0056 Acc: 0.8883

Epoch 65/87
----------
train Loss: 0.0011 Acc: 0.9792
val Loss: 0.0059 Acc: 0.8883

Epoch 66/87
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0057 Acc: 0.8883

Epoch 67/87
----------
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0055 Acc: 0.8883

Epoch 68/87
----------
train Loss: 0.0011 Acc: 0.9813
val Loss: 0.0047 Acc: 0.8883

Epoch 69/87
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0057 Acc: 0.8883

Epoch 70/87
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0061 Acc: 0.8883

Epoch 71/87
----------
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0046 Acc: 0.8883

Epoch 72/87
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0040 Acc: 0.8883

Epoch 73/87
----------
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0055 Acc: 0.8883

Epoch 74/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0086 Acc: 0.8883

Epoch 75/87
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0077 Acc: 0.8883

Epoch 76/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0052 Acc: 0.8883

Epoch 77/87
----------
train Loss: 0.0011 Acc: 0.9875
val Loss: 0.0047 Acc: 0.8883

Epoch 78/87
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0012 Acc: 0.9854
val Loss: 0.0063 Acc: 0.8883

Epoch 79/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0054 Acc: 0.8883

Epoch 80/87
----------
train Loss: 0.0011 Acc: 0.9917
val Loss: 0.0047 Acc: 0.8883

Epoch 81/87
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0057 Acc: 0.8883

Epoch 82/87
----------
train Loss: 0.0011 Acc: 0.9917
val Loss: 0.0061 Acc: 0.8883

Epoch 83/87
----------
train Loss: 0.0012 Acc: 0.9813
val Loss: 0.0058 Acc: 0.8883

Epoch 84/87
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0088 Acc: 0.8883

Epoch 85/87
----------
train Loss: 0.0011 Acc: 0.9834
val Loss: 0.0040 Acc: 0.8883

Epoch 86/87
----------
train Loss: 0.0011 Acc: 0.9896
val Loss: 0.0073 Acc: 0.8883

Epoch 87/87
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0010 Acc: 0.9917
val Loss: 0.0046 Acc: 0.8883

Training complete in 4m 29s
Best val Acc: 0.893204

---Testing---
Test accuracy: 0.953421
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 80 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 96 %
mean: 0.9398430984416501, std: 0.058233802886289814

Model saved in "./weights/shark_[0.98]_mean[0.98]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 53, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0216 Acc: 0.2310
val Loss: 0.0229 Acc: 0.5147

Epoch 1/52
----------
train Loss: 0.0135 Acc: 0.6300
val Loss: 0.0175 Acc: 0.6176

Epoch 2/52
----------
train Loss: 0.0086 Acc: 0.7464
val Loss: 0.0115 Acc: 0.7353

Epoch 3/52
----------
train Loss: 0.0066 Acc: 0.8207
val Loss: 0.0118 Acc: 0.6912

Epoch 4/52
----------
train Loss: 0.0057 Acc: 0.8530
val Loss: 0.0089 Acc: 0.8088

Epoch 5/52
----------
train Loss: 0.0047 Acc: 0.8659
val Loss: 0.0095 Acc: 0.7941

Epoch 6/52
----------
train Loss: 0.0043 Acc: 0.8821
val Loss: 0.0105 Acc: 0.7206

Epoch 7/52
----------
train Loss: 0.0047 Acc: 0.8627
val Loss: 0.0094 Acc: 0.7500

Epoch 8/52
----------
LR is set to 0.001
train Loss: 0.0043 Acc: 0.9047
val Loss: 0.0088 Acc: 0.7941

Epoch 9/52
----------
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0089 Acc: 0.7941

Epoch 10/52
----------
train Loss: 0.0036 Acc: 0.9031
val Loss: 0.0096 Acc: 0.7353

Epoch 11/52
----------
train Loss: 0.0031 Acc: 0.9111
val Loss: 0.0096 Acc: 0.7647

Epoch 12/52
----------
train Loss: 0.0031 Acc: 0.9208
val Loss: 0.0092 Acc: 0.7794

Epoch 13/52
----------
train Loss: 0.0032 Acc: 0.9063
val Loss: 0.0092 Acc: 0.7647

Epoch 14/52
----------
train Loss: 0.0031 Acc: 0.9257
val Loss: 0.0094 Acc: 0.7794

Epoch 15/52
----------
train Loss: 0.0034 Acc: 0.9160
val Loss: 0.0093 Acc: 0.7794

Epoch 16/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9370
val Loss: 0.0091 Acc: 0.7794

Epoch 17/52
----------
train Loss: 0.0034 Acc: 0.9305
val Loss: 0.0091 Acc: 0.7794

Epoch 18/52
----------
train Loss: 0.0031 Acc: 0.9354
val Loss: 0.0091 Acc: 0.7794

Epoch 19/52
----------
train Loss: 0.0032 Acc: 0.9208
val Loss: 0.0092 Acc: 0.7794

Epoch 20/52
----------
train Loss: 0.0028 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7794

Epoch 21/52
----------
train Loss: 0.0033 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7941

Epoch 22/52
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0091 Acc: 0.7941

Epoch 23/52
----------
train Loss: 0.0032 Acc: 0.9289
val Loss: 0.0090 Acc: 0.7941

Epoch 24/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0031 Acc: 0.9176
val Loss: 0.0091 Acc: 0.7794

Epoch 25/52
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0090 Acc: 0.7794

Epoch 26/52
----------
train Loss: 0.0030 Acc: 0.9338
val Loss: 0.0089 Acc: 0.7941

Epoch 27/52
----------
train Loss: 0.0030 Acc: 0.9289
val Loss: 0.0090 Acc: 0.7941

Epoch 28/52
----------
train Loss: 0.0029 Acc: 0.9257
val Loss: 0.0091 Acc: 0.7794

Epoch 29/52
----------
train Loss: 0.0032 Acc: 0.9338
val Loss: 0.0091 Acc: 0.7647

Epoch 30/52
----------
train Loss: 0.0031 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7794

Epoch 31/52
----------
train Loss: 0.0030 Acc: 0.9354
val Loss: 0.0091 Acc: 0.7794

Epoch 32/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0031 Acc: 0.9370
val Loss: 0.0092 Acc: 0.7794

Epoch 33/52
----------
train Loss: 0.0029 Acc: 0.9386
val Loss: 0.0091 Acc: 0.7941

Epoch 34/52
----------
train Loss: 0.0032 Acc: 0.9402
val Loss: 0.0091 Acc: 0.7794

Epoch 35/52
----------
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0091 Acc: 0.7941

Epoch 36/52
----------
train Loss: 0.0034 Acc: 0.9176
val Loss: 0.0091 Acc: 0.7941

Epoch 37/52
----------
train Loss: 0.0032 Acc: 0.9160
val Loss: 0.0092 Acc: 0.7941

Epoch 38/52
----------
train Loss: 0.0030 Acc: 0.9321
val Loss: 0.0091 Acc: 0.7794

Epoch 39/52
----------
train Loss: 0.0032 Acc: 0.9241
val Loss: 0.0091 Acc: 0.7794

Epoch 40/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0033 Acc: 0.9289
val Loss: 0.0092 Acc: 0.7794

Epoch 41/52
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0091 Acc: 0.7794

Epoch 42/52
----------
train Loss: 0.0031 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7794

Epoch 43/52
----------
train Loss: 0.0032 Acc: 0.9241
val Loss: 0.0092 Acc: 0.7794

Epoch 44/52
----------
train Loss: 0.0034 Acc: 0.9241
val Loss: 0.0093 Acc: 0.7794

Epoch 45/52
----------
train Loss: 0.0029 Acc: 0.9273
val Loss: 0.0091 Acc: 0.7794

Epoch 46/52
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7794

Epoch 47/52
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0090 Acc: 0.7794

Epoch 48/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0091 Acc: 0.7794

Epoch 49/52
----------
train Loss: 0.0033 Acc: 0.9241
val Loss: 0.0091 Acc: 0.7794

Epoch 50/52
----------
train Loss: 0.0030 Acc: 0.9354
val Loss: 0.0092 Acc: 0.7794

Epoch 51/52
----------
train Loss: 0.0035 Acc: 0.9354
val Loss: 0.0091 Acc: 0.7794

Epoch 52/52
----------
train Loss: 0.0033 Acc: 0.9095
val Loss: 0.0091 Acc: 0.7794

Training complete in 2m 35s
Best val Acc: 0.808824

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0043 Acc: 0.8885
val Loss: 0.0115 Acc: 0.7500

Epoch 1/52
----------
train Loss: 0.0026 Acc: 0.9370
val Loss: 0.0101 Acc: 0.7941

Epoch 2/52
----------
train Loss: 0.0016 Acc: 0.9742
val Loss: 0.0088 Acc: 0.8235

Epoch 3/52
----------
train Loss: 0.0010 Acc: 0.9871
val Loss: 0.0082 Acc: 0.7941

Epoch 4/52
----------
train Loss: 0.0013 Acc: 0.9725
val Loss: 0.0087 Acc: 0.8235

Epoch 5/52
----------
train Loss: 0.0008 Acc: 0.9822
val Loss: 0.0109 Acc: 0.8235

Epoch 6/52
----------
train Loss: 0.0011 Acc: 0.9887
val Loss: 0.0091 Acc: 0.8382

Epoch 7/52
----------
train Loss: 0.0006 Acc: 0.9935
val Loss: 0.0110 Acc: 0.7941

Epoch 8/52
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9871
val Loss: 0.0093 Acc: 0.8382

Epoch 9/52
----------
train Loss: 0.0007 Acc: 0.9871
val Loss: 0.0080 Acc: 0.8529

Epoch 10/52
----------
train Loss: 0.0003 Acc: 0.9935
val Loss: 0.0079 Acc: 0.8382

Epoch 11/52
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0079 Acc: 0.8382

Epoch 12/52
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8382

Epoch 13/52
----------
train Loss: 0.0009 Acc: 0.9903
val Loss: 0.0075 Acc: 0.8382

Epoch 14/52
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0085 Acc: 0.8235

Epoch 15/52
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0087 Acc: 0.8235

Epoch 16/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0084 Acc: 0.8235

Epoch 17/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0082 Acc: 0.8235

Epoch 18/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8235

Epoch 19/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0084 Acc: 0.8235

Epoch 20/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0084 Acc: 0.8235

Epoch 21/52
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0086 Acc: 0.8235

Epoch 22/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0085 Acc: 0.8235

Epoch 23/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0087 Acc: 0.8235

Epoch 24/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0089 Acc: 0.8235

Epoch 25/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0088 Acc: 0.8235

Epoch 26/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0086 Acc: 0.8235

Epoch 27/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0084 Acc: 0.8235

Epoch 28/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0085 Acc: 0.8235

Epoch 29/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0084 Acc: 0.8235

Epoch 30/52
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8235

Epoch 31/52
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0083 Acc: 0.8235

Epoch 32/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0083 Acc: 0.8235

Epoch 33/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0083 Acc: 0.8235

Epoch 34/52
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0084 Acc: 0.8235

Epoch 35/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8235

Epoch 36/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0087 Acc: 0.8235

Epoch 37/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0085 Acc: 0.8235

Epoch 38/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8235

Epoch 39/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8235

Epoch 40/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0083 Acc: 0.8235

Epoch 41/52
----------
train Loss: 0.0004 Acc: 0.9952
val Loss: 0.0085 Acc: 0.8235

Epoch 42/52
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0083 Acc: 0.8235

Epoch 43/52
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0084 Acc: 0.8235

Epoch 44/52
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0086 Acc: 0.8235

Epoch 45/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0085 Acc: 0.8235

Epoch 46/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8235

Epoch 47/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8235

Epoch 48/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8235

Epoch 49/52
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0087 Acc: 0.8235

Epoch 50/52
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8235

Epoch 51/52
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8235

Epoch 52/52
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0083 Acc: 0.8235

Training complete in 2m 46s
Best val Acc: 0.852941

---Testing---
Test accuracy: 0.982533
--------------------
Accuracy of Carcharhiniformes : 100 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 97 %
mean: 0.9748501936659831, std: 0.025938535413552233
--------------------

run info[val: 0.15, epoch: 64, randcrop: False, decay: 4]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0203 Acc: 0.2568
val Loss: 0.0292 Acc: 0.2816

Epoch 1/63
----------
train Loss: 0.0132 Acc: 0.5873
val Loss: 0.0166 Acc: 0.6019

Epoch 2/63
----------
train Loss: 0.0081 Acc: 0.7671
val Loss: 0.0103 Acc: 0.6990

Epoch 3/63
----------
train Loss: 0.0054 Acc: 0.8630
val Loss: 0.0082 Acc: 0.7087

Epoch 4/63
----------
LR is set to 0.001
train Loss: 0.0048 Acc: 0.8699
val Loss: 0.0097 Acc: 0.7476

Epoch 5/63
----------
train Loss: 0.0044 Acc: 0.8973
val Loss: 0.0226 Acc: 0.7573

Epoch 6/63
----------
train Loss: 0.0044 Acc: 0.8887
val Loss: 0.0199 Acc: 0.7573

Epoch 7/63
----------
train Loss: 0.0043 Acc: 0.8870
val Loss: 0.0165 Acc: 0.7670

Epoch 8/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0042 Acc: 0.8973
val Loss: 0.0145 Acc: 0.7670

Epoch 9/63
----------
train Loss: 0.0043 Acc: 0.8904
val Loss: 0.0084 Acc: 0.7670

Epoch 10/63
----------
train Loss: 0.0042 Acc: 0.8921
val Loss: 0.0120 Acc: 0.7767

Epoch 11/63
----------
train Loss: 0.0042 Acc: 0.9075
val Loss: 0.0167 Acc: 0.7767

Epoch 12/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0042 Acc: 0.9007
val Loss: 0.0084 Acc: 0.7767

Epoch 13/63
----------
train Loss: 0.0041 Acc: 0.8904
val Loss: 0.0158 Acc: 0.7670

Epoch 14/63
----------
train Loss: 0.0043 Acc: 0.8955
val Loss: 0.0113 Acc: 0.7767

Epoch 15/63
----------
train Loss: 0.0042 Acc: 0.8870
val Loss: 0.0159 Acc: 0.7767

Epoch 16/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0042 Acc: 0.8990
val Loss: 0.0132 Acc: 0.7767

Epoch 17/63
----------
train Loss: 0.0042 Acc: 0.8990
val Loss: 0.0120 Acc: 0.7767

Epoch 18/63
----------
train Loss: 0.0042 Acc: 0.8818
val Loss: 0.0087 Acc: 0.7767

Epoch 19/63
----------
train Loss: 0.0043 Acc: 0.8887
val Loss: 0.0161 Acc: 0.7767

Epoch 20/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0043 Acc: 0.8870
val Loss: 0.0192 Acc: 0.7767

Epoch 21/63
----------
train Loss: 0.0041 Acc: 0.9110
val Loss: 0.0088 Acc: 0.7670

Epoch 22/63
----------
train Loss: 0.0042 Acc: 0.8887
val Loss: 0.0094 Acc: 0.7670

Epoch 23/63
----------
train Loss: 0.0042 Acc: 0.9007
val Loss: 0.0160 Acc: 0.7573

Epoch 24/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0042 Acc: 0.8955
val Loss: 0.0075 Acc: 0.7670

Epoch 25/63
----------
train Loss: 0.0043 Acc: 0.8904
val Loss: 0.0127 Acc: 0.7573

Epoch 26/63
----------
train Loss: 0.0042 Acc: 0.9007
val Loss: 0.0091 Acc: 0.7573

Epoch 27/63
----------
train Loss: 0.0042 Acc: 0.9007
val Loss: 0.0194 Acc: 0.7573

Epoch 28/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0043 Acc: 0.8836
val Loss: 0.0205 Acc: 0.7670

Epoch 29/63
----------
train Loss: 0.0042 Acc: 0.9058
val Loss: 0.0160 Acc: 0.7670

Epoch 30/63
----------
train Loss: 0.0042 Acc: 0.8904
val Loss: 0.0160 Acc: 0.7767

Epoch 31/63
----------
train Loss: 0.0043 Acc: 0.8801
val Loss: 0.0127 Acc: 0.7573

Epoch 32/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0041 Acc: 0.8938
val Loss: 0.0125 Acc: 0.7670

Epoch 33/63
----------
train Loss: 0.0041 Acc: 0.9024
val Loss: 0.0177 Acc: 0.7670

Epoch 34/63
----------
train Loss: 0.0042 Acc: 0.8990
val Loss: 0.0081 Acc: 0.7670

Epoch 35/63
----------
train Loss: 0.0044 Acc: 0.8887
val Loss: 0.0106 Acc: 0.7573

Epoch 36/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0042 Acc: 0.8938
val Loss: 0.0125 Acc: 0.7670

Epoch 37/63
----------
train Loss: 0.0042 Acc: 0.8938
val Loss: 0.0110 Acc: 0.7670

Epoch 38/63
----------
train Loss: 0.0041 Acc: 0.8921
val Loss: 0.0236 Acc: 0.7670

Epoch 39/63
----------
train Loss: 0.0043 Acc: 0.9024
val Loss: 0.0136 Acc: 0.7670

Epoch 40/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0042 Acc: 0.9024
val Loss: 0.0121 Acc: 0.7670

Epoch 41/63
----------
train Loss: 0.0042 Acc: 0.8938
val Loss: 0.0208 Acc: 0.7670

Epoch 42/63
----------
train Loss: 0.0042 Acc: 0.8836
val Loss: 0.0105 Acc: 0.7670

Epoch 43/63
----------
train Loss: 0.0043 Acc: 0.8904
val Loss: 0.0164 Acc: 0.7670

Epoch 44/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0042 Acc: 0.8921
val Loss: 0.0158 Acc: 0.7670

Epoch 45/63
----------
train Loss: 0.0041 Acc: 0.8921
val Loss: 0.0101 Acc: 0.7573

Epoch 46/63
----------
train Loss: 0.0043 Acc: 0.8938
val Loss: 0.0135 Acc: 0.7670

Epoch 47/63
----------
train Loss: 0.0043 Acc: 0.8853
val Loss: 0.0199 Acc: 0.7573

Epoch 48/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0041 Acc: 0.8955
val Loss: 0.0098 Acc: 0.7573

Epoch 49/63
----------
train Loss: 0.0043 Acc: 0.8973
val Loss: 0.0154 Acc: 0.7670

Epoch 50/63
----------
train Loss: 0.0042 Acc: 0.8904
val Loss: 0.0158 Acc: 0.7573

Epoch 51/63
----------
train Loss: 0.0041 Acc: 0.9024
val Loss: 0.0099 Acc: 0.7767

Epoch 52/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0041 Acc: 0.9058
val Loss: 0.0148 Acc: 0.7573

Epoch 53/63
----------
train Loss: 0.0042 Acc: 0.8836
val Loss: 0.0081 Acc: 0.7767

Epoch 54/63
----------
train Loss: 0.0042 Acc: 0.8973
val Loss: 0.0203 Acc: 0.7767

Epoch 55/63
----------
train Loss: 0.0042 Acc: 0.8955
val Loss: 0.0140 Acc: 0.7767

Epoch 56/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0042 Acc: 0.9007
val Loss: 0.0075 Acc: 0.7767

Epoch 57/63
----------
train Loss: 0.0043 Acc: 0.8938
val Loss: 0.0110 Acc: 0.7767

Epoch 58/63
----------
train Loss: 0.0041 Acc: 0.9058
val Loss: 0.0140 Acc: 0.7670

Epoch 59/63
----------
train Loss: 0.0042 Acc: 0.8990
val Loss: 0.0212 Acc: 0.7670

Epoch 60/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0041 Acc: 0.9041
val Loss: 0.0246 Acc: 0.7670

Epoch 61/63
----------
train Loss: 0.0041 Acc: 0.8887
val Loss: 0.0168 Acc: 0.7670

Epoch 62/63
----------
train Loss: 0.0042 Acc: 0.9041
val Loss: 0.0111 Acc: 0.7573

Epoch 63/63
----------
train Loss: 0.0042 Acc: 0.8921
val Loss: 0.0087 Acc: 0.7573

Training complete in 3m 16s
Best val Acc: 0.776699

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0040 Acc: 0.8990
val Loss: 0.0152 Acc: 0.8155

Epoch 1/63
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0079 Acc: 0.8155

Epoch 2/63
----------
train Loss: 0.0010 Acc: 0.9795
val Loss: 0.0165 Acc: 0.8447

Epoch 3/63
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8544

Epoch 4/63
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9966
val Loss: 0.0058 Acc: 0.8447

Epoch 5/63
----------
train Loss: 0.0003 Acc: 0.9966
val Loss: 0.0136 Acc: 0.8350

Epoch 6/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0154 Acc: 0.8350

Epoch 7/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0126 Acc: 0.8350

Epoch 8/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0049 Acc: 0.8350

Epoch 9/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8350

Epoch 10/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 11/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8350

Epoch 12/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8350

Epoch 13/63
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0045 Acc: 0.8350

Epoch 14/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0147 Acc: 0.8350

Epoch 15/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0051 Acc: 0.8350

Epoch 16/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8350

Epoch 17/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8350

Epoch 18/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0135 Acc: 0.8350

Epoch 19/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 20/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8350

Epoch 21/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8350

Epoch 22/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8350

Epoch 23/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8350

Epoch 24/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0171 Acc: 0.8350

Epoch 25/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0045 Acc: 0.8350

Epoch 26/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 27/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8350

Epoch 28/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0199 Acc: 0.8350

Epoch 29/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 30/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 31/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0137 Acc: 0.8350

Epoch 32/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 33/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 34/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0157 Acc: 0.8350

Epoch 35/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 36/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0140 Acc: 0.8350

Epoch 37/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8350

Epoch 38/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0085 Acc: 0.8350

Epoch 39/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8350

Epoch 40/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8350

Epoch 41/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 42/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8350

Epoch 43/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 44/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 45/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0159 Acc: 0.8350

Epoch 46/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8350

Epoch 47/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0074 Acc: 0.8350

Epoch 48/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0066 Acc: 0.8350

Epoch 49/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 50/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 51/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 52/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0125 Acc: 0.8350

Epoch 53/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 54/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 55/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 56/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0161 Acc: 0.8350

Epoch 57/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8350

Epoch 58/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0187 Acc: 0.8350

Epoch 59/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 60/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0084 Acc: 0.8350

Epoch 61/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 62/63
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0100 Acc: 0.8350

Epoch 63/63
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Training complete in 3m 26s
Best val Acc: 0.854369

---Testing---
Test accuracy: 0.975255
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9667535315561632, std: 0.03073125645508669
--------------------

run info[val: 0.2, epoch: 57, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0214 Acc: 0.2618
val Loss: 0.0246 Acc: 0.2774

Epoch 1/56
----------
train Loss: 0.0141 Acc: 0.5945
val Loss: 0.0157 Acc: 0.6496

Epoch 2/56
----------
train Loss: 0.0093 Acc: 0.7382
val Loss: 0.0098 Acc: 0.8102

Epoch 3/56
----------
train Loss: 0.0062 Acc: 0.8400
val Loss: 0.0095 Acc: 0.7956

Epoch 4/56
----------
train Loss: 0.0049 Acc: 0.8673
val Loss: 0.0083 Acc: 0.8321

Epoch 5/56
----------
train Loss: 0.0040 Acc: 0.9145
val Loss: 0.0075 Acc: 0.8394

Epoch 6/56
----------
train Loss: 0.0034 Acc: 0.9145
val Loss: 0.0078 Acc: 0.8248

Epoch 7/56
----------
train Loss: 0.0032 Acc: 0.9200
val Loss: 0.0083 Acc: 0.8540

Epoch 8/56
----------
train Loss: 0.0030 Acc: 0.9273
val Loss: 0.0079 Acc: 0.8321

Epoch 9/56
----------
train Loss: 0.0026 Acc: 0.9491
val Loss: 0.0070 Acc: 0.8394

Epoch 10/56
----------
train Loss: 0.0028 Acc: 0.9364
val Loss: 0.0073 Acc: 0.8613

Epoch 11/56
----------
train Loss: 0.0027 Acc: 0.9309
val Loss: 0.0070 Acc: 0.8321

Epoch 12/56
----------
train Loss: 0.0024 Acc: 0.9527
val Loss: 0.0060 Acc: 0.8321

Epoch 13/56
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0071 Acc: 0.8394

Epoch 14/56
----------
LR is set to 0.001
train Loss: 0.0020 Acc: 0.9673
val Loss: 0.0078 Acc: 0.8321

Epoch 15/56
----------
train Loss: 0.0017 Acc: 0.9745
val Loss: 0.0069 Acc: 0.8321

Epoch 16/56
----------
train Loss: 0.0018 Acc: 0.9782
val Loss: 0.0084 Acc: 0.8394

Epoch 17/56
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0078 Acc: 0.8321

Epoch 18/56
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0074 Acc: 0.8394

Epoch 19/56
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0076 Acc: 0.8467

Epoch 20/56
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0065 Acc: 0.8394

Epoch 21/56
----------
train Loss: 0.0017 Acc: 0.9855
val Loss: 0.0070 Acc: 0.8394

Epoch 22/56
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0071 Acc: 0.8394

Epoch 23/56
----------
train Loss: 0.0015 Acc: 0.9818
val Loss: 0.0068 Acc: 0.8394

Epoch 24/56
----------
train Loss: 0.0017 Acc: 0.9800
val Loss: 0.0071 Acc: 0.8248

Epoch 25/56
----------
train Loss: 0.0018 Acc: 0.9764
val Loss: 0.0065 Acc: 0.8394

Epoch 26/56
----------
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0072 Acc: 0.8394

Epoch 27/56
----------
train Loss: 0.0016 Acc: 0.9873
val Loss: 0.0073 Acc: 0.8467

Epoch 28/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0065 Acc: 0.8467

Epoch 29/56
----------
train Loss: 0.0017 Acc: 0.9800
val Loss: 0.0075 Acc: 0.8467

Epoch 30/56
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0063 Acc: 0.8467

Epoch 31/56
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0060 Acc: 0.8467

Epoch 32/56
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8467

Epoch 33/56
----------
train Loss: 0.0017 Acc: 0.9800
val Loss: 0.0074 Acc: 0.8467

Epoch 34/56
----------
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0062 Acc: 0.8467

Epoch 35/56
----------
train Loss: 0.0016 Acc: 0.9909
val Loss: 0.0067 Acc: 0.8467

Epoch 36/56
----------
train Loss: 0.0017 Acc: 0.9855
val Loss: 0.0074 Acc: 0.8467

Epoch 37/56
----------
train Loss: 0.0016 Acc: 0.9782
val Loss: 0.0064 Acc: 0.8467

Epoch 38/56
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0065 Acc: 0.8467

Epoch 39/56
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0067 Acc: 0.8467

Epoch 40/56
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0060 Acc: 0.8467

Epoch 41/56
----------
train Loss: 0.0015 Acc: 0.9836
val Loss: 0.0063 Acc: 0.8467

Epoch 42/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9873
val Loss: 0.0066 Acc: 0.8467

Epoch 43/56
----------
train Loss: 0.0016 Acc: 0.9818
val Loss: 0.0071 Acc: 0.8467

Epoch 44/56
----------
train Loss: 0.0017 Acc: 0.9818
val Loss: 0.0071 Acc: 0.8467

Epoch 45/56
----------
train Loss: 0.0016 Acc: 0.9836
val Loss: 0.0074 Acc: 0.8467

Epoch 46/56
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0073 Acc: 0.8467

Epoch 47/56
----------
train Loss: 0.0016 Acc: 0.9745
val Loss: 0.0076 Acc: 0.8467

Epoch 48/56
----------
train Loss: 0.0015 Acc: 0.9836
val Loss: 0.0075 Acc: 0.8467

Epoch 49/56
----------
train Loss: 0.0015 Acc: 0.9836
val Loss: 0.0071 Acc: 0.8394

Epoch 50/56
----------
train Loss: 0.0017 Acc: 0.9855
val Loss: 0.0080 Acc: 0.8394

Epoch 51/56
----------
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0064 Acc: 0.8321

Epoch 52/56
----------
train Loss: 0.0015 Acc: 0.9800
val Loss: 0.0071 Acc: 0.8394

Epoch 53/56
----------
train Loss: 0.0017 Acc: 0.9782
val Loss: 0.0060 Acc: 0.8394

Epoch 54/56
----------
train Loss: 0.0016 Acc: 0.9800
val Loss: 0.0064 Acc: 0.8394

Epoch 55/56
----------
train Loss: 0.0017 Acc: 0.9836
val Loss: 0.0069 Acc: 0.8467

Epoch 56/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0080 Acc: 0.8467

Training complete in 2m 52s
Best val Acc: 0.861314

---Fine tuning.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0068 Acc: 0.8540

Epoch 1/56
----------
train Loss: 0.0012 Acc: 0.9855
val Loss: 0.0072 Acc: 0.8613

Epoch 2/56
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0063 Acc: 0.8540

Epoch 3/56
----------
train Loss: 0.0003 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8759

Epoch 4/56
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8467

Epoch 5/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8394

Epoch 6/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8540

Epoch 7/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8540

Epoch 8/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8613

Epoch 9/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8686

Epoch 10/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8759

Epoch 11/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 12/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 13/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 14/56
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 15/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8759

Epoch 16/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8686

Epoch 17/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 18/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 19/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 20/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 21/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8686

Epoch 22/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8686

Epoch 23/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8686

Epoch 24/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 25/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8686

Epoch 26/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 27/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8686

Epoch 28/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 29/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8686

Epoch 30/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 31/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8686

Epoch 32/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 33/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 34/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8686

Epoch 35/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 36/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8686

Epoch 37/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8686

Epoch 38/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 39/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 40/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 41/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 42/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 43/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 44/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 45/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 46/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8686

Epoch 47/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 48/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8686

Epoch 49/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 50/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 51/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 52/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 53/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 54/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 55/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 56/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Training complete in 3m 2s
Best val Acc: 0.875912

---Testing---
Test accuracy: 0.973799
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.966672750745075, std: 0.02642908348973817
--------------------

run info[val: 0.25, epoch: 58, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0218 Acc: 0.2616
val Loss: 0.0192 Acc: 0.4795

Epoch 1/57
----------
train Loss: 0.0158 Acc: 0.5271
val Loss: 0.0131 Acc: 0.6608

Epoch 2/57
----------
train Loss: 0.0118 Acc: 0.7016
val Loss: 0.0115 Acc: 0.6608

Epoch 3/57
----------
train Loss: 0.0089 Acc: 0.7132
val Loss: 0.0081 Acc: 0.7719

Epoch 4/57
----------
train Loss: 0.0067 Acc: 0.8023
val Loss: 0.0088 Acc: 0.7836

Epoch 5/57
----------
train Loss: 0.0068 Acc: 0.8217
val Loss: 0.0073 Acc: 0.8129

Epoch 6/57
----------
train Loss: 0.0049 Acc: 0.8702
val Loss: 0.0072 Acc: 0.7895

Epoch 7/57
----------
train Loss: 0.0048 Acc: 0.8857
val Loss: 0.0072 Acc: 0.8421

Epoch 8/57
----------
train Loss: 0.0051 Acc: 0.8779
val Loss: 0.0063 Acc: 0.8304

Epoch 9/57
----------
train Loss: 0.0041 Acc: 0.8837
val Loss: 0.0065 Acc: 0.8246

Epoch 10/57
----------
train Loss: 0.0040 Acc: 0.8876
val Loss: 0.0062 Acc: 0.8363

Epoch 11/57
----------
train Loss: 0.0035 Acc: 0.9089
val Loss: 0.0069 Acc: 0.8246

Epoch 12/57
----------
train Loss: 0.0034 Acc: 0.9128
val Loss: 0.0065 Acc: 0.8304

Epoch 13/57
----------
LR is set to 0.001
train Loss: 0.0027 Acc: 0.9438
val Loss: 0.0062 Acc: 0.8304

Epoch 14/57
----------
train Loss: 0.0028 Acc: 0.9438
val Loss: 0.0063 Acc: 0.8596

Epoch 15/57
----------
train Loss: 0.0027 Acc: 0.9419
val Loss: 0.0060 Acc: 0.8480

Epoch 16/57
----------
train Loss: 0.0027 Acc: 0.9554
val Loss: 0.0061 Acc: 0.8421

Epoch 17/57
----------
train Loss: 0.0027 Acc: 0.9593
val Loss: 0.0061 Acc: 0.8421

Epoch 18/57
----------
train Loss: 0.0024 Acc: 0.9477
val Loss: 0.0062 Acc: 0.8596

Epoch 19/57
----------
train Loss: 0.0024 Acc: 0.9438
val Loss: 0.0058 Acc: 0.8538

Epoch 20/57
----------
train Loss: 0.0029 Acc: 0.9496
val Loss: 0.0064 Acc: 0.8538

Epoch 21/57
----------
train Loss: 0.0027 Acc: 0.9419
val Loss: 0.0060 Acc: 0.8480

Epoch 22/57
----------
train Loss: 0.0025 Acc: 0.9419
val Loss: 0.0063 Acc: 0.8480

Epoch 23/57
----------
train Loss: 0.0024 Acc: 0.9516
val Loss: 0.0060 Acc: 0.8596

Epoch 24/57
----------
train Loss: 0.0029 Acc: 0.9516
val Loss: 0.0061 Acc: 0.8538

Epoch 25/57
----------
train Loss: 0.0027 Acc: 0.9535
val Loss: 0.0062 Acc: 0.8655

Epoch 26/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0022 Acc: 0.9593
val Loss: 0.0061 Acc: 0.8538

Epoch 27/57
----------
train Loss: 0.0027 Acc: 0.9574
val Loss: 0.0061 Acc: 0.8655

Epoch 28/57
----------
train Loss: 0.0034 Acc: 0.9341
val Loss: 0.0063 Acc: 0.8596

Epoch 29/57
----------
train Loss: 0.0030 Acc: 0.9477
val Loss: 0.0063 Acc: 0.8596

Epoch 30/57
----------
train Loss: 0.0026 Acc: 0.9457
val Loss: 0.0060 Acc: 0.8596

Epoch 31/57
----------
train Loss: 0.0026 Acc: 0.9360
val Loss: 0.0063 Acc: 0.8596

Epoch 32/57
----------
train Loss: 0.0025 Acc: 0.9651
val Loss: 0.0062 Acc: 0.8596

Epoch 33/57
----------
train Loss: 0.0025 Acc: 0.9554
val Loss: 0.0060 Acc: 0.8596

Epoch 34/57
----------
train Loss: 0.0024 Acc: 0.9516
val Loss: 0.0062 Acc: 0.8596

Epoch 35/57
----------
train Loss: 0.0021 Acc: 0.9535
val Loss: 0.0061 Acc: 0.8596

Epoch 36/57
----------
train Loss: 0.0025 Acc: 0.9554
val Loss: 0.0061 Acc: 0.8480

Epoch 37/57
----------
train Loss: 0.0024 Acc: 0.9632
val Loss: 0.0063 Acc: 0.8480

Epoch 38/57
----------
train Loss: 0.0026 Acc: 0.9535
val Loss: 0.0061 Acc: 0.8538

Epoch 39/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0024 Acc: 0.9516
val Loss: 0.0062 Acc: 0.8480

Epoch 40/57
----------
train Loss: 0.0030 Acc: 0.9399
val Loss: 0.0062 Acc: 0.8538

Epoch 41/57
----------
train Loss: 0.0024 Acc: 0.9516
val Loss: 0.0062 Acc: 0.8596

Epoch 42/57
----------
train Loss: 0.0024 Acc: 0.9457
val Loss: 0.0062 Acc: 0.8596

Epoch 43/57
----------
train Loss: 0.0025 Acc: 0.9535
val Loss: 0.0064 Acc: 0.8538

Epoch 44/57
----------
train Loss: 0.0031 Acc: 0.9283
val Loss: 0.0060 Acc: 0.8538

Epoch 45/57
----------
train Loss: 0.0026 Acc: 0.9438
val Loss: 0.0061 Acc: 0.8538

Epoch 46/57
----------
train Loss: 0.0026 Acc: 0.9438
val Loss: 0.0062 Acc: 0.8538

Epoch 47/57
----------
train Loss: 0.0025 Acc: 0.9574
val Loss: 0.0061 Acc: 0.8655

Epoch 48/57
----------
train Loss: 0.0022 Acc: 0.9535
val Loss: 0.0058 Acc: 0.8596

Epoch 49/57
----------
train Loss: 0.0028 Acc: 0.9516
val Loss: 0.0061 Acc: 0.8596

Epoch 50/57
----------
train Loss: 0.0020 Acc: 0.9632
val Loss: 0.0063 Acc: 0.8596

Epoch 51/57
----------
train Loss: 0.0027 Acc: 0.9516
val Loss: 0.0062 Acc: 0.8538

Epoch 52/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0026 Acc: 0.9457
val Loss: 0.0060 Acc: 0.8596

Epoch 53/57
----------
train Loss: 0.0028 Acc: 0.9302
val Loss: 0.0063 Acc: 0.8480

Epoch 54/57
----------
train Loss: 0.0032 Acc: 0.9302
val Loss: 0.0064 Acc: 0.8538

Epoch 55/57
----------
train Loss: 0.0027 Acc: 0.9302
val Loss: 0.0065 Acc: 0.8538

Epoch 56/57
----------
train Loss: 0.0031 Acc: 0.9438
val Loss: 0.0060 Acc: 0.8480

Epoch 57/57
----------
train Loss: 0.0024 Acc: 0.9535
val Loss: 0.0061 Acc: 0.8538

Training complete in 2m 53s
Best val Acc: 0.865497

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0025 Acc: 0.9457
val Loss: 0.0058 Acc: 0.8772

Epoch 1/57
----------
train Loss: 0.0021 Acc: 0.9593
val Loss: 0.0058 Acc: 0.8713

Epoch 2/57
----------
train Loss: 0.0012 Acc: 0.9690
val Loss: 0.0083 Acc: 0.8187

Epoch 3/57
----------
train Loss: 0.0009 Acc: 0.9767
val Loss: 0.0065 Acc: 0.8596

Epoch 4/57
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0057 Acc: 0.8713

Epoch 5/57
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0068 Acc: 0.8655

Epoch 6/57
----------
train Loss: 0.0008 Acc: 0.9845
val Loss: 0.0076 Acc: 0.8480

Epoch 7/57
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0064 Acc: 0.8538

Epoch 8/57
----------
train Loss: 0.0010 Acc: 0.9845
val Loss: 0.0060 Acc: 0.8421

Epoch 9/57
----------
train Loss: 0.0008 Acc: 0.9884
val Loss: 0.0074 Acc: 0.8363

Epoch 10/57
----------
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0079 Acc: 0.8246

Epoch 11/57
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0070 Acc: 0.8655

Epoch 12/57
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0073 Acc: 0.8421

Epoch 13/57
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9942
val Loss: 0.0076 Acc: 0.8363

Epoch 14/57
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0077 Acc: 0.8246

Epoch 15/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0076 Acc: 0.8421

Epoch 16/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0076 Acc: 0.8480

Epoch 17/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0071 Acc: 0.8480

Epoch 18/57
----------
train Loss: 0.0009 Acc: 0.9903
val Loss: 0.0072 Acc: 0.8480

Epoch 19/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0080 Acc: 0.8480

Epoch 20/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0078 Acc: 0.8421

Epoch 21/57
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0075 Acc: 0.8480

Epoch 22/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8480

Epoch 23/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8480

Epoch 24/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8480

Epoch 25/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0076 Acc: 0.8480

Epoch 26/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8480

Epoch 27/57
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0076 Acc: 0.8480

Epoch 28/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8538

Epoch 29/57
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0081 Acc: 0.8480

Epoch 30/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8480

Epoch 31/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8480

Epoch 32/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8480

Epoch 33/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8480

Epoch 34/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0081 Acc: 0.8480

Epoch 35/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8480

Epoch 36/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8480

Epoch 37/57
----------
train Loss: 0.0002 Acc: 0.9961
val Loss: 0.0071 Acc: 0.8480

Epoch 38/57
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0069 Acc: 0.8538

Epoch 39/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8538

Epoch 40/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8538

Epoch 41/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8480

Epoch 42/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8538

Epoch 43/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8480

Epoch 44/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8480

Epoch 45/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8480

Epoch 46/57
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8538

Epoch 47/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8538

Epoch 48/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8480

Epoch 49/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8480

Epoch 50/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8480

Epoch 51/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8480

Epoch 52/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8538

Epoch 53/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0072 Acc: 0.8480

Epoch 54/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8538

Epoch 55/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8538

Epoch 56/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8480

Epoch 57/57
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0074 Acc: 0.8538

Training complete in 3m 10s
Best val Acc: 0.877193

---Testing---
Test accuracy: 0.954876
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 84 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 81 %
Accuracy of Squatiniformes : 96 %
mean: 0.9397866042470422, std: 0.06456474328288075
--------------------

run info[val: 0.3, epoch: 97, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0213 Acc: 0.2141
val Loss: 0.0262 Acc: 0.3544

Epoch 1/96
----------
train Loss: 0.0161 Acc: 0.4761
val Loss: 0.0165 Acc: 0.5728

Epoch 2/96
----------
train Loss: 0.0106 Acc: 0.6590
val Loss: 0.0117 Acc: 0.6990

Epoch 3/96
----------
train Loss: 0.0071 Acc: 0.7983
val Loss: 0.0108 Acc: 0.8010

Epoch 4/96
----------
train Loss: 0.0053 Acc: 0.8441
val Loss: 0.0072 Acc: 0.8058

Epoch 5/96
----------
LR is set to 0.001
train Loss: 0.0047 Acc: 0.8794
val Loss: 0.0089 Acc: 0.8204

Epoch 6/96
----------
train Loss: 0.0046 Acc: 0.8753
val Loss: 0.0080 Acc: 0.8301

Epoch 7/96
----------
train Loss: 0.0042 Acc: 0.8919
val Loss: 0.0074 Acc: 0.8252

Epoch 8/96
----------
train Loss: 0.0042 Acc: 0.8960
val Loss: 0.0069 Acc: 0.8252

Epoch 9/96
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0078 Acc: 0.8301

Epoch 10/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0085 Acc: 0.8350

Epoch 11/96
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0069 Acc: 0.8350

Epoch 12/96
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0083 Acc: 0.8350

Epoch 13/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0068 Acc: 0.8350

Epoch 14/96
----------
train Loss: 0.0040 Acc: 0.9168
val Loss: 0.0061 Acc: 0.8350

Epoch 15/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0039 Acc: 0.9064
val Loss: 0.0079 Acc: 0.8350

Epoch 16/96
----------
train Loss: 0.0040 Acc: 0.8960
val Loss: 0.0100 Acc: 0.8350

Epoch 17/96
----------
train Loss: 0.0041 Acc: 0.9085
val Loss: 0.0103 Acc: 0.8350

Epoch 18/96
----------
train Loss: 0.0041 Acc: 0.9023
val Loss: 0.0081 Acc: 0.8350

Epoch 19/96
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0079 Acc: 0.8350

Epoch 20/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0040 Acc: 0.9210
val Loss: 0.0088 Acc: 0.8350

Epoch 21/96
----------
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0072 Acc: 0.8350

Epoch 22/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0066 Acc: 0.8350

Epoch 23/96
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0079 Acc: 0.8350

Epoch 24/96
----------
train Loss: 0.0041 Acc: 0.9023
val Loss: 0.0092 Acc: 0.8350

Epoch 25/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0041 Acc: 0.9002
val Loss: 0.0061 Acc: 0.8350

Epoch 26/96
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0089 Acc: 0.8301

Epoch 27/96
----------
train Loss: 0.0041 Acc: 0.9023
val Loss: 0.0101 Acc: 0.8350

Epoch 28/96
----------
train Loss: 0.0040 Acc: 0.9148
val Loss: 0.0094 Acc: 0.8350

Epoch 29/96
----------
train Loss: 0.0042 Acc: 0.8898
val Loss: 0.0091 Acc: 0.8350

Epoch 30/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0041 Acc: 0.9023
val Loss: 0.0090 Acc: 0.8350

Epoch 31/96
----------
train Loss: 0.0042 Acc: 0.8940
val Loss: 0.0067 Acc: 0.8350

Epoch 32/96
----------
train Loss: 0.0039 Acc: 0.9085
val Loss: 0.0102 Acc: 0.8350

Epoch 33/96
----------
train Loss: 0.0042 Acc: 0.8940
val Loss: 0.0073 Acc: 0.8350

Epoch 34/96
----------
train Loss: 0.0042 Acc: 0.8898
val Loss: 0.0067 Acc: 0.8350

Epoch 35/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0039 Acc: 0.9106
val Loss: 0.0073 Acc: 0.8350

Epoch 36/96
----------
train Loss: 0.0040 Acc: 0.9064
val Loss: 0.0063 Acc: 0.8350

Epoch 37/96
----------
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0069 Acc: 0.8350

Epoch 38/96
----------
train Loss: 0.0041 Acc: 0.9064
val Loss: 0.0099 Acc: 0.8350

Epoch 39/96
----------
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0068 Acc: 0.8350

Epoch 40/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0088 Acc: 0.8350

Epoch 41/96
----------
train Loss: 0.0039 Acc: 0.9106
val Loss: 0.0069 Acc: 0.8350

Epoch 42/96
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0072 Acc: 0.8350

Epoch 43/96
----------
train Loss: 0.0042 Acc: 0.8857
val Loss: 0.0061 Acc: 0.8350

Epoch 44/96
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0087 Acc: 0.8350

Epoch 45/96
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0042 Acc: 0.8981
val Loss: 0.0101 Acc: 0.8350

Epoch 46/96
----------
train Loss: 0.0040 Acc: 0.9064
val Loss: 0.0071 Acc: 0.8350

Epoch 47/96
----------
train Loss: 0.0041 Acc: 0.9002
val Loss: 0.0100 Acc: 0.8350

Epoch 48/96
----------
train Loss: 0.0042 Acc: 0.8836
val Loss: 0.0075 Acc: 0.8350

Epoch 49/96
----------
train Loss: 0.0041 Acc: 0.8877
val Loss: 0.0074 Acc: 0.8350

Epoch 50/96
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0069 Acc: 0.8350

Epoch 51/96
----------
train Loss: 0.0041 Acc: 0.9127
val Loss: 0.0097 Acc: 0.8350

Epoch 52/96
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0086 Acc: 0.8350

Epoch 53/96
----------
train Loss: 0.0042 Acc: 0.8940
val Loss: 0.0085 Acc: 0.8350

Epoch 54/96
----------
train Loss: 0.0041 Acc: 0.8940
val Loss: 0.0073 Acc: 0.8350

Epoch 55/96
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0038 Acc: 0.9106
val Loss: 0.0083 Acc: 0.8350

Epoch 56/96
----------
train Loss: 0.0041 Acc: 0.9085
val Loss: 0.0080 Acc: 0.8350

Epoch 57/96
----------
train Loss: 0.0039 Acc: 0.9148
val Loss: 0.0097 Acc: 0.8350

Epoch 58/96
----------
train Loss: 0.0043 Acc: 0.8877
val Loss: 0.0094 Acc: 0.8350

Epoch 59/96
----------
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0068 Acc: 0.8350

Epoch 60/96
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0039 Acc: 0.9064
val Loss: 0.0065 Acc: 0.8350

Epoch 61/96
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0084 Acc: 0.8350

Epoch 62/96
----------
train Loss: 0.0042 Acc: 0.8919
val Loss: 0.0095 Acc: 0.8350

Epoch 63/96
----------
train Loss: 0.0042 Acc: 0.9064
val Loss: 0.0084 Acc: 0.8350

Epoch 64/96
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0064 Acc: 0.8350

Epoch 65/96
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0070 Acc: 0.8350

Epoch 66/96
----------
train Loss: 0.0042 Acc: 0.9023
val Loss: 0.0067 Acc: 0.8350

Epoch 67/96
----------
train Loss: 0.0040 Acc: 0.9252
val Loss: 0.0107 Acc: 0.8350

Epoch 68/96
----------
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0072 Acc: 0.8350

Epoch 69/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0084 Acc: 0.8350

Epoch 70/96
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0039 Acc: 0.9168
val Loss: 0.0070 Acc: 0.8350

Epoch 71/96
----------
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0079 Acc: 0.8350

Epoch 72/96
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0068 Acc: 0.8350

Epoch 73/96
----------
train Loss: 0.0042 Acc: 0.9002
val Loss: 0.0077 Acc: 0.8350

Epoch 74/96
----------
train Loss: 0.0041 Acc: 0.9293
val Loss: 0.0068 Acc: 0.8350

Epoch 75/96
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0069 Acc: 0.8350

Epoch 76/96
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0103 Acc: 0.8350

Epoch 77/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0096 Acc: 0.8350

Epoch 78/96
----------
train Loss: 0.0041 Acc: 0.8960
val Loss: 0.0076 Acc: 0.8350

Epoch 79/96
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0063 Acc: 0.8350

Epoch 80/96
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0129 Acc: 0.8350

Epoch 81/96
----------
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0060 Acc: 0.8350

Epoch 82/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0128 Acc: 0.8350

Epoch 83/96
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0065 Acc: 0.8350

Epoch 84/96
----------
train Loss: 0.0042 Acc: 0.8877
val Loss: 0.0087 Acc: 0.8350

Epoch 85/96
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0071 Acc: 0.8350

Epoch 86/96
----------
train Loss: 0.0040 Acc: 0.8981
val Loss: 0.0083 Acc: 0.8350

Epoch 87/96
----------
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0065 Acc: 0.8350

Epoch 88/96
----------
train Loss: 0.0040 Acc: 0.9064
val Loss: 0.0098 Acc: 0.8350

Epoch 89/96
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0064 Acc: 0.8350

Epoch 90/96
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0041 Acc: 0.8898
val Loss: 0.0083 Acc: 0.8350

Epoch 91/96
----------
train Loss: 0.0041 Acc: 0.8960
val Loss: 0.0111 Acc: 0.8350

Epoch 92/96
----------
train Loss: 0.0042 Acc: 0.8940
val Loss: 0.0092 Acc: 0.8350

Epoch 93/96
----------
train Loss: 0.0040 Acc: 0.9168
val Loss: 0.0064 Acc: 0.8350

Epoch 94/96
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0092 Acc: 0.8350

Epoch 95/96
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0042 Acc: 0.9002
val Loss: 0.0098 Acc: 0.8350

Epoch 96/96
----------
train Loss: 0.0039 Acc: 0.9023
val Loss: 0.0066 Acc: 0.8350

Training complete in 4m 50s
Best val Acc: 0.834951

---Fine tuning.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0040 Acc: 0.8940
val Loss: 0.0059 Acc: 0.8592

Epoch 1/96
----------
train Loss: 0.0021 Acc: 0.9688
val Loss: 0.0045 Acc: 0.8689

Epoch 2/96
----------
train Loss: 0.0009 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8835

Epoch 3/96
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0099 Acc: 0.8932

Epoch 4/96
----------
train Loss: 0.0003 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 5/96
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8835

Epoch 6/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8835

Epoch 7/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 8/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8786

Epoch 9/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8786

Epoch 10/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8786

Epoch 11/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0031 Acc: 0.8786

Epoch 12/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8786

Epoch 13/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8786

Epoch 14/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 15/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 16/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8786

Epoch 17/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8835

Epoch 18/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8883

Epoch 19/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0091 Acc: 0.8786

Epoch 20/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8786

Epoch 21/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8835

Epoch 22/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0044 Acc: 0.8835

Epoch 23/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8835

Epoch 24/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8786

Epoch 25/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8883

Epoch 26/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 27/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0032 Acc: 0.8835

Epoch 28/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8835

Epoch 29/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8835

Epoch 30/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 31/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8786

Epoch 32/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0061 Acc: 0.8786

Epoch 33/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0043 Acc: 0.8786

Epoch 34/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 35/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0056 Acc: 0.8786

Epoch 36/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8786

Epoch 37/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0067 Acc: 0.8786

Epoch 38/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8786

Epoch 39/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8786

Epoch 40/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8835

Epoch 41/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0064 Acc: 0.8786

Epoch 42/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0046 Acc: 0.8786

Epoch 43/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8835

Epoch 44/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 45/96
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8883

Epoch 46/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8835

Epoch 47/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0032 Acc: 0.8786

Epoch 48/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8786

Epoch 49/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 50/96
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8786

Epoch 51/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0063 Acc: 0.8786

Epoch 52/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8786

Epoch 53/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0032 Acc: 0.8786

Epoch 54/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8786

Epoch 55/96
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0075 Acc: 0.8835

Epoch 56/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8835

Epoch 57/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 58/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0038 Acc: 0.8835

Epoch 59/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 60/96
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8786

Epoch 61/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0053 Acc: 0.8835

Epoch 62/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0118 Acc: 0.8883

Epoch 63/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 64/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 65/96
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0041 Acc: 0.8835

Epoch 66/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8835

Epoch 67/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0052 Acc: 0.8883

Epoch 68/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8786

Epoch 69/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0079 Acc: 0.8786

Epoch 70/96
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0031 Acc: 0.8786

Epoch 71/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8786

Epoch 72/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0044 Acc: 0.8786

Epoch 73/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0032 Acc: 0.8786

Epoch 74/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8786

Epoch 75/96
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0043 Acc: 0.8786

Epoch 76/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8786

Epoch 77/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8786

Epoch 78/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8835

Epoch 79/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0033 Acc: 0.8883

Epoch 80/96
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8883

Epoch 81/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8835

Epoch 82/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8835

Epoch 83/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8786

Epoch 84/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8786

Epoch 85/96
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0042 Acc: 0.8835

Epoch 86/96
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 87/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8835

Epoch 88/96
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0034 Acc: 0.8835

Epoch 89/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8786

Epoch 90/96
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8786

Epoch 91/96
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0058 Acc: 0.8786

Epoch 92/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8786

Epoch 93/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 94/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8786

Epoch 95/96
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0056 Acc: 0.8786

Epoch 96/96
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8835

Training complete in 5m 9s
Best val Acc: 0.893204

---Testing---
Test accuracy: 0.966521
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.9567572588426527, std: 0.048009285644812945

Model saved in "./weights/shark_[0.98]_mean[0.97]_std[0.03].save".
--------------------

run info[val: 0.1, epoch: 94, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0224 Acc: 0.2375
val Loss: 0.0242 Acc: 0.4412

Epoch 1/93
----------
train Loss: 0.0141 Acc: 0.5541
val Loss: 0.0166 Acc: 0.6176

Epoch 2/93
----------
train Loss: 0.0088 Acc: 0.7803
val Loss: 0.0119 Acc: 0.6618

Epoch 3/93
----------
train Loss: 0.0066 Acc: 0.8207
val Loss: 0.0117 Acc: 0.6765

Epoch 4/93
----------
train Loss: 0.0056 Acc: 0.8320
val Loss: 0.0113 Acc: 0.7353

Epoch 5/93
----------
LR is set to 0.001
train Loss: 0.0058 Acc: 0.8449
val Loss: 0.0103 Acc: 0.7647

Epoch 6/93
----------
train Loss: 0.0047 Acc: 0.8708
val Loss: 0.0099 Acc: 0.7647

Epoch 7/93
----------
train Loss: 0.0043 Acc: 0.8788
val Loss: 0.0097 Acc: 0.7353

Epoch 8/93
----------
train Loss: 0.0045 Acc: 0.8869
val Loss: 0.0095 Acc: 0.7794

Epoch 9/93
----------
train Loss: 0.0043 Acc: 0.8998
val Loss: 0.0095 Acc: 0.7647

Epoch 10/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0039 Acc: 0.8950
val Loss: 0.0096 Acc: 0.7647

Epoch 11/93
----------
train Loss: 0.0046 Acc: 0.8966
val Loss: 0.0096 Acc: 0.7647

Epoch 12/93
----------
train Loss: 0.0038 Acc: 0.8966
val Loss: 0.0096 Acc: 0.7647

Epoch 13/93
----------
train Loss: 0.0042 Acc: 0.8837
val Loss: 0.0096 Acc: 0.7647

Epoch 14/93
----------
train Loss: 0.0044 Acc: 0.9015
val Loss: 0.0095 Acc: 0.7647

Epoch 15/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0041 Acc: 0.9015
val Loss: 0.0095 Acc: 0.7647

Epoch 16/93
----------
train Loss: 0.0044 Acc: 0.8982
val Loss: 0.0096 Acc: 0.7647

Epoch 17/93
----------
train Loss: 0.0042 Acc: 0.9015
val Loss: 0.0096 Acc: 0.7647

Epoch 18/93
----------
train Loss: 0.0041 Acc: 0.9095
val Loss: 0.0095 Acc: 0.7647

Epoch 19/93
----------
train Loss: 0.0039 Acc: 0.8998
val Loss: 0.0095 Acc: 0.7647

Epoch 20/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0041 Acc: 0.8950
val Loss: 0.0096 Acc: 0.7647

Epoch 21/93
----------
train Loss: 0.0043 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 22/93
----------
train Loss: 0.0046 Acc: 0.8918
val Loss: 0.0095 Acc: 0.7647

Epoch 23/93
----------
train Loss: 0.0041 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 24/93
----------
train Loss: 0.0042 Acc: 0.9015
val Loss: 0.0096 Acc: 0.7647

Epoch 25/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0041 Acc: 0.9095
val Loss: 0.0097 Acc: 0.7647

Epoch 26/93
----------
train Loss: 0.0040 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 27/93
----------
train Loss: 0.0042 Acc: 0.8901
val Loss: 0.0097 Acc: 0.7647

Epoch 28/93
----------
train Loss: 0.0041 Acc: 0.8934
val Loss: 0.0097 Acc: 0.7647

Epoch 29/93
----------
train Loss: 0.0041 Acc: 0.8885
val Loss: 0.0096 Acc: 0.7647

Epoch 30/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0043 Acc: 0.8853
val Loss: 0.0097 Acc: 0.7647

Epoch 31/93
----------
train Loss: 0.0046 Acc: 0.8837
val Loss: 0.0096 Acc: 0.7647

Epoch 32/93
----------
train Loss: 0.0040 Acc: 0.8934
val Loss: 0.0097 Acc: 0.7647

Epoch 33/93
----------
train Loss: 0.0043 Acc: 0.8837
val Loss: 0.0096 Acc: 0.7647

Epoch 34/93
----------
train Loss: 0.0044 Acc: 0.8901
val Loss: 0.0095 Acc: 0.7647

Epoch 35/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0044 Acc: 0.8982
val Loss: 0.0093 Acc: 0.7647

Epoch 36/93
----------
train Loss: 0.0046 Acc: 0.8934
val Loss: 0.0095 Acc: 0.7647

Epoch 37/93
----------
train Loss: 0.0041 Acc: 0.9063
val Loss: 0.0096 Acc: 0.7647

Epoch 38/93
----------
train Loss: 0.0041 Acc: 0.8934
val Loss: 0.0096 Acc: 0.7647

Epoch 39/93
----------
train Loss: 0.0044 Acc: 0.8998
val Loss: 0.0096 Acc: 0.7647

Epoch 40/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0041 Acc: 0.8918
val Loss: 0.0097 Acc: 0.7647

Epoch 41/93
----------
train Loss: 0.0042 Acc: 0.8950
val Loss: 0.0095 Acc: 0.7647

Epoch 42/93
----------
train Loss: 0.0042 Acc: 0.8805
val Loss: 0.0095 Acc: 0.7647

Epoch 43/93
----------
train Loss: 0.0040 Acc: 0.8934
val Loss: 0.0096 Acc: 0.7647

Epoch 44/93
----------
train Loss: 0.0046 Acc: 0.9015
val Loss: 0.0096 Acc: 0.7500

Epoch 45/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0044 Acc: 0.8901
val Loss: 0.0095 Acc: 0.7647

Epoch 46/93
----------
train Loss: 0.0046 Acc: 0.8821
val Loss: 0.0094 Acc: 0.7647

Epoch 47/93
----------
train Loss: 0.0043 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 48/93
----------
train Loss: 0.0046 Acc: 0.8885
val Loss: 0.0096 Acc: 0.7647

Epoch 49/93
----------
train Loss: 0.0045 Acc: 0.8756
val Loss: 0.0096 Acc: 0.7647

Epoch 50/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0039 Acc: 0.9015
val Loss: 0.0096 Acc: 0.7647

Epoch 51/93
----------
train Loss: 0.0043 Acc: 0.8869
val Loss: 0.0097 Acc: 0.7647

Epoch 52/93
----------
train Loss: 0.0041 Acc: 0.9031
val Loss: 0.0097 Acc: 0.7647

Epoch 53/93
----------
train Loss: 0.0041 Acc: 0.8869
val Loss: 0.0095 Acc: 0.7647

Epoch 54/93
----------
train Loss: 0.0041 Acc: 0.8918
val Loss: 0.0096 Acc: 0.7647

Epoch 55/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0044 Acc: 0.8869
val Loss: 0.0097 Acc: 0.7647

Epoch 56/93
----------
train Loss: 0.0041 Acc: 0.8998
val Loss: 0.0098 Acc: 0.7647

Epoch 57/93
----------
train Loss: 0.0040 Acc: 0.8982
val Loss: 0.0096 Acc: 0.7647

Epoch 58/93
----------
train Loss: 0.0044 Acc: 0.9015
val Loss: 0.0097 Acc: 0.7647

Epoch 59/93
----------
train Loss: 0.0039 Acc: 0.9031
val Loss: 0.0097 Acc: 0.7647

Epoch 60/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0044 Acc: 0.8837
val Loss: 0.0096 Acc: 0.7647

Epoch 61/93
----------
train Loss: 0.0045 Acc: 0.8934
val Loss: 0.0095 Acc: 0.7647

Epoch 62/93
----------
train Loss: 0.0047 Acc: 0.8756
val Loss: 0.0095 Acc: 0.7647

Epoch 63/93
----------
train Loss: 0.0045 Acc: 0.8869
val Loss: 0.0096 Acc: 0.7647

Epoch 64/93
----------
train Loss: 0.0043 Acc: 0.8918
val Loss: 0.0096 Acc: 0.7647

Epoch 65/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0042 Acc: 0.8982
val Loss: 0.0096 Acc: 0.7647

Epoch 66/93
----------
train Loss: 0.0039 Acc: 0.8982
val Loss: 0.0096 Acc: 0.7647

Epoch 67/93
----------
train Loss: 0.0044 Acc: 0.8788
val Loss: 0.0097 Acc: 0.7647

Epoch 68/93
----------
train Loss: 0.0043 Acc: 0.9015
val Loss: 0.0096 Acc: 0.7647

Epoch 69/93
----------
train Loss: 0.0042 Acc: 0.9031
val Loss: 0.0095 Acc: 0.7647

Epoch 70/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0042 Acc: 0.9111
val Loss: 0.0097 Acc: 0.7647

Epoch 71/93
----------
train Loss: 0.0045 Acc: 0.9095
val Loss: 0.0096 Acc: 0.7647

Epoch 72/93
----------
train Loss: 0.0041 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 73/93
----------
train Loss: 0.0040 Acc: 0.8918
val Loss: 0.0096 Acc: 0.7647

Epoch 74/93
----------
train Loss: 0.0041 Acc: 0.9111
val Loss: 0.0095 Acc: 0.7647

Epoch 75/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0039 Acc: 0.9047
val Loss: 0.0095 Acc: 0.7647

Epoch 76/93
----------
train Loss: 0.0042 Acc: 0.9079
val Loss: 0.0097 Acc: 0.7647

Epoch 77/93
----------
train Loss: 0.0038 Acc: 0.8901
val Loss: 0.0095 Acc: 0.7647

Epoch 78/93
----------
train Loss: 0.0039 Acc: 0.8885
val Loss: 0.0095 Acc: 0.7647

Epoch 79/93
----------
train Loss: 0.0042 Acc: 0.8837
val Loss: 0.0095 Acc: 0.7647

Epoch 80/93
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0038 Acc: 0.8998
val Loss: 0.0096 Acc: 0.7647

Epoch 81/93
----------
train Loss: 0.0043 Acc: 0.8982
val Loss: 0.0096 Acc: 0.7647

Epoch 82/93
----------
train Loss: 0.0043 Acc: 0.8885
val Loss: 0.0095 Acc: 0.7647

Epoch 83/93
----------
train Loss: 0.0046 Acc: 0.8821
val Loss: 0.0094 Acc: 0.7647

Epoch 84/93
----------
train Loss: 0.0043 Acc: 0.8982
val Loss: 0.0095 Acc: 0.7647

Epoch 85/93
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0043 Acc: 0.9095
val Loss: 0.0097 Acc: 0.7647

Epoch 86/93
----------
train Loss: 0.0039 Acc: 0.8805
val Loss: 0.0097 Acc: 0.7647

Epoch 87/93
----------
train Loss: 0.0038 Acc: 0.9015
val Loss: 0.0097 Acc: 0.7647

Epoch 88/93
----------
train Loss: 0.0045 Acc: 0.8982
val Loss: 0.0097 Acc: 0.7647

Epoch 89/93
----------
train Loss: 0.0044 Acc: 0.9111
val Loss: 0.0096 Acc: 0.7647

Epoch 90/93
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0043 Acc: 0.8950
val Loss: 0.0096 Acc: 0.7647

Epoch 91/93
----------
train Loss: 0.0042 Acc: 0.9095
val Loss: 0.0095 Acc: 0.7647

Epoch 92/93
----------
train Loss: 0.0039 Acc: 0.8966
val Loss: 0.0094 Acc: 0.7647

Epoch 93/93
----------
train Loss: 0.0044 Acc: 0.8934
val Loss: 0.0095 Acc: 0.7647

Training complete in 4m 37s
Best val Acc: 0.779412

---Fine tuning.---
Epoch 0/93
----------
LR is set to 0.01
train Loss: 0.0042 Acc: 0.8950
val Loss: 0.0098 Acc: 0.8088

Epoch 1/93
----------
train Loss: 0.0022 Acc: 0.9483
val Loss: 0.0096 Acc: 0.8088

Epoch 2/93
----------
train Loss: 0.0013 Acc: 0.9709
val Loss: 0.0068 Acc: 0.8382

Epoch 3/93
----------
train Loss: 0.0008 Acc: 0.9903
val Loss: 0.0085 Acc: 0.8382

Epoch 4/93
----------
train Loss: 0.0005 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 5/93
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9952
val Loss: 0.0079 Acc: 0.8235

Epoch 6/93
----------
train Loss: 0.0007 Acc: 0.9919
val Loss: 0.0077 Acc: 0.8235

Epoch 7/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8235

Epoch 8/93
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0071 Acc: 0.8235

Epoch 9/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0073 Acc: 0.8235

Epoch 10/93
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0073 Acc: 0.8235

Epoch 11/93
----------
train Loss: 0.0003 Acc: 0.9935
val Loss: 0.0073 Acc: 0.8235

Epoch 12/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 13/93
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0074 Acc: 0.8235

Epoch 14/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 15/93
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0075 Acc: 0.8235

Epoch 16/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8235

Epoch 17/93
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0076 Acc: 0.8235

Epoch 18/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 19/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8235

Epoch 20/93
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0075 Acc: 0.8235

Epoch 21/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0073 Acc: 0.8235

Epoch 22/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8235

Epoch 23/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8235

Epoch 24/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 25/93
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0074 Acc: 0.8235

Epoch 26/93
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0075 Acc: 0.8235

Epoch 27/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0074 Acc: 0.8235

Epoch 28/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 29/93
----------
train Loss: 0.0004 Acc: 0.9935
val Loss: 0.0077 Acc: 0.8235

Epoch 30/93
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 31/93
----------
train Loss: 0.0005 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 32/93
----------
train Loss: 0.0005 Acc: 0.9968
val Loss: 0.0075 Acc: 0.8382

Epoch 33/93
----------
train Loss: 0.0002 Acc: 0.9952
val Loss: 0.0073 Acc: 0.8235

Epoch 34/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 35/93
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9935
val Loss: 0.0076 Acc: 0.8235

Epoch 36/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 37/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8235

Epoch 38/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8235

Epoch 39/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8235

Epoch 40/93
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 41/93
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0076 Acc: 0.8235

Epoch 42/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 43/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0074 Acc: 0.8235

Epoch 44/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8235

Epoch 45/93
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0073 Acc: 0.8235

Epoch 46/93
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0073 Acc: 0.8235

Epoch 47/93
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0074 Acc: 0.8235

Epoch 48/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8235

Epoch 49/93
----------
train Loss: 0.0005 Acc: 0.9935
val Loss: 0.0076 Acc: 0.8235

Epoch 50/93
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8235

Epoch 51/93
----------
train Loss: 0.0004 Acc: 0.9984
val Loss: 0.0073 Acc: 0.8235

Epoch 52/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8382

Epoch 53/93
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 54/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8235

Epoch 55/93
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 56/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 57/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 58/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 59/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 60/93
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 61/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 62/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 63/93
----------
train Loss: 0.0004 Acc: 0.9952
val Loss: 0.0075 Acc: 0.8235

Epoch 64/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 65/93
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0074 Acc: 0.8235

Epoch 66/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0073 Acc: 0.8235

Epoch 67/93
----------
train Loss: 0.0003 Acc: 0.9935
val Loss: 0.0074 Acc: 0.8235

Epoch 68/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 69/93
----------
train Loss: 0.0008 Acc: 0.9935
val Loss: 0.0076 Acc: 0.8088

Epoch 70/93
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8235

Epoch 71/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 72/93
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0075 Acc: 0.8382

Epoch 73/93
----------
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8088

Epoch 74/93
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 75/93
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8235

Epoch 76/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8235

Epoch 77/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 78/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8382

Epoch 79/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 80/93
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0003 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8235

Epoch 81/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8382

Epoch 82/93
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 83/93
----------
train Loss: 0.0004 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 84/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8235

Epoch 85/93
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 86/93
----------
train Loss: 0.0004 Acc: 0.9952
val Loss: 0.0076 Acc: 0.8235

Epoch 87/93
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8235

Epoch 88/93
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8235

Epoch 89/93
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8235

Epoch 90/93
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0004 Acc: 0.9952
val Loss: 0.0075 Acc: 0.8235

Epoch 91/93
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 92/93
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8235

Epoch 93/93
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8382

Training complete in 5m 0s
Best val Acc: 0.838235

---Testing---
Test accuracy: 0.960699
--------------------
Accuracy of Carcharhiniformes : 94 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 90 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 96 %
mean: 0.9522596511871504, std: 0.03201917359160339
--------------------

run info[val: 0.15, epoch: 95, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0203 Acc: 0.2397
val Loss: 0.0356 Acc: 0.3398

Epoch 1/94
----------
train Loss: 0.0138 Acc: 0.5839
val Loss: 0.0267 Acc: 0.5728

Epoch 2/94
----------
train Loss: 0.0080 Acc: 0.7432
val Loss: 0.0135 Acc: 0.7670

Epoch 3/94
----------
train Loss: 0.0058 Acc: 0.8510
val Loss: 0.0176 Acc: 0.7961

Epoch 4/94
----------
train Loss: 0.0045 Acc: 0.8682
val Loss: 0.0069 Acc: 0.8155

Epoch 5/94
----------
train Loss: 0.0038 Acc: 0.8990
val Loss: 0.0062 Acc: 0.7961

Epoch 6/94
----------
train Loss: 0.0035 Acc: 0.9041
val Loss: 0.0067 Acc: 0.8155

Epoch 7/94
----------
train Loss: 0.0029 Acc: 0.9212
val Loss: 0.0064 Acc: 0.7767

Epoch 8/94
----------
train Loss: 0.0025 Acc: 0.9298
val Loss: 0.0070 Acc: 0.8058

Epoch 9/94
----------
train Loss: 0.0024 Acc: 0.9521
val Loss: 0.0068 Acc: 0.7864

Epoch 10/94
----------
train Loss: 0.0022 Acc: 0.9538
val Loss: 0.0088 Acc: 0.8252

Epoch 11/94
----------
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0175 Acc: 0.7961

Epoch 12/94
----------
LR is set to 0.001
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0089 Acc: 0.7961

Epoch 13/94
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0095 Acc: 0.8058

Epoch 14/94
----------
train Loss: 0.0019 Acc: 0.9760
val Loss: 0.0059 Acc: 0.8155

Epoch 15/94
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0122 Acc: 0.8155

Epoch 16/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0059 Acc: 0.8155

Epoch 17/94
----------
train Loss: 0.0019 Acc: 0.9777
val Loss: 0.0191 Acc: 0.8058

Epoch 18/94
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0059 Acc: 0.8155

Epoch 19/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0060 Acc: 0.8058

Epoch 20/94
----------
train Loss: 0.0017 Acc: 0.9812
val Loss: 0.0063 Acc: 0.8058

Epoch 21/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0159 Acc: 0.8155

Epoch 22/94
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0093 Acc: 0.8155

Epoch 23/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0086 Acc: 0.8155

Epoch 24/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9812
val Loss: 0.0117 Acc: 0.8155

Epoch 25/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0059 Acc: 0.8058

Epoch 26/94
----------
train Loss: 0.0017 Acc: 0.9709
val Loss: 0.0068 Acc: 0.8155

Epoch 27/94
----------
train Loss: 0.0018 Acc: 0.9726
val Loss: 0.0084 Acc: 0.8155

Epoch 28/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0124 Acc: 0.8155

Epoch 29/94
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0064 Acc: 0.8155

Epoch 30/94
----------
train Loss: 0.0017 Acc: 0.9795
val Loss: 0.0074 Acc: 0.8155

Epoch 31/94
----------
train Loss: 0.0017 Acc: 0.9795
val Loss: 0.0084 Acc: 0.8155

Epoch 32/94
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0138 Acc: 0.8155

Epoch 33/94
----------
train Loss: 0.0019 Acc: 0.9623
val Loss: 0.0061 Acc: 0.8155

Epoch 34/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0103 Acc: 0.8155

Epoch 35/94
----------
train Loss: 0.0017 Acc: 0.9812
val Loss: 0.0059 Acc: 0.8155

Epoch 36/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0125 Acc: 0.8058

Epoch 37/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0061 Acc: 0.8058

Epoch 38/94
----------
train Loss: 0.0017 Acc: 0.9777
val Loss: 0.0341 Acc: 0.8155

Epoch 39/94
----------
train Loss: 0.0018 Acc: 0.9726
val Loss: 0.0066 Acc: 0.8155

Epoch 40/94
----------
train Loss: 0.0019 Acc: 0.9726
val Loss: 0.0056 Acc: 0.8155

Epoch 41/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0089 Acc: 0.8155

Epoch 42/94
----------
train Loss: 0.0018 Acc: 0.9640
val Loss: 0.0077 Acc: 0.8155

Epoch 43/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0121 Acc: 0.8155

Epoch 44/94
----------
train Loss: 0.0018 Acc: 0.9726
val Loss: 0.0097 Acc: 0.8155

Epoch 45/94
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0093 Acc: 0.8155

Epoch 46/94
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0072 Acc: 0.8155

Epoch 47/94
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0165 Acc: 0.8155

Epoch 48/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0096 Acc: 0.8155

Epoch 49/94
----------
train Loss: 0.0018 Acc: 0.9692
val Loss: 0.0064 Acc: 0.8155

Epoch 50/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0098 Acc: 0.8155

Epoch 51/94
----------
train Loss: 0.0016 Acc: 0.9829
val Loss: 0.0068 Acc: 0.8155

Epoch 52/94
----------
train Loss: 0.0017 Acc: 0.9846
val Loss: 0.0066 Acc: 0.8155

Epoch 53/94
----------
train Loss: 0.0017 Acc: 0.9777
val Loss: 0.0132 Acc: 0.8155

Epoch 54/94
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0066 Acc: 0.8155

Epoch 55/94
----------
train Loss: 0.0017 Acc: 0.9795
val Loss: 0.0152 Acc: 0.8155

Epoch 56/94
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0128 Acc: 0.8155

Epoch 57/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0119 Acc: 0.8155

Epoch 58/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0218 Acc: 0.8155

Epoch 59/94
----------
train Loss: 0.0018 Acc: 0.9795
val Loss: 0.0158 Acc: 0.8155

Epoch 60/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0018 Acc: 0.9777
val Loss: 0.0061 Acc: 0.8155

Epoch 61/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0056 Acc: 0.8155

Epoch 62/94
----------
train Loss: 0.0017 Acc: 0.9777
val Loss: 0.0213 Acc: 0.8155

Epoch 63/94
----------
train Loss: 0.0017 Acc: 0.9812
val Loss: 0.0058 Acc: 0.8155

Epoch 64/94
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0128 Acc: 0.8155

Epoch 65/94
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0075 Acc: 0.8155

Epoch 66/94
----------
train Loss: 0.0017 Acc: 0.9760
val Loss: 0.0059 Acc: 0.8155

Epoch 67/94
----------
train Loss: 0.0019 Acc: 0.9726
val Loss: 0.0088 Acc: 0.8155

Epoch 68/94
----------
train Loss: 0.0018 Acc: 0.9709
val Loss: 0.0071 Acc: 0.8155

Epoch 69/94
----------
train Loss: 0.0017 Acc: 0.9692
val Loss: 0.0061 Acc: 0.8155

Epoch 70/94
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0242 Acc: 0.8155

Epoch 71/94
----------
train Loss: 0.0018 Acc: 0.9675
val Loss: 0.0165 Acc: 0.8155

Epoch 72/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0055 Acc: 0.8155

Epoch 73/94
----------
train Loss: 0.0017 Acc: 0.9760
val Loss: 0.0059 Acc: 0.8155

Epoch 74/94
----------
train Loss: 0.0017 Acc: 0.9760
val Loss: 0.0056 Acc: 0.8155

Epoch 75/94
----------
train Loss: 0.0018 Acc: 0.9675
val Loss: 0.0103 Acc: 0.8155

Epoch 76/94
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0156 Acc: 0.8155

Epoch 77/94
----------
train Loss: 0.0017 Acc: 0.9692
val Loss: 0.0072 Acc: 0.8155

Epoch 78/94
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0118 Acc: 0.8155

Epoch 79/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0096 Acc: 0.8155

Epoch 80/94
----------
train Loss: 0.0017 Acc: 0.9829
val Loss: 0.0117 Acc: 0.8155

Epoch 81/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0061 Acc: 0.8155

Epoch 82/94
----------
train Loss: 0.0018 Acc: 0.9777
val Loss: 0.0101 Acc: 0.8155

Epoch 83/94
----------
train Loss: 0.0017 Acc: 0.9760
val Loss: 0.0059 Acc: 0.8155

Epoch 84/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0117 Acc: 0.8155

Epoch 85/94
----------
train Loss: 0.0018 Acc: 0.9658
val Loss: 0.0066 Acc: 0.8155

Epoch 86/94
----------
train Loss: 0.0018 Acc: 0.9777
val Loss: 0.0060 Acc: 0.8155

Epoch 87/94
----------
train Loss: 0.0018 Acc: 0.9743
val Loss: 0.0135 Acc: 0.8155

Epoch 88/94
----------
train Loss: 0.0018 Acc: 0.9760
val Loss: 0.0152 Acc: 0.8155

Epoch 89/94
----------
train Loss: 0.0017 Acc: 0.9829
val Loss: 0.0178 Acc: 0.8155

Epoch 90/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0185 Acc: 0.8155

Epoch 91/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0086 Acc: 0.8155

Epoch 92/94
----------
train Loss: 0.0018 Acc: 0.9692
val Loss: 0.0124 Acc: 0.8155

Epoch 93/94
----------
train Loss: 0.0018 Acc: 0.9726
val Loss: 0.0074 Acc: 0.8155

Epoch 94/94
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0056 Acc: 0.8155

Training complete in 4m 54s
Best val Acc: 0.825243

---Fine tuning.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0076 Acc: 0.8350

Epoch 1/94
----------
train Loss: 0.0009 Acc: 0.9914
val Loss: 0.0056 Acc: 0.8350

Epoch 2/94
----------
train Loss: 0.0004 Acc: 0.9966
val Loss: 0.0123 Acc: 0.8447

Epoch 3/94
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8350

Epoch 4/94
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8641

Epoch 5/94
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8447

Epoch 6/94
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8447

Epoch 7/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0149 Acc: 0.8447

Epoch 8/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8447

Epoch 9/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 10/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 11/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 12/94
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8544

Epoch 13/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 14/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8544

Epoch 15/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8544

Epoch 16/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 17/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 18/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 19/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 20/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8447

Epoch 21/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8447

Epoch 22/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8447

Epoch 23/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8447

Epoch 24/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0127 Acc: 0.8447

Epoch 25/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 26/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 27/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8447

Epoch 28/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 29/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 30/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8544

Epoch 31/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8544

Epoch 32/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 33/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8544

Epoch 34/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0091 Acc: 0.8544

Epoch 35/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 36/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8544

Epoch 37/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8447

Epoch 38/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0199 Acc: 0.8447

Epoch 39/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 40/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 41/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8447

Epoch 42/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0149 Acc: 0.8447

Epoch 43/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8447

Epoch 44/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0179 Acc: 0.8447

Epoch 45/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0181 Acc: 0.8447

Epoch 46/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 47/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 48/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8447

Epoch 49/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8447

Epoch 50/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8544

Epoch 51/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 52/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8544

Epoch 53/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8544

Epoch 54/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 55/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8447

Epoch 56/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8544

Epoch 57/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 58/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 59/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0150 Acc: 0.8544

Epoch 60/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8544

Epoch 61/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 62/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0138 Acc: 0.8447

Epoch 63/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 64/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 65/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8447

Epoch 66/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8447

Epoch 67/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 68/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8447

Epoch 69/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0244 Acc: 0.8447

Epoch 70/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 71/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 72/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 73/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8544

Epoch 74/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8544

Epoch 75/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8447

Epoch 76/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 77/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 78/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0314 Acc: 0.8447

Epoch 79/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8447

Epoch 80/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 81/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 82/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8544

Epoch 83/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8447

Epoch 84/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0207 Acc: 0.8544

Epoch 85/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 86/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8544

Epoch 87/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 88/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 89/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 90/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0141 Acc: 0.8447

Epoch 91/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 92/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0124 Acc: 0.8447

Epoch 93/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 94/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Training complete in 5m 28s
Best val Acc: 0.864078

---Testing---
Test accuracy: 0.979622
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9713274663932558, std: 0.029455169683012646
--------------------

run info[val: 0.2, epoch: 99, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0227 Acc: 0.2091
val Loss: 0.0274 Acc: 0.2044

Epoch 1/98
----------
train Loss: 0.0143 Acc: 0.5145
val Loss: 0.0160 Acc: 0.6131

Epoch 2/98
----------
train Loss: 0.0084 Acc: 0.7745
val Loss: 0.0108 Acc: 0.7445

Epoch 3/98
----------
train Loss: 0.0063 Acc: 0.8200
val Loss: 0.0093 Acc: 0.7518

Epoch 4/98
----------
train Loss: 0.0049 Acc: 0.8709
val Loss: 0.0087 Acc: 0.8248

Epoch 5/98
----------
train Loss: 0.0041 Acc: 0.8927
val Loss: 0.0086 Acc: 0.8613

Epoch 6/98
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.9182
val Loss: 0.0078 Acc: 0.8467

Epoch 7/98
----------
train Loss: 0.0033 Acc: 0.9109
val Loss: 0.0085 Acc: 0.8467

Epoch 8/98
----------
train Loss: 0.0037 Acc: 0.9127
val Loss: 0.0075 Acc: 0.8321

Epoch 9/98
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0070 Acc: 0.8321

Epoch 10/98
----------
train Loss: 0.0031 Acc: 0.9327
val Loss: 0.0078 Acc: 0.8394

Epoch 11/98
----------
train Loss: 0.0033 Acc: 0.9291
val Loss: 0.0072 Acc: 0.8394

Epoch 12/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9436
val Loss: 0.0078 Acc: 0.8394

Epoch 13/98
----------
train Loss: 0.0033 Acc: 0.9364
val Loss: 0.0076 Acc: 0.8394

Epoch 14/98
----------
train Loss: 0.0031 Acc: 0.9309
val Loss: 0.0069 Acc: 0.8467

Epoch 15/98
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0072 Acc: 0.8467

Epoch 16/98
----------
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0069 Acc: 0.8467

Epoch 17/98
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0082 Acc: 0.8467

Epoch 18/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0034 Acc: 0.9345
val Loss: 0.0074 Acc: 0.8394

Epoch 19/98
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0075 Acc: 0.8394

Epoch 20/98
----------
train Loss: 0.0033 Acc: 0.9327
val Loss: 0.0082 Acc: 0.8394

Epoch 21/98
----------
train Loss: 0.0033 Acc: 0.9218
val Loss: 0.0073 Acc: 0.8394

Epoch 22/98
----------
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0071 Acc: 0.8394

Epoch 23/98
----------
train Loss: 0.0033 Acc: 0.9291
val Loss: 0.0078 Acc: 0.8467

Epoch 24/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0093 Acc: 0.8467

Epoch 25/98
----------
train Loss: 0.0033 Acc: 0.9200
val Loss: 0.0085 Acc: 0.8467

Epoch 26/98
----------
train Loss: 0.0034 Acc: 0.9255
val Loss: 0.0070 Acc: 0.8467

Epoch 27/98
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0072 Acc: 0.8467

Epoch 28/98
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0070 Acc: 0.8394

Epoch 29/98
----------
train Loss: 0.0032 Acc: 0.9364
val Loss: 0.0077 Acc: 0.8467

Epoch 30/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0072 Acc: 0.8394

Epoch 31/98
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0084 Acc: 0.8467

Epoch 32/98
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0068 Acc: 0.8394

Epoch 33/98
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0090 Acc: 0.8394

Epoch 34/98
----------
train Loss: 0.0031 Acc: 0.9382
val Loss: 0.0086 Acc: 0.8394

Epoch 35/98
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0077 Acc: 0.8394

Epoch 36/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0032 Acc: 0.9309
val Loss: 0.0074 Acc: 0.8467

Epoch 37/98
----------
train Loss: 0.0032 Acc: 0.9382
val Loss: 0.0082 Acc: 0.8467

Epoch 38/98
----------
train Loss: 0.0031 Acc: 0.9400
val Loss: 0.0070 Acc: 0.8467

Epoch 39/98
----------
train Loss: 0.0032 Acc: 0.9236
val Loss: 0.0074 Acc: 0.8394

Epoch 40/98
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0076 Acc: 0.8467

Epoch 41/98
----------
train Loss: 0.0034 Acc: 0.9236
val Loss: 0.0084 Acc: 0.8467

Epoch 42/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0074 Acc: 0.8467

Epoch 43/98
----------
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0080 Acc: 0.8394

Epoch 44/98
----------
train Loss: 0.0033 Acc: 0.9327
val Loss: 0.0078 Acc: 0.8394

Epoch 45/98
----------
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0075 Acc: 0.8394

Epoch 46/98
----------
train Loss: 0.0034 Acc: 0.9091
val Loss: 0.0085 Acc: 0.8394

Epoch 47/98
----------
train Loss: 0.0033 Acc: 0.9200
val Loss: 0.0073 Acc: 0.8394

Epoch 48/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0033 Acc: 0.9236
val Loss: 0.0079 Acc: 0.8394

Epoch 49/98
----------
train Loss: 0.0032 Acc: 0.9436
val Loss: 0.0079 Acc: 0.8394

Epoch 50/98
----------
train Loss: 0.0032 Acc: 0.9345
val Loss: 0.0086 Acc: 0.8394

Epoch 51/98
----------
train Loss: 0.0032 Acc: 0.9218
val Loss: 0.0080 Acc: 0.8394

Epoch 52/98
----------
train Loss: 0.0034 Acc: 0.9327
val Loss: 0.0084 Acc: 0.8394

Epoch 53/98
----------
train Loss: 0.0032 Acc: 0.9236
val Loss: 0.0080 Acc: 0.8467

Epoch 54/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0033 Acc: 0.9236
val Loss: 0.0070 Acc: 0.8394

Epoch 55/98
----------
train Loss: 0.0032 Acc: 0.9182
val Loss: 0.0083 Acc: 0.8394

Epoch 56/98
----------
train Loss: 0.0033 Acc: 0.9309
val Loss: 0.0069 Acc: 0.8467

Epoch 57/98
----------
train Loss: 0.0034 Acc: 0.9255
val Loss: 0.0082 Acc: 0.8467

Epoch 58/98
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0078 Acc: 0.8467

Epoch 59/98
----------
train Loss: 0.0031 Acc: 0.9291
val Loss: 0.0074 Acc: 0.8467

Epoch 60/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0031 Acc: 0.9327
val Loss: 0.0074 Acc: 0.8394

Epoch 61/98
----------
train Loss: 0.0032 Acc: 0.9364
val Loss: 0.0090 Acc: 0.8394

Epoch 62/98
----------
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0087 Acc: 0.8394

Epoch 63/98
----------
train Loss: 0.0033 Acc: 0.9255
val Loss: 0.0081 Acc: 0.8394

Epoch 64/98
----------
train Loss: 0.0035 Acc: 0.9182
val Loss: 0.0090 Acc: 0.8394

Epoch 65/98
----------
train Loss: 0.0034 Acc: 0.9182
val Loss: 0.0072 Acc: 0.8394

Epoch 66/98
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0033 Acc: 0.9418
val Loss: 0.0076 Acc: 0.8394

Epoch 67/98
----------
train Loss: 0.0032 Acc: 0.9291
val Loss: 0.0089 Acc: 0.8394

Epoch 68/98
----------
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0090 Acc: 0.8394

Epoch 69/98
----------
train Loss: 0.0032 Acc: 0.9364
val Loss: 0.0077 Acc: 0.8394

Epoch 70/98
----------
train Loss: 0.0030 Acc: 0.9273
val Loss: 0.0087 Acc: 0.8467

Epoch 71/98
----------
train Loss: 0.0033 Acc: 0.9182
val Loss: 0.0071 Acc: 0.8394

Epoch 72/98
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0032 Acc: 0.9364
val Loss: 0.0071 Acc: 0.8467

Epoch 73/98
----------
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0076 Acc: 0.8467

Epoch 74/98
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0085 Acc: 0.8467

Epoch 75/98
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0068 Acc: 0.8467

Epoch 76/98
----------
train Loss: 0.0032 Acc: 0.9236
val Loss: 0.0072 Acc: 0.8467

Epoch 77/98
----------
train Loss: 0.0033 Acc: 0.9291
val Loss: 0.0071 Acc: 0.8467

Epoch 78/98
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0034 Acc: 0.9182
val Loss: 0.0080 Acc: 0.8467

Epoch 79/98
----------
train Loss: 0.0032 Acc: 0.9309
val Loss: 0.0068 Acc: 0.8467

Epoch 80/98
----------
train Loss: 0.0031 Acc: 0.9327
val Loss: 0.0079 Acc: 0.8394

Epoch 81/98
----------
train Loss: 0.0036 Acc: 0.9236
val Loss: 0.0088 Acc: 0.8394

Epoch 82/98
----------
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0079 Acc: 0.8394

Epoch 83/98
----------
train Loss: 0.0031 Acc: 0.9345
val Loss: 0.0076 Acc: 0.8394

Epoch 84/98
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0034 Acc: 0.9255
val Loss: 0.0077 Acc: 0.8394

Epoch 85/98
----------
train Loss: 0.0032 Acc: 0.9345
val Loss: 0.0083 Acc: 0.8394

Epoch 86/98
----------
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0078 Acc: 0.8394

Epoch 87/98
----------
train Loss: 0.0032 Acc: 0.9255
val Loss: 0.0072 Acc: 0.8467

Epoch 88/98
----------
train Loss: 0.0032 Acc: 0.9400
val Loss: 0.0078 Acc: 0.8467

Epoch 89/98
----------
train Loss: 0.0031 Acc: 0.9309
val Loss: 0.0073 Acc: 0.8467

Epoch 90/98
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0032 Acc: 0.9218
val Loss: 0.0074 Acc: 0.8394

Epoch 91/98
----------
train Loss: 0.0032 Acc: 0.9255
val Loss: 0.0081 Acc: 0.8394

Epoch 92/98
----------
train Loss: 0.0032 Acc: 0.9327
val Loss: 0.0083 Acc: 0.8394

Epoch 93/98
----------
train Loss: 0.0033 Acc: 0.9109
val Loss: 0.0079 Acc: 0.8467

Epoch 94/98
----------
train Loss: 0.0034 Acc: 0.9273
val Loss: 0.0071 Acc: 0.8394

Epoch 95/98
----------
train Loss: 0.0033 Acc: 0.9327
val Loss: 0.0084 Acc: 0.8394

Epoch 96/98
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0033 Acc: 0.9345
val Loss: 0.0079 Acc: 0.8467

Epoch 97/98
----------
train Loss: 0.0034 Acc: 0.9255
val Loss: 0.0073 Acc: 0.8467

Epoch 98/98
----------
train Loss: 0.0033 Acc: 0.9309
val Loss: 0.0077 Acc: 0.8467

Training complete in 4m 56s
Best val Acc: 0.861314

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0035 Acc: 0.9091
val Loss: 0.0075 Acc: 0.8248

Epoch 1/98
----------
train Loss: 0.0016 Acc: 0.9745
val Loss: 0.0055 Acc: 0.8832

Epoch 2/98
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0056 Acc: 0.8613

Epoch 3/98
----------
train Loss: 0.0003 Acc: 0.9982
val Loss: 0.0054 Acc: 0.8540

Epoch 4/98
----------
train Loss: 0.0003 Acc: 0.9982
val Loss: 0.0051 Acc: 0.8686

Epoch 5/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 6/98
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 7/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 8/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8832

Epoch 9/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8832

Epoch 10/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8832

Epoch 11/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8832

Epoch 12/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 13/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 14/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8759

Epoch 15/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 16/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8759

Epoch 17/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 18/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 19/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 20/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 21/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 22/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8759

Epoch 23/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8759

Epoch 24/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 25/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8832

Epoch 26/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8832

Epoch 27/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 28/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8759

Epoch 29/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8832

Epoch 30/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 31/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 32/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 33/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 34/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8759

Epoch 35/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 36/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 37/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8759

Epoch 38/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8759

Epoch 39/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 40/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8686

Epoch 41/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8686

Epoch 42/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8686

Epoch 43/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8686

Epoch 44/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 45/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 46/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 47/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 48/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 49/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8759

Epoch 50/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8759

Epoch 51/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 52/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 53/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 54/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 55/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 56/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 57/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 58/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8759

Epoch 59/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8686

Epoch 60/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8759

Epoch 61/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 62/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 63/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 64/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8759

Epoch 65/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 66/98
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 67/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 68/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 69/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 70/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 71/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8759

Epoch 72/98
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8759

Epoch 73/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 74/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 75/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 76/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8759

Epoch 77/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8759

Epoch 78/98
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8759

Epoch 79/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 80/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 81/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 82/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8686

Epoch 83/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 84/98
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 85/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 86/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 87/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 88/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8759

Epoch 89/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 90/98
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8759

Epoch 91/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8759

Epoch 92/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8759

Epoch 93/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 94/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 95/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8759

Epoch 96/98
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 97/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 98/98
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Training complete in 5m 11s
Best val Acc: 0.883212

---Testing---
Test accuracy: 0.959243
--------------------
Accuracy of Carcharhiniformes : 96 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 89 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 95 %
Accuracy of Squaliformes : 85 %
Accuracy of Squatiniformes : 97 %
mean: 0.9516467801596731, std: 0.04735221556895044
--------------------

run info[val: 0.25, epoch: 58, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0246 Acc: 0.1860
val Loss: 0.0226 Acc: 0.2690

Epoch 1/57
----------
train Loss: 0.0165 Acc: 0.4884
val Loss: 0.0135 Acc: 0.5906

Epoch 2/57
----------
train Loss: 0.0104 Acc: 0.6996
val Loss: 0.0091 Acc: 0.7719

Epoch 3/57
----------
train Loss: 0.0074 Acc: 0.8256
val Loss: 0.0076 Acc: 0.7719

Epoch 4/57
----------
train Loss: 0.0056 Acc: 0.8585
val Loss: 0.0071 Acc: 0.8129

Epoch 5/57
----------
train Loss: 0.0048 Acc: 0.8798
val Loss: 0.0069 Acc: 0.7778

Epoch 6/57
----------
train Loss: 0.0048 Acc: 0.8682
val Loss: 0.0067 Acc: 0.8363

Epoch 7/57
----------
train Loss: 0.0036 Acc: 0.9089
val Loss: 0.0072 Acc: 0.8421

Epoch 8/57
----------
train Loss: 0.0034 Acc: 0.9050
val Loss: 0.0071 Acc: 0.8304

Epoch 9/57
----------
train Loss: 0.0033 Acc: 0.9205
val Loss: 0.0058 Acc: 0.8421

Epoch 10/57
----------
train Loss: 0.0027 Acc: 0.9399
val Loss: 0.0061 Acc: 0.8480

Epoch 11/57
----------
train Loss: 0.0024 Acc: 0.9516
val Loss: 0.0066 Acc: 0.8070

Epoch 12/57
----------
train Loss: 0.0024 Acc: 0.9419
val Loss: 0.0058 Acc: 0.8421

Epoch 13/57
----------
train Loss: 0.0022 Acc: 0.9612
val Loss: 0.0060 Acc: 0.8538

Epoch 14/57
----------
LR is set to 0.001
train Loss: 0.0020 Acc: 0.9612
val Loss: 0.0058 Acc: 0.8480

Epoch 15/57
----------
train Loss: 0.0018 Acc: 0.9767
val Loss: 0.0060 Acc: 0.8421

Epoch 16/57
----------
train Loss: 0.0017 Acc: 0.9729
val Loss: 0.0061 Acc: 0.8421

Epoch 17/57
----------
train Loss: 0.0019 Acc: 0.9729
val Loss: 0.0059 Acc: 0.8538

Epoch 18/57
----------
train Loss: 0.0021 Acc: 0.9690
val Loss: 0.0059 Acc: 0.8421

Epoch 19/57
----------
train Loss: 0.0018 Acc: 0.9806
val Loss: 0.0058 Acc: 0.8480

Epoch 20/57
----------
train Loss: 0.0020 Acc: 0.9729
val Loss: 0.0059 Acc: 0.8480

Epoch 21/57
----------
train Loss: 0.0016 Acc: 0.9709
val Loss: 0.0056 Acc: 0.8480

Epoch 22/57
----------
train Loss: 0.0015 Acc: 0.9826
val Loss: 0.0055 Acc: 0.8480

Epoch 23/57
----------
train Loss: 0.0019 Acc: 0.9826
val Loss: 0.0058 Acc: 0.8480

Epoch 24/57
----------
train Loss: 0.0018 Acc: 0.9767
val Loss: 0.0058 Acc: 0.8421

Epoch 25/57
----------
train Loss: 0.0017 Acc: 0.9767
val Loss: 0.0059 Acc: 0.8421

Epoch 26/57
----------
train Loss: 0.0015 Acc: 0.9864
val Loss: 0.0058 Acc: 0.8538

Epoch 27/57
----------
train Loss: 0.0020 Acc: 0.9787
val Loss: 0.0058 Acc: 0.8480

Epoch 28/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8480

Epoch 29/57
----------
train Loss: 0.0018 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8538

Epoch 30/57
----------
train Loss: 0.0018 Acc: 0.9787
val Loss: 0.0057 Acc: 0.8538

Epoch 31/57
----------
train Loss: 0.0017 Acc: 0.9767
val Loss: 0.0057 Acc: 0.8538

Epoch 32/57
----------
train Loss: 0.0017 Acc: 0.9690
val Loss: 0.0060 Acc: 0.8480

Epoch 33/57
----------
train Loss: 0.0018 Acc: 0.9806
val Loss: 0.0059 Acc: 0.8480

Epoch 34/57
----------
train Loss: 0.0016 Acc: 0.9787
val Loss: 0.0057 Acc: 0.8480

Epoch 35/57
----------
train Loss: 0.0016 Acc: 0.9845
val Loss: 0.0059 Acc: 0.8480

Epoch 36/57
----------
train Loss: 0.0017 Acc: 0.9787
val Loss: 0.0059 Acc: 0.8421

Epoch 37/57
----------
train Loss: 0.0021 Acc: 0.9787
val Loss: 0.0062 Acc: 0.8480

Epoch 38/57
----------
train Loss: 0.0017 Acc: 0.9767
val Loss: 0.0062 Acc: 0.8538

Epoch 39/57
----------
train Loss: 0.0018 Acc: 0.9826
val Loss: 0.0058 Acc: 0.8538

Epoch 40/57
----------
train Loss: 0.0017 Acc: 0.9748
val Loss: 0.0056 Acc: 0.8480

Epoch 41/57
----------
train Loss: 0.0016 Acc: 0.9845
val Loss: 0.0056 Acc: 0.8480

Epoch 42/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9806
val Loss: 0.0058 Acc: 0.8480

Epoch 43/57
----------
train Loss: 0.0018 Acc: 0.9826
val Loss: 0.0058 Acc: 0.8421

Epoch 44/57
----------
train Loss: 0.0018 Acc: 0.9806
val Loss: 0.0062 Acc: 0.8480

Epoch 45/57
----------
train Loss: 0.0016 Acc: 0.9845
val Loss: 0.0061 Acc: 0.8480

Epoch 46/57
----------
train Loss: 0.0015 Acc: 0.9864
val Loss: 0.0059 Acc: 0.8480

Epoch 47/57
----------
train Loss: 0.0019 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8480

Epoch 48/57
----------
train Loss: 0.0019 Acc: 0.9845
val Loss: 0.0059 Acc: 0.8363

Epoch 49/57
----------
train Loss: 0.0019 Acc: 0.9767
val Loss: 0.0060 Acc: 0.8421

Epoch 50/57
----------
train Loss: 0.0019 Acc: 0.9748
val Loss: 0.0057 Acc: 0.8480

Epoch 51/57
----------
train Loss: 0.0017 Acc: 0.9806
val Loss: 0.0058 Acc: 0.8538

Epoch 52/57
----------
train Loss: 0.0015 Acc: 0.9767
val Loss: 0.0059 Acc: 0.8480

Epoch 53/57
----------
train Loss: 0.0017 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8480

Epoch 54/57
----------
train Loss: 0.0018 Acc: 0.9845
val Loss: 0.0057 Acc: 0.8480

Epoch 55/57
----------
train Loss: 0.0017 Acc: 0.9806
val Loss: 0.0060 Acc: 0.8480

Epoch 56/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9767
val Loss: 0.0057 Acc: 0.8480

Epoch 57/57
----------
train Loss: 0.0017 Acc: 0.9806
val Loss: 0.0057 Acc: 0.8421

Training complete in 2m 56s
Best val Acc: 0.853801

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0019 Acc: 0.9612
val Loss: 0.0062 Acc: 0.8596

Epoch 1/57
----------
train Loss: 0.0016 Acc: 0.9709
val Loss: 0.0060 Acc: 0.8596

Epoch 2/57
----------
train Loss: 0.0009 Acc: 0.9806
val Loss: 0.0083 Acc: 0.7895

Epoch 3/57
----------
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0062 Acc: 0.8480

Epoch 4/57
----------
train Loss: 0.0004 Acc: 0.9922
val Loss: 0.0061 Acc: 0.8480

Epoch 5/57
----------
train Loss: 0.0009 Acc: 0.9961
val Loss: 0.0053 Acc: 0.8655

Epoch 6/57
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0063 Acc: 0.8538

Epoch 7/57
----------
train Loss: 0.0004 Acc: 0.9942
val Loss: 0.0061 Acc: 0.8596

Epoch 8/57
----------
train Loss: 0.0004 Acc: 0.9981
val Loss: 0.0059 Acc: 0.8772

Epoch 9/57
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0063 Acc: 0.8655

Epoch 10/57
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0066 Acc: 0.8713

Epoch 11/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8655

Epoch 12/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 13/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8538

Epoch 14/57
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8596

Epoch 15/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8655

Epoch 16/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8655

Epoch 17/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8655

Epoch 18/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8713

Epoch 19/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8655

Epoch 20/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8655

Epoch 21/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8655

Epoch 22/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8655

Epoch 23/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8713

Epoch 24/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8655

Epoch 25/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8772

Epoch 26/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8772

Epoch 27/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8713

Epoch 28/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8713

Epoch 29/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8772

Epoch 30/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8713

Epoch 31/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 32/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8713

Epoch 33/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8713

Epoch 34/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8713

Epoch 35/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8713

Epoch 36/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8655

Epoch 37/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8713

Epoch 38/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 39/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8713

Epoch 40/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8713

Epoch 41/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 42/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Epoch 43/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8713

Epoch 44/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8655

Epoch 45/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8713

Epoch 46/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8713

Epoch 47/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8655

Epoch 48/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8655

Epoch 49/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8713

Epoch 50/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8655

Epoch 51/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8713

Epoch 52/57
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0059 Acc: 0.8713

Epoch 53/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8655

Epoch 54/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8596

Epoch 55/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8655

Epoch 56/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8713

Epoch 57/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8713

Training complete in 3m 2s
Best val Acc: 0.877193

---Testing---
Test accuracy: 0.969432
--------------------
Accuracy of Carcharhiniformes : 95 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.962514058560067, std: 0.03839434422790757
--------------------

run info[val: 0.3, epoch: 90, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0215 Acc: 0.2516
val Loss: 0.0273 Acc: 0.3155

Epoch 1/89
----------
train Loss: 0.0157 Acc: 0.5010
val Loss: 0.0182 Acc: 0.6650

Epoch 2/89
----------
train Loss: 0.0103 Acc: 0.7027
val Loss: 0.0128 Acc: 0.6893

Epoch 3/89
----------
train Loss: 0.0069 Acc: 0.8150
val Loss: 0.0070 Acc: 0.8301

Epoch 4/89
----------
train Loss: 0.0058 Acc: 0.8212
val Loss: 0.0099 Acc: 0.8058

Epoch 5/89
----------
LR is set to 0.001
train Loss: 0.0045 Acc: 0.8690
val Loss: 0.0081 Acc: 0.8107

Epoch 6/89
----------
train Loss: 0.0043 Acc: 0.8898
val Loss: 0.0069 Acc: 0.8204

Epoch 7/89
----------
train Loss: 0.0042 Acc: 0.8960
val Loss: 0.0062 Acc: 0.8252

Epoch 8/89
----------
train Loss: 0.0041 Acc: 0.9127
val Loss: 0.0081 Acc: 0.8301

Epoch 9/89
----------
train Loss: 0.0042 Acc: 0.9002
val Loss: 0.0058 Acc: 0.8350

Epoch 10/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0039 Acc: 0.9231
val Loss: 0.0082 Acc: 0.8398

Epoch 11/89
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0064 Acc: 0.8398

Epoch 12/89
----------
train Loss: 0.0039 Acc: 0.9023
val Loss: 0.0071 Acc: 0.8398

Epoch 13/89
----------
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0073 Acc: 0.8350

Epoch 14/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0071 Acc: 0.8350

Epoch 15/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0042 Acc: 0.9064
val Loss: 0.0076 Acc: 0.8350

Epoch 16/89
----------
train Loss: 0.0042 Acc: 0.8960
val Loss: 0.0070 Acc: 0.8350

Epoch 17/89
----------
train Loss: 0.0039 Acc: 0.9002
val Loss: 0.0059 Acc: 0.8350

Epoch 18/89
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0078 Acc: 0.8398

Epoch 19/89
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0086 Acc: 0.8398

Epoch 20/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0068 Acc: 0.8301

Epoch 21/89
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0094 Acc: 0.8350

Epoch 22/89
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0072 Acc: 0.8350

Epoch 23/89
----------
train Loss: 0.0039 Acc: 0.9189
val Loss: 0.0077 Acc: 0.8398

Epoch 24/89
----------
train Loss: 0.0041 Acc: 0.9148
val Loss: 0.0088 Acc: 0.8350

Epoch 25/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0075 Acc: 0.8398

Epoch 26/89
----------
train Loss: 0.0041 Acc: 0.9127
val Loss: 0.0077 Acc: 0.8398

Epoch 27/89
----------
train Loss: 0.0040 Acc: 0.9127
val Loss: 0.0060 Acc: 0.8398

Epoch 28/89
----------
train Loss: 0.0041 Acc: 0.9002
val Loss: 0.0075 Acc: 0.8398

Epoch 29/89
----------
train Loss: 0.0041 Acc: 0.9168
val Loss: 0.0084 Acc: 0.8350

Epoch 30/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0084 Acc: 0.8350

Epoch 31/89
----------
train Loss: 0.0043 Acc: 0.9023
val Loss: 0.0080 Acc: 0.8398

Epoch 32/89
----------
train Loss: 0.0042 Acc: 0.9023
val Loss: 0.0082 Acc: 0.8301

Epoch 33/89
----------
train Loss: 0.0040 Acc: 0.9127
val Loss: 0.0059 Acc: 0.8301

Epoch 34/89
----------
train Loss: 0.0040 Acc: 0.8898
val Loss: 0.0103 Acc: 0.8398

Epoch 35/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0065 Acc: 0.8350

Epoch 36/89
----------
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0091 Acc: 0.8398

Epoch 37/89
----------
train Loss: 0.0040 Acc: 0.9168
val Loss: 0.0071 Acc: 0.8398

Epoch 38/89
----------
train Loss: 0.0041 Acc: 0.9002
val Loss: 0.0092 Acc: 0.8350

Epoch 39/89
----------
train Loss: 0.0039 Acc: 0.9106
val Loss: 0.0077 Acc: 0.8350

Epoch 40/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0062 Acc: 0.8350

Epoch 41/89
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0074 Acc: 0.8398

Epoch 42/89
----------
train Loss: 0.0041 Acc: 0.9148
val Loss: 0.0062 Acc: 0.8301

Epoch 43/89
----------
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0071 Acc: 0.8398

Epoch 44/89
----------
train Loss: 0.0041 Acc: 0.8960
val Loss: 0.0059 Acc: 0.8350

Epoch 45/89
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0097 Acc: 0.8350

Epoch 46/89
----------
train Loss: 0.0041 Acc: 0.9127
val Loss: 0.0070 Acc: 0.8398

Epoch 47/89
----------
train Loss: 0.0041 Acc: 0.9085
val Loss: 0.0071 Acc: 0.8301

Epoch 48/89
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0059 Acc: 0.8350

Epoch 49/89
----------
train Loss: 0.0040 Acc: 0.8960
val Loss: 0.0062 Acc: 0.8398

Epoch 50/89
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0042 Acc: 0.8940
val Loss: 0.0083 Acc: 0.8447

Epoch 51/89
----------
train Loss: 0.0040 Acc: 0.9106
val Loss: 0.0079 Acc: 0.8252

Epoch 52/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0076 Acc: 0.8398

Epoch 53/89
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0073 Acc: 0.8350

Epoch 54/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0094 Acc: 0.8301

Epoch 55/89
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0039 Acc: 0.9106
val Loss: 0.0080 Acc: 0.8301

Epoch 56/89
----------
train Loss: 0.0040 Acc: 0.8898
val Loss: 0.0073 Acc: 0.8350

Epoch 57/89
----------
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0073 Acc: 0.8447

Epoch 58/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0093 Acc: 0.8350

Epoch 59/89
----------
train Loss: 0.0041 Acc: 0.9002
val Loss: 0.0091 Acc: 0.8398

Epoch 60/89
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0074 Acc: 0.8398

Epoch 61/89
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0083 Acc: 0.8398

Epoch 62/89
----------
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0061 Acc: 0.8398

Epoch 63/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0102 Acc: 0.8350

Epoch 64/89
----------
train Loss: 0.0039 Acc: 0.9210
val Loss: 0.0076 Acc: 0.8447

Epoch 65/89
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0040 Acc: 0.9127
val Loss: 0.0065 Acc: 0.8350

Epoch 66/89
----------
train Loss: 0.0041 Acc: 0.9210
val Loss: 0.0070 Acc: 0.8350

Epoch 67/89
----------
train Loss: 0.0040 Acc: 0.9002
val Loss: 0.0064 Acc: 0.8398

Epoch 68/89
----------
train Loss: 0.0039 Acc: 0.9168
val Loss: 0.0078 Acc: 0.8350

Epoch 69/89
----------
train Loss: 0.0041 Acc: 0.8940
val Loss: 0.0070 Acc: 0.8398

Epoch 70/89
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0041 Acc: 0.9085
val Loss: 0.0072 Acc: 0.8350

Epoch 71/89
----------
train Loss: 0.0042 Acc: 0.9023
val Loss: 0.0077 Acc: 0.8350

Epoch 72/89
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0066 Acc: 0.8350

Epoch 73/89
----------
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0069 Acc: 0.8301

Epoch 74/89
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0070 Acc: 0.8301

Epoch 75/89
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0040 Acc: 0.9023
val Loss: 0.0100 Acc: 0.8350

Epoch 76/89
----------
train Loss: 0.0040 Acc: 0.9044
val Loss: 0.0082 Acc: 0.8301

Epoch 77/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0074 Acc: 0.8398

Epoch 78/89
----------
train Loss: 0.0041 Acc: 0.8981
val Loss: 0.0081 Acc: 0.8301

Epoch 79/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0075 Acc: 0.8301

Epoch 80/89
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0042 Acc: 0.8919
val Loss: 0.0085 Acc: 0.8350

Epoch 81/89
----------
train Loss: 0.0041 Acc: 0.9106
val Loss: 0.0077 Acc: 0.8301

Epoch 82/89
----------
train Loss: 0.0040 Acc: 0.9127
val Loss: 0.0063 Acc: 0.8301

Epoch 83/89
----------
train Loss: 0.0039 Acc: 0.9106
val Loss: 0.0085 Acc: 0.8398

Epoch 84/89
----------
train Loss: 0.0041 Acc: 0.9044
val Loss: 0.0083 Acc: 0.8398

Epoch 85/89
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0042 Acc: 0.9044
val Loss: 0.0071 Acc: 0.8350

Epoch 86/89
----------
train Loss: 0.0040 Acc: 0.9085
val Loss: 0.0066 Acc: 0.8350

Epoch 87/89
----------
train Loss: 0.0041 Acc: 0.9064
val Loss: 0.0072 Acc: 0.8398

Epoch 88/89
----------
train Loss: 0.0040 Acc: 0.9168
val Loss: 0.0068 Acc: 0.8350

Epoch 89/89
----------
train Loss: 0.0043 Acc: 0.9044
val Loss: 0.0076 Acc: 0.8398

Training complete in 4m 36s
Best val Acc: 0.844660

---Fine tuning.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0041 Acc: 0.8940
val Loss: 0.0053 Acc: 0.8641

Epoch 1/89
----------
train Loss: 0.0022 Acc: 0.9647
val Loss: 0.0109 Acc: 0.8592

Epoch 2/89
----------
train Loss: 0.0009 Acc: 0.9958
val Loss: 0.0062 Acc: 0.8932

Epoch 3/89
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0039 Acc: 0.8835

Epoch 4/89
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8835

Epoch 5/89
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8835

Epoch 6/89
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8835

Epoch 7/89
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 8/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8883

Epoch 9/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8883

Epoch 10/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8883

Epoch 11/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8883

Epoch 12/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8883

Epoch 13/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8883

Epoch 14/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 15/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 16/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0031 Acc: 0.8835

Epoch 17/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8883

Epoch 18/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8883

Epoch 19/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 20/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 21/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8835

Epoch 22/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8835

Epoch 23/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8835

Epoch 24/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8835

Epoch 25/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8835

Epoch 26/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8883

Epoch 27/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8883

Epoch 28/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8883

Epoch 29/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8835

Epoch 30/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8835

Epoch 31/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 32/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8932

Epoch 33/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 34/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 35/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 36/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8883

Epoch 37/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8835

Epoch 38/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 39/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 40/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8835

Epoch 41/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 42/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 43/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 44/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 45/89
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 46/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8883

Epoch 47/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 48/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8883

Epoch 49/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8835

Epoch 50/89
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 51/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8835

Epoch 52/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8883

Epoch 53/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8883

Epoch 54/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8883

Epoch 55/89
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8883

Epoch 56/89
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8835

Epoch 57/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 58/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8835

Epoch 59/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8835

Epoch 60/89
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8835

Epoch 61/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8883

Epoch 62/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8883

Epoch 63/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 64/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8883

Epoch 65/89
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 66/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8883

Epoch 67/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 68/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8883

Epoch 69/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8835

Epoch 70/89
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8835

Epoch 71/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8835

Epoch 72/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8835

Epoch 73/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8883

Epoch 74/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 75/89
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8883

Epoch 76/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8883

Epoch 77/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8835

Epoch 78/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8883

Epoch 79/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8835

Epoch 80/89
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8835

Epoch 81/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8835

Epoch 82/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.8835

Epoch 83/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8835

Epoch 84/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.8835

Epoch 85/89
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.8835

Epoch 86/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8835

Epoch 87/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8835

Epoch 88/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8835

Epoch 89/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8835

Training complete in 4m 50s
Best val Acc: 0.893204

---Testing---
Test accuracy: 0.963610
--------------------
Accuracy of Carcharhiniformes : 100 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 82 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 87 %
Accuracy of Squatiniformes : 96 %
mean: 0.9501912059740565, std: 0.06083435545028682

Model saved in "./weights/shark_[0.98]_mean[0.97]_std[0.03].save".
--------------------

run info[val: 0.1, epoch: 58, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0229 Acc: 0.2100
val Loss: 0.0237 Acc: 0.3382

Epoch 1/57
----------
train Loss: 0.0132 Acc: 0.5994
val Loss: 0.0149 Acc: 0.6765

Epoch 2/57
----------
train Loss: 0.0075 Acc: 0.8142
val Loss: 0.0112 Acc: 0.7206

Epoch 3/57
----------
train Loss: 0.0057 Acc: 0.8498
val Loss: 0.0105 Acc: 0.7794

Epoch 4/57
----------
train Loss: 0.0051 Acc: 0.8756
val Loss: 0.0100 Acc: 0.7647

Epoch 5/57
----------
train Loss: 0.0042 Acc: 0.8950
val Loss: 0.0107 Acc: 0.7500

Epoch 6/57
----------
train Loss: 0.0039 Acc: 0.8982
val Loss: 0.0089 Acc: 0.8088

Epoch 7/57
----------
train Loss: 0.0037 Acc: 0.9144
val Loss: 0.0107 Acc: 0.7500

Epoch 8/57
----------
train Loss: 0.0033 Acc: 0.9354
val Loss: 0.0092 Acc: 0.7794

Epoch 9/57
----------
train Loss: 0.0029 Acc: 0.9483
val Loss: 0.0089 Acc: 0.7647

Epoch 10/57
----------
train Loss: 0.0032 Acc: 0.9418
val Loss: 0.0099 Acc: 0.7647

Epoch 11/57
----------
train Loss: 0.0029 Acc: 0.9338
val Loss: 0.0099 Acc: 0.7794

Epoch 12/57
----------
train Loss: 0.0027 Acc: 0.9386
val Loss: 0.0102 Acc: 0.7647

Epoch 13/57
----------
train Loss: 0.0024 Acc: 0.9418
val Loss: 0.0088 Acc: 0.7647

Epoch 14/57
----------
LR is set to 0.001
train Loss: 0.0020 Acc: 0.9790
val Loss: 0.0087 Acc: 0.7647

Epoch 15/57
----------
train Loss: 0.0019 Acc: 0.9790
val Loss: 0.0087 Acc: 0.7794

Epoch 16/57
----------
train Loss: 0.0018 Acc: 0.9742
val Loss: 0.0092 Acc: 0.7647

Epoch 17/57
----------
train Loss: 0.0017 Acc: 0.9806
val Loss: 0.0092 Acc: 0.7647

Epoch 18/57
----------
train Loss: 0.0020 Acc: 0.9758
val Loss: 0.0090 Acc: 0.7647

Epoch 19/57
----------
train Loss: 0.0019 Acc: 0.9758
val Loss: 0.0084 Acc: 0.7794

Epoch 20/57
----------
train Loss: 0.0015 Acc: 0.9855
val Loss: 0.0087 Acc: 0.7647

Epoch 21/57
----------
train Loss: 0.0017 Acc: 0.9790
val Loss: 0.0090 Acc: 0.7794

Epoch 22/57
----------
train Loss: 0.0016 Acc: 0.9790
val Loss: 0.0091 Acc: 0.7794

Epoch 23/57
----------
train Loss: 0.0016 Acc: 0.9774
val Loss: 0.0092 Acc: 0.7794

Epoch 24/57
----------
train Loss: 0.0016 Acc: 0.9822
val Loss: 0.0093 Acc: 0.7794

Epoch 25/57
----------
train Loss: 0.0016 Acc: 0.9790
val Loss: 0.0090 Acc: 0.7647

Epoch 26/57
----------
train Loss: 0.0018 Acc: 0.9790
val Loss: 0.0088 Acc: 0.7794

Epoch 27/57
----------
train Loss: 0.0017 Acc: 0.9790
val Loss: 0.0087 Acc: 0.7794

Epoch 28/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0015 Acc: 0.9822
val Loss: 0.0087 Acc: 0.7941

Epoch 29/57
----------
train Loss: 0.0017 Acc: 0.9758
val Loss: 0.0087 Acc: 0.7794

Epoch 30/57
----------
train Loss: 0.0016 Acc: 0.9758
val Loss: 0.0087 Acc: 0.7941

Epoch 31/57
----------
train Loss: 0.0018 Acc: 0.9838
val Loss: 0.0088 Acc: 0.7941

Epoch 32/57
----------
train Loss: 0.0016 Acc: 0.9806
val Loss: 0.0086 Acc: 0.7941

Epoch 33/57
----------
train Loss: 0.0016 Acc: 0.9790
val Loss: 0.0089 Acc: 0.7794

Epoch 34/57
----------
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0088 Acc: 0.7794

Epoch 35/57
----------
train Loss: 0.0016 Acc: 0.9822
val Loss: 0.0088 Acc: 0.7794

Epoch 36/57
----------
train Loss: 0.0014 Acc: 0.9871
val Loss: 0.0089 Acc: 0.7794

Epoch 37/57
----------
train Loss: 0.0015 Acc: 0.9871
val Loss: 0.0089 Acc: 0.7794

Epoch 38/57
----------
train Loss: 0.0017 Acc: 0.9774
val Loss: 0.0090 Acc: 0.7794

Epoch 39/57
----------
train Loss: 0.0015 Acc: 0.9822
val Loss: 0.0090 Acc: 0.7794

Epoch 40/57
----------
train Loss: 0.0016 Acc: 0.9838
val Loss: 0.0089 Acc: 0.7794

Epoch 41/57
----------
train Loss: 0.0018 Acc: 0.9790
val Loss: 0.0089 Acc: 0.7794

Epoch 42/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9790
val Loss: 0.0087 Acc: 0.7794

Epoch 43/57
----------
train Loss: 0.0017 Acc: 0.9871
val Loss: 0.0090 Acc: 0.7794

Epoch 44/57
----------
train Loss: 0.0016 Acc: 0.9822
val Loss: 0.0090 Acc: 0.7794

Epoch 45/57
----------
train Loss: 0.0016 Acc: 0.9838
val Loss: 0.0090 Acc: 0.7794

Epoch 46/57
----------
train Loss: 0.0015 Acc: 0.9790
val Loss: 0.0089 Acc: 0.7794

Epoch 47/57
----------
train Loss: 0.0014 Acc: 0.9806
val Loss: 0.0089 Acc: 0.7794

Epoch 48/57
----------
train Loss: 0.0017 Acc: 0.9838
val Loss: 0.0092 Acc: 0.7794

Epoch 49/57
----------
train Loss: 0.0018 Acc: 0.9806
val Loss: 0.0091 Acc: 0.7794

Epoch 50/57
----------
train Loss: 0.0015 Acc: 0.9822
val Loss: 0.0088 Acc: 0.7794

Epoch 51/57
----------
train Loss: 0.0015 Acc: 0.9806
val Loss: 0.0089 Acc: 0.7794

Epoch 52/57
----------
train Loss: 0.0019 Acc: 0.9806
val Loss: 0.0089 Acc: 0.7794

Epoch 53/57
----------
train Loss: 0.0016 Acc: 0.9822
val Loss: 0.0090 Acc: 0.7794

Epoch 54/57
----------
train Loss: 0.0016 Acc: 0.9709
val Loss: 0.0091 Acc: 0.7794

Epoch 55/57
----------
train Loss: 0.0018 Acc: 0.9838
val Loss: 0.0090 Acc: 0.7794

Epoch 56/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9758
val Loss: 0.0092 Acc: 0.7794

Epoch 57/57
----------
train Loss: 0.0016 Acc: 0.9855
val Loss: 0.0090 Acc: 0.7794

Training complete in 3m 0s
Best val Acc: 0.808824

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0031 Acc: 0.9192
val Loss: 0.0106 Acc: 0.7941

Epoch 1/57
----------
train Loss: 0.0017 Acc: 0.9677
val Loss: 0.0092 Acc: 0.7794

Epoch 2/57
----------
train Loss: 0.0011 Acc: 0.9806
val Loss: 0.0084 Acc: 0.8235

Epoch 3/57
----------
train Loss: 0.0004 Acc: 0.9984
val Loss: 0.0112 Acc: 0.7647

Epoch 4/57
----------
train Loss: 0.0004 Acc: 0.9935
val Loss: 0.0079 Acc: 0.8235

Epoch 5/57
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0084 Acc: 0.8088

Epoch 6/57
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0081 Acc: 0.8088

Epoch 7/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8088

Epoch 8/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8382

Epoch 9/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8382

Epoch 10/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8382

Epoch 11/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8382

Epoch 12/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8382

Epoch 13/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8382

Epoch 14/57
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 15/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8382

Epoch 16/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8529

Epoch 17/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8529

Epoch 18/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 19/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 20/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8529

Epoch 21/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8529

Epoch 22/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8529

Epoch 23/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 24/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 25/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 26/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 27/57
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 28/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 29/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 30/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8382

Epoch 31/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8529

Epoch 32/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 33/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8382

Epoch 34/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8529

Epoch 35/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 36/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 37/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 38/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 39/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 40/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 41/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 42/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 43/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 44/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 45/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 46/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 47/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8382

Epoch 48/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 49/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 50/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8529

Epoch 51/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8382

Epoch 52/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8382

Epoch 53/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8529

Epoch 54/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8529

Epoch 55/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8382

Epoch 56/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8382

Epoch 57/57
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8382

Training complete in 3m 15s
Best val Acc: 0.852941

---Testing---
Test accuracy: 0.985444
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 98 %
mean: 0.9797903850535429, std: 0.018670839575254096
--------------------

run info[val: 0.15, epoch: 93, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0209 Acc: 0.2432
val Loss: 0.0412 Acc: 0.2427

Epoch 1/92
----------
train Loss: 0.0136 Acc: 0.5616
val Loss: 0.0236 Acc: 0.6893

Epoch 2/92
----------
train Loss: 0.0078 Acc: 0.8202
val Loss: 0.0177 Acc: 0.7087

Epoch 3/92
----------
train Loss: 0.0057 Acc: 0.8151
val Loss: 0.0150 Acc: 0.7476

Epoch 4/92
----------
train Loss: 0.0044 Acc: 0.8767
val Loss: 0.0094 Acc: 0.7573

Epoch 5/92
----------
train Loss: 0.0038 Acc: 0.9092
val Loss: 0.0212 Acc: 0.7767

Epoch 6/92
----------
train Loss: 0.0032 Acc: 0.9229
val Loss: 0.0083 Acc: 0.7767

Epoch 7/92
----------
train Loss: 0.0031 Acc: 0.9161
val Loss: 0.0256 Acc: 0.7864

Epoch 8/92
----------
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0103 Acc: 0.7767

Epoch 9/92
----------
train Loss: 0.0023 Acc: 0.9538
val Loss: 0.0101 Acc: 0.7961

Epoch 10/92
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0126 Acc: 0.7961

Epoch 11/92
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0061 Acc: 0.7961

Epoch 12/92
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0062 Acc: 0.8058

Epoch 13/92
----------
train Loss: 0.0018 Acc: 0.9777
val Loss: 0.0066 Acc: 0.8155

Epoch 14/92
----------
LR is set to 0.001
train Loss: 0.0017 Acc: 0.9692
val Loss: 0.0092 Acc: 0.8155

Epoch 15/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0144 Acc: 0.8058

Epoch 16/92
----------
train Loss: 0.0017 Acc: 0.9795
val Loss: 0.0073 Acc: 0.8155

Epoch 17/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0076 Acc: 0.8155

Epoch 18/92
----------
train Loss: 0.0017 Acc: 0.9726
val Loss: 0.0160 Acc: 0.8058

Epoch 19/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0199 Acc: 0.7961

Epoch 20/92
----------
train Loss: 0.0016 Acc: 0.9829
val Loss: 0.0121 Acc: 0.7961

Epoch 21/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0073 Acc: 0.8058

Epoch 22/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0112 Acc: 0.7961

Epoch 23/92
----------
train Loss: 0.0017 Acc: 0.9760
val Loss: 0.0065 Acc: 0.7961

Epoch 24/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0073 Acc: 0.7961

Epoch 25/92
----------
train Loss: 0.0017 Acc: 0.9812
val Loss: 0.0087 Acc: 0.8058

Epoch 26/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0099 Acc: 0.7961

Epoch 27/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0143 Acc: 0.8058

Epoch 28/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9812
val Loss: 0.0108 Acc: 0.8058

Epoch 29/92
----------
train Loss: 0.0016 Acc: 0.9880
val Loss: 0.0066 Acc: 0.8058

Epoch 30/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0063 Acc: 0.8155

Epoch 31/92
----------
train Loss: 0.0015 Acc: 0.9795
val Loss: 0.0059 Acc: 0.8058

Epoch 32/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0094 Acc: 0.8058

Epoch 33/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0065 Acc: 0.8058

Epoch 34/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0121 Acc: 0.7961

Epoch 35/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0155 Acc: 0.8058

Epoch 36/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0103 Acc: 0.8058

Epoch 37/92
----------
train Loss: 0.0015 Acc: 0.9777
val Loss: 0.0118 Acc: 0.8155

Epoch 38/92
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0132 Acc: 0.8155

Epoch 39/92
----------
train Loss: 0.0017 Acc: 0.9846
val Loss: 0.0096 Acc: 0.8155

Epoch 40/92
----------
train Loss: 0.0017 Acc: 0.9777
val Loss: 0.0168 Acc: 0.8058

Epoch 41/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0114 Acc: 0.8155

Epoch 42/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0176 Acc: 0.8155

Epoch 43/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0106 Acc: 0.8058

Epoch 44/92
----------
train Loss: 0.0015 Acc: 0.9795
val Loss: 0.0067 Acc: 0.8155

Epoch 45/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0060 Acc: 0.8058

Epoch 46/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0224 Acc: 0.8058

Epoch 47/92
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0062 Acc: 0.8155

Epoch 48/92
----------
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0090 Acc: 0.8155

Epoch 49/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0089 Acc: 0.8058

Epoch 50/92
----------
train Loss: 0.0016 Acc: 0.9675
val Loss: 0.0152 Acc: 0.8155

Epoch 51/92
----------
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0068 Acc: 0.8155

Epoch 52/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0087 Acc: 0.8155

Epoch 53/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0178 Acc: 0.8058

Epoch 54/92
----------
train Loss: 0.0015 Acc: 0.9795
val Loss: 0.0130 Acc: 0.7961

Epoch 55/92
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0098 Acc: 0.8058

Epoch 56/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0108 Acc: 0.8058

Epoch 57/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0065 Acc: 0.8058

Epoch 58/92
----------
train Loss: 0.0015 Acc: 0.9795
val Loss: 0.0066 Acc: 0.8155

Epoch 59/92
----------
train Loss: 0.0016 Acc: 0.9812
val Loss: 0.0069 Acc: 0.8058

Epoch 60/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0143 Acc: 0.8058

Epoch 61/92
----------
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0060 Acc: 0.8058

Epoch 62/92
----------
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0245 Acc: 0.8058

Epoch 63/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0127 Acc: 0.7961

Epoch 64/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0113 Acc: 0.8058

Epoch 65/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0085 Acc: 0.8155

Epoch 66/92
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0078 Acc: 0.8155

Epoch 67/92
----------
train Loss: 0.0017 Acc: 0.9795
val Loss: 0.0184 Acc: 0.8155

Epoch 68/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0149 Acc: 0.8155

Epoch 69/92
----------
train Loss: 0.0016 Acc: 0.9829
val Loss: 0.0219 Acc: 0.8058

Epoch 70/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0135 Acc: 0.8058

Epoch 71/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0097 Acc: 0.8058

Epoch 72/92
----------
train Loss: 0.0014 Acc: 0.9812
val Loss: 0.0086 Acc: 0.8155

Epoch 73/92
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0066 Acc: 0.8058

Epoch 74/92
----------
train Loss: 0.0015 Acc: 0.9812
val Loss: 0.0059 Acc: 0.8155

Epoch 75/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0065 Acc: 0.7961

Epoch 76/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0061 Acc: 0.8058

Epoch 77/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0088 Acc: 0.8058

Epoch 78/92
----------
train Loss: 0.0016 Acc: 0.9829
val Loss: 0.0099 Acc: 0.7961

Epoch 79/92
----------
train Loss: 0.0015 Acc: 0.9829
val Loss: 0.0236 Acc: 0.8058

Epoch 80/92
----------
train Loss: 0.0016 Acc: 0.9795
val Loss: 0.0112 Acc: 0.8058

Epoch 81/92
----------
train Loss: 0.0016 Acc: 0.9760
val Loss: 0.0104 Acc: 0.8155

Epoch 82/92
----------
train Loss: 0.0015 Acc: 0.9846
val Loss: 0.0093 Acc: 0.8058

Epoch 83/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0069 Acc: 0.8155

Epoch 84/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0103 Acc: 0.8155

Epoch 85/92
----------
train Loss: 0.0015 Acc: 0.9863
val Loss: 0.0082 Acc: 0.8155

Epoch 86/92
----------
train Loss: 0.0016 Acc: 0.9812
val Loss: 0.0098 Acc: 0.8155

Epoch 87/92
----------
train Loss: 0.0015 Acc: 0.9743
val Loss: 0.0097 Acc: 0.8155

Epoch 88/92
----------
train Loss: 0.0014 Acc: 0.9829
val Loss: 0.0190 Acc: 0.8155

Epoch 89/92
----------
train Loss: 0.0016 Acc: 0.9743
val Loss: 0.0251 Acc: 0.8058

Epoch 90/92
----------
train Loss: 0.0017 Acc: 0.9743
val Loss: 0.0065 Acc: 0.8155

Epoch 91/92
----------
train Loss: 0.0016 Acc: 0.9829
val Loss: 0.0086 Acc: 0.8155

Epoch 92/92
----------
train Loss: 0.0015 Acc: 0.9880
val Loss: 0.0065 Acc: 0.8058

Training complete in 5m 5s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0017 Acc: 0.9709
val Loss: 0.0054 Acc: 0.8350

Epoch 1/92
----------
train Loss: 0.0007 Acc: 0.9932
val Loss: 0.0189 Acc: 0.8350

Epoch 2/92
----------
train Loss: 0.0004 Acc: 0.9949
val Loss: 0.0193 Acc: 0.8252

Epoch 3/92
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 4/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0105 Acc: 0.8252

Epoch 5/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8447

Epoch 6/92
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0187 Acc: 0.8350

Epoch 7/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8350

Epoch 8/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 9/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0186 Acc: 0.8350

Epoch 10/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 11/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 12/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8447

Epoch 13/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0177 Acc: 0.8447

Epoch 14/92
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8544

Epoch 15/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8544

Epoch 16/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8447

Epoch 17/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8447

Epoch 18/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8447

Epoch 19/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 20/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8447

Epoch 21/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 22/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0222 Acc: 0.8350

Epoch 23/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 24/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8350

Epoch 25/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8447

Epoch 26/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 27/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 28/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0296 Acc: 0.8447

Epoch 29/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 30/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8447

Epoch 31/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0192 Acc: 0.8447

Epoch 32/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0131 Acc: 0.8447

Epoch 33/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 34/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0190 Acc: 0.8447

Epoch 35/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8447

Epoch 36/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0196 Acc: 0.8447

Epoch 37/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 38/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 39/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 40/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8447

Epoch 41/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8350

Epoch 42/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 43/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0116 Acc: 0.8447

Epoch 44/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 45/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8447

Epoch 46/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 47/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 48/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8447

Epoch 49/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 50/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8447

Epoch 51/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8447

Epoch 52/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8447

Epoch 53/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 54/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0187 Acc: 0.8447

Epoch 55/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8350

Epoch 56/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 57/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8447

Epoch 58/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 59/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0187 Acc: 0.8447

Epoch 60/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8447

Epoch 61/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0119 Acc: 0.8447

Epoch 62/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0196 Acc: 0.8447

Epoch 63/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8447

Epoch 64/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0112 Acc: 0.8447

Epoch 65/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0157 Acc: 0.8447

Epoch 66/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0187 Acc: 0.8447

Epoch 67/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 68/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0118 Acc: 0.8350

Epoch 69/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 70/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8447

Epoch 71/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 72/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8447

Epoch 73/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0267 Acc: 0.8447

Epoch 74/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8447

Epoch 75/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8447

Epoch 76/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0249 Acc: 0.8447

Epoch 77/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0114 Acc: 0.8447

Epoch 78/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 79/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8544

Epoch 80/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8544

Epoch 81/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8544

Epoch 82/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0180 Acc: 0.8447

Epoch 83/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 84/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8447

Epoch 85/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8447

Epoch 86/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8447

Epoch 87/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0180 Acc: 0.8447

Epoch 88/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8350

Epoch 89/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0183 Acc: 0.8350

Epoch 90/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0146 Acc: 0.8350

Epoch 91/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0182 Acc: 0.8447

Epoch 92/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8447

Training complete in 5m 20s
Best val Acc: 0.854369

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 97 %
mean: 0.9702167350193667, std: 0.027094367413520004
--------------------

run info[val: 0.2, epoch: 90, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0225 Acc: 0.2400
val Loss: 0.0290 Acc: 0.1971

Epoch 1/89
----------
train Loss: 0.0149 Acc: 0.5255
val Loss: 0.0167 Acc: 0.6423

Epoch 2/89
----------
train Loss: 0.0090 Acc: 0.7618
val Loss: 0.0102 Acc: 0.7956

Epoch 3/89
----------
train Loss: 0.0064 Acc: 0.8455
val Loss: 0.0100 Acc: 0.7299

Epoch 4/89
----------
train Loss: 0.0057 Acc: 0.8418
val Loss: 0.0088 Acc: 0.7737

Epoch 5/89
----------
train Loss: 0.0049 Acc: 0.8618
val Loss: 0.0093 Acc: 0.8029

Epoch 6/89
----------
train Loss: 0.0043 Acc: 0.8764
val Loss: 0.0073 Acc: 0.8321

Epoch 7/89
----------
train Loss: 0.0039 Acc: 0.8982
val Loss: 0.0082 Acc: 0.8321

Epoch 8/89
----------
train Loss: 0.0036 Acc: 0.9036
val Loss: 0.0074 Acc: 0.8321

Epoch 9/89
----------
train Loss: 0.0033 Acc: 0.9309
val Loss: 0.0060 Acc: 0.8467

Epoch 10/89
----------
train Loss: 0.0032 Acc: 0.9200
val Loss: 0.0083 Acc: 0.8175

Epoch 11/89
----------
LR is set to 0.001
train Loss: 0.0030 Acc: 0.9400
val Loss: 0.0078 Acc: 0.8394

Epoch 12/89
----------
train Loss: 0.0029 Acc: 0.9345
val Loss: 0.0074 Acc: 0.8321

Epoch 13/89
----------
train Loss: 0.0028 Acc: 0.9382
val Loss: 0.0059 Acc: 0.8467

Epoch 14/89
----------
train Loss: 0.0026 Acc: 0.9455
val Loss: 0.0067 Acc: 0.8467

Epoch 15/89
----------
train Loss: 0.0028 Acc: 0.9473
val Loss: 0.0069 Acc: 0.8394

Epoch 16/89
----------
train Loss: 0.0027 Acc: 0.9491
val Loss: 0.0069 Acc: 0.8394

Epoch 17/89
----------
train Loss: 0.0025 Acc: 0.9509
val Loss: 0.0080 Acc: 0.8467

Epoch 18/89
----------
train Loss: 0.0024 Acc: 0.9527
val Loss: 0.0069 Acc: 0.8321

Epoch 19/89
----------
train Loss: 0.0027 Acc: 0.9364
val Loss: 0.0070 Acc: 0.8321

Epoch 20/89
----------
train Loss: 0.0024 Acc: 0.9545
val Loss: 0.0069 Acc: 0.8248

Epoch 21/89
----------
train Loss: 0.0027 Acc: 0.9400
val Loss: 0.0073 Acc: 0.8467

Epoch 22/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0028 Acc: 0.9418
val Loss: 0.0070 Acc: 0.8467

Epoch 23/89
----------
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0079 Acc: 0.8467

Epoch 24/89
----------
train Loss: 0.0026 Acc: 0.9455
val Loss: 0.0083 Acc: 0.8394

Epoch 25/89
----------
train Loss: 0.0027 Acc: 0.9400
val Loss: 0.0073 Acc: 0.8321

Epoch 26/89
----------
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0078 Acc: 0.8321

Epoch 27/89
----------
train Loss: 0.0028 Acc: 0.9509
val Loss: 0.0073 Acc: 0.8467

Epoch 28/89
----------
train Loss: 0.0025 Acc: 0.9364
val Loss: 0.0071 Acc: 0.8467

Epoch 29/89
----------
train Loss: 0.0024 Acc: 0.9527
val Loss: 0.0058 Acc: 0.8467

Epoch 30/89
----------
train Loss: 0.0029 Acc: 0.9345
val Loss: 0.0065 Acc: 0.8394

Epoch 31/89
----------
train Loss: 0.0024 Acc: 0.9509
val Loss: 0.0063 Acc: 0.8467

Epoch 32/89
----------
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0067 Acc: 0.8467

Epoch 33/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0024 Acc: 0.9491
val Loss: 0.0074 Acc: 0.8467

Epoch 34/89
----------
train Loss: 0.0027 Acc: 0.9455
val Loss: 0.0067 Acc: 0.8394

Epoch 35/89
----------
train Loss: 0.0029 Acc: 0.9436
val Loss: 0.0072 Acc: 0.8248

Epoch 36/89
----------
train Loss: 0.0024 Acc: 0.9491
val Loss: 0.0070 Acc: 0.8394

Epoch 37/89
----------
train Loss: 0.0027 Acc: 0.9400
val Loss: 0.0067 Acc: 0.8394

Epoch 38/89
----------
train Loss: 0.0030 Acc: 0.9309
val Loss: 0.0081 Acc: 0.8394

Epoch 39/89
----------
train Loss: 0.0026 Acc: 0.9491
val Loss: 0.0071 Acc: 0.8394

Epoch 40/89
----------
train Loss: 0.0028 Acc: 0.9364
val Loss: 0.0078 Acc: 0.8467

Epoch 41/89
----------
train Loss: 0.0028 Acc: 0.9291
val Loss: 0.0064 Acc: 0.8467

Epoch 42/89
----------
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0064 Acc: 0.8467

Epoch 43/89
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0085 Acc: 0.8467

Epoch 44/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0024 Acc: 0.9655
val Loss: 0.0070 Acc: 0.8394

Epoch 45/89
----------
train Loss: 0.0025 Acc: 0.9491
val Loss: 0.0078 Acc: 0.8467

Epoch 46/89
----------
train Loss: 0.0026 Acc: 0.9618
val Loss: 0.0075 Acc: 0.8394

Epoch 47/89
----------
train Loss: 0.0024 Acc: 0.9582
val Loss: 0.0060 Acc: 0.8467

Epoch 48/89
----------
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0077 Acc: 0.8321

Epoch 49/89
----------
train Loss: 0.0026 Acc: 0.9582
val Loss: 0.0066 Acc: 0.8321

Epoch 50/89
----------
train Loss: 0.0028 Acc: 0.9400
val Loss: 0.0071 Acc: 0.8467

Epoch 51/89
----------
train Loss: 0.0026 Acc: 0.9600
val Loss: 0.0059 Acc: 0.8467

Epoch 52/89
----------
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0069 Acc: 0.8467

Epoch 53/89
----------
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0073 Acc: 0.8467

Epoch 54/89
----------
train Loss: 0.0023 Acc: 0.9618
val Loss: 0.0068 Acc: 0.8467

Epoch 55/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9291
val Loss: 0.0078 Acc: 0.8467

Epoch 56/89
----------
train Loss: 0.0024 Acc: 0.9473
val Loss: 0.0075 Acc: 0.8467

Epoch 57/89
----------
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0076 Acc: 0.8467

Epoch 58/89
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0074 Acc: 0.8467

Epoch 59/89
----------
train Loss: 0.0026 Acc: 0.9473
val Loss: 0.0072 Acc: 0.8394

Epoch 60/89
----------
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0069 Acc: 0.8394

Epoch 61/89
----------
train Loss: 0.0026 Acc: 0.9491
val Loss: 0.0080 Acc: 0.8467

Epoch 62/89
----------
train Loss: 0.0023 Acc: 0.9545
val Loss: 0.0066 Acc: 0.8248

Epoch 63/89
----------
train Loss: 0.0025 Acc: 0.9582
val Loss: 0.0061 Acc: 0.8321

Epoch 64/89
----------
train Loss: 0.0026 Acc: 0.9436
val Loss: 0.0063 Acc: 0.8467

Epoch 65/89
----------
train Loss: 0.0026 Acc: 0.9473
val Loss: 0.0071 Acc: 0.8394

Epoch 66/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0073 Acc: 0.8321

Epoch 67/89
----------
train Loss: 0.0023 Acc: 0.9473
val Loss: 0.0065 Acc: 0.8394

Epoch 68/89
----------
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0072 Acc: 0.8321

Epoch 69/89
----------
train Loss: 0.0025 Acc: 0.9545
val Loss: 0.0066 Acc: 0.8394

Epoch 70/89
----------
train Loss: 0.0026 Acc: 0.9327
val Loss: 0.0072 Acc: 0.8321

Epoch 71/89
----------
train Loss: 0.0025 Acc: 0.9618
val Loss: 0.0078 Acc: 0.8394

Epoch 72/89
----------
train Loss: 0.0025 Acc: 0.9436
val Loss: 0.0069 Acc: 0.8394

Epoch 73/89
----------
train Loss: 0.0025 Acc: 0.9600
val Loss: 0.0078 Acc: 0.8467

Epoch 74/89
----------
train Loss: 0.0025 Acc: 0.9491
val Loss: 0.0073 Acc: 0.8467

Epoch 75/89
----------
train Loss: 0.0028 Acc: 0.9327
val Loss: 0.0066 Acc: 0.8467

Epoch 76/89
----------
train Loss: 0.0025 Acc: 0.9509
val Loss: 0.0076 Acc: 0.8467

Epoch 77/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0026 Acc: 0.9473
val Loss: 0.0075 Acc: 0.8467

Epoch 78/89
----------
train Loss: 0.0027 Acc: 0.9545
val Loss: 0.0067 Acc: 0.8467

Epoch 79/89
----------
train Loss: 0.0025 Acc: 0.9527
val Loss: 0.0070 Acc: 0.8467

Epoch 80/89
----------
train Loss: 0.0025 Acc: 0.9655
val Loss: 0.0062 Acc: 0.8394

Epoch 81/89
----------
train Loss: 0.0025 Acc: 0.9473
val Loss: 0.0070 Acc: 0.8394

Epoch 82/89
----------
train Loss: 0.0025 Acc: 0.9527
val Loss: 0.0064 Acc: 0.8394

Epoch 83/89
----------
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0074 Acc: 0.8467

Epoch 84/89
----------
train Loss: 0.0024 Acc: 0.9509
val Loss: 0.0071 Acc: 0.8394

Epoch 85/89
----------
train Loss: 0.0027 Acc: 0.9345
val Loss: 0.0071 Acc: 0.8394

Epoch 86/89
----------
train Loss: 0.0025 Acc: 0.9491
val Loss: 0.0069 Acc: 0.8248

Epoch 87/89
----------
train Loss: 0.0026 Acc: 0.9473
val Loss: 0.0066 Acc: 0.8321

Epoch 88/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0025 Acc: 0.9527
val Loss: 0.0070 Acc: 0.8540

Epoch 89/89
----------
train Loss: 0.0023 Acc: 0.9509
val Loss: 0.0067 Acc: 0.8394

Training complete in 4m 46s
Best val Acc: 0.854015

---Fine tuning.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0025 Acc: 0.9382
val Loss: 0.0072 Acc: 0.8613

Epoch 1/89
----------
train Loss: 0.0015 Acc: 0.9818
val Loss: 0.0065 Acc: 0.8759

Epoch 2/89
----------
train Loss: 0.0008 Acc: 0.9873
val Loss: 0.0077 Acc: 0.8540

Epoch 3/89
----------
train Loss: 0.0007 Acc: 0.9873
val Loss: 0.0055 Acc: 0.8686

Epoch 4/89
----------
train Loss: 0.0003 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8759

Epoch 5/89
----------
train Loss: 0.0003 Acc: 0.9945
val Loss: 0.0057 Acc: 0.8832

Epoch 6/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8759

Epoch 7/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 8/89
----------
train Loss: 0.0001 Acc: 0.9945
val Loss: 0.0051 Acc: 0.8686

Epoch 9/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 10/89
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0047 Acc: 0.8759

Epoch 11/89
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 12/89
----------
train Loss: 0.0001 Acc: 0.9964
val Loss: 0.0062 Acc: 0.8759

Epoch 13/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8686

Epoch 14/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0054 Acc: 0.8613

Epoch 15/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8613

Epoch 16/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8613

Epoch 17/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8613

Epoch 18/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8613

Epoch 19/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 20/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 21/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8686

Epoch 22/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 23/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 24/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 25/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8759

Epoch 26/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0062 Acc: 0.8759

Epoch 27/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8759

Epoch 28/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0056 Acc: 0.8759

Epoch 29/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8686

Epoch 30/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8759

Epoch 31/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 32/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8759

Epoch 33/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 34/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 35/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8686

Epoch 36/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8686

Epoch 37/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8686

Epoch 38/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0048 Acc: 0.8686

Epoch 39/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 40/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8613

Epoch 41/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8686

Epoch 42/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8613

Epoch 43/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 44/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 45/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8686

Epoch 46/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 47/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8686

Epoch 48/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 49/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 50/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 51/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0051 Acc: 0.8686

Epoch 52/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8759

Epoch 53/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8759

Epoch 54/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.8759

Epoch 55/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 56/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8759

Epoch 57/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8759

Epoch 58/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8759

Epoch 59/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8759

Epoch 60/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 61/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8759

Epoch 62/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0046 Acc: 0.8686

Epoch 63/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8759

Epoch 64/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8686

Epoch 65/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8686

Epoch 66/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 67/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8759

Epoch 68/89
----------
train Loss: 0.0001 Acc: 0.9982
val Loss: 0.0047 Acc: 0.8759

Epoch 69/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8759

Epoch 70/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8759

Epoch 71/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 72/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 73/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 74/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8759

Epoch 75/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8759

Epoch 76/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 77/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 78/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8686

Epoch 79/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 80/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8686

Epoch 81/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8686

Epoch 82/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8686

Epoch 83/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8686

Epoch 84/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8759

Epoch 85/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8759

Epoch 86/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8686

Epoch 87/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 88/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8686

Epoch 89/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8686

Training complete in 5m 3s
Best val Acc: 0.883212

---Testing---
Test accuracy: 0.976710
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.9668617566643882, std: 0.036268955356127654
--------------------

run info[val: 0.25, epoch: 83, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0244 Acc: 0.1996
val Loss: 0.0213 Acc: 0.3567

Epoch 1/82
----------
train Loss: 0.0172 Acc: 0.4709
val Loss: 0.0132 Acc: 0.6374

Epoch 2/82
----------
train Loss: 0.0102 Acc: 0.7306
val Loss: 0.0093 Acc: 0.7135

Epoch 3/82
----------
LR is set to 0.001
train Loss: 0.0077 Acc: 0.7849
val Loss: 0.0089 Acc: 0.7427

Epoch 4/82
----------
train Loss: 0.0073 Acc: 0.8101
val Loss: 0.0085 Acc: 0.7661

Epoch 5/82
----------
train Loss: 0.0067 Acc: 0.8333
val Loss: 0.0081 Acc: 0.7602

Epoch 6/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0064 Acc: 0.8624
val Loss: 0.0084 Acc: 0.7485

Epoch 7/82
----------
train Loss: 0.0070 Acc: 0.8488
val Loss: 0.0084 Acc: 0.7544

Epoch 8/82
----------
train Loss: 0.0062 Acc: 0.8566
val Loss: 0.0081 Acc: 0.7602

Epoch 9/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0061 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7661

Epoch 10/82
----------
train Loss: 0.0065 Acc: 0.8527
val Loss: 0.0083 Acc: 0.7602

Epoch 11/82
----------
train Loss: 0.0060 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7544

Epoch 12/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0062 Acc: 0.8527
val Loss: 0.0083 Acc: 0.7544

Epoch 13/82
----------
train Loss: 0.0063 Acc: 0.8585
val Loss: 0.0082 Acc: 0.7602

Epoch 14/82
----------
train Loss: 0.0069 Acc: 0.8372
val Loss: 0.0082 Acc: 0.7544

Epoch 15/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0062 Acc: 0.8585
val Loss: 0.0082 Acc: 0.7602

Epoch 16/82
----------
train Loss: 0.0057 Acc: 0.8624
val Loss: 0.0082 Acc: 0.7602

Epoch 17/82
----------
train Loss: 0.0063 Acc: 0.8547
val Loss: 0.0082 Acc: 0.7544

Epoch 18/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0063 Acc: 0.8547
val Loss: 0.0082 Acc: 0.7602

Epoch 19/82
----------
train Loss: 0.0064 Acc: 0.8682
val Loss: 0.0082 Acc: 0.7544

Epoch 20/82
----------
train Loss: 0.0068 Acc: 0.8585
val Loss: 0.0082 Acc: 0.7544

Epoch 21/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0063 Acc: 0.8605
val Loss: 0.0083 Acc: 0.7602

Epoch 22/82
----------
train Loss: 0.0070 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7602

Epoch 23/82
----------
train Loss: 0.0061 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7485

Epoch 24/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0067 Acc: 0.8469
val Loss: 0.0084 Acc: 0.7602

Epoch 25/82
----------
train Loss: 0.0065 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7602

Epoch 26/82
----------
train Loss: 0.0061 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7602

Epoch 27/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0066 Acc: 0.8585
val Loss: 0.0083 Acc: 0.7544

Epoch 28/82
----------
train Loss: 0.0066 Acc: 0.8430
val Loss: 0.0083 Acc: 0.7544

Epoch 29/82
----------
train Loss: 0.0066 Acc: 0.8527
val Loss: 0.0084 Acc: 0.7544

Epoch 30/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0061 Acc: 0.8721
val Loss: 0.0084 Acc: 0.7602

Epoch 31/82
----------
train Loss: 0.0060 Acc: 0.8508
val Loss: 0.0084 Acc: 0.7544

Epoch 32/82
----------
train Loss: 0.0065 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7602

Epoch 33/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0063 Acc: 0.8469
val Loss: 0.0081 Acc: 0.7661

Epoch 34/82
----------
train Loss: 0.0073 Acc: 0.8585
val Loss: 0.0083 Acc: 0.7602

Epoch 35/82
----------
train Loss: 0.0060 Acc: 0.8663
val Loss: 0.0081 Acc: 0.7544

Epoch 36/82
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0060 Acc: 0.8488
val Loss: 0.0081 Acc: 0.7544

Epoch 37/82
----------
train Loss: 0.0064 Acc: 0.8488
val Loss: 0.0082 Acc: 0.7544

Epoch 38/82
----------
train Loss: 0.0065 Acc: 0.8411
val Loss: 0.0084 Acc: 0.7485

Epoch 39/82
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0059 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7544

Epoch 40/82
----------
train Loss: 0.0058 Acc: 0.8682
val Loss: 0.0084 Acc: 0.7544

Epoch 41/82
----------
train Loss: 0.0067 Acc: 0.8585
val Loss: 0.0085 Acc: 0.7544

Epoch 42/82
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0064 Acc: 0.8508
val Loss: 0.0082 Acc: 0.7602

Epoch 43/82
----------
train Loss: 0.0065 Acc: 0.8469
val Loss: 0.0082 Acc: 0.7544

Epoch 44/82
----------
train Loss: 0.0065 Acc: 0.8547
val Loss: 0.0080 Acc: 0.7602

Epoch 45/82
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0061 Acc: 0.8527
val Loss: 0.0083 Acc: 0.7602

Epoch 46/82
----------
train Loss: 0.0062 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7544

Epoch 47/82
----------
train Loss: 0.0064 Acc: 0.8663
val Loss: 0.0085 Acc: 0.7544

Epoch 48/82
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0061 Acc: 0.8585
val Loss: 0.0082 Acc: 0.7602

Epoch 49/82
----------
train Loss: 0.0061 Acc: 0.8566
val Loss: 0.0081 Acc: 0.7544

Epoch 50/82
----------
train Loss: 0.0061 Acc: 0.8585
val Loss: 0.0082 Acc: 0.7544

Epoch 51/82
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0057 Acc: 0.8566
val Loss: 0.0083 Acc: 0.7602

Epoch 52/82
----------
train Loss: 0.0061 Acc: 0.8430
val Loss: 0.0082 Acc: 0.7602

Epoch 53/82
----------
train Loss: 0.0063 Acc: 0.8721
val Loss: 0.0082 Acc: 0.7602

Epoch 54/82
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0062 Acc: 0.8643
val Loss: 0.0080 Acc: 0.7544

Epoch 55/82
----------
train Loss: 0.0063 Acc: 0.8605
val Loss: 0.0082 Acc: 0.7602

Epoch 56/82
----------
train Loss: 0.0065 Acc: 0.8469
val Loss: 0.0082 Acc: 0.7602

Epoch 57/82
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0060 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7602

Epoch 58/82
----------
train Loss: 0.0065 Acc: 0.8469
val Loss: 0.0084 Acc: 0.7602

Epoch 59/82
----------
train Loss: 0.0062 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7602

Epoch 60/82
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0060 Acc: 0.8488
val Loss: 0.0081 Acc: 0.7602

Epoch 61/82
----------
train Loss: 0.0065 Acc: 0.8450
val Loss: 0.0080 Acc: 0.7602

Epoch 62/82
----------
train Loss: 0.0061 Acc: 0.8605
val Loss: 0.0081 Acc: 0.7544

Epoch 63/82
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0066 Acc: 0.8547
val Loss: 0.0084 Acc: 0.7544

Epoch 64/82
----------
train Loss: 0.0063 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7544

Epoch 65/82
----------
train Loss: 0.0061 Acc: 0.8508
val Loss: 0.0082 Acc: 0.7602

Epoch 66/82
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0062 Acc: 0.8527
val Loss: 0.0082 Acc: 0.7602

Epoch 67/82
----------
train Loss: 0.0063 Acc: 0.8624
val Loss: 0.0083 Acc: 0.7544

Epoch 68/82
----------
train Loss: 0.0060 Acc: 0.8527
val Loss: 0.0081 Acc: 0.7544

Epoch 69/82
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0062 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7544

Epoch 70/82
----------
train Loss: 0.0062 Acc: 0.8566
val Loss: 0.0083 Acc: 0.7544

Epoch 71/82
----------
train Loss: 0.0063 Acc: 0.8624
val Loss: 0.0083 Acc: 0.7544

Epoch 72/82
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0063 Acc: 0.8411
val Loss: 0.0081 Acc: 0.7602

Epoch 73/82
----------
train Loss: 0.0061 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7602

Epoch 74/82
----------
train Loss: 0.0063 Acc: 0.8624
val Loss: 0.0084 Acc: 0.7485

Epoch 75/82
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0067 Acc: 0.8643
val Loss: 0.0082 Acc: 0.7602

Epoch 76/82
----------
train Loss: 0.0065 Acc: 0.8566
val Loss: 0.0084 Acc: 0.7602

Epoch 77/82
----------
train Loss: 0.0064 Acc: 0.8566
val Loss: 0.0082 Acc: 0.7544

Epoch 78/82
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0066 Acc: 0.8469
val Loss: 0.0082 Acc: 0.7602

Epoch 79/82
----------
train Loss: 0.0061 Acc: 0.8585
val Loss: 0.0080 Acc: 0.7602

Epoch 80/82
----------
train Loss: 0.0064 Acc: 0.8372
val Loss: 0.0082 Acc: 0.7544

Epoch 81/82
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0065 Acc: 0.8605
val Loss: 0.0082 Acc: 0.7602

Epoch 82/82
----------
train Loss: 0.0062 Acc: 0.8508
val Loss: 0.0082 Acc: 0.7485

Training complete in 4m 21s
Best val Acc: 0.766082

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0056 Acc: 0.8643
val Loss: 0.0072 Acc: 0.8187

Epoch 1/82
----------
train Loss: 0.0035 Acc: 0.9205
val Loss: 0.0054 Acc: 0.8713

Epoch 2/82
----------
train Loss: 0.0018 Acc: 0.9612
val Loss: 0.0056 Acc: 0.8596

Epoch 3/82
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9922
val Loss: 0.0045 Acc: 0.8596

Epoch 4/82
----------
train Loss: 0.0008 Acc: 0.9922
val Loss: 0.0041 Acc: 0.8655

Epoch 5/82
----------
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0038 Acc: 0.8655

Epoch 6/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9903
val Loss: 0.0037 Acc: 0.8713

Epoch 7/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 8/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 9/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8713

Epoch 10/82
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0038 Acc: 0.8713

Epoch 11/82
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8713

Epoch 12/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0036 Acc: 0.8713

Epoch 13/82
----------
train Loss: 0.0007 Acc: 0.9903
val Loss: 0.0041 Acc: 0.8655

Epoch 14/82
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8713

Epoch 15/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0036 Acc: 0.8655

Epoch 16/82
----------
train Loss: 0.0004 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8713

Epoch 17/82
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8713

Epoch 18/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0037 Acc: 0.8713

Epoch 19/82
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0038 Acc: 0.8655

Epoch 20/82
----------
train Loss: 0.0005 Acc: 0.9922
val Loss: 0.0039 Acc: 0.8655

Epoch 21/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0040 Acc: 0.8655

Epoch 22/82
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8655

Epoch 23/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8655

Epoch 24/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0041 Acc: 0.8655

Epoch 25/82
----------
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0037 Acc: 0.8655

Epoch 26/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8655

Epoch 27/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8655

Epoch 28/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 29/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0038 Acc: 0.8713

Epoch 30/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8655

Epoch 31/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8655

Epoch 32/82
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8713

Epoch 33/82
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0008 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8713

Epoch 34/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0037 Acc: 0.8713

Epoch 35/82
----------
train Loss: 0.0009 Acc: 0.9903
val Loss: 0.0037 Acc: 0.8655

Epoch 36/82
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0037 Acc: 0.8713

Epoch 37/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8655

Epoch 38/82
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0038 Acc: 0.8655

Epoch 39/82
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0006 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8713

Epoch 40/82
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0036 Acc: 0.8713

Epoch 41/82
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8713

Epoch 42/82
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8713

Epoch 43/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 44/82
----------
train Loss: 0.0010 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8713

Epoch 45/82
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0036 Acc: 0.8713

Epoch 46/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0040 Acc: 0.8713

Epoch 47/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0036 Acc: 0.8655

Epoch 48/82
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8655

Epoch 49/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 50/82
----------
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0037 Acc: 0.8713

Epoch 51/82
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8713

Epoch 52/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8713

Epoch 53/82
----------
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0037 Acc: 0.8655

Epoch 54/82
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0037 Acc: 0.8655

Epoch 55/82
----------
train Loss: 0.0006 Acc: 0.9981
val Loss: 0.0039 Acc: 0.8713

Epoch 56/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 57/82
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8713

Epoch 58/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0040 Acc: 0.8655

Epoch 59/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0040 Acc: 0.8655

Epoch 60/82
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0040 Acc: 0.8655

Epoch 61/82
----------
train Loss: 0.0006 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8713

Epoch 62/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0038 Acc: 0.8713

Epoch 63/82
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0036 Acc: 0.8713

Epoch 64/82
----------
train Loss: 0.0005 Acc: 0.9922
val Loss: 0.0038 Acc: 0.8713

Epoch 65/82
----------
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0036 Acc: 0.8713

Epoch 66/82
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0039 Acc: 0.8713

Epoch 67/82
----------
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8655

Epoch 68/82
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8655

Epoch 69/82
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0007 Acc: 0.9961
val Loss: 0.0038 Acc: 0.8713

Epoch 70/82
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0037 Acc: 0.8655

Epoch 71/82
----------
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0039 Acc: 0.8655

Epoch 72/82
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0007 Acc: 0.9942
val Loss: 0.0038 Acc: 0.8713

Epoch 73/82
----------
train Loss: 0.0006 Acc: 0.9942
val Loss: 0.0038 Acc: 0.8713

Epoch 74/82
----------
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0037 Acc: 0.8772

Epoch 75/82
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0005 Acc: 0.9942
val Loss: 0.0040 Acc: 0.8713

Epoch 76/82
----------
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0037 Acc: 0.8655

Epoch 77/82
----------
train Loss: 0.0007 Acc: 0.9922
val Loss: 0.0038 Acc: 0.8655

Epoch 78/82
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0038 Acc: 0.8655

Epoch 79/82
----------
train Loss: 0.0006 Acc: 0.9961
val Loss: 0.0041 Acc: 0.8655

Epoch 80/82
----------
train Loss: 0.0005 Acc: 0.9961
val Loss: 0.0037 Acc: 0.8655

Epoch 81/82
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0006 Acc: 0.9922
val Loss: 0.0038 Acc: 0.8655

Epoch 82/82
----------
train Loss: 0.0007 Acc: 0.9981
val Loss: 0.0037 Acc: 0.8655

Training complete in 4m 39s
Best val Acc: 0.877193

---Testing---
Test accuracy: 0.966521
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9561784360994005, std: 0.044224574323318784
--------------------

run info[val: 0.3, epoch: 74, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/73
----------
LR is set to 0.01
train Loss: 0.0225 Acc: 0.1954
val Loss: 0.0281 Acc: 0.3204

Epoch 1/73
----------
train Loss: 0.0161 Acc: 0.4948
val Loss: 0.0177 Acc: 0.6214

Epoch 2/73
----------
train Loss: 0.0112 Acc: 0.6403
val Loss: 0.0151 Acc: 0.6796

Epoch 3/73
----------
train Loss: 0.0082 Acc: 0.7651
val Loss: 0.0108 Acc: 0.7573

Epoch 4/73
----------
LR is set to 0.001
train Loss: 0.0065 Acc: 0.7817
val Loss: 0.0085 Acc: 0.7864

Epoch 5/73
----------
train Loss: 0.0063 Acc: 0.8233
val Loss: 0.0106 Acc: 0.7670

Epoch 6/73
----------
train Loss: 0.0057 Acc: 0.8649
val Loss: 0.0124 Acc: 0.8058

Epoch 7/73
----------
train Loss: 0.0057 Acc: 0.8441
val Loss: 0.0085 Acc: 0.8107

Epoch 8/73
----------
LR is set to 0.00010000000000000002
train Loss: 0.0057 Acc: 0.8586
val Loss: 0.0090 Acc: 0.8010

Epoch 9/73
----------
train Loss: 0.0057 Acc: 0.8628
val Loss: 0.0086 Acc: 0.7961

Epoch 10/73
----------
train Loss: 0.0059 Acc: 0.8669
val Loss: 0.0080 Acc: 0.7961

Epoch 11/73
----------
train Loss: 0.0057 Acc: 0.8586
val Loss: 0.0101 Acc: 0.7913

Epoch 12/73
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0056 Acc: 0.8482
val Loss: 0.0092 Acc: 0.7913

Epoch 13/73
----------
train Loss: 0.0057 Acc: 0.8628
val Loss: 0.0081 Acc: 0.7913

Epoch 14/73
----------
train Loss: 0.0056 Acc: 0.8524
val Loss: 0.0083 Acc: 0.7864

Epoch 15/73
----------
train Loss: 0.0056 Acc: 0.8545
val Loss: 0.0084 Acc: 0.7864

Epoch 16/73
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0057 Acc: 0.8441
val Loss: 0.0102 Acc: 0.7913

Epoch 17/73
----------
train Loss: 0.0056 Acc: 0.8628
val Loss: 0.0081 Acc: 0.7864

Epoch 18/73
----------
train Loss: 0.0056 Acc: 0.8690
val Loss: 0.0099 Acc: 0.7864

Epoch 19/73
----------
train Loss: 0.0054 Acc: 0.8628
val Loss: 0.0113 Acc: 0.7816

Epoch 20/73
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0056 Acc: 0.8628
val Loss: 0.0099 Acc: 0.7816

Epoch 21/73
----------
train Loss: 0.0054 Acc: 0.8815
val Loss: 0.0076 Acc: 0.7864

Epoch 22/73
----------
train Loss: 0.0054 Acc: 0.8711
val Loss: 0.0115 Acc: 0.7864

Epoch 23/73
----------
train Loss: 0.0055 Acc: 0.8711
val Loss: 0.0079 Acc: 0.7864

Epoch 24/73
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0056 Acc: 0.8545
val Loss: 0.0094 Acc: 0.7816

Epoch 25/73
----------
train Loss: 0.0055 Acc: 0.8836
val Loss: 0.0106 Acc: 0.7864

Epoch 26/73
----------
train Loss: 0.0054 Acc: 0.8607
val Loss: 0.0088 Acc: 0.7816

Epoch 27/73
----------
train Loss: 0.0056 Acc: 0.8669
val Loss: 0.0091 Acc: 0.7816

Epoch 28/73
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0057 Acc: 0.8545
val Loss: 0.0090 Acc: 0.7864

Epoch 29/73
----------
train Loss: 0.0055 Acc: 0.8815
val Loss: 0.0085 Acc: 0.7864

Epoch 30/73
----------
train Loss: 0.0057 Acc: 0.8565
val Loss: 0.0089 Acc: 0.7913

Epoch 31/73
----------
train Loss: 0.0054 Acc: 0.8711
val Loss: 0.0123 Acc: 0.7913

Epoch 32/73
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0056 Acc: 0.8690
val Loss: 0.0095 Acc: 0.7913

Epoch 33/73
----------
train Loss: 0.0056 Acc: 0.8690
val Loss: 0.0102 Acc: 0.7913

Epoch 34/73
----------
train Loss: 0.0055 Acc: 0.8628
val Loss: 0.0107 Acc: 0.7913

Epoch 35/73
----------
train Loss: 0.0054 Acc: 0.8836
val Loss: 0.0075 Acc: 0.7864

Epoch 36/73
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0054 Acc: 0.8690
val Loss: 0.0106 Acc: 0.7864

Epoch 37/73
----------
train Loss: 0.0056 Acc: 0.8586
val Loss: 0.0071 Acc: 0.7864

Epoch 38/73
----------
train Loss: 0.0056 Acc: 0.8628
val Loss: 0.0094 Acc: 0.7864

Epoch 39/73
----------
train Loss: 0.0058 Acc: 0.8545
val Loss: 0.0073 Acc: 0.7913

Epoch 40/73
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0058 Acc: 0.8462
val Loss: 0.0090 Acc: 0.7913

Epoch 41/73
----------
train Loss: 0.0058 Acc: 0.8295
val Loss: 0.0114 Acc: 0.7913

Epoch 42/73
----------
train Loss: 0.0056 Acc: 0.8586
val Loss: 0.0093 Acc: 0.7864

Epoch 43/73
----------
train Loss: 0.0055 Acc: 0.8794
val Loss: 0.0086 Acc: 0.7864

Epoch 44/73
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0057 Acc: 0.8545
val Loss: 0.0101 Acc: 0.7864

Epoch 45/73
----------
train Loss: 0.0055 Acc: 0.8669
val Loss: 0.0076 Acc: 0.7864

Epoch 46/73
----------
train Loss: 0.0055 Acc: 0.8773
val Loss: 0.0088 Acc: 0.7864

Epoch 47/73
----------
train Loss: 0.0054 Acc: 0.8690
val Loss: 0.0101 Acc: 0.7913

Epoch 48/73
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0053 Acc: 0.8836
val Loss: 0.0096 Acc: 0.7913

Epoch 49/73
----------
train Loss: 0.0054 Acc: 0.8628
val Loss: 0.0097 Acc: 0.7913

Epoch 50/73
----------
train Loss: 0.0056 Acc: 0.8586
val Loss: 0.0073 Acc: 0.7864

Epoch 51/73
----------
train Loss: 0.0056 Acc: 0.8669
val Loss: 0.0082 Acc: 0.7864

Epoch 52/73
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0054 Acc: 0.8628
val Loss: 0.0128 Acc: 0.7864

Epoch 53/73
----------
train Loss: 0.0054 Acc: 0.8690
val Loss: 0.0106 Acc: 0.7913

Epoch 54/73
----------
train Loss: 0.0056 Acc: 0.8628
val Loss: 0.0099 Acc: 0.7913

Epoch 55/73
----------
train Loss: 0.0056 Acc: 0.8441
val Loss: 0.0091 Acc: 0.7913

Epoch 56/73
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0055 Acc: 0.8545
val Loss: 0.0125 Acc: 0.7913

Epoch 57/73
----------
train Loss: 0.0055 Acc: 0.8503
val Loss: 0.0093 Acc: 0.7913

Epoch 58/73
----------
train Loss: 0.0055 Acc: 0.8753
val Loss: 0.0087 Acc: 0.7913

Epoch 59/73
----------
train Loss: 0.0055 Acc: 0.8836
val Loss: 0.0096 Acc: 0.7864

Epoch 60/73
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0058 Acc: 0.8441
val Loss: 0.0100 Acc: 0.7913

Epoch 61/73
----------
train Loss: 0.0055 Acc: 0.8690
val Loss: 0.0104 Acc: 0.7913

Epoch 62/73
----------
train Loss: 0.0053 Acc: 0.8690
val Loss: 0.0090 Acc: 0.7961

Epoch 63/73
----------
train Loss: 0.0054 Acc: 0.8628
val Loss: 0.0088 Acc: 0.7864

Epoch 64/73
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0055 Acc: 0.8732
val Loss: 0.0082 Acc: 0.7961

Epoch 65/73
----------
train Loss: 0.0056 Acc: 0.8586
val Loss: 0.0083 Acc: 0.7913

Epoch 66/73
----------
train Loss: 0.0054 Acc: 0.8628
val Loss: 0.0096 Acc: 0.7864

Epoch 67/73
----------
train Loss: 0.0055 Acc: 0.8690
val Loss: 0.0119 Acc: 0.7913

Epoch 68/73
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0055 Acc: 0.8586
val Loss: 0.0115 Acc: 0.7913

Epoch 69/73
----------
train Loss: 0.0054 Acc: 0.8732
val Loss: 0.0078 Acc: 0.7913

Epoch 70/73
----------
train Loss: 0.0056 Acc: 0.8649
val Loss: 0.0098 Acc: 0.7913

Epoch 71/73
----------
train Loss: 0.0055 Acc: 0.8690
val Loss: 0.0080 Acc: 0.7864

Epoch 72/73
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0055 Acc: 0.8607
val Loss: 0.0077 Acc: 0.7864

Epoch 73/73
----------
train Loss: 0.0056 Acc: 0.8628
val Loss: 0.0087 Acc: 0.7864

Training complete in 3m 49s
Best val Acc: 0.810680

---Fine tuning.---
Epoch 0/73
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8565
val Loss: 0.0100 Acc: 0.7913

Epoch 1/73
----------
train Loss: 0.0034 Acc: 0.9252
val Loss: 0.0060 Acc: 0.8155

Epoch 2/73
----------
train Loss: 0.0020 Acc: 0.9605
val Loss: 0.0062 Acc: 0.8592

Epoch 3/73
----------
train Loss: 0.0011 Acc: 0.9792
val Loss: 0.0044 Acc: 0.8738

Epoch 4/73
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9896
val Loss: 0.0041 Acc: 0.8932

Epoch 5/73
----------
train Loss: 0.0008 Acc: 0.9896
val Loss: 0.0038 Acc: 0.9029

Epoch 6/73
----------
train Loss: 0.0007 Acc: 0.9896
val Loss: 0.0060 Acc: 0.9029

Epoch 7/73
----------
train Loss: 0.0006 Acc: 0.9896
val Loss: 0.0034 Acc: 0.8932

Epoch 8/73
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0037 Acc: 0.8883

Epoch 9/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0035 Acc: 0.8932

Epoch 10/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0034 Acc: 0.8932

Epoch 11/73
----------
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0045 Acc: 0.8932

Epoch 12/73
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8932

Epoch 13/73
----------
train Loss: 0.0007 Acc: 0.9896
val Loss: 0.0034 Acc: 0.8932

Epoch 14/73
----------
train Loss: 0.0007 Acc: 0.9896
val Loss: 0.0072 Acc: 0.8981

Epoch 15/73
----------
train Loss: 0.0006 Acc: 0.9979
val Loss: 0.0038 Acc: 0.8981

Epoch 16/73
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0071 Acc: 0.8981

Epoch 17/73
----------
train Loss: 0.0006 Acc: 0.9979
val Loss: 0.0043 Acc: 0.8981

Epoch 18/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0034 Acc: 0.8981

Epoch 19/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0034 Acc: 0.8981

Epoch 20/73
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0037 Acc: 0.8981

Epoch 21/73
----------
train Loss: 0.0006 Acc: 0.9875
val Loss: 0.0042 Acc: 0.8981

Epoch 22/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0046 Acc: 0.8981

Epoch 23/73
----------
train Loss: 0.0007 Acc: 0.9896
val Loss: 0.0055 Acc: 0.8981

Epoch 24/73
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9979
val Loss: 0.0037 Acc: 0.8981

Epoch 25/73
----------
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0052 Acc: 0.8981

Epoch 26/73
----------
train Loss: 0.0005 Acc: 0.9979
val Loss: 0.0036 Acc: 0.8981

Epoch 27/73
----------
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0069 Acc: 0.8981

Epoch 28/73
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0035 Acc: 0.8981

Epoch 29/73
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0062 Acc: 0.8981

Epoch 30/73
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0034 Acc: 0.8981

Epoch 31/73
----------
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0103 Acc: 0.8981

Epoch 32/73
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0035 Acc: 0.8981

Epoch 33/73
----------
train Loss: 0.0006 Acc: 0.9875
val Loss: 0.0060 Acc: 0.8981

Epoch 34/73
----------
train Loss: 0.0006 Acc: 0.9979
val Loss: 0.0059 Acc: 0.8981

Epoch 35/73
----------
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0057 Acc: 0.8981

Epoch 36/73
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0034 Acc: 0.8981

Epoch 37/73
----------
train Loss: 0.0006 Acc: 0.9854
val Loss: 0.0045 Acc: 0.8981

Epoch 38/73
----------
train Loss: 0.0006 Acc: 0.9875
val Loss: 0.0044 Acc: 0.8981

Epoch 39/73
----------
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0034 Acc: 0.8981

Epoch 40/73
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0071 Acc: 0.8932

Epoch 41/73
----------
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0134 Acc: 0.8981

Epoch 42/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0058 Acc: 0.8981

Epoch 43/73
----------
train Loss: 0.0007 Acc: 0.9896
val Loss: 0.0151 Acc: 0.8981

Epoch 44/73
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0048 Acc: 0.8932

Epoch 45/73
----------
train Loss: 0.0007 Acc: 0.9917
val Loss: 0.0036 Acc: 0.8981

Epoch 46/73
----------
train Loss: 0.0006 Acc: 0.9979
val Loss: 0.0048 Acc: 0.8981

Epoch 47/73
----------
train Loss: 0.0007 Acc: 0.9854
val Loss: 0.0035 Acc: 0.8981

Epoch 48/73
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0042 Acc: 0.8932

Epoch 49/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0041 Acc: 0.8932

Epoch 50/73
----------
train Loss: 0.0007 Acc: 0.9917
val Loss: 0.0046 Acc: 0.8981

Epoch 51/73
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0035 Acc: 0.8981

Epoch 52/73
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0055 Acc: 0.8932

Epoch 53/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0039 Acc: 0.8932

Epoch 54/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0076 Acc: 0.8932

Epoch 55/73
----------
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0035 Acc: 0.8981

Epoch 56/73
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0006 Acc: 0.9958
val Loss: 0.0046 Acc: 0.8981

Epoch 57/73
----------
train Loss: 0.0006 Acc: 0.9896
val Loss: 0.0035 Acc: 0.8981

Epoch 58/73
----------
train Loss: 0.0006 Acc: 0.9896
val Loss: 0.0065 Acc: 0.8981

Epoch 59/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0039 Acc: 0.8981

Epoch 60/73
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0007 Acc: 0.9854
val Loss: 0.0046 Acc: 0.8981

Epoch 61/73
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0036 Acc: 0.8981

Epoch 62/73
----------
train Loss: 0.0006 Acc: 0.9875
val Loss: 0.0104 Acc: 0.8981

Epoch 63/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0054 Acc: 0.8981

Epoch 64/73
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0078 Acc: 0.8981

Epoch 65/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0048 Acc: 0.8981

Epoch 66/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0037 Acc: 0.8981

Epoch 67/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0051 Acc: 0.8981

Epoch 68/73
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0007 Acc: 0.9854
val Loss: 0.0044 Acc: 0.8981

Epoch 69/73
----------
train Loss: 0.0007 Acc: 0.9875
val Loss: 0.0054 Acc: 0.8981

Epoch 70/73
----------
train Loss: 0.0006 Acc: 0.9917
val Loss: 0.0046 Acc: 0.8932

Epoch 71/73
----------
train Loss: 0.0005 Acc: 0.9958
val Loss: 0.0034 Acc: 0.8981

Epoch 72/73
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0006 Acc: 0.9875
val Loss: 0.0040 Acc: 0.8981

Epoch 73/73
----------
train Loss: 0.0006 Acc: 0.9938
val Loss: 0.0088 Acc: 0.8981

Training complete in 4m 0s
Best val Acc: 0.902913

---Testing---
Test accuracy: 0.965066
--------------------
Accuracy of Carcharhiniformes : 100 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 82 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 96 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 98 %
mean: 0.9556336329689386, std: 0.054108309224115976

Model saved in "./weights/shark_[0.99]_mean[0.98]_std[0.02].save".
--------------------

run info[val: 0.1, epoch: 70, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0213 Acc: 0.2488
val Loss: 0.0222 Acc: 0.5441

Epoch 1/69
----------
train Loss: 0.0136 Acc: 0.6204
val Loss: 0.0152 Acc: 0.6324

Epoch 2/69
----------
train Loss: 0.0092 Acc: 0.7674
val Loss: 0.0126 Acc: 0.7500

Epoch 3/69
----------
train Loss: 0.0066 Acc: 0.8110
val Loss: 0.0101 Acc: 0.7500

Epoch 4/69
----------
train Loss: 0.0057 Acc: 0.8191
val Loss: 0.0099 Acc: 0.8088

Epoch 5/69
----------
train Loss: 0.0046 Acc: 0.8772
val Loss: 0.0087 Acc: 0.7941

Epoch 6/69
----------
train Loss: 0.0042 Acc: 0.8934
val Loss: 0.0102 Acc: 0.7500

Epoch 7/69
----------
train Loss: 0.0033 Acc: 0.9208
val Loss: 0.0093 Acc: 0.7647

Epoch 8/69
----------
LR is set to 0.001
train Loss: 0.0037 Acc: 0.9128
val Loss: 0.0093 Acc: 0.7500

Epoch 9/69
----------
train Loss: 0.0029 Acc: 0.9370
val Loss: 0.0094 Acc: 0.7647

Epoch 10/69
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0093 Acc: 0.7500

Epoch 11/69
----------
train Loss: 0.0030 Acc: 0.9418
val Loss: 0.0096 Acc: 0.7500

Epoch 12/69
----------
train Loss: 0.0037 Acc: 0.9208
val Loss: 0.0094 Acc: 0.7500

Epoch 13/69
----------
train Loss: 0.0034 Acc: 0.9370
val Loss: 0.0094 Acc: 0.7647

Epoch 14/69
----------
train Loss: 0.0033 Acc: 0.9160
val Loss: 0.0092 Acc: 0.7500

Epoch 15/69
----------
train Loss: 0.0031 Acc: 0.9241
val Loss: 0.0093 Acc: 0.7500

Epoch 16/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0032 Acc: 0.9241
val Loss: 0.0094 Acc: 0.7500

Epoch 17/69
----------
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0094 Acc: 0.7500

Epoch 18/69
----------
train Loss: 0.0032 Acc: 0.9354
val Loss: 0.0092 Acc: 0.7500

Epoch 19/69
----------
train Loss: 0.0031 Acc: 0.9225
val Loss: 0.0094 Acc: 0.7500

Epoch 20/69
----------
train Loss: 0.0032 Acc: 0.9321
val Loss: 0.0094 Acc: 0.7500

Epoch 21/69
----------
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0094 Acc: 0.7500

Epoch 22/69
----------
train Loss: 0.0034 Acc: 0.9289
val Loss: 0.0092 Acc: 0.7500

Epoch 23/69
----------
train Loss: 0.0034 Acc: 0.9225
val Loss: 0.0092 Acc: 0.7500

Epoch 24/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0031 Acc: 0.9435
val Loss: 0.0094 Acc: 0.7500

Epoch 25/69
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0094 Acc: 0.7500

Epoch 26/69
----------
train Loss: 0.0034 Acc: 0.9208
val Loss: 0.0094 Acc: 0.7500

Epoch 27/69
----------
train Loss: 0.0032 Acc: 0.9338
val Loss: 0.0094 Acc: 0.7500

Epoch 28/69
----------
train Loss: 0.0036 Acc: 0.9111
val Loss: 0.0092 Acc: 0.7500

Epoch 29/69
----------
train Loss: 0.0030 Acc: 0.9338
val Loss: 0.0092 Acc: 0.7500

Epoch 30/69
----------
train Loss: 0.0030 Acc: 0.9241
val Loss: 0.0092 Acc: 0.7500

Epoch 31/69
----------
train Loss: 0.0029 Acc: 0.9435
val Loss: 0.0093 Acc: 0.7500

Epoch 32/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0031 Acc: 0.9208
val Loss: 0.0094 Acc: 0.7500

Epoch 33/69
----------
train Loss: 0.0032 Acc: 0.9241
val Loss: 0.0093 Acc: 0.7500

Epoch 34/69
----------
train Loss: 0.0029 Acc: 0.9257
val Loss: 0.0093 Acc: 0.7500

Epoch 35/69
----------
train Loss: 0.0033 Acc: 0.9289
val Loss: 0.0093 Acc: 0.7500

Epoch 36/69
----------
train Loss: 0.0030 Acc: 0.9225
val Loss: 0.0093 Acc: 0.7500

Epoch 37/69
----------
train Loss: 0.0032 Acc: 0.9370
val Loss: 0.0093 Acc: 0.7500

Epoch 38/69
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0091 Acc: 0.7500

Epoch 39/69
----------
train Loss: 0.0031 Acc: 0.9370
val Loss: 0.0092 Acc: 0.7500

Epoch 40/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0033 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7500

Epoch 41/69
----------
train Loss: 0.0033 Acc: 0.9225
val Loss: 0.0093 Acc: 0.7500

Epoch 42/69
----------
train Loss: 0.0030 Acc: 0.9370
val Loss: 0.0093 Acc: 0.7500

Epoch 43/69
----------
train Loss: 0.0030 Acc: 0.9192
val Loss: 0.0092 Acc: 0.7500

Epoch 44/69
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0092 Acc: 0.7500

Epoch 45/69
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0093 Acc: 0.7500

Epoch 46/69
----------
train Loss: 0.0031 Acc: 0.9321
val Loss: 0.0093 Acc: 0.7500

Epoch 47/69
----------
train Loss: 0.0030 Acc: 0.9338
val Loss: 0.0093 Acc: 0.7500

Epoch 48/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0034 Acc: 0.9321
val Loss: 0.0093 Acc: 0.7500

Epoch 49/69
----------
train Loss: 0.0034 Acc: 0.9111
val Loss: 0.0094 Acc: 0.7500

Epoch 50/69
----------
train Loss: 0.0033 Acc: 0.9160
val Loss: 0.0093 Acc: 0.7500

Epoch 51/69
----------
train Loss: 0.0028 Acc: 0.9338
val Loss: 0.0093 Acc: 0.7500

Epoch 52/69
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0093 Acc: 0.7500

Epoch 53/69
----------
train Loss: 0.0031 Acc: 0.9144
val Loss: 0.0093 Acc: 0.7500

Epoch 54/69
----------
train Loss: 0.0032 Acc: 0.9321
val Loss: 0.0092 Acc: 0.7500

Epoch 55/69
----------
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0092 Acc: 0.7500

Epoch 56/69
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0091 Acc: 0.7500

Epoch 57/69
----------
train Loss: 0.0033 Acc: 0.9370
val Loss: 0.0091 Acc: 0.7500

Epoch 58/69
----------
train Loss: 0.0033 Acc: 0.9192
val Loss: 0.0091 Acc: 0.7500

Epoch 59/69
----------
train Loss: 0.0029 Acc: 0.9386
val Loss: 0.0092 Acc: 0.7500

Epoch 60/69
----------
train Loss: 0.0033 Acc: 0.9321
val Loss: 0.0092 Acc: 0.7500

Epoch 61/69
----------
train Loss: 0.0029 Acc: 0.9386
val Loss: 0.0092 Acc: 0.7500

Epoch 62/69
----------
train Loss: 0.0031 Acc: 0.9386
val Loss: 0.0093 Acc: 0.7500

Epoch 63/69
----------
train Loss: 0.0034 Acc: 0.9176
val Loss: 0.0092 Acc: 0.7500

Epoch 64/69
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0036 Acc: 0.9128
val Loss: 0.0092 Acc: 0.7500

Epoch 65/69
----------
train Loss: 0.0030 Acc: 0.9289
val Loss: 0.0092 Acc: 0.7500

Epoch 66/69
----------
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0093 Acc: 0.7500

Epoch 67/69
----------
train Loss: 0.0032 Acc: 0.9289
val Loss: 0.0095 Acc: 0.7500

Epoch 68/69
----------
train Loss: 0.0036 Acc: 0.9241
val Loss: 0.0095 Acc: 0.7500

Epoch 69/69
----------
train Loss: 0.0031 Acc: 0.9321
val Loss: 0.0092 Acc: 0.7500

Training complete in 3m 39s
Best val Acc: 0.808824

---Fine tuning.---
Epoch 0/69
----------
LR is set to 0.01
train Loss: 0.0044 Acc: 0.8788
val Loss: 0.0096 Acc: 0.7794

Epoch 1/69
----------
train Loss: 0.0024 Acc: 0.9451
val Loss: 0.0094 Acc: 0.7794

Epoch 2/69
----------
train Loss: 0.0014 Acc: 0.9806
val Loss: 0.0079 Acc: 0.8088

Epoch 3/69
----------
train Loss: 0.0007 Acc: 0.9903
val Loss: 0.0071 Acc: 0.7941

Epoch 4/69
----------
train Loss: 0.0005 Acc: 0.9935
val Loss: 0.0071 Acc: 0.8235

Epoch 5/69
----------
train Loss: 0.0003 Acc: 0.9952
val Loss: 0.0073 Acc: 0.8382

Epoch 6/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8382

Epoch 7/69
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8235

Epoch 8/69
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8235

Epoch 9/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8382

Epoch 10/69
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8382

Epoch 11/69
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8235

Epoch 12/69
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8382

Epoch 13/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 14/69
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8088

Epoch 15/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.7941

Epoch 16/69
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 17/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 18/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 19/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8235

Epoch 20/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 21/69
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0076 Acc: 0.7941

Epoch 22/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 23/69
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0079 Acc: 0.8088

Epoch 24/69
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8088

Epoch 25/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 26/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8088

Epoch 27/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 28/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 29/69
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0074 Acc: 0.8088

Epoch 30/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8088

Epoch 31/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0078 Acc: 0.7941

Epoch 32/69
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.7941

Epoch 33/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 34/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.7941

Epoch 35/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.7941

Epoch 36/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8088

Epoch 37/69
----------
train Loss: 0.0001 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8088

Epoch 38/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 39/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 40/69
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 41/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8088

Epoch 42/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.7941

Epoch 43/69
----------
train Loss: 0.0001 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8088

Epoch 44/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 45/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 46/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0081 Acc: 0.7941

Epoch 47/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.7941

Epoch 48/69
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.7941

Epoch 49/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 50/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 51/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0076 Acc: 0.8088

Epoch 52/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0078 Acc: 0.7941

Epoch 53/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.7941

Epoch 54/69
----------
train Loss: 0.0006 Acc: 0.9952
val Loss: 0.0076 Acc: 0.8088

Epoch 55/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 56/69
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8088

Epoch 57/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.7941

Epoch 58/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.7941

Epoch 59/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8088

Epoch 60/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0077 Acc: 0.7941

Epoch 61/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8088

Epoch 62/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8235

Epoch 63/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0075 Acc: 0.8235

Epoch 64/69
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.7941

Epoch 65/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 66/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Epoch 67/69
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.7941

Epoch 68/69
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.7941

Epoch 69/69
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8088

Training complete in 3m 53s
Best val Acc: 0.838235

---Testing---
Test accuracy: 0.981077
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 98 %
Accuracy of Squatiniformes : 96 %
mean: 0.9754347124675631, std: 0.02140955543547634
--------------------

run info[val: 0.15, epoch: 88, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0211 Acc: 0.2705
val Loss: 0.0338 Acc: 0.4466

Epoch 1/87
----------
train Loss: 0.0146 Acc: 0.5154
val Loss: 0.0231 Acc: 0.5146

Epoch 2/87
----------
train Loss: 0.0092 Acc: 0.6832
val Loss: 0.0219 Acc: 0.7184

Epoch 3/87
----------
train Loss: 0.0064 Acc: 0.8271
val Loss: 0.0244 Acc: 0.7670

Epoch 4/87
----------
train Loss: 0.0049 Acc: 0.8664
val Loss: 0.0204 Acc: 0.7961

Epoch 5/87
----------
train Loss: 0.0043 Acc: 0.8750
val Loss: 0.0111 Acc: 0.8058

Epoch 6/87
----------
LR is set to 0.001
train Loss: 0.0038 Acc: 0.8938
val Loss: 0.0092 Acc: 0.8058

Epoch 7/87
----------
train Loss: 0.0037 Acc: 0.9007
val Loss: 0.0083 Acc: 0.8058

Epoch 8/87
----------
train Loss: 0.0037 Acc: 0.9024
val Loss: 0.0074 Acc: 0.7864

Epoch 9/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0078 Acc: 0.7961

Epoch 10/87
----------
train Loss: 0.0036 Acc: 0.9007
val Loss: 0.0128 Acc: 0.8058

Epoch 11/87
----------
train Loss: 0.0037 Acc: 0.9041
val Loss: 0.0088 Acc: 0.8058

Epoch 12/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0036 Acc: 0.8973
val Loss: 0.0168 Acc: 0.8058

Epoch 13/87
----------
train Loss: 0.0036 Acc: 0.8990
val Loss: 0.0126 Acc: 0.8058

Epoch 14/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0145 Acc: 0.8058

Epoch 15/87
----------
train Loss: 0.0034 Acc: 0.9281
val Loss: 0.0107 Acc: 0.8058

Epoch 16/87
----------
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0088 Acc: 0.8058

Epoch 17/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0094 Acc: 0.8058

Epoch 18/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0035 Acc: 0.8955
val Loss: 0.0080 Acc: 0.8058

Epoch 19/87
----------
train Loss: 0.0035 Acc: 0.8990
val Loss: 0.0067 Acc: 0.8058

Epoch 20/87
----------
train Loss: 0.0036 Acc: 0.9007
val Loss: 0.0082 Acc: 0.8058

Epoch 21/87
----------
train Loss: 0.0036 Acc: 0.8870
val Loss: 0.0065 Acc: 0.8058

Epoch 22/87
----------
train Loss: 0.0036 Acc: 0.9161
val Loss: 0.0100 Acc: 0.8058

Epoch 23/87
----------
train Loss: 0.0034 Acc: 0.9092
val Loss: 0.0079 Acc: 0.8058

Epoch 24/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.9058
val Loss: 0.0080 Acc: 0.8058

Epoch 25/87
----------
train Loss: 0.0037 Acc: 0.9024
val Loss: 0.0073 Acc: 0.8058

Epoch 26/87
----------
train Loss: 0.0036 Acc: 0.8990
val Loss: 0.0226 Acc: 0.8058

Epoch 27/87
----------
train Loss: 0.0037 Acc: 0.9007
val Loss: 0.0064 Acc: 0.8058

Epoch 28/87
----------
train Loss: 0.0036 Acc: 0.9007
val Loss: 0.0121 Acc: 0.8058

Epoch 29/87
----------
train Loss: 0.0034 Acc: 0.9110
val Loss: 0.0173 Acc: 0.8058

Epoch 30/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0037 Acc: 0.9075
val Loss: 0.0120 Acc: 0.8058

Epoch 31/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0066 Acc: 0.8058

Epoch 32/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0149 Acc: 0.8058

Epoch 33/87
----------
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0071 Acc: 0.8058

Epoch 34/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0083 Acc: 0.8058

Epoch 35/87
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0105 Acc: 0.8058

Epoch 36/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0035 Acc: 0.9058
val Loss: 0.0067 Acc: 0.8058

Epoch 37/87
----------
train Loss: 0.0035 Acc: 0.9110
val Loss: 0.0113 Acc: 0.8058

Epoch 38/87
----------
train Loss: 0.0035 Acc: 0.9041
val Loss: 0.0142 Acc: 0.8058

Epoch 39/87
----------
train Loss: 0.0035 Acc: 0.9127
val Loss: 0.0161 Acc: 0.8058

Epoch 40/87
----------
train Loss: 0.0037 Acc: 0.8973
val Loss: 0.0064 Acc: 0.8058

Epoch 41/87
----------
train Loss: 0.0036 Acc: 0.8973
val Loss: 0.0089 Acc: 0.8058

Epoch 42/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0036 Acc: 0.9058
val Loss: 0.0116 Acc: 0.8058

Epoch 43/87
----------
train Loss: 0.0036 Acc: 0.9161
val Loss: 0.0078 Acc: 0.8058

Epoch 44/87
----------
train Loss: 0.0035 Acc: 0.9041
val Loss: 0.0065 Acc: 0.8058

Epoch 45/87
----------
train Loss: 0.0034 Acc: 0.9092
val Loss: 0.0080 Acc: 0.8058

Epoch 46/87
----------
train Loss: 0.0035 Acc: 0.9024
val Loss: 0.0114 Acc: 0.8058

Epoch 47/87
----------
train Loss: 0.0034 Acc: 0.9178
val Loss: 0.0133 Acc: 0.8058

Epoch 48/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0035 Acc: 0.9127
val Loss: 0.0072 Acc: 0.8058

Epoch 49/87
----------
train Loss: 0.0037 Acc: 0.8921
val Loss: 0.0123 Acc: 0.8058

Epoch 50/87
----------
train Loss: 0.0038 Acc: 0.9058
val Loss: 0.0199 Acc: 0.8058

Epoch 51/87
----------
train Loss: 0.0033 Acc: 0.9178
val Loss: 0.0126 Acc: 0.8058

Epoch 52/87
----------
train Loss: 0.0037 Acc: 0.9058
val Loss: 0.0130 Acc: 0.8058

Epoch 53/87
----------
train Loss: 0.0036 Acc: 0.9041
val Loss: 0.0115 Acc: 0.8058

Epoch 54/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0035 Acc: 0.9110
val Loss: 0.0143 Acc: 0.8058

Epoch 55/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0103 Acc: 0.8058

Epoch 56/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0074 Acc: 0.8058

Epoch 57/87
----------
train Loss: 0.0037 Acc: 0.9041
val Loss: 0.0134 Acc: 0.8058

Epoch 58/87
----------
train Loss: 0.0035 Acc: 0.9178
val Loss: 0.0084 Acc: 0.8058

Epoch 59/87
----------
train Loss: 0.0035 Acc: 0.9092
val Loss: 0.0184 Acc: 0.8058

Epoch 60/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0035 Acc: 0.9110
val Loss: 0.0092 Acc: 0.8058

Epoch 61/87
----------
train Loss: 0.0039 Acc: 0.9041
val Loss: 0.0081 Acc: 0.8058

Epoch 62/87
----------
train Loss: 0.0034 Acc: 0.9041
val Loss: 0.0148 Acc: 0.8058

Epoch 63/87
----------
train Loss: 0.0035 Acc: 0.9195
val Loss: 0.0089 Acc: 0.8058

Epoch 64/87
----------
train Loss: 0.0036 Acc: 0.9161
val Loss: 0.0073 Acc: 0.8058

Epoch 65/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0069 Acc: 0.8058

Epoch 66/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0035 Acc: 0.8973
val Loss: 0.0122 Acc: 0.8058

Epoch 67/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0077 Acc: 0.8058

Epoch 68/87
----------
train Loss: 0.0038 Acc: 0.8973
val Loss: 0.0091 Acc: 0.8058

Epoch 69/87
----------
train Loss: 0.0036 Acc: 0.9041
val Loss: 0.0109 Acc: 0.8058

Epoch 70/87
----------
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0065 Acc: 0.8058

Epoch 71/87
----------
train Loss: 0.0036 Acc: 0.9144
val Loss: 0.0122 Acc: 0.8058

Epoch 72/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0037 Acc: 0.8990
val Loss: 0.0146 Acc: 0.8058

Epoch 73/87
----------
train Loss: 0.0036 Acc: 0.9041
val Loss: 0.0070 Acc: 0.8058

Epoch 74/87
----------
train Loss: 0.0036 Acc: 0.9092
val Loss: 0.0087 Acc: 0.8058

Epoch 75/87
----------
train Loss: 0.0035 Acc: 0.9092
val Loss: 0.0082 Acc: 0.8058

Epoch 76/87
----------
train Loss: 0.0035 Acc: 0.9024
val Loss: 0.0131 Acc: 0.8058

Epoch 77/87
----------
train Loss: 0.0036 Acc: 0.9075
val Loss: 0.0066 Acc: 0.8058

Epoch 78/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0036 Acc: 0.9110
val Loss: 0.0126 Acc: 0.8058

Epoch 79/87
----------
train Loss: 0.0036 Acc: 0.9075
val Loss: 0.0213 Acc: 0.8058

Epoch 80/87
----------
train Loss: 0.0035 Acc: 0.9144
val Loss: 0.0119 Acc: 0.8058

Epoch 81/87
----------
train Loss: 0.0036 Acc: 0.9024
val Loss: 0.0130 Acc: 0.8058

Epoch 82/87
----------
train Loss: 0.0036 Acc: 0.8955
val Loss: 0.0147 Acc: 0.8058

Epoch 83/87
----------
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0181 Acc: 0.8058

Epoch 84/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0088 Acc: 0.8058

Epoch 85/87
----------
train Loss: 0.0036 Acc: 0.8990
val Loss: 0.0149 Acc: 0.8058

Epoch 86/87
----------
train Loss: 0.0037 Acc: 0.8904
val Loss: 0.0068 Acc: 0.8058

Epoch 87/87
----------
train Loss: 0.0035 Acc: 0.8990
val Loss: 0.0070 Acc: 0.8058

Training complete in 4m 27s
Best val Acc: 0.805825

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0037 Acc: 0.8904
val Loss: 0.0126 Acc: 0.7864

Epoch 1/87
----------
train Loss: 0.0022 Acc: 0.9640
val Loss: 0.0136 Acc: 0.8544

Epoch 2/87
----------
train Loss: 0.0011 Acc: 0.9812
val Loss: 0.0076 Acc: 0.8350

Epoch 3/87
----------
train Loss: 0.0007 Acc: 0.9897
val Loss: 0.0060 Acc: 0.8058

Epoch 4/87
----------
train Loss: 0.0003 Acc: 0.9966
val Loss: 0.0056 Acc: 0.7961

Epoch 5/87
----------
train Loss: 0.0003 Acc: 0.9932
val Loss: 0.0062 Acc: 0.8447

Epoch 6/87
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8447

Epoch 7/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0070 Acc: 0.8447

Epoch 8/87
----------
train Loss: 0.0002 Acc: 0.9949
val Loss: 0.0048 Acc: 0.8447

Epoch 9/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8544

Epoch 10/87
----------
train Loss: 0.0002 Acc: 0.9949
val Loss: 0.0067 Acc: 0.8544

Epoch 11/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0113 Acc: 0.8447

Epoch 12/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 13/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 14/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0319 Acc: 0.8350

Epoch 15/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 16/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0122 Acc: 0.8350

Epoch 17/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0068 Acc: 0.8350

Epoch 18/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8350

Epoch 19/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0259 Acc: 0.8350

Epoch 20/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 21/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0114 Acc: 0.8350

Epoch 22/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 23/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0132 Acc: 0.8350

Epoch 24/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8350

Epoch 25/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 26/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 27/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 28/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 29/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8350

Epoch 30/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0214 Acc: 0.8447

Epoch 31/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0075 Acc: 0.8447

Epoch 32/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 33/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0130 Acc: 0.8350

Epoch 34/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0122 Acc: 0.8447

Epoch 35/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0182 Acc: 0.8350

Epoch 36/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0104 Acc: 0.8350

Epoch 37/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 38/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 39/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 40/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8350

Epoch 41/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 42/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 43/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0086 Acc: 0.8350

Epoch 44/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0075 Acc: 0.8350

Epoch 45/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 46/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 47/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 48/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 49/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0184 Acc: 0.8350

Epoch 50/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0109 Acc: 0.8350

Epoch 51/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 52/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 53/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0064 Acc: 0.8350

Epoch 54/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0158 Acc: 0.8350

Epoch 55/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0152 Acc: 0.8350

Epoch 56/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8350

Epoch 57/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0047 Acc: 0.8350

Epoch 58/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0232 Acc: 0.8350

Epoch 59/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 60/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0193 Acc: 0.8350

Epoch 61/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0052 Acc: 0.8350

Epoch 62/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0118 Acc: 0.8350

Epoch 63/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0137 Acc: 0.8350

Epoch 64/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8350

Epoch 65/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 66/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0048 Acc: 0.8350

Epoch 67/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0102 Acc: 0.8350

Epoch 68/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 69/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0049 Acc: 0.8350

Epoch 70/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 71/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0103 Acc: 0.8350

Epoch 72/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 73/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8350

Epoch 74/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0195 Acc: 0.8350

Epoch 75/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0051 Acc: 0.8350

Epoch 76/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0185 Acc: 0.8350

Epoch 77/87
----------
train Loss: 0.0002 Acc: 0.9983
val Loss: 0.0047 Acc: 0.8350

Epoch 78/87
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 79/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0135 Acc: 0.8350

Epoch 80/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8350

Epoch 81/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Epoch 82/87
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 83/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8350

Epoch 84/87
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8350

Epoch 85/87
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0046 Acc: 0.8350

Epoch 86/87
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0136 Acc: 0.8350

Epoch 87/87
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8350

Training complete in 4m 43s
Best val Acc: 0.854369

---Testing---
Test accuracy: 0.947598
--------------------
Accuracy of Carcharhiniformes : 93 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 87 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 95 %
Accuracy of Squaliformes : 85 %
Accuracy of Squatiniformes : 96 %
mean: 0.9377432713877434, std: 0.04625015255028296
--------------------

run info[val: 0.2, epoch: 59, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/58
----------
LR is set to 0.01
train Loss: 0.0231 Acc: 0.1855
val Loss: 0.0266 Acc: 0.4015

Epoch 1/58
----------
train Loss: 0.0148 Acc: 0.5618
val Loss: 0.0164 Acc: 0.5547

Epoch 2/58
----------
train Loss: 0.0094 Acc: 0.7018
val Loss: 0.0120 Acc: 0.7737

Epoch 3/58
----------
LR is set to 0.001
train Loss: 0.0070 Acc: 0.8273
val Loss: 0.0115 Acc: 0.7737

Epoch 4/58
----------
train Loss: 0.0067 Acc: 0.8382
val Loss: 0.0116 Acc: 0.7518

Epoch 5/58
----------
train Loss: 0.0065 Acc: 0.8309
val Loss: 0.0116 Acc: 0.7664

Epoch 6/58
----------
LR is set to 0.00010000000000000002
train Loss: 0.0063 Acc: 0.8418
val Loss: 0.0113 Acc: 0.7664

Epoch 7/58
----------
train Loss: 0.0064 Acc: 0.8236
val Loss: 0.0110 Acc: 0.7737

Epoch 8/58
----------
train Loss: 0.0065 Acc: 0.8327
val Loss: 0.0112 Acc: 0.7664

Epoch 9/58
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0063 Acc: 0.8418
val Loss: 0.0114 Acc: 0.7664

Epoch 10/58
----------
train Loss: 0.0064 Acc: 0.8418
val Loss: 0.0108 Acc: 0.7737

Epoch 11/58
----------
train Loss: 0.0063 Acc: 0.8327
val Loss: 0.0112 Acc: 0.7737

Epoch 12/58
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0061 Acc: 0.8455
val Loss: 0.0115 Acc: 0.7737

Epoch 13/58
----------
train Loss: 0.0065 Acc: 0.8527
val Loss: 0.0117 Acc: 0.7737

Epoch 14/58
----------
train Loss: 0.0063 Acc: 0.8345
val Loss: 0.0106 Acc: 0.7737

Epoch 15/58
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0064 Acc: 0.8309
val Loss: 0.0108 Acc: 0.7664

Epoch 16/58
----------
train Loss: 0.0063 Acc: 0.8455
val Loss: 0.0108 Acc: 0.7737

Epoch 17/58
----------
train Loss: 0.0065 Acc: 0.8255
val Loss: 0.0112 Acc: 0.7664

Epoch 18/58
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0064 Acc: 0.8382
val Loss: 0.0117 Acc: 0.7664

Epoch 19/58
----------
train Loss: 0.0063 Acc: 0.8273
val Loss: 0.0110 Acc: 0.7664

Epoch 20/58
----------
train Loss: 0.0064 Acc: 0.8455
val Loss: 0.0123 Acc: 0.7737

Epoch 21/58
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0062 Acc: 0.8418
val Loss: 0.0113 Acc: 0.7737

Epoch 22/58
----------
train Loss: 0.0064 Acc: 0.8473
val Loss: 0.0113 Acc: 0.7737

Epoch 23/58
----------
train Loss: 0.0064 Acc: 0.8309
val Loss: 0.0115 Acc: 0.7737

Epoch 24/58
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0064 Acc: 0.8327
val Loss: 0.0109 Acc: 0.7737

Epoch 25/58
----------
train Loss: 0.0062 Acc: 0.8491
val Loss: 0.0114 Acc: 0.7737

Epoch 26/58
----------
train Loss: 0.0066 Acc: 0.8345
val Loss: 0.0113 Acc: 0.7810

Epoch 27/58
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0066 Acc: 0.8218
val Loss: 0.0110 Acc: 0.7737

Epoch 28/58
----------
train Loss: 0.0066 Acc: 0.8218
val Loss: 0.0111 Acc: 0.7737

Epoch 29/58
----------
train Loss: 0.0064 Acc: 0.8382
val Loss: 0.0117 Acc: 0.7664

Epoch 30/58
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0064 Acc: 0.8418
val Loss: 0.0112 Acc: 0.7737

Epoch 31/58
----------
train Loss: 0.0064 Acc: 0.8327
val Loss: 0.0103 Acc: 0.7810

Epoch 32/58
----------
train Loss: 0.0064 Acc: 0.8582
val Loss: 0.0111 Acc: 0.7737

Epoch 33/58
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0062 Acc: 0.8418
val Loss: 0.0112 Acc: 0.7737

Epoch 34/58
----------
train Loss: 0.0064 Acc: 0.8527
val Loss: 0.0118 Acc: 0.7664

Epoch 35/58
----------
train Loss: 0.0064 Acc: 0.8455
val Loss: 0.0114 Acc: 0.7737

Epoch 36/58
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0065 Acc: 0.8400
val Loss: 0.0108 Acc: 0.7737

Epoch 37/58
----------
train Loss: 0.0063 Acc: 0.8418
val Loss: 0.0111 Acc: 0.7737

Epoch 38/58
----------
train Loss: 0.0065 Acc: 0.8309
val Loss: 0.0112 Acc: 0.7737

Epoch 39/58
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0066 Acc: 0.8055
val Loss: 0.0112 Acc: 0.7737

Epoch 40/58
----------
train Loss: 0.0064 Acc: 0.8364
val Loss: 0.0116 Acc: 0.7737

Epoch 41/58
----------
train Loss: 0.0063 Acc: 0.8418
val Loss: 0.0113 Acc: 0.7810

Epoch 42/58
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0061 Acc: 0.8527
val Loss: 0.0103 Acc: 0.7737

Epoch 43/58
----------
train Loss: 0.0064 Acc: 0.8527
val Loss: 0.0107 Acc: 0.7737

Epoch 44/58
----------
train Loss: 0.0062 Acc: 0.8455
val Loss: 0.0110 Acc: 0.7737

Epoch 45/58
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0062 Acc: 0.8564
val Loss: 0.0108 Acc: 0.7810

Epoch 46/58
----------
train Loss: 0.0061 Acc: 0.8527
val Loss: 0.0108 Acc: 0.7737

Epoch 47/58
----------
train Loss: 0.0064 Acc: 0.8564
val Loss: 0.0108 Acc: 0.7664

Epoch 48/58
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0065 Acc: 0.8382
val Loss: 0.0111 Acc: 0.7664

Epoch 49/58
----------
train Loss: 0.0065 Acc: 0.8345
val Loss: 0.0113 Acc: 0.7737

Epoch 50/58
----------
train Loss: 0.0065 Acc: 0.8327
val Loss: 0.0111 Acc: 0.7737

Epoch 51/58
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0065 Acc: 0.8182
val Loss: 0.0110 Acc: 0.7737

Epoch 52/58
----------
train Loss: 0.0065 Acc: 0.8436
val Loss: 0.0114 Acc: 0.7737

Epoch 53/58
----------
train Loss: 0.0062 Acc: 0.8382
val Loss: 0.0110 Acc: 0.7737

Epoch 54/58
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0061 Acc: 0.8473
val Loss: 0.0107 Acc: 0.7737

Epoch 55/58
----------
train Loss: 0.0065 Acc: 0.8236
val Loss: 0.0118 Acc: 0.7737

Epoch 56/58
----------
train Loss: 0.0064 Acc: 0.8491
val Loss: 0.0121 Acc: 0.7737

Epoch 57/58
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0064 Acc: 0.8327
val Loss: 0.0115 Acc: 0.7737

Epoch 58/58
----------
train Loss: 0.0064 Acc: 0.8291
val Loss: 0.0114 Acc: 0.7664

Training complete in 2m 54s
Best val Acc: 0.781022

---Fine tuning.---
Epoch 0/58
----------
LR is set to 0.01
train Loss: 0.0060 Acc: 0.8455
val Loss: 0.0084 Acc: 0.8175

Epoch 1/58
----------
train Loss: 0.0032 Acc: 0.9436
val Loss: 0.0056 Acc: 0.8540

Epoch 2/58
----------
train Loss: 0.0020 Acc: 0.9655
val Loss: 0.0055 Acc: 0.8467

Epoch 3/58
----------
LR is set to 0.001
train Loss: 0.0011 Acc: 0.9836
val Loss: 0.0060 Acc: 0.8540

Epoch 4/58
----------
train Loss: 0.0011 Acc: 0.9855
val Loss: 0.0054 Acc: 0.8686

Epoch 5/58
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0052 Acc: 0.8686

Epoch 6/58
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0057 Acc: 0.8613

Epoch 7/58
----------
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0053 Acc: 0.8613

Epoch 8/58
----------
train Loss: 0.0009 Acc: 0.9873
val Loss: 0.0054 Acc: 0.8613

Epoch 9/58
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9927
val Loss: 0.0057 Acc: 0.8613

Epoch 10/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0067 Acc: 0.8613

Epoch 11/58
----------
train Loss: 0.0010 Acc: 0.9836
val Loss: 0.0049 Acc: 0.8613

Epoch 12/58
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9909
val Loss: 0.0049 Acc: 0.8613

Epoch 13/58
----------
train Loss: 0.0010 Acc: 0.9909
val Loss: 0.0057 Acc: 0.8613

Epoch 14/58
----------
train Loss: 0.0009 Acc: 0.9927
val Loss: 0.0055 Acc: 0.8613

Epoch 15/58
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0007 Acc: 0.9964
val Loss: 0.0057 Acc: 0.8613

Epoch 16/58
----------
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0051 Acc: 0.8613

Epoch 17/58
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0046 Acc: 0.8686

Epoch 18/58
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9909
val Loss: 0.0050 Acc: 0.8613

Epoch 19/58
----------
train Loss: 0.0009 Acc: 0.9927
val Loss: 0.0060 Acc: 0.8613

Epoch 20/58
----------
train Loss: 0.0009 Acc: 0.9909
val Loss: 0.0050 Acc: 0.8613

Epoch 21/58
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0010 Acc: 0.9982
val Loss: 0.0049 Acc: 0.8613

Epoch 22/58
----------
train Loss: 0.0008 Acc: 0.9927
val Loss: 0.0048 Acc: 0.8613

Epoch 23/58
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0060 Acc: 0.8613

Epoch 24/58
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0009 Acc: 0.9909
val Loss: 0.0053 Acc: 0.8613

Epoch 25/58
----------
train Loss: 0.0010 Acc: 0.9945
val Loss: 0.0045 Acc: 0.8613

Epoch 26/58
----------
train Loss: 0.0009 Acc: 0.9927
val Loss: 0.0058 Acc: 0.8613

Epoch 27/58
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0050 Acc: 0.8613

Epoch 28/58
----------
train Loss: 0.0010 Acc: 0.9818
val Loss: 0.0059 Acc: 0.8613

Epoch 29/58
----------
train Loss: 0.0009 Acc: 0.9800
val Loss: 0.0062 Acc: 0.8613

Epoch 30/58
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0009 Acc: 0.9818
val Loss: 0.0042 Acc: 0.8613

Epoch 31/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0060 Acc: 0.8613

Epoch 32/58
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0056 Acc: 0.8613

Epoch 33/58
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0008 Acc: 0.9964
val Loss: 0.0061 Acc: 0.8613

Epoch 34/58
----------
train Loss: 0.0009 Acc: 0.9800
val Loss: 0.0062 Acc: 0.8613

Epoch 35/58
----------
train Loss: 0.0010 Acc: 0.9855
val Loss: 0.0057 Acc: 0.8613

Epoch 36/58
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0009 Acc: 0.9855
val Loss: 0.0042 Acc: 0.8613

Epoch 37/58
----------
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0049 Acc: 0.8686

Epoch 38/58
----------
train Loss: 0.0009 Acc: 0.9909
val Loss: 0.0063 Acc: 0.8613

Epoch 39/58
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0008 Acc: 0.9873
val Loss: 0.0056 Acc: 0.8613

Epoch 40/58
----------
train Loss: 0.0008 Acc: 0.9927
val Loss: 0.0059 Acc: 0.8613

Epoch 41/58
----------
train Loss: 0.0009 Acc: 0.9891
val Loss: 0.0047 Acc: 0.8613

Epoch 42/58
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0011 Acc: 0.9873
val Loss: 0.0051 Acc: 0.8613

Epoch 43/58
----------
train Loss: 0.0009 Acc: 0.9855
val Loss: 0.0044 Acc: 0.8613

Epoch 44/58
----------
train Loss: 0.0008 Acc: 0.9927
val Loss: 0.0047 Acc: 0.8613

Epoch 45/58
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0009 Acc: 0.9855
val Loss: 0.0060 Acc: 0.8613

Epoch 46/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0046 Acc: 0.8613

Epoch 47/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0056 Acc: 0.8613

Epoch 48/58
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0009 Acc: 0.9873
val Loss: 0.0063 Acc: 0.8613

Epoch 49/58
----------
train Loss: 0.0008 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8613

Epoch 50/58
----------
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0053 Acc: 0.8613

Epoch 51/58
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0010 Acc: 0.9909
val Loss: 0.0063 Acc: 0.8613

Epoch 52/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0055 Acc: 0.8613

Epoch 53/58
----------
train Loss: 0.0011 Acc: 0.9818
val Loss: 0.0052 Acc: 0.8613

Epoch 54/58
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0010 Acc: 0.9855
val Loss: 0.0048 Acc: 0.8686

Epoch 55/58
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0050 Acc: 0.8686

Epoch 56/58
----------
train Loss: 0.0009 Acc: 0.9909
val Loss: 0.0055 Acc: 0.8686

Epoch 57/58
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0010 Acc: 0.9873
val Loss: 0.0045 Acc: 0.8613

Epoch 58/58
----------
train Loss: 0.0008 Acc: 0.9927
val Loss: 0.0053 Acc: 0.8686

Training complete in 3m 4s
Best val Acc: 0.868613

---Testing---
Test accuracy: 0.967977
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 89 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 97 %
mean: 0.9591529177580926, std: 0.033085533504520814
--------------------

run info[val: 0.25, epoch: 76, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0228 Acc: 0.2035
val Loss: 0.0215 Acc: 0.2456

Epoch 1/75
----------
train Loss: 0.0152 Acc: 0.5717
val Loss: 0.0136 Acc: 0.5556

Epoch 2/75
----------
train Loss: 0.0102 Acc: 0.7209
val Loss: 0.0104 Acc: 0.6901

Epoch 3/75
----------
LR is set to 0.001
train Loss: 0.0081 Acc: 0.8120
val Loss: 0.0094 Acc: 0.7193

Epoch 4/75
----------
train Loss: 0.0070 Acc: 0.8178
val Loss: 0.0090 Acc: 0.7368

Epoch 5/75
----------
train Loss: 0.0068 Acc: 0.8353
val Loss: 0.0088 Acc: 0.7602

Epoch 6/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0074 Acc: 0.8527
val Loss: 0.0088 Acc: 0.7602

Epoch 7/75
----------
train Loss: 0.0067 Acc: 0.8411
val Loss: 0.0085 Acc: 0.7485

Epoch 8/75
----------
train Loss: 0.0066 Acc: 0.8295
val Loss: 0.0085 Acc: 0.7544

Epoch 9/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.8353
val Loss: 0.0087 Acc: 0.7544

Epoch 10/75
----------
train Loss: 0.0068 Acc: 0.8372
val Loss: 0.0087 Acc: 0.7602

Epoch 11/75
----------
train Loss: 0.0074 Acc: 0.8372
val Loss: 0.0086 Acc: 0.7602

Epoch 12/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0066 Acc: 0.8353
val Loss: 0.0087 Acc: 0.7602

Epoch 13/75
----------
train Loss: 0.0075 Acc: 0.8411
val Loss: 0.0086 Acc: 0.7602

Epoch 14/75
----------
train Loss: 0.0071 Acc: 0.8450
val Loss: 0.0086 Acc: 0.7544

Epoch 15/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0076 Acc: 0.8295
val Loss: 0.0087 Acc: 0.7602

Epoch 16/75
----------
train Loss: 0.0067 Acc: 0.8411
val Loss: 0.0086 Acc: 0.7661

Epoch 17/75
----------
train Loss: 0.0071 Acc: 0.8295
val Loss: 0.0086 Acc: 0.7544

Epoch 18/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0069 Acc: 0.8488
val Loss: 0.0085 Acc: 0.7602

Epoch 19/75
----------
train Loss: 0.0074 Acc: 0.8450
val Loss: 0.0089 Acc: 0.7544

Epoch 20/75
----------
train Loss: 0.0067 Acc: 0.8508
val Loss: 0.0088 Acc: 0.7602

Epoch 21/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0066 Acc: 0.8585
val Loss: 0.0088 Acc: 0.7544

Epoch 22/75
----------
train Loss: 0.0071 Acc: 0.8178
val Loss: 0.0089 Acc: 0.7544

Epoch 23/75
----------
train Loss: 0.0070 Acc: 0.8411
val Loss: 0.0087 Acc: 0.7544

Epoch 24/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0069 Acc: 0.8353
val Loss: 0.0089 Acc: 0.7427

Epoch 25/75
----------
train Loss: 0.0064 Acc: 0.8411
val Loss: 0.0087 Acc: 0.7544

Epoch 26/75
----------
train Loss: 0.0069 Acc: 0.8275
val Loss: 0.0087 Acc: 0.7544

Epoch 27/75
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0073 Acc: 0.8430
val Loss: 0.0088 Acc: 0.7544

Epoch 28/75
----------
train Loss: 0.0067 Acc: 0.8353
val Loss: 0.0088 Acc: 0.7427

Epoch 29/75
----------
train Loss: 0.0066 Acc: 0.8430
val Loss: 0.0086 Acc: 0.7602

Epoch 30/75
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0069 Acc: 0.8508
val Loss: 0.0088 Acc: 0.7602

Epoch 31/75
----------
train Loss: 0.0069 Acc: 0.8527
val Loss: 0.0087 Acc: 0.7661

Epoch 32/75
----------
train Loss: 0.0072 Acc: 0.8488
val Loss: 0.0087 Acc: 0.7544

Epoch 33/75
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0073 Acc: 0.8508
val Loss: 0.0088 Acc: 0.7661

Epoch 34/75
----------
train Loss: 0.0070 Acc: 0.8256
val Loss: 0.0087 Acc: 0.7661

Epoch 35/75
----------
train Loss: 0.0073 Acc: 0.8314
val Loss: 0.0088 Acc: 0.7602

Epoch 36/75
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0072 Acc: 0.8101
val Loss: 0.0088 Acc: 0.7602

Epoch 37/75
----------
train Loss: 0.0067 Acc: 0.8411
val Loss: 0.0085 Acc: 0.7602

Epoch 38/75
----------
train Loss: 0.0068 Acc: 0.8450
val Loss: 0.0085 Acc: 0.7602

Epoch 39/75
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0070 Acc: 0.8295
val Loss: 0.0086 Acc: 0.7602

Epoch 40/75
----------
train Loss: 0.0070 Acc: 0.8275
val Loss: 0.0088 Acc: 0.7544

Epoch 41/75
----------
train Loss: 0.0067 Acc: 0.8372
val Loss: 0.0084 Acc: 0.7544

Epoch 42/75
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0070 Acc: 0.8353
val Loss: 0.0090 Acc: 0.7485

Epoch 43/75
----------
train Loss: 0.0070 Acc: 0.8469
val Loss: 0.0090 Acc: 0.7485

Epoch 44/75
----------
train Loss: 0.0070 Acc: 0.8469
val Loss: 0.0089 Acc: 0.7544

Epoch 45/75
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0066 Acc: 0.8391
val Loss: 0.0088 Acc: 0.7544

Epoch 46/75
----------
train Loss: 0.0068 Acc: 0.8566
val Loss: 0.0089 Acc: 0.7602

Epoch 47/75
----------
train Loss: 0.0069 Acc: 0.8411
val Loss: 0.0086 Acc: 0.7544

Epoch 48/75
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0070 Acc: 0.8353
val Loss: 0.0086 Acc: 0.7661

Epoch 49/75
----------
train Loss: 0.0070 Acc: 0.8469
val Loss: 0.0086 Acc: 0.7661

Epoch 50/75
----------
train Loss: 0.0075 Acc: 0.8353
val Loss: 0.0087 Acc: 0.7544

Epoch 51/75
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0070 Acc: 0.8333
val Loss: 0.0087 Acc: 0.7602

Epoch 52/75
----------
train Loss: 0.0075 Acc: 0.8372
val Loss: 0.0086 Acc: 0.7661

Epoch 53/75
----------
train Loss: 0.0069 Acc: 0.8527
val Loss: 0.0087 Acc: 0.7602

Epoch 54/75
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0068 Acc: 0.8527
val Loss: 0.0086 Acc: 0.7602

Epoch 55/75
----------
train Loss: 0.0068 Acc: 0.8566
val Loss: 0.0087 Acc: 0.7544

Epoch 56/75
----------
train Loss: 0.0074 Acc: 0.8353
val Loss: 0.0088 Acc: 0.7485

Epoch 57/75
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0071 Acc: 0.8333
val Loss: 0.0087 Acc: 0.7427

Epoch 58/75
----------
train Loss: 0.0064 Acc: 0.8508
val Loss: 0.0088 Acc: 0.7485

Epoch 59/75
----------
train Loss: 0.0073 Acc: 0.8314
val Loss: 0.0089 Acc: 0.7544

Epoch 60/75
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0074 Acc: 0.8275
val Loss: 0.0087 Acc: 0.7485

Epoch 61/75
----------
train Loss: 0.0072 Acc: 0.8411
val Loss: 0.0086 Acc: 0.7602

Epoch 62/75
----------
train Loss: 0.0068 Acc: 0.8256
val Loss: 0.0088 Acc: 0.7661

Epoch 63/75
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0069 Acc: 0.8372
val Loss: 0.0088 Acc: 0.7602

Epoch 64/75
----------
train Loss: 0.0068 Acc: 0.8585
val Loss: 0.0086 Acc: 0.7485

Epoch 65/75
----------
train Loss: 0.0070 Acc: 0.8353
val Loss: 0.0087 Acc: 0.7602

Epoch 66/75
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0071 Acc: 0.8527
val Loss: 0.0085 Acc: 0.7544

Epoch 67/75
----------
train Loss: 0.0070 Acc: 0.8469
val Loss: 0.0087 Acc: 0.7544

Epoch 68/75
----------
train Loss: 0.0070 Acc: 0.8566
val Loss: 0.0088 Acc: 0.7544

Epoch 69/75
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0064 Acc: 0.8391
val Loss: 0.0086 Acc: 0.7602

Epoch 70/75
----------
train Loss: 0.0072 Acc: 0.8217
val Loss: 0.0089 Acc: 0.7661

Epoch 71/75
----------
train Loss: 0.0067 Acc: 0.8372
val Loss: 0.0087 Acc: 0.7602

Epoch 72/75
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0072 Acc: 0.8527
val Loss: 0.0089 Acc: 0.7544

Epoch 73/75
----------
train Loss: 0.0069 Acc: 0.8372
val Loss: 0.0086 Acc: 0.7602

Epoch 74/75
----------
train Loss: 0.0066 Acc: 0.8508
val Loss: 0.0087 Acc: 0.7544

Epoch 75/75
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0070 Acc: 0.8353
val Loss: 0.0087 Acc: 0.7427

Training complete in 3m 44s
Best val Acc: 0.766082

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0065 Acc: 0.8469
val Loss: 0.0073 Acc: 0.8070

Epoch 1/75
----------
train Loss: 0.0041 Acc: 0.9147
val Loss: 0.0067 Acc: 0.8187

Epoch 2/75
----------
train Loss: 0.0030 Acc: 0.9264
val Loss: 0.0058 Acc: 0.8538

Epoch 3/75
----------
LR is set to 0.001
train Loss: 0.0014 Acc: 0.9767
val Loss: 0.0048 Acc: 0.8596

Epoch 4/75
----------
train Loss: 0.0016 Acc: 0.9709
val Loss: 0.0048 Acc: 0.8655

Epoch 5/75
----------
train Loss: 0.0014 Acc: 0.9826
val Loss: 0.0045 Acc: 0.8713

Epoch 6/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0048 Acc: 0.8655

Epoch 7/75
----------
train Loss: 0.0011 Acc: 0.9806
val Loss: 0.0045 Acc: 0.8596

Epoch 8/75
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0050 Acc: 0.8596

Epoch 9/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0046 Acc: 0.8596

Epoch 10/75
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8596

Epoch 11/75
----------
train Loss: 0.0012 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8655

Epoch 12/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0051 Acc: 0.8596

Epoch 13/75
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0052 Acc: 0.8480

Epoch 14/75
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0051 Acc: 0.8480

Epoch 15/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0008 Acc: 0.9884
val Loss: 0.0051 Acc: 0.8480

Epoch 16/75
----------
train Loss: 0.0013 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8538

Epoch 17/75
----------
train Loss: 0.0015 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8713

Epoch 18/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0010 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8596

Epoch 19/75
----------
train Loss: 0.0015 Acc: 0.9884
val Loss: 0.0048 Acc: 0.8538

Epoch 20/75
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0047 Acc: 0.8596

Epoch 21/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9826
val Loss: 0.0044 Acc: 0.8655

Epoch 22/75
----------
train Loss: 0.0012 Acc: 0.9767
val Loss: 0.0047 Acc: 0.8655

Epoch 23/75
----------
train Loss: 0.0018 Acc: 0.9826
val Loss: 0.0046 Acc: 0.8596

Epoch 24/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9748
val Loss: 0.0048 Acc: 0.8596

Epoch 25/75
----------
train Loss: 0.0012 Acc: 0.9864
val Loss: 0.0051 Acc: 0.8596

Epoch 26/75
----------
train Loss: 0.0008 Acc: 0.9884
val Loss: 0.0047 Acc: 0.8596

Epoch 27/75
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0052 Acc: 0.8538

Epoch 28/75
----------
train Loss: 0.0015 Acc: 0.9806
val Loss: 0.0049 Acc: 0.8596

Epoch 29/75
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8596

Epoch 30/75
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0012 Acc: 0.9864
val Loss: 0.0047 Acc: 0.8538

Epoch 31/75
----------
train Loss: 0.0011 Acc: 0.9767
val Loss: 0.0050 Acc: 0.8596

Epoch 32/75
----------
train Loss: 0.0010 Acc: 0.9787
val Loss: 0.0047 Acc: 0.8538

Epoch 33/75
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0012 Acc: 0.9787
val Loss: 0.0050 Acc: 0.8480

Epoch 34/75
----------
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0046 Acc: 0.8655

Epoch 35/75
----------
train Loss: 0.0009 Acc: 0.9903
val Loss: 0.0049 Acc: 0.8596

Epoch 36/75
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0011 Acc: 0.9903
val Loss: 0.0051 Acc: 0.8596

Epoch 37/75
----------
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8655

Epoch 38/75
----------
train Loss: 0.0013 Acc: 0.9787
val Loss: 0.0045 Acc: 0.8655

Epoch 39/75
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0049 Acc: 0.8538

Epoch 40/75
----------
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8538

Epoch 41/75
----------
train Loss: 0.0012 Acc: 0.9845
val Loss: 0.0050 Acc: 0.8538

Epoch 42/75
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0011 Acc: 0.9806
val Loss: 0.0046 Acc: 0.8655

Epoch 43/75
----------
train Loss: 0.0009 Acc: 0.9826
val Loss: 0.0047 Acc: 0.8538

Epoch 44/75
----------
train Loss: 0.0009 Acc: 0.9845
val Loss: 0.0049 Acc: 0.8538

Epoch 45/75
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0012 Acc: 0.9767
val Loss: 0.0051 Acc: 0.8480

Epoch 46/75
----------
train Loss: 0.0012 Acc: 0.9787
val Loss: 0.0049 Acc: 0.8596

Epoch 47/75
----------
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0050 Acc: 0.8480

Epoch 48/75
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0009 Acc: 0.9826
val Loss: 0.0050 Acc: 0.8538

Epoch 49/75
----------
train Loss: 0.0010 Acc: 0.9787
val Loss: 0.0048 Acc: 0.8596

Epoch 50/75
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8713

Epoch 51/75
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0012 Acc: 0.9806
val Loss: 0.0050 Acc: 0.8480

Epoch 52/75
----------
train Loss: 0.0010 Acc: 0.9806
val Loss: 0.0048 Acc: 0.8480

Epoch 53/75
----------
train Loss: 0.0011 Acc: 0.9826
val Loss: 0.0047 Acc: 0.8596

Epoch 54/75
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0047 Acc: 0.8538

Epoch 55/75
----------
train Loss: 0.0009 Acc: 0.9787
val Loss: 0.0044 Acc: 0.8596

Epoch 56/75
----------
train Loss: 0.0010 Acc: 0.9845
val Loss: 0.0051 Acc: 0.8596

Epoch 57/75
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0013 Acc: 0.9748
val Loss: 0.0048 Acc: 0.8596

Epoch 58/75
----------
train Loss: 0.0013 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8596

Epoch 59/75
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0048 Acc: 0.8596

Epoch 60/75
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0012 Acc: 0.9806
val Loss: 0.0049 Acc: 0.8596

Epoch 61/75
----------
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0048 Acc: 0.8596

Epoch 62/75
----------
train Loss: 0.0011 Acc: 0.9826
val Loss: 0.0049 Acc: 0.8538

Epoch 63/75
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0010 Acc: 0.9903
val Loss: 0.0048 Acc: 0.8538

Epoch 64/75
----------
train Loss: 0.0008 Acc: 0.9864
val Loss: 0.0050 Acc: 0.8538

Epoch 65/75
----------
train Loss: 0.0008 Acc: 0.9922
val Loss: 0.0047 Acc: 0.8538

Epoch 66/75
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0046 Acc: 0.8538

Epoch 67/75
----------
train Loss: 0.0014 Acc: 0.9806
val Loss: 0.0049 Acc: 0.8596

Epoch 68/75
----------
train Loss: 0.0012 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8538

Epoch 69/75
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0048 Acc: 0.8538

Epoch 70/75
----------
train Loss: 0.0010 Acc: 0.9806
val Loss: 0.0045 Acc: 0.8538

Epoch 71/75
----------
train Loss: 0.0009 Acc: 0.9922
val Loss: 0.0048 Acc: 0.8596

Epoch 72/75
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0011 Acc: 0.9845
val Loss: 0.0047 Acc: 0.8596

Epoch 73/75
----------
train Loss: 0.0010 Acc: 0.9884
val Loss: 0.0046 Acc: 0.8538

Epoch 74/75
----------
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0051 Acc: 0.8596

Epoch 75/75
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0010 Acc: 0.9767
val Loss: 0.0048 Acc: 0.8596

Training complete in 3m 56s
Best val Acc: 0.871345

---Testing---
Test accuracy: 0.959243
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 92 %
Accuracy of Lamniformes : 89 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 96 %
mean: 0.9497418118205825, std: 0.033893558579007284
--------------------

run info[val: 0.3, epoch: 81, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0238 Acc: 0.1684
val Loss: 0.0294 Acc: 0.2718

Epoch 1/80
----------
train Loss: 0.0175 Acc: 0.4407
val Loss: 0.0180 Acc: 0.5583

Epoch 2/80
----------
train Loss: 0.0112 Acc: 0.6237
val Loss: 0.0139 Acc: 0.6650

Epoch 3/80
----------
train Loss: 0.0075 Acc: 0.7651
val Loss: 0.0086 Acc: 0.8155

Epoch 4/80
----------
train Loss: 0.0058 Acc: 0.8295
val Loss: 0.0097 Acc: 0.8010

Epoch 5/80
----------
train Loss: 0.0050 Acc: 0.8441
val Loss: 0.0069 Acc: 0.8301

Epoch 6/80
----------
train Loss: 0.0045 Acc: 0.8565
val Loss: 0.0108 Acc: 0.8398

Epoch 7/80
----------
train Loss: 0.0039 Acc: 0.8919
val Loss: 0.0072 Acc: 0.8204

Epoch 8/80
----------
LR is set to 0.001
train Loss: 0.0036 Acc: 0.9085
val Loss: 0.0075 Acc: 0.8204

Epoch 9/80
----------
train Loss: 0.0035 Acc: 0.8981
val Loss: 0.0064 Acc: 0.8350

Epoch 10/80
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0054 Acc: 0.8398

Epoch 11/80
----------
train Loss: 0.0033 Acc: 0.9272
val Loss: 0.0055 Acc: 0.8301

Epoch 12/80
----------
train Loss: 0.0033 Acc: 0.9272
val Loss: 0.0115 Acc: 0.8301

Epoch 13/80
----------
train Loss: 0.0032 Acc: 0.9210
val Loss: 0.0054 Acc: 0.8301

Epoch 14/80
----------
train Loss: 0.0032 Acc: 0.9231
val Loss: 0.0093 Acc: 0.8301

Epoch 15/80
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0071 Acc: 0.8350

Epoch 16/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0032 Acc: 0.9127
val Loss: 0.0077 Acc: 0.8301

Epoch 17/80
----------
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0055 Acc: 0.8350

Epoch 18/80
----------
train Loss: 0.0032 Acc: 0.9210
val Loss: 0.0063 Acc: 0.8350

Epoch 19/80
----------
train Loss: 0.0033 Acc: 0.9106
val Loss: 0.0067 Acc: 0.8350

Epoch 20/80
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0124 Acc: 0.8350

Epoch 21/80
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0083 Acc: 0.8350

Epoch 22/80
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0065 Acc: 0.8350

Epoch 23/80
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0052 Acc: 0.8350

Epoch 24/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0055 Acc: 0.8350

Epoch 25/80
----------
train Loss: 0.0033 Acc: 0.9272
val Loss: 0.0084 Acc: 0.8350

Epoch 26/80
----------
train Loss: 0.0032 Acc: 0.9231
val Loss: 0.0060 Acc: 0.8350

Epoch 27/80
----------
train Loss: 0.0032 Acc: 0.9293
val Loss: 0.0072 Acc: 0.8350

Epoch 28/80
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0052 Acc: 0.8301

Epoch 29/80
----------
train Loss: 0.0033 Acc: 0.9064
val Loss: 0.0056 Acc: 0.8301

Epoch 30/80
----------
train Loss: 0.0032 Acc: 0.9335
val Loss: 0.0068 Acc: 0.8301

Epoch 31/80
----------
train Loss: 0.0032 Acc: 0.9106
val Loss: 0.0059 Acc: 0.8301

Epoch 32/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0033 Acc: 0.9168
val Loss: 0.0050 Acc: 0.8350

Epoch 33/80
----------
train Loss: 0.0035 Acc: 0.9127
val Loss: 0.0053 Acc: 0.8350

Epoch 34/80
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0057 Acc: 0.8350

Epoch 35/80
----------
train Loss: 0.0031 Acc: 0.9231
val Loss: 0.0061 Acc: 0.8350

Epoch 36/80
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0064 Acc: 0.8350

Epoch 37/80
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0084 Acc: 0.8301

Epoch 38/80
----------
train Loss: 0.0031 Acc: 0.9252
val Loss: 0.0049 Acc: 0.8350

Epoch 39/80
----------
train Loss: 0.0033 Acc: 0.9314
val Loss: 0.0055 Acc: 0.8350

Epoch 40/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0061 Acc: 0.8350

Epoch 41/80
----------
train Loss: 0.0033 Acc: 0.9356
val Loss: 0.0065 Acc: 0.8350

Epoch 42/80
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0080 Acc: 0.8350

Epoch 43/80
----------
train Loss: 0.0033 Acc: 0.9168
val Loss: 0.0053 Acc: 0.8350

Epoch 44/80
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0074 Acc: 0.8350

Epoch 45/80
----------
train Loss: 0.0032 Acc: 0.9106
val Loss: 0.0059 Acc: 0.8398

Epoch 46/80
----------
train Loss: 0.0033 Acc: 0.9106
val Loss: 0.0107 Acc: 0.8350

Epoch 47/80
----------
train Loss: 0.0032 Acc: 0.9148
val Loss: 0.0112 Acc: 0.8350

Epoch 48/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0060 Acc: 0.8398

Epoch 49/80
----------
train Loss: 0.0034 Acc: 0.9106
val Loss: 0.0065 Acc: 0.8350

Epoch 50/80
----------
train Loss: 0.0032 Acc: 0.9210
val Loss: 0.0053 Acc: 0.8350

Epoch 51/80
----------
train Loss: 0.0036 Acc: 0.8940
val Loss: 0.0056 Acc: 0.8350

Epoch 52/80
----------
train Loss: 0.0033 Acc: 0.9148
val Loss: 0.0063 Acc: 0.8350

Epoch 53/80
----------
train Loss: 0.0032 Acc: 0.9106
val Loss: 0.0081 Acc: 0.8350

Epoch 54/80
----------
train Loss: 0.0033 Acc: 0.9106
val Loss: 0.0058 Acc: 0.8350

Epoch 55/80
----------
train Loss: 0.0034 Acc: 0.9168
val Loss: 0.0060 Acc: 0.8350

Epoch 56/80
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0088 Acc: 0.8350

Epoch 57/80
----------
train Loss: 0.0033 Acc: 0.9210
val Loss: 0.0079 Acc: 0.8301

Epoch 58/80
----------
train Loss: 0.0033 Acc: 0.9127
val Loss: 0.0051 Acc: 0.8350

Epoch 59/80
----------
train Loss: 0.0033 Acc: 0.9252
val Loss: 0.0077 Acc: 0.8350

Epoch 60/80
----------
train Loss: 0.0032 Acc: 0.9106
val Loss: 0.0065 Acc: 0.8350

Epoch 61/80
----------
train Loss: 0.0033 Acc: 0.9148
val Loss: 0.0064 Acc: 0.8350

Epoch 62/80
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0070 Acc: 0.8350

Epoch 63/80
----------
train Loss: 0.0034 Acc: 0.9106
val Loss: 0.0079 Acc: 0.8350

Epoch 64/80
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0034 Acc: 0.9168
val Loss: 0.0055 Acc: 0.8301

Epoch 65/80
----------
train Loss: 0.0033 Acc: 0.9189
val Loss: 0.0062 Acc: 0.8350

Epoch 66/80
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0101 Acc: 0.8350

Epoch 67/80
----------
train Loss: 0.0031 Acc: 0.9231
val Loss: 0.0053 Acc: 0.8301

Epoch 68/80
----------
train Loss: 0.0030 Acc: 0.9356
val Loss: 0.0056 Acc: 0.8350

Epoch 69/80
----------
train Loss: 0.0033 Acc: 0.9189
val Loss: 0.0060 Acc: 0.8350

Epoch 70/80
----------
train Loss: 0.0035 Acc: 0.9231
val Loss: 0.0105 Acc: 0.8350

Epoch 71/80
----------
train Loss: 0.0034 Acc: 0.9168
val Loss: 0.0060 Acc: 0.8350

Epoch 72/80
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0032 Acc: 0.9293
val Loss: 0.0051 Acc: 0.8350

Epoch 73/80
----------
train Loss: 0.0032 Acc: 0.9189
val Loss: 0.0091 Acc: 0.8350

Epoch 74/80
----------
train Loss: 0.0032 Acc: 0.9272
val Loss: 0.0088 Acc: 0.8350

Epoch 75/80
----------
train Loss: 0.0032 Acc: 0.9210
val Loss: 0.0087 Acc: 0.8350

Epoch 76/80
----------
train Loss: 0.0033 Acc: 0.9189
val Loss: 0.0082 Acc: 0.8350

Epoch 77/80
----------
train Loss: 0.0034 Acc: 0.9044
val Loss: 0.0059 Acc: 0.8350

Epoch 78/80
----------
train Loss: 0.0034 Acc: 0.9189
val Loss: 0.0089 Acc: 0.8350

Epoch 79/80
----------
train Loss: 0.0032 Acc: 0.9252
val Loss: 0.0076 Acc: 0.8350

Epoch 80/80
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0033 Acc: 0.9189
val Loss: 0.0053 Acc: 0.8350

Training complete in 3m 54s
Best val Acc: 0.839806

---Fine tuning.---
Epoch 0/80
----------
LR is set to 0.01
train Loss: 0.0038 Acc: 0.8940
val Loss: 0.0072 Acc: 0.8544

Epoch 1/80
----------
train Loss: 0.0024 Acc: 0.9563
val Loss: 0.0064 Acc: 0.8689

Epoch 2/80
----------
train Loss: 0.0013 Acc: 0.9730
val Loss: 0.0049 Acc: 0.8689

Epoch 3/80
----------
train Loss: 0.0008 Acc: 0.9854
val Loss: 0.0044 Acc: 0.9029

Epoch 4/80
----------
train Loss: 0.0005 Acc: 0.9938
val Loss: 0.0052 Acc: 0.8932

Epoch 5/80
----------
train Loss: 0.0004 Acc: 0.9938
val Loss: 0.0082 Acc: 0.8932

Epoch 6/80
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0100 Acc: 0.8981

Epoch 7/80
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0065 Acc: 0.9029

Epoch 8/80
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.9029

Epoch 9/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0060 Acc: 0.9029

Epoch 10/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 11/80
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9029

Epoch 12/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.9029

Epoch 13/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9029

Epoch 14/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9029

Epoch 15/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 16/80
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9029

Epoch 17/80
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9029

Epoch 18/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.9029

Epoch 19/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0033 Acc: 0.9029

Epoch 20/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 21/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.9029

Epoch 22/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9029

Epoch 23/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0044 Acc: 0.9029

Epoch 24/80
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 25/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9029

Epoch 26/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0065 Acc: 0.9029

Epoch 27/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9029

Epoch 28/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9029

Epoch 29/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9029

Epoch 30/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9029

Epoch 31/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0092 Acc: 0.9029

Epoch 32/80
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9029

Epoch 33/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0046 Acc: 0.9029

Epoch 34/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0061 Acc: 0.9029

Epoch 35/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0037 Acc: 0.9029

Epoch 36/80
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0032 Acc: 0.9029

Epoch 37/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 38/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 39/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 40/80
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0118 Acc: 0.9029

Epoch 41/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9029

Epoch 42/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9029

Epoch 43/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.9029

Epoch 44/80
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0033 Acc: 0.9029

Epoch 45/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0041 Acc: 0.9029

Epoch 46/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0135 Acc: 0.9029

Epoch 47/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 48/80
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 49/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9029

Epoch 50/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9029

Epoch 51/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9029

Epoch 52/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0037 Acc: 0.9029

Epoch 53/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9029

Epoch 54/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.9029

Epoch 55/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.9029

Epoch 56/80
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0036 Acc: 0.9029

Epoch 57/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.9029

Epoch 58/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.9029

Epoch 59/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 60/80
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0037 Acc: 0.9029

Epoch 61/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9029

Epoch 62/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 63/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0062 Acc: 0.9029

Epoch 64/80
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.9029

Epoch 65/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.9029

Epoch 66/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0059 Acc: 0.9029

Epoch 67/80
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0032 Acc: 0.9029

Epoch 68/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.9029

Epoch 69/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.9029

Epoch 70/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.9029

Epoch 71/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0068 Acc: 0.9029

Epoch 72/80
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0101 Acc: 0.9029

Epoch 73/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0069 Acc: 0.9029

Epoch 74/80
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0033 Acc: 0.9029

Epoch 75/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0064 Acc: 0.9029

Epoch 76/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9029

Epoch 77/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9029

Epoch 78/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.9029

Epoch 79/80
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0085 Acc: 0.9029

Epoch 80/80
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0034 Acc: 0.9029

Training complete in 4m 4s
Best val Acc: 0.902913

---Testing---
Test accuracy: 0.963610
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 97 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 96 %
mean: 0.9548194597995905, std: 0.04320019816284225

Model saved in "./weights/shark_[0.98]_mean[0.98]_std[0.02].save".
--------------------

run info[val: 0.1, epoch: 52, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/51
----------
LR is set to 0.01
train Loss: 0.0231 Acc: 0.2229
val Loss: 0.0256 Acc: 0.3971

Epoch 1/51
----------
train Loss: 0.0158 Acc: 0.5024
val Loss: 0.0145 Acc: 0.7059

Epoch 2/51
----------
train Loss: 0.0095 Acc: 0.7480
val Loss: 0.0137 Acc: 0.6765

Epoch 3/51
----------
train Loss: 0.0076 Acc: 0.7884
val Loss: 0.0118 Acc: 0.7353

Epoch 4/51
----------
train Loss: 0.0060 Acc: 0.8223
val Loss: 0.0094 Acc: 0.8235

Epoch 5/51
----------
train Loss: 0.0048 Acc: 0.8708
val Loss: 0.0092 Acc: 0.7794

Epoch 6/51
----------
train Loss: 0.0045 Acc: 0.8837
val Loss: 0.0100 Acc: 0.7941

Epoch 7/51
----------
train Loss: 0.0039 Acc: 0.8934
val Loss: 0.0093 Acc: 0.7794

Epoch 8/51
----------
LR is set to 0.001
train Loss: 0.0039 Acc: 0.9111
val Loss: 0.0096 Acc: 0.7941

Epoch 9/51
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0092 Acc: 0.7794

Epoch 10/51
----------
train Loss: 0.0030 Acc: 0.9289
val Loss: 0.0092 Acc: 0.7794

Epoch 11/51
----------
train Loss: 0.0029 Acc: 0.9370
val Loss: 0.0092 Acc: 0.7794

Epoch 12/51
----------
train Loss: 0.0029 Acc: 0.9402
val Loss: 0.0090 Acc: 0.7794

Epoch 13/51
----------
train Loss: 0.0032 Acc: 0.9273
val Loss: 0.0089 Acc: 0.7941

Epoch 14/51
----------
train Loss: 0.0035 Acc: 0.9208
val Loss: 0.0092 Acc: 0.7941

Epoch 15/51
----------
train Loss: 0.0031 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7941

Epoch 16/51
----------
LR is set to 0.00010000000000000002
train Loss: 0.0032 Acc: 0.9289
val Loss: 0.0092 Acc: 0.7941

Epoch 17/51
----------
train Loss: 0.0031 Acc: 0.9354
val Loss: 0.0092 Acc: 0.7941

Epoch 18/51
----------
train Loss: 0.0030 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7941

Epoch 19/51
----------
train Loss: 0.0033 Acc: 0.9370
val Loss: 0.0093 Acc: 0.7941

Epoch 20/51
----------
train Loss: 0.0030 Acc: 0.9273
val Loss: 0.0091 Acc: 0.7941

Epoch 21/51
----------
train Loss: 0.0034 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7941

Epoch 22/51
----------
train Loss: 0.0031 Acc: 0.9273
val Loss: 0.0092 Acc: 0.7941

Epoch 23/51
----------
train Loss: 0.0028 Acc: 0.9402
val Loss: 0.0092 Acc: 0.7794

Epoch 24/51
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0032 Acc: 0.9321
val Loss: 0.0091 Acc: 0.7941

Epoch 25/51
----------
train Loss: 0.0030 Acc: 0.9354
val Loss: 0.0091 Acc: 0.7794

Epoch 26/51
----------
train Loss: 0.0032 Acc: 0.9354
val Loss: 0.0093 Acc: 0.7794

Epoch 27/51
----------
train Loss: 0.0030 Acc: 0.9402
val Loss: 0.0092 Acc: 0.7794

Epoch 28/51
----------
train Loss: 0.0033 Acc: 0.9257
val Loss: 0.0091 Acc: 0.7794

Epoch 29/51
----------
train Loss: 0.0038 Acc: 0.9225
val Loss: 0.0091 Acc: 0.7794

Epoch 30/51
----------
train Loss: 0.0033 Acc: 0.9370
val Loss: 0.0093 Acc: 0.7794

Epoch 31/51
----------
train Loss: 0.0034 Acc: 0.9257
val Loss: 0.0091 Acc: 0.7647

Epoch 32/51
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0035 Acc: 0.9192
val Loss: 0.0090 Acc: 0.7941

Epoch 33/51
----------
train Loss: 0.0039 Acc: 0.9160
val Loss: 0.0089 Acc: 0.7941

Epoch 34/51
----------
train Loss: 0.0030 Acc: 0.9370
val Loss: 0.0088 Acc: 0.7941

Epoch 35/51
----------
train Loss: 0.0033 Acc: 0.9354
val Loss: 0.0089 Acc: 0.7941

Epoch 36/51
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0090 Acc: 0.7941

Epoch 37/51
----------
train Loss: 0.0030 Acc: 0.9515
val Loss: 0.0091 Acc: 0.7647

Epoch 38/51
----------
train Loss: 0.0031 Acc: 0.9225
val Loss: 0.0091 Acc: 0.7941

Epoch 39/51
----------
train Loss: 0.0030 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7647

Epoch 40/51
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9176
val Loss: 0.0091 Acc: 0.7794

Epoch 41/51
----------
train Loss: 0.0032 Acc: 0.9305
val Loss: 0.0092 Acc: 0.7647

Epoch 42/51
----------
train Loss: 0.0038 Acc: 0.9192
val Loss: 0.0090 Acc: 0.7794

Epoch 43/51
----------
train Loss: 0.0033 Acc: 0.9273
val Loss: 0.0090 Acc: 0.7941

Epoch 44/51
----------
train Loss: 0.0030 Acc: 0.9338
val Loss: 0.0090 Acc: 0.7941

Epoch 45/51
----------
train Loss: 0.0029 Acc: 0.9289
val Loss: 0.0091 Acc: 0.7941

Epoch 46/51
----------
train Loss: 0.0032 Acc: 0.9208
val Loss: 0.0091 Acc: 0.7794

Epoch 47/51
----------
train Loss: 0.0031 Acc: 0.9208
val Loss: 0.0093 Acc: 0.7647

Epoch 48/51
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0034 Acc: 0.9144
val Loss: 0.0091 Acc: 0.7794

Epoch 49/51
----------
train Loss: 0.0035 Acc: 0.9063
val Loss: 0.0093 Acc: 0.7647

Epoch 50/51
----------
train Loss: 0.0034 Acc: 0.9370
val Loss: 0.0092 Acc: 0.7794

Epoch 51/51
----------
train Loss: 0.0029 Acc: 0.9241
val Loss: 0.0091 Acc: 0.7794

Training complete in 2m 33s
Best val Acc: 0.823529

---Fine tuning.---
Epoch 0/51
----------
LR is set to 0.01
train Loss: 0.0042 Acc: 0.8901
val Loss: 0.0109 Acc: 0.7647

Epoch 1/51
----------
train Loss: 0.0029 Acc: 0.9305
val Loss: 0.0102 Acc: 0.7647

Epoch 2/51
----------
train Loss: 0.0017 Acc: 0.9564
val Loss: 0.0095 Acc: 0.8088

Epoch 3/51
----------
train Loss: 0.0011 Acc: 0.9725
val Loss: 0.0128 Acc: 0.7794

Epoch 4/51
----------
train Loss: 0.0006 Acc: 0.9871
val Loss: 0.0106 Acc: 0.7647

Epoch 5/51
----------
train Loss: 0.0006 Acc: 0.9952
val Loss: 0.0089 Acc: 0.8235

Epoch 6/51
----------
train Loss: 0.0004 Acc: 0.9935
val Loss: 0.0074 Acc: 0.8235

Epoch 7/51
----------
train Loss: 0.0004 Acc: 0.9919
val Loss: 0.0077 Acc: 0.8529

Epoch 8/51
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8529

Epoch 9/51
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0076 Acc: 0.8382

Epoch 10/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8382

Epoch 11/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8382

Epoch 12/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8235

Epoch 13/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8235

Epoch 14/51
----------
train Loss: 0.0003 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8235

Epoch 15/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8235

Epoch 16/51
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8235

Epoch 17/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 18/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 19/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8235

Epoch 20/51
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0083 Acc: 0.8235

Epoch 21/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8235

Epoch 22/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8235

Epoch 23/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0082 Acc: 0.8235

Epoch 24/51
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0081 Acc: 0.8235

Epoch 25/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8235

Epoch 26/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8235

Epoch 27/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 28/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8235

Epoch 29/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0077 Acc: 0.8235

Epoch 30/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8382

Epoch 31/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 32/51
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8382

Epoch 33/51
----------
train Loss: 0.0002 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8382

Epoch 34/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8235

Epoch 35/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8235

Epoch 36/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8235

Epoch 37/51
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8235

Epoch 38/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0078 Acc: 0.8382

Epoch 39/51
----------
train Loss: 0.0004 Acc: 0.9968
val Loss: 0.0078 Acc: 0.8382

Epoch 40/51
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0080 Acc: 0.8235

Epoch 41/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 42/51
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0077 Acc: 0.8382

Epoch 43/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8382

Epoch 44/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8235

Epoch 45/51
----------
train Loss: 0.0005 Acc: 0.9968
val Loss: 0.0079 Acc: 0.8235

Epoch 46/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8235

Epoch 47/51
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0079 Acc: 0.8235

Epoch 48/51
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0083 Acc: 0.8235

Epoch 49/51
----------
train Loss: 0.0002 Acc: 0.9968
val Loss: 0.0080 Acc: 0.8235

Epoch 50/51
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8235

Epoch 51/51
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8235

Training complete in 2m 42s
Best val Acc: 0.852941

---Testing---
Test accuracy: 0.969432
--------------------
Accuracy of Carcharhiniformes : 88 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 92 %
Accuracy of Lamniformes : 96 %
Accuracy of Orectolobiformes : 100 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 97 %
mean: 0.9660087719298246, std: 0.04121778053753926
--------------------

run info[val: 0.15, epoch: 77, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/76
----------
LR is set to 0.01
train Loss: 0.0203 Acc: 0.2586
val Loss: 0.0360 Acc: 0.3592

Epoch 1/76
----------
train Loss: 0.0132 Acc: 0.5753
val Loss: 0.0201 Acc: 0.6505

Epoch 2/76
----------
train Loss: 0.0081 Acc: 0.7757
val Loss: 0.0183 Acc: 0.6893

Epoch 3/76
----------
train Loss: 0.0056 Acc: 0.8339
val Loss: 0.0125 Acc: 0.7961

Epoch 4/76
----------
train Loss: 0.0043 Acc: 0.8630
val Loss: 0.0330 Acc: 0.7573

Epoch 5/76
----------
train Loss: 0.0037 Acc: 0.9075
val Loss: 0.0103 Acc: 0.8155

Epoch 6/76
----------
train Loss: 0.0031 Acc: 0.9110
val Loss: 0.0066 Acc: 0.7767

Epoch 7/76
----------
train Loss: 0.0030 Acc: 0.9298
val Loss: 0.0161 Acc: 0.7864

Epoch 8/76
----------
train Loss: 0.0026 Acc: 0.9264
val Loss: 0.0101 Acc: 0.7864

Epoch 9/76
----------
train Loss: 0.0025 Acc: 0.9435
val Loss: 0.0071 Acc: 0.8058

Epoch 10/76
----------
LR is set to 0.001
train Loss: 0.0022 Acc: 0.9572
val Loss: 0.0082 Acc: 0.8058

Epoch 11/76
----------
train Loss: 0.0022 Acc: 0.9486
val Loss: 0.0148 Acc: 0.7961

Epoch 12/76
----------
train Loss: 0.0021 Acc: 0.9692
val Loss: 0.0067 Acc: 0.8058

Epoch 13/76
----------
train Loss: 0.0020 Acc: 0.9572
val Loss: 0.0078 Acc: 0.8058

Epoch 14/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0062 Acc: 0.7961

Epoch 15/76
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0116 Acc: 0.8058

Epoch 16/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0107 Acc: 0.8058

Epoch 17/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0061 Acc: 0.7961

Epoch 18/76
----------
train Loss: 0.0021 Acc: 0.9589
val Loss: 0.0085 Acc: 0.8058

Epoch 19/76
----------
train Loss: 0.0019 Acc: 0.9640
val Loss: 0.0096 Acc: 0.7961

Epoch 20/76
----------
LR is set to 0.00010000000000000002
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0066 Acc: 0.8058

Epoch 21/76
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0062 Acc: 0.7961

Epoch 22/76
----------
train Loss: 0.0020 Acc: 0.9743
val Loss: 0.0109 Acc: 0.7961

Epoch 23/76
----------
train Loss: 0.0021 Acc: 0.9692
val Loss: 0.0175 Acc: 0.8058

Epoch 24/76
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0107 Acc: 0.7961

Epoch 25/76
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0122 Acc: 0.7961

Epoch 26/76
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0124 Acc: 0.8058

Epoch 27/76
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0105 Acc: 0.8058

Epoch 28/76
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0061 Acc: 0.8058

Epoch 29/76
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0108 Acc: 0.7961

Epoch 30/76
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0140 Acc: 0.8058

Epoch 31/76
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0319 Acc: 0.8058

Epoch 32/76
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0108 Acc: 0.8058

Epoch 33/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0086 Acc: 0.8058

Epoch 34/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0092 Acc: 0.8058

Epoch 35/76
----------
train Loss: 0.0019 Acc: 0.9640
val Loss: 0.0067 Acc: 0.8058

Epoch 36/76
----------
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0103 Acc: 0.8058

Epoch 37/76
----------
train Loss: 0.0019 Acc: 0.9726
val Loss: 0.0071 Acc: 0.8058

Epoch 38/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0249 Acc: 0.8058

Epoch 39/76
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0061 Acc: 0.8058

Epoch 40/76
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9726
val Loss: 0.0062 Acc: 0.8058

Epoch 41/76
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0202 Acc: 0.8058

Epoch 42/76
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0074 Acc: 0.8058

Epoch 43/76
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0075 Acc: 0.8058

Epoch 44/76
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0086 Acc: 0.8058

Epoch 45/76
----------
train Loss: 0.0020 Acc: 0.9675
val Loss: 0.0141 Acc: 0.8058

Epoch 46/76
----------
train Loss: 0.0021 Acc: 0.9658
val Loss: 0.0131 Acc: 0.8058

Epoch 47/76
----------
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0085 Acc: 0.8058

Epoch 48/76
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0063 Acc: 0.8058

Epoch 49/76
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0077 Acc: 0.8058

Epoch 50/76
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0111 Acc: 0.8058

Epoch 51/76
----------
train Loss: 0.0021 Acc: 0.9623
val Loss: 0.0081 Acc: 0.8058

Epoch 52/76
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0169 Acc: 0.8058

Epoch 53/76
----------
train Loss: 0.0020 Acc: 0.9743
val Loss: 0.0068 Acc: 0.8058

Epoch 54/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0087 Acc: 0.8058

Epoch 55/76
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0067 Acc: 0.8058

Epoch 56/76
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0070 Acc: 0.8058

Epoch 57/76
----------
train Loss: 0.0020 Acc: 0.9623
val Loss: 0.0070 Acc: 0.8058

Epoch 58/76
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0084 Acc: 0.8058

Epoch 59/76
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0091 Acc: 0.8058

Epoch 60/76
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0225 Acc: 0.8058

Epoch 61/76
----------
train Loss: 0.0020 Acc: 0.9606
val Loss: 0.0065 Acc: 0.8058

Epoch 62/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0109 Acc: 0.8058

Epoch 63/76
----------
train Loss: 0.0020 Acc: 0.9692
val Loss: 0.0309 Acc: 0.8058

Epoch 64/76
----------
train Loss: 0.0020 Acc: 0.9521
val Loss: 0.0141 Acc: 0.8058

Epoch 65/76
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0249 Acc: 0.8058

Epoch 66/76
----------
train Loss: 0.0020 Acc: 0.9658
val Loss: 0.0210 Acc: 0.8058

Epoch 67/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0060 Acc: 0.8058

Epoch 68/76
----------
train Loss: 0.0021 Acc: 0.9606
val Loss: 0.0147 Acc: 0.8058

Epoch 69/76
----------
train Loss: 0.0019 Acc: 0.9692
val Loss: 0.0110 Acc: 0.8058

Epoch 70/76
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0020 Acc: 0.9743
val Loss: 0.0066 Acc: 0.8058

Epoch 71/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0068 Acc: 0.8058

Epoch 72/76
----------
train Loss: 0.0020 Acc: 0.9726
val Loss: 0.0136 Acc: 0.8058

Epoch 73/76
----------
train Loss: 0.0019 Acc: 0.9675
val Loss: 0.0085 Acc: 0.8058

Epoch 74/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0074 Acc: 0.8058

Epoch 75/76
----------
train Loss: 0.0021 Acc: 0.9640
val Loss: 0.0170 Acc: 0.8058

Epoch 76/76
----------
train Loss: 0.0019 Acc: 0.9658
val Loss: 0.0166 Acc: 0.8058

Training complete in 3m 52s
Best val Acc: 0.815534

---Fine tuning.---
Epoch 0/76
----------
LR is set to 0.01
train Loss: 0.0032 Acc: 0.9195
val Loss: 0.0080 Acc: 0.8350

Epoch 1/76
----------
train Loss: 0.0015 Acc: 0.9726
val Loss: 0.0077 Acc: 0.8350

Epoch 2/76
----------
train Loss: 0.0007 Acc: 0.9932
val Loss: 0.0048 Acc: 0.8350

Epoch 3/76
----------
train Loss: 0.0003 Acc: 0.9983
val Loss: 0.0048 Acc: 0.8155

Epoch 4/76
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8252

Epoch 5/76
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0188 Acc: 0.8350

Epoch 6/76
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8544

Epoch 7/76
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0049 Acc: 0.8641

Epoch 8/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8447

Epoch 9/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8544

Epoch 10/76
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8544

Epoch 11/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0152 Acc: 0.8544

Epoch 12/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8544

Epoch 13/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8544

Epoch 14/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0104 Acc: 0.8544

Epoch 15/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0186 Acc: 0.8544

Epoch 16/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8641

Epoch 17/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0189 Acc: 0.8641

Epoch 18/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0087 Acc: 0.8641

Epoch 19/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8641

Epoch 20/76
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 21/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0059 Acc: 0.8641

Epoch 22/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0190 Acc: 0.8641

Epoch 23/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 24/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0190 Acc: 0.8641

Epoch 25/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 26/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0197 Acc: 0.8641

Epoch 27/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0135 Acc: 0.8641

Epoch 28/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8641

Epoch 29/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8641

Epoch 30/76
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 31/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8641

Epoch 32/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 33/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0263 Acc: 0.8641

Epoch 34/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0062 Acc: 0.8641

Epoch 35/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0108 Acc: 0.8641

Epoch 36/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8641

Epoch 37/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8641

Epoch 38/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8641

Epoch 39/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 40/76
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8641

Epoch 41/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0144 Acc: 0.8641

Epoch 42/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 43/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 44/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0154 Acc: 0.8641

Epoch 45/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0197 Acc: 0.8641

Epoch 46/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 47/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 48/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0215 Acc: 0.8641

Epoch 49/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 50/76
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 51/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0154 Acc: 0.8641

Epoch 52/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 53/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0115 Acc: 0.8641

Epoch 54/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0103 Acc: 0.8641

Epoch 55/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8641

Epoch 56/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 57/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 58/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 59/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 60/76
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 61/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 62/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8641

Epoch 63/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8641

Epoch 64/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 65/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0164 Acc: 0.8641

Epoch 66/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 67/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 68/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8641

Epoch 69/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0304 Acc: 0.8641

Epoch 70/76
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 71/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 72/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8641

Epoch 73/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 74/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0111 Acc: 0.8641

Epoch 75/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8641

Epoch 76/76
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8641

Training complete in 4m 6s
Best val Acc: 0.864078

---Testing---
Test accuracy: 0.979622
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9713274663932558, std: 0.029455169683012646
--------------------

run info[val: 0.2, epoch: 87, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0224 Acc: 0.2273
val Loss: 0.0242 Acc: 0.4161

Epoch 1/86
----------
train Loss: 0.0138 Acc: 0.5636
val Loss: 0.0158 Acc: 0.6496

Epoch 2/86
----------
train Loss: 0.0087 Acc: 0.7855
val Loss: 0.0116 Acc: 0.7372

Epoch 3/86
----------
LR is set to 0.001
train Loss: 0.0066 Acc: 0.8018
val Loss: 0.0115 Acc: 0.7591

Epoch 4/86
----------
train Loss: 0.0062 Acc: 0.8400
val Loss: 0.0111 Acc: 0.7445

Epoch 5/86
----------
train Loss: 0.0060 Acc: 0.8727
val Loss: 0.0102 Acc: 0.7299

Epoch 6/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0058 Acc: 0.8709
val Loss: 0.0115 Acc: 0.7372

Epoch 7/86
----------
train Loss: 0.0059 Acc: 0.8691
val Loss: 0.0104 Acc: 0.7445

Epoch 8/86
----------
train Loss: 0.0059 Acc: 0.8618
val Loss: 0.0102 Acc: 0.7372

Epoch 9/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0059 Acc: 0.8527
val Loss: 0.0110 Acc: 0.7372

Epoch 10/86
----------
train Loss: 0.0058 Acc: 0.8691
val Loss: 0.0105 Acc: 0.7445

Epoch 11/86
----------
train Loss: 0.0057 Acc: 0.8600
val Loss: 0.0108 Acc: 0.7372

Epoch 12/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0059 Acc: 0.8564
val Loss: 0.0103 Acc: 0.7518

Epoch 13/86
----------
train Loss: 0.0060 Acc: 0.8509
val Loss: 0.0106 Acc: 0.7372

Epoch 14/86
----------
train Loss: 0.0058 Acc: 0.8636
val Loss: 0.0113 Acc: 0.7226

Epoch 15/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0059 Acc: 0.8709
val Loss: 0.0112 Acc: 0.7299

Epoch 16/86
----------
train Loss: 0.0059 Acc: 0.8618
val Loss: 0.0104 Acc: 0.7299

Epoch 17/86
----------
train Loss: 0.0058 Acc: 0.8582
val Loss: 0.0111 Acc: 0.7372

Epoch 18/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0059 Acc: 0.8636
val Loss: 0.0107 Acc: 0.7518

Epoch 19/86
----------
train Loss: 0.0058 Acc: 0.8709
val Loss: 0.0105 Acc: 0.7445

Epoch 20/86
----------
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0111 Acc: 0.7226

Epoch 21/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0106 Acc: 0.7299

Epoch 22/86
----------
train Loss: 0.0056 Acc: 0.8655
val Loss: 0.0103 Acc: 0.7445

Epoch 23/86
----------
train Loss: 0.0060 Acc: 0.8673
val Loss: 0.0101 Acc: 0.7372

Epoch 24/86
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0109 Acc: 0.7445

Epoch 25/86
----------
train Loss: 0.0058 Acc: 0.8545
val Loss: 0.0104 Acc: 0.7372

Epoch 26/86
----------
train Loss: 0.0057 Acc: 0.8673
val Loss: 0.0115 Acc: 0.7299

Epoch 27/86
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0061 Acc: 0.8545
val Loss: 0.0105 Acc: 0.7372

Epoch 28/86
----------
train Loss: 0.0059 Acc: 0.8564
val Loss: 0.0102 Acc: 0.7299

Epoch 29/86
----------
train Loss: 0.0058 Acc: 0.8636
val Loss: 0.0105 Acc: 0.7372

Epoch 30/86
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0058 Acc: 0.8636
val Loss: 0.0105 Acc: 0.7372

Epoch 31/86
----------
train Loss: 0.0060 Acc: 0.8582
val Loss: 0.0107 Acc: 0.7299

Epoch 32/86
----------
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0104 Acc: 0.7372

Epoch 33/86
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0059 Acc: 0.8727
val Loss: 0.0111 Acc: 0.7299

Epoch 34/86
----------
train Loss: 0.0058 Acc: 0.8727
val Loss: 0.0113 Acc: 0.7226

Epoch 35/86
----------
train Loss: 0.0060 Acc: 0.8600
val Loss: 0.0103 Acc: 0.7226

Epoch 36/86
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0059 Acc: 0.8782
val Loss: 0.0108 Acc: 0.7372

Epoch 37/86
----------
train Loss: 0.0059 Acc: 0.8564
val Loss: 0.0101 Acc: 0.7445

Epoch 38/86
----------
train Loss: 0.0058 Acc: 0.8800
val Loss: 0.0104 Acc: 0.7299

Epoch 39/86
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0061 Acc: 0.8564
val Loss: 0.0107 Acc: 0.7372

Epoch 40/86
----------
train Loss: 0.0057 Acc: 0.8709
val Loss: 0.0103 Acc: 0.7372

Epoch 41/86
----------
train Loss: 0.0058 Acc: 0.8764
val Loss: 0.0098 Acc: 0.7445

Epoch 42/86
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0059 Acc: 0.8745
val Loss: 0.0115 Acc: 0.7299

Epoch 43/86
----------
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0109 Acc: 0.7372

Epoch 44/86
----------
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0114 Acc: 0.7372

Epoch 45/86
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0059 Acc: 0.8800
val Loss: 0.0106 Acc: 0.7445

Epoch 46/86
----------
train Loss: 0.0059 Acc: 0.8691
val Loss: 0.0107 Acc: 0.7372

Epoch 47/86
----------
train Loss: 0.0060 Acc: 0.8600
val Loss: 0.0100 Acc: 0.7372

Epoch 48/86
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0059 Acc: 0.8618
val Loss: 0.0110 Acc: 0.7445

Epoch 49/86
----------
train Loss: 0.0060 Acc: 0.8655
val Loss: 0.0099 Acc: 0.7445

Epoch 50/86
----------
train Loss: 0.0058 Acc: 0.8636
val Loss: 0.0105 Acc: 0.7372

Epoch 51/86
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0059 Acc: 0.8745
val Loss: 0.0102 Acc: 0.7372

Epoch 52/86
----------
train Loss: 0.0058 Acc: 0.8618
val Loss: 0.0103 Acc: 0.7445

Epoch 53/86
----------
train Loss: 0.0057 Acc: 0.8691
val Loss: 0.0113 Acc: 0.7372

Epoch 54/86
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0058 Acc: 0.8727
val Loss: 0.0116 Acc: 0.7372

Epoch 55/86
----------
train Loss: 0.0060 Acc: 0.8545
val Loss: 0.0109 Acc: 0.7299

Epoch 56/86
----------
train Loss: 0.0058 Acc: 0.8600
val Loss: 0.0107 Acc: 0.7372

Epoch 57/86
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0058 Acc: 0.8745
val Loss: 0.0107 Acc: 0.7445

Epoch 58/86
----------
train Loss: 0.0057 Acc: 0.8782
val Loss: 0.0109 Acc: 0.7372

Epoch 59/86
----------
train Loss: 0.0056 Acc: 0.8673
val Loss: 0.0107 Acc: 0.7299

Epoch 60/86
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0059 Acc: 0.8709
val Loss: 0.0119 Acc: 0.7299

Epoch 61/86
----------
train Loss: 0.0058 Acc: 0.8691
val Loss: 0.0112 Acc: 0.7299

Epoch 62/86
----------
train Loss: 0.0060 Acc: 0.8691
val Loss: 0.0114 Acc: 0.7299

Epoch 63/86
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0059 Acc: 0.8618
val Loss: 0.0104 Acc: 0.7372

Epoch 64/86
----------
train Loss: 0.0058 Acc: 0.8709
val Loss: 0.0106 Acc: 0.7372

Epoch 65/86
----------
train Loss: 0.0060 Acc: 0.8636
val Loss: 0.0107 Acc: 0.7372

Epoch 66/86
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0060 Acc: 0.8764
val Loss: 0.0111 Acc: 0.7299

Epoch 67/86
----------
train Loss: 0.0059 Acc: 0.8636
val Loss: 0.0105 Acc: 0.7299

Epoch 68/86
----------
train Loss: 0.0060 Acc: 0.8491
val Loss: 0.0105 Acc: 0.7299

Epoch 69/86
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0057 Acc: 0.8782
val Loss: 0.0112 Acc: 0.7299

Epoch 70/86
----------
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0109 Acc: 0.7372

Epoch 71/86
----------
train Loss: 0.0060 Acc: 0.8582
val Loss: 0.0108 Acc: 0.7445

Epoch 72/86
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0059 Acc: 0.8545
val Loss: 0.0106 Acc: 0.7445

Epoch 73/86
----------
train Loss: 0.0057 Acc: 0.8909
val Loss: 0.0111 Acc: 0.7372

Epoch 74/86
----------
train Loss: 0.0056 Acc: 0.8673
val Loss: 0.0100 Acc: 0.7299

Epoch 75/86
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0060 Acc: 0.8655
val Loss: 0.0100 Acc: 0.7445

Epoch 76/86
----------
train Loss: 0.0058 Acc: 0.8709
val Loss: 0.0108 Acc: 0.7372

Epoch 77/86
----------
train Loss: 0.0059 Acc: 0.8855
val Loss: 0.0111 Acc: 0.7226

Epoch 78/86
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0059 Acc: 0.8727
val Loss: 0.0108 Acc: 0.7226

Epoch 79/86
----------
train Loss: 0.0058 Acc: 0.8709
val Loss: 0.0120 Acc: 0.7299

Epoch 80/86
----------
train Loss: 0.0061 Acc: 0.8655
val Loss: 0.0108 Acc: 0.7226

Epoch 81/86
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0059 Acc: 0.8655
val Loss: 0.0109 Acc: 0.7299

Epoch 82/86
----------
train Loss: 0.0059 Acc: 0.8527
val Loss: 0.0100 Acc: 0.7299

Epoch 83/86
----------
train Loss: 0.0060 Acc: 0.8618
val Loss: 0.0108 Acc: 0.7372

Epoch 84/86
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0059 Acc: 0.8745
val Loss: 0.0106 Acc: 0.7372

Epoch 85/86
----------
train Loss: 0.0056 Acc: 0.8764
val Loss: 0.0109 Acc: 0.7372

Epoch 86/86
----------
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0103 Acc: 0.7372

Training complete in 4m 18s
Best val Acc: 0.759124

---Fine tuning.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0057 Acc: 0.8691
val Loss: 0.0087 Acc: 0.8175

Epoch 1/86
----------
train Loss: 0.0029 Acc: 0.9600
val Loss: 0.0071 Acc: 0.8321

Epoch 2/86
----------
train Loss: 0.0014 Acc: 0.9782
val Loss: 0.0064 Acc: 0.8467

Epoch 3/86
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9909
val Loss: 0.0058 Acc: 0.8540

Epoch 4/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0053 Acc: 0.8613

Epoch 5/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0049 Acc: 0.8540

Epoch 6/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0055 Acc: 0.8540

Epoch 7/86
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0050 Acc: 0.8540

Epoch 8/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0068 Acc: 0.8540

Epoch 9/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8540

Epoch 10/86
----------
train Loss: 0.0007 Acc: 0.9982
val Loss: 0.0048 Acc: 0.8540

Epoch 11/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8540

Epoch 12/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8540

Epoch 13/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0053 Acc: 0.8540

Epoch 14/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0053 Acc: 0.8540

Epoch 15/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0056 Acc: 0.8540

Epoch 16/86
----------
train Loss: 0.0005 Acc: 0.9927
val Loss: 0.0049 Acc: 0.8540

Epoch 17/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8540

Epoch 18/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8467

Epoch 19/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0047 Acc: 0.8467

Epoch 20/86
----------
train Loss: 0.0007 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8467

Epoch 21/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0057 Acc: 0.8467

Epoch 22/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0053 Acc: 0.8540

Epoch 23/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8540

Epoch 24/86
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0052 Acc: 0.8540

Epoch 25/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0048 Acc: 0.8540

Epoch 26/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0061 Acc: 0.8540

Epoch 27/86
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8540

Epoch 28/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8540

Epoch 29/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0044 Acc: 0.8540

Epoch 30/86
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8540

Epoch 31/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8540

Epoch 32/86
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0050 Acc: 0.8540

Epoch 33/86
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8540

Epoch 34/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0054 Acc: 0.8540

Epoch 35/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0051 Acc: 0.8467

Epoch 36/86
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8540

Epoch 37/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0056 Acc: 0.8467

Epoch 38/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8540

Epoch 39/86
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0047 Acc: 0.8540

Epoch 40/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8540

Epoch 41/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8540

Epoch 42/86
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0055 Acc: 0.8467

Epoch 43/86
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0060 Acc: 0.8540

Epoch 44/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8467

Epoch 45/86
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0007 Acc: 0.9945
val Loss: 0.0052 Acc: 0.8540

Epoch 46/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0045 Acc: 0.8540

Epoch 47/86
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0047 Acc: 0.8467

Epoch 48/86
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0062 Acc: 0.8467

Epoch 49/86
----------
train Loss: 0.0007 Acc: 0.9964
val Loss: 0.0056 Acc: 0.8467

Epoch 50/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8467

Epoch 51/86
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0052 Acc: 0.8467

Epoch 52/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0051 Acc: 0.8467

Epoch 53/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0049 Acc: 0.8467

Epoch 54/86
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0059 Acc: 0.8467

Epoch 55/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0045 Acc: 0.8467

Epoch 56/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0056 Acc: 0.8467

Epoch 57/86
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0057 Acc: 0.8540

Epoch 58/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8540

Epoch 59/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0055 Acc: 0.8540

Epoch 60/86
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0059 Acc: 0.8540

Epoch 61/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8613

Epoch 62/86
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0060 Acc: 0.8540

Epoch 63/86
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0061 Acc: 0.8613

Epoch 64/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0058 Acc: 0.8540

Epoch 65/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8467

Epoch 66/86
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0056 Acc: 0.8467

Epoch 67/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0049 Acc: 0.8540

Epoch 68/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0055 Acc: 0.8540

Epoch 69/86
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8540

Epoch 70/86
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8540

Epoch 71/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0053 Acc: 0.8540

Epoch 72/86
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8540

Epoch 73/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8540

Epoch 74/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0057 Acc: 0.8540

Epoch 75/86
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0051 Acc: 0.8467

Epoch 76/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0048 Acc: 0.8467

Epoch 77/86
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0060 Acc: 0.8467

Epoch 78/86
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0055 Acc: 0.8613

Epoch 79/86
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0055 Acc: 0.8613

Epoch 80/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8467

Epoch 81/86
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0006 Acc: 0.9909
val Loss: 0.0059 Acc: 0.8467

Epoch 82/86
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0058 Acc: 0.8540

Epoch 83/86
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8467

Epoch 84/86
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0046 Acc: 0.8540

Epoch 85/86
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0046 Acc: 0.8540

Epoch 86/86
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8467

Training complete in 4m 32s
Best val Acc: 0.861314

---Testing---
Test accuracy: 0.970888
--------------------
Accuracy of Carcharhiniformes : 97 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 94 %
mean: 0.9638156078879321, std: 0.029984393944860924
--------------------

run info[val: 0.25, epoch: 57, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0229 Acc: 0.2481
val Loss: 0.0203 Acc: 0.3041

Epoch 1/56
----------
train Loss: 0.0160 Acc: 0.5426
val Loss: 0.0136 Acc: 0.6608

Epoch 2/56
----------
train Loss: 0.0113 Acc: 0.6977
val Loss: 0.0096 Acc: 0.7310

Epoch 3/56
----------
train Loss: 0.0070 Acc: 0.8023
val Loss: 0.0074 Acc: 0.7836

Epoch 4/56
----------
train Loss: 0.0060 Acc: 0.8469
val Loss: 0.0073 Acc: 0.8129

Epoch 5/56
----------
train Loss: 0.0054 Acc: 0.8450
val Loss: 0.0070 Acc: 0.8304

Epoch 6/56
----------
train Loss: 0.0040 Acc: 0.9012
val Loss: 0.0075 Acc: 0.7895

Epoch 7/56
----------
train Loss: 0.0039 Acc: 0.8973
val Loss: 0.0067 Acc: 0.8363

Epoch 8/56
----------
train Loss: 0.0029 Acc: 0.9322
val Loss: 0.0061 Acc: 0.8363

Epoch 9/56
----------
train Loss: 0.0027 Acc: 0.9322
val Loss: 0.0062 Acc: 0.8421

Epoch 10/56
----------
train Loss: 0.0026 Acc: 0.9477
val Loss: 0.0062 Acc: 0.8480

Epoch 11/56
----------
train Loss: 0.0025 Acc: 0.9477
val Loss: 0.0061 Acc: 0.8363

Epoch 12/56
----------
LR is set to 0.001
train Loss: 0.0025 Acc: 0.9593
val Loss: 0.0060 Acc: 0.8480

Epoch 13/56
----------
train Loss: 0.0020 Acc: 0.9593
val Loss: 0.0057 Acc: 0.8421

Epoch 14/56
----------
train Loss: 0.0023 Acc: 0.9651
val Loss: 0.0056 Acc: 0.8421

Epoch 15/56
----------
train Loss: 0.0027 Acc: 0.9574
val Loss: 0.0057 Acc: 0.8421

Epoch 16/56
----------
train Loss: 0.0020 Acc: 0.9690
val Loss: 0.0059 Acc: 0.8480

Epoch 17/56
----------
train Loss: 0.0022 Acc: 0.9535
val Loss: 0.0058 Acc: 0.8480

Epoch 18/56
----------
train Loss: 0.0024 Acc: 0.9690
val Loss: 0.0058 Acc: 0.8538

Epoch 19/56
----------
train Loss: 0.0022 Acc: 0.9690
val Loss: 0.0059 Acc: 0.8538

Epoch 20/56
----------
train Loss: 0.0023 Acc: 0.9632
val Loss: 0.0058 Acc: 0.8596

Epoch 21/56
----------
train Loss: 0.0021 Acc: 0.9671
val Loss: 0.0061 Acc: 0.8480

Epoch 22/56
----------
train Loss: 0.0024 Acc: 0.9690
val Loss: 0.0061 Acc: 0.8421

Epoch 23/56
----------
train Loss: 0.0020 Acc: 0.9748
val Loss: 0.0056 Acc: 0.8421

Epoch 24/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8421

Epoch 25/56
----------
train Loss: 0.0026 Acc: 0.9651
val Loss: 0.0058 Acc: 0.8421

Epoch 26/56
----------
train Loss: 0.0023 Acc: 0.9748
val Loss: 0.0057 Acc: 0.8421

Epoch 27/56
----------
train Loss: 0.0027 Acc: 0.9671
val Loss: 0.0059 Acc: 0.8421

Epoch 28/56
----------
train Loss: 0.0024 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8363

Epoch 29/56
----------
train Loss: 0.0018 Acc: 0.9690
val Loss: 0.0061 Acc: 0.8421

Epoch 30/56
----------
train Loss: 0.0025 Acc: 0.9690
val Loss: 0.0060 Acc: 0.8363

Epoch 31/56
----------
train Loss: 0.0021 Acc: 0.9748
val Loss: 0.0061 Acc: 0.8363

Epoch 32/56
----------
train Loss: 0.0021 Acc: 0.9748
val Loss: 0.0059 Acc: 0.8421

Epoch 33/56
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0059 Acc: 0.8421

Epoch 34/56
----------
train Loss: 0.0020 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8480

Epoch 35/56
----------
train Loss: 0.0023 Acc: 0.9574
val Loss: 0.0058 Acc: 0.8421

Epoch 36/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0019 Acc: 0.9729
val Loss: 0.0056 Acc: 0.8421

Epoch 37/56
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0057 Acc: 0.8421

Epoch 38/56
----------
train Loss: 0.0019 Acc: 0.9651
val Loss: 0.0055 Acc: 0.8421

Epoch 39/56
----------
train Loss: 0.0020 Acc: 0.9748
val Loss: 0.0056 Acc: 0.8480

Epoch 40/56
----------
train Loss: 0.0021 Acc: 0.9690
val Loss: 0.0060 Acc: 0.8363

Epoch 41/56
----------
train Loss: 0.0019 Acc: 0.9748
val Loss: 0.0058 Acc: 0.8363

Epoch 42/56
----------
train Loss: 0.0020 Acc: 0.9632
val Loss: 0.0058 Acc: 0.8421

Epoch 43/56
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0059 Acc: 0.8421

Epoch 44/56
----------
train Loss: 0.0020 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8421

Epoch 45/56
----------
train Loss: 0.0020 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8480

Epoch 46/56
----------
train Loss: 0.0019 Acc: 0.9632
val Loss: 0.0057 Acc: 0.8538

Epoch 47/56
----------
train Loss: 0.0020 Acc: 0.9729
val Loss: 0.0058 Acc: 0.8421

Epoch 48/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0055 Acc: 0.8421

Epoch 49/56
----------
train Loss: 0.0028 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8538

Epoch 50/56
----------
train Loss: 0.0021 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8538

Epoch 51/56
----------
train Loss: 0.0022 Acc: 0.9729
val Loss: 0.0058 Acc: 0.8538

Epoch 52/56
----------
train Loss: 0.0024 Acc: 0.9729
val Loss: 0.0060 Acc: 0.8421

Epoch 53/56
----------
train Loss: 0.0023 Acc: 0.9748
val Loss: 0.0057 Acc: 0.8363

Epoch 54/56
----------
train Loss: 0.0020 Acc: 0.9748
val Loss: 0.0060 Acc: 0.8421

Epoch 55/56
----------
train Loss: 0.0027 Acc: 0.9690
val Loss: 0.0060 Acc: 0.8480

Epoch 56/56
----------
train Loss: 0.0023 Acc: 0.9671
val Loss: 0.0058 Acc: 0.8480

Training complete in 2m 48s
Best val Acc: 0.859649

---Fine tuning.---
Epoch 0/56
----------
LR is set to 0.01
train Loss: 0.0026 Acc: 0.9612
val Loss: 0.0055 Acc: 0.8655

Epoch 1/56
----------
train Loss: 0.0015 Acc: 0.9671
val Loss: 0.0083 Acc: 0.8012

Epoch 2/56
----------
train Loss: 0.0023 Acc: 0.9767
val Loss: 0.0052 Acc: 0.8772

Epoch 3/56
----------
train Loss: 0.0011 Acc: 0.9729
val Loss: 0.0057 Acc: 0.8596

Epoch 4/56
----------
train Loss: 0.0006 Acc: 0.9884
val Loss: 0.0054 Acc: 0.8538

Epoch 5/56
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0050 Acc: 0.8713

Epoch 6/56
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8655

Epoch 7/56
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0048 Acc: 0.8772

Epoch 8/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8830

Epoch 9/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8655

Epoch 10/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8889

Epoch 11/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9064

Epoch 12/56
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9064

Epoch 13/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 14/56
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8947

Epoch 15/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9064

Epoch 16/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9006

Epoch 17/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 18/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 19/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9006

Epoch 20/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9064

Epoch 21/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 22/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 23/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9064

Epoch 24/56
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8947

Epoch 25/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 26/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9064

Epoch 27/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9006

Epoch 28/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9006

Epoch 29/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8947

Epoch 30/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8947

Epoch 31/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9006

Epoch 32/56
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0046 Acc: 0.9064

Epoch 33/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 34/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9064

Epoch 35/56
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0045 Acc: 0.9006

Epoch 36/56
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 37/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 38/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 39/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.9006

Epoch 40/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8947

Epoch 41/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 42/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9123

Epoch 43/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9123

Epoch 44/56
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0043 Acc: 0.8889

Epoch 45/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8947

Epoch 46/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 47/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8947

Epoch 48/56
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8889

Epoch 49/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8947

Epoch 50/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8889

Epoch 51/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.9064

Epoch 52/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0043 Acc: 0.9006

Epoch 53/56
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.9006

Epoch 54/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Epoch 55/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.9006

Epoch 56/56
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.9006

Training complete in 2m 58s
Best val Acc: 0.912281

---Testing---
Test accuracy: 0.978166
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 92 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 97 %
mean: 0.9729590239326198, std: 0.01991039380033045
--------------------

run info[val: 0.3, epoch: 61, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0207 Acc: 0.2536
val Loss: 0.0233 Acc: 0.4029

Epoch 1/60
----------
train Loss: 0.0154 Acc: 0.5073
val Loss: 0.0171 Acc: 0.6845

Epoch 2/60
----------
train Loss: 0.0107 Acc: 0.7027
val Loss: 0.0123 Acc: 0.7184

Epoch 3/60
----------
train Loss: 0.0080 Acc: 0.7235
val Loss: 0.0096 Acc: 0.7913

Epoch 4/60
----------
train Loss: 0.0059 Acc: 0.8337
val Loss: 0.0086 Acc: 0.8301

Epoch 5/60
----------
train Loss: 0.0050 Acc: 0.8586
val Loss: 0.0066 Acc: 0.8641

Epoch 6/60
----------
train Loss: 0.0044 Acc: 0.8628
val Loss: 0.0086 Acc: 0.8301

Epoch 7/60
----------
train Loss: 0.0040 Acc: 0.8836
val Loss: 0.0069 Acc: 0.8738

Epoch 8/60
----------
train Loss: 0.0036 Acc: 0.9127
val Loss: 0.0065 Acc: 0.8689

Epoch 9/60
----------
train Loss: 0.0033 Acc: 0.9231
val Loss: 0.0112 Acc: 0.8447

Epoch 10/60
----------
train Loss: 0.0032 Acc: 0.9210
val Loss: 0.0050 Acc: 0.8738

Epoch 11/60
----------
train Loss: 0.0031 Acc: 0.9210
val Loss: 0.0050 Acc: 0.8544

Epoch 12/60
----------
train Loss: 0.0026 Acc: 0.9376
val Loss: 0.0081 Acc: 0.8641

Epoch 13/60
----------
LR is set to 0.001
train Loss: 0.0027 Acc: 0.9356
val Loss: 0.0062 Acc: 0.8738

Epoch 14/60
----------
train Loss: 0.0026 Acc: 0.9522
val Loss: 0.0052 Acc: 0.8689

Epoch 15/60
----------
train Loss: 0.0026 Acc: 0.9480
val Loss: 0.0080 Acc: 0.8641

Epoch 16/60
----------
train Loss: 0.0025 Acc: 0.9418
val Loss: 0.0087 Acc: 0.8641

Epoch 17/60
----------
train Loss: 0.0024 Acc: 0.9418
val Loss: 0.0072 Acc: 0.8641

Epoch 18/60
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0066 Acc: 0.8641

Epoch 19/60
----------
train Loss: 0.0023 Acc: 0.9563
val Loss: 0.0084 Acc: 0.8641

Epoch 20/60
----------
train Loss: 0.0025 Acc: 0.9522
val Loss: 0.0051 Acc: 0.8689

Epoch 21/60
----------
train Loss: 0.0022 Acc: 0.9584
val Loss: 0.0107 Acc: 0.8689

Epoch 22/60
----------
train Loss: 0.0024 Acc: 0.9397
val Loss: 0.0064 Acc: 0.8689

Epoch 23/60
----------
train Loss: 0.0023 Acc: 0.9522
val Loss: 0.0074 Acc: 0.8641

Epoch 24/60
----------
train Loss: 0.0023 Acc: 0.9584
val Loss: 0.0049 Acc: 0.8689

Epoch 25/60
----------
train Loss: 0.0025 Acc: 0.9501
val Loss: 0.0060 Acc: 0.8689

Epoch 26/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9543
val Loss: 0.0066 Acc: 0.8689

Epoch 27/60
----------
train Loss: 0.0024 Acc: 0.9563
val Loss: 0.0080 Acc: 0.8641

Epoch 28/60
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0094 Acc: 0.8689

Epoch 29/60
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0049 Acc: 0.8689

Epoch 30/60
----------
train Loss: 0.0025 Acc: 0.9605
val Loss: 0.0056 Acc: 0.8689

Epoch 31/60
----------
train Loss: 0.0022 Acc: 0.9563
val Loss: 0.0084 Acc: 0.8738

Epoch 32/60
----------
train Loss: 0.0022 Acc: 0.9563
val Loss: 0.0055 Acc: 0.8738

Epoch 33/60
----------
train Loss: 0.0024 Acc: 0.9522
val Loss: 0.0048 Acc: 0.8738

Epoch 34/60
----------
train Loss: 0.0026 Acc: 0.9356
val Loss: 0.0050 Acc: 0.8689

Epoch 35/60
----------
train Loss: 0.0024 Acc: 0.9626
val Loss: 0.0059 Acc: 0.8689

Epoch 36/60
----------
train Loss: 0.0024 Acc: 0.9418
val Loss: 0.0077 Acc: 0.8738

Epoch 37/60
----------
train Loss: 0.0023 Acc: 0.9709
val Loss: 0.0054 Acc: 0.8738

Epoch 38/60
----------
train Loss: 0.0024 Acc: 0.9543
val Loss: 0.0118 Acc: 0.8689

Epoch 39/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9439
val Loss: 0.0055 Acc: 0.8689

Epoch 40/60
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0064 Acc: 0.8689

Epoch 41/60
----------
train Loss: 0.0025 Acc: 0.9439
val Loss: 0.0081 Acc: 0.8689

Epoch 42/60
----------
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0053 Acc: 0.8689

Epoch 43/60
----------
train Loss: 0.0023 Acc: 0.9605
val Loss: 0.0063 Acc: 0.8689

Epoch 44/60
----------
train Loss: 0.0023 Acc: 0.9584
val Loss: 0.0051 Acc: 0.8689

Epoch 45/60
----------
train Loss: 0.0024 Acc: 0.9418
val Loss: 0.0093 Acc: 0.8689

Epoch 46/60
----------
train Loss: 0.0023 Acc: 0.9605
val Loss: 0.0061 Acc: 0.8641

Epoch 47/60
----------
train Loss: 0.0022 Acc: 0.9563
val Loss: 0.0050 Acc: 0.8641

Epoch 48/60
----------
train Loss: 0.0022 Acc: 0.9480
val Loss: 0.0064 Acc: 0.8689

Epoch 49/60
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0094 Acc: 0.8641

Epoch 50/60
----------
train Loss: 0.0025 Acc: 0.9459
val Loss: 0.0073 Acc: 0.8689

Epoch 51/60
----------
train Loss: 0.0023 Acc: 0.9459
val Loss: 0.0073 Acc: 0.8738

Epoch 52/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0024 Acc: 0.9439
val Loss: 0.0075 Acc: 0.8738

Epoch 53/60
----------
train Loss: 0.0024 Acc: 0.9397
val Loss: 0.0052 Acc: 0.8738

Epoch 54/60
----------
train Loss: 0.0023 Acc: 0.9439
val Loss: 0.0048 Acc: 0.8738

Epoch 55/60
----------
train Loss: 0.0023 Acc: 0.9501
val Loss: 0.0052 Acc: 0.8689

Epoch 56/60
----------
train Loss: 0.0024 Acc: 0.9543
val Loss: 0.0048 Acc: 0.8738

Epoch 57/60
----------
train Loss: 0.0024 Acc: 0.9522
val Loss: 0.0062 Acc: 0.8738

Epoch 58/60
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0097 Acc: 0.8738

Epoch 59/60
----------
train Loss: 0.0024 Acc: 0.9543
val Loss: 0.0050 Acc: 0.8738

Epoch 60/60
----------
train Loss: 0.0024 Acc: 0.9501
val Loss: 0.0050 Acc: 0.8738

Training complete in 2m 56s
Best val Acc: 0.873786

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0034 Acc: 0.9127
val Loss: 0.0071 Acc: 0.8010

Epoch 1/60
----------
train Loss: 0.0023 Acc: 0.9543
val Loss: 0.0048 Acc: 0.8592

Epoch 2/60
----------
train Loss: 0.0010 Acc: 0.9938
val Loss: 0.0080 Acc: 0.8932

Epoch 3/60
----------
train Loss: 0.0007 Acc: 0.9917
val Loss: 0.0063 Acc: 0.8835

Epoch 4/60
----------
train Loss: 0.0004 Acc: 0.9958
val Loss: 0.0048 Acc: 0.8932

Epoch 5/60
----------
train Loss: 0.0003 Acc: 0.9958
val Loss: 0.0098 Acc: 0.8883

Epoch 6/60
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0039 Acc: 0.8835

Epoch 7/60
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0042 Acc: 0.8786

Epoch 8/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8883

Epoch 9/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0073 Acc: 0.8738

Epoch 10/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0088 Acc: 0.8835

Epoch 11/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8738

Epoch 12/60
----------
train Loss: 0.0001 Acc: 0.9979
val Loss: 0.0039 Acc: 0.8835

Epoch 13/60
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 14/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8786

Epoch 15/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 16/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8786

Epoch 17/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 18/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8786

Epoch 19/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8786

Epoch 20/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8786

Epoch 21/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8786

Epoch 22/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8786

Epoch 23/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8786

Epoch 24/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0047 Acc: 0.8786

Epoch 25/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0077 Acc: 0.8786

Epoch 26/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 27/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0078 Acc: 0.8786

Epoch 28/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8786

Epoch 29/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 30/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0045 Acc: 0.8786

Epoch 31/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8786

Epoch 32/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0129 Acc: 0.8786

Epoch 33/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0066 Acc: 0.8786

Epoch 34/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8786

Epoch 35/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8786

Epoch 36/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8786

Epoch 37/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8786

Epoch 38/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 39/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8786

Epoch 40/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0076 Acc: 0.8786

Epoch 41/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.8786

Epoch 42/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0085 Acc: 0.8786

Epoch 43/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8786

Epoch 44/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8786

Epoch 45/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8786

Epoch 46/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 47/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 48/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8786

Epoch 49/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0044 Acc: 0.8786

Epoch 50/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0117 Acc: 0.8835

Epoch 51/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0090 Acc: 0.8786

Epoch 52/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8786

Epoch 53/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 54/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0068 Acc: 0.8786

Epoch 55/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8786

Epoch 56/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8835

Epoch 57/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0040 Acc: 0.8786

Epoch 58/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8835

Epoch 59/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8786

Epoch 60/60
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8786

Training complete in 3m 5s
Best val Acc: 0.893204

---Testing---
Test accuracy: 0.954876
--------------------
Accuracy of Carcharhiniformes : 96 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Lamniformes : 87 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 95 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 96 %
mean: 0.9460387259331978, std: 0.03916162632123012

Model saved in "./weights/shark_[0.98]_mean[0.97]_std[0.03].save".
--------------------

run info[val: 0.1, epoch: 85, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0226 Acc: 0.2391
val Loss: 0.0219 Acc: 0.4853

Epoch 1/84
----------
train Loss: 0.0133 Acc: 0.6397
val Loss: 0.0147 Acc: 0.6471

Epoch 2/84
----------
train Loss: 0.0080 Acc: 0.7787
val Loss: 0.0126 Acc: 0.7353

Epoch 3/84
----------
train Loss: 0.0057 Acc: 0.8417
val Loss: 0.0110 Acc: 0.7647

Epoch 4/84
----------
train Loss: 0.0046 Acc: 0.8772
val Loss: 0.0106 Acc: 0.7647

Epoch 5/84
----------
train Loss: 0.0039 Acc: 0.9241
val Loss: 0.0105 Acc: 0.7647

Epoch 6/84
----------
train Loss: 0.0035 Acc: 0.9160
val Loss: 0.0106 Acc: 0.7794

Epoch 7/84
----------
train Loss: 0.0036 Acc: 0.9160
val Loss: 0.0108 Acc: 0.7353

Epoch 8/84
----------
train Loss: 0.0027 Acc: 0.9483
val Loss: 0.0109 Acc: 0.7941

Epoch 9/84
----------
LR is set to 0.001
train Loss: 0.0026 Acc: 0.9467
val Loss: 0.0108 Acc: 0.7647

Epoch 10/84
----------
train Loss: 0.0028 Acc: 0.9548
val Loss: 0.0104 Acc: 0.7353

Epoch 11/84
----------
train Loss: 0.0027 Acc: 0.9532
val Loss: 0.0102 Acc: 0.7500

Epoch 12/84
----------
train Loss: 0.0023 Acc: 0.9499
val Loss: 0.0101 Acc: 0.7647

Epoch 13/84
----------
train Loss: 0.0023 Acc: 0.9532
val Loss: 0.0100 Acc: 0.7647

Epoch 14/84
----------
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0102 Acc: 0.7794

Epoch 15/84
----------
train Loss: 0.0025 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7647

Epoch 16/84
----------
train Loss: 0.0024 Acc: 0.9612
val Loss: 0.0101 Acc: 0.7647

Epoch 17/84
----------
train Loss: 0.0022 Acc: 0.9612
val Loss: 0.0102 Acc: 0.7794

Epoch 18/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0028 Acc: 0.9596
val Loss: 0.0100 Acc: 0.7794

Epoch 19/84
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0101 Acc: 0.7794

Epoch 20/84
----------
train Loss: 0.0023 Acc: 0.9580
val Loss: 0.0102 Acc: 0.7794

Epoch 21/84
----------
train Loss: 0.0026 Acc: 0.9548
val Loss: 0.0103 Acc: 0.7647

Epoch 22/84
----------
train Loss: 0.0024 Acc: 0.9564
val Loss: 0.0102 Acc: 0.7794

Epoch 23/84
----------
train Loss: 0.0026 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7794

Epoch 24/84
----------
train Loss: 0.0024 Acc: 0.9532
val Loss: 0.0103 Acc: 0.7647

Epoch 25/84
----------
train Loss: 0.0024 Acc: 0.9532
val Loss: 0.0103 Acc: 0.7794

Epoch 26/84
----------
train Loss: 0.0023 Acc: 0.9548
val Loss: 0.0103 Acc: 0.7794

Epoch 27/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0022 Acc: 0.9564
val Loss: 0.0103 Acc: 0.7794

Epoch 28/84
----------
train Loss: 0.0022 Acc: 0.9645
val Loss: 0.0104 Acc: 0.7794

Epoch 29/84
----------
train Loss: 0.0022 Acc: 0.9548
val Loss: 0.0103 Acc: 0.7647

Epoch 30/84
----------
train Loss: 0.0026 Acc: 0.9515
val Loss: 0.0102 Acc: 0.7794

Epoch 31/84
----------
train Loss: 0.0023 Acc: 0.9628
val Loss: 0.0100 Acc: 0.7794

Epoch 32/84
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0100 Acc: 0.7794

Epoch 33/84
----------
train Loss: 0.0023 Acc: 0.9628
val Loss: 0.0101 Acc: 0.7794

Epoch 34/84
----------
train Loss: 0.0022 Acc: 0.9677
val Loss: 0.0101 Acc: 0.7794

Epoch 35/84
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0102 Acc: 0.7794

Epoch 36/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9580
val Loss: 0.0102 Acc: 0.7647

Epoch 37/84
----------
train Loss: 0.0025 Acc: 0.9725
val Loss: 0.0100 Acc: 0.7794

Epoch 38/84
----------
train Loss: 0.0021 Acc: 0.9564
val Loss: 0.0099 Acc: 0.7794

Epoch 39/84
----------
train Loss: 0.0025 Acc: 0.9596
val Loss: 0.0101 Acc: 0.7794

Epoch 40/84
----------
train Loss: 0.0023 Acc: 0.9532
val Loss: 0.0102 Acc: 0.7794

Epoch 41/84
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0103 Acc: 0.7794

Epoch 42/84
----------
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0102 Acc: 0.7794

Epoch 43/84
----------
train Loss: 0.0025 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7794

Epoch 44/84
----------
train Loss: 0.0026 Acc: 0.9596
val Loss: 0.0101 Acc: 0.7794

Epoch 45/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0024 Acc: 0.9515
val Loss: 0.0103 Acc: 0.7794

Epoch 46/84
----------
train Loss: 0.0026 Acc: 0.9612
val Loss: 0.0104 Acc: 0.7647

Epoch 47/84
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0103 Acc: 0.7647

Epoch 48/84
----------
train Loss: 0.0025 Acc: 0.9612
val Loss: 0.0103 Acc: 0.7647

Epoch 49/84
----------
train Loss: 0.0025 Acc: 0.9564
val Loss: 0.0101 Acc: 0.7794

Epoch 50/84
----------
train Loss: 0.0021 Acc: 0.9693
val Loss: 0.0102 Acc: 0.7794

Epoch 51/84
----------
train Loss: 0.0024 Acc: 0.9548
val Loss: 0.0102 Acc: 0.7794

Epoch 52/84
----------
train Loss: 0.0027 Acc: 0.9596
val Loss: 0.0103 Acc: 0.7647

Epoch 53/84
----------
train Loss: 0.0023 Acc: 0.9677
val Loss: 0.0102 Acc: 0.7647

Epoch 54/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0023 Acc: 0.9548
val Loss: 0.0101 Acc: 0.7647

Epoch 55/84
----------
train Loss: 0.0021 Acc: 0.9645
val Loss: 0.0102 Acc: 0.7794

Epoch 56/84
----------
train Loss: 0.0023 Acc: 0.9645
val Loss: 0.0102 Acc: 0.7647

Epoch 57/84
----------
train Loss: 0.0026 Acc: 0.9580
val Loss: 0.0101 Acc: 0.7794

Epoch 58/84
----------
train Loss: 0.0027 Acc: 0.9548
val Loss: 0.0103 Acc: 0.7794

Epoch 59/84
----------
train Loss: 0.0022 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7794

Epoch 60/84
----------
train Loss: 0.0025 Acc: 0.9645
val Loss: 0.0102 Acc: 0.7647

Epoch 61/84
----------
train Loss: 0.0023 Acc: 0.9645
val Loss: 0.0100 Acc: 0.7794

Epoch 62/84
----------
train Loss: 0.0023 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7794

Epoch 63/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0024 Acc: 0.9580
val Loss: 0.0103 Acc: 0.7647

Epoch 64/84
----------
train Loss: 0.0027 Acc: 0.9564
val Loss: 0.0100 Acc: 0.7794

Epoch 65/84
----------
train Loss: 0.0022 Acc: 0.9709
val Loss: 0.0100 Acc: 0.7794

Epoch 66/84
----------
train Loss: 0.0021 Acc: 0.9661
val Loss: 0.0099 Acc: 0.7794

Epoch 67/84
----------
train Loss: 0.0022 Acc: 0.9661
val Loss: 0.0100 Acc: 0.7794

Epoch 68/84
----------
train Loss: 0.0022 Acc: 0.9580
val Loss: 0.0101 Acc: 0.7794

Epoch 69/84
----------
train Loss: 0.0024 Acc: 0.9548
val Loss: 0.0100 Acc: 0.7794

Epoch 70/84
----------
train Loss: 0.0024 Acc: 0.9645
val Loss: 0.0102 Acc: 0.7794

Epoch 71/84
----------
train Loss: 0.0022 Acc: 0.9612
val Loss: 0.0101 Acc: 0.7794

Epoch 72/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0026 Acc: 0.9483
val Loss: 0.0098 Acc: 0.7794

Epoch 73/84
----------
train Loss: 0.0025 Acc: 0.9645
val Loss: 0.0099 Acc: 0.7794

Epoch 74/84
----------
train Loss: 0.0022 Acc: 0.9564
val Loss: 0.0100 Acc: 0.7794

Epoch 75/84
----------
train Loss: 0.0025 Acc: 0.9645
val Loss: 0.0101 Acc: 0.7794

Epoch 76/84
----------
train Loss: 0.0026 Acc: 0.9564
val Loss: 0.0102 Acc: 0.7794

Epoch 77/84
----------
train Loss: 0.0021 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7794

Epoch 78/84
----------
train Loss: 0.0022 Acc: 0.9612
val Loss: 0.0103 Acc: 0.7647

Epoch 79/84
----------
train Loss: 0.0024 Acc: 0.9628
val Loss: 0.0102 Acc: 0.7647

Epoch 80/84
----------
train Loss: 0.0024 Acc: 0.9628
val Loss: 0.0102 Acc: 0.7794

Epoch 81/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0022 Acc: 0.9596
val Loss: 0.0102 Acc: 0.7647

Epoch 82/84
----------
train Loss: 0.0023 Acc: 0.9612
val Loss: 0.0102 Acc: 0.7794

Epoch 83/84
----------
train Loss: 0.0021 Acc: 0.9709
val Loss: 0.0102 Acc: 0.7794

Epoch 84/84
----------
train Loss: 0.0022 Acc: 0.9645
val Loss: 0.0102 Acc: 0.7647

Training complete in 4m 7s
Best val Acc: 0.794118

---Fine tuning.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0024 Acc: 0.9418
val Loss: 0.0092 Acc: 0.8088

Epoch 1/84
----------
train Loss: 0.0015 Acc: 0.9758
val Loss: 0.0076 Acc: 0.8529

Epoch 2/84
----------
train Loss: 0.0008 Acc: 0.9855
val Loss: 0.0074 Acc: 0.8529

Epoch 3/84
----------
train Loss: 0.0010 Acc: 0.9822
val Loss: 0.0111 Acc: 0.8235

Epoch 4/84
----------
train Loss: 0.0007 Acc: 0.9903
val Loss: 0.0111 Acc: 0.8088

Epoch 5/84
----------
train Loss: 0.0003 Acc: 0.9935
val Loss: 0.0168 Acc: 0.7353

Epoch 6/84
----------
train Loss: 0.0004 Acc: 0.9919
val Loss: 0.0137 Acc: 0.7794

Epoch 7/84
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0103 Acc: 0.8088

Epoch 8/84
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0095 Acc: 0.8382

Epoch 9/84
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8088

Epoch 10/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0092 Acc: 0.8235

Epoch 11/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 12/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 13/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 14/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 15/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 16/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 17/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 18/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8235

Epoch 19/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 20/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 21/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 22/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 23/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 24/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 25/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 26/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 27/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 28/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 29/84
----------
train Loss: 0.0001 Acc: 0.9984
val Loss: 0.0095 Acc: 0.8088

Epoch 30/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 31/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8235

Epoch 32/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8088

Epoch 33/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8088

Epoch 34/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 35/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 36/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8235

Epoch 37/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 38/84
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8235

Epoch 39/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8235

Epoch 40/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8235

Epoch 41/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8235

Epoch 42/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 43/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 44/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 45/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 46/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 47/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 48/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8235

Epoch 49/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8235

Epoch 50/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 51/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8088

Epoch 52/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 53/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 54/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8235

Epoch 55/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8235

Epoch 56/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8235

Epoch 57/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0093 Acc: 0.8235

Epoch 58/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 59/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 60/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 61/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 62/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 63/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 64/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 65/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 66/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0099 Acc: 0.8088

Epoch 67/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8088

Epoch 68/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 69/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 70/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8235

Epoch 71/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 72/84
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 73/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8088

Epoch 74/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 75/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0094 Acc: 0.8235

Epoch 76/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8088

Epoch 77/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 78/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0100 Acc: 0.8088

Epoch 79/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8088

Epoch 80/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 81/84
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Epoch 82/84
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0095 Acc: 0.8235

Epoch 83/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0097 Acc: 0.8235

Epoch 84/84
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0096 Acc: 0.8235

Training complete in 4m 38s
Best val Acc: 0.852941

---Testing---
Test accuracy: 0.973799
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Lamniformes : 89 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 92 %
Accuracy of Squatiniformes : 94 %
mean: 0.9666428464388549, std: 0.036697786249861405
--------------------

run info[val: 0.15, epoch: 90, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0205 Acc: 0.2740
val Loss: 0.0353 Acc: 0.2621

Epoch 1/89
----------
train Loss: 0.0135 Acc: 0.6079
val Loss: 0.0240 Acc: 0.5631

Epoch 2/89
----------
train Loss: 0.0087 Acc: 0.7277
val Loss: 0.0212 Acc: 0.7184

Epoch 3/89
----------
train Loss: 0.0060 Acc: 0.8390
val Loss: 0.0201 Acc: 0.7184

Epoch 4/89
----------
train Loss: 0.0051 Acc: 0.8510
val Loss: 0.0115 Acc: 0.7670

Epoch 5/89
----------
train Loss: 0.0047 Acc: 0.8545
val Loss: 0.0148 Acc: 0.7670

Epoch 6/89
----------
train Loss: 0.0037 Acc: 0.9007
val Loss: 0.0115 Acc: 0.7767

Epoch 7/89
----------
train Loss: 0.0036 Acc: 0.8990
val Loss: 0.0261 Acc: 0.7670

Epoch 8/89
----------
train Loss: 0.0032 Acc: 0.9092
val Loss: 0.0072 Acc: 0.7864

Epoch 9/89
----------
LR is set to 0.001
train Loss: 0.0032 Acc: 0.9024
val Loss: 0.0070 Acc: 0.7864

Epoch 10/89
----------
train Loss: 0.0029 Acc: 0.9298
val Loss: 0.0129 Acc: 0.7767

Epoch 11/89
----------
train Loss: 0.0030 Acc: 0.9281
val Loss: 0.0102 Acc: 0.7767

Epoch 12/89
----------
train Loss: 0.0029 Acc: 0.9178
val Loss: 0.0071 Acc: 0.7767

Epoch 13/89
----------
train Loss: 0.0029 Acc: 0.9281
val Loss: 0.0066 Acc: 0.7767

Epoch 14/89
----------
train Loss: 0.0028 Acc: 0.9349
val Loss: 0.0081 Acc: 0.7767

Epoch 15/89
----------
train Loss: 0.0029 Acc: 0.9264
val Loss: 0.0082 Acc: 0.7767

Epoch 16/89
----------
train Loss: 0.0029 Acc: 0.9281
val Loss: 0.0074 Acc: 0.7767

Epoch 17/89
----------
train Loss: 0.0028 Acc: 0.9366
val Loss: 0.0085 Acc: 0.7767

Epoch 18/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9315
val Loss: 0.0098 Acc: 0.7767

Epoch 19/89
----------
train Loss: 0.0028 Acc: 0.9281
val Loss: 0.0075 Acc: 0.7767

Epoch 20/89
----------
train Loss: 0.0028 Acc: 0.9298
val Loss: 0.0191 Acc: 0.7767

Epoch 21/89
----------
train Loss: 0.0027 Acc: 0.9384
val Loss: 0.0072 Acc: 0.7767

Epoch 22/89
----------
train Loss: 0.0028 Acc: 0.9401
val Loss: 0.0220 Acc: 0.7767

Epoch 23/89
----------
train Loss: 0.0026 Acc: 0.9366
val Loss: 0.0116 Acc: 0.7767

Epoch 24/89
----------
train Loss: 0.0028 Acc: 0.9229
val Loss: 0.0065 Acc: 0.7767

Epoch 25/89
----------
train Loss: 0.0027 Acc: 0.9281
val Loss: 0.0139 Acc: 0.7767

Epoch 26/89
----------
train Loss: 0.0028 Acc: 0.9195
val Loss: 0.0067 Acc: 0.7767

Epoch 27/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0026 Acc: 0.9418
val Loss: 0.0078 Acc: 0.7767

Epoch 28/89
----------
train Loss: 0.0027 Acc: 0.9401
val Loss: 0.0066 Acc: 0.7767

Epoch 29/89
----------
train Loss: 0.0028 Acc: 0.9384
val Loss: 0.0073 Acc: 0.7767

Epoch 30/89
----------
train Loss: 0.0028 Acc: 0.9298
val Loss: 0.0066 Acc: 0.7767

Epoch 31/89
----------
train Loss: 0.0027 Acc: 0.9401
val Loss: 0.0080 Acc: 0.7767

Epoch 32/89
----------
train Loss: 0.0029 Acc: 0.9281
val Loss: 0.0101 Acc: 0.7767

Epoch 33/89
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0191 Acc: 0.7767

Epoch 34/89
----------
train Loss: 0.0027 Acc: 0.9469
val Loss: 0.0069 Acc: 0.7767

Epoch 35/89
----------
train Loss: 0.0026 Acc: 0.9384
val Loss: 0.0146 Acc: 0.7767

Epoch 36/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0078 Acc: 0.7767

Epoch 37/89
----------
train Loss: 0.0028 Acc: 0.9127
val Loss: 0.0121 Acc: 0.7767

Epoch 38/89
----------
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0270 Acc: 0.7767

Epoch 39/89
----------
train Loss: 0.0028 Acc: 0.9384
val Loss: 0.0087 Acc: 0.7767

Epoch 40/89
----------
train Loss: 0.0030 Acc: 0.9264
val Loss: 0.0067 Acc: 0.7767

Epoch 41/89
----------
train Loss: 0.0029 Acc: 0.9349
val Loss: 0.0071 Acc: 0.7767

Epoch 42/89
----------
train Loss: 0.0028 Acc: 0.9127
val Loss: 0.0081 Acc: 0.7767

Epoch 43/89
----------
train Loss: 0.0026 Acc: 0.9452
val Loss: 0.0070 Acc: 0.7767

Epoch 44/89
----------
train Loss: 0.0029 Acc: 0.9298
val Loss: 0.0078 Acc: 0.7767

Epoch 45/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9315
val Loss: 0.0100 Acc: 0.7767

Epoch 46/89
----------
train Loss: 0.0029 Acc: 0.9332
val Loss: 0.0080 Acc: 0.7767

Epoch 47/89
----------
train Loss: 0.0027 Acc: 0.9469
val Loss: 0.0193 Acc: 0.7767

Epoch 48/89
----------
train Loss: 0.0028 Acc: 0.9384
val Loss: 0.0089 Acc: 0.7767

Epoch 49/89
----------
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0201 Acc: 0.7767

Epoch 50/89
----------
train Loss: 0.0028 Acc: 0.9401
val Loss: 0.0072 Acc: 0.7767

Epoch 51/89
----------
train Loss: 0.0027 Acc: 0.9332
val Loss: 0.0073 Acc: 0.7767

Epoch 52/89
----------
train Loss: 0.0027 Acc: 0.9366
val Loss: 0.0072 Acc: 0.7767

Epoch 53/89
----------
train Loss: 0.0028 Acc: 0.9469
val Loss: 0.0064 Acc: 0.7767

Epoch 54/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0028 Acc: 0.9401
val Loss: 0.0066 Acc: 0.7767

Epoch 55/89
----------
train Loss: 0.0027 Acc: 0.9452
val Loss: 0.0266 Acc: 0.7767

Epoch 56/89
----------
train Loss: 0.0027 Acc: 0.9366
val Loss: 0.0126 Acc: 0.7767

Epoch 57/89
----------
train Loss: 0.0027 Acc: 0.9418
val Loss: 0.0214 Acc: 0.7767

Epoch 58/89
----------
train Loss: 0.0028 Acc: 0.9332
val Loss: 0.0112 Acc: 0.7767

Epoch 59/89
----------
train Loss: 0.0027 Acc: 0.9384
val Loss: 0.0100 Acc: 0.7767

Epoch 60/89
----------
train Loss: 0.0028 Acc: 0.9264
val Loss: 0.0092 Acc: 0.7767

Epoch 61/89
----------
train Loss: 0.0028 Acc: 0.9298
val Loss: 0.0186 Acc: 0.7767

Epoch 62/89
----------
train Loss: 0.0028 Acc: 0.9366
val Loss: 0.0066 Acc: 0.7767

Epoch 63/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0028 Acc: 0.9264
val Loss: 0.0111 Acc: 0.7767

Epoch 64/89
----------
train Loss: 0.0028 Acc: 0.9315
val Loss: 0.0161 Acc: 0.7767

Epoch 65/89
----------
train Loss: 0.0027 Acc: 0.9401
val Loss: 0.0159 Acc: 0.7767

Epoch 66/89
----------
train Loss: 0.0026 Acc: 0.9366
val Loss: 0.0072 Acc: 0.7767

Epoch 67/89
----------
train Loss: 0.0025 Acc: 0.9503
val Loss: 0.0078 Acc: 0.7767

Epoch 68/89
----------
train Loss: 0.0028 Acc: 0.9298
val Loss: 0.0113 Acc: 0.7767

Epoch 69/89
----------
train Loss: 0.0028 Acc: 0.9332
val Loss: 0.0107 Acc: 0.7767

Epoch 70/89
----------
train Loss: 0.0028 Acc: 0.9366
val Loss: 0.0067 Acc: 0.7767

Epoch 71/89
----------
train Loss: 0.0026 Acc: 0.9366
val Loss: 0.0096 Acc: 0.7767

Epoch 72/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0186 Acc: 0.7767

Epoch 73/89
----------
train Loss: 0.0028 Acc: 0.9349
val Loss: 0.0147 Acc: 0.7767

Epoch 74/89
----------
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0193 Acc: 0.7767

Epoch 75/89
----------
train Loss: 0.0029 Acc: 0.9366
val Loss: 0.0120 Acc: 0.7767

Epoch 76/89
----------
train Loss: 0.0028 Acc: 0.9332
val Loss: 0.0067 Acc: 0.7767

Epoch 77/89
----------
train Loss: 0.0028 Acc: 0.9315
val Loss: 0.0184 Acc: 0.7767

Epoch 78/89
----------
train Loss: 0.0026 Acc: 0.9469
val Loss: 0.0186 Acc: 0.7767

Epoch 79/89
----------
train Loss: 0.0026 Acc: 0.9349
val Loss: 0.0079 Acc: 0.7767

Epoch 80/89
----------
train Loss: 0.0026 Acc: 0.9486
val Loss: 0.0121 Acc: 0.7767

Epoch 81/89
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0026 Acc: 0.9315
val Loss: 0.0113 Acc: 0.7767

Epoch 82/89
----------
train Loss: 0.0028 Acc: 0.9384
val Loss: 0.0216 Acc: 0.7767

Epoch 83/89
----------
train Loss: 0.0026 Acc: 0.9401
val Loss: 0.0081 Acc: 0.7767

Epoch 84/89
----------
train Loss: 0.0029 Acc: 0.9315
val Loss: 0.0174 Acc: 0.7767

Epoch 85/89
----------
train Loss: 0.0027 Acc: 0.9435
val Loss: 0.0066 Acc: 0.7767

Epoch 86/89
----------
train Loss: 0.0027 Acc: 0.9349
val Loss: 0.0176 Acc: 0.7767

Epoch 87/89
----------
train Loss: 0.0027 Acc: 0.9281
val Loss: 0.0170 Acc: 0.7767

Epoch 88/89
----------
train Loss: 0.0025 Acc: 0.9469
val Loss: 0.0264 Acc: 0.7767

Epoch 89/89
----------
train Loss: 0.0028 Acc: 0.9332
val Loss: 0.0065 Acc: 0.7767

Training complete in 4m 44s
Best val Acc: 0.786408

---Fine tuning.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0030 Acc: 0.9264
val Loss: 0.0082 Acc: 0.7864

Epoch 1/89
----------
train Loss: 0.0019 Acc: 0.9538
val Loss: 0.0060 Acc: 0.8252

Epoch 2/89
----------
train Loss: 0.0010 Acc: 0.9795
val Loss: 0.0062 Acc: 0.8447

Epoch 3/89
----------
train Loss: 0.0006 Acc: 0.9914
val Loss: 0.0194 Acc: 0.8641

Epoch 4/89
----------
train Loss: 0.0003 Acc: 0.9966
val Loss: 0.0047 Acc: 0.8544

Epoch 5/89
----------
train Loss: 0.0003 Acc: 0.9949
val Loss: 0.0273 Acc: 0.8544

Epoch 6/89
----------
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0115 Acc: 0.8544

Epoch 7/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0122 Acc: 0.8350

Epoch 8/89
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0059 Acc: 0.8544

Epoch 9/89
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8544

Epoch 10/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0300 Acc: 0.8544

Epoch 11/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0123 Acc: 0.8544

Epoch 12/89
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0053 Acc: 0.8447

Epoch 13/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0107 Acc: 0.8447

Epoch 14/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 15/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0185 Acc: 0.8447

Epoch 16/89
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0182 Acc: 0.8447

Epoch 17/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8447

Epoch 18/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0263 Acc: 0.8447

Epoch 19/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0192 Acc: 0.8447

Epoch 20/89
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0214 Acc: 0.8447

Epoch 21/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 22/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8447

Epoch 23/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 24/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0120 Acc: 0.8447

Epoch 25/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0203 Acc: 0.8447

Epoch 26/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0081 Acc: 0.8447

Epoch 27/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8447

Epoch 28/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0236 Acc: 0.8447

Epoch 29/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 30/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0082 Acc: 0.8447

Epoch 31/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 32/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 33/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0183 Acc: 0.8447

Epoch 34/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8447

Epoch 35/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 36/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0191 Acc: 0.8447

Epoch 37/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8447

Epoch 38/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0206 Acc: 0.8447

Epoch 39/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 40/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 41/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 42/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 43/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0211 Acc: 0.8447

Epoch 44/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 45/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8447

Epoch 46/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 47/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 48/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 49/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 50/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 51/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0230 Acc: 0.8447

Epoch 52/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 53/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8447

Epoch 54/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0102 Acc: 0.8447

Epoch 55/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0179 Acc: 0.8447

Epoch 56/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0205 Acc: 0.8447

Epoch 57/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8447

Epoch 58/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0151 Acc: 0.8447

Epoch 59/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0114 Acc: 0.8447

Epoch 60/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8447

Epoch 61/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 62/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 63/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 64/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0221 Acc: 0.8447

Epoch 65/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0245 Acc: 0.8447

Epoch 66/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 67/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0071 Acc: 0.8447

Epoch 68/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0080 Acc: 0.8447

Epoch 69/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0214 Acc: 0.8447

Epoch 70/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0191 Acc: 0.8447

Epoch 71/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 72/89
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0060 Acc: 0.8447

Epoch 73/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0098 Acc: 0.8447

Epoch 74/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0104 Acc: 0.8447

Epoch 75/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 76/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 77/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 78/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8447

Epoch 79/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0179 Acc: 0.8447

Epoch 80/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 81/89
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0058 Acc: 0.8447

Epoch 82/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0089 Acc: 0.8447

Epoch 83/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Epoch 84/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0186 Acc: 0.8447

Epoch 85/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0070 Acc: 0.8447

Epoch 86/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0113 Acc: 0.8447

Epoch 87/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0243 Acc: 0.8447

Epoch 88/89
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0149 Acc: 0.8447

Epoch 89/89
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8447

Training complete in 4m 53s
Best val Acc: 0.864078

---Testing---
Test accuracy: 0.976710
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 91 %
Accuracy of Orectolobiformes : 99 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.9686025753064785, std: 0.03428013020520591
--------------------

run info[val: 0.2, epoch: 61, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0221 Acc: 0.2327
val Loss: 0.0282 Acc: 0.3504

Epoch 1/60
----------
train Loss: 0.0158 Acc: 0.4909
val Loss: 0.0166 Acc: 0.6642

Epoch 2/60
----------
train Loss: 0.0093 Acc: 0.7164
val Loss: 0.0143 Acc: 0.6058

Epoch 3/60
----------
LR is set to 0.001
train Loss: 0.0077 Acc: 0.7473
val Loss: 0.0124 Acc: 0.7007

Epoch 4/60
----------
train Loss: 0.0064 Acc: 0.8327
val Loss: 0.0103 Acc: 0.7737

Epoch 5/60
----------
train Loss: 0.0060 Acc: 0.8582
val Loss: 0.0116 Acc: 0.7883

Epoch 6/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0061 Acc: 0.8509
val Loss: 0.0108 Acc: 0.7883

Epoch 7/60
----------
train Loss: 0.0060 Acc: 0.8527
val Loss: 0.0110 Acc: 0.7883

Epoch 8/60
----------
train Loss: 0.0061 Acc: 0.8491
val Loss: 0.0108 Acc: 0.7737

Epoch 9/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0059 Acc: 0.8473
val Loss: 0.0113 Acc: 0.7737

Epoch 10/60
----------
train Loss: 0.0060 Acc: 0.8491
val Loss: 0.0098 Acc: 0.7737

Epoch 11/60
----------
train Loss: 0.0060 Acc: 0.8436
val Loss: 0.0115 Acc: 0.7737

Epoch 12/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0056 Acc: 0.8709
val Loss: 0.0114 Acc: 0.7737

Epoch 13/60
----------
train Loss: 0.0058 Acc: 0.8582
val Loss: 0.0110 Acc: 0.7737

Epoch 14/60
----------
train Loss: 0.0058 Acc: 0.8527
val Loss: 0.0102 Acc: 0.7810

Epoch 15/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0058 Acc: 0.8655
val Loss: 0.0104 Acc: 0.7810

Epoch 16/60
----------
train Loss: 0.0059 Acc: 0.8618
val Loss: 0.0105 Acc: 0.7737

Epoch 17/60
----------
train Loss: 0.0058 Acc: 0.8618
val Loss: 0.0109 Acc: 0.7737

Epoch 18/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0057 Acc: 0.8582
val Loss: 0.0113 Acc: 0.7737

Epoch 19/60
----------
train Loss: 0.0060 Acc: 0.8509
val Loss: 0.0101 Acc: 0.7737

Epoch 20/60
----------
train Loss: 0.0057 Acc: 0.8691
val Loss: 0.0100 Acc: 0.7737

Epoch 21/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0058 Acc: 0.8691
val Loss: 0.0118 Acc: 0.7737

Epoch 22/60
----------
train Loss: 0.0057 Acc: 0.8527
val Loss: 0.0107 Acc: 0.7737

Epoch 23/60
----------
train Loss: 0.0058 Acc: 0.8509
val Loss: 0.0103 Acc: 0.7737

Epoch 24/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0057 Acc: 0.8636
val Loss: 0.0107 Acc: 0.7737

Epoch 25/60
----------
train Loss: 0.0058 Acc: 0.8691
val Loss: 0.0098 Acc: 0.7737

Epoch 26/60
----------
train Loss: 0.0059 Acc: 0.8527
val Loss: 0.0115 Acc: 0.7737

Epoch 27/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0060 Acc: 0.8618
val Loss: 0.0110 Acc: 0.7737

Epoch 28/60
----------
train Loss: 0.0057 Acc: 0.8636
val Loss: 0.0104 Acc: 0.7737

Epoch 29/60
----------
train Loss: 0.0059 Acc: 0.8782
val Loss: 0.0112 Acc: 0.7737

Epoch 30/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0060 Acc: 0.8491
val Loss: 0.0100 Acc: 0.7810

Epoch 31/60
----------
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0107 Acc: 0.7737

Epoch 32/60
----------
train Loss: 0.0058 Acc: 0.8582
val Loss: 0.0105 Acc: 0.7737

Epoch 33/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0058 Acc: 0.8564
val Loss: 0.0110 Acc: 0.7737

Epoch 34/60
----------
train Loss: 0.0059 Acc: 0.8473
val Loss: 0.0097 Acc: 0.7737

Epoch 35/60
----------
train Loss: 0.0059 Acc: 0.8545
val Loss: 0.0103 Acc: 0.7737

Epoch 36/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0059 Acc: 0.8509
val Loss: 0.0104 Acc: 0.7737

Epoch 37/60
----------
train Loss: 0.0062 Acc: 0.8473
val Loss: 0.0111 Acc: 0.7737

Epoch 38/60
----------
train Loss: 0.0059 Acc: 0.8636
val Loss: 0.0105 Acc: 0.7737

Epoch 39/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0108 Acc: 0.7737

Epoch 40/60
----------
train Loss: 0.0059 Acc: 0.8509
val Loss: 0.0098 Acc: 0.7737

Epoch 41/60
----------
train Loss: 0.0059 Acc: 0.8473
val Loss: 0.0103 Acc: 0.7737

Epoch 42/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0060 Acc: 0.8509
val Loss: 0.0104 Acc: 0.7810

Epoch 43/60
----------
train Loss: 0.0058 Acc: 0.8655
val Loss: 0.0106 Acc: 0.7737

Epoch 44/60
----------
train Loss: 0.0060 Acc: 0.8582
val Loss: 0.0113 Acc: 0.7737

Epoch 45/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0058 Acc: 0.8655
val Loss: 0.0111 Acc: 0.7737

Epoch 46/60
----------
train Loss: 0.0057 Acc: 0.8764
val Loss: 0.0105 Acc: 0.7737

Epoch 47/60
----------
train Loss: 0.0059 Acc: 0.8582
val Loss: 0.0108 Acc: 0.7737

Epoch 48/60
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0110 Acc: 0.7737

Epoch 49/60
----------
train Loss: 0.0057 Acc: 0.8618
val Loss: 0.0104 Acc: 0.7737

Epoch 50/60
----------
train Loss: 0.0057 Acc: 0.8400
val Loss: 0.0107 Acc: 0.7737

Epoch 51/60
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0059 Acc: 0.8527
val Loss: 0.0112 Acc: 0.7737

Epoch 52/60
----------
train Loss: 0.0058 Acc: 0.8673
val Loss: 0.0104 Acc: 0.7737

Epoch 53/60
----------
train Loss: 0.0058 Acc: 0.8527
val Loss: 0.0109 Acc: 0.7737

Epoch 54/60
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0058 Acc: 0.8564
val Loss: 0.0111 Acc: 0.7737

Epoch 55/60
----------
train Loss: 0.0059 Acc: 0.8491
val Loss: 0.0108 Acc: 0.7737

Epoch 56/60
----------
train Loss: 0.0057 Acc: 0.8509
val Loss: 0.0098 Acc: 0.7737

Epoch 57/60
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0060 Acc: 0.8455
val Loss: 0.0099 Acc: 0.7737

Epoch 58/60
----------
train Loss: 0.0058 Acc: 0.8655
val Loss: 0.0107 Acc: 0.7737

Epoch 59/60
----------
train Loss: 0.0058 Acc: 0.8545
val Loss: 0.0108 Acc: 0.7737

Epoch 60/60
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0058 Acc: 0.8545
val Loss: 0.0105 Acc: 0.7737

Training complete in 3m 2s
Best val Acc: 0.788321

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0056 Acc: 0.8509
val Loss: 0.0094 Acc: 0.7956

Epoch 1/60
----------
train Loss: 0.0027 Acc: 0.9618
val Loss: 0.0075 Acc: 0.8540

Epoch 2/60
----------
train Loss: 0.0013 Acc: 0.9800
val Loss: 0.0048 Acc: 0.8686

Epoch 3/60
----------
LR is set to 0.001
train Loss: 0.0007 Acc: 0.9945
val Loss: 0.0050 Acc: 0.8613

Epoch 4/60
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8613

Epoch 5/60
----------
train Loss: 0.0006 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8759

Epoch 6/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0051 Acc: 0.8759

Epoch 7/60
----------
train Loss: 0.0004 Acc: 0.9982
val Loss: 0.0051 Acc: 0.8759

Epoch 8/60
----------
train Loss: 0.0005 Acc: 0.9909
val Loss: 0.0051 Acc: 0.8686

Epoch 9/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0053 Acc: 0.8686

Epoch 10/60
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8686

Epoch 11/60
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8686

Epoch 12/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8686

Epoch 13/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8613

Epoch 14/60
----------
train Loss: 0.0006 Acc: 0.9964
val Loss: 0.0042 Acc: 0.8613

Epoch 15/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0053 Acc: 0.8613

Epoch 16/60
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8686

Epoch 17/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8613

Epoch 18/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8686

Epoch 19/60
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8686

Epoch 20/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0043 Acc: 0.8686

Epoch 21/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0054 Acc: 0.8686

Epoch 22/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0059 Acc: 0.8686

Epoch 23/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0057 Acc: 0.8686

Epoch 24/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0047 Acc: 0.8686

Epoch 25/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0047 Acc: 0.8686

Epoch 26/60
----------
train Loss: 0.0004 Acc: 0.9964
val Loss: 0.0046 Acc: 0.8686

Epoch 27/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8686

Epoch 28/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0055 Acc: 0.8686

Epoch 29/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0050 Acc: 0.8686

Epoch 30/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0046 Acc: 0.8686

Epoch 31/60
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8686

Epoch 32/60
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0047 Acc: 0.8686

Epoch 33/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0046 Acc: 0.8686

Epoch 34/60
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0055 Acc: 0.8686

Epoch 35/60
----------
train Loss: 0.0005 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8686

Epoch 36/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0049 Acc: 0.8686

Epoch 37/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0054 Acc: 0.8686

Epoch 38/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8686

Epoch 39/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 40/60
----------
train Loss: 0.0004 Acc: 0.9982
val Loss: 0.0053 Acc: 0.8686

Epoch 41/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0051 Acc: 0.8686

Epoch 42/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0004 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8686

Epoch 43/60
----------
train Loss: 0.0006 Acc: 0.9982
val Loss: 0.0067 Acc: 0.8686

Epoch 44/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0047 Acc: 0.8686

Epoch 45/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0005 Acc: 0.9927
val Loss: 0.0046 Acc: 0.8686

Epoch 46/60
----------
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0049 Acc: 0.8686

Epoch 47/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0039 Acc: 0.8686

Epoch 48/60
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0004 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8686

Epoch 49/60
----------
train Loss: 0.0005 Acc: 0.9927
val Loss: 0.0051 Acc: 0.8686

Epoch 50/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8686

Epoch 51/60
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0052 Acc: 0.8686

Epoch 52/60
----------
train Loss: 0.0004 Acc: 0.9982
val Loss: 0.0049 Acc: 0.8686

Epoch 53/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0056 Acc: 0.8686

Epoch 54/60
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0049 Acc: 0.8686

Epoch 55/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0057 Acc: 0.8686

Epoch 56/60
----------
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0046 Acc: 0.8686

Epoch 57/60
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0058 Acc: 0.8686

Epoch 58/60
----------
train Loss: 0.0005 Acc: 0.9982
val Loss: 0.0050 Acc: 0.8686

Epoch 59/60
----------
train Loss: 0.0005 Acc: 0.9945
val Loss: 0.0052 Acc: 0.8686

Epoch 60/60
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0005 Acc: 0.9964
val Loss: 0.0050 Acc: 0.8686

Training complete in 3m 12s
Best val Acc: 0.875912

---Testing---
Test accuracy: 0.970888
--------------------
Accuracy of Carcharhiniformes : 98 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 100 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 97 %
mean: 0.960422077922078, std: 0.04648503627205602
--------------------

run info[val: 0.25, epoch: 79, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/78
----------
LR is set to 0.01
train Loss: 0.0243 Acc: 0.2074
val Loss: 0.0213 Acc: 0.3216

Epoch 1/78
----------
train Loss: 0.0168 Acc: 0.5000
val Loss: 0.0149 Acc: 0.5848

Epoch 2/78
----------
train Loss: 0.0109 Acc: 0.6705
val Loss: 0.0099 Acc: 0.7135

Epoch 3/78
----------
train Loss: 0.0084 Acc: 0.7442
val Loss: 0.0097 Acc: 0.7427

Epoch 4/78
----------
train Loss: 0.0067 Acc: 0.8062
val Loss: 0.0072 Acc: 0.8187

Epoch 5/78
----------
train Loss: 0.0052 Acc: 0.8643
val Loss: 0.0067 Acc: 0.8363

Epoch 6/78
----------
train Loss: 0.0047 Acc: 0.8837
val Loss: 0.0065 Acc: 0.8304

Epoch 7/78
----------
train Loss: 0.0038 Acc: 0.8953
val Loss: 0.0065 Acc: 0.8480

Epoch 8/78
----------
train Loss: 0.0036 Acc: 0.9012
val Loss: 0.0063 Acc: 0.8304

Epoch 9/78
----------
train Loss: 0.0038 Acc: 0.9167
val Loss: 0.0065 Acc: 0.8480

Epoch 10/78
----------
train Loss: 0.0033 Acc: 0.9264
val Loss: 0.0061 Acc: 0.8421

Epoch 11/78
----------
LR is set to 0.001
train Loss: 0.0034 Acc: 0.9109
val Loss: 0.0060 Acc: 0.8421

Epoch 12/78
----------
train Loss: 0.0033 Acc: 0.9186
val Loss: 0.0062 Acc: 0.8363

Epoch 13/78
----------
train Loss: 0.0030 Acc: 0.9283
val Loss: 0.0063 Acc: 0.8421

Epoch 14/78
----------
train Loss: 0.0034 Acc: 0.9264
val Loss: 0.0061 Acc: 0.8480

Epoch 15/78
----------
train Loss: 0.0032 Acc: 0.9283
val Loss: 0.0058 Acc: 0.8421

Epoch 16/78
----------
train Loss: 0.0028 Acc: 0.9302
val Loss: 0.0060 Acc: 0.8363

Epoch 17/78
----------
train Loss: 0.0032 Acc: 0.9399
val Loss: 0.0063 Acc: 0.8363

Epoch 18/78
----------
train Loss: 0.0027 Acc: 0.9438
val Loss: 0.0065 Acc: 0.8363

Epoch 19/78
----------
train Loss: 0.0028 Acc: 0.9438
val Loss: 0.0060 Acc: 0.8363

Epoch 20/78
----------
train Loss: 0.0031 Acc: 0.9380
val Loss: 0.0061 Acc: 0.8480

Epoch 21/78
----------
train Loss: 0.0029 Acc: 0.9477
val Loss: 0.0062 Acc: 0.8421

Epoch 22/78
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9457
val Loss: 0.0061 Acc: 0.8480

Epoch 23/78
----------
train Loss: 0.0029 Acc: 0.9457
val Loss: 0.0060 Acc: 0.8480

Epoch 24/78
----------
train Loss: 0.0031 Acc: 0.9283
val Loss: 0.0058 Acc: 0.8538

Epoch 25/78
----------
train Loss: 0.0028 Acc: 0.9554
val Loss: 0.0061 Acc: 0.8480

Epoch 26/78
----------
train Loss: 0.0028 Acc: 0.9496
val Loss: 0.0060 Acc: 0.8538

Epoch 27/78
----------
train Loss: 0.0026 Acc: 0.9477
val Loss: 0.0062 Acc: 0.8480

Epoch 28/78
----------
train Loss: 0.0031 Acc: 0.9399
val Loss: 0.0059 Acc: 0.8363

Epoch 29/78
----------
train Loss: 0.0032 Acc: 0.9264
val Loss: 0.0058 Acc: 0.8421

Epoch 30/78
----------
train Loss: 0.0029 Acc: 0.9322
val Loss: 0.0060 Acc: 0.8480

Epoch 31/78
----------
train Loss: 0.0033 Acc: 0.9477
val Loss: 0.0060 Acc: 0.8363

Epoch 32/78
----------
train Loss: 0.0025 Acc: 0.9516
val Loss: 0.0061 Acc: 0.8421

Epoch 33/78
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0028 Acc: 0.9380
val Loss: 0.0061 Acc: 0.8538

Epoch 34/78
----------
train Loss: 0.0027 Acc: 0.9477
val Loss: 0.0061 Acc: 0.8538

Epoch 35/78
----------
train Loss: 0.0031 Acc: 0.9593
val Loss: 0.0060 Acc: 0.8480

Epoch 36/78
----------
train Loss: 0.0027 Acc: 0.9496
val Loss: 0.0057 Acc: 0.8421

Epoch 37/78
----------
train Loss: 0.0025 Acc: 0.9477
val Loss: 0.0058 Acc: 0.8421

Epoch 38/78
----------
train Loss: 0.0026 Acc: 0.9516
val Loss: 0.0062 Acc: 0.8421

Epoch 39/78
----------
train Loss: 0.0028 Acc: 0.9438
val Loss: 0.0057 Acc: 0.8421

Epoch 40/78
----------
train Loss: 0.0028 Acc: 0.9496
val Loss: 0.0059 Acc: 0.8480

Epoch 41/78
----------
train Loss: 0.0030 Acc: 0.9496
val Loss: 0.0057 Acc: 0.8538

Epoch 42/78
----------
train Loss: 0.0027 Acc: 0.9477
val Loss: 0.0058 Acc: 0.8480

Epoch 43/78
----------
train Loss: 0.0026 Acc: 0.9535
val Loss: 0.0058 Acc: 0.8480

Epoch 44/78
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0032 Acc: 0.9457
val Loss: 0.0058 Acc: 0.8421

Epoch 45/78
----------
train Loss: 0.0026 Acc: 0.9496
val Loss: 0.0057 Acc: 0.8421

Epoch 46/78
----------
train Loss: 0.0028 Acc: 0.9341
val Loss: 0.0060 Acc: 0.8480

Epoch 47/78
----------
train Loss: 0.0030 Acc: 0.9438
val Loss: 0.0058 Acc: 0.8421

Epoch 48/78
----------
train Loss: 0.0031 Acc: 0.9302
val Loss: 0.0062 Acc: 0.8421

Epoch 49/78
----------
train Loss: 0.0027 Acc: 0.9360
val Loss: 0.0058 Acc: 0.8480

Epoch 50/78
----------
train Loss: 0.0031 Acc: 0.9496
val Loss: 0.0059 Acc: 0.8538

Epoch 51/78
----------
train Loss: 0.0030 Acc: 0.9477
val Loss: 0.0059 Acc: 0.8538

Epoch 52/78
----------
train Loss: 0.0026 Acc: 0.9399
val Loss: 0.0059 Acc: 0.8363

Epoch 53/78
----------
train Loss: 0.0030 Acc: 0.9438
val Loss: 0.0059 Acc: 0.8421

Epoch 54/78
----------
train Loss: 0.0028 Acc: 0.9535
val Loss: 0.0058 Acc: 0.8421

Epoch 55/78
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9419
val Loss: 0.0060 Acc: 0.8480

Epoch 56/78
----------
train Loss: 0.0028 Acc: 0.9438
val Loss: 0.0062 Acc: 0.8421

Epoch 57/78
----------
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0061 Acc: 0.8421

Epoch 58/78
----------
train Loss: 0.0027 Acc: 0.9322
val Loss: 0.0057 Acc: 0.8480

Epoch 59/78
----------
train Loss: 0.0026 Acc: 0.9535
val Loss: 0.0061 Acc: 0.8421

Epoch 60/78
----------
train Loss: 0.0031 Acc: 0.9302
val Loss: 0.0062 Acc: 0.8480

Epoch 61/78
----------
train Loss: 0.0028 Acc: 0.9477
val Loss: 0.0060 Acc: 0.8480

Epoch 62/78
----------
train Loss: 0.0027 Acc: 0.9496
val Loss: 0.0064 Acc: 0.8480

Epoch 63/78
----------
train Loss: 0.0034 Acc: 0.9496
val Loss: 0.0063 Acc: 0.8421

Epoch 64/78
----------
train Loss: 0.0034 Acc: 0.9341
val Loss: 0.0061 Acc: 0.8421

Epoch 65/78
----------
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0061 Acc: 0.8421

Epoch 66/78
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0030 Acc: 0.9477
val Loss: 0.0056 Acc: 0.8421

Epoch 67/78
----------
train Loss: 0.0029 Acc: 0.9419
val Loss: 0.0060 Acc: 0.8421

Epoch 68/78
----------
train Loss: 0.0025 Acc: 0.9554
val Loss: 0.0059 Acc: 0.8421

Epoch 69/78
----------
train Loss: 0.0026 Acc: 0.9535
val Loss: 0.0060 Acc: 0.8538

Epoch 70/78
----------
train Loss: 0.0033 Acc: 0.9380
val Loss: 0.0059 Acc: 0.8538

Epoch 71/78
----------
train Loss: 0.0029 Acc: 0.9574
val Loss: 0.0060 Acc: 0.8538

Epoch 72/78
----------
train Loss: 0.0029 Acc: 0.9322
val Loss: 0.0060 Acc: 0.8421

Epoch 73/78
----------
train Loss: 0.0029 Acc: 0.9302
val Loss: 0.0060 Acc: 0.8421

Epoch 74/78
----------
train Loss: 0.0027 Acc: 0.9457
val Loss: 0.0059 Acc: 0.8421

Epoch 75/78
----------
train Loss: 0.0028 Acc: 0.9341
val Loss: 0.0058 Acc: 0.8480

Epoch 76/78
----------
train Loss: 0.0030 Acc: 0.9380
val Loss: 0.0061 Acc: 0.8480

Epoch 77/78
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0029 Acc: 0.9438
val Loss: 0.0059 Acc: 0.8596

Epoch 78/78
----------
train Loss: 0.0029 Acc: 0.9554
val Loss: 0.0061 Acc: 0.8480

Training complete in 3m 59s
Best val Acc: 0.859649

---Fine tuning.---
Epoch 0/78
----------
LR is set to 0.01
train Loss: 0.0024 Acc: 0.9535
val Loss: 0.0056 Acc: 0.8713

Epoch 1/78
----------
train Loss: 0.0021 Acc: 0.9651
val Loss: 0.0057 Acc: 0.8772

Epoch 2/78
----------
train Loss: 0.0019 Acc: 0.9593
val Loss: 0.0080 Acc: 0.8070

Epoch 3/78
----------
train Loss: 0.0015 Acc: 0.9496
val Loss: 0.0089 Acc: 0.7310

Epoch 4/78
----------
train Loss: 0.0008 Acc: 0.9826
val Loss: 0.0076 Acc: 0.8070

Epoch 5/78
----------
train Loss: 0.0010 Acc: 0.9826
val Loss: 0.0069 Acc: 0.8304

Epoch 6/78
----------
train Loss: 0.0009 Acc: 0.9884
val Loss: 0.0078 Acc: 0.8421

Epoch 7/78
----------
train Loss: 0.0007 Acc: 0.9826
val Loss: 0.0075 Acc: 0.8480

Epoch 8/78
----------
train Loss: 0.0007 Acc: 0.9806
val Loss: 0.0067 Acc: 0.8480

Epoch 9/78
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8655

Epoch 10/78
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0056 Acc: 0.8713

Epoch 11/78
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9942
val Loss: 0.0053 Acc: 0.8772

Epoch 12/78
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8889

Epoch 13/78
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8889

Epoch 14/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0052 Acc: 0.8889

Epoch 15/78
----------
train Loss: 0.0004 Acc: 0.9961
val Loss: 0.0056 Acc: 0.8947

Epoch 16/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8889

Epoch 17/78
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0057 Acc: 0.8889

Epoch 18/78
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0055 Acc: 0.9006

Epoch 19/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8947

Epoch 20/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0054 Acc: 0.9006

Epoch 21/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0053 Acc: 0.9006

Epoch 22/78
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 23/78
----------
train Loss: 0.0008 Acc: 0.9981
val Loss: 0.0051 Acc: 0.8889

Epoch 24/78
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0058 Acc: 0.9006

Epoch 25/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8947

Epoch 26/78
----------
train Loss: 0.0003 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9006

Epoch 27/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0058 Acc: 0.9006

Epoch 28/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8947

Epoch 29/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0055 Acc: 0.9006

Epoch 30/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8947

Epoch 31/78
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8947

Epoch 32/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0056 Acc: 0.8947

Epoch 33/78
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 34/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8889

Epoch 35/78
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0050 Acc: 0.8889

Epoch 36/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8947

Epoch 37/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8947

Epoch 38/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8947

Epoch 39/78
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8947

Epoch 40/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8947

Epoch 41/78
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0053 Acc: 0.8889

Epoch 42/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8947

Epoch 43/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8947

Epoch 44/78
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8947

Epoch 45/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0054 Acc: 0.8947

Epoch 46/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8947

Epoch 47/78
----------
train Loss: 0.0011 Acc: 0.9922
val Loss: 0.0051 Acc: 0.8947

Epoch 48/78
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0054 Acc: 0.9006

Epoch 49/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8947

Epoch 50/78
----------
train Loss: 0.0002 Acc: 0.9961
val Loss: 0.0053 Acc: 0.9006

Epoch 51/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 52/78
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0054 Acc: 0.8947

Epoch 53/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8947

Epoch 54/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8947

Epoch 55/78
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 56/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0050 Acc: 0.8947

Epoch 57/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8947

Epoch 58/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8889

Epoch 59/78
----------
train Loss: 0.0005 Acc: 0.9981
val Loss: 0.0055 Acc: 0.8947

Epoch 60/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 61/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.9006

Epoch 62/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0057 Acc: 0.9006

Epoch 63/78
----------
train Loss: 0.0003 Acc: 0.9981
val Loss: 0.0056 Acc: 0.9006

Epoch 64/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.9006

Epoch 65/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8947

Epoch 66/78
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9006

Epoch 67/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8889

Epoch 68/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8889

Epoch 69/78
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8947

Epoch 70/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.9006

Epoch 71/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8889

Epoch 72/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8889

Epoch 73/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8889

Epoch 74/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8889

Epoch 75/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8947

Epoch 76/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0052 Acc: 0.8889

Epoch 77/78
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8889

Epoch 78/78
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8947

Training complete in 4m 9s
Best val Acc: 0.900585

---Testing---
Test accuracy: 0.975255
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Lamniformes : 94 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 98 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 96 %
mean: 0.9712282766097673, std: 0.016458896061730146
--------------------

run info[val: 0.3, epoch: 55, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0214 Acc: 0.1746
val Loss: 0.0272 Acc: 0.3883

Epoch 1/54
----------
train Loss: 0.0152 Acc: 0.4927
val Loss: 0.0161 Acc: 0.6699

Epoch 2/54
----------
train Loss: 0.0100 Acc: 0.7110
val Loss: 0.0118 Acc: 0.7524

Epoch 3/54
----------
train Loss: 0.0069 Acc: 0.8087
val Loss: 0.0078 Acc: 0.7961

Epoch 4/54
----------
train Loss: 0.0054 Acc: 0.8711
val Loss: 0.0105 Acc: 0.8058

Epoch 5/54
----------
train Loss: 0.0047 Acc: 0.8420
val Loss: 0.0066 Acc: 0.8495

Epoch 6/54
----------
train Loss: 0.0041 Acc: 0.8919
val Loss: 0.0062 Acc: 0.8252

Epoch 7/54
----------
train Loss: 0.0035 Acc: 0.8940
val Loss: 0.0057 Acc: 0.8301

Epoch 8/54
----------
train Loss: 0.0030 Acc: 0.9439
val Loss: 0.0064 Acc: 0.8495

Epoch 9/54
----------
train Loss: 0.0029 Acc: 0.9293
val Loss: 0.0057 Acc: 0.8398

Epoch 10/54
----------
train Loss: 0.0025 Acc: 0.9439
val Loss: 0.0126 Acc: 0.8592

Epoch 11/54
----------
train Loss: 0.0022 Acc: 0.9605
val Loss: 0.0066 Acc: 0.8398

Epoch 12/54
----------
LR is set to 0.001
train Loss: 0.0021 Acc: 0.9605
val Loss: 0.0062 Acc: 0.8398

Epoch 13/54
----------
train Loss: 0.0021 Acc: 0.9501
val Loss: 0.0090 Acc: 0.8544

Epoch 14/54
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0070 Acc: 0.8592

Epoch 15/54
----------
train Loss: 0.0020 Acc: 0.9647
val Loss: 0.0063 Acc: 0.8641

Epoch 16/54
----------
train Loss: 0.0021 Acc: 0.9626
val Loss: 0.0065 Acc: 0.8544

Epoch 17/54
----------
train Loss: 0.0020 Acc: 0.9709
val Loss: 0.0063 Acc: 0.8592

Epoch 18/54
----------
train Loss: 0.0019 Acc: 0.9771
val Loss: 0.0056 Acc: 0.8447

Epoch 19/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0066 Acc: 0.8544

Epoch 20/54
----------
train Loss: 0.0017 Acc: 0.9751
val Loss: 0.0093 Acc: 0.8544

Epoch 21/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0085 Acc: 0.8544

Epoch 22/54
----------
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0066 Acc: 0.8592

Epoch 23/54
----------
train Loss: 0.0019 Acc: 0.9730
val Loss: 0.0067 Acc: 0.8641

Epoch 24/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0050 Acc: 0.8592

Epoch 25/54
----------
train Loss: 0.0019 Acc: 0.9813
val Loss: 0.0070 Acc: 0.8592

Epoch 26/54
----------
train Loss: 0.0020 Acc: 0.9667
val Loss: 0.0062 Acc: 0.8592

Epoch 27/54
----------
train Loss: 0.0018 Acc: 0.9688
val Loss: 0.0051 Acc: 0.8544

Epoch 28/54
----------
train Loss: 0.0019 Acc: 0.9626
val Loss: 0.0087 Acc: 0.8592

Epoch 29/54
----------
train Loss: 0.0019 Acc: 0.9667
val Loss: 0.0067 Acc: 0.8641

Epoch 30/54
----------
train Loss: 0.0018 Acc: 0.9688
val Loss: 0.0078 Acc: 0.8592

Epoch 31/54
----------
train Loss: 0.0019 Acc: 0.9626
val Loss: 0.0048 Acc: 0.8495

Epoch 32/54
----------
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0090 Acc: 0.8495

Epoch 33/54
----------
train Loss: 0.0019 Acc: 0.9667
val Loss: 0.0091 Acc: 0.8592

Epoch 34/54
----------
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0048 Acc: 0.8544

Epoch 35/54
----------
train Loss: 0.0018 Acc: 0.9771
val Loss: 0.0049 Acc: 0.8495

Epoch 36/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0055 Acc: 0.8495

Epoch 37/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0047 Acc: 0.8544

Epoch 38/54
----------
train Loss: 0.0019 Acc: 0.9730
val Loss: 0.0050 Acc: 0.8495

Epoch 39/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0053 Acc: 0.8592

Epoch 40/54
----------
train Loss: 0.0019 Acc: 0.9667
val Loss: 0.0058 Acc: 0.8641

Epoch 41/54
----------
train Loss: 0.0019 Acc: 0.9792
val Loss: 0.0067 Acc: 0.8544

Epoch 42/54
----------
train Loss: 0.0018 Acc: 0.9834
val Loss: 0.0080 Acc: 0.8544

Epoch 43/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0078 Acc: 0.8592

Epoch 44/54
----------
train Loss: 0.0020 Acc: 0.9626
val Loss: 0.0055 Acc: 0.8592

Epoch 45/54
----------
train Loss: 0.0019 Acc: 0.9709
val Loss: 0.0052 Acc: 0.8544

Epoch 46/54
----------
train Loss: 0.0019 Acc: 0.9647
val Loss: 0.0077 Acc: 0.8641

Epoch 47/54
----------
train Loss: 0.0018 Acc: 0.9626
val Loss: 0.0054 Acc: 0.8544

Epoch 48/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0018 Acc: 0.9688
val Loss: 0.0051 Acc: 0.8592

Epoch 49/54
----------
train Loss: 0.0019 Acc: 0.9771
val Loss: 0.0069 Acc: 0.8592

Epoch 50/54
----------
train Loss: 0.0018 Acc: 0.9771
val Loss: 0.0083 Acc: 0.8544

Epoch 51/54
----------
train Loss: 0.0020 Acc: 0.9667
val Loss: 0.0078 Acc: 0.8641

Epoch 52/54
----------
train Loss: 0.0019 Acc: 0.9626
val Loss: 0.0054 Acc: 0.8641

Epoch 53/54
----------
train Loss: 0.0019 Acc: 0.9730
val Loss: 0.0066 Acc: 0.8641

Epoch 54/54
----------
train Loss: 0.0019 Acc: 0.9688
val Loss: 0.0047 Acc: 0.8641

Training complete in 2m 42s
Best val Acc: 0.864078

---Fine tuning.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0019 Acc: 0.9626
val Loss: 0.0069 Acc: 0.8689

Epoch 1/54
----------
train Loss: 0.0010 Acc: 0.9875
val Loss: 0.0082 Acc: 0.8835

Epoch 2/54
----------
train Loss: 0.0005 Acc: 0.9979
val Loss: 0.0052 Acc: 0.8689

Epoch 3/54
----------
train Loss: 0.0002 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8738

Epoch 4/54
----------
train Loss: 0.0002 Acc: 0.9979
val Loss: 0.0055 Acc: 0.8786

Epoch 5/54
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8738

Epoch 6/54
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8786

Epoch 7/54
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8786

Epoch 8/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.8738

Epoch 9/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8835

Epoch 10/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8835

Epoch 11/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0109 Acc: 0.8883

Epoch 12/54
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8883

Epoch 13/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0074 Acc: 0.8883

Epoch 14/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0042 Acc: 0.8883

Epoch 15/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0079 Acc: 0.8883

Epoch 16/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 17/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8883

Epoch 18/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 19/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 20/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8883

Epoch 21/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8883

Epoch 22/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8883

Epoch 23/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 24/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0069 Acc: 0.8835

Epoch 25/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0039 Acc: 0.8883

Epoch 26/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8835

Epoch 27/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0072 Acc: 0.8835

Epoch 28/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8835

Epoch 29/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8835

Epoch 30/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0051 Acc: 0.8835

Epoch 31/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8835

Epoch 32/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0056 Acc: 0.8835

Epoch 33/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8835

Epoch 34/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0063 Acc: 0.8835

Epoch 35/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0055 Acc: 0.8883

Epoch 36/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 37/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0064 Acc: 0.8883

Epoch 38/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.8883

Epoch 39/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0101 Acc: 0.8883

Epoch 40/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.8835

Epoch 41/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8835

Epoch 42/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0110 Acc: 0.8835

Epoch 43/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8883

Epoch 44/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0067 Acc: 0.8883

Epoch 45/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8883

Epoch 46/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0054 Acc: 0.8883

Epoch 47/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0043 Acc: 0.8883

Epoch 48/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 49/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0061 Acc: 0.8883

Epoch 50/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0057 Acc: 0.8835

Epoch 51/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0048 Acc: 0.8883

Epoch 52/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0053 Acc: 0.8883

Epoch 53/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0037 Acc: 0.8883

Epoch 54/54
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0046 Acc: 0.8883

Training complete in 2m 51s
Best val Acc: 0.888350

---Testing---
Test accuracy: 0.966521
--------------------
Accuracy of Carcharhiniformes : 99 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Lamniformes : 85 %
Accuracy of Orectolobiformes : 98 %
Accuracy of Pristiformes : 99 %
Accuracy of Squaliformes : 89 %
Accuracy of Squatiniformes : 97 %
mean: 0.9547446350011698, std: 0.04759988129303533

Model saved in "./weights/shark_[0.98]_mean[0.97]_std[0.03].save".

Process finished with exit code 0
'''