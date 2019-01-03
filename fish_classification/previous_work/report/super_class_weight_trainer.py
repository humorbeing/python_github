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

train_this = 'super_class'
test_this = 'super_class'


def trainer(trainee, testee):
    dataset_dir = './datasets/'
    data_dir = dataset_dir+trainee
    test_dir = dataset_dir+testee

    val_s = [0.1, 0.2, 0.3]
    scale_size = 224
    all_time_best = 0
    all_time_best_acc = 0.0
    performance = list()

    for runs in range(3):
        valid_size = val_s[runs]
        batch_size = 100
        num_workers = 2
        EPOCH = np.random.randint(30, 100)
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

for _ in range(5):
    trainer(train_this, test_this)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/home/visbic/python/pytorch_playground/datasets/new fish/git-test/super_class_weight_trainer.py"
--------------------

run info[val: 0.1, epoch: 50, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0048 Acc: 0.8105
val Loss: 0.0023 Acc: 0.9315

Epoch 1/49
----------
train Loss: 0.0024 Acc: 0.9106
val Loss: 0.0028 Acc: 0.9041

Epoch 2/49
----------
train Loss: 0.0022 Acc: 0.9202
val Loss: 0.0022 Acc: 0.9498

Epoch 3/49
----------
train Loss: 0.0017 Acc: 0.9358
val Loss: 0.0019 Acc: 0.9498

Epoch 4/49
----------
train Loss: 0.0016 Acc: 0.9444
val Loss: 0.0021 Acc: 0.9269

Epoch 5/49
----------
train Loss: 0.0015 Acc: 0.9429
val Loss: 0.0015 Acc: 0.9726

Epoch 6/49
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9555
val Loss: 0.0016 Acc: 0.9589

Epoch 7/49
----------
train Loss: 0.0013 Acc: 0.9555
val Loss: 0.0017 Acc: 0.9589

Epoch 8/49
----------
train Loss: 0.0012 Acc: 0.9535
val Loss: 0.0017 Acc: 0.9589

Epoch 9/49
----------
train Loss: 0.0012 Acc: 0.9570
val Loss: 0.0013 Acc: 0.9589

Epoch 10/49
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0025 Acc: 0.9589

Epoch 11/49
----------
train Loss: 0.0012 Acc: 0.9631
val Loss: 0.0028 Acc: 0.9589

Epoch 12/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0014 Acc: 0.9589

Epoch 13/49
----------
train Loss: 0.0012 Acc: 0.9545
val Loss: 0.0014 Acc: 0.9589

Epoch 14/49
----------
train Loss: 0.0013 Acc: 0.9596
val Loss: 0.0024 Acc: 0.9589

Epoch 15/49
----------
train Loss: 0.0012 Acc: 0.9570
val Loss: 0.0016 Acc: 0.9589

Epoch 16/49
----------
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0015 Acc: 0.9589

Epoch 17/49
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0024 Acc: 0.9589

Epoch 18/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0013 Acc: 0.9515
val Loss: 0.0013 Acc: 0.9589

Epoch 19/49
----------
train Loss: 0.0012 Acc: 0.9530
val Loss: 0.0018 Acc: 0.9589

Epoch 20/49
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0014 Acc: 0.9589

Epoch 21/49
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0017 Acc: 0.9589

Epoch 22/49
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0027 Acc: 0.9589

Epoch 23/49
----------
train Loss: 0.0012 Acc: 0.9570
val Loss: 0.0013 Acc: 0.9589

Epoch 24/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0013 Acc: 0.9520
val Loss: 0.0015 Acc: 0.9589

Epoch 25/49
----------
train Loss: 0.0013 Acc: 0.9576
val Loss: 0.0025 Acc: 0.9589

Epoch 26/49
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0016 Acc: 0.9589

Epoch 27/49
----------
train Loss: 0.0012 Acc: 0.9586
val Loss: 0.0019 Acc: 0.9589

Epoch 28/49
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0013 Acc: 0.9589

Epoch 29/49
----------
train Loss: 0.0012 Acc: 0.9555
val Loss: 0.0014 Acc: 0.9589

Epoch 30/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0012 Acc: 0.9535
val Loss: 0.0017 Acc: 0.9589

Epoch 31/49
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0019 Acc: 0.9589

Epoch 32/49
----------
train Loss: 0.0013 Acc: 0.9520
val Loss: 0.0014 Acc: 0.9589

Epoch 33/49
----------
train Loss: 0.0012 Acc: 0.9570
val Loss: 0.0025 Acc: 0.9589

Epoch 34/49
----------
train Loss: 0.0012 Acc: 0.9560
val Loss: 0.0013 Acc: 0.9589

Epoch 35/49
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0026 Acc: 0.9589

Epoch 36/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0019 Acc: 0.9589

Epoch 37/49
----------
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0020 Acc: 0.9589

Epoch 38/49
----------
train Loss: 0.0012 Acc: 0.9570
val Loss: 0.0013 Acc: 0.9589

Epoch 39/49
----------
train Loss: 0.0012 Acc: 0.9611
val Loss: 0.0014 Acc: 0.9589

Epoch 40/49
----------
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0017 Acc: 0.9589

Epoch 41/49
----------
train Loss: 0.0011 Acc: 0.9656
val Loss: 0.0018 Acc: 0.9589

Epoch 42/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0024 Acc: 0.9589

Epoch 43/49
----------
train Loss: 0.0013 Acc: 0.9540
val Loss: 0.0015 Acc: 0.9589

Epoch 44/49
----------
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0013 Acc: 0.9589

Epoch 45/49
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0015 Acc: 0.9589

Epoch 46/49
----------
train Loss: 0.0011 Acc: 0.9565
val Loss: 0.0014 Acc: 0.9589

Epoch 47/49
----------
train Loss: 0.0012 Acc: 0.9540
val Loss: 0.0018 Acc: 0.9589

Epoch 48/49
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9555
val Loss: 0.0014 Acc: 0.9589

Epoch 49/49
----------
train Loss: 0.0012 Acc: 0.9545
val Loss: 0.0013 Acc: 0.9589

Best val Acc: 0.972603

---Fine tuning.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0012 Acc: 0.9555
val Loss: 0.0016 Acc: 0.9498

Epoch 1/49
----------
train Loss: 0.0006 Acc: 0.9773
val Loss: 0.0017 Acc: 0.9635

Epoch 2/49
----------
train Loss: 0.0004 Acc: 0.9914
val Loss: 0.0016 Acc: 0.9680

Epoch 3/49
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0017 Acc: 0.9726

Epoch 4/49
----------
train Loss: 0.0001 Acc: 0.9944
val Loss: 0.0016 Acc: 0.9543

Epoch 5/49
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0010 Acc: 0.9680

Epoch 6/49
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0020 Acc: 0.9726

Epoch 7/49
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0010 Acc: 0.9680

Epoch 8/49
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9680

Epoch 9/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 10/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 11/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 12/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0025 Acc: 0.9680

Epoch 13/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 14/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 15/49
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0010 Acc: 0.9680

Epoch 16/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 17/49
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0036 Acc: 0.9680

Epoch 18/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0010 Acc: 0.9680

Epoch 19/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 20/49
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0010 Acc: 0.9680

Epoch 21/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 22/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 23/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0030 Acc: 0.9680

Epoch 24/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 25/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 26/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0025 Acc: 0.9680

Epoch 27/49
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0040 Acc: 0.9680

Epoch 28/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 29/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 30/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 31/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 32/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 33/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0010 Acc: 0.9680

Epoch 34/49
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0010 Acc: 0.9680

Epoch 35/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 36/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0010 Acc: 0.9680

Epoch 37/49
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0018 Acc: 0.9680

Epoch 38/49
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0025 Acc: 0.9680

Epoch 39/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 40/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 41/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 42/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 43/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 44/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Epoch 45/49
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0010 Acc: 0.9680

Epoch 46/49
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0016 Acc: 0.9680

Epoch 47/49
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 48/49
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0010 Acc: 0.9680

Epoch 49/49
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0010 Acc: 0.9680

Best val Acc: 0.972603

---Testing---
Test accuracy: 0.995905
--------------------
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9942277769412238, std: 0.004692643306400678
--------------------

run info[val: 0.2, epoch: 46, randcrop: True, decay: 14]

---Training last layer.---
Epoch 0/45
----------
LR is set to 0.01
train Loss: 0.0061 Acc: 0.7396
val Loss: 0.0031 Acc: 0.9066

Epoch 1/45
----------
train Loss: 0.0023 Acc: 0.9119
val Loss: 0.0020 Acc: 0.9408

Epoch 2/45
----------
train Loss: 0.0018 Acc: 0.9352
val Loss: 0.0019 Acc: 0.9408

Epoch 3/45
----------
train Loss: 0.0016 Acc: 0.9471
val Loss: 0.0020 Acc: 0.9453

Epoch 4/45
----------
train Loss: 0.0018 Acc: 0.9301
val Loss: 0.0029 Acc: 0.8952

Epoch 5/45
----------
train Loss: 0.0016 Acc: 0.9369
val Loss: 0.0020 Acc: 0.9362

Epoch 6/45
----------
train Loss: 0.0013 Acc: 0.9517
val Loss: 0.0020 Acc: 0.9476

Epoch 7/45
----------
train Loss: 0.0013 Acc: 0.9574
val Loss: 0.0018 Acc: 0.9453

Epoch 8/45
----------
train Loss: 0.0012 Acc: 0.9568
val Loss: 0.0015 Acc: 0.9408

Epoch 9/45
----------
train Loss: 0.0012 Acc: 0.9551
val Loss: 0.0016 Acc: 0.9499

Epoch 10/45
----------
train Loss: 0.0010 Acc: 0.9659
val Loss: 0.0018 Acc: 0.9362

Epoch 11/45
----------
train Loss: 0.0012 Acc: 0.9585
val Loss: 0.0016 Acc: 0.9522

Epoch 12/45
----------
train Loss: 0.0010 Acc: 0.9670
val Loss: 0.0018 Acc: 0.9476

Epoch 13/45
----------
train Loss: 0.0011 Acc: 0.9653
val Loss: 0.0016 Acc: 0.9431

Epoch 14/45
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9670
val Loss: 0.0016 Acc: 0.9522

Epoch 15/45
----------
train Loss: 0.0010 Acc: 0.9687
val Loss: 0.0018 Acc: 0.9476

Epoch 16/45
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9499

Epoch 17/45
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0018 Acc: 0.9453

Epoch 18/45
----------
train Loss: 0.0009 Acc: 0.9693
val Loss: 0.0017 Acc: 0.9522

Epoch 19/45
----------
train Loss: 0.0010 Acc: 0.9636
val Loss: 0.0017 Acc: 0.9431

Epoch 20/45
----------
train Loss: 0.0009 Acc: 0.9750
val Loss: 0.0017 Acc: 0.9522

Epoch 21/45
----------
train Loss: 0.0010 Acc: 0.9676
val Loss: 0.0016 Acc: 0.9431

Epoch 22/45
----------
train Loss: 0.0010 Acc: 0.9699
val Loss: 0.0016 Acc: 0.9476

Epoch 23/45
----------
train Loss: 0.0009 Acc: 0.9738
val Loss: 0.0017 Acc: 0.9522

Epoch 24/45
----------
train Loss: 0.0010 Acc: 0.9693
val Loss: 0.0017 Acc: 0.9476

Epoch 25/45
----------
train Loss: 0.0009 Acc: 0.9665
val Loss: 0.0016 Acc: 0.9522

Epoch 26/45
----------
train Loss: 0.0009 Acc: 0.9727
val Loss: 0.0017 Acc: 0.9476

Epoch 27/45
----------
train Loss: 0.0009 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9522

Epoch 28/45
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9721
val Loss: 0.0017 Acc: 0.9522

Epoch 29/45
----------
train Loss: 0.0009 Acc: 0.9665
val Loss: 0.0017 Acc: 0.9544

Epoch 30/45
----------
train Loss: 0.0010 Acc: 0.9636
val Loss: 0.0016 Acc: 0.9499

Epoch 31/45
----------
train Loss: 0.0009 Acc: 0.9648
val Loss: 0.0016 Acc: 0.9522

Epoch 32/45
----------
train Loss: 0.0009 Acc: 0.9704
val Loss: 0.0016 Acc: 0.9476

Epoch 33/45
----------
train Loss: 0.0009 Acc: 0.9699
val Loss: 0.0016 Acc: 0.9499

Epoch 34/45
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0016 Acc: 0.9544

Epoch 35/45
----------
train Loss: 0.0009 Acc: 0.9733
val Loss: 0.0017 Acc: 0.9522

Epoch 36/45
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0020 Acc: 0.9499

Epoch 37/45
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0016 Acc: 0.9499

Epoch 38/45
----------
train Loss: 0.0010 Acc: 0.9665
val Loss: 0.0017 Acc: 0.9499

Epoch 39/45
----------
train Loss: 0.0009 Acc: 0.9716
val Loss: 0.0019 Acc: 0.9499

Epoch 40/45
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0023 Acc: 0.9499

Epoch 41/45
----------
train Loss: 0.0010 Acc: 0.9704
val Loss: 0.0017 Acc: 0.9544

Epoch 42/45
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0009 Acc: 0.9704
val Loss: 0.0017 Acc: 0.9522

Epoch 43/45
----------
train Loss: 0.0009 Acc: 0.9727
val Loss: 0.0016 Acc: 0.9476

Epoch 44/45
----------
train Loss: 0.0009 Acc: 0.9727
val Loss: 0.0016 Acc: 0.9522

Epoch 45/45
----------
train Loss: 0.0010 Acc: 0.9659
val Loss: 0.0017 Acc: 0.9522

Best val Acc: 0.954442

---Fine tuning.---
Epoch 0/45
----------
LR is set to 0.01
train Loss: 0.0013 Acc: 0.9608
val Loss: 0.0020 Acc: 0.9408

Epoch 1/45
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0027 Acc: 0.9431

Epoch 2/45
----------
train Loss: 0.0005 Acc: 0.9801
val Loss: 0.0014 Acc: 0.9681

Epoch 3/45
----------
train Loss: 0.0003 Acc: 0.9926
val Loss: 0.0014 Acc: 0.9658

Epoch 4/45
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0013 Acc: 0.9658

Epoch 5/45
----------
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0010 Acc: 0.9727

Epoch 6/45
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0010 Acc: 0.9772

Epoch 7/45
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9704

Epoch 8/45
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0017 Acc: 0.9658

Epoch 9/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0014 Acc: 0.9658

Epoch 10/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9658

Epoch 11/45
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0014 Acc: 0.9636

Epoch 12/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0020 Acc: 0.9681

Epoch 13/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9636

Epoch 14/45
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0015 Acc: 0.9636

Epoch 15/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9636

Epoch 16/45
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0016 Acc: 0.9681

Epoch 17/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0014 Acc: 0.9681

Epoch 18/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9681

Epoch 19/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9704

Epoch 20/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 21/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0017 Acc: 0.9704

Epoch 22/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9658

Epoch 23/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9658

Epoch 24/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 25/45
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0018 Acc: 0.9704

Epoch 26/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9704

Epoch 27/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9658

Epoch 28/45
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9681

Epoch 29/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 30/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9681

Epoch 31/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9704

Epoch 32/45
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0014 Acc: 0.9681

Epoch 33/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9704

Epoch 34/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9704

Epoch 35/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 36/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9658

Epoch 37/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9704

Epoch 38/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9681

Epoch 39/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 40/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9704

Epoch 41/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9704

Epoch 42/45
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9704

Epoch 43/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9681

Epoch 44/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0021 Acc: 0.9658

Epoch 45/45
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9681

Best val Acc: 0.977221

---Testing---
Test accuracy: 0.995450
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9926182469172983, std: 0.00942885250453149
--------------------

run info[val: 0.3, epoch: 54, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0059 Acc: 0.7492
val Loss: 0.0024 Acc: 0.9181

Epoch 1/53
----------
train Loss: 0.0023 Acc: 0.9214
val Loss: 0.0023 Acc: 0.9272

Epoch 2/53
----------
train Loss: 0.0022 Acc: 0.9207
val Loss: 0.0020 Acc: 0.9332

Epoch 3/53
----------
train Loss: 0.0018 Acc: 0.9409
val Loss: 0.0018 Acc: 0.9408

Epoch 4/53
----------
train Loss: 0.0017 Acc: 0.9357
val Loss: 0.0017 Acc: 0.9423

Epoch 5/53
----------
train Loss: 0.0016 Acc: 0.9344
val Loss: 0.0018 Acc: 0.9378

Epoch 6/53
----------
train Loss: 0.0014 Acc: 0.9480
val Loss: 0.0016 Acc: 0.9423

Epoch 7/53
----------
train Loss: 0.0013 Acc: 0.9526
val Loss: 0.0017 Acc: 0.9393

Epoch 8/53
----------
train Loss: 0.0012 Acc: 0.9623
val Loss: 0.0016 Acc: 0.9469

Epoch 9/53
----------
train Loss: 0.0011 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9469

Epoch 10/53
----------
LR is set to 0.001
train Loss: 0.0012 Acc: 0.9636
val Loss: 0.0016 Acc: 0.9439

Epoch 11/53
----------
train Loss: 0.0010 Acc: 0.9701
val Loss: 0.0016 Acc: 0.9514

Epoch 12/53
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0016 Acc: 0.9514

Epoch 13/53
----------
train Loss: 0.0010 Acc: 0.9714
val Loss: 0.0015 Acc: 0.9499

Epoch 14/53
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0017 Acc: 0.9484

Epoch 15/53
----------
train Loss: 0.0011 Acc: 0.9623
val Loss: 0.0015 Acc: 0.9514

Epoch 16/53
----------
train Loss: 0.0010 Acc: 0.9695
val Loss: 0.0016 Acc: 0.9454

Epoch 17/53
----------
train Loss: 0.0011 Acc: 0.9688
val Loss: 0.0016 Acc: 0.9514

Epoch 18/53
----------
train Loss: 0.0011 Acc: 0.9649
val Loss: 0.0015 Acc: 0.9499

Epoch 19/53
----------
train Loss: 0.0011 Acc: 0.9630
val Loss: 0.0015 Acc: 0.9484

Epoch 20/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0015 Acc: 0.9530

Epoch 21/53
----------
train Loss: 0.0011 Acc: 0.9649
val Loss: 0.0015 Acc: 0.9499

Epoch 22/53
----------
train Loss: 0.0011 Acc: 0.9649
val Loss: 0.0015 Acc: 0.9499

Epoch 23/53
----------
train Loss: 0.0011 Acc: 0.9675
val Loss: 0.0015 Acc: 0.9530

Epoch 24/53
----------
train Loss: 0.0011 Acc: 0.9623
val Loss: 0.0016 Acc: 0.9469

Epoch 25/53
----------
train Loss: 0.0010 Acc: 0.9662
val Loss: 0.0016 Acc: 0.9514

Epoch 26/53
----------
train Loss: 0.0010 Acc: 0.9662
val Loss: 0.0016 Acc: 0.9469

Epoch 27/53
----------
train Loss: 0.0010 Acc: 0.9695
val Loss: 0.0016 Acc: 0.9484

Epoch 28/53
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0015 Acc: 0.9469

Epoch 29/53
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9514

Epoch 30/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0011 Acc: 0.9708
val Loss: 0.0015 Acc: 0.9514

Epoch 31/53
----------
train Loss: 0.0010 Acc: 0.9688
val Loss: 0.0015 Acc: 0.9484

Epoch 32/53
----------
train Loss: 0.0010 Acc: 0.9688
val Loss: 0.0015 Acc: 0.9514

Epoch 33/53
----------
train Loss: 0.0010 Acc: 0.9714
val Loss: 0.0016 Acc: 0.9514

Epoch 34/53
----------
train Loss: 0.0010 Acc: 0.9721
val Loss: 0.0016 Acc: 0.9514

Epoch 35/53
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9530

Epoch 36/53
----------
train Loss: 0.0011 Acc: 0.9688
val Loss: 0.0016 Acc: 0.9454

Epoch 37/53
----------
train Loss: 0.0011 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9499

Epoch 38/53
----------
train Loss: 0.0011 Acc: 0.9656
val Loss: 0.0015 Acc: 0.9484

Epoch 39/53
----------
train Loss: 0.0010 Acc: 0.9675
val Loss: 0.0015 Acc: 0.9499

Epoch 40/53
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0015 Acc: 0.9514

Epoch 41/53
----------
train Loss: 0.0011 Acc: 0.9675
val Loss: 0.0015 Acc: 0.9469

Epoch 42/53
----------
train Loss: 0.0011 Acc: 0.9656
val Loss: 0.0016 Acc: 0.9499

Epoch 43/53
----------
train Loss: 0.0010 Acc: 0.9662
val Loss: 0.0015 Acc: 0.9530

Epoch 44/53
----------
train Loss: 0.0012 Acc: 0.9649
val Loss: 0.0015 Acc: 0.9454

Epoch 45/53
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0016 Acc: 0.9469

Epoch 46/53
----------
train Loss: 0.0011 Acc: 0.9656
val Loss: 0.0016 Acc: 0.9469

Epoch 47/53
----------
train Loss: 0.0009 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9514

Epoch 48/53
----------
train Loss: 0.0010 Acc: 0.9695
val Loss: 0.0015 Acc: 0.9439

Epoch 49/53
----------
train Loss: 0.0011 Acc: 0.9649
val Loss: 0.0016 Acc: 0.9530

Epoch 50/53
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0012 Acc: 0.9610
val Loss: 0.0016 Acc: 0.9469

Epoch 51/53
----------
train Loss: 0.0011 Acc: 0.9662
val Loss: 0.0015 Acc: 0.9469

Epoch 52/53
----------
train Loss: 0.0011 Acc: 0.9662
val Loss: 0.0015 Acc: 0.9484

Epoch 53/53
----------
train Loss: 0.0011 Acc: 0.9630
val Loss: 0.0016 Acc: 0.9499

Best val Acc: 0.952959

---Fine tuning.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0012 Acc: 0.9558
val Loss: 0.0016 Acc: 0.9469

Epoch 1/53
----------
train Loss: 0.0007 Acc: 0.9714
val Loss: 0.0015 Acc: 0.9530

Epoch 2/53
----------
train Loss: 0.0005 Acc: 0.9877
val Loss: 0.0013 Acc: 0.9651

Epoch 3/53
----------
train Loss: 0.0004 Acc: 0.9890
val Loss: 0.0016 Acc: 0.9545

Epoch 4/53
----------
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0018 Acc: 0.9484

Epoch 5/53
----------
train Loss: 0.0002 Acc: 0.9942
val Loss: 0.0010 Acc: 0.9712

Epoch 6/53
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0013 Acc: 0.9636

Epoch 7/53
----------
train Loss: 0.0002 Acc: 0.9955
val Loss: 0.0014 Acc: 0.9651

Epoch 8/53
----------
train Loss: 0.0001 Acc: 0.9961
val Loss: 0.0016 Acc: 0.9666

Epoch 9/53
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0013 Acc: 0.9605

Epoch 10/53
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0013 Acc: 0.9621

Epoch 11/53
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0013 Acc: 0.9636

Epoch 12/53
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0012 Acc: 0.9651

Epoch 13/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9666

Epoch 14/53
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0013 Acc: 0.9697

Epoch 15/53
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0011 Acc: 0.9681

Epoch 16/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9697

Epoch 17/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 18/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 19/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 20/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9666

Epoch 21/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 22/53
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9681

Epoch 23/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 24/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 25/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9697

Epoch 26/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9681

Epoch 27/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 28/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 29/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 30/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 31/53
----------
train Loss: 0.0000 Acc: 0.9987
val Loss: 0.0011 Acc: 0.9697

Epoch 32/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 33/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 34/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 35/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 36/53
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0012 Acc: 0.9681

Epoch 37/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 38/53
----------
train Loss: 0.0000 Acc: 0.9987
val Loss: 0.0011 Acc: 0.9681

Epoch 39/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9697

Epoch 40/53
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 41/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 42/53
----------
train Loss: 0.0000 Acc: 0.9987
val Loss: 0.0013 Acc: 0.9697

Epoch 43/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 44/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9681

Epoch 45/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 46/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 47/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9681

Epoch 48/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 49/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 50/53
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9697

Epoch 51/53
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9681

Epoch 52/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 53/53
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Best val Acc: 0.971168

---Testing---
Test accuracy: 0.989991
--------------------
Accuracy of Batoidea(ga_oo_lee) : 96 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9855658317981478, std: 0.013107916247339821

Model saved in "./weights/super_class_[1.0]_mean[0.99]_std[0.0].save".
--------------------

run info[val: 0.1, epoch: 99, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0059 Acc: 0.7377
val Loss: 0.0045 Acc: 0.9361

Epoch 1/98
----------
train Loss: 0.0021 Acc: 0.9161
val Loss: 0.0019 Acc: 0.9543

Epoch 2/98
----------
train Loss: 0.0018 Acc: 0.9328
val Loss: 0.0019 Acc: 0.9589

Epoch 3/98
----------
train Loss: 0.0016 Acc: 0.9394
val Loss: 0.0012 Acc: 0.9680

Epoch 4/98
----------
train Loss: 0.0017 Acc: 0.9449
val Loss: 0.0017 Acc: 0.9589

Epoch 5/98
----------
train Loss: 0.0015 Acc: 0.9409
val Loss: 0.0025 Acc: 0.9680

Epoch 6/98
----------
train Loss: 0.0014 Acc: 0.9490
val Loss: 0.0013 Acc: 0.9772

Epoch 7/98
----------
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0017 Acc: 0.9635

Epoch 8/98
----------
train Loss: 0.0013 Acc: 0.9520
val Loss: 0.0030 Acc: 0.9543

Epoch 9/98
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9576
val Loss: 0.0013 Acc: 0.9726

Epoch 10/98
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0014 Acc: 0.9635

Epoch 11/98
----------
train Loss: 0.0010 Acc: 0.9636
val Loss: 0.0012 Acc: 0.9726

Epoch 12/98
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0012 Acc: 0.9726

Epoch 13/98
----------
train Loss: 0.0011 Acc: 0.9596
val Loss: 0.0022 Acc: 0.9726

Epoch 14/98
----------
train Loss: 0.0011 Acc: 0.9576
val Loss: 0.0012 Acc: 0.9726

Epoch 15/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0014 Acc: 0.9680

Epoch 16/98
----------
train Loss: 0.0012 Acc: 0.9621
val Loss: 0.0013 Acc: 0.9726

Epoch 17/98
----------
train Loss: 0.0010 Acc: 0.9722
val Loss: 0.0015 Acc: 0.9726

Epoch 18/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0016 Acc: 0.9726

Epoch 19/98
----------
train Loss: 0.0011 Acc: 0.9586
val Loss: 0.0013 Acc: 0.9726

Epoch 20/98
----------
train Loss: 0.0012 Acc: 0.9560
val Loss: 0.0024 Acc: 0.9726

Epoch 21/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0023 Acc: 0.9726

Epoch 22/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0015 Acc: 0.9726

Epoch 23/98
----------
train Loss: 0.0010 Acc: 0.9666
val Loss: 0.0030 Acc: 0.9726

Epoch 24/98
----------
train Loss: 0.0010 Acc: 0.9666
val Loss: 0.0014 Acc: 0.9726

Epoch 25/98
----------
train Loss: 0.0010 Acc: 0.9606
val Loss: 0.0014 Acc: 0.9680

Epoch 26/98
----------
train Loss: 0.0012 Acc: 0.9535
val Loss: 0.0027 Acc: 0.9680

Epoch 27/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0013 Acc: 0.9680

Epoch 28/98
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0012 Acc: 0.9680

Epoch 29/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0012 Acc: 0.9726

Epoch 30/98
----------
train Loss: 0.0011 Acc: 0.9621
val Loss: 0.0014 Acc: 0.9680

Epoch 31/98
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0013 Acc: 0.9680

Epoch 32/98
----------
train Loss: 0.0010 Acc: 0.9722
val Loss: 0.0013 Acc: 0.9680

Epoch 33/98
----------
train Loss: 0.0011 Acc: 0.9565
val Loss: 0.0013 Acc: 0.9680

Epoch 34/98
----------
train Loss: 0.0010 Acc: 0.9687
val Loss: 0.0015 Acc: 0.9680

Epoch 35/98
----------
train Loss: 0.0011 Acc: 0.9621
val Loss: 0.0026 Acc: 0.9680

Epoch 36/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9626
val Loss: 0.0013 Acc: 0.9680

Epoch 37/98
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0012 Acc: 0.9726

Epoch 38/98
----------
train Loss: 0.0011 Acc: 0.9601
val Loss: 0.0016 Acc: 0.9726

Epoch 39/98
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0013 Acc: 0.9680

Epoch 40/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0012 Acc: 0.9680

Epoch 41/98
----------
train Loss: 0.0010 Acc: 0.9651
val Loss: 0.0013 Acc: 0.9680

Epoch 42/98
----------
train Loss: 0.0010 Acc: 0.9697
val Loss: 0.0016 Acc: 0.9680

Epoch 43/98
----------
train Loss: 0.0011 Acc: 0.9616
val Loss: 0.0013 Acc: 0.9726

Epoch 44/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0014 Acc: 0.9726

Epoch 45/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0012 Acc: 0.9680

Epoch 46/98
----------
train Loss: 0.0010 Acc: 0.9661
val Loss: 0.0013 Acc: 0.9680

Epoch 47/98
----------
train Loss: 0.0010 Acc: 0.9666
val Loss: 0.0013 Acc: 0.9726

Epoch 48/98
----------
train Loss: 0.0010 Acc: 0.9677
val Loss: 0.0012 Acc: 0.9726

Epoch 49/98
----------
train Loss: 0.0010 Acc: 0.9687
val Loss: 0.0015 Acc: 0.9680

Epoch 50/98
----------
train Loss: 0.0011 Acc: 0.9596
val Loss: 0.0012 Acc: 0.9680

Epoch 51/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0022 Acc: 0.9680

Epoch 52/98
----------
train Loss: 0.0012 Acc: 0.9641
val Loss: 0.0021 Acc: 0.9726

Epoch 53/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0016 Acc: 0.9726

Epoch 54/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0011 Acc: 0.9596
val Loss: 0.0014 Acc: 0.9680

Epoch 55/98
----------
train Loss: 0.0011 Acc: 0.9560
val Loss: 0.0014 Acc: 0.9680

Epoch 56/98
----------
train Loss: 0.0011 Acc: 0.9581
val Loss: 0.0019 Acc: 0.9680

Epoch 57/98
----------
train Loss: 0.0011 Acc: 0.9677
val Loss: 0.0015 Acc: 0.9680

Epoch 58/98
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0017 Acc: 0.9680

Epoch 59/98
----------
train Loss: 0.0011 Acc: 0.9601
val Loss: 0.0022 Acc: 0.9680

Epoch 60/98
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0012 Acc: 0.9726

Epoch 61/98
----------
train Loss: 0.0010 Acc: 0.9687
val Loss: 0.0012 Acc: 0.9726

Epoch 62/98
----------
train Loss: 0.0011 Acc: 0.9646
val Loss: 0.0012 Acc: 0.9726

Epoch 63/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0013 Acc: 0.9680

Epoch 64/98
----------
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0025 Acc: 0.9635

Epoch 65/98
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0012 Acc: 0.9680

Epoch 66/98
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0024 Acc: 0.9680

Epoch 67/98
----------
train Loss: 0.0012 Acc: 0.9616
val Loss: 0.0012 Acc: 0.9680

Epoch 68/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0024 Acc: 0.9726

Epoch 69/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0014 Acc: 0.9680

Epoch 70/98
----------
train Loss: 0.0011 Acc: 0.9606
val Loss: 0.0013 Acc: 0.9680

Epoch 71/98
----------
train Loss: 0.0010 Acc: 0.9717
val Loss: 0.0013 Acc: 0.9726

Epoch 72/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0017 Acc: 0.9726

Epoch 73/98
----------
train Loss: 0.0010 Acc: 0.9656
val Loss: 0.0025 Acc: 0.9726

Epoch 74/98
----------
train Loss: 0.0010 Acc: 0.9666
val Loss: 0.0012 Acc: 0.9726

Epoch 75/98
----------
train Loss: 0.0011 Acc: 0.9581
val Loss: 0.0013 Acc: 0.9726

Epoch 76/98
----------
train Loss: 0.0011 Acc: 0.9672
val Loss: 0.0013 Acc: 0.9680

Epoch 77/98
----------
train Loss: 0.0010 Acc: 0.9661
val Loss: 0.0012 Acc: 0.9726

Epoch 78/98
----------
train Loss: 0.0011 Acc: 0.9616
val Loss: 0.0013 Acc: 0.9680

Epoch 79/98
----------
train Loss: 0.0011 Acc: 0.9626
val Loss: 0.0013 Acc: 0.9680

Epoch 80/98
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9680

Epoch 81/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0013 Acc: 0.9680

Epoch 82/98
----------
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0012 Acc: 0.9680

Epoch 83/98
----------
train Loss: 0.0011 Acc: 0.9646
val Loss: 0.0012 Acc: 0.9726

Epoch 84/98
----------
train Loss: 0.0010 Acc: 0.9626
val Loss: 0.0016 Acc: 0.9680

Epoch 85/98
----------
train Loss: 0.0010 Acc: 0.9606
val Loss: 0.0013 Acc: 0.9726

Epoch 86/98
----------
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0014 Acc: 0.9680

Epoch 87/98
----------
train Loss: 0.0011 Acc: 0.9672
val Loss: 0.0014 Acc: 0.9680

Epoch 88/98
----------
train Loss: 0.0011 Acc: 0.9626
val Loss: 0.0012 Acc: 0.9726

Epoch 89/98
----------
train Loss: 0.0010 Acc: 0.9656
val Loss: 0.0014 Acc: 0.9726

Epoch 90/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0011 Acc: 0.9626
val Loss: 0.0016 Acc: 0.9680

Epoch 91/98
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0012 Acc: 0.9680

Epoch 92/98
----------
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0013 Acc: 0.9726

Epoch 93/98
----------
train Loss: 0.0011 Acc: 0.9621
val Loss: 0.0014 Acc: 0.9726

Epoch 94/98
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0026 Acc: 0.9726

Epoch 95/98
----------
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0012 Acc: 0.9726

Epoch 96/98
----------
train Loss: 0.0011 Acc: 0.9646
val Loss: 0.0013 Acc: 0.9726

Epoch 97/98
----------
train Loss: 0.0010 Acc: 0.9672
val Loss: 0.0013 Acc: 0.9680

Epoch 98/98
----------
train Loss: 0.0011 Acc: 0.9687
val Loss: 0.0013 Acc: 0.9680

Best val Acc: 0.977169

---Fine tuning.---
Epoch 0/98
----------
LR is set to 0.01
train Loss: 0.0012 Acc: 0.9555
val Loss: 0.0014 Acc: 0.9680

Epoch 1/98
----------
train Loss: 0.0005 Acc: 0.9838
val Loss: 0.0022 Acc: 0.9726

Epoch 2/98
----------
train Loss: 0.0003 Acc: 0.9924
val Loss: 0.0029 Acc: 0.9772

Epoch 3/98
----------
train Loss: 0.0002 Acc: 0.9955
val Loss: 0.0010 Acc: 0.9726

Epoch 4/98
----------
train Loss: 0.0002 Acc: 0.9944
val Loss: 0.0033 Acc: 0.9772

Epoch 5/98
----------
train Loss: 0.0002 Acc: 0.9939
val Loss: 0.0010 Acc: 0.9772

Epoch 6/98
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0008 Acc: 0.9726

Epoch 7/98
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0029 Acc: 0.9772

Epoch 8/98
----------
train Loss: 0.0001 Acc: 0.9970
val Loss: 0.0012 Acc: 0.9772

Epoch 9/98
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9772

Epoch 10/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0014 Acc: 0.9680

Epoch 11/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 12/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 13/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 14/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9680

Epoch 15/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0022 Acc: 0.9680

Epoch 16/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 17/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9726

Epoch 18/98
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0026 Acc: 0.9680

Epoch 19/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 20/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0030 Acc: 0.9680

Epoch 21/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9726

Epoch 22/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 23/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 24/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 25/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 26/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 27/98
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 28/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 29/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 30/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0025 Acc: 0.9680

Epoch 31/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 32/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 33/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0023 Acc: 0.9680

Epoch 34/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 35/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0017 Acc: 0.9680

Epoch 36/98
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 37/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9680

Epoch 38/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 39/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9680

Epoch 40/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9680

Epoch 41/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 42/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9680

Epoch 43/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9680

Epoch 44/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 45/98
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 46/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0024 Acc: 0.9680

Epoch 47/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 48/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0033 Acc: 0.9680

Epoch 49/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 50/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 51/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0020 Acc: 0.9726

Epoch 52/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0015 Acc: 0.9680

Epoch 53/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0014 Acc: 0.9680

Epoch 54/98
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 55/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 56/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 57/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 58/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 59/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9680

Epoch 60/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9726

Epoch 61/98
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0016 Acc: 0.9680

Epoch 62/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 63/98
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 64/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 65/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 66/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 67/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 68/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 69/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0031 Acc: 0.9680

Epoch 70/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 71/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 72/98
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 73/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 74/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 75/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 76/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 77/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9726

Epoch 78/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 79/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0030 Acc: 0.9680

Epoch 80/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9680

Epoch 81/98
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0013 Acc: 0.9680

Epoch 82/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 83/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 84/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 85/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 86/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 87/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0023 Acc: 0.9680

Epoch 88/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0033 Acc: 0.9680

Epoch 89/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0024 Acc: 0.9680

Epoch 90/98
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 91/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 92/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9726

Epoch 93/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 94/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 95/98
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0030 Acc: 0.9680

Epoch 96/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9680

Epoch 97/98
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 98/98
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Best val Acc: 0.977169

---Testing---
Test accuracy: 0.995450
--------------------
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9934614934163194, std: 0.005653731195798963
--------------------

run info[val: 0.2, epoch: 41, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/40
----------
LR is set to 0.01
train Loss: 0.0073 Acc: 0.7061
val Loss: 0.0041 Acc: 0.8724

Epoch 1/40
----------
train Loss: 0.0028 Acc: 0.8982
val Loss: 0.0026 Acc: 0.9317

Epoch 2/40
----------
train Loss: 0.0019 Acc: 0.9284
val Loss: 0.0020 Acc: 0.9408

Epoch 3/40
----------
LR is set to 0.001
train Loss: 0.0016 Acc: 0.9369
val Loss: 0.0020 Acc: 0.9385

Epoch 4/40
----------
train Loss: 0.0015 Acc: 0.9494
val Loss: 0.0019 Acc: 0.9408

Epoch 5/40
----------
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0020 Acc: 0.9362

Epoch 6/40
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9426
val Loss: 0.0021 Acc: 0.9339

Epoch 7/40
----------
train Loss: 0.0015 Acc: 0.9443
val Loss: 0.0020 Acc: 0.9408

Epoch 8/40
----------
train Loss: 0.0016 Acc: 0.9409
val Loss: 0.0020 Acc: 0.9385

Epoch 9/40
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0015 Acc: 0.9460
val Loss: 0.0020 Acc: 0.9408

Epoch 10/40
----------
train Loss: 0.0015 Acc: 0.9443
val Loss: 0.0018 Acc: 0.9408

Epoch 11/40
----------
train Loss: 0.0016 Acc: 0.9414
val Loss: 0.0018 Acc: 0.9362

Epoch 12/40
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0015 Acc: 0.9460
val Loss: 0.0018 Acc: 0.9362

Epoch 13/40
----------
train Loss: 0.0016 Acc: 0.9426
val Loss: 0.0021 Acc: 0.9385

Epoch 14/40
----------
train Loss: 0.0016 Acc: 0.9454
val Loss: 0.0020 Acc: 0.9408

Epoch 15/40
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0015 Acc: 0.9403
val Loss: 0.0021 Acc: 0.9385

Epoch 16/40
----------
train Loss: 0.0016 Acc: 0.9483
val Loss: 0.0019 Acc: 0.9385

Epoch 17/40
----------
train Loss: 0.0016 Acc: 0.9449
val Loss: 0.0019 Acc: 0.9385

Epoch 18/40
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0019 Acc: 0.9385

Epoch 19/40
----------
train Loss: 0.0015 Acc: 0.9471
val Loss: 0.0021 Acc: 0.9385

Epoch 20/40
----------
train Loss: 0.0016 Acc: 0.9431
val Loss: 0.0021 Acc: 0.9385

Epoch 21/40
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0018 Acc: 0.9362

Epoch 22/40
----------
train Loss: 0.0015 Acc: 0.9488
val Loss: 0.0021 Acc: 0.9385

Epoch 23/40
----------
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0019 Acc: 0.9362

Epoch 24/40
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0015 Acc: 0.9437
val Loss: 0.0020 Acc: 0.9408

Epoch 25/40
----------
train Loss: 0.0015 Acc: 0.9437
val Loss: 0.0018 Acc: 0.9408

Epoch 26/40
----------
train Loss: 0.0016 Acc: 0.9380
val Loss: 0.0018 Acc: 0.9408

Epoch 27/40
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0015 Acc: 0.9477
val Loss: 0.0020 Acc: 0.9362

Epoch 28/40
----------
train Loss: 0.0014 Acc: 0.9488
val Loss: 0.0021 Acc: 0.9408

Epoch 29/40
----------
train Loss: 0.0015 Acc: 0.9471
val Loss: 0.0018 Acc: 0.9385

Epoch 30/40
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0015 Acc: 0.9483
val Loss: 0.0019 Acc: 0.9362

Epoch 31/40
----------
train Loss: 0.0015 Acc: 0.9517
val Loss: 0.0022 Acc: 0.9385

Epoch 32/40
----------
train Loss: 0.0016 Acc: 0.9426
val Loss: 0.0019 Acc: 0.9385

Epoch 33/40
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0015 Acc: 0.9522
val Loss: 0.0019 Acc: 0.9362

Epoch 34/40
----------
train Loss: 0.0015 Acc: 0.9437
val Loss: 0.0021 Acc: 0.9408

Epoch 35/40
----------
train Loss: 0.0016 Acc: 0.9386
val Loss: 0.0019 Acc: 0.9408

Epoch 36/40
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0015 Acc: 0.9426
val Loss: 0.0020 Acc: 0.9408

Epoch 37/40
----------
train Loss: 0.0014 Acc: 0.9511
val Loss: 0.0019 Acc: 0.9385

Epoch 38/40
----------
train Loss: 0.0014 Acc: 0.9454
val Loss: 0.0018 Acc: 0.9385

Epoch 39/40
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0015 Acc: 0.9443
val Loss: 0.0021 Acc: 0.9385

Epoch 40/40
----------
train Loss: 0.0016 Acc: 0.9437
val Loss: 0.0018 Acc: 0.9385

Best val Acc: 0.940774

---Fine tuning.---
Epoch 0/40
----------
LR is set to 0.01
train Loss: 0.0014 Acc: 0.9466
val Loss: 0.0013 Acc: 0.9613

Epoch 1/40
----------
train Loss: 0.0006 Acc: 0.9795
val Loss: 0.0015 Acc: 0.9567

Epoch 2/40
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0014 Acc: 0.9544

Epoch 3/40
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9636

Epoch 4/40
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0012 Acc: 0.9681

Epoch 5/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9704

Epoch 6/40
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 7/40
----------
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0011 Acc: 0.9704

Epoch 8/40
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9704

Epoch 9/40
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0017 Acc: 0.9704

Epoch 10/40
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0015 Acc: 0.9727

Epoch 11/40
----------
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0013 Acc: 0.9727

Epoch 12/40
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0011 Acc: 0.9727

Epoch 13/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 14/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 15/40
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0012 Acc: 0.9727

Epoch 16/40
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 17/40
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 18/40
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0012 Acc: 0.9727

Epoch 19/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0015 Acc: 0.9704

Epoch 20/40
----------
train Loss: 0.0001 Acc: 0.9949
val Loss: 0.0013 Acc: 0.9704

Epoch 21/40
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0014 Acc: 0.9727

Epoch 22/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 23/40
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0011 Acc: 0.9704

Epoch 24/40
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0013 Acc: 0.9727

Epoch 25/40
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0011 Acc: 0.9704

Epoch 26/40
----------
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0012 Acc: 0.9727

Epoch 27/40
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0018 Acc: 0.9727

Epoch 28/40
----------
train Loss: 0.0002 Acc: 0.9955
val Loss: 0.0012 Acc: 0.9727

Epoch 29/40
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0011 Acc: 0.9704

Epoch 30/40
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0013 Acc: 0.9727

Epoch 31/40
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0017 Acc: 0.9727

Epoch 32/40
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0013 Acc: 0.9727

Epoch 33/40
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 34/40
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9704

Epoch 35/40
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9727

Epoch 36/40
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0011 Acc: 0.9727

Epoch 37/40
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0012 Acc: 0.9704

Epoch 38/40
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9704

Epoch 39/40
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9704

Epoch 40/40
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0011 Acc: 0.9727

Best val Acc: 0.972665

---Testing---
Test accuracy: 0.993176
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9890679114591175, std: 0.013454368724995057
--------------------

run info[val: 0.3, epoch: 87, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0059 Acc: 0.7719
val Loss: 0.0024 Acc: 0.9226

Epoch 1/86
----------
train Loss: 0.0022 Acc: 0.9220
val Loss: 0.0022 Acc: 0.9226

Epoch 2/86
----------
train Loss: 0.0016 Acc: 0.9396
val Loss: 0.0019 Acc: 0.9347

Epoch 3/86
----------
train Loss: 0.0015 Acc: 0.9506
val Loss: 0.0019 Acc: 0.9317

Epoch 4/86
----------
train Loss: 0.0014 Acc: 0.9519
val Loss: 0.0017 Acc: 0.9423

Epoch 5/86
----------
train Loss: 0.0013 Acc: 0.9636
val Loss: 0.0018 Acc: 0.9454

Epoch 6/86
----------
train Loss: 0.0012 Acc: 0.9649
val Loss: 0.0016 Acc: 0.9439

Epoch 7/86
----------
train Loss: 0.0014 Acc: 0.9526
val Loss: 0.0016 Acc: 0.9469

Epoch 8/86
----------
train Loss: 0.0012 Acc: 0.9623
val Loss: 0.0018 Acc: 0.9499

Epoch 9/86
----------
train Loss: 0.0011 Acc: 0.9675
val Loss: 0.0019 Acc: 0.9393

Epoch 10/86
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0015 Acc: 0.9439

Epoch 11/86
----------
train Loss: 0.0009 Acc: 0.9740
val Loss: 0.0016 Acc: 0.9469

Epoch 12/86
----------
train Loss: 0.0009 Acc: 0.9721
val Loss: 0.0017 Acc: 0.9439

Epoch 13/86
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9714
val Loss: 0.0015 Acc: 0.9514

Epoch 14/86
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9484

Epoch 15/86
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0016 Acc: 0.9439

Epoch 16/86
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0016 Acc: 0.9439

Epoch 17/86
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9423

Epoch 18/86
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0016 Acc: 0.9514

Epoch 19/86
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0015 Acc: 0.9469

Epoch 20/86
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9484

Epoch 21/86
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9439

Epoch 22/86
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9454

Epoch 23/86
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9484

Epoch 24/86
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9469

Epoch 25/86
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9454

Epoch 26/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9423

Epoch 27/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9469

Epoch 28/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9499

Epoch 29/86
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0015 Acc: 0.9499

Epoch 30/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9499

Epoch 31/86
----------
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9484

Epoch 32/86
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9454

Epoch 33/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9499

Epoch 34/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0014 Acc: 0.9439

Epoch 35/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0016 Acc: 0.9469

Epoch 36/86
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0015 Acc: 0.9454

Epoch 37/86
----------
train Loss: 0.0007 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9439

Epoch 38/86
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9469

Epoch 39/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9454

Epoch 40/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9530

Epoch 41/86
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9484

Epoch 42/86
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9454

Epoch 43/86
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0016 Acc: 0.9514

Epoch 44/86
----------
train Loss: 0.0008 Acc: 0.9831
val Loss: 0.0016 Acc: 0.9469

Epoch 45/86
----------
train Loss: 0.0007 Acc: 0.9799
val Loss: 0.0017 Acc: 0.9454

Epoch 46/86
----------
train Loss: 0.0007 Acc: 0.9805
val Loss: 0.0016 Acc: 0.9469

Epoch 47/86
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9484

Epoch 48/86
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9514

Epoch 49/86
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9499

Epoch 50/86
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0015 Acc: 0.9454

Epoch 51/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9439

Epoch 52/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9499

Epoch 53/86
----------
train Loss: 0.0007 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9469

Epoch 54/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9469

Epoch 55/86
----------
train Loss: 0.0007 Acc: 0.9857
val Loss: 0.0015 Acc: 0.9469

Epoch 56/86
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9469

Epoch 57/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9499

Epoch 58/86
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0014 Acc: 0.9484

Epoch 59/86
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9499

Epoch 60/86
----------
train Loss: 0.0008 Acc: 0.9844
val Loss: 0.0014 Acc: 0.9454

Epoch 61/86
----------
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9484

Epoch 62/86
----------
train Loss: 0.0007 Acc: 0.9844
val Loss: 0.0015 Acc: 0.9469

Epoch 63/86
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0014 Acc: 0.9439

Epoch 64/86
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9499

Epoch 65/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9469

Epoch 66/86
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9484

Epoch 67/86
----------
train Loss: 0.0007 Acc: 0.9844
val Loss: 0.0015 Acc: 0.9469

Epoch 68/86
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9454

Epoch 69/86
----------
train Loss: 0.0007 Acc: 0.9844
val Loss: 0.0015 Acc: 0.9454

Epoch 70/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9439

Epoch 71/86
----------
train Loss: 0.0008 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9484

Epoch 72/86
----------
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9454

Epoch 73/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9454

Epoch 74/86
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9484

Epoch 75/86
----------
train Loss: 0.0007 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9454

Epoch 76/86
----------
train Loss: 0.0007 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9454

Epoch 77/86
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9484

Epoch 78/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9469

Epoch 79/86
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9484

Epoch 80/86
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0016 Acc: 0.9469

Epoch 81/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0014 Acc: 0.9454

Epoch 82/86
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9439

Epoch 83/86
----------
train Loss: 0.0007 Acc: 0.9805
val Loss: 0.0016 Acc: 0.9454

Epoch 84/86
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9484

Epoch 85/86
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9499

Epoch 86/86
----------
train Loss: 0.0007 Acc: 0.9857
val Loss: 0.0015 Acc: 0.9469

Best val Acc: 0.952959

---Fine tuning.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0009 Acc: 0.9734
val Loss: 0.0014 Acc: 0.9651

Epoch 1/86
----------
train Loss: 0.0003 Acc: 0.9961
val Loss: 0.0013 Acc: 0.9651

Epoch 2/86
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0011 Acc: 0.9666

Epoch 3/86
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0012 Acc: 0.9636

Epoch 4/86
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0014 Acc: 0.9697

Epoch 5/86
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9681

Epoch 6/86
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9697

Epoch 7/86
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9681

Epoch 8/86
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9666

Epoch 9/86
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9712

Epoch 10/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9666

Epoch 11/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9651

Epoch 12/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9651

Epoch 13/86
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 14/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 15/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 16/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 17/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 18/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 19/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 20/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 21/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 22/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 23/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 24/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9712

Epoch 25/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 26/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 27/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 28/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 29/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 30/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 31/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 32/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 33/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 34/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 35/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 36/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 37/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 38/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 39/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 40/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 41/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 42/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 43/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 44/86
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9697

Epoch 45/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 46/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 47/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 48/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 49/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 50/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 51/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 52/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 53/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9666

Epoch 54/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 55/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 56/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 57/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 58/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 59/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 60/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 61/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 62/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 63/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 64/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 65/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 66/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 67/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 68/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 69/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 70/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 71/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 72/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 73/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9712

Epoch 74/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 75/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 76/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 77/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 78/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 79/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 80/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 81/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 82/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 83/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 84/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 85/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 86/86
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Best val Acc: 0.971168

---Testing---
Test accuracy: 0.991356
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9878646823728605, std: 0.01003886930852553

Model saved in "./weights/super_class_[1.0]_mean[0.99]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 60, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0051 Acc: 0.8024
val Loss: 0.0023 Acc: 0.9406

Epoch 1/59
----------
train Loss: 0.0021 Acc: 0.9293
val Loss: 0.0022 Acc: 0.9589

Epoch 2/59
----------
train Loss: 0.0019 Acc: 0.9358
val Loss: 0.0014 Acc: 0.9589

Epoch 3/59
----------
train Loss: 0.0016 Acc: 0.9414
val Loss: 0.0014 Acc: 0.9680

Epoch 4/59
----------
LR is set to 0.001
train Loss: 0.0014 Acc: 0.9515
val Loss: 0.0016 Acc: 0.9543

Epoch 5/59
----------
train Loss: 0.0013 Acc: 0.9525
val Loss: 0.0014 Acc: 0.9680

Epoch 6/59
----------
train Loss: 0.0015 Acc: 0.9439
val Loss: 0.0023 Acc: 0.9680

Epoch 7/59
----------
train Loss: 0.0013 Acc: 0.9520
val Loss: 0.0014 Acc: 0.9589

Epoch 8/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0014 Acc: 0.9525
val Loss: 0.0015 Acc: 0.9635

Epoch 9/59
----------
train Loss: 0.0013 Acc: 0.9500
val Loss: 0.0015 Acc: 0.9635

Epoch 10/59
----------
train Loss: 0.0014 Acc: 0.9480
val Loss: 0.0029 Acc: 0.9635

Epoch 11/59
----------
train Loss: 0.0014 Acc: 0.9510
val Loss: 0.0013 Acc: 0.9635

Epoch 12/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0014 Acc: 0.9505
val Loss: 0.0017 Acc: 0.9635

Epoch 13/59
----------
train Loss: 0.0013 Acc: 0.9555
val Loss: 0.0023 Acc: 0.9680

Epoch 14/59
----------
train Loss: 0.0014 Acc: 0.9505
val Loss: 0.0014 Acc: 0.9635

Epoch 15/59
----------
train Loss: 0.0014 Acc: 0.9480
val Loss: 0.0025 Acc: 0.9635

Epoch 16/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0014 Acc: 0.9439
val Loss: 0.0015 Acc: 0.9589

Epoch 17/59
----------
train Loss: 0.0013 Acc: 0.9515
val Loss: 0.0016 Acc: 0.9635

Epoch 18/59
----------
train Loss: 0.0014 Acc: 0.9510
val Loss: 0.0019 Acc: 0.9635

Epoch 19/59
----------
train Loss: 0.0014 Acc: 0.9500
val Loss: 0.0017 Acc: 0.9635

Epoch 20/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0014 Acc: 0.9510
val Loss: 0.0019 Acc: 0.9635

Epoch 21/59
----------
train Loss: 0.0014 Acc: 0.9515
val Loss: 0.0021 Acc: 0.9635

Epoch 22/59
----------
train Loss: 0.0014 Acc: 0.9520
val Loss: 0.0014 Acc: 0.9680

Epoch 23/59
----------
train Loss: 0.0014 Acc: 0.9530
val Loss: 0.0014 Acc: 0.9680

Epoch 24/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0019 Acc: 0.9680

Epoch 25/59
----------
train Loss: 0.0013 Acc: 0.9510
val Loss: 0.0017 Acc: 0.9680

Epoch 26/59
----------
train Loss: 0.0014 Acc: 0.9464
val Loss: 0.0023 Acc: 0.9635

Epoch 27/59
----------
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0013 Acc: 0.9635

Epoch 28/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0013 Acc: 0.9586
val Loss: 0.0013 Acc: 0.9635

Epoch 29/59
----------
train Loss: 0.0014 Acc: 0.9485
val Loss: 0.0014 Acc: 0.9635

Epoch 30/59
----------
train Loss: 0.0013 Acc: 0.9510
val Loss: 0.0015 Acc: 0.9680

Epoch 31/59
----------
train Loss: 0.0014 Acc: 0.9525
val Loss: 0.0016 Acc: 0.9635

Epoch 32/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0014 Acc: 0.9515
val Loss: 0.0022 Acc: 0.9589

Epoch 33/59
----------
train Loss: 0.0014 Acc: 0.9515
val Loss: 0.0017 Acc: 0.9680

Epoch 34/59
----------
train Loss: 0.0013 Acc: 0.9515
val Loss: 0.0014 Acc: 0.9635

Epoch 35/59
----------
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0015 Acc: 0.9635

Epoch 36/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0013 Acc: 0.9570
val Loss: 0.0015 Acc: 0.9635

Epoch 37/59
----------
train Loss: 0.0014 Acc: 0.9535
val Loss: 0.0014 Acc: 0.9635

Epoch 38/59
----------
train Loss: 0.0014 Acc: 0.9464
val Loss: 0.0015 Acc: 0.9635

Epoch 39/59
----------
train Loss: 0.0014 Acc: 0.9454
val Loss: 0.0024 Acc: 0.9635

Epoch 40/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0014 Acc: 0.9520
val Loss: 0.0014 Acc: 0.9635

Epoch 41/59
----------
train Loss: 0.0013 Acc: 0.9525
val Loss: 0.0022 Acc: 0.9680

Epoch 42/59
----------
train Loss: 0.0013 Acc: 0.9505
val Loss: 0.0015 Acc: 0.9635

Epoch 43/59
----------
train Loss: 0.0014 Acc: 0.9495
val Loss: 0.0014 Acc: 0.9635

Epoch 44/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0013 Acc: 0.9505
val Loss: 0.0015 Acc: 0.9635

Epoch 45/59
----------
train Loss: 0.0015 Acc: 0.9469
val Loss: 0.0025 Acc: 0.9680

Epoch 46/59
----------
train Loss: 0.0014 Acc: 0.9576
val Loss: 0.0016 Acc: 0.9635

Epoch 47/59
----------
train Loss: 0.0014 Acc: 0.9485
val Loss: 0.0015 Acc: 0.9635

Epoch 48/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0014 Acc: 0.9464
val Loss: 0.0017 Acc: 0.9635

Epoch 49/59
----------
train Loss: 0.0013 Acc: 0.9525
val Loss: 0.0015 Acc: 0.9635

Epoch 50/59
----------
train Loss: 0.0014 Acc: 0.9500
val Loss: 0.0014 Acc: 0.9635

Epoch 51/59
----------
train Loss: 0.0014 Acc: 0.9510
val Loss: 0.0015 Acc: 0.9680

Epoch 52/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0014 Acc: 0.9520
val Loss: 0.0013 Acc: 0.9635

Epoch 53/59
----------
train Loss: 0.0013 Acc: 0.9495
val Loss: 0.0013 Acc: 0.9635

Epoch 54/59
----------
train Loss: 0.0014 Acc: 0.9520
val Loss: 0.0023 Acc: 0.9680

Epoch 55/59
----------
train Loss: 0.0013 Acc: 0.9525
val Loss: 0.0013 Acc: 0.9635

Epoch 56/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0022 Acc: 0.9635

Epoch 57/59
----------
train Loss: 0.0013 Acc: 0.9500
val Loss: 0.0016 Acc: 0.9635

Epoch 58/59
----------
train Loss: 0.0013 Acc: 0.9606
val Loss: 0.0014 Acc: 0.9635

Epoch 59/59
----------
train Loss: 0.0014 Acc: 0.9464
val Loss: 0.0026 Acc: 0.9589

Best val Acc: 0.968037

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0014 Acc: 0.9464
val Loss: 0.0015 Acc: 0.9543

Epoch 1/59
----------
train Loss: 0.0007 Acc: 0.9742
val Loss: 0.0029 Acc: 0.9772

Epoch 2/59
----------
train Loss: 0.0004 Acc: 0.9874
val Loss: 0.0010 Acc: 0.9635

Epoch 3/59
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0019 Acc: 0.9772

Epoch 4/59
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0013 Acc: 0.9726

Epoch 5/59
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0033 Acc: 0.9726

Epoch 6/59
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0021 Acc: 0.9680

Epoch 7/59
----------
train Loss: 0.0001 Acc: 0.9970
val Loss: 0.0012 Acc: 0.9726

Epoch 8/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0036 Acc: 0.9726

Epoch 9/59
----------
train Loss: 0.0001 Acc: 0.9970
val Loss: 0.0013 Acc: 0.9726

Epoch 10/59
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0034 Acc: 0.9726

Epoch 11/59
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9726

Epoch 12/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0021 Acc: 0.9726

Epoch 13/59
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0013 Acc: 0.9726

Epoch 14/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9726

Epoch 15/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0018 Acc: 0.9726

Epoch 16/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0021 Acc: 0.9726

Epoch 17/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0015 Acc: 0.9726

Epoch 18/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0018 Acc: 0.9726

Epoch 19/59
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0012 Acc: 0.9726

Epoch 20/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 21/59
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9726

Epoch 22/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 23/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0021 Acc: 0.9680

Epoch 24/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0015 Acc: 0.9726

Epoch 25/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0017 Acc: 0.9726

Epoch 26/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0013 Acc: 0.9726

Epoch 27/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9726

Epoch 28/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9726

Epoch 29/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 30/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0021 Acc: 0.9726

Epoch 31/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9726

Epoch 32/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9726

Epoch 33/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0015 Acc: 0.9726

Epoch 34/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0018 Acc: 0.9726

Epoch 35/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0034 Acc: 0.9726

Epoch 36/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9726

Epoch 37/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 38/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 39/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0017 Acc: 0.9726

Epoch 40/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9726

Epoch 41/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9726

Epoch 42/59
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0016 Acc: 0.9726

Epoch 43/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 44/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0012 Acc: 0.9726

Epoch 45/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0023 Acc: 0.9726

Epoch 46/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0012 Acc: 0.9726

Epoch 47/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9726

Epoch 48/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 0.9970
val Loss: 0.0012 Acc: 0.9726

Epoch 49/59
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0015 Acc: 0.9726

Epoch 50/59
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0012 Acc: 0.9726

Epoch 51/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0016 Acc: 0.9726

Epoch 52/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 53/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0016 Acc: 0.9680

Epoch 54/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0016 Acc: 0.9726

Epoch 55/59
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0021 Acc: 0.9726

Epoch 56/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9680

Epoch 57/59
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 58/59
----------
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0033 Acc: 0.9726

Epoch 59/59
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0012 Acc: 0.9726

Best val Acc: 0.977169

---Testing---
Test accuracy: 0.983621
--------------------
Accuracy of Batoidea(ga_oo_lee) : 93 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.975890334541392, std: 0.028541368501967
--------------------

run info[val: 0.2, epoch: 72, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0054 Acc: 0.7663
val Loss: 0.0025 Acc: 0.9294

Epoch 1/71
----------
train Loss: 0.0021 Acc: 0.9261
val Loss: 0.0023 Acc: 0.9294

Epoch 2/71
----------
train Loss: 0.0018 Acc: 0.9335
val Loss: 0.0020 Acc: 0.9385

Epoch 3/71
----------
train Loss: 0.0017 Acc: 0.9437
val Loss: 0.0022 Acc: 0.9362

Epoch 4/71
----------
LR is set to 0.001
train Loss: 0.0015 Acc: 0.9466
val Loss: 0.0018 Acc: 0.9408

Epoch 5/71
----------
train Loss: 0.0015 Acc: 0.9534
val Loss: 0.0019 Acc: 0.9499

Epoch 6/71
----------
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0020 Acc: 0.9408

Epoch 7/71
----------
train Loss: 0.0012 Acc: 0.9602
val Loss: 0.0017 Acc: 0.9453

Epoch 8/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0014 Acc: 0.9494
val Loss: 0.0017 Acc: 0.9431

Epoch 9/71
----------
train Loss: 0.0014 Acc: 0.9540
val Loss: 0.0018 Acc: 0.9453

Epoch 10/71
----------
train Loss: 0.0014 Acc: 0.9557
val Loss: 0.0018 Acc: 0.9408

Epoch 11/71
----------
train Loss: 0.0013 Acc: 0.9562
val Loss: 0.0017 Acc: 0.9408

Epoch 12/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0012 Acc: 0.9562
val Loss: 0.0021 Acc: 0.9453

Epoch 13/71
----------
train Loss: 0.0013 Acc: 0.9574
val Loss: 0.0020 Acc: 0.9431

Epoch 14/71
----------
train Loss: 0.0014 Acc: 0.9534
val Loss: 0.0018 Acc: 0.9431

Epoch 15/71
----------
train Loss: 0.0013 Acc: 0.9551
val Loss: 0.0020 Acc: 0.9408

Epoch 16/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0014 Acc: 0.9540
val Loss: 0.0017 Acc: 0.9431

Epoch 17/71
----------
train Loss: 0.0013 Acc: 0.9540
val Loss: 0.0020 Acc: 0.9431

Epoch 18/71
----------
train Loss: 0.0013 Acc: 0.9540
val Loss: 0.0019 Acc: 0.9431

Epoch 19/71
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0019 Acc: 0.9408

Epoch 20/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0013 Acc: 0.9568
val Loss: 0.0022 Acc: 0.9431

Epoch 21/71
----------
train Loss: 0.0015 Acc: 0.9528
val Loss: 0.0018 Acc: 0.9476

Epoch 22/71
----------
train Loss: 0.0014 Acc: 0.9591
val Loss: 0.0020 Acc: 0.9431

Epoch 23/71
----------
train Loss: 0.0014 Acc: 0.9551
val Loss: 0.0017 Acc: 0.9408

Epoch 24/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9568
val Loss: 0.0017 Acc: 0.9431

Epoch 25/71
----------
train Loss: 0.0013 Acc: 0.9534
val Loss: 0.0018 Acc: 0.9453

Epoch 26/71
----------
train Loss: 0.0014 Acc: 0.9517
val Loss: 0.0018 Acc: 0.9431

Epoch 27/71
----------
train Loss: 0.0013 Acc: 0.9562
val Loss: 0.0018 Acc: 0.9431

Epoch 28/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0020 Acc: 0.9453

Epoch 29/71
----------
train Loss: 0.0014 Acc: 0.9596
val Loss: 0.0017 Acc: 0.9431

Epoch 30/71
----------
train Loss: 0.0014 Acc: 0.9591
val Loss: 0.0019 Acc: 0.9431

Epoch 31/71
----------
train Loss: 0.0014 Acc: 0.9505
val Loss: 0.0019 Acc: 0.9431

Epoch 32/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0014 Acc: 0.9517
val Loss: 0.0018 Acc: 0.9431

Epoch 33/71
----------
train Loss: 0.0014 Acc: 0.9551
val Loss: 0.0019 Acc: 0.9453

Epoch 34/71
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0017 Acc: 0.9408

Epoch 35/71
----------
train Loss: 0.0015 Acc: 0.9488
val Loss: 0.0017 Acc: 0.9453

Epoch 36/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0013 Acc: 0.9625
val Loss: 0.0018 Acc: 0.9431

Epoch 37/71
----------
train Loss: 0.0015 Acc: 0.9517
val Loss: 0.0019 Acc: 0.9431

Epoch 38/71
----------
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0018 Acc: 0.9453

Epoch 39/71
----------
train Loss: 0.0013 Acc: 0.9562
val Loss: 0.0020 Acc: 0.9431

Epoch 40/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0015 Acc: 0.9545
val Loss: 0.0017 Acc: 0.9431

Epoch 41/71
----------
train Loss: 0.0014 Acc: 0.9545
val Loss: 0.0019 Acc: 0.9408

Epoch 42/71
----------
train Loss: 0.0013 Acc: 0.9602
val Loss: 0.0017 Acc: 0.9408

Epoch 43/71
----------
train Loss: 0.0014 Acc: 0.9522
val Loss: 0.0018 Acc: 0.9408

Epoch 44/71
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0013 Acc: 0.9574
val Loss: 0.0018 Acc: 0.9408

Epoch 45/71
----------
train Loss: 0.0013 Acc: 0.9602
val Loss: 0.0018 Acc: 0.9431

Epoch 46/71
----------
train Loss: 0.0014 Acc: 0.9551
val Loss: 0.0018 Acc: 0.9431

Epoch 47/71
----------
train Loss: 0.0013 Acc: 0.9585
val Loss: 0.0018 Acc: 0.9408

Epoch 48/71
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0013 Acc: 0.9557
val Loss: 0.0018 Acc: 0.9431

Epoch 49/71
----------
train Loss: 0.0013 Acc: 0.9534
val Loss: 0.0017 Acc: 0.9408

Epoch 50/71
----------
train Loss: 0.0014 Acc: 0.9579
val Loss: 0.0018 Acc: 0.9408

Epoch 51/71
----------
train Loss: 0.0014 Acc: 0.9522
val Loss: 0.0019 Acc: 0.9431

Epoch 52/71
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0013 Acc: 0.9579
val Loss: 0.0018 Acc: 0.9408

Epoch 53/71
----------
train Loss: 0.0013 Acc: 0.9579
val Loss: 0.0020 Acc: 0.9431

Epoch 54/71
----------
train Loss: 0.0015 Acc: 0.9511
val Loss: 0.0017 Acc: 0.9453

Epoch 55/71
----------
train Loss: 0.0013 Acc: 0.9557
val Loss: 0.0018 Acc: 0.9408

Epoch 56/71
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0014 Acc: 0.9488
val Loss: 0.0019 Acc: 0.9453

Epoch 57/71
----------
train Loss: 0.0014 Acc: 0.9579
val Loss: 0.0021 Acc: 0.9431

Epoch 58/71
----------
train Loss: 0.0014 Acc: 0.9511
val Loss: 0.0020 Acc: 0.9408

Epoch 59/71
----------
train Loss: 0.0014 Acc: 0.9562
val Loss: 0.0020 Acc: 0.9431

Epoch 60/71
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0014 Acc: 0.9574
val Loss: 0.0018 Acc: 0.9408

Epoch 61/71
----------
train Loss: 0.0014 Acc: 0.9466
val Loss: 0.0019 Acc: 0.9431

Epoch 62/71
----------
train Loss: 0.0013 Acc: 0.9551
val Loss: 0.0021 Acc: 0.9431

Epoch 63/71
----------
train Loss: 0.0015 Acc: 0.9466
val Loss: 0.0018 Acc: 0.9431

Epoch 64/71
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0013 Acc: 0.9568
val Loss: 0.0018 Acc: 0.9431

Epoch 65/71
----------
train Loss: 0.0014 Acc: 0.9511
val Loss: 0.0017 Acc: 0.9431

Epoch 66/71
----------
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0018 Acc: 0.9431

Epoch 67/71
----------
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0022 Acc: 0.9431

Epoch 68/71
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0013 Acc: 0.9568
val Loss: 0.0018 Acc: 0.9431

Epoch 69/71
----------
train Loss: 0.0014 Acc: 0.9574
val Loss: 0.0021 Acc: 0.9431

Epoch 70/71
----------
train Loss: 0.0014 Acc: 0.9562
val Loss: 0.0018 Acc: 0.9453

Epoch 71/71
----------
train Loss: 0.0013 Acc: 0.9540
val Loss: 0.0018 Acc: 0.9431

Best val Acc: 0.949886

---Fine tuning.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0019 Acc: 0.9499

Epoch 1/71
----------
train Loss: 0.0008 Acc: 0.9710
val Loss: 0.0023 Acc: 0.9590

Epoch 2/71
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0008 Acc: 0.9727

Epoch 3/71
----------
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0011 Acc: 0.9749

Epoch 4/71
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0010 Acc: 0.9727

Epoch 5/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0013 Acc: 0.9727

Epoch 6/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9749

Epoch 7/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0013 Acc: 0.9749

Epoch 8/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0010 Acc: 0.9749

Epoch 9/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0011 Acc: 0.9727

Epoch 10/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9727

Epoch 11/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0010 Acc: 0.9749

Epoch 12/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9727

Epoch 13/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9749

Epoch 14/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9727

Epoch 15/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0013 Acc: 0.9727

Epoch 16/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9749

Epoch 17/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 18/71
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9749

Epoch 19/71
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0012 Acc: 0.9749

Epoch 20/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0010 Acc: 0.9749

Epoch 21/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9749

Epoch 22/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0013 Acc: 0.9749

Epoch 23/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9727

Epoch 24/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0010 Acc: 0.9749

Epoch 25/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9749

Epoch 26/71
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9727

Epoch 27/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 28/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9727

Epoch 29/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9749

Epoch 30/71
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0015 Acc: 0.9749

Epoch 31/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0010 Acc: 0.9749

Epoch 32/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 33/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0011 Acc: 0.9749

Epoch 34/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0011 Acc: 0.9749

Epoch 35/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9727

Epoch 36/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 37/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 38/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0017 Acc: 0.9749

Epoch 39/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9749

Epoch 40/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9727

Epoch 41/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9749

Epoch 42/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0015 Acc: 0.9749

Epoch 43/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9749

Epoch 44/71
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0015 Acc: 0.9749

Epoch 45/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0010 Acc: 0.9749

Epoch 46/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9727

Epoch 47/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9749

Epoch 48/71
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9749

Epoch 49/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9727

Epoch 50/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0012 Acc: 0.9727

Epoch 51/71
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 52/71
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0014 Acc: 0.9727

Epoch 53/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9749

Epoch 54/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0011 Acc: 0.9749

Epoch 55/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9727

Epoch 56/71
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 57/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9749

Epoch 58/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9749

Epoch 59/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9749

Epoch 60/71
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 61/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9749

Epoch 62/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9727

Epoch 63/71
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0014 Acc: 0.9749

Epoch 64/71
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0013 Acc: 0.9749

Epoch 65/71
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9749

Epoch 66/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9749

Epoch 67/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9749

Epoch 68/71
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9749

Epoch 69/71
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9749

Epoch 70/71
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9749

Epoch 71/71
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9727

Best val Acc: 0.974943

---Testing---
Test accuracy: 0.993631
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9902906891656794, std: 0.011045015421004362
--------------------

run info[val: 0.3, epoch: 93, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0060 Acc: 0.7492
val Loss: 0.0029 Acc: 0.9014

Epoch 1/92
----------
train Loss: 0.0021 Acc: 0.9272
val Loss: 0.0022 Acc: 0.9211

Epoch 2/92
----------
train Loss: 0.0016 Acc: 0.9448
val Loss: 0.0021 Acc: 0.9332

Epoch 3/92
----------
train Loss: 0.0014 Acc: 0.9526
val Loss: 0.0020 Acc: 0.9363

Epoch 4/92
----------
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0016 Acc: 0.9408

Epoch 5/92
----------
train Loss: 0.0011 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9484

Epoch 6/92
----------
train Loss: 0.0011 Acc: 0.9649
val Loss: 0.0018 Acc: 0.9454

Epoch 7/92
----------
train Loss: 0.0011 Acc: 0.9656
val Loss: 0.0017 Acc: 0.9484

Epoch 8/92
----------
train Loss: 0.0010 Acc: 0.9675
val Loss: 0.0016 Acc: 0.9530

Epoch 9/92
----------
train Loss: 0.0009 Acc: 0.9721
val Loss: 0.0016 Acc: 0.9530

Epoch 10/92
----------
train Loss: 0.0009 Acc: 0.9740
val Loss: 0.0016 Acc: 0.9499

Epoch 11/92
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9499

Epoch 12/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9545

Epoch 13/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9514

Epoch 14/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 15/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9514

Epoch 16/92
----------
train Loss: 0.0009 Acc: 0.9747
val Loss: 0.0015 Acc: 0.9530

Epoch 17/92
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9545

Epoch 18/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9530

Epoch 19/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9530

Epoch 20/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9545

Epoch 21/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9499

Epoch 22/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0016 Acc: 0.9499

Epoch 23/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9484

Epoch 24/92
----------
train Loss: 0.0009 Acc: 0.9734
val Loss: 0.0015 Acc: 0.9514

Epoch 25/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9514

Epoch 26/92
----------
train Loss: 0.0008 Acc: 0.9844
val Loss: 0.0015 Acc: 0.9530

Epoch 27/92
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9545

Epoch 28/92
----------
train Loss: 0.0009 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9499

Epoch 29/92
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9530

Epoch 30/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9514

Epoch 31/92
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9575

Epoch 32/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 33/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0016 Acc: 0.9530

Epoch 34/92
----------
train Loss: 0.0007 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9530

Epoch 35/92
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9530

Epoch 36/92
----------
train Loss: 0.0009 Acc: 0.9779
val Loss: 0.0016 Acc: 0.9514

Epoch 37/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9560

Epoch 38/92
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9530

Epoch 39/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9560

Epoch 40/92
----------
train Loss: 0.0008 Acc: 0.9766
val Loss: 0.0015 Acc: 0.9530

Epoch 41/92
----------
train Loss: 0.0009 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9514

Epoch 42/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9545

Epoch 43/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 44/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9530

Epoch 45/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 46/92
----------
train Loss: 0.0008 Acc: 0.9734
val Loss: 0.0015 Acc: 0.9530

Epoch 47/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9530

Epoch 48/92
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9575

Epoch 49/92
----------
train Loss: 0.0007 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9545

Epoch 50/92
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9545

Epoch 51/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 52/92
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0016 Acc: 0.9560

Epoch 53/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0016 Acc: 0.9545

Epoch 54/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9530

Epoch 55/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9545

Epoch 56/92
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9514

Epoch 57/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9545

Epoch 58/92
----------
train Loss: 0.0008 Acc: 0.9766
val Loss: 0.0015 Acc: 0.9530

Epoch 59/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9530

Epoch 60/92
----------
train Loss: 0.0008 Acc: 0.9747
val Loss: 0.0015 Acc: 0.9530

Epoch 61/92
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0016 Acc: 0.9530

Epoch 62/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9514

Epoch 63/92
----------
train Loss: 0.0008 Acc: 0.9766
val Loss: 0.0015 Acc: 0.9530

Epoch 64/92
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9530

Epoch 65/92
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9530

Epoch 66/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9545

Epoch 67/92
----------
train Loss: 0.0008 Acc: 0.9734
val Loss: 0.0016 Acc: 0.9530

Epoch 68/92
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9560

Epoch 69/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9530

Epoch 70/92
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9545

Epoch 71/92
----------
train Loss: 0.0008 Acc: 0.9747
val Loss: 0.0015 Acc: 0.9530

Epoch 72/92
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9560

Epoch 73/92
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0016 Acc: 0.9545

Epoch 74/92
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9545

Epoch 75/92
----------
train Loss: 0.0009 Acc: 0.9740
val Loss: 0.0015 Acc: 0.9560

Epoch 76/92
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0016 Acc: 0.9514

Epoch 77/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9530

Epoch 78/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9545

Epoch 79/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9560

Epoch 80/92
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0016 Acc: 0.9530

Epoch 81/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9560

Epoch 82/92
----------
train Loss: 0.0008 Acc: 0.9747
val Loss: 0.0015 Acc: 0.9560

Epoch 83/92
----------
train Loss: 0.0008 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9575

Epoch 84/92
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0015 Acc: 0.9530

Epoch 85/92
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9560

Epoch 86/92
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0015 Acc: 0.9530

Epoch 87/92
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0016 Acc: 0.9530

Epoch 88/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9530

Epoch 89/92
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0015 Acc: 0.9560

Epoch 90/92
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9545

Epoch 91/92
----------
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0016 Acc: 0.9545

Epoch 92/92
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9575

Best val Acc: 0.957511

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0009 Acc: 0.9708
val Loss: 0.0017 Acc: 0.9560

Epoch 1/92
----------
train Loss: 0.0004 Acc: 0.9890
val Loss: 0.0020 Acc: 0.9454

Epoch 2/92
----------
train Loss: 0.0002 Acc: 0.9955
val Loss: 0.0013 Acc: 0.9666

Epoch 3/92
----------
train Loss: 0.0002 Acc: 0.9981
val Loss: 0.0012 Acc: 0.9697

Epoch 4/92
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0014 Acc: 0.9636

Epoch 5/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9666

Epoch 6/92
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0012 Acc: 0.9666

Epoch 7/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9651

Epoch 8/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 9/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 10/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 11/92
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 12/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 13/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 14/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9666

Epoch 15/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 16/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 17/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 18/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 19/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 20/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 21/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 22/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 23/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 24/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 25/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 26/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 27/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 28/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 29/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 30/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 31/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 32/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 33/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 34/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 35/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 36/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 37/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 38/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 39/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 40/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 41/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 42/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 43/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 44/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 45/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 46/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 47/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 48/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 49/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 50/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 51/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 52/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 53/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 54/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 55/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 56/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 57/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 58/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 59/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9712

Epoch 60/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 61/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 62/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 63/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 64/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9681

Epoch 65/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 66/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 67/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 68/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 69/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 70/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9681

Epoch 71/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 72/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 73/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 74/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 75/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 76/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 77/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 78/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 79/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 80/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 81/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 82/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 83/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 84/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 85/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 86/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 87/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9697

Epoch 88/92
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 89/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 90/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 91/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 92/92
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9681

Best val Acc: 0.971168

---Testing---
Test accuracy: 0.991356
--------------------
Accuracy of Batoidea(ga_oo_lee) : 96 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9862838595258411, std: 0.01640689179896924

Model saved in "./weights/super_class_[0.99]_mean[0.99]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 61, randcrop: True, decay: 12]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0055 Acc: 0.7605
val Loss: 0.0025 Acc: 0.9315

Epoch 1/60
----------
train Loss: 0.0021 Acc: 0.9146
val Loss: 0.0019 Acc: 0.9543

Epoch 2/60
----------
train Loss: 0.0020 Acc: 0.9298
val Loss: 0.0020 Acc: 0.9543

Epoch 3/60
----------
train Loss: 0.0018 Acc: 0.9348
val Loss: 0.0031 Acc: 0.9589

Epoch 4/60
----------
train Loss: 0.0020 Acc: 0.9262
val Loss: 0.0018 Acc: 0.9543

Epoch 5/60
----------
train Loss: 0.0014 Acc: 0.9495
val Loss: 0.0017 Acc: 0.9543

Epoch 6/60
----------
train Loss: 0.0017 Acc: 0.9298
val Loss: 0.0038 Acc: 0.9543

Epoch 7/60
----------
train Loss: 0.0014 Acc: 0.9510
val Loss: 0.0016 Acc: 0.9589

Epoch 8/60
----------
train Loss: 0.0015 Acc: 0.9409
val Loss: 0.0016 Acc: 0.9543

Epoch 9/60
----------
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0013 Acc: 0.9635

Epoch 10/60
----------
train Loss: 0.0011 Acc: 0.9616
val Loss: 0.0015 Acc: 0.9680

Epoch 11/60
----------
train Loss: 0.0013 Acc: 0.9510
val Loss: 0.0017 Acc: 0.9680

Epoch 12/60
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9596
val Loss: 0.0020 Acc: 0.9543

Epoch 13/60
----------
train Loss: 0.0010 Acc: 0.9631
val Loss: 0.0016 Acc: 0.9635

Epoch 14/60
----------
train Loss: 0.0010 Acc: 0.9656
val Loss: 0.0015 Acc: 0.9635

Epoch 15/60
----------
train Loss: 0.0009 Acc: 0.9717
val Loss: 0.0025 Acc: 0.9635

Epoch 16/60
----------
train Loss: 0.0010 Acc: 0.9677
val Loss: 0.0014 Acc: 0.9635

Epoch 17/60
----------
train Loss: 0.0010 Acc: 0.9651
val Loss: 0.0016 Acc: 0.9589

Epoch 18/60
----------
train Loss: 0.0010 Acc: 0.9672
val Loss: 0.0015 Acc: 0.9589

Epoch 19/60
----------
train Loss: 0.0010 Acc: 0.9626
val Loss: 0.0015 Acc: 0.9589

Epoch 20/60
----------
train Loss: 0.0010 Acc: 0.9621
val Loss: 0.0016 Acc: 0.9589

Epoch 21/60
----------
train Loss: 0.0009 Acc: 0.9682
val Loss: 0.0014 Acc: 0.9635

Epoch 22/60
----------
train Loss: 0.0009 Acc: 0.9666
val Loss: 0.0020 Acc: 0.9589

Epoch 23/60
----------
train Loss: 0.0009 Acc: 0.9672
val Loss: 0.0014 Acc: 0.9635

Epoch 24/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9677
val Loss: 0.0021 Acc: 0.9635

Epoch 25/60
----------
train Loss: 0.0011 Acc: 0.9591
val Loss: 0.0030 Acc: 0.9635

Epoch 26/60
----------
train Loss: 0.0010 Acc: 0.9687
val Loss: 0.0013 Acc: 0.9635

Epoch 27/60
----------
train Loss: 0.0010 Acc: 0.9636
val Loss: 0.0013 Acc: 0.9635

Epoch 28/60
----------
train Loss: 0.0010 Acc: 0.9601
val Loss: 0.0013 Acc: 0.9589

Epoch 29/60
----------
train Loss: 0.0009 Acc: 0.9697
val Loss: 0.0014 Acc: 0.9589

Epoch 30/60
----------
train Loss: 0.0009 Acc: 0.9712
val Loss: 0.0026 Acc: 0.9635

Epoch 31/60
----------
train Loss: 0.0009 Acc: 0.9677
val Loss: 0.0015 Acc: 0.9635

Epoch 32/60
----------
train Loss: 0.0009 Acc: 0.9661
val Loss: 0.0016 Acc: 0.9635

Epoch 33/60
----------
train Loss: 0.0009 Acc: 0.9732
val Loss: 0.0014 Acc: 0.9635

Epoch 34/60
----------
train Loss: 0.0009 Acc: 0.9712
val Loss: 0.0017 Acc: 0.9635

Epoch 35/60
----------
train Loss: 0.0010 Acc: 0.9616
val Loss: 0.0015 Acc: 0.9635

Epoch 36/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0010 Acc: 0.9656
val Loss: 0.0013 Acc: 0.9589

Epoch 37/60
----------
train Loss: 0.0009 Acc: 0.9661
val Loss: 0.0016 Acc: 0.9589

Epoch 38/60
----------
train Loss: 0.0009 Acc: 0.9687
val Loss: 0.0013 Acc: 0.9589

Epoch 39/60
----------
train Loss: 0.0009 Acc: 0.9702
val Loss: 0.0016 Acc: 0.9635

Epoch 40/60
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0016 Acc: 0.9635

Epoch 41/60
----------
train Loss: 0.0010 Acc: 0.9697
val Loss: 0.0015 Acc: 0.9635

Epoch 42/60
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0026 Acc: 0.9635

Epoch 43/60
----------
train Loss: 0.0010 Acc: 0.9697
val Loss: 0.0014 Acc: 0.9589

Epoch 44/60
----------
train Loss: 0.0010 Acc: 0.9631
val Loss: 0.0014 Acc: 0.9635

Epoch 45/60
----------
train Loss: 0.0010 Acc: 0.9661
val Loss: 0.0016 Acc: 0.9635

Epoch 46/60
----------
train Loss: 0.0010 Acc: 0.9641
val Loss: 0.0019 Acc: 0.9589

Epoch 47/60
----------
train Loss: 0.0009 Acc: 0.9697
val Loss: 0.0014 Acc: 0.9589

Epoch 48/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0010 Acc: 0.9646
val Loss: 0.0017 Acc: 0.9635

Epoch 49/60
----------
train Loss: 0.0009 Acc: 0.9702
val Loss: 0.0024 Acc: 0.9589

Epoch 50/60
----------
train Loss: 0.0010 Acc: 0.9621
val Loss: 0.0019 Acc: 0.9635

Epoch 51/60
----------
train Loss: 0.0009 Acc: 0.9666
val Loss: 0.0017 Acc: 0.9635

Epoch 52/60
----------
train Loss: 0.0009 Acc: 0.9687
val Loss: 0.0016 Acc: 0.9635

Epoch 53/60
----------
train Loss: 0.0009 Acc: 0.9692
val Loss: 0.0018 Acc: 0.9589

Epoch 54/60
----------
train Loss: 0.0010 Acc: 0.9661
val Loss: 0.0014 Acc: 0.9589

Epoch 55/60
----------
train Loss: 0.0010 Acc: 0.9666
val Loss: 0.0026 Acc: 0.9589

Epoch 56/60
----------
train Loss: 0.0009 Acc: 0.9687
val Loss: 0.0016 Acc: 0.9589

Epoch 57/60
----------
train Loss: 0.0010 Acc: 0.9692
val Loss: 0.0013 Acc: 0.9589

Epoch 58/60
----------
train Loss: 0.0010 Acc: 0.9712
val Loss: 0.0013 Acc: 0.9589

Epoch 59/60
----------
train Loss: 0.0010 Acc: 0.9692
val Loss: 0.0014 Acc: 0.9589

Epoch 60/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 0.9677
val Loss: 0.0013 Acc: 0.9635

Best val Acc: 0.968037

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0011 Acc: 0.9586
val Loss: 0.0041 Acc: 0.9635

Epoch 1/60
----------
train Loss: 0.0005 Acc: 0.9833
val Loss: 0.0012 Acc: 0.9589

Epoch 2/60
----------
train Loss: 0.0004 Acc: 0.9833
val Loss: 0.0024 Acc: 0.9543

Epoch 3/60
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0012 Acc: 0.9772

Epoch 4/60
----------
train Loss: 0.0002 Acc: 0.9949
val Loss: 0.0009 Acc: 0.9772

Epoch 5/60
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0011 Acc: 0.9680

Epoch 6/60
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0012 Acc: 0.9680

Epoch 7/60
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9772

Epoch 8/60
----------
train Loss: 0.0001 Acc: 0.9970
val Loss: 0.0029 Acc: 0.9726

Epoch 9/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 10/60
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0013 Acc: 0.9680

Epoch 11/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9726

Epoch 12/60
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9726

Epoch 13/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 14/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0031 Acc: 0.9726

Epoch 15/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 16/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9680

Epoch 17/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9680

Epoch 18/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 19/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 20/60
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0014 Acc: 0.9680

Epoch 21/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 22/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9680

Epoch 23/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0032 Acc: 0.9680

Epoch 24/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0030 Acc: 0.9680

Epoch 25/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 26/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 27/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 28/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 29/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0019 Acc: 0.9680

Epoch 30/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 31/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0024 Acc: 0.9680

Epoch 32/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 33/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0031 Acc: 0.9680

Epoch 34/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0038 Acc: 0.9680

Epoch 35/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0030 Acc: 0.9680

Epoch 36/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 37/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0036 Acc: 0.9680

Epoch 38/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0035 Acc: 0.9680

Epoch 39/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 40/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 41/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 42/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 43/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 44/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 45/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 46/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 47/60
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0014 Acc: 0.9680

Epoch 48/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9680

Epoch 49/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0019 Acc: 0.9680

Epoch 50/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9680

Epoch 51/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 52/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 53/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 54/60
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 55/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 56/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0041 Acc: 0.9680

Epoch 57/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 58/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 59/60
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 60/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0019 Acc: 0.9680

Best val Acc: 0.977169

---Testing---
Test accuracy: 0.992721
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9901276046608439, std: 0.009280329720144757
--------------------

run info[val: 0.2, epoch: 69, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0056 Acc: 0.7737
val Loss: 0.0025 Acc: 0.9226

Epoch 1/68
----------
train Loss: 0.0020 Acc: 0.9221
val Loss: 0.0026 Acc: 0.9271

Epoch 2/68
----------
train Loss: 0.0017 Acc: 0.9397
val Loss: 0.0021 Acc: 0.9431

Epoch 3/68
----------
train Loss: 0.0016 Acc: 0.9420
val Loss: 0.0023 Acc: 0.9385

Epoch 4/68
----------
train Loss: 0.0015 Acc: 0.9454
val Loss: 0.0018 Acc: 0.9385

Epoch 5/68
----------
train Loss: 0.0014 Acc: 0.9460
val Loss: 0.0018 Acc: 0.9453

Epoch 6/68
----------
train Loss: 0.0014 Acc: 0.9528
val Loss: 0.0018 Acc: 0.9499

Epoch 7/68
----------
train Loss: 0.0013 Acc: 0.9574
val Loss: 0.0018 Acc: 0.9408

Epoch 8/68
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9557
val Loss: 0.0018 Acc: 0.9476

Epoch 9/68
----------
train Loss: 0.0011 Acc: 0.9659
val Loss: 0.0016 Acc: 0.9431

Epoch 10/68
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0017 Acc: 0.9476

Epoch 11/68
----------
train Loss: 0.0011 Acc: 0.9642
val Loss: 0.0018 Acc: 0.9453

Epoch 12/68
----------
train Loss: 0.0011 Acc: 0.9619
val Loss: 0.0016 Acc: 0.9453

Epoch 13/68
----------
train Loss: 0.0011 Acc: 0.9642
val Loss: 0.0019 Acc: 0.9476

Epoch 14/68
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0018 Acc: 0.9476

Epoch 15/68
----------
train Loss: 0.0012 Acc: 0.9568
val Loss: 0.0016 Acc: 0.9476

Epoch 16/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9704
val Loss: 0.0018 Acc: 0.9476

Epoch 17/68
----------
train Loss: 0.0011 Acc: 0.9665
val Loss: 0.0018 Acc: 0.9476

Epoch 18/68
----------
train Loss: 0.0011 Acc: 0.9619
val Loss: 0.0019 Acc: 0.9453

Epoch 19/68
----------
train Loss: 0.0011 Acc: 0.9642
val Loss: 0.0017 Acc: 0.9476

Epoch 20/68
----------
train Loss: 0.0010 Acc: 0.9653
val Loss: 0.0019 Acc: 0.9453

Epoch 21/68
----------
train Loss: 0.0010 Acc: 0.9733
val Loss: 0.0016 Acc: 0.9499

Epoch 22/68
----------
train Loss: 0.0011 Acc: 0.9653
val Loss: 0.0020 Acc: 0.9476

Epoch 23/68
----------
train Loss: 0.0011 Acc: 0.9659
val Loss: 0.0017 Acc: 0.9453

Epoch 24/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0011 Acc: 0.9613
val Loss: 0.0019 Acc: 0.9476

Epoch 25/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0016 Acc: 0.9453

Epoch 26/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0016 Acc: 0.9476

Epoch 27/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0016 Acc: 0.9453

Epoch 28/68
----------
train Loss: 0.0011 Acc: 0.9619
val Loss: 0.0018 Acc: 0.9522

Epoch 29/68
----------
train Loss: 0.0010 Acc: 0.9665
val Loss: 0.0020 Acc: 0.9476

Epoch 30/68
----------
train Loss: 0.0011 Acc: 0.9625
val Loss: 0.0017 Acc: 0.9476

Epoch 31/68
----------
train Loss: 0.0011 Acc: 0.9676
val Loss: 0.0018 Acc: 0.9499

Epoch 32/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0018 Acc: 0.9476

Epoch 33/68
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0017 Acc: 0.9453

Epoch 34/68
----------
train Loss: 0.0010 Acc: 0.9642
val Loss: 0.0016 Acc: 0.9476

Epoch 35/68
----------
train Loss: 0.0011 Acc: 0.9670
val Loss: 0.0016 Acc: 0.9453

Epoch 36/68
----------
train Loss: 0.0011 Acc: 0.9676
val Loss: 0.0016 Acc: 0.9453

Epoch 37/68
----------
train Loss: 0.0009 Acc: 0.9721
val Loss: 0.0018 Acc: 0.9453

Epoch 38/68
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0018 Acc: 0.9476

Epoch 39/68
----------
train Loss: 0.0012 Acc: 0.9568
val Loss: 0.0017 Acc: 0.9476

Epoch 40/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0011 Acc: 0.9613
val Loss: 0.0016 Acc: 0.9453

Epoch 41/68
----------
train Loss: 0.0012 Acc: 0.9636
val Loss: 0.0019 Acc: 0.9476

Epoch 42/68
----------
train Loss: 0.0011 Acc: 0.9653
val Loss: 0.0017 Acc: 0.9453

Epoch 43/68
----------
train Loss: 0.0011 Acc: 0.9733
val Loss: 0.0016 Acc: 0.9499

Epoch 44/68
----------
train Loss: 0.0010 Acc: 0.9665
val Loss: 0.0020 Acc: 0.9476

Epoch 45/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0016 Acc: 0.9453

Epoch 46/68
----------
train Loss: 0.0011 Acc: 0.9682
val Loss: 0.0017 Acc: 0.9453

Epoch 47/68
----------
train Loss: 0.0011 Acc: 0.9642
val Loss: 0.0018 Acc: 0.9476

Epoch 48/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9579
val Loss: 0.0018 Acc: 0.9476

Epoch 49/68
----------
train Loss: 0.0011 Acc: 0.9665
val Loss: 0.0016 Acc: 0.9431

Epoch 50/68
----------
train Loss: 0.0010 Acc: 0.9733
val Loss: 0.0016 Acc: 0.9476

Epoch 51/68
----------
train Loss: 0.0011 Acc: 0.9665
val Loss: 0.0016 Acc: 0.9476

Epoch 52/68
----------
train Loss: 0.0011 Acc: 0.9665
val Loss: 0.0017 Acc: 0.9476

Epoch 53/68
----------
train Loss: 0.0011 Acc: 0.9602
val Loss: 0.0019 Acc: 0.9453

Epoch 54/68
----------
train Loss: 0.0012 Acc: 0.9608
val Loss: 0.0018 Acc: 0.9499

Epoch 55/68
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0016 Acc: 0.9476

Epoch 56/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9630
val Loss: 0.0017 Acc: 0.9453

Epoch 57/68
----------
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0017 Acc: 0.9453

Epoch 58/68
----------
train Loss: 0.0010 Acc: 0.9704
val Loss: 0.0018 Acc: 0.9431

Epoch 59/68
----------
train Loss: 0.0012 Acc: 0.9670
val Loss: 0.0019 Acc: 0.9431

Epoch 60/68
----------
train Loss: 0.0011 Acc: 0.9682
val Loss: 0.0017 Acc: 0.9476

Epoch 61/68
----------
train Loss: 0.0011 Acc: 0.9619
val Loss: 0.0016 Acc: 0.9453

Epoch 62/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0018 Acc: 0.9476

Epoch 63/68
----------
train Loss: 0.0011 Acc: 0.9648
val Loss: 0.0019 Acc: 0.9499

Epoch 64/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9625
val Loss: 0.0016 Acc: 0.9476

Epoch 65/68
----------
train Loss: 0.0010 Acc: 0.9625
val Loss: 0.0017 Acc: 0.9453

Epoch 66/68
----------
train Loss: 0.0012 Acc: 0.9574
val Loss: 0.0017 Acc: 0.9453

Epoch 67/68
----------
train Loss: 0.0011 Acc: 0.9608
val Loss: 0.0018 Acc: 0.9499

Epoch 68/68
----------
train Loss: 0.0010 Acc: 0.9676
val Loss: 0.0017 Acc: 0.9476

Best val Acc: 0.952164

---Fine tuning.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0013 Acc: 0.9562
val Loss: 0.0016 Acc: 0.9431

Epoch 1/68
----------
train Loss: 0.0005 Acc: 0.9824
val Loss: 0.0013 Acc: 0.9636

Epoch 2/68
----------
train Loss: 0.0004 Acc: 0.9892
val Loss: 0.0015 Acc: 0.9613

Epoch 3/68
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0015 Acc: 0.9613

Epoch 4/68
----------
train Loss: 0.0002 Acc: 0.9955
val Loss: 0.0022 Acc: 0.9613

Epoch 5/68
----------
train Loss: 0.0001 Acc: 0.9932
val Loss: 0.0017 Acc: 0.9658

Epoch 6/68
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0022 Acc: 0.9681

Epoch 7/68
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0022 Acc: 0.9681

Epoch 8/68
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0019 Acc: 0.9681

Epoch 9/68
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0016 Acc: 0.9704

Epoch 10/68
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0017 Acc: 0.9681

Epoch 11/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9658

Epoch 12/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9658

Epoch 13/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9704

Epoch 14/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0021 Acc: 0.9704

Epoch 15/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 16/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9681

Epoch 17/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0018 Acc: 0.9658

Epoch 18/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9681

Epoch 19/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0020 Acc: 0.9704

Epoch 20/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9658

Epoch 21/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0019 Acc: 0.9681

Epoch 22/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9704

Epoch 23/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 24/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9681

Epoch 25/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0019 Acc: 0.9681

Epoch 26/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 27/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 28/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 29/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 30/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0019 Acc: 0.9681

Epoch 31/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0023 Acc: 0.9681

Epoch 32/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0016 Acc: 0.9681

Epoch 33/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0018 Acc: 0.9681

Epoch 34/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 35/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 36/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 37/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 38/68
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0020 Acc: 0.9681

Epoch 39/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0019 Acc: 0.9704

Epoch 40/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9681

Epoch 41/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 42/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0020 Acc: 0.9658

Epoch 43/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 44/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0023 Acc: 0.9704

Epoch 45/68
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0016 Acc: 0.9658

Epoch 46/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9681

Epoch 47/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 48/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9681

Epoch 49/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9704

Epoch 50/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0023 Acc: 0.9681

Epoch 51/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0019 Acc: 0.9681

Epoch 52/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0021 Acc: 0.9681

Epoch 53/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 54/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9681

Epoch 55/68
----------
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0016 Acc: 0.9681

Epoch 56/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0017 Acc: 0.9681

Epoch 57/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 58/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9704

Epoch 59/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0021 Acc: 0.9704

Epoch 60/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0017 Acc: 0.9681

Epoch 61/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9704

Epoch 62/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0019 Acc: 0.9704

Epoch 63/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9681

Epoch 64/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 0.9989
val Loss: 0.0020 Acc: 0.9681

Epoch 65/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0018 Acc: 0.9704

Epoch 66/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0019 Acc: 0.9681

Epoch 67/68
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0020 Acc: 0.9658

Epoch 68/68
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0017 Acc: 0.9681

Best val Acc: 0.970387

---Testing---
Test accuracy: 0.994086
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9906004785089261, std: 0.01129708679062659
--------------------

run info[val: 0.3, epoch: 95, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0058 Acc: 0.7596
val Loss: 0.0024 Acc: 0.9196

Epoch 1/94
----------
train Loss: 0.0026 Acc: 0.9058
val Loss: 0.0027 Acc: 0.8968

Epoch 2/94
----------
train Loss: 0.0020 Acc: 0.9331
val Loss: 0.0017 Acc: 0.9423

Epoch 3/94
----------
train Loss: 0.0015 Acc: 0.9552
val Loss: 0.0017 Acc: 0.9439

Epoch 4/94
----------
train Loss: 0.0013 Acc: 0.9513
val Loss: 0.0020 Acc: 0.9256

Epoch 5/94
----------
train Loss: 0.0014 Acc: 0.9513
val Loss: 0.0015 Acc: 0.9454

Epoch 6/94
----------
train Loss: 0.0012 Acc: 0.9578
val Loss: 0.0017 Acc: 0.9469

Epoch 7/94
----------
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0016 Acc: 0.9454

Epoch 8/94
----------
train Loss: 0.0013 Acc: 0.9597
val Loss: 0.0020 Acc: 0.9332

Epoch 9/94
----------
train Loss: 0.0012 Acc: 0.9545
val Loss: 0.0016 Acc: 0.9499

Epoch 10/94
----------
train Loss: 0.0010 Acc: 0.9682
val Loss: 0.0017 Acc: 0.9484

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9734
val Loss: 0.0015 Acc: 0.9439

Epoch 12/94
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9499

Epoch 13/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9514

Epoch 14/94
----------
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0014 Acc: 0.9499

Epoch 15/94
----------
train Loss: 0.0009 Acc: 0.9740
val Loss: 0.0015 Acc: 0.9469

Epoch 16/94
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9469

Epoch 17/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9499

Epoch 18/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9514

Epoch 19/94
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0015 Acc: 0.9469

Epoch 20/94
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9514

Epoch 21/94
----------
train Loss: 0.0008 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9484

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9530

Epoch 23/94
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0016 Acc: 0.9469

Epoch 24/94
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9484

Epoch 25/94
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0014 Acc: 0.9499

Epoch 26/94
----------
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0014 Acc: 0.9484

Epoch 27/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9499

Epoch 28/94
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0014 Acc: 0.9484

Epoch 29/94
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0015 Acc: 0.9499

Epoch 30/94
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0014 Acc: 0.9454

Epoch 31/94
----------
train Loss: 0.0008 Acc: 0.9844
val Loss: 0.0015 Acc: 0.9469

Epoch 32/94
----------
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0014 Acc: 0.9484

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0014 Acc: 0.9469

Epoch 34/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9454

Epoch 35/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9499

Epoch 36/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9499

Epoch 37/94
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0015 Acc: 0.9484

Epoch 38/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9484

Epoch 39/94
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0014 Acc: 0.9499

Epoch 40/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 41/94
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9484

Epoch 42/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0016 Acc: 0.9469

Epoch 43/94
----------
train Loss: 0.0007 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9484

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9484

Epoch 45/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9514

Epoch 46/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9469

Epoch 47/94
----------
train Loss: 0.0007 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9499

Epoch 48/94
----------
train Loss: 0.0008 Acc: 0.9766
val Loss: 0.0015 Acc: 0.9530

Epoch 49/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 50/94
----------
train Loss: 0.0009 Acc: 0.9760
val Loss: 0.0014 Acc: 0.9469

Epoch 51/94
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0014 Acc: 0.9469

Epoch 52/94
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9499

Epoch 53/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9484

Epoch 54/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0014 Acc: 0.9484

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0007 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9484

Epoch 56/94
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0014 Acc: 0.9484

Epoch 57/94
----------
train Loss: 0.0007 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9484

Epoch 58/94
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9484

Epoch 59/94
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0014 Acc: 0.9454

Epoch 60/94
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 61/94
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0014 Acc: 0.9469

Epoch 62/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0014 Acc: 0.9454

Epoch 63/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9469

Epoch 64/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 65/94
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0014 Acc: 0.9484

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9831
val Loss: 0.0015 Acc: 0.9469

Epoch 67/94
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0015 Acc: 0.9484

Epoch 68/94
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0014 Acc: 0.9469

Epoch 69/94
----------
train Loss: 0.0007 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 70/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0014 Acc: 0.9469

Epoch 71/94
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0014 Acc: 0.9499

Epoch 72/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9454

Epoch 73/94
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0014 Acc: 0.9469

Epoch 74/94
----------
train Loss: 0.0007 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9469

Epoch 75/94
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0015 Acc: 0.9499

Epoch 76/94
----------
train Loss: 0.0009 Acc: 0.9753
val Loss: 0.0015 Acc: 0.9484

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0015 Acc: 0.9484

Epoch 78/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0014 Acc: 0.9499

Epoch 79/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 80/94
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0014 Acc: 0.9469

Epoch 81/94
----------
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0014 Acc: 0.9469

Epoch 82/94
----------
train Loss: 0.0008 Acc: 0.9844
val Loss: 0.0014 Acc: 0.9484

Epoch 83/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 84/94
----------
train Loss: 0.0007 Acc: 0.9805
val Loss: 0.0015 Acc: 0.9469

Epoch 85/94
----------
train Loss: 0.0008 Acc: 0.9805
val Loss: 0.0014 Acc: 0.9469

Epoch 86/94
----------
train Loss: 0.0009 Acc: 0.9734
val Loss: 0.0015 Acc: 0.9454

Epoch 87/94
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0014 Acc: 0.9530

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0008 Acc: 0.9799
val Loss: 0.0015 Acc: 0.9484

Epoch 89/94
----------
train Loss: 0.0008 Acc: 0.9812
val Loss: 0.0015 Acc: 0.9484

Epoch 90/94
----------
train Loss: 0.0007 Acc: 0.9786
val Loss: 0.0015 Acc: 0.9484

Epoch 91/94
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0016 Acc: 0.9469

Epoch 92/94
----------
train Loss: 0.0008 Acc: 0.9818
val Loss: 0.0015 Acc: 0.9484

Epoch 93/94
----------
train Loss: 0.0007 Acc: 0.9799
val Loss: 0.0014 Acc: 0.9514

Epoch 94/94
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0015 Acc: 0.9484

Best val Acc: 0.952959

---Fine tuning.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0009 Acc: 0.9675
val Loss: 0.0021 Acc: 0.9408

Epoch 1/94
----------
train Loss: 0.0003 Acc: 0.9922
val Loss: 0.0014 Acc: 0.9621

Epoch 2/94
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0009 Acc: 0.9742

Epoch 3/94
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0009 Acc: 0.9666

Epoch 4/94
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0010 Acc: 0.9727

Epoch 5/94
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0016 Acc: 0.9651

Epoch 6/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9666

Epoch 7/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9681

Epoch 8/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9681

Epoch 9/94
----------
train Loss: 0.0000 Acc: 0.9987
val Loss: 0.0011 Acc: 0.9712

Epoch 10/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 12/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 13/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 14/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 15/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9757

Epoch 16/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9757

Epoch 17/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 18/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9742

Epoch 19/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 20/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 21/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 23/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9742

Epoch 24/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 25/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 26/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 27/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9727

Epoch 28/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 29/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 30/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 31/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 32/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 34/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 35/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 36/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 37/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9727

Epoch 38/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 39/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 40/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 41/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9712

Epoch 42/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 43/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 45/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 46/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9727

Epoch 47/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 48/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 49/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 50/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9727

Epoch 51/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 52/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 53/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 54/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9742

Epoch 56/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 57/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 58/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9697

Epoch 59/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 60/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 61/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 62/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 63/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 64/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 65/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 67/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 68/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 69/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 70/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 71/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0013 Acc: 0.9727

Epoch 72/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 73/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 74/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 75/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 76/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 78/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9712

Epoch 79/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 80/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 81/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 82/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9742

Epoch 83/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 84/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Epoch 85/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9742

Epoch 86/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9697

Epoch 87/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 89/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9712

Epoch 90/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 91/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9697

Epoch 92/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9742

Epoch 93/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9727

Epoch 94/94
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9727

Best val Acc: 0.975721

---Testing---
Test accuracy: 0.992721
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9894956984638691, std: 0.010503226269851581

Model saved in "./weights/super_class_[0.99]_mean[0.99]_std[0.01].save".
--------------------

run info[val: 0.1, epoch: 84, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0051 Acc: 0.7898
val Loss: 0.0021 Acc: 0.9315

Epoch 1/83
----------
train Loss: 0.0021 Acc: 0.9252
val Loss: 0.0023 Acc: 0.9635

Epoch 2/83
----------
train Loss: 0.0018 Acc: 0.9363
val Loss: 0.0026 Acc: 0.9635

Epoch 3/83
----------
train Loss: 0.0014 Acc: 0.9409
val Loss: 0.0016 Acc: 0.9726

Epoch 4/83
----------
train Loss: 0.0014 Acc: 0.9525
val Loss: 0.0014 Acc: 0.9680

Epoch 5/83
----------
train Loss: 0.0014 Acc: 0.9520
val Loss: 0.0015 Acc: 0.9589

Epoch 6/83
----------
LR is set to 0.001
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0015 Acc: 0.9680

Epoch 7/83
----------
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0019 Acc: 0.9635

Epoch 8/83
----------
train Loss: 0.0013 Acc: 0.9525
val Loss: 0.0026 Acc: 0.9635

Epoch 9/83
----------
train Loss: 0.0012 Acc: 0.9611
val Loss: 0.0014 Acc: 0.9635

Epoch 10/83
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0014 Acc: 0.9635

Epoch 11/83
----------
train Loss: 0.0013 Acc: 0.9560
val Loss: 0.0016 Acc: 0.9680

Epoch 12/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0011 Acc: 0.9606
val Loss: 0.0015 Acc: 0.9680

Epoch 13/83
----------
train Loss: 0.0013 Acc: 0.9570
val Loss: 0.0013 Acc: 0.9680

Epoch 14/83
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0018 Acc: 0.9680

Epoch 15/83
----------
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0015 Acc: 0.9635

Epoch 16/83
----------
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0015 Acc: 0.9680

Epoch 17/83
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0018 Acc: 0.9635

Epoch 18/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0012 Acc: 0.9611
val Loss: 0.0014 Acc: 0.9680

Epoch 19/83
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0017 Acc: 0.9635

Epoch 20/83
----------
train Loss: 0.0012 Acc: 0.9596
val Loss: 0.0016 Acc: 0.9680

Epoch 21/83
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0025 Acc: 0.9680

Epoch 22/83
----------
train Loss: 0.0012 Acc: 0.9540
val Loss: 0.0014 Acc: 0.9680

Epoch 23/83
----------
train Loss: 0.0012 Acc: 0.9631
val Loss: 0.0015 Acc: 0.9680

Epoch 24/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0014 Acc: 0.9635

Epoch 25/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0017 Acc: 0.9635

Epoch 26/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9680

Epoch 27/83
----------
train Loss: 0.0012 Acc: 0.9616
val Loss: 0.0014 Acc: 0.9635

Epoch 28/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0014 Acc: 0.9680

Epoch 29/83
----------
train Loss: 0.0012 Acc: 0.9646
val Loss: 0.0018 Acc: 0.9635

Epoch 30/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0024 Acc: 0.9635

Epoch 31/83
----------
train Loss: 0.0011 Acc: 0.9636
val Loss: 0.0013 Acc: 0.9635

Epoch 32/83
----------
train Loss: 0.0012 Acc: 0.9545
val Loss: 0.0025 Acc: 0.9635

Epoch 33/83
----------
train Loss: 0.0012 Acc: 0.9616
val Loss: 0.0014 Acc: 0.9635

Epoch 34/83
----------
train Loss: 0.0012 Acc: 0.9646
val Loss: 0.0019 Acc: 0.9680

Epoch 35/83
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0015 Acc: 0.9635

Epoch 36/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0015 Acc: 0.9635

Epoch 37/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0015 Acc: 0.9635

Epoch 38/83
----------
train Loss: 0.0012 Acc: 0.9560
val Loss: 0.0015 Acc: 0.9635

Epoch 39/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0018 Acc: 0.9635

Epoch 40/83
----------
train Loss: 0.0011 Acc: 0.9606
val Loss: 0.0020 Acc: 0.9635

Epoch 41/83
----------
train Loss: 0.0012 Acc: 0.9616
val Loss: 0.0016 Acc: 0.9635

Epoch 42/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0013 Acc: 0.9555
val Loss: 0.0015 Acc: 0.9635

Epoch 43/83
----------
train Loss: 0.0012 Acc: 0.9550
val Loss: 0.0016 Acc: 0.9635

Epoch 44/83
----------
train Loss: 0.0011 Acc: 0.9606
val Loss: 0.0013 Acc: 0.9635

Epoch 45/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0025 Acc: 0.9635

Epoch 46/83
----------
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0017 Acc: 0.9680

Epoch 47/83
----------
train Loss: 0.0011 Acc: 0.9616
val Loss: 0.0024 Acc: 0.9680

Epoch 48/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9586
val Loss: 0.0032 Acc: 0.9680

Epoch 49/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0013 Acc: 0.9635

Epoch 50/83
----------
train Loss: 0.0012 Acc: 0.9581
val Loss: 0.0016 Acc: 0.9635

Epoch 51/83
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0015 Acc: 0.9680

Epoch 52/83
----------
train Loss: 0.0012 Acc: 0.9636
val Loss: 0.0015 Acc: 0.9680

Epoch 53/83
----------
train Loss: 0.0012 Acc: 0.9565
val Loss: 0.0027 Acc: 0.9680

Epoch 54/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0011 Acc: 0.9621
val Loss: 0.0014 Acc: 0.9635

Epoch 55/83
----------
train Loss: 0.0011 Acc: 0.9621
val Loss: 0.0019 Acc: 0.9635

Epoch 56/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0015 Acc: 0.9680

Epoch 57/83
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0013 Acc: 0.9680

Epoch 58/83
----------
train Loss: 0.0011 Acc: 0.9596
val Loss: 0.0026 Acc: 0.9635

Epoch 59/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0015 Acc: 0.9635

Epoch 60/83
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0017 Acc: 0.9680

Epoch 61/83
----------
train Loss: 0.0012 Acc: 0.9611
val Loss: 0.0019 Acc: 0.9680

Epoch 62/83
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0016 Acc: 0.9635

Epoch 63/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0014 Acc: 0.9635

Epoch 64/83
----------
train Loss: 0.0011 Acc: 0.9641
val Loss: 0.0016 Acc: 0.9635

Epoch 65/83
----------
train Loss: 0.0012 Acc: 0.9601
val Loss: 0.0016 Acc: 0.9635

Epoch 66/83
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0012 Acc: 0.9641
val Loss: 0.0014 Acc: 0.9635

Epoch 67/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0017 Acc: 0.9680

Epoch 68/83
----------
train Loss: 0.0012 Acc: 0.9545
val Loss: 0.0014 Acc: 0.9680

Epoch 69/83
----------
train Loss: 0.0012 Acc: 0.9616
val Loss: 0.0014 Acc: 0.9635

Epoch 70/83
----------
train Loss: 0.0011 Acc: 0.9631
val Loss: 0.0022 Acc: 0.9635

Epoch 71/83
----------
train Loss: 0.0012 Acc: 0.9626
val Loss: 0.0015 Acc: 0.9635

Epoch 72/83
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0016 Acc: 0.9680

Epoch 73/83
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0015 Acc: 0.9635

Epoch 74/83
----------
train Loss: 0.0012 Acc: 0.9606
val Loss: 0.0018 Acc: 0.9635

Epoch 75/83
----------
train Loss: 0.0011 Acc: 0.9601
val Loss: 0.0025 Acc: 0.9635

Epoch 76/83
----------
train Loss: 0.0012 Acc: 0.9560
val Loss: 0.0019 Acc: 0.9635

Epoch 77/83
----------
train Loss: 0.0012 Acc: 0.9621
val Loss: 0.0023 Acc: 0.9680

Epoch 78/83
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0012 Acc: 0.9636
val Loss: 0.0013 Acc: 0.9680

Epoch 79/83
----------
train Loss: 0.0012 Acc: 0.9611
val Loss: 0.0013 Acc: 0.9635

Epoch 80/83
----------
train Loss: 0.0013 Acc: 0.9515
val Loss: 0.0013 Acc: 0.9680

Epoch 81/83
----------
train Loss: 0.0012 Acc: 0.9576
val Loss: 0.0015 Acc: 0.9635

Epoch 82/83
----------
train Loss: 0.0011 Acc: 0.9611
val Loss: 0.0015 Acc: 0.9635

Epoch 83/83
----------
train Loss: 0.0011 Acc: 0.9576
val Loss: 0.0014 Acc: 0.9635

Best val Acc: 0.972603

---Fine tuning.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0017 Acc: 0.9498

Epoch 1/83
----------
train Loss: 0.0008 Acc: 0.9661
val Loss: 0.0009 Acc: 0.9817

Epoch 2/83
----------
train Loss: 0.0004 Acc: 0.9833
val Loss: 0.0013 Acc: 0.9726

Epoch 3/83
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0023 Acc: 0.9680

Epoch 4/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0014 Acc: 0.9635

Epoch 5/83
----------
train Loss: 0.0001 Acc: 0.9965
val Loss: 0.0018 Acc: 0.9635

Epoch 6/83
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9726

Epoch 7/83
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0032 Acc: 0.9680

Epoch 8/83
----------
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0012 Acc: 0.9635

Epoch 9/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9635

Epoch 10/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0021 Acc: 0.9680

Epoch 11/83
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9680

Epoch 12/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9975
val Loss: 0.0012 Acc: 0.9680

Epoch 13/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0013 Acc: 0.9680

Epoch 14/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 15/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 16/83
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0017 Acc: 0.9680

Epoch 17/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0019 Acc: 0.9680

Epoch 18/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 19/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0044 Acc: 0.9680

Epoch 20/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 21/83
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0012 Acc: 0.9680

Epoch 22/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0019 Acc: 0.9680

Epoch 23/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 24/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 25/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0015 Acc: 0.9680

Epoch 26/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 27/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0011 Acc: 0.9680

Epoch 28/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 29/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0014 Acc: 0.9680

Epoch 30/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9680

Epoch 31/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0015 Acc: 0.9680

Epoch 32/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 33/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0024 Acc: 0.9680

Epoch 34/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 35/83
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 36/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9680

Epoch 37/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0011 Acc: 0.9680

Epoch 38/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 39/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0016 Acc: 0.9680

Epoch 40/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9680

Epoch 41/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 42/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9995
val Loss: 0.0015 Acc: 0.9680

Epoch 43/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0013 Acc: 0.9680

Epoch 44/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0020 Acc: 0.9680

Epoch 45/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 46/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0023 Acc: 0.9680

Epoch 47/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 48/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 49/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0032 Acc: 0.9680

Epoch 50/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0016 Acc: 0.9680

Epoch 51/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0012 Acc: 0.9680

Epoch 52/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 53/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 54/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9680

Epoch 55/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 56/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 57/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 58/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 59/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0016 Acc: 0.9680

Epoch 60/83
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9680

Epoch 61/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0012 Acc: 0.9680

Epoch 62/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 63/83
----------
train Loss: 0.0001 Acc: 0.9980
val Loss: 0.0014 Acc: 0.9680

Epoch 64/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 65/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0020 Acc: 0.9680

Epoch 66/83
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9680

Epoch 67/83
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 68/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0015 Acc: 0.9680

Epoch 69/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0021 Acc: 0.9680

Epoch 70/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 71/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 72/83
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0015 Acc: 0.9680

Epoch 73/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0014 Acc: 0.9680

Epoch 74/83
----------
train Loss: 0.0001 Acc: 0.9985
val Loss: 0.0011 Acc: 0.9680

Epoch 75/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0015 Acc: 0.9680

Epoch 76/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0011 Acc: 0.9680

Epoch 77/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0012 Acc: 0.9680

Epoch 78/83
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0011 Acc: 0.9680

Epoch 79/83
----------
train Loss: 0.0001 Acc: 0.9990
val Loss: 0.0012 Acc: 0.9680

Epoch 80/83
----------
train Loss: 0.0000 Acc: 0.9985
val Loss: 0.0015 Acc: 0.9680

Epoch 81/83
----------
train Loss: 0.0000 Acc: 0.9990
val Loss: 0.0011 Acc: 0.9680

Epoch 82/83
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0014 Acc: 0.9680

Epoch 83/83
----------
train Loss: 0.0000 Acc: 0.9995
val Loss: 0.0015 Acc: 0.9680

Best val Acc: 0.981735

---Testing---
Test accuracy: 0.990901
--------------------
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9893111278919503, std: 0.004939354322860906
--------------------

run info[val: 0.2, epoch: 49, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/48
----------
LR is set to 0.01
train Loss: 0.0061 Acc: 0.7482
val Loss: 0.0036 Acc: 0.8907

Epoch 1/48
----------
train Loss: 0.0022 Acc: 0.9227
val Loss: 0.0022 Acc: 0.9408

Epoch 2/48
----------
train Loss: 0.0018 Acc: 0.9363
val Loss: 0.0021 Acc: 0.9431

Epoch 3/48
----------
LR is set to 0.001
train Loss: 0.0015 Acc: 0.9471
val Loss: 0.0018 Acc: 0.9431

Epoch 4/48
----------
train Loss: 0.0015 Acc: 0.9483
val Loss: 0.0020 Acc: 0.9431

Epoch 5/48
----------
train Loss: 0.0015 Acc: 0.9443
val Loss: 0.0021 Acc: 0.9431

Epoch 6/48
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9471
val Loss: 0.0019 Acc: 0.9453

Epoch 7/48
----------
train Loss: 0.0014 Acc: 0.9494
val Loss: 0.0018 Acc: 0.9431

Epoch 8/48
----------
train Loss: 0.0015 Acc: 0.9483
val Loss: 0.0018 Acc: 0.9431

Epoch 9/48
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0017 Acc: 0.9437
val Loss: 0.0019 Acc: 0.9431

Epoch 10/48
----------
train Loss: 0.0014 Acc: 0.9505
val Loss: 0.0020 Acc: 0.9408

Epoch 11/48
----------
train Loss: 0.0015 Acc: 0.9414
val Loss: 0.0021 Acc: 0.9408

Epoch 12/48
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9420
val Loss: 0.0017 Acc: 0.9453

Epoch 13/48
----------
train Loss: 0.0014 Acc: 0.9449
val Loss: 0.0022 Acc: 0.9431

Epoch 14/48
----------
train Loss: 0.0014 Acc: 0.9500
val Loss: 0.0019 Acc: 0.9453

Epoch 15/48
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0021 Acc: 0.9408

Epoch 16/48
----------
train Loss: 0.0015 Acc: 0.9386
val Loss: 0.0019 Acc: 0.9431

Epoch 17/48
----------
train Loss: 0.0016 Acc: 0.9443
val Loss: 0.0019 Acc: 0.9431

Epoch 18/48
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0014 Acc: 0.9483
val Loss: 0.0020 Acc: 0.9408

Epoch 19/48
----------
train Loss: 0.0014 Acc: 0.9522
val Loss: 0.0019 Acc: 0.9408

Epoch 20/48
----------
train Loss: 0.0014 Acc: 0.9414
val Loss: 0.0021 Acc: 0.9431

Epoch 21/48
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0014 Acc: 0.9557
val Loss: 0.0018 Acc: 0.9431

Epoch 22/48
----------
train Loss: 0.0016 Acc: 0.9397
val Loss: 0.0019 Acc: 0.9431

Epoch 23/48
----------
train Loss: 0.0016 Acc: 0.9420
val Loss: 0.0018 Acc: 0.9408

Epoch 24/48
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0016 Acc: 0.9483
val Loss: 0.0020 Acc: 0.9408

Epoch 25/48
----------
train Loss: 0.0014 Acc: 0.9540
val Loss: 0.0021 Acc: 0.9408

Epoch 26/48
----------
train Loss: 0.0015 Acc: 0.9431
val Loss: 0.0020 Acc: 0.9431

Epoch 27/48
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0016 Acc: 0.9471
val Loss: 0.0020 Acc: 0.9408

Epoch 28/48
----------
train Loss: 0.0015 Acc: 0.9454
val Loss: 0.0018 Acc: 0.9453

Epoch 29/48
----------
train Loss: 0.0015 Acc: 0.9392
val Loss: 0.0020 Acc: 0.9408

Epoch 30/48
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0014 Acc: 0.9540
val Loss: 0.0020 Acc: 0.9408

Epoch 31/48
----------
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0021 Acc: 0.9431

Epoch 32/48
----------
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0019 Acc: 0.9408

Epoch 33/48
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0015 Acc: 0.9483
val Loss: 0.0020 Acc: 0.9408

Epoch 34/48
----------
train Loss: 0.0015 Acc: 0.9511
val Loss: 0.0018 Acc: 0.9408

Epoch 35/48
----------
train Loss: 0.0015 Acc: 0.9449
val Loss: 0.0018 Acc: 0.9431

Epoch 36/48
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0015 Acc: 0.9454
val Loss: 0.0019 Acc: 0.9453

Epoch 37/48
----------
train Loss: 0.0015 Acc: 0.9477
val Loss: 0.0019 Acc: 0.9408

Epoch 38/48
----------
train Loss: 0.0015 Acc: 0.9460
val Loss: 0.0021 Acc: 0.9408

Epoch 39/48
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0015 Acc: 0.9522
val Loss: 0.0020 Acc: 0.9408

Epoch 40/48
----------
train Loss: 0.0016 Acc: 0.9449
val Loss: 0.0020 Acc: 0.9408

Epoch 41/48
----------
train Loss: 0.0015 Acc: 0.9460
val Loss: 0.0018 Acc: 0.9408

Epoch 42/48
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0015 Acc: 0.9431
val Loss: 0.0020 Acc: 0.9408

Epoch 43/48
----------
train Loss: 0.0016 Acc: 0.9397
val Loss: 0.0021 Acc: 0.9431

Epoch 44/48
----------
train Loss: 0.0015 Acc: 0.9511
val Loss: 0.0024 Acc: 0.9408

Epoch 45/48
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0015 Acc: 0.9466
val Loss: 0.0022 Acc: 0.9408

Epoch 46/48
----------
train Loss: 0.0016 Acc: 0.9449
val Loss: 0.0018 Acc: 0.9408

Epoch 47/48
----------
train Loss: 0.0015 Acc: 0.9443
val Loss: 0.0019 Acc: 0.9408

Epoch 48/48
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0014 Acc: 0.9522
val Loss: 0.0021 Acc: 0.9431

Best val Acc: 0.945330

---Fine tuning.---
Epoch 0/48
----------
LR is set to 0.01
train Loss: 0.0014 Acc: 0.9494
val Loss: 0.0021 Acc: 0.9590

Epoch 1/48
----------
train Loss: 0.0006 Acc: 0.9818
val Loss: 0.0011 Acc: 0.9681

Epoch 2/48
----------
train Loss: 0.0003 Acc: 0.9892
val Loss: 0.0013 Acc: 0.9727

Epoch 3/48
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0013 Acc: 0.9704

Epoch 4/48
----------
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0011 Acc: 0.9704

Epoch 5/48
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0010 Acc: 0.9681

Epoch 6/48
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0012 Acc: 0.9704

Epoch 7/48
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 8/48
----------
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0009 Acc: 0.9704

Epoch 9/48
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9943
val Loss: 0.0010 Acc: 0.9704

Epoch 10/48
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0009 Acc: 0.9704

Epoch 11/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0009 Acc: 0.9704

Epoch 12/48
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0010 Acc: 0.9704

Epoch 13/48
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9704

Epoch 14/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0010 Acc: 0.9681

Epoch 15/48
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9966
val Loss: 0.0010 Acc: 0.9704

Epoch 16/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0009 Acc: 0.9704

Epoch 17/48
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 18/48
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0012 Acc: 0.9681

Epoch 19/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0010 Acc: 0.9681

Epoch 20/48
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0009 Acc: 0.9704

Epoch 21/48
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0011 Acc: 0.9704

Epoch 22/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0010 Acc: 0.9704

Epoch 23/48
----------
train Loss: 0.0001 Acc: 0.9943
val Loss: 0.0011 Acc: 0.9704

Epoch 24/48
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9949
val Loss: 0.0010 Acc: 0.9704

Epoch 25/48
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9704

Epoch 26/48
----------
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 27/48
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0010 Acc: 0.9704

Epoch 28/48
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0009 Acc: 0.9704

Epoch 29/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0011 Acc: 0.9704

Epoch 30/48
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0010 Acc: 0.9704

Epoch 31/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0009 Acc: 0.9704

Epoch 32/48
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0011 Acc: 0.9704

Epoch 33/48
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9977
val Loss: 0.0012 Acc: 0.9681

Epoch 34/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0009 Acc: 0.9704

Epoch 35/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0009 Acc: 0.9704

Epoch 36/48
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0009 Acc: 0.9704

Epoch 37/48
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9704

Epoch 38/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0012 Acc: 0.9704

Epoch 39/48
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0011 Acc: 0.9704

Epoch 40/48
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0011 Acc: 0.9681

Epoch 41/48
----------
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0011 Acc: 0.9704

Epoch 42/48
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0001 Acc: 0.9960
val Loss: 0.0011 Acc: 0.9704

Epoch 43/48
----------
train Loss: 0.0001 Acc: 0.9983
val Loss: 0.0011 Acc: 0.9681

Epoch 44/48
----------
train Loss: 0.0001 Acc: 0.9989
val Loss: 0.0009 Acc: 0.9704

Epoch 45/48
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0001 Acc: 0.9955
val Loss: 0.0009 Acc: 0.9704

Epoch 46/48
----------
train Loss: 0.0001 Acc: 0.9972
val Loss: 0.0009 Acc: 0.9704

Epoch 47/48
----------
train Loss: 0.0001 Acc: 0.9966
val Loss: 0.0009 Acc: 0.9704

Epoch 48/48
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 0.9977
val Loss: 0.0009 Acc: 0.9704

Best val Acc: 0.972665

---Testing---
Test accuracy: 0.991356
--------------------
Accuracy of Batoidea(ga_oo_lee) : 96 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9865649416921816, std: 0.01507159357303237
--------------------

run info[val: 0.3, epoch: 92, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0064 Acc: 0.7147
val Loss: 0.0025 Acc: 0.9226

Epoch 1/91
----------
train Loss: 0.0022 Acc: 0.9162
val Loss: 0.0022 Acc: 0.9256

Epoch 2/91
----------
train Loss: 0.0022 Acc: 0.9220
val Loss: 0.0019 Acc: 0.9363

Epoch 3/91
----------
train Loss: 0.0018 Acc: 0.9376
val Loss: 0.0018 Acc: 0.9439

Epoch 4/91
----------
train Loss: 0.0018 Acc: 0.9370
val Loss: 0.0017 Acc: 0.9408

Epoch 5/91
----------
LR is set to 0.001
train Loss: 0.0014 Acc: 0.9487
val Loss: 0.0017 Acc: 0.9439

Epoch 6/91
----------
train Loss: 0.0013 Acc: 0.9597
val Loss: 0.0017 Acc: 0.9454

Epoch 7/91
----------
train Loss: 0.0014 Acc: 0.9532
val Loss: 0.0016 Acc: 0.9469

Epoch 8/91
----------
train Loss: 0.0013 Acc: 0.9552
val Loss: 0.0016 Acc: 0.9408

Epoch 9/91
----------
train Loss: 0.0014 Acc: 0.9493
val Loss: 0.0016 Acc: 0.9454

Epoch 10/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0012 Acc: 0.9584
val Loss: 0.0016 Acc: 0.9423

Epoch 11/91
----------
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9423

Epoch 12/91
----------
train Loss: 0.0012 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9439

Epoch 13/91
----------
train Loss: 0.0013 Acc: 0.9558
val Loss: 0.0016 Acc: 0.9408

Epoch 14/91
----------
train Loss: 0.0013 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9439

Epoch 15/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0012 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9439

Epoch 16/91
----------
train Loss: 0.0014 Acc: 0.9513
val Loss: 0.0017 Acc: 0.9439

Epoch 17/91
----------
train Loss: 0.0013 Acc: 0.9604
val Loss: 0.0016 Acc: 0.9393

Epoch 18/91
----------
train Loss: 0.0014 Acc: 0.9565
val Loss: 0.0017 Acc: 0.9423

Epoch 19/91
----------
train Loss: 0.0013 Acc: 0.9617
val Loss: 0.0016 Acc: 0.9423

Epoch 20/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0015 Acc: 0.9439

Epoch 21/91
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9423

Epoch 22/91
----------
train Loss: 0.0013 Acc: 0.9584
val Loss: 0.0016 Acc: 0.9423

Epoch 23/91
----------
train Loss: 0.0012 Acc: 0.9610
val Loss: 0.0017 Acc: 0.9439

Epoch 24/91
----------
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9439

Epoch 25/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9423

Epoch 26/91
----------
train Loss: 0.0013 Acc: 0.9617
val Loss: 0.0016 Acc: 0.9423

Epoch 27/91
----------
train Loss: 0.0012 Acc: 0.9584
val Loss: 0.0016 Acc: 0.9423

Epoch 28/91
----------
train Loss: 0.0014 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9439

Epoch 29/91
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9423

Epoch 30/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0013 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9423

Epoch 31/91
----------
train Loss: 0.0012 Acc: 0.9578
val Loss: 0.0017 Acc: 0.9423

Epoch 32/91
----------
train Loss: 0.0013 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9439

Epoch 33/91
----------
train Loss: 0.0014 Acc: 0.9552
val Loss: 0.0016 Acc: 0.9439

Epoch 34/91
----------
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0017 Acc: 0.9423

Epoch 35/91
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0016 Acc: 0.9439

Epoch 36/91
----------
train Loss: 0.0014 Acc: 0.9519
val Loss: 0.0016 Acc: 0.9454

Epoch 37/91
----------
train Loss: 0.0012 Acc: 0.9493
val Loss: 0.0017 Acc: 0.9423

Epoch 38/91
----------
train Loss: 0.0012 Acc: 0.9532
val Loss: 0.0017 Acc: 0.9454

Epoch 39/91
----------
train Loss: 0.0012 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9439

Epoch 40/91
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9423

Epoch 41/91
----------
train Loss: 0.0013 Acc: 0.9610
val Loss: 0.0016 Acc: 0.9423

Epoch 42/91
----------
train Loss: 0.0015 Acc: 0.9526
val Loss: 0.0016 Acc: 0.9439

Epoch 43/91
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0017 Acc: 0.9423

Epoch 44/91
----------
train Loss: 0.0015 Acc: 0.9506
val Loss: 0.0016 Acc: 0.9423

Epoch 45/91
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9454

Epoch 46/91
----------
train Loss: 0.0013 Acc: 0.9558
val Loss: 0.0016 Acc: 0.9439

Epoch 47/91
----------
train Loss: 0.0013 Acc: 0.9623
val Loss: 0.0016 Acc: 0.9439

Epoch 48/91
----------
train Loss: 0.0013 Acc: 0.9552
val Loss: 0.0016 Acc: 0.9423

Epoch 49/91
----------
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9439

Epoch 50/91
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0014 Acc: 0.9526
val Loss: 0.0016 Acc: 0.9439

Epoch 51/91
----------
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9439

Epoch 52/91
----------
train Loss: 0.0013 Acc: 0.9617
val Loss: 0.0016 Acc: 0.9439

Epoch 53/91
----------
train Loss: 0.0012 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9423

Epoch 54/91
----------
train Loss: 0.0014 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9423

Epoch 55/91
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0013 Acc: 0.9610
val Loss: 0.0016 Acc: 0.9423

Epoch 56/91
----------
train Loss: 0.0013 Acc: 0.9565
val Loss: 0.0016 Acc: 0.9439

Epoch 57/91
----------
train Loss: 0.0012 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9454

Epoch 58/91
----------
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9423

Epoch 59/91
----------
train Loss: 0.0013 Acc: 0.9545
val Loss: 0.0016 Acc: 0.9423

Epoch 60/91
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0017 Acc: 0.9423

Epoch 61/91
----------
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9423

Epoch 62/91
----------
train Loss: 0.0012 Acc: 0.9656
val Loss: 0.0016 Acc: 0.9423

Epoch 63/91
----------
train Loss: 0.0012 Acc: 0.9604
val Loss: 0.0017 Acc: 0.9439

Epoch 64/91
----------
train Loss: 0.0012 Acc: 0.9591
val Loss: 0.0017 Acc: 0.9439

Epoch 65/91
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0012 Acc: 0.9604
val Loss: 0.0016 Acc: 0.9408

Epoch 66/91
----------
train Loss: 0.0012 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9423

Epoch 67/91
----------
train Loss: 0.0014 Acc: 0.9545
val Loss: 0.0017 Acc: 0.9423

Epoch 68/91
----------
train Loss: 0.0013 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9454

Epoch 69/91
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9423

Epoch 70/91
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0012 Acc: 0.9675
val Loss: 0.0016 Acc: 0.9423

Epoch 71/91
----------
train Loss: 0.0013 Acc: 0.9552
val Loss: 0.0016 Acc: 0.9454

Epoch 72/91
----------
train Loss: 0.0012 Acc: 0.9610
val Loss: 0.0016 Acc: 0.9439

Epoch 73/91
----------
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9439

Epoch 74/91
----------
train Loss: 0.0013 Acc: 0.9558
val Loss: 0.0016 Acc: 0.9439

Epoch 75/91
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9423

Epoch 76/91
----------
train Loss: 0.0014 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9423

Epoch 77/91
----------
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0017 Acc: 0.9423

Epoch 78/91
----------
train Loss: 0.0013 Acc: 0.9591
val Loss: 0.0016 Acc: 0.9408

Epoch 79/91
----------
train Loss: 0.0013 Acc: 0.9617
val Loss: 0.0016 Acc: 0.9408

Epoch 80/91
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0013 Acc: 0.9571
val Loss: 0.0016 Acc: 0.9423

Epoch 81/91
----------
train Loss: 0.0013 Acc: 0.9552
val Loss: 0.0016 Acc: 0.9439

Epoch 82/91
----------
train Loss: 0.0013 Acc: 0.9539
val Loss: 0.0016 Acc: 0.9439

Epoch 83/91
----------
train Loss: 0.0013 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9423

Epoch 84/91
----------
train Loss: 0.0013 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9408

Epoch 85/91
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0013 Acc: 0.9519
val Loss: 0.0016 Acc: 0.9439

Epoch 86/91
----------
train Loss: 0.0013 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9423

Epoch 87/91
----------
train Loss: 0.0012 Acc: 0.9597
val Loss: 0.0016 Acc: 0.9423

Epoch 88/91
----------
train Loss: 0.0014 Acc: 0.9578
val Loss: 0.0016 Acc: 0.9439

Epoch 89/91
----------
train Loss: 0.0012 Acc: 0.9584
val Loss: 0.0016 Acc: 0.9439

Epoch 90/91
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0013 Acc: 0.9519
val Loss: 0.0017 Acc: 0.9454

Epoch 91/91
----------
train Loss: 0.0014 Acc: 0.9584
val Loss: 0.0016 Acc: 0.9439

Best val Acc: 0.946889

---Fine tuning.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0015 Acc: 0.9506
val Loss: 0.0014 Acc: 0.9439

Epoch 1/91
----------
train Loss: 0.0007 Acc: 0.9740
val Loss: 0.0011 Acc: 0.9575

Epoch 2/91
----------
train Loss: 0.0004 Acc: 0.9916
val Loss: 0.0010 Acc: 0.9681

Epoch 3/91
----------
train Loss: 0.0002 Acc: 0.9948
val Loss: 0.0011 Acc: 0.9681

Epoch 4/91
----------
train Loss: 0.0002 Acc: 0.9961
val Loss: 0.0008 Acc: 0.9772

Epoch 5/91
----------
LR is set to 0.001
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9697

Epoch 6/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9712

Epoch 7/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9712

Epoch 8/91
----------
train Loss: 0.0001 Acc: 0.9968
val Loss: 0.0009 Acc: 0.9757

Epoch 9/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9727

Epoch 10/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 11/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 12/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 13/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 14/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9727

Epoch 15/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 16/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 17/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9742

Epoch 18/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 19/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 20/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 21/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 22/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9742

Epoch 23/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 24/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 25/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 26/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 27/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 28/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 29/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9742

Epoch 30/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9742

Epoch 31/91
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 32/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 33/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 34/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 35/91
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 36/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 37/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 38/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 39/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0009 Acc: 0.9742

Epoch 40/91
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 41/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 42/91
----------
train Loss: 0.0001 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 43/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 44/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 45/91
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 46/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 47/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 48/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 49/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0009 Acc: 0.9757

Epoch 50/91
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 51/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9742

Epoch 52/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 53/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 54/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 55/91
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 56/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 57/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 58/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9727

Epoch 59/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 60/91
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 61/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 62/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9727

Epoch 63/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 64/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 65/91
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 66/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 67/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9757

Epoch 68/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 69/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0009 Acc: 0.9757

Epoch 70/91
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 71/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 72/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0009 Acc: 0.9742

Epoch 73/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0009 Acc: 0.9757

Epoch 74/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 75/91
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0009 Acc: 0.9742

Epoch 76/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 77/91
----------
train Loss: 0.0000 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 78/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Epoch 79/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9757

Epoch 80/91
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 81/91
----------
train Loss: 0.0001 Acc: 0.9987
val Loss: 0.0008 Acc: 0.9742

Epoch 82/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0008 Acc: 0.9742

Epoch 83/91
----------
train Loss: 0.0001 Acc: 0.9974
val Loss: 0.0008 Acc: 0.9742

Epoch 84/91
----------
train Loss: 0.0001 Acc: 0.9981
val Loss: 0.0009 Acc: 0.9757

Epoch 85/91
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 86/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9742

Epoch 87/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0008 Acc: 0.9757

Epoch 88/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0009 Acc: 0.9757

Epoch 89/91
----------
train Loss: 0.0001 Acc: 0.9994
val Loss: 0.0009 Acc: 0.9757

Epoch 90/91
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9757

Epoch 91/91
----------
train Loss: 0.0000 Acc: 1.0000
val Loss: 0.0008 Acc: 0.9742

Best val Acc: 0.977238

---Testing---
Test accuracy: 0.992721
--------------------
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 100 %
mean: 0.9891448744332346, std: 0.010628871328699666

Model saved in "./weights/super_class_[0.99]_mean[0.99]_std[0.01].save".
Training complete in 359m 8s

Process finished with exit code 0
'''