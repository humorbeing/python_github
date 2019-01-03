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


train_this = 'tuna_fish'
test_this = 'tuna_fish'


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



'''/usr/bin/python3.5 "/home/visbic/python/pytorch_playground/datasets/new fish/git-test/weight_trainer.py"
--------------------

run info[val: 0.1, epoch: 62, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/61
----------
LR is set to 0.01
train Loss: 0.0267 Acc: 0.1476
val Loss: 0.0491 Acc: 0.2897

Epoch 1/61
----------
train Loss: 0.0222 Acc: 0.3044
val Loss: 0.0313 Acc: 0.3738

Epoch 2/61
----------
train Loss: 0.0191 Acc: 0.3767
val Loss: 0.0337 Acc: 0.4393

Epoch 3/61
----------
train Loss: 0.0164 Acc: 0.4974
val Loss: 0.0442 Acc: 0.4393

Epoch 4/61
----------
train Loss: 0.0146 Acc: 0.5490
val Loss: 0.0274 Acc: 0.4486

Epoch 5/61
----------
train Loss: 0.0134 Acc: 0.6017
val Loss: 0.0339 Acc: 0.4673

Epoch 6/61
----------
train Loss: 0.0123 Acc: 0.6460
val Loss: 0.0397 Acc: 0.4953

Epoch 7/61
----------
train Loss: 0.0114 Acc: 0.6698
val Loss: 0.0231 Acc: 0.5047

Epoch 8/61
----------
train Loss: 0.0110 Acc: 0.6821
val Loss: 0.0340 Acc: 0.5047

Epoch 9/61
----------
train Loss: 0.0105 Acc: 0.7059
val Loss: 0.0309 Acc: 0.4953

Epoch 10/61
----------
train Loss: 0.0099 Acc: 0.7224
val Loss: 0.0370 Acc: 0.5047

Epoch 11/61
----------
train Loss: 0.0100 Acc: 0.7193
val Loss: 0.0295 Acc: 0.4860

Epoch 12/61
----------
train Loss: 0.0098 Acc: 0.7090
val Loss: 0.0283 Acc: 0.4953

Epoch 13/61
----------
train Loss: 0.0093 Acc: 0.7286
val Loss: 0.0346 Acc: 0.4860

Epoch 14/61
----------
LR is set to 0.001
train Loss: 0.0086 Acc: 0.7730
val Loss: 0.0314 Acc: 0.5140

Epoch 15/61
----------
train Loss: 0.0083 Acc: 0.7699
val Loss: 0.0321 Acc: 0.5327

Epoch 16/61
----------
train Loss: 0.0083 Acc: 0.7802
val Loss: 0.0372 Acc: 0.5327

Epoch 17/61
----------
train Loss: 0.0083 Acc: 0.7946
val Loss: 0.0366 Acc: 0.5234

Epoch 18/61
----------
train Loss: 0.0081 Acc: 0.7843
val Loss: 0.0371 Acc: 0.5327

Epoch 19/61
----------
train Loss: 0.0082 Acc: 0.7967
val Loss: 0.0297 Acc: 0.5514

Epoch 20/61
----------
train Loss: 0.0081 Acc: 0.7905
val Loss: 0.0360 Acc: 0.5327

Epoch 21/61
----------
train Loss: 0.0080 Acc: 0.8039
val Loss: 0.0291 Acc: 0.5327

Epoch 22/61
----------
train Loss: 0.0083 Acc: 0.7750
val Loss: 0.0389 Acc: 0.5234

Epoch 23/61
----------
train Loss: 0.0081 Acc: 0.7946
val Loss: 0.0283 Acc: 0.5234

Epoch 24/61
----------
train Loss: 0.0081 Acc: 0.7967
val Loss: 0.0237 Acc: 0.5327

Epoch 25/61
----------
train Loss: 0.0081 Acc: 0.7988
val Loss: 0.0378 Acc: 0.5421

Epoch 26/61
----------
train Loss: 0.0081 Acc: 0.7936
val Loss: 0.0301 Acc: 0.5421

Epoch 27/61
----------
train Loss: 0.0078 Acc: 0.8029
val Loss: 0.0255 Acc: 0.5140

Epoch 28/61
----------
LR is set to 0.00010000000000000002
train Loss: 0.0078 Acc: 0.8039
val Loss: 0.0220 Acc: 0.5234

Epoch 29/61
----------
train Loss: 0.0080 Acc: 0.8019
val Loss: 0.0265 Acc: 0.5234

Epoch 30/61
----------
train Loss: 0.0080 Acc: 0.7998
val Loss: 0.0325 Acc: 0.5234

Epoch 31/61
----------
train Loss: 0.0079 Acc: 0.7998
val Loss: 0.0292 Acc: 0.5234

Epoch 32/61
----------
train Loss: 0.0077 Acc: 0.8060
val Loss: 0.0297 Acc: 0.5421

Epoch 33/61
----------
train Loss: 0.0081 Acc: 0.7895
val Loss: 0.0341 Acc: 0.5234

Epoch 34/61
----------
train Loss: 0.0079 Acc: 0.7936
val Loss: 0.0219 Acc: 0.5234

Epoch 35/61
----------
train Loss: 0.0079 Acc: 0.7946
val Loss: 0.0345 Acc: 0.5234

Epoch 36/61
----------
train Loss: 0.0078 Acc: 0.7946
val Loss: 0.0372 Acc: 0.5327

Epoch 37/61
----------
train Loss: 0.0079 Acc: 0.8122
val Loss: 0.0318 Acc: 0.5327

Epoch 38/61
----------
train Loss: 0.0081 Acc: 0.7936
val Loss: 0.0304 Acc: 0.5327

Epoch 39/61
----------
train Loss: 0.0077 Acc: 0.8132
val Loss: 0.0293 Acc: 0.5421

Epoch 40/61
----------
train Loss: 0.0078 Acc: 0.8029
val Loss: 0.0272 Acc: 0.5327

Epoch 41/61
----------
train Loss: 0.0080 Acc: 0.7895
val Loss: 0.0373 Acc: 0.5421

Epoch 42/61
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0079 Acc: 0.7946
val Loss: 0.0264 Acc: 0.5421

Epoch 43/61
----------
train Loss: 0.0077 Acc: 0.8163
val Loss: 0.0348 Acc: 0.5234

Epoch 44/61
----------
train Loss: 0.0079 Acc: 0.8050
val Loss: 0.0277 Acc: 0.5327

Epoch 45/61
----------
train Loss: 0.0078 Acc: 0.8070
val Loss: 0.0366 Acc: 0.5421

Epoch 46/61
----------
train Loss: 0.0081 Acc: 0.7946
val Loss: 0.0378 Acc: 0.5421

Epoch 47/61
----------
train Loss: 0.0078 Acc: 0.8132
val Loss: 0.0238 Acc: 0.5327

Epoch 48/61
----------
train Loss: 0.0079 Acc: 0.8029
val Loss: 0.0281 Acc: 0.5234

Epoch 49/61
----------
train Loss: 0.0078 Acc: 0.7967
val Loss: 0.0319 Acc: 0.5234

Epoch 50/61
----------
train Loss: 0.0079 Acc: 0.8060
val Loss: 0.0328 Acc: 0.5234

Epoch 51/61
----------
train Loss: 0.0080 Acc: 0.7957
val Loss: 0.0380 Acc: 0.5327

Epoch 52/61
----------
train Loss: 0.0077 Acc: 0.8153
val Loss: 0.0285 Acc: 0.5327

Epoch 53/61
----------
train Loss: 0.0077 Acc: 0.7967
val Loss: 0.0240 Acc: 0.5327

Epoch 54/61
----------
train Loss: 0.0077 Acc: 0.8122
val Loss: 0.0342 Acc: 0.5327

Epoch 55/61
----------
train Loss: 0.0080 Acc: 0.7895
val Loss: 0.0253 Acc: 0.5234

Epoch 56/61
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0079 Acc: 0.7998
val Loss: 0.0410 Acc: 0.5234

Epoch 57/61
----------
train Loss: 0.0080 Acc: 0.7936
val Loss: 0.0313 Acc: 0.5327

Epoch 58/61
----------
train Loss: 0.0079 Acc: 0.7977
val Loss: 0.0353 Acc: 0.5234

Epoch 59/61
----------
train Loss: 0.0080 Acc: 0.7926
val Loss: 0.0215 Acc: 0.5234

Epoch 60/61
----------
train Loss: 0.0078 Acc: 0.8122
val Loss: 0.0320 Acc: 0.5327

Epoch 61/61
----------
train Loss: 0.0078 Acc: 0.8091
val Loss: 0.0311 Acc: 0.5327

Training complete in 5m 45s
Best val Acc: 0.551402

---Fine tuning.---
Epoch 0/61
----------
LR is set to 0.01
train Loss: 0.0090 Acc: 0.7368
val Loss: 0.0342 Acc: 0.5234

Epoch 1/61
----------
train Loss: 0.0045 Acc: 0.8885
val Loss: 0.0281 Acc: 0.5888

Epoch 2/61
----------
train Loss: 0.0026 Acc: 0.9391
val Loss: 0.0202 Acc: 0.5701

Epoch 3/61
----------
train Loss: 0.0018 Acc: 0.9608
val Loss: 0.0336 Acc: 0.5794

Epoch 4/61
----------
train Loss: 0.0013 Acc: 0.9659
val Loss: 0.0241 Acc: 0.5607

Epoch 5/61
----------
train Loss: 0.0009 Acc: 0.9742
val Loss: 0.0271 Acc: 0.5888

Epoch 6/61
----------
train Loss: 0.0009 Acc: 0.9742
val Loss: 0.0245 Acc: 0.5140

Epoch 7/61
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0416 Acc: 0.5421

Epoch 8/61
----------
train Loss: 0.0006 Acc: 0.9783
val Loss: 0.0356 Acc: 0.5421

Epoch 9/61
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0186 Acc: 0.5327

Epoch 10/61
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0226 Acc: 0.5607

Epoch 11/61
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0250 Acc: 0.5794

Epoch 12/61
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0297 Acc: 0.5794

Epoch 13/61
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0452 Acc: 0.5794

Epoch 14/61
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0368 Acc: 0.5794

Epoch 15/61
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0369 Acc: 0.5794

Epoch 16/61
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0313 Acc: 0.5794

Epoch 17/61
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0395 Acc: 0.5794

Epoch 18/61
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0287 Acc: 0.5794

Epoch 19/61
----------
train Loss: 0.0004 Acc: 0.9794
val Loss: 0.0401 Acc: 0.5794

Epoch 20/61
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0179 Acc: 0.5794

Epoch 21/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0446 Acc: 0.5794

Epoch 22/61
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0460 Acc: 0.5794

Epoch 23/61
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0335 Acc: 0.5794

Epoch 24/61
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0354 Acc: 0.5794

Epoch 25/61
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0317 Acc: 0.5794

Epoch 26/61
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0253 Acc: 0.5701

Epoch 27/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0285 Acc: 0.5701

Epoch 28/61
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0390 Acc: 0.5701

Epoch 29/61
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0298 Acc: 0.5701

Epoch 30/61
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0494 Acc: 0.5701

Epoch 31/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0575 Acc: 0.5701

Epoch 32/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0323 Acc: 0.5701

Epoch 33/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0270 Acc: 0.5701

Epoch 34/61
----------
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0349 Acc: 0.5701

Epoch 35/61
----------
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0473 Acc: 0.5794

Epoch 36/61
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0349 Acc: 0.5701

Epoch 37/61
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0390 Acc: 0.5701

Epoch 38/61
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0463 Acc: 0.5794

Epoch 39/61
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0318 Acc: 0.5701

Epoch 40/61
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0354 Acc: 0.5701

Epoch 41/61
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0357 Acc: 0.5701

Epoch 42/61
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0302 Acc: 0.5701

Epoch 43/61
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0442 Acc: 0.5701

Epoch 44/61
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0317 Acc: 0.5794

Epoch 45/61
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0448 Acc: 0.5701

Epoch 46/61
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0290 Acc: 0.5794

Epoch 47/61
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0272 Acc: 0.5701

Epoch 48/61
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0248 Acc: 0.5701

Epoch 49/61
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0279 Acc: 0.5701

Epoch 50/61
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0368 Acc: 0.5701

Epoch 51/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0368 Acc: 0.5701

Epoch 52/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0292 Acc: 0.5701

Epoch 53/61
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0307 Acc: 0.5794

Epoch 54/61
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0257 Acc: 0.5701

Epoch 55/61
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0369 Acc: 0.5794

Epoch 56/61
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0338 Acc: 0.5794

Epoch 57/61
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0404 Acc: 0.5701

Epoch 58/61
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0279 Acc: 0.5701

Epoch 59/61
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0365 Acc: 0.5701

Epoch 60/61
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0196 Acc: 0.5701

Epoch 61/61
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0215 Acc: 0.5701

Training complete in 6m 11s
Best val Acc: 0.588785

---Testing---
Test accuracy: 0.877323
--------------------
Accuracy of Albacore tuna : 89 %
Accuracy of Atlantic bluefin tuna : 89 %
Accuracy of Bigeye tuna : 74 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 90 %
Accuracy of Mackerel tuna : 78 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 59 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.8542214699751353, std: 0.10151237141095497
--------------------

run info[val: 0.15, epoch: 66, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/65
----------
LR is set to 0.01
train Loss: 0.0291 Acc: 0.1617
val Loss: 0.0293 Acc: 0.2422

Epoch 1/65
----------
train Loss: 0.0240 Acc: 0.3126
val Loss: 0.0258 Acc: 0.3851

Epoch 2/65
----------
train Loss: 0.0204 Acc: 0.3902
val Loss: 0.0251 Acc: 0.3043

Epoch 3/65
----------
train Loss: 0.0184 Acc: 0.4656
val Loss: 0.0263 Acc: 0.3665

Epoch 4/65
----------
train Loss: 0.0171 Acc: 0.5202
val Loss: 0.0242 Acc: 0.3851

Epoch 5/65
----------
train Loss: 0.0156 Acc: 0.5443
val Loss: 0.0245 Acc: 0.3851

Epoch 6/65
----------
LR is set to 0.001
train Loss: 0.0143 Acc: 0.5749
val Loss: 0.0223 Acc: 0.4286

Epoch 7/65
----------
train Loss: 0.0138 Acc: 0.6044
val Loss: 0.0224 Acc: 0.4472

Epoch 8/65
----------
train Loss: 0.0139 Acc: 0.6284
val Loss: 0.0223 Acc: 0.4472

Epoch 9/65
----------
train Loss: 0.0138 Acc: 0.6175
val Loss: 0.0219 Acc: 0.4348

Epoch 10/65
----------
train Loss: 0.0135 Acc: 0.6197
val Loss: 0.0223 Acc: 0.4472

Epoch 11/65
----------
train Loss: 0.0129 Acc: 0.6470
val Loss: 0.0222 Acc: 0.4410

Epoch 12/65
----------
LR is set to 0.00010000000000000002
train Loss: 0.0131 Acc: 0.6361
val Loss: 0.0222 Acc: 0.4534

Epoch 13/65
----------
train Loss: 0.0129 Acc: 0.6339
val Loss: 0.0228 Acc: 0.4472

Epoch 14/65
----------
train Loss: 0.0139 Acc: 0.6197
val Loss: 0.0221 Acc: 0.4472

Epoch 15/65
----------
train Loss: 0.0136 Acc: 0.6393
val Loss: 0.0220 Acc: 0.4472

Epoch 16/65
----------
train Loss: 0.0134 Acc: 0.6383
val Loss: 0.0220 Acc: 0.4348

Epoch 17/65
----------
train Loss: 0.0133 Acc: 0.6393
val Loss: 0.0221 Acc: 0.4410

Epoch 18/65
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0130 Acc: 0.6317
val Loss: 0.0216 Acc: 0.4410

Epoch 19/65
----------
train Loss: 0.0134 Acc: 0.6219
val Loss: 0.0215 Acc: 0.4348

Epoch 20/65
----------
train Loss: 0.0133 Acc: 0.6284
val Loss: 0.0215 Acc: 0.4286

Epoch 21/65
----------
train Loss: 0.0132 Acc: 0.6339
val Loss: 0.0220 Acc: 0.4410

Epoch 22/65
----------
train Loss: 0.0130 Acc: 0.6273
val Loss: 0.0216 Acc: 0.4410

Epoch 23/65
----------
train Loss: 0.0132 Acc: 0.6328
val Loss: 0.0215 Acc: 0.4472

Epoch 24/65
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0133 Acc: 0.6317
val Loss: 0.0218 Acc: 0.4472

Epoch 25/65
----------
train Loss: 0.0132 Acc: 0.6546
val Loss: 0.0216 Acc: 0.4410

Epoch 26/65
----------
train Loss: 0.0131 Acc: 0.6437
val Loss: 0.0214 Acc: 0.4410

Epoch 27/65
----------
train Loss: 0.0131 Acc: 0.6459
val Loss: 0.0220 Acc: 0.4348

Epoch 28/65
----------
train Loss: 0.0134 Acc: 0.6295
val Loss: 0.0217 Acc: 0.4410

Epoch 29/65
----------
train Loss: 0.0130 Acc: 0.6393
val Loss: 0.0222 Acc: 0.4348

Epoch 30/65
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0133 Acc: 0.6536
val Loss: 0.0220 Acc: 0.4410

Epoch 31/65
----------
train Loss: 0.0132 Acc: 0.6481
val Loss: 0.0217 Acc: 0.4286

Epoch 32/65
----------
train Loss: 0.0131 Acc: 0.6273
val Loss: 0.0222 Acc: 0.4286

Epoch 33/65
----------
train Loss: 0.0135 Acc: 0.6590
val Loss: 0.0215 Acc: 0.4472

Epoch 34/65
----------
train Loss: 0.0127 Acc: 0.6579
val Loss: 0.0219 Acc: 0.4286

Epoch 35/65
----------
train Loss: 0.0134 Acc: 0.6219
val Loss: 0.0219 Acc: 0.4472

Epoch 36/65
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0131 Acc: 0.6383
val Loss: 0.0218 Acc: 0.4348

Epoch 37/65
----------
train Loss: 0.0130 Acc: 0.6404
val Loss: 0.0213 Acc: 0.4410

Epoch 38/65
----------
train Loss: 0.0134 Acc: 0.6404
val Loss: 0.0216 Acc: 0.4410

Epoch 39/65
----------
train Loss: 0.0133 Acc: 0.6415
val Loss: 0.0221 Acc: 0.4410

Epoch 40/65
----------
train Loss: 0.0131 Acc: 0.6568
val Loss: 0.0223 Acc: 0.4348

Epoch 41/65
----------
train Loss: 0.0129 Acc: 0.6295
val Loss: 0.0219 Acc: 0.4410

Epoch 42/65
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0133 Acc: 0.6492
val Loss: 0.0221 Acc: 0.4534

Epoch 43/65
----------
train Loss: 0.0130 Acc: 0.6251
val Loss: 0.0218 Acc: 0.4348

Epoch 44/65
----------
train Loss: 0.0129 Acc: 0.6393
val Loss: 0.0220 Acc: 0.4348

Epoch 45/65
----------
train Loss: 0.0127 Acc: 0.6590
val Loss: 0.0217 Acc: 0.4410

Epoch 46/65
----------
train Loss: 0.0130 Acc: 0.6437
val Loss: 0.0224 Acc: 0.4410

Epoch 47/65
----------
train Loss: 0.0131 Acc: 0.6481
val Loss: 0.0214 Acc: 0.4348

Epoch 48/65
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0131 Acc: 0.6284
val Loss: 0.0214 Acc: 0.4286

Epoch 49/65
----------
train Loss: 0.0131 Acc: 0.6415
val Loss: 0.0225 Acc: 0.4348

Epoch 50/65
----------
train Loss: 0.0133 Acc: 0.6383
val Loss: 0.0220 Acc: 0.4472

Epoch 51/65
----------
train Loss: 0.0130 Acc: 0.6361
val Loss: 0.0217 Acc: 0.4472

Epoch 52/65
----------
train Loss: 0.0130 Acc: 0.6306
val Loss: 0.0217 Acc: 0.4472

Epoch 53/65
----------
train Loss: 0.0129 Acc: 0.6470
val Loss: 0.0216 Acc: 0.4410

Epoch 54/65
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0131 Acc: 0.6503
val Loss: 0.0218 Acc: 0.4410

Epoch 55/65
----------
train Loss: 0.0131 Acc: 0.6372
val Loss: 0.0217 Acc: 0.4472

Epoch 56/65
----------
train Loss: 0.0133 Acc: 0.6383
val Loss: 0.0219 Acc: 0.4348

Epoch 57/65
----------
train Loss: 0.0132 Acc: 0.6339
val Loss: 0.0219 Acc: 0.4348

Epoch 58/65
----------
train Loss: 0.0130 Acc: 0.6404
val Loss: 0.0218 Acc: 0.4348

Epoch 59/65
----------
train Loss: 0.0128 Acc: 0.6481
val Loss: 0.0219 Acc: 0.4348

Epoch 60/65
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0133 Acc: 0.6284
val Loss: 0.0210 Acc: 0.4472

Epoch 61/65
----------
train Loss: 0.0136 Acc: 0.6328
val Loss: 0.0215 Acc: 0.4410

Epoch 62/65
----------
train Loss: 0.0133 Acc: 0.6295
val Loss: 0.0225 Acc: 0.4472

Epoch 63/65
----------
train Loss: 0.0131 Acc: 0.6404
val Loss: 0.0219 Acc: 0.4472

Epoch 64/65
----------
train Loss: 0.0132 Acc: 0.6525
val Loss: 0.0215 Acc: 0.4472

Epoch 65/65
----------
train Loss: 0.0130 Acc: 0.6557
val Loss: 0.0222 Acc: 0.4472

Training complete in 6m 6s
Best val Acc: 0.453416

---Fine tuning.---
Epoch 0/65
----------
LR is set to 0.01
train Loss: 0.0131 Acc: 0.6262
val Loss: 0.0219 Acc: 0.4472

Epoch 1/65
----------
train Loss: 0.0097 Acc: 0.7246
val Loss: 0.0203 Acc: 0.5093

Epoch 2/65
----------
train Loss: 0.0078 Acc: 0.7814
val Loss: 0.0234 Acc: 0.5031

Epoch 3/65
----------
train Loss: 0.0055 Acc: 0.8437
val Loss: 0.0236 Acc: 0.4969

Epoch 4/65
----------
train Loss: 0.0040 Acc: 0.8874
val Loss: 0.0242 Acc: 0.4907

Epoch 5/65
----------
train Loss: 0.0030 Acc: 0.9290
val Loss: 0.0238 Acc: 0.4783

Epoch 6/65
----------
LR is set to 0.001
train Loss: 0.0025 Acc: 0.9454
val Loss: 0.0233 Acc: 0.5280

Epoch 7/65
----------
train Loss: 0.0020 Acc: 0.9508
val Loss: 0.0233 Acc: 0.5093

Epoch 8/65
----------
train Loss: 0.0019 Acc: 0.9607
val Loss: 0.0220 Acc: 0.5093

Epoch 9/65
----------
train Loss: 0.0018 Acc: 0.9705
val Loss: 0.0215 Acc: 0.5217

Epoch 10/65
----------
train Loss: 0.0014 Acc: 0.9661
val Loss: 0.0207 Acc: 0.5217

Epoch 11/65
----------
train Loss: 0.0014 Acc: 0.9705
val Loss: 0.0228 Acc: 0.5217

Epoch 12/65
----------
LR is set to 0.00010000000000000002
train Loss: 0.0013 Acc: 0.9683
val Loss: 0.0228 Acc: 0.5031

Epoch 13/65
----------
train Loss: 0.0018 Acc: 0.9683
val Loss: 0.0211 Acc: 0.5280

Epoch 14/65
----------
train Loss: 0.0015 Acc: 0.9650
val Loss: 0.0214 Acc: 0.5155

Epoch 15/65
----------
train Loss: 0.0013 Acc: 0.9738
val Loss: 0.0216 Acc: 0.5217

Epoch 16/65
----------
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0227 Acc: 0.5155

Epoch 17/65
----------
train Loss: 0.0015 Acc: 0.9694
val Loss: 0.0223 Acc: 0.5031

Epoch 18/65
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0013 Acc: 0.9727
val Loss: 0.0225 Acc: 0.5155

Epoch 19/65
----------
train Loss: 0.0016 Acc: 0.9585
val Loss: 0.0218 Acc: 0.5155

Epoch 20/65
----------
train Loss: 0.0014 Acc: 0.9705
val Loss: 0.0223 Acc: 0.5155

Epoch 21/65
----------
train Loss: 0.0014 Acc: 0.9738
val Loss: 0.0226 Acc: 0.5280

Epoch 22/65
----------
train Loss: 0.0013 Acc: 0.9781
val Loss: 0.0216 Acc: 0.5093

Epoch 23/65
----------
train Loss: 0.0013 Acc: 0.9727
val Loss: 0.0223 Acc: 0.5031

Epoch 24/65
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9738
val Loss: 0.0224 Acc: 0.5031

Epoch 25/65
----------
train Loss: 0.0014 Acc: 0.9694
val Loss: 0.0228 Acc: 0.5280

Epoch 26/65
----------
train Loss: 0.0013 Acc: 0.9760
val Loss: 0.0229 Acc: 0.5217

Epoch 27/65
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0211 Acc: 0.5280

Epoch 28/65
----------
train Loss: 0.0014 Acc: 0.9639
val Loss: 0.0219 Acc: 0.5280

Epoch 29/65
----------
train Loss: 0.0017 Acc: 0.9607
val Loss: 0.0214 Acc: 0.5342

Epoch 30/65
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0015 Acc: 0.9781
val Loss: 0.0219 Acc: 0.5031

Epoch 31/65
----------
train Loss: 0.0015 Acc: 0.9705
val Loss: 0.0217 Acc: 0.5155

Epoch 32/65
----------
train Loss: 0.0013 Acc: 0.9749
val Loss: 0.0224 Acc: 0.5280

Epoch 33/65
----------
train Loss: 0.0012 Acc: 0.9727
val Loss: 0.0224 Acc: 0.5155

Epoch 34/65
----------
train Loss: 0.0015 Acc: 0.9683
val Loss: 0.0224 Acc: 0.5031

Epoch 35/65
----------
train Loss: 0.0014 Acc: 0.9650
val Loss: 0.0217 Acc: 0.5155

Epoch 36/65
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0013 Acc: 0.9683
val Loss: 0.0216 Acc: 0.5093

Epoch 37/65
----------
train Loss: 0.0013 Acc: 0.9694
val Loss: 0.0223 Acc: 0.5217

Epoch 38/65
----------
train Loss: 0.0013 Acc: 0.9770
val Loss: 0.0220 Acc: 0.5217

Epoch 39/65
----------
train Loss: 0.0017 Acc: 0.9683
val Loss: 0.0222 Acc: 0.5217

Epoch 40/65
----------
train Loss: 0.0015 Acc: 0.9716
val Loss: 0.0227 Acc: 0.5093

Epoch 41/65
----------
train Loss: 0.0014 Acc: 0.9683
val Loss: 0.0219 Acc: 0.4969

Epoch 42/65
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0221 Acc: 0.5093

Epoch 43/65
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0223 Acc: 0.5280

Epoch 44/65
----------
train Loss: 0.0014 Acc: 0.9716
val Loss: 0.0220 Acc: 0.5280

Epoch 45/65
----------
train Loss: 0.0015 Acc: 0.9672
val Loss: 0.0218 Acc: 0.5217

Epoch 46/65
----------
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0226 Acc: 0.5155

Epoch 47/65
----------
train Loss: 0.0013 Acc: 0.9705
val Loss: 0.0226 Acc: 0.5217

Epoch 48/65
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9705
val Loss: 0.0220 Acc: 0.5217

Epoch 49/65
----------
train Loss: 0.0014 Acc: 0.9683
val Loss: 0.0221 Acc: 0.5093

Epoch 50/65
----------
train Loss: 0.0015 Acc: 0.9705
val Loss: 0.0221 Acc: 0.5155

Epoch 51/65
----------
train Loss: 0.0014 Acc: 0.9628
val Loss: 0.0223 Acc: 0.5093

Epoch 52/65
----------
train Loss: 0.0012 Acc: 0.9683
val Loss: 0.0229 Acc: 0.5217

Epoch 53/65
----------
train Loss: 0.0015 Acc: 0.9683
val Loss: 0.0223 Acc: 0.5280

Epoch 54/65
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0219 Acc: 0.5280

Epoch 55/65
----------
train Loss: 0.0013 Acc: 0.9716
val Loss: 0.0225 Acc: 0.5155

Epoch 56/65
----------
train Loss: 0.0014 Acc: 0.9694
val Loss: 0.0223 Acc: 0.5217

Epoch 57/65
----------
train Loss: 0.0014 Acc: 0.9760
val Loss: 0.0230 Acc: 0.5155

Epoch 58/65
----------
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0232 Acc: 0.5217

Epoch 59/65
----------
train Loss: 0.0013 Acc: 0.9650
val Loss: 0.0227 Acc: 0.5093

Epoch 60/65
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0012 Acc: 0.9716
val Loss: 0.0213 Acc: 0.5093

Epoch 61/65
----------
train Loss: 0.0014 Acc: 0.9672
val Loss: 0.0213 Acc: 0.5093

Epoch 62/65
----------
train Loss: 0.0013 Acc: 0.9716
val Loss: 0.0224 Acc: 0.5155

Epoch 63/65
----------
train Loss: 0.0012 Acc: 0.9792
val Loss: 0.0209 Acc: 0.5031

Epoch 64/65
----------
train Loss: 0.0012 Acc: 0.9738
val Loss: 0.0218 Acc: 0.5031

Epoch 65/65
----------
train Loss: 0.0012 Acc: 0.9738
val Loss: 0.0226 Acc: 0.5093

Training complete in 6m 29s
Best val Acc: 0.534161

---Testing---
Test accuracy: 0.912639
--------------------
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 84 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8851594252730083, std: 0.07239385064452217
--------------------

run info[val: 0.2, epoch: 91, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/90
----------
LR is set to 0.01
train Loss: 0.0270 Acc: 0.1510
val Loss: 0.0321 Acc: 0.2326

Epoch 1/90
----------
train Loss: 0.0217 Acc: 0.3322
val Loss: 0.0289 Acc: 0.3488

Epoch 2/90
----------
train Loss: 0.0188 Acc: 0.4402
val Loss: 0.0258 Acc: 0.3907

Epoch 3/90
----------
train Loss: 0.0169 Acc: 0.4971
val Loss: 0.0255 Acc: 0.4000

Epoch 4/90
----------
train Loss: 0.0153 Acc: 0.5343
val Loss: 0.0249 Acc: 0.4233

Epoch 5/90
----------
train Loss: 0.0144 Acc: 0.5668
val Loss: 0.0255 Acc: 0.3953

Epoch 6/90
----------
train Loss: 0.0140 Acc: 0.5807
val Loss: 0.0232 Acc: 0.4279

Epoch 7/90
----------
train Loss: 0.0132 Acc: 0.5970
val Loss: 0.0209 Acc: 0.5163

Epoch 8/90
----------
train Loss: 0.0126 Acc: 0.6341
val Loss: 0.0258 Acc: 0.4372

Epoch 9/90
----------
train Loss: 0.0121 Acc: 0.6307
val Loss: 0.0231 Acc: 0.4605

Epoch 10/90
----------
train Loss: 0.0114 Acc: 0.6620
val Loss: 0.0214 Acc: 0.4651

Epoch 11/90
----------
LR is set to 0.001
train Loss: 0.0111 Acc: 0.6597
val Loss: 0.0241 Acc: 0.4605

Epoch 12/90
----------
train Loss: 0.0105 Acc: 0.7108
val Loss: 0.0219 Acc: 0.4930

Epoch 13/90
----------
train Loss: 0.0106 Acc: 0.6899
val Loss: 0.0228 Acc: 0.4977

Epoch 14/90
----------
train Loss: 0.0104 Acc: 0.7178
val Loss: 0.0252 Acc: 0.4837

Epoch 15/90
----------
train Loss: 0.0105 Acc: 0.7038
val Loss: 0.0218 Acc: 0.4837

Epoch 16/90
----------
train Loss: 0.0103 Acc: 0.7062
val Loss: 0.0237 Acc: 0.4837

Epoch 17/90
----------
train Loss: 0.0105 Acc: 0.7154
val Loss: 0.0228 Acc: 0.4977

Epoch 18/90
----------
train Loss: 0.0106 Acc: 0.6945
val Loss: 0.0242 Acc: 0.4884

Epoch 19/90
----------
train Loss: 0.0104 Acc: 0.7352
val Loss: 0.0271 Acc: 0.4884

Epoch 20/90
----------
train Loss: 0.0105 Acc: 0.6992
val Loss: 0.0231 Acc: 0.4837

Epoch 21/90
----------
train Loss: 0.0103 Acc: 0.7073
val Loss: 0.0226 Acc: 0.4791

Epoch 22/90
----------
LR is set to 0.00010000000000000002
train Loss: 0.0102 Acc: 0.7131
val Loss: 0.0248 Acc: 0.4884

Epoch 23/90
----------
train Loss: 0.0102 Acc: 0.7294
val Loss: 0.0235 Acc: 0.4884

Epoch 24/90
----------
train Loss: 0.0101 Acc: 0.7271
val Loss: 0.0225 Acc: 0.4837

Epoch 25/90
----------
train Loss: 0.0103 Acc: 0.7201
val Loss: 0.0249 Acc: 0.4884

Epoch 26/90
----------
train Loss: 0.0103 Acc: 0.6980
val Loss: 0.0218 Acc: 0.4930

Epoch 27/90
----------
train Loss: 0.0103 Acc: 0.7189
val Loss: 0.0234 Acc: 0.4930

Epoch 28/90
----------
train Loss: 0.0101 Acc: 0.7062
val Loss: 0.0230 Acc: 0.4884

Epoch 29/90
----------
train Loss: 0.0101 Acc: 0.7178
val Loss: 0.0252 Acc: 0.4884

Epoch 30/90
----------
train Loss: 0.0104 Acc: 0.7108
val Loss: 0.0237 Acc: 0.4884

Epoch 31/90
----------
train Loss: 0.0102 Acc: 0.7340
val Loss: 0.0235 Acc: 0.4884

Epoch 32/90
----------
train Loss: 0.0103 Acc: 0.7050
val Loss: 0.0272 Acc: 0.4837

Epoch 33/90
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0101 Acc: 0.7201
val Loss: 0.0248 Acc: 0.4884

Epoch 34/90
----------
train Loss: 0.0102 Acc: 0.7096
val Loss: 0.0235 Acc: 0.4791

Epoch 35/90
----------
train Loss: 0.0103 Acc: 0.7038
val Loss: 0.0230 Acc: 0.4837

Epoch 36/90
----------
train Loss: 0.0099 Acc: 0.7294
val Loss: 0.0210 Acc: 0.4884

Epoch 37/90
----------
train Loss: 0.0100 Acc: 0.7236
val Loss: 0.0233 Acc: 0.4791

Epoch 38/90
----------
train Loss: 0.0100 Acc: 0.7213
val Loss: 0.0241 Acc: 0.4837

Epoch 39/90
----------
train Loss: 0.0104 Acc: 0.7062
val Loss: 0.0235 Acc: 0.4791

Epoch 40/90
----------
train Loss: 0.0104 Acc: 0.7189
val Loss: 0.0222 Acc: 0.4837

Epoch 41/90
----------
train Loss: 0.0101 Acc: 0.7294
val Loss: 0.0239 Acc: 0.4884

Epoch 42/90
----------
train Loss: 0.0103 Acc: 0.7259
val Loss: 0.0226 Acc: 0.4977

Epoch 43/90
----------
train Loss: 0.0100 Acc: 0.7271
val Loss: 0.0215 Acc: 0.4884

Epoch 44/90
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0101 Acc: 0.7166
val Loss: 0.0227 Acc: 0.4837

Epoch 45/90
----------
train Loss: 0.0107 Acc: 0.6992
val Loss: 0.0229 Acc: 0.4837

Epoch 46/90
----------
train Loss: 0.0101 Acc: 0.7178
val Loss: 0.0228 Acc: 0.4884

Epoch 47/90
----------
train Loss: 0.0100 Acc: 0.7166
val Loss: 0.0227 Acc: 0.4837

Epoch 48/90
----------
train Loss: 0.0102 Acc: 0.7085
val Loss: 0.0227 Acc: 0.4884

Epoch 49/90
----------
train Loss: 0.0099 Acc: 0.7329
val Loss: 0.0215 Acc: 0.4884

Epoch 50/90
----------
train Loss: 0.0101 Acc: 0.7224
val Loss: 0.0261 Acc: 0.4791

Epoch 51/90
----------
train Loss: 0.0102 Acc: 0.7154
val Loss: 0.0224 Acc: 0.4884

Epoch 52/90
----------
train Loss: 0.0100 Acc: 0.7410
val Loss: 0.0227 Acc: 0.4884

Epoch 53/90
----------
train Loss: 0.0102 Acc: 0.7352
val Loss: 0.0223 Acc: 0.4884

Epoch 54/90
----------
train Loss: 0.0103 Acc: 0.7166
val Loss: 0.0221 Acc: 0.4930

Epoch 55/90
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0102 Acc: 0.7120
val Loss: 0.0228 Acc: 0.4930

Epoch 56/90
----------
train Loss: 0.0102 Acc: 0.7282
val Loss: 0.0237 Acc: 0.4930

Epoch 57/90
----------
train Loss: 0.0100 Acc: 0.7422
val Loss: 0.0221 Acc: 0.4837

Epoch 58/90
----------
train Loss: 0.0102 Acc: 0.7201
val Loss: 0.0245 Acc: 0.4884

Epoch 59/90
----------
train Loss: 0.0102 Acc: 0.7213
val Loss: 0.0211 Acc: 0.4837

Epoch 60/90
----------
train Loss: 0.0105 Acc: 0.6969
val Loss: 0.0215 Acc: 0.4884

Epoch 61/90
----------
train Loss: 0.0101 Acc: 0.7154
val Loss: 0.0199 Acc: 0.4884

Epoch 62/90
----------
train Loss: 0.0101 Acc: 0.7329
val Loss: 0.0226 Acc: 0.4791

Epoch 63/90
----------
train Loss: 0.0100 Acc: 0.7247
val Loss: 0.0221 Acc: 0.4837

Epoch 64/90
----------
train Loss: 0.0102 Acc: 0.7247
val Loss: 0.0247 Acc: 0.4930

Epoch 65/90
----------
train Loss: 0.0100 Acc: 0.7282
val Loss: 0.0211 Acc: 0.4837

Epoch 66/90
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0101 Acc: 0.7178
val Loss: 0.0255 Acc: 0.4791

Epoch 67/90
----------
train Loss: 0.0100 Acc: 0.7317
val Loss: 0.0220 Acc: 0.4791

Epoch 68/90
----------
train Loss: 0.0102 Acc: 0.7038
val Loss: 0.0237 Acc: 0.4837

Epoch 69/90
----------
train Loss: 0.0100 Acc: 0.7201
val Loss: 0.0244 Acc: 0.4884

Epoch 70/90
----------
train Loss: 0.0102 Acc: 0.7131
val Loss: 0.0266 Acc: 0.4884

Epoch 71/90
----------
train Loss: 0.0102 Acc: 0.7247
val Loss: 0.0248 Acc: 0.4884

Epoch 72/90
----------
train Loss: 0.0101 Acc: 0.7305
val Loss: 0.0231 Acc: 0.4837

Epoch 73/90
----------
train Loss: 0.0101 Acc: 0.7166
val Loss: 0.0259 Acc: 0.4884

Epoch 74/90
----------
train Loss: 0.0100 Acc: 0.7398
val Loss: 0.0229 Acc: 0.4837

Epoch 75/90
----------
train Loss: 0.0104 Acc: 0.7178
val Loss: 0.0198 Acc: 0.4930

Epoch 76/90
----------
train Loss: 0.0101 Acc: 0.7236
val Loss: 0.0216 Acc: 0.4884

Epoch 77/90
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0103 Acc: 0.7166
val Loss: 0.0248 Acc: 0.4884

Epoch 78/90
----------
train Loss: 0.0102 Acc: 0.7096
val Loss: 0.0253 Acc: 0.4930

Epoch 79/90
----------
train Loss: 0.0100 Acc: 0.7236
val Loss: 0.0231 Acc: 0.4837

Epoch 80/90
----------
train Loss: 0.0101 Acc: 0.7247
val Loss: 0.0212 Acc: 0.4837

Epoch 81/90
----------
train Loss: 0.0101 Acc: 0.7189
val Loss: 0.0209 Acc: 0.4884

Epoch 82/90
----------
train Loss: 0.0100 Acc: 0.7271
val Loss: 0.0213 Acc: 0.4884

Epoch 83/90
----------
train Loss: 0.0101 Acc: 0.7422
val Loss: 0.0221 Acc: 0.4884

Epoch 84/90
----------
train Loss: 0.0103 Acc: 0.7247
val Loss: 0.0229 Acc: 0.4837

Epoch 85/90
----------
train Loss: 0.0103 Acc: 0.6980
val Loss: 0.0229 Acc: 0.4837

Epoch 86/90
----------
train Loss: 0.0102 Acc: 0.7120
val Loss: 0.0222 Acc: 0.4837

Epoch 87/90
----------
train Loss: 0.0103 Acc: 0.7178
val Loss: 0.0238 Acc: 0.4930

Epoch 88/90
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0104 Acc: 0.7062
val Loss: 0.0229 Acc: 0.4884

Epoch 89/90
----------
train Loss: 0.0102 Acc: 0.7166
val Loss: 0.0231 Acc: 0.4884

Epoch 90/90
----------
train Loss: 0.0100 Acc: 0.7271
val Loss: 0.0218 Acc: 0.4837

Training complete in 8m 4s
Best val Acc: 0.516279

---Fine tuning.---
Epoch 0/90
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6283
val Loss: 0.0244 Acc: 0.4837

Epoch 1/90
----------
train Loss: 0.0092 Acc: 0.7213
val Loss: 0.0220 Acc: 0.5163

Epoch 2/90
----------
train Loss: 0.0061 Acc: 0.8316
val Loss: 0.0204 Acc: 0.5349

Epoch 3/90
----------
train Loss: 0.0043 Acc: 0.8862
val Loss: 0.0202 Acc: 0.5535

Epoch 4/90
----------
train Loss: 0.0027 Acc: 0.9396
val Loss: 0.0202 Acc: 0.5860

Epoch 5/90
----------
train Loss: 0.0022 Acc: 0.9535
val Loss: 0.0209 Acc: 0.5535

Epoch 6/90
----------
train Loss: 0.0018 Acc: 0.9593
val Loss: 0.0215 Acc: 0.5581

Epoch 7/90
----------
train Loss: 0.0014 Acc: 0.9605
val Loss: 0.0265 Acc: 0.5302

Epoch 8/90
----------
train Loss: 0.0011 Acc: 0.9710
val Loss: 0.0259 Acc: 0.5442

Epoch 9/90
----------
train Loss: 0.0010 Acc: 0.9779
val Loss: 0.0246 Acc: 0.5442

Epoch 10/90
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0267 Acc: 0.5721

Epoch 11/90
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0249 Acc: 0.5674

Epoch 12/90
----------
train Loss: 0.0008 Acc: 0.9756
val Loss: 0.0249 Acc: 0.5628

Epoch 13/90
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0262 Acc: 0.5488

Epoch 14/90
----------
train Loss: 0.0007 Acc: 0.9756
val Loss: 0.0227 Acc: 0.5488

Epoch 15/90
----------
train Loss: 0.0007 Acc: 0.9768
val Loss: 0.0271 Acc: 0.5535

Epoch 16/90
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0226 Acc: 0.5581

Epoch 17/90
----------
train Loss: 0.0007 Acc: 0.9768
val Loss: 0.0255 Acc: 0.5721

Epoch 18/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0254 Acc: 0.5674

Epoch 19/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0252 Acc: 0.5721

Epoch 20/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0213 Acc: 0.5674

Epoch 21/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0232 Acc: 0.5628

Epoch 22/90
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0296 Acc: 0.5628

Epoch 23/90
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0256 Acc: 0.5674

Epoch 24/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0222 Acc: 0.5767

Epoch 25/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0235 Acc: 0.5721

Epoch 26/90
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0247 Acc: 0.5628

Epoch 27/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0248 Acc: 0.5767

Epoch 28/90
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0250 Acc: 0.5581

Epoch 29/90
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0227 Acc: 0.5721

Epoch 30/90
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0211 Acc: 0.5721

Epoch 31/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0255 Acc: 0.5721

Epoch 32/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0234 Acc: 0.5628

Epoch 33/90
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0233 Acc: 0.5721

Epoch 34/90
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0215 Acc: 0.5628

Epoch 35/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0210 Acc: 0.5674

Epoch 36/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0212 Acc: 0.5628

Epoch 37/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0241 Acc: 0.5674

Epoch 38/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0263 Acc: 0.5674

Epoch 39/90
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0277 Acc: 0.5628

Epoch 40/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0256 Acc: 0.5674

Epoch 41/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0249 Acc: 0.5628

Epoch 42/90
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0222 Acc: 0.5581

Epoch 43/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0282 Acc: 0.5721

Epoch 44/90
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0206 Acc: 0.5721

Epoch 45/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0223 Acc: 0.5628

Epoch 46/90
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0197 Acc: 0.5628

Epoch 47/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0210 Acc: 0.5628

Epoch 48/90
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0237 Acc: 0.5674

Epoch 49/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0244 Acc: 0.5581

Epoch 50/90
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0232 Acc: 0.5581

Epoch 51/90
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0202 Acc: 0.5721

Epoch 52/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0207 Acc: 0.5581

Epoch 53/90
----------
train Loss: 0.0006 Acc: 0.9907
val Loss: 0.0218 Acc: 0.5814

Epoch 54/90
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0267 Acc: 0.5628

Epoch 55/90
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0286 Acc: 0.5581

Epoch 56/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0226 Acc: 0.5581

Epoch 57/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0223 Acc: 0.5628

Epoch 58/90
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0228 Acc: 0.5535

Epoch 59/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0260 Acc: 0.5581

Epoch 60/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5721

Epoch 61/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0288 Acc: 0.5581

Epoch 62/90
----------
train Loss: 0.0006 Acc: 0.9744
val Loss: 0.0233 Acc: 0.5628

Epoch 63/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0221 Acc: 0.5535

Epoch 64/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5581

Epoch 65/90
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0287 Acc: 0.5674

Epoch 66/90
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0238 Acc: 0.5674

Epoch 67/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0247 Acc: 0.5767

Epoch 68/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0235 Acc: 0.5628

Epoch 69/90
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0244 Acc: 0.5628

Epoch 70/90
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0259 Acc: 0.5628

Epoch 71/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0205 Acc: 0.5674

Epoch 72/90
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0209 Acc: 0.5628

Epoch 73/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0216 Acc: 0.5581

Epoch 74/90
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0272 Acc: 0.5581

Epoch 75/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0232 Acc: 0.5814

Epoch 76/90
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5581

Epoch 77/90
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0224 Acc: 0.5488

Epoch 78/90
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0211 Acc: 0.5581

Epoch 79/90
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0233 Acc: 0.5581

Epoch 80/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0223 Acc: 0.5535

Epoch 81/90
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0243 Acc: 0.5674

Epoch 82/90
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0231 Acc: 0.5721

Epoch 83/90
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0192 Acc: 0.5674

Epoch 84/90
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0244 Acc: 0.5674

Epoch 85/90
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0211 Acc: 0.5674

Epoch 86/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0237 Acc: 0.5674

Epoch 87/90
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0206 Acc: 0.5628

Epoch 88/90
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0216 Acc: 0.5674

Epoch 89/90
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0226 Acc: 0.5721

Epoch 90/90
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0229 Acc: 0.5674

Training complete in 8m 36s
Best val Acc: 0.586047

---Testing---
Test accuracy: 0.888476
--------------------
Accuracy of Albacore tuna : 89 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 78 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 79 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.84642503084099, std: 0.10881981092322239
--------------------

run info[val: 0.25, epoch: 55, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0290 Acc: 0.1462
val Loss: 0.0274 Acc: 0.1896

Epoch 1/54
----------
train Loss: 0.0256 Acc: 0.2627
val Loss: 0.0265 Acc: 0.2751

Epoch 2/54
----------
train Loss: 0.0229 Acc: 0.3482
val Loss: 0.0243 Acc: 0.3309

Epoch 3/54
----------
train Loss: 0.0213 Acc: 0.4188
val Loss: 0.0223 Acc: 0.3792

Epoch 4/54
----------
train Loss: 0.0180 Acc: 0.4511
val Loss: 0.0208 Acc: 0.4275

Epoch 5/54
----------
train Loss: 0.0163 Acc: 0.5118
val Loss: 0.0220 Acc: 0.3792

Epoch 6/54
----------
train Loss: 0.0175 Acc: 0.5019
val Loss: 0.0201 Acc: 0.4572

Epoch 7/54
----------
train Loss: 0.0159 Acc: 0.5638
val Loss: 0.0199 Acc: 0.4424

Epoch 8/54
----------
train Loss: 0.0151 Acc: 0.5799
val Loss: 0.0197 Acc: 0.4758

Epoch 9/54
----------
train Loss: 0.0147 Acc: 0.5948
val Loss: 0.0196 Acc: 0.4610

Epoch 10/54
----------
train Loss: 0.0131 Acc: 0.6084
val Loss: 0.0200 Acc: 0.4461

Epoch 11/54
----------
train Loss: 0.0128 Acc: 0.6121
val Loss: 0.0207 Acc: 0.4498

Epoch 12/54
----------
train Loss: 0.0132 Acc: 0.6233
val Loss: 0.0202 Acc: 0.4647

Epoch 13/54
----------
LR is set to 0.001
train Loss: 0.0125 Acc: 0.6580
val Loss: 0.0193 Acc: 0.4796

Epoch 14/54
----------
train Loss: 0.0116 Acc: 0.6927
val Loss: 0.0186 Acc: 0.4944

Epoch 15/54
----------
train Loss: 0.0111 Acc: 0.7051
val Loss: 0.0188 Acc: 0.4796

Epoch 16/54
----------
train Loss: 0.0112 Acc: 0.7038
val Loss: 0.0186 Acc: 0.4796

Epoch 17/54
----------
train Loss: 0.0109 Acc: 0.7038
val Loss: 0.0188 Acc: 0.4833

Epoch 18/54
----------
train Loss: 0.0109 Acc: 0.6976
val Loss: 0.0187 Acc: 0.4758

Epoch 19/54
----------
train Loss: 0.0104 Acc: 0.7113
val Loss: 0.0184 Acc: 0.4833

Epoch 20/54
----------
train Loss: 0.0110 Acc: 0.7138
val Loss: 0.0184 Acc: 0.4944

Epoch 21/54
----------
train Loss: 0.0103 Acc: 0.7150
val Loss: 0.0189 Acc: 0.4870

Epoch 22/54
----------
train Loss: 0.0105 Acc: 0.7212
val Loss: 0.0188 Acc: 0.4684

Epoch 23/54
----------
train Loss: 0.0103 Acc: 0.7212
val Loss: 0.0185 Acc: 0.4796

Epoch 24/54
----------
train Loss: 0.0104 Acc: 0.7398
val Loss: 0.0186 Acc: 0.4833

Epoch 25/54
----------
train Loss: 0.0104 Acc: 0.7323
val Loss: 0.0185 Acc: 0.4610

Epoch 26/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0111 Acc: 0.7200
val Loss: 0.0182 Acc: 0.4647

Epoch 27/54
----------
train Loss: 0.0105 Acc: 0.7299
val Loss: 0.0183 Acc: 0.4833

Epoch 28/54
----------
train Loss: 0.0104 Acc: 0.7212
val Loss: 0.0187 Acc: 0.4758

Epoch 29/54
----------
train Loss: 0.0108 Acc: 0.7125
val Loss: 0.0187 Acc: 0.4684

Epoch 30/54
----------
train Loss: 0.0102 Acc: 0.7385
val Loss: 0.0185 Acc: 0.4684

Epoch 31/54
----------
train Loss: 0.0109 Acc: 0.7175
val Loss: 0.0183 Acc: 0.4833

Epoch 32/54
----------
train Loss: 0.0103 Acc: 0.7311
val Loss: 0.0188 Acc: 0.4833

Epoch 33/54
----------
train Loss: 0.0115 Acc: 0.7323
val Loss: 0.0187 Acc: 0.4870

Epoch 34/54
----------
train Loss: 0.0098 Acc: 0.7336
val Loss: 0.0185 Acc: 0.4833

Epoch 35/54
----------
train Loss: 0.0107 Acc: 0.7323
val Loss: 0.0187 Acc: 0.4796

Epoch 36/54
----------
train Loss: 0.0112 Acc: 0.7237
val Loss: 0.0183 Acc: 0.4758

Epoch 37/54
----------
train Loss: 0.0101 Acc: 0.7249
val Loss: 0.0185 Acc: 0.4833

Epoch 38/54
----------
train Loss: 0.0098 Acc: 0.7472
val Loss: 0.0190 Acc: 0.4721

Epoch 39/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0099 Acc: 0.7150
val Loss: 0.0185 Acc: 0.4870

Epoch 40/54
----------
train Loss: 0.0117 Acc: 0.7286
val Loss: 0.0184 Acc: 0.4796

Epoch 41/54
----------
train Loss: 0.0107 Acc: 0.7348
val Loss: 0.0186 Acc: 0.4870

Epoch 42/54
----------
train Loss: 0.0108 Acc: 0.7249
val Loss: 0.0183 Acc: 0.4833

Epoch 43/54
----------
train Loss: 0.0106 Acc: 0.7261
val Loss: 0.0188 Acc: 0.4944

Epoch 44/54
----------
train Loss: 0.0104 Acc: 0.7472
val Loss: 0.0188 Acc: 0.4833

Epoch 45/54
----------
train Loss: 0.0110 Acc: 0.7274
val Loss: 0.0184 Acc: 0.4944

Epoch 46/54
----------
train Loss: 0.0104 Acc: 0.7212
val Loss: 0.0187 Acc: 0.4796

Epoch 47/54
----------
train Loss: 0.0106 Acc: 0.7187
val Loss: 0.0186 Acc: 0.4758

Epoch 48/54
----------
train Loss: 0.0098 Acc: 0.7274
val Loss: 0.0185 Acc: 0.4833

Epoch 49/54
----------
train Loss: 0.0099 Acc: 0.7175
val Loss: 0.0184 Acc: 0.4870

Epoch 50/54
----------
train Loss: 0.0102 Acc: 0.7485
val Loss: 0.0186 Acc: 0.4758

Epoch 51/54
----------
train Loss: 0.0108 Acc: 0.7447
val Loss: 0.0186 Acc: 0.4833

Epoch 52/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0105 Acc: 0.7224
val Loss: 0.0183 Acc: 0.4758

Epoch 53/54
----------
train Loss: 0.0101 Acc: 0.7361
val Loss: 0.0185 Acc: 0.4796

Epoch 54/54
----------
train Loss: 0.0097 Acc: 0.7398
val Loss: 0.0184 Acc: 0.4796

Training complete in 4m 59s
Best val Acc: 0.494424

---Fine tuning.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0119 Acc: 0.6890
val Loss: 0.0196 Acc: 0.4461

Epoch 1/54
----------
train Loss: 0.0106 Acc: 0.6964
val Loss: 0.0238 Acc: 0.4387

Epoch 2/54
----------
train Loss: 0.0091 Acc: 0.7385
val Loss: 0.0263 Acc: 0.3532

Epoch 3/54
----------
train Loss: 0.0092 Acc: 0.7720
val Loss: 0.0245 Acc: 0.4424

Epoch 4/54
----------
train Loss: 0.0097 Acc: 0.7472
val Loss: 0.0345 Acc: 0.3643

Epoch 5/54
----------
train Loss: 0.0092 Acc: 0.7385
val Loss: 0.0266 Acc: 0.4312

Epoch 6/54
----------
train Loss: 0.0086 Acc: 0.7336
val Loss: 0.0290 Acc: 0.3643

Epoch 7/54
----------
train Loss: 0.0070 Acc: 0.8067
val Loss: 0.0322 Acc: 0.4089

Epoch 8/54
----------
train Loss: 0.0057 Acc: 0.8476
val Loss: 0.0242 Acc: 0.4870

Epoch 9/54
----------
train Loss: 0.0055 Acc: 0.8587
val Loss: 0.0280 Acc: 0.4535

Epoch 10/54
----------
train Loss: 0.0096 Acc: 0.8278
val Loss: 0.0281 Acc: 0.4721

Epoch 11/54
----------
train Loss: 0.0099 Acc: 0.7423
val Loss: 0.0296 Acc: 0.4312

Epoch 12/54
----------
train Loss: 0.0102 Acc: 0.7757
val Loss: 0.0319 Acc: 0.4052

Epoch 13/54
----------
LR is set to 0.001
train Loss: 0.0074 Acc: 0.8426
val Loss: 0.0278 Acc: 0.4572

Epoch 14/54
----------
train Loss: 0.0046 Acc: 0.8612
val Loss: 0.0258 Acc: 0.4796

Epoch 15/54
----------
train Loss: 0.0035 Acc: 0.9046
val Loss: 0.0242 Acc: 0.4758

Epoch 16/54
----------
train Loss: 0.0033 Acc: 0.9294
val Loss: 0.0236 Acc: 0.4796

Epoch 17/54
----------
train Loss: 0.0023 Acc: 0.9480
val Loss: 0.0241 Acc: 0.4721

Epoch 18/54
----------
train Loss: 0.0038 Acc: 0.9480
val Loss: 0.0237 Acc: 0.4907

Epoch 19/54
----------
train Loss: 0.0018 Acc: 0.9566
val Loss: 0.0240 Acc: 0.4944

Epoch 20/54
----------
train Loss: 0.0024 Acc: 0.9566
val Loss: 0.0236 Acc: 0.4796

Epoch 21/54
----------
train Loss: 0.0021 Acc: 0.9492
val Loss: 0.0233 Acc: 0.4833

Epoch 22/54
----------
train Loss: 0.0017 Acc: 0.9542
val Loss: 0.0233 Acc: 0.4721

Epoch 23/54
----------
train Loss: 0.0015 Acc: 0.9653
val Loss: 0.0233 Acc: 0.4758

Epoch 24/54
----------
train Loss: 0.0024 Acc: 0.9542
val Loss: 0.0233 Acc: 0.4833

Epoch 25/54
----------
train Loss: 0.0014 Acc: 0.9678
val Loss: 0.0237 Acc: 0.4647

Epoch 26/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0020 Acc: 0.9703
val Loss: 0.0236 Acc: 0.4684

Epoch 27/54
----------
train Loss: 0.0016 Acc: 0.9727
val Loss: 0.0241 Acc: 0.4684

Epoch 28/54
----------
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0236 Acc: 0.4684

Epoch 29/54
----------
train Loss: 0.0022 Acc: 0.9579
val Loss: 0.0236 Acc: 0.4684

Epoch 30/54
----------
train Loss: 0.0013 Acc: 0.9678
val Loss: 0.0236 Acc: 0.4758

Epoch 31/54
----------
train Loss: 0.0012 Acc: 0.9703
val Loss: 0.0235 Acc: 0.4721

Epoch 32/54
----------
train Loss: 0.0011 Acc: 0.9740
val Loss: 0.0240 Acc: 0.4721

Epoch 33/54
----------
train Loss: 0.0017 Acc: 0.9665
val Loss: 0.0241 Acc: 0.4684

Epoch 34/54
----------
train Loss: 0.0016 Acc: 0.9777
val Loss: 0.0231 Acc: 0.4796

Epoch 35/54
----------
train Loss: 0.0013 Acc: 0.9678
val Loss: 0.0241 Acc: 0.4870

Epoch 36/54
----------
train Loss: 0.0016 Acc: 0.9715
val Loss: 0.0241 Acc: 0.4833

Epoch 37/54
----------
train Loss: 0.0014 Acc: 0.9715
val Loss: 0.0238 Acc: 0.4870

Epoch 38/54
----------
train Loss: 0.0019 Acc: 0.9703
val Loss: 0.0235 Acc: 0.4833

Epoch 39/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0012 Acc: 0.9740
val Loss: 0.0238 Acc: 0.4758

Epoch 40/54
----------
train Loss: 0.0013 Acc: 0.9802
val Loss: 0.0233 Acc: 0.4721

Epoch 41/54
----------
train Loss: 0.0013 Acc: 0.9777
val Loss: 0.0236 Acc: 0.4833

Epoch 42/54
----------
train Loss: 0.0021 Acc: 0.9616
val Loss: 0.0241 Acc: 0.4944

Epoch 43/54
----------
train Loss: 0.0011 Acc: 0.9752
val Loss: 0.0240 Acc: 0.4870

Epoch 44/54
----------
train Loss: 0.0021 Acc: 0.9628
val Loss: 0.0241 Acc: 0.4870

Epoch 45/54
----------
train Loss: 0.0013 Acc: 0.9715
val Loss: 0.0239 Acc: 0.4870

Epoch 46/54
----------
train Loss: 0.0022 Acc: 0.9715
val Loss: 0.0239 Acc: 0.4907

Epoch 47/54
----------
train Loss: 0.0014 Acc: 0.9765
val Loss: 0.0243 Acc: 0.4758

Epoch 48/54
----------
train Loss: 0.0012 Acc: 0.9715
val Loss: 0.0243 Acc: 0.4721

Epoch 49/54
----------
train Loss: 0.0012 Acc: 0.9678
val Loss: 0.0236 Acc: 0.4833

Epoch 50/54
----------
train Loss: 0.0015 Acc: 0.9752
val Loss: 0.0237 Acc: 0.4758

Epoch 51/54
----------
train Loss: 0.0014 Acc: 0.9752
val Loss: 0.0239 Acc: 0.4796

Epoch 52/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0014 Acc: 0.9665
val Loss: 0.0235 Acc: 0.4833

Epoch 53/54
----------
train Loss: 0.0015 Acc: 0.9703
val Loss: 0.0240 Acc: 0.4944

Epoch 54/54
----------
train Loss: 0.0016 Acc: 0.9740
val Loss: 0.0235 Acc: 0.4870

Training complete in 5m 20s
Best val Acc: 0.494424

---Testing---
Test accuracy: 0.845725
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 70 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 88 %
Accuracy of Longtail tuna : 88 %
Accuracy of Mackerel tuna : 76 %
Accuracy of Pacific bluefin tuna : 71 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 50 %
Accuracy of Southern bluefin tuna : 69 %
Accuracy of Yellowfin tuna : 92 %
mean: 0.8027653471064334, std: 0.11858786014432904
--------------------

run info[val: 0.3, epoch: 97, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0285 Acc: 0.1379
val Loss: 0.0290 Acc: 0.2391

Epoch 1/96
----------
train Loss: 0.0230 Acc: 0.3329
val Loss: 0.0264 Acc: 0.3416

Epoch 2/96
----------
train Loss: 0.0193 Acc: 0.4324
val Loss: 0.0244 Acc: 0.3230

Epoch 3/96
----------
train Loss: 0.0176 Acc: 0.4867
val Loss: 0.0223 Acc: 0.3851

Epoch 4/96
----------
train Loss: 0.0155 Acc: 0.5517
val Loss: 0.0222 Acc: 0.4379

Epoch 5/96
----------
train Loss: 0.0145 Acc: 0.5822
val Loss: 0.0220 Acc: 0.4565

Epoch 6/96
----------
LR is set to 0.001
train Loss: 0.0137 Acc: 0.6406
val Loss: 0.0214 Acc: 0.4534

Epoch 7/96
----------
train Loss: 0.0133 Acc: 0.6340
val Loss: 0.0214 Acc: 0.4410

Epoch 8/96
----------
train Loss: 0.0132 Acc: 0.6167
val Loss: 0.0225 Acc: 0.4317

Epoch 9/96
----------
train Loss: 0.0133 Acc: 0.6260
val Loss: 0.0221 Acc: 0.4317

Epoch 10/96
----------
train Loss: 0.0132 Acc: 0.6101
val Loss: 0.0216 Acc: 0.4348

Epoch 11/96
----------
train Loss: 0.0131 Acc: 0.6366
val Loss: 0.0206 Acc: 0.4472

Epoch 12/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0127 Acc: 0.6698
val Loss: 0.0215 Acc: 0.4503

Epoch 13/96
----------
train Loss: 0.0130 Acc: 0.6459
val Loss: 0.0210 Acc: 0.4472

Epoch 14/96
----------
train Loss: 0.0129 Acc: 0.6552
val Loss: 0.0200 Acc: 0.4441

Epoch 15/96
----------
train Loss: 0.0130 Acc: 0.6207
val Loss: 0.0201 Acc: 0.4410

Epoch 16/96
----------
train Loss: 0.0129 Acc: 0.6512
val Loss: 0.0214 Acc: 0.4441

Epoch 17/96
----------
train Loss: 0.0131 Acc: 0.6446
val Loss: 0.0212 Acc: 0.4410

Epoch 18/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0131 Acc: 0.6446
val Loss: 0.0211 Acc: 0.4472

Epoch 19/96
----------
train Loss: 0.0130 Acc: 0.6525
val Loss: 0.0208 Acc: 0.4441

Epoch 20/96
----------
train Loss: 0.0132 Acc: 0.6366
val Loss: 0.0209 Acc: 0.4410

Epoch 21/96
----------
train Loss: 0.0130 Acc: 0.6432
val Loss: 0.0210 Acc: 0.4410

Epoch 22/96
----------
train Loss: 0.0129 Acc: 0.6432
val Loss: 0.0207 Acc: 0.4410

Epoch 23/96
----------
train Loss: 0.0128 Acc: 0.6472
val Loss: 0.0200 Acc: 0.4441

Epoch 24/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0131 Acc: 0.6379
val Loss: 0.0209 Acc: 0.4379

Epoch 25/96
----------
train Loss: 0.0128 Acc: 0.6525
val Loss: 0.0203 Acc: 0.4379

Epoch 26/96
----------
train Loss: 0.0128 Acc: 0.6419
val Loss: 0.0206 Acc: 0.4379

Epoch 27/96
----------
train Loss: 0.0131 Acc: 0.6366
val Loss: 0.0210 Acc: 0.4441

Epoch 28/96
----------
train Loss: 0.0131 Acc: 0.6446
val Loss: 0.0206 Acc: 0.4410

Epoch 29/96
----------
train Loss: 0.0129 Acc: 0.6578
val Loss: 0.0219 Acc: 0.4441

Epoch 30/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0128 Acc: 0.6472
val Loss: 0.0213 Acc: 0.4441

Epoch 31/96
----------
train Loss: 0.0131 Acc: 0.6419
val Loss: 0.0215 Acc: 0.4441

Epoch 32/96
----------
train Loss: 0.0132 Acc: 0.6273
val Loss: 0.0214 Acc: 0.4441

Epoch 33/96
----------
train Loss: 0.0130 Acc: 0.6406
val Loss: 0.0210 Acc: 0.4441

Epoch 34/96
----------
train Loss: 0.0130 Acc: 0.6459
val Loss: 0.0218 Acc: 0.4441

Epoch 35/96
----------
train Loss: 0.0127 Acc: 0.6326
val Loss: 0.0214 Acc: 0.4379

Epoch 36/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0131 Acc: 0.6485
val Loss: 0.0201 Acc: 0.4379

Epoch 37/96
----------
train Loss: 0.0130 Acc: 0.6167
val Loss: 0.0197 Acc: 0.4472

Epoch 38/96
----------
train Loss: 0.0130 Acc: 0.6459
val Loss: 0.0210 Acc: 0.4441

Epoch 39/96
----------
train Loss: 0.0128 Acc: 0.6446
val Loss: 0.0211 Acc: 0.4441

Epoch 40/96
----------
train Loss: 0.0131 Acc: 0.6300
val Loss: 0.0216 Acc: 0.4472

Epoch 41/96
----------
train Loss: 0.0133 Acc: 0.6220
val Loss: 0.0200 Acc: 0.4348

Epoch 42/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0132 Acc: 0.6353
val Loss: 0.0212 Acc: 0.4410

Epoch 43/96
----------
train Loss: 0.0129 Acc: 0.6326
val Loss: 0.0212 Acc: 0.4410

Epoch 44/96
----------
train Loss: 0.0130 Acc: 0.6326
val Loss: 0.0203 Acc: 0.4379

Epoch 45/96
----------
train Loss: 0.0132 Acc: 0.6194
val Loss: 0.0220 Acc: 0.4410

Epoch 46/96
----------
train Loss: 0.0132 Acc: 0.6220
val Loss: 0.0208 Acc: 0.4441

Epoch 47/96
----------
train Loss: 0.0130 Acc: 0.6379
val Loss: 0.0209 Acc: 0.4410

Epoch 48/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0127 Acc: 0.6512
val Loss: 0.0210 Acc: 0.4472

Epoch 49/96
----------
train Loss: 0.0129 Acc: 0.6472
val Loss: 0.0208 Acc: 0.4441

Epoch 50/96
----------
train Loss: 0.0128 Acc: 0.6552
val Loss: 0.0207 Acc: 0.4410

Epoch 51/96
----------
train Loss: 0.0130 Acc: 0.6313
val Loss: 0.0212 Acc: 0.4410

Epoch 52/96
----------
train Loss: 0.0127 Acc: 0.6432
val Loss: 0.0204 Acc: 0.4410

Epoch 53/96
----------
train Loss: 0.0131 Acc: 0.6499
val Loss: 0.0210 Acc: 0.4441

Epoch 54/96
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0129 Acc: 0.6525
val Loss: 0.0209 Acc: 0.4472

Epoch 55/96
----------
train Loss: 0.0130 Acc: 0.6592
val Loss: 0.0207 Acc: 0.4441

Epoch 56/96
----------
train Loss: 0.0127 Acc: 0.6538
val Loss: 0.0210 Acc: 0.4441

Epoch 57/96
----------
train Loss: 0.0130 Acc: 0.6592
val Loss: 0.0194 Acc: 0.4441

Epoch 58/96
----------
train Loss: 0.0132 Acc: 0.6366
val Loss: 0.0209 Acc: 0.4441

Epoch 59/96
----------
train Loss: 0.0130 Acc: 0.6446
val Loss: 0.0211 Acc: 0.4379

Epoch 60/96
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0130 Acc: 0.6485
val Loss: 0.0207 Acc: 0.4410

Epoch 61/96
----------
train Loss: 0.0129 Acc: 0.6366
val Loss: 0.0213 Acc: 0.4410

Epoch 62/96
----------
train Loss: 0.0132 Acc: 0.6432
val Loss: 0.0212 Acc: 0.4379

Epoch 63/96
----------
train Loss: 0.0131 Acc: 0.6472
val Loss: 0.0212 Acc: 0.4410

Epoch 64/96
----------
train Loss: 0.0127 Acc: 0.6313
val Loss: 0.0203 Acc: 0.4379

Epoch 65/96
----------
train Loss: 0.0130 Acc: 0.6485
val Loss: 0.0217 Acc: 0.4441

Epoch 66/96
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0131 Acc: 0.6459
val Loss: 0.0208 Acc: 0.4410

Epoch 67/96
----------
train Loss: 0.0129 Acc: 0.6618
val Loss: 0.0203 Acc: 0.4441

Epoch 68/96
----------
train Loss: 0.0130 Acc: 0.6379
val Loss: 0.0204 Acc: 0.4410

Epoch 69/96
----------
train Loss: 0.0129 Acc: 0.6432
val Loss: 0.0208 Acc: 0.4410

Epoch 70/96
----------
train Loss: 0.0131 Acc: 0.6379
val Loss: 0.0213 Acc: 0.4503

Epoch 71/96
----------
train Loss: 0.0129 Acc: 0.6578
val Loss: 0.0206 Acc: 0.4441

Epoch 72/96
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0129 Acc: 0.6446
val Loss: 0.0209 Acc: 0.4441

Epoch 73/96
----------
train Loss: 0.0129 Acc: 0.6472
val Loss: 0.0210 Acc: 0.4472

Epoch 74/96
----------
train Loss: 0.0128 Acc: 0.6605
val Loss: 0.0209 Acc: 0.4441

Epoch 75/96
----------
train Loss: 0.0129 Acc: 0.6552
val Loss: 0.0211 Acc: 0.4441

Epoch 76/96
----------
train Loss: 0.0126 Acc: 0.6565
val Loss: 0.0220 Acc: 0.4410

Epoch 77/96
----------
train Loss: 0.0132 Acc: 0.6379
val Loss: 0.0213 Acc: 0.4441

Epoch 78/96
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0131 Acc: 0.6273
val Loss: 0.0220 Acc: 0.4441

Epoch 79/96
----------
train Loss: 0.0130 Acc: 0.6340
val Loss: 0.0208 Acc: 0.4410

Epoch 80/96
----------
train Loss: 0.0130 Acc: 0.6552
val Loss: 0.0207 Acc: 0.4441

Epoch 81/96
----------
train Loss: 0.0131 Acc: 0.6300
val Loss: 0.0215 Acc: 0.4472

Epoch 82/96
----------
train Loss: 0.0129 Acc: 0.6459
val Loss: 0.0202 Acc: 0.4441

Epoch 83/96
----------
train Loss: 0.0131 Acc: 0.6432
val Loss: 0.0202 Acc: 0.4379

Epoch 84/96
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0128 Acc: 0.6538
val Loss: 0.0207 Acc: 0.4441

Epoch 85/96
----------
train Loss: 0.0131 Acc: 0.6393
val Loss: 0.0209 Acc: 0.4472

Epoch 86/96
----------
train Loss: 0.0132 Acc: 0.6379
val Loss: 0.0220 Acc: 0.4410

Epoch 87/96
----------
train Loss: 0.0128 Acc: 0.6565
val Loss: 0.0208 Acc: 0.4410

Epoch 88/96
----------
train Loss: 0.0131 Acc: 0.6379
val Loss: 0.0217 Acc: 0.4410

Epoch 89/96
----------
train Loss: 0.0129 Acc: 0.6459
val Loss: 0.0204 Acc: 0.4441

Epoch 90/96
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0129 Acc: 0.6419
val Loss: 0.0209 Acc: 0.4379

Epoch 91/96
----------
train Loss: 0.0131 Acc: 0.6180
val Loss: 0.0213 Acc: 0.4410

Epoch 92/96
----------
train Loss: 0.0126 Acc: 0.6472
val Loss: 0.0208 Acc: 0.4441

Epoch 93/96
----------
train Loss: 0.0131 Acc: 0.6393
val Loss: 0.0217 Acc: 0.4379

Epoch 94/96
----------
train Loss: 0.0132 Acc: 0.6406
val Loss: 0.0222 Acc: 0.4441

Epoch 95/96
----------
train Loss: 0.0129 Acc: 0.6512
val Loss: 0.0215 Acc: 0.4441

Epoch 96/96
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0131 Acc: 0.6366
val Loss: 0.0211 Acc: 0.4410

Training complete in 8m 55s
Best val Acc: 0.456522

---Fine tuning.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0141 Acc: 0.5743
val Loss: 0.0199 Acc: 0.4876

Epoch 1/96
----------
train Loss: 0.0102 Acc: 0.7228
val Loss: 0.0248 Acc: 0.4689

Epoch 2/96
----------
train Loss: 0.0073 Acc: 0.7878
val Loss: 0.0228 Acc: 0.4689

Epoch 3/96
----------
train Loss: 0.0044 Acc: 0.8793
val Loss: 0.0204 Acc: 0.5217

Epoch 4/96
----------
train Loss: 0.0034 Acc: 0.9231
val Loss: 0.0184 Acc: 0.5248

Epoch 5/96
----------
train Loss: 0.0024 Acc: 0.9456
val Loss: 0.0199 Acc: 0.5342

Epoch 6/96
----------
LR is set to 0.001
train Loss: 0.0017 Acc: 0.9655
val Loss: 0.0189 Acc: 0.5342

Epoch 7/96
----------
train Loss: 0.0015 Acc: 0.9761
val Loss: 0.0179 Acc: 0.5404

Epoch 8/96
----------
train Loss: 0.0014 Acc: 0.9788
val Loss: 0.0182 Acc: 0.5404

Epoch 9/96
----------
train Loss: 0.0013 Acc: 0.9748
val Loss: 0.0199 Acc: 0.5435

Epoch 10/96
----------
train Loss: 0.0014 Acc: 0.9761
val Loss: 0.0201 Acc: 0.5373

Epoch 11/96
----------
train Loss: 0.0013 Acc: 0.9761
val Loss: 0.0196 Acc: 0.5404

Epoch 12/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0196 Acc: 0.5404

Epoch 13/96
----------
train Loss: 0.0012 Acc: 0.9867
val Loss: 0.0201 Acc: 0.5435

Epoch 14/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0208 Acc: 0.5404

Epoch 15/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0200 Acc: 0.5342

Epoch 16/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0187 Acc: 0.5373

Epoch 17/96
----------
train Loss: 0.0010 Acc: 0.9894
val Loss: 0.0188 Acc: 0.5435

Epoch 18/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0191 Acc: 0.5435

Epoch 19/96
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0191 Acc: 0.5435

Epoch 20/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0202 Acc: 0.5466

Epoch 21/96
----------
train Loss: 0.0012 Acc: 0.9841
val Loss: 0.0196 Acc: 0.5435

Epoch 22/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0202 Acc: 0.5404

Epoch 23/96
----------
train Loss: 0.0012 Acc: 0.9841
val Loss: 0.0192 Acc: 0.5373

Epoch 24/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0204 Acc: 0.5435

Epoch 25/96
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0197 Acc: 0.5435

Epoch 26/96
----------
train Loss: 0.0012 Acc: 0.9788
val Loss: 0.0212 Acc: 0.5435

Epoch 27/96
----------
train Loss: 0.0012 Acc: 0.9854
val Loss: 0.0191 Acc: 0.5404

Epoch 28/96
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0226 Acc: 0.5404

Epoch 29/96
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0201 Acc: 0.5435

Epoch 30/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0207 Acc: 0.5404

Epoch 31/96
----------
train Loss: 0.0010 Acc: 0.9894
val Loss: 0.0203 Acc: 0.5373

Epoch 32/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0195 Acc: 0.5342

Epoch 33/96
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0187 Acc: 0.5373

Epoch 34/96
----------
train Loss: 0.0011 Acc: 0.9775
val Loss: 0.0207 Acc: 0.5404

Epoch 35/96
----------
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0198 Acc: 0.5404

Epoch 36/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0195 Acc: 0.5435

Epoch 37/96
----------
train Loss: 0.0011 Acc: 0.9867
val Loss: 0.0199 Acc: 0.5404

Epoch 38/96
----------
train Loss: 0.0010 Acc: 0.9841
val Loss: 0.0198 Acc: 0.5373

Epoch 39/96
----------
train Loss: 0.0012 Acc: 0.9788
val Loss: 0.0192 Acc: 0.5404

Epoch 40/96
----------
train Loss: 0.0011 Acc: 0.9881
val Loss: 0.0199 Acc: 0.5311

Epoch 41/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0188 Acc: 0.5373

Epoch 42/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0202 Acc: 0.5404

Epoch 43/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5404

Epoch 44/96
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0194 Acc: 0.5404

Epoch 45/96
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0193 Acc: 0.5435

Epoch 46/96
----------
train Loss: 0.0010 Acc: 0.9867
val Loss: 0.0196 Acc: 0.5435

Epoch 47/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0203 Acc: 0.5435

Epoch 48/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0187 Acc: 0.5404

Epoch 49/96
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0192 Acc: 0.5404

Epoch 50/96
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0191 Acc: 0.5404

Epoch 51/96
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0204 Acc: 0.5435

Epoch 52/96
----------
train Loss: 0.0012 Acc: 0.9841
val Loss: 0.0195 Acc: 0.5404

Epoch 53/96
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0203 Acc: 0.5373

Epoch 54/96
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0191 Acc: 0.5435

Epoch 55/96
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0203 Acc: 0.5404

Epoch 56/96
----------
train Loss: 0.0012 Acc: 0.9881
val Loss: 0.0203 Acc: 0.5404

Epoch 57/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0188 Acc: 0.5404

Epoch 58/96
----------
train Loss: 0.0011 Acc: 0.9881
val Loss: 0.0189 Acc: 0.5404

Epoch 59/96
----------
train Loss: 0.0012 Acc: 0.9788
val Loss: 0.0204 Acc: 0.5373

Epoch 60/96
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0195 Acc: 0.5373

Epoch 61/96
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0198 Acc: 0.5404

Epoch 62/96
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0194 Acc: 0.5373

Epoch 63/96
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0188 Acc: 0.5373

Epoch 64/96
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0196 Acc: 0.5404

Epoch 65/96
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0210 Acc: 0.5435

Epoch 66/96
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0190 Acc: 0.5435

Epoch 67/96
----------
train Loss: 0.0011 Acc: 0.9735
val Loss: 0.0190 Acc: 0.5435

Epoch 68/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5404

Epoch 69/96
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0194 Acc: 0.5404

Epoch 70/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0195 Acc: 0.5435

Epoch 71/96
----------
train Loss: 0.0012 Acc: 0.9788
val Loss: 0.0192 Acc: 0.5435

Epoch 72/96
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0012 Acc: 0.9735
val Loss: 0.0188 Acc: 0.5373

Epoch 73/96
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0215 Acc: 0.5404

Epoch 74/96
----------
train Loss: 0.0010 Acc: 0.9907
val Loss: 0.0202 Acc: 0.5404

Epoch 75/96
----------
train Loss: 0.0010 Acc: 0.9867
val Loss: 0.0191 Acc: 0.5404

Epoch 76/96
----------
train Loss: 0.0012 Acc: 0.9894
val Loss: 0.0201 Acc: 0.5404

Epoch 77/96
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0193 Acc: 0.5404

Epoch 78/96
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0216 Acc: 0.5435

Epoch 79/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0209 Acc: 0.5404

Epoch 80/96
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0184 Acc: 0.5373

Epoch 81/96
----------
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5373

Epoch 82/96
----------
train Loss: 0.0011 Acc: 0.9867
val Loss: 0.0191 Acc: 0.5404

Epoch 83/96
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0193 Acc: 0.5435

Epoch 84/96
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0188 Acc: 0.5404

Epoch 85/96
----------
train Loss: 0.0011 Acc: 0.9854
val Loss: 0.0204 Acc: 0.5404

Epoch 86/96
----------
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0207 Acc: 0.5435

Epoch 87/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0192 Acc: 0.5435

Epoch 88/96
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0204 Acc: 0.5404

Epoch 89/96
----------
train Loss: 0.0011 Acc: 0.9788
val Loss: 0.0191 Acc: 0.5404

Epoch 90/96
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0200 Acc: 0.5435

Epoch 91/96
----------
train Loss: 0.0011 Acc: 0.9788
val Loss: 0.0200 Acc: 0.5404

Epoch 92/96
----------
train Loss: 0.0011 Acc: 0.9881
val Loss: 0.0195 Acc: 0.5404

Epoch 93/96
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0194 Acc: 0.5435

Epoch 94/96
----------
train Loss: 0.0011 Acc: 0.9867
val Loss: 0.0207 Acc: 0.5435

Epoch 95/96
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0197 Acc: 0.5373

Epoch 96/96
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0012 Acc: 0.9761
val Loss: 0.0194 Acc: 0.5373

Training complete in 9m 30s
Best val Acc: 0.546584

---Testing---
Test accuracy: 0.856877
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 83 %
Accuracy of Frigate tuna : 72 %
Accuracy of Little tunny : 85 %
Accuracy of Longtail tuna : 93 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 76 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 73 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8054949669827913, std: 0.1290008575548585

Model saved in "./weights/tuna_fish_[0.91]_mean[0.89]_std[0.07].save".
--------------------

run info[val: 0.1, epoch: 87, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0266 Acc: 0.1651
val Loss: 0.0439 Acc: 0.2897

Epoch 1/86
----------
train Loss: 0.0217 Acc: 0.3209
val Loss: 0.0336 Acc: 0.4579

Epoch 2/86
----------
train Loss: 0.0186 Acc: 0.4035
val Loss: 0.0330 Acc: 0.5140

Epoch 3/86
----------
train Loss: 0.0164 Acc: 0.4964
val Loss: 0.0360 Acc: 0.3925

Epoch 4/86
----------
train Loss: 0.0153 Acc: 0.5212
val Loss: 0.0325 Acc: 0.4112

Epoch 5/86
----------
train Loss: 0.0145 Acc: 0.5583
val Loss: 0.0346 Acc: 0.4953

Epoch 6/86
----------
train Loss: 0.0142 Acc: 0.5645
val Loss: 0.0290 Acc: 0.4766

Epoch 7/86
----------
train Loss: 0.0127 Acc: 0.6244
val Loss: 0.0294 Acc: 0.5140

Epoch 8/86
----------
train Loss: 0.0122 Acc: 0.6522
val Loss: 0.0268 Acc: 0.5234

Epoch 9/86
----------
train Loss: 0.0117 Acc: 0.6543
val Loss: 0.0333 Acc: 0.5140

Epoch 10/86
----------
train Loss: 0.0116 Acc: 0.6522
val Loss: 0.0260 Acc: 0.5140

Epoch 11/86
----------
LR is set to 0.001
train Loss: 0.0112 Acc: 0.6801
val Loss: 0.0253 Acc: 0.5514

Epoch 12/86
----------
train Loss: 0.0104 Acc: 0.7100
val Loss: 0.0327 Acc: 0.4766

Epoch 13/86
----------
train Loss: 0.0108 Acc: 0.6883
val Loss: 0.0338 Acc: 0.4860

Epoch 14/86
----------
train Loss: 0.0106 Acc: 0.7007
val Loss: 0.0253 Acc: 0.5140

Epoch 15/86
----------
train Loss: 0.0106 Acc: 0.7100
val Loss: 0.0305 Acc: 0.5047

Epoch 16/86
----------
train Loss: 0.0106 Acc: 0.7079
val Loss: 0.0244 Acc: 0.4953

Epoch 17/86
----------
train Loss: 0.0105 Acc: 0.7028
val Loss: 0.0276 Acc: 0.5047

Epoch 18/86
----------
train Loss: 0.0103 Acc: 0.7110
val Loss: 0.0312 Acc: 0.5047

Epoch 19/86
----------
train Loss: 0.0104 Acc: 0.6997
val Loss: 0.0284 Acc: 0.5047

Epoch 20/86
----------
train Loss: 0.0101 Acc: 0.7203
val Loss: 0.0368 Acc: 0.5047

Epoch 21/86
----------
train Loss: 0.0104 Acc: 0.6966
val Loss: 0.0278 Acc: 0.5047

Epoch 22/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0103 Acc: 0.7059
val Loss: 0.0256 Acc: 0.4953

Epoch 23/86
----------
train Loss: 0.0105 Acc: 0.7121
val Loss: 0.0255 Acc: 0.4860

Epoch 24/86
----------
train Loss: 0.0101 Acc: 0.7172
val Loss: 0.0317 Acc: 0.4860

Epoch 25/86
----------
train Loss: 0.0103 Acc: 0.7059
val Loss: 0.0229 Acc: 0.4953

Epoch 26/86
----------
train Loss: 0.0100 Acc: 0.7245
val Loss: 0.0302 Acc: 0.4860

Epoch 27/86
----------
train Loss: 0.0103 Acc: 0.6987
val Loss: 0.0310 Acc: 0.4860

Epoch 28/86
----------
train Loss: 0.0102 Acc: 0.7234
val Loss: 0.0256 Acc: 0.4860

Epoch 29/86
----------
train Loss: 0.0102 Acc: 0.7152
val Loss: 0.0254 Acc: 0.4953

Epoch 30/86
----------
train Loss: 0.0102 Acc: 0.7276
val Loss: 0.0332 Acc: 0.4953

Epoch 31/86
----------
train Loss: 0.0100 Acc: 0.7172
val Loss: 0.0251 Acc: 0.4953

Epoch 32/86
----------
train Loss: 0.0104 Acc: 0.6842
val Loss: 0.0298 Acc: 0.4860

Epoch 33/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0103 Acc: 0.7059
val Loss: 0.0252 Acc: 0.4953

Epoch 34/86
----------
train Loss: 0.0100 Acc: 0.7121
val Loss: 0.0283 Acc: 0.4860

Epoch 35/86
----------
train Loss: 0.0102 Acc: 0.7183
val Loss: 0.0243 Acc: 0.4953

Epoch 36/86
----------
train Loss: 0.0103 Acc: 0.7203
val Loss: 0.0260 Acc: 0.4860

Epoch 37/86
----------
train Loss: 0.0102 Acc: 0.7162
val Loss: 0.0284 Acc: 0.4860

Epoch 38/86
----------
train Loss: 0.0101 Acc: 0.7172
val Loss: 0.0256 Acc: 0.4860

Epoch 39/86
----------
train Loss: 0.0103 Acc: 0.7152
val Loss: 0.0270 Acc: 0.4953

Epoch 40/86
----------
train Loss: 0.0099 Acc: 0.7193
val Loss: 0.0219 Acc: 0.4860

Epoch 41/86
----------
train Loss: 0.0103 Acc: 0.7121
val Loss: 0.0218 Acc: 0.4860

Epoch 42/86
----------
train Loss: 0.0102 Acc: 0.7110
val Loss: 0.0266 Acc: 0.4860

Epoch 43/86
----------
train Loss: 0.0104 Acc: 0.7069
val Loss: 0.0287 Acc: 0.4953

Epoch 44/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0103 Acc: 0.7079
val Loss: 0.0298 Acc: 0.4953

Epoch 45/86
----------
train Loss: 0.0102 Acc: 0.6987
val Loss: 0.0244 Acc: 0.4860

Epoch 46/86
----------
train Loss: 0.0104 Acc: 0.6945
val Loss: 0.0369 Acc: 0.4953

Epoch 47/86
----------
train Loss: 0.0101 Acc: 0.7090
val Loss: 0.0324 Acc: 0.4953

Epoch 48/86
----------
train Loss: 0.0101 Acc: 0.7255
val Loss: 0.0262 Acc: 0.4860

Epoch 49/86
----------
train Loss: 0.0105 Acc: 0.7110
val Loss: 0.0279 Acc: 0.4860

Epoch 50/86
----------
train Loss: 0.0103 Acc: 0.7028
val Loss: 0.0348 Acc: 0.4860

Epoch 51/86
----------
train Loss: 0.0104 Acc: 0.6945
val Loss: 0.0323 Acc: 0.4860

Epoch 52/86
----------
train Loss: 0.0104 Acc: 0.7018
val Loss: 0.0310 Acc: 0.4860

Epoch 53/86
----------
train Loss: 0.0102 Acc: 0.7245
val Loss: 0.0407 Acc: 0.4860

Epoch 54/86
----------
train Loss: 0.0103 Acc: 0.7224
val Loss: 0.0250 Acc: 0.4860

Epoch 55/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0102 Acc: 0.7059
val Loss: 0.0314 Acc: 0.4860

Epoch 56/86
----------
train Loss: 0.0102 Acc: 0.7069
val Loss: 0.0318 Acc: 0.4860

Epoch 57/86
----------
train Loss: 0.0102 Acc: 0.7172
val Loss: 0.0388 Acc: 0.4860

Epoch 58/86
----------
train Loss: 0.0104 Acc: 0.7152
val Loss: 0.0307 Acc: 0.4953

Epoch 59/86
----------
train Loss: 0.0103 Acc: 0.6966
val Loss: 0.0236 Acc: 0.4860

Epoch 60/86
----------
train Loss: 0.0100 Acc: 0.7307
val Loss: 0.0252 Acc: 0.4860

Epoch 61/86
----------
train Loss: 0.0102 Acc: 0.7121
val Loss: 0.0233 Acc: 0.4860

Epoch 62/86
----------
train Loss: 0.0102 Acc: 0.7183
val Loss: 0.0272 Acc: 0.4860

Epoch 63/86
----------
train Loss: 0.0102 Acc: 0.7265
val Loss: 0.0296 Acc: 0.4860

Epoch 64/86
----------
train Loss: 0.0102 Acc: 0.7121
val Loss: 0.0245 Acc: 0.4860

Epoch 65/86
----------
train Loss: 0.0102 Acc: 0.7276
val Loss: 0.0264 Acc: 0.4860

Epoch 66/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0099 Acc: 0.7327
val Loss: 0.0255 Acc: 0.4860

Epoch 67/86
----------
train Loss: 0.0102 Acc: 0.7193
val Loss: 0.0308 Acc: 0.4953

Epoch 68/86
----------
train Loss: 0.0101 Acc: 0.7018
val Loss: 0.0308 Acc: 0.4953

Epoch 69/86
----------
train Loss: 0.0104 Acc: 0.7007
val Loss: 0.0294 Acc: 0.4860

Epoch 70/86
----------
train Loss: 0.0101 Acc: 0.7224
val Loss: 0.0289 Acc: 0.4860

Epoch 71/86
----------
train Loss: 0.0103 Acc: 0.7110
val Loss: 0.0269 Acc: 0.4953

Epoch 72/86
----------
train Loss: 0.0101 Acc: 0.7245
val Loss: 0.0289 Acc: 0.4953

Epoch 73/86
----------
train Loss: 0.0102 Acc: 0.7110
val Loss: 0.0324 Acc: 0.4860

Epoch 74/86
----------
train Loss: 0.0102 Acc: 0.7172
val Loss: 0.0302 Acc: 0.4860

Epoch 75/86
----------
train Loss: 0.0102 Acc: 0.6997
val Loss: 0.0234 Acc: 0.4860

Epoch 76/86
----------
train Loss: 0.0102 Acc: 0.7018
val Loss: 0.0284 Acc: 0.4860

Epoch 77/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0101 Acc: 0.7121
val Loss: 0.0312 Acc: 0.4953

Epoch 78/86
----------
train Loss: 0.0102 Acc: 0.7172
val Loss: 0.0293 Acc: 0.4860

Epoch 79/86
----------
train Loss: 0.0103 Acc: 0.7090
val Loss: 0.0259 Acc: 0.4860

Epoch 80/86
----------
train Loss: 0.0103 Acc: 0.7162
val Loss: 0.0259 Acc: 0.4953

Epoch 81/86
----------
train Loss: 0.0102 Acc: 0.7131
val Loss: 0.0311 Acc: 0.4860

Epoch 82/86
----------
train Loss: 0.0099 Acc: 0.7234
val Loss: 0.0266 Acc: 0.4860

Epoch 83/86
----------
train Loss: 0.0103 Acc: 0.7214
val Loss: 0.0381 Acc: 0.4860

Epoch 84/86
----------
train Loss: 0.0103 Acc: 0.6966
val Loss: 0.0286 Acc: 0.4953

Epoch 85/86
----------
train Loss: 0.0101 Acc: 0.7152
val Loss: 0.0294 Acc: 0.4860

Epoch 86/86
----------
train Loss: 0.0103 Acc: 0.7255
val Loss: 0.0360 Acc: 0.4953

Training complete in 8m 6s
Best val Acc: 0.551402

---Fine tuning.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0110 Acc: 0.6574
val Loss: 0.0398 Acc: 0.4673

Epoch 1/86
----------
train Loss: 0.0072 Acc: 0.7926
val Loss: 0.0317 Acc: 0.4673

Epoch 2/86
----------
train Loss: 0.0050 Acc: 0.8658
val Loss: 0.0284 Acc: 0.5794

Epoch 3/86
----------
train Loss: 0.0034 Acc: 0.9071
val Loss: 0.0257 Acc: 0.5888

Epoch 4/86
----------
train Loss: 0.0024 Acc: 0.9412
val Loss: 0.0277 Acc: 0.5607

Epoch 5/86
----------
train Loss: 0.0018 Acc: 0.9536
val Loss: 0.0290 Acc: 0.5607

Epoch 6/86
----------
train Loss: 0.0015 Acc: 0.9639
val Loss: 0.0207 Acc: 0.5701

Epoch 7/86
----------
train Loss: 0.0013 Acc: 0.9649
val Loss: 0.0277 Acc: 0.5514

Epoch 8/86
----------
train Loss: 0.0012 Acc: 0.9711
val Loss: 0.0339 Acc: 0.5888

Epoch 9/86
----------
train Loss: 0.0010 Acc: 0.9690
val Loss: 0.0335 Acc: 0.6075

Epoch 10/86
----------
train Loss: 0.0009 Acc: 0.9711
val Loss: 0.0428 Acc: 0.5981

Epoch 11/86
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0535 Acc: 0.5794

Epoch 12/86
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0362 Acc: 0.5981

Epoch 13/86
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0195 Acc: 0.6075

Epoch 14/86
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0558 Acc: 0.5981

Epoch 15/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0229 Acc: 0.5794

Epoch 16/86
----------
train Loss: 0.0007 Acc: 0.9783
val Loss: 0.0265 Acc: 0.5794

Epoch 17/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0332 Acc: 0.5794

Epoch 18/86
----------
train Loss: 0.0006 Acc: 0.9763
val Loss: 0.0423 Acc: 0.5981

Epoch 19/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0340 Acc: 0.5794

Epoch 20/86
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0257 Acc: 0.5888

Epoch 21/86
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0372 Acc: 0.5888

Epoch 22/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0429 Acc: 0.5888

Epoch 23/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0410 Acc: 0.5794

Epoch 24/86
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0222 Acc: 0.5888

Epoch 25/86
----------
train Loss: 0.0006 Acc: 0.9794
val Loss: 0.0281 Acc: 0.5888

Epoch 26/86
----------
train Loss: 0.0006 Acc: 0.9856
val Loss: 0.0400 Acc: 0.5888

Epoch 27/86
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0263 Acc: 0.5888

Epoch 28/86
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0303 Acc: 0.5888

Epoch 29/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0303 Acc: 0.5888

Epoch 30/86
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0414 Acc: 0.5981

Epoch 31/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0292 Acc: 0.5888

Epoch 32/86
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0571 Acc: 0.5888

Epoch 33/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0317 Acc: 0.5888

Epoch 34/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0286 Acc: 0.5888

Epoch 35/86
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0418 Acc: 0.5888

Epoch 36/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0319 Acc: 0.5888

Epoch 37/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0429 Acc: 0.5888

Epoch 38/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0293 Acc: 0.5888

Epoch 39/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0391 Acc: 0.5888

Epoch 40/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0227 Acc: 0.5888

Epoch 41/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0358 Acc: 0.5888

Epoch 42/86
----------
train Loss: 0.0004 Acc: 0.9917
val Loss: 0.0414 Acc: 0.5888

Epoch 43/86
----------
train Loss: 0.0005 Acc: 0.9794
val Loss: 0.0484 Acc: 0.5888

Epoch 44/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0421 Acc: 0.6075

Epoch 45/86
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0456 Acc: 0.5888

Epoch 46/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0303 Acc: 0.5981

Epoch 47/86
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0276 Acc: 0.5888

Epoch 48/86
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0268 Acc: 0.5888

Epoch 49/86
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0241 Acc: 0.5888

Epoch 50/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0448 Acc: 0.5794

Epoch 51/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0323 Acc: 0.5981

Epoch 52/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0449 Acc: 0.5888

Epoch 53/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0341 Acc: 0.5888

Epoch 54/86
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0411 Acc: 0.5888

Epoch 55/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0310 Acc: 0.5981

Epoch 56/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0268 Acc: 0.5981

Epoch 57/86
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0251 Acc: 0.5888

Epoch 58/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0330 Acc: 0.5888

Epoch 59/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0286 Acc: 0.5888

Epoch 60/86
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0374 Acc: 0.5888

Epoch 61/86
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0387 Acc: 0.5888

Epoch 62/86
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0495 Acc: 0.5888

Epoch 63/86
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0322 Acc: 0.5888

Epoch 64/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0485 Acc: 0.5888

Epoch 65/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0369 Acc: 0.5981

Epoch 66/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0209 Acc: 0.5888

Epoch 67/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0344 Acc: 0.5888

Epoch 68/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0365 Acc: 0.5888

Epoch 69/86
----------
train Loss: 0.0006 Acc: 0.9856
val Loss: 0.0268 Acc: 0.5888

Epoch 70/86
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0339 Acc: 0.5888

Epoch 71/86
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0381 Acc: 0.5888

Epoch 72/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0449 Acc: 0.5888

Epoch 73/86
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0341 Acc: 0.5888

Epoch 74/86
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0285 Acc: 0.5888

Epoch 75/86
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0480 Acc: 0.5888

Epoch 76/86
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0316 Acc: 0.5888

Epoch 77/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0296 Acc: 0.5888

Epoch 78/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0434 Acc: 0.5888

Epoch 79/86
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0298 Acc: 0.5888

Epoch 80/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0245 Acc: 0.5981

Epoch 81/86
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0380 Acc: 0.5888

Epoch 82/86
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0280 Acc: 0.5888

Epoch 83/86
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0204 Acc: 0.5981

Epoch 84/86
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0383 Acc: 0.5794

Epoch 85/86
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0313 Acc: 0.5888

Epoch 86/86
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0352 Acc: 0.5888

Training complete in 8m 44s
Best val Acc: 0.607477

---Testing---
Test accuracy: 0.945167
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 91 %
Accuracy of Blackfin tuna : 99 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 86 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 92 %
Accuracy of Pacific bluefin tuna : 98 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 85 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.9287342960917374, std: 0.05515410274289879
--------------------

run info[val: 0.15, epoch: 72, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0285 Acc: 0.1519
val Loss: 0.0291 Acc: 0.3043

Epoch 1/71
----------
train Loss: 0.0237 Acc: 0.3344
val Loss: 0.0265 Acc: 0.3292

Epoch 2/71
----------
train Loss: 0.0202 Acc: 0.4142
val Loss: 0.0256 Acc: 0.3478

Epoch 3/71
----------
train Loss: 0.0185 Acc: 0.4689
val Loss: 0.0242 Acc: 0.4099

Epoch 4/71
----------
train Loss: 0.0162 Acc: 0.5290
val Loss: 0.0250 Acc: 0.3292

Epoch 5/71
----------
train Loss: 0.0165 Acc: 0.5126
val Loss: 0.0215 Acc: 0.4161

Epoch 6/71
----------
train Loss: 0.0148 Acc: 0.5421
val Loss: 0.0224 Acc: 0.4037

Epoch 7/71
----------
train Loss: 0.0134 Acc: 0.6317
val Loss: 0.0210 Acc: 0.4161

Epoch 8/71
----------
LR is set to 0.001
train Loss: 0.0126 Acc: 0.6033
val Loss: 0.0208 Acc: 0.4410

Epoch 9/71
----------
train Loss: 0.0125 Acc: 0.6503
val Loss: 0.0214 Acc: 0.4224

Epoch 10/71
----------
train Loss: 0.0124 Acc: 0.6863
val Loss: 0.0213 Acc: 0.4410

Epoch 11/71
----------
train Loss: 0.0120 Acc: 0.6689
val Loss: 0.0211 Acc: 0.4099

Epoch 12/71
----------
train Loss: 0.0121 Acc: 0.6754
val Loss: 0.0209 Acc: 0.4286

Epoch 13/71
----------
train Loss: 0.0123 Acc: 0.6699
val Loss: 0.0213 Acc: 0.4348

Epoch 14/71
----------
train Loss: 0.0121 Acc: 0.6645
val Loss: 0.0211 Acc: 0.4161

Epoch 15/71
----------
train Loss: 0.0120 Acc: 0.6765
val Loss: 0.0205 Acc: 0.4161

Epoch 16/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0118 Acc: 0.6634
val Loss: 0.0214 Acc: 0.4224

Epoch 17/71
----------
train Loss: 0.0121 Acc: 0.6612
val Loss: 0.0215 Acc: 0.4161

Epoch 18/71
----------
train Loss: 0.0118 Acc: 0.6667
val Loss: 0.0205 Acc: 0.4224

Epoch 19/71
----------
train Loss: 0.0123 Acc: 0.6623
val Loss: 0.0207 Acc: 0.4286

Epoch 20/71
----------
train Loss: 0.0118 Acc: 0.6743
val Loss: 0.0207 Acc: 0.4286

Epoch 21/71
----------
train Loss: 0.0117 Acc: 0.6689
val Loss: 0.0210 Acc: 0.4224

Epoch 22/71
----------
train Loss: 0.0116 Acc: 0.6842
val Loss: 0.0206 Acc: 0.4348

Epoch 23/71
----------
train Loss: 0.0123 Acc: 0.6798
val Loss: 0.0213 Acc: 0.4286

Epoch 24/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0122 Acc: 0.6579
val Loss: 0.0210 Acc: 0.4410

Epoch 25/71
----------
train Loss: 0.0117 Acc: 0.6787
val Loss: 0.0209 Acc: 0.4224

Epoch 26/71
----------
train Loss: 0.0122 Acc: 0.6776
val Loss: 0.0206 Acc: 0.4286

Epoch 27/71
----------
train Loss: 0.0121 Acc: 0.6776
val Loss: 0.0209 Acc: 0.4410

Epoch 28/71
----------
train Loss: 0.0120 Acc: 0.6645
val Loss: 0.0209 Acc: 0.4161

Epoch 29/71
----------
train Loss: 0.0120 Acc: 0.6732
val Loss: 0.0212 Acc: 0.4224

Epoch 30/71
----------
train Loss: 0.0121 Acc: 0.6656
val Loss: 0.0208 Acc: 0.4348

Epoch 31/71
----------
train Loss: 0.0120 Acc: 0.6874
val Loss: 0.0208 Acc: 0.4348

Epoch 32/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0115 Acc: 0.6765
val Loss: 0.0210 Acc: 0.4348

Epoch 33/71
----------
train Loss: 0.0121 Acc: 0.6754
val Loss: 0.0208 Acc: 0.4348

Epoch 34/71
----------
train Loss: 0.0121 Acc: 0.6765
val Loss: 0.0211 Acc: 0.4161

Epoch 35/71
----------
train Loss: 0.0117 Acc: 0.6787
val Loss: 0.0207 Acc: 0.4410

Epoch 36/71
----------
train Loss: 0.0119 Acc: 0.6667
val Loss: 0.0211 Acc: 0.4348

Epoch 37/71
----------
train Loss: 0.0118 Acc: 0.6776
val Loss: 0.0216 Acc: 0.4161

Epoch 38/71
----------
train Loss: 0.0118 Acc: 0.6590
val Loss: 0.0205 Acc: 0.4161

Epoch 39/71
----------
train Loss: 0.0119 Acc: 0.6721
val Loss: 0.0211 Acc: 0.4224

Epoch 40/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0120 Acc: 0.6863
val Loss: 0.0214 Acc: 0.4224

Epoch 41/71
----------
train Loss: 0.0119 Acc: 0.6667
val Loss: 0.0211 Acc: 0.4286

Epoch 42/71
----------
train Loss: 0.0120 Acc: 0.6689
val Loss: 0.0209 Acc: 0.4286

Epoch 43/71
----------
train Loss: 0.0124 Acc: 0.6732
val Loss: 0.0206 Acc: 0.4286

Epoch 44/71
----------
train Loss: 0.0124 Acc: 0.6776
val Loss: 0.0210 Acc: 0.4348

Epoch 45/71
----------
train Loss: 0.0119 Acc: 0.6787
val Loss: 0.0216 Acc: 0.4161

Epoch 46/71
----------
train Loss: 0.0118 Acc: 0.6689
val Loss: 0.0207 Acc: 0.4286

Epoch 47/71
----------
train Loss: 0.0119 Acc: 0.6678
val Loss: 0.0204 Acc: 0.4348

Epoch 48/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0119 Acc: 0.6689
val Loss: 0.0214 Acc: 0.4348

Epoch 49/71
----------
train Loss: 0.0124 Acc: 0.6699
val Loss: 0.0207 Acc: 0.4348

Epoch 50/71
----------
train Loss: 0.0119 Acc: 0.6907
val Loss: 0.0208 Acc: 0.4224

Epoch 51/71
----------
train Loss: 0.0118 Acc: 0.6699
val Loss: 0.0207 Acc: 0.4286

Epoch 52/71
----------
train Loss: 0.0118 Acc: 0.6754
val Loss: 0.0202 Acc: 0.4286

Epoch 53/71
----------
train Loss: 0.0118 Acc: 0.6798
val Loss: 0.0209 Acc: 0.4161

Epoch 54/71
----------
train Loss: 0.0118 Acc: 0.6940
val Loss: 0.0203 Acc: 0.4286

Epoch 55/71
----------
train Loss: 0.0117 Acc: 0.6852
val Loss: 0.0206 Acc: 0.4161

Epoch 56/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0120 Acc: 0.6874
val Loss: 0.0207 Acc: 0.4099

Epoch 57/71
----------
train Loss: 0.0118 Acc: 0.6710
val Loss: 0.0207 Acc: 0.4224

Epoch 58/71
----------
train Loss: 0.0114 Acc: 0.6874
val Loss: 0.0217 Acc: 0.4224

Epoch 59/71
----------
train Loss: 0.0118 Acc: 0.6645
val Loss: 0.0210 Acc: 0.4161

Epoch 60/71
----------
train Loss: 0.0121 Acc: 0.6820
val Loss: 0.0209 Acc: 0.4161

Epoch 61/71
----------
train Loss: 0.0116 Acc: 0.6918
val Loss: 0.0208 Acc: 0.4224

Epoch 62/71
----------
train Loss: 0.0119 Acc: 0.6710
val Loss: 0.0210 Acc: 0.4161

Epoch 63/71
----------
train Loss: 0.0117 Acc: 0.6787
val Loss: 0.0203 Acc: 0.4410

Epoch 64/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0118 Acc: 0.6699
val Loss: 0.0214 Acc: 0.4161

Epoch 65/71
----------
train Loss: 0.0122 Acc: 0.6874
val Loss: 0.0212 Acc: 0.4161

Epoch 66/71
----------
train Loss: 0.0125 Acc: 0.6754
val Loss: 0.0206 Acc: 0.4099

Epoch 67/71
----------
train Loss: 0.0118 Acc: 0.6798
val Loss: 0.0215 Acc: 0.4224

Epoch 68/71
----------
train Loss: 0.0126 Acc: 0.6667
val Loss: 0.0211 Acc: 0.4410

Epoch 69/71
----------
train Loss: 0.0122 Acc: 0.6831
val Loss: 0.0214 Acc: 0.4348

Epoch 70/71
----------
train Loss: 0.0121 Acc: 0.6678
val Loss: 0.0209 Acc: 0.4286

Epoch 71/71
----------
train Loss: 0.0119 Acc: 0.6852
val Loss: 0.0213 Acc: 0.4161

Training complete in 6m 48s
Best val Acc: 0.440994

---Fine tuning.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0128 Acc: 0.6361
val Loss: 0.0215 Acc: 0.4658

Epoch 1/71
----------
train Loss: 0.0094 Acc: 0.7268
val Loss: 0.0253 Acc: 0.4037

Epoch 2/71
----------
train Loss: 0.0075 Acc: 0.7628
val Loss: 0.0221 Acc: 0.4783

Epoch 3/71
----------
train Loss: 0.0054 Acc: 0.8623
val Loss: 0.0246 Acc: 0.4348

Epoch 4/71
----------
train Loss: 0.0039 Acc: 0.9027
val Loss: 0.0236 Acc: 0.4534

Epoch 5/71
----------
train Loss: 0.0033 Acc: 0.9137
val Loss: 0.0241 Acc: 0.4720

Epoch 6/71
----------
train Loss: 0.0028 Acc: 0.9344
val Loss: 0.0265 Acc: 0.4596

Epoch 7/71
----------
train Loss: 0.0022 Acc: 0.9421
val Loss: 0.0245 Acc: 0.4907

Epoch 8/71
----------
LR is set to 0.001
train Loss: 0.0018 Acc: 0.9574
val Loss: 0.0231 Acc: 0.4969

Epoch 9/71
----------
train Loss: 0.0015 Acc: 0.9661
val Loss: 0.0234 Acc: 0.5093

Epoch 10/71
----------
train Loss: 0.0011 Acc: 0.9738
val Loss: 0.0233 Acc: 0.5093

Epoch 11/71
----------
train Loss: 0.0011 Acc: 0.9760
val Loss: 0.0223 Acc: 0.5093

Epoch 12/71
----------
train Loss: 0.0013 Acc: 0.9727
val Loss: 0.0226 Acc: 0.5217

Epoch 13/71
----------
train Loss: 0.0010 Acc: 0.9781
val Loss: 0.0229 Acc: 0.5155

Epoch 14/71
----------
train Loss: 0.0012 Acc: 0.9770
val Loss: 0.0226 Acc: 0.5031

Epoch 15/71
----------
train Loss: 0.0011 Acc: 0.9738
val Loss: 0.0229 Acc: 0.5093

Epoch 16/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0234 Acc: 0.5031

Epoch 17/71
----------
train Loss: 0.0009 Acc: 0.9781
val Loss: 0.0240 Acc: 0.5031

Epoch 18/71
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0234 Acc: 0.5031

Epoch 19/71
----------
train Loss: 0.0009 Acc: 0.9760
val Loss: 0.0234 Acc: 0.5031

Epoch 20/71
----------
train Loss: 0.0009 Acc: 0.9814
val Loss: 0.0227 Acc: 0.4969

Epoch 21/71
----------
train Loss: 0.0014 Acc: 0.9760
val Loss: 0.0227 Acc: 0.4969

Epoch 22/71
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0226 Acc: 0.4969

Epoch 23/71
----------
train Loss: 0.0010 Acc: 0.9792
val Loss: 0.0219 Acc: 0.4969

Epoch 24/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0231 Acc: 0.4969

Epoch 25/71
----------
train Loss: 0.0015 Acc: 0.9760
val Loss: 0.0226 Acc: 0.5031

Epoch 26/71
----------
train Loss: 0.0014 Acc: 0.9770
val Loss: 0.0232 Acc: 0.5093

Epoch 27/71
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0233 Acc: 0.4969

Epoch 28/71
----------
train Loss: 0.0009 Acc: 0.9814
val Loss: 0.0236 Acc: 0.5031

Epoch 29/71
----------
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0235 Acc: 0.5031

Epoch 30/71
----------
train Loss: 0.0012 Acc: 0.9694
val Loss: 0.0235 Acc: 0.4969

Epoch 31/71
----------
train Loss: 0.0009 Acc: 0.9781
val Loss: 0.0227 Acc: 0.5093

Epoch 32/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0009 Acc: 0.9836
val Loss: 0.0229 Acc: 0.5093

Epoch 33/71
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0229 Acc: 0.4969

Epoch 34/71
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0235 Acc: 0.5093

Epoch 35/71
----------
train Loss: 0.0009 Acc: 0.9770
val Loss: 0.0233 Acc: 0.5093

Epoch 36/71
----------
train Loss: 0.0009 Acc: 0.9770
val Loss: 0.0241 Acc: 0.4969

Epoch 37/71
----------
train Loss: 0.0012 Acc: 0.9683
val Loss: 0.0231 Acc: 0.5155

Epoch 38/71
----------
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0231 Acc: 0.5155

Epoch 39/71
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0224 Acc: 0.5155

Epoch 40/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0011 Acc: 0.9770
val Loss: 0.0227 Acc: 0.5093

Epoch 41/71
----------
train Loss: 0.0009 Acc: 0.9858
val Loss: 0.0234 Acc: 0.5031

Epoch 42/71
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0228 Acc: 0.5031

Epoch 43/71
----------
train Loss: 0.0010 Acc: 0.9738
val Loss: 0.0227 Acc: 0.5031

Epoch 44/71
----------
train Loss: 0.0011 Acc: 0.9727
val Loss: 0.0230 Acc: 0.5093

Epoch 45/71
----------
train Loss: 0.0009 Acc: 0.9781
val Loss: 0.0219 Acc: 0.4969

Epoch 46/71
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0232 Acc: 0.4969

Epoch 47/71
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0228 Acc: 0.5031

Epoch 48/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9869
val Loss: 0.0231 Acc: 0.5093

Epoch 49/71
----------
train Loss: 0.0011 Acc: 0.9749
val Loss: 0.0232 Acc: 0.4969

Epoch 50/71
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0235 Acc: 0.5031

Epoch 51/71
----------
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0224 Acc: 0.5093

Epoch 52/71
----------
train Loss: 0.0011 Acc: 0.9705
val Loss: 0.0229 Acc: 0.5093

Epoch 53/71
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0233 Acc: 0.5093

Epoch 54/71
----------
train Loss: 0.0012 Acc: 0.9760
val Loss: 0.0233 Acc: 0.4969

Epoch 55/71
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0227 Acc: 0.4907

Epoch 56/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0226 Acc: 0.4907

Epoch 57/71
----------
train Loss: 0.0009 Acc: 0.9727
val Loss: 0.0232 Acc: 0.5093

Epoch 58/71
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0224 Acc: 0.5093

Epoch 59/71
----------
train Loss: 0.0010 Acc: 0.9749
val Loss: 0.0223 Acc: 0.5031

Epoch 60/71
----------
train Loss: 0.0011 Acc: 0.9803
val Loss: 0.0240 Acc: 0.4969

Epoch 61/71
----------
train Loss: 0.0009 Acc: 0.9825
val Loss: 0.0232 Acc: 0.5031

Epoch 62/71
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0231 Acc: 0.5031

Epoch 63/71
----------
train Loss: 0.0009 Acc: 0.9727
val Loss: 0.0227 Acc: 0.4969

Epoch 64/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0229 Acc: 0.4969

Epoch 65/71
----------
train Loss: 0.0010 Acc: 0.9803
val Loss: 0.0235 Acc: 0.5093

Epoch 66/71
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0225 Acc: 0.5031

Epoch 67/71
----------
train Loss: 0.0010 Acc: 0.9803
val Loss: 0.0235 Acc: 0.4969

Epoch 68/71
----------
train Loss: 0.0011 Acc: 0.9738
val Loss: 0.0222 Acc: 0.4969

Epoch 69/71
----------
train Loss: 0.0010 Acc: 0.9770
val Loss: 0.0234 Acc: 0.4907

Epoch 70/71
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0229 Acc: 0.4969

Epoch 71/71
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0232 Acc: 0.4969

Training complete in 7m 17s
Best val Acc: 0.521739

---Testing---
Test accuracy: 0.911710
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 88 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 97 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 84 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8860989002120396, std: 0.07084352237278334
--------------------

run info[val: 0.2, epoch: 80, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0274 Acc: 0.1301
val Loss: 0.0319 Acc: 0.2791

Epoch 1/79
----------
train Loss: 0.0225 Acc: 0.3066
val Loss: 0.0306 Acc: 0.3860

Epoch 2/79
----------
train Loss: 0.0198 Acc: 0.3984
val Loss: 0.0280 Acc: 0.3953

Epoch 3/79
----------
train Loss: 0.0170 Acc: 0.4959
val Loss: 0.0279 Acc: 0.4000

Epoch 4/79
----------
train Loss: 0.0163 Acc: 0.5261
val Loss: 0.0239 Acc: 0.4279

Epoch 5/79
----------
train Loss: 0.0147 Acc: 0.5331
val Loss: 0.0264 Acc: 0.4558

Epoch 6/79
----------
train Loss: 0.0141 Acc: 0.5796
val Loss: 0.0267 Acc: 0.4605

Epoch 7/79
----------
train Loss: 0.0132 Acc: 0.6121
val Loss: 0.0268 Acc: 0.4233

Epoch 8/79
----------
train Loss: 0.0124 Acc: 0.6144
val Loss: 0.0244 Acc: 0.5023

Epoch 9/79
----------
train Loss: 0.0119 Acc: 0.6469
val Loss: 0.0251 Acc: 0.4698

Epoch 10/79
----------
train Loss: 0.0115 Acc: 0.6643
val Loss: 0.0250 Acc: 0.5070

Epoch 11/79
----------
train Loss: 0.0111 Acc: 0.6806
val Loss: 0.0243 Acc: 0.4791

Epoch 12/79
----------
train Loss: 0.0107 Acc: 0.6841
val Loss: 0.0241 Acc: 0.4977

Epoch 13/79
----------
LR is set to 0.001
train Loss: 0.0103 Acc: 0.7108
val Loss: 0.0213 Acc: 0.4930

Epoch 14/79
----------
train Loss: 0.0101 Acc: 0.7317
val Loss: 0.0224 Acc: 0.4977

Epoch 15/79
----------
train Loss: 0.0100 Acc: 0.7317
val Loss: 0.0242 Acc: 0.4930

Epoch 16/79
----------
train Loss: 0.0100 Acc: 0.7224
val Loss: 0.0222 Acc: 0.5023

Epoch 17/79
----------
train Loss: 0.0099 Acc: 0.7329
val Loss: 0.0234 Acc: 0.4744

Epoch 18/79
----------
train Loss: 0.0102 Acc: 0.7201
val Loss: 0.0251 Acc: 0.5023

Epoch 19/79
----------
train Loss: 0.0101 Acc: 0.7305
val Loss: 0.0211 Acc: 0.4930

Epoch 20/79
----------
train Loss: 0.0098 Acc: 0.7317
val Loss: 0.0222 Acc: 0.5023

Epoch 21/79
----------
train Loss: 0.0095 Acc: 0.7352
val Loss: 0.0264 Acc: 0.4837

Epoch 22/79
----------
train Loss: 0.0095 Acc: 0.7340
val Loss: 0.0238 Acc: 0.4837

Epoch 23/79
----------
train Loss: 0.0100 Acc: 0.7178
val Loss: 0.0216 Acc: 0.5023

Epoch 24/79
----------
train Loss: 0.0097 Acc: 0.7294
val Loss: 0.0223 Acc: 0.4977

Epoch 25/79
----------
train Loss: 0.0096 Acc: 0.7340
val Loss: 0.0241 Acc: 0.4698

Epoch 26/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0098 Acc: 0.7422
val Loss: 0.0217 Acc: 0.4791

Epoch 27/79
----------
train Loss: 0.0096 Acc: 0.7178
val Loss: 0.0258 Acc: 0.4837

Epoch 28/79
----------
train Loss: 0.0096 Acc: 0.7247
val Loss: 0.0235 Acc: 0.4791

Epoch 29/79
----------
train Loss: 0.0095 Acc: 0.7433
val Loss: 0.0223 Acc: 0.4837

Epoch 30/79
----------
train Loss: 0.0096 Acc: 0.7398
val Loss: 0.0257 Acc: 0.4837

Epoch 31/79
----------
train Loss: 0.0095 Acc: 0.7445
val Loss: 0.0210 Acc: 0.4977

Epoch 32/79
----------
train Loss: 0.0095 Acc: 0.7503
val Loss: 0.0233 Acc: 0.4930

Epoch 33/79
----------
train Loss: 0.0095 Acc: 0.7387
val Loss: 0.0229 Acc: 0.4837

Epoch 34/79
----------
train Loss: 0.0095 Acc: 0.7387
val Loss: 0.0229 Acc: 0.4791

Epoch 35/79
----------
train Loss: 0.0098 Acc: 0.7329
val Loss: 0.0229 Acc: 0.4930

Epoch 36/79
----------
train Loss: 0.0098 Acc: 0.7364
val Loss: 0.0225 Acc: 0.4930

Epoch 37/79
----------
train Loss: 0.0096 Acc: 0.7375
val Loss: 0.0230 Acc: 0.4884

Epoch 38/79
----------
train Loss: 0.0095 Acc: 0.7422
val Loss: 0.0247 Acc: 0.4930

Epoch 39/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0095 Acc: 0.7445
val Loss: 0.0228 Acc: 0.4977

Epoch 40/79
----------
train Loss: 0.0097 Acc: 0.7236
val Loss: 0.0241 Acc: 0.4884

Epoch 41/79
----------
train Loss: 0.0098 Acc: 0.7294
val Loss: 0.0218 Acc: 0.4884

Epoch 42/79
----------
train Loss: 0.0095 Acc: 0.7456
val Loss: 0.0228 Acc: 0.4884

Epoch 43/79
----------
train Loss: 0.0095 Acc: 0.7456
val Loss: 0.0239 Acc: 0.4930

Epoch 44/79
----------
train Loss: 0.0094 Acc: 0.7619
val Loss: 0.0231 Acc: 0.4930

Epoch 45/79
----------
train Loss: 0.0095 Acc: 0.7305
val Loss: 0.0245 Acc: 0.4837

Epoch 46/79
----------
train Loss: 0.0093 Acc: 0.7410
val Loss: 0.0233 Acc: 0.5023

Epoch 47/79
----------
train Loss: 0.0094 Acc: 0.7387
val Loss: 0.0257 Acc: 0.4837

Epoch 48/79
----------
train Loss: 0.0096 Acc: 0.7317
val Loss: 0.0265 Acc: 0.4791

Epoch 49/79
----------
train Loss: 0.0094 Acc: 0.7503
val Loss: 0.0242 Acc: 0.4884

Epoch 50/79
----------
train Loss: 0.0094 Acc: 0.7398
val Loss: 0.0226 Acc: 0.4884

Epoch 51/79
----------
train Loss: 0.0093 Acc: 0.7503
val Loss: 0.0209 Acc: 0.4884

Epoch 52/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0097 Acc: 0.7259
val Loss: 0.0219 Acc: 0.4884

Epoch 53/79
----------
train Loss: 0.0095 Acc: 0.7433
val Loss: 0.0224 Acc: 0.4930

Epoch 54/79
----------
train Loss: 0.0095 Acc: 0.7340
val Loss: 0.0235 Acc: 0.5023

Epoch 55/79
----------
train Loss: 0.0097 Acc: 0.7561
val Loss: 0.0220 Acc: 0.5023

Epoch 56/79
----------
train Loss: 0.0096 Acc: 0.7619
val Loss: 0.0232 Acc: 0.4930

Epoch 57/79
----------
train Loss: 0.0098 Acc: 0.7189
val Loss: 0.0241 Acc: 0.4930

Epoch 58/79
----------
train Loss: 0.0094 Acc: 0.7526
val Loss: 0.0249 Acc: 0.4977

Epoch 59/79
----------
train Loss: 0.0094 Acc: 0.7433
val Loss: 0.0228 Acc: 0.4930

Epoch 60/79
----------
train Loss: 0.0098 Acc: 0.7491
val Loss: 0.0224 Acc: 0.5023

Epoch 61/79
----------
train Loss: 0.0096 Acc: 0.7410
val Loss: 0.0223 Acc: 0.4884

Epoch 62/79
----------
train Loss: 0.0096 Acc: 0.7305
val Loss: 0.0251 Acc: 0.4977

Epoch 63/79
----------
train Loss: 0.0097 Acc: 0.7364
val Loss: 0.0259 Acc: 0.4977

Epoch 64/79
----------
train Loss: 0.0097 Acc: 0.7456
val Loss: 0.0232 Acc: 0.4977

Epoch 65/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0097 Acc: 0.7224
val Loss: 0.0237 Acc: 0.4744

Epoch 66/79
----------
train Loss: 0.0096 Acc: 0.7201
val Loss: 0.0217 Acc: 0.4837

Epoch 67/79
----------
train Loss: 0.0095 Acc: 0.7538
val Loss: 0.0244 Acc: 0.4837

Epoch 68/79
----------
train Loss: 0.0096 Acc: 0.7317
val Loss: 0.0248 Acc: 0.4837

Epoch 69/79
----------
train Loss: 0.0095 Acc: 0.7340
val Loss: 0.0245 Acc: 0.5023

Epoch 70/79
----------
train Loss: 0.0097 Acc: 0.7503
val Loss: 0.0239 Acc: 0.4977

Epoch 71/79
----------
train Loss: 0.0093 Acc: 0.7526
val Loss: 0.0222 Acc: 0.4930

Epoch 72/79
----------
train Loss: 0.0095 Acc: 0.7375
val Loss: 0.0215 Acc: 0.4930

Epoch 73/79
----------
train Loss: 0.0095 Acc: 0.7445
val Loss: 0.0250 Acc: 0.4837

Epoch 74/79
----------
train Loss: 0.0097 Acc: 0.7352
val Loss: 0.0237 Acc: 0.4791

Epoch 75/79
----------
train Loss: 0.0093 Acc: 0.7596
val Loss: 0.0243 Acc: 0.4837

Epoch 76/79
----------
train Loss: 0.0096 Acc: 0.7456
val Loss: 0.0238 Acc: 0.4930

Epoch 77/79
----------
train Loss: 0.0099 Acc: 0.7178
val Loss: 0.0236 Acc: 0.4884

Epoch 78/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0095 Acc: 0.7538
val Loss: 0.0244 Acc: 0.4884

Epoch 79/79
----------
train Loss: 0.0095 Acc: 0.7480
val Loss: 0.0222 Acc: 0.4791

Training complete in 7m 18s
Best val Acc: 0.506977

---Fine tuning.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0108 Acc: 0.6852
val Loss: 0.0231 Acc: 0.4791

Epoch 1/79
----------
train Loss: 0.0077 Acc: 0.7909
val Loss: 0.0219 Acc: 0.5302

Epoch 2/79
----------
train Loss: 0.0049 Acc: 0.8653
val Loss: 0.0236 Acc: 0.5256

Epoch 3/79
----------
train Loss: 0.0033 Acc: 0.9106
val Loss: 0.0217 Acc: 0.5535

Epoch 4/79
----------
train Loss: 0.0026 Acc: 0.9384
val Loss: 0.0224 Acc: 0.5721

Epoch 5/79
----------
train Loss: 0.0021 Acc: 0.9524
val Loss: 0.0241 Acc: 0.5209

Epoch 6/79
----------
train Loss: 0.0016 Acc: 0.9640
val Loss: 0.0288 Acc: 0.5256

Epoch 7/79
----------
train Loss: 0.0015 Acc: 0.9628
val Loss: 0.0241 Acc: 0.5163

Epoch 8/79
----------
train Loss: 0.0010 Acc: 0.9756
val Loss: 0.0219 Acc: 0.5721

Epoch 9/79
----------
train Loss: 0.0010 Acc: 0.9744
val Loss: 0.0238 Acc: 0.5674

Epoch 10/79
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0236 Acc: 0.5442

Epoch 11/79
----------
train Loss: 0.0009 Acc: 0.9779
val Loss: 0.0288 Acc: 0.5628

Epoch 12/79
----------
train Loss: 0.0008 Acc: 0.9744
val Loss: 0.0242 Acc: 0.5349

Epoch 13/79
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5488

Epoch 14/79
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0242 Acc: 0.5488

Epoch 15/79
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0239 Acc: 0.5442

Epoch 16/79
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0249 Acc: 0.5628

Epoch 17/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0226 Acc: 0.5581

Epoch 18/79
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0265 Acc: 0.5488

Epoch 19/79
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0264 Acc: 0.5535

Epoch 20/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0255 Acc: 0.5535

Epoch 21/79
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0261 Acc: 0.5535

Epoch 22/79
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0286 Acc: 0.5535

Epoch 23/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0219 Acc: 0.5535

Epoch 24/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0276 Acc: 0.5442

Epoch 25/79
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0192 Acc: 0.5535

Epoch 26/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0233 Acc: 0.5442

Epoch 27/79
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0201 Acc: 0.5488

Epoch 28/79
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0253 Acc: 0.5442

Epoch 29/79
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0223 Acc: 0.5442

Epoch 30/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0265 Acc: 0.5535

Epoch 31/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0210 Acc: 0.5628

Epoch 32/79
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0244 Acc: 0.5488

Epoch 33/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0243 Acc: 0.5488

Epoch 34/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0235 Acc: 0.5535

Epoch 35/79
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0252 Acc: 0.5535

Epoch 36/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0255 Acc: 0.5535

Epoch 37/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5674

Epoch 38/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0242 Acc: 0.5581

Epoch 39/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0230 Acc: 0.5535

Epoch 40/79
----------
train Loss: 0.0005 Acc: 0.9768
val Loss: 0.0298 Acc: 0.5442

Epoch 41/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0239 Acc: 0.5488

Epoch 42/79
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0238 Acc: 0.5535

Epoch 43/79
----------
train Loss: 0.0004 Acc: 0.9895
val Loss: 0.0219 Acc: 0.5442

Epoch 44/79
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0285 Acc: 0.5488

Epoch 45/79
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0285 Acc: 0.5488

Epoch 46/79
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0291 Acc: 0.5488

Epoch 47/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0225 Acc: 0.5535

Epoch 48/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0277 Acc: 0.5581

Epoch 49/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5535

Epoch 50/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0240 Acc: 0.5628

Epoch 51/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0235 Acc: 0.5535

Epoch 52/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0207 Acc: 0.5581

Epoch 53/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0207 Acc: 0.5488

Epoch 54/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0258 Acc: 0.5581

Epoch 55/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0262 Acc: 0.5535

Epoch 56/79
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0271 Acc: 0.5581

Epoch 57/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0270 Acc: 0.5535

Epoch 58/79
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0207 Acc: 0.5628

Epoch 59/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0214 Acc: 0.5535

Epoch 60/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0208 Acc: 0.5535

Epoch 61/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0264 Acc: 0.5488

Epoch 62/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0243 Acc: 0.5581

Epoch 63/79
----------
train Loss: 0.0004 Acc: 0.9849
val Loss: 0.0266 Acc: 0.5535

Epoch 64/79
----------
train Loss: 0.0005 Acc: 0.9768
val Loss: 0.0248 Acc: 0.5535

Epoch 65/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0244 Acc: 0.5488

Epoch 66/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0246 Acc: 0.5535

Epoch 67/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0217 Acc: 0.5442

Epoch 68/79
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0217 Acc: 0.5581

Epoch 69/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0217 Acc: 0.5535

Epoch 70/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0239 Acc: 0.5535

Epoch 71/79
----------
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0250 Acc: 0.5535

Epoch 72/79
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0240 Acc: 0.5628

Epoch 73/79
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0270 Acc: 0.5628

Epoch 74/79
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0241 Acc: 0.5581

Epoch 75/79
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0243 Acc: 0.5581

Epoch 76/79
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0264 Acc: 0.5628

Epoch 77/79
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0301 Acc: 0.5581

Epoch 78/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0229 Acc: 0.5628

Epoch 79/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0271 Acc: 0.5535

Training complete in 7m 35s
Best val Acc: 0.572093

---Testing---
Test accuracy: 0.883829
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 68 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 83 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 71 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.8400260674462172, std: 0.12239448356534195
--------------------

run info[val: 0.25, epoch: 86, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0288 Acc: 0.1574
val Loss: 0.0256 Acc: 0.3160

Epoch 1/85
----------
train Loss: 0.0235 Acc: 0.3222
val Loss: 0.0250 Acc: 0.3011

Epoch 2/85
----------
train Loss: 0.0204 Acc: 0.4002
val Loss: 0.0221 Acc: 0.3941

Epoch 3/85
----------
train Loss: 0.0186 Acc: 0.4511
val Loss: 0.0212 Acc: 0.3643

Epoch 4/85
----------
train Loss: 0.0182 Acc: 0.5006
val Loss: 0.0223 Acc: 0.4089

Epoch 5/85
----------
LR is set to 0.001
train Loss: 0.0160 Acc: 0.5564
val Loss: 0.0202 Acc: 0.4498

Epoch 6/85
----------
train Loss: 0.0143 Acc: 0.6159
val Loss: 0.0196 Acc: 0.4238

Epoch 7/85
----------
train Loss: 0.0148 Acc: 0.6084
val Loss: 0.0194 Acc: 0.4201

Epoch 8/85
----------
train Loss: 0.0139 Acc: 0.6134
val Loss: 0.0194 Acc: 0.4312

Epoch 9/85
----------
train Loss: 0.0137 Acc: 0.6369
val Loss: 0.0194 Acc: 0.4312

Epoch 10/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0139 Acc: 0.6468
val Loss: 0.0196 Acc: 0.4349

Epoch 11/85
----------
train Loss: 0.0135 Acc: 0.6295
val Loss: 0.0193 Acc: 0.4312

Epoch 12/85
----------
train Loss: 0.0136 Acc: 0.6481
val Loss: 0.0195 Acc: 0.4498

Epoch 13/85
----------
train Loss: 0.0136 Acc: 0.6481
val Loss: 0.0191 Acc: 0.4424

Epoch 14/85
----------
train Loss: 0.0134 Acc: 0.6543
val Loss: 0.0194 Acc: 0.4349

Epoch 15/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0135 Acc: 0.6518
val Loss: 0.0196 Acc: 0.4349

Epoch 16/85
----------
train Loss: 0.0130 Acc: 0.6406
val Loss: 0.0194 Acc: 0.4275

Epoch 17/85
----------
train Loss: 0.0138 Acc: 0.6468
val Loss: 0.0194 Acc: 0.4349

Epoch 18/85
----------
train Loss: 0.0138 Acc: 0.6295
val Loss: 0.0194 Acc: 0.4387

Epoch 19/85
----------
train Loss: 0.0139 Acc: 0.6456
val Loss: 0.0194 Acc: 0.4424

Epoch 20/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0135 Acc: 0.6592
val Loss: 0.0194 Acc: 0.4387

Epoch 21/85
----------
train Loss: 0.0133 Acc: 0.6320
val Loss: 0.0194 Acc: 0.4349

Epoch 22/85
----------
train Loss: 0.0135 Acc: 0.6419
val Loss: 0.0193 Acc: 0.4238

Epoch 23/85
----------
train Loss: 0.0139 Acc: 0.6629
val Loss: 0.0195 Acc: 0.4312

Epoch 24/85
----------
train Loss: 0.0143 Acc: 0.6406
val Loss: 0.0193 Acc: 0.4275

Epoch 25/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0142 Acc: 0.6382
val Loss: 0.0194 Acc: 0.4424

Epoch 26/85
----------
train Loss: 0.0139 Acc: 0.6295
val Loss: 0.0194 Acc: 0.4424

Epoch 27/85
----------
train Loss: 0.0136 Acc: 0.6295
val Loss: 0.0191 Acc: 0.4387

Epoch 28/85
----------
train Loss: 0.0140 Acc: 0.6456
val Loss: 0.0191 Acc: 0.4387

Epoch 29/85
----------
train Loss: 0.0135 Acc: 0.6406
val Loss: 0.0193 Acc: 0.4387

Epoch 30/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0135 Acc: 0.6506
val Loss: 0.0194 Acc: 0.4424

Epoch 31/85
----------
train Loss: 0.0138 Acc: 0.6419
val Loss: 0.0191 Acc: 0.4312

Epoch 32/85
----------
train Loss: 0.0145 Acc: 0.6431
val Loss: 0.0192 Acc: 0.4275

Epoch 33/85
----------
train Loss: 0.0143 Acc: 0.6456
val Loss: 0.0192 Acc: 0.4461

Epoch 34/85
----------
train Loss: 0.0132 Acc: 0.6444
val Loss: 0.0190 Acc: 0.4424

Epoch 35/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0140 Acc: 0.6506
val Loss: 0.0192 Acc: 0.4349

Epoch 36/85
----------
train Loss: 0.0148 Acc: 0.6382
val Loss: 0.0193 Acc: 0.4424

Epoch 37/85
----------
train Loss: 0.0131 Acc: 0.6444
val Loss: 0.0195 Acc: 0.4349

Epoch 38/85
----------
train Loss: 0.0132 Acc: 0.6394
val Loss: 0.0194 Acc: 0.4349

Epoch 39/85
----------
train Loss: 0.0138 Acc: 0.6481
val Loss: 0.0194 Acc: 0.4238

Epoch 40/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0138 Acc: 0.6394
val Loss: 0.0192 Acc: 0.4387

Epoch 41/85
----------
train Loss: 0.0132 Acc: 0.6506
val Loss: 0.0194 Acc: 0.4387

Epoch 42/85
----------
train Loss: 0.0133 Acc: 0.6518
val Loss: 0.0196 Acc: 0.4312

Epoch 43/85
----------
train Loss: 0.0141 Acc: 0.6394
val Loss: 0.0193 Acc: 0.4387

Epoch 44/85
----------
train Loss: 0.0136 Acc: 0.6518
val Loss: 0.0193 Acc: 0.4275

Epoch 45/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0139 Acc: 0.6555
val Loss: 0.0196 Acc: 0.4238

Epoch 46/85
----------
train Loss: 0.0143 Acc: 0.6444
val Loss: 0.0196 Acc: 0.4312

Epoch 47/85
----------
train Loss: 0.0133 Acc: 0.6419
val Loss: 0.0195 Acc: 0.4424

Epoch 48/85
----------
train Loss: 0.0141 Acc: 0.6444
val Loss: 0.0193 Acc: 0.4387

Epoch 49/85
----------
train Loss: 0.0131 Acc: 0.6394
val Loss: 0.0193 Acc: 0.4275

Epoch 50/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0133 Acc: 0.6444
val Loss: 0.0194 Acc: 0.4201

Epoch 51/85
----------
train Loss: 0.0135 Acc: 0.6444
val Loss: 0.0192 Acc: 0.4275

Epoch 52/85
----------
train Loss: 0.0138 Acc: 0.6444
val Loss: 0.0193 Acc: 0.4387

Epoch 53/85
----------
train Loss: 0.0137 Acc: 0.6530
val Loss: 0.0191 Acc: 0.4387

Epoch 54/85
----------
train Loss: 0.0142 Acc: 0.6555
val Loss: 0.0193 Acc: 0.4275

Epoch 55/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0136 Acc: 0.6518
val Loss: 0.0194 Acc: 0.4349

Epoch 56/85
----------
train Loss: 0.0136 Acc: 0.6357
val Loss: 0.0195 Acc: 0.4387

Epoch 57/85
----------
train Loss: 0.0136 Acc: 0.6369
val Loss: 0.0195 Acc: 0.4312

Epoch 58/85
----------
train Loss: 0.0130 Acc: 0.6506
val Loss: 0.0193 Acc: 0.4387

Epoch 59/85
----------
train Loss: 0.0135 Acc: 0.6568
val Loss: 0.0194 Acc: 0.4461

Epoch 60/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0131 Acc: 0.6493
val Loss: 0.0191 Acc: 0.4424

Epoch 61/85
----------
train Loss: 0.0135 Acc: 0.6406
val Loss: 0.0192 Acc: 0.4349

Epoch 62/85
----------
train Loss: 0.0137 Acc: 0.6394
val Loss: 0.0196 Acc: 0.4349

Epoch 63/85
----------
train Loss: 0.0135 Acc: 0.6543
val Loss: 0.0197 Acc: 0.4349

Epoch 64/85
----------
train Loss: 0.0143 Acc: 0.6369
val Loss: 0.0195 Acc: 0.4387

Epoch 65/85
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0138 Acc: 0.6394
val Loss: 0.0193 Acc: 0.4424

Epoch 66/85
----------
train Loss: 0.0136 Acc: 0.6406
val Loss: 0.0195 Acc: 0.4387

Epoch 67/85
----------
train Loss: 0.0133 Acc: 0.6493
val Loss: 0.0192 Acc: 0.4275

Epoch 68/85
----------
train Loss: 0.0143 Acc: 0.6419
val Loss: 0.0194 Acc: 0.4424

Epoch 69/85
----------
train Loss: 0.0138 Acc: 0.6506
val Loss: 0.0198 Acc: 0.4312

Epoch 70/85
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0136 Acc: 0.6382
val Loss: 0.0194 Acc: 0.4424

Epoch 71/85
----------
train Loss: 0.0136 Acc: 0.6456
val Loss: 0.0194 Acc: 0.4498

Epoch 72/85
----------
train Loss: 0.0149 Acc: 0.6444
val Loss: 0.0195 Acc: 0.4387

Epoch 73/85
----------
train Loss: 0.0140 Acc: 0.6493
val Loss: 0.0193 Acc: 0.4424

Epoch 74/85
----------
train Loss: 0.0140 Acc: 0.6468
val Loss: 0.0193 Acc: 0.4461

Epoch 75/85
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0140 Acc: 0.6394
val Loss: 0.0194 Acc: 0.4312

Epoch 76/85
----------
train Loss: 0.0137 Acc: 0.6518
val Loss: 0.0193 Acc: 0.4387

Epoch 77/85
----------
train Loss: 0.0137 Acc: 0.6468
val Loss: 0.0195 Acc: 0.4424

Epoch 78/85
----------
train Loss: 0.0132 Acc: 0.6419
val Loss: 0.0193 Acc: 0.4424

Epoch 79/85
----------
train Loss: 0.0140 Acc: 0.6382
val Loss: 0.0193 Acc: 0.4387

Epoch 80/85
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0137 Acc: 0.6530
val Loss: 0.0193 Acc: 0.4498

Epoch 81/85
----------
train Loss: 0.0136 Acc: 0.6394
val Loss: 0.0192 Acc: 0.4387

Epoch 82/85
----------
train Loss: 0.0141 Acc: 0.6357
val Loss: 0.0193 Acc: 0.4312

Epoch 83/85
----------
train Loss: 0.0137 Acc: 0.6382
val Loss: 0.0194 Acc: 0.4461

Epoch 84/85
----------
train Loss: 0.0139 Acc: 0.6344
val Loss: 0.0192 Acc: 0.4461

Epoch 85/85
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0140 Acc: 0.6357
val Loss: 0.0191 Acc: 0.4498

Training complete in 7m 51s
Best val Acc: 0.449814

---Fine tuning.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0153 Acc: 0.6010
val Loss: 0.0221 Acc: 0.4201

Epoch 1/85
----------
train Loss: 0.0134 Acc: 0.6183
val Loss: 0.0260 Acc: 0.3457

Epoch 2/85
----------
train Loss: 0.0114 Acc: 0.6853
val Loss: 0.0210 Acc: 0.4424

Epoch 3/85
----------
train Loss: 0.0079 Acc: 0.7695
val Loss: 0.0245 Acc: 0.4387

Epoch 4/85
----------
train Loss: 0.0060 Acc: 0.8092
val Loss: 0.0230 Acc: 0.3903

Epoch 5/85
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.8823
val Loss: 0.0194 Acc: 0.5093

Epoch 6/85
----------
train Loss: 0.0036 Acc: 0.9294
val Loss: 0.0180 Acc: 0.5576

Epoch 7/85
----------
train Loss: 0.0029 Acc: 0.9455
val Loss: 0.0182 Acc: 0.5539

Epoch 8/85
----------
train Loss: 0.0031 Acc: 0.9442
val Loss: 0.0187 Acc: 0.5465

Epoch 9/85
----------
train Loss: 0.0029 Acc: 0.9542
val Loss: 0.0178 Acc: 0.5428

Epoch 10/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9690
val Loss: 0.0181 Acc: 0.5428

Epoch 11/85
----------
train Loss: 0.0031 Acc: 0.9542
val Loss: 0.0178 Acc: 0.5390

Epoch 12/85
----------
train Loss: 0.0027 Acc: 0.9579
val Loss: 0.0183 Acc: 0.5428

Epoch 13/85
----------
train Loss: 0.0025 Acc: 0.9579
val Loss: 0.0183 Acc: 0.5428

Epoch 14/85
----------
train Loss: 0.0022 Acc: 0.9665
val Loss: 0.0183 Acc: 0.5353

Epoch 15/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0019 Acc: 0.9665
val Loss: 0.0182 Acc: 0.5428

Epoch 16/85
----------
train Loss: 0.0024 Acc: 0.9641
val Loss: 0.0182 Acc: 0.5502

Epoch 17/85
----------
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0182 Acc: 0.5502

Epoch 18/85
----------
train Loss: 0.0027 Acc: 0.9653
val Loss: 0.0186 Acc: 0.5353

Epoch 19/85
----------
train Loss: 0.0026 Acc: 0.9653
val Loss: 0.0179 Acc: 0.5502

Epoch 20/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0022 Acc: 0.9641
val Loss: 0.0183 Acc: 0.5428

Epoch 21/85
----------
train Loss: 0.0020 Acc: 0.9616
val Loss: 0.0186 Acc: 0.5465

Epoch 22/85
----------
train Loss: 0.0041 Acc: 0.9591
val Loss: 0.0182 Acc: 0.5390

Epoch 23/85
----------
train Loss: 0.0025 Acc: 0.9665
val Loss: 0.0186 Acc: 0.5353

Epoch 24/85
----------
train Loss: 0.0022 Acc: 0.9628
val Loss: 0.0182 Acc: 0.5576

Epoch 25/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0020 Acc: 0.9703
val Loss: 0.0182 Acc: 0.5465

Epoch 26/85
----------
train Loss: 0.0026 Acc: 0.9641
val Loss: 0.0183 Acc: 0.5539

Epoch 27/85
----------
train Loss: 0.0021 Acc: 0.9603
val Loss: 0.0181 Acc: 0.5390

Epoch 28/85
----------
train Loss: 0.0024 Acc: 0.9665
val Loss: 0.0185 Acc: 0.5576

Epoch 29/85
----------
train Loss: 0.0022 Acc: 0.9665
val Loss: 0.0185 Acc: 0.5539

Epoch 30/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0018 Acc: 0.9690
val Loss: 0.0179 Acc: 0.5576

Epoch 31/85
----------
train Loss: 0.0038 Acc: 0.9603
val Loss: 0.0188 Acc: 0.5428

Epoch 32/85
----------
train Loss: 0.0021 Acc: 0.9678
val Loss: 0.0179 Acc: 0.5576

Epoch 33/85
----------
train Loss: 0.0020 Acc: 0.9727
val Loss: 0.0182 Acc: 0.5353

Epoch 34/85
----------
train Loss: 0.0021 Acc: 0.9628
val Loss: 0.0181 Acc: 0.5390

Epoch 35/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0019 Acc: 0.9603
val Loss: 0.0182 Acc: 0.5502

Epoch 36/85
----------
train Loss: 0.0024 Acc: 0.9703
val Loss: 0.0181 Acc: 0.5353

Epoch 37/85
----------
train Loss: 0.0027 Acc: 0.9603
val Loss: 0.0181 Acc: 0.5576

Epoch 38/85
----------
train Loss: 0.0021 Acc: 0.9603
val Loss: 0.0185 Acc: 0.5576

Epoch 39/85
----------
train Loss: 0.0025 Acc: 0.9727
val Loss: 0.0182 Acc: 0.5613

Epoch 40/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0030 Acc: 0.9665
val Loss: 0.0184 Acc: 0.5539

Epoch 41/85
----------
train Loss: 0.0019 Acc: 0.9665
val Loss: 0.0180 Acc: 0.5390

Epoch 42/85
----------
train Loss: 0.0024 Acc: 0.9678
val Loss: 0.0181 Acc: 0.5390

Epoch 43/85
----------
train Loss: 0.0018 Acc: 0.9678
val Loss: 0.0179 Acc: 0.5688

Epoch 44/85
----------
train Loss: 0.0029 Acc: 0.9653
val Loss: 0.0186 Acc: 0.5576

Epoch 45/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0020 Acc: 0.9603
val Loss: 0.0183 Acc: 0.5539

Epoch 46/85
----------
train Loss: 0.0022 Acc: 0.9678
val Loss: 0.0181 Acc: 0.5502

Epoch 47/85
----------
train Loss: 0.0019 Acc: 0.9703
val Loss: 0.0180 Acc: 0.5465

Epoch 48/85
----------
train Loss: 0.0020 Acc: 0.9616
val Loss: 0.0178 Acc: 0.5613

Epoch 49/85
----------
train Loss: 0.0021 Acc: 0.9665
val Loss: 0.0183 Acc: 0.5539

Epoch 50/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0022 Acc: 0.9579
val Loss: 0.0180 Acc: 0.5613

Epoch 51/85
----------
train Loss: 0.0036 Acc: 0.9603
val Loss: 0.0184 Acc: 0.5539

Epoch 52/85
----------
train Loss: 0.0020 Acc: 0.9703
val Loss: 0.0180 Acc: 0.5576

Epoch 53/85
----------
train Loss: 0.0026 Acc: 0.9678
val Loss: 0.0182 Acc: 0.5651

Epoch 54/85
----------
train Loss: 0.0022 Acc: 0.9678
val Loss: 0.0182 Acc: 0.5539

Epoch 55/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0019 Acc: 0.9628
val Loss: 0.0180 Acc: 0.5502

Epoch 56/85
----------
train Loss: 0.0046 Acc: 0.9603
val Loss: 0.0182 Acc: 0.5651

Epoch 57/85
----------
train Loss: 0.0023 Acc: 0.9653
val Loss: 0.0185 Acc: 0.5688

Epoch 58/85
----------
train Loss: 0.0023 Acc: 0.9665
val Loss: 0.0182 Acc: 0.5502

Epoch 59/85
----------
train Loss: 0.0026 Acc: 0.9641
val Loss: 0.0186 Acc: 0.5465

Epoch 60/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0023 Acc: 0.9678
val Loss: 0.0182 Acc: 0.5316

Epoch 61/85
----------
train Loss: 0.0023 Acc: 0.9616
val Loss: 0.0177 Acc: 0.5390

Epoch 62/85
----------
train Loss: 0.0029 Acc: 0.9665
val Loss: 0.0181 Acc: 0.5353

Epoch 63/85
----------
train Loss: 0.0021 Acc: 0.9727
val Loss: 0.0181 Acc: 0.5390

Epoch 64/85
----------
train Loss: 0.0025 Acc: 0.9665
val Loss: 0.0181 Acc: 0.5502

Epoch 65/85
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0025 Acc: 0.9653
val Loss: 0.0185 Acc: 0.5465

Epoch 66/85
----------
train Loss: 0.0030 Acc: 0.9678
val Loss: 0.0182 Acc: 0.5465

Epoch 67/85
----------
train Loss: 0.0029 Acc: 0.9678
val Loss: 0.0185 Acc: 0.5465

Epoch 68/85
----------
train Loss: 0.0023 Acc: 0.9653
val Loss: 0.0181 Acc: 0.5539

Epoch 69/85
----------
train Loss: 0.0024 Acc: 0.9665
val Loss: 0.0182 Acc: 0.5539

Epoch 70/85
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0024 Acc: 0.9628
val Loss: 0.0184 Acc: 0.5613

Epoch 71/85
----------
train Loss: 0.0028 Acc: 0.9678
val Loss: 0.0181 Acc: 0.5576

Epoch 72/85
----------
train Loss: 0.0025 Acc: 0.9616
val Loss: 0.0187 Acc: 0.5465

Epoch 73/85
----------
train Loss: 0.0022 Acc: 0.9641
val Loss: 0.0183 Acc: 0.5465

Epoch 74/85
----------
train Loss: 0.0022 Acc: 0.9628
val Loss: 0.0184 Acc: 0.5539

Epoch 75/85
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0024 Acc: 0.9628
val Loss: 0.0178 Acc: 0.5576

Epoch 76/85
----------
train Loss: 0.0024 Acc: 0.9703
val Loss: 0.0180 Acc: 0.5465

Epoch 77/85
----------
train Loss: 0.0023 Acc: 0.9653
val Loss: 0.0186 Acc: 0.5428

Epoch 78/85
----------
train Loss: 0.0019 Acc: 0.9653
val Loss: 0.0182 Acc: 0.5353

Epoch 79/85
----------
train Loss: 0.0036 Acc: 0.9628
val Loss: 0.0185 Acc: 0.5651

Epoch 80/85
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0035 Acc: 0.9653
val Loss: 0.0189 Acc: 0.5613

Epoch 81/85
----------
train Loss: 0.0024 Acc: 0.9690
val Loss: 0.0181 Acc: 0.5576

Epoch 82/85
----------
train Loss: 0.0021 Acc: 0.9641
val Loss: 0.0180 Acc: 0.5539

Epoch 83/85
----------
train Loss: 0.0020 Acc: 0.9690
val Loss: 0.0182 Acc: 0.5613

Epoch 84/85
----------
train Loss: 0.0024 Acc: 0.9616
val Loss: 0.0178 Acc: 0.5576

Epoch 85/85
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0179 Acc: 0.5539

Training complete in 8m 22s
Best val Acc: 0.568773

---Testing---
Test accuracy: 0.868959
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 74 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 62 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 77 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8140069050382049, std: 0.1450216317870659
--------------------

run info[val: 0.3, epoch: 72, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0282 Acc: 0.1273
val Loss: 0.0321 Acc: 0.1491

Epoch 1/71
----------
train Loss: 0.0240 Acc: 0.2361
val Loss: 0.0280 Acc: 0.2640

Epoch 2/71
----------
train Loss: 0.0207 Acc: 0.3594
val Loss: 0.0237 Acc: 0.3571

Epoch 3/71
----------
LR is set to 0.001
train Loss: 0.0172 Acc: 0.4867
val Loss: 0.0232 Acc: 0.3602

Epoch 4/71
----------
train Loss: 0.0164 Acc: 0.5345
val Loss: 0.0237 Acc: 0.3696

Epoch 5/71
----------
train Loss: 0.0162 Acc: 0.5517
val Loss: 0.0231 Acc: 0.3727

Epoch 6/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0161 Acc: 0.5451
val Loss: 0.0226 Acc: 0.3758

Epoch 7/71
----------
train Loss: 0.0162 Acc: 0.5318
val Loss: 0.0237 Acc: 0.3758

Epoch 8/71
----------
train Loss: 0.0160 Acc: 0.5531
val Loss: 0.0233 Acc: 0.3727

Epoch 9/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0161 Acc: 0.5504
val Loss: 0.0232 Acc: 0.3727

Epoch 10/71
----------
train Loss: 0.0162 Acc: 0.5517
val Loss: 0.0234 Acc: 0.3727

Epoch 11/71
----------
train Loss: 0.0162 Acc: 0.5504
val Loss: 0.0236 Acc: 0.3727

Epoch 12/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0159 Acc: 0.5292
val Loss: 0.0245 Acc: 0.3696

Epoch 13/71
----------
train Loss: 0.0162 Acc: 0.5451
val Loss: 0.0235 Acc: 0.3727

Epoch 14/71
----------
train Loss: 0.0160 Acc: 0.5743
val Loss: 0.0237 Acc: 0.3758

Epoch 15/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0161 Acc: 0.5557
val Loss: 0.0227 Acc: 0.3727

Epoch 16/71
----------
train Loss: 0.0160 Acc: 0.5504
val Loss: 0.0225 Acc: 0.3696

Epoch 17/71
----------
train Loss: 0.0162 Acc: 0.5477
val Loss: 0.0238 Acc: 0.3758

Epoch 18/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0161 Acc: 0.5544
val Loss: 0.0240 Acc: 0.3758

Epoch 19/71
----------
train Loss: 0.0160 Acc: 0.5451
val Loss: 0.0233 Acc: 0.3727

Epoch 20/71
----------
train Loss: 0.0161 Acc: 0.5584
val Loss: 0.0234 Acc: 0.3727

Epoch 21/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0160 Acc: 0.5358
val Loss: 0.0244 Acc: 0.3727

Epoch 22/71
----------
train Loss: 0.0160 Acc: 0.5438
val Loss: 0.0232 Acc: 0.3758

Epoch 23/71
----------
train Loss: 0.0162 Acc: 0.5531
val Loss: 0.0225 Acc: 0.3727

Epoch 24/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0162 Acc: 0.5464
val Loss: 0.0236 Acc: 0.3727

Epoch 25/71
----------
train Loss: 0.0161 Acc: 0.5531
val Loss: 0.0233 Acc: 0.3758

Epoch 26/71
----------
train Loss: 0.0160 Acc: 0.5570
val Loss: 0.0241 Acc: 0.3758

Epoch 27/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0161 Acc: 0.5544
val Loss: 0.0248 Acc: 0.3758

Epoch 28/71
----------
train Loss: 0.0161 Acc: 0.5584
val Loss: 0.0235 Acc: 0.3727

Epoch 29/71
----------
train Loss: 0.0161 Acc: 0.5424
val Loss: 0.0230 Acc: 0.3696

Epoch 30/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0161 Acc: 0.5517
val Loss: 0.0224 Acc: 0.3758

Epoch 31/71
----------
train Loss: 0.0160 Acc: 0.5517
val Loss: 0.0233 Acc: 0.3727

Epoch 32/71
----------
train Loss: 0.0162 Acc: 0.5504
val Loss: 0.0223 Acc: 0.3696

Epoch 33/71
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0161 Acc: 0.5544
val Loss: 0.0238 Acc: 0.3758

Epoch 34/71
----------
train Loss: 0.0160 Acc: 0.5464
val Loss: 0.0241 Acc: 0.3758

Epoch 35/71
----------
train Loss: 0.0160 Acc: 0.5544
val Loss: 0.0243 Acc: 0.3758

Epoch 36/71
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0160 Acc: 0.5424
val Loss: 0.0234 Acc: 0.3727

Epoch 37/71
----------
train Loss: 0.0162 Acc: 0.5491
val Loss: 0.0222 Acc: 0.3727

Epoch 38/71
----------
train Loss: 0.0159 Acc: 0.5557
val Loss: 0.0229 Acc: 0.3696

Epoch 39/71
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0161 Acc: 0.5517
val Loss: 0.0237 Acc: 0.3727

Epoch 40/71
----------
train Loss: 0.0162 Acc: 0.5279
val Loss: 0.0236 Acc: 0.3727

Epoch 41/71
----------
train Loss: 0.0161 Acc: 0.5504
val Loss: 0.0241 Acc: 0.3727

Epoch 42/71
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0161 Acc: 0.5345
val Loss: 0.0226 Acc: 0.3758

Epoch 43/71
----------
train Loss: 0.0161 Acc: 0.5517
val Loss: 0.0233 Acc: 0.3758

Epoch 44/71
----------
train Loss: 0.0161 Acc: 0.5438
val Loss: 0.0228 Acc: 0.3727

Epoch 45/71
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0161 Acc: 0.5438
val Loss: 0.0229 Acc: 0.3727

Epoch 46/71
----------
train Loss: 0.0161 Acc: 0.5438
val Loss: 0.0225 Acc: 0.3727

Epoch 47/71
----------
train Loss: 0.0161 Acc: 0.5438
val Loss: 0.0230 Acc: 0.3727

Epoch 48/71
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0161 Acc: 0.5570
val Loss: 0.0236 Acc: 0.3727

Epoch 49/71
----------
train Loss: 0.0161 Acc: 0.5477
val Loss: 0.0237 Acc: 0.3727

Epoch 50/71
----------
train Loss: 0.0159 Acc: 0.5557
val Loss: 0.0239 Acc: 0.3727

Epoch 51/71
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0160 Acc: 0.5570
val Loss: 0.0228 Acc: 0.3758

Epoch 52/71
----------
train Loss: 0.0163 Acc: 0.5424
val Loss: 0.0228 Acc: 0.3727

Epoch 53/71
----------
train Loss: 0.0160 Acc: 0.5438
val Loss: 0.0238 Acc: 0.3758

Epoch 54/71
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0162 Acc: 0.5451
val Loss: 0.0234 Acc: 0.3696

Epoch 55/71
----------
train Loss: 0.0160 Acc: 0.5477
val Loss: 0.0226 Acc: 0.3727

Epoch 56/71
----------
train Loss: 0.0160 Acc: 0.5345
val Loss: 0.0229 Acc: 0.3727

Epoch 57/71
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0163 Acc: 0.5424
val Loss: 0.0239 Acc: 0.3727

Epoch 58/71
----------
train Loss: 0.0161 Acc: 0.5491
val Loss: 0.0224 Acc: 0.3727

Epoch 59/71
----------
train Loss: 0.0161 Acc: 0.5584
val Loss: 0.0228 Acc: 0.3727

Epoch 60/71
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0162 Acc: 0.5265
val Loss: 0.0246 Acc: 0.3696

Epoch 61/71
----------
train Loss: 0.0159 Acc: 0.5570
val Loss: 0.0230 Acc: 0.3696

Epoch 62/71
----------
train Loss: 0.0161 Acc: 0.5610
val Loss: 0.0229 Acc: 0.3727

Epoch 63/71
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0161 Acc: 0.5584
val Loss: 0.0242 Acc: 0.3696

Epoch 64/71
----------
train Loss: 0.0161 Acc: 0.5504
val Loss: 0.0235 Acc: 0.3727

Epoch 65/71
----------
train Loss: 0.0161 Acc: 0.5544
val Loss: 0.0231 Acc: 0.3758

Epoch 66/71
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0161 Acc: 0.5491
val Loss: 0.0227 Acc: 0.3758

Epoch 67/71
----------
train Loss: 0.0160 Acc: 0.5491
val Loss: 0.0243 Acc: 0.3696

Epoch 68/71
----------
train Loss: 0.0161 Acc: 0.5491
val Loss: 0.0224 Acc: 0.3727

Epoch 69/71
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0160 Acc: 0.5650
val Loss: 0.0234 Acc: 0.3758

Epoch 70/71
----------
train Loss: 0.0160 Acc: 0.5477
val Loss: 0.0240 Acc: 0.3758

Epoch 71/71
----------
train Loss: 0.0161 Acc: 0.5623
val Loss: 0.0222 Acc: 0.3758

Training complete in 6m 39s
Best val Acc: 0.375776

---Fine tuning.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0160 Acc: 0.5345
val Loss: 0.0213 Acc: 0.4255

Epoch 1/71
----------
train Loss: 0.0109 Acc: 0.7162
val Loss: 0.0194 Acc: 0.4907

Epoch 2/71
----------
train Loss: 0.0069 Acc: 0.8395
val Loss: 0.0194 Acc: 0.5031

Epoch 3/71
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.9085
val Loss: 0.0182 Acc: 0.5155

Epoch 4/71
----------
train Loss: 0.0039 Acc: 0.9297
val Loss: 0.0166 Acc: 0.5217

Epoch 5/71
----------
train Loss: 0.0034 Acc: 0.9523
val Loss: 0.0171 Acc: 0.5373

Epoch 6/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0170 Acc: 0.5435

Epoch 7/71
----------
train Loss: 0.0035 Acc: 0.9403
val Loss: 0.0170 Acc: 0.5466

Epoch 8/71
----------
train Loss: 0.0032 Acc: 0.9576
val Loss: 0.0169 Acc: 0.5435

Epoch 9/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0171 Acc: 0.5466

Epoch 10/71
----------
train Loss: 0.0033 Acc: 0.9523
val Loss: 0.0173 Acc: 0.5497

Epoch 11/71
----------
train Loss: 0.0032 Acc: 0.9562
val Loss: 0.0174 Acc: 0.5466

Epoch 12/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.9509
val Loss: 0.0178 Acc: 0.5435

Epoch 13/71
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0175 Acc: 0.5404

Epoch 14/71
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0186 Acc: 0.5435

Epoch 15/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0177 Acc: 0.5435

Epoch 16/71
----------
train Loss: 0.0033 Acc: 0.9483
val Loss: 0.0174 Acc: 0.5435

Epoch 17/71
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0179 Acc: 0.5466

Epoch 18/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.9629
val Loss: 0.0187 Acc: 0.5435

Epoch 19/71
----------
train Loss: 0.0034 Acc: 0.9483
val Loss: 0.0167 Acc: 0.5435

Epoch 20/71
----------
train Loss: 0.0032 Acc: 0.9576
val Loss: 0.0175 Acc: 0.5466

Epoch 21/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9615
val Loss: 0.0162 Acc: 0.5466

Epoch 22/71
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0190 Acc: 0.5435

Epoch 23/71
----------
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0180 Acc: 0.5466

Epoch 24/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0031 Acc: 0.9536
val Loss: 0.0169 Acc: 0.5435

Epoch 25/71
----------
train Loss: 0.0033 Acc: 0.9536
val Loss: 0.0181 Acc: 0.5435

Epoch 26/71
----------
train Loss: 0.0033 Acc: 0.9509
val Loss: 0.0167 Acc: 0.5466

Epoch 27/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0173 Acc: 0.5466

Epoch 28/71
----------
train Loss: 0.0033 Acc: 0.9469
val Loss: 0.0176 Acc: 0.5435

Epoch 29/71
----------
train Loss: 0.0033 Acc: 0.9523
val Loss: 0.0182 Acc: 0.5466

Epoch 30/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0031 Acc: 0.9509
val Loss: 0.0171 Acc: 0.5435

Epoch 31/71
----------
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0168 Acc: 0.5466

Epoch 32/71
----------
train Loss: 0.0033 Acc: 0.9509
val Loss: 0.0173 Acc: 0.5466

Epoch 33/71
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0175 Acc: 0.5404

Epoch 34/71
----------
train Loss: 0.0033 Acc: 0.9562
val Loss: 0.0165 Acc: 0.5435

Epoch 35/71
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0174 Acc: 0.5435

Epoch 36/71
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0032 Acc: 0.9615
val Loss: 0.0183 Acc: 0.5435

Epoch 37/71
----------
train Loss: 0.0033 Acc: 0.9549
val Loss: 0.0182 Acc: 0.5435

Epoch 38/71
----------
train Loss: 0.0031 Acc: 0.9536
val Loss: 0.0171 Acc: 0.5466

Epoch 39/71
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0176 Acc: 0.5435

Epoch 40/71
----------
train Loss: 0.0033 Acc: 0.9416
val Loss: 0.0166 Acc: 0.5466

Epoch 41/71
----------
train Loss: 0.0033 Acc: 0.9496
val Loss: 0.0163 Acc: 0.5435

Epoch 42/71
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0033 Acc: 0.9443
val Loss: 0.0176 Acc: 0.5404

Epoch 43/71
----------
train Loss: 0.0032 Acc: 0.9602
val Loss: 0.0188 Acc: 0.5404

Epoch 44/71
----------
train Loss: 0.0033 Acc: 0.9469
val Loss: 0.0171 Acc: 0.5435

Epoch 45/71
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0033 Acc: 0.9469
val Loss: 0.0168 Acc: 0.5435

Epoch 46/71
----------
train Loss: 0.0034 Acc: 0.9496
val Loss: 0.0183 Acc: 0.5466

Epoch 47/71
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0170 Acc: 0.5435

Epoch 48/71
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0033 Acc: 0.9483
val Loss: 0.0173 Acc: 0.5466

Epoch 49/71
----------
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0180 Acc: 0.5528

Epoch 50/71
----------
train Loss: 0.0033 Acc: 0.9576
val Loss: 0.0165 Acc: 0.5404

Epoch 51/71
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0033 Acc: 0.9523
val Loss: 0.0184 Acc: 0.5404

Epoch 52/71
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0180 Acc: 0.5435

Epoch 53/71
----------
train Loss: 0.0033 Acc: 0.9576
val Loss: 0.0178 Acc: 0.5435

Epoch 54/71
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0033 Acc: 0.9483
val Loss: 0.0176 Acc: 0.5435

Epoch 55/71
----------
train Loss: 0.0033 Acc: 0.9443
val Loss: 0.0168 Acc: 0.5404

Epoch 56/71
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0176 Acc: 0.5435

Epoch 57/71
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0166 Acc: 0.5435

Epoch 58/71
----------
train Loss: 0.0033 Acc: 0.9576
val Loss: 0.0175 Acc: 0.5435

Epoch 59/71
----------
train Loss: 0.0031 Acc: 0.9602
val Loss: 0.0177 Acc: 0.5435

Epoch 60/71
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0034 Acc: 0.9456
val Loss: 0.0177 Acc: 0.5466

Epoch 61/71
----------
train Loss: 0.0033 Acc: 0.9416
val Loss: 0.0182 Acc: 0.5466

Epoch 62/71
----------
train Loss: 0.0033 Acc: 0.9443
val Loss: 0.0170 Acc: 0.5435

Epoch 63/71
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0174 Acc: 0.5404

Epoch 64/71
----------
train Loss: 0.0032 Acc: 0.9562
val Loss: 0.0177 Acc: 0.5466

Epoch 65/71
----------
train Loss: 0.0033 Acc: 0.9496
val Loss: 0.0176 Acc: 0.5435

Epoch 66/71
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0033 Acc: 0.9562
val Loss: 0.0167 Acc: 0.5404

Epoch 67/71
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0177 Acc: 0.5404

Epoch 68/71
----------
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0174 Acc: 0.5466

Epoch 69/71
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0034 Acc: 0.9483
val Loss: 0.0177 Acc: 0.5435

Epoch 70/71
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0183 Acc: 0.5466

Epoch 71/71
----------
train Loss: 0.0033 Acc: 0.9562
val Loss: 0.0171 Acc: 0.5466

Training complete in 7m 4s
Best val Acc: 0.552795

---Testing---
Test accuracy: 0.835502
--------------------
Accuracy of Albacore tuna : 81 %
Accuracy of Atlantic bluefin tuna : 73 %
Accuracy of Bigeye tuna : 68 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 85 %
Accuracy of Longtail tuna : 92 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 67 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna : 14 %
Accuracy of Southern bluefin tuna : 73 %
Accuracy of Yellowfin tuna : 94 %
mean: 0.7645644821285458, std: 0.19549055800605086

Model saved in "./weights/tuna_fish_[0.95]_mean[0.93]_std[0.06].save".
--------------------

run info[val: 0.1, epoch: 90, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0262 Acc: 0.1703
val Loss: 0.0399 Acc: 0.3458

Epoch 1/89
----------
train Loss: 0.0212 Acc: 0.3457
val Loss: 0.0425 Acc: 0.3551

Epoch 2/89
----------
train Loss: 0.0178 Acc: 0.4489
val Loss: 0.0267 Acc: 0.4299

Epoch 3/89
----------
train Loss: 0.0153 Acc: 0.5490
val Loss: 0.0320 Acc: 0.4299

Epoch 4/89
----------
train Loss: 0.0142 Acc: 0.5810
val Loss: 0.0320 Acc: 0.4206

Epoch 5/89
----------
train Loss: 0.0130 Acc: 0.6140
val Loss: 0.0316 Acc: 0.4860

Epoch 6/89
----------
train Loss: 0.0123 Acc: 0.6233
val Loss: 0.0372 Acc: 0.4393

Epoch 7/89
----------
train Loss: 0.0116 Acc: 0.6749
val Loss: 0.0349 Acc: 0.5140

Epoch 8/89
----------
train Loss: 0.0109 Acc: 0.6842
val Loss: 0.0254 Acc: 0.4953

Epoch 9/89
----------
train Loss: 0.0108 Acc: 0.6729
val Loss: 0.0296 Acc: 0.5140

Epoch 10/89
----------
train Loss: 0.0102 Acc: 0.6997
val Loss: 0.0252 Acc: 0.5140

Epoch 11/89
----------
train Loss: 0.0099 Acc: 0.7110
val Loss: 0.0331 Acc: 0.5421

Epoch 12/89
----------
LR is set to 0.001
train Loss: 0.0095 Acc: 0.7172
val Loss: 0.0298 Acc: 0.5140

Epoch 13/89
----------
train Loss: 0.0089 Acc: 0.7606
val Loss: 0.0264 Acc: 0.5047

Epoch 14/89
----------
train Loss: 0.0088 Acc: 0.7678
val Loss: 0.0299 Acc: 0.5140

Epoch 15/89
----------
train Loss: 0.0090 Acc: 0.7688
val Loss: 0.0227 Acc: 0.5140

Epoch 16/89
----------
train Loss: 0.0087 Acc: 0.7740
val Loss: 0.0343 Acc: 0.5140

Epoch 17/89
----------
train Loss: 0.0087 Acc: 0.7761
val Loss: 0.0305 Acc: 0.5047

Epoch 18/89
----------
train Loss: 0.0089 Acc: 0.7554
val Loss: 0.0210 Acc: 0.5047

Epoch 19/89
----------
train Loss: 0.0088 Acc: 0.7740
val Loss: 0.0350 Acc: 0.5140

Epoch 20/89
----------
train Loss: 0.0087 Acc: 0.7874
val Loss: 0.0358 Acc: 0.5140

Epoch 21/89
----------
train Loss: 0.0086 Acc: 0.7761
val Loss: 0.0301 Acc: 0.5140

Epoch 22/89
----------
train Loss: 0.0087 Acc: 0.7740
val Loss: 0.0243 Acc: 0.5047

Epoch 23/89
----------
train Loss: 0.0086 Acc: 0.7761
val Loss: 0.0330 Acc: 0.5047

Epoch 24/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0085 Acc: 0.7730
val Loss: 0.0266 Acc: 0.5047

Epoch 25/89
----------
train Loss: 0.0085 Acc: 0.7853
val Loss: 0.0269 Acc: 0.5047

Epoch 26/89
----------
train Loss: 0.0085 Acc: 0.7864
val Loss: 0.0260 Acc: 0.5047

Epoch 27/89
----------
train Loss: 0.0086 Acc: 0.7843
val Loss: 0.0274 Acc: 0.5047

Epoch 28/89
----------
train Loss: 0.0086 Acc: 0.7864
val Loss: 0.0296 Acc: 0.5047

Epoch 29/89
----------
train Loss: 0.0086 Acc: 0.7822
val Loss: 0.0325 Acc: 0.5047

Epoch 30/89
----------
train Loss: 0.0085 Acc: 0.7781
val Loss: 0.0280 Acc: 0.5047

Epoch 31/89
----------
train Loss: 0.0085 Acc: 0.7864
val Loss: 0.0317 Acc: 0.5047

Epoch 32/89
----------
train Loss: 0.0085 Acc: 0.7833
val Loss: 0.0321 Acc: 0.5047

Epoch 33/89
----------
train Loss: 0.0084 Acc: 0.7843
val Loss: 0.0242 Acc: 0.5047

Epoch 34/89
----------
train Loss: 0.0083 Acc: 0.7967
val Loss: 0.0263 Acc: 0.5047

Epoch 35/89
----------
train Loss: 0.0087 Acc: 0.7719
val Loss: 0.0287 Acc: 0.5047

Epoch 36/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0085 Acc: 0.7771
val Loss: 0.0289 Acc: 0.5140

Epoch 37/89
----------
train Loss: 0.0085 Acc: 0.7884
val Loss: 0.0324 Acc: 0.5140

Epoch 38/89
----------
train Loss: 0.0083 Acc: 0.8008
val Loss: 0.0274 Acc: 0.5140

Epoch 39/89
----------
train Loss: 0.0085 Acc: 0.7750
val Loss: 0.0361 Acc: 0.5047

Epoch 40/89
----------
train Loss: 0.0085 Acc: 0.7688
val Loss: 0.0272 Acc: 0.5047

Epoch 41/89
----------
train Loss: 0.0085 Acc: 0.7853
val Loss: 0.0301 Acc: 0.5047

Epoch 42/89
----------
train Loss: 0.0087 Acc: 0.7802
val Loss: 0.0309 Acc: 0.5140

Epoch 43/89
----------
train Loss: 0.0084 Acc: 0.7864
val Loss: 0.0354 Acc: 0.5140

Epoch 44/89
----------
train Loss: 0.0085 Acc: 0.7822
val Loss: 0.0294 Acc: 0.5047

Epoch 45/89
----------
train Loss: 0.0085 Acc: 0.7874
val Loss: 0.0282 Acc: 0.5140

Epoch 46/89
----------
train Loss: 0.0085 Acc: 0.7781
val Loss: 0.0292 Acc: 0.5140

Epoch 47/89
----------
train Loss: 0.0085 Acc: 0.7884
val Loss: 0.0257 Acc: 0.5140

Epoch 48/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0086 Acc: 0.7915
val Loss: 0.0268 Acc: 0.5140

Epoch 49/89
----------
train Loss: 0.0084 Acc: 0.7998
val Loss: 0.0349 Acc: 0.5047

Epoch 50/89
----------
train Loss: 0.0084 Acc: 0.7792
val Loss: 0.0239 Acc: 0.5047

Epoch 51/89
----------
train Loss: 0.0085 Acc: 0.7833
val Loss: 0.0292 Acc: 0.5047

Epoch 52/89
----------
train Loss: 0.0084 Acc: 0.7905
val Loss: 0.0260 Acc: 0.5047

Epoch 53/89
----------
train Loss: 0.0083 Acc: 0.7874
val Loss: 0.0409 Acc: 0.5047

Epoch 54/89
----------
train Loss: 0.0084 Acc: 0.7833
val Loss: 0.0262 Acc: 0.5047

Epoch 55/89
----------
train Loss: 0.0087 Acc: 0.7761
val Loss: 0.0351 Acc: 0.5047

Epoch 56/89
----------
train Loss: 0.0086 Acc: 0.7822
val Loss: 0.0258 Acc: 0.5047

Epoch 57/89
----------
train Loss: 0.0085 Acc: 0.7802
val Loss: 0.0296 Acc: 0.5047

Epoch 58/89
----------
train Loss: 0.0085 Acc: 0.7812
val Loss: 0.0358 Acc: 0.5047

Epoch 59/89
----------
train Loss: 0.0085 Acc: 0.7905
val Loss: 0.0234 Acc: 0.5047

Epoch 60/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0086 Acc: 0.7792
val Loss: 0.0286 Acc: 0.5047

Epoch 61/89
----------
train Loss: 0.0085 Acc: 0.7833
val Loss: 0.0264 Acc: 0.5140

Epoch 62/89
----------
train Loss: 0.0083 Acc: 0.7792
val Loss: 0.0324 Acc: 0.5140

Epoch 63/89
----------
train Loss: 0.0084 Acc: 0.7843
val Loss: 0.0336 Acc: 0.5047

Epoch 64/89
----------
train Loss: 0.0084 Acc: 0.7905
val Loss: 0.0239 Acc: 0.5140

Epoch 65/89
----------
train Loss: 0.0084 Acc: 0.7915
val Loss: 0.0254 Acc: 0.5140

Epoch 66/89
----------
train Loss: 0.0083 Acc: 0.8029
val Loss: 0.0291 Acc: 0.5047

Epoch 67/89
----------
train Loss: 0.0085 Acc: 0.7833
val Loss: 0.0284 Acc: 0.5047

Epoch 68/89
----------
train Loss: 0.0083 Acc: 0.7915
val Loss: 0.0303 Acc: 0.5047

Epoch 69/89
----------
train Loss: 0.0084 Acc: 0.7936
val Loss: 0.0218 Acc: 0.5140

Epoch 70/89
----------
train Loss: 0.0084 Acc: 0.7874
val Loss: 0.0289 Acc: 0.5047

Epoch 71/89
----------
train Loss: 0.0086 Acc: 0.7853
val Loss: 0.0430 Acc: 0.5140

Epoch 72/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0087 Acc: 0.7781
val Loss: 0.0240 Acc: 0.5047

Epoch 73/89
----------
train Loss: 0.0085 Acc: 0.7853
val Loss: 0.0289 Acc: 0.5047

Epoch 74/89
----------
train Loss: 0.0085 Acc: 0.7812
val Loss: 0.0257 Acc: 0.5140

Epoch 75/89
----------
train Loss: 0.0086 Acc: 0.7843
val Loss: 0.0264 Acc: 0.5047

Epoch 76/89
----------
train Loss: 0.0083 Acc: 0.7833
val Loss: 0.0314 Acc: 0.5140

Epoch 77/89
----------
train Loss: 0.0085 Acc: 0.7853
val Loss: 0.0234 Acc: 0.5047

Epoch 78/89
----------
train Loss: 0.0084 Acc: 0.7864
val Loss: 0.0332 Acc: 0.5140

Epoch 79/89
----------
train Loss: 0.0085 Acc: 0.7915
val Loss: 0.0292 Acc: 0.5140

Epoch 80/89
----------
train Loss: 0.0084 Acc: 0.7905
val Loss: 0.0225 Acc: 0.5047

Epoch 81/89
----------
train Loss: 0.0084 Acc: 0.7781
val Loss: 0.0304 Acc: 0.5140

Epoch 82/89
----------
train Loss: 0.0085 Acc: 0.7895
val Loss: 0.0312 Acc: 0.5140

Epoch 83/89
----------
train Loss: 0.0084 Acc: 0.7915
val Loss: 0.0281 Acc: 0.5140

Epoch 84/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0084 Acc: 0.7874
val Loss: 0.0330 Acc: 0.5047

Epoch 85/89
----------
train Loss: 0.0086 Acc: 0.7657
val Loss: 0.0379 Acc: 0.5140

Epoch 86/89
----------
train Loss: 0.0084 Acc: 0.7864
val Loss: 0.0238 Acc: 0.5047

Epoch 87/89
----------
train Loss: 0.0085 Acc: 0.7905
val Loss: 0.0360 Acc: 0.5140

Epoch 88/89
----------
train Loss: 0.0085 Acc: 0.7822
val Loss: 0.0316 Acc: 0.5047

Epoch 89/89
----------
train Loss: 0.0085 Acc: 0.7761
val Loss: 0.0413 Acc: 0.5140

Training complete in 8m 21s
Best val Acc: 0.542056

---Fine tuning.---
Epoch 0/89
----------
LR is set to 0.01
train Loss: 0.0099 Acc: 0.7100
val Loss: 0.0210 Acc: 0.5047

Epoch 1/89
----------
train Loss: 0.0051 Acc: 0.8772
val Loss: 0.0265 Acc: 0.5234

Epoch 2/89
----------
train Loss: 0.0032 Acc: 0.9329
val Loss: 0.0224 Acc: 0.5607

Epoch 3/89
----------
train Loss: 0.0017 Acc: 0.9598
val Loss: 0.0317 Acc: 0.5327

Epoch 4/89
----------
train Loss: 0.0013 Acc: 0.9711
val Loss: 0.0261 Acc: 0.5701

Epoch 5/89
----------
train Loss: 0.0011 Acc: 0.9732
val Loss: 0.0277 Acc: 0.5421

Epoch 6/89
----------
train Loss: 0.0009 Acc: 0.9783
val Loss: 0.0334 Acc: 0.5327

Epoch 7/89
----------
train Loss: 0.0007 Acc: 0.9804
val Loss: 0.0246 Acc: 0.5514

Epoch 8/89
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0336 Acc: 0.5701

Epoch 9/89
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0236 Acc: 0.5794

Epoch 10/89
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0310 Acc: 0.5701

Epoch 11/89
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0275 Acc: 0.5981

Epoch 12/89
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9794
val Loss: 0.0313 Acc: 0.5981

Epoch 13/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0284 Acc: 0.5981

Epoch 14/89
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0432 Acc: 0.5888

Epoch 15/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0328 Acc: 0.5888

Epoch 16/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0317 Acc: 0.5794

Epoch 17/89
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0482 Acc: 0.5888

Epoch 18/89
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0401 Acc: 0.5794

Epoch 19/89
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0330 Acc: 0.5888

Epoch 20/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0268 Acc: 0.5981

Epoch 21/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0366 Acc: 0.5981

Epoch 22/89
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0194 Acc: 0.5981

Epoch 23/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0351 Acc: 0.6075

Epoch 24/89
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0490 Acc: 0.6075

Epoch 25/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0348 Acc: 0.5981

Epoch 26/89
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0469 Acc: 0.5981

Epoch 27/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0311 Acc: 0.5981

Epoch 28/89
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0338 Acc: 0.5981

Epoch 29/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0282 Acc: 0.5981

Epoch 30/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0297 Acc: 0.6075

Epoch 31/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0414 Acc: 0.5981

Epoch 32/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0305 Acc: 0.5794

Epoch 33/89
----------
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0239 Acc: 0.6075

Epoch 34/89
----------
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0251 Acc: 0.5981

Epoch 35/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0404 Acc: 0.5888

Epoch 36/89
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0274 Acc: 0.5981

Epoch 37/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0335 Acc: 0.5981

Epoch 38/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0221 Acc: 0.5981

Epoch 39/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0188 Acc: 0.5888

Epoch 40/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0326 Acc: 0.6075

Epoch 41/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0260 Acc: 0.5794

Epoch 42/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0308 Acc: 0.5888

Epoch 43/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0426 Acc: 0.6075

Epoch 44/89
----------
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0182 Acc: 0.6075

Epoch 45/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0257 Acc: 0.5701

Epoch 46/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0317 Acc: 0.5888

Epoch 47/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0374 Acc: 0.5888

Epoch 48/89
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0355 Acc: 0.5701

Epoch 49/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0327 Acc: 0.6075

Epoch 50/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0312 Acc: 0.5981

Epoch 51/89
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0334 Acc: 0.5981

Epoch 52/89
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0232 Acc: 0.5888

Epoch 53/89
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0277 Acc: 0.5794

Epoch 54/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0388 Acc: 0.5888

Epoch 55/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0342 Acc: 0.5888

Epoch 56/89
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0347 Acc: 0.5981

Epoch 57/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0357 Acc: 0.5888

Epoch 58/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0264 Acc: 0.5981

Epoch 59/89
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0294 Acc: 0.5981

Epoch 60/89
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0365 Acc: 0.5981

Epoch 61/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0300 Acc: 0.5888

Epoch 62/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0262 Acc: 0.5888

Epoch 63/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0290 Acc: 0.5888

Epoch 64/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0314 Acc: 0.5888

Epoch 65/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0368 Acc: 0.5794

Epoch 66/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0268 Acc: 0.5981

Epoch 67/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0277 Acc: 0.5888

Epoch 68/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0432 Acc: 0.6075

Epoch 69/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0279 Acc: 0.5981

Epoch 70/89
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0224 Acc: 0.5981

Epoch 71/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0306 Acc: 0.6075

Epoch 72/89
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0426 Acc: 0.5981

Epoch 73/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0465 Acc: 0.5981

Epoch 74/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0287 Acc: 0.5981

Epoch 75/89
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0416 Acc: 0.5701

Epoch 76/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0394 Acc: 0.5888

Epoch 77/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0282 Acc: 0.5981

Epoch 78/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0478 Acc: 0.5888

Epoch 79/89
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0327 Acc: 0.5794

Epoch 80/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0333 Acc: 0.5981

Epoch 81/89
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0337 Acc: 0.5981

Epoch 82/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0233 Acc: 0.5794

Epoch 83/89
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0412 Acc: 0.5888

Epoch 84/89
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0261 Acc: 0.5794

Epoch 85/89
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0262 Acc: 0.5888

Epoch 86/89
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0300 Acc: 0.5888

Epoch 87/89
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0285 Acc: 0.5888

Epoch 88/89
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0401 Acc: 0.5888

Epoch 89/89
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0400 Acc: 0.6075

Training complete in 8m 59s
Best val Acc: 0.607477

---Testing---
Test accuracy: 0.948885
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 89 %
Accuracy of Bigeye tuna : 89 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 92 %
Accuracy of Pacific bluefin tuna : 98 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 88 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.931022245203559, std: 0.05155254587486178
--------------------

run info[val: 0.15, epoch: 63, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/62
----------
LR is set to 0.01
train Loss: 0.0277 Acc: 0.1891
val Loss: 0.0281 Acc: 0.2919

Epoch 1/62
----------
train Loss: 0.0226 Acc: 0.3410
val Loss: 0.0249 Acc: 0.3540

Epoch 2/62
----------
train Loss: 0.0191 Acc: 0.4295
val Loss: 0.0251 Acc: 0.3602

Epoch 3/62
----------
train Loss: 0.0177 Acc: 0.4863
val Loss: 0.0249 Acc: 0.3354

Epoch 4/62
----------
LR is set to 0.001
train Loss: 0.0173 Acc: 0.4962
val Loss: 0.0222 Acc: 0.3851

Epoch 5/62
----------
train Loss: 0.0156 Acc: 0.5552
val Loss: 0.0219 Acc: 0.4037

Epoch 6/62
----------
train Loss: 0.0160 Acc: 0.5727
val Loss: 0.0221 Acc: 0.4224

Epoch 7/62
----------
train Loss: 0.0153 Acc: 0.5880
val Loss: 0.0221 Acc: 0.3975

Epoch 8/62
----------
LR is set to 0.00010000000000000002
train Loss: 0.0153 Acc: 0.5781
val Loss: 0.0218 Acc: 0.3789

Epoch 9/62
----------
train Loss: 0.0152 Acc: 0.5705
val Loss: 0.0216 Acc: 0.3913

Epoch 10/62
----------
train Loss: 0.0153 Acc: 0.5880
val Loss: 0.0219 Acc: 0.3913

Epoch 11/62
----------
train Loss: 0.0153 Acc: 0.5967
val Loss: 0.0213 Acc: 0.3975

Epoch 12/62
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0148 Acc: 0.5945
val Loss: 0.0219 Acc: 0.4099

Epoch 13/62
----------
train Loss: 0.0151 Acc: 0.5781
val Loss: 0.0222 Acc: 0.4037

Epoch 14/62
----------
train Loss: 0.0150 Acc: 0.5858
val Loss: 0.0222 Acc: 0.3913

Epoch 15/62
----------
train Loss: 0.0154 Acc: 0.5978
val Loss: 0.0219 Acc: 0.3975

Epoch 16/62
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0152 Acc: 0.5694
val Loss: 0.0221 Acc: 0.4037

Epoch 17/62
----------
train Loss: 0.0151 Acc: 0.5913
val Loss: 0.0217 Acc: 0.3975

Epoch 18/62
----------
train Loss: 0.0148 Acc: 0.5836
val Loss: 0.0215 Acc: 0.4037

Epoch 19/62
----------
train Loss: 0.0151 Acc: 0.5836
val Loss: 0.0219 Acc: 0.4161

Epoch 20/62
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0153 Acc: 0.5814
val Loss: 0.0220 Acc: 0.3975

Epoch 21/62
----------
train Loss: 0.0152 Acc: 0.5956
val Loss: 0.0217 Acc: 0.3913

Epoch 22/62
----------
train Loss: 0.0150 Acc: 0.5781
val Loss: 0.0216 Acc: 0.4099

Epoch 23/62
----------
train Loss: 0.0151 Acc: 0.5934
val Loss: 0.0224 Acc: 0.4037

Epoch 24/62
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0150 Acc: 0.5738
val Loss: 0.0225 Acc: 0.3913

Epoch 25/62
----------
train Loss: 0.0147 Acc: 0.6011
val Loss: 0.0217 Acc: 0.3975

Epoch 26/62
----------
train Loss: 0.0153 Acc: 0.5902
val Loss: 0.0220 Acc: 0.4037

Epoch 27/62
----------
train Loss: 0.0151 Acc: 0.5880
val Loss: 0.0222 Acc: 0.3913

Epoch 28/62
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0154 Acc: 0.5891
val Loss: 0.0217 Acc: 0.3975

Epoch 29/62
----------
train Loss: 0.0150 Acc: 0.5902
val Loss: 0.0221 Acc: 0.4037

Epoch 30/62
----------
train Loss: 0.0147 Acc: 0.5847
val Loss: 0.0220 Acc: 0.3913

Epoch 31/62
----------
train Loss: 0.0150 Acc: 0.5934
val Loss: 0.0221 Acc: 0.3975

Epoch 32/62
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0153 Acc: 0.5836
val Loss: 0.0220 Acc: 0.4099

Epoch 33/62
----------
train Loss: 0.0147 Acc: 0.5705
val Loss: 0.0219 Acc: 0.3975

Epoch 34/62
----------
train Loss: 0.0153 Acc: 0.5803
val Loss: 0.0217 Acc: 0.3913

Epoch 35/62
----------
train Loss: 0.0155 Acc: 0.5967
val Loss: 0.0218 Acc: 0.4161

Epoch 36/62
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0150 Acc: 0.5628
val Loss: 0.0221 Acc: 0.3975

Epoch 37/62
----------
train Loss: 0.0155 Acc: 0.5880
val Loss: 0.0222 Acc: 0.3975

Epoch 38/62
----------
train Loss: 0.0148 Acc: 0.5967
val Loss: 0.0220 Acc: 0.4099

Epoch 39/62
----------
train Loss: 0.0155 Acc: 0.5760
val Loss: 0.0212 Acc: 0.4037

Epoch 40/62
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0147 Acc: 0.6055
val Loss: 0.0227 Acc: 0.3851

Epoch 41/62
----------
train Loss: 0.0149 Acc: 0.5781
val Loss: 0.0216 Acc: 0.3913

Epoch 42/62
----------
train Loss: 0.0149 Acc: 0.5902
val Loss: 0.0219 Acc: 0.3975

Epoch 43/62
----------
train Loss: 0.0152 Acc: 0.5934
val Loss: 0.0221 Acc: 0.4037

Epoch 44/62
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0147 Acc: 0.5770
val Loss: 0.0223 Acc: 0.3975

Epoch 45/62
----------
train Loss: 0.0149 Acc: 0.5869
val Loss: 0.0222 Acc: 0.3975

Epoch 46/62
----------
train Loss: 0.0153 Acc: 0.5792
val Loss: 0.0222 Acc: 0.4099

Epoch 47/62
----------
train Loss: 0.0150 Acc: 0.5934
val Loss: 0.0214 Acc: 0.4037

Epoch 48/62
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0152 Acc: 0.5847
val Loss: 0.0217 Acc: 0.4161

Epoch 49/62
----------
train Loss: 0.0150 Acc: 0.6033
val Loss: 0.0216 Acc: 0.4099

Epoch 50/62
----------
train Loss: 0.0148 Acc: 0.5825
val Loss: 0.0218 Acc: 0.4161

Epoch 51/62
----------
train Loss: 0.0155 Acc: 0.5705
val Loss: 0.0220 Acc: 0.4099

Epoch 52/62
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0147 Acc: 0.5770
val Loss: 0.0221 Acc: 0.4037

Epoch 53/62
----------
train Loss: 0.0153 Acc: 0.5913
val Loss: 0.0218 Acc: 0.4037

Epoch 54/62
----------
train Loss: 0.0152 Acc: 0.5738
val Loss: 0.0217 Acc: 0.4037

Epoch 55/62
----------
train Loss: 0.0149 Acc: 0.5956
val Loss: 0.0220 Acc: 0.4224

Epoch 56/62
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0148 Acc: 0.5792
val Loss: 0.0216 Acc: 0.4099

Epoch 57/62
----------
train Loss: 0.0155 Acc: 0.5716
val Loss: 0.0218 Acc: 0.4099

Epoch 58/62
----------
train Loss: 0.0148 Acc: 0.5989
val Loss: 0.0221 Acc: 0.3913

Epoch 59/62
----------
train Loss: 0.0157 Acc: 0.5880
val Loss: 0.0218 Acc: 0.4099

Epoch 60/62
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0146 Acc: 0.6066
val Loss: 0.0218 Acc: 0.3975

Epoch 61/62
----------
train Loss: 0.0157 Acc: 0.5814
val Loss: 0.0219 Acc: 0.3975

Epoch 62/62
----------
train Loss: 0.0151 Acc: 0.5760
val Loss: 0.0219 Acc: 0.4161

Training complete in 5m 50s
Best val Acc: 0.422360

---Fine tuning.---
Epoch 0/62
----------
LR is set to 0.01
train Loss: 0.0149 Acc: 0.5749
val Loss: 0.0204 Acc: 0.4658

Epoch 1/62
----------
train Loss: 0.0111 Acc: 0.6809
val Loss: 0.0197 Acc: 0.4658

Epoch 2/62
----------
train Loss: 0.0078 Acc: 0.7749
val Loss: 0.0185 Acc: 0.5342

Epoch 3/62
----------
train Loss: 0.0060 Acc: 0.8383
val Loss: 0.0203 Acc: 0.5528

Epoch 4/62
----------
LR is set to 0.001
train Loss: 0.0043 Acc: 0.8874
val Loss: 0.0196 Acc: 0.5652

Epoch 5/62
----------
train Loss: 0.0038 Acc: 0.8995
val Loss: 0.0186 Acc: 0.5590

Epoch 6/62
----------
train Loss: 0.0036 Acc: 0.9235
val Loss: 0.0195 Acc: 0.5652

Epoch 7/62
----------
train Loss: 0.0031 Acc: 0.9301
val Loss: 0.0183 Acc: 0.5466

Epoch 8/62
----------
LR is set to 0.00010000000000000002
train Loss: 0.0029 Acc: 0.9454
val Loss: 0.0188 Acc: 0.5404

Epoch 9/62
----------
train Loss: 0.0033 Acc: 0.9355
val Loss: 0.0184 Acc: 0.5590

Epoch 10/62
----------
train Loss: 0.0029 Acc: 0.9399
val Loss: 0.0187 Acc: 0.5528

Epoch 11/62
----------
train Loss: 0.0030 Acc: 0.9246
val Loss: 0.0184 Acc: 0.5590

Epoch 12/62
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0030 Acc: 0.9290
val Loss: 0.0183 Acc: 0.5652

Epoch 13/62
----------
train Loss: 0.0032 Acc: 0.9421
val Loss: 0.0192 Acc: 0.5466

Epoch 14/62
----------
train Loss: 0.0028 Acc: 0.9333
val Loss: 0.0180 Acc: 0.5528

Epoch 15/62
----------
train Loss: 0.0031 Acc: 0.9333
val Loss: 0.0177 Acc: 0.5590

Epoch 16/62
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0030 Acc: 0.9388
val Loss: 0.0183 Acc: 0.5528

Epoch 17/62
----------
train Loss: 0.0031 Acc: 0.9311
val Loss: 0.0186 Acc: 0.5590

Epoch 18/62
----------
train Loss: 0.0035 Acc: 0.9268
val Loss: 0.0186 Acc: 0.5466

Epoch 19/62
----------
train Loss: 0.0031 Acc: 0.9202
val Loss: 0.0179 Acc: 0.5528

Epoch 20/62
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0029 Acc: 0.9410
val Loss: 0.0186 Acc: 0.5466

Epoch 21/62
----------
train Loss: 0.0030 Acc: 0.9301
val Loss: 0.0180 Acc: 0.5404

Epoch 22/62
----------
train Loss: 0.0029 Acc: 0.9399
val Loss: 0.0183 Acc: 0.5466

Epoch 23/62
----------
train Loss: 0.0028 Acc: 0.9366
val Loss: 0.0183 Acc: 0.5404

Epoch 24/62
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.9377
val Loss: 0.0190 Acc: 0.5466

Epoch 25/62
----------
train Loss: 0.0029 Acc: 0.9322
val Loss: 0.0189 Acc: 0.5466

Epoch 26/62
----------
train Loss: 0.0032 Acc: 0.9344
val Loss: 0.0184 Acc: 0.5466

Epoch 27/62
----------
train Loss: 0.0029 Acc: 0.9388
val Loss: 0.0190 Acc: 0.5404

Epoch 28/62
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9355
val Loss: 0.0185 Acc: 0.5528

Epoch 29/62
----------
train Loss: 0.0028 Acc: 0.9344
val Loss: 0.0187 Acc: 0.5528

Epoch 30/62
----------
train Loss: 0.0031 Acc: 0.9355
val Loss: 0.0185 Acc: 0.5466

Epoch 31/62
----------
train Loss: 0.0029 Acc: 0.9388
val Loss: 0.0182 Acc: 0.5466

Epoch 32/62
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0034 Acc: 0.9257
val Loss: 0.0185 Acc: 0.5404

Epoch 33/62
----------
train Loss: 0.0030 Acc: 0.9410
val Loss: 0.0186 Acc: 0.5528

Epoch 34/62
----------
train Loss: 0.0031 Acc: 0.9366
val Loss: 0.0191 Acc: 0.5466

Epoch 35/62
----------
train Loss: 0.0030 Acc: 0.9475
val Loss: 0.0190 Acc: 0.5466

Epoch 36/62
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0031 Acc: 0.9333
val Loss: 0.0193 Acc: 0.5466

Epoch 37/62
----------
train Loss: 0.0033 Acc: 0.9279
val Loss: 0.0184 Acc: 0.5528

Epoch 38/62
----------
train Loss: 0.0028 Acc: 0.9443
val Loss: 0.0184 Acc: 0.5466

Epoch 39/62
----------
train Loss: 0.0030 Acc: 0.9432
val Loss: 0.0182 Acc: 0.5466

Epoch 40/62
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9290
val Loss: 0.0186 Acc: 0.5466

Epoch 41/62
----------
train Loss: 0.0028 Acc: 0.9333
val Loss: 0.0179 Acc: 0.5528

Epoch 42/62
----------
train Loss: 0.0030 Acc: 0.9290
val Loss: 0.0188 Acc: 0.5466

Epoch 43/62
----------
train Loss: 0.0030 Acc: 0.9344
val Loss: 0.0181 Acc: 0.5528

Epoch 44/62
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0031 Acc: 0.9366
val Loss: 0.0182 Acc: 0.5528

Epoch 45/62
----------
train Loss: 0.0033 Acc: 0.9388
val Loss: 0.0194 Acc: 0.5466

Epoch 46/62
----------
train Loss: 0.0034 Acc: 0.9290
val Loss: 0.0192 Acc: 0.5466

Epoch 47/62
----------
train Loss: 0.0029 Acc: 0.9311
val Loss: 0.0184 Acc: 0.5652

Epoch 48/62
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0032 Acc: 0.9410
val Loss: 0.0186 Acc: 0.5652

Epoch 49/62
----------
train Loss: 0.0029 Acc: 0.9454
val Loss: 0.0183 Acc: 0.5466

Epoch 50/62
----------
train Loss: 0.0030 Acc: 0.9344
val Loss: 0.0193 Acc: 0.5466

Epoch 51/62
----------
train Loss: 0.0030 Acc: 0.9290
val Loss: 0.0181 Acc: 0.5528

Epoch 52/62
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0033 Acc: 0.9333
val Loss: 0.0181 Acc: 0.5404

Epoch 53/62
----------
train Loss: 0.0030 Acc: 0.9322
val Loss: 0.0185 Acc: 0.5404

Epoch 54/62
----------
train Loss: 0.0032 Acc: 0.9290
val Loss: 0.0190 Acc: 0.5404

Epoch 55/62
----------
train Loss: 0.0029 Acc: 0.9355
val Loss: 0.0186 Acc: 0.5466

Epoch 56/62
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0030 Acc: 0.9355
val Loss: 0.0190 Acc: 0.5528

Epoch 57/62
----------
train Loss: 0.0030 Acc: 0.9410
val Loss: 0.0184 Acc: 0.5466

Epoch 58/62
----------
train Loss: 0.0034 Acc: 0.9355
val Loss: 0.0192 Acc: 0.5466

Epoch 59/62
----------
train Loss: 0.0029 Acc: 0.9410
val Loss: 0.0190 Acc: 0.5466

Epoch 60/62
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0034 Acc: 0.9301
val Loss: 0.0189 Acc: 0.5466

Epoch 61/62
----------
train Loss: 0.0031 Acc: 0.9344
val Loss: 0.0196 Acc: 0.5466

Epoch 62/62
----------
train Loss: 0.0028 Acc: 0.9344
val Loss: 0.0187 Acc: 0.5528

Training complete in 6m 11s
Best val Acc: 0.565217

---Testing---
Test accuracy: 0.861524
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 53 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 88 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 72 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna : 35 %
Accuracy of Southern bluefin tuna : 77 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.7974053288790964, std: 0.17067152127432314
--------------------

run info[val: 0.2, epoch: 86, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0273 Acc: 0.1312
val Loss: 0.0325 Acc: 0.2512

Epoch 1/85
----------
train Loss: 0.0217 Acc: 0.3449
val Loss: 0.0269 Acc: 0.3581

Epoch 2/85
----------
train Loss: 0.0186 Acc: 0.4437
val Loss: 0.0281 Acc: 0.3953

Epoch 3/85
----------
train Loss: 0.0167 Acc: 0.4843
val Loss: 0.0243 Acc: 0.4186

Epoch 4/85
----------
train Loss: 0.0159 Acc: 0.5087
val Loss: 0.0239 Acc: 0.4419

Epoch 5/85
----------
train Loss: 0.0145 Acc: 0.5528
val Loss: 0.0244 Acc: 0.4186

Epoch 6/85
----------
train Loss: 0.0134 Acc: 0.6039
val Loss: 0.0238 Acc: 0.4279

Epoch 7/85
----------
train Loss: 0.0129 Acc: 0.6376
val Loss: 0.0254 Acc: 0.4558

Epoch 8/85
----------
train Loss: 0.0127 Acc: 0.6272
val Loss: 0.0244 Acc: 0.4698

Epoch 9/85
----------
train Loss: 0.0121 Acc: 0.6318
val Loss: 0.0238 Acc: 0.4698

Epoch 10/85
----------
LR is set to 0.001
train Loss: 0.0115 Acc: 0.6690
val Loss: 0.0235 Acc: 0.4791

Epoch 11/85
----------
train Loss: 0.0111 Acc: 0.6887
val Loss: 0.0227 Acc: 0.5116

Epoch 12/85
----------
train Loss: 0.0111 Acc: 0.6852
val Loss: 0.0218 Acc: 0.4930

Epoch 13/85
----------
train Loss: 0.0109 Acc: 0.6806
val Loss: 0.0245 Acc: 0.4837

Epoch 14/85
----------
train Loss: 0.0110 Acc: 0.6887
val Loss: 0.0219 Acc: 0.4977

Epoch 15/85
----------
train Loss: 0.0107 Acc: 0.7050
val Loss: 0.0256 Acc: 0.4791

Epoch 16/85
----------
train Loss: 0.0107 Acc: 0.7108
val Loss: 0.0233 Acc: 0.4884

Epoch 17/85
----------
train Loss: 0.0108 Acc: 0.7027
val Loss: 0.0259 Acc: 0.4930

Epoch 18/85
----------
train Loss: 0.0106 Acc: 0.6887
val Loss: 0.0244 Acc: 0.5070

Epoch 19/85
----------
train Loss: 0.0107 Acc: 0.7062
val Loss: 0.0205 Acc: 0.4977

Epoch 20/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0104 Acc: 0.7166
val Loss: 0.0209 Acc: 0.4977

Epoch 21/85
----------
train Loss: 0.0106 Acc: 0.7038
val Loss: 0.0204 Acc: 0.4884

Epoch 22/85
----------
train Loss: 0.0108 Acc: 0.6957
val Loss: 0.0257 Acc: 0.4930

Epoch 23/85
----------
train Loss: 0.0106 Acc: 0.7038
val Loss: 0.0235 Acc: 0.4884

Epoch 24/85
----------
train Loss: 0.0105 Acc: 0.7143
val Loss: 0.0231 Acc: 0.4884

Epoch 25/85
----------
train Loss: 0.0105 Acc: 0.7224
val Loss: 0.0261 Acc: 0.4884

Epoch 26/85
----------
train Loss: 0.0105 Acc: 0.7015
val Loss: 0.0241 Acc: 0.4930

Epoch 27/85
----------
train Loss: 0.0107 Acc: 0.7073
val Loss: 0.0241 Acc: 0.4930

Epoch 28/85
----------
train Loss: 0.0106 Acc: 0.7259
val Loss: 0.0216 Acc: 0.4930

Epoch 29/85
----------
train Loss: 0.0105 Acc: 0.7027
val Loss: 0.0224 Acc: 0.4977

Epoch 30/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0103 Acc: 0.7154
val Loss: 0.0218 Acc: 0.4884

Epoch 31/85
----------
train Loss: 0.0102 Acc: 0.7294
val Loss: 0.0221 Acc: 0.4977

Epoch 32/85
----------
train Loss: 0.0105 Acc: 0.7108
val Loss: 0.0198 Acc: 0.5023

Epoch 33/85
----------
train Loss: 0.0105 Acc: 0.6934
val Loss: 0.0250 Acc: 0.4930

Epoch 34/85
----------
train Loss: 0.0103 Acc: 0.7189
val Loss: 0.0246 Acc: 0.4977

Epoch 35/85
----------
train Loss: 0.0106 Acc: 0.6934
val Loss: 0.0215 Acc: 0.4977

Epoch 36/85
----------
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0223 Acc: 0.4930

Epoch 37/85
----------
train Loss: 0.0104 Acc: 0.7178
val Loss: 0.0263 Acc: 0.4930

Epoch 38/85
----------
train Loss: 0.0104 Acc: 0.7096
val Loss: 0.0240 Acc: 0.5023

Epoch 39/85
----------
train Loss: 0.0102 Acc: 0.7154
val Loss: 0.0244 Acc: 0.4930

Epoch 40/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0104 Acc: 0.7050
val Loss: 0.0243 Acc: 0.4884

Epoch 41/85
----------
train Loss: 0.0107 Acc: 0.7085
val Loss: 0.0221 Acc: 0.4837

Epoch 42/85
----------
train Loss: 0.0104 Acc: 0.7050
val Loss: 0.0202 Acc: 0.4884

Epoch 43/85
----------
train Loss: 0.0103 Acc: 0.7236
val Loss: 0.0215 Acc: 0.4930

Epoch 44/85
----------
train Loss: 0.0104 Acc: 0.7154
val Loss: 0.0217 Acc: 0.4884

Epoch 45/85
----------
train Loss: 0.0105 Acc: 0.7213
val Loss: 0.0249 Acc: 0.4977

Epoch 46/85
----------
train Loss: 0.0106 Acc: 0.6992
val Loss: 0.0244 Acc: 0.4977

Epoch 47/85
----------
train Loss: 0.0106 Acc: 0.7015
val Loss: 0.0232 Acc: 0.4930

Epoch 48/85
----------
train Loss: 0.0105 Acc: 0.7120
val Loss: 0.0234 Acc: 0.4930

Epoch 49/85
----------
train Loss: 0.0106 Acc: 0.7247
val Loss: 0.0216 Acc: 0.4930

Epoch 50/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0107 Acc: 0.6852
val Loss: 0.0211 Acc: 0.4977

Epoch 51/85
----------
train Loss: 0.0104 Acc: 0.7201
val Loss: 0.0219 Acc: 0.4977

Epoch 52/85
----------
train Loss: 0.0105 Acc: 0.7027
val Loss: 0.0230 Acc: 0.4930

Epoch 53/85
----------
train Loss: 0.0106 Acc: 0.6980
val Loss: 0.0259 Acc: 0.4884

Epoch 54/85
----------
train Loss: 0.0107 Acc: 0.7131
val Loss: 0.0226 Acc: 0.4837

Epoch 55/85
----------
train Loss: 0.0105 Acc: 0.7073
val Loss: 0.0234 Acc: 0.4884

Epoch 56/85
----------
train Loss: 0.0107 Acc: 0.7108
val Loss: 0.0205 Acc: 0.4930

Epoch 57/85
----------
train Loss: 0.0105 Acc: 0.7143
val Loss: 0.0245 Acc: 0.4930

Epoch 58/85
----------
train Loss: 0.0106 Acc: 0.7003
val Loss: 0.0254 Acc: 0.4977

Epoch 59/85
----------
train Loss: 0.0103 Acc: 0.7305
val Loss: 0.0228 Acc: 0.4930

Epoch 60/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0107 Acc: 0.7050
val Loss: 0.0202 Acc: 0.4884

Epoch 61/85
----------
train Loss: 0.0106 Acc: 0.7213
val Loss: 0.0232 Acc: 0.4977

Epoch 62/85
----------
train Loss: 0.0106 Acc: 0.7062
val Loss: 0.0219 Acc: 0.4977

Epoch 63/85
----------
train Loss: 0.0105 Acc: 0.7247
val Loss: 0.0235 Acc: 0.4977

Epoch 64/85
----------
train Loss: 0.0107 Acc: 0.6934
val Loss: 0.0213 Acc: 0.4884

Epoch 65/85
----------
train Loss: 0.0106 Acc: 0.7178
val Loss: 0.0257 Acc: 0.4930

Epoch 66/85
----------
train Loss: 0.0103 Acc: 0.6969
val Loss: 0.0245 Acc: 0.4930

Epoch 67/85
----------
train Loss: 0.0105 Acc: 0.7050
val Loss: 0.0248 Acc: 0.4930

Epoch 68/85
----------
train Loss: 0.0105 Acc: 0.6992
val Loss: 0.0221 Acc: 0.5070

Epoch 69/85
----------
train Loss: 0.0104 Acc: 0.7120
val Loss: 0.0207 Acc: 0.4977

Epoch 70/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0104 Acc: 0.7096
val Loss: 0.0223 Acc: 0.4977

Epoch 71/85
----------
train Loss: 0.0104 Acc: 0.6992
val Loss: 0.0219 Acc: 0.4977

Epoch 72/85
----------
train Loss: 0.0104 Acc: 0.7224
val Loss: 0.0245 Acc: 0.4977

Epoch 73/85
----------
train Loss: 0.0105 Acc: 0.7213
val Loss: 0.0236 Acc: 0.4930

Epoch 74/85
----------
train Loss: 0.0105 Acc: 0.7178
val Loss: 0.0231 Acc: 0.4930

Epoch 75/85
----------
train Loss: 0.0108 Acc: 0.6911
val Loss: 0.0223 Acc: 0.4977

Epoch 76/85
----------
train Loss: 0.0106 Acc: 0.6945
val Loss: 0.0221 Acc: 0.4977

Epoch 77/85
----------
train Loss: 0.0104 Acc: 0.7120
val Loss: 0.0236 Acc: 0.4930

Epoch 78/85
----------
train Loss: 0.0105 Acc: 0.6945
val Loss: 0.0227 Acc: 0.4884

Epoch 79/85
----------
train Loss: 0.0108 Acc: 0.6922
val Loss: 0.0247 Acc: 0.4884

Epoch 80/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0105 Acc: 0.7038
val Loss: 0.0228 Acc: 0.4977

Epoch 81/85
----------
train Loss: 0.0104 Acc: 0.7143
val Loss: 0.0214 Acc: 0.4977

Epoch 82/85
----------
train Loss: 0.0106 Acc: 0.7073
val Loss: 0.0217 Acc: 0.4977

Epoch 83/85
----------
train Loss: 0.0106 Acc: 0.7131
val Loss: 0.0232 Acc: 0.4977

Epoch 84/85
----------
train Loss: 0.0106 Acc: 0.7143
val Loss: 0.0242 Acc: 0.4884

Epoch 85/85
----------
train Loss: 0.0105 Acc: 0.7189
val Loss: 0.0221 Acc: 0.4977

Training complete in 7m 38s
Best val Acc: 0.511628

---Fine tuning.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0112 Acc: 0.6818
val Loss: 0.0263 Acc: 0.4698

Epoch 1/85
----------
train Loss: 0.0076 Acc: 0.7909
val Loss: 0.0252 Acc: 0.4512

Epoch 2/85
----------
train Loss: 0.0056 Acc: 0.8432
val Loss: 0.0208 Acc: 0.5163

Epoch 3/85
----------
train Loss: 0.0036 Acc: 0.9013
val Loss: 0.0203 Acc: 0.5116

Epoch 4/85
----------
train Loss: 0.0027 Acc: 0.9315
val Loss: 0.0202 Acc: 0.5674

Epoch 5/85
----------
train Loss: 0.0018 Acc: 0.9535
val Loss: 0.0233 Acc: 0.4977

Epoch 6/85
----------
train Loss: 0.0014 Acc: 0.9663
val Loss: 0.0211 Acc: 0.5395

Epoch 7/85
----------
train Loss: 0.0012 Acc: 0.9744
val Loss: 0.0234 Acc: 0.5535

Epoch 8/85
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0233 Acc: 0.5116

Epoch 9/85
----------
train Loss: 0.0011 Acc: 0.9768
val Loss: 0.0234 Acc: 0.5535

Epoch 10/85
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9849
val Loss: 0.0229 Acc: 0.5535

Epoch 11/85
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0210 Acc: 0.5488

Epoch 12/85
----------
train Loss: 0.0008 Acc: 0.9756
val Loss: 0.0298 Acc: 0.5535

Epoch 13/85
----------
train Loss: 0.0008 Acc: 0.9768
val Loss: 0.0262 Acc: 0.5581

Epoch 14/85
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0234 Acc: 0.5581

Epoch 15/85
----------
train Loss: 0.0008 Acc: 0.9779
val Loss: 0.0210 Acc: 0.5535

Epoch 16/85
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0234 Acc: 0.5581

Epoch 17/85
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0228 Acc: 0.5581

Epoch 18/85
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0250 Acc: 0.5767

Epoch 19/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0258 Acc: 0.5581

Epoch 20/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0262 Acc: 0.5628

Epoch 21/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0243 Acc: 0.5581

Epoch 22/85
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0254 Acc: 0.5581

Epoch 23/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0262 Acc: 0.5535

Epoch 24/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0217 Acc: 0.5628

Epoch 25/85
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0221 Acc: 0.5628

Epoch 26/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0249 Acc: 0.5628

Epoch 27/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0217 Acc: 0.5581

Epoch 28/85
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0225 Acc: 0.5581

Epoch 29/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0236 Acc: 0.5581

Epoch 30/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0006 Acc: 0.9768
val Loss: 0.0263 Acc: 0.5628

Epoch 31/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0261 Acc: 0.5674

Epoch 32/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0266 Acc: 0.5628

Epoch 33/85
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0274 Acc: 0.5628

Epoch 34/85
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0228 Acc: 0.5628

Epoch 35/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0247 Acc: 0.5581

Epoch 36/85
----------
train Loss: 0.0006 Acc: 0.9768
val Loss: 0.0257 Acc: 0.5581

Epoch 37/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0233 Acc: 0.5581

Epoch 38/85
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0234 Acc: 0.5628

Epoch 39/85
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0211 Acc: 0.5628

Epoch 40/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0207 Acc: 0.5628

Epoch 41/85
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0274 Acc: 0.5628

Epoch 42/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0267 Acc: 0.5628

Epoch 43/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0231 Acc: 0.5581

Epoch 44/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0228 Acc: 0.5581

Epoch 45/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0253 Acc: 0.5581

Epoch 46/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0225 Acc: 0.5628

Epoch 47/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0266 Acc: 0.5581

Epoch 48/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0279 Acc: 0.5581

Epoch 49/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0225 Acc: 0.5628

Epoch 50/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0264 Acc: 0.5581

Epoch 51/85
----------
train Loss: 0.0006 Acc: 0.9872
val Loss: 0.0235 Acc: 0.5628

Epoch 52/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0229 Acc: 0.5535

Epoch 53/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0223 Acc: 0.5628

Epoch 54/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0262 Acc: 0.5628

Epoch 55/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0213 Acc: 0.5674

Epoch 56/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0239 Acc: 0.5628

Epoch 57/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0274 Acc: 0.5628

Epoch 58/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0235 Acc: 0.5628

Epoch 59/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0267 Acc: 0.5628

Epoch 60/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0229 Acc: 0.5628

Epoch 61/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0221 Acc: 0.5535

Epoch 62/85
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0258 Acc: 0.5581

Epoch 63/85
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5628

Epoch 64/85
----------
train Loss: 0.0007 Acc: 0.9779
val Loss: 0.0266 Acc: 0.5581

Epoch 65/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0227 Acc: 0.5581

Epoch 66/85
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0227 Acc: 0.5628

Epoch 67/85
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0253 Acc: 0.5628

Epoch 68/85
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0281 Acc: 0.5628

Epoch 69/85
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0272 Acc: 0.5628

Epoch 70/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0304 Acc: 0.5628

Epoch 71/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0232 Acc: 0.5674

Epoch 72/85
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0244 Acc: 0.5628

Epoch 73/85
----------
train Loss: 0.0005 Acc: 0.9919
val Loss: 0.0263 Acc: 0.5581

Epoch 74/85
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0231 Acc: 0.5581

Epoch 75/85
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0246 Acc: 0.5628

Epoch 76/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0272 Acc: 0.5628

Epoch 77/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0255 Acc: 0.5674

Epoch 78/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0280 Acc: 0.5628

Epoch 79/85
----------
train Loss: 0.0006 Acc: 0.9768
val Loss: 0.0270 Acc: 0.5581

Epoch 80/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0296 Acc: 0.5535

Epoch 81/85
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0229 Acc: 0.5581

Epoch 82/85
----------
train Loss: 0.0006 Acc: 0.9768
val Loss: 0.0224 Acc: 0.5628

Epoch 83/85
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0241 Acc: 0.5581

Epoch 84/85
----------
train Loss: 0.0007 Acc: 0.9756
val Loss: 0.0226 Acc: 0.5581

Epoch 85/85
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5628

Training complete in 8m 7s
Best val Acc: 0.576744

---Testing---
Test accuracy: 0.902416
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 96 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 84 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 77 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.8703470419628597, std: 0.08815115368127432
--------------------

run info[val: 0.25, epoch: 64, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0294 Acc: 0.1487
val Loss: 0.0274 Acc: 0.2156

Epoch 1/63
----------
train Loss: 0.0258 Acc: 0.2800
val Loss: 0.0240 Acc: 0.3346

Epoch 2/63
----------
train Loss: 0.0223 Acc: 0.3618
val Loss: 0.0246 Acc: 0.3123

Epoch 3/63
----------
LR is set to 0.001
train Loss: 0.0208 Acc: 0.4213
val Loss: 0.0213 Acc: 0.4015

Epoch 4/63
----------
train Loss: 0.0186 Acc: 0.4548
val Loss: 0.0207 Acc: 0.4015

Epoch 5/63
----------
train Loss: 0.0175 Acc: 0.4944
val Loss: 0.0207 Acc: 0.4052

Epoch 6/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0173 Acc: 0.5143
val Loss: 0.0206 Acc: 0.4126

Epoch 7/63
----------
train Loss: 0.0177 Acc: 0.5167
val Loss: 0.0206 Acc: 0.4089

Epoch 8/63
----------
train Loss: 0.0178 Acc: 0.5304
val Loss: 0.0202 Acc: 0.4015

Epoch 9/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0178 Acc: 0.5192
val Loss: 0.0203 Acc: 0.4052

Epoch 10/63
----------
train Loss: 0.0175 Acc: 0.5167
val Loss: 0.0204 Acc: 0.4015

Epoch 11/63
----------
train Loss: 0.0173 Acc: 0.5304
val Loss: 0.0202 Acc: 0.4015

Epoch 12/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0174 Acc: 0.5143
val Loss: 0.0205 Acc: 0.4015

Epoch 13/63
----------
train Loss: 0.0174 Acc: 0.5204
val Loss: 0.0202 Acc: 0.4015

Epoch 14/63
----------
train Loss: 0.0184 Acc: 0.5167
val Loss: 0.0205 Acc: 0.4015

Epoch 15/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0172 Acc: 0.5093
val Loss: 0.0202 Acc: 0.3978

Epoch 16/63
----------
train Loss: 0.0178 Acc: 0.5304
val Loss: 0.0206 Acc: 0.4015

Epoch 17/63
----------
train Loss: 0.0184 Acc: 0.5081
val Loss: 0.0203 Acc: 0.3978

Epoch 18/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0176 Acc: 0.5118
val Loss: 0.0207 Acc: 0.4015

Epoch 19/63
----------
train Loss: 0.0184 Acc: 0.5266
val Loss: 0.0203 Acc: 0.4052

Epoch 20/63
----------
train Loss: 0.0175 Acc: 0.5192
val Loss: 0.0205 Acc: 0.4089

Epoch 21/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0174 Acc: 0.5242
val Loss: 0.0205 Acc: 0.3941

Epoch 22/63
----------
train Loss: 0.0180 Acc: 0.5192
val Loss: 0.0205 Acc: 0.3941

Epoch 23/63
----------
train Loss: 0.0171 Acc: 0.5192
val Loss: 0.0205 Acc: 0.4015

Epoch 24/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0177 Acc: 0.5130
val Loss: 0.0203 Acc: 0.4089

Epoch 25/63
----------
train Loss: 0.0178 Acc: 0.5279
val Loss: 0.0207 Acc: 0.4089

Epoch 26/63
----------
train Loss: 0.0183 Acc: 0.5204
val Loss: 0.0205 Acc: 0.4052

Epoch 27/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0175 Acc: 0.5242
val Loss: 0.0202 Acc: 0.4052

Epoch 28/63
----------
train Loss: 0.0183 Acc: 0.5180
val Loss: 0.0204 Acc: 0.3941

Epoch 29/63
----------
train Loss: 0.0173 Acc: 0.5279
val Loss: 0.0203 Acc: 0.3978

Epoch 30/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0176 Acc: 0.5341
val Loss: 0.0203 Acc: 0.4052

Epoch 31/63
----------
train Loss: 0.0183 Acc: 0.5229
val Loss: 0.0204 Acc: 0.3978

Epoch 32/63
----------
train Loss: 0.0181 Acc: 0.5217
val Loss: 0.0204 Acc: 0.4015

Epoch 33/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0185 Acc: 0.5143
val Loss: 0.0204 Acc: 0.3978

Epoch 34/63
----------
train Loss: 0.0175 Acc: 0.5093
val Loss: 0.0203 Acc: 0.3978

Epoch 35/63
----------
train Loss: 0.0180 Acc: 0.5081
val Loss: 0.0207 Acc: 0.3978

Epoch 36/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0172 Acc: 0.5204
val Loss: 0.0205 Acc: 0.3941

Epoch 37/63
----------
train Loss: 0.0175 Acc: 0.5217
val Loss: 0.0201 Acc: 0.3978

Epoch 38/63
----------
train Loss: 0.0175 Acc: 0.5130
val Loss: 0.0204 Acc: 0.4052

Epoch 39/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0182 Acc: 0.5143
val Loss: 0.0206 Acc: 0.3978

Epoch 40/63
----------
train Loss: 0.0189 Acc: 0.5118
val Loss: 0.0206 Acc: 0.4015

Epoch 41/63
----------
train Loss: 0.0187 Acc: 0.5242
val Loss: 0.0205 Acc: 0.3941

Epoch 42/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0181 Acc: 0.5366
val Loss: 0.0204 Acc: 0.4052

Epoch 43/63
----------
train Loss: 0.0179 Acc: 0.5192
val Loss: 0.0204 Acc: 0.4015

Epoch 44/63
----------
train Loss: 0.0174 Acc: 0.5192
val Loss: 0.0200 Acc: 0.3978

Epoch 45/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0173 Acc: 0.5192
val Loss: 0.0204 Acc: 0.3941

Epoch 46/63
----------
train Loss: 0.0188 Acc: 0.5217
val Loss: 0.0203 Acc: 0.3941

Epoch 47/63
----------
train Loss: 0.0172 Acc: 0.5266
val Loss: 0.0203 Acc: 0.3978

Epoch 48/63
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0183 Acc: 0.5266
val Loss: 0.0204 Acc: 0.3941

Epoch 49/63
----------
train Loss: 0.0183 Acc: 0.5180
val Loss: 0.0203 Acc: 0.3941

Epoch 50/63
----------
train Loss: 0.0183 Acc: 0.5254
val Loss: 0.0205 Acc: 0.4015

Epoch 51/63
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0185 Acc: 0.5180
val Loss: 0.0205 Acc: 0.4015

Epoch 52/63
----------
train Loss: 0.0177 Acc: 0.5180
val Loss: 0.0205 Acc: 0.4052

Epoch 53/63
----------
train Loss: 0.0183 Acc: 0.5192
val Loss: 0.0204 Acc: 0.4015

Epoch 54/63
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0180 Acc: 0.5204
val Loss: 0.0205 Acc: 0.4052

Epoch 55/63
----------
train Loss: 0.0182 Acc: 0.5266
val Loss: 0.0205 Acc: 0.4015

Epoch 56/63
----------
train Loss: 0.0180 Acc: 0.5130
val Loss: 0.0204 Acc: 0.3941

Epoch 57/63
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0177 Acc: 0.5130
val Loss: 0.0203 Acc: 0.3941

Epoch 58/63
----------
train Loss: 0.0173 Acc: 0.5043
val Loss: 0.0203 Acc: 0.4015

Epoch 59/63
----------
train Loss: 0.0172 Acc: 0.5043
val Loss: 0.0204 Acc: 0.4089

Epoch 60/63
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0175 Acc: 0.5167
val Loss: 0.0203 Acc: 0.3978

Epoch 61/63
----------
train Loss: 0.0183 Acc: 0.5291
val Loss: 0.0202 Acc: 0.4015

Epoch 62/63
----------
train Loss: 0.0172 Acc: 0.5118
val Loss: 0.0205 Acc: 0.4052

Epoch 63/63
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0180 Acc: 0.5118
val Loss: 0.0205 Acc: 0.3941

Training complete in 5m 50s
Best val Acc: 0.412639

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0184 Acc: 0.5031
val Loss: 0.0233 Acc: 0.3197

Epoch 1/63
----------
train Loss: 0.0176 Acc: 0.4907
val Loss: 0.0247 Acc: 0.3903

Epoch 2/63
----------
train Loss: 0.0128 Acc: 0.6258
val Loss: 0.0248 Acc: 0.3903

Epoch 3/63
----------
LR is set to 0.001
train Loss: 0.0104 Acc: 0.7076
val Loss: 0.0185 Acc: 0.4870

Epoch 4/63
----------
train Loss: 0.0083 Acc: 0.7893
val Loss: 0.0173 Acc: 0.5167

Epoch 5/63
----------
train Loss: 0.0074 Acc: 0.8129
val Loss: 0.0169 Acc: 0.5465

Epoch 6/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0080 Acc: 0.8253
val Loss: 0.0167 Acc: 0.5390

Epoch 7/63
----------
train Loss: 0.0065 Acc: 0.8389
val Loss: 0.0168 Acc: 0.5502

Epoch 8/63
----------
train Loss: 0.0074 Acc: 0.8278
val Loss: 0.0164 Acc: 0.5502

Epoch 9/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0063 Acc: 0.8414
val Loss: 0.0163 Acc: 0.5390

Epoch 10/63
----------
train Loss: 0.0065 Acc: 0.8364
val Loss: 0.0163 Acc: 0.5316

Epoch 11/63
----------
train Loss: 0.0071 Acc: 0.8426
val Loss: 0.0162 Acc: 0.5390

Epoch 12/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0067 Acc: 0.8414
val Loss: 0.0163 Acc: 0.5465

Epoch 13/63
----------
train Loss: 0.0062 Acc: 0.8501
val Loss: 0.0165 Acc: 0.5502

Epoch 14/63
----------
train Loss: 0.0062 Acc: 0.8426
val Loss: 0.0165 Acc: 0.5353

Epoch 15/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0064 Acc: 0.8290
val Loss: 0.0168 Acc: 0.5428

Epoch 16/63
----------
train Loss: 0.0072 Acc: 0.8253
val Loss: 0.0167 Acc: 0.5428

Epoch 17/63
----------
train Loss: 0.0071 Acc: 0.8315
val Loss: 0.0165 Acc: 0.5465

Epoch 18/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0066 Acc: 0.8426
val Loss: 0.0167 Acc: 0.5279

Epoch 19/63
----------
train Loss: 0.0072 Acc: 0.8327
val Loss: 0.0164 Acc: 0.5167

Epoch 20/63
----------
train Loss: 0.0072 Acc: 0.8377
val Loss: 0.0166 Acc: 0.5390

Epoch 21/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0082 Acc: 0.8401
val Loss: 0.0164 Acc: 0.5390

Epoch 22/63
----------
train Loss: 0.0071 Acc: 0.8439
val Loss: 0.0165 Acc: 0.5465

Epoch 23/63
----------
train Loss: 0.0067 Acc: 0.8340
val Loss: 0.0167 Acc: 0.5502

Epoch 24/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0068 Acc: 0.8501
val Loss: 0.0162 Acc: 0.5428

Epoch 25/63
----------
train Loss: 0.0066 Acc: 0.8439
val Loss: 0.0163 Acc: 0.5539

Epoch 26/63
----------
train Loss: 0.0071 Acc: 0.8377
val Loss: 0.0165 Acc: 0.5502

Epoch 27/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0066 Acc: 0.8315
val Loss: 0.0162 Acc: 0.5502

Epoch 28/63
----------
train Loss: 0.0067 Acc: 0.8414
val Loss: 0.0164 Acc: 0.5539

Epoch 29/63
----------
train Loss: 0.0069 Acc: 0.8327
val Loss: 0.0165 Acc: 0.5353

Epoch 30/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0071 Acc: 0.8401
val Loss: 0.0168 Acc: 0.5502

Epoch 31/63
----------
train Loss: 0.0073 Acc: 0.8352
val Loss: 0.0165 Acc: 0.5539

Epoch 32/63
----------
train Loss: 0.0067 Acc: 0.8451
val Loss: 0.0163 Acc: 0.5390

Epoch 33/63
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0068 Acc: 0.8451
val Loss: 0.0167 Acc: 0.5465

Epoch 34/63
----------
train Loss: 0.0061 Acc: 0.8488
val Loss: 0.0166 Acc: 0.5539

Epoch 35/63
----------
train Loss: 0.0079 Acc: 0.8203
val Loss: 0.0163 Acc: 0.5539

Epoch 36/63
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0069 Acc: 0.8364
val Loss: 0.0166 Acc: 0.5316

Epoch 37/63
----------
train Loss: 0.0063 Acc: 0.8364
val Loss: 0.0164 Acc: 0.5428

Epoch 38/63
----------
train Loss: 0.0068 Acc: 0.8340
val Loss: 0.0164 Acc: 0.5390

Epoch 39/63
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0070 Acc: 0.8414
val Loss: 0.0163 Acc: 0.5428

Epoch 40/63
----------
train Loss: 0.0072 Acc: 0.8414
val Loss: 0.0166 Acc: 0.5390

Epoch 41/63
----------
train Loss: 0.0067 Acc: 0.8401
val Loss: 0.0167 Acc: 0.5353

Epoch 42/63
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0068 Acc: 0.8401
val Loss: 0.0167 Acc: 0.5390

Epoch 43/63
----------
train Loss: 0.0070 Acc: 0.8414
val Loss: 0.0165 Acc: 0.5465

Epoch 44/63
----------
train Loss: 0.0067 Acc: 0.8290
val Loss: 0.0163 Acc: 0.5539

Epoch 45/63
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0073 Acc: 0.8315
val Loss: 0.0165 Acc: 0.5465

Epoch 46/63
----------
train Loss: 0.0064 Acc: 0.8463
val Loss: 0.0162 Acc: 0.5390

Epoch 47/63
----------
train Loss: 0.0062 Acc: 0.8488
val Loss: 0.0164 Acc: 0.5465

Epoch 48/63
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0066 Acc: 0.8426
val Loss: 0.0164 Acc: 0.5353

Epoch 49/63
----------
train Loss: 0.0064 Acc: 0.8327
val Loss: 0.0164 Acc: 0.5539

Epoch 50/63
----------
train Loss: 0.0074 Acc: 0.8389
val Loss: 0.0166 Acc: 0.5502

Epoch 51/63
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0067 Acc: 0.8340
val Loss: 0.0161 Acc: 0.5502

Epoch 52/63
----------
train Loss: 0.0063 Acc: 0.8389
val Loss: 0.0162 Acc: 0.5390

Epoch 53/63
----------
train Loss: 0.0086 Acc: 0.8290
val Loss: 0.0166 Acc: 0.5613

Epoch 54/63
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0079 Acc: 0.8278
val Loss: 0.0166 Acc: 0.5279

Epoch 55/63
----------
train Loss: 0.0066 Acc: 0.8327
val Loss: 0.0168 Acc: 0.5316

Epoch 56/63
----------
train Loss: 0.0069 Acc: 0.8315
val Loss: 0.0163 Acc: 0.5390

Epoch 57/63
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0073 Acc: 0.8525
val Loss: 0.0168 Acc: 0.5465

Epoch 58/63
----------
train Loss: 0.0067 Acc: 0.8327
val Loss: 0.0164 Acc: 0.5390

Epoch 59/63
----------
train Loss: 0.0071 Acc: 0.8302
val Loss: 0.0166 Acc: 0.5576

Epoch 60/63
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0066 Acc: 0.8451
val Loss: 0.0165 Acc: 0.5465

Epoch 61/63
----------
train Loss: 0.0072 Acc: 0.8265
val Loss: 0.0162 Acc: 0.5390

Epoch 62/63
----------
train Loss: 0.0075 Acc: 0.8290
val Loss: 0.0165 Acc: 0.5316

Epoch 63/63
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0075 Acc: 0.8178
val Loss: 0.0162 Acc: 0.5279

Training complete in 6m 15s
Best val Acc: 0.561338

---Testing---
Test accuracy: 0.791822
--------------------
Accuracy of Albacore tuna : 79 %
Accuracy of Atlantic bluefin tuna : 70 %
Accuracy of Bigeye tuna : 70 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 37 %
Accuracy of Little tunny : 77 %
Accuracy of Longtail tuna : 84 %
Accuracy of Mackerel tuna : 72 %
Accuracy of Pacific bluefin tuna : 53 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna :  0 %
Accuracy of Southern bluefin tuna : 68 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.6958395834155287, std: 0.24233888459716602
--------------------

run info[val: 0.3, epoch: 96, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0280 Acc: 0.1326
val Loss: 0.0332 Acc: 0.1894

Epoch 1/95
----------
train Loss: 0.0231 Acc: 0.2891
val Loss: 0.0287 Acc: 0.3012

Epoch 2/95
----------
train Loss: 0.0202 Acc: 0.3767
val Loss: 0.0260 Acc: 0.3230

Epoch 3/95
----------
LR is set to 0.001
train Loss: 0.0174 Acc: 0.4655
val Loss: 0.0239 Acc: 0.3789

Epoch 4/95
----------
train Loss: 0.0163 Acc: 0.5517
val Loss: 0.0249 Acc: 0.3975

Epoch 5/95
----------
train Loss: 0.0164 Acc: 0.5265
val Loss: 0.0233 Acc: 0.3913

Epoch 6/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0235 Acc: 0.3913

Epoch 7/95
----------
train Loss: 0.0159 Acc: 0.5424
val Loss: 0.0236 Acc: 0.3913

Epoch 8/95
----------
train Loss: 0.0160 Acc: 0.5544
val Loss: 0.0235 Acc: 0.3851

Epoch 9/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0158 Acc: 0.5345
val Loss: 0.0232 Acc: 0.3820

Epoch 10/95
----------
train Loss: 0.0158 Acc: 0.5411
val Loss: 0.0237 Acc: 0.3820

Epoch 11/95
----------
train Loss: 0.0158 Acc: 0.5504
val Loss: 0.0227 Acc: 0.3851

Epoch 12/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0159 Acc: 0.5504
val Loss: 0.0235 Acc: 0.3851

Epoch 13/95
----------
train Loss: 0.0157 Acc: 0.5371
val Loss: 0.0232 Acc: 0.3820

Epoch 14/95
----------
train Loss: 0.0157 Acc: 0.5504
val Loss: 0.0236 Acc: 0.3820

Epoch 15/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0155 Acc: 0.5703
val Loss: 0.0230 Acc: 0.3851

Epoch 16/95
----------
train Loss: 0.0156 Acc: 0.5477
val Loss: 0.0226 Acc: 0.3851

Epoch 17/95
----------
train Loss: 0.0158 Acc: 0.5597
val Loss: 0.0232 Acc: 0.3851

Epoch 18/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0157 Acc: 0.5491
val Loss: 0.0248 Acc: 0.3820

Epoch 19/95
----------
train Loss: 0.0158 Acc: 0.5411
val Loss: 0.0225 Acc: 0.3820

Epoch 20/95
----------
train Loss: 0.0156 Acc: 0.5597
val Loss: 0.0240 Acc: 0.3851

Epoch 21/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0158 Acc: 0.5477
val Loss: 0.0241 Acc: 0.3851

Epoch 22/95
----------
train Loss: 0.0160 Acc: 0.5411
val Loss: 0.0235 Acc: 0.3851

Epoch 23/95
----------
train Loss: 0.0157 Acc: 0.5424
val Loss: 0.0236 Acc: 0.3820

Epoch 24/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0157 Acc: 0.5424
val Loss: 0.0233 Acc: 0.3820

Epoch 25/95
----------
train Loss: 0.0156 Acc: 0.5491
val Loss: 0.0239 Acc: 0.3820

Epoch 26/95
----------
train Loss: 0.0158 Acc: 0.5557
val Loss: 0.0250 Acc: 0.3820

Epoch 27/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0158 Acc: 0.5531
val Loss: 0.0238 Acc: 0.3820

Epoch 28/95
----------
train Loss: 0.0160 Acc: 0.5464
val Loss: 0.0233 Acc: 0.3851

Epoch 29/95
----------
train Loss: 0.0158 Acc: 0.5584
val Loss: 0.0230 Acc: 0.3851

Epoch 30/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0157 Acc: 0.5663
val Loss: 0.0225 Acc: 0.3820

Epoch 31/95
----------
train Loss: 0.0157 Acc: 0.5477
val Loss: 0.0229 Acc: 0.3851

Epoch 32/95
----------
train Loss: 0.0157 Acc: 0.5584
val Loss: 0.0229 Acc: 0.3851

Epoch 33/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0159 Acc: 0.5491
val Loss: 0.0230 Acc: 0.3851

Epoch 34/95
----------
train Loss: 0.0158 Acc: 0.5504
val Loss: 0.0232 Acc: 0.3820

Epoch 35/95
----------
train Loss: 0.0158 Acc: 0.5544
val Loss: 0.0238 Acc: 0.3820

Epoch 36/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0158 Acc: 0.5570
val Loss: 0.0236 Acc: 0.3820

Epoch 37/95
----------
train Loss: 0.0157 Acc: 0.5531
val Loss: 0.0237 Acc: 0.3851

Epoch 38/95
----------
train Loss: 0.0160 Acc: 0.5424
val Loss: 0.0233 Acc: 0.3851

Epoch 39/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0157 Acc: 0.5557
val Loss: 0.0232 Acc: 0.3851

Epoch 40/95
----------
train Loss: 0.0158 Acc: 0.5438
val Loss: 0.0230 Acc: 0.3820

Epoch 41/95
----------
train Loss: 0.0158 Acc: 0.5464
val Loss: 0.0244 Acc: 0.3820

Epoch 42/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0157 Acc: 0.5504
val Loss: 0.0246 Acc: 0.3820

Epoch 43/95
----------
train Loss: 0.0158 Acc: 0.5411
val Loss: 0.0235 Acc: 0.3820

Epoch 44/95
----------
train Loss: 0.0159 Acc: 0.5477
val Loss: 0.0227 Acc: 0.3820

Epoch 45/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0159 Acc: 0.5597
val Loss: 0.0231 Acc: 0.3820

Epoch 46/95
----------
train Loss: 0.0158 Acc: 0.5438
val Loss: 0.0243 Acc: 0.3851

Epoch 47/95
----------
train Loss: 0.0159 Acc: 0.5491
val Loss: 0.0236 Acc: 0.3851

Epoch 48/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0159 Acc: 0.5438
val Loss: 0.0223 Acc: 0.3851

Epoch 49/95
----------
train Loss: 0.0158 Acc: 0.5610
val Loss: 0.0228 Acc: 0.3851

Epoch 50/95
----------
train Loss: 0.0159 Acc: 0.5584
val Loss: 0.0225 Acc: 0.3851

Epoch 51/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0158 Acc: 0.5424
val Loss: 0.0233 Acc: 0.3851

Epoch 52/95
----------
train Loss: 0.0159 Acc: 0.5610
val Loss: 0.0243 Acc: 0.3820

Epoch 53/95
----------
train Loss: 0.0157 Acc: 0.5531
val Loss: 0.0229 Acc: 0.3820

Epoch 54/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0156 Acc: 0.5570
val Loss: 0.0232 Acc: 0.3851

Epoch 55/95
----------
train Loss: 0.0158 Acc: 0.5411
val Loss: 0.0240 Acc: 0.3820

Epoch 56/95
----------
train Loss: 0.0158 Acc: 0.5504
val Loss: 0.0241 Acc: 0.3820

Epoch 57/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0159 Acc: 0.5477
val Loss: 0.0228 Acc: 0.3820

Epoch 58/95
----------
train Loss: 0.0158 Acc: 0.5531
val Loss: 0.0231 Acc: 0.3820

Epoch 59/95
----------
train Loss: 0.0157 Acc: 0.5517
val Loss: 0.0240 Acc: 0.3820

Epoch 60/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0158 Acc: 0.5570
val Loss: 0.0229 Acc: 0.3851

Epoch 61/95
----------
train Loss: 0.0159 Acc: 0.5531
val Loss: 0.0258 Acc: 0.3820

Epoch 62/95
----------
train Loss: 0.0158 Acc: 0.5477
val Loss: 0.0224 Acc: 0.3820

Epoch 63/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0159 Acc: 0.5464
val Loss: 0.0226 Acc: 0.3820

Epoch 64/95
----------
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0243 Acc: 0.3820

Epoch 65/95
----------
train Loss: 0.0157 Acc: 0.5424
val Loss: 0.0238 Acc: 0.3820

Epoch 66/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0158 Acc: 0.5438
val Loss: 0.0233 Acc: 0.3820

Epoch 67/95
----------
train Loss: 0.0158 Acc: 0.5544
val Loss: 0.0229 Acc: 0.3820

Epoch 68/95
----------
train Loss: 0.0157 Acc: 0.5531
val Loss: 0.0233 Acc: 0.3820

Epoch 69/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0158 Acc: 0.5345
val Loss: 0.0230 Acc: 0.3820

Epoch 70/95
----------
train Loss: 0.0157 Acc: 0.5570
val Loss: 0.0228 Acc: 0.3820

Epoch 71/95
----------
train Loss: 0.0159 Acc: 0.5411
val Loss: 0.0232 Acc: 0.3820

Epoch 72/95
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0158 Acc: 0.5544
val Loss: 0.0237 Acc: 0.3820

Epoch 73/95
----------
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0234 Acc: 0.3820

Epoch 74/95
----------
train Loss: 0.0159 Acc: 0.5531
val Loss: 0.0233 Acc: 0.3820

Epoch 75/95
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0159 Acc: 0.5385
val Loss: 0.0230 Acc: 0.3820

Epoch 76/95
----------
train Loss: 0.0158 Acc: 0.5464
val Loss: 0.0240 Acc: 0.3820

Epoch 77/95
----------
train Loss: 0.0157 Acc: 0.5557
val Loss: 0.0233 Acc: 0.3820

Epoch 78/95
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0158 Acc: 0.5544
val Loss: 0.0240 Acc: 0.3851

Epoch 79/95
----------
train Loss: 0.0158 Acc: 0.5570
val Loss: 0.0223 Acc: 0.3851

Epoch 80/95
----------
train Loss: 0.0158 Acc: 0.5570
val Loss: 0.0230 Acc: 0.3820

Epoch 81/95
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0158 Acc: 0.5464
val Loss: 0.0232 Acc: 0.3820

Epoch 82/95
----------
train Loss: 0.0159 Acc: 0.5438
val Loss: 0.0238 Acc: 0.3820

Epoch 83/95
----------
train Loss: 0.0156 Acc: 0.5544
val Loss: 0.0230 Acc: 0.3851

Epoch 84/95
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0226 Acc: 0.3820

Epoch 85/95
----------
train Loss: 0.0157 Acc: 0.5531
val Loss: 0.0235 Acc: 0.3851

Epoch 86/95
----------
train Loss: 0.0158 Acc: 0.5637
val Loss: 0.0245 Acc: 0.3851

Epoch 87/95
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0236 Acc: 0.3820

Epoch 88/95
----------
train Loss: 0.0159 Acc: 0.5544
val Loss: 0.0230 Acc: 0.3820

Epoch 89/95
----------
train Loss: 0.0157 Acc: 0.5663
val Loss: 0.0241 Acc: 0.3851

Epoch 90/95
----------
LR is set to 1.0000000000000017e-32
train Loss: 0.0157 Acc: 0.5398
val Loss: 0.0234 Acc: 0.3820

Epoch 91/95
----------
train Loss: 0.0158 Acc: 0.5623
val Loss: 0.0230 Acc: 0.3820

Epoch 92/95
----------
train Loss: 0.0158 Acc: 0.5623
val Loss: 0.0238 Acc: 0.3820

Epoch 93/95
----------
LR is set to 1.0000000000000016e-33
train Loss: 0.0159 Acc: 0.5597
val Loss: 0.0239 Acc: 0.3820

Epoch 94/95
----------
train Loss: 0.0158 Acc: 0.5464
val Loss: 0.0233 Acc: 0.3820

Epoch 95/95
----------
train Loss: 0.0159 Acc: 0.5477
val Loss: 0.0248 Acc: 0.3820

Training complete in 8m 52s
Best val Acc: 0.397516

---Fine tuning.---
Epoch 0/95
----------
LR is set to 0.01
train Loss: 0.0159 Acc: 0.5451
val Loss: 0.0226 Acc: 0.4224

Epoch 1/95
----------
train Loss: 0.0112 Acc: 0.6883
val Loss: 0.0208 Acc: 0.4565

Epoch 2/95
----------
train Loss: 0.0070 Acc: 0.8263
val Loss: 0.0204 Acc: 0.4969

Epoch 3/95
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.8979
val Loss: 0.0200 Acc: 0.5000

Epoch 4/95
----------
train Loss: 0.0038 Acc: 0.9271
val Loss: 0.0172 Acc: 0.5155

Epoch 5/95
----------
train Loss: 0.0033 Acc: 0.9456
val Loss: 0.0175 Acc: 0.5155

Epoch 6/95
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9403
val Loss: 0.0188 Acc: 0.5217

Epoch 7/95
----------
train Loss: 0.0033 Acc: 0.9536
val Loss: 0.0182 Acc: 0.5280

Epoch 8/95
----------
train Loss: 0.0033 Acc: 0.9536
val Loss: 0.0182 Acc: 0.5248

Epoch 9/95
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0031 Acc: 0.9509
val Loss: 0.0186 Acc: 0.5280

Epoch 10/95
----------
train Loss: 0.0032 Acc: 0.9602
val Loss: 0.0170 Acc: 0.5280

Epoch 11/95
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0191 Acc: 0.5280

Epoch 12/95
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0033 Acc: 0.9509
val Loss: 0.0179 Acc: 0.5248

Epoch 13/95
----------
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0179 Acc: 0.5217

Epoch 14/95
----------
train Loss: 0.0032 Acc: 0.9430
val Loss: 0.0191 Acc: 0.5217

Epoch 15/95
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0192 Acc: 0.5217

Epoch 16/95
----------
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0168 Acc: 0.5280

Epoch 17/95
----------
train Loss: 0.0033 Acc: 0.9523
val Loss: 0.0174 Acc: 0.5248

Epoch 18/95
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0174 Acc: 0.5248

Epoch 19/95
----------
train Loss: 0.0032 Acc: 0.9589
val Loss: 0.0184 Acc: 0.5248

Epoch 20/95
----------
train Loss: 0.0031 Acc: 0.9602
val Loss: 0.0176 Acc: 0.5280

Epoch 21/95
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0031 Acc: 0.9509
val Loss: 0.0190 Acc: 0.5280

Epoch 22/95
----------
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0177 Acc: 0.5248

Epoch 23/95
----------
train Loss: 0.0032 Acc: 0.9629
val Loss: 0.0182 Acc: 0.5280

Epoch 24/95
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0182 Acc: 0.5280

Epoch 25/95
----------
train Loss: 0.0031 Acc: 0.9549
val Loss: 0.0188 Acc: 0.5280

Epoch 26/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0177 Acc: 0.5248

Epoch 27/95
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0033 Acc: 0.9443
val Loss: 0.0178 Acc: 0.5311

Epoch 28/95
----------
train Loss: 0.0033 Acc: 0.9562
val Loss: 0.0184 Acc: 0.5280

Epoch 29/95
----------
train Loss: 0.0033 Acc: 0.9562
val Loss: 0.0191 Acc: 0.5248

Epoch 30/95
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0032 Acc: 0.9443
val Loss: 0.0175 Acc: 0.5217

Epoch 31/95
----------
train Loss: 0.0031 Acc: 0.9589
val Loss: 0.0173 Acc: 0.5280

Epoch 32/95
----------
train Loss: 0.0031 Acc: 0.9589
val Loss: 0.0180 Acc: 0.5217

Epoch 33/95
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0190 Acc: 0.5248

Epoch 34/95
----------
train Loss: 0.0033 Acc: 0.9456
val Loss: 0.0182 Acc: 0.5248

Epoch 35/95
----------
train Loss: 0.0031 Acc: 0.9549
val Loss: 0.0182 Acc: 0.5248

Epoch 36/95
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0172 Acc: 0.5280

Epoch 37/95
----------
train Loss: 0.0030 Acc: 0.9602
val Loss: 0.0185 Acc: 0.5248

Epoch 38/95
----------
train Loss: 0.0032 Acc: 0.9576
val Loss: 0.0187 Acc: 0.5248

Epoch 39/95
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0030 Acc: 0.9523
val Loss: 0.0178 Acc: 0.5248

Epoch 40/95
----------
train Loss: 0.0033 Acc: 0.9483
val Loss: 0.0176 Acc: 0.5280

Epoch 41/95
----------
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0186 Acc: 0.5248

Epoch 42/95
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0032 Acc: 0.9589
val Loss: 0.0184 Acc: 0.5248

Epoch 43/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0194 Acc: 0.5280

Epoch 44/95
----------
train Loss: 0.0031 Acc: 0.9496
val Loss: 0.0187 Acc: 0.5248

Epoch 45/95
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0187 Acc: 0.5217

Epoch 46/95
----------
train Loss: 0.0032 Acc: 0.9430
val Loss: 0.0186 Acc: 0.5248

Epoch 47/95
----------
train Loss: 0.0033 Acc: 0.9536
val Loss: 0.0180 Acc: 0.5280

Epoch 48/95
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0033 Acc: 0.9509
val Loss: 0.0173 Acc: 0.5280

Epoch 49/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0179 Acc: 0.5280

Epoch 50/95
----------
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0186 Acc: 0.5217

Epoch 51/95
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0031 Acc: 0.9549
val Loss: 0.0184 Acc: 0.5248

Epoch 52/95
----------
train Loss: 0.0031 Acc: 0.9576
val Loss: 0.0187 Acc: 0.5217

Epoch 53/95
----------
train Loss: 0.0032 Acc: 0.9589
val Loss: 0.0172 Acc: 0.5186

Epoch 54/95
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0031 Acc: 0.9523
val Loss: 0.0183 Acc: 0.5217

Epoch 55/95
----------
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0180 Acc: 0.5217

Epoch 56/95
----------
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0179 Acc: 0.5248

Epoch 57/95
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0032 Acc: 0.9576
val Loss: 0.0177 Acc: 0.5217

Epoch 58/95
----------
train Loss: 0.0033 Acc: 0.9456
val Loss: 0.0183 Acc: 0.5280

Epoch 59/95
----------
train Loss: 0.0033 Acc: 0.9496
val Loss: 0.0190 Acc: 0.5248

Epoch 60/95
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0177 Acc: 0.5217

Epoch 61/95
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0186 Acc: 0.5311

Epoch 62/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0185 Acc: 0.5311

Epoch 63/95
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0190 Acc: 0.5342

Epoch 64/95
----------
train Loss: 0.0032 Acc: 0.9443
val Loss: 0.0175 Acc: 0.5217

Epoch 65/95
----------
train Loss: 0.0032 Acc: 0.9562
val Loss: 0.0183 Acc: 0.5248

Epoch 66/95
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0032 Acc: 0.9589
val Loss: 0.0175 Acc: 0.5280

Epoch 67/95
----------
train Loss: 0.0032 Acc: 0.9562
val Loss: 0.0184 Acc: 0.5217

Epoch 68/95
----------
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0182 Acc: 0.5217

Epoch 69/95
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0032 Acc: 0.9589
val Loss: 0.0182 Acc: 0.5248

Epoch 70/95
----------
train Loss: 0.0031 Acc: 0.9589
val Loss: 0.0172 Acc: 0.5248

Epoch 71/95
----------
train Loss: 0.0032 Acc: 0.9576
val Loss: 0.0183 Acc: 0.5186

Epoch 72/95
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0191 Acc: 0.5217

Epoch 73/95
----------
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0174 Acc: 0.5186

Epoch 74/95
----------
train Loss: 0.0032 Acc: 0.9562
val Loss: 0.0180 Acc: 0.5217

Epoch 75/95
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0169 Acc: 0.5248

Epoch 76/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0178 Acc: 0.5280

Epoch 77/95
----------
train Loss: 0.0034 Acc: 0.9483
val Loss: 0.0179 Acc: 0.5280

Epoch 78/95
----------
LR is set to 1.0000000000000015e-28
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0176 Acc: 0.5280

Epoch 79/95
----------
train Loss: 0.0031 Acc: 0.9589
val Loss: 0.0169 Acc: 0.5248

Epoch 80/95
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0175 Acc: 0.5248

Epoch 81/95
----------
LR is set to 1.0000000000000015e-29
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0182 Acc: 0.5186

Epoch 82/95
----------
train Loss: 0.0031 Acc: 0.9562
val Loss: 0.0172 Acc: 0.5217

Epoch 83/95
----------
train Loss: 0.0031 Acc: 0.9536
val Loss: 0.0177 Acc: 0.5217

Epoch 84/95
----------
LR is set to 1.0000000000000015e-30
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0181 Acc: 0.5248

Epoch 85/95
----------
train Loss: 0.0032 Acc: 0.9456
val Loss: 0.0188 Acc: 0.5217

Epoch 86/95
----------
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0184 Acc: 0.5248

Epoch 87/95
----------
LR is set to 1.0000000000000016e-31
train Loss: 0.0032 Acc: 0.9483
val Loss: 0.0194 Acc: 0.5186

Epoch 88/95
----------
train Loss: 0.0031 Acc: 0.9576
val Loss: 0.0169 Acc: 0.5248

Epoch 89/95
----------
train Loss: 0.0032 Acc: 0.9549
val Loss: 0.0186 Acc: 0.5248

Epoch 90/95
----------
LR is set to 1.0000000000000017e-32
train Loss: 0.0032 Acc: 0.9509
val Loss: 0.0181 Acc: 0.5248

Epoch 91/95
----------
train Loss: 0.0032 Acc: 0.9523
val Loss: 0.0174 Acc: 0.5217

Epoch 92/95
----------
train Loss: 0.0031 Acc: 0.9523
val Loss: 0.0177 Acc: 0.5217

Epoch 93/95
----------
LR is set to 1.0000000000000016e-33
train Loss: 0.0031 Acc: 0.9496
val Loss: 0.0179 Acc: 0.5280

Epoch 94/95
----------
train Loss: 0.0032 Acc: 0.9496
val Loss: 0.0184 Acc: 0.5248

Epoch 95/95
----------
train Loss: 0.0032 Acc: 0.9536
val Loss: 0.0179 Acc: 0.5248

Training complete in 9m 25s
Best val Acc: 0.534161

---Testing---
Test accuracy: 0.830855
--------------------
Accuracy of Albacore tuna : 82 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of Bigeye tuna : 73 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 83 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 85 %
Accuracy of Longtail tuna : 93 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 55 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 28 %
Accuracy of Southern bluefin tuna : 65 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.7679158032691341, std: 0.17284938640014363

Model saved in "./weights/tuna_fish_[0.95]_mean[0.93]_std[0.05].save".
--------------------

run info[val: 0.1, epoch: 53, randcrop: False, decay: 10]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0260 Acc: 0.1775
val Loss: 0.0375 Acc: 0.2617

Epoch 1/52
----------
train Loss: 0.0208 Acc: 0.3457
val Loss: 0.0375 Acc: 0.4206

Epoch 2/52
----------
train Loss: 0.0180 Acc: 0.4324
val Loss: 0.0269 Acc: 0.4393

Epoch 3/52
----------
train Loss: 0.0156 Acc: 0.5232
val Loss: 0.0399 Acc: 0.4206

Epoch 4/52
----------
train Loss: 0.0143 Acc: 0.5769
val Loss: 0.0317 Acc: 0.4673

Epoch 5/52
----------
train Loss: 0.0133 Acc: 0.6202
val Loss: 0.0302 Acc: 0.4486

Epoch 6/52
----------
train Loss: 0.0127 Acc: 0.6171
val Loss: 0.0326 Acc: 0.4766

Epoch 7/52
----------
train Loss: 0.0118 Acc: 0.6502
val Loss: 0.0290 Acc: 0.5327

Epoch 8/52
----------
train Loss: 0.0112 Acc: 0.6832
val Loss: 0.0301 Acc: 0.5327

Epoch 9/52
----------
train Loss: 0.0103 Acc: 0.7141
val Loss: 0.0329 Acc: 0.4673

Epoch 10/52
----------
LR is set to 0.001
train Loss: 0.0102 Acc: 0.7203
val Loss: 0.0302 Acc: 0.5234

Epoch 11/52
----------
train Loss: 0.0097 Acc: 0.7389
val Loss: 0.0322 Acc: 0.5421

Epoch 12/52
----------
train Loss: 0.0095 Acc: 0.7534
val Loss: 0.0314 Acc: 0.5327

Epoch 13/52
----------
train Loss: 0.0097 Acc: 0.7554
val Loss: 0.0239 Acc: 0.5327

Epoch 14/52
----------
train Loss: 0.0095 Acc: 0.7513
val Loss: 0.0278 Acc: 0.5421

Epoch 15/52
----------
train Loss: 0.0097 Acc: 0.7472
val Loss: 0.0364 Acc: 0.5234

Epoch 16/52
----------
train Loss: 0.0096 Acc: 0.7337
val Loss: 0.0294 Acc: 0.5327

Epoch 17/52
----------
train Loss: 0.0094 Acc: 0.7647
val Loss: 0.0421 Acc: 0.5421

Epoch 18/52
----------
train Loss: 0.0094 Acc: 0.7554
val Loss: 0.0316 Acc: 0.5421

Epoch 19/52
----------
train Loss: 0.0095 Acc: 0.7523
val Loss: 0.0281 Acc: 0.5327

Epoch 20/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0094 Acc: 0.7461
val Loss: 0.0314 Acc: 0.5327

Epoch 21/52
----------
train Loss: 0.0092 Acc: 0.7503
val Loss: 0.0267 Acc: 0.5327

Epoch 22/52
----------
train Loss: 0.0093 Acc: 0.7564
val Loss: 0.0382 Acc: 0.5421

Epoch 23/52
----------
train Loss: 0.0093 Acc: 0.7554
val Loss: 0.0258 Acc: 0.5327

Epoch 24/52
----------
train Loss: 0.0093 Acc: 0.7564
val Loss: 0.0262 Acc: 0.5327

Epoch 25/52
----------
train Loss: 0.0093 Acc: 0.7513
val Loss: 0.0263 Acc: 0.5421

Epoch 26/52
----------
train Loss: 0.0093 Acc: 0.7688
val Loss: 0.0303 Acc: 0.5234

Epoch 27/52
----------
train Loss: 0.0093 Acc: 0.7595
val Loss: 0.0305 Acc: 0.5421

Epoch 28/52
----------
train Loss: 0.0092 Acc: 0.7523
val Loss: 0.0354 Acc: 0.5421

Epoch 29/52
----------
train Loss: 0.0093 Acc: 0.7503
val Loss: 0.0280 Acc: 0.5234

Epoch 30/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0092 Acc: 0.7822
val Loss: 0.0249 Acc: 0.5234

Epoch 31/52
----------
train Loss: 0.0093 Acc: 0.7595
val Loss: 0.0296 Acc: 0.5234

Epoch 32/52
----------
train Loss: 0.0092 Acc: 0.7492
val Loss: 0.0308 Acc: 0.5421

Epoch 33/52
----------
train Loss: 0.0091 Acc: 0.7709
val Loss: 0.0359 Acc: 0.5421

Epoch 34/52
----------
train Loss: 0.0093 Acc: 0.7523
val Loss: 0.0349 Acc: 0.5327

Epoch 35/52
----------
train Loss: 0.0093 Acc: 0.7564
val Loss: 0.0293 Acc: 0.5327

Epoch 36/52
----------
train Loss: 0.0093 Acc: 0.7606
val Loss: 0.0317 Acc: 0.5421

Epoch 37/52
----------
train Loss: 0.0093 Acc: 0.7585
val Loss: 0.0226 Acc: 0.5327

Epoch 38/52
----------
train Loss: 0.0093 Acc: 0.7595
val Loss: 0.0275 Acc: 0.5421

Epoch 39/52
----------
train Loss: 0.0094 Acc: 0.7492
val Loss: 0.0325 Acc: 0.5327

Epoch 40/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0092 Acc: 0.7678
val Loss: 0.0286 Acc: 0.5421

Epoch 41/52
----------
train Loss: 0.0094 Acc: 0.7544
val Loss: 0.0310 Acc: 0.5421

Epoch 42/52
----------
train Loss: 0.0093 Acc: 0.7678
val Loss: 0.0359 Acc: 0.5327

Epoch 43/52
----------
train Loss: 0.0090 Acc: 0.7616
val Loss: 0.0325 Acc: 0.5421

Epoch 44/52
----------
train Loss: 0.0091 Acc: 0.7719
val Loss: 0.0210 Acc: 0.5327

Epoch 45/52
----------
train Loss: 0.0093 Acc: 0.7575
val Loss: 0.0245 Acc: 0.5421

Epoch 46/52
----------
train Loss: 0.0092 Acc: 0.7647
val Loss: 0.0292 Acc: 0.5421

Epoch 47/52
----------
train Loss: 0.0095 Acc: 0.7513
val Loss: 0.0367 Acc: 0.5327

Epoch 48/52
----------
train Loss: 0.0091 Acc: 0.7626
val Loss: 0.0368 Acc: 0.5327

Epoch 49/52
----------
train Loss: 0.0092 Acc: 0.7616
val Loss: 0.0346 Acc: 0.5234

Epoch 50/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0094 Acc: 0.7564
val Loss: 0.0326 Acc: 0.5234

Epoch 51/52
----------
train Loss: 0.0093 Acc: 0.7637
val Loss: 0.0253 Acc: 0.5327

Epoch 52/52
----------
train Loss: 0.0092 Acc: 0.7472
val Loss: 0.0358 Acc: 0.5327

Training complete in 4m 55s
Best val Acc: 0.542056

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0103 Acc: 0.6987
val Loss: 0.0267 Acc: 0.5140

Epoch 1/52
----------
train Loss: 0.0055 Acc: 0.8596
val Loss: 0.0201 Acc: 0.5701

Epoch 2/52
----------
train Loss: 0.0032 Acc: 0.9257
val Loss: 0.0321 Acc: 0.5047

Epoch 3/52
----------
train Loss: 0.0020 Acc: 0.9494
val Loss: 0.0307 Acc: 0.6168

Epoch 4/52
----------
train Loss: 0.0014 Acc: 0.9618
val Loss: 0.0363 Acc: 0.5327

Epoch 5/52
----------
train Loss: 0.0009 Acc: 0.9783
val Loss: 0.0237 Acc: 0.5421

Epoch 6/52
----------
train Loss: 0.0010 Acc: 0.9773
val Loss: 0.0474 Acc: 0.5607

Epoch 7/52
----------
train Loss: 0.0008 Acc: 0.9711
val Loss: 0.0335 Acc: 0.5701

Epoch 8/52
----------
train Loss: 0.0007 Acc: 0.9773
val Loss: 0.0279 Acc: 0.5981

Epoch 9/52
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0302 Acc: 0.5794

Epoch 10/52
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0357 Acc: 0.5794

Epoch 11/52
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0349 Acc: 0.5794

Epoch 12/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0256 Acc: 0.5888

Epoch 13/52
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0370 Acc: 0.5794

Epoch 14/52
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0318 Acc: 0.5794

Epoch 15/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0346 Acc: 0.5888

Epoch 16/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0402 Acc: 0.5981

Epoch 17/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0362 Acc: 0.5981

Epoch 18/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0523 Acc: 0.5981

Epoch 19/52
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0372 Acc: 0.5888

Epoch 20/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0347 Acc: 0.5888

Epoch 21/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0322 Acc: 0.5888

Epoch 22/52
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0356 Acc: 0.5888

Epoch 23/52
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0196 Acc: 0.5794

Epoch 24/52
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0384 Acc: 0.5981

Epoch 25/52
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0386 Acc: 0.5794

Epoch 26/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0315 Acc: 0.5794

Epoch 27/52
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0350 Acc: 0.5794

Epoch 28/52
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0304 Acc: 0.5794

Epoch 29/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0345 Acc: 0.5794

Epoch 30/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0272 Acc: 0.5794

Epoch 31/52
----------
train Loss: 0.0004 Acc: 0.9804
val Loss: 0.0334 Acc: 0.5888

Epoch 32/52
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0309 Acc: 0.5794

Epoch 33/52
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0228 Acc: 0.5888

Epoch 34/52
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0230 Acc: 0.5888

Epoch 35/52
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0266 Acc: 0.5888

Epoch 36/52
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0292 Acc: 0.5888

Epoch 37/52
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0334 Acc: 0.5888

Epoch 38/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0257 Acc: 0.5794

Epoch 39/52
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0353 Acc: 0.5794

Epoch 40/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0434 Acc: 0.5794

Epoch 41/52
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0303 Acc: 0.5794

Epoch 42/52
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0353 Acc: 0.5888

Epoch 43/52
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0394 Acc: 0.5888

Epoch 44/52
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0278 Acc: 0.5888

Epoch 45/52
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0298 Acc: 0.5794

Epoch 46/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0283 Acc: 0.5888

Epoch 47/52
----------
train Loss: 0.0003 Acc: 0.9835
val Loss: 0.0343 Acc: 0.5794

Epoch 48/52
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0412 Acc: 0.5888

Epoch 49/52
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0295 Acc: 0.5794

Epoch 50/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0345 Acc: 0.5888

Epoch 51/52
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0273 Acc: 0.5888

Epoch 52/52
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0228 Acc: 0.5794

Training complete in 5m 17s
Best val Acc: 0.616822

---Testing---
Test accuracy: 0.924721
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 95 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 72 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 79 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8970182780341623, std: 0.08420328904773504
--------------------

run info[val: 0.15, epoch: 80, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0282 Acc: 0.1672
val Loss: 0.0291 Acc: 0.2733

Epoch 1/79
----------
train Loss: 0.0230 Acc: 0.3290
val Loss: 0.0243 Acc: 0.3416

Epoch 2/79
----------
train Loss: 0.0202 Acc: 0.4077
val Loss: 0.0256 Acc: 0.3354

Epoch 3/79
----------
train Loss: 0.0187 Acc: 0.4579
val Loss: 0.0242 Acc: 0.3602

Epoch 4/79
----------
train Loss: 0.0166 Acc: 0.4973
val Loss: 0.0241 Acc: 0.3665

Epoch 5/79
----------
train Loss: 0.0158 Acc: 0.5290
val Loss: 0.0219 Acc: 0.4410

Epoch 6/79
----------
train Loss: 0.0144 Acc: 0.5913
val Loss: 0.0217 Acc: 0.3913

Epoch 7/79
----------
train Loss: 0.0144 Acc: 0.5934
val Loss: 0.0228 Acc: 0.4224

Epoch 8/79
----------
train Loss: 0.0128 Acc: 0.6175
val Loss: 0.0224 Acc: 0.4348

Epoch 9/79
----------
train Loss: 0.0131 Acc: 0.6197
val Loss: 0.0212 Acc: 0.4472

Epoch 10/79
----------
train Loss: 0.0126 Acc: 0.6197
val Loss: 0.0215 Acc: 0.4596

Epoch 11/79
----------
train Loss: 0.0124 Acc: 0.6525
val Loss: 0.0219 Acc: 0.4161

Epoch 12/79
----------
train Loss: 0.0122 Acc: 0.6590
val Loss: 0.0237 Acc: 0.3913

Epoch 13/79
----------
LR is set to 0.001
train Loss: 0.0115 Acc: 0.6546
val Loss: 0.0211 Acc: 0.4783

Epoch 14/79
----------
train Loss: 0.0110 Acc: 0.7104
val Loss: 0.0210 Acc: 0.4658

Epoch 15/79
----------
train Loss: 0.0107 Acc: 0.7104
val Loss: 0.0214 Acc: 0.4410

Epoch 16/79
----------
train Loss: 0.0100 Acc: 0.7169
val Loss: 0.0213 Acc: 0.4348

Epoch 17/79
----------
train Loss: 0.0106 Acc: 0.7016
val Loss: 0.0207 Acc: 0.4472

Epoch 18/79
----------
train Loss: 0.0102 Acc: 0.7246
val Loss: 0.0216 Acc: 0.4534

Epoch 19/79
----------
train Loss: 0.0100 Acc: 0.7301
val Loss: 0.0211 Acc: 0.4534

Epoch 20/79
----------
train Loss: 0.0102 Acc: 0.7421
val Loss: 0.0205 Acc: 0.4534

Epoch 21/79
----------
train Loss: 0.0106 Acc: 0.7421
val Loss: 0.0214 Acc: 0.4472

Epoch 22/79
----------
train Loss: 0.0106 Acc: 0.7104
val Loss: 0.0209 Acc: 0.4658

Epoch 23/79
----------
train Loss: 0.0101 Acc: 0.7333
val Loss: 0.0217 Acc: 0.4783

Epoch 24/79
----------
train Loss: 0.0100 Acc: 0.7443
val Loss: 0.0210 Acc: 0.4472

Epoch 25/79
----------
train Loss: 0.0104 Acc: 0.7246
val Loss: 0.0211 Acc: 0.4410

Epoch 26/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0106 Acc: 0.7224
val Loss: 0.0207 Acc: 0.4472

Epoch 27/79
----------
train Loss: 0.0101 Acc: 0.7301
val Loss: 0.0209 Acc: 0.4472

Epoch 28/79
----------
train Loss: 0.0101 Acc: 0.7224
val Loss: 0.0215 Acc: 0.4472

Epoch 29/79
----------
train Loss: 0.0099 Acc: 0.7301
val Loss: 0.0214 Acc: 0.4472

Epoch 30/79
----------
train Loss: 0.0102 Acc: 0.7377
val Loss: 0.0210 Acc: 0.4410

Epoch 31/79
----------
train Loss: 0.0100 Acc: 0.7246
val Loss: 0.0211 Acc: 0.4472

Epoch 32/79
----------
train Loss: 0.0098 Acc: 0.7322
val Loss: 0.0213 Acc: 0.4410

Epoch 33/79
----------
train Loss: 0.0101 Acc: 0.7322
val Loss: 0.0215 Acc: 0.4410

Epoch 34/79
----------
train Loss: 0.0100 Acc: 0.7311
val Loss: 0.0209 Acc: 0.4410

Epoch 35/79
----------
train Loss: 0.0097 Acc: 0.7333
val Loss: 0.0219 Acc: 0.4472

Epoch 36/79
----------
train Loss: 0.0103 Acc: 0.7366
val Loss: 0.0214 Acc: 0.4348

Epoch 37/79
----------
train Loss: 0.0102 Acc: 0.7355
val Loss: 0.0210 Acc: 0.4286

Epoch 38/79
----------
train Loss: 0.0102 Acc: 0.7355
val Loss: 0.0205 Acc: 0.4348

Epoch 39/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0101 Acc: 0.7169
val Loss: 0.0205 Acc: 0.4410

Epoch 40/79
----------
train Loss: 0.0101 Acc: 0.7213
val Loss: 0.0216 Acc: 0.4410

Epoch 41/79
----------
train Loss: 0.0099 Acc: 0.7432
val Loss: 0.0210 Acc: 0.4410

Epoch 42/79
----------
train Loss: 0.0099 Acc: 0.7443
val Loss: 0.0210 Acc: 0.4348

Epoch 43/79
----------
train Loss: 0.0097 Acc: 0.7421
val Loss: 0.0206 Acc: 0.4534

Epoch 44/79
----------
train Loss: 0.0099 Acc: 0.7421
val Loss: 0.0207 Acc: 0.4472

Epoch 45/79
----------
train Loss: 0.0098 Acc: 0.7443
val Loss: 0.0209 Acc: 0.4534

Epoch 46/79
----------
train Loss: 0.0106 Acc: 0.7311
val Loss: 0.0219 Acc: 0.4534

Epoch 47/79
----------
train Loss: 0.0101 Acc: 0.7454
val Loss: 0.0207 Acc: 0.4472

Epoch 48/79
----------
train Loss: 0.0096 Acc: 0.7530
val Loss: 0.0216 Acc: 0.4410

Epoch 49/79
----------
train Loss: 0.0102 Acc: 0.7311
val Loss: 0.0207 Acc: 0.4348

Epoch 50/79
----------
train Loss: 0.0096 Acc: 0.7366
val Loss: 0.0209 Acc: 0.4348

Epoch 51/79
----------
train Loss: 0.0098 Acc: 0.7246
val Loss: 0.0209 Acc: 0.4348

Epoch 52/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0098 Acc: 0.7454
val Loss: 0.0210 Acc: 0.4534

Epoch 53/79
----------
train Loss: 0.0100 Acc: 0.7301
val Loss: 0.0212 Acc: 0.4348

Epoch 54/79
----------
train Loss: 0.0103 Acc: 0.7311
val Loss: 0.0217 Acc: 0.4472

Epoch 55/79
----------
train Loss: 0.0099 Acc: 0.7311
val Loss: 0.0212 Acc: 0.4534

Epoch 56/79
----------
train Loss: 0.0096 Acc: 0.7475
val Loss: 0.0210 Acc: 0.4534

Epoch 57/79
----------
train Loss: 0.0101 Acc: 0.7257
val Loss: 0.0215 Acc: 0.4348

Epoch 58/79
----------
train Loss: 0.0099 Acc: 0.7530
val Loss: 0.0211 Acc: 0.4472

Epoch 59/79
----------
train Loss: 0.0099 Acc: 0.7355
val Loss: 0.0212 Acc: 0.4534

Epoch 60/79
----------
train Loss: 0.0098 Acc: 0.7454
val Loss: 0.0215 Acc: 0.4348

Epoch 61/79
----------
train Loss: 0.0101 Acc: 0.7202
val Loss: 0.0213 Acc: 0.4410

Epoch 62/79
----------
train Loss: 0.0101 Acc: 0.7344
val Loss: 0.0219 Acc: 0.4410

Epoch 63/79
----------
train Loss: 0.0100 Acc: 0.7388
val Loss: 0.0210 Acc: 0.4534

Epoch 64/79
----------
train Loss: 0.0102 Acc: 0.7301
val Loss: 0.0216 Acc: 0.4472

Epoch 65/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0100 Acc: 0.7399
val Loss: 0.0208 Acc: 0.4348

Epoch 66/79
----------
train Loss: 0.0100 Acc: 0.7344
val Loss: 0.0215 Acc: 0.4410

Epoch 67/79
----------
train Loss: 0.0104 Acc: 0.7410
val Loss: 0.0207 Acc: 0.4348

Epoch 68/79
----------
train Loss: 0.0100 Acc: 0.7497
val Loss: 0.0216 Acc: 0.4410

Epoch 69/79
----------
train Loss: 0.0103 Acc: 0.7344
val Loss: 0.0213 Acc: 0.4472

Epoch 70/79
----------
train Loss: 0.0098 Acc: 0.7268
val Loss: 0.0213 Acc: 0.4348

Epoch 71/79
----------
train Loss: 0.0100 Acc: 0.7399
val Loss: 0.0215 Acc: 0.4348

Epoch 72/79
----------
train Loss: 0.0104 Acc: 0.7180
val Loss: 0.0217 Acc: 0.4348

Epoch 73/79
----------
train Loss: 0.0101 Acc: 0.7279
val Loss: 0.0215 Acc: 0.4410

Epoch 74/79
----------
train Loss: 0.0099 Acc: 0.7301
val Loss: 0.0209 Acc: 0.4472

Epoch 75/79
----------
train Loss: 0.0097 Acc: 0.7530
val Loss: 0.0211 Acc: 0.4410

Epoch 76/79
----------
train Loss: 0.0096 Acc: 0.7366
val Loss: 0.0213 Acc: 0.4410

Epoch 77/79
----------
train Loss: 0.0102 Acc: 0.7180
val Loss: 0.0210 Acc: 0.4286

Epoch 78/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0099 Acc: 0.7563
val Loss: 0.0217 Acc: 0.4348

Epoch 79/79
----------
train Loss: 0.0100 Acc: 0.7410
val Loss: 0.0214 Acc: 0.4348

Training complete in 7m 22s
Best val Acc: 0.478261

---Fine tuning.---
Epoch 0/79
----------
LR is set to 0.01
train Loss: 0.0111 Acc: 0.6743
val Loss: 0.0254 Acc: 0.4099

Epoch 1/79
----------
train Loss: 0.0087 Acc: 0.7475
val Loss: 0.0205 Acc: 0.4845

Epoch 2/79
----------
train Loss: 0.0061 Acc: 0.8230
val Loss: 0.0230 Acc: 0.4534

Epoch 3/79
----------
train Loss: 0.0043 Acc: 0.8809
val Loss: 0.0210 Acc: 0.5404

Epoch 4/79
----------
train Loss: 0.0034 Acc: 0.9115
val Loss: 0.0223 Acc: 0.4845

Epoch 5/79
----------
train Loss: 0.0027 Acc: 0.9333
val Loss: 0.0236 Acc: 0.4907

Epoch 6/79
----------
train Loss: 0.0025 Acc: 0.9344
val Loss: 0.0223 Acc: 0.5217

Epoch 7/79
----------
train Loss: 0.0024 Acc: 0.9257
val Loss: 0.0243 Acc: 0.5031

Epoch 8/79
----------
train Loss: 0.0018 Acc: 0.9508
val Loss: 0.0248 Acc: 0.4907

Epoch 9/79
----------
train Loss: 0.0014 Acc: 0.9552
val Loss: 0.0256 Acc: 0.5155

Epoch 10/79
----------
train Loss: 0.0012 Acc: 0.9661
val Loss: 0.0258 Acc: 0.5342

Epoch 11/79
----------
train Loss: 0.0011 Acc: 0.9705
val Loss: 0.0249 Acc: 0.5093

Epoch 12/79
----------
train Loss: 0.0010 Acc: 0.9705
val Loss: 0.0246 Acc: 0.5217

Epoch 13/79
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0252 Acc: 0.5217

Epoch 14/79
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0245 Acc: 0.5342

Epoch 15/79
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0239 Acc: 0.5280

Epoch 16/79
----------
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0236 Acc: 0.5466

Epoch 17/79
----------
train Loss: 0.0011 Acc: 0.9847
val Loss: 0.0245 Acc: 0.5404

Epoch 18/79
----------
train Loss: 0.0006 Acc: 0.9749
val Loss: 0.0246 Acc: 0.5342

Epoch 19/79
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0238 Acc: 0.5590

Epoch 20/79
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0253 Acc: 0.5528

Epoch 21/79
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0242 Acc: 0.5466

Epoch 22/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0247 Acc: 0.5404

Epoch 23/79
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0249 Acc: 0.5466

Epoch 24/79
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0245 Acc: 0.5466

Epoch 25/79
----------
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0249 Acc: 0.5466

Epoch 26/79
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0238 Acc: 0.5404

Epoch 27/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0254 Acc: 0.5342

Epoch 28/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0256 Acc: 0.5342

Epoch 29/79
----------
train Loss: 0.0006 Acc: 0.9847
val Loss: 0.0238 Acc: 0.5404

Epoch 30/79
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0246 Acc: 0.5280

Epoch 31/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0244 Acc: 0.5342

Epoch 32/79
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0253 Acc: 0.5466

Epoch 33/79
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0251 Acc: 0.5466

Epoch 34/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0256 Acc: 0.5342

Epoch 35/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5466

Epoch 36/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5342

Epoch 37/79
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0250 Acc: 0.5652

Epoch 38/79
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0256 Acc: 0.5404

Epoch 39/79
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0256 Acc: 0.5404

Epoch 40/79
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0249 Acc: 0.5528

Epoch 41/79
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0254 Acc: 0.5466

Epoch 42/79
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0244 Acc: 0.5404

Epoch 43/79
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0249 Acc: 0.5404

Epoch 44/79
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0251 Acc: 0.5466

Epoch 45/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0242 Acc: 0.5404

Epoch 46/79
----------
train Loss: 0.0006 Acc: 0.9847
val Loss: 0.0253 Acc: 0.5466

Epoch 47/79
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0250 Acc: 0.5466

Epoch 48/79
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0249 Acc: 0.5466

Epoch 49/79
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0239 Acc: 0.5466

Epoch 50/79
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0242 Acc: 0.5404

Epoch 51/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0256 Acc: 0.5466

Epoch 52/79
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0244 Acc: 0.5466

Epoch 53/79
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0252 Acc: 0.5404

Epoch 54/79
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0241 Acc: 0.5342

Epoch 55/79
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0245 Acc: 0.5342

Epoch 56/79
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0235 Acc: 0.5404

Epoch 57/79
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0244 Acc: 0.5466

Epoch 58/79
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0251 Acc: 0.5590

Epoch 59/79
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0243 Acc: 0.5528

Epoch 60/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0251 Acc: 0.5652

Epoch 61/79
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0246 Acc: 0.5466

Epoch 62/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5466

Epoch 63/79
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0244 Acc: 0.5404

Epoch 64/79
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0248 Acc: 0.5404

Epoch 65/79
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0009 Acc: 0.9770
val Loss: 0.0257 Acc: 0.5528

Epoch 66/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0261 Acc: 0.5466

Epoch 67/79
----------
train Loss: 0.0006 Acc: 0.9760
val Loss: 0.0259 Acc: 0.5342

Epoch 68/79
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0238 Acc: 0.5466

Epoch 69/79
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0238 Acc: 0.5466

Epoch 70/79
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0257 Acc: 0.5404

Epoch 71/79
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0235 Acc: 0.5466

Epoch 72/79
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0235 Acc: 0.5404

Epoch 73/79
----------
train Loss: 0.0007 Acc: 0.9749
val Loss: 0.0250 Acc: 0.5528

Epoch 74/79
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0252 Acc: 0.5466

Epoch 75/79
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0243 Acc: 0.5466

Epoch 76/79
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0246 Acc: 0.5466

Epoch 77/79
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0248 Acc: 0.5466

Epoch 78/79
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0251 Acc: 0.5528

Epoch 79/79
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0254 Acc: 0.5466

Training complete in 7m 51s
Best val Acc: 0.565217

---Testing---
Test accuracy: 0.921004
--------------------
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 82 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 88 %
Accuracy of Pacific bluefin tuna : 92 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 82 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8965653736591032, std: 0.07104634866600779
--------------------

run info[val: 0.2, epoch: 71, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0276 Acc: 0.1359
val Loss: 0.0334 Acc: 0.2326

Epoch 1/70
----------
train Loss: 0.0219 Acc: 0.3449
val Loss: 0.0262 Acc: 0.3628

Epoch 2/70
----------
train Loss: 0.0181 Acc: 0.4623
val Loss: 0.0251 Acc: 0.3767

Epoch 3/70
----------
train Loss: 0.0158 Acc: 0.5343
val Loss: 0.0258 Acc: 0.3953

Epoch 4/70
----------
train Loss: 0.0144 Acc: 0.5830
val Loss: 0.0263 Acc: 0.4326

Epoch 5/70
----------
train Loss: 0.0132 Acc: 0.6086
val Loss: 0.0233 Acc: 0.4186

Epoch 6/70
----------
train Loss: 0.0124 Acc: 0.6620
val Loss: 0.0228 Acc: 0.4558

Epoch 7/70
----------
train Loss: 0.0115 Acc: 0.6562
val Loss: 0.0222 Acc: 0.4512

Epoch 8/70
----------
train Loss: 0.0108 Acc: 0.7038
val Loss: 0.0243 Acc: 0.4558

Epoch 9/70
----------
train Loss: 0.0105 Acc: 0.7085
val Loss: 0.0263 Acc: 0.4512

Epoch 10/70
----------
train Loss: 0.0101 Acc: 0.7062
val Loss: 0.0245 Acc: 0.4651

Epoch 11/70
----------
train Loss: 0.0100 Acc: 0.7305
val Loss: 0.0256 Acc: 0.4651

Epoch 12/70
----------
train Loss: 0.0094 Acc: 0.7468
val Loss: 0.0258 Acc: 0.4698

Epoch 13/70
----------
LR is set to 0.001
train Loss: 0.0087 Acc: 0.7828
val Loss: 0.0221 Acc: 0.4744

Epoch 14/70
----------
train Loss: 0.0085 Acc: 0.7782
val Loss: 0.0250 Acc: 0.4884

Epoch 15/70
----------
train Loss: 0.0085 Acc: 0.7816
val Loss: 0.0219 Acc: 0.4930

Epoch 16/70
----------
train Loss: 0.0084 Acc: 0.7921
val Loss: 0.0244 Acc: 0.4791

Epoch 17/70
----------
train Loss: 0.0082 Acc: 0.7956
val Loss: 0.0262 Acc: 0.4930

Epoch 18/70
----------
train Loss: 0.0083 Acc: 0.7898
val Loss: 0.0248 Acc: 0.4977

Epoch 19/70
----------
train Loss: 0.0082 Acc: 0.7991
val Loss: 0.0236 Acc: 0.4837

Epoch 20/70
----------
train Loss: 0.0082 Acc: 0.8026
val Loss: 0.0246 Acc: 0.4884

Epoch 21/70
----------
train Loss: 0.0082 Acc: 0.8002
val Loss: 0.0236 Acc: 0.4884

Epoch 22/70
----------
train Loss: 0.0080 Acc: 0.8107
val Loss: 0.0266 Acc: 0.4791

Epoch 23/70
----------
train Loss: 0.0082 Acc: 0.8026
val Loss: 0.0237 Acc: 0.4837

Epoch 24/70
----------
train Loss: 0.0081 Acc: 0.8060
val Loss: 0.0245 Acc: 0.4930

Epoch 25/70
----------
train Loss: 0.0081 Acc: 0.8107
val Loss: 0.0248 Acc: 0.4930

Epoch 26/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0081 Acc: 0.7840
val Loss: 0.0238 Acc: 0.4884

Epoch 27/70
----------
train Loss: 0.0081 Acc: 0.8095
val Loss: 0.0229 Acc: 0.4837

Epoch 28/70
----------
train Loss: 0.0080 Acc: 0.8072
val Loss: 0.0237 Acc: 0.4930

Epoch 29/70
----------
train Loss: 0.0080 Acc: 0.8211
val Loss: 0.0241 Acc: 0.4977

Epoch 30/70
----------
train Loss: 0.0081 Acc: 0.8072
val Loss: 0.0256 Acc: 0.4977

Epoch 31/70
----------
train Loss: 0.0080 Acc: 0.7979
val Loss: 0.0220 Acc: 0.4930

Epoch 32/70
----------
train Loss: 0.0081 Acc: 0.8014
val Loss: 0.0238 Acc: 0.4977

Epoch 33/70
----------
train Loss: 0.0079 Acc: 0.8072
val Loss: 0.0242 Acc: 0.4837

Epoch 34/70
----------
train Loss: 0.0081 Acc: 0.8165
val Loss: 0.0242 Acc: 0.4930

Epoch 35/70
----------
train Loss: 0.0080 Acc: 0.8084
val Loss: 0.0235 Acc: 0.4977

Epoch 36/70
----------
train Loss: 0.0081 Acc: 0.8118
val Loss: 0.0253 Acc: 0.4930

Epoch 37/70
----------
train Loss: 0.0079 Acc: 0.8072
val Loss: 0.0238 Acc: 0.4930

Epoch 38/70
----------
train Loss: 0.0081 Acc: 0.8060
val Loss: 0.0241 Acc: 0.4837

Epoch 39/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0079 Acc: 0.8084
val Loss: 0.0247 Acc: 0.4930

Epoch 40/70
----------
train Loss: 0.0079 Acc: 0.8246
val Loss: 0.0261 Acc: 0.4930

Epoch 41/70
----------
train Loss: 0.0081 Acc: 0.8002
val Loss: 0.0226 Acc: 0.4977

Epoch 42/70
----------
train Loss: 0.0079 Acc: 0.8142
val Loss: 0.0240 Acc: 0.4930

Epoch 43/70
----------
train Loss: 0.0079 Acc: 0.8026
val Loss: 0.0228 Acc: 0.4977

Epoch 44/70
----------
train Loss: 0.0082 Acc: 0.8037
val Loss: 0.0272 Acc: 0.4930

Epoch 45/70
----------
train Loss: 0.0079 Acc: 0.8142
val Loss: 0.0237 Acc: 0.4884

Epoch 46/70
----------
train Loss: 0.0079 Acc: 0.8037
val Loss: 0.0217 Acc: 0.4884

Epoch 47/70
----------
train Loss: 0.0080 Acc: 0.8049
val Loss: 0.0249 Acc: 0.4837

Epoch 48/70
----------
train Loss: 0.0081 Acc: 0.8037
val Loss: 0.0246 Acc: 0.4884

Epoch 49/70
----------
train Loss: 0.0080 Acc: 0.8072
val Loss: 0.0224 Acc: 0.4884

Epoch 50/70
----------
train Loss: 0.0081 Acc: 0.8014
val Loss: 0.0226 Acc: 0.5023

Epoch 51/70
----------
train Loss: 0.0080 Acc: 0.8037
val Loss: 0.0244 Acc: 0.4977

Epoch 52/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0080 Acc: 0.8095
val Loss: 0.0248 Acc: 0.4930

Epoch 53/70
----------
train Loss: 0.0079 Acc: 0.8188
val Loss: 0.0237 Acc: 0.4977

Epoch 54/70
----------
train Loss: 0.0081 Acc: 0.8049
val Loss: 0.0279 Acc: 0.4930

Epoch 55/70
----------
train Loss: 0.0082 Acc: 0.8037
val Loss: 0.0239 Acc: 0.4930

Epoch 56/70
----------
train Loss: 0.0080 Acc: 0.8130
val Loss: 0.0264 Acc: 0.4930

Epoch 57/70
----------
train Loss: 0.0080 Acc: 0.8049
val Loss: 0.0236 Acc: 0.4930

Epoch 58/70
----------
train Loss: 0.0079 Acc: 0.7991
val Loss: 0.0226 Acc: 0.4977

Epoch 59/70
----------
train Loss: 0.0081 Acc: 0.8014
val Loss: 0.0235 Acc: 0.4884

Epoch 60/70
----------
train Loss: 0.0080 Acc: 0.8188
val Loss: 0.0231 Acc: 0.4884

Epoch 61/70
----------
train Loss: 0.0079 Acc: 0.8002
val Loss: 0.0229 Acc: 0.4930

Epoch 62/70
----------
train Loss: 0.0080 Acc: 0.8072
val Loss: 0.0234 Acc: 0.4930

Epoch 63/70
----------
train Loss: 0.0081 Acc: 0.8002
val Loss: 0.0232 Acc: 0.4930

Epoch 64/70
----------
train Loss: 0.0081 Acc: 0.7979
val Loss: 0.0253 Acc: 0.4884

Epoch 65/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0080 Acc: 0.8130
val Loss: 0.0232 Acc: 0.4884

Epoch 66/70
----------
train Loss: 0.0081 Acc: 0.8014
val Loss: 0.0239 Acc: 0.4930

Epoch 67/70
----------
train Loss: 0.0081 Acc: 0.8072
val Loss: 0.0216 Acc: 0.4837

Epoch 68/70
----------
train Loss: 0.0080 Acc: 0.8142
val Loss: 0.0234 Acc: 0.4930

Epoch 69/70
----------
train Loss: 0.0080 Acc: 0.8002
val Loss: 0.0239 Acc: 0.4884

Epoch 70/70
----------
train Loss: 0.0080 Acc: 0.8118
val Loss: 0.0233 Acc: 0.4930

Training complete in 6m 18s
Best val Acc: 0.502326

---Fine tuning.---
Epoch 0/70
----------
LR is set to 0.01
train Loss: 0.0086 Acc: 0.7666
val Loss: 0.0252 Acc: 0.5023

Epoch 1/70
----------
train Loss: 0.0048 Acc: 0.8780
val Loss: 0.0251 Acc: 0.4837

Epoch 2/70
----------
train Loss: 0.0027 Acc: 0.9443
val Loss: 0.0228 Acc: 0.4791

Epoch 3/70
----------
train Loss: 0.0016 Acc: 0.9663
val Loss: 0.0202 Acc: 0.5209

Epoch 4/70
----------
train Loss: 0.0011 Acc: 0.9756
val Loss: 0.0250 Acc: 0.5349

Epoch 5/70
----------
train Loss: 0.0009 Acc: 0.9733
val Loss: 0.0252 Acc: 0.5395

Epoch 6/70
----------
train Loss: 0.0007 Acc: 0.9849
val Loss: 0.0246 Acc: 0.5209

Epoch 7/70
----------
train Loss: 0.0008 Acc: 0.9791
val Loss: 0.0248 Acc: 0.5395

Epoch 8/70
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5488

Epoch 9/70
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5535

Epoch 10/70
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0249 Acc: 0.5581

Epoch 11/70
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0253 Acc: 0.5488

Epoch 12/70
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0261 Acc: 0.5349

Epoch 13/70
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9791
val Loss: 0.0245 Acc: 0.5302

Epoch 14/70
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0248 Acc: 0.5442

Epoch 15/70
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0256 Acc: 0.5442

Epoch 16/70
----------
train Loss: 0.0004 Acc: 0.9849
val Loss: 0.0237 Acc: 0.5488

Epoch 17/70
----------
train Loss: 0.0003 Acc: 0.9803
val Loss: 0.0256 Acc: 0.5442

Epoch 18/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0207 Acc: 0.5442

Epoch 19/70
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0261 Acc: 0.5488

Epoch 20/70
----------
train Loss: 0.0003 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5442

Epoch 21/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0241 Acc: 0.5442

Epoch 22/70
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0244 Acc: 0.5535

Epoch 23/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0265 Acc: 0.5488

Epoch 24/70
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0235 Acc: 0.5535

Epoch 25/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0273 Acc: 0.5581

Epoch 26/70
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0217 Acc: 0.5628

Epoch 27/70
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0270 Acc: 0.5581

Epoch 28/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0210 Acc: 0.5628

Epoch 29/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0239 Acc: 0.5581

Epoch 30/70
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0262 Acc: 0.5535

Epoch 31/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0243 Acc: 0.5628

Epoch 32/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0294 Acc: 0.5628

Epoch 33/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0241 Acc: 0.5581

Epoch 34/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0229 Acc: 0.5535

Epoch 35/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0285 Acc: 0.5628

Epoch 36/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0253 Acc: 0.5581

Epoch 37/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0240 Acc: 0.5488

Epoch 38/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0235 Acc: 0.5488

Epoch 39/70
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0235 Acc: 0.5442

Epoch 40/70
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0260 Acc: 0.5535

Epoch 41/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0244 Acc: 0.5628

Epoch 42/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0257 Acc: 0.5535

Epoch 43/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0266 Acc: 0.5628

Epoch 44/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0257 Acc: 0.5535

Epoch 45/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0280 Acc: 0.5628

Epoch 46/70
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0232 Acc: 0.5628

Epoch 47/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0258 Acc: 0.5442

Epoch 48/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0259 Acc: 0.5581

Epoch 49/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0241 Acc: 0.5395

Epoch 50/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0232 Acc: 0.5488

Epoch 51/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0265 Acc: 0.5628

Epoch 52/70
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0218 Acc: 0.5535

Epoch 53/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0279 Acc: 0.5581

Epoch 54/70
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0254 Acc: 0.5535

Epoch 55/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0273 Acc: 0.5581

Epoch 56/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0225 Acc: 0.5581

Epoch 57/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0245 Acc: 0.5581

Epoch 58/70
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0237 Acc: 0.5581

Epoch 59/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0253 Acc: 0.5628

Epoch 60/70
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0233 Acc: 0.5628

Epoch 61/70
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0276 Acc: 0.5535

Epoch 62/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0262 Acc: 0.5628

Epoch 63/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0255 Acc: 0.5581

Epoch 64/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0262 Acc: 0.5581

Epoch 65/70
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0230 Acc: 0.5628

Epoch 66/70
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0242 Acc: 0.5628

Epoch 67/70
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0265 Acc: 0.5628

Epoch 68/70
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0248 Acc: 0.5535

Epoch 69/70
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0248 Acc: 0.5581

Epoch 70/70
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0282 Acc: 0.5535

Training complete in 6m 40s
Best val Acc: 0.562791

---Testing---
Test accuracy: 0.901487
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 82 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 97 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 79 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8674849855538218, std: 0.08920887357523284
--------------------

run info[val: 0.25, epoch: 97, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0289 Acc: 0.1537
val Loss: 0.0263 Acc: 0.2900

Epoch 1/96
----------
train Loss: 0.0253 Acc: 0.3123
val Loss: 0.0235 Acc: 0.3494

Epoch 2/96
----------
train Loss: 0.0212 Acc: 0.3804
val Loss: 0.0244 Acc: 0.3420

Epoch 3/96
----------
train Loss: 0.0203 Acc: 0.4325
val Loss: 0.0216 Acc: 0.3829

Epoch 4/96
----------
train Loss: 0.0177 Acc: 0.4796
val Loss: 0.0226 Acc: 0.3606

Epoch 5/96
----------
train Loss: 0.0170 Acc: 0.5527
val Loss: 0.0209 Acc: 0.4015

Epoch 6/96
----------
train Loss: 0.0164 Acc: 0.5440
val Loss: 0.0223 Acc: 0.3941

Epoch 7/96
----------
train Loss: 0.0169 Acc: 0.5527
val Loss: 0.0227 Acc: 0.4015

Epoch 8/96
----------
train Loss: 0.0139 Acc: 0.5849
val Loss: 0.0219 Acc: 0.4312

Epoch 9/96
----------
train Loss: 0.0127 Acc: 0.6642
val Loss: 0.0196 Acc: 0.4349

Epoch 10/96
----------
train Loss: 0.0115 Acc: 0.6753
val Loss: 0.0198 Acc: 0.4461

Epoch 11/96
----------
train Loss: 0.0115 Acc: 0.6853
val Loss: 0.0208 Acc: 0.4461

Epoch 12/96
----------
train Loss: 0.0104 Acc: 0.6989
val Loss: 0.0202 Acc: 0.4089

Epoch 13/96
----------
LR is set to 0.001
train Loss: 0.0107 Acc: 0.7162
val Loss: 0.0195 Acc: 0.4535

Epoch 14/96
----------
train Loss: 0.0098 Acc: 0.7757
val Loss: 0.0188 Acc: 0.4944

Epoch 15/96
----------
train Loss: 0.0090 Acc: 0.7720
val Loss: 0.0188 Acc: 0.4907

Epoch 16/96
----------
train Loss: 0.0091 Acc: 0.7993
val Loss: 0.0188 Acc: 0.4758

Epoch 17/96
----------
train Loss: 0.0094 Acc: 0.7819
val Loss: 0.0187 Acc: 0.4870

Epoch 18/96
----------
train Loss: 0.0092 Acc: 0.7732
val Loss: 0.0192 Acc: 0.4870

Epoch 19/96
----------
train Loss: 0.0093 Acc: 0.7708
val Loss: 0.0187 Acc: 0.4870

Epoch 20/96
----------
train Loss: 0.0087 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4833

Epoch 21/96
----------
train Loss: 0.0099 Acc: 0.7906
val Loss: 0.0190 Acc: 0.4870

Epoch 22/96
----------
train Loss: 0.0085 Acc: 0.7831
val Loss: 0.0186 Acc: 0.4796

Epoch 23/96
----------
train Loss: 0.0098 Acc: 0.7844
val Loss: 0.0186 Acc: 0.4610

Epoch 24/96
----------
train Loss: 0.0096 Acc: 0.7993
val Loss: 0.0193 Acc: 0.4907

Epoch 25/96
----------
train Loss: 0.0090 Acc: 0.7918
val Loss: 0.0191 Acc: 0.4833

Epoch 26/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0092 Acc: 0.7819
val Loss: 0.0190 Acc: 0.4796

Epoch 27/96
----------
train Loss: 0.0098 Acc: 0.8042
val Loss: 0.0188 Acc: 0.4870

Epoch 28/96
----------
train Loss: 0.0097 Acc: 0.7844
val Loss: 0.0190 Acc: 0.4907

Epoch 29/96
----------
train Loss: 0.0085 Acc: 0.7918
val Loss: 0.0192 Acc: 0.4907

Epoch 30/96
----------
train Loss: 0.0093 Acc: 0.7931
val Loss: 0.0189 Acc: 0.4870

Epoch 31/96
----------
train Loss: 0.0090 Acc: 0.8079
val Loss: 0.0187 Acc: 0.4907

Epoch 32/96
----------
train Loss: 0.0102 Acc: 0.7770
val Loss: 0.0187 Acc: 0.4870

Epoch 33/96
----------
train Loss: 0.0090 Acc: 0.7968
val Loss: 0.0188 Acc: 0.4796

Epoch 34/96
----------
train Loss: 0.0087 Acc: 0.7968
val Loss: 0.0189 Acc: 0.4721

Epoch 35/96
----------
train Loss: 0.0086 Acc: 0.7906
val Loss: 0.0186 Acc: 0.4758

Epoch 36/96
----------
train Loss: 0.0085 Acc: 0.7918
val Loss: 0.0190 Acc: 0.4721

Epoch 37/96
----------
train Loss: 0.0084 Acc: 0.7993
val Loss: 0.0186 Acc: 0.4758

Epoch 38/96
----------
train Loss: 0.0090 Acc: 0.8017
val Loss: 0.0191 Acc: 0.4758

Epoch 39/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0085 Acc: 0.8092
val Loss: 0.0190 Acc: 0.4870

Epoch 40/96
----------
train Loss: 0.0091 Acc: 0.7906
val Loss: 0.0186 Acc: 0.4647

Epoch 41/96
----------
train Loss: 0.0086 Acc: 0.8030
val Loss: 0.0187 Acc: 0.4796

Epoch 42/96
----------
train Loss: 0.0084 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4833

Epoch 43/96
----------
train Loss: 0.0087 Acc: 0.7881
val Loss: 0.0187 Acc: 0.4796

Epoch 44/96
----------
train Loss: 0.0087 Acc: 0.7968
val Loss: 0.0189 Acc: 0.4907

Epoch 45/96
----------
train Loss: 0.0087 Acc: 0.8005
val Loss: 0.0188 Acc: 0.4796

Epoch 46/96
----------
train Loss: 0.0087 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4721

Epoch 47/96
----------
train Loss: 0.0091 Acc: 0.7968
val Loss: 0.0188 Acc: 0.4907

Epoch 48/96
----------
train Loss: 0.0091 Acc: 0.8055
val Loss: 0.0192 Acc: 0.4870

Epoch 49/96
----------
train Loss: 0.0087 Acc: 0.7918
val Loss: 0.0188 Acc: 0.4758

Epoch 50/96
----------
train Loss: 0.0090 Acc: 0.7856
val Loss: 0.0187 Acc: 0.4907

Epoch 51/96
----------
train Loss: 0.0090 Acc: 0.8042
val Loss: 0.0188 Acc: 0.4833

Epoch 52/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0099 Acc: 0.7918
val Loss: 0.0188 Acc: 0.4721

Epoch 53/96
----------
train Loss: 0.0088 Acc: 0.8005
val Loss: 0.0192 Acc: 0.4833

Epoch 54/96
----------
train Loss: 0.0089 Acc: 0.7931
val Loss: 0.0188 Acc: 0.4796

Epoch 55/96
----------
train Loss: 0.0095 Acc: 0.7943
val Loss: 0.0187 Acc: 0.4833

Epoch 56/96
----------
train Loss: 0.0086 Acc: 0.7980
val Loss: 0.0187 Acc: 0.4833

Epoch 57/96
----------
train Loss: 0.0090 Acc: 0.8055
val Loss: 0.0185 Acc: 0.4907

Epoch 58/96
----------
train Loss: 0.0092 Acc: 0.8005
val Loss: 0.0188 Acc: 0.4870

Epoch 59/96
----------
train Loss: 0.0099 Acc: 0.7831
val Loss: 0.0186 Acc: 0.4870

Epoch 60/96
----------
train Loss: 0.0088 Acc: 0.8079
val Loss: 0.0185 Acc: 0.4758

Epoch 61/96
----------
train Loss: 0.0090 Acc: 0.8005
val Loss: 0.0192 Acc: 0.4796

Epoch 62/96
----------
train Loss: 0.0095 Acc: 0.7844
val Loss: 0.0186 Acc: 0.4684

Epoch 63/96
----------
train Loss: 0.0102 Acc: 0.7844
val Loss: 0.0189 Acc: 0.4758

Epoch 64/96
----------
train Loss: 0.0090 Acc: 0.7943
val Loss: 0.0190 Acc: 0.4758

Epoch 65/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0087 Acc: 0.7993
val Loss: 0.0190 Acc: 0.4721

Epoch 66/96
----------
train Loss: 0.0091 Acc: 0.7980
val Loss: 0.0187 Acc: 0.4721

Epoch 67/96
----------
train Loss: 0.0095 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4796

Epoch 68/96
----------
train Loss: 0.0084 Acc: 0.7943
val Loss: 0.0187 Acc: 0.4833

Epoch 69/96
----------
train Loss: 0.0096 Acc: 0.8055
val Loss: 0.0189 Acc: 0.4684

Epoch 70/96
----------
train Loss: 0.0085 Acc: 0.8067
val Loss: 0.0187 Acc: 0.4684

Epoch 71/96
----------
train Loss: 0.0098 Acc: 0.7906
val Loss: 0.0187 Acc: 0.4796

Epoch 72/96
----------
train Loss: 0.0087 Acc: 0.8042
val Loss: 0.0188 Acc: 0.4684

Epoch 73/96
----------
train Loss: 0.0083 Acc: 0.8116
val Loss: 0.0185 Acc: 0.4870

Epoch 74/96
----------
train Loss: 0.0095 Acc: 0.7931
val Loss: 0.0190 Acc: 0.4758

Epoch 75/96
----------
train Loss: 0.0093 Acc: 0.8005
val Loss: 0.0189 Acc: 0.4833

Epoch 76/96
----------
train Loss: 0.0083 Acc: 0.8116
val Loss: 0.0184 Acc: 0.4796

Epoch 77/96
----------
train Loss: 0.0088 Acc: 0.7980
val Loss: 0.0189 Acc: 0.4796

Epoch 78/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0084 Acc: 0.8154
val Loss: 0.0186 Acc: 0.4944

Epoch 79/96
----------
train Loss: 0.0090 Acc: 0.7968
val Loss: 0.0188 Acc: 0.4796

Epoch 80/96
----------
train Loss: 0.0096 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4684

Epoch 81/96
----------
train Loss: 0.0091 Acc: 0.7881
val Loss: 0.0187 Acc: 0.4796

Epoch 82/96
----------
train Loss: 0.0088 Acc: 0.7993
val Loss: 0.0186 Acc: 0.4796

Epoch 83/96
----------
train Loss: 0.0097 Acc: 0.7906
val Loss: 0.0188 Acc: 0.4721

Epoch 84/96
----------
train Loss: 0.0091 Acc: 0.7955
val Loss: 0.0188 Acc: 0.4721

Epoch 85/96
----------
train Loss: 0.0094 Acc: 0.7856
val Loss: 0.0188 Acc: 0.4758

Epoch 86/96
----------
train Loss: 0.0085 Acc: 0.7931
val Loss: 0.0190 Acc: 0.4758

Epoch 87/96
----------
train Loss: 0.0101 Acc: 0.7819
val Loss: 0.0187 Acc: 0.4758

Epoch 88/96
----------
train Loss: 0.0086 Acc: 0.7943
val Loss: 0.0184 Acc: 0.4721

Epoch 89/96
----------
train Loss: 0.0085 Acc: 0.7893
val Loss: 0.0189 Acc: 0.4758

Epoch 90/96
----------
train Loss: 0.0088 Acc: 0.7893
val Loss: 0.0185 Acc: 0.4796

Epoch 91/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0100 Acc: 0.7943
val Loss: 0.0188 Acc: 0.4833

Epoch 92/96
----------
train Loss: 0.0087 Acc: 0.8005
val Loss: 0.0188 Acc: 0.4833

Epoch 93/96
----------
train Loss: 0.0089 Acc: 0.7893
val Loss: 0.0187 Acc: 0.4796

Epoch 94/96
----------
train Loss: 0.0088 Acc: 0.8005
val Loss: 0.0190 Acc: 0.4796

Epoch 95/96
----------
train Loss: 0.0085 Acc: 0.8141
val Loss: 0.0188 Acc: 0.4684

Epoch 96/96
----------
train Loss: 0.0093 Acc: 0.8005
val Loss: 0.0189 Acc: 0.4758

Training complete in 8m 49s
Best val Acc: 0.494424

---Fine tuning.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0102 Acc: 0.7546
val Loss: 0.0246 Acc: 0.3494

Epoch 1/96
----------
train Loss: 0.0095 Acc: 0.7348
val Loss: 0.0271 Acc: 0.4126

Epoch 2/96
----------
train Loss: 0.0067 Acc: 0.7918
val Loss: 0.0273 Acc: 0.4164

Epoch 3/96
----------
train Loss: 0.0049 Acc: 0.8662
val Loss: 0.0279 Acc: 0.4424

Epoch 4/96
----------
train Loss: 0.0053 Acc: 0.8699
val Loss: 0.0251 Acc: 0.4833

Epoch 5/96
----------
train Loss: 0.0044 Acc: 0.8637
val Loss: 0.0280 Acc: 0.4349

Epoch 6/96
----------
train Loss: 0.0042 Acc: 0.9095
val Loss: 0.0272 Acc: 0.4981

Epoch 7/96
----------
train Loss: 0.0055 Acc: 0.8340
val Loss: 0.0297 Acc: 0.4089

Epoch 8/96
----------
train Loss: 0.0045 Acc: 0.8984
val Loss: 0.0254 Acc: 0.4833

Epoch 9/96
----------
train Loss: 0.0048 Acc: 0.8699
val Loss: 0.0275 Acc: 0.3680

Epoch 10/96
----------
train Loss: 0.0037 Acc: 0.9071
val Loss: 0.0337 Acc: 0.3941

Epoch 11/96
----------
train Loss: 0.0048 Acc: 0.8786
val Loss: 0.0380 Acc: 0.3978

Epoch 12/96
----------
train Loss: 0.0033 Acc: 0.9244
val Loss: 0.0366 Acc: 0.4164

Epoch 13/96
----------
LR is set to 0.001
train Loss: 0.0020 Acc: 0.9517
val Loss: 0.0274 Acc: 0.4535

Epoch 14/96
----------
train Loss: 0.0019 Acc: 0.9591
val Loss: 0.0239 Acc: 0.5019

Epoch 15/96
----------
train Loss: 0.0011 Acc: 0.9690
val Loss: 0.0232 Acc: 0.4981

Epoch 16/96
----------
train Loss: 0.0015 Acc: 0.9789
val Loss: 0.0240 Acc: 0.5093

Epoch 17/96
----------
train Loss: 0.0018 Acc: 0.9802
val Loss: 0.0230 Acc: 0.5093

Epoch 18/96
----------
train Loss: 0.0011 Acc: 0.9839
val Loss: 0.0229 Acc: 0.5390

Epoch 19/96
----------
train Loss: 0.0018 Acc: 0.9765
val Loss: 0.0233 Acc: 0.5167

Epoch 20/96
----------
train Loss: 0.0014 Acc: 0.9851
val Loss: 0.0229 Acc: 0.5279

Epoch 21/96
----------
train Loss: 0.0008 Acc: 0.9851
val Loss: 0.0224 Acc: 0.5167

Epoch 22/96
----------
train Loss: 0.0009 Acc: 0.9839
val Loss: 0.0224 Acc: 0.5353

Epoch 23/96
----------
train Loss: 0.0007 Acc: 0.9827
val Loss: 0.0230 Acc: 0.5167

Epoch 24/96
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0234 Acc: 0.5167

Epoch 25/96
----------
train Loss: 0.0011 Acc: 0.9839
val Loss: 0.0231 Acc: 0.5204

Epoch 26/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0225 Acc: 0.5353

Epoch 27/96
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0224 Acc: 0.5428

Epoch 28/96
----------
train Loss: 0.0007 Acc: 0.9789
val Loss: 0.0227 Acc: 0.5242

Epoch 29/96
----------
train Loss: 0.0006 Acc: 0.9839
val Loss: 0.0222 Acc: 0.5390

Epoch 30/96
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0222 Acc: 0.5316

Epoch 31/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0223 Acc: 0.5167

Epoch 32/96
----------
train Loss: 0.0006 Acc: 0.9901
val Loss: 0.0223 Acc: 0.5353

Epoch 33/96
----------
train Loss: 0.0008 Acc: 0.9839
val Loss: 0.0223 Acc: 0.5316

Epoch 34/96
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0228 Acc: 0.5279

Epoch 35/96
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0225 Acc: 0.5353

Epoch 36/96
----------
train Loss: 0.0006 Acc: 0.9888
val Loss: 0.0222 Acc: 0.5390

Epoch 37/96
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0223 Acc: 0.5353

Epoch 38/96
----------
train Loss: 0.0006 Acc: 0.9851
val Loss: 0.0223 Acc: 0.5279

Epoch 39/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9851
val Loss: 0.0225 Acc: 0.5316

Epoch 40/96
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0223 Acc: 0.5204

Epoch 41/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0231 Acc: 0.5279

Epoch 42/96
----------
train Loss: 0.0004 Acc: 0.9864
val Loss: 0.0228 Acc: 0.5353

Epoch 43/96
----------
train Loss: 0.0017 Acc: 0.9851
val Loss: 0.0223 Acc: 0.5279

Epoch 44/96
----------
train Loss: 0.0004 Acc: 0.9864
val Loss: 0.0224 Acc: 0.5390

Epoch 45/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0226 Acc: 0.5316

Epoch 46/96
----------
train Loss: 0.0008 Acc: 0.9851
val Loss: 0.0223 Acc: 0.5353

Epoch 47/96
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0221 Acc: 0.5316

Epoch 48/96
----------
train Loss: 0.0005 Acc: 0.9851
val Loss: 0.0222 Acc: 0.5428

Epoch 49/96
----------
train Loss: 0.0016 Acc: 0.9839
val Loss: 0.0226 Acc: 0.5428

Epoch 50/96
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0227 Acc: 0.5204

Epoch 51/96
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0226 Acc: 0.5353

Epoch 52/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9827
val Loss: 0.0229 Acc: 0.5279

Epoch 53/96
----------
train Loss: 0.0007 Acc: 0.9839
val Loss: 0.0231 Acc: 0.5242

Epoch 54/96
----------
train Loss: 0.0008 Acc: 0.9851
val Loss: 0.0230 Acc: 0.5242

Epoch 55/96
----------
train Loss: 0.0022 Acc: 0.9839
val Loss: 0.0228 Acc: 0.5279

Epoch 56/96
----------
train Loss: 0.0004 Acc: 0.9851
val Loss: 0.0220 Acc: 0.5353

Epoch 57/96
----------
train Loss: 0.0023 Acc: 0.9839
val Loss: 0.0226 Acc: 0.5390

Epoch 58/96
----------
train Loss: 0.0006 Acc: 0.9926
val Loss: 0.0224 Acc: 0.5353

Epoch 59/96
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0225 Acc: 0.5390

Epoch 60/96
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0228 Acc: 0.5428

Epoch 61/96
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0229 Acc: 0.5204

Epoch 62/96
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0225 Acc: 0.5204

Epoch 63/96
----------
train Loss: 0.0009 Acc: 0.9851
val Loss: 0.0229 Acc: 0.5204

Epoch 64/96
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0225 Acc: 0.5279

Epoch 65/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0011 Acc: 0.9876
val Loss: 0.0223 Acc: 0.5279

Epoch 66/96
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0220 Acc: 0.5279

Epoch 67/96
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0224 Acc: 0.5316

Epoch 68/96
----------
train Loss: 0.0009 Acc: 0.9839
val Loss: 0.0229 Acc: 0.5204

Epoch 69/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0216 Acc: 0.5242

Epoch 70/96
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0221 Acc: 0.5316

Epoch 71/96
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0225 Acc: 0.5353

Epoch 72/96
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0226 Acc: 0.5279

Epoch 73/96
----------
train Loss: 0.0004 Acc: 0.9864
val Loss: 0.0221 Acc: 0.5279

Epoch 74/96
----------
train Loss: 0.0005 Acc: 0.9926
val Loss: 0.0223 Acc: 0.5353

Epoch 75/96
----------
train Loss: 0.0005 Acc: 0.9901
val Loss: 0.0223 Acc: 0.5353

Epoch 76/96
----------
train Loss: 0.0006 Acc: 0.9839
val Loss: 0.0225 Acc: 0.5316

Epoch 77/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0230 Acc: 0.5390

Epoch 78/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0225 Acc: 0.5390

Epoch 79/96
----------
train Loss: 0.0008 Acc: 0.9864
val Loss: 0.0222 Acc: 0.5390

Epoch 80/96
----------
train Loss: 0.0006 Acc: 0.9827
val Loss: 0.0221 Acc: 0.5353

Epoch 81/96
----------
train Loss: 0.0004 Acc: 0.9913
val Loss: 0.0225 Acc: 0.5316

Epoch 82/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0230 Acc: 0.5353

Epoch 83/96
----------
train Loss: 0.0011 Acc: 0.9888
val Loss: 0.0225 Acc: 0.5390

Epoch 84/96
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0234 Acc: 0.5353

Epoch 85/96
----------
train Loss: 0.0013 Acc: 0.9864
val Loss: 0.0228 Acc: 0.5279

Epoch 86/96
----------
train Loss: 0.0010 Acc: 0.9888
val Loss: 0.0224 Acc: 0.5316

Epoch 87/96
----------
train Loss: 0.0005 Acc: 0.9926
val Loss: 0.0222 Acc: 0.5353

Epoch 88/96
----------
train Loss: 0.0010 Acc: 0.9876
val Loss: 0.0230 Acc: 0.5390

Epoch 89/96
----------
train Loss: 0.0019 Acc: 0.9839
val Loss: 0.0230 Acc: 0.5316

Epoch 90/96
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0226 Acc: 0.5353

Epoch 91/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0222 Acc: 0.5353

Epoch 92/96
----------
train Loss: 0.0006 Acc: 0.9963
val Loss: 0.0226 Acc: 0.5316

Epoch 93/96
----------
train Loss: 0.0008 Acc: 0.9839
val Loss: 0.0226 Acc: 0.5242

Epoch 94/96
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0222 Acc: 0.5316

Epoch 95/96
----------
train Loss: 0.0006 Acc: 0.9888
val Loss: 0.0220 Acc: 0.5242

Epoch 96/96
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0222 Acc: 0.5204

Training complete in 9m 24s
Best val Acc: 0.542751

---Testing---
Test accuracy: 0.874535
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 92 %
Accuracy of Longtail tuna : 93 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 73 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.8351013251641783, std: 0.10891522695942317
--------------------

run info[val: 0.3, epoch: 92, randcrop: True, decay: 6]

---Training last layer.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0273 Acc: 0.1592
val Loss: 0.0290 Acc: 0.2671

Epoch 1/91
----------
train Loss: 0.0230 Acc: 0.2997
val Loss: 0.0254 Acc: 0.3323

Epoch 2/91
----------
train Loss: 0.0194 Acc: 0.4297
val Loss: 0.0241 Acc: 0.3696

Epoch 3/91
----------
train Loss: 0.0170 Acc: 0.4854
val Loss: 0.0223 Acc: 0.3758

Epoch 4/91
----------
train Loss: 0.0159 Acc: 0.5279
val Loss: 0.0225 Acc: 0.4161

Epoch 5/91
----------
train Loss: 0.0149 Acc: 0.5623
val Loss: 0.0211 Acc: 0.4379

Epoch 6/91
----------
LR is set to 0.001
train Loss: 0.0137 Acc: 0.6220
val Loss: 0.0202 Acc: 0.4472

Epoch 7/91
----------
train Loss: 0.0133 Acc: 0.6220
val Loss: 0.0221 Acc: 0.4472

Epoch 8/91
----------
train Loss: 0.0132 Acc: 0.6326
val Loss: 0.0202 Acc: 0.4503

Epoch 9/91
----------
train Loss: 0.0134 Acc: 0.6141
val Loss: 0.0205 Acc: 0.4596

Epoch 10/91
----------
train Loss: 0.0130 Acc: 0.6459
val Loss: 0.0205 Acc: 0.4596

Epoch 11/91
----------
train Loss: 0.0131 Acc: 0.6247
val Loss: 0.0216 Acc: 0.4596

Epoch 12/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0131 Acc: 0.6233
val Loss: 0.0204 Acc: 0.4565

Epoch 13/91
----------
train Loss: 0.0129 Acc: 0.6366
val Loss: 0.0209 Acc: 0.4596

Epoch 14/91
----------
train Loss: 0.0129 Acc: 0.6618
val Loss: 0.0209 Acc: 0.4596

Epoch 15/91
----------
train Loss: 0.0131 Acc: 0.6340
val Loss: 0.0216 Acc: 0.4565

Epoch 16/91
----------
train Loss: 0.0129 Acc: 0.6353
val Loss: 0.0213 Acc: 0.4565

Epoch 17/91
----------
train Loss: 0.0129 Acc: 0.6684
val Loss: 0.0216 Acc: 0.4596

Epoch 18/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0129 Acc: 0.6313
val Loss: 0.0208 Acc: 0.4596

Epoch 19/91
----------
train Loss: 0.0126 Acc: 0.6525
val Loss: 0.0207 Acc: 0.4565

Epoch 20/91
----------
train Loss: 0.0130 Acc: 0.6406
val Loss: 0.0212 Acc: 0.4565

Epoch 21/91
----------
train Loss: 0.0129 Acc: 0.6353
val Loss: 0.0201 Acc: 0.4565

Epoch 22/91
----------
train Loss: 0.0128 Acc: 0.6459
val Loss: 0.0202 Acc: 0.4565

Epoch 23/91
----------
train Loss: 0.0130 Acc: 0.6406
val Loss: 0.0217 Acc: 0.4565

Epoch 24/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0129 Acc: 0.6552
val Loss: 0.0218 Acc: 0.4534

Epoch 25/91
----------
train Loss: 0.0128 Acc: 0.6406
val Loss: 0.0204 Acc: 0.4565

Epoch 26/91
----------
train Loss: 0.0129 Acc: 0.6393
val Loss: 0.0204 Acc: 0.4596

Epoch 27/91
----------
train Loss: 0.0130 Acc: 0.6419
val Loss: 0.0212 Acc: 0.4596

Epoch 28/91
----------
train Loss: 0.0126 Acc: 0.6340
val Loss: 0.0217 Acc: 0.4596

Epoch 29/91
----------
train Loss: 0.0130 Acc: 0.6379
val Loss: 0.0217 Acc: 0.4627

Epoch 30/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0131 Acc: 0.6485
val Loss: 0.0204 Acc: 0.4596

Epoch 31/91
----------
train Loss: 0.0128 Acc: 0.6552
val Loss: 0.0213 Acc: 0.4534

Epoch 32/91
----------
train Loss: 0.0130 Acc: 0.6432
val Loss: 0.0214 Acc: 0.4596

Epoch 33/91
----------
train Loss: 0.0128 Acc: 0.6605
val Loss: 0.0206 Acc: 0.4596

Epoch 34/91
----------
train Loss: 0.0130 Acc: 0.6512
val Loss: 0.0223 Acc: 0.4596

Epoch 35/91
----------
train Loss: 0.0127 Acc: 0.6485
val Loss: 0.0219 Acc: 0.4565

Epoch 36/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0130 Acc: 0.6326
val Loss: 0.0212 Acc: 0.4596

Epoch 37/91
----------
train Loss: 0.0129 Acc: 0.6631
val Loss: 0.0204 Acc: 0.4596

Epoch 38/91
----------
train Loss: 0.0130 Acc: 0.6393
val Loss: 0.0208 Acc: 0.4534

Epoch 39/91
----------
train Loss: 0.0129 Acc: 0.6538
val Loss: 0.0199 Acc: 0.4565

Epoch 40/91
----------
train Loss: 0.0130 Acc: 0.6313
val Loss: 0.0213 Acc: 0.4565

Epoch 41/91
----------
train Loss: 0.0129 Acc: 0.6353
val Loss: 0.0209 Acc: 0.4596

Epoch 42/91
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0131 Acc: 0.6366
val Loss: 0.0209 Acc: 0.4596

Epoch 43/91
----------
train Loss: 0.0131 Acc: 0.6286
val Loss: 0.0198 Acc: 0.4534

Epoch 44/91
----------
train Loss: 0.0129 Acc: 0.6432
val Loss: 0.0212 Acc: 0.4596

Epoch 45/91
----------
train Loss: 0.0129 Acc: 0.6353
val Loss: 0.0222 Acc: 0.4596

Epoch 46/91
----------
train Loss: 0.0126 Acc: 0.6525
val Loss: 0.0205 Acc: 0.4627

Epoch 47/91
----------
train Loss: 0.0127 Acc: 0.6512
val Loss: 0.0200 Acc: 0.4565

Epoch 48/91
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0129 Acc: 0.6260
val Loss: 0.0211 Acc: 0.4596

Epoch 49/91
----------
train Loss: 0.0128 Acc: 0.6366
val Loss: 0.0209 Acc: 0.4565

Epoch 50/91
----------
train Loss: 0.0130 Acc: 0.6366
val Loss: 0.0215 Acc: 0.4565

Epoch 51/91
----------
train Loss: 0.0130 Acc: 0.6353
val Loss: 0.0210 Acc: 0.4596

Epoch 52/91
----------
train Loss: 0.0130 Acc: 0.6499
val Loss: 0.0208 Acc: 0.4596

Epoch 53/91
----------
train Loss: 0.0128 Acc: 0.6326
val Loss: 0.0215 Acc: 0.4565

Epoch 54/91
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0130 Acc: 0.6393
val Loss: 0.0213 Acc: 0.4565

Epoch 55/91
----------
train Loss: 0.0131 Acc: 0.6233
val Loss: 0.0194 Acc: 0.4596

Epoch 56/91
----------
train Loss: 0.0130 Acc: 0.6485
val Loss: 0.0217 Acc: 0.4534

Epoch 57/91
----------
train Loss: 0.0129 Acc: 0.6459
val Loss: 0.0207 Acc: 0.4627

Epoch 58/91
----------
train Loss: 0.0128 Acc: 0.6578
val Loss: 0.0219 Acc: 0.4534

Epoch 59/91
----------
train Loss: 0.0126 Acc: 0.6485
val Loss: 0.0217 Acc: 0.4565

Epoch 60/91
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0129 Acc: 0.6406
val Loss: 0.0213 Acc: 0.4627

Epoch 61/91
----------
train Loss: 0.0129 Acc: 0.6472
val Loss: 0.0205 Acc: 0.4565

Epoch 62/91
----------
train Loss: 0.0129 Acc: 0.6512
val Loss: 0.0218 Acc: 0.4596

Epoch 63/91
----------
train Loss: 0.0127 Acc: 0.6485
val Loss: 0.0199 Acc: 0.4596

Epoch 64/91
----------
train Loss: 0.0129 Acc: 0.6300
val Loss: 0.0208 Acc: 0.4565

Epoch 65/91
----------
train Loss: 0.0133 Acc: 0.6220
val Loss: 0.0212 Acc: 0.4565

Epoch 66/91
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0129 Acc: 0.6485
val Loss: 0.0200 Acc: 0.4596

Epoch 67/91
----------
train Loss: 0.0127 Acc: 0.6366
val Loss: 0.0210 Acc: 0.4565

Epoch 68/91
----------
train Loss: 0.0130 Acc: 0.6393
val Loss: 0.0216 Acc: 0.4534

Epoch 69/91
----------
train Loss: 0.0130 Acc: 0.6552
val Loss: 0.0215 Acc: 0.4596

Epoch 70/91
----------
train Loss: 0.0128 Acc: 0.6313
val Loss: 0.0212 Acc: 0.4596

Epoch 71/91
----------
train Loss: 0.0129 Acc: 0.6313
val Loss: 0.0212 Acc: 0.4596

Epoch 72/91
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0128 Acc: 0.6353
val Loss: 0.0208 Acc: 0.4596

Epoch 73/91
----------
train Loss: 0.0127 Acc: 0.6512
val Loss: 0.0220 Acc: 0.4596

Epoch 74/91
----------
train Loss: 0.0128 Acc: 0.6459
val Loss: 0.0209 Acc: 0.4596

Epoch 75/91
----------
train Loss: 0.0128 Acc: 0.6340
val Loss: 0.0212 Acc: 0.4565

Epoch 76/91
----------
train Loss: 0.0130 Acc: 0.6472
val Loss: 0.0217 Acc: 0.4565

Epoch 77/91
----------
train Loss: 0.0129 Acc: 0.6300
val Loss: 0.0203 Acc: 0.4534

Epoch 78/91
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0126 Acc: 0.6512
val Loss: 0.0221 Acc: 0.4565

Epoch 79/91
----------
train Loss: 0.0129 Acc: 0.6419
val Loss: 0.0214 Acc: 0.4565

Epoch 80/91
----------
train Loss: 0.0130 Acc: 0.6605
val Loss: 0.0200 Acc: 0.4565

Epoch 81/91
----------
train Loss: 0.0128 Acc: 0.6618
val Loss: 0.0213 Acc: 0.4596

Epoch 82/91
----------
train Loss: 0.0129 Acc: 0.6459
val Loss: 0.0210 Acc: 0.4596

Epoch 83/91
----------
train Loss: 0.0131 Acc: 0.6446
val Loss: 0.0210 Acc: 0.4596

Epoch 84/91
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0130 Acc: 0.6446
val Loss: 0.0216 Acc: 0.4596

Epoch 85/91
----------
train Loss: 0.0127 Acc: 0.6353
val Loss: 0.0205 Acc: 0.4596

Epoch 86/91
----------
train Loss: 0.0131 Acc: 0.6326
val Loss: 0.0202 Acc: 0.4534

Epoch 87/91
----------
train Loss: 0.0127 Acc: 0.6432
val Loss: 0.0202 Acc: 0.4596

Epoch 88/91
----------
train Loss: 0.0129 Acc: 0.6379
val Loss: 0.0214 Acc: 0.4534

Epoch 89/91
----------
train Loss: 0.0128 Acc: 0.6578
val Loss: 0.0206 Acc: 0.4534

Epoch 90/91
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0131 Acc: 0.6313
val Loss: 0.0202 Acc: 0.4534

Epoch 91/91
----------
train Loss: 0.0127 Acc: 0.6618
val Loss: 0.0211 Acc: 0.4534

Training complete in 8m 29s
Best val Acc: 0.462733

---Fine tuning.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0128 Acc: 0.6393
val Loss: 0.0213 Acc: 0.4317

Epoch 1/91
----------
train Loss: 0.0097 Acc: 0.7454
val Loss: 0.0198 Acc: 0.4969

Epoch 2/91
----------
train Loss: 0.0066 Acc: 0.8143
val Loss: 0.0201 Acc: 0.4565

Epoch 3/91
----------
train Loss: 0.0046 Acc: 0.8886
val Loss: 0.0201 Acc: 0.5124

Epoch 4/91
----------
train Loss: 0.0033 Acc: 0.9151
val Loss: 0.0195 Acc: 0.5093

Epoch 5/91
----------
train Loss: 0.0021 Acc: 0.9602
val Loss: 0.0205 Acc: 0.5124

Epoch 6/91
----------
LR is set to 0.001
train Loss: 0.0017 Acc: 0.9682
val Loss: 0.0192 Acc: 0.5466

Epoch 7/91
----------
train Loss: 0.0017 Acc: 0.9668
val Loss: 0.0201 Acc: 0.5528

Epoch 8/91
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0205 Acc: 0.5497

Epoch 9/91
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0197 Acc: 0.5466

Epoch 10/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0213 Acc: 0.5404

Epoch 11/91
----------
train Loss: 0.0012 Acc: 0.9788
val Loss: 0.0191 Acc: 0.5528

Epoch 12/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0012 Acc: 0.9761
val Loss: 0.0195 Acc: 0.5528

Epoch 13/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0213 Acc: 0.5528

Epoch 14/91
----------
train Loss: 0.0012 Acc: 0.9854
val Loss: 0.0192 Acc: 0.5559

Epoch 15/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0203 Acc: 0.5528

Epoch 16/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0218 Acc: 0.5497

Epoch 17/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0204 Acc: 0.5497

Epoch 18/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0010 Acc: 0.9801
val Loss: 0.0207 Acc: 0.5497

Epoch 19/91
----------
train Loss: 0.0010 Acc: 0.9841
val Loss: 0.0204 Acc: 0.5466

Epoch 20/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0222 Acc: 0.5559

Epoch 21/91
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0212 Acc: 0.5497

Epoch 22/91
----------
train Loss: 0.0010 Acc: 0.9867
val Loss: 0.0207 Acc: 0.5559

Epoch 23/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0186 Acc: 0.5559

Epoch 24/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0201 Acc: 0.5528

Epoch 25/91
----------
train Loss: 0.0010 Acc: 0.9867
val Loss: 0.0197 Acc: 0.5528

Epoch 26/91
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0205 Acc: 0.5466

Epoch 27/91
----------
train Loss: 0.0010 Acc: 0.9894
val Loss: 0.0207 Acc: 0.5497

Epoch 28/91
----------
train Loss: 0.0011 Acc: 0.9748
val Loss: 0.0210 Acc: 0.5497

Epoch 29/91
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0222 Acc: 0.5497

Epoch 30/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0010 Acc: 0.9881
val Loss: 0.0197 Acc: 0.5528

Epoch 31/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0213 Acc: 0.5497

Epoch 32/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0209 Acc: 0.5528

Epoch 33/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0202 Acc: 0.5466

Epoch 34/91
----------
train Loss: 0.0011 Acc: 0.9788
val Loss: 0.0199 Acc: 0.5435

Epoch 35/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0209 Acc: 0.5466

Epoch 36/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0010 Acc: 0.9881
val Loss: 0.0208 Acc: 0.5466

Epoch 37/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0215 Acc: 0.5497

Epoch 38/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0203 Acc: 0.5497

Epoch 39/91
----------
train Loss: 0.0011 Acc: 0.9761
val Loss: 0.0193 Acc: 0.5528

Epoch 40/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0206 Acc: 0.5528

Epoch 41/91
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0207 Acc: 0.5559

Epoch 42/91
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0011 Acc: 0.9788
val Loss: 0.0204 Acc: 0.5497

Epoch 43/91
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0192 Acc: 0.5497

Epoch 44/91
----------
train Loss: 0.0010 Acc: 0.9841
val Loss: 0.0201 Acc: 0.5497

Epoch 45/91
----------
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0210 Acc: 0.5435

Epoch 46/91
----------
train Loss: 0.0010 Acc: 0.9881
val Loss: 0.0211 Acc: 0.5528

Epoch 47/91
----------
train Loss: 0.0010 Acc: 0.9854
val Loss: 0.0193 Acc: 0.5528

Epoch 48/91
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0220 Acc: 0.5497

Epoch 49/91
----------
train Loss: 0.0009 Acc: 0.9828
val Loss: 0.0209 Acc: 0.5466

Epoch 50/91
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0208 Acc: 0.5435

Epoch 51/91
----------
train Loss: 0.0010 Acc: 0.9881
val Loss: 0.0200 Acc: 0.5528

Epoch 52/91
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0208 Acc: 0.5466

Epoch 53/91
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5435

Epoch 54/91
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0010 Acc: 0.9867
val Loss: 0.0201 Acc: 0.5435

Epoch 55/91
----------
train Loss: 0.0011 Acc: 0.9788
val Loss: 0.0195 Acc: 0.5466

Epoch 56/91
----------
train Loss: 0.0011 Acc: 0.9775
val Loss: 0.0198 Acc: 0.5497

Epoch 57/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0200 Acc: 0.5497

Epoch 58/91
----------
train Loss: 0.0012 Acc: 0.9814
val Loss: 0.0193 Acc: 0.5528

Epoch 59/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0204 Acc: 0.5559

Epoch 60/91
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0012 Acc: 0.9761
val Loss: 0.0212 Acc: 0.5528

Epoch 61/91
----------
train Loss: 0.0011 Acc: 0.9761
val Loss: 0.0216 Acc: 0.5528

Epoch 62/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0200 Acc: 0.5528

Epoch 63/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0214 Acc: 0.5435

Epoch 64/91
----------
train Loss: 0.0009 Acc: 0.9828
val Loss: 0.0207 Acc: 0.5528

Epoch 65/91
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0203 Acc: 0.5528

Epoch 66/91
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0010 Acc: 0.9881
val Loss: 0.0189 Acc: 0.5559

Epoch 67/91
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0193 Acc: 0.5528

Epoch 68/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0204 Acc: 0.5528

Epoch 69/91
----------
train Loss: 0.0009 Acc: 0.9854
val Loss: 0.0202 Acc: 0.5466

Epoch 70/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0205 Acc: 0.5466

Epoch 71/91
----------
train Loss: 0.0011 Acc: 0.9775
val Loss: 0.0198 Acc: 0.5435

Epoch 72/91
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0010 Acc: 0.9920
val Loss: 0.0202 Acc: 0.5435

Epoch 73/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0185 Acc: 0.5466

Epoch 74/91
----------
train Loss: 0.0011 Acc: 0.9748
val Loss: 0.0201 Acc: 0.5497

Epoch 75/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0188 Acc: 0.5497

Epoch 76/91
----------
train Loss: 0.0009 Acc: 0.9867
val Loss: 0.0202 Acc: 0.5466

Epoch 77/91
----------
train Loss: 0.0012 Acc: 0.9801
val Loss: 0.0213 Acc: 0.5466

Epoch 78/91
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0011 Acc: 0.9761
val Loss: 0.0197 Acc: 0.5497

Epoch 79/91
----------
train Loss: 0.0011 Acc: 0.9841
val Loss: 0.0210 Acc: 0.5528

Epoch 80/91
----------
train Loss: 0.0010 Acc: 0.9828
val Loss: 0.0190 Acc: 0.5497

Epoch 81/91
----------
train Loss: 0.0011 Acc: 0.9814
val Loss: 0.0213 Acc: 0.5497

Epoch 82/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0201 Acc: 0.5497

Epoch 83/91
----------
train Loss: 0.0012 Acc: 0.9775
val Loss: 0.0209 Acc: 0.5497

Epoch 84/91
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0012 Acc: 0.9841
val Loss: 0.0197 Acc: 0.5435

Epoch 85/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0203 Acc: 0.5404

Epoch 86/91
----------
train Loss: 0.0011 Acc: 0.9828
val Loss: 0.0197 Acc: 0.5497

Epoch 87/91
----------
train Loss: 0.0013 Acc: 0.9788
val Loss: 0.0212 Acc: 0.5528

Epoch 88/91
----------
train Loss: 0.0010 Acc: 0.9841
val Loss: 0.0201 Acc: 0.5559

Epoch 89/91
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0201 Acc: 0.5528

Epoch 90/91
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0010 Acc: 0.9841
val Loss: 0.0214 Acc: 0.5497

Epoch 91/91
----------
train Loss: 0.0011 Acc: 0.9801
val Loss: 0.0200 Acc: 0.5528

Training complete in 8m 59s
Best val Acc: 0.555901

---Testing---
Test accuracy: 0.859665
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 75 %
Accuracy of Bigeye tuna : 79 %
Accuracy of Blackfin tuna : 95 %
Accuracy of Bullet tuna : 87 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 84 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 76 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 73 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8071644570722116, std: 0.13207426196424443

Model saved in "./weights/tuna_fish_[0.92]_mean[0.9]_std[0.08].save".
--------------------

run info[val: 0.1, epoch: 50, randcrop: True, decay: 12]

---Training last layer.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0257 Acc: 0.1868
val Loss: 0.0403 Acc: 0.3178

Epoch 1/49
----------
train Loss: 0.0217 Acc: 0.2982
val Loss: 0.0361 Acc: 0.4393

Epoch 2/49
----------
train Loss: 0.0181 Acc: 0.4231
val Loss: 0.0374 Acc: 0.3738

Epoch 3/49
----------
train Loss: 0.0162 Acc: 0.5150
val Loss: 0.0281 Acc: 0.4766

Epoch 4/49
----------
train Loss: 0.0149 Acc: 0.5397
val Loss: 0.0244 Acc: 0.5047

Epoch 5/49
----------
train Loss: 0.0143 Acc: 0.5624
val Loss: 0.0345 Acc: 0.4579

Epoch 6/49
----------
train Loss: 0.0137 Acc: 0.5810
val Loss: 0.0304 Acc: 0.4953

Epoch 7/49
----------
train Loss: 0.0130 Acc: 0.5924
val Loss: 0.0243 Acc: 0.5607

Epoch 8/49
----------
train Loss: 0.0129 Acc: 0.6140
val Loss: 0.0286 Acc: 0.4673

Epoch 9/49
----------
train Loss: 0.0120 Acc: 0.6512
val Loss: 0.0226 Acc: 0.5140

Epoch 10/49
----------
train Loss: 0.0117 Acc: 0.6460
val Loss: 0.0303 Acc: 0.4860

Epoch 11/49
----------
train Loss: 0.0116 Acc: 0.6419
val Loss: 0.0238 Acc: 0.5327

Epoch 12/49
----------
LR is set to 0.001
train Loss: 0.0106 Acc: 0.6821
val Loss: 0.0237 Acc: 0.5234

Epoch 13/49
----------
train Loss: 0.0106 Acc: 0.6976
val Loss: 0.0294 Acc: 0.5140

Epoch 14/49
----------
train Loss: 0.0104 Acc: 0.6935
val Loss: 0.0336 Acc: 0.5140

Epoch 15/49
----------
train Loss: 0.0100 Acc: 0.7152
val Loss: 0.0339 Acc: 0.5140

Epoch 16/49
----------
train Loss: 0.0102 Acc: 0.6987
val Loss: 0.0299 Acc: 0.5140

Epoch 17/49
----------
train Loss: 0.0097 Acc: 0.7461
val Loss: 0.0271 Acc: 0.5140

Epoch 18/49
----------
train Loss: 0.0102 Acc: 0.7121
val Loss: 0.0269 Acc: 0.5140

Epoch 19/49
----------
train Loss: 0.0103 Acc: 0.6976
val Loss: 0.0254 Acc: 0.5140

Epoch 20/49
----------
train Loss: 0.0102 Acc: 0.7028
val Loss: 0.0261 Acc: 0.5234

Epoch 21/49
----------
train Loss: 0.0102 Acc: 0.7100
val Loss: 0.0330 Acc: 0.5140

Epoch 22/49
----------
train Loss: 0.0101 Acc: 0.7162
val Loss: 0.0244 Acc: 0.5140

Epoch 23/49
----------
train Loss: 0.0100 Acc: 0.7214
val Loss: 0.0281 Acc: 0.5047

Epoch 24/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0100 Acc: 0.7327
val Loss: 0.0216 Acc: 0.5047

Epoch 25/49
----------
train Loss: 0.0096 Acc: 0.7430
val Loss: 0.0239 Acc: 0.5047

Epoch 26/49
----------
train Loss: 0.0101 Acc: 0.7141
val Loss: 0.0322 Acc: 0.5047

Epoch 27/49
----------
train Loss: 0.0098 Acc: 0.7317
val Loss: 0.0331 Acc: 0.5047

Epoch 28/49
----------
train Loss: 0.0098 Acc: 0.7276
val Loss: 0.0333 Acc: 0.5047

Epoch 29/49
----------
train Loss: 0.0097 Acc: 0.7203
val Loss: 0.0288 Acc: 0.5047

Epoch 30/49
----------
train Loss: 0.0098 Acc: 0.7224
val Loss: 0.0468 Acc: 0.5047

Epoch 31/49
----------
train Loss: 0.0099 Acc: 0.7059
val Loss: 0.0278 Acc: 0.5140

Epoch 32/49
----------
train Loss: 0.0100 Acc: 0.7255
val Loss: 0.0316 Acc: 0.5047

Epoch 33/49
----------
train Loss: 0.0097 Acc: 0.7348
val Loss: 0.0290 Acc: 0.5047

Epoch 34/49
----------
train Loss: 0.0101 Acc: 0.7183
val Loss: 0.0405 Acc: 0.5047

Epoch 35/49
----------
train Loss: 0.0097 Acc: 0.7265
val Loss: 0.0330 Acc: 0.5047

Epoch 36/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0099 Acc: 0.7121
val Loss: 0.0230 Acc: 0.5047

Epoch 37/49
----------
train Loss: 0.0098 Acc: 0.7276
val Loss: 0.0305 Acc: 0.5047

Epoch 38/49
----------
train Loss: 0.0098 Acc: 0.7245
val Loss: 0.0299 Acc: 0.5047

Epoch 39/49
----------
train Loss: 0.0100 Acc: 0.7110
val Loss: 0.0321 Acc: 0.5047

Epoch 40/49
----------
train Loss: 0.0099 Acc: 0.7214
val Loss: 0.0337 Acc: 0.5047

Epoch 41/49
----------
train Loss: 0.0099 Acc: 0.7152
val Loss: 0.0365 Acc: 0.5047

Epoch 42/49
----------
train Loss: 0.0100 Acc: 0.7214
val Loss: 0.0254 Acc: 0.5047

Epoch 43/49
----------
train Loss: 0.0100 Acc: 0.7255
val Loss: 0.0267 Acc: 0.5047

Epoch 44/49
----------
train Loss: 0.0101 Acc: 0.7079
val Loss: 0.0292 Acc: 0.5047

Epoch 45/49
----------
train Loss: 0.0101 Acc: 0.7131
val Loss: 0.0293 Acc: 0.5047

Epoch 46/49
----------
train Loss: 0.0098 Acc: 0.7327
val Loss: 0.0252 Acc: 0.5047

Epoch 47/49
----------
train Loss: 0.0101 Acc: 0.7059
val Loss: 0.0331 Acc: 0.5047

Epoch 48/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0099 Acc: 0.7152
val Loss: 0.0249 Acc: 0.5047

Epoch 49/49
----------
train Loss: 0.0098 Acc: 0.7183
val Loss: 0.0277 Acc: 0.5047

Training complete in 4m 38s
Best val Acc: 0.560748

---Fine tuning.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0124 Acc: 0.6130
val Loss: 0.0290 Acc: 0.5607

Epoch 1/49
----------
train Loss: 0.0086 Acc: 0.7368
val Loss: 0.0352 Acc: 0.4766

Epoch 2/49
----------
train Loss: 0.0062 Acc: 0.8256
val Loss: 0.0233 Acc: 0.5701

Epoch 3/49
----------
train Loss: 0.0040 Acc: 0.8885
val Loss: 0.0249 Acc: 0.5514

Epoch 4/49
----------
train Loss: 0.0030 Acc: 0.9236
val Loss: 0.0295 Acc: 0.5794

Epoch 5/49
----------
train Loss: 0.0023 Acc: 0.9391
val Loss: 0.0299 Acc: 0.5701

Epoch 6/49
----------
train Loss: 0.0017 Acc: 0.9587
val Loss: 0.0187 Acc: 0.5421

Epoch 7/49
----------
train Loss: 0.0014 Acc: 0.9628
val Loss: 0.0375 Acc: 0.5607

Epoch 8/49
----------
train Loss: 0.0013 Acc: 0.9628
val Loss: 0.0327 Acc: 0.5514

Epoch 9/49
----------
train Loss: 0.0009 Acc: 0.9763
val Loss: 0.0376 Acc: 0.5421

Epoch 10/49
----------
train Loss: 0.0010 Acc: 0.9721
val Loss: 0.0347 Acc: 0.5140

Epoch 11/49
----------
train Loss: 0.0008 Acc: 0.9835
val Loss: 0.0384 Acc: 0.5607

Epoch 12/49
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0369 Acc: 0.5607

Epoch 13/49
----------
train Loss: 0.0006 Acc: 0.9783
val Loss: 0.0315 Acc: 0.5701

Epoch 14/49
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0423 Acc: 0.5607

Epoch 15/49
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0356 Acc: 0.5607

Epoch 16/49
----------
train Loss: 0.0006 Acc: 0.9794
val Loss: 0.0328 Acc: 0.5607

Epoch 17/49
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0292 Acc: 0.5607

Epoch 18/49
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0441 Acc: 0.5514

Epoch 19/49
----------
train Loss: 0.0006 Acc: 0.9773
val Loss: 0.0243 Acc: 0.5514

Epoch 20/49
----------
train Loss: 0.0006 Acc: 0.9783
val Loss: 0.0500 Acc: 0.5514

Epoch 21/49
----------
train Loss: 0.0005 Acc: 0.9783
val Loss: 0.0214 Acc: 0.5607

Epoch 22/49
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0221 Acc: 0.5514

Epoch 23/49
----------
train Loss: 0.0006 Acc: 0.9783
val Loss: 0.0269 Acc: 0.5607

Epoch 24/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0405 Acc: 0.5607

Epoch 25/49
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0371 Acc: 0.5514

Epoch 26/49
----------
train Loss: 0.0006 Acc: 0.9773
val Loss: 0.0186 Acc: 0.5607

Epoch 27/49
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0293 Acc: 0.5514

Epoch 28/49
----------
train Loss: 0.0006 Acc: 0.9804
val Loss: 0.0267 Acc: 0.5514

Epoch 29/49
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0388 Acc: 0.5607

Epoch 30/49
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0350 Acc: 0.5607

Epoch 31/49
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0363 Acc: 0.5607

Epoch 32/49
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0460 Acc: 0.5607

Epoch 33/49
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0458 Acc: 0.5607

Epoch 34/49
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0238 Acc: 0.5607

Epoch 35/49
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0467 Acc: 0.5607

Epoch 36/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0343 Acc: 0.5607

Epoch 37/49
----------
train Loss: 0.0005 Acc: 0.9783
val Loss: 0.0332 Acc: 0.5607

Epoch 38/49
----------
train Loss: 0.0005 Acc: 0.9804
val Loss: 0.0391 Acc: 0.5514

Epoch 39/49
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0243 Acc: 0.5607

Epoch 40/49
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0312 Acc: 0.5607

Epoch 41/49
----------
train Loss: 0.0006 Acc: 0.9752
val Loss: 0.0342 Acc: 0.5607

Epoch 42/49
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0311 Acc: 0.5607

Epoch 43/49
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0340 Acc: 0.5607

Epoch 44/49
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0284 Acc: 0.5607

Epoch 45/49
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0429 Acc: 0.5607

Epoch 46/49
----------
train Loss: 0.0005 Acc: 0.9783
val Loss: 0.0357 Acc: 0.5607

Epoch 47/49
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0203 Acc: 0.5607

Epoch 48/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0285 Acc: 0.5607

Epoch 49/49
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0325 Acc: 0.5607

Training complete in 4m 57s
Best val Acc: 0.579439

---Testing---
Test accuracy: 0.904275
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 60 %
Accuracy of Bigeye tuna : 79 %
Accuracy of Blackfin tuna : 95 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 62 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 93 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 92 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 87 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.8700254028965658, std: 0.11917883871412396
--------------------

run info[val: 0.15, epoch: 92, randcrop: True, decay: 14]

---Training last layer.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0285 Acc: 0.1628
val Loss: 0.0278 Acc: 0.2795

Epoch 1/91
----------
train Loss: 0.0229 Acc: 0.3202
val Loss: 0.0247 Acc: 0.3168

Epoch 2/91
----------
train Loss: 0.0205 Acc: 0.3956
val Loss: 0.0238 Acc: 0.4224

Epoch 3/91
----------
train Loss: 0.0171 Acc: 0.5082
val Loss: 0.0249 Acc: 0.3913

Epoch 4/91
----------
train Loss: 0.0167 Acc: 0.5060
val Loss: 0.0225 Acc: 0.4534

Epoch 5/91
----------
train Loss: 0.0160 Acc: 0.5432
val Loss: 0.0231 Acc: 0.4224

Epoch 6/91
----------
train Loss: 0.0144 Acc: 0.5716
val Loss: 0.0241 Acc: 0.4348

Epoch 7/91
----------
train Loss: 0.0142 Acc: 0.5956
val Loss: 0.0244 Acc: 0.4224

Epoch 8/91
----------
train Loss: 0.0134 Acc: 0.6120
val Loss: 0.0241 Acc: 0.4161

Epoch 9/91
----------
train Loss: 0.0138 Acc: 0.5858
val Loss: 0.0218 Acc: 0.4410

Epoch 10/91
----------
train Loss: 0.0120 Acc: 0.6383
val Loss: 0.0216 Acc: 0.4658

Epoch 11/91
----------
train Loss: 0.0121 Acc: 0.6721
val Loss: 0.0223 Acc: 0.4410

Epoch 12/91
----------
train Loss: 0.0116 Acc: 0.6656
val Loss: 0.0226 Acc: 0.4596

Epoch 13/91
----------
train Loss: 0.0117 Acc: 0.6645
val Loss: 0.0223 Acc: 0.4410

Epoch 14/91
----------
LR is set to 0.001
train Loss: 0.0105 Acc: 0.7049
val Loss: 0.0228 Acc: 0.4534

Epoch 15/91
----------
train Loss: 0.0105 Acc: 0.7093
val Loss: 0.0220 Acc: 0.4720

Epoch 16/91
----------
train Loss: 0.0098 Acc: 0.7290
val Loss: 0.0219 Acc: 0.4596

Epoch 17/91
----------
train Loss: 0.0104 Acc: 0.7202
val Loss: 0.0218 Acc: 0.4596

Epoch 18/91
----------
train Loss: 0.0101 Acc: 0.7158
val Loss: 0.0225 Acc: 0.4472

Epoch 19/91
----------
train Loss: 0.0102 Acc: 0.7115
val Loss: 0.0226 Acc: 0.4596

Epoch 20/91
----------
train Loss: 0.0103 Acc: 0.7301
val Loss: 0.0218 Acc: 0.4596

Epoch 21/91
----------
train Loss: 0.0101 Acc: 0.7290
val Loss: 0.0217 Acc: 0.4534

Epoch 22/91
----------
train Loss: 0.0098 Acc: 0.7148
val Loss: 0.0216 Acc: 0.4596

Epoch 23/91
----------
train Loss: 0.0101 Acc: 0.7257
val Loss: 0.0218 Acc: 0.4720

Epoch 24/91
----------
train Loss: 0.0101 Acc: 0.7104
val Loss: 0.0216 Acc: 0.4658

Epoch 25/91
----------
train Loss: 0.0104 Acc: 0.7333
val Loss: 0.0209 Acc: 0.4596

Epoch 26/91
----------
train Loss: 0.0097 Acc: 0.7508
val Loss: 0.0216 Acc: 0.4596

Epoch 27/91
----------
train Loss: 0.0095 Acc: 0.7366
val Loss: 0.0225 Acc: 0.4658

Epoch 28/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0100 Acc: 0.7432
val Loss: 0.0219 Acc: 0.4596

Epoch 29/91
----------
train Loss: 0.0103 Acc: 0.7180
val Loss: 0.0217 Acc: 0.4534

Epoch 30/91
----------
train Loss: 0.0099 Acc: 0.7333
val Loss: 0.0223 Acc: 0.4472

Epoch 31/91
----------
train Loss: 0.0098 Acc: 0.7530
val Loss: 0.0225 Acc: 0.4596

Epoch 32/91
----------
train Loss: 0.0098 Acc: 0.7541
val Loss: 0.0220 Acc: 0.4596

Epoch 33/91
----------
train Loss: 0.0098 Acc: 0.7410
val Loss: 0.0217 Acc: 0.4472

Epoch 34/91
----------
train Loss: 0.0096 Acc: 0.7410
val Loss: 0.0212 Acc: 0.4658

Epoch 35/91
----------
train Loss: 0.0093 Acc: 0.7519
val Loss: 0.0214 Acc: 0.4658

Epoch 36/91
----------
train Loss: 0.0097 Acc: 0.7497
val Loss: 0.0219 Acc: 0.4658

Epoch 37/91
----------
train Loss: 0.0100 Acc: 0.7541
val Loss: 0.0215 Acc: 0.4720

Epoch 38/91
----------
train Loss: 0.0098 Acc: 0.7399
val Loss: 0.0214 Acc: 0.4534

Epoch 39/91
----------
train Loss: 0.0101 Acc: 0.7169
val Loss: 0.0215 Acc: 0.4720

Epoch 40/91
----------
train Loss: 0.0098 Acc: 0.7421
val Loss: 0.0214 Acc: 0.4783

Epoch 41/91
----------
train Loss: 0.0100 Acc: 0.7311
val Loss: 0.0218 Acc: 0.4720

Epoch 42/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0094 Acc: 0.7344
val Loss: 0.0213 Acc: 0.4720

Epoch 43/91
----------
train Loss: 0.0101 Acc: 0.7235
val Loss: 0.0211 Acc: 0.4658

Epoch 44/91
----------
train Loss: 0.0097 Acc: 0.7366
val Loss: 0.0215 Acc: 0.4720

Epoch 45/91
----------
train Loss: 0.0101 Acc: 0.7366
val Loss: 0.0216 Acc: 0.4596

Epoch 46/91
----------
train Loss: 0.0095 Acc: 0.7508
val Loss: 0.0213 Acc: 0.4783

Epoch 47/91
----------
train Loss: 0.0098 Acc: 0.7475
val Loss: 0.0218 Acc: 0.4658

Epoch 48/91
----------
train Loss: 0.0094 Acc: 0.7486
val Loss: 0.0218 Acc: 0.4658

Epoch 49/91
----------
train Loss: 0.0097 Acc: 0.7355
val Loss: 0.0223 Acc: 0.4720

Epoch 50/91
----------
train Loss: 0.0099 Acc: 0.7563
val Loss: 0.0218 Acc: 0.4596

Epoch 51/91
----------
train Loss: 0.0099 Acc: 0.7224
val Loss: 0.0220 Acc: 0.4596

Epoch 52/91
----------
train Loss: 0.0099 Acc: 0.7333
val Loss: 0.0213 Acc: 0.4720

Epoch 53/91
----------
train Loss: 0.0099 Acc: 0.7388
val Loss: 0.0219 Acc: 0.4783

Epoch 54/91
----------
train Loss: 0.0099 Acc: 0.7399
val Loss: 0.0217 Acc: 0.4596

Epoch 55/91
----------
train Loss: 0.0096 Acc: 0.7628
val Loss: 0.0224 Acc: 0.4783

Epoch 56/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0097 Acc: 0.7421
val Loss: 0.0220 Acc: 0.4596

Epoch 57/91
----------
train Loss: 0.0099 Acc: 0.7355
val Loss: 0.0213 Acc: 0.4720

Epoch 58/91
----------
train Loss: 0.0095 Acc: 0.7486
val Loss: 0.0218 Acc: 0.4658

Epoch 59/91
----------
train Loss: 0.0096 Acc: 0.7268
val Loss: 0.0216 Acc: 0.4596

Epoch 60/91
----------
train Loss: 0.0104 Acc: 0.7333
val Loss: 0.0219 Acc: 0.4783

Epoch 61/91
----------
train Loss: 0.0102 Acc: 0.7366
val Loss: 0.0224 Acc: 0.4658

Epoch 62/91
----------
train Loss: 0.0100 Acc: 0.7530
val Loss: 0.0217 Acc: 0.4596

Epoch 63/91
----------
train Loss: 0.0095 Acc: 0.7475
val Loss: 0.0218 Acc: 0.4720

Epoch 64/91
----------
train Loss: 0.0105 Acc: 0.7464
val Loss: 0.0225 Acc: 0.4658

Epoch 65/91
----------
train Loss: 0.0101 Acc: 0.7421
val Loss: 0.0227 Acc: 0.4658

Epoch 66/91
----------
train Loss: 0.0098 Acc: 0.7454
val Loss: 0.0218 Acc: 0.4720

Epoch 67/91
----------
train Loss: 0.0101 Acc: 0.7377
val Loss: 0.0209 Acc: 0.4783

Epoch 68/91
----------
train Loss: 0.0096 Acc: 0.7607
val Loss: 0.0219 Acc: 0.4658

Epoch 69/91
----------
train Loss: 0.0097 Acc: 0.7421
val Loss: 0.0218 Acc: 0.4596

Epoch 70/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0101 Acc: 0.7213
val Loss: 0.0221 Acc: 0.4658

Epoch 71/91
----------
train Loss: 0.0100 Acc: 0.7333
val Loss: 0.0212 Acc: 0.4720

Epoch 72/91
----------
train Loss: 0.0100 Acc: 0.7541
val Loss: 0.0222 Acc: 0.4596

Epoch 73/91
----------
train Loss: 0.0098 Acc: 0.7519
val Loss: 0.0218 Acc: 0.4658

Epoch 74/91
----------
train Loss: 0.0100 Acc: 0.7508
val Loss: 0.0226 Acc: 0.4658

Epoch 75/91
----------
train Loss: 0.0103 Acc: 0.7388
val Loss: 0.0217 Acc: 0.4534

Epoch 76/91
----------
train Loss: 0.0095 Acc: 0.7475
val Loss: 0.0220 Acc: 0.4658

Epoch 77/91
----------
train Loss: 0.0098 Acc: 0.7530
val Loss: 0.0220 Acc: 0.4658

Epoch 78/91
----------
train Loss: 0.0096 Acc: 0.7508
val Loss: 0.0213 Acc: 0.4658

Epoch 79/91
----------
train Loss: 0.0097 Acc: 0.7388
val Loss: 0.0215 Acc: 0.4720

Epoch 80/91
----------
train Loss: 0.0099 Acc: 0.7410
val Loss: 0.0211 Acc: 0.4658

Epoch 81/91
----------
train Loss: 0.0101 Acc: 0.7322
val Loss: 0.0216 Acc: 0.4658

Epoch 82/91
----------
train Loss: 0.0098 Acc: 0.7301
val Loss: 0.0216 Acc: 0.4658

Epoch 83/91
----------
train Loss: 0.0099 Acc: 0.7180
val Loss: 0.0215 Acc: 0.4596

Epoch 84/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0098 Acc: 0.7410
val Loss: 0.0215 Acc: 0.4534

Epoch 85/91
----------
train Loss: 0.0095 Acc: 0.7333
val Loss: 0.0221 Acc: 0.4534

Epoch 86/91
----------
train Loss: 0.0095 Acc: 0.7454
val Loss: 0.0214 Acc: 0.4596

Epoch 87/91
----------
train Loss: 0.0095 Acc: 0.7410
val Loss: 0.0212 Acc: 0.4596

Epoch 88/91
----------
train Loss: 0.0097 Acc: 0.7519
val Loss: 0.0219 Acc: 0.4658

Epoch 89/91
----------
train Loss: 0.0098 Acc: 0.7475
val Loss: 0.0210 Acc: 0.4658

Epoch 90/91
----------
train Loss: 0.0098 Acc: 0.7268
val Loss: 0.0213 Acc: 0.4658

Epoch 91/91
----------
train Loss: 0.0101 Acc: 0.7279
val Loss: 0.0218 Acc: 0.4720

Training complete in 8m 30s
Best val Acc: 0.478261

---Fine tuning.---
Epoch 0/91
----------
LR is set to 0.01
train Loss: 0.0099 Acc: 0.7148
val Loss: 0.0244 Acc: 0.3851

Epoch 1/91
----------
train Loss: 0.0079 Acc: 0.7672
val Loss: 0.0239 Acc: 0.4410

Epoch 2/91
----------
train Loss: 0.0063 Acc: 0.8328
val Loss: 0.0247 Acc: 0.4472

Epoch 3/91
----------
train Loss: 0.0054 Acc: 0.8514
val Loss: 0.0235 Acc: 0.4783

Epoch 4/91
----------
train Loss: 0.0043 Acc: 0.8776
val Loss: 0.0287 Acc: 0.4596

Epoch 5/91
----------
train Loss: 0.0030 Acc: 0.9224
val Loss: 0.0238 Acc: 0.4720

Epoch 6/91
----------
train Loss: 0.0025 Acc: 0.9355
val Loss: 0.0235 Acc: 0.5155

Epoch 7/91
----------
train Loss: 0.0023 Acc: 0.9399
val Loss: 0.0250 Acc: 0.5155

Epoch 8/91
----------
train Loss: 0.0021 Acc: 0.9486
val Loss: 0.0261 Acc: 0.4720

Epoch 9/91
----------
train Loss: 0.0023 Acc: 0.9486
val Loss: 0.0324 Acc: 0.3851

Epoch 10/91
----------
train Loss: 0.0018 Acc: 0.9530
val Loss: 0.0260 Acc: 0.4969

Epoch 11/91
----------
train Loss: 0.0016 Acc: 0.9563
val Loss: 0.0242 Acc: 0.5280

Epoch 12/91
----------
train Loss: 0.0011 Acc: 0.9661
val Loss: 0.0238 Acc: 0.5652

Epoch 13/91
----------
train Loss: 0.0013 Acc: 0.9661
val Loss: 0.0265 Acc: 0.5404

Epoch 14/91
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9716
val Loss: 0.0261 Acc: 0.5342

Epoch 15/91
----------
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0267 Acc: 0.5466

Epoch 16/91
----------
train Loss: 0.0009 Acc: 0.9760
val Loss: 0.0277 Acc: 0.5466

Epoch 17/91
----------
train Loss: 0.0010 Acc: 0.9803
val Loss: 0.0263 Acc: 0.5342

Epoch 18/91
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0297 Acc: 0.5404

Epoch 19/91
----------
train Loss: 0.0007 Acc: 0.9781
val Loss: 0.0266 Acc: 0.5466

Epoch 20/91
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0263 Acc: 0.5342

Epoch 21/91
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0270 Acc: 0.4969

Epoch 22/91
----------
train Loss: 0.0007 Acc: 0.9760
val Loss: 0.0273 Acc: 0.5155

Epoch 23/91
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0262 Acc: 0.5342

Epoch 24/91
----------
train Loss: 0.0006 Acc: 0.9760
val Loss: 0.0275 Acc: 0.5652

Epoch 25/91
----------
train Loss: 0.0006 Acc: 0.9869
val Loss: 0.0253 Acc: 0.5590

Epoch 26/91
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0270 Acc: 0.5590

Epoch 27/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0273 Acc: 0.5342

Epoch 28/91
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0283 Acc: 0.5590

Epoch 29/91
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0279 Acc: 0.5466

Epoch 30/91
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0293 Acc: 0.5528

Epoch 31/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0269 Acc: 0.5590

Epoch 32/91
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0263 Acc: 0.5528

Epoch 33/91
----------
train Loss: 0.0006 Acc: 0.9869
val Loss: 0.0276 Acc: 0.5466

Epoch 34/91
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0283 Acc: 0.5342

Epoch 35/91
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0263 Acc: 0.5466

Epoch 36/91
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0265 Acc: 0.5404

Epoch 37/91
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0266 Acc: 0.5466

Epoch 38/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0275 Acc: 0.5590

Epoch 39/91
----------
train Loss: 0.0006 Acc: 0.9858
val Loss: 0.0274 Acc: 0.5528

Epoch 40/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0276 Acc: 0.5590

Epoch 41/91
----------
train Loss: 0.0007 Acc: 0.9781
val Loss: 0.0275 Acc: 0.5528

Epoch 42/91
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0269 Acc: 0.5466

Epoch 43/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0275 Acc: 0.5528

Epoch 44/91
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0287 Acc: 0.5590

Epoch 45/91
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0285 Acc: 0.5528

Epoch 46/91
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0282 Acc: 0.5528

Epoch 47/91
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0269 Acc: 0.5528

Epoch 48/91
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0284 Acc: 0.5652

Epoch 49/91
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0274 Acc: 0.5652

Epoch 50/91
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0267 Acc: 0.5528

Epoch 51/91
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0277 Acc: 0.5466

Epoch 52/91
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0256 Acc: 0.5528

Epoch 53/91
----------
train Loss: 0.0007 Acc: 0.9749
val Loss: 0.0271 Acc: 0.5528

Epoch 54/91
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0268 Acc: 0.5590

Epoch 55/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0271 Acc: 0.5528

Epoch 56/91
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0288 Acc: 0.5466

Epoch 57/91
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0287 Acc: 0.5466

Epoch 58/91
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0273 Acc: 0.5590

Epoch 59/91
----------
train Loss: 0.0006 Acc: 0.9858
val Loss: 0.0273 Acc: 0.5590

Epoch 60/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0271 Acc: 0.5590

Epoch 61/91
----------
train Loss: 0.0009 Acc: 0.9836
val Loss: 0.0279 Acc: 0.5466

Epoch 62/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0272 Acc: 0.5466

Epoch 63/91
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0270 Acc: 0.5590

Epoch 64/91
----------
train Loss: 0.0005 Acc: 0.9891
val Loss: 0.0264 Acc: 0.5590

Epoch 65/91
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0264 Acc: 0.5590

Epoch 66/91
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0272 Acc: 0.5590

Epoch 67/91
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0271 Acc: 0.5466

Epoch 68/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0272 Acc: 0.5590

Epoch 69/91
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0266 Acc: 0.5590

Epoch 70/91
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9891
val Loss: 0.0274 Acc: 0.5590

Epoch 71/91
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0260 Acc: 0.5528

Epoch 72/91
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0266 Acc: 0.5590

Epoch 73/91
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0275 Acc: 0.5528

Epoch 74/91
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0283 Acc: 0.5466

Epoch 75/91
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0280 Acc: 0.5404

Epoch 76/91
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0276 Acc: 0.5466

Epoch 77/91
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0282 Acc: 0.5652

Epoch 78/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0278 Acc: 0.5466

Epoch 79/91
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0270 Acc: 0.5590

Epoch 80/91
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0288 Acc: 0.5590

Epoch 81/91
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0270 Acc: 0.5528

Epoch 82/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0268 Acc: 0.5466

Epoch 83/91
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0270 Acc: 0.5528

Epoch 84/91
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0275 Acc: 0.5652

Epoch 85/91
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0286 Acc: 0.5590

Epoch 86/91
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0270 Acc: 0.5528

Epoch 87/91
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0279 Acc: 0.5404

Epoch 88/91
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0269 Acc: 0.5528

Epoch 89/91
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0273 Acc: 0.5590

Epoch 90/91
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0275 Acc: 0.5528

Epoch 91/91
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0286 Acc: 0.5528

Training complete in 8m 60s
Best val Acc: 0.565217

---Testing---
Test accuracy: 0.914498
--------------------
Accuracy of Albacore tuna : 91 %
Accuracy of Atlantic bluefin tuna : 82 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 86 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 92 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Skipjack tuna : 97 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 85 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.892375855762033, std: 0.06569770827987455
--------------------

run info[val: 0.2, epoch: 66, randcrop: False, decay: 7]

---Training last layer.---
Epoch 0/65
----------
LR is set to 0.01
train Loss: 0.0275 Acc: 0.1661
val Loss: 0.0314 Acc: 0.3163

Epoch 1/65
----------
train Loss: 0.0218 Acc: 0.3508
val Loss: 0.0298 Acc: 0.3953

Epoch 2/65
----------
train Loss: 0.0184 Acc: 0.4530
val Loss: 0.0274 Acc: 0.3814

Epoch 3/65
----------
train Loss: 0.0162 Acc: 0.5041
val Loss: 0.0249 Acc: 0.4233

Epoch 4/65
----------
train Loss: 0.0144 Acc: 0.5772
val Loss: 0.0241 Acc: 0.4465

Epoch 5/65
----------
train Loss: 0.0135 Acc: 0.5958
val Loss: 0.0236 Acc: 0.4791

Epoch 6/65
----------
train Loss: 0.0128 Acc: 0.6388
val Loss: 0.0260 Acc: 0.4465

Epoch 7/65
----------
LR is set to 0.001
train Loss: 0.0118 Acc: 0.6829
val Loss: 0.0231 Acc: 0.5070

Epoch 8/65
----------
train Loss: 0.0113 Acc: 0.6876
val Loss: 0.0213 Acc: 0.4884

Epoch 9/65
----------
train Loss: 0.0112 Acc: 0.6911
val Loss: 0.0245 Acc: 0.4791

Epoch 10/65
----------
train Loss: 0.0111 Acc: 0.7027
val Loss: 0.0227 Acc: 0.4744

Epoch 11/65
----------
train Loss: 0.0109 Acc: 0.6992
val Loss: 0.0254 Acc: 0.5023

Epoch 12/65
----------
train Loss: 0.0108 Acc: 0.7236
val Loss: 0.0217 Acc: 0.4791

Epoch 13/65
----------
train Loss: 0.0108 Acc: 0.7213
val Loss: 0.0246 Acc: 0.4977

Epoch 14/65
----------
LR is set to 0.00010000000000000002
train Loss: 0.0108 Acc: 0.6969
val Loss: 0.0228 Acc: 0.4977

Epoch 15/65
----------
train Loss: 0.0108 Acc: 0.7050
val Loss: 0.0233 Acc: 0.4837

Epoch 16/65
----------
train Loss: 0.0108 Acc: 0.7027
val Loss: 0.0231 Acc: 0.4884

Epoch 17/65
----------
train Loss: 0.0106 Acc: 0.7178
val Loss: 0.0251 Acc: 0.4791

Epoch 18/65
----------
train Loss: 0.0108 Acc: 0.6992
val Loss: 0.0223 Acc: 0.4837

Epoch 19/65
----------
train Loss: 0.0106 Acc: 0.7224
val Loss: 0.0251 Acc: 0.4837

Epoch 20/65
----------
train Loss: 0.0109 Acc: 0.7015
val Loss: 0.0243 Acc: 0.4930

Epoch 21/65
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0108 Acc: 0.7131
val Loss: 0.0239 Acc: 0.4884

Epoch 22/65
----------
train Loss: 0.0107 Acc: 0.7189
val Loss: 0.0230 Acc: 0.4791

Epoch 23/65
----------
train Loss: 0.0107 Acc: 0.7213
val Loss: 0.0218 Acc: 0.4884

Epoch 24/65
----------
train Loss: 0.0107 Acc: 0.7247
val Loss: 0.0220 Acc: 0.4837

Epoch 25/65
----------
train Loss: 0.0108 Acc: 0.7224
val Loss: 0.0211 Acc: 0.4930

Epoch 26/65
----------
train Loss: 0.0107 Acc: 0.7189
val Loss: 0.0229 Acc: 0.4837

Epoch 27/65
----------
train Loss: 0.0106 Acc: 0.7294
val Loss: 0.0243 Acc: 0.4837

Epoch 28/65
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0108 Acc: 0.7154
val Loss: 0.0230 Acc: 0.4837

Epoch 29/65
----------
train Loss: 0.0107 Acc: 0.7213
val Loss: 0.0229 Acc: 0.4837

Epoch 30/65
----------
train Loss: 0.0106 Acc: 0.7178
val Loss: 0.0228 Acc: 0.4791

Epoch 31/65
----------
train Loss: 0.0107 Acc: 0.7027
val Loss: 0.0228 Acc: 0.4791

Epoch 32/65
----------
train Loss: 0.0108 Acc: 0.7178
val Loss: 0.0244 Acc: 0.4791

Epoch 33/65
----------
train Loss: 0.0106 Acc: 0.7154
val Loss: 0.0237 Acc: 0.4837

Epoch 34/65
----------
train Loss: 0.0107 Acc: 0.7178
val Loss: 0.0240 Acc: 0.4837

Epoch 35/65
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0108 Acc: 0.7096
val Loss: 0.0243 Acc: 0.4837

Epoch 36/65
----------
train Loss: 0.0107 Acc: 0.7096
val Loss: 0.0232 Acc: 0.4791

Epoch 37/65
----------
train Loss: 0.0109 Acc: 0.7166
val Loss: 0.0242 Acc: 0.4837

Epoch 38/65
----------
train Loss: 0.0107 Acc: 0.7247
val Loss: 0.0215 Acc: 0.4791

Epoch 39/65
----------
train Loss: 0.0106 Acc: 0.7189
val Loss: 0.0229 Acc: 0.4837

Epoch 40/65
----------
train Loss: 0.0108 Acc: 0.7224
val Loss: 0.0234 Acc: 0.4884

Epoch 41/65
----------
train Loss: 0.0106 Acc: 0.7143
val Loss: 0.0220 Acc: 0.4837

Epoch 42/65
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0107 Acc: 0.7166
val Loss: 0.0251 Acc: 0.4884

Epoch 43/65
----------
train Loss: 0.0107 Acc: 0.7120
val Loss: 0.0268 Acc: 0.4791

Epoch 44/65
----------
train Loss: 0.0109 Acc: 0.6992
val Loss: 0.0264 Acc: 0.4837

Epoch 45/65
----------
train Loss: 0.0108 Acc: 0.7073
val Loss: 0.0261 Acc: 0.4791

Epoch 46/65
----------
train Loss: 0.0107 Acc: 0.7050
val Loss: 0.0231 Acc: 0.4791

Epoch 47/65
----------
train Loss: 0.0108 Acc: 0.6992
val Loss: 0.0235 Acc: 0.4791

Epoch 48/65
----------
train Loss: 0.0106 Acc: 0.7178
val Loss: 0.0226 Acc: 0.4837

Epoch 49/65
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0106 Acc: 0.7131
val Loss: 0.0238 Acc: 0.4837

Epoch 50/65
----------
train Loss: 0.0107 Acc: 0.7271
val Loss: 0.0246 Acc: 0.4884

Epoch 51/65
----------
train Loss: 0.0109 Acc: 0.7062
val Loss: 0.0204 Acc: 0.4791

Epoch 52/65
----------
train Loss: 0.0107 Acc: 0.7015
val Loss: 0.0234 Acc: 0.4791

Epoch 53/65
----------
train Loss: 0.0108 Acc: 0.7096
val Loss: 0.0247 Acc: 0.4837

Epoch 54/65
----------
train Loss: 0.0109 Acc: 0.7038
val Loss: 0.0239 Acc: 0.4837

Epoch 55/65
----------
train Loss: 0.0107 Acc: 0.7096
val Loss: 0.0278 Acc: 0.4791

Epoch 56/65
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0108 Acc: 0.6934
val Loss: 0.0228 Acc: 0.4837

Epoch 57/65
----------
train Loss: 0.0108 Acc: 0.7073
val Loss: 0.0233 Acc: 0.4837

Epoch 58/65
----------
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0219 Acc: 0.4884

Epoch 59/65
----------
train Loss: 0.0108 Acc: 0.7027
val Loss: 0.0234 Acc: 0.4791

Epoch 60/65
----------
train Loss: 0.0105 Acc: 0.7213
val Loss: 0.0227 Acc: 0.4791

Epoch 61/65
----------
train Loss: 0.0107 Acc: 0.7038
val Loss: 0.0258 Acc: 0.4837

Epoch 62/65
----------
train Loss: 0.0108 Acc: 0.7108
val Loss: 0.0232 Acc: 0.4837

Epoch 63/65
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0108 Acc: 0.7154
val Loss: 0.0223 Acc: 0.4884

Epoch 64/65
----------
train Loss: 0.0107 Acc: 0.7166
val Loss: 0.0242 Acc: 0.4837

Epoch 65/65
----------
train Loss: 0.0108 Acc: 0.6969
val Loss: 0.0233 Acc: 0.4884

Training complete in 5m 52s
Best val Acc: 0.506977

---Fine tuning.---
Epoch 0/65
----------
LR is set to 0.01
train Loss: 0.0112 Acc: 0.6911
val Loss: 0.0232 Acc: 0.4930

Epoch 1/65
----------
train Loss: 0.0071 Acc: 0.8188
val Loss: 0.0247 Acc: 0.4465

Epoch 2/65
----------
train Loss: 0.0042 Acc: 0.9013
val Loss: 0.0225 Acc: 0.5209

Epoch 3/65
----------
train Loss: 0.0024 Acc: 0.9443
val Loss: 0.0209 Acc: 0.5907

Epoch 4/65
----------
train Loss: 0.0016 Acc: 0.9663
val Loss: 0.0227 Acc: 0.5953

Epoch 5/65
----------
train Loss: 0.0011 Acc: 0.9768
val Loss: 0.0232 Acc: 0.5488

Epoch 6/65
----------
train Loss: 0.0010 Acc: 0.9791
val Loss: 0.0206 Acc: 0.5907

Epoch 7/65
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0249 Acc: 0.5907

Epoch 8/65
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0227 Acc: 0.5628

Epoch 9/65
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0210 Acc: 0.5628

Epoch 10/65
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0269 Acc: 0.5581

Epoch 11/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0232 Acc: 0.5628

Epoch 12/65
----------
train Loss: 0.0006 Acc: 0.9779
val Loss: 0.0208 Acc: 0.5721

Epoch 13/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0217 Acc: 0.5767

Epoch 14/65
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0218 Acc: 0.5767

Epoch 15/65
----------
train Loss: 0.0004 Acc: 0.9919
val Loss: 0.0249 Acc: 0.5721

Epoch 16/65
----------
train Loss: 0.0004 Acc: 0.9849
val Loss: 0.0200 Acc: 0.5721

Epoch 17/65
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0214 Acc: 0.5721

Epoch 18/65
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0212 Acc: 0.5767

Epoch 19/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0237 Acc: 0.5767

Epoch 20/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0183 Acc: 0.5721

Epoch 21/65
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0205 Acc: 0.5767

Epoch 22/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0238 Acc: 0.5721

Epoch 23/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0208 Acc: 0.5674

Epoch 24/65
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0204 Acc: 0.5767

Epoch 25/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0218 Acc: 0.5767

Epoch 26/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0200 Acc: 0.5721

Epoch 27/65
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0222 Acc: 0.5674

Epoch 28/65
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0203 Acc: 0.5721

Epoch 29/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0201 Acc: 0.5814

Epoch 30/65
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0208 Acc: 0.5814

Epoch 31/65
----------
train Loss: 0.0005 Acc: 0.9779
val Loss: 0.0266 Acc: 0.5767

Epoch 32/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0209 Acc: 0.5767

Epoch 33/65
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0173 Acc: 0.5674

Epoch 34/65
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0209 Acc: 0.5628

Epoch 35/65
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0240 Acc: 0.5674

Epoch 36/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0205 Acc: 0.5767

Epoch 37/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0261 Acc: 0.5721

Epoch 38/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0227 Acc: 0.5767

Epoch 39/65
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0213 Acc: 0.5721

Epoch 40/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0218 Acc: 0.5721

Epoch 41/65
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0201 Acc: 0.5767

Epoch 42/65
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0231 Acc: 0.5721

Epoch 43/65
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0203 Acc: 0.5767

Epoch 44/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0229 Acc: 0.5767

Epoch 45/65
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0227 Acc: 0.5814

Epoch 46/65
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0204 Acc: 0.5767

Epoch 47/65
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0212 Acc: 0.5767

Epoch 48/65
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0231 Acc: 0.5721

Epoch 49/65
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0215 Acc: 0.5674

Epoch 50/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0223 Acc: 0.5767

Epoch 51/65
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5674

Epoch 52/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0214 Acc: 0.5814

Epoch 53/65
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0218 Acc: 0.5721

Epoch 54/65
----------
train Loss: 0.0004 Acc: 0.9895
val Loss: 0.0220 Acc: 0.5767

Epoch 55/65
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0238 Acc: 0.5767

Epoch 56/65
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0205 Acc: 0.5767

Epoch 57/65
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0254 Acc: 0.5721

Epoch 58/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0207 Acc: 0.5721

Epoch 59/65
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0236 Acc: 0.5767

Epoch 60/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0257 Acc: 0.5721

Epoch 61/65
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0195 Acc: 0.5674

Epoch 62/65
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0227 Acc: 0.5721

Epoch 63/65
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0217 Acc: 0.5721

Epoch 64/65
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0195 Acc: 0.5721

Epoch 65/65
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0235 Acc: 0.5767

Training complete in 6m 13s
Best val Acc: 0.595349

---Testing---
Test accuracy: 0.902416
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 96 %
Accuracy of Mackerel tuna : 88 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8701341220509213, std: 0.08819326429188332
--------------------

run info[val: 0.25, epoch: 83, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0289 Acc: 0.1537
val Loss: 0.0282 Acc: 0.2082

Epoch 1/82
----------
train Loss: 0.0244 Acc: 0.2714
val Loss: 0.0248 Acc: 0.3346

Epoch 2/82
----------
train Loss: 0.0215 Acc: 0.3953
val Loss: 0.0226 Acc: 0.3866

Epoch 3/82
----------
train Loss: 0.0189 Acc: 0.4486
val Loss: 0.0238 Acc: 0.3606

Epoch 4/82
----------
train Loss: 0.0182 Acc: 0.4585
val Loss: 0.0225 Acc: 0.3680

Epoch 5/82
----------
train Loss: 0.0194 Acc: 0.4325
val Loss: 0.0213 Acc: 0.4312

Epoch 6/82
----------
train Loss: 0.0166 Acc: 0.5229
val Loss: 0.0200 Acc: 0.4275

Epoch 7/82
----------
train Loss: 0.0145 Acc: 0.5874
val Loss: 0.0207 Acc: 0.4572

Epoch 8/82
----------
train Loss: 0.0147 Acc: 0.6171
val Loss: 0.0202 Acc: 0.4684

Epoch 9/82
----------
train Loss: 0.0137 Acc: 0.6208
val Loss: 0.0213 Acc: 0.4126

Epoch 10/82
----------
train Loss: 0.0142 Acc: 0.6159
val Loss: 0.0203 Acc: 0.4610

Epoch 11/82
----------
LR is set to 0.001
train Loss: 0.0118 Acc: 0.6815
val Loss: 0.0200 Acc: 0.4647

Epoch 12/82
----------
train Loss: 0.0117 Acc: 0.6679
val Loss: 0.0194 Acc: 0.4498

Epoch 13/82
----------
train Loss: 0.0120 Acc: 0.6753
val Loss: 0.0189 Acc: 0.4647

Epoch 14/82
----------
train Loss: 0.0115 Acc: 0.7001
val Loss: 0.0193 Acc: 0.4721

Epoch 15/82
----------
train Loss: 0.0116 Acc: 0.6989
val Loss: 0.0191 Acc: 0.4796

Epoch 16/82
----------
train Loss: 0.0118 Acc: 0.6939
val Loss: 0.0193 Acc: 0.4684

Epoch 17/82
----------
train Loss: 0.0112 Acc: 0.6865
val Loss: 0.0194 Acc: 0.4758

Epoch 18/82
----------
train Loss: 0.0118 Acc: 0.6753
val Loss: 0.0192 Acc: 0.4796

Epoch 19/82
----------
train Loss: 0.0116 Acc: 0.6914
val Loss: 0.0190 Acc: 0.4721

Epoch 20/82
----------
train Loss: 0.0118 Acc: 0.7138
val Loss: 0.0191 Acc: 0.4833

Epoch 21/82
----------
train Loss: 0.0116 Acc: 0.6865
val Loss: 0.0192 Acc: 0.4684

Epoch 22/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0112 Acc: 0.7150
val Loss: 0.0193 Acc: 0.4796

Epoch 23/82
----------
train Loss: 0.0110 Acc: 0.7038
val Loss: 0.0192 Acc: 0.4721

Epoch 24/82
----------
train Loss: 0.0108 Acc: 0.7249
val Loss: 0.0190 Acc: 0.4796

Epoch 25/82
----------
train Loss: 0.0113 Acc: 0.6902
val Loss: 0.0191 Acc: 0.4721

Epoch 26/82
----------
train Loss: 0.0108 Acc: 0.7299
val Loss: 0.0188 Acc: 0.4721

Epoch 27/82
----------
train Loss: 0.0113 Acc: 0.7063
val Loss: 0.0190 Acc: 0.4721

Epoch 28/82
----------
train Loss: 0.0109 Acc: 0.7224
val Loss: 0.0190 Acc: 0.4684

Epoch 29/82
----------
train Loss: 0.0121 Acc: 0.6877
val Loss: 0.0191 Acc: 0.4572

Epoch 30/82
----------
train Loss: 0.0110 Acc: 0.6989
val Loss: 0.0192 Acc: 0.4796

Epoch 31/82
----------
train Loss: 0.0115 Acc: 0.7162
val Loss: 0.0186 Acc: 0.4833

Epoch 32/82
----------
train Loss: 0.0111 Acc: 0.7113
val Loss: 0.0190 Acc: 0.4647

Epoch 33/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0103 Acc: 0.7212
val Loss: 0.0191 Acc: 0.4684

Epoch 34/82
----------
train Loss: 0.0117 Acc: 0.6952
val Loss: 0.0192 Acc: 0.4721

Epoch 35/82
----------
train Loss: 0.0109 Acc: 0.7125
val Loss: 0.0189 Acc: 0.4796

Epoch 36/82
----------
train Loss: 0.0111 Acc: 0.7076
val Loss: 0.0191 Acc: 0.4758

Epoch 37/82
----------
train Loss: 0.0115 Acc: 0.6939
val Loss: 0.0189 Acc: 0.4796

Epoch 38/82
----------
train Loss: 0.0112 Acc: 0.7038
val Loss: 0.0190 Acc: 0.4684

Epoch 39/82
----------
train Loss: 0.0122 Acc: 0.6964
val Loss: 0.0190 Acc: 0.4647

Epoch 40/82
----------
train Loss: 0.0111 Acc: 0.7051
val Loss: 0.0191 Acc: 0.4684

Epoch 41/82
----------
train Loss: 0.0115 Acc: 0.7038
val Loss: 0.0189 Acc: 0.4572

Epoch 42/82
----------
train Loss: 0.0128 Acc: 0.6914
val Loss: 0.0189 Acc: 0.4684

Epoch 43/82
----------
train Loss: 0.0110 Acc: 0.7150
val Loss: 0.0191 Acc: 0.4572

Epoch 44/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0116 Acc: 0.7113
val Loss: 0.0187 Acc: 0.4647

Epoch 45/82
----------
train Loss: 0.0112 Acc: 0.7051
val Loss: 0.0188 Acc: 0.4758

Epoch 46/82
----------
train Loss: 0.0112 Acc: 0.7001
val Loss: 0.0189 Acc: 0.4758

Epoch 47/82
----------
train Loss: 0.0110 Acc: 0.6976
val Loss: 0.0187 Acc: 0.4721

Epoch 48/82
----------
train Loss: 0.0116 Acc: 0.7076
val Loss: 0.0191 Acc: 0.4647

Epoch 49/82
----------
train Loss: 0.0106 Acc: 0.7224
val Loss: 0.0191 Acc: 0.4758

Epoch 50/82
----------
train Loss: 0.0121 Acc: 0.7200
val Loss: 0.0190 Acc: 0.4721

Epoch 51/82
----------
train Loss: 0.0110 Acc: 0.6976
val Loss: 0.0187 Acc: 0.4647

Epoch 52/82
----------
train Loss: 0.0112 Acc: 0.6964
val Loss: 0.0190 Acc: 0.4647

Epoch 53/82
----------
train Loss: 0.0122 Acc: 0.7113
val Loss: 0.0187 Acc: 0.4684

Epoch 54/82
----------
train Loss: 0.0107 Acc: 0.7138
val Loss: 0.0188 Acc: 0.4796

Epoch 55/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0118 Acc: 0.6989
val Loss: 0.0186 Acc: 0.4758

Epoch 56/82
----------
train Loss: 0.0110 Acc: 0.6964
val Loss: 0.0190 Acc: 0.4758

Epoch 57/82
----------
train Loss: 0.0107 Acc: 0.7113
val Loss: 0.0190 Acc: 0.4870

Epoch 58/82
----------
train Loss: 0.0116 Acc: 0.7063
val Loss: 0.0188 Acc: 0.4721

Epoch 59/82
----------
train Loss: 0.0112 Acc: 0.6877
val Loss: 0.0189 Acc: 0.4610

Epoch 60/82
----------
train Loss: 0.0112 Acc: 0.7113
val Loss: 0.0190 Acc: 0.4758

Epoch 61/82
----------
train Loss: 0.0115 Acc: 0.7026
val Loss: 0.0192 Acc: 0.4684

Epoch 62/82
----------
train Loss: 0.0114 Acc: 0.6914
val Loss: 0.0190 Acc: 0.4907

Epoch 63/82
----------
train Loss: 0.0111 Acc: 0.6989
val Loss: 0.0187 Acc: 0.4684

Epoch 64/82
----------
train Loss: 0.0113 Acc: 0.7100
val Loss: 0.0191 Acc: 0.4758

Epoch 65/82
----------
train Loss: 0.0109 Acc: 0.7113
val Loss: 0.0189 Acc: 0.4721

Epoch 66/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0112 Acc: 0.7076
val Loss: 0.0188 Acc: 0.4684

Epoch 67/82
----------
train Loss: 0.0113 Acc: 0.7088
val Loss: 0.0189 Acc: 0.4684

Epoch 68/82
----------
train Loss: 0.0113 Acc: 0.7113
val Loss: 0.0191 Acc: 0.4758

Epoch 69/82
----------
train Loss: 0.0106 Acc: 0.7237
val Loss: 0.0187 Acc: 0.4684

Epoch 70/82
----------
train Loss: 0.0109 Acc: 0.7038
val Loss: 0.0186 Acc: 0.4647

Epoch 71/82
----------
train Loss: 0.0111 Acc: 0.7125
val Loss: 0.0191 Acc: 0.4721

Epoch 72/82
----------
train Loss: 0.0107 Acc: 0.7150
val Loss: 0.0189 Acc: 0.4610

Epoch 73/82
----------
train Loss: 0.0116 Acc: 0.7001
val Loss: 0.0190 Acc: 0.4684

Epoch 74/82
----------
train Loss: 0.0110 Acc: 0.7001
val Loss: 0.0191 Acc: 0.4684

Epoch 75/82
----------
train Loss: 0.0112 Acc: 0.7100
val Loss: 0.0187 Acc: 0.4721

Epoch 76/82
----------
train Loss: 0.0116 Acc: 0.7150
val Loss: 0.0187 Acc: 0.4796

Epoch 77/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0113 Acc: 0.7026
val Loss: 0.0187 Acc: 0.4758

Epoch 78/82
----------
train Loss: 0.0111 Acc: 0.7076
val Loss: 0.0188 Acc: 0.4647

Epoch 79/82
----------
train Loss: 0.0123 Acc: 0.7125
val Loss: 0.0191 Acc: 0.4610

Epoch 80/82
----------
train Loss: 0.0115 Acc: 0.6976
val Loss: 0.0188 Acc: 0.4684

Epoch 81/82
----------
train Loss: 0.0113 Acc: 0.7001
val Loss: 0.0191 Acc: 0.4684

Epoch 82/82
----------
train Loss: 0.0113 Acc: 0.7113
val Loss: 0.0190 Acc: 0.4684

Training complete in 7m 34s
Best val Acc: 0.490706

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6828
val Loss: 0.0207 Acc: 0.4498

Epoch 1/82
----------
train Loss: 0.0128 Acc: 0.6543
val Loss: 0.0323 Acc: 0.3903

Epoch 2/82
----------
train Loss: 0.0128 Acc: 0.6766
val Loss: 0.0283 Acc: 0.3829

Epoch 3/82
----------
train Loss: 0.0110 Acc: 0.7088
val Loss: 0.0271 Acc: 0.3866

Epoch 4/82
----------
train Loss: 0.0109 Acc: 0.6914
val Loss: 0.0457 Acc: 0.2974

Epoch 5/82
----------
train Loss: 0.0104 Acc: 0.7224
val Loss: 0.0307 Acc: 0.4126

Epoch 6/82
----------
train Loss: 0.0094 Acc: 0.7224
val Loss: 0.0324 Acc: 0.4238

Epoch 7/82
----------
train Loss: 0.0097 Acc: 0.7423
val Loss: 0.0443 Acc: 0.3606

Epoch 8/82
----------
train Loss: 0.0114 Acc: 0.7138
val Loss: 0.0347 Acc: 0.3197

Epoch 9/82
----------
train Loss: 0.0128 Acc: 0.6890
val Loss: 0.0296 Acc: 0.4498

Epoch 10/82
----------
train Loss: 0.0135 Acc: 0.6667
val Loss: 0.0374 Acc: 0.3086

Epoch 11/82
----------
LR is set to 0.001
train Loss: 0.0098 Acc: 0.6989
val Loss: 0.0269 Acc: 0.4164

Epoch 12/82
----------
train Loss: 0.0071 Acc: 0.8030
val Loss: 0.0227 Acc: 0.5019

Epoch 13/82
----------
train Loss: 0.0051 Acc: 0.8525
val Loss: 0.0208 Acc: 0.5242

Epoch 14/82
----------
train Loss: 0.0041 Acc: 0.8934
val Loss: 0.0203 Acc: 0.5428

Epoch 15/82
----------
train Loss: 0.0041 Acc: 0.9033
val Loss: 0.0208 Acc: 0.5353

Epoch 16/82
----------
train Loss: 0.0035 Acc: 0.9182
val Loss: 0.0200 Acc: 0.5316

Epoch 17/82
----------
train Loss: 0.0035 Acc: 0.9257
val Loss: 0.0210 Acc: 0.5316

Epoch 18/82
----------
train Loss: 0.0032 Acc: 0.9343
val Loss: 0.0205 Acc: 0.5093

Epoch 19/82
----------
train Loss: 0.0027 Acc: 0.9368
val Loss: 0.0205 Acc: 0.5130

Epoch 20/82
----------
train Loss: 0.0027 Acc: 0.9480
val Loss: 0.0205 Acc: 0.5130

Epoch 21/82
----------
train Loss: 0.0029 Acc: 0.9430
val Loss: 0.0214 Acc: 0.4870

Epoch 22/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0031 Acc: 0.9467
val Loss: 0.0217 Acc: 0.4981

Epoch 23/82
----------
train Loss: 0.0027 Acc: 0.9480
val Loss: 0.0214 Acc: 0.4981

Epoch 24/82
----------
train Loss: 0.0025 Acc: 0.9467
val Loss: 0.0211 Acc: 0.5167

Epoch 25/82
----------
train Loss: 0.0022 Acc: 0.9492
val Loss: 0.0209 Acc: 0.5204

Epoch 26/82
----------
train Loss: 0.0023 Acc: 0.9504
val Loss: 0.0217 Acc: 0.4981

Epoch 27/82
----------
train Loss: 0.0021 Acc: 0.9455
val Loss: 0.0208 Acc: 0.4981

Epoch 28/82
----------
train Loss: 0.0025 Acc: 0.9529
val Loss: 0.0212 Acc: 0.5204

Epoch 29/82
----------
train Loss: 0.0025 Acc: 0.9455
val Loss: 0.0207 Acc: 0.5056

Epoch 30/82
----------
train Loss: 0.0028 Acc: 0.9492
val Loss: 0.0215 Acc: 0.5242

Epoch 31/82
----------
train Loss: 0.0024 Acc: 0.9517
val Loss: 0.0211 Acc: 0.5204

Epoch 32/82
----------
train Loss: 0.0027 Acc: 0.9665
val Loss: 0.0209 Acc: 0.5279

Epoch 33/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0023 Acc: 0.9492
val Loss: 0.0213 Acc: 0.5204

Epoch 34/82
----------
train Loss: 0.0022 Acc: 0.9504
val Loss: 0.0206 Acc: 0.5167

Epoch 35/82
----------
train Loss: 0.0035 Acc: 0.9542
val Loss: 0.0212 Acc: 0.5130

Epoch 36/82
----------
train Loss: 0.0024 Acc: 0.9492
val Loss: 0.0217 Acc: 0.5130

Epoch 37/82
----------
train Loss: 0.0029 Acc: 0.9467
val Loss: 0.0217 Acc: 0.5056

Epoch 38/82
----------
train Loss: 0.0024 Acc: 0.9529
val Loss: 0.0206 Acc: 0.5242

Epoch 39/82
----------
train Loss: 0.0026 Acc: 0.9480
val Loss: 0.0213 Acc: 0.5204

Epoch 40/82
----------
train Loss: 0.0023 Acc: 0.9579
val Loss: 0.0206 Acc: 0.5390

Epoch 41/82
----------
train Loss: 0.0020 Acc: 0.9579
val Loss: 0.0208 Acc: 0.5316

Epoch 42/82
----------
train Loss: 0.0023 Acc: 0.9554
val Loss: 0.0215 Acc: 0.5316

Epoch 43/82
----------
train Loss: 0.0024 Acc: 0.9442
val Loss: 0.0206 Acc: 0.5279

Epoch 44/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0025 Acc: 0.9566
val Loss: 0.0206 Acc: 0.5167

Epoch 45/82
----------
train Loss: 0.0031 Acc: 0.9504
val Loss: 0.0202 Acc: 0.5316

Epoch 46/82
----------
train Loss: 0.0032 Acc: 0.9418
val Loss: 0.0206 Acc: 0.5316

Epoch 47/82
----------
train Loss: 0.0027 Acc: 0.9504
val Loss: 0.0212 Acc: 0.5242

Epoch 48/82
----------
train Loss: 0.0028 Acc: 0.9579
val Loss: 0.0211 Acc: 0.5242

Epoch 49/82
----------
train Loss: 0.0030 Acc: 0.9467
val Loss: 0.0209 Acc: 0.5279

Epoch 50/82
----------
train Loss: 0.0024 Acc: 0.9492
val Loss: 0.0208 Acc: 0.5093

Epoch 51/82
----------
train Loss: 0.0021 Acc: 0.9492
val Loss: 0.0206 Acc: 0.5167

Epoch 52/82
----------
train Loss: 0.0020 Acc: 0.9566
val Loss: 0.0213 Acc: 0.5279

Epoch 53/82
----------
train Loss: 0.0019 Acc: 0.9628
val Loss: 0.0209 Acc: 0.5242

Epoch 54/82
----------
train Loss: 0.0024 Acc: 0.9480
val Loss: 0.0208 Acc: 0.5167

Epoch 55/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9504
val Loss: 0.0210 Acc: 0.5242

Epoch 56/82
----------
train Loss: 0.0030 Acc: 0.9455
val Loss: 0.0209 Acc: 0.5167

Epoch 57/82
----------
train Loss: 0.0021 Acc: 0.9579
val Loss: 0.0212 Acc: 0.5204

Epoch 58/82
----------
train Loss: 0.0032 Acc: 0.9591
val Loss: 0.0212 Acc: 0.5353

Epoch 59/82
----------
train Loss: 0.0022 Acc: 0.9641
val Loss: 0.0212 Acc: 0.5242

Epoch 60/82
----------
train Loss: 0.0026 Acc: 0.9603
val Loss: 0.0215 Acc: 0.5242

Epoch 61/82
----------
train Loss: 0.0025 Acc: 0.9628
val Loss: 0.0209 Acc: 0.5279

Epoch 62/82
----------
train Loss: 0.0032 Acc: 0.9529
val Loss: 0.0215 Acc: 0.5204

Epoch 63/82
----------
train Loss: 0.0025 Acc: 0.9591
val Loss: 0.0213 Acc: 0.5167

Epoch 64/82
----------
train Loss: 0.0021 Acc: 0.9480
val Loss: 0.0210 Acc: 0.5204

Epoch 65/82
----------
train Loss: 0.0027 Acc: 0.9554
val Loss: 0.0208 Acc: 0.5204

Epoch 66/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0029 Acc: 0.9566
val Loss: 0.0208 Acc: 0.5316

Epoch 67/82
----------
train Loss: 0.0019 Acc: 0.9616
val Loss: 0.0210 Acc: 0.5279

Epoch 68/82
----------
train Loss: 0.0025 Acc: 0.9579
val Loss: 0.0214 Acc: 0.5279

Epoch 69/82
----------
train Loss: 0.0019 Acc: 0.9665
val Loss: 0.0210 Acc: 0.5279

Epoch 70/82
----------
train Loss: 0.0027 Acc: 0.9517
val Loss: 0.0211 Acc: 0.5204

Epoch 71/82
----------
train Loss: 0.0028 Acc: 0.9566
val Loss: 0.0211 Acc: 0.5242

Epoch 72/82
----------
train Loss: 0.0019 Acc: 0.9603
val Loss: 0.0210 Acc: 0.5316

Epoch 73/82
----------
train Loss: 0.0023 Acc: 0.9542
val Loss: 0.0206 Acc: 0.5316

Epoch 74/82
----------
train Loss: 0.0031 Acc: 0.9554
val Loss: 0.0209 Acc: 0.5130

Epoch 75/82
----------
train Loss: 0.0023 Acc: 0.9579
val Loss: 0.0218 Acc: 0.5093

Epoch 76/82
----------
train Loss: 0.0026 Acc: 0.9492
val Loss: 0.0214 Acc: 0.5279

Epoch 77/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0021 Acc: 0.9616
val Loss: 0.0211 Acc: 0.5242

Epoch 78/82
----------
train Loss: 0.0023 Acc: 0.9554
val Loss: 0.0208 Acc: 0.5242

Epoch 79/82
----------
train Loss: 0.0017 Acc: 0.9542
val Loss: 0.0207 Acc: 0.5316

Epoch 80/82
----------
train Loss: 0.0023 Acc: 0.9591
val Loss: 0.0211 Acc: 0.5242

Epoch 81/82
----------
train Loss: 0.0024 Acc: 0.9455
val Loss: 0.0210 Acc: 0.5242

Epoch 82/82
----------
train Loss: 0.0037 Acc: 0.9504
val Loss: 0.0205 Acc: 0.5130

Training complete in 8m 4s
Best val Acc: 0.542751

---Testing---
Test accuracy: 0.837361
--------------------
Accuracy of Albacore tuna : 82 %
Accuracy of Atlantic bluefin tuna : 79 %
Accuracy of Bigeye tuna : 74 %
Accuracy of Blackfin tuna : 89 %
Accuracy of Bullet tuna : 87 %
Accuracy of Frigate tuna : 58 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 90 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 69 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 35 %
Accuracy of Southern bluefin tuna : 68 %
Accuracy of Yellowfin tuna : 92 %
mean: 0.7835823703978037, std: 0.15706413086811558
--------------------

run info[val: 0.3, epoch: 69, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0270 Acc: 0.1724
val Loss: 0.0303 Acc: 0.2112

Epoch 1/68
----------
train Loss: 0.0227 Acc: 0.3249
val Loss: 0.0263 Acc: 0.3602

Epoch 2/68
----------
train Loss: 0.0197 Acc: 0.4284
val Loss: 0.0231 Acc: 0.3758

Epoch 3/68
----------
train Loss: 0.0175 Acc: 0.4841
val Loss: 0.0227 Acc: 0.4037

Epoch 4/68
----------
train Loss: 0.0162 Acc: 0.5146
val Loss: 0.0226 Acc: 0.4317

Epoch 5/68
----------
train Loss: 0.0148 Acc: 0.5491
val Loss: 0.0209 Acc: 0.4161

Epoch 6/68
----------
train Loss: 0.0138 Acc: 0.5955
val Loss: 0.0199 Acc: 0.4441

Epoch 7/68
----------
train Loss: 0.0132 Acc: 0.6141
val Loss: 0.0239 Acc: 0.4534

Epoch 8/68
----------
train Loss: 0.0125 Acc: 0.6340
val Loss: 0.0201 Acc: 0.4410

Epoch 9/68
----------
train Loss: 0.0124 Acc: 0.6379
val Loss: 0.0208 Acc: 0.4752

Epoch 10/68
----------
train Loss: 0.0113 Acc: 0.6883
val Loss: 0.0221 Acc: 0.4783

Epoch 11/68
----------
LR is set to 0.001
train Loss: 0.0110 Acc: 0.6790
val Loss: 0.0204 Acc: 0.4938

Epoch 12/68
----------
train Loss: 0.0104 Acc: 0.7188
val Loss: 0.0201 Acc: 0.4845

Epoch 13/68
----------
train Loss: 0.0107 Acc: 0.7042
val Loss: 0.0201 Acc: 0.4969

Epoch 14/68
----------
train Loss: 0.0105 Acc: 0.7042
val Loss: 0.0202 Acc: 0.4907

Epoch 15/68
----------
train Loss: 0.0103 Acc: 0.7095
val Loss: 0.0203 Acc: 0.5000

Epoch 16/68
----------
train Loss: 0.0105 Acc: 0.7109
val Loss: 0.0210 Acc: 0.4938

Epoch 17/68
----------
train Loss: 0.0107 Acc: 0.7082
val Loss: 0.0201 Acc: 0.4907

Epoch 18/68
----------
train Loss: 0.0106 Acc: 0.7175
val Loss: 0.0211 Acc: 0.5062

Epoch 19/68
----------
train Loss: 0.0105 Acc: 0.7294
val Loss: 0.0206 Acc: 0.4907

Epoch 20/68
----------
train Loss: 0.0105 Acc: 0.7188
val Loss: 0.0214 Acc: 0.5000

Epoch 21/68
----------
train Loss: 0.0104 Acc: 0.7281
val Loss: 0.0201 Acc: 0.4907

Epoch 22/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0105 Acc: 0.7493
val Loss: 0.0208 Acc: 0.4938

Epoch 23/68
----------
train Loss: 0.0102 Acc: 0.7202
val Loss: 0.0220 Acc: 0.5000

Epoch 24/68
----------
train Loss: 0.0101 Acc: 0.7440
val Loss: 0.0205 Acc: 0.4907

Epoch 25/68
----------
train Loss: 0.0102 Acc: 0.7241
val Loss: 0.0193 Acc: 0.4938

Epoch 26/68
----------
train Loss: 0.0103 Acc: 0.7281
val Loss: 0.0206 Acc: 0.4969

Epoch 27/68
----------
train Loss: 0.0106 Acc: 0.7082
val Loss: 0.0199 Acc: 0.4938

Epoch 28/68
----------
train Loss: 0.0103 Acc: 0.7347
val Loss: 0.0209 Acc: 0.4938

Epoch 29/68
----------
train Loss: 0.0102 Acc: 0.7241
val Loss: 0.0214 Acc: 0.4938

Epoch 30/68
----------
train Loss: 0.0100 Acc: 0.7308
val Loss: 0.0198 Acc: 0.4907

Epoch 31/68
----------
train Loss: 0.0102 Acc: 0.7480
val Loss: 0.0206 Acc: 0.4938

Epoch 32/68
----------
train Loss: 0.0101 Acc: 0.7467
val Loss: 0.0212 Acc: 0.4938

Epoch 33/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0104 Acc: 0.7069
val Loss: 0.0211 Acc: 0.4938

Epoch 34/68
----------
train Loss: 0.0102 Acc: 0.7308
val Loss: 0.0209 Acc: 0.4969

Epoch 35/68
----------
train Loss: 0.0103 Acc: 0.7374
val Loss: 0.0213 Acc: 0.4938

Epoch 36/68
----------
train Loss: 0.0103 Acc: 0.7255
val Loss: 0.0208 Acc: 0.4969

Epoch 37/68
----------
train Loss: 0.0101 Acc: 0.7347
val Loss: 0.0203 Acc: 0.4969

Epoch 38/68
----------
train Loss: 0.0102 Acc: 0.7334
val Loss: 0.0198 Acc: 0.4907

Epoch 39/68
----------
train Loss: 0.0102 Acc: 0.7188
val Loss: 0.0200 Acc: 0.4876

Epoch 40/68
----------
train Loss: 0.0100 Acc: 0.7308
val Loss: 0.0203 Acc: 0.4814

Epoch 41/68
----------
train Loss: 0.0101 Acc: 0.7228
val Loss: 0.0206 Acc: 0.4845

Epoch 42/68
----------
train Loss: 0.0100 Acc: 0.7361
val Loss: 0.0196 Acc: 0.4938

Epoch 43/68
----------
train Loss: 0.0098 Acc: 0.7334
val Loss: 0.0204 Acc: 0.4907

Epoch 44/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0103 Acc: 0.7202
val Loss: 0.0201 Acc: 0.4938

Epoch 45/68
----------
train Loss: 0.0101 Acc: 0.7334
val Loss: 0.0202 Acc: 0.4907

Epoch 46/68
----------
train Loss: 0.0102 Acc: 0.7321
val Loss: 0.0202 Acc: 0.4876

Epoch 47/68
----------
train Loss: 0.0103 Acc: 0.7228
val Loss: 0.0218 Acc: 0.4938

Epoch 48/68
----------
train Loss: 0.0100 Acc: 0.7321
val Loss: 0.0203 Acc: 0.4969

Epoch 49/68
----------
train Loss: 0.0099 Acc: 0.7467
val Loss: 0.0217 Acc: 0.4907

Epoch 50/68
----------
train Loss: 0.0103 Acc: 0.7281
val Loss: 0.0215 Acc: 0.4938

Epoch 51/68
----------
train Loss: 0.0105 Acc: 0.7135
val Loss: 0.0208 Acc: 0.4969

Epoch 52/68
----------
train Loss: 0.0102 Acc: 0.7202
val Loss: 0.0208 Acc: 0.4938

Epoch 53/68
----------
train Loss: 0.0101 Acc: 0.7454
val Loss: 0.0202 Acc: 0.4907

Epoch 54/68
----------
train Loss: 0.0103 Acc: 0.7347
val Loss: 0.0209 Acc: 0.4845

Epoch 55/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0103 Acc: 0.7321
val Loss: 0.0210 Acc: 0.4876

Epoch 56/68
----------
train Loss: 0.0101 Acc: 0.7321
val Loss: 0.0216 Acc: 0.4907

Epoch 57/68
----------
train Loss: 0.0102 Acc: 0.7387
val Loss: 0.0216 Acc: 0.4907

Epoch 58/68
----------
train Loss: 0.0100 Acc: 0.7440
val Loss: 0.0210 Acc: 0.4907

Epoch 59/68
----------
train Loss: 0.0105 Acc: 0.7162
val Loss: 0.0203 Acc: 0.4938

Epoch 60/68
----------
train Loss: 0.0100 Acc: 0.7387
val Loss: 0.0201 Acc: 0.4938

Epoch 61/68
----------
train Loss: 0.0098 Acc: 0.7374
val Loss: 0.0220 Acc: 0.4938

Epoch 62/68
----------
train Loss: 0.0104 Acc: 0.7188
val Loss: 0.0194 Acc: 0.5000

Epoch 63/68
----------
train Loss: 0.0103 Acc: 0.7294
val Loss: 0.0206 Acc: 0.4938

Epoch 64/68
----------
train Loss: 0.0100 Acc: 0.7149
val Loss: 0.0207 Acc: 0.4907

Epoch 65/68
----------
train Loss: 0.0103 Acc: 0.7347
val Loss: 0.0203 Acc: 0.4938

Epoch 66/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0104 Acc: 0.7241
val Loss: 0.0215 Acc: 0.4907

Epoch 67/68
----------
train Loss: 0.0099 Acc: 0.7347
val Loss: 0.0195 Acc: 0.4938

Epoch 68/68
----------
train Loss: 0.0102 Acc: 0.7414
val Loss: 0.0213 Acc: 0.4938

Training complete in 6m 22s
Best val Acc: 0.506211

---Fine tuning.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0107 Acc: 0.7215
val Loss: 0.0216 Acc: 0.4814

Epoch 1/68
----------
train Loss: 0.0077 Acc: 0.7931
val Loss: 0.0229 Acc: 0.4472

Epoch 2/68
----------
train Loss: 0.0048 Acc: 0.8833
val Loss: 0.0220 Acc: 0.4938

Epoch 3/68
----------
train Loss: 0.0033 Acc: 0.9218
val Loss: 0.0194 Acc: 0.4938

Epoch 4/68
----------
train Loss: 0.0022 Acc: 0.9456
val Loss: 0.0213 Acc: 0.5217

Epoch 5/68
----------
train Loss: 0.0015 Acc: 0.9682
val Loss: 0.0215 Acc: 0.4907

Epoch 6/68
----------
train Loss: 0.0013 Acc: 0.9788
val Loss: 0.0226 Acc: 0.5342

Epoch 7/68
----------
train Loss: 0.0011 Acc: 0.9761
val Loss: 0.0236 Acc: 0.5031

Epoch 8/68
----------
train Loss: 0.0009 Acc: 0.9814
val Loss: 0.0230 Acc: 0.5155

Epoch 9/68
----------
train Loss: 0.0007 Acc: 0.9841
val Loss: 0.0230 Acc: 0.5373

Epoch 10/68
----------
train Loss: 0.0006 Acc: 0.9854
val Loss: 0.0215 Acc: 0.5435

Epoch 11/68
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9894
val Loss: 0.0217 Acc: 0.5435

Epoch 12/68
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0220 Acc: 0.5559

Epoch 13/68
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0230 Acc: 0.5590

Epoch 14/68
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0222 Acc: 0.5590

Epoch 15/68
----------
train Loss: 0.0006 Acc: 0.9841
val Loss: 0.0216 Acc: 0.5559

Epoch 16/68
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0223 Acc: 0.5559

Epoch 17/68
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0218 Acc: 0.5590

Epoch 18/68
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0206 Acc: 0.5497

Epoch 19/68
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0232 Acc: 0.5497

Epoch 20/68
----------
train Loss: 0.0004 Acc: 0.9854
val Loss: 0.0211 Acc: 0.5497

Epoch 21/68
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0221 Acc: 0.5466

Epoch 22/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0219 Acc: 0.5466

Epoch 23/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0205 Acc: 0.5497

Epoch 24/68
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0211 Acc: 0.5466

Epoch 25/68
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0223 Acc: 0.5466

Epoch 26/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0216 Acc: 0.5497

Epoch 27/68
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0227 Acc: 0.5497

Epoch 28/68
----------
train Loss: 0.0006 Acc: 0.9841
val Loss: 0.0224 Acc: 0.5497

Epoch 29/68
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0213 Acc: 0.5497

Epoch 30/68
----------
train Loss: 0.0005 Acc: 0.9841
val Loss: 0.0224 Acc: 0.5466

Epoch 31/68
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0229 Acc: 0.5466

Epoch 32/68
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0206 Acc: 0.5466

Epoch 33/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0223 Acc: 0.5466

Epoch 34/68
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0231 Acc: 0.5497

Epoch 35/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0212 Acc: 0.5497

Epoch 36/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0208 Acc: 0.5497

Epoch 37/68
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0217 Acc: 0.5466

Epoch 38/68
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0226 Acc: 0.5497

Epoch 39/68
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0209 Acc: 0.5466

Epoch 40/68
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0200 Acc: 0.5466

Epoch 41/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0235 Acc: 0.5435

Epoch 42/68
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0205 Acc: 0.5466

Epoch 43/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0213 Acc: 0.5497

Epoch 44/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0211 Acc: 0.5497

Epoch 45/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0215 Acc: 0.5497

Epoch 46/68
----------
train Loss: 0.0004 Acc: 0.9828
val Loss: 0.0246 Acc: 0.5497

Epoch 47/68
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0222 Acc: 0.5497

Epoch 48/68
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0215 Acc: 0.5466

Epoch 49/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0205 Acc: 0.5466

Epoch 50/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0222 Acc: 0.5466

Epoch 51/68
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0226 Acc: 0.5435

Epoch 52/68
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0229 Acc: 0.5466

Epoch 53/68
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0226 Acc: 0.5497

Epoch 54/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0222 Acc: 0.5497

Epoch 55/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0212 Acc: 0.5466

Epoch 56/68
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0209 Acc: 0.5404

Epoch 57/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0229 Acc: 0.5435

Epoch 58/68
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0212 Acc: 0.5435

Epoch 59/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0201 Acc: 0.5466

Epoch 60/68
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0222 Acc: 0.5466

Epoch 61/68
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0222 Acc: 0.5497

Epoch 62/68
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0227 Acc: 0.5497

Epoch 63/68
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0229 Acc: 0.5435

Epoch 64/68
----------
train Loss: 0.0005 Acc: 0.9841
val Loss: 0.0226 Acc: 0.5466

Epoch 65/68
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0223 Acc: 0.5466

Epoch 66/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0221 Acc: 0.5497

Epoch 67/68
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0223 Acc: 0.5466

Epoch 68/68
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0210 Acc: 0.5435

Training complete in 6m 46s
Best val Acc: 0.559006

---Testing---
Test accuracy: 0.862454
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 79 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 86 %
Accuracy of Frigate tuna : 72 %
Accuracy of Little tunny : 84 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 75 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8123097377945748, std: 0.12845938078370267

Model saved in "./weights/tuna_fish_[0.91]_mean[0.89]_std[0.07].save".
--------------------

run info[val: 0.1, epoch: 65, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0271 Acc: 0.1269
val Loss: 0.0325 Acc: 0.3364

Epoch 1/64
----------
train Loss: 0.0213 Acc: 0.3457
val Loss: 0.0338 Acc: 0.4299

Epoch 2/64
----------
train Loss: 0.0175 Acc: 0.4696
val Loss: 0.0373 Acc: 0.4393

Epoch 3/64
----------
train Loss: 0.0160 Acc: 0.5057
val Loss: 0.0270 Acc: 0.4393

Epoch 4/64
----------
train Loss: 0.0146 Acc: 0.5480
val Loss: 0.0284 Acc: 0.4579

Epoch 5/64
----------
train Loss: 0.0134 Acc: 0.6027
val Loss: 0.0275 Acc: 0.4673

Epoch 6/64
----------
train Loss: 0.0123 Acc: 0.6171
val Loss: 0.0244 Acc: 0.4766

Epoch 7/64
----------
train Loss: 0.0117 Acc: 0.6605
val Loss: 0.0333 Acc: 0.4860

Epoch 8/64
----------
train Loss: 0.0111 Acc: 0.6842
val Loss: 0.0282 Acc: 0.4860

Epoch 9/64
----------
LR is set to 0.001
train Loss: 0.0105 Acc: 0.6987
val Loss: 0.0301 Acc: 0.5140

Epoch 10/64
----------
train Loss: 0.0100 Acc: 0.7348
val Loss: 0.0254 Acc: 0.4953

Epoch 11/64
----------
train Loss: 0.0099 Acc: 0.7337
val Loss: 0.0353 Acc: 0.5140

Epoch 12/64
----------
train Loss: 0.0100 Acc: 0.7183
val Loss: 0.0289 Acc: 0.5140

Epoch 13/64
----------
train Loss: 0.0099 Acc: 0.7234
val Loss: 0.0332 Acc: 0.5234

Epoch 14/64
----------
train Loss: 0.0099 Acc: 0.7348
val Loss: 0.0282 Acc: 0.4953

Epoch 15/64
----------
train Loss: 0.0096 Acc: 0.7492
val Loss: 0.0227 Acc: 0.5140

Epoch 16/64
----------
train Loss: 0.0096 Acc: 0.7534
val Loss: 0.0385 Acc: 0.5234

Epoch 17/64
----------
train Loss: 0.0097 Acc: 0.7337
val Loss: 0.0329 Acc: 0.5047

Epoch 18/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0096 Acc: 0.7513
val Loss: 0.0262 Acc: 0.4953

Epoch 19/64
----------
train Loss: 0.0097 Acc: 0.7296
val Loss: 0.0332 Acc: 0.4953

Epoch 20/64
----------
train Loss: 0.0097 Acc: 0.7492
val Loss: 0.0281 Acc: 0.5047

Epoch 21/64
----------
train Loss: 0.0097 Acc: 0.7389
val Loss: 0.0259 Acc: 0.5234

Epoch 22/64
----------
train Loss: 0.0095 Acc: 0.7564
val Loss: 0.0252 Acc: 0.5234

Epoch 23/64
----------
train Loss: 0.0096 Acc: 0.7420
val Loss: 0.0285 Acc: 0.5140

Epoch 24/64
----------
train Loss: 0.0097 Acc: 0.7441
val Loss: 0.0261 Acc: 0.5234

Epoch 25/64
----------
train Loss: 0.0097 Acc: 0.7420
val Loss: 0.0289 Acc: 0.4953

Epoch 26/64
----------
train Loss: 0.0096 Acc: 0.7358
val Loss: 0.0366 Acc: 0.5047

Epoch 27/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0096 Acc: 0.7441
val Loss: 0.0284 Acc: 0.5140

Epoch 28/64
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0233 Acc: 0.5140

Epoch 29/64
----------
train Loss: 0.0096 Acc: 0.7358
val Loss: 0.0286 Acc: 0.5234

Epoch 30/64
----------
train Loss: 0.0095 Acc: 0.7637
val Loss: 0.0237 Acc: 0.5234

Epoch 31/64
----------
train Loss: 0.0096 Acc: 0.7575
val Loss: 0.0306 Acc: 0.5234

Epoch 32/64
----------
train Loss: 0.0096 Acc: 0.7430
val Loss: 0.0277 Acc: 0.5047

Epoch 33/64
----------
train Loss: 0.0096 Acc: 0.7368
val Loss: 0.0269 Acc: 0.5234

Epoch 34/64
----------
train Loss: 0.0096 Acc: 0.7523
val Loss: 0.0341 Acc: 0.5234

Epoch 35/64
----------
train Loss: 0.0096 Acc: 0.7637
val Loss: 0.0313 Acc: 0.5327

Epoch 36/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0227 Acc: 0.5234

Epoch 37/64
----------
train Loss: 0.0097 Acc: 0.7482
val Loss: 0.0401 Acc: 0.5140

Epoch 38/64
----------
train Loss: 0.0096 Acc: 0.7430
val Loss: 0.0331 Acc: 0.5140

Epoch 39/64
----------
train Loss: 0.0096 Acc: 0.7482
val Loss: 0.0339 Acc: 0.5140

Epoch 40/64
----------
train Loss: 0.0096 Acc: 0.7482
val Loss: 0.0277 Acc: 0.5047

Epoch 41/64
----------
train Loss: 0.0096 Acc: 0.7503
val Loss: 0.0278 Acc: 0.5234

Epoch 42/64
----------
train Loss: 0.0095 Acc: 0.7461
val Loss: 0.0305 Acc: 0.5327

Epoch 43/64
----------
train Loss: 0.0097 Acc: 0.7461
val Loss: 0.0309 Acc: 0.5140

Epoch 44/64
----------
train Loss: 0.0097 Acc: 0.7410
val Loss: 0.0287 Acc: 0.5234

Epoch 45/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0097 Acc: 0.7461
val Loss: 0.0301 Acc: 0.5327

Epoch 46/64
----------
train Loss: 0.0096 Acc: 0.7399
val Loss: 0.0279 Acc: 0.5140

Epoch 47/64
----------
train Loss: 0.0096 Acc: 0.7513
val Loss: 0.0364 Acc: 0.5234

Epoch 48/64
----------
train Loss: 0.0096 Acc: 0.7379
val Loss: 0.0296 Acc: 0.5234

Epoch 49/64
----------
train Loss: 0.0094 Acc: 0.7554
val Loss: 0.0261 Acc: 0.5327

Epoch 50/64
----------
train Loss: 0.0097 Acc: 0.7492
val Loss: 0.0305 Acc: 0.5234

Epoch 51/64
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0227 Acc: 0.5327

Epoch 52/64
----------
train Loss: 0.0095 Acc: 0.7513
val Loss: 0.0323 Acc: 0.5327

Epoch 53/64
----------
train Loss: 0.0097 Acc: 0.7534
val Loss: 0.0369 Acc: 0.4953

Epoch 54/64
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0098 Acc: 0.7327
val Loss: 0.0293 Acc: 0.5234

Epoch 55/64
----------
train Loss: 0.0096 Acc: 0.7379
val Loss: 0.0235 Acc: 0.5047

Epoch 56/64
----------
train Loss: 0.0097 Acc: 0.7461
val Loss: 0.0337 Acc: 0.5234

Epoch 57/64
----------
train Loss: 0.0097 Acc: 0.7389
val Loss: 0.0257 Acc: 0.5047

Epoch 58/64
----------
train Loss: 0.0095 Acc: 0.7564
val Loss: 0.0195 Acc: 0.5047

Epoch 59/64
----------
train Loss: 0.0096 Acc: 0.7441
val Loss: 0.0279 Acc: 0.5140

Epoch 60/64
----------
train Loss: 0.0095 Acc: 0.7523
val Loss: 0.0298 Acc: 0.5140

Epoch 61/64
----------
train Loss: 0.0097 Acc: 0.7410
val Loss: 0.0271 Acc: 0.5140

Epoch 62/64
----------
train Loss: 0.0096 Acc: 0.7513
val Loss: 0.0275 Acc: 0.5234

Epoch 63/64
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0096 Acc: 0.7472
val Loss: 0.0365 Acc: 0.5327

Epoch 64/64
----------
train Loss: 0.0096 Acc: 0.7564
val Loss: 0.0195 Acc: 0.5140

Training complete in 6m 2s
Best val Acc: 0.532710

---Fine tuning.---
Epoch 0/64
----------
LR is set to 0.01
train Loss: 0.0101 Acc: 0.6873
val Loss: 0.0231 Acc: 0.5514

Epoch 1/64
----------
train Loss: 0.0059 Acc: 0.8524
val Loss: 0.0241 Acc: 0.5794

Epoch 2/64
----------
train Loss: 0.0034 Acc: 0.9195
val Loss: 0.0254 Acc: 0.5421

Epoch 3/64
----------
train Loss: 0.0020 Acc: 0.9567
val Loss: 0.0271 Acc: 0.6355

Epoch 4/64
----------
train Loss: 0.0014 Acc: 0.9680
val Loss: 0.0365 Acc: 0.5514

Epoch 5/64
----------
train Loss: 0.0010 Acc: 0.9711
val Loss: 0.0162 Acc: 0.5888

Epoch 6/64
----------
train Loss: 0.0009 Acc: 0.9783
val Loss: 0.0263 Acc: 0.5794

Epoch 7/64
----------
train Loss: 0.0008 Acc: 0.9742
val Loss: 0.0391 Acc: 0.5701

Epoch 8/64
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0314 Acc: 0.5234

Epoch 9/64
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0335 Acc: 0.5327

Epoch 10/64
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0242 Acc: 0.5421

Epoch 11/64
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0311 Acc: 0.5421

Epoch 12/64
----------
train Loss: 0.0005 Acc: 0.9794
val Loss: 0.0335 Acc: 0.5421

Epoch 13/64
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0250 Acc: 0.5607

Epoch 14/64
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0286 Acc: 0.5607

Epoch 15/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0255 Acc: 0.5607

Epoch 16/64
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0425 Acc: 0.5514

Epoch 17/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0418 Acc: 0.5701

Epoch 18/64
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0233 Acc: 0.5607

Epoch 19/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0448 Acc: 0.5607

Epoch 20/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0440 Acc: 0.5607

Epoch 21/64
----------
train Loss: 0.0003 Acc: 0.9845
val Loss: 0.0263 Acc: 0.5607

Epoch 22/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0267 Acc: 0.5607

Epoch 23/64
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0250 Acc: 0.5607

Epoch 24/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0331 Acc: 0.5514

Epoch 25/64
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0295 Acc: 0.5514

Epoch 26/64
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0443 Acc: 0.5607

Epoch 27/64
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0275 Acc: 0.5607

Epoch 28/64
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0317 Acc: 0.5607

Epoch 29/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0182 Acc: 0.5607

Epoch 30/64
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0306 Acc: 0.5607

Epoch 31/64
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0370 Acc: 0.5607

Epoch 32/64
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0279 Acc: 0.5607

Epoch 33/64
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0323 Acc: 0.5607

Epoch 34/64
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0316 Acc: 0.5514

Epoch 35/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0443 Acc: 0.5514

Epoch 36/64
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0238 Acc: 0.5607

Epoch 37/64
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0235 Acc: 0.5514

Epoch 38/64
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0313 Acc: 0.5607

Epoch 39/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0256 Acc: 0.5514

Epoch 40/64
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0198 Acc: 0.5514

Epoch 41/64
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0221 Acc: 0.5607

Epoch 42/64
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0223 Acc: 0.5514

Epoch 43/64
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0255 Acc: 0.5514

Epoch 44/64
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0253 Acc: 0.5514

Epoch 45/64
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0300 Acc: 0.5514

Epoch 46/64
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0411 Acc: 0.5607

Epoch 47/64
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0205 Acc: 0.5514

Epoch 48/64
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0323 Acc: 0.5607

Epoch 49/64
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0364 Acc: 0.5701

Epoch 50/64
----------
train Loss: 0.0003 Acc: 0.9938
val Loss: 0.0271 Acc: 0.5607

Epoch 51/64
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0486 Acc: 0.5607

Epoch 52/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0425 Acc: 0.5514

Epoch 53/64
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0349 Acc: 0.5701

Epoch 54/64
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0316 Acc: 0.5701

Epoch 55/64
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0230 Acc: 0.5701

Epoch 56/64
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0279 Acc: 0.5607

Epoch 57/64
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0275 Acc: 0.5514

Epoch 58/64
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0244 Acc: 0.5514

Epoch 59/64
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0232 Acc: 0.5421

Epoch 60/64
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0414 Acc: 0.5607

Epoch 61/64
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0261 Acc: 0.5514

Epoch 62/64
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0339 Acc: 0.5701

Epoch 63/64
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0248 Acc: 0.5514

Epoch 64/64
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0244 Acc: 0.5607

Training complete in 6m 28s
Best val Acc: 0.635514

---Testing---
Test accuracy: 0.936803
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 90 %
Accuracy of Bigeye tuna : 89 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 95 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 97 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 84 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.915811720582159, std: 0.0584354943558201
--------------------

run info[val: 0.15, epoch: 61, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0277 Acc: 0.1792
val Loss: 0.0291 Acc: 0.2236

Epoch 1/60
----------
train Loss: 0.0231 Acc: 0.2962
val Loss: 0.0265 Acc: 0.3478

Epoch 2/60
----------
train Loss: 0.0198 Acc: 0.4142
val Loss: 0.0237 Acc: 0.3789

Epoch 3/60
----------
train Loss: 0.0179 Acc: 0.4754
val Loss: 0.0232 Acc: 0.3913

Epoch 4/60
----------
train Loss: 0.0166 Acc: 0.5355
val Loss: 0.0223 Acc: 0.4348

Epoch 5/60
----------
train Loss: 0.0160 Acc: 0.5344
val Loss: 0.0240 Acc: 0.4161

Epoch 6/60
----------
train Loss: 0.0151 Acc: 0.5563
val Loss: 0.0230 Acc: 0.4534

Epoch 7/60
----------
train Loss: 0.0142 Acc: 0.5781
val Loss: 0.0227 Acc: 0.4286

Epoch 8/60
----------
train Loss: 0.0136 Acc: 0.6033
val Loss: 0.0216 Acc: 0.4720

Epoch 9/60
----------
LR is set to 0.001
train Loss: 0.0129 Acc: 0.6437
val Loss: 0.0207 Acc: 0.5155

Epoch 10/60
----------
train Loss: 0.0124 Acc: 0.6754
val Loss: 0.0214 Acc: 0.4720

Epoch 11/60
----------
train Loss: 0.0124 Acc: 0.6787
val Loss: 0.0210 Acc: 0.4472

Epoch 12/60
----------
train Loss: 0.0124 Acc: 0.6721
val Loss: 0.0208 Acc: 0.4720

Epoch 13/60
----------
train Loss: 0.0117 Acc: 0.6721
val Loss: 0.0203 Acc: 0.4720

Epoch 14/60
----------
train Loss: 0.0120 Acc: 0.6667
val Loss: 0.0214 Acc: 0.4783

Epoch 15/60
----------
train Loss: 0.0116 Acc: 0.6831
val Loss: 0.0210 Acc: 0.4720

Epoch 16/60
----------
train Loss: 0.0117 Acc: 0.6962
val Loss: 0.0211 Acc: 0.4658

Epoch 17/60
----------
train Loss: 0.0116 Acc: 0.6863
val Loss: 0.0204 Acc: 0.4658

Epoch 18/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0116 Acc: 0.7060
val Loss: 0.0205 Acc: 0.4720

Epoch 19/60
----------
train Loss: 0.0113 Acc: 0.6842
val Loss: 0.0215 Acc: 0.4658

Epoch 20/60
----------
train Loss: 0.0121 Acc: 0.6710
val Loss: 0.0208 Acc: 0.4658

Epoch 21/60
----------
train Loss: 0.0116 Acc: 0.6896
val Loss: 0.0211 Acc: 0.4720

Epoch 22/60
----------
train Loss: 0.0113 Acc: 0.6940
val Loss: 0.0210 Acc: 0.4720

Epoch 23/60
----------
train Loss: 0.0118 Acc: 0.6940
val Loss: 0.0212 Acc: 0.4720

Epoch 24/60
----------
train Loss: 0.0116 Acc: 0.6929
val Loss: 0.0201 Acc: 0.4658

Epoch 25/60
----------
train Loss: 0.0116 Acc: 0.6984
val Loss: 0.0209 Acc: 0.4658

Epoch 26/60
----------
train Loss: 0.0111 Acc: 0.7027
val Loss: 0.0211 Acc: 0.4658

Epoch 27/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0113 Acc: 0.7082
val Loss: 0.0211 Acc: 0.4658

Epoch 28/60
----------
train Loss: 0.0114 Acc: 0.6842
val Loss: 0.0219 Acc: 0.4658

Epoch 29/60
----------
train Loss: 0.0111 Acc: 0.6951
val Loss: 0.0212 Acc: 0.4658

Epoch 30/60
----------
train Loss: 0.0111 Acc: 0.6918
val Loss: 0.0208 Acc: 0.4658

Epoch 31/60
----------
train Loss: 0.0113 Acc: 0.7060
val Loss: 0.0211 Acc: 0.4658

Epoch 32/60
----------
train Loss: 0.0113 Acc: 0.6940
val Loss: 0.0207 Acc: 0.4658

Epoch 33/60
----------
train Loss: 0.0114 Acc: 0.6929
val Loss: 0.0205 Acc: 0.4720

Epoch 34/60
----------
train Loss: 0.0113 Acc: 0.7126
val Loss: 0.0211 Acc: 0.4720

Epoch 35/60
----------
train Loss: 0.0110 Acc: 0.6852
val Loss: 0.0212 Acc: 0.4720

Epoch 36/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0115 Acc: 0.6874
val Loss: 0.0209 Acc: 0.4658

Epoch 37/60
----------
train Loss: 0.0113 Acc: 0.6951
val Loss: 0.0210 Acc: 0.4658

Epoch 38/60
----------
train Loss: 0.0118 Acc: 0.6973
val Loss: 0.0210 Acc: 0.4658

Epoch 39/60
----------
train Loss: 0.0109 Acc: 0.6896
val Loss: 0.0203 Acc: 0.4658

Epoch 40/60
----------
train Loss: 0.0114 Acc: 0.6831
val Loss: 0.0210 Acc: 0.4658

Epoch 41/60
----------
train Loss: 0.0117 Acc: 0.6896
val Loss: 0.0202 Acc: 0.4658

Epoch 42/60
----------
train Loss: 0.0113 Acc: 0.7005
val Loss: 0.0213 Acc: 0.4596

Epoch 43/60
----------
train Loss: 0.0114 Acc: 0.7005
val Loss: 0.0209 Acc: 0.4658

Epoch 44/60
----------
train Loss: 0.0115 Acc: 0.6863
val Loss: 0.0210 Acc: 0.4658

Epoch 45/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0118 Acc: 0.7049
val Loss: 0.0209 Acc: 0.4720

Epoch 46/60
----------
train Loss: 0.0114 Acc: 0.6962
val Loss: 0.0207 Acc: 0.4720

Epoch 47/60
----------
train Loss: 0.0117 Acc: 0.6798
val Loss: 0.0210 Acc: 0.4658

Epoch 48/60
----------
train Loss: 0.0110 Acc: 0.6984
val Loss: 0.0208 Acc: 0.4658

Epoch 49/60
----------
train Loss: 0.0112 Acc: 0.6973
val Loss: 0.0211 Acc: 0.4658

Epoch 50/60
----------
train Loss: 0.0114 Acc: 0.6962
val Loss: 0.0207 Acc: 0.4720

Epoch 51/60
----------
train Loss: 0.0116 Acc: 0.6995
val Loss: 0.0207 Acc: 0.4658

Epoch 52/60
----------
train Loss: 0.0114 Acc: 0.7005
val Loss: 0.0204 Acc: 0.4658

Epoch 53/60
----------
train Loss: 0.0114 Acc: 0.6973
val Loss: 0.0211 Acc: 0.4658

Epoch 54/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0115 Acc: 0.6852
val Loss: 0.0208 Acc: 0.4658

Epoch 55/60
----------
train Loss: 0.0113 Acc: 0.7027
val Loss: 0.0209 Acc: 0.4658

Epoch 56/60
----------
train Loss: 0.0113 Acc: 0.6984
val Loss: 0.0208 Acc: 0.4658

Epoch 57/60
----------
train Loss: 0.0114 Acc: 0.6842
val Loss: 0.0209 Acc: 0.4658

Epoch 58/60
----------
train Loss: 0.0116 Acc: 0.6918
val Loss: 0.0211 Acc: 0.4658

Epoch 59/60
----------
train Loss: 0.0114 Acc: 0.7016
val Loss: 0.0205 Acc: 0.4658

Epoch 60/60
----------
train Loss: 0.0113 Acc: 0.6896
val Loss: 0.0205 Acc: 0.4658

Training complete in 5m 38s
Best val Acc: 0.515528

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0120 Acc: 0.6525
val Loss: 0.0231 Acc: 0.4161

Epoch 1/60
----------
train Loss: 0.0099 Acc: 0.7049
val Loss: 0.0260 Acc: 0.4037

Epoch 2/60
----------
train Loss: 0.0072 Acc: 0.7880
val Loss: 0.0240 Acc: 0.5155

Epoch 3/60
----------
train Loss: 0.0049 Acc: 0.8656
val Loss: 0.0217 Acc: 0.4410

Epoch 4/60
----------
train Loss: 0.0039 Acc: 0.9093
val Loss: 0.0230 Acc: 0.5155

Epoch 5/60
----------
train Loss: 0.0037 Acc: 0.9060
val Loss: 0.0238 Acc: 0.4783

Epoch 6/60
----------
train Loss: 0.0027 Acc: 0.9268
val Loss: 0.0246 Acc: 0.4783

Epoch 7/60
----------
train Loss: 0.0019 Acc: 0.9475
val Loss: 0.0278 Acc: 0.4907

Epoch 8/60
----------
train Loss: 0.0016 Acc: 0.9607
val Loss: 0.0247 Acc: 0.5404

Epoch 9/60
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9705
val Loss: 0.0246 Acc: 0.5280

Epoch 10/60
----------
train Loss: 0.0010 Acc: 0.9803
val Loss: 0.0259 Acc: 0.5280

Epoch 11/60
----------
train Loss: 0.0012 Acc: 0.9749
val Loss: 0.0243 Acc: 0.5217

Epoch 12/60
----------
train Loss: 0.0012 Acc: 0.9683
val Loss: 0.0255 Acc: 0.5217

Epoch 13/60
----------
train Loss: 0.0009 Acc: 0.9770
val Loss: 0.0258 Acc: 0.5280

Epoch 14/60
----------
train Loss: 0.0012 Acc: 0.9770
val Loss: 0.0263 Acc: 0.5280

Epoch 15/60
----------
train Loss: 0.0009 Acc: 0.9858
val Loss: 0.0258 Acc: 0.5155

Epoch 16/60
----------
train Loss: 0.0010 Acc: 0.9803
val Loss: 0.0263 Acc: 0.5155

Epoch 17/60
----------
train Loss: 0.0009 Acc: 0.9749
val Loss: 0.0260 Acc: 0.5217

Epoch 18/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9781
val Loss: 0.0246 Acc: 0.5217

Epoch 19/60
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0252 Acc: 0.5280

Epoch 20/60
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0247 Acc: 0.5217

Epoch 21/60
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0258 Acc: 0.5155

Epoch 22/60
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0271 Acc: 0.5217

Epoch 23/60
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0260 Acc: 0.5217

Epoch 24/60
----------
train Loss: 0.0007 Acc: 0.9836
val Loss: 0.0261 Acc: 0.5342

Epoch 25/60
----------
train Loss: 0.0008 Acc: 0.9858
val Loss: 0.0249 Acc: 0.5280

Epoch 26/60
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0251 Acc: 0.5280

Epoch 27/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9760
val Loss: 0.0252 Acc: 0.5217

Epoch 28/60
----------
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0252 Acc: 0.5280

Epoch 29/60
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0257 Acc: 0.5155

Epoch 30/60
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0259 Acc: 0.5280

Epoch 31/60
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0242 Acc: 0.5217

Epoch 32/60
----------
train Loss: 0.0007 Acc: 0.9738
val Loss: 0.0261 Acc: 0.5342

Epoch 33/60
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0260 Acc: 0.5280

Epoch 34/60
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0261 Acc: 0.5280

Epoch 35/60
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0259 Acc: 0.5155

Epoch 36/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0009 Acc: 0.9847
val Loss: 0.0264 Acc: 0.5280

Epoch 37/60
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0269 Acc: 0.5217

Epoch 38/60
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0256 Acc: 0.5217

Epoch 39/60
----------
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0267 Acc: 0.5280

Epoch 40/60
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5217

Epoch 41/60
----------
train Loss: 0.0010 Acc: 0.9749
val Loss: 0.0256 Acc: 0.5217

Epoch 42/60
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0250 Acc: 0.5155

Epoch 43/60
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0259 Acc: 0.5217

Epoch 44/60
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0249 Acc: 0.5217

Epoch 45/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5342

Epoch 46/60
----------
train Loss: 0.0010 Acc: 0.9760
val Loss: 0.0243 Acc: 0.5280

Epoch 47/60
----------
train Loss: 0.0007 Acc: 0.9858
val Loss: 0.0263 Acc: 0.5155

Epoch 48/60
----------
train Loss: 0.0008 Acc: 0.9749
val Loss: 0.0258 Acc: 0.5155

Epoch 49/60
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5217

Epoch 50/60
----------
train Loss: 0.0008 Acc: 0.9727
val Loss: 0.0247 Acc: 0.5280

Epoch 51/60
----------
train Loss: 0.0008 Acc: 0.9847
val Loss: 0.0235 Acc: 0.5342

Epoch 52/60
----------
train Loss: 0.0008 Acc: 0.9891
val Loss: 0.0254 Acc: 0.5280

Epoch 53/60
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0250 Acc: 0.5342

Epoch 54/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0245 Acc: 0.5342

Epoch 55/60
----------
train Loss: 0.0008 Acc: 0.9792
val Loss: 0.0258 Acc: 0.5217

Epoch 56/60
----------
train Loss: 0.0009 Acc: 0.9814
val Loss: 0.0248 Acc: 0.5217

Epoch 57/60
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0260 Acc: 0.5280

Epoch 58/60
----------
train Loss: 0.0010 Acc: 0.9825
val Loss: 0.0248 Acc: 0.5155

Epoch 59/60
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5217

Epoch 60/60
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0258 Acc: 0.5280

Training complete in 5m 59s
Best val Acc: 0.540373

---Testing---
Test accuracy: 0.904275
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 76 %
Accuracy of Bigeye tuna : 88 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 79 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 78 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8755181094113489, std: 0.08091734270648049
--------------------

run info[val: 0.2, epoch: 52, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/51
----------
LR is set to 0.01
train Loss: 0.0279 Acc: 0.1382
val Loss: 0.0319 Acc: 0.2419

Epoch 1/51
----------
train Loss: 0.0220 Acc: 0.3322
val Loss: 0.0284 Acc: 0.3535

Epoch 2/51
----------
train Loss: 0.0187 Acc: 0.4413
val Loss: 0.0284 Acc: 0.3442

Epoch 3/51
----------
train Loss: 0.0164 Acc: 0.4866
val Loss: 0.0224 Acc: 0.4000

Epoch 4/51
----------
train Loss: 0.0145 Acc: 0.5389
val Loss: 0.0256 Acc: 0.4093

Epoch 5/51
----------
train Loss: 0.0134 Acc: 0.6016
val Loss: 0.0235 Acc: 0.4465

Epoch 6/51
----------
LR is set to 0.001
train Loss: 0.0125 Acc: 0.6295
val Loss: 0.0223 Acc: 0.4791

Epoch 7/51
----------
train Loss: 0.0118 Acc: 0.6643
val Loss: 0.0223 Acc: 0.4465

Epoch 8/51
----------
train Loss: 0.0118 Acc: 0.6655
val Loss: 0.0243 Acc: 0.4372

Epoch 9/51
----------
train Loss: 0.0118 Acc: 0.6911
val Loss: 0.0218 Acc: 0.4558

Epoch 10/51
----------
train Loss: 0.0117 Acc: 0.6771
val Loss: 0.0242 Acc: 0.4605

Epoch 11/51
----------
train Loss: 0.0116 Acc: 0.6911
val Loss: 0.0232 Acc: 0.4605

Epoch 12/51
----------
LR is set to 0.00010000000000000002
train Loss: 0.0113 Acc: 0.6969
val Loss: 0.0234 Acc: 0.4605

Epoch 13/51
----------
train Loss: 0.0117 Acc: 0.6818
val Loss: 0.0244 Acc: 0.4605

Epoch 14/51
----------
train Loss: 0.0115 Acc: 0.6945
val Loss: 0.0230 Acc: 0.4605

Epoch 15/51
----------
train Loss: 0.0116 Acc: 0.6864
val Loss: 0.0285 Acc: 0.4651

Epoch 16/51
----------
train Loss: 0.0115 Acc: 0.6794
val Loss: 0.0240 Acc: 0.4605

Epoch 17/51
----------
train Loss: 0.0116 Acc: 0.6852
val Loss: 0.0240 Acc: 0.4512

Epoch 18/51
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0115 Acc: 0.6771
val Loss: 0.0248 Acc: 0.4558

Epoch 19/51
----------
train Loss: 0.0114 Acc: 0.6841
val Loss: 0.0245 Acc: 0.4558

Epoch 20/51
----------
train Loss: 0.0115 Acc: 0.6934
val Loss: 0.0233 Acc: 0.4512

Epoch 21/51
----------
train Loss: 0.0114 Acc: 0.6922
val Loss: 0.0243 Acc: 0.4558

Epoch 22/51
----------
train Loss: 0.0114 Acc: 0.6945
val Loss: 0.0245 Acc: 0.4558

Epoch 23/51
----------
train Loss: 0.0115 Acc: 0.6794
val Loss: 0.0244 Acc: 0.4605

Epoch 24/51
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0116 Acc: 0.6934
val Loss: 0.0229 Acc: 0.4605

Epoch 25/51
----------
train Loss: 0.0115 Acc: 0.6922
val Loss: 0.0239 Acc: 0.4651

Epoch 26/51
----------
train Loss: 0.0115 Acc: 0.6864
val Loss: 0.0248 Acc: 0.4651

Epoch 27/51
----------
train Loss: 0.0116 Acc: 0.6783
val Loss: 0.0234 Acc: 0.4651

Epoch 28/51
----------
train Loss: 0.0114 Acc: 0.7015
val Loss: 0.0219 Acc: 0.4558

Epoch 29/51
----------
train Loss: 0.0115 Acc: 0.6760
val Loss: 0.0239 Acc: 0.4651

Epoch 30/51
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0116 Acc: 0.6794
val Loss: 0.0229 Acc: 0.4558

Epoch 31/51
----------
train Loss: 0.0114 Acc: 0.6934
val Loss: 0.0226 Acc: 0.4605

Epoch 32/51
----------
train Loss: 0.0115 Acc: 0.6922
val Loss: 0.0229 Acc: 0.4605

Epoch 33/51
----------
train Loss: 0.0112 Acc: 0.7015
val Loss: 0.0224 Acc: 0.4651

Epoch 34/51
----------
train Loss: 0.0114 Acc: 0.6864
val Loss: 0.0221 Acc: 0.4465

Epoch 35/51
----------
train Loss: 0.0113 Acc: 0.6992
val Loss: 0.0252 Acc: 0.4605

Epoch 36/51
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0114 Acc: 0.6957
val Loss: 0.0216 Acc: 0.4558

Epoch 37/51
----------
train Loss: 0.0115 Acc: 0.7050
val Loss: 0.0251 Acc: 0.4605

Epoch 38/51
----------
train Loss: 0.0113 Acc: 0.7085
val Loss: 0.0254 Acc: 0.4651

Epoch 39/51
----------
train Loss: 0.0115 Acc: 0.6864
val Loss: 0.0254 Acc: 0.4605

Epoch 40/51
----------
train Loss: 0.0115 Acc: 0.6887
val Loss: 0.0238 Acc: 0.4605

Epoch 41/51
----------
train Loss: 0.0114 Acc: 0.6841
val Loss: 0.0235 Acc: 0.4605

Epoch 42/51
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0114 Acc: 0.6852
val Loss: 0.0221 Acc: 0.4605

Epoch 43/51
----------
train Loss: 0.0112 Acc: 0.7096
val Loss: 0.0248 Acc: 0.4558

Epoch 44/51
----------
train Loss: 0.0115 Acc: 0.6783
val Loss: 0.0256 Acc: 0.4558

Epoch 45/51
----------
train Loss: 0.0115 Acc: 0.6783
val Loss: 0.0224 Acc: 0.4605

Epoch 46/51
----------
train Loss: 0.0117 Acc: 0.6818
val Loss: 0.0225 Acc: 0.4558

Epoch 47/51
----------
train Loss: 0.0115 Acc: 0.6899
val Loss: 0.0230 Acc: 0.4558

Epoch 48/51
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0114 Acc: 0.6876
val Loss: 0.0236 Acc: 0.4512

Epoch 49/51
----------
train Loss: 0.0114 Acc: 0.7015
val Loss: 0.0233 Acc: 0.4605

Epoch 50/51
----------
train Loss: 0.0113 Acc: 0.6911
val Loss: 0.0244 Acc: 0.4558

Epoch 51/51
----------
train Loss: 0.0115 Acc: 0.6760
val Loss: 0.0230 Acc: 0.4465

Training complete in 4m 37s
Best val Acc: 0.479070

---Fine tuning.---
Epoch 0/51
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6307
val Loss: 0.0211 Acc: 0.4744

Epoch 1/51
----------
train Loss: 0.0074 Acc: 0.7979
val Loss: 0.0238 Acc: 0.5023

Epoch 2/51
----------
train Loss: 0.0043 Acc: 0.8873
val Loss: 0.0212 Acc: 0.5442

Epoch 3/51
----------
train Loss: 0.0026 Acc: 0.9408
val Loss: 0.0240 Acc: 0.5395

Epoch 4/51
----------
train Loss: 0.0017 Acc: 0.9617
val Loss: 0.0209 Acc: 0.5116

Epoch 5/51
----------
train Loss: 0.0013 Acc: 0.9698
val Loss: 0.0244 Acc: 0.5302

Epoch 6/51
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9756
val Loss: 0.0217 Acc: 0.5349

Epoch 7/51
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0210 Acc: 0.5349

Epoch 8/51
----------
train Loss: 0.0008 Acc: 0.9872
val Loss: 0.0231 Acc: 0.5302

Epoch 9/51
----------
train Loss: 0.0007 Acc: 0.9826
val Loss: 0.0231 Acc: 0.5302

Epoch 10/51
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0208 Acc: 0.5302

Epoch 11/51
----------
train Loss: 0.0007 Acc: 0.9837
val Loss: 0.0223 Acc: 0.5349

Epoch 12/51
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0194 Acc: 0.5395

Epoch 13/51
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0187 Acc: 0.5395

Epoch 14/51
----------
train Loss: 0.0006 Acc: 0.9872
val Loss: 0.0227 Acc: 0.5395

Epoch 15/51
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0187 Acc: 0.5395

Epoch 16/51
----------
train Loss: 0.0006 Acc: 0.9884
val Loss: 0.0186 Acc: 0.5395

Epoch 17/51
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0196 Acc: 0.5349

Epoch 18/51
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0200 Acc: 0.5349

Epoch 19/51
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0250 Acc: 0.5395

Epoch 20/51
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0230 Acc: 0.5442

Epoch 21/51
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0234 Acc: 0.5349

Epoch 22/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0208 Acc: 0.5442

Epoch 23/51
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0236 Acc: 0.5302

Epoch 24/51
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0198 Acc: 0.5349

Epoch 25/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0201 Acc: 0.5302

Epoch 26/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0208 Acc: 0.5302

Epoch 27/51
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0230 Acc: 0.5302

Epoch 28/51
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0208 Acc: 0.5349

Epoch 29/51
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0200 Acc: 0.5442

Epoch 30/51
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0239 Acc: 0.5442

Epoch 31/51
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0208 Acc: 0.5349

Epoch 32/51
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0218 Acc: 0.5349

Epoch 33/51
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0219 Acc: 0.5349

Epoch 34/51
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0196 Acc: 0.5349

Epoch 35/51
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0208 Acc: 0.5349

Epoch 36/51
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9872
val Loss: 0.0208 Acc: 0.5349

Epoch 37/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0190 Acc: 0.5395

Epoch 38/51
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0229 Acc: 0.5349

Epoch 39/51
----------
train Loss: 0.0007 Acc: 0.9849
val Loss: 0.0216 Acc: 0.5442

Epoch 40/51
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0240 Acc: 0.5395

Epoch 41/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0215 Acc: 0.5349

Epoch 42/51
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0213 Acc: 0.5302

Epoch 43/51
----------
train Loss: 0.0007 Acc: 0.9849
val Loss: 0.0212 Acc: 0.5349

Epoch 44/51
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0253 Acc: 0.5395

Epoch 45/51
----------
train Loss: 0.0006 Acc: 0.9895
val Loss: 0.0230 Acc: 0.5395

Epoch 46/51
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0201 Acc: 0.5488

Epoch 47/51
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0201 Acc: 0.5442

Epoch 48/51
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9884
val Loss: 0.0246 Acc: 0.5349

Epoch 49/51
----------
train Loss: 0.0006 Acc: 0.9872
val Loss: 0.0204 Acc: 0.5302

Epoch 50/51
----------
train Loss: 0.0007 Acc: 0.9826
val Loss: 0.0221 Acc: 0.5395

Epoch 51/51
----------
train Loss: 0.0006 Acc: 0.9861
val Loss: 0.0212 Acc: 0.5442

Training complete in 4m 54s
Best val Acc: 0.548837

---Testing---
Test accuracy: 0.898699
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 79 %
Accuracy of Bigeye tuna : 86 %
Accuracy of Blackfin tuna : 95 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 88 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 78 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8693820233745305, std: 0.08485372935414025
--------------------

run info[val: 0.25, epoch: 69, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0285 Acc: 0.1945
val Loss: 0.0270 Acc: 0.2119

Epoch 1/68
----------
train Loss: 0.0251 Acc: 0.2850
val Loss: 0.0231 Acc: 0.3346

Epoch 2/68
----------
train Loss: 0.0209 Acc: 0.3730
val Loss: 0.0238 Acc: 0.3569

Epoch 3/68
----------
train Loss: 0.0204 Acc: 0.4250
val Loss: 0.0248 Acc: 0.3532

Epoch 4/68
----------
LR is set to 0.001
train Loss: 0.0193 Acc: 0.4771
val Loss: 0.0216 Acc: 0.4164

Epoch 5/68
----------
train Loss: 0.0174 Acc: 0.5217
val Loss: 0.0210 Acc: 0.3941

Epoch 6/68
----------
train Loss: 0.0166 Acc: 0.5167
val Loss: 0.0198 Acc: 0.4349

Epoch 7/68
----------
train Loss: 0.0162 Acc: 0.5489
val Loss: 0.0201 Acc: 0.4201

Epoch 8/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0161 Acc: 0.5576
val Loss: 0.0199 Acc: 0.4275

Epoch 9/68
----------
train Loss: 0.0157 Acc: 0.5762
val Loss: 0.0198 Acc: 0.4238

Epoch 10/68
----------
train Loss: 0.0161 Acc: 0.5713
val Loss: 0.0196 Acc: 0.4164

Epoch 11/68
----------
train Loss: 0.0157 Acc: 0.5861
val Loss: 0.0198 Acc: 0.4275

Epoch 12/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0158 Acc: 0.5651
val Loss: 0.0196 Acc: 0.4201

Epoch 13/68
----------
train Loss: 0.0162 Acc: 0.5651
val Loss: 0.0199 Acc: 0.4238

Epoch 14/68
----------
train Loss: 0.0166 Acc: 0.5626
val Loss: 0.0199 Acc: 0.4238

Epoch 15/68
----------
train Loss: 0.0160 Acc: 0.5762
val Loss: 0.0198 Acc: 0.4201

Epoch 16/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0159 Acc: 0.5675
val Loss: 0.0199 Acc: 0.4275

Epoch 17/68
----------
train Loss: 0.0162 Acc: 0.5787
val Loss: 0.0198 Acc: 0.4275

Epoch 18/68
----------
train Loss: 0.0164 Acc: 0.5874
val Loss: 0.0198 Acc: 0.4164

Epoch 19/68
----------
train Loss: 0.0159 Acc: 0.5675
val Loss: 0.0196 Acc: 0.4275

Epoch 20/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0161 Acc: 0.5923
val Loss: 0.0198 Acc: 0.4275

Epoch 21/68
----------
train Loss: 0.0157 Acc: 0.5973
val Loss: 0.0197 Acc: 0.4238

Epoch 22/68
----------
train Loss: 0.0154 Acc: 0.5725
val Loss: 0.0197 Acc: 0.4201

Epoch 23/68
----------
train Loss: 0.0169 Acc: 0.5936
val Loss: 0.0197 Acc: 0.4238

Epoch 24/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0155 Acc: 0.5663
val Loss: 0.0199 Acc: 0.4201

Epoch 25/68
----------
train Loss: 0.0162 Acc: 0.5737
val Loss: 0.0201 Acc: 0.4275

Epoch 26/68
----------
train Loss: 0.0162 Acc: 0.5824
val Loss: 0.0197 Acc: 0.4275

Epoch 27/68
----------
train Loss: 0.0167 Acc: 0.5713
val Loss: 0.0200 Acc: 0.4201

Epoch 28/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0163 Acc: 0.5787
val Loss: 0.0200 Acc: 0.4201

Epoch 29/68
----------
train Loss: 0.0157 Acc: 0.5985
val Loss: 0.0197 Acc: 0.4201

Epoch 30/68
----------
train Loss: 0.0157 Acc: 0.5613
val Loss: 0.0197 Acc: 0.4349

Epoch 31/68
----------
train Loss: 0.0157 Acc: 0.5663
val Loss: 0.0197 Acc: 0.4238

Epoch 32/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0160 Acc: 0.5874
val Loss: 0.0198 Acc: 0.4312

Epoch 33/68
----------
train Loss: 0.0166 Acc: 0.5638
val Loss: 0.0197 Acc: 0.4275

Epoch 34/68
----------
train Loss: 0.0158 Acc: 0.5836
val Loss: 0.0198 Acc: 0.4349

Epoch 35/68
----------
train Loss: 0.0160 Acc: 0.5539
val Loss: 0.0199 Acc: 0.4201

Epoch 36/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0159 Acc: 0.5799
val Loss: 0.0199 Acc: 0.4164

Epoch 37/68
----------
train Loss: 0.0171 Acc: 0.5774
val Loss: 0.0200 Acc: 0.4126

Epoch 38/68
----------
train Loss: 0.0156 Acc: 0.5725
val Loss: 0.0199 Acc: 0.4201

Epoch 39/68
----------
train Loss: 0.0165 Acc: 0.5799
val Loss: 0.0198 Acc: 0.4275

Epoch 40/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0169 Acc: 0.5762
val Loss: 0.0201 Acc: 0.4349

Epoch 41/68
----------
train Loss: 0.0153 Acc: 0.5713
val Loss: 0.0198 Acc: 0.4126

Epoch 42/68
----------
train Loss: 0.0165 Acc: 0.5613
val Loss: 0.0199 Acc: 0.4349

Epoch 43/68
----------
train Loss: 0.0158 Acc: 0.5898
val Loss: 0.0199 Acc: 0.4164

Epoch 44/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0163 Acc: 0.5663
val Loss: 0.0199 Acc: 0.4164

Epoch 45/68
----------
train Loss: 0.0155 Acc: 0.5750
val Loss: 0.0196 Acc: 0.4201

Epoch 46/68
----------
train Loss: 0.0154 Acc: 0.5787
val Loss: 0.0199 Acc: 0.4275

Epoch 47/68
----------
train Loss: 0.0160 Acc: 0.5663
val Loss: 0.0199 Acc: 0.4238

Epoch 48/68
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0159 Acc: 0.5774
val Loss: 0.0200 Acc: 0.4201

Epoch 49/68
----------
train Loss: 0.0163 Acc: 0.5564
val Loss: 0.0197 Acc: 0.4238

Epoch 50/68
----------
train Loss: 0.0160 Acc: 0.5774
val Loss: 0.0197 Acc: 0.4238

Epoch 51/68
----------
train Loss: 0.0165 Acc: 0.5527
val Loss: 0.0197 Acc: 0.4238

Epoch 52/68
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0156 Acc: 0.5626
val Loss: 0.0198 Acc: 0.4164

Epoch 53/68
----------
train Loss: 0.0161 Acc: 0.5626
val Loss: 0.0199 Acc: 0.4238

Epoch 54/68
----------
train Loss: 0.0157 Acc: 0.5737
val Loss: 0.0198 Acc: 0.4275

Epoch 55/68
----------
train Loss: 0.0160 Acc: 0.5713
val Loss: 0.0197 Acc: 0.4238

Epoch 56/68
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0160 Acc: 0.5861
val Loss: 0.0198 Acc: 0.4238

Epoch 57/68
----------
train Loss: 0.0157 Acc: 0.5849
val Loss: 0.0196 Acc: 0.4164

Epoch 58/68
----------
train Loss: 0.0165 Acc: 0.5638
val Loss: 0.0196 Acc: 0.4312

Epoch 59/68
----------
train Loss: 0.0154 Acc: 0.5725
val Loss: 0.0197 Acc: 0.4201

Epoch 60/68
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0160 Acc: 0.5576
val Loss: 0.0200 Acc: 0.4238

Epoch 61/68
----------
train Loss: 0.0166 Acc: 0.5576
val Loss: 0.0194 Acc: 0.4238

Epoch 62/68
----------
train Loss: 0.0158 Acc: 0.5874
val Loss: 0.0196 Acc: 0.4238

Epoch 63/68
----------
train Loss: 0.0160 Acc: 0.5601
val Loss: 0.0198 Acc: 0.4238

Epoch 64/68
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0166 Acc: 0.5675
val Loss: 0.0196 Acc: 0.4164

Epoch 65/68
----------
train Loss: 0.0166 Acc: 0.5725
val Loss: 0.0200 Acc: 0.4126

Epoch 66/68
----------
train Loss: 0.0162 Acc: 0.5551
val Loss: 0.0197 Acc: 0.4201

Epoch 67/68
----------
train Loss: 0.0163 Acc: 0.5700
val Loss: 0.0197 Acc: 0.4312

Epoch 68/68
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0168 Acc: 0.5737
val Loss: 0.0199 Acc: 0.4238

Training complete in 6m 16s
Best val Acc: 0.434944

---Fine tuning.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0166 Acc: 0.5353
val Loss: 0.0185 Acc: 0.4721

Epoch 1/68
----------
train Loss: 0.0129 Acc: 0.6332
val Loss: 0.0213 Acc: 0.4498

Epoch 2/68
----------
train Loss: 0.0105 Acc: 0.7076
val Loss: 0.0203 Acc: 0.4833

Epoch 3/68
----------
train Loss: 0.0097 Acc: 0.7286
val Loss: 0.0209 Acc: 0.4424

Epoch 4/68
----------
LR is set to 0.001
train Loss: 0.0077 Acc: 0.8141
val Loss: 0.0170 Acc: 0.5242

Epoch 5/68
----------
train Loss: 0.0064 Acc: 0.8451
val Loss: 0.0160 Acc: 0.5874

Epoch 6/68
----------
train Loss: 0.0056 Acc: 0.8649
val Loss: 0.0157 Acc: 0.6059

Epoch 7/68
----------
train Loss: 0.0052 Acc: 0.8910
val Loss: 0.0157 Acc: 0.6022

Epoch 8/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0055 Acc: 0.8984
val Loss: 0.0158 Acc: 0.6022

Epoch 9/68
----------
train Loss: 0.0047 Acc: 0.9021
val Loss: 0.0156 Acc: 0.5985

Epoch 10/68
----------
train Loss: 0.0044 Acc: 0.9058
val Loss: 0.0155 Acc: 0.5911

Epoch 11/68
----------
train Loss: 0.0048 Acc: 0.9033
val Loss: 0.0156 Acc: 0.5874

Epoch 12/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0055 Acc: 0.9157
val Loss: 0.0156 Acc: 0.6022

Epoch 13/68
----------
train Loss: 0.0043 Acc: 0.9083
val Loss: 0.0154 Acc: 0.5874

Epoch 14/68
----------
train Loss: 0.0042 Acc: 0.9219
val Loss: 0.0156 Acc: 0.5948

Epoch 15/68
----------
train Loss: 0.0055 Acc: 0.9095
val Loss: 0.0152 Acc: 0.6059

Epoch 16/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0052 Acc: 0.9157
val Loss: 0.0154 Acc: 0.6022

Epoch 17/68
----------
train Loss: 0.0041 Acc: 0.9120
val Loss: 0.0158 Acc: 0.5948

Epoch 18/68
----------
train Loss: 0.0049 Acc: 0.9083
val Loss: 0.0157 Acc: 0.5911

Epoch 19/68
----------
train Loss: 0.0045 Acc: 0.9083
val Loss: 0.0157 Acc: 0.5762

Epoch 20/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0049 Acc: 0.8996
val Loss: 0.0159 Acc: 0.5874

Epoch 21/68
----------
train Loss: 0.0049 Acc: 0.9108
val Loss: 0.0158 Acc: 0.5948

Epoch 22/68
----------
train Loss: 0.0043 Acc: 0.9046
val Loss: 0.0154 Acc: 0.5948

Epoch 23/68
----------
train Loss: 0.0044 Acc: 0.8947
val Loss: 0.0151 Acc: 0.5911

Epoch 24/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0043 Acc: 0.9182
val Loss: 0.0155 Acc: 0.5911

Epoch 25/68
----------
train Loss: 0.0043 Acc: 0.9046
val Loss: 0.0156 Acc: 0.5911

Epoch 26/68
----------
train Loss: 0.0053 Acc: 0.9120
val Loss: 0.0153 Acc: 0.5911

Epoch 27/68
----------
train Loss: 0.0046 Acc: 0.9120
val Loss: 0.0154 Acc: 0.5799

Epoch 28/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0045 Acc: 0.9046
val Loss: 0.0155 Acc: 0.5874

Epoch 29/68
----------
train Loss: 0.0042 Acc: 0.9244
val Loss: 0.0153 Acc: 0.5911

Epoch 30/68
----------
train Loss: 0.0050 Acc: 0.9071
val Loss: 0.0153 Acc: 0.5985

Epoch 31/68
----------
train Loss: 0.0047 Acc: 0.9108
val Loss: 0.0155 Acc: 0.5985

Epoch 32/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0051 Acc: 0.9009
val Loss: 0.0157 Acc: 0.5948

Epoch 33/68
----------
train Loss: 0.0045 Acc: 0.9021
val Loss: 0.0155 Acc: 0.5874

Epoch 34/68
----------
train Loss: 0.0048 Acc: 0.9170
val Loss: 0.0158 Acc: 0.5836

Epoch 35/68
----------
train Loss: 0.0045 Acc: 0.9058
val Loss: 0.0156 Acc: 0.5911

Epoch 36/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0041 Acc: 0.9170
val Loss: 0.0153 Acc: 0.5911

Epoch 37/68
----------
train Loss: 0.0042 Acc: 0.9170
val Loss: 0.0155 Acc: 0.5948

Epoch 38/68
----------
train Loss: 0.0044 Acc: 0.9108
val Loss: 0.0155 Acc: 0.5836

Epoch 39/68
----------
train Loss: 0.0039 Acc: 0.9046
val Loss: 0.0156 Acc: 0.5874

Epoch 40/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0052 Acc: 0.9009
val Loss: 0.0154 Acc: 0.5985

Epoch 41/68
----------
train Loss: 0.0044 Acc: 0.9058
val Loss: 0.0153 Acc: 0.5874

Epoch 42/68
----------
train Loss: 0.0040 Acc: 0.9058
val Loss: 0.0151 Acc: 0.5911

Epoch 43/68
----------
train Loss: 0.0046 Acc: 0.9046
val Loss: 0.0153 Acc: 0.5948

Epoch 44/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0043 Acc: 0.9071
val Loss: 0.0154 Acc: 0.5874

Epoch 45/68
----------
train Loss: 0.0045 Acc: 0.9058
val Loss: 0.0157 Acc: 0.5985

Epoch 46/68
----------
train Loss: 0.0042 Acc: 0.9207
val Loss: 0.0157 Acc: 0.6022

Epoch 47/68
----------
train Loss: 0.0044 Acc: 0.9108
val Loss: 0.0155 Acc: 0.5911

Epoch 48/68
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0043 Acc: 0.9195
val Loss: 0.0158 Acc: 0.5874

Epoch 49/68
----------
train Loss: 0.0046 Acc: 0.9133
val Loss: 0.0150 Acc: 0.5799

Epoch 50/68
----------
train Loss: 0.0057 Acc: 0.9071
val Loss: 0.0155 Acc: 0.5911

Epoch 51/68
----------
train Loss: 0.0048 Acc: 0.9133
val Loss: 0.0152 Acc: 0.5836

Epoch 52/68
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0044 Acc: 0.9021
val Loss: 0.0153 Acc: 0.5948

Epoch 53/68
----------
train Loss: 0.0045 Acc: 0.9095
val Loss: 0.0158 Acc: 0.5836

Epoch 54/68
----------
train Loss: 0.0044 Acc: 0.9207
val Loss: 0.0161 Acc: 0.6022

Epoch 55/68
----------
train Loss: 0.0044 Acc: 0.9133
val Loss: 0.0155 Acc: 0.6022

Epoch 56/68
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0048 Acc: 0.9108
val Loss: 0.0155 Acc: 0.5874

Epoch 57/68
----------
train Loss: 0.0052 Acc: 0.9046
val Loss: 0.0155 Acc: 0.5948

Epoch 58/68
----------
train Loss: 0.0048 Acc: 0.9009
val Loss: 0.0158 Acc: 0.5948

Epoch 59/68
----------
train Loss: 0.0049 Acc: 0.9133
val Loss: 0.0152 Acc: 0.5799

Epoch 60/68
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0040 Acc: 0.9157
val Loss: 0.0156 Acc: 0.5911

Epoch 61/68
----------
train Loss: 0.0044 Acc: 0.9095
val Loss: 0.0157 Acc: 0.5985

Epoch 62/68
----------
train Loss: 0.0040 Acc: 0.9108
val Loss: 0.0156 Acc: 0.5911

Epoch 63/68
----------
train Loss: 0.0044 Acc: 0.9120
val Loss: 0.0158 Acc: 0.5911

Epoch 64/68
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0051 Acc: 0.9033
val Loss: 0.0153 Acc: 0.5948

Epoch 65/68
----------
train Loss: 0.0045 Acc: 0.9046
val Loss: 0.0158 Acc: 0.5948

Epoch 66/68
----------
train Loss: 0.0047 Acc: 0.9095
val Loss: 0.0157 Acc: 0.5911

Epoch 67/68
----------
train Loss: 0.0043 Acc: 0.9157
val Loss: 0.0153 Acc: 0.5948

Epoch 68/68
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0046 Acc: 0.9120
val Loss: 0.0154 Acc: 0.5911

Training complete in 6m 42s
Best val Acc: 0.605948

---Testing---
Test accuracy: 0.842937
--------------------
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 68 %
Accuracy of Blackfin tuna : 95 %
Accuracy of Bullet tuna : 96 %
Accuracy of Frigate tuna : 34 %
Accuracy of Little tunny : 88 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 68 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 21 %
Accuracy of Southern bluefin tuna : 63 %
Accuracy of Yellowfin tuna : 94 %
mean: 0.7613267468834745, std: 0.22550725353514503
--------------------

run info[val: 0.3, epoch: 67, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/66
----------
LR is set to 0.01
train Loss: 0.0288 Acc: 0.1260
val Loss: 0.0302 Acc: 0.2236

Epoch 1/66
----------
train Loss: 0.0234 Acc: 0.2785
val Loss: 0.0277 Acc: 0.3261

Epoch 2/66
----------
train Loss: 0.0194 Acc: 0.4297
val Loss: 0.0258 Acc: 0.3696

Epoch 3/66
----------
train Loss: 0.0173 Acc: 0.4894
val Loss: 0.0243 Acc: 0.3944

Epoch 4/66
----------
train Loss: 0.0156 Acc: 0.5398
val Loss: 0.0231 Acc: 0.3447

Epoch 5/66
----------
train Loss: 0.0140 Acc: 0.5875
val Loss: 0.0210 Acc: 0.4224

Epoch 6/66
----------
train Loss: 0.0127 Acc: 0.6592
val Loss: 0.0218 Acc: 0.4503

Epoch 7/66
----------
train Loss: 0.0120 Acc: 0.6684
val Loss: 0.0215 Acc: 0.4472

Epoch 8/66
----------
train Loss: 0.0113 Acc: 0.6910
val Loss: 0.0210 Acc: 0.4286

Epoch 9/66
----------
LR is set to 0.001
train Loss: 0.0108 Acc: 0.6976
val Loss: 0.0224 Acc: 0.4441

Epoch 10/66
----------
train Loss: 0.0104 Acc: 0.7294
val Loss: 0.0205 Acc: 0.4627

Epoch 11/66
----------
train Loss: 0.0102 Acc: 0.7467
val Loss: 0.0213 Acc: 0.4845

Epoch 12/66
----------
train Loss: 0.0102 Acc: 0.7546
val Loss: 0.0205 Acc: 0.4876

Epoch 13/66
----------
train Loss: 0.0101 Acc: 0.7480
val Loss: 0.0216 Acc: 0.4720

Epoch 14/66
----------
train Loss: 0.0101 Acc: 0.7493
val Loss: 0.0196 Acc: 0.4720

Epoch 15/66
----------
train Loss: 0.0101 Acc: 0.7427
val Loss: 0.0206 Acc: 0.4907

Epoch 16/66
----------
train Loss: 0.0098 Acc: 0.7401
val Loss: 0.0201 Acc: 0.4814

Epoch 17/66
----------
train Loss: 0.0099 Acc: 0.7613
val Loss: 0.0206 Acc: 0.4720

Epoch 18/66
----------
LR is set to 0.00010000000000000002
train Loss: 0.0097 Acc: 0.7560
val Loss: 0.0202 Acc: 0.4720

Epoch 19/66
----------
train Loss: 0.0097 Acc: 0.7533
val Loss: 0.0215 Acc: 0.4720

Epoch 20/66
----------
train Loss: 0.0096 Acc: 0.7785
val Loss: 0.0204 Acc: 0.4814

Epoch 21/66
----------
train Loss: 0.0098 Acc: 0.7480
val Loss: 0.0211 Acc: 0.4783

Epoch 22/66
----------
train Loss: 0.0098 Acc: 0.7639
val Loss: 0.0209 Acc: 0.4658

Epoch 23/66
----------
train Loss: 0.0097 Acc: 0.7401
val Loss: 0.0206 Acc: 0.4752

Epoch 24/66
----------
train Loss: 0.0098 Acc: 0.7454
val Loss: 0.0214 Acc: 0.4689

Epoch 25/66
----------
train Loss: 0.0097 Acc: 0.7666
val Loss: 0.0212 Acc: 0.4689

Epoch 26/66
----------
train Loss: 0.0097 Acc: 0.7772
val Loss: 0.0212 Acc: 0.4752

Epoch 27/66
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0099 Acc: 0.7586
val Loss: 0.0201 Acc: 0.4783

Epoch 28/66
----------
train Loss: 0.0099 Acc: 0.7546
val Loss: 0.0203 Acc: 0.4689

Epoch 29/66
----------
train Loss: 0.0096 Acc: 0.7573
val Loss: 0.0198 Acc: 0.4658

Epoch 30/66
----------
train Loss: 0.0098 Acc: 0.7586
val Loss: 0.0201 Acc: 0.4783

Epoch 31/66
----------
train Loss: 0.0097 Acc: 0.7493
val Loss: 0.0218 Acc: 0.4752

Epoch 32/66
----------
train Loss: 0.0098 Acc: 0.7586
val Loss: 0.0209 Acc: 0.4689

Epoch 33/66
----------
train Loss: 0.0100 Acc: 0.7480
val Loss: 0.0207 Acc: 0.4689

Epoch 34/66
----------
train Loss: 0.0096 Acc: 0.7599
val Loss: 0.0210 Acc: 0.4689

Epoch 35/66
----------
train Loss: 0.0097 Acc: 0.7626
val Loss: 0.0215 Acc: 0.4689

Epoch 36/66
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0099 Acc: 0.7573
val Loss: 0.0213 Acc: 0.4720

Epoch 37/66
----------
train Loss: 0.0097 Acc: 0.7692
val Loss: 0.0219 Acc: 0.4783

Epoch 38/66
----------
train Loss: 0.0098 Acc: 0.7666
val Loss: 0.0205 Acc: 0.4752

Epoch 39/66
----------
train Loss: 0.0098 Acc: 0.7427
val Loss: 0.0204 Acc: 0.4752

Epoch 40/66
----------
train Loss: 0.0097 Acc: 0.7599
val Loss: 0.0198 Acc: 0.4627

Epoch 41/66
----------
train Loss: 0.0099 Acc: 0.7573
val Loss: 0.0202 Acc: 0.4720

Epoch 42/66
----------
train Loss: 0.0097 Acc: 0.7653
val Loss: 0.0196 Acc: 0.4783

Epoch 43/66
----------
train Loss: 0.0097 Acc: 0.7759
val Loss: 0.0205 Acc: 0.4689

Epoch 44/66
----------
train Loss: 0.0097 Acc: 0.7560
val Loss: 0.0219 Acc: 0.4720

Epoch 45/66
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0099 Acc: 0.7533
val Loss: 0.0198 Acc: 0.4752

Epoch 46/66
----------
train Loss: 0.0099 Acc: 0.7414
val Loss: 0.0206 Acc: 0.4845

Epoch 47/66
----------
train Loss: 0.0097 Acc: 0.7639
val Loss: 0.0211 Acc: 0.4752

Epoch 48/66
----------
train Loss: 0.0100 Acc: 0.7414
val Loss: 0.0212 Acc: 0.4720

Epoch 49/66
----------
train Loss: 0.0097 Acc: 0.7613
val Loss: 0.0209 Acc: 0.4689

Epoch 50/66
----------
train Loss: 0.0098 Acc: 0.7586
val Loss: 0.0199 Acc: 0.4783

Epoch 51/66
----------
train Loss: 0.0097 Acc: 0.7573
val Loss: 0.0208 Acc: 0.4752

Epoch 52/66
----------
train Loss: 0.0097 Acc: 0.7573
val Loss: 0.0220 Acc: 0.4752

Epoch 53/66
----------
train Loss: 0.0098 Acc: 0.7639
val Loss: 0.0200 Acc: 0.4814

Epoch 54/66
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0095 Acc: 0.7666
val Loss: 0.0206 Acc: 0.4752

Epoch 55/66
----------
train Loss: 0.0096 Acc: 0.7626
val Loss: 0.0203 Acc: 0.4752

Epoch 56/66
----------
train Loss: 0.0098 Acc: 0.7560
val Loss: 0.0209 Acc: 0.4720

Epoch 57/66
----------
train Loss: 0.0098 Acc: 0.7480
val Loss: 0.0206 Acc: 0.4752

Epoch 58/66
----------
train Loss: 0.0097 Acc: 0.7520
val Loss: 0.0203 Acc: 0.4689

Epoch 59/66
----------
train Loss: 0.0095 Acc: 0.7719
val Loss: 0.0206 Acc: 0.4689

Epoch 60/66
----------
train Loss: 0.0097 Acc: 0.7507
val Loss: 0.0207 Acc: 0.4752

Epoch 61/66
----------
train Loss: 0.0099 Acc: 0.7613
val Loss: 0.0214 Acc: 0.4783

Epoch 62/66
----------
train Loss: 0.0099 Acc: 0.7706
val Loss: 0.0209 Acc: 0.4752

Epoch 63/66
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0099 Acc: 0.7666
val Loss: 0.0211 Acc: 0.4783

Epoch 64/66
----------
train Loss: 0.0096 Acc: 0.7639
val Loss: 0.0203 Acc: 0.4720

Epoch 65/66
----------
train Loss: 0.0099 Acc: 0.7427
val Loss: 0.0213 Acc: 0.4720

Epoch 66/66
----------
train Loss: 0.0099 Acc: 0.7573
val Loss: 0.0201 Acc: 0.4783

Training complete in 6m 9s
Best val Acc: 0.490683

---Fine tuning.---
Epoch 0/66
----------
LR is set to 0.01
train Loss: 0.0103 Acc: 0.7082
val Loss: 0.0216 Acc: 0.4410

Epoch 1/66
----------
train Loss: 0.0062 Acc: 0.8462
val Loss: 0.0202 Acc: 0.4814

Epoch 2/66
----------
train Loss: 0.0035 Acc: 0.9231
val Loss: 0.0214 Acc: 0.4658

Epoch 3/66
----------
train Loss: 0.0019 Acc: 0.9642
val Loss: 0.0199 Acc: 0.5217

Epoch 4/66
----------
train Loss: 0.0012 Acc: 0.9828
val Loss: 0.0202 Acc: 0.4969

Epoch 5/66
----------
train Loss: 0.0010 Acc: 0.9775
val Loss: 0.0197 Acc: 0.5217

Epoch 6/66
----------
train Loss: 0.0007 Acc: 0.9881
val Loss: 0.0202 Acc: 0.5217

Epoch 7/66
----------
train Loss: 0.0007 Acc: 0.9854
val Loss: 0.0202 Acc: 0.5186

Epoch 8/66
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0203 Acc: 0.5248

Epoch 9/66
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0196 Acc: 0.5280

Epoch 10/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0200 Acc: 0.5280

Epoch 11/66
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0201 Acc: 0.5280

Epoch 12/66
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0203 Acc: 0.5311

Epoch 13/66
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0189 Acc: 0.5342

Epoch 14/66
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0210 Acc: 0.5311

Epoch 15/66
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0210 Acc: 0.5311

Epoch 16/66
----------
train Loss: 0.0003 Acc: 0.9960
val Loss: 0.0212 Acc: 0.5373

Epoch 17/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0200 Acc: 0.5373

Epoch 18/66
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0189 Acc: 0.5342

Epoch 19/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0221 Acc: 0.5311

Epoch 20/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0202 Acc: 0.5311

Epoch 21/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0199 Acc: 0.5280

Epoch 22/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0203 Acc: 0.5311

Epoch 23/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0206 Acc: 0.5280

Epoch 24/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0199 Acc: 0.5311

Epoch 25/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0194 Acc: 0.5311

Epoch 26/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0203 Acc: 0.5342

Epoch 27/66
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0205 Acc: 0.5311

Epoch 28/66
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0203 Acc: 0.5280

Epoch 29/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0202 Acc: 0.5311

Epoch 30/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0213 Acc: 0.5342

Epoch 31/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0201 Acc: 0.5311

Epoch 32/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0217 Acc: 0.5373

Epoch 33/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0204 Acc: 0.5311

Epoch 34/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0207 Acc: 0.5311

Epoch 35/66
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0214 Acc: 0.5311

Epoch 36/66
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0205 Acc: 0.5373

Epoch 37/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0227 Acc: 0.5404

Epoch 38/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0217 Acc: 0.5435

Epoch 39/66
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0211 Acc: 0.5373

Epoch 40/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0230 Acc: 0.5404

Epoch 41/66
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0211 Acc: 0.5342

Epoch 42/66
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0223 Acc: 0.5280

Epoch 43/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0203 Acc: 0.5311

Epoch 44/66
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0218 Acc: 0.5280

Epoch 45/66
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0227 Acc: 0.5373

Epoch 46/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0197 Acc: 0.5311

Epoch 47/66
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0207 Acc: 0.5373

Epoch 48/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0190 Acc: 0.5373

Epoch 49/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0197 Acc: 0.5342

Epoch 50/66
----------
train Loss: 0.0002 Acc: 0.9960
val Loss: 0.0205 Acc: 0.5404

Epoch 51/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0202 Acc: 0.5404

Epoch 52/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0198 Acc: 0.5373

Epoch 53/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0208 Acc: 0.5373

Epoch 54/66
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0218 Acc: 0.5373

Epoch 55/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0220 Acc: 0.5311

Epoch 56/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0215 Acc: 0.5280

Epoch 57/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0186 Acc: 0.5280

Epoch 58/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0218 Acc: 0.5311

Epoch 59/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0207 Acc: 0.5280

Epoch 60/66
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0198 Acc: 0.5248

Epoch 61/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0204 Acc: 0.5311

Epoch 62/66
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0204 Acc: 0.5373

Epoch 63/66
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0205 Acc: 0.5280

Epoch 64/66
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0209 Acc: 0.5280

Epoch 65/66
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0200 Acc: 0.5280

Epoch 66/66
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0200 Acc: 0.5248

Training complete in 6m 34s
Best val Acc: 0.543478

---Testing---
Test accuracy: 0.857807
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 75 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 81 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 80 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 76 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.8095063603354029, std: 0.1279690284491019

Model saved in "./weights/tuna_fish_[0.94]_mean[0.92]_std[0.06].save".
--------------------

run info[val: 0.1, epoch: 69, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0261 Acc: 0.1734
val Loss: 0.0405 Acc: 0.3084

Epoch 1/68
----------
train Loss: 0.0211 Acc: 0.3457
val Loss: 0.0281 Acc: 0.4673

Epoch 2/68
----------
train Loss: 0.0177 Acc: 0.4644
val Loss: 0.0315 Acc: 0.4299

Epoch 3/68
----------
train Loss: 0.0157 Acc: 0.5057
val Loss: 0.0323 Acc: 0.4860

Epoch 4/68
----------
train Loss: 0.0140 Acc: 0.5851
val Loss: 0.0227 Acc: 0.5140

Epoch 5/68
----------
train Loss: 0.0136 Acc: 0.5965
val Loss: 0.0266 Acc: 0.4673

Epoch 6/68
----------
LR is set to 0.001
train Loss: 0.0125 Acc: 0.6244
val Loss: 0.0252 Acc: 0.5234

Epoch 7/68
----------
train Loss: 0.0118 Acc: 0.6625
val Loss: 0.0328 Acc: 0.5140

Epoch 8/68
----------
train Loss: 0.0116 Acc: 0.6594
val Loss: 0.0280 Acc: 0.5047

Epoch 9/68
----------
train Loss: 0.0117 Acc: 0.6780
val Loss: 0.0291 Acc: 0.5421

Epoch 10/68
----------
train Loss: 0.0116 Acc: 0.6863
val Loss: 0.0370 Acc: 0.5140

Epoch 11/68
----------
train Loss: 0.0116 Acc: 0.6718
val Loss: 0.0370 Acc: 0.5140

Epoch 12/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0114 Acc: 0.6821
val Loss: 0.0304 Acc: 0.5140

Epoch 13/68
----------
train Loss: 0.0113 Acc: 0.6832
val Loss: 0.0314 Acc: 0.5234

Epoch 14/68
----------
train Loss: 0.0115 Acc: 0.6883
val Loss: 0.0310 Acc: 0.5047

Epoch 15/68
----------
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0320 Acc: 0.5140

Epoch 16/68
----------
train Loss: 0.0115 Acc: 0.6873
val Loss: 0.0293 Acc: 0.5140

Epoch 17/68
----------
train Loss: 0.0115 Acc: 0.6687
val Loss: 0.0258 Acc: 0.5140

Epoch 18/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0116 Acc: 0.6708
val Loss: 0.0295 Acc: 0.5234

Epoch 19/68
----------
train Loss: 0.0114 Acc: 0.6966
val Loss: 0.0345 Acc: 0.5047

Epoch 20/68
----------
train Loss: 0.0115 Acc: 0.6801
val Loss: 0.0244 Acc: 0.5140

Epoch 21/68
----------
train Loss: 0.0113 Acc: 0.6760
val Loss: 0.0253 Acc: 0.5140

Epoch 22/68
----------
train Loss: 0.0112 Acc: 0.6987
val Loss: 0.0298 Acc: 0.5047

Epoch 23/68
----------
train Loss: 0.0114 Acc: 0.6811
val Loss: 0.0326 Acc: 0.5047

Epoch 24/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0288 Acc: 0.5047

Epoch 25/68
----------
train Loss: 0.0114 Acc: 0.6780
val Loss: 0.0299 Acc: 0.5047

Epoch 26/68
----------
train Loss: 0.0114 Acc: 0.6811
val Loss: 0.0249 Acc: 0.5047

Epoch 27/68
----------
train Loss: 0.0114 Acc: 0.6770
val Loss: 0.0337 Acc: 0.5047

Epoch 28/68
----------
train Loss: 0.0115 Acc: 0.6852
val Loss: 0.0256 Acc: 0.5047

Epoch 29/68
----------
train Loss: 0.0114 Acc: 0.6914
val Loss: 0.0282 Acc: 0.5140

Epoch 30/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0113 Acc: 0.6770
val Loss: 0.0297 Acc: 0.5047

Epoch 31/68
----------
train Loss: 0.0115 Acc: 0.6739
val Loss: 0.0282 Acc: 0.5140

Epoch 32/68
----------
train Loss: 0.0113 Acc: 0.7018
val Loss: 0.0238 Acc: 0.5047

Epoch 33/68
----------
train Loss: 0.0112 Acc: 0.6997
val Loss: 0.0286 Acc: 0.5047

Epoch 34/68
----------
train Loss: 0.0112 Acc: 0.6894
val Loss: 0.0276 Acc: 0.5140

Epoch 35/68
----------
train Loss: 0.0115 Acc: 0.6821
val Loss: 0.0304 Acc: 0.5140

Epoch 36/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0114 Acc: 0.6852
val Loss: 0.0295 Acc: 0.5047

Epoch 37/68
----------
train Loss: 0.0114 Acc: 0.6873
val Loss: 0.0345 Acc: 0.5047

Epoch 38/68
----------
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0267 Acc: 0.5140

Epoch 39/68
----------
train Loss: 0.0115 Acc: 0.6832
val Loss: 0.0338 Acc: 0.5140

Epoch 40/68
----------
train Loss: 0.0114 Acc: 0.6801
val Loss: 0.0279 Acc: 0.5047

Epoch 41/68
----------
train Loss: 0.0113 Acc: 0.6987
val Loss: 0.0288 Acc: 0.5047

Epoch 42/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0114 Acc: 0.7028
val Loss: 0.0303 Acc: 0.5047

Epoch 43/68
----------
train Loss: 0.0114 Acc: 0.6904
val Loss: 0.0321 Acc: 0.5047

Epoch 44/68
----------
train Loss: 0.0113 Acc: 0.6904
val Loss: 0.0254 Acc: 0.5140

Epoch 45/68
----------
train Loss: 0.0116 Acc: 0.6852
val Loss: 0.0311 Acc: 0.5047

Epoch 46/68
----------
train Loss: 0.0114 Acc: 0.6894
val Loss: 0.0259 Acc: 0.5140

Epoch 47/68
----------
train Loss: 0.0114 Acc: 0.6966
val Loss: 0.0310 Acc: 0.5047

Epoch 48/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0113 Acc: 0.6842
val Loss: 0.0300 Acc: 0.5047

Epoch 49/68
----------
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0280 Acc: 0.5140

Epoch 50/68
----------
train Loss: 0.0114 Acc: 0.6873
val Loss: 0.0246 Acc: 0.5047

Epoch 51/68
----------
train Loss: 0.0115 Acc: 0.6791
val Loss: 0.0246 Acc: 0.5047

Epoch 52/68
----------
train Loss: 0.0113 Acc: 0.6997
val Loss: 0.0309 Acc: 0.5047

Epoch 53/68
----------
train Loss: 0.0114 Acc: 0.6791
val Loss: 0.0301 Acc: 0.5047

Epoch 54/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0329 Acc: 0.5047

Epoch 55/68
----------
train Loss: 0.0114 Acc: 0.6945
val Loss: 0.0381 Acc: 0.5047

Epoch 56/68
----------
train Loss: 0.0112 Acc: 0.7018
val Loss: 0.0270 Acc: 0.5047

Epoch 57/68
----------
train Loss: 0.0114 Acc: 0.6883
val Loss: 0.0286 Acc: 0.5047

Epoch 58/68
----------
train Loss: 0.0113 Acc: 0.6863
val Loss: 0.0246 Acc: 0.5140

Epoch 59/68
----------
train Loss: 0.0115 Acc: 0.6780
val Loss: 0.0267 Acc: 0.5047

Epoch 60/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0115 Acc: 0.6749
val Loss: 0.0331 Acc: 0.5047

Epoch 61/68
----------
train Loss: 0.0114 Acc: 0.6729
val Loss: 0.0239 Acc: 0.5047

Epoch 62/68
----------
train Loss: 0.0114 Acc: 0.6945
val Loss: 0.0299 Acc: 0.5047

Epoch 63/68
----------
train Loss: 0.0115 Acc: 0.6832
val Loss: 0.0290 Acc: 0.5140

Epoch 64/68
----------
train Loss: 0.0114 Acc: 0.6935
val Loss: 0.0303 Acc: 0.5047

Epoch 65/68
----------
train Loss: 0.0113 Acc: 0.6852
val Loss: 0.0244 Acc: 0.5047

Epoch 66/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0114 Acc: 0.6842
val Loss: 0.0370 Acc: 0.5047

Epoch 67/68
----------
train Loss: 0.0112 Acc: 0.6945
val Loss: 0.0280 Acc: 0.5140

Epoch 68/68
----------
train Loss: 0.0113 Acc: 0.6780
val Loss: 0.0347 Acc: 0.5140

Training complete in 6m 24s
Best val Acc: 0.542056

---Fine tuning.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6522
val Loss: 0.0380 Acc: 0.4579

Epoch 1/68
----------
train Loss: 0.0073 Acc: 0.7988
val Loss: 0.0272 Acc: 0.5607

Epoch 2/68
----------
train Loss: 0.0041 Acc: 0.9071
val Loss: 0.0181 Acc: 0.5701

Epoch 3/68
----------
train Loss: 0.0025 Acc: 0.9443
val Loss: 0.0365 Acc: 0.5607

Epoch 4/68
----------
train Loss: 0.0017 Acc: 0.9608
val Loss: 0.0304 Acc: 0.5514

Epoch 5/68
----------
train Loss: 0.0011 Acc: 0.9763
val Loss: 0.0312 Acc: 0.5794

Epoch 6/68
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0328 Acc: 0.5514

Epoch 7/68
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0214 Acc: 0.5514

Epoch 8/68
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0278 Acc: 0.5514

Epoch 9/68
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0179 Acc: 0.5514

Epoch 10/68
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0391 Acc: 0.5514

Epoch 11/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0330 Acc: 0.5607

Epoch 12/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0238 Acc: 0.5607

Epoch 13/68
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0266 Acc: 0.5607

Epoch 14/68
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0286 Acc: 0.5607

Epoch 15/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0194 Acc: 0.5607

Epoch 16/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0391 Acc: 0.5607

Epoch 17/68
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0322 Acc: 0.5607

Epoch 18/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0316 Acc: 0.5514

Epoch 19/68
----------
train Loss: 0.0006 Acc: 0.9886
val Loss: 0.0184 Acc: 0.5514

Epoch 20/68
----------
train Loss: 0.0006 Acc: 0.9845
val Loss: 0.0396 Acc: 0.5514

Epoch 21/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0301 Acc: 0.5514

Epoch 22/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0424 Acc: 0.5607

Epoch 23/68
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0269 Acc: 0.5607

Epoch 24/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0296 Acc: 0.5514

Epoch 25/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0244 Acc: 0.5607

Epoch 26/68
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0295 Acc: 0.5514

Epoch 27/68
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0304 Acc: 0.5607

Epoch 28/68
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0180 Acc: 0.5607

Epoch 29/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0313 Acc: 0.5607

Epoch 30/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0366 Acc: 0.5607

Epoch 31/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0252 Acc: 0.5514

Epoch 32/68
----------
train Loss: 0.0006 Acc: 0.9845
val Loss: 0.0254 Acc: 0.5514

Epoch 33/68
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0268 Acc: 0.5607

Epoch 34/68
----------
train Loss: 0.0006 Acc: 0.9856
val Loss: 0.0340 Acc: 0.5514

Epoch 35/68
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0294 Acc: 0.5607

Epoch 36/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0403 Acc: 0.5607

Epoch 37/68
----------
train Loss: 0.0006 Acc: 0.9794
val Loss: 0.0389 Acc: 0.5514

Epoch 38/68
----------
train Loss: 0.0006 Acc: 0.9856
val Loss: 0.0213 Acc: 0.5514

Epoch 39/68
----------
train Loss: 0.0006 Acc: 0.9856
val Loss: 0.0177 Acc: 0.5514

Epoch 40/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0221 Acc: 0.5514

Epoch 41/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0360 Acc: 0.5514

Epoch 42/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0207 Acc: 0.5514

Epoch 43/68
----------
train Loss: 0.0006 Acc: 0.9845
val Loss: 0.0337 Acc: 0.5514

Epoch 44/68
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0270 Acc: 0.5514

Epoch 45/68
----------
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0245 Acc: 0.5514

Epoch 46/68
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0207 Acc: 0.5514

Epoch 47/68
----------
train Loss: 0.0006 Acc: 0.9845
val Loss: 0.0356 Acc: 0.5514

Epoch 48/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9886
val Loss: 0.0232 Acc: 0.5514

Epoch 49/68
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0328 Acc: 0.5514

Epoch 50/68
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0207 Acc: 0.5514

Epoch 51/68
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0392 Acc: 0.5514

Epoch 52/68
----------
train Loss: 0.0005 Acc: 0.9917
val Loss: 0.0304 Acc: 0.5607

Epoch 53/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0217 Acc: 0.5607

Epoch 54/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0006 Acc: 0.9835
val Loss: 0.0352 Acc: 0.5514

Epoch 55/68
----------
train Loss: 0.0006 Acc: 0.9794
val Loss: 0.0384 Acc: 0.5514

Epoch 56/68
----------
train Loss: 0.0005 Acc: 0.9866
val Loss: 0.0247 Acc: 0.5607

Epoch 57/68
----------
train Loss: 0.0006 Acc: 0.9794
val Loss: 0.0454 Acc: 0.5607

Epoch 58/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0201 Acc: 0.5607

Epoch 59/68
----------
train Loss: 0.0005 Acc: 0.9886
val Loss: 0.0337 Acc: 0.5514

Epoch 60/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0419 Acc: 0.5514

Epoch 61/68
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0230 Acc: 0.5514

Epoch 62/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0268 Acc: 0.5607

Epoch 63/68
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0286 Acc: 0.5514

Epoch 64/68
----------
train Loss: 0.0006 Acc: 0.9845
val Loss: 0.0316 Acc: 0.5514

Epoch 65/68
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0251 Acc: 0.5514

Epoch 66/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0369 Acc: 0.5514

Epoch 67/68
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0364 Acc: 0.5514

Epoch 68/68
----------
train Loss: 0.0006 Acc: 0.9866
val Loss: 0.0274 Acc: 0.5607

Training complete in 6m 51s
Best val Acc: 0.579439

---Testing---
Test accuracy: 0.939591
--------------------
Accuracy of Albacore tuna : 94 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 86 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 94 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 89 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.9187945177580453, std: 0.06154148489701587
--------------------

run info[val: 0.15, epoch: 93, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0284 Acc: 0.1727
val Loss: 0.0287 Acc: 0.2919

Epoch 1/92
----------
train Loss: 0.0234 Acc: 0.3301
val Loss: 0.0248 Acc: 0.3727

Epoch 2/92
----------
train Loss: 0.0195 Acc: 0.4350
val Loss: 0.0250 Acc: 0.3416

Epoch 3/92
----------
train Loss: 0.0188 Acc: 0.4503
val Loss: 0.0232 Acc: 0.3851

Epoch 4/92
----------
train Loss: 0.0167 Acc: 0.5257
val Loss: 0.0251 Acc: 0.3975

Epoch 5/92
----------
train Loss: 0.0154 Acc: 0.5585
val Loss: 0.0225 Acc: 0.4472

Epoch 6/92
----------
train Loss: 0.0148 Acc: 0.5650
val Loss: 0.0219 Acc: 0.4099

Epoch 7/92
----------
train Loss: 0.0148 Acc: 0.5661
val Loss: 0.0223 Acc: 0.4410

Epoch 8/92
----------
train Loss: 0.0139 Acc: 0.6087
val Loss: 0.0227 Acc: 0.4596

Epoch 9/92
----------
train Loss: 0.0130 Acc: 0.6251
val Loss: 0.0211 Acc: 0.4534

Epoch 10/92
----------
train Loss: 0.0129 Acc: 0.6197
val Loss: 0.0224 Acc: 0.4534

Epoch 11/92
----------
train Loss: 0.0131 Acc: 0.6372
val Loss: 0.0240 Acc: 0.4410

Epoch 12/92
----------
train Loss: 0.0115 Acc: 0.6699
val Loss: 0.0233 Acc: 0.4472

Epoch 13/92
----------
LR is set to 0.001
train Loss: 0.0110 Acc: 0.6831
val Loss: 0.0216 Acc: 0.4658

Epoch 14/92
----------
train Loss: 0.0107 Acc: 0.7268
val Loss: 0.0220 Acc: 0.4658

Epoch 15/92
----------
train Loss: 0.0104 Acc: 0.7115
val Loss: 0.0214 Acc: 0.4596

Epoch 16/92
----------
train Loss: 0.0106 Acc: 0.7104
val Loss: 0.0211 Acc: 0.4658

Epoch 17/92
----------
train Loss: 0.0109 Acc: 0.7137
val Loss: 0.0213 Acc: 0.4845

Epoch 18/92
----------
train Loss: 0.0102 Acc: 0.7093
val Loss: 0.0215 Acc: 0.4720

Epoch 19/92
----------
train Loss: 0.0102 Acc: 0.7246
val Loss: 0.0218 Acc: 0.4348

Epoch 20/92
----------
train Loss: 0.0105 Acc: 0.7279
val Loss: 0.0217 Acc: 0.4907

Epoch 21/92
----------
train Loss: 0.0103 Acc: 0.7246
val Loss: 0.0217 Acc: 0.4969

Epoch 22/92
----------
train Loss: 0.0107 Acc: 0.7049
val Loss: 0.0209 Acc: 0.4783

Epoch 23/92
----------
train Loss: 0.0101 Acc: 0.7486
val Loss: 0.0216 Acc: 0.4658

Epoch 24/92
----------
train Loss: 0.0102 Acc: 0.7410
val Loss: 0.0208 Acc: 0.4410

Epoch 25/92
----------
train Loss: 0.0097 Acc: 0.7279
val Loss: 0.0220 Acc: 0.4783

Epoch 26/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0102 Acc: 0.7246
val Loss: 0.0215 Acc: 0.4845

Epoch 27/92
----------
train Loss: 0.0102 Acc: 0.7399
val Loss: 0.0213 Acc: 0.4783

Epoch 28/92
----------
train Loss: 0.0099 Acc: 0.7290
val Loss: 0.0218 Acc: 0.4907

Epoch 29/92
----------
train Loss: 0.0106 Acc: 0.7279
val Loss: 0.0215 Acc: 0.4534

Epoch 30/92
----------
train Loss: 0.0098 Acc: 0.7432
val Loss: 0.0214 Acc: 0.4534

Epoch 31/92
----------
train Loss: 0.0101 Acc: 0.7366
val Loss: 0.0214 Acc: 0.4596

Epoch 32/92
----------
train Loss: 0.0101 Acc: 0.7268
val Loss: 0.0213 Acc: 0.4596

Epoch 33/92
----------
train Loss: 0.0104 Acc: 0.7322
val Loss: 0.0214 Acc: 0.4720

Epoch 34/92
----------
train Loss: 0.0105 Acc: 0.7388
val Loss: 0.0216 Acc: 0.4472

Epoch 35/92
----------
train Loss: 0.0105 Acc: 0.7301
val Loss: 0.0212 Acc: 0.4658

Epoch 36/92
----------
train Loss: 0.0102 Acc: 0.7333
val Loss: 0.0208 Acc: 0.4720

Epoch 37/92
----------
train Loss: 0.0104 Acc: 0.7366
val Loss: 0.0212 Acc: 0.4783

Epoch 38/92
----------
train Loss: 0.0102 Acc: 0.7213
val Loss: 0.0215 Acc: 0.4783

Epoch 39/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0104 Acc: 0.7344
val Loss: 0.0211 Acc: 0.4658

Epoch 40/92
----------
train Loss: 0.0102 Acc: 0.7443
val Loss: 0.0222 Acc: 0.4720

Epoch 41/92
----------
train Loss: 0.0107 Acc: 0.7301
val Loss: 0.0217 Acc: 0.4720

Epoch 42/92
----------
train Loss: 0.0101 Acc: 0.7290
val Loss: 0.0220 Acc: 0.4720

Epoch 43/92
----------
train Loss: 0.0103 Acc: 0.7454
val Loss: 0.0210 Acc: 0.4720

Epoch 44/92
----------
train Loss: 0.0104 Acc: 0.7530
val Loss: 0.0216 Acc: 0.4720

Epoch 45/92
----------
train Loss: 0.0106 Acc: 0.7311
val Loss: 0.0216 Acc: 0.4720

Epoch 46/92
----------
train Loss: 0.0100 Acc: 0.7475
val Loss: 0.0221 Acc: 0.4658

Epoch 47/92
----------
train Loss: 0.0103 Acc: 0.7399
val Loss: 0.0213 Acc: 0.4720

Epoch 48/92
----------
train Loss: 0.0102 Acc: 0.7355
val Loss: 0.0208 Acc: 0.4534

Epoch 49/92
----------
train Loss: 0.0099 Acc: 0.7257
val Loss: 0.0211 Acc: 0.4472

Epoch 50/92
----------
train Loss: 0.0103 Acc: 0.7322
val Loss: 0.0220 Acc: 0.4658

Epoch 51/92
----------
train Loss: 0.0098 Acc: 0.7519
val Loss: 0.0222 Acc: 0.4596

Epoch 52/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0100 Acc: 0.7475
val Loss: 0.0212 Acc: 0.4845

Epoch 53/92
----------
train Loss: 0.0100 Acc: 0.7322
val Loss: 0.0214 Acc: 0.4658

Epoch 54/92
----------
train Loss: 0.0103 Acc: 0.7443
val Loss: 0.0213 Acc: 0.4907

Epoch 55/92
----------
train Loss: 0.0099 Acc: 0.7497
val Loss: 0.0220 Acc: 0.4783

Epoch 56/92
----------
train Loss: 0.0101 Acc: 0.7432
val Loss: 0.0214 Acc: 0.4720

Epoch 57/92
----------
train Loss: 0.0099 Acc: 0.7585
val Loss: 0.0216 Acc: 0.4720

Epoch 58/92
----------
train Loss: 0.0098 Acc: 0.7486
val Loss: 0.0219 Acc: 0.4472

Epoch 59/92
----------
train Loss: 0.0100 Acc: 0.7421
val Loss: 0.0211 Acc: 0.4596

Epoch 60/92
----------
train Loss: 0.0099 Acc: 0.7279
val Loss: 0.0211 Acc: 0.4658

Epoch 61/92
----------
train Loss: 0.0101 Acc: 0.7322
val Loss: 0.0220 Acc: 0.4658

Epoch 62/92
----------
train Loss: 0.0103 Acc: 0.7279
val Loss: 0.0209 Acc: 0.4534

Epoch 63/92
----------
train Loss: 0.0102 Acc: 0.7388
val Loss: 0.0213 Acc: 0.4596

Epoch 64/92
----------
train Loss: 0.0100 Acc: 0.7235
val Loss: 0.0210 Acc: 0.4969

Epoch 65/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0102 Acc: 0.7366
val Loss: 0.0209 Acc: 0.4596

Epoch 66/92
----------
train Loss: 0.0098 Acc: 0.7508
val Loss: 0.0214 Acc: 0.4720

Epoch 67/92
----------
train Loss: 0.0097 Acc: 0.7519
val Loss: 0.0215 Acc: 0.4658

Epoch 68/92
----------
train Loss: 0.0101 Acc: 0.7432
val Loss: 0.0211 Acc: 0.4596

Epoch 69/92
----------
train Loss: 0.0105 Acc: 0.7301
val Loss: 0.0218 Acc: 0.4472

Epoch 70/92
----------
train Loss: 0.0103 Acc: 0.7224
val Loss: 0.0217 Acc: 0.4472

Epoch 71/92
----------
train Loss: 0.0101 Acc: 0.7301
val Loss: 0.0218 Acc: 0.4534

Epoch 72/92
----------
train Loss: 0.0098 Acc: 0.7399
val Loss: 0.0211 Acc: 0.4720

Epoch 73/92
----------
train Loss: 0.0105 Acc: 0.7290
val Loss: 0.0209 Acc: 0.4534

Epoch 74/92
----------
train Loss: 0.0104 Acc: 0.7410
val Loss: 0.0209 Acc: 0.4783

Epoch 75/92
----------
train Loss: 0.0102 Acc: 0.7355
val Loss: 0.0219 Acc: 0.4658

Epoch 76/92
----------
train Loss: 0.0106 Acc: 0.7355
val Loss: 0.0213 Acc: 0.4658

Epoch 77/92
----------
train Loss: 0.0100 Acc: 0.7301
val Loss: 0.0217 Acc: 0.4596

Epoch 78/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0101 Acc: 0.7475
val Loss: 0.0215 Acc: 0.4720

Epoch 79/92
----------
train Loss: 0.0100 Acc: 0.7410
val Loss: 0.0213 Acc: 0.4658

Epoch 80/92
----------
train Loss: 0.0101 Acc: 0.7454
val Loss: 0.0211 Acc: 0.4720

Epoch 81/92
----------
train Loss: 0.0100 Acc: 0.7388
val Loss: 0.0211 Acc: 0.4658

Epoch 82/92
----------
train Loss: 0.0100 Acc: 0.7508
val Loss: 0.0216 Acc: 0.4534

Epoch 83/92
----------
train Loss: 0.0103 Acc: 0.7290
val Loss: 0.0216 Acc: 0.4534

Epoch 84/92
----------
train Loss: 0.0107 Acc: 0.7235
val Loss: 0.0223 Acc: 0.4596

Epoch 85/92
----------
train Loss: 0.0103 Acc: 0.7246
val Loss: 0.0219 Acc: 0.4783

Epoch 86/92
----------
train Loss: 0.0105 Acc: 0.7432
val Loss: 0.0217 Acc: 0.4596

Epoch 87/92
----------
train Loss: 0.0104 Acc: 0.7246
val Loss: 0.0216 Acc: 0.4720

Epoch 88/92
----------
train Loss: 0.0103 Acc: 0.7311
val Loss: 0.0210 Acc: 0.4783

Epoch 89/92
----------
train Loss: 0.0099 Acc: 0.7410
val Loss: 0.0224 Acc: 0.4596

Epoch 90/92
----------
train Loss: 0.0099 Acc: 0.7421
val Loss: 0.0214 Acc: 0.4783

Epoch 91/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0099 Acc: 0.7508
val Loss: 0.0215 Acc: 0.4720

Epoch 92/92
----------
train Loss: 0.0101 Acc: 0.7388
val Loss: 0.0218 Acc: 0.4658

Training complete in 8m 34s
Best val Acc: 0.496894

---Fine tuning.---
Epoch 0/92
----------
LR is set to 0.01
train Loss: 0.0113 Acc: 0.7005
val Loss: 0.0239 Acc: 0.4783

Epoch 1/92
----------
train Loss: 0.0089 Acc: 0.7443
val Loss: 0.0278 Acc: 0.4286

Epoch 2/92
----------
train Loss: 0.0070 Acc: 0.8098
val Loss: 0.0231 Acc: 0.4783

Epoch 3/92
----------
train Loss: 0.0052 Acc: 0.8645
val Loss: 0.0255 Acc: 0.4658

Epoch 4/92
----------
train Loss: 0.0043 Acc: 0.8656
val Loss: 0.0220 Acc: 0.5093

Epoch 5/92
----------
train Loss: 0.0034 Acc: 0.9148
val Loss: 0.0241 Acc: 0.5031

Epoch 6/92
----------
train Loss: 0.0032 Acc: 0.9148
val Loss: 0.0263 Acc: 0.5031

Epoch 7/92
----------
train Loss: 0.0031 Acc: 0.9235
val Loss: 0.0251 Acc: 0.4907

Epoch 8/92
----------
train Loss: 0.0027 Acc: 0.9301
val Loss: 0.0301 Acc: 0.4286

Epoch 9/92
----------
train Loss: 0.0020 Acc: 0.9388
val Loss: 0.0271 Acc: 0.5031

Epoch 10/92
----------
train Loss: 0.0018 Acc: 0.9486
val Loss: 0.0262 Acc: 0.4596

Epoch 11/92
----------
train Loss: 0.0013 Acc: 0.9596
val Loss: 0.0245 Acc: 0.5466

Epoch 12/92
----------
train Loss: 0.0012 Acc: 0.9694
val Loss: 0.0266 Acc: 0.4907

Epoch 13/92
----------
LR is set to 0.001
train Loss: 0.0011 Acc: 0.9727
val Loss: 0.0267 Acc: 0.5342

Epoch 14/92
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0260 Acc: 0.5342

Epoch 15/92
----------
train Loss: 0.0008 Acc: 0.9749
val Loss: 0.0239 Acc: 0.5528

Epoch 16/92
----------
train Loss: 0.0007 Acc: 0.9781
val Loss: 0.0259 Acc: 0.5528

Epoch 17/92
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0275 Acc: 0.5466

Epoch 18/92
----------
train Loss: 0.0007 Acc: 0.9858
val Loss: 0.0259 Acc: 0.5342

Epoch 19/92
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0259 Acc: 0.5404

Epoch 20/92
----------
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0252 Acc: 0.5466

Epoch 21/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0255 Acc: 0.5528

Epoch 22/92
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0265 Acc: 0.5528

Epoch 23/92
----------
train Loss: 0.0007 Acc: 0.9847
val Loss: 0.0247 Acc: 0.5590

Epoch 24/92
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0249 Acc: 0.5466

Epoch 25/92
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0268 Acc: 0.5466

Epoch 26/92
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0273 Acc: 0.5466

Epoch 27/92
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0265 Acc: 0.5466

Epoch 28/92
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0268 Acc: 0.5404

Epoch 29/92
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0255 Acc: 0.5404

Epoch 30/92
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0274 Acc: 0.5342

Epoch 31/92
----------
train Loss: 0.0008 Acc: 0.9781
val Loss: 0.0262 Acc: 0.5466

Epoch 32/92
----------
train Loss: 0.0007 Acc: 0.9792
val Loss: 0.0260 Acc: 0.5590

Epoch 33/92
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0270 Acc: 0.5528

Epoch 34/92
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0257 Acc: 0.5528

Epoch 35/92
----------
train Loss: 0.0008 Acc: 0.9847
val Loss: 0.0261 Acc: 0.5404

Epoch 36/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0256 Acc: 0.5466

Epoch 37/92
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0282 Acc: 0.5404

Epoch 38/92
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0265 Acc: 0.5342

Epoch 39/92
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0260 Acc: 0.5342

Epoch 40/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0260 Acc: 0.5466

Epoch 41/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0258 Acc: 0.5466

Epoch 42/92
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0253 Acc: 0.5528

Epoch 43/92
----------
train Loss: 0.0009 Acc: 0.9869
val Loss: 0.0260 Acc: 0.5404

Epoch 44/92
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0257 Acc: 0.5404

Epoch 45/92
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0268 Acc: 0.5528

Epoch 46/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0274 Acc: 0.5528

Epoch 47/92
----------
train Loss: 0.0007 Acc: 0.9770
val Loss: 0.0258 Acc: 0.5404

Epoch 48/92
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0268 Acc: 0.5528

Epoch 49/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0262 Acc: 0.5466

Epoch 50/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0251 Acc: 0.5466

Epoch 51/92
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0257 Acc: 0.5466

Epoch 52/92
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0265 Acc: 0.5466

Epoch 53/92
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0254 Acc: 0.5404

Epoch 54/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0250 Acc: 0.5466

Epoch 55/92
----------
train Loss: 0.0006 Acc: 0.9869
val Loss: 0.0251 Acc: 0.5528

Epoch 56/92
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0267 Acc: 0.5466

Epoch 57/92
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0268 Acc: 0.5652

Epoch 58/92
----------
train Loss: 0.0008 Acc: 0.9749
val Loss: 0.0260 Acc: 0.5590

Epoch 59/92
----------
train Loss: 0.0007 Acc: 0.9836
val Loss: 0.0275 Acc: 0.5590

Epoch 60/92
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0259 Acc: 0.5466

Epoch 61/92
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0266 Acc: 0.5590

Epoch 62/92
----------
train Loss: 0.0007 Acc: 0.9847
val Loss: 0.0271 Acc: 0.5528

Epoch 63/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0262 Acc: 0.5528

Epoch 64/92
----------
train Loss: 0.0008 Acc: 0.9869
val Loss: 0.0256 Acc: 0.5466

Epoch 65/92
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0254 Acc: 0.5466

Epoch 66/92
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0252 Acc: 0.5528

Epoch 67/92
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5528

Epoch 68/92
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0274 Acc: 0.5590

Epoch 69/92
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0251 Acc: 0.5528

Epoch 70/92
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0255 Acc: 0.5528

Epoch 71/92
----------
train Loss: 0.0007 Acc: 0.9781
val Loss: 0.0269 Acc: 0.5590

Epoch 72/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0268 Acc: 0.5466

Epoch 73/92
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0275 Acc: 0.5466

Epoch 74/92
----------
train Loss: 0.0006 Acc: 0.9760
val Loss: 0.0248 Acc: 0.5528

Epoch 75/92
----------
train Loss: 0.0006 Acc: 0.9869
val Loss: 0.0260 Acc: 0.5528

Epoch 76/92
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0272 Acc: 0.5528

Epoch 77/92
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0273 Acc: 0.5590

Epoch 78/92
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0268 Acc: 0.5652

Epoch 79/92
----------
train Loss: 0.0006 Acc: 0.9781
val Loss: 0.0271 Acc: 0.5466

Epoch 80/92
----------
train Loss: 0.0007 Acc: 0.9770
val Loss: 0.0253 Acc: 0.5528

Epoch 81/92
----------
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0260 Acc: 0.5528

Epoch 82/92
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0260 Acc: 0.5466

Epoch 83/92
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0251 Acc: 0.5590

Epoch 84/92
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0260 Acc: 0.5466

Epoch 85/92
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0269 Acc: 0.5342

Epoch 86/92
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0269 Acc: 0.5466

Epoch 87/92
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0272 Acc: 0.5528

Epoch 88/92
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0254 Acc: 0.5466

Epoch 89/92
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0264 Acc: 0.5466

Epoch 90/92
----------
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0265 Acc: 0.5466

Epoch 91/92
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0266 Acc: 0.5466

Epoch 92/92
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0259 Acc: 0.5404

Training complete in 9m 7s
Best val Acc: 0.565217

---Testing---
Test accuracy: 0.921004
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 89 %
Accuracy of Little tunny : 97 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 84 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.8983643313772601, std: 0.06253681015603103
--------------------

run info[val: 0.2, epoch: 95, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0271 Acc: 0.1487
val Loss: 0.0310 Acc: 0.3023

Epoch 1/94
----------
train Loss: 0.0216 Acc: 0.3298
val Loss: 0.0285 Acc: 0.3163

Epoch 2/94
----------
train Loss: 0.0182 Acc: 0.4390
val Loss: 0.0264 Acc: 0.4000

Epoch 3/94
----------
train Loss: 0.0163 Acc: 0.5285
val Loss: 0.0235 Acc: 0.4326

Epoch 4/94
----------
train Loss: 0.0142 Acc: 0.5923
val Loss: 0.0239 Acc: 0.4186

Epoch 5/94
----------
train Loss: 0.0134 Acc: 0.6179
val Loss: 0.0241 Acc: 0.4093

Epoch 6/94
----------
train Loss: 0.0129 Acc: 0.6179
val Loss: 0.0269 Acc: 0.4000

Epoch 7/94
----------
train Loss: 0.0119 Acc: 0.6562
val Loss: 0.0250 Acc: 0.4419

Epoch 8/94
----------
train Loss: 0.0111 Acc: 0.6783
val Loss: 0.0236 Acc: 0.4698

Epoch 9/94
----------
train Loss: 0.0109 Acc: 0.6783
val Loss: 0.0228 Acc: 0.4605

Epoch 10/94
----------
train Loss: 0.0101 Acc: 0.7131
val Loss: 0.0233 Acc: 0.4651

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0093 Acc: 0.7433
val Loss: 0.0234 Acc: 0.4605

Epoch 12/94
----------
train Loss: 0.0091 Acc: 0.7689
val Loss: 0.0213 Acc: 0.4698

Epoch 13/94
----------
train Loss: 0.0090 Acc: 0.7747
val Loss: 0.0227 Acc: 0.4884

Epoch 14/94
----------
train Loss: 0.0091 Acc: 0.7689
val Loss: 0.0237 Acc: 0.4791

Epoch 15/94
----------
train Loss: 0.0090 Acc: 0.7677
val Loss: 0.0228 Acc: 0.4698

Epoch 16/94
----------
train Loss: 0.0090 Acc: 0.7561
val Loss: 0.0232 Acc: 0.4744

Epoch 17/94
----------
train Loss: 0.0089 Acc: 0.7782
val Loss: 0.0235 Acc: 0.4698

Epoch 18/94
----------
train Loss: 0.0089 Acc: 0.7782
val Loss: 0.0231 Acc: 0.4791

Epoch 19/94
----------
train Loss: 0.0087 Acc: 0.7747
val Loss: 0.0220 Acc: 0.4791

Epoch 20/94
----------
train Loss: 0.0088 Acc: 0.7851
val Loss: 0.0252 Acc: 0.4884

Epoch 21/94
----------
train Loss: 0.0088 Acc: 0.7724
val Loss: 0.0209 Acc: 0.4744

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0087 Acc: 0.7875
val Loss: 0.0248 Acc: 0.4698

Epoch 23/94
----------
train Loss: 0.0087 Acc: 0.7840
val Loss: 0.0204 Acc: 0.4744

Epoch 24/94
----------
train Loss: 0.0087 Acc: 0.7735
val Loss: 0.0226 Acc: 0.4791

Epoch 25/94
----------
train Loss: 0.0086 Acc: 0.7944
val Loss: 0.0228 Acc: 0.4698

Epoch 26/94
----------
train Loss: 0.0087 Acc: 0.7921
val Loss: 0.0232 Acc: 0.4744

Epoch 27/94
----------
train Loss: 0.0088 Acc: 0.7816
val Loss: 0.0233 Acc: 0.4791

Epoch 28/94
----------
train Loss: 0.0087 Acc: 0.7758
val Loss: 0.0251 Acc: 0.4744

Epoch 29/94
----------
train Loss: 0.0086 Acc: 0.7909
val Loss: 0.0213 Acc: 0.4837

Epoch 30/94
----------
train Loss: 0.0086 Acc: 0.7944
val Loss: 0.0219 Acc: 0.4698

Epoch 31/94
----------
train Loss: 0.0087 Acc: 0.7805
val Loss: 0.0230 Acc: 0.4698

Epoch 32/94
----------
train Loss: 0.0087 Acc: 0.7700
val Loss: 0.0231 Acc: 0.4698

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0086 Acc: 0.7909
val Loss: 0.0268 Acc: 0.4837

Epoch 34/94
----------
train Loss: 0.0087 Acc: 0.7793
val Loss: 0.0237 Acc: 0.4791

Epoch 35/94
----------
train Loss: 0.0086 Acc: 0.7851
val Loss: 0.0244 Acc: 0.4744

Epoch 36/94
----------
train Loss: 0.0087 Acc: 0.7863
val Loss: 0.0225 Acc: 0.4698

Epoch 37/94
----------
train Loss: 0.0086 Acc: 0.7909
val Loss: 0.0241 Acc: 0.4744

Epoch 38/94
----------
train Loss: 0.0087 Acc: 0.7828
val Loss: 0.0252 Acc: 0.4698

Epoch 39/94
----------
train Loss: 0.0087 Acc: 0.7782
val Loss: 0.0219 Acc: 0.4744

Epoch 40/94
----------
train Loss: 0.0087 Acc: 0.7793
val Loss: 0.0247 Acc: 0.4884

Epoch 41/94
----------
train Loss: 0.0087 Acc: 0.7851
val Loss: 0.0248 Acc: 0.4837

Epoch 42/94
----------
train Loss: 0.0085 Acc: 0.7909
val Loss: 0.0248 Acc: 0.4698

Epoch 43/94
----------
train Loss: 0.0087 Acc: 0.7805
val Loss: 0.0246 Acc: 0.4744

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0087 Acc: 0.7828
val Loss: 0.0218 Acc: 0.4837

Epoch 45/94
----------
train Loss: 0.0086 Acc: 0.7851
val Loss: 0.0216 Acc: 0.4698

Epoch 46/94
----------
train Loss: 0.0088 Acc: 0.7840
val Loss: 0.0237 Acc: 0.4744

Epoch 47/94
----------
train Loss: 0.0087 Acc: 0.7816
val Loss: 0.0242 Acc: 0.4698

Epoch 48/94
----------
train Loss: 0.0087 Acc: 0.7863
val Loss: 0.0249 Acc: 0.4744

Epoch 49/94
----------
train Loss: 0.0085 Acc: 0.7944
val Loss: 0.0234 Acc: 0.4698

Epoch 50/94
----------
train Loss: 0.0086 Acc: 0.7886
val Loss: 0.0233 Acc: 0.4837

Epoch 51/94
----------
train Loss: 0.0087 Acc: 0.7886
val Loss: 0.0242 Acc: 0.4791

Epoch 52/94
----------
train Loss: 0.0088 Acc: 0.7805
val Loss: 0.0236 Acc: 0.4884

Epoch 53/94
----------
train Loss: 0.0086 Acc: 0.7875
val Loss: 0.0215 Acc: 0.4837

Epoch 54/94
----------
train Loss: 0.0087 Acc: 0.7898
val Loss: 0.0221 Acc: 0.4698

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0089 Acc: 0.7816
val Loss: 0.0249 Acc: 0.4791

Epoch 56/94
----------
train Loss: 0.0087 Acc: 0.7758
val Loss: 0.0245 Acc: 0.4744

Epoch 57/94
----------
train Loss: 0.0086 Acc: 0.7840
val Loss: 0.0230 Acc: 0.4744

Epoch 58/94
----------
train Loss: 0.0087 Acc: 0.7770
val Loss: 0.0230 Acc: 0.4698

Epoch 59/94
----------
train Loss: 0.0086 Acc: 0.7805
val Loss: 0.0231 Acc: 0.4791

Epoch 60/94
----------
train Loss: 0.0087 Acc: 0.7967
val Loss: 0.0233 Acc: 0.4698

Epoch 61/94
----------
train Loss: 0.0087 Acc: 0.7782
val Loss: 0.0232 Acc: 0.4698

Epoch 62/94
----------
train Loss: 0.0086 Acc: 0.8037
val Loss: 0.0228 Acc: 0.4698

Epoch 63/94
----------
train Loss: 0.0087 Acc: 0.7933
val Loss: 0.0242 Acc: 0.4651

Epoch 64/94
----------
train Loss: 0.0086 Acc: 0.7991
val Loss: 0.0238 Acc: 0.4791

Epoch 65/94
----------
train Loss: 0.0086 Acc: 0.7898
val Loss: 0.0265 Acc: 0.4744

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0088 Acc: 0.7700
val Loss: 0.0255 Acc: 0.4744

Epoch 67/94
----------
train Loss: 0.0088 Acc: 0.7840
val Loss: 0.0259 Acc: 0.4744

Epoch 68/94
----------
train Loss: 0.0087 Acc: 0.7782
val Loss: 0.0255 Acc: 0.4837

Epoch 69/94
----------
train Loss: 0.0085 Acc: 0.7979
val Loss: 0.0214 Acc: 0.4884

Epoch 70/94
----------
train Loss: 0.0087 Acc: 0.7724
val Loss: 0.0237 Acc: 0.4744

Epoch 71/94
----------
train Loss: 0.0088 Acc: 0.7898
val Loss: 0.0216 Acc: 0.4744

Epoch 72/94
----------
train Loss: 0.0086 Acc: 0.7944
val Loss: 0.0230 Acc: 0.4791

Epoch 73/94
----------
train Loss: 0.0087 Acc: 0.7828
val Loss: 0.0211 Acc: 0.4698

Epoch 74/94
----------
train Loss: 0.0086 Acc: 0.7828
val Loss: 0.0254 Acc: 0.4837

Epoch 75/94
----------
train Loss: 0.0089 Acc: 0.7747
val Loss: 0.0248 Acc: 0.4884

Epoch 76/94
----------
train Loss: 0.0087 Acc: 0.7828
val Loss: 0.0237 Acc: 0.4884

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0086 Acc: 0.7967
val Loss: 0.0219 Acc: 0.4837

Epoch 78/94
----------
train Loss: 0.0086 Acc: 0.7851
val Loss: 0.0215 Acc: 0.4791

Epoch 79/94
----------
train Loss: 0.0086 Acc: 0.7956
val Loss: 0.0239 Acc: 0.4651

Epoch 80/94
----------
train Loss: 0.0084 Acc: 0.7898
val Loss: 0.0223 Acc: 0.4651

Epoch 81/94
----------
train Loss: 0.0085 Acc: 0.7793
val Loss: 0.0240 Acc: 0.4744

Epoch 82/94
----------
train Loss: 0.0087 Acc: 0.7840
val Loss: 0.0242 Acc: 0.4791

Epoch 83/94
----------
train Loss: 0.0085 Acc: 0.7840
val Loss: 0.0261 Acc: 0.4744

Epoch 84/94
----------
train Loss: 0.0088 Acc: 0.7840
val Loss: 0.0225 Acc: 0.4698

Epoch 85/94
----------
train Loss: 0.0086 Acc: 0.7805
val Loss: 0.0220 Acc: 0.4791

Epoch 86/94
----------
train Loss: 0.0087 Acc: 0.7816
val Loss: 0.0243 Acc: 0.4791

Epoch 87/94
----------
train Loss: 0.0087 Acc: 0.7747
val Loss: 0.0228 Acc: 0.4698

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0087 Acc: 0.7875
val Loss: 0.0231 Acc: 0.4884

Epoch 89/94
----------
train Loss: 0.0087 Acc: 0.7735
val Loss: 0.0218 Acc: 0.4744

Epoch 90/94
----------
train Loss: 0.0087 Acc: 0.7805
val Loss: 0.0251 Acc: 0.4744

Epoch 91/94
----------
train Loss: 0.0085 Acc: 0.7956
val Loss: 0.0238 Acc: 0.4791

Epoch 92/94
----------
train Loss: 0.0086 Acc: 0.7863
val Loss: 0.0225 Acc: 0.4744

Epoch 93/94
----------
train Loss: 0.0086 Acc: 0.7909
val Loss: 0.0248 Acc: 0.4791

Epoch 94/94
----------
train Loss: 0.0086 Acc: 0.7840
val Loss: 0.0222 Acc: 0.4698

Training complete in 8m 24s
Best val Acc: 0.488372

---Fine tuning.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0095 Acc: 0.7305
val Loss: 0.0246 Acc: 0.4791

Epoch 1/94
----------
train Loss: 0.0054 Acc: 0.8618
val Loss: 0.0254 Acc: 0.4744

Epoch 2/94
----------
train Loss: 0.0029 Acc: 0.9396
val Loss: 0.0257 Acc: 0.5070

Epoch 3/94
----------
train Loss: 0.0018 Acc: 0.9617
val Loss: 0.0220 Acc: 0.5581

Epoch 4/94
----------
train Loss: 0.0013 Acc: 0.9721
val Loss: 0.0206 Acc: 0.5349

Epoch 5/94
----------
train Loss: 0.0011 Acc: 0.9710
val Loss: 0.0234 Acc: 0.5256

Epoch 6/94
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0260 Acc: 0.5442

Epoch 7/94
----------
train Loss: 0.0007 Acc: 0.9768
val Loss: 0.0274 Acc: 0.5302

Epoch 8/94
----------
train Loss: 0.0007 Acc: 0.9768
val Loss: 0.0257 Acc: 0.5395

Epoch 9/94
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0222 Acc: 0.5302

Epoch 10/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0202 Acc: 0.5349

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0259 Acc: 0.5581

Epoch 12/94
----------
train Loss: 0.0004 Acc: 0.9895
val Loss: 0.0244 Acc: 0.5581

Epoch 13/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0239 Acc: 0.5488

Epoch 14/94
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0254 Acc: 0.5349

Epoch 15/94
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0238 Acc: 0.5442

Epoch 16/94
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0244 Acc: 0.5442

Epoch 17/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0243 Acc: 0.5442

Epoch 18/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0215 Acc: 0.5442

Epoch 19/94
----------
train Loss: 0.0004 Acc: 0.9791
val Loss: 0.0244 Acc: 0.5442

Epoch 20/94
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0269 Acc: 0.5535

Epoch 21/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0207 Acc: 0.5535

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0247 Acc: 0.5488

Epoch 23/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0229 Acc: 0.5442

Epoch 24/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0191 Acc: 0.5442

Epoch 25/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0225 Acc: 0.5488

Epoch 26/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0262 Acc: 0.5442

Epoch 27/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0283 Acc: 0.5488

Epoch 28/94
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0277 Acc: 0.5488

Epoch 29/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0253 Acc: 0.5395

Epoch 30/94
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0251 Acc: 0.5442

Epoch 31/94
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0233 Acc: 0.5442

Epoch 32/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0208 Acc: 0.5442

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0239 Acc: 0.5442

Epoch 34/94
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0226 Acc: 0.5442

Epoch 35/94
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0267 Acc: 0.5442

Epoch 36/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0250 Acc: 0.5488

Epoch 37/94
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0228 Acc: 0.5442

Epoch 38/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0233 Acc: 0.5442

Epoch 39/94
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0242 Acc: 0.5488

Epoch 40/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0222 Acc: 0.5442

Epoch 41/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0264 Acc: 0.5442

Epoch 42/94
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0222 Acc: 0.5442

Epoch 43/94
----------
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0248 Acc: 0.5488

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0225 Acc: 0.5395

Epoch 45/94
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0274 Acc: 0.5442

Epoch 46/94
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0236 Acc: 0.5442

Epoch 47/94
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0231 Acc: 0.5442

Epoch 48/94
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0239 Acc: 0.5442

Epoch 49/94
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0263 Acc: 0.5442

Epoch 50/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0232 Acc: 0.5442

Epoch 51/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0257 Acc: 0.5488

Epoch 52/94
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0256 Acc: 0.5488

Epoch 53/94
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0276 Acc: 0.5488

Epoch 54/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0212 Acc: 0.5442

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0267 Acc: 0.5442

Epoch 56/94
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0275 Acc: 0.5488

Epoch 57/94
----------
train Loss: 0.0003 Acc: 0.9826
val Loss: 0.0287 Acc: 0.5535

Epoch 58/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0198 Acc: 0.5442

Epoch 59/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0241 Acc: 0.5442

Epoch 60/94
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0259 Acc: 0.5395

Epoch 61/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0262 Acc: 0.5535

Epoch 62/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0245 Acc: 0.5442

Epoch 63/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0241 Acc: 0.5395

Epoch 64/94
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0247 Acc: 0.5395

Epoch 65/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0231 Acc: 0.5442

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0207 Acc: 0.5442

Epoch 67/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0239 Acc: 0.5395

Epoch 68/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0273 Acc: 0.5442

Epoch 69/94
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0243 Acc: 0.5442

Epoch 70/94
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0282 Acc: 0.5442

Epoch 71/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0225 Acc: 0.5488

Epoch 72/94
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0256 Acc: 0.5395

Epoch 73/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0210 Acc: 0.5442

Epoch 74/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0242 Acc: 0.5488

Epoch 75/94
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0231 Acc: 0.5488

Epoch 76/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0219 Acc: 0.5535

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0271 Acc: 0.5488

Epoch 78/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0244 Acc: 0.5535

Epoch 79/94
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0235 Acc: 0.5442

Epoch 80/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0248 Acc: 0.5488

Epoch 81/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0235 Acc: 0.5442

Epoch 82/94
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0233 Acc: 0.5442

Epoch 83/94
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0252 Acc: 0.5395

Epoch 84/94
----------
train Loss: 0.0003 Acc: 0.9837
val Loss: 0.0248 Acc: 0.5488

Epoch 85/94
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0225 Acc: 0.5488

Epoch 86/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0253 Acc: 0.5488

Epoch 87/94
----------
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0256 Acc: 0.5395

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0242 Acc: 0.5442

Epoch 89/94
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0230 Acc: 0.5442

Epoch 90/94
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0214 Acc: 0.5488

Epoch 91/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0272 Acc: 0.5395

Epoch 92/94
----------
train Loss: 0.0003 Acc: 0.9861
val Loss: 0.0257 Acc: 0.5488

Epoch 93/94
----------
train Loss: 0.0003 Acc: 0.9884
val Loss: 0.0227 Acc: 0.5488

Epoch 94/94
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0269 Acc: 0.5442

Training complete in 8m 57s
Best val Acc: 0.558140

---Testing---
Test accuracy: 0.883829
--------------------
Accuracy of Albacore tuna : 81 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 86 %
Accuracy of Frigate tuna : 86 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 78 %
Accuracy of Pacific bluefin tuna : 73 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 82 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8479094599896166, std: 0.09359585756054443
--------------------

run info[val: 0.25, epoch: 68, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/67
----------
LR is set to 0.01
train Loss: 0.0285 Acc: 0.1660
val Loss: 0.0278 Acc: 0.2156

Epoch 1/67
----------
train Loss: 0.0250 Acc: 0.2924
val Loss: 0.0240 Acc: 0.3420

Epoch 2/67
----------
train Loss: 0.0210 Acc: 0.3965
val Loss: 0.0214 Acc: 0.4089

Epoch 3/67
----------
train Loss: 0.0180 Acc: 0.4994
val Loss: 0.0203 Acc: 0.4126

Epoch 4/67
----------
train Loss: 0.0170 Acc: 0.5118
val Loss: 0.0215 Acc: 0.3866

Epoch 5/67
----------
train Loss: 0.0160 Acc: 0.5465
val Loss: 0.0220 Acc: 0.3903

Epoch 6/67
----------
train Loss: 0.0158 Acc: 0.5774
val Loss: 0.0203 Acc: 0.4498

Epoch 7/67
----------
train Loss: 0.0144 Acc: 0.6097
val Loss: 0.0197 Acc: 0.4387

Epoch 8/67
----------
train Loss: 0.0134 Acc: 0.6245
val Loss: 0.0203 Acc: 0.4387

Epoch 9/67
----------
train Loss: 0.0132 Acc: 0.6307
val Loss: 0.0211 Acc: 0.4424

Epoch 10/67
----------
train Loss: 0.0129 Acc: 0.6431
val Loss: 0.0204 Acc: 0.4349

Epoch 11/67
----------
train Loss: 0.0118 Acc: 0.6791
val Loss: 0.0216 Acc: 0.3978

Epoch 12/67
----------
LR is set to 0.001
train Loss: 0.0113 Acc: 0.6791
val Loss: 0.0196 Acc: 0.4684

Epoch 13/67
----------
train Loss: 0.0103 Acc: 0.7447
val Loss: 0.0193 Acc: 0.4684

Epoch 14/67
----------
train Loss: 0.0102 Acc: 0.7472
val Loss: 0.0188 Acc: 0.4535

Epoch 15/67
----------
train Loss: 0.0096 Acc: 0.7658
val Loss: 0.0188 Acc: 0.4610

Epoch 16/67
----------
train Loss: 0.0101 Acc: 0.7745
val Loss: 0.0187 Acc: 0.4647

Epoch 17/67
----------
train Loss: 0.0103 Acc: 0.7546
val Loss: 0.0187 Acc: 0.4758

Epoch 18/67
----------
train Loss: 0.0090 Acc: 0.7770
val Loss: 0.0185 Acc: 0.4721

Epoch 19/67
----------
train Loss: 0.0102 Acc: 0.7807
val Loss: 0.0188 Acc: 0.4758

Epoch 20/67
----------
train Loss: 0.0099 Acc: 0.7658
val Loss: 0.0189 Acc: 0.4684

Epoch 21/67
----------
train Loss: 0.0091 Acc: 0.7584
val Loss: 0.0185 Acc: 0.4870

Epoch 22/67
----------
train Loss: 0.0099 Acc: 0.7695
val Loss: 0.0187 Acc: 0.4870

Epoch 23/67
----------
train Loss: 0.0107 Acc: 0.7794
val Loss: 0.0191 Acc: 0.4758

Epoch 24/67
----------
LR is set to 0.00010000000000000002
train Loss: 0.0090 Acc: 0.7770
val Loss: 0.0188 Acc: 0.4758

Epoch 25/67
----------
train Loss: 0.0091 Acc: 0.7819
val Loss: 0.0186 Acc: 0.4684

Epoch 26/67
----------
train Loss: 0.0099 Acc: 0.7844
val Loss: 0.0189 Acc: 0.4610

Epoch 27/67
----------
train Loss: 0.0091 Acc: 0.7782
val Loss: 0.0187 Acc: 0.4610

Epoch 28/67
----------
train Loss: 0.0101 Acc: 0.7918
val Loss: 0.0188 Acc: 0.4758

Epoch 29/67
----------
train Loss: 0.0093 Acc: 0.7943
val Loss: 0.0186 Acc: 0.4758

Epoch 30/67
----------
train Loss: 0.0092 Acc: 0.7869
val Loss: 0.0189 Acc: 0.4758

Epoch 31/67
----------
train Loss: 0.0091 Acc: 0.7906
val Loss: 0.0184 Acc: 0.4944

Epoch 32/67
----------
train Loss: 0.0095 Acc: 0.7918
val Loss: 0.0188 Acc: 0.4796

Epoch 33/67
----------
train Loss: 0.0091 Acc: 0.7980
val Loss: 0.0188 Acc: 0.4758

Epoch 34/67
----------
train Loss: 0.0091 Acc: 0.7918
val Loss: 0.0185 Acc: 0.4796

Epoch 35/67
----------
train Loss: 0.0101 Acc: 0.7869
val Loss: 0.0188 Acc: 0.4721

Epoch 36/67
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0089 Acc: 0.7732
val Loss: 0.0186 Acc: 0.4758

Epoch 37/67
----------
train Loss: 0.0096 Acc: 0.7670
val Loss: 0.0186 Acc: 0.4796

Epoch 38/67
----------
train Loss: 0.0094 Acc: 0.7831
val Loss: 0.0188 Acc: 0.4833

Epoch 39/67
----------
train Loss: 0.0095 Acc: 0.7819
val Loss: 0.0190 Acc: 0.4758

Epoch 40/67
----------
train Loss: 0.0096 Acc: 0.7918
val Loss: 0.0187 Acc: 0.4796

Epoch 41/67
----------
train Loss: 0.0091 Acc: 0.7943
val Loss: 0.0188 Acc: 0.4796

Epoch 42/67
----------
train Loss: 0.0089 Acc: 0.7869
val Loss: 0.0187 Acc: 0.4870

Epoch 43/67
----------
train Loss: 0.0086 Acc: 0.7906
val Loss: 0.0188 Acc: 0.4684

Epoch 44/67
----------
train Loss: 0.0091 Acc: 0.7770
val Loss: 0.0189 Acc: 0.4647

Epoch 45/67
----------
train Loss: 0.0095 Acc: 0.7794
val Loss: 0.0189 Acc: 0.4833

Epoch 46/67
----------
train Loss: 0.0097 Acc: 0.7931
val Loss: 0.0186 Acc: 0.4721

Epoch 47/67
----------
train Loss: 0.0094 Acc: 0.7807
val Loss: 0.0186 Acc: 0.4833

Epoch 48/67
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0090 Acc: 0.7893
val Loss: 0.0186 Acc: 0.4870

Epoch 49/67
----------
train Loss: 0.0095 Acc: 0.7918
val Loss: 0.0187 Acc: 0.4721

Epoch 50/67
----------
train Loss: 0.0091 Acc: 0.7856
val Loss: 0.0186 Acc: 0.4647

Epoch 51/67
----------
train Loss: 0.0095 Acc: 0.7831
val Loss: 0.0186 Acc: 0.4684

Epoch 52/67
----------
train Loss: 0.0092 Acc: 0.7794
val Loss: 0.0189 Acc: 0.4647

Epoch 53/67
----------
train Loss: 0.0095 Acc: 0.7807
val Loss: 0.0188 Acc: 0.4684

Epoch 54/67
----------
train Loss: 0.0094 Acc: 0.7770
val Loss: 0.0185 Acc: 0.4721

Epoch 55/67
----------
train Loss: 0.0101 Acc: 0.7881
val Loss: 0.0184 Acc: 0.4758

Epoch 56/67
----------
train Loss: 0.0099 Acc: 0.7732
val Loss: 0.0187 Acc: 0.4907

Epoch 57/67
----------
train Loss: 0.0086 Acc: 0.7794
val Loss: 0.0189 Acc: 0.4758

Epoch 58/67
----------
train Loss: 0.0094 Acc: 0.7943
val Loss: 0.0191 Acc: 0.4721

Epoch 59/67
----------
train Loss: 0.0089 Acc: 0.7980
val Loss: 0.0188 Acc: 0.4833

Epoch 60/67
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0086 Acc: 0.7881
val Loss: 0.0191 Acc: 0.4870

Epoch 61/67
----------
train Loss: 0.0089 Acc: 0.7869
val Loss: 0.0185 Acc: 0.4833

Epoch 62/67
----------
train Loss: 0.0092 Acc: 0.7856
val Loss: 0.0184 Acc: 0.4610

Epoch 63/67
----------
train Loss: 0.0095 Acc: 0.7893
val Loss: 0.0186 Acc: 0.4758

Epoch 64/67
----------
train Loss: 0.0087 Acc: 0.7831
val Loss: 0.0185 Acc: 0.4796

Epoch 65/67
----------
train Loss: 0.0089 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4796

Epoch 66/67
----------
train Loss: 0.0090 Acc: 0.7831
val Loss: 0.0186 Acc: 0.4721

Epoch 67/67
----------
train Loss: 0.0091 Acc: 0.7732
val Loss: 0.0188 Acc: 0.4833

Training complete in 6m 12s
Best val Acc: 0.494424

---Fine tuning.---
Epoch 0/67
----------
LR is set to 0.01
train Loss: 0.0110 Acc: 0.7460
val Loss: 0.0217 Acc: 0.4201

Epoch 1/67
----------
train Loss: 0.0094 Acc: 0.7088
val Loss: 0.0267 Acc: 0.4164

Epoch 2/67
----------
train Loss: 0.0092 Acc: 0.7398
val Loss: 0.0215 Acc: 0.4870

Epoch 3/67
----------
train Loss: 0.0056 Acc: 0.8401
val Loss: 0.0222 Acc: 0.4461

Epoch 4/67
----------
train Loss: 0.0047 Acc: 0.8662
val Loss: 0.0219 Acc: 0.4944

Epoch 5/67
----------
train Loss: 0.0033 Acc: 0.9232
val Loss: 0.0233 Acc: 0.4572

Epoch 6/67
----------
train Loss: 0.0025 Acc: 0.9380
val Loss: 0.0236 Acc: 0.4907

Epoch 7/67
----------
train Loss: 0.0020 Acc: 0.9591
val Loss: 0.0274 Acc: 0.4572

Epoch 8/67
----------
train Loss: 0.0018 Acc: 0.9628
val Loss: 0.0276 Acc: 0.4758

Epoch 9/67
----------
train Loss: 0.0038 Acc: 0.9343
val Loss: 0.0289 Acc: 0.4647

Epoch 10/67
----------
train Loss: 0.0046 Acc: 0.8649
val Loss: 0.0288 Acc: 0.4275

Epoch 11/67
----------
train Loss: 0.0037 Acc: 0.9083
val Loss: 0.0341 Acc: 0.4275

Epoch 12/67
----------
LR is set to 0.001
train Loss: 0.0026 Acc: 0.9318
val Loss: 0.0266 Acc: 0.4758

Epoch 13/67
----------
train Loss: 0.0025 Acc: 0.9480
val Loss: 0.0238 Acc: 0.5316

Epoch 14/67
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0249 Acc: 0.5353

Epoch 15/67
----------
train Loss: 0.0015 Acc: 0.9703
val Loss: 0.0242 Acc: 0.5353

Epoch 16/67
----------
train Loss: 0.0012 Acc: 0.9765
val Loss: 0.0236 Acc: 0.5316

Epoch 17/67
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0237 Acc: 0.5316

Epoch 18/67
----------
train Loss: 0.0010 Acc: 0.9827
val Loss: 0.0231 Acc: 0.5390

Epoch 19/67
----------
train Loss: 0.0010 Acc: 0.9814
val Loss: 0.0229 Acc: 0.5613

Epoch 20/67
----------
train Loss: 0.0014 Acc: 0.9789
val Loss: 0.0237 Acc: 0.5465

Epoch 21/67
----------
train Loss: 0.0009 Acc: 0.9802
val Loss: 0.0246 Acc: 0.5316

Epoch 22/67
----------
train Loss: 0.0009 Acc: 0.9814
val Loss: 0.0233 Acc: 0.5465

Epoch 23/67
----------
train Loss: 0.0009 Acc: 0.9802
val Loss: 0.0233 Acc: 0.5613

Epoch 24/67
----------
LR is set to 0.00010000000000000002
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0237 Acc: 0.5428

Epoch 25/67
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0240 Acc: 0.5428

Epoch 26/67
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0242 Acc: 0.5465

Epoch 27/67
----------
train Loss: 0.0016 Acc: 0.9901
val Loss: 0.0245 Acc: 0.5502

Epoch 28/67
----------
train Loss: 0.0007 Acc: 0.9901
val Loss: 0.0239 Acc: 0.5539

Epoch 29/67
----------
train Loss: 0.0016 Acc: 0.9876
val Loss: 0.0237 Acc: 0.5539

Epoch 30/67
----------
train Loss: 0.0006 Acc: 0.9839
val Loss: 0.0235 Acc: 0.5539

Epoch 31/67
----------
train Loss: 0.0011 Acc: 0.9888
val Loss: 0.0232 Acc: 0.5428

Epoch 32/67
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0233 Acc: 0.5465

Epoch 33/67
----------
train Loss: 0.0008 Acc: 0.9864
val Loss: 0.0237 Acc: 0.5390

Epoch 34/67
----------
train Loss: 0.0008 Acc: 0.9901
val Loss: 0.0234 Acc: 0.5428

Epoch 35/67
----------
train Loss: 0.0008 Acc: 0.9851
val Loss: 0.0229 Acc: 0.5390

Epoch 36/67
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9851
val Loss: 0.0235 Acc: 0.5428

Epoch 37/67
----------
train Loss: 0.0010 Acc: 0.9901
val Loss: 0.0241 Acc: 0.5502

Epoch 38/67
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0233 Acc: 0.5502

Epoch 39/67
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0240 Acc: 0.5465

Epoch 40/67
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0232 Acc: 0.5390

Epoch 41/67
----------
train Loss: 0.0016 Acc: 0.9864
val Loss: 0.0234 Acc: 0.5390

Epoch 42/67
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0228 Acc: 0.5390

Epoch 43/67
----------
train Loss: 0.0011 Acc: 0.9827
val Loss: 0.0231 Acc: 0.5353

Epoch 44/67
----------
train Loss: 0.0008 Acc: 0.9888
val Loss: 0.0233 Acc: 0.5502

Epoch 45/67
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0239 Acc: 0.5502

Epoch 46/67
----------
train Loss: 0.0005 Acc: 0.9901
val Loss: 0.0235 Acc: 0.5539

Epoch 47/67
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0238 Acc: 0.5502

Epoch 48/67
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0019 Acc: 0.9901
val Loss: 0.0245 Acc: 0.5502

Epoch 49/67
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0232 Acc: 0.5390

Epoch 50/67
----------
train Loss: 0.0010 Acc: 0.9839
val Loss: 0.0233 Acc: 0.5539

Epoch 51/67
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0239 Acc: 0.5502

Epoch 52/67
----------
train Loss: 0.0005 Acc: 0.9913
val Loss: 0.0233 Acc: 0.5576

Epoch 53/67
----------
train Loss: 0.0027 Acc: 0.9814
val Loss: 0.0231 Acc: 0.5651

Epoch 54/67
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0231 Acc: 0.5502

Epoch 55/67
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0238 Acc: 0.5390

Epoch 56/67
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0240 Acc: 0.5576

Epoch 57/67
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0238 Acc: 0.5465

Epoch 58/67
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0233 Acc: 0.5613

Epoch 59/67
----------
train Loss: 0.0007 Acc: 0.9851
val Loss: 0.0240 Acc: 0.5576

Epoch 60/67
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0006 Acc: 0.9901
val Loss: 0.0246 Acc: 0.5651

Epoch 61/67
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0240 Acc: 0.5576

Epoch 62/67
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0241 Acc: 0.5613

Epoch 63/67
----------
train Loss: 0.0007 Acc: 0.9864
val Loss: 0.0246 Acc: 0.5502

Epoch 64/67
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0238 Acc: 0.5539

Epoch 65/67
----------
train Loss: 0.0006 Acc: 0.9901
val Loss: 0.0240 Acc: 0.5539

Epoch 66/67
----------
train Loss: 0.0013 Acc: 0.9864
val Loss: 0.0236 Acc: 0.5465

Epoch 67/67
----------
train Loss: 0.0006 Acc: 0.9901
val Loss: 0.0243 Acc: 0.5539

Training complete in 6m 38s
Best val Acc: 0.565056

---Testing---
Test accuracy: 0.882900
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of Bigeye tuna : 77 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 72 %
Accuracy of Little tunny : 92 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 84 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 78 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8403648604007306, std: 0.10535168911114895
--------------------

run info[val: 0.3, epoch: 58, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0273 Acc: 0.1406
val Loss: 0.0293 Acc: 0.1988

Epoch 1/57
----------
train Loss: 0.0229 Acc: 0.3103
val Loss: 0.0260 Acc: 0.3354

Epoch 2/57
----------
train Loss: 0.0188 Acc: 0.4523
val Loss: 0.0243 Acc: 0.3975

Epoch 3/57
----------
train Loss: 0.0161 Acc: 0.5358
val Loss: 0.0218 Acc: 0.3820

Epoch 4/57
----------
train Loss: 0.0148 Acc: 0.5796
val Loss: 0.0224 Acc: 0.3944

Epoch 5/57
----------
train Loss: 0.0139 Acc: 0.6021
val Loss: 0.0217 Acc: 0.4286

Epoch 6/57
----------
LR is set to 0.001
train Loss: 0.0127 Acc: 0.6379
val Loss: 0.0216 Acc: 0.4472

Epoch 7/57
----------
train Loss: 0.0122 Acc: 0.6857
val Loss: 0.0214 Acc: 0.4379

Epoch 8/57
----------
train Loss: 0.0123 Acc: 0.6645
val Loss: 0.0216 Acc: 0.4193

Epoch 9/57
----------
train Loss: 0.0118 Acc: 0.6857
val Loss: 0.0215 Acc: 0.4441

Epoch 10/57
----------
train Loss: 0.0119 Acc: 0.6751
val Loss: 0.0202 Acc: 0.4472

Epoch 11/57
----------
train Loss: 0.0117 Acc: 0.6870
val Loss: 0.0210 Acc: 0.4503

Epoch 12/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0117 Acc: 0.6910
val Loss: 0.0200 Acc: 0.4472

Epoch 13/57
----------
train Loss: 0.0116 Acc: 0.6777
val Loss: 0.0206 Acc: 0.4503

Epoch 14/57
----------
train Loss: 0.0117 Acc: 0.6950
val Loss: 0.0206 Acc: 0.4565

Epoch 15/57
----------
train Loss: 0.0117 Acc: 0.6963
val Loss: 0.0206 Acc: 0.4565

Epoch 16/57
----------
train Loss: 0.0117 Acc: 0.6897
val Loss: 0.0213 Acc: 0.4534

Epoch 17/57
----------
train Loss: 0.0117 Acc: 0.7016
val Loss: 0.0214 Acc: 0.4596

Epoch 18/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0116 Acc: 0.6963
val Loss: 0.0212 Acc: 0.4534

Epoch 19/57
----------
train Loss: 0.0117 Acc: 0.6857
val Loss: 0.0206 Acc: 0.4534

Epoch 20/57
----------
train Loss: 0.0117 Acc: 0.6870
val Loss: 0.0217 Acc: 0.4503

Epoch 21/57
----------
train Loss: 0.0114 Acc: 0.7016
val Loss: 0.0212 Acc: 0.4534

Epoch 22/57
----------
train Loss: 0.0116 Acc: 0.6923
val Loss: 0.0219 Acc: 0.4472

Epoch 23/57
----------
train Loss: 0.0116 Acc: 0.6883
val Loss: 0.0212 Acc: 0.4534

Epoch 24/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0114 Acc: 0.7069
val Loss: 0.0198 Acc: 0.4534

Epoch 25/57
----------
train Loss: 0.0117 Acc: 0.6936
val Loss: 0.0213 Acc: 0.4503

Epoch 26/57
----------
train Loss: 0.0115 Acc: 0.7016
val Loss: 0.0216 Acc: 0.4534

Epoch 27/57
----------
train Loss: 0.0116 Acc: 0.6936
val Loss: 0.0218 Acc: 0.4565

Epoch 28/57
----------
train Loss: 0.0117 Acc: 0.6923
val Loss: 0.0208 Acc: 0.4534

Epoch 29/57
----------
train Loss: 0.0116 Acc: 0.6883
val Loss: 0.0205 Acc: 0.4534

Epoch 30/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0114 Acc: 0.7016
val Loss: 0.0213 Acc: 0.4534

Epoch 31/57
----------
train Loss: 0.0115 Acc: 0.6963
val Loss: 0.0215 Acc: 0.4534

Epoch 32/57
----------
train Loss: 0.0115 Acc: 0.7003
val Loss: 0.0204 Acc: 0.4565

Epoch 33/57
----------
train Loss: 0.0117 Acc: 0.6923
val Loss: 0.0204 Acc: 0.4534

Epoch 34/57
----------
train Loss: 0.0117 Acc: 0.6923
val Loss: 0.0206 Acc: 0.4565

Epoch 35/57
----------
train Loss: 0.0117 Acc: 0.6936
val Loss: 0.0215 Acc: 0.4565

Epoch 36/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0117 Acc: 0.6857
val Loss: 0.0216 Acc: 0.4534

Epoch 37/57
----------
train Loss: 0.0115 Acc: 0.6923
val Loss: 0.0205 Acc: 0.4596

Epoch 38/57
----------
train Loss: 0.0116 Acc: 0.6963
val Loss: 0.0210 Acc: 0.4503

Epoch 39/57
----------
train Loss: 0.0115 Acc: 0.6910
val Loss: 0.0211 Acc: 0.4534

Epoch 40/57
----------
train Loss: 0.0118 Acc: 0.6897
val Loss: 0.0214 Acc: 0.4565

Epoch 41/57
----------
train Loss: 0.0115 Acc: 0.6883
val Loss: 0.0209 Acc: 0.4534

Epoch 42/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0116 Acc: 0.6897
val Loss: 0.0211 Acc: 0.4534

Epoch 43/57
----------
train Loss: 0.0116 Acc: 0.6870
val Loss: 0.0214 Acc: 0.4534

Epoch 44/57
----------
train Loss: 0.0116 Acc: 0.6844
val Loss: 0.0210 Acc: 0.4534

Epoch 45/57
----------
train Loss: 0.0116 Acc: 0.6870
val Loss: 0.0212 Acc: 0.4565

Epoch 46/57
----------
train Loss: 0.0116 Acc: 0.6897
val Loss: 0.0206 Acc: 0.4565

Epoch 47/57
----------
train Loss: 0.0116 Acc: 0.7056
val Loss: 0.0202 Acc: 0.4565

Epoch 48/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0118 Acc: 0.6897
val Loss: 0.0212 Acc: 0.4534

Epoch 49/57
----------
train Loss: 0.0117 Acc: 0.6804
val Loss: 0.0200 Acc: 0.4534

Epoch 50/57
----------
train Loss: 0.0116 Acc: 0.6963
val Loss: 0.0213 Acc: 0.4534

Epoch 51/57
----------
train Loss: 0.0117 Acc: 0.7056
val Loss: 0.0199 Acc: 0.4596

Epoch 52/57
----------
train Loss: 0.0116 Acc: 0.6923
val Loss: 0.0214 Acc: 0.4596

Epoch 53/57
----------
train Loss: 0.0118 Acc: 0.6897
val Loss: 0.0215 Acc: 0.4565

Epoch 54/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0117 Acc: 0.6830
val Loss: 0.0200 Acc: 0.4565

Epoch 55/57
----------
train Loss: 0.0117 Acc: 0.6857
val Loss: 0.0206 Acc: 0.4534

Epoch 56/57
----------
train Loss: 0.0118 Acc: 0.6976
val Loss: 0.0208 Acc: 0.4534

Epoch 57/57
----------
train Loss: 0.0117 Acc: 0.7095
val Loss: 0.0210 Acc: 0.4534

Training complete in 5m 21s
Best val Acc: 0.459627

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0120 Acc: 0.6645
val Loss: 0.0202 Acc: 0.4938

Epoch 1/57
----------
train Loss: 0.0074 Acc: 0.8355
val Loss: 0.0193 Acc: 0.5000

Epoch 2/57
----------
train Loss: 0.0045 Acc: 0.9019
val Loss: 0.0196 Acc: 0.4876

Epoch 3/57
----------
train Loss: 0.0024 Acc: 0.9562
val Loss: 0.0206 Acc: 0.5062

Epoch 4/57
----------
train Loss: 0.0016 Acc: 0.9655
val Loss: 0.0192 Acc: 0.5062

Epoch 5/57
----------
train Loss: 0.0010 Acc: 0.9748
val Loss: 0.0200 Acc: 0.5280

Epoch 6/57
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9867
val Loss: 0.0193 Acc: 0.5311

Epoch 7/57
----------
train Loss: 0.0006 Acc: 0.9854
val Loss: 0.0196 Acc: 0.5280

Epoch 8/57
----------
train Loss: 0.0006 Acc: 0.9894
val Loss: 0.0188 Acc: 0.5248

Epoch 9/57
----------
train Loss: 0.0006 Acc: 0.9920
val Loss: 0.0193 Acc: 0.5280

Epoch 10/57
----------
train Loss: 0.0005 Acc: 0.9947
val Loss: 0.0205 Acc: 0.5280

Epoch 11/57
----------
train Loss: 0.0006 Acc: 0.9867
val Loss: 0.0192 Acc: 0.5311

Epoch 12/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0194 Acc: 0.5342

Epoch 13/57
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0207 Acc: 0.5280

Epoch 14/57
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0201 Acc: 0.5311

Epoch 15/57
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0200 Acc: 0.5311

Epoch 16/57
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0204 Acc: 0.5280

Epoch 17/57
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0185 Acc: 0.5280

Epoch 18/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0198 Acc: 0.5311

Epoch 19/57
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0204 Acc: 0.5311

Epoch 20/57
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0206 Acc: 0.5311

Epoch 21/57
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0204 Acc: 0.5311

Epoch 22/57
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0200 Acc: 0.5311

Epoch 23/57
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0211 Acc: 0.5342

Epoch 24/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0202 Acc: 0.5280

Epoch 25/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0200 Acc: 0.5280

Epoch 26/57
----------
train Loss: 0.0005 Acc: 0.9854
val Loss: 0.0186 Acc: 0.5280

Epoch 27/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0188 Acc: 0.5280

Epoch 28/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0199 Acc: 0.5280

Epoch 29/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0189 Acc: 0.5311

Epoch 30/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0210 Acc: 0.5311

Epoch 31/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0191 Acc: 0.5280

Epoch 32/57
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0193 Acc: 0.5342

Epoch 33/57
----------
train Loss: 0.0005 Acc: 0.9960
val Loss: 0.0200 Acc: 0.5311

Epoch 34/57
----------
train Loss: 0.0005 Acc: 0.9947
val Loss: 0.0210 Acc: 0.5311

Epoch 35/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0199 Acc: 0.5280

Epoch 36/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0195 Acc: 0.5280

Epoch 37/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0200 Acc: 0.5311

Epoch 38/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0189 Acc: 0.5311

Epoch 39/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0202 Acc: 0.5311

Epoch 40/57
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0197 Acc: 0.5280

Epoch 41/57
----------
train Loss: 0.0005 Acc: 0.9934
val Loss: 0.0192 Acc: 0.5280

Epoch 42/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0184 Acc: 0.5280

Epoch 43/57
----------
train Loss: 0.0006 Acc: 0.9920
val Loss: 0.0203 Acc: 0.5280

Epoch 44/57
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0194 Acc: 0.5311

Epoch 45/57
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0180 Acc: 0.5311

Epoch 46/57
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0189 Acc: 0.5311

Epoch 47/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0182 Acc: 0.5280

Epoch 48/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0212 Acc: 0.5311

Epoch 49/57
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0192 Acc: 0.5311

Epoch 50/57
----------
train Loss: 0.0005 Acc: 0.9920
val Loss: 0.0196 Acc: 0.5248

Epoch 51/57
----------
train Loss: 0.0005 Acc: 0.9947
val Loss: 0.0204 Acc: 0.5311

Epoch 52/57
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0211 Acc: 0.5373

Epoch 53/57
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0199 Acc: 0.5311

Epoch 54/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0188 Acc: 0.5311

Epoch 55/57
----------
train Loss: 0.0005 Acc: 0.9947
val Loss: 0.0198 Acc: 0.5280

Epoch 56/57
----------
train Loss: 0.0005 Acc: 0.9947
val Loss: 0.0195 Acc: 0.5311

Epoch 57/57
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0187 Acc: 0.5311

Training complete in 5m 39s
Best val Acc: 0.537267

---Testing---
Test accuracy: 0.855948
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 81 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 78 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 75 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.8063585774100275, std: 0.12603323822473908

Model saved in "./weights/tuna_fish_[0.94]_mean[0.92]_std[0.06].save".
--------------------

run info[val: 0.1, epoch: 61, randcrop: True, decay: 4]

---Training last layer.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0262 Acc: 0.2074
val Loss: 0.0415 Acc: 0.2897

Epoch 1/60
----------
train Loss: 0.0213 Acc: 0.3560
val Loss: 0.0364 Acc: 0.4112

Epoch 2/60
----------
train Loss: 0.0188 Acc: 0.4035
val Loss: 0.0359 Acc: 0.4486

Epoch 3/60
----------
train Loss: 0.0165 Acc: 0.5201
val Loss: 0.0354 Acc: 0.4673

Epoch 4/60
----------
LR is set to 0.001
train Loss: 0.0153 Acc: 0.5408
val Loss: 0.0257 Acc: 0.4393

Epoch 5/60
----------
train Loss: 0.0146 Acc: 0.5624
val Loss: 0.0279 Acc: 0.4673

Epoch 6/60
----------
train Loss: 0.0147 Acc: 0.5655
val Loss: 0.0370 Acc: 0.4953

Epoch 7/60
----------
train Loss: 0.0144 Acc: 0.5924
val Loss: 0.0310 Acc: 0.4579

Epoch 8/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0143 Acc: 0.5748
val Loss: 0.0293 Acc: 0.4486

Epoch 9/60
----------
train Loss: 0.0143 Acc: 0.5841
val Loss: 0.0325 Acc: 0.4579

Epoch 10/60
----------
train Loss: 0.0143 Acc: 0.5800
val Loss: 0.0366 Acc: 0.4579

Epoch 11/60
----------
train Loss: 0.0143 Acc: 0.5862
val Loss: 0.0271 Acc: 0.4579

Epoch 12/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0144 Acc: 0.5666
val Loss: 0.0275 Acc: 0.4673

Epoch 13/60
----------
train Loss: 0.0144 Acc: 0.5851
val Loss: 0.0353 Acc: 0.4579

Epoch 14/60
----------
train Loss: 0.0142 Acc: 0.5882
val Loss: 0.0301 Acc: 0.4579

Epoch 15/60
----------
train Loss: 0.0139 Acc: 0.5862
val Loss: 0.0319 Acc: 0.4673

Epoch 16/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0143 Acc: 0.5666
val Loss: 0.0308 Acc: 0.4673

Epoch 17/60
----------
train Loss: 0.0141 Acc: 0.5872
val Loss: 0.0279 Acc: 0.4579

Epoch 18/60
----------
train Loss: 0.0142 Acc: 0.5769
val Loss: 0.0336 Acc: 0.4579

Epoch 19/60
----------
train Loss: 0.0143 Acc: 0.5686
val Loss: 0.0355 Acc: 0.4673

Epoch 20/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0143 Acc: 0.5779
val Loss: 0.0325 Acc: 0.4766

Epoch 21/60
----------
train Loss: 0.0143 Acc: 0.5789
val Loss: 0.0317 Acc: 0.4579

Epoch 22/60
----------
train Loss: 0.0140 Acc: 0.5748
val Loss: 0.0303 Acc: 0.4766

Epoch 23/60
----------
train Loss: 0.0141 Acc: 0.5789
val Loss: 0.0335 Acc: 0.4766

Epoch 24/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0144 Acc: 0.5676
val Loss: 0.0287 Acc: 0.4673

Epoch 25/60
----------
train Loss: 0.0142 Acc: 0.5728
val Loss: 0.0319 Acc: 0.4673

Epoch 26/60
----------
train Loss: 0.0143 Acc: 0.5614
val Loss: 0.0374 Acc: 0.4673

Epoch 27/60
----------
train Loss: 0.0143 Acc: 0.5789
val Loss: 0.0276 Acc: 0.4579

Epoch 28/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0144 Acc: 0.5573
val Loss: 0.0291 Acc: 0.4579

Epoch 29/60
----------
train Loss: 0.0143 Acc: 0.5686
val Loss: 0.0354 Acc: 0.4673

Epoch 30/60
----------
train Loss: 0.0142 Acc: 0.5789
val Loss: 0.0274 Acc: 0.4673

Epoch 31/60
----------
train Loss: 0.0145 Acc: 0.5635
val Loss: 0.0270 Acc: 0.4579

Epoch 32/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0143 Acc: 0.5955
val Loss: 0.0301 Acc: 0.4579

Epoch 33/60
----------
train Loss: 0.0144 Acc: 0.5779
val Loss: 0.0263 Acc: 0.4673

Epoch 34/60
----------
train Loss: 0.0143 Acc: 0.5717
val Loss: 0.0305 Acc: 0.4579

Epoch 35/60
----------
train Loss: 0.0143 Acc: 0.5820
val Loss: 0.0369 Acc: 0.4579

Epoch 36/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0142 Acc: 0.5831
val Loss: 0.0318 Acc: 0.4766

Epoch 37/60
----------
train Loss: 0.0141 Acc: 0.6006
val Loss: 0.0281 Acc: 0.4673

Epoch 38/60
----------
train Loss: 0.0142 Acc: 0.5820
val Loss: 0.0265 Acc: 0.4766

Epoch 39/60
----------
train Loss: 0.0143 Acc: 0.5552
val Loss: 0.0317 Acc: 0.4766

Epoch 40/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0143 Acc: 0.5810
val Loss: 0.0337 Acc: 0.4766

Epoch 41/60
----------
train Loss: 0.0143 Acc: 0.5810
val Loss: 0.0391 Acc: 0.4766

Epoch 42/60
----------
train Loss: 0.0142 Acc: 0.5810
val Loss: 0.0275 Acc: 0.4766

Epoch 43/60
----------
train Loss: 0.0143 Acc: 0.5717
val Loss: 0.0373 Acc: 0.4766

Epoch 44/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0141 Acc: 0.5851
val Loss: 0.0289 Acc: 0.4766

Epoch 45/60
----------
train Loss: 0.0144 Acc: 0.5955
val Loss: 0.0288 Acc: 0.4673

Epoch 46/60
----------
train Loss: 0.0142 Acc: 0.5769
val Loss: 0.0302 Acc: 0.4673

Epoch 47/60
----------
train Loss: 0.0142 Acc: 0.5831
val Loss: 0.0348 Acc: 0.4766

Epoch 48/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0143 Acc: 0.5872
val Loss: 0.0271 Acc: 0.4579

Epoch 49/60
----------
train Loss: 0.0142 Acc: 0.5882
val Loss: 0.0261 Acc: 0.4579

Epoch 50/60
----------
train Loss: 0.0142 Acc: 0.5717
val Loss: 0.0322 Acc: 0.4579

Epoch 51/60
----------
train Loss: 0.0143 Acc: 0.5728
val Loss: 0.0339 Acc: 0.4579

Epoch 52/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0142 Acc: 0.5800
val Loss: 0.0302 Acc: 0.4579

Epoch 53/60
----------
train Loss: 0.0144 Acc: 0.5820
val Loss: 0.0287 Acc: 0.4673

Epoch 54/60
----------
train Loss: 0.0143 Acc: 0.5872
val Loss: 0.0381 Acc: 0.4673

Epoch 55/60
----------
train Loss: 0.0144 Acc: 0.5831
val Loss: 0.0258 Acc: 0.4579

Epoch 56/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0141 Acc: 0.5779
val Loss: 0.0319 Acc: 0.4673

Epoch 57/60
----------
train Loss: 0.0144 Acc: 0.5759
val Loss: 0.0264 Acc: 0.4673

Epoch 58/60
----------
train Loss: 0.0143 Acc: 0.5759
val Loss: 0.0312 Acc: 0.4579

Epoch 59/60
----------
train Loss: 0.0144 Acc: 0.5851
val Loss: 0.0329 Acc: 0.4579

Epoch 60/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0141 Acc: 0.5872
val Loss: 0.0306 Acc: 0.4766

Training complete in 5m 40s
Best val Acc: 0.495327

---Fine tuning.---
Epoch 0/60
----------
LR is set to 0.01
train Loss: 0.0144 Acc: 0.5542
val Loss: 0.0251 Acc: 0.5607

Epoch 1/60
----------
train Loss: 0.0102 Acc: 0.6956
val Loss: 0.0235 Acc: 0.5607

Epoch 2/60
----------
train Loss: 0.0071 Acc: 0.7926
val Loss: 0.0200 Acc: 0.5327

Epoch 3/60
----------
train Loss: 0.0050 Acc: 0.8762
val Loss: 0.0282 Acc: 0.5701

Epoch 4/60
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.9123
val Loss: 0.0296 Acc: 0.5701

Epoch 5/60
----------
train Loss: 0.0032 Acc: 0.9267
val Loss: 0.0189 Acc: 0.5794

Epoch 6/60
----------
train Loss: 0.0029 Acc: 0.9340
val Loss: 0.0282 Acc: 0.5981

Epoch 7/60
----------
train Loss: 0.0028 Acc: 0.9401
val Loss: 0.0277 Acc: 0.5888

Epoch 8/60
----------
LR is set to 0.00010000000000000002
train Loss: 0.0024 Acc: 0.9577
val Loss: 0.0198 Acc: 0.5794

Epoch 9/60
----------
train Loss: 0.0025 Acc: 0.9556
val Loss: 0.0317 Acc: 0.5701

Epoch 10/60
----------
train Loss: 0.0027 Acc: 0.9463
val Loss: 0.0255 Acc: 0.5701

Epoch 11/60
----------
train Loss: 0.0026 Acc: 0.9474
val Loss: 0.0268 Acc: 0.5701

Epoch 12/60
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0025 Acc: 0.9515
val Loss: 0.0469 Acc: 0.5701

Epoch 13/60
----------
train Loss: 0.0025 Acc: 0.9546
val Loss: 0.0380 Acc: 0.5888

Epoch 14/60
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0162 Acc: 0.5794

Epoch 15/60
----------
train Loss: 0.0026 Acc: 0.9494
val Loss: 0.0374 Acc: 0.5794

Epoch 16/60
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0026 Acc: 0.9453
val Loss: 0.0302 Acc: 0.5701

Epoch 17/60
----------
train Loss: 0.0023 Acc: 0.9680
val Loss: 0.0310 Acc: 0.5794

Epoch 18/60
----------
train Loss: 0.0026 Acc: 0.9525
val Loss: 0.0406 Acc: 0.5794

Epoch 19/60
----------
train Loss: 0.0025 Acc: 0.9515
val Loss: 0.0201 Acc: 0.5794

Epoch 20/60
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0026 Acc: 0.9525
val Loss: 0.0241 Acc: 0.5794

Epoch 21/60
----------
train Loss: 0.0026 Acc: 0.9443
val Loss: 0.0401 Acc: 0.5701

Epoch 22/60
----------
train Loss: 0.0026 Acc: 0.9505
val Loss: 0.0304 Acc: 0.5888

Epoch 23/60
----------
train Loss: 0.0025 Acc: 0.9463
val Loss: 0.0360 Acc: 0.5888

Epoch 24/60
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0025 Acc: 0.9536
val Loss: 0.0298 Acc: 0.5794

Epoch 25/60
----------
train Loss: 0.0026 Acc: 0.9474
val Loss: 0.0271 Acc: 0.5701

Epoch 26/60
----------
train Loss: 0.0027 Acc: 0.9453
val Loss: 0.0182 Acc: 0.5794

Epoch 27/60
----------
train Loss: 0.0026 Acc: 0.9494
val Loss: 0.0247 Acc: 0.5794

Epoch 28/60
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0026 Acc: 0.9463
val Loss: 0.0167 Acc: 0.5701

Epoch 29/60
----------
train Loss: 0.0026 Acc: 0.9474
val Loss: 0.0171 Acc: 0.5794

Epoch 30/60
----------
train Loss: 0.0026 Acc: 0.9556
val Loss: 0.0227 Acc: 0.5701

Epoch 31/60
----------
train Loss: 0.0026 Acc: 0.9525
val Loss: 0.0213 Acc: 0.5701

Epoch 32/60
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0026 Acc: 0.9515
val Loss: 0.0323 Acc: 0.5794

Epoch 33/60
----------
train Loss: 0.0026 Acc: 0.9453
val Loss: 0.0285 Acc: 0.5701

Epoch 34/60
----------
train Loss: 0.0025 Acc: 0.9567
val Loss: 0.0261 Acc: 0.5888

Epoch 35/60
----------
train Loss: 0.0028 Acc: 0.9329
val Loss: 0.0274 Acc: 0.5794

Epoch 36/60
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0025 Acc: 0.9556
val Loss: 0.0305 Acc: 0.5888

Epoch 37/60
----------
train Loss: 0.0026 Acc: 0.9494
val Loss: 0.0406 Acc: 0.5794

Epoch 38/60
----------
train Loss: 0.0026 Acc: 0.9484
val Loss: 0.0370 Acc: 0.5701

Epoch 39/60
----------
train Loss: 0.0026 Acc: 0.9474
val Loss: 0.0332 Acc: 0.5888

Epoch 40/60
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0027 Acc: 0.9443
val Loss: 0.0273 Acc: 0.5701

Epoch 41/60
----------
train Loss: 0.0027 Acc: 0.9494
val Loss: 0.0328 Acc: 0.5701

Epoch 42/60
----------
train Loss: 0.0026 Acc: 0.9432
val Loss: 0.0207 Acc: 0.5701

Epoch 43/60
----------
train Loss: 0.0027 Acc: 0.9484
val Loss: 0.0202 Acc: 0.5888

Epoch 44/60
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0025 Acc: 0.9536
val Loss: 0.0298 Acc: 0.5888

Epoch 45/60
----------
train Loss: 0.0026 Acc: 0.9546
val Loss: 0.0274 Acc: 0.5794

Epoch 46/60
----------
train Loss: 0.0026 Acc: 0.9536
val Loss: 0.0370 Acc: 0.5794

Epoch 47/60
----------
train Loss: 0.0026 Acc: 0.9505
val Loss: 0.0346 Acc: 0.5794

Epoch 48/60
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0026 Acc: 0.9536
val Loss: 0.0302 Acc: 0.5701

Epoch 49/60
----------
train Loss: 0.0025 Acc: 0.9608
val Loss: 0.0199 Acc: 0.5888

Epoch 50/60
----------
train Loss: 0.0025 Acc: 0.9484
val Loss: 0.0340 Acc: 0.5794

Epoch 51/60
----------
train Loss: 0.0026 Acc: 0.9536
val Loss: 0.0287 Acc: 0.5794

Epoch 52/60
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0026 Acc: 0.9484
val Loss: 0.0267 Acc: 0.5888

Epoch 53/60
----------
train Loss: 0.0027 Acc: 0.9463
val Loss: 0.0231 Acc: 0.5888

Epoch 54/60
----------
train Loss: 0.0025 Acc: 0.9484
val Loss: 0.0255 Acc: 0.5888

Epoch 55/60
----------
train Loss: 0.0025 Acc: 0.9515
val Loss: 0.0271 Acc: 0.5888

Epoch 56/60
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0025 Acc: 0.9505
val Loss: 0.0364 Acc: 0.5888

Epoch 57/60
----------
train Loss: 0.0027 Acc: 0.9505
val Loss: 0.0212 Acc: 0.5794

Epoch 58/60
----------
train Loss: 0.0026 Acc: 0.9515
val Loss: 0.0246 Acc: 0.5981

Epoch 59/60
----------
train Loss: 0.0025 Acc: 0.9525
val Loss: 0.0257 Acc: 0.5888

Epoch 60/60
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0026 Acc: 0.9422
val Loss: 0.0247 Acc: 0.5794

Training complete in 6m 5s
Best val Acc: 0.598131

---Testing---
Test accuracy: 0.924721
--------------------
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 85 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 84 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 50 %
Accuracy of Southern bluefin tuna : 86 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.8824968953873767, std: 0.12190261217106955
--------------------

run info[val: 0.15, epoch: 76, randcrop: False, decay: 14]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0286 Acc: 0.1694
val Loss: 0.0276 Acc: 0.2484

Epoch 1/75
----------
train Loss: 0.0228 Acc: 0.3355
val Loss: 0.0254 Acc: 0.2422

Epoch 2/75
----------
train Loss: 0.0196 Acc: 0.4186
val Loss: 0.0244 Acc: 0.3602

Epoch 3/75
----------
train Loss: 0.0172 Acc: 0.5016
val Loss: 0.0233 Acc: 0.3540

Epoch 4/75
----------
train Loss: 0.0155 Acc: 0.5399
val Loss: 0.0230 Acc: 0.4099

Epoch 5/75
----------
train Loss: 0.0143 Acc: 0.5913
val Loss: 0.0234 Acc: 0.4099

Epoch 6/75
----------
train Loss: 0.0136 Acc: 0.5913
val Loss: 0.0229 Acc: 0.4161

Epoch 7/75
----------
train Loss: 0.0135 Acc: 0.6317
val Loss: 0.0241 Acc: 0.4286

Epoch 8/75
----------
train Loss: 0.0126 Acc: 0.6317
val Loss: 0.0238 Acc: 0.3975

Epoch 9/75
----------
train Loss: 0.0120 Acc: 0.6536
val Loss: 0.0220 Acc: 0.4224

Epoch 10/75
----------
train Loss: 0.0101 Acc: 0.7027
val Loss: 0.0232 Acc: 0.3975

Epoch 11/75
----------
train Loss: 0.0101 Acc: 0.7137
val Loss: 0.0229 Acc: 0.4224

Epoch 12/75
----------
train Loss: 0.0106 Acc: 0.7016
val Loss: 0.0232 Acc: 0.4596

Epoch 13/75
----------
train Loss: 0.0097 Acc: 0.7388
val Loss: 0.0227 Acc: 0.4534

Epoch 14/75
----------
LR is set to 0.001
train Loss: 0.0094 Acc: 0.7213
val Loss: 0.0223 Acc: 0.4658

Epoch 15/75
----------
train Loss: 0.0084 Acc: 0.7956
val Loss: 0.0222 Acc: 0.4720

Epoch 16/75
----------
train Loss: 0.0086 Acc: 0.7923
val Loss: 0.0219 Acc: 0.4658

Epoch 17/75
----------
train Loss: 0.0084 Acc: 0.7978
val Loss: 0.0220 Acc: 0.4658

Epoch 18/75
----------
train Loss: 0.0090 Acc: 0.7956
val Loss: 0.0212 Acc: 0.4783

Epoch 19/75
----------
train Loss: 0.0083 Acc: 0.8077
val Loss: 0.0221 Acc: 0.4720

Epoch 20/75
----------
train Loss: 0.0083 Acc: 0.8055
val Loss: 0.0224 Acc: 0.4720

Epoch 21/75
----------
train Loss: 0.0087 Acc: 0.7967
val Loss: 0.0216 Acc: 0.4534

Epoch 22/75
----------
train Loss: 0.0082 Acc: 0.8044
val Loss: 0.0218 Acc: 0.4658

Epoch 23/75
----------
train Loss: 0.0088 Acc: 0.8087
val Loss: 0.0225 Acc: 0.4720

Epoch 24/75
----------
train Loss: 0.0083 Acc: 0.8175
val Loss: 0.0228 Acc: 0.4845

Epoch 25/75
----------
train Loss: 0.0081 Acc: 0.8098
val Loss: 0.0222 Acc: 0.4658

Epoch 26/75
----------
train Loss: 0.0084 Acc: 0.8120
val Loss: 0.0225 Acc: 0.4907

Epoch 27/75
----------
train Loss: 0.0088 Acc: 0.8164
val Loss: 0.0216 Acc: 0.4658

Epoch 28/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0085 Acc: 0.8055
val Loss: 0.0210 Acc: 0.4720

Epoch 29/75
----------
train Loss: 0.0081 Acc: 0.8197
val Loss: 0.0219 Acc: 0.4720

Epoch 30/75
----------
train Loss: 0.0081 Acc: 0.8131
val Loss: 0.0221 Acc: 0.4534

Epoch 31/75
----------
train Loss: 0.0085 Acc: 0.7989
val Loss: 0.0217 Acc: 0.4720

Epoch 32/75
----------
train Loss: 0.0080 Acc: 0.8120
val Loss: 0.0223 Acc: 0.4783

Epoch 33/75
----------
train Loss: 0.0087 Acc: 0.8098
val Loss: 0.0226 Acc: 0.4658

Epoch 34/75
----------
train Loss: 0.0081 Acc: 0.8164
val Loss: 0.0219 Acc: 0.4596

Epoch 35/75
----------
train Loss: 0.0084 Acc: 0.8066
val Loss: 0.0223 Acc: 0.4783

Epoch 36/75
----------
train Loss: 0.0083 Acc: 0.8295
val Loss: 0.0222 Acc: 0.4720

Epoch 37/75
----------
train Loss: 0.0084 Acc: 0.8066
val Loss: 0.0224 Acc: 0.4658

Epoch 38/75
----------
train Loss: 0.0078 Acc: 0.8098
val Loss: 0.0214 Acc: 0.4596

Epoch 39/75
----------
train Loss: 0.0080 Acc: 0.8153
val Loss: 0.0220 Acc: 0.4720

Epoch 40/75
----------
train Loss: 0.0082 Acc: 0.8208
val Loss: 0.0216 Acc: 0.4720

Epoch 41/75
----------
train Loss: 0.0083 Acc: 0.8087
val Loss: 0.0220 Acc: 0.4596

Epoch 42/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0080 Acc: 0.8186
val Loss: 0.0216 Acc: 0.4720

Epoch 43/75
----------
train Loss: 0.0083 Acc: 0.8219
val Loss: 0.0222 Acc: 0.4720

Epoch 44/75
----------
train Loss: 0.0084 Acc: 0.8175
val Loss: 0.0226 Acc: 0.4783

Epoch 45/75
----------
train Loss: 0.0081 Acc: 0.8186
val Loss: 0.0217 Acc: 0.4720

Epoch 46/75
----------
train Loss: 0.0084 Acc: 0.8131
val Loss: 0.0228 Acc: 0.4596

Epoch 47/75
----------
train Loss: 0.0080 Acc: 0.8109
val Loss: 0.0217 Acc: 0.4845

Epoch 48/75
----------
train Loss: 0.0080 Acc: 0.8197
val Loss: 0.0222 Acc: 0.4720

Epoch 49/75
----------
train Loss: 0.0083 Acc: 0.8142
val Loss: 0.0224 Acc: 0.4720

Epoch 50/75
----------
train Loss: 0.0082 Acc: 0.8044
val Loss: 0.0222 Acc: 0.4596

Epoch 51/75
----------
train Loss: 0.0082 Acc: 0.8033
val Loss: 0.0221 Acc: 0.4720

Epoch 52/75
----------
train Loss: 0.0083 Acc: 0.8175
val Loss: 0.0219 Acc: 0.4658

Epoch 53/75
----------
train Loss: 0.0081 Acc: 0.8164
val Loss: 0.0227 Acc: 0.4720

Epoch 54/75
----------
train Loss: 0.0091 Acc: 0.8000
val Loss: 0.0221 Acc: 0.4783

Epoch 55/75
----------
train Loss: 0.0081 Acc: 0.7989
val Loss: 0.0218 Acc: 0.4907

Epoch 56/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0083 Acc: 0.7967
val Loss: 0.0217 Acc: 0.4783

Epoch 57/75
----------
train Loss: 0.0082 Acc: 0.8240
val Loss: 0.0219 Acc: 0.4720

Epoch 58/75
----------
train Loss: 0.0076 Acc: 0.8251
val Loss: 0.0220 Acc: 0.4720

Epoch 59/75
----------
train Loss: 0.0080 Acc: 0.8109
val Loss: 0.0221 Acc: 0.4783

Epoch 60/75
----------
train Loss: 0.0080 Acc: 0.8087
val Loss: 0.0217 Acc: 0.4783

Epoch 61/75
----------
train Loss: 0.0079 Acc: 0.8175
val Loss: 0.0220 Acc: 0.4845

Epoch 62/75
----------
train Loss: 0.0081 Acc: 0.8077
val Loss: 0.0215 Acc: 0.4658

Epoch 63/75
----------
train Loss: 0.0079 Acc: 0.8230
val Loss: 0.0219 Acc: 0.4783

Epoch 64/75
----------
train Loss: 0.0078 Acc: 0.8251
val Loss: 0.0215 Acc: 0.4658

Epoch 65/75
----------
train Loss: 0.0081 Acc: 0.8197
val Loss: 0.0224 Acc: 0.4720

Epoch 66/75
----------
train Loss: 0.0082 Acc: 0.8240
val Loss: 0.0219 Acc: 0.4596

Epoch 67/75
----------
train Loss: 0.0079 Acc: 0.8164
val Loss: 0.0223 Acc: 0.4845

Epoch 68/75
----------
train Loss: 0.0081 Acc: 0.8164
val Loss: 0.0217 Acc: 0.4720

Epoch 69/75
----------
train Loss: 0.0084 Acc: 0.8044
val Loss: 0.0219 Acc: 0.4658

Epoch 70/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0081 Acc: 0.8219
val Loss: 0.0216 Acc: 0.4783

Epoch 71/75
----------
train Loss: 0.0078 Acc: 0.8317
val Loss: 0.0227 Acc: 0.4907

Epoch 72/75
----------
train Loss: 0.0081 Acc: 0.8219
val Loss: 0.0224 Acc: 0.4658

Epoch 73/75
----------
train Loss: 0.0082 Acc: 0.8098
val Loss: 0.0218 Acc: 0.4658

Epoch 74/75
----------
train Loss: 0.0087 Acc: 0.8022
val Loss: 0.0221 Acc: 0.4658

Epoch 75/75
----------
train Loss: 0.0082 Acc: 0.7902
val Loss: 0.0220 Acc: 0.4658

Training complete in 7m 1s
Best val Acc: 0.490683

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0092 Acc: 0.7683
val Loss: 0.0225 Acc: 0.4472

Epoch 1/75
----------
train Loss: 0.0067 Acc: 0.8109
val Loss: 0.0290 Acc: 0.4161

Epoch 2/75
----------
train Loss: 0.0048 Acc: 0.8710
val Loss: 0.0250 Acc: 0.3975

Epoch 3/75
----------
train Loss: 0.0033 Acc: 0.9202
val Loss: 0.0247 Acc: 0.4720

Epoch 4/75
----------
train Loss: 0.0022 Acc: 0.9410
val Loss: 0.0239 Acc: 0.4658

Epoch 5/75
----------
train Loss: 0.0013 Acc: 0.9705
val Loss: 0.0234 Acc: 0.4907

Epoch 6/75
----------
train Loss: 0.0015 Acc: 0.9639
val Loss: 0.0223 Acc: 0.4969

Epoch 7/75
----------
train Loss: 0.0014 Acc: 0.9683
val Loss: 0.0244 Acc: 0.5031

Epoch 8/75
----------
train Loss: 0.0012 Acc: 0.9716
val Loss: 0.0232 Acc: 0.4658

Epoch 9/75
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0227 Acc: 0.4969

Epoch 10/75
----------
train Loss: 0.0009 Acc: 0.9781
val Loss: 0.0248 Acc: 0.4783

Epoch 11/75
----------
train Loss: 0.0008 Acc: 0.9770
val Loss: 0.0261 Acc: 0.4720

Epoch 12/75
----------
train Loss: 0.0009 Acc: 0.9770
val Loss: 0.0237 Acc: 0.5093

Epoch 13/75
----------
train Loss: 0.0010 Acc: 0.9749
val Loss: 0.0239 Acc: 0.5404

Epoch 14/75
----------
LR is set to 0.001
train Loss: 0.0009 Acc: 0.9792
val Loss: 0.0224 Acc: 0.5404

Epoch 15/75
----------
train Loss: 0.0006 Acc: 0.9792
val Loss: 0.0234 Acc: 0.5590

Epoch 16/75
----------
train Loss: 0.0006 Acc: 0.9760
val Loss: 0.0234 Acc: 0.5528

Epoch 17/75
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0235 Acc: 0.5466

Epoch 18/75
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0239 Acc: 0.5404

Epoch 19/75
----------
train Loss: 0.0009 Acc: 0.9803
val Loss: 0.0244 Acc: 0.5342

Epoch 20/75
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0257 Acc: 0.5342

Epoch 21/75
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0250 Acc: 0.5280

Epoch 22/75
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0252 Acc: 0.5217

Epoch 23/75
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0247 Acc: 0.5404

Epoch 24/75
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0241 Acc: 0.5466

Epoch 25/75
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0241 Acc: 0.5404

Epoch 26/75
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0247 Acc: 0.5342

Epoch 27/75
----------
train Loss: 0.0003 Acc: 0.9858
val Loss: 0.0238 Acc: 0.5342

Epoch 28/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0241 Acc: 0.5404

Epoch 29/75
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0246 Acc: 0.5342

Epoch 30/75
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0246 Acc: 0.5280

Epoch 31/75
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0239 Acc: 0.5217

Epoch 32/75
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0240 Acc: 0.5342

Epoch 33/75
----------
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0238 Acc: 0.5217

Epoch 34/75
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0237 Acc: 0.5280

Epoch 35/75
----------
train Loss: 0.0003 Acc: 0.9814
val Loss: 0.0234 Acc: 0.5342

Epoch 36/75
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0244 Acc: 0.5217

Epoch 37/75
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0244 Acc: 0.5093

Epoch 38/75
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0241 Acc: 0.5342

Epoch 39/75
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0253 Acc: 0.5280

Epoch 40/75
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0244 Acc: 0.5217

Epoch 41/75
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0246 Acc: 0.5342

Epoch 42/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5155

Epoch 43/75
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0248 Acc: 0.5280

Epoch 44/75
----------
train Loss: 0.0003 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5280

Epoch 45/75
----------
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0244 Acc: 0.5342

Epoch 46/75
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0243 Acc: 0.5217

Epoch 47/75
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0242 Acc: 0.5280

Epoch 48/75
----------
train Loss: 0.0003 Acc: 0.9836
val Loss: 0.0244 Acc: 0.5404

Epoch 49/75
----------
train Loss: 0.0003 Acc: 0.9858
val Loss: 0.0236 Acc: 0.5342

Epoch 50/75
----------
train Loss: 0.0003 Acc: 0.9836
val Loss: 0.0242 Acc: 0.5280

Epoch 51/75
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0246 Acc: 0.5342

Epoch 52/75
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0238 Acc: 0.5342

Epoch 53/75
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0240 Acc: 0.5217

Epoch 54/75
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0239 Acc: 0.5404

Epoch 55/75
----------
train Loss: 0.0004 Acc: 0.9792
val Loss: 0.0238 Acc: 0.5217

Epoch 56/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0236 Acc: 0.5217

Epoch 57/75
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0251 Acc: 0.5217

Epoch 58/75
----------
train Loss: 0.0004 Acc: 0.9781
val Loss: 0.0232 Acc: 0.5342

Epoch 59/75
----------
train Loss: 0.0004 Acc: 0.9792
val Loss: 0.0242 Acc: 0.5342

Epoch 60/75
----------
train Loss: 0.0003 Acc: 0.9847
val Loss: 0.0245 Acc: 0.5155

Epoch 61/75
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0254 Acc: 0.5093

Epoch 62/75
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0252 Acc: 0.5217

Epoch 63/75
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0259 Acc: 0.5031

Epoch 64/75
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0246 Acc: 0.5217

Epoch 65/75
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5155

Epoch 66/75
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0253 Acc: 0.5217

Epoch 67/75
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0227 Acc: 0.5280

Epoch 68/75
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0250 Acc: 0.5217

Epoch 69/75
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0242 Acc: 0.5155

Epoch 70/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0257 Acc: 0.5217

Epoch 71/75
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5217

Epoch 72/75
----------
train Loss: 0.0003 Acc: 0.9836
val Loss: 0.0245 Acc: 0.5217

Epoch 73/75
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0246 Acc: 0.5217

Epoch 74/75
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0241 Acc: 0.5280

Epoch 75/75
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0254 Acc: 0.5342

Training complete in 7m 30s
Best val Acc: 0.559006

---Testing---
Test accuracy: 0.920074
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 89 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 89 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 90 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.900602401661474, std: 0.0568498561472144
--------------------

run info[val: 0.2, epoch: 56, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0276 Acc: 0.1533
val Loss: 0.0320 Acc: 0.2279

Epoch 1/55
----------
train Loss: 0.0224 Acc: 0.2997
val Loss: 0.0272 Acc: 0.3628

Epoch 2/55
----------
train Loss: 0.0194 Acc: 0.4042
val Loss: 0.0278 Acc: 0.3721

Epoch 3/55
----------
train Loss: 0.0172 Acc: 0.4715
val Loss: 0.0244 Acc: 0.3907

Epoch 4/55
----------
train Loss: 0.0150 Acc: 0.5563
val Loss: 0.0241 Acc: 0.4326

Epoch 5/55
----------
train Loss: 0.0141 Acc: 0.5807
val Loss: 0.0237 Acc: 0.4372

Epoch 6/55
----------
train Loss: 0.0135 Acc: 0.5819
val Loss: 0.0228 Acc: 0.4233

Epoch 7/55
----------
train Loss: 0.0125 Acc: 0.6190
val Loss: 0.0239 Acc: 0.4512

Epoch 8/55
----------
train Loss: 0.0123 Acc: 0.6400
val Loss: 0.0230 Acc: 0.4698

Epoch 9/55
----------
train Loss: 0.0115 Acc: 0.6736
val Loss: 0.0228 Acc: 0.4512

Epoch 10/55
----------
train Loss: 0.0115 Acc: 0.6690
val Loss: 0.0256 Acc: 0.4837

Epoch 11/55
----------
train Loss: 0.0111 Acc: 0.6725
val Loss: 0.0232 Acc: 0.4512

Epoch 12/55
----------
train Loss: 0.0106 Acc: 0.6969
val Loss: 0.0230 Acc: 0.4884

Epoch 13/55
----------
LR is set to 0.001
train Loss: 0.0102 Acc: 0.7143
val Loss: 0.0201 Acc: 0.4837

Epoch 14/55
----------
train Loss: 0.0097 Acc: 0.7294
val Loss: 0.0236 Acc: 0.4884

Epoch 15/55
----------
train Loss: 0.0101 Acc: 0.7224
val Loss: 0.0234 Acc: 0.5070

Epoch 16/55
----------
train Loss: 0.0099 Acc: 0.7375
val Loss: 0.0215 Acc: 0.4930

Epoch 17/55
----------
train Loss: 0.0100 Acc: 0.7294
val Loss: 0.0227 Acc: 0.4977

Epoch 18/55
----------
train Loss: 0.0097 Acc: 0.7329
val Loss: 0.0216 Acc: 0.5023

Epoch 19/55
----------
train Loss: 0.0097 Acc: 0.7352
val Loss: 0.0233 Acc: 0.5023

Epoch 20/55
----------
train Loss: 0.0097 Acc: 0.7561
val Loss: 0.0215 Acc: 0.5023

Epoch 21/55
----------
train Loss: 0.0099 Acc: 0.7305
val Loss: 0.0227 Acc: 0.4977

Epoch 22/55
----------
train Loss: 0.0097 Acc: 0.7247
val Loss: 0.0256 Acc: 0.5070

Epoch 23/55
----------
train Loss: 0.0095 Acc: 0.7282
val Loss: 0.0210 Acc: 0.4977

Epoch 24/55
----------
train Loss: 0.0095 Acc: 0.7259
val Loss: 0.0232 Acc: 0.5023

Epoch 25/55
----------
train Loss: 0.0095 Acc: 0.7329
val Loss: 0.0248 Acc: 0.5163

Epoch 26/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0099 Acc: 0.7375
val Loss: 0.0239 Acc: 0.5023

Epoch 27/55
----------
train Loss: 0.0098 Acc: 0.7271
val Loss: 0.0219 Acc: 0.5070

Epoch 28/55
----------
train Loss: 0.0095 Acc: 0.7468
val Loss: 0.0221 Acc: 0.5023

Epoch 29/55
----------
train Loss: 0.0094 Acc: 0.7526
val Loss: 0.0240 Acc: 0.5116

Epoch 30/55
----------
train Loss: 0.0095 Acc: 0.7340
val Loss: 0.0241 Acc: 0.5070

Epoch 31/55
----------
train Loss: 0.0096 Acc: 0.7329
val Loss: 0.0237 Acc: 0.5070

Epoch 32/55
----------
train Loss: 0.0093 Acc: 0.7422
val Loss: 0.0253 Acc: 0.5070

Epoch 33/55
----------
train Loss: 0.0096 Acc: 0.7398
val Loss: 0.0249 Acc: 0.5070

Epoch 34/55
----------
train Loss: 0.0094 Acc: 0.7561
val Loss: 0.0202 Acc: 0.5070

Epoch 35/55
----------
train Loss: 0.0096 Acc: 0.7468
val Loss: 0.0224 Acc: 0.5070

Epoch 36/55
----------
train Loss: 0.0096 Acc: 0.7573
val Loss: 0.0224 Acc: 0.5070

Epoch 37/55
----------
train Loss: 0.0098 Acc: 0.7305
val Loss: 0.0213 Acc: 0.5116

Epoch 38/55
----------
train Loss: 0.0094 Acc: 0.7340
val Loss: 0.0242 Acc: 0.5070

Epoch 39/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0095 Acc: 0.7468
val Loss: 0.0230 Acc: 0.4977

Epoch 40/55
----------
train Loss: 0.0098 Acc: 0.7247
val Loss: 0.0238 Acc: 0.4977

Epoch 41/55
----------
train Loss: 0.0093 Acc: 0.7364
val Loss: 0.0218 Acc: 0.5023

Epoch 42/55
----------
train Loss: 0.0097 Acc: 0.7340
val Loss: 0.0239 Acc: 0.5023

Epoch 43/55
----------
train Loss: 0.0097 Acc: 0.7271
val Loss: 0.0237 Acc: 0.5070

Epoch 44/55
----------
train Loss: 0.0097 Acc: 0.7282
val Loss: 0.0252 Acc: 0.5116

Epoch 45/55
----------
train Loss: 0.0093 Acc: 0.7596
val Loss: 0.0268 Acc: 0.5116

Epoch 46/55
----------
train Loss: 0.0094 Acc: 0.7410
val Loss: 0.0224 Acc: 0.5070

Epoch 47/55
----------
train Loss: 0.0094 Acc: 0.7503
val Loss: 0.0228 Acc: 0.5023

Epoch 48/55
----------
train Loss: 0.0095 Acc: 0.7282
val Loss: 0.0227 Acc: 0.5070

Epoch 49/55
----------
train Loss: 0.0093 Acc: 0.7538
val Loss: 0.0230 Acc: 0.5070

Epoch 50/55
----------
train Loss: 0.0095 Acc: 0.7480
val Loss: 0.0236 Acc: 0.5070

Epoch 51/55
----------
train Loss: 0.0095 Acc: 0.7456
val Loss: 0.0224 Acc: 0.5070

Epoch 52/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0096 Acc: 0.7317
val Loss: 0.0208 Acc: 0.5070

Epoch 53/55
----------
train Loss: 0.0096 Acc: 0.7352
val Loss: 0.0212 Acc: 0.5070

Epoch 54/55
----------
train Loss: 0.0097 Acc: 0.7224
val Loss: 0.0217 Acc: 0.5116

Epoch 55/55
----------
train Loss: 0.0096 Acc: 0.7387
val Loss: 0.0221 Acc: 0.5023

Training complete in 4m 58s
Best val Acc: 0.516279

---Fine tuning.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0101 Acc: 0.7050
val Loss: 0.0208 Acc: 0.5023

Epoch 1/55
----------
train Loss: 0.0069 Acc: 0.8026
val Loss: 0.0246 Acc: 0.4930

Epoch 2/55
----------
train Loss: 0.0046 Acc: 0.8804
val Loss: 0.0225 Acc: 0.4884

Epoch 3/55
----------
train Loss: 0.0030 Acc: 0.9257
val Loss: 0.0231 Acc: 0.5349

Epoch 4/55
----------
train Loss: 0.0023 Acc: 0.9443
val Loss: 0.0257 Acc: 0.5116

Epoch 5/55
----------
train Loss: 0.0018 Acc: 0.9477
val Loss: 0.0230 Acc: 0.5163

Epoch 6/55
----------
train Loss: 0.0014 Acc: 0.9675
val Loss: 0.0247 Acc: 0.5163

Epoch 7/55
----------
train Loss: 0.0012 Acc: 0.9710
val Loss: 0.0251 Acc: 0.5256

Epoch 8/55
----------
train Loss: 0.0010 Acc: 0.9768
val Loss: 0.0248 Acc: 0.5442

Epoch 9/55
----------
train Loss: 0.0009 Acc: 0.9768
val Loss: 0.0219 Acc: 0.5674

Epoch 10/55
----------
train Loss: 0.0010 Acc: 0.9744
val Loss: 0.0212 Acc: 0.5721

Epoch 11/55
----------
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0249 Acc: 0.5814

Epoch 12/55
----------
train Loss: 0.0008 Acc: 0.9768
val Loss: 0.0231 Acc: 0.5628

Epoch 13/55
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5628

Epoch 14/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0231 Acc: 0.5535

Epoch 15/55
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0226 Acc: 0.5581

Epoch 16/55
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0216 Acc: 0.5628

Epoch 17/55
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0263 Acc: 0.5628

Epoch 18/55
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0248 Acc: 0.5535

Epoch 19/55
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0247 Acc: 0.5581

Epoch 20/55
----------
train Loss: 0.0006 Acc: 0.9826
val Loss: 0.0225 Acc: 0.5628

Epoch 21/55
----------
train Loss: 0.0004 Acc: 0.9919
val Loss: 0.0269 Acc: 0.5721

Epoch 22/55
----------
train Loss: 0.0006 Acc: 0.9837
val Loss: 0.0274 Acc: 0.5767

Epoch 23/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0251 Acc: 0.5907

Epoch 24/55
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0249 Acc: 0.5814

Epoch 25/55
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0234 Acc: 0.5767

Epoch 26/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0262 Acc: 0.5814

Epoch 27/55
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0280 Acc: 0.5814

Epoch 28/55
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0287 Acc: 0.5767

Epoch 29/55
----------
train Loss: 0.0005 Acc: 0.9768
val Loss: 0.0250 Acc: 0.5814

Epoch 30/55
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0273 Acc: 0.5767

Epoch 31/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0276 Acc: 0.5721

Epoch 32/55
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0271 Acc: 0.5721

Epoch 33/55
----------
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0242 Acc: 0.5860

Epoch 34/55
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0293 Acc: 0.5767

Epoch 35/55
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0242 Acc: 0.5814

Epoch 36/55
----------
train Loss: 0.0005 Acc: 0.9779
val Loss: 0.0253 Acc: 0.5814

Epoch 37/55
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0255 Acc: 0.5767

Epoch 38/55
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0229 Acc: 0.5814

Epoch 39/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9779
val Loss: 0.0239 Acc: 0.5814

Epoch 40/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0315 Acc: 0.5860

Epoch 41/55
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0238 Acc: 0.5767

Epoch 42/55
----------
train Loss: 0.0004 Acc: 0.9849
val Loss: 0.0216 Acc: 0.5674

Epoch 43/55
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0259 Acc: 0.5814

Epoch 44/55
----------
train Loss: 0.0005 Acc: 0.9779
val Loss: 0.0258 Acc: 0.5814

Epoch 45/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0277 Acc: 0.5907

Epoch 46/55
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0246 Acc: 0.5814

Epoch 47/55
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0252 Acc: 0.5860

Epoch 48/55
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0260 Acc: 0.5814

Epoch 49/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0239 Acc: 0.5814

Epoch 50/55
----------
train Loss: 0.0004 Acc: 0.9849
val Loss: 0.0229 Acc: 0.5814

Epoch 51/55
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0244 Acc: 0.5814

Epoch 52/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 0.9826
val Loss: 0.0207 Acc: 0.5767

Epoch 53/55
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0246 Acc: 0.5767

Epoch 54/55
----------
train Loss: 0.0004 Acc: 0.9895
val Loss: 0.0263 Acc: 0.5721

Epoch 55/55
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0290 Acc: 0.5721

Training complete in 5m 18s
Best val Acc: 0.590698

---Testing---
Test accuracy: 0.907063
--------------------
Accuracy of Albacore tuna : 87 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 89 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 95 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 80 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.8740050353279513, std: 0.08607551444211213
--------------------

run info[val: 0.25, epoch: 88, randcrop: False, decay: 13]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0287 Acc: 0.1512
val Loss: 0.0262 Acc: 0.2602

Epoch 1/87
----------
train Loss: 0.0239 Acc: 0.3123
val Loss: 0.0237 Acc: 0.3011

Epoch 2/87
----------
train Loss: 0.0199 Acc: 0.4015
val Loss: 0.0224 Acc: 0.3903

Epoch 3/87
----------
train Loss: 0.0194 Acc: 0.4535
val Loss: 0.0205 Acc: 0.3643

Epoch 4/87
----------
train Loss: 0.0176 Acc: 0.4969
val Loss: 0.0212 Acc: 0.4126

Epoch 5/87
----------
train Loss: 0.0156 Acc: 0.5353
val Loss: 0.0208 Acc: 0.4498

Epoch 6/87
----------
train Loss: 0.0140 Acc: 0.6109
val Loss: 0.0209 Acc: 0.4089

Epoch 7/87
----------
train Loss: 0.0148 Acc: 0.5787
val Loss: 0.0242 Acc: 0.3755

Epoch 8/87
----------
train Loss: 0.0153 Acc: 0.5477
val Loss: 0.0217 Acc: 0.4052

Epoch 9/87
----------
train Loss: 0.0127 Acc: 0.6146
val Loss: 0.0205 Acc: 0.4572

Epoch 10/87
----------
train Loss: 0.0128 Acc: 0.6753
val Loss: 0.0215 Acc: 0.4275

Epoch 11/87
----------
train Loss: 0.0116 Acc: 0.6803
val Loss: 0.0214 Acc: 0.4461

Epoch 12/87
----------
train Loss: 0.0118 Acc: 0.6568
val Loss: 0.0203 Acc: 0.4610

Epoch 13/87
----------
LR is set to 0.001
train Loss: 0.0098 Acc: 0.7224
val Loss: 0.0195 Acc: 0.4758

Epoch 14/87
----------
train Loss: 0.0103 Acc: 0.7559
val Loss: 0.0191 Acc: 0.4721

Epoch 15/87
----------
train Loss: 0.0104 Acc: 0.7534
val Loss: 0.0191 Acc: 0.4647

Epoch 16/87
----------
train Loss: 0.0097 Acc: 0.7807
val Loss: 0.0190 Acc: 0.4944

Epoch 17/87
----------
train Loss: 0.0093 Acc: 0.7720
val Loss: 0.0188 Acc: 0.4833

Epoch 18/87
----------
train Loss: 0.0091 Acc: 0.7819
val Loss: 0.0187 Acc: 0.4870

Epoch 19/87
----------
train Loss: 0.0091 Acc: 0.7720
val Loss: 0.0190 Acc: 0.4721

Epoch 20/87
----------
train Loss: 0.0090 Acc: 0.7993
val Loss: 0.0189 Acc: 0.4796

Epoch 21/87
----------
train Loss: 0.0097 Acc: 0.7658
val Loss: 0.0192 Acc: 0.4833

Epoch 22/87
----------
train Loss: 0.0095 Acc: 0.7807
val Loss: 0.0191 Acc: 0.4721

Epoch 23/87
----------
train Loss: 0.0092 Acc: 0.7881
val Loss: 0.0190 Acc: 0.4684

Epoch 24/87
----------
train Loss: 0.0091 Acc: 0.7856
val Loss: 0.0190 Acc: 0.4758

Epoch 25/87
----------
train Loss: 0.0085 Acc: 0.7893
val Loss: 0.0188 Acc: 0.4721

Epoch 26/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0091 Acc: 0.7856
val Loss: 0.0191 Acc: 0.4647

Epoch 27/87
----------
train Loss: 0.0091 Acc: 0.7757
val Loss: 0.0191 Acc: 0.4758

Epoch 28/87
----------
train Loss: 0.0087 Acc: 0.7980
val Loss: 0.0193 Acc: 0.4833

Epoch 29/87
----------
train Loss: 0.0088 Acc: 0.7906
val Loss: 0.0188 Acc: 0.4833

Epoch 30/87
----------
train Loss: 0.0094 Acc: 0.7980
val Loss: 0.0190 Acc: 0.4833

Epoch 31/87
----------
train Loss: 0.0088 Acc: 0.7869
val Loss: 0.0187 Acc: 0.4944

Epoch 32/87
----------
train Loss: 0.0085 Acc: 0.7831
val Loss: 0.0187 Acc: 0.4796

Epoch 33/87
----------
train Loss: 0.0084 Acc: 0.7931
val Loss: 0.0190 Acc: 0.4721

Epoch 34/87
----------
train Loss: 0.0097 Acc: 0.7881
val Loss: 0.0188 Acc: 0.4758

Epoch 35/87
----------
train Loss: 0.0086 Acc: 0.7893
val Loss: 0.0186 Acc: 0.4796

Epoch 36/87
----------
train Loss: 0.0087 Acc: 0.7844
val Loss: 0.0190 Acc: 0.4796

Epoch 37/87
----------
train Loss: 0.0083 Acc: 0.8017
val Loss: 0.0189 Acc: 0.4758

Epoch 38/87
----------
train Loss: 0.0087 Acc: 0.7819
val Loss: 0.0186 Acc: 0.4796

Epoch 39/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0085 Acc: 0.7844
val Loss: 0.0192 Acc: 0.4758

Epoch 40/87
----------
train Loss: 0.0098 Acc: 0.7943
val Loss: 0.0189 Acc: 0.4833

Epoch 41/87
----------
train Loss: 0.0093 Acc: 0.7955
val Loss: 0.0191 Acc: 0.4758

Epoch 42/87
----------
train Loss: 0.0090 Acc: 0.7918
val Loss: 0.0188 Acc: 0.4833

Epoch 43/87
----------
train Loss: 0.0094 Acc: 0.7918
val Loss: 0.0190 Acc: 0.4944

Epoch 44/87
----------
train Loss: 0.0090 Acc: 0.7906
val Loss: 0.0189 Acc: 0.4833

Epoch 45/87
----------
train Loss: 0.0092 Acc: 0.7869
val Loss: 0.0190 Acc: 0.4647

Epoch 46/87
----------
train Loss: 0.0087 Acc: 0.7893
val Loss: 0.0191 Acc: 0.4944

Epoch 47/87
----------
train Loss: 0.0087 Acc: 0.7918
val Loss: 0.0191 Acc: 0.4796

Epoch 48/87
----------
train Loss: 0.0093 Acc: 0.7980
val Loss: 0.0190 Acc: 0.4907

Epoch 49/87
----------
train Loss: 0.0086 Acc: 0.8017
val Loss: 0.0189 Acc: 0.4796

Epoch 50/87
----------
train Loss: 0.0090 Acc: 0.7893
val Loss: 0.0186 Acc: 0.4907

Epoch 51/87
----------
train Loss: 0.0092 Acc: 0.7993
val Loss: 0.0188 Acc: 0.4870

Epoch 52/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0085 Acc: 0.8067
val Loss: 0.0189 Acc: 0.4981

Epoch 53/87
----------
train Loss: 0.0086 Acc: 0.7869
val Loss: 0.0188 Acc: 0.4907

Epoch 54/87
----------
train Loss: 0.0089 Acc: 0.7943
val Loss: 0.0190 Acc: 0.4833

Epoch 55/87
----------
train Loss: 0.0099 Acc: 0.7980
val Loss: 0.0192 Acc: 0.4907

Epoch 56/87
----------
train Loss: 0.0087 Acc: 0.8067
val Loss: 0.0190 Acc: 0.4833

Epoch 57/87
----------
train Loss: 0.0091 Acc: 0.7918
val Loss: 0.0190 Acc: 0.4758

Epoch 58/87
----------
train Loss: 0.0083 Acc: 0.8042
val Loss: 0.0190 Acc: 0.4796

Epoch 59/87
----------
train Loss: 0.0091 Acc: 0.7757
val Loss: 0.0188 Acc: 0.4796

Epoch 60/87
----------
train Loss: 0.0090 Acc: 0.7881
val Loss: 0.0188 Acc: 0.4721

Epoch 61/87
----------
train Loss: 0.0085 Acc: 0.7931
val Loss: 0.0188 Acc: 0.4796

Epoch 62/87
----------
train Loss: 0.0085 Acc: 0.7968
val Loss: 0.0186 Acc: 0.4833

Epoch 63/87
----------
train Loss: 0.0087 Acc: 0.7831
val Loss: 0.0188 Acc: 0.4758

Epoch 64/87
----------
train Loss: 0.0094 Acc: 0.7980
val Loss: 0.0188 Acc: 0.4758

Epoch 65/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0094 Acc: 0.7918
val Loss: 0.0189 Acc: 0.4870

Epoch 66/87
----------
train Loss: 0.0087 Acc: 0.7931
val Loss: 0.0190 Acc: 0.4758

Epoch 67/87
----------
train Loss: 0.0087 Acc: 0.7931
val Loss: 0.0191 Acc: 0.4721

Epoch 68/87
----------
train Loss: 0.0085 Acc: 0.8092
val Loss: 0.0188 Acc: 0.4796

Epoch 69/87
----------
train Loss: 0.0089 Acc: 0.7831
val Loss: 0.0190 Acc: 0.4833

Epoch 70/87
----------
train Loss: 0.0090 Acc: 0.7831
val Loss: 0.0190 Acc: 0.4796

Epoch 71/87
----------
train Loss: 0.0100 Acc: 0.7819
val Loss: 0.0188 Acc: 0.4833

Epoch 72/87
----------
train Loss: 0.0086 Acc: 0.7931
val Loss: 0.0189 Acc: 0.4758

Epoch 73/87
----------
train Loss: 0.0090 Acc: 0.7943
val Loss: 0.0191 Acc: 0.4833

Epoch 74/87
----------
train Loss: 0.0088 Acc: 0.7968
val Loss: 0.0189 Acc: 0.4944

Epoch 75/87
----------
train Loss: 0.0082 Acc: 0.7968
val Loss: 0.0191 Acc: 0.4796

Epoch 76/87
----------
train Loss: 0.0089 Acc: 0.7943
val Loss: 0.0187 Acc: 0.4796

Epoch 77/87
----------
train Loss: 0.0088 Acc: 0.7782
val Loss: 0.0189 Acc: 0.4870

Epoch 78/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0085 Acc: 0.7906
val Loss: 0.0189 Acc: 0.4721

Epoch 79/87
----------
train Loss: 0.0089 Acc: 0.7893
val Loss: 0.0189 Acc: 0.5019

Epoch 80/87
----------
train Loss: 0.0085 Acc: 0.7906
val Loss: 0.0188 Acc: 0.4833

Epoch 81/87
----------
train Loss: 0.0091 Acc: 0.8067
val Loss: 0.0189 Acc: 0.4870

Epoch 82/87
----------
train Loss: 0.0095 Acc: 0.7881
val Loss: 0.0188 Acc: 0.4796

Epoch 83/87
----------
train Loss: 0.0087 Acc: 0.7819
val Loss: 0.0187 Acc: 0.4796

Epoch 84/87
----------
train Loss: 0.0085 Acc: 0.7968
val Loss: 0.0187 Acc: 0.4758

Epoch 85/87
----------
train Loss: 0.0091 Acc: 0.7980
val Loss: 0.0188 Acc: 0.4833

Epoch 86/87
----------
train Loss: 0.0089 Acc: 0.7955
val Loss: 0.0190 Acc: 0.4796

Epoch 87/87
----------
train Loss: 0.0091 Acc: 0.7980
val Loss: 0.0188 Acc: 0.4796

Training complete in 7m 60s
Best val Acc: 0.501859

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0106 Acc: 0.7596
val Loss: 0.0245 Acc: 0.4052

Epoch 1/87
----------
train Loss: 0.0087 Acc: 0.7274
val Loss: 0.0313 Acc: 0.3532

Epoch 2/87
----------
train Loss: 0.0057 Acc: 0.8327
val Loss: 0.0257 Acc: 0.4312

Epoch 3/87
----------
train Loss: 0.0045 Acc: 0.8724
val Loss: 0.0231 Acc: 0.4461

Epoch 4/87
----------
train Loss: 0.0036 Acc: 0.9083
val Loss: 0.0251 Acc: 0.4498

Epoch 5/87
----------
train Loss: 0.0049 Acc: 0.8736
val Loss: 0.0268 Acc: 0.4349

Epoch 6/87
----------
train Loss: 0.0040 Acc: 0.8872
val Loss: 0.0252 Acc: 0.4610

Epoch 7/87
----------
train Loss: 0.0038 Acc: 0.8984
val Loss: 0.0263 Acc: 0.4833

Epoch 8/87
----------
train Loss: 0.0038 Acc: 0.9046
val Loss: 0.0323 Acc: 0.4126

Epoch 9/87
----------
train Loss: 0.0040 Acc: 0.9219
val Loss: 0.0275 Acc: 0.4275

Epoch 10/87
----------
train Loss: 0.0040 Acc: 0.8934
val Loss: 0.0308 Acc: 0.4201

Epoch 11/87
----------
train Loss: 0.0035 Acc: 0.9033
val Loss: 0.0293 Acc: 0.4461

Epoch 12/87
----------
train Loss: 0.0050 Acc: 0.8959
val Loss: 0.0327 Acc: 0.3866

Epoch 13/87
----------
LR is set to 0.001
train Loss: 0.0018 Acc: 0.9504
val Loss: 0.0305 Acc: 0.4461

Epoch 14/87
----------
train Loss: 0.0027 Acc: 0.9616
val Loss: 0.0285 Acc: 0.4535

Epoch 15/87
----------
train Loss: 0.0015 Acc: 0.9566
val Loss: 0.0282 Acc: 0.4535

Epoch 16/87
----------
train Loss: 0.0028 Acc: 0.9752
val Loss: 0.0261 Acc: 0.4647

Epoch 17/87
----------
train Loss: 0.0009 Acc: 0.9839
val Loss: 0.0257 Acc: 0.4944

Epoch 18/87
----------
train Loss: 0.0016 Acc: 0.9752
val Loss: 0.0253 Acc: 0.4833

Epoch 19/87
----------
train Loss: 0.0013 Acc: 0.9777
val Loss: 0.0250 Acc: 0.4572

Epoch 20/87
----------
train Loss: 0.0009 Acc: 0.9789
val Loss: 0.0240 Acc: 0.4981

Epoch 21/87
----------
train Loss: 0.0011 Acc: 0.9864
val Loss: 0.0240 Acc: 0.5019

Epoch 22/87
----------
train Loss: 0.0009 Acc: 0.9802
val Loss: 0.0240 Acc: 0.5093

Epoch 23/87
----------
train Loss: 0.0008 Acc: 0.9839
val Loss: 0.0238 Acc: 0.5242

Epoch 24/87
----------
train Loss: 0.0015 Acc: 0.9765
val Loss: 0.0237 Acc: 0.5279

Epoch 25/87
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0249 Acc: 0.5019

Epoch 26/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0012 Acc: 0.9789
val Loss: 0.0255 Acc: 0.5093

Epoch 27/87
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0247 Acc: 0.5167

Epoch 28/87
----------
train Loss: 0.0005 Acc: 0.9901
val Loss: 0.0248 Acc: 0.5056

Epoch 29/87
----------
train Loss: 0.0008 Acc: 0.9827
val Loss: 0.0246 Acc: 0.5167

Epoch 30/87
----------
train Loss: 0.0010 Acc: 0.9839
val Loss: 0.0248 Acc: 0.5167

Epoch 31/87
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0251 Acc: 0.5204

Epoch 32/87
----------
train Loss: 0.0009 Acc: 0.9839
val Loss: 0.0242 Acc: 0.5242

Epoch 33/87
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0250 Acc: 0.5279

Epoch 34/87
----------
train Loss: 0.0032 Acc: 0.9802
val Loss: 0.0241 Acc: 0.5316

Epoch 35/87
----------
train Loss: 0.0009 Acc: 0.9839
val Loss: 0.0248 Acc: 0.5204

Epoch 36/87
----------
train Loss: 0.0013 Acc: 0.9839
val Loss: 0.0244 Acc: 0.5130

Epoch 37/87
----------
train Loss: 0.0006 Acc: 0.9864
val Loss: 0.0244 Acc: 0.5242

Epoch 38/87
----------
train Loss: 0.0006 Acc: 0.9851
val Loss: 0.0243 Acc: 0.5204

Epoch 39/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0242 Acc: 0.5130

Epoch 40/87
----------
train Loss: 0.0007 Acc: 0.9901
val Loss: 0.0245 Acc: 0.5204

Epoch 41/87
----------
train Loss: 0.0012 Acc: 0.9851
val Loss: 0.0247 Acc: 0.5242

Epoch 42/87
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0250 Acc: 0.5056

Epoch 43/87
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0241 Acc: 0.5130

Epoch 44/87
----------
train Loss: 0.0007 Acc: 0.9802
val Loss: 0.0241 Acc: 0.5242

Epoch 45/87
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0246 Acc: 0.5130

Epoch 46/87
----------
train Loss: 0.0005 Acc: 0.9839
val Loss: 0.0249 Acc: 0.5130

Epoch 47/87
----------
train Loss: 0.0004 Acc: 0.9913
val Loss: 0.0244 Acc: 0.5204

Epoch 48/87
----------
train Loss: 0.0006 Acc: 0.9901
val Loss: 0.0238 Acc: 0.5167

Epoch 49/87
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0247 Acc: 0.5242

Epoch 50/87
----------
train Loss: 0.0008 Acc: 0.9888
val Loss: 0.0249 Acc: 0.5204

Epoch 51/87
----------
train Loss: 0.0011 Acc: 0.9876
val Loss: 0.0240 Acc: 0.5056

Epoch 52/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0011 Acc: 0.9876
val Loss: 0.0243 Acc: 0.5130

Epoch 53/87
----------
train Loss: 0.0006 Acc: 0.9888
val Loss: 0.0241 Acc: 0.5167

Epoch 54/87
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0247 Acc: 0.5130

Epoch 55/87
----------
train Loss: 0.0011 Acc: 0.9851
val Loss: 0.0237 Acc: 0.5130

Epoch 56/87
----------
train Loss: 0.0016 Acc: 0.9864
val Loss: 0.0247 Acc: 0.5130

Epoch 57/87
----------
train Loss: 0.0005 Acc: 0.9901
val Loss: 0.0242 Acc: 0.5242

Epoch 58/87
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0239 Acc: 0.5204

Epoch 59/87
----------
train Loss: 0.0010 Acc: 0.9864
val Loss: 0.0246 Acc: 0.5353

Epoch 60/87
----------
train Loss: 0.0012 Acc: 0.9827
val Loss: 0.0251 Acc: 0.5242

Epoch 61/87
----------
train Loss: 0.0006 Acc: 0.9851
val Loss: 0.0242 Acc: 0.5279

Epoch 62/87
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0244 Acc: 0.5316

Epoch 63/87
----------
train Loss: 0.0005 Acc: 0.9926
val Loss: 0.0239 Acc: 0.5242

Epoch 64/87
----------
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0244 Acc: 0.5242

Epoch 65/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0243 Acc: 0.5279

Epoch 66/87
----------
train Loss: 0.0006 Acc: 0.9876
val Loss: 0.0247 Acc: 0.5279

Epoch 67/87
----------
train Loss: 0.0004 Acc: 0.9851
val Loss: 0.0250 Acc: 0.5130

Epoch 68/87
----------
train Loss: 0.0009 Acc: 0.9888
val Loss: 0.0241 Acc: 0.5279

Epoch 69/87
----------
train Loss: 0.0007 Acc: 0.9913
val Loss: 0.0253 Acc: 0.5204

Epoch 70/87
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0240 Acc: 0.5167

Epoch 71/87
----------
train Loss: 0.0005 Acc: 0.9864
val Loss: 0.0241 Acc: 0.5130

Epoch 72/87
----------
train Loss: 0.0005 Acc: 0.9876
val Loss: 0.0244 Acc: 0.5130

Epoch 73/87
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0242 Acc: 0.5167

Epoch 74/87
----------
train Loss: 0.0008 Acc: 0.9888
val Loss: 0.0242 Acc: 0.5167

Epoch 75/87
----------
train Loss: 0.0008 Acc: 0.9876
val Loss: 0.0243 Acc: 0.5242

Epoch 76/87
----------
train Loss: 0.0011 Acc: 0.9876
val Loss: 0.0248 Acc: 0.5279

Epoch 77/87
----------
train Loss: 0.0004 Acc: 0.9888
val Loss: 0.0238 Acc: 0.5242

Epoch 78/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0007 Acc: 0.9876
val Loss: 0.0250 Acc: 0.5167

Epoch 79/87
----------
train Loss: 0.0006 Acc: 0.9888
val Loss: 0.0250 Acc: 0.5242

Epoch 80/87
----------
train Loss: 0.0004 Acc: 0.9913
val Loss: 0.0246 Acc: 0.5167

Epoch 81/87
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0247 Acc: 0.5167

Epoch 82/87
----------
train Loss: 0.0004 Acc: 0.9901
val Loss: 0.0248 Acc: 0.5204

Epoch 83/87
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0237 Acc: 0.5204

Epoch 84/87
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0237 Acc: 0.5167

Epoch 85/87
----------
train Loss: 0.0005 Acc: 0.9851
val Loss: 0.0240 Acc: 0.5279

Epoch 86/87
----------
train Loss: 0.0005 Acc: 0.9913
val Loss: 0.0247 Acc: 0.5242

Epoch 87/87
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0249 Acc: 0.5167

Training complete in 8m 35s
Best val Acc: 0.535316

---Testing---
Test accuracy: 0.875465
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 84 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 70 %
Accuracy of Yellowfin tuna : 94 %
mean: 0.8357088414433459, std: 0.10789668511162265
--------------------

run info[val: 0.3, epoch: 97, randcrop: False, decay: 12]

---Training last layer.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0280 Acc: 0.1525
val Loss: 0.0306 Acc: 0.2081

Epoch 1/96
----------
train Loss: 0.0235 Acc: 0.2692
val Loss: 0.0256 Acc: 0.3292

Epoch 2/96
----------
train Loss: 0.0194 Acc: 0.4576
val Loss: 0.0259 Acc: 0.3509

Epoch 3/96
----------
train Loss: 0.0169 Acc: 0.5239
val Loss: 0.0235 Acc: 0.3634

Epoch 4/96
----------
train Loss: 0.0150 Acc: 0.5570
val Loss: 0.0213 Acc: 0.4037

Epoch 5/96
----------
train Loss: 0.0142 Acc: 0.5875
val Loss: 0.0219 Acc: 0.4441

Epoch 6/96
----------
train Loss: 0.0129 Acc: 0.6446
val Loss: 0.0236 Acc: 0.3975

Epoch 7/96
----------
train Loss: 0.0123 Acc: 0.6446
val Loss: 0.0215 Acc: 0.4410

Epoch 8/96
----------
train Loss: 0.0116 Acc: 0.6698
val Loss: 0.0215 Acc: 0.4286

Epoch 9/96
----------
train Loss: 0.0103 Acc: 0.7029
val Loss: 0.0209 Acc: 0.4534

Epoch 10/96
----------
train Loss: 0.0101 Acc: 0.7255
val Loss: 0.0205 Acc: 0.4503

Epoch 11/96
----------
train Loss: 0.0096 Acc: 0.7414
val Loss: 0.0219 Acc: 0.4596

Epoch 12/96
----------
LR is set to 0.001
train Loss: 0.0093 Acc: 0.7560
val Loss: 0.0222 Acc: 0.4565

Epoch 13/96
----------
train Loss: 0.0087 Acc: 0.7785
val Loss: 0.0205 Acc: 0.4596

Epoch 14/96
----------
train Loss: 0.0090 Acc: 0.7812
val Loss: 0.0206 Acc: 0.4596

Epoch 15/96
----------
train Loss: 0.0088 Acc: 0.7865
val Loss: 0.0208 Acc: 0.4689

Epoch 16/96
----------
train Loss: 0.0086 Acc: 0.7905
val Loss: 0.0208 Acc: 0.4627

Epoch 17/96
----------
train Loss: 0.0088 Acc: 0.7958
val Loss: 0.0215 Acc: 0.4627

Epoch 18/96
----------
train Loss: 0.0086 Acc: 0.8090
val Loss: 0.0205 Acc: 0.4565

Epoch 19/96
----------
train Loss: 0.0085 Acc: 0.7984
val Loss: 0.0219 Acc: 0.4596

Epoch 20/96
----------
train Loss: 0.0085 Acc: 0.8117
val Loss: 0.0219 Acc: 0.4503

Epoch 21/96
----------
train Loss: 0.0085 Acc: 0.7997
val Loss: 0.0210 Acc: 0.4689

Epoch 22/96
----------
train Loss: 0.0085 Acc: 0.7905
val Loss: 0.0204 Acc: 0.4627

Epoch 23/96
----------
train Loss: 0.0085 Acc: 0.8090
val Loss: 0.0204 Acc: 0.4565

Epoch 24/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0085 Acc: 0.7997
val Loss: 0.0199 Acc: 0.4534

Epoch 25/96
----------
train Loss: 0.0083 Acc: 0.7944
val Loss: 0.0212 Acc: 0.4534

Epoch 26/96
----------
train Loss: 0.0084 Acc: 0.8077
val Loss: 0.0203 Acc: 0.4596

Epoch 27/96
----------
train Loss: 0.0082 Acc: 0.8011
val Loss: 0.0199 Acc: 0.4627

Epoch 28/96
----------
train Loss: 0.0084 Acc: 0.7971
val Loss: 0.0193 Acc: 0.4565

Epoch 29/96
----------
train Loss: 0.0084 Acc: 0.7997
val Loss: 0.0202 Acc: 0.4565

Epoch 30/96
----------
train Loss: 0.0085 Acc: 0.8037
val Loss: 0.0202 Acc: 0.4596

Epoch 31/96
----------
train Loss: 0.0083 Acc: 0.8103
val Loss: 0.0211 Acc: 0.4596

Epoch 32/96
----------
train Loss: 0.0083 Acc: 0.8183
val Loss: 0.0209 Acc: 0.4596

Epoch 33/96
----------
train Loss: 0.0082 Acc: 0.8170
val Loss: 0.0207 Acc: 0.4596

Epoch 34/96
----------
train Loss: 0.0084 Acc: 0.7997
val Loss: 0.0196 Acc: 0.4596

Epoch 35/96
----------
train Loss: 0.0082 Acc: 0.8276
val Loss: 0.0210 Acc: 0.4596

Epoch 36/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0085 Acc: 0.8223
val Loss: 0.0210 Acc: 0.4627

Epoch 37/96
----------
train Loss: 0.0081 Acc: 0.8050
val Loss: 0.0223 Acc: 0.4596

Epoch 38/96
----------
train Loss: 0.0083 Acc: 0.8156
val Loss: 0.0211 Acc: 0.4565

Epoch 39/96
----------
train Loss: 0.0084 Acc: 0.8103
val Loss: 0.0207 Acc: 0.4596

Epoch 40/96
----------
train Loss: 0.0082 Acc: 0.8077
val Loss: 0.0212 Acc: 0.4565

Epoch 41/96
----------
train Loss: 0.0085 Acc: 0.8183
val Loss: 0.0216 Acc: 0.4627

Epoch 42/96
----------
train Loss: 0.0085 Acc: 0.7931
val Loss: 0.0205 Acc: 0.4596

Epoch 43/96
----------
train Loss: 0.0084 Acc: 0.8170
val Loss: 0.0206 Acc: 0.4627

Epoch 44/96
----------
train Loss: 0.0085 Acc: 0.8064
val Loss: 0.0216 Acc: 0.4596

Epoch 45/96
----------
train Loss: 0.0083 Acc: 0.8077
val Loss: 0.0211 Acc: 0.4596

Epoch 46/96
----------
train Loss: 0.0084 Acc: 0.8050
val Loss: 0.0213 Acc: 0.4627

Epoch 47/96
----------
train Loss: 0.0084 Acc: 0.8156
val Loss: 0.0211 Acc: 0.4596

Epoch 48/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0083 Acc: 0.7984
val Loss: 0.0221 Acc: 0.4627

Epoch 49/96
----------
train Loss: 0.0083 Acc: 0.7984
val Loss: 0.0211 Acc: 0.4596

Epoch 50/96
----------
train Loss: 0.0083 Acc: 0.8130
val Loss: 0.0218 Acc: 0.4596

Epoch 51/96
----------
train Loss: 0.0083 Acc: 0.8183
val Loss: 0.0198 Acc: 0.4596

Epoch 52/96
----------
train Loss: 0.0082 Acc: 0.8050
val Loss: 0.0219 Acc: 0.4596

Epoch 53/96
----------
train Loss: 0.0082 Acc: 0.8037
val Loss: 0.0224 Acc: 0.4627

Epoch 54/96
----------
train Loss: 0.0084 Acc: 0.8050
val Loss: 0.0203 Acc: 0.4627

Epoch 55/96
----------
train Loss: 0.0083 Acc: 0.8064
val Loss: 0.0206 Acc: 0.4658

Epoch 56/96
----------
train Loss: 0.0083 Acc: 0.8090
val Loss: 0.0217 Acc: 0.4565

Epoch 57/96
----------
train Loss: 0.0084 Acc: 0.8064
val Loss: 0.0205 Acc: 0.4596

Epoch 58/96
----------
train Loss: 0.0082 Acc: 0.8011
val Loss: 0.0208 Acc: 0.4658

Epoch 59/96
----------
train Loss: 0.0085 Acc: 0.8090
val Loss: 0.0210 Acc: 0.4627

Epoch 60/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0082 Acc: 0.8223
val Loss: 0.0203 Acc: 0.4565

Epoch 61/96
----------
train Loss: 0.0083 Acc: 0.8064
val Loss: 0.0209 Acc: 0.4596

Epoch 62/96
----------
train Loss: 0.0084 Acc: 0.8024
val Loss: 0.0210 Acc: 0.4596

Epoch 63/96
----------
train Loss: 0.0083 Acc: 0.8143
val Loss: 0.0214 Acc: 0.4596

Epoch 64/96
----------
train Loss: 0.0082 Acc: 0.8183
val Loss: 0.0212 Acc: 0.4689

Epoch 65/96
----------
train Loss: 0.0084 Acc: 0.8050
val Loss: 0.0209 Acc: 0.4627

Epoch 66/96
----------
train Loss: 0.0084 Acc: 0.7918
val Loss: 0.0218 Acc: 0.4627

Epoch 67/96
----------
train Loss: 0.0084 Acc: 0.8077
val Loss: 0.0210 Acc: 0.4627

Epoch 68/96
----------
train Loss: 0.0082 Acc: 0.8050
val Loss: 0.0226 Acc: 0.4658

Epoch 69/96
----------
train Loss: 0.0083 Acc: 0.8103
val Loss: 0.0205 Acc: 0.4596

Epoch 70/96
----------
train Loss: 0.0083 Acc: 0.8011
val Loss: 0.0217 Acc: 0.4627

Epoch 71/96
----------
train Loss: 0.0084 Acc: 0.7997
val Loss: 0.0220 Acc: 0.4658

Epoch 72/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0081 Acc: 0.8090
val Loss: 0.0211 Acc: 0.4596

Epoch 73/96
----------
train Loss: 0.0082 Acc: 0.8170
val Loss: 0.0214 Acc: 0.4596

Epoch 74/96
----------
train Loss: 0.0083 Acc: 0.8090
val Loss: 0.0208 Acc: 0.4627

Epoch 75/96
----------
train Loss: 0.0083 Acc: 0.8037
val Loss: 0.0208 Acc: 0.4658

Epoch 76/96
----------
train Loss: 0.0084 Acc: 0.7984
val Loss: 0.0214 Acc: 0.4627

Epoch 77/96
----------
train Loss: 0.0084 Acc: 0.8011
val Loss: 0.0206 Acc: 0.4596

Epoch 78/96
----------
train Loss: 0.0082 Acc: 0.8130
val Loss: 0.0206 Acc: 0.4627

Epoch 79/96
----------
train Loss: 0.0085 Acc: 0.8064
val Loss: 0.0212 Acc: 0.4627

Epoch 80/96
----------
train Loss: 0.0084 Acc: 0.7971
val Loss: 0.0212 Acc: 0.4627

Epoch 81/96
----------
train Loss: 0.0082 Acc: 0.8103
val Loss: 0.0197 Acc: 0.4627

Epoch 82/96
----------
train Loss: 0.0081 Acc: 0.8196
val Loss: 0.0219 Acc: 0.4658

Epoch 83/96
----------
train Loss: 0.0082 Acc: 0.8077
val Loss: 0.0211 Acc: 0.4596

Epoch 84/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0084 Acc: 0.8037
val Loss: 0.0216 Acc: 0.4596

Epoch 85/96
----------
train Loss: 0.0085 Acc: 0.7971
val Loss: 0.0208 Acc: 0.4565

Epoch 86/96
----------
train Loss: 0.0084 Acc: 0.8037
val Loss: 0.0215 Acc: 0.4627

Epoch 87/96
----------
train Loss: 0.0083 Acc: 0.7944
val Loss: 0.0212 Acc: 0.4596

Epoch 88/96
----------
train Loss: 0.0083 Acc: 0.8236
val Loss: 0.0216 Acc: 0.4627

Epoch 89/96
----------
train Loss: 0.0083 Acc: 0.8170
val Loss: 0.0207 Acc: 0.4627

Epoch 90/96
----------
train Loss: 0.0084 Acc: 0.8103
val Loss: 0.0198 Acc: 0.4596

Epoch 91/96
----------
train Loss: 0.0084 Acc: 0.8249
val Loss: 0.0210 Acc: 0.4596

Epoch 92/96
----------
train Loss: 0.0083 Acc: 0.8196
val Loss: 0.0215 Acc: 0.4596

Epoch 93/96
----------
train Loss: 0.0084 Acc: 0.8130
val Loss: 0.0206 Acc: 0.4565

Epoch 94/96
----------
train Loss: 0.0086 Acc: 0.8064
val Loss: 0.0200 Acc: 0.4534

Epoch 95/96
----------
train Loss: 0.0084 Acc: 0.8103
val Loss: 0.0206 Acc: 0.4627

Epoch 96/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0084 Acc: 0.8117
val Loss: 0.0203 Acc: 0.4658

Training complete in 8m 57s
Best val Acc: 0.468944

---Fine tuning.---
Epoch 0/96
----------
LR is set to 0.01
train Loss: 0.0092 Acc: 0.7520
val Loss: 0.0205 Acc: 0.4752

Epoch 1/96
----------
train Loss: 0.0048 Acc: 0.8966
val Loss: 0.0219 Acc: 0.4224

Epoch 2/96
----------
train Loss: 0.0027 Acc: 0.9456
val Loss: 0.0210 Acc: 0.5155

Epoch 3/96
----------
train Loss: 0.0015 Acc: 0.9695
val Loss: 0.0204 Acc: 0.5186

Epoch 4/96
----------
train Loss: 0.0012 Acc: 0.9708
val Loss: 0.0204 Acc: 0.5155

Epoch 5/96
----------
train Loss: 0.0010 Acc: 0.9788
val Loss: 0.0214 Acc: 0.5280

Epoch 6/96
----------
train Loss: 0.0007 Acc: 0.9854
val Loss: 0.0208 Acc: 0.5186

Epoch 7/96
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0207 Acc: 0.5311

Epoch 8/96
----------
train Loss: 0.0005 Acc: 0.9854
val Loss: 0.0199 Acc: 0.5217

Epoch 9/96
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0231 Acc: 0.5217

Epoch 10/96
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0202 Acc: 0.5311

Epoch 11/96
----------
train Loss: 0.0005 Acc: 0.9854
val Loss: 0.0218 Acc: 0.5280

Epoch 12/96
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0211 Acc: 0.5435

Epoch 13/96
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0217 Acc: 0.5373

Epoch 14/96
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0214 Acc: 0.5466

Epoch 15/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0221 Acc: 0.5466

Epoch 16/96
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0209 Acc: 0.5373

Epoch 17/96
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0238 Acc: 0.5373

Epoch 18/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0200 Acc: 0.5404

Epoch 19/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0222 Acc: 0.5311

Epoch 20/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0209 Acc: 0.5373

Epoch 21/96
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0229 Acc: 0.5280

Epoch 22/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0227 Acc: 0.5280

Epoch 23/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0206 Acc: 0.5311

Epoch 24/96
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0209 Acc: 0.5373

Epoch 25/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0201 Acc: 0.5248

Epoch 26/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0219 Acc: 0.5248

Epoch 27/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0221 Acc: 0.5342

Epoch 28/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0217 Acc: 0.5342

Epoch 29/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0238 Acc: 0.5311

Epoch 30/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0214 Acc: 0.5373

Epoch 31/96
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0216 Acc: 0.5373

Epoch 32/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0209 Acc: 0.5311

Epoch 33/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0217 Acc: 0.5280

Epoch 34/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0214 Acc: 0.5342

Epoch 35/96
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0219 Acc: 0.5311

Epoch 36/96
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0220 Acc: 0.5311

Epoch 37/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0215 Acc: 0.5342

Epoch 38/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0220 Acc: 0.5311

Epoch 39/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0215 Acc: 0.5311

Epoch 40/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0201 Acc: 0.5373

Epoch 41/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0213 Acc: 0.5311

Epoch 42/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0214 Acc: 0.5342

Epoch 43/96
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0218 Acc: 0.5280

Epoch 44/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0219 Acc: 0.5280

Epoch 45/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0223 Acc: 0.5311

Epoch 46/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0207 Acc: 0.5280

Epoch 47/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0233 Acc: 0.5342

Epoch 48/96
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0211 Acc: 0.5217

Epoch 49/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0228 Acc: 0.5248

Epoch 50/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0211 Acc: 0.5311

Epoch 51/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0211 Acc: 0.5311

Epoch 52/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0222 Acc: 0.5280

Epoch 53/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0219 Acc: 0.5248

Epoch 54/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0212 Acc: 0.5248

Epoch 55/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0227 Acc: 0.5217

Epoch 56/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0214 Acc: 0.5311

Epoch 57/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0211 Acc: 0.5311

Epoch 58/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0215 Acc: 0.5280

Epoch 59/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0216 Acc: 0.5248

Epoch 60/96
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0227 Acc: 0.5280

Epoch 61/96
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0218 Acc: 0.5280

Epoch 62/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0234 Acc: 0.5311

Epoch 63/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0207 Acc: 0.5311

Epoch 64/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0205 Acc: 0.5373

Epoch 65/96
----------
train Loss: 0.0002 Acc: 0.9960
val Loss: 0.0200 Acc: 0.5342

Epoch 66/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0204 Acc: 0.5342

Epoch 67/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0215 Acc: 0.5280

Epoch 68/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0220 Acc: 0.5342

Epoch 69/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0209 Acc: 0.5311

Epoch 70/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0216 Acc: 0.5280

Epoch 71/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0238 Acc: 0.5280

Epoch 72/96
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0223 Acc: 0.5280

Epoch 73/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0227 Acc: 0.5280

Epoch 74/96
----------
train Loss: 0.0002 Acc: 0.9960
val Loss: 0.0228 Acc: 0.5311

Epoch 75/96
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0211 Acc: 0.5311

Epoch 76/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0205 Acc: 0.5280

Epoch 77/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0221 Acc: 0.5342

Epoch 78/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0211 Acc: 0.5311

Epoch 79/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0204 Acc: 0.5373

Epoch 80/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0202 Acc: 0.5280

Epoch 81/96
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0210 Acc: 0.5342

Epoch 82/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0222 Acc: 0.5342

Epoch 83/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0224 Acc: 0.5342

Epoch 84/96
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0213 Acc: 0.5280

Epoch 85/96
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0217 Acc: 0.5373

Epoch 86/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0224 Acc: 0.5373

Epoch 87/96
----------
train Loss: 0.0002 Acc: 0.9947
val Loss: 0.0207 Acc: 0.5280

Epoch 88/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0207 Acc: 0.5217

Epoch 89/96
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0201 Acc: 0.5342

Epoch 90/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0190 Acc: 0.5311

Epoch 91/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0220 Acc: 0.5311

Epoch 92/96
----------
train Loss: 0.0002 Acc: 0.9973
val Loss: 0.0234 Acc: 0.5248

Epoch 93/96
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0231 Acc: 0.5280

Epoch 94/96
----------
train Loss: 0.0002 Acc: 0.9867
val Loss: 0.0217 Acc: 0.5248

Epoch 95/96
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0207 Acc: 0.5280

Epoch 96/96
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0212 Acc: 0.5248

Training complete in 9m 29s
Best val Acc: 0.546584

---Testing---
Test accuracy: 0.857807
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of Bigeye tuna : 79 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 80 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 80 %
Accuracy of Pacific bluefin tuna : 76 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 76 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8082991619611223, std: 0.125509547858142

Model saved in "./weights/tuna_fish_[0.92]_mean[0.88]_std[0.12].save".
--------------------

run info[val: 0.1, epoch: 84, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0271 Acc: 0.1703
val Loss: 0.0442 Acc: 0.3271

Epoch 1/83
----------
train Loss: 0.0216 Acc: 0.3179
val Loss: 0.0390 Acc: 0.3925

Epoch 2/83
----------
train Loss: 0.0180 Acc: 0.4530
val Loss: 0.0357 Acc: 0.4112

Epoch 3/83
----------
train Loss: 0.0165 Acc: 0.4665
val Loss: 0.0296 Acc: 0.4393

Epoch 4/83
----------
train Loss: 0.0151 Acc: 0.5294
val Loss: 0.0322 Acc: 0.4299

Epoch 5/83
----------
train Loss: 0.0135 Acc: 0.5738
val Loss: 0.0275 Acc: 0.5047

Epoch 6/83
----------
train Loss: 0.0125 Acc: 0.6161
val Loss: 0.0307 Acc: 0.4579

Epoch 7/83
----------
train Loss: 0.0117 Acc: 0.6543
val Loss: 0.0229 Acc: 0.4860

Epoch 8/83
----------
train Loss: 0.0111 Acc: 0.6574
val Loss: 0.0238 Acc: 0.4579

Epoch 9/83
----------
LR is set to 0.001
train Loss: 0.0110 Acc: 0.6574
val Loss: 0.0374 Acc: 0.5421

Epoch 10/83
----------
train Loss: 0.0101 Acc: 0.7183
val Loss: 0.0432 Acc: 0.4953

Epoch 11/83
----------
train Loss: 0.0101 Acc: 0.7059
val Loss: 0.0355 Acc: 0.5327

Epoch 12/83
----------
train Loss: 0.0100 Acc: 0.7307
val Loss: 0.0237 Acc: 0.5140

Epoch 13/83
----------
train Loss: 0.0100 Acc: 0.7255
val Loss: 0.0282 Acc: 0.5234

Epoch 14/83
----------
train Loss: 0.0099 Acc: 0.7307
val Loss: 0.0292 Acc: 0.5327

Epoch 15/83
----------
train Loss: 0.0097 Acc: 0.7441
val Loss: 0.0279 Acc: 0.5421

Epoch 16/83
----------
train Loss: 0.0096 Acc: 0.7379
val Loss: 0.0311 Acc: 0.5421

Epoch 17/83
----------
train Loss: 0.0098 Acc: 0.7451
val Loss: 0.0281 Acc: 0.5327

Epoch 18/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0097 Acc: 0.7358
val Loss: 0.0315 Acc: 0.5140

Epoch 19/83
----------
train Loss: 0.0095 Acc: 0.7492
val Loss: 0.0340 Acc: 0.5140

Epoch 20/83
----------
train Loss: 0.0096 Acc: 0.7472
val Loss: 0.0291 Acc: 0.5327

Epoch 21/83
----------
train Loss: 0.0098 Acc: 0.7389
val Loss: 0.0311 Acc: 0.5421

Epoch 22/83
----------
train Loss: 0.0097 Acc: 0.7327
val Loss: 0.0304 Acc: 0.5234

Epoch 23/83
----------
train Loss: 0.0095 Acc: 0.7430
val Loss: 0.0353 Acc: 0.5327

Epoch 24/83
----------
train Loss: 0.0096 Acc: 0.7265
val Loss: 0.0339 Acc: 0.5140

Epoch 25/83
----------
train Loss: 0.0095 Acc: 0.7482
val Loss: 0.0243 Acc: 0.5327

Epoch 26/83
----------
train Loss: 0.0098 Acc: 0.7379
val Loss: 0.0290 Acc: 0.5234

Epoch 27/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0095 Acc: 0.7534
val Loss: 0.0272 Acc: 0.5234

Epoch 28/83
----------
train Loss: 0.0095 Acc: 0.7441
val Loss: 0.0247 Acc: 0.5234

Epoch 29/83
----------
train Loss: 0.0098 Acc: 0.7265
val Loss: 0.0287 Acc: 0.5327

Epoch 30/83
----------
train Loss: 0.0095 Acc: 0.7348
val Loss: 0.0262 Acc: 0.5234

Epoch 31/83
----------
train Loss: 0.0095 Acc: 0.7389
val Loss: 0.0285 Acc: 0.5234

Epoch 32/83
----------
train Loss: 0.0096 Acc: 0.7389
val Loss: 0.0231 Acc: 0.5234

Epoch 33/83
----------
train Loss: 0.0096 Acc: 0.7482
val Loss: 0.0267 Acc: 0.5234

Epoch 34/83
----------
train Loss: 0.0095 Acc: 0.7451
val Loss: 0.0316 Acc: 0.5140

Epoch 35/83
----------
train Loss: 0.0095 Acc: 0.7337
val Loss: 0.0292 Acc: 0.5421

Epoch 36/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0096 Acc: 0.7348
val Loss: 0.0307 Acc: 0.5327

Epoch 37/83
----------
train Loss: 0.0095 Acc: 0.7389
val Loss: 0.0346 Acc: 0.5327

Epoch 38/83
----------
train Loss: 0.0095 Acc: 0.7544
val Loss: 0.0277 Acc: 0.5327

Epoch 39/83
----------
train Loss: 0.0095 Acc: 0.7389
val Loss: 0.0396 Acc: 0.5421

Epoch 40/83
----------
train Loss: 0.0095 Acc: 0.7430
val Loss: 0.0329 Acc: 0.5327

Epoch 41/83
----------
train Loss: 0.0097 Acc: 0.7461
val Loss: 0.0266 Acc: 0.5327

Epoch 42/83
----------
train Loss: 0.0094 Acc: 0.7472
val Loss: 0.0282 Acc: 0.5327

Epoch 43/83
----------
train Loss: 0.0097 Acc: 0.7503
val Loss: 0.0277 Acc: 0.5234

Epoch 44/83
----------
train Loss: 0.0094 Acc: 0.7389
val Loss: 0.0272 Acc: 0.5327

Epoch 45/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0096 Acc: 0.7358
val Loss: 0.0247 Acc: 0.5327

Epoch 46/83
----------
train Loss: 0.0096 Acc: 0.7379
val Loss: 0.0234 Acc: 0.5421

Epoch 47/83
----------
train Loss: 0.0094 Acc: 0.7616
val Loss: 0.0270 Acc: 0.5421

Epoch 48/83
----------
train Loss: 0.0096 Acc: 0.7368
val Loss: 0.0274 Acc: 0.5327

Epoch 49/83
----------
train Loss: 0.0096 Acc: 0.7482
val Loss: 0.0266 Acc: 0.5327

Epoch 50/83
----------
train Loss: 0.0095 Acc: 0.7575
val Loss: 0.0264 Acc: 0.5234

Epoch 51/83
----------
train Loss: 0.0095 Acc: 0.7451
val Loss: 0.0284 Acc: 0.5421

Epoch 52/83
----------
train Loss: 0.0097 Acc: 0.7399
val Loss: 0.0267 Acc: 0.5234

Epoch 53/83
----------
train Loss: 0.0095 Acc: 0.7317
val Loss: 0.0424 Acc: 0.5421

Epoch 54/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0095 Acc: 0.7523
val Loss: 0.0337 Acc: 0.5327

Epoch 55/83
----------
train Loss: 0.0095 Acc: 0.7503
val Loss: 0.0301 Acc: 0.5327

Epoch 56/83
----------
train Loss: 0.0096 Acc: 0.7451
val Loss: 0.0291 Acc: 0.5234

Epoch 57/83
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0262 Acc: 0.5327

Epoch 58/83
----------
train Loss: 0.0095 Acc: 0.7503
val Loss: 0.0304 Acc: 0.5327

Epoch 59/83
----------
train Loss: 0.0095 Acc: 0.7585
val Loss: 0.0319 Acc: 0.5234

Epoch 60/83
----------
train Loss: 0.0096 Acc: 0.7420
val Loss: 0.0261 Acc: 0.5327

Epoch 61/83
----------
train Loss: 0.0095 Acc: 0.7544
val Loss: 0.0320 Acc: 0.5234

Epoch 62/83
----------
train Loss: 0.0097 Acc: 0.7327
val Loss: 0.0303 Acc: 0.5234

Epoch 63/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0094 Acc: 0.7523
val Loss: 0.0279 Acc: 0.5327

Epoch 64/83
----------
train Loss: 0.0096 Acc: 0.7451
val Loss: 0.0223 Acc: 0.5421

Epoch 65/83
----------
train Loss: 0.0096 Acc: 0.7523
val Loss: 0.0311 Acc: 0.5327

Epoch 66/83
----------
train Loss: 0.0096 Acc: 0.7451
val Loss: 0.0366 Acc: 0.5327

Epoch 67/83
----------
train Loss: 0.0095 Acc: 0.7523
val Loss: 0.0267 Acc: 0.5234

Epoch 68/83
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0336 Acc: 0.5327

Epoch 69/83
----------
train Loss: 0.0095 Acc: 0.7389
val Loss: 0.0409 Acc: 0.5327

Epoch 70/83
----------
train Loss: 0.0095 Acc: 0.7482
val Loss: 0.0245 Acc: 0.5421

Epoch 71/83
----------
train Loss: 0.0096 Acc: 0.7441
val Loss: 0.0263 Acc: 0.5234

Epoch 72/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0096 Acc: 0.7441
val Loss: 0.0362 Acc: 0.5421

Epoch 73/83
----------
train Loss: 0.0096 Acc: 0.7327
val Loss: 0.0326 Acc: 0.5421

Epoch 74/83
----------
train Loss: 0.0096 Acc: 0.7420
val Loss: 0.0366 Acc: 0.5234

Epoch 75/83
----------
train Loss: 0.0097 Acc: 0.7348
val Loss: 0.0239 Acc: 0.5327

Epoch 76/83
----------
train Loss: 0.0095 Acc: 0.7564
val Loss: 0.0276 Acc: 0.5234

Epoch 77/83
----------
train Loss: 0.0096 Acc: 0.7399
val Loss: 0.0313 Acc: 0.5327

Epoch 78/83
----------
train Loss: 0.0097 Acc: 0.7523
val Loss: 0.0327 Acc: 0.5327

Epoch 79/83
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0368 Acc: 0.5234

Epoch 80/83
----------
train Loss: 0.0096 Acc: 0.7461
val Loss: 0.0221 Acc: 0.5234

Epoch 81/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0097 Acc: 0.7317
val Loss: 0.0283 Acc: 0.5327

Epoch 82/83
----------
train Loss: 0.0095 Acc: 0.7420
val Loss: 0.0342 Acc: 0.5234

Epoch 83/83
----------
train Loss: 0.0096 Acc: 0.7399
val Loss: 0.0307 Acc: 0.5140

Training complete in 7m 48s
Best val Acc: 0.542056

---Fine tuning.---
Epoch 0/83
----------
LR is set to 0.01
train Loss: 0.0105 Acc: 0.6852
val Loss: 0.0237 Acc: 0.5327

Epoch 1/83
----------
train Loss: 0.0062 Acc: 0.8287
val Loss: 0.0324 Acc: 0.5140

Epoch 2/83
----------
train Loss: 0.0034 Acc: 0.9205
val Loss: 0.0337 Acc: 0.5327

Epoch 3/83
----------
train Loss: 0.0018 Acc: 0.9587
val Loss: 0.0313 Acc: 0.5514

Epoch 4/83
----------
train Loss: 0.0014 Acc: 0.9711
val Loss: 0.0401 Acc: 0.5607

Epoch 5/83
----------
train Loss: 0.0010 Acc: 0.9752
val Loss: 0.0440 Acc: 0.5701

Epoch 6/83
----------
train Loss: 0.0010 Acc: 0.9752
val Loss: 0.0297 Acc: 0.5327

Epoch 7/83
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0466 Acc: 0.5327

Epoch 8/83
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0312 Acc: 0.5327

Epoch 9/83
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0294 Acc: 0.5327

Epoch 10/83
----------
train Loss: 0.0006 Acc: 0.9783
val Loss: 0.0341 Acc: 0.5514

Epoch 11/83
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0262 Acc: 0.5514

Epoch 12/83
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0242 Acc: 0.5421

Epoch 13/83
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0355 Acc: 0.5421

Epoch 14/83
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0305 Acc: 0.5421

Epoch 15/83
----------
train Loss: 0.0004 Acc: 0.9814
val Loss: 0.0299 Acc: 0.5421

Epoch 16/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0303 Acc: 0.5421

Epoch 17/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0359 Acc: 0.5327

Epoch 18/83
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0294 Acc: 0.5327

Epoch 19/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0377 Acc: 0.5421

Epoch 20/83
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0318 Acc: 0.5421

Epoch 21/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0174 Acc: 0.5421

Epoch 22/83
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0344 Acc: 0.5421

Epoch 23/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0318 Acc: 0.5327

Epoch 24/83
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0302 Acc: 0.5327

Epoch 25/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0293 Acc: 0.5327

Epoch 26/83
----------
train Loss: 0.0004 Acc: 0.9804
val Loss: 0.0195 Acc: 0.5421

Epoch 27/83
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0334 Acc: 0.5421

Epoch 28/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0356 Acc: 0.5327

Epoch 29/83
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0237 Acc: 0.5327

Epoch 30/83
----------
train Loss: 0.0004 Acc: 0.9804
val Loss: 0.0233 Acc: 0.5327

Epoch 31/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0277 Acc: 0.5327

Epoch 32/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0240 Acc: 0.5327

Epoch 33/83
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0393 Acc: 0.5421

Epoch 34/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0180 Acc: 0.5327

Epoch 35/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0219 Acc: 0.5327

Epoch 36/83
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0271 Acc: 0.5327

Epoch 37/83
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0292 Acc: 0.5327

Epoch 38/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0321 Acc: 0.5327

Epoch 39/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0386 Acc: 0.5327

Epoch 40/83
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0453 Acc: 0.5327

Epoch 41/83
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0338 Acc: 0.5327

Epoch 42/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0290 Acc: 0.5327

Epoch 43/83
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0405 Acc: 0.5327

Epoch 44/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0229 Acc: 0.5327

Epoch 45/83
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0352 Acc: 0.5421

Epoch 46/83
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0308 Acc: 0.5421

Epoch 47/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0331 Acc: 0.5421

Epoch 48/83
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0444 Acc: 0.5327

Epoch 49/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0263 Acc: 0.5327

Epoch 50/83
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0358 Acc: 0.5327

Epoch 51/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0373 Acc: 0.5327

Epoch 52/83
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0369 Acc: 0.5327

Epoch 53/83
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0287 Acc: 0.5327

Epoch 54/83
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0340 Acc: 0.5327

Epoch 55/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0330 Acc: 0.5327

Epoch 56/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0273 Acc: 0.5327

Epoch 57/83
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0306 Acc: 0.5327

Epoch 58/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0353 Acc: 0.5327

Epoch 59/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0339 Acc: 0.5327

Epoch 60/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0432 Acc: 0.5327

Epoch 61/83
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0288 Acc: 0.5421

Epoch 62/83
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0410 Acc: 0.5327

Epoch 63/83
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0289 Acc: 0.5327

Epoch 64/83
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0325 Acc: 0.5327

Epoch 65/83
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0303 Acc: 0.5327

Epoch 66/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0338 Acc: 0.5327

Epoch 67/83
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0407 Acc: 0.5327

Epoch 68/83
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0253 Acc: 0.5327

Epoch 69/83
----------
train Loss: 0.0003 Acc: 0.9876
val Loss: 0.0207 Acc: 0.5327

Epoch 70/83
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0384 Acc: 0.5327

Epoch 71/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0326 Acc: 0.5327

Epoch 72/83
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0370 Acc: 0.5421

Epoch 73/83
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0343 Acc: 0.5421

Epoch 74/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0262 Acc: 0.5421

Epoch 75/83
----------
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0401 Acc: 0.5421

Epoch 76/83
----------
train Loss: 0.0004 Acc: 0.9835
val Loss: 0.0312 Acc: 0.5421

Epoch 77/83
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0362 Acc: 0.5421

Epoch 78/83
----------
train Loss: 0.0004 Acc: 0.9897
val Loss: 0.0346 Acc: 0.5327

Epoch 79/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0512 Acc: 0.5327

Epoch 80/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0344 Acc: 0.5327

Epoch 81/83
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0341 Acc: 0.5327

Epoch 82/83
----------
train Loss: 0.0004 Acc: 0.9845
val Loss: 0.0291 Acc: 0.5327

Epoch 83/83
----------
train Loss: 0.0004 Acc: 0.9866
val Loss: 0.0379 Acc: 0.5327

Training complete in 8m 20s
Best val Acc: 0.570093

---Testing---
Test accuracy: 0.938662
--------------------
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 92 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 79 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 100 %
Accuracy of Mackerel tuna : 96 %
Accuracy of Pacific bluefin tuna : 94 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Yellowfin tuna : 98 %
mean: 0.9193904735958629, std: 0.06839562274268349
--------------------

run info[val: 0.15, epoch: 95, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0288 Acc: 0.1519
val Loss: 0.0295 Acc: 0.2484

Epoch 1/94
----------
train Loss: 0.0233 Acc: 0.2951
val Loss: 0.0243 Acc: 0.3540

Epoch 2/94
----------
train Loss: 0.0200 Acc: 0.4197
val Loss: 0.0231 Acc: 0.4161

Epoch 3/94
----------
train Loss: 0.0170 Acc: 0.5027
val Loss: 0.0239 Acc: 0.4037

Epoch 4/94
----------
train Loss: 0.0158 Acc: 0.5432
val Loss: 0.0229 Acc: 0.4410

Epoch 5/94
----------
train Loss: 0.0144 Acc: 0.5978
val Loss: 0.0235 Acc: 0.3665

Epoch 6/94
----------
train Loss: 0.0136 Acc: 0.6230
val Loss: 0.0218 Acc: 0.4410

Epoch 7/94
----------
train Loss: 0.0125 Acc: 0.6656
val Loss: 0.0218 Acc: 0.4410

Epoch 8/94
----------
train Loss: 0.0119 Acc: 0.6743
val Loss: 0.0210 Acc: 0.4596

Epoch 9/94
----------
train Loss: 0.0114 Acc: 0.6710
val Loss: 0.0234 Acc: 0.4348

Epoch 10/94
----------
train Loss: 0.0119 Acc: 0.6667
val Loss: 0.0222 Acc: 0.4410

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0107 Acc: 0.7301
val Loss: 0.0224 Acc: 0.4658

Epoch 12/94
----------
train Loss: 0.0095 Acc: 0.7705
val Loss: 0.0216 Acc: 0.4907

Epoch 13/94
----------
train Loss: 0.0097 Acc: 0.7650
val Loss: 0.0218 Acc: 0.4658

Epoch 14/94
----------
train Loss: 0.0097 Acc: 0.7694
val Loss: 0.0205 Acc: 0.4410

Epoch 15/94
----------
train Loss: 0.0097 Acc: 0.7694
val Loss: 0.0213 Acc: 0.4720

Epoch 16/94
----------
train Loss: 0.0095 Acc: 0.7628
val Loss: 0.0212 Acc: 0.4534

Epoch 17/94
----------
train Loss: 0.0097 Acc: 0.7661
val Loss: 0.0218 Acc: 0.4720

Epoch 18/94
----------
train Loss: 0.0091 Acc: 0.7770
val Loss: 0.0208 Acc: 0.4720

Epoch 19/94
----------
train Loss: 0.0098 Acc: 0.7443
val Loss: 0.0214 Acc: 0.4845

Epoch 20/94
----------
train Loss: 0.0094 Acc: 0.7792
val Loss: 0.0214 Acc: 0.4534

Epoch 21/94
----------
train Loss: 0.0090 Acc: 0.7650
val Loss: 0.0212 Acc: 0.4534

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0087 Acc: 0.7902
val Loss: 0.0214 Acc: 0.4534

Epoch 23/94
----------
train Loss: 0.0090 Acc: 0.7781
val Loss: 0.0211 Acc: 0.4658

Epoch 24/94
----------
train Loss: 0.0096 Acc: 0.7869
val Loss: 0.0210 Acc: 0.4658

Epoch 25/94
----------
train Loss: 0.0096 Acc: 0.7694
val Loss: 0.0213 Acc: 0.4845

Epoch 26/94
----------
train Loss: 0.0093 Acc: 0.7727
val Loss: 0.0212 Acc: 0.4783

Epoch 27/94
----------
train Loss: 0.0090 Acc: 0.7956
val Loss: 0.0211 Acc: 0.4783

Epoch 28/94
----------
train Loss: 0.0090 Acc: 0.7814
val Loss: 0.0215 Acc: 0.4783

Epoch 29/94
----------
train Loss: 0.0097 Acc: 0.7760
val Loss: 0.0221 Acc: 0.4783

Epoch 30/94
----------
train Loss: 0.0092 Acc: 0.7760
val Loss: 0.0214 Acc: 0.4783

Epoch 31/94
----------
train Loss: 0.0088 Acc: 0.7923
val Loss: 0.0217 Acc: 0.4845

Epoch 32/94
----------
train Loss: 0.0093 Acc: 0.7781
val Loss: 0.0219 Acc: 0.4720

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0089 Acc: 0.7770
val Loss: 0.0218 Acc: 0.4658

Epoch 34/94
----------
train Loss: 0.0090 Acc: 0.7792
val Loss: 0.0217 Acc: 0.4783

Epoch 35/94
----------
train Loss: 0.0091 Acc: 0.7858
val Loss: 0.0210 Acc: 0.4783

Epoch 36/94
----------
train Loss: 0.0088 Acc: 0.7902
val Loss: 0.0217 Acc: 0.4845

Epoch 37/94
----------
train Loss: 0.0094 Acc: 0.7836
val Loss: 0.0206 Acc: 0.4845

Epoch 38/94
----------
train Loss: 0.0093 Acc: 0.7770
val Loss: 0.0220 Acc: 0.4845

Epoch 39/94
----------
train Loss: 0.0091 Acc: 0.7891
val Loss: 0.0207 Acc: 0.4783

Epoch 40/94
----------
train Loss: 0.0089 Acc: 0.7781
val Loss: 0.0218 Acc: 0.4720

Epoch 41/94
----------
train Loss: 0.0090 Acc: 0.7913
val Loss: 0.0212 Acc: 0.4783

Epoch 42/94
----------
train Loss: 0.0088 Acc: 0.7923
val Loss: 0.0211 Acc: 0.4720

Epoch 43/94
----------
train Loss: 0.0087 Acc: 0.7825
val Loss: 0.0207 Acc: 0.4783

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0093 Acc: 0.7781
val Loss: 0.0213 Acc: 0.4907

Epoch 45/94
----------
train Loss: 0.0091 Acc: 0.7760
val Loss: 0.0206 Acc: 0.4720

Epoch 46/94
----------
train Loss: 0.0093 Acc: 0.7617
val Loss: 0.0206 Acc: 0.4658

Epoch 47/94
----------
train Loss: 0.0089 Acc: 0.7825
val Loss: 0.0206 Acc: 0.4596

Epoch 48/94
----------
train Loss: 0.0091 Acc: 0.7923
val Loss: 0.0217 Acc: 0.4845

Epoch 49/94
----------
train Loss: 0.0094 Acc: 0.7825
val Loss: 0.0209 Acc: 0.4845

Epoch 50/94
----------
train Loss: 0.0092 Acc: 0.7836
val Loss: 0.0214 Acc: 0.4783

Epoch 51/94
----------
train Loss: 0.0092 Acc: 0.7803
val Loss: 0.0215 Acc: 0.4783

Epoch 52/94
----------
train Loss: 0.0090 Acc: 0.7760
val Loss: 0.0214 Acc: 0.4783

Epoch 53/94
----------
train Loss: 0.0092 Acc: 0.7880
val Loss: 0.0212 Acc: 0.4596

Epoch 54/94
----------
train Loss: 0.0094 Acc: 0.7683
val Loss: 0.0212 Acc: 0.4720

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0094 Acc: 0.7825
val Loss: 0.0211 Acc: 0.4783

Epoch 56/94
----------
train Loss: 0.0094 Acc: 0.7683
val Loss: 0.0215 Acc: 0.4783

Epoch 57/94
----------
train Loss: 0.0088 Acc: 0.7945
val Loss: 0.0210 Acc: 0.4783

Epoch 58/94
----------
train Loss: 0.0094 Acc: 0.7760
val Loss: 0.0212 Acc: 0.4783

Epoch 59/94
----------
train Loss: 0.0089 Acc: 0.7880
val Loss: 0.0210 Acc: 0.4720

Epoch 60/94
----------
train Loss: 0.0090 Acc: 0.7814
val Loss: 0.0213 Acc: 0.4907

Epoch 61/94
----------
train Loss: 0.0089 Acc: 0.7902
val Loss: 0.0213 Acc: 0.4720

Epoch 62/94
----------
train Loss: 0.0090 Acc: 0.7825
val Loss: 0.0212 Acc: 0.4720

Epoch 63/94
----------
train Loss: 0.0095 Acc: 0.7847
val Loss: 0.0212 Acc: 0.4720

Epoch 64/94
----------
train Loss: 0.0099 Acc: 0.7672
val Loss: 0.0213 Acc: 0.4720

Epoch 65/94
----------
train Loss: 0.0089 Acc: 0.7913
val Loss: 0.0223 Acc: 0.4783

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0089 Acc: 0.7814
val Loss: 0.0213 Acc: 0.4720

Epoch 67/94
----------
train Loss: 0.0093 Acc: 0.7749
val Loss: 0.0211 Acc: 0.4783

Epoch 68/94
----------
train Loss: 0.0089 Acc: 0.7891
val Loss: 0.0206 Acc: 0.4783

Epoch 69/94
----------
train Loss: 0.0089 Acc: 0.7781
val Loss: 0.0210 Acc: 0.4783

Epoch 70/94
----------
train Loss: 0.0088 Acc: 0.7923
val Loss: 0.0207 Acc: 0.4845

Epoch 71/94
----------
train Loss: 0.0091 Acc: 0.7716
val Loss: 0.0217 Acc: 0.4783

Epoch 72/94
----------
train Loss: 0.0091 Acc: 0.7880
val Loss: 0.0209 Acc: 0.4720

Epoch 73/94
----------
train Loss: 0.0092 Acc: 0.7738
val Loss: 0.0211 Acc: 0.4720

Epoch 74/94
----------
train Loss: 0.0098 Acc: 0.7716
val Loss: 0.0207 Acc: 0.4783

Epoch 75/94
----------
train Loss: 0.0090 Acc: 0.7738
val Loss: 0.0207 Acc: 0.4720

Epoch 76/94
----------
train Loss: 0.0091 Acc: 0.7956
val Loss: 0.0213 Acc: 0.4720

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0094 Acc: 0.7738
val Loss: 0.0211 Acc: 0.4783

Epoch 78/94
----------
train Loss: 0.0093 Acc: 0.7694
val Loss: 0.0210 Acc: 0.4783

Epoch 79/94
----------
train Loss: 0.0092 Acc: 0.7738
val Loss: 0.0212 Acc: 0.4720

Epoch 80/94
----------
train Loss: 0.0092 Acc: 0.7934
val Loss: 0.0211 Acc: 0.4720

Epoch 81/94
----------
train Loss: 0.0090 Acc: 0.7792
val Loss: 0.0206 Acc: 0.4658

Epoch 82/94
----------
train Loss: 0.0091 Acc: 0.7770
val Loss: 0.0214 Acc: 0.4783

Epoch 83/94
----------
train Loss: 0.0095 Acc: 0.7902
val Loss: 0.0214 Acc: 0.4845

Epoch 84/94
----------
train Loss: 0.0091 Acc: 0.7749
val Loss: 0.0215 Acc: 0.4720

Epoch 85/94
----------
train Loss: 0.0093 Acc: 0.7880
val Loss: 0.0206 Acc: 0.4845

Epoch 86/94
----------
train Loss: 0.0088 Acc: 0.7858
val Loss: 0.0213 Acc: 0.4907

Epoch 87/94
----------
train Loss: 0.0086 Acc: 0.8000
val Loss: 0.0211 Acc: 0.4783

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0093 Acc: 0.7781
val Loss: 0.0205 Acc: 0.4720

Epoch 89/94
----------
train Loss: 0.0090 Acc: 0.7738
val Loss: 0.0209 Acc: 0.4658

Epoch 90/94
----------
train Loss: 0.0089 Acc: 0.7814
val Loss: 0.0215 Acc: 0.4783

Epoch 91/94
----------
train Loss: 0.0094 Acc: 0.7738
val Loss: 0.0210 Acc: 0.4783

Epoch 92/94
----------
train Loss: 0.0093 Acc: 0.7869
val Loss: 0.0216 Acc: 0.4720

Epoch 93/94
----------
train Loss: 0.0092 Acc: 0.7814
val Loss: 0.0210 Acc: 0.4720

Epoch 94/94
----------
train Loss: 0.0090 Acc: 0.7650
val Loss: 0.0213 Acc: 0.4783

Training complete in 8m 44s
Best val Acc: 0.490683

---Fine tuning.---
Epoch 0/94
----------
LR is set to 0.01
train Loss: 0.0099 Acc: 0.7235
val Loss: 0.0218 Acc: 0.4596

Epoch 1/94
----------
train Loss: 0.0059 Acc: 0.8557
val Loss: 0.0214 Acc: 0.5155

Epoch 2/94
----------
train Loss: 0.0036 Acc: 0.9158
val Loss: 0.0220 Acc: 0.4907

Epoch 3/94
----------
train Loss: 0.0029 Acc: 0.9432
val Loss: 0.0227 Acc: 0.4907

Epoch 4/94
----------
train Loss: 0.0023 Acc: 0.9432
val Loss: 0.0240 Acc: 0.5155

Epoch 5/94
----------
train Loss: 0.0017 Acc: 0.9661
val Loss: 0.0247 Acc: 0.5093

Epoch 6/94
----------
train Loss: 0.0014 Acc: 0.9694
val Loss: 0.0253 Acc: 0.5155

Epoch 7/94
----------
train Loss: 0.0016 Acc: 0.9716
val Loss: 0.0258 Acc: 0.4720

Epoch 8/94
----------
train Loss: 0.0020 Acc: 0.9410
val Loss: 0.0286 Acc: 0.4410

Epoch 9/94
----------
train Loss: 0.0017 Acc: 0.9683
val Loss: 0.0253 Acc: 0.4845

Epoch 10/94
----------
train Loss: 0.0015 Acc: 0.9749
val Loss: 0.0268 Acc: 0.5031

Epoch 11/94
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9803
val Loss: 0.0233 Acc: 0.5342

Epoch 12/94
----------
train Loss: 0.0007 Acc: 0.9781
val Loss: 0.0246 Acc: 0.5342

Epoch 13/94
----------
train Loss: 0.0006 Acc: 0.9760
val Loss: 0.0251 Acc: 0.5280

Epoch 14/94
----------
train Loss: 0.0006 Acc: 0.9825
val Loss: 0.0236 Acc: 0.5155

Epoch 15/94
----------
train Loss: 0.0007 Acc: 0.9847
val Loss: 0.0234 Acc: 0.5155

Epoch 16/94
----------
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0239 Acc: 0.5155

Epoch 17/94
----------
train Loss: 0.0006 Acc: 0.9770
val Loss: 0.0248 Acc: 0.5155

Epoch 18/94
----------
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0241 Acc: 0.5155

Epoch 19/94
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0240 Acc: 0.5217

Epoch 20/94
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0252 Acc: 0.5217

Epoch 21/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0234 Acc: 0.5155

Epoch 22/94
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0227 Acc: 0.5217

Epoch 23/94
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0234 Acc: 0.5155

Epoch 24/94
----------
train Loss: 0.0005 Acc: 0.9792
val Loss: 0.0245 Acc: 0.5155

Epoch 25/94
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0250 Acc: 0.5093

Epoch 26/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0250 Acc: 0.5155

Epoch 27/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0236 Acc: 0.5217

Epoch 28/94
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0243 Acc: 0.5217

Epoch 29/94
----------
train Loss: 0.0006 Acc: 0.9803
val Loss: 0.0238 Acc: 0.5280

Epoch 30/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0248 Acc: 0.5155

Epoch 31/94
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0239 Acc: 0.5155

Epoch 32/94
----------
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0236 Acc: 0.5217

Epoch 33/94
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0235 Acc: 0.5280

Epoch 34/94
----------
train Loss: 0.0006 Acc: 0.9858
val Loss: 0.0251 Acc: 0.5155

Epoch 35/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0252 Acc: 0.5093

Epoch 36/94
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0243 Acc: 0.5217

Epoch 37/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0233 Acc: 0.5155

Epoch 38/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0237 Acc: 0.5217

Epoch 39/94
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0233 Acc: 0.5217

Epoch 40/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0257 Acc: 0.5155

Epoch 41/94
----------
train Loss: 0.0006 Acc: 0.9858
val Loss: 0.0242 Acc: 0.5093

Epoch 42/94
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0246 Acc: 0.5217

Epoch 43/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0239 Acc: 0.5217

Epoch 44/94
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0245 Acc: 0.5155

Epoch 45/94
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0240 Acc: 0.5217

Epoch 46/94
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0243 Acc: 0.5155

Epoch 47/94
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0245 Acc: 0.5155

Epoch 48/94
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0254 Acc: 0.5155

Epoch 49/94
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0252 Acc: 0.5280

Epoch 50/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0246 Acc: 0.5217

Epoch 51/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0246 Acc: 0.5280

Epoch 52/94
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0241 Acc: 0.5217

Epoch 53/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0258 Acc: 0.5217

Epoch 54/94
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0244 Acc: 0.5093

Epoch 55/94
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0239 Acc: 0.5155

Epoch 56/94
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0245 Acc: 0.5155

Epoch 57/94
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0231 Acc: 0.5093

Epoch 58/94
----------
train Loss: 0.0004 Acc: 0.9847
val Loss: 0.0231 Acc: 0.5155

Epoch 59/94
----------
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0246 Acc: 0.5217

Epoch 60/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0241 Acc: 0.5155

Epoch 61/94
----------
train Loss: 0.0004 Acc: 0.9836
val Loss: 0.0242 Acc: 0.5217

Epoch 62/94
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0236 Acc: 0.5217

Epoch 63/94
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0240 Acc: 0.5155

Epoch 64/94
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0246 Acc: 0.5217

Epoch 65/94
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0247 Acc: 0.5155

Epoch 66/94
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0241 Acc: 0.5155

Epoch 67/94
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0243 Acc: 0.5217

Epoch 68/94
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0243 Acc: 0.5155

Epoch 69/94
----------
train Loss: 0.0007 Acc: 0.9836
val Loss: 0.0260 Acc: 0.5155

Epoch 70/94
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0258 Acc: 0.5155

Epoch 71/94
----------
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0244 Acc: 0.5155

Epoch 72/94
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0241 Acc: 0.5280

Epoch 73/94
----------
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0253 Acc: 0.5155

Epoch 74/94
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0238 Acc: 0.5093

Epoch 75/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0252 Acc: 0.5093

Epoch 76/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5217

Epoch 77/94
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5217

Epoch 78/94
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0249 Acc: 0.5155

Epoch 79/94
----------
train Loss: 0.0005 Acc: 0.9880
val Loss: 0.0235 Acc: 0.5155

Epoch 80/94
----------
train Loss: 0.0004 Acc: 0.9803
val Loss: 0.0244 Acc: 0.5217

Epoch 81/94
----------
train Loss: 0.0004 Acc: 0.9902
val Loss: 0.0249 Acc: 0.5217

Epoch 82/94
----------
train Loss: 0.0006 Acc: 0.9869
val Loss: 0.0234 Acc: 0.5217

Epoch 83/94
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0241 Acc: 0.5217

Epoch 84/94
----------
train Loss: 0.0005 Acc: 0.9869
val Loss: 0.0242 Acc: 0.5155

Epoch 85/94
----------
train Loss: 0.0005 Acc: 0.9825
val Loss: 0.0250 Acc: 0.5217

Epoch 86/94
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0244 Acc: 0.5093

Epoch 87/94
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0240 Acc: 0.5217

Epoch 88/94
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0006 Acc: 0.9814
val Loss: 0.0245 Acc: 0.5093

Epoch 89/94
----------
train Loss: 0.0005 Acc: 0.9847
val Loss: 0.0238 Acc: 0.5093

Epoch 90/94
----------
train Loss: 0.0005 Acc: 0.9858
val Loss: 0.0247 Acc: 0.5155

Epoch 91/94
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0246 Acc: 0.5217

Epoch 92/94
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0249 Acc: 0.5217

Epoch 93/94
----------
train Loss: 0.0004 Acc: 0.9825
val Loss: 0.0249 Acc: 0.5031

Epoch 94/94
----------
train Loss: 0.0006 Acc: 0.9847
val Loss: 0.0244 Acc: 0.5217

Training complete in 9m 22s
Best val Acc: 0.534161

---Testing---
Test accuracy: 0.908922
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 87 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 87 %
Accuracy of Frigate tuna : 79 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 88 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8811891815205666, std: 0.07016833730588443
--------------------

run info[val: 0.2, epoch: 76, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0270 Acc: 0.1731
val Loss: 0.0343 Acc: 0.2651

Epoch 1/75
----------
train Loss: 0.0226 Acc: 0.3333
val Loss: 0.0293 Acc: 0.3488

Epoch 2/75
----------
train Loss: 0.0187 Acc: 0.4204
val Loss: 0.0274 Acc: 0.4372

Epoch 3/75
----------
LR is set to 0.001
train Loss: 0.0166 Acc: 0.5273
val Loss: 0.0266 Acc: 0.4186

Epoch 4/75
----------
train Loss: 0.0159 Acc: 0.5354
val Loss: 0.0275 Acc: 0.4000

Epoch 5/75
----------
train Loss: 0.0156 Acc: 0.5285
val Loss: 0.0247 Acc: 0.3907

Epoch 6/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0155 Acc: 0.5470
val Loss: 0.0255 Acc: 0.3953

Epoch 7/75
----------
train Loss: 0.0155 Acc: 0.5587
val Loss: 0.0256 Acc: 0.3953

Epoch 8/75
----------
train Loss: 0.0155 Acc: 0.5575
val Loss: 0.0263 Acc: 0.3953

Epoch 9/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0155 Acc: 0.5377
val Loss: 0.0247 Acc: 0.3953

Epoch 10/75
----------
train Loss: 0.0154 Acc: 0.5726
val Loss: 0.0254 Acc: 0.4000

Epoch 11/75
----------
train Loss: 0.0154 Acc: 0.5517
val Loss: 0.0233 Acc: 0.4000

Epoch 12/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0152 Acc: 0.5656
val Loss: 0.0260 Acc: 0.3953

Epoch 13/75
----------
train Loss: 0.0154 Acc: 0.5563
val Loss: 0.0258 Acc: 0.3907

Epoch 14/75
----------
train Loss: 0.0154 Acc: 0.5563
val Loss: 0.0254 Acc: 0.3907

Epoch 15/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0154 Acc: 0.5656
val Loss: 0.0252 Acc: 0.3907

Epoch 16/75
----------
train Loss: 0.0154 Acc: 0.5598
val Loss: 0.0246 Acc: 0.3907

Epoch 17/75
----------
train Loss: 0.0154 Acc: 0.5563
val Loss: 0.0254 Acc: 0.3907

Epoch 18/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0155 Acc: 0.5598
val Loss: 0.0280 Acc: 0.3907

Epoch 19/75
----------
train Loss: 0.0154 Acc: 0.5610
val Loss: 0.0233 Acc: 0.3907

Epoch 20/75
----------
train Loss: 0.0154 Acc: 0.5587
val Loss: 0.0267 Acc: 0.3953

Epoch 21/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0155 Acc: 0.5575
val Loss: 0.0267 Acc: 0.4000

Epoch 22/75
----------
train Loss: 0.0154 Acc: 0.5645
val Loss: 0.0267 Acc: 0.3907

Epoch 23/75
----------
train Loss: 0.0153 Acc: 0.5587
val Loss: 0.0252 Acc: 0.3907

Epoch 24/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0153 Acc: 0.5494
val Loss: 0.0235 Acc: 0.3907

Epoch 25/75
----------
train Loss: 0.0155 Acc: 0.5528
val Loss: 0.0266 Acc: 0.3907

Epoch 26/75
----------
train Loss: 0.0155 Acc: 0.5668
val Loss: 0.0267 Acc: 0.3907

Epoch 27/75
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0153 Acc: 0.5679
val Loss: 0.0255 Acc: 0.3953

Epoch 28/75
----------
train Loss: 0.0154 Acc: 0.5598
val Loss: 0.0260 Acc: 0.3953

Epoch 29/75
----------
train Loss: 0.0155 Acc: 0.5598
val Loss: 0.0258 Acc: 0.3907

Epoch 30/75
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0152 Acc: 0.5772
val Loss: 0.0242 Acc: 0.3907

Epoch 31/75
----------
train Loss: 0.0154 Acc: 0.5540
val Loss: 0.0237 Acc: 0.3953

Epoch 32/75
----------
train Loss: 0.0154 Acc: 0.5552
val Loss: 0.0245 Acc: 0.3953

Epoch 33/75
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0155 Acc: 0.5633
val Loss: 0.0253 Acc: 0.3953

Epoch 34/75
----------
train Loss: 0.0154 Acc: 0.5633
val Loss: 0.0259 Acc: 0.3953

Epoch 35/75
----------
train Loss: 0.0155 Acc: 0.5575
val Loss: 0.0268 Acc: 0.3953

Epoch 36/75
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0153 Acc: 0.5691
val Loss: 0.0241 Acc: 0.3953

Epoch 37/75
----------
train Loss: 0.0155 Acc: 0.5563
val Loss: 0.0273 Acc: 0.3907

Epoch 38/75
----------
train Loss: 0.0155 Acc: 0.5575
val Loss: 0.0251 Acc: 0.3907

Epoch 39/75
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0153 Acc: 0.5645
val Loss: 0.0257 Acc: 0.4047

Epoch 40/75
----------
train Loss: 0.0153 Acc: 0.5668
val Loss: 0.0251 Acc: 0.4000

Epoch 41/75
----------
train Loss: 0.0154 Acc: 0.5633
val Loss: 0.0239 Acc: 0.3907

Epoch 42/75
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0152 Acc: 0.5668
val Loss: 0.0257 Acc: 0.3953

Epoch 43/75
----------
train Loss: 0.0155 Acc: 0.5587
val Loss: 0.0262 Acc: 0.3953

Epoch 44/75
----------
train Loss: 0.0153 Acc: 0.5714
val Loss: 0.0273 Acc: 0.3953

Epoch 45/75
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0154 Acc: 0.5738
val Loss: 0.0271 Acc: 0.3953

Epoch 46/75
----------
train Loss: 0.0156 Acc: 0.5540
val Loss: 0.0276 Acc: 0.3953

Epoch 47/75
----------
train Loss: 0.0154 Acc: 0.5552
val Loss: 0.0261 Acc: 0.3860

Epoch 48/75
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0156 Acc: 0.5482
val Loss: 0.0280 Acc: 0.3907

Epoch 49/75
----------
train Loss: 0.0154 Acc: 0.5621
val Loss: 0.0245 Acc: 0.3953

Epoch 50/75
----------
train Loss: 0.0153 Acc: 0.5668
val Loss: 0.0255 Acc: 0.3907

Epoch 51/75
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0154 Acc: 0.5726
val Loss: 0.0252 Acc: 0.3953

Epoch 52/75
----------
train Loss: 0.0155 Acc: 0.5575
val Loss: 0.0251 Acc: 0.3953

Epoch 53/75
----------
train Loss: 0.0154 Acc: 0.5517
val Loss: 0.0248 Acc: 0.4000

Epoch 54/75
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0155 Acc: 0.5540
val Loss: 0.0247 Acc: 0.3953

Epoch 55/75
----------
train Loss: 0.0154 Acc: 0.5459
val Loss: 0.0239 Acc: 0.3953

Epoch 56/75
----------
train Loss: 0.0153 Acc: 0.5575
val Loss: 0.0253 Acc: 0.3953

Epoch 57/75
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0153 Acc: 0.5645
val Loss: 0.0223 Acc: 0.3953

Epoch 58/75
----------
train Loss: 0.0154 Acc: 0.5598
val Loss: 0.0243 Acc: 0.3953

Epoch 59/75
----------
train Loss: 0.0155 Acc: 0.5528
val Loss: 0.0247 Acc: 0.3953

Epoch 60/75
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0156 Acc: 0.5482
val Loss: 0.0270 Acc: 0.3907

Epoch 61/75
----------
train Loss: 0.0154 Acc: 0.5517
val Loss: 0.0255 Acc: 0.3953

Epoch 62/75
----------
train Loss: 0.0153 Acc: 0.5575
val Loss: 0.0249 Acc: 0.3953

Epoch 63/75
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0153 Acc: 0.5691
val Loss: 0.0260 Acc: 0.3953

Epoch 64/75
----------
train Loss: 0.0154 Acc: 0.5668
val Loss: 0.0245 Acc: 0.3953

Epoch 65/75
----------
train Loss: 0.0153 Acc: 0.5668
val Loss: 0.0239 Acc: 0.3953

Epoch 66/75
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0153 Acc: 0.5633
val Loss: 0.0253 Acc: 0.3953

Epoch 67/75
----------
train Loss: 0.0154 Acc: 0.5494
val Loss: 0.0259 Acc: 0.3953

Epoch 68/75
----------
train Loss: 0.0153 Acc: 0.5563
val Loss: 0.0252 Acc: 0.4000

Epoch 69/75
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0154 Acc: 0.5610
val Loss: 0.0263 Acc: 0.3953

Epoch 70/75
----------
train Loss: 0.0154 Acc: 0.5494
val Loss: 0.0287 Acc: 0.3953

Epoch 71/75
----------
train Loss: 0.0155 Acc: 0.5598
val Loss: 0.0245 Acc: 0.4000

Epoch 72/75
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0154 Acc: 0.5587
val Loss: 0.0277 Acc: 0.3953

Epoch 73/75
----------
train Loss: 0.0155 Acc: 0.5552
val Loss: 0.0249 Acc: 0.3907

Epoch 74/75
----------
train Loss: 0.0154 Acc: 0.5645
val Loss: 0.0248 Acc: 0.3953

Epoch 75/75
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0155 Acc: 0.5436
val Loss: 0.0266 Acc: 0.4000

Training complete in 6m 44s
Best val Acc: 0.437209

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0165 Acc: 0.4936
val Loss: 0.0229 Acc: 0.4791

Epoch 1/75
----------
train Loss: 0.0103 Acc: 0.7178
val Loss: 0.0233 Acc: 0.4930

Epoch 2/75
----------
train Loss: 0.0067 Acc: 0.8362
val Loss: 0.0233 Acc: 0.5395

Epoch 3/75
----------
LR is set to 0.001
train Loss: 0.0041 Acc: 0.9199
val Loss: 0.0202 Acc: 0.5628

Epoch 4/75
----------
train Loss: 0.0036 Acc: 0.9338
val Loss: 0.0182 Acc: 0.5581

Epoch 5/75
----------
train Loss: 0.0034 Acc: 0.9489
val Loss: 0.0215 Acc: 0.5581

Epoch 6/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.9477
val Loss: 0.0189 Acc: 0.5581

Epoch 7/75
----------
train Loss: 0.0032 Acc: 0.9512
val Loss: 0.0203 Acc: 0.5535

Epoch 8/75
----------
train Loss: 0.0032 Acc: 0.9466
val Loss: 0.0179 Acc: 0.5581

Epoch 9/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0033 Acc: 0.9431
val Loss: 0.0176 Acc: 0.5581

Epoch 10/75
----------
train Loss: 0.0032 Acc: 0.9559
val Loss: 0.0201 Acc: 0.5581

Epoch 11/75
----------
train Loss: 0.0031 Acc: 0.9431
val Loss: 0.0199 Acc: 0.5628

Epoch 12/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0032 Acc: 0.9582
val Loss: 0.0182 Acc: 0.5581

Epoch 13/75
----------
train Loss: 0.0032 Acc: 0.9431
val Loss: 0.0190 Acc: 0.5581

Epoch 14/75
----------
train Loss: 0.0031 Acc: 0.9512
val Loss: 0.0212 Acc: 0.5581

Epoch 15/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0030 Acc: 0.9547
val Loss: 0.0216 Acc: 0.5581

Epoch 16/75
----------
train Loss: 0.0031 Acc: 0.9489
val Loss: 0.0245 Acc: 0.5581

Epoch 17/75
----------
train Loss: 0.0029 Acc: 0.9593
val Loss: 0.0187 Acc: 0.5581

Epoch 18/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0031 Acc: 0.9570
val Loss: 0.0186 Acc: 0.5581

Epoch 19/75
----------
train Loss: 0.0033 Acc: 0.9489
val Loss: 0.0195 Acc: 0.5535

Epoch 20/75
----------
train Loss: 0.0032 Acc: 0.9501
val Loss: 0.0204 Acc: 0.5535

Epoch 21/75
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0032 Acc: 0.9419
val Loss: 0.0189 Acc: 0.5535

Epoch 22/75
----------
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0204 Acc: 0.5581

Epoch 23/75
----------
train Loss: 0.0031 Acc: 0.9501
val Loss: 0.0232 Acc: 0.5581

Epoch 24/75
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0031 Acc: 0.9570
val Loss: 0.0190 Acc: 0.5535

Epoch 25/75
----------
train Loss: 0.0030 Acc: 0.9512
val Loss: 0.0183 Acc: 0.5628

Epoch 26/75
----------
train Loss: 0.0032 Acc: 0.9501
val Loss: 0.0188 Acc: 0.5581

Epoch 27/75
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0030 Acc: 0.9512
val Loss: 0.0188 Acc: 0.5581

Epoch 28/75
----------
train Loss: 0.0032 Acc: 0.9524
val Loss: 0.0210 Acc: 0.5535

Epoch 29/75
----------
train Loss: 0.0031 Acc: 0.9419
val Loss: 0.0183 Acc: 0.5535

Epoch 30/75
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0031 Acc: 0.9454
val Loss: 0.0218 Acc: 0.5581

Epoch 31/75
----------
train Loss: 0.0032 Acc: 0.9501
val Loss: 0.0205 Acc: 0.5581

Epoch 32/75
----------
train Loss: 0.0031 Acc: 0.9559
val Loss: 0.0178 Acc: 0.5581

Epoch 33/75
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0032 Acc: 0.9477
val Loss: 0.0189 Acc: 0.5581

Epoch 34/75
----------
train Loss: 0.0032 Acc: 0.9454
val Loss: 0.0197 Acc: 0.5581

Epoch 35/75
----------
train Loss: 0.0031 Acc: 0.9501
val Loss: 0.0196 Acc: 0.5581

Epoch 36/75
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0032 Acc: 0.9419
val Loss: 0.0190 Acc: 0.5581

Epoch 37/75
----------
train Loss: 0.0031 Acc: 0.9466
val Loss: 0.0196 Acc: 0.5581

Epoch 38/75
----------
train Loss: 0.0030 Acc: 0.9605
val Loss: 0.0205 Acc: 0.5581

Epoch 39/75
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0033 Acc: 0.9431
val Loss: 0.0177 Acc: 0.5628

Epoch 40/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0189 Acc: 0.5535

Epoch 41/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0186 Acc: 0.5581

Epoch 42/75
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0031 Acc: 0.9466
val Loss: 0.0199 Acc: 0.5628

Epoch 43/75
----------
train Loss: 0.0031 Acc: 0.9431
val Loss: 0.0185 Acc: 0.5581

Epoch 44/75
----------
train Loss: 0.0033 Acc: 0.9466
val Loss: 0.0209 Acc: 0.5628

Epoch 45/75
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0031 Acc: 0.9443
val Loss: 0.0188 Acc: 0.5628

Epoch 46/75
----------
train Loss: 0.0030 Acc: 0.9512
val Loss: 0.0197 Acc: 0.5581

Epoch 47/75
----------
train Loss: 0.0031 Acc: 0.9535
val Loss: 0.0205 Acc: 0.5581

Epoch 48/75
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0032 Acc: 0.9431
val Loss: 0.0191 Acc: 0.5581

Epoch 49/75
----------
train Loss: 0.0032 Acc: 0.9501
val Loss: 0.0188 Acc: 0.5535

Epoch 50/75
----------
train Loss: 0.0031 Acc: 0.9501
val Loss: 0.0202 Acc: 0.5581

Epoch 51/75
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0031 Acc: 0.9512
val Loss: 0.0186 Acc: 0.5581

Epoch 52/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0188 Acc: 0.5581

Epoch 53/75
----------
train Loss: 0.0031 Acc: 0.9443
val Loss: 0.0198 Acc: 0.5628

Epoch 54/75
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0032 Acc: 0.9443
val Loss: 0.0178 Acc: 0.5581

Epoch 55/75
----------
train Loss: 0.0032 Acc: 0.9419
val Loss: 0.0206 Acc: 0.5628

Epoch 56/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0182 Acc: 0.5628

Epoch 57/75
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0192 Acc: 0.5535

Epoch 58/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0206 Acc: 0.5581

Epoch 59/75
----------
train Loss: 0.0032 Acc: 0.9443
val Loss: 0.0176 Acc: 0.5674

Epoch 60/75
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0031 Acc: 0.9501
val Loss: 0.0179 Acc: 0.5628

Epoch 61/75
----------
train Loss: 0.0031 Acc: 0.9384
val Loss: 0.0212 Acc: 0.5628

Epoch 62/75
----------
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0175 Acc: 0.5581

Epoch 63/75
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0032 Acc: 0.9512
val Loss: 0.0194 Acc: 0.5581

Epoch 64/75
----------
train Loss: 0.0031 Acc: 0.9535
val Loss: 0.0174 Acc: 0.5581

Epoch 65/75
----------
train Loss: 0.0030 Acc: 0.9535
val Loss: 0.0197 Acc: 0.5581

Epoch 66/75
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0031 Acc: 0.9466
val Loss: 0.0189 Acc: 0.5581

Epoch 67/75
----------
train Loss: 0.0032 Acc: 0.9477
val Loss: 0.0206 Acc: 0.5628

Epoch 68/75
----------
train Loss: 0.0033 Acc: 0.9454
val Loss: 0.0216 Acc: 0.5628

Epoch 69/75
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0031 Acc: 0.9477
val Loss: 0.0193 Acc: 0.5581

Epoch 70/75
----------
train Loss: 0.0031 Acc: 0.9501
val Loss: 0.0196 Acc: 0.5581

Epoch 71/75
----------
train Loss: 0.0031 Acc: 0.9524
val Loss: 0.0202 Acc: 0.5581

Epoch 72/75
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0030 Acc: 0.9477
val Loss: 0.0199 Acc: 0.5581

Epoch 73/75
----------
train Loss: 0.0033 Acc: 0.9501
val Loss: 0.0178 Acc: 0.5581

Epoch 74/75
----------
train Loss: 0.0031 Acc: 0.9547
val Loss: 0.0206 Acc: 0.5581

Epoch 75/75
----------
LR is set to 1.0000000000000015e-27
train Loss: 0.0031 Acc: 0.9419
val Loss: 0.0180 Acc: 0.5628

Training complete in 7m 10s
Best val Acc: 0.567442

---Testing---
Test accuracy: 0.880112
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 82 %
Accuracy of Bigeye tuna : 77 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 92 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 92 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 35 %
Accuracy of Southern bluefin tuna : 78 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8293228620723883, std: 0.14874702100989462
--------------------

run info[val: 0.25, epoch: 87, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0292 Acc: 0.1413
val Loss: 0.0274 Acc: 0.2082

Epoch 1/86
----------
train Loss: 0.0254 Acc: 0.2652
val Loss: 0.0276 Acc: 0.2454

Epoch 2/86
----------
train Loss: 0.0215 Acc: 0.3705
val Loss: 0.0221 Acc: 0.3457

Epoch 3/86
----------
train Loss: 0.0191 Acc: 0.4424
val Loss: 0.0212 Acc: 0.4424

Epoch 4/86
----------
train Loss: 0.0180 Acc: 0.5031
val Loss: 0.0201 Acc: 0.4275

Epoch 5/86
----------
train Loss: 0.0155 Acc: 0.5477
val Loss: 0.0213 Acc: 0.4387

Epoch 6/86
----------
train Loss: 0.0163 Acc: 0.5787
val Loss: 0.0200 Acc: 0.4089

Epoch 7/86
----------
train Loss: 0.0146 Acc: 0.5799
val Loss: 0.0207 Acc: 0.4424

Epoch 8/86
----------
train Loss: 0.0147 Acc: 0.6010
val Loss: 0.0195 Acc: 0.4647

Epoch 9/86
----------
train Loss: 0.0146 Acc: 0.6072
val Loss: 0.0195 Acc: 0.4572

Epoch 10/86
----------
LR is set to 0.001
train Loss: 0.0143 Acc: 0.6196
val Loss: 0.0187 Acc: 0.4796

Epoch 11/86
----------
train Loss: 0.0122 Acc: 0.6530
val Loss: 0.0187 Acc: 0.4944

Epoch 12/86
----------
train Loss: 0.0136 Acc: 0.6704
val Loss: 0.0191 Acc: 0.4647

Epoch 13/86
----------
train Loss: 0.0120 Acc: 0.6605
val Loss: 0.0187 Acc: 0.4944

Epoch 14/86
----------
train Loss: 0.0124 Acc: 0.6753
val Loss: 0.0189 Acc: 0.4944

Epoch 15/86
----------
train Loss: 0.0125 Acc: 0.6716
val Loss: 0.0187 Acc: 0.4833

Epoch 16/86
----------
train Loss: 0.0118 Acc: 0.6766
val Loss: 0.0187 Acc: 0.4870

Epoch 17/86
----------
train Loss: 0.0125 Acc: 0.6729
val Loss: 0.0184 Acc: 0.4870

Epoch 18/86
----------
train Loss: 0.0130 Acc: 0.6865
val Loss: 0.0184 Acc: 0.4833

Epoch 19/86
----------
train Loss: 0.0120 Acc: 0.6853
val Loss: 0.0183 Acc: 0.4796

Epoch 20/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0121 Acc: 0.6667
val Loss: 0.0183 Acc: 0.4758

Epoch 21/86
----------
train Loss: 0.0117 Acc: 0.6877
val Loss: 0.0184 Acc: 0.4833

Epoch 22/86
----------
train Loss: 0.0133 Acc: 0.6667
val Loss: 0.0186 Acc: 0.4870

Epoch 23/86
----------
train Loss: 0.0123 Acc: 0.6853
val Loss: 0.0185 Acc: 0.4981

Epoch 24/86
----------
train Loss: 0.0134 Acc: 0.7038
val Loss: 0.0183 Acc: 0.5056

Epoch 25/86
----------
train Loss: 0.0117 Acc: 0.7038
val Loss: 0.0183 Acc: 0.4758

Epoch 26/86
----------
train Loss: 0.0115 Acc: 0.7051
val Loss: 0.0187 Acc: 0.4944

Epoch 27/86
----------
train Loss: 0.0125 Acc: 0.6815
val Loss: 0.0185 Acc: 0.4981

Epoch 28/86
----------
train Loss: 0.0118 Acc: 0.7038
val Loss: 0.0187 Acc: 0.4870

Epoch 29/86
----------
train Loss: 0.0116 Acc: 0.6902
val Loss: 0.0184 Acc: 0.4981

Epoch 30/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0116 Acc: 0.7001
val Loss: 0.0186 Acc: 0.4944

Epoch 31/86
----------
train Loss: 0.0118 Acc: 0.6976
val Loss: 0.0183 Acc: 0.5056

Epoch 32/86
----------
train Loss: 0.0118 Acc: 0.7001
val Loss: 0.0186 Acc: 0.4944

Epoch 33/86
----------
train Loss: 0.0114 Acc: 0.6890
val Loss: 0.0188 Acc: 0.4944

Epoch 34/86
----------
train Loss: 0.0118 Acc: 0.7125
val Loss: 0.0186 Acc: 0.5019

Epoch 35/86
----------
train Loss: 0.0120 Acc: 0.7125
val Loss: 0.0183 Acc: 0.4981

Epoch 36/86
----------
train Loss: 0.0124 Acc: 0.6791
val Loss: 0.0187 Acc: 0.5019

Epoch 37/86
----------
train Loss: 0.0111 Acc: 0.6964
val Loss: 0.0184 Acc: 0.4907

Epoch 38/86
----------
train Loss: 0.0113 Acc: 0.7051
val Loss: 0.0185 Acc: 0.5019

Epoch 39/86
----------
train Loss: 0.0116 Acc: 0.7001
val Loss: 0.0186 Acc: 0.5056

Epoch 40/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0122 Acc: 0.6976
val Loss: 0.0182 Acc: 0.4944

Epoch 41/86
----------
train Loss: 0.0115 Acc: 0.6902
val Loss: 0.0184 Acc: 0.5019

Epoch 42/86
----------
train Loss: 0.0115 Acc: 0.7014
val Loss: 0.0183 Acc: 0.4981

Epoch 43/86
----------
train Loss: 0.0111 Acc: 0.7200
val Loss: 0.0185 Acc: 0.4907

Epoch 44/86
----------
train Loss: 0.0116 Acc: 0.6840
val Loss: 0.0182 Acc: 0.4981

Epoch 45/86
----------
train Loss: 0.0114 Acc: 0.6890
val Loss: 0.0183 Acc: 0.5093

Epoch 46/86
----------
train Loss: 0.0119 Acc: 0.7125
val Loss: 0.0182 Acc: 0.4944

Epoch 47/86
----------
train Loss: 0.0113 Acc: 0.7100
val Loss: 0.0184 Acc: 0.4944

Epoch 48/86
----------
train Loss: 0.0121 Acc: 0.6840
val Loss: 0.0184 Acc: 0.4981

Epoch 49/86
----------
train Loss: 0.0115 Acc: 0.7001
val Loss: 0.0185 Acc: 0.5130

Epoch 50/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0114 Acc: 0.6840
val Loss: 0.0186 Acc: 0.5019

Epoch 51/86
----------
train Loss: 0.0119 Acc: 0.6952
val Loss: 0.0186 Acc: 0.5019

Epoch 52/86
----------
train Loss: 0.0116 Acc: 0.7026
val Loss: 0.0185 Acc: 0.5019

Epoch 53/86
----------
train Loss: 0.0122 Acc: 0.6914
val Loss: 0.0183 Acc: 0.5093

Epoch 54/86
----------
train Loss: 0.0111 Acc: 0.7026
val Loss: 0.0184 Acc: 0.5093

Epoch 55/86
----------
train Loss: 0.0114 Acc: 0.6902
val Loss: 0.0186 Acc: 0.5093

Epoch 56/86
----------
train Loss: 0.0117 Acc: 0.7076
val Loss: 0.0182 Acc: 0.5093

Epoch 57/86
----------
train Loss: 0.0125 Acc: 0.6927
val Loss: 0.0182 Acc: 0.5093

Epoch 58/86
----------
train Loss: 0.0124 Acc: 0.6741
val Loss: 0.0183 Acc: 0.4981

Epoch 59/86
----------
train Loss: 0.0116 Acc: 0.6989
val Loss: 0.0184 Acc: 0.4944

Epoch 60/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0116 Acc: 0.6989
val Loss: 0.0185 Acc: 0.4907

Epoch 61/86
----------
train Loss: 0.0118 Acc: 0.7038
val Loss: 0.0185 Acc: 0.4870

Epoch 62/86
----------
train Loss: 0.0114 Acc: 0.6989
val Loss: 0.0183 Acc: 0.4907

Epoch 63/86
----------
train Loss: 0.0122 Acc: 0.7162
val Loss: 0.0189 Acc: 0.5056

Epoch 64/86
----------
train Loss: 0.0120 Acc: 0.7100
val Loss: 0.0185 Acc: 0.5019

Epoch 65/86
----------
train Loss: 0.0121 Acc: 0.6766
val Loss: 0.0182 Acc: 0.5056

Epoch 66/86
----------
train Loss: 0.0114 Acc: 0.6815
val Loss: 0.0183 Acc: 0.5019

Epoch 67/86
----------
train Loss: 0.0120 Acc: 0.7162
val Loss: 0.0183 Acc: 0.4944

Epoch 68/86
----------
train Loss: 0.0119 Acc: 0.6877
val Loss: 0.0185 Acc: 0.4981

Epoch 69/86
----------
train Loss: 0.0113 Acc: 0.6976
val Loss: 0.0185 Acc: 0.4981

Epoch 70/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0128 Acc: 0.6828
val Loss: 0.0188 Acc: 0.5019

Epoch 71/86
----------
train Loss: 0.0121 Acc: 0.6902
val Loss: 0.0185 Acc: 0.4870

Epoch 72/86
----------
train Loss: 0.0115 Acc: 0.7001
val Loss: 0.0186 Acc: 0.4944

Epoch 73/86
----------
train Loss: 0.0111 Acc: 0.6877
val Loss: 0.0185 Acc: 0.4944

Epoch 74/86
----------
train Loss: 0.0115 Acc: 0.7026
val Loss: 0.0186 Acc: 0.4758

Epoch 75/86
----------
train Loss: 0.0127 Acc: 0.7150
val Loss: 0.0186 Acc: 0.4944

Epoch 76/86
----------
train Loss: 0.0112 Acc: 0.7076
val Loss: 0.0184 Acc: 0.4907

Epoch 77/86
----------
train Loss: 0.0118 Acc: 0.7001
val Loss: 0.0185 Acc: 0.4907

Epoch 78/86
----------
train Loss: 0.0124 Acc: 0.6877
val Loss: 0.0186 Acc: 0.4870

Epoch 79/86
----------
train Loss: 0.0120 Acc: 0.7088
val Loss: 0.0183 Acc: 0.4981

Epoch 80/86
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0112 Acc: 0.6914
val Loss: 0.0185 Acc: 0.4907

Epoch 81/86
----------
train Loss: 0.0111 Acc: 0.6766
val Loss: 0.0181 Acc: 0.5019

Epoch 82/86
----------
train Loss: 0.0109 Acc: 0.6964
val Loss: 0.0185 Acc: 0.5130

Epoch 83/86
----------
train Loss: 0.0119 Acc: 0.7026
val Loss: 0.0185 Acc: 0.4907

Epoch 84/86
----------
train Loss: 0.0114 Acc: 0.6828
val Loss: 0.0184 Acc: 0.4833

Epoch 85/86
----------
train Loss: 0.0117 Acc: 0.7038
val Loss: 0.0181 Acc: 0.4981

Epoch 86/86
----------
train Loss: 0.0114 Acc: 0.6927
val Loss: 0.0186 Acc: 0.5056

Training complete in 7m 57s
Best val Acc: 0.513011

---Fine tuning.---
Epoch 0/86
----------
LR is set to 0.01
train Loss: 0.0139 Acc: 0.6431
val Loss: 0.0209 Acc: 0.4610

Epoch 1/86
----------
train Loss: 0.0113 Acc: 0.6803
val Loss: 0.0252 Acc: 0.3866

Epoch 2/86
----------
train Loss: 0.0091 Acc: 0.7237
val Loss: 0.0222 Acc: 0.4201

Epoch 3/86
----------
train Loss: 0.0087 Acc: 0.8055
val Loss: 0.0247 Acc: 0.4424

Epoch 4/86
----------
train Loss: 0.0084 Acc: 0.7782
val Loss: 0.0324 Acc: 0.3643

Epoch 5/86
----------
train Loss: 0.0096 Acc: 0.7472
val Loss: 0.0270 Acc: 0.4089

Epoch 6/86
----------
train Loss: 0.0095 Acc: 0.7509
val Loss: 0.0297 Acc: 0.3941

Epoch 7/86
----------
train Loss: 0.0071 Acc: 0.7881
val Loss: 0.0257 Acc: 0.4424

Epoch 8/86
----------
train Loss: 0.0077 Acc: 0.8017
val Loss: 0.0275 Acc: 0.4126

Epoch 9/86
----------
train Loss: 0.0083 Acc: 0.7869
val Loss: 0.0322 Acc: 0.3643

Epoch 10/86
----------
LR is set to 0.001
train Loss: 0.0071 Acc: 0.8240
val Loss: 0.0260 Acc: 0.4758

Epoch 11/86
----------
train Loss: 0.0042 Acc: 0.8996
val Loss: 0.0218 Acc: 0.5056

Epoch 12/86
----------
train Loss: 0.0029 Acc: 0.9368
val Loss: 0.0223 Acc: 0.5204

Epoch 13/86
----------
train Loss: 0.0026 Acc: 0.9331
val Loss: 0.0221 Acc: 0.5093

Epoch 14/86
----------
train Loss: 0.0022 Acc: 0.9566
val Loss: 0.0219 Acc: 0.5204

Epoch 15/86
----------
train Loss: 0.0022 Acc: 0.9442
val Loss: 0.0231 Acc: 0.5242

Epoch 16/86
----------
train Loss: 0.0019 Acc: 0.9566
val Loss: 0.0229 Acc: 0.5167

Epoch 17/86
----------
train Loss: 0.0019 Acc: 0.9665
val Loss: 0.0221 Acc: 0.5242

Epoch 18/86
----------
train Loss: 0.0019 Acc: 0.9554
val Loss: 0.0227 Acc: 0.5130

Epoch 19/86
----------
train Loss: 0.0018 Acc: 0.9665
val Loss: 0.0223 Acc: 0.5167

Epoch 20/86
----------
LR is set to 0.00010000000000000002
train Loss: 0.0013 Acc: 0.9690
val Loss: 0.0225 Acc: 0.5242

Epoch 21/86
----------
train Loss: 0.0019 Acc: 0.9703
val Loss: 0.0229 Acc: 0.5204

Epoch 22/86
----------
train Loss: 0.0022 Acc: 0.9665
val Loss: 0.0226 Acc: 0.5279

Epoch 23/86
----------
train Loss: 0.0015 Acc: 0.9715
val Loss: 0.0232 Acc: 0.5130

Epoch 24/86
----------
train Loss: 0.0015 Acc: 0.9591
val Loss: 0.0231 Acc: 0.5130

Epoch 25/86
----------
train Loss: 0.0025 Acc: 0.9678
val Loss: 0.0231 Acc: 0.5130

Epoch 26/86
----------
train Loss: 0.0019 Acc: 0.9740
val Loss: 0.0230 Acc: 0.5167

Epoch 27/86
----------
train Loss: 0.0019 Acc: 0.9740
val Loss: 0.0227 Acc: 0.5204

Epoch 28/86
----------
train Loss: 0.0017 Acc: 0.9703
val Loss: 0.0226 Acc: 0.5242

Epoch 29/86
----------
train Loss: 0.0015 Acc: 0.9752
val Loss: 0.0227 Acc: 0.5242

Epoch 30/86
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0013 Acc: 0.9789
val Loss: 0.0228 Acc: 0.5279

Epoch 31/86
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0231 Acc: 0.5242

Epoch 32/86
----------
train Loss: 0.0015 Acc: 0.9641
val Loss: 0.0226 Acc: 0.5204

Epoch 33/86
----------
train Loss: 0.0017 Acc: 0.9678
val Loss: 0.0220 Acc: 0.5130

Epoch 34/86
----------
train Loss: 0.0016 Acc: 0.9715
val Loss: 0.0219 Acc: 0.5167

Epoch 35/86
----------
train Loss: 0.0016 Acc: 0.9752
val Loss: 0.0224 Acc: 0.5204

Epoch 36/86
----------
train Loss: 0.0017 Acc: 0.9752
val Loss: 0.0224 Acc: 0.5204

Epoch 37/86
----------
train Loss: 0.0019 Acc: 0.9628
val Loss: 0.0228 Acc: 0.5204

Epoch 38/86
----------
train Loss: 0.0017 Acc: 0.9616
val Loss: 0.0227 Acc: 0.5242

Epoch 39/86
----------
train Loss: 0.0013 Acc: 0.9703
val Loss: 0.0227 Acc: 0.5167

Epoch 40/86
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9727
val Loss: 0.0229 Acc: 0.5167

Epoch 41/86
----------
train Loss: 0.0013 Acc: 0.9678
val Loss: 0.0216 Acc: 0.5167

Epoch 42/86
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0228 Acc: 0.5204

Epoch 43/86
----------
train Loss: 0.0019 Acc: 0.9715
val Loss: 0.0233 Acc: 0.5204

Epoch 44/86
----------
train Loss: 0.0023 Acc: 0.9703
val Loss: 0.0229 Acc: 0.5167

Epoch 45/86
----------
train Loss: 0.0021 Acc: 0.9740
val Loss: 0.0229 Acc: 0.5167

Epoch 46/86
----------
train Loss: 0.0014 Acc: 0.9789
val Loss: 0.0228 Acc: 0.5204

Epoch 47/86
----------
train Loss: 0.0016 Acc: 0.9703
val Loss: 0.0223 Acc: 0.5130

Epoch 48/86
----------
train Loss: 0.0018 Acc: 0.9653
val Loss: 0.0231 Acc: 0.5204

Epoch 49/86
----------
train Loss: 0.0014 Acc: 0.9765
val Loss: 0.0226 Acc: 0.5130

Epoch 50/86
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0012 Acc: 0.9827
val Loss: 0.0225 Acc: 0.5204

Epoch 51/86
----------
train Loss: 0.0015 Acc: 0.9703
val Loss: 0.0228 Acc: 0.5204

Epoch 52/86
----------
train Loss: 0.0016 Acc: 0.9789
val Loss: 0.0225 Acc: 0.5204

Epoch 53/86
----------
train Loss: 0.0012 Acc: 0.9727
val Loss: 0.0226 Acc: 0.5279

Epoch 54/86
----------
train Loss: 0.0012 Acc: 0.9690
val Loss: 0.0222 Acc: 0.5167

Epoch 55/86
----------
train Loss: 0.0017 Acc: 0.9703
val Loss: 0.0217 Acc: 0.5093

Epoch 56/86
----------
train Loss: 0.0018 Acc: 0.9715
val Loss: 0.0231 Acc: 0.5130

Epoch 57/86
----------
train Loss: 0.0017 Acc: 0.9740
val Loss: 0.0220 Acc: 0.5204

Epoch 58/86
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0230 Acc: 0.5279

Epoch 59/86
----------
train Loss: 0.0020 Acc: 0.9641
val Loss: 0.0226 Acc: 0.5204

Epoch 60/86
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0019 Acc: 0.9715
val Loss: 0.0222 Acc: 0.5167

Epoch 61/86
----------
train Loss: 0.0017 Acc: 0.9752
val Loss: 0.0224 Acc: 0.5130

Epoch 62/86
----------
train Loss: 0.0018 Acc: 0.9665
val Loss: 0.0226 Acc: 0.5130

Epoch 63/86
----------
train Loss: 0.0016 Acc: 0.9703
val Loss: 0.0228 Acc: 0.5167

Epoch 64/86
----------
train Loss: 0.0016 Acc: 0.9765
val Loss: 0.0224 Acc: 0.5167

Epoch 65/86
----------
train Loss: 0.0016 Acc: 0.9690
val Loss: 0.0225 Acc: 0.5242

Epoch 66/86
----------
train Loss: 0.0015 Acc: 0.9740
val Loss: 0.0227 Acc: 0.5167

Epoch 67/86
----------
train Loss: 0.0017 Acc: 0.9789
val Loss: 0.0229 Acc: 0.5167

Epoch 68/86
----------
train Loss: 0.0015 Acc: 0.9603
val Loss: 0.0221 Acc: 0.5204

Epoch 69/86
----------
train Loss: 0.0014 Acc: 0.9703
val Loss: 0.0227 Acc: 0.5242

Epoch 70/86
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0025 Acc: 0.9740
val Loss: 0.0228 Acc: 0.5130

Epoch 71/86
----------
train Loss: 0.0012 Acc: 0.9752
val Loss: 0.0226 Acc: 0.5130

Epoch 72/86
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0224 Acc: 0.5130

Epoch 73/86
----------
train Loss: 0.0013 Acc: 0.9715
val Loss: 0.0225 Acc: 0.5204

Epoch 74/86
----------
train Loss: 0.0011 Acc: 0.9740
val Loss: 0.0223 Acc: 0.5167

Epoch 75/86
----------
train Loss: 0.0014 Acc: 0.9765
val Loss: 0.0229 Acc: 0.5130

Epoch 76/86
----------
train Loss: 0.0027 Acc: 0.9690
val Loss: 0.0231 Acc: 0.5242

Epoch 77/86
----------
train Loss: 0.0024 Acc: 0.9653
val Loss: 0.0233 Acc: 0.5242

Epoch 78/86
----------
train Loss: 0.0013 Acc: 0.9715
val Loss: 0.0224 Acc: 0.5167

Epoch 79/86
----------
train Loss: 0.0026 Acc: 0.9665
val Loss: 0.0223 Acc: 0.5167

Epoch 80/86
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0013 Acc: 0.9703
val Loss: 0.0224 Acc: 0.5093

Epoch 81/86
----------
train Loss: 0.0014 Acc: 0.9690
val Loss: 0.0225 Acc: 0.5167

Epoch 82/86
----------
train Loss: 0.0015 Acc: 0.9715
val Loss: 0.0225 Acc: 0.5204

Epoch 83/86
----------
train Loss: 0.0027 Acc: 0.9703
val Loss: 0.0214 Acc: 0.5130

Epoch 84/86
----------
train Loss: 0.0014 Acc: 0.9703
val Loss: 0.0226 Acc: 0.5167

Epoch 85/86
----------
train Loss: 0.0017 Acc: 0.9727
val Loss: 0.0223 Acc: 0.5167

Epoch 86/86
----------
train Loss: 0.0013 Acc: 0.9752
val Loss: 0.0219 Acc: 0.5204

Training complete in 8m 27s
Best val Acc: 0.527881

---Testing---
Test accuracy: 0.862454
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 67 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 91 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 68 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 92 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 50 %
Accuracy of Southern bluefin tuna : 75 %
Accuracy of Yellowfin tuna : 93 %
mean: 0.8194773133299968, std: 0.12660821205197867
--------------------

run info[val: 0.3, epoch: 53, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0269 Acc: 0.1658
val Loss: 0.0306 Acc: 0.2609

Epoch 1/52
----------
train Loss: 0.0227 Acc: 0.3196
val Loss: 0.0262 Acc: 0.3043

Epoch 2/52
----------
train Loss: 0.0195 Acc: 0.4204
val Loss: 0.0242 Acc: 0.3602

Epoch 3/52
----------
LR is set to 0.001
train Loss: 0.0180 Acc: 0.4496
val Loss: 0.0247 Acc: 0.3820

Epoch 4/52
----------
train Loss: 0.0172 Acc: 0.4907
val Loss: 0.0246 Acc: 0.3696

Epoch 5/52
----------
train Loss: 0.0171 Acc: 0.4947
val Loss: 0.0238 Acc: 0.3882

Epoch 6/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0168 Acc: 0.5106
val Loss: 0.0232 Acc: 0.3913

Epoch 7/52
----------
train Loss: 0.0168 Acc: 0.5119
val Loss: 0.0240 Acc: 0.3913

Epoch 8/52
----------
train Loss: 0.0169 Acc: 0.5080
val Loss: 0.0236 Acc: 0.3975

Epoch 9/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0167 Acc: 0.5212
val Loss: 0.0237 Acc: 0.4006

Epoch 10/52
----------
train Loss: 0.0173 Acc: 0.5027
val Loss: 0.0236 Acc: 0.3913

Epoch 11/52
----------
train Loss: 0.0168 Acc: 0.5013
val Loss: 0.0236 Acc: 0.3944

Epoch 12/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0168 Acc: 0.5066
val Loss: 0.0238 Acc: 0.3882

Epoch 13/52
----------
train Loss: 0.0167 Acc: 0.5252
val Loss: 0.0244 Acc: 0.3944

Epoch 14/52
----------
train Loss: 0.0167 Acc: 0.4987
val Loss: 0.0225 Acc: 0.4006

Epoch 15/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0168 Acc: 0.5013
val Loss: 0.0232 Acc: 0.3913

Epoch 16/52
----------
train Loss: 0.0167 Acc: 0.5066
val Loss: 0.0246 Acc: 0.3882

Epoch 17/52
----------
train Loss: 0.0166 Acc: 0.5292
val Loss: 0.0228 Acc: 0.3944

Epoch 18/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0169 Acc: 0.5040
val Loss: 0.0228 Acc: 0.3913

Epoch 19/52
----------
train Loss: 0.0167 Acc: 0.5332
val Loss: 0.0229 Acc: 0.4006

Epoch 20/52
----------
train Loss: 0.0169 Acc: 0.5146
val Loss: 0.0234 Acc: 0.3882

Epoch 21/52
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0167 Acc: 0.5279
val Loss: 0.0226 Acc: 0.3975

Epoch 22/52
----------
train Loss: 0.0169 Acc: 0.5119
val Loss: 0.0242 Acc: 0.3944

Epoch 23/52
----------
train Loss: 0.0166 Acc: 0.5146
val Loss: 0.0241 Acc: 0.4006

Epoch 24/52
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0168 Acc: 0.5186
val Loss: 0.0230 Acc: 0.3975

Epoch 25/52
----------
train Loss: 0.0167 Acc: 0.5146
val Loss: 0.0228 Acc: 0.3944

Epoch 26/52
----------
train Loss: 0.0168 Acc: 0.5265
val Loss: 0.0238 Acc: 0.4006

Epoch 27/52
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0169 Acc: 0.5212
val Loss: 0.0234 Acc: 0.3944

Epoch 28/52
----------
train Loss: 0.0169 Acc: 0.5053
val Loss: 0.0238 Acc: 0.3975

Epoch 29/52
----------
train Loss: 0.0169 Acc: 0.5225
val Loss: 0.0235 Acc: 0.3944

Epoch 30/52
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0170 Acc: 0.5040
val Loss: 0.0224 Acc: 0.4037

Epoch 31/52
----------
train Loss: 0.0170 Acc: 0.5225
val Loss: 0.0232 Acc: 0.4037

Epoch 32/52
----------
train Loss: 0.0167 Acc: 0.5199
val Loss: 0.0243 Acc: 0.4037

Epoch 33/52
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0166 Acc: 0.5358
val Loss: 0.0239 Acc: 0.3913

Epoch 34/52
----------
train Loss: 0.0167 Acc: 0.5199
val Loss: 0.0243 Acc: 0.3975

Epoch 35/52
----------
train Loss: 0.0167 Acc: 0.5292
val Loss: 0.0228 Acc: 0.3944

Epoch 36/52
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0166 Acc: 0.5424
val Loss: 0.0231 Acc: 0.3913

Epoch 37/52
----------
train Loss: 0.0166 Acc: 0.5252
val Loss: 0.0237 Acc: 0.4006

Epoch 38/52
----------
train Loss: 0.0168 Acc: 0.5133
val Loss: 0.0237 Acc: 0.3913

Epoch 39/52
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0169 Acc: 0.5146
val Loss: 0.0227 Acc: 0.3944

Epoch 40/52
----------
train Loss: 0.0165 Acc: 0.5212
val Loss: 0.0232 Acc: 0.3975

Epoch 41/52
----------
train Loss: 0.0165 Acc: 0.5133
val Loss: 0.0233 Acc: 0.4037

Epoch 42/52
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0169 Acc: 0.5000
val Loss: 0.0229 Acc: 0.3975

Epoch 43/52
----------
train Loss: 0.0166 Acc: 0.5119
val Loss: 0.0228 Acc: 0.3944

Epoch 44/52
----------
train Loss: 0.0168 Acc: 0.5172
val Loss: 0.0242 Acc: 0.3882

Epoch 45/52
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0168 Acc: 0.5199
val Loss: 0.0234 Acc: 0.3944

Epoch 46/52
----------
train Loss: 0.0170 Acc: 0.5106
val Loss: 0.0246 Acc: 0.4006

Epoch 47/52
----------
train Loss: 0.0168 Acc: 0.5159
val Loss: 0.0230 Acc: 0.3913

Epoch 48/52
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0169 Acc: 0.5292
val Loss: 0.0246 Acc: 0.3975

Epoch 49/52
----------
train Loss: 0.0169 Acc: 0.5239
val Loss: 0.0236 Acc: 0.3913

Epoch 50/52
----------
train Loss: 0.0166 Acc: 0.5265
val Loss: 0.0232 Acc: 0.3975

Epoch 51/52
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0167 Acc: 0.5252
val Loss: 0.0237 Acc: 0.3944

Epoch 52/52
----------
train Loss: 0.0166 Acc: 0.5305
val Loss: 0.0235 Acc: 0.3975

Training complete in 4m 52s
Best val Acc: 0.403727

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0170 Acc: 0.5040
val Loss: 0.0242 Acc: 0.4472

Epoch 1/52
----------
train Loss: 0.0131 Acc: 0.6141
val Loss: 0.0220 Acc: 0.4255

Epoch 2/52
----------
train Loss: 0.0096 Acc: 0.7135
val Loss: 0.0191 Acc: 0.4783

Epoch 3/52
----------
LR is set to 0.001
train Loss: 0.0069 Acc: 0.8170
val Loss: 0.0203 Acc: 0.5155

Epoch 4/52
----------
train Loss: 0.0059 Acc: 0.8581
val Loss: 0.0178 Acc: 0.5217

Epoch 5/52
----------
train Loss: 0.0059 Acc: 0.8727
val Loss: 0.0178 Acc: 0.5373

Epoch 6/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0054 Acc: 0.8899
val Loss: 0.0172 Acc: 0.5404

Epoch 7/52
----------
train Loss: 0.0055 Acc: 0.8899
val Loss: 0.0169 Acc: 0.5404

Epoch 8/52
----------
train Loss: 0.0052 Acc: 0.8966
val Loss: 0.0193 Acc: 0.5342

Epoch 9/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0053 Acc: 0.8846
val Loss: 0.0166 Acc: 0.5342

Epoch 10/52
----------
train Loss: 0.0053 Acc: 0.8992
val Loss: 0.0171 Acc: 0.5404

Epoch 11/52
----------
train Loss: 0.0054 Acc: 0.8899
val Loss: 0.0185 Acc: 0.5404

Epoch 12/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0053 Acc: 0.8939
val Loss: 0.0178 Acc: 0.5373

Epoch 13/52
----------
train Loss: 0.0052 Acc: 0.8859
val Loss: 0.0178 Acc: 0.5311

Epoch 14/52
----------
train Loss: 0.0051 Acc: 0.8926
val Loss: 0.0171 Acc: 0.5342

Epoch 15/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0054 Acc: 0.8780
val Loss: 0.0178 Acc: 0.5373

Epoch 16/52
----------
train Loss: 0.0050 Acc: 0.9005
val Loss: 0.0188 Acc: 0.5311

Epoch 17/52
----------
train Loss: 0.0052 Acc: 0.8939
val Loss: 0.0185 Acc: 0.5280

Epoch 18/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0053 Acc: 0.8926
val Loss: 0.0179 Acc: 0.5373

Epoch 19/52
----------
train Loss: 0.0056 Acc: 0.8873
val Loss: 0.0178 Acc: 0.5404

Epoch 20/52
----------
train Loss: 0.0051 Acc: 0.8926
val Loss: 0.0181 Acc: 0.5404

Epoch 21/52
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0053 Acc: 0.8939
val Loss: 0.0186 Acc: 0.5280

Epoch 22/52
----------
train Loss: 0.0054 Acc: 0.8727
val Loss: 0.0190 Acc: 0.5280

Epoch 23/52
----------
train Loss: 0.0051 Acc: 0.8966
val Loss: 0.0173 Acc: 0.5311

Epoch 24/52
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0054 Acc: 0.8740
val Loss: 0.0175 Acc: 0.5311

Epoch 25/52
----------
train Loss: 0.0051 Acc: 0.8859
val Loss: 0.0171 Acc: 0.5311

Epoch 26/52
----------
train Loss: 0.0052 Acc: 0.8846
val Loss: 0.0170 Acc: 0.5311

Epoch 27/52
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0050 Acc: 0.9032
val Loss: 0.0182 Acc: 0.5280

Epoch 28/52
----------
train Loss: 0.0051 Acc: 0.8966
val Loss: 0.0165 Acc: 0.5342

Epoch 29/52
----------
train Loss: 0.0052 Acc: 0.8992
val Loss: 0.0173 Acc: 0.5373

Epoch 30/52
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0054 Acc: 0.8859
val Loss: 0.0182 Acc: 0.5435

Epoch 31/52
----------
train Loss: 0.0054 Acc: 0.8780
val Loss: 0.0180 Acc: 0.5342

Epoch 32/52
----------
train Loss: 0.0053 Acc: 0.8873
val Loss: 0.0178 Acc: 0.5311

Epoch 33/52
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0053 Acc: 0.8767
val Loss: 0.0175 Acc: 0.5342

Epoch 34/52
----------
train Loss: 0.0054 Acc: 0.8846
val Loss: 0.0182 Acc: 0.5373

Epoch 35/52
----------
train Loss: 0.0053 Acc: 0.8714
val Loss: 0.0170 Acc: 0.5373

Epoch 36/52
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0052 Acc: 0.8740
val Loss: 0.0183 Acc: 0.5373

Epoch 37/52
----------
train Loss: 0.0054 Acc: 0.8859
val Loss: 0.0185 Acc: 0.5404

Epoch 38/52
----------
train Loss: 0.0051 Acc: 0.8912
val Loss: 0.0173 Acc: 0.5373

Epoch 39/52
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0054 Acc: 0.8846
val Loss: 0.0174 Acc: 0.5342

Epoch 40/52
----------
train Loss: 0.0052 Acc: 0.8899
val Loss: 0.0176 Acc: 0.5373

Epoch 41/52
----------
train Loss: 0.0052 Acc: 0.9058
val Loss: 0.0162 Acc: 0.5280

Epoch 42/52
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0052 Acc: 0.8886
val Loss: 0.0183 Acc: 0.5373

Epoch 43/52
----------
train Loss: 0.0053 Acc: 0.8793
val Loss: 0.0179 Acc: 0.5404

Epoch 44/52
----------
train Loss: 0.0051 Acc: 0.8952
val Loss: 0.0177 Acc: 0.5373

Epoch 45/52
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0054 Acc: 0.8873
val Loss: 0.0174 Acc: 0.5342

Epoch 46/52
----------
train Loss: 0.0052 Acc: 0.9032
val Loss: 0.0184 Acc: 0.5373

Epoch 47/52
----------
train Loss: 0.0053 Acc: 0.8886
val Loss: 0.0181 Acc: 0.5373

Epoch 48/52
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0050 Acc: 0.8992
val Loss: 0.0192 Acc: 0.5342

Epoch 49/52
----------
train Loss: 0.0053 Acc: 0.8966
val Loss: 0.0188 Acc: 0.5404

Epoch 50/52
----------
train Loss: 0.0053 Acc: 0.8833
val Loss: 0.0175 Acc: 0.5404

Epoch 51/52
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0054 Acc: 0.8926
val Loss: 0.0177 Acc: 0.5373

Epoch 52/52
----------
train Loss: 0.0052 Acc: 0.9019
val Loss: 0.0180 Acc: 0.5342

Training complete in 5m 10s
Best val Acc: 0.543478

---Testing---
Test accuracy: 0.798327
--------------------
Accuracy of Albacore tuna : 81 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 61 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 84 %
Accuracy of Frigate tuna : 62 %
Accuracy of Little tunny : 82 %
Accuracy of Longtail tuna : 87 %
Accuracy of Mackerel tuna : 68 %
Accuracy of Pacific bluefin tuna : 50 %
Accuracy of Skipjack tuna : 89 %
Accuracy of Slender tuna :  7 %
Accuracy of Southern bluefin tuna : 57 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.7152373757383409, std: 0.22505615951036262

Model saved in "./weights/tuna_fish_[0.94]_mean[0.92]_std[0.07].save".
--------------------

run info[val: 0.1, epoch: 51, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/50
----------
LR is set to 0.01
train Loss: 0.0268 Acc: 0.1610
val Loss: 0.0368 Acc: 0.3271

Epoch 1/50
----------
train Loss: 0.0216 Acc: 0.3323
val Loss: 0.0343 Acc: 0.4673

Epoch 2/50
----------
train Loss: 0.0185 Acc: 0.4180
val Loss: 0.0285 Acc: 0.4860

Epoch 3/50
----------
train Loss: 0.0169 Acc: 0.4706
val Loss: 0.0263 Acc: 0.4673

Epoch 4/50
----------
train Loss: 0.0156 Acc: 0.5335
val Loss: 0.0275 Acc: 0.4953

Epoch 5/50
----------
train Loss: 0.0144 Acc: 0.5738
val Loss: 0.0339 Acc: 0.4860

Epoch 6/50
----------
train Loss: 0.0135 Acc: 0.5996
val Loss: 0.0295 Acc: 0.5234

Epoch 7/50
----------
train Loss: 0.0130 Acc: 0.6182
val Loss: 0.0311 Acc: 0.4673

Epoch 8/50
----------
LR is set to 0.001
train Loss: 0.0120 Acc: 0.6429
val Loss: 0.0298 Acc: 0.4953

Epoch 9/50
----------
train Loss: 0.0115 Acc: 0.6698
val Loss: 0.0299 Acc: 0.4953

Epoch 10/50
----------
train Loss: 0.0120 Acc: 0.6553
val Loss: 0.0249 Acc: 0.4953

Epoch 11/50
----------
train Loss: 0.0117 Acc: 0.6615
val Loss: 0.0307 Acc: 0.4953

Epoch 12/50
----------
train Loss: 0.0115 Acc: 0.6718
val Loss: 0.0321 Acc: 0.4953

Epoch 13/50
----------
train Loss: 0.0115 Acc: 0.6667
val Loss: 0.0254 Acc: 0.4953

Epoch 14/50
----------
train Loss: 0.0114 Acc: 0.6563
val Loss: 0.0286 Acc: 0.4953

Epoch 15/50
----------
train Loss: 0.0114 Acc: 0.6801
val Loss: 0.0266 Acc: 0.5140

Epoch 16/50
----------
LR is set to 0.00010000000000000002
train Loss: 0.0114 Acc: 0.6687
val Loss: 0.0299 Acc: 0.5047

Epoch 17/50
----------
train Loss: 0.0114 Acc: 0.6615
val Loss: 0.0277 Acc: 0.5047

Epoch 18/50
----------
train Loss: 0.0115 Acc: 0.6646
val Loss: 0.0271 Acc: 0.4860

Epoch 19/50
----------
train Loss: 0.0113 Acc: 0.6749
val Loss: 0.0323 Acc: 0.4860

Epoch 20/50
----------
train Loss: 0.0115 Acc: 0.6729
val Loss: 0.0299 Acc: 0.5047

Epoch 21/50
----------
train Loss: 0.0113 Acc: 0.6698
val Loss: 0.0220 Acc: 0.4953

Epoch 22/50
----------
train Loss: 0.0113 Acc: 0.6821
val Loss: 0.0348 Acc: 0.4860

Epoch 23/50
----------
train Loss: 0.0117 Acc: 0.6594
val Loss: 0.0246 Acc: 0.4953

Epoch 24/50
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0114 Acc: 0.6605
val Loss: 0.0395 Acc: 0.4953

Epoch 25/50
----------
train Loss: 0.0112 Acc: 0.6832
val Loss: 0.0289 Acc: 0.4860

Epoch 26/50
----------
train Loss: 0.0115 Acc: 0.6615
val Loss: 0.0298 Acc: 0.4953

Epoch 27/50
----------
train Loss: 0.0117 Acc: 0.6708
val Loss: 0.0316 Acc: 0.4953

Epoch 28/50
----------
train Loss: 0.0113 Acc: 0.6646
val Loss: 0.0268 Acc: 0.4953

Epoch 29/50
----------
train Loss: 0.0113 Acc: 0.6811
val Loss: 0.0327 Acc: 0.4953

Epoch 30/50
----------
train Loss: 0.0114 Acc: 0.6687
val Loss: 0.0211 Acc: 0.4953

Epoch 31/50
----------
train Loss: 0.0112 Acc: 0.6832
val Loss: 0.0297 Acc: 0.4953

Epoch 32/50
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0115 Acc: 0.6718
val Loss: 0.0300 Acc: 0.4953

Epoch 33/50
----------
train Loss: 0.0113 Acc: 0.6729
val Loss: 0.0286 Acc: 0.4953

Epoch 34/50
----------
train Loss: 0.0112 Acc: 0.6873
val Loss: 0.0312 Acc: 0.4953

Epoch 35/50
----------
train Loss: 0.0113 Acc: 0.6749
val Loss: 0.0352 Acc: 0.4953

Epoch 36/50
----------
train Loss: 0.0114 Acc: 0.6842
val Loss: 0.0245 Acc: 0.4953

Epoch 37/50
----------
train Loss: 0.0115 Acc: 0.6749
val Loss: 0.0342 Acc: 0.4953

Epoch 38/50
----------
train Loss: 0.0116 Acc: 0.6698
val Loss: 0.0299 Acc: 0.4953

Epoch 39/50
----------
train Loss: 0.0115 Acc: 0.6801
val Loss: 0.0233 Acc: 0.4953

Epoch 40/50
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0114 Acc: 0.6543
val Loss: 0.0278 Acc: 0.4953

Epoch 41/50
----------
train Loss: 0.0115 Acc: 0.6708
val Loss: 0.0301 Acc: 0.4953

Epoch 42/50
----------
train Loss: 0.0114 Acc: 0.6811
val Loss: 0.0288 Acc: 0.4860

Epoch 43/50
----------
train Loss: 0.0112 Acc: 0.6873
val Loss: 0.0348 Acc: 0.4860

Epoch 44/50
----------
train Loss: 0.0116 Acc: 0.6656
val Loss: 0.0308 Acc: 0.4953

Epoch 45/50
----------
train Loss: 0.0116 Acc: 0.6584
val Loss: 0.0343 Acc: 0.4953

Epoch 46/50
----------
train Loss: 0.0113 Acc: 0.6749
val Loss: 0.0274 Acc: 0.4953

Epoch 47/50
----------
train Loss: 0.0115 Acc: 0.6687
val Loss: 0.0314 Acc: 0.4860

Epoch 48/50
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0113 Acc: 0.6821
val Loss: 0.0251 Acc: 0.4953

Epoch 49/50
----------
train Loss: 0.0111 Acc: 0.6997
val Loss: 0.0347 Acc: 0.4953

Epoch 50/50
----------
train Loss: 0.0112 Acc: 0.6811
val Loss: 0.0300 Acc: 0.4953

Training complete in 4m 44s
Best val Acc: 0.523364

---Fine tuning.---
Epoch 0/50
----------
LR is set to 0.01
train Loss: 0.0127 Acc: 0.6089
val Loss: 0.0330 Acc: 0.5140

Epoch 1/50
----------
train Loss: 0.0088 Acc: 0.7245
val Loss: 0.0384 Acc: 0.5701

Epoch 2/50
----------
train Loss: 0.0057 Acc: 0.8400
val Loss: 0.0231 Acc: 0.5701

Epoch 3/50
----------
train Loss: 0.0040 Acc: 0.8906
val Loss: 0.0223 Acc: 0.5888

Epoch 4/50
----------
train Loss: 0.0031 Acc: 0.9154
val Loss: 0.0305 Acc: 0.5607

Epoch 5/50
----------
train Loss: 0.0021 Acc: 0.9536
val Loss: 0.0254 Acc: 0.5888

Epoch 6/50
----------
train Loss: 0.0016 Acc: 0.9659
val Loss: 0.0263 Acc: 0.5888

Epoch 7/50
----------
train Loss: 0.0013 Acc: 0.9628
val Loss: 0.0253 Acc: 0.5888

Epoch 8/50
----------
LR is set to 0.001
train Loss: 0.0010 Acc: 0.9763
val Loss: 0.0227 Acc: 0.5888

Epoch 9/50
----------
train Loss: 0.0010 Acc: 0.9773
val Loss: 0.0307 Acc: 0.5888

Epoch 10/50
----------
train Loss: 0.0010 Acc: 0.9804
val Loss: 0.0280 Acc: 0.5888

Epoch 11/50
----------
train Loss: 0.0010 Acc: 0.9721
val Loss: 0.0284 Acc: 0.5701

Epoch 12/50
----------
train Loss: 0.0009 Acc: 0.9752
val Loss: 0.0339 Acc: 0.5888

Epoch 13/50
----------
train Loss: 0.0009 Acc: 0.9773
val Loss: 0.0245 Acc: 0.5794

Epoch 14/50
----------
train Loss: 0.0008 Acc: 0.9752
val Loss: 0.0261 Acc: 0.5888

Epoch 15/50
----------
train Loss: 0.0007 Acc: 0.9835
val Loss: 0.0240 Acc: 0.5888

Epoch 16/50
----------
LR is set to 0.00010000000000000002
train Loss: 0.0009 Acc: 0.9721
val Loss: 0.0287 Acc: 0.5888

Epoch 17/50
----------
train Loss: 0.0007 Acc: 0.9773
val Loss: 0.0376 Acc: 0.5888

Epoch 18/50
----------
train Loss: 0.0008 Acc: 0.9783
val Loss: 0.0337 Acc: 0.5888

Epoch 19/50
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0245 Acc: 0.5888

Epoch 20/50
----------
train Loss: 0.0009 Acc: 0.9732
val Loss: 0.0241 Acc: 0.5888

Epoch 21/50
----------
train Loss: 0.0008 Acc: 0.9814
val Loss: 0.0417 Acc: 0.5888

Epoch 22/50
----------
train Loss: 0.0008 Acc: 0.9783
val Loss: 0.0233 Acc: 0.5888

Epoch 23/50
----------
train Loss: 0.0008 Acc: 0.9732
val Loss: 0.0249 Acc: 0.5888

Epoch 24/50
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0210 Acc: 0.5888

Epoch 25/50
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0317 Acc: 0.5888

Epoch 26/50
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0166 Acc: 0.5888

Epoch 27/50
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0226 Acc: 0.5888

Epoch 28/50
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0385 Acc: 0.5888

Epoch 29/50
----------
train Loss: 0.0007 Acc: 0.9783
val Loss: 0.0419 Acc: 0.5888

Epoch 30/50
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0252 Acc: 0.5888

Epoch 31/50
----------
train Loss: 0.0008 Acc: 0.9794
val Loss: 0.0227 Acc: 0.5888

Epoch 32/50
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0007 Acc: 0.9845
val Loss: 0.0256 Acc: 0.5888

Epoch 33/50
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0364 Acc: 0.5888

Epoch 34/50
----------
train Loss: 0.0007 Acc: 0.9835
val Loss: 0.0186 Acc: 0.5888

Epoch 35/50
----------
train Loss: 0.0008 Acc: 0.9763
val Loss: 0.0289 Acc: 0.5888

Epoch 36/50
----------
train Loss: 0.0008 Acc: 0.9794
val Loss: 0.0411 Acc: 0.5888

Epoch 37/50
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0232 Acc: 0.5888

Epoch 38/50
----------
train Loss: 0.0008 Acc: 0.9835
val Loss: 0.0311 Acc: 0.5888

Epoch 39/50
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0320 Acc: 0.5888

Epoch 40/50
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0007 Acc: 0.9856
val Loss: 0.0271 Acc: 0.5888

Epoch 41/50
----------
train Loss: 0.0008 Acc: 0.9804
val Loss: 0.0215 Acc: 0.5888

Epoch 42/50
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0437 Acc: 0.5888

Epoch 43/50
----------
train Loss: 0.0008 Acc: 0.9794
val Loss: 0.0395 Acc: 0.5888

Epoch 44/50
----------
train Loss: 0.0008 Acc: 0.9773
val Loss: 0.0209 Acc: 0.5888

Epoch 45/50
----------
train Loss: 0.0007 Acc: 0.9845
val Loss: 0.0194 Acc: 0.5888

Epoch 46/50
----------
train Loss: 0.0008 Acc: 0.9783
val Loss: 0.0343 Acc: 0.5888

Epoch 47/50
----------
train Loss: 0.0007 Acc: 0.9835
val Loss: 0.0384 Acc: 0.5888

Epoch 48/50
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9794
val Loss: 0.0220 Acc: 0.5888

Epoch 49/50
----------
train Loss: 0.0007 Acc: 0.9804
val Loss: 0.0321 Acc: 0.5888

Epoch 50/50
----------
train Loss: 0.0007 Acc: 0.9825
val Loss: 0.0237 Acc: 0.5888

Training complete in 5m 4s
Best val Acc: 0.588785

---Testing---
Test accuracy: 0.905204
--------------------
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 73 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 95 %
Accuracy of Frigate tuna : 58 %
Accuracy of Little tunny : 98 %
Accuracy of Longtail tuna : 96 %
Accuracy of Mackerel tuna : 76 %
Accuracy of Pacific bluefin tuna : 73 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 90 %
Accuracy of Yellowfin tuna : 97 %
mean: 0.8634050864230584, std: 0.12026983493383916
--------------------

run info[val: 0.15, epoch: 60, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0270 Acc: 0.2066
val Loss: 0.0282 Acc: 0.2609

Epoch 1/59
----------
train Loss: 0.0219 Acc: 0.3530
val Loss: 0.0258 Acc: 0.3106

Epoch 2/59
----------
train Loss: 0.0187 Acc: 0.4514
val Loss: 0.0242 Acc: 0.3602

Epoch 3/59
----------
LR is set to 0.001
train Loss: 0.0166 Acc: 0.5104
val Loss: 0.0226 Acc: 0.3851

Epoch 4/59
----------
train Loss: 0.0160 Acc: 0.5596
val Loss: 0.0233 Acc: 0.3789

Epoch 5/59
----------
train Loss: 0.0164 Acc: 0.5650
val Loss: 0.0231 Acc: 0.3727

Epoch 6/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0159 Acc: 0.5836
val Loss: 0.0236 Acc: 0.3540

Epoch 7/59
----------
train Loss: 0.0157 Acc: 0.5607
val Loss: 0.0232 Acc: 0.3789

Epoch 8/59
----------
train Loss: 0.0150 Acc: 0.5869
val Loss: 0.0230 Acc: 0.3851

Epoch 9/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0156 Acc: 0.5836
val Loss: 0.0231 Acc: 0.3789

Epoch 10/59
----------
train Loss: 0.0153 Acc: 0.5738
val Loss: 0.0231 Acc: 0.3789

Epoch 11/59
----------
train Loss: 0.0153 Acc: 0.5781
val Loss: 0.0236 Acc: 0.3913

Epoch 12/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0154 Acc: 0.5803
val Loss: 0.0232 Acc: 0.3727

Epoch 13/59
----------
train Loss: 0.0160 Acc: 0.5585
val Loss: 0.0236 Acc: 0.3789

Epoch 14/59
----------
train Loss: 0.0158 Acc: 0.5749
val Loss: 0.0234 Acc: 0.3789

Epoch 15/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0156 Acc: 0.5727
val Loss: 0.0234 Acc: 0.3851

Epoch 16/59
----------
train Loss: 0.0157 Acc: 0.5770
val Loss: 0.0233 Acc: 0.3851

Epoch 17/59
----------
train Loss: 0.0159 Acc: 0.5770
val Loss: 0.0232 Acc: 0.3727

Epoch 18/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0155 Acc: 0.5825
val Loss: 0.0234 Acc: 0.3789

Epoch 19/59
----------
train Loss: 0.0157 Acc: 0.5847
val Loss: 0.0232 Acc: 0.3727

Epoch 20/59
----------
train Loss: 0.0156 Acc: 0.5792
val Loss: 0.0239 Acc: 0.3851

Epoch 21/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0159 Acc: 0.5858
val Loss: 0.0235 Acc: 0.3913

Epoch 22/59
----------
train Loss: 0.0161 Acc: 0.5716
val Loss: 0.0235 Acc: 0.3789

Epoch 23/59
----------
train Loss: 0.0154 Acc: 0.5869
val Loss: 0.0233 Acc: 0.3851

Epoch 24/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0155 Acc: 0.5738
val Loss: 0.0237 Acc: 0.3851

Epoch 25/59
----------
train Loss: 0.0159 Acc: 0.5694
val Loss: 0.0228 Acc: 0.3727

Epoch 26/59
----------
train Loss: 0.0156 Acc: 0.5672
val Loss: 0.0234 Acc: 0.3727

Epoch 27/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0155 Acc: 0.5770
val Loss: 0.0232 Acc: 0.3789

Epoch 28/59
----------
train Loss: 0.0155 Acc: 0.5770
val Loss: 0.0234 Acc: 0.3789

Epoch 29/59
----------
train Loss: 0.0158 Acc: 0.5738
val Loss: 0.0231 Acc: 0.3851

Epoch 30/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0155 Acc: 0.5803
val Loss: 0.0228 Acc: 0.3851

Epoch 31/59
----------
train Loss: 0.0157 Acc: 0.5814
val Loss: 0.0231 Acc: 0.3913

Epoch 32/59
----------
train Loss: 0.0155 Acc: 0.5650
val Loss: 0.0232 Acc: 0.3727

Epoch 33/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0158 Acc: 0.5847
val Loss: 0.0233 Acc: 0.3851

Epoch 34/59
----------
train Loss: 0.0153 Acc: 0.5814
val Loss: 0.0229 Acc: 0.3727

Epoch 35/59
----------
train Loss: 0.0159 Acc: 0.5880
val Loss: 0.0234 Acc: 0.3727

Epoch 36/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0155 Acc: 0.5683
val Loss: 0.0238 Acc: 0.3851

Epoch 37/59
----------
train Loss: 0.0157 Acc: 0.5836
val Loss: 0.0233 Acc: 0.3851

Epoch 38/59
----------
train Loss: 0.0155 Acc: 0.5770
val Loss: 0.0232 Acc: 0.3851

Epoch 39/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0157 Acc: 0.5694
val Loss: 0.0227 Acc: 0.3913

Epoch 40/59
----------
train Loss: 0.0157 Acc: 0.5694
val Loss: 0.0233 Acc: 0.3789

Epoch 41/59
----------
train Loss: 0.0153 Acc: 0.5694
val Loss: 0.0232 Acc: 0.3975

Epoch 42/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0156 Acc: 0.5661
val Loss: 0.0233 Acc: 0.3913

Epoch 43/59
----------
train Loss: 0.0155 Acc: 0.5891
val Loss: 0.0229 Acc: 0.3727

Epoch 44/59
----------
train Loss: 0.0153 Acc: 0.5705
val Loss: 0.0234 Acc: 0.3789

Epoch 45/59
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0155 Acc: 0.5825
val Loss: 0.0234 Acc: 0.3913

Epoch 46/59
----------
train Loss: 0.0156 Acc: 0.5705
val Loss: 0.0227 Acc: 0.3789

Epoch 47/59
----------
train Loss: 0.0158 Acc: 0.5628
val Loss: 0.0230 Acc: 0.3727

Epoch 48/59
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0155 Acc: 0.5749
val Loss: 0.0231 Acc: 0.3913

Epoch 49/59
----------
train Loss: 0.0157 Acc: 0.5803
val Loss: 0.0232 Acc: 0.3789

Epoch 50/59
----------
train Loss: 0.0159 Acc: 0.5563
val Loss: 0.0230 Acc: 0.3727

Epoch 51/59
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0156 Acc: 0.5705
val Loss: 0.0233 Acc: 0.3851

Epoch 52/59
----------
train Loss: 0.0154 Acc: 0.5694
val Loss: 0.0229 Acc: 0.3727

Epoch 53/59
----------
train Loss: 0.0157 Acc: 0.5858
val Loss: 0.0228 Acc: 0.3789

Epoch 54/59
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0154 Acc: 0.5749
val Loss: 0.0233 Acc: 0.3851

Epoch 55/59
----------
train Loss: 0.0153 Acc: 0.5945
val Loss: 0.0236 Acc: 0.3789

Epoch 56/59
----------
train Loss: 0.0155 Acc: 0.5803
val Loss: 0.0235 Acc: 0.3789

Epoch 57/59
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0154 Acc: 0.5749
val Loss: 0.0233 Acc: 0.3789

Epoch 58/59
----------
train Loss: 0.0155 Acc: 0.5705
val Loss: 0.0228 Acc: 0.3851

Epoch 59/59
----------
train Loss: 0.0157 Acc: 0.5945
val Loss: 0.0232 Acc: 0.3789

Training complete in 5m 32s
Best val Acc: 0.397516

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0157 Acc: 0.5421
val Loss: 0.0223 Acc: 0.4037

Epoch 1/59
----------
train Loss: 0.0103 Acc: 0.7005
val Loss: 0.0219 Acc: 0.4658

Epoch 2/59
----------
train Loss: 0.0068 Acc: 0.8328
val Loss: 0.0199 Acc: 0.4907

Epoch 3/59
----------
LR is set to 0.001
train Loss: 0.0043 Acc: 0.9016
val Loss: 0.0186 Acc: 0.5031

Epoch 4/59
----------
train Loss: 0.0036 Acc: 0.9388
val Loss: 0.0184 Acc: 0.5155

Epoch 5/59
----------
train Loss: 0.0031 Acc: 0.9454
val Loss: 0.0181 Acc: 0.5342

Epoch 6/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0030 Acc: 0.9464
val Loss: 0.0177 Acc: 0.5404

Epoch 7/59
----------
train Loss: 0.0030 Acc: 0.9497
val Loss: 0.0184 Acc: 0.5404

Epoch 8/59
----------
train Loss: 0.0029 Acc: 0.9541
val Loss: 0.0183 Acc: 0.5342

Epoch 9/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0030 Acc: 0.9475
val Loss: 0.0189 Acc: 0.5466

Epoch 10/59
----------
train Loss: 0.0029 Acc: 0.9410
val Loss: 0.0183 Acc: 0.5404

Epoch 11/59
----------
train Loss: 0.0034 Acc: 0.9410
val Loss: 0.0187 Acc: 0.5404

Epoch 12/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0029 Acc: 0.9443
val Loss: 0.0188 Acc: 0.5342

Epoch 13/59
----------
train Loss: 0.0030 Acc: 0.9552
val Loss: 0.0185 Acc: 0.5404

Epoch 14/59
----------
train Loss: 0.0030 Acc: 0.9497
val Loss: 0.0187 Acc: 0.5342

Epoch 15/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0027 Acc: 0.9585
val Loss: 0.0188 Acc: 0.5280

Epoch 16/59
----------
train Loss: 0.0029 Acc: 0.9464
val Loss: 0.0186 Acc: 0.5342

Epoch 17/59
----------
train Loss: 0.0029 Acc: 0.9541
val Loss: 0.0185 Acc: 0.5466

Epoch 18/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0036 Acc: 0.9410
val Loss: 0.0185 Acc: 0.5466

Epoch 19/59
----------
train Loss: 0.0029 Acc: 0.9552
val Loss: 0.0182 Acc: 0.5342

Epoch 20/59
----------
train Loss: 0.0032 Acc: 0.9530
val Loss: 0.0184 Acc: 0.5466

Epoch 21/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0030 Acc: 0.9464
val Loss: 0.0189 Acc: 0.5342

Epoch 22/59
----------
train Loss: 0.0032 Acc: 0.9421
val Loss: 0.0190 Acc: 0.5342

Epoch 23/59
----------
train Loss: 0.0032 Acc: 0.9410
val Loss: 0.0187 Acc: 0.5466

Epoch 24/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0029 Acc: 0.9464
val Loss: 0.0191 Acc: 0.5342

Epoch 25/59
----------
train Loss: 0.0031 Acc: 0.9486
val Loss: 0.0190 Acc: 0.5590

Epoch 26/59
----------
train Loss: 0.0029 Acc: 0.9497
val Loss: 0.0188 Acc: 0.5528

Epoch 27/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0030 Acc: 0.9377
val Loss: 0.0185 Acc: 0.5528

Epoch 28/59
----------
train Loss: 0.0030 Acc: 0.9585
val Loss: 0.0187 Acc: 0.5342

Epoch 29/59
----------
train Loss: 0.0029 Acc: 0.9563
val Loss: 0.0188 Acc: 0.5404

Epoch 30/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0030 Acc: 0.9497
val Loss: 0.0191 Acc: 0.5404

Epoch 31/59
----------
train Loss: 0.0033 Acc: 0.9497
val Loss: 0.0185 Acc: 0.5342

Epoch 32/59
----------
train Loss: 0.0032 Acc: 0.9508
val Loss: 0.0192 Acc: 0.5280

Epoch 33/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0030 Acc: 0.9530
val Loss: 0.0192 Acc: 0.5342

Epoch 34/59
----------
train Loss: 0.0030 Acc: 0.9410
val Loss: 0.0189 Acc: 0.5342

Epoch 35/59
----------
train Loss: 0.0031 Acc: 0.9475
val Loss: 0.0185 Acc: 0.5404

Epoch 36/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0031 Acc: 0.9508
val Loss: 0.0184 Acc: 0.5342

Epoch 37/59
----------
train Loss: 0.0029 Acc: 0.9552
val Loss: 0.0189 Acc: 0.5342

Epoch 38/59
----------
train Loss: 0.0030 Acc: 0.9508
val Loss: 0.0190 Acc: 0.5342

Epoch 39/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0029 Acc: 0.9552
val Loss: 0.0191 Acc: 0.5280

Epoch 40/59
----------
train Loss: 0.0029 Acc: 0.9497
val Loss: 0.0185 Acc: 0.5590

Epoch 41/59
----------
train Loss: 0.0031 Acc: 0.9464
val Loss: 0.0188 Acc: 0.5342

Epoch 42/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0033 Acc: 0.9486
val Loss: 0.0188 Acc: 0.5404

Epoch 43/59
----------
train Loss: 0.0030 Acc: 0.9497
val Loss: 0.0184 Acc: 0.5528

Epoch 44/59
----------
train Loss: 0.0028 Acc: 0.9541
val Loss: 0.0187 Acc: 0.5466

Epoch 45/59
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0032 Acc: 0.9366
val Loss: 0.0195 Acc: 0.5404

Epoch 46/59
----------
train Loss: 0.0029 Acc: 0.9475
val Loss: 0.0185 Acc: 0.5466

Epoch 47/59
----------
train Loss: 0.0031 Acc: 0.9421
val Loss: 0.0188 Acc: 0.5404

Epoch 48/59
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0029 Acc: 0.9443
val Loss: 0.0187 Acc: 0.5404

Epoch 49/59
----------
train Loss: 0.0031 Acc: 0.9563
val Loss: 0.0192 Acc: 0.5342

Epoch 50/59
----------
train Loss: 0.0034 Acc: 0.9443
val Loss: 0.0185 Acc: 0.5342

Epoch 51/59
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0030 Acc: 0.9519
val Loss: 0.0187 Acc: 0.5342

Epoch 52/59
----------
train Loss: 0.0032 Acc: 0.9486
val Loss: 0.0195 Acc: 0.5466

Epoch 53/59
----------
train Loss: 0.0036 Acc: 0.9421
val Loss: 0.0190 Acc: 0.5280

Epoch 54/59
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0028 Acc: 0.9519
val Loss: 0.0185 Acc: 0.5404

Epoch 55/59
----------
train Loss: 0.0029 Acc: 0.9486
val Loss: 0.0193 Acc: 0.5404

Epoch 56/59
----------
train Loss: 0.0031 Acc: 0.9475
val Loss: 0.0186 Acc: 0.5466

Epoch 57/59
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0031 Acc: 0.9410
val Loss: 0.0186 Acc: 0.5404

Epoch 58/59
----------
train Loss: 0.0030 Acc: 0.9486
val Loss: 0.0190 Acc: 0.5528

Epoch 59/59
----------
train Loss: 0.0035 Acc: 0.9454
val Loss: 0.0184 Acc: 0.5590

Training complete in 5m 53s
Best val Acc: 0.559006

---Testing---
Test accuracy: 0.892193
--------------------
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 82 %
Accuracy of Bigeye tuna : 79 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 79 %
Accuracy of Little tunny : 94 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 84 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 35 %
Accuracy of Southern bluefin tuna : 77 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8410154881763361, std: 0.1518566408977773
--------------------

run info[val: 0.2, epoch: 86, randcrop: False, decay: 7]

---Training last layer.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0267 Acc: 0.1940
val Loss: 0.0343 Acc: 0.2605

Epoch 1/85
----------
train Loss: 0.0219 Acc: 0.3322
val Loss: 0.0273 Acc: 0.3116

Epoch 2/85
----------
train Loss: 0.0181 Acc: 0.4611
val Loss: 0.0267 Acc: 0.3721

Epoch 3/85
----------
train Loss: 0.0160 Acc: 0.5145
val Loss: 0.0244 Acc: 0.4186

Epoch 4/85
----------
train Loss: 0.0145 Acc: 0.5714
val Loss: 0.0257 Acc: 0.4186

Epoch 5/85
----------
train Loss: 0.0137 Acc: 0.5796
val Loss: 0.0245 Acc: 0.4419

Epoch 6/85
----------
train Loss: 0.0127 Acc: 0.6225
val Loss: 0.0265 Acc: 0.4047

Epoch 7/85
----------
LR is set to 0.001
train Loss: 0.0118 Acc: 0.6434
val Loss: 0.0254 Acc: 0.4279

Epoch 8/85
----------
train Loss: 0.0112 Acc: 0.7027
val Loss: 0.0242 Acc: 0.4605

Epoch 9/85
----------
train Loss: 0.0112 Acc: 0.6899
val Loss: 0.0256 Acc: 0.4605

Epoch 10/85
----------
train Loss: 0.0110 Acc: 0.7027
val Loss: 0.0231 Acc: 0.4884

Epoch 11/85
----------
train Loss: 0.0109 Acc: 0.7096
val Loss: 0.0251 Acc: 0.4558

Epoch 12/85
----------
train Loss: 0.0109 Acc: 0.7131
val Loss: 0.0251 Acc: 0.4744

Epoch 13/85
----------
train Loss: 0.0110 Acc: 0.7073
val Loss: 0.0229 Acc: 0.4651

Epoch 14/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0108 Acc: 0.7027
val Loss: 0.0221 Acc: 0.4744

Epoch 15/85
----------
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0218 Acc: 0.4651

Epoch 16/85
----------
train Loss: 0.0107 Acc: 0.7224
val Loss: 0.0258 Acc: 0.4651

Epoch 17/85
----------
train Loss: 0.0108 Acc: 0.7120
val Loss: 0.0216 Acc: 0.4651

Epoch 18/85
----------
train Loss: 0.0107 Acc: 0.7131
val Loss: 0.0229 Acc: 0.4744

Epoch 19/85
----------
train Loss: 0.0107 Acc: 0.7259
val Loss: 0.0252 Acc: 0.4698

Epoch 20/85
----------
train Loss: 0.0108 Acc: 0.7062
val Loss: 0.0242 Acc: 0.4651

Epoch 21/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0217 Acc: 0.4698

Epoch 22/85
----------
train Loss: 0.0108 Acc: 0.7143
val Loss: 0.0225 Acc: 0.4651

Epoch 23/85
----------
train Loss: 0.0106 Acc: 0.7282
val Loss: 0.0260 Acc: 0.4651

Epoch 24/85
----------
train Loss: 0.0106 Acc: 0.7329
val Loss: 0.0266 Acc: 0.4651

Epoch 25/85
----------
train Loss: 0.0106 Acc: 0.7294
val Loss: 0.0268 Acc: 0.4698

Epoch 26/85
----------
train Loss: 0.0108 Acc: 0.7201
val Loss: 0.0223 Acc: 0.4651

Epoch 27/85
----------
train Loss: 0.0108 Acc: 0.7108
val Loss: 0.0250 Acc: 0.4651

Epoch 28/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0108 Acc: 0.7166
val Loss: 0.0238 Acc: 0.4651

Epoch 29/85
----------
train Loss: 0.0107 Acc: 0.7294
val Loss: 0.0241 Acc: 0.4698

Epoch 30/85
----------
train Loss: 0.0108 Acc: 0.7224
val Loss: 0.0226 Acc: 0.4698

Epoch 31/85
----------
train Loss: 0.0108 Acc: 0.7143
val Loss: 0.0242 Acc: 0.4744

Epoch 32/85
----------
train Loss: 0.0108 Acc: 0.7178
val Loss: 0.0232 Acc: 0.4651

Epoch 33/85
----------
train Loss: 0.0108 Acc: 0.7236
val Loss: 0.0235 Acc: 0.4698

Epoch 34/85
----------
train Loss: 0.0108 Acc: 0.7178
val Loss: 0.0221 Acc: 0.4698

Epoch 35/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0107 Acc: 0.7189
val Loss: 0.0236 Acc: 0.4744

Epoch 36/85
----------
train Loss: 0.0107 Acc: 0.7236
val Loss: 0.0232 Acc: 0.4651

Epoch 37/85
----------
train Loss: 0.0108 Acc: 0.7015
val Loss: 0.0250 Acc: 0.4651

Epoch 38/85
----------
train Loss: 0.0107 Acc: 0.7166
val Loss: 0.0250 Acc: 0.4651

Epoch 39/85
----------
train Loss: 0.0107 Acc: 0.7131
val Loss: 0.0235 Acc: 0.4651

Epoch 40/85
----------
train Loss: 0.0106 Acc: 0.7340
val Loss: 0.0237 Acc: 0.4651

Epoch 41/85
----------
train Loss: 0.0105 Acc: 0.7282
val Loss: 0.0246 Acc: 0.4698

Epoch 42/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0108 Acc: 0.7120
val Loss: 0.0252 Acc: 0.4651

Epoch 43/85
----------
train Loss: 0.0108 Acc: 0.6992
val Loss: 0.0231 Acc: 0.4651

Epoch 44/85
----------
train Loss: 0.0108 Acc: 0.7259
val Loss: 0.0246 Acc: 0.4651

Epoch 45/85
----------
train Loss: 0.0108 Acc: 0.7120
val Loss: 0.0224 Acc: 0.4651

Epoch 46/85
----------
train Loss: 0.0109 Acc: 0.7120
val Loss: 0.0270 Acc: 0.4651

Epoch 47/85
----------
train Loss: 0.0108 Acc: 0.7096
val Loss: 0.0239 Acc: 0.4698

Epoch 48/85
----------
train Loss: 0.0109 Acc: 0.7050
val Loss: 0.0260 Acc: 0.4651

Epoch 49/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0107 Acc: 0.7085
val Loss: 0.0240 Acc: 0.4651

Epoch 50/85
----------
train Loss: 0.0108 Acc: 0.7143
val Loss: 0.0242 Acc: 0.4651

Epoch 51/85
----------
train Loss: 0.0107 Acc: 0.7131
val Loss: 0.0256 Acc: 0.4651

Epoch 52/85
----------
train Loss: 0.0105 Acc: 0.7096
val Loss: 0.0234 Acc: 0.4651

Epoch 53/85
----------
train Loss: 0.0108 Acc: 0.7120
val Loss: 0.0227 Acc: 0.4744

Epoch 54/85
----------
train Loss: 0.0106 Acc: 0.7224
val Loss: 0.0239 Acc: 0.4651

Epoch 55/85
----------
train Loss: 0.0108 Acc: 0.7108
val Loss: 0.0253 Acc: 0.4744

Epoch 56/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0107 Acc: 0.7189
val Loss: 0.0242 Acc: 0.4744

Epoch 57/85
----------
train Loss: 0.0106 Acc: 0.7131
val Loss: 0.0250 Acc: 0.4698

Epoch 58/85
----------
train Loss: 0.0108 Acc: 0.7224
val Loss: 0.0257 Acc: 0.4791

Epoch 59/85
----------
train Loss: 0.0107 Acc: 0.7201
val Loss: 0.0249 Acc: 0.4744

Epoch 60/85
----------
train Loss: 0.0106 Acc: 0.7364
val Loss: 0.0223 Acc: 0.4744

Epoch 61/85
----------
train Loss: 0.0107 Acc: 0.7166
val Loss: 0.0263 Acc: 0.4651

Epoch 62/85
----------
train Loss: 0.0107 Acc: 0.7166
val Loss: 0.0228 Acc: 0.4744

Epoch 63/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0107 Acc: 0.7108
val Loss: 0.0221 Acc: 0.4698

Epoch 64/85
----------
train Loss: 0.0106 Acc: 0.7131
val Loss: 0.0247 Acc: 0.4744

Epoch 65/85
----------
train Loss: 0.0108 Acc: 0.7154
val Loss: 0.0243 Acc: 0.4698

Epoch 66/85
----------
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0207 Acc: 0.4698

Epoch 67/85
----------
train Loss: 0.0107 Acc: 0.7259
val Loss: 0.0237 Acc: 0.4744

Epoch 68/85
----------
train Loss: 0.0107 Acc: 0.7224
val Loss: 0.0249 Acc: 0.4698

Epoch 69/85
----------
train Loss: 0.0107 Acc: 0.7178
val Loss: 0.0242 Acc: 0.4791

Epoch 70/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0108 Acc: 0.7189
val Loss: 0.0249 Acc: 0.4651

Epoch 71/85
----------
train Loss: 0.0108 Acc: 0.7143
val Loss: 0.0228 Acc: 0.4698

Epoch 72/85
----------
train Loss: 0.0109 Acc: 0.7131
val Loss: 0.0228 Acc: 0.4698

Epoch 73/85
----------
train Loss: 0.0107 Acc: 0.7003
val Loss: 0.0253 Acc: 0.4698

Epoch 74/85
----------
train Loss: 0.0107 Acc: 0.7213
val Loss: 0.0243 Acc: 0.4698

Epoch 75/85
----------
train Loss: 0.0110 Acc: 0.7131
val Loss: 0.0217 Acc: 0.4651

Epoch 76/85
----------
train Loss: 0.0107 Acc: 0.7154
val Loss: 0.0234 Acc: 0.4698

Epoch 77/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0109 Acc: 0.7154
val Loss: 0.0241 Acc: 0.4698

Epoch 78/85
----------
train Loss: 0.0108 Acc: 0.7073
val Loss: 0.0269 Acc: 0.4698

Epoch 79/85
----------
train Loss: 0.0106 Acc: 0.7213
val Loss: 0.0235 Acc: 0.4698

Epoch 80/85
----------
train Loss: 0.0108 Acc: 0.7096
val Loss: 0.0227 Acc: 0.4744

Epoch 81/85
----------
train Loss: 0.0107 Acc: 0.7282
val Loss: 0.0251 Acc: 0.4698

Epoch 82/85
----------
train Loss: 0.0106 Acc: 0.7236
val Loss: 0.0237 Acc: 0.4698

Epoch 83/85
----------
train Loss: 0.0107 Acc: 0.7224
val Loss: 0.0240 Acc: 0.4698

Epoch 84/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0107 Acc: 0.7131
val Loss: 0.0232 Acc: 0.4651

Epoch 85/85
----------
train Loss: 0.0107 Acc: 0.7201
val Loss: 0.0270 Acc: 0.4698

Training complete in 7m 41s
Best val Acc: 0.488372

---Fine tuning.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0113 Acc: 0.6690
val Loss: 0.0242 Acc: 0.5256

Epoch 1/85
----------
train Loss: 0.0068 Acc: 0.8165
val Loss: 0.0191 Acc: 0.5302

Epoch 2/85
----------
train Loss: 0.0038 Acc: 0.9164
val Loss: 0.0202 Acc: 0.5302

Epoch 3/85
----------
train Loss: 0.0020 Acc: 0.9640
val Loss: 0.0217 Acc: 0.5349

Epoch 4/85
----------
train Loss: 0.0015 Acc: 0.9686
val Loss: 0.0241 Acc: 0.5209

Epoch 5/85
----------
train Loss: 0.0011 Acc: 0.9756
val Loss: 0.0225 Acc: 0.5581

Epoch 6/85
----------
train Loss: 0.0009 Acc: 0.9779
val Loss: 0.0238 Acc: 0.5442

Epoch 7/85
----------
LR is set to 0.001
train Loss: 0.0007 Acc: 0.9803
val Loss: 0.0226 Acc: 0.5442

Epoch 8/85
----------
train Loss: 0.0007 Acc: 0.9791
val Loss: 0.0235 Acc: 0.5395

Epoch 9/85
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0204 Acc: 0.5442

Epoch 10/85
----------
train Loss: 0.0006 Acc: 0.9791
val Loss: 0.0241 Acc: 0.5395

Epoch 11/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0239 Acc: 0.5442

Epoch 12/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0227 Acc: 0.5535

Epoch 13/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0210 Acc: 0.5488

Epoch 14/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0242 Acc: 0.5535

Epoch 15/85
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0241 Acc: 0.5535

Epoch 16/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0206 Acc: 0.5581

Epoch 17/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0247 Acc: 0.5581

Epoch 18/85
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0213 Acc: 0.5581

Epoch 19/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0200 Acc: 0.5581

Epoch 20/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0244 Acc: 0.5581

Epoch 21/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0232 Acc: 0.5488

Epoch 22/85
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0262 Acc: 0.5535

Epoch 23/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0203 Acc: 0.5535

Epoch 24/85
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0224 Acc: 0.5535

Epoch 25/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0265 Acc: 0.5581

Epoch 26/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0223 Acc: 0.5488

Epoch 27/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0234 Acc: 0.5581

Epoch 28/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0217 Acc: 0.5581

Epoch 29/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0231 Acc: 0.5488

Epoch 30/85
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0265 Acc: 0.5581

Epoch 31/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0246 Acc: 0.5581

Epoch 32/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0233 Acc: 0.5581

Epoch 33/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0229 Acc: 0.5581

Epoch 34/85
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0226 Acc: 0.5535

Epoch 35/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0242 Acc: 0.5535

Epoch 36/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0225 Acc: 0.5581

Epoch 37/85
----------
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0205 Acc: 0.5535

Epoch 38/85
----------
train Loss: 0.0005 Acc: 0.9872
val Loss: 0.0243 Acc: 0.5535

Epoch 39/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0282 Acc: 0.5535

Epoch 40/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0189 Acc: 0.5535

Epoch 41/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0215 Acc: 0.5535

Epoch 42/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0200 Acc: 0.5581

Epoch 43/85
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0218 Acc: 0.5581

Epoch 44/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0261 Acc: 0.5628

Epoch 45/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0196 Acc: 0.5581

Epoch 46/85
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0230 Acc: 0.5535

Epoch 47/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0214 Acc: 0.5488

Epoch 48/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0220 Acc: 0.5628

Epoch 49/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0188 Acc: 0.5581

Epoch 50/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0239 Acc: 0.5581

Epoch 51/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0219 Acc: 0.5581

Epoch 52/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0242 Acc: 0.5581

Epoch 53/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0218 Acc: 0.5535

Epoch 54/85
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0208 Acc: 0.5581

Epoch 55/85
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0254 Acc: 0.5535

Epoch 56/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0247 Acc: 0.5488

Epoch 57/85
----------
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0258 Acc: 0.5535

Epoch 58/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0202 Acc: 0.5581

Epoch 59/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0227 Acc: 0.5535

Epoch 60/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0251 Acc: 0.5581

Epoch 61/85
----------
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0261 Acc: 0.5535

Epoch 62/85
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0202 Acc: 0.5581

Epoch 63/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0215 Acc: 0.5535

Epoch 64/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0242 Acc: 0.5535

Epoch 65/85
----------
train Loss: 0.0005 Acc: 0.9884
val Loss: 0.0226 Acc: 0.5535

Epoch 66/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0256 Acc: 0.5535

Epoch 67/85
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0234 Acc: 0.5535

Epoch 68/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0207 Acc: 0.5535

Epoch 69/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0225 Acc: 0.5488

Epoch 70/85
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0004 Acc: 0.9930
val Loss: 0.0194 Acc: 0.5535

Epoch 71/85
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0233 Acc: 0.5581

Epoch 72/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0230 Acc: 0.5488

Epoch 73/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0207 Acc: 0.5488

Epoch 74/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0243 Acc: 0.5488

Epoch 75/85
----------
train Loss: 0.0004 Acc: 0.9861
val Loss: 0.0244 Acc: 0.5488

Epoch 76/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0227 Acc: 0.5535

Epoch 77/85
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0243 Acc: 0.5535

Epoch 78/85
----------
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0205 Acc: 0.5535

Epoch 79/85
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0224 Acc: 0.5535

Epoch 80/85
----------
train Loss: 0.0005 Acc: 0.9861
val Loss: 0.0213 Acc: 0.5535

Epoch 81/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0198 Acc: 0.5581

Epoch 82/85
----------
train Loss: 0.0005 Acc: 0.9837
val Loss: 0.0229 Acc: 0.5535

Epoch 83/85
----------
train Loss: 0.0005 Acc: 0.9814
val Loss: 0.0216 Acc: 0.5535

Epoch 84/85
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0005 Acc: 0.9849
val Loss: 0.0247 Acc: 0.5535

Epoch 85/85
----------
train Loss: 0.0005 Acc: 0.9803
val Loss: 0.0240 Acc: 0.5535

Training complete in 8m 4s
Best val Acc: 0.562791

---Testing---
Test accuracy: 0.899628
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of Bigeye tuna : 83 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 90 %
Accuracy of Frigate tuna : 82 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 64 %
Accuracy of Southern bluefin tuna : 80 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8688433356209717, std: 0.08189525553554858
--------------------

run info[val: 0.25, epoch: 88, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0289 Acc: 0.1400
val Loss: 0.0281 Acc: 0.2193

Epoch 1/87
----------
train Loss: 0.0261 Acc: 0.2701
val Loss: 0.0241 Acc: 0.2900

Epoch 2/87
----------
train Loss: 0.0225 Acc: 0.3730
val Loss: 0.0214 Acc: 0.3941

Epoch 3/87
----------
train Loss: 0.0210 Acc: 0.4548
val Loss: 0.0216 Acc: 0.4201

Epoch 4/87
----------
train Loss: 0.0191 Acc: 0.4647
val Loss: 0.0219 Acc: 0.3941

Epoch 5/87
----------
train Loss: 0.0193 Acc: 0.4981
val Loss: 0.0205 Acc: 0.4387

Epoch 6/87
----------
train Loss: 0.0162 Acc: 0.5514
val Loss: 0.0220 Acc: 0.3903

Epoch 7/87
----------
LR is set to 0.001
train Loss: 0.0162 Acc: 0.5353
val Loss: 0.0194 Acc: 0.4610

Epoch 8/87
----------
train Loss: 0.0155 Acc: 0.6183
val Loss: 0.0195 Acc: 0.4610

Epoch 9/87
----------
train Loss: 0.0137 Acc: 0.6121
val Loss: 0.0194 Acc: 0.4461

Epoch 10/87
----------
train Loss: 0.0140 Acc: 0.6022
val Loss: 0.0191 Acc: 0.4833

Epoch 11/87
----------
train Loss: 0.0131 Acc: 0.6530
val Loss: 0.0190 Acc: 0.4684

Epoch 12/87
----------
train Loss: 0.0136 Acc: 0.6369
val Loss: 0.0186 Acc: 0.4610

Epoch 13/87
----------
train Loss: 0.0142 Acc: 0.6221
val Loss: 0.0191 Acc: 0.4758

Epoch 14/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0132 Acc: 0.6283
val Loss: 0.0189 Acc: 0.4758

Epoch 15/87
----------
train Loss: 0.0144 Acc: 0.6307
val Loss: 0.0185 Acc: 0.4796

Epoch 16/87
----------
train Loss: 0.0137 Acc: 0.6419
val Loss: 0.0185 Acc: 0.4684

Epoch 17/87
----------
train Loss: 0.0142 Acc: 0.6208
val Loss: 0.0185 Acc: 0.4647

Epoch 18/87
----------
train Loss: 0.0128 Acc: 0.6456
val Loss: 0.0184 Acc: 0.4684

Epoch 19/87
----------
train Loss: 0.0139 Acc: 0.6654
val Loss: 0.0189 Acc: 0.4796

Epoch 20/87
----------
train Loss: 0.0137 Acc: 0.6642
val Loss: 0.0188 Acc: 0.4647

Epoch 21/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0131 Acc: 0.6431
val Loss: 0.0187 Acc: 0.4833

Epoch 22/87
----------
train Loss: 0.0139 Acc: 0.6543
val Loss: 0.0189 Acc: 0.4833

Epoch 23/87
----------
train Loss: 0.0126 Acc: 0.6357
val Loss: 0.0189 Acc: 0.4721

Epoch 24/87
----------
train Loss: 0.0139 Acc: 0.6543
val Loss: 0.0188 Acc: 0.4833

Epoch 25/87
----------
train Loss: 0.0132 Acc: 0.6419
val Loss: 0.0184 Acc: 0.4758

Epoch 26/87
----------
train Loss: 0.0127 Acc: 0.6369
val Loss: 0.0189 Acc: 0.4758

Epoch 27/87
----------
train Loss: 0.0128 Acc: 0.6605
val Loss: 0.0186 Acc: 0.4758

Epoch 28/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0130 Acc: 0.6543
val Loss: 0.0186 Acc: 0.4647

Epoch 29/87
----------
train Loss: 0.0136 Acc: 0.6518
val Loss: 0.0186 Acc: 0.4870

Epoch 30/87
----------
train Loss: 0.0140 Acc: 0.6592
val Loss: 0.0185 Acc: 0.4758

Epoch 31/87
----------
train Loss: 0.0130 Acc: 0.6592
val Loss: 0.0187 Acc: 0.4870

Epoch 32/87
----------
train Loss: 0.0138 Acc: 0.6629
val Loss: 0.0184 Acc: 0.4721

Epoch 33/87
----------
train Loss: 0.0125 Acc: 0.6369
val Loss: 0.0186 Acc: 0.4796

Epoch 34/87
----------
train Loss: 0.0134 Acc: 0.6369
val Loss: 0.0190 Acc: 0.4721

Epoch 35/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0133 Acc: 0.6481
val Loss: 0.0190 Acc: 0.4721

Epoch 36/87
----------
train Loss: 0.0131 Acc: 0.6357
val Loss: 0.0189 Acc: 0.4721

Epoch 37/87
----------
train Loss: 0.0127 Acc: 0.6605
val Loss: 0.0184 Acc: 0.4721

Epoch 38/87
----------
train Loss: 0.0132 Acc: 0.6481
val Loss: 0.0185 Acc: 0.4870

Epoch 39/87
----------
train Loss: 0.0134 Acc: 0.6332
val Loss: 0.0188 Acc: 0.4796

Epoch 40/87
----------
train Loss: 0.0130 Acc: 0.6406
val Loss: 0.0185 Acc: 0.4833

Epoch 41/87
----------
train Loss: 0.0134 Acc: 0.6394
val Loss: 0.0188 Acc: 0.4684

Epoch 42/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0133 Acc: 0.6468
val Loss: 0.0186 Acc: 0.4796

Epoch 43/87
----------
train Loss: 0.0132 Acc: 0.6357
val Loss: 0.0185 Acc: 0.4833

Epoch 44/87
----------
train Loss: 0.0130 Acc: 0.6431
val Loss: 0.0185 Acc: 0.4870

Epoch 45/87
----------
train Loss: 0.0137 Acc: 0.6555
val Loss: 0.0185 Acc: 0.4796

Epoch 46/87
----------
train Loss: 0.0135 Acc: 0.6456
val Loss: 0.0186 Acc: 0.4907

Epoch 47/87
----------
train Loss: 0.0128 Acc: 0.6580
val Loss: 0.0189 Acc: 0.4796

Epoch 48/87
----------
train Loss: 0.0128 Acc: 0.6518
val Loss: 0.0188 Acc: 0.4833

Epoch 49/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0136 Acc: 0.6382
val Loss: 0.0187 Acc: 0.4721

Epoch 50/87
----------
train Loss: 0.0135 Acc: 0.6456
val Loss: 0.0188 Acc: 0.4796

Epoch 51/87
----------
train Loss: 0.0134 Acc: 0.6382
val Loss: 0.0183 Acc: 0.4684

Epoch 52/87
----------
train Loss: 0.0133 Acc: 0.6530
val Loss: 0.0188 Acc: 0.4796

Epoch 53/87
----------
train Loss: 0.0130 Acc: 0.6270
val Loss: 0.0185 Acc: 0.4796

Epoch 54/87
----------
train Loss: 0.0128 Acc: 0.6691
val Loss: 0.0185 Acc: 0.4833

Epoch 55/87
----------
train Loss: 0.0140 Acc: 0.6245
val Loss: 0.0188 Acc: 0.4796

Epoch 56/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0139 Acc: 0.6320
val Loss: 0.0187 Acc: 0.4721

Epoch 57/87
----------
train Loss: 0.0130 Acc: 0.6431
val Loss: 0.0188 Acc: 0.4758

Epoch 58/87
----------
train Loss: 0.0133 Acc: 0.6419
val Loss: 0.0187 Acc: 0.4758

Epoch 59/87
----------
train Loss: 0.0132 Acc: 0.6382
val Loss: 0.0185 Acc: 0.4721

Epoch 60/87
----------
train Loss: 0.0135 Acc: 0.6506
val Loss: 0.0186 Acc: 0.4684

Epoch 61/87
----------
train Loss: 0.0128 Acc: 0.6691
val Loss: 0.0191 Acc: 0.4758

Epoch 62/87
----------
train Loss: 0.0139 Acc: 0.6419
val Loss: 0.0186 Acc: 0.4758

Epoch 63/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0126 Acc: 0.6580
val Loss: 0.0186 Acc: 0.4684

Epoch 64/87
----------
train Loss: 0.0131 Acc: 0.6506
val Loss: 0.0184 Acc: 0.4758

Epoch 65/87
----------
train Loss: 0.0139 Acc: 0.6468
val Loss: 0.0189 Acc: 0.4647

Epoch 66/87
----------
train Loss: 0.0137 Acc: 0.6518
val Loss: 0.0185 Acc: 0.4796

Epoch 67/87
----------
train Loss: 0.0127 Acc: 0.6493
val Loss: 0.0185 Acc: 0.4684

Epoch 68/87
----------
train Loss: 0.0132 Acc: 0.6654
val Loss: 0.0185 Acc: 0.4610

Epoch 69/87
----------
train Loss: 0.0129 Acc: 0.6481
val Loss: 0.0184 Acc: 0.4796

Epoch 70/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0133 Acc: 0.6468
val Loss: 0.0185 Acc: 0.4758

Epoch 71/87
----------
train Loss: 0.0147 Acc: 0.6159
val Loss: 0.0188 Acc: 0.4833

Epoch 72/87
----------
train Loss: 0.0129 Acc: 0.6307
val Loss: 0.0188 Acc: 0.4758

Epoch 73/87
----------
train Loss: 0.0128 Acc: 0.6592
val Loss: 0.0187 Acc: 0.4721

Epoch 74/87
----------
train Loss: 0.0126 Acc: 0.6468
val Loss: 0.0187 Acc: 0.4796

Epoch 75/87
----------
train Loss: 0.0133 Acc: 0.6642
val Loss: 0.0187 Acc: 0.4721

Epoch 76/87
----------
train Loss: 0.0131 Acc: 0.6555
val Loss: 0.0188 Acc: 0.4721

Epoch 77/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0138 Acc: 0.6431
val Loss: 0.0187 Acc: 0.4907

Epoch 78/87
----------
train Loss: 0.0130 Acc: 0.6481
val Loss: 0.0186 Acc: 0.4758

Epoch 79/87
----------
train Loss: 0.0137 Acc: 0.6518
val Loss: 0.0188 Acc: 0.4647

Epoch 80/87
----------
train Loss: 0.0146 Acc: 0.6431
val Loss: 0.0185 Acc: 0.4721

Epoch 81/87
----------
train Loss: 0.0139 Acc: 0.6518
val Loss: 0.0186 Acc: 0.4870

Epoch 82/87
----------
train Loss: 0.0134 Acc: 0.6592
val Loss: 0.0187 Acc: 0.4907

Epoch 83/87
----------
train Loss: 0.0138 Acc: 0.6493
val Loss: 0.0184 Acc: 0.4758

Epoch 84/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0138 Acc: 0.6530
val Loss: 0.0183 Acc: 0.4833

Epoch 85/87
----------
train Loss: 0.0133 Acc: 0.6617
val Loss: 0.0188 Acc: 0.4721

Epoch 86/87
----------
train Loss: 0.0133 Acc: 0.6481
val Loss: 0.0187 Acc: 0.4833

Epoch 87/87
----------
train Loss: 0.0141 Acc: 0.6704
val Loss: 0.0186 Acc: 0.4796

Training complete in 8m 2s
Best val Acc: 0.490706

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0146 Acc: 0.6022
val Loss: 0.0185 Acc: 0.4796

Epoch 1/87
----------
train Loss: 0.0109 Acc: 0.6927
val Loss: 0.0206 Acc: 0.4349

Epoch 2/87
----------
train Loss: 0.0075 Acc: 0.7968
val Loss: 0.0195 Acc: 0.4870

Epoch 3/87
----------
train Loss: 0.0064 Acc: 0.8104
val Loss: 0.0225 Acc: 0.5242

Epoch 4/87
----------
train Loss: 0.0060 Acc: 0.8092
val Loss: 0.0304 Acc: 0.4387

Epoch 5/87
----------
train Loss: 0.0058 Acc: 0.8463
val Loss: 0.0230 Acc: 0.5167

Epoch 6/87
----------
train Loss: 0.0046 Acc: 0.8538
val Loss: 0.0288 Acc: 0.3978

Epoch 7/87
----------
LR is set to 0.001
train Loss: 0.0035 Acc: 0.9133
val Loss: 0.0218 Acc: 0.4907

Epoch 8/87
----------
train Loss: 0.0030 Acc: 0.9257
val Loss: 0.0189 Acc: 0.5465

Epoch 9/87
----------
train Loss: 0.0036 Acc: 0.9517
val Loss: 0.0198 Acc: 0.5502

Epoch 10/87
----------
train Loss: 0.0020 Acc: 0.9492
val Loss: 0.0190 Acc: 0.5576

Epoch 11/87
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0187 Acc: 0.5725

Epoch 12/87
----------
train Loss: 0.0019 Acc: 0.9603
val Loss: 0.0186 Acc: 0.5613

Epoch 13/87
----------
train Loss: 0.0014 Acc: 0.9641
val Loss: 0.0187 Acc: 0.5725

Epoch 14/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0017 Acc: 0.9703
val Loss: 0.0187 Acc: 0.5688

Epoch 15/87
----------
train Loss: 0.0015 Acc: 0.9641
val Loss: 0.0188 Acc: 0.5725

Epoch 16/87
----------
train Loss: 0.0018 Acc: 0.9665
val Loss: 0.0186 Acc: 0.5725

Epoch 17/87
----------
train Loss: 0.0023 Acc: 0.9715
val Loss: 0.0188 Acc: 0.5725

Epoch 18/87
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0187 Acc: 0.5836

Epoch 19/87
----------
train Loss: 0.0020 Acc: 0.9678
val Loss: 0.0190 Acc: 0.5725

Epoch 20/87
----------
train Loss: 0.0016 Acc: 0.9715
val Loss: 0.0188 Acc: 0.5762

Epoch 21/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9678
val Loss: 0.0190 Acc: 0.5911

Epoch 22/87
----------
train Loss: 0.0016 Acc: 0.9765
val Loss: 0.0191 Acc: 0.5799

Epoch 23/87
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0187 Acc: 0.5799

Epoch 24/87
----------
train Loss: 0.0015 Acc: 0.9690
val Loss: 0.0187 Acc: 0.5725

Epoch 25/87
----------
train Loss: 0.0014 Acc: 0.9827
val Loss: 0.0185 Acc: 0.5725

Epoch 26/87
----------
train Loss: 0.0015 Acc: 0.9715
val Loss: 0.0187 Acc: 0.5762

Epoch 27/87
----------
train Loss: 0.0018 Acc: 0.9789
val Loss: 0.0187 Acc: 0.5762

Epoch 28/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0015 Acc: 0.9765
val Loss: 0.0188 Acc: 0.5762

Epoch 29/87
----------
train Loss: 0.0013 Acc: 0.9765
val Loss: 0.0185 Acc: 0.5613

Epoch 30/87
----------
train Loss: 0.0012 Acc: 0.9740
val Loss: 0.0189 Acc: 0.5762

Epoch 31/87
----------
train Loss: 0.0017 Acc: 0.9715
val Loss: 0.0186 Acc: 0.5576

Epoch 32/87
----------
train Loss: 0.0020 Acc: 0.9653
val Loss: 0.0192 Acc: 0.5688

Epoch 33/87
----------
train Loss: 0.0020 Acc: 0.9740
val Loss: 0.0190 Acc: 0.5688

Epoch 34/87
----------
train Loss: 0.0017 Acc: 0.9765
val Loss: 0.0191 Acc: 0.5725

Epoch 35/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0019 Acc: 0.9727
val Loss: 0.0190 Acc: 0.5651

Epoch 36/87
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0189 Acc: 0.5836

Epoch 37/87
----------
train Loss: 0.0013 Acc: 0.9690
val Loss: 0.0187 Acc: 0.5725

Epoch 38/87
----------
train Loss: 0.0020 Acc: 0.9715
val Loss: 0.0187 Acc: 0.5799

Epoch 39/87
----------
train Loss: 0.0017 Acc: 0.9678
val Loss: 0.0185 Acc: 0.5799

Epoch 40/87
----------
train Loss: 0.0019 Acc: 0.9765
val Loss: 0.0186 Acc: 0.5762

Epoch 41/87
----------
train Loss: 0.0015 Acc: 0.9715
val Loss: 0.0181 Acc: 0.5762

Epoch 42/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0016 Acc: 0.9616
val Loss: 0.0190 Acc: 0.5762

Epoch 43/87
----------
train Loss: 0.0018 Acc: 0.9765
val Loss: 0.0190 Acc: 0.5613

Epoch 44/87
----------
train Loss: 0.0013 Acc: 0.9777
val Loss: 0.0187 Acc: 0.5762

Epoch 45/87
----------
train Loss: 0.0016 Acc: 0.9678
val Loss: 0.0185 Acc: 0.5799

Epoch 46/87
----------
train Loss: 0.0015 Acc: 0.9715
val Loss: 0.0190 Acc: 0.5725

Epoch 47/87
----------
train Loss: 0.0018 Acc: 0.9703
val Loss: 0.0188 Acc: 0.5725

Epoch 48/87
----------
train Loss: 0.0014 Acc: 0.9777
val Loss: 0.0189 Acc: 0.5725

Epoch 49/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0015 Acc: 0.9740
val Loss: 0.0189 Acc: 0.5762

Epoch 50/87
----------
train Loss: 0.0016 Acc: 0.9752
val Loss: 0.0192 Acc: 0.5725

Epoch 51/87
----------
train Loss: 0.0015 Acc: 0.9752
val Loss: 0.0187 Acc: 0.5799

Epoch 52/87
----------
train Loss: 0.0026 Acc: 0.9727
val Loss: 0.0190 Acc: 0.5688

Epoch 53/87
----------
train Loss: 0.0016 Acc: 0.9727
val Loss: 0.0190 Acc: 0.5613

Epoch 54/87
----------
train Loss: 0.0017 Acc: 0.9752
val Loss: 0.0189 Acc: 0.5762

Epoch 55/87
----------
train Loss: 0.0015 Acc: 0.9740
val Loss: 0.0194 Acc: 0.5725

Epoch 56/87
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0014 Acc: 0.9740
val Loss: 0.0190 Acc: 0.5688

Epoch 57/87
----------
train Loss: 0.0021 Acc: 0.9678
val Loss: 0.0185 Acc: 0.5725

Epoch 58/87
----------
train Loss: 0.0014 Acc: 0.9703
val Loss: 0.0185 Acc: 0.5762

Epoch 59/87
----------
train Loss: 0.0015 Acc: 0.9839
val Loss: 0.0183 Acc: 0.5688

Epoch 60/87
----------
train Loss: 0.0015 Acc: 0.9740
val Loss: 0.0180 Acc: 0.5725

Epoch 61/87
----------
train Loss: 0.0019 Acc: 0.9727
val Loss: 0.0188 Acc: 0.5762

Epoch 62/87
----------
train Loss: 0.0023 Acc: 0.9690
val Loss: 0.0190 Acc: 0.5725

Epoch 63/87
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0022 Acc: 0.9752
val Loss: 0.0188 Acc: 0.5651

Epoch 64/87
----------
train Loss: 0.0013 Acc: 0.9765
val Loss: 0.0182 Acc: 0.5725

Epoch 65/87
----------
train Loss: 0.0014 Acc: 0.9703
val Loss: 0.0186 Acc: 0.5688

Epoch 66/87
----------
train Loss: 0.0025 Acc: 0.9616
val Loss: 0.0185 Acc: 0.5539

Epoch 67/87
----------
train Loss: 0.0016 Acc: 0.9641
val Loss: 0.0186 Acc: 0.5725

Epoch 68/87
----------
train Loss: 0.0014 Acc: 0.9653
val Loss: 0.0184 Acc: 0.5688

Epoch 69/87
----------
train Loss: 0.0017 Acc: 0.9752
val Loss: 0.0186 Acc: 0.5836

Epoch 70/87
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0017 Acc: 0.9690
val Loss: 0.0179 Acc: 0.5762

Epoch 71/87
----------
train Loss: 0.0012 Acc: 0.9765
val Loss: 0.0189 Acc: 0.5799

Epoch 72/87
----------
train Loss: 0.0018 Acc: 0.9752
val Loss: 0.0189 Acc: 0.5836

Epoch 73/87
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0192 Acc: 0.5688

Epoch 74/87
----------
train Loss: 0.0013 Acc: 0.9727
val Loss: 0.0188 Acc: 0.5725

Epoch 75/87
----------
train Loss: 0.0013 Acc: 0.9703
val Loss: 0.0189 Acc: 0.5725

Epoch 76/87
----------
train Loss: 0.0021 Acc: 0.9715
val Loss: 0.0189 Acc: 0.5725

Epoch 77/87
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0019 Acc: 0.9727
val Loss: 0.0188 Acc: 0.5799

Epoch 78/87
----------
train Loss: 0.0018 Acc: 0.9727
val Loss: 0.0188 Acc: 0.5799

Epoch 79/87
----------
train Loss: 0.0014 Acc: 0.9789
val Loss: 0.0186 Acc: 0.5799

Epoch 80/87
----------
train Loss: 0.0019 Acc: 0.9690
val Loss: 0.0189 Acc: 0.5762

Epoch 81/87
----------
train Loss: 0.0017 Acc: 0.9678
val Loss: 0.0187 Acc: 0.5762

Epoch 82/87
----------
train Loss: 0.0018 Acc: 0.9690
val Loss: 0.0188 Acc: 0.5762

Epoch 83/87
----------
train Loss: 0.0014 Acc: 0.9727
val Loss: 0.0190 Acc: 0.5762

Epoch 84/87
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0015 Acc: 0.9703
val Loss: 0.0191 Acc: 0.5725

Epoch 85/87
----------
train Loss: 0.0014 Acc: 0.9715
val Loss: 0.0187 Acc: 0.5799

Epoch 86/87
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0188 Acc: 0.5799

Epoch 87/87
----------
train Loss: 0.0018 Acc: 0.9628
val Loss: 0.0186 Acc: 0.5799

Training complete in 8m 32s
Best val Acc: 0.591078

---Testing---
Test accuracy: 0.881970
--------------------
Accuracy of Albacore tuna : 86 %
Accuracy of Atlantic bluefin tuna : 87 %
Accuracy of Bigeye tuna : 74 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 93 %
Accuracy of Frigate tuna : 65 %
Accuracy of Little tunny : 92 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 57 %
Accuracy of Southern bluefin tuna : 75 %
Accuracy of Yellowfin tuna : 96 %
mean: 0.8380764989807383, std: 0.11774593515228245
--------------------

run info[val: 0.3, epoch: 88, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0272 Acc: 0.1485
val Loss: 0.0290 Acc: 0.2422

Epoch 1/87
----------
train Loss: 0.0228 Acc: 0.3156
val Loss: 0.0264 Acc: 0.3292

Epoch 2/87
----------
train Loss: 0.0199 Acc: 0.4085
val Loss: 0.0253 Acc: 0.3696

Epoch 3/87
----------
train Loss: 0.0179 Acc: 0.4589
val Loss: 0.0227 Acc: 0.4317

Epoch 4/87
----------
train Loss: 0.0160 Acc: 0.5292
val Loss: 0.0221 Acc: 0.4068

Epoch 5/87
----------
train Loss: 0.0150 Acc: 0.5703
val Loss: 0.0215 Acc: 0.4130

Epoch 6/87
----------
train Loss: 0.0137 Acc: 0.6101
val Loss: 0.0219 Acc: 0.4441

Epoch 7/87
----------
train Loss: 0.0136 Acc: 0.6034
val Loss: 0.0219 Acc: 0.4565

Epoch 8/87
----------
train Loss: 0.0124 Acc: 0.6459
val Loss: 0.0198 Acc: 0.4596

Epoch 9/87
----------
train Loss: 0.0120 Acc: 0.6631
val Loss: 0.0212 Acc: 0.4752

Epoch 10/87
----------
train Loss: 0.0117 Acc: 0.6963
val Loss: 0.0209 Acc: 0.4472

Epoch 11/87
----------
LR is set to 0.001
train Loss: 0.0110 Acc: 0.6976
val Loss: 0.0216 Acc: 0.4596

Epoch 12/87
----------
train Loss: 0.0108 Acc: 0.7029
val Loss: 0.0202 Acc: 0.4876

Epoch 13/87
----------
train Loss: 0.0105 Acc: 0.7334
val Loss: 0.0209 Acc: 0.4814

Epoch 14/87
----------
train Loss: 0.0106 Acc: 0.7069
val Loss: 0.0202 Acc: 0.4845

Epoch 15/87
----------
train Loss: 0.0108 Acc: 0.7188
val Loss: 0.0201 Acc: 0.4814

Epoch 16/87
----------
train Loss: 0.0106 Acc: 0.7294
val Loss: 0.0202 Acc: 0.4876

Epoch 17/87
----------
train Loss: 0.0105 Acc: 0.7122
val Loss: 0.0216 Acc: 0.4876

Epoch 18/87
----------
train Loss: 0.0103 Acc: 0.7361
val Loss: 0.0210 Acc: 0.4876

Epoch 19/87
----------
train Loss: 0.0100 Acc: 0.7480
val Loss: 0.0206 Acc: 0.4845

Epoch 20/87
----------
train Loss: 0.0101 Acc: 0.7308
val Loss: 0.0203 Acc: 0.4783

Epoch 21/87
----------
train Loss: 0.0099 Acc: 0.7387
val Loss: 0.0209 Acc: 0.4783

Epoch 22/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0101 Acc: 0.7387
val Loss: 0.0203 Acc: 0.4783

Epoch 23/87
----------
train Loss: 0.0101 Acc: 0.7374
val Loss: 0.0213 Acc: 0.4845

Epoch 24/87
----------
train Loss: 0.0100 Acc: 0.7321
val Loss: 0.0212 Acc: 0.4845

Epoch 25/87
----------
train Loss: 0.0103 Acc: 0.7347
val Loss: 0.0216 Acc: 0.4845

Epoch 26/87
----------
train Loss: 0.0100 Acc: 0.7255
val Loss: 0.0202 Acc: 0.4845

Epoch 27/87
----------
train Loss: 0.0101 Acc: 0.7228
val Loss: 0.0215 Acc: 0.4876

Epoch 28/87
----------
train Loss: 0.0104 Acc: 0.7281
val Loss: 0.0200 Acc: 0.4845

Epoch 29/87
----------
train Loss: 0.0103 Acc: 0.7374
val Loss: 0.0199 Acc: 0.4876

Epoch 30/87
----------
train Loss: 0.0102 Acc: 0.7308
val Loss: 0.0206 Acc: 0.4845

Epoch 31/87
----------
train Loss: 0.0103 Acc: 0.7255
val Loss: 0.0201 Acc: 0.4845

Epoch 32/87
----------
train Loss: 0.0102 Acc: 0.7308
val Loss: 0.0197 Acc: 0.4907

Epoch 33/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0102 Acc: 0.7281
val Loss: 0.0199 Acc: 0.4907

Epoch 34/87
----------
train Loss: 0.0100 Acc: 0.7440
val Loss: 0.0209 Acc: 0.4845

Epoch 35/87
----------
train Loss: 0.0102 Acc: 0.7202
val Loss: 0.0209 Acc: 0.4907

Epoch 36/87
----------
train Loss: 0.0104 Acc: 0.7255
val Loss: 0.0205 Acc: 0.4845

Epoch 37/87
----------
train Loss: 0.0103 Acc: 0.7308
val Loss: 0.0200 Acc: 0.4845

Epoch 38/87
----------
train Loss: 0.0102 Acc: 0.7215
val Loss: 0.0207 Acc: 0.4907

Epoch 39/87
----------
train Loss: 0.0101 Acc: 0.7294
val Loss: 0.0206 Acc: 0.4876

Epoch 40/87
----------
train Loss: 0.0103 Acc: 0.7294
val Loss: 0.0193 Acc: 0.4876

Epoch 41/87
----------
train Loss: 0.0103 Acc: 0.7281
val Loss: 0.0204 Acc: 0.4907

Epoch 42/87
----------
train Loss: 0.0101 Acc: 0.7268
val Loss: 0.0199 Acc: 0.4876

Epoch 43/87
----------
train Loss: 0.0103 Acc: 0.7321
val Loss: 0.0200 Acc: 0.4876

Epoch 44/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0102 Acc: 0.7202
val Loss: 0.0219 Acc: 0.4845

Epoch 45/87
----------
train Loss: 0.0100 Acc: 0.7321
val Loss: 0.0213 Acc: 0.4876

Epoch 46/87
----------
train Loss: 0.0104 Acc: 0.7321
val Loss: 0.0204 Acc: 0.4876

Epoch 47/87
----------
train Loss: 0.0102 Acc: 0.7427
val Loss: 0.0214 Acc: 0.4845

Epoch 48/87
----------
train Loss: 0.0103 Acc: 0.7294
val Loss: 0.0209 Acc: 0.4907

Epoch 49/87
----------
train Loss: 0.0101 Acc: 0.7454
val Loss: 0.0215 Acc: 0.4907

Epoch 50/87
----------
train Loss: 0.0103 Acc: 0.7135
val Loss: 0.0199 Acc: 0.4845

Epoch 51/87
----------
train Loss: 0.0102 Acc: 0.7281
val Loss: 0.0206 Acc: 0.4876

Epoch 52/87
----------
train Loss: 0.0103 Acc: 0.7414
val Loss: 0.0198 Acc: 0.4876

Epoch 53/87
----------
train Loss: 0.0105 Acc: 0.7109
val Loss: 0.0214 Acc: 0.4876

Epoch 54/87
----------
train Loss: 0.0104 Acc: 0.7387
val Loss: 0.0195 Acc: 0.4845

Epoch 55/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0103 Acc: 0.7308
val Loss: 0.0203 Acc: 0.4845

Epoch 56/87
----------
train Loss: 0.0106 Acc: 0.7082
val Loss: 0.0212 Acc: 0.4845

Epoch 57/87
----------
train Loss: 0.0103 Acc: 0.7308
val Loss: 0.0204 Acc: 0.4845

Epoch 58/87
----------
train Loss: 0.0101 Acc: 0.7281
val Loss: 0.0209 Acc: 0.4907

Epoch 59/87
----------
train Loss: 0.0101 Acc: 0.7308
val Loss: 0.0204 Acc: 0.4876

Epoch 60/87
----------
train Loss: 0.0105 Acc: 0.7095
val Loss: 0.0199 Acc: 0.4876

Epoch 61/87
----------
train Loss: 0.0102 Acc: 0.7149
val Loss: 0.0192 Acc: 0.4907

Epoch 62/87
----------
train Loss: 0.0102 Acc: 0.7546
val Loss: 0.0201 Acc: 0.4876

Epoch 63/87
----------
train Loss: 0.0100 Acc: 0.7414
val Loss: 0.0206 Acc: 0.4876

Epoch 64/87
----------
train Loss: 0.0104 Acc: 0.7294
val Loss: 0.0196 Acc: 0.4969

Epoch 65/87
----------
train Loss: 0.0100 Acc: 0.7454
val Loss: 0.0202 Acc: 0.4814

Epoch 66/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0100 Acc: 0.7414
val Loss: 0.0197 Acc: 0.4845

Epoch 67/87
----------
train Loss: 0.0102 Acc: 0.7414
val Loss: 0.0197 Acc: 0.4907

Epoch 68/87
----------
train Loss: 0.0102 Acc: 0.7427
val Loss: 0.0209 Acc: 0.4876

Epoch 69/87
----------
train Loss: 0.0100 Acc: 0.7520
val Loss: 0.0212 Acc: 0.4907

Epoch 70/87
----------
train Loss: 0.0103 Acc: 0.7162
val Loss: 0.0206 Acc: 0.4907

Epoch 71/87
----------
train Loss: 0.0101 Acc: 0.7414
val Loss: 0.0197 Acc: 0.4907

Epoch 72/87
----------
train Loss: 0.0101 Acc: 0.7454
val Loss: 0.0206 Acc: 0.4845

Epoch 73/87
----------
train Loss: 0.0103 Acc: 0.7241
val Loss: 0.0204 Acc: 0.4876

Epoch 74/87
----------
train Loss: 0.0100 Acc: 0.7321
val Loss: 0.0205 Acc: 0.4907

Epoch 75/87
----------
train Loss: 0.0102 Acc: 0.7109
val Loss: 0.0203 Acc: 0.4845

Epoch 76/87
----------
train Loss: 0.0101 Acc: 0.7467
val Loss: 0.0202 Acc: 0.4876

Epoch 77/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0102 Acc: 0.7414
val Loss: 0.0206 Acc: 0.4907

Epoch 78/87
----------
train Loss: 0.0103 Acc: 0.7387
val Loss: 0.0190 Acc: 0.4938

Epoch 79/87
----------
train Loss: 0.0103 Acc: 0.7215
val Loss: 0.0201 Acc: 0.4938

Epoch 80/87
----------
train Loss: 0.0102 Acc: 0.7401
val Loss: 0.0199 Acc: 0.4938

Epoch 81/87
----------
train Loss: 0.0099 Acc: 0.7321
val Loss: 0.0203 Acc: 0.4876

Epoch 82/87
----------
train Loss: 0.0102 Acc: 0.7361
val Loss: 0.0197 Acc: 0.4876

Epoch 83/87
----------
train Loss: 0.0102 Acc: 0.7347
val Loss: 0.0211 Acc: 0.4876

Epoch 84/87
----------
train Loss: 0.0104 Acc: 0.7255
val Loss: 0.0194 Acc: 0.4876

Epoch 85/87
----------
train Loss: 0.0101 Acc: 0.7347
val Loss: 0.0208 Acc: 0.4876

Epoch 86/87
----------
train Loss: 0.0104 Acc: 0.7188
val Loss: 0.0221 Acc: 0.4876

Epoch 87/87
----------
train Loss: 0.0102 Acc: 0.7241
val Loss: 0.0201 Acc: 0.4845

Training complete in 8m 6s
Best val Acc: 0.496894

---Fine tuning.---
Epoch 0/87
----------
LR is set to 0.01
train Loss: 0.0110 Acc: 0.6883
val Loss: 0.0204 Acc: 0.4534

Epoch 1/87
----------
train Loss: 0.0076 Acc: 0.7798
val Loss: 0.0226 Acc: 0.4814

Epoch 2/87
----------
train Loss: 0.0050 Acc: 0.8687
val Loss: 0.0241 Acc: 0.4503

Epoch 3/87
----------
train Loss: 0.0034 Acc: 0.9164
val Loss: 0.0201 Acc: 0.5062

Epoch 4/87
----------
train Loss: 0.0023 Acc: 0.9523
val Loss: 0.0205 Acc: 0.5248

Epoch 5/87
----------
train Loss: 0.0016 Acc: 0.9668
val Loss: 0.0213 Acc: 0.5186

Epoch 6/87
----------
train Loss: 0.0014 Acc: 0.9708
val Loss: 0.0194 Acc: 0.5466

Epoch 7/87
----------
train Loss: 0.0010 Acc: 0.9801
val Loss: 0.0248 Acc: 0.5217

Epoch 8/87
----------
train Loss: 0.0010 Acc: 0.9775
val Loss: 0.0223 Acc: 0.5342

Epoch 9/87
----------
train Loss: 0.0009 Acc: 0.9775
val Loss: 0.0226 Acc: 0.5311

Epoch 10/87
----------
train Loss: 0.0006 Acc: 0.9881
val Loss: 0.0245 Acc: 0.5124

Epoch 11/87
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0228 Acc: 0.5280

Epoch 12/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0225 Acc: 0.5342

Epoch 13/87
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0224 Acc: 0.5404

Epoch 14/87
----------
train Loss: 0.0005 Acc: 0.9894
val Loss: 0.0208 Acc: 0.5404

Epoch 15/87
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0232 Acc: 0.5466

Epoch 16/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0234 Acc: 0.5435

Epoch 17/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0219 Acc: 0.5497

Epoch 18/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0217 Acc: 0.5466

Epoch 19/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0217 Acc: 0.5497

Epoch 20/87
----------
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0251 Acc: 0.5466

Epoch 21/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0227 Acc: 0.5435

Epoch 22/87
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0233 Acc: 0.5404

Epoch 23/87
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0244 Acc: 0.5404

Epoch 24/87
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0238 Acc: 0.5435

Epoch 25/87
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0236 Acc: 0.5497

Epoch 26/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0232 Acc: 0.5497

Epoch 27/87
----------
train Loss: 0.0005 Acc: 0.9841
val Loss: 0.0235 Acc: 0.5497

Epoch 28/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0225 Acc: 0.5404

Epoch 29/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0223 Acc: 0.5435

Epoch 30/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0242 Acc: 0.5466

Epoch 31/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0244 Acc: 0.5404

Epoch 32/87
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0250 Acc: 0.5435

Epoch 33/87
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9934
val Loss: 0.0233 Acc: 0.5435

Epoch 34/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0238 Acc: 0.5466

Epoch 35/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0257 Acc: 0.5435

Epoch 36/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0238 Acc: 0.5435

Epoch 37/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0243 Acc: 0.5435

Epoch 38/87
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0220 Acc: 0.5435

Epoch 39/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0238 Acc: 0.5435

Epoch 40/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0236 Acc: 0.5404

Epoch 41/87
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0235 Acc: 0.5435

Epoch 42/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0226 Acc: 0.5497

Epoch 43/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0254 Acc: 0.5435

Epoch 44/87
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0247 Acc: 0.5435

Epoch 45/87
----------
train Loss: 0.0005 Acc: 0.9881
val Loss: 0.0215 Acc: 0.5404

Epoch 46/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0226 Acc: 0.5466

Epoch 47/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0211 Acc: 0.5466

Epoch 48/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0244 Acc: 0.5435

Epoch 49/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0240 Acc: 0.5466

Epoch 50/87
----------
train Loss: 0.0003 Acc: 0.9987
val Loss: 0.0243 Acc: 0.5404

Epoch 51/87
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0227 Acc: 0.5435

Epoch 52/87
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0217 Acc: 0.5404

Epoch 53/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0225 Acc: 0.5404

Epoch 54/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0236 Acc: 0.5404

Epoch 55/87
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0221 Acc: 0.5404

Epoch 56/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0245 Acc: 0.5435

Epoch 57/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0232 Acc: 0.5466

Epoch 58/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0211 Acc: 0.5435

Epoch 59/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0240 Acc: 0.5404

Epoch 60/87
----------
train Loss: 0.0004 Acc: 0.9854
val Loss: 0.0246 Acc: 0.5373

Epoch 61/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0214 Acc: 0.5435

Epoch 62/87
----------
train Loss: 0.0004 Acc: 0.9841
val Loss: 0.0227 Acc: 0.5404

Epoch 63/87
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0244 Acc: 0.5404

Epoch 64/87
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0223 Acc: 0.5435

Epoch 65/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0218 Acc: 0.5435

Epoch 66/87
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0240 Acc: 0.5497

Epoch 67/87
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0241 Acc: 0.5466

Epoch 68/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0230 Acc: 0.5435

Epoch 69/87
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0227 Acc: 0.5435

Epoch 70/87
----------
train Loss: 0.0005 Acc: 0.9867
val Loss: 0.0232 Acc: 0.5466

Epoch 71/87
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0249 Acc: 0.5466

Epoch 72/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0238 Acc: 0.5466

Epoch 73/87
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0232 Acc: 0.5528

Epoch 74/87
----------
train Loss: 0.0004 Acc: 0.9920
val Loss: 0.0234 Acc: 0.5435

Epoch 75/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0228 Acc: 0.5435

Epoch 76/87
----------
train Loss: 0.0004 Acc: 0.9894
val Loss: 0.0222 Acc: 0.5373

Epoch 77/87
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0231 Acc: 0.5466

Epoch 78/87
----------
train Loss: 0.0003 Acc: 0.9947
val Loss: 0.0226 Acc: 0.5466

Epoch 79/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0227 Acc: 0.5466

Epoch 80/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0225 Acc: 0.5497

Epoch 81/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0238 Acc: 0.5435

Epoch 82/87
----------
train Loss: 0.0004 Acc: 0.9907
val Loss: 0.0244 Acc: 0.5404

Epoch 83/87
----------
train Loss: 0.0004 Acc: 0.9947
val Loss: 0.0231 Acc: 0.5435

Epoch 84/87
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0236 Acc: 0.5435

Epoch 85/87
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0228 Acc: 0.5435

Epoch 86/87
----------
train Loss: 0.0004 Acc: 0.9934
val Loss: 0.0246 Acc: 0.5435

Epoch 87/87
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0241 Acc: 0.5497

Training complete in 8m 35s
Best val Acc: 0.552795

---Testing---
Test accuracy: 0.860595
--------------------
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 79 %
Accuracy of Bigeye tuna : 80 %
Accuracy of Blackfin tuna : 93 %
Accuracy of Bullet tuna : 87 %
Accuracy of Frigate tuna : 75 %
Accuracy of Little tunny : 87 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 76 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Skipjack tuna : 90 %
Accuracy of Slender tuna : 42 %
Accuracy of Southern bluefin tuna : 70 %
Accuracy of Yellowfin tuna : 95 %
mean: 0.8122168810546987, std: 0.12994988226240511

Model saved in "./weights/tuna_fish_[0.91]_mean[0.86]_std[0.12].save".

Process finished with exit code 0
'''