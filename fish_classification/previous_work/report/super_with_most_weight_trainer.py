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

train_this = 'super_with_most'
test_this = 'super_with_most'


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


'''/usr/bin/python3.5 "/home/visbic/python/pytorch_playground/datasets/new fish/git-test/super_with_most_weight_trainer.py"
--------------------

run info[val: 0.1, epoch: 76, randcrop: False, decay: 11]

---Training last layer.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0117 Acc: 0.6378
val Loss: 0.0077 Acc: 0.7576

Epoch 1/75
----------
train Loss: 0.0067 Acc: 0.7774
val Loss: 0.0068 Acc: 0.7784

Epoch 2/75
----------
train Loss: 0.0057 Acc: 0.8026
val Loss: 0.0067 Acc: 0.7978

Epoch 3/75
----------
train Loss: 0.0052 Acc: 0.8262
val Loss: 0.0066 Acc: 0.7742

Epoch 4/75
----------
train Loss: 0.0049 Acc: 0.8342
val Loss: 0.0057 Acc: 0.8449

Epoch 5/75
----------
train Loss: 0.0044 Acc: 0.8543
val Loss: 0.0053 Acc: 0.8407

Epoch 6/75
----------
train Loss: 0.0046 Acc: 0.8397
val Loss: 0.0053 Acc: 0.8338

Epoch 7/75
----------
train Loss: 0.0041 Acc: 0.8611
val Loss: 0.0057 Acc: 0.8172

Epoch 8/75
----------
train Loss: 0.0041 Acc: 0.8652
val Loss: 0.0050 Acc: 0.8380

Epoch 9/75
----------
train Loss: 0.0039 Acc: 0.8709
val Loss: 0.0052 Acc: 0.8504

Epoch 10/75
----------
train Loss: 0.0038 Acc: 0.8749
val Loss: 0.0050 Acc: 0.8476

Epoch 11/75
----------
LR is set to 0.001
train Loss: 0.0034 Acc: 0.8935
val Loss: 0.0049 Acc: 0.8393

Epoch 12/75
----------
train Loss: 0.0034 Acc: 0.8908
val Loss: 0.0050 Acc: 0.8463

Epoch 13/75
----------
train Loss: 0.0033 Acc: 0.8943
val Loss: 0.0046 Acc: 0.8435

Epoch 14/75
----------
train Loss: 0.0033 Acc: 0.8928
val Loss: 0.0048 Acc: 0.8449

Epoch 15/75
----------
train Loss: 0.0033 Acc: 0.8908
val Loss: 0.0047 Acc: 0.8449

Epoch 16/75
----------
train Loss: 0.0034 Acc: 0.8934
val Loss: 0.0049 Acc: 0.8463

Epoch 17/75
----------
train Loss: 0.0033 Acc: 0.8969
val Loss: 0.0048 Acc: 0.8476

Epoch 18/75
----------
train Loss: 0.0033 Acc: 0.8935
val Loss: 0.0046 Acc: 0.8518

Epoch 19/75
----------
train Loss: 0.0034 Acc: 0.8917
val Loss: 0.0049 Acc: 0.8490

Epoch 20/75
----------
train Loss: 0.0033 Acc: 0.8952
val Loss: 0.0047 Acc: 0.8449

Epoch 21/75
----------
train Loss: 0.0033 Acc: 0.8898
val Loss: 0.0048 Acc: 0.8560

Epoch 22/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0033 Acc: 0.8957
val Loss: 0.0048 Acc: 0.8490

Epoch 23/75
----------
train Loss: 0.0033 Acc: 0.8951
val Loss: 0.0053 Acc: 0.8504

Epoch 24/75
----------
train Loss: 0.0032 Acc: 0.8942
val Loss: 0.0049 Acc: 0.8490

Epoch 25/75
----------
train Loss: 0.0033 Acc: 0.8945
val Loss: 0.0048 Acc: 0.8490

Epoch 26/75
----------
train Loss: 0.0032 Acc: 0.8954
val Loss: 0.0047 Acc: 0.8490

Epoch 27/75
----------
train Loss: 0.0032 Acc: 0.8957
val Loss: 0.0047 Acc: 0.8490

Epoch 28/75
----------
train Loss: 0.0033 Acc: 0.8965
val Loss: 0.0049 Acc: 0.8504

Epoch 29/75
----------
train Loss: 0.0033 Acc: 0.8923
val Loss: 0.0051 Acc: 0.8518

Epoch 30/75
----------
train Loss: 0.0033 Acc: 0.8934
val Loss: 0.0049 Acc: 0.8421

Epoch 31/75
----------
train Loss: 0.0033 Acc: 0.8968
val Loss: 0.0050 Acc: 0.8532

Epoch 32/75
----------
train Loss: 0.0032 Acc: 0.8942
val Loss: 0.0049 Acc: 0.8490

Epoch 33/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0033 Acc: 0.8974
val Loss: 0.0052 Acc: 0.8504

Epoch 34/75
----------
train Loss: 0.0033 Acc: 0.8980
val Loss: 0.0056 Acc: 0.8463

Epoch 35/75
----------
train Loss: 0.0032 Acc: 0.8974
val Loss: 0.0048 Acc: 0.8504

Epoch 36/75
----------
train Loss: 0.0033 Acc: 0.8962
val Loss: 0.0049 Acc: 0.8504

Epoch 37/75
----------
train Loss: 0.0032 Acc: 0.8991
val Loss: 0.0046 Acc: 0.8546

Epoch 38/75
----------
train Loss: 0.0032 Acc: 0.9003
val Loss: 0.0046 Acc: 0.8449

Epoch 39/75
----------
train Loss: 0.0033 Acc: 0.8968
val Loss: 0.0049 Acc: 0.8518

Epoch 40/75
----------
train Loss: 0.0032 Acc: 0.8972
val Loss: 0.0049 Acc: 0.8476

Epoch 41/75
----------
train Loss: 0.0032 Acc: 0.8963
val Loss: 0.0048 Acc: 0.8518

Epoch 42/75
----------
train Loss: 0.0033 Acc: 0.8932
val Loss: 0.0046 Acc: 0.8532

Epoch 43/75
----------
train Loss: 0.0033 Acc: 0.8962
val Loss: 0.0047 Acc: 0.8532

Epoch 44/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0033 Acc: 0.8991
val Loss: 0.0048 Acc: 0.8518

Epoch 45/75
----------
train Loss: 0.0033 Acc: 0.8925
val Loss: 0.0046 Acc: 0.8518

Epoch 46/75
----------
train Loss: 0.0033 Acc: 0.8932
val Loss: 0.0049 Acc: 0.8504

Epoch 47/75
----------
train Loss: 0.0032 Acc: 0.8978
val Loss: 0.0048 Acc: 0.8504

Epoch 48/75
----------
train Loss: 0.0033 Acc: 0.8978
val Loss: 0.0052 Acc: 0.8518

Epoch 49/75
----------
train Loss: 0.0032 Acc: 0.8972
val Loss: 0.0047 Acc: 0.8490

Epoch 50/75
----------
train Loss: 0.0033 Acc: 0.8945
val Loss: 0.0049 Acc: 0.8476

Epoch 51/75
----------
train Loss: 0.0032 Acc: 0.8934
val Loss: 0.0047 Acc: 0.8476

Epoch 52/75
----------
train Loss: 0.0033 Acc: 0.8932
val Loss: 0.0049 Acc: 0.8504

Epoch 53/75
----------
train Loss: 0.0033 Acc: 0.8988
val Loss: 0.0049 Acc: 0.8490

Epoch 54/75
----------
train Loss: 0.0033 Acc: 0.8949
val Loss: 0.0051 Acc: 0.8490

Epoch 55/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0033 Acc: 0.8923
val Loss: 0.0050 Acc: 0.8490

Epoch 56/75
----------
train Loss: 0.0033 Acc: 0.8935
val Loss: 0.0050 Acc: 0.8476

Epoch 57/75
----------
train Loss: 0.0033 Acc: 0.8942
val Loss: 0.0049 Acc: 0.8435

Epoch 58/75
----------
train Loss: 0.0033 Acc: 0.8948
val Loss: 0.0048 Acc: 0.8476

Epoch 59/75
----------
train Loss: 0.0032 Acc: 0.8960
val Loss: 0.0047 Acc: 0.8490

Epoch 60/75
----------
train Loss: 0.0033 Acc: 0.8968
val Loss: 0.0050 Acc: 0.8463

Epoch 61/75
----------
train Loss: 0.0032 Acc: 0.8980
val Loss: 0.0051 Acc: 0.8518

Epoch 62/75
----------
train Loss: 0.0033 Acc: 0.8958
val Loss: 0.0048 Acc: 0.8490

Epoch 63/75
----------
train Loss: 0.0033 Acc: 0.8943
val Loss: 0.0048 Acc: 0.8490

Epoch 64/75
----------
train Loss: 0.0033 Acc: 0.8922
val Loss: 0.0050 Acc: 0.8504

Epoch 65/75
----------
train Loss: 0.0033 Acc: 0.8955
val Loss: 0.0048 Acc: 0.8504

Epoch 66/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0033 Acc: 0.8972
val Loss: 0.0049 Acc: 0.8490

Epoch 67/75
----------
train Loss: 0.0033 Acc: 0.8952
val Loss: 0.0050 Acc: 0.8463

Epoch 68/75
----------
train Loss: 0.0033 Acc: 0.8958
val Loss: 0.0046 Acc: 0.8490

Epoch 69/75
----------
train Loss: 0.0032 Acc: 0.8992
val Loss: 0.0051 Acc: 0.8476

Epoch 70/75
----------
train Loss: 0.0032 Acc: 0.8983
val Loss: 0.0047 Acc: 0.8476

Epoch 71/75
----------
train Loss: 0.0033 Acc: 0.8938
val Loss: 0.0047 Acc: 0.8504

Epoch 72/75
----------
train Loss: 0.0032 Acc: 0.8954
val Loss: 0.0049 Acc: 0.8463

Epoch 73/75
----------
train Loss: 0.0033 Acc: 0.8923
val Loss: 0.0050 Acc: 0.8532

Epoch 74/75
----------
train Loss: 0.0033 Acc: 0.8931
val Loss: 0.0046 Acc: 0.8463

Epoch 75/75
----------
train Loss: 0.0033 Acc: 0.8922
val Loss: 0.0046 Acc: 0.8518

Best val Acc: 0.855956

---Fine tuning.---
Epoch 0/75
----------
LR is set to 0.01
train Loss: 0.0045 Acc: 0.8465
val Loss: 0.0056 Acc: 0.8366

Epoch 1/75
----------
train Loss: 0.0021 Acc: 0.9242
val Loss: 0.0051 Acc: 0.8795

Epoch 2/75
----------
train Loss: 0.0011 Acc: 0.9614
val Loss: 0.0038 Acc: 0.8989

Epoch 3/75
----------
train Loss: 0.0005 Acc: 0.9800
val Loss: 0.0036 Acc: 0.9197

Epoch 4/75
----------
train Loss: 0.0004 Acc: 0.9809
val Loss: 0.0033 Acc: 0.9141

Epoch 5/75
----------
train Loss: 0.0004 Acc: 0.9843
val Loss: 0.0031 Acc: 0.9238

Epoch 6/75
----------
train Loss: 0.0003 Acc: 0.9855
val Loss: 0.0032 Acc: 0.9238

Epoch 7/75
----------
train Loss: 0.0003 Acc: 0.9854
val Loss: 0.0035 Acc: 0.9211

Epoch 8/75
----------
train Loss: 0.0003 Acc: 0.9874
val Loss: 0.0033 Acc: 0.9238

Epoch 9/75
----------
train Loss: 0.0002 Acc: 0.9868
val Loss: 0.0035 Acc: 0.9252

Epoch 10/75
----------
train Loss: 0.0002 Acc: 0.9860
val Loss: 0.0035 Acc: 0.9238

Epoch 11/75
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0033 Acc: 0.9224

Epoch 12/75
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0034 Acc: 0.9252

Epoch 13/75
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0036 Acc: 0.9238

Epoch 14/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9238

Epoch 15/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9252

Epoch 16/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0032 Acc: 0.9238

Epoch 17/75
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0034 Acc: 0.9238

Epoch 18/75
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0035 Acc: 0.9238

Epoch 19/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0040 Acc: 0.9238

Epoch 20/75
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9252

Epoch 21/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9224

Epoch 22/75
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0038 Acc: 0.9280

Epoch 23/75
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0035 Acc: 0.9238

Epoch 24/75
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0035 Acc: 0.9280

Epoch 25/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0032 Acc: 0.9238

Epoch 26/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0036 Acc: 0.9238

Epoch 27/75
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0032 Acc: 0.9252

Epoch 28/75
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0032 Acc: 0.9238

Epoch 29/75
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0044 Acc: 0.9224

Epoch 30/75
----------
train Loss: 0.0002 Acc: 0.9868
val Loss: 0.0037 Acc: 0.9224

Epoch 31/75
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0035 Acc: 0.9238

Epoch 32/75
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0033 Acc: 0.9238

Epoch 33/75
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0035 Acc: 0.9238

Epoch 34/75
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0033 Acc: 0.9238

Epoch 35/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9224

Epoch 36/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0036 Acc: 0.9238

Epoch 37/75
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0041 Acc: 0.9252

Epoch 38/75
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0033 Acc: 0.9238

Epoch 39/75
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0034 Acc: 0.9224

Epoch 40/75
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0038 Acc: 0.9238

Epoch 41/75
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9238

Epoch 42/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0035 Acc: 0.9224

Epoch 43/75
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0039 Acc: 0.9238

Epoch 44/75
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0037 Acc: 0.9238

Epoch 45/75
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0039 Acc: 0.9238

Epoch 46/75
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0035 Acc: 0.9238

Epoch 47/75
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9252

Epoch 48/75
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9266

Epoch 49/75
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0034 Acc: 0.9238

Epoch 50/75
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0036 Acc: 0.9224

Epoch 51/75
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0039 Acc: 0.9238

Epoch 52/75
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0034 Acc: 0.9252

Epoch 53/75
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0037 Acc: 0.9238

Epoch 54/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0033 Acc: 0.9224

Epoch 55/75
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0032 Acc: 0.9266

Epoch 56/75
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0040 Acc: 0.9252

Epoch 57/75
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0041 Acc: 0.9238

Epoch 58/75
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0034 Acc: 0.9238

Epoch 59/75
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9238

Epoch 60/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9238

Epoch 61/75
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9252

Epoch 62/75
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0033 Acc: 0.9252

Epoch 63/75
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0035 Acc: 0.9224

Epoch 64/75
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0035 Acc: 0.9252

Epoch 65/75
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0037 Acc: 0.9238

Epoch 66/75
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0038 Acc: 0.9224

Epoch 67/75
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9252

Epoch 68/75
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0043 Acc: 0.9238

Epoch 69/75
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0032 Acc: 0.9224

Epoch 70/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0032 Acc: 0.9224

Epoch 71/75
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0036 Acc: 0.9252

Epoch 72/75
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9224

Epoch 73/75
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9238

Epoch 74/75
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0032 Acc: 0.9238

Epoch 75/75
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0033 Acc: 0.9252

Best val Acc: 0.927978

---Testing---
Test accuracy: 0.982553
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 99 %
Accuracy of Batoidea(ga_oo_lee) : 99 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 99 %
Accuracy of SHARK on boat : 100 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 49 %
Accuracy of   ray : 79 %
Accuracy of rough : 94 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9392793371504601, std: 0.13413751768151963
--------------------

run info[val: 0.2, epoch: 55, randcrop: True, decay: 5]

---Training last layer.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6315
val Loss: 0.0086 Acc: 0.7237

Epoch 1/54
----------
train Loss: 0.0076 Acc: 0.7484
val Loss: 0.0071 Acc: 0.7784

Epoch 2/54
----------
train Loss: 0.0066 Acc: 0.7740
val Loss: 0.0071 Acc: 0.7666

Epoch 3/54
----------
train Loss: 0.0061 Acc: 0.7885
val Loss: 0.0072 Acc: 0.7639

Epoch 4/54
----------
train Loss: 0.0058 Acc: 0.7989
val Loss: 0.0062 Acc: 0.7839

Epoch 5/54
----------
LR is set to 0.001
train Loss: 0.0052 Acc: 0.8240
val Loss: 0.0059 Acc: 0.7936

Epoch 6/54
----------
train Loss: 0.0051 Acc: 0.8292
val Loss: 0.0060 Acc: 0.7992

Epoch 7/54
----------
train Loss: 0.0051 Acc: 0.8247
val Loss: 0.0060 Acc: 0.7999

Epoch 8/54
----------
train Loss: 0.0050 Acc: 0.8358
val Loss: 0.0060 Acc: 0.7943

Epoch 9/54
----------
train Loss: 0.0051 Acc: 0.8204
val Loss: 0.0059 Acc: 0.7999

Epoch 10/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0049 Acc: 0.8323
val Loss: 0.0060 Acc: 0.8006

Epoch 11/54
----------
train Loss: 0.0049 Acc: 0.8354
val Loss: 0.0059 Acc: 0.7999

Epoch 12/54
----------
train Loss: 0.0050 Acc: 0.8269
val Loss: 0.0060 Acc: 0.8019

Epoch 13/54
----------
train Loss: 0.0049 Acc: 0.8384
val Loss: 0.0059 Acc: 0.8012

Epoch 14/54
----------
train Loss: 0.0049 Acc: 0.8403
val Loss: 0.0059 Acc: 0.8026

Epoch 15/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0049 Acc: 0.8396
val Loss: 0.0060 Acc: 0.8012

Epoch 16/54
----------
train Loss: 0.0049 Acc: 0.8340
val Loss: 0.0058 Acc: 0.7999

Epoch 17/54
----------
train Loss: 0.0049 Acc: 0.8385
val Loss: 0.0059 Acc: 0.8033

Epoch 18/54
----------
train Loss: 0.0049 Acc: 0.8356
val Loss: 0.0059 Acc: 0.8012

Epoch 19/54
----------
train Loss: 0.0049 Acc: 0.8378
val Loss: 0.0059 Acc: 0.8012

Epoch 20/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0049 Acc: 0.8413
val Loss: 0.0058 Acc: 0.8006

Epoch 21/54
----------
train Loss: 0.0050 Acc: 0.8347
val Loss: 0.0059 Acc: 0.8012

Epoch 22/54
----------
train Loss: 0.0049 Acc: 0.8384
val Loss: 0.0059 Acc: 0.8033

Epoch 23/54
----------
train Loss: 0.0050 Acc: 0.8323
val Loss: 0.0059 Acc: 0.8040

Epoch 24/54
----------
train Loss: 0.0049 Acc: 0.8337
val Loss: 0.0060 Acc: 0.8012

Epoch 25/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0049 Acc: 0.8364
val Loss: 0.0059 Acc: 0.8026

Epoch 26/54
----------
train Loss: 0.0049 Acc: 0.8416
val Loss: 0.0058 Acc: 0.8026

Epoch 27/54
----------
train Loss: 0.0049 Acc: 0.8425
val Loss: 0.0059 Acc: 0.8040

Epoch 28/54
----------
train Loss: 0.0049 Acc: 0.8390
val Loss: 0.0060 Acc: 0.8040

Epoch 29/54
----------
train Loss: 0.0049 Acc: 0.8392
val Loss: 0.0059 Acc: 0.8012

Epoch 30/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0049 Acc: 0.8401
val Loss: 0.0059 Acc: 0.7999

Epoch 31/54
----------
train Loss: 0.0049 Acc: 0.8411
val Loss: 0.0059 Acc: 0.8006

Epoch 32/54
----------
train Loss: 0.0049 Acc: 0.8377
val Loss: 0.0059 Acc: 0.8026

Epoch 33/54
----------
train Loss: 0.0050 Acc: 0.8399
val Loss: 0.0059 Acc: 0.8006

Epoch 34/54
----------
train Loss: 0.0049 Acc: 0.8358
val Loss: 0.0059 Acc: 0.8033

Epoch 35/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0049 Acc: 0.8311
val Loss: 0.0060 Acc: 0.8006

Epoch 36/54
----------
train Loss: 0.0049 Acc: 0.8408
val Loss: 0.0061 Acc: 0.8006

Epoch 37/54
----------
train Loss: 0.0048 Acc: 0.8453
val Loss: 0.0059 Acc: 0.8026

Epoch 38/54
----------
train Loss: 0.0049 Acc: 0.8375
val Loss: 0.0059 Acc: 0.8012

Epoch 39/54
----------
train Loss: 0.0049 Acc: 0.8389
val Loss: 0.0060 Acc: 0.8054

Epoch 40/54
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0049 Acc: 0.8373
val Loss: 0.0059 Acc: 0.8012

Epoch 41/54
----------
train Loss: 0.0048 Acc: 0.8441
val Loss: 0.0059 Acc: 0.8026

Epoch 42/54
----------
train Loss: 0.0049 Acc: 0.8416
val Loss: 0.0059 Acc: 0.8006

Epoch 43/54
----------
train Loss: 0.0049 Acc: 0.8397
val Loss: 0.0059 Acc: 0.8012

Epoch 44/54
----------
train Loss: 0.0049 Acc: 0.8366
val Loss: 0.0059 Acc: 0.8033

Epoch 45/54
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0049 Acc: 0.8378
val Loss: 0.0059 Acc: 0.8012

Epoch 46/54
----------
train Loss: 0.0049 Acc: 0.8366
val Loss: 0.0059 Acc: 0.8040

Epoch 47/54
----------
train Loss: 0.0049 Acc: 0.8318
val Loss: 0.0060 Acc: 0.8006

Epoch 48/54
----------
train Loss: 0.0048 Acc: 0.8404
val Loss: 0.0059 Acc: 0.8012

Epoch 49/54
----------
train Loss: 0.0049 Acc: 0.8368
val Loss: 0.0059 Acc: 0.8026

Epoch 50/54
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0050 Acc: 0.8371
val Loss: 0.0061 Acc: 0.8054

Epoch 51/54
----------
train Loss: 0.0050 Acc: 0.8344
val Loss: 0.0058 Acc: 0.8026

Epoch 52/54
----------
train Loss: 0.0050 Acc: 0.8396
val Loss: 0.0059 Acc: 0.8026

Epoch 53/54
----------
train Loss: 0.0049 Acc: 0.8297
val Loss: 0.0059 Acc: 0.8012

Epoch 54/54
----------
train Loss: 0.0049 Acc: 0.8337
val Loss: 0.0059 Acc: 0.8019

Best val Acc: 0.805402

---Fine tuning.---
Epoch 0/54
----------
LR is set to 0.01
train Loss: 0.0054 Acc: 0.8179
val Loss: 0.0067 Acc: 0.7950

Epoch 1/54
----------
train Loss: 0.0028 Acc: 0.9005
val Loss: 0.0040 Acc: 0.8788

Epoch 2/54
----------
train Loss: 0.0018 Acc: 0.9401
val Loss: 0.0042 Acc: 0.8698

Epoch 3/54
----------
train Loss: 0.0013 Acc: 0.9576
val Loss: 0.0036 Acc: 0.8947

Epoch 4/54
----------
train Loss: 0.0009 Acc: 0.9690
val Loss: 0.0038 Acc: 0.8975

Epoch 5/54
----------
LR is set to 0.001
train Loss: 0.0006 Acc: 0.9808
val Loss: 0.0034 Acc: 0.9134

Epoch 6/54
----------
train Loss: 0.0005 Acc: 0.9836
val Loss: 0.0035 Acc: 0.9148

Epoch 7/54
----------
train Loss: 0.0004 Acc: 0.9834
val Loss: 0.0033 Acc: 0.9176

Epoch 8/54
----------
train Loss: 0.0004 Acc: 0.9844
val Loss: 0.0032 Acc: 0.9211

Epoch 9/54
----------
train Loss: 0.0004 Acc: 0.9855
val Loss: 0.0033 Acc: 0.9169

Epoch 10/54
----------
LR is set to 0.00010000000000000002
train Loss: 0.0004 Acc: 0.9862
val Loss: 0.0032 Acc: 0.9176

Epoch 11/54
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0033 Acc: 0.9183

Epoch 12/54
----------
train Loss: 0.0003 Acc: 0.9888
val Loss: 0.0033 Acc: 0.9176

Epoch 13/54
----------
train Loss: 0.0004 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9176

Epoch 14/54
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0033 Acc: 0.9169

Epoch 15/54
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9856
val Loss: 0.0033 Acc: 0.9183

Epoch 16/54
----------
train Loss: 0.0004 Acc: 0.9865
val Loss: 0.0033 Acc: 0.9176

Epoch 17/54
----------
train Loss: 0.0003 Acc: 0.9870
val Loss: 0.0033 Acc: 0.9183

Epoch 18/54
----------
train Loss: 0.0004 Acc: 0.9851
val Loss: 0.0033 Acc: 0.9183

Epoch 19/54
----------
train Loss: 0.0003 Acc: 0.9872
val Loss: 0.0033 Acc: 0.9183

Epoch 20/54
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0004 Acc: 0.9874
val Loss: 0.0034 Acc: 0.9183

Epoch 21/54
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0033 Acc: 0.9183

Epoch 22/54
----------
train Loss: 0.0004 Acc: 0.9848
val Loss: 0.0033 Acc: 0.9176

Epoch 23/54
----------
train Loss: 0.0003 Acc: 0.9860
val Loss: 0.0032 Acc: 0.9183

Epoch 24/54
----------
train Loss: 0.0004 Acc: 0.9879
val Loss: 0.0035 Acc: 0.9176

Epoch 25/54
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9865
val Loss: 0.0034 Acc: 0.9190

Epoch 26/54
----------
train Loss: 0.0004 Acc: 0.9860
val Loss: 0.0033 Acc: 0.9176

Epoch 27/54
----------
train Loss: 0.0004 Acc: 0.9863
val Loss: 0.0034 Acc: 0.9176

Epoch 28/54
----------
train Loss: 0.0004 Acc: 0.9868
val Loss: 0.0033 Acc: 0.9169

Epoch 29/54
----------
train Loss: 0.0004 Acc: 0.9860
val Loss: 0.0033 Acc: 0.9176

Epoch 30/54
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0004 Acc: 0.9863
val Loss: 0.0033 Acc: 0.9169

Epoch 31/54
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0033 Acc: 0.9176

Epoch 32/54
----------
train Loss: 0.0004 Acc: 0.9870
val Loss: 0.0034 Acc: 0.9162

Epoch 33/54
----------
train Loss: 0.0003 Acc: 0.9865
val Loss: 0.0033 Acc: 0.9176

Epoch 34/54
----------
train Loss: 0.0004 Acc: 0.9853
val Loss: 0.0033 Acc: 0.9183

Epoch 35/54
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0004 Acc: 0.9862
val Loss: 0.0033 Acc: 0.9169

Epoch 36/54
----------
train Loss: 0.0004 Acc: 0.9862
val Loss: 0.0032 Acc: 0.9183

Epoch 37/54
----------
train Loss: 0.0004 Acc: 0.9841
val Loss: 0.0033 Acc: 0.9183

Epoch 38/54
----------
train Loss: 0.0004 Acc: 0.9863
val Loss: 0.0033 Acc: 0.9183

Epoch 39/54
----------
train Loss: 0.0004 Acc: 0.9870
val Loss: 0.0033 Acc: 0.9176

Epoch 40/54
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0033 Acc: 0.9169

Epoch 41/54
----------
train Loss: 0.0004 Acc: 0.9853
val Loss: 0.0033 Acc: 0.9190

Epoch 42/54
----------
train Loss: 0.0004 Acc: 0.9855
val Loss: 0.0035 Acc: 0.9176

Epoch 43/54
----------
train Loss: 0.0004 Acc: 0.9875
val Loss: 0.0033 Acc: 0.9169

Epoch 44/54
----------
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0033 Acc: 0.9176

Epoch 45/54
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0004 Acc: 0.9848
val Loss: 0.0032 Acc: 0.9183

Epoch 46/54
----------
train Loss: 0.0003 Acc: 0.9856
val Loss: 0.0033 Acc: 0.9176

Epoch 47/54
----------
train Loss: 0.0004 Acc: 0.9870
val Loss: 0.0033 Acc: 0.9169

Epoch 48/54
----------
train Loss: 0.0003 Acc: 0.9874
val Loss: 0.0033 Acc: 0.9183

Epoch 49/54
----------
train Loss: 0.0004 Acc: 0.9867
val Loss: 0.0034 Acc: 0.9169

Epoch 50/54
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0033 Acc: 0.9183

Epoch 51/54
----------
train Loss: 0.0003 Acc: 0.9855
val Loss: 0.0034 Acc: 0.9169

Epoch 52/54
----------
train Loss: 0.0004 Acc: 0.9870
val Loss: 0.0034 Acc: 0.9183

Epoch 53/54
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0033 Acc: 0.9169

Epoch 54/54
----------
train Loss: 0.0004 Acc: 0.9875
val Loss: 0.0032 Acc: 0.9183

Best val Acc: 0.921053

---Testing---
Test accuracy: 0.975076
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 97 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 96 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 94 %
Accuracy of mullet : 34 %
Accuracy of   ray : 82 %
Accuracy of rough : 85 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.917945927055625, std: 0.16842137732051893
--------------------

run info[val: 0.3, epoch: 46, randcrop: True, decay: 13]

---Training last layer.---
Epoch 0/45
----------
LR is set to 0.01
train Loss: 0.0130 Acc: 0.6050
val Loss: 0.0087 Acc: 0.7101

Epoch 1/45
----------
train Loss: 0.0078 Acc: 0.7373
val Loss: 0.0075 Acc: 0.7479

Epoch 2/45
----------
train Loss: 0.0067 Acc: 0.7733
val Loss: 0.0066 Acc: 0.7849

Epoch 3/45
----------
train Loss: 0.0062 Acc: 0.7864
val Loss: 0.0064 Acc: 0.7872

Epoch 4/45
----------
train Loss: 0.0058 Acc: 0.8046
val Loss: 0.0065 Acc: 0.7733

Epoch 5/45
----------
train Loss: 0.0057 Acc: 0.8058
val Loss: 0.0063 Acc: 0.7807

Epoch 6/45
----------
train Loss: 0.0052 Acc: 0.8277
val Loss: 0.0062 Acc: 0.7867

Epoch 7/45
----------
train Loss: 0.0051 Acc: 0.8232
val Loss: 0.0059 Acc: 0.8001

Epoch 8/45
----------
train Loss: 0.0050 Acc: 0.8297
val Loss: 0.0059 Acc: 0.7941

Epoch 9/45
----------
train Loss: 0.0049 Acc: 0.8339
val Loss: 0.0058 Acc: 0.7973

Epoch 10/45
----------
train Loss: 0.0047 Acc: 0.8410
val Loss: 0.0056 Acc: 0.7978

Epoch 11/45
----------
train Loss: 0.0047 Acc: 0.8374
val Loss: 0.0061 Acc: 0.7946

Epoch 12/45
----------
train Loss: 0.0048 Acc: 0.8358
val Loss: 0.0056 Acc: 0.8024

Epoch 13/45
----------
LR is set to 0.001
train Loss: 0.0041 Acc: 0.8619
val Loss: 0.0055 Acc: 0.8047

Epoch 14/45
----------
train Loss: 0.0041 Acc: 0.8631
val Loss: 0.0056 Acc: 0.8047

Epoch 15/45
----------
train Loss: 0.0040 Acc: 0.8697
val Loss: 0.0055 Acc: 0.8024

Epoch 16/45
----------
train Loss: 0.0040 Acc: 0.8671
val Loss: 0.0055 Acc: 0.8038

Epoch 17/45
----------
train Loss: 0.0041 Acc: 0.8687
val Loss: 0.0056 Acc: 0.8066

Epoch 18/45
----------
train Loss: 0.0041 Acc: 0.8635
val Loss: 0.0055 Acc: 0.8102

Epoch 19/45
----------
train Loss: 0.0041 Acc: 0.8631
val Loss: 0.0055 Acc: 0.8135

Epoch 20/45
----------
train Loss: 0.0040 Acc: 0.8667
val Loss: 0.0055 Acc: 0.8052

Epoch 21/45
----------
train Loss: 0.0041 Acc: 0.8687
val Loss: 0.0055 Acc: 0.8061

Epoch 22/45
----------
train Loss: 0.0040 Acc: 0.8655
val Loss: 0.0055 Acc: 0.8047

Epoch 23/45
----------
train Loss: 0.0040 Acc: 0.8639
val Loss: 0.0055 Acc: 0.8056

Epoch 24/45
----------
train Loss: 0.0040 Acc: 0.8718
val Loss: 0.0055 Acc: 0.8061

Epoch 25/45
----------
train Loss: 0.0040 Acc: 0.8681
val Loss: 0.0055 Acc: 0.8075

Epoch 26/45
----------
LR is set to 0.00010000000000000002
train Loss: 0.0040 Acc: 0.8726
val Loss: 0.0055 Acc: 0.8061

Epoch 27/45
----------
train Loss: 0.0040 Acc: 0.8701
val Loss: 0.0055 Acc: 0.8042

Epoch 28/45
----------
train Loss: 0.0040 Acc: 0.8659
val Loss: 0.0055 Acc: 0.8061

Epoch 29/45
----------
train Loss: 0.0040 Acc: 0.8673
val Loss: 0.0055 Acc: 0.8075

Epoch 30/45
----------
train Loss: 0.0040 Acc: 0.8703
val Loss: 0.0055 Acc: 0.8075

Epoch 31/45
----------
train Loss: 0.0040 Acc: 0.8730
val Loss: 0.0055 Acc: 0.8070

Epoch 32/45
----------
train Loss: 0.0040 Acc: 0.8703
val Loss: 0.0055 Acc: 0.8061

Epoch 33/45
----------
train Loss: 0.0039 Acc: 0.8764
val Loss: 0.0055 Acc: 0.8089

Epoch 34/45
----------
train Loss: 0.0040 Acc: 0.8617
val Loss: 0.0055 Acc: 0.8066

Epoch 35/45
----------
train Loss: 0.0039 Acc: 0.8730
val Loss: 0.0055 Acc: 0.8075

Epoch 36/45
----------
train Loss: 0.0040 Acc: 0.8697
val Loss: 0.0055 Acc: 0.8066

Epoch 37/45
----------
train Loss: 0.0039 Acc: 0.8681
val Loss: 0.0055 Acc: 0.8070

Epoch 38/45
----------
train Loss: 0.0040 Acc: 0.8655
val Loss: 0.0055 Acc: 0.8084

Epoch 39/45
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0041 Acc: 0.8612
val Loss: 0.0055 Acc: 0.8038

Epoch 40/45
----------
train Loss: 0.0040 Acc: 0.8665
val Loss: 0.0055 Acc: 0.8089

Epoch 41/45
----------
train Loss: 0.0040 Acc: 0.8661
val Loss: 0.0054 Acc: 0.8089

Epoch 42/45
----------
train Loss: 0.0040 Acc: 0.8637
val Loss: 0.0054 Acc: 0.8070

Epoch 43/45
----------
train Loss: 0.0040 Acc: 0.8667
val Loss: 0.0054 Acc: 0.8061

Epoch 44/45
----------
train Loss: 0.0039 Acc: 0.8703
val Loss: 0.0054 Acc: 0.8047

Epoch 45/45
----------
train Loss: 0.0040 Acc: 0.8716
val Loss: 0.0055 Acc: 0.8066

Best val Acc: 0.813481

---Fine tuning.---
Epoch 0/45
----------
LR is set to 0.01
train Loss: 0.0054 Acc: 0.8240
val Loss: 0.0071 Acc: 0.7715

Epoch 1/45
----------
train Loss: 0.0032 Acc: 0.8857
val Loss: 0.0050 Acc: 0.8495

Epoch 2/45
----------
train Loss: 0.0018 Acc: 0.9397
val Loss: 0.0044 Acc: 0.8652

Epoch 3/45
----------
train Loss: 0.0012 Acc: 0.9577
val Loss: 0.0043 Acc: 0.8809

Epoch 4/45
----------
train Loss: 0.0009 Acc: 0.9672
val Loss: 0.0037 Acc: 0.8943

Epoch 5/45
----------
train Loss: 0.0007 Acc: 0.9705
val Loss: 0.0040 Acc: 0.8989

Epoch 6/45
----------
train Loss: 0.0006 Acc: 0.9786
val Loss: 0.0039 Acc: 0.9012

Epoch 7/45
----------
train Loss: 0.0006 Acc: 0.9778
val Loss: 0.0038 Acc: 0.9017

Epoch 8/45
----------
train Loss: 0.0004 Acc: 0.9838
val Loss: 0.0038 Acc: 0.9086

Epoch 9/45
----------
train Loss: 0.0004 Acc: 0.9858
val Loss: 0.0039 Acc: 0.9095

Epoch 10/45
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0039 Acc: 0.9063

Epoch 11/45
----------
train Loss: 0.0003 Acc: 0.9871
val Loss: 0.0040 Acc: 0.9104

Epoch 12/45
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0040 Acc: 0.9090

Epoch 13/45
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0039 Acc: 0.9109

Epoch 14/45
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0039 Acc: 0.9109

Epoch 15/45
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0039 Acc: 0.9114

Epoch 16/45
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0039 Acc: 0.9123

Epoch 17/45
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0039 Acc: 0.9132

Epoch 18/45
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0040 Acc: 0.9137

Epoch 19/45
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0040 Acc: 0.9137

Epoch 20/45
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0040 Acc: 0.9141

Epoch 21/45
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0039 Acc: 0.9151

Epoch 22/45
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0040 Acc: 0.9137

Epoch 23/45
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0040 Acc: 0.9141

Epoch 24/45
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0040 Acc: 0.9141

Epoch 25/45
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0039 Acc: 0.9137

Epoch 26/45
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0039 Acc: 0.9146

Epoch 27/45
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0040 Acc: 0.9141

Epoch 28/45
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0041 Acc: 0.9137

Epoch 29/45
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0040 Acc: 0.9137

Epoch 30/45
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0039 Acc: 0.9141

Epoch 31/45
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0040 Acc: 0.9141

Epoch 32/45
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0040 Acc: 0.9155

Epoch 33/45
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0040 Acc: 0.9137

Epoch 34/45
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0040 Acc: 0.9146

Epoch 35/45
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0040 Acc: 0.9127

Epoch 36/45
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0040 Acc: 0.9146

Epoch 37/45
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0040 Acc: 0.9141

Epoch 38/45
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0041 Acc: 0.9132

Epoch 39/45
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0040 Acc: 0.9132

Epoch 40/45
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0040 Acc: 0.9132

Epoch 41/45
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0040 Acc: 0.9141

Epoch 42/45
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0040 Acc: 0.9146

Epoch 43/45
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0040 Acc: 0.9137

Epoch 44/45
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0040 Acc: 0.9151

Epoch 45/45
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0040 Acc: 0.9137

Best val Acc: 0.915512

---Testing---
Test accuracy: 0.968707
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 97 %
Accuracy of Batoidea(ga_oo_lee) : 96 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 95 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 91 %
Accuracy of mullet : 34 %
Accuracy of   ray : 80 %
Accuracy of rough : 82 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.909757469820211, std: 0.16888549326974886

Model saved in "./weights/super_with_most_[0.98]_mean[0.94]_std[0.13].save".
--------------------

run info[val: 0.1, epoch: 73, randcrop: False, decay: 8]

---Training last layer.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0117 Acc: 0.6409
val Loss: 0.0084 Acc: 0.7618

Epoch 1/72
----------
train Loss: 0.0067 Acc: 0.7752
val Loss: 0.0068 Acc: 0.7978

Epoch 2/72
----------
train Loss: 0.0057 Acc: 0.8063
val Loss: 0.0064 Acc: 0.8047

Epoch 3/72
----------
train Loss: 0.0051 Acc: 0.8338
val Loss: 0.0060 Acc: 0.8241

Epoch 4/72
----------
train Loss: 0.0051 Acc: 0.8268
val Loss: 0.0055 Acc: 0.8435

Epoch 5/72
----------
train Loss: 0.0046 Acc: 0.8432
val Loss: 0.0058 Acc: 0.8269

Epoch 6/72
----------
train Loss: 0.0045 Acc: 0.8488
val Loss: 0.0059 Acc: 0.8144

Epoch 7/72
----------
train Loss: 0.0043 Acc: 0.8555
val Loss: 0.0050 Acc: 0.8532

Epoch 8/72
----------
LR is set to 0.001
train Loss: 0.0038 Acc: 0.8794
val Loss: 0.0051 Acc: 0.8532

Epoch 9/72
----------
train Loss: 0.0037 Acc: 0.8814
val Loss: 0.0052 Acc: 0.8449

Epoch 10/72
----------
train Loss: 0.0037 Acc: 0.8831
val Loss: 0.0049 Acc: 0.8490

Epoch 11/72
----------
train Loss: 0.0037 Acc: 0.8812
val Loss: 0.0051 Acc: 0.8490

Epoch 12/72
----------
train Loss: 0.0037 Acc: 0.8771
val Loss: 0.0051 Acc: 0.8490

Epoch 13/72
----------
train Loss: 0.0037 Acc: 0.8826
val Loss: 0.0050 Acc: 0.8463

Epoch 14/72
----------
train Loss: 0.0037 Acc: 0.8800
val Loss: 0.0053 Acc: 0.8518

Epoch 15/72
----------
train Loss: 0.0037 Acc: 0.8871
val Loss: 0.0051 Acc: 0.8463

Epoch 16/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0036 Acc: 0.8886
val Loss: 0.0054 Acc: 0.8449

Epoch 17/72
----------
train Loss: 0.0035 Acc: 0.8897
val Loss: 0.0048 Acc: 0.8476

Epoch 18/72
----------
train Loss: 0.0036 Acc: 0.8845
val Loss: 0.0051 Acc: 0.8449

Epoch 19/72
----------
train Loss: 0.0036 Acc: 0.8843
val Loss: 0.0055 Acc: 0.8463

Epoch 20/72
----------
train Loss: 0.0035 Acc: 0.8908
val Loss: 0.0048 Acc: 0.8463

Epoch 21/72
----------
train Loss: 0.0036 Acc: 0.8838
val Loss: 0.0050 Acc: 0.8504

Epoch 22/72
----------
train Loss: 0.0036 Acc: 0.8851
val Loss: 0.0049 Acc: 0.8449

Epoch 23/72
----------
train Loss: 0.0036 Acc: 0.8849
val Loss: 0.0051 Acc: 0.8407

Epoch 24/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0036 Acc: 0.8838
val Loss: 0.0048 Acc: 0.8504

Epoch 25/72
----------
train Loss: 0.0035 Acc: 0.8871
val Loss: 0.0049 Acc: 0.8435

Epoch 26/72
----------
train Loss: 0.0036 Acc: 0.8845
val Loss: 0.0049 Acc: 0.8463

Epoch 27/72
----------
train Loss: 0.0036 Acc: 0.8851
val Loss: 0.0047 Acc: 0.8421

Epoch 28/72
----------
train Loss: 0.0036 Acc: 0.8831
val Loss: 0.0053 Acc: 0.8435

Epoch 29/72
----------
train Loss: 0.0036 Acc: 0.8846
val Loss: 0.0053 Acc: 0.8449

Epoch 30/72
----------
train Loss: 0.0035 Acc: 0.8843
val Loss: 0.0051 Acc: 0.8463

Epoch 31/72
----------
train Loss: 0.0036 Acc: 0.8845
val Loss: 0.0052 Acc: 0.8463

Epoch 32/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0036 Acc: 0.8866
val Loss: 0.0051 Acc: 0.8463

Epoch 33/72
----------
train Loss: 0.0036 Acc: 0.8868
val Loss: 0.0049 Acc: 0.8435

Epoch 34/72
----------
train Loss: 0.0036 Acc: 0.8866
val Loss: 0.0050 Acc: 0.8463

Epoch 35/72
----------
train Loss: 0.0036 Acc: 0.8858
val Loss: 0.0051 Acc: 0.8435

Epoch 36/72
----------
train Loss: 0.0036 Acc: 0.8851
val Loss: 0.0052 Acc: 0.8449

Epoch 37/72
----------
train Loss: 0.0036 Acc: 0.8877
val Loss: 0.0047 Acc: 0.8435

Epoch 38/72
----------
train Loss: 0.0036 Acc: 0.8858
val Loss: 0.0048 Acc: 0.8463

Epoch 39/72
----------
train Loss: 0.0035 Acc: 0.8882
val Loss: 0.0054 Acc: 0.8463

Epoch 40/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0036 Acc: 0.8866
val Loss: 0.0051 Acc: 0.8476

Epoch 41/72
----------
train Loss: 0.0036 Acc: 0.8846
val Loss: 0.0050 Acc: 0.8435

Epoch 42/72
----------
train Loss: 0.0035 Acc: 0.8857
val Loss: 0.0051 Acc: 0.8421

Epoch 43/72
----------
train Loss: 0.0035 Acc: 0.8860
val Loss: 0.0048 Acc: 0.8449

Epoch 44/72
----------
train Loss: 0.0035 Acc: 0.8888
val Loss: 0.0049 Acc: 0.8490

Epoch 45/72
----------
train Loss: 0.0036 Acc: 0.8814
val Loss: 0.0051 Acc: 0.8463

Epoch 46/72
----------
train Loss: 0.0036 Acc: 0.8880
val Loss: 0.0050 Acc: 0.8463

Epoch 47/72
----------
train Loss: 0.0036 Acc: 0.8835
val Loss: 0.0051 Acc: 0.8463

Epoch 48/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0036 Acc: 0.8832
val Loss: 0.0052 Acc: 0.8490

Epoch 49/72
----------
train Loss: 0.0036 Acc: 0.8857
val Loss: 0.0047 Acc: 0.8476

Epoch 50/72
----------
train Loss: 0.0036 Acc: 0.8863
val Loss: 0.0049 Acc: 0.8435

Epoch 51/72
----------
train Loss: 0.0035 Acc: 0.8872
val Loss: 0.0051 Acc: 0.8463

Epoch 52/72
----------
train Loss: 0.0036 Acc: 0.8831
val Loss: 0.0055 Acc: 0.8463

Epoch 53/72
----------
train Loss: 0.0035 Acc: 0.8895
val Loss: 0.0051 Acc: 0.8393

Epoch 54/72
----------
train Loss: 0.0035 Acc: 0.8888
val Loss: 0.0051 Acc: 0.8463

Epoch 55/72
----------
train Loss: 0.0036 Acc: 0.8848
val Loss: 0.0048 Acc: 0.8463

Epoch 56/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0036 Acc: 0.8880
val Loss: 0.0051 Acc: 0.8490

Epoch 57/72
----------
train Loss: 0.0036 Acc: 0.8843
val Loss: 0.0047 Acc: 0.8490

Epoch 58/72
----------
train Loss: 0.0035 Acc: 0.8883
val Loss: 0.0050 Acc: 0.8463

Epoch 59/72
----------
train Loss: 0.0036 Acc: 0.8872
val Loss: 0.0050 Acc: 0.8449

Epoch 60/72
----------
train Loss: 0.0036 Acc: 0.8848
val Loss: 0.0051 Acc: 0.8449

Epoch 61/72
----------
train Loss: 0.0036 Acc: 0.8831
val Loss: 0.0049 Acc: 0.8490

Epoch 62/72
----------
train Loss: 0.0036 Acc: 0.8843
val Loss: 0.0055 Acc: 0.8421

Epoch 63/72
----------
train Loss: 0.0035 Acc: 0.8883
val Loss: 0.0050 Acc: 0.8449

Epoch 64/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0036 Acc: 0.8837
val Loss: 0.0051 Acc: 0.8463

Epoch 65/72
----------
train Loss: 0.0036 Acc: 0.8894
val Loss: 0.0050 Acc: 0.8435

Epoch 66/72
----------
train Loss: 0.0036 Acc: 0.8860
val Loss: 0.0054 Acc: 0.8435

Epoch 67/72
----------
train Loss: 0.0036 Acc: 0.8834
val Loss: 0.0051 Acc: 0.8463

Epoch 68/72
----------
train Loss: 0.0036 Acc: 0.8837
val Loss: 0.0048 Acc: 0.8476

Epoch 69/72
----------
train Loss: 0.0035 Acc: 0.8848
val Loss: 0.0047 Acc: 0.8490

Epoch 70/72
----------
train Loss: 0.0035 Acc: 0.8868
val Loss: 0.0049 Acc: 0.8490

Epoch 71/72
----------
train Loss: 0.0036 Acc: 0.8860
val Loss: 0.0049 Acc: 0.8463

Epoch 72/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0036 Acc: 0.8858
val Loss: 0.0049 Acc: 0.8476

Best val Acc: 0.853186

---Fine tuning.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0046 Acc: 0.8478
val Loss: 0.0056 Acc: 0.8518

Epoch 1/72
----------
train Loss: 0.0019 Acc: 0.9351
val Loss: 0.0044 Acc: 0.9058

Epoch 2/72
----------
train Loss: 0.0010 Acc: 0.9637
val Loss: 0.0039 Acc: 0.8961

Epoch 3/72
----------
train Loss: 0.0006 Acc: 0.9785
val Loss: 0.0034 Acc: 0.9114

Epoch 4/72
----------
train Loss: 0.0004 Acc: 0.9823
val Loss: 0.0033 Acc: 0.9114

Epoch 5/72
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0036 Acc: 0.9183

Epoch 6/72
----------
train Loss: 0.0003 Acc: 0.9849
val Loss: 0.0033 Acc: 0.9155

Epoch 7/72
----------
train Loss: 0.0003 Acc: 0.9862
val Loss: 0.0035 Acc: 0.9238

Epoch 8/72
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0034 Acc: 0.9197

Epoch 9/72
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0041 Acc: 0.9211

Epoch 10/72
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0040 Acc: 0.9224

Epoch 11/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9238

Epoch 12/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9211

Epoch 13/72
----------
train Loss: 0.0002 Acc: 0.9868
val Loss: 0.0037 Acc: 0.9238

Epoch 14/72
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9224

Epoch 15/72
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0039 Acc: 0.9224

Epoch 16/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0036 Acc: 0.9224

Epoch 17/72
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0037 Acc: 0.9224

Epoch 18/72
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0037 Acc: 0.9224

Epoch 19/72
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0034 Acc: 0.9238

Epoch 20/72
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0034 Acc: 0.9238

Epoch 21/72
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0035 Acc: 0.9238

Epoch 22/72
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0035 Acc: 0.9224

Epoch 23/72
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0037 Acc: 0.9238

Epoch 24/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0037 Acc: 0.9238

Epoch 25/72
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0035 Acc: 0.9224

Epoch 26/72
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9238

Epoch 27/72
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0036 Acc: 0.9224

Epoch 28/72
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0036 Acc: 0.9224

Epoch 29/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9238

Epoch 30/72
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0043 Acc: 0.9224

Epoch 31/72
----------
train Loss: 0.0002 Acc: 0.9902
val Loss: 0.0036 Acc: 0.9224

Epoch 32/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9224

Epoch 33/72
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0042 Acc: 0.9224

Epoch 34/72
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0034 Acc: 0.9224

Epoch 35/72
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0042 Acc: 0.9224

Epoch 36/72
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9238

Epoch 37/72
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0035 Acc: 0.9238

Epoch 38/72
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0041 Acc: 0.9211

Epoch 39/72
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0039 Acc: 0.9224

Epoch 40/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9238

Epoch 41/72
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0040 Acc: 0.9224

Epoch 42/72
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9238

Epoch 43/72
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0040 Acc: 0.9224

Epoch 44/72
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0046 Acc: 0.9238

Epoch 45/72
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0043 Acc: 0.9224

Epoch 46/72
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0035 Acc: 0.9211

Epoch 47/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9211

Epoch 48/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0043 Acc: 0.9224

Epoch 49/72
----------
train Loss: 0.0002 Acc: 0.9902
val Loss: 0.0041 Acc: 0.9224

Epoch 50/72
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0040 Acc: 0.9238

Epoch 51/72
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0034 Acc: 0.9224

Epoch 52/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9238

Epoch 53/72
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0035 Acc: 0.9224

Epoch 54/72
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0034 Acc: 0.9224

Epoch 55/72
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0039 Acc: 0.9238

Epoch 56/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0042 Acc: 0.9224

Epoch 57/72
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0037 Acc: 0.9238

Epoch 58/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9224

Epoch 59/72
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0036 Acc: 0.9224

Epoch 60/72
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0046 Acc: 0.9238

Epoch 61/72
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0036 Acc: 0.9224

Epoch 62/72
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0036 Acc: 0.9224

Epoch 63/72
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9238

Epoch 64/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0038 Acc: 0.9224

Epoch 65/72
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0039 Acc: 0.9224

Epoch 66/72
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9238

Epoch 67/72
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0035 Acc: 0.9238

Epoch 68/72
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0035 Acc: 0.9224

Epoch 69/72
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0038 Acc: 0.9224

Epoch 70/72
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0035 Acc: 0.9224

Epoch 71/72
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0042 Acc: 0.9224

Epoch 72/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0037 Acc: 0.9224

Best val Acc: 0.923823

---Testing---
Test accuracy: 0.981723
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 99 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 99 %
Accuracy of SHARK on boat : 100 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 95 %
Accuracy of mullet : 76 %
Accuracy of   ray : 68 %
Accuracy of rough : 94 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9495890743421352, std: 0.0949079256742326
--------------------

run info[val: 0.2, epoch: 56, randcrop: False, decay: 8]

---Training last layer.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0125 Acc: 0.6293
val Loss: 0.0082 Acc: 0.7417

Epoch 1/55
----------
train Loss: 0.0068 Acc: 0.7715
val Loss: 0.0068 Acc: 0.7805

Epoch 2/55
----------
train Loss: 0.0060 Acc: 0.7982
val Loss: 0.0066 Acc: 0.7936

Epoch 3/55
----------
train Loss: 0.0053 Acc: 0.8219
val Loss: 0.0061 Acc: 0.7916

Epoch 4/55
----------
train Loss: 0.0049 Acc: 0.8406
val Loss: 0.0066 Acc: 0.7749

Epoch 5/55
----------
train Loss: 0.0046 Acc: 0.8449
val Loss: 0.0059 Acc: 0.8089

Epoch 6/55
----------
train Loss: 0.0046 Acc: 0.8434
val Loss: 0.0055 Acc: 0.8158

Epoch 7/55
----------
train Loss: 0.0043 Acc: 0.8517
val Loss: 0.0057 Acc: 0.8054

Epoch 8/55
----------
LR is set to 0.001
train Loss: 0.0038 Acc: 0.8757
val Loss: 0.0054 Acc: 0.8262

Epoch 9/55
----------
train Loss: 0.0037 Acc: 0.8792
val Loss: 0.0055 Acc: 0.8227

Epoch 10/55
----------
train Loss: 0.0037 Acc: 0.8808
val Loss: 0.0053 Acc: 0.8317

Epoch 11/55
----------
train Loss: 0.0036 Acc: 0.8846
val Loss: 0.0054 Acc: 0.8241

Epoch 12/55
----------
train Loss: 0.0036 Acc: 0.8825
val Loss: 0.0053 Acc: 0.8296

Epoch 13/55
----------
train Loss: 0.0037 Acc: 0.8823
val Loss: 0.0053 Acc: 0.8269

Epoch 14/55
----------
train Loss: 0.0036 Acc: 0.8827
val Loss: 0.0054 Acc: 0.8255

Epoch 15/55
----------
train Loss: 0.0037 Acc: 0.8844
val Loss: 0.0054 Acc: 0.8255

Epoch 16/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0037 Acc: 0.8842
val Loss: 0.0053 Acc: 0.8234

Epoch 17/55
----------
train Loss: 0.0036 Acc: 0.8884
val Loss: 0.0052 Acc: 0.8255

Epoch 18/55
----------
train Loss: 0.0036 Acc: 0.8889
val Loss: 0.0054 Acc: 0.8269

Epoch 19/55
----------
train Loss: 0.0036 Acc: 0.8856
val Loss: 0.0052 Acc: 0.8269

Epoch 20/55
----------
train Loss: 0.0035 Acc: 0.8879
val Loss: 0.0053 Acc: 0.8283

Epoch 21/55
----------
train Loss: 0.0036 Acc: 0.8830
val Loss: 0.0054 Acc: 0.8248

Epoch 22/55
----------
train Loss: 0.0036 Acc: 0.8849
val Loss: 0.0053 Acc: 0.8227

Epoch 23/55
----------
train Loss: 0.0036 Acc: 0.8873
val Loss: 0.0054 Acc: 0.8276

Epoch 24/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0036 Acc: 0.8840
val Loss: 0.0054 Acc: 0.8283

Epoch 25/55
----------
train Loss: 0.0036 Acc: 0.8832
val Loss: 0.0053 Acc: 0.8234

Epoch 26/55
----------
train Loss: 0.0036 Acc: 0.8840
val Loss: 0.0053 Acc: 0.8276

Epoch 27/55
----------
train Loss: 0.0036 Acc: 0.8870
val Loss: 0.0053 Acc: 0.8248

Epoch 28/55
----------
train Loss: 0.0036 Acc: 0.8846
val Loss: 0.0053 Acc: 0.8269

Epoch 29/55
----------
train Loss: 0.0036 Acc: 0.8906
val Loss: 0.0053 Acc: 0.8283

Epoch 30/55
----------
train Loss: 0.0036 Acc: 0.8849
val Loss: 0.0053 Acc: 0.8248

Epoch 31/55
----------
train Loss: 0.0036 Acc: 0.8834
val Loss: 0.0053 Acc: 0.8269

Epoch 32/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0036 Acc: 0.8889
val Loss: 0.0052 Acc: 0.8289

Epoch 33/55
----------
train Loss: 0.0036 Acc: 0.8814
val Loss: 0.0053 Acc: 0.8248

Epoch 34/55
----------
train Loss: 0.0036 Acc: 0.8887
val Loss: 0.0052 Acc: 0.8255

Epoch 35/55
----------
train Loss: 0.0036 Acc: 0.8858
val Loss: 0.0052 Acc: 0.8283

Epoch 36/55
----------
train Loss: 0.0036 Acc: 0.8889
val Loss: 0.0053 Acc: 0.8241

Epoch 37/55
----------
train Loss: 0.0036 Acc: 0.8866
val Loss: 0.0053 Acc: 0.8255

Epoch 38/55
----------
train Loss: 0.0036 Acc: 0.8846
val Loss: 0.0053 Acc: 0.8262

Epoch 39/55
----------
train Loss: 0.0036 Acc: 0.8873
val Loss: 0.0053 Acc: 0.8269

Epoch 40/55
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0036 Acc: 0.8868
val Loss: 0.0053 Acc: 0.8255

Epoch 41/55
----------
train Loss: 0.0036 Acc: 0.8832
val Loss: 0.0053 Acc: 0.8262

Epoch 42/55
----------
train Loss: 0.0036 Acc: 0.8918
val Loss: 0.0052 Acc: 0.8283

Epoch 43/55
----------
train Loss: 0.0036 Acc: 0.8887
val Loss: 0.0054 Acc: 0.8262

Epoch 44/55
----------
train Loss: 0.0036 Acc: 0.8856
val Loss: 0.0053 Acc: 0.8255

Epoch 45/55
----------
train Loss: 0.0036 Acc: 0.8880
val Loss: 0.0052 Acc: 0.8262

Epoch 46/55
----------
train Loss: 0.0035 Acc: 0.8832
val Loss: 0.0053 Acc: 0.8276

Epoch 47/55
----------
train Loss: 0.0036 Acc: 0.8842
val Loss: 0.0053 Acc: 0.8227

Epoch 48/55
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0036 Acc: 0.8894
val Loss: 0.0052 Acc: 0.8276

Epoch 49/55
----------
train Loss: 0.0036 Acc: 0.8885
val Loss: 0.0054 Acc: 0.8269

Epoch 50/55
----------
train Loss: 0.0036 Acc: 0.8872
val Loss: 0.0053 Acc: 0.8241

Epoch 51/55
----------
train Loss: 0.0036 Acc: 0.8861
val Loss: 0.0053 Acc: 0.8289

Epoch 52/55
----------
train Loss: 0.0035 Acc: 0.8906
val Loss: 0.0054 Acc: 0.8241

Epoch 53/55
----------
train Loss: 0.0036 Acc: 0.8885
val Loss: 0.0053 Acc: 0.8255

Epoch 54/55
----------
train Loss: 0.0036 Acc: 0.8870
val Loss: 0.0052 Acc: 0.8213

Epoch 55/55
----------
train Loss: 0.0036 Acc: 0.8884
val Loss: 0.0052 Acc: 0.8248

Best val Acc: 0.831717

---Fine tuning.---
Epoch 0/55
----------
LR is set to 0.01
train Loss: 0.0044 Acc: 0.8510
val Loss: 0.0051 Acc: 0.8518

Epoch 1/55
----------
train Loss: 0.0019 Acc: 0.9370
val Loss: 0.0043 Acc: 0.8740

Epoch 2/55
----------
train Loss: 0.0010 Acc: 0.9697
val Loss: 0.0036 Acc: 0.9037

Epoch 3/55
----------
train Loss: 0.0006 Acc: 0.9775
val Loss: 0.0034 Acc: 0.9058

Epoch 4/55
----------
train Loss: 0.0004 Acc: 0.9844
val Loss: 0.0033 Acc: 0.9148

Epoch 5/55
----------
train Loss: 0.0003 Acc: 0.9868
val Loss: 0.0035 Acc: 0.9107

Epoch 6/55
----------
train Loss: 0.0003 Acc: 0.9858
val Loss: 0.0036 Acc: 0.9127

Epoch 7/55
----------
train Loss: 0.0002 Acc: 0.9879
val Loss: 0.0036 Acc: 0.9134

Epoch 8/55
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9127

Epoch 9/55
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0035 Acc: 0.9141

Epoch 10/55
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0034 Acc: 0.9148

Epoch 11/55
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0035 Acc: 0.9141

Epoch 12/55
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0036 Acc: 0.9148

Epoch 13/55
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0036 Acc: 0.9134

Epoch 14/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9134

Epoch 15/55
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0035 Acc: 0.9141

Epoch 16/55
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9134

Epoch 17/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9148

Epoch 18/55
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0037 Acc: 0.9141

Epoch 19/55
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0038 Acc: 0.9141

Epoch 20/55
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9155

Epoch 21/55
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0035 Acc: 0.9162

Epoch 22/55
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0036 Acc: 0.9148

Epoch 23/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0035 Acc: 0.9141

Epoch 24/55
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0035 Acc: 0.9141

Epoch 25/55
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0036 Acc: 0.9148

Epoch 26/55
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0036 Acc: 0.9141

Epoch 27/55
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0035 Acc: 0.9141

Epoch 28/55
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0036 Acc: 0.9148

Epoch 29/55
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0036 Acc: 0.9148

Epoch 30/55
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0036 Acc: 0.9148

Epoch 31/55
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0036 Acc: 0.9148

Epoch 32/55
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0035 Acc: 0.9148

Epoch 33/55
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0036 Acc: 0.9155

Epoch 34/55
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0036 Acc: 0.9148

Epoch 35/55
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0036 Acc: 0.9141

Epoch 36/55
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9148

Epoch 37/55
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0036 Acc: 0.9134

Epoch 38/55
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0036 Acc: 0.9141

Epoch 39/55
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0038 Acc: 0.9141

Epoch 40/55
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0036 Acc: 0.9155

Epoch 41/55
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0037 Acc: 0.9155

Epoch 42/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9141

Epoch 43/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9162

Epoch 44/55
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0036 Acc: 0.9141

Epoch 45/55
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0035 Acc: 0.9141

Epoch 46/55
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0035 Acc: 0.9148

Epoch 47/55
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9148

Epoch 48/55
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0037 Acc: 0.9148

Epoch 49/55
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0036 Acc: 0.9134

Epoch 50/55
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9148

Epoch 51/55
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0035 Acc: 0.9148

Epoch 52/55
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0036 Acc: 0.9148

Epoch 53/55
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0035 Acc: 0.9141

Epoch 54/55
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0036 Acc: 0.9155

Epoch 55/55
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0035 Acc: 0.9141

Best val Acc: 0.916205

---Testing---
Test accuracy: 0.975492
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 98 %
Accuracy of Batoidea(ga_oo_lee) : 97 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 92 %
Accuracy of mullet : 38 %
Accuracy of   ray : 82 %
Accuracy of rough : 84 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9200031903241559, std: 0.15760126059583188
--------------------

run info[val: 0.3, epoch: 73, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0131 Acc: 0.6078
val Loss: 0.0088 Acc: 0.7184

Epoch 1/72
----------
train Loss: 0.0071 Acc: 0.7644
val Loss: 0.0068 Acc: 0.7839

Epoch 2/72
----------
train Loss: 0.0059 Acc: 0.7987
val Loss: 0.0067 Acc: 0.7581

Epoch 3/72
----------
LR is set to 0.001
train Loss: 0.0052 Acc: 0.8356
val Loss: 0.0062 Acc: 0.7955

Epoch 4/72
----------
train Loss: 0.0051 Acc: 0.8386
val Loss: 0.0062 Acc: 0.7895

Epoch 5/72
----------
train Loss: 0.0050 Acc: 0.8422
val Loss: 0.0061 Acc: 0.7969

Epoch 6/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0049 Acc: 0.8412
val Loss: 0.0061 Acc: 0.7982

Epoch 7/72
----------
train Loss: 0.0049 Acc: 0.8430
val Loss: 0.0061 Acc: 0.8038

Epoch 8/72
----------
train Loss: 0.0049 Acc: 0.8414
val Loss: 0.0061 Acc: 0.8015

Epoch 9/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0049 Acc: 0.8426
val Loss: 0.0061 Acc: 0.8019

Epoch 10/72
----------
train Loss: 0.0049 Acc: 0.8438
val Loss: 0.0061 Acc: 0.7992

Epoch 11/72
----------
train Loss: 0.0049 Acc: 0.8451
val Loss: 0.0061 Acc: 0.8024

Epoch 12/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0049 Acc: 0.8404
val Loss: 0.0061 Acc: 0.8019

Epoch 13/72
----------
train Loss: 0.0049 Acc: 0.8436
val Loss: 0.0062 Acc: 0.7982

Epoch 14/72
----------
train Loss: 0.0049 Acc: 0.8471
val Loss: 0.0061 Acc: 0.8006

Epoch 15/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0049 Acc: 0.8483
val Loss: 0.0061 Acc: 0.8006

Epoch 16/72
----------
train Loss: 0.0049 Acc: 0.8469
val Loss: 0.0061 Acc: 0.8024

Epoch 17/72
----------
train Loss: 0.0049 Acc: 0.8412
val Loss: 0.0062 Acc: 0.7964

Epoch 18/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0049 Acc: 0.8436
val Loss: 0.0061 Acc: 0.8029

Epoch 19/72
----------
train Loss: 0.0049 Acc: 0.8461
val Loss: 0.0062 Acc: 0.8033

Epoch 20/72
----------
train Loss: 0.0049 Acc: 0.8473
val Loss: 0.0061 Acc: 0.8015

Epoch 21/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0049 Acc: 0.8453
val Loss: 0.0061 Acc: 0.7973

Epoch 22/72
----------
train Loss: 0.0049 Acc: 0.8400
val Loss: 0.0061 Acc: 0.8019

Epoch 23/72
----------
train Loss: 0.0049 Acc: 0.8404
val Loss: 0.0061 Acc: 0.7964

Epoch 24/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0049 Acc: 0.8445
val Loss: 0.0061 Acc: 0.8015

Epoch 25/72
----------
train Loss: 0.0049 Acc: 0.8481
val Loss: 0.0061 Acc: 0.8038

Epoch 26/72
----------
train Loss: 0.0049 Acc: 0.8439
val Loss: 0.0061 Acc: 0.7973

Epoch 27/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0049 Acc: 0.8453
val Loss: 0.0061 Acc: 0.8024

Epoch 28/72
----------
train Loss: 0.0049 Acc: 0.8457
val Loss: 0.0061 Acc: 0.8042

Epoch 29/72
----------
train Loss: 0.0049 Acc: 0.8461
val Loss: 0.0061 Acc: 0.8015

Epoch 30/72
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0049 Acc: 0.8465
val Loss: 0.0061 Acc: 0.8029

Epoch 31/72
----------
train Loss: 0.0048 Acc: 0.8495
val Loss: 0.0061 Acc: 0.8019

Epoch 32/72
----------
train Loss: 0.0049 Acc: 0.8451
val Loss: 0.0061 Acc: 0.8001

Epoch 33/72
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0049 Acc: 0.8424
val Loss: 0.0061 Acc: 0.8019

Epoch 34/72
----------
train Loss: 0.0049 Acc: 0.8438
val Loss: 0.0061 Acc: 0.8019

Epoch 35/72
----------
train Loss: 0.0049 Acc: 0.8445
val Loss: 0.0062 Acc: 0.8029

Epoch 36/72
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0049 Acc: 0.8499
val Loss: 0.0061 Acc: 0.8006

Epoch 37/72
----------
train Loss: 0.0049 Acc: 0.8469
val Loss: 0.0061 Acc: 0.8010

Epoch 38/72
----------
train Loss: 0.0048 Acc: 0.8489
val Loss: 0.0061 Acc: 0.8001

Epoch 39/72
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0049 Acc: 0.8447
val Loss: 0.0061 Acc: 0.8029

Epoch 40/72
----------
train Loss: 0.0049 Acc: 0.8451
val Loss: 0.0061 Acc: 0.8029

Epoch 41/72
----------
train Loss: 0.0049 Acc: 0.8507
val Loss: 0.0062 Acc: 0.8047

Epoch 42/72
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0049 Acc: 0.8457
val Loss: 0.0061 Acc: 0.8010

Epoch 43/72
----------
train Loss: 0.0049 Acc: 0.8511
val Loss: 0.0061 Acc: 0.7982

Epoch 44/72
----------
train Loss: 0.0049 Acc: 0.8461
val Loss: 0.0061 Acc: 0.8010

Epoch 45/72
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0049 Acc: 0.8449
val Loss: 0.0061 Acc: 0.8001

Epoch 46/72
----------
train Loss: 0.0049 Acc: 0.8511
val Loss: 0.0062 Acc: 0.8006

Epoch 47/72
----------
train Loss: 0.0049 Acc: 0.8436
val Loss: 0.0061 Acc: 0.8015

Epoch 48/72
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0049 Acc: 0.8487
val Loss: 0.0061 Acc: 0.8038

Epoch 49/72
----------
train Loss: 0.0048 Acc: 0.8509
val Loss: 0.0061 Acc: 0.8029

Epoch 50/72
----------
train Loss: 0.0049 Acc: 0.8459
val Loss: 0.0061 Acc: 0.7992

Epoch 51/72
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0049 Acc: 0.8424
val Loss: 0.0061 Acc: 0.8042

Epoch 52/72
----------
train Loss: 0.0049 Acc: 0.8473
val Loss: 0.0061 Acc: 0.7992

Epoch 53/72
----------
train Loss: 0.0049 Acc: 0.8509
val Loss: 0.0061 Acc: 0.8029

Epoch 54/72
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0049 Acc: 0.8457
val Loss: 0.0061 Acc: 0.8015

Epoch 55/72
----------
train Loss: 0.0049 Acc: 0.8505
val Loss: 0.0062 Acc: 0.7992

Epoch 56/72
----------
train Loss: 0.0049 Acc: 0.8436
val Loss: 0.0061 Acc: 0.8001

Epoch 57/72
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0049 Acc: 0.8449
val Loss: 0.0061 Acc: 0.8010

Epoch 58/72
----------
train Loss: 0.0049 Acc: 0.8436
val Loss: 0.0061 Acc: 0.8006

Epoch 59/72
----------
train Loss: 0.0049 Acc: 0.8489
val Loss: 0.0061 Acc: 0.8006

Epoch 60/72
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0049 Acc: 0.8432
val Loss: 0.0061 Acc: 0.8019

Epoch 61/72
----------
train Loss: 0.0049 Acc: 0.8534
val Loss: 0.0061 Acc: 0.8019

Epoch 62/72
----------
train Loss: 0.0049 Acc: 0.8467
val Loss: 0.0061 Acc: 0.8001

Epoch 63/72
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0049 Acc: 0.8459
val Loss: 0.0061 Acc: 0.8015

Epoch 64/72
----------
train Loss: 0.0048 Acc: 0.8519
val Loss: 0.0061 Acc: 0.8015

Epoch 65/72
----------
train Loss: 0.0049 Acc: 0.8410
val Loss: 0.0061 Acc: 0.8015

Epoch 66/72
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0049 Acc: 0.8471
val Loss: 0.0061 Acc: 0.7978

Epoch 67/72
----------
train Loss: 0.0049 Acc: 0.8426
val Loss: 0.0061 Acc: 0.8006

Epoch 68/72
----------
train Loss: 0.0049 Acc: 0.8426
val Loss: 0.0062 Acc: 0.8029

Epoch 69/72
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0049 Acc: 0.8519
val Loss: 0.0061 Acc: 0.7950

Epoch 70/72
----------
train Loss: 0.0049 Acc: 0.8461
val Loss: 0.0061 Acc: 0.7978

Epoch 71/72
----------
train Loss: 0.0049 Acc: 0.8459
val Loss: 0.0062 Acc: 0.7969

Epoch 72/72
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0049 Acc: 0.8453
val Loss: 0.0061 Acc: 0.8015

Best val Acc: 0.804709

---Fine tuning.---
Epoch 0/72
----------
LR is set to 0.01
train Loss: 0.0047 Acc: 0.8430
val Loss: 0.0058 Acc: 0.8061

Epoch 1/72
----------
train Loss: 0.0020 Acc: 0.9298
val Loss: 0.0039 Acc: 0.8753

Epoch 2/72
----------
train Loss: 0.0011 Acc: 0.9660
val Loss: 0.0045 Acc: 0.8698

Epoch 3/72
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9832
val Loss: 0.0033 Acc: 0.8966

Epoch 4/72
----------
train Loss: 0.0004 Acc: 0.9875
val Loss: 0.0032 Acc: 0.8980

Epoch 5/72
----------
train Loss: 0.0004 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9007

Epoch 6/72
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9003

Epoch 7/72
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0032 Acc: 0.9007

Epoch 8/72
----------
train Loss: 0.0004 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9021

Epoch 9/72
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0032 Acc: 0.9040

Epoch 10/72
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9030

Epoch 11/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9026

Epoch 12/72
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9877
val Loss: 0.0032 Acc: 0.9044

Epoch 13/72
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9021

Epoch 14/72
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9003

Epoch 15/72
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9003

Epoch 16/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0033 Acc: 0.8984

Epoch 17/72
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0032 Acc: 0.8994

Epoch 18/72
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0032 Acc: 0.9017

Epoch 19/72
----------
train Loss: 0.0003 Acc: 0.9887
val Loss: 0.0033 Acc: 0.9007

Epoch 20/72
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0033 Acc: 0.9003

Epoch 21/72
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9030

Epoch 22/72
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0032 Acc: 0.9044

Epoch 23/72
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0032 Acc: 0.8989

Epoch 24/72
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9017

Epoch 25/72
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0032 Acc: 0.9012

Epoch 26/72
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9049

Epoch 27/72
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0032 Acc: 0.9007

Epoch 28/72
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0032 Acc: 0.9012

Epoch 29/72
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0032 Acc: 0.9040

Epoch 30/72
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0032 Acc: 0.9035

Epoch 31/72
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9026

Epoch 32/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9021

Epoch 33/72
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9021

Epoch 34/72
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0032 Acc: 0.9030

Epoch 35/72
----------
train Loss: 0.0003 Acc: 0.9883
val Loss: 0.0033 Acc: 0.9021

Epoch 36/72
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0032 Acc: 0.9026

Epoch 37/72
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0033 Acc: 0.9021

Epoch 38/72
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0032 Acc: 0.9026

Epoch 39/72
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0032 Acc: 0.9012

Epoch 40/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9012

Epoch 41/72
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9026

Epoch 42/72
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0032 Acc: 0.9030

Epoch 43/72
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0032 Acc: 0.9017

Epoch 44/72
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0032 Acc: 0.9021

Epoch 45/72
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0004 Acc: 0.9881
val Loss: 0.0033 Acc: 0.8994

Epoch 46/72
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0033 Acc: 0.8994

Epoch 47/72
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.8998

Epoch 48/72
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0032 Acc: 0.9040

Epoch 49/72
----------
train Loss: 0.0004 Acc: 0.9887
val Loss: 0.0033 Acc: 0.9003

Epoch 50/72
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0032 Acc: 0.9040

Epoch 51/72
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0032 Acc: 0.9026

Epoch 52/72
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0032 Acc: 0.9026

Epoch 53/72
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0033 Acc: 0.9021

Epoch 54/72
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0033 Acc: 0.9021

Epoch 55/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0033 Acc: 0.9026

Epoch 56/72
----------
train Loss: 0.0003 Acc: 0.9889
val Loss: 0.0032 Acc: 0.9007

Epoch 57/72
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0032 Acc: 0.9007

Epoch 58/72
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0033 Acc: 0.9021

Epoch 59/72
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0032 Acc: 0.9021

Epoch 60/72
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0032 Acc: 0.8998

Epoch 61/72
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9035

Epoch 62/72
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0033 Acc: 0.9012

Epoch 63/72
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9021

Epoch 64/72
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0032 Acc: 0.9007

Epoch 65/72
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0032 Acc: 0.9003

Epoch 66/72
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0033 Acc: 0.8998

Epoch 67/72
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0032 Acc: 0.9007

Epoch 68/72
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0032 Acc: 0.8980

Epoch 69/72
----------
LR is set to 1.0000000000000013e-25
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0033 Acc: 0.9021

Epoch 70/72
----------
train Loss: 0.0004 Acc: 0.9899
val Loss: 0.0032 Acc: 0.9017

Epoch 71/72
----------
train Loss: 0.0004 Acc: 0.9891
val Loss: 0.0032 Acc: 0.9049

Epoch 72/72
----------
LR is set to 1.0000000000000015e-26
train Loss: 0.0004 Acc: 0.9885
val Loss: 0.0032 Acc: 0.9017

Best val Acc: 0.904894

---Testing---
Test accuracy: 0.964968
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 98 %
Accuracy of Batoidea(ga_oo_lee) : 95 %
Accuracy of   DOL : 99 %
Accuracy of   LAG : 97 %
Accuracy of   NoF : 94 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 89 %
Accuracy of mullet : 34 %
Accuracy of   ray : 79 %
Accuracy of rough : 82 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9039652218796091, std: 0.1677842403299422

Model saved in "./weights/super_with_most_[0.98]_mean[0.95]_std[0.09].save".
--------------------

run info[val: 0.1, epoch: 53, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0125 Acc: 0.6169
val Loss: 0.0081 Acc: 0.7687

Epoch 1/52
----------
train Loss: 0.0075 Acc: 0.7415
val Loss: 0.0076 Acc: 0.7770

Epoch 2/52
----------
train Loss: 0.0066 Acc: 0.7737
val Loss: 0.0065 Acc: 0.7950

Epoch 3/52
----------
train Loss: 0.0060 Acc: 0.7931
val Loss: 0.0061 Acc: 0.7978

Epoch 4/52
----------
train Loss: 0.0056 Acc: 0.8086
val Loss: 0.0064 Acc: 0.8047

Epoch 5/52
----------
train Loss: 0.0054 Acc: 0.8140
val Loss: 0.0057 Acc: 0.8338

Epoch 6/52
----------
train Loss: 0.0052 Acc: 0.8206
val Loss: 0.0058 Acc: 0.8158

Epoch 7/52
----------
train Loss: 0.0048 Acc: 0.8345
val Loss: 0.0061 Acc: 0.8241

Epoch 8/52
----------
LR is set to 0.001
train Loss: 0.0046 Acc: 0.8400
val Loss: 0.0052 Acc: 0.8393

Epoch 9/52
----------
train Loss: 0.0046 Acc: 0.8451
val Loss: 0.0055 Acc: 0.8449

Epoch 10/52
----------
train Loss: 0.0046 Acc: 0.8400
val Loss: 0.0063 Acc: 0.8393

Epoch 11/52
----------
train Loss: 0.0045 Acc: 0.8498
val Loss: 0.0052 Acc: 0.8421

Epoch 12/52
----------
train Loss: 0.0044 Acc: 0.8531
val Loss: 0.0055 Acc: 0.8407

Epoch 13/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0056 Acc: 0.8407

Epoch 14/52
----------
train Loss: 0.0045 Acc: 0.8506
val Loss: 0.0052 Acc: 0.8407

Epoch 15/52
----------
train Loss: 0.0045 Acc: 0.8511
val Loss: 0.0063 Acc: 0.8393

Epoch 16/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0045 Acc: 0.8505
val Loss: 0.0055 Acc: 0.8421

Epoch 17/52
----------
train Loss: 0.0044 Acc: 0.8535
val Loss: 0.0059 Acc: 0.8490

Epoch 18/52
----------
train Loss: 0.0044 Acc: 0.8591
val Loss: 0.0050 Acc: 0.8366

Epoch 19/52
----------
train Loss: 0.0044 Acc: 0.8486
val Loss: 0.0053 Acc: 0.8435

Epoch 20/52
----------
train Loss: 0.0045 Acc: 0.8480
val Loss: 0.0055 Acc: 0.8476

Epoch 21/52
----------
train Loss: 0.0044 Acc: 0.8528
val Loss: 0.0056 Acc: 0.8421

Epoch 22/52
----------
train Loss: 0.0044 Acc: 0.8542
val Loss: 0.0053 Acc: 0.8476

Epoch 23/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0055 Acc: 0.8463

Epoch 24/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0044 Acc: 0.8505
val Loss: 0.0053 Acc: 0.8380

Epoch 25/52
----------
train Loss: 0.0044 Acc: 0.8460
val Loss: 0.0054 Acc: 0.8449

Epoch 26/52
----------
train Loss: 0.0045 Acc: 0.8503
val Loss: 0.0055 Acc: 0.8421

Epoch 27/52
----------
train Loss: 0.0044 Acc: 0.8555
val Loss: 0.0056 Acc: 0.8463

Epoch 28/52
----------
train Loss: 0.0045 Acc: 0.8458
val Loss: 0.0053 Acc: 0.8435

Epoch 29/52
----------
train Loss: 0.0044 Acc: 0.8532
val Loss: 0.0053 Acc: 0.8476

Epoch 30/52
----------
train Loss: 0.0044 Acc: 0.8502
val Loss: 0.0053 Acc: 0.8476

Epoch 31/52
----------
train Loss: 0.0044 Acc: 0.8489
val Loss: 0.0051 Acc: 0.8435

Epoch 32/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0044 Acc: 0.8498
val Loss: 0.0053 Acc: 0.8435

Epoch 33/52
----------
train Loss: 0.0045 Acc: 0.8482
val Loss: 0.0051 Acc: 0.8490

Epoch 34/52
----------
train Loss: 0.0044 Acc: 0.8498
val Loss: 0.0057 Acc: 0.8476

Epoch 35/52
----------
train Loss: 0.0045 Acc: 0.8482
val Loss: 0.0052 Acc: 0.8435

Epoch 36/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0053 Acc: 0.8449

Epoch 37/52
----------
train Loss: 0.0045 Acc: 0.8463
val Loss: 0.0054 Acc: 0.8449

Epoch 38/52
----------
train Loss: 0.0044 Acc: 0.8543
val Loss: 0.0058 Acc: 0.8421

Epoch 39/52
----------
train Loss: 0.0043 Acc: 0.8529
val Loss: 0.0053 Acc: 0.8449

Epoch 40/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0044 Acc: 0.8531
val Loss: 0.0052 Acc: 0.8476

Epoch 41/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0057 Acc: 0.8435

Epoch 42/52
----------
train Loss: 0.0044 Acc: 0.8512
val Loss: 0.0051 Acc: 0.8463

Epoch 43/52
----------
train Loss: 0.0044 Acc: 0.8545
val Loss: 0.0055 Acc: 0.8463

Epoch 44/52
----------
train Loss: 0.0044 Acc: 0.8502
val Loss: 0.0055 Acc: 0.8490

Epoch 45/52
----------
train Loss: 0.0043 Acc: 0.8571
val Loss: 0.0055 Acc: 0.8476

Epoch 46/52
----------
train Loss: 0.0044 Acc: 0.8506
val Loss: 0.0053 Acc: 0.8490

Epoch 47/52
----------
train Loss: 0.0045 Acc: 0.8455
val Loss: 0.0053 Acc: 0.8449

Epoch 48/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0045 Acc: 0.8491
val Loss: 0.0053 Acc: 0.8421

Epoch 49/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0051 Acc: 0.8449

Epoch 50/52
----------
train Loss: 0.0044 Acc: 0.8483
val Loss: 0.0051 Acc: 0.8449

Epoch 51/52
----------
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0053 Acc: 0.8449

Epoch 52/52
----------
train Loss: 0.0044 Acc: 0.8554
val Loss: 0.0054 Acc: 0.8463

Best val Acc: 0.849030

---Fine tuning.---
Epoch 0/52
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8129
val Loss: 0.0119 Acc: 0.6939

Epoch 1/52
----------
train Loss: 0.0030 Acc: 0.8991
val Loss: 0.0043 Acc: 0.8809

Epoch 2/52
----------
train Loss: 0.0016 Acc: 0.9431
val Loss: 0.0036 Acc: 0.9030

Epoch 3/52
----------
train Loss: 0.0010 Acc: 0.9637
val Loss: 0.0036 Acc: 0.9003

Epoch 4/52
----------
train Loss: 0.0008 Acc: 0.9705
val Loss: 0.0035 Acc: 0.9155

Epoch 5/52
----------
train Loss: 0.0007 Acc: 0.9751
val Loss: 0.0035 Acc: 0.9127

Epoch 6/52
----------
train Loss: 0.0005 Acc: 0.9791
val Loss: 0.0034 Acc: 0.9169

Epoch 7/52
----------
train Loss: 0.0005 Acc: 0.9809
val Loss: 0.0039 Acc: 0.9127

Epoch 8/52
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9842
val Loss: 0.0043 Acc: 0.9169

Epoch 9/52
----------
train Loss: 0.0003 Acc: 0.9848
val Loss: 0.0034 Acc: 0.9238

Epoch 10/52
----------
train Loss: 0.0003 Acc: 0.9877
val Loss: 0.0035 Acc: 0.9197

Epoch 11/52
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0034 Acc: 0.9197

Epoch 12/52
----------
train Loss: 0.0003 Acc: 0.9862
val Loss: 0.0042 Acc: 0.9197

Epoch 13/52
----------
train Loss: 0.0003 Acc: 0.9863
val Loss: 0.0034 Acc: 0.9211

Epoch 14/52
----------
train Loss: 0.0003 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9183

Epoch 15/52
----------
train Loss: 0.0003 Acc: 0.9855
val Loss: 0.0039 Acc: 0.9197

Epoch 16/52
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9882
val Loss: 0.0036 Acc: 0.9211

Epoch 17/52
----------
train Loss: 0.0003 Acc: 0.9854
val Loss: 0.0037 Acc: 0.9211

Epoch 18/52
----------
train Loss: 0.0003 Acc: 0.9863
val Loss: 0.0037 Acc: 0.9197

Epoch 19/52
----------
train Loss: 0.0003 Acc: 0.9883
val Loss: 0.0036 Acc: 0.9197

Epoch 20/52
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0034 Acc: 0.9211

Epoch 21/52
----------
train Loss: 0.0003 Acc: 0.9863
val Loss: 0.0037 Acc: 0.9211

Epoch 22/52
----------
train Loss: 0.0003 Acc: 0.9866
val Loss: 0.0044 Acc: 0.9211

Epoch 23/52
----------
train Loss: 0.0003 Acc: 0.9882
val Loss: 0.0035 Acc: 0.9211

Epoch 24/52
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9865
val Loss: 0.0036 Acc: 0.9211

Epoch 25/52
----------
train Loss: 0.0003 Acc: 0.9878
val Loss: 0.0035 Acc: 0.9211

Epoch 26/52
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0038 Acc: 0.9211

Epoch 27/52
----------
train Loss: 0.0003 Acc: 0.9883
val Loss: 0.0034 Acc: 0.9197

Epoch 28/52
----------
train Loss: 0.0003 Acc: 0.9871
val Loss: 0.0034 Acc: 0.9211

Epoch 29/52
----------
train Loss: 0.0003 Acc: 0.9874
val Loss: 0.0038 Acc: 0.9211

Epoch 30/52
----------
train Loss: 0.0003 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9197

Epoch 31/52
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9211

Epoch 32/52
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0038 Acc: 0.9211

Epoch 33/52
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0034 Acc: 0.9197

Epoch 34/52
----------
train Loss: 0.0003 Acc: 0.9860
val Loss: 0.0037 Acc: 0.9211

Epoch 35/52
----------
train Loss: 0.0003 Acc: 0.9868
val Loss: 0.0034 Acc: 0.9211

Epoch 36/52
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9211

Epoch 37/52
----------
train Loss: 0.0003 Acc: 0.9878
val Loss: 0.0039 Acc: 0.9197

Epoch 38/52
----------
train Loss: 0.0003 Acc: 0.9863
val Loss: 0.0035 Acc: 0.9211

Epoch 39/52
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0035 Acc: 0.9197

Epoch 40/52
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9882
val Loss: 0.0037 Acc: 0.9197

Epoch 41/52
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0034 Acc: 0.9211

Epoch 42/52
----------
train Loss: 0.0003 Acc: 0.9855
val Loss: 0.0039 Acc: 0.9211

Epoch 43/52
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0035 Acc: 0.9224

Epoch 44/52
----------
train Loss: 0.0003 Acc: 0.9880
val Loss: 0.0034 Acc: 0.9197

Epoch 45/52
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0034 Acc: 0.9224

Epoch 46/52
----------
train Loss: 0.0002 Acc: 0.9866
val Loss: 0.0034 Acc: 0.9211

Epoch 47/52
----------
train Loss: 0.0003 Acc: 0.9868
val Loss: 0.0039 Acc: 0.9211

Epoch 48/52
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9880
val Loss: 0.0041 Acc: 0.9197

Epoch 49/52
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0038 Acc: 0.9197

Epoch 50/52
----------
train Loss: 0.0003 Acc: 0.9860
val Loss: 0.0034 Acc: 0.9197

Epoch 51/52
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0034 Acc: 0.9183

Epoch 52/52
----------
train Loss: 0.0003 Acc: 0.9874
val Loss: 0.0043 Acc: 0.9197

Best val Acc: 0.923823

---Testing---
Test accuracy: 0.981307
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 98 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 100 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 28 %
Accuracy of   ray : 87 %
Accuracy of rough : 93 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9277978383735357, std: 0.18197467762530561
--------------------

run info[val: 0.2, epoch: 72, randcrop: False, decay: 7]

---Training last layer.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0124 Acc: 0.6276
val Loss: 0.0083 Acc: 0.7403

Epoch 1/71
----------
train Loss: 0.0069 Acc: 0.7674
val Loss: 0.0066 Acc: 0.7881

Epoch 2/71
----------
train Loss: 0.0059 Acc: 0.8010
val Loss: 0.0072 Acc: 0.7486

Epoch 3/71
----------
train Loss: 0.0054 Acc: 0.8209
val Loss: 0.0063 Acc: 0.7819

Epoch 4/71
----------
train Loss: 0.0049 Acc: 0.8352
val Loss: 0.0061 Acc: 0.8158

Epoch 5/71
----------
train Loss: 0.0046 Acc: 0.8480
val Loss: 0.0067 Acc: 0.7735

Epoch 6/71
----------
train Loss: 0.0045 Acc: 0.8525
val Loss: 0.0056 Acc: 0.8151

Epoch 7/71
----------
LR is set to 0.001
train Loss: 0.0038 Acc: 0.8789
val Loss: 0.0054 Acc: 0.8213

Epoch 8/71
----------
train Loss: 0.0039 Acc: 0.8763
val Loss: 0.0054 Acc: 0.8213

Epoch 9/71
----------
train Loss: 0.0039 Acc: 0.8737
val Loss: 0.0054 Acc: 0.8248

Epoch 10/71
----------
train Loss: 0.0038 Acc: 0.8773
val Loss: 0.0053 Acc: 0.8199

Epoch 11/71
----------
train Loss: 0.0038 Acc: 0.8775
val Loss: 0.0053 Acc: 0.8248

Epoch 12/71
----------
train Loss: 0.0038 Acc: 0.8818
val Loss: 0.0053 Acc: 0.8303

Epoch 13/71
----------
train Loss: 0.0038 Acc: 0.8771
val Loss: 0.0053 Acc: 0.8199

Epoch 14/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0037 Acc: 0.8863
val Loss: 0.0053 Acc: 0.8220

Epoch 15/71
----------
train Loss: 0.0037 Acc: 0.8835
val Loss: 0.0053 Acc: 0.8227

Epoch 16/71
----------
train Loss: 0.0038 Acc: 0.8837
val Loss: 0.0053 Acc: 0.8248

Epoch 17/71
----------
train Loss: 0.0037 Acc: 0.8799
val Loss: 0.0052 Acc: 0.8234

Epoch 18/71
----------
train Loss: 0.0037 Acc: 0.8820
val Loss: 0.0053 Acc: 0.8255

Epoch 19/71
----------
train Loss: 0.0038 Acc: 0.8792
val Loss: 0.0053 Acc: 0.8220

Epoch 20/71
----------
train Loss: 0.0037 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8248

Epoch 21/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0038 Acc: 0.8828
val Loss: 0.0054 Acc: 0.8234

Epoch 22/71
----------
train Loss: 0.0037 Acc: 0.8835
val Loss: 0.0053 Acc: 0.8241

Epoch 23/71
----------
train Loss: 0.0037 Acc: 0.8859
val Loss: 0.0054 Acc: 0.8255

Epoch 24/71
----------
train Loss: 0.0037 Acc: 0.8795
val Loss: 0.0053 Acc: 0.8276

Epoch 25/71
----------
train Loss: 0.0037 Acc: 0.8856
val Loss: 0.0053 Acc: 0.8255

Epoch 26/71
----------
train Loss: 0.0037 Acc: 0.8842
val Loss: 0.0054 Acc: 0.8227

Epoch 27/71
----------
train Loss: 0.0037 Acc: 0.8835
val Loss: 0.0053 Acc: 0.8276

Epoch 28/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0037 Acc: 0.8825
val Loss: 0.0053 Acc: 0.8255

Epoch 29/71
----------
train Loss: 0.0037 Acc: 0.8825
val Loss: 0.0054 Acc: 0.8255

Epoch 30/71
----------
train Loss: 0.0037 Acc: 0.8835
val Loss: 0.0054 Acc: 0.8248

Epoch 31/71
----------
train Loss: 0.0037 Acc: 0.8808
val Loss: 0.0053 Acc: 0.8227

Epoch 32/71
----------
train Loss: 0.0037 Acc: 0.8827
val Loss: 0.0054 Acc: 0.8227

Epoch 33/71
----------
train Loss: 0.0037 Acc: 0.8825
val Loss: 0.0053 Acc: 0.8262

Epoch 34/71
----------
train Loss: 0.0038 Acc: 0.8839
val Loss: 0.0054 Acc: 0.8227

Epoch 35/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0037 Acc: 0.8844
val Loss: 0.0053 Acc: 0.8241

Epoch 36/71
----------
train Loss: 0.0037 Acc: 0.8837
val Loss: 0.0053 Acc: 0.8248

Epoch 37/71
----------
train Loss: 0.0037 Acc: 0.8809
val Loss: 0.0053 Acc: 0.8227

Epoch 38/71
----------
train Loss: 0.0037 Acc: 0.8806
val Loss: 0.0052 Acc: 0.8255

Epoch 39/71
----------
train Loss: 0.0038 Acc: 0.8814
val Loss: 0.0053 Acc: 0.8234

Epoch 40/71
----------
train Loss: 0.0038 Acc: 0.8830
val Loss: 0.0054 Acc: 0.8227

Epoch 41/71
----------
train Loss: 0.0037 Acc: 0.8785
val Loss: 0.0053 Acc: 0.8234

Epoch 42/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0037 Acc: 0.8851
val Loss: 0.0053 Acc: 0.8227

Epoch 43/71
----------
train Loss: 0.0037 Acc: 0.8866
val Loss: 0.0054 Acc: 0.8255

Epoch 44/71
----------
train Loss: 0.0037 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8241

Epoch 45/71
----------
train Loss: 0.0037 Acc: 0.8851
val Loss: 0.0054 Acc: 0.8213

Epoch 46/71
----------
train Loss: 0.0037 Acc: 0.8847
val Loss: 0.0053 Acc: 0.8255

Epoch 47/71
----------
train Loss: 0.0038 Acc: 0.8804
val Loss: 0.0053 Acc: 0.8241

Epoch 48/71
----------
train Loss: 0.0037 Acc: 0.8814
val Loss: 0.0053 Acc: 0.8213

Epoch 49/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0038 Acc: 0.8783
val Loss: 0.0053 Acc: 0.8234

Epoch 50/71
----------
train Loss: 0.0037 Acc: 0.8808
val Loss: 0.0054 Acc: 0.8255

Epoch 51/71
----------
train Loss: 0.0037 Acc: 0.8849
val Loss: 0.0053 Acc: 0.8255

Epoch 52/71
----------
train Loss: 0.0037 Acc: 0.8858
val Loss: 0.0053 Acc: 0.8213

Epoch 53/71
----------
train Loss: 0.0037 Acc: 0.8823
val Loss: 0.0053 Acc: 0.8234

Epoch 54/71
----------
train Loss: 0.0037 Acc: 0.8847
val Loss: 0.0053 Acc: 0.8241

Epoch 55/71
----------
train Loss: 0.0038 Acc: 0.8832
val Loss: 0.0053 Acc: 0.8234

Epoch 56/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0037 Acc: 0.8821
val Loss: 0.0053 Acc: 0.8234

Epoch 57/71
----------
train Loss: 0.0037 Acc: 0.8827
val Loss: 0.0053 Acc: 0.8248

Epoch 58/71
----------
train Loss: 0.0037 Acc: 0.8820
val Loss: 0.0054 Acc: 0.8255

Epoch 59/71
----------
train Loss: 0.0037 Acc: 0.8814
val Loss: 0.0054 Acc: 0.8255

Epoch 60/71
----------
train Loss: 0.0037 Acc: 0.8846
val Loss: 0.0054 Acc: 0.8255

Epoch 61/71
----------
train Loss: 0.0037 Acc: 0.8846
val Loss: 0.0053 Acc: 0.8213

Epoch 62/71
----------
train Loss: 0.0037 Acc: 0.8816
val Loss: 0.0054 Acc: 0.8220

Epoch 63/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0037 Acc: 0.8839
val Loss: 0.0053 Acc: 0.8269

Epoch 64/71
----------
train Loss: 0.0037 Acc: 0.8846
val Loss: 0.0053 Acc: 0.8296

Epoch 65/71
----------
train Loss: 0.0037 Acc: 0.8816
val Loss: 0.0054 Acc: 0.8227

Epoch 66/71
----------
train Loss: 0.0037 Acc: 0.8828
val Loss: 0.0053 Acc: 0.8276

Epoch 67/71
----------
train Loss: 0.0037 Acc: 0.8868
val Loss: 0.0054 Acc: 0.8234

Epoch 68/71
----------
train Loss: 0.0038 Acc: 0.8792
val Loss: 0.0053 Acc: 0.8241

Epoch 69/71
----------
train Loss: 0.0037 Acc: 0.8795
val Loss: 0.0054 Acc: 0.8234

Epoch 70/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0038 Acc: 0.8842
val Loss: 0.0053 Acc: 0.8234

Epoch 71/71
----------
train Loss: 0.0037 Acc: 0.8828
val Loss: 0.0053 Acc: 0.8248

Best val Acc: 0.830332

---Fine tuning.---
Epoch 0/71
----------
LR is set to 0.01
train Loss: 0.0048 Acc: 0.8489
val Loss: 0.0061 Acc: 0.8324

Epoch 1/71
----------
train Loss: 0.0021 Acc: 0.9294
val Loss: 0.0039 Acc: 0.8947

Epoch 2/71
----------
train Loss: 0.0010 Acc: 0.9659
val Loss: 0.0039 Acc: 0.8934

Epoch 3/71
----------
train Loss: 0.0006 Acc: 0.9780
val Loss: 0.0036 Acc: 0.9044

Epoch 4/71
----------
train Loss: 0.0004 Acc: 0.9832
val Loss: 0.0036 Acc: 0.9044

Epoch 5/71
----------
train Loss: 0.0003 Acc: 0.9848
val Loss: 0.0036 Acc: 0.9072

Epoch 6/71
----------
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9114

Epoch 7/71
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9114

Epoch 8/71
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9107

Epoch 9/71
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9127

Epoch 10/71
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0037 Acc: 0.9107

Epoch 11/71
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9120

Epoch 12/71
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0038 Acc: 0.9127

Epoch 13/71
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0037 Acc: 0.9141

Epoch 14/71
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0037 Acc: 0.9127

Epoch 15/71
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0036 Acc: 0.9120

Epoch 16/71
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0037 Acc: 0.9127

Epoch 17/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9120

Epoch 18/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9127

Epoch 19/71
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0036 Acc: 0.9134

Epoch 20/71
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0036 Acc: 0.9120

Epoch 21/71
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0040 Acc: 0.9127

Epoch 22/71
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0038 Acc: 0.9120

Epoch 23/71
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0038 Acc: 0.9114

Epoch 24/71
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0038 Acc: 0.9141

Epoch 25/71
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0037 Acc: 0.9127

Epoch 26/71
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9134

Epoch 27/71
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0038 Acc: 0.9114

Epoch 28/71
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9127

Epoch 29/71
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9120

Epoch 30/71
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0039 Acc: 0.9141

Epoch 31/71
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0037 Acc: 0.9134

Epoch 32/71
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0037 Acc: 0.9107

Epoch 33/71
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0036 Acc: 0.9134

Epoch 34/71
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9127

Epoch 35/71
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0036 Acc: 0.9148

Epoch 36/71
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9134

Epoch 37/71
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0036 Acc: 0.9127

Epoch 38/71
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0036 Acc: 0.9134

Epoch 39/71
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0038 Acc: 0.9127

Epoch 40/71
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0037 Acc: 0.9134

Epoch 41/71
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0036 Acc: 0.9148

Epoch 42/71
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0037 Acc: 0.9120

Epoch 43/71
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9120

Epoch 44/71
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0036 Acc: 0.9120

Epoch 45/71
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9134

Epoch 46/71
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0036 Acc: 0.9127

Epoch 47/71
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0040 Acc: 0.9127

Epoch 48/71
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0036 Acc: 0.9114

Epoch 49/71
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9127

Epoch 50/71
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9120

Epoch 51/71
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0036 Acc: 0.9127

Epoch 52/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9120

Epoch 53/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9127

Epoch 54/71
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0037 Acc: 0.9127

Epoch 55/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9107

Epoch 56/71
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9127

Epoch 57/71
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0038 Acc: 0.9120

Epoch 58/71
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0039 Acc: 0.9134

Epoch 59/71
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0037 Acc: 0.9134

Epoch 60/71
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0038 Acc: 0.9134

Epoch 61/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0038 Acc: 0.9120

Epoch 62/71
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9134

Epoch 63/71
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9114

Epoch 64/71
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0036 Acc: 0.9120

Epoch 65/71
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0037 Acc: 0.9120

Epoch 66/71
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0037 Acc: 0.9134

Epoch 67/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0038 Acc: 0.9120

Epoch 68/71
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0036 Acc: 0.9127

Epoch 69/71
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0036 Acc: 0.9127

Epoch 70/71
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9120

Epoch 71/71
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9127

Best val Acc: 0.914820

---Testing---
Test accuracy: 0.975215
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 98 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 94 %
Accuracy of mullet : 36 %
Accuracy of   ray : 81 %
Accuracy of rough : 86 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9208205696805346, std: 0.16295979003653227
--------------------

run info[val: 0.3, epoch: 50, randcrop: True, decay: 7]

---Training last layer.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0131 Acc: 0.6119
val Loss: 0.0090 Acc: 0.6967

Epoch 1/49
----------
train Loss: 0.0078 Acc: 0.7330
val Loss: 0.0071 Acc: 0.7632

Epoch 2/49
----------
train Loss: 0.0067 Acc: 0.7725
val Loss: 0.0066 Acc: 0.7664

Epoch 3/49
----------
train Loss: 0.0063 Acc: 0.7907
val Loss: 0.0088 Acc: 0.7211

Epoch 4/49
----------
train Loss: 0.0063 Acc: 0.7830
val Loss: 0.0088 Acc: 0.7244

Epoch 5/49
----------
train Loss: 0.0060 Acc: 0.7979
val Loss: 0.0060 Acc: 0.8015

Epoch 6/49
----------
train Loss: 0.0054 Acc: 0.8095
val Loss: 0.0062 Acc: 0.7899

Epoch 7/49
----------
LR is set to 0.001
train Loss: 0.0047 Acc: 0.8424
val Loss: 0.0058 Acc: 0.7987

Epoch 8/49
----------
train Loss: 0.0049 Acc: 0.8374
val Loss: 0.0058 Acc: 0.8019

Epoch 9/49
----------
train Loss: 0.0047 Acc: 0.8400
val Loss: 0.0057 Acc: 0.8029

Epoch 10/49
----------
train Loss: 0.0047 Acc: 0.8439
val Loss: 0.0057 Acc: 0.8033

Epoch 11/49
----------
train Loss: 0.0048 Acc: 0.8420
val Loss: 0.0057 Acc: 0.8038

Epoch 12/49
----------
train Loss: 0.0046 Acc: 0.8538
val Loss: 0.0058 Acc: 0.7996

Epoch 13/49
----------
train Loss: 0.0046 Acc: 0.8532
val Loss: 0.0057 Acc: 0.8006

Epoch 14/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0047 Acc: 0.8467
val Loss: 0.0057 Acc: 0.7996

Epoch 15/49
----------
train Loss: 0.0046 Acc: 0.8517
val Loss: 0.0057 Acc: 0.8019

Epoch 16/49
----------
train Loss: 0.0047 Acc: 0.8501
val Loss: 0.0057 Acc: 0.8033

Epoch 17/49
----------
train Loss: 0.0046 Acc: 0.8511
val Loss: 0.0057 Acc: 0.8015

Epoch 18/49
----------
train Loss: 0.0047 Acc: 0.8430
val Loss: 0.0057 Acc: 0.8001

Epoch 19/49
----------
train Loss: 0.0046 Acc: 0.8538
val Loss: 0.0057 Acc: 0.7987

Epoch 20/49
----------
train Loss: 0.0046 Acc: 0.8483
val Loss: 0.0057 Acc: 0.8033

Epoch 21/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0046 Acc: 0.8491
val Loss: 0.0057 Acc: 0.8019

Epoch 22/49
----------
train Loss: 0.0047 Acc: 0.8461
val Loss: 0.0057 Acc: 0.8001

Epoch 23/49
----------
train Loss: 0.0046 Acc: 0.8505
val Loss: 0.0057 Acc: 0.8015

Epoch 24/49
----------
train Loss: 0.0047 Acc: 0.8441
val Loss: 0.0057 Acc: 0.8052

Epoch 25/49
----------
train Loss: 0.0046 Acc: 0.8453
val Loss: 0.0057 Acc: 0.8047

Epoch 26/49
----------
train Loss: 0.0046 Acc: 0.8471
val Loss: 0.0057 Acc: 0.8006

Epoch 27/49
----------
train Loss: 0.0046 Acc: 0.8489
val Loss: 0.0057 Acc: 0.8006

Epoch 28/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0047 Acc: 0.8475
val Loss: 0.0057 Acc: 0.8029

Epoch 29/49
----------
train Loss: 0.0046 Acc: 0.8449
val Loss: 0.0057 Acc: 0.8015

Epoch 30/49
----------
train Loss: 0.0046 Acc: 0.8493
val Loss: 0.0057 Acc: 0.8001

Epoch 31/49
----------
train Loss: 0.0045 Acc: 0.8536
val Loss: 0.0057 Acc: 0.8029

Epoch 32/49
----------
train Loss: 0.0046 Acc: 0.8525
val Loss: 0.0057 Acc: 0.8024

Epoch 33/49
----------
train Loss: 0.0045 Acc: 0.8517
val Loss: 0.0057 Acc: 0.8019

Epoch 34/49
----------
train Loss: 0.0047 Acc: 0.8477
val Loss: 0.0057 Acc: 0.8019

Epoch 35/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0047 Acc: 0.8453
val Loss: 0.0057 Acc: 0.8024

Epoch 36/49
----------
train Loss: 0.0045 Acc: 0.8483
val Loss: 0.0057 Acc: 0.8038

Epoch 37/49
----------
train Loss: 0.0046 Acc: 0.8449
val Loss: 0.0057 Acc: 0.8033

Epoch 38/49
----------
train Loss: 0.0046 Acc: 0.8471
val Loss: 0.0057 Acc: 0.8010

Epoch 39/49
----------
train Loss: 0.0047 Acc: 0.8517
val Loss: 0.0057 Acc: 0.8015

Epoch 40/49
----------
train Loss: 0.0046 Acc: 0.8438
val Loss: 0.0057 Acc: 0.8029

Epoch 41/49
----------
train Loss: 0.0047 Acc: 0.8459
val Loss: 0.0057 Acc: 0.8019

Epoch 42/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0046 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8006

Epoch 43/49
----------
train Loss: 0.0046 Acc: 0.8491
val Loss: 0.0057 Acc: 0.8042

Epoch 44/49
----------
train Loss: 0.0047 Acc: 0.8424
val Loss: 0.0057 Acc: 0.8029

Epoch 45/49
----------
train Loss: 0.0045 Acc: 0.8534
val Loss: 0.0057 Acc: 0.8019

Epoch 46/49
----------
train Loss: 0.0046 Acc: 0.8473
val Loss: 0.0057 Acc: 0.8038

Epoch 47/49
----------
train Loss: 0.0046 Acc: 0.8449
val Loss: 0.0057 Acc: 0.8024

Epoch 48/49
----------
train Loss: 0.0045 Acc: 0.8540
val Loss: 0.0057 Acc: 0.8038

Epoch 49/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0045 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8052

Best val Acc: 0.805171

---Fine tuning.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0051 Acc: 0.8269
val Loss: 0.0093 Acc: 0.7184

Epoch 1/49
----------
train Loss: 0.0028 Acc: 0.9045
val Loss: 0.0053 Acc: 0.8366

Epoch 2/49
----------
train Loss: 0.0017 Acc: 0.9430
val Loss: 0.0045 Acc: 0.8647

Epoch 3/49
----------
train Loss: 0.0011 Acc: 0.9616
val Loss: 0.0046 Acc: 0.8707

Epoch 4/49
----------
train Loss: 0.0009 Acc: 0.9697
val Loss: 0.0044 Acc: 0.8707

Epoch 5/49
----------
train Loss: 0.0009 Acc: 0.9666
val Loss: 0.0042 Acc: 0.8906

Epoch 6/49
----------
train Loss: 0.0006 Acc: 0.9767
val Loss: 0.0037 Acc: 0.9072

Epoch 7/49
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9834
val Loss: 0.0037 Acc: 0.9141

Epoch 8/49
----------
train Loss: 0.0003 Acc: 0.9862
val Loss: 0.0037 Acc: 0.9123

Epoch 9/49
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9141

Epoch 10/49
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9137

Epoch 11/49
----------
train Loss: 0.0003 Acc: 0.9871
val Loss: 0.0037 Acc: 0.9127

Epoch 12/49
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0038 Acc: 0.9114

Epoch 13/49
----------
train Loss: 0.0003 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9127

Epoch 14/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9883
val Loss: 0.0037 Acc: 0.9137

Epoch 15/49
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0038 Acc: 0.9137

Epoch 16/49
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9146

Epoch 17/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9141

Epoch 18/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9127

Epoch 19/49
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9127

Epoch 20/49
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9123

Epoch 21/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9877
val Loss: 0.0038 Acc: 0.9123

Epoch 22/49
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0037 Acc: 0.9127

Epoch 23/49
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0037 Acc: 0.9127

Epoch 24/49
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0037 Acc: 0.9123

Epoch 25/49
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0038 Acc: 0.9137

Epoch 26/49
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9137

Epoch 27/49
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9132

Epoch 28/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9877
val Loss: 0.0037 Acc: 0.9132

Epoch 29/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9137

Epoch 30/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9127

Epoch 31/49
----------
train Loss: 0.0003 Acc: 0.9887
val Loss: 0.0037 Acc: 0.9137

Epoch 32/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9123

Epoch 33/49
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9132

Epoch 34/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9123

Epoch 35/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0037 Acc: 0.9132

Epoch 36/49
----------
train Loss: 0.0003 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9127

Epoch 37/49
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9141

Epoch 38/49
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0037 Acc: 0.9127

Epoch 39/49
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9127

Epoch 40/49
----------
train Loss: 0.0003 Acc: 0.9887
val Loss: 0.0037 Acc: 0.9137

Epoch 41/49
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0038 Acc: 0.9123

Epoch 42/49
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9132

Epoch 43/49
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9118

Epoch 44/49
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9141

Epoch 45/49
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0038 Acc: 0.9127

Epoch 46/49
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0038 Acc: 0.9123

Epoch 47/49
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0037 Acc: 0.9127

Epoch 48/49
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9132

Epoch 49/49
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0038 Acc: 0.9127

Best val Acc: 0.914589

---Testing---
Test accuracy: 0.968153
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 97 %
Accuracy of Batoidea(ga_oo_lee) : 96 %
Accuracy of   DOL : 97 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 95 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 92 %
Accuracy of mullet : 38 %
Accuracy of   ray : 77 %
Accuracy of rough : 82 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9094519353112732, std: 0.15799609825016733

Model saved in "./weights/super_with_most_[0.98]_mean[0.93]_std[0.18].save".
--------------------

run info[val: 0.1, epoch: 64, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0122 Acc: 0.6334
val Loss: 0.0085 Acc: 0.7341

Epoch 1/63
----------
train Loss: 0.0069 Acc: 0.7622
val Loss: 0.0072 Acc: 0.7729

Epoch 2/63
----------
train Loss: 0.0058 Acc: 0.8028
val Loss: 0.0068 Acc: 0.7881

Epoch 3/63
----------
train Loss: 0.0051 Acc: 0.8314
val Loss: 0.0055 Acc: 0.8407

Epoch 4/63
----------
train Loss: 0.0048 Acc: 0.8425
val Loss: 0.0057 Acc: 0.8490

Epoch 5/63
----------
train Loss: 0.0047 Acc: 0.8400
val Loss: 0.0057 Acc: 0.8172

Epoch 6/63
----------
train Loss: 0.0044 Acc: 0.8562
val Loss: 0.0055 Acc: 0.8296

Epoch 7/63
----------
train Loss: 0.0044 Acc: 0.8515
val Loss: 0.0053 Acc: 0.8352

Epoch 8/63
----------
train Loss: 0.0041 Acc: 0.8605
val Loss: 0.0056 Acc: 0.8393

Epoch 9/63
----------
LR is set to 0.001
train Loss: 0.0036 Acc: 0.8845
val Loss: 0.0049 Acc: 0.8518

Epoch 10/63
----------
train Loss: 0.0036 Acc: 0.8845
val Loss: 0.0049 Acc: 0.8421

Epoch 11/63
----------
train Loss: 0.0036 Acc: 0.8843
val Loss: 0.0051 Acc: 0.8476

Epoch 12/63
----------
train Loss: 0.0036 Acc: 0.8814
val Loss: 0.0050 Acc: 0.8504

Epoch 13/63
----------
train Loss: 0.0035 Acc: 0.8889
val Loss: 0.0048 Acc: 0.8490

Epoch 14/63
----------
train Loss: 0.0035 Acc: 0.8889
val Loss: 0.0047 Acc: 0.8421

Epoch 15/63
----------
train Loss: 0.0035 Acc: 0.8908
val Loss: 0.0049 Acc: 0.8421

Epoch 16/63
----------
train Loss: 0.0035 Acc: 0.8862
val Loss: 0.0048 Acc: 0.8463

Epoch 17/63
----------
train Loss: 0.0035 Acc: 0.8860
val Loss: 0.0049 Acc: 0.8504

Epoch 18/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0035 Acc: 0.8898
val Loss: 0.0051 Acc: 0.8421

Epoch 19/63
----------
train Loss: 0.0035 Acc: 0.8858
val Loss: 0.0050 Acc: 0.8449

Epoch 20/63
----------
train Loss: 0.0035 Acc: 0.8906
val Loss: 0.0052 Acc: 0.8435

Epoch 21/63
----------
train Loss: 0.0035 Acc: 0.8858
val Loss: 0.0051 Acc: 0.8463

Epoch 22/63
----------
train Loss: 0.0035 Acc: 0.8885
val Loss: 0.0050 Acc: 0.8518

Epoch 23/63
----------
train Loss: 0.0035 Acc: 0.8897
val Loss: 0.0049 Acc: 0.8476

Epoch 24/63
----------
train Loss: 0.0034 Acc: 0.8935
val Loss: 0.0054 Acc: 0.8476

Epoch 25/63
----------
train Loss: 0.0035 Acc: 0.8875
val Loss: 0.0054 Acc: 0.8476

Epoch 26/63
----------
train Loss: 0.0035 Acc: 0.8888
val Loss: 0.0050 Acc: 0.8463

Epoch 27/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0035 Acc: 0.8895
val Loss: 0.0049 Acc: 0.8421

Epoch 28/63
----------
train Loss: 0.0035 Acc: 0.8898
val Loss: 0.0047 Acc: 0.8463

Epoch 29/63
----------
train Loss: 0.0034 Acc: 0.8934
val Loss: 0.0048 Acc: 0.8463

Epoch 30/63
----------
train Loss: 0.0035 Acc: 0.8897
val Loss: 0.0049 Acc: 0.8449

Epoch 31/63
----------
train Loss: 0.0035 Acc: 0.8894
val Loss: 0.0048 Acc: 0.8476

Epoch 32/63
----------
train Loss: 0.0034 Acc: 0.8923
val Loss: 0.0047 Acc: 0.8476

Epoch 33/63
----------
train Loss: 0.0035 Acc: 0.8854
val Loss: 0.0049 Acc: 0.8463

Epoch 34/63
----------
train Loss: 0.0035 Acc: 0.8885
val Loss: 0.0048 Acc: 0.8421

Epoch 35/63
----------
train Loss: 0.0035 Acc: 0.8909
val Loss: 0.0048 Acc: 0.8476

Epoch 36/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0035 Acc: 0.8892
val Loss: 0.0051 Acc: 0.8476

Epoch 37/63
----------
train Loss: 0.0035 Acc: 0.8894
val Loss: 0.0048 Acc: 0.8435

Epoch 38/63
----------
train Loss: 0.0035 Acc: 0.8905
val Loss: 0.0047 Acc: 0.8463

Epoch 39/63
----------
train Loss: 0.0034 Acc: 0.8909
val Loss: 0.0048 Acc: 0.8463

Epoch 40/63
----------
train Loss: 0.0036 Acc: 0.8854
val Loss: 0.0048 Acc: 0.8449

Epoch 41/63
----------
train Loss: 0.0035 Acc: 0.8889
val Loss: 0.0051 Acc: 0.8476

Epoch 42/63
----------
train Loss: 0.0035 Acc: 0.8903
val Loss: 0.0055 Acc: 0.8490

Epoch 43/63
----------
train Loss: 0.0034 Acc: 0.8911
val Loss: 0.0053 Acc: 0.8449

Epoch 44/63
----------
train Loss: 0.0035 Acc: 0.8894
val Loss: 0.0049 Acc: 0.8435

Epoch 45/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0035 Acc: 0.8914
val Loss: 0.0048 Acc: 0.8463

Epoch 46/63
----------
train Loss: 0.0034 Acc: 0.8895
val Loss: 0.0051 Acc: 0.8435

Epoch 47/63
----------
train Loss: 0.0034 Acc: 0.8905
val Loss: 0.0051 Acc: 0.8490

Epoch 48/63
----------
train Loss: 0.0035 Acc: 0.8908
val Loss: 0.0048 Acc: 0.8518

Epoch 49/63
----------
train Loss: 0.0035 Acc: 0.8875
val Loss: 0.0050 Acc: 0.8490

Epoch 50/63
----------
train Loss: 0.0035 Acc: 0.8875
val Loss: 0.0049 Acc: 0.8504

Epoch 51/63
----------
train Loss: 0.0034 Acc: 0.8914
val Loss: 0.0053 Acc: 0.8490

Epoch 52/63
----------
train Loss: 0.0035 Acc: 0.8900
val Loss: 0.0048 Acc: 0.8476

Epoch 53/63
----------
train Loss: 0.0034 Acc: 0.8877
val Loss: 0.0048 Acc: 0.8435

Epoch 54/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0034 Acc: 0.8951
val Loss: 0.0053 Acc: 0.8463

Epoch 55/63
----------
train Loss: 0.0034 Acc: 0.8931
val Loss: 0.0050 Acc: 0.8490

Epoch 56/63
----------
train Loss: 0.0035 Acc: 0.8903
val Loss: 0.0051 Acc: 0.8463

Epoch 57/63
----------
train Loss: 0.0035 Acc: 0.8898
val Loss: 0.0048 Acc: 0.8435

Epoch 58/63
----------
train Loss: 0.0035 Acc: 0.8877
val Loss: 0.0050 Acc: 0.8435

Epoch 59/63
----------
train Loss: 0.0035 Acc: 0.8880
val Loss: 0.0048 Acc: 0.8435

Epoch 60/63
----------
train Loss: 0.0035 Acc: 0.8894
val Loss: 0.0049 Acc: 0.8463

Epoch 61/63
----------
train Loss: 0.0035 Acc: 0.8869
val Loss: 0.0049 Acc: 0.8463

Epoch 62/63
----------
train Loss: 0.0034 Acc: 0.8948
val Loss: 0.0051 Acc: 0.8449

Epoch 63/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0035 Acc: 0.8906
val Loss: 0.0049 Acc: 0.8449

Best val Acc: 0.851801

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0045 Acc: 0.8477
val Loss: 0.0074 Acc: 0.7867

Epoch 1/63
----------
train Loss: 0.0020 Acc: 0.9328
val Loss: 0.0040 Acc: 0.8906

Epoch 2/63
----------
train Loss: 0.0010 Acc: 0.9652
val Loss: 0.0041 Acc: 0.9030

Epoch 3/63
----------
train Loss: 0.0006 Acc: 0.9780
val Loss: 0.0032 Acc: 0.9127

Epoch 4/63
----------
train Loss: 0.0005 Acc: 0.9798
val Loss: 0.0036 Acc: 0.9086

Epoch 5/63
----------
train Loss: 0.0004 Acc: 0.9837
val Loss: 0.0038 Acc: 0.9169

Epoch 6/63
----------
train Loss: 0.0003 Acc: 0.9869
val Loss: 0.0038 Acc: 0.9127

Epoch 7/63
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0041 Acc: 0.9141

Epoch 8/63
----------
train Loss: 0.0003 Acc: 0.9871
val Loss: 0.0033 Acc: 0.9155

Epoch 9/63
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0034 Acc: 0.9183

Epoch 10/63
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0034 Acc: 0.9141

Epoch 11/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0043 Acc: 0.9169

Epoch 12/63
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0042 Acc: 0.9141

Epoch 13/63
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9169

Epoch 14/63
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0037 Acc: 0.9183

Epoch 15/63
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0037 Acc: 0.9197

Epoch 16/63
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0033 Acc: 0.9197

Epoch 17/63
----------
train Loss: 0.0002 Acc: 0.9866
val Loss: 0.0038 Acc: 0.9183

Epoch 18/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0036 Acc: 0.9183

Epoch 19/63
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0033 Acc: 0.9197

Epoch 20/63
----------
train Loss: 0.0002 Acc: 0.9902
val Loss: 0.0035 Acc: 0.9197

Epoch 21/63
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0034 Acc: 0.9169

Epoch 22/63
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0036 Acc: 0.9197

Epoch 23/63
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0035 Acc: 0.9169

Epoch 24/63
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0043 Acc: 0.9197

Epoch 25/63
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0036 Acc: 0.9197

Epoch 26/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0042 Acc: 0.9197

Epoch 27/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0034 Acc: 0.9197

Epoch 28/63
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0033 Acc: 0.9169

Epoch 29/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0034 Acc: 0.9211

Epoch 30/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0040 Acc: 0.9197

Epoch 31/63
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0036 Acc: 0.9197

Epoch 32/63
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0035 Acc: 0.9183

Epoch 33/63
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0049 Acc: 0.9155

Epoch 34/63
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9169

Epoch 35/63
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0033 Acc: 0.9197

Epoch 36/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0038 Acc: 0.9155

Epoch 37/63
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0035 Acc: 0.9224

Epoch 38/63
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0035 Acc: 0.9197

Epoch 39/63
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0039 Acc: 0.9211

Epoch 40/63
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0038 Acc: 0.9183

Epoch 41/63
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0037 Acc: 0.9224

Epoch 42/63
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0035 Acc: 0.9183

Epoch 43/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0034 Acc: 0.9183

Epoch 44/63
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0038 Acc: 0.9169

Epoch 45/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0034 Acc: 0.9169

Epoch 46/63
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0033 Acc: 0.9211

Epoch 47/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0035 Acc: 0.9183

Epoch 48/63
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0038 Acc: 0.9169

Epoch 49/63
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0035 Acc: 0.9183

Epoch 50/63
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0033 Acc: 0.9197

Epoch 51/63
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0038 Acc: 0.9183

Epoch 52/63
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0035 Acc: 0.9183

Epoch 53/63
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0040 Acc: 0.9197

Epoch 54/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0038 Acc: 0.9197

Epoch 55/63
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0035 Acc: 0.9211

Epoch 56/63
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9155

Epoch 57/63
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0040 Acc: 0.9169

Epoch 58/63
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9211

Epoch 59/63
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0043 Acc: 0.9224

Epoch 60/63
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0034 Acc: 0.9155

Epoch 61/63
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0034 Acc: 0.9169

Epoch 62/63
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0034 Acc: 0.9224

Epoch 63/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0033 Acc: 0.9169

Best val Acc: 0.922438

---Testing---
Test accuracy: 0.981999
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 100 %
Accuracy of Batoidea(ga_oo_lee) : 99 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 100 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 47 %
Accuracy of   ray : 80 %
Accuracy of rough : 93 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9375979888134512, std: 0.13912180780455266
--------------------

run info[val: 0.2, epoch: 35, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/34
----------
LR is set to 0.01
train Loss: 0.0125 Acc: 0.6248
val Loss: 0.0082 Acc: 0.7299

Epoch 1/34
----------
train Loss: 0.0068 Acc: 0.7788
val Loss: 0.0070 Acc: 0.7729

Epoch 2/34
----------
train Loss: 0.0058 Acc: 0.8094
val Loss: 0.0064 Acc: 0.7936

Epoch 3/34
----------
train Loss: 0.0054 Acc: 0.8204
val Loss: 0.0065 Acc: 0.7846

Epoch 4/34
----------
train Loss: 0.0050 Acc: 0.8325
val Loss: 0.0061 Acc: 0.7902

Epoch 5/34
----------
train Loss: 0.0046 Acc: 0.8474
val Loss: 0.0056 Acc: 0.8068

Epoch 6/34
----------
LR is set to 0.001
train Loss: 0.0041 Acc: 0.8692
val Loss: 0.0054 Acc: 0.8269

Epoch 7/34
----------
train Loss: 0.0041 Acc: 0.8719
val Loss: 0.0055 Acc: 0.8227

Epoch 8/34
----------
train Loss: 0.0040 Acc: 0.8712
val Loss: 0.0055 Acc: 0.8213

Epoch 9/34
----------
train Loss: 0.0040 Acc: 0.8728
val Loss: 0.0055 Acc: 0.8234

Epoch 10/34
----------
train Loss: 0.0040 Acc: 0.8733
val Loss: 0.0055 Acc: 0.8199

Epoch 11/34
----------
train Loss: 0.0040 Acc: 0.8728
val Loss: 0.0054 Acc: 0.8262

Epoch 12/34
----------
LR is set to 0.00010000000000000002
train Loss: 0.0040 Acc: 0.8756
val Loss: 0.0055 Acc: 0.8255

Epoch 13/34
----------
train Loss: 0.0040 Acc: 0.8768
val Loss: 0.0054 Acc: 0.8255

Epoch 14/34
----------
train Loss: 0.0039 Acc: 0.8783
val Loss: 0.0054 Acc: 0.8241

Epoch 15/34
----------
train Loss: 0.0039 Acc: 0.8785
val Loss: 0.0054 Acc: 0.8227

Epoch 16/34
----------
train Loss: 0.0039 Acc: 0.8766
val Loss: 0.0054 Acc: 0.8234

Epoch 17/34
----------
train Loss: 0.0039 Acc: 0.8832
val Loss: 0.0054 Acc: 0.8262

Epoch 18/34
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0039 Acc: 0.8789
val Loss: 0.0054 Acc: 0.8255

Epoch 19/34
----------
train Loss: 0.0039 Acc: 0.8768
val Loss: 0.0054 Acc: 0.8262

Epoch 20/34
----------
train Loss: 0.0039 Acc: 0.8790
val Loss: 0.0054 Acc: 0.8269

Epoch 21/34
----------
train Loss: 0.0039 Acc: 0.8782
val Loss: 0.0054 Acc: 0.8213

Epoch 22/34
----------
train Loss: 0.0039 Acc: 0.8771
val Loss: 0.0054 Acc: 0.8248

Epoch 23/34
----------
train Loss: 0.0039 Acc: 0.8783
val Loss: 0.0055 Acc: 0.8213

Epoch 24/34
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0038 Acc: 0.8801
val Loss: 0.0054 Acc: 0.8255

Epoch 25/34
----------
train Loss: 0.0039 Acc: 0.8776
val Loss: 0.0054 Acc: 0.8255

Epoch 26/34
----------
train Loss: 0.0039 Acc: 0.8783
val Loss: 0.0055 Acc: 0.8241

Epoch 27/34
----------
train Loss: 0.0039 Acc: 0.8782
val Loss: 0.0054 Acc: 0.8241

Epoch 28/34
----------
train Loss: 0.0039 Acc: 0.8757
val Loss: 0.0054 Acc: 0.8248

Epoch 29/34
----------
train Loss: 0.0040 Acc: 0.8738
val Loss: 0.0053 Acc: 0.8276

Epoch 30/34
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0040 Acc: 0.8750
val Loss: 0.0054 Acc: 0.8248

Epoch 31/34
----------
train Loss: 0.0040 Acc: 0.8709
val Loss: 0.0054 Acc: 0.8234

Epoch 32/34
----------
train Loss: 0.0039 Acc: 0.8723
val Loss: 0.0055 Acc: 0.8227

Epoch 33/34
----------
train Loss: 0.0039 Acc: 0.8789
val Loss: 0.0054 Acc: 0.8269

Epoch 34/34
----------
train Loss: 0.0039 Acc: 0.8775
val Loss: 0.0055 Acc: 0.8241

Best val Acc: 0.827562

---Fine tuning.---
Epoch 0/34
----------
LR is set to 0.01
train Loss: 0.0044 Acc: 0.8520
val Loss: 0.0056 Acc: 0.8324

Epoch 1/34
----------
train Loss: 0.0021 Acc: 0.9296
val Loss: 0.0040 Acc: 0.8712

Epoch 2/34
----------
train Loss: 0.0010 Acc: 0.9643
val Loss: 0.0033 Acc: 0.9079

Epoch 3/34
----------
train Loss: 0.0006 Acc: 0.9822
val Loss: 0.0031 Acc: 0.9155

Epoch 4/34
----------
train Loss: 0.0004 Acc: 0.9843
val Loss: 0.0033 Acc: 0.9197

Epoch 5/34
----------
train Loss: 0.0004 Acc: 0.9855
val Loss: 0.0034 Acc: 0.9169

Epoch 6/34
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0033 Acc: 0.9211

Epoch 7/34
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0032 Acc: 0.9217

Epoch 8/34
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0032 Acc: 0.9217

Epoch 9/34
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0032 Acc: 0.9217

Epoch 10/34
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0033 Acc: 0.9204

Epoch 11/34
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9224

Epoch 12/34
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0032 Acc: 0.9231

Epoch 13/34
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0032 Acc: 0.9231

Epoch 14/34
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0032 Acc: 0.9231

Epoch 15/34
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0033 Acc: 0.9217

Epoch 16/34
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0033 Acc: 0.9231

Epoch 17/34
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0031 Acc: 0.9231

Epoch 18/34
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0031 Acc: 0.9224

Epoch 19/34
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0035 Acc: 0.9204

Epoch 20/34
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0033 Acc: 0.9217

Epoch 21/34
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0033 Acc: 0.9224

Epoch 22/34
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0032 Acc: 0.9238

Epoch 23/34
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0034 Acc: 0.9217

Epoch 24/34
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9217

Epoch 25/34
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0033 Acc: 0.9231

Epoch 26/34
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0033 Acc: 0.9217

Epoch 27/34
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0031 Acc: 0.9224

Epoch 28/34
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0033 Acc: 0.9217

Epoch 29/34
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0032 Acc: 0.9211

Epoch 30/34
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0032 Acc: 0.9245

Epoch 31/34
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0032 Acc: 0.9238

Epoch 32/34
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0032 Acc: 0.9211

Epoch 33/34
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0032 Acc: 0.9217

Epoch 34/34
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0032 Acc: 0.9238

Best val Acc: 0.924515

---Testing---
Test accuracy: 0.977153
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 99 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 94 %
Accuracy of mullet : 37 %
Accuracy of   ray : 82 %
Accuracy of rough : 86 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.923508168366057, std: 0.1605225048925157
--------------------

run info[val: 0.3, epoch: 79, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/78
----------
LR is set to 0.01
train Loss: 0.0129 Acc: 0.6086
val Loss: 0.0082 Acc: 0.7345

Epoch 1/78
----------
train Loss: 0.0071 Acc: 0.7745
val Loss: 0.0081 Acc: 0.7142

Epoch 2/78
----------
train Loss: 0.0060 Acc: 0.8004
val Loss: 0.0065 Acc: 0.7807

Epoch 3/78
----------
train Loss: 0.0054 Acc: 0.8259
val Loss: 0.0063 Acc: 0.7849

Epoch 4/78
----------
train Loss: 0.0049 Acc: 0.8398
val Loss: 0.0061 Acc: 0.8010

Epoch 5/78
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.8612
val Loss: 0.0058 Acc: 0.8056

Epoch 6/78
----------
train Loss: 0.0043 Acc: 0.8641
val Loss: 0.0058 Acc: 0.8038

Epoch 7/78
----------
train Loss: 0.0043 Acc: 0.8649
val Loss: 0.0058 Acc: 0.8019

Epoch 8/78
----------
train Loss: 0.0043 Acc: 0.8663
val Loss: 0.0058 Acc: 0.8029

Epoch 9/78
----------
train Loss: 0.0043 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8047

Epoch 10/78
----------
LR is set to 0.00010000000000000002
train Loss: 0.0042 Acc: 0.8734
val Loss: 0.0057 Acc: 0.8084

Epoch 11/78
----------
train Loss: 0.0042 Acc: 0.8756
val Loss: 0.0057 Acc: 0.8042

Epoch 12/78
----------
train Loss: 0.0042 Acc: 0.8722
val Loss: 0.0057 Acc: 0.8056

Epoch 13/78
----------
train Loss: 0.0042 Acc: 0.8705
val Loss: 0.0057 Acc: 0.8070

Epoch 14/78
----------
train Loss: 0.0042 Acc: 0.8754
val Loss: 0.0058 Acc: 0.8029

Epoch 15/78
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0041 Acc: 0.8730
val Loss: 0.0057 Acc: 0.8061

Epoch 16/78
----------
train Loss: 0.0042 Acc: 0.8712
val Loss: 0.0057 Acc: 0.8075

Epoch 17/78
----------
train Loss: 0.0042 Acc: 0.8728
val Loss: 0.0057 Acc: 0.8056

Epoch 18/78
----------
train Loss: 0.0042 Acc: 0.8706
val Loss: 0.0057 Acc: 0.8052

Epoch 19/78
----------
train Loss: 0.0042 Acc: 0.8724
val Loss: 0.0057 Acc: 0.8052

Epoch 20/78
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0042 Acc: 0.8671
val Loss: 0.0057 Acc: 0.8047

Epoch 21/78
----------
train Loss: 0.0042 Acc: 0.8734
val Loss: 0.0057 Acc: 0.8066

Epoch 22/78
----------
train Loss: 0.0042 Acc: 0.8720
val Loss: 0.0057 Acc: 0.8079

Epoch 23/78
----------
train Loss: 0.0041 Acc: 0.8754
val Loss: 0.0057 Acc: 0.8061

Epoch 24/78
----------
train Loss: 0.0041 Acc: 0.8718
val Loss: 0.0057 Acc: 0.8061

Epoch 25/78
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0042 Acc: 0.8675
val Loss: 0.0057 Acc: 0.8075

Epoch 26/78
----------
train Loss: 0.0042 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8052

Epoch 27/78
----------
train Loss: 0.0042 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8038

Epoch 28/78
----------
train Loss: 0.0042 Acc: 0.8649
val Loss: 0.0057 Acc: 0.8075

Epoch 29/78
----------
train Loss: 0.0042 Acc: 0.8714
val Loss: 0.0057 Acc: 0.8047

Epoch 30/78
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0041 Acc: 0.8695
val Loss: 0.0057 Acc: 0.8038

Epoch 31/78
----------
train Loss: 0.0042 Acc: 0.8720
val Loss: 0.0057 Acc: 0.8052

Epoch 32/78
----------
train Loss: 0.0042 Acc: 0.8689
val Loss: 0.0057 Acc: 0.8070

Epoch 33/78
----------
train Loss: 0.0041 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8052

Epoch 34/78
----------
train Loss: 0.0042 Acc: 0.8681
val Loss: 0.0057 Acc: 0.8047

Epoch 35/78
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0042 Acc: 0.8701
val Loss: 0.0057 Acc: 0.8042

Epoch 36/78
----------
train Loss: 0.0042 Acc: 0.8722
val Loss: 0.0057 Acc: 0.8029

Epoch 37/78
----------
train Loss: 0.0042 Acc: 0.8726
val Loss: 0.0057 Acc: 0.8066

Epoch 38/78
----------
train Loss: 0.0042 Acc: 0.8703
val Loss: 0.0057 Acc: 0.8042

Epoch 39/78
----------
train Loss: 0.0041 Acc: 0.8724
val Loss: 0.0057 Acc: 0.8066

Epoch 40/78
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0042 Acc: 0.8705
val Loss: 0.0057 Acc: 0.8042

Epoch 41/78
----------
train Loss: 0.0041 Acc: 0.8742
val Loss: 0.0057 Acc: 0.8061

Epoch 42/78
----------
train Loss: 0.0042 Acc: 0.8708
val Loss: 0.0057 Acc: 0.8052

Epoch 43/78
----------
train Loss: 0.0042 Acc: 0.8653
val Loss: 0.0057 Acc: 0.8061

Epoch 44/78
----------
train Loss: 0.0043 Acc: 0.8693
val Loss: 0.0057 Acc: 0.8042

Epoch 45/78
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0042 Acc: 0.8665
val Loss: 0.0057 Acc: 0.8084

Epoch 46/78
----------
train Loss: 0.0041 Acc: 0.8758
val Loss: 0.0057 Acc: 0.8084

Epoch 47/78
----------
train Loss: 0.0042 Acc: 0.8695
val Loss: 0.0057 Acc: 0.8056

Epoch 48/78
----------
train Loss: 0.0042 Acc: 0.8744
val Loss: 0.0057 Acc: 0.8075

Epoch 49/78
----------
train Loss: 0.0041 Acc: 0.8732
val Loss: 0.0057 Acc: 0.8010

Epoch 50/78
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0041 Acc: 0.8744
val Loss: 0.0057 Acc: 0.8084

Epoch 51/78
----------
train Loss: 0.0041 Acc: 0.8770
val Loss: 0.0057 Acc: 0.8070

Epoch 52/78
----------
train Loss: 0.0042 Acc: 0.8705
val Loss: 0.0057 Acc: 0.8066

Epoch 53/78
----------
train Loss: 0.0042 Acc: 0.8703
val Loss: 0.0057 Acc: 0.8089

Epoch 54/78
----------
train Loss: 0.0041 Acc: 0.8730
val Loss: 0.0057 Acc: 0.8056

Epoch 55/78
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0042 Acc: 0.8691
val Loss: 0.0057 Acc: 0.8075

Epoch 56/78
----------
train Loss: 0.0042 Acc: 0.8712
val Loss: 0.0057 Acc: 0.8084

Epoch 57/78
----------
train Loss: 0.0042 Acc: 0.8683
val Loss: 0.0058 Acc: 0.8052

Epoch 58/78
----------
train Loss: 0.0042 Acc: 0.8730
val Loss: 0.0057 Acc: 0.8070

Epoch 59/78
----------
train Loss: 0.0042 Acc: 0.8752
val Loss: 0.0057 Acc: 0.8075

Epoch 60/78
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0042 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8066

Epoch 61/78
----------
train Loss: 0.0042 Acc: 0.8716
val Loss: 0.0057 Acc: 0.8024

Epoch 62/78
----------
train Loss: 0.0042 Acc: 0.8697
val Loss: 0.0057 Acc: 0.8038

Epoch 63/78
----------
train Loss: 0.0042 Acc: 0.8665
val Loss: 0.0057 Acc: 0.8079

Epoch 64/78
----------
train Loss: 0.0042 Acc: 0.8693
val Loss: 0.0057 Acc: 0.8056

Epoch 65/78
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0042 Acc: 0.8708
val Loss: 0.0057 Acc: 0.8042

Epoch 66/78
----------
train Loss: 0.0041 Acc: 0.8732
val Loss: 0.0057 Acc: 0.8061

Epoch 67/78
----------
train Loss: 0.0042 Acc: 0.8712
val Loss: 0.0057 Acc: 0.8079

Epoch 68/78
----------
train Loss: 0.0042 Acc: 0.8679
val Loss: 0.0057 Acc: 0.8061

Epoch 69/78
----------
train Loss: 0.0042 Acc: 0.8706
val Loss: 0.0057 Acc: 0.8075

Epoch 70/78
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0042 Acc: 0.8703
val Loss: 0.0057 Acc: 0.8070

Epoch 71/78
----------
train Loss: 0.0042 Acc: 0.8736
val Loss: 0.0057 Acc: 0.8047

Epoch 72/78
----------
train Loss: 0.0042 Acc: 0.8706
val Loss: 0.0057 Acc: 0.8056

Epoch 73/78
----------
train Loss: 0.0041 Acc: 0.8712
val Loss: 0.0057 Acc: 0.8047

Epoch 74/78
----------
train Loss: 0.0042 Acc: 0.8708
val Loss: 0.0057 Acc: 0.8047

Epoch 75/78
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0041 Acc: 0.8772
val Loss: 0.0057 Acc: 0.8061

Epoch 76/78
----------
train Loss: 0.0042 Acc: 0.8691
val Loss: 0.0057 Acc: 0.8038

Epoch 77/78
----------
train Loss: 0.0041 Acc: 0.8726
val Loss: 0.0057 Acc: 0.8084

Epoch 78/78
----------
train Loss: 0.0042 Acc: 0.8701
val Loss: 0.0058 Acc: 0.8070

Best val Acc: 0.808864

---Fine tuning.---
Epoch 0/78
----------
LR is set to 0.01
train Loss: 0.0045 Acc: 0.8528
val Loss: 0.0055 Acc: 0.8163

Epoch 1/78
----------
train Loss: 0.0020 Acc: 0.9345
val Loss: 0.0041 Acc: 0.8730

Epoch 2/78
----------
train Loss: 0.0010 Acc: 0.9688
val Loss: 0.0039 Acc: 0.8901

Epoch 3/78
----------
train Loss: 0.0006 Acc: 0.9777
val Loss: 0.0035 Acc: 0.8984

Epoch 4/78
----------
train Loss: 0.0004 Acc: 0.9852
val Loss: 0.0035 Acc: 0.8984

Epoch 5/78
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0034 Acc: 0.9017

Epoch 6/78
----------
train Loss: 0.0002 Acc: 0.9887
val Loss: 0.0034 Acc: 0.9035

Epoch 7/78
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9054

Epoch 8/78
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0034 Acc: 0.9054

Epoch 9/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9054

Epoch 10/78
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0033 Acc: 0.9054

Epoch 11/78
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0034 Acc: 0.9049

Epoch 12/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0033 Acc: 0.9063

Epoch 13/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9054

Epoch 14/78
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0033 Acc: 0.9067

Epoch 15/78
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0034 Acc: 0.9058

Epoch 16/78
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0034 Acc: 0.9049

Epoch 17/78
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0033 Acc: 0.9058

Epoch 18/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9049

Epoch 19/78
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0034 Acc: 0.9058

Epoch 20/78
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9058

Epoch 21/78
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0034 Acc: 0.9058

Epoch 22/78
----------
train Loss: 0.0002 Acc: 0.9933
val Loss: 0.0034 Acc: 0.9049

Epoch 23/78
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9054

Epoch 24/78
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0034 Acc: 0.9054

Epoch 25/78
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0034 Acc: 0.9072

Epoch 26/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9035

Epoch 27/78
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0034 Acc: 0.9063

Epoch 28/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9054

Epoch 29/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0033 Acc: 0.9054

Epoch 30/78
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0034 Acc: 0.9054

Epoch 31/78
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0034 Acc: 0.9058

Epoch 32/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0034 Acc: 0.9044

Epoch 33/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9054

Epoch 34/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9049

Epoch 35/78
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0034 Acc: 0.9063

Epoch 36/78
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0034 Acc: 0.9054

Epoch 37/78
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0034 Acc: 0.9049

Epoch 38/78
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0034 Acc: 0.9058

Epoch 39/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0034 Acc: 0.9049

Epoch 40/78
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0034 Acc: 0.9044

Epoch 41/78
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0034 Acc: 0.9044

Epoch 42/78
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0033 Acc: 0.9058

Epoch 43/78
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0034 Acc: 0.9058

Epoch 44/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9058

Epoch 45/78
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0034 Acc: 0.9054

Epoch 46/78
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0034 Acc: 0.9054

Epoch 47/78
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0033 Acc: 0.9054

Epoch 48/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9044

Epoch 49/78
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0034 Acc: 0.9054

Epoch 50/78
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0033 Acc: 0.9058

Epoch 51/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0033 Acc: 0.9063

Epoch 52/78
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0034 Acc: 0.9058

Epoch 53/78
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9058

Epoch 54/78
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0034 Acc: 0.9058

Epoch 55/78
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0034 Acc: 0.9054

Epoch 56/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9054

Epoch 57/78
----------
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0034 Acc: 0.9063

Epoch 58/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0034 Acc: 0.9049

Epoch 59/78
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0034 Acc: 0.9063

Epoch 60/78
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9044

Epoch 61/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9063

Epoch 62/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0034 Acc: 0.9063

Epoch 63/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9049

Epoch 64/78
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9054

Epoch 65/78
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0034 Acc: 0.9067

Epoch 66/78
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0034 Acc: 0.9067

Epoch 67/78
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0033 Acc: 0.9044

Epoch 68/78
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9063

Epoch 69/78
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0034 Acc: 0.9063

Epoch 70/78
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9058

Epoch 71/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0034 Acc: 0.9054

Epoch 72/78
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9072

Epoch 73/78
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0034 Acc: 0.9049

Epoch 74/78
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0034 Acc: 0.9044

Epoch 75/78
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0034 Acc: 0.9054

Epoch 76/78
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0033 Acc: 0.9058

Epoch 77/78
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0033 Acc: 0.9063

Epoch 78/78
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0033 Acc: 0.9054

Best val Acc: 0.907202

---Testing---
Test accuracy: 0.966214
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 97 %
Accuracy of Batoidea(ga_oo_lee) : 95 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 96 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 98 %
Accuracy of holocephalan : 89 %
Accuracy of mullet : 38 %
Accuracy of   ray : 78 %
Accuracy of rough : 82 %
Accuracy of shark : 98 %
Accuracy of tuna_fish : 99 %
mean: 0.9087354987977762, std: 0.15810502484995997

Model saved in "./weights/super_with_most_[0.98]_mean[0.94]_std[0.14].save".
--------------------

run info[val: 0.1, epoch: 86, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0118 Acc: 0.6386
val Loss: 0.0082 Acc: 0.7715

Epoch 1/85
----------
train Loss: 0.0068 Acc: 0.7697
val Loss: 0.0069 Acc: 0.7922

Epoch 2/85
----------
train Loss: 0.0058 Acc: 0.8048
val Loss: 0.0070 Acc: 0.7936

Epoch 3/85
----------
train Loss: 0.0052 Acc: 0.8251
val Loss: 0.0059 Acc: 0.8324

Epoch 4/85
----------
train Loss: 0.0049 Acc: 0.8375
val Loss: 0.0056 Acc: 0.8352

Epoch 5/85
----------
train Loss: 0.0047 Acc: 0.8434
val Loss: 0.0058 Acc: 0.8352

Epoch 6/85
----------
train Loss: 0.0044 Acc: 0.8482
val Loss: 0.0059 Acc: 0.8116

Epoch 7/85
----------
train Loss: 0.0042 Acc: 0.8615
val Loss: 0.0054 Acc: 0.8338

Epoch 8/85
----------
train Loss: 0.0041 Acc: 0.8634
val Loss: 0.0053 Acc: 0.8296

Epoch 9/85
----------
LR is set to 0.001
train Loss: 0.0036 Acc: 0.8815
val Loss: 0.0049 Acc: 0.8476

Epoch 10/85
----------
train Loss: 0.0036 Acc: 0.8805
val Loss: 0.0048 Acc: 0.8407

Epoch 11/85
----------
train Loss: 0.0036 Acc: 0.8878
val Loss: 0.0049 Acc: 0.8407

Epoch 12/85
----------
train Loss: 0.0036 Acc: 0.8872
val Loss: 0.0048 Acc: 0.8435

Epoch 13/85
----------
train Loss: 0.0035 Acc: 0.8846
val Loss: 0.0051 Acc: 0.8393

Epoch 14/85
----------
train Loss: 0.0036 Acc: 0.8828
val Loss: 0.0052 Acc: 0.8463

Epoch 15/85
----------
train Loss: 0.0036 Acc: 0.8849
val Loss: 0.0051 Acc: 0.8449

Epoch 16/85
----------
train Loss: 0.0035 Acc: 0.8882
val Loss: 0.0049 Acc: 0.8407

Epoch 17/85
----------
train Loss: 0.0034 Acc: 0.8888
val Loss: 0.0051 Acc: 0.8463

Epoch 18/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0035 Acc: 0.8848
val Loss: 0.0049 Acc: 0.8435

Epoch 19/85
----------
train Loss: 0.0034 Acc: 0.8917
val Loss: 0.0049 Acc: 0.8421

Epoch 20/85
----------
train Loss: 0.0035 Acc: 0.8888
val Loss: 0.0046 Acc: 0.8463

Epoch 21/85
----------
train Loss: 0.0035 Acc: 0.8871
val Loss: 0.0051 Acc: 0.8407

Epoch 22/85
----------
train Loss: 0.0035 Acc: 0.8865
val Loss: 0.0049 Acc: 0.8476

Epoch 23/85
----------
train Loss: 0.0035 Acc: 0.8909
val Loss: 0.0046 Acc: 0.8421

Epoch 24/85
----------
train Loss: 0.0035 Acc: 0.8914
val Loss: 0.0048 Acc: 0.8435

Epoch 25/85
----------
train Loss: 0.0035 Acc: 0.8889
val Loss: 0.0052 Acc: 0.8449

Epoch 26/85
----------
train Loss: 0.0035 Acc: 0.8846
val Loss: 0.0050 Acc: 0.8435

Epoch 27/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0034 Acc: 0.8945
val Loss: 0.0049 Acc: 0.8449

Epoch 28/85
----------
train Loss: 0.0035 Acc: 0.8863
val Loss: 0.0048 Acc: 0.8463

Epoch 29/85
----------
train Loss: 0.0035 Acc: 0.8865
val Loss: 0.0049 Acc: 0.8449

Epoch 30/85
----------
train Loss: 0.0035 Acc: 0.8852
val Loss: 0.0054 Acc: 0.8449

Epoch 31/85
----------
train Loss: 0.0035 Acc: 0.8889
val Loss: 0.0055 Acc: 0.8463

Epoch 32/85
----------
train Loss: 0.0034 Acc: 0.8934
val Loss: 0.0050 Acc: 0.8407

Epoch 33/85
----------
train Loss: 0.0035 Acc: 0.8878
val Loss: 0.0051 Acc: 0.8435

Epoch 34/85
----------
train Loss: 0.0035 Acc: 0.8892
val Loss: 0.0049 Acc: 0.8435

Epoch 35/85
----------
train Loss: 0.0035 Acc: 0.8822
val Loss: 0.0050 Acc: 0.8449

Epoch 36/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0034 Acc: 0.8905
val Loss: 0.0050 Acc: 0.8449

Epoch 37/85
----------
train Loss: 0.0035 Acc: 0.8882
val Loss: 0.0052 Acc: 0.8435

Epoch 38/85
----------
train Loss: 0.0035 Acc: 0.8880
val Loss: 0.0048 Acc: 0.8407

Epoch 39/85
----------
train Loss: 0.0035 Acc: 0.8912
val Loss: 0.0049 Acc: 0.8476

Epoch 40/85
----------
train Loss: 0.0035 Acc: 0.8857
val Loss: 0.0052 Acc: 0.8435

Epoch 41/85
----------
train Loss: 0.0035 Acc: 0.8878
val Loss: 0.0049 Acc: 0.8463

Epoch 42/85
----------
train Loss: 0.0035 Acc: 0.8849
val Loss: 0.0051 Acc: 0.8449

Epoch 43/85
----------
train Loss: 0.0034 Acc: 0.8898
val Loss: 0.0050 Acc: 0.8435

Epoch 44/85
----------
train Loss: 0.0035 Acc: 0.8912
val Loss: 0.0048 Acc: 0.8435

Epoch 45/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0034 Acc: 0.8908
val Loss: 0.0050 Acc: 0.8421

Epoch 46/85
----------
train Loss: 0.0035 Acc: 0.8868
val Loss: 0.0048 Acc: 0.8421

Epoch 47/85
----------
train Loss: 0.0034 Acc: 0.8894
val Loss: 0.0048 Acc: 0.8421

Epoch 48/85
----------
train Loss: 0.0035 Acc: 0.8882
val Loss: 0.0054 Acc: 0.8476

Epoch 49/85
----------
train Loss: 0.0035 Acc: 0.8868
val Loss: 0.0049 Acc: 0.8435

Epoch 50/85
----------
train Loss: 0.0034 Acc: 0.8877
val Loss: 0.0049 Acc: 0.8449

Epoch 51/85
----------
train Loss: 0.0034 Acc: 0.8905
val Loss: 0.0049 Acc: 0.8449

Epoch 52/85
----------
train Loss: 0.0034 Acc: 0.8915
val Loss: 0.0050 Acc: 0.8504

Epoch 53/85
----------
train Loss: 0.0035 Acc: 0.8868
val Loss: 0.0051 Acc: 0.8421

Epoch 54/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0034 Acc: 0.8892
val Loss: 0.0047 Acc: 0.8449

Epoch 55/85
----------
train Loss: 0.0035 Acc: 0.8891
val Loss: 0.0050 Acc: 0.8476

Epoch 56/85
----------
train Loss: 0.0034 Acc: 0.8925
val Loss: 0.0047 Acc: 0.8449

Epoch 57/85
----------
train Loss: 0.0034 Acc: 0.8909
val Loss: 0.0049 Acc: 0.8476

Epoch 58/85
----------
train Loss: 0.0034 Acc: 0.8903
val Loss: 0.0049 Acc: 0.8463

Epoch 59/85
----------
train Loss: 0.0034 Acc: 0.8934
val Loss: 0.0048 Acc: 0.8435

Epoch 60/85
----------
train Loss: 0.0035 Acc: 0.8895
val Loss: 0.0048 Acc: 0.8476

Epoch 61/85
----------
train Loss: 0.0034 Acc: 0.8892
val Loss: 0.0049 Acc: 0.8421

Epoch 62/85
----------
train Loss: 0.0035 Acc: 0.8900
val Loss: 0.0050 Acc: 0.8435

Epoch 63/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0034 Acc: 0.8914
val Loss: 0.0047 Acc: 0.8449

Epoch 64/85
----------
train Loss: 0.0035 Acc: 0.8863
val Loss: 0.0051 Acc: 0.8407

Epoch 65/85
----------
train Loss: 0.0035 Acc: 0.8858
val Loss: 0.0051 Acc: 0.8449

Epoch 66/85
----------
train Loss: 0.0034 Acc: 0.8874
val Loss: 0.0049 Acc: 0.8463

Epoch 67/85
----------
train Loss: 0.0035 Acc: 0.8885
val Loss: 0.0052 Acc: 0.8449

Epoch 68/85
----------
train Loss: 0.0034 Acc: 0.8917
val Loss: 0.0050 Acc: 0.8421

Epoch 69/85
----------
train Loss: 0.0034 Acc: 0.8900
val Loss: 0.0048 Acc: 0.8476

Epoch 70/85
----------
train Loss: 0.0034 Acc: 0.8866
val Loss: 0.0048 Acc: 0.8449

Epoch 71/85
----------
train Loss: 0.0035 Acc: 0.8906
val Loss: 0.0049 Acc: 0.8435

Epoch 72/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0035 Acc: 0.8926
val Loss: 0.0047 Acc: 0.8421

Epoch 73/85
----------
train Loss: 0.0035 Acc: 0.8923
val Loss: 0.0050 Acc: 0.8435

Epoch 74/85
----------
train Loss: 0.0034 Acc: 0.8877
val Loss: 0.0047 Acc: 0.8449

Epoch 75/85
----------
train Loss: 0.0035 Acc: 0.8849
val Loss: 0.0055 Acc: 0.8476

Epoch 76/85
----------
train Loss: 0.0035 Acc: 0.8885
val Loss: 0.0050 Acc: 0.8463

Epoch 77/85
----------
train Loss: 0.0035 Acc: 0.8883
val Loss: 0.0051 Acc: 0.8449

Epoch 78/85
----------
train Loss: 0.0034 Acc: 0.8866
val Loss: 0.0049 Acc: 0.8435

Epoch 79/85
----------
train Loss: 0.0034 Acc: 0.8905
val Loss: 0.0049 Acc: 0.8449

Epoch 80/85
----------
train Loss: 0.0035 Acc: 0.8880
val Loss: 0.0048 Acc: 0.8421

Epoch 81/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0034 Acc: 0.8923
val Loss: 0.0050 Acc: 0.8463

Epoch 82/85
----------
train Loss: 0.0034 Acc: 0.8922
val Loss: 0.0050 Acc: 0.8449

Epoch 83/85
----------
train Loss: 0.0035 Acc: 0.8871
val Loss: 0.0053 Acc: 0.8435

Epoch 84/85
----------
train Loss: 0.0034 Acc: 0.8908
val Loss: 0.0051 Acc: 0.8393

Epoch 85/85
----------
train Loss: 0.0035 Acc: 0.8906
val Loss: 0.0053 Acc: 0.8435

Best val Acc: 0.850416

---Fine tuning.---
Epoch 0/85
----------
LR is set to 0.01
train Loss: 0.0044 Acc: 0.8560
val Loss: 0.0052 Acc: 0.8310

Epoch 1/85
----------
train Loss: 0.0020 Acc: 0.9314
val Loss: 0.0044 Acc: 0.8781

Epoch 2/85
----------
train Loss: 0.0011 Acc: 0.9600
val Loss: 0.0043 Acc: 0.9030

Epoch 3/85
----------
train Loss: 0.0006 Acc: 0.9775
val Loss: 0.0034 Acc: 0.9127

Epoch 4/85
----------
train Loss: 0.0004 Acc: 0.9805
val Loss: 0.0038 Acc: 0.9044

Epoch 5/85
----------
train Loss: 0.0004 Acc: 0.9828
val Loss: 0.0034 Acc: 0.9155

Epoch 6/85
----------
train Loss: 0.0003 Acc: 0.9831
val Loss: 0.0039 Acc: 0.9183

Epoch 7/85
----------
train Loss: 0.0003 Acc: 0.9851
val Loss: 0.0036 Acc: 0.9141

Epoch 8/85
----------
train Loss: 0.0002 Acc: 0.9871
val Loss: 0.0034 Acc: 0.9224

Epoch 9/85
----------
LR is set to 0.001
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0033 Acc: 0.9197

Epoch 10/85
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0034 Acc: 0.9211

Epoch 11/85
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0033 Acc: 0.9197

Epoch 12/85
----------
train Loss: 0.0002 Acc: 0.9871
val Loss: 0.0034 Acc: 0.9183

Epoch 13/85
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0038 Acc: 0.9211

Epoch 14/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0041 Acc: 0.9197

Epoch 15/85
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0036 Acc: 0.9211

Epoch 16/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0039 Acc: 0.9211

Epoch 17/85
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0036 Acc: 0.9197

Epoch 18/85
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9211

Epoch 19/85
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0034 Acc: 0.9197

Epoch 20/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0035 Acc: 0.9197

Epoch 21/85
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0048 Acc: 0.9197

Epoch 22/85
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0037 Acc: 0.9211

Epoch 23/85
----------
train Loss: 0.0002 Acc: 0.9868
val Loss: 0.0035 Acc: 0.9197

Epoch 24/85
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0034 Acc: 0.9197

Epoch 25/85
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0039 Acc: 0.9197

Epoch 26/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0035 Acc: 0.9197

Epoch 27/85
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9863
val Loss: 0.0041 Acc: 0.9197

Epoch 28/85
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0035 Acc: 0.9197

Epoch 29/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0036 Acc: 0.9197

Epoch 30/85
----------
train Loss: 0.0002 Acc: 0.9871
val Loss: 0.0036 Acc: 0.9211

Epoch 31/85
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0040 Acc: 0.9211

Epoch 32/85
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0049 Acc: 0.9211

Epoch 33/85
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0037 Acc: 0.9211

Epoch 34/85
----------
train Loss: 0.0002 Acc: 0.9871
val Loss: 0.0035 Acc: 0.9211

Epoch 35/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0046 Acc: 0.9211

Epoch 36/85
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0034 Acc: 0.9197

Epoch 37/85
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0040 Acc: 0.9211

Epoch 38/85
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0035 Acc: 0.9197

Epoch 39/85
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0035 Acc: 0.9197

Epoch 40/85
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0040 Acc: 0.9211

Epoch 41/85
----------
train Loss: 0.0002 Acc: 0.9868
val Loss: 0.0034 Acc: 0.9211

Epoch 42/85
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0035 Acc: 0.9197

Epoch 43/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0034 Acc: 0.9211

Epoch 44/85
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9211

Epoch 45/85
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0042 Acc: 0.9197

Epoch 46/85
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0040 Acc: 0.9197

Epoch 47/85
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9211

Epoch 48/85
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9211

Epoch 49/85
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0036 Acc: 0.9197

Epoch 50/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0039 Acc: 0.9211

Epoch 51/85
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0036 Acc: 0.9211

Epoch 52/85
----------
train Loss: 0.0002 Acc: 0.9869
val Loss: 0.0034 Acc: 0.9224

Epoch 53/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0034 Acc: 0.9197

Epoch 54/85
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0037 Acc: 0.9211

Epoch 55/85
----------
train Loss: 0.0002 Acc: 0.9866
val Loss: 0.0047 Acc: 0.9197

Epoch 56/85
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0042 Acc: 0.9211

Epoch 57/85
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0040 Acc: 0.9211

Epoch 58/85
----------
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0034 Acc: 0.9197

Epoch 59/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0038 Acc: 0.9224

Epoch 60/85
----------
train Loss: 0.0002 Acc: 0.9869
val Loss: 0.0035 Acc: 0.9197

Epoch 61/85
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0034 Acc: 0.9197

Epoch 62/85
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0037 Acc: 0.9197

Epoch 63/85
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0039 Acc: 0.9197

Epoch 64/85
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0035 Acc: 0.9211

Epoch 65/85
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0038 Acc: 0.9197

Epoch 66/85
----------
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0037 Acc: 0.9211

Epoch 67/85
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9197

Epoch 68/85
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0040 Acc: 0.9211

Epoch 69/85
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0034 Acc: 0.9211

Epoch 70/85
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0039 Acc: 0.9197

Epoch 71/85
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0039 Acc: 0.9211

Epoch 72/85
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0034 Acc: 0.9211

Epoch 73/85
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0040 Acc: 0.9197

Epoch 74/85
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0037 Acc: 0.9197

Epoch 75/85
----------
train Loss: 0.0002 Acc: 0.9872
val Loss: 0.0038 Acc: 0.9197

Epoch 76/85
----------
train Loss: 0.0002 Acc: 0.9877
val Loss: 0.0040 Acc: 0.9211

Epoch 77/85
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0044 Acc: 0.9197

Epoch 78/85
----------
train Loss: 0.0002 Acc: 0.9892
val Loss: 0.0040 Acc: 0.9197

Epoch 79/85
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0042 Acc: 0.9197

Epoch 80/85
----------
train Loss: 0.0002 Acc: 0.9878
val Loss: 0.0038 Acc: 0.9211

Epoch 81/85
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9902
val Loss: 0.0034 Acc: 0.9211

Epoch 82/85
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0036 Acc: 0.9197

Epoch 83/85
----------
train Loss: 0.0002 Acc: 0.9885
val Loss: 0.0034 Acc: 0.9197

Epoch 84/85
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9211

Epoch 85/85
----------
train Loss: 0.0002 Acc: 0.9880
val Loss: 0.0038 Acc: 0.9211

Best val Acc: 0.922438

---Testing---
Test accuracy: 0.981861
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 99 %
Accuracy of Batoidea(ga_oo_lee) : 99 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 100 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 96 %
Accuracy of mullet :  9 %
Accuracy of   ray : 94 %
Accuracy of rough : 93 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.920331978172955, std: 0.23010555609853495
--------------------

run info[val: 0.2, epoch: 74, randcrop: True, decay: 10]

---Training last layer.---
Epoch 0/73
----------
LR is set to 0.01
train Loss: 0.0125 Acc: 0.6118
val Loss: 0.0094 Acc: 0.7064

Epoch 1/73
----------
train Loss: 0.0077 Acc: 0.7413
val Loss: 0.0078 Acc: 0.7327

Epoch 2/73
----------
train Loss: 0.0065 Acc: 0.7754
val Loss: 0.0070 Acc: 0.7639

Epoch 3/73
----------
train Loss: 0.0061 Acc: 0.7908
val Loss: 0.0065 Acc: 0.7791

Epoch 4/73
----------
train Loss: 0.0057 Acc: 0.8034
val Loss: 0.0066 Acc: 0.7839

Epoch 5/73
----------
train Loss: 0.0055 Acc: 0.8141
val Loss: 0.0061 Acc: 0.7950

Epoch 6/73
----------
train Loss: 0.0052 Acc: 0.8221
val Loss: 0.0065 Acc: 0.7839

Epoch 7/73
----------
train Loss: 0.0053 Acc: 0.8143
val Loss: 0.0059 Acc: 0.8012

Epoch 8/73
----------
train Loss: 0.0050 Acc: 0.8250
val Loss: 0.0059 Acc: 0.8006

Epoch 9/73
----------
train Loss: 0.0048 Acc: 0.8382
val Loss: 0.0062 Acc: 0.8068

Epoch 10/73
----------
LR is set to 0.001
train Loss: 0.0044 Acc: 0.8513
val Loss: 0.0056 Acc: 0.8102

Epoch 11/73
----------
train Loss: 0.0042 Acc: 0.8579
val Loss: 0.0055 Acc: 0.8102

Epoch 12/73
----------
train Loss: 0.0044 Acc: 0.8525
val Loss: 0.0055 Acc: 0.8158

Epoch 13/73
----------
train Loss: 0.0043 Acc: 0.8593
val Loss: 0.0056 Acc: 0.8144

Epoch 14/73
----------
train Loss: 0.0043 Acc: 0.8555
val Loss: 0.0055 Acc: 0.8165

Epoch 15/73
----------
train Loss: 0.0043 Acc: 0.8610
val Loss: 0.0056 Acc: 0.8179

Epoch 16/73
----------
train Loss: 0.0043 Acc: 0.8558
val Loss: 0.0054 Acc: 0.8199

Epoch 17/73
----------
train Loss: 0.0043 Acc: 0.8581
val Loss: 0.0055 Acc: 0.8199

Epoch 18/73
----------
train Loss: 0.0043 Acc: 0.8564
val Loss: 0.0055 Acc: 0.8193

Epoch 19/73
----------
train Loss: 0.0041 Acc: 0.8641
val Loss: 0.0056 Acc: 0.8109

Epoch 20/73
----------
LR is set to 0.00010000000000000002
train Loss: 0.0043 Acc: 0.8558
val Loss: 0.0055 Acc: 0.8144

Epoch 21/73
----------
train Loss: 0.0042 Acc: 0.8622
val Loss: 0.0055 Acc: 0.8137

Epoch 22/73
----------
train Loss: 0.0042 Acc: 0.8603
val Loss: 0.0055 Acc: 0.8199

Epoch 23/73
----------
train Loss: 0.0042 Acc: 0.8569
val Loss: 0.0055 Acc: 0.8151

Epoch 24/73
----------
train Loss: 0.0043 Acc: 0.8576
val Loss: 0.0055 Acc: 0.8172

Epoch 25/73
----------
train Loss: 0.0042 Acc: 0.8617
val Loss: 0.0054 Acc: 0.8186

Epoch 26/73
----------
train Loss: 0.0042 Acc: 0.8574
val Loss: 0.0055 Acc: 0.8116

Epoch 27/73
----------
train Loss: 0.0042 Acc: 0.8628
val Loss: 0.0054 Acc: 0.8158

Epoch 28/73
----------
train Loss: 0.0042 Acc: 0.8610
val Loss: 0.0055 Acc: 0.8179

Epoch 29/73
----------
train Loss: 0.0042 Acc: 0.8648
val Loss: 0.0055 Acc: 0.8116

Epoch 30/73
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0042 Acc: 0.8615
val Loss: 0.0054 Acc: 0.8151

Epoch 31/73
----------
train Loss: 0.0041 Acc: 0.8643
val Loss: 0.0054 Acc: 0.8193

Epoch 32/73
----------
train Loss: 0.0042 Acc: 0.8609
val Loss: 0.0055 Acc: 0.8151

Epoch 33/73
----------
train Loss: 0.0041 Acc: 0.8626
val Loss: 0.0054 Acc: 0.8179

Epoch 34/73
----------
train Loss: 0.0042 Acc: 0.8574
val Loss: 0.0054 Acc: 0.8199

Epoch 35/73
----------
train Loss: 0.0041 Acc: 0.8593
val Loss: 0.0055 Acc: 0.8158

Epoch 36/73
----------
train Loss: 0.0041 Acc: 0.8654
val Loss: 0.0055 Acc: 0.8165

Epoch 37/73
----------
train Loss: 0.0041 Acc: 0.8612
val Loss: 0.0055 Acc: 0.8172

Epoch 38/73
----------
train Loss: 0.0041 Acc: 0.8624
val Loss: 0.0054 Acc: 0.8165

Epoch 39/73
----------
train Loss: 0.0042 Acc: 0.8634
val Loss: 0.0055 Acc: 0.8144

Epoch 40/73
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0042 Acc: 0.8586
val Loss: 0.0054 Acc: 0.8144

Epoch 41/73
----------
train Loss: 0.0042 Acc: 0.8576
val Loss: 0.0055 Acc: 0.8144

Epoch 42/73
----------
train Loss: 0.0041 Acc: 0.8628
val Loss: 0.0055 Acc: 0.8165

Epoch 43/73
----------
train Loss: 0.0042 Acc: 0.8598
val Loss: 0.0055 Acc: 0.8165

Epoch 44/73
----------
train Loss: 0.0042 Acc: 0.8612
val Loss: 0.0054 Acc: 0.8151

Epoch 45/73
----------
train Loss: 0.0041 Acc: 0.8609
val Loss: 0.0054 Acc: 0.8144

Epoch 46/73
----------
train Loss: 0.0042 Acc: 0.8631
val Loss: 0.0055 Acc: 0.8123

Epoch 47/73
----------
train Loss: 0.0042 Acc: 0.8577
val Loss: 0.0055 Acc: 0.8206

Epoch 48/73
----------
train Loss: 0.0041 Acc: 0.8673
val Loss: 0.0055 Acc: 0.8172

Epoch 49/73
----------
train Loss: 0.0042 Acc: 0.8589
val Loss: 0.0054 Acc: 0.8186

Epoch 50/73
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0041 Acc: 0.8591
val Loss: 0.0054 Acc: 0.8151

Epoch 51/73
----------
train Loss: 0.0042 Acc: 0.8576
val Loss: 0.0054 Acc: 0.8137

Epoch 52/73
----------
train Loss: 0.0042 Acc: 0.8600
val Loss: 0.0054 Acc: 0.8179

Epoch 53/73
----------
train Loss: 0.0042 Acc: 0.8570
val Loss: 0.0054 Acc: 0.8193

Epoch 54/73
----------
train Loss: 0.0042 Acc: 0.8636
val Loss: 0.0055 Acc: 0.8151

Epoch 55/73
----------
train Loss: 0.0043 Acc: 0.8588
val Loss: 0.0054 Acc: 0.8165

Epoch 56/73
----------
train Loss: 0.0041 Acc: 0.8695
val Loss: 0.0054 Acc: 0.8165

Epoch 57/73
----------
train Loss: 0.0042 Acc: 0.8596
val Loss: 0.0054 Acc: 0.8206

Epoch 58/73
----------
train Loss: 0.0042 Acc: 0.8614
val Loss: 0.0054 Acc: 0.8186

Epoch 59/73
----------
train Loss: 0.0042 Acc: 0.8626
val Loss: 0.0055 Acc: 0.8179

Epoch 60/73
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0042 Acc: 0.8622
val Loss: 0.0054 Acc: 0.8213

Epoch 61/73
----------
train Loss: 0.0042 Acc: 0.8622
val Loss: 0.0054 Acc: 0.8220

Epoch 62/73
----------
train Loss: 0.0042 Acc: 0.8621
val Loss: 0.0054 Acc: 0.8179

Epoch 63/73
----------
train Loss: 0.0042 Acc: 0.8558
val Loss: 0.0054 Acc: 0.8172

Epoch 64/73
----------
train Loss: 0.0042 Acc: 0.8634
val Loss: 0.0055 Acc: 0.8158

Epoch 65/73
----------
train Loss: 0.0042 Acc: 0.8550
val Loss: 0.0054 Acc: 0.8206

Epoch 66/73
----------
train Loss: 0.0042 Acc: 0.8586
val Loss: 0.0055 Acc: 0.8158

Epoch 67/73
----------
train Loss: 0.0042 Acc: 0.8612
val Loss: 0.0055 Acc: 0.8158

Epoch 68/73
----------
train Loss: 0.0042 Acc: 0.8634
val Loss: 0.0056 Acc: 0.8193

Epoch 69/73
----------
train Loss: 0.0041 Acc: 0.8622
val Loss: 0.0054 Acc: 0.8193

Epoch 70/73
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0041 Acc: 0.8612
val Loss: 0.0055 Acc: 0.8193

Epoch 71/73
----------
train Loss: 0.0041 Acc: 0.8621
val Loss: 0.0054 Acc: 0.8151

Epoch 72/73
----------
train Loss: 0.0042 Acc: 0.8596
val Loss: 0.0054 Acc: 0.8137

Epoch 73/73
----------
train Loss: 0.0042 Acc: 0.8564
val Loss: 0.0054 Acc: 0.8186

Best val Acc: 0.822022

---Fine tuning.---
Epoch 0/73
----------
LR is set to 0.01
train Loss: 0.0053 Acc: 0.8197
val Loss: 0.0081 Acc: 0.7770

Epoch 1/73
----------
train Loss: 0.0030 Acc: 0.8958
val Loss: 0.0046 Acc: 0.8774

Epoch 2/73
----------
train Loss: 0.0017 Acc: 0.9396
val Loss: 0.0037 Acc: 0.9030

Epoch 3/73
----------
train Loss: 0.0011 Acc: 0.9640
val Loss: 0.0038 Acc: 0.8885

Epoch 4/73
----------
train Loss: 0.0009 Acc: 0.9687
val Loss: 0.0040 Acc: 0.8864

Epoch 5/73
----------
train Loss: 0.0006 Acc: 0.9753
val Loss: 0.0035 Acc: 0.9079

Epoch 6/73
----------
train Loss: 0.0006 Acc: 0.9784
val Loss: 0.0038 Acc: 0.9065

Epoch 7/73
----------
train Loss: 0.0005 Acc: 0.9801
val Loss: 0.0038 Acc: 0.9058

Epoch 8/73
----------
train Loss: 0.0004 Acc: 0.9829
val Loss: 0.0039 Acc: 0.9065

Epoch 9/73
----------
train Loss: 0.0004 Acc: 0.9817
val Loss: 0.0041 Acc: 0.9058

Epoch 10/73
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9870
val Loss: 0.0037 Acc: 0.9134

Epoch 11/73
----------
train Loss: 0.0003 Acc: 0.9874
val Loss: 0.0039 Acc: 0.9120

Epoch 12/73
----------
train Loss: 0.0003 Acc: 0.9879
val Loss: 0.0038 Acc: 0.9127

Epoch 13/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0039 Acc: 0.9100

Epoch 14/73
----------
train Loss: 0.0003 Acc: 0.9868
val Loss: 0.0039 Acc: 0.9114

Epoch 15/73
----------
train Loss: 0.0002 Acc: 0.9879
val Loss: 0.0039 Acc: 0.9100

Epoch 16/73
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0039 Acc: 0.9114

Epoch 17/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9127

Epoch 18/73
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0038 Acc: 0.9127

Epoch 19/73
----------
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0039 Acc: 0.9141

Epoch 20/73
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9875
val Loss: 0.0038 Acc: 0.9148

Epoch 21/73
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0038 Acc: 0.9134

Epoch 22/73
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0039 Acc: 0.9141

Epoch 23/73
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0038 Acc: 0.9155

Epoch 24/73
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0038 Acc: 0.9148

Epoch 25/73
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0038 Acc: 0.9127

Epoch 26/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0038 Acc: 0.9155

Epoch 27/73
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0038 Acc: 0.9134

Epoch 28/73
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0041 Acc: 0.9127

Epoch 29/73
----------
train Loss: 0.0002 Acc: 0.9888
val Loss: 0.0039 Acc: 0.9127

Epoch 30/73
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0038 Acc: 0.9120

Epoch 31/73
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0038 Acc: 0.9155

Epoch 32/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9134

Epoch 33/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0039 Acc: 0.9134

Epoch 34/73
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0038 Acc: 0.9141

Epoch 35/73
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0040 Acc: 0.9148

Epoch 36/73
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0038 Acc: 0.9134

Epoch 37/73
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0038 Acc: 0.9120

Epoch 38/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0038 Acc: 0.9127

Epoch 39/73
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9134

Epoch 40/73
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9874
val Loss: 0.0039 Acc: 0.9148

Epoch 41/73
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0038 Acc: 0.9134

Epoch 42/73
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9141

Epoch 43/73
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0039 Acc: 0.9148

Epoch 44/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0039 Acc: 0.9134

Epoch 45/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9141

Epoch 46/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9155

Epoch 47/73
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0039 Acc: 0.9155

Epoch 48/73
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0039 Acc: 0.9141

Epoch 49/73
----------
train Loss: 0.0002 Acc: 0.9882
val Loss: 0.0039 Acc: 0.9141

Epoch 50/73
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0039 Acc: 0.9134

Epoch 51/73
----------
train Loss: 0.0002 Acc: 0.9900
val Loss: 0.0038 Acc: 0.9134

Epoch 52/73
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0038 Acc: 0.9148

Epoch 53/73
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0038 Acc: 0.9155

Epoch 54/73
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0039 Acc: 0.9127

Epoch 55/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0038 Acc: 0.9127

Epoch 56/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0039 Acc: 0.9141

Epoch 57/73
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9141

Epoch 58/73
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0038 Acc: 0.9148

Epoch 59/73
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0040 Acc: 0.9148

Epoch 60/73
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0037 Acc: 0.9148

Epoch 61/73
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0038 Acc: 0.9134

Epoch 62/73
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0037 Acc: 0.9134

Epoch 63/73
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0037 Acc: 0.9141

Epoch 64/73
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0038 Acc: 0.9134

Epoch 65/73
----------
train Loss: 0.0002 Acc: 0.9881
val Loss: 0.0038 Acc: 0.9148

Epoch 66/73
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0039 Acc: 0.9148

Epoch 67/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0040 Acc: 0.9134

Epoch 68/73
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0039 Acc: 0.9127

Epoch 69/73
----------
train Loss: 0.0002 Acc: 0.9894
val Loss: 0.0040 Acc: 0.9141

Epoch 70/73
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0039 Acc: 0.9134

Epoch 71/73
----------
train Loss: 0.0002 Acc: 0.9898
val Loss: 0.0038 Acc: 0.9148

Epoch 72/73
----------
train Loss: 0.0002 Acc: 0.9886
val Loss: 0.0039 Acc: 0.9141

Epoch 73/73
----------
train Loss: 0.0002 Acc: 0.9896
val Loss: 0.0037 Acc: 0.9148

Best val Acc: 0.915512

---Testing---
Test accuracy: 0.975215
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 99 %
Accuracy of Batoidea(ga_oo_lee) : 98 %
Accuracy of   DOL : 100 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 90 %
Accuracy of mullet : 34 %
Accuracy of   ray : 83 %
Accuracy of rough : 83 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9168375814223682, std: 0.16931424231293632
--------------------

run info[val: 0.3, epoch: 83, randcrop: True, decay: 8]

---Training last layer.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0133 Acc: 0.6040
val Loss: 0.0095 Acc: 0.6962

Epoch 1/82
----------
train Loss: 0.0080 Acc: 0.7358
val Loss: 0.0074 Acc: 0.7613

Epoch 2/82
----------
train Loss: 0.0066 Acc: 0.7700
val Loss: 0.0067 Acc: 0.7613

Epoch 3/82
----------
train Loss: 0.0063 Acc: 0.7834
val Loss: 0.0065 Acc: 0.7812

Epoch 4/82
----------
train Loss: 0.0057 Acc: 0.8064
val Loss: 0.0064 Acc: 0.7881

Epoch 5/82
----------
train Loss: 0.0054 Acc: 0.8125
val Loss: 0.0066 Acc: 0.7669

Epoch 6/82
----------
train Loss: 0.0053 Acc: 0.8246
val Loss: 0.0063 Acc: 0.7802

Epoch 7/82
----------
train Loss: 0.0052 Acc: 0.8163
val Loss: 0.0060 Acc: 0.7973

Epoch 8/82
----------
LR is set to 0.001
train Loss: 0.0047 Acc: 0.8461
val Loss: 0.0058 Acc: 0.8015

Epoch 9/82
----------
train Loss: 0.0046 Acc: 0.8503
val Loss: 0.0058 Acc: 0.7982

Epoch 10/82
----------
train Loss: 0.0046 Acc: 0.8465
val Loss: 0.0057 Acc: 0.7978

Epoch 11/82
----------
train Loss: 0.0046 Acc: 0.8515
val Loss: 0.0057 Acc: 0.8019

Epoch 12/82
----------
train Loss: 0.0046 Acc: 0.8507
val Loss: 0.0058 Acc: 0.8001

Epoch 13/82
----------
train Loss: 0.0045 Acc: 0.8467
val Loss: 0.0057 Acc: 0.8015

Epoch 14/82
----------
train Loss: 0.0045 Acc: 0.8550
val Loss: 0.0057 Acc: 0.8019

Epoch 15/82
----------
train Loss: 0.0046 Acc: 0.8530
val Loss: 0.0057 Acc: 0.8066

Epoch 16/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0044 Acc: 0.8515
val Loss: 0.0057 Acc: 0.8029

Epoch 17/82
----------
train Loss: 0.0044 Acc: 0.8564
val Loss: 0.0057 Acc: 0.8047

Epoch 18/82
----------
train Loss: 0.0045 Acc: 0.8548
val Loss: 0.0057 Acc: 0.8015

Epoch 19/82
----------
train Loss: 0.0044 Acc: 0.8519
val Loss: 0.0056 Acc: 0.8042

Epoch 20/82
----------
train Loss: 0.0045 Acc: 0.8513
val Loss: 0.0057 Acc: 0.8033

Epoch 21/82
----------
train Loss: 0.0045 Acc: 0.8517
val Loss: 0.0057 Acc: 0.8024

Epoch 22/82
----------
train Loss: 0.0046 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8006

Epoch 23/82
----------
train Loss: 0.0044 Acc: 0.8548
val Loss: 0.0057 Acc: 0.8033

Epoch 24/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0044 Acc: 0.8625
val Loss: 0.0057 Acc: 0.8042

Epoch 25/82
----------
train Loss: 0.0044 Acc: 0.8558
val Loss: 0.0057 Acc: 0.8038

Epoch 26/82
----------
train Loss: 0.0045 Acc: 0.8564
val Loss: 0.0057 Acc: 0.8015

Epoch 27/82
----------
train Loss: 0.0045 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8015

Epoch 28/82
----------
train Loss: 0.0044 Acc: 0.8517
val Loss: 0.0056 Acc: 0.8070

Epoch 29/82
----------
train Loss: 0.0045 Acc: 0.8542
val Loss: 0.0056 Acc: 0.8033

Epoch 30/82
----------
train Loss: 0.0045 Acc: 0.8521
val Loss: 0.0056 Acc: 0.8029

Epoch 31/82
----------
train Loss: 0.0044 Acc: 0.8612
val Loss: 0.0057 Acc: 0.8052

Epoch 32/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0044 Acc: 0.8562
val Loss: 0.0057 Acc: 0.8047

Epoch 33/82
----------
train Loss: 0.0044 Acc: 0.8548
val Loss: 0.0056 Acc: 0.8047

Epoch 34/82
----------
train Loss: 0.0045 Acc: 0.8542
val Loss: 0.0056 Acc: 0.8024

Epoch 35/82
----------
train Loss: 0.0044 Acc: 0.8554
val Loss: 0.0057 Acc: 0.8029

Epoch 36/82
----------
train Loss: 0.0045 Acc: 0.8509
val Loss: 0.0057 Acc: 0.8042

Epoch 37/82
----------
train Loss: 0.0044 Acc: 0.8538
val Loss: 0.0057 Acc: 0.8019

Epoch 38/82
----------
train Loss: 0.0045 Acc: 0.8521
val Loss: 0.0057 Acc: 0.8019

Epoch 39/82
----------
train Loss: 0.0044 Acc: 0.8562
val Loss: 0.0057 Acc: 0.8029

Epoch 40/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0045 Acc: 0.8527
val Loss: 0.0057 Acc: 0.8052

Epoch 41/82
----------
train Loss: 0.0045 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8010

Epoch 42/82
----------
train Loss: 0.0045 Acc: 0.8538
val Loss: 0.0057 Acc: 0.8029

Epoch 43/82
----------
train Loss: 0.0045 Acc: 0.8489
val Loss: 0.0057 Acc: 0.8024

Epoch 44/82
----------
train Loss: 0.0044 Acc: 0.8548
val Loss: 0.0057 Acc: 0.8015

Epoch 45/82
----------
train Loss: 0.0045 Acc: 0.8507
val Loss: 0.0057 Acc: 0.8010

Epoch 46/82
----------
train Loss: 0.0045 Acc: 0.8471
val Loss: 0.0057 Acc: 0.8015

Epoch 47/82
----------
train Loss: 0.0045 Acc: 0.8556
val Loss: 0.0056 Acc: 0.8038

Epoch 48/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0044 Acc: 0.8521
val Loss: 0.0056 Acc: 0.8033

Epoch 49/82
----------
train Loss: 0.0044 Acc: 0.8525
val Loss: 0.0057 Acc: 0.8029

Epoch 50/82
----------
train Loss: 0.0044 Acc: 0.8536
val Loss: 0.0056 Acc: 0.8042

Epoch 51/82
----------
train Loss: 0.0044 Acc: 0.8544
val Loss: 0.0057 Acc: 0.7996

Epoch 52/82
----------
train Loss: 0.0045 Acc: 0.8540
val Loss: 0.0057 Acc: 0.8056

Epoch 53/82
----------
train Loss: 0.0045 Acc: 0.8558
val Loss: 0.0056 Acc: 0.8042

Epoch 54/82
----------
train Loss: 0.0045 Acc: 0.8532
val Loss: 0.0057 Acc: 0.8006

Epoch 55/82
----------
train Loss: 0.0044 Acc: 0.8542
val Loss: 0.0057 Acc: 0.8047

Epoch 56/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0044 Acc: 0.8576
val Loss: 0.0057 Acc: 0.8029

Epoch 57/82
----------
train Loss: 0.0044 Acc: 0.8594
val Loss: 0.0056 Acc: 0.8033

Epoch 58/82
----------
train Loss: 0.0045 Acc: 0.8525
val Loss: 0.0057 Acc: 0.8024

Epoch 59/82
----------
train Loss: 0.0045 Acc: 0.8499
val Loss: 0.0057 Acc: 0.8038

Epoch 60/82
----------
train Loss: 0.0046 Acc: 0.8467
val Loss: 0.0057 Acc: 0.8033

Epoch 61/82
----------
train Loss: 0.0045 Acc: 0.8493
val Loss: 0.0057 Acc: 0.8019

Epoch 62/82
----------
train Loss: 0.0044 Acc: 0.8560
val Loss: 0.0057 Acc: 0.8029

Epoch 63/82
----------
train Loss: 0.0044 Acc: 0.8580
val Loss: 0.0056 Acc: 0.8042

Epoch 64/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0044 Acc: 0.8511
val Loss: 0.0057 Acc: 0.7992

Epoch 65/82
----------
train Loss: 0.0045 Acc: 0.8491
val Loss: 0.0057 Acc: 0.8066

Epoch 66/82
----------
train Loss: 0.0045 Acc: 0.8501
val Loss: 0.0057 Acc: 0.8029

Epoch 67/82
----------
train Loss: 0.0044 Acc: 0.8566
val Loss: 0.0057 Acc: 0.8024

Epoch 68/82
----------
train Loss: 0.0045 Acc: 0.8505
val Loss: 0.0057 Acc: 0.8038

Epoch 69/82
----------
train Loss: 0.0045 Acc: 0.8499
val Loss: 0.0057 Acc: 0.8015

Epoch 70/82
----------
train Loss: 0.0044 Acc: 0.8570
val Loss: 0.0056 Acc: 0.8033

Epoch 71/82
----------
train Loss: 0.0044 Acc: 0.8562
val Loss: 0.0056 Acc: 0.8047

Epoch 72/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0044 Acc: 0.8546
val Loss: 0.0056 Acc: 0.8052

Epoch 73/82
----------
train Loss: 0.0044 Acc: 0.8582
val Loss: 0.0056 Acc: 0.8042

Epoch 74/82
----------
train Loss: 0.0044 Acc: 0.8546
val Loss: 0.0057 Acc: 0.7996

Epoch 75/82
----------
train Loss: 0.0045 Acc: 0.8546
val Loss: 0.0056 Acc: 0.8042

Epoch 76/82
----------
train Loss: 0.0045 Acc: 0.8538
val Loss: 0.0056 Acc: 0.8024

Epoch 77/82
----------
train Loss: 0.0045 Acc: 0.8489
val Loss: 0.0057 Acc: 0.7996

Epoch 78/82
----------
train Loss: 0.0045 Acc: 0.8548
val Loss: 0.0056 Acc: 0.7987

Epoch 79/82
----------
train Loss: 0.0045 Acc: 0.8554
val Loss: 0.0057 Acc: 0.8038

Epoch 80/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0044 Acc: 0.8538
val Loss: 0.0057 Acc: 0.8024

Epoch 81/82
----------
train Loss: 0.0045 Acc: 0.8540
val Loss: 0.0057 Acc: 0.8001

Epoch 82/82
----------
train Loss: 0.0045 Acc: 0.8495
val Loss: 0.0057 Acc: 0.8066

Best val Acc: 0.807018

---Fine tuning.---
Epoch 0/82
----------
LR is set to 0.01
train Loss: 0.0052 Acc: 0.8258
val Loss: 0.0075 Acc: 0.7502

Epoch 1/82
----------
train Loss: 0.0031 Acc: 0.8934
val Loss: 0.0045 Acc: 0.8573

Epoch 2/82
----------
train Loss: 0.0017 Acc: 0.9424
val Loss: 0.0039 Acc: 0.8887

Epoch 3/82
----------
train Loss: 0.0012 Acc: 0.9589
val Loss: 0.0038 Acc: 0.8915

Epoch 4/82
----------
train Loss: 0.0009 Acc: 0.9703
val Loss: 0.0037 Acc: 0.8938

Epoch 5/82
----------
train Loss: 0.0008 Acc: 0.9733
val Loss: 0.0042 Acc: 0.8989

Epoch 6/82
----------
train Loss: 0.0005 Acc: 0.9812
val Loss: 0.0039 Acc: 0.8984

Epoch 7/82
----------
train Loss: 0.0004 Acc: 0.9830
val Loss: 0.0037 Acc: 0.9003

Epoch 8/82
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9862
val Loss: 0.0038 Acc: 0.9040

Epoch 9/82
----------
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0038 Acc: 0.9058

Epoch 10/82
----------
train Loss: 0.0003 Acc: 0.9881
val Loss: 0.0038 Acc: 0.9063

Epoch 11/82
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9086

Epoch 12/82
----------
train Loss: 0.0003 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9104

Epoch 13/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9100

Epoch 14/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0038 Acc: 0.9109

Epoch 15/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9090

Epoch 16/82
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9081

Epoch 17/82
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9100

Epoch 18/82
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0037 Acc: 0.9104

Epoch 19/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9086

Epoch 20/82
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9109

Epoch 21/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9114

Epoch 22/82
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0037 Acc: 0.9104

Epoch 23/82
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9114

Epoch 24/82
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9123

Epoch 25/82
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0037 Acc: 0.9118

Epoch 26/82
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9114

Epoch 27/82
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9114

Epoch 28/82
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0038 Acc: 0.9118

Epoch 29/82
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9123

Epoch 30/82
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9114

Epoch 31/82
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9104

Epoch 32/82
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9114

Epoch 33/82
----------
train Loss: 0.0002 Acc: 0.9891
val Loss: 0.0037 Acc: 0.9127

Epoch 34/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9118

Epoch 35/82
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9127

Epoch 36/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9104

Epoch 37/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9118

Epoch 38/82
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0038 Acc: 0.9104

Epoch 39/82
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0038 Acc: 0.9104

Epoch 40/82
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0037 Acc: 0.9109

Epoch 41/82
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9123

Epoch 42/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9114

Epoch 43/82
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9109

Epoch 44/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9114

Epoch 45/82
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9118

Epoch 46/82
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9127

Epoch 47/82
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0038 Acc: 0.9132

Epoch 48/82
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9883
val Loss: 0.0037 Acc: 0.9114

Epoch 49/82
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0037 Acc: 0.9109

Epoch 50/82
----------
train Loss: 0.0002 Acc: 0.9887
val Loss: 0.0038 Acc: 0.9114

Epoch 51/82
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9118

Epoch 52/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0038 Acc: 0.9123

Epoch 53/82
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9114

Epoch 54/82
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9109

Epoch 55/82
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9118

Epoch 56/82
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9118

Epoch 57/82
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0037 Acc: 0.9109

Epoch 58/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0037 Acc: 0.9114

Epoch 59/82
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9118

Epoch 60/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9118

Epoch 61/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0037 Acc: 0.9118

Epoch 62/82
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0038 Acc: 0.9114

Epoch 63/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9118

Epoch 64/82
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9118

Epoch 65/82
----------
train Loss: 0.0002 Acc: 0.9895
val Loss: 0.0038 Acc: 0.9118

Epoch 66/82
----------
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9127

Epoch 67/82
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9114

Epoch 68/82
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0037 Acc: 0.9127

Epoch 69/82
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0037 Acc: 0.9114

Epoch 70/82
----------
train Loss: 0.0002 Acc: 0.9887
val Loss: 0.0037 Acc: 0.9118

Epoch 71/82
----------
train Loss: 0.0002 Acc: 0.9893
val Loss: 0.0037 Acc: 0.9109

Epoch 72/82
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9897
val Loss: 0.0037 Acc: 0.9127

Epoch 73/82
----------
train Loss: 0.0002 Acc: 0.9901
val Loss: 0.0038 Acc: 0.9109

Epoch 74/82
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0036 Acc: 0.9118

Epoch 75/82
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0036 Acc: 0.9118

Epoch 76/82
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0037 Acc: 0.9123

Epoch 77/82
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0037 Acc: 0.9095

Epoch 78/82
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9114

Epoch 79/82
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0037 Acc: 0.9104

Epoch 80/82
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0037 Acc: 0.9114

Epoch 81/82
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0037 Acc: 0.9118

Epoch 82/82
----------
train Loss: 0.0002 Acc: 0.9889
val Loss: 0.0037 Acc: 0.9118

Best val Acc: 0.913204

---Testing---
Test accuracy: 0.968014
--------------------
Accuracy of   ALB : 99 %
Accuracy of   BET : 96 %
Accuracy of Batoidea(ga_oo_lee) : 95 %
Accuracy of   DOL : 98 %
Accuracy of   LAG : 98 %
Accuracy of   NoF : 96 %
Accuracy of SHARK on boat : 99 %
Accuracy of   YFT : 99 %
Accuracy of holocephalan : 90 %
Accuracy of mullet : 37 %
Accuracy of   ray : 80 %
Accuracy of rough : 80 %
Accuracy of shark : 99 %
Accuracy of tuna_fish : 99 %
mean: 0.9074832011328671, std: 0.16069190936760944

Model saved in "./weights/super_with_most_[0.98]_mean[0.92]_std[0.23].save".
Training complete in 1594m 45s

Process finished with exit code 0
'''