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

train_this = 'all_in_one'
test_this = 'all_in_one'


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


'''/usr/bin/python3.5 "/home/visbic/python/pytorch_playground/datasets/new fish/git-test/all_in_one_weight_trainer.py"
--------------------

run info[val: 0.1, epoch: 47, randcrop: False, decay: 5]

---Training last layer.---
Epoch 0/46
----------
LR is set to 0.01
train Loss: 0.0224 Acc: 0.4403
val Loss: 0.0167 Acc: 0.5831

Epoch 1/46
----------
train Loss: 0.0133 Acc: 0.6355
val Loss: 0.0140 Acc: 0.6488

Epoch 2/46
----------
train Loss: 0.0111 Acc: 0.6905
val Loss: 0.0131 Acc: 0.6610

Epoch 3/46
----------
train Loss: 0.0099 Acc: 0.7209
val Loss: 0.0122 Acc: 0.6918

Epoch 4/46
----------
train Loss: 0.0091 Acc: 0.7452
val Loss: 0.0119 Acc: 0.6886

Epoch 5/46
----------
LR is set to 0.001
train Loss: 0.0082 Acc: 0.7765
val Loss: 0.0114 Acc: 0.7056

Epoch 6/46
----------
train Loss: 0.0081 Acc: 0.7775
val Loss: 0.0115 Acc: 0.7072

Epoch 7/46
----------
train Loss: 0.0080 Acc: 0.7815
val Loss: 0.0115 Acc: 0.6999

Epoch 8/46
----------
train Loss: 0.0079 Acc: 0.7833
val Loss: 0.0111 Acc: 0.7080

Epoch 9/46
----------
train Loss: 0.0079 Acc: 0.7826
val Loss: 0.0110 Acc: 0.7129

Epoch 10/46
----------
LR is set to 0.00010000000000000002
train Loss: 0.0079 Acc: 0.7830
val Loss: 0.0112 Acc: 0.7145

Epoch 11/46
----------
train Loss: 0.0079 Acc: 0.7859
val Loss: 0.0114 Acc: 0.7080

Epoch 12/46
----------
train Loss: 0.0078 Acc: 0.7864
val Loss: 0.0113 Acc: 0.7121

Epoch 13/46
----------
train Loss: 0.0078 Acc: 0.7905
val Loss: 0.0111 Acc: 0.7088

Epoch 14/46
----------
train Loss: 0.0078 Acc: 0.7906
val Loss: 0.0113 Acc: 0.7072

Epoch 15/46
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0078 Acc: 0.7877
val Loss: 0.0114 Acc: 0.7137

Epoch 16/46
----------
train Loss: 0.0078 Acc: 0.7882
val Loss: 0.0111 Acc: 0.7121

Epoch 17/46
----------
train Loss: 0.0078 Acc: 0.7895
val Loss: 0.0112 Acc: 0.7153

Epoch 18/46
----------
train Loss: 0.0078 Acc: 0.7864
val Loss: 0.0112 Acc: 0.7072

Epoch 19/46
----------
train Loss: 0.0078 Acc: 0.7933
val Loss: 0.0112 Acc: 0.7088

Epoch 20/46
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0078 Acc: 0.7871
val Loss: 0.0111 Acc: 0.7088

Epoch 21/46
----------
train Loss: 0.0078 Acc: 0.7863
val Loss: 0.0111 Acc: 0.7080

Epoch 22/46
----------
train Loss: 0.0078 Acc: 0.7842
val Loss: 0.0112 Acc: 0.7121

Epoch 23/46
----------
train Loss: 0.0078 Acc: 0.7895
val Loss: 0.0112 Acc: 0.7080

Epoch 24/46
----------
train Loss: 0.0078 Acc: 0.7886
val Loss: 0.0113 Acc: 0.7056

Epoch 25/46
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0078 Acc: 0.7898
val Loss: 0.0113 Acc: 0.7056

Epoch 26/46
----------
train Loss: 0.0078 Acc: 0.7903
val Loss: 0.0113 Acc: 0.7129

Epoch 27/46
----------
train Loss: 0.0078 Acc: 0.7883
val Loss: 0.0112 Acc: 0.7105

Epoch 28/46
----------
train Loss: 0.0078 Acc: 0.7889
val Loss: 0.0112 Acc: 0.7145

Epoch 29/46
----------
train Loss: 0.0078 Acc: 0.7821
val Loss: 0.0111 Acc: 0.7080

Epoch 30/46
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0078 Acc: 0.7865
val Loss: 0.0112 Acc: 0.7072

Epoch 31/46
----------
train Loss: 0.0078 Acc: 0.7904
val Loss: 0.0113 Acc: 0.7072

Epoch 32/46
----------
train Loss: 0.0078 Acc: 0.7868
val Loss: 0.0111 Acc: 0.7088

Epoch 33/46
----------
train Loss: 0.0077 Acc: 0.7939
val Loss: 0.0112 Acc: 0.7113

Epoch 34/46
----------
train Loss: 0.0078 Acc: 0.7878
val Loss: 0.0111 Acc: 0.7072

Epoch 35/46
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0078 Acc: 0.7898
val Loss: 0.0114 Acc: 0.7113

Epoch 36/46
----------
train Loss: 0.0078 Acc: 0.7867
val Loss: 0.0112 Acc: 0.7105

Epoch 37/46
----------
train Loss: 0.0078 Acc: 0.7914
val Loss: 0.0114 Acc: 0.7064

Epoch 38/46
----------
train Loss: 0.0078 Acc: 0.7907
val Loss: 0.0112 Acc: 0.7072

Epoch 39/46
----------
train Loss: 0.0078 Acc: 0.7923
val Loss: 0.0114 Acc: 0.7097

Epoch 40/46
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0078 Acc: 0.7876
val Loss: 0.0112 Acc: 0.7121

Epoch 41/46
----------
train Loss: 0.0078 Acc: 0.7875
val Loss: 0.0112 Acc: 0.7105

Epoch 42/46
----------
train Loss: 0.0078 Acc: 0.7914
val Loss: 0.0112 Acc: 0.7048

Epoch 43/46
----------
train Loss: 0.0078 Acc: 0.7888
val Loss: 0.0113 Acc: 0.7097

Epoch 44/46
----------
train Loss: 0.0078 Acc: 0.7884
val Loss: 0.0113 Acc: 0.7105

Epoch 45/46
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0078 Acc: 0.7869
val Loss: 0.0114 Acc: 0.7056

Epoch 46/46
----------
train Loss: 0.0078 Acc: 0.7876
val Loss: 0.0111 Acc: 0.7048

Best val Acc: 0.715328

---Fine tuning.---
Epoch 0/46
----------
LR is set to 0.01
train Loss: 0.0082 Acc: 0.7508
val Loss: 0.0125 Acc: 0.6910

Epoch 1/46
----------
train Loss: 0.0040 Acc: 0.8759
val Loss: 0.0084 Acc: 0.7802

Epoch 2/46
----------
train Loss: 0.0021 Acc: 0.9386
val Loss: 0.0084 Acc: 0.7908

Epoch 3/46
----------
train Loss: 0.0014 Acc: 0.9623
val Loss: 0.0079 Acc: 0.7981

Epoch 4/46
----------
train Loss: 0.0008 Acc: 0.9776
val Loss: 0.0075 Acc: 0.8118

Epoch 5/46
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9862
val Loss: 0.0071 Acc: 0.8248

Epoch 6/46
----------
train Loss: 0.0004 Acc: 0.9876
val Loss: 0.0070 Acc: 0.8248

Epoch 7/46
----------
train Loss: 0.0004 Acc: 0.9889
val Loss: 0.0072 Acc: 0.8191

Epoch 8/46
----------
train Loss: 0.0004 Acc: 0.9892
val Loss: 0.0070 Acc: 0.8183

Epoch 9/46
----------
train Loss: 0.0004 Acc: 0.9889
val Loss: 0.0072 Acc: 0.8167

Epoch 10/46
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0068 Acc: 0.8208

Epoch 11/46
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0072 Acc: 0.8183

Epoch 12/46
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0069 Acc: 0.8200

Epoch 13/46
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0074 Acc: 0.8200

Epoch 14/46
----------
train Loss: 0.0003 Acc: 0.9902
val Loss: 0.0070 Acc: 0.8175

Epoch 15/46
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9892
val Loss: 0.0071 Acc: 0.8208

Epoch 16/46
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0070 Acc: 0.8208

Epoch 17/46
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0073 Acc: 0.8224

Epoch 18/46
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0069 Acc: 0.8208

Epoch 19/46
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0073 Acc: 0.8224

Epoch 20/46
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0070 Acc: 0.8248

Epoch 21/46
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0072 Acc: 0.8200

Epoch 22/46
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0069 Acc: 0.8224

Epoch 23/46
----------
train Loss: 0.0003 Acc: 0.9902
val Loss: 0.0069 Acc: 0.8224

Epoch 24/46
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0072 Acc: 0.8200

Epoch 25/46
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0074 Acc: 0.8208

Epoch 26/46
----------
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8208

Epoch 27/46
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0069 Acc: 0.8208

Epoch 28/46
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0071 Acc: 0.8216

Epoch 29/46
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0071 Acc: 0.8208

Epoch 30/46
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0070 Acc: 0.8191

Epoch 31/46
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0070 Acc: 0.8216

Epoch 32/46
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0070 Acc: 0.8224

Epoch 33/46
----------
train Loss: 0.0003 Acc: 0.9902
val Loss: 0.0075 Acc: 0.8175

Epoch 34/46
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0069 Acc: 0.8224

Epoch 35/46
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0072 Acc: 0.8216

Epoch 36/46
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0070 Acc: 0.8208

Epoch 37/46
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0070 Acc: 0.8216

Epoch 38/46
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0071 Acc: 0.8232

Epoch 39/46
----------
train Loss: 0.0003 Acc: 0.9896
val Loss: 0.0073 Acc: 0.8191

Epoch 40/46
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0073 Acc: 0.8200

Epoch 41/46
----------
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0069 Acc: 0.8191

Epoch 42/46
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0072 Acc: 0.8200

Epoch 43/46
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0069 Acc: 0.8216

Epoch 44/46
----------
train Loss: 0.0003 Acc: 0.9900
val Loss: 0.0072 Acc: 0.8191

Epoch 45/46
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0069 Acc: 0.8175

Epoch 46/46
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0072 Acc: 0.8208

Best val Acc: 0.824818

---Testing---
Test accuracy: 0.971702
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 81 %
Accuracy of   BET : 99 %
Accuracy of Bigeye tuna : 89 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 98 %
Accuracy of Carcharhiniformes : 97 %
Accuracy of Conger myriaster : 99 %
Accuracy of   DOL : 97 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 98 %
Accuracy of Frigate tuna : 82 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexagrammos agrammus : 96 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 99 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 91 %
Accuracy of Larimichthys polyactis : 99 %
Accuracy of Lateolabrax japonicus : 98 %
Accuracy of Little tunny : 92 %
Accuracy of Longtail tuna : 98 %
Accuracy of Mackerel tuna : 86 %
Accuracy of Miichthys miiuy : 97 %
Accuracy of Mugil cephalus : 99 %
Accuracy of Myliobatiformes : 92 %
Accuracy of   NoF : 98 %
Accuracy of Oncorhynchus keta : 95 %
Accuracy of Oncorhynchus masou : 96 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pacific bluefin tuna : 78 %
Accuracy of Paralichthys olivaceus : 95 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 99 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 93 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 88 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 95 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 98 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 34 %
Accuracy of   ray : 84 %
Accuracy of rough : 94 %
Accuracy of shark : 98 %
mean: 0.9455795540545925, std: 0.09580371778491963
--------------------

run info[val: 0.2, epoch: 54, randcrop: True, decay: 14]

---Training last layer.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0236 Acc: 0.4158
val Loss: 0.0168 Acc: 0.5677

Epoch 1/53
----------
train Loss: 0.0147 Acc: 0.5960
val Loss: 0.0140 Acc: 0.6180

Epoch 2/53
----------
train Loss: 0.0124 Acc: 0.6490
val Loss: 0.0130 Acc: 0.6525

Epoch 3/53
----------
train Loss: 0.0110 Acc: 0.6842
val Loss: 0.0122 Acc: 0.6614

Epoch 4/53
----------
train Loss: 0.0104 Acc: 0.6991
val Loss: 0.0123 Acc: 0.6594

Epoch 5/53
----------
train Loss: 0.0098 Acc: 0.7148
val Loss: 0.0116 Acc: 0.6764

Epoch 6/53
----------
train Loss: 0.0093 Acc: 0.7248
val Loss: 0.0113 Acc: 0.6809

Epoch 7/53
----------
train Loss: 0.0089 Acc: 0.7362
val Loss: 0.0115 Acc: 0.6752

Epoch 8/53
----------
train Loss: 0.0085 Acc: 0.7458
val Loss: 0.0108 Acc: 0.6995

Epoch 9/53
----------
train Loss: 0.0083 Acc: 0.7550
val Loss: 0.0109 Acc: 0.6991

Epoch 10/53
----------
train Loss: 0.0081 Acc: 0.7642
val Loss: 0.0111 Acc: 0.6841

Epoch 11/53
----------
train Loss: 0.0078 Acc: 0.7743
val Loss: 0.0109 Acc: 0.6934

Epoch 12/53
----------
train Loss: 0.0077 Acc: 0.7737
val Loss: 0.0106 Acc: 0.6991

Epoch 13/53
----------
train Loss: 0.0075 Acc: 0.7748
val Loss: 0.0105 Acc: 0.7097

Epoch 14/53
----------
LR is set to 0.001
train Loss: 0.0069 Acc: 0.7986
val Loss: 0.0102 Acc: 0.7133

Epoch 15/53
----------
train Loss: 0.0069 Acc: 0.8026
val Loss: 0.0102 Acc: 0.7165

Epoch 16/53
----------
train Loss: 0.0068 Acc: 0.8000
val Loss: 0.0103 Acc: 0.7153

Epoch 17/53
----------
train Loss: 0.0068 Acc: 0.8019
val Loss: 0.0102 Acc: 0.7141

Epoch 18/53
----------
train Loss: 0.0068 Acc: 0.8029
val Loss: 0.0102 Acc: 0.7137

Epoch 19/53
----------
train Loss: 0.0068 Acc: 0.8015
val Loss: 0.0102 Acc: 0.7165

Epoch 20/53
----------
train Loss: 0.0067 Acc: 0.8004
val Loss: 0.0102 Acc: 0.7153

Epoch 21/53
----------
train Loss: 0.0068 Acc: 0.8003
val Loss: 0.0103 Acc: 0.7165

Epoch 22/53
----------
train Loss: 0.0068 Acc: 0.8026
val Loss: 0.0103 Acc: 0.7182

Epoch 23/53
----------
train Loss: 0.0067 Acc: 0.8065
val Loss: 0.0103 Acc: 0.7153

Epoch 24/53
----------
train Loss: 0.0067 Acc: 0.8064
val Loss: 0.0102 Acc: 0.7141

Epoch 25/53
----------
train Loss: 0.0067 Acc: 0.8018
val Loss: 0.0101 Acc: 0.7178

Epoch 26/53
----------
train Loss: 0.0067 Acc: 0.8069
val Loss: 0.0102 Acc: 0.7165

Epoch 27/53
----------
train Loss: 0.0067 Acc: 0.8033
val Loss: 0.0102 Acc: 0.7182

Epoch 28/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0066 Acc: 0.8062
val Loss: 0.0101 Acc: 0.7182

Epoch 29/53
----------
train Loss: 0.0066 Acc: 0.8105
val Loss: 0.0101 Acc: 0.7174

Epoch 30/53
----------
train Loss: 0.0067 Acc: 0.8053
val Loss: 0.0101 Acc: 0.7186

Epoch 31/53
----------
train Loss: 0.0066 Acc: 0.8101
val Loss: 0.0101 Acc: 0.7165

Epoch 32/53
----------
train Loss: 0.0066 Acc: 0.8103
val Loss: 0.0102 Acc: 0.7141

Epoch 33/53
----------
train Loss: 0.0066 Acc: 0.8123
val Loss: 0.0102 Acc: 0.7190

Epoch 34/53
----------
train Loss: 0.0066 Acc: 0.8106
val Loss: 0.0102 Acc: 0.7186

Epoch 35/53
----------
train Loss: 0.0066 Acc: 0.8079
val Loss: 0.0102 Acc: 0.7161

Epoch 36/53
----------
train Loss: 0.0066 Acc: 0.8078
val Loss: 0.0102 Acc: 0.7153

Epoch 37/53
----------
train Loss: 0.0067 Acc: 0.8012
val Loss: 0.0102 Acc: 0.7170

Epoch 38/53
----------
train Loss: 0.0066 Acc: 0.8081
val Loss: 0.0102 Acc: 0.7165

Epoch 39/53
----------
train Loss: 0.0066 Acc: 0.8051
val Loss: 0.0102 Acc: 0.7129

Epoch 40/53
----------
train Loss: 0.0066 Acc: 0.8108
val Loss: 0.0101 Acc: 0.7153

Epoch 41/53
----------
train Loss: 0.0066 Acc: 0.8085
val Loss: 0.0101 Acc: 0.7178

Epoch 42/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0066 Acc: 0.8101
val Loss: 0.0102 Acc: 0.7182

Epoch 43/53
----------
train Loss: 0.0066 Acc: 0.8089
val Loss: 0.0102 Acc: 0.7174

Epoch 44/53
----------
train Loss: 0.0067 Acc: 0.8042
val Loss: 0.0102 Acc: 0.7165

Epoch 45/53
----------
train Loss: 0.0066 Acc: 0.8088
val Loss: 0.0102 Acc: 0.7161

Epoch 46/53
----------
train Loss: 0.0066 Acc: 0.8072
val Loss: 0.0101 Acc: 0.7145

Epoch 47/53
----------
train Loss: 0.0066 Acc: 0.8101
val Loss: 0.0102 Acc: 0.7157

Epoch 48/53
----------
train Loss: 0.0066 Acc: 0.8088
val Loss: 0.0101 Acc: 0.7174

Epoch 49/53
----------
train Loss: 0.0065 Acc: 0.8065
val Loss: 0.0101 Acc: 0.7161

Epoch 50/53
----------
train Loss: 0.0066 Acc: 0.8074
val Loss: 0.0102 Acc: 0.7182

Epoch 51/53
----------
train Loss: 0.0066 Acc: 0.8105
val Loss: 0.0102 Acc: 0.7174

Epoch 52/53
----------
train Loss: 0.0067 Acc: 0.8015
val Loss: 0.0101 Acc: 0.7137

Epoch 53/53
----------
train Loss: 0.0066 Acc: 0.8122
val Loss: 0.0102 Acc: 0.7165

Best val Acc: 0.718978

---Fine tuning.---
Epoch 0/53
----------
LR is set to 0.01
train Loss: 0.0096 Acc: 0.7071
val Loss: 0.0132 Acc: 0.6594

Epoch 1/53
----------
train Loss: 0.0055 Acc: 0.8301
val Loss: 0.0098 Acc: 0.7405

Epoch 2/53
----------
train Loss: 0.0032 Acc: 0.9000
val Loss: 0.0086 Acc: 0.7753

Epoch 3/53
----------
train Loss: 0.0022 Acc: 0.9354
val Loss: 0.0083 Acc: 0.7883

Epoch 4/53
----------
train Loss: 0.0016 Acc: 0.9529
val Loss: 0.0080 Acc: 0.8001

Epoch 5/53
----------
train Loss: 0.0012 Acc: 0.9659
val Loss: 0.0082 Acc: 0.8033

Epoch 6/53
----------
train Loss: 0.0010 Acc: 0.9710
val Loss: 0.0085 Acc: 0.8021

Epoch 7/53
----------
train Loss: 0.0009 Acc: 0.9730
val Loss: 0.0079 Acc: 0.8167

Epoch 8/53
----------
train Loss: 0.0007 Acc: 0.9794
val Loss: 0.0078 Acc: 0.8175

Epoch 9/53
----------
train Loss: 0.0006 Acc: 0.9830
val Loss: 0.0080 Acc: 0.8139

Epoch 10/53
----------
train Loss: 0.0005 Acc: 0.9826
val Loss: 0.0085 Acc: 0.8062

Epoch 11/53
----------
train Loss: 0.0005 Acc: 0.9835
val Loss: 0.0080 Acc: 0.8106

Epoch 12/53
----------
train Loss: 0.0005 Acc: 0.9859
val Loss: 0.0080 Acc: 0.8195

Epoch 13/53
----------
train Loss: 0.0004 Acc: 0.9859
val Loss: 0.0080 Acc: 0.8163

Epoch 14/53
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9886
val Loss: 0.0080 Acc: 0.8167

Epoch 15/53
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0080 Acc: 0.8171

Epoch 16/53
----------
train Loss: 0.0002 Acc: 0.9902
val Loss: 0.0080 Acc: 0.8204

Epoch 17/53
----------
train Loss: 0.0003 Acc: 0.9892
val Loss: 0.0079 Acc: 0.8191

Epoch 18/53
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0080 Acc: 0.8183

Epoch 19/53
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0080 Acc: 0.8191

Epoch 20/53
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0080 Acc: 0.8191

Epoch 21/53
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0081 Acc: 0.8200

Epoch 22/53
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0080 Acc: 0.8200

Epoch 23/53
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0080 Acc: 0.8191

Epoch 24/53
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0080 Acc: 0.8204

Epoch 25/53
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0081 Acc: 0.8204

Epoch 26/53
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0080 Acc: 0.8195

Epoch 27/53
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0081 Acc: 0.8212

Epoch 28/53
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0081 Acc: 0.8208

Epoch 29/53
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0081 Acc: 0.8200

Epoch 30/53
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0080 Acc: 0.8212

Epoch 31/53
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0081 Acc: 0.8220

Epoch 32/53
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0081 Acc: 0.8204

Epoch 33/53
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0082 Acc: 0.8187

Epoch 34/53
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0081 Acc: 0.8208

Epoch 35/53
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0081 Acc: 0.8204

Epoch 36/53
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0081 Acc: 0.8212

Epoch 37/53
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0081 Acc: 0.8204

Epoch 38/53
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0082 Acc: 0.8191

Epoch 39/53
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0082 Acc: 0.8212

Epoch 40/53
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0080 Acc: 0.8195

Epoch 41/53
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0081 Acc: 0.8204

Epoch 42/53
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0081 Acc: 0.8200

Epoch 43/53
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0082 Acc: 0.8200

Epoch 44/53
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0081 Acc: 0.8200

Epoch 45/53
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0081 Acc: 0.8208

Epoch 46/53
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0081 Acc: 0.8204

Epoch 47/53
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0081 Acc: 0.8212

Epoch 48/53
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0081 Acc: 0.8200

Epoch 49/53
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0081 Acc: 0.8212

Epoch 50/53
----------
train Loss: 0.0002 Acc: 0.9930
val Loss: 0.0081 Acc: 0.8212

Epoch 51/53
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0081 Acc: 0.8200

Epoch 52/53
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0081 Acc: 0.8204

Epoch 53/53
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0081 Acc: 0.8200

Best val Acc: 0.821979

---Testing---
Test accuracy: 0.958404
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 89 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of   BET : 98 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 89 %
Accuracy of Carcharhiniformes : 90 %
Accuracy of Conger myriaster : 97 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 96 %
Accuracy of Epinephelus septemfasciatus : 95 %
Accuracy of Frigate tuna : 75 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexagrammos agrammus : 92 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 97 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 98 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 97 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 92 %
Accuracy of Mackerel tuna : 80 %
Accuracy of Miichthys miiuy : 94 %
Accuracy of Mugil cephalus : 97 %
Accuracy of Myliobatiformes : 90 %
Accuracy of   NoF : 97 %
Accuracy of Oncorhynchus keta : 94 %
Accuracy of Oncorhynchus masou : 93 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 93 %
Accuracy of Pacific bluefin tuna : 86 %
Accuracy of Paralichthys olivaceus : 94 %
Accuracy of Pleuronectidae : 96 %
Accuracy of Pristiformes : 96 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 98 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 81 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 86 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 93 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 93 %
Accuracy of holocephalan : 92 %
Accuracy of mullet : 49 %
Accuracy of   ray : 75 %
Accuracy of rough : 86 %
Accuracy of shark : 97 %
mean: 0.9285487472648799, std: 0.08389972956305135
--------------------

run info[val: 0.3, epoch: 58, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0246 Acc: 0.4021
val Loss: 0.0171 Acc: 0.5472

Epoch 1/57
----------
train Loss: 0.0152 Acc: 0.5844
val Loss: 0.0142 Acc: 0.6175

Epoch 2/57
----------
train Loss: 0.0129 Acc: 0.6399
val Loss: 0.0130 Acc: 0.6372

Epoch 3/57
----------
LR is set to 0.001
train Loss: 0.0112 Acc: 0.6862
val Loss: 0.0123 Acc: 0.6607

Epoch 4/57
----------
train Loss: 0.0111 Acc: 0.6987
val Loss: 0.0123 Acc: 0.6623

Epoch 5/57
----------
train Loss: 0.0110 Acc: 0.7030
val Loss: 0.0122 Acc: 0.6653

Epoch 6/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0108 Acc: 0.7055
val Loss: 0.0122 Acc: 0.6688

Epoch 7/57
----------
train Loss: 0.0108 Acc: 0.7028
val Loss: 0.0121 Acc: 0.6645

Epoch 8/57
----------
train Loss: 0.0108 Acc: 0.7019
val Loss: 0.0121 Acc: 0.6645

Epoch 9/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0107 Acc: 0.7062
val Loss: 0.0121 Acc: 0.6675

Epoch 10/57
----------
train Loss: 0.0107 Acc: 0.7072
val Loss: 0.0122 Acc: 0.6642

Epoch 11/57
----------
train Loss: 0.0108 Acc: 0.7059
val Loss: 0.0121 Acc: 0.6640

Epoch 12/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0107 Acc: 0.7092
val Loss: 0.0121 Acc: 0.6645

Epoch 13/57
----------
train Loss: 0.0107 Acc: 0.7080
val Loss: 0.0122 Acc: 0.6669

Epoch 14/57
----------
train Loss: 0.0108 Acc: 0.7057
val Loss: 0.0122 Acc: 0.6640

Epoch 15/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0108 Acc: 0.7070
val Loss: 0.0121 Acc: 0.6669

Epoch 16/57
----------
train Loss: 0.0108 Acc: 0.7058
val Loss: 0.0121 Acc: 0.6661

Epoch 17/57
----------
train Loss: 0.0108 Acc: 0.7078
val Loss: 0.0121 Acc: 0.6659

Epoch 18/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0107 Acc: 0.7100
val Loss: 0.0122 Acc: 0.6664

Epoch 19/57
----------
train Loss: 0.0107 Acc: 0.7092
val Loss: 0.0121 Acc: 0.6648

Epoch 20/57
----------
train Loss: 0.0107 Acc: 0.7077
val Loss: 0.0121 Acc: 0.6661

Epoch 21/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0108 Acc: 0.7055
val Loss: 0.0121 Acc: 0.6648

Epoch 22/57
----------
train Loss: 0.0108 Acc: 0.7079
val Loss: 0.0121 Acc: 0.6688

Epoch 23/57
----------
train Loss: 0.0108 Acc: 0.7018
val Loss: 0.0121 Acc: 0.6677

Epoch 24/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0108 Acc: 0.7064
val Loss: 0.0121 Acc: 0.6667

Epoch 25/57
----------
train Loss: 0.0107 Acc: 0.7101
val Loss: 0.0122 Acc: 0.6664

Epoch 26/57
----------
train Loss: 0.0107 Acc: 0.7045
val Loss: 0.0122 Acc: 0.6615

Epoch 27/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0107 Acc: 0.7080
val Loss: 0.0121 Acc: 0.6675

Epoch 28/57
----------
train Loss: 0.0107 Acc: 0.7081
val Loss: 0.0121 Acc: 0.6677

Epoch 29/57
----------
train Loss: 0.0107 Acc: 0.7091
val Loss: 0.0121 Acc: 0.6642

Epoch 30/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0108 Acc: 0.7043
val Loss: 0.0122 Acc: 0.6621

Epoch 31/57
----------
train Loss: 0.0107 Acc: 0.7028
val Loss: 0.0122 Acc: 0.6645

Epoch 32/57
----------
train Loss: 0.0108 Acc: 0.7059
val Loss: 0.0121 Acc: 0.6642

Epoch 33/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0108 Acc: 0.7042
val Loss: 0.0121 Acc: 0.6640

Epoch 34/57
----------
train Loss: 0.0108 Acc: 0.7016
val Loss: 0.0121 Acc: 0.6632

Epoch 35/57
----------
train Loss: 0.0108 Acc: 0.7076
val Loss: 0.0121 Acc: 0.6659

Epoch 36/57
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0107 Acc: 0.7130
val Loss: 0.0122 Acc: 0.6632

Epoch 37/57
----------
train Loss: 0.0107 Acc: 0.7089
val Loss: 0.0121 Acc: 0.6659

Epoch 38/57
----------
train Loss: 0.0108 Acc: 0.7071
val Loss: 0.0121 Acc: 0.6637

Epoch 39/57
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0108 Acc: 0.7079
val Loss: 0.0121 Acc: 0.6648

Epoch 40/57
----------
train Loss: 0.0108 Acc: 0.7099
val Loss: 0.0122 Acc: 0.6640

Epoch 41/57
----------
train Loss: 0.0107 Acc: 0.7092
val Loss: 0.0122 Acc: 0.6664

Epoch 42/57
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0107 Acc: 0.7114
val Loss: 0.0122 Acc: 0.6642

Epoch 43/57
----------
train Loss: 0.0107 Acc: 0.7041
val Loss: 0.0122 Acc: 0.6645

Epoch 44/57
----------
train Loss: 0.0107 Acc: 0.7094
val Loss: 0.0122 Acc: 0.6634

Epoch 45/57
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0107 Acc: 0.7072
val Loss: 0.0122 Acc: 0.6629

Epoch 46/57
----------
train Loss: 0.0108 Acc: 0.7066
val Loss: 0.0121 Acc: 0.6656

Epoch 47/57
----------
train Loss: 0.0108 Acc: 0.7036
val Loss: 0.0121 Acc: 0.6672

Epoch 48/57
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0108 Acc: 0.7048
val Loss: 0.0121 Acc: 0.6661

Epoch 49/57
----------
train Loss: 0.0108 Acc: 0.7107
val Loss: 0.0121 Acc: 0.6648

Epoch 50/57
----------
train Loss: 0.0108 Acc: 0.7051
val Loss: 0.0121 Acc: 0.6686

Epoch 51/57
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0108 Acc: 0.7013
val Loss: 0.0122 Acc: 0.6621

Epoch 52/57
----------
train Loss: 0.0107 Acc: 0.7060
val Loss: 0.0122 Acc: 0.6664

Epoch 53/57
----------
train Loss: 0.0107 Acc: 0.7018
val Loss: 0.0121 Acc: 0.6656

Epoch 54/57
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0108 Acc: 0.7109
val Loss: 0.0121 Acc: 0.6661

Epoch 55/57
----------
train Loss: 0.0107 Acc: 0.7003
val Loss: 0.0122 Acc: 0.6669

Epoch 56/57
----------
train Loss: 0.0108 Acc: 0.7073
val Loss: 0.0121 Acc: 0.6691

Epoch 57/57
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0108 Acc: 0.7054
val Loss: 0.0122 Acc: 0.6656

Best val Acc: 0.669100

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0101 Acc: 0.6962
val Loss: 0.0114 Acc: 0.6696

Epoch 1/57
----------
train Loss: 0.0062 Acc: 0.8116
val Loss: 0.0090 Acc: 0.7502

Epoch 2/57
----------
train Loss: 0.0040 Acc: 0.8776
val Loss: 0.0075 Acc: 0.7843

Epoch 3/57
----------
LR is set to 0.001
train Loss: 0.0023 Acc: 0.9397
val Loss: 0.0066 Acc: 0.8181

Epoch 4/57
----------
train Loss: 0.0020 Acc: 0.9505
val Loss: 0.0065 Acc: 0.8200

Epoch 5/57
----------
train Loss: 0.0018 Acc: 0.9581
val Loss: 0.0064 Acc: 0.8235

Epoch 6/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0017 Acc: 0.9600
val Loss: 0.0064 Acc: 0.8210

Epoch 7/57
----------
train Loss: 0.0017 Acc: 0.9603
val Loss: 0.0064 Acc: 0.8224

Epoch 8/57
----------
train Loss: 0.0016 Acc: 0.9640
val Loss: 0.0064 Acc: 0.8216

Epoch 9/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9631
val Loss: 0.0064 Acc: 0.8218

Epoch 10/57
----------
train Loss: 0.0016 Acc: 0.9615
val Loss: 0.0064 Acc: 0.8235

Epoch 11/57
----------
train Loss: 0.0016 Acc: 0.9635
val Loss: 0.0064 Acc: 0.8216

Epoch 12/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9615
val Loss: 0.0064 Acc: 0.8224

Epoch 13/57
----------
train Loss: 0.0016 Acc: 0.9639
val Loss: 0.0064 Acc: 0.8205

Epoch 14/57
----------
train Loss: 0.0016 Acc: 0.9634
val Loss: 0.0064 Acc: 0.8227

Epoch 15/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0016 Acc: 0.9617
val Loss: 0.0064 Acc: 0.8232

Epoch 16/57
----------
train Loss: 0.0016 Acc: 0.9609
val Loss: 0.0064 Acc: 0.8221

Epoch 17/57
----------
train Loss: 0.0016 Acc: 0.9639
val Loss: 0.0064 Acc: 0.8245

Epoch 18/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0016 Acc: 0.9662
val Loss: 0.0064 Acc: 0.8243

Epoch 19/57
----------
train Loss: 0.0017 Acc: 0.9609
val Loss: 0.0063 Acc: 0.8237

Epoch 20/57
----------
train Loss: 0.0016 Acc: 0.9637
val Loss: 0.0064 Acc: 0.8235

Epoch 21/57
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0016 Acc: 0.9627
val Loss: 0.0064 Acc: 0.8227

Epoch 22/57
----------
train Loss: 0.0016 Acc: 0.9611
val Loss: 0.0064 Acc: 0.8224

Epoch 23/57
----------
train Loss: 0.0016 Acc: 0.9629
val Loss: 0.0064 Acc: 0.8221

Epoch 24/57
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0016 Acc: 0.9633
val Loss: 0.0064 Acc: 0.8237

Epoch 25/57
----------
train Loss: 0.0017 Acc: 0.9615
val Loss: 0.0064 Acc: 0.8227

Epoch 26/57
----------
train Loss: 0.0016 Acc: 0.9631
val Loss: 0.0064 Acc: 0.8229

Epoch 27/57
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0016 Acc: 0.9611
val Loss: 0.0064 Acc: 0.8227

Epoch 28/57
----------
train Loss: 0.0017 Acc: 0.9610
val Loss: 0.0064 Acc: 0.8237

Epoch 29/57
----------
train Loss: 0.0016 Acc: 0.9637
val Loss: 0.0064 Acc: 0.8232

Epoch 30/57
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0016 Acc: 0.9641
val Loss: 0.0064 Acc: 0.8218

Epoch 31/57
----------
train Loss: 0.0016 Acc: 0.9588
val Loss: 0.0064 Acc: 0.8243

Epoch 32/57
----------
train Loss: 0.0016 Acc: 0.9659
val Loss: 0.0064 Acc: 0.8224

Epoch 33/57
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0016 Acc: 0.9622
val Loss: 0.0064 Acc: 0.8221

Epoch 34/57
----------
train Loss: 0.0016 Acc: 0.9627
val Loss: 0.0064 Acc: 0.8221

Epoch 35/57
----------
train Loss: 0.0016 Acc: 0.9650
val Loss: 0.0064 Acc: 0.8216

Epoch 36/57
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0016 Acc: 0.9654
val Loss: 0.0064 Acc: 0.8218

Epoch 37/57
----------
train Loss: 0.0017 Acc: 0.9624
val Loss: 0.0064 Acc: 0.8224

Epoch 38/57
----------
train Loss: 0.0016 Acc: 0.9626
val Loss: 0.0064 Acc: 0.8218

Epoch 39/57
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0016 Acc: 0.9650
val Loss: 0.0064 Acc: 0.8221

Epoch 40/57
----------
train Loss: 0.0016 Acc: 0.9604
val Loss: 0.0064 Acc: 0.8227

Epoch 41/57
----------
train Loss: 0.0016 Acc: 0.9619
val Loss: 0.0064 Acc: 0.8229

Epoch 42/57
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0016 Acc: 0.9647
val Loss: 0.0064 Acc: 0.8216

Epoch 43/57
----------
train Loss: 0.0016 Acc: 0.9642
val Loss: 0.0064 Acc: 0.8210

Epoch 44/57
----------
train Loss: 0.0016 Acc: 0.9611
val Loss: 0.0064 Acc: 0.8221

Epoch 45/57
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0016 Acc: 0.9634
val Loss: 0.0064 Acc: 0.8227

Epoch 46/57
----------
train Loss: 0.0016 Acc: 0.9619
val Loss: 0.0064 Acc: 0.8232

Epoch 47/57
----------
train Loss: 0.0016 Acc: 0.9636
val Loss: 0.0064 Acc: 0.8221

Epoch 48/57
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0016 Acc: 0.9633
val Loss: 0.0064 Acc: 0.8210

Epoch 49/57
----------
train Loss: 0.0016 Acc: 0.9629
val Loss: 0.0064 Acc: 0.8216

Epoch 50/57
----------
train Loss: 0.0016 Acc: 0.9643
val Loss: 0.0064 Acc: 0.8216

Epoch 51/57
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0016 Acc: 0.9621
val Loss: 0.0064 Acc: 0.8237

Epoch 52/57
----------
train Loss: 0.0016 Acc: 0.9658
val Loss: 0.0064 Acc: 0.8237

Epoch 53/57
----------
train Loss: 0.0016 Acc: 0.9629
val Loss: 0.0064 Acc: 0.8210

Epoch 54/57
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0016 Acc: 0.9655
val Loss: 0.0063 Acc: 0.8224

Epoch 55/57
----------
train Loss: 0.0016 Acc: 0.9624
val Loss: 0.0064 Acc: 0.8197

Epoch 56/57
----------
train Loss: 0.0016 Acc: 0.9635
val Loss: 0.0064 Acc: 0.8205

Epoch 57/57
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0016 Acc: 0.9631
val Loss: 0.0064 Acc: 0.8208

Best val Acc: 0.824547

---Testing---
Test accuracy: 0.929620
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 82 %
Accuracy of Atlantic bluefin tuna : 70 %
Accuracy of   BET : 95 %
Accuracy of Bigeye tuna : 71 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 86 %
Accuracy of Carcharhiniformes : 89 %
Accuracy of Conger myriaster : 97 %
Accuracy of   DOL : 96 %
Accuracy of Dasyatiformes : 96 %
Accuracy of Epinephelus septemfasciatus : 95 %
Accuracy of Frigate tuna : 44 %
Accuracy of Heterodontiformes : 94 %
Accuracy of Hexagrammos agrammus : 84 %
Accuracy of Hexanchiformes : 92 %
Accuracy of Konosirus punctatus : 96 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 89 %
Accuracy of Larimichthys polyactis : 97 %
Accuracy of Lateolabrax japonicus : 94 %
Accuracy of Little tunny : 77 %
Accuracy of Longtail tuna : 89 %
Accuracy of Mackerel tuna : 60 %
Accuracy of Miichthys miiuy : 92 %
Accuracy of Mugil cephalus : 94 %
Accuracy of Myliobatiformes : 81 %
Accuracy of   NoF : 94 %
Accuracy of Oncorhynchus keta : 86 %
Accuracy of Oncorhynchus masou : 92 %
Accuracy of Oplegnathus fasciatus : 98 %
Accuracy of Orectolobiformes : 95 %
Accuracy of Pacific bluefin tuna : 65 %
Accuracy of Paralichthys olivaceus : 91 %
Accuracy of Pleuronectidae : 94 %
Accuracy of Pristiformes : 93 %
Accuracy of Rajiformes : 91 %
Accuracy of Rhinobatiformes : 92 %
Accuracy of SHARK on boat : 97 %
Accuracy of Scomber japonicus : 95 %
Accuracy of Sebastes inermis : 98 %
Accuracy of Seriola quinqueradiata : 96 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 21 %
Accuracy of Southern bluefin tuna : 48 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 93 %
Accuracy of Stephanolepis cirrhifer : 98 %
Accuracy of Tetraodon or Diodon : 98 %
Accuracy of Thunnus orientalis : 76 %
Accuracy of Torpediniformes : 95 %
Accuracy of Trachurus japonicus : 90 %
Accuracy of   YFT : 97 %
Accuracy of Yellowfin tuna : 93 %
Accuracy of holocephalan : 88 %
Accuracy of mullet : 27 %
Accuracy of   ray : 75 %
Accuracy of rough : 76 %
Accuracy of shark : 95 %
mean: 0.866086710567472, std: 0.16554517291354492

Model saved in "./weights/all_in_one_[0.97]_mean[0.95]_std[0.1].save".
--------------------

run info[val: 0.1, epoch: 60, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0228 Acc: 0.4363
val Loss: 0.0170 Acc: 0.5742

Epoch 1/59
----------
train Loss: 0.0141 Acc: 0.6161
val Loss: 0.0141 Acc: 0.6245

Epoch 2/59
----------
train Loss: 0.0120 Acc: 0.6551
val Loss: 0.0131 Acc: 0.6586

Epoch 3/59
----------
LR is set to 0.001
train Loss: 0.0106 Acc: 0.7026
val Loss: 0.0125 Acc: 0.6723

Epoch 4/59
----------
train Loss: 0.0104 Acc: 0.7106
val Loss: 0.0123 Acc: 0.6772

Epoch 5/59
----------
train Loss: 0.0103 Acc: 0.7142
val Loss: 0.0127 Acc: 0.6764

Epoch 6/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0103 Acc: 0.7099
val Loss: 0.0127 Acc: 0.6723

Epoch 7/59
----------
train Loss: 0.0102 Acc: 0.7175
val Loss: 0.0123 Acc: 0.6764

Epoch 8/59
----------
train Loss: 0.0103 Acc: 0.7120
val Loss: 0.0125 Acc: 0.6796

Epoch 9/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0102 Acc: 0.7177
val Loss: 0.0125 Acc: 0.6780

Epoch 10/59
----------
train Loss: 0.0102 Acc: 0.7193
val Loss: 0.0125 Acc: 0.6748

Epoch 11/59
----------
train Loss: 0.0102 Acc: 0.7138
val Loss: 0.0126 Acc: 0.6740

Epoch 12/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0102 Acc: 0.7151
val Loss: 0.0123 Acc: 0.6788

Epoch 13/59
----------
train Loss: 0.0102 Acc: 0.7160
val Loss: 0.0126 Acc: 0.6748

Epoch 14/59
----------
train Loss: 0.0102 Acc: 0.7146
val Loss: 0.0124 Acc: 0.6691

Epoch 15/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0102 Acc: 0.7163
val Loss: 0.0124 Acc: 0.6715

Epoch 16/59
----------
train Loss: 0.0102 Acc: 0.7165
val Loss: 0.0125 Acc: 0.6723

Epoch 17/59
----------
train Loss: 0.0102 Acc: 0.7190
val Loss: 0.0126 Acc: 0.6772

Epoch 18/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0102 Acc: 0.7197
val Loss: 0.0122 Acc: 0.6723

Epoch 19/59
----------
train Loss: 0.0102 Acc: 0.7137
val Loss: 0.0123 Acc: 0.6740

Epoch 20/59
----------
train Loss: 0.0102 Acc: 0.7159
val Loss: 0.0122 Acc: 0.6756

Epoch 21/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0102 Acc: 0.7157
val Loss: 0.0125 Acc: 0.6756

Epoch 22/59
----------
train Loss: 0.0102 Acc: 0.7199
val Loss: 0.0124 Acc: 0.6748

Epoch 23/59
----------
train Loss: 0.0102 Acc: 0.7177
val Loss: 0.0122 Acc: 0.6748

Epoch 24/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0102 Acc: 0.7178
val Loss: 0.0125 Acc: 0.6756

Epoch 25/59
----------
train Loss: 0.0102 Acc: 0.7164
val Loss: 0.0124 Acc: 0.6748

Epoch 26/59
----------
train Loss: 0.0102 Acc: 0.7186
val Loss: 0.0123 Acc: 0.6723

Epoch 27/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0102 Acc: 0.7171
val Loss: 0.0122 Acc: 0.6772

Epoch 28/59
----------
train Loss: 0.0102 Acc: 0.7154
val Loss: 0.0124 Acc: 0.6796

Epoch 29/59
----------
train Loss: 0.0101 Acc: 0.7186
val Loss: 0.0123 Acc: 0.6732

Epoch 30/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0102 Acc: 0.7144
val Loss: 0.0124 Acc: 0.6723

Epoch 31/59
----------
train Loss: 0.0102 Acc: 0.7149
val Loss: 0.0123 Acc: 0.6756

Epoch 32/59
----------
train Loss: 0.0101 Acc: 0.7184
val Loss: 0.0126 Acc: 0.6740

Epoch 33/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0102 Acc: 0.7124
val Loss: 0.0125 Acc: 0.6788

Epoch 34/59
----------
train Loss: 0.0101 Acc: 0.7149
val Loss: 0.0124 Acc: 0.6715

Epoch 35/59
----------
train Loss: 0.0101 Acc: 0.7196
val Loss: 0.0124 Acc: 0.6772

Epoch 36/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0102 Acc: 0.7150
val Loss: 0.0124 Acc: 0.6756

Epoch 37/59
----------
train Loss: 0.0102 Acc: 0.7148
val Loss: 0.0121 Acc: 0.6788

Epoch 38/59
----------
train Loss: 0.0102 Acc: 0.7196
val Loss: 0.0125 Acc: 0.6813

Epoch 39/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0103 Acc: 0.7134
val Loss: 0.0124 Acc: 0.6740

Epoch 40/59
----------
train Loss: 0.0102 Acc: 0.7162
val Loss: 0.0123 Acc: 0.6756

Epoch 41/59
----------
train Loss: 0.0102 Acc: 0.7174
val Loss: 0.0125 Acc: 0.6732

Epoch 42/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0101 Acc: 0.7214
val Loss: 0.0123 Acc: 0.6723

Epoch 43/59
----------
train Loss: 0.0102 Acc: 0.7209
val Loss: 0.0127 Acc: 0.6805

Epoch 44/59
----------
train Loss: 0.0102 Acc: 0.7163
val Loss: 0.0126 Acc: 0.6764

Epoch 45/59
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0102 Acc: 0.7113
val Loss: 0.0121 Acc: 0.6780

Epoch 46/59
----------
train Loss: 0.0102 Acc: 0.7205
val Loss: 0.0124 Acc: 0.6788

Epoch 47/59
----------
train Loss: 0.0102 Acc: 0.7126
val Loss: 0.0123 Acc: 0.6723

Epoch 48/59
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0101 Acc: 0.7230
val Loss: 0.0123 Acc: 0.6788

Epoch 49/59
----------
train Loss: 0.0101 Acc: 0.7235
val Loss: 0.0123 Acc: 0.6723

Epoch 50/59
----------
train Loss: 0.0101 Acc: 0.7162
val Loss: 0.0122 Acc: 0.6772

Epoch 51/59
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0102 Acc: 0.7170
val Loss: 0.0124 Acc: 0.6756

Epoch 52/59
----------
train Loss: 0.0102 Acc: 0.7156
val Loss: 0.0124 Acc: 0.6748

Epoch 53/59
----------
train Loss: 0.0102 Acc: 0.7141
val Loss: 0.0124 Acc: 0.6740

Epoch 54/59
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0103 Acc: 0.7162
val Loss: 0.0125 Acc: 0.6756

Epoch 55/59
----------
train Loss: 0.0102 Acc: 0.7197
val Loss: 0.0123 Acc: 0.6723

Epoch 56/59
----------
train Loss: 0.0102 Acc: 0.7111
val Loss: 0.0125 Acc: 0.6764

Epoch 57/59
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0102 Acc: 0.7149
val Loss: 0.0123 Acc: 0.6788

Epoch 58/59
----------
train Loss: 0.0102 Acc: 0.7153
val Loss: 0.0124 Acc: 0.6764

Epoch 59/59
----------
train Loss: 0.0102 Acc: 0.7157
val Loss: 0.0126 Acc: 0.6723

Best val Acc: 0.681265

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0098 Acc: 0.7046
val Loss: 0.0130 Acc: 0.6472

Epoch 1/59
----------
train Loss: 0.0057 Acc: 0.8265
val Loss: 0.0093 Acc: 0.7543

Epoch 2/59
----------
train Loss: 0.0038 Acc: 0.8836
val Loss: 0.0078 Acc: 0.7997

Epoch 3/59
----------
LR is set to 0.001
train Loss: 0.0022 Acc: 0.9426
val Loss: 0.0069 Acc: 0.8273

Epoch 4/59
----------
train Loss: 0.0018 Acc: 0.9530
val Loss: 0.0069 Acc: 0.8281

Epoch 5/59
----------
train Loss: 0.0017 Acc: 0.9553
val Loss: 0.0068 Acc: 0.8329

Epoch 6/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9625
val Loss: 0.0068 Acc: 0.8321

Epoch 7/59
----------
train Loss: 0.0015 Acc: 0.9617
val Loss: 0.0068 Acc: 0.8354

Epoch 8/59
----------
train Loss: 0.0015 Acc: 0.9646
val Loss: 0.0071 Acc: 0.8354

Epoch 9/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0015 Acc: 0.9626
val Loss: 0.0066 Acc: 0.8362

Epoch 10/59
----------
train Loss: 0.0015 Acc: 0.9604
val Loss: 0.0067 Acc: 0.8321

Epoch 11/59
----------
train Loss: 0.0015 Acc: 0.9625
val Loss: 0.0066 Acc: 0.8313

Epoch 12/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0015 Acc: 0.9613
val Loss: 0.0067 Acc: 0.8370

Epoch 13/59
----------
train Loss: 0.0015 Acc: 0.9634
val Loss: 0.0071 Acc: 0.8321

Epoch 14/59
----------
train Loss: 0.0015 Acc: 0.9633
val Loss: 0.0066 Acc: 0.8362

Epoch 15/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0015 Acc: 0.9622
val Loss: 0.0067 Acc: 0.8329

Epoch 16/59
----------
train Loss: 0.0015 Acc: 0.9629
val Loss: 0.0067 Acc: 0.8354

Epoch 17/59
----------
train Loss: 0.0015 Acc: 0.9632
val Loss: 0.0067 Acc: 0.8362

Epoch 18/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0015 Acc: 0.9606
val Loss: 0.0068 Acc: 0.8354

Epoch 19/59
----------
train Loss: 0.0015 Acc: 0.9632
val Loss: 0.0069 Acc: 0.8337

Epoch 20/59
----------
train Loss: 0.0015 Acc: 0.9601
val Loss: 0.0067 Acc: 0.8337

Epoch 21/59
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0016 Acc: 0.9616
val Loss: 0.0066 Acc: 0.8337

Epoch 22/59
----------
train Loss: 0.0015 Acc: 0.9624
val Loss: 0.0068 Acc: 0.8337

Epoch 23/59
----------
train Loss: 0.0015 Acc: 0.9648
val Loss: 0.0069 Acc: 0.8362

Epoch 24/59
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0015 Acc: 0.9621
val Loss: 0.0065 Acc: 0.8362

Epoch 25/59
----------
train Loss: 0.0015 Acc: 0.9621
val Loss: 0.0069 Acc: 0.8362

Epoch 26/59
----------
train Loss: 0.0015 Acc: 0.9623
val Loss: 0.0068 Acc: 0.8321

Epoch 27/59
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0015 Acc: 0.9625
val Loss: 0.0067 Acc: 0.8337

Epoch 28/59
----------
train Loss: 0.0015 Acc: 0.9641
val Loss: 0.0067 Acc: 0.8329

Epoch 29/59
----------
train Loss: 0.0015 Acc: 0.9648
val Loss: 0.0069 Acc: 0.8337

Epoch 30/59
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0015 Acc: 0.9628
val Loss: 0.0067 Acc: 0.8370

Epoch 31/59
----------
train Loss: 0.0016 Acc: 0.9612
val Loss: 0.0066 Acc: 0.8337

Epoch 32/59
----------
train Loss: 0.0015 Acc: 0.9617
val Loss: 0.0067 Acc: 0.8337

Epoch 33/59
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0015 Acc: 0.9653
val Loss: 0.0067 Acc: 0.8378

Epoch 34/59
----------
train Loss: 0.0015 Acc: 0.9638
val Loss: 0.0068 Acc: 0.8329

Epoch 35/59
----------
train Loss: 0.0015 Acc: 0.9640
val Loss: 0.0068 Acc: 0.8329

Epoch 36/59
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0015 Acc: 0.9625
val Loss: 0.0067 Acc: 0.8329

Epoch 37/59
----------
train Loss: 0.0015 Acc: 0.9640
val Loss: 0.0068 Acc: 0.8329

Epoch 38/59
----------
train Loss: 0.0015 Acc: 0.9643
val Loss: 0.0067 Acc: 0.8337

Epoch 39/59
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0016 Acc: 0.9614
val Loss: 0.0066 Acc: 0.8305

Epoch 40/59
----------
train Loss: 0.0015 Acc: 0.9654
val Loss: 0.0065 Acc: 0.8354

Epoch 41/59
----------
train Loss: 0.0015 Acc: 0.9641
val Loss: 0.0067 Acc: 0.8345

Epoch 42/59
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0015 Acc: 0.9606
val Loss: 0.0070 Acc: 0.8305

Epoch 43/59
----------
train Loss: 0.0015 Acc: 0.9619
val Loss: 0.0066 Acc: 0.8345

Epoch 44/59
----------
train Loss: 0.0015 Acc: 0.9639
val Loss: 0.0069 Acc: 0.8321

Epoch 45/59
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0015 Acc: 0.9637
val Loss: 0.0067 Acc: 0.8329

Epoch 46/59
----------
train Loss: 0.0015 Acc: 0.9635
val Loss: 0.0067 Acc: 0.8337

Epoch 47/59
----------
train Loss: 0.0015 Acc: 0.9642
val Loss: 0.0068 Acc: 0.8337

Epoch 48/59
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0015 Acc: 0.9631
val Loss: 0.0067 Acc: 0.8337

Epoch 49/59
----------
train Loss: 0.0015 Acc: 0.9621
val Loss: 0.0069 Acc: 0.8329

Epoch 50/59
----------
train Loss: 0.0015 Acc: 0.9627
val Loss: 0.0066 Acc: 0.8354

Epoch 51/59
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0015 Acc: 0.9646
val Loss: 0.0068 Acc: 0.8321

Epoch 52/59
----------
train Loss: 0.0015 Acc: 0.9656
val Loss: 0.0067 Acc: 0.8354

Epoch 53/59
----------
train Loss: 0.0015 Acc: 0.9616
val Loss: 0.0068 Acc: 0.8329

Epoch 54/59
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0015 Acc: 0.9626
val Loss: 0.0067 Acc: 0.8370

Epoch 55/59
----------
train Loss: 0.0015 Acc: 0.9610
val Loss: 0.0067 Acc: 0.8378

Epoch 56/59
----------
train Loss: 0.0015 Acc: 0.9632
val Loss: 0.0068 Acc: 0.8345

Epoch 57/59
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0015 Acc: 0.9645
val Loss: 0.0068 Acc: 0.8370

Epoch 58/59
----------
train Loss: 0.0015 Acc: 0.9633
val Loss: 0.0066 Acc: 0.8362

Epoch 59/59
----------
train Loss: 0.0015 Acc: 0.9628
val Loss: 0.0067 Acc: 0.8345

Best val Acc: 0.837794

---Testing---
Test accuracy: 0.960026
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 78 %
Accuracy of   BET : 96 %
Accuracy of Bigeye tuna : 77 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 95 %
Accuracy of Carcharhiniformes : 97 %
Accuracy of Conger myriaster : 100 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 98 %
Accuracy of Frigate tuna : 65 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexagrammos agrammus : 94 %
Accuracy of Hexanchiformes : 94 %
Accuracy of Konosirus punctatus : 98 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 91 %
Accuracy of Larimichthys polyactis : 99 %
Accuracy of Lateolabrax japonicus : 98 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Miichthys miiuy : 96 %
Accuracy of Mugil cephalus : 98 %
Accuracy of Myliobatiformes : 92 %
Accuracy of   NoF : 96 %
Accuracy of Oncorhynchus keta : 93 %
Accuracy of Oncorhynchus masou : 96 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pacific bluefin tuna : 63 %
Accuracy of Paralichthys olivaceus : 96 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 97 %
Accuracy of Rajiformes : 96 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 99 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 35 %
Accuracy of Southern bluefin tuna : 57 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 98 %
Accuracy of Tetraodon or Diodon : 100 %
Accuracy of Thunnus orientalis : 78 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 95 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 96 %
Accuracy of holocephalan : 95 %
Accuracy of mullet : 20 %
Accuracy of   ray : 88 %
Accuracy of rough : 91 %
Accuracy of shark : 97 %
mean: 0.9130257616232768, std: 0.14928274726324306
--------------------

run info[val: 0.2, epoch: 78, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/77
----------
LR is set to 0.01
train Loss: 0.0229 Acc: 0.4312
val Loss: 0.0165 Acc: 0.5535

Epoch 1/77
----------
train Loss: 0.0137 Acc: 0.6314
val Loss: 0.0138 Acc: 0.6298

Epoch 2/77
----------
train Loss: 0.0115 Acc: 0.6835
val Loss: 0.0125 Acc: 0.6521

Epoch 3/77
----------
train Loss: 0.0100 Acc: 0.7218
val Loss: 0.0121 Acc: 0.6756

Epoch 4/77
----------
train Loss: 0.0094 Acc: 0.7333
val Loss: 0.0117 Acc: 0.6800

Epoch 5/77
----------
train Loss: 0.0086 Acc: 0.7631
val Loss: 0.0113 Acc: 0.6882

Epoch 6/77
----------
LR is set to 0.001
train Loss: 0.0078 Acc: 0.7844
val Loss: 0.0109 Acc: 0.6995

Epoch 7/77
----------
train Loss: 0.0078 Acc: 0.7892
val Loss: 0.0109 Acc: 0.7007

Epoch 8/77
----------
train Loss: 0.0077 Acc: 0.7904
val Loss: 0.0109 Acc: 0.7028

Epoch 9/77
----------
train Loss: 0.0076 Acc: 0.7921
val Loss: 0.0109 Acc: 0.7036

Epoch 10/77
----------
train Loss: 0.0076 Acc: 0.7962
val Loss: 0.0108 Acc: 0.7040

Epoch 11/77
----------
train Loss: 0.0075 Acc: 0.7985
val Loss: 0.0109 Acc: 0.6979

Epoch 12/77
----------
LR is set to 0.00010000000000000002
train Loss: 0.0075 Acc: 0.7959
val Loss: 0.0108 Acc: 0.7048

Epoch 13/77
----------
train Loss: 0.0074 Acc: 0.7977
val Loss: 0.0108 Acc: 0.7048

Epoch 14/77
----------
train Loss: 0.0075 Acc: 0.7980
val Loss: 0.0108 Acc: 0.7052

Epoch 15/77
----------
train Loss: 0.0074 Acc: 0.8022
val Loss: 0.0108 Acc: 0.7076

Epoch 16/77
----------
train Loss: 0.0074 Acc: 0.7990
val Loss: 0.0108 Acc: 0.7060

Epoch 17/77
----------
train Loss: 0.0074 Acc: 0.8020
val Loss: 0.0108 Acc: 0.7076

Epoch 18/77
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.7997
val Loss: 0.0107 Acc: 0.7080

Epoch 19/77
----------
train Loss: 0.0074 Acc: 0.8045
val Loss: 0.0108 Acc: 0.7076

Epoch 20/77
----------
train Loss: 0.0074 Acc: 0.8023
val Loss: 0.0108 Acc: 0.7056

Epoch 21/77
----------
train Loss: 0.0074 Acc: 0.8017
val Loss: 0.0108 Acc: 0.7064

Epoch 22/77
----------
train Loss: 0.0074 Acc: 0.7993
val Loss: 0.0107 Acc: 0.7068

Epoch 23/77
----------
train Loss: 0.0075 Acc: 0.8001
val Loss: 0.0108 Acc: 0.7056

Epoch 24/77
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0074 Acc: 0.8002
val Loss: 0.0108 Acc: 0.7036

Epoch 25/77
----------
train Loss: 0.0074 Acc: 0.7987
val Loss: 0.0108 Acc: 0.7072

Epoch 26/77
----------
train Loss: 0.0074 Acc: 0.8026
val Loss: 0.0109 Acc: 0.7060

Epoch 27/77
----------
train Loss: 0.0074 Acc: 0.8011
val Loss: 0.0108 Acc: 0.7052

Epoch 28/77
----------
train Loss: 0.0074 Acc: 0.8046
val Loss: 0.0108 Acc: 0.7044

Epoch 29/77
----------
train Loss: 0.0074 Acc: 0.8052
val Loss: 0.0108 Acc: 0.7064

Epoch 30/77
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0074 Acc: 0.8010
val Loss: 0.0108 Acc: 0.7048

Epoch 31/77
----------
train Loss: 0.0074 Acc: 0.7986
val Loss: 0.0108 Acc: 0.7048

Epoch 32/77
----------
train Loss: 0.0074 Acc: 0.8004
val Loss: 0.0108 Acc: 0.7015

Epoch 33/77
----------
train Loss: 0.0074 Acc: 0.8023
val Loss: 0.0108 Acc: 0.7040

Epoch 34/77
----------
train Loss: 0.0074 Acc: 0.8000
val Loss: 0.0108 Acc: 0.7040

Epoch 35/77
----------
train Loss: 0.0074 Acc: 0.8016
val Loss: 0.0107 Acc: 0.7068

Epoch 36/77
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0074 Acc: 0.7977
val Loss: 0.0108 Acc: 0.7036

Epoch 37/77
----------
train Loss: 0.0074 Acc: 0.8001
val Loss: 0.0108 Acc: 0.7052

Epoch 38/77
----------
train Loss: 0.0074 Acc: 0.7996
val Loss: 0.0108 Acc: 0.7068

Epoch 39/77
----------
train Loss: 0.0074 Acc: 0.7995
val Loss: 0.0108 Acc: 0.7068

Epoch 40/77
----------
train Loss: 0.0074 Acc: 0.7997
val Loss: 0.0108 Acc: 0.7084

Epoch 41/77
----------
train Loss: 0.0075 Acc: 0.7973
val Loss: 0.0108 Acc: 0.7044

Epoch 42/77
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0074 Acc: 0.8025
val Loss: 0.0108 Acc: 0.7019

Epoch 43/77
----------
train Loss: 0.0074 Acc: 0.8018
val Loss: 0.0108 Acc: 0.7060

Epoch 44/77
----------
train Loss: 0.0074 Acc: 0.8016
val Loss: 0.0108 Acc: 0.7056

Epoch 45/77
----------
train Loss: 0.0074 Acc: 0.8012
val Loss: 0.0108 Acc: 0.7032

Epoch 46/77
----------
train Loss: 0.0074 Acc: 0.8018
val Loss: 0.0108 Acc: 0.7068

Epoch 47/77
----------
train Loss: 0.0075 Acc: 0.7973
val Loss: 0.0108 Acc: 0.7084

Epoch 48/77
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0074 Acc: 0.7992
val Loss: 0.0108 Acc: 0.7060

Epoch 49/77
----------
train Loss: 0.0074 Acc: 0.8010
val Loss: 0.0108 Acc: 0.7072

Epoch 50/77
----------
train Loss: 0.0074 Acc: 0.8036
val Loss: 0.0107 Acc: 0.7068

Epoch 51/77
----------
train Loss: 0.0074 Acc: 0.8031
val Loss: 0.0108 Acc: 0.7080

Epoch 52/77
----------
train Loss: 0.0074 Acc: 0.7998
val Loss: 0.0108 Acc: 0.7072

Epoch 53/77
----------
train Loss: 0.0074 Acc: 0.8015
val Loss: 0.0107 Acc: 0.7084

Epoch 54/77
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0074 Acc: 0.7980
val Loss: 0.0108 Acc: 0.7056

Epoch 55/77
----------
train Loss: 0.0074 Acc: 0.8025
val Loss: 0.0108 Acc: 0.7092

Epoch 56/77
----------
train Loss: 0.0074 Acc: 0.8049
val Loss: 0.0108 Acc: 0.7036

Epoch 57/77
----------
train Loss: 0.0074 Acc: 0.8006
val Loss: 0.0108 Acc: 0.7056

Epoch 58/77
----------
train Loss: 0.0074 Acc: 0.7995
val Loss: 0.0108 Acc: 0.7056

Epoch 59/77
----------
train Loss: 0.0074 Acc: 0.7990
val Loss: 0.0108 Acc: 0.7052

Epoch 60/77
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0074 Acc: 0.8022
val Loss: 0.0107 Acc: 0.7032

Epoch 61/77
----------
train Loss: 0.0075 Acc: 0.8007
val Loss: 0.0108 Acc: 0.7056

Epoch 62/77
----------
train Loss: 0.0074 Acc: 0.7999
val Loss: 0.0108 Acc: 0.7064

Epoch 63/77
----------
train Loss: 0.0074 Acc: 0.7972
val Loss: 0.0108 Acc: 0.7048

Epoch 64/77
----------
train Loss: 0.0074 Acc: 0.8015
val Loss: 0.0108 Acc: 0.7052

Epoch 65/77
----------
train Loss: 0.0074 Acc: 0.8039
val Loss: 0.0107 Acc: 0.7044

Epoch 66/77
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0074 Acc: 0.7987
val Loss: 0.0108 Acc: 0.7060

Epoch 67/77
----------
train Loss: 0.0074 Acc: 0.8009
val Loss: 0.0108 Acc: 0.7088

Epoch 68/77
----------
train Loss: 0.0074 Acc: 0.7990
val Loss: 0.0108 Acc: 0.7044

Epoch 69/77
----------
train Loss: 0.0074 Acc: 0.8020
val Loss: 0.0108 Acc: 0.7044

Epoch 70/77
----------
train Loss: 0.0074 Acc: 0.7973
val Loss: 0.0108 Acc: 0.7068

Epoch 71/77
----------
train Loss: 0.0074 Acc: 0.8028
val Loss: 0.0108 Acc: 0.7048

Epoch 72/77
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0074 Acc: 0.8003
val Loss: 0.0108 Acc: 0.7056

Epoch 73/77
----------
train Loss: 0.0074 Acc: 0.8034
val Loss: 0.0107 Acc: 0.7084

Epoch 74/77
----------
train Loss: 0.0075 Acc: 0.7981
val Loss: 0.0108 Acc: 0.7088

Epoch 75/77
----------
train Loss: 0.0074 Acc: 0.8007
val Loss: 0.0108 Acc: 0.7036

Epoch 76/77
----------
train Loss: 0.0074 Acc: 0.8025
val Loss: 0.0108 Acc: 0.7068

Epoch 77/77
----------
train Loss: 0.0075 Acc: 0.7991
val Loss: 0.0108 Acc: 0.7052

Best val Acc: 0.709246

---Fine tuning.---
Epoch 0/77
----------
LR is set to 0.01
train Loss: 0.0083 Acc: 0.7531
val Loss: 0.0106 Acc: 0.7060

Epoch 1/77
----------
train Loss: 0.0039 Acc: 0.8810
val Loss: 0.0083 Acc: 0.7676

Epoch 2/77
----------
train Loss: 0.0020 Acc: 0.9428
val Loss: 0.0080 Acc: 0.7810

Epoch 3/77
----------
train Loss: 0.0012 Acc: 0.9670
val Loss: 0.0078 Acc: 0.7903

Epoch 4/77
----------
train Loss: 0.0009 Acc: 0.9774
val Loss: 0.0072 Acc: 0.8074

Epoch 5/77
----------
train Loss: 0.0006 Acc: 0.9849
val Loss: 0.0074 Acc: 0.8098

Epoch 6/77
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9880
val Loss: 0.0070 Acc: 0.8187

Epoch 7/77
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0068 Acc: 0.8236

Epoch 8/77
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0068 Acc: 0.8240

Epoch 9/77
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0068 Acc: 0.8236

Epoch 10/77
----------
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8260

Epoch 11/77
----------
train Loss: 0.0003 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8248

Epoch 12/77
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8240

Epoch 13/77
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0068 Acc: 0.8228

Epoch 14/77
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0068 Acc: 0.8277

Epoch 15/77
----------
train Loss: 0.0003 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8256

Epoch 16/77
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0068 Acc: 0.8236

Epoch 17/77
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0068 Acc: 0.8248

Epoch 18/77
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8252

Epoch 19/77
----------
train Loss: 0.0002 Acc: 0.9940
val Loss: 0.0068 Acc: 0.8244

Epoch 20/77
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8281

Epoch 21/77
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8228

Epoch 22/77
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0069 Acc: 0.8268

Epoch 23/77
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8260

Epoch 24/77
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0069 Acc: 0.8256

Epoch 25/77
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0068 Acc: 0.8268

Epoch 26/77
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8256

Epoch 27/77
----------
train Loss: 0.0003 Acc: 0.9915
val Loss: 0.0068 Acc: 0.8268

Epoch 28/77
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0069 Acc: 0.8264

Epoch 29/77
----------
train Loss: 0.0002 Acc: 0.9930
val Loss: 0.0068 Acc: 0.8256

Epoch 30/77
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0069 Acc: 0.8268

Epoch 31/77
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0069 Acc: 0.8244

Epoch 32/77
----------
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8264

Epoch 33/77
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0069 Acc: 0.8252

Epoch 34/77
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8252

Epoch 35/77
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0069 Acc: 0.8252

Epoch 36/77
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8244

Epoch 37/77
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0068 Acc: 0.8236

Epoch 38/77
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0068 Acc: 0.8264

Epoch 39/77
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0068 Acc: 0.8248

Epoch 40/77
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0069 Acc: 0.8264

Epoch 41/77
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8252

Epoch 42/77
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0068 Acc: 0.8244

Epoch 43/77
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8264

Epoch 44/77
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0069 Acc: 0.8248

Epoch 45/77
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0068 Acc: 0.8248

Epoch 46/77
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8236

Epoch 47/77
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8260

Epoch 48/77
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0069 Acc: 0.8256

Epoch 49/77
----------
train Loss: 0.0003 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8252

Epoch 50/77
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8260

Epoch 51/77
----------
train Loss: 0.0003 Acc: 0.9925
val Loss: 0.0069 Acc: 0.8244

Epoch 52/77
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8236

Epoch 53/77
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8268

Epoch 54/77
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8264

Epoch 55/77
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8260

Epoch 56/77
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0069 Acc: 0.8252

Epoch 57/77
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0068 Acc: 0.8273

Epoch 58/77
----------
train Loss: 0.0003 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8256

Epoch 59/77
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8240

Epoch 60/77
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8240

Epoch 61/77
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0069 Acc: 0.8252

Epoch 62/77
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8248

Epoch 63/77
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0069 Acc: 0.8256

Epoch 64/77
----------
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8244

Epoch 65/77
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8252

Epoch 66/77
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0003 Acc: 0.9924
val Loss: 0.0069 Acc: 0.8256

Epoch 67/77
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8268

Epoch 68/77
----------
train Loss: 0.0003 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8248

Epoch 69/77
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8256

Epoch 70/77
----------
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0068 Acc: 0.8244

Epoch 71/77
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0069 Acc: 0.8232

Epoch 72/77
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0069 Acc: 0.8256

Epoch 73/77
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0068 Acc: 0.8228

Epoch 74/77
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8264

Epoch 75/77
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8252

Epoch 76/77
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8256

Epoch 77/77
----------
train Loss: 0.0003 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8256

Best val Acc: 0.828062

---Testing---
Test accuracy: 0.959539
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 79 %
Accuracy of   BET : 97 %
Accuracy of Bigeye tuna : 86 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 89 %
Accuracy of Carcharhiniformes : 93 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 96 %
Accuracy of Epinephelus septemfasciatus : 95 %
Accuracy of Frigate tuna : 82 %
Accuracy of Heterodontiformes : 96 %
Accuracy of Hexagrammos agrammus : 92 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Konosirus punctatus : 97 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 98 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 96 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 94 %
Accuracy of Mackerel tuna : 78 %
Accuracy of Miichthys miiuy : 95 %
Accuracy of Mugil cephalus : 96 %
Accuracy of Myliobatiformes : 89 %
Accuracy of   NoF : 97 %
Accuracy of Oncorhynchus keta : 95 %
Accuracy of Oncorhynchus masou : 95 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 95 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Paralichthys olivaceus : 94 %
Accuracy of Pleuronectidae : 95 %
Accuracy of Pristiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 98 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 76 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 85 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 94 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 95 %
Accuracy of holocephalan : 92 %
Accuracy of mullet : 43 %
Accuracy of   ray : 77 %
Accuracy of rough : 87 %
Accuracy of shark : 97 %
mean: 0.9293158287585886, std: 0.08888161135766945
--------------------

run info[val: 0.3, epoch: 91, randcrop: False, decay: 4]

---Training last layer.---
Epoch 0/90
----------
LR is set to 0.01
train Loss: 0.0241 Acc: 0.4087
val Loss: 0.0175 Acc: 0.5258

Epoch 1/90
----------
train Loss: 0.0144 Acc: 0.6159
val Loss: 0.0139 Acc: 0.6318

Epoch 2/90
----------
train Loss: 0.0119 Acc: 0.6743
val Loss: 0.0127 Acc: 0.6540

Epoch 3/90
----------
train Loss: 0.0105 Acc: 0.7132
val Loss: 0.0120 Acc: 0.6680

Epoch 4/90
----------
LR is set to 0.001
train Loss: 0.0093 Acc: 0.7513
val Loss: 0.0115 Acc: 0.6796

Epoch 5/90
----------
train Loss: 0.0091 Acc: 0.7570
val Loss: 0.0114 Acc: 0.6880

Epoch 6/90
----------
train Loss: 0.0090 Acc: 0.7599
val Loss: 0.0114 Acc: 0.6864

Epoch 7/90
----------
train Loss: 0.0090 Acc: 0.7578
val Loss: 0.0114 Acc: 0.6880

Epoch 8/90
----------
LR is set to 0.00010000000000000002
train Loss: 0.0089 Acc: 0.7611
val Loss: 0.0114 Acc: 0.6875

Epoch 9/90
----------
train Loss: 0.0089 Acc: 0.7625
val Loss: 0.0113 Acc: 0.6875

Epoch 10/90
----------
train Loss: 0.0089 Acc: 0.7651
val Loss: 0.0113 Acc: 0.6878

Epoch 11/90
----------
train Loss: 0.0088 Acc: 0.7630
val Loss: 0.0113 Acc: 0.6910

Epoch 12/90
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0089 Acc: 0.7659
val Loss: 0.0113 Acc: 0.6872

Epoch 13/90
----------
train Loss: 0.0089 Acc: 0.7665
val Loss: 0.0113 Acc: 0.6918

Epoch 14/90
----------
train Loss: 0.0088 Acc: 0.7612
val Loss: 0.0113 Acc: 0.6875

Epoch 15/90
----------
train Loss: 0.0088 Acc: 0.7629
val Loss: 0.0113 Acc: 0.6923

Epoch 16/90
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0088 Acc: 0.7674
val Loss: 0.0113 Acc: 0.6910

Epoch 17/90
----------
train Loss: 0.0088 Acc: 0.7703
val Loss: 0.0113 Acc: 0.6899

Epoch 18/90
----------
train Loss: 0.0088 Acc: 0.7659
val Loss: 0.0113 Acc: 0.6905

Epoch 19/90
----------
train Loss: 0.0088 Acc: 0.7662
val Loss: 0.0113 Acc: 0.6926

Epoch 20/90
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0088 Acc: 0.7660
val Loss: 0.0113 Acc: 0.6878

Epoch 21/90
----------
train Loss: 0.0089 Acc: 0.7618
val Loss: 0.0113 Acc: 0.6915

Epoch 22/90
----------
train Loss: 0.0088 Acc: 0.7663
val Loss: 0.0113 Acc: 0.6888

Epoch 23/90
----------
train Loss: 0.0088 Acc: 0.7674
val Loss: 0.0113 Acc: 0.6899

Epoch 24/90
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0088 Acc: 0.7697
val Loss: 0.0113 Acc: 0.6921

Epoch 25/90
----------
train Loss: 0.0088 Acc: 0.7693
val Loss: 0.0113 Acc: 0.6894

Epoch 26/90
----------
train Loss: 0.0088 Acc: 0.7685
val Loss: 0.0113 Acc: 0.6878

Epoch 27/90
----------
train Loss: 0.0088 Acc: 0.7719
val Loss: 0.0113 Acc: 0.6883

Epoch 28/90
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0087 Acc: 0.7697
val Loss: 0.0113 Acc: 0.6926

Epoch 29/90
----------
train Loss: 0.0088 Acc: 0.7638
val Loss: 0.0113 Acc: 0.6902

Epoch 30/90
----------
train Loss: 0.0089 Acc: 0.7662
val Loss: 0.0114 Acc: 0.6905

Epoch 31/90
----------
train Loss: 0.0088 Acc: 0.7613
val Loss: 0.0113 Acc: 0.6891

Epoch 32/90
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0088 Acc: 0.7678
val Loss: 0.0113 Acc: 0.6905

Epoch 33/90
----------
train Loss: 0.0088 Acc: 0.7652
val Loss: 0.0113 Acc: 0.6899

Epoch 34/90
----------
train Loss: 0.0088 Acc: 0.7662
val Loss: 0.0113 Acc: 0.6886

Epoch 35/90
----------
train Loss: 0.0088 Acc: 0.7685
val Loss: 0.0113 Acc: 0.6896

Epoch 36/90
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0088 Acc: 0.7665
val Loss: 0.0113 Acc: 0.6894

Epoch 37/90
----------
train Loss: 0.0088 Acc: 0.7704
val Loss: 0.0113 Acc: 0.6905

Epoch 38/90
----------
train Loss: 0.0088 Acc: 0.7685
val Loss: 0.0113 Acc: 0.6891

Epoch 39/90
----------
train Loss: 0.0088 Acc: 0.7675
val Loss: 0.0113 Acc: 0.6910

Epoch 40/90
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0089 Acc: 0.7651
val Loss: 0.0113 Acc: 0.6894

Epoch 41/90
----------
train Loss: 0.0088 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6915

Epoch 42/90
----------
train Loss: 0.0088 Acc: 0.7652
val Loss: 0.0113 Acc: 0.6929

Epoch 43/90
----------
train Loss: 0.0088 Acc: 0.7667
val Loss: 0.0113 Acc: 0.6899

Epoch 44/90
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0088 Acc: 0.7653
val Loss: 0.0113 Acc: 0.6883

Epoch 45/90
----------
train Loss: 0.0088 Acc: 0.7697
val Loss: 0.0113 Acc: 0.6921

Epoch 46/90
----------
train Loss: 0.0088 Acc: 0.7707
val Loss: 0.0113 Acc: 0.6888

Epoch 47/90
----------
train Loss: 0.0088 Acc: 0.7663
val Loss: 0.0113 Acc: 0.6883

Epoch 48/90
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0089 Acc: 0.7643
val Loss: 0.0113 Acc: 0.6907

Epoch 49/90
----------
train Loss: 0.0089 Acc: 0.7651
val Loss: 0.0113 Acc: 0.6918

Epoch 50/90
----------
train Loss: 0.0089 Acc: 0.7642
val Loss: 0.0113 Acc: 0.6891

Epoch 51/90
----------
train Loss: 0.0088 Acc: 0.7644
val Loss: 0.0113 Acc: 0.6913

Epoch 52/90
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0088 Acc: 0.7682
val Loss: 0.0113 Acc: 0.6899

Epoch 53/90
----------
train Loss: 0.0089 Acc: 0.7642
val Loss: 0.0113 Acc: 0.6883

Epoch 54/90
----------
train Loss: 0.0089 Acc: 0.7623
val Loss: 0.0113 Acc: 0.6910

Epoch 55/90
----------
train Loss: 0.0088 Acc: 0.7641
val Loss: 0.0113 Acc: 0.6918

Epoch 56/90
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0088 Acc: 0.7641
val Loss: 0.0113 Acc: 0.6894

Epoch 57/90
----------
train Loss: 0.0088 Acc: 0.7627
val Loss: 0.0113 Acc: 0.6910

Epoch 58/90
----------
train Loss: 0.0088 Acc: 0.7674
val Loss: 0.0113 Acc: 0.6899

Epoch 59/90
----------
train Loss: 0.0088 Acc: 0.7621
val Loss: 0.0113 Acc: 0.6913

Epoch 60/90
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0088 Acc: 0.7675
val Loss: 0.0113 Acc: 0.6913

Epoch 61/90
----------
train Loss: 0.0088 Acc: 0.7689
val Loss: 0.0113 Acc: 0.6915

Epoch 62/90
----------
train Loss: 0.0089 Acc: 0.7651
val Loss: 0.0113 Acc: 0.6926

Epoch 63/90
----------
train Loss: 0.0088 Acc: 0.7691
val Loss: 0.0113 Acc: 0.6905

Epoch 64/90
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0088 Acc: 0.7682
val Loss: 0.0113 Acc: 0.6894

Epoch 65/90
----------
train Loss: 0.0088 Acc: 0.7633
val Loss: 0.0114 Acc: 0.6888

Epoch 66/90
----------
train Loss: 0.0088 Acc: 0.7619
val Loss: 0.0113 Acc: 0.6921

Epoch 67/90
----------
train Loss: 0.0089 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6894

Epoch 68/90
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0089 Acc: 0.7637
val Loss: 0.0113 Acc: 0.6905

Epoch 69/90
----------
train Loss: 0.0088 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6907

Epoch 70/90
----------
train Loss: 0.0088 Acc: 0.7651
val Loss: 0.0113 Acc: 0.6902

Epoch 71/90
----------
train Loss: 0.0089 Acc: 0.7688
val Loss: 0.0113 Acc: 0.6880

Epoch 72/90
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0088 Acc: 0.7648
val Loss: 0.0114 Acc: 0.6875

Epoch 73/90
----------
train Loss: 0.0088 Acc: 0.7662
val Loss: 0.0113 Acc: 0.6923

Epoch 74/90
----------
train Loss: 0.0088 Acc: 0.7671
val Loss: 0.0113 Acc: 0.6894

Epoch 75/90
----------
train Loss: 0.0088 Acc: 0.7685
val Loss: 0.0113 Acc: 0.6942

Epoch 76/90
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0088 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6886

Epoch 77/90
----------
train Loss: 0.0088 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6896

Epoch 78/90
----------
train Loss: 0.0088 Acc: 0.7716
val Loss: 0.0113 Acc: 0.6883

Epoch 79/90
----------
train Loss: 0.0088 Acc: 0.7649
val Loss: 0.0113 Acc: 0.6907

Epoch 80/90
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0088 Acc: 0.7703
val Loss: 0.0113 Acc: 0.6907

Epoch 81/90
----------
train Loss: 0.0088 Acc: 0.7678
val Loss: 0.0113 Acc: 0.6915

Epoch 82/90
----------
train Loss: 0.0088 Acc: 0.7669
val Loss: 0.0113 Acc: 0.6918

Epoch 83/90
----------
train Loss: 0.0088 Acc: 0.7634
val Loss: 0.0113 Acc: 0.6880

Epoch 84/90
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0088 Acc: 0.7678
val Loss: 0.0113 Acc: 0.6896

Epoch 85/90
----------
train Loss: 0.0088 Acc: 0.7677
val Loss: 0.0114 Acc: 0.6910

Epoch 86/90
----------
train Loss: 0.0088 Acc: 0.7695
val Loss: 0.0113 Acc: 0.6918

Epoch 87/90
----------
train Loss: 0.0088 Acc: 0.7697
val Loss: 0.0113 Acc: 0.6926

Epoch 88/90
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0088 Acc: 0.7684
val Loss: 0.0113 Acc: 0.6886

Epoch 89/90
----------
train Loss: 0.0088 Acc: 0.7673
val Loss: 0.0113 Acc: 0.6915

Epoch 90/90
----------
train Loss: 0.0088 Acc: 0.7691
val Loss: 0.0113 Acc: 0.6907

Best val Acc: 0.694242

---Fine tuning.---
Epoch 0/90
----------
LR is set to 0.01
train Loss: 0.0086 Acc: 0.7431
val Loss: 0.0095 Acc: 0.7234

Epoch 1/90
----------
train Loss: 0.0043 Acc: 0.8739
val Loss: 0.0079 Acc: 0.7751

Epoch 2/90
----------
train Loss: 0.0023 Acc: 0.9378
val Loss: 0.0076 Acc: 0.7821

Epoch 3/90
----------
train Loss: 0.0015 Acc: 0.9610
val Loss: 0.0072 Acc: 0.8008

Epoch 4/90
----------
LR is set to 0.001
train Loss: 0.0008 Acc: 0.9811
val Loss: 0.0066 Acc: 0.8181

Epoch 5/90
----------
train Loss: 0.0007 Acc: 0.9861
val Loss: 0.0065 Acc: 0.8197

Epoch 6/90
----------
train Loss: 0.0006 Acc: 0.9870
val Loss: 0.0065 Acc: 0.8256

Epoch 7/90
----------
train Loss: 0.0006 Acc: 0.9881
val Loss: 0.0064 Acc: 0.8243

Epoch 8/90
----------
LR is set to 0.00010000000000000002
train Loss: 0.0005 Acc: 0.9911
val Loss: 0.0064 Acc: 0.8224

Epoch 9/90
----------
train Loss: 0.0005 Acc: 0.9903
val Loss: 0.0064 Acc: 0.8240

Epoch 10/90
----------
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0064 Acc: 0.8227

Epoch 11/90
----------
train Loss: 0.0005 Acc: 0.9889
val Loss: 0.0065 Acc: 0.8229

Epoch 12/90
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0005 Acc: 0.9896
val Loss: 0.0064 Acc: 0.8251

Epoch 13/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0064 Acc: 0.8245

Epoch 14/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8245

Epoch 15/90
----------
train Loss: 0.0005 Acc: 0.9903
val Loss: 0.0064 Acc: 0.8243

Epoch 16/90
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0064 Acc: 0.8224

Epoch 17/90
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8245

Epoch 18/90
----------
train Loss: 0.0005 Acc: 0.9891
val Loss: 0.0064 Acc: 0.8243

Epoch 19/90
----------
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0064 Acc: 0.8243

Epoch 20/90
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8248

Epoch 21/90
----------
train Loss: 0.0005 Acc: 0.9903
val Loss: 0.0064 Acc: 0.8237

Epoch 22/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0064 Acc: 0.8237

Epoch 23/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0064 Acc: 0.8240

Epoch 24/90
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0064 Acc: 0.8224

Epoch 25/90
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8245

Epoch 26/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0064 Acc: 0.8254

Epoch 27/90
----------
train Loss: 0.0005 Acc: 0.9913
val Loss: 0.0064 Acc: 0.8237

Epoch 28/90
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8243

Epoch 29/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8248

Epoch 30/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8245

Epoch 31/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0065 Acc: 0.8237

Epoch 32/90
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0064 Acc: 0.8245

Epoch 33/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0064 Acc: 0.8245

Epoch 34/90
----------
train Loss: 0.0005 Acc: 0.9909
val Loss: 0.0064 Acc: 0.8243

Epoch 35/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0065 Acc: 0.8254

Epoch 36/90
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0005 Acc: 0.9905
val Loss: 0.0064 Acc: 0.8248

Epoch 37/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0064 Acc: 0.8237

Epoch 38/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0064 Acc: 0.8254

Epoch 39/90
----------
train Loss: 0.0005 Acc: 0.9905
val Loss: 0.0064 Acc: 0.8256

Epoch 40/90
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0064 Acc: 0.8224

Epoch 41/90
----------
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0065 Acc: 0.8240

Epoch 42/90
----------
train Loss: 0.0005 Acc: 0.9891
val Loss: 0.0064 Acc: 0.8243

Epoch 43/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0065 Acc: 0.8235

Epoch 44/90
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8227

Epoch 45/90
----------
train Loss: 0.0005 Acc: 0.9885
val Loss: 0.0064 Acc: 0.8259

Epoch 46/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0065 Acc: 0.8245

Epoch 47/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0065 Acc: 0.8229

Epoch 48/90
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8240

Epoch 49/90
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8245

Epoch 50/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0064 Acc: 0.8254

Epoch 51/90
----------
train Loss: 0.0005 Acc: 0.9892
val Loss: 0.0065 Acc: 0.8235

Epoch 52/90
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0005 Acc: 0.9905
val Loss: 0.0064 Acc: 0.8248

Epoch 53/90
----------
train Loss: 0.0005 Acc: 0.9905
val Loss: 0.0064 Acc: 0.8229

Epoch 54/90
----------
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0065 Acc: 0.8224

Epoch 55/90
----------
train Loss: 0.0005 Acc: 0.9890
val Loss: 0.0064 Acc: 0.8243

Epoch 56/90
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0065 Acc: 0.8232

Epoch 57/90
----------
train Loss: 0.0005 Acc: 0.9885
val Loss: 0.0064 Acc: 0.8251

Epoch 58/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0064 Acc: 0.8237

Epoch 59/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8229

Epoch 60/90
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0005 Acc: 0.9910
val Loss: 0.0064 Acc: 0.8237

Epoch 61/90
----------
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0064 Acc: 0.8259

Epoch 62/90
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0064 Acc: 0.8256

Epoch 63/90
----------
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0064 Acc: 0.8232

Epoch 64/90
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0064 Acc: 0.8237

Epoch 65/90
----------
train Loss: 0.0005 Acc: 0.9907
val Loss: 0.0064 Acc: 0.8235

Epoch 66/90
----------
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0064 Acc: 0.8243

Epoch 67/90
----------
train Loss: 0.0005 Acc: 0.9896
val Loss: 0.0065 Acc: 0.8227

Epoch 68/90
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0064 Acc: 0.8243

Epoch 69/90
----------
train Loss: 0.0005 Acc: 0.9888
val Loss: 0.0064 Acc: 0.8245

Epoch 70/90
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8245

Epoch 71/90
----------
train Loss: 0.0005 Acc: 0.9893
val Loss: 0.0064 Acc: 0.8235

Epoch 72/90
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0064 Acc: 0.8248

Epoch 73/90
----------
train Loss: 0.0005 Acc: 0.9897
val Loss: 0.0064 Acc: 0.8248

Epoch 74/90
----------
train Loss: 0.0005 Acc: 0.9909
val Loss: 0.0064 Acc: 0.8235

Epoch 75/90
----------
train Loss: 0.0005 Acc: 0.9904
val Loss: 0.0065 Acc: 0.8235

Epoch 76/90
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0005 Acc: 0.9896
val Loss: 0.0064 Acc: 0.8232

Epoch 77/90
----------
train Loss: 0.0005 Acc: 0.9900
val Loss: 0.0064 Acc: 0.8227

Epoch 78/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0064 Acc: 0.8235

Epoch 79/90
----------
train Loss: 0.0005 Acc: 0.9905
val Loss: 0.0064 Acc: 0.8267

Epoch 80/90
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0005 Acc: 0.9892
val Loss: 0.0064 Acc: 0.8218

Epoch 81/90
----------
train Loss: 0.0005 Acc: 0.9892
val Loss: 0.0064 Acc: 0.8259

Epoch 82/90
----------
train Loss: 0.0005 Acc: 0.9898
val Loss: 0.0064 Acc: 0.8221

Epoch 83/90
----------
train Loss: 0.0005 Acc: 0.9912
val Loss: 0.0064 Acc: 0.8248

Epoch 84/90
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0005 Acc: 0.9902
val Loss: 0.0065 Acc: 0.8240

Epoch 85/90
----------
train Loss: 0.0005 Acc: 0.9910
val Loss: 0.0064 Acc: 0.8229

Epoch 86/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0064 Acc: 0.8240

Epoch 87/90
----------
train Loss: 0.0005 Acc: 0.9896
val Loss: 0.0064 Acc: 0.8235

Epoch 88/90
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0005 Acc: 0.9903
val Loss: 0.0064 Acc: 0.8235

Epoch 89/90
----------
train Loss: 0.0005 Acc: 0.9899
val Loss: 0.0064 Acc: 0.8240

Epoch 90/90
----------
train Loss: 0.0005 Acc: 0.9895
val Loss: 0.0064 Acc: 0.8240

Best val Acc: 0.826710

---Testing---
Test accuracy: 0.941620
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 83 %
Accuracy of Atlantic bluefin tuna : 75 %
Accuracy of   BET : 98 %
Accuracy of Bigeye tuna : 82 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 83 %
Accuracy of Carcharhiniformes : 90 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 94 %
Accuracy of Frigate tuna : 62 %
Accuracy of Heterodontiformes : 93 %
Accuracy of Hexagrammos agrammus : 83 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Konosirus punctatus : 96 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 96 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 94 %
Accuracy of Little tunny : 82 %
Accuracy of Longtail tuna : 87 %
Accuracy of Mackerel tuna : 72 %
Accuracy of Miichthys miiuy : 93 %
Accuracy of Mugil cephalus : 93 %
Accuracy of Myliobatiformes : 85 %
Accuracy of   NoF : 96 %
Accuracy of Oncorhynchus keta : 88 %
Accuracy of Oncorhynchus masou : 92 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 95 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Paralichthys olivaceus : 90 %
Accuracy of Pleuronectidae : 94 %
Accuracy of Pristiformes : 94 %
Accuracy of Rajiformes : 94 %
Accuracy of Rhinobatiformes : 95 %
Accuracy of SHARK on boat : 99 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 98 %
Accuracy of Seriola quinqueradiata : 95 %
Accuracy of Skipjack tuna : 91 %
Accuracy of Slender tuna : 71 %
Accuracy of Southern bluefin tuna : 61 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 98 %
Accuracy of Tetraodon or Diodon : 98 %
Accuracy of Thunnus orientalis : 77 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 92 %
Accuracy of   YFT : 98 %
Accuracy of Yellowfin tuna : 93 %
Accuracy of holocephalan : 91 %
Accuracy of mullet : 41 %
Accuracy of   ray : 75 %
Accuracy of rough : 79 %
Accuracy of shark : 95 %
mean: 0.8974072804089772, std: 0.11326492761624402

Model saved in "./weights/all_in_one_[0.96]_mean[0.91]_std[0.15].save".
--------------------

run info[val: 0.1, epoch: 69, randcrop: False, decay: 3]

---Training last layer.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0223 Acc: 0.4444
val Loss: 0.0167 Acc: 0.5807

Epoch 1/68
----------
train Loss: 0.0134 Acc: 0.6405
val Loss: 0.0142 Acc: 0.6318

Epoch 2/68
----------
train Loss: 0.0112 Acc: 0.6856
val Loss: 0.0128 Acc: 0.6683

Epoch 3/68
----------
LR is set to 0.001
train Loss: 0.0097 Acc: 0.7393
val Loss: 0.0121 Acc: 0.6869

Epoch 4/68
----------
train Loss: 0.0096 Acc: 0.7400
val Loss: 0.0120 Acc: 0.6926

Epoch 5/68
----------
train Loss: 0.0094 Acc: 0.7457
val Loss: 0.0120 Acc: 0.6869

Epoch 6/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0093 Acc: 0.7498
val Loss: 0.0121 Acc: 0.6878

Epoch 7/68
----------
train Loss: 0.0093 Acc: 0.7496
val Loss: 0.0120 Acc: 0.6959

Epoch 8/68
----------
train Loss: 0.0093 Acc: 0.7507
val Loss: 0.0120 Acc: 0.6934

Epoch 9/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0092 Acc: 0.7540
val Loss: 0.0120 Acc: 0.6983

Epoch 10/68
----------
train Loss: 0.0093 Acc: 0.7509
val Loss: 0.0121 Acc: 0.6926

Epoch 11/68
----------
train Loss: 0.0092 Acc: 0.7529
val Loss: 0.0119 Acc: 0.6894

Epoch 12/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0093 Acc: 0.7525
val Loss: 0.0119 Acc: 0.6959

Epoch 13/68
----------
train Loss: 0.0093 Acc: 0.7524
val Loss: 0.0120 Acc: 0.6886

Epoch 14/68
----------
train Loss: 0.0092 Acc: 0.7512
val Loss: 0.0120 Acc: 0.6967

Epoch 15/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0093 Acc: 0.7491
val Loss: 0.0121 Acc: 0.6975

Epoch 16/68
----------
train Loss: 0.0093 Acc: 0.7464
val Loss: 0.0120 Acc: 0.6959

Epoch 17/68
----------
train Loss: 0.0092 Acc: 0.7510
val Loss: 0.0120 Acc: 0.6926

Epoch 18/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0093 Acc: 0.7514
val Loss: 0.0119 Acc: 0.6983

Epoch 19/68
----------
train Loss: 0.0093 Acc: 0.7468
val Loss: 0.0120 Acc: 0.6942

Epoch 20/68
----------
train Loss: 0.0093 Acc: 0.7523
val Loss: 0.0122 Acc: 0.6926

Epoch 21/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0093 Acc: 0.7484
val Loss: 0.0120 Acc: 0.6959

Epoch 22/68
----------
train Loss: 0.0093 Acc: 0.7523
val Loss: 0.0121 Acc: 0.6926

Epoch 23/68
----------
train Loss: 0.0092 Acc: 0.7526
val Loss: 0.0119 Acc: 0.6942

Epoch 24/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0093 Acc: 0.7529
val Loss: 0.0121 Acc: 0.6934

Epoch 25/68
----------
train Loss: 0.0093 Acc: 0.7508
val Loss: 0.0121 Acc: 0.6926

Epoch 26/68
----------
train Loss: 0.0093 Acc: 0.7519
val Loss: 0.0121 Acc: 0.6951

Epoch 27/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0093 Acc: 0.7499
val Loss: 0.0119 Acc: 0.6934

Epoch 28/68
----------
train Loss: 0.0092 Acc: 0.7505
val Loss: 0.0122 Acc: 0.6934

Epoch 29/68
----------
train Loss: 0.0092 Acc: 0.7521
val Loss: 0.0122 Acc: 0.6918

Epoch 30/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0093 Acc: 0.7486
val Loss: 0.0120 Acc: 0.6999

Epoch 31/68
----------
train Loss: 0.0093 Acc: 0.7504
val Loss: 0.0121 Acc: 0.6942

Epoch 32/68
----------
train Loss: 0.0093 Acc: 0.7493
val Loss: 0.0120 Acc: 0.6926

Epoch 33/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0093 Acc: 0.7515
val Loss: 0.0120 Acc: 0.6926

Epoch 34/68
----------
train Loss: 0.0092 Acc: 0.7505
val Loss: 0.0119 Acc: 0.6934

Epoch 35/68
----------
train Loss: 0.0092 Acc: 0.7502
val Loss: 0.0121 Acc: 0.6975

Epoch 36/68
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0093 Acc: 0.7455
val Loss: 0.0119 Acc: 0.6967

Epoch 37/68
----------
train Loss: 0.0093 Acc: 0.7487
val Loss: 0.0121 Acc: 0.6926

Epoch 38/68
----------
train Loss: 0.0092 Acc: 0.7543
val Loss: 0.0119 Acc: 0.6926

Epoch 39/68
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0093 Acc: 0.7508
val Loss: 0.0119 Acc: 0.6926

Epoch 40/68
----------
train Loss: 0.0093 Acc: 0.7487
val Loss: 0.0119 Acc: 0.6926

Epoch 41/68
----------
train Loss: 0.0092 Acc: 0.7547
val Loss: 0.0120 Acc: 0.6934

Epoch 42/68
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0093 Acc: 0.7493
val Loss: 0.0121 Acc: 0.6959

Epoch 43/68
----------
train Loss: 0.0093 Acc: 0.7543
val Loss: 0.0119 Acc: 0.6942

Epoch 44/68
----------
train Loss: 0.0093 Acc: 0.7480
val Loss: 0.0119 Acc: 0.6951

Epoch 45/68
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0093 Acc: 0.7522
val Loss: 0.0120 Acc: 0.6967

Epoch 46/68
----------
train Loss: 0.0093 Acc: 0.7488
val Loss: 0.0122 Acc: 0.6902

Epoch 47/68
----------
train Loss: 0.0093 Acc: 0.7491
val Loss: 0.0119 Acc: 0.6967

Epoch 48/68
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0093 Acc: 0.7507
val Loss: 0.0121 Acc: 0.6902

Epoch 49/68
----------
train Loss: 0.0093 Acc: 0.7470
val Loss: 0.0121 Acc: 0.6918

Epoch 50/68
----------
train Loss: 0.0093 Acc: 0.7483
val Loss: 0.0117 Acc: 0.6951

Epoch 51/68
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0092 Acc: 0.7514
val Loss: 0.0120 Acc: 0.6951

Epoch 52/68
----------
train Loss: 0.0093 Acc: 0.7459
val Loss: 0.0121 Acc: 0.6910

Epoch 53/68
----------
train Loss: 0.0093 Acc: 0.7509
val Loss: 0.0118 Acc: 0.6951

Epoch 54/68
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0093 Acc: 0.7454
val Loss: 0.0120 Acc: 0.6942

Epoch 55/68
----------
train Loss: 0.0093 Acc: 0.7546
val Loss: 0.0120 Acc: 0.6934

Epoch 56/68
----------
train Loss: 0.0093 Acc: 0.7496
val Loss: 0.0119 Acc: 0.6942

Epoch 57/68
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0093 Acc: 0.7501
val Loss: 0.0119 Acc: 0.6942

Epoch 58/68
----------
train Loss: 0.0092 Acc: 0.7493
val Loss: 0.0120 Acc: 0.6934

Epoch 59/68
----------
train Loss: 0.0093 Acc: 0.7494
val Loss: 0.0119 Acc: 0.6926

Epoch 60/68
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0092 Acc: 0.7497
val Loss: 0.0119 Acc: 0.6942

Epoch 61/68
----------
train Loss: 0.0092 Acc: 0.7515
val Loss: 0.0120 Acc: 0.6934

Epoch 62/68
----------
train Loss: 0.0092 Acc: 0.7497
val Loss: 0.0119 Acc: 0.6934

Epoch 63/68
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0093 Acc: 0.7464
val Loss: 0.0120 Acc: 0.6951

Epoch 64/68
----------
train Loss: 0.0092 Acc: 0.7493
val Loss: 0.0119 Acc: 0.6910

Epoch 65/68
----------
train Loss: 0.0093 Acc: 0.7504
val Loss: 0.0118 Acc: 0.6951

Epoch 66/68
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0093 Acc: 0.7536
val Loss: 0.0121 Acc: 0.6951

Epoch 67/68
----------
train Loss: 0.0093 Acc: 0.7495
val Loss: 0.0120 Acc: 0.6942

Epoch 68/68
----------
train Loss: 0.0093 Acc: 0.7492
val Loss: 0.0120 Acc: 0.6894

Best val Acc: 0.699919

---Fine tuning.---
Epoch 0/68
----------
LR is set to 0.01
train Loss: 0.0087 Acc: 0.7377
val Loss: 0.0104 Acc: 0.7234

Epoch 1/68
----------
train Loss: 0.0042 Acc: 0.8725
val Loss: 0.0086 Acc: 0.7729

Epoch 2/68
----------
train Loss: 0.0024 Acc: 0.9288
val Loss: 0.0082 Acc: 0.7875

Epoch 3/68
----------
LR is set to 0.001
train Loss: 0.0013 Acc: 0.9694
val Loss: 0.0068 Acc: 0.8289

Epoch 4/68
----------
train Loss: 0.0010 Acc: 0.9774
val Loss: 0.0070 Acc: 0.8305

Epoch 5/68
----------
train Loss: 0.0009 Acc: 0.9805
val Loss: 0.0067 Acc: 0.8289

Epoch 6/68
----------
LR is set to 0.00010000000000000002
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0068 Acc: 0.8281

Epoch 7/68
----------
train Loss: 0.0008 Acc: 0.9834
val Loss: 0.0070 Acc: 0.8273

Epoch 8/68
----------
train Loss: 0.0008 Acc: 0.9839
val Loss: 0.0068 Acc: 0.8329

Epoch 9/68
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0008 Acc: 0.9829
val Loss: 0.0067 Acc: 0.8313

Epoch 10/68
----------
train Loss: 0.0008 Acc: 0.9827
val Loss: 0.0068 Acc: 0.8313

Epoch 11/68
----------
train Loss: 0.0008 Acc: 0.9821
val Loss: 0.0067 Acc: 0.8313

Epoch 12/68
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0008 Acc: 0.9829
val Loss: 0.0071 Acc: 0.8273

Epoch 13/68
----------
train Loss: 0.0008 Acc: 0.9830
val Loss: 0.0065 Acc: 0.8289

Epoch 14/68
----------
train Loss: 0.0008 Acc: 0.9826
val Loss: 0.0068 Acc: 0.8273

Epoch 15/68
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0071 Acc: 0.8289

Epoch 16/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0067 Acc: 0.8281

Epoch 17/68
----------
train Loss: 0.0008 Acc: 0.9841
val Loss: 0.0067 Acc: 0.8297

Epoch 18/68
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0068 Acc: 0.8281

Epoch 19/68
----------
train Loss: 0.0008 Acc: 0.9839
val Loss: 0.0067 Acc: 0.8297

Epoch 20/68
----------
train Loss: 0.0008 Acc: 0.9824
val Loss: 0.0069 Acc: 0.8313

Epoch 21/68
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0069 Acc: 0.8329

Epoch 22/68
----------
train Loss: 0.0008 Acc: 0.9828
val Loss: 0.0067 Acc: 0.8281

Epoch 23/68
----------
train Loss: 0.0008 Acc: 0.9829
val Loss: 0.0069 Acc: 0.8297

Epoch 24/68
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0008 Acc: 0.9840
val Loss: 0.0066 Acc: 0.8321

Epoch 25/68
----------
train Loss: 0.0008 Acc: 0.9845
val Loss: 0.0067 Acc: 0.8313

Epoch 26/68
----------
train Loss: 0.0008 Acc: 0.9825
val Loss: 0.0068 Acc: 0.8297

Epoch 27/68
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0069 Acc: 0.8289

Epoch 28/68
----------
train Loss: 0.0008 Acc: 0.9850
val Loss: 0.0069 Acc: 0.8281

Epoch 29/68
----------
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0068 Acc: 0.8305

Epoch 30/68
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0065 Acc: 0.8297

Epoch 31/68
----------
train Loss: 0.0008 Acc: 0.9828
val Loss: 0.0068 Acc: 0.8305

Epoch 32/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0066 Acc: 0.8289

Epoch 33/68
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0067 Acc: 0.8305

Epoch 34/68
----------
train Loss: 0.0008 Acc: 0.9824
val Loss: 0.0068 Acc: 0.8305

Epoch 35/68
----------
train Loss: 0.0008 Acc: 0.9821
val Loss: 0.0068 Acc: 0.8321

Epoch 36/68
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0071 Acc: 0.8273

Epoch 37/68
----------
train Loss: 0.0008 Acc: 0.9824
val Loss: 0.0070 Acc: 0.8289

Epoch 38/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0069 Acc: 0.8289

Epoch 39/68
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0008 Acc: 0.9836
val Loss: 0.0066 Acc: 0.8281

Epoch 40/68
----------
train Loss: 0.0008 Acc: 0.9830
val Loss: 0.0067 Acc: 0.8297

Epoch 41/68
----------
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0068 Acc: 0.8289

Epoch 42/68
----------
LR is set to 1.0000000000000008e-16
train Loss: 0.0008 Acc: 0.9824
val Loss: 0.0068 Acc: 0.8289

Epoch 43/68
----------
train Loss: 0.0008 Acc: 0.9820
val Loss: 0.0067 Acc: 0.8264

Epoch 44/68
----------
train Loss: 0.0008 Acc: 0.9850
val Loss: 0.0067 Acc: 0.8281

Epoch 45/68
----------
LR is set to 1.0000000000000008e-17
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0067 Acc: 0.8289

Epoch 46/68
----------
train Loss: 0.0008 Acc: 0.9837
val Loss: 0.0068 Acc: 0.8289

Epoch 47/68
----------
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0071 Acc: 0.8313

Epoch 48/68
----------
LR is set to 1.0000000000000008e-18
train Loss: 0.0008 Acc: 0.9838
val Loss: 0.0067 Acc: 0.8281

Epoch 49/68
----------
train Loss: 0.0008 Acc: 0.9841
val Loss: 0.0068 Acc: 0.8273

Epoch 50/68
----------
train Loss: 0.0008 Acc: 0.9832
val Loss: 0.0067 Acc: 0.8313

Epoch 51/68
----------
LR is set to 1.000000000000001e-19
train Loss: 0.0008 Acc: 0.9842
val Loss: 0.0067 Acc: 0.8297

Epoch 52/68
----------
train Loss: 0.0008 Acc: 0.9829
val Loss: 0.0066 Acc: 0.8305

Epoch 53/68
----------
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0066 Acc: 0.8305

Epoch 54/68
----------
LR is set to 1.000000000000001e-20
train Loss: 0.0008 Acc: 0.9830
val Loss: 0.0066 Acc: 0.8297

Epoch 55/68
----------
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0067 Acc: 0.8281

Epoch 56/68
----------
train Loss: 0.0008 Acc: 0.9820
val Loss: 0.0068 Acc: 0.8281

Epoch 57/68
----------
LR is set to 1.000000000000001e-21
train Loss: 0.0008 Acc: 0.9815
val Loss: 0.0069 Acc: 0.8281

Epoch 58/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0072 Acc: 0.8329

Epoch 59/68
----------
train Loss: 0.0008 Acc: 0.9823
val Loss: 0.0066 Acc: 0.8305

Epoch 60/68
----------
LR is set to 1.0000000000000012e-22
train Loss: 0.0008 Acc: 0.9841
val Loss: 0.0067 Acc: 0.8313

Epoch 61/68
----------
train Loss: 0.0008 Acc: 0.9828
val Loss: 0.0066 Acc: 0.8297

Epoch 62/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0067 Acc: 0.8321

Epoch 63/68
----------
LR is set to 1.0000000000000013e-23
train Loss: 0.0008 Acc: 0.9831
val Loss: 0.0066 Acc: 0.8289

Epoch 64/68
----------
train Loss: 0.0008 Acc: 0.9841
val Loss: 0.0068 Acc: 0.8313

Epoch 65/68
----------
train Loss: 0.0008 Acc: 0.9833
val Loss: 0.0069 Acc: 0.8273

Epoch 66/68
----------
LR is set to 1.0000000000000012e-24
train Loss: 0.0008 Acc: 0.9841
val Loss: 0.0068 Acc: 0.8264

Epoch 67/68
----------
train Loss: 0.0008 Acc: 0.9830
val Loss: 0.0067 Acc: 0.8313

Epoch 68/68
----------
train Loss: 0.0008 Acc: 0.9822
val Loss: 0.0069 Acc: 0.8297

Best val Acc: 0.832928

---Testing---
Test accuracy: 0.968945
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 92 %
Accuracy of Atlantic bluefin tuna : 87 %
Accuracy of   BET : 99 %
Accuracy of Bigeye tuna : 86 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 96 %
Accuracy of Carcharhiniformes : 96 %
Accuracy of Conger myriaster : 99 %
Accuracy of   DOL : 100 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 98 %
Accuracy of Frigate tuna : 72 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexagrammos agrammus : 97 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 99 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 96 %
Accuracy of Larimichthys polyactis : 99 %
Accuracy of Lateolabrax japonicus : 98 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 84 %
Accuracy of Miichthys miiuy : 97 %
Accuracy of Mugil cephalus : 99 %
Accuracy of Myliobatiformes : 90 %
Accuracy of   NoF : 98 %
Accuracy of Oncorhynchus keta : 95 %
Accuracy of Oncorhynchus masou : 95 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 96 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Paralichthys olivaceus : 98 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 97 %
Accuracy of Rajiformes : 100 %
Accuracy of Rhinobatiformes : 98 %
Accuracy of SHARK on boat : 99 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 99 %
Accuracy of Skipjack tuna : 97 %
Accuracy of Slender tuna : 71 %
Accuracy of Southern bluefin tuna : 73 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 93 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 82 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 94 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 97 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 22 %
Accuracy of   ray : 88 %
Accuracy of rough : 91 %
Accuracy of shark : 98 %
mean: 0.9373879293860156, std: 0.1163320830621631
--------------------

run info[val: 0.2, epoch: 44, randcrop: True, decay: 12]

---Training last layer.---
Epoch 0/43
----------
LR is set to 0.01
train Loss: 0.0235 Acc: 0.4132
val Loss: 0.0168 Acc: 0.5677

Epoch 1/43
----------
train Loss: 0.0145 Acc: 0.6050
val Loss: 0.0141 Acc: 0.6237

Epoch 2/43
----------
train Loss: 0.0123 Acc: 0.6520
val Loss: 0.0135 Acc: 0.6237

Epoch 3/43
----------
train Loss: 0.0112 Acc: 0.6814
val Loss: 0.0123 Acc: 0.6553

Epoch 4/43
----------
train Loss: 0.0102 Acc: 0.7095
val Loss: 0.0117 Acc: 0.6606

Epoch 5/43
----------
train Loss: 0.0097 Acc: 0.7175
val Loss: 0.0113 Acc: 0.6813

Epoch 6/43
----------
train Loss: 0.0091 Acc: 0.7325
val Loss: 0.0115 Acc: 0.6821

Epoch 7/43
----------
train Loss: 0.0088 Acc: 0.7414
val Loss: 0.0111 Acc: 0.6878

Epoch 8/43
----------
train Loss: 0.0086 Acc: 0.7494
val Loss: 0.0109 Acc: 0.6983

Epoch 9/43
----------
train Loss: 0.0083 Acc: 0.7511
val Loss: 0.0108 Acc: 0.6942

Epoch 10/43
----------
train Loss: 0.0080 Acc: 0.7662
val Loss: 0.0110 Acc: 0.6926

Epoch 11/43
----------
train Loss: 0.0079 Acc: 0.7659
val Loss: 0.0106 Acc: 0.7024

Epoch 12/43
----------
LR is set to 0.001
train Loss: 0.0072 Acc: 0.7891
val Loss: 0.0103 Acc: 0.7105

Epoch 13/43
----------
train Loss: 0.0072 Acc: 0.7930
val Loss: 0.0103 Acc: 0.7080

Epoch 14/43
----------
train Loss: 0.0072 Acc: 0.7917
val Loss: 0.0103 Acc: 0.7121

Epoch 15/43
----------
train Loss: 0.0071 Acc: 0.7926
val Loss: 0.0103 Acc: 0.7097

Epoch 16/43
----------
train Loss: 0.0071 Acc: 0.7906
val Loss: 0.0103 Acc: 0.7121

Epoch 17/43
----------
train Loss: 0.0071 Acc: 0.7924
val Loss: 0.0103 Acc: 0.7101

Epoch 18/43
----------
train Loss: 0.0071 Acc: 0.7977
val Loss: 0.0103 Acc: 0.7088

Epoch 19/43
----------
train Loss: 0.0071 Acc: 0.7939
val Loss: 0.0103 Acc: 0.7084

Epoch 20/43
----------
train Loss: 0.0070 Acc: 0.7969
val Loss: 0.0103 Acc: 0.7084

Epoch 21/43
----------
train Loss: 0.0070 Acc: 0.7996
val Loss: 0.0103 Acc: 0.7125

Epoch 22/43
----------
train Loss: 0.0070 Acc: 0.7968
val Loss: 0.0103 Acc: 0.7141

Epoch 23/43
----------
train Loss: 0.0070 Acc: 0.7986
val Loss: 0.0103 Acc: 0.7125

Epoch 24/43
----------
LR is set to 0.00010000000000000002
train Loss: 0.0069 Acc: 0.8027
val Loss: 0.0103 Acc: 0.7117

Epoch 25/43
----------
train Loss: 0.0069 Acc: 0.7982
val Loss: 0.0102 Acc: 0.7101

Epoch 26/43
----------
train Loss: 0.0070 Acc: 0.7965
val Loss: 0.0103 Acc: 0.7117

Epoch 27/43
----------
train Loss: 0.0069 Acc: 0.8030
val Loss: 0.0103 Acc: 0.7097

Epoch 28/43
----------
train Loss: 0.0069 Acc: 0.8024
val Loss: 0.0102 Acc: 0.7113

Epoch 29/43
----------
train Loss: 0.0069 Acc: 0.7999
val Loss: 0.0102 Acc: 0.7084

Epoch 30/43
----------
train Loss: 0.0069 Acc: 0.7982
val Loss: 0.0102 Acc: 0.7080

Epoch 31/43
----------
train Loss: 0.0069 Acc: 0.8000
val Loss: 0.0102 Acc: 0.7101

Epoch 32/43
----------
train Loss: 0.0070 Acc: 0.7947
val Loss: 0.0102 Acc: 0.7145

Epoch 33/43
----------
train Loss: 0.0069 Acc: 0.8007
val Loss: 0.0102 Acc: 0.7109

Epoch 34/43
----------
train Loss: 0.0069 Acc: 0.7955
val Loss: 0.0103 Acc: 0.7121

Epoch 35/43
----------
train Loss: 0.0069 Acc: 0.7969
val Loss: 0.0102 Acc: 0.7121

Epoch 36/43
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0069 Acc: 0.8012
val Loss: 0.0102 Acc: 0.7092

Epoch 37/43
----------
train Loss: 0.0069 Acc: 0.8009
val Loss: 0.0102 Acc: 0.7105

Epoch 38/43
----------
train Loss: 0.0068 Acc: 0.8019
val Loss: 0.0102 Acc: 0.7105

Epoch 39/43
----------
train Loss: 0.0069 Acc: 0.8009
val Loss: 0.0103 Acc: 0.7117

Epoch 40/43
----------
train Loss: 0.0069 Acc: 0.8033
val Loss: 0.0102 Acc: 0.7137

Epoch 41/43
----------
train Loss: 0.0069 Acc: 0.8019
val Loss: 0.0103 Acc: 0.7121

Epoch 42/43
----------
train Loss: 0.0069 Acc: 0.8011
val Loss: 0.0102 Acc: 0.7088

Epoch 43/43
----------
train Loss: 0.0069 Acc: 0.8022
val Loss: 0.0102 Acc: 0.7088

Best val Acc: 0.714517

---Fine tuning.---
Epoch 0/43
----------
LR is set to 0.01
train Loss: 0.0090 Acc: 0.7232
val Loss: 0.0128 Acc: 0.6655

Epoch 1/43
----------
train Loss: 0.0054 Acc: 0.8320
val Loss: 0.0095 Acc: 0.7579

Epoch 2/43
----------
train Loss: 0.0032 Acc: 0.8979
val Loss: 0.0083 Acc: 0.7822

Epoch 3/43
----------
train Loss: 0.0022 Acc: 0.9345
val Loss: 0.0083 Acc: 0.7895

Epoch 4/43
----------
train Loss: 0.0016 Acc: 0.9516
val Loss: 0.0085 Acc: 0.7741

Epoch 5/43
----------
train Loss: 0.0012 Acc: 0.9652
val Loss: 0.0083 Acc: 0.7952

Epoch 6/43
----------
train Loss: 0.0010 Acc: 0.9712
val Loss: 0.0082 Acc: 0.8005

Epoch 7/43
----------
train Loss: 0.0008 Acc: 0.9772
val Loss: 0.0081 Acc: 0.7997

Epoch 8/43
----------
train Loss: 0.0007 Acc: 0.9802
val Loss: 0.0081 Acc: 0.8054

Epoch 9/43
----------
train Loss: 0.0005 Acc: 0.9846
val Loss: 0.0081 Acc: 0.8135

Epoch 10/43
----------
train Loss: 0.0005 Acc: 0.9845
val Loss: 0.0081 Acc: 0.8127

Epoch 11/43
----------
train Loss: 0.0005 Acc: 0.9824
val Loss: 0.0082 Acc: 0.8102

Epoch 12/43
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9878
val Loss: 0.0079 Acc: 0.8187

Epoch 13/43
----------
train Loss: 0.0003 Acc: 0.9885
val Loss: 0.0079 Acc: 0.8212

Epoch 14/43
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0079 Acc: 0.8228

Epoch 15/43
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0079 Acc: 0.8228

Epoch 16/43
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0078 Acc: 0.8228

Epoch 17/43
----------
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0079 Acc: 0.8220

Epoch 18/43
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0079 Acc: 0.8232

Epoch 19/43
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0079 Acc: 0.8240

Epoch 20/43
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0078 Acc: 0.8228

Epoch 21/43
----------
train Loss: 0.0002 Acc: 0.9904
val Loss: 0.0079 Acc: 0.8232

Epoch 22/43
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0079 Acc: 0.8248

Epoch 23/43
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0079 Acc: 0.8216

Epoch 24/43
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0079 Acc: 0.8220

Epoch 25/43
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0079 Acc: 0.8244

Epoch 26/43
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0079 Acc: 0.8220

Epoch 27/43
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0079 Acc: 0.8224

Epoch 28/43
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0079 Acc: 0.8208

Epoch 29/43
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0079 Acc: 0.8232

Epoch 30/43
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0078 Acc: 0.8204

Epoch 31/43
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0079 Acc: 0.8224

Epoch 32/43
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0080 Acc: 0.8216

Epoch 33/43
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0080 Acc: 0.8236

Epoch 34/43
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0079 Acc: 0.8224

Epoch 35/43
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0079 Acc: 0.8220

Epoch 36/43
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0078 Acc: 0.8240

Epoch 37/43
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0079 Acc: 0.8220

Epoch 38/43
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0079 Acc: 0.8216

Epoch 39/43
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0079 Acc: 0.8224

Epoch 40/43
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0078 Acc: 0.8232

Epoch 41/43
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0080 Acc: 0.8248

Epoch 42/43
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0079 Acc: 0.8228

Epoch 43/43
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0079 Acc: 0.8220

Best val Acc: 0.824818

---Testing---
Test accuracy: 0.958648
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 92 %
Accuracy of Atlantic bluefin tuna : 71 %
Accuracy of   BET : 98 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 99 %
Accuracy of Bullet tuna : 87 %
Accuracy of Carcharhiniformes : 92 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 94 %
Accuracy of Frigate tuna : 82 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexagrammos agrammus : 92 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 96 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 98 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 97 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 92 %
Accuracy of Mackerel tuna : 76 %
Accuracy of Miichthys miiuy : 95 %
Accuracy of Mugil cephalus : 97 %
Accuracy of Myliobatiformes : 89 %
Accuracy of   NoF : 96 %
Accuracy of Oncorhynchus keta : 93 %
Accuracy of Oncorhynchus masou : 92 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 94 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Paralichthys olivaceus : 94 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 97 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 96 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 98 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 80 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 93 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 100 %
Accuracy of Thunnus orientalis : 86 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 94 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 95 %
Accuracy of holocephalan : 93 %
Accuracy of mullet : 35 %
Accuracy of   ray : 79 %
Accuracy of rough : 88 %
Accuracy of shark : 97 %
mean: 0.9276429997234261, std: 0.09867908639567796
--------------------

run info[val: 0.3, epoch: 60, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0242 Acc: 0.4131
val Loss: 0.0170 Acc: 0.5610

Epoch 1/59
----------
train Loss: 0.0145 Acc: 0.6149
val Loss: 0.0140 Acc: 0.6231

Epoch 2/59
----------
train Loss: 0.0119 Acc: 0.6774
val Loss: 0.0134 Acc: 0.6218

Epoch 3/59
----------
train Loss: 0.0105 Acc: 0.7108
val Loss: 0.0119 Acc: 0.6750

Epoch 4/59
----------
train Loss: 0.0095 Acc: 0.7372
val Loss: 0.0115 Acc: 0.6845

Epoch 5/59
----------
train Loss: 0.0090 Acc: 0.7492
val Loss: 0.0111 Acc: 0.6856

Epoch 6/59
----------
train Loss: 0.0084 Acc: 0.7695
val Loss: 0.0109 Acc: 0.6923

Epoch 7/59
----------
train Loss: 0.0081 Acc: 0.7716
val Loss: 0.0107 Acc: 0.6959

Epoch 8/59
----------
train Loss: 0.0076 Acc: 0.7852
val Loss: 0.0107 Acc: 0.7072

Epoch 9/59
----------
LR is set to 0.001
train Loss: 0.0070 Acc: 0.8109
val Loss: 0.0103 Acc: 0.7156

Epoch 10/59
----------
train Loss: 0.0068 Acc: 0.8177
val Loss: 0.0103 Acc: 0.7145

Epoch 11/59
----------
train Loss: 0.0068 Acc: 0.8164
val Loss: 0.0103 Acc: 0.7170

Epoch 12/59
----------
train Loss: 0.0068 Acc: 0.8169
val Loss: 0.0102 Acc: 0.7164

Epoch 13/59
----------
train Loss: 0.0068 Acc: 0.8193
val Loss: 0.0103 Acc: 0.7186

Epoch 14/59
----------
train Loss: 0.0068 Acc: 0.8182
val Loss: 0.0102 Acc: 0.7159

Epoch 15/59
----------
train Loss: 0.0067 Acc: 0.8163
val Loss: 0.0102 Acc: 0.7183

Epoch 16/59
----------
train Loss: 0.0067 Acc: 0.8205
val Loss: 0.0102 Acc: 0.7172

Epoch 17/59
----------
train Loss: 0.0067 Acc: 0.8206
val Loss: 0.0102 Acc: 0.7186

Epoch 18/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0066 Acc: 0.8231
val Loss: 0.0102 Acc: 0.7202

Epoch 19/59
----------
train Loss: 0.0066 Acc: 0.8281
val Loss: 0.0102 Acc: 0.7180

Epoch 20/59
----------
train Loss: 0.0066 Acc: 0.8201
val Loss: 0.0102 Acc: 0.7210

Epoch 21/59
----------
train Loss: 0.0065 Acc: 0.8246
val Loss: 0.0102 Acc: 0.7183

Epoch 22/59
----------
train Loss: 0.0065 Acc: 0.8249
val Loss: 0.0102 Acc: 0.7188

Epoch 23/59
----------
train Loss: 0.0065 Acc: 0.8275
val Loss: 0.0102 Acc: 0.7205

Epoch 24/59
----------
train Loss: 0.0066 Acc: 0.8265
val Loss: 0.0102 Acc: 0.7161

Epoch 25/59
----------
train Loss: 0.0065 Acc: 0.8237
val Loss: 0.0102 Acc: 0.7175

Epoch 26/59
----------
train Loss: 0.0066 Acc: 0.8273
val Loss: 0.0102 Acc: 0.7210

Epoch 27/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0065 Acc: 0.8244
val Loss: 0.0102 Acc: 0.7207

Epoch 28/59
----------
train Loss: 0.0065 Acc: 0.8307
val Loss: 0.0102 Acc: 0.7191

Epoch 29/59
----------
train Loss: 0.0065 Acc: 0.8300
val Loss: 0.0102 Acc: 0.7178

Epoch 30/59
----------
train Loss: 0.0065 Acc: 0.8294
val Loss: 0.0102 Acc: 0.7180

Epoch 31/59
----------
train Loss: 0.0065 Acc: 0.8243
val Loss: 0.0102 Acc: 0.7164

Epoch 32/59
----------
train Loss: 0.0065 Acc: 0.8256
val Loss: 0.0102 Acc: 0.7197

Epoch 33/59
----------
train Loss: 0.0065 Acc: 0.8271
val Loss: 0.0101 Acc: 0.7188

Epoch 34/59
----------
train Loss: 0.0066 Acc: 0.8263
val Loss: 0.0102 Acc: 0.7172

Epoch 35/59
----------
train Loss: 0.0065 Acc: 0.8245
val Loss: 0.0102 Acc: 0.7194

Epoch 36/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0065 Acc: 0.8253
val Loss: 0.0102 Acc: 0.7183

Epoch 37/59
----------
train Loss: 0.0065 Acc: 0.8260
val Loss: 0.0102 Acc: 0.7167

Epoch 38/59
----------
train Loss: 0.0065 Acc: 0.8284
val Loss: 0.0102 Acc: 0.7199

Epoch 39/59
----------
train Loss: 0.0065 Acc: 0.8293
val Loss: 0.0102 Acc: 0.7191

Epoch 40/59
----------
train Loss: 0.0065 Acc: 0.8241
val Loss: 0.0101 Acc: 0.7188

Epoch 41/59
----------
train Loss: 0.0066 Acc: 0.8216
val Loss: 0.0102 Acc: 0.7170

Epoch 42/59
----------
train Loss: 0.0066 Acc: 0.8234
val Loss: 0.0102 Acc: 0.7186

Epoch 43/59
----------
train Loss: 0.0066 Acc: 0.8240
val Loss: 0.0102 Acc: 0.7191

Epoch 44/59
----------
train Loss: 0.0066 Acc: 0.8255
val Loss: 0.0102 Acc: 0.7183

Epoch 45/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0065 Acc: 0.8230
val Loss: 0.0102 Acc: 0.7170

Epoch 46/59
----------
train Loss: 0.0066 Acc: 0.8270
val Loss: 0.0102 Acc: 0.7197

Epoch 47/59
----------
train Loss: 0.0065 Acc: 0.8296
val Loss: 0.0102 Acc: 0.7205

Epoch 48/59
----------
train Loss: 0.0066 Acc: 0.8231
val Loss: 0.0102 Acc: 0.7191

Epoch 49/59
----------
train Loss: 0.0065 Acc: 0.8234
val Loss: 0.0102 Acc: 0.7221

Epoch 50/59
----------
train Loss: 0.0066 Acc: 0.8241
val Loss: 0.0102 Acc: 0.7207

Epoch 51/59
----------
train Loss: 0.0066 Acc: 0.8259
val Loss: 0.0102 Acc: 0.7207

Epoch 52/59
----------
train Loss: 0.0066 Acc: 0.8265
val Loss: 0.0101 Acc: 0.7218

Epoch 53/59
----------
train Loss: 0.0066 Acc: 0.8231
val Loss: 0.0102 Acc: 0.7183

Epoch 54/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0065 Acc: 0.8273
val Loss: 0.0102 Acc: 0.7191

Epoch 55/59
----------
train Loss: 0.0066 Acc: 0.8248
val Loss: 0.0102 Acc: 0.7197

Epoch 56/59
----------
train Loss: 0.0065 Acc: 0.8252
val Loss: 0.0101 Acc: 0.7183

Epoch 57/59
----------
train Loss: 0.0065 Acc: 0.8228
val Loss: 0.0102 Acc: 0.7186

Epoch 58/59
----------
train Loss: 0.0066 Acc: 0.8278
val Loss: 0.0102 Acc: 0.7210

Epoch 59/59
----------
train Loss: 0.0065 Acc: 0.8281
val Loss: 0.0102 Acc: 0.7188

Best val Acc: 0.722087

---Fine tuning.---
Epoch 0/59
----------
LR is set to 0.01
train Loss: 0.0083 Acc: 0.7516
val Loss: 0.0115 Acc: 0.6775

Epoch 1/59
----------
train Loss: 0.0039 Acc: 0.8823
val Loss: 0.0088 Acc: 0.7656

Epoch 2/59
----------
train Loss: 0.0021 Acc: 0.9398
val Loss: 0.0076 Acc: 0.7910

Epoch 3/59
----------
train Loss: 0.0011 Acc: 0.9719
val Loss: 0.0070 Acc: 0.8105

Epoch 4/59
----------
train Loss: 0.0009 Acc: 0.9786
val Loss: 0.0070 Acc: 0.8194

Epoch 5/59
----------
train Loss: 0.0006 Acc: 0.9833
val Loss: 0.0070 Acc: 0.8167

Epoch 6/59
----------
train Loss: 0.0005 Acc: 0.9854
val Loss: 0.0068 Acc: 0.8210

Epoch 7/59
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0068 Acc: 0.8232

Epoch 8/59
----------
train Loss: 0.0004 Acc: 0.9886
val Loss: 0.0071 Acc: 0.8254

Epoch 9/59
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0069 Acc: 0.8267

Epoch 10/59
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0068 Acc: 0.8283

Epoch 11/59
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8321

Epoch 12/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8300

Epoch 13/59
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8321

Epoch 14/59
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8324

Epoch 15/59
----------
train Loss: 0.0002 Acc: 0.9936
val Loss: 0.0068 Acc: 0.8324

Epoch 16/59
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8327

Epoch 17/59
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8335

Epoch 18/59
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8329

Epoch 19/59
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8335

Epoch 20/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8324

Epoch 21/59
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8329

Epoch 22/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8340

Epoch 23/59
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8340

Epoch 24/59
----------
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0068 Acc: 0.8332

Epoch 25/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8332

Epoch 26/59
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8316

Epoch 27/59
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0068 Acc: 0.8332

Epoch 28/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8327

Epoch 29/59
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8356

Epoch 30/59
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8327

Epoch 31/59
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8335

Epoch 32/59
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8324

Epoch 33/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8327

Epoch 34/59
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0068 Acc: 0.8337

Epoch 35/59
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0068 Acc: 0.8343

Epoch 36/59
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8340

Epoch 37/59
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0068 Acc: 0.8335

Epoch 38/59
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8335

Epoch 39/59
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8335

Epoch 40/59
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8329

Epoch 41/59
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8343

Epoch 42/59
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8332

Epoch 43/59
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8332

Epoch 44/59
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8332

Epoch 45/59
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0067 Acc: 0.8340

Epoch 46/59
----------
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0068 Acc: 0.8351

Epoch 47/59
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8335

Epoch 48/59
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8321

Epoch 49/59
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8337

Epoch 50/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8332

Epoch 51/59
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8351

Epoch 52/59
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8362

Epoch 53/59
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8332

Epoch 54/59
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8337

Epoch 55/59
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0068 Acc: 0.8351

Epoch 56/59
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8332

Epoch 57/59
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0068 Acc: 0.8348

Epoch 58/59
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8343

Epoch 59/59
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8318

Best val Acc: 0.836172

---Testing---
Test accuracy: 0.945836
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 73 %
Accuracy of   BET : 97 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 96 %
Accuracy of Bullet tuna : 83 %
Accuracy of Carcharhiniformes : 92 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 94 %
Accuracy of Frigate tuna : 65 %
Accuracy of Heterodontiformes : 92 %
Accuracy of Hexagrammos agrammus : 83 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 96 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 94 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 95 %
Accuracy of Little tunny : 84 %
Accuracy of Longtail tuna : 89 %
Accuracy of Mackerel tuna : 72 %
Accuracy of Miichthys miiuy : 95 %
Accuracy of Mugil cephalus : 95 %
Accuracy of Myliobatiformes : 85 %
Accuracy of   NoF : 96 %
Accuracy of Oncorhynchus keta : 89 %
Accuracy of Oncorhynchus masou : 92 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 96 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Paralichthys olivaceus : 91 %
Accuracy of Pleuronectidae : 93 %
Accuracy of Pristiformes : 94 %
Accuracy of Rajiformes : 94 %
Accuracy of Rhinobatiformes : 93 %
Accuracy of SHARK on boat : 99 %
Accuracy of Scomber japonicus : 96 %
Accuracy of Sebastes inermis : 98 %
Accuracy of Seriola quinqueradiata : 94 %
Accuracy of Skipjack tuna : 93 %
Accuracy of Slender tuna : 71 %
Accuracy of Southern bluefin tuna : 68 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 97 %
Accuracy of Tetraodon or Diodon : 100 %
Accuracy of Thunnus orientalis : 81 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 91 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 94 %
Accuracy of holocephalan : 91 %
Accuracy of mullet : 44 %
Accuracy of   ray : 73 %
Accuracy of rough : 80 %
Accuracy of shark : 97 %
mean: 0.9044931141801882, std: 0.10535639172111633

Model saved in "./weights/all_in_one_[0.97]_mean[0.94]_std[0.12].save".
--------------------

run info[val: 0.1, epoch: 64, randcrop: False, decay: 6]

---Training last layer.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0225 Acc: 0.4450
val Loss: 0.0165 Acc: 0.5799

Epoch 1/63
----------
train Loss: 0.0134 Acc: 0.6302
val Loss: 0.0144 Acc: 0.6188

Epoch 2/63
----------
train Loss: 0.0112 Acc: 0.6891
val Loss: 0.0131 Acc: 0.6586

Epoch 3/63
----------
train Loss: 0.0100 Acc: 0.7166
val Loss: 0.0121 Acc: 0.6740

Epoch 4/63
----------
train Loss: 0.0091 Acc: 0.7425
val Loss: 0.0118 Acc: 0.7015

Epoch 5/63
----------
train Loss: 0.0087 Acc: 0.7510
val Loss: 0.0115 Acc: 0.6934

Epoch 6/63
----------
LR is set to 0.001
train Loss: 0.0078 Acc: 0.7832
val Loss: 0.0112 Acc: 0.7097

Epoch 7/63
----------
train Loss: 0.0076 Acc: 0.7886
val Loss: 0.0112 Acc: 0.7129

Epoch 8/63
----------
train Loss: 0.0076 Acc: 0.7926
val Loss: 0.0111 Acc: 0.7129

Epoch 9/63
----------
train Loss: 0.0076 Acc: 0.7920
val Loss: 0.0114 Acc: 0.7129

Epoch 10/63
----------
train Loss: 0.0075 Acc: 0.7936
val Loss: 0.0113 Acc: 0.7105

Epoch 11/63
----------
train Loss: 0.0075 Acc: 0.7929
val Loss: 0.0111 Acc: 0.7186

Epoch 12/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0074 Acc: 0.7991
val Loss: 0.0110 Acc: 0.7129

Epoch 13/63
----------
train Loss: 0.0074 Acc: 0.7974
val Loss: 0.0109 Acc: 0.7178

Epoch 14/63
----------
train Loss: 0.0074 Acc: 0.8010
val Loss: 0.0110 Acc: 0.7194

Epoch 15/63
----------
train Loss: 0.0074 Acc: 0.7986
val Loss: 0.0108 Acc: 0.7153

Epoch 16/63
----------
train Loss: 0.0074 Acc: 0.7966
val Loss: 0.0110 Acc: 0.7210

Epoch 17/63
----------
train Loss: 0.0073 Acc: 0.8028
val Loss: 0.0111 Acc: 0.7178

Epoch 18/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0074 Acc: 0.7985
val Loss: 0.0109 Acc: 0.7194

Epoch 19/63
----------
train Loss: 0.0074 Acc: 0.7946
val Loss: 0.0112 Acc: 0.7170

Epoch 20/63
----------
train Loss: 0.0074 Acc: 0.7986
val Loss: 0.0109 Acc: 0.7202

Epoch 21/63
----------
train Loss: 0.0073 Acc: 0.8002
val Loss: 0.0110 Acc: 0.7202

Epoch 22/63
----------
train Loss: 0.0073 Acc: 0.7982
val Loss: 0.0109 Acc: 0.7210

Epoch 23/63
----------
train Loss: 0.0074 Acc: 0.7950
val Loss: 0.0108 Acc: 0.7210

Epoch 24/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0073 Acc: 0.7990
val Loss: 0.0111 Acc: 0.7178

Epoch 25/63
----------
train Loss: 0.0074 Acc: 0.7993
val Loss: 0.0110 Acc: 0.7137

Epoch 26/63
----------
train Loss: 0.0073 Acc: 0.8018
val Loss: 0.0109 Acc: 0.7170

Epoch 27/63
----------
train Loss: 0.0074 Acc: 0.7981
val Loss: 0.0110 Acc: 0.7153

Epoch 28/63
----------
train Loss: 0.0074 Acc: 0.7977
val Loss: 0.0110 Acc: 0.7186

Epoch 29/63
----------
train Loss: 0.0073 Acc: 0.7978
val Loss: 0.0111 Acc: 0.7129

Epoch 30/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0073 Acc: 0.8014
val Loss: 0.0109 Acc: 0.7129

Epoch 31/63
----------
train Loss: 0.0074 Acc: 0.8001
val Loss: 0.0112 Acc: 0.7178

Epoch 32/63
----------
train Loss: 0.0074 Acc: 0.8034
val Loss: 0.0108 Acc: 0.7161

Epoch 33/63
----------
train Loss: 0.0073 Acc: 0.8002
val Loss: 0.0110 Acc: 0.7161

Epoch 34/63
----------
train Loss: 0.0074 Acc: 0.7976
val Loss: 0.0113 Acc: 0.7186

Epoch 35/63
----------
train Loss: 0.0073 Acc: 0.8010
val Loss: 0.0112 Acc: 0.7170

Epoch 36/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0074 Acc: 0.7996
val Loss: 0.0110 Acc: 0.7170

Epoch 37/63
----------
train Loss: 0.0073 Acc: 0.7994
val Loss: 0.0110 Acc: 0.7145

Epoch 38/63
----------
train Loss: 0.0073 Acc: 0.7997
val Loss: 0.0109 Acc: 0.7194

Epoch 39/63
----------
train Loss: 0.0074 Acc: 0.8017
val Loss: 0.0109 Acc: 0.7210

Epoch 40/63
----------
train Loss: 0.0074 Acc: 0.7927
val Loss: 0.0111 Acc: 0.7161

Epoch 41/63
----------
train Loss: 0.0074 Acc: 0.7971
val Loss: 0.0110 Acc: 0.7170

Epoch 42/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0073 Acc: 0.8008
val Loss: 0.0110 Acc: 0.7194

Epoch 43/63
----------
train Loss: 0.0074 Acc: 0.7999
val Loss: 0.0111 Acc: 0.7186

Epoch 44/63
----------
train Loss: 0.0073 Acc: 0.8012
val Loss: 0.0111 Acc: 0.7129

Epoch 45/63
----------
train Loss: 0.0073 Acc: 0.8020
val Loss: 0.0110 Acc: 0.7161

Epoch 46/63
----------
train Loss: 0.0074 Acc: 0.8023
val Loss: 0.0111 Acc: 0.7202

Epoch 47/63
----------
train Loss: 0.0073 Acc: 0.8007
val Loss: 0.0112 Acc: 0.7145

Epoch 48/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0073 Acc: 0.7996
val Loss: 0.0110 Acc: 0.7170

Epoch 49/63
----------
train Loss: 0.0073 Acc: 0.7975
val Loss: 0.0109 Acc: 0.7202

Epoch 50/63
----------
train Loss: 0.0073 Acc: 0.7981
val Loss: 0.0110 Acc: 0.7186

Epoch 51/63
----------
train Loss: 0.0074 Acc: 0.8009
val Loss: 0.0112 Acc: 0.7170

Epoch 52/63
----------
train Loss: 0.0074 Acc: 0.8032
val Loss: 0.0111 Acc: 0.7186

Epoch 53/63
----------
train Loss: 0.0073 Acc: 0.8002
val Loss: 0.0110 Acc: 0.7170

Epoch 54/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0074 Acc: 0.7955
val Loss: 0.0109 Acc: 0.7153

Epoch 55/63
----------
train Loss: 0.0073 Acc: 0.7999
val Loss: 0.0109 Acc: 0.7145

Epoch 56/63
----------
train Loss: 0.0074 Acc: 0.8005
val Loss: 0.0111 Acc: 0.7161

Epoch 57/63
----------
train Loss: 0.0073 Acc: 0.7984
val Loss: 0.0108 Acc: 0.7145

Epoch 58/63
----------
train Loss: 0.0074 Acc: 0.7995
val Loss: 0.0109 Acc: 0.7194

Epoch 59/63
----------
train Loss: 0.0073 Acc: 0.8010
val Loss: 0.0111 Acc: 0.7210

Epoch 60/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0074 Acc: 0.7980
val Loss: 0.0112 Acc: 0.7161

Epoch 61/63
----------
train Loss: 0.0073 Acc: 0.8006
val Loss: 0.0108 Acc: 0.7186

Epoch 62/63
----------
train Loss: 0.0073 Acc: 0.7989
val Loss: 0.0110 Acc: 0.7178

Epoch 63/63
----------
train Loss: 0.0073 Acc: 0.8020
val Loss: 0.0112 Acc: 0.7234

Best val Acc: 0.723439

---Fine tuning.---
Epoch 0/63
----------
LR is set to 0.01
train Loss: 0.0083 Acc: 0.7474
val Loss: 0.0105 Acc: 0.7275

Epoch 1/63
----------
train Loss: 0.0039 Acc: 0.8825
val Loss: 0.0084 Acc: 0.7875

Epoch 2/63
----------
train Loss: 0.0022 Acc: 0.9338
val Loss: 0.0077 Acc: 0.8045

Epoch 3/63
----------
train Loss: 0.0013 Acc: 0.9634
val Loss: 0.0083 Acc: 0.8037

Epoch 4/63
----------
train Loss: 0.0008 Acc: 0.9786
val Loss: 0.0073 Acc: 0.8135

Epoch 5/63
----------
train Loss: 0.0006 Acc: 0.9841
val Loss: 0.0079 Acc: 0.8118

Epoch 6/63
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9865
val Loss: 0.0074 Acc: 0.8232

Epoch 7/63
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0073 Acc: 0.8264

Epoch 8/63
----------
train Loss: 0.0003 Acc: 0.9892
val Loss: 0.0072 Acc: 0.8273

Epoch 9/63
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8297

Epoch 10/63
----------
train Loss: 0.0003 Acc: 0.9902
val Loss: 0.0073 Acc: 0.8264

Epoch 11/63
----------
train Loss: 0.0003 Acc: 0.9896
val Loss: 0.0075 Acc: 0.8256

Epoch 12/63
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0073 Acc: 0.8281

Epoch 13/63
----------
train Loss: 0.0003 Acc: 0.9923
val Loss: 0.0076 Acc: 0.8289

Epoch 14/63
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0074 Acc: 0.8289

Epoch 15/63
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0073 Acc: 0.8264

Epoch 16/63
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0074 Acc: 0.8289

Epoch 17/63
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0075 Acc: 0.8264

Epoch 18/63
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0077 Acc: 0.8281

Epoch 19/63
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0071 Acc: 0.8264

Epoch 20/63
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0072 Acc: 0.8289

Epoch 21/63
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0074 Acc: 0.8281

Epoch 22/63
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0073 Acc: 0.8289

Epoch 23/63
----------
train Loss: 0.0003 Acc: 0.9919
val Loss: 0.0074 Acc: 0.8273

Epoch 24/63
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0072 Acc: 0.8256

Epoch 25/63
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0073 Acc: 0.8281

Epoch 26/63
----------
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0072 Acc: 0.8256

Epoch 27/63
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0076 Acc: 0.8264

Epoch 28/63
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0072 Acc: 0.8281

Epoch 29/63
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0075 Acc: 0.8281

Epoch 30/63
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9915
val Loss: 0.0073 Acc: 0.8289

Epoch 31/63
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0073 Acc: 0.8281

Epoch 32/63
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0073 Acc: 0.8289

Epoch 33/63
----------
train Loss: 0.0003 Acc: 0.9927
val Loss: 0.0073 Acc: 0.8281

Epoch 34/63
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0072 Acc: 0.8256

Epoch 35/63
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0074 Acc: 0.8264

Epoch 36/63
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0071 Acc: 0.8281

Epoch 37/63
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0076 Acc: 0.8273

Epoch 38/63
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0074 Acc: 0.8273

Epoch 39/63
----------
train Loss: 0.0003 Acc: 0.9918
val Loss: 0.0071 Acc: 0.8297

Epoch 40/63
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0072 Acc: 0.8305

Epoch 41/63
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0075 Acc: 0.8264

Epoch 42/63
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0077 Acc: 0.8297

Epoch 43/63
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0075 Acc: 0.8256

Epoch 44/63
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0073 Acc: 0.8297

Epoch 45/63
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0073 Acc: 0.8256

Epoch 46/63
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0076 Acc: 0.8273

Epoch 47/63
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0071 Acc: 0.8264

Epoch 48/63
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0003 Acc: 0.9920
val Loss: 0.0074 Acc: 0.8305

Epoch 49/63
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0072 Acc: 0.8264

Epoch 50/63
----------
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0075 Acc: 0.8281

Epoch 51/63
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0075 Acc: 0.8256

Epoch 52/63
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0073 Acc: 0.8264

Epoch 53/63
----------
train Loss: 0.0003 Acc: 0.9911
val Loss: 0.0074 Acc: 0.8281

Epoch 54/63
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0003 Acc: 0.9922
val Loss: 0.0071 Acc: 0.8248

Epoch 55/63
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0072 Acc: 0.8297

Epoch 56/63
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0073 Acc: 0.8273

Epoch 57/63
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0076 Acc: 0.8281

Epoch 58/63
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0073 Acc: 0.8281

Epoch 59/63
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0071 Acc: 0.8313

Epoch 60/63
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0074 Acc: 0.8297

Epoch 61/63
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8273

Epoch 62/63
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0080 Acc: 0.8273

Epoch 63/63
----------
train Loss: 0.0003 Acc: 0.9915
val Loss: 0.0076 Acc: 0.8264

Best val Acc: 0.831306

---Testing---
Test accuracy: 0.975270
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 89 %
Accuracy of   BET : 99 %
Accuracy of Bigeye tuna : 92 %
Accuracy of Blackfin tuna : 99 %
Accuracy of Bullet tuna : 100 %
Accuracy of Carcharhiniformes : 97 %
Accuracy of Conger myriaster : 100 %
Accuracy of   DOL : 100 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 98 %
Accuracy of Frigate tuna : 82 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexagrammos agrammus : 97 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 98 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 98 %
Accuracy of Larimichthys polyactis : 99 %
Accuracy of Lateolabrax japonicus : 98 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 96 %
Accuracy of Mackerel tuna : 92 %
Accuracy of Miichthys miiuy : 98 %
Accuracy of Mugil cephalus : 100 %
Accuracy of Myliobatiformes : 93 %
Accuracy of   NoF : 98 %
Accuracy of Oncorhynchus keta : 97 %
Accuracy of Oncorhynchus masou : 96 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pacific bluefin tuna : 88 %
Accuracy of Paralichthys olivaceus : 96 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 98 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 98 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 99 %
Accuracy of Skipjack tuna : 96 %
Accuracy of Slender tuna : 92 %
Accuracy of Southern bluefin tuna : 84 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 94 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 90 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 96 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 98 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 44 %
Accuracy of   ray : 80 %
Accuracy of rough : 92 %
Accuracy of shark : 98 %
mean: 0.9563000289709446, std: 0.0799545107815518
--------------------

run info[val: 0.2, epoch: 58, randcrop: True, decay: 9]

---Training last layer.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0237 Acc: 0.4231
val Loss: 0.0169 Acc: 0.5572

Epoch 1/57
----------
train Loss: 0.0147 Acc: 0.5993
val Loss: 0.0146 Acc: 0.6010

Epoch 2/57
----------
train Loss: 0.0124 Acc: 0.6536
val Loss: 0.0128 Acc: 0.6504

Epoch 3/57
----------
train Loss: 0.0111 Acc: 0.6860
val Loss: 0.0125 Acc: 0.6460

Epoch 4/57
----------
train Loss: 0.0102 Acc: 0.7052
val Loss: 0.0119 Acc: 0.6715

Epoch 5/57
----------
train Loss: 0.0096 Acc: 0.7236
val Loss: 0.0116 Acc: 0.6695

Epoch 6/57
----------
train Loss: 0.0092 Acc: 0.7278
val Loss: 0.0112 Acc: 0.6857

Epoch 7/57
----------
train Loss: 0.0088 Acc: 0.7401
val Loss: 0.0111 Acc: 0.6951

Epoch 8/57
----------
train Loss: 0.0084 Acc: 0.7565
val Loss: 0.0112 Acc: 0.6878

Epoch 9/57
----------
LR is set to 0.001
train Loss: 0.0079 Acc: 0.7708
val Loss: 0.0105 Acc: 0.7052

Epoch 10/57
----------
train Loss: 0.0078 Acc: 0.7761
val Loss: 0.0106 Acc: 0.7080

Epoch 11/57
----------
train Loss: 0.0077 Acc: 0.7806
val Loss: 0.0106 Acc: 0.7076

Epoch 12/57
----------
train Loss: 0.0076 Acc: 0.7895
val Loss: 0.0105 Acc: 0.7076

Epoch 13/57
----------
train Loss: 0.0077 Acc: 0.7801
val Loss: 0.0105 Acc: 0.7072

Epoch 14/57
----------
train Loss: 0.0076 Acc: 0.7806
val Loss: 0.0105 Acc: 0.7060

Epoch 15/57
----------
train Loss: 0.0076 Acc: 0.7823
val Loss: 0.0105 Acc: 0.7101

Epoch 16/57
----------
train Loss: 0.0076 Acc: 0.7747
val Loss: 0.0105 Acc: 0.7060

Epoch 17/57
----------
train Loss: 0.0076 Acc: 0.7809
val Loss: 0.0105 Acc: 0.7064

Epoch 18/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0076 Acc: 0.7860
val Loss: 0.0105 Acc: 0.7105

Epoch 19/57
----------
train Loss: 0.0075 Acc: 0.7818
val Loss: 0.0105 Acc: 0.7056

Epoch 20/57
----------
train Loss: 0.0075 Acc: 0.7871
val Loss: 0.0105 Acc: 0.7072

Epoch 21/57
----------
train Loss: 0.0076 Acc: 0.7802
val Loss: 0.0105 Acc: 0.7072

Epoch 22/57
----------
train Loss: 0.0074 Acc: 0.7906
val Loss: 0.0105 Acc: 0.7064

Epoch 23/57
----------
train Loss: 0.0075 Acc: 0.7876
val Loss: 0.0105 Acc: 0.7084

Epoch 24/57
----------
train Loss: 0.0075 Acc: 0.7860
val Loss: 0.0105 Acc: 0.7125

Epoch 25/57
----------
train Loss: 0.0075 Acc: 0.7895
val Loss: 0.0104 Acc: 0.7072

Epoch 26/57
----------
train Loss: 0.0075 Acc: 0.7842
val Loss: 0.0105 Acc: 0.7097

Epoch 27/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0075 Acc: 0.7889
val Loss: 0.0104 Acc: 0.7092

Epoch 28/57
----------
train Loss: 0.0075 Acc: 0.7870
val Loss: 0.0105 Acc: 0.7080

Epoch 29/57
----------
train Loss: 0.0075 Acc: 0.7876
val Loss: 0.0104 Acc: 0.7072

Epoch 30/57
----------
train Loss: 0.0075 Acc: 0.7866
val Loss: 0.0104 Acc: 0.7101

Epoch 31/57
----------
train Loss: 0.0075 Acc: 0.7850
val Loss: 0.0104 Acc: 0.7101

Epoch 32/57
----------
train Loss: 0.0075 Acc: 0.7858
val Loss: 0.0105 Acc: 0.7088

Epoch 33/57
----------
train Loss: 0.0075 Acc: 0.7855
val Loss: 0.0104 Acc: 0.7088

Epoch 34/57
----------
train Loss: 0.0075 Acc: 0.7846
val Loss: 0.0104 Acc: 0.7084

Epoch 35/57
----------
train Loss: 0.0075 Acc: 0.7831
val Loss: 0.0105 Acc: 0.7092

Epoch 36/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0075 Acc: 0.7869
val Loss: 0.0105 Acc: 0.7084

Epoch 37/57
----------
train Loss: 0.0075 Acc: 0.7867
val Loss: 0.0104 Acc: 0.7084

Epoch 38/57
----------
train Loss: 0.0074 Acc: 0.7892
val Loss: 0.0104 Acc: 0.7105

Epoch 39/57
----------
train Loss: 0.0075 Acc: 0.7838
val Loss: 0.0105 Acc: 0.7064

Epoch 40/57
----------
train Loss: 0.0075 Acc: 0.7876
val Loss: 0.0105 Acc: 0.7076

Epoch 41/57
----------
train Loss: 0.0075 Acc: 0.7865
val Loss: 0.0104 Acc: 0.7092

Epoch 42/57
----------
train Loss: 0.0075 Acc: 0.7891
val Loss: 0.0104 Acc: 0.7080

Epoch 43/57
----------
train Loss: 0.0076 Acc: 0.7809
val Loss: 0.0105 Acc: 0.7109

Epoch 44/57
----------
train Loss: 0.0074 Acc: 0.7866
val Loss: 0.0104 Acc: 0.7088

Epoch 45/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0075 Acc: 0.7866
val Loss: 0.0104 Acc: 0.7101

Epoch 46/57
----------
train Loss: 0.0074 Acc: 0.7887
val Loss: 0.0104 Acc: 0.7068

Epoch 47/57
----------
train Loss: 0.0074 Acc: 0.7873
val Loss: 0.0105 Acc: 0.7076

Epoch 48/57
----------
train Loss: 0.0076 Acc: 0.7787
val Loss: 0.0104 Acc: 0.7076

Epoch 49/57
----------
train Loss: 0.0075 Acc: 0.7874
val Loss: 0.0105 Acc: 0.7076

Epoch 50/57
----------
train Loss: 0.0075 Acc: 0.7853
val Loss: 0.0105 Acc: 0.7125

Epoch 51/57
----------
train Loss: 0.0075 Acc: 0.7872
val Loss: 0.0105 Acc: 0.7092

Epoch 52/57
----------
train Loss: 0.0075 Acc: 0.7820
val Loss: 0.0105 Acc: 0.7084

Epoch 53/57
----------
train Loss: 0.0074 Acc: 0.7885
val Loss: 0.0104 Acc: 0.7092

Epoch 54/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0075 Acc: 0.7821
val Loss: 0.0104 Acc: 0.7088

Epoch 55/57
----------
train Loss: 0.0075 Acc: 0.7845
val Loss: 0.0105 Acc: 0.7092

Epoch 56/57
----------
train Loss: 0.0075 Acc: 0.7885
val Loss: 0.0105 Acc: 0.7113

Epoch 57/57
----------
train Loss: 0.0075 Acc: 0.7900
val Loss: 0.0105 Acc: 0.7097

Best val Acc: 0.712490

---Fine tuning.---
Epoch 0/57
----------
LR is set to 0.01
train Loss: 0.0091 Acc: 0.7192
val Loss: 0.0105 Acc: 0.7170

Epoch 1/57
----------
train Loss: 0.0053 Acc: 0.8342
val Loss: 0.0094 Acc: 0.7405

Epoch 2/57
----------
train Loss: 0.0032 Acc: 0.9014
val Loss: 0.0083 Acc: 0.7847

Epoch 3/57
----------
train Loss: 0.0022 Acc: 0.9370
val Loss: 0.0083 Acc: 0.7871

Epoch 4/57
----------
train Loss: 0.0017 Acc: 0.9528
val Loss: 0.0079 Acc: 0.8021

Epoch 5/57
----------
train Loss: 0.0012 Acc: 0.9626
val Loss: 0.0075 Acc: 0.8017

Epoch 6/57
----------
train Loss: 0.0010 Acc: 0.9715
val Loss: 0.0079 Acc: 0.8045

Epoch 7/57
----------
train Loss: 0.0008 Acc: 0.9761
val Loss: 0.0077 Acc: 0.8143

Epoch 8/57
----------
train Loss: 0.0007 Acc: 0.9814
val Loss: 0.0077 Acc: 0.8191

Epoch 9/57
----------
LR is set to 0.001
train Loss: 0.0005 Acc: 0.9851
val Loss: 0.0074 Acc: 0.8260

Epoch 10/57
----------
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0074 Acc: 0.8281

Epoch 11/57
----------
train Loss: 0.0004 Acc: 0.9882
val Loss: 0.0074 Acc: 0.8293

Epoch 12/57
----------
train Loss: 0.0004 Acc: 0.9884
val Loss: 0.0073 Acc: 0.8289

Epoch 13/57
----------
train Loss: 0.0004 Acc: 0.9898
val Loss: 0.0073 Acc: 0.8301

Epoch 14/57
----------
train Loss: 0.0004 Acc: 0.9892
val Loss: 0.0074 Acc: 0.8345

Epoch 15/57
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8313

Epoch 16/57
----------
train Loss: 0.0003 Acc: 0.9891
val Loss: 0.0074 Acc: 0.8313

Epoch 17/57
----------
train Loss: 0.0004 Acc: 0.9878
val Loss: 0.0074 Acc: 0.8337

Epoch 18/57
----------
LR is set to 0.00010000000000000002
train Loss: 0.0003 Acc: 0.9900
val Loss: 0.0074 Acc: 0.8321

Epoch 19/57
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0074 Acc: 0.8325

Epoch 20/57
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0073 Acc: 0.8333

Epoch 21/57
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0074 Acc: 0.8325

Epoch 22/57
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0074 Acc: 0.8325

Epoch 23/57
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0074 Acc: 0.8309

Epoch 24/57
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0073 Acc: 0.8313

Epoch 25/57
----------
train Loss: 0.0003 Acc: 0.9900
val Loss: 0.0075 Acc: 0.8325

Epoch 26/57
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0074 Acc: 0.8321

Epoch 27/57
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0074 Acc: 0.8317

Epoch 28/57
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0074 Acc: 0.8317

Epoch 29/57
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0074 Acc: 0.8325

Epoch 30/57
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0074 Acc: 0.8317

Epoch 31/57
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0074 Acc: 0.8313

Epoch 32/57
----------
train Loss: 0.0003 Acc: 0.9897
val Loss: 0.0074 Acc: 0.8305

Epoch 33/57
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0074 Acc: 0.8313

Epoch 34/57
----------
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0073 Acc: 0.8309

Epoch 35/57
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0074 Acc: 0.8321

Epoch 36/57
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0073 Acc: 0.8325

Epoch 37/57
----------
train Loss: 0.0003 Acc: 0.9896
val Loss: 0.0074 Acc: 0.8313

Epoch 38/57
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0074 Acc: 0.8329

Epoch 39/57
----------
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0074 Acc: 0.8325

Epoch 40/57
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8325

Epoch 41/57
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0073 Acc: 0.8329

Epoch 42/57
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0074 Acc: 0.8313

Epoch 43/57
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0073 Acc: 0.8301

Epoch 44/57
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0074 Acc: 0.8317

Epoch 45/57
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8325

Epoch 46/57
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0074 Acc: 0.8297

Epoch 47/57
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0075 Acc: 0.8313

Epoch 48/57
----------
train Loss: 0.0003 Acc: 0.9898
val Loss: 0.0074 Acc: 0.8325

Epoch 49/57
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0074 Acc: 0.8309

Epoch 50/57
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0074 Acc: 0.8321

Epoch 51/57
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0074 Acc: 0.8317

Epoch 52/57
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0073 Acc: 0.8329

Epoch 53/57
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0074 Acc: 0.8309

Epoch 54/57
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0074 Acc: 0.8337

Epoch 55/57
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0074 Acc: 0.8313

Epoch 56/57
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0075 Acc: 0.8325

Epoch 57/57
----------
train Loss: 0.0003 Acc: 0.9900
val Loss: 0.0073 Acc: 0.8313

Best val Acc: 0.834550

---Testing---
Test accuracy: 0.959621
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 88 %
Accuracy of Atlantic bluefin tuna : 73 %
Accuracy of   BET : 98 %
Accuracy of Bigeye tuna : 89 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 86 %
Accuracy of Carcharhiniformes : 94 %
Accuracy of Conger myriaster : 97 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 95 %
Accuracy of Frigate tuna : 79 %
Accuracy of Heterodontiformes : 98 %
Accuracy of Hexagrammos agrammus : 92 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Konosirus punctatus : 97 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 96 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 97 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 95 %
Accuracy of Mackerel tuna : 80 %
Accuracy of Miichthys miiuy : 94 %
Accuracy of Mugil cephalus : 97 %
Accuracy of Myliobatiformes : 90 %
Accuracy of   NoF : 97 %
Accuracy of Oncorhynchus keta : 92 %
Accuracy of Oncorhynchus masou : 94 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 96 %
Accuracy of Pacific bluefin tuna : 80 %
Accuracy of Paralichthys olivaceus : 95 %
Accuracy of Pleuronectidae : 98 %
Accuracy of Pristiformes : 96 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 97 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 75 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 92 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 100 %
Accuracy of Thunnus orientalis : 87 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 94 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 95 %
Accuracy of holocephalan : 93 %
Accuracy of mullet : 50 %
Accuracy of   ray : 74 %
Accuracy of rough : 87 %
Accuracy of shark : 97 %
mean: 0.929845341025519, std: 0.08700553619145131
--------------------

run info[val: 0.3, epoch: 47, randcrop: False, decay: 8]

---Training last layer.---
Epoch 0/46
----------
LR is set to 0.01
train Loss: 0.0243 Acc: 0.4076
val Loss: 0.0170 Acc: 0.5712

Epoch 1/46
----------
train Loss: 0.0145 Acc: 0.6161
val Loss: 0.0140 Acc: 0.6229

Epoch 2/46
----------
train Loss: 0.0118 Acc: 0.6814
val Loss: 0.0126 Acc: 0.6531

Epoch 3/46
----------
train Loss: 0.0105 Acc: 0.7086
val Loss: 0.0121 Acc: 0.6669

Epoch 4/46
----------
train Loss: 0.0097 Acc: 0.7239
val Loss: 0.0118 Acc: 0.6659

Epoch 5/46
----------
train Loss: 0.0090 Acc: 0.7451
val Loss: 0.0116 Acc: 0.6705

Epoch 6/46
----------
train Loss: 0.0085 Acc: 0.7615
val Loss: 0.0111 Acc: 0.6880

Epoch 7/46
----------
train Loss: 0.0079 Acc: 0.7717
val Loss: 0.0108 Acc: 0.6948

Epoch 8/46
----------
LR is set to 0.001
train Loss: 0.0072 Acc: 0.8010
val Loss: 0.0105 Acc: 0.7086

Epoch 9/46
----------
train Loss: 0.0071 Acc: 0.8104
val Loss: 0.0104 Acc: 0.7132

Epoch 10/46
----------
train Loss: 0.0071 Acc: 0.8128
val Loss: 0.0104 Acc: 0.7097

Epoch 11/46
----------
train Loss: 0.0070 Acc: 0.8129
val Loss: 0.0103 Acc: 0.7126

Epoch 12/46
----------
train Loss: 0.0071 Acc: 0.8072
val Loss: 0.0104 Acc: 0.7132

Epoch 13/46
----------
train Loss: 0.0069 Acc: 0.8167
val Loss: 0.0104 Acc: 0.7091

Epoch 14/46
----------
train Loss: 0.0070 Acc: 0.8097
val Loss: 0.0103 Acc: 0.7107

Epoch 15/46
----------
train Loss: 0.0069 Acc: 0.8156
val Loss: 0.0103 Acc: 0.7118

Epoch 16/46
----------
LR is set to 0.00010000000000000002
train Loss: 0.0069 Acc: 0.8141
val Loss: 0.0103 Acc: 0.7129

Epoch 17/46
----------
train Loss: 0.0069 Acc: 0.8156
val Loss: 0.0103 Acc: 0.7142

Epoch 18/46
----------
train Loss: 0.0068 Acc: 0.8176
val Loss: 0.0103 Acc: 0.7121

Epoch 19/46
----------
train Loss: 0.0069 Acc: 0.8186
val Loss: 0.0103 Acc: 0.7129

Epoch 20/46
----------
train Loss: 0.0069 Acc: 0.8149
val Loss: 0.0103 Acc: 0.7126

Epoch 21/46
----------
train Loss: 0.0069 Acc: 0.8148
val Loss: 0.0103 Acc: 0.7156

Epoch 22/46
----------
train Loss: 0.0069 Acc: 0.8162
val Loss: 0.0103 Acc: 0.7118

Epoch 23/46
----------
train Loss: 0.0068 Acc: 0.8193
val Loss: 0.0103 Acc: 0.7140

Epoch 24/46
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0068 Acc: 0.8198
val Loss: 0.0103 Acc: 0.7129

Epoch 25/46
----------
train Loss: 0.0068 Acc: 0.8209
val Loss: 0.0103 Acc: 0.7121

Epoch 26/46
----------
train Loss: 0.0068 Acc: 0.8179
val Loss: 0.0103 Acc: 0.7129

Epoch 27/46
----------
train Loss: 0.0069 Acc: 0.8146
val Loss: 0.0103 Acc: 0.7126

Epoch 28/46
----------
train Loss: 0.0068 Acc: 0.8208
val Loss: 0.0103 Acc: 0.7142

Epoch 29/46
----------
train Loss: 0.0068 Acc: 0.8190
val Loss: 0.0103 Acc: 0.7132

Epoch 30/46
----------
train Loss: 0.0069 Acc: 0.8125
val Loss: 0.0103 Acc: 0.7142

Epoch 31/46
----------
train Loss: 0.0068 Acc: 0.8170
val Loss: 0.0103 Acc: 0.7110

Epoch 32/46
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0068 Acc: 0.8222
val Loss: 0.0103 Acc: 0.7134

Epoch 33/46
----------
train Loss: 0.0068 Acc: 0.8183
val Loss: 0.0103 Acc: 0.7164

Epoch 34/46
----------
train Loss: 0.0068 Acc: 0.8215
val Loss: 0.0103 Acc: 0.7140

Epoch 35/46
----------
train Loss: 0.0068 Acc: 0.8198
val Loss: 0.0103 Acc: 0.7107

Epoch 36/46
----------
train Loss: 0.0068 Acc: 0.8174
val Loss: 0.0103 Acc: 0.7115

Epoch 37/46
----------
train Loss: 0.0069 Acc: 0.8192
val Loss: 0.0103 Acc: 0.7091

Epoch 38/46
----------
train Loss: 0.0069 Acc: 0.8160
val Loss: 0.0103 Acc: 0.7132

Epoch 39/46
----------
train Loss: 0.0068 Acc: 0.8180
val Loss: 0.0103 Acc: 0.7121

Epoch 40/46
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0068 Acc: 0.8182
val Loss: 0.0103 Acc: 0.7124

Epoch 41/46
----------
train Loss: 0.0068 Acc: 0.8180
val Loss: 0.0103 Acc: 0.7118

Epoch 42/46
----------
train Loss: 0.0068 Acc: 0.8174
val Loss: 0.0103 Acc: 0.7145

Epoch 43/46
----------
train Loss: 0.0068 Acc: 0.8194
val Loss: 0.0103 Acc: 0.7124

Epoch 44/46
----------
train Loss: 0.0068 Acc: 0.8154
val Loss: 0.0103 Acc: 0.7134

Epoch 45/46
----------
train Loss: 0.0069 Acc: 0.8140
val Loss: 0.0103 Acc: 0.7121

Epoch 46/46
----------
train Loss: 0.0068 Acc: 0.8176
val Loss: 0.0103 Acc: 0.7124

Best val Acc: 0.716410

---Fine tuning.---
Epoch 0/46
----------
LR is set to 0.01
train Loss: 0.0084 Acc: 0.7448
val Loss: 0.0110 Acc: 0.6929

Epoch 1/46
----------
train Loss: 0.0040 Acc: 0.8819
val Loss: 0.0091 Acc: 0.7489

Epoch 2/46
----------
train Loss: 0.0021 Acc: 0.9390
val Loss: 0.0076 Acc: 0.7935

Epoch 3/46
----------
train Loss: 0.0012 Acc: 0.9678
val Loss: 0.0070 Acc: 0.8021

Epoch 4/46
----------
train Loss: 0.0009 Acc: 0.9779
val Loss: 0.0073 Acc: 0.8081

Epoch 5/46
----------
train Loss: 0.0006 Acc: 0.9836
val Loss: 0.0071 Acc: 0.8127

Epoch 6/46
----------
train Loss: 0.0005 Acc: 0.9856
val Loss: 0.0070 Acc: 0.8116

Epoch 7/46
----------
train Loss: 0.0004 Acc: 0.9870
val Loss: 0.0069 Acc: 0.8189

Epoch 8/46
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9899
val Loss: 0.0068 Acc: 0.8205

Epoch 9/46
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0068 Acc: 0.8229

Epoch 10/46
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0068 Acc: 0.8216

Epoch 11/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8213

Epoch 12/46
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0068 Acc: 0.8205

Epoch 13/46
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8221

Epoch 14/46
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0068 Acc: 0.8210

Epoch 15/46
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8229

Epoch 16/46
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0068 Acc: 0.8191

Epoch 17/46
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0068 Acc: 0.8232

Epoch 18/46
----------
train Loss: 0.0002 Acc: 0.9934
val Loss: 0.0068 Acc: 0.8243

Epoch 19/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8229

Epoch 20/46
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8235

Epoch 21/46
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0068 Acc: 0.8216

Epoch 22/46
----------
train Loss: 0.0002 Acc: 0.9933
val Loss: 0.0069 Acc: 0.8232

Epoch 23/46
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0068 Acc: 0.8237

Epoch 24/46
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8210

Epoch 25/46
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0069 Acc: 0.8221

Epoch 26/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8221

Epoch 27/46
----------
train Loss: 0.0002 Acc: 0.9926
val Loss: 0.0068 Acc: 0.8218

Epoch 28/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0069 Acc: 0.8229

Epoch 29/46
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8224

Epoch 30/46
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8229

Epoch 31/46
----------
train Loss: 0.0002 Acc: 0.9932
val Loss: 0.0068 Acc: 0.8221

Epoch 32/46
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0068 Acc: 0.8224

Epoch 33/46
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0068 Acc: 0.8235

Epoch 34/46
----------
train Loss: 0.0002 Acc: 0.9937
val Loss: 0.0068 Acc: 0.8229

Epoch 35/46
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0068 Acc: 0.8237

Epoch 36/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8237

Epoch 37/46
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0069 Acc: 0.8224

Epoch 38/46
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0068 Acc: 0.8245

Epoch 39/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8229

Epoch 40/46
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9935
val Loss: 0.0068 Acc: 0.8213

Epoch 41/46
----------
train Loss: 0.0002 Acc: 0.9931
val Loss: 0.0068 Acc: 0.8227

Epoch 42/46
----------
train Loss: 0.0002 Acc: 0.9933
val Loss: 0.0068 Acc: 0.8227

Epoch 43/46
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0069 Acc: 0.8240

Epoch 44/46
----------
train Loss: 0.0002 Acc: 0.9929
val Loss: 0.0068 Acc: 0.8224

Epoch 45/46
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0069 Acc: 0.8202

Epoch 46/46
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0068 Acc: 0.8245

Best val Acc: 0.824547

---Testing---
Test accuracy: 0.942269
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 84 %
Accuracy of Atlantic bluefin tuna : 75 %
Accuracy of   BET : 97 %
Accuracy of Bigeye tuna : 86 %
Accuracy of Blackfin tuna : 95 %
Accuracy of Bullet tuna : 86 %
Accuracy of Carcharhiniformes : 91 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 98 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 95 %
Accuracy of Frigate tuna : 62 %
Accuracy of Heterodontiformes : 93 %
Accuracy of Hexagrammos agrammus : 84 %
Accuracy of Hexanchiformes : 98 %
Accuracy of Konosirus punctatus : 96 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 92 %
Accuracy of Larimichthys polyactis : 97 %
Accuracy of Lateolabrax japonicus : 94 %
Accuracy of Little tunny : 82 %
Accuracy of Longtail tuna : 88 %
Accuracy of Mackerel tuna : 74 %
Accuracy of Miichthys miiuy : 92 %
Accuracy of Mugil cephalus : 93 %
Accuracy of Myliobatiformes : 86 %
Accuracy of   NoF : 96 %
Accuracy of Oncorhynchus keta : 88 %
Accuracy of Oncorhynchus masou : 92 %
Accuracy of Oplegnathus fasciatus : 98 %
Accuracy of Orectolobiformes : 94 %
Accuracy of Pacific bluefin tuna : 75 %
Accuracy of Paralichthys olivaceus : 90 %
Accuracy of Pleuronectidae : 93 %
Accuracy of Pristiformes : 95 %
Accuracy of Rajiformes : 92 %
Accuracy of Rhinobatiformes : 95 %
Accuracy of SHARK on boat : 99 %
Accuracy of Scomber japonicus : 96 %
Accuracy of Sebastes inermis : 98 %
Accuracy of Seriola quinqueradiata : 93 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 78 %
Accuracy of Southern bluefin tuna : 68 %
Accuracy of Squaliformes : 94 %
Accuracy of Squatiniformes : 90 %
Accuracy of Stephanolepis cirrhifer : 97 %
Accuracy of Tetraodon or Diodon : 98 %
Accuracy of Thunnus orientalis : 78 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 91 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 95 %
Accuracy of holocephalan : 91 %
Accuracy of mullet : 47 %
Accuracy of   ray : 70 %
Accuracy of rough : 80 %
Accuracy of shark : 97 %
mean: 0.9007327732596209, std: 0.10339654490784025

Model saved in "./weights/all_in_one_[0.98]_mean[0.96]_std[0.08].save".
--------------------

run info[val: 0.1, epoch: 50, randcrop: False, decay: 9]

---Training last layer.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0224 Acc: 0.4415
val Loss: 0.0168 Acc: 0.5669

Epoch 1/49
----------
train Loss: 0.0135 Acc: 0.6344
val Loss: 0.0138 Acc: 0.6464

Epoch 2/49
----------
train Loss: 0.0111 Acc: 0.6936
val Loss: 0.0127 Acc: 0.6748

Epoch 3/49
----------
train Loss: 0.0101 Acc: 0.7175
val Loss: 0.0123 Acc: 0.6788

Epoch 4/49
----------
train Loss: 0.0091 Acc: 0.7406
val Loss: 0.0118 Acc: 0.6902

Epoch 5/49
----------
train Loss: 0.0086 Acc: 0.7586
val Loss: 0.0118 Acc: 0.6837

Epoch 6/49
----------
train Loss: 0.0082 Acc: 0.7650
val Loss: 0.0116 Acc: 0.7072

Epoch 7/49
----------
train Loss: 0.0078 Acc: 0.7792
val Loss: 0.0110 Acc: 0.7137

Epoch 8/49
----------
train Loss: 0.0075 Acc: 0.7842
val Loss: 0.0111 Acc: 0.7024

Epoch 9/49
----------
LR is set to 0.001
train Loss: 0.0068 Acc: 0.8092
val Loss: 0.0105 Acc: 0.7251

Epoch 10/49
----------
train Loss: 0.0067 Acc: 0.8129
val Loss: 0.0110 Acc: 0.7324

Epoch 11/49
----------
train Loss: 0.0067 Acc: 0.8198
val Loss: 0.0106 Acc: 0.7283

Epoch 12/49
----------
train Loss: 0.0066 Acc: 0.8155
val Loss: 0.0105 Acc: 0.7234

Epoch 13/49
----------
train Loss: 0.0066 Acc: 0.8153
val Loss: 0.0107 Acc: 0.7210

Epoch 14/49
----------
train Loss: 0.0065 Acc: 0.8229
val Loss: 0.0108 Acc: 0.7267

Epoch 15/49
----------
train Loss: 0.0066 Acc: 0.8185
val Loss: 0.0107 Acc: 0.7332

Epoch 16/49
----------
train Loss: 0.0066 Acc: 0.8168
val Loss: 0.0105 Acc: 0.7210

Epoch 17/49
----------
train Loss: 0.0065 Acc: 0.8201
val Loss: 0.0104 Acc: 0.7234

Epoch 18/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0064 Acc: 0.8245
val Loss: 0.0104 Acc: 0.7259

Epoch 19/49
----------
train Loss: 0.0064 Acc: 0.8245
val Loss: 0.0106 Acc: 0.7259

Epoch 20/49
----------
train Loss: 0.0064 Acc: 0.8242
val Loss: 0.0106 Acc: 0.7291

Epoch 21/49
----------
train Loss: 0.0064 Acc: 0.8263
val Loss: 0.0107 Acc: 0.7275

Epoch 22/49
----------
train Loss: 0.0064 Acc: 0.8238
val Loss: 0.0105 Acc: 0.7315

Epoch 23/49
----------
train Loss: 0.0065 Acc: 0.8217
val Loss: 0.0109 Acc: 0.7218

Epoch 24/49
----------
train Loss: 0.0064 Acc: 0.8216
val Loss: 0.0106 Acc: 0.7315

Epoch 25/49
----------
train Loss: 0.0064 Acc: 0.8212
val Loss: 0.0106 Acc: 0.7242

Epoch 26/49
----------
train Loss: 0.0064 Acc: 0.8268
val Loss: 0.0106 Acc: 0.7332

Epoch 27/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0065 Acc: 0.8215
val Loss: 0.0105 Acc: 0.7299

Epoch 28/49
----------
train Loss: 0.0064 Acc: 0.8247
val Loss: 0.0108 Acc: 0.7315

Epoch 29/49
----------
train Loss: 0.0064 Acc: 0.8254
val Loss: 0.0106 Acc: 0.7299

Epoch 30/49
----------
train Loss: 0.0064 Acc: 0.8240
val Loss: 0.0106 Acc: 0.7291

Epoch 31/49
----------
train Loss: 0.0064 Acc: 0.8235
val Loss: 0.0106 Acc: 0.7283

Epoch 32/49
----------
train Loss: 0.0065 Acc: 0.8221
val Loss: 0.0104 Acc: 0.7299

Epoch 33/49
----------
train Loss: 0.0064 Acc: 0.8227
val Loss: 0.0107 Acc: 0.7283

Epoch 34/49
----------
train Loss: 0.0065 Acc: 0.8229
val Loss: 0.0106 Acc: 0.7251

Epoch 35/49
----------
train Loss: 0.0064 Acc: 0.8267
val Loss: 0.0105 Acc: 0.7242

Epoch 36/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0065 Acc: 0.8232
val Loss: 0.0108 Acc: 0.7259

Epoch 37/49
----------
train Loss: 0.0064 Acc: 0.8231
val Loss: 0.0105 Acc: 0.7307

Epoch 38/49
----------
train Loss: 0.0064 Acc: 0.8254
val Loss: 0.0105 Acc: 0.7299

Epoch 39/49
----------
train Loss: 0.0064 Acc: 0.8250
val Loss: 0.0106 Acc: 0.7283

Epoch 40/49
----------
train Loss: 0.0065 Acc: 0.8232
val Loss: 0.0105 Acc: 0.7234

Epoch 41/49
----------
train Loss: 0.0065 Acc: 0.8237
val Loss: 0.0107 Acc: 0.7251

Epoch 42/49
----------
train Loss: 0.0064 Acc: 0.8232
val Loss: 0.0106 Acc: 0.7283

Epoch 43/49
----------
train Loss: 0.0064 Acc: 0.8248
val Loss: 0.0105 Acc: 0.7267

Epoch 44/49
----------
train Loss: 0.0064 Acc: 0.8257
val Loss: 0.0108 Acc: 0.7267

Epoch 45/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0064 Acc: 0.8216
val Loss: 0.0105 Acc: 0.7283

Epoch 46/49
----------
train Loss: 0.0064 Acc: 0.8241
val Loss: 0.0105 Acc: 0.7251

Epoch 47/49
----------
train Loss: 0.0063 Acc: 0.8271
val Loss: 0.0106 Acc: 0.7324

Epoch 48/49
----------
train Loss: 0.0064 Acc: 0.8254
val Loss: 0.0104 Acc: 0.7267

Epoch 49/49
----------
train Loss: 0.0064 Acc: 0.8231
val Loss: 0.0106 Acc: 0.7299

Best val Acc: 0.733171

---Fine tuning.---
Epoch 0/49
----------
LR is set to 0.01
train Loss: 0.0083 Acc: 0.7477
val Loss: 0.0125 Acc: 0.6926

Epoch 1/49
----------
train Loss: 0.0039 Acc: 0.8767
val Loss: 0.0108 Acc: 0.7356

Epoch 2/49
----------
train Loss: 0.0021 Acc: 0.9398
val Loss: 0.0080 Acc: 0.7835

Epoch 3/49
----------
train Loss: 0.0012 Acc: 0.9654
val Loss: 0.0085 Acc: 0.8094

Epoch 4/49
----------
train Loss: 0.0008 Acc: 0.9784
val Loss: 0.0076 Acc: 0.8200

Epoch 5/49
----------
train Loss: 0.0006 Acc: 0.9843
val Loss: 0.0078 Acc: 0.8175

Epoch 6/49
----------
train Loss: 0.0005 Acc: 0.9843
val Loss: 0.0079 Acc: 0.8264

Epoch 7/49
----------
train Loss: 0.0004 Acc: 0.9872
val Loss: 0.0077 Acc: 0.8248

Epoch 8/49
----------
train Loss: 0.0004 Acc: 0.9869
val Loss: 0.0078 Acc: 0.8175

Epoch 9/49
----------
LR is set to 0.001
train Loss: 0.0003 Acc: 0.9895
val Loss: 0.0079 Acc: 0.8224

Epoch 10/49
----------
train Loss: 0.0002 Acc: 0.9904
val Loss: 0.0076 Acc: 0.8240

Epoch 11/49
----------
train Loss: 0.0002 Acc: 0.9899
val Loss: 0.0076 Acc: 0.8281

Epoch 12/49
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0078 Acc: 0.8240

Epoch 13/49
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0076 Acc: 0.8248

Epoch 14/49
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0080 Acc: 0.8240

Epoch 15/49
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0076 Acc: 0.8289

Epoch 16/49
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0076 Acc: 0.8256

Epoch 17/49
----------
train Loss: 0.0002 Acc: 0.9903
val Loss: 0.0075 Acc: 0.8289

Epoch 18/49
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0077 Acc: 0.8289

Epoch 19/49
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0078 Acc: 0.8264

Epoch 20/49
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0078 Acc: 0.8264

Epoch 21/49
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0079 Acc: 0.8289

Epoch 22/49
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0075 Acc: 0.8313

Epoch 23/49
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0080 Acc: 0.8248

Epoch 24/49
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0075 Acc: 0.8289

Epoch 25/49
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0077 Acc: 0.8281

Epoch 26/49
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0075 Acc: 0.8289

Epoch 27/49
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0075 Acc: 0.8240

Epoch 28/49
----------
train Loss: 0.0002 Acc: 0.9928
val Loss: 0.0076 Acc: 0.8289

Epoch 29/49
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0080 Acc: 0.8273

Epoch 30/49
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0079 Acc: 0.8256

Epoch 31/49
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0079 Acc: 0.8264

Epoch 32/49
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0078 Acc: 0.8273

Epoch 33/49
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0080 Acc: 0.8240

Epoch 34/49
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0076 Acc: 0.8281

Epoch 35/49
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0082 Acc: 0.8273

Epoch 36/49
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0076 Acc: 0.8248

Epoch 37/49
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0077 Acc: 0.8248

Epoch 38/49
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0077 Acc: 0.8297

Epoch 39/49
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0078 Acc: 0.8256

Epoch 40/49
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0076 Acc: 0.8264

Epoch 41/49
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0076 Acc: 0.8273

Epoch 42/49
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0080 Acc: 0.8273

Epoch 43/49
----------
train Loss: 0.0002 Acc: 0.9927
val Loss: 0.0076 Acc: 0.8232

Epoch 44/49
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0076 Acc: 0.8240

Epoch 45/49
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0079 Acc: 0.8273

Epoch 46/49
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0079 Acc: 0.8248

Epoch 47/49
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0077 Acc: 0.8273

Epoch 48/49
----------
train Loss: 0.0002 Acc: 0.9905
val Loss: 0.0078 Acc: 0.8256

Epoch 49/49
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0078 Acc: 0.8273

Best val Acc: 0.831306

---Testing---
Test accuracy: 0.975999
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 93 %
Accuracy of Atlantic bluefin tuna : 87 %
Accuracy of   BET : 99 %
Accuracy of Bigeye tuna : 92 %
Accuracy of Blackfin tuna : 98 %
Accuracy of Bullet tuna : 100 %
Accuracy of Carcharhiniformes : 98 %
Accuracy of Conger myriaster : 99 %
Accuracy of   DOL : 100 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 98 %
Accuracy of Frigate tuna : 79 %
Accuracy of Heterodontiformes : 100 %
Accuracy of Hexagrammos agrammus : 97 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 99 %
Accuracy of   LAG : 100 %
Accuracy of Lamniformes : 100 %
Accuracy of Larimichthys polyactis : 99 %
Accuracy of Lateolabrax japonicus : 98 %
Accuracy of Little tunny : 91 %
Accuracy of Longtail tuna : 97 %
Accuracy of Mackerel tuna : 92 %
Accuracy of Miichthys miiuy : 97 %
Accuracy of Mugil cephalus : 99 %
Accuracy of Myliobatiformes : 92 %
Accuracy of   NoF : 99 %
Accuracy of Oncorhynchus keta : 97 %
Accuracy of Oncorhynchus masou : 96 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 97 %
Accuracy of Pacific bluefin tuna : 90 %
Accuracy of Paralichthys olivaceus : 97 %
Accuracy of Pleuronectidae : 97 %
Accuracy of Pristiformes : 98 %
Accuracy of Rajiformes : 97 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 99 %
Accuracy of Skipjack tuna : 95 %
Accuracy of Slender tuna : 92 %
Accuracy of Southern bluefin tuna : 88 %
Accuracy of Squaliformes : 100 %
Accuracy of Squatiniformes : 94 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 89 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 96 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 98 %
Accuracy of holocephalan : 96 %
Accuracy of mullet : 57 %
Accuracy of   ray : 75 %
Accuracy of rough : 93 %
Accuracy of shark : 99 %
mean: 0.9579019243831696, std: 0.06940145793509542
--------------------

run info[val: 0.2, epoch: 85, randcrop: True, decay: 11]

---Training last layer.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0237 Acc: 0.4202
val Loss: 0.0168 Acc: 0.5697

Epoch 1/84
----------
train Loss: 0.0145 Acc: 0.6016
val Loss: 0.0142 Acc: 0.6200

Epoch 2/84
----------
train Loss: 0.0123 Acc: 0.6491
val Loss: 0.0132 Acc: 0.6322

Epoch 3/84
----------
train Loss: 0.0110 Acc: 0.6846
val Loss: 0.0121 Acc: 0.6586

Epoch 4/84
----------
train Loss: 0.0104 Acc: 0.7017
val Loss: 0.0117 Acc: 0.6784

Epoch 5/84
----------
train Loss: 0.0097 Acc: 0.7162
val Loss: 0.0114 Acc: 0.6776

Epoch 6/84
----------
train Loss: 0.0092 Acc: 0.7346
val Loss: 0.0112 Acc: 0.6914

Epoch 7/84
----------
train Loss: 0.0088 Acc: 0.7421
val Loss: 0.0110 Acc: 0.6959

Epoch 8/84
----------
train Loss: 0.0086 Acc: 0.7441
val Loss: 0.0109 Acc: 0.6975

Epoch 9/84
----------
train Loss: 0.0082 Acc: 0.7603
val Loss: 0.0108 Acc: 0.6987

Epoch 10/84
----------
train Loss: 0.0081 Acc: 0.7603
val Loss: 0.0109 Acc: 0.6946

Epoch 11/84
----------
LR is set to 0.001
train Loss: 0.0075 Acc: 0.7827
val Loss: 0.0104 Acc: 0.7092

Epoch 12/84
----------
train Loss: 0.0074 Acc: 0.7853
val Loss: 0.0103 Acc: 0.7092

Epoch 13/84
----------
train Loss: 0.0073 Acc: 0.7889
val Loss: 0.0104 Acc: 0.7101

Epoch 14/84
----------
train Loss: 0.0074 Acc: 0.7855
val Loss: 0.0104 Acc: 0.7117

Epoch 15/84
----------
train Loss: 0.0073 Acc: 0.7879
val Loss: 0.0104 Acc: 0.7133

Epoch 16/84
----------
train Loss: 0.0072 Acc: 0.7915
val Loss: 0.0103 Acc: 0.7153

Epoch 17/84
----------
train Loss: 0.0073 Acc: 0.7910
val Loss: 0.0104 Acc: 0.7141

Epoch 18/84
----------
train Loss: 0.0073 Acc: 0.7886
val Loss: 0.0104 Acc: 0.7129

Epoch 19/84
----------
train Loss: 0.0072 Acc: 0.7927
val Loss: 0.0103 Acc: 0.7133

Epoch 20/84
----------
train Loss: 0.0071 Acc: 0.7913
val Loss: 0.0103 Acc: 0.7137

Epoch 21/84
----------
train Loss: 0.0072 Acc: 0.7895
val Loss: 0.0104 Acc: 0.7084

Epoch 22/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0071 Acc: 0.7948
val Loss: 0.0103 Acc: 0.7088

Epoch 23/84
----------
train Loss: 0.0071 Acc: 0.7955
val Loss: 0.0103 Acc: 0.7117

Epoch 24/84
----------
train Loss: 0.0071 Acc: 0.7937
val Loss: 0.0103 Acc: 0.7157

Epoch 25/84
----------
train Loss: 0.0071 Acc: 0.8044
val Loss: 0.0103 Acc: 0.7153

Epoch 26/84
----------
train Loss: 0.0071 Acc: 0.7954
val Loss: 0.0103 Acc: 0.7161

Epoch 27/84
----------
train Loss: 0.0071 Acc: 0.7956
val Loss: 0.0103 Acc: 0.7129

Epoch 28/84
----------
train Loss: 0.0071 Acc: 0.7915
val Loss: 0.0103 Acc: 0.7121

Epoch 29/84
----------
train Loss: 0.0072 Acc: 0.7950
val Loss: 0.0103 Acc: 0.7153

Epoch 30/84
----------
train Loss: 0.0071 Acc: 0.7963
val Loss: 0.0103 Acc: 0.7121

Epoch 31/84
----------
train Loss: 0.0070 Acc: 0.8002
val Loss: 0.0103 Acc: 0.7133

Epoch 32/84
----------
train Loss: 0.0071 Acc: 0.7957
val Loss: 0.0103 Acc: 0.7137

Epoch 33/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0070 Acc: 0.7945
val Loss: 0.0103 Acc: 0.7170

Epoch 34/84
----------
train Loss: 0.0070 Acc: 0.7999
val Loss: 0.0103 Acc: 0.7137

Epoch 35/84
----------
train Loss: 0.0071 Acc: 0.7983
val Loss: 0.0103 Acc: 0.7170

Epoch 36/84
----------
train Loss: 0.0070 Acc: 0.7967
val Loss: 0.0103 Acc: 0.7141

Epoch 37/84
----------
train Loss: 0.0071 Acc: 0.7986
val Loss: 0.0103 Acc: 0.7145

Epoch 38/84
----------
train Loss: 0.0070 Acc: 0.7945
val Loss: 0.0103 Acc: 0.7141

Epoch 39/84
----------
train Loss: 0.0071 Acc: 0.7924
val Loss: 0.0103 Acc: 0.7157

Epoch 40/84
----------
train Loss: 0.0071 Acc: 0.7899
val Loss: 0.0103 Acc: 0.7157

Epoch 41/84
----------
train Loss: 0.0071 Acc: 0.7979
val Loss: 0.0103 Acc: 0.7137

Epoch 42/84
----------
train Loss: 0.0071 Acc: 0.7931
val Loss: 0.0103 Acc: 0.7141

Epoch 43/84
----------
train Loss: 0.0069 Acc: 0.8033
val Loss: 0.0103 Acc: 0.7149

Epoch 44/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0070 Acc: 0.8010
val Loss: 0.0103 Acc: 0.7149

Epoch 45/84
----------
train Loss: 0.0071 Acc: 0.7963
val Loss: 0.0103 Acc: 0.7129

Epoch 46/84
----------
train Loss: 0.0070 Acc: 0.7988
val Loss: 0.0103 Acc: 0.7145

Epoch 47/84
----------
train Loss: 0.0070 Acc: 0.7982
val Loss: 0.0103 Acc: 0.7133

Epoch 48/84
----------
train Loss: 0.0071 Acc: 0.7935
val Loss: 0.0103 Acc: 0.7161

Epoch 49/84
----------
train Loss: 0.0071 Acc: 0.7968
val Loss: 0.0103 Acc: 0.7137

Epoch 50/84
----------
train Loss: 0.0071 Acc: 0.7962
val Loss: 0.0103 Acc: 0.7133

Epoch 51/84
----------
train Loss: 0.0070 Acc: 0.7989
val Loss: 0.0103 Acc: 0.7153

Epoch 52/84
----------
train Loss: 0.0071 Acc: 0.7924
val Loss: 0.0103 Acc: 0.7145

Epoch 53/84
----------
train Loss: 0.0070 Acc: 0.7975
val Loss: 0.0103 Acc: 0.7145

Epoch 54/84
----------
train Loss: 0.0070 Acc: 0.7986
val Loss: 0.0103 Acc: 0.7153

Epoch 55/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0071 Acc: 0.7948
val Loss: 0.0103 Acc: 0.7161

Epoch 56/84
----------
train Loss: 0.0070 Acc: 0.7985
val Loss: 0.0103 Acc: 0.7161

Epoch 57/84
----------
train Loss: 0.0071 Acc: 0.7979
val Loss: 0.0103 Acc: 0.7157

Epoch 58/84
----------
train Loss: 0.0071 Acc: 0.7988
val Loss: 0.0103 Acc: 0.7149

Epoch 59/84
----------
train Loss: 0.0070 Acc: 0.8002
val Loss: 0.0103 Acc: 0.7161

Epoch 60/84
----------
train Loss: 0.0071 Acc: 0.7926
val Loss: 0.0103 Acc: 0.7174

Epoch 61/84
----------
train Loss: 0.0070 Acc: 0.7990
val Loss: 0.0103 Acc: 0.7170

Epoch 62/84
----------
train Loss: 0.0070 Acc: 0.7980
val Loss: 0.0103 Acc: 0.7125

Epoch 63/84
----------
train Loss: 0.0070 Acc: 0.7980
val Loss: 0.0103 Acc: 0.7145

Epoch 64/84
----------
train Loss: 0.0070 Acc: 0.8019
val Loss: 0.0103 Acc: 0.7174

Epoch 65/84
----------
train Loss: 0.0072 Acc: 0.7947
val Loss: 0.0103 Acc: 0.7149

Epoch 66/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0071 Acc: 0.7962
val Loss: 0.0103 Acc: 0.7141

Epoch 67/84
----------
train Loss: 0.0071 Acc: 0.7969
val Loss: 0.0103 Acc: 0.7149

Epoch 68/84
----------
train Loss: 0.0071 Acc: 0.7946
val Loss: 0.0103 Acc: 0.7145

Epoch 69/84
----------
train Loss: 0.0070 Acc: 0.7966
val Loss: 0.0103 Acc: 0.7141

Epoch 70/84
----------
train Loss: 0.0071 Acc: 0.7911
val Loss: 0.0103 Acc: 0.7121

Epoch 71/84
----------
train Loss: 0.0071 Acc: 0.7998
val Loss: 0.0103 Acc: 0.7149

Epoch 72/84
----------
train Loss: 0.0070 Acc: 0.7967
val Loss: 0.0103 Acc: 0.7137

Epoch 73/84
----------
train Loss: 0.0070 Acc: 0.7998
val Loss: 0.0103 Acc: 0.7157

Epoch 74/84
----------
train Loss: 0.0070 Acc: 0.7952
val Loss: 0.0103 Acc: 0.7161

Epoch 75/84
----------
train Loss: 0.0071 Acc: 0.7971
val Loss: 0.0103 Acc: 0.7170

Epoch 76/84
----------
train Loss: 0.0071 Acc: 0.7952
val Loss: 0.0104 Acc: 0.7125

Epoch 77/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0070 Acc: 0.7992
val Loss: 0.0103 Acc: 0.7165

Epoch 78/84
----------
train Loss: 0.0071 Acc: 0.7925
val Loss: 0.0103 Acc: 0.7141

Epoch 79/84
----------
train Loss: 0.0071 Acc: 0.7946
val Loss: 0.0103 Acc: 0.7157

Epoch 80/84
----------
train Loss: 0.0070 Acc: 0.7990
val Loss: 0.0103 Acc: 0.7145

Epoch 81/84
----------
train Loss: 0.0071 Acc: 0.7924
val Loss: 0.0103 Acc: 0.7129

Epoch 82/84
----------
train Loss: 0.0071 Acc: 0.7966
val Loss: 0.0103 Acc: 0.7149

Epoch 83/84
----------
train Loss: 0.0070 Acc: 0.7991
val Loss: 0.0103 Acc: 0.7145

Epoch 84/84
----------
train Loss: 0.0071 Acc: 0.7970
val Loss: 0.0103 Acc: 0.7149

Best val Acc: 0.717356

---Fine tuning.---
Epoch 0/84
----------
LR is set to 0.01
train Loss: 0.0091 Acc: 0.7184
val Loss: 0.0129 Acc: 0.6545

Epoch 1/84
----------
train Loss: 0.0053 Acc: 0.8317
val Loss: 0.0097 Acc: 0.7397

Epoch 2/84
----------
train Loss: 0.0033 Acc: 0.9009
val Loss: 0.0089 Acc: 0.7733

Epoch 3/84
----------
train Loss: 0.0023 Acc: 0.9319
val Loss: 0.0085 Acc: 0.7916

Epoch 4/84
----------
train Loss: 0.0018 Acc: 0.9477
val Loss: 0.0083 Acc: 0.7908

Epoch 5/84
----------
train Loss: 0.0012 Acc: 0.9626
val Loss: 0.0084 Acc: 0.7952

Epoch 6/84
----------
train Loss: 0.0010 Acc: 0.9699
val Loss: 0.0080 Acc: 0.8025

Epoch 7/84
----------
train Loss: 0.0009 Acc: 0.9745
val Loss: 0.0082 Acc: 0.8049

Epoch 8/84
----------
train Loss: 0.0007 Acc: 0.9822
val Loss: 0.0079 Acc: 0.8200

Epoch 9/84
----------
train Loss: 0.0006 Acc: 0.9810
val Loss: 0.0080 Acc: 0.8127

Epoch 10/84
----------
train Loss: 0.0006 Acc: 0.9811
val Loss: 0.0083 Acc: 0.8122

Epoch 11/84
----------
LR is set to 0.001
train Loss: 0.0004 Acc: 0.9860
val Loss: 0.0078 Acc: 0.8179

Epoch 12/84
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0078 Acc: 0.8228

Epoch 13/84
----------
train Loss: 0.0003 Acc: 0.9894
val Loss: 0.0078 Acc: 0.8208

Epoch 14/84
----------
train Loss: 0.0003 Acc: 0.9892
val Loss: 0.0077 Acc: 0.8248

Epoch 15/84
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0078 Acc: 0.8252

Epoch 16/84
----------
train Loss: 0.0003 Acc: 0.9901
val Loss: 0.0078 Acc: 0.8268

Epoch 17/84
----------
train Loss: 0.0003 Acc: 0.9906
val Loss: 0.0078 Acc: 0.8281

Epoch 18/84
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0078 Acc: 0.8220

Epoch 19/84
----------
train Loss: 0.0003 Acc: 0.9905
val Loss: 0.0078 Acc: 0.8240

Epoch 20/84
----------
train Loss: 0.0003 Acc: 0.9903
val Loss: 0.0078 Acc: 0.8256

Epoch 21/84
----------
train Loss: 0.0003 Acc: 0.9900
val Loss: 0.0077 Acc: 0.8248

Epoch 22/84
----------
LR is set to 0.00010000000000000002
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0077 Acc: 0.8268

Epoch 23/84
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0077 Acc: 0.8248

Epoch 24/84
----------
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0078 Acc: 0.8240

Epoch 25/84
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0078 Acc: 0.8248

Epoch 26/84
----------
train Loss: 0.0003 Acc: 0.9904
val Loss: 0.0077 Acc: 0.8248

Epoch 27/84
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0077 Acc: 0.8244

Epoch 28/84
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0077 Acc: 0.8244

Epoch 29/84
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0077 Acc: 0.8228

Epoch 30/84
----------
train Loss: 0.0002 Acc: 0.9908
val Loss: 0.0077 Acc: 0.8248

Epoch 31/84
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0078 Acc: 0.8252

Epoch 32/84
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0078 Acc: 0.8252

Epoch 33/84
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0077 Acc: 0.8260

Epoch 34/84
----------
train Loss: 0.0003 Acc: 0.9915
val Loss: 0.0078 Acc: 0.8256

Epoch 35/84
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0078 Acc: 0.8228

Epoch 36/84
----------
train Loss: 0.0003 Acc: 0.9914
val Loss: 0.0078 Acc: 0.8240

Epoch 37/84
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0077 Acc: 0.8264

Epoch 38/84
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0077 Acc: 0.8260

Epoch 39/84
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0078 Acc: 0.8232

Epoch 40/84
----------
train Loss: 0.0002 Acc: 0.9924
val Loss: 0.0077 Acc: 0.8256

Epoch 41/84
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0077 Acc: 0.8240

Epoch 42/84
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0077 Acc: 0.8244

Epoch 43/84
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0078 Acc: 0.8252

Epoch 44/84
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0003 Acc: 0.9908
val Loss: 0.0078 Acc: 0.8228

Epoch 45/84
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0078 Acc: 0.8264

Epoch 46/84
----------
train Loss: 0.0002 Acc: 0.9923
val Loss: 0.0078 Acc: 0.8252

Epoch 47/84
----------
train Loss: 0.0002 Acc: 0.9914
val Loss: 0.0078 Acc: 0.8260

Epoch 48/84
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0078 Acc: 0.8240

Epoch 49/84
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0078 Acc: 0.8268

Epoch 50/84
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0078 Acc: 0.8264

Epoch 51/84
----------
train Loss: 0.0003 Acc: 0.9917
val Loss: 0.0078 Acc: 0.8244

Epoch 52/84
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0077 Acc: 0.8268

Epoch 53/84
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0078 Acc: 0.8260

Epoch 54/84
----------
train Loss: 0.0002 Acc: 0.9911
val Loss: 0.0077 Acc: 0.8264

Epoch 55/84
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0078 Acc: 0.8240

Epoch 56/84
----------
train Loss: 0.0002 Acc: 0.9916
val Loss: 0.0078 Acc: 0.8260

Epoch 57/84
----------
train Loss: 0.0003 Acc: 0.9893
val Loss: 0.0077 Acc: 0.8248

Epoch 58/84
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0078 Acc: 0.8252

Epoch 59/84
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0078 Acc: 0.8252

Epoch 60/84
----------
train Loss: 0.0003 Acc: 0.9925
val Loss: 0.0078 Acc: 0.8248

Epoch 61/84
----------
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0078 Acc: 0.8244

Epoch 62/84
----------
train Loss: 0.0002 Acc: 0.9915
val Loss: 0.0078 Acc: 0.8244

Epoch 63/84
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0078 Acc: 0.8264

Epoch 64/84
----------
train Loss: 0.0003 Acc: 0.9910
val Loss: 0.0078 Acc: 0.8240

Epoch 65/84
----------
train Loss: 0.0002 Acc: 0.9909
val Loss: 0.0078 Acc: 0.8240

Epoch 66/84
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0002 Acc: 0.9912
val Loss: 0.0077 Acc: 0.8228

Epoch 67/84
----------
train Loss: 0.0002 Acc: 0.9907
val Loss: 0.0078 Acc: 0.8236

Epoch 68/84
----------
train Loss: 0.0002 Acc: 0.9917
val Loss: 0.0078 Acc: 0.8268

Epoch 69/84
----------
train Loss: 0.0002 Acc: 0.9922
val Loss: 0.0078 Acc: 0.8252

Epoch 70/84
----------
train Loss: 0.0003 Acc: 0.9916
val Loss: 0.0078 Acc: 0.8252

Epoch 71/84
----------
train Loss: 0.0002 Acc: 0.9925
val Loss: 0.0078 Acc: 0.8256

Epoch 72/84
----------
train Loss: 0.0002 Acc: 0.9910
val Loss: 0.0078 Acc: 0.8240

Epoch 73/84
----------
train Loss: 0.0002 Acc: 0.9913
val Loss: 0.0078 Acc: 0.8220

Epoch 74/84
----------
train Loss: 0.0002 Acc: 0.9920
val Loss: 0.0078 Acc: 0.8252

Epoch 75/84
----------
train Loss: 0.0002 Acc: 0.9921
val Loss: 0.0078 Acc: 0.8240

Epoch 76/84
----------
train Loss: 0.0003 Acc: 0.9912
val Loss: 0.0077 Acc: 0.8260

Epoch 77/84
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0003 Acc: 0.9909
val Loss: 0.0077 Acc: 0.8228

Epoch 78/84
----------
train Loss: 0.0003 Acc: 0.9913
val Loss: 0.0077 Acc: 0.8256

Epoch 79/84
----------
train Loss: 0.0002 Acc: 0.9919
val Loss: 0.0078 Acc: 0.8260

Epoch 80/84
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0079 Acc: 0.8256

Epoch 81/84
----------
train Loss: 0.0003 Acc: 0.9907
val Loss: 0.0078 Acc: 0.8252

Epoch 82/84
----------
train Loss: 0.0003 Acc: 0.9915
val Loss: 0.0077 Acc: 0.8256

Epoch 83/84
----------
train Loss: 0.0002 Acc: 0.9918
val Loss: 0.0077 Acc: 0.8252

Epoch 84/84
----------
train Loss: 0.0002 Acc: 0.9906
val Loss: 0.0078 Acc: 0.8240

Best val Acc: 0.828062

---Testing---
Test accuracy: 0.959053
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 89 %
Accuracy of Atlantic bluefin tuna : 71 %
Accuracy of   BET : 98 %
Accuracy of Bigeye tuna : 85 %
Accuracy of Blackfin tuna : 97 %
Accuracy of Bullet tuna : 87 %
Accuracy of Carcharhiniformes : 93 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 99 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 94 %
Accuracy of Frigate tuna : 79 %
Accuracy of Heterodontiformes : 97 %
Accuracy of Hexagrammos agrammus : 92 %
Accuracy of Hexanchiformes : 100 %
Accuracy of Konosirus punctatus : 97 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 96 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 96 %
Accuracy of Little tunny : 90 %
Accuracy of Longtail tuna : 93 %
Accuracy of Mackerel tuna : 82 %
Accuracy of Miichthys miiuy : 96 %
Accuracy of Mugil cephalus : 98 %
Accuracy of Myliobatiformes : 88 %
Accuracy of   NoF : 97 %
Accuracy of Oncorhynchus keta : 93 %
Accuracy of Oncorhynchus masou : 93 %
Accuracy of Oplegnathus fasciatus : 98 %
Accuracy of Orectolobiformes : 95 %
Accuracy of Pacific bluefin tuna : 82 %
Accuracy of Paralichthys olivaceus : 95 %
Accuracy of Pleuronectidae : 96 %
Accuracy of Pristiformes : 97 %
Accuracy of Rajiformes : 98 %
Accuracy of Rhinobatiformes : 97 %
Accuracy of SHARK on boat : 100 %
Accuracy of Scomber japonicus : 97 %
Accuracy of Sebastes inermis : 99 %
Accuracy of Seriola quinqueradiata : 98 %
Accuracy of Skipjack tuna : 92 %
Accuracy of Slender tuna : 85 %
Accuracy of Southern bluefin tuna : 80 %
Accuracy of Squaliformes : 96 %
Accuracy of Squatiniformes : 93 %
Accuracy of Stephanolepis cirrhifer : 99 %
Accuracy of Tetraodon or Diodon : 100 %
Accuracy of Thunnus orientalis : 84 %
Accuracy of Torpediniformes : 98 %
Accuracy of Trachurus japonicus : 94 %
Accuracy of   YFT : 99 %
Accuracy of Yellowfin tuna : 94 %
Accuracy of holocephalan : 93 %
Accuracy of mullet : 40 %
Accuracy of   ray : 78 %
Accuracy of rough : 86 %
Accuracy of shark : 98 %
mean: 0.9283703502985956, std: 0.09438124658953587
--------------------

run info[val: 0.3, epoch: 42, randcrop: True, decay: 3]

---Training last layer.---
Epoch 0/41
----------
LR is set to 0.01
train Loss: 0.0249 Acc: 0.3897
val Loss: 0.0171 Acc: 0.5496

Epoch 1/41
----------
train Loss: 0.0152 Acc: 0.5917
val Loss: 0.0143 Acc: 0.6056

Epoch 2/41
----------
train Loss: 0.0129 Acc: 0.6347
val Loss: 0.0128 Acc: 0.6469

Epoch 3/41
----------
LR is set to 0.001
train Loss: 0.0112 Acc: 0.6952
val Loss: 0.0123 Acc: 0.6596

Epoch 4/41
----------
train Loss: 0.0110 Acc: 0.7016
val Loss: 0.0122 Acc: 0.6629

Epoch 5/41
----------
train Loss: 0.0109 Acc: 0.6996
val Loss: 0.0121 Acc: 0.6650

Epoch 6/41
----------
LR is set to 0.00010000000000000002
train Loss: 0.0108 Acc: 0.7058
val Loss: 0.0121 Acc: 0.6664

Epoch 7/41
----------
train Loss: 0.0108 Acc: 0.7066
val Loss: 0.0121 Acc: 0.6653

Epoch 8/41
----------
train Loss: 0.0107 Acc: 0.7102
val Loss: 0.0121 Acc: 0.6642

Epoch 9/41
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0108 Acc: 0.7032
val Loss: 0.0121 Acc: 0.6640

Epoch 10/41
----------
train Loss: 0.0107 Acc: 0.7094
val Loss: 0.0121 Acc: 0.6640

Epoch 11/41
----------
train Loss: 0.0107 Acc: 0.7057
val Loss: 0.0121 Acc: 0.6659

Epoch 12/41
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0107 Acc: 0.7086
val Loss: 0.0121 Acc: 0.6669

Epoch 13/41
----------
train Loss: 0.0108 Acc: 0.7056
val Loss: 0.0121 Acc: 0.6672

Epoch 14/41
----------
train Loss: 0.0108 Acc: 0.6997
val Loss: 0.0121 Acc: 0.6656

Epoch 15/41
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0108 Acc: 0.7048
val Loss: 0.0121 Acc: 0.6656

Epoch 16/41
----------
train Loss: 0.0108 Acc: 0.7034
val Loss: 0.0121 Acc: 0.6688

Epoch 17/41
----------
train Loss: 0.0107 Acc: 0.7103
val Loss: 0.0121 Acc: 0.6669

Epoch 18/41
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0107 Acc: 0.7078
val Loss: 0.0121 Acc: 0.6659

Epoch 19/41
----------
train Loss: 0.0108 Acc: 0.7072
val Loss: 0.0121 Acc: 0.6669

Epoch 20/41
----------
train Loss: 0.0107 Acc: 0.7121
val Loss: 0.0121 Acc: 0.6661

Epoch 21/41
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0107 Acc: 0.7020
val Loss: 0.0121 Acc: 0.6677

Epoch 22/41
----------
train Loss: 0.0108 Acc: 0.7048
val Loss: 0.0121 Acc: 0.6645

Epoch 23/41
----------
train Loss: 0.0107 Acc: 0.7103
val Loss: 0.0121 Acc: 0.6645

Epoch 24/41
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0107 Acc: 0.7129
val Loss: 0.0121 Acc: 0.6672

Epoch 25/41
----------
train Loss: 0.0107 Acc: 0.7102
val Loss: 0.0121 Acc: 0.6650

Epoch 26/41
----------
train Loss: 0.0108 Acc: 0.7027
val Loss: 0.0121 Acc: 0.6669

Epoch 27/41
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0107 Acc: 0.7099
val Loss: 0.0121 Acc: 0.6664

Epoch 28/41
----------
train Loss: 0.0107 Acc: 0.7074
val Loss: 0.0121 Acc: 0.6667

Epoch 29/41
----------
train Loss: 0.0107 Acc: 0.7050
val Loss: 0.0121 Acc: 0.6688

Epoch 30/41
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0108 Acc: 0.7043
val Loss: 0.0121 Acc: 0.6653

Epoch 31/41
----------
train Loss: 0.0108 Acc: 0.7054
val Loss: 0.0121 Acc: 0.6650

Epoch 32/41
----------
train Loss: 0.0107 Acc: 0.7055
val Loss: 0.0121 Acc: 0.6677

Epoch 33/41
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0107 Acc: 0.7104
val Loss: 0.0121 Acc: 0.6677

Epoch 34/41
----------
train Loss: 0.0107 Acc: 0.7081
val Loss: 0.0121 Acc: 0.6683

Epoch 35/41
----------
train Loss: 0.0108 Acc: 0.7036
val Loss: 0.0121 Acc: 0.6648

Epoch 36/41
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0108 Acc: 0.7085
val Loss: 0.0121 Acc: 0.6659

Epoch 37/41
----------
train Loss: 0.0107 Acc: 0.7135
val Loss: 0.0121 Acc: 0.6664

Epoch 38/41
----------
train Loss: 0.0108 Acc: 0.7041
val Loss: 0.0121 Acc: 0.6680

Epoch 39/41
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0107 Acc: 0.7049
val Loss: 0.0121 Acc: 0.6659

Epoch 40/41
----------
train Loss: 0.0107 Acc: 0.7082
val Loss: 0.0121 Acc: 0.6656

Epoch 41/41
----------
train Loss: 0.0108 Acc: 0.7066
val Loss: 0.0121 Acc: 0.6650

Best val Acc: 0.668829

---Fine tuning.---
Epoch 0/41
----------
LR is set to 0.01
train Loss: 0.0100 Acc: 0.7034
val Loss: 0.0103 Acc: 0.7007

Epoch 1/41
----------
train Loss: 0.0060 Acc: 0.8139
val Loss: 0.0084 Acc: 0.7634

Epoch 2/41
----------
train Loss: 0.0040 Acc: 0.8806
val Loss: 0.0081 Acc: 0.7689

Epoch 3/41
----------
LR is set to 0.001
train Loss: 0.0024 Acc: 0.9350
val Loss: 0.0066 Acc: 0.8172

Epoch 4/41
----------
train Loss: 0.0019 Acc: 0.9508
val Loss: 0.0065 Acc: 0.8151

Epoch 5/41
----------
train Loss: 0.0018 Acc: 0.9567
val Loss: 0.0064 Acc: 0.8191

Epoch 6/41
----------
LR is set to 0.00010000000000000002
train Loss: 0.0016 Acc: 0.9629
val Loss: 0.0064 Acc: 0.8210

Epoch 7/41
----------
train Loss: 0.0016 Acc: 0.9598
val Loss: 0.0064 Acc: 0.8221

Epoch 8/41
----------
train Loss: 0.0016 Acc: 0.9625
val Loss: 0.0064 Acc: 0.8205

Epoch 9/41
----------
LR is set to 1.0000000000000003e-05
train Loss: 0.0016 Acc: 0.9653
val Loss: 0.0064 Acc: 0.8191

Epoch 10/41
----------
train Loss: 0.0016 Acc: 0.9634
val Loss: 0.0064 Acc: 0.8191

Epoch 11/41
----------
train Loss: 0.0016 Acc: 0.9631
val Loss: 0.0064 Acc: 0.8208

Epoch 12/41
----------
LR is set to 1.0000000000000002e-06
train Loss: 0.0016 Acc: 0.9640
val Loss: 0.0064 Acc: 0.8213

Epoch 13/41
----------
train Loss: 0.0015 Acc: 0.9664
val Loss: 0.0064 Acc: 0.8221

Epoch 14/41
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0064 Acc: 0.8213

Epoch 15/41
----------
LR is set to 1.0000000000000002e-07
train Loss: 0.0016 Acc: 0.9631
val Loss: 0.0064 Acc: 0.8221

Epoch 16/41
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0064 Acc: 0.8216

Epoch 17/41
----------
train Loss: 0.0016 Acc: 0.9646
val Loss: 0.0064 Acc: 0.8216

Epoch 18/41
----------
LR is set to 1.0000000000000004e-08
train Loss: 0.0016 Acc: 0.9632
val Loss: 0.0064 Acc: 0.8205

Epoch 19/41
----------
train Loss: 0.0016 Acc: 0.9639
val Loss: 0.0064 Acc: 0.8216

Epoch 20/41
----------
train Loss: 0.0016 Acc: 0.9612
val Loss: 0.0064 Acc: 0.8221

Epoch 21/41
----------
LR is set to 1.0000000000000005e-09
train Loss: 0.0016 Acc: 0.9642
val Loss: 0.0064 Acc: 0.8205

Epoch 22/41
----------
train Loss: 0.0016 Acc: 0.9658
val Loss: 0.0064 Acc: 0.8213

Epoch 23/41
----------
train Loss: 0.0016 Acc: 0.9624
val Loss: 0.0064 Acc: 0.8213

Epoch 24/41
----------
LR is set to 1.0000000000000006e-10
train Loss: 0.0016 Acc: 0.9614
val Loss: 0.0064 Acc: 0.8186

Epoch 25/41
----------
train Loss: 0.0016 Acc: 0.9637
val Loss: 0.0064 Acc: 0.8208

Epoch 26/41
----------
train Loss: 0.0017 Acc: 0.9613
val Loss: 0.0064 Acc: 0.8216

Epoch 27/41
----------
LR is set to 1.0000000000000004e-11
train Loss: 0.0016 Acc: 0.9634
val Loss: 0.0064 Acc: 0.8218

Epoch 28/41
----------
train Loss: 0.0016 Acc: 0.9621
val Loss: 0.0064 Acc: 0.8227

Epoch 29/41
----------
train Loss: 0.0016 Acc: 0.9641
val Loss: 0.0064 Acc: 0.8216

Epoch 30/41
----------
LR is set to 1.0000000000000006e-12
train Loss: 0.0016 Acc: 0.9643
val Loss: 0.0064 Acc: 0.8208

Epoch 31/41
----------
train Loss: 0.0016 Acc: 0.9614
val Loss: 0.0064 Acc: 0.8229

Epoch 32/41
----------
train Loss: 0.0016 Acc: 0.9636
val Loss: 0.0064 Acc: 0.8208

Epoch 33/41
----------
LR is set to 1.0000000000000007e-13
train Loss: 0.0016 Acc: 0.9647
val Loss: 0.0064 Acc: 0.8218

Epoch 34/41
----------
train Loss: 0.0016 Acc: 0.9648
val Loss: 0.0064 Acc: 0.8202

Epoch 35/41
----------
train Loss: 0.0016 Acc: 0.9625
val Loss: 0.0064 Acc: 0.8202

Epoch 36/41
----------
LR is set to 1.0000000000000006e-14
train Loss: 0.0016 Acc: 0.9618
val Loss: 0.0064 Acc: 0.8205

Epoch 37/41
----------
train Loss: 0.0016 Acc: 0.9603
val Loss: 0.0064 Acc: 0.8210

Epoch 38/41
----------
train Loss: 0.0016 Acc: 0.9625
val Loss: 0.0064 Acc: 0.8200

Epoch 39/41
----------
LR is set to 1.0000000000000007e-15
train Loss: 0.0016 Acc: 0.9650
val Loss: 0.0064 Acc: 0.8194

Epoch 40/41
----------
train Loss: 0.0016 Acc: 0.9628
val Loss: 0.0064 Acc: 0.8227

Epoch 41/41
----------
train Loss: 0.0016 Acc: 0.9624
val Loss: 0.0064 Acc: 0.8191

Best val Acc: 0.822925

---Testing---
Test accuracy: 0.928809
--------------------
Accuracy of   ALB : 99 %
Accuracy of Albacore tuna : 81 %
Accuracy of Atlantic bluefin tuna : 67 %
Accuracy of   BET : 94 %
Accuracy of Bigeye tuna : 76 %
Accuracy of Blackfin tuna : 94 %
Accuracy of Bullet tuna : 84 %
Accuracy of Carcharhiniformes : 90 %
Accuracy of Conger myriaster : 98 %
Accuracy of   DOL : 96 %
Accuracy of Dasyatiformes : 100 %
Accuracy of Epinephelus septemfasciatus : 94 %
Accuracy of Frigate tuna : 34 %
Accuracy of Heterodontiformes : 94 %
Accuracy of Hexagrammos agrammus : 83 %
Accuracy of Hexanchiformes : 96 %
Accuracy of Konosirus punctatus : 95 %
Accuracy of   LAG : 98 %
Accuracy of Lamniformes : 91 %
Accuracy of Larimichthys polyactis : 98 %
Accuracy of Lateolabrax japonicus : 93 %
Accuracy of Little tunny : 80 %
Accuracy of Longtail tuna : 88 %
Accuracy of Mackerel tuna : 60 %
Accuracy of Miichthys miiuy : 93 %
Accuracy of Mugil cephalus : 94 %
Accuracy of Myliobatiformes : 81 %
Accuracy of   NoF : 93 %
Accuracy of Oncorhynchus keta : 89 %
Accuracy of Oncorhynchus masou : 91 %
Accuracy of Oplegnathus fasciatus : 99 %
Accuracy of Orectolobiformes : 94 %
Accuracy of Pacific bluefin tuna : 59 %
Accuracy of Paralichthys olivaceus : 92 %
Accuracy of Pleuronectidae : 93 %
Accuracy of Pristiformes : 93 %
Accuracy of Rajiformes : 92 %
Accuracy of Rhinobatiformes : 91 %
Accuracy of SHARK on boat : 96 %
Accuracy of Scomber japonicus : 95 %
Accuracy of Sebastes inermis : 98 %
Accuracy of Seriola quinqueradiata : 95 %
Accuracy of Skipjack tuna : 87 %
Accuracy of Slender tuna : 50 %
Accuracy of Southern bluefin tuna : 54 %
Accuracy of Squaliformes : 90 %
Accuracy of Squatiniformes : 88 %
Accuracy of Stephanolepis cirrhifer : 97 %
Accuracy of Tetraodon or Diodon : 99 %
Accuracy of Thunnus orientalis : 75 %
Accuracy of Torpediniformes : 97 %
Accuracy of Trachurus japonicus : 90 %
Accuracy of   YFT : 96 %
Accuracy of Yellowfin tuna : 95 %
Accuracy of holocephalan : 90 %
Accuracy of mullet : 27 %
Accuracy of   ray : 79 %
Accuracy of rough : 76 %
Accuracy of shark : 95 %
mean: 0.8690191689999098, std: 0.1539877787874487

Model saved in "./weights/all_in_one_[0.98]_mean[0.96]_std[0.07].save".
Training complete in 2070m 41s

Process finished with exit code 0
'''