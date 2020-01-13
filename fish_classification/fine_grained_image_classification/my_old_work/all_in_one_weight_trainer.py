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
