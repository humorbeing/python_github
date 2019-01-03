# pytorch_playground
import torch
from torch.autograd import Variable
# torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
# more
import numpy as np
import copy

def train_set_test_set(
        dataset_path,
        test_set_ratio,
        batch_size,
        random_seed=0,
        val_set_ratio=0.3,
        val_random_seed=None,
        image_size=224,
        num_workers=2
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = datasets.ImageFolder(dataset_path, transform_train)
    dataset_test = datasets.ImageFolder(dataset_path, transform_test)

    num_data = len(dataset_train)
    indices = list(range(num_data))
    split = int(np.floor(test_set_ratio * num_data))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    np.random.seed()
    train_idx, test_idx = indices[split:], indices[:split]

    test_sampler = SubsetRandomSampler(test_idx)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )

    val_split = int(np.floor(val_set_ratio * len(train_idx)))
    if val_random_seed:
        np.random.seed(val_random_seed)
        np.random.shuffle(train_idx)
        np.random.seed()
    else:
        np.random.shuffle(train_idx)

    train_idx, val_idx = train_idx[val_split:], train_idx[:val_split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers
    )
    dataset_loaders = {
        'train': train_loader,
        'val': val_loader,
        # 'test': test_loader
    }
    dataset_sizes = {
        'train': len(train_idx),
        'val': len(val_idx),
        'test': len(test_idx)
    }
    dataset_classes = dataset_train.classes

    return dataset_loaders, test_loader, dataset_sizes, dataset_classes


def train_model(
        model,
        dataset_loaders,
        dataset_sizes,
        criterion,
        optimizer,
        learning_rate,
        lr_scheduler,
        decay,
        gpu=False,
        num_epochs=25
):
    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,
                                         lr_decay_epoch=decay,
                                         init_lr=learning_rate)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataset_loaders[phase]:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def exp_lr_scheduler(
        optimizer,
        epoch,
        lr_decay_epoch=7,
        init_lr=0.01,
):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def test_model(
        model_conv,
        test_loader,
        dataset_sizes,
        batch_size,
        gpu
):
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

    test_acc = running_corrects / dataset_sizes['test']

    print("Test accuracy: {:4f}".format(test_acc))
    print('-' * 20)