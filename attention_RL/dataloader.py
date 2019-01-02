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


image_size = 224

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

def test_set(testset_path):
    test_dataset = datasets.ImageFolder(testset_path, transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=20,
                                                  num_workers=1)
    return test_loader

def train_set_test_set(
        dataset_path,
        batch_size=1,
        val_set_ratio=0.3,
        random_seed=None,
        # image_size=224,
        num_workers=2
):

    # trainset_path = dataset_path + 'train'
    trainset_path = dataset_path
    # testset_path = dataset_path + 'test'

    dataset_train = datasets.ImageFolder(trainset_path, transform_train)
    dataset_val = datasets.ImageFolder(trainset_path, transform_test)
    # dataset_test = datasets.ImageFolder(testset_path, transform_test)

    num_data = len(dataset_train)
    indices = list(range(num_data))
    split = int(np.floor(val_set_ratio * num_data))
    if random_seed:
        print('random seed is:', random_seed)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.seed()
    else:
        print('These is no random seed given.')
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers
    )

    dataset_loaders = {
        0: train_loader,
        1: val_loader,
        # 'test': test_loader
    }
    dataset_sizes = {
        'train': len(train_idx),
        'val': len(valid_idx),
        # 'test': len(test_idx)
    }
    dataset_classes = dataset_train.classes

    return dataset_loaders, dataset_sizes, dataset_classes

