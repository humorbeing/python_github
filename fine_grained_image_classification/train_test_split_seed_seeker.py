from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision
import torch
import numpy as np



# Hyper-parameter

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
valid_transform = transforms.Compose([
    transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
# dataset_path = '/media/ray/SSD/workspace/python/dataset/fish/original'
dataset_path = '/media/ray/SSD/workspace/python/dataset/CUB_200_2011/CUB_200_2011/images'
o = datasets.ImageFolder(dataset_path,valid_transform)

# print(len(original_dataset))
# print(len(original_dataset.classes))
# print(original_dataset.class_to_idx)
# print(original_dataset.imgs)

# original_dataset = original_dataset.transform(valid_transform)
def train_test(original_dataset, seed_in):
    num_data = len(original_dataset)
    # num_data = 10
    indices = list(range(num_data))
    split = int(np.floor(test_set_size * num_data))
    np.random.seed(seed_in)
    np.random.shuffle(indices)
    np.random.seed()
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        original_dataset,
        # batch_size=batch_size,
        sampler=train_sampler,
        # num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        original_dataset,
        # batch_size=batch_size,
        sampler=test_sampler,
        # num_workers=num_workers
    )
    return train_loader, test_loader

# tn, tt = train_test(o, 111)
# del o


def loader_class_counter(loader):
    detail = {}
    for _, l in loader:
        label = l.numpy()[0]
        if label in detail:
            detail[label] += 1
        else:
            detail[label] = 1
    return detail

# train = loader_class_counter(tn)
# test = loader_class_counter(tt)


def split_result(tr, te):
    re = []
    for i in range(27):
        if i in te:
            n_te = te[i]
        else:
            n_te = 0
        if i in tr:
            n_tr = tr[i]
        else:
            n_tr = 0
        r = n_te / (n_te + n_tr)
        re.append(r)

    return re

# r = split_result(train, test)
# print(
#     "mean: {}, var: {}, max: {}, min {} | seed: 111".format(
#         round(np.mean(r), 3),
#         round(np.var(r), 6),
#         round(np.max(r), 3),
#         round(np.min(r), 3)
#     )
# )
split_schedule = [0.3]
min_var = 99
the_seed = 0
for sch in split_schedule:
    test_set_size = sch
    for seed in range(5000):
        train_loader, test_loader = train_test(o, seed)
        train_count = loader_class_counter(train_loader)
        test_count = loader_class_counter(test_loader)
        result = split_result(train_count, test_count)
        var = np.var(result)
        if var < min_var:
            min_var = var
            the_seed = seed
        print('[{}] The min var: {}, best seed: {} | on seed: {}'.format(
            sch,
            round(min_var, 6),
            the_seed,
            seed
        ))






