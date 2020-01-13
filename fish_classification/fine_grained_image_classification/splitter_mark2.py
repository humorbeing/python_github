import os
import errno
import numpy as np
import shutil


def splitter(source_path, target_path, new_dataset_name):

    origin_dataset_path = source_path
    origin_dataset_path += '/'
    dataset_new_path = target_path
    dataset_new_path += '/'
    dataset_origin_name = new_dataset_name
    origin_classes = os.listdir(origin_dataset_path)

    new_dataset_name = dataset_origin_name + '_train_test_split'

    new_target_dataset = dataset_new_path + new_dataset_name + '/'

    train_path = new_target_dataset + 'train/'
    test_path = new_target_dataset + 'test/'


    def copy_each(
            cls,
            origin_path,
            train_path,
            test_path,
            random_seed,
            test_size
    ):
        elements = os.listdir(origin_path + cls)
        num_train = len(elements)
        indices = list(range(num_train))
        split = int(np.floor(test_size * num_train))
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
        # np.random.seed()
        train_idx, test_idx = indices[split:], indices[:split]

        for idx in train_idx:
            file_name = elements[idx]
            src = origin_path + cls + '/' + file_name
            dst = train_path + cls
            shutil.copy2(src, dst)

        for idx in test_idx:
            file_name = elements[idx]
            src = origin_path + cls + '/' + file_name
            dst = test_path + cls
            shutil.copy2(src, dst)

    def make_target_set_per_class(
            cls,
            origin_path,
            train_path,
            test_path,
            random_seed,
            test_size
    ):
        try:
            os.makedirs(train_path + cls)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(test_path + cls)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        copy_each(
            cls,
            origin_path,
            train_path,
            test_path,
            random_seed,
            test_size
        )



    for cls in origin_classes:
        make_target_set_per_class(
            cls,
            origin_dataset_path,
            train_path,
            test_path,
            random_seed=5,
            test_size=0.3
        )


# src = '/media/ray/SSD/workspace/python/dataset/original/fish/original'
src = '/media/ray/SSD/workspace/python/dataset/original/CUB_200_2011/CUB_200_2011/images'
dst = '/media/ray/SSD/workspace/python/dataset/trainset_testset'
# name = 'fish_dataset'
name = 'bird_dataset'

splitter(src, dst, name)