import torch
from trainer import Trainer
from dataloader import train_set_test_set
from dataloader import test_set

dataset_path = '/media/ray/SSD/workspace/python/dataset/trainset_testset/fish_dataset_train_test_split/train'
d, s, c = train_set_test_set(
    dataset_path, 128
)
t_p = '/media/ray/SSD/workspace/python/dataset/trainset_testset/fish_dataset_train_test_split/test'
t_set = test_set(t_p)
trainer = Trainer(d)
trainer.train()
trainer.test(t_set)