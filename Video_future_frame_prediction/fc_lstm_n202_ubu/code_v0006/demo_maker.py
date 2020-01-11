import torch
from models import lstm_copy
import numpy as np
device = torch.device("cpu")
model_path = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/Data_save_here/fc_lstm_model_save/model_save/ED_R_01-both.save'

model = torch.load(model_path)
model.cpu()
# x = torch.randn((10, 100, 64, 64))
# x1, x2 = model(x)
# print(x1.shape)
# if type(x2) == int:
#     print(x2)
# else:
#     print(x2.shape)

data_path = '../../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'
data_set = np.load(data_path + 'mnist_test_seq.npy')
# test_set = data_set[:, :1000]
# train_set = data_set[:, 1000:7000]
# valid_set = data_set[:, 7000:]
# test_set = data_set[:, :1000]
train_set = data_set[:, :9000]
valid_set = data_set[:, 9000:]
del data_set


def input_target_maker(batch, device):
    batch = batch / 255.
    input_x = batch[:10, :, :, :]
    pred_target = batch[10:, :, :, :]
    rec_target = np.flip(batch[:10, :, :, :], axis=0)
    rec_target = np.ascontiguousarray(rec_target)
    input_x = torch.Tensor(input_x).to(device)
    pred_target = torch.Tensor(pred_target).to(device)
    rec_target = torch.Tensor(rec_target).to(device)
    return input_x, rec_target, pred_target

one_x = train_set[:, 0:1]
# one_x = train_set[:, 0]
print(one_x.shape)

x, r, p = input_target_maker(one_x, device=device)
# x1, x2 = model(x)
# x = x2
x = p
print(x.shape)

from utility import show_images
# x = x.numpy()
x = x.detach().numpy()
x = x[:, 0]
show_images(x)

from utility import make_gif

make_gif(x)