import torch
from models import lstm_copy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
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

def make_gif(imgs, save_name='make',path='./imgs/'):
    plt.gray()
    fig, ax = plt.subplots(
        1, 1,
        gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.set_tight_layout(True)
    def images(i):
        ax.imshow(imgs[i])
        return ax
    anim = FuncAnimation(
        fig, images,
        frames=np.arange(len(imgs)), interval=500)
    if not os.path.exists(path):
        os.makedirs(path)
    anim.save(path+save_name+'.gif', dpi=80, writer='imagemagick')
    plt.show()

# make a tensor to img interface

def show_images(imgs, save_name='save', path='./imgs/'):
    show_size = len(imgs)
    plt.gray()
    fig, axs = plt.subplots(1, show_size,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    # axs[0].set_title('Epoch:' + str(epoch))
    for n in range(show_size):
        axs[n].imshow(imgs[n])
        axs[n].axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * show_size * 0.25)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+save_name+'.png')
    plt.show()

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
one_x = valid_set[:, 0:1]

# ground truth
make_gif(one_x[:, 0], 'ground_truth')
show_images(one_x[:, 0], 'ground_truth')

x, r, p = input_target_maker(one_x, device=device)
r, p = model(x)

r = r.detach().numpy()
r = np.flip(r, axis=0)
p = p.detach().numpy()
y = np.concatenate((r, p), axis=0)
print(y.shape)
# y = r
y = y[:, 0]
show_images(y, 'pred')
make_gif(y, 'pred')


