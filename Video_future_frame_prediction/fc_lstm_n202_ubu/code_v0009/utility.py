import torch.nn as nn
import torch.nn.init as init
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def make_gif(imgs, save_name='make',path='./'):
    path += 'imgs/'
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
    # plt.show()


def make_gif2(img1, img2, save_name='make',path='./'):
    path += 'imgs/'
    plt.gray()
    fig, ax = plt.subplots(
        1, 2,
        gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.set_tight_layout(True)
    ax[0].axis('off')
    ax[1].axis('off')
    def images(i):

        ax[0].imshow(img1[i])
        ax[1].imshow(img2[i])
        return ax
    anim = FuncAnimation(
        fig, images,
        frames=np.arange(len(img1)), interval=500)
    if not os.path.exists(path):
        os.makedirs(path)
    anim.save(path+save_name+'.gif', dpi=80, writer='imagemagick')
    # anim.save(path+save_name+'.gif', dpi=80)#, writer='imagemagick')
    # plt.show()

# make a tensor to img interface

def show_images(imgs, save_name='save', path='./'):
    path += 'imgs/'
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




def show_result(args, model, data, device=torch.device("cpu")):
    if args.mode == 'both':
        one_x = data[:, 0:1]
        x, _, _ = input_target_maker(one_x, device=device)
        model.to(device)
        model.eval()
        r, p = model(x)
        r = r.detach().numpy()
        r = np.flip(r, axis=0)
        p = p.detach().numpy()
        y = np.concatenate((r, p), axis=0)
        # show_images(y[:, 0], args.this_name+'pred', path=args.log_path)
        # show_images(one_x[:, 0], args.this_name+'ground_truth', path=args.log_path)
        make_gif2(one_x[:, 0], y[:, 0], args.this_name, path=args.log_path)


if __name__ == '__main__':
    a = np.random.random((20, 64, 64))
    b = np.random.random((20, 64, 64))
    # print(a.shape)
    # print(b)
    show_images(a)
    make_gif2(a, b)
