from model_v0003 import FC_LSTM
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'
# path = '../data_here/'
data_set = np.load(path + 'mnist_test_seq.npy')
data_set = np.transpose(data_set,(1,0,2,3))
test_set = data_set[:1000]
train_set = data_set[1000:7000]
valid_set = data_set[7000:]
del data_set

model = FC_LSTM().to(device)
optimizer = optim.Adam(model.parameters())



def make_gif(imgs):
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
    anim.save('make.gif', dpi=80, writer='imagemagick')
    plt.show()

def show_images(imgs):
    show_size = len(imgs)
    plt.gray()
    fig, axs = plt.subplots(1, show_size,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    # axs[0].set_title('Epoch:' + str(epoch))
    for n in range(show_size):
        axs[n].imshow(imgs[n])
        axs[n].axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * show_size * 0.25)
    plt.savefig('save.png')
    plt.show()

def input_target_maker(batch, device):
    batch = batch / 255.
    input_x = batch[:, :10, :, :]
    pred_target = batch[:, 10:, :, :]
    rec_target = np.flip(batch[:, :10, :, :], axis=1)
    rec_target = np.ascontiguousarray(rec_target)
    # make_gif(input_x[0])
    # show_images(batch[0])
    # show_images(input_x[0])
    # show_images(pred_target[0])
    # show_images(rec_target[0])
    input_x = torch.Tensor(input_x).to(device)
    pred_target = torch.Tensor(pred_target).to(device)
    rec_target = torch.Tensor(rec_target).to(device)
    return input_x, rec_target, pred_target

# mse = F.mse_loss()
# bce = F.binary_cross_entropy
batch_size = 100
EPOCH = 200
best_loss = 99999
for e in range(EPOCH):
    rec_loss = []
    pred_loss = []
    idx = np.random.permutation(len(train_set))  # i.i.d. sampling
    for i in range(len(train_set) // batch_size):
        model.train()
        input_x, rec_target, pred_target = input_target_maker(train_set[idx[i:i + batch_size]], device)

        optimizer.zero_grad()
        rec, pred = model(input_x)
        loss_recon = F.mse_loss(rec, rec_target)
        loss_pred = F.mse_loss(pred, pred_target)
        loss = loss_pred + loss_recon
        loss.backward()
        optimizer.step()
    for i in range(len(valid_set) // batch_size):
        with torch.no_grad():
            model.eval()
            input_x, rec_target, pred_target = input_target_maker(valid_set[i:i + batch_size], device)
            rec, pred = model(input_x)
            loss_recon = F.mse_loss(rec, rec_target)
            loss_pred = F.mse_loss(pred, pred_target)
            loss = loss_pred + loss_recon
        rec_loss.append(loss_recon.item())
        pred_loss.append(loss_pred.item())

    rec_l = np.mean(rec_loss)
    pred_l = np.mean(pred_loss)
    print('Epoch: {}, recon loss: {}, pred loss: {}'.format(e, rec_l, pred_l))
    total_loss = 0.9 * rec_l + pred_l
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model, './model_cnn_fcLSTM.save')