from models.simple_encoder_decoder import FC_LSTM as m1
from models.lstmcell_simple_encoder_decoder import FC_LSTM as m2
# from models.lstmcell_cnn_lstm_encoder_decoder import FC_LSTM as m3
from models.lstmcell_cnn_lstm_encoder_decoder_v0002 import FC_LSTM as m3
from models.copycode_de import FC_LSTM as m4
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
from utility import Log

name = 'copy_de'
this_group = 'AO_ori_recon'
this_name = name +'_'+ this_group
batch_size = 10
EPOCH = 1


seed = 6
np.random.seed(seed)
torch.manual_seed(seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'
save_path = path+'fc_lstm_model_save/model_save/'

data_set = np.load(path + 'mnist_test_seq.npy')
# data_set = np.transpose(data_set,(1,0,2,3))
test_set = data_set[:, :1000]
train_set = data_set[:, 1000:7000]
valid_set = data_set[:, 7000:]
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


def running(m, o, n, this_n, batch_size=100, epoch=100):
    this_name = n + '_' + this_n
    model = m().to(device)
    optimizer = o(model.parameters())

    log = Log(this_name)
    # best_loss = 99999
    for e in range(epoch):
        train_rec_loss = []
        rec_loss = []
        # pred_loss = []
        idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
        for i in range(len(train_set[0]) // batch_size):
            model.train()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    train_set[:, idx[i:i + batch_size]], device)

            optimizer.zero_grad()
            rec = model(input_x)
            loss_recon = F.binary_cross_entropy(rec, rec_target)
            loss = loss_recon
            loss.backward()
            optimizer.step()
            train_rec_loss.append(loss.item())
        e_train_rec_loss = np.mean(train_rec_loss)

        for i in range(len(valid_set[0]) // batch_size):
            with torch.no_grad():
                model.eval()
                input_x, rec_target, pred_target =\
                    input_target_maker(
                        valid_set[:, i:i + batch_size], device)
                rec = model(input_x)
                loss_recon = F.binary_cross_entropy(rec, rec_target)

                loss = loss_recon
            rec_loss.append(loss_recon.item())


        rec_l = np.mean(rec_loss)

        log_string = 'Epoch: {}, train ls: {}, eval ls: {}'.format(e, e_train_rec_loss, rec_l)
        log.log(log_string)

oz = []
nz = []

oz.append(optim.Adam)
nz.append('Adam')

oz.append(optim.Adadelta)
nz.append('Adadelta')

oz.append(optim.Adagrad)
nz.append('Adagrad')

oz.append(optim.Adamax)
nz.append('Adamax')

oz.append(optim.ASGD)
nz.append('ASGD')

oz.append(optim.SparseAdam)
nz.append('SparseAdam')

oz.append(optim.SGD)
nz.append('SGD')

oz.append(optim.RMSprop)
nz.append('RMSprop')

oz.append(optim.Rprop)
nz.append('Rprop')

# oz.append(optim.LBFGS)
# nz.append('LBFGS')


for (o, n) in zip(oz, nz):
    running(m4, o, n, this_name, batch_size, EPOCH)