from models.simple_encoder_decoder import FC_LSTM as m1
from models.lstmcell_simple_encoder_decoder import FC_LSTM as m2
# from models.lstmcell_cnn_lstm_encoder_decoder import FC_LSTM as m3
from models.lstmcell_cnn_lstm_encoder_decoder_v0002 import FC_LSTM as m3
from models.lstmcell_EDP_v0001 import FC_LSTM as m4
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
from utility import Log

name = 'lstmcell_v0001'
this_group = 'EDP'  # already 55, means input is (-0.5, 0.5)
this_name = name +'_'+ this_group
batch_size = 200
EPOCH = 200
recon_loss_weight = 0.8

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
    batch = (batch - 0.5) / 0.5
    input_x = batch[:10, :, :, :]
    pred_target = batch[10:, :, :, :]
    rec_target = np.flip(batch[:10, :, :, :], axis=0)
    rec_target = np.ascontiguousarray(rec_target)
    input_x = torch.Tensor(input_x).to(device)
    pred_target = torch.Tensor(pred_target).to(device)
    rec_target = torch.Tensor(rec_target).to(device)
    return input_x, rec_target, pred_target

model = m4().to(device)
optimizer = optim.Adam(model.parameters())

log = Log(this_name)
best_loss = 99999
for e in range(EPOCH):
    train_rec_loss = []
    train_pred_loss = []
    rec_loss = []
    pred_loss = []
    idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
    for i in range(len(train_set[0]) // batch_size):
        model.train()
        input_x, rec_target, pred_target =\
            input_target_maker(
                train_set[:, idx[i:i + batch_size]], device)

        optimizer.zero_grad()
        rec, pred = model(input_x)
        loss_recon = F.mse_loss(rec, rec_target)
        loss_pred = F.mse_loss(pred, pred_target)
        loss = recon_loss_weight * loss_recon + loss_pred
        loss.backward()
        optimizer.step()
        train_rec_loss.append(loss_recon.item())
        train_pred_loss.append(loss_pred.item())
    e_train_rec_loss = np.mean(train_rec_loss)
    e_train_pred_loss = np.mean(train_pred_loss)

    for i in range(len(valid_set[0]) // batch_size):
        with torch.no_grad():
            model.eval()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    valid_set[:, i:i + batch_size], device)
            rec, pred = model(input_x)
            loss_recon = F.mse_loss(rec, rec_target)
            loss_pred = F.mse_loss(pred, pred_target)
            loss = loss_recon + loss_pred

        rec_loss.append(loss_recon.item())
        pred_loss.append(loss_pred.item())


    e_eval_rec_loss = np.mean(rec_loss)
    e_eval_pred_loss = np.mean(pred_loss)

    log_string = 'Epoch: {}, train_re: {}, train_pr: {}, eval_rs: {}, eval_pr: {}'\
        .format(e, e_train_rec_loss, e_train_pred_loss, e_eval_rec_loss, e_eval_pred_loss)
    log.log(log_string)
    total_loss = recon_loss_weight * e_eval_rec_loss + e_eval_pred_loss
    if total_loss < best_loss:
        best_loss = total_loss

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path+this_name+'.save')