from models.lstmcell_simple_encoder_decoder import FC_LSTM

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'

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

model = FC_LSTM().to(device)
optimizer = optim.Adam(model.parameters())

batch_size = 20
EPOCH = 200
best_loss = 99999
for e in range(EPOCH):
    rec_loss = []
    pred_loss = []
    idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
    for i in range(len(train_set[0]) // batch_size):
        model.train()
        input_x, rec_target, pred_target =\
            input_target_maker(
                train_set[:, idx[i:i + batch_size]], device)
        # print(input_x.shape)
        optimizer.zero_grad()
        rec = model(input_x)
        # print(rec.shape)
        # print(rec_target.shape)
        loss_recon = F.mse_loss(rec, rec_target)
        # loss_pred = F.mse_loss(pred, pred_target)
        loss = loss_recon
        loss.backward()
        optimizer.step()


    for i in range(len(valid_set[0]) // batch_size):
        with torch.no_grad():
            model.eval()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    valid_set[:, i:i + batch_size], device)
            rec = model(input_x)
            loss_recon = F.mse_loss(rec, rec_target)
            # loss_pred = F.mse_loss(pred, pred_target)
            loss = loss_recon
        rec_loss.append(loss_recon.item())
        # pred_loss.append(loss_pred.item())

    rec_l = np.mean(rec_loss)
    # pred_l = np.mean(pred_loss)
    print('Epoch: {}, recon loss: {}'
          .format(e, rec_l))
    total_loss = rec_l
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model, './model_save/lstmcell_simple_encoder_decoder.save')