from model_v0002 import FC_LSTM
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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


def input_target_maker(batch, device):
    batch = batch / 255.
    input_x = batch[:, :10, :, :]
    pred_target = batch[:, 10:, :, :]
    rec_target = np.flip(batch[:, :10, :, :], axis=1)
    rec_target = np.ascontiguousarray(rec_target)
    input_x = torch.Tensor(input_x).to(device)
    pred_target = torch.Tensor(pred_target).to(device)
    rec_target = torch.Tensor(rec_target).to(device)
    return input_x, rec_target, pred_target

# mse = F.mse_loss()
# bce = F.binary_cross_entropy
batch_size = 10
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
        pred_loss.append(pred_loss.item())

    rec_l = np.mean(rec_loss)
    pred_l = np.mean(pred_loss)
    print('Epoch: {}, recon loss: {}, pred loss: {}'.format(e, rec_l, pred_l))
    total_loss = 0.9 * rec_l + pred_l
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model, './model.save')