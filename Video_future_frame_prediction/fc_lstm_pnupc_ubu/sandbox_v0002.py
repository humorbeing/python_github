from model_v0001 import FC_LSTM
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '../../__SSSSTTTTOOOORRRREEEE/Data_save_here/'
# path = '../data_here/'
train_set = np.load(path + 'mnist_test_seq.npy')
train_set = np.transpose(train_set,(1,0,2,3))
# print(train_set.shape)
model = FC_LSTM().to(device)
optimizer = optim.Adam(model.parameters())
# mse = F.mse_loss()
# bce = F.binary_cross_entropy
batch_size = 100
idx = np.random.permutation(len(train_set))  # i.i.d. sampling
EPOCH = 200
for e in range(EPOCH):
    losses = []
    for i in range(len(train_set) // batch_size):
        batch = train_set[idx[i:i + batch_size]]
        batch = batch/255.
        batch = torch.Tensor(batch).to(device)
        optimizer.zero_grad()
        x = model(batch)
        loss = F.mse_loss(x, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ave_loss = np.mean(losses)
    print('Epoch {}. Loss {}'.format(e, ave_loss))
