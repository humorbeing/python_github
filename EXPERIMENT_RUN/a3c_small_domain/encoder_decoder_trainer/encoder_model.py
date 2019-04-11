import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from this_utility import *

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # self.kl_tolerance = 0.5
        self.z_size = 32
        self.encoder = nn.Sequential(
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # self.z_layer = nn.Linear(32, self.z_size)
        self.num_actions = 2
        self.lstm = nn.LSTM(
            self.z_size,
            self.z_size * self.num_actions,
            batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.ReLU()
        )

        self.apply(weights_init)
        self.train()
        self.is_cuda = False

    def reconstruction_error_f(self, z, inputs):
        """

        :param z: z is predicted latent vector
            shape: [batch, z_size]
            i.e.: [1000, 32]
        :param inputs: inputs is target, z need to be
            reconstructed as similar as this inputs
            shape: [batch, channel, x(y), y(x)]
            i.e. [1000, 3, 42, 42]
        :return: a real number, here, it's torch.tensor
        """

        x_hat = self.decoder(z)
        r_loss = F.mse_loss(x_hat, inputs)
        return r_loss


    def get_z(self, state):
        state = state.astype(np.float32)
        state = state / 255.0
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.encoder(state)
        z = self.z_layer(x)

        z = z.detach().numpy()
        z = z.squeeze(0)
        # print(z.shape)

        return z

    def realtime_trainer(self, state):
        state = state.astype(np.float32)
        state = state / 255.0
        state = torch.from_numpy(state)

        z = self.encoder(state)
        return z

    def encoder_decoder_trainer(self, state1):
        state1 = state1.astype(np.float32)
        state1 = state1 / 255.0
        state1 = torch.from_numpy(state1)

        if self.is_cuda:
            state1 = state1.cuda()


        z1 = self.encoder(state1)
        x_hat = self.decoder(z1)
        # print(z1.shape)
        # print(x_hat.shape)
        loss = F.mse_loss(x_hat, state1)
        # print(loss)
        return loss
    def sequence_trainer(self, state1, action, state2):
        state1 = state1.astype(np.float32)
        state1 = state1 / 255.0
        state1 = torch.from_numpy(state1)
        state2 = state2.astype(np.float32)
        state2 = state2 / 255.0
        state2 = torch.from_numpy(state2)

        actions = action.T

        actions = actions.astype(np.float32)

        actions = torch.from_numpy(actions)


        if self.is_cuda:
            state1 = state1.cuda()
            state2 = state2.cuda()
            actions = actions.cuda()
        not_actions = actions * (-1) + 1
        z1 = self.encoder(state1)

        z1 = z1.unsqueeze(0)
        self.h, self.c = self.lstm(z1)


        z2 = self.h.squeeze(0)

        pred_action_pair = []
        for i in range(self.num_actions):
            pred = z2[:, i*self.z_size:(i+1)*self.z_size]
            pred_action_pair.append(pred)

        preds = []
        for i in range(self.num_actions):
            pred = self.decoder(pred_action_pair[i])
            preds.append(pred)

        errors = []
        for i in range(self.num_actions):
            error = preds[i] - state2
            error = error**2
            error = torch.sum(error, dim=1)
            errors.append(error)

        mini_losses = []
        maxi_losses = []

        for i in range(self.num_actions):
            mi_loss = errors[i] * actions[i]

            mi_loss = torch.mean(mi_loss)

            mini_losses.append(mi_loss)
            ma_loss = errors[i] * not_actions[i]

            ma_loss = torch.mean(ma_loss)

            maxi_losses.append(ma_loss)

        mini_loss = sum(mini_losses)
        maxi_loss = sum(maxi_losses)

        lllambda = 0.85
        maxi_loss = maxi_loss * (-1)

        loss = mini_loss + maxi_loss * lllambda

        return loss, mini_loss, maxi_loss

    def forward(self, state):

        state = state.astype(np.float32)
        state = state/255.0
        state = torch.from_numpy(state)

        if self.is_cuda:
            state = state.cuda()
        x = self.encoder(state)
        z = self.z_layer(x)

        self.r_loss = self.reconstruction_error_f(z, state)

        return z

if __name__ == '__main__':
    # toy_path = '/mnt/36D4F815D4F7D559/workspace/python_github/__SSSSShare_DDDate_set/pong-ram'
    # VAE_DATA_PATH = toy_path
    filelist = os.listdir(VAE_DATA_PATH)
    # filelist = os.listdir(toy_path)

    log_name = 'enc_dec_1'
    save_name = log_name
    log = Log(log_name)

    lr = 0.001
    is_cuda = True
    # is_cuda = False
    is_save = True
    # is_save = False

    m = encoder()
    if is_cuda:
        m = m.cuda()
        m.is_cuda = True
    optimizer = optim.Adam(m.parameters(), lr=lr, weight_decay=0.0001)
    np.random.shuffle(filelist)
    cuts = filelist[:500]
    EPOCH = 1000
    thre = 0.01
    best_mi = 999
    so_far_best = 999
    for epoch in range(EPOCH):
        lss = []
        miss = []
        mass = []
        for filename in cuts:

            onefilepath = os.path.join(VAE_DATA_PATH, filename)
            raw_data = np.load(onefilepath)
            raw_ob = raw_data['obs']
            raw_action = raw_data['action']

            actions = raw_action - 2
            actions = one_hot(actions)

            input_ob = raw_ob[:-1, :]
            actions = actions[:-1, :]
            output_ob = raw_ob[1:, :]

            loss = m.encoder_decoder_trainer(input_ob)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            lss.append(loss)
            # mi = mi.item()
            # miss.append(mi)
            # ma = ma.item()
            # mass.append(ma)

        mean_ls = np.mean(lss)
        # mean_mi = np.mean(miss)
        # mean_ma = np.mean(mass) * (-1)

        log_string = "epoch: {}, loss: {:0.4f}".format(
            epoch, mean_ls)
        log.log(log_string)


        if mean_ls < so_far_best:
            so_far_best = mean_ls
            if is_save:
                save_this_model(m, save_name)
