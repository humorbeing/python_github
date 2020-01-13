import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from this_utility import *

class RNN_WM_competing(nn.Module):
    def __init__(self):
        super(RNN_WM_competing, self).__init__()
        # self.kl_tolerance = 0.5
        self.z_size = 32
        self.encoder = nn.Sequential(
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # self.z_layer = nn.Linear(32, self.z_size)
        self.num_actions = 2
        self.encoder_lstm = nn.LSTM(
            self.z_size,
            self.z_size,
            batch_first=True)
        self.a0_lstm = nn.LSTM(
            self.z_size,
            self.z_size,
            batch_first=True)
        self.a0_layer = nn.Sequential(
            nn.Linear(self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),

        )
        self.a1_lstm = nn.LSTM(
            self.z_size,
            self.z_size,
            batch_first=True)
        self.a1_layer = nn.Sequential(
            nn.Linear(self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 90),
            nn.ReLU(),
            nn.Linear(90, 128),
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


    def sequence_trainer(self, state1, action, state2):
        state1 = state1.astype(np.float32)
        state1 = state1 / 255.0
        state1 = torch.from_numpy(state1)
        state2 = state2.astype(np.float32)
        state2 = state2 / 255.0
        state2 = torch.from_numpy(state2)
        # state1 = state1.unsqueeze(0)
        # print(state1.shape)
        # print('state1', state1, 'len', len(state1))
        # print('state2', state2, 'len', len(state2))
        # print('action', action, 'len', len(action))
        actions = action.T
        # print(actions.shape)
        actions = actions.astype(np.float32)
        # state1 = state1 / 255.0
        actions = torch.from_numpy(actions)
        # print('actions 0', actions[0])
        # print('actions 1', actions[1])

        if self.is_cuda:
            state1 = state1.cuda()
            state2 = state2.cuda()
            actions = actions.cuda()
        not_actions = actions * (-1) + 1
        z1 = self.encoder(state1)
        # print('z1', z1)
        z1 = z1.unsqueeze(0)
        self.h, self.c = self.encoder_lstm(z1)
        # print('h', self.h)
        # print('h shape', self.h.shape)
        z1 = self.h
        # print('z2', z2)
        # print(z2.shape)
        # pred_action_pair = []
        # for i in range(self.num_actions):
        #     pred = z2[:, i*self.z_size:(i+1)*self.z_size]
        #     pred_action_pair.append(pred)
        # print(len(pred_action_pair))
        # print('z2 first', pred_action_pair[0])
        # print('---   ' * 20)
        # print('z2 second', pred_action_pair[1])
        preds = []
        a0_h, a0_c = self.a0_lstm(z1)
        za0 = a0_h.squeeze(0)
        za0 = self.a0_layer(za0)
        pred0 = self.decoder(za0)
        # print(za0.shape)
        # print(pred0.shape)
        preds.append(pred0)

        a1_h, a1_c = self.a1_lstm(z1)
        za1 = a1_h.squeeze(0)
        za1 = self.a1_layer(za1)
        pred1 = self.decoder(za1)
        # print(za0.shape)
        # print(pred0.shape)
        preds.append(pred1)

        # print(state2.shape)
        # for i in range(self.num_actions):
        #     pred = self.decoder(pred_action_pair[i])
        #     preds.append(pred)
        # print('preds 0', preds[0])
        # print('preds 1', preds[1])
        errors = []
        for i in range(self.num_actions):
            error = preds[i] - state2
            error = error**2
            error = torch.sum(error, dim=1)
            errors.append(error)
        # print('error 0', errors[0])
        # print('error 1', errors[1])
        # print(errors[0].shape)
        # print(actions.shape)

        # print('non actions 0', not_actions[0])
        # print('non actions 1', not_actions[1])
        # actions = action
        num_seq = len(actions)
        # print(actions.shape)

        mini_losses = []
        maxi_losses = []

        for i in range(self.num_actions):
            mi_loss = errors[i] * actions[i]
            # print('mi loss', i, mi_loss)
            mi_loss = torch.mean(mi_loss)
            # print('mean mi loss', i, mi_loss)
            mini_losses.append(mi_loss)
            ma_loss = errors[i] * not_actions[i]
            # print('mx loss', i, ma_loss)
            ma_loss = torch.mean(ma_loss)
            # print('mean mx loss', i, ma_loss)
            maxi_losses.append(ma_loss)
        # print(losses)
        mini_loss = sum(mini_losses)
        maxi_loss = sum(maxi_losses)
        # loss = torch.exp(loss)
        # print(loss)
        # a = errors[0][0]
        # print(a)
        # print(a*2)
        # b = torch.Tensor([2])
        # print(b)
        # print(a*2)

        # print(loss)
        # print(errors)
        # print(z)
        # print(state1[9])
        # print(state2[8])
        # print(state2[9])
        # print('yo')
        lllambda = 0.85
        maxi_loss = maxi_loss * (-1)
        # loss = mini_loss * 20 + maxi_loss * (-1)
        loss = mini_loss + maxi_loss * lllambda
        # loss = mini_loss
        # loss = maxi_loss*(-1)
        return loss, mini_loss, maxi_loss

    def forward(self, state):
        # print(state.shape)
        # ss('hi')
        state = state.astype(np.float32)
        state = state/255.0
        state = torch.from_numpy(state)
        # state = state.unsqueeze(0)
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
    # print(filelist)
    # ss('new')
    log_name = 'WM_competing_loss_more_layer_lambda_0.85'
    log = Log(log_name)
    # ss('hi')
    lr = 0.001
    m = RNN_WM_competing()
    is_cuda = False
    is_cuda = True
    if is_cuda:
        m = m.cuda()
        m.is_cuda = True
    optimizer = optim.Adam(m.parameters(), lr=lr, weight_decay=0.0001)
    cuts = filelist[:50]
    # print(cuts)
    EPOCH = 1000
    for epoch in range(EPOCH):
        lss = []
        miss = []
        mass = []
        for filename in cuts:
            # print(filename)
            # filename = filelist[0]
            # print(filename)

            # for filename in filelist:
            onefilepath = os.path.join(VAE_DATA_PATH, filename)
            raw_data = np.load(onefilepath)
            raw_ob = raw_data['obs']
            raw_action = raw_data['action']
            # print(raw_ob.shape)
            # print(raw_action.shape)
            #     ss('ho')
            # one = one_hot(0)
            # print(one)
            actions = raw_action - 2
            actions = one_hot(actions)
            # print(actions)
            # x = 52
            # print(actions[x], raw_action[x])
            input_ob = raw_ob[:-1, :]
            actions = actions[:-1, :]
            output_ob = raw_ob[1:, :]
            # x = 99
            # print(raw_ob[x], input_ob[x])
            # print(raw_action[x], actions[x])
            # print(raw_ob[x+1], output_ob[x])
            # print(len(input_ob))
            # print(len(actions))
            # print(len(output_ob))

            loss, mi, ma = m.sequence_trainer(input_ob, actions, output_ob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            lss.append(loss)
            mi = mi.item()
            miss.append(mi)
            ma = ma.item()
            mass.append(ma)
        log_string = "epoch: {}, loss: {:0.4f}, minimizing loss: {:0.4f}, maximizing loss: {:0.4f}".format(
            epoch, np.mean(loss), np.mean(mi), np.mean(ma*(-1)))
        log.log(log_string)
    ss('hi')
    dataset = creat_dataset(filelist)
    N = len(dataset)
    # ss(N)
    batch_size = 5000
    EPOCH = 100
    is_cuda = True
    # is_cuda = False
    lr = 0.0001
    # is_load = False
    is_load = True
    is_save = True
    # is_save = False
    num_batches = int(np.floor(N / batch_size))

    encoder = ENCODER()
    if is_load:
        encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=lambda storage, loc: storage))
    encoder.train()
    if is_cuda:
        encoder.cuda()
        encoder.is_cuda = True
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(EPOCH):
        # np.random.shuffle(dataset)
        # kl_loss_s = []
        r_loss_s = []
        for idx in range(num_batches):
            batch = dataset[idx * batch_size:(idx + 1) * batch_size]
            # batch = torch.from_numpy(batch)
            # if is_cuda:
            #     batch = batch.cuda()
            z = encoder(batch)
            # ss()
            # kl_loss = vae_model.kl_loss
            r_loss = encoder.r_loss
            # kl_loss_s.append(kl_loss.item())
            r_loss_s.append(r_loss.item())
            loss = r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_string = 'epoch: {}, Reconstruct loss: {}'.format(
            epoch, np.mean(r_loss_s))
        log.log(log_string)
        if (epoch + 1) % 20 == 0:
            if is_save:
                save_this_model(encoder, 'encoder')
    log.end()
    if is_save:
        save_this_model(encoder, 'encoder')
