"""
loss is going to infinity negative, i will try exp()
"""
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


    def sequence_trainer(self, state1, action, state2):
        state1 = state1.astype(np.float32)
        state1 = state1 / 255.0
        state1 = torch.from_numpy(state1)
        state2 = state2.astype(np.float32)
        state2 = state2 / 255.0
        state2 = torch.from_numpy(state2)
        # state1 = state1.unsqueeze(0)
        # print(state1.shape)
        z1 = self.encoder(state1)
        z1 = z1.unsqueeze(0)
        self.h, self.c = self.lstm(z1)
        z2 = self.h.squeeze(0)
        # print(z2.shape)
        pred_action_pair = []
        for i in range(self.num_actions):
            pred = z2[:, i*self.z_size:(i+1)*self.z_size]
            pred_action_pair.append(pred)
        # print(len(pred_action_pair))
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
        # print(errors[0].shape)
        # print(actions.shape)
        actions = action*2 - 1
        # print(actions.shape)
        actions = actions.T
        # print(actions.shape)
        actions = actions.astype(np.float32)
        # state1 = state1 / 255.0
        actions = torch.from_numpy(actions)
        losses = []
        for i in range(self.num_actions):
            loss = errors[i]*actions[i]
            # print(loss.shape)
            # print(torch.sum(loss))
            # print(torch.mean(loss))

            loss = torch.mean(loss)
            # print(loss)
            losses.append(loss)
        # print(losses)
        loss = sum(losses)
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
        return loss

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
    filelist = os.listdir(VAE_DATA_PATH)
    # print(filelist)
    # ss('new')
    log = Log('WM_competing_loss')
    # ss('hi')
    lr = 0.001
    m = RNN_WM_competing()
    optimizer = optim.Adam(m.parameters(), lr=lr)
    while True:
        filename = filelist[0]
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

        loss = m.sequence_trainer(input_ob, actions, output_ob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
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
