import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from this_utility import *

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.kl_tolerance = 0.5
        self.z_size = 16
        self.vae_encoder = nn.Sequential(
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.en_log_var = nn.Linear(32, self.z_size)
        self.en_mu = nn.Linear(32, self.z_size)
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.apply(weights_init)
        self.train()
        self.is_cuda = False

    def kl_loss_f(self, mu, logvar):
        """

        :param mu: mu is mean value of Gaussian distribution of latent vector
            shape: [batch, z_size]
            i.e. [1000, 32]
        :param logvar: logvar is log of variance
            standard deviation AKA sigma is square loot variance
            std can not be negative
            variance can not be negative, log variance can BE negative
            shape [batch, z_size]
            i.e. [1000, 32]
        :return: ONE real number, it's in torch.tensor
        """

        kl_loss = 1 + logvar
        kl_loss -= mu.pow(2)
        kl_loss -= logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        kl_loss = torch.clamp(kl_loss, min=self.kl_tolerance * self.z_size)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

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

        x_hat = self.vae_decoder(z)
        r_loss = F.mse_loss(x_hat, inputs)
        return r_loss


    def get_z(self, state):
        state = state.astype(np.float32)
        state = state / 255.0
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)

        x = self.vae_encoder(state)
        mu = self.en_mu(x)
        logvar = self.en_log_var(x)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn(*sigma.size())
        # if self.is_cuda:
        #     epsilon = epsilon.cuda()
        z = mu + epsilon * sigma
        z = z.detach().numpy()
        z = z.squeeze(0)
        # print(z.shape)

        return z

    def forward(self, state):
        # print(state.shape)
        # ss('hi')
        state = state.astype(np.float32)
        state = state/255.0
        state = torch.from_numpy(state)
        # state = state.unsqueeze(0)
        if self.is_cuda:
            state = state.cuda()
        x = self.vae_encoder(state)
        mu = self.en_mu(x)
        logvar = self.en_log_var(x)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn(*sigma.size())
        if self.is_cuda:
            epsilon = epsilon.cuda()
        z = mu + epsilon * sigma

        self.kl_loss = self.kl_loss_f(mu, logvar)
        self.r_loss = self.reconstruction_error_f(z, state)

        return z

if __name__ == '__main__':
    filelist = os.listdir(VAE_DATA_PATH)
    log = Log('vae_loss')
    # ss('hi')
    def creat_dataset(filelist, MAX=1000000):
        np.random.shuffle(filelist)
        data = None
        for filename in filelist:
            onefilepath = os.path.join(VAE_DATA_PATH, filename)
            raw_data = np.load(onefilepath)['obs']
            if data is None:
                data = raw_data
            else:
                data = np.concatenate((data, raw_data), axis=0)
            print('loading:', len(data))
            if len(data) > MAX:
                break
        return data


    dataset = creat_dataset(filelist)
    N = len(dataset)
    # ss(N)
    batch_size = 50000
    EPOCH = 100
    is_cuda = True
    # is_cuda = False
    lr = 0.0001
    # is_load = False
    is_load = True
    is_save = True
    # is_save = False
    num_batches = int(np.floor(N / batch_size))

    vae_model = VAE()
    if is_load:
        vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=lambda storage, loc: storage))
    vae_model.train()
    if is_cuda:
        vae_model.cuda()
        vae_model.is_cuda = True
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    for epoch in range(EPOCH):
        # np.random.shuffle(dataset)
        kl_loss_s = []
        r_loss_s = []
        for idx in range(num_batches):
            batch = dataset[idx * batch_size:(idx + 1) * batch_size]
            # batch = torch.from_numpy(batch)
            # if is_cuda:
            #     batch = batch.cuda()
            z = vae_model(batch)
            # ss()
            kl_loss = vae_model.kl_loss
            r_loss = vae_model.r_loss
            kl_loss_s.append(kl_loss.item())
            r_loss_s.append(r_loss.item())
            loss = kl_loss + r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_string = 'epoch: {}, KL loss: {:.2f}, Reconstruct loss: {}'.format(
            epoch, np.mean(kl_loss_s), np.mean(r_loss_s))
        log.log(log_string)
        if (epoch + 1) % 20 == 0:
            if is_save:
                save_vae_model(vae_model)
    log.end()
    if is_save:
        save_vae_model(vae_model)
