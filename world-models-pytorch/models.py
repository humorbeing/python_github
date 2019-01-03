import torch
from torch import nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.nz = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU())
        self.logvar_fc = nn.Linear(in_features=256 * 2 * 2, out_features=self.nz)
        self.mu_fc = nn.Linear(in_features=256 * 2 * 2, out_features=self.nz)

        self.decode_fc = nn.Linear(in_features=32, out_features=1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid())

        self._initialize_weights()

    def forward(self, x):
        vec = self.encoder(x)

        # flatten
        vec = vec.view(vec.size(0), -1)
        mu = self.mu_fc(vec)
        logvar = self.logvar_fc(vec)
        sigma = torch.exp(logvar/2.0)
        z = self.reparameterize(mu, sigma)
        im = self.decode_fc(z)

        # reshape into im
        im = im[:, :, None, None]

        xr = self.decoder(im)

        return xr, mu, logvar, z


    def decode_this(self, z):
        # pass
        im = self.decode_fc(z)

        # reshape into im
        im = im[:, :, None, None]

        xr = self.decoder(im)
        return xr

    def reparameterize(self, mu, sigma):
        if self.training:
            eps = Variable(torch.randn(*sigma.size()))
            if sigma.is_cuda:
                eps = eps.cuda()
            z = mu + eps * sigma
            return z
        else:
            return mu

    def _initialize_weights(self):
        # pass
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()