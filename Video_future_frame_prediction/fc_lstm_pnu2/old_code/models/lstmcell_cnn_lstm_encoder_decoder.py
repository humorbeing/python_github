# lstmcell, 3 layers, with cnn de deconv(convtraspose)

# encoder - decoder
# v0005

import torch
import torch.nn as nn
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1),
        )
        self.recon_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2),
        )
        self.en1 = nn.LSTMCell(512, 512)
        self.en2 = nn.LSTMCell(512, 512)
        self.en3 = nn.LSTMCell(512, 512)

        self.de1 = nn.LSTMCell(512, 512)
        self.de2 = nn.LSTMCell(512, 512)
        self.de3 = nn.LSTMCell(512, 512)


    def forward(self, x, future_step=10):
        # x in is [seq=10, batch, 64, 64]
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]
        h_e1 = torch.zeros((batch_size, 512)).to(device)
        c_e1 = torch.zeros((batch_size, 512)).to(device)
        h_e2 = torch.zeros((batch_size, 512)).to(device)
        c_e2 = torch.zeros((batch_size, 512)).to(device)
        h_e3 = torch.zeros((batch_size, 512)).to(device)
        c_e3 = torch.zeros((batch_size, 512)).to(device)

        for seq in range(seq_size):
            im = x[seq]
            im = torch.reshape(im, (batch_size, 1, 64, 64))
            im = self.cnn(im)
            im = torch.reshape(im, (batch_size, 512))

            h_e1, c_e1 = self.en1(im, (h_e1, c_e1))
            h_e2, c_e2 = self.en2(h_e1, (h_e2, c_e2))
            h_e3, c_e3 = self.en3(h_e2, (h_e3, c_e3))
        # print(h_e3.shape)
        h_d1 = h_e3
        # h_d1 = torch.zeros((batch_size, 512)).to(device)
        c_d1 = torch.zeros((batch_size, 512)).to(device)
        h_d2 = torch.zeros((batch_size, 512)).to(device)
        c_d2 = torch.zeros((batch_size, 512)).to(device)
        h_d3 = torch.zeros((batch_size, 512)).to(device)
        c_d3 = torch.zeros((batch_size, 512)).to(device)

        zero_input = torch.zeros((batch_size, 512)).to(device)
        outputs = []
        for seq in range(future_step):
            h_d1, c_d1 = self.de1(zero_input, (h_d1, c_d1))
            h_d2, c_d2 = self.de2(h_d1, (h_d2, c_d2))
            h_d3, c_d3 = self.de3(h_d2, (h_d3, c_d3))
            im = torch.reshape(h_d3, (batch_size, 512, 1, 1))
            recon = self.recon_convtranspose(im)
            recon = torch.reshape(recon, (batch_size, 64, 64))
            outputs.append(recon)
        outputs = torch.stack(outputs)
        return outputs




if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((10,100,64,64))
    x1 = model(x)


    print(x1.shape)
