# lstmcell, 3 layers, with cnn de deconv(convtraspose)

# encoder - decoder
# v0005

import torch
import torch.nn as nn
import numpy as np

class cnn(nn.Module):
    def __init__(self, args):
        super(cnn, self).__init__()
        self.args = args
        self.hidden = args.hidden

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 10, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 10, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden, 10, stride=1),
            nn.ReLU(),
        )
        self.recon_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(self.hidden, 64, 10, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 10, stride=2),
            nn.ReLU(),            
            nn.ConvTranspose2d(64, 1, 10, stride=2),
            # nn.Tanh()
        )
        self.en1 = nn.LSTMCell(self.hidden, self.hidden)
        self.en2 = nn.LSTMCell(self.hidden, self.hidden)
        self.en3 = nn.LSTMCell(self.hidden, self.hidden)

        self.de1 = nn.LSTMCell(self.hidden, self.hidden)
        self.de2 = nn.LSTMCell(self.hidden, self.hidden)
        self.de3 = nn.LSTMCell(self.hidden, self.hidden)

        self.pre1 = nn.LSTMCell(self.hidden, self.hidden)
        self.pre2 = nn.LSTMCell(self.hidden, self.hidden)
        self.pre3 = nn.LSTMCell(self.hidden, self.hidden)

        self.pred_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(self.hidden, 64, 10, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 10, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 10, stride=2),
            # nn.Tanh()
        )
    def forward(self, x, future_step=10):
        # x in is [seq=10, batch, 64, 64]
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]
        h_e1 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e1 = torch.zeros((batch_size, self.hidden)).to(device)
        h_e2 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e2 = torch.zeros((batch_size, self.hidden)).to(device)
        h_e3 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e3 = torch.zeros((batch_size, self.hidden)).to(device)

        for seq in range(seq_size):
            im = x[seq]
            im = torch.reshape(im, (batch_size, 1, 64, 64))
            im = self.cnn(im)
            im = torch.reshape(im, (batch_size, self.hidden))

            h_e1, c_e1 = self.en1(im, (h_e1, c_e1))
            h_e2, c_e2 = self.en2(h_e1, (h_e2, c_e2))
            h_e3, c_e3 = self.en3(h_e2, (h_e3, c_e3))
        # print(h_e3.shape)
        h_d1 = h_e3
        # h_d1 = torch.zeros((batch_size, 512)).to(device)
        c_d1 = torch.zeros((batch_size, self.hidden)).to(device)
        h_d2 = torch.zeros((batch_size, self.hidden)).to(device)
        c_d2 = torch.zeros((batch_size, self.hidden)).to(device)
        h_d3 = torch.zeros((batch_size, self.hidden)).to(device)
        c_d3 = torch.zeros((batch_size, self.hidden)).to(device)

        zero_input = torch.zeros((batch_size, self.hidden)).to(device)
        outputs = []
        for seq in range(future_step):
            h_d1, c_d1 = self.de1(zero_input, (h_d1, c_d1))
            h_d2, c_d2 = self.de2(h_d1, (h_d2, c_d2))
            h_d3, c_d3 = self.de3(h_d2, (h_d3, c_d3))
            z = h_d3
            zero_input = z
            im = torch.reshape(z, (batch_size, self.hidden, 1, 1))
            recon = self.recon_convtranspose(im)
            recon = torch.reshape(recon, (batch_size, 64, 64))
            outputs.append(recon)
        outputs = torch.stack(outputs)

        h_p1 = h_e3
        # h_p1 = torch.zeros((batch_size, 512)).to(device)
        c_p1 = torch.zeros((batch_size, self.hidden)).to(device)
        h_p2 = torch.zeros((batch_size, self.hidden)).to(device)
        c_p2 = torch.zeros((batch_size, self.hidden)).to(device)
        h_p3 = torch.zeros((batch_size, self.hidden)).to(device)
        c_p3 = torch.zeros((batch_size, self.hidden)).to(device)

        zero_input = torch.zeros((batch_size, self.hidden)).to(device)
        pre_outputs = []
        for seq in range(future_step):
            h_p1, c_p1 = self.pre1(zero_input, (h_p1, c_p1))
            h_p2, c_p2 = self.pre2(h_p1, (h_p2, c_p2))
            h_p3, c_p3 = self.pre3(h_p2, (h_p3, c_p3))
            z = h_p3
            zero_input = z
            im = torch.reshape(z, (batch_size, self.hidden, 1, 1))
            recon = self.pred_convtranspose(im)
            recon = torch.reshape(recon, (batch_size, 64, 64))
            pre_outputs.append(recon)
        pre_outputs = torch.stack(pre_outputs)
        return outputs, pre_outputs




if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace()
    # args.mode = 'both'  # 'recon' / 'pred' / 'both'
    # args.zero_input = False
    # args.last_activation = 'non'  # tanh / sigmoid / 'non'
    args.hidden = 256
    # model = lstm_v0001(args)
    model = cnn(args)
    x = torch.randn((10, 100, 64, 64))
    x1, x2 = model(x)
    print(x1.shape)
    if type(x2) == int:
        print(x2)
    else:
        print(x2.shape)
