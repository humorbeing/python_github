
# encoder decoder, only 1 layer, flatten style
# lstmcell
# v0004
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class lstm_v0001(nn.Module):
    def __init__(self, args):
        super(lstm_v0001, self).__init__()
        self.args = args
        self.fc_en1 = nn.Linear(4096, 1024)
        self.fc_en2 = nn.Linear(1024, self.args.hidden)
        self.en1 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.en2 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.en3 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.de1 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.de2 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.de3 = nn.LSTMCell(self.args.hidden, self.args.hidden)
        self.fc_de1 = nn.Linear(self.args.hidden, 1024)
        self.fc_de2 = nn.Linear(1024, 4096)
        if self.args.mode == 'both':
            self.pre1 = nn.LSTMCell(self.args.hidden, self.args.hidden)
            self.pre2 = nn.LSTMCell(self.args.hidden, self.args.hidden)
            self.pre3 = nn.LSTMCell(self.args.hidden, self.args.hidden)
            self.fc_pre1 = nn.Linear(self.args.hidden, 1024)
            self.fc_pre2 = nn.Linear(1024, 4096)


    def forward(self, x, future_step=10):
        # x in is [seq=10, batch, 64, 64]
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]
        h_e1 = torch.zeros((batch_size, self.args.hidden)).to(device)
        c_e1 = torch.zeros((batch_size, self.args.hidden)).to(device)
        h_e2 = torch.zeros((batch_size, self.args.hidden)).to(device)
        c_e2 = torch.zeros((batch_size, self.args.hidden)).to(device)
        h_e3 = torch.zeros((batch_size, self.args.hidden)).to(device)
        c_e3 = torch.zeros((batch_size, self.args.hidden)).to(device)
        x = x.reshape((seq_size, batch_size, -1))
        # print(x.shape)
        for seq in range(seq_size):
            x_ = self.fc_en1(x[seq])
            x_ = F.relu(x_)
            x_ = self.fc_en2(x_)
            x_ = F.relu(x_)
            h_e1, c_e1 = self.en1(x_, (h_e1, c_e1))
            h_e2, c_e2 = self.en2(h_e1, (h_e2, c_e2))
            h_e3, c_e3 = self.en3(h_e2, (h_e3, c_e3))

        h_d1 = h_e3
        # h_d1 = torch.zeros((batch_size, 512)).to(device)
        c_d1 = torch.zeros((batch_size, self.args.hidden)).to(device)
        h_d2 = torch.zeros((batch_size, self.args.hidden)).to(device)
        c_d2 = torch.zeros((batch_size, self.args.hidden)).to(device)
        h_d3 = torch.zeros((batch_size, self.args.hidden)).to(device)
        c_d3 = torch.zeros((batch_size, self.args.hidden)).to(device)

        zero_input = torch.zeros((batch_size, self.args.hidden)).to(device)
        outputs = []
        for seq in range(future_step):
            h_d1, c_d1 = self.de1(zero_input, (h_d1, c_d1))
            h_d2, c_d2 = self.de2(h_d1, (h_d2, c_d2))
            h_d3, c_d3 = self.de3(h_d2, (h_d3, c_d3))
            if not self.args.zero_input:
                zero_input = h_d3

            z = self.fc_de1(h_d3)
            z = F.relu(z)
            z = self.fc_de2(z)

            z = torch.reshape(z, (batch_size, 64, 64))

            outputs.append(z)
        outputs = torch.stack(outputs)

        if self.args.mode == 'both':
            h_p1 = h_e3
            # h_p1 = torch.zeros((batch_size, 512)).to(device)
            c_p1 = torch.zeros((batch_size, self.args.hidden)).to(device)
            h_p2 = torch.zeros((batch_size, self.args.hidden)).to(device)
            c_p2 = torch.zeros((batch_size, self.args.hidden)).to(device)
            h_p3 = torch.zeros((batch_size, self.args.hidden)).to(device)
            c_p3 = torch.zeros((batch_size, self.args.hidden)).to(device)

            zero_input = torch.zeros((batch_size, self.args.hidden)).to(device)
            pre_outputs = []
            for seq in range(future_step):
                h_p1, c_p1 = self.pre1(zero_input, (h_p1, c_p1))
                h_p2, c_p2 = self.pre2(h_p1, (h_p2, c_p2))
                h_p3, c_p3 = self.pre3(h_p2, (h_p3, c_p3))
                if not self.args.zero_input:
                    zero_input = h_p3
                z = self.fc_pre1(h_p3)
                z = F.relu(z)
                z = self.fc_pre2(z)

                z = torch.reshape(z, (batch_size, 64, 64))
                pre_outputs.append(z)
            pre_outputs = torch.stack(pre_outputs)
            return outputs, pre_outputs
        else:
            return outputs, 0


class lstm_copy(nn.Module):
    def __init__(self, args):
        super(lstm_copy, self).__init__()
        self.hidden = args.hidden
        self.args = args
        self.en1 = nn.LSTMCell(4096, self.hidden)
        self.en2 = nn.LSTMCell(self.hidden, self.hidden)
        self.en3 = nn.LSTMCell(self.hidden, self.hidden)

        self.de1 = nn.LSTMCell(4096, self.hidden)
        self.de2 = nn.LSTMCell(self.hidden, self.hidden)
        self.de3 = nn.LSTMCell(self.hidden, 4096)

        if args.mode == 'both':
            self.pre1 = nn.LSTMCell(4096, self.hidden)
            self.pre2 = nn.LSTMCell(self.hidden, self.hidden)
            self.pre3 = nn.LSTMCell(self.hidden, 4096)


    def forward(self, x, future_step=10):
        # x in is [seq=10, batch, 64, 64]
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]
        x = x.reshape((seq_size, batch_size, -1))
        h_e1 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e1 = torch.zeros((batch_size, self.hidden)).to(device)
        h_e2 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e2 = torch.zeros((batch_size, self.hidden)).to(device)
        h_e3 = torch.zeros((batch_size, self.hidden)).to(device)
        c_e3 = torch.zeros((batch_size, self.hidden)).to(device)

        # print(x.shape)
        for seq in range(seq_size):

            h_e1, c_e1 = self.en1(x[seq], (h_e1, c_e1))
            h_e2, c_e2 = self.en2(h_e1, (h_e2, c_e2))
            h_e3, c_e3 = self.en3(h_e2, (h_e3, c_e3))
        # print(h_e3.shape)
        h_d1 = h_e3
        # h_d1 = torch.zeros((batch_size, 512)).to(device)
        c_d1 = torch.zeros((batch_size, self.hidden)).to(device)
        h_d2 = torch.zeros((batch_size, self.hidden)).to(device)
        c_d2 = torch.zeros((batch_size, self.hidden)).to(device)
        h_d3 = torch.zeros((batch_size, 4096)).to(device)
        c_d3 = torch.zeros((batch_size, 4096)).to(device)

        zero_input = torch.zeros((batch_size, 4096)).to(device)
        recon_outputs = []
        for seq in range(future_step):
            h_d1, c_d1 = self.de1(zero_input, (h_d1, c_d1))
            h_d2, c_d2 = self.de2(h_d1, (h_d2, c_d2))
            h_d3, c_d3 = self.de3(h_d2, (h_d3, c_d3))
            z = h_d3
            if self.args.last_activation == 'sigmoid':
                z = torch.sigmoid(z)
            elif self.args.last_activation == '100s':
                z = z * 100
                z = torch.sigmoid(z)
            else:
                # print('non1')
                pass
            if not self.args.zero_input:
                zero_input = z
            z = torch.reshape(z, (batch_size, 64, 64))
            recon_outputs.append(z)

        recon_outputs = torch.stack(recon_outputs)
        # return recon_outputs
        if self.args.mode == 'both':
            h_p1 = h_e3
            # h_p1 = torch.zeros((batch_size, 512)).to(device)
            c_p1 = torch.zeros((batch_size, self.hidden)).to(device)
            h_p2 = torch.zeros((batch_size, self.hidden)).to(device)
            c_p2 = torch.zeros((batch_size, self.hidden)).to(device)
            h_p3 = torch.zeros((batch_size, 4096)).to(device)
            c_p3 = torch.zeros((batch_size, 4096)).to(device)

            pre_outputs = []
            for seq in range(future_step):
                h_p1, c_p1 = self.pre1(zero_input, (h_p1, c_p1))
                h_p2, c_p2 = self.pre2(h_p1, (h_p2, c_p2))
                h_p3, c_p3 = self.pre3(h_p2, (h_p3, c_p3))
                z = h_p3
                if self.args.last_activation == 'sigmoid':
                    z = torch.sigmoid(z)
                elif self.args.last_activation == '100s':
                    z = z * 100
                    z = torch.sigmoid(z)
                else:
                    # print('non2')
                    pass
                if not self.args.zero_input:
                    zero_input = z
                z = torch.reshape(z, (batch_size, 64, 64))
                # print(x_.shape)
                pre_outputs.append(z)
            pre_outputs = torch.stack(pre_outputs)

            return recon_outputs, pre_outputs
        else:
            return recon_outputs, 0

if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace()
    args.mode = 'both'  # 'recon' / 'pred' / 'both'
    args.zero_input = False
    args.last_activation = 'non'  # 100s / sigmoid / 'non'
    args.hidden = 2048
    model = lstm_v0001(args)
    # model = lstm_copy(args)
    x = torch.randn((10, 100, 64, 64))
    x1, x2 = model(x)
    print(x1.shape)
    if type(x2) == int:
        print(x2)
    else:
        print(x2.shape)
    # model = FC_LSTM(256)
    # x = torch.randn((10, 100, 64, 64))
    # x1, x2 = model(x)
    #
    # print(x1.shape)
    # print(x2.shape)