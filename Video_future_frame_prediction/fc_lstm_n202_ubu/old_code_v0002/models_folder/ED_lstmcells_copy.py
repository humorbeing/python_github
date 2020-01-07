
# encoder decoder, only 1 layer, flatten style
# lstmcell
# v0004
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self, args):
        super(FC_LSTM, self).__init__()
        self.hidden = args.hidden

        self.en1 = nn.LSTMCell(4096, self.hidden)
        self.en2 = nn.LSTMCell(self.hidden, self.hidden)
        self.en3 = nn.LSTMCell(self.hidden, self.hidden)

        self.de1 = nn.LSTMCell(4096, self.hidden)
        self.de2 = nn.LSTMCell(self.hidden, self.hidden)
        self.de3 = nn.LSTMCell(self.hidden, 4096)

        # self.pre1 = nn.LSTMCell(4096, hidden)
        # self.pre2 = nn.LSTMCell(hidden, hidden)
        # self.pre3 = nn.LSTMCell(hidden, 4096)


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
            z = torch.sigmoid(z)
            z = torch.reshape(z, (batch_size, 64, 64))
            recon_outputs.append(z)

        recon_outputs = torch.stack(recon_outputs)
        # outputs = torch.reshape(outputs, (seq_size, batch_size, 64, 64))

        # h_p1 = h_e3
        # # h_p1 = torch.zeros((batch_size, 512)).to(device)
        # c_p1 = torch.zeros((batch_size, self.hidden)).to(device)
        # h_p2 = torch.zeros((batch_size, self.hidden)).to(device)
        # c_p2 = torch.zeros((batch_size, self.hidden)).to(device)
        # h_p3 = torch.zeros((batch_size, 4096)).to(device)
        # c_p3 = torch.zeros((batch_size, 4096)).to(device)
        #
        # pre_outputs = []
        # for seq in range(future_step):
        #     h_p1, c_p1 = self.pre1(zero_input, (h_p1, c_p1))
        #     h_p2, c_p2 = self.pre2(h_p1, (h_p2, c_p2))
        #     h_p3, c_p3 = self.pre3(h_p2, (h_p3, c_p3))
        #     z = h_p3
        #     # z = torch.tanh(z)
        #     z = torch.reshape(z, (batch_size, 64, 64))
        #     # print(x_.shape)
        #     pre_outputs.append(z)
        # pre_outputs = torch.stack(pre_outputs)
        return recon_outputs
        # return recon_outputs, pre_outputs


if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((10, 100, 64, 64))
    x1 = model(x)
    print(x1.shape)
    # x1, x2 = model(x)
    # print(x1.shape)
    # print(x2.shape)