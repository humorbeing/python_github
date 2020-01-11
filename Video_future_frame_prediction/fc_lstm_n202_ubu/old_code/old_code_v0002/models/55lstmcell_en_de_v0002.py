
# encoder decoder, only 1 layer, flatten style
# lstmcell
# v0004
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self, hidden=256):
        super(FC_LSTM, self).__init__()
        self.hidden = hidden
        self.fc_en1 = nn.Linear(4096, 1024)
        self.fc_en2 = nn.Linear(1024, hidden)
        self.en1 = nn.LSTMCell(hidden, hidden)
        self.en2 = nn.LSTMCell(hidden, hidden)
        self.en3 = nn.LSTMCell(hidden, hidden)
        self.de1 = nn.LSTMCell(hidden, hidden)
        self.de2 = nn.LSTMCell(hidden, hidden)
        self.de3 = nn.LSTMCell(hidden, hidden)
        self.fc_de1 = nn.Linear(hidden, 1024)
        self.fc_de2 = nn.Linear(1024, 4096)

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
            x_ = self.fc_de1(h_d3)
            x_ = F.relu(x_)
            x_ = self.fc_de2(x_)
            x_ = torch.tanh(x_)
            x_ = torch.reshape(x_, (batch_size, 64, 64))
            # print(x_.shape)
            outputs.append(x_)
        outputs = torch.stack(outputs)
        # outputs = torch.reshape(outputs, (seq_size, batch_size, 64, 64))

        return outputs


if __name__ == "__main__":
    model = FC_LSTM(256)
    x = torch.randn((10, 100, 64, 64))
    x = model(x)

    print(x.shape)