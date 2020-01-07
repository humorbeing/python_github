
# encoder decoder, only 1 layer, flatten style
# lstmcell
# v0004
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        self.encoder_lstmcell = nn.LSTMCell(4096, 4096)
        self.decoder_lstmcell = nn.LSTMCell(4096, 4096)


    def forward(self, x, future_step=10):
        # x in is [seq=10, batch, 64, 64]
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]
        h_e = torch.zeros((batch_size, 4096)).to(device)
        c_e = torch.zeros((batch_size, 4096)).to(device)
        x = x.reshape((seq_size, batch_size, -1))
        # print(x.shape)
        for seq in range(seq_size):
            h_e, c_e = self.encoder_lstmcell(x[seq], (h_e, c_e))
        # print(h_e.shape)
        h_d = h_e
        c_d = torch.zeros((batch_size, 4096)).to(device)

        zero_input = torch.zeros((batch_size, 4096)).to(device)
        outputs = []
        for seq in range(future_step):
            h_d, c_d = self.decoder_lstmcell(zero_input, (h_d, c_d))
            x_ = torch.tanh(h_d)
            outputs.append(x_)
        outputs = torch.stack(outputs)
        outputs = torch.reshape(outputs, (seq_size, batch_size, 64, 64))

        return outputs


if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((10, 100, 64, 64))
    x = model(x)

    print(x.shape)