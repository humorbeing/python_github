# 1-i-a from readme.md
# the first model
# encoder decoder, only 1 layer, flatten style
# v0001 +
# renew with seq first lstm
import torch
import torch.nn as nn
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(4096, 4096)
        self.decoder_lstm = nn.LSTM(4096, 4096)

    def forward(self, x):
        device = next(self.parameters()).device
        seq_size = x.shape[0]
        batch_size = x.shape[1]

        x = x.view((seq_size,batch_size,-1))
        all_h, (last_h, last_c) = self.encoder_lstm(x)
        x = torch.zeros((seq_size,batch_size, 4096)).to(device)
        new_c = torch.zeros((1,batch_size,4096)).to(device)
        all_h, _ = self.decoder_lstm(x, (last_h, new_c))
        all_h = all_h.view((seq_size,batch_size,64,64))
        return all_h


if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((20, 100, 64, 64))
    x = model(x)

    print(x.shape)