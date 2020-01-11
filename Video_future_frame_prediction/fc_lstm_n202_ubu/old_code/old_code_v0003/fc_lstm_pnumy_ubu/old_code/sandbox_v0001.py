# 1-i-a from readme.md

import torch
import torch.nn as nn
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(4096, 4096, batch_first=True)
        self.decoder_lstm = nn.LSTM(4096, 4096, batch_first=True)


    def forward(self, x):
        # x is [batch,seq=20,64,64]
        x = x.view((x.shape[0],x.shape[1],-1))
        all_h, (last_h, last_c) = self.encoder_lstm(x)
        x = torch.zeros((100, 20, 4096))
        new_c = torch.zeros((1,100,4096))
        all_h, _ = self.decoder_lstm(x, (last_h, new_c))
        all_h = all_h.view((100,20,64,64))
        return all_h


model = FC_LSTM()
x = torch.randn((100,20,64,64))
x = model(x)

print(x.shape)