# 1-i-a from readme.md

import torch
import torch.nn as nn
import numpy as np

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        self.num_lstm_layers = 1
        self.encoder_lstm = nn.LSTM(
            4096, 4096,
            num_layers=self.num_lstm_layers,
            batch_first=True)
        self.decoder_lstm = nn.LSTM(
            4096, 4096,
            num_layers=self.num_lstm_layers,
            batch_first=True)
        self.predictor_lstm = nn.LSTM(
            4096, 4096,
            num_layers=self.num_lstm_layers,
            batch_first=True)

    def forward(self, x, future_step=10):
        self.device = next(self.parameters()).device
        batch_size = x.shape[0]
        seq_size = x.shape[1]
        x = x.view((batch_size,seq_size,-1))
        all_h, (last_h, last_c) = self.encoder_lstm(x)
        x = torch.zeros((batch_size, seq_size, 4096)).to(self.device)
        new_c = torch.zeros((self.num_lstm_layers, batch_size, 4096)).to(self.device)
        dec, _ = self.decoder_lstm(x, (last_h, new_c))
        dec = dec.view((batch_size,seq_size,64,64))
        pred, _ = self.predictor_lstm(x, (last_h, new_c))
        pred = pred.view((batch_size, seq_size, 64, 64))
        return dec, pred

if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((100,20,64,64))
    x1, x2 = model(x)

    print(x1.shape)