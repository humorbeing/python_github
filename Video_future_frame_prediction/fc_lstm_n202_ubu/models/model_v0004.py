# v0004
# plan: created image become input.

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
            nn.Conv2d(256, 256, 3, stride=1),
            nn.ReLU(),
        )
        self.recon_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2),
        )
        self.pred_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2),
        )
        self.num_lstm_layers = 3
        self.encoder_lstm = nn.LSTM(
            256, 256,
            num_layers=self.num_lstm_layers,
            batch_first=True)
        self.decoder_lstm = nn.LSTM(
            256, 256,
            num_layers=self.num_lstm_layers,
            batch_first=True)
        self.predictor_lstm = nn.LSTM(
            256, 256,
            num_layers=self.num_lstm_layers,
            batch_first=True)

    def forward(self, x, future_step=10):
        self.device = next(self.parameters()).device
        batch_size = x.shape[0]
        seq_size = x.shape[1]
        x = x.reshape((batch_size * seq_size,1,64,64))
        x = self.cnn(x)
        x = x.reshape((batch_size, seq_size, -1))
        all_h, (last_h, last_c) = self.encoder_lstm(x)

        x = torch.zeros((batch_size, seq_size, 256)).to(self.device)
        new_c = torch.zeros((self.num_lstm_layers, batch_size, 256)).to(self.device)
        dec, _ = self.decoder_lstm(x, (last_h, new_c))

        dec = dec.reshape((batch_size * seq_size, 256, 1, 1))

        dec = self.recon_convtranspose(dec)

        dec = dec.reshape((batch_size,seq_size,64,64))

        pred_x = torch.zeros((batch_size, future_step, 256)).to(self.device)
        pred, _ = self.predictor_lstm(pred_x, (last_h, new_c))
        pred = pred.reshape((batch_size * seq_size, 256, 1, 1))

        pred = self.pred_convtranspose(pred)

        pred = pred.reshape((batch_size, seq_size, 64, 64))
        return dec, pred

if __name__ == "__main__":
    model = FC_LSTM()
    x = torch.randn((100,10,64,64))
    x1, x2 = model(x)


    print(x1.shape)
    print(x2.shape)