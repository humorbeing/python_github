import numpy as np



i = np.identity(5)
print(i.shape)
print(i)
print(i[0])

data = np.identity(5)
x = data[:-1]
y = data[1:]

for i in range(4):
    print('--')
    print(x[i])
    print(y[i])


import torch.nn as nn
import torch

class ENCODER(nn.Module):
    def __init__(self):
        super(ENCODER, self).__init__()
        # self.kl_tolerance = 0.5

        self.lstm = nn.LSTM(5, 5, batch_first=True)
        self.lstm = nn.LSTMCell(5, 5)

    def forward(self, x, hx, cx):
        x = x.astype(np.float32)
        # state = state / 255.0
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        print(x.shape)
        hx, cx, = self.lstm(x, (hx, cx))
        # hx, cx = self.lstm(x)
        return hx, cx


mo = ENCODER()
hx = torch.zeros(1, 1, 5)
cx = torch.zeros(1, 1, 5)
# print('hi')
hx, cx = mo(x, hx, cx)
print(hx)


