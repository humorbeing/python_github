import torch
import torch.nn as nn
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.len_seq = 20
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, 4096, 3, stride=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(4096, 2048, batch_first=True)

    def forward(self, x):
        y = self.cnn(x)
        y = y.view(-1, self.len_seq, 4096)
        y = self.lstm(y)
        return y

m = FC_LSTM()





x = torch.randn((100,1,64,64))
print(x.shape)
y, (h, c) = m(x)
print(y.shape)
print(h.shape)
a = y[:,-1,:]
print(a.shape)
b = h.view(5,2048)
# b = b +1
print(b.shape)
print((a==b).all())
c = np.array([1,2]) == np.array([1,4])

print(c.any())
print(c.all())
