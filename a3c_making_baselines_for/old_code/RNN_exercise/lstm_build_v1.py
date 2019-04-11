import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
h = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
hidden = h

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    # print(out.shape)
    # print(hidden[0].shape)
    # print(hidden[1].shape)
    print(out)
    # print(hidden[0])
    # print('1')
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
hidden = h
out, hidden = lstm(inputs, hidden)
print('1'*20)
print(out)
print(hidden)