"""
checking pytorch installation and GPU computing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class test_model(torch.nn.Module):
    """
    GPU testing model
    """
    def __init__(self, num_layers=3, num_inputs=10, num_outputs=10, num_mid_nodes=10, is_cuda=True):
        super(test_model, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            if is_cuda:
                layer = nn.Linear(in_features=num_mid_nodes, out_features=num_mid_nodes).cuda()
            else:
                layer = nn.Linear(in_features=num_mid_nodes, out_features=num_mid_nodes)
            self.layers.append(layer)
        self.start_layer = nn.Linear(in_features=num_inputs, out_features=num_mid_nodes)
        self.last_layer = nn.Linear(in_features=num_mid_nodes, out_features=num_outputs)

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        x = self.start_layer(inputs)
        x = F.relu(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.last_layer(x)
        return x


is_cuda = False
batch = 100
input_size = 100000
output_size = 10
x = np.random.randn(batch, input_size)
y = np.random.randn(batch, output_size)

x = torch.from_numpy(x)
x = x.type(torch.float)
y = torch.from_numpy(y)
y = y.type(torch.float)


m = test_model(
    num_layers=100,
    num_inputs=input_size,
    num_outputs=output_size,
    num_mid_nodes=1000,
    is_cuda=is_cuda)

if is_cuda:
    x = x.cuda()
    y = y.cuda()
    m = m.cuda()
optimizer = optim.Adam(m.parameters())

while True:
    y_hat = m(x)
    loss = F.mse_loss(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


