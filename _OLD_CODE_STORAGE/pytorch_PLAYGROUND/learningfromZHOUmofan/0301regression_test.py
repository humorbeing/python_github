import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer1
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer1
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer1

        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, xx):
        xx = F.relu(self.hidden1(xx))  # activation function, relu
        xx = F.relu(self.hidden2(xx))  # activation function, relu
        xx = F.relu(self.hidden3(xx))  # activation function, relu

        xx = self.predict(xx)  # linear output
        return xx

net = Net(n_feature=1, n_hidden=100, n_output=1)  # define network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.02, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()  # this is for regression mean square loss

plt.ion()
plt.show()

for t in range(5000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()        # back propagation, compute gradients
    optimizer.step()       # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
