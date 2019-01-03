import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_datasets = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = Data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
print(train_datasets.train_data.size())
# print(train_datasets.train_data[0])
test_datasets = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_loader = Data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)
print(test_datasets.test_data.size())
# print(test_datasets.train_data.size())
# print(test_datasets.test_data[0])
print(test_datasets.test_labels.size())

# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]
# test_y = test_data.test_labels[:2000]
# test_x, test_y = test_x.cuda(), test_y.cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(20 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
correct_sum = 0
for step, (x, y) in enumerate(test_loader):
    t_x = Variable(x).cuda()
    print(t_x)
    t_y = Variable(y).cuda()
    output = cnn(t_x)
    pred_y = torch.max(output, 1)[1].data.squeeze()
    correct_sum += sum(pred_y == t_y)
print(correct_sum)
# cnn.cpu()
# test_output = cnn(test_x[500:510])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[500:510].numpy(), 'real number')
# if step % 50 == 0:
#     test_output = cnn(test_x)
#     pred_y = torch.max(test_output, 1)[1].data.squeeze()
#     accuracy = sum(pred_y == test_y) / test_y.size(0)
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
