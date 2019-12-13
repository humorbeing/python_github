import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# MNIST Dataset
transform = transforms.Compose([
    # HxWxC -> CxHxW, [28x28] -> [1x28x28], [0,255] -> [0,1]
    transforms.ToTensor(),
    # mean=(0.5,0.5,0.5) and std. It is for 3 channels
    #mnist only has one channel
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
path = '../../__SSSSShare_DDDate_set/mnist_data/'
train_dataset = datasets.MNIST(root=path, train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root=path, train=False,
                              transform=transform,
                              download=False)
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# Formula: O=(I-1)xS - 2P + K
# -----------------------
# l1: I=1, O=3, K=3, S=1
# l2: I=3, O=5, K=3, S=1
# l3: I=5, O=12, K=4, S=2
# l4: I=12, O=26, K=4, S=2
# l5: I=26, O=28, K=3, S=1
# -----------------------
# l1: I=1, O=3, K=3, S=2
# l2: I=3, O=7, K=3, S=2
# l3: I=7, O=13,K=3, S=2, P=1
# l4: I=13,O=28,K=4, S=2
# ------------------------
# l1: I=1, O=4, K=4, S=2
# l2: I=4, O=9, K=3, S=2
# l3: I=9, O=28,K=4, S=3
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 3, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 3, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 3, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 3, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 3, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        x = x.squeeze()
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# hyper parameter
batch_size = 100
n_epoch = 200
lr = 0.0002

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# build network
# z_dim = 100
G = generator(64)
D = discriminator(64)

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.to(device)
D.to(device)

# optimizer
# G_optimizer = optim.Adam(G.parameters(), lr = lr)
# D_optimizer = optim.Adam(D.parameters(), lr = lr)
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

DL = []
GL = []
show_size = 10
for epoch in range(n_epoch):
    D_losses = []
    G_losses = []
    for x, _ in train_loader:
        x = x.to(device)
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        z = z.to(device)
        x_fake = G(z)
        D_loss = torch.mean(torch.log(D(x)) + torch.log(1-D(x_fake)))
        D_loss = D_loss * (-1)
        # D.zero_grad()
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())

        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        z = z.to(device)
        x_fake = G(z)
        G_loss = torch.mean(torch.log(D(x_fake)))
        G_loss = G_loss * (-1)
        # G.zero_grad()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())

        # break
    DL.append(np.mean(D_losses))
    GL.append(np.mean(G_losses))


    if epoch % 10 == 0:
        if batch_size < show_size:
            show_size = batch_size
        imgs = x_fake[:show_size]
        imgs = imgs.cpu().detach().numpy()
        imgs = np.reshape(imgs, (-1, 28, 28))
        plt.gray()
        fig, axs = plt.subplots(1, show_size,
                        gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0].set_title('Epoch:'+ str(epoch))
        for n in range(show_size):
            axs[n].imshow(imgs[n])
            axs[n].axis('off')
        fig.set_size_inches(np.array(fig.get_size_inches()) * show_size* 0.25)
        plt.show()

    # break
# plt.plot(range(n_epoch), D_losses)
plt.plot(range(len(DL)), DL, 'r')
plt.plot(range(len(GL)), GL, 'b')
plt.legend(('Discriminator', 'Generator'),
           loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.show()
