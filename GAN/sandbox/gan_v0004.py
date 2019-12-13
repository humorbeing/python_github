import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self,n_node=128):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(110, n_node),
            nn.LeakyReLU(0.2),
            nn.Linear(n_node, n_node*2),
            nn.BatchNorm1d(n_node*2, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(n_node*2, n_node*4),
            nn.BatchNorm1d(n_node*4, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(n_node*4, n_node*8),
            nn.BatchNorm1d(n_node*8, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(n_node*8, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # print(labels)
        # a = self.label_emb(labels)
        # print(a.shape)
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        # print(noise.shape)
        # print(gen_input.shape)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear((10 + 784), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# hyper parameter
# batch_size = 100
batch_size = 100
n_epoch = 62
lr = 0.0002

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

torch.manual_seed(6)
# build network
# z_dim = 100
G = Generator(32)
D = Discriminator()

# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
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
    print('epoch:', epoch)
    D_losses = []
    G_losses = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        z = torch.randn((batch_size, 100))
        z = z.to(device)
        fake_label = torch.randint(0, 10, (batch_size,)).to(device)
        x_fake = G(z, fake_label)
        G_loss = torch.mean(torch.log(D(x_fake, fake_label)))
        G_loss = G_loss * (-1)
        # G.zero_grad()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())

        z = torch.randn((batch_size, 100))
        z = z.to(device)
        fake_label = torch.randint(0, 10, (batch_size,)).to(device)
        # print(fake_label.shape)
        # print(z.shape)
        x_fake = G(z, fake_label)
        D_loss = torch.mean(torch.log(D(x, y)) + torch.log(1-D(x_fake, fake_label)))
        D_loss = D_loss * (-1)
#         # D.zero_grad()
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())
#
#         z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
#         z = z.to(device)
#         x_fake = G(z)
#         G_loss = torch.mean(torch.log(D(x_fake)))
#         G_loss = G_loss * (-1)
#         # G.zero_grad()
#         G_optimizer.zero_grad()
#         G_loss.backward()
#         G_optimizer.step()
#         G_losses.append(G_loss.item())


    DL.append(np.mean(D_losses))
    GL.append(np.mean(G_losses))
#
#
    if epoch % 10 == 0:
        if batch_size < show_size:
            show_size = batch_size
        imgs = x_fake[:show_size]
        imgs = imgs.cpu().detach().numpy()
        imgs = np.reshape(imgs, (-1, 28, 28))
        plt.gray()
        fig, axs = plt.subplots(1, show_size,
                        gridspec_kw={'hspace': 0, 'wspace': 0})

        for n in range(show_size):
            axs[n].imshow(imgs[n])
            axs[n].set_title(str(fake_label[n].item()))

            axs[n].axis('off')
        axs[0].set_title('Epoch:' + str(epoch))
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
