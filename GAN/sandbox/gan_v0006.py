import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Encoder(nn.Module):
    def __init__(self,n_node=128):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(784, n_node*2)
        self.lin2 = nn.Linear(n_node*2, n_node)
        self.lin3gauss = nn.Linear(n_node, 100)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

class Decoder(nn.Module):
    def __init__(self,n_node=128):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(100, n_node)
        self.lin2 = nn.Linear(n_node, n_node)
        self.lin3 = nn.Linear(n_node, 784)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = self.lin3(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, n_node=128):
        super(Discriminator, self).__init__()

        self.lin1 = nn.Linear(110, n_node)
        self.lin2 = nn.Linear(n_node, n_node)
        self.lin3 = nn.Linear(n_node, 1)

    def forward(self, x, z):
        x = torch.cat((x, z), -1)
        # print(x.shape)
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))


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

EPS = 1e-15
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# hyper parameter
batch_size = 100
# batch_size = 2
n_epoch = 72
lr = 0.0002

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

torch.manual_seed(6)
# build network
# z_dim = 100
encoder = Encoder()
decoder = Decoder()
D = Discriminator()

encoder.to(device)
decoder.to(device)
D.to(device)

# Set learning rates
gen_lr = 0.0001
reg_lr = 0.00005

#encode/decode optimizers
optim_decoder = torch.optim.Adam(decoder.parameters(), lr=gen_lr)
optim_encoder_w_de = torch.optim.Adam(encoder.parameters(), lr=gen_lr)
#regularizing optimizers
optim_encoder_w_D = torch.optim.Adam(encoder.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D.parameters(), lr=reg_lr)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]
DL = []
GL = []
RL = []
show_size = 10
for epoch in range(n_epoch):
    print('epoch:', epoch)
    D_losses = []
    G_losses = []
    Recon_losses = []
    for x, y in train_loader:
        x = x.to(device)
        # print(y)

        # label = F.one_hot(y, num_class=10)
        # print(label)
        x = x.view(batch_size, -1)
        z_real = encoder(x)
        x_fake = decoder(z_real)
        recon_loss = F.mse_loss(x_fake+EPS, x+EPS)

        encoder.zero_grad()
        decoder.zero_grad()
        recon_loss.backward()
        optim_decoder.step()
        optim_encoder_w_de.step()
        Recon_losses.append(recon_loss.item())

        # Discriminator
        encoder.eval()
        z_sample = torch.randn(batch_size, 100) * 5.
        z_sample = z_sample.to(device)
        x_label = one_hot_embedding(y, num_classes=10)
        # print(label)
        x_label = x_label.to(device)
        # print(y)
        sample_label = torch.randint(0, 10, (batch_size,))
        z_label = one_hot_embedding(sample_label, num_classes=10)
        z_label = z_label.to(device)

        D_pos = D(z_sample, z_label)

        z_real = encoder(x)
        D_nag = D(z_real, x_label)

        D_loss = -torch.mean(torch.log(D_pos+EPS) + torch.log(1 - D_nag+EPS))

        D_loss.backward()
        optim_D.step()
        D_losses.append(D_loss.item())
        # Generator
        encoder.train()
        z_real = encoder(x)
        D_nag = D(z_real, x_label)

        G_loss = -torch.mean(torch.log(D_nag+EPS))

        G_loss.backward()
        optim_encoder_w_D.step()
        G_losses.append(G_loss.item())


        # break
    DL.append(np.mean(D_losses))
    GL.append(np.mean(G_losses))
    RL.append(np.mean(Recon_losses))

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
            axs[n].set_title(str(y[n].item()))

            axs[n].axis('off')
        axs[0].set_title('Epoch:' + str(epoch))
        fig.set_size_inches(np.array(fig.get_size_inches()) * show_size* 0.25)
        plt.show()

    # break
plt.plot(range(len(DL)), DL, 'r')
plt.plot(range(len(GL)), GL, 'b')
plt.plot(range(len(RL)), RL, 'g')
plt.legend(('Discriminator', 'Generator', 'Reconst'),
           loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.show()
