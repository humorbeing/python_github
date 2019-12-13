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
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


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
z_dim = 100
mnist_dim = 28 * 28
G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


DL = []
GL = []
show_size = 10
for epoch in range(n_epoch):
    print(epoch)
    D_losses = []
    G_losses = []
    for batch_idx, (x, _) in enumerate(train_loader):
        z = torch.randn(batch_size, z_dim).to(device)
        x_fake = G(z).to(device)
        x_real = x.view(-1, mnist_dim).to(device)
        D_loss = torch.mean(torch.log(D(x_real)) + torch.log(1-D(x_fake)))
        D_loss = D_loss * (-1)
        # D.zero_grad()
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())

        z = torch.randn(batch_size, z_dim).to(device)
        x_fake = G(z).to(device)
        G_loss = torch.mean(torch.log(D(x_fake)))
        G_loss = G_loss * (-1)
        # G.zero_grad()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())
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

# plt.plot(range(n_epoch), D_losses)
plt.plot(range(len(DL)), DL, 'r')
plt.plot(range(len(GL)), GL, 'b')
plt.legend(('Discriminator', 'Generator'),
           loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.show()