from to_import import ROOT_PATH
from to_import import vae_dataset_loader
from to_import import vae_loss
from models import VAE

import torch
from torch import optim
from torch.autograd import Variable

EPOCH = 200
Learning_rate = 1e-4

dataset_path = '/media/ray/SSD/workspace/python/dataset/save_here/carracing/v2/VAE_trainset/rollout_v0_0.npz'
dataloader = vae_dataset_loader(dataset_path, batch_size=512)

dst = ROOT_PATH + 'model/'
vae_save_name = 'vae_model.save'
is_new_model = False
if is_new_model:
    V = VAE()
    V = V.cuda()
    V.train()
else:
    V = torch.load(dst + vae_save_name)
    V = V.cuda()
    V.train()

optimizer = optim.Adam(V.parameters(), lr=Learning_rate)

for e in range(EPOCH):
    for batch_idx, state in enumerate(dataloader):

        x = Variable(state.cuda())
        optimizer.zero_grad()
        output, mu, logvar, z = V(x)
        # loss = F.mse_loss(output, x)
        # loss = F.binary_cross_entropy_with_logits(output, x)
        loss = vae_loss(x, output, mu, logvar)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(state), len(dataloader.dataset),
                   100. * batch_idx / len(dataloader),
                # loss.data[0]
                loss.item()
            ))
torch.save(V, dst + vae_save_name)