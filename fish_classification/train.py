from dataloader import loader, num_classes
from model import CNN
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

model = CNN(num_classes)
optimizer = optim.Adam(model.parameters())

EPOCH = 100

for e in range(EPOCH):
    losses = []
    for x, y in loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print('Epoch {}, loss {.4f}'.format(e, np.mean(losses)))

# torch.save(model, "Path")