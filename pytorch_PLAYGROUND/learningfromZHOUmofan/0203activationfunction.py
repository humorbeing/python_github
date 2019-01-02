import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)
x_np = x.data.numpy()

y_reLu = F.relu(x)
y_sigmoid = F.sigmoid(x)
y_tanH = F.tanh(x)
y_softPlus = F.softplus(x)
y_leakyReLu = F.leaky_relu(x)
# y_other = F.

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))

plt.subplot(221)
plt.plot(x_np, y_reLu.data.numpy(), c='red', label='relu')
# plt.plot(x_np, x_np, c='red', label='x_np')
# plt.plot(x_np, y_leakyReLu.data.numpy(), c='red', label='leaky_relu')
plt.ylim((-1, 6))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid.data.numpy(), c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanH.data.numpy(), c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softPlus.data.numpy(), c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()


