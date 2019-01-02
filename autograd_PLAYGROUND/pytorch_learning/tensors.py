import torch
# a = torch.FloatTensor(5, 7)
a = torch.randn(5, 7)
# print(a)
size_ = a.size()
print(size_)
print(size_[0])
print(type(size_))
print(type(size_[0]))
a.fill_(3.5)
print(a)
b = a.add_(2)
print(a)
print(b)

b = a[0, 3]
print(b)
b = a[:, 3:5]
print(b)

x = torch.ones(5, 5)
print(x)

z = torch.Tensor(5, 2)
z[:, 0] = 10
z[:, 1] = 100
print(z)

x.index_add_(1, torch.LongTensor([4, 0]), z)
print(x)

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 5, out=a)
print(a)
print(b)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    b = a.cpu()
    print(type(a))
    print(type(b))