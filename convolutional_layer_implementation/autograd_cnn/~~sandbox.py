import torch
from torch.autograd import Variable
import numpy as np

# a = [i for i in range(12)]
# a = np.array(a)
# print(a.shape)
# b = torch.from_numpy(a)
# print(b)
# a = a.reshape(2, 3, 2)
#
# b = b.view(2, 3, 2)
# print(b)
# g = V(torch.ones(5, 5, 5))
# v = V(torch.zeros(5, 5, 5))
# print(b[0][1][1])
# print(a[0][1][1])
# print(b[0, 1, 1])
# print(a[0, 1, 1])
# print(v[0][1][1])
# print(type(g))
# # g = V(torch.from_numpy(np.array([5])))
# # g[0][1][1] = v[0][1][0]
# n = torch.ones(7, 1)
# m = torch.zeros(1, 5)
# m = m + 2
# print(m)
# print(n)
# print(torch.matmul(n, m))
a = [i-7 for i in range(12)]
a = torch.Tensor(a)
a = a.view(3, 4)
# a = torch.max(a)
a = Variable(a)
b = Variable(torch.zeros(3, 4))
print(a)
for i in range(3):
    for j in range(4):
        if (a[i, j]<0).data.numpy():
            b[i, j] = 0
a = torch.mul(a, b)

print(a)