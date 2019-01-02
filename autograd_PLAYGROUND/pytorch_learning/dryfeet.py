import torch
from torch.autograd import Variable

x = Variable(
    torch.ones(2, 2),
    requires_grad=True
)

print(x)
print(x.data)
print(x.grad)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

out.backward()
print(x.grad)
print(y.grad)
print(z.grad)
print(out.grad)

# aha, I defined x as require_grad=True.
print(x)
x = torch.ones(12)
x = x.view(-1, 2, 1, 2)
print(x)
x = x.view(6, 2)
print(x)
y = x.numpy()
print(y.reshape(-1, 4))

x = Variable(torch.Tensor([1., 2., 3.]), requires_grad=False)
y = Variable(torch.Tensor([4., 5., 6.]), requires_grad=True)
z = x + y
print(torch.from_numpy(z.data.numpy()))
print(z.grad_fn)
s = z.sum()
print(s)
print(s.grad_fn)
import numpy as np
x = Variable(torch.Tensor(np.array([
    0., 0.5, 1., 1.5, 2.
]) * np.pi),
             requires_grad=True)
out = torch.sin(x)
# print(x.grad)
out.backward(torch.Tensor([0, 1, -1, 0, 0.]))
print(out)
print(x.grad)
print(torch.cos(x))

