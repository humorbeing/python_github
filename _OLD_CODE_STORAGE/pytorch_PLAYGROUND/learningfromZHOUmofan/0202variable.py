import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

print('tensor: \n', tensor)
print('\nvariable:\n', variable)

t_out = torch.mean(tensor*tensor)  # x**2
v_out = torch.mean(variable*variable)  # x**2

print('t_out: ', t_out)
print('v_out: \n', v_out)

v_out.backward()
# v_out = 1/4 * sum (var*var) = 1/4 * sum (var**2)
# d(v_out)/d(var) = 1/4 * var *2
print(variable.grad)
print(variable)
print(variable.data)
print(variable.data.numpy())



