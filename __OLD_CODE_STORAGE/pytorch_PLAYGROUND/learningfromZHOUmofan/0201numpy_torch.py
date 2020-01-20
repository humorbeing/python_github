import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    'numpy\n', np_data,
    '\ntorch\n', torch_data,
    '\ntensor2array\n', tensor2array,
)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32bit
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),
    # 'torch: ', torch.abs(data),  #data is list, torch need tensor
    '\ntorch: ', torch.abs(tensor),
)
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),
    '\ntorch: ', torch.sin(tensor),
)
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor),
)
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nnumpy:\n', np.matmul(data, data),
    '\nnumpy:\n', np.dot(data, data),
    '\nnumpy:\n', np.array(data).dot(data),
    '\ntorch:\n', torch.mm(tensor, tensor),
    '\ntorch:\n', torch.dot(tensor, tensor),
)
