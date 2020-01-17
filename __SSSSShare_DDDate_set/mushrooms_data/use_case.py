import numpy as np
mushroomsfile = 'mushrooms.csv'
data_array = np.genfromtxt(
    mushroomsfile, delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]

print(data_array.shape)
print(X.shape)
print(Y.shape)
print('-'*20)
print(data_array[0])
print(X[0])
print(Y[0])
print('`.'*20)
print(data_array[1])
print(X[1])
print(Y[1])
