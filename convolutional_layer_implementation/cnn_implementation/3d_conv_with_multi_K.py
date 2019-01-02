import numpy as np


in1 = [i for i in range(18)]
V = np.array(in1).reshape([2, 3, 3])
print(len(V))
print(len(V[0]))
print(len(V[0][0]))

# in2 = np.random.random([4, 4])
# K = np.random.random([3, 3])
filt = [i for i in range(40)]
K = np.array(filt).reshape([5, 2, 2, 2])
# print(in1)*---
# print(in2)
# print(filt)
stride = 1
# in_len = len(V)
# fi_len = len(K)
out_len = (3 - 2) / stride + 1
out_len = int(out_len)
out = np.zeros([5, out_len, out_len])


def Z(K, V, s, i, j, k):
    kernel_channel = 2
    kernel_row = 2
    kernel_column = 2
    z = 0
    for l in range(kernel_channel):
        for m in range(kernel_row):
            for n in range(kernel_column):
                z += V[l][j*s+m][k*s+n] * K[i][l][m][n]
    return z


for output_channel in range(5):
    for output_row in range(out_len):
        for output_col in range(out_len):
            out[output_channel][output_row][output_col] = \
                Z(K, V, stride, output_channel, output_row,output_col)



print(out)
