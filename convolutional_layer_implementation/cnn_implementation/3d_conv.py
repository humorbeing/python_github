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


def Z(K, V, i, j, k, stride=1):
    k_shape = K.shape
    # kernel_out_channel = k_shape[0]
    kernel_channel = k_shape[1]
    kernel_row = k_shape[2]
    kernel_column = k_shape[3]
    z = 0
    for l in range(kernel_channel):
        for m in range(kernel_row):
            for n in range(kernel_column):
                z += V[l][j*stride+m][k*stride+n] * K[i][l][m][n]
    return z


for output_channel in range(5):
    for output_row in range(out_len):
        for output_col in range(out_len):
            out[output_channel][output_row][output_col] = \
                Z(K, V, output_channel, output_row, output_col)



print(out)


def delta_K(G, V, K, i, j, k, l, stride=1):
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]
    output = 0
    for m in range(kernel_row):
        for n in range(kernel_col):
            output += G[i][m][n] * V[j][m*stride+k][n*stride+l]
    return output


def delta_V(K, G, i, j, k, stride=1):
    kernel_output_channel = K.shape[0]
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]
    l_range = G.shape[1]
    n_range = G.shape[2]
    output = 0
    for q in range(kernel_output_channel):
        for m in range(kernel_row):
            l = j - m
            if 0 <= l < l_range:
                for p in range(kernel_col):
                    n = k - p
                    if 0 <= n < n_range:
                        output += K[q][i][m][p] * G[q][l][n]
    return output


d_K = delta_K(out, V, K, 0, 0, 0, 0)
print(d_K)
d_K = delta_V(K, out, 0, 0, 0)
print(d_K)