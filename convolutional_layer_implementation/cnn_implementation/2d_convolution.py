import numpy as np


in1 = [i for i in range(9)]
I = np.array(in1).reshape([3, 3])
# in2 = np.random.random([4, 4])
# K = np.random.random([3, 3])
filt = [i for i in range(4)]
K = np.array(filt).reshape([2, 2])
# print(in1)
# print(in2)
# print(filt)
stride = 1
in_len = len(I)
fi_len = len(K)
out_len = (in_len - fi_len) / stride + 1
out_len = int(out_len)
out = np.zeros([out_len, out_len])


def S(I_in, K_in, i, j):
    k_row = len(K_in)
    k_col = len(K_in[0])
    s = 0
    for m in range(k_row):
        for n in range(k_col):
            s += I_in[i+m][j+n] * K_in[m][n]
    return s


for row in range(out_len):
    for col in range(out_len):
        out[row][col] = S(I, K, row, col)



print(out)
