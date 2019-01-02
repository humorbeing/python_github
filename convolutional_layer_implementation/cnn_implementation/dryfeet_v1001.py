import numpy as np


in1 = [i for i in range(216)]
V = np.array(in1).reshape([3, 6, 12])
in_shape = V.shape
print(in_shape)
in_channel = in_shape[0]
in_row = in_shape[1]
in_column = in_shape[2]
pool_row = 2
pool_column = 3
out_channel = in_channel
out_row = int(in_row / pool_row)
out_column = int(in_column / pool_column)
out_shape = (out_channel, out_row, out_column)
print(out_shape)
output = np.zeros(out_shape)
choose_tensor = np.zeros(in_shape)
for out_c in range(out_channel):
    for out_r in range(out_row):
        for out_col in range(out_column):
            pass
