import numpy as np
A = np.array(([[i*10+j for j in range(16)] for i in range(16)]))
print(A)
def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

#swap_cols(A,1,5)

#print(A)
b = [0 for i in range(16)]
for i in range(4):
    for j in range(4):
        b[4*i+j] = i + j*4
print(b)
B = np.array(([[A[i][j] for j in b] for i in range(16)]))
print(B)
d = [0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
C = np.array(([[A[i][j] for j in d] for i in range(16)]))
print(d)
print(A)
#A[0][0]=99
print(A)
print(C)
