import numpy as np
sudoku = np.array([
          [0,0,4,3],
          [3,0,2,0],
          [0,3,0,2],
          [0,2,0,4],
          ])

print(sudoku)
A = np.array([[1 for i in range(16)] for j in range(16)])
#print(A)
for i in range(4):
    for j in range(4):
        if sudoku[i][j] != 0:
            for k in range(1,5):
                if sudoku[i][j] == k:
                    for l in range(16):
                        A[l][4*i+j] = 0
                    for l in range( (k-1)*4,(k*4-1) ):
                        if sum(A[l]) != 999:
                            for m in range(16):
                                A[l][m] = 0
                            A[l][4*i + j] = 999
                            break
a = [i for i in range(16)]
b = [0 for i in range(16)]
for i in range(4):
    for j in range(4):
        b[4*i+j] = i + j*4
c = [0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
#print(b)
#print(A)
B = np.array(([[A[i][j] for j in b] for i in range(16)]))
C = np.array(([[A[i][j] for j in c] for i in range(16)]))
print('A'*79)
print(A)
print('B'*79)
print(B)
print('C'*79)
print(C)
