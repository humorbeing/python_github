import numpy as np
'''
sudoku = [
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],

           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],

           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
          ]
'''
sudoku = [
           [0,2,0, 0,0,8, 0,0,0],
           [0,3,0, 4,2,0, 0,1,0],
           [4,0,0, 0,0,0, 5,3,0],

           [6,0,8, 0,0,5, 0,7,0],
           [7,9,0, 0,0,0, 2,8,4],
           [2,0,0, 0,7,4, 0,0,0],

           [8,0,0, 3,5,6, 0,2,0],
           [0,5,0, 0,4,0, 3,0,0],
           [0,0,4, 0,0,0, 0,0,7],
          ]
print(np.array(sudoku))
A = np.array([[1 for i in range(81)] for j in range(81)])
#print(A)
for i in range(9):
    for j in range(9):
        if sudoku[i][j] != 0:
            for k in range(1,10):
                if sudoku[i][j] == k:
                    for l in range(81):
                        A[l][9*i+j] = 0
                    for l in range( (k-1)*9,(k*9) ):
                        if sum(A[l]) != 999:
                            for m in range(81):
                                A[l][m] = 0
                            A[l][9*i + j] = 999
                            break
print(np.array(A))
#print([i for i in range(9,18)])
row = [i for i in range(81)]
column = [0 for i in range(81)]
for i in range(9):
    for j in range(9):
        column[9*i+j] = i + j*9
firstblock = [0,1,2,9,10,11,18,19,20]
block = []
for i in range(3):
    for j in range(3):
        for k in firstblock:
            block.append(k+3*j+27*i)
print(block)
##print(column)
