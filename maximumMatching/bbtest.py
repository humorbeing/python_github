import numpy as np
with open('matching.inp','r') as f:
    a = f.readlines()
def arr(s):
    tem = []
    b = s.split(' ')
    for i in b:
        if i != ' ' and i != '\n':
            tem.append(int(i))
    return tem
N = arr(a[0])[0]
M = arr(a[0])[1]
G = []
for i in range(N):
    tem = []
    A = arr(a[i+1])
    for j in range(A[1]):
        tem.append(A[j+2]-1)
    G.append(tem)
A = [[0 for j in range(M)] for i in range(N)]
for i in range(N):
    for j in range(len(G[i])):
        A[i][G[i][j]] = 1
A = np.array(A)
'''
def subtract(E,F):
    tem = []
    line = []
    for i in range(len(E)):
        for j in range(len(E[0])):
            line.append(E[i][j]-F[i][j])
        tem.append(line)
        line = []
    return tem
'''
def waterforward(S):
    Water = []
    delta_water = []
    for i in range(N):
        if sum(A[i]) == 0:
            Water.append([0 for j in range(M)])
        else:
            if len(S) != 0:
                Water.append(A[i]*[float(sum(S[i]))/sum(A[i]) for j in range(M)])
            else:
                Water.append(A[i]*[1000.0/sum(A[i]) for j in range(M)])
    if len(S) != 0:
        delta_water = np.subtract(Water,S)
    else:
        delta_water = Water
    return np.array(Water),np.array(delta_water)

def waterbackward(S):
    Water = []
    delta_water = []
    columnwatersum = [0 for i in range(M)]
    columnnumbersum = [0 for i in range(M)]
    B = A.T
    for i in range(N):
        for j in range(M):
            columnwatersum[j] += S[i][j]
            columnnumbersum[j] += A[i][j]
    for i in range(M):
        if columnnumbersum[i] == 0:
            Water.append([0 for j in range(N)])
        else:
            Water.append(B[i]*[float(columnwatersum[i])/columnnumbersum[i] for j in range(N)])
    Water = np.array(Water)
    delta_water = np.subtract(Water.T,S)

    return np.array(Water.T),np.array(delta_water)

x = []
for i in range(100):
    x,_ = waterforward(x)
    x,_ = waterbackward(x)
print(np.around(x, decimals=1))
