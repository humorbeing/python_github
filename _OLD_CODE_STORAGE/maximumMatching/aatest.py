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
'''
def transpo(S):
    newX = len(S[0])
    newY = len(S)
    tem = [[0 for j in range(newY)] for i in range(newX)]
    for i in range(newX):
        for j in range(newY):
            tem[i][j] = S[j][i]
    return tem
B = transpo(A)
def p(aa):
    print(np.array(aa))
    print('/*-'*15)

def waterone(D, p):
    tem = []
    for i in range(len(D)):
        tem.append(D[i]*p)
    return tem

def subtr(E,F):
    tem = []
    line = []
    for i in range(len(E)):
        for j in range(len(E[0])):
            line.append(E[i][j]-F[i][j])
        tem.append(line)
        line = []
    return tem

def water(M, S=[]):
    Water = []
    delta_water = []
    for i in range(len(M)):
        if sum(M[i]) == 0:
            Water.append([0 for i in range(len(M[i]))])
        else:
            if S:
                Water.append(waterone(M[i],(float(sum(S[i]))/sum(M[i]))))
            else:
                Water.append(waterone(M[i],(1000.0/sum(M[i]))))
    if S:
        delta_water = subtr(Water,S)
    else:
        delta_water = Water
    return Water,delta_water


thisiseven=True
tem = []
ma = []
x = []
y = []
for i in range(10):
    if thisiseven:
        thisiseven = False
        if i == 0:
            x,y = water(A)
        else:
            x,y = water(A,x)
        ma.append(x)
        tem.append(y)
    else:
        thisiseven = True
        x,y = water(B,transpo(x))
        x = transpo(x)
        y = transpo(y)
        ma.append(x)
        tem.append(y)
p(tem)
p(ma)
'''
