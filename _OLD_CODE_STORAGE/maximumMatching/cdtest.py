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
Lookingfor = 0
MaxMatching = 0
G = []
edgesum = 0
for i in range(N):
    tem = []
    A = arr(a[i+1])
    edgesum += A[1]
    for j in range(A[1]):
        tem.append(A[j+2]-1)
    G.append(tem)
A = [[0 for j in range(M)] for i in range(N)]
for i in range(N):
    for j in range(len(G[i])):
        A[i][G[i][j]] = 1
A = np.array(A)
nn = N
gotone = 0
for i in range(nn):
    if np.sum(A[i-gotone]) == 0:
        A = np.delete(A,i-gotone,0)#!!! - - !!!
        gotone += 1
        N -= 1
A = A.T
gotone = 0
mm = M
for i in range(mm):
    if np.sum(A[i-gotone]) == 0:
        A = np.delete(A,i-gotone,0)
        gotone += 1
        M -= 1
if N>M:
    A = A.T
else:
    N,M = M,N
tem = []
ping = [0 for i in range(N)]
Xform = np.array([[0 for j in range(edgesum)] for i in range(edgesum)])
donefor = 0
for i in A:
    n = np.sum(i)
    for j in range(n):
        for k in range(n):
            Xform[j+donefor][k+donefor] = 1
    donefor += n
    on = 0
    while n > 0:
        if i[on] == 1:
            n -= 1
            ping[on] = 1
            tem.append(ping)
            ping = [0 for j in range(N)]
        on += 1
half_c = np.array(tem).T
tem = []
Yform = np.array([[0 for j in range(edgesum)] for i in range(edgesum)])
donefor = 0
ping = [0 for i in range(edgesum)]
for i in half_c:
    n = np.sum(i)
    for j in range(n):
        for k in range(n):
            Yform[j+donefor][k+donefor] = 1
    donefor += n
    on = 0
    while n > 0:
        if i[on] == 1:
            n -= 1
            ping[on] = 1
            tem.append(ping)
            ping = [0 for j in range(edgesum)]
        on += 1
forwardConnector = np.array(tem)
backwardConnector = forwardConnector.T
Xscaler = np.array([[np.sum(Xform[i])] for i in range(edgesum)])
Yscaler = np.array([[np.sum(Yform[i])] for i in range(edgesum)])
Xway = np.array(([[1.0 for i in range(edgesum)]])).T
Yway = np.array(([[1.0 for i in range(edgesum)]])).T
def oneoverx(S):
    tem = []
    for i in S:
        if i[0] == 0:
            tem.append(0.0)
        else:
            tem.append(1.0/i[0])
    return np.array([tem]).T

Wx = oneoverx(np.dot(Xform,Xway))
Wy = oneoverx(np.dot(Yform,Yway))
X = np.array(([[600.0] for i in range(edgesum)]))
Xsendingforforward = Wx*X
YGetfromforwardX = np.dot(forwardConnector,Xsendingforforward)
Y = np.dot(Yform,YGetfromforwardX)
XGetfrombackwardY = np.dot(backwardConnector,Wy*Y)
delta_Y = oneoverx(Y)*YGetfromforwardX*Yscaler
Yway = delta_Y*Yway
Xsum = np.dot(Xform,(oneoverx(Xsendingforforward)*XGetfrombackwardY))

Xtocheck = np.subtract(Xsum,(oneoverx(Xsendingforforward)*XGetfrombackwardY))
delta_X = oneoverx(Xsum)*Xtocheck*Xscaler
for i in range(len(Xscaler)):
    if Xscaler[i][0] == 1:
        delta_X[i][0] = 1
Xway = delta_X*Xway
print( np.dot(backwardConnector,(np.dot(forwardConnector,Xway)*Yway)) )
print(np.dot(backwardConnector,Yway)*Xway)
