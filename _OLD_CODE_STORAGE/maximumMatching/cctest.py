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
#print(A)
#print(N)
tem = A# !!! - - !!!
for i in range(len(A)):
    if np.sum(A[i]) == 0:
        tem = np.delete(A,i,0)#!!! - - !!!
        N -= 1

A = np.array(tem).T
#print(A)
tem = A#!!!!!!!!!! - - !!!!!!!!!!!
for i in range(len(A)):
    if np.sum(A[i]) == 0:
        tem = np.delete(A,i,0)
        M -= 1
#print(M)
print(N)
print(M)
if N>M:
    A = np.array(tem).T
else:
    A = np.array(tem)
    N,M = M,N
print(N)
print(M)
#print(A)
#print(len(A))
#print(len(A[1]))
#print(A[1])
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
            print(on)
            print(i[on])
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
connector = np.array(tem)
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
YGetfromforwardX = np.dot(connector,Wx*X)
Y = np.dot(Yform,YGetfromforwardX)
XGetfrombackwardY = np.dot(connector.T,Wy*Y)
print(YGetfromforwardX)
print(XGetfrombackwardY)
