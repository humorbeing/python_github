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
B = []
C = []
X = np.array(([0.0 for i in range(edgesum)]))
Wx = np.array(([[1.0 for i in range(edgesum)]])).T
Wy = np.array(([[1.0 for i in range(edgesum)]])).T
Y = np.array(([0.0 for i in range(edgesum)]))
for i in range(len(A)):
    if sum(A[i]) == 0:
        B = np.delete(A,i,0)
        N -= 1
A = B
A = A.T
for i in range(len(A)):
    if sum(A[i]) == 0:
        C = np.delete(A,i,0)
        M -= 1
A = C
if N>M:
    A = A.T
else:
    N,M = M,N
on = 0
n = 0
for i in A:
    n = sum(i)
    for j in range(n):
        X[on] = 1.0/n
        on += 1
X = np.array([X]).T
B = A.T
on = 0
n = 0
for i in B:
    n = sum(i)
    for j in range(n):
        Y[on] = 1.0/n
        on += 1
Y = np.array([Y]).T
connector = np.array([[0 for j in range(edgesum)] for i in range(edgesum)])
tem = []
ping = [0 for i in range(N)]
for i in A:
    n = sum(i)
    on = 0
    while n > 0:
        if i[on] == 1:
            n -= 1
            ping[on] = 1
            tem.append(ping)
            ping = [0 for i in range(N)]
        on += 1
half_c = np.array(tem).T
tem = []
ping = [0 for i in range(edgesum)]
for i in half_c:
    n = sum(i)
    on = 0
    while n > 0:
        if i[on] == 1:
            n -= 1
            ping[on] = 1
            tem.append(ping)
            ping = [0 for i in range(edgesum)]
        on += 1
connector = np.array(tem).T
print(connector)
print(X)
print(Y)
def forwardtomid()
