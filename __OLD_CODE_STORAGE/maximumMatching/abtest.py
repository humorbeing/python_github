import numpy as np

'''
A = [i for i in range(10)]

print(A)
del A[0]
print(A)

A = np.array(A).T

print(A)
A = np.delete(A,1)
print(A)
'''
'''
A = [0 for i in range(5)]
print(A)
A[1] = 1
print(A)
A = A*[0,0,0,0,0]
print(A)

n = 1
i = 0
while n>0:

    i += 1
    print(i)
    if i == 5:
        n = 0
    print(i)
    print(i)

A = [1.0/(i+1) for i in range(5)]
B = [10 for i in range(5)]
A = np.array([A]).T
B = np.array([B]).T
print(A*B)
'''

#A = np.array(([[0 for i in range(10)] for j in range(10)]))
#A[0] = A[0] + 1
#print(A)
#A = [i for i in range(10)]
#if 5 in A:
#    print(5)
#print(11 in A)

'''
a = 6
b = 6
c = 6
d = 5
e = 6
print(a==b==c==d==e)
'''
'''

#print(A)
print(np.sum(np.subtract(A,B)))
print(np.sum(np.abs(np.subtract(A,B))))
'''
A = [[1],[1],[1]]
B = [[0,2,1]]

def dot(S, T):
    #nxm () mxv = n x v
    n = len(S)
    m = len(T)
    v = len(T[0])
    tem = [[0 for j in range(v)] for i in range(n)]
    s = 0
    for i in range(n):
        for j in range(v):
            for k in range(m):
                s += S[i][k]*T[k][j]
            tem[i][j] = s
            s = 0
    return tem
print(dot(A,B))
print(np.dot(A,B))
