import numpy as np

A = np.random.random((4,5))-0.5
#print(A)
#A = np.maximum(A,0)
#print(A)
def a(A):
    aa = []
    aaa = []
    for i in A:
        aa = []
        for j in i:
            if j > 0:
                aa.append(j)
            else:
                aa.append(0)
        aaa.append(aa)
    return np.array(aaa)
#print(A)
#print(a([1]))
#print len([10])
#print(A)

A = np.random.random((1,2))
B = np.random.random((1,2))
print(A)
print(A.T)
print(A.transpose())
'''
#print(B)
print(B.T)
#print(C)
C = A - B

print(C)
print(np.subtract(A,B.T))
'''
