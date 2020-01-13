import numpy as np
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
Wxh = np.random.random((4,2))-0.5
X = np.array(([[1],[1]]))

#print(a(Wxh))
#print(a(np.dot(Wxh,X)))

a = 2
b = 3

#print((a+b,))
#print((a+b))
#print(())
n = 4
bias = np.array(([[0.01]*n])).T
#print (bias)

#print(np.subtract([2,1],[1,1]))


x = np.array(([[-5],[-4],[3],[5],[0],[5]]))
def z(X):
    return np.maximum(0,0)
def o(X):
    a = []
    for i in X:
        if i[0]>0:
            a.append(1)
        else:
            a.append(0.01)
    return np.array(([a])).T
'''
print(x)
print(len(x[0]))
print(np.maximum(0.01*x,x))
print(z(x))
'''
#print(o(x))
print(z(x))
