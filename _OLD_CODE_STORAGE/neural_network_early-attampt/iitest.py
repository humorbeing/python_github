#dot product for 3d matrix
import numpy as np

A = np.array(([
               [
                [1,2,5],
                [3,4,6]

               ]

              ]))
B = np.array(([
               [
                [1],
                [3]

               ],
               [
                [2],
                [4],

               ],
               [
                [1],
                [3],

               ]
              ]))
C = np.array(([
               [2],
               [3],
               [4]
              ]))
'''
print(A)
print('-'*5)
#print(B)
print(C)
print('-'*5)
print (np.dot(A,C))
'''
'''
A = np.array(([0,1,0],[0,1,1],[0,0,1]))
print(A)
B = []
print('-'*5)
for i in range(5):
    B.append(A)
print(B)
B = np.array(B)
print('-'*5)
print(B)

print(len(B))

'''
#print(np.random.random((3,4,1))[2])
A = [1 for i in range(4)]
B = [A for i in range(5)]

B = np.array(B)

#print(B)
a = 6
b = 5
c = 5
#C = [[[k+10*j+100*i for k in range(c)] for j in range(b)] for i in range(a)]
C = np.random.random((6,5,5))-0.5
C = np.array(C)
#print(C)
#print(np.dot(B,C))
D = np.random.random((6,5))
#print(C)
#print( np.dot( (np.dot(D,B)),C))
#print(np.dot(C,B))
#print('-'*5)
tem = []
for i in C:
    tem.append(np.dot(i,B))

tem = np.array(tem)
#print(tem)
def si(X):
    return 1/(1+np.exp(-X))
sig = si(tem)
#print(sig)
W = np.random.random((6,5,5))
sum = [[0.0 for j in range(len(sig[0][0]))] for i in range(len(sig[0]))]
#print(np.add(tem0,W))
tem = []
for w in W:
    #print(w)
    for i in sig:
        #print(np.dot(w,i))
        sum = np.add(sum,(np.dot(w,i)))
    tem.append(sum)
    sum = sum*0
outB = np.array(tem)
#print(si(outB))
sig = si(outB)
tarL = 5
Wf = np.random.random((tarL,3,5))
Wb = np.random.random((tarL,4,2))
tarX = 3
tarY = 2
sum = [[0.0 for j in range(tarY)] for i in range(tarX)]
tem = []
for i in range(tarL):
    for j in sig:
        sum = np.add( sum,( (np.dot( (np.dot(Wf[i],j)),Wb[i] ))) )
    tem.append(sum)
    sum = sum*0
sl = np.array(tem)
sig = si(sl)
#print(sig)
tem = []
for i in sig:
    for j in i:
        for k in j:
            tem.append(k)
tocf = np.array([tem]).T
#print(tocf)
tarS = 50
t = len(tocf)
wc = np.random.random((tarS,t))
FC = np.dot(wc,tocf)
y = 2
wy = np.random.random((y,len(FC)))
yHat = np.dot(wy,FC)
#print(np.array([[[sig]][0][0][0]]).shape)
#print(yHat.T[0])
#print(np.array((([yHat.T[0]]))))
a1=1
b1=2
c1=3
A = [1 for i in range(a1)]
B = [A for i in range(b1)]

B = np.array([B]*c1)

#print(B)
a = 3
b = 1
c = 2
#C = [[[k+10*j+100*i for k in range(c)] for j in range(b)] for i in range(a)]
C = np.random.random((a,b,c))-0.5
def M(X, Y):
    tem = []
    summ = [[0.0 for j in range(len(Y[0][0]))] for i in range(len(X[0]))]
    for i in X:
        for x in Y:
            summ = np.add(summ,np.dot(i,x))
        tem.append(summ)
        summ = summ*0
    return np.array(tem)


#print("{} o {} = {}.".format(C.shape,B.shape,(np.dot(C,B).shape)))
#print("{} * {} = {}.".format(C.shape,B.shape,((C*B).shape)))
#print("{} M {} = {}.".format(C.shape,B.shape,(M(C,B).shape)))
#print(B.shape)
#print(M(C,B))
#print(np.dot(C,B).shape)
#print((C*B).shape)
