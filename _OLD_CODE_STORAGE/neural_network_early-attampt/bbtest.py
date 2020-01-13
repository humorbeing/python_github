import numpy as np
#print(np.random.random((3,4)))
#print(np.array([[0,1,1,0]]).T)
X = np.array(([3,5],[5,1],[10,2]),dtype=float)
Y = np.array(([3,5],[5,1],[10,2]),dtype=float).T
#print(np.dot(X,Y))
def sigmoid(z):
#    pass
    return 1/(1+np.exp(-z))

'''
print (sigmoid(1))
print (sigmoid(np.array([-1,0,1])))
print (sigmoid(np.random.random((4,4))))
'''
#print(np.random.random((3,4)))

#A = np.array(([2,2],[3,3],[4,4]))
#print(A)
B = np.array(([2],[1]))
#print(B)
#print(np.dot(A,B))
A = np.array(([[2,2,5],[3,3,4],[4,4,3]]))
A = np.array(([[3,4],[2,16]]))
invA = np.linalg.inv(A)
#print (invA)
#print(np.dot(A,invA))
A = np.array(([[-1],[2]]))
B = np.array(([[1],[2]]))
#print(A)
#print(B)
#print(A*B)
def relu(x):

    if x>0:
        return x
    else:
        return 0
print (relu(A))
