import numpy as np
A = np.array(([[1,2],[3,4]]))
B = np.array(([[1,2],[3,4]]))

#print(np.dot(A,B))
#print(A*B)
C = np.array(([[1,1]]))
E = np.dot(C,A)
D = [[1 for j in range(4)] for i in range(2)]
D = np.array(D)

print(np.dot(E,D).T)
