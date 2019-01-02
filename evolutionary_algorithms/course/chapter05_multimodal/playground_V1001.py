import numpy as np

a = np.array([[5,5],[0,0]])
b = np.array([[6,4],[0,0]])

print(a[0]-b[0])
print(np.linalg.norm(a-b))
a = np.array([1 for i in range(5)])
b = np.array([999 for i in range(5)])
while True:
    print(np.concatenate((a, b)))