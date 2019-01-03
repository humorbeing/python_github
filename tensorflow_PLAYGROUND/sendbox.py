s = [[2,2],[2,2]]
print(s)
print(s[0])
print(*s[0])
s = [32]
print(s)
print(s[0])
print(*s)

import numpy as np

print(np.random.randn(10))
print(np.random.normal())


a = [1, 2]
b = a[1:500]
print(b)
import numpy as np

c = np.random.permutation(3)
# a = np.random.randn([2,2,3])
print(c)
a = np.array([[1,2,3],[4,5,6],[9,2,3]])
b = a[c]
print(b)
b = [5,5]
print(np.random.randn(5))
# print(np.random.randn(b))
# print(np.random.randn(np.array(b)))
c = np.array(b)
# print(np.random.randn(c))
print(np.random.randn(*c))
print(np.random.randn(*[5,5]))


print([5,5])
print(*[5,5])

a = np.array([[1, 2]])
b = np.array([[3, 4]])
c = np.array([[5],[6]])
print(a.shape)
print(b.shape)
print(c.shape)
print(a*b)
print(a*c)
print(c*a)
print(np.dot(a,c))
print(np.multiply(a,b))