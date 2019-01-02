import numpy as np

# n = 2
# for _ in range(500):
#     u = np.random.random()
#     if u > 0.5:
#         b = (2*u)**(1/(n+1))
#     else:
#         b = (2*(1-u))**(-(1/(n+1)))
#     print(b)
e = [0, 0]
d = np.array(np.random.random(size=2))
print(d)
e[0], e[1] = d[1], d[0]
e[0] *= -1
e = e / np.linalg.norm(e)
d = d / np.linalg.norm(d)
print(e)
print(d)
print(np.dot(d, e))
