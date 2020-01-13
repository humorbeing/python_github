from collections import namedtuple

a = namedtuple('Point', ['x', 'y'])
p = a(11, y=22)
print(p)
print(type(a))
print(type(p))
print(p[0])
print(p.x)
print(p.y)

