data = {
    'a': 1,
    'b': 2,
    'c': 3,
}

try:
    d = data['d']
except KeyError:
    d = 0
try:
    e = data['e']
except KeyError:
    e = 5
# print(d,e)
import numpy as np
# while True:
#     print(np.random.randint(1, 3))
a = 'TWUM71200043'
print(a[-5:])