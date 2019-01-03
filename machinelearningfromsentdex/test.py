'''
a = [i for i in range(10)]
print(a)
print(a[:])
print(a[:3])
print(a[3:])
print(a[:-3])
print(a[-3:])
print(a[:3:])
'''
'''
exDict = {'Jack':15, 'Bob':22, 'Alice':12, 'Kevin':17}

print(exDict)
print(exDict['Jack'])

exDict['Tim'] = 14
print(exDict)

exDict['Tim'] = 15
print(exDict)
del exDict['Tim']
print(exDict)
'''
'''
exDict = {'Jack':[15,'blonde'], 'Bob':[22,'brown'], 'Alice':[12,'black'], 'Kevin':[17,'red']}

print(exDict['Jack'])
print(exDict['Jack'][1])
'''

import numpy as np

#t = np.array([-1,-1])
#w = np.array([30,30])
#w_t= w*t
#a = np.linalg.norm([4,4,3,3,2,2,1,0])
#print(a)
print("hello",np.array([i for i in range(10)]))
print([4,4,3,3,2,2,1,0][::-1])
