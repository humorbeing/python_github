import numpy as np
import random



#if random.random() < 20:

'''

s = 0
n = 50000
so = []
for _ in range(1000):
    s = 0
    for i in range(n):
        #if np.random.randint(0,1000) < 10:
        if random.random() < 0.02:
            s += 1
    so.append(s/n)
    print(s/n)

print('Min:',np.sort(so)[0],'Max:',np.sort(so)[-1])
'''
'''
i = [[],0]
i[0] = [5,6]
i[0].append(7)
print (i)
'''
'''
dot = {1:{1:True,
          2:True,
          3:True,
          4:True,
          },
       2:{1:True},
       3:{1:True},
       4:{1:True},
       5:{1:True},
       6:{1:True},
       7:{1:True},
       8:{1:True},
       9:{1:True},
       10:{1:True},

       }

print (dot[1][4])
'''
'''
dot = {1:{},2:{}}
dot[1][1] = True
dot[1][2] = True
dot[2][1] = True
dot[2][2] = True

print (dot)
'''
'''
dot = {1:{},2:{}}
dot[1] = {i:True for i in range(10)}

print(dot)'''
#bugged = {i:False for i in range(100)}
#print(bugged)

if False or False or True:
    print(1)
