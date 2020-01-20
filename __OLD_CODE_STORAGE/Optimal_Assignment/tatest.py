from random import randint
'''
a = [[0,0,5],[4,4,4],[4,4,4]]
m = 0
print(a)
print(max(max(a)))
def findmax(s):
    global m
    #n = int(s)
    if s>m:
        m = s
    return s
for i in a:
    for j in i:
        c = findmax(j)

print(m)


a = [i*j+k for i in range(2) for j in range(2) for k in range(2)]
b = [[i*j+k for i in range(2) for j in range(2)] for k in range(2)]
c = [[[i*j+k] for i in range(2) for j in range(2)] for k in range(2)]
d = [[i*j+k] for i in range(2) for j in range(2) for k in range(2)]
print(a)
print(b)
print(c)
print(d)


a = [100*i+10*j+k for i in range(2) for j in range(3) for k in range(4)]
b = [[100*i+10*j+k for i in range(2) for j in range(3)] for k in range(4)]
c = [[[100*i+10*j+k] for i in range(2) for j in range(3)] for k in range(4)]
d = [[100*i+10*j+k] for i in range(2) for j in range(3) for k in range(4)]
e = [[[100*i+10*j+k for i in range(2) for j in range(3) for k in range(4)]]]
f = [[[100*i+10*j+k for i in range(2) for j in range(3)] for k in range(4)]]
g = [[[100*i+10*j+k]] for i in range(2) for j in range(3) for k in range(4)]
print('-'*79)
print(a)
print('-'*79)
print(b)
print('-'*79)
print(c)
print('-'*79)
print(d)
print('-'*79)
print(e)
print('-'*79)
print(f)
print('-'*79)
print(g)
print('-'*79)
'''

print(str(randint(1, 1000)) for i in range(8))
