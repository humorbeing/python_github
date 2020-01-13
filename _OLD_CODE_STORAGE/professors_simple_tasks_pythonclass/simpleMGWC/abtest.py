import numpy as np
'''
A = [i for i in range(10)]
b = [2,3]
print(A)
print(b)
print( b in A)

print( not 2 in A)
B = A
print(A == B)
a = [0,1,1,0]
b = [0,1,1,0]
c = [0,1,0,1]
d = [1,1,0,1]
D = [a,b,c]
print(D)
print(a==b)
print(a==c)
print(a in D)
print(d in D)
E = set()
E.add(3)
#E.add(a)
print(E)

g = 1
def test():
    if g == 1:
        return [1]
    else:
        return False

if test():
    print('here')
else:
    print('false?')
'''
'''
lisst = []
a = [0,1,1,0]
b = [0,1,1,0]
c = [0,1,0,1]
d = [1,1,0,1]
lisst.append(a)
#print(lisst)
a = c
#print(lisst)
'''
'''
class cc(object):
    def __init__(self):
        self.lisst = []
        self.c = [0,1,0,1]
    def aa(self):
        self.bb()
    def bb(self):
        #a = [0,1,1,0]
        b = [0,1,1,0]
        c = [0,1,0,1]
        d = [1,1,0,1]
        a = self.c

        self.lisst.append(a)
        print(self.lisst)
        a[0]=9

        print(self.lisst)

tt = cc()
tt.aa()
'''

'''
a = [0,1]
#b = [0 for i in range(2)]
#b = []
b = a

#b[0] = a[0]
#b[1] = a[1]

#print(a)
def givelistto(s):
    b = [0 for i in range(len(s))]
    for i in range(len(s)):
        b[i]=s[i]
    return b
c = givelistto(a)
b[0]=1

print(a)

c[0] = 2
print(a)


a = []
a.append(b[1])
'''
a = np.array([1,2,3,4])
#b=a
b = np.array(a)
b[0]=9
print(a)
print(b)
