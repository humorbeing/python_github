'''
A = [
    [0,99],
    [0,98],
    [0,97]
    ]

b = [sum(i) == 99 for i in A]
print(b)
if (any(b)):
    print('true for if any b')
else:
    print('false for if any b')
#print( sum(A[0:2]) == 99)
'''
a = [i for i in range(10)]
print(a)
print(a.count(0))
a[9] = 0
print(a.count(0))
