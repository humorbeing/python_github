# a = ['N', int(1)]
a = ['N', 1]
print(a)
for i, c in enumerate(a):
    if c != 'N':
        # print(c)
        a[i] += 1
print(a)
