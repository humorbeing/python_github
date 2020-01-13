def a(b):
    print(b)


c = a
# d = a()

c('a')
# d('b')

a = [1,2]
b = [3,4]

for i,j in zip(a, b):
    print(i)
    print(j)

for (i,j) in zip(a, b):
    print(i)
    print(j)

for i in zip(a, b):
    print(i)
    # print(j)