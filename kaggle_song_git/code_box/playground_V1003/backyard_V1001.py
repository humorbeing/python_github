import numpy as np

# np.arange(185, 12173850, )
a = np.linspace(185, 12174000, 50)
print(a)
for i in range(4):

    # print('i is :', i, ' value:', a[i])
    pass
def d():
    x = 0
    while True:
        if x == 5:
            return 5
        else:
            x += 1

print(d())

# while True:
#     print(int(np.random.uniform(1980, 2016)))

a ='hi'
print(len(a))
# while True:
#     a = np.random.poisson(2016, 1000)
#     print(np.mean(a))

a = ['1','one','one','one']
b = set(a)
b.add('yir')
a = 'aa|one'
c = a.split('|')
print(c)
print(b)
for i in c:
    b.add(i)
print(b)
# while True:
#     print(np.random.sample(a))