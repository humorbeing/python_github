a = [i for i in range(1, 10)]
print(a)
for i in a:
    print(i)
for i in enumerate(a):
    print(i)
for i, j in enumerate(a):
    print(i)
    print(j)

b = [i for i in range(2, 11)]
z = zip(a, b)
l = list(z)
print(l)
z = zip(a, b)
c, d = zip(*z)
print(list(c))
print(list(d))
z = zip(a, b)
print('len l: ', len(l))
for i, (j, k) in enumerate(z):
    print(i)
    print(j)
    print(k)


import torch.cuda as Cuda

print(Cuda.is_available())
