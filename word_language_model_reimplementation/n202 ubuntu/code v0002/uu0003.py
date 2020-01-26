a = [1,2]
b = (3,4)

print(type(a))
print(type(b))
def c():
    d = [6,7]
    for i in d:
        yield i
def e():
    yield 8
    yield 9
# for _ in range(2):
#     for i in a:
#         print(i)
#     for i in b:
#         print(i)
#     for i in c():
#         print(i)
#     for i in e():
#         print(i)

a = list(range(10))
print(a[1::2])