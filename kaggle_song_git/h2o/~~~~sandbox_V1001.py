a = 1

def b(x):
    # global x
    print(x)
    del x

b(a)
del a