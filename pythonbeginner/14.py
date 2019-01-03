x = 6

def example():
    global x
    z = 5
    print(z)
    print(x+z)
    x += z #only with global x
    print(x)

    globx = x
    print(globx)
    globx+=5
    print(globx)

    return globx ##some ppl hate the idea globle x. use this.

x = example()

print(x)
