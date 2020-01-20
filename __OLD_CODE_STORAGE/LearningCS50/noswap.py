def main():
    x=1
    y=2

    print("x is {}".format(x))
    print("y is {}".format(y))
    print("swapping...")
    swap(x,y)
    print("swapped")
    print("x is {}".format(x))
    print("y is {}".format(y))

def swap(a, b): #doesn't work
    tmp = a     #python don't have pointers
    a = b       #can't really solve it like C
    b = tmp

if __name__ == "__main__":
    main()
