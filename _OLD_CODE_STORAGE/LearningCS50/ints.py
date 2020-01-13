

print("x is: ", end="")
x = input()
print("Same x plz: ")
x1 = int(input())
y = input("y is: ")
y1 = int(input("Same y plz: "))

print("[x,y]{} plus {} is {}".format(x, y, x + y))
print("[int(x,y)]{} plus {} is {}".format(x, y, int(x) + int(y)))
print("[same x,y]{} plus {} is {}".format(x, y, x1 + y1))
