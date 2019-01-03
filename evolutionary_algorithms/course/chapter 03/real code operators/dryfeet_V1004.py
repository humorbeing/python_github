import numpy as np

# n = 5
# def poly():
#     u = np.random.random()
#     if u < 0.5:
#         v = ((2 * u)**(1/(n+1))) - 1
#     else:
#         v = 1 - (2*(1-u)) ** (1/(n+1))
#     return v
# for _ in range(500):
#     print(poly())

min_x = 0
max_x = 1


def fix_x(point_in):
    x_in = point_in
    # y_in = point_in[1]
    if x_in < min_x:
        x_in = 2 * min_x - x_in
    elif x_in > max_x:
        x_in = 2 * max_x - x_in
    else:
        pass

    return x_in

def fix(x):
    old = x
    while True:
        new = fix_x(old)
        if new == old:
            break
        else:
            old = new
    return new

# print(fix(100000.123))
if [2.1,3.0] == [2.1,3.1]:
    print('same')

a = np.array([2,3])
b = [2,3]
if all(a == b):
    print('same')