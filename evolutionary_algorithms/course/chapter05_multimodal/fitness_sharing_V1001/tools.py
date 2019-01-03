import numpy as np


def is_fixed(point_in, boundary_in):
    x_in = point_in[0]
    y_in = point_in[1]
    if x_in < boundary_in[0]:
        x_in = 2 * boundary_in[0] - x_in
    elif x_in > boundary_in[1]:
        x_in = 2 * boundary_in[1] - x_in
    else:
        pass
    if y_in < boundary_in[2]:
        y_in = 2 * boundary_in[2] - y_in
    elif y_in > boundary_in[3]:
        y_in = 2 * boundary_in[3] - y_in
    else:
        pass

    return np.array([x_in, y_in])


def fix_one(point_in, boundary_in):
    old = point_in
    while True:
        new = is_fixed(old, boundary_in)
        if all(new == old):
            break
        else:
            old = new
    return new


def reflect_fix(lambda_gen_in, boundary_in):
    lambda_gen = []
    for i in lambda_gen_in:
        x = fix_one(i, boundary_in)
        lambda_gen.append(x)
    return np.array(lambda_gen)


def fix_x(point_in, boundary_in):
    x_in = point_in[0]
    y_in = point_in[1]
    if x_in < boundary_in[0]:
        x_in = boundary_in[0]
    elif x_in > boundary_in[1]:
        x_in = boundary_in[1]
    else:
        pass
    if y_in < boundary_in[2]:
        y_in = boundary_in[2]
    elif y_in > boundary_in[3]:
        y_in = boundary_in[3]
    else:
        pass
    return np.array([x_in, y_in])


def boundary_fix(lambda_gen_in, boundary_in):
    lambda_gen = []
    for i in lambda_gen_in:
        x = fix_x(i, boundary_in)
        lambda_gen.append(x)
    return np.array(lambda_gen)


fixs = {
    0: reflect_fix,
    1: boundary_fix,
}


def fix(choice, lambda_gen_in, boundary_in):
    return fixs[choice](lambda_gen_in, boundary_in)
