import numpy as np


def get_U1(X, W):
    U1 = X @ W
    return U1


def get_U2(U1, B):
    U2 = U1 + B
    return U2


def relu(X):
    x_in = np.array(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            if x_in[i][j] > 0:
                pass
            else:
                x_in[i][j] = 0
    return x_in


def get_Y(U2):
    return relu(U2)


minibatch_size = 5
number_node = 10


def layer(X, W, B):
    U1 = get_U1(X, W)
    U2 = get_U2(U1, B)
    Y = get_Y(U2)
    return Y


a = np.array([i for i in range(20)]).reshape((5, 4))
print(a)
b = np.array([i for i in range(40)]).reshape((4, 10))
print(b)
c = np.array([1 for i in range(50)]).reshape((5, 10))
print(c)

y = layer(a,b,c)
print(y)