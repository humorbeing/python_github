import numpy as np

def relu(X):
    matrix_shape = X.shape
    F_matrix = np.zeros(matrix_shape)
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            if X[i][j] > 0:
                F_matrix[i][j] = 1
    return F_matrix


def layer(X, W, B):
    # print('X:', X.shape)
    # print('W:', W.shape)
    # print('B:', B.shape)
    # U = X @ W
    U = np.matmul(X, W)
    V = U + B

    F = relu(V)
    Y = np.multiply(F, V)
    return Y


def back_layer(dJdY, X, W, B):
    # print('djdy:', dJdY.shape)
    print('X:', X.shape)
    print('W:', W.shape)
    print('B:', B.shape)
    # U = X @ W
    U = np.matmul(X, W)
    V = U + B

    F = relu(V)
    A = np.multiply(F, dJdY)
    Delta_B = A
    # Delta_W = X.T @ A
    Delta_W = np.matmul(X.T, A)
    # Delta_X = A @ W.T
    Delta_X = np.matmul(A, W.T)
    return Delta_X, Delta_W, Delta_B


def W_B(X, node):
    b = len(X)
    f = len(X[0])
    n = node
    # print('b, f, n:', b, f, n)
    W = np.random.random([f, n])
    B = np.zeros([b, n])
    # print('W shape:', W.shape)
    # print('B shape:', B.shape)
    return W, B

a = [i for i in range(6)]
train_set = np.array(a).reshape((1, 6))

# layers = 5
nodes = [10, 15, 20, 25, 30, 1]
# nodes = [4, 5]
weights = []
baises = []
Xs = []
x = train_set
for i in nodes:
    # print('creating nodes:', i)
    w, b = W_B(x, i)
    weights.append(w)
    baises.append(b)
    x = b
X = train_set

for i in range(len(nodes)):
    # print('layer:', i)
    Xs.append(X)
    Y = layer(X, weights[i], baises[i])
    X = Y
    # Xs.append(X)
print('size of Xs:', len(Xs))
print(Y)
print(Y.shape)

dy = Y
for i in range(len(nodes)):
    j = len(nodes) - i - 1
    print('layer:', j)
    dx, dw, db = back_layer(dy, Xs[j], weights[j], baises[j])
    dy = dx