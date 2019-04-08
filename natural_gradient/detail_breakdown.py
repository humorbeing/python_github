import numpy as np
from sklearn.utils import shuffle

X0 = np.random.randn(100, 2) - 1
X1 = np.random.randn(100, 2) + 1
X = np.vstack([X0, X1])
t = np.vstack([np.zeros([100, 1]), np.ones([100, 1])])

X, t = shuffle(X, t)

X_train, X_test = X[:150], X[150:]
t_train, t_test = t[:150], t[150:]
# print(X_train[149], X_test[0])
# print(X[149], X[150])
# print(len(X_train), len(X_test))

# Initialize weight
W = np.random.randn(2, 1) * 0.01
W1 = W
alpha = 0.01

def sigm(x):
    return 1/(1+np.exp(-x))


def NLL(y, t):
    return -np.mean(t*np.log(y) + (1-t)*np.log(1-y))


    # Forward
X_train = np.array([[2, 1]])
W = np.array([[-1], [1]])
t_train = np.array([[1]])
print('X shape', X_train.shape)
print('W shape', W.shape)
z = X_train @ W
print('z shape', z.shape)
print('z', z)
y = sigm(z)
print('y shape', y.shape)
print('y', y)
loss = NLL(y, t_train)
print('loss', loss)
m = y.shape[0]
# Loss
# print('Loss:', loss)
dy = (y-t_train)/(m * (y - y*y))
dz = sigm(z)*(1-sigm(z))
dW = X_train.T @ (dz * dy)

# grad_loglik_z = (t_train - y) / (y - y * y) * dz
# grad_loglik_W = grad_loglik_z * X_train
# F = np.cov(grad_loglik_W.T)

# Step
# W = W - alpha * np.linalg.inv(F) @ dW

W = W - alpha * dW
