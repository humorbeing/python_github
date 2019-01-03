import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[0, 1, 2, 3]])
W = np.random.random((1, 2))
learning_rate = 0.1
epoch = 100
print('Weight Before Training: {}'.format(W[0]))
for _ in range(epoch):
    dW = -1*(Y-W.dot(X)).dot(X.T)*1
    W -= learning_rate*dW
print('Weight After Training:  {}'.format(W[0]))
