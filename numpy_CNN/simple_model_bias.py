import numpy as np
import matplotlib.pyplot as plt


def loss_function(y, yhat):
    loss = 0.5 * (y - yhat) ** 2
    return loss


def back_loss_function(y, yhat):
    back = yhat - y
    return back


def function1(x, w):
    output = x * w
    return output


def back_function1(gradient, w):
    g = gradient * w
    return g


def function1_weight_gradient(gradient, x):
    g = gradient * x
    return g


def function2(x, w):
    output = x + w
    return output


def back_function2(gradient):
    return gradient


def function2_weight_gradient(gradient):
    return gradient


trainset = [
    [6, 3],
    [4, 2]
]

# weight initialization
W1 = 0.9
W2 = 0.8
W3 = 0.2
W4 = 0.1

# Hyperparameters
learning_rate = 0.002
EPOCH = 20

# logging
loss_log = []
for epoch in range(EPOCH):
    loss_s = []
    for x, y in trainset:
        # forward propagation start
        forward = function1(x, W1)  # U1
        U1 = forward
        forward = function1(forward, W2)  # U2
        U2 = forward
        forward = function1(forward, W3)  # U3
        U3 = forward
        forward = function2(forward, W4)  # yhat
        yhat = forward
        # forward propagation end
        J = loss_function(y, yhat)

        # back propagation start
        gradient = back_loss_function(y, yhat)  # dJdyhat
        dJdW4 = function2_weight_gradient(gradient)

        gradient = back_function2(gradient)  # dJdU3
        dJdW3 = function1_weight_gradient(gradient, U2)

        gradient = back_function1(gradient, W3)  # dJdU2
        dJdW2 = function1_weight_gradient(gradient, U1)

        gradient = back_function1(gradient, W2)  # dJdU1
        dJdW1 = function1_weight_gradient(gradient, x)
        # back propagation end

        W1 = W1 - (learning_rate * dJdW1)
        W2 = W2 - (learning_rate * dJdW2)
        W3 = W3 - (learning_rate * dJdW3)
        W4 = W4 - (learning_rate * dJdW4)

        loss_s.append(J)

    ep_loss = np.mean(loss_s)
    loss_log.append(ep_loss)

plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()