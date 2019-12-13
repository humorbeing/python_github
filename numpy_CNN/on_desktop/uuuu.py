

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


def back_function1_weight(gradient, x):
    g = gradient * x
    return g


trainset = [
    [6, 3],
    [4, 2]
]

W1 = 0.9
W2 = 0.8
W3 = 0.2
learning_rate = 0.002
EPOCH = 20

for epoch in range(EPOCH):
    loss_s = []
    for x, y in trainset:
        forward = function1(x, W1)  # U1
        U1 = forward
        forward = function1(forward, W2)  # U2
        U2 = forward
        forward = function1(forward, W3)  # yhat
        yhat = forward
        J = loss_function(y, yhat)

        gradient = back_loss_function(y, yhat)  # dJdU3
        dJdW3 = back_function1_weight(gradient, U2)

        gradient = back_function1(gradient, W3)  # dJdU2
        dJdW2 = back_function1_weight(gradient, U1)

        gradient = back_function1(gradient, W2)  # dJdU1
        dJdW1 = back_function1_weight(gradient, x)

        W1 = W1 - (learning_rate * dJdW1)
        W2 = W2 - (learning_rate * dJdW2)
        W3 = W3 - (learning_rate * dJdW3)

        loss_s.append(J)

    ep_loss = (loss_s[0] + loss_s[1]) / 2
    print(ep_loss)
