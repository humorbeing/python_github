import numpy as np
import matplotlib.pyplot as plt

# Making Training set starts
def make_trainingset():
    samples = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        samples.append([a, b, c, d, e])

    samples = np.array(samples)

    def makelabel(x):
        count = sum(x)
        if count < 2:
            return np.array([1, 0, 0, 0])
        elif count == 2:
            return np.array([0, 1, 0, 0])
        elif count == 3:
            return np.array([0, 0, 1, 0])
        else:
            return np.array([0, 0, 0, 1])

    trainingset = []
    for x in samples:
        trainingset.append([x, makelabel(x)])

    trainingset = np.array(trainingset)
    return trainingset


# Making Traninig set end
D = make_trainingset()  # <--training set
a = [1, -10, 5]


# Utility
def batch_parser(batch):
    b_size = len(batch)
    x = []
    y = []
    for i in range(b_size):
        x.append(batch[i, 0])
        y.append(batch[i, 1])
    return np.array(x), np.array(y)


def g(x, w):
    output = np.matmul(x, w)
    return output

def back_g(gradient, w):
    output = np.matmul(gradient, w.T)
    return output

def g_weight_gradient(gradient, x):
    output = np.matmul(x.T, gradient)
    return output

def relu(x):
    y = x > 0
    y = y * 1
    output = np.multiply(x, y)
    return output, y


def one_softmax(X):
    X_exp = [np.exp(i) for i in X]
    sum_exp = np.sum(X_exp)
    output = [np.round(i / sum_exp, 4) for i in X_exp]
    output = np.array(output)
    return output


def softmax(x):
    output = []
    for i in x:
        output.append(one_softmax(i))

    return np.array(output)

def one_cross_entropy(Y_hat, Y):
    output = np.log(np.matmul(Y,Y_hat)) * (-1)
    output = round(output, 4)

    return output

def cross_entropy(y, yhat):
    output = []
    for i in range(len(y)):
        output.append(one_cross_entropy(yhat[i],y[i]))

    return np.array(output).mean()
# a = 0 + 1e-8
# print(a)

def Delta_one_cross_entropy(Y_hat, Y):
    output = -1/(Y_hat + 0.0000000001)
    output = np.multiply(output, Y)
    return output

def back_cross_entropy(y, yhat):
    output = []
    for i in range(len(y)):
        output.append(Delta_one_cross_entropy(yhat[i], y[i]))
    return np.array(output)

def Delta_one_softmax(gradient ,S):
    output_size = len(S)
    output = np.zeros(shape=(output_size, output_size))
    for i in range(output_size):
        for j in range(output_size):
            if i == j:
                output[i][j] = S[i] * (1 - S[j])
            else:
                output[i][j] = (-1) * S[i] * S[j]
    new_gradient = np.matmul(gradient, output)
    return new_gradient

def back_softmax(gradient, S):
    output = []
    for i in range(len(gradient)):
        output.append(Delta_one_softmax(gradient[i], S[i]))
    return np.array(output)

def back_relu(gradient, F):
    return np.multiply(gradient, F)



batch_size = 3
idx = np.random.permutation(len(D))  # i.i.d. sampling

# initialize weight
W1 = np.random.random(size=(5, 9))
W2 = np.random.random(size=(9, 4))
# Hyperparameters
learning_rate = 0.002
EPOCH = 1000

# logging
loss_log = []
for epoch in range(EPOCH):
    loss_s = []
    for i in range(len(D) // batch_size):
        batch = D[idx[i:i + batch_size]]
        x, y = batch_parser(batch)
        # forward propagation start
        forward = g(x, W1)
        forward, F = relu(forward)
        U2 = forward
        forward = g(forward, W2)
        yhat = softmax(forward)
        # forward propagation end
        J = cross_entropy(y, yhat)
        # back prop
        gradient = back_cross_entropy(y, yhat)
        gradient = back_softmax(gradient, yhat)
        dJdW2 = g_weight_gradient(gradient, U2)
        gradient = back_g(gradient, W2)
        gradient = back_relu(gradient, F)
        dJdW1 = g_weight_gradient(gradient, x)

        W1 = W1 - (learning_rate * dJdW1)
        W2 = W2 - (learning_rate * dJdW2)


        loss_s.append(J)

    ep_loss = np.mean(loss_s)
    loss_log.append(ep_loss)

plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

