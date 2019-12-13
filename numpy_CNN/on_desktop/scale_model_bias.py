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



# Utility
def batch_parser(batch):
    b_size = len(batch)
    x = []
    y = []
    for i in range(b_size):
        x.append(batch[i, 0])
        y.append(batch[i, 1])
    return np.array(x), np.array(y)

# function definition
def g(x, w):
    output = np.matmul(x, w)
    return output

def back_g(gradient, w):
    output = np.matmul(gradient, w.T)
    return output

def g_weight_gradient(gradient, x):
    output = np.matmul(x.T, gradient)
    return output

def bias(x, b):
    # compare = True
    compare = False
    b_size = len(x)
    B = np.array([b[0] for _ in range(b_size)])
    output2 = x + b
    output1 = x + B
    if compare:
        print(output1==output2)
        if (output1==output2).all():
            print('its all same')
        else:
            print('its not all same, some of them are not same')
    return output1

def back_bias(gradient):
    return gradient

def bias_weight_gradient(gradient):
    g = np.mean(gradient, axis=0, keepdims=True)
    return g

def relu(x):
    y = x > 0
    y = y * 1
    output = np.multiply(x, y)
    return output, y

def back_relu(gradient, F):
    return np.multiply(gradient, F)

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

def one_cross_entropy(Y_hat, Y):
    output = np.log(np.matmul(Y,Y_hat) + 1e-10) * (-1)
    output = round(output, 4)

    return output

def cross_entropy(y, yhat):
    output = []
    for i in range(len(y)):
        output.append(one_cross_entropy(yhat[i],y[i]))

    return np.array(output).mean()

def Delta_one_cross_entropy(Y_hat, Y):
    output = -1/(Y_hat + 1e-10)
    output = np.multiply(output, Y)
    return output

def back_cross_entropy(y, yhat):
    output = []
    batch_s = len(y)
    for i in range(len(y)):
        output.append(Delta_one_cross_entropy(yhat[i], y[i]))
    return np.array(output) / batch_s

# bad seed: [6], good seed [7]
np.random.seed(7)
# Layer setup
l1_outputs = 12
l2_outputs = 9
# initialize weight
W1 = np.random.random(size=(5, l1_outputs))
W2 = np.random.random(size=(l1_outputs, l2_outputs))
W3 = np.random.random(size=(l2_outputs, 4))
B1 = np.zeros((1, l1_outputs))
B2 = np.zeros((1, l2_outputs))
B3 = np.zeros((1, 4))
# Hyperparameters
learning_rate = 0.01
EPOCH = 500
batch_size = 3

# logging
loss_log = []
for epoch in range(EPOCH):
    loss_s = []
    idx = np.random.permutation(len(D))  # i.i.d. sampling
    for i in range(len(D) // batch_size):
        batch = D[idx[i:i + batch_size]]
        x, y = batch_parser(batch)
        # forward propagation start
        forward = g(x, W1)
        forward = bias(forward, B1)


        forward, F1 = relu(forward)
        U3 = forward
        forward = g(forward, W2)
        forward = bias(forward, B2)
        forward, F2 = relu(forward)
        U6 = forward
        forward = g(forward, W3)
        forward = bias(forward, B3)
        yhat = softmax(forward)
        # forward propagation end
        J = cross_entropy(y, yhat)
        # back prop start
        gradient = back_cross_entropy(y, yhat)  # dJdyhat
        gradient = back_softmax(gradient, yhat)  # dJdU8
        dJdB3 = bias_weight_gradient(gradient)

        gradient = back_bias(gradient)  # dJdU7
        dJdW3 = g_weight_gradient(gradient, U6)

        gradient = back_g(gradient, W3)  # dJdU6
        gradient = back_relu(gradient, F2)  # dJdU5
        dJdB2 = bias_weight_gradient(gradient)

        gradient = back_bias(gradient)  # dJdU4
        dJdW2 = g_weight_gradient(gradient, U3)

        gradient = back_g(gradient, W2)  # dJdU3
        gradient = back_relu(gradient, F1)  # dJdU2
        dJdB1 = bias_weight_gradient(gradient)

        gradient = back_bias(gradient)  # dJdU1
        dJdW1 = g_weight_gradient(gradient, x)
        # back prop end

        # update weight
        W1 = W1 - (learning_rate * dJdW1)
        W2 = W2 - (learning_rate * dJdW2)
        W3 = W3 - (learning_rate * dJdW3)

        B1 = B1 - (learning_rate * dJdB1)
        B2 = B2 - (learning_rate * dJdB2)
        B3 = B3 - (learning_rate * dJdB3)

        loss_s.append(J)

    ep_loss = np.mean(loss_s)
    loss_log.append(ep_loss)

plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

