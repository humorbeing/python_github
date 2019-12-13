import numpy as np
import matplotlib.pyplot as plt
EPS = 0.0000000000001
# EPS = 1e-12

plot_title = 'Numpy CNN (Conv Bias + L2 regu + Batch training)'
# trainset = np.load('./simpleset_1per.npy')  # 1 image per class, total 10 images
trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images

# Utility functions
def normalize_image(image_in):
    output_image = image_in / 255.0
    return output_image

def batch_sample_label_separator(batch):
    if len(batch.shape)==1:
        batch = np.expand_dims(batch, axis=0)
    b_size = len(batch)
    x = []
    y = []

    for i in range(b_size):
        # adding a dimension into input image
        # So: 28x28 -> 1x28x28
        sample = np.expand_dims(batch[i, 0], axis=0)
        # sample = np.reshape(batch[i, 0],(1,28,28))
        x.append(sample)
        y.append(batch[i, 1])
    return np.array(x), np.array(y)

def get_one_hot(targets, nb_classes=10):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

# forward and backward functions
def Z(K, V, i, j, k, stride):
    k_shape = K.shape
    kernel_channel = k_shape[1]
    kernel_row = k_shape[2]
    kernel_column = k_shape[3]
    z = 0
    for l in range(kernel_channel):
        for m in range(kernel_row):
            for n in range(kernel_column):
                z += V[l][j*stride+m][k*stride+n] * K[i][l][m][n]
    return z


def conv(V, K, stride=1):
    output_channel = K.shape[0]
    kernel_channel = K.shape[1]
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]

    input_channel = V.shape[0]
    input_row = V.shape[1]
    input_col = V.shape[2]
    assert (input_channel == kernel_channel), "channel doesn't match!"
    output_row = int((input_row - kernel_row) / stride + 1)
    output_col = int((input_col - kernel_col) / stride + 1)
    output = np.zeros(shape=(output_channel, output_row, output_col))
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i][j][k] = Z(K, V, i, j, k, stride=stride)
    return output

def batch_conv(x, k):
    output = []
    for i in range(len(x)):
        output.append(conv(x[i], k))
    return np.array(output)

def delta_V(K, G, i, j, k):
    kernel_output_channel = K.shape[0]
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]
    l_range = G.shape[1]
    n_range = G.shape[2]
    output = 0
    for q in range(kernel_output_channel):
        for m in range(kernel_row):
            l = j - m
            if 0 <= l < l_range:
                for p in range(kernel_col):
                    n = k - p
                    if 0 <= n < n_range:
                        output += K[q][i][m][p] * G[q][l][n]
    return output

def back_conv(G, K, V_shape):
    DELTA_V = np.zeros(shape=(V_shape))

    for i in range(DELTA_V.shape[0]):
        for j in range(DELTA_V.shape[1]):
            for k in range(DELTA_V.shape[2]):
                DELTA_V[i][j][k] = delta_V(K, G, i, j, k)
    return DELTA_V

def batch_back_conv(G, K, V_shape):
    output = []
    for i in range(len(G)):
        output.append(back_conv(G[i], K, V_shape))
    return np.array(output)

def delta_K(G, V, K_shape, i, j, k, l, stride=1):
    kernel_row = K_shape[2]
    kernel_col = K_shape[3]
    output = 0
    for m in range(kernel_row):
        for n in range(kernel_col):
            output += G[i][m][n] * V[j][m*stride+k][n*stride+l]
    return output


def conv_weight_gradient(G, V, K_shape):
    DELTA_K = np.zeros(shape=(K_shape))
    for i in range(DELTA_K.shape[0]):
        for j in range(DELTA_K.shape[1]):
            for k in range(DELTA_K.shape[2]):
                for l in range(DELTA_K.shape[3]):
                    DELTA_K[i][j][k][l] = delta_K(G, V, K_shape, i, j, k, l)
    return DELTA_K

def batch_conv_weight_gradient(G, V, K_shape):
    output = []
    for i in range(len(G)):
        output.append(conv_weight_gradient(G[i],V[i],K_shape))
    g = np.mean(np.array(output), axis=0)
    return g

def conv_bias(x, b):
    y_dimension = x.shape[1]
    x_dimension = x.shape[2]
    m = np.ones(shape=(1, y_dimension * x_dimension))
    mb = np.matmul(b, m)
    mb = mb.reshape(x.shape)
    output = x + mb
    return output

def batch_conv_bias(x, b):
    output = []
    for i in range(len(x)):
        output.append(conv_bias(x[i], b))
    return np.array(output)

def back_conv_bias(x):
    return x

def conv_bias_weight_gradient(gradient):
    z_dimension = gradient.shape[0]
    y_dimension = gradient.shape[1]
    x_dimension = gradient.shape[2]
    g = gradient.reshape((z_dimension, y_dimension * x_dimension))
    m = np.ones(shape=(y_dimension * x_dimension, 1))
    output = np.matmul(g, m)
    return output

def batch_conv_bias_weight_gradient(gradient):
    output = []
    for i in range(len(gradient)):
        output.append(conv_bias_weight_gradient(gradient[i]))
    g = np.mean(np.array(output), axis=0)
    return g

def relu(x):
    y = x > 0
    y = y * 1
    output = np.multiply(x, y)
    return output

def back_relu(gradient, x):
    y = x > 0
    y = y * 1
    return np.multiply(gradient, y)

def Pool(V, i, j, k, kernel):
    values = []
    locations = []
    for m in range(kernel):
        for n in range(kernel):
            row = j * kernel + m
            col = k * kernel + n
            values.append(V[i][row][col])
            locations.append((i, row, col))
    max_value = max(values)
    ind = values.index(max_value)
    location = locations[ind]
    return max_value, location


def max_pooling(V, kernel=2):
    input_channel = V.shape[0]
    input_row = V.shape[1]
    input_col = V.shape[2]

    output_channel = input_channel
    output_row = int(input_row / kernel)
    output_col = int(input_col / kernel)

    output = np.zeros(shape=(output_channel, output_row, output_col))
    pool_matrix = np.zeros(shape=(input_channel, input_row, input_col))
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i][j][k], loc = Pool(V, i, j, k, kernel=kernel)
                pool_matrix[loc[0]][loc[1]][loc[2]] = 1
    return output, pool_matrix

def batch_max_pooling(V):
    output = []
    pooling_matrix = []
    for i in range(len(V)):
        o, m = max_pooling(V[i])
        output.append(o)
        pooling_matrix.append(m)
    return np.array(output), np.array(pooling_matrix)

def back_max_pooling(gradient, pool_matrix, kernel=2):
    output = np.zeros(pool_matrix.shape)
    input_channel = gradient.shape[0]
    input_row = gradient.shape[1]
    input_col = gradient.shape[2]
    for i in range(input_channel):
        for j in range(input_row):
            for k in range(input_col):
                for m in range(kernel):
                    for n in range(kernel):
                        row = j * kernel + m
                        col = k * kernel + n
                        output[i][row][col] = gradient[i][j][k]
    output = np.multiply(output, pool_matrix)
    return output

def batch_back_max_pooling(gradient, pool_matrix):
    output = []
    for i in range(len(gradient)):
        output.append(back_max_pooling(gradient[i], pool_matrix[i]))
    return np.array(output)

def tensor_to_vector(x):
    tensor_shape = x.shape
    output = np.reshape(x, (tensor_shape[0], -1))
    return output, tensor_shape

def vector_to_tensor(x, shape):
    output = np.reshape(x, shape)
    return output

def MLP(x, w):
    output = np.matmul(x, w)
    return output

def back_MLP(gradient, w):
    output = np.matmul(gradient, w.T)
    return output

def MLP_weight_gradient(gradient, x):
    output = np.matmul(x.T, gradient)
    return output / len(gradient)

def MLP_bias(x, bias):
    output = x + bias
    return output

def back_MLP_bias(gradient):
    return gradient

def MLP_bias_weight_gradient(gradient):
    g = np.mean(gradient, axis=0, keepdims=True)
    return g

def softmax(X):
    X_exp = [np.exp(i) for i in X]
    sum_exp = np.sum(X_exp)
    output = [(i / (sum_exp + EPS)) for i in X_exp]
    output = np.array(output)
    return output


def batch_softmax(x):
    output = []
    for i in x:
        output.append(softmax(i))

    return np.array(output)

def back_softmax(gradient ,S):
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

def batch_back_softmax(gradient, S):
    output = []
    for i in range(len(gradient)):
        output.append(back_softmax(gradient[i], S[i]))
    return np.array(output)

def cross_entropy(Y, Y_hat):
    output = np.log(np.matmul(Y,Y_hat) + EPS) * (-1)
    return output

def batch_cross_entropy(y, yhat):
    output = []
    for i in range(len(y)):
        output.append(cross_entropy(y[i], yhat[i]))

    return np.array(output).mean()

def back_cross_entropy(Y, Y_hat):
    output = -1/(Y_hat + EPS)
    output = np.multiply(output, Y)
    return output

def batch_back_cross_entropy(y, yhat):
    output = []
    for i in range(len(y)):
        output.append(back_cross_entropy(y[i], yhat[i]))
    return np.array(output)

# Hyperparameters
learning_rate = 0.006
lambda_regu = 0.00002
EPOCH = 200
batch_size = 20

# Weight initialization
np.random.seed(10)
adjust = 0.033  # importance of random weight init: bigger than 0.05, explode gradient, less than 0.01, can't train much
K1 = np.random.random(size=(5, 1, 5, 5)) * adjust  # K1 is kernel of first conv layer, or its the parameters of first conv layer. shape is (out channel, input channel, kernel size, kernel size)
K2 = np.random.random(size=(10, 5, 5, 5)) * adjust  # K2 is kernel of second conv layer
W1 = np.random.random(size=(1000, 100)) * adjust
W2 = np.random.random(size=(100, 10)) * adjust

B1 = np.zeros(shape=(K1.shape[0], 1))  # B1 is bias of first conv layer. Need same number of bias as output channel number, so each output can has a bias, which is K1.shape[0]
B2 = np.zeros(shape=(K2.shape[0], 1))  # B2 is bias of second conv layer.
B3 = np.zeros(shape=(1, W1.shape[1]))  # B3 is bias of first MLP layer. since input is 1000, and transform to 100, we need 100 bias, which is W1.shape[1]
B4 = np.zeros(shape=(1, W2.shape[1]))  # B4 is bias of second MLP layer.

loss_log = []
for epoch in range(EPOCH):
    # print(epoch)
    loss_s = []
    idx = np.random.permutation(len(trainset))  # i.i.d. sampling
    for i in range(len(trainset) // batch_size):
        batch = trainset[idx[i:i + batch_size]]
        x, y = batch_sample_label_separator(batch)
        x = normalize_image(x)
        label = get_one_hot(y)

        # START: feed forward propagation
        forward = batch_conv(x, K1)
        forward = batch_conv_bias(forward, B1)
        U2 = forward
        forward = relu(forward)

        U3 = forward
        forward = batch_conv(forward, K2)
        forward = batch_conv_bias(forward, B2)
        U5 = forward
        forward = relu(forward)

        forward, pooling_matrix = batch_max_pooling(forward)
        forward, tensor_shape = tensor_to_vector(forward)

        U8 = forward
        forward = MLP(forward, W1)
        forward = MLP_bias(forward, B3)
        U9 = forward
        forward = relu(forward)

        U11 = forward
        forward = MLP(forward, W2)
        forward = MLP_bias(forward, B4)
        forward = batch_softmax(forward)
        # End: feed forward propagation
        J = batch_cross_entropy(label, forward)
        # Start: back propagation
        gradient = batch_back_cross_entropy(label, forward)
        gradient = batch_back_softmax(gradient, forward)

        dJdB4 = MLP_bias_weight_gradient(gradient)
        gradient = back_MLP_bias(gradient)
        dJdW2 = MLP_weight_gradient(gradient, U11)

        gradient = back_MLP(gradient, W2)
        gradient = back_relu(gradient, U9)

        dJdB3 = MLP_bias_weight_gradient(gradient)
        gradient = back_MLP_bias(gradient)
        dJdW1 = MLP_weight_gradient(gradient, U8)

        gradient = back_MLP(gradient, W1)
        gradient = vector_to_tensor(gradient, tensor_shape)
        gradient = batch_back_max_pooling(gradient, pooling_matrix)
        gradient = back_relu(gradient, U5)

        dJdB2 = batch_conv_bias_weight_gradient(gradient)
        gradient = back_conv_bias(gradient)
        dJdK2 = batch_conv_weight_gradient(gradient, U3, K2.shape)

        gradient = batch_back_conv(gradient, K2, U3[0].shape)
        gradient = back_relu(gradient, U2)

        dJdB1 = batch_conv_bias_weight_gradient(gradient)
        gradient = back_conv_bias(gradient)
        dJdK1 = batch_conv_weight_gradient(gradient, x, K1.shape)
        # End: back propagation

        # This is L2 regularization.
        dJdK1 += K1 * 2 * lambda_regu
        dJdK2 += K2 * 2 * lambda_regu
        dJdW1 += W1 * 2 * lambda_regu
        dJdW2 += W2 * 2 * lambda_regu

        # Updating kernel parameter
        K1 = K1 - learning_rate * dJdK1
        K2 = K2 - learning_rate * dJdK2
        W1 = W1 - learning_rate * dJdW1
        W2 = W2 - learning_rate * dJdW2

        # --->!!! DO NOT !!!<--- apply regularization on bias term
        # dJdB1 += B1 * 2 * lambda_regu
        # dJdB2 += B2 * 2 * lambda_regu
        # dJdB3 += B3 * 2 * lambda_regu
        # dJdB4 += B4 * 2 * lambda_regu

        # Updating bias parameter
        scaling_conv_bias = 0.5  # unsure of conv bias layer's back-prop.
        B1 = B1 - learning_rate * dJdB1 * scaling_conv_bias
        B2 = B2 - learning_rate * dJdB2 * scaling_conv_bias
        B3 = B3 - learning_rate * dJdB3
        B4 = B4 - learning_rate * dJdB4
        loss_s.append(J)

    ep_loss = np.mean(loss_s)
    print('epoch: {}. loss: {}'.format(epoch, ep_loss))
    loss_log.append(ep_loss)

plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title(plot_title)
plt.show()

