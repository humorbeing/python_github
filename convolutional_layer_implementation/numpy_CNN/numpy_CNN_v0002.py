import numpy as np
import matplotlib.pyplot as plt
EPS = 0.0000000000001
# EPS = 1e-12

plot_title = 'Numpy CNN (Conv Bias + L2 regu + Batch training)'
trainset = np.load('./simpleset_1per.npy')  # 1 image per class, total 10 images
# trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images

# Utility functions
def normalize_image_(image_in):
    output_image = image_in / 255.0
    return output_image

def normalize_image(image_in):
    output_image = np.array([image_in])  # add a dimension: (28 x 28) to (1 x 28 x 28)
    output_image = output_image / 255
    return output_image


def batch_sample_label_separator(batch):
    b_size = len(batch)
    x = []
    y = []
    for i in range(b_size):
        # adding a dimension into input image
        # So: 28x28 -> 1x28x28
        sample = np.reshape(batch[i, 0],(1,28,28))
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

def delta_V_(K, G, i, j, k):
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

def back_conv_(G, K, V_shape):
    # print(V_shape)
    DELTA_V = np.zeros(shape=(V_shape))

    for i in range(DELTA_V.shape[0]):
        for j in range(DELTA_V.shape[1]):
            for k in range(DELTA_V.shape[2]):
                DELTA_V[i][j][k] = delta_V_(K, G, i, j, k)
    return DELTA_V

def batch_back_conv(G, K, V_shape):
    output = []
    for i in range(len(G)):
        output.append(back_conv_(G[i], K, V_shape))
    return np.array(output)

def delta_K_(G, V, K_shape, i, j, k, l, stride=1):
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
                    DELTA_K[i][j][k][l] = delta_K_(G, V, K_shape, i, j, k, l)
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

def relu_(x):
    y = x > 0
    y = y * 1
    output = np.multiply(x, y)
    return output

def back_relu_(gradient, x):
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
    return output

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
    # print('in')
    # print(Y.shape)
    # print(Y_hat.shape)
    # print('out')
    output = np.log(np.matmul(Y, Y_hat) + EPS) * (-1)
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
    batch_s = len(y)
    for i in range(len(y)):
        output.append(back_cross_entropy(y[i], yhat[i]))
    return np.array(output)  / batch_s







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
np.random.seed()

epoch = 100  # epoch
learning_rate = 0.006  # learning rate
lambda_regu = 0.00002  # regularization coefficient
# trainset = trainset[0:3]
def ad(x):
    return np.expand_dims(x, axis=0)

loss_log = []
for e in range(epoch):
    # print(e)
    e_loss = []
    for image, label in trainset:

        one_hot_label = get_one_hot(np.array([label]))

        Image = normalize_image_(image)# Image is used in calculating dJ/dK1. [SHAPE] Image: (1,28,28)
        x = ad(ad(Image))

        O = batch_conv(x, K1)
        O = batch_conv_bias(O, B1)  # E1. [SHAPE] B1: (5,1), E1: (5,24,24)
        U2 = O
        O = relu_(O)  # C1. F1 is used in back-propagate this relu layer. [SHAPE] C1: (5,24,24), F1: (5,24,24)


        C1 = O  # C1 is used in calculating dJ/dK2. [SHAPE] C1: (5,24,24)
        O = batch_conv(O, K2)  # U2. [SHAPE] K2: (10,5,5,5), U2: (10,20,20)
        O = batch_conv_bias(O, B2)  # E2. [SHAPE] B2: (10,1), E2: (10,20,20)
        U5 = O
        O = relu_(O)  # C2. F2 is used in back-propagate this relu layer. [SHAPE] C2: (10,20,20), F2: (10,20,20)

        O, pm = batch_max_pooling(O)  # P1. L1 is used in back-propagate this max pooling layer. [SHAPE] P1: (10,10,10), L1: (10,20,20)
        O, ten_s = tensor_to_vector(O)  # P2. [SHAPE] P2: (1,1000)

        P2 = O  # P2 is used in calculating dJ/dW1. [SHAPE] P2: (1,1000)
        O = MLP(O, W1)  # U3. [SHAPE] W1: (1000,100), U3: (1,100)
        O = MLP_bias(O, B3)  # E3. [SHAPE] B3: (1,100), E3: (1,100)
        U9 = O
        O = relu_(O)  # C3. F3 is used in back-propagate this relu layer. [SHAPE] C3: (1,100), F3: (1,100)

        C3 = O  # C3 is used in calculating dJ/dW2. [SHAPE] C3: (1,100)
        O = MLP(O, W2)  # U4. [SHAPE] W2: (100,10), U4:(1,10)
        O = MLP_bias(O, B4)  # E4. [SHAPE] B4: (1,10), E4: (1,10)
        O = batch_softmax(O)  # Y_hat. [SHAPE] Y_hat: (1,10)

        J = batch_cross_entropy(one_hot_label, O)  # error. [SHAPE] J: one number


        D = batch_back_cross_entropy(one_hot_label, O)  # dJ/dY_hat. [SHAPE] dJ/dY_hat: (1,10)
        D = batch_back_softmax(D, O)


        dJdB4 = MLP_bias_weight_gradient(D)
        dJdW2 = MLP_weight_gradient(D, C3)  # dJ/dW2. Here dU4/dW2 = C3(transpose and go front), then dJ/dW2= dJ/dU4 X dU4/dW2. [SHAPE] dU4/dW2 = C3(transpose): (100,1), dJ/dE4=dJ/dU4: (1,10), dJ/dW2: (100,10)

        D = back_MLP(D, W2)  # dJ/dC3. Here, dU4/dC3 = W2(transpose), then dJ/dC3=dJ/dU4 X dU4/dC3. [SHAPE] dU4/dC3 = W2(transpose): (10,100), dJ/dU4: (1,10), dJ/dC3: (1,100)
        D = back_relu_(D, U9)

        dJdB3 = MLP_bias_weight_gradient(D)
        dJdW1 = MLP_weight_gradient(D, P2)  # dJ/dW1. dU3/dW1 = P2(transpose and go front), then dJ/dW1 = dJ/dU3 X dU3/dW1. [SHAPE] dU3/dW1=P2(transpose): (1000,1), dJ/dU3: (1,100), dJ/dW1: (1000,100)

        D = back_MLP(D, W1)  # dJ/dP2. dJ/dP2 = dJ/dU3 X dU3/dP2, where dU3/dP2=W1(transpose). [SHAPE] dU3/dP2=W1(transpose): (100,1000), dJ/dU3: (1,100), dJ/dP2: (1,1000)
        D = vector_to_tensor(D, ten_s)  # dJ/dP1. dJ/dP1 = dJ/dP2 X dP2/dP1. dP2/dP1 is reshaping operation. P1 offers target tensor shape. [SHAPE] P1: (10,10,10), dJ/dP1: (10,10,10)
        D = batch_back_max_pooling(D, pm)
        D = back_relu_(D, U5)

        dJdB2 = batch_conv_bias_weight_gradient(D)
        dJdK2 = batch_conv_weight_gradient(D, C1, K2.shape)

        D = batch_back_conv(D, K2, C1[0].shape)
        D = back_relu_(D, U2)

        dJdB1 = batch_conv_bias_weight_gradient(D)  # dJ/dB1. dJ/dB1 = dJ/dE1 X dE1/dB1. [SHAPE] dJ/dB1: (5,1)
        dJdK1 = batch_conv_weight_gradient(D, x, K1.shape)


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

        # Updating bias parameter
        scaling_conv_bias = 1.0  # unsure of conv bias layer's back-prop.
        B1 = B1 - learning_rate * dJdB1 * scaling_conv_bias
        B2 = B2 - learning_rate * dJdB2 * scaling_conv_bias
        B3 = B3 - learning_rate * dJdB3
        B4 = B4 - learning_rate * dJdB4
        # break
        e_loss.append(J)

    loss_log.append(np.mean(e_loss))
    print('epoch:{}. loss: {}'.format(e, J))
plt.plot(loss_log)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title(plot_title)
plt.show()