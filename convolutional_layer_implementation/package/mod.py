

from torchvision import datasets
import numpy as np

# mnist = datasets.MNIST('../storage/mnist/', download=True)
train_set = datasets.MNIST('../storage/mnist/', train=True)
test_set = datasets.MNIST('../storage/mnist/', train=False)


def relu(X):
    matrix_shape = X.shape
    F_matrix = np.zeros(matrix_shape)
    if len(matrix_shape) == 2:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                if X[i][j] > 0:
                    F_matrix[i][j] = 1
    else:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                for k in range(matrix_shape[2]):
                    if X[i][j][k] > 0:
                        F_matrix[i][j][k] = 1
    output = np.multiply(F_matrix, X)
    return output, F_matrix


def Z(K, V, i, j, k, stride):
    k_shape = K.shape
    # kernel_out_channel = k_shape[0]
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
    # print('Convolution layer:')
    # print('[-]K shape:', K.shape)
    # print('[-]V shape:', V.shape)
    output_channel = K.shape[0]
    kernel_channel = K.shape[1]
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]

    input_channel = V.shape[0]
    input_row = V.shape[1]
    input_col = V.shape[2]
    assert (input_channel == kernel_channel), "channel doesn't match!"
    # stride = 1
    output_row = int((input_row - kernel_row) / stride + 1)
    output_col = int((input_col - kernel_col) / stride + 1)
    output = np.zeros(shape=(output_channel, output_row, output_col))
    # print(output.shape)
    # print('[-]output shape:', output.shape)
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i][j][k] = Z(K, V, i, j, k, stride=stride)
    return output


def Pool(V, i, j, k, kernel):
    # print('[--]pooling:')
    values = []
    locations = []
    for m in range(kernel):
        for n in range(kernel):
            row = j * kernel + m
            col = k * kernel + n
            values.append(V[i][row][col])
            locations.append((i, row, col))
    # print('[---]values:', values)
    # print('[---]locations:', locations)
    max_value = max(values)
    ind = values.index(max_value)
    location = locations[ind]
    # print('[---]max:', max_value)
    # print('[---]index:', ind)
    # print('[---]location:', location)
    return max_value, location


def max_pooling(V, kernel=2):
    # print('Max pooling layer:')
    # print('[-]V shape:', V.shape)
    input_channel = V.shape[0]
    input_row = V.shape[1]
    input_col = V.shape[2]

    output_channel = input_channel
    output_row = int(input_row / kernel)
    output_col = int(input_col / kernel)

    output = np.zeros(shape=(output_channel, output_row, output_col))
    # print('[-]output shape:', output.shape)
    pool_matrix = np.zeros(shape=(input_channel, input_row, input_col))
    # print('[-]pool matrix shape:', pool_matrix.shape)
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i][j][k], loc = Pool(V, i, j, k, kernel=kernel)
                pool_matrix[loc[0]][loc[1]][loc[2]] = 1
    return output, pool_matrix


def max_pooling_back(DELTA, pool_matrix, kernel=2):
    # print(DELTA.shape)
    # print(pool_matrix.shape)
    output = np.zeros(pool_matrix.shape)
    input_channel = DELTA.shape[0]
    input_row = DELTA.shape[1]
    input_col = DELTA.shape[2]
    for i in range(input_channel):
        for j in range(input_row):
            for k in range(input_col):
                for m in range(kernel):
                    for n in range(kernel):
                        row = j * kernel + m
                        col = k * kernel + n
                        output[i][row][col] = DELTA[i][j][k]
    output = np.multiply(output, pool_matrix)
    return output


def normalize_image(image_in):
    output_image = np.array(image_in)
    output_image = np.array([output_image])
    output_image = output_image/255
    return output_image


def tensor_to_1d_vector_transpose(tensor):
    # print('Tensor to Vector.T:')
    s = 1
    for i in tensor.shape:
        s = s * i
    output = tensor.reshape((1, s))
    # print('[-]tensor shape:', tensor.shape)
    # print('[-]number of elements:', s)
    # print('[-]output shape:', output.shape)
    return output


def vector_to_tensor(vector, tensor):
    output = vector.reshape(tensor.shape)
    return output


def MLP(X, W):
    # print('Fully connected layer:')
    output = np.matmul(X, W)
    # print('[-]input shape:', X.shape)
    # print('[-]W shape:', W.shape)
    # print('[-]output shape:', output.shape)
    return output


def softmax(X):
    # print('softmax:')
    X_exp = [np.exp(i) for i in X]
    # print(X)
    # print(X_exp)
    sum_exp = np.sum(X_exp)
    # print(sum_exp)
    output = [np.round(i / sum_exp, 4) for i in X_exp]
    output = np.array(output)
    # print('[-]output shape:', output.shape)
    # print('[-]output:', output)
    return output


def predict(Y_hat):

    return np.argmax(Y_hat)


def cross_entropy(Y_hat, Y):
    output = np.log(Y_hat[0][Y]) * (-1)
    output = round(output, 4)
    return output


def Delta_cross_entropy(Y_hat, Y):
    output = np.zeros(shape=(Y_hat.shape))
    # print('output:', output.shape)
    output[0][Y] = -1 * (1 / Y_hat[0][Y])
    # print(output)
    return output


def Delta_softmax(S):
    output_size = len(S[0])
    output = np.zeros(shape=(output_size, output_size))
    for i in range(output_size):
        for j in range(output_size):
            if i == j:
                output[i][j] = S[0][i] * (1 - S[0][j])
            else:
                output[i][j] = (-1) * S[0][i] * S[0][j]
    # print(output)
    return output


def MLP_back(Delta, X, W, matrix_f):
    gradient = np.multiply(matrix_f, Delta)
    Delta_W = np.matmul(X.T, gradient)
    Delta_X = np.matmul(gradient, W.T)
    return Delta_X, Delta_W


def delta_K(G, V, K, i, j, k, l, stride=1):
    kernel_row = K.shape[2]
    kernel_col = K.shape[3]
    output = 0
    for m in range(kernel_row):
        for n in range(kernel_col):
            output += G[i][m][n] * V[j][m*stride+k][n*stride+l]
    return output


def delta_V(K, G, i, j, k, stride=1):
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


def conv_back(DELTA, V, K, relu_matrix):
    G = np.multiply(DELTA, relu_matrix)
    DELTA_K = np.zeros(shape=(K.shape))
    DELTA_V = np.zeros(shape=(V.shape))
    # print(DELTA_K.shape)
    for i in range(DELTA_K.shape[0]):
        for j in range(DELTA_K.shape[1]):
            for k in range(DELTA_K.shape[2]):
                for l in range(DELTA_K.shape[3]):
                    DELTA_K[i][j][k][l] = delta_K(G, V, K, i, j, k, l)

    for i in range(DELTA_V.shape[0]):
        for j in range(DELTA_V.shape[1]):
            for k in range(DELTA_V.shape[2]):
                DELTA_V[i][j][k] = delta_V(K, G, i, j, k)
    return DELTA_V, DELTA_K


def MLP_bias(x, b):
    batch_size = x.shape[0]
    # node_size = b.shape[0]
    m = np.zeros(shape=(batch_size, 1))
    m = m + 1
    mb = np.matmul(m, b)
    y = x + mb
    return y


def MLP_bias_back(gradient):
    batch_size = gradient.shape[0]
    m = np.zeros(shape=(1, batch_size))
    m = m + 1
    y = np.matmul(m, gradient)
    return y


def conv_bias(x, b):
    # z_dimension = x.shape[0]
    y_dimension = x.shape[1]
    x_dimension = x.shape[2]
    m = np.zeros(shape=(1, y_dimension * x_dimension))
    m = m + 1
    mb = np.matmul(b, m)
    mb = mb.reshape(x.shape)
    output = x + mb
    # print(output.shape)
    return output


def conv_bias_back(gradient):
    z_dimension = gradient.shape[0]
    y_dimension = gradient.shape[1]
    x_dimension = gradient.shape[2]
    g = gradient.reshape((z_dimension, y_dimension * x_dimension))
    m = np.zeros(shape=(y_dimension * x_dimension, 1))
    m = m + 1
    output = np.matmul(g, m)
    # print(output.shape)
    return output



adjust = 0.01

K1 = np.random.random(size=(5, 1, 5, 5)) * adjust
K2 = np.random.random(size=(10, 5, 5, 5)) * adjust
W1 = np.random.random(size=(1000, 100)) * adjust
W2 = np.random.random(size=(100, 10)) * adjust

B_K1 = np.zeros(shape=(K1.shape[0], 1))
B_K2 = np.zeros(shape=(K2.shape[0], 1))
B_W1 = np.zeros(shape=(1, W1.shape[1]))
B_W2 = np.zeros(shape=(1, W2.shape[1]))


epoch = 100
l = 0.01
min_error = 999
lambda_regu = 0.02

# for e in range(epoch):
#     for case, (image, label) in enumerate(train_set):
#         Image = normalize_image(image)
#         O = conv(Image, K1)  # U1
#         O = conv_bias(O, B_K1)
#         O, mf1 = relu(O)  # C1
#         C1 = O
#         O = conv(O, K2)  # U2
#         O = conv_bias(O, B_K2)
#         O, mf2 = relu(O)  # C2
#
#         O, mp1 = max_pooling(O)  # P1
#         P1 = O
#         O = tensor_to_1d_vector_transpose(O)  # P2
#         P2 = O
#         O = MLP(O, W1)  # U3
#         O = MLP_bias(O, B_W1)
#         O, mf3 = relu(O)  # F1
#         F1 = O
#
#         O = MLP(O, W2)  # U4
#         O = MLP_bias(O, B_W2)
#         # O, mf4 = relu(O)  # F2
#
#         O = softmax(O)  # Y_hat
#         prediction = predict(O)  # prediction
#         J = cross_entropy(O, label)  # error
#         # print('Prediction:', prediction)
#         # print('Answer:', label)
#         if J < min_error:
#             min_error = J
#         print('Error:[', J, ']  -->Epoch:', e, ' Case:', case, '<--| min:[', min_error, ']')
#         D = Delta_cross_entropy(O, label)  # dJ/DY_hat
#         Temp = Delta_softmax(O)  # dY_hat/DF2
#         # print(Temp.shape)
#         D = np.matmul(D, Temp)
#         del Temp
#         Delta_B_W2 = MLP_bias_back(D)
#         Delta_W2 = np.matmul(F1.T, D)
#         D = np.matmul(D, W2.T)
#         # D, Delta_W2 = MLP_back(D, F1, W2, mf4)
#         # print('DELTA_W2:', Delta_W2.shape)
#         del F1  # , mf4
#         Delta_B_W1 = MLP_bias_back(np.multiply(D, mf3))
#         D, Delta_W1 = MLP_back(D, P2, W1, mf3)
#         # print('DELTA_W1:', Delta_W1.shape)
#         del P2, mf3
#         D = vector_to_tensor(D, P1)
#         del P1
#
#         D = max_pooling_back(D, mp1)
#         # for i in range(10):
#         #     print(mp1[i][6:9, 6:9])
#         del mp1
#         Delta_B_K2 = conv_bias_back(np.multiply(mf2, D))
#         D, Delta_K2 = conv_back(D, C1, K2, mf2)
#         del C1, mf2
#         # print('DELTA_K2:', Delta_K2.shape)
#
#         Delta_B_K1 = conv_bias_back(np.multiply(mf1, D))
#         D, Delta_K1 = conv_back(D, Image, K1, mf1)
#         # print('DELTA_K1:', Delta_K1.shape)
#         # print('Delta shape:', D.shape)
#
#         Delta_W1 += W1 * 2 * lambda_regu
#         Delta_W2 += W2 * 2 * lambda_regu
#         Delta_K1 += K1 * 2 * lambda_regu
#         Delta_K2 += K2 * 2 * lambda_regu
#
#
#         W1 = W1 - l * Delta_W1
#         W2 = W2 - l * Delta_W2
#         K1 = K1 - l * Delta_K1
#         K2 = K2 - l * Delta_K2
#
#         # Delta_B_W1 += B_W1 * 2 * lambda_regu
#         # Delta_B_W2 += B_W2 * 2 * lambda_regu
#         # Delta_B_K1 += B_K1 * 2 * lambda_regu
#         # Delta_B_K2 += B_K2 * 2 * lambda_regu
#
#         B_W1 = B_W1 - l * Delta_B_W1
#         B_W2 = B_W2 - l * Delta_B_W2
#         B_K1 = B_K1 - l * Delta_B_K1
#         B_K2 = B_K2 - l * Delta_B_K2
#         # print(D[1][6:10, 6:10])
#         # print('OUTPUT shape:', O.shape)
#         # print(O)
#
#         # break
#     # break