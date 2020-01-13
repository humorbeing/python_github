''' By: From VB lab
    This is a learning attempt for mathematics and mechanism
involved in CNN. It has far more bugs and problems to be considered as
an applicable implementation. But as a pure numpy based and able to train on mnist dataset,
its a good CNN implementation for study purpose.
    Experiment: So, during training, we can observe the loss value is
dropping. And model can be overfitted with small number of training samples.
    I am very certain that there are some "wrong" codes out there in this code.
Considering all these dancing among multi dimension manipulations, its almost
certain, that I have made some mistakes somewhere. The question is, "Are you good
enough to find them?", "Do you understand CNN well enough to spot them?"

- no speed concern (need 10 days to run one epoch of 60000 mnist)
- no testing (evaluated by observing training loss value)
- no normalization or standardization
- no batch (one by one training, so its a SGD basically)
- no overflow or divide by 0 protection (eventually will fail)
'''
import numpy as np  # One and only import. This is Numpy-based, pure math implementation.
'''
# some interesting implementations: easy to hard
# back-prop relu layer using matrix operation (idea only)
# back-prop max pooling layer using matrix operation (idea only)
# back-prop conv bias layer using matrix operation (idea only)
# forward-prop conv layer using matrix operation
# back-prop conv layer using matrix operation
'''
trainset = np.load('./simpleset_1per.npy')  # 1 image per class, total 10 images
# trainset = np.load('./simpleset_10per.npy')  # 10 image per class, total 100 images

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


def max_pooling_back(DELTA, pool_matrix, kernel=2):
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
    output_image = np.array([image_in])  # add a dimension: (28 x 28) to (1 x 28 x 28)
    output_image = output_image / 255
    return output_image


def tensor_to_1d_vector_transpose(tensor):
    s = 1
    for i in tensor.shape:
        s = s * i
    output = tensor.reshape((1, s))
    return output


def vector_to_tensor(vector, tensor):
    output = vector.reshape(tensor.shape)
    return output


def MLP(X, W):
    output = np.matmul(X, W)
    return output


def softmax(X):
    X_exp = [np.exp(i) for i in X]
    sum_exp = np.sum(X_exp)
    output = [np.round(i / sum_exp, 4) for i in X_exp]
    output = np.array(output)
    return output


def predict(Y_hat):
    return np.argmax(Y_hat)


def cross_entropy(Y_hat, Y):
    output = np.log(Y_hat[0][Y] + 1e-5) * (-1)
    output = round(output, 4)
    return output


def Delta_cross_entropy(Y_hat, Y):
    output = np.zeros(shape=(Y_hat.shape))
    # print('output:', output.shape)
    output[0][Y] = -1 * (1 / (Y_hat[0][Y] + 1e-5))
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
    return output


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


def conv_back(G, V, K):
    DELTA_K = np.zeros(shape=(K.shape))
    DELTA_V = np.zeros(shape=(V.shape))
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
    m = np.ones(shape=(batch_size, 1))
    mb = np.matmul(m, b)
    y = x + mb
    return y


def MLP_bias_back(gradient):
    batch_size = gradient.shape[0]
    m = np.ones(shape=(1, batch_size))
    y = np.matmul(m, gradient)
    return y


def conv_bias(x, b):
    y_dimension = x.shape[1]
    x_dimension = x.shape[2]
    m = np.ones(shape=(1, y_dimension * x_dimension))
    mb = np.matmul(b, m)
    mb = mb.reshape(x.shape)
    output = x + mb
    return output

# conv_bias_back MIGHT BE a wrong implementation.
# even though model's training process works as intended
# idea: add up all gradient in a certain channel(z dimension).
# every point in the same channel(z dimension) is added by the same bias.
# concern is, the graident MIGHT BE too big, since they are all added up.
def conv_bias_back(gradient):
    z_dimension = gradient.shape[0]
    y_dimension = gradient.shape[1]
    x_dimension = gradient.shape[2]
    g = gradient.reshape((z_dimension, y_dimension * x_dimension))
    m = np.ones(shape=(y_dimension * x_dimension, 1))
    output = np.matmul(g, m)
    return output


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


epoch = 100  # epoch
l = 0.006  # learning rate
lambda_regu = 0.00002  # regularization coefficient

for e in range(epoch):
    errors = []
    for image, label in trainset:

        # - - - Start of forward propagation - - -
        # Convolution 1
        Image = normalize_image(image)  # Image is used in calculating dJ/dK1. [SHAPE] Image: (1,28,28)
        O = conv(Image, K1)  # U1. [SHAPE] K1: (5,1,5,5), U1: (5,24,24)
        O = conv_bias(O, B1)  # E1. [SHAPE] B1: (5,1), E1: (5,24,24)
        O, F1 = relu(O)  # C1. F1 is used in back-propagate this relu layer. [SHAPE] C1: (5,24,24), F1: (5,24,24)

        # Convolution 2
        C1 = O  # C1 is used in calculating dJ/dK2. [SHAPE] C1: (5,24,24)
        O = conv(O, K2)  # U2. [SHAPE] K2: (10,5,5,5), U2: (10,20,20)
        O = conv_bias(O, B2)  # E2. [SHAPE] B2: (10,1), E2: (10,20,20)
        O, F2 = relu(O)  # C2. F2 is used in back-propagate this relu layer. [SHAPE] C2: (10,20,20), F2: (10,20,20)

        # Pooling 1 and reshaping
        O, L1 = max_pooling(O)  # P1. L1 is used in back-propagate this max pooling layer. [SHAPE] P1: (10,10,10), L1: (10,20,20)
        P1 = O  # [SHAPE] P1: (10,10,10)
        O = tensor_to_1d_vector_transpose(O)  # P2. [SHAPE] P2: (1,1000)

        # MLP 1
        P2 = O  # P2 is used in calculating dJ/dW1. [SHAPE] P2: (1,1000)
        O = MLP(O, W1)  # U3. [SHAPE] W1: (1000,100), U3: (1,100)
        O = MLP_bias(O, B3)  # E3. [SHAPE] B3: (1,100), E3: (1,100)
        O, F3 = relu(O)  # C3. F3 is used in back-propagate this relu layer. [SHAPE] C3: (1,100), F3: (1,100)

        # MLP 2
        C3 = O  # C3 is used in calculating dJ/dW2. [SHAPE] C3: (1,100)
        O = MLP(O, W2)  # U4. [SHAPE] W2: (100,10), U4:(1,10)
        O = MLP_bias(O, B4)  # E4. [SHAPE] B4: (1,10), E4: (1,10)
        # O, mf4 = relu(O)  # !!! NOTICE: NO relu activation function for last layer

        # softmax
        O = softmax(O)  # Y_hat. [SHAPE] Y_hat: (1,10)
        # prediction = predict(O)  # prediction. [SHAPE] prediction: one number
        # - - - End of forward propagation - - -

        # - - - Start of back propagation - - -
        # loss function
        J = cross_entropy(O, label)  # error. [SHAPE] J: one number
        errors.append(J)  # training progress logging purpose

        D = Delta_cross_entropy(O, label)  # dJ/dY_hat. [SHAPE] dJ/dY_hat: (1,10)
        Temp = Delta_softmax(O)  # dY_hat/dE4. [SHAPE] dY_hat/dE4: (10,10)
        D = np.matmul(D, Temp)  # dJ/dE4. [SHAPE] dJ/dE4: (1,10)

        # !!! About "del": The purpose for del is not save memory space. It is a checking test, to see if I understand which matrix or gradient is needed until which point.
        del Temp

        Delta_B4 = MLP_bias_back(D)  # dJ/dB4. [SHAPE] dJ/dB4: (1,10)
        Delta_W2 = np.matmul(C3.T, D)  # dJ/dW2. Here dU4/dW2 = C3(transpose and go front), then dJ/dW2= dJ/dU4 X dU4/dW2. [SHAPE] dU4/dW2 = C3(transpose): (100,1), dJ/dE4=dJ/dU4: (1,10), dJ/dW2: (100,10)
        D = np.matmul(D, W2.T)  # dJ/dC3. Here, dU4/dC3 = W2(transpose), then dJ/dC3=dJ/dU4 X dU4/dC3. [SHAPE] dU4/dC3 = W2(transpose): (10,100), dJ/dU4: (1,10), dJ/dC3: (1,100)
        del C3

        D = np.multiply(D, F3)  # dJ/dE3. Here, its element wise multiplication. dJ/dE3 = dJ/dC3 X dC3/dE3, where dC3/dE3 is F3 with element wise multiply gradient. [SHAPE] F3: (1,100), dJ/dC3: (1,100), dJ/dE3=dJ/dU3: (1,100)
        Delta_B3 = MLP_bias_back(D)  # dJ/dB3. [SHAPE] dJ/dB3: (1,100)
        Delta_W1 = np.matmul(P2.T, D)  # dJ/dW1. dU3/dW1 = P2(transpose and go front), then dJ/dW1 = dJ/dU3 X dU3/dW1. [SHAPE] dU3/dW1=P2(transpose): (1000,1), dJ/dU3: (1,100), dJ/dW1: (1000,100)
        D = np.matmul(D, W1.T)  # dJ/dP2. dJ/dP2 = dJ/dU3 X dU3/dP2, where dU3/dP2=W1(transpose). [SHAPE] dU3/dP2=W1(transpose): (100,1000), dJ/dU3: (1,100), dJ/dP2: (1,1000)
        del P2, F3

        D = vector_to_tensor(D, P1)  # dJ/dP1. dJ/dP1 = dJ/dP2 X dP2/dP1. dP2/dP1 is reshaping operation. P1 offers target tensor shape. [SHAPE] P1: (10,10,10), dJ/dP1: (10,10,10)
        del P1

        D = max_pooling_back(D, L1)  # dJ/dC2. dJ/dC2 = dJ/dP1 X dP1/dC2. dP1/dC2 is upsampling gradient with pooling stride and applying max pooling mask L1. [SHAPE] L1: (10,20,20), dJ/dC2: (10,20,20)
        del L1

        D = np.multiply(F2, D)  # dJ/dE2=dJ/U2. dJ/dE2 = dJ/dC2 X dC2/dE2. dC2/dE2 is F2 relu mask apply on gradient. [SHAPE] F2: (10,20,20), dJ/dE2=dJ/dU2: (10,20,20)
        Delta_B2 = conv_bias_back(D)  # dJ/dB2. dJ/dB2 = dJ/dE2 X dE2/dB2. [SHAPE] dJ/dB2: (10,1)
        D, Delta_K2 = conv_back(D, C1, K2)  # dJ/dC1 and dJ/dK2. Its all math, check "deep learning" book. [SHAPE] C1: (5,24,24), K2: (10,5,5,5), dJ/dC1: (5,24,24), dJ/dK2: (10,5,5,5)
        del C1, F2
        D = np.multiply(F1, D)  # dJ/dE1=dJ/dU1. dJ/E1=dJ/C1 X dC1/dE1. dC1/dE1 is F1 relu mask apply on gradient. [SHAPE] F1: (5,24,24), dJ/dE1=dJ/dU1: (5,24,24)
        Delta_B1 = conv_bias_back(D)  # dJ/dB1. dJ/dB1 = dJ/dE1 X dE1/dB1. [SHAPE] dJ/dB1: (5,1)
        _, Delta_K1 = conv_back(D, Image, K1)  # dJ/dK1. [SHAPE] dJ/dK1: (5,1,5,5)
        # - - - End of back propagation - - -

        # This is L2 regularization.
        Delta_K1 += K1 * 2 * lambda_regu
        Delta_K2 += K2 * 2 * lambda_regu
        Delta_W1 += W1 * 2 * lambda_regu
        Delta_W2 += W2 * 2 * lambda_regu

        # Updating kernel parameter
        K1 = K1 - l * Delta_K1
        K2 = K2 - l * Delta_K2
        W1 = W1 - l * Delta_W1
        W2 = W2 - l * Delta_W2

        # DO NOT apply regularization on bias term
        # Delta_B1 += B1 * 2 * lambda_regu
        # Delta_B2 += B2 * 2 * lambda_regu
        # Delta_B3 += B3 * 2 * lambda_regu
        # Delta_B4 += B4 * 2 * lambda_regu

        # Updating bias parameter
        scaling_conv_bias = 1  # unsure of conv bias layer's back-prop.
        B1 = B1 - l * Delta_B1 * scaling_conv_bias
        B2 = B2 - l * Delta_B2 * scaling_conv_bias
        B3 = B3 - l * Delta_B3
        B4 = B4 - l * Delta_B4

        # break
        # by breaking here, train model with same first sample over and over.
        # overfitting model. good way of testing model is capable of learning.
    print('Epoch: {}, Mean Loss: {}'.format(e, np.array(errors).mean()))
