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
    print('Convolution layer:')
    print('[-]K shape:', K.shape)
    print('[-]V shape:', V.shape)
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
    print('[-]output shape:', output.shape)
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
    print('Max pooling layer:')
    print('[-]V shape:', V.shape)
    input_channel = V.shape[0]
    input_row = V.shape[1]
    input_col = V.shape[2]

    output_channel = input_channel
    output_row = int(input_row / kernel)
    output_col = int(input_col / kernel)

    output = np.zeros(shape=(output_channel, output_row, output_col))
    print('[-]output shape:', output.shape)
    pool_matrix = np.zeros(shape=(input_channel, input_row, input_col))
    print('[-]pool matrix shape:', pool_matrix.shape)
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i][j][k], loc = Pool(V, i, j, k, kernel=kernel)
                pool_matrix[loc[0]][loc[1]][loc[2]] = 1
    return output, pool_matrix


def normalize_image(image_in):
    output_image = np.array(image_in)
    output_image = np.array([output_image])
    output_image = output_image/255
    return output_image


K1 = np.random.random(size=(5, 1, 5, 5))
K2 = np.random.random(size=(10, 5, 5, 5))


for image, label in train_set:
    Image = normalize_image(image)
    O = conv(Image, K1)  # U1
    O, mf1 = relu(O)  # C1

    O = conv(O, K2)  # U2
    O, mf2 = relu(O)  # C2

    O, mp1 = max_pooling(O)  # P1
    print(O.shape)
    break