import torch
from torch.autograd import Variable
import numpy as np


def relu(X):
    matrix_shape = X.size()
    # print(len(matrix_shape))
    # print(matrix_shape[0])
    F_matrix = Variable(torch.zeros(matrix_shape))
    if len(matrix_shape) == 2:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                if X[i][j] > 0:
                    F_matrix[i, j] = 1
    else:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                for k in range(matrix_shape[2]):
                    # print((X[i][j][k] > 0).data.numpy())
                    if (X[i][j][k] > 0).data.numpy():
                        F_matrix[i, j, k] = 1
    output = torch.mul(F_matrix, X)
    return output  # , F_matrix


def tensor_vector(x):
    output = x.view(1, -1)
    return output


def Z(K, V, i, j, k, stride):
    k_shape = K.size()
    # kernel_out_channel = k_shape[0]
    kernel_channel = k_shape[1]
    kernel_row = k_shape[2]
    kernel_column = k_shape[3]
    z = 0
    for l in range(kernel_channel):
        for m in range(kernel_row):
            for n in range(kernel_column):
                z += V[l][j*stride+m][k*stride+n] * K[i][l][m][n]
    # print('[-]', type(z), z)
    return z


def convolution(V, K, stride=1):
    # print('Convolution layer:')
    # print('[-]K shape:', K.shape)
    # print('[-]V shape:', V.shape)
    output_channel = K.size()[0]
    kernel_channel = K.size()[1]
    kernel_row = K.size()[2]
    kernel_col = K.size()[3]

    input_channel = V.size()[0]
    input_row = V.size()[1]
    input_col = V.size()[2]
    assert (input_channel == kernel_channel), "channel doesn't match!"
    # stride = 1
    output_row = int((input_row - kernel_row) / stride + 1)
    output_col = int((input_col - kernel_col) / stride + 1)
    output = Variable(torch.zeros(output_channel, output_row, output_col))
    # print(output.shape)
    # print('[-]output shape:', output.shape)
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i, j, k] = Z(K, V, i, j, k, stride=stride)
    return output


def conv_bias(x, b):
    # z_dimension = x.shape[0]
    y_dimension = x.size()[1]
    x_dimension = x.size()[2]
    m = torch.zeros(1, y_dimension * x_dimension)
    m = m + 1
    m = Variable(m)
    mb = torch.matmul(b, m)
    mb = mb.view(x.size())
    output = x + mb
    # print(output.shape)
    return output


def MLP_bias(x, b):
    batch_size = x.size()[0]
    # node_size = b.shape[0]
    m = torch.zeros(batch_size, 1)
    m = m + 1
    m = Variable(m)
    mb = torch.matmul(m, b)
    y = x + mb
    return y


def softmax(X):
    # print('softmax:')
    X_exp = torch.exp(X)
    # sss = 0
    # for i in X:
    #     sss += torch.exp(i)
    # print(X)
    # print(X_exp)
    sum_exp = torch.sum(X_exp)
    # print(sum_exp)
    output = X_exp / sum_exp
    # output = Variable(torch.from_numpy(np.array(output)))
    # output = np.array(output)
    # print('[-]output shape:', output.shape)
    # print('[-]output:', output)
    return output


def cross_entropy(Y_hat, Y):
    output = torch.log(Y_hat[0][Y]) * (-1)
    # output = round(output, 4)
    return output


def normalize_image(image_in):
    output_image = np.array(image_in)
    output_image = np.array([output_image])
    output_image = output_image/255
    output_image = torch.from_numpy(output_image)
    # output_image = Variable(output_image)
    return output_image


def Pool(V, i, j, k, kernel):
    # print('[--]pooling:')
    values = Variable(torch.zeros(kernel, kernel))
    # locations = []
    for m in range(kernel):
        for n in range(kernel):
            row = j * kernel + m
            col = k * kernel + n
            # print(V[i][row][col])
            values[m, n] = V[i][row][col]
            # locations.append((i, row, col))
    # print('[---]values:', values)
    # print('[---]locations:', locations)
    # values = torch.from_numpy(np.array(values))
    max_value = torch.max(values)
    # ind = values.index(max_value)
    # location = locations[ind]
    # print('[---]max:', max_value)
    # print('[---]index:', ind)
    # print('[---]location:', location)
    return max_value


def max_pooling(V, kernel=2):
    # print('Max pooling layer:')
    # print('[-]V shape:', V.shape)
    input_channel = V.size()[0]
    input_row = V.size()[1]
    input_col = V.size()[2]

    output_channel = input_channel
    output_row = int(input_row / kernel)
    output_col = int(input_col / kernel)

    output = Variable(torch.zeros(output_channel, output_row, output_col))
    # print('[-]output shape:', output.shape)
    # pool_matrix = np.zeros(shape=(input_channel, input_row, input_col))
    # print('[-]pool matrix shape:', pool_matrix.shape)
    for i in range(output_channel):
        for j in range(output_row):
            for k in range(output_col):
                output[i, j, k] = Pool(V, i, j, k, kernel=kernel)
                # pool_matrix[loc[0]][loc[1]][loc[2]] = 1
    return output
