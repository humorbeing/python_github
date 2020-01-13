import numpy as np
X = np.array(([3,5],[5,1],[10,2]),dtype=float)
Y = np.array(([75],[82],[93]),dtype=float)
#X[1] = [3.,4.]
#print(X[1])
#test 8,3
#scale
X = X/np.amax(X, axis=0)
Y = Y/100
#print(X)
#print(Y)

class Neural_Network(object):
    def __init__(self):
        #define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #weights (parameters)
        self.W1 = np.random.random((self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random((self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):
        #progagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

NN = Neural_Network()
yHat = NN.forward(X)

#print(Y)
#print(yHat)

def sigmoidPrime(z):
    #derivative of sigmoid Function
    return np.exp(-z)/((1+np.exp(-z))**2)
