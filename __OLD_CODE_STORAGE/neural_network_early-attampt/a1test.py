#testing last bias to be viable.
import numpy as np
class NN():
    def __init__(self, node = 3, layer = 3):
        self.isNew = True
        self.inputlayersize = 0
        self.outputLayersize = 0
        self.stepsize = 0.02
        self.S = node
        self.N = layer
        self.L = [[] for i in range(self.N)]
        self.W = [[] for i in range(self.N)]
        self.B = [[] for i in range(self.N)]
        self.delta_W = [[] for i in range(self.N)]
        self.dW = [[] for i in range(self.N)]
        self.dB = [[] for i in range(self.N)]

    def init(self, xsize, ysize):
        self.inputlayersize = xsize
        self.outputLayersize = ysize
        Lsize = []
        for i in range(self.N):
            if i == (self.N-1):
                Lsize.append(ysize)
            else:
                Lsize.append(self.S)
        for i in range(self.N):
            if i == 0:
                self.W[i] = np.random.random((Lsize[i],xsize))
            else:
                self.W[i] = np.random.random((Lsize[i],Lsize[i-1]))
            self.B[i] = np.array(([[0.01]*Lsize[i]])).T

    def forward(self, X, Y):
        if self.isNew or self.inputlayersize != len(X) or self.outputLayersize != len(Y):
            self.init(len(X),len(Y))
            self.isNew = False

        self.Y = np.array(([Y])).T
        self.X = np.array(([X])).T

        for i in range(self.N):
            if i == 0:
                self.L[i] = self.activationFunction(np.add( (np.dot(self.W[i], self.X)), self.B[i] ))
            else:
                self.L[i] = self.activationFunction(np.add( (np.dot(self.W[i], self.L[i-1])), self.B[i] ))
        self.backward()

    def backward(self):
        for i in range(self.N):
            n = self.N-1-i
            if n == self.N-1:
                self.delta_W[n] = (-1)*( np.subtract(self.Y,self.L[self.N-1]) ) * self.d_activationFunction(np.add( (np.dot(self.W[n], self.L[n-1])), self.B[n] ))
            elif n == 0:
                self.delta_W[n] = np.dot( (self.W[n+1].T), self.delta_W[n+1] ) * self.d_activationFunction(np.add( (np.dot(self.W[n], self.X)), self.B[n] ))
            else:
                self.delta_W[n] = np.dot( (self.W[n+1].T), self.delta_W[n+1] ) * self.d_activationFunction(np.add( (np.dot(self.W[n], self.L[n-1])), self.B[n] ))

        for i in range(self.N):
            if i == 0:
                self.dW[i] = np.dot( self.delta_W[i], self.X.T )
            else:
                self.dW[i] = np.dot( self.delta_W[i], self.L[i-1].T )
            self.dB[i] = self.delta_W[i]

        for i in range(self.N):
            self.W[i] = np.subtract( self.W[i], self.stepsize*self.dW[i] )
            self.B[i] = np.subtract( self.B[i], self.stepsize*self.dB[i] )

    def activationFunction(self, X):
        return np.maximum(0.01*X,X)

    def d_activationFunction(self, X):
        tem = []
        for i in X:
            if i[0]>0:
                tem.append(1)
            else:
                tem.append(0.01)
        return np.array(([tem])).T

    def test(self,X):
        self.X = np.array(([X])).T
        for i in range(self.N):
            if i == 0:
                self.L[i] = self.activationFunction(np.add( (np.dot(self.W[i], self.X)), self.B[i] ))
            else:
                self.L[i] = self.activationFunction(np.add( (np.dot(self.W[i], self.L[i-1])), self.B[i] ))
        print("{} is {} |".format(X,self.L[self.N-1].T[0]))

nn = NN()

for i in range(5000):
    nn.forward([0,0,0,0],[0])
    nn.forward([0,0,0,1],[1])
    nn.forward([0,0,1,0],[2])
    nn.forward([0,0,1,1],[3])
    nn.forward([0,1,0,0],[4])

nn.test([0,0,0,0])
nn.test([0,0,0,1])
nn.test([0,0,1,0])
nn.test([0,0,1,1])
nn.test([0,1,0,0])

for i in range(5000):
    nn.forward([0,0,0,0],[0,1])
    nn.forward([1,0,0,0],[1,2])
    nn.forward([0,1,0,0],[2,3])
    nn.forward([1,1,0,0],[3,4])
    nn.forward([1,0,1,0],[5,6])

nn.test([0,0,0,0])
nn.test([1,0,0,0])
nn.test([0,1,0,0])
nn.test([1,1,0,0])
nn.test([0,0,1,0])#print("o_O? ",end="")
nn.test([1,0,1,0])

for i in range(5000):
    nn.forward([0,0,0],[1])
    nn.forward([0,0,1],[2])
    nn.forward([0,1,0],[3])
    nn.forward([0,1,1],[4])

nn.test([0,0,0])
nn.test([0,0,1])
nn.test([0,1,0])
nn.test([0,1,1])

for i in range(5000):
    nn.forward([0,0,0],[0,1])
    nn.forward([0,0,1],[1,2])
    nn.forward([0,1,0],[2,3])
    nn.forward([0,1,1],[3,4])
    nn.forward([1,0,0],[4,5])

nn.test([0,0,0])
nn.test([0,0,1])
nn.test([0,1,0])
nn.test([0,1,1])
nn.test([1,0,0])
