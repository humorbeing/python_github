import numpy as np
class NN():
    def __init__(self):
        self.inputSize = 3
        self.outputSize = 2
        self.W0 = np.random.random((5,3,3))
        self.W1 = np.random.random((6,3,3))
        self.W20 = np.random.random((4,2,3))
        self.W21 = np.random.random((4,3,2))
        self.W3 = np.random.random((50,16))
        self.W4 = np.random.random((2,50))

    def forward(self, X, Y):
        self.X = np.array([X])
        self.Y = np.array([Y]).T
        self.L0 = self.activationFunction( self.M(self.W0,self.X) )
        self.L1 = self.activationFunction( self.M(self.W1,self.L0) )
        self.L2 = self.activationFunction( self.M( ( self.M( (self.W20),(self.L1) ) ),(self.W21) ) )
        self.L3 = self.activationFunction( self.F(self.W3,self.L2) )
        self.L4 = self.activationFunction( np.dot(self.W4,self.L3))
        print(self.L4)
        self.backward()
    def backward(self):
        pass
    def activationFunction(self, X):
        return np.maximum(0.01*X,X)
        #return 1/(1+np.exp(-X))
    def d_activationFunction(self, X):
        pass
    def M(self, X, Y):
        tem = []
        summ = [[0.0 for j in range(len(Y[0][0]))] for i in range(len(X[0]))]
        for i in X:
            for x in Y:
                summ = np.add(summ,np.dot(i,x))
            tem.append(summ)
            summ = summ*0
        return np.array(tem)

    def F(self, X, Y):
        tem = []
        for i in Y:
            for j in i:
                for k in j:
                    tem.append(k)
        return np.dot(X,(np.array(tem)))
    def test(self, X):
        pass
nn = NN()
X = np.array(([[0,1,0],[0,1,0],[0,1,0]]))
nn.forward(X,[0,1])
