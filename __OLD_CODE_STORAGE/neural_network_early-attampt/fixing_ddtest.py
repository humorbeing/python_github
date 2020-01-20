import numpy as np
class NN():
    def __init__(self):
        self.inputlayersize = 2
        self.outputLayersize = 1
        #self.w0 = np.random.random()
        #self.w1 = np.random.random()
        #self.w2 = np.random.random()
        self.W = np.random.random((1,2))
        self.step = 0.05
    def matrix_relu(self, x):
        columnInARow = []
        Row = []
        for i in x:
            columnInARow = []
            for j in i:
                if j > 0:
                    columnInARow.append(j)
                else:
                    columnInARow.append(0)
            Row.append(aa)
        return np.array(Row)
    def relu(self, x):
        return x
        '''
        if x > 0:
            return x
        else:
            return 0
        '''
    def err(self):
        return 1/2*((self.y - self.yHat)**2)

    def forward(self, X, y):

        self.X = np.array([X]).T
        self.node_II_0out = np.dot(self.W, self.X)
        self.yHat = self.relu(self.node_II_0out)
        self.y = np.array([[y]])


        return self.err()

    def backward(self):
        # if self.node_II_0out > 0:
        #     dedw = np.array((np.array(([self.X])).T) * (-1) * (self.y - self.yHat) * (1))
        #
        #     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!![1,1] [[1,1]]
        # else:
        #     dedw = np.array([[0],[0]])
        dedw = np.array((np.array(([self.X])).T) * (-1) * (self.y - self.yHat) * (1))

        return dedw

    def train(self):
        dw = self.backward()

        self.W = np.subtract(self.W,self.step*dw)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(self.W)


    def showw(self):
        print("W is {}.".format(self.W))
    def test(self, X):
        print("{} to {}".format(X,self.relu(np.dot(self.W, np.array(X).T))))
nn = NN()
#for i in range()
for i in range(1):

    nn.forward([0,0],0)
    nn.train()

    nn.forward([0,1],1)
    nn.train()

    nn.forward([1,0],2)
    nn.train()
    nn.forward([1,1],3)
    nn.train()


nn.showw()

nn.test([0,0])
nn.test([0,1])
nn.test([1,0])
nn.test([1,1])
