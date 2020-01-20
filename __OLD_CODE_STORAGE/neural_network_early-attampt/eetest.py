import numpy as np
class NN():
    def __init__(self):
        self.inputlayersize = 2
        self.hiddenlayersize = 6#!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.outputLayersize = 2
        #self.w0 = np.random.random()
        #self.w1 = np.random.random()
        #self.w2 = np.random.random()
        self.Wxh = np.random.random((self.hiddenlayersize,2))
        self.Why = np.random.random((2,self.hiddenlayersize))
        self.step = 0.05
    def matrix_relu(self, x):
        columnInARow = []
        Row = []
        for i in x:
            columnInARow = []
            #print(x)
            #print(i)
            for j in i:
                if j > 0:
                    columnInARow.append(j)
                else:
                    columnInARow.append(0.01)
            Row.append(columnInARow)
        return np.array(Row)
    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def err(self):
        erro = 0
        #for i in range(2):
        #    erro += 1/2*((self.Y[i] - self.yHat[i])**2)
        return erro
    def savezero(self, X):
        z = []
        if X[0] == 0:
            z.append(0.0001)
        else:
            z.append(X[0])
        if X[1] == 0:
            z.append(0.0001)
        else:
            z.append(X[1])
        #print(X)
        #print(z)
        return np.array(z)

    def forward(self, X, Y):
        self.Y = np.array(([Y])).T#!!!!!!!!!!!!!
        #print(Y)
        #print(self.Y)
        X = self.savezero(X)
        self.X = np.array(([X])).T#!!!!!!!!!!!!!!

        self.Hi = np.dot(self.Wxh, self.X)

        #print(self.Wxh)
        #print(self.X)
        #print(self.Hi)
        self.Ho = self.matrix_relu(self.Hi)



        self.yHat = np.dot(self.Why, self.Ho)
        #print(self.Why)
        #print(self.Ho)
        #print(self.yHat)
        #print(np.subtract(self.Y,self.yHat))
        return self.err()

    def gWxh(self):
        dWxh =   np.dot( (np.dot( (np.array(self.Why).T),( (-1)*( np.subtract(self.Y,self.yHat) )))*(1) ), (np.array((self.X)).T))
        for i in range(len(self.Hi)):
            if self.Hi[i][0]>0:
                pass
            else:
                #print(dWxh)
                #print(dWxh[i])
                #print(np.array(([0,0])))
                dWxh[i]=np.array(([0.0001,0.0001]))

        #print(dWxh)

        return dWxh

    def backward(self):
        dWhy = (-1)*(np.dot((np.subtract(self.Y,self.yHat)*(1)),np.array((self.Ho)).T))
        dWxh = self.gWxh()#!!!!dont' forget self.
        return dWhy,dWxh



    def train(self):
        dWhy ,dWxh= self.backward()
        #print(dWhy)
        #print(self.Why)
        self.Why = np.subtract(self.Why,self.step*dWhy)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print('-'*10)
        #print(np.subtract(self.Why,self.step*dWhy))
        #print('-'*10)
        #print(self.Why)
        #print('-'*20)
        #print(self.Wxh)
        #print('='*10)
        self.Wxh = np.subtract(self.Wxh,self.step*dWxh)
        #print(self.Wxh)
        #print('='*20)


    def showw(self):
        print("Wxh is {}.".format(self.Wxh))
        print("Why is {}.".format(self.Why))
    def test(self, X):
        X = np.array(([X])).T#!!!!!!!!!!!!!!

        Hi = np.dot(self.Wxh, X)

        #print(self.Wxh)
        #print(self.X)
        #print(self.Hi)
        Ho = self.matrix_relu(Hi)



        yHat = np.dot(self.Why, Ho)
        print("{} to {}".format(X.T,yHat.T))

nn = NN()
#for i in range()
for i in range(5000):

    nn.forward([0,0],[0,1])
    nn.train()

    nn.forward([0,1],[1,2])
    nn.train()

    nn.forward([1,0],[2,3])
    nn.train()
    nn.forward([1,1],[3,4])
    nn.train()


#nn.showw()

nn.test([0,0])
nn.test([0,1])
nn.test([1,0])
nn.test([1,1])
