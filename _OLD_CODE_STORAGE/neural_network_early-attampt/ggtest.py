import numpy as np
class NN():
    def __init__(self):
        self.isNew = True
        self.inputlayersize = 0
        self.outputLayersize = 0
    def init(self, xsize, ysize):
        self.inputlayersize = xsize
        self.hiddenlayers = 4
        #self.hiddenlayersize = 6
        self.hiddenlayers_sizes = [5,6,7,4]
        self.hiddenlayer = [0 for i in range(self.hiddenlayers)]
        for i in range(self.hiddenlayers):
            self.hiddenlayer[i] = self.hiddenlayers_sizes[i]
        self.outputLayersize = ysize

        #print(self.hiddenlayer)
        self.stepsize = 0.001
        self.Wxh0 = np.random.random((self.hiddenlayer[0],self.inputlayersize))
        self.Wh0h1 = np.random.random((self.hiddenlayer[1],self.hiddenlayer[0]))
        self.Wh1h2 = np.random.random((self.hiddenlayer[2],self.hiddenlayer[1]))
        self.Wh2h3 = np.random.random((self.hiddenlayer[3],self.hiddenlayer[2]))
        self.Wh3y = np.random.random((self.outputLayersize,self.hiddenlayer[3]))
        '''
        print(self.Wxh0)
        print(self.Wh0h1)
        print(self.Wh1h2)
        print(self.Wh2h3)
        print(self.Wh3y)
        print(self.Bxh0)
        print(self.Bh0h1)
        print(self.Bh1h2)
        print(self.Bh2h3)
        print(self.Bh3y)
        '''
        self.Bxh0 = np.array(([[0.01]*self.hiddenlayer[0]])).T
        self.Bh0h1 = np.array(([[0.01]*self.hiddenlayer[1]])).T
        self.Bh1h2 = np.array(([[0.01]*self.hiddenlayer[2]])).T
        self.Bh2h3 = np.array(([[0.01]*self.hiddenlayer[3]])).T
        self.Bh3y = np.array(([[0.01]*self.outputLayersize])).T

    def forward(self, X, Y):
        if self.isNew or self.inputlayersize != len(X) or self.outputLayersize != len(Y):
            self.init(len(X),len(Y))
            self.isNew = False

        self.Y = np.array(([Y])).T
        self.X = np.array(([X])).T
        self.Hi_0 = np.add( (np.dot(self.Wxh0, self.X)), self.Bxh0 )
        self.Ho_0 = self.activationFunction(self.Hi_0)
        self.Hi_1 = np.add( (np.dot(self.Wh0h1, self.Ho_0)), self.Bh0h1 )
        self.Ho_1 = self.activationFunction(self.Hi_1)
        self.Hi_2 = np.add( (np.dot(self.Wh1h2, self.Ho_1)), self.Bh1h2 )
        self.Ho_2 = self.activationFunction(self.Hi_2)
        self.Hi_3 = np.add( (np.dot(self.Wh2h3, self.Ho_2)), self.Bh2h3 )
        self.Ho_3 = self.activationFunction(self.Hi_3)
        self.Bh3y = self.Bh3y*0
        self.yHat = np.add( (np.dot(self.Wh3y, self.Ho_3)), self.Bh3y )
        #print((self.Y-self.yHat)**2)
        self.backward()

    def backward(self):
        self.delta_Wh3y = (-1)*( np.subtract(self.Y,self.yHat) )
        #self.delta_Wh3y = np.abs( np.subtract(self.Y,self.yHat) )
        self.delta_Wh2h3 = np.dot(self.Wh3y.T,self.delta_Wh3y) * self.d_activationFunction(self.Hi_3)
        self.delta_Wh1h2 = np.dot(self.Wh2h3.T,self.delta_Wh2h3) * self.d_activationFunction(self.Hi_2)
        self.delta_Wh0h1 = np.dot(self.Wh1h2.T,self.delta_Wh1h2) * self.d_activationFunction(self.Hi_1)
        self.delta_Wxh0 = np.dot(self.Wh0h1.T,self.delta_Wh0h1) * self.d_activationFunction(self.Hi_0)

        self.dWh3y = np.dot( self.delta_Wh3y, self.Ho_3.T )
        self.dBh3y = np.abs(self.delta_Wh3y)
        #!!!!!!!!!!!!!!!!!!!!!!WHY WHY WHY WHY???????????????
        #Bh3y didn't go thru activationfunction.,and the number is infuencing too much.
        #self.dBh3y = self.delta_Wh3y
        self.dWh2h3 = np.dot( self.delta_Wh2h3, self.Ho_2.T )
        self.dBh2h3 = self.delta_Wh2h3
        self.dWh1h2 = np.dot( self.delta_Wh1h2, self.Ho_1.T )
        self.dBh1h2 = self.delta_Wh1h2
        self.dWh0h1 = np.dot( self.delta_Wh0h1, self.Ho_0.T )
        self.dBh0h1 = self.delta_Wh0h1
        self.dWxh0 = np.dot( self.delta_Wxh0, self.X.T )
        self.dBxh0 = self.delta_Wxh0

        #print('test:Bh3y {} | dBh3y {}'.format(self.Bh3y,self.dBh3y))
        #self.Bh3y = 0
        self.Wh3y = np.subtract( self.Wh3y, self.stepsize*self.dWh3y )
        self.Bh3y = np.subtract( self.Bh3y, self.stepsize*self.dBh3y )
        self.Wh2h3 = np.subtract( self.Wh2h3, self.stepsize*self.dWh2h3 )
        self.Bh2h3 = np.subtract( self.Bh2h3, self.stepsize*self.dBh2h3 )
        self.Wh1h2 = np.subtract( self.Wh1h2, self.stepsize*self.dWh1h2 )
        self.Bh1h2 = np.subtract( self.Bh1h2, self.stepsize*self.dBh1h2 )
        self.Wh0h1 = np.subtract( self.Wh0h1, self.stepsize*self.dWh0h1 )
        self.Bh0h1 = np.subtract( self.Bh0h1, self.stepsize*self.dBh0h1 )
        self.Wxh0 = np.subtract( self.Wxh0, self.stepsize*self.dWxh0 )
        self.Bxh0 = np.subtract( self.Bxh0, self.stepsize*self.dBxh0 )
        self.Bh3y = self.Bh3y*0

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
        self.Hi_0 = np.add( (np.dot(self.Wxh0, self.X)), self.Bxh0 )
        self.Ho_0 = self.activationFunction(self.Hi_0)
        self.Hi_1 = np.add( (np.dot(self.Wh0h1, self.Ho_0)), self.Bh0h1 )
        self.Ho_1 = self.activationFunction(self.Hi_1)
        self.Hi_2 = np.add( (np.dot(self.Wh1h2, self.Ho_1)), self.Bh1h2 )
        self.Ho_2 = self.activationFunction(self.Hi_2)
        self.Hi_3 = np.add( (np.dot(self.Wh2h3, self.Ho_2)), self.Bh2h3 )
        self.Ho_3 = self.activationFunction(self.Hi_3)
        self.yHat = np.add( (np.dot(self.Wh3y, self.Ho_3)), self.Bh3y )
        print("{} is {} |".format(X,self.yHat.T[0]))
        '''
        print(self.Bxh0)
        print(self.Bh0h1)
        print(self.Bh1h2)
        print(self.Bh2h3)
        print(self.Bh3y)
        '''
nn = NN()

for i in range(5000):
    nn.forward([0,0,0,0],[1])
    nn.forward([1,0,0,0],[2])
    nn.forward([0,1,0,0],[5])
    nn.forward([1,1,0,0],[10])
    nn.forward([1,0,1,0],[26])

nn.test([0,0,0,0])
nn.test([1,0,0,0])
nn.test([0,1,0,0])
nn.test([1,1,0,0])
print("o_O? ",end="")
nn.test([0,0,1,0])
nn.test([1,0,1,0])

print('*/'*20)
for i in range(5000):
    nn.forward([0,0,0],[1])
    nn.forward([0,0,1],[2])
    nn.forward([0,1,0],[3])
    nn.forward([0,1,1],[4])


nn.test([0,0,0])
nn.test([0,0,1])
nn.test([0,1,0])
nn.test([0,1,1])

print('*/'*20)
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
