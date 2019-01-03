import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[2],[3]])
class NN():
    def __init__(self):
        self.inputlayersize = 2
        self.A_layersize = 2
        self.Z_layersize =1
        self.W__A = 2*np.random.random((2,2))
        self.W_AZ = 2*np.random.random((2,1))
    def relu(z):
        if z>0:
            return z
        else:
            return 0
    # def forward(X):
    #     # I_AZ = ztoz(np.dot(X,W__A))
    #     # z = np.dot(I_AZ,W_AZ)
    #     return z
    #
    # def backward(Y):
    #     ab = ifs
