import numpy as np


class NN(object):
    def __init__(self):
        self.c = 0
        self.yHat = 0
        self.y = 0
        self.x0 = 0
        self.x1 = 0
        self.w0 = np.random.random()
        self.w1 = np.random.random()
        self.step = 0.05

    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def error_function(self):
        return 1/2*((self.y - self.yHat)**2)

    def forward(self, x0, x1, y):
        self.c = self.w0*x0 + self.w1*x1
        self.yHat = self.relu(self.c)
        self.y = y
        self.x0 = x0
        self.x1 = x1
        return self.error_function()

    def backward(self):
        if self.c > 0:
            dedw0 = (-1)*(self.y - self.yHat)*1*self.x0
            dedw1 = (-1)*(self.y - self.yHat)*1*self.x1
        else:
            dedw0 = 0
            dedw1 = 0
        return dedw0, dedw1

    def train(self):
        dw0, dw1 = self.backward()
        self.w0 -= self.step*dw0
        self.w1 -= self.step*dw1

    def show_weight(self):
        print("w0 is {}, w1 is {}.".format(self.w0, self.w1))

    def test(self, x0, x1):
        print("[{},{}] to {}".format(x0, x1, self.relu(self.w0*x0 + self.w1*x1)))
nn = NN()

for i in range(5000):
    nn.forward(0, 0, 0)
    nn.train()
    nn.forward(0, 1, 1)
    nn.train()
    nn.forward(1, 0, 2)
    nn.train()
    nn.forward(1, 1, 3)
    nn.train()

nn.show_weight()
nn.test(0, 0)
nn.test(0, 1)
nn.test(1, 0)
nn.test(1, 1)
