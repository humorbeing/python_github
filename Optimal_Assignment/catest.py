import numpy as np#just to formated print
class hungarian:
    def __init__(self):
        with open('assign.in','r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.matrix = []
        for i in a:
            self.matrix.append([int(j) for j in i.split()])
        #self.matrix = self.matrix[0]
        print(np.array(self.matrix))
        print(self.matrix)
        print(self.matrix[0][1])

oa = hungarian()
