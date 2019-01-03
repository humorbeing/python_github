import numpy as np#just to formated print
class hungarian:
    def __init__(self):
        with open('assign.in','r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.matrix = []
        self.maxinput = 0
        def findmax(s):
            n = int(s)
            if n>self.maxinput:
                self.maxinput = n
            return n
        for i in a:
            self.matrix.append([findmax(j) for j in i.split()])
        print(self.N)
        print(self.matrix)
        print(self.maxinput)
        #print(map(int.__sub__, , b))
        self.minimum_value_matching_matrix = [[(self.maxinput - element) for element in row] for row in self.matrix]
        #print([[(self.maxinput - element) for element in row] for row in self.matrix])
        print(self.minimum_value_matching_matrix)
oa = hungarian()
