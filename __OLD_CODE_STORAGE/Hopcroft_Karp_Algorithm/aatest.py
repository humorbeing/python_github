class hopcroft_karp:
    def __init__(self):
        with open('matching.inp','r') as f:
            a = f.readlines()
        def arr(s):
            tem = []
            b = s.split(' ')
            for i in b:
                if i != ' ' and i != '\n':
                    tem.append(int(i))
            return tem
        self.N = arr(a[0])[0]
        self.M = arr(a[0])[1]
        if self.N == 0 or self.M == 0:
            pass
        else:
            self.A = [[] for i in range(self.N)]
            self.B = [[] for i in range(self.M)]
            for i in range(self.N):
                tem = []
                C = arr(a[i+1])
                #self.edgesum += A[1]
                for j in range(C[1]):
                    self.A[i].append(C[j+2])
                    self.B[C[j+2]-1].append(i+1)

        self.Aside = [-1 for i in range(self.N)]
        self.Bside = [-1 for i in range(self.M)]
        self.count = 0
    def start(self):
        for i in range(self.N):
            for j in range(len(self.A[i])):
                if self.Bside[self.A[i][j]-1] == -1:
                    self.count += 1
                    self.Bside[self.A[i][j]-1] = i
                    self.Aside[i] = self.A[i][j]


    def show(self):
        print(self.count)
        print(self.Aside)
        print(self.Bside)
        print(self.A)
        print(self.B)
    def run(self):
        self.start()
        self.show()
mm = hopcroft_karp()
mm.run()
