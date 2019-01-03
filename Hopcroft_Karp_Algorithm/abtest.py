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
            G = []
            self.edgesum = 0
            for i in range(self.N):
                tem = []
                A = arr(a[i+1])
                self.edgesum += A[1]
                for j in range(A[1]):
                    tem.append(A[j+2]-1)
                G.append(tem)
            A = [[0 for j in range(self.M)] for i in range(self.N)]
            for i in range(self.N):
                for j in range(len(G[i])):
                    A[i][G[i][j]] = 1
            self.A = A
            nn = self.N
            gotone = 0
            for i in range(nn):
                if sum(self.A[i-gotone]) == 0:
                    del self.A[i-gotone]
                    gotone += 1
                    self.N -= 1
            self.A = self.transpose(self.A)
            gotone = 0
            mm = self.M
            for i in range(mm):
                if sum(self.A[i-gotone]) == 0:
                    del self.A[i-gotone]
                    gotone += 1
                    self.M -= 1
            if self.N>self.M:
                self.A = self.transpose(self.A)
            else:
                self.N,self.M = self.M,self.N

            self.L = [[] for i in range(self.N)]
            self.R = [[] for i in range(self.M)]
            for i in range(self.N):
                for j in range(self.M):
                    if self.A[i][j] == 1:
                        self.L[i].append(j)
                        self.R[j].append(i)

            self.Lside = [-1 for i in range(self.N)]
            self.Rside = [-1 for i in range(self.M)]
            self.count = 0
            self.stillneedtime = True
    def start(self):
        for i in range(self.N):
            for j in range(len(self.L[i])):
                if self.Rside[self.L[i][j]-1] == -1:
                    self.count += 1
                    self.Rside[self.L[i][j]-1] = i
                    self.Lside[i] = self.L[i][j]
        if self.count == self.M:
            self.stillneedtime = False
    def lookfor(self):
        if self.stillneedtime:
            print('still need time')
        else:
            print('got {}'.format(self.count))
        #while self.stillneedtime:
            #for i in range(self.M):
                #if self.Rside
    def show(self):
        print(self.count)
        print(self.Lside)
        print(self.Rside)
        #print(self.A)
        print(self.L)
        print(self.R)
        #print(self.B)
    def run(self):
        self.start()
        #self.show()
        self.lookfor()
    def transpose(self,S):
        tem = [[0 for j in range(len(S))] for i in range(len(S[0]))]
        for i in range(len(S)):
            for j in range(len(S[0])):
                tem[j][i] = S[i][j]
        return tem
mm = hopcroft_karp()
mm.run()
