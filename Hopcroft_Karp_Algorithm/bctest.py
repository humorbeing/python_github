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
            if self.N<self.M:
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
            self.newpath = []
            self.waitingorder = 0
            self.gotsome = False
            self.gotnothing = False
            #self.foundit = False
            self.tryonnewpath = 0
    def start(self):
        for i in range(self.N):
            for j in range(len(self.L[i])):
                if self.Rside[self.L[i][j]] == -1 and self.Lside[i] == -1:
                    self.count += 1
                    self.Rside[self.L[i][j]] = i
                    self.Lside[i] = self.L[i][j]

        #print(self.count)
        print(self.N)
        print(self.L)
        print(self.Lside)

        print(self.M)
        print(self.R)
        print(self.Rside)

        if self.count == self.N:
            print('got full at start {}'.format(self.count))
            self.stillneedtime = False
    def lookfor(self):
        self.startbackchase(8)
        self.stillneedtime = False

        #self.nn = 0
        #self.Lside = [9,-1,4,5,1,3,2,0,7]
        #self.Rside = [7,4,6,5,2,3,-1,8,-1,0]
        while self.stillneedtime:# and self.nn<50:
            tem = []
            #self.nn += 1
            for i in range(self.N):
                if self.Lside[i] == -1:
                    for j in self.L[i]:
                        tem.append(j)

            if len(tem)>0:
                self.LtoRlookingforconnectedorend(self.cleantem(tem))
            if self.gotnothing:
                print('no change?,we have {}'.format(self.count))
                self.stillneedtime = False
        #print('we got {}'.format(self.count))

    def LtoRlookingforconnectedorend(self, Rsides):
        #print('in lookfor >>>>')
        tem = []
        #print(Rsides)
        for i in Rsides:
            if self.Rside[i] == -1:
                self.gotsome = True
                #self.startbackchase(i)
                self.stillneedtime = False
                print('got candi as {}.'.format(i))
            else:
                tem.append(self.Rside[i])
        if not self.gotsome:
            self.RtoLlookingforwakeup(self.cleantem(tem))

    def RtoLlookingforwakeup(self, Ls):
        #print('in lookfor <<<<<<')
        #print(Ls)
        tem = []
        for i in Ls:
            for j in self.L[i]:
                if i != self.Rside[j]:
                    tem.append(j)
        if len(tem)>0:
            self.LtoRlookingforconnectedorend(self.cleantem(tem))
        else:
            #print(len(tem))
            self.gotnothing = True


    def startbackchase(self, founds):
        self.newpath.append(founds)
        self.RtoLbackchaselookingforconnectionormatch(self.R[founds])

    def RtoLbackchaselookingforconnectionormatch(self, Lseed):
        print(self.newpath)
        print('HHHHHHHH <<<<<<')
        #print(self.Lside[Lseed])
        print(Lseed)
        print(self.Lside[Lseed])
        if self.Lside[Lseed] == -1:
            self.newpath.append(Lseed)
            #print('we are here')
            self.updatepath()
        else:
            self.newpath.append(Lseed)
            self.LtoRbackchaselookingforcleanones(self.Lside[Lseed])

    def LtoRbackchaselookingforcleanones(self, Rs):
        print(self.newpath)
        self.newpath.append(Rs)
        print("HHHHHHHH >>>>>>")
        print(Rs)
        print(self.R[Rs])
        isclean = True
        whatnothing = True
        for i in self.R[Rs]:
            #print('out')
            #print(i)
            if i != Rs:
                #print('in')
                #print(i)
                for j in range(len(self.newpath)):
                    '''
                    print('2*j+1 < len(self.newpath')
                    print(i)
                    print(self.Lside[i])
                    print(self.newpath)
                    #print()
                    print(2*j+1,len(self.newpath))
                    print(2*j+1 < len(self.newpath))
                    '''
                    if 2*j+1 < len(self.newpath):
                        #print('see')
                        #print(self.newpath[2*j +1])
                        if i == self.newpath[2*j +1]:
                            isclean = False
                if isclean:
                    whatnothing = False
                    self.RtoLbackchaselookingforconnectionormatch(i)
                    break
                else:
                    isclean = True
            isclean = True
        if whatnothing:
            self.tryonnewpath += 1
            n = self.newpath[0]
            self.newpath = []
            self.startbackchase(n)
    def updatepath(self):
        #print('old one')

        #print(self.Rside)
        #print(self.Lside)
        #print(self.newpath)
        for i in range(int(len(self.newpath)/2)):
            self.Rside[self.newpath[i*2]] = self.newpath[i*2+1]
            self.Lside[self.newpath[i*2+1]] = self.newpath[i*2]
        #print('new one')
        #print(self.Rside)
        #print(self.Lside)
        self.count += 1
        self.newpath = []
        if self.count == self.N:
            print('we got {} for max count at end'.format(self.count))
        else:
            self.stillneedtime = True
            self.gotsome = False
            self.lookfor()

    def show(self):
        print(self.N)
        print(self.M)
        #print(self.count)
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
    def cleantem(self,t):
        tem = []
        norepeat = True
        for i in t:
            if tem:
                for j in tem:
                    if i == j:
                        norepeat = False
                if norepeat:
                    tem.append(i)
                else:
                    norepeat = True
            else:
                tem.append(i)
        return tem


mm = hopcroft_karp()
mm.run()
