import numpy as np
class MaximumMatching():
    def __init__(self):
        #controls
        self.counting = 0
        self.checking = 3
        self.end = 10
        self.stillneedtime = True
        #np.set_printoptions(precision=1)
        self.A = []
        self.N = 0
        self.M = 0
        self.edgesum = 0
        #Graph A, size of N,M, # of edges(edgesum)
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
        self.A = np.array(A)
        nn = self.N
        gotone = 0
        for i in range(nn):
            if np.sum(self.A[i-gotone]) == 0:
                self.A = np.delete(self.A,i-gotone,0)#!!! - - !!!
                gotone += 1
                self.N -= 1
        self.A = self.A.T

        gotone = 0
        mm = self.M
        for i in range(mm):
            if np.sum(self.A[i-gotone]) == 0:
                self.A = np.delete(self.A,i-gotone,0)
                gotone += 1
                self.M -= 1
        if self.N>self.M:
            self.A = self.A.T
        else:
            self.N,self.M = self.M,self.N
        self.setupForm()

    def setupForm(self):
        self.Xform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
        self.Yform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
        self.toshow = np.array(([[0 for j in range(self.edgesum)] for i in range(self.M)]))
        self.forwardConnector = []
        self.backwardConnector = []

        tem = []
        ping = [0 for i in range(self.N)]
        donefor = 0
        for i in self.A:
            n = np.sum(i)
            for j in range(n):
                for k in range(n):
                    self.Xform[j+donefor][k+donefor] = 1
            donefor += n
            on = 0
            while n > 0:
                if i[on] == 1:
                    n -= 1
                    ping[on] = 1
                    tem.append(ping)
                    ping = [0 for j in range(self.N)]
                on += 1
        half_c = np.array(tem).T
        tem = []
        donefor = 0
        row = 0
        ping = [0 for i in range(self.edgesum)]
        for i in range(self.M):
            for j in range(np.sum(half_c[i])):
                self.toshow[i][j+donefor] = 1
            donefor += np.sum(half_c[i])
        donefor = 0
        for i in half_c:
            n = np.sum(i)
            for j in range(n):
                for k in range(n):
                    self.Yform[j+donefor][k+donefor] = 1
            donefor += n
            on = 0
            while n > 0:
                if i[on] == 1:
                    n -= 1
                    ping[on] = 1
                    tem.append(ping)
                    ping = [0 for j in range(self.edgesum)]
                on += 1
        self.forwardConnector = np.array(tem)
        self.backwardConnector = self.forwardConnector.T
        self.setupOperators()

    def setupOperators(self):
        self.Xscaler = np.array([[np.sum(self.Xform[i])] for i in range(self.edgesum)])
        self.Yscaler = np.array([[np.sum(self.Yform[i])] for i in range(self.edgesum)])
        self.Xway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Yway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.delta_X = np.array(([[0.0 for i in range(self.edgesum)]])).T
        self.delta_Y = np.array(([[0.0 for i in range(self.edgesum)]])).T
        self.Xfix = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Yfix = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway
        self.Xdone = np.array(([[0 for i in range(self.edgesum)]])).T
        self.Ydone = np.array(([[0 for i in range(self.edgesum)]])).T

    def forwardandbackward(self):
        self.X = 12
        self.FLOW = np.array(([[self.X] for i in range(self.edgesum)]))
        self.Xsendingforforward = self.Wx*self.FLOW
        self.YGetfromforwardX = np.dot(self.forwardConnector,self.Xsendingforforward)
        self.backFLOW = np.dot(self.Yform,self.YGetfromforwardX)
        self.Ysendingforbackward = self.Wy*self.backFLOW
        self.XGetfrombackwardY = np.dot(self.backwardConnector,self.Ysendingforbackward)
        self.receiveFlow = np.dot(self.Xform,self.XGetfrombackwardY)

    def Xtraining(self):
        self.delta_X = self.oneoverx(self.receiveFlow)*self.XGetfrombackwardY
        #print('XXXXXXXXXXXX')
        #print(self.delta_X)
        #self.Xvote()
        #print(np.dot(self.Xform,self.delta_X))
    def Ytraining(self):
        self.delta_Y = self.oneoverx(self.backFLOW)*self.YGetfromforwardX
        print('YYYYYYYYYYY')
        print(self.delta_Y)
        self.Yvote()

    def learning(self):
        self.Xway = np.add( ( np.dot(self.backwardConnector,self.delta_X)),(self.delta_Y) )*0.5
        self.Yway = np.dot(self.forwardConnector,self.Xway)
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway

    def train(self):
        self.counting += 1
        self.forwardandbackward()
        self.onetoonecheck()
        print('/'*15 + '   **|**   ' + '\\'*15)
        #self.showflow()
        #self.showweight()
        self.Xtraining()
        self.Ytraining()
        #self.showweight()
        self.learning()
        #self.showflow()
        #self.showweight()
        #self.show()

    def showflow(self):
        np.set_printoptions(precision=1)
        DDDDDDDDDD = []
        DDDDDDDDDD.append(self.FLOW.T[0])
        DDDDDDDDDD.append(self.Xsendingforforward.T[0])
        DDDDDDDDDD.append(self.XGetfrombackwardY.T[0])
        #DDDDDDDDDD.append(self.YGetfromforwardX.T[0])
        DDDDDDDDDD.append(np.dot(self.backwardConnector,self.Ysendingforbackward).T[0])
        DDDDDDDDDD.append(np.dot(self.backwardConnector,self.YGetfromforwardX).T[0])
        DDDDDDDDDD.append(np.dot(self.backwardConnector,self.backFLOW).T[0])
        print('--XXXXXXXXXXXXXXXXXXXXXXXXXXXXX--')
        print('  XXXX  >>>  <<< | <<<  >>>  YYYY')
        print(np.array(DDDDDDDDDD).T)

        ##################################
        ##################################
        DDDDDDDDDD = []
        DDDDDDDDDD.append(self.FLOW.T[0])
        DDDDDDDDDD.append(np.dot(self.forwardConnector,self.Xsendingforforward).T[0])
        DDDDDDDDDD.append(np.dot(self.forwardConnector,self.XGetfrombackwardY).T[0])
        #DDDDDDDDDD.append(self.YGetfromforwardX.T[0])
        DDDDDDDDDD.append(self.Ysendingforbackward.T[0])
        DDDDDDDDDD.append(self.YGetfromforwardX.T[0])
        DDDDDDDDDD.append(self.backFLOW.T[0])
        #print('--YYYYYYYYYYYYYYYYYYYYYYYYYYYYY--')
        #print('  XXXX  >>>  <<< | <<<  >>>  YYYY')
        #print(np.array(DDDDDDDDDD).T)
        #print(np.around(np.array(DDDDDDDDDD).T, decimals = 1))

    def showweight(self):
        np.set_printoptions(precision=1)
        XXXXXXXXXXXXXXXX = []
        XXXXXXXXXXXXXXXX.append(self.Wx.T[0])
        XXXXXXXXXXXXXXXX.append(self.Xscaler.T[0])
        XXXXXXXXXXXXXXXX.append(self.delta_X.T[0])
        XXXXXXXXXXXXXXXX.append(self.Xway.T[0])
        XXXXXXXXXXXXXXXX.append(self.Yway.T[0])
        XXXXXXXXXXXXXXXX.append(self.delta_Y.T[0])
        XXXXXXXXXXXXXXXX.append(self.Yscaler.T[0])
        XXXXXXXXXXXXXXXX.append(self.Wy.T[0])
        print('   Wx | Xs | Dx |-Xw-|-Yw-| Dy | Ys |  Wy ')
        print(np.array(XXXXXXXXXXXXXXXX).T)

    def show(self):
        np.set_printoptions(precision=1)
        print(np.dot(self.toshow,self.YGetfromforwardX) )

    def oneoverx(self, S):
        tem = []
        for i in S:
            if i[0] == 0:
                tem.append(0.0)
            else:
                tem.append(1.0/i[0])
        return np.array([tem]).T
    def onetoonecheck(self):
        for i in range(self.edgesum):
            #print(self.lookingbackward(i))
            if self.Ydone[i][0] == 0:
                if self.X == self.YGetfromforwardX[i][0] == self.Ysendingforbackward[i][0] == self.Xsendingforforward[self.lookingbackward(i)][0] == self.XGetfrombackwardY[self.lookingbackward(i)][0]:
                    print('got one 1on1: {}'.format(i))
                    self.Ydone[i][0] = 1
                    self.Yfix[i][0] = 1
                    self.Xdone[self.lookingbackward(i)][0] = 1
                    self.Xfix[self.lookingbackward(i)][0] = 1
                    #print(self.Ydone)
    def lookingforward(self, s):
        for i in range(self.edgesum):
            if self.backwardConnector[s][i] == 1:
                return i
                break
    def lookingbackward(self, s):
        for i in range(self.edgesum):
            if self.forwardConnector[s][i] == 1:
                return i
                break
    def Xvote(self):
        n = 0
        voted = 0
        votingfor = 0
        donefor = 0
        winner = 0
        tie = []
        compare = 0
        needvote = False
        for i in range(self.N):
            n = np.sum(self.Xform[donefor])
            for j in range(n):
                if self.delta_X[j + donefor] > compare:
                    needvote = False
                    tie = []
                    winner = j + donefor
                    compare = self.delta_X[j + donefor]
                elif self.delta_X[j + donefor] == compare:
                    needvote = True
                    tie.append(winner)
                    winner = j + donefor
            if needvote:
                tie.append(winner)
                print('Its tie in ({} - {}), in {}N.'.format(donefor,n + donefor -1,n))
                print(tie)
                print('----- tie end ------')

            else:
                print('Winner is {} in ({} - {}), in {}N.'.format(winner,donefor,n + donefor -1,n))
                #pass
            compare = 0
            donefor += n
    def Yvote(self):
        n = 0
        voted = 0
        votingfor = 0
        donefor = 0
        winner = 0
        tie = []
        compare = 0
        needvote = False
        for i in range(self.M):
            n = np.sum(self.Yform[donefor])
            '''
            print('i and n and donefor and i+donefor')
            print(i)
            print(n)
            print(donefor)
            print(i+donefor)
            print('end')
            '''
            for j in range(n):
                if self.delta_Y[j + donefor] > compare:
                    needvote = False
                    tie = []
                    winner = j + donefor
                    compare = self.delta_Y[j + donefor]
                elif self.delta_Y[j + donefor] == compare:
                    needvote = True
                    tie.append(winner)
                    winner = j + donefor
            if needvote:
                tie.append(winner)
                print('Its tie in ({} - {}), in {}N.'.format(donefor,n + donefor -1,n))
                print(tie)
                print('----- tie end ------')

            else:
                print('Winner is {} in ({} - {}), in {}N.'.format(winner,donefor,n + donefor -1,n))
                #pass
            compare = 0
            donefor += n

mm = MaximumMatching()
for i in range(2):
    mm.train()
