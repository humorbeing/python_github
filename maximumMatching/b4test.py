#fixed 0
import numpy as np
class MaximumMatching():
    def __init__(self):

        #np.set_printoptions(precision=1)
        self.notyet = True
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
        if self.N == 0 or self.M == 0:
            self.notyet = False
            print('byebye0')
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
        #controls
        self.needcheck = False
        self.notyet = True
        self.counting = 0
        self.sure = 15
        self.ding = 0
        self.finalcheck = 0
        self.finalconfirm = 9
        self.bestguess = 0

        self.Xscaler = np.array([[np.sum(self.Xform[i])] for i in range(self.edgesum)])
        self.Yscaler = np.array([[np.sum(self.Yform[i])] for i in range(self.edgesum)])
        self.Xway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Yway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.delta_X = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.delta_Y = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Xfix = np.array(([[99 for i in range(self.edgesum)]])).T
        self.Yfix = np.array(([[99 for i in range(self.edgesum)]])).T
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway
        self.Xdone = np.array(([[0 for i in range(self.edgesum)]])).T
        self.Ydone = np.array(([[0 for i in range(self.edgesum)]])).T
        self.Xcheck = []
        self.Ycheck = []
        self.Xcompare = []
        self.Ycompare = []
        self.oldxn = 0
        self.newxn = 0
        self.oldyn = 0
        self.newyn = 0

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
        self.Xcompare = self.Xcheck
        self.Xcheck = self.delta_X
        self.Xvote()

    def Ytraining(self):
        self.delta_Y = self.oneoverx(self.backFLOW)*self.YGetfromforwardX
        self.Ycompare = self.Ycheck
        self.Ycheck = self.delta_Y
        self.Yvote()
        #print(self.oneoverx(np.dot(self.Yform,self.Ycompare))*self.Ycompare)
    def learning(self):
        self.Ykiller()
        self.Xway = np.add( ( np.dot(self.backwardConnector,self.delta_Y)),(self.delta_X) )*0.5
        self.Yway = np.dot(self.forwardConnector,self.Xway)
        self.oldWx = self.Wx
        self.oldWy = self.Wy
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway

        self.oldxn = self.newxn
        self.oldyn = self.newyn
        self.newxn = np.sum ( np.abs( (np.subtract( (self.oldWx),(self.Wx) )) ) )
        self.newyn = np.sum ( np.abs( (np.subtract( (self.oldWy),(self.Wy) )) ) )
        #print(self.newxn,self.oldxn,self.newyn,self.oldyn)
        #self.newxn = np.sum ( np.abs( (np.subtract( (self.Xcheck),(self.Xcompare) )) ) )
        #self.newyn = np.sum ( np.abs( (np.subtract( (self.Ycheck),(self.Ycompare) )) ) )
        if self.oldxn - self.newxn == 0:
            self.ding += 1
        if self.oldyn - self.newyn == 0:
            self.ding += 1
        if self.oldxn == self.newxn == 0:
            self.ding += 2
        if self.oldyn == self.newyn == 0:
            self.ding += 2
        if self.oldxn == self.newxn == self.oldyn == self.newyn:
            self.ding += 3
        if self.oldxn == self.newxn == self.oldyn == self.newyn == 0:
            self.ding += 5
        if np.abs(self.oldxn - self.oldyn) - np.abs(self.newxn - self.newyn) == 0:
            self.ding += 1


        if self.ding > self.sure:
            self.needcheck = True
        '''
        if self.oldxn - self.newxn == 0:
            if self.oldyn - self.newyn == 0:
                self.needcheck = True
        '''

    def train(self):
        while self.notyet:
            if not self.needcheck:
                self.counting += 1
                self.forwardandbackward()
                self.Ytraining()
                self.Xtraining()
                self.learning()
                self.onetoonecheck()
            else:
                print('needcheck from {}'.format(self.counting))
                #print(self.oneoverx(np.dot(self.Yform,self.Ycompare))*self.Ycompare)
                #self.showcheck()
                #print(np.sum(self.oneoverx(np.dot(self.Yform,self.Ycompare))*self.Ycompare))
                self.letscheck()
                #self.notyet = False

    def letscheck(self):
        y1 = np.sum(self.oneoverx(np.dot(self.Yform,self.Ycompare))*self.Ycompare)
        y2 = np.sum(self.oneoverx(np.dot(self.Yform,self.Ycheck))*self.Ycheck)
        #y3 = np.sum(self.oneoverx(np.dot(self.Yform,self.delta_Y))*self.delta_Y)
        x1 = np.sum(self.oneoverx(np.dot(self.Xform,self.Xcompare))*self.Xcompare)
        x2 = np.sum(self.oneoverx(np.dot(self.Xform,self.Xcheck))*self.Xcheck)
        #x3 = np.sum(self.oneoverx(np.dot(self.Xform,self.delta_X))*self.delta_X)
        #y2 = 0
        if y1 == y2 and x1 == x2:
            print('oh,its {}'.format(y1))
            self.notyet = False
        else:
            if y1>self.bestguess:
                self.bestguess = y1
            self.finalcheck += 1
            self.needcheck = False

        if self.finalcheck>self.finalconfirm:
            print('sorry,bestguess is...{}'.format(self.bestguess))
            self.notyet = False



    def showcheck(self):
        np.set_printoptions(precision=1)
        DDDDDDDDDD = []
        DDDDDDDDDD.append(self.Xcompare.T[0])
        DDDDDDDDDD.append(self.Xcheck.T[0])
        DDDDDDDDDD.append(self.delta_X.T[0])
        print('--XXXXXXXXXXXXXXXXXXXXXXXXXXXXX--')
        #print('  XXXX  >>>  <<< | <<<  >>>  YYYY')
        print(np.array(DDDDDDDDDD).T)

        DDDDDDDDDD = []
        DDDDDDDDDD.append(self.Ycompare.T[0])
        DDDDDDDDDD.append(self.Ycheck.T[0])
        DDDDDDDDDD.append(self.delta_Y.T[0])
        print('--YYYYYYYYYYYYYYYYYYYYYYYYYYYYY--')
        #print('  XXXX  >>>  <<< | <<<  >>>  YYYY')
        print(np.array(DDDDDDDDDD).T)

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
                    #print('got one 1on1: {}'.format(i))
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
                self.delta_X[tie[0]] = self.delta_X[tie[0]]*n
            else:
                self.delta_X[winner] = self.delta_X[winner]*n
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
                self.delta_Y[tie[0]] = self.delta_Y[tie[0]]*n
            else:
                self.delta_Y[winner] = self.delta_Y[winner]*n
            compare = 0
            donefor += n
    def Ykiller(self):
        for i in range(self.edgesum):
            if self.Ydone[i][0] == 0:
                if self.delta_Y[i][0] < 0.1:
                    #print('killing one. yw {} is dead'.format(i))
                    #print(self.delta_Y)
                    self.delta_Y[i][0] = 0
                    self.delta_X[self.lookingbackward(i)][0] = 0
                    self.Ydone[i][0] =1
                    self.Xdone[self.lookingbackward(i)][0] =1
                    self.Yfix[i][0] = 0
                    self.Xfix[self.lookingbackward(i)][0] = 0
                    #print('after')
                    #print(self.delta_Y)
    def Yend(self):
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
                    self.delta_Y[tie[0]] = self.delta_Y[tie[0]]*n
                else:
                    self.delta_Y[winner] = self.delta_Y[winner]*n
                compare = 0
                donefor += n
mm = MaximumMatching()
for i in range(11):
    mm.train()
