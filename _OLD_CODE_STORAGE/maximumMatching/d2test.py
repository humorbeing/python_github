#import numpy as np

class MaximumMatching():
    def __init__(self):
        self.notyet = True
        self.A = []
        self.N = 0
        self.M = 0
        self.edgesum = 0
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
            self.output(0)
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
            self.setupForm()
    def setupForm(self):
        self.Xform = [[0 for j in range(self.edgesum)] for i in range(self.edgesum)]
        self.Yform = [[0 for j in range(self.edgesum)] for i in range(self.edgesum)]
        self.toshow = [[0 for j in range(self.edgesum)] for i in range(self.M)]
        self.forwardConnector = []
        self.backwardConnector = []

        tem = []
        ping = [0 for i in range(self.N)]
        donefor = 0
        for i in self.A:
            n = sum(i)
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
        half_c = self.transpose(tem)
        tem = []
        donefor = 0
        row = 0
        ping = [0 for i in range(self.edgesum)]
        for i in range(self.M):
            for j in range(sum(half_c[i])):
                self.toshow[i][j+donefor] = 1
            donefor += sum(half_c[i])
        donefor = 0
        for i in half_c:
            n = sum(i)
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
        self.forwardConnector = tem
        self.backwardConnector = self.transpose(self.forwardConnector)
        self.setupOperators()
    def setupOperators(self):
        self.needcheck = False
        self.counting = 0
        self.sure = 15
        self.ding = 0
        self.finalcheck = 0
        self.finalconfirm = 9
        self.bestguess = 0
        self.Xway = self.transpose([[1.0 for i in range(self.edgesum)]])
        self.Yway = self.transpose([[1.0 for i in range(self.edgesum)]])
        self.delta_X = self.transpose([[1.0 for i in range(self.edgesum)]])
        self.delta_Y = self.transpose([[1.0 for i in range(self.edgesum)]])
        self.Xfix = self.transpose([[99 for i in range(self.edgesum)]])
        self.Yfix = self.transpose([[99 for i in range(self.edgesum)]])
        self.Wx = self.mult(self.oneoverx(self.dot(self.Xform,self.Xway)),self.Xway)
        self.Wy = self.mult(self.oneoverx(self.dot(self.Yform,self.Yway)),self.Yway)
        self.Xdone = self.transpose([[0 for i in range(self.edgesum)]])
        self.Ydone = self.transpose([[0 for i in range(self.edgesum)]])
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
        self.FLOW = [[self.X] for i in range(self.edgesum)]
        self.Xsendingforforward = self.mult(self.Wx,self.FLOW)
        self.YGetfromforwardX = self.dot(self.forwardConnector,self.Xsendingforforward)
        self.backFLOW = self.dot(self.Yform,self.YGetfromforwardX)
        self.Ysendingforbackward = self.mult(self.Wy,self.backFLOW)
        self.XGetfrombackwardY = self.dot(self.backwardConnector,self.Ysendingforbackward)
        self.receiveFlow = self.dot(self.Xform,self.XGetfrombackwardY)
    def Xtraining(self):
        self.delta_X = self.mult(self.oneoverx(self.receiveFlow),self.XGetfrombackwardY)
        self.Xcompare = self.Xcheck
        self.Xcheck = self.delta_X
        self.Xvote()
    def Ytraining(self):
        self.delta_Y = self.mult(self.oneoverx(self.backFLOW),self.YGetfromforwardX)
        self.Ycompare = self.Ycheck
        self.Ycheck = self.delta_Y
        self.Yvote()
    def learning(self):
        self.Ykiller()
        self.Xway = self.multn( self.add( ( self.dot(self.backwardConnector,self.delta_Y)),(self.delta_X) ), 0.5 )
        self.Yway = self.dot(self.forwardConnector,self.Xway)
        self.oldWx = self.Wx
        self.oldWy = self.Wy
        self.Wx = self.mult(self.oneoverx(self.dot(self.Xform,self.Xway)),self.Xway)
        self.Wy = self.mult(self.oneoverx(self.dot(self.Yform,self.Yway)),self.Yway)
        self.oldxn = self.newxn
        self.oldyn = self.newyn
        self.newxn = self.sum ( self.abs( (self.subtract( (self.oldWx),(self.Wx) )) ) )
        self.newyn = self.sum ( self.abs( (self.subtract( (self.oldWy),(self.Wy) )) ) )
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
        if self.absn(self.oldxn - self.oldyn) - self.absn(self.newxn - self.newyn) == 0:
            self.ding += 1
        if self.ding > self.sure:
            self.needcheck = True
    def train(self):
        while self.notyet:
            if not self.needcheck:
                self.forwardandbackward()
                self.Ytraining()
                self.Xtraining()
                self.learning()
                self.onetoonecheck()
            else:
                self.letscheck()
    def letscheck(self):
        y1 = self.sum(self.mult(self.oneoverx(self.dot(self.Yform,self.Ycompare)),self.Ycompare))
        y2 = self.sum(self.mult(self.oneoverx(self.dot(self.Yform,self.Ycheck)),self.Ycheck))
        x1 = self.sum(self.mult(self.oneoverx(self.dot(self.Xform,self.Xcompare)),self.Xcompare))
        x2 = self.sum(self.mult(self.oneoverx(self.dot(self.Xform,self.Xcheck)),self.Xcheck))
        if y1 == y2 and x1 == x2:
            self.output(y1)
            self.notyet = False
        else:
            if y1>self.bestguess:
                self.bestguess = y1
            self.finalcheck += 1
            self.needcheck = False

        if self.finalcheck>self.finalconfirm:
            self.output(self.bestguess)
            self.notyet = False
    def oneoverx(self, S):
        tem = []
        for i in S:
            if i[0] == 0:
                tem.append(0.0)
            else:
                tem.append(1.0/i[0])
        return self.transpose([tem])
    def onetoonecheck(self):
        for i in range(self.edgesum):
            if self.Ydone[i][0] == 0:
                if self.X == self.YGetfromforwardX[i][0] == self.Ysendingforbackward[i][0] == self.Xsendingforforward[self.lookingbackward(i)][0] == self.XGetfrombackwardY[self.lookingbackward(i)][0]:
                    self.Ydone[i][0] = 1
                    self.Yfix[i][0] = 1
                    self.Xdone[self.lookingbackward(i)][0] = 1
                    self.Xfix[self.lookingbackward(i)][0] = 1
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
            n = sum(self.Xform[donefor])
            for j in range(n):
                if self.delta_X[j + donefor][0] > compare:
                    needvote = False
                    tie = []
                    winner = j + donefor
                    compare = self.delta_X[j + donefor][0]
                elif self.delta_X[j + donefor][0] == compare:
                    needvote = True
                    tie.append(winner)
                    winner = j + donefor
            if needvote:
                tie.append(winner)
                self.delta_X[tie[0]][0] = self.delta_X[tie[0]][0]*n
            else:
                self.delta_X[winner][0] = self.delta_X[winner][0]*n
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
            n = sum(self.Yform[donefor])
            for j in range(n):
                if self.delta_Y[j + donefor][0] > compare:
                    needvote = False
                    tie = []
                    winner = j + donefor
                    compare = self.delta_Y[j + donefor][0]
                elif self.delta_Y[j + donefor][0] == compare:
                    needvote = True
                    tie.append(winner)
                    winner = j + donefor
            if needvote:
                tie.append(winner)
                self.delta_Y[tie[0]][0] = self.delta_Y[tie[0]][0]*n
            else:
                self.delta_Y[winner][0] = self.delta_Y[winner][0]*n
            compare = 0
            donefor += n
    def Ykiller(self):
        for i in range(self.edgesum):
            if self.Ydone[i][0] == 0:
                if self.delta_Y[i][0] < 0.1:
                    self.delta_Y[i][0] = 0
                    self.delta_X[self.lookingbackward(i)][0] = 0
                    self.Ydone[i][0] =1
                    self.Xdone[self.lookingbackward(i)][0] =1
                    self.Yfix[i][0] = 0
                    self.Xfix[self.lookingbackward(i)][0] = 0
    def output(self,s):
        with open('matching.out','w') as f:
            f.write(str(int(s)))
    def transpose(self,S):
        tem = [[0 for j in range(len(S))] for i in range(len(S[0]))]
        for i in range(len(S)):
            for j in range(len(S[0])):
                tem[j][i] = S[i][j]
        return tem
    def dot(self, S, T):
        #nxm () mxv = n x v
        n = len(S)
        m = len(T)
        v = len(T[0])
        tem = [[0 for j in range(v)] for i in range(n)]
        s = 0
        for i in range(n):
            for j in range(v):
                for k in range(m):
                    s += S[i][k]*T[k][j]
                tem[i][j] = s
                s = 0
        return tem
    def mult(self, S, T):
        tem = [[] for i in range(len(S))]
        for i in range(len(S)):
            tem[i].append(S[i][0]*T[i][0])
        return tem
    def add(self, S, T):
        tem = [[] for i in range(len(S))]
        for i in range(len(S)):
            tem[i].append(S[i][0]+T[i][0])
        return tem
    def subtract(self, S, T):
        tem = [[] for i in range(len(S))]
        for i in range(len(S)):
            tem[i].append(S[i][0]-T[i][0])
        return tem
    def multn(self, S, n):
        tem = [[] for i in range(len(S))]
        for i in range(len(S)):
            tem[i].append(S[i][0]*n)
        return tem
    def absn(self, s):
        if s >=0:
            return s
        else:
            return s*(-1)
    def abs(self, S):
        tem = [[] for i in range(len(S))]
        for i in range(len(S)):
            if S[i][0]>=0:
                tem[i].append(S[i][0])
            else:
                tem[i].append(S[i][0]*(-1))
        return tem
    def sum(self, S):
        s = 0
        for i in range(len(S)):
            s += S[i][0]
        return s
mm = MaximumMatching()
mm.train()
