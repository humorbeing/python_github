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
        #self.Xform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
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
        #self.Yform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
        #self.toshow = np.array(([[0 for j in range(self.edgesum)] for i in range(self.M)]))
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
        #print(self.backwardConnector)
        #print(self.forwardConnector)

    def onetoonecheck(self):
        for i in range(self.edgesum):
            #print(self.lookingbackward(i))
            if self.Ydone[i][0] == 0:
                if self.X == self.YGetfromforwardX[i][0] == self.Ysendingforbackward[i][0] == self.Xsendingforforward[self.lookingbackward(i)][0] == self.XGetfrombackwardY[self.lookingbackward(i)][0]:
                    print('got one 1on1: {}'.format(i))
                    self.Ydone[i][0] = 1
                    self.Xdone[self.lookingbackward(i)][0] = 1
                    #print(self.Ydone)
    '''
    def voting(self):
        on = 0
        n = 0
        for i in range(self.M):
            n = np.sum(self.Yform[i])
            if n == 1:
                self.delta_Y[on][0] = 1
                on += n
            else:
                for j in range(n):
    '''

    def Xtraining(self):
        Xsum = np.dot(self.Xform,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        Xtocheck = np.subtract(Xsum,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        self.delta_X = self.oneoverx(Xsum)*Xtocheck*self.Xscaler
        for i in range(len(self.Xscaler)):
            if self.Xscaler[i][0] == 1:
                self.delta_X[i][0] = 1
        self.Xway = self.delta_X*self.Xway
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway

    def Ytraining(self):
        self.delta_Y = self.oneoverx(self.backFLOW)*self.YGetfromforwardX*self.Yscaler
        self.Yway = self.delta_Y*self.Yway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway
        if self.counting > self.checking:
            test = []
            testform = []
            found = False
            for i in range(len(self.Wy)):
                if self.delta_Y[i][0] == 1:
                    if self.Wy[i][0] != 1:
                        test.append(i)
                        found = True
            if found:
                testform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
                for i in range(len(test)):
                    for j in range(self.edgesum):
                        if test[i] == j:
                            testform[j] = testform[j] + 1
                            testform[j][j] = 0
                testform = self.Yform*testform
                for i in range(len(test)):
                    if test[i] > 0:
                        for j in range(self.edgesum):
                            if testform[test[i]][j] == 1:
                                if j in test:
                                    if self.Wy[j][0] == self.Wy[test[i]][0]:
                                        self.Wy[test[i]][0] = self.Wy[test[i]][0]*2
                                        self.Wy[j][0] = 0
                                        test[i] = -1
                                        if test[i+1] == j:
                                            test[i+1] = -1
            Wyy = self.Wy
            YYscaler = self.Yscaler
            for i in range(len(Wyy)):
                if Wyy[i][0]<0.1:
                    Wyy[i][0] = 0
                    self.Yfix[i][0] = 0
                    self.Xfix = np.dot(self.forwardConnector,self.Yfix)
                    self.Xway = self.Xfix
                    self.Yway = self.Yfix
                    YYscaler[i][0] = 0
            self.Yscaler = self.oneoverx(YYscaler)*np.dot(self.Yform,YYscaler)
            self.Wy = self.oneoverx(np.dot(self.Yform,Wyy))*Wyy

    def learning(self):
        self.Xway = np.add( ( np.dot(self.backwardConnector,self.Wy)),(self.Wy) )*0.5
        #self.Xway = np.dot(self.backwardConnector,self.Yway)*self.Xway
        for i in range(self.edgesum):
            if self.Wx[i][0] == 1:
                if self.delta_X[i][0] == 1:
                    if np.dot(self.forwardConnector,self.delta_Y)[i][0] == 1:
                        if np.dot(self.forwardConnector,self.Wy)[i][0] == 1:
                            self.Xway[i][0] = 1
        '''
        for i in range(len(self.Xway)):
            if self.Xway[i][0] > 9999:
                self.Xway = self.Xway*0.0001
            elif self.Xway[i][0] < 0.0001:
                self.Xway = self.Xway*1000
        '''
        self.Yway = np.dot(self.forwardConnector,self.Xway)
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        self.Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway

    def forward(self):
        self.oldx = self.Xway
        self.oldy = self.Yway

        '''
        if self.counting > self.checking:
            test = []
            testform = []
            found = False
            for i in range(len(self.Wx)):
                if self.delta_X[i][0] == 1:
                    if self.Wx[i][0] != 1:
                        test.append(i)
                        found = True
            if found:
                testform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
                for i in range(len(test)):
                    for j in range(self.edgesum):
                        if test[i] == j:
                            testform[j] = testform[j] + 1
                            testform[j][j] = 0
                testform = self.Xform*testform
                for i in range(len(test)):
                    if test[i] > 0:
                        for j in range(self.edgesum):
                            if testform[test[i]][j] == 1:
                                if j in test:
                                    if self.Wx[j][0] == self.Wx[test[i]][0]:
                                        self.Wx[test[i]][0] = self.Wx[test[i]][0]*2
                                        self.Wx[j][0] = 0
                                        test[i] = -1
                                        if test[i+1] == j:
                                            test[i+1] = -1
            Wxx = self.Wx
            XXscaler = self.Xscaler
            for i in range(len(Wxx)):
                if Wxx[i][0]<0.1:
                    Wxx[i][0] = 0
                    self.Xway = np.array(([[1.0 for i in range(self.edgesum)]])).T
                    self.Xway[i][0] = 0
                    XXscaler[i][0] = 0
            self.Xscaler = self.oneoverx(XXscaler)*np.dot(self.Xform,XXscaler)
            self.Wx = self.oneoverx(np.dot(self.Xform,Wxx))*Wxx
        '''

        Wxx = self.Wx
        XXscaler = self.Xscaler
        for i in range(len(Wxx)):
            if Wxx[i][0]<0.1:
                Wxx[i][0] = 0
                self.Xfix[i][0] = 0
                self.Yfix = np.dot(self.backwardConnector,self.Xfix)
                self.Xway = self.Xfix
                self.Yway = self.Yfix
                XXscaler[i][0] = 0
        self.Xscaler = self.oneoverx(XXscaler)*np.dot(self.Xform,XXscaler)
        self.Wx = self.oneoverx(np.dot(self.Xform,Wxx))*Wxx
        self.xchange = self.Xway

    def backward(self):
        XXXXXXXXXXXXXXXX = []
        YYYYYYYYYYYYYYYY = []
        SSSSSSSSSSSSSSSS = []

        if self.counting > self.checking:
            test = []
            testform = []
            found = False
            for i in range(len(self.Wy)):
                if self.delta_Y[i][0] == 1:
                    if self.Wy[i][0] != 1:
                        test.append(i)
                        found = True
            if found:
                testform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
                for i in range(len(test)):
                    for j in range(self.edgesum):
                        if test[i] == j:
                            testform[j] = testform[j] + 1
                            testform[j][j] = 0
                testform = self.Yform*testform
                for i in range(len(test)):
                    if test[i] > 0:
                        for j in range(self.edgesum):
                            if testform[test[i]][j] == 1:
                                if j in test:
                                    if self.Wy[j][0] == self.Wy[test[i]][0]:
                                        self.Wy[test[i]][0] = self.Wy[test[i]][0]*2
                                        self.Wy[j][0] = 0
                                        test[i] = -1
                                        if test[i+1] == j:
                                            test[i+1] = -1
            Wyy = self.Wy
            YYscaler = self.Yscaler
            for i in range(len(Wyy)):
                if Wyy[i][0]<0.1:
                    Wyy[i][0] = 0
                    self.Yfix[i][0] = 0
                    self.Xfix = np.dot(self.forwardConnector,self.Yfix)
                    self.Xway = self.Xfix
                    self.Yway = self.Yfix
                    YYscaler[i][0] = 0
            self.Yscaler = self.oneoverx(YYscaler)*np.dot(self.Yform,YYscaler)
            self.Wy = self.oneoverx(np.dot(self.Yform,Wyy))*Wyy
        self.ychange = self.Yway

        self.delta_Y = self.oneoverx(Y)*self.YGetfromforwardX*self.Yscaler
        self.Yway = self.delta_Y*self.Yway
        Xsum = np.dot(self.Xform,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        Xtocheck = np.subtract(Xsum,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        self.delta_X = self.oneoverx(Xsum)*Xtocheck*self.Xscaler
        for i in range(len(self.Xscaler)):
            if self.Xscaler[i][0] == 1:
                self.delta_X[i][0] = 1

        XXXXXXXXXXXXXXXX.append(self.Wx.T[0])
        #XXXXXXXXXXXXXXXX.append(self.Xway.T[0])
        XXXXXXXXXXXXXXXX.append(self.Xscaler.T[0])
        XXXXXXXXXXXXXXXX.append(self.delta_X.T[0])

        self.Xway = self.delta_X*self.Xway
        self.Xway = np.dot(self.backwardConnector,self.Yway)*self.Xway
        for i in range(self.edgesum):
            if self.Wx[i][0] == 1:
                if self.delta_X[i][0] == 1:
                    if np.dot(self.forwardConnector,self.delta_Y)[i][0] == 1:
                        if np.dot(self.forwardConnector,self.Wy)[i][0] == 1:
                            self.Xway[i][0] = 1
        for i in range(len(self.Xway)):
            if self.Xway[i][0] > 9999:
                self.Xway = self.Xway*0.0001
            elif self.Xway[i][0] < 0.0001:
                self.Xway = self.Xway*1000
        self.Yway = np.dot(self.forwardConnector,self.Xway)
        #XXXXXXXXXXXXXXXX.append(['|' for i in range(self.edgesum)])

        #XXXXXXXXXXXXXXXX.append(self.Xway.T[0])

        XXXXXXXXXXXXXXXX.append(self.delta_Y.T[0])
        XXXXXXXXXXXXXXXX.append(self.Yscaler.T[0])
        XXXXXXXXXXXXXXXX.append(self.Wy.T[0])

        #SSSSSSSSSSSSSSSS.append(np.array(XXXXXXXXXXXXXXXX).T)
        #SSSSSSSSSSSSSSSS.append(np.array(YYYYYYYYYYYYYYYY).T)
        np.set_printoptions(precision=1)
        print('   Wx | Xs | Dx | Dy | Ys |  Wy ')
        print(np.array(XXXXXXXXXXXXXXXX).T)
        SSSSSSSSSSSSSSSS.append(self.oldx.T[0])
        SSSSSSSSSSSSSSSS.append(self.xchange.T[0])
        SSSSSSSSSSSSSSSS.append(self.Xway.T[0])
        SSSSSSSSSSSSSSSS.append(self.Yway.T[0])
        SSSSSSSSSSSSSSSS.append(self.ychange.T[0])
        SSSSSSSSSSSSSSSS.append(self.oldy.T[0])
        print(np.array(SSSSSSSSSSSSSSSS).T)

    def train(self):
        self.counting += 1
        self.forwardandbackward()
        self.onetoonecheck()
        #self.showflow()
        self.showweight()
        self.Xtraining()
        self.Ytraining()
        self.showweight()
        self.learning()
        #self.showflow()
        self.showweight()
        '''
        if self.counting > self.end:
            self.stillneedtime = False
            #with open('matching.out','w') as f:
            #    f.write(str(int(sum(self.Yscaler.T[0]))))
            print('ended before complete, with {} trainning'.format(self.counting))
            print(sum(self.Yscaler.T[0]))
        if self.stillneedtime:
            self.counting += 1
            self.forward()
            self.backward()
            if np.sum(np.subtract(self.Xscaler,self.Wx)) == 0:
                if np.sum(np.subtract(self.Yscaler,self.Wy)) == 0:
                    self.stillneedtime = True
                    #with open('matching.out','w') as f:
                    #    f.write(str(int(sum(self.Yscaler.T[0]))))
                    print( "FOUND IT. -- {} -- !!".format( int( sum( self.Yscaler.T[0] ) ) ) )
        '''

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
        print('-/-\\'*15)
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
        print('--YYYYYYYYYYYYYYYYYYYYYYYYYYYYY--')
        print('  XXXX  >>>  <<< | <<<  >>>  YYYY')
        print(np.array(DDDDDDDDDD).T)
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
        #print('   Wx | Xs | Dx |-Xw-|-Yw-| Dy | Ys |  Wy ')
        #print(np.array(XXXXXXXXXXXXXXXX).T)

    def oneoverx(self, S):
        tem = []
        for i in S:
            if i[0] == 0:
                tem.append(0.0)
            else:
                tem.append(1.0/i[0])
        return np.array([tem]).T

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


mm = MaximumMatching()
for i in range(500):
    mm.train()
mm.showflow()
'''
while mm.stillneedtime and mm.counting < mm.end+2:
    mm.train()
'''
