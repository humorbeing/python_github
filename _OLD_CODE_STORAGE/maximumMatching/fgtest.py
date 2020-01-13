import numpy as np
class MM():
    def __init__(self):
        np.set_printoptions(precision=1)
        self.counting = 0
        self.checking = 3
        self.end = 5
        self.stillneedtime = True
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

        tem = []
        ping = [0 for i in range(self.N)]
        self.Xform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
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
        self.Yform = np.array([[0 for j in range(self.edgesum)] for i in range(self.edgesum)])
        self.toshow = np.array(([[0 for j in range(self.edgesum)] for i in range(self.M)]))
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
        self.Xscaler = np.array([[np.sum(self.Xform[i])] for i in range(self.edgesum)])
        self.Yscaler = np.array([[np.sum(self.Yform[i])] for i in range(self.edgesum)])
        self.Xway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Yway = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.delta_X = np.array(([[0.0 for i in range(self.edgesum)]])).T
        self.delta_Y = np.array(([[0.0 for i in range(self.edgesum)]])).T
        self.Xfix = np.array(([[1.0 for i in range(self.edgesum)]])).T
        self.Yfix = np.array(([[1.0 for i in range(self.edgesum)]])).T

    def forward(self):
        self.oldx = self.Xway
        self.oldy = self.Yway
        self.Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
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
        X = np.array(([[6.0] for i in range(self.edgesum)]))
        self.Xsendingforforward = self.Wx*X
        self.YGetfromforwardX = np.dot(self.forwardConnector,self.Xsendingforforward)

    def backward(self):
        XXXXXXXXXXXXXXXX = []
        YYYYYYYYYYYYYYYY = []
        SSSSSSSSSSSSSSSS = []
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
        self.ychange = self.Yway
        Y = np.dot(self.Yform,self.YGetfromforwardX)
        self.XGetfrombackwardY = np.dot(self.backwardConnector,self.Wy*Y)
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
                    print(sum(self.Yscaler.T[0]))
    def oneoverx(self, S):
        tem = []
        for i in S:
            if i[0] == 0:
                tem.append(0.0)
            else:
                tem.append(1.0/i[0])
        return np.array([tem]).T

mm = MM()
while mm.stillneedtime and mm.counting < mm.end+2:
    mm.train()