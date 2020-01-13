import numpy as np
class MM():
    def __init__(self):
        np.set_printoptions(precision=1)
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
        self.Lookingfor = 0
        self.MaxMatching = 0
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
        #############################################################
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

        self.showing = []
    def forward(self):
        Wx = self.oneoverx(np.dot(self.Xform,self.Xway))*self.Xway
        #self.showing.append(Wx.T[0])
        #print(np.dot(self.Xform,self.Xway))
        Wxx = Wx
        #print(self.Xscaler)
        XXscaler = self.Xscaler
        for i in range(len(Wxx)):
            if Wxx[i][0]<0.1:
                Wxx[i][0] = 0
                self.Xway[i][0] = 0
                XXscaler[i][0] = 0
        self.Xscaler = self.oneoverx(XXscaler)*np.dot(self.Xform,XXscaler)
        #print(self.Xscaler)

        Wx = self.oneoverx(np.dot(self.Xform,Wxx))*Wxx
        #print(np.dot(self.Xform,Wx))
        #print(self.Xscaler)
        #print(np.around(Wx, decimals=2))
        X = np.array(([[600.0] for i in range(self.edgesum)]))
        #print(Wx.T[0])

        self.Xsendingforforward = Wx*X
        self.YGetfromforwardX = np.dot(self.forwardConnector,self.Xsendingforforward)
        #print(self.Xsendingforforward.T[0])
        #self.showing.append(X.T[0])
        self.showing.append(Wx.T[0])
        self.showing.append(self.Xway.T[0])
        #self.showing.append(self.Xsendingforforward.T[0])
        #self.showing.append(self.YGetfromforwardX.T[0])




        #print(np.dot(self.toshow,self.YGetfromforwardX))
    def backward(self):
        show_d_x = []
        Wy = self.oneoverx(np.dot(self.Yform,self.Yway))*self.Yway

        #self.showing.append(self.Yway.T[0])
        #self.showing.append(Wy.T[0])
        Wyy = Wy
        YYscaler = self.Yscaler
        for i in range(len(Wyy)):
            if Wyy[i][0]<0.1:
                Wyy[i][0] = 0
                self.Yway[i][0] = 0
                YYscaler[i][0] = 0
        self.Yscaler = self.oneoverx(YYscaler)*np.dot(self.Yform,YYscaler)
        Wy = self.oneoverx(np.dot(self.Yform,Wyy))*Wyy
        #print(np.around(Wy, decimals=2))
        Y = np.dot(self.Yform,self.YGetfromforwardX)
        self.XGetfrombackwardY = np.dot(self.backwardConnector,Wy*Y)

        #learning
        delta_Y = self.oneoverx(Y)*self.YGetfromforwardX*self.Yscaler
        self.Yway = delta_Y*self.Yway
        Xsum = np.dot(self.Xform,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        #print(Xsum)
        show_d_x.append(self.Xsendingforforward.T[0])
        show_d_x.append(self.XGetfrombackwardY.T[0])
        show_d_x.append(Xsum.T[0])
        Xtocheck = np.subtract(Xsum,(self.oneoverx(self.Xsendingforforward)*self.XGetfrombackwardY))
        #print(Xtocheck)
        #show_d_x.append(Xsum.T[0])
        #show_d_x.append(Xsum.T[0])
        show_d_x.append(Xtocheck.T[0])

        delta_X = self.oneoverx(Xsum)*Xtocheck*self.Xscaler

        #show_d_x.append(self.oneoverx(Xsum).T[0])
        #show_d_x.append(Xtocheck.T[0])
        show_d_x.append(self.Xscaler.T[0])
        show_d_x.append(delta_X.T[0])
        self.showing.append(delta_X.T[0])
        for i in range(len(self.Xscaler)):
            if self.Xscaler[i][0] == 1:
                delta_X[i][0] = 1
        #print(delta_X)
        #show_d_x.append(delta_X.T[0])

        self.showing.append(delta_X.T[0])
        self.Xway = delta_X*self.Xway
        self.showing.append(self.Xway.T[0])
        #self.showing.append(self.Yway.T[0])
        self.showing.append(np.dot(self.backwardConnector,self.Yway).T[0])
        #print(self.Xway)
        self.Xway = np.dot(self.backwardConnector,self.Yway)*self.Xway
        self.showing.append(self.Xway.T[0])
        '''
        for i in range(len(self.Xway)):
            if self.Xway[i][0] > 999.0:
                #print(self.Xway)
                self.Xway = self.Xway*0.1
                #print(self.Xway)
            elif self.Xway[i][0] < 0.001:
                #print(self.Xway)
                self.Xway[i][0] = 1
                #print(self.Xway)
        '''

        self.Yway = np.dot(self.forwardConnector,self.Xway)
        #self.showing.append(Wy.T[0])
        #print(np.around((np.array(show_d_x).T), decimals=1))
        print(np.around((np.array(self.showing).T), decimals=1))
        self.showing = []
    def train(self):
        self.forward()
        self.backward()
    def oneoverx(self, S):
        tem = []
        for i in S:
            if i[0] == 0:
                tem.append(0.0)
            else:
                tem.append(1.0/i[0])
        return np.array([tem]).T
    def show(self):
        #np.set_printoptions(precision=1)

        print(np.dot(self.toshow,self.YGetfromforwardX) )
        #print(self.N)
        #print(self.M)
        pass
mm = MM()
for i in range(5):
    mm.train()
mm.show()
