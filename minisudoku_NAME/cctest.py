import numpy as np
class mini_sudoku:
    def __init__(self,S):
        self.sudoku = S
        self.A = [[1 for i in range(16)] for j in range(16)]
        for i in range(4):
            for j in range(4):
                if self.sudoku[i][j] != 0:
                    for k in range(1,5):
                        if self.sudoku[i][j] == k:
                            for l in range(16):
                                self.A[l][4*i+j] = 0
                            for l in range( (k-1)*4,(k*4-1) ):
                                if sum(self.A[l]) != 999:
                                    for m in range(16):
                                        self.A[l][m] = 0
                                    self.A[l][4*i + j] = 999
                                    break
        print(np.array(self.sudoku))
        print(np.array(self.A))
        self.row = [i for i in range(16)]
        self.column = [0 for i in range(16)]
        for i in range(4):
            for j in range(4):
                self.column[4*i+j] = i + j*4
        self.block = [0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
        print(self.row)
        print(self.column)
        print(self.block)

    def check4(self,Row):
        #check = []
        for leftnum in range(4):
            for top4block in range(4):
                #for i in range(4):
                check = [sum(self.A[4*leftnum+i][Row[top4block*4+j]] for j in range(4)) == 999 for i in range(4)]
                #print('b'*79)
                #print(b)
                if any(check):
                    #print('number {} in column-block {} is occupied'.format(leftnum+1,top4block+1))
                    for i in range(4):
                        for j in range(4):
                            if self.A[4*leftnum+i][Row[top4block*4+j]] != 999:
                                self.A[4*leftnum+i][Row[top4block*4+j]] = 0
                    #pass
                '''
                else:
                    #print('number {} in column-block {} is good to go'.format(leftnum+1,top4block+1))
                    for i in range(4):
                        b = [self.A[4*leftnum+i][Row[top4block*4+j]] for j in range(4)]
                        print(b)
                        print(b.count(0))
                    pass# good to go
                #print('-'*79)
                '''

    def setup4(self):
        #check = []
        for leftnum in range(4):
            for top4block in range(4):
                #for i in range(4):
                check = [sum(self.A[4*leftnum+i][top4block*4+j] for j in range(4)) == 999 for i in range(4)]
                #print('b'*79)
                #print(b)
                if any(check):
                    pass
                else:
                    for i in range(4):
                        n = [self.A[4*leftnum+i][top4block*4+j] for j in range(4)].count(0)
                        if n == 4:
                            pass
                        else:
                            for j in range(4):
                                if self.A[4*leftnum+i][top4block*4+j] != 0:
                                    self.A[4*leftnum+i][top4block*4+j] = 100*(n+1)
    def printme(self):
        print(np.array(self.A))

##running from here####
sudoku = [
          [0,0,4,3],
          [3,0,2,0],
          [0,3,0,2],
          [0,2,0,4],
          ]
s = mini_sudoku(sudoku)
s.check4(s.row)
s.check4(s.column)
s.check4(s.block)
s.printme()
s.setup4()
s.printme()
