import numpy as np
class Sudoku:
    def __init__(self,S):
        self.MM = 999
        self.sudoku = S
        self.A = np.array([[1 for i in range(81)] for j in range(81)])
        for i in range(9):
            for j in range(9):
                if self.sudoku[i][j] != 0:
                    for k in range(1,10):
                        if self.sudoku[i][j] == k:
                            for l in range(81):
                                self.A[l][9*i+j] = 0
                            for l in range( (k-1)*9,(k*9) ):
                                if sum(self.A[l]) != self.MM:
                                    for m in range(81):
                                        self.A[l][m] = 0
                                    self.A[l][9*i + j] = self.MM
                                    break
        self.row = [i for i in range(81)]
        self.column = [0 for i in range(81)]
        for i in range(9):
            for j in range(9):
                self.column[9*i+j] = i + j*9
        firstblock = [0,1,2,9,10,11,18,19,20]
        self.block = []
        for i in range(3):
            for j in range(3):
                for k in firstblock:
                    self.block.append(k+3*j+27*i)
        #print(self.block)
    def checkblockfor999(self,Row):
        #check = []
        for leftnum in range(9):
            for top4block in range(9):
                #for i in range(4):
                check = [sum(self.A[9*leftnum+i][Row[top4block*9+j]] for j in range(9)) == self.MM for i in range(9)]
                #print('b'*79)
                #print(b)
                if any(check):
                    #print('number {} in column-block {} is occupied'.format(leftnum+1,top4block+1))
                    for i in range(9):
                        for j in range(9):
                            if self.A[9*leftnum+i][Row[top4block*9+j]] != self.MM:
                                self.A[9*leftnum+i][Row[top4block*9+j]] = 0
    def checkblockfor1(self,Row):
        gotanewone = False
        for leftnum in range(9):
            for top4block in range(9):
                for i in range(9):
                    if sum([self.A[9*leftnum+i][Row[top4block*9+j]] for j in range(9)]) == 1:
                        gotanewone = True
                        for j in range(9):
                            if self.A[9*leftnum+i][Row[top4block*9+j]] == 1:
                                for l in range(81):
                                    self.A[9*leftnum+i][l] = 0
                                    self.A[l][Row[top4block*9+j]] = 0
                                    self.A[9*leftnum+i][Row[top4block*9+j]] =999
                                break
                        break
        return gotanewone
    def checking1(self):
        oktogo = True
        n = 0
        while oktogo:
            self.checkblockfor999(self.row)
            self.outputme(self.row,'IIIrow'+str(n))
            #self.checkblockfor999(self.row)
            isrow = self.checkblockfor1(self.row)
            self.outputme(self.row,'OOOOIIIrow'+str(n))
            #self.outputme(self.row,'row1')
            self.checkblockfor999(self.column)
            self.outputme(self.column,'IIIcolumn'+str(n))
            ischeck = self.checkblockfor1(self.column)
            self.outputme(self.column,'OOOOIIIcolumn'+str(n))
            #self.outputme(self.column,'column1')
            self.checkblockfor999(self.block)
            self.outputme(self.block,'IIIblock'+str(n))
            isblock = self.checkblockfor1(self.block)
            self.outputme(self.block,'OOOOIIIblock'+str(n))
            n += 1
            if any([isrow,ischeck,isblock]):
                pass
            else:
                oktogo = False

    def setupblock(self,Row):
        #check = []
        for leftnum in range(9):
            for top4block in range(9):
                #for i in range(4):
                check = [sum(self.A[9*leftnum+i][Row[top4block*9+j]] for j in range(9)) == self.MM for i in range(9)]
                #print('b'*79)
                #print(b)
                if any(check):
                    pass
                else:
                    for i in range(9):
                        n = [self.A[9*leftnum+i][Row[top4block*9+j]] for j in range(9)].count(0)
                        if n == 9:
                            pass
                        else:
                            for j in range(9):
                                if self.A[9*leftnum+i][Row[top4block*9+j]] != 0:
                                    b = sum([self.A[9*leftnum+i][Row[top4block*9+k]] for k in range(9)])
                                    #self.A[9*leftnum+i][Row[top4block*9+j]] += 25*(n+1)
                                    #self.A[9*leftnum+i][Row[top4block*9+j]] += (n+1)**3
                                    #self.A[9*leftnum+i][Row[top4block*9+j]] *= n**2
                                    self.A[9*leftnum+i][Row[top4block*9+j]] += n *(self.A[9*leftnum+i][Row[top4block*9+j]]/float(b))
                                    #self.A[9*leftnum+i][Row[top4block*9+j]] += n

    def init_optimal_assignment(self):
        self.N = 81
        self.matrix = self.A
        self.maxinput = self.MM
        self.minimum_value_matching_matrix = [[(self.maxinput - element) for element in row] for row in self.matrix]
        self.operator()
    def operator(self):
        self.L = [[] for i in range(self.N)]
        self.match_list = []
        self.L_set = set([i for i in range(self.N)])
        self.free_set = set([i for i in range(self.N)])
        self.tree_list = []
        self.bfs_visited_edges_list = []
        self.dfs_lvl = 0
        self.dfs_visited_edges_list = []
        self.dfs_trying_path_list = []
    def step_one(self):
        for i in range(self.N):
            min_value_in_row = min(self.minimum_value_matching_matrix[i])
            self.minimum_value_matching_matrix[i] = [(element - min_value_in_row) for element in self.minimum_value_matching_matrix[i]]
        self.step_two()
    def step_two(self):
        for i in range(self.N):
            minimum_value_in_column = min([j[i] for j in self.minimum_value_matching_matrix])
            for j in range(self.N):
                self.minimum_value_matching_matrix[j][i] = self.minimum_value_matching_matrix[j][i] - minimum_value_in_column
        self.step_three()
    def step_three(self):
        self.operator()
        for i in range(self.N):
            for j in range(self.N):
                if self.minimum_value_matching_matrix[i][j] == 0:
                    self.L[i].append(j)
        self.BreadthFirstSearch(self.L_set)
        if self.N == len(self.match_list):
            self.step_five()
        else:
            self.step_four()
    def step_four(self):
        row_remain = [i for i in range(self.N)]
        column_remain = [i for i in range(self.N)]
        row_removed = []
        column_removed = []
        for i in self.match_list:
            row_remain.remove(i[0])
            row_removed.append(i[0])
        for i in row_remain:
            for j in self.L[i]:
                if j in column_remain:
                    column_remain.remove(j)
                    column_removed.append(j)
                    for k in self.match_list:
                        if k[1] == j:
                            row_remain.append(k[0])
                            row_removed.remove(k[0])
        tem = [self.minimum_value_matching_matrix[i][j] for i in row_remain for j in column_remain]
        mini_value_4 = min(tem)
        for i in row_remain:
            for j in column_remain:
                self.minimum_value_matching_matrix[i][j] = self.minimum_value_matching_matrix[i][j] - mini_value_4
        for i in row_removed:
            for j in column_removed:
                self.minimum_value_matching_matrix[i][j] = self.minimum_value_matching_matrix[i][j] + mini_value_4
        self.step_three()
    def step_five(self):
        #optimal_value = 0
        #self.match_list = sorted(self.match_list)
        step = [' ' for i in range(81)]
        for i in self.match_list:
            for j in range(9):
                for k in range(9):
                    if j*9+k == i[1]:
                        step[j*9+k] = str(int(int(i[0])/9)+1)
        for i in range(9):
            for j in range(9):
                print(' '+step[i*9+j]+' ',end='')

            print()

    def BreadthFirstSearch(self,left_set):
        edge_list = []
        vertix_candidate_set = set()
        viable_vertix_set = set()
        for i in left_set:
            for j in self.L[i]:
                if not [i,j] in self.match_list:
                    if not [i,j] in self.bfs_visited_edges_list:
                        edge_list.append([i,j])
                        self.bfs_visited_edges_list.append([i,j])
                        vertix_candidate_set.add(j)
        if vertix_candidate_set:
            self.tree_list.append(edge_list)
            for i in vertix_candidate_set:
                if i in self.free_set:
                    viable_vertix_set.add(i)
            if viable_vertix_set:
                for i in viable_vertix_set:
                    self.dfs_visited_edges_list = []
                    self.dfs_trying_path_list = []
                    self.dfs_lvl = 0
                    self.DepthFirstSearch(i)
                self.tree_list = []
                self.bfs_visited_edges_list = []
                self.BreadthFirstSearch(self.L_set)
            else:
                rightside_vertix_lookingfor_edge_list = []
                rightside_looking_leftside_vertix_candidate_set = set()
                for i in vertix_candidate_set:
                    for j in self.match_list:
                        if j[1] == i:
                            if not j in self.bfs_visited_edges_list:
                                rightside_vertix_lookingfor_edge_list.append(j)
                                self.bfs_visited_edges_list.append(j)
                                rightside_looking_leftside_vertix_candidate_set.add(j[0])
                            break
                if rightside_vertix_lookingfor_edge_list:
                    self.tree_list.append(rightside_vertix_lookingfor_edge_list)
                    self.BreadthFirstSearch(rightside_looking_leftside_vertix_candidate_set)
    def DepthFirstSearch(self, vertix):
        for i in self.tree_list[len(self.tree_list)-1-self.dfs_lvl]:
            if i[1] == vertix:
                if len(self.tree_list)-1-self.dfs_lvl == 0:
                    if i[0] in self.L_set:
                        self.dfs_trying_path_list.append(i)
                        self.update()
                        break
                else:
                    if not i in self.dfs_visited_edges_list:
                        self.dfs_visited_edges_list.append(i)
                        self.dfs_trying_path_list.append(i)
                        for j in self.match_list:
                            if j[0] == i[0]:
                                if not j in self.dfs_visited_edges_list:
                                    self.dfs_visited_edges_list.append(j)
                                    self.dfs_trying_path_list.append(j)
                                    self.dfs_lvl += 2
                                    self.DepthFirstSearch(j[1])
                                    break
    def update(self):
            blacklist = []
            for i in self.dfs_trying_path_list:
                if i in self.match_list:
                    blacklist.append(i)
            for i in self.dfs_trying_path_list:
                self.match_list.append(i)
            for i in blacklist:
                self.match_list.remove(i)
                self.match_list.remove(i)
            for i in range(len(self.tree_list)):
                if self.dfs_trying_path_list[i] in self.tree_list[i]:
                    self.tree_list[i].remove(self.dfs_trying_path_list[i])
            self.L_set.discard(self.dfs_trying_path_list[-1][0])
            self.free_set.discard(self.dfs_trying_path_list[0][1])

    def outputme(self, row, s):
        line1 = []
        #line1.append('   ')
        for i in range(81):
            for j in range(9):
                for k in range(9):
                    if row[i] == j*9+k:
                        line1.append( (str(j+1)+str(k+1)).rjust(4) )
        allline = [[' ' for i in range(81)] for j in range(81)]
        for i in range(81):
            for j in range(81):
                allline[i][j] = str(self.A[i][row[j]]).rjust(4)

        with open(s+'.txt','w') as f:
            f.write('   ')
            for i in range(9):
                for j in range(9):
                    f.write(line1[i*9+j])
                f.write('|')
            f.write('\n')
            for i in range(9):
                for j in range(9):
                    f.write(str(i+1)+': ')
                    for k in range(9):
                        for l in range(9):
                            f.write(allline[i*9+j][k*9+l])
                        f.write('|')
                    f.write('\n')
                f.write('-'*340)
                f.write('\n')
    def run(self):
        #self.outputme(self.row)
        self.checkblockfor999(self.row)
        #self.outputme(self.row)
        self.checkblockfor999(self.column)
        #self.outputme(self.column)
        self.checkblockfor999(self.block)
        self.checking1()
        '''
        self.outputme(self.block,'block999')
        self.outputme(self.column,'column999')
        self.outputme(self.row,'row999')
        self.checkblockfor1(self.row)
        self.outputme(self.row,'row1')
        self.checkblockfor1(self.column)
        self.outputme(self.column,'column1')
        self.checkblockfor1(self.block)
        self.outputme(self.block,'block1')
        #self.printme()
        '''
        self.outputme(self.row,'row1')
        #self.checkblockfor1(self.column)
        self.outputme(self.column,'column1')
        #self.checkblockfor1(self.block)
        self.outputme(self.block,'block1')
        for i in range(1):
            self.setupblock(self.row)
            self.setupblock(self.column)
            self.setupblock(self.block)

        self.outputme(self.row,'finalrow')
        for i in range(5):
            self.A[i+40][52] = 20
        self.init_optimal_assignment()
        self.step_one()
        #self.outputme(self.block)
'''
sudoku = [
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],

           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],

           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,0, 0,0,0],
          ]

sudoku = [
           [0,2,0, 0,0,8, 0,0,0],
           [0,3,0, 4,2,0, 0,1,0],
           [4,0,0, 0,0,0, 5,3,0],

           [6,0,8, 0,0,5, 0,7,0],
           [7,9,0, 0,0,0, 2,8,4],
           [2,0,0, 0,7,4, 0,0,0],

           [8,0,0, 3,5,6, 0,2,0],
           [0,5,0, 0,4,0, 3,0,0],
           [0,0,4, 0,0,0, 0,0,7],
          ]
'''
sudoku = [
           [0,3,2, 0,0,0, 0,0,0],
           [4,0,0, 8,3,7, 0,6,0],
           [0,0,0, 5,0,0, 0,0,0],

           [2,0,4, 0,0,0, 0,0,0],
           [0,7,0, 0,0,3, 5,0,4],
           [0,0,8, 0,2,0, 0,9,7],

           [0,6,0, 0,0,0, 0,0,8],
           [0,0,0, 2,0,0, 0,0,0],
           [7,0,0, 4,0,0, 0,3,6],
          ]
s = Sudoku(sudoku)
s.run()
