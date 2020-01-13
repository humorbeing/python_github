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

    def init_optimal_assignment(self):
        self.N = 16
        self.matrix = self.A
        self.maxinput = 999
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
        optimal_value = 0
        #self.match_list = sorted(self.match_list)
        for i in self.match_list:
            print(i)
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
#s.printme()
s.setup4()
#s.printme()
s.init_optimal_assignment()
s.step_one()
