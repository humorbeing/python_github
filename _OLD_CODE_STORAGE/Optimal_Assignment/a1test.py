class hungarian:
    def __init__(self):
        with open('assign.txt','r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.matrix = []
        self.maxinput = 0
        def findmax(s):
            n = int(s)
            if n>self.maxinput:
                self.maxinput = n
            return n
        for i in a:
            self.matrix.append([findmax(j) for j in i.split()])
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
        s=[]
        for i in self.match_list:
            optimal_value += self.matrix[i[0]][i[1]]
            s.append(' '.join((str(i[0]),str(i[1]))))
        s = '\n'.join(s)
        s = '\n'.join((s,str(optimal_value)))
        with open('assign.out','w') as f:
            f.writelines(s)
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
optimal_assingment = hungarian()
optimal_assingment.step_one()
