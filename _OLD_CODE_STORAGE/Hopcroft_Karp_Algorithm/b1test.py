class hopcroft_karp:
    def __init__(self):
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
        self.A = A
        self.L = [[] for i in range(self.N)]
        self.R = [[] for i in range(self.M)]
        for i in range(self.N):
            for j in range(self.M):
                if self.A[i][j] == 1:
                    self.L[i].append(j)
                    self.R[j].append(i)
        self.operator()

    def operator(self):
        self.match_list = []
        self.L_set = set([i for i in range(self.N)])
        self.free_set = set([i for i in range(self.M)])
        self.tree_list = []
        self.bfs_visited_edges_list = []
        self.dfs_lvl = 0
        self.dfs_visited_edges_list = []
        self.dfs_trying_path_list = []

    def run(self):
        self.BreadthFirstSearch(self.L_set)
        with open('matching.out','w') as f:
            f.write(str(len(self.match_list)))
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
            self.match_list = [j for j in self.match_list if j != i]
        for i in range(len(self.tree_list)):
            for j in range(len(self.tree_list[i])):
                if self.dfs_trying_path_list[i] == self.tree_list[i][j]:
                    del self.tree_list[i][j]
                    break
        self.L_set.discard(self.dfs_trying_path_list[-1][0])
        self.free_set.discard(self.dfs_trying_path_list[0][1])
mm = hopcroft_karp()
mm.run()
