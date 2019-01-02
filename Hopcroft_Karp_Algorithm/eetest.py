import numpy as np
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
        if self.N == 0 or self.M == 0:
            pass
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
            '''
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
            if self.N<self.M:
                self.A = self.transpose(self.A)
            else:
                self.N,self.M = self.M,self.N
            '''
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
        self.path_list = []
        self.L_set = set([i for i in range(self.N)])
        self.free_set = set([i for i in range(self.M)])
        self.R_set = set([i for i in range(self.M)])
        self.tree_list = []
        self.dfs_lvl = 0
        self.dfs_visited_edges_list = []
        self.dfs_trying_path_list = []
        #####jug
        self.path_empty = False


    def run(self):
        print("-RRRRRRRun it")
        print("-L nodes connected edges")
        print(self.L)
        print("-Lset nodes set aviable")
        print(self.L_set)
        print("-"*79)
        print("-R nodes and edges")
        print(self.R)
        print("-freeset nodes set aviable")
        print(self.free_set)
        print("-"*79)
        #where it begins
        while not self.path_empty:
            print('-WWW while loop')
            if self.L_set:
                self.BreadthFirstSearch(self.L_set)
                self.path_empty = True
            else:
                pass#end
    def BreadthFirstSearch(self,left_set):
        print('--BBB breadth first search')
        edge_list = []
        vertix_candidate_set = set()
        viable_vertix_set = set()
        for i in left_set:
            #print('--i: {} in leftset {}'.format(i,leftset))
            for j in self.L[i]:
                edge_list.append([i,j])
                vertix_candidate_set.add(j)
        print('--after LF, cadidate nodes: {}'.format(vertix_candidate_set))
        self.tree_list.append(edge_list)
        #checking for usable vertex
        for i in vertix_candidate_set:
            print('--LF viable:i {} in Freeset {}.'.format(i,self.free_set))
            if i in self.free_set:
                viable_vertix_set.add(i)
        print('--Viable x we found: {}'.format(viable_vertix_set))
        print('--if viable is not empty?')
        if viable_vertix_set:
            print('--found x, let do DFS now')
            print('--should we stop outside loop for update?')

            for i in viable_vertix_set:
                print('--reseting dfs_visited_edges_list for i {}'.format(i))
                self.dfs_visited_edges_list = []
                print('--reseting dfs_trying_path_list for i {}'.format(i))
                self.dfs_trying_path_list = []
                print('--reseting dfs_lvl to 0 for i {}'.format(i))
                self.dfs_lvl = 0
                print('--sending i {} to DFS'.format(i))
                self.DepthFirstSearch(i)

        else:
            print('--there is no viable x this round, looking for more')
    def DepthFirstSearch(self, vertix):
        print('---DDD depth first search')
        print('---use tree and vertice we got to look for match')
        print('---tree is {}'.format(self.tree_list))
        print('---vertix is {}'.format(vertix))
        print('---DFS_lvl is {} which is on lvl {} in tree with {} lvl'.format(self.dfs_lvl,(len(self.tree_list)-1-self.dfs_lvl),len(self.tree_list)))
        for i in self.tree_list[len(self.tree_list)-1-self.dfs_lvl]:
            #print('OOOOOOO test tree edge {} and right side node {}'.format(i,i[1]))
            print('---if i {}-left node {} is in self.L_set {}'.format(i,i[0],self.L_set))
            print('---AND i {}-right node {} is as same as vertix {} we checking'.format(i,i[1],vertix))
            if i[0] in self.L_set and i[1] == vertix:
                print('---yes it is. appending to trying path {}'.format(self.dfs_trying_path_list))
                self.dfs_trying_path_list.append(i)
                print('---after append {}'.format(self.dfs_trying_path_list))
                self.update()
                break
            else:
                print('---nope')

    def update(self):
        print('----UUUUUUUpdating')
        print('----update self.match_list {}'.format(self.match_list))
        print('----with self.dfs_trying_path_list {}'.format(self.dfs_trying_path_list))


    def transpose(self,S):
        tem = [[0 for j in range(len(S))] for i in range(len(S[0]))]
        for i in range(len(S)):
            for j in range(len(S[0])):
                tem[j][i] = S[i][j]
        return tem

mm = hopcroft_karp()
mm.run()
