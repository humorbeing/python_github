#import numpy as np
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
        self.bfs_visited_edges_list = []
        self.dfs_lvl = 0
        self.dfs_visited_edges_list = []
        self.dfs_trying_path_list = []
        #####jug
        self.path_empty = False
        self.nn = 0
        self.limit = 500

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
                for i in range(3):
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
                if not [i,j] in self.match_list:
                    print('--is [{},{}] edge in bfs_visited: {}'.format(i,j,self.bfs_visited_edges_list))
                    if not [i,j] in self.bfs_visited_edges_list:

                        edge_list.append([i,j])
                        self.bfs_visited_edges_list.append([i,j])
                        print('--no,adding this to bfs visited list: {}'.format(self.bfs_visited_edges_list))
                        vertix_candidate_set.add(j)
        print('--is candidate empty??')
        if vertix_candidate_set:
            print('--its not empty,{} to right side possible nodes: {}'.format(left_set,vertix_candidate_set))

            self.tree_list.append(edge_list)
            #checking for usable vertex
            for i in vertix_candidate_set:
                print('--if this candidate i: {} is in free_set(Right side open node): {} as open node.'.format(i,self.free_set))
                if i in self.free_set:
                    print('--yes')
                    viable_vertix_set.add(i)
                else:
                    print('--no')
            print('--did we find any open nodes??')
            if viable_vertix_set:
                print('--yes.open nodes are: {}'.format(viable_vertix_set))
                #print('--should we stop outside loop for update?')
                print('--sending to DepthFirst search.')
                for i in viable_vertix_set:
                    #print('--reseting dfs_visited_edges_list for i {}'.format(i))
                    self.dfs_visited_edges_list = []
                    #print('--reseting dfs_trying_path_list for i {}'.format(i))
                    self.dfs_trying_path_list = []
                    #print('--reseting dfs_lvl to 0 for i {}'.format(i))
                    self.dfs_lvl = 0
                    #print('--sending i {} to DFS'.format(i))
                    self.DepthFirstSearch(i)

                print('A'*79)
                print('A'*79)
                print('--self.matching {}'.format(self.match_list))
                self.tree_list = []
                self.bfs_visited_edges_list = []
                print('--self.lL_set: {}'.format(self.L_set))
                print('--self.free_set: {}'.format(self.free_set))
                print('--!!!starting a new finding from after match.')
                #self.BreadthFirstSearch(self.L_set)
            else:
                rightside_vertix_lookingfor_edge_list = []
                rightside_looking_leftside_vertix_candidate_set = set()
                print('--there is no open node on right side this round, looking for more')
                print('--tree or edge we got this round {}'.format(self.tree_list))
                print('--sending candidate {} to find connected neighbors'.format(vertix_candidate_set))
                for i in vertix_candidate_set:
                    for j in self.match_list:
                        if j[1] == i:
                            print('--is [{}] edge in bfs_visited: {}'.format(j,self.bfs_visited_edges_list))
                            if not j in self.bfs_visited_edges_list:
                                rightside_vertix_lookingfor_edge_list.append(j)
                                self.bfs_visited_edges_list.append(j)
                                print('--no,adding this to bfs visited list: {}'.format(self.bfs_visited_edges_list))
                                rightside_looking_leftside_vertix_candidate_set.add(j[0])
                            break
                print('--got rightside_vertix_lookingfor_edge_list {}'.format(rightside_vertix_lookingfor_edge_list))
                print('--got rightside_looking_leftside_vertix_candidate_set {}'.format(rightside_looking_leftside_vertix_candidate_set))
                if rightside_vertix_lookingfor_edge_list:
                    self.tree_list.append(rightside_vertix_lookingfor_edge_list)
                    print('--tree list updated {}'.format(self.tree_list))
                    print('--starting a new BFS from candidates')
                    self.BreadthFirstSearch(rightside_looking_leftside_vertix_candidate_set)
                    '''
                    self.nn += 1
                    if self.nn < self.limit:
                        print('timer is {}'.format(self.nn))
                        self.BreadthFirstSearch(rightside_looking_leftside_vertix_candidate_set)
                    else:
                        print('T'*79)
                        print('T'*79)
                        print('T'*79)
                    '''
                else:
                    print('nothing <<<<----- on right side vertices go left side')
                    print('!'*79)
                    print('*'*79)
                    print('!'*79)
                    print('The bipartite graph is:')
                    print()
                    print("- {} L nodes connected edges".format(self.N))
                    print(self.L)
                    print("-"*79)
                    print()
                    print("- {} R nodes connected edges".format(self.M))
                    print(self.R)
                    print("-"*79)
                    print()
                    print('we got match as {}'.format(self.match_list))
                    print()
                    print('The Maximum matching is {}.'.format(len(self.match_list)))

        else:
            print('----->>> nothing on left side vertices go right side')
            print('!'*79)
            print('*'*79)
            print('!'*79)
            print('The bipartite graph is:')
            print()
            print("- {} L nodes connected edges".format(self.N))
            print(self.L)
            print("-"*79)
            print()
            print("- {} R nodes connected edges".format(self.M))
            print(self.R)
            print("-"*79)
            print()
            print('we got match as {}'.format(self.match_list))
            print()
            print('The Maximum matching is {}.'.format(len(self.match_list)))

    def DepthFirstSearch(self, vertix):
        print('---DDD depth first search')
        print('---use tree and vertice we got to look for match')
        print('---tree is {}'.format(self.tree_list))
        print('---vertix is {}'.format(vertix))
        print('---DFS_lvl is {} which is on lvl {} in tree with {} lvl'.format(self.dfs_lvl,(len(self.tree_list)-1-self.dfs_lvl),len(self.tree_list)))
        print('---leaves are {}'.format(self.tree_list[len(self.tree_list)-1-self.dfs_lvl]))
        for i in self.tree_list[len(self.tree_list)-1-self.dfs_lvl]:
            #print('OOOOOOO test tree edge {} and right side node {}'.format(i,i[1]))
            print('---i {}-right node {} is as same as vertix {} we checking'.format(i,i[1],vertix))
            if i[1] == vertix:
                print('---yes,same num. are we on top lvl as tree layer {} = 0?'.format(len(self.tree_list)-1-self.dfs_lvl))
                if len(self.tree_list)-1-self.dfs_lvl == 0:
                    print('---yes,if i {}-left node {} is in self.L_set {}'.format(i,i[0],self.L_set))
                    if i[0] in self.L_set:
                        print('---yes it is. appending to trying path {}'.format(self.dfs_trying_path_list))
                        self.dfs_trying_path_list.append(i)
                        print('---after append {}'.format(self.dfs_trying_path_list))
                        #print('---updating to self.path_list {}'.format(self.path_list))
                        #self.path_list.append(self.dfs_trying_path_list)
                        #print('---updated self.path_list {}'.format(self.path_list))
                        print('---one by one update?')
                        self.update()
                        break
                else:
                    print('---Nope, checking if i {} is not in visited {} is {}'.format(i,self.dfs_visited_edges_list,not i in self.dfs_visited_edges_list))
                    if not i in self.dfs_visited_edges_list:
                        print('---its not in there.this {} is our guy'.format(i))
                        print('---update visited list {}'.format(self.dfs_visited_edges_list))
                        self.dfs_visited_edges_list.append(i)
                        print('---done visted list {}'.format(self.dfs_visited_edges_list))
                        print('---update trying list {}'.format(self.dfs_trying_path_list))
                        self.dfs_trying_path_list.append(i)
                        print('---done trying list {}'.format(self.dfs_trying_path_list))
                        print('---looking thru matchlist {} for connected node for [{}, X]'.format(self.match_list, i[0]))
                        for j in self.match_list:
                            if j[0] == i[0]:
                                print('---{} is it. see if we visited it before {}'.format(j,self.dfs_visited_edges_list))
                                if not j in self.dfs_visited_edges_list:
                                    print('---nope.this is new path')
                                    print('---update visited list {}'.format(self.dfs_visited_edges_list))
                                    self.dfs_visited_edges_list.append(j)
                                    print('---done visted list {}'.format(self.dfs_visited_edges_list))
                                    print('---update trying list {}'.format(self.dfs_trying_path_list))
                                    self.dfs_trying_path_list.append(j)
                                    print('---done trying list {}'.format(self.dfs_trying_path_list))
                                    print('---sending to next lvl')
                                    self.dfs_lvl += 2
                                    self.DepthFirstSearch(j[1])
                                    break

    def update(self):
        print('----UUUUUUUpdating')
        print('----updating match list')
        print('----with self.dfs_trying_path_list {}'.format(self.dfs_trying_path_list))
        print('----updating self.match_list {}'.format(self.match_list))
        blacklist = []
        for i in self.dfs_trying_path_list:
            if i in self.match_list:
                blacklist.append(i)
        for i in self.dfs_trying_path_list:
            self.match_list.append(i)
        for i in blacklist:
            print('----blacklist is {},try remove from self.match_list {}'.format(i,self.match_list))
            self.match_list = [j for j in self.match_list if j != i]
            print('----removed from self.match_list {}'.format(self.match_list))
        print('----updated self.match_list {}'.format(self.match_list))



        print('----updating tree')
        print('----updating self.dfs_trying_path_list {} to tree'.format(self.dfs_trying_path_list))
        print('----tree is {}'.format(self.tree_list))
        for i in range(len(self.tree_list)):
            for j in range(len(self.tree_list[i])):
                if self.dfs_trying_path_list[i] == self.tree_list[i][j]:
                    del self.tree_list[i][j]
                    break
        print('----updated tree is {}'.format(self.tree_list))

        print('----now updating lset and free_set,lset first')
        print('----removing trying_path_list[-1][0] {} from L_set {}'.format(self.dfs_trying_path_list[-1][0],self.L_set))
        self.L_set.discard(self.dfs_trying_path_list[-1][0])
        print('----new L_set is {}'.format(self.L_set))
        print('----removing trying_path_list[0][1] {} from freeset {}'.format(self.dfs_trying_path_list[0][1],self.free_set))
        self.free_set.discard(self.dfs_trying_path_list[0][1])
        print('----new freeset is {}'.format(self.free_set))



    def test(self):
        self.tree_list = [[[1,8],[2,2],[2,8]]]
        self.dfs_trying_path_list = [[2,2]]
        self.match_list = [[4,1],[3,0]]
        self.update()

mm = hopcroft_karp()
mm.run()
#mm.test()
