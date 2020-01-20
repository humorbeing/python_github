import numpy as np#just to formated print
class hungarian:
    def __init__(self):
        with open('assign.in','r') as f:
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
        print('INIT--'*5)
        print('/'*79)
        print('\\'*79)
        print('self.N: {}'.format(self.N))
        print('self.matrix:\n {}'.format(np.array(self.matrix)))
        print('*'*79)
        print('self.maxinput: {}'.format(self.maxinput))
        self.minimum_value_matching_matrix = [[(self.maxinput - element) for element in row] for row in self.matrix]
        print('self.mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('*'*79)
        print('did self.matrix change:\n {}'.format(np.array(self.matrix)))
        print(' --INIT END-- '*5)
        print('/'*79)
        print('\\'*79)
        self.operator()

    def operator(self):
        ##########maximum matching operators#########
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
        #min_value_in_row = 0
        print('-in step 11111111111111111111111111111111')
        print('-old mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        for i in range(self.N):
            min_value_in_row = min(self.minimum_value_matching_matrix[i])
            print('-min value in this row: {} is {}'.format(self.minimum_value_matching_matrix[i],min_value_in_row))
            #print('-subtracting mini value on each row.')
            self.minimum_value_matching_matrix[i] = [(element - min_value_in_row) for element in self.minimum_value_matching_matrix[i]]

        print('-new mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('-did step 1 change self.matrix?:\n {}'.format(np.array(self.matrix)))
        print('-END step 1111111111111111111111111111111111111')
        self.step_two()

    def step_two(self):
        print('--in step 2222222222222222222222222222222')
        print('--old mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))


        for i in range(self.N):

            minimum_value_in_column = min([j[i] for j in self.minimum_value_matching_matrix])
            print('--the minimum value in {} column {} is {}'.format(i,[j[i] for j in self.minimum_value_matching_matrix],minimum_value_in_column))
            for j in range(self.N):
                self.minimum_value_matching_matrix[j][i] = self.minimum_value_matching_matrix[j][i] - minimum_value_in_column



        print('--new mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('--did step 2 change self.matrix?:\n {}'.format(np.array(self.matrix)))
        print('--END step 2222222222222222222222222222222222')
        self.step_three()

    def step_three(self):
        print('---in step 3333333333333333333333333')
        for i in range(self.N):
            for j in range(self.N):
                if self.minimum_value_matching_matrix[i][j] == 0:
                    self.L[i].append(j)
        print('---maked self.L {} from self.mini matrix: \n {}'.format(self.L,np.array(self.minimum_value_matching_matrix)))
        print('---running Maximum matching')
        self.BreadthFirstSearch(self.L_set)
        print('---is maximum matching {} == N: {} ?'.format(len(self.match_list),self.N))
        if self.N == len(self.match_list):
            print('---yes,we got the answer')
            print('---END step 3333333333333333333333333333333')
            print('---go step 5')
        else:
            print('---nope,is N bigger?')
            if self.N > len(self.match_list):
                print('---yes')
                print('---END step 3333333333333333333333333333333')
                #print('---yes, we go step 4')
                self.step_four()
            else:
                print('---NO? ERROR')
                print('---END step 3333333333333333333333333333333')

        #print(self.match_list)

    def step_four(self):
        print('----in step 4444444444444444444444444444444')
    ##########maximum match#############
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
    ##########some functions#############
    def test(self):
        #print(self.minimum_value_matching_matrix[::][1])
        pass
optimal_assingment = hungarian()
optimal_assingment.step_one()
#optimal_assingment.test()
