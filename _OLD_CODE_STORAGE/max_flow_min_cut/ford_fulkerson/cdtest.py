# import collections.counter as Counter
class FordFulkerson:
    def __init__(self):
        with open('network.inp', 'r') as f:
            a = f.readlines()
        # print(a)
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        # self.graph = [['X' for _ in range(self.N)] for _ in range(self.N)]
        self.residual_matrix = [['X' for _ in range(self.N)] for _ in range(self.N)]
        # self.residual_matrix[0][0] = 0
        for i in a:
            b = i.split()
            for j in range(int(b[1])):
                # self.graph[int(b[0]) - 1][int(b[j * 2 + 2]) - 1] = int(b[j * 2 + 3])
                # self.graph[int(b[j * 2 + 2]) - 1][int(b[0]) - 1] = int(b[j * 2 + 3]) * -1
                self.residual_matrix[int(b[0])-1][int(b[j*2+2])-1] = int(b[j*2+3])
                self.residual_matrix[int(b[j * 2 + 2]) - 1][int(b[0]) - 1] = 0
                pass
        self.path = []
        self.deadend = []
        self.visited = []
        self.min_cut = []
        self.findall = False
        self.recursion_n = 0
        # self.goodtogo = True
        pass

    def looking_for_augment_path_in_path_list(self):
        # need_recursion = True
        # print(self.graph)
        if self.path:
            if self.path[-1] != self.N - 1:
                got_one = False
                for i in range(self.N):
                    if self.residual_matrix[self.path[-1]][i] != 'X':
                        if self.residual_matrix[self.path[-1]][i] != 0:
                            if i in self.deadend:
                                pass  # deadend
                            else:
                                if i in self.path:
                                    pass
                                else:
                                    self.path.append(i)
                                    got_one = True
                                    break
                if got_one:
                    pass
                    # self.looking_for_augment_path_in_path_list()
                else:
                    pass
                    self.deadend.append(self.path[-1])
                    del self.path[-1]
                    # self.looking_for_augment_path_in_path_list()
            else:
                pass
                self.update_path()
                self.path = []
                self.deadend = []
                # self.looking_for_augment_path_in_path_list()
        else:
            if 0 in self.deadend:
                self.deadend = []
                self.findall = True
                pass
            else:
                self.path.append(0)
                # self.looking_for_augment_path_in_path_list()
        if self.findall:
            pass
        else:
            if self.recursion_n < 950:
                self.recursion_n += 1
                self.looking_for_augment_path_in_path_list()

    def update_path(self):
        min_capacity = min([abs(self.residual_matrix[self.path[i]][self.path[i+1]]) for i in range(len(self.path)-1)])
        for i in range(len(self.path) - 1):
            if self.residual_matrix[self.path[i]][self.path[i+1]] > 0 \
                    or self.residual_matrix[self.path[i+1]][self.path[i]] < 0:
                self.residual_matrix[self.path[i]][self.path[i + 1]] -= min_capacity
                self.residual_matrix[self.path[i + 1]][self.path[i]] -= min_capacity
            else:
                self.residual_matrix[self.path[i]][self.path[i + 1]] += min_capacity
                self.residual_matrix[self.path[i + 1]][self.path[i]] += min_capacity
        pass

    def min_cut_node_search(self, node):
        self.visited.append(node)
        for i in range(self.N):
            if i not in self.visited:
                if self.residual_matrix[node][i] != 'X':
                    if self.residual_matrix[node][i] != 0:
                        if self.residual_matrix[i][node] != 0:
                            self.min_cut_node_search(i)
        pass

    def summary(self):
        max_flow = 0
        for i in self.residual_matrix[-1]:
            if i != 'X':
                max_flow += i * (-1)
        for i in self.visited:
            for j, flow_potential in enumerate(self.residual_matrix[i]):
                if j not in self.visited:
                    if flow_potential == 0:
                        if self.residual_matrix[j][i] < 0:
                            self.min_cut.append([i, j])
        print(max_flow, len(self.min_cut))
        for i in self.min_cut:
            print(i[0]+1, i[1]+1, (self.residual_matrix[i[1]][i[0]] * (-1)))

    def testrun(self):
        # cnt = Counter()
        # for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
        #     cnt[word] += 1
        # print(cnt)
        while not self.findall:
            self.looking_for_augment_path_in_path_list()
            self.recursion_n = 0
        self.min_cut_node_search(0)
        self.summary()
        # self.path = [0, 1, 2, 3, 6]
        # self.update_path()
        pass
ff = FordFulkerson()
ff.testrun()

