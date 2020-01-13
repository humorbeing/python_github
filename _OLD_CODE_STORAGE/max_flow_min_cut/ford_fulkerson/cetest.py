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
        pass

    def looking_for_augment_path_in_path_list(self):
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
                    self.looking_for_augment_path_in_path_list()
                else:
                    pass
                    self.deadend.append(self.path[-1])
                    del self.path[-1]
                    self.looking_for_augment_path_in_path_list()
            else:
                pass
                self.update_path()
                self.path = []
                self.deadend = []
                self.looking_for_augment_path_in_path_list()
        else:
            if 0 in self.deadend:
                self.deadend = []
                pass
            else:
                self.path.append(0)
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
        # self.residual_matrix = [['X', 'X', 'X', 0, 'X', 0, 0, 0, 60, 'X', 'X'], ['X', 'X', 0, 44, -14, -25, 2, -30, 8, 14, 3], ['X', 10, 'X', -27, 0, 0, 1, 92, -15, 10, 'X'], [-61, -6, 2, 'X', 0, 0, -13, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 96, -62, 29, 68, 61, 16], [-76, 10, 67, -1, 0, 'X', 0, 0, 47, 13, 'X'], [-52, 0, -4, 67, 22, 18, 'X', -19, 96, 87, 'X'], [-97, 0, 0, 0, 0, -3, 56, 'X', 18, 3, 10], [-8, -11, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -15, 0, -6, 28, 'X', 0], ['X', -26, 'X', 'X', -71, 'X', 'X', -43, -69, -85, 'X']]
        self.looking_for_augment_path_in_path_list()
        self.min_cut_node_search(0)
        self.summary()
        # self.path = [0, 1, 2, 3, 6]
        # self.update_path()
        pass
ff = FordFulkerson()
ff.testrun()
'''
1 <class 'list'>: [['X', 'X', 'X', 0, 'X', 0, 0, 0, 68, 'X', 'X'], ['X', 'X', -2, 50, -14, -25, 2, -30, 0, 14, 3], ['X', 8, 'X', -23, 0, -6, 1, 92, -15, 10, 'X'], [-61, 0, 6, 'X', 0, 0, -15, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 96, -58, 29, 68, 61, 20], [-76, 10, 61, -1, 0, 'X', 0, 3, 47, 16, 'X'], [-52, 0, -4, 65, 26, 18, 'X', -17, 96, 87, 'X'], [-97, 0, 0, 0, 0, 0, 58, 'X', 18, 0, 14], [0, -19, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -12, 0, -9, 28, 'X', 0], ['X', -26, 'X', 'X', -67, 'X', 'X', -39, -69, -85, 'X']]
2 <class 'list'>: [['X', 'X', 'X', 0, 'X', 0, 0, 0, 66, 'X', 'X'], ['X', 'X', 0, 50, -14, -25, 2, -30, 2, 14, 3], ['X', 10, 'X', -21, 0, -6, 1, 92, -15, 10, 'X'], [-61, 0, 8, 'X', 0, 0, -13, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 94, -60, 29, 68, 61, 20], [-76, 10, 61, -1, -2, 'X', 0, 1, 47, 16, 'X'], [-52, 0, -4, 67, 24, 18, 'X', -17, 96, 87, 'X'], [-97, 0, 0, 0, 0, -2, 58, 'X', 18, 0, 12], [-2, -17, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -12, 0, -9, 28, 'X', 0], ['X', -26, 'X', 'X', -67, 'X', 'X', -41, -69, -85, 'X']]
3 <class 'list'>: [['X', 'X', 'X', 0, 'X', 0, 0, 0, 64, 'X', 'X'], ['X', 'X', 0, 48, -14, -25, 2, -30, 4, 14, 3], ['X', 10, 'X', -23, 0, -4, 1, 92, -15, 10, 'X'], [-61, -2, 6, 'X', 0, 0, -13, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 96, -58, 29, 68, 61, 20], [-76, 10, 63, -1, 0, 'X', 0, 1, 47, 16, 'X'], [-52, 0, -4, 67, 26, 18, 'X', -15, 96, 87, 'X'], [-97, 0, 0, 0, 0, -2, 60, 'X', 18, 0, 10], [-4, -15, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -12, 0, -9, 28, 'X', 0], ['X', -26, 'X', 'X', -67, 'X', 'X', -43, -69, -85, 'X']]
4 <class 'list'>: [['X', 'X', 'X', 0, 'X', 0, 0, 0, 63, 'X', 'X'], ['X', 'X', 0, 47, -14, -25, 2, -30, 5, 14, 3], ['X', 10, 'X', -24, 0, -3, 1, 92, -15, 10, 'X'], [-61, -3, 5, 'X', 0, 0, -13, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 96, -59, 29, 68, 61, 19], [-76, 10, 64, -1, 0, 'X', 0, 0, 47, 16, 'X'], [-52, 0, -4, 67, 25, 18, 'X', -16, 96, 87, 'X'], [-97, 0, 0, 0, 0, -3, 59, 'X', 18, 0, 10], [-5, -14, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -12, 0, -9, 28, 'X', 0], ['X', -26, 'X', 'X', -68, 'X', 'X', -43, -69, -85, 'X']]
5 <class 'list'>: [['X', 'X', 'X', 0, 'X', 0, 0, 0, 60, 'X', 'X'], ['X', 'X', 0, 44, -14, -25, 2, -30, 8, 14, 3], ['X', 10, 'X', -27, 0, 0, 1, 92, -15, 10, 'X'], [-61, -6, 2, 'X', 0, 0, -13, 27, 0, 0, 'X'], ['X', 15, 7, -23, 'X', 96, -62, 29, 68, 61, 16], [-76, 10, 67, -1, 0, 'X', 0, 0, 47, 13, 'X'], [-52, 0, -4, 67, 22, 18, 'X', -19, 96, 87, 'X'], [-97, 0, 0, 0, 0, -3, 56, 'X', 18, 3, 10], [-8, -11, 50, -29, 0, -34, 0, -2, 'X', 0, 0], ['X', -26, -38, 33, 0, -15, 0, -6, 28, 'X', 0], ['X', -26, 'X', 'X', -71, 'X', 'X', -43, -69, -85, 'X']]

'''