# import collections.counter as Counter
import time
class Edmonds_Karp:
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
        self.recursion_n = 0
        self.visited = []
        self.layers = []
        self.backward = {}
        self.path = []
        self.deadend = []
        self.min_cut = []
        self.findall = False
        self.reach_layer_end = False
        self.update_all = False
        pass

    def BFS_shortest_path(self):
        self.recursion_n += 1
        # reach_end = False
        # gotnothing = False
        if self.layers:
            tem = []
            n = len(self.layers[-1])
            for i in range(n):
                for j in range(self.N):
                    if self.residual_matrix[self.layers[-1][i]][j] != 'X':
                        if self.residual_matrix[self.layers[-1][i]][j] != 0:
                            if j in self.visited:
                                pass  # deadend
                            else:
                                if j not in tem:
                                    tem.append(j)
                                # self.visited.append(j)
                                if j in self.backward:
                                    self.backward[j].append(self.layers[-1][i])
                                else:
                                    self.backward[j] = [self.layers[-1][i]]
            if tem:
                self.layers.append(tem)
                for i in tem:
                    self.visited.append(i)
                if self.N - 1 in tem:
                    self.reach_layer_end = True
            else:
                self.findall = True
            pass
        else:
            self.visited.append(0)
            self.layers.append([0])
            pass
        if self.reach_layer_end or self.findall:
            self.visited = []
            # self.look_for_path()
            self.deadend = []
            # self.backward.clear()
            # self.layers = []
            # self.BFS_shortest_path()
            # print('end')
            pass
        else:
            if self.recursion_n < 950:

                self.BFS_shortest_path()
            else:
                pass
        pass

    def look_for_path(self):
        self.recursion_n += 1
        gotone = False
        if self.path:
            if self.path[-1] in self.backward:
                for i in self.backward[self.path[-1]]:
                    if i not in self.deadend:
                        gotone = True
                        self.path.append(i)
                        break
            else:
                pass
                # self.deadend.append(self.path[-1])
                # del self.path[-1]
        else:
            if self.N - 1 in self.backward:
                if self.N - 1 not in self.deadend:
                    gotone = True
                    self.path.append(self.N - 1)
        if gotone:
            if self.path[-1] == 0:
                self.path.reverse()
                self.update_path()
                self.path = []
                # self.look_for_path()
                # self.backward.clear()
                pass
            else:
                self.look_for_path()
        else:
            if self.path:
                self.deadend.append(self.path[-1])
                del self.path[-1]
                self.look_for_path()
            else:
                self.update_all = True
                pass

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
            if self.residual_matrix[self.path[i]][self.path[i + 1]] == 0:
                self.backward[self.path[i+1]].remove(self.path[i])
                if self.backward[self.path[i+1]]:
                    pass
                else:
                    del self.backward[self.path[i+1]]
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
        # self.looking_for_augment_path_in_path_list()
        # self.min_cut_node_search(0)
        # self.summary()
        # self.path = [0, 1, 2, 3, 6]
        # self.update_path()
        start_time = time.time()
        while not self.findall:
            self.BFS_shortest_path()
            self.recursion_n = 0
            while not self.update_all:
                self.look_for_path()
                pass
            self.update_all = False
            self.recursion_n = 0
            self.visited = []
            self.layers = []
            self.backward = {}
            self.path = []
            self.deadend = []
            # self.min_cut = []
            # self.findall = False
            self.reach_layer_end = False
            # self.recursion_n = 0
        self.min_cut_node_search(0)
        self.summary()
        print("--- %s seconds ---" % round(time.time() - start_time, 2))

        pass
ek = Edmonds_Karp()
ek.testrun()
