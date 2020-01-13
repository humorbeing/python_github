class Dinic:
    def __init__(self):
        with open('network.inp', 'r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.residual_matrix = [['X' for _ in range(self.N)] for _ in range(self.N)]
        for i in a:
            b = i.split()
            for j in range(int(b[1])):
                self.residual_matrix[int(b[0])-1][int(b[j*2+2])-1] = int(b[j*2+3])
                self.residual_matrix[int(b[j * 2 + 2]) - 1][int(b[0]) - 1] = 0
        self.visited = []
        self.layers = []
        self.backward = {}
        self.path = []
        self.deadend = []
        self.min_cut = []
        self.findall = False
        self.reach_layer_end = False
        self.update_all = False

    def BFS_shortest_path(self):
        if self.layers:
            tem = []
            n = len(self.layers[-1])
            for i in range(n):
                for j in range(self.N):
                    if self.residual_matrix[self.layers[-1][i]][j] != 'X':
                        if self.residual_matrix[self.layers[-1][i]][j] != 0:
                            if j not in self.visited:
                                if j not in tem:
                                    tem.append(j)
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
        else:
            self.visited.append(0)
            self.layers.append([0])
        if self.reach_layer_end or self.findall:
            self.deadend = []
        else:
            self.BFS_shortest_path()

    def look_for_path(self):
        gotone = False
        if self.path:
            if self.path[-1] in self.backward:
                for i in self.backward[self.path[-1]]:
                    if i not in self.deadend:
                        gotone = True
                        self.path.append(i)
                        break
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
            else:
                self.look_for_path()
        else:
            if self.path:
                self.deadend.append(self.path[-1])
                del self.path[-1]
                self.look_for_path()
            else:
                self.update_all = True

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
                if not self.backward[self.path[i+1]]:
                    del self.backward[self.path[i+1]]

    def min_cut_node_search(self, node):
        self.visited.append(node)
        for i in range(self.N):
            if i not in self.visited:
                if self.residual_matrix[node][i] != 'X':
                    if self.residual_matrix[node][i] != 0:
                        self.min_cut_node_search(i)

    def summary(self):
        s = []
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
        s.append(str(max_flow)+' '+str(len(self.min_cut))+'\n')
        for i in self.min_cut:
            s.append(str(i[0]+1)+' '+str(i[1]+1)+' '+str(self.residual_matrix[i[1]][i[0]] * (-1))+'\n')
        with open('network.out', 'w') as f:
            f.writelines(s)

    def run(self):
        while not self.findall:
            self.BFS_shortest_path()
            if self.reach_layer_end:
                while not self.update_all:
                    self.look_for_path()
            self.update_all = False
            self.visited = []
            self.layers = []
            self.backward = {}
            self.path = []
            self.deadend = []
            self.reach_layer_end = False
        self.min_cut_node_search(0)
        self.summary()

d = Dinic()
d.run()
