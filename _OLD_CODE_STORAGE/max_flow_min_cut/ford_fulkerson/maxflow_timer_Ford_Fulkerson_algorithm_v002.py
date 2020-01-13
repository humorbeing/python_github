import time
class FordFulkerson:
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
        self.path = []
        self.deadend = []
        self.visited = []
        self.min_cut = []
        self.findall = False
        self.reach_layer_end = False

    def looking_for_augment_path_in_path_list(self):
        if self.path:
            if self.path[-1] != self.N - 1:
                got_one = False
                for i in range(self.N):
                    if self.residual_matrix[self.path[-1]][i] != 'X':
                        if self.residual_matrix[self.path[-1]][i] != 0:
                            if i not in self.deadend:
                                if i not in self.path:
                                    self.path.append(i)
                                    got_one = True
                                    break
                if not got_one:
                    self.deadend.append(self.path[-1])
                    del self.path[-1]
            else:
                self.reach_layer_end = True
        else:
            if 0 in self.deadend:
                self.findall = True
            else:
                self.path.append(0)
        if self.reach_layer_end or self.findall:
            self.visited = []
            self.deadend = []
        else:
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
        start_time = time.time()
        while not self.findall:
            self.looking_for_augment_path_in_path_list()
            if self.path:
                self.update_path()
            self.path = []
            self.deadend = []
            self.visited = []
            self.reach_layer_end = False
        self.min_cut_node_search(0)
        self.summary()
        print("--- %s seconds ---" % round(time.time() - start_time, 2))

ff = FordFulkerson()
ff.run()
