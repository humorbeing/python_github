# import collections.counter as Counter
class FordFulkerson:
    def __init__(self):
        with open('network.inp', 'r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.graph = [['X' for j in range(self.N)] for i in range(self.N)]
        self.residual_matrix = [['X' for j in range(self.N)] for i in range(self.N)]
        # self.residual_matrix[0][0] = 0
        for i in a:
            b = i.split()
            for j in range(int(b[1])):
                self.graph[int(b[0]) - 1][int(b[j * 2 + 2]) - 1] = int(b[j * 2 + 3])
                self.graph[int(b[j * 2 + 2]) - 1][int(b[0]) - 1] = int(b[j * 2 + 3]) * -1
                self.residual_matrix[int(b[0])-1][int(b[j*2+2])-1] = int(b[j*2+3])
                self.residual_matrix[int(b[j * 2 + 2]) - 1][int(b[0]) - 1] = 0
                pass
        self.path = []
        self.deadend = []
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
                    self.deadend.append(self.path[-1])
                    del self.path[-1]
                    self.looking_for_augment_path_in_path_list()
            else:
                pass
        else:
            if 0 in self.deadend:
                pass
            else:
                self.path.append(0)
                self.looking_for_augment_path_in_path_list()

    def test(self):
        # cnt = Counter()
        # for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
        #     cnt[word] += 1
        # print(cnt)
        self.looking_for_augment_path_in_path_list()
        pass
ff = FordFulkerson()
ff.test()

