import random

N = 50
p = 0.70

M = N - 2
def makeS():
    global M, p
    A = [[-1 for i in range(M)] for j in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j:
                if random.random() < p:
                    if A[i][j] == -1:
                        A[i][j] = int(random.random()*100)
                        A[j][i] = 0
                    else:
                        if random.random() > 0.5:
                            A[i][j] = int(random.random() * 100)
                            A[j][i] = 0
    for i in range(M):
        if any([j > 0 for j in A[i]]):
            # print(i)
            pass
        else:

            l = [j for j in range(M) if A[i][j] == -1]
            l.remove(i)
            random.shuffle(l)

            if i < int(M/2):
                chose = i + 1
            else:
                chose = i - 1
            if l:
                chose = l[0]

            A[i][chose] = int(random.random() * 100)
            A[chose][i] = 0
    for i in range(M):
        if A[i].count(0) == 0:
            l = [j for j in range(M) if A[i][j] == -1]
            l.remove(i)
            random.shuffle(l)
            if i < int(M/2):
                chose = i + 1
            else:
                chose = i - 1
            if l:
                chose = l[0]

            A[i][chose] = 0
            A[chose][i] = int(random.random() * 100)
    return A


def checkS(SS):
    global M
    good = True
    for i in range(M):
        if any([j > 0 for j in A[i]]):
            pass
        else:
            good = False
        if SS[i].count(0) > 0:
            pass
        else:
            good = False
        if SS[i].count(-1) == 0:
            good = False
    return good
A = makeS()
while not checkS(A):
    A = makeS()


a = [i for i in range(M)]
random.shuffle(a)
x = int(M*p)
y = int(M*p/2)

z = random.randint(y, x)
if z < 2:
    z = 2
iii = a[0:z]
iii = sorted(iii)

def makeooo():
    global M, p
    b = [i for i in range(M)]
    random.shuffle(b)
    x = int(M*p)
    y = int(M*p/2)

    zz = random.randint(y, x)
    if zz < 2:
        zz = 2

    return b[0:zz]
ooo = makeooo()
ooo = sorted(ooo)
while iii == ooo:
    ooo = makeooo()
    ooo = sorted(ooo)

outputstring = []
outputstring.append(str(N)+'\n')
s = '1 '+str(len(iii))
for i in iii:
    s += ' '+str(i+2)+' '+str(int(random.random()*100))
s += '\n'
outputstring.append(s)
for i in range(M):
    s = str(i+2)
    n = M - (A[i].count(-1) + A[i].count(0))
    if i in ooo:
        n += 1
    s += ' '+str(n)
    for j in range(M):
        if A[i][j] > 0:
            s += ' '+str(j+2)+' '+str(A[i][j])
    if i in ooo:
        s += ' '+str(N)+' '+str(int(random.random() * 100))
    s += '\n'
    outputstring.append(s)
pass

# firstline = "{} {} \n".format(N,M)
# def lineto(B):
#     sum = 0
#     s = ""
#     for i in range(M):
#         if B[i] == 1:
#             sum += 1
#             s += str(i+1)+' '
#     return "{} ".format(sum)+s
with open('network.inp', 'w') as f:
    f.writelines(outputstring)
    # for i in range(N):
    #     f.write("{} ".format(i+1)+lineto(A[i])+'\n')
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
            f.write('from FF: '+s[0])

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
        print("FF:--- %s seconds ---" % round(time.time() - start_time, 2))

ff = FordFulkerson()
ff.run()

class Edmonds_Karp:
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
                pass
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
            self.visited = []
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
        with open('network.out', 'a') as f:
            f.write('from EK: '+s[0])

    def run(self):
        start_time = time.time()
        while not self.findall:
            self.BFS_shortest_path()
            if self.reach_layer_end:
                self.look_for_path()
            self.visited = []
            self.layers = []
            self.backward = {}
            self.path = []
            self.deadend = []
            self.reach_layer_end = False
        self.min_cut_node_search(0)
        self.summary()
        print("EK:--- %s seconds ---" % round(time.time() - start_time, 2))

ek = Edmonds_Karp()
ek.run()

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
        with open('network.out', 'a') as f:
            f.write('from DD: '+s[0])

    def run(self):
        start_time = time.time()
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
        print("DD: --- %s seconds ---" % round(time.time() - start_time, 2))
d = Dinic()
d.run()
