# import collections.counter as Counter
# import produce_input
import random

N = 9
# M = N - 2
p = 0.90


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
        self.looking_for_augment_path_in_path_list()
        self.min_cut_node_search(0)
        self.summary()
        # self.path = [0, 1, 2, 3, 6]
        # self.update_path()
        pass



import random

# N = 15
M = N - 2
# p = 0.20


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
ff = FordFulkerson()
ff.testrun()

