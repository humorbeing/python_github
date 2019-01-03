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

                else:
                    pass
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
        pass

    def min_cut_node_search(self, node):
        self.visited.append(node)
        for i in range(self.N):
            if i not in self.visited:
                if self.residual_matrix[node][i] != 'X':
                    if self.residual_matrix[node][i] != 0:
                        if self.residual_matrix[i][node] != 0:
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

    def testrun(self):
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
        # self.path = [0, 1, 2, 3, 6]
        # self.update_path()
        pass
ff = FordFulkerson()
ff.testrun()

'''/usr/bin/python3.5 /media/ray/PNU@myPC@DDDDD/workspace/python/max_flow_min_cut/ford_fulkerson/dbtest.py
11037 220
17 1000 74
5 1000 56
6 1000 79
14 1000 50
12 1000 76
21 1000 80
26 1000 66
30 1000 9
32 1000 16
33 1000 63
38 1000 51
37 1000 66
39 1000 52
132 1000 80
48 1000 53
63 1000 83
67 1000 71
65 1000 50
50 1000 79
84 1000 10
75 1000 45
82 1000 88
109 1000 16
89 1000 19
93 1000 38
97 1000 11
96 1000 6
112 1000 58
105 1000 75
114 1000 69
120 1000 38
133 1000 69
117 1000 61
119 1000 33
126 1000 17
124 1000 89
172 1000 42
140 1000 57
142 1000 43
150 1000 48
173 1000 70
163 1000 29
151 1000 36
152 1000 75
175 1000 27
174 1000 76
180 1000 16
179 1000 36
218 1000 14
197 1000 33
198 1000 11
201 1000 72
204 1000 80
213 1000 60
219 1000 48
217 1000 93
232 1000 51
235 1000 5
230 1000 99
229 1000 62
236 1000 83
241 1000 6
242 1000 62
243 1000 94
250 1000 15
261 1000 73
281 1000 47
280 1000 11
307 1000 7
286 1000 26
285 1000 10
278 1000 71
290 1000 83
288 1000 18
291 1000 9
306 1000 63
313 1000 31
317 1000 23
328 1000 40
349 1000 2
333 1000 38
351 1000 51
334 1000 9
348 1000 98
338 1000 18
364 1000 55
356 1000 71
357 1000 12
370 1000 60
371 1000 95
369 1000 63
375 1000 12
377 1000 12
381 1000 85
390 1000 73
392 1000 58
393 1000 60
396 1000 85
389 1000 77
406 1000 40
397 1000 35
421 1000 50
412 1000 63
400 1000 28
417 1000 33
426 1000 22
425 1000 33
429 1000 89
435 1000 37
443 1000 88
446 1000 56
448 1000 67
450 1000 23
460 1000 20
463 1000 47
471 1000 89
504 1000 86
491 1000 7
518 1000 70
481 1000 5
492 1000 5
500 1000 45
501 1000 81
507 1000 82
515 1000 18
511 1000 87
520 1000 81
523 1000 60
525 1000 40
526 1000 41
537 1000 85
534 1000 57
542 1000 65
546 1000 84
551 1000 81
554 1000 17
552 1000 40
550 1000 96
590 1000 84
572 1000 64
582 1000 19
574 1000 82
600 1000 36
579 1000 82
589 1000 67
586 1000 23
594 1000 60
601 1000 94
604 1000 26
606 1000 3
617 1000 1
619 1000 34
636 1000 15
637 1000 79
642 1000 17
649 1000 6
651 1000 83
652 1000 99
646 1000 48
665 1000 85
663 1000 21
678 1000 21
689 1000 63
687 1000 19
688 1000 67
695 1000 18
701 1000 65
705 1000 6
711 1000 46
721 1000 14
720 1000 44
731 1000 51
730 1000 81
733 1000 53
735 1000 97
740 1000 99
739 1000 98
744 1000 5
751 1000 99
755 1000 69
752 1000 7
761 1000 27
765 1000 8
759 1000 64
775 1000 83
780 1000 30
782 1000 44
788 1000 56
796 1000 38
813 1000 18
816 1000 76
819 1000 16
814 1000 8
826 1000 73
839 1000 4
828 1000 43
843 1000 7
845 1000 50
847 1000 39
858 1000 77
859 1000 5
862 1000 23
877 1000 50
881 1000 86
911 1000 94
918 1000 37
904 1000 83
913 1000 90
923 1000 80
920 1000 94
932 1000 66
935 1000 61
956 1000 68
952 1000 31
975 1000 89
977 1000 72
984 1000 5
983 1000 47
994 1000 53
988 1000 54
--- 18770.92 seconds ---

Process finished with exit code 0
'''