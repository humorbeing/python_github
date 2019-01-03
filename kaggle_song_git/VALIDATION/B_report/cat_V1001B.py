import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle
from catboost import CatBoostClassifier


since = time.time()
result = {}
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_top2.csv'

df = read_df(load_name)
show_df(df)

train, val = fake_df(df)
del df

X = train.drop('target', axis=1)
Y = train['target']
vX = val.drop('target', axis=1)
vY = val['target']
cat_feature = np.where(X.dtypes == 'category')[0]
del train, val

model = CatBoostClassifier(
    iterations=260, learning_rate=0.3,
    depth=16, logging_level='Verbose',
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',
    od_wait=100,
)
model.fit(
    X, Y,
    cat_features=cat_feature,
    eval_set=(vX, vY)
)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION/cat_V1001B.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
  return f(*args, **kwds)

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float16
ITC_msno_log10_1        float16
top2_in_song           category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
0: learn: 0.8273337	test: 0.6674815	bestTest: 0.6674815 (0)	total: 1m 26s	remaining: 6h 12m 26s
1: learn: 0.8332374	test: 0.6733523	bestTest: 0.6733523 (1)	total: 2m 50s	remaining: 6h 5m 43s
2: learn: 0.8353196	test: 0.6739257	bestTest: 0.6739257 (2)	total: 4m 1s	remaining: 5h 44m 39s
3: learn: 0.8370516	test: 0.6768654	bestTest: 0.6768654 (3)	total: 5m 27s	remaining: 5h 48m 51s
4: learn: 0.8378079	test: 0.677253	bestTest: 0.677253 (4)	total: 6m 32s	remaining: 5h 33m 53s
5: learn: 0.8391146	test: 0.6783794	bestTest: 0.6783794 (5)	total: 7m 53s	remaining: 5h 33m 59s
6: learn: 0.8401685	test: 0.6795271	bestTest: 0.6795271 (6)	total: 9m 3s	remaining: 5h 27m 25s
7: learn: 0.8406144	test: 0.6801458	bestTest: 0.6801458 (7)	total: 10m 15s	remaining: 5h 23m 6s
8: learn: 0.8418009	test: 0.6811896	bestTest: 0.6811896 (8)	total: 11m 35s	remaining: 5h 23m 26s
9: learn: 0.8427239	test: 0.6820257	bestTest: 0.6820257 (9)	total: 12m 58s	remaining: 5h 24m 26s
10: learn: 0.8436988	test: 0.6841346	bestTest: 0.6841346 (10)	total: 14m 37s	remaining: 5h 31m 6s
11: learn: 0.8444833	test: 0.6852268	bestTest: 0.6852268 (11)	total: 16m 1s	remaining: 5h 31m 10s
12: learn: 0.8448595	test: 0.6862138	bestTest: 0.6862138 (12)	total: 17m 21s	remaining: 5h 29m 45s
13: learn: 0.8450538	test: 0.6868917	bestTest: 0.6868917 (13)	total: 18m 30s	remaining: 5h 25m 20s
14: learn: 0.84513	test: 0.6873191	bestTest: 0.6873191 (14)	total: 18m 59s	remaining: 5h 10m 16s
15: learn: 0.8451297	test: 0.6873363	bestTest: 0.6873363 (15)	total: 19m 5s	remaining: 4h 51m 8s
16: learn: 0.8451295	test: 0.6873483	bestTest: 0.6873483 (16)	total: 19m 10s	remaining: 4h 34m 5s
17: learn: 0.8451294	test: 0.6873566	bestTest: 0.6873566 (17)	total: 19m 15s	remaining: 4h 18m 55s
18: learn: 0.8451293	test: 0.6873624	bestTest: 0.6873624 (18)	total: 19m 20s	remaining: 4h 5m 21s
19: learn: 0.8451292	test: 0.6873664	bestTest: 0.6873664 (19)	total: 19m 25s	remaining: 3h 53m 7s
20: learn: 0.8451292	test: 0.6873692	bestTest: 0.6873692 (20)	total: 19m 30s	remaining: 3h 42m 3s
21: learn: 0.8451291	test: 0.6873712	bestTest: 0.6873712 (21)	total: 19m 35s	remaining: 3h 31m 58s
22: learn: 0.8451291	test: 0.6873726	bestTest: 0.6873726 (22)	total: 19m 40s	remaining: 3h 22m 46s
23: learn: 0.8451291	test: 0.6873735	bestTest: 0.6873735 (23)	total: 19m 45s	remaining: 3h 14m 20s
24: learn: 0.8451288	test: 0.6874305	bestTest: 0.6874305 (24)	total: 19m 51s	remaining: 3h 6m 40s
25: learn: 0.8457075	test: 0.6884917	bestTest: 0.6884917 (25)	total: 21m 19s	remaining: 3h 11m 57s
26: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (26)	total: 22m 59s	remaining: 3h 18m 27s
27: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (27)	total: 23m 4s	remaining: 3h 11m 8s
28: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (28)	total: 23m 9s	remaining: 3h 4m 29s
29: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (29)	total: 23m 13s	remaining: 2h 58m 7s
30: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (30)	total: 23m 18s	remaining: 2h 52m 9s
31: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (31)	total: 23m 22s	remaining: 2h 46m 34s
32: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (32)	total: 23m 27s	remaining: 2h 41m 18s
33: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (33)	total: 23m 31s	remaining: 2h 36m 21s
34: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (34)	total: 23m 35s	remaining: 2h 31m 41s
35: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (35)	total: 23m 40s	remaining: 2h 27m 16s
36: learn: 0.8463994	test: 0.6889989	bestTest: 0.6889989 (36)	total: 23m 44s	remaining: 2h 23m 5s
37: learn: 0.8467623	test: 0.6895744	bestTest: 0.6895744 (37)	total: 25m 18s	remaining: 2h 27m 52s
38: learn: 0.8472803	test: 0.6898084	bestTest: 0.6898084 (38)	total: 26m 40s	remaining: 2h 31m 9s
39: learn: 0.8477648	test: 0.6902333	bestTest: 0.6902333 (39)	total: 28m 12s	remaining: 2h 35m 7s
40: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (40)	total: 29m 32s	remaining: 2h 37m 49s
41: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (41)	total: 29m 37s	remaining: 2h 33m 44s
42: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (42)	total: 29m 41s	remaining: 2h 29m 51s
43: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (43)	total: 29m 47s	remaining: 2h 26m 13s
44: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (44)	total: 29m 52s	remaining: 2h 22m 44s
45: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (45)	total: 29m 56s	remaining: 2h 19m 19s
46: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (46)	total: 30m 1s	remaining: 2h 16m 6s
47: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (47)	total: 30m 6s	remaining: 2h 12m 57s
48: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (48)	total: 30m 11s	remaining: 2h 9m 59s
49: learn: 0.8480763	test: 0.6906336	bestTest: 0.6906336 (49)	total: 30m 15s	remaining: 2h 7m 5s
50: learn: 0.8484246	test: 0.6911672	bestTest: 0.6911672 (50)	total: 31m 54s	remaining: 2h 10m 46s
51: learn: 0.8489462	test: 0.6913323	bestTest: 0.6913323 (51)	total: 33m 25s	remaining: 2h 13m 43s
52: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (52)	total: 34m 41s	remaining: 2h 15m 31s
53: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (53)	total: 34m 46s	remaining: 2h 12m 38s
54: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (54)	total: 34m 50s	remaining: 2h 9m 52s
55: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (55)	total: 34m 56s	remaining: 2h 7m 15s
56: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (56)	total: 35m 1s	remaining: 2h 4m 44s
57: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (57)	total: 35m 5s	remaining: 2h 2m 14s
58: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (58)	total: 35m 11s	remaining: 1h 59m 51s
59: learn: 0.8491961	test: 0.6915922	bestTest: 0.6915922 (59)	total: 35m 16s	remaining: 1h 57m 33s
60: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (60)	total: 36m 30s	remaining: 1h 59m 6s
61: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (61)	total: 36m 34s	remaining: 1h 56m 49s
62: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (62)	total: 36m 40s	remaining: 1h 54m 40s
63: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (63)	total: 36m 44s	remaining: 1h 52m 32s
64: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (64)	total: 36m 49s	remaining: 1h 50m 27s
65: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (65)	total: 36m 55s	remaining: 1h 48m 31s
66: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (66)	total: 36m 59s	remaining: 1h 46m 33s
67: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (67)	total: 37m 4s	remaining: 1h 44m 41s
68: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (68)	total: 37m 9s	remaining: 1h 42m 50s
69: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (69)	total: 37m 13s	remaining: 1h 41m 2s
70: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (70)	total: 37m 17s	remaining: 1h 39m 17s
71: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (71)	total: 37m 22s	remaining: 1h 37m 34s
72: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (72)	total: 37m 26s	remaining: 1h 35m 54s
73: learn: 0.8496593	test: 0.6919127	bestTest: 0.6919127 (73)	total: 37m 30s	remaining: 1h 34m 17s
74: learn: 0.8498004	test: 0.6919629	bestTest: 0.6919629 (74)	total: 38m 46s	remaining: 1h 35m 38s
75: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (75)	total: 38m 51s	remaining: 1h 34m 4s
76: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (76)	total: 38m 56s	remaining: 1h 32m 33s
77: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (77)	total: 39m 1s	remaining: 1h 31m 3s
78: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (78)	total: 39m 5s	remaining: 1h 29m 34s
79: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (79)	total: 39m 10s	remaining: 1h 28m 9s
80: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (80)	total: 39m 15s	remaining: 1h 26m 44s
81: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (81)	total: 39m 20s	remaining: 1h 25m 24s
82: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (82)	total: 39m 25s	remaining: 1h 24m 5s
83: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (83)	total: 39m 30s	remaining: 1h 22m 47s
84: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (84)	total: 39m 35s	remaining: 1h 21m 31s
85: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (85)	total: 39m 41s	remaining: 1h 20m 17s
86: learn: 0.8498005	test: 0.6919629	bestTest: 0.6919629 (86)	total: 39m 45s	remaining: 1h 19m 3s
87: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (87)	total: 41m 4s	remaining: 1h 20m 16s
88: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 10s	remaining: 1h 19m 6s
89: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 20s	remaining: 1h 18m 5s
90: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 24s	remaining: 1h 16m 54s
91: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 29s	remaining: 1h 15m 45s
92: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 33s	remaining: 1h 14m 37s
93: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 38s	remaining: 1h 13m 32s
94: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 43s	remaining: 1h 12m 28s
95: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 48s	remaining: 1h 11m 25s
96: learn: 0.8500418	test: 0.6921084	bestTest: 0.6921084 (88)	total: 41m 53s	remaining: 1h 10m 24s
97: learn: 0.8502191	test: 0.6921372	bestTest: 0.6921372 (97)	total: 43m	remaining: 1h 11m 6s
98: learn: 0.8505581	test: 0.6923644	bestTest: 0.6923644 (98)	total: 44m 9s	remaining: 1h 11m 49s
99: learn: 0.8509137	test: 0.6925948	bestTest: 0.6925948 (99)	total: 45m 42s	remaining: 1h 13m 8s
100: learn: 0.8512995	test: 0.6928493	bestTest: 0.6928493 (100)	total: 47m 20s	remaining: 1h 14m 31s
101: learn: 0.8513855	test: 0.6929912	bestTest: 0.6929912 (101)	total: 48m 37s	remaining: 1h 15m 19s
102: learn: 0.8516764	test: 0.6930568	bestTest: 0.6930568 (102)	total: 49m 58s	remaining: 1h 16m 10s
103: learn: 0.8518016	test: 0.6932093	bestTest: 0.6932093 (103)	total: 51m 18s	remaining: 1h 16m 57s
104: learn: 0.8521449	test: 0.6933708	bestTest: 0.6933708 (104)	total: 52m 55s	remaining: 1h 18m 7s
105: learn: 0.8523282	test: 0.6934194	bestTest: 0.6934194 (105)	total: 54m 18s	remaining: 1h 18m 54s
106: learn: 0.8524328	test: 0.6936029	bestTest: 0.6936029 (106)	total: 55m 42s	remaining: 1h 19m 38s
107: learn: 0.8524328	test: 0.6936029	bestTest: 0.6936029 (107)	total: 55m 47s	remaining: 1h 18m 31s
108: learn: 0.852433	test: 0.6936025	bestTest: 0.6936029 (107)	total: 55m 56s	remaining: 1h 17m 30s
109: learn: 0.852433	test: 0.6936025	bestTest: 0.6936029 (107)	total: 56m 1s	remaining: 1h 16m 23s
110: learn: 0.852433	test: 0.6936025	bestTest: 0.6936029 (107)	total: 56m 7s	remaining: 1h 15m 20s
111: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 14s	remaining: 1h 14m 19s
112: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 20s	remaining: 1h 13m 17s
113: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 25s	remaining: 1h 12m 15s
114: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 30s	remaining: 1h 11m 14s
115: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 34s	remaining: 1h 10m 13s
116: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 38s	remaining: 1h 9m 14s
117: learn: 0.852433	test: 0.6936023	bestTest: 0.6936029 (107)	total: 56m 43s	remaining: 1h 8m 16s
118: learn: 0.8525529	test: 0.6937068	bestTest: 0.6937068 (118)	total: 58m 8s	remaining: 1h 8m 52s
119: learn: 0.8526407	test: 0.6937478	bestTest: 0.6937478 (119)	total: 59m 26s	remaining: 1h 9m 21s
120: learn: 0.8527945	test: 0.6938973	bestTest: 0.6938973 (120)	total: 1h 49s	remaining: 1h 9m 52s
121: learn: 0.85305	test: 0.6943682	bestTest: 0.6943682 (121)	total: 1h 2m 11s	remaining: 1h 10m 21s
122: learn: 0.8530501	test: 0.694368	bestTest: 0.6943682 (121)	total: 1h 2m 22s	remaining: 1h 9m 28s
123: learn: 0.8530501	test: 0.694368	bestTest: 0.6943682 (121)	total: 1h 2m 31s	remaining: 1h 8m 34s
124: learn: 0.8530501	test: 0.6943771	bestTest: 0.6943771 (124)	total: 1h 2m 43s	remaining: 1h 7m 44s
125: learn: 0.8530501	test: 0.6943771	bestTest: 0.6943771 (125)	total: 1h 2m 48s	remaining: 1h 6m 47s
126: learn: 0.8530501	test: 0.694377	bestTest: 0.6943771 (125)	total: 1h 2m 57s	remaining: 1h 5m 55s
127: learn: 0.8530501	test: 0.694377	bestTest: 0.6943771 (125)	total: 1h 3m 5s	remaining: 1h 5m 4s
128: learn: 0.8530501	test: 0.694377	bestTest: 0.6943771 (125)	total: 1h 3m 10s	remaining: 1h 4m 9s
129: learn: 0.8530501	test: 0.694377	bestTest: 0.6943771 (125)	total: 1h 3m 19s	remaining: 1h 3m 19s
130: learn: 0.8530501	test: 0.694377	bestTest: 0.6943771 (125)	total: 1h 3m 28s	remaining: 1h 2m 29s
131: learn: 0.8530886	test: 0.6944221	bestTest: 0.6944221 (131)	total: 1h 4m 20s	remaining: 1h 2m 23s
132: learn: 0.8532652	test: 0.6945488	bestTest: 0.6945488 (132)	total: 1h 5m 40s	remaining: 1h 2m 42s
133: learn: 0.8532654	test: 0.6945496	bestTest: 0.6945496 (133)	total: 1h 5m 54s	remaining: 1h 1m 58s
134: learn: 0.8532654	test: 0.6945496	bestTest: 0.6945496 (134)	total: 1h 5m 58s	remaining: 1h 1m 5s
135: learn: 0.8532654	test: 0.6945496	bestTest: 0.6945496 (135)	total: 1h 6m 3s	remaining: 1h 13s
136: learn: 0.8532655	test: 0.6945496	bestTest: 0.6945496 (135)	total: 1h 6m 10s	remaining: 59m 24s
137: learn: 0.8532655	test: 0.6945496	bestTest: 0.6945496 (135)	total: 1h 6m 15s	remaining: 58m 34s
138: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 22s	remaining: 57m 46s
139: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 29s	remaining: 56m 59s
140: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 35s	remaining: 56m 12s
141: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 41s	remaining: 55m 25s
142: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 49s	remaining: 54m 40s
143: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 6m 56s	remaining: 53m 55s
144: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 7m 3s	remaining: 53m 11s
145: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 7m 10s	remaining: 52m 27s
146: learn: 0.8532655	test: 0.6945495	bestTest: 0.6945496 (135)	total: 1h 7m 18s	remaining: 51m 44s
147: learn: 0.8533759	test: 0.6945586	bestTest: 0.6945586 (147)	total: 1h 8m 29s	remaining: 51m 50s
148: learn: 0.8535669	test: 0.6948177	bestTest: 0.6948177 (148)	total: 1h 9m 52s	remaining: 52m 3s
149: learn: 0.8535669	test: 0.6948177	bestTest: 0.6948177 (149)	total: 1h 9m 58s	remaining: 51m 19s
150: learn: 0.8536821	test: 0.6947662	bestTest: 0.6948177 (149)	total: 1h 11m 7s	remaining: 51m 20s
151: learn: 0.8538172	test: 0.6947839	bestTest: 0.6948177 (149)	total: 1h 12m 20s	remaining: 51m 24s
152: learn: 0.8538473	test: 0.6947738	bestTest: 0.6948177 (149)	total: 1h 13m 26s	remaining: 51m 21s
153: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 14m 51s	remaining: 51m 31s
154: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m	remaining: 50m 49s
155: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 10s	remaining: 50m 6s
156: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 15s	remaining: 49m 22s
157: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 22s	remaining: 48m 39s
158: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 29s	remaining: 47m 57s
159: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 37s	remaining: 47m 15s
160: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 44s	remaining: 46m 34s
161: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 51s	remaining: 45m 53s
162: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 15m 59s	remaining: 45m 13s
163: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 16m 9s	remaining: 44m 34s
164: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 16m 16s	remaining: 43m 54s
165: learn: 0.8540325	test: 0.6948127	bestTest: 0.6948177 (149)	total: 1h 16m 20s	remaining: 43m 14s
166: learn: 0.8540353	test: 0.694801	bestTest: 0.6948177 (149)	total: 1h 17m 20s	remaining: 43m 4s
167: learn: 0.8540774	test: 0.6948421	bestTest: 0.6948421 (167)	total: 1h 18m 29s	remaining: 42m 58s
168: learn: 0.85408	test: 0.6948486	bestTest: 0.6948486 (168)	total: 1h 18m 46s	remaining: 42m 25s
169: learn: 0.8540847	test: 0.6948732	bestTest: 0.6948732 (169)	total: 1h 19m 44s	remaining: 42m 12s
170: learn: 0.8542859	test: 0.6949825	bestTest: 0.6949825 (170)	total: 1h 20m 56s	remaining: 42m 7s
171: learn: 0.8542859	test: 0.6949825	bestTest: 0.6949825 (171)	total: 1h 21m 3s	remaining: 41m 28s
172: learn: 0.8542859	test: 0.6949825	bestTest: 0.6949825 (172)	total: 1h 21m 12s	remaining: 40m 50s
173: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (173)	total: 1h 21m 20s	remaining: 40m 11s
174: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (174)	total: 1h 21m 25s	remaining: 39m 32s
175: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (175)	total: 1h 21m 33s	remaining: 38m 55s
176: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (176)	total: 1h 21m 38s	remaining: 38m 17s
177: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (177)	total: 1h 21m 43s	remaining: 37m 39s
178: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (178)	total: 1h 21m 48s	remaining: 37m 1s
179: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (179)	total: 1h 21m 53s	remaining: 36m 23s
180: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (180)	total: 1h 21m 57s	remaining: 35m 46s
181: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (181)	total: 1h 22m 1s	remaining: 35m 9s
182: learn: 0.8542859	test: 0.6949826	bestTest: 0.6949826 (182)	total: 1h 22m 6s	remaining: 34m 32s
183: learn: 0.8543027	test: 0.6949936	bestTest: 0.6949936 (183)	total: 1h 23m 12s	remaining: 34m 22s
184: learn: 0.8544327	test: 0.6949554	bestTest: 0.6949936 (183)	total: 1h 24m 34s	remaining: 34m 17s
185: learn: 0.8544327	test: 0.6949555	bestTest: 0.6949936 (183)	total: 1h 24m 45s	remaining: 33m 43s
186: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 24m 53s	remaining: 33m 8s
187: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 2s	remaining: 32m 34s
188: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 11s	remaining: 32m
189: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 15s	remaining: 31m 24s
190: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 24s	remaining: 30m 51s
191: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 28s	remaining: 30m 16s
192: learn: 0.8544327	test: 0.6949658	bestTest: 0.6949936 (183)	total: 1h 25m 36s	remaining: 29m 43s
193: learn: 0.8544662	test: 0.6949812	bestTest: 0.6949936 (183)	total: 1h 26m 51s	remaining: 29m 33s
194: learn: 0.854478	test: 0.6950061	bestTest: 0.6950061 (194)	total: 1h 27m 51s	remaining: 29m 17s
195: learn: 0.8547132	test: 0.6949508	bestTest: 0.6950061 (194)	total: 1h 29m 11s	remaining: 29m 7s
196: learn: 0.8549441	test: 0.695066	bestTest: 0.695066 (196)	total: 1h 30m 19s	remaining: 28m 53s
197: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950665 (197)	total: 1h 31m 36s	remaining: 28m 41s
198: learn: 0.8551236	test: 0.6950666	bestTest: 0.6950666 (198)	total: 1h 31m 47s	remaining: 28m 8s
199: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 31m 59s	remaining: 27m 35s
200: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 3s	remaining: 27m 1s
201: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 12s	remaining: 26m 28s
202: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 22s	remaining: 25m 56s
203: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 26s	remaining: 25m 22s
204: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 30s	remaining: 24m 49s
205: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 37s	remaining: 24m 16s
206: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 42s	remaining: 23m 44s
207: learn: 0.8551236	test: 0.6950665	bestTest: 0.6950666 (198)	total: 1h 32m 46s	remaining: 23m 11s
208: learn: 0.8554128	test: 0.695176	bestTest: 0.695176 (208)	total: 1h 34m 24s	remaining: 23m 2s
209: learn: 0.8554933	test: 0.695264	bestTest: 0.695264 (209)	total: 1h 35m 32s	remaining: 22m 44s
210: learn: 0.8556041	test: 0.6952744	bestTest: 0.6952744 (210)	total: 1h 36m 48s	remaining: 22m 28s
211: learn: 0.855669	test: 0.695296	bestTest: 0.695296 (211)	total: 1h 38m 18s	remaining: 22m 15s
212: learn: 0.8556797	test: 0.6953172	bestTest: 0.6953172 (212)	total: 1h 39m 33s	remaining: 21m 58s
213: learn: 0.8558328	test: 0.6954296	bestTest: 0.6954296 (213)	total: 1h 40m 44s	remaining: 21m 39s
214: learn: 0.8560648	test: 0.6954593	bestTest: 0.6954593 (214)	total: 1h 42m 20s	remaining: 21m 25s
215: learn: 0.8560932	test: 0.6954677	bestTest: 0.6954677 (215)	total: 1h 43m 31s	remaining: 21m 5s
216: learn: 0.8561191	test: 0.6954983	bestTest: 0.6954983 (216)	total: 1h 45m	remaining: 20m 48s
217: learn: 0.8561385	test: 0.6957105	bestTest: 0.6957105 (217)	total: 1h 46m 24s	remaining: 20m 30s
218: learn: 0.8561735	test: 0.6956832	bestTest: 0.6957105 (217)	total: 1h 47m 40s	remaining: 20m 9s
219: learn: 0.8562226	test: 0.6957385	bestTest: 0.6957385 (219)	total: 1h 48m 41s	remaining: 19m 45s
220: learn: 0.8562368	test: 0.6957466	bestTest: 0.6957466 (220)	total: 1h 49m 41s	remaining: 19m 21s
221: learn: 0.8563669	test: 0.6957719	bestTest: 0.6957719 (221)	total: 1h 50m 51s	remaining: 18m 58s
222: learn: 0.856463	test: 0.6957267	bestTest: 0.6957719 (221)	total: 1h 52m 9s	remaining: 18m 36s
223: learn: 0.8564894	test: 0.6957371	bestTest: 0.6957719 (221)	total: 1h 53m 13s	remaining: 18m 11s
224: learn: 0.8565518	test: 0.6957762	bestTest: 0.6957762 (224)	total: 1h 54m 30s	remaining: 17m 48s
225: learn: 0.8565722	test: 0.6957732	bestTest: 0.6957762 (224)	total: 1h 55m 49s	remaining: 17m 25s
226: learn: 0.8566112	test: 0.6957247	bestTest: 0.6957762 (224)	total: 1h 56m 59s	remaining: 17m
227: learn: 0.8566353	test: 0.6957429	bestTest: 0.6957762 (224)	total: 1h 58m 19s	remaining: 16m 36s
228: learn: 0.8566758	test: 0.6957756	bestTest: 0.6957762 (224)	total: 1h 59m 28s	remaining: 16m 10s
229: learn: 0.8568307	test: 0.695698	bestTest: 0.6957762 (224)	total: 2h 39s	remaining: 15m 44s
230: learn: 0.8568465	test: 0.695672	bestTest: 0.6957762 (224)	total: 2h 1m 34s	remaining: 15m 15s
231: learn: 0.8568964	test: 0.6957113	bestTest: 0.6957762 (224)	total: 2h 2m 37s	remaining: 14m 47s
232: learn: 0.856901	test: 0.6957074	bestTest: 0.6957762 (224)	total: 2h 3m 42s	remaining: 14m 20s
233: learn: 0.8570756	test: 0.6956651	bestTest: 0.6957762 (224)	total: 2h 4m 50s	remaining: 13m 52s
234: learn: 0.8571566	test: 0.6957819	bestTest: 0.6957819 (234)	total: 2h 6m 25s	remaining: 13m 26s
235: learn: 0.857164	test: 0.695786	bestTest: 0.695786 (235)	total: 2h 7m 26s	remaining: 12m 57s
236: learn: 0.8572122	test: 0.6957863	bestTest: 0.6957863 (236)	total: 2h 8m 33s	remaining: 12m 28s
237: learn: 0.8573232	test: 0.6958079	bestTest: 0.6958079 (237)	total: 2h 9m 53s	remaining: 12m
238: learn: 0.8573232	test: 0.6958079	bestTest: 0.6958079 (238)	total: 2h 9m 58s	remaining: 11m 25s
239: learn: 0.8573414	test: 0.695842	bestTest: 0.695842 (239)	total: 2h 11m 27s	remaining: 10m 57s
240: learn: 0.8573422	test: 0.6957973	bestTest: 0.695842 (239)	total: 2h 12m 25s	remaining: 10m 26s
241: learn: 0.8573517	test: 0.6957997	bestTest: 0.695842 (239)	total: 2h 13m 44s	remaining: 9m 56s
242: learn: 0.8573595	test: 0.695819	bestTest: 0.695842 (239)	total: 2h 14m 36s	remaining: 9m 24s
243: learn: 0.8574308	test: 0.6958458	bestTest: 0.6958458 (243)	total: 2h 15m 40s	remaining: 8m 53s
244: learn: 0.8574364	test: 0.695862	bestTest: 0.695862 (244)	total: 2h 16m 43s	remaining: 8m 22s
245: learn: 0.8576137	test: 0.6958746	bestTest: 0.6958746 (245)	total: 2h 17m 58s	remaining: 7m 51s
246: learn: 0.857858	test: 0.695833	bestTest: 0.6958746 (245)	total: 2h 19m 34s	remaining: 7m 20s
247: learn: 0.8580222	test: 0.6958315	bestTest: 0.6958746 (245)	total: 2h 21m 2s	remaining: 6m 49s
248: learn: 0.8581789	test: 0.6958019	bestTest: 0.6958746 (245)	total: 2h 22m 35s	remaining: 6m 17s
249: learn: 0.8582382	test: 0.6958596	bestTest: 0.6958746 (245)	total: 2h 23m 43s	remaining: 5m 44s
250: learn: 0.8583013	test: 0.695789	bestTest: 0.6958746 (245)	total: 2h 24m 59s	remaining: 5m 11s
251: learn: 0.8583466	test: 0.6957926	bestTest: 0.6958746 (245)	total: 2h 25m 52s	remaining: 4m 37s
252: learn: 0.8583493	test: 0.6957943	bestTest: 0.6958746 (245)	total: 2h 27m 10s	remaining: 4m 4s
253: learn: 0.8583515	test: 0.6957975	bestTest: 0.6958746 (245)	total: 2h 28m 28s	remaining: 3m 30s
254: learn: 0.8583733	test: 0.6958419	bestTest: 0.6958746 (245)	total: 2h 29m 45s	remaining: 2m 56s
255: learn: 0.8585258	test: 0.6958636	bestTest: 0.6958746 (245)	total: 2h 31m 8s	remaining: 2m 21s
256: learn: 0.8586623	test: 0.695843	bestTest: 0.6958746 (245)	total: 2h 32m 34s	remaining: 1m 46s
257: learn: 0.8586708	test: 0.6958113	bestTest: 0.6958746 (245)	total: 2h 33m 37s	remaining: 1m 11s
258: learn: 0.8587783	test: 0.6958421	bestTest: 0.6958746 (245)	total: 2h 34m 46s	remaining: 35.9s
259: learn: 0.8588568	test: 0.6958358	bestTest: 0.6958746 (245)	total: 2h 36m 6s	remaining: 0us

bestTest = 0.6958745936
bestIteration = 245

Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION/cat_V1001B.py", line 42, in <module>
    eval_set=(vX, vY)
  File "/usr/local/lib/python3.5/dist-packages/catboost/core.py", line 1291, in fit
    self._fit(X, y, cat_features, None, sample_weight, None, None, baseline, use_best_model, eval_set, verbose, logging_level, plot)
  File "/usr/local/lib/python3.5/dist-packages/catboost/core.py", line 572, in _fit
    setattr(self, "_feature_importance", self.get_feature_importance(X))
  File "/usr/local/lib/python3.5/dist-packages/catboost/core.py", line 874, in get_feature_importance
    fstr = self._calc_fstr(X, fstr_type, thread_count)
  File "_catboost.pyx", line 941, in _catboost._CatBoostBase._calc_fstr
  File "_catboost.pyx", line 802, in _catboost._CatBoost._calc_fstr
  File "_catboost.pyx", line 804, in _catboost._CatBoost._calc_fstr
_catboost.CatboostError: cxxrt::bad_alloc

Process finished with exit code 1
'''