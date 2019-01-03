import sys
sys.path.insert(0, '../')
from me import *
from real_cat_top2 import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier



since = time.time()
print()
print('This is [no drill] training.')
print()
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'final_train_real.csv'
train = read_df(load_name)
show_df(train)
load_name = 'final_test_real.csv'
test = read_df(load_name)
show_df(test)


K = 3
dfs = divide_df(train, K)
del train
dfs_collector = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dfs_collector.append(dc)

test_collector = pd.DataFrame()
test_collector['id'] = test['id']


# !!!!!!!!!!!!!!!!!!!!!!!!!

dfs_collector, test_collector, r = CatC_top2_1(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = CatR_top2_1(
    K, dfs, dfs_collector, test, test_collector
)



#-----------------------------

dfs_collector, test_collector, r = CatC_top2_2(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = CatR_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
#


# !!!!!!!!!!!!!!!!!!!!!!!!!

print(test_collector.head())
print(test_collector.tail())
save_name = 'Cat'
save_here = '../saves/feature/level1/'
for i in range(K):
    save_train = save_here + 'train' + str(i+1) + '/'
    save_df(dfs_collector[i], name=save_name,
            save_to=save_train)

save_df(test_collector, name=save_name,
            save_to=save_here+'test/')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/home/vblab/untitled/bin/python /home/vblab/workplace/python/kagglebigdata/Final_LVL1_MAKER/C_top2.py
/usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.6
  return f(*args, **kwds)

This is [no drill] training.


>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                             category
song_id                          category
source_system_tab                category
source_screen_name               category
source_type                      category
target                              uint8
artist_name                      category
language                         category
song_year                        category
top2_in_song                     category
top3_in_song                     category
membership_days                     int64
ISC_song_year                       int64
ISC_song_country_ln               float64
ITC_msno_log10_1                  float32
ITC_song_id_log10_1               float32
ITC_source_system_tab_log10_1     float32
OinC_language                     float32
dtype: object
number of rows: 7377418
number of columns: 18

'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'artist_name',
'language',
'song_year',
'top2_in_song',
'top3_in_song',
'membership_days',
'ISC_song_year',
'ISC_song_country_ln',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'OinC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                             category
song_id                          category
source_system_tab                category
source_screen_name               category
source_type                      category
id                               category
artist_name                      category
language                         category
song_year                        category
top2_in_song                     category
top3_in_song                     category
membership_days                     int64
ISC_song_year                       int64
ISC_song_country_ln               float64
ITC_msno_log10_1                  float32
ITC_song_id_log10_1               float32
ITC_source_system_tab_log10_1     float32
OinC_language                     float32
dtype: object
number of rows: 2556790
number of columns: 18

'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'id',
'artist_name',
'language',
'song_year',
'top2_in_song',
'top3_in_song',
'membership_days',
'ISC_song_year',
'ISC_song_country_ln',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'OinC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: CatC_top2_1  k-fold: 1 / 3

0: learn: 0.7794465	total: 10.9s	remaining: 27m 10s
1: learn: 0.7943124	total: 19.6s	remaining: 24m 9s
2: learn: 0.8000242	total: 28s	remaining: 22m 52s
3: learn: 0.8031233	total: 37.4s	remaining: 22m 43s
4: learn: 0.8056411	total: 45.7s	remaining: 22m 5s
5: learn: 0.8071511	total: 55.5s	remaining: 22m 11s
6: learn: 0.8086366	total: 1m 6s	remaining: 22m 45s
7: learn: 0.8103456	total: 1m 16s	remaining: 22m 46s
8: learn: 0.8110976	total: 1m 26s	remaining: 22m 35s
9: learn: 0.8125434	total: 1m 37s	remaining: 22m 42s
10: learn: 0.8132661	total: 1m 47s	remaining: 22m 43s
11: learn: 0.8135831	total: 1m 57s	remaining: 22m 30s
12: learn: 0.8135831	total: 2m	remaining: 21m 7s
13: learn: 0.8135831	total: 2m 3s	remaining: 19m 55s
14: learn: 0.8139379	total: 2m 11s	remaining: 19m 43s
15: learn: 0.8139379	total: 2m 14s	remaining: 18m 45s
16: learn: 0.8139379	total: 2m 17s	remaining: 17m 52s
17: learn: 0.8139379	total: 2m 19s	remaining: 17m 5s
18: learn: 0.8139379	total: 2m 22s	remaining: 16m 23s
19: learn: 0.8139379	total: 2m 25s	remaining: 15m 45s
20: learn: 0.8139379	total: 2m 28s	remaining: 15m 10s
21: learn: 0.8139379	total: 2m 30s	remaining: 14m 38s
22: learn: 0.8139379	total: 2m 33s	remaining: 14m 9s
23: learn: 0.8139379	total: 2m 36s	remaining: 13m 41s
24: learn: 0.8139379	total: 2m 39s	remaining: 13m 16s
25: learn: 0.8139379	total: 2m 42s	remaining: 12m 53s
26: learn: 0.8139379	total: 2m 45s	remaining: 12m 32s
27: learn: 0.8139379	total: 2m 47s	remaining: 12m 11s
28: learn: 0.8146751	total: 2m 58s	remaining: 12m 25s
29: learn: 0.8149466	total: 3m 8s	remaining: 12m 35s
30: learn: 0.8153544	total: 3m 19s	remaining: 12m 46s
31: learn: 0.8153544	total: 3m 22s	remaining: 12m 26s
32: learn: 0.8153544	total: 3m 25s	remaining: 12m 7s
33: learn: 0.8153544	total: 3m 27s	remaining: 11m 49s
34: learn: 0.8153544	total: 3m 30s	remaining: 11m 32s
35: learn: 0.8153544	total: 3m 33s	remaining: 11m 16s
36: learn: 0.8159811	total: 3m 43s	remaining: 11m 23s
37: learn: 0.8159811	total: 3m 46s	remaining: 11m 8s
38: learn: 0.8164412	total: 3m 57s	remaining: 11m 15s
39: learn: 0.8164412	total: 4m	remaining: 11m
40: learn: 0.8164412	total: 4m 3s	remaining: 10m 46s
41: learn: 0.8164412	total: 4m 5s	remaining: 10m 32s
42: learn: 0.8164412	total: 4m 8s	remaining: 10m 18s
43: learn: 0.8164412	total: 4m 11s	remaining: 10m 5s
44: learn: 0.8164412	total: 4m 14s	remaining: 9m 53s
45: learn: 0.8164412	total: 4m 16s	remaining: 9m 41s
46: learn: 0.8164412	total: 4m 19s	remaining: 9m 29s
47: learn: 0.8164412	total: 4m 22s	remaining: 9m 18s
48: learn: 0.8164412	total: 4m 25s	remaining: 9m 7s
49: learn: 0.8164412	total: 4m 28s	remaining: 8m 56s
50: learn: 0.8167454	total: 4m 38s	remaining: 9m
51: learn: 0.8169548	total: 4m 48s	remaining: 9m 2s
52: learn: 0.8170606	total: 4m 56s	remaining: 9m 2s
53: learn: 0.8170606	total: 4m 59s	remaining: 8m 52s
54: learn: 0.8170606	total: 5m 2s	remaining: 8m 41s
55: learn: 0.8170606	total: 5m 4s	remaining: 8m 31s
56: learn: 0.8170606	total: 5m 7s	remaining: 8m 21s
57: learn: 0.8170606	total: 5m 10s	remaining: 8m 12s
58: learn: 0.8170606	total: 5m 13s	remaining: 8m 2s
59: learn: 0.8170606	total: 5m 15s	remaining: 7m 53s
60: learn: 0.8170606	total: 5m 18s	remaining: 7m 45s
61: learn: 0.8175213	total: 5m 29s	remaining: 7m 47s
62: learn: 0.8175214	total: 5m 34s	remaining: 7m 41s
63: learn: 0.8175214	total: 5m 36s	remaining: 7m 32s
64: learn: 0.8175214	total: 5m 39s	remaining: 7m 24s
65: learn: 0.8175214	total: 5m 42s	remaining: 7m 15s
66: learn: 0.8175214	total: 5m 45s	remaining: 7m 7s
67: learn: 0.8175214	total: 5m 48s	remaining: 7m
68: learn: 0.8175214	total: 5m 51s	remaining: 6m 52s
69: learn: 0.8175214	total: 5m 54s	remaining: 6m 44s
70: learn: 0.8175214	total: 5m 56s	remaining: 6m 37s
71: learn: 0.8175214	total: 5m 59s	remaining: 6m 29s
72: learn: 0.8175214	total: 6m 2s	remaining: 6m 22s
73: learn: 0.8179775	total: 6m 12s	remaining: 6m 22s
74: learn: 0.8179775	total: 6m 17s	remaining: 6m 17s
75: learn: 0.8179775	total: 6m 20s	remaining: 6m 10s
76: learn: 0.8179775	total: 6m 23s	remaining: 6m 3s
77: learn: 0.8179775	total: 6m 27s	remaining: 5m 57s
78: learn: 0.8179775	total: 6m 30s	remaining: 5m 50s
79: learn: 0.8179777	total: 6m 35s	remaining: 5m 46s
80: learn: 0.8179777	total: 6m 38s	remaining: 5m 39s
81: learn: 0.8179777	total: 6m 41s	remaining: 5m 32s
82: learn: 0.8179777	total: 6m 44s	remaining: 5m 26s
83: learn: 0.8179777	total: 6m 48s	remaining: 5m 20s
84: learn: 0.8179777	total: 6m 50s	remaining: 5m 14s
85: learn: 0.8180636	total: 7m	remaining: 5m 13s
86: learn: 0.8185016	total: 7m 11s	remaining: 5m 12s
87: learn: 0.818927	total: 7m 21s	remaining: 5m 10s
88: learn: 0.8192768	total: 7m 32s	remaining: 5m 9s
89: learn: 0.8196634	total: 7m 41s	remaining: 5m 7s
90: learn: 0.8199	total: 7m 51s	remaining: 5m 5s
91: learn: 0.8200842	total: 8m 1s	remaining: 5m 3s
92: learn: 0.8203653	total: 8m 12s	remaining: 5m 1s
93: learn: 0.8205223	total: 8m 22s	remaining: 4m 59s
94: learn: 0.8207407	total: 8m 32s	remaining: 4m 56s
95: learn: 0.8209887	total: 8m 43s	remaining: 4m 54s
96: learn: 0.8210929	total: 8m 52s	remaining: 4m 50s
97: learn: 0.8212181	total: 9m 3s	remaining: 4m 48s
98: learn: 0.8212705	total: 9m 12s	remaining: 4m 44s
99: learn: 0.8214445	total: 9m 22s	remaining: 4m 41s
100: learn: 0.8216228	total: 9m 31s	remaining: 4m 37s
101: learn: 0.8217026	total: 9m 40s	remaining: 4m 33s
102: learn: 0.8218203	total: 9m 51s	remaining: 4m 29s
103: learn: 0.8219688	total: 10m	remaining: 4m 25s
104: learn: 0.8222276	total: 10m 11s	remaining: 4m 22s
105: learn: 0.8223531	total: 10m 20s	remaining: 4m 17s
106: learn: 0.8223531	total: 10m 22s	remaining: 4m 10s
107: learn: 0.8223532	total: 10m 26s	remaining: 4m 3s
108: learn: 0.8223533	total: 10m 29s	remaining: 3m 56s
109: learn: 0.8223533	total: 10m 32s	remaining: 3m 50s
110: learn: 0.8223533	total: 10m 35s	remaining: 3m 43s
111: learn: 0.8223534	total: 10m 40s	remaining: 3m 37s
112: learn: 0.8223534	total: 10m 42s	remaining: 3m 30s
113: learn: 0.8223534	total: 10m 46s	remaining: 3m 24s
114: learn: 0.8223534	total: 10m 49s	remaining: 3m 17s
115: learn: 0.8223535	total: 10m 53s	remaining: 3m 11s
116: learn: 0.8223948	total: 11m 3s	remaining: 3m 7s
117: learn: 0.8223948	total: 11m 5s	remaining: 3m
118: learn: 0.8223948	total: 11m 8s	remaining: 2m 54s
119: learn: 0.8225265	total: 11m 19s	remaining: 2m 49s
120: learn: 0.8225983	total: 11m 28s	remaining: 2m 44s
121: learn: 0.8226299	total: 11m 38s	remaining: 2m 40s
122: learn: 0.8228323	total: 11m 48s	remaining: 2m 35s
123: learn: 0.8228323	total: 11m 51s	remaining: 2m 29s
124: learn: 0.8228323	total: 11m 54s	remaining: 2m 22s
125: learn: 0.8228323	total: 11m 57s	remaining: 2m 16s
126: learn: 0.8228324	total: 12m 2s	remaining: 2m 10s
127: learn: 0.8228974	total: 12m 11s	remaining: 2m 5s
128: learn: 0.8230272	total: 12m 21s	remaining: 2m
129: learn: 0.8231212	total: 12m 30s	remaining: 1m 55s
130: learn: 0.8232011	total: 12m 42s	remaining: 1m 50s
131: learn: 0.8232922	total: 12m 51s	remaining: 1m 45s
132: learn: 0.8234803	total: 13m 1s	remaining: 1m 39s
133: learn: 0.823578	total: 13m 10s	remaining: 1m 34s
134: learn: 0.8238034	total: 13m 21s	remaining: 1m 29s
135: learn: 0.8239565	total: 13m 32s	remaining: 1m 23s
136: learn: 0.8240893	total: 13m 41s	remaining: 1m 17s
137: learn: 0.8241652	total: 13m 51s	remaining: 1m 12s
138: learn: 0.8242488	total: 14m 1s	remaining: 1m 6s
139: learn: 0.8243459	total: 14m 10s	remaining: 1m
140: learn: 0.8243874	total: 14m 20s	remaining: 54.9s
141: learn: 0.824408	total: 14m 29s	remaining: 49s
142: learn: 0.8244191	total: 14m 38s	remaining: 43s
143: learn: 0.8244593	total: 14m 48s	remaining: 37s
144: learn: 0.8245618	total: 14m 57s	remaining: 30.9s
145: learn: 0.8246764	total: 15m 9s	remaining: 24.9s
146: learn: 0.8247053	total: 15m 18s	remaining: 18.7s
147: learn: 0.8248101	total: 15m 28s	remaining: 12.5s
148: learn: 0.8248519	total: 15m 38s	remaining: 6.3s
149: learn: 0.8249015	total: 15m 49s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1
0        1     0.473860
3        1     0.510477
6        1     0.755119
9        1     0.917777
12       0     0.785798
# # # # # # # # # # 
0.406444439508
0.538442862135
0.101725357987
0.0956202283953
0.0908211148766
# # # # # # # # # # 

in model: CatC_top2_1  k-fold: 2 / 3

0: learn: 0.7800181	total: 11s	remaining: 27m 19s
1: learn: 0.7949452	total: 20.2s	remaining: 24m 57s
2: learn: 0.7997702	total: 30.7s	remaining: 25m 2s
3: learn: 0.8028883	total: 39.4s	remaining: 23m 57s
4: learn: 0.8055134	total: 48.2s	remaining: 23m 16s
5: learn: 0.8074224	total: 58.8s	remaining: 23m 31s
6: learn: 0.8086234	total: 1m 8s	remaining: 23m 13s
7: learn: 0.8097511	total: 1m 20s	remaining: 23m 44s
8: learn: 0.8110702	total: 1m 30s	remaining: 23m 44s
9: learn: 0.811581	total: 1m 41s	remaining: 23m 38s
10: learn: 0.811581	total: 1m 44s	remaining: 21m 56s
11: learn: 0.811581	total: 1m 47s	remaining: 20m 31s
12: learn: 0.8124332	total: 1m 57s	remaining: 20m 36s
13: learn: 0.813223	total: 2m 9s	remaining: 20m 55s
14: learn: 0.813223	total: 2m 12s	remaining: 19m 48s
15: learn: 0.813223	total: 2m 14s	remaining: 18m 49s
16: learn: 0.813223	total: 2m 17s	remaining: 17m 57s
17: learn: 0.813223	total: 2m 20s	remaining: 17m 10s
18: learn: 0.813223	total: 2m 23s	remaining: 16m 27s
19: learn: 0.813223	total: 2m 26s	remaining: 15m 50s
20: learn: 0.813223	total: 2m 29s	remaining: 15m 15s
21: learn: 0.813223	total: 2m 31s	remaining: 14m 43s
22: learn: 0.813223	total: 2m 34s	remaining: 14m 14s
23: learn: 0.813223	total: 2m 37s	remaining: 13m 46s
24: learn: 0.813223	total: 2m 40s	remaining: 13m 21s
25: learn: 0.813223	total: 2m 43s	remaining: 12m 57s
26: learn: 0.813223	total: 2m 45s	remaining: 12m 35s
27: learn: 0.813223	total: 2m 48s	remaining: 12m 15s
28: learn: 0.8135665	total: 2m 56s	remaining: 12m 15s
29: learn: 0.8142593	total: 3m 6s	remaining: 12m 24s
30: learn: 0.8145416	total: 3m 16s	remaining: 12m 35s
31: learn: 0.8152213	total: 3m 27s	remaining: 12m 43s
32: learn: 0.8159483	total: 3m 36s	remaining: 12m 48s
33: learn: 0.8162811	total: 3m 45s	remaining: 12m 49s
34: learn: 0.8166825	total: 3m 56s	remaining: 12m 58s
35: learn: 0.8166825	total: 3m 59s	remaining: 12m 38s
36: learn: 0.8166825	total: 4m 2s	remaining: 12m 20s
37: learn: 0.8166825	total: 4m 5s	remaining: 12m 2s
38: learn: 0.8166825	total: 4m 7s	remaining: 11m 45s
39: learn: 0.8166825	total: 4m 10s	remaining: 11m 29s
40: learn: 0.8166825	total: 4m 13s	remaining: 11m 14s
41: learn: 0.8166825	total: 4m 16s	remaining: 10m 59s
42: learn: 0.8166825	total: 4m 19s	remaining: 10m 44s
43: learn: 0.8166825	total: 4m 21s	remaining: 10m 30s
44: learn: 0.8166825	total: 4m 24s	remaining: 10m 17s
45: learn: 0.8166825	total: 4m 27s	remaining: 10m 4s
46: learn: 0.8166825	total: 4m 31s	remaining: 9m 54s
47: learn: 0.8166825	total: 4m 34s	remaining: 9m 43s
48: learn: 0.817163	total: 4m 44s	remaining: 9m 46s
49: learn: 0.817163	total: 4m 47s	remaining: 9m 34s
50: learn: 0.817163	total: 4m 50s	remaining: 9m 23s
51: learn: 0.817163	total: 4m 52s	remaining: 9m 12s
52: learn: 0.817163	total: 4m 55s	remaining: 9m 1s
53: learn: 0.817163	total: 4m 58s	remaining: 8m 50s
54: learn: 0.817163	total: 5m 1s	remaining: 8m 40s
55: learn: 0.817163	total: 5m 4s	remaining: 8m 30s
56: learn: 0.817163	total: 5m 7s	remaining: 8m 20s
57: learn: 0.817163	total: 5m 9s	remaining: 8m 11s
58: learn: 0.817163	total: 5m 12s	remaining: 8m 2s
59: learn: 0.817163	total: 5m 15s	remaining: 7m 53s
60: learn: 0.817381	total: 5m 24s	remaining: 7m 53s
61: learn: 0.8174893	total: 5m 33s	remaining: 7m 53s
62: learn: 0.8174893	total: 5m 36s	remaining: 7m 44s
63: learn: 0.8174893	total: 5m 39s	remaining: 7m 35s
64: learn: 0.8174893	total: 5m 41s	remaining: 7m 27s
65: learn: 0.8174893	total: 5m 44s	remaining: 7m 18s
66: learn: 0.8174893	total: 5m 47s	remaining: 7m 10s
67: learn: 0.8174893	total: 5m 50s	remaining: 7m 2s
68: learn: 0.8175997	total: 5m 57s	remaining: 6m 59s
69: learn: 0.8179858	total: 6m 8s	remaining: 7m
70: learn: 0.8179858	total: 6m 11s	remaining: 6m 52s
71: learn: 0.8179858	total: 6m 13s	remaining: 6m 45s
72: learn: 0.8179858	total: 6m 16s	remaining: 6m 37s
73: learn: 0.8179858	total: 6m 19s	remaining: 6m 29s
74: learn: 0.8179858	total: 6m 22s	remaining: 6m 22s
75: learn: 0.8179858	total: 6m 25s	remaining: 6m 15s
76: learn: 0.8179858	total: 6m 27s	remaining: 6m 7s
77: learn: 0.8179858	total: 6m 30s	remaining: 6m
78: learn: 0.8179858	total: 6m 33s	remaining: 5m 53s
79: learn: 0.8179858	total: 6m 36s	remaining: 5m 46s
80: learn: 0.8182708	total: 6m 45s	remaining: 5m 45s
81: learn: 0.8184241	total: 6m 55s	remaining: 5m 44s
82: learn: 0.818578	total: 7m 4s	remaining: 5m 42s
83: learn: 0.8186476	total: 7m 13s	remaining: 5m 40s
84: learn: 0.8189598	total: 7m 23s	remaining: 5m 39s
85: learn: 0.8189598	total: 7m 26s	remaining: 5m 32s
86: learn: 0.8189598	total: 7m 29s	remaining: 5m 25s
87: learn: 0.8189598	total: 7m 32s	remaining: 5m 18s
88: learn: 0.8189598	total: 7m 35s	remaining: 5m 12s
89: learn: 0.8189598	total: 7m 38s	remaining: 5m 5s
90: learn: 0.8189598	total: 7m 41s	remaining: 4m 58s
91: learn: 0.8194775	total: 7m 51s	remaining: 4m 57s
92: learn: 0.8197725	total: 8m	remaining: 4m 54s
93: learn: 0.8200955	total: 8m 11s	remaining: 4m 52s
94: learn: 0.8203347	total: 8m 20s	remaining: 4m 49s
95: learn: 0.820524	total: 8m 30s	remaining: 4m 47s
96: learn: 0.8208174	total: 8m 41s	remaining: 4m 44s
97: learn: 0.8210089	total: 8m 50s	remaining: 4m 41s
98: learn: 0.821155	total: 8m 59s	remaining: 4m 37s
99: learn: 0.8213334	total: 9m 8s	remaining: 4m 34s
100: learn: 0.8214702	total: 9m 17s	remaining: 4m 30s
101: learn: 0.8216797	total: 9m 26s	remaining: 4m 26s
102: learn: 0.8218208	total: 9m 36s	remaining: 4m 22s
103: learn: 0.8219824	total: 9m 45s	remaining: 4m 18s
104: learn: 0.8222426	total: 9m 56s	remaining: 4m 15s
105: learn: 0.8223247	total: 10m 6s	remaining: 4m 11s
106: learn: 0.8223914	total: 10m 15s	remaining: 4m 7s
107: learn: 0.8226059	total: 10m 25s	remaining: 4m 3s
108: learn: 0.8226996	total: 10m 33s	remaining: 3m 58s
109: learn: 0.8226996	total: 10m 36s	remaining: 3m 51s
110: learn: 0.8226996	total: 10m 38s	remaining: 3m 44s
111: learn: 0.8227689	total: 10m 48s	remaining: 3m 40s
112: learn: 0.8227691	total: 10m 52s	remaining: 3m 33s
113: learn: 0.8227692	total: 10m 55s	remaining: 3m 27s
114: learn: 0.8227692	total: 10m 58s	remaining: 3m 20s
115: learn: 0.8228198	total: 11m 8s	remaining: 3m 15s
116: learn: 0.8228198	total: 11m 10s	remaining: 3m 9s
117: learn: 0.8228198	total: 11m 13s	remaining: 3m 2s
118: learn: 0.8228198	total: 11m 16s	remaining: 2m 56s
119: learn: 0.8228198	total: 11m 19s	remaining: 2m 49s
120: learn: 0.8228198	total: 11m 21s	remaining: 2m 43s
121: learn: 0.8228198	total: 11m 24s	remaining: 2m 37s
122: learn: 0.8228198	total: 11m 27s	remaining: 2m 30s
123: learn: 0.8228198	total: 11m 30s	remaining: 2m 24s
124: learn: 0.8228198	total: 11m 32s	remaining: 2m 18s
125: learn: 0.8228198	total: 11m 35s	remaining: 2m 12s
126: learn: 0.8228897	total: 11m 46s	remaining: 2m 8s
127: learn: 0.8228897	total: 11m 49s	remaining: 2m 1s
128: learn: 0.8228897	total: 11m 52s	remaining: 1m 55s
129: learn: 0.8228897	total: 11m 55s	remaining: 1m 50s
130: learn: 0.8228897	total: 11m 58s	remaining: 1m 44s
131: learn: 0.8228897	total: 12m 1s	remaining: 1m 38s
132: learn: 0.8229361	total: 12m 9s	remaining: 1m 33s
133: learn: 0.8229361	total: 12m 12s	remaining: 1m 27s
134: learn: 0.8229361	total: 12m 15s	remaining: 1m 21s
135: learn: 0.8229361	total: 12m 18s	remaining: 1m 16s
136: learn: 0.8229361	total: 12m 21s	remaining: 1m 10s
137: learn: 0.8229361	total: 12m 23s	remaining: 1m 4s
138: learn: 0.8229361	total: 12m 26s	remaining: 59.1s
139: learn: 0.8230456	total: 12m 36s	remaining: 54s
140: learn: 0.8232097	total: 12m 46s	remaining: 48.9s
141: learn: 0.8233505	total: 12m 55s	remaining: 43.7s
142: learn: 0.8234133	total: 13m 5s	remaining: 38.4s
143: learn: 0.8234639	total: 13m 16s	remaining: 33.2s
144: learn: 0.8234904	total: 13m 25s	remaining: 27.8s
145: learn: 0.8235477	total: 13m 34s	remaining: 22.3s
146: learn: 0.8236664	total: 13m 44s	remaining: 16.8s
147: learn: 0.8238007	total: 13m 51s	remaining: 11.2s
148: learn: 0.8238972	total: 14m 2s	remaining: 5.65s
149: learn: 0.8239376	total: 14m 11s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1
1        1     0.772383
4        1     0.489014
7        1     0.521924
10       1     0.946141
13       1     0.819401
# # # # # # # # # # 
0.704306818594
0.874437450256
0.21834603006
0.162179979404
0.143437151765
# # # # # # # # # # 

in model: CatC_top2_1  k-fold: 3 / 3

0: learn: 0.7799098	total: 10.6s	remaining: 26m 26s
1: learn: 0.795177	total: 20.1s	remaining: 24m 47s
2: learn: 0.8003194	total: 30.3s	remaining: 24m 42s
3: learn: 0.8031415	total: 39.8s	remaining: 24m 14s
4: learn: 0.805612	total: 48.9s	remaining: 23m 39s
5: learn: 0.8072177	total: 57.7s	remaining: 23m 3s
6: learn: 0.808944	total: 1m 7s	remaining: 22m 59s
7: learn: 0.8102232	total: 1m 19s	remaining: 23m 29s
8: learn: 0.8106958	total: 1m 28s	remaining: 23m 2s
9: learn: 0.8119093	total: 1m 38s	remaining: 22m 52s
10: learn: 0.8125524	total: 1m 48s	remaining: 22m 53s
11: learn: 0.8133467	total: 2m	remaining: 23m
12: learn: 0.8133467	total: 2m 2s	remaining: 21m 34s
13: learn: 0.8133467	total: 2m 5s	remaining: 20m 19s
14: learn: 0.8133467	total: 2m 8s	remaining: 19m 15s
15: learn: 0.8137349	total: 2m 17s	remaining: 19m 13s
16: learn: 0.8137349	total: 2m 20s	remaining: 18m 18s
17: learn: 0.8137349	total: 2m 23s	remaining: 17m 29s
18: learn: 0.8137349	total: 2m 26s	remaining: 16m 46s
19: learn: 0.8137349	total: 2m 28s	remaining: 16m 6s
20: learn: 0.8137349	total: 2m 31s	remaining: 15m 31s
21: learn: 0.8137349	total: 2m 34s	remaining: 14m 57s
22: learn: 0.8137349	total: 2m 37s	remaining: 14m 32s
23: learn: 0.8144389	total: 2m 48s	remaining: 14m 46s
24: learn: 0.8147947	total: 2m 58s	remaining: 14m 50s
25: learn: 0.8153941	total: 3m 9s	remaining: 15m 1s
26: learn: 0.815818	total: 3m 18s	remaining: 15m 5s
27: learn: 0.8160399	total: 3m 28s	remaining: 15m 7s
28: learn: 0.8160399	total: 3m 30s	remaining: 14m 40s
29: learn: 0.8160399	total: 3m 33s	remaining: 14m 14s
30: learn: 0.8160399	total: 3m 36s	remaining: 13m 51s
31: learn: 0.8160399	total: 3m 39s	remaining: 13m 28s
32: learn: 0.8160399	total: 3m 42s	remaining: 13m 7s
33: learn: 0.8160399	total: 3m 44s	remaining: 12m 47s
34: learn: 0.8160399	total: 3m 47s	remaining: 12m 28s
35: learn: 0.8160399	total: 3m 50s	remaining: 12m 9s
36: learn: 0.8160399	total: 3m 53s	remaining: 11m 52s
37: learn: 0.8160399	total: 3m 56s	remaining: 11m 35s
38: learn: 0.8160399	total: 3m 59s	remaining: 11m 21s
39: learn: 0.8160399	total: 4m 2s	remaining: 11m 6s
40: learn: 0.8165629	total: 4m 13s	remaining: 11m 13s
41: learn: 0.8165629	total: 4m 16s	remaining: 10m 58s
42: learn: 0.8165629	total: 4m 18s	remaining: 10m 44s
43: learn: 0.8165629	total: 4m 21s	remaining: 10m 30s
44: learn: 0.8165629	total: 4m 24s	remaining: 10m 16s
45: learn: 0.8165629	total: 4m 27s	remaining: 10m 3s
46: learn: 0.8165629	total: 4m 29s	remaining: 9m 51s
47: learn: 0.8165629	total: 4m 32s	remaining: 9m 39s
48: learn: 0.8165629	total: 4m 35s	remaining: 9m 27s
49: learn: 0.8165629	total: 4m 38s	remaining: 9m 16s
50: learn: 0.8165629	total: 4m 40s	remaining: 9m 5s
51: learn: 0.8165629	total: 4m 43s	remaining: 8m 54s
52: learn: 0.8165629	total: 4m 47s	remaining: 8m 45s
53: learn: 0.8165629	total: 4m 49s	remaining: 8m 35s
54: learn: 0.8165629	total: 4m 54s	remaining: 8m 27s
55: learn: 0.8167511	total: 5m 3s	remaining: 8m 29s
56: learn: 0.8167511	total: 5m 6s	remaining: 8m 19s
57: learn: 0.8167511	total: 5m 8s	remaining: 8m 9s
58: learn: 0.8167511	total: 5m 11s	remaining: 8m
59: learn: 0.8167511	total: 5m 14s	remaining: 7m 51s
60: learn: 0.8167511	total: 5m 17s	remaining: 7m 42s
61: learn: 0.8167511	total: 5m 19s	remaining: 7m 34s
62: learn: 0.8167511	total: 5m 23s	remaining: 7m 26s
63: learn: 0.8167511	total: 5m 26s	remaining: 7m 18s
64: learn: 0.8167511	total: 5m 28s	remaining: 7m 9s
65: learn: 0.8172339	total: 5m 38s	remaining: 7m 11s
66: learn: 0.8175018	total: 5m 49s	remaining: 7m 12s
67: learn: 0.8178015	total: 5m 59s	remaining: 7m 13s
68: learn: 0.8180739	total: 6m 9s	remaining: 7m 13s
69: learn: 0.8181755	total: 6m 19s	remaining: 7m 13s
70: learn: 0.8184101	total: 6m 28s	remaining: 7m 12s
71: learn: 0.8188313	total: 6m 38s	remaining: 7m 11s
72: learn: 0.818832	total: 6m 47s	remaining: 7m 9s
73: learn: 0.8189831	total: 6m 56s	remaining: 7m 8s
74: learn: 0.8189831	total: 7m 1s	remaining: 7m 1s
75: learn: 0.8189831	total: 7m 4s	remaining: 6m 53s
76: learn: 0.8189831	total: 7m 8s	remaining: 6m 46s
77: learn: 0.8189831	total: 7m 11s	remaining: 6m 38s
78: learn: 0.8189831	total: 7m 15s	remaining: 6m 30s
79: learn: 0.8189831	total: 7m 18s	remaining: 6m 23s
80: learn: 0.8191468	total: 7m 28s	remaining: 6m 22s
81: learn: 0.8191469	total: 7m 34s	remaining: 6m 17s
82: learn: 0.8191469	total: 7m 37s	remaining: 6m 9s
83: learn: 0.8191469	total: 7m 42s	remaining: 6m 3s
84: learn: 0.819341	total: 7m 52s	remaining: 6m 1s
85: learn: 0.8193534	total: 8m	remaining: 5m 57s
86: learn: 0.819545	total: 8m 10s	remaining: 5m 54s
87: learn: 0.8198047	total: 8m 17s	remaining: 5m 50s
88: learn: 0.8202049	total: 8m 26s	remaining: 5m 47s
89: learn: 0.8204357	total: 8m 37s	remaining: 5m 44s
90: learn: 0.8206582	total: 8m 47s	remaining: 5m 41s
91: learn: 0.8207471	total: 8m 54s	remaining: 5m 36s
92: learn: 0.8208958	total: 9m 4s	remaining: 5m 33s
93: learn: 0.8209466	total: 9m 11s	remaining: 5m 28s
94: learn: 0.8210504	total: 9m 21s	remaining: 5m 25s
95: learn: 0.8211036	total: 9m 30s	remaining: 5m 21s
96: learn: 0.8212097	total: 9m 40s	remaining: 5m 17s
97: learn: 0.8214513	total: 9m 50s	remaining: 5m 13s
98: learn: 0.8215364	total: 9m 58s	remaining: 5m 8s
99: learn: 0.8216634	total: 10m 8s	remaining: 5m 4s
100: learn: 0.8218868	total: 10m 18s	remaining: 4m 59s
101: learn: 0.821953	total: 10m 28s	remaining: 4m 55s
102: learn: 0.8220326	total: 10m 38s	remaining: 4m 51s
103: learn: 0.8220326	total: 10m 42s	remaining: 4m 44s
104: learn: 0.8220326	total: 10m 46s	remaining: 4m 36s
105: learn: 0.8220326	total: 10m 49s	remaining: 4m 29s
106: learn: 0.8220326	total: 10m 53s	remaining: 4m 22s
107: learn: 0.8220326	total: 10m 57s	remaining: 4m 15s
108: learn: 0.8220326	total: 11m	remaining: 4m 8s
109: learn: 0.8220326	total: 11m 5s	remaining: 4m 1s
110: learn: 0.8220699	total: 11m 12s	remaining: 3m 56s
111: learn: 0.8222947	total: 11m 21s	remaining: 3m 51s
112: learn: 0.8224563	total: 11m 31s	remaining: 3m 46s
113: learn: 0.8224961	total: 11m 41s	remaining: 3m 41s
114: learn: 0.8225772	total: 11m 49s	remaining: 3m 36s
115: learn: 0.822633	total: 11m 58s	remaining: 3m 30s
116: learn: 0.8226895	total: 12m 8s	remaining: 3m 25s
117: learn: 0.8228726	total: 12m 18s	remaining: 3m 20s
118: learn: 0.8230523	total: 12m 28s	remaining: 3m 14s
119: learn: 0.8232104	total: 12m 35s	remaining: 3m 8s
120: learn: 0.8233153	total: 12m 43s	remaining: 3m 3s
121: learn: 0.823376	total: 12m 53s	remaining: 2m 57s
122: learn: 0.8234413	total: 13m 4s	remaining: 2m 52s
123: learn: 0.8235633	total: 13m 12s	remaining: 2m 46s
124: learn: 0.8236517	total: 13m 21s	remaining: 2m 40s
125: learn: 0.8238046	total: 13m 31s	remaining: 2m 34s
126: learn: 0.8238889	total: 13m 40s	remaining: 2m 28s
127: learn: 0.8239704	total: 13m 50s	remaining: 2m 22s
128: learn: 0.8240024	total: 14m	remaining: 2m 16s
129: learn: 0.8240524	total: 14m 9s	remaining: 2m 10s
130: learn: 0.8241732	total: 14m 18s	remaining: 2m 4s
131: learn: 0.8242315	total: 14m 27s	remaining: 1m 58s
132: learn: 0.8242582	total: 14m 39s	remaining: 1m 52s
133: learn: 0.8242859	total: 14m 49s	remaining: 1m 46s
134: learn: 0.8243955	total: 14m 58s	remaining: 1m 39s
135: learn: 0.8243955	total: 15m 2s	remaining: 1m 32s
136: learn: 0.8243955	total: 15m 5s	remaining: 1m 25s
137: learn: 0.8243955	total: 15m 11s	remaining: 1m 19s
138: learn: 0.8243955	total: 15m 15s	remaining: 1m 12s
139: learn: 0.8243955	total: 15m 20s	remaining: 1m 5s
140: learn: 0.8244252	total: 15m 28s	remaining: 59.3s
141: learn: 0.8244252	total: 15m 32s	remaining: 52.5s
142: learn: 0.8244252	total: 15m 35s	remaining: 45.8s
143: learn: 0.8244276	total: 15m 42s	remaining: 39.3s
144: learn: 0.8244618	total: 15m 49s	remaining: 32.7s
145: learn: 0.8245722	total: 15m 59s	remaining: 26.3s
146: learn: 0.8246151	total: 16m 7s	remaining: 19.7s
147: learn: 0.8246673	total: 16m 15s	remaining: 13.2s
148: learn: 0.8247541	total: 16m 26s	remaining: 6.62s
149: learn: 0.8247829	total: 16m 35s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1
2        1     0.745452
5        1     0.731993
8        1     0.892745
11       1     0.937748
14       1     0.962778
# # # # # # # # # # 
1.01276833269
1.31392029914
0.267841915511
0.232190856647
0.210964557312
# # # # # # # # # # 
  id  CatC_top2_1
0  0     0.337589
1  1     0.437973
2  2     0.089281
3  3     0.077397
4  4     0.070322

in model: CatR_top2_1  k-fold: 1 / 3

0: learn: 0.7955779	total: 15.7s	remaining: 28m 30s
1: learn: 0.8003107	total: 28.8s	remaining: 25m 54s
2: learn: 0.8023413	total: 41.7s	remaining: 24m 47s
3: learn: 0.804291	total: 52.7s	remaining: 23m 15s
4: learn: 0.8049642	total: 1m 4s	remaining: 22m 25s
5: learn: 0.805299	total: 1m 14s	remaining: 21m 31s
6: learn: 0.8060125	total: 1m 25s	remaining: 20m 57s
7: learn: 0.8064847	total: 1m 36s	remaining: 20m 35s
8: learn: 0.8069958	total: 1m 49s	remaining: 20m 27s
9: learn: 0.8073345	total: 2m 2s	remaining: 20m 27s
10: learn: 0.8078456	total: 2m 14s	remaining: 20m 7s
11: learn: 0.8082706	total: 2m 26s	remaining: 19m 57s
12: learn: 0.8084788	total: 2m 39s	remaining: 19m 48s
13: learn: 0.8087057	total: 2m 52s	remaining: 19m 43s
14: learn: 0.8087825	total: 3m 4s	remaining: 19m 26s
15: learn: 0.8091926	total: 3m 16s	remaining: 19m 13s
16: learn: 0.8096132	total: 3m 28s	remaining: 18m 59s
17: learn: 0.8101162	total: 3m 43s	remaining: 19m 1s
18: learn: 0.8103713	total: 3m 56s	remaining: 18m 53s
19: learn: 0.8106777	total: 4m 9s	remaining: 18m 44s
20: learn: 0.8109036	total: 4m 20s	remaining: 18m 25s
21: learn: 0.8112901	total: 4m 33s	remaining: 18m 12s
22: learn: 0.8115723	total: 4m 45s	remaining: 17m 59s
23: learn: 0.8117817	total: 4m 57s	remaining: 17m 44s
24: learn: 0.8121044	total: 5m 10s	remaining: 17m 35s
25: learn: 0.8122552	total: 5m 22s	remaining: 17m 22s
26: learn: 0.8123769	total: 5m 33s	remaining: 17m 4s
27: learn: 0.8125675	total: 5m 44s	remaining: 16m 48s
28: learn: 0.812893	total: 5m 57s	remaining: 16m 37s
29: learn: 0.8129651	total: 6m 8s	remaining: 16m 23s
30: learn: 0.8132254	total: 6m 21s	remaining: 16m 12s
31: learn: 0.8134064	total: 6m 34s	remaining: 16m 1s
32: learn: 0.8135286	total: 6m 47s	remaining: 15m 50s
33: learn: 0.8137949	total: 7m	remaining: 15m 40s
34: learn: 0.813916	total: 7m 14s	remaining: 15m 30s
35: learn: 0.8140733	total: 7m 28s	remaining: 15m 22s
36: learn: 0.8143345	total: 7m 41s	remaining: 15m 10s
37: learn: 0.8145296	total: 7m 53s	remaining: 14m 58s
38: learn: 0.8146517	total: 8m 8s	remaining: 14m 49s
39: learn: 0.8148744	total: 8m 21s	remaining: 14m 37s
40: learn: 0.8150485	total: 8m 33s	remaining: 14m 24s
41: learn: 0.8151844	total: 8m 46s	remaining: 14m 12s
42: learn: 0.8153552	total: 8m 57s	remaining: 13m 58s
43: learn: 0.8155077	total: 9m 11s	remaining: 13m 46s
44: learn: 0.8157248	total: 9m 24s	remaining: 13m 35s
45: learn: 0.8159107	total: 9m 38s	remaining: 13m 25s
46: learn: 0.8160649	total: 9m 53s	remaining: 13m 15s
47: learn: 0.8162007	total: 10m 9s	remaining: 13m 7s
48: learn: 0.8163386	total: 10m 21s	remaining: 12m 53s
49: learn: 0.8165128	total: 10m 34s	remaining: 12m 41s
50: learn: 0.8166321	total: 10m 49s	remaining: 12m 30s
51: learn: 0.8167546	total: 11m 1s	remaining: 12m 17s
52: learn: 0.8168911	total: 11m 13s	remaining: 12m 4s
53: learn: 0.8170713	total: 11m 27s	remaining: 11m 53s
54: learn: 0.8171763	total: 11m 40s	remaining: 11m 40s
55: learn: 0.8172984	total: 11m 53s	remaining: 11m 27s
56: learn: 0.8174672	total: 12m 10s	remaining: 11m 19s
57: learn: 0.8176073	total: 12m 26s	remaining: 11m 9s
58: learn: 0.8177255	total: 12m 45s	remaining: 11m 1s
59: learn: 0.8179088	total: 13m	remaining: 10m 50s
60: learn: 0.8180216	total: 13m 17s	remaining: 10m 40s
61: learn: 0.8181683	total: 13m 32s	remaining: 10m 29s
62: learn: 0.8183147	total: 13m 49s	remaining: 10m 18s
63: learn: 0.8184326	total: 14m 5s	remaining: 10m 7s
64: learn: 0.8185489	total: 14m 20s	remaining: 9m 55s
65: learn: 0.8186962	total: 14m 37s	remaining: 9m 45s
66: learn: 0.8188759	total: 14m 52s	remaining: 9m 32s
67: learn: 0.8190617	total: 15m 7s	remaining: 9m 20s
68: learn: 0.8191867	total: 15m 21s	remaining: 9m 7s
69: learn: 0.8193318	total: 15m 34s	remaining: 8m 54s
70: learn: 0.8194567	total: 15m 53s	remaining: 8m 43s
71: learn: 0.819526	total: 16m 11s	remaining: 8m 32s
72: learn: 0.819662	total: 16m 25s	remaining: 8m 19s
73: learn: 0.8197885	total: 16m 39s	remaining: 8m 6s
74: learn: 0.8198997	total: 16m 56s	remaining: 7m 54s
75: learn: 0.8199757	total: 17m 13s	remaining: 7m 42s
76: learn: 0.8200503	total: 17m 28s	remaining: 7m 29s
77: learn: 0.8201893	total: 17m 42s	remaining: 7m 16s
78: learn: 0.8202976	total: 17m 58s	remaining: 7m 3s
79: learn: 0.8204232	total: 18m 14s	remaining: 6m 50s
80: learn: 0.8205224	total: 18m 28s	remaining: 6m 37s
81: learn: 0.8206485	total: 18m 43s	remaining: 6m 23s
82: learn: 0.8207418	total: 18m 57s	remaining: 6m 10s
83: learn: 0.820836	total: 19m 15s	remaining: 5m 57s
84: learn: 0.8209517	total: 19m 29s	remaining: 5m 43s
85: learn: 0.8210354	total: 19m 45s	remaining: 5m 30s
86: learn: 0.821143	total: 20m 1s	remaining: 5m 17s
87: learn: 0.8212299	total: 20m 17s	remaining: 5m 4s
88: learn: 0.8213063	total: 20m 31s	remaining: 4m 50s
89: learn: 0.8213831	total: 20m 50s	remaining: 4m 37s
90: learn: 0.821468	total: 21m 5s	remaining: 4m 24s
91: learn: 0.8215394	total: 21m 20s	remaining: 4m 10s
92: learn: 0.8216334	total: 21m 35s	remaining: 3m 56s
93: learn: 0.8216842	total: 21m 51s	remaining: 3m 43s
94: learn: 0.8217607	total: 22m 7s	remaining: 3m 29s
95: learn: 0.8218656	total: 22m 23s	remaining: 3m 15s
96: learn: 0.8219561	total: 22m 38s	remaining: 3m 2s
97: learn: 0.8220163	total: 22m 52s	remaining: 2m 48s
98: learn: 0.8221229	total: 23m 7s	remaining: 2m 34s
99: learn: 0.8222026	total: 23m 23s	remaining: 2m 20s
100: learn: 0.8222847	total: 23m 40s	remaining: 2m 6s
101: learn: 0.8223535	total: 23m 56s	remaining: 1m 52s
102: learn: 0.8224176	total: 24m 13s	remaining: 1m 38s
103: learn: 0.8224867	total: 24m 28s	remaining: 1m 24s
104: learn: 0.8225677	total: 24m 42s	remaining: 1m 10s
105: learn: 0.8226229	total: 24m 55s	remaining: 56.4s
106: learn: 0.8226927	total: 25m 14s	remaining: 42.5s
107: learn: 0.8227494	total: 25m 29s	remaining: 28.3s
108: learn: 0.8228055	total: 25m 44s	remaining: 14.2s
109: learn: 0.8228686	total: 25m 59s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1
0        1     0.473860     0.483480
3        1     0.510477     0.531194
6        1     0.755119     0.754216
9        1     0.917777     0.911412
12       0     0.785798     0.832348
# # # # # # # # # # 
0.395011027162
0.580597447835
0.181427122851
0.072761761001
0.0758261508055
# # # # # # # # # # 

in model: CatR_top2_1  k-fold: 2 / 3

0: learn: 0.7959657	total: 14.5s	remaining: 26m 21s
1: learn: 0.7988873	total: 26.9s	remaining: 24m 11s
2: learn: 0.8014029	total: 37s	remaining: 21m 59s
3: learn: 0.8037018	total: 49.3s	remaining: 21m 45s
4: learn: 0.8042586	total: 58.6s	remaining: 20m 31s
5: learn: 0.8050152	total: 1m 10s	remaining: 20m 21s
6: learn: 0.8054099	total: 1m 24s	remaining: 20m 38s
7: learn: 0.8054503	total: 1m 35s	remaining: 20m 19s
8: learn: 0.8063138	total: 1m 48s	remaining: 20m 12s
9: learn: 0.8067852	total: 1m 59s	remaining: 19m 57s
10: learn: 0.8071387	total: 2m 10s	remaining: 19m 36s
11: learn: 0.8073032	total: 2m 20s	remaining: 19m 9s
12: learn: 0.8078279	total: 2m 31s	remaining: 18m 49s
13: learn: 0.8082258	total: 2m 43s	remaining: 18m 44s
14: learn: 0.8083941	total: 2m 55s	remaining: 18m 33s
15: learn: 0.8088192	total: 3m 8s	remaining: 18m 27s
16: learn: 0.8091141	total: 3m 20s	remaining: 18m 17s
17: learn: 0.8094397	total: 3m 32s	remaining: 18m 4s
18: learn: 0.8096671	total: 3m 47s	remaining: 18m 8s
19: learn: 0.8099523	total: 3m 58s	remaining: 17m 51s
20: learn: 0.8102882	total: 4m 10s	remaining: 17m 40s
21: learn: 0.8103811	total: 4m 22s	remaining: 17m 28s
22: learn: 0.8105325	total: 4m 35s	remaining: 17m 20s
23: learn: 0.8109074	total: 4m 47s	remaining: 17m 11s
24: learn: 0.8112494	total: 4m 58s	remaining: 16m 54s
25: learn: 0.8115205	total: 5m 9s	remaining: 16m 41s
26: learn: 0.8117157	total: 5m 22s	remaining: 16m 30s
27: learn: 0.8119533	total: 5m 35s	remaining: 16m 22s
28: learn: 0.8121369	total: 5m 50s	remaining: 16m 18s
29: learn: 0.8123152	total: 6m 1s	remaining: 16m 5s
30: learn: 0.8125547	total: 6m 14s	remaining: 15m 53s
31: learn: 0.8128526	total: 6m 27s	remaining: 15m 43s
32: learn: 0.8130387	total: 6m 41s	remaining: 15m 36s
33: learn: 0.813214	total: 6m 53s	remaining: 15m 25s
34: learn: 0.8134189	total: 7m 5s	remaining: 15m 12s
35: learn: 0.8136881	total: 7m 19s	remaining: 15m 3s
36: learn: 0.8138926	total: 7m 32s	remaining: 14m 53s
37: learn: 0.8140034	total: 7m 45s	remaining: 14m 42s
38: learn: 0.8141308	total: 7m 59s	remaining: 14m 32s
39: learn: 0.8142689	total: 8m 11s	remaining: 14m 19s
40: learn: 0.8145154	total: 8m 23s	remaining: 14m 7s
41: learn: 0.8146794	total: 8m 36s	remaining: 13m 56s
42: learn: 0.814791	total: 8m 50s	remaining: 13m 45s
43: learn: 0.8149337	total: 9m 3s	remaining: 13m 34s
44: learn: 0.8151584	total: 9m 16s	remaining: 13m 23s
45: learn: 0.815298	total: 9m 31s	remaining: 13m 14s
46: learn: 0.8154298	total: 9m 45s	remaining: 13m 4s
47: learn: 0.8155619	total: 9m 57s	remaining: 12m 51s
48: learn: 0.8157688	total: 10m 9s	remaining: 12m 39s
49: learn: 0.8158747	total: 10m 22s	remaining: 12m 27s
50: learn: 0.8160377	total: 10m 36s	remaining: 12m 16s
51: learn: 0.8162013	total: 10m 51s	remaining: 12m 6s
52: learn: 0.8162995	total: 11m 6s	remaining: 11m 56s
53: learn: 0.8164346	total: 11m 20s	remaining: 11m 45s
54: learn: 0.8165193	total: 11m 34s	remaining: 11m 34s
55: learn: 0.8166147	total: 11m 48s	remaining: 11m 23s
56: learn: 0.816783	total: 12m 6s	remaining: 11m 15s
57: learn: 0.8169115	total: 12m 21s	remaining: 11m 4s
58: learn: 0.8171298	total: 12m 38s	remaining: 10m 55s
59: learn: 0.817236	total: 12m 52s	remaining: 10m 43s
60: learn: 0.817363	total: 13m 7s	remaining: 10m 32s
61: learn: 0.8175079	total: 13m 22s	remaining: 10m 21s
62: learn: 0.8176309	total: 13m 38s	remaining: 10m 10s
63: learn: 0.8177777	total: 13m 52s	remaining: 9m 58s
64: learn: 0.8178657	total: 14m 7s	remaining: 9m 46s
65: learn: 0.8179824	total: 14m 22s	remaining: 9m 35s
66: learn: 0.8181394	total: 14m 41s	remaining: 9m 25s
67: learn: 0.8183091	total: 14m 54s	remaining: 9m 12s
68: learn: 0.8185079	total: 15m 9s	remaining: 9m
69: learn: 0.8186352	total: 15m 26s	remaining: 8m 49s
70: learn: 0.8187583	total: 15m 42s	remaining: 8m 37s
71: learn: 0.8188736	total: 15m 57s	remaining: 8m 25s
72: learn: 0.819021	total: 16m 11s	remaining: 8m 12s
73: learn: 0.8191318	total: 16m 28s	remaining: 8m 1s
74: learn: 0.8192244	total: 16m 45s	remaining: 7m 49s
75: learn: 0.8193245	total: 17m	remaining: 7m 36s
76: learn: 0.8194476	total: 17m 14s	remaining: 7m 23s
77: learn: 0.8195576	total: 17m 31s	remaining: 7m 11s
78: learn: 0.8196809	total: 17m 45s	remaining: 6m 58s
79: learn: 0.8197699	total: 18m 1s	remaining: 6m 45s
80: learn: 0.8198604	total: 18m 19s	remaining: 6m 33s
81: learn: 0.8199627	total: 18m 34s	remaining: 6m 20s
82: learn: 0.8200604	total: 18m 49s	remaining: 6m 7s
83: learn: 0.8201474	total: 19m 3s	remaining: 5m 53s
84: learn: 0.8202351	total: 19m 18s	remaining: 5m 40s
85: learn: 0.8203266	total: 19m 37s	remaining: 5m 28s
86: learn: 0.8204137	total: 19m 51s	remaining: 5m 15s
87: learn: 0.8204901	total: 20m 9s	remaining: 5m 2s
88: learn: 0.8205611	total: 20m 23s	remaining: 4m 48s
89: learn: 0.8206326	total: 20m 37s	remaining: 4m 35s
90: learn: 0.820701	total: 20m 55s	remaining: 4m 22s
91: learn: 0.8207563	total: 21m 10s	remaining: 4m 8s
92: learn: 0.8208416	total: 21m 24s	remaining: 3m 54s
93: learn: 0.8208971	total: 21m 40s	remaining: 3m 41s
94: learn: 0.8209819	total: 21m 57s	remaining: 3m 27s
95: learn: 0.8210765	total: 22m 12s	remaining: 3m 14s
96: learn: 0.8211503	total: 22m 27s	remaining: 3m
97: learn: 0.8212644	total: 22m 44s	remaining: 2m 47s
98: learn: 0.82138	total: 23m	remaining: 2m 33s
99: learn: 0.8215009	total: 23m 17s	remaining: 2m 19s
100: learn: 0.8215751	total: 23m 33s	remaining: 2m 5s
101: learn: 0.821639	total: 23m 50s	remaining: 1m 52s
102: learn: 0.8217342	total: 24m 5s	remaining: 1m 38s
103: learn: 0.8217833	total: 24m 22s	remaining: 1m 24s
104: learn: 0.8218588	total: 24m 39s	remaining: 1m 10s
105: learn: 0.8219531	total: 24m 52s	remaining: 56.3s
106: learn: 0.8220291	total: 25m 6s	remaining: 42.2s
107: learn: 0.8221025	total: 25m 22s	remaining: 28.2s
108: learn: 0.8221705	total: 25m 36s	remaining: 14.1s
109: learn: 0.8222307	total: 25m 52s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1
1        1     0.772383     0.772278
4        1     0.489014     0.501482
7        1     0.521924     0.514333
10       1     0.946141     0.922728
13       1     0.819401     0.835823
# # # # # # # # # # 
0.730413506999
0.996398013033
0.350647280158
0.130401508812
0.1310354144
# # # # # # # # # # 

in model: CatR_top2_1  k-fold: 3 / 3

0: learn: 0.793929	total: 14.9s	remaining: 27m
1: learn: 0.7992116	total: 28.3s	remaining: 25m 28s
2: learn: 0.8011962	total: 40.9s	remaining: 24m 17s
3: learn: 0.8017492	total: 52.3s	remaining: 23m 4s
4: learn: 0.8036197	total: 1m 5s	remaining: 22m 54s
5: learn: 0.8047634	total: 1m 17s	remaining: 22m 17s
6: learn: 0.8054068	total: 1m 28s	remaining: 21m 41s
7: learn: 0.8061471	total: 1m 42s	remaining: 21m 43s
8: learn: 0.8062973	total: 1m 53s	remaining: 21m 14s
9: learn: 0.8070622	total: 2m 6s	remaining: 21m 5s
10: learn: 0.8076404	total: 2m 19s	remaining: 20m 53s
11: learn: 0.8078821	total: 2m 31s	remaining: 20m 40s
12: learn: 0.8082386	total: 2m 44s	remaining: 20m 29s
13: learn: 0.8087453	total: 2m 55s	remaining: 20m 3s
14: learn: 0.8090828	total: 3m 7s	remaining: 19m 47s
15: learn: 0.8093792	total: 3m 19s	remaining: 19m 34s
16: learn: 0.8096186	total: 3m 29s	remaining: 19m 5s
17: learn: 0.8098772	total: 3m 39s	remaining: 18m 44s
18: learn: 0.8101327	total: 3m 51s	remaining: 18m 26s
19: learn: 0.8103351	total: 4m 1s	remaining: 18m 8s
20: learn: 0.8107279	total: 4m 15s	remaining: 18m 2s
21: learn: 0.8109851	total: 4m 29s	remaining: 17m 59s
22: learn: 0.8112772	total: 4m 43s	remaining: 17m 51s
23: learn: 0.8116164	total: 4m 56s	remaining: 17m 44s
24: learn: 0.811757	total: 5m 9s	remaining: 17m 31s
25: learn: 0.811902	total: 5m 24s	remaining: 17m 26s
26: learn: 0.8120762	total: 5m 35s	remaining: 17m 11s
27: learn: 0.8122577	total: 5m 47s	remaining: 16m 58s
28: learn: 0.8125539	total: 5m 59s	remaining: 16m 43s
29: learn: 0.8128583	total: 6m 10s	remaining: 16m 27s
30: learn: 0.8130798	total: 6m 20s	remaining: 16m 9s
31: learn: 0.8132566	total: 6m 30s	remaining: 15m 52s
32: learn: 0.8135355	total: 6m 47s	remaining: 15m 51s
33: learn: 0.8136987	total: 7m 1s	remaining: 15m 41s
34: learn: 0.8138049	total: 7m 14s	remaining: 15m 30s
35: learn: 0.8139679	total: 7m 28s	remaining: 15m 21s
36: learn: 0.8142178	total: 7m 41s	remaining: 15m 11s
37: learn: 0.8144413	total: 7m 56s	remaining: 15m 2s
38: learn: 0.8146364	total: 8m 13s	remaining: 14m 57s
39: learn: 0.8148851	total: 8m 28s	remaining: 14m 49s
40: learn: 0.8150228	total: 8m 42s	remaining: 14m 38s
41: learn: 0.8151817	total: 8m 55s	remaining: 14m 27s
42: learn: 0.8153072	total: 9m 10s	remaining: 14m 17s
43: learn: 0.8155137	total: 9m 23s	remaining: 14m 4s
44: learn: 0.8156308	total: 9m 36s	remaining: 13m 52s
45: learn: 0.81579	total: 9m 51s	remaining: 13m 42s
46: learn: 0.8159392	total: 10m 3s	remaining: 13m 29s
47: learn: 0.8160864	total: 10m 17s	remaining: 13m 17s
48: learn: 0.8162262	total: 10m 29s	remaining: 13m 4s
49: learn: 0.8163589	total: 10m 42s	remaining: 12m 51s
50: learn: 0.816515	total: 10m 56s	remaining: 12m 39s
51: learn: 0.816612	total: 11m 7s	remaining: 12m 24s
52: learn: 0.8167148	total: 11m 21s	remaining: 12m 12s
53: learn: 0.8168946	total: 11m 35s	remaining: 12m 1s
54: learn: 0.8170489	total: 11m 50s	remaining: 11m 50s
55: learn: 0.8171602	total: 12m 3s	remaining: 11m 37s
56: learn: 0.8173007	total: 12m 17s	remaining: 11m 25s
57: learn: 0.8174469	total: 12m 28s	remaining: 11m 11s
58: learn: 0.8175201	total: 12m 43s	remaining: 10m 59s
59: learn: 0.8175759	total: 12m 52s	remaining: 10m 43s
60: learn: 0.8176957	total: 13m 3s	remaining: 10m 29s
61: learn: 0.8178621	total: 13m 20s	remaining: 10m 19s
62: learn: 0.8179485	total: 13m 37s	remaining: 10m 9s
63: learn: 0.8180886	total: 13m 52s	remaining: 9m 58s
64: learn: 0.8182224	total: 14m 7s	remaining: 9m 46s
65: learn: 0.8183803	total: 14m 22s	remaining: 9m 35s
66: learn: 0.8184921	total: 14m 37s	remaining: 9m 22s
67: learn: 0.8186321	total: 14m 54s	remaining: 9m 12s
68: learn: 0.8187015	total: 15m 7s	remaining: 8m 59s
69: learn: 0.8188501	total: 15m 24s	remaining: 8m 48s
70: learn: 0.8189417	total: 15m 40s	remaining: 8m 36s
71: learn: 0.8190841	total: 15m 53s	remaining: 8m 23s
72: learn: 0.8191576	total: 16m 8s	remaining: 8m 11s
73: learn: 0.8192392	total: 16m 23s	remaining: 7m 58s
74: learn: 0.819341	total: 16m 39s	remaining: 7m 46s
75: learn: 0.8194695	total: 16m 53s	remaining: 7m 33s
76: learn: 0.8195982	total: 17m 10s	remaining: 7m 21s
77: learn: 0.8197124	total: 17m 25s	remaining: 7m 8s
78: learn: 0.8198176	total: 17m 40s	remaining: 6m 56s
79: learn: 0.8199219	total: 17m 53s	remaining: 6m 42s
80: learn: 0.8200401	total: 18m 10s	remaining: 6m 30s
81: learn: 0.820135	total: 18m 27s	remaining: 6m 18s
82: learn: 0.8202295	total: 18m 42s	remaining: 6m 5s
83: learn: 0.8203906	total: 18m 58s	remaining: 5m 52s
84: learn: 0.8205047	total: 19m 15s	remaining: 5m 39s
85: learn: 0.8205838	total: 19m 31s	remaining: 5m 26s
86: learn: 0.8206633	total: 19m 47s	remaining: 5m 14s
87: learn: 0.820725	total: 20m 2s	remaining: 5m
88: learn: 0.8207839	total: 20m 17s	remaining: 4m 47s
89: learn: 0.8208543	total: 20m 34s	remaining: 4m 34s
90: learn: 0.8209771	total: 20m 51s	remaining: 4m 21s
91: learn: 0.8210585	total: 21m 7s	remaining: 4m 7s
92: learn: 0.8211905	total: 21m 22s	remaining: 3m 54s
93: learn: 0.8212887	total: 21m 39s	remaining: 3m 41s
94: learn: 0.8213632	total: 21m 53s	remaining: 3m 27s
95: learn: 0.8214697	total: 22m 10s	remaining: 3m 14s
96: learn: 0.8215284	total: 22m 27s	remaining: 3m
97: learn: 0.8216033	total: 22m 42s	remaining: 2m 46s
98: learn: 0.8216693	total: 23m	remaining: 2m 33s
99: learn: 0.8217494	total: 23m 16s	remaining: 2m 19s
100: learn: 0.8218186	total: 23m 30s	remaining: 2m 5s
101: learn: 0.8218984	total: 23m 47s	remaining: 1m 51s
102: learn: 0.8219865	total: 24m 4s	remaining: 1m 38s
103: learn: 0.8220617	total: 24m 20s	remaining: 1m 24s
104: learn: 0.8221137	total: 24m 37s	remaining: 1m 10s
105: learn: 0.8221779	total: 24m 55s	remaining: 56.4s
106: learn: 0.8222128	total: 25m 14s	remaining: 42.5s
107: learn: 0.8223047	total: 25m 32s	remaining: 28.4s
108: learn: 0.8223697	total: 25m 49s	remaining: 14.2s
109: learn: 0.8224568	total: 26m 6s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1
2        1     0.745452     0.731489
5        1     0.731993     0.613060
8        1     0.892745     0.888683
11       1     0.937748     0.933345
14       1     0.962778     0.961574
# # # # # # # # # # 
1.03290244768
1.47774009533
0.394258561202
0.192886060481
0.198433877984
# # # # # # # # # # 
  id  CatC_top2_1  CatR_top2_1
0  0     0.337589     0.344301
1  1     0.437973     0.492580
2  2     0.089281     0.131420
3  3     0.077397     0.064295
4  4     0.070322     0.066145

in model: CatC_top2_2  k-fold: 1 / 3

0: learn: 0.7529124	total: 8.54s	remaining: 14m 5s
1: learn: 0.787416	total: 14.8s	remaining: 12m 6s
2: learn: 0.7925106	total: 22.9s	remaining: 12m 19s
3: learn: 0.7977076	total: 31.2s	remaining: 12m 28s
4: learn: 0.8001228	total: 37.9s	remaining: 12m
5: learn: 0.8024522	total: 45.5s	remaining: 11m 52s
6: learn: 0.8039185	total: 52.1s	remaining: 11m 31s
7: learn: 0.8039185	total: 55.4s	remaining: 10m 36s
8: learn: 0.8039185	total: 58.8s	remaining: 9m 54s
9: learn: 0.8050416	total: 1m 4s	remaining: 9m 43s
10: learn: 0.8050416	total: 1m 7s	remaining: 9m 6s
11: learn: 0.8050416	total: 1m 10s	remaining: 8m 34s
12: learn: 0.8050416	total: 1m 12s	remaining: 8m 7s
13: learn: 0.8050418	total: 1m 16s	remaining: 7m 50s
14: learn: 0.8050423	total: 1m 21s	remaining: 7m 43s
15: learn: 0.806027	total: 1m 29s	remaining: 7m 50s
16: learn: 0.806027	total: 1m 32s	remaining: 7m 33s
17: learn: 0.806027	total: 1m 35s	remaining: 7m 15s
18: learn: 0.806027	total: 1m 38s	remaining: 6m 59s
19: learn: 0.8060271	total: 1m 42s	remaining: 6m 50s
20: learn: 0.807877	total: 1m 49s	remaining: 6m 53s
21: learn: 0.8087576	total: 1m 57s	remaining: 6m 56s
22: learn: 0.8087576	total: 2m	remaining: 6m 43s
23: learn: 0.8087576	total: 2m 3s	remaining: 6m 31s
24: learn: 0.8087576	total: 2m 6s	remaining: 6m 19s
25: learn: 0.8087576	total: 2m 9s	remaining: 6m 7s
26: learn: 0.8096001	total: 2m 16s	remaining: 6m 7s
27: learn: 0.8096002	total: 2m 21s	remaining: 6m 2s
28: learn: 0.8096002	total: 2m 26s	remaining: 5m 58s
29: learn: 0.8096002	total: 2m 29s	remaining: 5m 49s
30: learn: 0.8096003	total: 2m 33s	remaining: 5m 41s
31: learn: 0.8096003	total: 2m 36s	remaining: 5m 31s
32: learn: 0.810189	total: 2m 42s	remaining: 5m 30s
33: learn: 0.8106598	total: 2m 48s	remaining: 5m 27s
34: learn: 0.8115544	total: 2m 57s	remaining: 5m 29s
35: learn: 0.8120403	total: 3m 5s	remaining: 5m 30s
36: learn: 0.8124765	total: 3m 13s	remaining: 5m 29s
37: learn: 0.8131506	total: 3m 21s	remaining: 5m 29s
38: learn: 0.8136091	total: 3m 29s	remaining: 5m 26s
39: learn: 0.8140838	total: 3m 36s	remaining: 5m 24s
40: learn: 0.8145272	total: 3m 44s	remaining: 5m 22s
41: learn: 0.8151234	total: 3m 51s	remaining: 5m 19s
42: learn: 0.8152862	total: 3m 59s	remaining: 5m 17s
43: learn: 0.8154421	total: 4m 6s	remaining: 5m 13s
44: learn: 0.8154421	total: 4m 12s	remaining: 5m 8s
45: learn: 0.8157078	total: 4m 20s	remaining: 5m 5s
46: learn: 0.8161123	total: 4m 28s	remaining: 5m 2s
47: learn: 0.8166349	total: 4m 34s	remaining: 4m 57s
48: learn: 0.8168691	total: 4m 40s	remaining: 4m 52s
49: learn: 0.8171895	total: 4m 47s	remaining: 4m 47s
50: learn: 0.8173102	total: 4m 55s	remaining: 4m 44s
51: learn: 0.8176161	total: 5m 2s	remaining: 4m 39s
52: learn: 0.8178319	total: 5m 9s	remaining: 4m 34s
53: learn: 0.8180716	total: 5m 16s	remaining: 4m 29s
54: learn: 0.8182651	total: 5m 24s	remaining: 4m 25s
55: learn: 0.8183388	total: 5m 32s	remaining: 4m 21s
56: learn: 0.8184705	total: 5m 41s	remaining: 4m 17s
57: learn: 0.8190396	total: 5m 48s	remaining: 4m 12s
58: learn: 0.8190396	total: 5m 54s	remaining: 4m 6s
59: learn: 0.8190396	total: 5m 59s	remaining: 3m 59s
60: learn: 0.8190396	total: 6m 4s	remaining: 3m 53s
61: learn: 0.8190396	total: 6m 8s	remaining: 3m 46s
62: learn: 0.8190396	total: 6m 13s	remaining: 3m 39s
63: learn: 0.8193469	total: 6m 20s	remaining: 3m 34s
64: learn: 0.819347	total: 6m 27s	remaining: 3m 28s
65: learn: 0.819347	total: 6m 30s	remaining: 3m 20s
66: learn: 0.819347	total: 6m 35s	remaining: 3m 14s
67: learn: 0.819347	total: 6m 40s	remaining: 3m 8s
68: learn: 0.8195078	total: 6m 48s	remaining: 3m 3s
69: learn: 0.8197836	total: 6m 54s	remaining: 2m 57s
70: learn: 0.819851	total: 7m 3s	remaining: 2m 52s
71: learn: 0.8199022	total: 7m 10s	remaining: 2m 47s
72: learn: 0.8200982	total: 7m 16s	remaining: 2m 41s
73: learn: 0.820571	total: 7m 23s	remaining: 2m 35s
74: learn: 0.8207137	total: 7m 32s	remaining: 2m 30s
75: learn: 0.8207871	total: 7m 39s	remaining: 2m 25s
76: learn: 0.82094	total: 7m 45s	remaining: 2m 19s
77: learn: 0.8210638	total: 7m 52s	remaining: 2m 13s
78: learn: 0.8211255	total: 8m	remaining: 2m 7s
79: learn: 0.8212573	total: 8m 7s	remaining: 2m 1s
80: learn: 0.8213808	total: 8m 15s	remaining: 1m 56s
81: learn: 0.8214952	total: 8m 21s	remaining: 1m 50s
82: learn: 0.8215783	total: 8m 29s	remaining: 1m 44s
83: learn: 0.8216765	total: 8m 36s	remaining: 1m 38s
84: learn: 0.8217341	total: 8m 43s	remaining: 1m 32s
85: learn: 0.8218059	total: 8m 51s	remaining: 1m 26s
86: learn: 0.8219314	total: 8m 59s	remaining: 1m 20s
87: learn: 0.8220221	total: 9m 5s	remaining: 1m 14s
88: learn: 0.8221019	total: 9m 11s	remaining: 1m 8s
89: learn: 0.8221484	total: 9m 17s	remaining: 1m 1s
90: learn: 0.8221985	total: 9m 23s	remaining: 55.7s
91: learn: 0.8222818	total: 9m 30s	remaining: 49.6s
92: learn: 0.8223836	total: 9m 36s	remaining: 43.4s
93: learn: 0.8224381	total: 9m 43s	remaining: 37.3s
94: learn: 0.82247	total: 9m 50s	remaining: 31.1s
95: learn: 0.8225924	total: 9m 56s	remaining: 24.9s
96: learn: 0.8226009	total: 10m 1s	remaining: 18.6s
97: learn: 0.8227732	total: 10m 6s	remaining: 12.4s
98: learn: 0.8228631	total: 10m 12s	remaining: 6.19s
99: learn: 0.8229064	total: 10m 19s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2
0        1     0.473860     0.483480     0.414084
3        1     0.510477     0.531194     0.480202
6        1     0.755119     0.754216     0.747641
9        1     0.917777     0.911412     0.908166
12       0     0.785798     0.832348     0.868293
# # # # # # # # # # 
0.421249403088
0.548749989333
0.0631959468065
0.0956415290134
0.074011296864
# # # # # # # # # # 

in model: CatC_top2_2  k-fold: 2 / 3

0: learn: 0.7528418	total: 7.74s	remaining: 12m 46s
1: learn: 0.7832925	total: 14.6s	remaining: 11m 56s
2: learn: 0.791984	total: 21.6s	remaining: 11m 37s
3: learn: 0.7978224	total: 29s	remaining: 11m 34s
4: learn: 0.8001886	total: 36.4s	remaining: 11m 32s
5: learn: 0.8001886	total: 39s	remaining: 10m 10s
6: learn: 0.8001886	total: 41.4s	remaining: 9m 10s
7: learn: 0.8001886	total: 43.9s	remaining: 8m 24s
8: learn: 0.8015642	total: 51s	remaining: 8m 35s
9: learn: 0.8030177	total: 58.7s	remaining: 8m 47s
10: learn: 0.8043933	total: 1m 5s	remaining: 8m 52s
11: learn: 0.8056252	total: 1m 12s	remaining: 8m 51s
12: learn: 0.8056252	total: 1m 15s	remaining: 8m 22s
13: learn: 0.8056252	total: 1m 17s	remaining: 7m 57s
14: learn: 0.8056252	total: 1m 20s	remaining: 7m 35s
15: learn: 0.8056252	total: 1m 23s	remaining: 7m 16s
16: learn: 0.8056252	total: 1m 25s	remaining: 6m 59s
17: learn: 0.8056252	total: 1m 28s	remaining: 6m 43s
18: learn: 0.8066192	total: 1m 34s	remaining: 6m 44s
19: learn: 0.8066192	total: 1m 37s	remaining: 6m 29s
20: learn: 0.8066192	total: 1m 40s	remaining: 6m 16s
21: learn: 0.8066192	total: 1m 42s	remaining: 6m 4s
22: learn: 0.8066192	total: 1m 45s	remaining: 5m 53s
23: learn: 0.8075423	total: 1m 52s	remaining: 5m 56s
24: learn: 0.8075423	total: 1m 55s	remaining: 5m 46s
25: learn: 0.8075423	total: 1m 58s	remaining: 5m 36s
26: learn: 0.8075423	total: 2m	remaining: 5m 26s
27: learn: 0.8075423	total: 2m 3s	remaining: 5m 17s
28: learn: 0.8075423	total: 2m 6s	remaining: 5m 9s
29: learn: 0.8075423	total: 2m 9s	remaining: 5m 1s
30: learn: 0.8095389	total: 2m 15s	remaining: 5m 1s
31: learn: 0.810548	total: 2m 22s	remaining: 5m 3s
32: learn: 0.8111352	total: 2m 28s	remaining: 5m 2s
33: learn: 0.8117715	total: 2m 36s	remaining: 5m 3s
34: learn: 0.8117715	total: 2m 39s	remaining: 4m 56s
35: learn: 0.8117715	total: 2m 43s	remaining: 4m 49s
36: learn: 0.8117715	total: 2m 45s	remaining: 4m 42s
37: learn: 0.8117715	total: 2m 48s	remaining: 4m 35s
38: learn: 0.8117718	total: 2m 52s	remaining: 4m 30s
39: learn: 0.8117718	total: 2m 55s	remaining: 4m 23s
40: learn: 0.8122804	total: 3m 3s	remaining: 4m 23s
41: learn: 0.8126446	total: 3m 10s	remaining: 4m 22s
42: learn: 0.8129824	total: 3m 17s	remaining: 4m 21s
43: learn: 0.8132987	total: 3m 23s	remaining: 4m 19s
44: learn: 0.8139463	total: 3m 30s	remaining: 4m 17s
45: learn: 0.8141258	total: 3m 37s	remaining: 4m 15s
46: learn: 0.814126	total: 3m 43s	remaining: 4m 11s
47: learn: 0.814126	total: 3m 46s	remaining: 4m 5s
48: learn: 0.814126	total: 3m 49s	remaining: 3m 59s
49: learn: 0.814126	total: 3m 53s	remaining: 3m 53s
50: learn: 0.8143588	total: 3m 58s	remaining: 3m 49s
51: learn: 0.8143588	total: 4m 2s	remaining: 3m 43s
52: learn: 0.8143588	total: 4m 5s	remaining: 3m 37s
53: learn: 0.8150779	total: 4m 12s	remaining: 3m 34s
54: learn: 0.815501	total: 4m 19s	remaining: 3m 32s
55: learn: 0.8160185	total: 4m 26s	remaining: 3m 29s
56: learn: 0.8161302	total: 4m 33s	remaining: 3m 26s
57: learn: 0.8163478	total: 4m 40s	remaining: 3m 23s
58: learn: 0.8165041	total: 4m 46s	remaining: 3m 19s
59: learn: 0.8169171	total: 4m 53s	remaining: 3m 15s
60: learn: 0.8169826	total: 5m	remaining: 3m 12s
61: learn: 0.8172036	total: 5m 7s	remaining: 3m 8s
62: learn: 0.8174124	total: 5m 14s	remaining: 3m 4s
63: learn: 0.8176215	total: 5m 22s	remaining: 3m 1s
64: learn: 0.8176215	total: 5m 25s	remaining: 2m 55s
65: learn: 0.8176215	total: 5m 28s	remaining: 2m 49s
66: learn: 0.8176215	total: 5m 32s	remaining: 2m 43s
67: learn: 0.8181768	total: 5m 38s	remaining: 2m 39s
68: learn: 0.8183304	total: 5m 43s	remaining: 2m 34s
69: learn: 0.8184389	total: 5m 49s	remaining: 2m 29s
70: learn: 0.8186152	total: 5m 56s	remaining: 2m 25s
71: learn: 0.8187832	total: 6m 2s	remaining: 2m 20s
72: learn: 0.8188258	total: 6m 9s	remaining: 2m 16s
73: learn: 0.8189963	total: 6m 16s	remaining: 2m 12s
74: learn: 0.8189963	total: 6m 18s	remaining: 2m 6s
75: learn: 0.8190615	total: 6m 25s	remaining: 2m 1s
76: learn: 0.8190615	total: 6m 28s	remaining: 1m 56s
77: learn: 0.8190615	total: 6m 32s	remaining: 1m 50s
78: learn: 0.8190615	total: 6m 35s	remaining: 1m 45s
79: learn: 0.8192049	total: 6m 42s	remaining: 1m 40s
80: learn: 0.819357	total: 6m 50s	remaining: 1m 36s
81: learn: 0.8193885	total: 6m 55s	remaining: 1m 31s
82: learn: 0.8194976	total: 7m 3s	remaining: 1m 26s
83: learn: 0.8194981	total: 7m 9s	remaining: 1m 21s
84: learn: 0.8194981	total: 7m 11s	remaining: 1m 16s
85: learn: 0.8195824	total: 7m 19s	remaining: 1m 11s
86: learn: 0.8198505	total: 7m 26s	remaining: 1m 6s
87: learn: 0.8199149	total: 7m 33s	remaining: 1m 1s
88: learn: 0.8200169	total: 7m 39s	remaining: 56.8s
89: learn: 0.820075	total: 7m 46s	remaining: 51.8s
90: learn: 0.8202295	total: 7m 54s	remaining: 46.9s
91: learn: 0.8202793	total: 8m 1s	remaining: 41.8s
92: learn: 0.8203318	total: 8m 7s	remaining: 36.7s
93: learn: 0.8204104	total: 8m 15s	remaining: 31.6s
94: learn: 0.8204671	total: 8m 21s	remaining: 26.4s
95: learn: 0.8205303	total: 8m 28s	remaining: 21.2s
96: learn: 0.8205552	total: 8m 35s	remaining: 15.9s
97: learn: 0.8205988	total: 8m 41s	remaining: 10.7s
98: learn: 0.8208256	total: 8m 48s	remaining: 5.34s
99: learn: 0.8209275	total: 8m 56s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2
1        1     0.772383     0.772278     0.753011
4        1     0.489014     0.501482     0.538394
7        1     0.521924     0.514333     0.567273
10       1     0.946141     0.922728     0.943774
13       1     0.819401     0.835823     0.869749
# # # # # # # # # # 
0.812213191514
0.916604270395
0.156260546559
0.159906305657
0.117963713862
# # # # # # # # # # 

in model: CatC_top2_2  k-fold: 3 / 3

0: learn: 0.7538327	total: 7.8s	remaining: 12m 52s
1: learn: 0.7867077	total: 13.7s	remaining: 11m 10s
2: learn: 0.7932634	total: 20.3s	remaining: 10m 55s
3: learn: 0.7984191	total: 27.4s	remaining: 10m 58s
4: learn: 0.8001515	total: 33.4s	remaining: 10m 34s
5: learn: 0.8017544	total: 39.9s	remaining: 10m 24s
6: learn: 0.8017544	total: 42.3s	remaining: 9m 22s
7: learn: 0.8017544	total: 44.8s	remaining: 8m 34s
8: learn: 0.8017544	total: 47.2s	remaining: 7m 57s
9: learn: 0.8017544	total: 49.7s	remaining: 7m 27s
10: learn: 0.8017544	total: 52.2s	remaining: 7m 2s
11: learn: 0.8017544	total: 54.8s	remaining: 6m 41s
12: learn: 0.8017544	total: 57.2s	remaining: 6m 22s
13: learn: 0.8028527	total: 1m 4s	remaining: 6m 33s
14: learn: 0.8041487	total: 1m 9s	remaining: 6m 34s
15: learn: 0.8053985	total: 1m 15s	remaining: 6m 38s
16: learn: 0.8066983	total: 1m 22s	remaining: 6m 41s
17: learn: 0.8066983	total: 1m 24s	remaining: 6m 26s
18: learn: 0.8066983	total: 1m 28s	remaining: 6m 16s
19: learn: 0.8066983	total: 1m 30s	remaining: 6m 3s
20: learn: 0.8066983	total: 1m 33s	remaining: 5m 52s
21: learn: 0.8066983	total: 1m 36s	remaining: 5m 41s
22: learn: 0.8066983	total: 1m 39s	remaining: 5m 32s
23: learn: 0.8073278	total: 1m 45s	remaining: 5m 33s
24: learn: 0.8082877	total: 1m 52s	remaining: 5m 37s
25: learn: 0.8091835	total: 1m 58s	remaining: 5m 36s
26: learn: 0.8097278	total: 2m 4s	remaining: 5m 37s
27: learn: 0.8100327	total: 2m 11s	remaining: 5m 39s
28: learn: 0.8112756	total: 2m 18s	remaining: 5m 38s
29: learn: 0.8115514	total: 2m 25s	remaining: 5m 40s
30: learn: 0.8115514	total: 2m 28s	remaining: 5m 31s
31: learn: 0.8121071	total: 2m 35s	remaining: 5m 31s
32: learn: 0.8123786	total: 2m 42s	remaining: 5m 30s
33: learn: 0.8126111	total: 2m 48s	remaining: 5m 27s
34: learn: 0.8130691	total: 2m 54s	remaining: 5m 24s
35: learn: 0.8137047	total: 3m 1s	remaining: 5m 22s
36: learn: 0.8137047	total: 3m 4s	remaining: 5m 14s
37: learn: 0.8137047	total: 3m 8s	remaining: 5m 7s
38: learn: 0.8137047	total: 3m 11s	remaining: 4m 59s
39: learn: 0.8140982	total: 3m 18s	remaining: 4m 57s
40: learn: 0.8140982	total: 3m 22s	remaining: 4m 50s
41: learn: 0.8140982	total: 3m 26s	remaining: 4m 45s
42: learn: 0.8140982	total: 3m 30s	remaining: 4m 39s
43: learn: 0.8140982	total: 3m 35s	remaining: 4m 33s
44: learn: 0.8140982	total: 3m 38s	remaining: 4m 27s
45: learn: 0.8142704	total: 3m 45s	remaining: 4m 24s
46: learn: 0.8144605	total: 3m 50s	remaining: 4m 19s
47: learn: 0.8144608	total: 3m 56s	remaining: 4m 15s
48: learn: 0.8145379	total: 4m 2s	remaining: 4m 11s
49: learn: 0.8145379	total: 4m 5s	remaining: 4m 5s
50: learn: 0.8145379	total: 4m 8s	remaining: 3m 58s
51: learn: 0.8145379	total: 4m 12s	remaining: 3m 53s
52: learn: 0.8145379	total: 4m 15s	remaining: 3m 46s
53: learn: 0.8145379	total: 4m 19s	remaining: 3m 41s
54: learn: 0.8145379	total: 4m 23s	remaining: 3m 35s
55: learn: 0.8147721	total: 4m 30s	remaining: 3m 32s
56: learn: 0.8153356	total: 4m 36s	remaining: 3m 28s
57: learn: 0.8155237	total: 4m 43s	remaining: 3m 25s
58: learn: 0.816107	total: 4m 50s	remaining: 3m 21s
59: learn: 0.816107	total: 4m 54s	remaining: 3m 16s
60: learn: 0.816107	total: 4m 58s	remaining: 3m 10s
61: learn: 0.816107	total: 5m	remaining: 3m 4s
62: learn: 0.816107	total: 5m 4s	remaining: 2m 58s
63: learn: 0.816107	total: 5m 10s	remaining: 2m 54s
64: learn: 0.816107	total: 5m 13s	remaining: 2m 48s
65: learn: 0.8169211	total: 5m 19s	remaining: 2m 44s
66: learn: 0.8170547	total: 5m 25s	remaining: 2m 40s
67: learn: 0.8171644	total: 5m 32s	remaining: 2m 36s
68: learn: 0.8173554	total: 5m 38s	remaining: 2m 32s
69: learn: 0.8176359	total: 5m 46s	remaining: 2m 28s
70: learn: 0.8179834	total: 5m 54s	remaining: 2m 24s
71: learn: 0.8181473	total: 6m	remaining: 2m 20s
72: learn: 0.8183069	total: 6m 7s	remaining: 2m 15s
73: learn: 0.8186571	total: 6m 14s	remaining: 2m 11s
74: learn: 0.8187825	total: 6m 21s	remaining: 2m 7s
75: learn: 0.8189081	total: 6m 28s	remaining: 2m 2s
76: learn: 0.8189815	total: 6m 35s	remaining: 1m 58s
77: learn: 0.8190875	total: 6m 41s	remaining: 1m 53s
78: learn: 0.8191473	total: 6m 48s	remaining: 1m 48s
79: learn: 0.8193134	total: 6m 54s	remaining: 1m 43s
80: learn: 0.8195165	total: 7m 1s	remaining: 1m 38s
81: learn: 0.8195461	total: 7m 8s	remaining: 1m 34s
82: learn: 0.8197188	total: 7m 15s	remaining: 1m 29s
83: learn: 0.8198643	total: 7m 21s	remaining: 1m 24s
84: learn: 0.8200054	total: 7m 27s	remaining: 1m 19s
85: learn: 0.8200462	total: 7m 33s	remaining: 1m 13s
86: learn: 0.820109	total: 7m 40s	remaining: 1m 8s
87: learn: 0.8203584	total: 7m 47s	remaining: 1m 3s
88: learn: 0.820522	total: 7m 54s	remaining: 58.7s
89: learn: 0.820585	total: 8m 1s	remaining: 53.5s
90: learn: 0.8207102	total: 8m 8s	remaining: 48.3s
91: learn: 0.8208864	total: 8m 14s	remaining: 43s
92: learn: 0.8209404	total: 8m 21s	remaining: 37.7s
93: learn: 0.8210379	total: 8m 28s	remaining: 32.4s
94: learn: 0.8210732	total: 8m 35s	remaining: 27.1s
95: learn: 0.8211171	total: 8m 40s	remaining: 21.7s
96: learn: 0.8212067	total: 8m 47s	remaining: 16.3s
97: learn: 0.8212359	total: 8m 53s	remaining: 10.9s
98: learn: 0.8213371	total: 9m	remaining: 5.46s
99: learn: 0.8213966	total: 9m 7s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2
2        1     0.745452     0.731489     0.711371
5        1     0.731993     0.613060     0.697563
8        1     0.892745     0.888683     0.964299
11       1     0.937748     0.933345     0.935016
14       1     0.962778     0.961574     0.981358
# # # # # # # # # # 
1.15863973207
1.30350667807
0.224253332108
0.249952307054
0.205060027919
# # # # # # # # # # 
  id  CatC_top2_1  CatR_top2_1  CatC_top2_2
0  0     0.337589     0.344301     0.386213
1  1     0.437973     0.492580     0.434502
2  2     0.089281     0.131420     0.074751
3  3     0.077397     0.064295     0.083317
4  4     0.070322     0.066145     0.068353

in model: CatR_top2_2  k-fold: 1 / 3

0: learn: 0.8057489	total: 41.2s	remaining: 33m 39s
1: learn: 0.8136916	total: 1m 32s	remaining: 36m 59s
2: learn: 0.8175286	total: 2m 31s	remaining: 39m 33s
3: learn: 0.820591	total: 3m 14s	remaining: 37m 13s
4: learn: 0.8223374	total: 4m 2s	remaining: 36m 20s
5: learn: 0.8236899	total: 4m 47s	remaining: 35m 10s
6: learn: 0.8245106	total: 5m 46s	remaining: 35m 28s
7: learn: 0.8253496	total: 6m 35s	remaining: 34m 37s
8: learn: 0.8261911	total: 7m 25s	remaining: 33m 48s
9: learn: 0.8270143	total: 8m 16s	remaining: 33m 4s
10: learn: 0.8274099	total: 9m	remaining: 31m 57s
11: learn: 0.8274278	total: 9m 5s	remaining: 28m 48s
12: learn: 0.8281202	total: 9m 54s	remaining: 28m 11s
13: learn: 0.8286371	total: 10m 29s	remaining: 26m 57s
14: learn: 0.8288465	total: 10m 59s	remaining: 25m 38s
15: learn: 0.8294587	total: 12m	remaining: 25m 30s
16: learn: 0.8294576	total: 12m 2s	remaining: 23m 23s
17: learn: 0.8294573	total: 12m 5s	remaining: 21m 30s
18: learn: 0.829485	total: 12m 13s	remaining: 19m 56s
19: learn: 0.8298337	total: 12m 59s	remaining: 19m 29s
20: learn: 0.8304626	total: 13m 47s	remaining: 19m 2s
21: learn: 0.8309323	total: 14m 35s	remaining: 18m 33s
22: learn: 0.8313942	total: 15m 30s	remaining: 18m 12s
23: learn: 0.8317068	total: 16m 1s	remaining: 17m 21s
24: learn: 0.8324487	total: 16m 56s	remaining: 16m 56s
25: learn: 0.8328701	total: 17m 41s	remaining: 16m 20s
26: learn: 0.8330798	total: 18m 23s	remaining: 15m 40s
27: learn: 0.8334357	total: 19m 14s	remaining: 15m 6s
28: learn: 0.8335572	total: 20m 6s	remaining: 14m 33s
29: learn: 0.8338354	total: 20m 56s	remaining: 13m 57s
30: learn: 0.8341744	total: 21m 31s	remaining: 13m 11s
31: learn: 0.8342187	total: 22m 8s	remaining: 12m 27s
32: learn: 0.8343758	total: 22m 45s	remaining: 11m 43s
33: learn: 0.8347133	total: 23m 28s	remaining: 11m 3s
34: learn: 0.8353153	total: 24m 19s	remaining: 10m 25s
35: learn: 0.8354404	total: 24m 48s	remaining: 9m 38s
36: learn: 0.8358703	total: 25m 44s	remaining: 9m 2s
37: learn: 0.8359637	total: 26m 36s	remaining: 8m 24s
38: learn: 0.835964	total: 27m 16s	remaining: 7m 41s
39: learn: 0.8359641	total: 27m 34s	remaining: 6m 53s
40: learn: 0.8359643	total: 27m 53s	remaining: 6m 7s
41: learn: 0.836213	total: 28m 40s	remaining: 5m 27s
42: learn: 0.8366168	total: 29m 23s	remaining: 4m 47s
43: learn: 0.8369908	total: 30m 17s	remaining: 4m 7s
44: learn: 0.8369908	total: 30m 31s	remaining: 3m 23s
45: learn: 0.837197	total: 31m 14s	remaining: 2m 43s
46: learn: 0.837273	total: 31m 48s	remaining: 2m 1s
47: learn: 0.8375121	total: 32m 38s	remaining: 1m 21s
48: learn: 0.8377585	total: 33m 22s	remaining: 40.9s
49: learn: 0.8377585	total: 33m 24s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
0        1     0.473860     0.483480     0.414084     0.388706
3        1     0.510477     0.531194     0.480202     0.543768
6        1     0.755119     0.754216     0.747641     0.781373
9        1     0.917777     0.911412     0.908166     0.920171
12       0     0.785798     0.832348     0.868293     0.673296
# # # # # # # # # # 
0.335184422425
0.462493434975
0.0286856206714
0.120432696402
0.0822334100316
# # # # # # # # # # 

in model: CatR_top2_2  k-fold: 2 / 3

0: learn: 0.8031097	total: 45.7s	remaining: 37m 17s
1: learn: 0.811687	total: 1m 37s	remaining: 39m 4s
2: learn: 0.81625	total: 2m 34s	remaining: 40m 13s
3: learn: 0.8188966	total: 3m 21s	remaining: 38m 38s
4: learn: 0.8204931	total: 3m 58s	remaining: 35m 46s
5: learn: 0.821901	total: 4m 49s	remaining: 35m 19s
6: learn: 0.8233753	total: 5m 40s	remaining: 34m 53s
7: learn: 0.824217	total: 6m 18s	remaining: 33m 6s
8: learn: 0.825404	total: 6m 45s	remaining: 30m 48s
9: learn: 0.8264552	total: 7m 40s	remaining: 30m 42s
10: learn: 0.8268719	total: 8m 2s	remaining: 28m 30s
11: learn: 0.8275997	total: 8m 54s	remaining: 28m 13s
12: learn: 0.8281319	total: 9m 46s	remaining: 27m 49s
13: learn: 0.8286988	total: 10m 32s	remaining: 27m 7s
14: learn: 0.8288395	total: 10m 55s	remaining: 25m 28s
15: learn: 0.8288393	total: 10m 57s	remaining: 23m 18s
16: learn: 0.8288393	total: 11m 1s	remaining: 21m 24s
17: learn: 0.8288393	total: 11m 4s	remaining: 19m 41s
18: learn: 0.8288393	total: 11m 7s	remaining: 18m 9s
19: learn: 0.8288393	total: 11m 11s	remaining: 16m 47s
20: learn: 0.8288393	total: 11m 14s	remaining: 15m 30s
21: learn: 0.8288388	total: 11m 17s	remaining: 14m 22s
22: learn: 0.8292101	total: 11m 51s	remaining: 13m 55s
23: learn: 0.8299795	total: 12m 37s	remaining: 13m 40s
24: learn: 0.8304528	total: 13m 28s	remaining: 13m 28s
25: learn: 0.830988	total: 14m 12s	remaining: 13m 7s
26: learn: 0.8313991	total: 15m 10s	remaining: 12m 55s
27: learn: 0.8314006	total: 15m 45s	remaining: 12m 22s
28: learn: 0.8314009	total: 15m 53s	remaining: 11m 30s
29: learn: 0.8318792	total: 16m 43s	remaining: 11m 8s
30: learn: 0.8318793	total: 16m 50s	remaining: 10m 19s
31: learn: 0.8318793	total: 16m 56s	remaining: 9m 31s
32: learn: 0.8326253	total: 17m 47s	remaining: 9m 9s
33: learn: 0.8329271	total: 18m 27s	remaining: 8m 41s
34: learn: 0.8332313	total: 19m 4s	remaining: 8m 10s
35: learn: 0.833557	total: 19m 40s	remaining: 7m 39s
36: learn: 0.8336681	total: 20m 23s	remaining: 7m 9s
37: learn: 0.8337334	total: 21m 5s	remaining: 6m 39s
38: learn: 0.8337523	total: 21m 39s	remaining: 6m 6s
39: learn: 0.8341598	total: 22m 37s	remaining: 5m 39s
40: learn: 0.8343324	total: 23m 27s	remaining: 5m 8s
41: learn: 0.8343623	total: 24m 10s	remaining: 4m 36s
42: learn: 0.8346414	total: 24m 49s	remaining: 4m 2s
43: learn: 0.8348457	total: 25m 31s	remaining: 3m 28s
44: learn: 0.8349171	total: 26m 10s	remaining: 2m 54s
45: learn: 0.8349793	total: 27m 5s	remaining: 2m 21s
46: learn: 0.83498	total: 27m 42s	remaining: 1m 46s
47: learn: 0.8354504	total: 28m 27s	remaining: 1m 11s
48: learn: 0.8356948	total: 29m 8s	remaining: 35.7s
49: learn: 0.8356941	total: 29m 40s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
1        1     0.772383     0.772278     0.753011     0.807876
4        1     0.489014     0.501482     0.538394     0.625692
7        1     0.521924     0.514333     0.567273     0.648481
10       1     0.946141     0.922728     0.943774     0.912291
13       1     0.819401     0.835823     0.869749     0.830727
# # # # # # # # # # 
0.558207983621
0.763424288103
0.159308443545
0.176239214637
0.104106563343
# # # # # # # # # # 

in model: CatR_top2_2  k-fold: 3 / 3

0: learn: 0.8017045	total: 54.6s	remaining: 44m 36s
1: learn: 0.8112948	total: 1m 43s	remaining: 41m 30s
2: learn: 0.8170578	total: 2m 46s	remaining: 43m 32s
3: learn: 0.8204033	total: 3m 51s	remaining: 44m 17s
4: learn: 0.8229934	total: 4m 58s	remaining: 44m 46s
5: learn: 0.8239991	total: 5m 46s	remaining: 42m 24s
6: learn: 0.8249465	total: 6m 48s	remaining: 41m 46s
7: learn: 0.8261827	total: 7m 48s	remaining: 40m 58s
8: learn: 0.8272404	total: 8m 47s	remaining: 40m 5s
9: learn: 0.8274876	total: 9m 21s	remaining: 37m 25s
10: learn: 0.8275467	total: 9m 34s	remaining: 33m 56s
11: learn: 0.828188	total: 9m 58s	remaining: 31m 33s
12: learn: 0.8286696	total: 10m 42s	remaining: 30m 29s
13: learn: 0.8286687	total: 10m 47s	remaining: 27m 44s
14: learn: 0.8290592	total: 11m 50s	remaining: 27m 36s
15: learn: 0.829613	total: 12m 43s	remaining: 27m 2s
16: learn: 0.8296493	total: 12m 57s	remaining: 25m 8s
17: learn: 0.8302254	total: 14m 4s	remaining: 25m
18: learn: 0.8306245	total: 14m 52s	remaining: 24m 16s
19: learn: 0.830624	total: 14m 55s	remaining: 22m 23s
20: learn: 0.8306897	total: 15m 20s	remaining: 21m 11s
21: learn: 0.8312595	total: 16m 13s	remaining: 20m 39s
22: learn: 0.8315999	total: 17m 10s	remaining: 20m 10s
23: learn: 0.8323215	total: 18m 14s	remaining: 19m 46s
24: learn: 0.832682	total: 19m 3s	remaining: 19m 3s
25: learn: 0.832995	total: 19m 49s	remaining: 18m 18s
26: learn: 0.8332204	total: 20m 38s	remaining: 17m 34s
27: learn: 0.8334899	total: 21m 41s	remaining: 17m 2s
28: learn: 0.8334923	total: 22m 15s	remaining: 16m 7s
29: learn: 0.8337513	total: 23m	remaining: 15m 20s
30: learn: 0.8337599	total: 23m 41s	remaining: 14m 30s
31: learn: 0.8337842	total: 24m 16s	remaining: 13m 39s
32: learn: 0.8342215	total: 25m 14s	remaining: 12m 59s
33: learn: 0.8346038	total: 26m	remaining: 12m 14s
34: learn: 0.8347522	total: 26m 53s	remaining: 11m 31s
35: learn: 0.8348999	total: 27m 32s	remaining: 10m 42s
36: learn: 0.8350274	total: 28m 20s	remaining: 9m 57s
37: learn: 0.8351409	total: 29m 10s	remaining: 9m 12s
38: learn: 0.8352471	total: 29m 56s	remaining: 8m 26s
39: learn: 0.8353143	total: 30m 41s	remaining: 7m 40s
40: learn: 0.835368	total: 31m 30s	remaining: 6m 54s
41: learn: 0.8353686	total: 31m 49s	remaining: 6m 3s
42: learn: 0.8357762	total: 32m 31s	remaining: 5m 17s
43: learn: 0.8357763	total: 32m 41s	remaining: 4m 27s
44: learn: 0.8357875	total: 33m 23s	remaining: 3m 42s
45: learn: 0.8360454	total: 34m 13s	remaining: 2m 58s
46: learn: 0.8362057	total: 34m 48s	remaining: 2m 13s
47: learn: 0.8362551	total: 35m 30s	remaining: 1m 28s
48: learn: 0.8364101	total: 36m 6s	remaining: 44.2s
49: learn: 0.836428	total: 36m 50s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',
- - - - - - - - - - 
    target  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
2        1     0.745452     0.731489     0.711371     0.773749
5        1     0.731993     0.613060     0.697563     0.861547
8        1     0.892745     0.888683     0.964299     0.841809
11       1     0.937748     0.933345     0.935016     0.938189
14       1     0.962778     0.961574     0.981358     0.976964
# # # # # # # # # # 
0.887847920716
1.139986173
0.204168888892
0.25342768143
0.133472553783
# # # # # # # # # # 
  id  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
0  0     0.337589     0.344301     0.386213     0.295949
1  1     0.437973     0.492580     0.434502     0.379995
2  2     0.089281     0.131420     0.074751     0.068056
3  3     0.077397     0.064295     0.083317     0.084476
4  4     0.070322     0.066145     0.068353     0.044491
  id  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
0  0     0.337589     0.344301     0.386213     0.295949
1  1     0.437973     0.492580     0.434502     0.379995
2  2     0.089281     0.131420     0.074751     0.068056
3  3     0.077397     0.064295     0.083317     0.084476
4  4     0.070322     0.066145     0.068353     0.044491
              id  CatC_top2_1  CatR_top2_1  CatC_top2_2  CatR_top2_2
2556785  2556785     0.204839     0.197316     0.258217     0.220821
2556786  2556786     0.261589     0.244128     0.269934     0.332703
2556787  2556787     0.199951     0.211342     0.222797     0.134544
2556788  2556788     0.207564     0.195193     0.278250     0.066861
2556789  2556789     0.277306     0.242064     0.290152     0.159867
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
CatC_top2_1    float64
CatR_top2_1    float64
CatC_top2_2    float64
CatR_top2_2    float64
dtype: object
number of columns: 5
number of data: 2459140
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
CatC_top2_1    float64
CatR_top2_1    float64
CatC_top2_2    float64
CatR_top2_2    float64
dtype: object
number of columns: 5
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
CatC_top2_1    float64
CatR_top2_1    float64
CatC_top2_2    float64
CatR_top2_2    float64
dtype: object
number of columns: 5
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
id             category
CatC_top2_1     float64
CatR_top2_1     float64
CatC_top2_2     float64
CatR_top2_2     float64
dtype: object
number of columns: 5
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.

[timer]: complete in 276m 58s

Process finished with exit code 0
'''