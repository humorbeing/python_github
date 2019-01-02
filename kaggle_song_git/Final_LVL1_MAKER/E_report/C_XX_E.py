import sys
sys.path.insert(0, '../')
from me import *
from real_cat_XX import *
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
save_name = 'Cat_XX'
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


'''/home/vblab/untitled/bin/python /home/vblab/workplace/python/kagglebigdata/Final_LVL1_MAKER/C_XX.py
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

in model: CatC_XX_1  k-fold: 1 / 3

0: learn: 0.7507187	total: 7.47s	remaining: 18m 33s
1: learn: 0.7604845	total: 12.4s	remaining: 15m 19s
2: learn: 0.7629023	total: 16.7s	remaining: 13m 37s
3: learn: 0.7654786	total: 22.4s	remaining: 13m 37s
4: learn: 0.7668764	total: 27.7s	remaining: 13m 24s
5: learn: 0.7671732	total: 32.2s	remaining: 12m 52s
6: learn: 0.7709129	total: 37.9s	remaining: 12m 54s
7: learn: 0.7729521	total: 45.5s	remaining: 13m 27s
8: learn: 0.773769	total: 50.2s	remaining: 13m 7s
9: learn: 0.7740306	total: 55.4s	remaining: 12m 55s
10: learn: 0.7741603	total: 1m	remaining: 12m 38s
11: learn: 0.7745734	total: 1m 4s	remaining: 12m 27s
12: learn: 0.7745553	total: 1m 9s	remaining: 12m 8s
13: learn: 0.7756808	total: 1m 14s	remaining: 11m 59s
14: learn: 0.776343	total: 1m 19s	remaining: 11m 59s
15: learn: 0.7777359	total: 1m 25s	remaining: 11m 56s
16: learn: 0.7781018	total: 1m 30s	remaining: 11m 49s
17: learn: 0.7781018	total: 1m 33s	remaining: 11m 25s
18: learn: 0.7781018	total: 1m 36s	remaining: 11m 3s
19: learn: 0.7781018	total: 1m 38s	remaining: 10m 43s
20: learn: 0.7781018	total: 1m 41s	remaining: 10m 24s
21: learn: 0.7781018	total: 1m 44s	remaining: 10m 8s
22: learn: 0.7781018	total: 1m 47s	remaining: 9m 52s
23: learn: 0.7781021	total: 1m 50s	remaining: 9m 40s
24: learn: 0.7781021	total: 1m 53s	remaining: 9m 26s
25: learn: 0.7781021	total: 1m 56s	remaining: 9m 17s
26: learn: 0.7781021	total: 2m	remaining: 9m 6s
27: learn: 0.7784232	total: 2m 4s	remaining: 9m 3s
28: learn: 0.7796371	total: 2m 9s	remaining: 9m 2s
29: learn: 0.7796404	total: 2m 13s	remaining: 8m 52s
30: learn: 0.7796404	total: 2m 15s	remaining: 8m 41s
31: learn: 0.7803408	total: 2m 22s	remaining: 8m 44s
32: learn: 0.7807745	total: 2m 27s	remaining: 8m 42s
33: learn: 0.7812968	total: 2m 32s	remaining: 8m 40s
34: learn: 0.7818486	total: 2m 39s	remaining: 8m 45s
35: learn: 0.7822174	total: 2m 46s	remaining: 8m 48s
36: learn: 0.7823468	total: 2m 54s	remaining: 8m 52s
37: learn: 0.7824192	total: 3m	remaining: 8m 50s
38: learn: 0.7824217	total: 3m 3s	remaining: 8m 41s
39: learn: 0.782423	total: 3m 6s	remaining: 8m 31s
40: learn: 0.7826813	total: 3m 12s	remaining: 8m 31s
41: learn: 0.7826813	total: 3m 15s	remaining: 8m 21s
42: learn: 0.7826813	total: 3m 17s	remaining: 8m 12s
43: learn: 0.7826813	total: 3m 21s	remaining: 8m 4s
44: learn: 0.7828538	total: 3m 27s	remaining: 8m 3s
45: learn: 0.7828538	total: 3m 30s	remaining: 7m 55s
46: learn: 0.7828538	total: 3m 33s	remaining: 7m 47s
47: learn: 0.7828538	total: 3m 36s	remaining: 7m 39s
48: learn: 0.7828538	total: 3m 39s	remaining: 7m 32s
49: learn: 0.7828538	total: 3m 42s	remaining: 7m 24s
50: learn: 0.7828538	total: 3m 45s	remaining: 7m 17s
51: learn: 0.7828538	total: 3m 48s	remaining: 7m 11s
52: learn: 0.7828538	total: 3m 51s	remaining: 7m 4s
53: learn: 0.7833495	total: 3m 57s	remaining: 7m 3s
54: learn: 0.7833495	total: 4m	remaining: 6m 55s
55: learn: 0.7833495	total: 4m 3s	remaining: 6m 48s
56: learn: 0.7833495	total: 4m 6s	remaining: 6m 41s
57: learn: 0.7833495	total: 4m 8s	remaining: 6m 34s
58: learn: 0.7833495	total: 4m 11s	remaining: 6m 28s
59: learn: 0.7833495	total: 4m 15s	remaining: 6m 23s
60: learn: 0.7833495	total: 4m 18s	remaining: 6m 17s
61: learn: 0.7833495	total: 4m 21s	remaining: 6m 11s
62: learn: 0.7833495	total: 4m 24s	remaining: 6m 5s
63: learn: 0.7833495	total: 4m 27s	remaining: 5m 59s
64: learn: 0.7833495	total: 4m 29s	remaining: 5m 53s
65: learn: 0.7833495	total: 4m 32s	remaining: 5m 47s
66: learn: 0.7837312	total: 4m 39s	remaining: 5m 46s
67: learn: 0.7837312	total: 4m 42s	remaining: 5m 40s
68: learn: 0.7837312	total: 4m 46s	remaining: 5m 36s
69: learn: 0.7837312	total: 4m 49s	remaining: 5m 31s
70: learn: 0.7837312	total: 4m 53s	remaining: 5m 26s
71: learn: 0.7837312	total: 4m 56s	remaining: 5m 21s
72: learn: 0.7837312	total: 4m 59s	remaining: 5m 15s
73: learn: 0.7837312	total: 5m 2s	remaining: 5m 10s
74: learn: 0.7837312	total: 5m 5s	remaining: 5m 5s
75: learn: 0.7837312	total: 5m 8s	remaining: 5m
76: learn: 0.7837312	total: 5m 10s	remaining: 4m 54s
77: learn: 0.7838237	total: 5m 17s	remaining: 4m 52s
78: learn: 0.7839791	total: 5m 23s	remaining: 4m 50s
79: learn: 0.7840269	total: 5m 28s	remaining: 4m 47s
80: learn: 0.7842462	total: 5m 35s	remaining: 4m 45s
81: learn: 0.7846486	total: 5m 42s	remaining: 4m 43s
82: learn: 0.7850655	total: 5m 48s	remaining: 4m 41s
83: learn: 0.7851418	total: 5m 54s	remaining: 4m 38s
84: learn: 0.7852937	total: 5m 59s	remaining: 4m 35s
85: learn: 0.785375	total: 6m 4s	remaining: 4m 31s
86: learn: 0.7854562	total: 6m 10s	remaining: 4m 28s
87: learn: 0.7857549	total: 6m 17s	remaining: 4m 26s
88: learn: 0.786224	total: 6m 24s	remaining: 4m 23s
89: learn: 0.7862466	total: 6m 29s	remaining: 4m 19s
90: learn: 0.786447	total: 6m 35s	remaining: 4m 16s
91: learn: 0.7866847	total: 6m 42s	remaining: 4m 13s
92: learn: 0.786759	total: 6m 48s	remaining: 4m 10s
93: learn: 0.7869525	total: 6m 55s	remaining: 4m 7s
94: learn: 0.7871007	total: 7m 2s	remaining: 4m 4s
95: learn: 0.7871007	total: 7m 5s	remaining: 3m 59s
96: learn: 0.7871007	total: 7m 8s	remaining: 3m 54s
97: learn: 0.7871007	total: 7m 11s	remaining: 3m 48s
98: learn: 0.7871007	total: 7m 14s	remaining: 3m 43s
99: learn: 0.7871007	total: 7m 17s	remaining: 3m 38s
100: learn: 0.7871007	total: 7m 20s	remaining: 3m 33s
101: learn: 0.7871007	total: 7m 22s	remaining: 3m 28s
102: learn: 0.7871007	total: 7m 25s	remaining: 3m 23s
103: learn: 0.7871007	total: 7m 28s	remaining: 3m 18s
104: learn: 0.7871007	total: 7m 31s	remaining: 3m 13s
105: learn: 0.7871007	total: 7m 34s	remaining: 3m 8s
106: learn: 0.7871304	total: 7m 41s	remaining: 3m 5s
107: learn: 0.7871304	total: 7m 44s	remaining: 3m
108: learn: 0.7871304	total: 7m 47s	remaining: 2m 55s
109: learn: 0.7871304	total: 7m 50s	remaining: 2m 50s
110: learn: 0.7871304	total: 7m 52s	remaining: 2m 46s
111: learn: 0.7871304	total: 7m 56s	remaining: 2m 41s
112: learn: 0.7871304	total: 7m 59s	remaining: 2m 36s
113: learn: 0.787181	total: 8m 5s	remaining: 2m 33s
114: learn: 0.7872636	total: 8m 11s	remaining: 2m 29s
115: learn: 0.7873422	total: 8m 18s	remaining: 2m 26s
116: learn: 0.7873422	total: 8m 21s	remaining: 2m 21s
117: learn: 0.7873422	total: 8m 24s	remaining: 2m 16s
118: learn: 0.7873422	total: 8m 27s	remaining: 2m 12s
119: learn: 0.7873422	total: 8m 30s	remaining: 2m 7s
120: learn: 0.7873422	total: 8m 32s	remaining: 2m 2s
121: learn: 0.7873422	total: 8m 35s	remaining: 1m 58s
122: learn: 0.7873422	total: 8m 38s	remaining: 1m 53s
123: learn: 0.7873422	total: 8m 40s	remaining: 1m 49s
124: learn: 0.7873422	total: 8m 43s	remaining: 1m 44s
125: learn: 0.7873422	total: 8m 46s	remaining: 1m 40s
126: learn: 0.7873422	total: 8m 49s	remaining: 1m 35s
127: learn: 0.7876645	total: 8m 56s	remaining: 1m 32s
128: learn: 0.7878116	total: 9m 2s	remaining: 1m 28s
129: learn: 0.7878574	total: 9m 7s	remaining: 1m 24s
130: learn: 0.7878944	total: 9m 14s	remaining: 1m 20s
131: learn: 0.7880602	total: 9m 20s	remaining: 1m 16s
132: learn: 0.7881614	total: 9m 28s	remaining: 1m 12s
133: learn: 0.7882586	total: 9m 34s	remaining: 1m 8s
134: learn: 0.788281	total: 9m 40s	remaining: 1m 4s
135: learn: 0.7884447	total: 9m 47s	remaining: 1m
136: learn: 0.7884447	total: 9m 50s	remaining: 56s
137: learn: 0.7884447	total: 9m 52s	remaining: 51.5s
138: learn: 0.7884447	total: 9m 55s	remaining: 47.2s
139: learn: 0.7884447	total: 9m 58s	remaining: 42.8s
140: learn: 0.7884447	total: 10m 1s	remaining: 38.4s
141: learn: 0.7884447	total: 10m 4s	remaining: 34.1s
142: learn: 0.7884447	total: 10m 7s	remaining: 29.8s
143: learn: 0.7884447	total: 10m 10s	remaining: 25.4s
144: learn: 0.7884447	total: 10m 13s	remaining: 21.2s
145: learn: 0.7884447	total: 10m 16s	remaining: 16.9s
146: learn: 0.7885186	total: 10m 23s	remaining: 12.7s
147: learn: 0.7886459	total: 10m 30s	remaining: 8.52s
148: learn: 0.7886721	total: 10m 37s	remaining: 4.28s
149: learn: 0.7887228	total: 10m 44s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1
0        1   0.689373
3        1   0.608090
6        1   0.931559
9        1   0.916619
12       0   0.934667
# # # # # # # # # # 
0.362444826115
0.497574056748
0.152550219721
0.0445036238807
0.127338567016
# # # # # # # # # # 

in model: CatC_XX_1  k-fold: 2 / 3

0: learn: 0.7511082	total: 7.52s	remaining: 18m 40s
1: learn: 0.7597428	total: 13.4s	remaining: 16m 33s
2: learn: 0.7632343	total: 18s	remaining: 14m 39s
3: learn: 0.7656372	total: 22.5s	remaining: 13m 41s
4: learn: 0.7665585	total: 27.8s	remaining: 13m 26s
5: learn: 0.7669475	total: 32.3s	remaining: 12m 55s
6: learn: 0.7672808	total: 37.3s	remaining: 12m 42s
7: learn: 0.7696209	total: 44.2s	remaining: 13m 5s
8: learn: 0.7700924	total: 49.4s	remaining: 12m 54s
9: learn: 0.7729523	total: 55.7s	remaining: 12m 59s
10: learn: 0.7735924	total: 1m	remaining: 12m 44s
11: learn: 0.7756077	total: 1m 5s	remaining: 12m 37s
12: learn: 0.776445	total: 1m 10s	remaining: 12m 26s
13: learn: 0.7772184	total: 1m 16s	remaining: 12m 27s
14: learn: 0.7773253	total: 1m 21s	remaining: 12m 17s
15: learn: 0.7773253	total: 1m 24s	remaining: 11m 49s
16: learn: 0.7774786	total: 1m 29s	remaining: 11m 44s
17: learn: 0.7774786	total: 1m 32s	remaining: 11m 20s
18: learn: 0.7789292	total: 1m 38s	remaining: 11m 19s
19: learn: 0.7789292	total: 1m 41s	remaining: 10m 58s
20: learn: 0.7789292	total: 1m 44s	remaining: 10m 39s
21: learn: 0.7789292	total: 1m 46s	remaining: 10m 22s
22: learn: 0.7789292	total: 1m 49s	remaining: 10m 7s
23: learn: 0.7789292	total: 1m 52s	remaining: 9m 52s
24: learn: 0.7789292	total: 1m 55s	remaining: 9m 39s
25: learn: 0.7789292	total: 1m 59s	remaining: 9m 27s
26: learn: 0.7789293	total: 2m 2s	remaining: 9m 16s
27: learn: 0.7795093	total: 2m 7s	remaining: 9m 14s
28: learn: 0.7802898	total: 2m 14s	remaining: 9m 21s
29: learn: 0.781089	total: 2m 22s	remaining: 9m 28s
30: learn: 0.7811712	total: 2m 27s	remaining: 9m 24s
31: learn: 0.7814618	total: 2m 32s	remaining: 9m 22s
32: learn: 0.7816444	total: 2m 38s	remaining: 9m 21s
33: learn: 0.7819263	total: 2m 44s	remaining: 9m 20s
34: learn: 0.7819263	total: 2m 46s	remaining: 9m 8s
35: learn: 0.7819263	total: 2m 49s	remaining: 8m 57s
36: learn: 0.7819263	total: 2m 52s	remaining: 8m 46s
37: learn: 0.7819263	total: 2m 55s	remaining: 8m 36s
38: learn: 0.7819263	total: 2m 58s	remaining: 8m 26s
39: learn: 0.7819263	total: 3m 1s	remaining: 8m 18s
40: learn: 0.7819263	total: 3m 4s	remaining: 8m 9s
41: learn: 0.7819263	total: 3m 7s	remaining: 8m 1s
42: learn: 0.7819263	total: 3m 9s	remaining: 7m 52s
43: learn: 0.7819281	total: 3m 12s	remaining: 7m 44s
44: learn: 0.7823392	total: 3m 19s	remaining: 7m 46s
45: learn: 0.7824701	total: 3m 25s	remaining: 7m 45s
46: learn: 0.7829132	total: 3m 32s	remaining: 7m 45s
47: learn: 0.7829132	total: 3m 34s	remaining: 7m 36s
48: learn: 0.7829132	total: 3m 37s	remaining: 7m 28s
49: learn: 0.7829132	total: 3m 40s	remaining: 7m 20s
50: learn: 0.7829132	total: 3m 43s	remaining: 7m 13s
51: learn: 0.7829132	total: 3m 45s	remaining: 7m 5s
52: learn: 0.7829132	total: 3m 48s	remaining: 6m 58s
53: learn: 0.7829132	total: 3m 51s	remaining: 6m 51s
54: learn: 0.7829132	total: 3m 54s	remaining: 6m 45s
55: learn: 0.7829132	total: 3m 57s	remaining: 6m 38s
56: learn: 0.7829132	total: 4m	remaining: 6m 32s
57: learn: 0.7829132	total: 4m 3s	remaining: 6m 26s
58: learn: 0.7829132	total: 4m 6s	remaining: 6m 20s
59: learn: 0.7829139	total: 4m 10s	remaining: 6m 15s
60: learn: 0.7829142	total: 4m 13s	remaining: 6m 10s
61: learn: 0.7829144	total: 4m 17s	remaining: 6m 4s
62: learn: 0.7831386	total: 4m 24s	remaining: 6m 5s
63: learn: 0.7835661	total: 4m 31s	remaining: 6m 5s
64: learn: 0.7835661	total: 4m 34s	remaining: 5m 58s
65: learn: 0.7835661	total: 4m 37s	remaining: 5m 52s
66: learn: 0.7835661	total: 4m 40s	remaining: 5m 47s
67: learn: 0.7835661	total: 4m 43s	remaining: 5m 41s
68: learn: 0.7835661	total: 4m 45s	remaining: 5m 35s
69: learn: 0.7835661	total: 4m 48s	remaining: 5m 29s
70: learn: 0.7835661	total: 4m 51s	remaining: 5m 24s
71: learn: 0.7835661	total: 4m 54s	remaining: 5m 18s
72: learn: 0.7835661	total: 4m 56s	remaining: 5m 13s
73: learn: 0.7835661	total: 4m 59s	remaining: 5m 7s
74: learn: 0.7835661	total: 5m 2s	remaining: 5m 2s
75: learn: 0.7835661	total: 5m 5s	remaining: 4m 57s
76: learn: 0.7835661	total: 5m 8s	remaining: 4m 52s
77: learn: 0.7835661	total: 5m 10s	remaining: 4m 46s
78: learn: 0.7838092	total: 5m 17s	remaining: 4m 45s
79: learn: 0.7839902	total: 5m 25s	remaining: 4m 44s
80: learn: 0.784216	total: 5m 31s	remaining: 4m 42s
81: learn: 0.7844065	total: 5m 37s	remaining: 4m 39s
82: learn: 0.7844678	total: 5m 44s	remaining: 4m 38s
83: learn: 0.7845487	total: 5m 51s	remaining: 4m 35s
84: learn: 0.7850411	total: 5m 57s	remaining: 4m 33s
85: learn: 0.7853092	total: 6m 4s	remaining: 4m 31s
86: learn: 0.7853926	total: 6m 9s	remaining: 4m 27s
87: learn: 0.7854048	total: 6m 15s	remaining: 4m 24s
88: learn: 0.7856744	total: 6m 22s	remaining: 4m 21s
89: learn: 0.7857638	total: 6m 28s	remaining: 4m 19s
90: learn: 0.7858246	total: 6m 35s	remaining: 4m 16s
91: learn: 0.785977	total: 6m 42s	remaining: 4m 13s
92: learn: 0.7862663	total: 6m 49s	remaining: 4m 10s
93: learn: 0.7864449	total: 6m 55s	remaining: 4m 7s
94: learn: 0.7864978	total: 7m	remaining: 4m 3s
95: learn: 0.7867685	total: 7m 8s	remaining: 4m
96: learn: 0.7868355	total: 7m 13s	remaining: 3m 56s
97: learn: 0.7868702	total: 7m 17s	remaining: 3m 52s
98: learn: 0.7869126	total: 7m 23s	remaining: 3m 48s
99: learn: 0.7869979	total: 7m 28s	remaining: 3m 44s
100: learn: 0.78702	total: 7m 33s	remaining: 3m 40s
101: learn: 0.7871063	total: 7m 40s	remaining: 3m 36s
102: learn: 0.787181	total: 7m 47s	remaining: 3m 33s
103: learn: 0.7872406	total: 7m 53s	remaining: 3m 29s
104: learn: 0.7873924	total: 8m	remaining: 3m 25s
105: learn: 0.7874671	total: 8m 7s	remaining: 3m 22s
106: learn: 0.787537	total: 8m 15s	remaining: 3m 18s
107: learn: 0.7876691	total: 8m 22s	remaining: 3m 15s
108: learn: 0.7877114	total: 8m 29s	remaining: 3m 11s
109: learn: 0.7878241	total: 8m 36s	remaining: 3m 7s
110: learn: 0.7878996	total: 8m 43s	remaining: 3m 3s
111: learn: 0.7879386	total: 8m 50s	remaining: 3m
112: learn: 0.7880312	total: 8m 57s	remaining: 2m 55s
113: learn: 0.7881328	total: 9m 4s	remaining: 2m 52s
114: learn: 0.7881484	total: 9m 11s	remaining: 2m 47s
115: learn: 0.7881773	total: 9m 18s	remaining: 2m 43s
116: learn: 0.7882634	total: 9m 26s	remaining: 2m 39s
117: learn: 0.7883225	total: 9m 34s	remaining: 2m 35s
118: learn: 0.7883688	total: 9m 40s	remaining: 2m 31s
119: learn: 0.7883978	total: 9m 49s	remaining: 2m 27s
120: learn: 0.7884379	total: 9m 56s	remaining: 2m 23s
121: learn: 0.7885982	total: 10m 4s	remaining: 2m 18s
122: learn: 0.7885982	total: 10m 7s	remaining: 2m 13s
123: learn: 0.7885982	total: 10m 10s	remaining: 2m 8s
124: learn: 0.7885982	total: 10m 13s	remaining: 2m 2s
125: learn: 0.7885982	total: 10m 16s	remaining: 1m 57s
126: learn: 0.7885982	total: 10m 19s	remaining: 1m 52s
127: learn: 0.7886022	total: 10m 26s	remaining: 1m 47s
128: learn: 0.7886238	total: 10m 32s	remaining: 1m 42s
129: learn: 0.7886373	total: 10m 38s	remaining: 1m 38s
130: learn: 0.7887549	total: 10m 45s	remaining: 1m 33s
131: learn: 0.7888629	total: 10m 53s	remaining: 1m 29s
132: learn: 0.7889493	total: 11m 1s	remaining: 1m 24s
133: learn: 0.7889962	total: 11m 8s	remaining: 1m 19s
134: learn: 0.7890069	total: 11m 16s	remaining: 1m 15s
135: learn: 0.7891449	total: 11m 24s	remaining: 1m 10s
136: learn: 0.7891449	total: 11m 27s	remaining: 1m 5s
137: learn: 0.789145	total: 11m 31s	remaining: 1m
138: learn: 0.789145	total: 11m 34s	remaining: 54.9s
139: learn: 0.789145	total: 11m 37s	remaining: 49.8s
140: learn: 0.789145	total: 11m 40s	remaining: 44.7s
141: learn: 0.789145	total: 11m 43s	remaining: 39.7s
142: learn: 0.7891473	total: 11m 50s	remaining: 34.8s
143: learn: 0.7892788	total: 11m 57s	remaining: 29.9s
144: learn: 0.7892788	total: 12m	remaining: 24.9s
145: learn: 0.7892788	total: 12m 3s	remaining: 19.8s
146: learn: 0.7892788	total: 12m 6s	remaining: 14.8s
147: learn: 0.7892788	total: 12m 9s	remaining: 9.86s
148: learn: 0.7892788	total: 12m 12s	remaining: 4.92s
149: learn: 0.7892788	total: 12m 15s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1
1        1   0.767281
4        1   0.563007
7        1   0.626437
10       1   0.812196
13       1   0.802438
# # # # # # # # # # 
0.704994749289
0.925188573934
0.295064519056
0.0782924020908
0.198817455786
# # # # # # # # # # 

in model: CatC_XX_1  k-fold: 3 / 3

0: learn: 0.751309	total: 8.12s	remaining: 20m 10s
1: learn: 0.7598838	total: 14.6s	remaining: 18m
2: learn: 0.7629138	total: 20.5s	remaining: 16m 43s
3: learn: 0.7655272	total: 26s	remaining: 15m 47s
4: learn: 0.7658614	total: 30.9s	remaining: 14m 56s
5: learn: 0.766238	total: 35.7s	remaining: 14m 17s
6: learn: 0.7670473	total: 40.9s	remaining: 13m 56s
7: learn: 0.7674059	total: 46.3s	remaining: 13m 41s
8: learn: 0.7708141	total: 53.3s	remaining: 13m 54s
9: learn: 0.7728214	total: 1m 1s	remaining: 14m 16s
10: learn: 0.7739216	total: 1m 6s	remaining: 14m
11: learn: 0.7743607	total: 1m 11s	remaining: 13m 44s
12: learn: 0.7759029	total: 1m 17s	remaining: 13m 39s
13: learn: 0.7767922	total: 1m 24s	remaining: 13m 42s
14: learn: 0.7773586	total: 1m 29s	remaining: 13m 29s
15: learn: 0.7773586	total: 1m 33s	remaining: 13m
16: learn: 0.7776791	total: 1m 39s	remaining: 12m 54s
17: learn: 0.7776791	total: 1m 42s	remaining: 12m 28s
18: learn: 0.7776791	total: 1m 45s	remaining: 12m 3s
19: learn: 0.7776791	total: 1m 47s	remaining: 11m 41s
20: learn: 0.7776791	total: 1m 50s	remaining: 11m 21s
21: learn: 0.7776791	total: 1m 53s	remaining: 11m 2s
22: learn: 0.7776791	total: 1m 56s	remaining: 10m 45s
23: learn: 0.7776791	total: 1m 59s	remaining: 10m 29s
24: learn: 0.7776791	total: 2m 3s	remaining: 10m 16s
25: learn: 0.7776791	total: 2m 6s	remaining: 10m 1s
26: learn: 0.7776791	total: 2m 9s	remaining: 9m 48s
27: learn: 0.7776791	total: 2m 12s	remaining: 9m 36s
28: learn: 0.777885	total: 2m 17s	remaining: 9m 34s
29: learn: 0.777884	total: 2m 21s	remaining: 9m 24s
30: learn: 0.7782444	total: 2m 26s	remaining: 9m 20s
31: learn: 0.7782444	total: 2m 29s	remaining: 9m 9s
32: learn: 0.7782444	total: 2m 32s	remaining: 8m 59s
33: learn: 0.7782444	total: 2m 35s	remaining: 8m 48s
34: learn: 0.7782499	total: 2m 39s	remaining: 8m 43s
35: learn: 0.7782499	total: 2m 42s	remaining: 8m 33s
36: learn: 0.7782499	total: 2m 45s	remaining: 8m 24s
37: learn: 0.7782499	total: 2m 48s	remaining: 8m 15s
38: learn: 0.7782977	total: 2m 53s	remaining: 8m 14s
39: learn: 0.7782987	total: 2m 57s	remaining: 8m 6s
40: learn: 0.779588	total: 3m 4s	remaining: 8m 10s
41: learn: 0.7800977	total: 3m 12s	remaining: 8m 14s
42: learn: 0.780803	total: 3m 19s	remaining: 8m 17s
43: learn: 0.7813395	total: 3m 27s	remaining: 8m 19s
44: learn: 0.7815016	total: 3m 35s	remaining: 8m 21s
45: learn: 0.7815476	total: 3m 41s	remaining: 8m 20s
46: learn: 0.781893	total: 3m 47s	remaining: 8m 18s
47: learn: 0.7825864	total: 3m 53s	remaining: 8m 16s
48: learn: 0.7829568	total: 4m	remaining: 8m 15s
49: learn: 0.7834628	total: 4m 6s	remaining: 8m 13s
50: learn: 0.7838359	total: 4m 14s	remaining: 8m 13s
51: learn: 0.7838359	total: 4m 17s	remaining: 8m 4s
52: learn: 0.7838359	total: 4m 19s	remaining: 7m 55s
53: learn: 0.7838359	total: 4m 22s	remaining: 7m 47s
54: learn: 0.7838359	total: 4m 25s	remaining: 7m 39s
55: learn: 0.7838359	total: 4m 28s	remaining: 7m 31s
56: learn: 0.7838359	total: 4m 32s	remaining: 7m 24s
57: learn: 0.7838359	total: 4m 35s	remaining: 7m 16s
58: learn: 0.7838359	total: 4m 38s	remaining: 7m 9s
59: learn: 0.7838359	total: 4m 41s	remaining: 7m 1s
60: learn: 0.7838359	total: 4m 44s	remaining: 6m 54s
61: learn: 0.7838359	total: 4m 47s	remaining: 6m 47s
62: learn: 0.7838359	total: 4m 50s	remaining: 6m 40s
63: learn: 0.7838359	total: 4m 52s	remaining: 6m 33s
64: learn: 0.7838359	total: 4m 55s	remaining: 6m 27s
65: learn: 0.7838954	total: 5m 2s	remaining: 6m 25s
66: learn: 0.7841128	total: 5m 10s	remaining: 6m 24s
67: learn: 0.7841128	total: 5m 13s	remaining: 6m 17s
68: learn: 0.7841128	total: 5m 16s	remaining: 6m 11s
69: learn: 0.7841128	total: 5m 19s	remaining: 6m 5s
70: learn: 0.7841128	total: 5m 22s	remaining: 5m 59s
71: learn: 0.7841128	total: 5m 26s	remaining: 5m 53s
72: learn: 0.7841128	total: 5m 28s	remaining: 5m 47s
73: learn: 0.7841128	total: 5m 31s	remaining: 5m 40s
74: learn: 0.7843447	total: 5m 37s	remaining: 5m 37s
75: learn: 0.7843447	total: 5m 39s	remaining: 5m 30s
76: learn: 0.7843447	total: 5m 42s	remaining: 5m 24s
77: learn: 0.7843447	total: 5m 45s	remaining: 5m 18s
78: learn: 0.7843447	total: 5m 48s	remaining: 5m 13s
79: learn: 0.7846628	total: 5m 55s	remaining: 5m 10s
80: learn: 0.7848669	total: 6m 2s	remaining: 5m 8s
81: learn: 0.7849342	total: 6m 9s	remaining: 5m 6s
82: learn: 0.7850361	total: 6m 14s	remaining: 5m 2s
83: learn: 0.7851338	total: 6m 21s	remaining: 4m 59s
84: learn: 0.7851338	total: 6m 24s	remaining: 4m 53s
85: learn: 0.7851677	total: 6m 29s	remaining: 4m 50s
86: learn: 0.785651	total: 6m 36s	remaining: 4m 47s
87: learn: 0.785651	total: 6m 39s	remaining: 4m 41s
88: learn: 0.785651	total: 6m 42s	remaining: 4m 35s
89: learn: 0.785651	total: 6m 44s	remaining: 4m 29s
90: learn: 0.785651	total: 6m 47s	remaining: 4m 24s
91: learn: 0.785651	total: 6m 50s	remaining: 4m 18s
92: learn: 0.785651	total: 6m 53s	remaining: 4m 13s
93: learn: 0.785651	total: 6m 55s	remaining: 4m 7s
94: learn: 0.785651	total: 6m 58s	remaining: 4m 2s
95: learn: 0.785651	total: 7m 1s	remaining: 3m 57s
96: learn: 0.7856511	total: 7m 4s	remaining: 3m 51s
97: learn: 0.7856511	total: 7m 7s	remaining: 3m 46s
98: learn: 0.7856511	total: 7m 10s	remaining: 3m 41s
99: learn: 0.7857323	total: 7m 16s	remaining: 3m 38s
100: learn: 0.7857323	total: 7m 18s	remaining: 3m 32s
101: learn: 0.7857323	total: 7m 21s	remaining: 3m 27s
102: learn: 0.7857323	total: 7m 24s	remaining: 3m 22s
103: learn: 0.7857323	total: 7m 27s	remaining: 3m 17s
104: learn: 0.7857323	total: 7m 29s	remaining: 3m 12s
105: learn: 0.7857323	total: 7m 32s	remaining: 3m 8s
106: learn: 0.7857323	total: 7m 35s	remaining: 3m 3s
107: learn: 0.7859569	total: 7m 42s	remaining: 2m 59s
108: learn: 0.7859977	total: 7m 48s	remaining: 2m 56s
109: learn: 0.7861943	total: 7m 53s	remaining: 2m 52s
110: learn: 0.7863972	total: 8m	remaining: 2m 48s
111: learn: 0.7863972	total: 8m 2s	remaining: 2m 43s
112: learn: 0.7864627	total: 8m 9s	remaining: 2m 40s
113: learn: 0.7866781	total: 8m 16s	remaining: 2m 36s
114: learn: 0.7869032	total: 8m 23s	remaining: 2m 33s
115: learn: 0.7869032	total: 8m 26s	remaining: 2m 28s
116: learn: 0.7869032	total: 8m 29s	remaining: 2m 23s
117: learn: 0.7869032	total: 8m 31s	remaining: 2m 18s
118: learn: 0.7869032	total: 8m 34s	remaining: 2m 14s
119: learn: 0.7870136	total: 8m 41s	remaining: 2m 10s
120: learn: 0.7871699	total: 8m 47s	remaining: 2m 6s
121: learn: 0.7873834	total: 8m 54s	remaining: 2m 2s
122: learn: 0.7874029	total: 8m 59s	remaining: 1m 58s
123: learn: 0.7874994	total: 9m 6s	remaining: 1m 54s
124: learn: 0.7874994	total: 9m 9s	remaining: 1m 49s
125: learn: 0.7874994	total: 9m 12s	remaining: 1m 45s
126: learn: 0.7874994	total: 9m 14s	remaining: 1m 40s
127: learn: 0.7875805	total: 9m 20s	remaining: 1m 36s
128: learn: 0.7875805	total: 9m 23s	remaining: 1m 31s
129: learn: 0.7875805	total: 9m 26s	remaining: 1m 27s
130: learn: 0.7875805	total: 9m 28s	remaining: 1m 22s
131: learn: 0.7875805	total: 9m 31s	remaining: 1m 17s
132: learn: 0.7875805	total: 9m 34s	remaining: 1m 13s
133: learn: 0.7875805	total: 9m 37s	remaining: 1m 8s
134: learn: 0.7875805	total: 9m 39s	remaining: 1m 4s
135: learn: 0.7875805	total: 9m 42s	remaining: 60s
136: learn: 0.787659	total: 9m 48s	remaining: 55.9s
137: learn: 0.7876606	total: 9m 53s	remaining: 51.6s
138: learn: 0.7877058	total: 10m 1s	remaining: 47.6s
139: learn: 0.7877518	total: 10m 8s	remaining: 43.4s
140: learn: 0.7877526	total: 10m 13s	remaining: 39.2s
141: learn: 0.7878174	total: 10m 19s	remaining: 34.9s
142: learn: 0.7878533	total: 10m 25s	remaining: 30.6s
143: learn: 0.7878877	total: 10m 32s	remaining: 26.3s
144: learn: 0.7879969	total: 10m 38s	remaining: 22s
145: learn: 0.7880462	total: 10m 44s	remaining: 17.7s
146: learn: 0.7880462	total: 10m 48s	remaining: 13.2s
147: learn: 0.7880462	total: 10m 50s	remaining: 8.79s
148: learn: 0.7880462	total: 10m 53s	remaining: 4.38s
149: learn: 0.7880462	total: 10m 56s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1
2        1   0.844959
5        1   0.822139
8        1   0.911415
11       1   0.814345
14       1   0.891612
# # # # # # # # # # 
0.997018716586
1.3122431256
0.414450656018
0.115131149029
0.289416788829
# # # # # # # # # # 
  id  CatC_XX_1
0  0   0.332340
1  1   0.437414
2  2   0.138150
3  3   0.038377
4  4   0.096472

in model: CatR_XX_1  k-fold: 1 / 3

0: learn: 0.7636618	total: 10.5s	remaining: 19m 5s
1: learn: 0.7699975	total: 17.9s	remaining: 16m 9s
2: learn: 0.7726168	total: 24.5s	remaining: 14m 34s
3: learn: 0.7745021	total: 31.6s	remaining: 13m 57s
4: learn: 0.7747345	total: 38.3s	remaining: 13m 23s
5: learn: 0.7744249	total: 45s	remaining: 13m
6: learn: 0.7752125	total: 51.8s	remaining: 12m 42s
7: learn: 0.7757101	total: 58.5s	remaining: 12m 25s
8: learn: 0.7758982	total: 1m 5s	remaining: 12m 20s
9: learn: 0.7763951	total: 1m 12s	remaining: 12m 4s
10: learn: 0.776636	total: 1m 19s	remaining: 11m 51s
11: learn: 0.7769167	total: 1m 25s	remaining: 11m 35s
12: learn: 0.7771756	total: 1m 31s	remaining: 11m 21s
13: learn: 0.7772655	total: 1m 37s	remaining: 11m 8s
14: learn: 0.7773573	total: 1m 43s	remaining: 10m 54s
15: learn: 0.7776741	total: 1m 49s	remaining: 10m 42s
16: learn: 0.777869	total: 1m 55s	remaining: 10m 33s
17: learn: 0.7777916	total: 2m 1s	remaining: 10m 21s
18: learn: 0.7779883	total: 2m 7s	remaining: 10m 11s
19: learn: 0.7781593	total: 2m 13s	remaining: 10m 1s
20: learn: 0.7783259	total: 2m 18s	remaining: 9m 48s
21: learn: 0.7784406	total: 2m 24s	remaining: 9m 39s
22: learn: 0.778718	total: 2m 31s	remaining: 9m 34s
23: learn: 0.7788017	total: 2m 37s	remaining: 9m 24s
24: learn: 0.7789118	total: 2m 43s	remaining: 9m 14s
25: learn: 0.7790356	total: 2m 48s	remaining: 9m 4s
26: learn: 0.7793104	total: 2m 54s	remaining: 8m 55s
27: learn: 0.7793749	total: 2m 59s	remaining: 8m 46s
28: learn: 0.7796179	total: 3m 7s	remaining: 8m 42s
29: learn: 0.7797307	total: 3m 12s	remaining: 8m 33s
30: learn: 0.7798071	total: 3m 18s	remaining: 8m 25s
31: learn: 0.7799338	total: 3m 23s	remaining: 8m 16s
32: learn: 0.7799864	total: 3m 29s	remaining: 8m 7s
33: learn: 0.7801759	total: 3m 34s	remaining: 7m 59s
34: learn: 0.7802587	total: 3m 40s	remaining: 7m 51s
35: learn: 0.7803222	total: 3m 46s	remaining: 7m 45s
36: learn: 0.780443	total: 3m 52s	remaining: 7m 38s
37: learn: 0.7804772	total: 3m 57s	remaining: 7m 30s
38: learn: 0.7805985	total: 4m 3s	remaining: 7m 22s
39: learn: 0.7806491	total: 4m 9s	remaining: 7m 15s
40: learn: 0.7806716	total: 4m 14s	remaining: 7m 8s
41: learn: 0.7807548	total: 4m 20s	remaining: 7m 1s
42: learn: 0.7809095	total: 4m 25s	remaining: 6m 54s
43: learn: 0.7811096	total: 4m 32s	remaining: 6m 48s
44: learn: 0.7812673	total: 4m 38s	remaining: 6m 41s
45: learn: 0.7813922	total: 4m 43s	remaining: 6m 34s
46: learn: 0.781535	total: 4m 49s	remaining: 6m 27s
47: learn: 0.781598	total: 4m 54s	remaining: 6m 20s
48: learn: 0.7816722	total: 5m	remaining: 6m 13s
49: learn: 0.7817469	total: 5m 5s	remaining: 6m 6s
50: learn: 0.7818407	total: 5m 12s	remaining: 6m 1s
51: learn: 0.7818991	total: 5m 17s	remaining: 5m 54s
52: learn: 0.7819838	total: 5m 23s	remaining: 5m 47s
53: learn: 0.7820245	total: 5m 28s	remaining: 5m 40s
54: learn: 0.782075	total: 5m 34s	remaining: 5m 34s
55: learn: 0.7821695	total: 5m 40s	remaining: 5m 27s
56: learn: 0.7822089	total: 5m 45s	remaining: 5m 21s
57: learn: 0.7822485	total: 5m 51s	remaining: 5m 15s
58: learn: 0.7822892	total: 5m 57s	remaining: 5m 8s
59: learn: 0.7823594	total: 6m 2s	remaining: 5m 2s
60: learn: 0.7824095	total: 6m 8s	remaining: 4m 55s
61: learn: 0.7825988	total: 6m 14s	remaining: 4m 49s
62: learn: 0.7826711	total: 6m 20s	remaining: 4m 43s
63: learn: 0.7829222	total: 6m 28s	remaining: 4m 39s
64: learn: 0.7830894	total: 6m 37s	remaining: 4m 35s
65: learn: 0.7832556	total: 6m 43s	remaining: 4m 29s
66: learn: 0.783475	total: 6m 51s	remaining: 4m 24s
67: learn: 0.7835814	total: 6m 57s	remaining: 4m 18s
68: learn: 0.7836501	total: 7m 6s	remaining: 4m 13s
69: learn: 0.7837383	total: 7m 14s	remaining: 4m 8s
70: learn: 0.7839476	total: 7m 25s	remaining: 4m 4s
71: learn: 0.7841143	total: 7m 32s	remaining: 3m 59s
72: learn: 0.784299	total: 7m 42s	remaining: 3m 54s
73: learn: 0.7843737	total: 7m 51s	remaining: 3m 49s
74: learn: 0.7845675	total: 8m	remaining: 3m 44s
75: learn: 0.7847014	total: 8m 9s	remaining: 3m 39s
76: learn: 0.7848923	total: 8m 19s	remaining: 3m 34s
77: learn: 0.7849907	total: 8m 28s	remaining: 3m 28s
78: learn: 0.7851397	total: 8m 39s	remaining: 3m 23s
79: learn: 0.7852887	total: 8m 47s	remaining: 3m 17s
80: learn: 0.7854131	total: 8m 54s	remaining: 3m 11s
81: learn: 0.7855217	total: 9m 4s	remaining: 3m 5s
82: learn: 0.7856126	total: 9m 13s	remaining: 2m 59s
83: learn: 0.7857245	total: 9m 22s	remaining: 2m 54s
84: learn: 0.7858434	total: 9m 31s	remaining: 2m 48s
85: learn: 0.7859312	total: 9m 40s	remaining: 2m 42s
86: learn: 0.7860012	total: 9m 49s	remaining: 2m 35s
87: learn: 0.7861284	total: 9m 58s	remaining: 2m 29s
88: learn: 0.7862107	total: 10m 6s	remaining: 2m 23s
89: learn: 0.7863081	total: 10m 13s	remaining: 2m 16s
90: learn: 0.7864215	total: 10m 22s	remaining: 2m 9s
91: learn: 0.7865561	total: 10m 31s	remaining: 2m 3s
92: learn: 0.7866361	total: 10m 41s	remaining: 1m 57s
93: learn: 0.7867282	total: 10m 50s	remaining: 1m 50s
94: learn: 0.7868395	total: 10m 59s	remaining: 1m 44s
95: learn: 0.7869137	total: 11m 6s	remaining: 1m 37s
96: learn: 0.7870144	total: 11m 15s	remaining: 1m 30s
97: learn: 0.7871451	total: 11m 23s	remaining: 1m 23s
98: learn: 0.7872297	total: 11m 31s	remaining: 1m 16s
99: learn: 0.7872845	total: 11m 39s	remaining: 1m 9s
100: learn: 0.7873381	total: 11m 48s	remaining: 1m 3s
101: learn: 0.7874335	total: 11m 56s	remaining: 56.2s
102: learn: 0.7874972	total: 12m 5s	remaining: 49.3s
103: learn: 0.7875977	total: 12m 15s	remaining: 42.4s
104: learn: 0.7876514	total: 12m 23s	remaining: 35.4s
105: learn: 0.7877241	total: 12m 31s	remaining: 28.4s
106: learn: 0.7878129	total: 12m 41s	remaining: 21.3s
107: learn: 0.787893	total: 12m 47s	remaining: 14.2s
108: learn: 0.7879373	total: 12m 57s	remaining: 7.13s
109: learn: 0.7880375	total: 13m 5s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1
0        1   0.689373   0.553690
3        1   0.608090   0.571467
6        1   0.931559   0.914058
9        1   0.916619   0.916289
12       0   0.934667   0.938171
# # # # # # # # # # 
0.363341283731
0.520424654989
0.138824589053
0.0399507603292
0.119995654533
# # # # # # # # # # 

in model: CatR_XX_1  k-fold: 2 / 3

0: learn: 0.7607246	total: 10.2s	remaining: 18m 34s
1: learn: 0.7652334	total: 18.1s	remaining: 16m 17s
2: learn: 0.7679992	total: 24.2s	remaining: 14m 24s
3: learn: 0.7703644	total: 30.4s	remaining: 13m 24s
4: learn: 0.7729693	total: 37s	remaining: 12m 56s
5: learn: 0.7736621	total: 43.1s	remaining: 12m 27s
6: learn: 0.7743165	total: 50.9s	remaining: 12m 28s
7: learn: 0.7745791	total: 57.1s	remaining: 12m 8s
8: learn: 0.7753689	total: 1m 3s	remaining: 11m 55s
9: learn: 0.7755546	total: 1m 10s	remaining: 11m 41s
10: learn: 0.7760425	total: 1m 16s	remaining: 11m 29s
11: learn: 0.7762562	total: 1m 22s	remaining: 11m 16s
12: learn: 0.7765478	total: 1m 29s	remaining: 11m 9s
13: learn: 0.7767603	total: 1m 35s	remaining: 10m 56s
14: learn: 0.7771802	total: 1m 42s	remaining: 10m 46s
15: learn: 0.7772766	total: 1m 47s	remaining: 10m 34s
16: learn: 0.777399	total: 1m 54s	remaining: 10m 24s
17: learn: 0.7777441	total: 2m	remaining: 10m 15s
18: learn: 0.7777775	total: 2m 5s	remaining: 10m 3s
19: learn: 0.7779572	total: 2m 11s	remaining: 9m 53s
20: learn: 0.7781739	total: 2m 17s	remaining: 9m 42s
21: learn: 0.7782359	total: 2m 23s	remaining: 9m 32s
22: learn: 0.778377	total: 2m 28s	remaining: 9m 23s
23: learn: 0.7785815	total: 2m 34s	remaining: 9m 13s
24: learn: 0.7789127	total: 2m 42s	remaining: 9m 11s
25: learn: 0.7789974	total: 2m 47s	remaining: 9m 2s
26: learn: 0.779324	total: 2m 54s	remaining: 8m 55s
27: learn: 0.7793577	total: 2m 59s	remaining: 8m 45s
28: learn: 0.7794031	total: 3m 5s	remaining: 8m 37s
29: learn: 0.7794016	total: 3m 10s	remaining: 8m 28s
30: learn: 0.7796384	total: 3m 16s	remaining: 8m 20s
31: learn: 0.7797058	total: 3m 22s	remaining: 8m 12s
32: learn: 0.7797856	total: 3m 27s	remaining: 8m 4s
33: learn: 0.7798112	total: 3m 34s	remaining: 7m 58s
34: learn: 0.7798437	total: 3m 39s	remaining: 7m 51s
35: learn: 0.7800193	total: 3m 45s	remaining: 7m 44s
36: learn: 0.7800967	total: 3m 51s	remaining: 7m 36s
37: learn: 0.7801488	total: 3m 56s	remaining: 7m 29s
38: learn: 0.7802763	total: 4m 2s	remaining: 7m 22s
39: learn: 0.7805066	total: 4m 10s	remaining: 7m 17s
40: learn: 0.7805419	total: 4m 15s	remaining: 7m 10s
41: learn: 0.7805746	total: 4m 21s	remaining: 7m 2s
42: learn: 0.7807385	total: 4m 27s	remaining: 6m 57s
43: learn: 0.7809252	total: 4m 33s	remaining: 6m 50s
44: learn: 0.7810327	total: 4m 39s	remaining: 6m 44s
45: learn: 0.7810482	total: 4m 45s	remaining: 6m 37s
46: learn: 0.7811367	total: 4m 50s	remaining: 6m 30s
47: learn: 0.7811913	total: 4m 56s	remaining: 6m 23s
48: learn: 0.7812727	total: 5m 2s	remaining: 6m 16s
49: learn: 0.7814671	total: 5m 9s	remaining: 6m 11s
50: learn: 0.7815976	total: 5m 15s	remaining: 6m 5s
51: learn: 0.7817565	total: 5m 21s	remaining: 5m 58s
52: learn: 0.781802	total: 5m 26s	remaining: 5m 51s
53: learn: 0.7818861	total: 5m 32s	remaining: 5m 44s
54: learn: 0.7819798	total: 5m 38s	remaining: 5m 38s
55: learn: 0.7820748	total: 5m 43s	remaining: 5m 31s
56: learn: 0.7821539	total: 5m 50s	remaining: 5m 26s
57: learn: 0.7821913	total: 5m 58s	remaining: 5m 21s
58: learn: 0.7822818	total: 6m 3s	remaining: 5m 14s
59: learn: 0.7825542	total: 6m 12s	remaining: 5m 10s
60: learn: 0.7827483	total: 6m 19s	remaining: 5m 4s
61: learn: 0.7828565	total: 6m 28s	remaining: 5m 1s
62: learn: 0.7829582	total: 6m 35s	remaining: 4m 55s
63: learn: 0.7830292	total: 6m 43s	remaining: 4m 50s
64: learn: 0.7831394	total: 6m 49s	remaining: 4m 43s
65: learn: 0.7833802	total: 6m 58s	remaining: 4m 39s
66: learn: 0.7835133	total: 7m 5s	remaining: 4m 32s
67: learn: 0.7836606	total: 7m 13s	remaining: 4m 27s
68: learn: 0.7837655	total: 7m 20s	remaining: 4m 21s
69: learn: 0.7838566	total: 7m 27s	remaining: 4m 15s
70: learn: 0.783916	total: 7m 37s	remaining: 4m 11s
71: learn: 0.7841061	total: 7m 46s	remaining: 4m 6s
72: learn: 0.7842343	total: 7m 55s	remaining: 4m 1s
73: learn: 0.7843451	total: 8m 4s	remaining: 3m 55s
74: learn: 0.784405	total: 8m 12s	remaining: 3m 49s
75: learn: 0.7844771	total: 8m 20s	remaining: 3m 44s
76: learn: 0.7846345	total: 8m 29s	remaining: 3m 38s
77: learn: 0.7848131	total: 8m 37s	remaining: 3m 32s
78: learn: 0.7849258	total: 8m 47s	remaining: 3m 26s
79: learn: 0.7850768	total: 8m 53s	remaining: 3m 20s
80: learn: 0.785182	total: 8m 59s	remaining: 3m 13s
81: learn: 0.7852334	total: 9m 9s	remaining: 3m 7s
82: learn: 0.7853704	total: 9m 18s	remaining: 3m 1s
83: learn: 0.7855111	total: 9m 29s	remaining: 2m 56s
84: learn: 0.7855852	total: 9m 39s	remaining: 2m 50s
85: learn: 0.7856868	total: 9m 49s	remaining: 2m 44s
86: learn: 0.7857624	total: 9m 58s	remaining: 2m 38s
87: learn: 0.7858772	total: 10m 7s	remaining: 2m 31s
88: learn: 0.7859835	total: 10m 16s	remaining: 2m 25s
89: learn: 0.7861171	total: 10m 25s	remaining: 2m 19s
90: learn: 0.7862263	total: 10m 35s	remaining: 2m 12s
91: learn: 0.7863898	total: 10m 42s	remaining: 2m 5s
92: learn: 0.7864509	total: 10m 51s	remaining: 1m 59s
93: learn: 0.7865071	total: 11m	remaining: 1m 52s
94: learn: 0.7866114	total: 11m 6s	remaining: 1m 45s
95: learn: 0.7866705	total: 11m 13s	remaining: 1m 38s
96: learn: 0.7867415	total: 11m 23s	remaining: 1m 31s
97: learn: 0.7868257	total: 11m 32s	remaining: 1m 24s
98: learn: 0.7868781	total: 11m 40s	remaining: 1m 17s
99: learn: 0.7869478	total: 11m 49s	remaining: 1m 10s
100: learn: 0.7870078	total: 11m 58s	remaining: 1m 4s
101: learn: 0.7871226	total: 12m 7s	remaining: 57s
102: learn: 0.7871686	total: 12m 15s	remaining: 50s
103: learn: 0.7872749	total: 12m 24s	remaining: 42.9s
104: learn: 0.7873216	total: 12m 31s	remaining: 35.8s
105: learn: 0.7874021	total: 12m 40s	remaining: 28.7s
106: learn: 0.7874457	total: 12m 48s	remaining: 21.5s
107: learn: 0.787514	total: 12m 56s	remaining: 14.4s
108: learn: 0.7875786	total: 13m 4s	remaining: 7.2s
109: learn: 0.7876151	total: 13m 13s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1
1        1   0.767281   0.763207
4        1   0.563007   0.551187
7        1   0.626437   0.560496
10       1   0.812196   0.822267
13       1   0.802438   0.822619
# # # # # # # # # # 
0.69465258926
0.998732813273
0.277197849623
0.0678992745853
0.174519930155
# # # # # # # # # # 

in model: CatR_XX_1  k-fold: 3 / 3

0: learn: 0.7624317	total: 10.2s	remaining: 18m 26s
1: learn: 0.7705112	total: 18.4s	remaining: 16m 31s
2: learn: 0.7722929	total: 24.7s	remaining: 14m 41s
3: learn: 0.7733135	total: 31.8s	remaining: 14m 3s
4: learn: 0.7741095	total: 38.8s	remaining: 13m 34s
5: learn: 0.7746078	total: 45s	remaining: 12m 59s
6: learn: 0.775334	total: 51.6s	remaining: 12m 38s
7: learn: 0.7753963	total: 57.6s	remaining: 12m 14s
8: learn: 0.7758229	total: 1m 3s	remaining: 11m 57s
9: learn: 0.7761143	total: 1m 9s	remaining: 11m 39s
10: learn: 0.7763909	total: 1m 16s	remaining: 11m 27s
11: learn: 0.7765566	total: 1m 22s	remaining: 11m 17s
12: learn: 0.7767485	total: 1m 29s	remaining: 11m 5s
13: learn: 0.7769732	total: 1m 35s	remaining: 10m 53s
14: learn: 0.7775074	total: 1m 41s	remaining: 10m 44s
15: learn: 0.777614	total: 1m 47s	remaining: 10m 34s
16: learn: 0.7775405	total: 1m 53s	remaining: 10m 21s
17: learn: 0.777764	total: 1m 59s	remaining: 10m 11s
18: learn: 0.7778574	total: 2m 5s	remaining: 10m 2s
19: learn: 0.7780849	total: 2m 11s	remaining: 9m 53s
20: learn: 0.7782539	total: 2m 17s	remaining: 9m 43s
21: learn: 0.7784298	total: 2m 23s	remaining: 9m 32s
22: learn: 0.7785726	total: 2m 28s	remaining: 9m 22s
23: learn: 0.7787579	total: 2m 34s	remaining: 9m 14s
24: learn: 0.7790765	total: 2m 42s	remaining: 9m 12s
25: learn: 0.7791889	total: 2m 48s	remaining: 9m 3s
26: learn: 0.7793167	total: 2m 53s	remaining: 8m 54s
27: learn: 0.7794384	total: 2m 59s	remaining: 8m 45s
28: learn: 0.7794752	total: 3m 4s	remaining: 8m 36s
29: learn: 0.7795724	total: 3m 10s	remaining: 8m 27s
30: learn: 0.7798684	total: 3m 16s	remaining: 8m 20s
31: learn: 0.7798903	total: 3m 22s	remaining: 8m 13s
32: learn: 0.7800337	total: 3m 28s	remaining: 8m 5s
33: learn: 0.7800984	total: 3m 33s	remaining: 7m 57s
34: learn: 0.7803699	total: 3m 39s	remaining: 7m 51s
35: learn: 0.7804631	total: 3m 45s	remaining: 7m 43s
36: learn: 0.7805036	total: 3m 51s	remaining: 7m 35s
37: learn: 0.7805726	total: 3m 57s	remaining: 7m 30s
38: learn: 0.780736	total: 4m 3s	remaining: 7m 23s
39: learn: 0.7807636	total: 4m 9s	remaining: 7m 16s
40: learn: 0.7809672	total: 4m 16s	remaining: 7m 11s
41: learn: 0.7810404	total: 4m 22s	remaining: 7m 5s
42: learn: 0.7811596	total: 4m 28s	remaining: 6m 57s
43: learn: 0.781339	total: 4m 34s	remaining: 6m 51s
44: learn: 0.7814082	total: 4m 41s	remaining: 6m 46s
45: learn: 0.7815004	total: 4m 47s	remaining: 6m 39s
46: learn: 0.7816474	total: 4m 53s	remaining: 6m 33s
47: learn: 0.7817288	total: 4m 59s	remaining: 6m 26s
48: learn: 0.7818894	total: 5m 5s	remaining: 6m 19s
49: learn: 0.7819603	total: 5m 11s	remaining: 6m 14s
50: learn: 0.7821097	total: 5m 18s	remaining: 6m 8s
51: learn: 0.7822393	total: 5m 24s	remaining: 6m 2s
52: learn: 0.7823889	total: 5m 30s	remaining: 5m 55s
53: learn: 0.7824925	total: 5m 35s	remaining: 5m 48s
54: learn: 0.7825508	total: 5m 41s	remaining: 5m 41s
55: learn: 0.7826404	total: 5m 46s	remaining: 5m 34s
56: learn: 0.7828107	total: 5m 53s	remaining: 5m 28s
57: learn: 0.7829581	total: 6m	remaining: 5m 23s
58: learn: 0.7830267	total: 6m 8s	remaining: 5m 18s
59: learn: 0.7831004	total: 6m 15s	remaining: 5m 12s
60: learn: 0.7832526	total: 6m 21s	remaining: 5m 6s
61: learn: 0.7834723	total: 6m 29s	remaining: 5m 1s
62: learn: 0.7836566	total: 6m 35s	remaining: 4m 55s
63: learn: 0.7837286	total: 6m 44s	remaining: 4m 50s
64: learn: 0.7838055	total: 6m 51s	remaining: 4m 45s
65: learn: 0.783936	total: 6m 58s	remaining: 4m 38s
66: learn: 0.7841362	total: 7m 6s	remaining: 4m 33s
67: learn: 0.7842782	total: 7m 13s	remaining: 4m 27s
68: learn: 0.7843456	total: 7m 19s	remaining: 4m 21s
69: learn: 0.7844645	total: 7m 27s	remaining: 4m 15s
70: learn: 0.7846008	total: 7m 33s	remaining: 4m 9s
71: learn: 0.7846573	total: 7m 41s	remaining: 4m 3s
72: learn: 0.7848242	total: 7m 49s	remaining: 3m 58s
73: learn: 0.7849017	total: 7m 59s	remaining: 3m 53s
74: learn: 0.7850693	total: 8m 6s	remaining: 3m 47s
75: learn: 0.7852053	total: 8m 16s	remaining: 3m 42s
76: learn: 0.7852977	total: 8m 23s	remaining: 3m 35s
77: learn: 0.785398	total: 8m 29s	remaining: 3m 29s
78: learn: 0.7855269	total: 8m 38s	remaining: 3m 23s
79: learn: 0.7856012	total: 8m 48s	remaining: 3m 18s
80: learn: 0.7857486	total: 8m 57s	remaining: 3m 12s
81: learn: 0.7859166	total: 9m 7s	remaining: 3m 7s
82: learn: 0.785993	total: 9m 16s	remaining: 3m 1s
83: learn: 0.7861443	total: 9m 26s	remaining: 2m 55s
84: learn: 0.786269	total: 9m 34s	remaining: 2m 49s
85: learn: 0.786361	total: 9m 45s	remaining: 2m 43s
86: learn: 0.7865215	total: 9m 54s	remaining: 2m 37s
87: learn: 0.7865778	total: 10m 3s	remaining: 2m 30s
88: learn: 0.7866849	total: 10m 12s	remaining: 2m 24s
89: learn: 0.7867603	total: 10m 20s	remaining: 2m 17s
90: learn: 0.7868266	total: 10m 28s	remaining: 2m 11s
91: learn: 0.7869203	total: 10m 37s	remaining: 2m 4s
92: learn: 0.7870179	total: 10m 45s	remaining: 1m 58s
93: learn: 0.7870679	total: 10m 54s	remaining: 1m 51s
94: learn: 0.7871781	total: 11m 3s	remaining: 1m 44s
95: learn: 0.787277	total: 11m 13s	remaining: 1m 38s
96: learn: 0.7873554	total: 11m 21s	remaining: 1m 31s
97: learn: 0.7874127	total: 11m 30s	remaining: 1m 24s
98: learn: 0.7875051	total: 11m 37s	remaining: 1m 17s
99: learn: 0.7876083	total: 11m 44s	remaining: 1m 10s
100: learn: 0.7876598	total: 11m 52s	remaining: 1m 3s
101: learn: 0.7877197	total: 12m 1s	remaining: 56.6s
102: learn: 0.7877951	total: 12m 11s	remaining: 49.7s
103: learn: 0.7878721	total: 12m 20s	remaining: 42.7s
104: learn: 0.7879422	total: 12m 28s	remaining: 35.6s
105: learn: 0.7879958	total: 12m 38s	remaining: 28.6s
106: learn: 0.7880315	total: 12m 47s	remaining: 21.5s
107: learn: 0.7880869	total: 12m 55s	remaining: 14.4s
108: learn: 0.7882082	total: 13m 7s	remaining: 7.22s
109: learn: 0.7882496	total: 13m 16s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1
2        1   0.844959   0.864025
5        1   0.822139   0.690211
8        1   0.911415   0.896298
11       1   0.814345   0.855633
14       1   0.891612   0.903683
# # # # # # # # # # 
0.978520390685
1.43668184228
0.363171124733
0.102872123148
0.263505646695
# # # # # # # # # # 
  id  CatC_XX_1  CatR_XX_1
0  0   0.332340   0.326173
1  1   0.437414   0.478894
2  2   0.138150   0.121057
3  3   0.038377   0.034291
4  4   0.096472   0.087835

in model: CatC_XX_2  k-fold: 1 / 3

0: learn: 0.7257918	total: 5.08s	remaining: 8m 23s
1: learn: 0.7429249	total: 9.56s	remaining: 7m 48s
2: learn: 0.7482801	total: 13s	remaining: 6m 58s
3: learn: 0.7507258	total: 16.4s	remaining: 6m 34s
4: learn: 0.7547514	total: 21.8s	remaining: 6m 54s
5: learn: 0.7567027	total: 26.3s	remaining: 6m 51s
6: learn: 0.7649877	total: 31.4s	remaining: 6m 56s
7: learn: 0.7659567	total: 34.9s	remaining: 6m 41s
8: learn: 0.7659567	total: 37.7s	remaining: 6m 20s
9: learn: 0.7659567	total: 40.2s	remaining: 6m 1s
10: learn: 0.7659567	total: 42.8s	remaining: 5m 46s
11: learn: 0.7659567	total: 45.3s	remaining: 5m 32s
12: learn: 0.7659567	total: 47.9s	remaining: 5m 20s
13: learn: 0.7675076	total: 51.9s	remaining: 5m 19s
14: learn: 0.7684506	total: 56.3s	remaining: 5m 19s
15: learn: 0.7701163	total: 1m	remaining: 5m 15s
16: learn: 0.7706549	total: 1m 4s	remaining: 5m 14s
17: learn: 0.7712628	total: 1m 7s	remaining: 5m 9s
18: learn: 0.7720399	total: 1m 12s	remaining: 5m 9s
19: learn: 0.773084	total: 1m 17s	remaining: 5m 10s
20: learn: 0.773151	total: 1m 21s	remaining: 5m 6s
21: learn: 0.7756663	total: 1m 25s	remaining: 5m 4s
22: learn: 0.7756663	total: 1m 28s	remaining: 4m 56s
23: learn: 0.7756663	total: 1m 31s	remaining: 4m 49s
24: learn: 0.7756663	total: 1m 34s	remaining: 4m 42s
25: learn: 0.7756663	total: 1m 36s	remaining: 4m 35s
26: learn: 0.7756663	total: 1m 39s	remaining: 4m 29s
27: learn: 0.776021	total: 1m 45s	remaining: 4m 30s
28: learn: 0.776021	total: 1m 48s	remaining: 4m 24s
29: learn: 0.7760666	total: 1m 52s	remaining: 4m 22s
30: learn: 0.7764543	total: 1m 57s	remaining: 4m 21s
31: learn: 0.7767455	total: 2m 1s	remaining: 4m 18s
32: learn: 0.7772919	total: 2m 6s	remaining: 4m 17s
33: learn: 0.7775082	total: 2m 11s	remaining: 4m 15s
34: learn: 0.7776722	total: 2m 15s	remaining: 4m 11s
35: learn: 0.7779133	total: 2m 18s	remaining: 4m 6s
36: learn: 0.7779355	total: 2m 22s	remaining: 4m 3s
37: learn: 0.7779879	total: 2m 27s	remaining: 3m 59s
38: learn: 0.7785172	total: 2m 32s	remaining: 3m 58s
39: learn: 0.7787564	total: 2m 37s	remaining: 3m 55s
40: learn: 0.7788812	total: 2m 40s	remaining: 3m 51s
41: learn: 0.7792276	total: 2m 46s	remaining: 3m 49s
42: learn: 0.7792892	total: 2m 49s	remaining: 3m 45s
43: learn: 0.7794439	total: 2m 54s	remaining: 3m 41s
44: learn: 0.7794806	total: 2m 57s	remaining: 3m 37s
45: learn: 0.7794931	total: 3m 2s	remaining: 3m 33s
46: learn: 0.7796278	total: 3m 7s	remaining: 3m 31s
47: learn: 0.7796926	total: 3m 11s	remaining: 3m 27s
48: learn: 0.7797274	total: 3m 15s	remaining: 3m 23s
49: learn: 0.7800769	total: 3m 20s	remaining: 3m 20s
50: learn: 0.7802828	total: 3m 25s	remaining: 3m 17s
51: learn: 0.7803416	total: 3m 30s	remaining: 3m 14s
52: learn: 0.7804893	total: 3m 35s	remaining: 3m 10s
53: learn: 0.7806212	total: 3m 39s	remaining: 3m 7s
54: learn: 0.7807558	total: 3m 44s	remaining: 3m 4s
55: learn: 0.7807558	total: 3m 47s	remaining: 2m 59s
56: learn: 0.7807558	total: 3m 50s	remaining: 2m 53s
57: learn: 0.7808118	total: 3m 55s	remaining: 2m 50s
58: learn: 0.7808118	total: 3m 58s	remaining: 2m 45s
59: learn: 0.7809311	total: 4m 3s	remaining: 2m 42s
60: learn: 0.7809551	total: 4m 7s	remaining: 2m 38s
61: learn: 0.7810055	total: 4m 11s	remaining: 2m 34s
62: learn: 0.7810639	total: 4m 15s	remaining: 2m 30s
63: learn: 0.7812242	total: 4m 20s	remaining: 2m 26s
64: learn: 0.781415	total: 4m 26s	remaining: 2m 23s
65: learn: 0.7815941	total: 4m 30s	remaining: 2m 19s
66: learn: 0.7816918	total: 4m 35s	remaining: 2m 15s
67: learn: 0.7817269	total: 4m 39s	remaining: 2m 11s
68: learn: 0.7818449	total: 4m 45s	remaining: 2m 8s
69: learn: 0.7818449	total: 4m 49s	remaining: 2m 3s
70: learn: 0.7818457	total: 4m 52s	remaining: 1m 59s
71: learn: 0.7818459	total: 4m 56s	remaining: 1m 55s
72: learn: 0.7818514	total: 5m	remaining: 1m 51s
73: learn: 0.7819633	total: 5m 5s	remaining: 1m 47s
74: learn: 0.782108	total: 5m 10s	remaining: 1m 43s
75: learn: 0.7821197	total: 5m 15s	remaining: 1m 39s
76: learn: 0.7821806	total: 5m 21s	remaining: 1m 35s
77: learn: 0.7822311	total: 5m 25s	remaining: 1m 31s
78: learn: 0.78226	total: 5m 29s	remaining: 1m 27s
79: learn: 0.7823841	total: 5m 35s	remaining: 1m 23s
80: learn: 0.7824036	total: 5m 40s	remaining: 1m 19s
81: learn: 0.7824833	total: 5m 44s	remaining: 1m 15s
82: learn: 0.7824853	total: 5m 48s	remaining: 1m 11s
83: learn: 0.7826387	total: 5m 52s	remaining: 1m 7s
84: learn: 0.7826389	total: 5m 56s	remaining: 1m 2s
85: learn: 0.7827368	total: 6m 1s	remaining: 58.8s
86: learn: 0.7827538	total: 6m 5s	remaining: 54.6s
87: learn: 0.7828013	total: 6m 9s	remaining: 50.4s
88: learn: 0.7828569	total: 6m 13s	remaining: 46.2s
89: learn: 0.7828723	total: 6m 18s	remaining: 42.1s
90: learn: 0.7829518	total: 6m 23s	remaining: 37.9s
91: learn: 0.7829564	total: 6m 28s	remaining: 33.8s
92: learn: 0.7833939	total: 6m 33s	remaining: 29.6s
93: learn: 0.7837532	total: 6m 38s	remaining: 25.4s
94: learn: 0.7837796	total: 6m 42s	remaining: 21.2s
95: learn: 0.783966	total: 6m 46s	remaining: 16.9s
96: learn: 0.783966	total: 6m 49s	remaining: 12.7s
97: learn: 0.7840025	total: 6m 54s	remaining: 8.45s
98: learn: 0.7840042	total: 6m 58s	remaining: 4.23s
99: learn: 0.7840086	total: 7m 3s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2
0        1   0.689373   0.553690   0.655145
3        1   0.608090   0.571467   0.569290
6        1   0.931559   0.914058   0.856881
9        1   0.916619   0.916289   0.901704
12       0   0.934667   0.938171   0.923478
# # # # # # # # # # 
0.39403362187
0.450405051679
0.14528841668
0.0436074329405
0.0999985139086
# # # # # # # # # # 

in model: CatC_XX_2  k-fold: 2 / 3

0: learn: 0.7258341	total: 5.07s	remaining: 8m 21s
1: learn: 0.7432292	total: 9.62s	remaining: 7m 51s
2: learn: 0.7488982	total: 13.5s	remaining: 7m 17s
3: learn: 0.7509838	total: 17.1s	remaining: 6m 50s
4: learn: 0.7610166	total: 21.6s	remaining: 6m 49s
5: learn: 0.7627503	total: 25.4s	remaining: 6m 38s
6: learn: 0.7661607	total: 30.6s	remaining: 6m 46s
7: learn: 0.7669083	total: 34.5s	remaining: 6m 36s
8: learn: 0.7679558	total: 38.2s	remaining: 6m 26s
9: learn: 0.7679974	total: 42.1s	remaining: 6m 18s
10: learn: 0.7682889	total: 46.1s	remaining: 6m 13s
11: learn: 0.7689859	total: 50.2s	remaining: 6m 7s
12: learn: 0.7689862	total: 53.1s	remaining: 5m 55s
13: learn: 0.7689862	total: 55.7s	remaining: 5m 42s
14: learn: 0.7689862	total: 58.5s	remaining: 5m 31s
15: learn: 0.7689862	total: 1m 1s	remaining: 5m 22s
16: learn: 0.7693152	total: 1m 5s	remaining: 5m 17s
17: learn: 0.7700519	total: 1m 10s	remaining: 5m 18s
18: learn: 0.7707515	total: 1m 14s	remaining: 5m 17s
19: learn: 0.7731055	total: 1m 19s	remaining: 5m 17s
20: learn: 0.7731056	total: 1m 22s	remaining: 5m 10s
21: learn: 0.7731056	total: 1m 25s	remaining: 5m 2s
22: learn: 0.7731056	total: 1m 27s	remaining: 4m 54s
23: learn: 0.7731056	total: 1m 30s	remaining: 4m 46s
24: learn: 0.7731056	total: 1m 33s	remaining: 4m 39s
25: learn: 0.7731056	total: 1m 35s	remaining: 4m 32s
26: learn: 0.7743509	total: 1m 40s	remaining: 4m 30s
27: learn: 0.7743509	total: 1m 42s	remaining: 4m 24s
28: learn: 0.7743509	total: 1m 45s	remaining: 4m 18s
29: learn: 0.7743509	total: 1m 48s	remaining: 4m 12s
30: learn: 0.7743509	total: 1m 50s	remaining: 4m 6s
31: learn: 0.7743509	total: 1m 53s	remaining: 4m 1s
32: learn: 0.7751452	total: 1m 57s	remaining: 3m 59s
33: learn: 0.7755143	total: 2m 3s	remaining: 3m 58s
34: learn: 0.7758015	total: 2m 7s	remaining: 3m 56s
35: learn: 0.7762852	total: 2m 12s	remaining: 3m 55s
36: learn: 0.7766351	total: 2m 16s	remaining: 3m 52s
37: learn: 0.7768092	total: 2m 20s	remaining: 3m 49s
38: learn: 0.7779017	total: 2m 24s	remaining: 3m 45s
39: learn: 0.7782629	total: 2m 28s	remaining: 3m 42s
40: learn: 0.7782642	total: 2m 31s	remaining: 3m 38s
41: learn: 0.7784678	total: 2m 35s	remaining: 3m 34s
42: learn: 0.778666	total: 2m 39s	remaining: 3m 31s
43: learn: 0.7788193	total: 2m 43s	remaining: 3m 28s
44: learn: 0.7794001	total: 2m 48s	remaining: 3m 25s
45: learn: 0.7795434	total: 2m 53s	remaining: 3m 23s
46: learn: 0.7797816	total: 2m 57s	remaining: 3m 20s
47: learn: 0.7798784	total: 3m 2s	remaining: 3m 17s
48: learn: 0.7801449	total: 3m 6s	remaining: 3m 14s
49: learn: 0.7802091	total: 3m 10s	remaining: 3m 10s
50: learn: 0.7804197	total: 3m 15s	remaining: 3m 8s
51: learn: 0.7804197	total: 3m 18s	remaining: 3m 3s
52: learn: 0.7804197	total: 3m 21s	remaining: 2m 58s
53: learn: 0.7804197	total: 3m 24s	remaining: 2m 53s
54: learn: 0.7804197	total: 3m 26s	remaining: 2m 49s
55: learn: 0.7806949	total: 3m 31s	remaining: 2m 46s
56: learn: 0.7806949	total: 3m 34s	remaining: 2m 41s
57: learn: 0.7806949	total: 3m 37s	remaining: 2m 37s
58: learn: 0.7806949	total: 3m 39s	remaining: 2m 32s
59: learn: 0.7806949	total: 3m 42s	remaining: 2m 28s
60: learn: 0.7806949	total: 3m 45s	remaining: 2m 24s
61: learn: 0.7806949	total: 3m 48s	remaining: 2m 19s
62: learn: 0.7809701	total: 3m 52s	remaining: 2m 16s
63: learn: 0.7810812	total: 3m 57s	remaining: 2m 13s
64: learn: 0.7811902	total: 4m 1s	remaining: 2m 10s
65: learn: 0.7812626	total: 4m 7s	remaining: 2m 7s
66: learn: 0.7814947	total: 4m 12s	remaining: 2m 4s
67: learn: 0.7815645	total: 4m 17s	remaining: 2m 1s
68: learn: 0.7816939	total: 4m 22s	remaining: 1m 57s
69: learn: 0.7817462	total: 4m 26s	remaining: 1m 54s
70: learn: 0.7817596	total: 4m 30s	remaining: 1m 50s
71: learn: 0.7818591	total: 4m 34s	remaining: 1m 46s
72: learn: 0.7819204	total: 4m 39s	remaining: 1m 43s
73: learn: 0.782039	total: 4m 44s	remaining: 1m 39s
74: learn: 0.7821836	total: 4m 49s	remaining: 1m 36s
75: learn: 0.7822763	total: 4m 53s	remaining: 1m 32s
76: learn: 0.7824408	total: 4m 57s	remaining: 1m 28s
77: learn: 0.7825296	total: 5m 1s	remaining: 1m 25s
78: learn: 0.78255	total: 5m 6s	remaining: 1m 21s
79: learn: 0.7825528	total: 5m 9s	remaining: 1m 17s
80: learn: 0.7826395	total: 5m 14s	remaining: 1m 13s
81: learn: 0.7826753	total: 5m 18s	remaining: 1m 9s
82: learn: 0.7826918	total: 5m 22s	remaining: 1m 6s
83: learn: 0.7827214	total: 5m 27s	remaining: 1m 2s
84: learn: 0.7827214	total: 5m 29s	remaining: 58.2s
85: learn: 0.7827214	total: 5m 32s	remaining: 54.1s
86: learn: 0.7827604	total: 5m 38s	remaining: 50.5s
87: learn: 0.7827627	total: 5m 42s	remaining: 46.7s
88: learn: 0.7828013	total: 5m 46s	remaining: 42.9s
89: learn: 0.7828532	total: 5m 51s	remaining: 39s
90: learn: 0.7828612	total: 5m 56s	remaining: 35.2s
91: learn: 0.7828863	total: 6m	remaining: 31.3s
92: learn: 0.7833304	total: 6m 4s	remaining: 27.5s
93: learn: 0.7833304	total: 6m 7s	remaining: 23.5s
94: learn: 0.7833304	total: 6m 10s	remaining: 19.5s
95: learn: 0.7833304	total: 6m 13s	remaining: 15.5s
96: learn: 0.7833304	total: 6m 15s	remaining: 11.6s
97: learn: 0.7833818	total: 6m 20s	remaining: 7.76s
98: learn: 0.783511	total: 6m 24s	remaining: 3.88s
99: learn: 0.7835623	total: 6m 28s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2
1        1   0.767281   0.763207   0.751800
4        1   0.563007   0.551187   0.633751
7        1   0.626437   0.560496   0.652828
10       1   0.812196   0.822267   0.826902
13       1   0.802438   0.822619   0.785917
# # # # # # # # # # 
0.720351896145
0.900733233463
0.30445633116
0.0778155291133
0.16931550513
# # # # # # # # # # 

in model: CatC_XX_2  k-fold: 3 / 3

0: learn: 0.7257974	total: 5.03s	remaining: 8m 17s
1: learn: 0.7430587	total: 9.59s	remaining: 7m 49s
2: learn: 0.7489683	total: 13.2s	remaining: 7m 5s
3: learn: 0.7517128	total: 16.8s	remaining: 6m 42s
4: learn: 0.7563321	total: 21.7s	remaining: 6m 52s
5: learn: 0.7591357	total: 25.3s	remaining: 6m 36s
6: learn: 0.7599093	total: 28.8s	remaining: 6m 22s
7: learn: 0.7600368	total: 33s	remaining: 6m 19s
8: learn: 0.7610809	total: 37.3s	remaining: 6m 16s
9: learn: 0.7610809	total: 39.8s	remaining: 5m 57s
10: learn: 0.7610809	total: 42.2s	remaining: 5m 41s
11: learn: 0.7610809	total: 44.7s	remaining: 5m 27s
12: learn: 0.7610809	total: 47.2s	remaining: 5m 16s
13: learn: 0.7610809	total: 49.8s	remaining: 5m 5s
14: learn: 0.7611895	total: 53.5s	remaining: 5m 3s
15: learn: 0.7625861	total: 57.4s	remaining: 5m 1s
16: learn: 0.7625861	total: 60s	remaining: 4m 52s
17: learn: 0.7625861	total: 1m 2s	remaining: 4m 44s
18: learn: 0.7625861	total: 1m 5s	remaining: 4m 37s
19: learn: 0.7625861	total: 1m 7s	remaining: 4m 30s
20: learn: 0.7626265	total: 1m 11s	remaining: 4m 29s
21: learn: 0.7629185	total: 1m 15s	remaining: 4m 26s
22: learn: 0.7632155	total: 1m 18s	remaining: 4m 24s
23: learn: 0.7632452	total: 1m 22s	remaining: 4m 21s
24: learn: 0.7637194	total: 1m 27s	remaining: 4m 21s
25: learn: 0.7637195	total: 1m 30s	remaining: 4m 16s
26: learn: 0.7637327	total: 1m 33s	remaining: 4m 13s
27: learn: 0.7637327	total: 1m 36s	remaining: 4m 8s
28: learn: 0.7637357	total: 1m 39s	remaining: 4m 4s
29: learn: 0.7637937	total: 1m 43s	remaining: 4m
30: learn: 0.7637978	total: 1m 46s	remaining: 3m 58s
31: learn: 0.7639703	total: 1m 50s	remaining: 3m 55s
32: learn: 0.7663185	total: 1m 54s	remaining: 3m 53s
33: learn: 0.7663185	total: 1m 57s	remaining: 3m 48s
34: learn: 0.7663185	total: 2m	remaining: 3m 44s
35: learn: 0.7663185	total: 2m 3s	remaining: 3m 40s
36: learn: 0.7663185	total: 2m 6s	remaining: 3m 35s
37: learn: 0.7663185	total: 2m 9s	remaining: 3m 30s
38: learn: 0.7676271	total: 2m 13s	remaining: 3m 28s
39: learn: 0.7688542	total: 2m 18s	remaining: 3m 28s
40: learn: 0.7691919	total: 2m 23s	remaining: 3m 25s
41: learn: 0.7696481	total: 2m 26s	remaining: 3m 22s
42: learn: 0.7709512	total: 2m 30s	remaining: 3m 19s
43: learn: 0.771251	total: 2m 35s	remaining: 3m 17s
44: learn: 0.771308	total: 2m 39s	remaining: 3m 14s
45: learn: 0.7716784	total: 2m 43s	remaining: 3m 11s
46: learn: 0.7718644	total: 2m 47s	remaining: 3m 8s
47: learn: 0.77199	total: 2m 51s	remaining: 3m 5s
48: learn: 0.7724199	total: 2m 56s	remaining: 3m 3s
49: learn: 0.7724914	total: 3m 3s	remaining: 3m 3s
50: learn: 0.7725658	total: 3m 7s	remaining: 3m
51: learn: 0.7726108	total: 3m 11s	remaining: 2m 56s
52: learn: 0.7727954	total: 3m 17s	remaining: 2m 55s
53: learn: 0.7727954	total: 3m 20s	remaining: 2m 51s
54: learn: 0.7727954	total: 3m 24s	remaining: 2m 46s
55: learn: 0.7727954	total: 3m 27s	remaining: 2m 42s
56: learn: 0.7727954	total: 3m 30s	remaining: 2m 38s
57: learn: 0.776373	total: 3m 35s	remaining: 2m 36s
58: learn: 0.776589	total: 3m 40s	remaining: 2m 33s
59: learn: 0.7765899	total: 3m 45s	remaining: 2m 30s
60: learn: 0.7765903	total: 3m 48s	remaining: 2m 26s
61: learn: 0.7765905	total: 3m 51s	remaining: 2m 22s
62: learn: 0.7765906	total: 3m 54s	remaining: 2m 17s
63: learn: 0.7770183	total: 3m 59s	remaining: 2m 14s
64: learn: 0.7772899	total: 4m 3s	remaining: 2m 11s
65: learn: 0.7774545	total: 4m 7s	remaining: 2m 7s
66: learn: 0.7776266	total: 4m 11s	remaining: 2m 3s
67: learn: 0.7776287	total: 4m 14s	remaining: 1m 59s
68: learn: 0.7776295	total: 4m 18s	remaining: 1m 56s
69: learn: 0.7776299	total: 4m 22s	remaining: 1m 52s
70: learn: 0.7777169	total: 4m 26s	remaining: 1m 48s
71: learn: 0.7778349	total: 4m 30s	remaining: 1m 45s
72: learn: 0.7779507	total: 4m 34s	remaining: 1m 41s
73: learn: 0.7780921	total: 4m 39s	remaining: 1m 38s
74: learn: 0.7780941	total: 4m 43s	remaining: 1m 34s
75: learn: 0.7780948	total: 4m 46s	remaining: 1m 30s
76: learn: 0.7780953	total: 4m 49s	remaining: 1m 26s
77: learn: 0.7780954	total: 4m 53s	remaining: 1m 22s
78: learn: 0.7780955	total: 4m 56s	remaining: 1m 18s
79: learn: 0.7780955	total: 4m 59s	remaining: 1m 14s
80: learn: 0.7781572	total: 5m 4s	remaining: 1m 11s
81: learn: 0.7784445	total: 5m 9s	remaining: 1m 7s
82: learn: 0.7784445	total: 5m 11s	remaining: 1m 3s
83: learn: 0.7784445	total: 5m 14s	remaining: 59.9s
84: learn: 0.7784445	total: 5m 17s	remaining: 56s
85: learn: 0.7784445	total: 5m 20s	remaining: 52.1s
86: learn: 0.7784445	total: 5m 22s	remaining: 48.3s
87: learn: 0.7784445	total: 5m 25s	remaining: 44.4s
88: learn: 0.7785846	total: 5m 29s	remaining: 40.7s
89: learn: 0.7786425	total: 5m 33s	remaining: 37s
90: learn: 0.7787016	total: 5m 37s	remaining: 33.4s
91: learn: 0.7787016	total: 5m 40s	remaining: 29.6s
92: learn: 0.7787866	total: 5m 44s	remaining: 25.9s
93: learn: 0.7788216	total: 5m 48s	remaining: 22.3s
94: learn: 0.7788447	total: 5m 52s	remaining: 18.6s
95: learn: 0.7789343	total: 5m 57s	remaining: 14.9s
96: learn: 0.7789568	total: 6m 1s	remaining: 11.2s
97: learn: 0.7790788	total: 6m 6s	remaining: 7.49s
98: learn: 0.7790938	total: 6m 12s	remaining: 3.76s
99: learn: 0.7791414	total: 6m 16s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2
2        1   0.844959   0.864025   0.891801
5        1   0.822139   0.690211   0.769481
8        1   0.911415   0.896298   0.904809
11       1   0.814345   0.855633   0.855232
14       1   0.891612   0.903683   0.924271
# # # # # # # # # # 
1.02154528771
1.3384535831
0.42539596739
0.117279108711
0.296001578358
# # # # # # # # # # 
  id  CatC_XX_1  CatR_XX_1  CatC_XX_2
0  0   0.332340   0.326173   0.340515
1  1   0.437414   0.478894   0.446151
2  2   0.138150   0.121057   0.141799
3  3   0.038377   0.034291   0.039093
4  4   0.096472   0.087835   0.098667

in model: CatR_XX_2  k-fold: 1 / 3

0: learn: 0.7624731	total: 11s	remaining: 8m 58s
1: learn: 0.7777629	total: 36.1s	remaining: 14m 26s
2: learn: 0.7841077	total: 58.6s	remaining: 15m 17s
3: learn: 0.7855141	total: 1m 21s	remaining: 15m 41s
4: learn: 0.7870923	total: 1m 46s	remaining: 16m
5: learn: 0.7888732	total: 2m 11s	remaining: 16m 6s
6: learn: 0.7888843	total: 2m 14s	remaining: 13m 45s
7: learn: 0.7888878	total: 2m 18s	remaining: 12m 4s
8: learn: 0.7888885	total: 2m 20s	remaining: 10m 40s
9: learn: 0.7894661	total: 2m 30s	remaining: 10m 3s
10: learn: 0.7904987	total: 2m 51s	remaining: 10m 9s
11: learn: 0.7905698	total: 2m 58s	remaining: 9m 23s
12: learn: 0.7928036	total: 3m 23s	remaining: 9m 40s
13: learn: 0.7928841	total: 3m 29s	remaining: 8m 59s
14: learn: 0.7929125	total: 3m 34s	remaining: 8m 19s
15: learn: 0.7931065	total: 3m 44s	remaining: 7m 57s
16: learn: 0.7931074	total: 3m 47s	remaining: 7m 22s
17: learn: 0.7931076	total: 3m 50s	remaining: 6m 48s
18: learn: 0.7931077	total: 3m 52s	remaining: 6m 19s
19: learn: 0.7940588	total: 4m 19s	remaining: 6m 29s
20: learn: 0.7943339	total: 4m 46s	remaining: 6m 36s
21: learn: 0.7948147	total: 5m 11s	remaining: 6m 36s
22: learn: 0.7948207	total: 5m 14s	remaining: 6m 9s
23: learn: 0.7949125	total: 5m 33s	remaining: 6m 1s
24: learn: 0.7949239	total: 5m 40s	remaining: 5m 40s
25: learn: 0.7949246	total: 5m 44s	remaining: 5m 17s
26: learn: 0.7949248	total: 5m 46s	remaining: 4m 55s
27: learn: 0.7950695	total: 6m 8s	remaining: 4m 49s
28: learn: 0.7957907	total: 6m 33s	remaining: 4m 44s
29: learn: 0.7971688	total: 6m 55s	remaining: 4m 37s
30: learn: 0.7972463	total: 7m 9s	remaining: 4m 23s
31: learn: 0.7972566	total: 7m 28s	remaining: 4m 12s
32: learn: 0.7981236	total: 7m 44s	remaining: 3m 59s
33: learn: 0.7981236	total: 7m 46s	remaining: 3m 39s
34: learn: 0.7982553	total: 8m 4s	remaining: 3m 27s
35: learn: 0.7983037	total: 8m 23s	remaining: 3m 15s
36: learn: 0.7983068	total: 8m 37s	remaining: 3m 1s
37: learn: 0.7988196	total: 9m	remaining: 2m 50s
38: learn: 0.7988288	total: 9m 12s	remaining: 2m 35s
39: learn: 0.7988298	total: 9m 23s	remaining: 2m 20s
40: learn: 0.7988882	total: 9m 44s	remaining: 2m 8s
41: learn: 0.7992464	total: 10m 4s	remaining: 1m 55s
42: learn: 0.7997278	total: 10m 31s	remaining: 1m 42s
43: learn: 0.7999077	total: 10m 54s	remaining: 1m 29s
44: learn: 0.8000974	total: 11m 17s	remaining: 1m 15s
45: learn: 0.8000974	total: 11m 20s	remaining: 59.1s
46: learn: 0.8000974	total: 11m 22s	remaining: 43.6s
47: learn: 0.8000974	total: 11m 25s	remaining: 28.5s
48: learn: 0.8000974	total: 11m 27s	remaining: 14s
49: learn: 0.8000974	total: 11m 29s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
0        1   0.689373   0.553690   0.655145   0.481753
3        1   0.608090   0.571467   0.569290   0.533063
6        1   0.931559   0.914058   0.856881   0.930974
9        1   0.916619   0.916289   0.901704   0.904606
12       0   0.934667   0.938171   0.923478   0.895320
# # # # # # # # # # 
0.334430621735
0.429345983835
0.213504590102
0.132596207591
0.143405040498
# # # # # # # # # # 

in model: CatR_XX_2  k-fold: 2 / 3

0: learn: 0.7736944	total: 27.1s	remaining: 22m 5s
1: learn: 0.7801506	total: 50.3s	remaining: 20m 7s
2: learn: 0.7858227	total: 1m 16s	remaining: 19m 50s
3: learn: 0.7875548	total: 1m 38s	remaining: 18m 53s
4: learn: 0.7891328	total: 2m 4s	remaining: 18m 38s
5: learn: 0.7901078	total: 2m 28s	remaining: 18m 9s
6: learn: 0.7913742	total: 2m 47s	remaining: 17m 9s
7: learn: 0.7913778	total: 2m 52s	remaining: 15m 6s
8: learn: 0.7913892	total: 2m 56s	remaining: 13m 24s
9: learn: 0.7916163	total: 3m 5s	remaining: 12m 21s
10: learn: 0.7916166	total: 3m 7s	remaining: 11m 6s
11: learn: 0.7916167	total: 3m 10s	remaining: 10m 2s
12: learn: 0.7916199	total: 3m 13s	remaining: 9m 10s
13: learn: 0.7924468	total: 3m 37s	remaining: 9m 19s
14: learn: 0.79256	total: 3m 42s	remaining: 8m 38s
15: learn: 0.7925603	total: 3m 45s	remaining: 7m 59s
16: learn: 0.7925597	total: 3m 48s	remaining: 7m 24s
17: learn: 0.7934963	total: 4m 19s	remaining: 7m 41s
18: learn: 0.7943963	total: 4m 47s	remaining: 7m 49s
19: learn: 0.795199	total: 5m 13s	remaining: 7m 50s
20: learn: 0.7952033	total: 5m 17s	remaining: 7m 17s
21: learn: 0.7952033	total: 5m 18s	remaining: 6m 45s
22: learn: 0.7953055	total: 5m 33s	remaining: 6m 30s
23: learn: 0.7956814	total: 5m 48s	remaining: 6m 17s
24: learn: 0.7959058	total: 6m 10s	remaining: 6m 10s
25: learn: 0.7959144	total: 6m 16s	remaining: 5m 47s
26: learn: 0.7959159	total: 6m 20s	remaining: 5m 24s
27: learn: 0.7959205	total: 6m 25s	remaining: 5m 2s
28: learn: 0.7959216	total: 6m 29s	remaining: 4m 42s
29: learn: 0.7959828	total: 6m 48s	remaining: 4m 32s
30: learn: 0.7963147	total: 7m 12s	remaining: 4m 25s
31: learn: 0.7968842	total: 7m 38s	remaining: 4m 17s
32: learn: 0.7974503	total: 8m 4s	remaining: 4m 9s
33: learn: 0.7974609	total: 8m 18s	remaining: 3m 54s
34: learn: 0.7974621	total: 8m 25s	remaining: 3m 36s
35: learn: 0.7974661	total: 8m 40s	remaining: 3m 22s
36: learn: 0.7974676	total: 8m 50s	remaining: 3m 6s
37: learn: 0.7976131	total: 9m 9s	remaining: 2m 53s
38: learn: 0.7981052	total: 9m 28s	remaining: 2m 40s
39: learn: 0.7981918	total: 9m 49s	remaining: 2m 27s
40: learn: 0.7981925	total: 9m 54s	remaining: 2m 10s
41: learn: 0.7981925	total: 9m 56s	remaining: 1m 53s
42: learn: 0.7981925	total: 9m 57s	remaining: 1m 37s
43: learn: 0.7981932	total: 10m 2s	remaining: 1m 22s
44: learn: 0.7985284	total: 10m 24s	remaining: 1m 9s
45: learn: 0.7985844	total: 10m 42s	remaining: 55.9s
46: learn: 0.7986336	total: 11m 3s	remaining: 42.3s
47: learn: 0.7986707	total: 11m 25s	remaining: 28.5s
48: learn: 0.7989577	total: 11m 48s	remaining: 14.4s
49: learn: 0.7992902	total: 12m 15s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
1        1   0.767281   0.763207   0.751800   0.861751
4        1   0.563007   0.551187   0.633751   0.814859
7        1   0.626437   0.560496   0.652828   0.735203
10       1   0.812196   0.822267   0.826902   0.788409
13       1   0.802438   0.822619   0.785917   0.879429
# # # # # # # # # # 
0.686368234866
0.813446821849
0.386994195518
0.177039138815
0.142656374677
# # # # # # # # # # 

in model: CatR_XX_2  k-fold: 3 / 3

0: learn: 0.7724365	total: 24.5s	remaining: 20m
1: learn: 0.7810181	total: 46.6s	remaining: 18m 38s
2: learn: 0.7846924	total: 1m 10s	remaining: 18m 17s
3: learn: 0.7867814	total: 1m 37s	remaining: 18m 37s
4: learn: 0.7894248	total: 1m 59s	remaining: 17m 54s
5: learn: 0.790421	total: 2m 25s	remaining: 17m 45s
6: learn: 0.7904652	total: 2m 30s	remaining: 15m 26s
7: learn: 0.790465	total: 2m 33s	remaining: 13m 24s
8: learn: 0.7915239	total: 2m 56s	remaining: 13m 24s
9: learn: 0.7926932	total: 3m 22s	remaining: 13m 31s
10: learn: 0.7927033	total: 3m 26s	remaining: 12m 12s
11: learn: 0.7927145	total: 3m 29s	remaining: 11m 4s
12: learn: 0.7937204	total: 3m 50s	remaining: 10m 57s
13: learn: 0.7938268	total: 3m 58s	remaining: 10m 12s
14: learn: 0.7942072	total: 4m 25s	remaining: 10m 18s
15: learn: 0.7942116	total: 4m 27s	remaining: 9m 28s
16: learn: 0.7944922	total: 4m 49s	remaining: 9m 21s
17: learn: 0.7945	total: 4m 52s	remaining: 8m 39s
18: learn: 0.794574	total: 4m 59s	remaining: 8m 9s
19: learn: 0.7945778	total: 5m 3s	remaining: 7m 34s
20: learn: 0.7948232	total: 5m 23s	remaining: 7m 27s
21: learn: 0.7948247	total: 5m 27s	remaining: 6m 56s
22: learn: 0.7948251	total: 5m 30s	remaining: 6m 28s
23: learn: 0.7950175	total: 5m 49s	remaining: 6m 18s
24: learn: 0.7950458	total: 6m 5s	remaining: 6m 5s
25: learn: 0.7955759	total: 6m 32s	remaining: 6m 2s
26: learn: 0.7955788	total: 6m 47s	remaining: 5m 46s
27: learn: 0.796111	total: 7m 6s	remaining: 5m 35s
28: learn: 0.7961953	total: 7m 23s	remaining: 5m 21s
29: learn: 0.796673	total: 7m 46s	remaining: 5m 11s
30: learn: 0.796673	total: 7m 48s	remaining: 4m 47s
31: learn: 0.796673	total: 7m 50s	remaining: 4m 24s
32: learn: 0.796673	total: 7m 52s	remaining: 4m 3s
33: learn: 0.796673	total: 7m 54s	remaining: 3m 43s
34: learn: 0.796673	total: 7m 56s	remaining: 3m 24s
35: learn: 0.796673	total: 7m 58s	remaining: 3m 5s
36: learn: 0.796673	total: 8m	remaining: 2m 48s
37: learn: 0.796673	total: 8m 1s	remaining: 2m 32s
38: learn: 0.796673	total: 8m 3s	remaining: 2m 16s
39: learn: 0.796673	total: 8m 5s	remaining: 2m 1s
40: learn: 0.796673	total: 8m 7s	remaining: 1m 47s
41: learn: 0.796673	total: 8m 9s	remaining: 1m 33s
42: learn: 0.796673	total: 8m 11s	remaining: 1m 20s
43: learn: 0.796673	total: 8m 13s	remaining: 1m 7s
44: learn: 0.796673	total: 8m 15s	remaining: 55s
45: learn: 0.796673	total: 8m 17s	remaining: 43.2s
46: learn: 0.796673	total: 8m 18s	remaining: 31.9s
47: learn: 0.796673	total: 8m 20s	remaining: 20.9s
48: learn: 0.796673	total: 8m 22s	remaining: 10.3s
49: learn: 0.796673	total: 8m 24s	remaining: 0us
- - - - - - - - - - 
'msno',
'song_id',
'language',
'top3_in_song',
'ITC_source_system_tab_log10_1',
'ISC_song_country_ln',
'membership_days',
'ISC_song_year',
'OinC_language',
- - - - - - - - - - 
    target  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
2        1   0.844959   0.864025   0.891801   0.853083
5        1   0.822139   0.690211   0.769481   0.789782
8        1   0.911415   0.896298   0.904809   0.897755
11       1   0.814345   0.855633   0.855232   0.814420
14       1   0.891612   0.903683   0.924271   0.868322
# # # # # # # # # # 
1.00267220555
1.16884178568
0.50218233654
0.173510253413
0.209737444996
# # # # # # # # # # 
  id  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
0  0   0.332340   0.326173   0.340515   0.334224
1  1   0.437414   0.478894   0.446151   0.389614
2  2   0.138150   0.121057   0.141799   0.167394
3  3   0.038377   0.034291   0.039093   0.057837
4  4   0.096472   0.087835   0.098667   0.069912
  id  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
0  0   0.332340   0.326173   0.340515   0.334224
1  1   0.437414   0.478894   0.446151   0.389614
2  2   0.138150   0.121057   0.141799   0.167394
3  3   0.038377   0.034291   0.039093   0.057837
4  4   0.096472   0.087835   0.098667   0.069912
              id  CatC_XX_1  CatR_XX_1  CatC_XX_2  CatR_XX_2
2556785  2556785   0.085902   0.080783   0.089260   0.089856
2556786  2556786   0.284520   0.315343   0.296637   0.291052
2556787  2556787   0.316310   0.351873   0.338929   0.313283
2556788  2556788   0.230976   0.221815   0.286898   0.190540
2556789  2556789   0.280266   0.315343   0.293894   0.281278
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target         uint8
CatC_XX_1    float64
CatR_XX_1    float64
CatC_XX_2    float64
CatR_XX_2    float64
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
target         uint8
CatC_XX_1    float64
CatR_XX_1    float64
CatC_XX_2    float64
CatR_XX_2    float64
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
target         uint8
CatC_XX_1    float64
CatR_XX_1    float64
CatC_XX_2    float64
CatR_XX_2    float64
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
id           category
CatC_XX_1     float64
CatR_XX_1     float64
CatC_XX_2     float64
CatR_XX_2     float64
dtype: object
number of columns: 5
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.

[timer]: complete in 139m 48s

Process finished with exit code 0
'''