import sys
sys.path.insert(0, '../')
from me import *
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
df = read_df(load_name)

on = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
df = df[on]
show_df(df)

# !!!!!!!!!!!!!!!!!!!!!!!!!

iterations = 200
learning_rate = 0.3
depth = 6
estimate = 0.6925

model, cols = train_cat(df, iterations,
                        learning_rate=learning_rate,
                        depth=depth)
del df

# !!!!!!!!!!!!!!!!!!!!!!!!!


print('training complete.')
print('Making prediction')

load_name = 'final_test_real.csv'
df = read_df(load_name)

cols.remove('target')
cols.append('id')
df = df[cols]

test = df.drop(['id'], axis=1)
ids = df['id'].values
del df

p = cat_predict(model, test)
del model

print('prediction done.')
print('creating submission')
subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = p
del p


model_time = str(int(time.time()))
model_name = '_cat_'
model_name = '[]_'+str(estimate)+model_name
model_name = model_name + '_' + model_time
subm.to_csv(save_dir+'submission/'+model_name+'.csv',
            index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

# 0.68151
'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/Final_Trainer/single_cat_trainer_B.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
  return f(*args, **kwds)

This is [no drill] training.


>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                   category
song_id                category
source_screen_name     category
source_type            category
target                    uint8
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
source_system_tab      category
dtype: object
number of rows: 7377418
number of columns: 11

'msno',
'song_id',
'source_screen_name',
'source_type',
'target',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
0: learn: 0.7902133	total: 21.8s	remaining: 1h 12m 24s
1: learn: 0.80348	total: 42.6s	remaining: 1h 10m 19s
2: learn: 0.8101132	total: 59.1s	remaining: 1h 4m 39s
3: learn: 0.8132112	total: 1m 17s	remaining: 1h 3m 17s
4: learn: 0.815022	total: 1m 37s	remaining: 1h 3m 11s
5: learn: 0.8173046	total: 1m 56s	remaining: 1h 2m 58s
6: learn: 0.819063	total: 2m 26s	remaining: 1h 7m 31s
7: learn: 0.8202257	total: 3m 11s	remaining: 1h 16m 32s
8: learn: 0.8212173	total: 3m 35s	remaining: 1h 16m 10s
9: learn: 0.8212173	total: 3m 40s	remaining: 1h 9m 58s
10: learn: 0.8212173	total: 3m 46s	remaining: 1h 4m 53s
11: learn: 0.8212173	total: 3m 52s	remaining: 1h 38s
12: learn: 0.8212173	total: 3m 57s	remaining: 57m
13: learn: 0.8212173	total: 4m 3s	remaining: 53m 53s
14: learn: 0.8212173	total: 4m 9s	remaining: 51m 11s
15: learn: 0.8212173	total: 4m 14s	remaining: 48m 49s
16: learn: 0.8212173	total: 4m 20s	remaining: 46m 41s
17: learn: 0.8212173	total: 4m 25s	remaining: 44m 48s
18: learn: 0.8212173	total: 4m 31s	remaining: 43m 6s
19: learn: 0.8216604	total: 4m 51s	remaining: 43m 47s
20: learn: 0.8225536	total: 5m 12s	remaining: 44m 26s
21: learn: 0.8225536	total: 5m 18s	remaining: 42m 56s
22: learn: 0.8225536	total: 5m 24s	remaining: 41m 33s
23: learn: 0.8225536	total: 5m 29s	remaining: 40m 17s
24: learn: 0.8225536	total: 5m 35s	remaining: 39m 7s
25: learn: 0.8225536	total: 5m 40s	remaining: 38m 1s
26: learn: 0.8225536	total: 5m 46s	remaining: 37m
27: learn: 0.8225536	total: 5m 52s	remaining: 36m 2s
28: learn: 0.8225536	total: 5m 57s	remaining: 35m 9s
29: learn: 0.8228368	total: 6m 18s	remaining: 35m 45s
30: learn: 0.8228368	total: 6m 24s	remaining: 34m 54s
31: learn: 0.8228369	total: 6m 31s	remaining: 34m 16s
32: learn: 0.8228369	total: 6m 37s	remaining: 33m 29s
33: learn: 0.8228369	total: 6m 46s	remaining: 33m 5s
34: learn: 0.8228369	total: 6m 52s	remaining: 32m 23s
35: learn: 0.8228369	total: 6m 57s	remaining: 31m 43s
36: learn: 0.8228369	total: 7m 3s	remaining: 31m 5s
37: learn: 0.8235956	total: 7m 25s	remaining: 31m 38s
38: learn: 0.8235956	total: 7m 32s	remaining: 31m 7s
39: learn: 0.8235956	total: 7m 37s	remaining: 30m 31s
40: learn: 0.8235956	total: 7m 43s	remaining: 29m 57s
41: learn: 0.8235956	total: 7m 49s	remaining: 29m 24s
42: learn: 0.8235956	total: 7m 54s	remaining: 28m 53s
43: learn: 0.8235956	total: 8m	remaining: 28m 22s
44: learn: 0.8235956	total: 8m 5s	remaining: 27m 53s
45: learn: 0.8235956	total: 8m 13s	remaining: 27m 31s
46: learn: 0.8235956	total: 8m 18s	remaining: 27m 3s
47: learn: 0.8235956	total: 8m 24s	remaining: 26m 37s
48: learn: 0.8235956	total: 8m 30s	remaining: 26m 11s
49: learn: 0.8235956	total: 8m 37s	remaining: 25m 52s
50: learn: 0.8235956	total: 8m 43s	remaining: 25m 28s
51: learn: 0.8235956	total: 8m 48s	remaining: 25m 4s
52: learn: 0.8235957	total: 8m 57s	remaining: 24m 51s
53: learn: 0.8245336	total: 9m 20s	remaining: 25m 15s
54: learn: 0.8250445	total: 9m 42s	remaining: 25m 35s
55: learn: 0.8255143	total: 10m 4s	remaining: 25m 53s
56: learn: 0.8255143	total: 10m 9s	remaining: 25m 30s
57: learn: 0.8258322	total: 10m 29s	remaining: 25m 40s
58: learn: 0.8261958	total: 10m 49s	remaining: 25m 51s
59: learn: 0.8263419	total: 11m 8s	remaining: 25m 59s
60: learn: 0.8267379	total: 11m 25s	remaining: 26m 3s
61: learn: 0.8270624	total: 11m 47s	remaining: 26m 14s
62: learn: 0.8273971	total: 12m 9s	remaining: 26m 26s
63: learn: 0.8277412	total: 12m 33s	remaining: 26m 41s
64: learn: 0.8279211	total: 12m 55s	remaining: 26m 50s
65: learn: 0.828042	total: 13m 15s	remaining: 26m 54s
66: learn: 0.8284486	total: 13m 38s	remaining: 27m 4s
67: learn: 0.82862	total: 13m 59s	remaining: 27m 10s
68: learn: 0.828814	total: 14m 19s	remaining: 27m 12s
69: learn: 0.828814	total: 14m 25s	remaining: 26m 47s
70: learn: 0.828814	total: 14m 31s	remaining: 26m 22s
71: learn: 0.8288143	total: 14m 42s	remaining: 26m 9s
72: learn: 0.8288454	total: 15m 2s	remaining: 26m 10s
73: learn: 0.829148	total: 15m 25s	remaining: 26m 16s
74: learn: 0.8292789	total: 15m 47s	remaining: 26m 19s
75: learn: 0.8295294	total: 16m 7s	remaining: 26m 17s
76: learn: 0.8295296	total: 16m 18s	remaining: 26m 3s
77: learn: 0.8295297	total: 16m 31s	remaining: 25m 51s
78: learn: 0.8295298	total: 16m 41s	remaining: 25m 34s
79: learn: 0.8295298	total: 16m 51s	remaining: 25m 17s
80: learn: 0.8295298	total: 17m 2s	remaining: 25m 1s
81: learn: 0.8295298	total: 17m 12s	remaining: 24m 45s
82: learn: 0.8298143	total: 17m 32s	remaining: 24m 43s
83: learn: 0.8300436	total: 17m 51s	remaining: 24m 40s
84: learn: 0.8302886	total: 18m 17s	remaining: 24m 44s
85: learn: 0.8303993	total: 18m 37s	remaining: 24m 41s
86: learn: 0.8306031	total: 18m 59s	remaining: 24m 40s
87: learn: 0.8308184	total: 19m 20s	remaining: 24m 36s
88: learn: 0.8308184	total: 19m 33s	remaining: 24m 24s
89: learn: 0.8308185	total: 19m 46s	remaining: 24m 10s
90: learn: 0.8308185	total: 20m	remaining: 23m 57s
91: learn: 0.8308185	total: 20m 12s	remaining: 23m 43s
92: learn: 0.8308185	total: 20m 25s	remaining: 23m 29s
93: learn: 0.8308186	total: 20m 37s	remaining: 23m 15s
94: learn: 0.8309447	total: 20m 59s	remaining: 23m 12s
95: learn: 0.8311207	total: 21m 21s	remaining: 23m 8s
96: learn: 0.8312421	total: 21m 40s	remaining: 23m
97: learn: 0.8313654	total: 22m 2s	remaining: 22m 56s
98: learn: 0.8315736	total: 22m 22s	remaining: 22m 50s
99: learn: 0.8315736	total: 22m 31s	remaining: 22m 31s
100: learn: 0.8315739	total: 22m 47s	remaining: 22m 20s
101: learn: 0.8315739	total: 23m 6s	remaining: 22m 12s
102: learn: 0.8315739	total: 23m 21s	remaining: 22m
103: learn: 0.8315739	total: 23m 37s	remaining: 21m 48s
104: learn: 0.8315739	total: 23m 43s	remaining: 21m 28s
105: learn: 0.8315739	total: 23m 49s	remaining: 21m 7s
106: learn: 0.8315739	total: 23m 55s	remaining: 20m 47s
107: learn: 0.8315739	total: 24m 1s	remaining: 20m 28s
108: learn: 0.8315739	total: 24m 12s	remaining: 20m 12s
109: learn: 0.8316614	total: 24m 28s	remaining: 20m 1s
110: learn: 0.8316615	total: 24m 41s	remaining: 19m 47s
111: learn: 0.8316615	total: 24m 53s	remaining: 19m 33s
112: learn: 0.8316615	total: 24m 59s	remaining: 19m 14s
113: learn: 0.8316615	total: 25m 4s	remaining: 18m 55s
114: learn: 0.8316615	total: 25m 14s	remaining: 18m 39s
115: learn: 0.8316615	total: 25m 20s	remaining: 18m 21s
116: learn: 0.8316615	total: 25m 32s	remaining: 18m 7s
117: learn: 0.8317142	total: 25m 51s	remaining: 17m 58s
118: learn: 0.831869	total: 26m 7s	remaining: 17m 47s
119: learn: 0.8319975	total: 26m 27s	remaining: 17m 38s
120: learn: 0.8320418	total: 26m 45s	remaining: 17m 28s
121: learn: 0.8321034	total: 27m 2s	remaining: 17m 17s
122: learn: 0.8321779	total: 27m 24s	remaining: 17m 9s
123: learn: 0.8322361	total: 27m 49s	remaining: 17m 3s
124: learn: 0.8323605	total: 28m 9s	remaining: 16m 53s
125: learn: 0.8324661	total: 28m 28s	remaining: 16m 43s
126: learn: 0.8325971	total: 28m 49s	remaining: 16m 34s
127: learn: 0.8327113	total: 29m 10s	remaining: 16m 24s
128: learn: 0.8327781	total: 29m 29s	remaining: 16m 13s
129: learn: 0.8327781	total: 29m 38s	remaining: 15m 57s
130: learn: 0.8327781	total: 29m 50s	remaining: 15m 42s
131: learn: 0.8327781	total: 29m 58s	remaining: 15m 26s
132: learn: 0.8328163	total: 30m 16s	remaining: 15m 14s
133: learn: 0.832858	total: 30m 34s	remaining: 15m 3s
134: learn: 0.8328581	total: 30m 43s	remaining: 14m 47s
135: learn: 0.8328581	total: 30m 53s	remaining: 14m 32s
136: learn: 0.8328581	total: 30m 59s	remaining: 14m 15s
137: learn: 0.8328581	total: 31m 9s	remaining: 13m 59s
138: learn: 0.8328581	total: 31m 19s	remaining: 13m 44s
139: learn: 0.8329308	total: 31m 39s	remaining: 13m 33s
140: learn: 0.8329655	total: 31m 59s	remaining: 13m 23s
141: learn: 0.8331093	total: 32m 19s	remaining: 13m 12s
142: learn: 0.8332081	total: 32m 43s	remaining: 13m 2s
143: learn: 0.8332567	total: 33m 8s	remaining: 12m 53s
144: learn: 0.8333222	total: 33m 31s	remaining: 12m 43s
145: learn: 0.8333833	total: 33m 49s	remaining: 12m 30s
146: learn: 0.8334691	total: 34m 9s	remaining: 12m 19s
147: learn: 0.8334888	total: 34m 25s	remaining: 12m 5s
148: learn: 0.8335374	total: 34m 48s	remaining: 11m 54s
149: learn: 0.8336167	total: 35m 5s	remaining: 11m 41s
150: learn: 0.8336167	total: 35m 14s	remaining: 11m 26s
151: learn: 0.8336167	total: 35m 22s	remaining: 11m 10s
152: learn: 0.8336167	total: 35m 31s	remaining: 10m 54s
153: learn: 0.8336167	total: 35m 45s	remaining: 10m 40s
154: learn: 0.8336167	total: 35m 53s	remaining: 10m 25s
155: learn: 0.8336167	total: 36m 5s	remaining: 10m 10s
156: learn: 0.8336167	total: 36m 15s	remaining: 9m 55s
157: learn: 0.8336472	total: 36m 37s	remaining: 9m 44s
158: learn: 0.8336473	total: 36m 49s	remaining: 9m 29s
159: learn: 0.8336473	total: 37m 1s	remaining: 9m 15s
160: learn: 0.8336683	total: 37m 23s	remaining: 9m 3s
161: learn: 0.8336683	total: 37m 35s	remaining: 8m 49s
162: learn: 0.8336683	total: 37m 41s	remaining: 8m 33s
163: learn: 0.8336683	total: 37m 46s	remaining: 8m 17s
164: learn: 0.8337447	total: 38m 8s	remaining: 8m 5s
165: learn: 0.8337909	total: 38m 30s	remaining: 7m 53s
166: learn: 0.833875	total: 38m 48s	remaining: 7m 40s
167: learn: 0.8339074	total: 39m 8s	remaining: 7m 27s
168: learn: 0.8340146	total: 39m 27s	remaining: 7m 14s
169: learn: 0.8340341	total: 39m 45s	remaining: 7m
170: learn: 0.8340593	total: 40m 1s	remaining: 6m 47s
171: learn: 0.8340742	total: 40m 19s	remaining: 6m 33s
172: learn: 0.8341193	total: 40m 41s	remaining: 6m 21s
173: learn: 0.8341956	total: 41m	remaining: 6m 7s
174: learn: 0.8342571	total: 41m 22s	remaining: 5m 54s
175: learn: 0.8343124	total: 41m 43s	remaining: 5m 41s
176: learn: 0.8343129	total: 42m 2s	remaining: 5m 27s
177: learn: 0.8343375	total: 42m 20s	remaining: 5m 14s
178: learn: 0.8343852	total: 42m 41s	remaining: 5m
179: learn: 0.8344479	total: 43m 4s	remaining: 4m 47s
180: learn: 0.8344625	total: 43m 25s	remaining: 4m 33s
181: learn: 0.8346064	total: 43m 46s	remaining: 4m 19s
182: learn: 0.8346201	total: 44m 4s	remaining: 4m 5s
183: learn: 0.8346499	total: 44m 25s	remaining: 3m 51s
184: learn: 0.8347261	total: 44m 47s	remaining: 3m 37s
185: learn: 0.8347792	total: 45m 10s	remaining: 3m 23s
186: learn: 0.8348295	total: 45m 31s	remaining: 3m 9s
187: learn: 0.8348784	total: 45m 49s	remaining: 2m 55s
188: learn: 0.8348798	total: 46m 3s	remaining: 2m 40s
189: learn: 0.8348823	total: 46m 19s	remaining: 2m 26s
190: learn: 0.8348978	total: 46m 39s	remaining: 2m 11s
191: learn: 0.8349341	total: 46m 56s	remaining: 1m 57s
192: learn: 0.8349348	total: 47m 9s	remaining: 1m 42s
193: learn: 0.8349785	total: 47m 28s	remaining: 1m 28s
194: learn: 0.8350067	total: 47m 48s	remaining: 1m 13s
195: learn: 0.8350128	total: 48m 3s	remaining: 58.8s
196: learn: 0.8350159	total: 48m 15s	remaining: 44.1s
197: learn: 0.8350574	total: 48m 34s	remaining: 29.4s
198: learn: 0.8350971	total: 48m 56s	remaining: 14.8s
199: learn: 0.8351458	total: 49m 16s	remaining: 0us
training complete.
Making prediction
prediction done.
creating submission
[complete] submission name: []_0.6925_cat__1513283740.csv.

[timer]: complete in 185m 14s

Process finished with exit code 0
'''