import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

barebone = True
# barebone = False
if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('target')
    df.drop(ccc, axis=1, inplace=True)

inner = False


inner = [
    '[0.67982]_0.6788_Light_gbdt_1512750240.csv',
    '[0.62259]_0.6246_Light_gbdt_1512859793.csv'
]


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    df1.rename(columns={'target': on}, inplace=True)
    # print(df1.head(10))
    df = df.join(df1)
    del df1


if inner:
    for i in inner:
        insert_this(i)

print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 5000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 15
max_depth = 10

lambda_l1 = 0
lambda_l2 = 0.3


bagging_fraction = 0.8
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.8
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}
# on = [
#     'msno',
#     'song_id',
#     'target',
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'language',
#     'artist_name',
#     'song_count',
#     'member_count',
#     'song_year',
# ]
# df = df[on]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print('Our guest selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()


length = len(df)
train_size = 0.76
train_set = df.head(int(length*train_size))
val_set = df.drop(train_set.index)
del df

train_set = train_set.sample(frac=1)
X_tr = train_set.drop(['target'], axis=1)
Y_tr = train_set['target'].values

X_val = val_set.drop(['target'], axis=1)
Y_val = val_set['target'].values

del train_set, val_set

t = len(Y_tr)
t1 = sum(Y_tr)
t0 = t - t1
print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
t = len(Y_val)
t1 = sum(Y_val)
t0 = t - t1
print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
print()
print()

train_set = lgb.Dataset(X_tr, Y_tr)
val_set = lgb.Dataset(X_val, Y_val)
# train_set.max_bin = max_bin
# val_set.max_bin = max_bin

del X_tr, Y_tr, X_val, Y_val

params['metric'] = 'auc'
params['verbose'] = -1
params['objective'] = 'binary'

print('Training...')

model = lgb.train(params,
                  train_set,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping_rounds,
                  valid_sets=val_set,
                  verbose_eval=verbose_eval,
                  )

print('best score:', model.best_score['valid_0']['auc'])
print('best iteration:', model.best_iteration)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1006/one_round_train_V1003.py
What we got:
target          uint8
1512750240    float64
1512859793    float64
dtype: object
number of rows: 7377418
number of columns: 3

Our guest selection:
target          uint8
1512750240    float64
1512859793    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.811177
[20]	valid_0's auc: 0.811528
[30]	valid_0's auc: 0.811908
[40]	valid_0's auc: 0.812289
[50]	valid_0's auc: 0.812687
[60]	valid_0's auc: 0.813079
[70]	valid_0's auc: 0.813468
[80]	valid_0's auc: 0.813851
[90]	valid_0's auc: 0.814228
[100]	valid_0's auc: 0.814596
[110]	valid_0's auc: 0.814951
[120]	valid_0's auc: 0.815296
[130]	valid_0's auc: 0.815623
[140]	valid_0's auc: 0.815937
[150]	valid_0's auc: 0.816236
[160]	valid_0's auc: 0.816515
[170]	valid_0's auc: 0.816777
[180]	valid_0's auc: 0.817023
[190]	valid_0's auc: 0.817256
[200]	valid_0's auc: 0.817474
[210]	valid_0's auc: 0.817677
[220]	valid_0's auc: 0.817866
[230]	valid_0's auc: 0.818043
[240]	valid_0's auc: 0.818208
[250]	valid_0's auc: 0.818359
[260]	valid_0's auc: 0.818501
[270]	valid_0's auc: 0.818631
[280]	valid_0's auc: 0.818752
[290]	valid_0's auc: 0.818862
[300]	valid_0's auc: 0.818964
[310]	valid_0's auc: 0.819057
[320]	valid_0's auc: 0.819144
[330]	valid_0's auc: 0.819225
[340]	valid_0's auc: 0.819297
[350]	valid_0's auc: 0.819365
[360]	valid_0's auc: 0.819426
[370]	valid_0's auc: 0.819483
[380]	valid_0's auc: 0.819535
[390]	valid_0's auc: 0.819582
[400]	valid_0's auc: 0.819625
[410]	valid_0's auc: 0.819664
[420]	valid_0's auc: 0.819699
[430]	valid_0's auc: 0.819732
[440]	valid_0's auc: 0.819762
[450]	valid_0's auc: 0.819789
[460]	valid_0's auc: 0.819813
[470]	valid_0's auc: 0.819835
[480]	valid_0's auc: 0.819855
[490]	valid_0's auc: 0.819873
[500]	valid_0's auc: 0.81989
[510]	valid_0's auc: 0.819905
[520]	valid_0's auc: 0.819919
[530]	valid_0's auc: 0.819932
[540]	valid_0's auc: 0.819943
[550]	valid_0's auc: 0.819954
[560]	valid_0's auc: 0.819963
[570]	valid_0's auc: 0.819971
[580]	valid_0's auc: 0.819979
[590]	valid_0's auc: 0.819985
[600]	valid_0's auc: 0.819991
[610]	valid_0's auc: 0.819997
[620]	valid_0's auc: 0.820001
[630]	valid_0's auc: 0.820005
[640]	valid_0's auc: 0.820009
[650]	valid_0's auc: 0.820012
[660]	valid_0's auc: 0.820016
[670]	valid_0's auc: 0.820018
[680]	valid_0's auc: 0.82002
[690]	valid_0's auc: 0.820022
[700]	valid_0's auc: 0.820024
[710]	valid_0's auc: 0.820026
[720]	valid_0's auc: 0.820027
[730]	valid_0's auc: 0.820027
[740]	valid_0's auc: 0.820029
[750]	valid_0's auc: 0.820029
[760]	valid_0's auc: 0.82003
[770]	valid_0's auc: 0.820031
[780]	valid_0's auc: 0.820032
[790]	valid_0's auc: 0.820032
[800]	valid_0's auc: 0.820032
[810]	valid_0's auc: 0.820033
[820]	valid_0's auc: 0.820032
[830]	valid_0's auc: 0.820032
[840]	valid_0's auc: 0.820032
[850]	valid_0's auc: 0.820033
[860]	valid_0's auc: 0.820033
[870]	valid_0's auc: 0.820033
[880]	valid_0's auc: 0.820033
[890]	valid_0's auc: 0.820033
[900]	valid_0's auc: 0.820032
[910]	valid_0's auc: 0.820032
Early stopping, best iteration is:
[861]	valid_0's auc: 0.820033
best score: 0.820033068762
best iteration: 861

[timer]: complete in 7m 45s

Process finished with exit code 0
'''
