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
print('What we got:')
print(df.dtypes)
print('number of columns:', len(df.columns))
num_boost_round = 500000
early_stopping_rounds = 1000
verbose_eval = 10
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'verbose': -1,
    'num_leaves': 2**10,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 2**10,
    'max_depth': -1,
}
df = df[['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         # 'fake_liked_song_count'
         ]]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')


print()
print('our guest:')
print()
print(df.dtypes)
print('number of columns:', len(df.columns))
print()
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
del X_tr, Y_tr, X_val, Y_val


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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/drill_train_and_compare_V1001/B_one_round_train_V1001.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
artist_name             object
language              category
dtype: object
number of columns: 8

our guest:

msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
dtype: object
number of columns: 8


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 1000 rounds.
[10]	valid_0's auc: 0.657203
[20]	valid_0's auc: 0.658811
[30]	valid_0's auc: 0.660038
[40]	valid_0's auc: 0.661069
[50]	valid_0's auc: 0.661662
[60]	valid_0's auc: 0.662236
[70]	valid_0's auc: 0.662828
[80]	valid_0's auc: 0.663618
[90]	valid_0's auc: 0.664073
[100]	valid_0's auc: 0.664543
[110]	valid_0's auc: 0.665194
[120]	valid_0's auc: 0.665554
[130]	valid_0's auc: 0.665782
[140]	valid_0's auc: 0.666015
[150]	valid_0's auc: 0.666363
[160]	valid_0's auc: 0.666537
[170]	valid_0's auc: 0.666838
[180]	valid_0's auc: 0.667167
[190]	valid_0's auc: 0.667431
[200]	valid_0's auc: 0.667748
[210]	valid_0's auc: 0.667988
[220]	valid_0's auc: 0.668259
[230]	valid_0's auc: 0.668503
[240]	valid_0's auc: 0.668747
[250]	valid_0's auc: 0.669121
[260]	valid_0's auc: 0.669473
[270]	valid_0's auc: 0.66977
[280]	valid_0's auc: 0.670004
[290]	valid_0's auc: 0.670125
[300]	valid_0's auc: 0.670251
[310]	valid_0's auc: 0.670368
[320]	valid_0's auc: 0.670585
[330]	valid_0's auc: 0.670825
[340]	valid_0's auc: 0.671013
[350]	valid_0's auc: 0.671168
[360]	valid_0's auc: 0.671292
[370]	valid_0's auc: 0.67139
[380]	valid_0's auc: 0.671476
[390]	valid_0's auc: 0.671577
[400]	valid_0's auc: 0.671678
[410]	valid_0's auc: 0.671754
[420]	valid_0's auc: 0.671825
[430]	valid_0's auc: 0.671901
[440]	valid_0's auc: 0.671992
[450]	valid_0's auc: 0.672089
[460]	valid_0's auc: 0.672165
[470]	valid_0's auc: 0.672231
[480]	valid_0's auc: 0.672297
[490]	valid_0's auc: 0.672329
[500]	valid_0's auc: 0.672385
[510]	valid_0's auc: 0.672408
[520]	valid_0's auc: 0.672413
[530]	valid_0's auc: 0.672438
[540]	valid_0's auc: 0.672454
[550]	valid_0's auc: 0.672506
[560]	valid_0's auc: 0.67255
[570]	valid_0's auc: 0.672608
[580]	valid_0's auc: 0.672628
[590]	valid_0's auc: 0.672669
[600]	valid_0's auc: 0.672702
[610]	valid_0's auc: 0.672745
[620]	valid_0's auc: 0.67276
[630]	valid_0's auc: 0.672791
[640]	valid_0's auc: 0.672832
[650]	valid_0's auc: 0.672851
[660]	valid_0's auc: 0.672846
[670]	valid_0's auc: 0.672889
[680]	valid_0's auc: 0.6729
[690]	valid_0's auc: 0.672938
[700]	valid_0's auc: 0.672946
[710]	valid_0's auc: 0.672957
[720]	valid_0's auc: 0.672992
[730]	valid_0's auc: 0.673001
[740]	valid_0's auc: 0.673023
[750]	valid_0's auc: 0.67302
[760]	valid_0's auc: 0.673018
[770]	valid_0's auc: 0.672999
[780]	valid_0's auc: 0.673004
[790]	valid_0's auc: 0.673015
[800]	valid_0's auc: 0.673
[810]	valid_0's auc: 0.672996
[820]	valid_0's auc: 0.672991
[830]	valid_0's auc: 0.672997
[840]	valid_0's auc: 0.672989
[850]	valid_0's auc: 0.672978

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''