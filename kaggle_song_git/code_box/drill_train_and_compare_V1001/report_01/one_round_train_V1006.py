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
early_stopping_rounds = 50
verbose_eval = 10
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'verbose': -1,
    'num_leaves': 511,

    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 255,
    'max_depth': -1,
}
df = df[[
         # 'msno',
         # 'song_id',
         'target',
         # 'source_system_tab',
         # 'source_screen_name',
         # 'source_type',
         # 'language',
         # 'artist_name',
         'fake_song_count',
         'fake_member_count',
         'fake_artist_count',
         'fake_language_count',
         'fake_genre_ids_count',
         'fake_source_system_tab_count',
         'fake_source_screen_name_count',
         'fake_source_type_count'
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_train_V1001.py
What we got:
msno                               object
song_id                            object
source_system_tab                  object
source_screen_name                 object
source_type                        object
target                              uint8
fake_member_count                   int64
member_count                        int64
genre_ids                          object
artist_name                        object
language                         category
fake_song_count                     int64
fake_artist_count                   int64
fake_language_count                 int64
fake_genre_ids_count                int64
fake_source_system_tab_count        int64
fake_source_screen_name_count       int64
fake_source_type_count              int64
dtype: object
number of columns: 18

our guest:

target                           uint8
fake_song_count                  int64
fake_member_count                int64
fake_artist_count                int64
fake_language_count              int64
fake_genre_ids_count             int64
fake_source_system_tab_count     int64
fake_source_screen_name_count    int64
fake_source_type_count           int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642745
[20]	valid_0's auc: 0.64433
[30]	valid_0's auc: 0.64576
[40]	valid_0's auc: 0.646888
[50]	valid_0's auc: 0.647924
[60]	valid_0's auc: 0.648665
[70]	valid_0's auc: 0.649093
[80]	valid_0's auc: 0.649618
[90]	valid_0's auc: 0.650147
[100]	valid_0's auc: 0.65031
[110]	valid_0's auc: 0.650396
[120]	valid_0's auc: 0.650508
[130]	valid_0's auc: 0.650549
[140]	valid_0's auc: 0.650665
[150]	valid_0's auc: 0.650729
[160]	valid_0's auc: 0.650785
[170]	valid_0's auc: 0.650904
[180]	valid_0's auc: 0.650908
[190]	valid_0's auc: 0.651047
[200]	valid_0's auc: 0.651079
[210]	valid_0's auc: 0.651197
[220]	valid_0's auc: 0.651229
[230]	valid_0's auc: 0.651254
[240]	valid_0's auc: 0.651338
[250]	valid_0's auc: 0.651333
[260]	valid_0's auc: 0.651355
[270]	valid_0's auc: 0.651436
[280]	valid_0's auc: 0.651494
[290]	valid_0's auc: 0.651416
[300]	valid_0's auc: 0.651443
[310]	valid_0's auc: 0.651474
[320]	valid_0's auc: 0.651497
[330]	valid_0's auc: 0.651524
[340]	valid_0's auc: 0.651554
[350]	valid_0's auc: 0.651537
[360]	valid_0's auc: 0.651536
[370]	valid_0's auc: 0.65153
[380]	valid_0's auc: 0.65153
[390]	valid_0's auc: 0.65152
Early stopping, best iteration is:
[342]	valid_0's auc: 0.651565
best score: 0.651565372171
best iteration: 342

[timer]: complete in 4m 35s

Process finished with exit code 0
'''