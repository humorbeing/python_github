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
    'num_leaves': 100,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 255,
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
         'fake_member_count'
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/0.6246_one_round_train_V1002.py
What we got:
msno                     object
song_id                  object
source_system_tab        object
source_screen_name       object
source_type              object
target                    uint8
fake_member_count         int64
member_count              int64
artist_name              object
language               category
song_count                int64
artist_count              int64
language_count            int64
fake_song_count           int64
fake_artist_count         int64
fake_language_count       int64
dtype: object
number of columns: 16

our guest:

msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_member_count        int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.64732
[20]	valid_0's auc: 0.653785
[30]	valid_0's auc: 0.658854
[40]	valid_0's auc: 0.663185
[50]	valid_0's auc: 0.666229
[60]	valid_0's auc: 0.667959
[70]	valid_0's auc: 0.669482
[80]	valid_0's auc: 0.670514
[90]	valid_0's auc: 0.671211
[100]	valid_0's auc: 0.671823
[110]	valid_0's auc: 0.672309
[120]	valid_0's auc: 0.672819
[130]	valid_0's auc: 0.67303
[140]	valid_0's auc: 0.67329
[150]	valid_0's auc: 0.673446
[160]	valid_0's auc: 0.673802
[170]	valid_0's auc: 0.673977
[180]	valid_0's auc: 0.674113
[190]	valid_0's auc: 0.674162
[200]	valid_0's auc: 0.674157
[210]	valid_0's auc: 0.674181
[220]	valid_0's auc: 0.674229
[230]	valid_0's auc: 0.674365
[240]	valid_0's auc: 0.674439
[250]	valid_0's auc: 0.674561
[260]	valid_0's auc: 0.674587
[270]	valid_0's auc: 0.674629
[280]	valid_0's auc: 0.674691
[290]	valid_0's auc: 0.674659
[300]	valid_0's auc: 0.674683
[310]	valid_0's auc: 0.674683
[320]	valid_0's auc: 0.674708
[330]	valid_0's auc: 0.674764
[340]	valid_0's auc: 0.674782
[350]	valid_0's auc: 0.674777
[360]	valid_0's auc: 0.674775
[370]	valid_0's auc: 0.674867
[380]	valid_0's auc: 0.674838
[390]	valid_0's auc: 0.674838
[400]	valid_0's auc: 0.674879
[410]	valid_0's auc: 0.674879
[420]	valid_0's auc: 0.674867
[430]	valid_0's auc: 0.674909
[440]	valid_0's auc: 0.674916
[450]	valid_0's auc: 0.674898
[460]	valid_0's auc: 0.674893
[470]	valid_0's auc: 0.674864
[480]	valid_0's auc: 0.674902
Early stopping, best iteration is:
[435]	valid_0's auc: 0.674933
best score: 0.674932967897
best iteration: 435

[timer]: complete in 8m 35s

Process finished with exit code 0
'''