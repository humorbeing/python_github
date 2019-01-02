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
    'num_leaves': 127,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 15,
    'max_depth': -1,
}
on = ['msno',
      'song_id',
      'target',
      'source_system_tab',
      'source_screen_name',
      'source_type',
      'language',
      'artist_name',
      'fake_song_count',
      'fake_member_count',
      # new members
      'time',
      'fake_source_type_count',
      # 'fake_artist_count',
      # 'fake_source_system_tab_count',
      # 'fake_source_screen_name_count',
      # 'fake_source_type_count',
      # 'fake_genre_ids_count',
      # 'genre_ids',
      # 'fake_language_count',
      ]
df = df[on]
fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'fake_song_count',
         'fake_member_count',
         ]

for w in df.columns:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df = df[toto]

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype('category')

        print()
        print()
        print('After selection:')
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
        del train_set, val_set
        print('complete on:', w)
        print()
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
        del dt
        df = df[on]

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1002/one_in_column_member_count_int.py
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
fake_genre_type_count               int64
fake_song_count                     int64
fake_artist_count                   int64
fake_genre_ids_count                int64
time                                int64
fake_source_system_tab_count        int64
fake_source_screen_name_count       int64
fake_source_type_count              int64
dtype: object
number of columns: 19
working on: time


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
fake_member_count        int64
time                     int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.655562
[20]	valid_0's auc: 0.663668
[30]	valid_0's auc: 0.669291
[40]	valid_0's auc: 0.673866
[50]	valid_0's auc: 0.676843
[60]	valid_0's auc: 0.678645
[70]	valid_0's auc: 0.679702
[80]	valid_0's auc: 0.680274
[90]	valid_0's auc: 0.680784
[100]	valid_0's auc: 0.680282
[110]	valid_0's auc: 0.679901
[120]	valid_0's auc: 0.679946
[130]	valid_0's auc: 0.679339
[140]	valid_0's auc: 0.678813
Early stopping, best iteration is:
[90]	valid_0's auc: 0.680784
best score: 0.680784176193
best iteration: 90
complete on: time

working on: fake_source_type_count


After selection:
msno                      category
song_id                   category
target                       uint8
source_system_tab         category
source_screen_name        category
source_type               category
language                  category
artist_name               category
fake_song_count              int64
fake_member_count            int64
fake_source_type_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656582
[20]	valid_0's auc: 0.664259
[30]	valid_0's auc: 0.670359
[40]	valid_0's auc: 0.675043
[50]	valid_0's auc: 0.678632
[60]	valid_0's auc: 0.680504
[70]	valid_0's auc: 0.681425
[80]	valid_0's auc: 0.682289
[90]	valid_0's auc: 0.682768
[100]	valid_0's auc: 0.683203
[110]	valid_0's auc: 0.683456
[120]	valid_0's auc: 0.683655
[130]	valid_0's auc: 0.683727
[140]	valid_0's auc: 0.683931
[150]	valid_0's auc: 0.684089
[160]	valid_0's auc: 0.68425
[170]	valid_0's auc: 0.684386
[180]	valid_0's auc: 0.684375
[190]	valid_0's auc: 0.684346
[200]	valid_0's auc: 0.684379
[210]	valid_0's auc: 0.684426
[220]	valid_0's auc: 0.684458
[230]	valid_0's auc: 0.684404
[240]	valid_0's auc: 0.684402
[250]	valid_0's auc: 0.684349
[260]	valid_0's auc: 0.684367
[270]	valid_0's auc: 0.684347
Early stopping, best iteration is:
[221]	valid_0's auc: 0.684496
best score: 0.684496392419
best iteration: 221
complete on: fake_source_type_count


[timer]: complete in 11m 2s

Process finished with exit code 0
'''