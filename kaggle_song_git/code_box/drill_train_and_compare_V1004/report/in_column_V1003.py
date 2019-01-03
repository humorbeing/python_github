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
      # candidate
      # 'fake_artist_count',
      # 'fake_source_screen_name_count',
      # new members
      'fake_genre_type_count',
      'fake_top1',
      'fake_top1_count',
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
fake_top1                        category
fake_top1_count                     int64
fake_source_screen_name_count       int64
dtype: object
number of columns: 17
working on: fake_genre_type_count


After selection:
msno                     category
song_id                  category
target                      uint8
source_system_tab        category
source_screen_name       category
source_type              category
language                 category
artist_name              category
fake_song_count             int64
fake_member_count           int64
fake_genre_type_count       int64
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
[10]	valid_0's auc: 0.656789
[20]	valid_0's auc: 0.664062
[30]	valid_0's auc: 0.67051
[40]	valid_0's auc: 0.674953
[50]	valid_0's auc: 0.678415
[60]	valid_0's auc: 0.680289
[70]	valid_0's auc: 0.681242
[80]	valid_0's auc: 0.682031
[90]	valid_0's auc: 0.682433
[100]	valid_0's auc: 0.682879
[110]	valid_0's auc: 0.683034
[120]	valid_0's auc: 0.683277
[130]	valid_0's auc: 0.683388
[140]	valid_0's auc: 0.683513
[150]	valid_0's auc: 0.683689
[160]	valid_0's auc: 0.683788
[170]	valid_0's auc: 0.68386
[180]	valid_0's auc: 0.683898
[190]	valid_0's auc: 0.683966
[200]	valid_0's auc: 0.6841
[210]	valid_0's auc: 0.684063
[220]	valid_0's auc: 0.684054
[230]	valid_0's auc: 0.684081
[240]	valid_0's auc: 0.68414
[250]	valid_0's auc: 0.684088
[260]	valid_0's auc: 0.684097
[270]	valid_0's auc: 0.684084
[280]	valid_0's auc: 0.684056
[290]	valid_0's auc: 0.684056
Early stopping, best iteration is:
[241]	valid_0's auc: 0.684161
best score: 0.684160974414
best iteration: 241
complete on: fake_genre_type_count

working on: fake_top1


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
fake_top1             category
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.65677
[20]	valid_0's auc: 0.663694
[30]	valid_0's auc: 0.670282
[40]	valid_0's auc: 0.674957
[50]	valid_0's auc: 0.678194
[60]	valid_0's auc: 0.680164
[70]	valid_0's auc: 0.681585
[80]	valid_0's auc: 0.682212
[90]	valid_0's auc: 0.682707
[100]	valid_0's auc: 0.68295
[110]	valid_0's auc: 0.68335
[120]	valid_0's auc: 0.683525
[130]	valid_0's auc: 0.68377
[140]	valid_0's auc: 0.683989
[150]	valid_0's auc: 0.684204
[160]	valid_0's auc: 0.684386
[170]	valid_0's auc: 0.684365
[180]	valid_0's auc: 0.68455
[190]	valid_0's auc: 0.684557
[200]	valid_0's auc: 0.684645
[210]	valid_0's auc: 0.684607
[220]	valid_0's auc: 0.684658
[230]	valid_0's auc: 0.684774
[240]	valid_0's auc: 0.684822
[250]	valid_0's auc: 0.684767
[260]	valid_0's auc: 0.684754
[270]	valid_0's auc: 0.684733
[280]	valid_0's auc: 0.684734
[290]	valid_0's auc: 0.684742
Early stopping, best iteration is:
[242]	valid_0's auc: 0.684835
best score: 0.684834769856
best iteration: 242
complete on: fake_top1

working on: fake_top1_count


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
fake_top1_count          int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656417
[20]	valid_0's auc: 0.663867
[30]	valid_0's auc: 0.670206
[40]	valid_0's auc: 0.675232
[50]	valid_0's auc: 0.678291
[60]	valid_0's auc: 0.680281
[70]	valid_0's auc: 0.681563
[80]	valid_0's auc: 0.682337
[90]	valid_0's auc: 0.682873
[100]	valid_0's auc: 0.683117
[110]	valid_0's auc: 0.683245
[120]	valid_0's auc: 0.683358
[130]	valid_0's auc: 0.683552
[140]	valid_0's auc: 0.68366
[150]	valid_0's auc: 0.683805
[160]	valid_0's auc: 0.683998
[170]	valid_0's auc: 0.683993
[180]	valid_0's auc: 0.684048
[190]	valid_0's auc: 0.684173
[200]	valid_0's auc: 0.684091
[210]	valid_0's auc: 0.684118
[220]	valid_0's auc: 0.684127
[230]	valid_0's auc: 0.684093
Early stopping, best iteration is:
[189]	valid_0's auc: 0.684189
best score: 0.684189374992
best iteration: 189
complete on: fake_top1_count


[timer]: complete in 20m 18s

Process finished with exit code 0
'''
