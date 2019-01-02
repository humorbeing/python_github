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
         'fake_song_count',
         # 'fake_artist_count',
         'fake_member_count',
         'fake_source_system_tab_count',
         'fake_source_screen_name_count',
         'fake_source_type_count',
         'fake_genre_ids_count',
         'genre_ids',
         'fake_artist_count',
         # 'fake_language_count',
         ]]
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
        df = df[['msno',
                 'song_id',
                 'target',
                 'source_system_tab',
                 'source_screen_name',
                 'source_type',
                 'language',
                 'artist_name',
                 'fake_song_count',
                 'fake_artist_count',
                 'fake_member_count',
                 'fake_source_system_tab_count',
                 'fake_source_screen_name_count',
                 'fake_source_type_count',
                 'fake_genre_ids_count',
                 'genre_ids',
                 'fake_artist_count',
                 # 'fake_language_count',
                 ]]

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_trainer_V1004.py
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
working on: fake_source_system_tab_count


After selection:
msno                            category
song_id                         category
target                             uint8
source_system_tab               category
source_screen_name              category
source_type                     category
language                        category
artist_name                     category
fake_song_count                    int64
fake_member_count                  int64
fake_source_system_tab_count       int64
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
[10]	valid_0's auc: 0.654455
[20]	valid_0's auc: 0.661795
[30]	valid_0's auc: 0.668014
[40]	valid_0's auc: 0.672875
[50]	valid_0's auc: 0.676685
[60]	valid_0's auc: 0.679104
[70]	valid_0's auc: 0.680228
[80]	valid_0's auc: 0.681058
[90]	valid_0's auc: 0.681685
[100]	valid_0's auc: 0.682238
[110]	valid_0's auc: 0.682396
[120]	valid_0's auc: 0.682868
[130]	valid_0's auc: 0.683207
[140]	valid_0's auc: 0.683379
[150]	valid_0's auc: 0.683621
[160]	valid_0's auc: 0.683768
[170]	valid_0's auc: 0.683822
[180]	valid_0's auc: 0.683909
[190]	valid_0's auc: 0.683984
[200]	valid_0's auc: 0.684024
[210]	valid_0's auc: 0.684025
[220]	valid_0's auc: 0.684116
[230]	valid_0's auc: 0.684132
[240]	valid_0's auc: 0.684207
[250]	valid_0's auc: 0.684247
[260]	valid_0's auc: 0.684237
[270]	valid_0's auc: 0.684285
[280]	valid_0's auc: 0.684348
[290]	valid_0's auc: 0.684373
[300]	valid_0's auc: 0.684395
[310]	valid_0's auc: 0.68436
[320]	valid_0's auc: 0.684322
[330]	valid_0's auc: 0.684286
[340]	valid_0's auc: 0.684285
[350]	valid_0's auc: 0.684302
Early stopping, best iteration is:
[301]	valid_0's auc: 0.684402
best score: 0.684402155248
best iteration: 301
complete on: fake_source_system_tab_count

working on: fake_source_screen_name_count


After selection:
msno                             category
song_id                          category
target                              uint8
source_system_tab                category
source_screen_name               category
source_type                      category
language                         category
artist_name                      category
fake_song_count                     int64
fake_member_count                   int64
fake_source_screen_name_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.654388
[20]	valid_0's auc: 0.661643
[30]	valid_0's auc: 0.668679
[40]	valid_0's auc: 0.673597
[50]	valid_0's auc: 0.676959
[60]	valid_0's auc: 0.679368
[70]	valid_0's auc: 0.680567
[80]	valid_0's auc: 0.681374
[90]	valid_0's auc: 0.682124
[100]	valid_0's auc: 0.682699
[110]	valid_0's auc: 0.682949
[120]	valid_0's auc: 0.683408
[130]	valid_0's auc: 0.683604
[140]	valid_0's auc: 0.683738
[150]	valid_0's auc: 0.683958
[160]	valid_0's auc: 0.684206
[170]	valid_0's auc: 0.684302
[180]	valid_0's auc: 0.684385
[190]	valid_0's auc: 0.684436
[200]	valid_0's auc: 0.684548
[210]	valid_0's auc: 0.684542
[220]	valid_0's auc: 0.684723
[230]	valid_0's auc: 0.684796
[240]	valid_0's auc: 0.684858
[250]	valid_0's auc: 0.68497
[260]	valid_0's auc: 0.684956
[270]	valid_0's auc: 0.684993
[280]	valid_0's auc: 0.685058
[290]	valid_0's auc: 0.685181
[300]	valid_0's auc: 0.685219
[310]	valid_0's auc: 0.685263
[320]	valid_0's auc: 0.685283
[330]	valid_0's auc: 0.68528
[340]	valid_0's auc: 0.685283
[350]	valid_0's auc: 0.685325
[360]	valid_0's auc: 0.685282
[370]	valid_0's auc: 0.685287
[380]	valid_0's auc: 0.685262
[390]	valid_0's auc: 0.685274
Early stopping, best iteration is:
[346]	valid_0's auc: 0.685332
best score: 0.68533195436
best iteration: 346
complete on: fake_source_screen_name_count

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
[10]	valid_0's auc: 0.654282
[20]	valid_0's auc: 0.661392
[30]	valid_0's auc: 0.668384
[40]	valid_0's auc: 0.672938
[50]	valid_0's auc: 0.676704
[60]	valid_0's auc: 0.679043
[70]	valid_0's auc: 0.680573
[80]	valid_0's auc: 0.681369
[90]	valid_0's auc: 0.681976
[100]	valid_0's auc: 0.682624
[110]	valid_0's auc: 0.683099
[120]	valid_0's auc: 0.683393
[130]	valid_0's auc: 0.683664
[140]	valid_0's auc: 0.683926
[150]	valid_0's auc: 0.68407
[160]	valid_0's auc: 0.684189
[170]	valid_0's auc: 0.684301
[180]	valid_0's auc: 0.684465
[190]	valid_0's auc: 0.684522
[200]	valid_0's auc: 0.684531
[210]	valid_0's auc: 0.684658
[220]	valid_0's auc: 0.684736
[230]	valid_0's auc: 0.684805
[240]	valid_0's auc: 0.684801
[250]	valid_0's auc: 0.684792
[260]	valid_0's auc: 0.684808
[270]	valid_0's auc: 0.684779
[280]	valid_0's auc: 0.68483
Early stopping, best iteration is:
[236]	valid_0's auc: 0.684834
best score: 0.684833813909
best iteration: 236
complete on: fake_source_type_count

working on: fake_genre_ids_count


After selection:
msno                    category
song_id                 category
target                     uint8
source_system_tab       category
source_screen_name      category
source_type             category
language                category
artist_name             category
fake_song_count            int64
fake_member_count          int64
fake_genre_ids_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.654551
[20]	valid_0's auc: 0.661716
[30]	valid_0's auc: 0.668233
[40]	valid_0's auc: 0.673099
[50]	valid_0's auc: 0.676779
[60]	valid_0's auc: 0.678523
[70]	valid_0's auc: 0.680028
[80]	valid_0's auc: 0.68102
[90]	valid_0's auc: 0.681722
[100]	valid_0's auc: 0.682253
[110]	valid_0's auc: 0.682813
[120]	valid_0's auc: 0.683039
[130]	valid_0's auc: 0.683382
[140]	valid_0's auc: 0.683656
[150]	valid_0's auc: 0.683806
[160]	valid_0's auc: 0.684018
[170]	valid_0's auc: 0.684251
[180]	valid_0's auc: 0.684284
[190]	valid_0's auc: 0.684427
[200]	valid_0's auc: 0.684482
[210]	valid_0's auc: 0.6845
[220]	valid_0's auc: 0.684548
[230]	valid_0's auc: 0.684568
[240]	valid_0's auc: 0.684598
[250]	valid_0's auc: 0.684647
[260]	valid_0's auc: 0.684752
[270]	valid_0's auc: 0.684811
[280]	valid_0's auc: 0.684783
[290]	valid_0's auc: 0.684869
[300]	valid_0's auc: 0.684946
[310]	valid_0's auc: 0.68499
[320]	valid_0's auc: 0.685011
[330]	valid_0's auc: 0.685006
[340]	valid_0's auc: 0.684963
[350]	valid_0's auc: 0.684981
[360]	valid_0's auc: 0.68496
[370]	valid_0's auc: 0.68498
Early stopping, best iteration is:
[326]	valid_0's auc: 0.685026
best score: 0.685026276789
best iteration: 326
complete on: fake_genre_ids_count

working on: genre_ids


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
genre_ids             category
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.65453
[20]	valid_0's auc: 0.662114
[30]	valid_0's auc: 0.668907
[40]	valid_0's auc: 0.673863
[50]	valid_0's auc: 0.67721
[60]	valid_0's auc: 0.679222
[70]	valid_0's auc: 0.680634
[80]	valid_0's auc: 0.681501
[90]	valid_0's auc: 0.682366
[100]	valid_0's auc: 0.682722
[110]	valid_0's auc: 0.683168
[120]	valid_0's auc: 0.683462
[130]	valid_0's auc: 0.683653
[140]	valid_0's auc: 0.683912
[150]	valid_0's auc: 0.684084
[160]	valid_0's auc: 0.684175
[170]	valid_0's auc: 0.684347
[180]	valid_0's auc: 0.68444
[190]	valid_0's auc: 0.684532
[200]	valid_0's auc: 0.684616
[210]	valid_0's auc: 0.684648
[220]	valid_0's auc: 0.684773
[230]	valid_0's auc: 0.684847
[240]	valid_0's auc: 0.684847
[250]	valid_0's auc: 0.684913
[260]	valid_0's auc: 0.684975
[270]	valid_0's auc: 0.684972
[280]	valid_0's auc: 0.685019
[290]	valid_0's auc: 0.684975
[300]	valid_0's auc: 0.685014
[310]	valid_0's auc: 0.684991
[320]	valid_0's auc: 0.685054
[330]	valid_0's auc: 0.685068
[340]	valid_0's auc: 0.685071
[350]	valid_0's auc: 0.685088
[360]	valid_0's auc: 0.685066
[370]	valid_0's auc: 0.685044
[380]	valid_0's auc: 0.684997
[390]	valid_0's auc: 0.685006
Early stopping, best iteration is:
[346]	valid_0's auc: 0.685104
best score: 0.685104098387
best iteration: 346
complete on: genre_ids

working on: fake_artist_count
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_trainer_V1004.py", line 83, in <module>
    if df[col].dtype == object:
  File "/usr/local/lib/python3.5/dist-packages/pandas/core/generic.py", line 3081, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'dtype'

Process finished with exit code 1
'''