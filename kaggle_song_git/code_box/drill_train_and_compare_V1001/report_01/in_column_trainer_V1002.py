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
         'fake_artist_count',
         'fake_member_count',
         'fake_language_count',
         ]]
fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
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
                 'fake_language_count',
                 ]]

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_trainer_V1002.py
What we got:
msno                            object
song_id                         object
source_system_tab               object
source_screen_name              object
source_type                     object
target                           uint8
artist_name                     object
language                      category
fake_song_count                  int64
fake_liked_song_count            int64
fake_disliked_song_count         int64
fake_artist_count                int64
fake_liked_artist_count          int64
fake_disliked_artist_count       int64
fake_member_count                int64
fake_liked_member_count          int64
fake_disliked_member_count       int64
fake_language_count              int64
dtype: object
number of columns: 18
working on: fake_song_count


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
[10]	valid_0's auc: 0.650608
[20]	valid_0's auc: 0.658385
[30]	valid_0's auc: 0.664199
[40]	valid_0's auc: 0.669777
[50]	valid_0's auc: 0.673191
[60]	valid_0's auc: 0.675198
[70]	valid_0's auc: 0.676747
[80]	valid_0's auc: 0.677779
[90]	valid_0's auc: 0.678702
[100]	valid_0's auc: 0.679215
[110]	valid_0's auc: 0.679641
[120]	valid_0's auc: 0.680114
[130]	valid_0's auc: 0.680304
[140]	valid_0's auc: 0.680482
[150]	valid_0's auc: 0.680695
[160]	valid_0's auc: 0.680743
[170]	valid_0's auc: 0.680984
[180]	valid_0's auc: 0.68111
[190]	valid_0's auc: 0.681297
[200]	valid_0's auc: 0.681447
[210]	valid_0's auc: 0.681454
[220]	valid_0's auc: 0.681674
[230]	valid_0's auc: 0.681724
[240]	valid_0's auc: 0.68185
[250]	valid_0's auc: 0.682039
[260]	valid_0's auc: 0.682067
[270]	valid_0's auc: 0.682065
[280]	valid_0's auc: 0.682064
[290]	valid_0's auc: 0.682082
[300]	valid_0's auc: 0.682084
[310]	valid_0's auc: 0.682121
[320]	valid_0's auc: 0.682104
[330]	valid_0's auc: 0.682086
[340]	valid_0's auc: 0.682057
[350]	valid_0's auc: 0.682119
[360]	valid_0's auc: 0.682097
[370]	valid_0's auc: 0.682085
[380]	valid_0's auc: 0.682078
[390]	valid_0's auc: 0.682052
[400]	valid_0's auc: 0.682049
Early stopping, best iteration is:
[356]	valid_0's auc: 0.682133
best score: 0.682133480211
best iteration: 356
complete on: fake_song_count

working on: fake_artist_count


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_artist_count        int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642794
[20]	valid_0's auc: 0.648496
[30]	valid_0's auc: 0.654887
[40]	valid_0's auc: 0.659138
[50]	valid_0's auc: 0.662173
[60]	valid_0's auc: 0.664282
[70]	valid_0's auc: 0.665863
[80]	valid_0's auc: 0.666922
[90]	valid_0's auc: 0.667708
[100]	valid_0's auc: 0.668252
[110]	valid_0's auc: 0.668799
[120]	valid_0's auc: 0.669091
[130]	valid_0's auc: 0.669351
[140]	valid_0's auc: 0.66981
[150]	valid_0's auc: 0.669966
[160]	valid_0's auc: 0.670148
[170]	valid_0's auc: 0.670248
[180]	valid_0's auc: 0.670299
[190]	valid_0's auc: 0.670348
[200]	valid_0's auc: 0.670415
[210]	valid_0's auc: 0.670512
[220]	valid_0's auc: 0.670633
[230]	valid_0's auc: 0.670685
[240]	valid_0's auc: 0.670732
[250]	valid_0's auc: 0.670815
[260]	valid_0's auc: 0.670834
[270]	valid_0's auc: 0.670869
[280]	valid_0's auc: 0.670863
[290]	valid_0's auc: 0.670915
[300]	valid_0's auc: 0.670951
[310]	valid_0's auc: 0.670998
[320]	valid_0's auc: 0.671047
[330]	valid_0's auc: 0.671051
[340]	valid_0's auc: 0.671034
[350]	valid_0's auc: 0.671053
[360]	valid_0's auc: 0.671095
[370]	valid_0's auc: 0.671105
[380]	valid_0's auc: 0.671183
[390]	valid_0's auc: 0.671164
[400]	valid_0's auc: 0.671154
[410]	valid_0's auc: 0.671161
[420]	valid_0's auc: 0.671164
[430]	valid_0's auc: 0.671132
Early stopping, best iteration is:
[383]	valid_0's auc: 0.67119
best score: 0.671190027266
best iteration: 383
complete on: fake_artist_count

working on: fake_member_count


After selection:
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
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.644515
[20]	valid_0's auc: 0.650929
[30]	valid_0's auc: 0.65613
[40]	valid_0's auc: 0.660658
[50]	valid_0's auc: 0.66368
[60]	valid_0's auc: 0.665679
[70]	valid_0's auc: 0.666973
[80]	valid_0's auc: 0.667715
[90]	valid_0's auc: 0.668551
[100]	valid_0's auc: 0.669285
[110]	valid_0's auc: 0.669845
[120]	valid_0's auc: 0.670311
[130]	valid_0's auc: 0.670616
[140]	valid_0's auc: 0.670732
[150]	valid_0's auc: 0.671014
[160]	valid_0's auc: 0.671098
[170]	valid_0's auc: 0.671233
[180]	valid_0's auc: 0.671365
[190]	valid_0's auc: 0.671512
[200]	valid_0's auc: 0.671588
[210]	valid_0's auc: 0.671609
[220]	valid_0's auc: 0.671615
[230]	valid_0's auc: 0.671605
[240]	valid_0's auc: 0.671636
[250]	valid_0's auc: 0.67168
[260]	valid_0's auc: 0.671731
[270]	valid_0's auc: 0.671759
[280]	valid_0's auc: 0.671777
[290]	valid_0's auc: 0.671802
[300]	valid_0's auc: 0.671831
[310]	valid_0's auc: 0.671873
[320]	valid_0's auc: 0.67189
[330]	valid_0's auc: 0.67192
[340]	valid_0's auc: 0.67195
[350]	valid_0's auc: 0.671926
[360]	valid_0's auc: 0.671903
[370]	valid_0's auc: 0.67191
[380]	valid_0's auc: 0.671913
[390]	valid_0's auc: 0.671891
Early stopping, best iteration is:
[341]	valid_0's auc: 0.671956
best score: 0.671955526257
best iteration: 341
complete on: fake_member_count

working on: fake_language_count


After selection:
msno                   category
song_id                category
target                    uint8
source_system_tab      category
source_screen_name     category
source_type            category
language               category
artist_name            category
fake_language_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.64435
[20]	valid_0's auc: 0.650545
[30]	valid_0's auc: 0.656514
[40]	valid_0's auc: 0.661133
[50]	valid_0's auc: 0.663931
[60]	valid_0's auc: 0.666292
[70]	valid_0's auc: 0.667279
[80]	valid_0's auc: 0.668207
[90]	valid_0's auc: 0.669226
[100]	valid_0's auc: 0.669816
[110]	valid_0's auc: 0.670316
[120]	valid_0's auc: 0.670761
[130]	valid_0's auc: 0.671008
[140]	valid_0's auc: 0.671161
[150]	valid_0's auc: 0.671411
[160]	valid_0's auc: 0.671574
[170]	valid_0's auc: 0.671692
[180]	valid_0's auc: 0.671809
[190]	valid_0's auc: 0.671917
[200]	valid_0's auc: 0.671977
[210]	valid_0's auc: 0.671985
[220]	valid_0's auc: 0.67215
[230]	valid_0's auc: 0.672156
[240]	valid_0's auc: 0.672191
[250]	valid_0's auc: 0.672197
[260]	valid_0's auc: 0.672196
[270]	valid_0's auc: 0.672225
[280]	valid_0's auc: 0.672253
[290]	valid_0's auc: 0.672303
[300]	valid_0's auc: 0.67233
[310]	valid_0's auc: 0.672377
[320]	valid_0's auc: 0.672402
[330]	valid_0's auc: 0.672352
[340]	valid_0's auc: 0.672383
[350]	valid_0's auc: 0.672358
[360]	valid_0's auc: 0.672389
Early stopping, best iteration is:
[314]	valid_0's auc: 0.672417
best score: 0.672417348845
best iteration: 314
complete on: fake_language_count


[timer]: complete in 31m 36s

Process finished with exit code 0
'''
