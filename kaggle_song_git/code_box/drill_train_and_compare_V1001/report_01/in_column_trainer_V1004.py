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
         'fake_song_count',
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

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_trainer_V1004.py
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
fake_song_count          int64
fake_artist_count        int64
dtype: object
number of columns: 10


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650206
[20]	valid_0's auc: 0.658113
[30]	valid_0's auc: 0.665007
[40]	valid_0's auc: 0.669849
[50]	valid_0's auc: 0.673005
[60]	valid_0's auc: 0.675189
[70]	valid_0's auc: 0.676438
[80]	valid_0's auc: 0.677804
[90]	valid_0's auc: 0.678665
[100]	valid_0's auc: 0.679334
[110]	valid_0's auc: 0.679821
[120]	valid_0's auc: 0.680013
[130]	valid_0's auc: 0.680204
[140]	valid_0's auc: 0.680534
[150]	valid_0's auc: 0.680771
[160]	valid_0's auc: 0.680834
[170]	valid_0's auc: 0.681055
[180]	valid_0's auc: 0.681266
[190]	valid_0's auc: 0.681353
[200]	valid_0's auc: 0.681592
[210]	valid_0's auc: 0.681715
[220]	valid_0's auc: 0.681741
[230]	valid_0's auc: 0.681874
[240]	valid_0's auc: 0.681915
[250]	valid_0's auc: 0.681938
[260]	valid_0's auc: 0.681974
[270]	valid_0's auc: 0.682077
[280]	valid_0's auc: 0.682003
[290]	valid_0's auc: 0.682062
[300]	valid_0's auc: 0.682111
[310]	valid_0's auc: 0.682164
[320]	valid_0's auc: 0.682088
[330]	valid_0's auc: 0.682059
[340]	valid_0's auc: 0.682051
[350]	valid_0's auc: 0.682039
[360]	valid_0's auc: 0.682045
Early stopping, best iteration is:
[310]	valid_0's auc: 0.682164
best score: 0.682164413482
best iteration: 310
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
fake_song_count          int64
fake_member_count        int64
dtype: object
number of columns: 10


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.654307
[20]	valid_0's auc: 0.661787
[30]	valid_0's auc: 0.668684
[40]	valid_0's auc: 0.673484
[50]	valid_0's auc: 0.67688
[60]	valid_0's auc: 0.679177
[70]	valid_0's auc: 0.680817
[80]	valid_0's auc: 0.681808
[90]	valid_0's auc: 0.682456
[100]	valid_0's auc: 0.682952
[110]	valid_0's auc: 0.683325
[120]	valid_0's auc: 0.683601
[130]	valid_0's auc: 0.683867
[140]	valid_0's auc: 0.684044
[150]	valid_0's auc: 0.684329
[160]	valid_0's auc: 0.684414
[170]	valid_0's auc: 0.684453
[180]	valid_0's auc: 0.684451
[190]	valid_0's auc: 0.684483
[200]	valid_0's auc: 0.684598
[210]	valid_0's auc: 0.684649
[220]	valid_0's auc: 0.684768
[230]	valid_0's auc: 0.684751
[240]	valid_0's auc: 0.684837
[250]	valid_0's auc: 0.684895
[260]	valid_0's auc: 0.684921
[270]	valid_0's auc: 0.684929
[280]	valid_0's auc: 0.684971
[290]	valid_0's auc: 0.684977
[300]	valid_0's auc: 0.685017
[310]	valid_0's auc: 0.685019
[320]	valid_0's auc: 0.684984
[330]	valid_0's auc: 0.684978
[340]	valid_0's auc: 0.684983
[350]	valid_0's auc: 0.684986
Early stopping, best iteration is:
[307]	valid_0's auc: 0.685051
best score: 0.685051083788
best iteration: 307
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
fake_song_count           int64
fake_language_count       int64
dtype: object
number of columns: 10


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650968
[20]	valid_0's auc: 0.657817
[30]	valid_0's auc: 0.664716
[40]	valid_0's auc: 0.669737
[50]	valid_0's auc: 0.673303
[60]	valid_0's auc: 0.675503
[70]	valid_0's auc: 0.676967
[80]	valid_0's auc: 0.678007
[90]	valid_0's auc: 0.678693
[100]	valid_0's auc: 0.679214
[110]	valid_0's auc: 0.679875
[120]	valid_0's auc: 0.680188
[130]	valid_0's auc: 0.680649
[140]	valid_0's auc: 0.680855
[150]	valid_0's auc: 0.681019
[160]	valid_0's auc: 0.681192
[170]	valid_0's auc: 0.681392
[180]	valid_0's auc: 0.681439
[190]	valid_0's auc: 0.681513
[200]	valid_0's auc: 0.681626
[210]	valid_0's auc: 0.681747
[220]	valid_0's auc: 0.681776
[230]	valid_0's auc: 0.681786
[240]	valid_0's auc: 0.681809
[250]	valid_0's auc: 0.681822
[260]	valid_0's auc: 0.681808
[270]	valid_0's auc: 0.681835
[280]	valid_0's auc: 0.681899
[290]	valid_0's auc: 0.681876
[300]	valid_0's auc: 0.681894
[310]	valid_0's auc: 0.681902
[320]	valid_0's auc: 0.681906
[330]	valid_0's auc: 0.682077
[340]	valid_0's auc: 0.682061
[350]	valid_0's auc: 0.682021
[360]	valid_0's auc: 0.682004
[370]	valid_0's auc: 0.681957
[380]	valid_0's auc: 0.681958
Early stopping, best iteration is:
[331]	valid_0's auc: 0.682088
best score: 0.682088233134
best iteration: 331
complete on: fake_language_count


[timer]: complete in 23m 56s

Process finished with exit code 0
'''
