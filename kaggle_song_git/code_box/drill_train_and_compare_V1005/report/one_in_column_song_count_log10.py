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
print('number of rows:', len(df))
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
    'num_leaves': 2**6-1,
    'max_depth': 5,
}

fixed = [
         'target',
]
cols = df.columns
result = {}
for w in cols:
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
        result[w] = model.best_score['valid_0']['auc']
        print()
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
        del dt


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))
# reversed(sorted_x)
# print(sorted_x)
for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1005/one_in_column_V1002.py
What we got:
target                       uint8
song_length_log10          float64
song_year_log10            float64
ISC_genre_ids_log10        float64
ISCZ_genre_ids_log10       float64
ISC_top1_in_song_log10     float64
ISC_top2_in_song_log10     float64
ISC_top3_in_song_log10     float64
ISC_artist_name_log10      float64
ISCZ_artist_name_log10     float64
ISC_composer_log10         float64
ISCZ_composer_log10        float64
ISC_lyricist_log10         float64
ISCZ_lyricist_log10        float64
ISC_name_log10             float64
ISCZ_name_log10            float64
ISC_language_log10         float64
ISC_isrc_log10             float64
ISCZ_isrc_log10            float64
ISC_song_country_log10     float64
ISCZ_song_country_log10    float64
ISC_rc_log10               float64
ISCZ_rc_log10              float64
ISC_isrc_rest_log10        float64
ISCZ_isrc_rest_log10       float64
ISC_song_year_log10        float64
ISCZ_song_year_log10       float64
dtype: object
number of rows: 7377418
number of columns: 27
working on: song_length_log10


After selection:
target                 uint8
song_length_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.514802
[20]	valid_0's auc: 0.521633
[30]	valid_0's auc: 0.525816
[40]	valid_0's auc: 0.527664
[50]	valid_0's auc: 0.528743
[60]	valid_0's auc: 0.529469
[70]	valid_0's auc: 0.529737
[80]	valid_0's auc: 0.530436
[90]	valid_0's auc: 0.530596
[100]	valid_0's auc: 0.530885
[110]	valid_0's auc: 0.531061
[120]	valid_0's auc: 0.531042
[130]	valid_0's auc: 0.531097
[140]	valid_0's auc: 0.530916
[150]	valid_0's auc: 0.530843
[160]	valid_0's auc: 0.530699
Early stopping, best iteration is:
[117]	valid_0's auc: 0.531174
best score: 0.531174076926
best iteration: 117
complete on: song_length_log10

working on: song_year_log10


After selection:
target               uint8
song_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.538058
[20]	valid_0's auc: 0.53806
[30]	valid_0's auc: 0.538055
[40]	valid_0's auc: 0.538055
[50]	valid_0's auc: 0.538055
[60]	valid_0's auc: 0.538055
[70]	valid_0's auc: 0.538055
Early stopping, best iteration is:
[23]	valid_0's auc: 0.53806
best score: 0.538060451128
best iteration: 23
complete on: song_year_log10

working on: ISC_genre_ids_log10


After selection:
target                   uint8
ISC_genre_ids_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526476
[20]	valid_0's auc: 0.526527
[30]	valid_0's auc: 0.527078
[40]	valid_0's auc: 0.527213
[50]	valid_0's auc: 0.527297
[60]	valid_0's auc: 0.52731
[70]	valid_0's auc: 0.527626
[80]	valid_0's auc: 0.527554
[90]	valid_0's auc: 0.527571
[100]	valid_0's auc: 0.527587
[110]	valid_0's auc: 0.527585
Early stopping, best iteration is:
[69]	valid_0's auc: 0.527627
best score: 0.527626686412
best iteration: 69
complete on: ISC_genre_ids_log10

working on: ISCZ_genre_ids_log10


After selection:
target                    uint8
ISCZ_genre_ids_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525733
[20]	valid_0's auc: 0.526845
[30]	valid_0's auc: 0.526986
[40]	valid_0's auc: 0.527195
[50]	valid_0's auc: 0.52716
[60]	valid_0's auc: 0.527212
[70]	valid_0's auc: 0.527204
[80]	valid_0's auc: 0.527225
[90]	valid_0's auc: 0.527247
[100]	valid_0's auc: 0.527545
[110]	valid_0's auc: 0.527562
[120]	valid_0's auc: 0.527564
[130]	valid_0's auc: 0.527567
[140]	valid_0's auc: 0.527577
[150]	valid_0's auc: 0.527573
[160]	valid_0's auc: 0.527609
[170]	valid_0's auc: 0.527616
[180]	valid_0's auc: 0.527622
[190]	valid_0's auc: 0.527622
[200]	valid_0's auc: 0.527677
[210]	valid_0's auc: 0.527675
[220]	valid_0's auc: 0.527675
[230]	valid_0's auc: 0.52769
[240]	valid_0's auc: 0.527691
[250]	valid_0's auc: 0.527666
[260]	valid_0's auc: 0.52766
[270]	valid_0's auc: 0.52766
[280]	valid_0's auc: 0.527673
Early stopping, best iteration is:
[237]	valid_0's auc: 0.527691
best score: 0.527690851179
best iteration: 237
complete on: ISCZ_genre_ids_log10

working on: ISC_top1_in_song_log10


After selection:
target                      uint8
ISC_top1_in_song_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.52572
[20]	valid_0's auc: 0.52578
[30]	valid_0's auc: 0.525857
[40]	valid_0's auc: 0.526202
[50]	valid_0's auc: 0.526064
[60]	valid_0's auc: 0.526076
[70]	valid_0's auc: 0.526074
[80]	valid_0's auc: 0.526146
[90]	valid_0's auc: 0.526171
Early stopping, best iteration is:
[44]	valid_0's auc: 0.52621
best score: 0.526209798343
best iteration: 44
complete on: ISC_top1_in_song_log10

working on: ISC_top2_in_song_log10


After selection:
target                      uint8
ISC_top2_in_song_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.527637
[20]	valid_0's auc: 0.528206
[30]	valid_0's auc: 0.52793
[40]	valid_0's auc: 0.527937
[50]	valid_0's auc: 0.527992
[60]	valid_0's auc: 0.528312
[70]	valid_0's auc: 0.528333
[80]	valid_0's auc: 0.528329
[90]	valid_0's auc: 0.52833
[100]	valid_0's auc: 0.52833
[110]	valid_0's auc: 0.528384
[120]	valid_0's auc: 0.528384
[130]	valid_0's auc: 0.52838
[140]	valid_0's auc: 0.52838
[150]	valid_0's auc: 0.528379
Early stopping, best iteration is:
[107]	valid_0's auc: 0.528387
best score: 0.528386874735
best iteration: 107
complete on: ISC_top2_in_song_log10

working on: ISC_top3_in_song_log10


After selection:
target                      uint8
ISC_top3_in_song_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.523101
[20]	valid_0's auc: 0.523668
[30]	valid_0's auc: 0.523675
[40]	valid_0's auc: 0.523716
[50]	valid_0's auc: 0.523737
[60]	valid_0's auc: 0.523741
[70]	valid_0's auc: 0.523774
[80]	valid_0's auc: 0.524062
[90]	valid_0's auc: 0.524066
[100]	valid_0's auc: 0.524111
[110]	valid_0's auc: 0.524162
[120]	valid_0's auc: 0.524169
[130]	valid_0's auc: 0.524168
[140]	valid_0's auc: 0.524164
[150]	valid_0's auc: 0.524193
[160]	valid_0's auc: 0.524195
[170]	valid_0's auc: 0.524195
[180]	valid_0's auc: 0.524196
[190]	valid_0's auc: 0.524205
[200]	valid_0's auc: 0.524205
[210]	valid_0's auc: 0.524204
[220]	valid_0's auc: 0.524204
[230]	valid_0's auc: 0.524204
Early stopping, best iteration is:
[189]	valid_0's auc: 0.524205
best score: 0.524204801236
best iteration: 189
complete on: ISC_top3_in_song_log10

working on: ISC_artist_name_log10


After selection:
target                     uint8
ISC_artist_name_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.52066
[20]	valid_0's auc: 0.52691
[30]	valid_0's auc: 0.528696
[40]	valid_0's auc: 0.530797
[50]	valid_0's auc: 0.532433
[60]	valid_0's auc: 0.534376
[70]	valid_0's auc: 0.535128
[80]	valid_0's auc: 0.535797
[90]	valid_0's auc: 0.536382
[100]	valid_0's auc: 0.537061
[110]	valid_0's auc: 0.537668
[120]	valid_0's auc: 0.53811
[130]	valid_0's auc: 0.538539
[140]	valid_0's auc: 0.538729
[150]	valid_0's auc: 0.538977
[160]	valid_0's auc: 0.539172
[170]	valid_0's auc: 0.539288
[180]	valid_0's auc: 0.539379
[190]	valid_0's auc: 0.539534
[200]	valid_0's auc: 0.539563
[210]	valid_0's auc: 0.539672
[220]	valid_0's auc: 0.539724
[230]	valid_0's auc: 0.539778
[240]	valid_0's auc: 0.53982
[250]	valid_0's auc: 0.539868
[260]	valid_0's auc: 0.539954
[270]	valid_0's auc: 0.539989
[280]	valid_0's auc: 0.539978
[290]	valid_0's auc: 0.539992
[300]	valid_0's auc: 0.539974
[310]	valid_0's auc: 0.539986
[320]	valid_0's auc: 0.54004
[330]	valid_0's auc: 0.540069
[340]	valid_0's auc: 0.540096
[350]	valid_0's auc: 0.540185
[360]	valid_0's auc: 0.540185
[370]	valid_0's auc: 0.540276
[380]	valid_0's auc: 0.540282
[390]	valid_0's auc: 0.540305
[400]	valid_0's auc: 0.540242
[410]	valid_0's auc: 0.540322
[420]	valid_0's auc: 0.540297
[430]	valid_0's auc: 0.540283
[440]	valid_0's auc: 0.540282
[450]	valid_0's auc: 0.540317
[460]	valid_0's auc: 0.540323
[470]	valid_0's auc: 0.540335
[480]	valid_0's auc: 0.540322
[490]	valid_0's auc: 0.54032
[500]	valid_0's auc: 0.540333
[510]	valid_0's auc: 0.540321
[520]	valid_0's auc: 0.540344
[530]	valid_0's auc: 0.540323
[540]	valid_0's auc: 0.540318
[550]	valid_0's auc: 0.540334
[560]	valid_0's auc: 0.54034
[570]	valid_0's auc: 0.540337
Early stopping, best iteration is:
[525]	valid_0's auc: 0.540351
best score: 0.540350746139
best iteration: 525
complete on: ISC_artist_name_log10

working on: ISCZ_artist_name_log10


After selection:
target                      uint8
ISCZ_artist_name_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.520539
[20]	valid_0's auc: 0.526452
[30]	valid_0's auc: 0.528995
[40]	valid_0's auc: 0.530495
[50]	valid_0's auc: 0.53192
[60]	valid_0's auc: 0.533367
[70]	valid_0's auc: 0.534389
[80]	valid_0's auc: 0.535167
[90]	valid_0's auc: 0.535972
[100]	valid_0's auc: 0.536484
[110]	valid_0's auc: 0.536922
[120]	valid_0's auc: 0.537435
[130]	valid_0's auc: 0.538054
[140]	valid_0's auc: 0.538408
[150]	valid_0's auc: 0.5386
[160]	valid_0's auc: 0.538712
[170]	valid_0's auc: 0.53883
[180]	valid_0's auc: 0.538834
[190]	valid_0's auc: 0.53919
[200]	valid_0's auc: 0.539311
[210]	valid_0's auc: 0.539405
[220]	valid_0's auc: 0.539434
[230]	valid_0's auc: 0.539449
[240]	valid_0's auc: 0.539488
[250]	valid_0's auc: 0.539537
[260]	valid_0's auc: 0.539549
[270]	valid_0's auc: 0.53966
[280]	valid_0's auc: 0.539676
[290]	valid_0's auc: 0.53972
[300]	valid_0's auc: 0.539747
[310]	valid_0's auc: 0.539819
[320]	valid_0's auc: 0.539904
[330]	valid_0's auc: 0.539942
[340]	valid_0's auc: 0.540016
[350]	valid_0's auc: 0.540038
[360]	valid_0's auc: 0.540057
[370]	valid_0's auc: 0.540106
[380]	valid_0's auc: 0.540105
[390]	valid_0's auc: 0.54011
[400]	valid_0's auc: 0.540123
[410]	valid_0's auc: 0.540123
[420]	valid_0's auc: 0.540116
[430]	valid_0's auc: 0.540114
[440]	valid_0's auc: 0.54011
[450]	valid_0's auc: 0.540099
Early stopping, best iteration is:
[403]	valid_0's auc: 0.540158
best score: 0.540157514377
best iteration: 403
complete on: ISCZ_artist_name_log10

working on: ISC_composer_log10


After selection:
target                  uint8
ISC_composer_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529858
[20]	valid_0's auc: 0.531781
[30]	valid_0's auc: 0.532789
[40]	valid_0's auc: 0.534141
[50]	valid_0's auc: 0.534754
[60]	valid_0's auc: 0.535392
[70]	valid_0's auc: 0.535468
[80]	valid_0's auc: 0.535444
[90]	valid_0's auc: 0.535748
[100]	valid_0's auc: 0.53586
[110]	valid_0's auc: 0.536062
[120]	valid_0's auc: 0.536519
[130]	valid_0's auc: 0.536689
[140]	valid_0's auc: 0.536905
[150]	valid_0's auc: 0.536926
[160]	valid_0's auc: 0.536976
[170]	valid_0's auc: 0.537041
[180]	valid_0's auc: 0.537129
[190]	valid_0's auc: 0.53717
[200]	valid_0's auc: 0.537188
[210]	valid_0's auc: 0.537217
[220]	valid_0's auc: 0.537222
[230]	valid_0's auc: 0.537265
[240]	valid_0's auc: 0.537266
[250]	valid_0's auc: 0.537257
[260]	valid_0's auc: 0.537212
[270]	valid_0's auc: 0.537186
[280]	valid_0's auc: 0.537203
Early stopping, best iteration is:
[237]	valid_0's auc: 0.537288
best score: 0.537287860791
best iteration: 237
complete on: ISC_composer_log10

working on: ISCZ_composer_log10


After selection:
target                   uint8
ISCZ_composer_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526093
[20]	valid_0's auc: 0.531062
[30]	valid_0's auc: 0.533256
[40]	valid_0's auc: 0.533834
[50]	valid_0's auc: 0.534118
[60]	valid_0's auc: 0.534801
[70]	valid_0's auc: 0.534945
[80]	valid_0's auc: 0.535267
[90]	valid_0's auc: 0.535578
[100]	valid_0's auc: 0.535565
[110]	valid_0's auc: 0.535715
[120]	valid_0's auc: 0.535731
[130]	valid_0's auc: 0.535962
[140]	valid_0's auc: 0.536051
[150]	valid_0's auc: 0.536113
[160]	valid_0's auc: 0.536485
[170]	valid_0's auc: 0.536489
[180]	valid_0's auc: 0.536514
[190]	valid_0's auc: 0.53659
[200]	valid_0's auc: 0.536544
[210]	valid_0's auc: 0.536736
[220]	valid_0's auc: 0.536738
[230]	valid_0's auc: 0.53687
[240]	valid_0's auc: 0.536888
[250]	valid_0's auc: 0.53687
[260]	valid_0's auc: 0.53694
[270]	valid_0's auc: 0.537016
[280]	valid_0's auc: 0.537063
[290]	valid_0's auc: 0.537111
[300]	valid_0's auc: 0.537114
[310]	valid_0's auc: 0.537145
[320]	valid_0's auc: 0.537156
[330]	valid_0's auc: 0.537172
[340]	valid_0's auc: 0.537176
[350]	valid_0's auc: 0.537173
[360]	valid_0's auc: 0.537166
[370]	valid_0's auc: 0.53717
[380]	valid_0's auc: 0.537193
[390]	valid_0's auc: 0.537194
[400]	valid_0's auc: 0.537254
[410]	valid_0's auc: 0.537219
[420]	valid_0's auc: 0.537219
[430]	valid_0's auc: 0.537232
[440]	valid_0's auc: 0.537226
[450]	valid_0's auc: 0.537226
Early stopping, best iteration is:
[401]	valid_0's auc: 0.537262
best score: 0.537262118627
best iteration: 401
complete on: ISCZ_composer_log10

working on: ISC_lyricist_log10


After selection:
target                  uint8
ISC_lyricist_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.52941
[20]	valid_0's auc: 0.529862
[30]	valid_0's auc: 0.529671
[40]	valid_0's auc: 0.529837
[50]	valid_0's auc: 0.53
[60]	valid_0's auc: 0.53016
[70]	valid_0's auc: 0.530402
[80]	valid_0's auc: 0.530589
[90]	valid_0's auc: 0.530946
[100]	valid_0's auc: 0.531038
[110]	valid_0's auc: 0.531107
[120]	valid_0's auc: 0.531055
[130]	valid_0's auc: 0.53109
[140]	valid_0's auc: 0.531174
[150]	valid_0's auc: 0.531292
[160]	valid_0's auc: 0.53131
[170]	valid_0's auc: 0.531297
[180]	valid_0's auc: 0.531334
[190]	valid_0's auc: 0.531335
[200]	valid_0's auc: 0.531497
[210]	valid_0's auc: 0.531509
[220]	valid_0's auc: 0.531592
[230]	valid_0's auc: 0.531636
[240]	valid_0's auc: 0.531707
[250]	valid_0's auc: 0.531697
[260]	valid_0's auc: 0.531679
[270]	valid_0's auc: 0.531679
[280]	valid_0's auc: 0.531655
Early stopping, best iteration is:
[237]	valid_0's auc: 0.531707
best score: 0.531707440591
best iteration: 237
complete on: ISC_lyricist_log10

working on: ISCZ_lyricist_log10


After selection:
target                   uint8
ISCZ_lyricist_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529393
[20]	valid_0's auc: 0.530103
[30]	valid_0's auc: 0.529712
[40]	valid_0's auc: 0.529703
[50]	valid_0's auc: 0.530302
[60]	valid_0's auc: 0.530348
[70]	valid_0's auc: 0.530474
[80]	valid_0's auc: 0.530654
[90]	valid_0's auc: 0.530669
[100]	valid_0's auc: 0.530792
[110]	valid_0's auc: 0.530932
[120]	valid_0's auc: 0.531098
[130]	valid_0's auc: 0.531151
[140]	valid_0's auc: 0.531232
[150]	valid_0's auc: 0.531265
[160]	valid_0's auc: 0.531226
[170]	valid_0's auc: 0.531328
[180]	valid_0's auc: 0.531339
[190]	valid_0's auc: 0.531394
[200]	valid_0's auc: 0.531452
[210]	valid_0's auc: 0.53141
[220]	valid_0's auc: 0.531519
[230]	valid_0's auc: 0.531517
[240]	valid_0's auc: 0.531517
[250]	valid_0's auc: 0.531496
[260]	valid_0's auc: 0.531489
Early stopping, best iteration is:
[217]	valid_0's auc: 0.531539
best score: 0.531539457107
best iteration: 217
complete on: ISCZ_lyricist_log10

working on: ISC_name_log10


After selection:
target              uint8
ISC_name_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506615
[20]	valid_0's auc: 0.507944
[30]	valid_0's auc: 0.508421
[40]	valid_0's auc: 0.508514
[50]	valid_0's auc: 0.509044
[60]	valid_0's auc: 0.509127
[70]	valid_0's auc: 0.509484
[80]	valid_0's auc: 0.509625
[90]	valid_0's auc: 0.509644
[100]	valid_0's auc: 0.509827
[110]	valid_0's auc: 0.509764
[120]	valid_0's auc: 0.509867
[130]	valid_0's auc: 0.509931
[140]	valid_0's auc: 0.510037
[150]	valid_0's auc: 0.510162
[160]	valid_0's auc: 0.510306
[170]	valid_0's auc: 0.510336
[180]	valid_0's auc: 0.510366
[190]	valid_0's auc: 0.510393
[200]	valid_0's auc: 0.510419
[210]	valid_0's auc: 0.510432
[220]	valid_0's auc: 0.510451
[230]	valid_0's auc: 0.510459
[240]	valid_0's auc: 0.510449
[250]	valid_0's auc: 0.51047
[260]	valid_0's auc: 0.51046
[270]	valid_0's auc: 0.510478
[280]	valid_0's auc: 0.510486
[290]	valid_0's auc: 0.510449
[300]	valid_0's auc: 0.510454
[310]	valid_0's auc: 0.510446
[320]	valid_0's auc: 0.510448
[330]	valid_0's auc: 0.510452
Early stopping, best iteration is:
[281]	valid_0's auc: 0.510486
best score: 0.510486215639
best iteration: 281
complete on: ISC_name_log10

working on: ISCZ_name_log10


After selection:
target               uint8
ISCZ_name_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506192
[20]	valid_0's auc: 0.507878
[30]	valid_0's auc: 0.508123
[40]	valid_0's auc: 0.508555
[50]	valid_0's auc: 0.508839
[60]	valid_0's auc: 0.509194
[70]	valid_0's auc: 0.509364
[80]	valid_0's auc: 0.509462
[90]	valid_0's auc: 0.509521
[100]	valid_0's auc: 0.509652
[110]	valid_0's auc: 0.509708
[120]	valid_0's auc: 0.509693
[130]	valid_0's auc: 0.509717
[140]	valid_0's auc: 0.509738
[150]	valid_0's auc: 0.509794
[160]	valid_0's auc: 0.509857
[170]	valid_0's auc: 0.509922
[180]	valid_0's auc: 0.509944
[190]	valid_0's auc: 0.509986
[200]	valid_0's auc: 0.510058
[210]	valid_0's auc: 0.510063
[220]	valid_0's auc: 0.510044
[230]	valid_0's auc: 0.510058
[240]	valid_0's auc: 0.510097
[250]	valid_0's auc: 0.510094
[260]	valid_0's auc: 0.510112
[270]	valid_0's auc: 0.510106
[280]	valid_0's auc: 0.510117
[290]	valid_0's auc: 0.51012
[300]	valid_0's auc: 0.510104
[310]	valid_0's auc: 0.510108
[320]	valid_0's auc: 0.51011
[330]	valid_0's auc: 0.510111
[340]	valid_0's auc: 0.510119
Early stopping, best iteration is:
[292]	valid_0's auc: 0.51012
best score: 0.510119863181
best iteration: 292
complete on: ISCZ_name_log10

working on: ISC_language_log10


After selection:
target                  uint8
ISC_language_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.524765
[20]	valid_0's auc: 0.524711
[30]	valid_0's auc: 0.524766
[40]	valid_0's auc: 0.524766
[50]	valid_0's auc: 0.524767
[60]	valid_0's auc: 0.524768
[70]	valid_0's auc: 0.524768
[80]	valid_0's auc: 0.524768
[90]	valid_0's auc: 0.524768
[100]	valid_0's auc: 0.524768
Early stopping, best iteration is:
[57]	valid_0's auc: 0.524768
best score: 0.524767621195
best iteration: 57
complete on: ISC_language_log10

working on: ISC_isrc_log10


After selection:
target              uint8
ISC_isrc_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503213
[20]	valid_0's auc: 0.503222
[30]	valid_0's auc: 0.503258
[40]	valid_0's auc: 0.503264
[50]	valid_0's auc: 0.503274
[60]	valid_0's auc: 0.503274
[70]	valid_0's auc: 0.50328
[80]	valid_0's auc: 0.50328
[90]	valid_0's auc: 0.50328
[100]	valid_0's auc: 0.503281
[110]	valid_0's auc: 0.503281
[120]	valid_0's auc: 0.503281
[130]	valid_0's auc: 0.503281
[140]	valid_0's auc: 0.503281
[150]	valid_0's auc: 0.503281
[160]	valid_0's auc: 0.503281
[170]	valid_0's auc: 0.503281
Early stopping, best iteration is:
[121]	valid_0's auc: 0.503281
best score: 0.503281002111
best iteration: 121
complete on: ISC_isrc_log10

working on: ISCZ_isrc_log10


After selection:
target               uint8
ISCZ_isrc_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503256
[20]	valid_0's auc: 0.50329
[30]	valid_0's auc: 0.503256
[40]	valid_0's auc: 0.503277
[50]	valid_0's auc: 0.503273
[60]	valid_0's auc: 0.503282
[70]	valid_0's auc: 0.503281
[80]	valid_0's auc: 0.503281
Early stopping, best iteration is:
[37]	valid_0's auc: 0.503297
best score: 0.503297119792
best iteration: 37
complete on: ISCZ_isrc_log10

working on: ISC_song_country_log10


After selection:
target                      uint8
ISC_song_country_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525537
[20]	valid_0's auc: 0.525582
[30]	valid_0's auc: 0.52585
[40]	valid_0's auc: 0.525891
[50]	valid_0's auc: 0.526085
[60]	valid_0's auc: 0.526107
[70]	valid_0's auc: 0.526109
[80]	valid_0's auc: 0.526131
[90]	valid_0's auc: 0.52612
[100]	valid_0's auc: 0.526158
[110]	valid_0's auc: 0.526159
[120]	valid_0's auc: 0.526061
[130]	valid_0's auc: 0.52606
[140]	valid_0's auc: 0.52606
[150]	valid_0's auc: 0.52605
[160]	valid_0's auc: 0.526049
Early stopping, best iteration is:
[113]	valid_0's auc: 0.526159
best score: 0.526158644716
best iteration: 113
complete on: ISC_song_country_log10

working on: ISCZ_song_country_log10


After selection:
target                       uint8
ISCZ_song_country_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525727
[20]	valid_0's auc: 0.525566
[30]	valid_0's auc: 0.525816
[40]	valid_0's auc: 0.52589
[50]	valid_0's auc: 0.52591
[60]	valid_0's auc: 0.526088
[70]	valid_0's auc: 0.526109
[80]	valid_0's auc: 0.526137
[90]	valid_0's auc: 0.526156
[100]	valid_0's auc: 0.526156
[110]	valid_0's auc: 0.526153
[120]	valid_0's auc: 0.526154
[130]	valid_0's auc: 0.52604
Early stopping, best iteration is:
[86]	valid_0's auc: 0.526204
best score: 0.526204438216
best iteration: 86
complete on: ISCZ_song_country_log10

working on: ISC_rc_log10


After selection:
target            uint8
ISC_rc_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.532448
[20]	valid_0's auc: 0.534773
[30]	valid_0's auc: 0.536011
[40]	valid_0's auc: 0.536351
[50]	valid_0's auc: 0.53736
[60]	valid_0's auc: 0.53813
[70]	valid_0's auc: 0.538682
[80]	valid_0's auc: 0.539004
[90]	valid_0's auc: 0.539677
[100]	valid_0's auc: 0.53977
[110]	valid_0's auc: 0.539813
[120]	valid_0's auc: 0.54016
[130]	valid_0's auc: 0.540598
[140]	valid_0's auc: 0.540894
[150]	valid_0's auc: 0.540968
[160]	valid_0's auc: 0.54104
[170]	valid_0's auc: 0.541372
[180]	valid_0's auc: 0.54123
[190]	valid_0's auc: 0.541314
[200]	valid_0's auc: 0.541316
[210]	valid_0's auc: 0.541334
[220]	valid_0's auc: 0.541352
Early stopping, best iteration is:
[170]	valid_0's auc: 0.541372
best score: 0.541371676405
best iteration: 170
complete on: ISC_rc_log10

working on: ISCZ_rc_log10


After selection:
target             uint8
ISCZ_rc_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.533544
[20]	valid_0's auc: 0.537049
[30]	valid_0's auc: 0.537975
[40]	valid_0's auc: 0.538702
[50]	valid_0's auc: 0.539571
[60]	valid_0's auc: 0.54024
[70]	valid_0's auc: 0.540825
[80]	valid_0's auc: 0.540893
[90]	valid_0's auc: 0.540916
[100]	valid_0's auc: 0.540473
[110]	valid_0's auc: 0.540972
[120]	valid_0's auc: 0.541324
[130]	valid_0's auc: 0.541841
[140]	valid_0's auc: 0.541836
[150]	valid_0's auc: 0.541897
[160]	valid_0's auc: 0.541765
[170]	valid_0's auc: 0.541798
[180]	valid_0's auc: 0.541774
[190]	valid_0's auc: 0.54183
Early stopping, best iteration is:
[147]	valid_0's auc: 0.5419
best score: 0.541900454552
best iteration: 147
complete on: ISCZ_rc_log10

working on: ISC_isrc_rest_log10


After selection:
target                   uint8
ISC_isrc_rest_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51166
[20]	valid_0's auc: 0.512529
[30]	valid_0's auc: 0.513976
[40]	valid_0's auc: 0.515331
[50]	valid_0's auc: 0.517042
[60]	valid_0's auc: 0.5167
[70]	valid_0's auc: 0.517088
[80]	valid_0's auc: 0.517793
[90]	valid_0's auc: 0.517854
[100]	valid_0's auc: 0.517811
[110]	valid_0's auc: 0.518294
[120]	valid_0's auc: 0.518456
[130]	valid_0's auc: 0.518372
[140]	valid_0's auc: 0.518444
[150]	valid_0's auc: 0.51852
[160]	valid_0's auc: 0.518641
[170]	valid_0's auc: 0.518869
[180]	valid_0's auc: 0.518949
[190]	valid_0's auc: 0.518956
[200]	valid_0's auc: 0.518909
[210]	valid_0's auc: 0.518918
[220]	valid_0's auc: 0.51877
[230]	valid_0's auc: 0.518777
[240]	valid_0's auc: 0.518746
[250]	valid_0's auc: 0.518656
Early stopping, best iteration is:
[208]	valid_0's auc: 0.518992
best score: 0.518991746193
best iteration: 208
complete on: ISC_isrc_rest_log10

working on: ISCZ_isrc_rest_log10


After selection:
target                    uint8
ISCZ_isrc_rest_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512622
[20]	valid_0's auc: 0.513709
[30]	valid_0's auc: 0.515012
[40]	valid_0's auc: 0.515308
[50]	valid_0's auc: 0.515727
[60]	valid_0's auc: 0.51691
[70]	valid_0's auc: 0.517191
[80]	valid_0's auc: 0.517326
[90]	valid_0's auc: 0.517447
[100]	valid_0's auc: 0.517944
[110]	valid_0's auc: 0.517922
[120]	valid_0's auc: 0.517375
[130]	valid_0's auc: 0.517401
[140]	valid_0's auc: 0.517422
[150]	valid_0's auc: 0.517464
Early stopping, best iteration is:
[103]	valid_0's auc: 0.517961
best score: 0.517961179087
best iteration: 103
complete on: ISCZ_isrc_rest_log10

working on: ISC_song_year_log10


After selection:
target                   uint8
ISC_song_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537977
[20]	valid_0's auc: 0.538023
[30]	valid_0's auc: 0.538043
[40]	valid_0's auc: 0.538077
[50]	valid_0's auc: 0.538055
[60]	valid_0's auc: 0.538055
[70]	valid_0's auc: 0.538055
[80]	valid_0's auc: 0.538055
[90]	valid_0's auc: 0.538055
Early stopping, best iteration is:
[43]	valid_0's auc: 0.538077
best score: 0.53807736876
best iteration: 43
complete on: ISC_song_year_log10

working on: ISCZ_song_year_log10


After selection:
target                    uint8
ISCZ_song_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.515549
[20]	valid_0's auc: 0.533867
[30]	valid_0's auc: 0.538028
[40]	valid_0's auc: 0.538041
[50]	valid_0's auc: 0.538054
[60]	valid_0's auc: 0.538055
[70]	valid_0's auc: 0.538055
[80]	valid_0's auc: 0.538055
[90]	valid_0's auc: 0.538055
[100]	valid_0's auc: 0.538055
[110]	valid_0's auc: 0.538055
[120]	valid_0's auc: 0.538055
[130]	valid_0's auc: 0.538055
[140]	valid_0's auc: 0.538055
[150]	valid_0's auc: 0.538055
[160]	valid_0's auc: 0.538055
Early stopping, best iteration is:
[115]	valid_0's auc: 0.538055
best score: 0.538055249532
best iteration: 115
complete on: ISCZ_song_year_log10

                       ISC_isrc_log10:  0.503281002111
                      ISCZ_isrc_log10:  0.503297119792
                      ISCZ_name_log10:  0.510119863181
                       ISC_name_log10:  0.510486215639
                 ISCZ_isrc_rest_log10:  0.517961179087
                  ISC_isrc_rest_log10:  0.518991746193
               ISC_top3_in_song_log10:  0.524204801236
                   ISC_language_log10:  0.524767621195
               ISC_song_country_log10:  0.526158644716
              ISCZ_song_country_log10:  0.526204438216
               ISC_top1_in_song_log10:  0.526209798343
                  ISC_genre_ids_log10:  0.527626686412
                 ISCZ_genre_ids_log10:  0.527690851179
               ISC_top2_in_song_log10:  0.528386874735
                    song_length_log10:  0.531174076926
                  ISCZ_lyricist_log10:  0.531539457107
                   ISC_lyricist_log10:  0.531707440591
                  ISCZ_composer_log10:  0.537262118627
                   ISC_composer_log10:  0.537287860791
                 ISCZ_song_year_log10:  0.538055249532
                      song_year_log10:  0.538060451128
                  ISC_song_year_log10:  0.53807736876
               ISCZ_artist_name_log10:  0.540157514377
                ISC_artist_name_log10:  0.540350746139
                         ISC_rc_log10:  0.541371676405
                        ISCZ_rc_log10:  0.541900454552

[timer]: complete in 48m 58s

Process finished with exit code 0
'''
