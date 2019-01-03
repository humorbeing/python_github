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
target                    uint8
song_length_ln          float64
song_year_ln            float64
ISC_genre_ids_ln        float64
ISCZ_genre_ids_ln       float64
ISC_top1_in_song_ln     float64
ISC_top2_in_song_ln     float64
ISC_top3_in_song_ln     float64
ISC_artist_name_ln      float64
ISCZ_artist_name_ln     float64
ISC_composer_ln         float64
ISCZ_composer_ln        float64
ISC_lyricist_ln         float64
ISCZ_lyricist_ln        float64
ISC_name_ln             float64
ISCZ_name_ln            float64
ISC_language_ln         float64
ISC_isrc_ln             float64
ISCZ_isrc_ln            float64
ISC_song_country_ln     float64
ISCZ_song_country_ln    float64
ISC_rc_ln               float64
ISCZ_rc_ln              float64
ISC_isrc_rest_ln        float64
ISCZ_isrc_rest_ln       float64
ISC_song_year_ln        float64
ISCZ_song_year_ln       float64
dtype: object
number of rows: 7377418
number of columns: 27
working on: song_length_ln


After selection:
target              uint8
song_length_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.518346
[20]	valid_0's auc: 0.522109
[30]	valid_0's auc: 0.523872
[40]	valid_0's auc: 0.52591
[50]	valid_0's auc: 0.526644
[60]	valid_0's auc: 0.52757
[70]	valid_0's auc: 0.52803
[80]	valid_0's auc: 0.528336
[90]	valid_0's auc: 0.529136
[100]	valid_0's auc: 0.529484
[110]	valid_0's auc: 0.529372
[120]	valid_0's auc: 0.529455
[130]	valid_0's auc: 0.529639
[140]	valid_0's auc: 0.529641
[150]	valid_0's auc: 0.529587
[160]	valid_0's auc: 0.529666
[170]	valid_0's auc: 0.529746
[180]	valid_0's auc: 0.529766
[190]	valid_0's auc: 0.529691
Early stopping, best iteration is:
[148]	valid_0's auc: 0.529799
best score: 0.529798839113
best iteration: 148
complete on: song_length_ln

working on: song_year_ln


After selection:
target            uint8
song_year_ln    float64
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
best score: 0.538060450866
best iteration: 23
complete on: song_year_ln

working on: ISC_genre_ids_ln


After selection:
target                uint8
ISC_genre_ids_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526488
[20]	valid_0's auc: 0.52699
[30]	valid_0's auc: 0.527081
[40]	valid_0's auc: 0.527282
[50]	valid_0's auc: 0.527318
[60]	valid_0's auc: 0.527268
[70]	valid_0's auc: 0.52756
[80]	valid_0's auc: 0.527566
[90]	valid_0's auc: 0.527585
[100]	valid_0's auc: 0.527579
[110]	valid_0's auc: 0.527577
[120]	valid_0's auc: 0.527569
[130]	valid_0's auc: 0.527569
[140]	valid_0's auc: 0.527573
Early stopping, best iteration is:
[94]	valid_0's auc: 0.527586
best score: 0.52758560988
best iteration: 94
complete on: ISC_genre_ids_ln

working on: ISCZ_genre_ids_ln


After selection:
target                 uint8
ISCZ_genre_ids_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525729
[20]	valid_0's auc: 0.526874
[30]	valid_0's auc: 0.526983
[40]	valid_0's auc: 0.527114
[50]	valid_0's auc: 0.527149
[60]	valid_0's auc: 0.527201
[70]	valid_0's auc: 0.527239
[80]	valid_0's auc: 0.527236
[90]	valid_0's auc: 0.527237
[100]	valid_0's auc: 0.527241
[110]	valid_0's auc: 0.527545
[120]	valid_0's auc: 0.527537
[130]	valid_0's auc: 0.52754
[140]	valid_0's auc: 0.527539
[150]	valid_0's auc: 0.527539
[160]	valid_0's auc: 0.527541
[170]	valid_0's auc: 0.527577
[180]	valid_0's auc: 0.527577
[190]	valid_0's auc: 0.527577
[200]	valid_0's auc: 0.527594
[210]	valid_0's auc: 0.527599
[220]	valid_0's auc: 0.527654
[230]	valid_0's auc: 0.52767
[240]	valid_0's auc: 0.527671
[250]	valid_0's auc: 0.527662
[260]	valid_0's auc: 0.527661
[270]	valid_0's auc: 0.527672
[280]	valid_0's auc: 0.527662
[290]	valid_0's auc: 0.527618
[300]	valid_0's auc: 0.527619
[310]	valid_0's auc: 0.527601
Early stopping, best iteration is:
[269]	valid_0's auc: 0.527672
best score: 0.527671906756
best iteration: 269
complete on: ISCZ_genre_ids_ln

working on: ISC_top1_in_song_ln


After selection:
target                   uint8
ISC_top1_in_song_ln    float64
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
[40]	valid_0's auc: 0.526
[50]	valid_0's auc: 0.52602
[60]	valid_0's auc: 0.526034
[70]	valid_0's auc: 0.526053
[80]	valid_0's auc: 0.526077
Early stopping, best iteration is:
[36]	valid_0's auc: 0.526186
best score: 0.526185542306
best iteration: 36
complete on: ISC_top1_in_song_ln

working on: ISC_top2_in_song_ln


After selection:
target                   uint8
ISC_top2_in_song_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.527641
[20]	valid_0's auc: 0.528212
[30]	valid_0's auc: 0.527932
[40]	valid_0's auc: 0.527945
[50]	valid_0's auc: 0.527999
[60]	valid_0's auc: 0.528005
[70]	valid_0's auc: 0.528012
Early stopping, best iteration is:
[20]	valid_0's auc: 0.528212
best score: 0.528211756335
best iteration: 20
complete on: ISC_top2_in_song_ln

working on: ISC_top3_in_song_ln


After selection:
target                   uint8
ISC_top3_in_song_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.5231
[20]	valid_0's auc: 0.523667
[30]	valid_0's auc: 0.523682
[40]	valid_0's auc: 0.523716
[50]	valid_0's auc: 0.523723
[60]	valid_0's auc: 0.523747
[70]	valid_0's auc: 0.523773
[80]	valid_0's auc: 0.523787
[90]	valid_0's auc: 0.523788
[100]	valid_0's auc: 0.524125
[110]	valid_0's auc: 0.524125
[120]	valid_0's auc: 0.524128
[130]	valid_0's auc: 0.524169
[140]	valid_0's auc: 0.524197
[150]	valid_0's auc: 0.524198
[160]	valid_0's auc: 0.524199
[170]	valid_0's auc: 0.524198
[180]	valid_0's auc: 0.524199
[190]	valid_0's auc: 0.524198
[200]	valid_0's auc: 0.524207
[210]	valid_0's auc: 0.524209
[220]	valid_0's auc: 0.524209
[230]	valid_0's auc: 0.524209
[240]	valid_0's auc: 0.524207
[250]	valid_0's auc: 0.524208
[260]	valid_0's auc: 0.524208
[270]	valid_0's auc: 0.524208
[280]	valid_0's auc: 0.524206
Early stopping, best iteration is:
[230]	valid_0's auc: 0.524209
best score: 0.524209239032
best iteration: 230
complete on: ISC_top3_in_song_ln

working on: ISC_artist_name_ln


After selection:
target                  uint8
ISC_artist_name_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.52087
[20]	valid_0's auc: 0.5265
[30]	valid_0's auc: 0.527775
[40]	valid_0's auc: 0.529851
[50]	valid_0's auc: 0.532021
[60]	valid_0's auc: 0.532884
[70]	valid_0's auc: 0.5346
[80]	valid_0's auc: 0.535402
[90]	valid_0's auc: 0.536074
[100]	valid_0's auc: 0.536564
[110]	valid_0's auc: 0.536655
[120]	valid_0's auc: 0.536841
[130]	valid_0's auc: 0.537222
[140]	valid_0's auc: 0.537488
[150]	valid_0's auc: 0.537568
[160]	valid_0's auc: 0.53798
[170]	valid_0's auc: 0.538104
[180]	valid_0's auc: 0.538256
[190]	valid_0's auc: 0.538339
[200]	valid_0's auc: 0.538435
[210]	valid_0's auc: 0.538441
[220]	valid_0's auc: 0.538539
[230]	valid_0's auc: 0.538621
[240]	valid_0's auc: 0.538624
[250]	valid_0's auc: 0.538789
[260]	valid_0's auc: 0.538813
[270]	valid_0's auc: 0.538865
[280]	valid_0's auc: 0.538895
[290]	valid_0's auc: 0.53896
[300]	valid_0's auc: 0.538961
[310]	valid_0's auc: 0.539015
[320]	valid_0's auc: 0.539061
[330]	valid_0's auc: 0.539144
[340]	valid_0's auc: 0.539077
[350]	valid_0's auc: 0.539075
[360]	valid_0's auc: 0.539109
[370]	valid_0's auc: 0.539107
[380]	valid_0's auc: 0.539101
Early stopping, best iteration is:
[330]	valid_0's auc: 0.539144
best score: 0.539143949452
best iteration: 330
complete on: ISC_artist_name_ln

working on: ISCZ_artist_name_ln


After selection:
target                   uint8
ISCZ_artist_name_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.520682
[20]	valid_0's auc: 0.526583
[30]	valid_0's auc: 0.529025
[40]	valid_0's auc: 0.530759
[50]	valid_0's auc: 0.532817
[60]	valid_0's auc: 0.533148
[70]	valid_0's auc: 0.534297
[80]	valid_0's auc: 0.535593
[90]	valid_0's auc: 0.535889
[100]	valid_0's auc: 0.536281
[110]	valid_0's auc: 0.53727
[120]	valid_0's auc: 0.537651
[130]	valid_0's auc: 0.538163
[140]	valid_0's auc: 0.538341
[150]	valid_0's auc: 0.53847
[160]	valid_0's auc: 0.538599
[170]	valid_0's auc: 0.538871
[180]	valid_0's auc: 0.538997
[190]	valid_0's auc: 0.539169
[200]	valid_0's auc: 0.539266
[210]	valid_0's auc: 0.53928
[220]	valid_0's auc: 0.539361
[230]	valid_0's auc: 0.539428
[240]	valid_0's auc: 0.539501
[250]	valid_0's auc: 0.539539
[260]	valid_0's auc: 0.539614
[270]	valid_0's auc: 0.539611
[280]	valid_0's auc: 0.539618
[290]	valid_0's auc: 0.539719
[300]	valid_0's auc: 0.539748
[310]	valid_0's auc: 0.539822
[320]	valid_0's auc: 0.53981
[330]	valid_0's auc: 0.539859
[340]	valid_0's auc: 0.539915
[350]	valid_0's auc: 0.539943
[360]	valid_0's auc: 0.539915
[370]	valid_0's auc: 0.539926
[380]	valid_0's auc: 0.539955
[390]	valid_0's auc: 0.53999
[400]	valid_0's auc: 0.539938
[410]	valid_0's auc: 0.53994
[420]	valid_0's auc: 0.539932
[430]	valid_0's auc: 0.539942
Early stopping, best iteration is:
[388]	valid_0's auc: 0.540001
best score: 0.540000784154
best iteration: 388
complete on: ISCZ_artist_name_ln

working on: ISC_composer_ln


After selection:
target               uint8
ISC_composer_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529859
[20]	valid_0's auc: 0.531781
[30]	valid_0's auc: 0.533104
[40]	valid_0's auc: 0.533793
[50]	valid_0's auc: 0.534502
[60]	valid_0's auc: 0.535371
[70]	valid_0's auc: 0.535659
[80]	valid_0's auc: 0.535709
[90]	valid_0's auc: 0.535945
[100]	valid_0's auc: 0.53618
[110]	valid_0's auc: 0.53609
[120]	valid_0's auc: 0.536131
[130]	valid_0's auc: 0.536124
[140]	valid_0's auc: 0.536135
[150]	valid_0's auc: 0.536445
[160]	valid_0's auc: 0.536667
[170]	valid_0's auc: 0.536731
[180]	valid_0's auc: 0.536727
[190]	valid_0's auc: 0.536814
[200]	valid_0's auc: 0.536899
[210]	valid_0's auc: 0.537012
[220]	valid_0's auc: 0.537041
[230]	valid_0's auc: 0.537113
[240]	valid_0's auc: 0.537161
[250]	valid_0's auc: 0.53719
[260]	valid_0's auc: 0.537203
[270]	valid_0's auc: 0.537205
[280]	valid_0's auc: 0.537201
[290]	valid_0's auc: 0.53716
[300]	valid_0's auc: 0.537168
[310]	valid_0's auc: 0.537188
[320]	valid_0's auc: 0.537209
[330]	valid_0's auc: 0.537214
[340]	valid_0's auc: 0.537227
[350]	valid_0's auc: 0.537246
[360]	valid_0's auc: 0.537226
[370]	valid_0's auc: 0.537236
[380]	valid_0's auc: 0.53726
[390]	valid_0's auc: 0.537253
[400]	valid_0's auc: 0.537252
[410]	valid_0's auc: 0.537243
[420]	valid_0's auc: 0.537256
[430]	valid_0's auc: 0.537257
Early stopping, best iteration is:
[380]	valid_0's auc: 0.53726
best score: 0.537260279417
best iteration: 380
complete on: ISC_composer_ln

working on: ISCZ_composer_ln


After selection:
target                uint8
ISCZ_composer_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526091
[20]	valid_0's auc: 0.532601
[30]	valid_0's auc: 0.53333
[40]	valid_0's auc: 0.534097
[50]	valid_0's auc: 0.534778
[60]	valid_0's auc: 0.535114
[70]	valid_0's auc: 0.535114
[80]	valid_0's auc: 0.535433
[90]	valid_0's auc: 0.535435
[100]	valid_0's auc: 0.535643
[110]	valid_0's auc: 0.535908
[120]	valid_0's auc: 0.535958
[130]	valid_0's auc: 0.536002
[140]	valid_0's auc: 0.536065
[150]	valid_0's auc: 0.536154
[160]	valid_0's auc: 0.536392
[170]	valid_0's auc: 0.536616
[180]	valid_0's auc: 0.536651
[190]	valid_0's auc: 0.536728
[200]	valid_0's auc: 0.53682
[210]	valid_0's auc: 0.537008
[220]	valid_0's auc: 0.537115
[230]	valid_0's auc: 0.53715
[240]	valid_0's auc: 0.53717
[250]	valid_0's auc: 0.537159
[260]	valid_0's auc: 0.537126
[270]	valid_0's auc: 0.537134
[280]	valid_0's auc: 0.537153
Early stopping, best iteration is:
[231]	valid_0's auc: 0.537196
best score: 0.537195713588
best iteration: 231
complete on: ISCZ_composer_ln

working on: ISC_lyricist_ln


After selection:
target               uint8
ISC_lyricist_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529429
[20]	valid_0's auc: 0.530145
[30]	valid_0's auc: 0.529888
[40]	valid_0's auc: 0.52988
[50]	valid_0's auc: 0.530054
[60]	valid_0's auc: 0.530281
[70]	valid_0's auc: 0.530416
[80]	valid_0's auc: 0.53076
[90]	valid_0's auc: 0.530971
[100]	valid_0's auc: 0.531111
[110]	valid_0's auc: 0.531127
[120]	valid_0's auc: 0.531221
[130]	valid_0's auc: 0.531218
[140]	valid_0's auc: 0.531233
[150]	valid_0's auc: 0.531301
[160]	valid_0's auc: 0.531332
[170]	valid_0's auc: 0.531341
[180]	valid_0's auc: 0.531375
[190]	valid_0's auc: 0.531425
[200]	valid_0's auc: 0.531545
[210]	valid_0's auc: 0.531547
[220]	valid_0's auc: 0.531534
[230]	valid_0's auc: 0.531594
[240]	valid_0's auc: 0.531587
[250]	valid_0's auc: 0.531587
[260]	valid_0's auc: 0.531622
[270]	valid_0's auc: 0.531623
[280]	valid_0's auc: 0.531603
[290]	valid_0's auc: 0.531589
[300]	valid_0's auc: 0.531581
Early stopping, best iteration is:
[256]	valid_0's auc: 0.531624
best score: 0.531624474116
best iteration: 256
complete on: ISC_lyricist_ln

working on: ISCZ_lyricist_ln


After selection:
target                uint8
ISCZ_lyricist_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529429
[20]	valid_0's auc: 0.530141
[30]	valid_0's auc: 0.529754
[40]	valid_0's auc: 0.52974
[50]	valid_0's auc: 0.530354
[60]	valid_0's auc: 0.530395
[70]	valid_0's auc: 0.530351
[80]	valid_0's auc: 0.530678
[90]	valid_0's auc: 0.530729
[100]	valid_0's auc: 0.530945
[110]	valid_0's auc: 0.530955
[120]	valid_0's auc: 0.531116
[130]	valid_0's auc: 0.531193
[140]	valid_0's auc: 0.531223
[150]	valid_0's auc: 0.531292
[160]	valid_0's auc: 0.531296
[170]	valid_0's auc: 0.531325
[180]	valid_0's auc: 0.531325
[190]	valid_0's auc: 0.531411
[200]	valid_0's auc: 0.531456
[210]	valid_0's auc: 0.53149
[220]	valid_0's auc: 0.531536
[230]	valid_0's auc: 0.53157
[240]	valid_0's auc: 0.531568
[250]	valid_0's auc: 0.531565
[260]	valid_0's auc: 0.531628
[270]	valid_0's auc: 0.531627
[280]	valid_0's auc: 0.531621
[290]	valid_0's auc: 0.531619
[300]	valid_0's auc: 0.531684
[310]	valid_0's auc: 0.531679
[320]	valid_0's auc: 0.531676
[330]	valid_0's auc: 0.531651
[340]	valid_0's auc: 0.531643
[350]	valid_0's auc: 0.531644
Early stopping, best iteration is:
[305]	valid_0's auc: 0.531687
best score: 0.531687144493
best iteration: 305
complete on: ISCZ_lyricist_ln

working on: ISC_name_ln


After selection:
target           uint8
ISC_name_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506654
[20]	valid_0's auc: 0.507915
[30]	valid_0's auc: 0.508176
[40]	valid_0's auc: 0.508449
[50]	valid_0's auc: 0.508731
[60]	valid_0's auc: 0.508783
[70]	valid_0's auc: 0.509075
[80]	valid_0's auc: 0.50928
[90]	valid_0's auc: 0.509411
[100]	valid_0's auc: 0.50946
[110]	valid_0's auc: 0.509554
[120]	valid_0's auc: 0.509603
[130]	valid_0's auc: 0.509582
[140]	valid_0's auc: 0.509685
[150]	valid_0's auc: 0.509798
[160]	valid_0's auc: 0.509816
[170]	valid_0's auc: 0.509838
[180]	valid_0's auc: 0.509845
[190]	valid_0's auc: 0.509869
[200]	valid_0's auc: 0.509855
[210]	valid_0's auc: 0.509837
[220]	valid_0's auc: 0.509886
[230]	valid_0's auc: 0.509896
[240]	valid_0's auc: 0.509895
Early stopping, best iteration is:
[198]	valid_0's auc: 0.509909
best score: 0.50990911284
best iteration: 198
complete on: ISC_name_ln

working on: ISCZ_name_ln


After selection:
target            uint8
ISCZ_name_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506214
[20]	valid_0's auc: 0.507844
[30]	valid_0's auc: 0.508203
[40]	valid_0's auc: 0.508734
[50]	valid_0's auc: 0.50935
[60]	valid_0's auc: 0.509524
[70]	valid_0's auc: 0.509713
[80]	valid_0's auc: 0.509732
[90]	valid_0's auc: 0.50983
[100]	valid_0's auc: 0.509922
[110]	valid_0's auc: 0.509649
[120]	valid_0's auc: 0.509652
[130]	valid_0's auc: 0.509762
[140]	valid_0's auc: 0.510124
[150]	valid_0's auc: 0.510143
[160]	valid_0's auc: 0.510157
[170]	valid_0's auc: 0.510163
[180]	valid_0's auc: 0.510173
[190]	valid_0's auc: 0.510197
[200]	valid_0's auc: 0.510271
[210]	valid_0's auc: 0.510256
[220]	valid_0's auc: 0.510242
[230]	valid_0's auc: 0.510249
[240]	valid_0's auc: 0.51026
[250]	valid_0's auc: 0.510275
[260]	valid_0's auc: 0.510281
[270]	valid_0's auc: 0.510271
[280]	valid_0's auc: 0.510286
[290]	valid_0's auc: 0.510274
[300]	valid_0's auc: 0.510269
[310]	valid_0's auc: 0.510278
[320]	valid_0's auc: 0.510216
Early stopping, best iteration is:
[278]	valid_0's auc: 0.510287
best score: 0.510287486507
best iteration: 278
complete on: ISCZ_name_ln

working on: ISC_language_ln


After selection:
target               uint8
ISC_language_ln    float64
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
complete on: ISC_language_ln

working on: ISC_isrc_ln


After selection:
target           uint8
ISC_isrc_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503227
[20]	valid_0's auc: 0.503218
[30]	valid_0's auc: 0.503258
[40]	valid_0's auc: 0.503256
[50]	valid_0's auc: 0.503263
Early stopping, best iteration is:
[4]	valid_0's auc: 0.503264
best score: 0.503263668467
best iteration: 4
complete on: ISC_isrc_ln

working on: ISCZ_isrc_ln


After selection:
target            uint8
ISCZ_isrc_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503262
[20]	valid_0's auc: 0.503303
[30]	valid_0's auc: 0.503299
[40]	valid_0's auc: 0.50327
[50]	valid_0's auc: 0.503281
[60]	valid_0's auc: 0.503282
[70]	valid_0's auc: 0.503283
Early stopping, best iteration is:
[25]	valid_0's auc: 0.503305
best score: 0.50330451676
best iteration: 25
complete on: ISCZ_isrc_ln

working on: ISC_song_country_ln


After selection:
target                   uint8
ISC_song_country_ln    float64
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
[30]	valid_0's auc: 0.525788
[40]	valid_0's auc: 0.525927
[50]	valid_0's auc: 0.526081
[60]	valid_0's auc: 0.526106
[70]	valid_0's auc: 0.526104
[80]	valid_0's auc: 0.526118
[90]	valid_0's auc: 0.52614
[100]	valid_0's auc: 0.526139
[110]	valid_0's auc: 0.526135
[120]	valid_0's auc: 0.526136
[130]	valid_0's auc: 0.526135
[140]	valid_0's auc: 0.526037
Early stopping, best iteration is:
[96]	valid_0's auc: 0.526186
best score: 0.526186418173
best iteration: 96
complete on: ISC_song_country_ln

working on: ISCZ_song_country_ln


After selection:
target                    uint8
ISCZ_song_country_ln    float64
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
[30]	valid_0's auc: 0.525815
[40]	valid_0's auc: 0.525886
[50]	valid_0's auc: 0.525884
[60]	valid_0's auc: 0.526085
[70]	valid_0's auc: 0.526108
[80]	valid_0's auc: 0.52613
[90]	valid_0's auc: 0.526172
[100]	valid_0's auc: 0.526164
[110]	valid_0's auc: 0.526153
[120]	valid_0's auc: 0.526052
[130]	valid_0's auc: 0.526052
Early stopping, best iteration is:
[88]	valid_0's auc: 0.526175
best score: 0.526175441658
best iteration: 88
complete on: ISCZ_song_country_ln

working on: ISC_rc_ln


After selection:
target         uint8
ISC_rc_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.532782
[20]	valid_0's auc: 0.534685
[30]	valid_0's auc: 0.536354
[40]	valid_0's auc: 0.53761
[50]	valid_0's auc: 0.538404
[60]	valid_0's auc: 0.539053
[70]	valid_0's auc: 0.539541
[80]	valid_0's auc: 0.539652
[90]	valid_0's auc: 0.540225
[100]	valid_0's auc: 0.540423
[110]	valid_0's auc: 0.540471
[120]	valid_0's auc: 0.540891
[130]	valid_0's auc: 0.541026
[140]	valid_0's auc: 0.540954
[150]	valid_0's auc: 0.541497
[160]	valid_0's auc: 0.541808
[170]	valid_0's auc: 0.541262
[180]	valid_0's auc: 0.541938
[190]	valid_0's auc: 0.541314
[200]	valid_0's auc: 0.541385
[210]	valid_0's auc: 0.542006
[220]	valid_0's auc: 0.541401
[230]	valid_0's auc: 0.542042
[240]	valid_0's auc: 0.542046
[250]	valid_0's auc: 0.542083
[260]	valid_0's auc: 0.542103
[270]	valid_0's auc: 0.542109
[280]	valid_0's auc: 0.542101
[290]	valid_0's auc: 0.54211
[300]	valid_0's auc: 0.542092
[310]	valid_0's auc: 0.542109
[320]	valid_0's auc: 0.542113
[330]	valid_0's auc: 0.542129
[340]	valid_0's auc: 0.542103
[350]	valid_0's auc: 0.542115
[360]	valid_0's auc: 0.542123
[370]	valid_0's auc: 0.542085
[380]	valid_0's auc: 0.542089
Early stopping, best iteration is:
[333]	valid_0's auc: 0.542144
best score: 0.54214397067
best iteration: 333
complete on: ISC_rc_ln

working on: ISCZ_rc_ln


After selection:
target          uint8
ISCZ_rc_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.533004
[20]	valid_0's auc: 0.537151
[30]	valid_0's auc: 0.537857
[40]	valid_0's auc: 0.539077
[50]	valid_0's auc: 0.540127
[60]	valid_0's auc: 0.540261
[70]	valid_0's auc: 0.540134
[80]	valid_0's auc: 0.540219
[90]	valid_0's auc: 0.540414
[100]	valid_0's auc: 0.54041
[110]	valid_0's auc: 0.540217
Early stopping, best iteration is:
[65]	valid_0's auc: 0.540623
best score: 0.540623037823
best iteration: 65
complete on: ISCZ_rc_ln

working on: ISC_isrc_rest_ln


After selection:
target                uint8
ISC_isrc_rest_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.511454
[20]	valid_0's auc: 0.514476
[30]	valid_0's auc: 0.515266
[40]	valid_0's auc: 0.515669
[50]	valid_0's auc: 0.516813
[60]	valid_0's auc: 0.517207
[70]	valid_0's auc: 0.517579
[80]	valid_0's auc: 0.517772
[90]	valid_0's auc: 0.517821
[100]	valid_0's auc: 0.517477
[110]	valid_0's auc: 0.517943
[120]	valid_0's auc: 0.517808
[130]	valid_0's auc: 0.51799
Early stopping, best iteration is:
[84]	valid_0's auc: 0.518074
best score: 0.518073676422
best iteration: 84
complete on: ISC_isrc_rest_ln

working on: ISCZ_isrc_rest_ln


After selection:
target                 uint8
ISCZ_isrc_rest_ln    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512835
[20]	valid_0's auc: 0.514511
[30]	valid_0's auc: 0.515193
[40]	valid_0's auc: 0.515908
[50]	valid_0's auc: 0.516988
[60]	valid_0's auc: 0.517431
[70]	valid_0's auc: 0.517439
[80]	valid_0's auc: 0.517835
[90]	valid_0's auc: 0.517394
[100]	valid_0's auc: 0.517547
[110]	valid_0's auc: 0.517867
[120]	valid_0's auc: 0.517858
[130]	valid_0's auc: 0.517895
[140]	valid_0's auc: 0.517947
[150]	valid_0's auc: 0.517807
[160]	valid_0's auc: 0.517552
[170]	valid_0's auc: 0.517661
[180]	valid_0's auc: 0.517626
Early stopping, best iteration is:
[137]	valid_0's auc: 0.518014
best score: 0.518013659578
best iteration: 137
complete on: ISCZ_isrc_rest_ln

working on: ISC_song_year_ln


After selection:
target                uint8
ISC_song_year_ln    float64
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
[42]	valid_0's auc: 0.538077
best score: 0.538077320024
best iteration: 42
complete on: ISC_song_year_ln

working on: ISCZ_song_year_ln


After selection:
target                 uint8
ISCZ_song_year_ln    float64
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
Early stopping, best iteration is:
[93]	valid_0's auc: 0.538055
best score: 0.538055201301
best iteration: 93
complete on: ISCZ_song_year_ln

                          ISC_isrc_ln:  0.503263668467
                         ISCZ_isrc_ln:  0.50330451676
                          ISC_name_ln:  0.50990911284
                         ISCZ_name_ln:  0.510287486507
                    ISCZ_isrc_rest_ln:  0.518013659578
                     ISC_isrc_rest_ln:  0.518073676422
                  ISC_top3_in_song_ln:  0.524209239032
                      ISC_language_ln:  0.524767621195
                 ISCZ_song_country_ln:  0.526175441658
                  ISC_top1_in_song_ln:  0.526185542306
                  ISC_song_country_ln:  0.526186418173
                     ISC_genre_ids_ln:  0.52758560988
                    ISCZ_genre_ids_ln:  0.527671906756
                  ISC_top2_in_song_ln:  0.528211756335
                       song_length_ln:  0.529798839113
                      ISC_lyricist_ln:  0.531624474116
                     ISCZ_lyricist_ln:  0.531687144493
                     ISCZ_composer_ln:  0.537195713588
                      ISC_composer_ln:  0.537260279417
                    ISCZ_song_year_ln:  0.538055201301
                         song_year_ln:  0.538060450866
                     ISC_song_year_ln:  0.538077320024
                   ISC_artist_name_ln:  0.539143949452
                  ISCZ_artist_name_ln:  0.540000784154
                           ISCZ_rc_ln:  0.540623037823
                            ISC_rc_ln:  0.54214397067

[timer]: complete in 46m 21s

Process finished with exit code 0
'''
