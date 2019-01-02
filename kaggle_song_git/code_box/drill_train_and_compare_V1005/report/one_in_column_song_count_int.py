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
target                 uint8
song_length          float64
song_year              int64
ISC_genre_ids          int64
ISCZ_genre_ids         int64
ISC_top1_in_song       int64
ISC_top2_in_song       int64
ISC_top3_in_song       int64
ISC_artist_name        int64
ISCZ_artist_name       int64
ISC_composer           int64
ISCZ_composer          int64
ISC_lyricist           int64
ISCZ_lyricist          int64
ISC_name               int64
ISCZ_name              int64
ISC_language           int64
ISC_isrc               int64
ISCZ_isrc              int64
ISC_song_country       int64
ISCZ_song_country      int64
ISC_rc                 int64
ISCZ_rc                int64
ISC_isrc_rest          int64
ISCZ_isrc_rest         int64
ISC_song_year          int64
ISCZ_song_year         int64
dtype: object
number of rows: 7377418
number of columns: 27
working on: song_length


After selection:
target           uint8
song_length    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.518111
[20]	valid_0's auc: 0.52097
[30]	valid_0's auc: 0.52369
[40]	valid_0's auc: 0.526169
[50]	valid_0's auc: 0.526896
[60]	valid_0's auc: 0.527425
[70]	valid_0's auc: 0.527927
[80]	valid_0's auc: 0.528747
[90]	valid_0's auc: 0.529009
[100]	valid_0's auc: 0.529665
[110]	valid_0's auc: 0.530119
[120]	valid_0's auc: 0.530315
[130]	valid_0's auc: 0.530341
[140]	valid_0's auc: 0.530493
[150]	valid_0's auc: 0.530464
[160]	valid_0's auc: 0.53047
[170]	valid_0's auc: 0.530462
[180]	valid_0's auc: 0.53045
[190]	valid_0's auc: 0.530535
[200]	valid_0's auc: 0.530576
[210]	valid_0's auc: 0.530571
[220]	valid_0's auc: 0.530543
[230]	valid_0's auc: 0.53049
[240]	valid_0's auc: 0.530444
[250]	valid_0's auc: 0.530512
[260]	valid_0's auc: 0.530525
Early stopping, best iteration is:
[213]	valid_0's auc: 0.530617
best score: 0.530617005442
best iteration: 213
complete on: song_length

working on: song_year


After selection:
target       uint8
song_year    int64
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
best score: 0.53806045281
best iteration: 23
complete on: song_year

working on: ISC_genre_ids


After selection:
target           uint8
ISC_genre_ids    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526498
[20]	valid_0's auc: 0.527
[30]	valid_0's auc: 0.5271
[40]	valid_0's auc: 0.527253
[50]	valid_0's auc: 0.527322
[60]	valid_0's auc: 0.527332
[70]	valid_0's auc: 0.527556
[80]	valid_0's auc: 0.527593
[90]	valid_0's auc: 0.527624
[100]	valid_0's auc: 0.527615
[110]	valid_0's auc: 0.527608
[120]	valid_0's auc: 0.527603
[130]	valid_0's auc: 0.527615
[140]	valid_0's auc: 0.527611
Early stopping, best iteration is:
[92]	valid_0's auc: 0.527627
best score: 0.52762699931
best iteration: 92
complete on: ISC_genre_ids

working on: ISCZ_genre_ids


After selection:
target            uint8
ISCZ_genre_ids    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525727
[20]	valid_0's auc: 0.526842
[30]	valid_0's auc: 0.526988
[40]	valid_0's auc: 0.527189
[50]	valid_0's auc: 0.527166
[60]	valid_0's auc: 0.52719
[70]	valid_0's auc: 0.527225
[80]	valid_0's auc: 0.527282
[90]	valid_0's auc: 0.527296
[100]	valid_0's auc: 0.527292
[110]	valid_0's auc: 0.527296
[120]	valid_0's auc: 0.527309
[130]	valid_0's auc: 0.527583
[140]	valid_0's auc: 0.527585
[150]	valid_0's auc: 0.527589
[160]	valid_0's auc: 0.527593
[170]	valid_0's auc: 0.527596
[180]	valid_0's auc: 0.527606
[190]	valid_0's auc: 0.527601
[200]	valid_0's auc: 0.527606
[210]	valid_0's auc: 0.52761
[220]	valid_0's auc: 0.527609
[230]	valid_0's auc: 0.527609
[240]	valid_0's auc: 0.527662
[250]	valid_0's auc: 0.527669
[260]	valid_0's auc: 0.527664
[270]	valid_0's auc: 0.527624
[280]	valid_0's auc: 0.527641
[290]	valid_0's auc: 0.527636
Early stopping, best iteration is:
[245]	valid_0's auc: 0.527671
best score: 0.527671219365
best iteration: 245
complete on: ISCZ_genre_ids

working on: ISC_top1_in_song


After selection:
target              uint8
ISC_top1_in_song    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525721
[20]	valid_0's auc: 0.525779
[30]	valid_0's auc: 0.525856
[40]	valid_0's auc: 0.526198
[50]	valid_0's auc: 0.526031
[60]	valid_0's auc: 0.526087
[70]	valid_0's auc: 0.526084
[80]	valid_0's auc: 0.526088
[90]	valid_0's auc: 0.526144
Early stopping, best iteration is:
[45]	valid_0's auc: 0.526217
best score: 0.526216813637
best iteration: 45
complete on: ISC_top1_in_song

working on: ISC_top2_in_song


After selection:
target              uint8
ISC_top2_in_song    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.527641
[20]	valid_0's auc: 0.52821
[30]	valid_0's auc: 0.527929
[40]	valid_0's auc: 0.527939
[50]	valid_0's auc: 0.527944
[60]	valid_0's auc: 0.527996
[70]	valid_0's auc: 0.528302
[80]	valid_0's auc: 0.528316
[90]	valid_0's auc: 0.528328
[100]	valid_0's auc: 0.528328
[110]	valid_0's auc: 0.528377
[120]	valid_0's auc: 0.52838
[130]	valid_0's auc: 0.528382
[140]	valid_0's auc: 0.528378
[150]	valid_0's auc: 0.528378
[160]	valid_0's auc: 0.528378
[170]	valid_0's auc: 0.528378
Early stopping, best iteration is:
[121]	valid_0's auc: 0.528384
best score: 0.528383803801
best iteration: 121
complete on: ISC_top2_in_song

working on: ISC_top3_in_song


After selection:
target              uint8
ISC_top3_in_song    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.523079
[20]	valid_0's auc: 0.523669
[30]	valid_0's auc: 0.523676
[40]	valid_0's auc: 0.523726
[50]	valid_0's auc: 0.523738
[60]	valid_0's auc: 0.523761
[70]	valid_0's auc: 0.523778
[80]	valid_0's auc: 0.524066
[90]	valid_0's auc: 0.524074
[100]	valid_0's auc: 0.524125
[110]	valid_0's auc: 0.524125
[120]	valid_0's auc: 0.524173
[130]	valid_0's auc: 0.524173
[140]	valid_0's auc: 0.524185
[150]	valid_0's auc: 0.524185
[160]	valid_0's auc: 0.524184
[170]	valid_0's auc: 0.524197
[180]	valid_0's auc: 0.5242
[190]	valid_0's auc: 0.5242
[200]	valid_0's auc: 0.524201
[210]	valid_0's auc: 0.524201
[220]	valid_0's auc: 0.524201
[230]	valid_0's auc: 0.524209
[240]	valid_0's auc: 0.524209
[250]	valid_0's auc: 0.524209
[260]	valid_0's auc: 0.524209
[270]	valid_0's auc: 0.524208
[280]	valid_0's auc: 0.524207
Early stopping, best iteration is:
[230]	valid_0's auc: 0.524209
best score: 0.524209242714
best iteration: 230
complete on: ISC_top3_in_song

working on: ISC_artist_name


After selection:
target             uint8
ISC_artist_name    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.522074
[20]	valid_0's auc: 0.526539
[30]	valid_0's auc: 0.527737
[40]	valid_0's auc: 0.530679
[50]	valid_0's auc: 0.532658
[60]	valid_0's auc: 0.533533
[70]	valid_0's auc: 0.534806
[80]	valid_0's auc: 0.535665
[90]	valid_0's auc: 0.536689
[100]	valid_0's auc: 0.537142
[110]	valid_0's auc: 0.537538
[120]	valid_0's auc: 0.537654
[130]	valid_0's auc: 0.537963
[140]	valid_0's auc: 0.538138
[150]	valid_0's auc: 0.538452
[160]	valid_0's auc: 0.538644
[170]	valid_0's auc: 0.538785
[180]	valid_0's auc: 0.538809
[190]	valid_0's auc: 0.538964
[200]	valid_0's auc: 0.539008
[210]	valid_0's auc: 0.539104
[220]	valid_0's auc: 0.539153
[230]	valid_0's auc: 0.539165
[240]	valid_0's auc: 0.539212
[250]	valid_0's auc: 0.539352
[260]	valid_0's auc: 0.539349
[270]	valid_0's auc: 0.539414
[280]	valid_0's auc: 0.539498
[290]	valid_0's auc: 0.539538
[300]	valid_0's auc: 0.539557
[310]	valid_0's auc: 0.539544
[320]	valid_0's auc: 0.539567
[330]	valid_0's auc: 0.539584
[340]	valid_0's auc: 0.539605
[350]	valid_0's auc: 0.539614
[360]	valid_0's auc: 0.539614
[370]	valid_0's auc: 0.539635
[380]	valid_0's auc: 0.539617
[390]	valid_0's auc: 0.539597
[400]	valid_0's auc: 0.539595
[410]	valid_0's auc: 0.539598
[420]	valid_0's auc: 0.539593
Early stopping, best iteration is:
[371]	valid_0's auc: 0.539639
best score: 0.539639472699
best iteration: 371
complete on: ISC_artist_name

working on: ISCZ_artist_name


After selection:
target              uint8
ISCZ_artist_name    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.520638
[20]	valid_0's auc: 0.527053
[30]	valid_0's auc: 0.528705
[40]	valid_0's auc: 0.530819
[50]	valid_0's auc: 0.533068
[60]	valid_0's auc: 0.534164
[70]	valid_0's auc: 0.53535
[80]	valid_0's auc: 0.535799
[90]	valid_0's auc: 0.536572
[100]	valid_0's auc: 0.537281
[110]	valid_0's auc: 0.537894
[120]	valid_0's auc: 0.538274
[130]	valid_0's auc: 0.538725
[140]	valid_0's auc: 0.538919
[150]	valid_0's auc: 0.539265
[160]	valid_0's auc: 0.539337
[170]	valid_0's auc: 0.539404
[180]	valid_0's auc: 0.539525
[190]	valid_0's auc: 0.539575
[200]	valid_0's auc: 0.5397
[210]	valid_0's auc: 0.539837
[220]	valid_0's auc: 0.53994
[230]	valid_0's auc: 0.539965
[240]	valid_0's auc: 0.540037
[250]	valid_0's auc: 0.539985
[260]	valid_0's auc: 0.540038
[270]	valid_0's auc: 0.54012
[280]	valid_0's auc: 0.540176
[290]	valid_0's auc: 0.540181
[300]	valid_0's auc: 0.540213
[310]	valid_0's auc: 0.540292
[320]	valid_0's auc: 0.540346
[330]	valid_0's auc: 0.540372
[340]	valid_0's auc: 0.540384
[350]	valid_0's auc: 0.540345
[360]	valid_0's auc: 0.540395
[370]	valid_0's auc: 0.540369
[380]	valid_0's auc: 0.540394
[390]	valid_0's auc: 0.540431
[400]	valid_0's auc: 0.540531
[410]	valid_0's auc: 0.540549
[420]	valid_0's auc: 0.540526
[430]	valid_0's auc: 0.540495
[440]	valid_0's auc: 0.5405
[450]	valid_0's auc: 0.5405
Early stopping, best iteration is:
[409]	valid_0's auc: 0.54055
best score: 0.54055033529
best iteration: 409
complete on: ISCZ_artist_name

working on: ISC_composer


After selection:
target          uint8
ISC_composer    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529859
[20]	valid_0's auc: 0.531609
[30]	valid_0's auc: 0.532922
[40]	valid_0's auc: 0.533979
[50]	valid_0's auc: 0.534908
[60]	valid_0's auc: 0.535231
[70]	valid_0's auc: 0.535424
[80]	valid_0's auc: 0.535628
[90]	valid_0's auc: 0.535724
[100]	valid_0's auc: 0.535714
[110]	valid_0's auc: 0.535979
[120]	valid_0's auc: 0.53607
[130]	valid_0's auc: 0.536112
[140]	valid_0's auc: 0.536145
[150]	valid_0's auc: 0.536192
[160]	valid_0's auc: 0.536262
[170]	valid_0's auc: 0.536266
[180]	valid_0's auc: 0.536622
[190]	valid_0's auc: 0.536717
[200]	valid_0's auc: 0.536747
[210]	valid_0's auc: 0.536753
[220]	valid_0's auc: 0.536953
[230]	valid_0's auc: 0.537032
[240]	valid_0's auc: 0.537106
[250]	valid_0's auc: 0.537141
[260]	valid_0's auc: 0.537152
[270]	valid_0's auc: 0.537187
[280]	valid_0's auc: 0.537137
[290]	valid_0's auc: 0.53719
[300]	valid_0's auc: 0.537189
[310]	valid_0's auc: 0.537183
[320]	valid_0's auc: 0.537185
[330]	valid_0's auc: 0.53723
[340]	valid_0's auc: 0.537233
[350]	valid_0's auc: 0.537242
[360]	valid_0's auc: 0.537258
[370]	valid_0's auc: 0.53726
[380]	valid_0's auc: 0.53726
[390]	valid_0's auc: 0.537263
[400]	valid_0's auc: 0.537316
[410]	valid_0's auc: 0.537289
[420]	valid_0's auc: 0.537272
[430]	valid_0's auc: 0.537285
[440]	valid_0's auc: 0.537305
Early stopping, best iteration is:
[398]	valid_0's auc: 0.537316
best score: 0.53731602596
best iteration: 398
complete on: ISC_composer

working on: ISCZ_composer


After selection:
target           uint8
ISCZ_composer    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526093
[20]	valid_0's auc: 0.531063
[30]	valid_0's auc: 0.533257
[40]	valid_0's auc: 0.533836
[50]	valid_0's auc: 0.534209
[60]	valid_0's auc: 0.534955
[70]	valid_0's auc: 0.535205
[80]	valid_0's auc: 0.535308
[90]	valid_0's auc: 0.535691
[100]	valid_0's auc: 0.535721
[110]	valid_0's auc: 0.535811
[120]	valid_0's auc: 0.535972
[130]	valid_0's auc: 0.536467
[140]	valid_0's auc: 0.536511
[150]	valid_0's auc: 0.536649
[160]	valid_0's auc: 0.536841
[170]	valid_0's auc: 0.536952
[180]	valid_0's auc: 0.53698
[190]	valid_0's auc: 0.536992
[200]	valid_0's auc: 0.537006
[210]	valid_0's auc: 0.537082
[220]	valid_0's auc: 0.537091
[230]	valid_0's auc: 0.537131
[240]	valid_0's auc: 0.537119
[250]	valid_0's auc: 0.537175
[260]	valid_0's auc: 0.537151
[270]	valid_0's auc: 0.5372
[280]	valid_0's auc: 0.537203
[290]	valid_0's auc: 0.537215
[300]	valid_0's auc: 0.537219
[310]	valid_0's auc: 0.537186
[320]	valid_0's auc: 0.537217
[330]	valid_0's auc: 0.537225
[340]	valid_0's auc: 0.537226
[350]	valid_0's auc: 0.537233
[360]	valid_0's auc: 0.537226
[370]	valid_0's auc: 0.537229
[380]	valid_0's auc: 0.537229
[390]	valid_0's auc: 0.53724
[400]	valid_0's auc: 0.53724
[410]	valid_0's auc: 0.53724
[420]	valid_0's auc: 0.53723
[430]	valid_0's auc: 0.53723
[440]	valid_0's auc: 0.537223
[450]	valid_0's auc: 0.537223
[460]	valid_0's auc: 0.537258
[470]	valid_0's auc: 0.537263
[480]	valid_0's auc: 0.537237
[490]	valid_0's auc: 0.537247
[500]	valid_0's auc: 0.537254
[510]	valid_0's auc: 0.537284
[520]	valid_0's auc: 0.537293
[530]	valid_0's auc: 0.537293
[540]	valid_0's auc: 0.537293
[550]	valid_0's auc: 0.537292
[560]	valid_0's auc: 0.537293
[570]	valid_0's auc: 0.537293
[580]	valid_0's auc: 0.537292
[590]	valid_0's auc: 0.537292
[600]	valid_0's auc: 0.537292
[610]	valid_0's auc: 0.537292
[620]	valid_0's auc: 0.537292
Early stopping, best iteration is:
[571]	valid_0's auc: 0.537294
best score: 0.537293588042
best iteration: 571
complete on: ISCZ_composer

working on: ISC_lyricist


After selection:
target          uint8
ISC_lyricist    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529533
[20]	valid_0's auc: 0.529821
[30]	valid_0's auc: 0.529733
[40]	valid_0's auc: 0.529652
[50]	valid_0's auc: 0.529968
[60]	valid_0's auc: 0.530267
[70]	valid_0's auc: 0.53045
[80]	valid_0's auc: 0.530715
[90]	valid_0's auc: 0.530875
[100]	valid_0's auc: 0.531078
[110]	valid_0's auc: 0.531093
[120]	valid_0's auc: 0.53117
[130]	valid_0's auc: 0.531287
[140]	valid_0's auc: 0.531355
[150]	valid_0's auc: 0.53129
[160]	valid_0's auc: 0.531316
[170]	valid_0's auc: 0.531295
[180]	valid_0's auc: 0.531276
[190]	valid_0's auc: 0.53132
Early stopping, best iteration is:
[143]	valid_0's auc: 0.531359
best score: 0.531359008768
best iteration: 143
complete on: ISC_lyricist

working on: ISCZ_lyricist


After selection:
target           uint8
ISCZ_lyricist    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529445
[20]	valid_0's auc: 0.529901
[30]	valid_0's auc: 0.529443
[40]	valid_0's auc: 0.529759
[50]	valid_0's auc: 0.53018
[60]	valid_0's auc: 0.530422
[70]	valid_0's auc: 0.530638
[80]	valid_0's auc: 0.5308
[90]	valid_0's auc: 0.531052
[100]	valid_0's auc: 0.531206
[110]	valid_0's auc: 0.53123
[120]	valid_0's auc: 0.531217
[130]	valid_0's auc: 0.531252
[140]	valid_0's auc: 0.531333
[150]	valid_0's auc: 0.531321
[160]	valid_0's auc: 0.531392
[170]	valid_0's auc: 0.531433
[180]	valid_0's auc: 0.531432
[190]	valid_0's auc: 0.531493
[200]	valid_0's auc: 0.531566
[210]	valid_0's auc: 0.531573
[220]	valid_0's auc: 0.531704
[230]	valid_0's auc: 0.531695
[240]	valid_0's auc: 0.531695
[250]	valid_0's auc: 0.531694
[260]	valid_0's auc: 0.531692
Early stopping, best iteration is:
[218]	valid_0's auc: 0.531704
best score: 0.531704068301
best iteration: 218
complete on: ISCZ_lyricist

working on: ISC_name


After selection:
target      uint8
ISC_name    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506614
[20]	valid_0's auc: 0.507937
[30]	valid_0's auc: 0.508423
[40]	valid_0's auc: 0.508668
[50]	valid_0's auc: 0.509038
[60]	valid_0's auc: 0.509067
[70]	valid_0's auc: 0.508873
[80]	valid_0's auc: 0.509392
[90]	valid_0's auc: 0.509366
[100]	valid_0's auc: 0.509542
[110]	valid_0's auc: 0.509615
[120]	valid_0's auc: 0.509706
[130]	valid_0's auc: 0.509854
[140]	valid_0's auc: 0.509869
[150]	valid_0's auc: 0.509923
[160]	valid_0's auc: 0.50999
[170]	valid_0's auc: 0.510078
[180]	valid_0's auc: 0.51009
[190]	valid_0's auc: 0.51014
[200]	valid_0's auc: 0.51015
[210]	valid_0's auc: 0.510163
[220]	valid_0's auc: 0.510162
[230]	valid_0's auc: 0.51018
[240]	valid_0's auc: 0.510143
[250]	valid_0's auc: 0.510133
[260]	valid_0's auc: 0.510134
[270]	valid_0's auc: 0.510145
[280]	valid_0's auc: 0.510134
Early stopping, best iteration is:
[238]	valid_0's auc: 0.510186
best score: 0.510186407599
best iteration: 238
complete on: ISC_name

working on: ISCZ_name


After selection:
target       uint8
ISCZ_name    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506189
[20]	valid_0's auc: 0.507866
[30]	valid_0's auc: 0.508113
[40]	valid_0's auc: 0.508225
[50]	valid_0's auc: 0.508506
[60]	valid_0's auc: 0.508733
[70]	valid_0's auc: 0.508872
[80]	valid_0's auc: 0.509161
[90]	valid_0's auc: 0.509458
[100]	valid_0's auc: 0.509725
[110]	valid_0's auc: 0.509754
[120]	valid_0's auc: 0.509836
[130]	valid_0's auc: 0.509921
[140]	valid_0's auc: 0.509938
[150]	valid_0's auc: 0.509958
[160]	valid_0's auc: 0.509971
[170]	valid_0's auc: 0.509972
[180]	valid_0's auc: 0.510008
[190]	valid_0's auc: 0.51005
[200]	valid_0's auc: 0.51007
[210]	valid_0's auc: 0.510062
[220]	valid_0's auc: 0.510069
[230]	valid_0's auc: 0.510061
[240]	valid_0's auc: 0.510068
[250]	valid_0's auc: 0.510077
[260]	valid_0's auc: 0.510054
[270]	valid_0's auc: 0.510053
Early stopping, best iteration is:
[224]	valid_0's auc: 0.510082
best score: 0.510081624055
best iteration: 224
complete on: ISCZ_name

working on: ISC_language


After selection:
target          uint8
ISC_language    int64
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
complete on: ISC_language

working on: ISC_isrc


After selection:
target      uint8
ISC_isrc    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503221
[20]	valid_0's auc: 0.503207
[30]	valid_0's auc: 0.503262
[40]	valid_0's auc: 0.503261
[50]	valid_0's auc: 0.503267
[60]	valid_0's auc: 0.503271
[70]	valid_0's auc: 0.503281
[80]	valid_0's auc: 0.503286
[90]	valid_0's auc: 0.503286
[100]	valid_0's auc: 0.503282
[110]	valid_0's auc: 0.503282
[120]	valid_0's auc: 0.503283
[130]	valid_0's auc: 0.503283
Early stopping, best iteration is:
[87]	valid_0's auc: 0.503286
best score: 0.50328577303
best iteration: 87
complete on: ISC_isrc

working on: ISCZ_isrc


After selection:
target       uint8
ISCZ_isrc    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503245
[20]	valid_0's auc: 0.503272
[30]	valid_0's auc: 0.503245
[40]	valid_0's auc: 0.503269
[50]	valid_0's auc: 0.503259
[60]	valid_0's auc: 0.503265
Early stopping, best iteration is:
[18]	valid_0's auc: 0.503281
best score: 0.503280957959
best iteration: 18
complete on: ISCZ_isrc

working on: ISC_song_country


After selection:
target              uint8
ISC_song_country    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525537
[20]	valid_0's auc: 0.525583
[30]	valid_0's auc: 0.525623
[40]	valid_0's auc: 0.525898
[50]	valid_0's auc: 0.525929
[60]	valid_0's auc: 0.526099
[70]	valid_0's auc: 0.526091
[80]	valid_0's auc: 0.526071
[90]	valid_0's auc: 0.526126
[100]	valid_0's auc: 0.526126
[110]	valid_0's auc: 0.526127
[120]	valid_0's auc: 0.526029
[130]	valid_0's auc: 0.526029
Early stopping, best iteration is:
[86]	valid_0's auc: 0.526136
best score: 0.526136016164
best iteration: 86
complete on: ISC_song_country

working on: ISCZ_song_country


After selection:
target               uint8
ISCZ_song_country    int64
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
[40]	valid_0's auc: 0.525888
[50]	valid_0's auc: 0.525884
[60]	valid_0's auc: 0.526077
[70]	valid_0's auc: 0.5261
[80]	valid_0's auc: 0.526122
[90]	valid_0's auc: 0.526107
[100]	valid_0's auc: 0.526151
[110]	valid_0's auc: 0.526053
[120]	valid_0's auc: 0.526038
[130]	valid_0's auc: 0.526029
[140]	valid_0's auc: 0.526026
[150]	valid_0's auc: 0.526029
Early stopping, best iteration is:
[102]	valid_0's auc: 0.526152
best score: 0.526152404452
best iteration: 102
complete on: ISCZ_song_country

working on: ISC_rc


After selection:
target    uint8
ISC_rc    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.53276
[20]	valid_0's auc: 0.534319
[30]	valid_0's auc: 0.535717
[40]	valid_0's auc: 0.536996
[50]	valid_0's auc: 0.538026
[60]	valid_0's auc: 0.538674
[70]	valid_0's auc: 0.539017
[80]	valid_0's auc: 0.539452
[90]	valid_0's auc: 0.539735
[100]	valid_0's auc: 0.539985
[110]	valid_0's auc: 0.539983
[120]	valid_0's auc: 0.540292
[130]	valid_0's auc: 0.540693
[140]	valid_0's auc: 0.540705
[150]	valid_0's auc: 0.540712
[160]	valid_0's auc: 0.540667
[170]	valid_0's auc: 0.540631
[180]	valid_0's auc: 0.540648
[190]	valid_0's auc: 0.541248
[200]	valid_0's auc: 0.541344
[210]	valid_0's auc: 0.541347
[220]	valid_0's auc: 0.54132
[230]	valid_0's auc: 0.54128
[240]	valid_0's auc: 0.541294
[250]	valid_0's auc: 0.541305
Early stopping, best iteration is:
[209]	valid_0's auc: 0.541377
best score: 0.541376994377
best iteration: 209
complete on: ISC_rc

working on: ISCZ_rc


After selection:
target     uint8
ISCZ_rc    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.532586
[20]	valid_0's auc: 0.536123
[30]	valid_0's auc: 0.538946
[40]	valid_0's auc: 0.538994
[50]	valid_0's auc: 0.539766
[60]	valid_0's auc: 0.540106
[70]	valid_0's auc: 0.540445
[80]	valid_0's auc: 0.540428
[90]	valid_0's auc: 0.540899
[100]	valid_0's auc: 0.541469
[110]	valid_0's auc: 0.541653
[120]	valid_0's auc: 0.541665
[130]	valid_0's auc: 0.541682
[140]	valid_0's auc: 0.541781
[150]	valid_0's auc: 0.541771
[160]	valid_0's auc: 0.541768
[170]	valid_0's auc: 0.541864
[180]	valid_0's auc: 0.541831
[190]	valid_0's auc: 0.541723
[200]	valid_0's auc: 0.541915
[210]	valid_0's auc: 0.541831
[220]	valid_0's auc: 0.541829
[230]	valid_0's auc: 0.541769
[240]	valid_0's auc: 0.541799
[250]	valid_0's auc: 0.541862
Early stopping, best iteration is:
[205]	valid_0's auc: 0.541932
best score: 0.541931933803
best iteration: 205
complete on: ISCZ_rc

working on: ISC_isrc_rest


After selection:
target           uint8
ISC_isrc_rest    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.510871
[20]	valid_0's auc: 0.511084
[30]	valid_0's auc: 0.514634
[40]	valid_0's auc: 0.514454
[50]	valid_0's auc: 0.515661
[60]	valid_0's auc: 0.516904
[70]	valid_0's auc: 0.516896
[80]	valid_0's auc: 0.516934
[90]	valid_0's auc: 0.517363
[100]	valid_0's auc: 0.51756
[110]	valid_0's auc: 0.517693
[120]	valid_0's auc: 0.517733
[130]	valid_0's auc: 0.517683
[140]	valid_0's auc: 0.517583
[150]	valid_0's auc: 0.517631
[160]	valid_0's auc: 0.517674
Early stopping, best iteration is:
[119]	valid_0's auc: 0.517983
best score: 0.517982829119
best iteration: 119
complete on: ISC_isrc_rest

working on: ISCZ_isrc_rest


After selection:
target            uint8
ISCZ_isrc_rest    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512231
[20]	valid_0's auc: 0.514147
[30]	valid_0's auc: 0.516094
[40]	valid_0's auc: 0.516813
[50]	valid_0's auc: 0.517379
[60]	valid_0's auc: 0.518217
[70]	valid_0's auc: 0.51828
[80]	valid_0's auc: 0.518163
[90]	valid_0's auc: 0.518373
[100]	valid_0's auc: 0.518975
[110]	valid_0's auc: 0.518999
[120]	valid_0's auc: 0.519227
[130]	valid_0's auc: 0.519225
[140]	valid_0's auc: 0.51935
[150]	valid_0's auc: 0.519437
[160]	valid_0's auc: 0.519555
[170]	valid_0's auc: 0.519585
[180]	valid_0's auc: 0.519247
[190]	valid_0's auc: 0.519124
[200]	valid_0's auc: 0.519314
[210]	valid_0's auc: 0.519273
[220]	valid_0's auc: 0.519159
Early stopping, best iteration is:
[172]	valid_0's auc: 0.519601
best score: 0.519601095544
best iteration: 172
complete on: ISCZ_isrc_rest

working on: ISC_song_year


After selection:
target           uint8
ISC_song_year    int64
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
Early stopping, best iteration is:
[39]	valid_0's auc: 0.538077
best score: 0.538077461366
best iteration: 39
complete on: ISC_song_year

working on: ISCZ_song_year


After selection:
target            uint8
ISCZ_song_year    int64
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
[118]	valid_0's auc: 0.538055
best score: 0.538055228378
best iteration: 118
complete on: ISCZ_song_year

                            ISCZ_isrc:  0.503280957959
                             ISC_isrc:  0.50328577303
                            ISCZ_name:  0.510081624055
                             ISC_name:  0.510186407599
                        ISC_isrc_rest:  0.517982829119
                       ISCZ_isrc_rest:  0.519601095544
                     ISC_top3_in_song:  0.524209242714
                         ISC_language:  0.524767621195
                     ISC_song_country:  0.526136016164
                    ISCZ_song_country:  0.526152404452
                     ISC_top1_in_song:  0.526216813637
                        ISC_genre_ids:  0.52762699931
                       ISCZ_genre_ids:  0.527671219365
                     ISC_top2_in_song:  0.528383803801
                          song_length:  0.530617005442
                         ISC_lyricist:  0.531359008768
                        ISCZ_lyricist:  0.531704068301
                        ISCZ_composer:  0.537293588042
                         ISC_composer:  0.53731602596
                       ISCZ_song_year:  0.538055228378
                            song_year:  0.53806045281
                        ISC_song_year:  0.538077461366
                      ISC_artist_name:  0.539639472699
                     ISCZ_artist_name:  0.54055033529
                               ISC_rc:  0.541376994377
                              ISCZ_rc:  0.541931933803

[timer]: complete in 40m 57s

Process finished with exit code 0
'''