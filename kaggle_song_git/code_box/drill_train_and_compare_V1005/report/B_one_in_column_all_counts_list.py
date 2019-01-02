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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/drill_train_and_compare_V1005/one_in_column_V1002.py
What we got:
target                          uint8
membership_days                 int64
bd_log10                      float64
expiration_month_log10        float64
IMC_expiration_month_log10    float64
bd_fixed_log10                float64
age_guess_log10               float64
bd_range_log10                float64
age_guess_range_log10         float64
bd_fixed_range_log10          float64
IMC_bd_log10                  float64
IMC_bd_fixed_log10            float64
IMC_age_guess_log10           float64
IMC_bd_range_log10            float64
IMC_bd_fixed_range_log10      float64
IMC_age_guess_range_log10     float64
IMC_membership_days_log10     float64
song_year                       int64
ISC_genre_ids                   int64
ISC_top1_in_song                int64
ISC_top2_in_song                int64
ISC_top3_in_song                int64
ISCZ_artist_name                int64
ISC_composer                    int64
ISCZ_lyricist                   int64
ISC_language                    int64
ISCZ_rc                         int64
ISCZ_isrc_rest                  int64
ISC_song_year                   int64
ISCZ_song_year                  int64
song_length_log10             float64
ISCZ_genre_ids_log10          float64
ISC_artist_name_log10         float64
ISCZ_composer_log10           float64
ISC_lyricist_log10            float64
ISC_name_log10                float64
ISCZ_name_ln                  float64
ISC_song_country_ln           float64
ISCZ_song_country_log10       float64
ISC_rc_ln                     float64
ISC_isrc_rest_log10           float64
dtype: object
number of rows: 7377418
number of columns: 41
working on: membership_days


After selection:
target             uint8
membership_days    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.511595
[20]	valid_0's auc: 0.513722
[30]	valid_0's auc: 0.514783
[40]	valid_0's auc: 0.515079
[50]	valid_0's auc: 0.516269
[60]	valid_0's auc: 0.516746
[70]	valid_0's auc: 0.51692
[80]	valid_0's auc: 0.517184
[90]	valid_0's auc: 0.517213
[100]	valid_0's auc: 0.517223
[110]	valid_0's auc: 0.517394
[120]	valid_0's auc: 0.517559
[130]	valid_0's auc: 0.517729
[140]	valid_0's auc: 0.517856
[150]	valid_0's auc: 0.51779
[160]	valid_0's auc: 0.517851
[170]	valid_0's auc: 0.51788
[180]	valid_0's auc: 0.517897
[190]	valid_0's auc: 0.51786
[200]	valid_0's auc: 0.51788
[210]	valid_0's auc: 0.518021
[220]	valid_0's auc: 0.517998
[230]	valid_0's auc: 0.517955
[240]	valid_0's auc: 0.517955
[250]	valid_0's auc: 0.517949
[260]	valid_0's auc: 0.51792
Early stopping, best iteration is:
[210]	valid_0's auc: 0.518021
best score: 0.518020750052
best iteration: 210
complete on: membership_days

working on: bd_log10


After selection:
target        uint8
bd_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.515973
[20]	valid_0's auc: 0.516215
[30]	valid_0's auc: 0.5162
[40]	valid_0's auc: 0.516312
[50]	valid_0's auc: 0.516317
[60]	valid_0's auc: 0.516321
[70]	valid_0's auc: 0.516191
[80]	valid_0's auc: 0.516107
[90]	valid_0's auc: 0.516111
Early stopping, best iteration is:
[45]	valid_0's auc: 0.516325
best score: 0.516325224149
best iteration: 45
complete on: bd_log10

working on: expiration_month_log10


After selection:
target                      uint8
expiration_month_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513365
[20]	valid_0's auc: 0.513493
[30]	valid_0's auc: 0.513493
[40]	valid_0's auc: 0.513512
[50]	valid_0's auc: 0.513512
[60]	valid_0's auc: 0.513447
[70]	valid_0's auc: 0.513447
[80]	valid_0's auc: 0.513447
Early stopping, best iteration is:
[37]	valid_0's auc: 0.513512
best score: 0.51351211197
best iteration: 37
complete on: expiration_month_log10

working on: IMC_expiration_month_log10


After selection:
target                          uint8
IMC_expiration_month_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513469
[20]	valid_0's auc: 0.513488
[30]	valid_0's auc: 0.513488
[40]	valid_0's auc: 0.513423
[50]	valid_0's auc: 0.513423
[60]	valid_0's auc: 0.513447
Early stopping, best iteration is:
[17]	valid_0's auc: 0.513488
best score: 0.513488378472
best iteration: 17
complete on: IMC_expiration_month_log10

working on: bd_fixed_log10


After selection:
target              uint8
bd_fixed_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51525
[20]	valid_0's auc: 0.515585
[30]	valid_0's auc: 0.515543
[40]	valid_0's auc: 0.515547
[50]	valid_0's auc: 0.515489
[60]	valid_0's auc: 0.515479
Early stopping, best iteration is:
[19]	valid_0's auc: 0.515803
best score: 0.515803351591
best iteration: 19
complete on: bd_fixed_log10

working on: age_guess_log10


After selection:
target               uint8
age_guess_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51346
[20]	valid_0's auc: 0.513576
[30]	valid_0's auc: 0.513534
[40]	valid_0's auc: 0.513493
[50]	valid_0's auc: 0.513466
[60]	valid_0's auc: 0.513527
[70]	valid_0's auc: 0.513644
[80]	valid_0's auc: 0.513634
[90]	valid_0's auc: 0.513635
[100]	valid_0's auc: 0.513612
[110]	valid_0's auc: 0.51361
[120]	valid_0's auc: 0.513604
Early stopping, best iteration is:
[71]	valid_0's auc: 0.513652
best score: 0.513651814109
best iteration: 71
complete on: age_guess_log10

working on: bd_range_log10


After selection:
target              uint8
bd_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512364
[20]	valid_0's auc: 0.512364
[30]	valid_0's auc: 0.512364
[40]	valid_0's auc: 0.512364
[50]	valid_0's auc: 0.512364
Early stopping, best iteration is:
[1]	valid_0's auc: 0.512364
best score: 0.512363873294
best iteration: 1
complete on: bd_range_log10

working on: age_guess_range_log10


After selection:
target                     uint8
age_guess_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.508907
[20]	valid_0's auc: 0.508907
[30]	valid_0's auc: 0.508907
[40]	valid_0's auc: 0.508907
[50]	valid_0's auc: 0.508907
Early stopping, best iteration is:
[1]	valid_0's auc: 0.508907
best score: 0.508907354054
best iteration: 1
complete on: age_guess_range_log10

working on: bd_fixed_range_log10


After selection:
target                    uint8
bd_fixed_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512343
[20]	valid_0's auc: 0.512343
[30]	valid_0's auc: 0.512343
[40]	valid_0's auc: 0.512343
[50]	valid_0's auc: 0.512343
Early stopping, best iteration is:
[1]	valid_0's auc: 0.512343
best score: 0.512342947511
best iteration: 1
complete on: bd_fixed_range_log10

working on: IMC_bd_log10


After selection:
target            uint8
IMC_bd_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51311
[20]	valid_0's auc: 0.513655
[30]	valid_0's auc: 0.513422
[40]	valid_0's auc: 0.513494
[50]	valid_0's auc: 0.514583
[60]	valid_0's auc: 0.514682
[70]	valid_0's auc: 0.51472
[80]	valid_0's auc: 0.514792
[90]	valid_0's auc: 0.514733
[100]	valid_0's auc: 0.514684
[110]	valid_0's auc: 0.514717
[120]	valid_0's auc: 0.514718
[130]	valid_0's auc: 0.514726
Early stopping, best iteration is:
[81]	valid_0's auc: 0.514827
best score: 0.514826834671
best iteration: 81
complete on: IMC_bd_log10

working on: IMC_bd_fixed_log10


After selection:
target                  uint8
IMC_bd_fixed_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512708
[20]	valid_0's auc: 0.514267
[30]	valid_0's auc: 0.514304
[40]	valid_0's auc: 0.514245
[50]	valid_0's auc: 0.51416
[60]	valid_0's auc: 0.514299
[70]	valid_0's auc: 0.514335
Early stopping, best iteration is:
[29]	valid_0's auc: 0.514374
best score: 0.514373696215
best iteration: 29
complete on: IMC_bd_fixed_log10

working on: IMC_age_guess_log10


After selection:
target                   uint8
IMC_age_guess_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51227
[20]	valid_0's auc: 0.512304
[30]	valid_0's auc: 0.51298
[40]	valid_0's auc: 0.513346
[50]	valid_0's auc: 0.513465
[60]	valid_0's auc: 0.513425
[70]	valid_0's auc: 0.513595
[80]	valid_0's auc: 0.513592
[90]	valid_0's auc: 0.513586
[100]	valid_0's auc: 0.513585
[110]	valid_0's auc: 0.513639
[120]	valid_0's auc: 0.513595
[130]	valid_0's auc: 0.513608
[140]	valid_0's auc: 0.513602
[150]	valid_0's auc: 0.513603
[160]	valid_0's auc: 0.513609
Early stopping, best iteration is:
[110]	valid_0's auc: 0.513639
best score: 0.513638951741
best iteration: 110
complete on: IMC_age_guess_log10

working on: IMC_bd_range_log10


After selection:
target                  uint8
IMC_bd_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512364
[20]	valid_0's auc: 0.512364
[30]	valid_0's auc: 0.512364
[40]	valid_0's auc: 0.512364
[50]	valid_0's auc: 0.512364
Early stopping, best iteration is:
[1]	valid_0's auc: 0.512364
best score: 0.512363873294
best iteration: 1
complete on: IMC_bd_range_log10

working on: IMC_bd_fixed_range_log10


After selection:
target                        uint8
IMC_bd_fixed_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512343
[20]	valid_0's auc: 0.512343
[30]	valid_0's auc: 0.512343
[40]	valid_0's auc: 0.512343
[50]	valid_0's auc: 0.512343
Early stopping, best iteration is:
[1]	valid_0's auc: 0.512343
best score: 0.512342947511
best iteration: 1
complete on: IMC_bd_fixed_range_log10

working on: IMC_age_guess_range_log10


After selection:
target                         uint8
IMC_age_guess_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.508907
[20]	valid_0's auc: 0.508907
[30]	valid_0's auc: 0.508907
[40]	valid_0's auc: 0.508907
[50]	valid_0's auc: 0.508907
Early stopping, best iteration is:
[1]	valid_0's auc: 0.508907
best score: 0.508907354054
best iteration: 1
complete on: IMC_age_guess_range_log10

working on: IMC_membership_days_log10


After selection:
target                         uint8
IMC_membership_days_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512455
[20]	valid_0's auc: 0.512795
[30]	valid_0's auc: 0.512694
[40]	valid_0's auc: 0.512643
[50]	valid_0's auc: 0.512637
[60]	valid_0's auc: 0.512711
[70]	valid_0's auc: 0.51243
Early stopping, best iteration is:
[22]	valid_0's auc: 0.513053
best score: 0.51305298913
best iteration: 22
complete on: IMC_membership_days_log10

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
best score: 0.538060446124
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
[10]	valid_0's auc: 0.526477
[20]	valid_0's auc: 0.52653
[30]	valid_0's auc: 0.527141
[40]	valid_0's auc: 0.527268
[50]	valid_0's auc: 0.527287
[60]	valid_0's auc: 0.527362
[70]	valid_0's auc: 0.52736
[80]	valid_0's auc: 0.527281
[90]	valid_0's auc: 0.527295
[100]	valid_0's auc: 0.52731
[110]	valid_0's auc: 0.527294
Early stopping, best iteration is:
[60]	valid_0's auc: 0.527362
best score: 0.52736185413
best iteration: 60
complete on: ISC_genre_ids

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
[10]	valid_0's auc: 0.525723
[20]	valid_0's auc: 0.525776
[30]	valid_0's auc: 0.52586
[40]	valid_0's auc: 0.526202
[50]	valid_0's auc: 0.52603
[60]	valid_0's auc: 0.52603
[70]	valid_0's auc: 0.526061
[80]	valid_0's auc: 0.526085
[90]	valid_0's auc: 0.526145
Early stopping, best iteration is:
[44]	valid_0's auc: 0.526218
best score: 0.526218173781
best iteration: 44
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
[10]	valid_0's auc: 0.527636
[20]	valid_0's auc: 0.52821
[30]	valid_0's auc: 0.527934
[40]	valid_0's auc: 0.527944
[50]	valid_0's auc: 0.527964
[60]	valid_0's auc: 0.52801
[70]	valid_0's auc: 0.528015
Early stopping, best iteration is:
[20]	valid_0's auc: 0.52821
best score: 0.528210374046
best iteration: 20
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
[10]	valid_0's auc: 0.52312
[20]	valid_0's auc: 0.523664
[30]	valid_0's auc: 0.52367
[40]	valid_0's auc: 0.523714
[50]	valid_0's auc: 0.523739
[60]	valid_0's auc: 0.523726
[70]	valid_0's auc: 0.523763
[80]	valid_0's auc: 0.523761
[90]	valid_0's auc: 0.524053
[100]	valid_0's auc: 0.524121
[110]	valid_0's auc: 0.524122
[120]	valid_0's auc: 0.52417
[130]	valid_0's auc: 0.524168
[140]	valid_0's auc: 0.524167
[150]	valid_0's auc: 0.524183
[160]	valid_0's auc: 0.524183
[170]	valid_0's auc: 0.524197
[180]	valid_0's auc: 0.524197
[190]	valid_0's auc: 0.524199
[200]	valid_0's auc: 0.524199
[210]	valid_0's auc: 0.524199
[220]	valid_0's auc: 0.524199
[230]	valid_0's auc: 0.524198
[240]	valid_0's auc: 0.524198
[250]	valid_0's auc: 0.524207
[260]	valid_0's auc: 0.524204
[270]	valid_0's auc: 0.524204
[280]	valid_0's auc: 0.524204
[290]	valid_0's auc: 0.524204
Early stopping, best iteration is:
[247]	valid_0's auc: 0.524207
best score: 0.524206569763
best iteration: 247
complete on: ISC_top3_in_song

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
[10]	valid_0's auc: 0.522
[20]	valid_0's auc: 0.525844
[30]	valid_0's auc: 0.527613
[40]	valid_0's auc: 0.530785
[50]	valid_0's auc: 0.532568
[60]	valid_0's auc: 0.534437
[70]	valid_0's auc: 0.535433
[80]	valid_0's auc: 0.536287
[90]	valid_0's auc: 0.536809
[100]	valid_0's auc: 0.537457
[110]	valid_0's auc: 0.537709
[120]	valid_0's auc: 0.537929
[130]	valid_0's auc: 0.538182
[140]	valid_0's auc: 0.538562
[150]	valid_0's auc: 0.538778
[160]	valid_0's auc: 0.538964
[170]	valid_0's auc: 0.539036
[180]	valid_0's auc: 0.539177
[190]	valid_0's auc: 0.539264
[200]	valid_0's auc: 0.539331
[210]	valid_0's auc: 0.539411
[220]	valid_0's auc: 0.539461
[230]	valid_0's auc: 0.539606
[240]	valid_0's auc: 0.539658
[250]	valid_0's auc: 0.53967
[260]	valid_0's auc: 0.539733
[270]	valid_0's auc: 0.53978
[280]	valid_0's auc: 0.539804
[290]	valid_0's auc: 0.539755
[300]	valid_0's auc: 0.539838
[310]	valid_0's auc: 0.539848
[320]	valid_0's auc: 0.539885
[330]	valid_0's auc: 0.539925
[340]	valid_0's auc: 0.539928
[350]	valid_0's auc: 0.539934
[360]	valid_0's auc: 0.539951
[370]	valid_0's auc: 0.539963
[380]	valid_0's auc: 0.539978
[390]	valid_0's auc: 0.539968
[400]	valid_0's auc: 0.540003
[410]	valid_0's auc: 0.540004
[420]	valid_0's auc: 0.540005
[430]	valid_0's auc: 0.540003
[440]	valid_0's auc: 0.540009
[450]	valid_0's auc: 0.540011
[460]	valid_0's auc: 0.540003
[470]	valid_0's auc: 0.540033
[480]	valid_0's auc: 0.540039
[490]	valid_0's auc: 0.540064
[500]	valid_0's auc: 0.540052
[510]	valid_0's auc: 0.540063
[520]	valid_0's auc: 0.540046
[530]	valid_0's auc: 0.540038
[540]	valid_0's auc: 0.539984
Early stopping, best iteration is:
[494]	valid_0's auc: 0.540069
best score: 0.54006905136
best iteration: 494
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
[10]	valid_0's auc: 0.529819
[20]	valid_0's auc: 0.531424
[30]	valid_0's auc: 0.53284
[40]	valid_0's auc: 0.533953
[50]	valid_0's auc: 0.535038
[60]	valid_0's auc: 0.535238
[70]	valid_0's auc: 0.535453
[80]	valid_0's auc: 0.535712
[90]	valid_0's auc: 0.536052
[100]	valid_0's auc: 0.536307
[110]	valid_0's auc: 0.536011
[120]	valid_0's auc: 0.536088
[130]	valid_0's auc: 0.536112
[140]	valid_0's auc: 0.536132
[150]	valid_0's auc: 0.536369
[160]	valid_0's auc: 0.536622
[170]	valid_0's auc: 0.536726
[180]	valid_0's auc: 0.536978
[190]	valid_0's auc: 0.536997
[200]	valid_0's auc: 0.537042
[210]	valid_0's auc: 0.537157
[220]	valid_0's auc: 0.53717
[230]	valid_0's auc: 0.537218
[240]	valid_0's auc: 0.537208
[250]	valid_0's auc: 0.537178
[260]	valid_0's auc: 0.53719
[270]	valid_0's auc: 0.537141
[280]	valid_0's auc: 0.53714
Early stopping, best iteration is:
[232]	valid_0's auc: 0.537225
best score: 0.537224682105
best iteration: 232
complete on: ISC_composer

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
[10]	valid_0's auc: 0.529424
[20]	valid_0's auc: 0.529841
[30]	valid_0's auc: 0.529702
[40]	valid_0's auc: 0.52995
[50]	valid_0's auc: 0.530062
[60]	valid_0's auc: 0.530463
[70]	valid_0's auc: 0.530493
[80]	valid_0's auc: 0.530618
[90]	valid_0's auc: 0.530812
[100]	valid_0's auc: 0.531186
[110]	valid_0's auc: 0.531197
[120]	valid_0's auc: 0.531216
[130]	valid_0's auc: 0.531267
[140]	valid_0's auc: 0.531357
[150]	valid_0's auc: 0.531406
[160]	valid_0's auc: 0.531404
[170]	valid_0's auc: 0.531391
[180]	valid_0's auc: 0.531481
[190]	valid_0's auc: 0.531393
[200]	valid_0's auc: 0.531487
[210]	valid_0's auc: 0.531527
[220]	valid_0's auc: 0.531558
[230]	valid_0's auc: 0.5316
[240]	valid_0's auc: 0.531664
[250]	valid_0's auc: 0.531666
[260]	valid_0's auc: 0.531651
[270]	valid_0's auc: 0.531636
[280]	valid_0's auc: 0.531663
[290]	valid_0's auc: 0.531656
[300]	valid_0's auc: 0.531652
[310]	valid_0's auc: 0.531655
[320]	valid_0's auc: 0.531654
Early stopping, best iteration is:
[275]	valid_0's auc: 0.531697
best score: 0.5316974855
best iteration: 275
complete on: ISCZ_lyricist

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
[10]	valid_0's auc: 0.533085
[20]	valid_0's auc: 0.537321
[30]	valid_0's auc: 0.538517
[40]	valid_0's auc: 0.539039
[50]	valid_0's auc: 0.540348
[60]	valid_0's auc: 0.540427
[70]	valid_0's auc: 0.540904
[80]	valid_0's auc: 0.540936
[90]	valid_0's auc: 0.541718
[100]	valid_0's auc: 0.541688
[110]	valid_0's auc: 0.542005
[120]	valid_0's auc: 0.542109
[130]	valid_0's auc: 0.542414
[140]	valid_0's auc: 0.542473
[150]	valid_0's auc: 0.542403
[160]	valid_0's auc: 0.542425
[170]	valid_0's auc: 0.54246
[180]	valid_0's auc: 0.542419
[190]	valid_0's auc: 0.542471
Early stopping, best iteration is:
[140]	valid_0's auc: 0.542473
best score: 0.542473145051
best iteration: 140
complete on: ISCZ_rc

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
[10]	valid_0's auc: 0.512375
[20]	valid_0's auc: 0.514026
[30]	valid_0's auc: 0.515353
[40]	valid_0's auc: 0.515469
[50]	valid_0's auc: 0.516591
[60]	valid_0's auc: 0.517772
[70]	valid_0's auc: 0.517546
[80]	valid_0's auc: 0.51751
[90]	valid_0's auc: 0.517771
[100]	valid_0's auc: 0.518039
[110]	valid_0's auc: 0.51799
[120]	valid_0's auc: 0.518127
[130]	valid_0's auc: 0.51806
[140]	valid_0's auc: 0.518119
[150]	valid_0's auc: 0.517999
[160]	valid_0's auc: 0.517689
Early stopping, best iteration is:
[119]	valid_0's auc: 0.518214
best score: 0.518213616979
best iteration: 119
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
[90]	valid_0's auc: 0.538055
Early stopping, best iteration is:
[42]	valid_0's auc: 0.538077
best score: 0.538077328194
best iteration: 42
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
[170]	valid_0's auc: 0.538055
Early stopping, best iteration is:
[121]	valid_0's auc: 0.538055
best score: 0.538055387574
best iteration: 121
complete on: ISCZ_song_year

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
[10]	valid_0's auc: 0.511402
[20]	valid_0's auc: 0.518822
[30]	valid_0's auc: 0.522305
[40]	valid_0's auc: 0.524391
[50]	valid_0's auc: 0.525375
[60]	valid_0's auc: 0.52605
[70]	valid_0's auc: 0.52648
[80]	valid_0's auc: 0.527058
[90]	valid_0's auc: 0.526996
[100]	valid_0's auc: 0.527075
[110]	valid_0's auc: 0.527303
[120]	valid_0's auc: 0.527189
[130]	valid_0's auc: 0.527379
[140]	valid_0's auc: 0.527401
[150]	valid_0's auc: 0.527118
[160]	valid_0's auc: 0.527262
[170]	valid_0's auc: 0.527035
[180]	valid_0's auc: 0.527181
[190]	valid_0's auc: 0.527142
Early stopping, best iteration is:
[140]	valid_0's auc: 0.527401
best score: 0.527400763551
best iteration: 140
complete on: song_length_log10

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
[10]	valid_0's auc: 0.525727
[20]	valid_0's auc: 0.526842
[30]	valid_0's auc: 0.526984
[40]	valid_0's auc: 0.527183
[50]	valid_0's auc: 0.527157
[60]	valid_0's auc: 0.527204
[70]	valid_0's auc: 0.527234
[80]	valid_0's auc: 0.527238
[90]	valid_0's auc: 0.527248
[100]	valid_0's auc: 0.527268
[110]	valid_0's auc: 0.527554
[120]	valid_0's auc: 0.527543
[130]	valid_0's auc: 0.527541
[140]	valid_0's auc: 0.527537
[150]	valid_0's auc: 0.527539
Early stopping, best iteration is:
[101]	valid_0's auc: 0.527563
best score: 0.527563227322
best iteration: 101
complete on: ISCZ_genre_ids_log10

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
[10]	valid_0's auc: 0.522442
[20]	valid_0's auc: 0.526356
[30]	valid_0's auc: 0.527836
[40]	valid_0's auc: 0.531607
[50]	valid_0's auc: 0.533912
[60]	valid_0's auc: 0.534644
[70]	valid_0's auc: 0.53554
[80]	valid_0's auc: 0.536364
[90]	valid_0's auc: 0.536881
[100]	valid_0's auc: 0.537343
[110]	valid_0's auc: 0.537691
[120]	valid_0's auc: 0.538011
[130]	valid_0's auc: 0.538105
[140]	valid_0's auc: 0.538516
[150]	valid_0's auc: 0.53857
[160]	valid_0's auc: 0.538822
[170]	valid_0's auc: 0.538886
[180]	valid_0's auc: 0.539002
[190]	valid_0's auc: 0.53912
[200]	valid_0's auc: 0.539248
[210]	valid_0's auc: 0.539301
[220]	valid_0's auc: 0.53935
[230]	valid_0's auc: 0.53941
[240]	valid_0's auc: 0.539511
[250]	valid_0's auc: 0.539551
[260]	valid_0's auc: 0.539604
[270]	valid_0's auc: 0.539664
[280]	valid_0's auc: 0.53976
[290]	valid_0's auc: 0.539852
[300]	valid_0's auc: 0.539811
[310]	valid_0's auc: 0.539824
[320]	valid_0's auc: 0.539851
[330]	valid_0's auc: 0.539887
[340]	valid_0's auc: 0.539969
[350]	valid_0's auc: 0.540028
[360]	valid_0's auc: 0.539976
[370]	valid_0's auc: 0.53999
[380]	valid_0's auc: 0.540017
[390]	valid_0's auc: 0.54003
[400]	valid_0's auc: 0.540012
[410]	valid_0's auc: 0.540014
[420]	valid_0's auc: 0.540125
[430]	valid_0's auc: 0.540105
[440]	valid_0's auc: 0.540118
[450]	valid_0's auc: 0.540126
[460]	valid_0's auc: 0.540107
Early stopping, best iteration is:
[419]	valid_0's auc: 0.540129
best score: 0.540129338866
best iteration: 419
complete on: ISC_artist_name_log10

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
[20]	valid_0's auc: 0.531063
[30]	valid_0's auc: 0.53323
[40]	valid_0's auc: 0.534336
[50]	valid_0's auc: 0.534783
[60]	valid_0's auc: 0.534683
[70]	valid_0's auc: 0.535039
[80]	valid_0's auc: 0.535393
[90]	valid_0's auc: 0.535707
[100]	valid_0's auc: 0.535741
[110]	valid_0's auc: 0.535763
[120]	valid_0's auc: 0.535965
[130]	valid_0's auc: 0.536267
[140]	valid_0's auc: 0.536327
[150]	valid_0's auc: 0.536691
[160]	valid_0's auc: 0.536799
[170]	valid_0's auc: 0.536844
[180]	valid_0's auc: 0.536841
[190]	valid_0's auc: 0.536845
[200]	valid_0's auc: 0.537023
[210]	valid_0's auc: 0.537088
[220]	valid_0's auc: 0.537063
[230]	valid_0's auc: 0.537064
[240]	valid_0's auc: 0.537123
[250]	valid_0's auc: 0.537151
[260]	valid_0's auc: 0.537148
[270]	valid_0's auc: 0.53718
[280]	valid_0's auc: 0.537277
[290]	valid_0's auc: 0.537277
[300]	valid_0's auc: 0.537279
[310]	valid_0's auc: 0.537357
[320]	valid_0's auc: 0.537355
[330]	valid_0's auc: 0.537303
[340]	valid_0's auc: 0.537305
[350]	valid_0's auc: 0.537307
[360]	valid_0's auc: 0.537305
Early stopping, best iteration is:
[312]	valid_0's auc: 0.537358
best score: 0.537357554901
best iteration: 312
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
[10]	valid_0's auc: 0.529393
[20]	valid_0's auc: 0.530103
[30]	valid_0's auc: 0.529834
[40]	valid_0's auc: 0.529558
[50]	valid_0's auc: 0.529933
[60]	valid_0's auc: 0.530356
[70]	valid_0's auc: 0.530445
[80]	valid_0's auc: 0.530642
[90]	valid_0's auc: 0.53078
[100]	valid_0's auc: 0.531066
[110]	valid_0's auc: 0.531067
[120]	valid_0's auc: 0.531065
[130]	valid_0's auc: 0.531055
[140]	valid_0's auc: 0.53113
[150]	valid_0's auc: 0.531186
[160]	valid_0's auc: 0.531215
[170]	valid_0's auc: 0.531227
[180]	valid_0's auc: 0.53123
[190]	valid_0's auc: 0.531292
[200]	valid_0's auc: 0.53134
[210]	valid_0's auc: 0.531452
[220]	valid_0's auc: 0.531471
[230]	valid_0's auc: 0.531515
[240]	valid_0's auc: 0.531521
[250]	valid_0's auc: 0.531523
[260]	valid_0's auc: 0.531583
[270]	valid_0's auc: 0.531583
[280]	valid_0's auc: 0.531617
[290]	valid_0's auc: 0.531618
[300]	valid_0's auc: 0.531616
[310]	valid_0's auc: 0.531605
[320]	valid_0's auc: 0.531598
[330]	valid_0's auc: 0.531598
[340]	valid_0's auc: 0.531564
Early stopping, best iteration is:
[295]	valid_0's auc: 0.531618
best score: 0.531618369673
best iteration: 295
complete on: ISC_lyricist_log10

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
[10]	valid_0's auc: 0.506651
[20]	valid_0's auc: 0.507881
[30]	valid_0's auc: 0.508389
[40]	valid_0's auc: 0.508405
[50]	valid_0's auc: 0.508924
[60]	valid_0's auc: 0.509148
[70]	valid_0's auc: 0.509332
[80]	valid_0's auc: 0.50946
[90]	valid_0's auc: 0.509552
[100]	valid_0's auc: 0.50967
[110]	valid_0's auc: 0.509683
[120]	valid_0's auc: 0.509713
[130]	valid_0's auc: 0.509839
[140]	valid_0's auc: 0.509899
[150]	valid_0's auc: 0.509976
[160]	valid_0's auc: 0.510007
[170]	valid_0's auc: 0.510037
[180]	valid_0's auc: 0.510045
[190]	valid_0's auc: 0.510014
[200]	valid_0's auc: 0.510029
[210]	valid_0's auc: 0.510035
[220]	valid_0's auc: 0.51006
[230]	valid_0's auc: 0.510111
[240]	valid_0's auc: 0.510117
[250]	valid_0's auc: 0.510146
[260]	valid_0's auc: 0.510178
[270]	valid_0's auc: 0.51017
[280]	valid_0's auc: 0.510185
[290]	valid_0's auc: 0.510174
[300]	valid_0's auc: 0.510175
[310]	valid_0's auc: 0.510181
[320]	valid_0's auc: 0.510111
[330]	valid_0's auc: 0.510127
Early stopping, best iteration is:
[282]	valid_0's auc: 0.510186
best score: 0.510185901677
best iteration: 282
complete on: ISC_name_log10

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
[10]	valid_0's auc: 0.506223
[20]	valid_0's auc: 0.507916
[30]	valid_0's auc: 0.508204
[40]	valid_0's auc: 0.508719
[50]	valid_0's auc: 0.508927
[60]	valid_0's auc: 0.509313
[70]	valid_0's auc: 0.509545
[80]	valid_0's auc: 0.509595
[90]	valid_0's auc: 0.509686
[100]	valid_0's auc: 0.509664
[110]	valid_0's auc: 0.509616
[120]	valid_0's auc: 0.509671
[130]	valid_0's auc: 0.509796
[140]	valid_0's auc: 0.509846
[150]	valid_0's auc: 0.509876
[160]	valid_0's auc: 0.509885
[170]	valid_0's auc: 0.509981
[180]	valid_0's auc: 0.509994
[190]	valid_0's auc: 0.510052
[200]	valid_0's auc: 0.510069
[210]	valid_0's auc: 0.510049
[220]	valid_0's auc: 0.510102
[230]	valid_0's auc: 0.510128
[240]	valid_0's auc: 0.510142
[250]	valid_0's auc: 0.510152
[260]	valid_0's auc: 0.510161
[270]	valid_0's auc: 0.510128
[280]	valid_0's auc: 0.510143
[290]	valid_0's auc: 0.510149
[300]	valid_0's auc: 0.510155
[310]	valid_0's auc: 0.510164
[320]	valid_0's auc: 0.510157
[330]	valid_0's auc: 0.510167
[340]	valid_0's auc: 0.510178
[350]	valid_0's auc: 0.51018
[360]	valid_0's auc: 0.510188
[370]	valid_0's auc: 0.510187
[380]	valid_0's auc: 0.510188
[390]	valid_0's auc: 0.510187
[400]	valid_0's auc: 0.510183
[410]	valid_0's auc: 0.51012
Early stopping, best iteration is:
[363]	valid_0's auc: 0.510188
best score: 0.510188193885
best iteration: 363
complete on: ISCZ_name_ln

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
[30]	valid_0's auc: 0.525778
[40]	valid_0's auc: 0.525916
[50]	valid_0's auc: 0.526066
[60]	valid_0's auc: 0.526083
[70]	valid_0's auc: 0.526111
[80]	valid_0's auc: 0.526146
[90]	valid_0's auc: 0.526175
[100]	valid_0's auc: 0.526127
[110]	valid_0's auc: 0.526138
[120]	valid_0's auc: 0.52604
[130]	valid_0's auc: 0.52604
Early stopping, best iteration is:
[85]	valid_0's auc: 0.526185
best score: 0.526184562009
best iteration: 85
complete on: ISC_song_country_ln

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
[30]	valid_0's auc: 0.525815
[40]	valid_0's auc: 0.525888
[50]	valid_0's auc: 0.525884
[60]	valid_0's auc: 0.526077
[70]	valid_0's auc: 0.5261
[80]	valid_0's auc: 0.526092
[90]	valid_0's auc: 0.526155
[100]	valid_0's auc: 0.526109
[110]	valid_0's auc: 0.526143
[120]	valid_0's auc: 0.526042
[130]	valid_0's auc: 0.526042
Early stopping, best iteration is:
[84]	valid_0's auc: 0.526156
best score: 0.526155821048
best iteration: 84
complete on: ISCZ_song_country_log10

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
[10]	valid_0's auc: 0.533156
[20]	valid_0's auc: 0.536288
[30]	valid_0's auc: 0.538086
[40]	valid_0's auc: 0.53851
[50]	valid_0's auc: 0.539173
[60]	valid_0's auc: 0.539503
[70]	valid_0's auc: 0.539831
[80]	valid_0's auc: 0.539548
[90]	valid_0's auc: 0.539723
[100]	valid_0's auc: 0.540318
[110]	valid_0's auc: 0.540404
[120]	valid_0's auc: 0.540277
[130]	valid_0's auc: 0.540587
[140]	valid_0's auc: 0.541109
[150]	valid_0's auc: 0.541165
[160]	valid_0's auc: 0.541346
[170]	valid_0's auc: 0.541381
[180]	valid_0's auc: 0.541925
[190]	valid_0's auc: 0.541986
[200]	valid_0's auc: 0.541992
[210]	valid_0's auc: 0.542029
[220]	valid_0's auc: 0.542025
[230]	valid_0's auc: 0.541986
[240]	valid_0's auc: 0.541904
[250]	valid_0's auc: 0.541924
Early stopping, best iteration is:
[206]	valid_0's auc: 0.542072
best score: 0.542071898261
best iteration: 206
complete on: ISC_rc_ln

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
[10]	valid_0's auc: 0.510038
[20]	valid_0's auc: 0.513081
[30]	valid_0's auc: 0.514664
[40]	valid_0's auc: 0.515044
[50]	valid_0's auc: 0.516103
[60]	valid_0's auc: 0.516559
[70]	valid_0's auc: 0.516778
[80]	valid_0's auc: 0.517105
[90]	valid_0's auc: 0.517176
[100]	valid_0's auc: 0.517342
[110]	valid_0's auc: 0.517356
[120]	valid_0's auc: 0.517791
[130]	valid_0's auc: 0.517292
[140]	valid_0's auc: 0.517218
[150]	valid_0's auc: 0.517396
[160]	valid_0's auc: 0.517364
[170]	valid_0's auc: 0.517339
Early stopping, best iteration is:
[121]	valid_0's auc: 0.517808
best score: 0.517807645686
best iteration: 121
complete on: ISC_isrc_rest_log10

            IMC_age_guess_range_log10:  0.508907354054
                age_guess_range_log10:  0.508907354054
                       ISC_name_log10:  0.510185901677
                         ISCZ_name_ln:  0.510188193885
             IMC_bd_fixed_range_log10:  0.512342947511
                 bd_fixed_range_log10:  0.512342947511
                   IMC_bd_range_log10:  0.512363873294
                       bd_range_log10:  0.512363873294
            IMC_membership_days_log10:  0.51305298913
           IMC_expiration_month_log10:  0.513488378472
               expiration_month_log10:  0.51351211197
                  IMC_age_guess_log10:  0.513638951741
                      age_guess_log10:  0.513651814109
                   IMC_bd_fixed_log10:  0.514373696215
                         IMC_bd_log10:  0.514826834671
                       bd_fixed_log10:  0.515803351591
                             bd_log10:  0.516325224149
                  ISC_isrc_rest_log10:  0.517807645686
                      membership_days:  0.518020750052
                       ISCZ_isrc_rest:  0.518213616979
                     ISC_top3_in_song:  0.524206569763
                         ISC_language:  0.524767621195
              ISCZ_song_country_log10:  0.526155821048
                  ISC_song_country_ln:  0.526184562009
                     ISC_top1_in_song:  0.526218173781
                        ISC_genre_ids:  0.52736185413
                    song_length_log10:  0.527400763551
                 ISCZ_genre_ids_log10:  0.527563227322
                     ISC_top2_in_song:  0.528210374046
                   ISC_lyricist_log10:  0.531618369673
                        ISCZ_lyricist:  0.5316974855
                         ISC_composer:  0.537224682105
                  ISCZ_composer_log10:  0.537357554901
                       ISCZ_song_year:  0.538055387574
                            song_year:  0.538060446124
                        ISC_song_year:  0.538077328194
                     ISCZ_artist_name:  0.54006905136
                ISC_artist_name_log10:  0.540129338866
                            ISC_rc_ln:  0.542071898261
                              ISCZ_rc:  0.542473145051

[timer]: complete in 62m 26s

Process finished with exit code 0
'''
