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
target                               uint8
bd_log10                           float64
membership_days_log10              float64
membership_days_range_log10        float64
registration_year_log10            float64
registration_month_log10           float64
registration_date_log10            float64
expiration_year_log10              float64
expiration_month_log10             float64
expiration_date_log10              float64
IMC_city_log10                     float64
IMC_gender_log10                   float64
IMCZ_gender_log10                  float64
IMC_registered_via_log10           float64
IMC_registration_year_log10        float64
IMC_registration_month_log10       float64
IMC_registration_date_log10        float64
IMC_expiration_year_log10          float64
IMC_expiration_month_log10         float64
IMC_expiration_date_log10          float64
bd_fixed_log10                     float64
age_guess_log10                    float64
bd_range_log10                     float64
age_guess_range_log10              float64
bd_fixed_range_log10               float64
IMC_bd_log10                       float64
IMC_bd_fixed_log10                 float64
IMC_age_guess_log10                float64
IMC_bd_range_log10                 float64
IMC_bd_fixed_range_log10           float64
IMC_age_guess_range_log10          float64
IMC_membership_days_log10          float64
IMC_membership_days_range_log10    float64
dtype: object
number of columns: 33
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
[10]	valid_0's auc: 0.515996
[20]	valid_0's auc: 0.516234
[30]	valid_0's auc: 0.516224
[40]	valid_0's auc: 0.516375
[50]	valid_0's auc: 0.51635
[60]	valid_0's auc: 0.516215
[70]	valid_0's auc: 0.516142
[80]	valid_0's auc: 0.516146
[90]	valid_0's auc: 0.516154
Early stopping, best iteration is:
[40]	valid_0's auc: 0.516375
best score: 0.516375479796
best iteration: 40
complete on: bd_log10

working on: membership_days_log10


After selection:
target                     uint8
membership_days_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512257
[20]	valid_0's auc: 0.514051
[30]	valid_0's auc: 0.514952
[40]	valid_0's auc: 0.515606
[50]	valid_0's auc: 0.516066
[60]	valid_0's auc: 0.516569
[70]	valid_0's auc: 0.516711
[80]	valid_0's auc: 0.516992
[90]	valid_0's auc: 0.516888
[100]	valid_0's auc: 0.517024
[110]	valid_0's auc: 0.517024
[120]	valid_0's auc: 0.517079
[130]	valid_0's auc: 0.517178
[140]	valid_0's auc: 0.517198
[150]	valid_0's auc: 0.517305
[160]	valid_0's auc: 0.517463
[170]	valid_0's auc: 0.517484
[180]	valid_0's auc: 0.517447
[190]	valid_0's auc: 0.517487
[200]	valid_0's auc: 0.517556
[210]	valid_0's auc: 0.517599
[220]	valid_0's auc: 0.517711
[230]	valid_0's auc: 0.517721
[240]	valid_0's auc: 0.517785
[250]	valid_0's auc: 0.517781
[260]	valid_0's auc: 0.517796
[270]	valid_0's auc: 0.517878
[280]	valid_0's auc: 0.517889
[290]	valid_0's auc: 0.517918
[300]	valid_0's auc: 0.517916
[310]	valid_0's auc: 0.517917
[320]	valid_0's auc: 0.517922
[330]	valid_0's auc: 0.517952
[340]	valid_0's auc: 0.517961
[350]	valid_0's auc: 0.517956
[360]	valid_0's auc: 0.517982
[370]	valid_0's auc: 0.518
[380]	valid_0's auc: 0.518005
[390]	valid_0's auc: 0.517967
[400]	valid_0's auc: 0.517966
[410]	valid_0's auc: 0.517975
[420]	valid_0's auc: 0.517992
Early stopping, best iteration is:
[376]	valid_0's auc: 0.518022
best score: 0.5180218469
best iteration: 376
complete on: membership_days_log10

working on: membership_days_range_log10


After selection:
target                           uint8
membership_days_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.49795
[20]	valid_0's auc: 0.49795
[30]	valid_0's auc: 0.49795
[40]	valid_0's auc: 0.49795
[50]	valid_0's auc: 0.49795
Early stopping, best iteration is:
[1]	valid_0's auc: 0.49795
best score: 0.497950102921
best iteration: 1
complete on: membership_days_range_log10

working on: registration_year_log10


After selection:
target                       uint8
registration_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.503056
[20]	valid_0's auc: 0.50299
[30]	valid_0's auc: 0.50299
[40]	valid_0's auc: 0.50299
[50]	valid_0's auc: 0.50299
Early stopping, best iteration is:
[1]	valid_0's auc: 0.503559
best score: 0.503559229892
best iteration: 1
complete on: registration_year_log10

working on: registration_month_log10


After selection:
target                        uint8
registration_month_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505293
[20]	valid_0's auc: 0.504862
[30]	valid_0's auc: 0.505149
[40]	valid_0's auc: 0.505149
[50]	valid_0's auc: 0.505328
[60]	valid_0's auc: 0.505328
[70]	valid_0's auc: 0.505328
[80]	valid_0's auc: 0.505328
[90]	valid_0's auc: 0.505328
Early stopping, best iteration is:
[45]	valid_0's auc: 0.505328
best score: 0.505328020014
best iteration: 45
complete on: registration_month_log10

working on: registration_date_log10


After selection:
target                       uint8
registration_date_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.504573
[20]	valid_0's auc: 0.505173
[30]	valid_0's auc: 0.504848
[40]	valid_0's auc: 0.50504
[50]	valid_0's auc: 0.505018
[60]	valid_0's auc: 0.505027
[70]	valid_0's auc: 0.50496
Early stopping, best iteration is:
[22]	valid_0's auc: 0.505176
best score: 0.50517598203
best iteration: 22
complete on: registration_date_log10

working on: expiration_year_log10


After selection:
target                     uint8
expiration_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506714
[20]	valid_0's auc: 0.506714
[30]	valid_0's auc: 0.506714
[40]	valid_0's auc: 0.506714
[50]	valid_0's auc: 0.506714
Early stopping, best iteration is:
[1]	valid_0's auc: 0.506714
best score: 0.50671362948
best iteration: 1
complete on: expiration_year_log10

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

working on: expiration_date_log10


After selection:
target                     uint8
expiration_date_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.507095
[20]	valid_0's auc: 0.507201
[30]	valid_0's auc: 0.507153
[40]	valid_0's auc: 0.507227
[50]	valid_0's auc: 0.507114
Early stopping, best iteration is:
[1]	valid_0's auc: 0.507429
best score: 0.507429045473
best iteration: 1
complete on: expiration_date_log10

working on: IMC_city_log10


After selection:
target              uint8
IMC_city_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505394
[20]	valid_0's auc: 0.50566
[30]	valid_0's auc: 0.505651
[40]	valid_0's auc: 0.505868
[50]	valid_0's auc: 0.505915
[60]	valid_0's auc: 0.505602
[70]	valid_0's auc: 0.505612
[80]	valid_0's auc: 0.505612
[90]	valid_0's auc: 0.505612
Early stopping, best iteration is:
[49]	valid_0's auc: 0.505915
best score: 0.505914654431
best iteration: 49
complete on: IMC_city_log10

working on: IMC_gender_log10


After selection:
target                uint8
IMC_gender_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.504738
[20]	valid_0's auc: 0.504738
[30]	valid_0's auc: 0.504738
[40]	valid_0's auc: 0.504738
[50]	valid_0's auc: 0.504738
Early stopping, best iteration is:
[1]	valid_0's auc: 0.504738
best score: 0.504738069576
best iteration: 1
complete on: IMC_gender_log10

working on: IMCZ_gender_log10


After selection:
target                 uint8
IMCZ_gender_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.504738
[20]	valid_0's auc: 0.504738
[30]	valid_0's auc: 0.504738
[40]	valid_0's auc: 0.504738
[50]	valid_0's auc: 0.504738
Early stopping, best iteration is:
[1]	valid_0's auc: 0.504738
best score: 0.504738069576
best iteration: 1
complete on: IMCZ_gender_log10

working on: IMC_registered_via_log10


After selection:
target                        uint8
IMC_registered_via_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.502389
[20]	valid_0's auc: 0.502389
[30]	valid_0's auc: 0.502389
[40]	valid_0's auc: 0.502389
[50]	valid_0's auc: 0.502389
Early stopping, best iteration is:
[1]	valid_0's auc: 0.502389
best score: 0.502388935196
best iteration: 1
complete on: IMC_registered_via_log10

working on: IMC_registration_year_log10


After selection:
target                           uint8
IMC_registration_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.50299
[20]	valid_0's auc: 0.50299
[30]	valid_0's auc: 0.50299
[40]	valid_0's auc: 0.50299
[50]	valid_0's auc: 0.50299
Early stopping, best iteration is:
[1]	valid_0's auc: 0.503047
best score: 0.503046936669
best iteration: 1
complete on: IMC_registration_year_log10

working on: IMC_registration_month_log10


After selection:
target                            uint8
IMC_registration_month_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505278
[20]	valid_0's auc: 0.505328
[30]	valid_0's auc: 0.505328
[40]	valid_0's auc: 0.505328
[50]	valid_0's auc: 0.505328
[60]	valid_0's auc: 0.505328
Early stopping, best iteration is:
[12]	valid_0's auc: 0.505328
best score: 0.505328020014
best iteration: 12
complete on: IMC_registration_month_log10

working on: IMC_registration_date_log10


After selection:
target                           uint8
IMC_registration_date_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505
[20]	valid_0's auc: 0.504839
[30]	valid_0's auc: 0.504789
[40]	valid_0's auc: 0.504832
[50]	valid_0's auc: 0.504779
[60]	valid_0's auc: 0.504787
Early stopping, best iteration is:
[10]	valid_0's auc: 0.505
best score: 0.505000331532
best iteration: 10
complete on: IMC_registration_date_log10

working on: IMC_expiration_year_log10


After selection:
target                         uint8
IMC_expiration_year_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506714
[20]	valid_0's auc: 0.506714
[30]	valid_0's auc: 0.506714
[40]	valid_0's auc: 0.506714
[50]	valid_0's auc: 0.506714
[60]	valid_0's auc: 0.506713
[70]	valid_0's auc: 0.506713
[80]	valid_0's auc: 0.506713
[90]	valid_0's auc: 0.506713
[100]	valid_0's auc: 0.506713
Early stopping, best iteration is:
[54]	valid_0's auc: 0.506714
best score: 0.506714004239
best iteration: 54
complete on: IMC_expiration_year_log10

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

working on: IMC_expiration_date_log10


After selection:
target                         uint8
IMC_expiration_date_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506243
[20]	valid_0's auc: 0.506513
[30]	valid_0's auc: 0.506778
[40]	valid_0's auc: 0.506779
[50]	valid_0's auc: 0.506622
[60]	valid_0's auc: 0.506805
[70]	valid_0's auc: 0.506815
Early stopping, best iteration is:
[28]	valid_0's auc: 0.506822
best score: 0.506822161527
best iteration: 28
complete on: IMC_expiration_date_log10

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
[10]	valid_0's auc: 0.516468
[20]	valid_0's auc: 0.516553
[30]	valid_0's auc: 0.516584
[40]	valid_0's auc: 0.516509
[50]	valid_0's auc: 0.516488
[60]	valid_0's auc: 0.516486
[70]	valid_0's auc: 0.516576
[80]	valid_0's auc: 0.516519
[90]	valid_0's auc: 0.516512
[100]	valid_0's auc: 0.516517
[110]	valid_0's auc: 0.516517
Early stopping, best iteration is:
[66]	valid_0's auc: 0.516603
best score: 0.51660269976
best iteration: 66
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
[10]	valid_0's auc: 0.51157
[20]	valid_0's auc: 0.51157
[30]	valid_0's auc: 0.51157
[40]	valid_0's auc: 0.51157
[50]	valid_0's auc: 0.51157
Early stopping, best iteration is:
[1]	valid_0's auc: 0.51157
best score: 0.511569676428
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
[10]	valid_0's auc: 0.5145
[20]	valid_0's auc: 0.516102
[30]	valid_0's auc: 0.516665
[40]	valid_0's auc: 0.516653
[50]	valid_0's auc: 0.516476
[60]	valid_0's auc: 0.516455
[70]	valid_0's auc: 0.516539
[80]	valid_0's auc: 0.516516
Early stopping, best iteration is:
[32]	valid_0's auc: 0.516782
best score: 0.51678232349
best iteration: 32
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
[10]	valid_0's auc: 0.51157
[20]	valid_0's auc: 0.51157
[30]	valid_0's auc: 0.51157
[40]	valid_0's auc: 0.51157
[50]	valid_0's auc: 0.51157
Early stopping, best iteration is:
[1]	valid_0's auc: 0.51157
best score: 0.511569676428
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

working on: IMC_membership_days_range_log10


After selection:
target                               uint8
IMC_membership_days_range_log10    float64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.49795
[20]	valid_0's auc: 0.49795
[30]	valid_0's auc: 0.49795
[40]	valid_0's auc: 0.49795
[50]	valid_0's auc: 0.49795
Early stopping, best iteration is:
[1]	valid_0's auc: 0.49795
best score: 0.497950102921
best iteration: 1
complete on: IMC_membership_days_range_log10

          membership_days_range_log10:  0.497950102921
      IMC_membership_days_range_log10:  0.497950102921
             IMC_registered_via_log10:  0.502388935196
          IMC_registration_year_log10:  0.503046936669
              registration_year_log10:  0.503559229892
                     IMC_gender_log10:  0.504738069576
                    IMCZ_gender_log10:  0.504738069576
          IMC_registration_date_log10:  0.505000331532
              registration_date_log10:  0.50517598203
             registration_month_log10:  0.505328020014
         IMC_registration_month_log10:  0.505328020014
                       IMC_city_log10:  0.505914654431
                expiration_year_log10:  0.50671362948
            IMC_expiration_year_log10:  0.506714004239
            IMC_expiration_date_log10:  0.506822161527
                expiration_date_log10:  0.507429045473
            IMC_age_guess_range_log10:  0.511569676428
                age_guess_range_log10:  0.511569676428
                 bd_fixed_range_log10:  0.512342947511
             IMC_bd_fixed_range_log10:  0.512342947511
                   IMC_bd_range_log10:  0.512363873294
                       bd_range_log10:  0.512363873294
            IMC_membership_days_log10:  0.51305298913
           IMC_expiration_month_log10:  0.513488378472
               expiration_month_log10:  0.51351211197
                   IMC_bd_fixed_log10:  0.514373696215
                         IMC_bd_log10:  0.514826834671
                       bd_fixed_log10:  0.515803351591
                             bd_log10:  0.516375479796
                      age_guess_log10:  0.51660269976
                  IMC_age_guess_log10:  0.51678232349
                membership_days_log10:  0.5180218469

[timer]: complete in 38m 22s

Process finished with exit code 0
'''