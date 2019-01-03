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
# dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
# df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
#
# del dt
# print('What we got:')
# print(df.dtypes)
# print('number of columns:', len(df.columns))
# print(type(df.head()))
# df = df.drop(['song_count', 'liked_song_count',
#               'disliked_song_count', 'artist_count',
#               'liked_artist_count', 'disliked_artist_count'], axis=1)
# df = df[['mn', 'sn', 'target']]
# df = df[['msno', 'song_id', 'language', 'target']]
# df['language'] = df['language'].astype('category')
# working_on = ['source_system_tab',
#               'source_screen_name',
#               'source_type',
#               'genre_ids',
#               'composer',
#               'lyricist',
#               'rc',
#               ]
'''
msno                        object
song_id                     object
source_system_tab           object
source_screen_name          object
source_type                 object
target                       uint8
city                         uint8
registered_via               uint8
mn                           int64
age                           int8
age_range                     int8
membership_days              int64
membership_days_range         int8
registration_year            int64
registration_month           int64
registration_date            int64
expiration_year              int64
expiration_month             int64
expiration_date              int64
sex                           int8
sex_guess                     int8
song_length                  int64
genre_ids                   object
artist_name                 object
composer                    object
lyricist                    object
language                      int8
sn                           int64
lyricists_count               int8
composer_count                int8
genre_ids_count               int8
length_range                 int64
length_bin_range             int64
length_chunk_range           int64
song_year                    int64
song_year_bin_range          int64
song_year_chunk_range        int64
song_country                object
rc                          object
artist_composer               int8
artist_composer_lyricist      int8
song_count                   int64
liked_song_count             int64
disliked_song_count          int64
artist_count                 int64
liked_artist_count           int64
disliked_artist_count        int64
'''
# working_on = [
#     'city',
#     'registered_via',
#     'membership_days_range',
#     'sex',
#     'sex_guess',
#     'length_range',
#     'length_bin_range',
#     'length_chunk_range',
#     'song_year_bin_range',
#     'song_year_chunk_range',
# ]

working_on = [
    'age',
    'membership_days',
    'membership_days_range',
    'registration_year',
    'registration_month',
    'registration_date',
    'expiration_year',
    'expiration_month',
    'expiration_date',
    'song_length',
]
for w in working_on:
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)

    del dt
    print('working on:', w)
    df = df[['msno', 'song_id', w, 'target']]
    # df[w] = df[w].astype('category')
    # df = df[['city', 'age', 'target']]
    print("Train test and validation sets")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')
            # test[col] = test[col].astype('category')

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
    # print(train_set.head(100))
    # print(len(train_set))
    # print(len(val_set))
    del df
    train_set = train_set.sample(frac=1)
    X_tr = train_set.drop(['target'], axis=1)
    Y_tr = train_set['target'].values

    X_val = val_set.drop(['target'], axis=1)
    Y_val = val_set['target'].values

    del train_set, val_set
    # X_test = test.drop(['id'], axis=1)
    # ids = test['id'].values
    # X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train,
    #                                             train_size=0.000001,
    #                                             shuffle=True,
    #                                             random_state=555,
    #                                             )
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
    # del X_train, Y_train

    train_set = lgb.Dataset(X_tr, Y_tr)
    val_set = lgb.Dataset(X_val, Y_val)
    del X_tr, Y_tr, X_val, Y_val

    # train_set = lgb.Dataset(X_train, Y_train,
    #                         categorical_feature=[0, 1],
    #                         )
    print('Training...')
    params = {'objective': 'binary',
              'metric': 'auc',
              # 'metric': 'binary_logloss',
              'boosting': 'gbdt',
              'learning_rate': 0.1,
              # 'verbosity': -1,
              'verbose': -1,
              'num_leaves': 100,

              'bagging_fraction': 0.8,
              'bagging_freq': 2,
              'bagging_seed': 1,
              'feature_fraction': 0.8,
              'feature_fraction_seed': 1,
              'max_bin': 63,
              'max_depth': -1,
              # 'min_data': 500,
              # 'min_hessian': 0.05,
              # 'num_rounds': 500,
              # "min_data_in_leaf": 1,
              # 'min_data': 1,
              # 'min_data_in_bin': 1,
              # 'lambda_l2': 0.5,
              # 'device': 'gpu',
              # 'gpu_platform_id': 0,
              # 'gpu_device_id': 0,
              # 'sparse_threshold': 1.0,
              # 'categorical_feature': (0,1,2,3),
              }
    model = lgb.train(params,
                      train_set,
                      num_boost_round=50000,
                      early_stopping_rounds=200,
                      valid_sets=val_set,
                      verbose_eval=10,
                      )
    # model_name = 'model_V1001'
    # pickle.dump(model, open(save_dir+model_name+'.save', "wb"))
    # print('model saved as: ', save_dir, model_name)
    del train_set, val_set
    print('complete on:', w)
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1201.py
working on: age
Train test and validation sets


After selection:
msno       category
song_id    category
age            int8
target        uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.600163
[20]	valid_0's auc: 0.60773
[30]	valid_0's auc: 0.612312
[40]	valid_0's auc: 0.615242
[50]	valid_0's auc: 0.617441
[60]	valid_0's auc: 0.6183
[70]	valid_0's auc: 0.618788
[80]	valid_0's auc: 0.619118
[90]	valid_0's auc: 0.619571
[100]	valid_0's auc: 0.61968
[110]	valid_0's auc: 0.619687
[120]	valid_0's auc: 0.619788
[130]	valid_0's auc: 0.619783
[140]	valid_0's auc: 0.619738
[150]	valid_0's auc: 0.619704
[160]	valid_0's auc: 0.619703
[170]	valid_0's auc: 0.619614
[180]	valid_0's auc: 0.61962
[190]	valid_0's auc: 0.619554
[200]	valid_0's auc: 0.619545
[210]	valid_0's auc: 0.619493
[220]	valid_0's auc: 0.619465
[230]	valid_0's auc: 0.619378
[240]	valid_0's auc: 0.619349
[250]	valid_0's auc: 0.619316
[260]	valid_0's auc: 0.619285
[270]	valid_0's auc: 0.619248
[280]	valid_0's auc: 0.619202
[290]	valid_0's auc: 0.619167
[300]	valid_0's auc: 0.619161
[310]	valid_0's auc: 0.619074
[320]	valid_0's auc: 0.619076
[330]	valid_0's auc: 0.619076
Early stopping, best iteration is:
[136]	valid_0's auc: 0.619793
complete on: age
working on: membership_days
Train test and validation sets


After selection:
msno               category
song_id            category
membership_days       int64
target                uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.60463
[20]	valid_0's auc: 0.611447
[30]	valid_0's auc: 0.614306
[40]	valid_0's auc: 0.616746
[50]	valid_0's auc: 0.618039
[60]	valid_0's auc: 0.618568
[70]	valid_0's auc: 0.61917
[80]	valid_0's auc: 0.619528
[90]	valid_0's auc: 0.620007
[100]	valid_0's auc: 0.62018
[110]	valid_0's auc: 0.620188
[120]	valid_0's auc: 0.620118
[130]	valid_0's auc: 0.620069
[140]	valid_0's auc: 0.620013
[150]	valid_0's auc: 0.620038
[160]	valid_0's auc: 0.619988
[170]	valid_0's auc: 0.619967
[180]	valid_0's auc: 0.619924
[190]	valid_0's auc: 0.619896
[200]	valid_0's auc: 0.619857
[210]	valid_0's auc: 0.619727
[220]	valid_0's auc: 0.619704
[230]	valid_0's auc: 0.619678
[240]	valid_0's auc: 0.619679
[250]	valid_0's auc: 0.619622
[260]	valid_0's auc: 0.619601
[270]	valid_0's auc: 0.619609
[280]	valid_0's auc: 0.619597
[290]	valid_0's auc: 0.619565
[300]	valid_0's auc: 0.619546
Early stopping, best iteration is:
[104]	valid_0's auc: 0.620279
complete on: membership_days
working on: membership_days_range
Train test and validation sets


After selection:
msno                     category
song_id                  category
membership_days_range        int8
target                      uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.602212
[20]	valid_0's auc: 0.609363
[30]	valid_0's auc: 0.614045
[40]	valid_0's auc: 0.616609
[50]	valid_0's auc: 0.618024
[60]	valid_0's auc: 0.618528
[70]	valid_0's auc: 0.618739
[80]	valid_0's auc: 0.619013
[90]	valid_0's auc: 0.619529
[100]	valid_0's auc: 0.619522
[110]	valid_0's auc: 0.619561
[120]	valid_0's auc: 0.619564
[130]	valid_0's auc: 0.619549
[140]	valid_0's auc: 0.61947
[150]	valid_0's auc: 0.619445
[160]	valid_0's auc: 0.619356
[170]	valid_0's auc: 0.61928
[180]	valid_0's auc: 0.619235
[190]	valid_0's auc: 0.619141
[200]	valid_0's auc: 0.619067
[210]	valid_0's auc: 0.618917
[220]	valid_0's auc: 0.61886
[230]	valid_0's auc: 0.618814
[240]	valid_0's auc: 0.61878
[250]	valid_0's auc: 0.618714
[260]	valid_0's auc: 0.618693
[270]	valid_0's auc: 0.618675
[280]	valid_0's auc: 0.61864
[290]	valid_0's auc: 0.618607
Early stopping, best iteration is:
[93]	valid_0's auc: 0.619617
complete on: membership_days_range
working on: registration_year
Train test and validation sets


After selection:
msno                 category
song_id              category
registration_year       int64
target                  uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.599448
[20]	valid_0's auc: 0.607328
[30]	valid_0's auc: 0.611708
[40]	valid_0's auc: 0.614381
[50]	valid_0's auc: 0.615842
[60]	valid_0's auc: 0.616567
[70]	valid_0's auc: 0.617169
[80]	valid_0's auc: 0.617483
[90]	valid_0's auc: 0.618018
[100]	valid_0's auc: 0.618087
[110]	valid_0's auc: 0.618054
[120]	valid_0's auc: 0.618184
[130]	valid_0's auc: 0.618182
[140]	valid_0's auc: 0.618099
[150]	valid_0's auc: 0.618137
[160]	valid_0's auc: 0.618083
[170]	valid_0's auc: 0.618015
[180]	valid_0's auc: 0.617998
[190]	valid_0's auc: 0.617912
[200]	valid_0's auc: 0.617717
[210]	valid_0's auc: 0.617647
[220]	valid_0's auc: 0.617622
[230]	valid_0's auc: 0.617528
[240]	valid_0's auc: 0.617535
[250]	valid_0's auc: 0.617601
[260]	valid_0's auc: 0.617543
[270]	valid_0's auc: 0.617521
[280]	valid_0's auc: 0.61748
[290]	valid_0's auc: 0.617451
[300]	valid_0's auc: 0.617401
[310]	valid_0's auc: 0.617405
[320]	valid_0's auc: 0.617416
Early stopping, best iteration is:
[124]	valid_0's auc: 0.618247
complete on: registration_year
working on: registration_month
Train test and validation sets


After selection:
msno                  category
song_id               category
registration_month       int64
target                   uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601451
[20]	valid_0's auc: 0.608507
[30]	valid_0's auc: 0.613259
[40]	valid_0's auc: 0.615925
[50]	valid_0's auc: 0.6174
[60]	valid_0's auc: 0.618068
[70]	valid_0's auc: 0.618449
[80]	valid_0's auc: 0.618701
[90]	valid_0's auc: 0.619149
[100]	valid_0's auc: 0.619122
[110]	valid_0's auc: 0.619158
[120]	valid_0's auc: 0.619246
[130]	valid_0's auc: 0.619266
[140]	valid_0's auc: 0.619129
[150]	valid_0's auc: 0.619096
[160]	valid_0's auc: 0.619092
[170]	valid_0's auc: 0.619054
[180]	valid_0's auc: 0.619021
[190]	valid_0's auc: 0.618986
[200]	valid_0's auc: 0.618913
[210]	valid_0's auc: 0.618926
[220]	valid_0's auc: 0.618857
[230]	valid_0's auc: 0.618828
[240]	valid_0's auc: 0.618852
[250]	valid_0's auc: 0.618823
[260]	valid_0's auc: 0.618829
[270]	valid_0's auc: 0.618783
[280]	valid_0's auc: 0.618774
[290]	valid_0's auc: 0.61868
[300]	valid_0's auc: 0.618673
[310]	valid_0's auc: 0.618749
[320]	valid_0's auc: 0.61868
Early stopping, best iteration is:
[125]	valid_0's auc: 0.619299
complete on: registration_month
working on: registration_date
Train test and validation sets


After selection:
msno                 category
song_id              category
registration_date       int64
target                  uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.600205
[20]	valid_0's auc: 0.607828
[30]	valid_0's auc: 0.612871
[40]	valid_0's auc: 0.615564
[50]	valid_0's auc: 0.6172
[60]	valid_0's auc: 0.617893
[70]	valid_0's auc: 0.618101
[80]	valid_0's auc: 0.618203
[90]	valid_0's auc: 0.61841
[100]	valid_0's auc: 0.618466
[110]	valid_0's auc: 0.618349
[120]	valid_0's auc: 0.618392
[130]	valid_0's auc: 0.618315
[140]	valid_0's auc: 0.618266
[150]	valid_0's auc: 0.618218
[160]	valid_0's auc: 0.618166
[170]	valid_0's auc: 0.618123
[180]	valid_0's auc: 0.618093
[190]	valid_0's auc: 0.618041
[200]	valid_0's auc: 0.618017
[210]	valid_0's auc: 0.617908
[220]	valid_0's auc: 0.617858
[230]	valid_0's auc: 0.617746
[240]	valid_0's auc: 0.617668
[250]	valid_0's auc: 0.617653
[260]	valid_0's auc: 0.617693
[270]	valid_0's auc: 0.617674
[280]	valid_0's auc: 0.617636
[290]	valid_0's auc: 0.617685
Early stopping, best iteration is:
[97]	valid_0's auc: 0.618541
complete on: registration_date
working on: expiration_year
Train test and validation sets


After selection:
msno               category
song_id            category
expiration_year       int64
target                uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.605261
[20]	valid_0's auc: 0.612544
[30]	valid_0's auc: 0.616764
[40]	valid_0's auc: 0.619271
[50]	valid_0's auc: 0.620429
[60]	valid_0's auc: 0.620832
[70]	valid_0's auc: 0.621113
[80]	valid_0's auc: 0.621228
[90]	valid_0's auc: 0.621544
[100]	valid_0's auc: 0.621584
[110]	valid_0's auc: 0.621569
[120]	valid_0's auc: 0.621583
[130]	valid_0's auc: 0.621482
[140]	valid_0's auc: 0.621396
[150]	valid_0's auc: 0.621409
[160]	valid_0's auc: 0.621363
[170]	valid_0's auc: 0.621288
[180]	valid_0's auc: 0.621278
[190]	valid_0's auc: 0.621208
[200]	valid_0's auc: 0.621205
[210]	valid_0's auc: 0.621215
[220]	valid_0's auc: 0.621143
[230]	valid_0's auc: 0.621148
[240]	valid_0's auc: 0.621093
[250]	valid_0's auc: 0.62108
[260]	valid_0's auc: 0.62105
[270]	valid_0's auc: 0.621039
[280]	valid_0's auc: 0.621024
[290]	valid_0's auc: 0.621017
Early stopping, best iteration is:
[97]	valid_0's auc: 0.621659
complete on: expiration_year
working on: expiration_month
Train test and validation sets


After selection:
msno                category
song_id             category
expiration_month       int64
target                 uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.603121
[20]	valid_0's auc: 0.608419
[30]	valid_0's auc: 0.615103
[40]	valid_0's auc: 0.617625
[50]	valid_0's auc: 0.61899
[60]	valid_0's auc: 0.619379
[70]	valid_0's auc: 0.619502
[80]	valid_0's auc: 0.619741
[90]	valid_0's auc: 0.620271
[100]	valid_0's auc: 0.62041
[110]	valid_0's auc: 0.620357
[120]	valid_0's auc: 0.620483
[130]	valid_0's auc: 0.620485
[140]	valid_0's auc: 0.620462
[150]	valid_0's auc: 0.620509
[160]	valid_0's auc: 0.620528
[170]	valid_0's auc: 0.620551
[180]	valid_0's auc: 0.620547
[190]	valid_0's auc: 0.62047
[200]	valid_0's auc: 0.620408
[210]	valid_0's auc: 0.620427
[220]	valid_0's auc: 0.620324
[230]	valid_0's auc: 0.620299
[240]	valid_0's auc: 0.620306
[250]	valid_0's auc: 0.620336
[260]	valid_0's auc: 0.620294
[270]	valid_0's auc: 0.620266
[280]	valid_0's auc: 0.620295
[290]	valid_0's auc: 0.620261
[300]	valid_0's auc: 0.620204
[310]	valid_0's auc: 0.62016
[320]	valid_0's auc: 0.620144
[330]	valid_0's auc: 0.620035
[340]	valid_0's auc: 0.620025
[350]	valid_0's auc: 0.620018
[360]	valid_0's auc: 0.620018
[370]	valid_0's auc: 0.620001
Early stopping, best iteration is:
[175]	valid_0's auc: 0.620572
complete on: expiration_month
working on: expiration_date
Train test and validation sets


After selection:
msno               category
song_id            category
expiration_date       int64
target                uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601417
[20]	valid_0's auc: 0.609133
[30]	valid_0's auc: 0.613892
[40]	valid_0's auc: 0.61678
[50]	valid_0's auc: 0.618134
[60]	valid_0's auc: 0.619077
[70]	valid_0's auc: 0.619312
[80]	valid_0's auc: 0.619461
[90]	valid_0's auc: 0.619686
[100]	valid_0's auc: 0.619727
[110]	valid_0's auc: 0.619609
[120]	valid_0's auc: 0.619702
[130]	valid_0's auc: 0.619564
[140]	valid_0's auc: 0.619484
[150]	valid_0's auc: 0.619497
[160]	valid_0's auc: 0.619417
[170]	valid_0's auc: 0.619433
[180]	valid_0's auc: 0.619431
[190]	valid_0's auc: 0.619378
[200]	valid_0's auc: 0.619279
[210]	valid_0's auc: 0.619252
[220]	valid_0's auc: 0.619171
[230]	valid_0's auc: 0.61916
[240]	valid_0's auc: 0.619103
[250]	valid_0's auc: 0.619092
[260]	valid_0's auc: 0.619031
[270]	valid_0's auc: 0.618969
[280]	valid_0's auc: 0.61891
[290]	valid_0's auc: 0.618864
[300]	valid_0's auc: 0.618852
Early stopping, best iteration is:
[100]	valid_0's auc: 0.619727
complete on: expiration_date
working on: song_length
Train test and validation sets


After selection:
msno           category
song_id        category
song_length       int64
target            uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.60374
[20]	valid_0's auc: 0.609845
[30]	valid_0's auc: 0.61443
[40]	valid_0's auc: 0.617157
[50]	valid_0's auc: 0.6187
[60]	valid_0's auc: 0.619331
[70]	valid_0's auc: 0.61992
[80]	valid_0's auc: 0.62004
[90]	valid_0's auc: 0.620314
[100]	valid_0's auc: 0.620455
[110]	valid_0's auc: 0.620385
[120]	valid_0's auc: 0.620309
[130]	valid_0's auc: 0.620287
[140]	valid_0's auc: 0.620132
[150]	valid_0's auc: 0.62012
[160]	valid_0's auc: 0.620041
[170]	valid_0's auc: 0.62004
[180]	valid_0's auc: 0.620012
[190]	valid_0's auc: 0.619939
[200]	valid_0's auc: 0.619882
[210]	valid_0's auc: 0.619886
[220]	valid_0's auc: 0.61986
[230]	valid_0's auc: 0.619847
[240]	valid_0's auc: 0.619793
[250]	valid_0's auc: 0.619784
[260]	valid_0's auc: 0.619735
[270]	valid_0's auc: 0.619713
[280]	valid_0's auc: 0.619698
[290]	valid_0's auc: 0.619642
[300]	valid_0's auc: 0.619609
Early stopping, best iteration is:
[104]	valid_0's auc: 0.620462
complete on: song_length

[timer]: complete in 141m 21s

Process finished with exit code 0
'''