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
working_on = [
    'city',
    'registered_via',
    'membership_days_range',
    'sex',
    'sex_guess',
    'length_range',
    'length_bin_range',
    'length_chunk_range',
    'song_year_bin_range',
    'song_year_chunk_range',
]

# working_on = [
#     'age',
#     'membership_days',
#     'membership_days_range',
#     'registration_year',
#     'registration_month',
#     'registration_date',
#     'expiration_year',
#     'expiration_month',
#     'expiration_date',
#     'song_length',
# ]
for w in working_on:
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)

    del dt
    print('working on:', w)
    df = df[['msno', 'song_id', w, 'target']]
    df[w] = df[w].astype('category')
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


'''/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/training_V1201.py"
working on: city
Train test and validation sets


After selection:
msno       category
song_id    category
city       category
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
[10]	valid_0's auc: 0.60301
[20]	valid_0's auc: 0.609574
[30]	valid_0's auc: 0.613521
[40]	valid_0's auc: 0.616091
[50]	valid_0's auc: 0.617375
[60]	valid_0's auc: 0.617652
[70]	valid_0's auc: 0.618124
[80]	valid_0's auc: 0.618245
[90]	valid_0's auc: 0.618831
[100]	valid_0's auc: 0.618905
[110]	valid_0's auc: 0.618802
[120]	valid_0's auc: 0.618944
[130]	valid_0's auc: 0.618938
[140]	valid_0's auc: 0.618846
[150]	valid_0's auc: 0.618841
[160]	valid_0's auc: 0.618732
[170]	valid_0's auc: 0.618742
[180]	valid_0's auc: 0.618736
[190]	valid_0's auc: 0.618702
[200]	valid_0's auc: 0.618625
[210]	valid_0's auc: 0.61858
[220]	valid_0's auc: 0.618512
[230]	valid_0's auc: 0.618512
[240]	valid_0's auc: 0.61845
[250]	valid_0's auc: 0.618395
[260]	valid_0's auc: 0.618388
[270]	valid_0's auc: 0.618303
[280]	valid_0's auc: 0.618307
[290]	valid_0's auc: 0.61831
[300]	valid_0's auc: 0.618281
[310]	valid_0's auc: 0.618237
[320]	valid_0's auc: 0.618222
Early stopping, best iteration is:
[124]	valid_0's auc: 0.618979
complete on: city
working on: registered_via
Train test and validation sets


After selection:
msno              category
song_id           category
registered_via    category
target               uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.603374
[20]	valid_0's auc: 0.609568
[30]	valid_0's auc: 0.614654
[40]	valid_0's auc: 0.617419
[50]	valid_0's auc: 0.619009
[60]	valid_0's auc: 0.619596
[70]	valid_0's auc: 0.619998
[80]	valid_0's auc: 0.619996
[90]	valid_0's auc: 0.620473
[100]	valid_0's auc: 0.620403
[110]	valid_0's auc: 0.620323
[120]	valid_0's auc: 0.620373
[130]	valid_0's auc: 0.620252
[140]	valid_0's auc: 0.620151
[150]	valid_0's auc: 0.620125
[160]	valid_0's auc: 0.620022
[170]	valid_0's auc: 0.619982
[180]	valid_0's auc: 0.619993
[190]	valid_0's auc: 0.619949
[200]	valid_0's auc: 0.61988
[210]	valid_0's auc: 0.619812
[220]	valid_0's auc: 0.619794
[230]	valid_0's auc: 0.619779
[240]	valid_0's auc: 0.61968
[250]	valid_0's auc: 0.619625
[260]	valid_0's auc: 0.61963
[270]	valid_0's auc: 0.619604
[280]	valid_0's auc: 0.619554
[290]	valid_0's auc: 0.619571
Early stopping, best iteration is:
[90]	valid_0's auc: 0.620473
complete on: registered_via
working on: membership_days_range
Train test and validation sets


After selection:
msno                     category
song_id                  category
membership_days_range    category
target                      uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.600319
[20]	valid_0's auc: 0.610793
[30]	valid_0's auc: 0.6149
[40]	valid_0's auc: 0.616988
[50]	valid_0's auc: 0.618132
[60]	valid_0's auc: 0.618477
[70]	valid_0's auc: 0.61869
[80]	valid_0's auc: 0.619074
[90]	valid_0's auc: 0.619514
[100]	valid_0's auc: 0.619518
[110]	valid_0's auc: 0.619441
[120]	valid_0's auc: 0.619546
[130]	valid_0's auc: 0.619561
[140]	valid_0's auc: 0.619455
[150]	valid_0's auc: 0.619504
[160]	valid_0's auc: 0.619423
[170]	valid_0's auc: 0.619353
[180]	valid_0's auc: 0.619287
[190]	valid_0's auc: 0.619206
[200]	valid_0's auc: 0.619119
[210]	valid_0's auc: 0.619055
[220]	valid_0's auc: 0.619077
[230]	valid_0's auc: 0.619063
[240]	valid_0's auc: 0.619045
[250]	valid_0's auc: 0.619004
[260]	valid_0's auc: 0.618944
[270]	valid_0's auc: 0.618909
[280]	valid_0's auc: 0.618857
[290]	valid_0's auc: 0.618792
[300]	valid_0's auc: 0.618737
[310]	valid_0's auc: 0.618719
[320]	valid_0's auc: 0.618724
Early stopping, best iteration is:
[124]	valid_0's auc: 0.619597
complete on: membership_days_range
working on: sex
Train test and validation sets


After selection:
msno       category
song_id    category
sex        category
target        uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.604529
[20]	valid_0's auc: 0.610296
[30]	valid_0's auc: 0.614771
[40]	valid_0's auc: 0.617304
[50]	valid_0's auc: 0.618345
[60]	valid_0's auc: 0.618822
[70]	valid_0's auc: 0.619176
[80]	valid_0's auc: 0.619381
[90]	valid_0's auc: 0.619722
[100]	valid_0's auc: 0.619747
[110]	valid_0's auc: 0.61973
[120]	valid_0's auc: 0.619758
[130]	valid_0's auc: 0.619774
[140]	valid_0's auc: 0.619646
[150]	valid_0's auc: 0.619671
[160]	valid_0's auc: 0.619554
[170]	valid_0's auc: 0.619531
[180]	valid_0's auc: 0.61948
[190]	valid_0's auc: 0.619423
[200]	valid_0's auc: 0.619371
[210]	valid_0's auc: 0.619339
[220]	valid_0's auc: 0.619299
[230]	valid_0's auc: 0.619268
[240]	valid_0's auc: 0.619203
[250]	valid_0's auc: 0.619143
[260]	valid_0's auc: 0.619119
[270]	valid_0's auc: 0.619154
[280]	valid_0's auc: 0.619109
[290]	valid_0's auc: 0.619067
Early stopping, best iteration is:
[96]	valid_0's auc: 0.619824
complete on: sex
working on: sex_guess
Train test and validation sets


After selection:
msno         category
song_id      category
sex_guess    category
target          uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.600649
[20]	valid_0's auc: 0.608973
[30]	valid_0's auc: 0.613482
[40]	valid_0's auc: 0.616271
[50]	valid_0's auc: 0.617457
[60]	valid_0's auc: 0.618502
[70]	valid_0's auc: 0.618945
[80]	valid_0's auc: 0.619112
[90]	valid_0's auc: 0.619303
[100]	valid_0's auc: 0.619187
[110]	valid_0's auc: 0.619048
[120]	valid_0's auc: 0.619146
[130]	valid_0's auc: 0.61915
[140]	valid_0's auc: 0.619109
[150]	valid_0's auc: 0.61907
[160]	valid_0's auc: 0.619002
[170]	valid_0's auc: 0.618921
[180]	valid_0's auc: 0.618898
[190]	valid_0's auc: 0.618883
[200]	valid_0's auc: 0.618863
[210]	valid_0's auc: 0.618846
[220]	valid_0's auc: 0.618827
[230]	valid_0's auc: 0.618803
[240]	valid_0's auc: 0.618691
[250]	valid_0's auc: 0.618688
[260]	valid_0's auc: 0.618629
[270]	valid_0's auc: 0.618608
[280]	valid_0's auc: 0.618557
[290]	valid_0's auc: 0.618544
Early stopping, best iteration is:
[93]	valid_0's auc: 0.619345
complete on: sex_guess
working on: length_range
Train test and validation sets


After selection:
msno            category
song_id         category
length_range    category
target             uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.598496
[20]	valid_0's auc: 0.608119
[30]	valid_0's auc: 0.612804
[40]	valid_0's auc: 0.615973
[50]	valid_0's auc: 0.6174
[60]	valid_0's auc: 0.618563
[70]	valid_0's auc: 0.619004
[80]	valid_0's auc: 0.619194
[90]	valid_0's auc: 0.61946
[100]	valid_0's auc: 0.619556
[110]	valid_0's auc: 0.619465
[120]	valid_0's auc: 0.619443
[130]	valid_0's auc: 0.619388
[140]	valid_0's auc: 0.619379
[150]	valid_0's auc: 0.619368
[160]	valid_0's auc: 0.619267
[170]	valid_0's auc: 0.619127
[180]	valid_0's auc: 0.619081
[190]	valid_0's auc: 0.619041
[200]	valid_0's auc: 0.61898
[210]	valid_0's auc: 0.618966
[220]	valid_0's auc: 0.618914
[230]	valid_0's auc: 0.618859
[240]	valid_0's auc: 0.618834
[250]	valid_0's auc: 0.618865
[260]	valid_0's auc: 0.61882
[270]	valid_0's auc: 0.618786
[280]	valid_0's auc: 0.618751
[290]	valid_0's auc: 0.618693
Early stopping, best iteration is:
[98]	valid_0's auc: 0.619601
complete on: length_range
working on: length_bin_range
Train test and validation sets


After selection:
msno                category
song_id             category
length_bin_range    category
target                 uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601062
[20]	valid_0's auc: 0.60944
[30]	valid_0's auc: 0.614176
[40]	valid_0's auc: 0.616725
[50]	valid_0's auc: 0.61778
[60]	valid_0's auc: 0.618523
[70]	valid_0's auc: 0.618829
[80]	valid_0's auc: 0.61904
[90]	valid_0's auc: 0.619252
[100]	valid_0's auc: 0.619221
[110]	valid_0's auc: 0.619139
[120]	valid_0's auc: 0.619224
[130]	valid_0's auc: 0.619129
[140]	valid_0's auc: 0.619022
[150]	valid_0's auc: 0.618986
[160]	valid_0's auc: 0.618913
[170]	valid_0's auc: 0.618821
[180]	valid_0's auc: 0.61885
[190]	valid_0's auc: 0.618849
[200]	valid_0's auc: 0.618773
[210]	valid_0's auc: 0.618692
[220]	valid_0's auc: 0.618709
[230]	valid_0's auc: 0.618608
[240]	valid_0's auc: 0.618554
[250]	valid_0's auc: 0.618526
[260]	valid_0's auc: 0.61848
[270]	valid_0's auc: 0.618417
[280]	valid_0's auc: 0.618379
[290]	valid_0's auc: 0.618364
Early stopping, best iteration is:
[96]	valid_0's auc: 0.61928
complete on: length_bin_range
working on: length_chunk_range
Train test and validation sets


After selection:
msno                  category
song_id               category
length_chunk_range    category
target                   uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.599934
[20]	valid_0's auc: 0.608417
[30]	valid_0's auc: 0.613271
[40]	valid_0's auc: 0.616392
[50]	valid_0's auc: 0.618114
[60]	valid_0's auc: 0.619054
[70]	valid_0's auc: 0.619603
[80]	valid_0's auc: 0.61983
[90]	valid_0's auc: 0.620067
[100]	valid_0's auc: 0.620139
[110]	valid_0's auc: 0.620105
[120]	valid_0's auc: 0.620095
[130]	valid_0's auc: 0.620079
[140]	valid_0's auc: 0.62006
[150]	valid_0's auc: 0.619987
[160]	valid_0's auc: 0.619912
[170]	valid_0's auc: 0.619865
[180]	valid_0's auc: 0.619813
[190]	valid_0's auc: 0.619738
[200]	valid_0's auc: 0.619678
[210]	valid_0's auc: 0.619676
[220]	valid_0's auc: 0.619656
[230]	valid_0's auc: 0.619616
[240]	valid_0's auc: 0.619589
[250]	valid_0's auc: 0.619554
[260]	valid_0's auc: 0.619516
[270]	valid_0's auc: 0.619485
[280]	valid_0's auc: 0.619502
[290]	valid_0's auc: 0.619533
Early stopping, best iteration is:
[96]	valid_0's auc: 0.620195
complete on: length_chunk_range
working on: song_year_bin_range
Train test and validation sets


After selection:
msno                   category
song_id                category
song_year_bin_range    category
target                    uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601528
[20]	valid_0's auc: 0.609651
[30]	valid_0's auc: 0.614407
[40]	valid_0's auc: 0.616741
[50]	valid_0's auc: 0.618188
[60]	valid_0's auc: 0.619028
[70]	valid_0's auc: 0.619324
[80]	valid_0's auc: 0.619618
[90]	valid_0's auc: 0.619846
[100]	valid_0's auc: 0.619843
[110]	valid_0's auc: 0.619844
[120]	valid_0's auc: 0.619845
[130]	valid_0's auc: 0.619772
[140]	valid_0's auc: 0.619681
[150]	valid_0's auc: 0.619664
[160]	valid_0's auc: 0.619619
[170]	valid_0's auc: 0.619523
[180]	valid_0's auc: 0.619561
[190]	valid_0's auc: 0.619494
[200]	valid_0's auc: 0.619433
[210]	valid_0's auc: 0.619411
[220]	valid_0's auc: 0.619417
[230]	valid_0's auc: 0.619376
[240]	valid_0's auc: 0.619291
[250]	valid_0's auc: 0.619303
[260]	valid_0's auc: 0.619212
[270]	valid_0's auc: 0.619146
[280]	valid_0's auc: 0.619054
[290]	valid_0's auc: 0.61901
[300]	valid_0's auc: 0.619036
Early stopping, best iteration is:
[105]	valid_0's auc: 0.619927
complete on: song_year_bin_range
working on: song_year_chunk_range
Train test and validation sets


After selection:
msno                     category
song_id                  category
song_year_chunk_range    category
target                      uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.606833
[20]	valid_0's auc: 0.612657
[30]	valid_0's auc: 0.616647
[40]	valid_0's auc: 0.619368
[50]	valid_0's auc: 0.620898
[60]	valid_0's auc: 0.622019
[70]	valid_0's auc: 0.623007
[80]	valid_0's auc: 0.623127
[90]	valid_0's auc: 0.62338
[100]	valid_0's auc: 0.623403
[110]	valid_0's auc: 0.623511
[120]	valid_0's auc: 0.623623
[130]	valid_0's auc: 0.623602
[140]	valid_0's auc: 0.623399
[150]	valid_0's auc: 0.623387
[160]	valid_0's auc: 0.623337
[170]	valid_0's auc: 0.623354
[180]	valid_0's auc: 0.62331
[190]	valid_0's auc: 0.623303
[200]	valid_0's auc: 0.62326
[210]	valid_0's auc: 0.623169
[220]	valid_0's auc: 0.623187
[230]	valid_0's auc: 0.623137
[240]	valid_0's auc: 0.623103
[250]	valid_0's auc: 0.623124
[260]	valid_0's auc: 0.623075
[270]	valid_0's auc: 0.623071
[280]	valid_0's auc: 0.623087
[290]	valid_0's auc: 0.623098
[300]	valid_0's auc: 0.623072
[310]	valid_0's auc: 0.623074
[320]	valid_0's auc: 0.623032
Early stopping, best iteration is:
[122]	valid_0's auc: 0.62365
complete on: song_year_chunk_range

[timer]: complete in 140m 53s

Process finished with exit code 0
'''