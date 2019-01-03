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

# working_on = [
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'source_system_tab_guess',
#     'source_screen_name_guess',
#     'source_type_guess',
# ]
fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         ]
# working_on = ['language',
#               'artist_name']
# if True:
#     w = 'nothing'
for w in df.columns:
    if w in fixed:

        # print('yoyo:', w)
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        # toto = fixed
        toto.append(w)
        df = df[toto]
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

                  # 'bagging_fraction': 0.8,
                  # 'bagging_freq': 2,
                  # 'bagging_seed': 1,
                  # 'feature_fraction': 0.8,
                  # 'feature_fraction_seed': 1,
                  'max_bin': 255,
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
                          num_boost_round=500000,
                          early_stopping_rounds=50,
                          valid_sets=val_set,
                          verbose_eval=10,
                          )
        # model_name = 'model_V1001'
        # pickle.dump(model, open(save_dir+model_name+'.save', "wb"))
        # print('model saved as: ', save_dir, model_name)
        del train_set, val_set
        print('complete on:', w)
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
        del dt


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1203.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
registered_via        category
genre_ids               object
artist_name             object
language              category
length_bin_range      category
length_chunk_range    category
dtype: object
number of columns: 12
working on: registered_via
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
registered_via        category
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
[10]	valid_0's auc: 0.644196
[20]	valid_0's auc: 0.650625
[30]	valid_0's auc: 0.656604
[40]	valid_0's auc: 0.661211
[50]	valid_0's auc: 0.663885
[60]	valid_0's auc: 0.665841
[70]	valid_0's auc: 0.667044
[80]	valid_0's auc: 0.66786
[90]	valid_0's auc: 0.668879
[100]	valid_0's auc: 0.66943
[110]	valid_0's auc: 0.669694
[120]	valid_0's auc: 0.670133
[130]	valid_0's auc: 0.670301
[140]	valid_0's auc: 0.670725
[150]	valid_0's auc: 0.670844
[160]	valid_0's auc: 0.670972
[170]	valid_0's auc: 0.671088
[180]	valid_0's auc: 0.671226
[190]	valid_0's auc: 0.671254
[200]	valid_0's auc: 0.671405
[210]	valid_0's auc: 0.671435
[220]	valid_0's auc: 0.671524
[230]	valid_0's auc: 0.671585
[240]	valid_0's auc: 0.671567
[250]	valid_0's auc: 0.671599
[260]	valid_0's auc: 0.671688
[270]	valid_0's auc: 0.671705
[280]	valid_0's auc: 0.671813
[290]	valid_0's auc: 0.671809
[300]	valid_0's auc: 0.671822
[310]	valid_0's auc: 0.671802
[320]	valid_0's auc: 0.671834
[330]	valid_0's auc: 0.671863
[340]	valid_0's auc: 0.671853
[350]	valid_0's auc: 0.671829
[360]	valid_0's auc: 0.671825
[370]	valid_0's auc: 0.671825
[380]	valid_0's auc: 0.671851
Early stopping, best iteration is:
[334]	valid_0's auc: 0.671884
complete on: registered_via
working on: genre_ids
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
genre_ids             category
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.644303
[20]	valid_0's auc: 0.649955
[30]	valid_0's auc: 0.655557
[40]	valid_0's auc: 0.660229
[50]	valid_0's auc: 0.663515
[60]	valid_0's auc: 0.665649
[70]	valid_0's auc: 0.667062
[80]	valid_0's auc: 0.668262
[90]	valid_0's auc: 0.668992
[100]	valid_0's auc: 0.669632
[110]	valid_0's auc: 0.670199
[120]	valid_0's auc: 0.670888
[130]	valid_0's auc: 0.670839
[140]	valid_0's auc: 0.671166
[150]	valid_0's auc: 0.671309
[160]	valid_0's auc: 0.671473
[170]	valid_0's auc: 0.671614
[180]	valid_0's auc: 0.671736
[190]	valid_0's auc: 0.671813
[200]	valid_0's auc: 0.671959
[210]	valid_0's auc: 0.672097
[220]	valid_0's auc: 0.672016
[230]	valid_0's auc: 0.672062
[240]	valid_0's auc: 0.672138
[250]	valid_0's auc: 0.672208
[260]	valid_0's auc: 0.67232
[270]	valid_0's auc: 0.672341
[280]	valid_0's auc: 0.672382
[290]	valid_0's auc: 0.672478
[300]	valid_0's auc: 0.672604
[310]	valid_0's auc: 0.672611
[320]	valid_0's auc: 0.672593
[330]	valid_0's auc: 0.672635
[340]	valid_0's auc: 0.672714
[350]	valid_0's auc: 0.672772
[360]	valid_0's auc: 0.672751
[370]	valid_0's auc: 0.672721
[380]	valid_0's auc: 0.67272
[390]	valid_0's auc: 0.672712
[400]	valid_0's auc: 0.672737
Early stopping, best iteration is:
[350]	valid_0's auc: 0.672772
complete on: genre_ids
working on: length_bin_range
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
length_bin_range      category
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.644072
[20]	valid_0's auc: 0.650499
[30]	valid_0's auc: 0.656543
[40]	valid_0's auc: 0.660402
[50]	valid_0's auc: 0.663645
[60]	valid_0's auc: 0.665309
[70]	valid_0's auc: 0.666822
[80]	valid_0's auc: 0.667938
[90]	valid_0's auc: 0.668587
[100]	valid_0's auc: 0.669139
[110]	valid_0's auc: 0.669704
[120]	valid_0's auc: 0.670194
[130]	valid_0's auc: 0.670488
[140]	valid_0's auc: 0.670815
[150]	valid_0's auc: 0.671023
[160]	valid_0's auc: 0.671207
[170]	valid_0's auc: 0.671411
[180]	valid_0's auc: 0.671525
[190]	valid_0's auc: 0.671583
[200]	valid_0's auc: 0.671634
[210]	valid_0's auc: 0.671683
[220]	valid_0's auc: 0.671701
[230]	valid_0's auc: 0.671725
[240]	valid_0's auc: 0.671808
[250]	valid_0's auc: 0.671893
[260]	valid_0's auc: 0.6719
[270]	valid_0's auc: 0.671892
[280]	valid_0's auc: 0.671937
[290]	valid_0's auc: 0.671951
[300]	valid_0's auc: 0.671972
[310]	valid_0's auc: 0.671991
[320]	valid_0's auc: 0.672036
[330]	valid_0's auc: 0.672063
[340]	valid_0's auc: 0.672071
[350]	valid_0's auc: 0.672123
[360]	valid_0's auc: 0.672171
[370]	valid_0's auc: 0.672202
[380]	valid_0's auc: 0.672193
[390]	valid_0's auc: 0.672175
[400]	valid_0's auc: 0.672183
[410]	valid_0's auc: 0.672171
[420]	valid_0's auc: 0.672163
[430]	valid_0's auc: 0.67215
[440]	valid_0's auc: 0.6721
[450]	valid_0's auc: 0.672073
Early stopping, best iteration is:
[403]	valid_0's auc: 0.672221
complete on: length_bin_range
working on: length_chunk_range
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
length_chunk_range    category
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.643679
[20]	valid_0's auc: 0.650461
[30]	valid_0's auc: 0.656419
[40]	valid_0's auc: 0.660979
[50]	valid_0's auc: 0.663923
[60]	valid_0's auc: 0.665952
[70]	valid_0's auc: 0.667293
[80]	valid_0's auc: 0.668355
[90]	valid_0's auc: 0.669155
[100]	valid_0's auc: 0.669848
[110]	valid_0's auc: 0.670328
[120]	valid_0's auc: 0.670605
[130]	valid_0's auc: 0.670814
[140]	valid_0's auc: 0.671129
[150]	valid_0's auc: 0.671256
[160]	valid_0's auc: 0.671505
[170]	valid_0's auc: 0.671667
[180]	valid_0's auc: 0.671739
[190]	valid_0's auc: 0.67184
[200]	valid_0's auc: 0.67183
[210]	valid_0's auc: 0.671932
[220]	valid_0's auc: 0.672029
[230]	valid_0's auc: 0.672048
[240]	valid_0's auc: 0.672042
[250]	valid_0's auc: 0.672049
[260]	valid_0's auc: 0.672084
[270]	valid_0's auc: 0.6721
[280]	valid_0's auc: 0.672095
[290]	valid_0's auc: 0.672087
[300]	valid_0's auc: 0.672168
[310]	valid_0's auc: 0.672207
[320]	valid_0's auc: 0.672257
[330]	valid_0's auc: 0.672245
[340]	valid_0's auc: 0.672273
[350]	valid_0's auc: 0.672335
[360]	valid_0's auc: 0.672343
[370]	valid_0's auc: 0.672385
[380]	valid_0's auc: 0.672365
[390]	valid_0's auc: 0.672436
[400]	valid_0's auc: 0.672471
[410]	valid_0's auc: 0.672469
[420]	valid_0's auc: 0.672431
[430]	valid_0's auc: 0.672463
[440]	valid_0's auc: 0.67246
[450]	valid_0's auc: 0.67247
Early stopping, best iteration is:
[405]	valid_0's auc: 0.672499
complete on: length_chunk_range

[timer]: complete in 35m 26s

Process finished with exit code 0
'''