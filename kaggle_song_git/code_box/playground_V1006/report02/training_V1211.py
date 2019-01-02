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
msno                       object
song_id                    object
source_system_tab          object
source_screen_name         object
source_type                object
target                      uint8
artist_name                object
language                 category
song_count                  int64
liked_song_count            int64
disliked_song_count         int64
artist_count                int64
liked_artist_count          int64
disliked_artist_count       int64
dtype: object
number of columns: 14
working on: song_count
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
song_count               int64
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
[10]	valid_0's auc: 0.65028
[20]	valid_0's auc: 0.657541
[30]	valid_0's auc: 0.664293
[40]	valid_0's auc: 0.669637
[50]	valid_0's auc: 0.672747
[60]	valid_0's auc: 0.674994
[70]	valid_0's auc: 0.676403
[80]	valid_0's auc: 0.677391
[90]	valid_0's auc: 0.67814
[100]	valid_0's auc: 0.678857
[110]	valid_0's auc: 0.67927
[120]	valid_0's auc: 0.679571
[130]	valid_0's auc: 0.679953
[140]	valid_0's auc: 0.680217
[150]	valid_0's auc: 0.680411
[160]	valid_0's auc: 0.680611
[170]	valid_0's auc: 0.680639
[180]	valid_0's auc: 0.68075
[190]	valid_0's auc: 0.680809
[200]	valid_0's auc: 0.680984
[210]	valid_0's auc: 0.681122
[220]	valid_0's auc: 0.681164
[230]	valid_0's auc: 0.681206
[240]	valid_0's auc: 0.681272
[250]	valid_0's auc: 0.681254
[260]	valid_0's auc: 0.681345
[270]	valid_0's auc: 0.681319
[280]	valid_0's auc: 0.68128
[290]	valid_0's auc: 0.681262
[300]	valid_0's auc: 0.681368
[310]	valid_0's auc: 0.68133
[320]	valid_0's auc: 0.681306
[330]	valid_0's auc: 0.68136
[340]	valid_0's auc: 0.681345
[350]	valid_0's auc: 0.681608
[360]	valid_0's auc: 0.681597
[370]	valid_0's auc: 0.681613
[380]	valid_0's auc: 0.681757
[390]	valid_0's auc: 0.681756
[400]	valid_0's auc: 0.681746
[410]	valid_0's auc: 0.681789
[420]	valid_0's auc: 0.681905
[430]	valid_0's auc: 0.681879
[440]	valid_0's auc: 0.681869
[450]	valid_0's auc: 0.681826
[460]	valid_0's auc: 0.681902
Early stopping, best iteration is:
[416]	valid_0's auc: 0.681909
complete on: song_count
working on: liked_song_count
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
liked_song_count         int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.666057
[20]	valid_0's auc: 0.673587
[30]	valid_0's auc: 0.67957
[40]	valid_0's auc: 0.684913
[50]	valid_0's auc: 0.688252
[60]	valid_0's auc: 0.690462
[70]	valid_0's auc: 0.691896
[80]	valid_0's auc: 0.693159
[90]	valid_0's auc: 0.694028
[100]	valid_0's auc: 0.694574
[110]	valid_0's auc: 0.695012
[120]	valid_0's auc: 0.695379
[130]	valid_0's auc: 0.695714
[140]	valid_0's auc: 0.696042
[150]	valid_0's auc: 0.696258
[160]	valid_0's auc: 0.696489
[170]	valid_0's auc: 0.6966
[180]	valid_0's auc: 0.696781
[190]	valid_0's auc: 0.696918
[200]	valid_0's auc: 0.697047
[210]	valid_0's auc: 0.697079
[220]	valid_0's auc: 0.69718
[230]	valid_0's auc: 0.697345
[240]	valid_0's auc: 0.697531
[250]	valid_0's auc: 0.697585
[260]	valid_0's auc: 0.697639
[270]	valid_0's auc: 0.697657
[280]	valid_0's auc: 0.697726
[290]	valid_0's auc: 0.697814
[300]	valid_0's auc: 0.697864
[310]	valid_0's auc: 0.697954
[320]	valid_0's auc: 0.698028
[330]	valid_0's auc: 0.698083
[340]	valid_0's auc: 0.698103
[350]	valid_0's auc: 0.698105
[360]	valid_0's auc: 0.698142
[370]	valid_0's auc: 0.698218
[380]	valid_0's auc: 0.698269
[390]	valid_0's auc: 0.698273
[400]	valid_0's auc: 0.698367
[410]	valid_0's auc: 0.698367
[420]	valid_0's auc: 0.698392
[430]	valid_0's auc: 0.698432
[440]	valid_0's auc: 0.698414
[450]	valid_0's auc: 0.69838
[460]	valid_0's auc: 0.698432
[470]	valid_0's auc: 0.698392
[480]	valid_0's auc: 0.698329
[490]	valid_0's auc: 0.698333
[500]	valid_0's auc: 0.698307
[510]	valid_0's auc: 0.698295
Early stopping, best iteration is:
[463]	valid_0's auc: 0.698463
complete on: liked_song_count
working on: disliked_song_count
Train test and validation sets


After selection:
msno                   category
song_id                category
target                    uint8
source_system_tab      category
source_screen_name     category
source_type            category
language               category
artist_name            category
disliked_song_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.659844
[20]	valid_0's auc: 0.666512
[30]	valid_0's auc: 0.673344
[40]	valid_0's auc: 0.678186
[50]	valid_0's auc: 0.681758
[60]	valid_0's auc: 0.683837
[70]	valid_0's auc: 0.685483
[80]	valid_0's auc: 0.686627
[90]	valid_0's auc: 0.687357
[100]	valid_0's auc: 0.687986
[110]	valid_0's auc: 0.688504
[120]	valid_0's auc: 0.688807
[130]	valid_0's auc: 0.689056
[140]	valid_0's auc: 0.689288
[150]	valid_0's auc: 0.689609
[160]	valid_0's auc: 0.689815
[170]	valid_0's auc: 0.690082
[180]	valid_0's auc: 0.690265
[190]	valid_0's auc: 0.690498
[200]	valid_0's auc: 0.690624
[210]	valid_0's auc: 0.690764
[220]	valid_0's auc: 0.6909
[230]	valid_0's auc: 0.690985
[240]	valid_0's auc: 0.691075
[250]	valid_0's auc: 0.691102
[260]	valid_0's auc: 0.691129
[270]	valid_0's auc: 0.691225
[280]	valid_0's auc: 0.691256
[290]	valid_0's auc: 0.691227
[300]	valid_0's auc: 0.691207
[310]	valid_0's auc: 0.691248
[320]	valid_0's auc: 0.6913
[330]	valid_0's auc: 0.691312
[340]	valid_0's auc: 0.691327
[350]	valid_0's auc: 0.691321
[360]	valid_0's auc: 0.691347
[370]	valid_0's auc: 0.691337
[380]	valid_0's auc: 0.69145
[390]	valid_0's auc: 0.691457
[400]	valid_0's auc: 0.69148
[410]	valid_0's auc: 0.69147
[420]	valid_0's auc: 0.691456
[430]	valid_0's auc: 0.69143
[440]	valid_0's auc: 0.691405
[450]	valid_0's auc: 0.691414
Early stopping, best iteration is:
[406]	valid_0's auc: 0.691499
complete on: disliked_song_count
working on: artist_count
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
artist_count             int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642543
[20]	valid_0's auc: 0.649303
[30]	valid_0's auc: 0.654964
[40]	valid_0's auc: 0.659609
[50]	valid_0's auc: 0.662918
[60]	valid_0's auc: 0.665113
[70]	valid_0's auc: 0.666496
[80]	valid_0's auc: 0.667676
[90]	valid_0's auc: 0.66842
[100]	valid_0's auc: 0.668897
[110]	valid_0's auc: 0.669327
[120]	valid_0's auc: 0.669426
[130]	valid_0's auc: 0.669823
[140]	valid_0's auc: 0.670004
[150]	valid_0's auc: 0.670182
[160]	valid_0's auc: 0.670414
[170]	valid_0's auc: 0.670482
[180]	valid_0's auc: 0.670781
[190]	valid_0's auc: 0.670907
[200]	valid_0's auc: 0.670997
[210]	valid_0's auc: 0.671111
[220]	valid_0's auc: 0.671123
[230]	valid_0's auc: 0.671176
[240]	valid_0's auc: 0.671236
[250]	valid_0's auc: 0.671315
[260]	valid_0's auc: 0.671356
[270]	valid_0's auc: 0.671337
[280]	valid_0's auc: 0.671403
[290]	valid_0's auc: 0.671416
[300]	valid_0's auc: 0.67144
[310]	valid_0's auc: 0.671458
[320]	valid_0's auc: 0.671532
[330]	valid_0's auc: 0.671512
[340]	valid_0's auc: 0.671488
[350]	valid_0's auc: 0.671448
[360]	valid_0's auc: 0.671496
[370]	valid_0's auc: 0.671582
[380]	valid_0's auc: 0.671639
[390]	valid_0's auc: 0.671652
[400]	valid_0's auc: 0.671664
[410]	valid_0's auc: 0.671683
[420]	valid_0's auc: 0.671724
[430]	valid_0's auc: 0.671723
[440]	valid_0's auc: 0.67172
[450]	valid_0's auc: 0.671751
[460]	valid_0's auc: 0.671779
[470]	valid_0's auc: 0.671759
[480]	valid_0's auc: 0.671737
[490]	valid_0's auc: 0.671733
[500]	valid_0's auc: 0.6717
[510]	valid_0's auc: 0.67171
[520]	valid_0's auc: 0.671727
Early stopping, best iteration is:
[475]	valid_0's auc: 0.671782
complete on: artist_count
working on: liked_artist_count
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
liked_artist_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.644148
[20]	valid_0's auc: 0.650084
[30]	valid_0's auc: 0.655677
[40]	valid_0's auc: 0.660299
[50]	valid_0's auc: 0.663505
[60]	valid_0's auc: 0.66595
[70]	valid_0's auc: 0.667161
[80]	valid_0's auc: 0.668304
[90]	valid_0's auc: 0.668886
[100]	valid_0's auc: 0.669428
[110]	valid_0's auc: 0.670029
[120]	valid_0's auc: 0.670338
[130]	valid_0's auc: 0.670617
[140]	valid_0's auc: 0.670809
[150]	valid_0's auc: 0.671224
[160]	valid_0's auc: 0.671426
[170]	valid_0's auc: 0.671688
[180]	valid_0's auc: 0.671869
[190]	valid_0's auc: 0.672008
[200]	valid_0's auc: 0.672019
[210]	valid_0's auc: 0.672159
[220]	valid_0's auc: 0.672344
[230]	valid_0's auc: 0.672495
[240]	valid_0's auc: 0.672575
[250]	valid_0's auc: 0.672613
[260]	valid_0's auc: 0.672649
[270]	valid_0's auc: 0.672706
[280]	valid_0's auc: 0.672757
[290]	valid_0's auc: 0.672769
[300]	valid_0's auc: 0.672874
[310]	valid_0's auc: 0.672955
[320]	valid_0's auc: 0.672996
[330]	valid_0's auc: 0.673035
[340]	valid_0's auc: 0.673035
[350]	valid_0's auc: 0.673093
[360]	valid_0's auc: 0.673108
[370]	valid_0's auc: 0.673106
[380]	valid_0's auc: 0.673121
[390]	valid_0's auc: 0.673179
[400]	valid_0's auc: 0.673205
[410]	valid_0's auc: 0.673312
[420]	valid_0's auc: 0.673265
[430]	valid_0's auc: 0.673215
[440]	valid_0's auc: 0.673188
[450]	valid_0's auc: 0.673114
[460]	valid_0's auc: 0.673109
Early stopping, best iteration is:
[415]	valid_0's auc: 0.673324
complete on: liked_artist_count
working on: disliked_artist_count
Train test and validation sets


After selection:
msno                     category
song_id                  category
target                      uint8
source_system_tab        category
source_screen_name       category
source_type              category
language                 category
artist_name              category
disliked_artist_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.641696
[20]	valid_0's auc: 0.648875
[30]	valid_0's auc: 0.6546
[40]	valid_0's auc: 0.659374
[50]	valid_0's auc: 0.662282
[60]	valid_0's auc: 0.664484
[70]	valid_0's auc: 0.666408
[80]	valid_0's auc: 0.667335
[90]	valid_0's auc: 0.667995
[100]	valid_0's auc: 0.668568
[110]	valid_0's auc: 0.669298
[120]	valid_0's auc: 0.669596
[130]	valid_0's auc: 0.669913
[140]	valid_0's auc: 0.670149
[150]	valid_0's auc: 0.67039
[160]	valid_0's auc: 0.670592
[170]	valid_0's auc: 0.670745
[180]	valid_0's auc: 0.670877
[190]	valid_0's auc: 0.670969
[200]	valid_0's auc: 0.67108
[210]	valid_0's auc: 0.671168
[220]	valid_0's auc: 0.671285
[230]	valid_0's auc: 0.671381
[240]	valid_0's auc: 0.671521
[250]	valid_0's auc: 0.671589
[260]	valid_0's auc: 0.6717
[270]	valid_0's auc: 0.671747
[280]	valid_0's auc: 0.671753
[290]	valid_0's auc: 0.671738
[300]	valid_0's auc: 0.671752
[310]	valid_0's auc: 0.671775
[320]	valid_0's auc: 0.671824
[330]	valid_0's auc: 0.671818
[340]	valid_0's auc: 0.67181
[350]	valid_0's auc: 0.671836
[360]	valid_0's auc: 0.671877
[370]	valid_0's auc: 0.671922
[380]	valid_0's auc: 0.671874
[390]	valid_0's auc: 0.671892
[400]	valid_0's auc: 0.671897
[410]	valid_0's auc: 0.671903
[420]	valid_0's auc: 0.671883
Early stopping, best iteration is:
[371]	valid_0's auc: 0.671924
complete on: disliked_artist_count

[timer]: complete in 56m 24s

Process finished with exit code 0
'''

# 0.63453 with best one