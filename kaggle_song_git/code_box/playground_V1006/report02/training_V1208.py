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
         ]
# working_on = ['language',
#               'artist_name']
# if True:
#     w = 'nothing'
for w in df.columns:
    if w in fixed:
        print('yoyo:', w)
    else:
        print('working on:', w)
        toto = fixed
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
                          num_boost_round=50000,
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1202.py
working on: language
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
dtype: object
number of columns: 7


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.643428
[20]	valid_0's auc: 0.649736
[30]	valid_0's auc: 0.655106
[40]	valid_0's auc: 0.65985
[50]	valid_0's auc: 0.662858
[60]	valid_0's auc: 0.664886
[70]	valid_0's auc: 0.666261
[80]	valid_0's auc: 0.667207
[90]	valid_0's auc: 0.668155
[100]	valid_0's auc: 0.6687
[110]	valid_0's auc: 0.669164
[120]	valid_0's auc: 0.669617
[130]	valid_0's auc: 0.669895
[140]	valid_0's auc: 0.670012
[150]	valid_0's auc: 0.670099
[160]	valid_0's auc: 0.670197
[170]	valid_0's auc: 0.670173
[180]	valid_0's auc: 0.670202
[190]	valid_0's auc: 0.670283
[200]	valid_0's auc: 0.67041
[210]	valid_0's auc: 0.670409
[220]	valid_0's auc: 0.670374
[230]	valid_0's auc: 0.670281
[240]	valid_0's auc: 0.670294
[250]	valid_0's auc: 0.670273
Early stopping, best iteration is:
[207]	valid_0's auc: 0.670422
complete on: language
working on: artist_name
Train test and validation sets


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
artist_name           category
dtype: object
number of columns: 7


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642046
[20]	valid_0's auc: 0.647626
[30]	valid_0's auc: 0.6543
[40]	valid_0's auc: 0.658047
[50]	valid_0's auc: 0.661047
[60]	valid_0's auc: 0.662898
[70]	valid_0's auc: 0.663872
[80]	valid_0's auc: 0.664763
[90]	valid_0's auc: 0.665349
[100]	valid_0's auc: 0.665881
[110]	valid_0's auc: 0.666437
[120]	valid_0's auc: 0.666688
[130]	valid_0's auc: 0.667072
[140]	valid_0's auc: 0.667399
[150]	valid_0's auc: 0.667574
[160]	valid_0's auc: 0.667717
[170]	valid_0's auc: 0.667861
[180]	valid_0's auc: 0.667891
[190]	valid_0's auc: 0.667922
[200]	valid_0's auc: 0.667929
[210]	valid_0's auc: 0.668088
[220]	valid_0's auc: 0.668157
[230]	valid_0's auc: 0.668151
[240]	valid_0's auc: 0.668203
[250]	valid_0's auc: 0.668217
[260]	valid_0's auc: 0.668254
[270]	valid_0's auc: 0.6683
[280]	valid_0's auc: 0.668342
[290]	valid_0's auc: 0.668412
[300]	valid_0's auc: 0.668379
[310]	valid_0's auc: 0.668423
[320]	valid_0's auc: 0.668434
[330]	valid_0's auc: 0.668467
[340]	valid_0's auc: 0.668435
[350]	valid_0's auc: 0.668481
[360]	valid_0's auc: 0.668509
[370]	valid_0's auc: 0.668488
[380]	valid_0's auc: 0.668498
[390]	valid_0's auc: 0.668493
[400]	valid_0's auc: 0.668503
[410]	valid_0's auc: 0.668508
[420]	valid_0's auc: 0.668453
[430]	valid_0's auc: 0.668451
[440]	valid_0's auc: 0.668459
[450]	valid_0's auc: 0.668516
[460]	valid_0's auc: 0.668552
[470]	valid_0's auc: 0.668521
[480]	valid_0's auc: 0.668503
[490]	valid_0's auc: 0.668515
[500]	valid_0's auc: 0.668532
[510]	valid_0's auc: 0.668499
Early stopping, best iteration is:
[460]	valid_0's auc: 0.668552
complete on: artist_name

[timer]: complete in 17m 9s

Process finished with exit code 0
'''