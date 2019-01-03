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

working_on = [
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'source_system_tab_guess',
    'source_screen_name_guess',
    'source_type_guess',
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
working on: source_system_tab
Train test and validation sets


After selection:
msno                 category
song_id              category
source_system_tab    category
target                  uint8
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
[10]	valid_0's auc: 0.640507
[20]	valid_0's auc: 0.64403
[30]	valid_0's auc: 0.647886
[40]	valid_0's auc: 0.651008
[50]	valid_0's auc: 0.65275
[60]	valid_0's auc: 0.654031
[70]	valid_0's auc: 0.655212
[80]	valid_0's auc: 0.65561
[90]	valid_0's auc: 0.655869
[100]	valid_0's auc: 0.656259
[110]	valid_0's auc: 0.656417
[120]	valid_0's auc: 0.656526
[130]	valid_0's auc: 0.656548
[140]	valid_0's auc: 0.656576
[150]	valid_0's auc: 0.656502
[160]	valid_0's auc: 0.656416
[170]	valid_0's auc: 0.656265
[180]	valid_0's auc: 0.656156
[190]	valid_0's auc: 0.656125
[200]	valid_0's auc: 0.656056
[210]	valid_0's auc: 0.656054
[220]	valid_0's auc: 0.656054
[230]	valid_0's auc: 0.655951
[240]	valid_0's auc: 0.655867
[250]	valid_0's auc: 0.655841
[260]	valid_0's auc: 0.655769
[270]	valid_0's auc: 0.65573
[280]	valid_0's auc: 0.655707
[290]	valid_0's auc: 0.655676
[300]	valid_0's auc: 0.655652
[310]	valid_0's auc: 0.655591
[320]	valid_0's auc: 0.655528
[330]	valid_0's auc: 0.655468
[340]	valid_0's auc: 0.655465
Early stopping, best iteration is:
[140]	valid_0's auc: 0.656576
complete on: source_system_tab
working on: source_screen_name
Train test and validation sets


After selection:
msno                  category
song_id               category
source_screen_name    category
target                   uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.642579
[20]	valid_0's auc: 0.646483
[30]	valid_0's auc: 0.649929
[40]	valid_0's auc: 0.653431
[50]	valid_0's auc: 0.655633
[60]	valid_0's auc: 0.657237
[70]	valid_0's auc: 0.658478
[80]	valid_0's auc: 0.658911
[90]	valid_0's auc: 0.659186
[100]	valid_0's auc: 0.659542
[110]	valid_0's auc: 0.65969
[120]	valid_0's auc: 0.65968
[130]	valid_0's auc: 0.659836
[140]	valid_0's auc: 0.659919
[150]	valid_0's auc: 0.65984
[160]	valid_0's auc: 0.659922
[170]	valid_0's auc: 0.659946
[180]	valid_0's auc: 0.659857
[190]	valid_0's auc: 0.659821
[200]	valid_0's auc: 0.659844
[210]	valid_0's auc: 0.659901
[220]	valid_0's auc: 0.659882
[230]	valid_0's auc: 0.659945
[240]	valid_0's auc: 0.659867
[250]	valid_0's auc: 0.659754
[260]	valid_0's auc: 0.65973
[270]	valid_0's auc: 0.659664
[280]	valid_0's auc: 0.659581
[290]	valid_0's auc: 0.659556
[300]	valid_0's auc: 0.659572
[310]	valid_0's auc: 0.659553
[320]	valid_0's auc: 0.659528
[330]	valid_0's auc: 0.659526
[340]	valid_0's auc: 0.659479
[350]	valid_0's auc: 0.65945
[360]	valid_0's auc: 0.65944
Early stopping, best iteration is:
[164]	valid_0's auc: 0.659958
complete on: source_screen_name
working on: source_type
Train test and validation sets


After selection:
msno           category
song_id        category
source_type    category
target            uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.644172
[20]	valid_0's auc: 0.647735
[30]	valid_0's auc: 0.651274
[40]	valid_0's auc: 0.655124
[50]	valid_0's auc: 0.657675
[60]	valid_0's auc: 0.65943
[70]	valid_0's auc: 0.660615
[80]	valid_0's auc: 0.661201
[90]	valid_0's auc: 0.661675
[100]	valid_0's auc: 0.662028
[110]	valid_0's auc: 0.662298
[120]	valid_0's auc: 0.662564
[130]	valid_0's auc: 0.662698
[140]	valid_0's auc: 0.66283
[150]	valid_0's auc: 0.662889
[160]	valid_0's auc: 0.66289
[170]	valid_0's auc: 0.662942
[180]	valid_0's auc: 0.662853
[190]	valid_0's auc: 0.662794
[200]	valid_0's auc: 0.662877
[210]	valid_0's auc: 0.662872
[220]	valid_0's auc: 0.662853
[230]	valid_0's auc: 0.662797
[240]	valid_0's auc: 0.662782
[250]	valid_0's auc: 0.662703
[260]	valid_0's auc: 0.662708
[270]	valid_0's auc: 0.662654
[280]	valid_0's auc: 0.662638
[290]	valid_0's auc: 0.662632
[300]	valid_0's auc: 0.662678
[310]	valid_0's auc: 0.662634
[320]	valid_0's auc: 0.662651
[330]	valid_0's auc: 0.662655
[340]	valid_0's auc: 0.662633
[350]	valid_0's auc: 0.662664
[360]	valid_0's auc: 0.66268
[370]	valid_0's auc: 0.662649
Early stopping, best iteration is:
[173]	valid_0's auc: 0.662949
complete on: source_type
working on: source_system_tab_guess
Train test and validation sets


After selection:
msno                       category
song_id                    category
source_system_tab_guess    category
target                        uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.640032
[20]	valid_0's auc: 0.643427
[30]	valid_0's auc: 0.646835
[40]	valid_0's auc: 0.650111
[50]	valid_0's auc: 0.652094
[60]	valid_0's auc: 0.653663
[70]	valid_0's auc: 0.654471
[80]	valid_0's auc: 0.654652
[90]	valid_0's auc: 0.65505
[100]	valid_0's auc: 0.655323
[110]	valid_0's auc: 0.655455
[120]	valid_0's auc: 0.655559
[130]	valid_0's auc: 0.655531
[140]	valid_0's auc: 0.655678
[150]	valid_0's auc: 0.655594
[160]	valid_0's auc: 0.655511
[170]	valid_0's auc: 0.655439
[180]	valid_0's auc: 0.655287
[190]	valid_0's auc: 0.655259
[200]	valid_0's auc: 0.655244
[210]	valid_0's auc: 0.65518
[220]	valid_0's auc: 0.655117
[230]	valid_0's auc: 0.655152
[240]	valid_0's auc: 0.655255
[250]	valid_0's auc: 0.655228
[260]	valid_0's auc: 0.655169
[270]	valid_0's auc: 0.655113
[280]	valid_0's auc: 0.655034
[290]	valid_0's auc: 0.654957
[300]	valid_0's auc: 0.654897
[310]	valid_0's auc: 0.654859
[320]	valid_0's auc: 0.654807
[330]	valid_0's auc: 0.654791
[340]	valid_0's auc: 0.654756
Early stopping, best iteration is:
[141]	valid_0's auc: 0.655695
complete on: source_system_tab_guess
working on: source_screen_name_guess
Train test and validation sets


After selection:
msno                        category
song_id                     category
source_screen_name_guess    category
target                         uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.638809
[20]	valid_0's auc: 0.643306
[30]	valid_0's auc: 0.647073
[40]	valid_0's auc: 0.649757
[50]	valid_0's auc: 0.652012
[60]	valid_0's auc: 0.653398
[70]	valid_0's auc: 0.654296
[80]	valid_0's auc: 0.654984
[90]	valid_0's auc: 0.655157
[100]	valid_0's auc: 0.655587
[110]	valid_0's auc: 0.655871
[120]	valid_0's auc: 0.656107
[130]	valid_0's auc: 0.656324
[140]	valid_0's auc: 0.65632
[150]	valid_0's auc: 0.656298
[160]	valid_0's auc: 0.656233
[170]	valid_0's auc: 0.65609
[180]	valid_0's auc: 0.65602
[190]	valid_0's auc: 0.655985
[200]	valid_0's auc: 0.65601
[210]	valid_0's auc: 0.656073
[220]	valid_0's auc: 0.656095
[230]	valid_0's auc: 0.655997
[240]	valid_0's auc: 0.655917
[250]	valid_0's auc: 0.655906
[260]	valid_0's auc: 0.655897
[270]	valid_0's auc: 0.655862
[280]	valid_0's auc: 0.655885
[290]	valid_0's auc: 0.655825
[300]	valid_0's auc: 0.655789
[310]	valid_0's auc: 0.655814
[320]	valid_0's auc: 0.655859
[330]	valid_0's auc: 0.655888
Early stopping, best iteration is:
[133]	valid_0's auc: 0.656373
complete on: source_screen_name_guess
working on: source_type_guess
Train test and validation sets


After selection:
msno                 category
song_id              category
source_type_guess    category
target                  uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.644903
[20]	valid_0's auc: 0.647889
[30]	valid_0's auc: 0.651684
[40]	valid_0's auc: 0.655457
[50]	valid_0's auc: 0.657616
[60]	valid_0's auc: 0.659317
[70]	valid_0's auc: 0.660553
[80]	valid_0's auc: 0.661226
[90]	valid_0's auc: 0.661622
[100]	valid_0's auc: 0.662189
[110]	valid_0's auc: 0.662553
[120]	valid_0's auc: 0.662819
[130]	valid_0's auc: 0.663037
[140]	valid_0's auc: 0.663219
[150]	valid_0's auc: 0.66325
[160]	valid_0's auc: 0.66322
[170]	valid_0's auc: 0.663239
[180]	valid_0's auc: 0.663153
[190]	valid_0's auc: 0.66309
[200]	valid_0's auc: 0.663118
[210]	valid_0's auc: 0.663099
[220]	valid_0's auc: 0.663176
[230]	valid_0's auc: 0.663105
[240]	valid_0's auc: 0.663111
[250]	valid_0's auc: 0.663002
[260]	valid_0's auc: 0.662963
[270]	valid_0's auc: 0.662946
[280]	valid_0's auc: 0.662941
[290]	valid_0's auc: 0.662866
[300]	valid_0's auc: 0.662907
[310]	valid_0's auc: 0.662958
[320]	valid_0's auc: 0.662952
[330]	valid_0's auc: 0.662951
[340]	valid_0's auc: 0.662883
Early stopping, best iteration is:
[147]	valid_0's auc: 0.663287
complete on: source_type_guess

[timer]: complete in 82m 19s

Process finished with exit code 0
'''