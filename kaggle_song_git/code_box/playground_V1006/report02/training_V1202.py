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
working_on = ['source_system_tab',
              'source_screen_name',
              'source_type',
              'genre_ids',
              'composer',
              'lyricist',
              'rc',
              ]
for w in working_on:
    dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
    df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)

    del dt
    print('working on:', w)
    df = df[['msno', 'song_id', w, 'target']]
    # df['age_range'] = df['age_range'].astype('category')
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
[10]	valid_0's auc: 0.639775
[20]	valid_0's auc: 0.644231
[30]	valid_0's auc: 0.64707
[40]	valid_0's auc: 0.650229
[50]	valid_0's auc: 0.652009
[60]	valid_0's auc: 0.653438
[70]	valid_0's auc: 0.654475
[80]	valid_0's auc: 0.655036
[90]	valid_0's auc: 0.655226
[100]	valid_0's auc: 0.655408
[110]	valid_0's auc: 0.65549
[120]	valid_0's auc: 0.65553
[130]	valid_0's auc: 0.655556
[140]	valid_0's auc: 0.65559
[150]	valid_0's auc: 0.655521
[160]	valid_0's auc: 0.655389
[170]	valid_0's auc: 0.655329
[180]	valid_0's auc: 0.655272
[190]	valid_0's auc: 0.655229
[200]	valid_0's auc: 0.655194
[210]	valid_0's auc: 0.655109
[220]	valid_0's auc: 0.655069
[230]	valid_0's auc: 0.654975
[240]	valid_0's auc: 0.654899
[250]	valid_0's auc: 0.654801
[260]	valid_0's auc: 0.6547
[270]	valid_0's auc: 0.65468
[280]	valid_0's auc: 0.654672
[290]	valid_0's auc: 0.654595
[300]	valid_0's auc: 0.654539
[310]	valid_0's auc: 0.654483
[320]	valid_0's auc: 0.654441
[330]	valid_0's auc: 0.654423
Early stopping, best iteration is:
[132]	valid_0's auc: 0.655608
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
[10]	valid_0's auc: 0.640014
[20]	valid_0's auc: 0.64386
[30]	valid_0's auc: 0.647826
[40]	valid_0's auc: 0.651099
[50]	valid_0's auc: 0.653012
[60]	valid_0's auc: 0.654308
[70]	valid_0's auc: 0.655378
[80]	valid_0's auc: 0.65594
[90]	valid_0's auc: 0.656208
[100]	valid_0's auc: 0.656632
[110]	valid_0's auc: 0.656938
[120]	valid_0's auc: 0.657025
[130]	valid_0's auc: 0.657067
[140]	valid_0's auc: 0.657044
[150]	valid_0's auc: 0.65705
[160]	valid_0's auc: 0.657094
[170]	valid_0's auc: 0.657137
[180]	valid_0's auc: 0.657059
[190]	valid_0's auc: 0.657009
[200]	valid_0's auc: 0.656939
[210]	valid_0's auc: 0.656929
[220]	valid_0's auc: 0.657028
[230]	valid_0's auc: 0.656944
[240]	valid_0's auc: 0.656846
[250]	valid_0's auc: 0.656797
[260]	valid_0's auc: 0.656769
[270]	valid_0's auc: 0.656732
[280]	valid_0's auc: 0.656721
[290]	valid_0's auc: 0.656684
[300]	valid_0's auc: 0.656665
[310]	valid_0's auc: 0.6567
[320]	valid_0's auc: 0.656703
[330]	valid_0's auc: 0.656701
[340]	valid_0's auc: 0.656654
[350]	valid_0's auc: 0.656659
[360]	valid_0's auc: 0.656601
[370]	valid_0's auc: 0.656593
Early stopping, best iteration is:
[173]	valid_0's auc: 0.657147
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
[10]	valid_0's auc: 0.643814
[20]	valid_0's auc: 0.647112
[30]	valid_0's auc: 0.651301
[40]	valid_0's auc: 0.654401
[50]	valid_0's auc: 0.657008
[60]	valid_0's auc: 0.658755
[70]	valid_0's auc: 0.659883
[80]	valid_0's auc: 0.660574
[90]	valid_0's auc: 0.660745
[100]	valid_0's auc: 0.66126
[110]	valid_0's auc: 0.661758
[120]	valid_0's auc: 0.661979
[130]	valid_0's auc: 0.662201
[140]	valid_0's auc: 0.662432
[150]	valid_0's auc: 0.662438
[160]	valid_0's auc: 0.662464
[170]	valid_0's auc: 0.662442
[180]	valid_0's auc: 0.662334
[190]	valid_0's auc: 0.662283
[200]	valid_0's auc: 0.66228
[210]	valid_0's auc: 0.662414
[220]	valid_0's auc: 0.662393
[230]	valid_0's auc: 0.662399
[240]	valid_0's auc: 0.662364
[250]	valid_0's auc: 0.662261
[260]	valid_0's auc: 0.662269
[270]	valid_0's auc: 0.662214
[280]	valid_0's auc: 0.662184
[290]	valid_0's auc: 0.662133
[300]	valid_0's auc: 0.662154
[310]	valid_0's auc: 0.662133
[320]	valid_0's auc: 0.662081
[330]	valid_0's auc: 0.662142
[340]	valid_0's auc: 0.662073
[350]	valid_0's auc: 0.662073
Early stopping, best iteration is:
[151]	valid_0's auc: 0.662505
complete on: source_type
working on: genre_ids
Train test and validation sets


After selection:
msno         category
song_id      category
genre_ids    category
target          uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.60503
[20]	valid_0's auc: 0.612995
[30]	valid_0's auc: 0.616781
[40]	valid_0's auc: 0.618918
[50]	valid_0's auc: 0.620853
[60]	valid_0's auc: 0.622161
[70]	valid_0's auc: 0.623076
[80]	valid_0's auc: 0.623279
[90]	valid_0's auc: 0.623809
[100]	valid_0's auc: 0.62386
[110]	valid_0's auc: 0.623823
[120]	valid_0's auc: 0.623906
[130]	valid_0's auc: 0.623937
[140]	valid_0's auc: 0.623842
[150]	valid_0's auc: 0.623866
[160]	valid_0's auc: 0.623793
[170]	valid_0's auc: 0.623806
[180]	valid_0's auc: 0.623881
[190]	valid_0's auc: 0.623858
[200]	valid_0's auc: 0.623783
[210]	valid_0's auc: 0.62391
[220]	valid_0's auc: 0.623921
[230]	valid_0's auc: 0.624023
[240]	valid_0's auc: 0.623959
[250]	valid_0's auc: 0.624067
[260]	valid_0's auc: 0.624036
[270]	valid_0's auc: 0.623941
[280]	valid_0's auc: 0.623929
[290]	valid_0's auc: 0.623955
[300]	valid_0's auc: 0.623961
[310]	valid_0's auc: 0.624026
[320]	valid_0's auc: 0.62402
[330]	valid_0's auc: 0.624063
[340]	valid_0's auc: 0.624023
[350]	valid_0's auc: 0.624035
[360]	valid_0's auc: 0.624029
[370]	valid_0's auc: 0.624064
[380]	valid_0's auc: 0.624026
[390]	valid_0's auc: 0.624025
[400]	valid_0's auc: 0.624006
[410]	valid_0's auc: 0.623993
[420]	valid_0's auc: 0.624005
[430]	valid_0's auc: 0.623997
[440]	valid_0's auc: 0.623997
Early stopping, best iteration is:
[249]	valid_0's auc: 0.624077
complete on: genre_ids
working on: composer
Train test and validation sets


After selection:
msno        category
song_id     category
composer    category
target         uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601011
[20]	valid_0's auc: 0.609157
[30]	valid_0's auc: 0.613806
[40]	valid_0's auc: 0.617025
[50]	valid_0's auc: 0.618454
[60]	valid_0's auc: 0.619345
[70]	valid_0's auc: 0.619734
[80]	valid_0's auc: 0.619918
[90]	valid_0's auc: 0.620137
[100]	valid_0's auc: 0.620134
[110]	valid_0's auc: 0.620153
[120]	valid_0's auc: 0.620225
[130]	valid_0's auc: 0.620112
[140]	valid_0's auc: 0.62008
[150]	valid_0's auc: 0.620149
[160]	valid_0's auc: 0.620103
[170]	valid_0's auc: 0.620109
[180]	valid_0's auc: 0.620058
[190]	valid_0's auc: 0.620001
[200]	valid_0's auc: 0.61994
[210]	valid_0's auc: 0.619954
[220]	valid_0's auc: 0.619901
[230]	valid_0's auc: 0.61988
[240]	valid_0's auc: 0.619875
[250]	valid_0's auc: 0.619875
[260]	valid_0's auc: 0.619851
[270]	valid_0's auc: 0.619814
[280]	valid_0's auc: 0.619798
[290]	valid_0's auc: 0.619781
Early stopping, best iteration is:
[93]	valid_0's auc: 0.620254
complete on: composer
working on: lyricist
Train test and validation sets


After selection:
msno        category
song_id     category
lyricist    category
target         uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.601801
[20]	valid_0's auc: 0.610103
[30]	valid_0's auc: 0.615101
[40]	valid_0's auc: 0.617712
[50]	valid_0's auc: 0.618839
[60]	valid_0's auc: 0.619678
[70]	valid_0's auc: 0.619994
[80]	valid_0's auc: 0.620213
[90]	valid_0's auc: 0.620585
[100]	valid_0's auc: 0.620594
[110]	valid_0's auc: 0.620567
[120]	valid_0's auc: 0.620597
[130]	valid_0's auc: 0.620571
[140]	valid_0's auc: 0.62054
[150]	valid_0's auc: 0.62049
[160]	valid_0's auc: 0.620481
[170]	valid_0's auc: 0.620481
[180]	valid_0's auc: 0.620314
[190]	valid_0's auc: 0.620289
[200]	valid_0's auc: 0.62028
[210]	valid_0's auc: 0.620235
[220]	valid_0's auc: 0.620208
[230]	valid_0's auc: 0.620189
[240]	valid_0's auc: 0.620187
[250]	valid_0's auc: 0.620149
[260]	valid_0's auc: 0.620056
[270]	valid_0's auc: 0.620076
[280]	valid_0's auc: 0.62008
[290]	valid_0's auc: 0.620105
[300]	valid_0's auc: 0.620099
[310]	valid_0's auc: 0.619962
[320]	valid_0's auc: 0.620001
Early stopping, best iteration is:
[125]	valid_0's auc: 0.620638
complete on: lyricist
working on: rc
Train test and validation sets


After selection:
msno       category
song_id    category
rc         category
target        uint8
dtype: object
number of columns: 4


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.605952
[20]	valid_0's auc: 0.611511
[30]	valid_0's auc: 0.616031
[40]	valid_0's auc: 0.618621
[50]	valid_0's auc: 0.620558
[60]	valid_0's auc: 0.621448
[70]	valid_0's auc: 0.621597
[80]	valid_0's auc: 0.621762
[90]	valid_0's auc: 0.622015
[100]	valid_0's auc: 0.622139
[110]	valid_0's auc: 0.622088
[120]	valid_0's auc: 0.622168
[130]	valid_0's auc: 0.622159
[140]	valid_0's auc: 0.622091
[150]	valid_0's auc: 0.62209
[160]	valid_0's auc: 0.622116
[170]	valid_0's auc: 0.622061
[180]	valid_0's auc: 0.622043
[190]	valid_0's auc: 0.622023
[200]	valid_0's auc: 0.621937
[210]	valid_0's auc: 0.621952
[220]	valid_0's auc: 0.621975
[230]	valid_0's auc: 0.622029
[240]	valid_0's auc: 0.621944
[250]	valid_0's auc: 0.621908
[260]	valid_0's auc: 0.621904
[270]	valid_0's auc: 0.62188
[280]	valid_0's auc: 0.621915
[290]	valid_0's auc: 0.621929
[300]	valid_0's auc: 0.621915
[310]	valid_0's auc: 0.621898
[320]	valid_0's auc: 0.621924
Early stopping, best iteration is:
[126]	valid_0's auc: 0.622204
complete on: rc

[timer]: complete in 97m 42s

Process finished with exit code 0
'''