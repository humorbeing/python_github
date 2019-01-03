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
    'num_leaves': 127,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 15,
    'max_depth': -1,
}
df['song_year'] = df['song_year'].astype('category')
on = ['msno',
      'song_id',
      'target',
      'source_system_tab',
      'source_screen_name',
      'source_type',
      'language',
      'artist_name',
      'fake_song_count',
      'fake_member_count',
      # candidate
      # 'fake_artist_count',
      # 'fake_source_screen_name_count',
      # new members
      # 'fake_genre_type_count',
      # 'fake_top1',
      
      'song_year',
      'song_country',
      'fake_song_year_count',
      'fake_song_country_count',
      'fake_top1_count',
      ]
df = df[on]
fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'fake_song_count',
         'fake_member_count',
         ]

for w in df.columns:
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
        print()
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
        del dt
        df = df[on]

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1002/one_in_column_member_count_int.py
What we got:
msno                               object
song_id                            object
source_system_tab                  object
source_screen_name                 object
source_type                        object
target                              uint8
fake_member_count                   int64
member_count                        int64
genre_ids                          object
artist_name                        object
language                         category
fake_genre_type_count               int64
song_year                           int64
song_country                       object
fake_song_count                     int64
fake_artist_count                   int64
fake_song_year_count                int64
fake_song_country_count             int64
fake_top1                        category
fake_top1_count                     int64
fake_source_screen_name_count       int64
dtype: object
number of columns: 21
working on: song_year


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
fake_member_count        int64
song_year             category
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656604
[20]	valid_0's auc: 0.66469
[30]	valid_0's auc: 0.670479
[40]	valid_0's auc: 0.675372
[50]	valid_0's auc: 0.678545
[60]	valid_0's auc: 0.680608
[70]	valid_0's auc: 0.681642
[80]	valid_0's auc: 0.682486
[90]	valid_0's auc: 0.683006
[100]	valid_0's auc: 0.683414
[110]	valid_0's auc: 0.683722
[120]	valid_0's auc: 0.684019
[130]	valid_0's auc: 0.684188
[140]	valid_0's auc: 0.684271
[150]	valid_0's auc: 0.684364
[160]	valid_0's auc: 0.684551
[170]	valid_0's auc: 0.684534
[180]	valid_0's auc: 0.684581
[190]	valid_0's auc: 0.684708
[200]	valid_0's auc: 0.684753
[210]	valid_0's auc: 0.684893
[220]	valid_0's auc: 0.684972
[230]	valid_0's auc: 0.685001
[240]	valid_0's auc: 0.685019
[250]	valid_0's auc: 0.685025
[260]	valid_0's auc: 0.685144
[270]	valid_0's auc: 0.685103
[280]	valid_0's auc: 0.685047
[290]	valid_0's auc: 0.685007
[300]	valid_0's auc: 0.684965
[310]	valid_0's auc: 0.684946
Early stopping, best iteration is:
[261]	valid_0's auc: 0.685149
best score: 0.685149216507
best iteration: 261
complete on: song_year

working on: song_country


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
fake_member_count        int64
song_country          category
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656609
[20]	valid_0's auc: 0.664015
[30]	valid_0's auc: 0.670237
[40]	valid_0's auc: 0.675032
[50]	valid_0's auc: 0.678599
[60]	valid_0's auc: 0.680308
[70]	valid_0's auc: 0.681675
[80]	valid_0's auc: 0.682331
[90]	valid_0's auc: 0.68279
[100]	valid_0's auc: 0.683193
[110]	valid_0's auc: 0.683391
[120]	valid_0's auc: 0.68356
[130]	valid_0's auc: 0.683646
[140]	valid_0's auc: 0.68373
[150]	valid_0's auc: 0.683841
[160]	valid_0's auc: 0.684127
[170]	valid_0's auc: 0.684216
[180]	valid_0's auc: 0.684303
[190]	valid_0's auc: 0.684417
[200]	valid_0's auc: 0.684374
[210]	valid_0's auc: 0.684459
[220]	valid_0's auc: 0.684462
[230]	valid_0's auc: 0.68445
[240]	valid_0's auc: 0.684537
[250]	valid_0's auc: 0.684452
[260]	valid_0's auc: 0.684369
[270]	valid_0's auc: 0.68443
[280]	valid_0's auc: 0.684395
[290]	valid_0's auc: 0.684298
Early stopping, best iteration is:
[240]	valid_0's auc: 0.684537
best score: 0.684536647695
best iteration: 240
complete on: song_country

working on: fake_song_year_count


After selection:
msno                    category
song_id                 category
target                     uint8
source_system_tab       category
source_screen_name      category
source_type             category
language                category
artist_name             category
fake_song_count            int64
fake_member_count          int64
fake_song_year_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656787
[20]	valid_0's auc: 0.664121
[30]	valid_0's auc: 0.670144
[40]	valid_0's auc: 0.675205
[50]	valid_0's auc: 0.67828
[60]	valid_0's auc: 0.68045
[70]	valid_0's auc: 0.681428
[80]	valid_0's auc: 0.682195
[90]	valid_0's auc: 0.682701
[100]	valid_0's auc: 0.683122
[110]	valid_0's auc: 0.683513
[120]	valid_0's auc: 0.683717
[130]	valid_0's auc: 0.68398
[140]	valid_0's auc: 0.684289
[150]	valid_0's auc: 0.684358
[160]	valid_0's auc: 0.68443
[170]	valid_0's auc: 0.684544
[180]	valid_0's auc: 0.684564
[190]	valid_0's auc: 0.684618
[200]	valid_0's auc: 0.684617
[210]	valid_0's auc: 0.684614
[220]	valid_0's auc: 0.684597
[230]	valid_0's auc: 0.684737
[240]	valid_0's auc: 0.684695
[250]	valid_0's auc: 0.684606
[260]	valid_0's auc: 0.684621
[270]	valid_0's auc: 0.684524
[280]	valid_0's auc: 0.684524
Early stopping, best iteration is:
[233]	valid_0's auc: 0.684741
best score: 0.684741228659
best iteration: 233
complete on: fake_song_year_count

working on: fake_song_country_count


After selection:
msno                       category
song_id                    category
target                        uint8
source_system_tab          category
source_screen_name         category
source_type                category
language                   category
artist_name                category
fake_song_count               int64
fake_member_count             int64
fake_song_country_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656257
[20]	valid_0's auc: 0.664465
[30]	valid_0's auc: 0.670484
[40]	valid_0's auc: 0.675147
[50]	valid_0's auc: 0.678293
[60]	valid_0's auc: 0.68081
[70]	valid_0's auc: 0.68202
[80]	valid_0's auc: 0.682799
[90]	valid_0's auc: 0.683145
[100]	valid_0's auc: 0.683474
[110]	valid_0's auc: 0.68385
[120]	valid_0's auc: 0.683925
[130]	valid_0's auc: 0.684034
[140]	valid_0's auc: 0.684069
[150]	valid_0's auc: 0.684166
[160]	valid_0's auc: 0.684246
[170]	valid_0's auc: 0.68435
[180]	valid_0's auc: 0.684405
[190]	valid_0's auc: 0.684495
[200]	valid_0's auc: 0.684527
[210]	valid_0's auc: 0.684543
[220]	valid_0's auc: 0.684564
[230]	valid_0's auc: 0.68454
[240]	valid_0's auc: 0.684606
[250]	valid_0's auc: 0.684703
[260]	valid_0's auc: 0.68465
[270]	valid_0's auc: 0.684594
[280]	valid_0's auc: 0.684568
[290]	valid_0's auc: 0.684561
[300]	valid_0's auc: 0.684511
Early stopping, best iteration is:
[252]	valid_0's auc: 0.684715
best score: 0.684714580792
best iteration: 252
complete on: fake_song_country_count

working on: fake_top1_count


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
fake_member_count        int64
fake_top1_count          int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656603
[20]	valid_0's auc: 0.663786
[30]	valid_0's auc: 0.670166
[40]	valid_0's auc: 0.675376
[50]	valid_0's auc: 0.678375
[60]	valid_0's auc: 0.680391
[70]	valid_0's auc: 0.681776
[80]	valid_0's auc: 0.682454
[90]	valid_0's auc: 0.683034
[100]	valid_0's auc: 0.683308
[110]	valid_0's auc: 0.683507
[120]	valid_0's auc: 0.683574
[130]	valid_0's auc: 0.683651
[140]	valid_0's auc: 0.683939
[150]	valid_0's auc: 0.683971
[160]	valid_0's auc: 0.684161
[170]	valid_0's auc: 0.68422
[180]	valid_0's auc: 0.684249
[190]	valid_0's auc: 0.684231
[200]	valid_0's auc: 0.684386
[210]	valid_0's auc: 0.684466
[220]	valid_0's auc: 0.684483
[230]	valid_0's auc: 0.68449
[240]	valid_0's auc: 0.684455
[250]	valid_0's auc: 0.684477
[260]	valid_0's auc: 0.684437
[270]	valid_0's auc: 0.68443
Early stopping, best iteration is:
[225]	valid_0's auc: 0.684516
best score: 0.68451588171
best iteration: 225
complete on: fake_top1_count


[timer]: complete in 35m 11s

Process finished with exit code 0
'''
