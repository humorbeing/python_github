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
      # new members
      'fake_artist_count',
      'fake_source_system_tab_count',
      'fake_source_screen_name_count',
      'fake_source_type_count',
      'fake_genre_ids_count',
      'genre_ids',
      'fake_language_count',
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
fake_song_count                     int64
fake_artist_count                   int64
fake_language_count                 int64
fake_genre_ids_count                int64
fake_source_system_tab_count        int64
fake_source_screen_name_count       int64
fake_source_type_count              int64
dtype: object
number of columns: 18
working on: fake_artist_count


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
fake_artist_count        int64
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
[10]	valid_0's auc: 0.656842
[20]	valid_0's auc: 0.663867
[30]	valid_0's auc: 0.670106
[40]	valid_0's auc: 0.674802
[50]	valid_0's auc: 0.678027
[60]	valid_0's auc: 0.680168
[70]	valid_0's auc: 0.681462
[80]	valid_0's auc: 0.682305
[90]	valid_0's auc: 0.682806
[100]	valid_0's auc: 0.683133
[110]	valid_0's auc: 0.683483
[120]	valid_0's auc: 0.683595
[130]	valid_0's auc: 0.683809
[140]	valid_0's auc: 0.683779
[150]	valid_0's auc: 0.683889
[160]	valid_0's auc: 0.684127
[170]	valid_0's auc: 0.684147
[180]	valid_0's auc: 0.684137
[190]	valid_0's auc: 0.684233
[200]	valid_0's auc: 0.684146
[210]	valid_0's auc: 0.6842
[220]	valid_0's auc: 0.684227
[230]	valid_0's auc: 0.684169
Early stopping, best iteration is:
[189]	valid_0's auc: 0.684262
best score: 0.684261872238
best iteration: 189
complete on: fake_artist_count

working on: fake_source_system_tab_count


After selection:
msno                            category
song_id                         category
target                             uint8
source_system_tab               category
source_screen_name              category
source_type                     category
language                        category
artist_name                     category
fake_song_count                    int64
fake_member_count                  int64
fake_source_system_tab_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656651
[20]	valid_0's auc: 0.663808
[30]	valid_0's auc: 0.670064
[40]	valid_0's auc: 0.674772
[50]	valid_0's auc: 0.677814
[60]	valid_0's auc: 0.680037
[70]	valid_0's auc: 0.681066
[80]	valid_0's auc: 0.681765
[90]	valid_0's auc: 0.682283
[100]	valid_0's auc: 0.682581
[110]	valid_0's auc: 0.682753
[120]	valid_0's auc: 0.682921
[130]	valid_0's auc: 0.682968
[140]	valid_0's auc: 0.683139
[150]	valid_0's auc: 0.68332
[160]	valid_0's auc: 0.683466
[170]	valid_0's auc: 0.683482
[180]	valid_0's auc: 0.683475
[190]	valid_0's auc: 0.68373
[200]	valid_0's auc: 0.683781
[210]	valid_0's auc: 0.683753
[220]	valid_0's auc: 0.683735
[230]	valid_0's auc: 0.683733
[240]	valid_0's auc: 0.683703
[250]	valid_0's auc: 0.683675
Early stopping, best iteration is:
[200]	valid_0's auc: 0.683781
best score: 0.683781372382
best iteration: 200
complete on: fake_source_system_tab_count

working on: fake_source_screen_name_count


After selection:
msno                             category
song_id                          category
target                              uint8
source_system_tab                category
source_screen_name               category
source_type                      category
language                         category
artist_name                      category
fake_song_count                     int64
fake_member_count                   int64
fake_source_screen_name_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.657005
[20]	valid_0's auc: 0.664625
[30]	valid_0's auc: 0.670435
[40]	valid_0's auc: 0.67492
[50]	valid_0's auc: 0.678443
[60]	valid_0's auc: 0.6801
[70]	valid_0's auc: 0.68126
[80]	valid_0's auc: 0.681923
[90]	valid_0's auc: 0.68258
[100]	valid_0's auc: 0.683104
[110]	valid_0's auc: 0.683257
[120]	valid_0's auc: 0.683466
[130]	valid_0's auc: 0.683488
[140]	valid_0's auc: 0.683717
[150]	valid_0's auc: 0.683885
[160]	valid_0's auc: 0.683939
[170]	valid_0's auc: 0.684077
[180]	valid_0's auc: 0.684038
[190]	valid_0's auc: 0.684112
[200]	valid_0's auc: 0.684142
[210]	valid_0's auc: 0.684171
[220]	valid_0's auc: 0.684214
[230]	valid_0's auc: 0.684264
[240]	valid_0's auc: 0.684208
[250]	valid_0's auc: 0.684204
[260]	valid_0's auc: 0.68416
[270]	valid_0's auc: 0.684133
[280]	valid_0's auc: 0.684197
Early stopping, best iteration is:
[231]	valid_0's auc: 0.684271
best score: 0.684271133875
best iteration: 231
complete on: fake_source_screen_name_count

working on: fake_source_type_count


After selection:
msno                      category
song_id                   category
target                       uint8
source_system_tab         category
source_screen_name        category
source_type               category
language                  category
artist_name               category
fake_song_count              int64
fake_member_count            int64
fake_source_type_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656631
[20]	valid_0's auc: 0.664324
[30]	valid_0's auc: 0.670439
[40]	valid_0's auc: 0.674991
[50]	valid_0's auc: 0.67811
[60]	valid_0's auc: 0.680264
[70]	valid_0's auc: 0.681221
[80]	valid_0's auc: 0.681827
[90]	valid_0's auc: 0.6823
[100]	valid_0's auc: 0.682597
[110]	valid_0's auc: 0.682894
[120]	valid_0's auc: 0.682984
[130]	valid_0's auc: 0.683054
[140]	valid_0's auc: 0.683246
[150]	valid_0's auc: 0.683366
[160]	valid_0's auc: 0.683499
[170]	valid_0's auc: 0.683467
[180]	valid_0's auc: 0.683549
[190]	valid_0's auc: 0.683538
[200]	valid_0's auc: 0.683584
[210]	valid_0's auc: 0.683583
[220]	valid_0's auc: 0.683552
[230]	valid_0's auc: 0.683573
[240]	valid_0's auc: 0.683564
[250]	valid_0's auc: 0.683633
[260]	valid_0's auc: 0.683593
[270]	valid_0's auc: 0.683511
[280]	valid_0's auc: 0.683543
[290]	valid_0's auc: 0.683557
[300]	valid_0's auc: 0.683507
Early stopping, best iteration is:
[254]	valid_0's auc: 0.683654
best score: 0.683653768894
best iteration: 254
complete on: fake_source_type_count

working on: fake_genre_ids_count


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
fake_genre_ids_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.65644
[20]	valid_0's auc: 0.664319
[30]	valid_0's auc: 0.670196
[40]	valid_0's auc: 0.674819
[50]	valid_0's auc: 0.677843
[60]	valid_0's auc: 0.679895
[70]	valid_0's auc: 0.681131
[80]	valid_0's auc: 0.681843
[90]	valid_0's auc: 0.682363
[100]	valid_0's auc: 0.682648
[110]	valid_0's auc: 0.682945
[120]	valid_0's auc: 0.683222
[130]	valid_0's auc: 0.683208
[140]	valid_0's auc: 0.683366
[150]	valid_0's auc: 0.683531
[160]	valid_0's auc: 0.683601
[170]	valid_0's auc: 0.683663
[180]	valid_0's auc: 0.683622
[190]	valid_0's auc: 0.683736
[200]	valid_0's auc: 0.6839
[210]	valid_0's auc: 0.683956
[220]	valid_0's auc: 0.683998
[230]	valid_0's auc: 0.68402
[240]	valid_0's auc: 0.683959
[250]	valid_0's auc: 0.68391
[260]	valid_0's auc: 0.684049
[270]	valid_0's auc: 0.683969
[280]	valid_0's auc: 0.683981
[290]	valid_0's auc: 0.684044
[300]	valid_0's auc: 0.684047
[310]	valid_0's auc: 0.683925
[320]	valid_0's auc: 0.683934
[330]	valid_0's auc: 0.683917
[340]	valid_0's auc: 0.683834
Early stopping, best iteration is:
[295]	valid_0's auc: 0.6841
best score: 0.684100062705
best iteration: 295
complete on: fake_genre_ids_count

working on: genre_ids


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
genre_ids             category
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.656156
[20]	valid_0's auc: 0.663569
[30]	valid_0's auc: 0.670294
[40]	valid_0's auc: 0.674958
[50]	valid_0's auc: 0.678209
[60]	valid_0's auc: 0.680142
[70]	valid_0's auc: 0.681176
[80]	valid_0's auc: 0.681936
[90]	valid_0's auc: 0.682335
[100]	valid_0's auc: 0.682796
[110]	valid_0's auc: 0.683122
[120]	valid_0's auc: 0.683269
[130]	valid_0's auc: 0.683527
[140]	valid_0's auc: 0.683604
[150]	valid_0's auc: 0.683663
[160]	valid_0's auc: 0.683712
[170]	valid_0's auc: 0.683766
[180]	valid_0's auc: 0.683892
[190]	valid_0's auc: 0.684007
[200]	valid_0's auc: 0.684073
[210]	valid_0's auc: 0.684094
[220]	valid_0's auc: 0.684136
[230]	valid_0's auc: 0.684162
[240]	valid_0's auc: 0.684162
[250]	valid_0's auc: 0.684221
[260]	valid_0's auc: 0.684245
[270]	valid_0's auc: 0.684213
[280]	valid_0's auc: 0.684218
[290]	valid_0's auc: 0.684172
[300]	valid_0's auc: 0.684171
[310]	valid_0's auc: 0.684213
Early stopping, best iteration is:
[260]	valid_0's auc: 0.684245
best score: 0.684244970769
best iteration: 260
complete on: genre_ids

working on: fake_language_count


After selection:
msno                   category
song_id                category
target                    uint8
source_system_tab      category
source_screen_name     category
source_type            category
language               category
artist_name            category
fake_song_count           int64
fake_member_count         int64
fake_language_count       int64
dtype: object
number of columns: 11


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.65662
[20]	valid_0's auc: 0.663823
[30]	valid_0's auc: 0.669989
[40]	valid_0's auc: 0.675049
[50]	valid_0's auc: 0.678078
[60]	valid_0's auc: 0.680022
[70]	valid_0's auc: 0.681066
[80]	valid_0's auc: 0.681862
[90]	valid_0's auc: 0.682388
[100]	valid_0's auc: 0.682792
[110]	valid_0's auc: 0.682938
[120]	valid_0's auc: 0.683236
[130]	valid_0's auc: 0.683276
[140]	valid_0's auc: 0.683431
[150]	valid_0's auc: 0.683696
[160]	valid_0's auc: 0.68371
[170]	valid_0's auc: 0.68378
[180]	valid_0's auc: 0.683768
[190]	valid_0's auc: 0.683842
[200]	valid_0's auc: 0.683864
[210]	valid_0's auc: 0.683864
[220]	valid_0's auc: 0.683911
[230]	valid_0's auc: 0.683999
[240]	valid_0's auc: 0.683966
[250]	valid_0's auc: 0.683976
[260]	valid_0's auc: 0.683898
[270]	valid_0's auc: 0.683933
[280]	valid_0's auc: 0.683879
Early stopping, best iteration is:
[232]	valid_0's auc: 0.684024
best score: 0.68402387099
best iteration: 232
complete on: fake_language_count


[timer]: complete in 46m 59s

Process finished with exit code 0
'''