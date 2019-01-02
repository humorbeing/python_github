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
# df['song_year'] = df['song_year'].astype('category')
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

      'song_year', # int
      # 'song_country',
      'fake_song_year_count', #00
      'fake_song_country_count', # 00
      'fake_top1_count', # 00
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
song_year                int64
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
[10]	valid_0's auc: 0.65672
[20]	valid_0's auc: 0.664253
[30]	valid_0's auc: 0.670273
[40]	valid_0's auc: 0.675096
[50]	valid_0's auc: 0.678089
[60]	valid_0's auc: 0.680022
[70]	valid_0's auc: 0.681114
[80]	valid_0's auc: 0.682087
[90]	valid_0's auc: 0.682659
[100]	valid_0's auc: 0.683028
[110]	valid_0's auc: 0.683456
[120]	valid_0's auc: 0.683902
[130]	valid_0's auc: 0.684416
[140]	valid_0's auc: 0.684546
[150]	valid_0's auc: 0.684751
[160]	valid_0's auc: 0.684837
[170]	valid_0's auc: 0.684933
[180]	valid_0's auc: 0.684886
[190]	valid_0's auc: 0.684871
[200]	valid_0's auc: 0.684882
[210]	valid_0's auc: 0.685012
[220]	valid_0's auc: 0.685052
[230]	valid_0's auc: 0.68504
[240]	valid_0's auc: 0.685135
[250]	valid_0's auc: 0.685118
[260]	valid_0's auc: 0.685102
[270]	valid_0's auc: 0.685178
[280]	valid_0's auc: 0.685072
[290]	valid_0's auc: 0.685187
[300]	valid_0's auc: 0.685263
[310]	valid_0's auc: 0.685343
[320]	valid_0's auc: 0.685455
[330]	valid_0's auc: 0.685383
[340]	valid_0's auc: 0.685331
[350]	valid_0's auc: 0.685283
[360]	valid_0's auc: 0.685254
Early stopping, best iteration is:
[317]	valid_0's auc: 0.685459
best score: 0.685458809134
best iteration: 317
complete on: song_year

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
[10]	valid_0's auc: 0.656817
[20]	valid_0's auc: 0.664285
[30]	valid_0's auc: 0.6703
[40]	valid_0's auc: 0.674994
[50]	valid_0's auc: 0.678157
[60]	valid_0's auc: 0.679982
[70]	valid_0's auc: 0.681228
[80]	valid_0's auc: 0.681869
[90]	valid_0's auc: 0.682203
[100]	valid_0's auc: 0.682767
[110]	valid_0's auc: 0.682932
[120]	valid_0's auc: 0.683281
[130]	valid_0's auc: 0.683301
[140]	valid_0's auc: 0.683471
[150]	valid_0's auc: 0.683673
[160]	valid_0's auc: 0.68384
[170]	valid_0's auc: 0.683915
[180]	valid_0's auc: 0.684003
[190]	valid_0's auc: 0.683985
[200]	valid_0's auc: 0.683961
[210]	valid_0's auc: 0.684096
[220]	valid_0's auc: 0.684135
[230]	valid_0's auc: 0.684167
[240]	valid_0's auc: 0.684256
[250]	valid_0's auc: 0.684231
[260]	valid_0's auc: 0.684271
[270]	valid_0's auc: 0.684327
[280]	valid_0's auc: 0.684405
[290]	valid_0's auc: 0.684427
[300]	valid_0's auc: 0.684473
[310]	valid_0's auc: 0.684479
[320]	valid_0's auc: 0.684439
[330]	valid_0's auc: 0.684458
[340]	valid_0's auc: 0.684439
[350]	valid_0's auc: 0.684362
Early stopping, best iteration is:
[305]	valid_0's auc: 0.684514
best score: 0.684514155936
best iteration: 305
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
[10]	valid_0's auc: 0.656647
[20]	valid_0's auc: 0.663934
[30]	valid_0's auc: 0.669909
[40]	valid_0's auc: 0.67456
[50]	valid_0's auc: 0.677662
[60]	valid_0's auc: 0.679924
[70]	valid_0's auc: 0.680951
[80]	valid_0's auc: 0.681728
[90]	valid_0's auc: 0.682232
[100]	valid_0's auc: 0.682791
[110]	valid_0's auc: 0.682984
[120]	valid_0's auc: 0.683308
[130]	valid_0's auc: 0.683351
[140]	valid_0's auc: 0.683503
[150]	valid_0's auc: 0.68356
[160]	valid_0's auc: 0.68365
[170]	valid_0's auc: 0.683656
[180]	valid_0's auc: 0.683712
[190]	valid_0's auc: 0.683726
[200]	valid_0's auc: 0.683718
[210]	valid_0's auc: 0.68383
[220]	valid_0's auc: 0.683788
[230]	valid_0's auc: 0.683832
[240]	valid_0's auc: 0.683871
[250]	valid_0's auc: 0.683898
[260]	valid_0's auc: 0.683908
[270]	valid_0's auc: 0.683931
[280]	valid_0's auc: 0.683933
[290]	valid_0's auc: 0.683872
[300]	valid_0's auc: 0.68386
[310]	valid_0's auc: 0.683945
[320]	valid_0's auc: 0.683889
[330]	valid_0's auc: 0.683872
[340]	valid_0's auc: 0.683857
[350]	valid_0's auc: 0.683831
Early stopping, best iteration is:
[303]	valid_0's auc: 0.683978
best score: 0.683977861133
best iteration: 303
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
[10]	valid_0's auc: 0.656497
[20]	valid_0's auc: 0.66391
[30]	valid_0's auc: 0.670123
[40]	valid_0's auc: 0.674666
[50]	valid_0's auc: 0.677853
[60]	valid_0's auc: 0.680086
[70]	valid_0's auc: 0.681122
[80]	valid_0's auc: 0.681878
[90]	valid_0's auc: 0.682349
[100]	valid_0's auc: 0.682848
[110]	valid_0's auc: 0.68311
[120]	valid_0's auc: 0.68321
[130]	valid_0's auc: 0.683371
[140]	valid_0's auc: 0.683653
[150]	valid_0's auc: 0.683758
[160]	valid_0's auc: 0.68371
[170]	valid_0's auc: 0.683703
[180]	valid_0's auc: 0.683737
[190]	valid_0's auc: 0.683872
[200]	valid_0's auc: 0.683889
[210]	valid_0's auc: 0.683847
[220]	valid_0's auc: 0.683873
[230]	valid_0's auc: 0.68388
[240]	valid_0's auc: 0.683877
[250]	valid_0's auc: 0.683971
[260]	valid_0's auc: 0.68399
[270]	valid_0's auc: 0.683992
[280]	valid_0's auc: 0.683931
[290]	valid_0's auc: 0.683996
[300]	valid_0's auc: 0.683978
Early stopping, best iteration is:
[259]	valid_0's auc: 0.684009
best score: 0.684008556989
best iteration: 259
complete on: fake_top1_count


[timer]: complete in 29m 30s

Process finished with exit code 0
'''