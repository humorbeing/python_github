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
save_dir = '../saves01/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
# df.drop('gender', axis=1, inplace=True)
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
    'num_leaves': 2**6-1,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 15,
    'max_depth': 5,
}
# df['song_year'] = df['song_year'].astype('category')
on = [

      'target',
      'isrc_rest',
      'top1_in_song',
    'top2_in_song',
    'top3_in_song'
      # 'msno',
      # 'song_id',
      # 'source_system_tab',
      # 'source_screen_name',
      # 'source_type',
      # 'language',
      # 'artist_name',
      # 'fake_song_count',
      # 'fake_member_count',
      # candidate
      # 'fake_artist_count',
      # 'fake_source_screen_name_count',
      # new members
      # 'fake_genre_type_count',
      # 'fake_top1',

      # 'song_year', # int
      # 'song_country',
      # 'fake_song_year_count', #00
      # 'fake_song_country_count', # 00
      # 'fake_top1_count', # 00
      ]
df = df[on]

fixed = [
         # 'msno',
         # 'song_id',
         'target',
         # 'source_system_tab',
         # 'source_screen_name',
         # 'source_type',
         # 'language',
         # 'artist_name',
         # 'fake_song_count',
         # 'fake_member_count',
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
        # df.drop('gender', axis=1, inplace=True)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1003/one_in_column_member_count_int.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
expiration_month      category
genre_ids               object
artist_name             object
composer                object
lyricist                object
language              category
name                    object
song_year             category
song_country          category
rc                    category
isrc_rest             category
top1_in_song          category
top2_in_song          category
top3_in_song          category
dtype: object
number of columns: 20
working on: isrc_rest


After selection:
target          uint8
isrc_rest    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529978
[20]	valid_0's auc: 0.535596
[30]	valid_0's auc: 0.53731
[40]	valid_0's auc: 0.538467
[50]	valid_0's auc: 0.539553
[60]	valid_0's auc: 0.540325
[70]	valid_0's auc: 0.540879
[80]	valid_0's auc: 0.541225
[90]	valid_0's auc: 0.54165
[100]	valid_0's auc: 0.542009
[110]	valid_0's auc: 0.543692
[120]	valid_0's auc: 0.544653
[130]	valid_0's auc: 0.545857
[140]	valid_0's auc: 0.547632
[150]	valid_0's auc: 0.548576
[160]	valid_0's auc: 0.549595
[170]	valid_0's auc: 0.549657
[180]	valid_0's auc: 0.549726
[190]	valid_0's auc: 0.550383
[200]	valid_0's auc: 0.550554
[210]	valid_0's auc: 0.550619
[220]	valid_0's auc: 0.55075
[230]	valid_0's auc: 0.550928
[240]	valid_0's auc: 0.551019
[250]	valid_0's auc: 0.550989
[260]	valid_0's auc: 0.551116
[270]	valid_0's auc: 0.551259
[280]	valid_0's auc: 0.551459
[290]	valid_0's auc: 0.551182
[300]	valid_0's auc: 0.55135
[310]	valid_0's auc: 0.551307
[320]	valid_0's auc: 0.551431
[330]	valid_0's auc: 0.551513
[340]	valid_0's auc: 0.551647
[350]	valid_0's auc: 0.551826
[360]	valid_0's auc: 0.551898
[370]	valid_0's auc: 0.55181
[380]	valid_0's auc: 0.55182
[390]	valid_0's auc: 0.551779
[400]	valid_0's auc: 0.551866
[410]	valid_0's auc: 0.551962
[420]	valid_0's auc: 0.552791
[430]	valid_0's auc: 0.552956
[440]	valid_0's auc: 0.553285
[450]	valid_0's auc: 0.553304
[460]	valid_0's auc: 0.553399
[470]	valid_0's auc: 0.554359
[480]	valid_0's auc: 0.5544
[490]	valid_0's auc: 0.554405
[500]	valid_0's auc: 0.555029
[510]	valid_0's auc: 0.555026
[520]	valid_0's auc: 0.555066
[530]	valid_0's auc: 0.555169
[540]	valid_0's auc: 0.555163
[550]	valid_0's auc: 0.555183
[560]	valid_0's auc: 0.55518
[570]	valid_0's auc: 0.555805
[580]	valid_0's auc: 0.55581
[590]	valid_0's auc: 0.555833
[600]	valid_0's auc: 0.555878
[610]	valid_0's auc: 0.555894
[620]	valid_0's auc: 0.555877
[630]	valid_0's auc: 0.555899
[640]	valid_0's auc: 0.555901
[650]	valid_0's auc: 0.556034
[660]	valid_0's auc: 0.556028
[670]	valid_0's auc: 0.555993
[680]	valid_0's auc: 0.556015
[690]	valid_0's auc: 0.556004
Early stopping, best iteration is:
[646]	valid_0's auc: 0.556045
best score: 0.556045123688
best iteration: 646
complete on: isrc_rest

working on: top1_in_song


After selection:
target             uint8
top1_in_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.526064
[20]	valid_0's auc: 0.526063
[30]	valid_0's auc: 0.526063
[40]	valid_0's auc: 0.526063
[50]	valid_0's auc: 0.526063
Early stopping, best iteration is:
[1]	valid_0's auc: 0.526698
best score: 0.526698024873
best iteration: 1
complete on: top1_in_song

working on: top2_in_song


After selection:
target             uint8
top2_in_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.527672
[20]	valid_0's auc: 0.527672
[30]	valid_0's auc: 0.52829
[40]	valid_0's auc: 0.52829
[50]	valid_0's auc: 0.52829
[60]	valid_0's auc: 0.52829
[70]	valid_0's auc: 0.52829
Early stopping, best iteration is:
[27]	valid_0's auc: 0.52829
best score: 0.528290186128
best iteration: 27
complete on: top2_in_song

working on: top3_in_song


After selection:
target             uint8
top3_in_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.524067
[20]	valid_0's auc: 0.524075
[30]	valid_0's auc: 0.524075
[40]	valid_0's auc: 0.524075
[50]	valid_0's auc: 0.524075
Early stopping, best iteration is:
[3]	valid_0's auc: 0.525001
best score: 0.525001347
best iteration: 3
complete on: top3_in_song


[timer]: complete in 6m 44s

Process finished with exit code 0
'''