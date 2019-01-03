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
print()

fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language'
         ]


for w in df.columns:
    if w in fixed:
        pass
    else:
        print()
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df = df[toto]

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype('category')

        print()
        print('This rounds guests:')
        print(df.dtypes)
        print('number of columns:', len(df.columns))
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

        train_set = lgb.Dataset(X_tr, Y_tr)
        val_set = lgb.Dataset(X_val, Y_val)
        del X_tr, Y_tr, X_val, Y_val

        print('Training...')
        print()
        params = {'objective': 'binary',
                  'metric': 'auc',
                  'boosting': 'gbdt',
                  'learning_rate': 0.1,
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

                  }
        model = lgb.train(params,
                          train_set,
                          num_boost_round=500000,
                          early_stopping_rounds=50,
                          valid_sets=val_set,
                          verbose_eval=10,
                          )

        del train_set, val_set
        print()
        print('complete on:', w)
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)

        del dt


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1301.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
artist_name             object
language              category
dtype: object
number of columns: 8


working on: artist_name

This rounds guests:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
dtype: object
number of columns: 8

Training...

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.644099
[20]	valid_0's auc: 0.650248
[30]	valid_0's auc: 0.656097
[40]	valid_0's auc: 0.660687
[50]	valid_0's auc: 0.66367
[60]	valid_0's auc: 0.665752
[70]	valid_0's auc: 0.666845
[80]	valid_0's auc: 0.667755
[90]	valid_0's auc: 0.668657
[100]	valid_0's auc: 0.669479
[110]	valid_0's auc: 0.670126
[120]	valid_0's auc: 0.670538
[130]	valid_0's auc: 0.670907
[140]	valid_0's auc: 0.671173
[150]	valid_0's auc: 0.671426
[160]	valid_0's auc: 0.671605
[170]	valid_0's auc: 0.671635
[180]	valid_0's auc: 0.671748
[190]	valid_0's auc: 0.671809
[200]	valid_0's auc: 0.671877
[210]	valid_0's auc: 0.672017
[220]	valid_0's auc: 0.672064
[230]	valid_0's auc: 0.67209
[240]	valid_0's auc: 0.67211
[250]	valid_0's auc: 0.672142
[260]	valid_0's auc: 0.672177
[270]	valid_0's auc: 0.672217
[280]	valid_0's auc: 0.67222
[290]	valid_0's auc: 0.67223
[300]	valid_0's auc: 0.67222
[310]	valid_0's auc: 0.672263
[320]	valid_0's auc: 0.672316
[330]	valid_0's auc: 0.672361
[340]	valid_0's auc: 0.672353
[350]	valid_0's auc: 0.672414
[360]	valid_0's auc: 0.672359
[370]	valid_0's auc: 0.672379
[380]	valid_0's auc: 0.672409
[390]	valid_0's auc: 0.672431
[400]	valid_0's auc: 0.67245
[410]	valid_0's auc: 0.672459
[420]	valid_0's auc: 0.672468
[430]	valid_0's auc: 0.672464
[440]	valid_0's auc: 0.672439
[450]	valid_0's auc: 0.672412
[460]	valid_0's auc: 0.672514
[470]	valid_0's auc: 0.672483
[480]	valid_0's auc: 0.672517
[490]	valid_0's auc: 0.672493
[500]	valid_0's auc: 0.672483
[510]	valid_0's auc: 0.672469
[520]	valid_0's auc: 0.672447
[530]	valid_0's auc: 0.672451
Early stopping, best iteration is:
[482]	valid_0's auc: 0.672521

complete on: artist_name

[timer]: complete in 11m 28s

Process finished with exit code 0
'''
