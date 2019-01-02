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
    'num_leaves': 100,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 255,
    'max_depth': -1,
}
df = df[['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'fake_song_count',
         'fake_artist_count',
         'fake_member_count',
         'fake_language_count',
         ]]
# fixed = ['msno',
#          'song_id',
#          'target',
#          'source_system_tab',
#          'source_screen_name',
#          'source_type',
#          'language',
#          'artist_name',
#          ]
fixed = ['target']

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
        df = df[['msno',
                 'song_id',
                 'target',
                 'source_system_tab',
                 'source_screen_name',
                 'source_type',
                 'language',
                 'artist_name',
                 'fake_song_count',
                 'fake_artist_count',
                 'fake_member_count',
                 'fake_language_count',
                 ]]

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_trainer_V1003.py
What we got:
msno                     object
song_id                  object
source_system_tab        object
source_screen_name       object
source_type              object
target                    uint8
fake_member_count         int64
member_count              int64
artist_name              object
language               category
song_count                int64
artist_count              int64
language_count            int64
fake_song_count           int64
fake_artist_count         int64
fake_language_count       int64
dtype: object
number of columns: 16
working on: msno


After selection:
target       uint8
msno      category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.569313
[20]	valid_0's auc: 0.581944
[30]	valid_0's auc: 0.588316
[40]	valid_0's auc: 0.590814
[50]	valid_0's auc: 0.591551
[60]	valid_0's auc: 0.592437
[70]	valid_0's auc: 0.592988
[80]	valid_0's auc: 0.592892
[90]	valid_0's auc: 0.592802
[100]	valid_0's auc: 0.592602
[110]	valid_0's auc: 0.592596
[120]	valid_0's auc: 0.592531
Early stopping, best iteration is:
[70]	valid_0's auc: 0.592988
best score: 0.592987509598
best iteration: 70
complete on: msno

working on: song_id


After selection:
target        uint8
song_id    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.563449
[20]	valid_0's auc: 0.565455
[30]	valid_0's auc: 0.56651
[40]	valid_0's auc: 0.566814
[50]	valid_0's auc: 0.566866
[60]	valid_0's auc: 0.566818
[70]	valid_0's auc: 0.566698
[80]	valid_0's auc: 0.566596
[90]	valid_0's auc: 0.566557
Early stopping, best iteration is:
[49]	valid_0's auc: 0.566875
best score: 0.566875302683
best iteration: 49
complete on: song_id

working on: source_system_tab


After selection:
target                  uint8
source_system_tab    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.585163
[20]	valid_0's auc: 0.58519
[30]	valid_0's auc: 0.58519
[40]	valid_0's auc: 0.585177
[50]	valid_0's auc: 0.585177
[60]	valid_0's auc: 0.585177
Early stopping, best iteration is:
[14]	valid_0's auc: 0.58519
best score: 0.585189781567
best iteration: 14
complete on: source_system_tab

working on: source_screen_name


After selection:
target                   uint8
source_screen_name    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.593275
[20]	valid_0's auc: 0.5934
[30]	valid_0's auc: 0.5934
[40]	valid_0's auc: 0.5934
[50]	valid_0's auc: 0.5934
[60]	valid_0's auc: 0.5934
[70]	valid_0's auc: 0.5934
Early stopping, best iteration is:
[20]	valid_0's auc: 0.5934
best score: 0.593399932324
best iteration: 20
complete on: source_screen_name

working on: source_type


After selection:
target            uint8
source_type    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.591709
[20]	valid_0's auc: 0.591709
[30]	valid_0's auc: 0.591709
[40]	valid_0's auc: 0.591709
[50]	valid_0's auc: 0.591709
Early stopping, best iteration is:
[1]	valid_0's auc: 0.591709
best score: 0.591708884203
best iteration: 1
complete on: source_type

working on: language


After selection:
target         uint8
language    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.524767
[20]	valid_0's auc: 0.524768
[30]	valid_0's auc: 0.524768
[40]	valid_0's auc: 0.524768
[50]	valid_0's auc: 0.524768
[60]	valid_0's auc: 0.524768
Early stopping, best iteration is:
[12]	valid_0's auc: 0.524768
best score: 0.52476758398
best iteration: 12
complete on: language

working on: artist_name


After selection:
target            uint8
artist_name    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.563353
[20]	valid_0's auc: 0.5633
[30]	valid_0's auc: 0.563256
[40]	valid_0's auc: 0.563246
[50]	valid_0's auc: 0.563238
[60]	valid_0's auc: 0.563206
Early stopping, best iteration is:
[10]	valid_0's auc: 0.563353
best score: 0.563353005738
best iteration: 10
complete on: artist_name

working on: fake_song_count


After selection:
target             uint8
fake_song_count    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.578838
[20]	valid_0's auc: 0.578855
[30]	valid_0's auc: 0.578852
[40]	valid_0's auc: 0.578852
[50]	valid_0's auc: 0.578851
Early stopping, best iteration is:
[4]	valid_0's auc: 0.578901
best score: 0.578901255897
best iteration: 4
complete on: fake_song_count

working on: fake_artist_count


After selection:
target               uint8
fake_artist_count    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.564416
[20]	valid_0's auc: 0.564393
[30]	valid_0's auc: 0.564393
[40]	valid_0's auc: 0.564391
[50]	valid_0's auc: 0.564387
Early stopping, best iteration is:
[5]	valid_0's auc: 0.564431
best score: 0.564431158044
best iteration: 5
complete on: fake_artist_count

working on: fake_member_count


After selection:
target               uint8
fake_member_count    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537412
[20]	valid_0's auc: 0.5374
[30]	valid_0's auc: 0.537398
[40]	valid_0's auc: 0.537399
[50]	valid_0's auc: 0.537406
[60]	valid_0's auc: 0.537406
Early stopping, best iteration is:
[10]	valid_0's auc: 0.537412
best score: 0.537411788204
best iteration: 10
complete on: fake_member_count

working on: fake_language_count


After selection:
target                 uint8
fake_language_count    int64
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.524767
[20]	valid_0's auc: 0.524768
[30]	valid_0's auc: 0.524768
[40]	valid_0's auc: 0.524768
[50]	valid_0's auc: 0.524768
[60]	valid_0's auc: 0.524768
Early stopping, best iteration is:
[12]	valid_0's auc: 0.524768
best score: 0.52476758398
best iteration: 12
complete on: fake_language_count


[timer]: complete in 16m 27s

Process finished with exit code 0
'''
