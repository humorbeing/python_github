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
         'language',
         # 'artist_name',
         ]

boosting = 'gbdt'
learning_rate = 0.1
num_leaves = 100
# bagging_fraction = 0
# bagging_freq = 0
# bagging_seed = 0
# feature_fraction = 0
# feature_fraction_seed = 0
max_depth = -1
lambda_l2 = 0
lambda_l1 = 0

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
                  'boosting': boosting,
                  'learning_rate': learning_rate,
                  'verbose': -1,
                  'num_leaves': num_leaves,

                  # 'bagging_fraction': bagging_fraction,
                  # 'bagging_freq': bagging_freq,
                  # 'bagging_seed': bagging_seed,
                  # 'feature_fraction': feature_fraction,
                  # 'feature_fraction_seed': feature_fraction_seed,
                  'max_bin': 255,
                  'max_depth': max_depth,
                  # 'min_data': 500,
                  # 'min_hessian': 0.05,
                  # 'num_rounds': 500,
                  # "min_data_in_leaf": 1,
                  # 'min_data': 1,
                  # 'min_data_in_bin': 1,
                  'lambda_l2': lambda_l2,
                  'lambda_l1': lambda_l1

                  }
        model = lgb.train(params,
                          train_set,
                          num_boost_round=2,
                          early_stopping_rounds=1,
                          valid_sets=val_set,
                          verbose_eval=1,
                          )
        print(model.best_score['valid_0']['auc'])
        # print(type(model.best_iteration))
        # print(model.best_iteration)
        # print(model.attr('learning_rate'))
        # li = model.eval_valid()
        # print('len list:', len(li))
        # print('max list:', max(li))
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


