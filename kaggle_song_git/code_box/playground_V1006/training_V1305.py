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
         'artist_name',
         ]

boosting = 'gbdt'
learning_rate = 0.1
num_leaves = 100
bagging_fraction = 0.9
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.9
feature_fraction_seed = 2
max_depth = -1
lambda_l2 = 0
lambda_l1 = 0

b_s = ['gbdt', 'rf', 'dart', 'goss']
lr_s = [0.3, 0.1, 0.05, 0.01, 0.005, 0.001]
# lr_s = [0.3, 10, 1000, 1000000, 100000000000, 0.001]
# lr_s = [20, 19, 18, 17, 16, 15]
nl_s = [100, 150, 200, 50, 75, 25]
md_s = [-1, 10, 13, 20, 25]


df = df[fixed]

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

train_set = lgb.Dataset(X_tr, Y_tr, free_raw_data=False)
val_set = lgb.Dataset(X_val, Y_val, free_raw_data=False)
del X_tr, Y_tr, X_val, Y_val

print('Training...')
print()

runs = 0


def ccc(x):
    # print(x)
    # print(lr_s[runs])
    return lr_s[runs]


while True:
    boosting = b_s[0]
    # learning_rate = lr_s[np.random.randint(0, 6)]
    num_leaves = nl_s[np.random.randint(0, 6)]
    max_depth = md_s[np.random.randint(0, 5)]
    # lambda_l1 = np.random.random()
    # lambda_l2 = np.random.random()
    params = {
              # 'objective': 'binary',
              # 'metric': 'auc',
              'boosting': boosting,
              # 'learning_rate': learning_rate,
              # 'verbose': -1,
              'num_leaves': num_leaves,

              # 'bagging_fraction': bagging_fraction,
              # 'bagging_freq': bagging_freq,
              # 'bagging_seed': bagging_seed,
              # 'feature_fraction': feature_fraction,
              # 'feature_fraction_seed': feature_fraction_seed,
              # 'max_bin': 255,
              'max_depth': max_depth,
              # 'min_data': 500,
              # 'min_hessian': 0.05,
              # 'num_rounds': 500,
              # "min_data_in_leaf": 1,
              # 'min_data': 1,
              # 'min_data_in_bin': 1,
              # 'lambda_l2': lambda_l2,
              # 'lambda_l1': lambda_l1

              }
    print()
    print('>'*50)
    print('------------Parameters-----------')
    print()
    for dd in params:
        print(dd.ljust(20), ':', params[dd])
    print()
    params['metric'] = 'auc'
    params['max_bin'] = 255
    params['verbose'] = -1
    params['objective'] = 'binary'
    params['learning_rate'] = lr_s[0]
    print('learning rate:',lr_s[0])
    model = lgb.train(params,
                      train_set,
                      num_boost_round=50000,
                      early_stopping_rounds=50,
                      # learning_rates=ccc,
                      valid_sets=val_set,
                      verbose_eval=10,
                      )
    for i in range(5):
        runs = i+1
        # params['learning_rate'] = lr_s[i+1]
        print('learning rate:', lr_s[i+1])
        # model.set
        model = lgb.train(params,
                          train_set,
                          num_boost_round=50000,
                          init_model=model,
                          learning_rates=ccc,
                          early_stopping_rounds=50,
                          valid_sets=val_set,
                          verbose_eval=10,
                          )
    print(model.best_score['valid_0']['auc'])
    # print(type(model.best_iteration))
    # print(model.best_iteration)
    print()
    print('<'*50)
# li = model.eval_valid()
# print('len list:', len(li))
# print('max list:', max(li))
del train_set, val_set
print()
# print('complete on:', w)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


