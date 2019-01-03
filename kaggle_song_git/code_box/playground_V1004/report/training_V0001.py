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
# df = df.drop(['song_count', 'liked_song_count',
#               'disliked_song_count', 'artist_count',
#               'liked_artist_count', 'disliked_artist_count'], axis=1)
# df = df[['mn', 'sn', 'target']]
df = df[['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'liked_song_count'
         ]]
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


X_tr = df.drop(['target'], axis=1)
Y_tr = df['target'].values


del df
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
# X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train,
#                                             train_size=0.000001,
#                                             shuffle=True,
#                                             random_state=555,
#                                             )
# t = len(Y_tr)
# t1 = sum(Y_tr)
# t0 = t - t1
# print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
# print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
# t = len(Y_val)
# t1 = sum(Y_val)
# t0 = t - t1
# print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
# print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
# print()
# print()
# del X_train, Y_train

train_set = lgb.Dataset(X_tr, Y_tr)
# val_set = lgb.Dataset(X_val, Y_val)
del X_tr, Y_tr, #X_val, Y_val

# train_set = lgb.Dataset(X_train, Y_train,
#                         categorical_feature=[0, 1],
#                         )
print('Training...')
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
          # # 'num_rounds': 500,
          # # "min_data_in_leaf": 1,
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
                  num_boost_round=80,
                  # early_stopping_rounds=50,
                  # valid_sets=val_set,
                  # verbose_eval=10,
                  # nfold=5,
                  )
model_name = 'model_V1001'
pickle.dump(model, open(save_dir+model_name+'.save', "wb"))
print('model saved as: ', save_dir, model_name)
# print(model)
# print(type(model))
# print(len(model))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

#0.63451

