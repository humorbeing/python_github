import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle


since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_fillna3'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)

del dt

df = df.drop(['song_count', 'liked_song_count',
              'disliked_song_count', 'artist_count',
              'liked_artist_count', 'disliked_artist_count'], axis=1)

print("Train test and validation sets")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')
        # test[col] = test[col].astype('category')

print(df.dtypes)
train = df.sample(frac=0.95, random_state=5)
val = df.drop(train.index)
# print('df len: ', len(df))
del df
X_train = train.drop(['target'], axis=1)
y_train = train['target'].values
X_val = val.drop(['target'], axis=1)
Y_val = val['target'].values

# print('train len:', len(train))
# print('val len: ', len(val))
del train, val
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values


# del train, test; gc.collect();

train_set = lgb.Dataset(X_train, y_train)
val_set = lgb.Dataset(X_train, y_train)

print('Processed data...')
params = {'objective': 'binary',
          'metric': 'binary_logloss',
          'boosting': 'gbdt',
          'learning_rate': 0.1,
          'verbose': 0,
          'num_leaves': 150,
          'bagging_fraction': 0.95,
          'bagging_freq': 1,
          'bagging_seed': 1,
          'feature_fraction': 0.9,
          'feature_fraction_seed': 1,
          'max_bin': 256,
          'max_depth': 11,
          'num_rounds': 300,
          'metric': 'auc',
          # 'device': 'gpu',
          # 'gpu_platform_id': 0,
          # 'gpu_device_id': 0,
         }
model = lgb.train(params, train_set=train_set,
                  valid_sets=val_set, verbose_eval=5)
pickle.dump(model, open(save_dir+'model_V1002.save', "wb"))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


# 67