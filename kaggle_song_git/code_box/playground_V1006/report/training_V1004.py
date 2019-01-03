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
print(df.dtypes)
# df = df.drop(['song_count', 'liked_song_count',
#               'disliked_song_count', 'artist_count',
#               'liked_artist_count', 'disliked_artist_count'], axis=1)
# df = df[['mn', 'sn', 'target']]
df = df[['msno', 'song_id', 'target']]
# df = df[['city', 'age', 'target']]
print("Train test and validation sets")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')
        # test[col] = test[col].astype('category')

print(df.dtypes)
# train = df.sample(frac=0.6, random_state=5)
# val = df.drop(train.index)
# print('df len: ', len(df))
# del df
X_train = df.drop(['target'], axis=1)
Y_train = df['target'].values
# X_val = val.drop(['target'], axis=1)
# Y_val = val['target'].values

# print('train len:', len(train))
# print('val len: ', len(val))
del df
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train,
                                            train_size=0.65,
                                            shuffle=True,
                                            random_state=222,
                                            )

del X_train, Y_train

train_set = lgb.Dataset(X_tr, Y_tr)
val_set = lgb.Dataset(X_val, Y_val)


# train_set = lgb.Dataset(X_train, Y_train,
#                         categorical_feature=[0, 1],
#                         )
print('Processed data...')
params = {'objective': 'binary',
          'metric': 'auc',
          # 'metric': 'binary_logloss',
          'boosting': 'gbdt',
          'learning_rate': 0.1,
          # 'verbosity': -1,
          'verbose': -1,
          # 'record': True,
          'num_leaves': 100,

          'bagging_fraction': 0.8,
          'bagging_freq': 2,
          'bagging_seed': 1,
          'feature_fraction': 0.8,
          'feature_fraction_seed': 1,
          'max_bin': 63,
          'max_depth': 10,
          # 'min_data': 500,
          'min_hessian': 0.05,
          # 'num_rounds': 500,
          # "min_data_in_leaf": 1,

          # 'lambda_l2': 0.5,
          # 'device': 'gpu',
          # 'gpu_platform_id': 0,
          # 'gpu_device_id': 0,
          # 'sparse_threshold': 1.0,
          # 'categorical_feature': (0,1,2,3),
         }
model = lgb.train(params,
                  train_set,
                  num_boost_round=300,
                  early_stopping_rounds=50,
                  valid_sets=val_set,
                  verbose_eval=10,
                  # nfold=5,
                  )
pickle.dump(model, open(save_dir+'model_V1001.save', "wb"))
# print(model)
# print(type(model))
# print(len(model))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/gbdt_random_V1001.py"
msno                        object
song_id                     object
source_system_tab           object
source_screen_name          object
source_type                 object
target                       uint8
city                         uint8
registered_via               uint8
mn                           int64
age                           int8
age_range                     int8
membership_days              int64
membership_days_range         int8
registration_year            int64
registration_month           int64
registration_date            int64
expiration_year              int64
expiration_month             int64
expiration_date              int64
sex                           int8
sex_guess                     int8
song_length                  int64
genre_ids                   object
artist_name                 object
composer                    object
lyricist                    object
language                      int8
sn                           int64
lyricists_count               int8
composer_count                int8
genre_ids_count               int8
length_range                 int64
length_bin_range             int64
length_chunk_range           int64
song_year                    int64
song_year_bin_range          int64
song_year_chunk_range        int64
song_country                object
rc                          object
artist_composer               int8
artist_composer_lyricist      int8
song_count                   int64
liked_song_count             int64
disliked_song_count          int64
artist_count                 int64
liked_artist_count           int64
disliked_artist_count        int64
dtype: object
Train test and validation sets
msno       category
song_id    category
target        uint8
dtype: object
/usr/local/lib/python3.5/dist-packages/sklearn/model_selection/_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
Processed data...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.587576
[20]	valid_0's auc: 0.607598
[30]	valid_0's auc: 0.622732
[40]	valid_0's auc: 0.632464
[50]	valid_0's auc: 0.641024
[60]	valid_0's auc: 0.648649
[70]	valid_0's auc: 0.65569
[80]	valid_0's auc: 0.660779
[90]	valid_0's auc: 0.664983
[100]	valid_0's auc: 0.668785
[110]	valid_0's auc: 0.672148
[120]	valid_0's auc: 0.675173
[130]	valid_0's auc: 0.677954
[140]	valid_0's auc: 0.680177
[150]	valid_0's auc: 0.682281
[160]	valid_0's auc: 0.684057
[170]	valid_0's auc: 0.686143
[180]	valid_0's auc: 0.687661
[190]	valid_0's auc: 0.689338
[200]	valid_0's auc: 0.690897
[210]	valid_0's auc: 0.692322
[220]	valid_0's auc: 0.69359
[230]	valid_0's auc: 0.694746
[240]	valid_0's auc: 0.695734
[250]	valid_0's auc: 0.696681
[260]	valid_0's auc: 0.697614
[270]	valid_0's auc: 0.698497
[280]	valid_0's auc: 0.699294
[290]	valid_0's auc: 0.700132
[300]	valid_0's auc: 0.700882

[timer]: complete in 4m 1s

Process finished with exit code 0
'''

# 0.61189
