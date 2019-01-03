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
# print(type(df.head()))
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

print()
print()
print('After selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()
print()
length = len(df)
train_size = 0.75
train_set = df.head(int(length*train_size))
val_set = df.drop(train_set.index)
# print(train_set.head(100))
# print(len(train_set))
# print(len(val_set))
del df

X_tr = train_set.drop(['target'], axis=1)
Y_tr = train_set['target'].values

X_val = val_set.drop(['target'], axis=1)
Y_val = val_set['target'].values

del train_set, val_set
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
# X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train,
#                                             train_size=0.000001,
#                                             shuffle=True,
#                                             random_state=555,
#                                             )
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
# del X_train, Y_train

train_set = lgb.Dataset(X_tr, Y_tr)
val_set = lgb.Dataset(X_val, Y_val)
del X_tr, Y_tr, X_val, Y_val

# train_set = lgb.Dataset(X_train, Y_train,
#                         categorical_feature=[0, 1],
#                         )
print('Training...')
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
          'min_data': 1,
          'min_data_in_bin': 1,
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
                  # early_stopping_rounds=50,
                  valid_sets=val_set,
                  verbose_eval=10,
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


'''
/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/training_V1101.py"
What we got:
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
number of columns: 47
Train test and validation sets


After selection:
msno       category
song_id    category
target        uint8
dtype: object
number of columns: 3


train size: 5533063 number of 1: 2963461 number of 0: 2569602
train: 1 in all: 0.535591407508 0 in all: 0.464408592492 1/0: 1.15327626613
val size: 1844355 number of 1: 751195 number of 0: 1093160
val: 1 in all: 0.407294148903 0 in all: 0.592705851097 1/0: 0.687177540342


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
[10]	valid_0's auc: 0.548976
[20]	valid_0's auc: 0.558078
[30]	valid_0's auc: 0.565018
[40]	valid_0's auc: 0.569481
[50]	valid_0's auc: 0.573896
[60]	valid_0's auc: 0.578518
[70]	valid_0's auc: 0.581707
[80]	valid_0's auc: 0.584726
[90]	valid_0's auc: 0.587213
[100]	valid_0's auc: 0.589231
[110]	valid_0's auc: 0.591398
[120]	valid_0's auc: 0.593473
[130]	valid_0's auc: 0.595739
[140]	valid_0's auc: 0.597507
[150]	valid_0's auc: 0.598914
[160]	valid_0's auc: 0.600184
[170]	valid_0's auc: 0.601351
[180]	valid_0's auc: 0.602088
[190]	valid_0's auc: 0.60301
[200]	valid_0's auc: 0.603949
[210]	valid_0's auc: 0.604911
[220]	valid_0's auc: 0.605802
[230]	valid_0's auc: 0.606791
[240]	valid_0's auc: 0.607582
[250]	valid_0's auc: 0.608433
[260]	valid_0's auc: 0.6091
[270]	valid_0's auc: 0.609826
[280]	valid_0's auc: 0.610362
[290]	valid_0's auc: 0.61093
[300]	valid_0's auc: 0.611465
model saved as:  ../saves/ model_V1001

[timer]: complete in 3m 38s

Process finished with exit code 0
'''