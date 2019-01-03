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
train_size = 0.76
train_set = df.head(int(length*train_size))
val_set = df.drop(train_set.index)
# print(train_set.head(100))
# print(len(train_set))
# print(len(val_set))
del df
train_set = train_set.sample(frac=1)
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
                  num_boost_round=50000,
                  early_stopping_rounds=50,
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


'''/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/training_V1002.py"
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


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.549549
[20]	valid_0's auc: 0.559352
[30]	valid_0's auc: 0.56549
[40]	valid_0's auc: 0.569764
[50]	valid_0's auc: 0.573978
[60]	valid_0's auc: 0.578411
[70]	valid_0's auc: 0.582373
[80]	valid_0's auc: 0.584851
[90]	valid_0's auc: 0.587744
[100]	valid_0's auc: 0.589397
[110]	valid_0's auc: 0.59177
[120]	valid_0's auc: 0.593779
[130]	valid_0's auc: 0.595588
[140]	valid_0's auc: 0.596917
[150]	valid_0's auc: 0.598737
[160]	valid_0's auc: 0.599788
[170]	valid_0's auc: 0.600757
[180]	valid_0's auc: 0.601942
[190]	valid_0's auc: 0.602763
[200]	valid_0's auc: 0.604006
[210]	valid_0's auc: 0.604761
[220]	valid_0's auc: 0.605546
[230]	valid_0's auc: 0.606236
[240]	valid_0's auc: 0.606899
[250]	valid_0's auc: 0.607776
[260]	valid_0's auc: 0.608468
[270]	valid_0's auc: 0.609362
[280]	valid_0's auc: 0.610081
[290]	valid_0's auc: 0.610414
[300]	valid_0's auc: 0.610985
[310]	valid_0's auc: 0.611558
[320]	valid_0's auc: 0.612207
[330]	valid_0's auc: 0.612628
[340]	valid_0's auc: 0.613166
[350]	valid_0's auc: 0.613576
[360]	valid_0's auc: 0.61407
[370]	valid_0's auc: 0.614456
[380]	valid_0's auc: 0.614789
[390]	valid_0's auc: 0.615123
[400]	valid_0's auc: 0.615362
[410]	valid_0's auc: 0.615573
[420]	valid_0's auc: 0.615808
[430]	valid_0's auc: 0.616259
[440]	valid_0's auc: 0.616664
[450]	valid_0's auc: 0.616861
[460]	valid_0's auc: 0.617084
[470]	valid_0's auc: 0.617368
[480]	valid_0's auc: 0.617531
[490]	valid_0's auc: 0.617762
[500]	valid_0's auc: 0.617915
[510]	valid_0's auc: 0.618077
[520]	valid_0's auc: 0.618253
[530]	valid_0's auc: 0.618489
[540]	valid_0's auc: 0.618634
[550]	valid_0's auc: 0.618789
[560]	valid_0's auc: 0.61905
[570]	valid_0's auc: 0.619245
[580]	valid_0's auc: 0.619373
[590]	valid_0's auc: 0.61945
[600]	valid_0's auc: 0.619646
[610]	valid_0's auc: 0.619695
[620]	valid_0's auc: 0.619932
[630]	valid_0's auc: 0.620111
[640]	valid_0's auc: 0.620334
[650]	valid_0's auc: 0.620415
[660]	valid_0's auc: 0.620519
[670]	valid_0's auc: 0.620577
[680]	valid_0's auc: 0.620717
[690]	valid_0's auc: 0.620769
[700]	valid_0's auc: 0.620844
[710]	valid_0's auc: 0.620906
[720]	valid_0's auc: 0.621108
[730]	valid_0's auc: 0.621198
[740]	valid_0's auc: 0.621222
[750]	valid_0's auc: 0.621233
[760]	valid_0's auc: 0.621302
[770]	valid_0's auc: 0.621422
[780]	valid_0's auc: 0.621502
[790]	valid_0's auc: 0.621553
[800]	valid_0's auc: 0.621731
[810]	valid_0's auc: 0.621758
[820]	valid_0's auc: 0.621778
[830]	valid_0's auc: 0.621808
[840]	valid_0's auc: 0.62188
[850]	valid_0's auc: 0.62191
[860]	valid_0's auc: 0.621993
[870]	valid_0's auc: 0.621998
[880]	valid_0's auc: 0.622024
[890]	valid_0's auc: 0.622136
[900]	valid_0's auc: 0.622241
[910]	valid_0's auc: 0.622295
[920]	valid_0's auc: 0.622338
[930]	valid_0's auc: 0.622416
[940]	valid_0's auc: 0.622428
[950]	valid_0's auc: 0.622472
[960]	valid_0's auc: 0.622483
[970]	valid_0's auc: 0.622527
[980]	valid_0's auc: 0.622578
[990]	valid_0's auc: 0.622637
[1000]	valid_0's auc: 0.622699
[1010]	valid_0's auc: 0.622699
[1020]	valid_0's auc: 0.622744
[1030]	valid_0's auc: 0.622755
[1040]	valid_0's auc: 0.622845
[1050]	valid_0's auc: 0.622872
[1060]	valid_0's auc: 0.622881
[1070]	valid_0's auc: 0.62293
[1080]	valid_0's auc: 0.622946
[1090]	valid_0's auc: 0.622964
[1100]	valid_0's auc: 0.623022
[1110]	valid_0's auc: 0.623047
[1120]	valid_0's auc: 0.623055
[1130]	valid_0's auc: 0.623118
[1140]	valid_0's auc: 0.623126
[1150]	valid_0's auc: 0.623102
[1160]	valid_0's auc: 0.62311
[1170]	valid_0's auc: 0.623073
[1180]	valid_0's auc: 0.623079
[1190]	valid_0's auc: 0.623076
Early stopping, best iteration is:
[1142]	valid_0's auc: 0.623136
model saved as:  ../saves/ model_V1001

[timer]: complete in 13m 11s

Process finished with exit code 0
'''