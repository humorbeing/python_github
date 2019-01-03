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
         # 'language',
         'artist_name',
         'composer',
         'lyricist',
         'song_year',
         'top1_in_song',
         'top2_in_song',
         'top3_in_song',
         'language',
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
lr_s = [0.02, 0.01, 0.01, 0.02, 0.02]
nl_s = [ 511, 1023, 1023,  511,  511]
md_s = [  -1,   -1,   11,   -1,   -1]
l2_s = [   0,    0,    0,  0.2,    0]
l1_s = [   0,    0,    0,    0,  0.2]
mb_s = [ 255,  511,  511,  255,  255]
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

train_set = lgb.Dataset(X_tr, Y_tr)
val_set = lgb.Dataset(X_val, Y_val)
del X_tr, Y_tr, X_val, Y_val

print('Training...')
print()

for i in range(5):
    boosting = b_s[0]
    learning_rate = lr_s[i]
    num_leaves = nl_s[i]
    max_depth = md_s[i]
    lambda_l1 = l1_s[i]
    lambda_l2 = l2_s[i]
    max_bin = mb_s[i]
    train_set.max_bin = max_bin
    val_set.max_bin = max_bin
    params = {
              'boosting': boosting,
              'learning_rate': learning_rate,
              'num_leaves': num_leaves,
              'bagging_fraction': bagging_fraction,
              'bagging_freq': bagging_freq,
              'bagging_seed': bagging_seed,
              'feature_fraction': feature_fraction,
              'feature_fraction_seed': feature_fraction_seed,
              'max_bin': max_bin,
              'max_depth': max_depth,
              'lambda_l2': lambda_l2,
              'lambda_l1': lambda_l1
              }
    print()
    print('>'*50)
    print('------------Parameters-----------')
    print()
    for dd in params:
        print(dd.ljust(20), ':', params[dd])
    print()
    params['metric'] = 'auc'
    # params['max_bin'] = 255
    params['verbose'] = -1
    params['objective'] = 'binary'

    model = lgb.train(params,
                      train_set,
                      num_boost_round=50000,
                      early_stopping_rounds=200,
                      valid_sets=val_set,
                      verbose_eval=10,
                      )

    print('best score:', model.best_score['valid_0']['auc'])

    print('best iteration:', model.best_iteration)
    print()
    print('<'*50)
    print()
    time_elapsed = time.time() - since
    print('[timer]: complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    since = time.time()


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1001.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
expiration_month      category
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
number of columns: 19


This rounds guests:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
artist_name           category
composer              category
lyricist              category
song_year             category
top1_in_song          category
top2_in_song          category
top3_in_song          category
language              category
dtype: object
number of columns: 14

Training...


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 511
boosting             : gbdt
feature_fraction     : 0.9
bagging_fraction     : 0.9
feature_fraction_seed : 2
max_depth            : -1
learning_rate        : 0.02
lambda_l2            : 0
bagging_seed         : 2
lambda_l1            : 0
bagging_freq         : 2
max_bin              : 255

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:648: LGBMDeprecationWarning: The `max_bin` parameter is deprecated and will be removed in 2.0.12 version. Please use `params` to pass this parameter.
  'Please use `params` to pass this parameter.', LGBMDeprecationWarning)
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.657547
[20]	valid_0's auc: 0.659002
[30]	valid_0's auc: 0.660143
[40]	valid_0's auc: 0.661537
[50]	valid_0's auc: 0.662423
[60]	valid_0's auc: 0.663177
[70]	valid_0's auc: 0.66459
[80]	valid_0's auc: 0.66567
[90]	valid_0's auc: 0.666618
[100]	valid_0's auc: 0.667665
[110]	valid_0's auc: 0.668532
[120]	valid_0's auc: 0.669309
[130]	valid_0's auc: 0.670284
[140]	valid_0's auc: 0.671043
[150]	valid_0's auc: 0.671644
[160]	valid_0's auc: 0.672017
[170]	valid_0's auc: 0.672572
[180]	valid_0's auc: 0.673088
[190]	valid_0's auc: 0.673529
[200]	valid_0's auc: 0.673936
[210]	valid_0's auc: 0.674256
[220]	valid_0's auc: 0.674615
[230]	valid_0's auc: 0.674758
[240]	valid_0's auc: 0.675149
[250]	valid_0's auc: 0.67542
[260]	valid_0's auc: 0.675664
[270]	valid_0's auc: 0.675839
[280]	valid_0's auc: 0.675948
[290]	valid_0's auc: 0.676059
[300]	valid_0's auc: 0.676161
[310]	valid_0's auc: 0.676289
[320]	valid_0's auc: 0.676343
[330]	valid_0's auc: 0.676455
[340]	valid_0's auc: 0.676569
[350]	valid_0's auc: 0.676632
[360]	valid_0's auc: 0.676637
[370]	valid_0's auc: 0.676695
[380]	valid_0's auc: 0.676715
[390]	valid_0's auc: 0.676813
[400]	valid_0's auc: 0.676904
[410]	valid_0's auc: 0.676916
[420]	valid_0's auc: 0.676954
[430]	valid_0's auc: 0.676951
[440]	valid_0's auc: 0.676974
[450]	valid_0's auc: 0.676962
[460]	valid_0's auc: 0.677019
[470]	valid_0's auc: 0.677015
[480]	valid_0's auc: 0.677051
[490]	valid_0's auc: 0.677099
[500]	valid_0's auc: 0.677094
[510]	valid_0's auc: 0.677114
[520]	valid_0's auc: 0.677099
[530]	valid_0's auc: 0.677104
[540]	valid_0's auc: 0.677092
[550]	valid_0's auc: 0.677113
[560]	valid_0's auc: 0.677113
[570]	valid_0's auc: 0.677137
[580]	valid_0's auc: 0.677155
[590]	valid_0's auc: 0.677149
[600]	valid_0's auc: 0.677129
[610]	valid_0's auc: 0.677113
[620]	valid_0's auc: 0.677103
[630]	valid_0's auc: 0.677116
[640]	valid_0's auc: 0.677119
[650]	valid_0's auc: 0.677146
[660]	valid_0's auc: 0.677145
[670]	valid_0's auc: 0.677147
[680]	valid_0's auc: 0.677138
[690]	valid_0's auc: 0.677125
[700]	valid_0's auc: 0.677127
[710]	valid_0's auc: 0.677131
[720]	valid_0's auc: 0.677141
[730]	valid_0's auc: 0.677126
[740]	valid_0's auc: 0.677131
[750]	valid_0's auc: 0.677125
[760]	valid_0's auc: 0.677124
[770]	valid_0's auc: 0.677128
[780]	valid_0's auc: 0.677139
Early stopping, best iteration is:
[584]	valid_0's auc: 0.677164
best score: 0.677163500998
best iteration: 584

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 80m 35s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 1023
boosting             : gbdt
feature_fraction     : 0.9
bagging_fraction     : 0.9
feature_fraction_seed : 2
max_depth            : -1
learning_rate        : 0.01
lambda_l2            : 0
bagging_seed         : 2
lambda_l1            : 0
bagging_freq         : 2
max_bin              : 511

Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.663237
[20]	valid_0's auc: 0.663409
[30]	valid_0's auc: 0.664132
[40]	valid_0's auc: 0.664626
[50]	valid_0's auc: 0.665184
[60]	valid_0's auc: 0.665477
[70]	valid_0's auc: 0.666027
[80]	valid_0's auc: 0.666404
[90]	valid_0's auc: 0.666624
[100]	valid_0's auc: 0.667307
[110]	valid_0's auc: 0.667541
[120]	valid_0's auc: 0.667883
[130]	valid_0's auc: 0.668325
[140]	valid_0's auc: 0.66869
[150]	valid_0's auc: 0.669082
[160]	valid_0's auc: 0.669372
[170]	valid_0's auc: 0.66977
[180]	valid_0's auc: 0.670262
[190]	valid_0's auc: 0.670606
[200]	valid_0's auc: 0.670812
[210]	valid_0's auc: 0.671143
[220]	valid_0's auc: 0.671527
[230]	valid_0's auc: 0.671764
[240]	valid_0's auc: 0.672136
[250]	valid_0's auc: 0.672483
[260]	valid_0's auc: 0.672706
[270]	valid_0's auc: 0.672944
[280]	valid_0's auc: 0.673165
[290]	valid_0's auc: 0.673335
[300]	valid_0's auc: 0.673516
[310]	valid_0's auc: 0.67364
[320]	valid_0's auc: 0.673827
[330]	valid_0's auc: 0.67392
[340]	valid_0's auc: 0.674087
[350]	valid_0's auc: 0.674209
[360]	valid_0's auc: 0.674287
[370]	valid_0's auc: 0.674403
[380]	valid_0's auc: 0.674501
[390]	valid_0's auc: 0.674628
[400]	valid_0's auc: 0.67474
[410]	valid_0's auc: 0.674829
[420]	valid_0's auc: 0.674912
[430]	valid_0's auc: 0.675043
[440]	valid_0's auc: 0.675138
[450]	valid_0's auc: 0.675221
[460]	valid_0's auc: 0.675287
[470]	valid_0's auc: 0.67536
[480]	valid_0's auc: 0.675414
[490]	valid_0's auc: 0.675476
[500]	valid_0's auc: 0.675493
[510]	valid_0's auc: 0.675584
[520]	valid_0's auc: 0.675672
[530]	valid_0's auc: 0.675729
[540]	valid_0's auc: 0.675784
[550]	valid_0's auc: 0.675824
[560]	valid_0's auc: 0.675866
[570]	valid_0's auc: 0.67595
[580]	valid_0's auc: 0.676005
[590]	valid_0's auc: 0.676033
[600]	valid_0's auc: 0.676072
[610]	valid_0's auc: 0.676092
[620]	valid_0's auc: 0.67611
[630]	valid_0's auc: 0.676159
[640]	valid_0's auc: 0.676191
[650]	valid_0's auc: 0.676207
[660]	valid_0's auc: 0.676244
[670]	valid_0's auc: 0.676253
[680]	valid_0's auc: 0.676267
[690]	valid_0's auc: 0.676277
[700]	valid_0's auc: 0.676291
[710]	valid_0's auc: 0.676297
[720]	valid_0's auc: 0.676313
[730]	valid_0's auc: 0.676333
[740]	valid_0's auc: 0.676342
[750]	valid_0's auc: 0.676363
[760]	valid_0's auc: 0.676358
[770]	valid_0's auc: 0.676378
[780]	valid_0's auc: 0.6764
[790]	valid_0's auc: 0.676412
[800]	valid_0's auc: 0.676399
[810]	valid_0's auc: 0.676418
[820]	valid_0's auc: 0.676424
[830]	valid_0's auc: 0.676438
[840]	valid_0's auc: 0.676478
[850]	valid_0's auc: 0.676468
[860]	valid_0's auc: 0.676463
[870]	valid_0's auc: 0.676473
[880]	valid_0's auc: 0.676455
[890]	valid_0's auc: 0.676457
[900]	valid_0's auc: 0.676472
[910]	valid_0's auc: 0.676487
[920]	valid_0's auc: 0.676507
[930]	valid_0's auc: 0.676518
[940]	valid_0's auc: 0.67653
[950]	valid_0's auc: 0.676541
[960]	valid_0's auc: 0.676545
[970]	valid_0's auc: 0.676537
[980]	valid_0's auc: 0.676537
[990]	valid_0's auc: 0.676538
[1000]	valid_0's auc: 0.676536
[1010]	valid_0's auc: 0.676533
[1020]	valid_0's auc: 0.67653
[1030]	valid_0's auc: 0.676504
[1040]	valid_0's auc: 0.676525
[1050]	valid_0's auc: 0.676514
[1060]	valid_0's auc: 0.676538
[1070]	valid_0's auc: 0.676555
[1080]	valid_0's auc: 0.676548
[1090]	valid_0's auc: 0.676536
[1100]	valid_0's auc: 0.676539
[1110]	valid_0's auc: 0.676549
[1120]	valid_0's auc: 0.676538
[1130]	valid_0's auc: 0.676528
[1140]	valid_0's auc: 0.676526
[1150]	valid_0's auc: 0.676527
[1160]	valid_0's auc: 0.67651
[1170]	valid_0's auc: 0.676514
[1180]	valid_0's auc: 0.676519
[1190]	valid_0's auc: 0.676519
[1200]	valid_0's auc: 0.676529
[1210]	valid_0's auc: 0.676524
[1220]	valid_0's auc: 0.676524
[1230]	valid_0's auc: 0.67652
[1240]	valid_0's auc: 0.676519
[1250]	valid_0's auc: 0.676528
[1260]	valid_0's auc: 0.676529
[1270]	valid_0's auc: 0.676528
Early stopping, best iteration is:
[1073]	valid_0's auc: 0.676559
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1001.py", line 135, in <module>
    verbose_eval=10,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1679, in _save_model_to_string
    ptr_string_buffer))
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 48, in _safe_call
    raise LightGBMError(_LIB.LGBM_GetLastError())
lightgbm.basic.LightGBMError: b'std::bad_alloc'

Process finished with exit code 1
'''