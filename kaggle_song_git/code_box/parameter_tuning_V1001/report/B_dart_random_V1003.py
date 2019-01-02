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
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
print()

# on = [
#     'msno',
#     'song_id',
#     'target',
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'language',
#     'artist_name',
#     'song_count',
#     'member_count',
#     'song_year',
# ]
# df = df[on]

num_boost_round = 5000
early_stopping_rounds = 50
verbose_eval = 10


bagging_fraction = 0.8
bagging_freq = 4
bagging_seed = 2
feature_fraction = 0.8
feature_fraction_seed = 2

b_s = ['gbdt', 'rf', 'dart', 'goss']
lr_s = [0.5, 0.3, 0.1, 0.3, 0.1]
nl_s = [511,1023, 511, 511, 511]
md_s = [ -1,  10,  11,  -1,  10]
l2_s = [  0,   0,   0,   0, 0.3]
l1_s = [  0,   0,   0, 0.3,   0]
xg_s = [False,True,True,False,False]
# mb_s = [511, 511, 255, 255, 127]


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
    inner_time = time.time()
    boosting = b_s[2]
    learning_rate = lr_s[i]
    num_leaves = nl_s[i]
    max_depth = md_s[i]
    lambda_l1 = l1_s[i]
    lambda_l2 = l2_s[i]
    xgboost_dart_mode = xg_s[i]
    # max_bin = mb_s[i]
    # train_set.max_bin = max_bin
    # val_set.max_bin = max_bin
    params = {
        'boosting': boosting,

        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,

        # 'max_bin': max_bin,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'xgboost_dart_mode': xgboost_dart_mode,

        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'bagging_seed': bagging_seed,
        'feature_fraction': feature_fraction,
        'feature_fraction_seed': feature_fraction_seed,
    }
    print()
    print('>'*50)
    print('------------Parameters-----------')
    print('round:', i)
    print()
    for dd in params:
        print(dd.ljust(20), ':', params[dd])
    print()
    params['metric'] = 'auc'
    params['verbose'] = -1
    params['objective'] = 'binary'

    model = lgb.train(params,
                      train_set,
                      num_boost_round=num_boost_round,
                      early_stopping_rounds=early_stopping_rounds,
                      valid_sets=val_set,
                      verbose_eval=verbose_eval,
                      )

    print('best score:', model.best_score['valid_0']['auc'])

    print('best iteration:', model.best_iteration)
    print()
    print('<'*50)

    print()
    inner_time_elapsed = time.time() - inner_time
    print('round:', i, 'complete in {:.0f}m {:.0f}s'.format(
        inner_time_elapsed // 60, inner_time_elapsed % 60))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
since = time.time()

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/B_dart_random_V1002.py
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
number of rows: 7377418
number of columns: 19


This rounds guests:
msno                  category
song_id               category
source_system_tab     category
source_screen_name    category
source_type           category
target                   uint8
expiration_month      category
artist_name           category
composer              category
lyricist              category
language              category
name                  category
song_year             category
song_country          category
rc                    category
isrc_rest             category
top1_in_song          category
top2_in_song          category
top3_in_song          category
dtype: object
number of columns: 19

Training...


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 0

max_depth            : 20
bagging_seed         : 2
bagging_freq         : 4
lambda_l2            : 0
lambda_l1            : 0
learning_rate        : 0.5
feature_fraction     : 0.8
feature_fraction_seed : 2
boosting             : dart
num_leaves           : 511
bagging_fraction     : 0.8
xgboost_dart_mode    : False

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.649665
[20]	valid_0's auc: 0.656075
[30]	valid_0's auc: 0.65888
[40]	valid_0's auc: 0.66058
[50]	valid_0's auc: 0.662306
[60]	valid_0's auc: 0.663897
[70]	valid_0's auc: 0.666949
[80]	valid_0's auc: 0.668215
[90]	valid_0's auc: 0.668642
[100]	valid_0's auc: 0.669043
[110]	valid_0's auc: 0.669449
[120]	valid_0's auc: 0.670157
[130]	valid_0's auc: 0.670261
[140]	valid_0's auc: 0.670513
[150]	valid_0's auc: 0.67081
[160]	valid_0's auc: 0.671461
[170]	valid_0's auc: 0.671954
[180]	valid_0's auc: 0.671877
[190]	valid_0's auc: 0.671837
[200]	valid_0's auc: 0.672124
[210]	valid_0's auc: 0.671891
[220]	valid_0's auc: 0.672285
[230]	valid_0's auc: 0.672453
[240]	valid_0's auc: 0.672028
[250]	valid_0's auc: 0.672063
[260]	valid_0's auc: 0.672249
[270]	valid_0's auc: 0.672462
[280]	valid_0's auc: 0.672628
[290]	valid_0's auc: 0.672813
[300]	valid_0's auc: 0.673104
[310]	valid_0's auc: 0.673546
[320]	valid_0's auc: 0.673437
[330]	valid_0's auc: 0.673446
[340]	valid_0's auc: 0.673474
[350]	valid_0's auc: 0.67347
Early stopping, best iteration is:
[309]	valid_0's auc: 0.67366
best score: 0.673660485376
best iteration: 309

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 0 complete in 56m 14s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 1

max_depth            : 20
bagging_seed         : 2
bagging_freq         : 4
lambda_l2            : 0
lambda_l1            : 0
learning_rate        : 0.3
feature_fraction     : 0.8
feature_fraction_seed : 2
boosting             : dart
num_leaves           : 511
bagging_fraction     : 0.8
xgboost_dart_mode    : True

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.649207
[20]	valid_0's auc: 0.655152
[30]	valid_0's auc: 0.658564
[40]	valid_0's auc: 0.660404
[50]	valid_0's auc: 0.661683
[60]	valid_0's auc: 0.663238
[70]	valid_0's auc: 0.664163
[80]	valid_0's auc: 0.664858
[90]	valid_0's auc: 0.665303
[100]	valid_0's auc: 0.666393
[110]	valid_0's auc: 0.66688
[120]	valid_0's auc: 0.667333
[130]	valid_0's auc: 0.667838
[140]	valid_0's auc: 0.668124
[150]	valid_0's auc: 0.667987
[160]	valid_0's auc: 0.668175
[170]	valid_0's auc: 0.668847
[180]	valid_0's auc: 0.669028
[190]	valid_0's auc: 0.669219
[200]	valid_0's auc: 0.669283
[210]	valid_0's auc: 0.669484
[220]	valid_0's auc: 0.669521
[230]	valid_0's auc: 0.669658
[240]	valid_0's auc: 0.669843
[250]	valid_0's auc: 0.669835
[260]	valid_0's auc: 0.669974
[270]	valid_0's auc: 0.670426
[280]	valid_0's auc: 0.670474
[290]	valid_0's auc: 0.670601
[300]	valid_0's auc: 0.670913
[310]	valid_0's auc: 0.670913
[320]	valid_0's auc: 0.671244
[330]	valid_0's auc: 0.671246
[340]	valid_0's auc: 0.671354
[350]	valid_0's auc: 0.671307
[360]	valid_0's auc: 0.671374
[370]	valid_0's auc: 0.671364
[380]	valid_0's auc: 0.6714
[390]	valid_0's auc: 0.671542
[400]	valid_0's auc: 0.671617
[410]	valid_0's auc: 0.671848
[420]	valid_0's auc: 0.671871
[430]	valid_0's auc: 0.671897
[440]	valid_0's auc: 0.671692
[450]	valid_0's auc: 0.671756
[460]	valid_0's auc: 0.671812
[470]	valid_0's auc: 0.671912
[480]	valid_0's auc: 0.672124
[490]	valid_0's auc: 0.672143
[500]	valid_0's auc: 0.672114
[510]	valid_0's auc: 0.672121
[520]	valid_0's auc: 0.672115
[530]	valid_0's auc: 0.672005
[540]	valid_0's auc: 0.672027
[550]	valid_0's auc: 0.672359
[560]	valid_0's auc: 0.671806
[570]	valid_0's auc: 0.671802
[580]	valid_0's auc: 0.671867
[590]	valid_0's auc: 0.671894
[600]	valid_0's auc: 0.671764
Early stopping, best iteration is:
[555]	valid_0's auc: 0.672377
best score: 0.672376918219
best iteration: 555

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 1 complete in 294m 42s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 2

max_depth            : 11
bagging_seed         : 2
bagging_freq         : 4
lambda_l2            : 0
lambda_l1            : 0
learning_rate        : 0.3
feature_fraction     : 0.8
feature_fraction_seed : 2
boosting             : dart
num_leaves           : 127
bagging_fraction     : 0.8
xgboost_dart_mode    : True

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.640415
[20]	valid_0's auc: 0.647097
[30]	valid_0's auc: 0.650951
[40]	valid_0's auc: 0.651794
[50]	valid_0's auc: 0.652836
[60]	valid_0's auc: 0.654401
[70]	valid_0's auc: 0.655395
[80]	valid_0's auc: 0.656572
[90]	valid_0's auc: 0.656942
[100]	valid_0's auc: 0.657893
[110]	valid_0's auc: 0.65835
[120]	valid_0's auc: 0.659071
[130]	valid_0's auc: 0.659394
[140]	valid_0's auc: 0.660111
[150]	valid_0's auc: 0.660485
[160]	valid_0's auc: 0.660618
[170]	valid_0's auc: 0.661088
[180]	valid_0's auc: 0.661421
[190]	valid_0's auc: 0.661653
[200]	valid_0's auc: 0.662004
[210]	valid_0's auc: 0.662184
[220]	valid_0's auc: 0.662389
[230]	valid_0's auc: 0.662606
[240]	valid_0's auc: 0.663259
[250]	valid_0's auc: 0.663349
[260]	valid_0's auc: 0.663604
[270]	valid_0's auc: 0.66392
[280]	valid_0's auc: 0.664138
[290]	valid_0's auc: 0.66465
[300]	valid_0's auc: 0.664676
[310]	valid_0's auc: 0.66473
[320]	valid_0's auc: 0.664896
[330]	valid_0's auc: 0.665041
[340]	valid_0's auc: 0.66523
[350]	valid_0's auc: 0.665305
[360]	valid_0's auc: 0.665364
[370]	valid_0's auc: 0.665396
[380]	valid_0's auc: 0.66551
[390]	valid_0's auc: 0.665753
[400]	valid_0's auc: 0.66638
[410]	valid_0's auc: 0.666467
[420]	valid_0's auc: 0.666547
[430]	valid_0's auc: 0.666973
[440]	valid_0's auc: 0.667074
[450]	valid_0's auc: 0.667213
[460]	valid_0's auc: 0.667225
[470]	valid_0's auc: 0.667431
[480]	valid_0's auc: 0.667768
[490]	valid_0's auc: 0.667686
[500]	valid_0's auc: 0.66774
[510]	valid_0's auc: 0.66782
[520]	valid_0's auc: 0.667882
[530]	valid_0's auc: 0.667864
[540]	valid_0's auc: 0.667897
[550]	valid_0's auc: 0.667936
[560]	valid_0's auc: 0.667699
[570]	valid_0's auc: 0.667847
[580]	valid_0's auc: 0.667909
[590]	valid_0's auc: 0.668197
[600]	valid_0's auc: 0.668184
[610]	valid_0's auc: 0.668221
[620]	valid_0's auc: 0.66825
[630]	valid_0's auc: 0.668275
[640]	valid_0's auc: 0.668291
[650]	valid_0's auc: 0.668307
[660]	valid_0's auc: 0.668304
[670]	valid_0's auc: 0.668883
'''
