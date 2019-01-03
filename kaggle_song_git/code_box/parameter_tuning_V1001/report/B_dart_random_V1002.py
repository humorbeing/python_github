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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/dart_random_V1002.py
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

bagging_freq         : 4
max_depth            : -1
num_leaves           : 511
xgboost_dart_mode    : False
lambda_l1            : 0
feature_fraction     : 0.8
bagging_seed         : 2
feature_fraction_seed : 2
boosting             : dart
learning_rate        : 0.5
lambda_l2            : 0
bagging_fraction     : 0.8

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.670359
[20]	valid_0's auc: 0.672265
[30]	valid_0's auc: 0.671833
[40]	valid_0's auc: 0.671006
[50]	valid_0's auc: 0.671448
[60]	valid_0's auc: 0.671637
[70]	valid_0's auc: 0.672231
Early stopping, best iteration is:
[22]	valid_0's auc: 0.672581
best score: 0.672580637071
best iteration: 22

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 0 complete in 19m 54s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 1

bagging_freq         : 4
max_depth            : 10
num_leaves           : 1023
xgboost_dart_mode    : True
lambda_l1            : 0
feature_fraction     : 0.8
bagging_seed         : 2
feature_fraction_seed : 2
boosting             : dart
learning_rate        : 0.3
lambda_l2            : 0
bagging_fraction     : 0.8

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.638814
[20]	valid_0's auc: 0.644766
[30]	valid_0's auc: 0.648088
[40]	valid_0's auc: 0.650619
[50]	valid_0's auc: 0.651658
[60]	valid_0's auc: 0.652686
[70]	valid_0's auc: 0.653782
[80]	valid_0's auc: 0.654849
[90]	valid_0's auc: 0.655314
[100]	valid_0's auc: 0.656334
[110]	valid_0's auc: 0.657067
[120]	valid_0's auc: 0.657891
[130]	valid_0's auc: 0.658349
[140]	valid_0's auc: 0.65886
[150]	valid_0's auc: 0.659178
[160]	valid_0's auc: 0.659357
[170]	valid_0's auc: 0.659832
[180]	valid_0's auc: 0.660144
[190]	valid_0's auc: 0.660634
[200]	valid_0's auc: 0.660878
[210]	valid_0's auc: 0.661203
[220]	valid_0's auc: 0.661536
[230]	valid_0's auc: 0.661752
[240]	valid_0's auc: 0.662257
[250]	valid_0's auc: 0.662455
[260]	valid_0's auc: 0.662835
[270]	valid_0's auc: 0.66292
[280]	valid_0's auc: 0.663032
[290]	valid_0's auc: 0.663371
[300]	valid_0's auc: 0.663555
[310]	valid_0's auc: 0.663603
[320]	valid_0's auc: 0.663821
[330]	valid_0's auc: 0.663963
[340]	valid_0's auc: 0.664395
[350]	valid_0's auc: 0.66449
[360]	valid_0's auc: 0.664586
[370]	valid_0's auc: 0.66468
[380]	valid_0's auc: 0.664898
[390]	valid_0's auc: 0.665296
[400]	valid_0's auc: 0.665521
[410]	valid_0's auc: 0.665618
[420]	valid_0's auc: 0.665599
[430]	valid_0's auc: 0.666034
[440]	valid_0's auc: 0.666112
[450]	valid_0's auc: 0.66619
[460]	valid_0's auc: 0.666195
[470]	valid_0's auc: 0.666404
[480]	valid_0's auc: 0.666748
[490]	valid_0's auc: 0.666882
[500]	valid_0's auc: 0.667059
[510]	valid_0's auc: 0.667172
[520]	valid_0's auc: 0.667209
[530]	valid_0's auc: 0.6673
[540]	valid_0's auc: 0.667286
[550]	valid_0's auc: 0.667265
[560]	valid_0's auc: 0.667433
[570]	valid_0's auc: 0.667479
[580]	valid_0's auc: 0.667677
[590]	valid_0's auc: 0.667739
[600]	valid_0's auc: 0.66782
[610]	valid_0's auc: 0.667917
[620]	valid_0's auc: 0.668084
[630]	valid_0's auc: 0.668094
[640]	valid_0's auc: 0.66807
[650]	valid_0's auc: 0.668343
[660]	valid_0's auc: 0.66835
[670]	valid_0's auc: 0.668388
[680]	valid_0's auc: 0.668478
[690]	valid_0's auc: 0.668481
[700]	valid_0's auc: 0.66871
[710]	valid_0's auc: 0.668719
[720]	valid_0's auc: 0.668699
[730]	valid_0's auc: 0.668969
[740]	valid_0's auc: 0.669035
[750]	valid_0's auc: 0.669012
[760]	valid_0's auc: 0.669094
[770]	valid_0's auc: 0.669091
[780]	valid_0's auc: 0.669259
[790]	valid_0's auc: 0.66923
[800]	valid_0's auc: 0.669338
[810]	valid_0's auc: 0.669346
[820]	valid_0's auc: 0.669351
[830]	valid_0's auc: 0.669448
[840]	valid_0's auc: 0.669468
[850]	valid_0's auc: 0.669469
[860]	valid_0's auc: 0.669574
[870]	valid_0's auc: 0.669699
[880]	valid_0's auc: 0.669844
[890]	valid_0's auc: 0.669931
[900]	valid_0's auc: 0.669866
[910]	valid_0's auc: 0.66986
[920]	valid_0's auc: 0.669918
[930]	valid_0's auc: 0.669925
Early stopping, best iteration is:
[889]	valid_0's auc: 0.669951
best score: 0.669950689997
best iteration: 889

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 1 complete in 447m 14s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 2

bagging_freq         : 4
max_depth            : 11
num_leaves           : 511
xgboost_dart_mode    : True
lambda_l1            : 0
feature_fraction     : 0.8
bagging_seed         : 2
feature_fraction_seed : 2
boosting             : dart
learning_rate        : 0.1
lambda_l2            : 0
bagging_fraction     : 0.8

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.634315
[20]	valid_0's auc: 0.638792
[30]	valid_0's auc: 0.642967
[40]	valid_0's auc: 0.645721
[50]	valid_0's auc: 0.646737
[60]	valid_0's auc: 0.647853
[70]	valid_0's auc: 0.648591
[80]	valid_0's auc: 0.64942
[90]	valid_0's auc: 0.64998
[100]	valid_0's auc: 0.650931
[110]	valid_0's auc: 0.651154
[120]	valid_0's auc: 0.651627
[130]	valid_0's auc: 0.651974
[140]	valid_0's auc: 0.652644
[150]	valid_0's auc: 0.653045
[160]	valid_0's auc: 0.653198
[170]	valid_0's auc: 0.653865
[180]	valid_0's auc: 0.654171
[190]	valid_0's auc: 0.654632
[200]	valid_0's auc: 0.655059
[210]	valid_0's auc: 0.655199
[220]	valid_0's auc: 0.655722
[230]	valid_0's auc: 0.656089
[240]	valid_0's auc: 0.656641
[250]	valid_0's auc: 0.656868
[260]	valid_0's auc: 0.65723
[270]	valid_0's auc: 0.65749
[280]	valid_0's auc: 0.657558
[290]	valid_0's auc: 0.657907
[300]	valid_0's auc: 0.658088
[310]	valid_0's auc: 0.65816
[320]	valid_0's auc: 0.658524
[330]	valid_0's auc: 0.658807
[340]	valid_0's auc: 0.659026
[350]	valid_0's auc: 0.659236
[360]	valid_0's auc: 0.659466
[370]	valid_0's auc: 0.659555
[380]	valid_0's auc: 0.659745
[390]	valid_0's auc: 0.659945
[400]	valid_0's auc: 0.660247
[410]	valid_0's auc: 0.660379
[420]	valid_0's auc: 0.660498
[430]	valid_0's auc: 0.660707
[440]	valid_0's auc: 0.661012
[450]	valid_0's auc: 0.66116
[460]	valid_0's auc: 0.661231
[470]	valid_0's auc: 0.661378
[480]	valid_0's auc: 0.661571
[490]	valid_0's auc: 0.661555
[500]	valid_0's auc: 0.661716
[510]	valid_0's auc: 0.661836
[520]	valid_0's auc: 0.661971
[530]	valid_0's auc: 0.662157
[540]	valid_0's auc: 0.66221
[550]	valid_0's auc: 0.662314
[560]	valid_0's auc: 0.662426
[570]	valid_0's auc: 0.662607
[580]	valid_0's auc: 0.662781
[590]	valid_0's auc: 0.662866
[600]	valid_0's auc: 0.663048
[610]	valid_0's auc: 0.663169
[620]	valid_0's auc: 0.663238
[630]	valid_0's auc: 0.663247
[640]	valid_0's auc: 0.663307
[650]	valid_0's auc: 0.663339
[660]	valid_0's auc: 0.663418
[670]	valid_0's auc: 0.663567
[680]	valid_0's auc: 0.663606
[690]	valid_0's auc: 0.663665
[700]	valid_0's auc: 0.663749
[710]	valid_0's auc: 0.663769
[720]	valid_0's auc: 0.663884
[730]	valid_0's auc: 0.663953
[740]	valid_0's auc: 0.664039
[750]	valid_0's auc: 0.664046
[760]	valid_0's auc: 0.6642
[770]	valid_0's auc: 0.664244
[780]	valid_0's auc: 0.664324
[790]	valid_0's auc: 0.66437
[800]	valid_0's auc: 0.664561
[810]	valid_0's auc: 0.6646
[820]	valid_0's auc: 0.664641
[830]	valid_0's auc: 0.664683
[840]	valid_0's auc: 0.664713
[850]	valid_0's auc: 0.664788
[860]	valid_0's auc: 0.664818
[870]	valid_0's auc: 0.664932
[880]	valid_0's auc: 0.665017
[890]	valid_0's auc: 0.665125
[900]	valid_0's auc: 0.665159
[910]	valid_0's auc: 0.665259
[920]	valid_0's auc: 0.665282
[930]	valid_0's auc: 0.665273
[940]	valid_0's auc: 0.665279
[950]	valid_0's auc: 0.66528
[960]	valid_0's auc: 0.665358
[970]	valid_0's auc: 0.665399
[980]	valid_0's auc: 0.665423
[990]	valid_0's auc: 0.665511
[1000]	valid_0's auc: 0.66556
[1010]	valid_0's auc: 0.665558
[1020]	valid_0's auc: 0.665607
[1030]	valid_0's auc: 0.665623
[1040]	valid_0's auc: 0.665665
[1050]	valid_0's auc: 0.665735
[1060]	valid_0's auc: 0.665797
[1070]	valid_0's auc: 0.66588
'''