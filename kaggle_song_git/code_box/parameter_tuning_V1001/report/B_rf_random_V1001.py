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
lr_s = [0.5, 0.1,0.02, 0.3, 0.2]
nl_s = [511,1023, 511, 511, 511]
md_s = [ -1,  10,  11,  -1,  10]
l2_s = [  0,   0,   0,   0, 0.3]
l1_s = [  0,   0,   0, 0.3,   0]
# mb_s = [ 511,  511,  255, 255, 127]


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
    boosting = b_s[1]
    learning_rate = lr_s[i]
    num_leaves = nl_s[i]
    max_depth = md_s[i]
    lambda_l1 = l1_s[i]
    lambda_l2 = l2_s[i]
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/rf_random_V1001.py
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

bagging_fraction     : 0.8
lambda_l1            : 0
bagging_freq         : 4
num_leaves           : 511
bagging_seed         : 2
max_depth            : -1
lambda_l2            : 0
feature_fraction_seed : 2
learning_rate        : 0.5
feature_fraction     : 0.8
boosting             : rf

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.655548
[20]	valid_0's auc: 0.656137
[30]	valid_0's auc: 0.657045
[40]	valid_0's auc: 0.658208
[50]	valid_0's auc: 0.658261
[60]	valid_0's auc: 0.658617
[70]	valid_0's auc: 0.659092
[80]	valid_0's auc: 0.659294
[90]	valid_0's auc: 0.659459
[100]	valid_0's auc: 0.659367
[110]	valid_0's auc: 0.65928
[120]	valid_0's auc: 0.659236
[130]	valid_0's auc: 0.659195
[140]	valid_0's auc: 0.659276
Early stopping, best iteration is:
[94]	valid_0's auc: 0.659524
best score: 0.659523842736
best iteration: 94

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 0 complete in 11m 29s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 1

bagging_fraction     : 0.8
lambda_l1            : 0
bagging_freq         : 4
num_leaves           : 1023
bagging_seed         : 2
max_depth            : 10
lambda_l2            : 0
feature_fraction_seed : 2
learning_rate        : 0.1
feature_fraction     : 0.8
boosting             : rf

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.627434
[20]	valid_0's auc: 0.627644
[30]	valid_0's auc: 0.627455
[40]	valid_0's auc: 0.627895
[50]	valid_0's auc: 0.627983
[60]	valid_0's auc: 0.628097
[70]	valid_0's auc: 0.62834
[80]	valid_0's auc: 0.628526
[90]	valid_0's auc: 0.628606
[100]	valid_0's auc: 0.628494
[110]	valid_0's auc: 0.628609
[120]	valid_0's auc: 0.628727
[130]	valid_0's auc: 0.628564
[140]	valid_0's auc: 0.628689
[150]	valid_0's auc: 0.628673
[160]	valid_0's auc: 0.628804
[170]	valid_0's auc: 0.628757
[180]	valid_0's auc: 0.628728
[190]	valid_0's auc: 0.628756
[200]	valid_0's auc: 0.62877
[210]	valid_0's auc: 0.628811
[220]	valid_0's auc: 0.628798
[230]	valid_0's auc: 0.628805
[240]	valid_0's auc: 0.6288
[250]	valid_0's auc: 0.628758
[260]	valid_0's auc: 0.628734
[270]	valid_0's auc: 0.628787
[280]	valid_0's auc: 0.628792
Early stopping, best iteration is:
[234]	valid_0's auc: 0.628824
best score: 0.628824192997
best iteration: 234

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 1 complete in 4m 42s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 2

bagging_fraction     : 0.8
lambda_l1            : 0
bagging_freq         : 4
num_leaves           : 511
bagging_seed         : 2
max_depth            : 11
lambda_l2            : 0
feature_fraction_seed : 2
learning_rate        : 0.02
feature_fraction     : 0.8
boosting             : rf

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.629068
[20]	valid_0's auc: 0.629078
[30]	valid_0's auc: 0.628858
[40]	valid_0's auc: 0.6297
[50]	valid_0's auc: 0.629687
[60]	valid_0's auc: 0.62965
[70]	valid_0's auc: 0.629938
[80]	valid_0's auc: 0.630005
[90]	valid_0's auc: 0.630105
[100]	valid_0's auc: 0.630119
[110]	valid_0's auc: 0.63008
[120]	valid_0's auc: 0.630148
[130]	valid_0's auc: 0.630065
[140]	valid_0's auc: 0.630087
[150]	valid_0's auc: 0.630069
[160]	valid_0's auc: 0.630153
[170]	valid_0's auc: 0.630139
[180]	valid_0's auc: 0.630125
[190]	valid_0's auc: 0.630122
[200]	valid_0's auc: 0.630144
[210]	valid_0's auc: 0.630149
[220]	valid_0's auc: 0.63012
Early stopping, best iteration is:
[176]	valid_0's auc: 0.630158
best score: 0.630157944455
best iteration: 176

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 2 complete in 4m 4s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 3

bagging_fraction     : 0.8
lambda_l1            : 0.3
bagging_freq         : 4
num_leaves           : 511
bagging_seed         : 2
max_depth            : -1
lambda_l2            : 0
feature_fraction_seed : 2
learning_rate        : 0.3
feature_fraction     : 0.8
boosting             : rf

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.655515
[20]	valid_0's auc: 0.65611
[30]	valid_0's auc: 0.657009
[40]	valid_0's auc: 0.658137
[50]	valid_0's auc: 0.658236
[60]	valid_0's auc: 0.658606
[70]	valid_0's auc: 0.659085
[80]	valid_0's auc: 0.659334
[90]	valid_0's auc: 0.65949
[100]	valid_0's auc: 0.659395
[110]	valid_0's auc: 0.659303
[120]	valid_0's auc: 0.65926
[130]	valid_0's auc: 0.659217
[140]	valid_0's auc: 0.659292
Early stopping, best iteration is:
[94]	valid_0's auc: 0.659552
best score: 0.659552140066
best iteration: 94

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 3 complete in 11m 22s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 4

bagging_fraction     : 0.8
lambda_l1            : 0
bagging_freq         : 4
num_leaves           : 511
bagging_seed         : 2
max_depth            : 10
lambda_l2            : 0.3
feature_fraction_seed : 2
learning_rate        : 0.2
feature_fraction     : 0.8
boosting             : rf

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.627426
[20]	valid_0's auc: 0.627638
[30]	valid_0's auc: 0.627453
[40]	valid_0's auc: 0.627883
[50]	valid_0's auc: 0.627985
[60]	valid_0's auc: 0.628105
[70]	valid_0's auc: 0.62834
[80]	valid_0's auc: 0.628524
[90]	valid_0's auc: 0.6286
[100]	valid_0's auc: 0.628491
[110]	valid_0's auc: 0.628603
[120]	valid_0's auc: 0.628726
[130]	valid_0's auc: 0.628561
[140]	valid_0's auc: 0.628681
[150]	valid_0's auc: 0.628666
[160]	valid_0's auc: 0.628797
[170]	valid_0's auc: 0.628749
[180]	valid_0's auc: 0.628723
[190]	valid_0's auc: 0.628756
[200]	valid_0's auc: 0.628764
[210]	valid_0's auc: 0.628807
[220]	valid_0's auc: 0.628793
[230]	valid_0's auc: 0.628799
[240]	valid_0's auc: 0.628797
[250]	valid_0's auc: 0.62875
Early stopping, best iteration is:
[208]	valid_0's auc: 0.628815
best score: 0.628814568051
best iteration: 208

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 4 complete in 4m 14s

[timer]: complete in 36m 26s

Process finished with exit code 0
'''