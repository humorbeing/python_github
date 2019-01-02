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
lr_s = [ 0.3, 0.5, 0.03,  0.3, 0.2]
nl_s = [   7, 1023,  511, 127, 511]
md_s = [  63,   10,   11,   5,  10]
l2_s = [   0,  0.4,  0.1,   2, 0.1]
l1_s = [   0,    0,    0,   0, 0.1]
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
    boosting = b_s[0]
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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1005.py
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

bagging_seed         : 2
feature_fraction_seed : 2
lambda_l1            : 0
boosting             : gbdt
bagging_freq         : 4
feature_fraction     : 0.8
lambda_l2            : 0
learning_rate        : 0.3
bagging_fraction     : 0.8
num_leaves           : 7
max_depth            : 63

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.625478
[20]	valid_0's auc: 0.632108
[30]	valid_0's auc: 0.637183
[40]	valid_0's auc: 0.640467
[50]	valid_0's auc: 0.642546
[60]	valid_0's auc: 0.644794
[70]	valid_0's auc: 0.64627
[80]	valid_0's auc: 0.6477
[90]	valid_0's auc: 0.648661
[100]	valid_0's auc: 0.650014
[110]	valid_0's auc: 0.651213
[120]	valid_0's auc: 0.651868
[130]	valid_0's auc: 0.652644
[140]	valid_0's auc: 0.653326
[150]	valid_0's auc: 0.653903
[160]	valid_0's auc: 0.654219
[170]	valid_0's auc: 0.654622
[180]	valid_0's auc: 0.655236
[190]	valid_0's auc: 0.65542
[200]	valid_0's auc: 0.655769
[210]	valid_0's auc: 0.656311
[220]	valid_0's auc: 0.656601
[230]	valid_0's auc: 0.657044
[240]	valid_0's auc: 0.656994
[250]	valid_0's auc: 0.657216
[260]	valid_0's auc: 0.657508
[270]	valid_0's auc: 0.657662
[280]	valid_0's auc: 0.657834
[290]	valid_0's auc: 0.658037
[300]	valid_0's auc: 0.658155
[310]	valid_0's auc: 0.65831
[320]	valid_0's auc: 0.658604
[330]	valid_0's auc: 0.65875
[340]	valid_0's auc: 0.658865
[350]	valid_0's auc: 0.659171
[360]	valid_0's auc: 0.659216
[370]	valid_0's auc: 0.659414
[380]	valid_0's auc: 0.65941
[390]	valid_0's auc: 0.659522
[400]	valid_0's auc: 0.659627
[410]	valid_0's auc: 0.659789
[420]	valid_0's auc: 0.66006
[430]	valid_0's auc: 0.660285
[440]	valid_0's auc: 0.660397
[450]	valid_0's auc: 0.660508
[460]	valid_0's auc: 0.660572
[470]	valid_0's auc: 0.660881
[480]	valid_0's auc: 0.660911
[490]	valid_0's auc: 0.661097
[500]	valid_0's auc: 0.661099
[510]	valid_0's auc: 0.66111
[520]	valid_0's auc: 0.661289
[530]	valid_0's auc: 0.661343
[540]	valid_0's auc: 0.661397
[550]	valid_0's auc: 0.661413
[560]	valid_0's auc: 0.6615
[570]	valid_0's auc: 0.661441
[580]	valid_0's auc: 0.661489
[590]	valid_0's auc: 0.661535
[600]	valid_0's auc: 0.661652
[610]	valid_0's auc: 0.661729
[620]	valid_0's auc: 0.661767
[630]	valid_0's auc: 0.661778
[640]	valid_0's auc: 0.661791
[650]	valid_0's auc: 0.661793
[660]	valid_0's auc: 0.661834
[670]	valid_0's auc: 0.661897
[680]	valid_0's auc: 0.661844
[690]	valid_0's auc: 0.661825
[700]	valid_0's auc: 0.661816
[710]	valid_0's auc: 0.6621
[720]	valid_0's auc: 0.662137
[730]	valid_0's auc: 0.662151
[740]	valid_0's auc: 0.662201
[750]	valid_0's auc: 0.66224
[760]	valid_0's auc: 0.662221
[770]	valid_0's auc: 0.662324
[780]	valid_0's auc: 0.662339
[790]	valid_0's auc: 0.662422
[800]	valid_0's auc: 0.662408
[810]	valid_0's auc: 0.662583
[820]	valid_0's auc: 0.66258
[830]	valid_0's auc: 0.662586
[840]	valid_0's auc: 0.662595
[850]	valid_0's auc: 0.662656
[860]	valid_0's auc: 0.662721
[870]	valid_0's auc: 0.662794
[880]	valid_0's auc: 0.662793
[890]	valid_0's auc: 0.662859
[900]	valid_0's auc: 0.662828
[910]	valid_0's auc: 0.662908
[920]	valid_0's auc: 0.662979
[930]	valid_0's auc: 0.663011
[940]	valid_0's auc: 0.663001
[950]	valid_0's auc: 0.663042
[960]	valid_0's auc: 0.662984
[970]	valid_0's auc: 0.662969
[980]	valid_0's auc: 0.663059
[990]	valid_0's auc: 0.663112
[1000]	valid_0's auc: 0.663068
[1010]	valid_0's auc: 0.66305
[1020]	valid_0's auc: 0.663137
[1030]	valid_0's auc: 0.663147
[1040]	valid_0's auc: 0.663128
[1050]	valid_0's auc: 0.663168
[1060]	valid_0's auc: 0.663161
[1070]	valid_0's auc: 0.663196
[1080]	valid_0's auc: 0.66326
[1090]	valid_0's auc: 0.663253
[1100]	valid_0's auc: 0.663125
[1110]	valid_0's auc: 0.663086
[1120]	valid_0's auc: 0.663139
[1130]	valid_0's auc: 0.663137
[1140]	valid_0's auc: 0.663158
Early stopping, best iteration is:
[1092]	valid_0's auc: 0.663265
best score: 0.663265263598
best iteration: 1092

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 0 complete in 9m 10s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 1

bagging_seed         : 2
feature_fraction_seed : 2
lambda_l1            : 0
boosting             : gbdt
bagging_freq         : 4
feature_fraction     : 0.8
lambda_l2            : 0.4
learning_rate        : 0.5
bagging_fraction     : 0.8
num_leaves           : 1023
max_depth            : 10

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642016
[20]	valid_0's auc: 0.648264
[30]	valid_0's auc: 0.651979
[40]	valid_0's auc: 0.654965
[50]	valid_0's auc: 0.656323
[60]	valid_0's auc: 0.657852
[70]	valid_0's auc: 0.658712
[80]	valid_0's auc: 0.660338
[90]	valid_0's auc: 0.660847
[100]	valid_0's auc: 0.661667
[110]	valid_0's auc: 0.662342
[120]	valid_0's auc: 0.662476
[130]	valid_0's auc: 0.663109
[140]	valid_0's auc: 0.663295
[150]	valid_0's auc: 0.663422
[160]	valid_0's auc: 0.663698
[170]	valid_0's auc: 0.663685
[180]	valid_0's auc: 0.663525
[190]	valid_0's auc: 0.663645
[200]	valid_0's auc: 0.664857
[210]	valid_0's auc: 0.665178
[220]	valid_0's auc: 0.665354
[230]	valid_0's auc: 0.665647
[240]	valid_0's auc: 0.665875
[250]	valid_0's auc: 0.665624
[260]	valid_0's auc: 0.665564
[270]	valid_0's auc: 0.665466
[280]	valid_0's auc: 0.665538
[290]	valid_0's auc: 0.665783
Early stopping, best iteration is:
[243]	valid_0's auc: 0.665955
best score: 0.665954907643
best iteration: 243

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 1 complete in 3m 37s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 2

bagging_seed         : 2
feature_fraction_seed : 2
lambda_l1            : 0
boosting             : gbdt
bagging_freq         : 4
feature_fraction     : 0.8
lambda_l2            : 0.1
learning_rate        : 0.03
bagging_fraction     : 0.8
num_leaves           : 511
max_depth            : 11

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.63097
[20]	valid_0's auc: 0.633218
[30]	valid_0's auc: 0.6354
[40]	valid_0's auc: 0.637486
[50]	valid_0's auc: 0.63913
[60]	valid_0's auc: 0.640996
[70]	valid_0's auc: 0.642972
[80]	valid_0's auc: 0.644147
[90]	valid_0's auc: 0.645353
[100]	valid_0's auc: 0.646509
[110]	valid_0's auc: 0.647343
[120]	valid_0's auc: 0.64801
[130]	valid_0's auc: 0.648624
[140]	valid_0's auc: 0.649161
[150]	valid_0's auc: 0.649632
[160]	valid_0's auc: 0.650049
[170]	valid_0's auc: 0.650552
[180]	valid_0's auc: 0.65101
[190]	valid_0's auc: 0.651367
[200]	valid_0's auc: 0.651669
[210]	valid_0's auc: 0.652054
[220]	valid_0's auc: 0.652381
[230]	valid_0's auc: 0.652697
[240]	valid_0's auc: 0.652939
[250]	valid_0's auc: 0.653277
[260]	valid_0's auc: 0.653508
[270]	valid_0's auc: 0.653762
[280]	valid_0's auc: 0.653983
[290]	valid_0's auc: 0.654212
[300]	valid_0's auc: 0.654461
[310]	valid_0's auc: 0.654687
[320]	valid_0's auc: 0.65488
[330]	valid_0's auc: 0.655148
[340]	valid_0's auc: 0.655387
[350]	valid_0's auc: 0.655635
[360]	valid_0's auc: 0.655827
[370]	valid_0's auc: 0.656023
[380]	valid_0's auc: 0.656231
[390]	valid_0's auc: 0.656417
[400]	valid_0's auc: 0.656546
[410]	valid_0's auc: 0.656748
[420]	valid_0's auc: 0.656926
[430]	valid_0's auc: 0.657133
[440]	valid_0's auc: 0.657275
[450]	valid_0's auc: 0.657476
[460]	valid_0's auc: 0.657638
[470]	valid_0's auc: 0.657749
[480]	valid_0's auc: 0.65789
[490]	valid_0's auc: 0.658088
[500]	valid_0's auc: 0.658199
[510]	valid_0's auc: 0.658367
[520]	valid_0's auc: 0.65856
[530]	valid_0's auc: 0.658747
[540]	valid_0's auc: 0.658827
[550]	valid_0's auc: 0.658948
[560]	valid_0's auc: 0.659047
[570]	valid_0's auc: 0.659149
[580]	valid_0's auc: 0.659251
[590]	valid_0's auc: 0.659356
[600]	valid_0's auc: 0.659512
[610]	valid_0's auc: 0.65963
[620]	valid_0's auc: 0.659731
[630]	valid_0's auc: 0.659863
[640]	valid_0's auc: 0.659987
[650]	valid_0's auc: 0.660111
[660]	valid_0's auc: 0.66028
[670]	valid_0's auc: 0.660401
[680]	valid_0's auc: 0.660533
[690]	valid_0's auc: 0.66061
[700]	valid_0's auc: 0.660681
[710]	valid_0's auc: 0.660792
[720]	valid_0's auc: 0.660925
[730]	valid_0's auc: 0.661019
[740]	valid_0's auc: 0.661087
[750]	valid_0's auc: 0.661187
[760]	valid_0's auc: 0.661303
[770]	valid_0's auc: 0.661417
[780]	valid_0's auc: 0.661508
[790]	valid_0's auc: 0.661624
[800]	valid_0's auc: 0.661718
[810]	valid_0's auc: 0.66183
[820]	valid_0's auc: 0.661916
[830]	valid_0's auc: 0.661995
[840]	valid_0's auc: 0.662062
[850]	valid_0's auc: 0.662156
[860]	valid_0's auc: 0.662239
[870]	valid_0's auc: 0.66231
[880]	valid_0's auc: 0.66239
[890]	valid_0's auc: 0.662476
[900]	valid_0's auc: 0.662587
[910]	valid_0's auc: 0.662691
[920]	valid_0's auc: 0.662748
[930]	valid_0's auc: 0.662794
[940]	valid_0's auc: 0.66281
[950]	valid_0's auc: 0.662883
[960]	valid_0's auc: 0.662946
[970]	valid_0's auc: 0.66303
[980]	valid_0's auc: 0.663141
[990]	valid_0's auc: 0.663229
[1000]	valid_0's auc: 0.663288
[1010]	valid_0's auc: 0.663346
[1020]	valid_0's auc: 0.663412
[1030]	valid_0's auc: 0.663456
[1040]	valid_0's auc: 0.663503
[1050]	valid_0's auc: 0.663558
[1060]	valid_0's auc: 0.663605
[1070]	valid_0's auc: 0.66366
[1080]	valid_0's auc: 0.663738
[1090]	valid_0's auc: 0.663786
[1100]	valid_0's auc: 0.663846
[1110]	valid_0's auc: 0.663892
[1120]	valid_0's auc: 0.663934
[1130]	valid_0's auc: 0.663984
[1140]	valid_0's auc: 0.664028
[1150]	valid_0's auc: 0.664104
[1160]	valid_0's auc: 0.66419
[1170]	valid_0's auc: 0.664244
[1180]	valid_0's auc: 0.664307
[1190]	valid_0's auc: 0.664371
[1200]	valid_0's auc: 0.6644
[1210]	valid_0's auc: 0.664436
[1220]	valid_0's auc: 0.664472
[1230]	valid_0's auc: 0.664534
[1240]	valid_0's auc: 0.664576
[1250]	valid_0's auc: 0.664641
[1260]	valid_0's auc: 0.664696
[1270]	valid_0's auc: 0.664752
[1280]	valid_0's auc: 0.664789
[1290]	valid_0's auc: 0.664818
[1300]	valid_0's auc: 0.664889
[1310]	valid_0's auc: 0.664934
[1320]	valid_0's auc: 0.665007
[1330]	valid_0's auc: 0.665064
[1340]	valid_0's auc: 0.6651
[1350]	valid_0's auc: 0.665126
[1360]	valid_0's auc: 0.665166
[1370]	valid_0's auc: 0.665193
[1380]	valid_0's auc: 0.665221
[1390]	valid_0's auc: 0.66525
[1400]	valid_0's auc: 0.665273
[1410]	valid_0's auc: 0.665303
[1420]	valid_0's auc: 0.665361
[1430]	valid_0's auc: 0.665417
[1440]	valid_0's auc: 0.665441
[1450]	valid_0's auc: 0.665458
[1460]	valid_0's auc: 0.665526
[1470]	valid_0's auc: 0.665566
[1480]	valid_0's auc: 0.665611
[1490]	valid_0's auc: 0.665655
[1500]	valid_0's auc: 0.665697
[1510]	valid_0's auc: 0.665715
[1520]	valid_0's auc: 0.665768
[1530]	valid_0's auc: 0.665804
[1540]	valid_0's auc: 0.665881
[1550]	valid_0's auc: 0.665902
[1560]	valid_0's auc: 0.665917
[1570]	valid_0's auc: 0.665957
[1580]	valid_0's auc: 0.665972
[1590]	valid_0's auc: 0.666018
[1600]	valid_0's auc: 0.666053
[1610]	valid_0's auc: 0.666095
[1620]	valid_0's auc: 0.666155
[1630]	valid_0's auc: 0.666174
[1640]	valid_0's auc: 0.666198
[1650]	valid_0's auc: 0.666229
[1660]	valid_0's auc: 0.666269
[1670]	valid_0's auc: 0.666297
[1680]	valid_0's auc: 0.666318
[1690]	valid_0's auc: 0.666353
[1700]	valid_0's auc: 0.666383
[1710]	valid_0's auc: 0.666414
[1720]	valid_0's auc: 0.666446
[1730]	valid_0's auc: 0.666469
[1740]	valid_0's auc: 0.666495
[1750]	valid_0's auc: 0.666518
[1760]	valid_0's auc: 0.666569
[1770]	valid_0's auc: 0.666634
[1780]	valid_0's auc: 0.666665
[1790]	valid_0's auc: 0.666694
[1800]	valid_0's auc: 0.666694
[1810]	valid_0's auc: 0.666717
[1820]	valid_0's auc: 0.666744
[1830]	valid_0's auc: 0.666758
[1840]	valid_0's auc: 0.666764
[1850]	valid_0's auc: 0.666802
[1860]	valid_0's auc: 0.666827
[1870]	valid_0's auc: 0.666859
[1880]	valid_0's auc: 0.666867
[1890]	valid_0's auc: 0.66687
[1900]	valid_0's auc: 0.666909
[1910]	valid_0's auc: 0.666937
[1920]	valid_0's auc: 0.666969
[1930]	valid_0's auc: 0.666982
[1940]	valid_0's auc: 0.667015
[1950]	valid_0's auc: 0.667044
[1960]	valid_0's auc: 0.667083
[1970]	valid_0's auc: 0.667131
[1980]	valid_0's auc: 0.667146
[1990]	valid_0's auc: 0.667169
[2000]	valid_0's auc: 0.667232
[2010]	valid_0's auc: 0.667246
[2020]	valid_0's auc: 0.667275
[2030]	valid_0's auc: 0.667297
[2040]	valid_0's auc: 0.667326
[2050]	valid_0's auc: 0.667332
[2060]	valid_0's auc: 0.667354
[2070]	valid_0's auc: 0.667378
[2080]	valid_0's auc: 0.667398
[2090]	valid_0's auc: 0.667423
[2100]	valid_0's auc: 0.667474
[2110]	valid_0's auc: 0.66749
[2120]	valid_0's auc: 0.667517
[2130]	valid_0's auc: 0.667543
[2140]	valid_0's auc: 0.667566
[2150]	valid_0's auc: 0.667586
[2160]	valid_0's auc: 0.667602
[2170]	valid_0's auc: 0.667606
[2180]	valid_0's auc: 0.667614
[2190]	valid_0's auc: 0.667646
[2200]	valid_0's auc: 0.667659
[2210]	valid_0's auc: 0.667681
[2220]	valid_0's auc: 0.667704
[2230]	valid_0's auc: 0.667714
[2240]	valid_0's auc: 0.667743
[2250]	valid_0's auc: 0.66779
[2260]	valid_0's auc: 0.667799
[2270]	valid_0's auc: 0.667831
[2280]	valid_0's auc: 0.66784
[2290]	valid_0's auc: 0.667866
[2300]	valid_0's auc: 0.667884
[2310]	valid_0's auc: 0.667889
[2320]	valid_0's auc: 0.667909
[2330]	valid_0's auc: 0.667915
[2340]	valid_0's auc: 0.667957
[2350]	valid_0's auc: 0.66798
[2360]	valid_0's auc: 0.667991
[2370]	valid_0's auc: 0.668026
[2380]	valid_0's auc: 0.668051
[2390]	valid_0's auc: 0.668117
[2400]	valid_0's auc: 0.668146
[2410]	valid_0's auc: 0.668161
[2420]	valid_0's auc: 0.668192
[2430]	valid_0's auc: 0.668224
[2440]	valid_0's auc: 0.668251
[2450]	valid_0's auc: 0.668265
[2460]	valid_0's auc: 0.668291
[2470]	valid_0's auc: 0.6683
[2480]	valid_0's auc: 0.668314
[2490]	valid_0's auc: 0.668324
[2500]	valid_0's auc: 0.668344
[2510]	valid_0's auc: 0.668379
[2520]	valid_0's auc: 0.668388
[2530]	valid_0's auc: 0.668397
[2540]	valid_0's auc: 0.66842
[2550]	valid_0's auc: 0.668439
[2560]	valid_0's auc: 0.668484
[2570]	valid_0's auc: 0.668486
[2580]	valid_0's auc: 0.668495
[2590]	valid_0's auc: 0.668537
[2600]	valid_0's auc: 0.668566
[2610]	valid_0's auc: 0.668576
[2620]	valid_0's auc: 0.668605
[2630]	valid_0's auc: 0.668617
[2640]	valid_0's auc: 0.668644
[2650]	valid_0's auc: 0.668656
[2660]	valid_0's auc: 0.668663
[2670]	valid_0's auc: 0.66869
[2680]	valid_0's auc: 0.668705
[2690]	valid_0's auc: 0.668727
[2700]	valid_0's auc: 0.668739
[2710]	valid_0's auc: 0.668764
[2720]	valid_0's auc: 0.668777
[2730]	valid_0's auc: 0.668783
[2740]	valid_0's auc: 0.66879
[2750]	valid_0's auc: 0.668828
[2760]	valid_0's auc: 0.668838
[2770]	valid_0's auc: 0.668896
[2780]	valid_0's auc: 0.668924
[2790]	valid_0's auc: 0.668928
[2800]	valid_0's auc: 0.668926
[2810]	valid_0's auc: 0.668943
[2820]	valid_0's auc: 0.668959
[2830]	valid_0's auc: 0.66898
[2840]	valid_0's auc: 0.668991
[2850]	valid_0's auc: 0.668995
[2860]	valid_0's auc: 0.669002
[2870]	valid_0's auc: 0.669025
[2880]	valid_0's auc: 0.669049
[2890]	valid_0's auc: 0.669074
[2900]	valid_0's auc: 0.669122
[2910]	valid_0's auc: 0.669175
[2920]	valid_0's auc: 0.669181
[2930]	valid_0's auc: 0.669192
[2940]	valid_0's auc: 0.66919
[2950]	valid_0's auc: 0.669222
[2960]	valid_0's auc: 0.669212
[2970]	valid_0's auc: 0.66922
[2980]	valid_0's auc: 0.669264
[2990]	valid_0's auc: 0.669294
[3000]	valid_0's auc: 0.66931
[3010]	valid_0's auc: 0.669323
[3020]	valid_0's auc: 0.669332
[3030]	valid_0's auc: 0.669353
[3040]	valid_0's auc: 0.669368
[3050]	valid_0's auc: 0.669385
[3060]	valid_0's auc: 0.669397
[3070]	valid_0's auc: 0.669383
[3080]	valid_0's auc: 0.669401
[3090]	valid_0's auc: 0.669411
[3100]	valid_0's auc: 0.669416
[3110]	valid_0's auc: 0.669422
[3120]	valid_0's auc: 0.669417
[3130]	valid_0's auc: 0.669443
[3140]	valid_0's auc: 0.669453
[3150]	valid_0's auc: 0.669465
[3160]	valid_0's auc: 0.669469
[3170]	valid_0's auc: 0.669483
[3180]	valid_0's auc: 0.669507
[3190]	valid_0's auc: 0.669529
[3200]	valid_0's auc: 0.669536
[3210]	valid_0's auc: 0.669583
[3220]	valid_0's auc: 0.669605
[3230]	valid_0's auc: 0.66962
[3240]	valid_0's auc: 0.669629
[3250]	valid_0's auc: 0.669647
[3260]	valid_0's auc: 0.669661
[3270]	valid_0's auc: 0.669669
[3280]	valid_0's auc: 0.669671
[3290]	valid_0's auc: 0.669685
[3300]	valid_0's auc: 0.669695
[3310]	valid_0's auc: 0.669713
[3320]	valid_0's auc: 0.669715
[3330]	valid_0's auc: 0.669757
[3340]	valid_0's auc: 0.669785
[3350]	valid_0's auc: 0.6698
[3360]	valid_0's auc: 0.669794
[3370]	valid_0's auc: 0.669808
[3380]	valid_0's auc: 0.66981
[3390]	valid_0's auc: 0.669821
[3400]	valid_0's auc: 0.669822
[3410]	valid_0's auc: 0.669844
[3420]	valid_0's auc: 0.66985
[3430]	valid_0's auc: 0.669857
[3440]	valid_0's auc: 0.669872
[3450]	valid_0's auc: 0.669888
[3460]	valid_0's auc: 0.669905
[3470]	valid_0's auc: 0.669909
[3480]	valid_0's auc: 0.669902
[3490]	valid_0's auc: 0.669912
[3500]	valid_0's auc: 0.669926
[3510]	valid_0's auc: 0.669953
[3520]	valid_0's auc: 0.669953
[3530]	valid_0's auc: 0.669971
[3540]	valid_0's auc: 0.669985
[3550]	valid_0's auc: 0.669987
[3560]	valid_0's auc: 0.670009
[3570]	valid_0's auc: 0.670012
[3580]	valid_0's auc: 0.670015
[3590]	valid_0's auc: 0.670041
[3600]	valid_0's auc: 0.67005
[3610]	valid_0's auc: 0.670029
[3620]	valid_0's auc: 0.670063
[3630]	valid_0's auc: 0.67007
[3640]	valid_0's auc: 0.670073
[3650]	valid_0's auc: 0.670079
[3660]	valid_0's auc: 0.670082
[3670]	valid_0's auc: 0.670079
[3680]	valid_0's auc: 0.670076
[3690]	valid_0's auc: 0.670086
[3700]	valid_0's auc: 0.670091
[3710]	valid_0's auc: 0.670107
[3720]	valid_0's auc: 0.670118
[3730]	valid_0's auc: 0.670132
[3740]	valid_0's auc: 0.670133
[3750]	valid_0's auc: 0.670134
[3760]	valid_0's auc: 0.670146
[3770]	valid_0's auc: 0.670154
[3780]	valid_0's auc: 0.670163
[3790]	valid_0's auc: 0.670174
[3800]	valid_0's auc: 0.670201
[3810]	valid_0's auc: 0.670227
[3820]	valid_0's auc: 0.670244
[3830]	valid_0's auc: 0.670207
[3840]	valid_0's auc: 0.670224
[3850]	valid_0's auc: 0.670235
[3860]	valid_0's auc: 0.670268
[3870]	valid_0's auc: 0.670276
[3880]	valid_0's auc: 0.6703
[3890]	valid_0's auc: 0.67032
[3900]	valid_0's auc: 0.670328
[3910]	valid_0's auc: 0.670333
[3920]	valid_0's auc: 0.670331
[3930]	valid_0's auc: 0.670334
[3940]	valid_0's auc: 0.67032
[3950]	valid_0's auc: 0.670326
[3960]	valid_0's auc: 0.670318
Early stopping, best iteration is:
[3914]	valid_0's auc: 0.670337
best score: 0.670336747688
best iteration: 3914

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 2 complete in 52m 7s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 3

bagging_seed         : 2
feature_fraction_seed : 2
lambda_l1            : 0
boosting             : gbdt
bagging_freq         : 4
feature_fraction     : 0.8
lambda_l2            : 2
learning_rate        : 0.3
bagging_fraction     : 0.8
num_leaves           : 127
max_depth            : 5

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.631277
[20]	valid_0's auc: 0.636959
[30]	valid_0's auc: 0.640765
[40]	valid_0's auc: 0.643271
[50]	valid_0's auc: 0.64552
[60]	valid_0's auc: 0.64733
[70]	valid_0's auc: 0.648701
[80]	valid_0's auc: 0.649824
[90]	valid_0's auc: 0.650895
[100]	valid_0's auc: 0.651878
[110]	valid_0's auc: 0.65283
[120]	valid_0's auc: 0.653415
[130]	valid_0's auc: 0.654109
[140]	valid_0's auc: 0.654534
[150]	valid_0's auc: 0.655134
[160]	valid_0's auc: 0.655418
[170]	valid_0's auc: 0.655915
[180]	valid_0's auc: 0.656444
[190]	valid_0's auc: 0.656864
[200]	valid_0's auc: 0.657164
[210]	valid_0's auc: 0.657645
[220]	valid_0's auc: 0.657928
[230]	valid_0's auc: 0.658186
[240]	valid_0's auc: 0.658481
[250]	valid_0's auc: 0.658737
[260]	valid_0's auc: 0.65914
[270]	valid_0's auc: 0.659233
[280]	valid_0's auc: 0.659434
[290]	valid_0's auc: 0.659651
[300]	valid_0's auc: 0.659797
[310]	valid_0's auc: 0.66005
[320]	valid_0's auc: 0.660356
[330]	valid_0's auc: 0.660478
[340]	valid_0's auc: 0.660524
[350]	valid_0's auc: 0.660833
[360]	valid_0's auc: 0.660938
[370]	valid_0's auc: 0.661078
[380]	valid_0's auc: 0.661269
[390]	valid_0's auc: 0.661394
[400]	valid_0's auc: 0.661474
[410]	valid_0's auc: 0.661741
[420]	valid_0's auc: 0.661893
[430]	valid_0's auc: 0.66213
[440]	valid_0's auc: 0.662252
[450]	valid_0's auc: 0.662357
[460]	valid_0's auc: 0.662536
[470]	valid_0's auc: 0.662592
[480]	valid_0's auc: 0.662743
[490]	valid_0's auc: 0.662868
[500]	valid_0's auc: 0.663055
[510]	valid_0's auc: 0.663089
[520]	valid_0's auc: 0.663135
[530]	valid_0's auc: 0.66315
[540]	valid_0's auc: 0.663423
[550]	valid_0's auc: 0.663424
[560]	valid_0's auc: 0.663675
[570]	valid_0's auc: 0.663584
[580]	valid_0's auc: 0.663652
[590]	valid_0's auc: 0.663708
[600]	valid_0's auc: 0.663817
[610]	valid_0's auc: 0.663935
[620]	valid_0's auc: 0.664053
[630]	valid_0's auc: 0.664215
[640]	valid_0's auc: 0.664218
[650]	valid_0's auc: 0.664227
[660]	valid_0's auc: 0.66418
[670]	valid_0's auc: 0.664277
[680]	valid_0's auc: 0.664312
[690]	valid_0's auc: 0.66438
[700]	valid_0's auc: 0.664363
[710]	valid_0's auc: 0.664739
[720]	valid_0's auc: 0.664735
[730]	valid_0's auc: 0.664828
[740]	valid_0's auc: 0.664942
[750]	valid_0's auc: 0.66503
[760]	valid_0's auc: 0.665072
[770]	valid_0's auc: 0.665111
[780]	valid_0's auc: 0.665203
[790]	valid_0's auc: 0.66525
[800]	valid_0's auc: 0.665241
[810]	valid_0's auc: 0.665401
[820]	valid_0's auc: 0.66545
[830]	valid_0's auc: 0.665448
[840]	valid_0's auc: 0.665511
[850]	valid_0's auc: 0.665525
[860]	valid_0's auc: 0.665487
[870]	valid_0's auc: 0.665483
[880]	valid_0's auc: 0.665524
[890]	valid_0's auc: 0.665552
[900]	valid_0's auc: 0.665519
[910]	valid_0's auc: 0.665633
[920]	valid_0's auc: 0.665647
[930]	valid_0's auc: 0.665639
[940]	valid_0's auc: 0.665664
[950]	valid_0's auc: 0.665608
[960]	valid_0's auc: 0.665554
[970]	valid_0's auc: 0.665553
[980]	valid_0's auc: 0.665612
Early stopping, best iteration is:
[937]	valid_0's auc: 0.665675
best score: 0.665675057574
best iteration: 937

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 3 complete in 8m 13s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 4

bagging_seed         : 2
feature_fraction_seed : 2
lambda_l1            : 0.1
boosting             : gbdt
bagging_freq         : 4
feature_fraction     : 0.8
lambda_l2            : 0.1
learning_rate        : 0.2
bagging_fraction     : 0.8
num_leaves           : 511
max_depth            : 10

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.63796
[20]	valid_0's auc: 0.644014
[30]	valid_0's auc: 0.647363
[40]	valid_0's auc: 0.649515
[50]	valid_0's auc: 0.651325
[60]	valid_0's auc: 0.652767
[70]	valid_0's auc: 0.653992
[80]	valid_0's auc: 0.655761
[90]	valid_0's auc: 0.656545
[100]	valid_0's auc: 0.657625
[110]	valid_0's auc: 0.658601
[120]	valid_0's auc: 0.65904
[130]	valid_0's auc: 0.659768
[140]	valid_0's auc: 0.660353
[150]	valid_0's auc: 0.660747
[160]	valid_0's auc: 0.661021
[170]	valid_0's auc: 0.661403
[180]	valid_0's auc: 0.661784
[190]	valid_0's auc: 0.661933
[200]	valid_0's auc: 0.662268
[210]	valid_0's auc: 0.662583
[220]	valid_0's auc: 0.662827
[230]	valid_0's auc: 0.663127
[240]	valid_0's auc: 0.663401
[250]	valid_0's auc: 0.663606
[260]	valid_0's auc: 0.663766
[270]	valid_0's auc: 0.664095
[280]	valid_0's auc: 0.664519
[290]	valid_0's auc: 0.6649
[300]	valid_0's auc: 0.665095
[310]	valid_0's auc: 0.665286
[320]	valid_0's auc: 0.665392
[330]	valid_0's auc: 0.665524
[340]	valid_0's auc: 0.665759
[350]	valid_0's auc: 0.665811
[360]	valid_0's auc: 0.665962
[370]	valid_0's auc: 0.665996
[380]	valid_0's auc: 0.666084
[390]	valid_0's auc: 0.666096
[400]	valid_0's auc: 0.666203
[410]	valid_0's auc: 0.666305
[420]	valid_0's auc: 0.666588
[430]	valid_0's auc: 0.666794
[440]	valid_0's auc: 0.666961
[450]	valid_0's auc: 0.667016
[460]	valid_0's auc: 0.667104
[470]	valid_0's auc: 0.667305
[480]	valid_0's auc: 0.667404
[490]	valid_0's auc: 0.667499
[500]	valid_0's auc: 0.667509
[510]	valid_0's auc: 0.667555
[520]	valid_0's auc: 0.66754
[530]	valid_0's auc: 0.667704
[540]	valid_0's auc: 0.667772
[550]	valid_0's auc: 0.667804
[560]	valid_0's auc: 0.667883
[570]	valid_0's auc: 0.668013
[580]	valid_0's auc: 0.668108
[590]	valid_0's auc: 0.668126
[600]	valid_0's auc: 0.668094
[610]	valid_0's auc: 0.668195
[620]	valid_0's auc: 0.668187
[630]	valid_0's auc: 0.668269
[640]	valid_0's auc: 0.668363
[650]	valid_0's auc: 0.668149
[660]	valid_0's auc: 0.668196
[670]	valid_0's auc: 0.668303
[680]	valid_0's auc: 0.668292
[690]	valid_0's auc: 0.668387
[700]	valid_0's auc: 0.668337
[710]	valid_0's auc: 0.668506
[720]	valid_0's auc: 0.668489
[730]	valid_0's auc: 0.668558
[740]	valid_0's auc: 0.668616
[750]	valid_0's auc: 0.66867
[760]	valid_0's auc: 0.668661
[770]	valid_0's auc: 0.668714
[780]	valid_0's auc: 0.668899
[790]	valid_0's auc: 0.668934
[800]	valid_0's auc: 0.668924
[810]	valid_0's auc: 0.668958
[820]	valid_0's auc: 0.668986
[830]	valid_0's auc: 0.668989
[840]	valid_0's auc: 0.669026
[850]	valid_0's auc: 0.669019
[860]	valid_0's auc: 0.669009
[870]	valid_0's auc: 0.669001
[880]	valid_0's auc: 0.668934
[890]	valid_0's auc: 0.668936
[900]	valid_0's auc: 0.66893
[910]	valid_0's auc: 0.669068
[920]	valid_0's auc: 0.669103
[930]	valid_0's auc: 0.669085
[940]	valid_0's auc: 0.669147
[950]	valid_0's auc: 0.669126
[960]	valid_0's auc: 0.669134
[970]	valid_0's auc: 0.669237
[980]	valid_0's auc: 0.669235
[990]	valid_0's auc: 0.669277
[1000]	valid_0's auc: 0.669255
[1010]	valid_0's auc: 0.669275
[1020]	valid_0's auc: 0.669355
[1030]	valid_0's auc: 0.669385
[1040]	valid_0's auc: 0.669389
[1050]	valid_0's auc: 0.669448
[1060]	valid_0's auc: 0.669489
[1070]	valid_0's auc: 0.669546
[1080]	valid_0's auc: 0.669632
[1090]	valid_0's auc: 0.669639
[1100]	valid_0's auc: 0.66963
[1110]	valid_0's auc: 0.669655
[1120]	valid_0's auc: 0.669783
[1130]	valid_0's auc: 0.669803
[1140]	valid_0's auc: 0.669807
[1150]	valid_0's auc: 0.669835
[1160]	valid_0's auc: 0.669819
[1170]	valid_0's auc: 0.669864
[1180]	valid_0's auc: 0.669861
[1190]	valid_0's auc: 0.669835
[1200]	valid_0's auc: 0.669833
[1210]	valid_0's auc: 0.669868
[1220]	valid_0's auc: 0.669878
[1230]	valid_0's auc: 0.669898
[1240]	valid_0's auc: 0.669963
[1250]	valid_0's auc: 0.669972
[1260]	valid_0's auc: 0.669984
[1270]	valid_0's auc: 0.669992
[1280]	valid_0's auc: 0.669995
[1290]	valid_0's auc: 0.669958
[1300]	valid_0's auc: 0.669988
Early stopping, best iteration is:
[1254]	valid_0's auc: 0.670009
best score: 0.670008987809
best iteration: 1254

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 4 complete in 16m 3s

[timer]: complete in 89m 52s

Process finished with exit code 0
'''
