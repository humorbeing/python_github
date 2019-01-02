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

num_boost_round = 500000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 511
max_depth = -1

max_bin = 255
lambda_l1 = 0
lambda_l2 = 0.2


bagging_fraction = 0.9
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.9
feature_fraction_seed = 2

b_s = ['gbdt', 'rf', 'dart', 'goss']
lr_s = [0.05, 0.03, 0.02, 0.3, 0.1]
nl_s = [1023,  1023,  511, 511, 511]
md_s = [  -1,    10,   11,  -1, 10]
l2_s = [   0,  0.4,  0.1, 2, 0.3]
l1_s = [   0,    0,    0, 0, 0.3]
mb_s = [ 511,  511,  255, 255, 127]


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
        'max_depth': max_depth,

        'max_bin': max_bin,
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
    time_elapsed = time.time() - since
    print('[timer]: complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    since = time.time()

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1004.py
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

bagging_fraction     : 0.9
bagging_seed         : 2
lambda_l1            : 0
bagging_freq         : 2
max_depth            : -1
boosting             : gbdt
learning_rate        : 0.05
max_bin              : 511
num_leaves           : 1023
lambda_l2            : 0
feature_fraction     : 0.9
feature_fraction_seed : 2

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:648: LGBMDeprecationWarning: The `max_bin` parameter is deprecated and will be removed in 2.0.12 version. Please use `params` to pass this parameter.
  'Please use `params` to pass this parameter.', LGBMDeprecationWarning)
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.662822
[20]	valid_0's auc: 0.6668
[30]	valid_0's auc: 0.669465
[40]	valid_0's auc: 0.671841
[50]	valid_0's auc: 0.67335
[60]	valid_0's auc: 0.674758
[70]	valid_0's auc: 0.675863
[80]	valid_0's auc: 0.676456
[90]	valid_0's auc: 0.676982
[100]	valid_0's auc: 0.677306
[110]	valid_0's auc: 0.677447
[120]	valid_0's auc: 0.677593
[130]	valid_0's auc: 0.677666
[140]	valid_0's auc: 0.677809
[150]	valid_0's auc: 0.677947
[160]	valid_0's auc: 0.677872
[170]	valid_0's auc: 0.677752
[180]	valid_0's auc: 0.677638
[190]	valid_0's auc: 0.677576
[200]	valid_0's auc: 0.677562
Early stopping, best iteration is:
[150]	valid_0's auc: 0.677947
best score: 0.677946699266
best iteration: 150

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 35m 30s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

bagging_fraction     : 0.9
bagging_seed         : 2
lambda_l1            : 0
bagging_freq         : 2
max_depth            : 10
boosting             : gbdt
learning_rate        : 0.03
max_bin              : 511
num_leaves           : 1023
lambda_l2            : 0.4
feature_fraction     : 0.9
feature_fraction_seed : 2

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.625744
[20]	valid_0's auc: 0.631693
[30]	valid_0's auc: 0.633774
[40]	valid_0's auc: 0.635437
[50]	valid_0's auc: 0.637627
[60]	valid_0's auc: 0.639676
[70]	valid_0's auc: 0.641407
[80]	valid_0's auc: 0.642896
[90]	valid_0's auc: 0.643868
[100]	valid_0's auc: 0.645009
[110]	valid_0's auc: 0.646153
[120]	valid_0's auc: 0.646876
[130]	valid_0's auc: 0.647517
[140]	valid_0's auc: 0.64811
[150]	valid_0's auc: 0.648497
[160]	valid_0's auc: 0.648974
[170]	valid_0's auc: 0.649407
[180]	valid_0's auc: 0.649883
[190]	valid_0's auc: 0.650192
[200]	valid_0's auc: 0.650578
[210]	valid_0's auc: 0.650829
[220]	valid_0's auc: 0.651076
[230]	valid_0's auc: 0.65139
[240]	valid_0's auc: 0.651686
[250]	valid_0's auc: 0.651918
[260]	valid_0's auc: 0.652225
[270]	valid_0's auc: 0.652465
[280]	valid_0's auc: 0.652727
[290]	valid_0's auc: 0.652959
[300]	valid_0's auc: 0.65313
[310]	valid_0's auc: 0.653385
[320]	valid_0's auc: 0.653627
[330]	valid_0's auc: 0.653829
[340]	valid_0's auc: 0.654045
[350]	valid_0's auc: 0.654263
[360]	valid_0's auc: 0.654508
[370]	valid_0's auc: 0.654715
[380]	valid_0's auc: 0.654901
[390]	valid_0's auc: 0.655108
[400]	valid_0's auc: 0.655284
[410]	valid_0's auc: 0.655539
[420]	valid_0's auc: 0.655748
[430]	valid_0's auc: 0.65598
[440]	valid_0's auc: 0.656131
[450]	valid_0's auc: 0.65633
[460]	valid_0's auc: 0.656503
[470]	valid_0's auc: 0.656618
[480]	valid_0's auc: 0.656818
[490]	valid_0's auc: 0.656959
[500]	valid_0's auc: 0.657058
[510]	valid_0's auc: 0.65721
[520]	valid_0's auc: 0.657337
[530]	valid_0's auc: 0.657497
[540]	valid_0's auc: 0.657599
[550]	valid_0's auc: 0.657723
[560]	valid_0's auc: 0.657869
[570]	valid_0's auc: 0.657996
[580]	valid_0's auc: 0.658168
[590]	valid_0's auc: 0.658328
[600]	valid_0's auc: 0.658444
[610]	valid_0's auc: 0.658515
[620]	valid_0's auc: 0.658637
[630]	valid_0's auc: 0.658762
[640]	valid_0's auc: 0.658867
[650]	valid_0's auc: 0.658982
[660]	valid_0's auc: 0.659117
[670]	valid_0's auc: 0.659226
[680]	valid_0's auc: 0.659316
[690]	valid_0's auc: 0.659421
[700]	valid_0's auc: 0.659526
[710]	valid_0's auc: 0.659619
[720]	valid_0's auc: 0.659747
[730]	valid_0's auc: 0.659887
[740]	valid_0's auc: 0.66
[750]	valid_0's auc: 0.660124
[760]	valid_0's auc: 0.660244
[770]	valid_0's auc: 0.660362
[780]	valid_0's auc: 0.660465
[790]	valid_0's auc: 0.660542
[800]	valid_0's auc: 0.660685
[810]	valid_0's auc: 0.660766
[820]	valid_0's auc: 0.660845
[830]	valid_0's auc: 0.660944
[840]	valid_0's auc: 0.661035
[850]	valid_0's auc: 0.661118
[860]	valid_0's auc: 0.661205
[870]	valid_0's auc: 0.661291
[880]	valid_0's auc: 0.661382
[890]	valid_0's auc: 0.661449
[900]	valid_0's auc: 0.661533
[910]	valid_0's auc: 0.661593
[920]	valid_0's auc: 0.661651
[930]	valid_0's auc: 0.661745
[940]	valid_0's auc: 0.661798
[950]	valid_0's auc: 0.661871
[960]	valid_0's auc: 0.661938
[970]	valid_0's auc: 0.662014
[980]	valid_0's auc: 0.662087
[990]	valid_0's auc: 0.662171
[1000]	valid_0's auc: 0.662236
[1010]	valid_0's auc: 0.662285
[1020]	valid_0's auc: 0.662403
[1030]	valid_0's auc: 0.662468
[1040]	valid_0's auc: 0.66255
[1050]	valid_0's auc: 0.662604
[1060]	valid_0's auc: 0.66265
[1070]	valid_0's auc: 0.662745
[1080]	valid_0's auc: 0.662788
[1090]	valid_0's auc: 0.662876
[1100]	valid_0's auc: 0.662972
[1110]	valid_0's auc: 0.663045
[1120]	valid_0's auc: 0.663093
[1130]	valid_0's auc: 0.66314
[1140]	valid_0's auc: 0.663193
[1150]	valid_0's auc: 0.663227
[1160]	valid_0's auc: 0.66331
[1170]	valid_0's auc: 0.663376
[1180]	valid_0's auc: 0.663403
[1190]	valid_0's auc: 0.663475
[1200]	valid_0's auc: 0.663536
[1210]	valid_0's auc: 0.663604
[1220]	valid_0's auc: 0.663712
[1230]	valid_0's auc: 0.663769
[1240]	valid_0's auc: 0.66384
[1250]	valid_0's auc: 0.663909
[1260]	valid_0's auc: 0.663957
[1270]	valid_0's auc: 0.664008
[1280]	valid_0's auc: 0.66405
[1290]	valid_0's auc: 0.664113
[1300]	valid_0's auc: 0.664165
[1310]	valid_0's auc: 0.664243
[1320]	valid_0's auc: 0.664292
[1330]	valid_0's auc: 0.664336
[1340]	valid_0's auc: 0.664377
[1350]	valid_0's auc: 0.664426
[1360]	valid_0's auc: 0.664473
[1370]	valid_0's auc: 0.664539
[1380]	valid_0's auc: 0.664575
[1390]	valid_0's auc: 0.664688
[1400]	valid_0's auc: 0.664744
[1410]	valid_0's auc: 0.664769
[1420]	valid_0's auc: 0.664818
[1430]	valid_0's auc: 0.66487
[1440]	valid_0's auc: 0.664928
[1450]	valid_0's auc: 0.664973
[1460]	valid_0's auc: 0.665011
[1470]	valid_0's auc: 0.66505
[1480]	valid_0's auc: 0.665088
[1490]	valid_0's auc: 0.665161
[1500]	valid_0's auc: 0.665195
[1510]	valid_0's auc: 0.665222
[1520]	valid_0's auc: 0.665284
[1530]	valid_0's auc: 0.665313
[1540]	valid_0's auc: 0.665378
[1550]	valid_0's auc: 0.665432
[1560]	valid_0's auc: 0.665463
[1570]	valid_0's auc: 0.66549
[1580]	valid_0's auc: 0.665548
[1590]	valid_0's auc: 0.665589
[1600]	valid_0's auc: 0.665628
[1610]	valid_0's auc: 0.665659
[1620]	valid_0's auc: 0.665701
[1630]	valid_0's auc: 0.665733
[1640]	valid_0's auc: 0.665826
[1650]	valid_0's auc: 0.665859
[1660]	valid_0's auc: 0.665909
[1670]	valid_0's auc: 0.665966
[1680]	valid_0's auc: 0.666002
[1690]	valid_0's auc: 0.666058
[1700]	valid_0's auc: 0.666091
[1710]	valid_0's auc: 0.666113
[1720]	valid_0's auc: 0.666149
[1730]	valid_0's auc: 0.666181
[1740]	valid_0's auc: 0.666209
[1750]	valid_0's auc: 0.666272
[1760]	valid_0's auc: 0.666312
[1770]	valid_0's auc: 0.666344
[1780]	valid_0's auc: 0.666369
[1790]	valid_0's auc: 0.666403
[1800]	valid_0's auc: 0.666464
[1810]	valid_0's auc: 0.666483
[1820]	valid_0's auc: 0.666515
[1830]	valid_0's auc: 0.666548
[1840]	valid_0's auc: 0.666586
[1850]	valid_0's auc: 0.666618
[1860]	valid_0's auc: 0.666668
[1870]	valid_0's auc: 0.666707
[1880]	valid_0's auc: 0.666743
[1890]	valid_0's auc: 0.666808
[1900]	valid_0's auc: 0.666836
[1910]	valid_0's auc: 0.666884
[1920]	valid_0's auc: 0.666927
[1930]	valid_0's auc: 0.666981
[1940]	valid_0's auc: 0.667015
[1950]	valid_0's auc: 0.66704
[1960]	valid_0's auc: 0.667085
[1970]	valid_0's auc: 0.667131
[1980]	valid_0's auc: 0.667169
[1990]	valid_0's auc: 0.667215
[2000]	valid_0's auc: 0.667239
[2010]	valid_0's auc: 0.667257
[2020]	valid_0's auc: 0.667277
[2030]	valid_0's auc: 0.6673
[2040]	valid_0's auc: 0.667314
[2050]	valid_0's auc: 0.667375
[2060]	valid_0's auc: 0.667429
[2070]	valid_0's auc: 0.66746
[2080]	valid_0's auc: 0.667469
[2090]	valid_0's auc: 0.667489
[2100]	valid_0's auc: 0.667532
[2110]	valid_0's auc: 0.667588
[2120]	valid_0's auc: 0.667638
[2130]	valid_0's auc: 0.667665
[2140]	valid_0's auc: 0.667693
[2150]	valid_0's auc: 0.667717
[2160]	valid_0's auc: 0.667733
[2170]	valid_0's auc: 0.667772
[2180]	valid_0's auc: 0.667815
[2190]	valid_0's auc: 0.66784
'''
