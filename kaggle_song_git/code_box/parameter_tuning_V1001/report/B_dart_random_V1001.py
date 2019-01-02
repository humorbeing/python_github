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
         'language',
         'artist_name',
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
lr_s = [0.3, 0.1, 0.07, 0.02]
nl_s = [100, 300, 500, 700]
md_s = [-1, 10, 20, 30]


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

while True:
    boosting = b_s[2]
    learning_rate = lr_s[np.random.randint(0, 4)]
    num_leaves = nl_s[np.random.randint(0, 4)]
    max_depth = md_s[np.random.randint(0, 4)]

    params = {
              'boosting': boosting,
              'learning_rate': learning_rate,
              'num_leaves': num_leaves,
              # 'bagging_fraction': bagging_fraction,
              # 'bagging_freq': bagging_freq,
              # 'bagging_seed': bagging_seed,
              # 'feature_fraction': feature_fraction,
              # 'feature_fraction_seed': feature_fraction_seed,
              # 'max_bin': 255,
              'max_depth': max_depth,
              }
    print()
    print('>'*50)
    print('------------Parameters-----------')
    print()
    for dd in params:
        print(dd.ljust(20), ':', params[dd])
    print()
    params['metric'] = 'auc'
    params['max_bin'] = 255
    params['verbose'] = -1
    params['objective'] = 'binary'

    model = lgb.train(params,
                      train_set,
                      num_boost_round=500000,
                      early_stopping_rounds=100,
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_trying_V1001/B_dart_random_V1001.py
What we got:
msno                    object
song_id                 object
source_system_tab       object
source_screen_name      object
source_type             object
target                   uint8
artist_name             object
language              category
dtype: object
number of columns: 8


This rounds guests:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
dtype: object
number of columns: 8

Training...


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

boosting             : dart
learning_rate        : 0.07
max_depth            : 30
num_leaves           : 300

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 100 rounds.
[10]	valid_0's auc: 0.642827
[20]	valid_0's auc: 0.648476
[30]	valid_0's auc: 0.651396
[40]	valid_0's auc: 0.65285
[50]	valid_0's auc: 0.653409
[60]	valid_0's auc: 0.654383
[70]	valid_0's auc: 0.654913
[80]	valid_0's auc: 0.65559
[90]	valid_0's auc: 0.655716
[100]	valid_0's auc: 0.656081
[110]	valid_0's auc: 0.656207
[120]	valid_0's auc: 0.657138
[130]	valid_0's auc: 0.65751
[140]	valid_0's auc: 0.65781
[150]	valid_0's auc: 0.658167
[160]	valid_0's auc: 0.658221
[170]	valid_0's auc: 0.65886
[180]	valid_0's auc: 0.659348
[190]	valid_0's auc: 0.660152
[200]	valid_0's auc: 0.660487
[210]	valid_0's auc: 0.660658
[220]	valid_0's auc: 0.661091
[230]	valid_0's auc: 0.661308
[240]	valid_0's auc: 0.662064
[250]	valid_0's auc: 0.662365
[260]	valid_0's auc: 0.662875
[270]	valid_0's auc: 0.663717
[280]	valid_0's auc: 0.664026
[290]	valid_0's auc: 0.664618
[300]	valid_0's auc: 0.66474
[310]	valid_0's auc: 0.664632
[320]	valid_0's auc: 0.664984
[330]	valid_0's auc: 0.66526
[340]	valid_0's auc: 0.665298
[350]	valid_0's auc: 0.66559
[360]	valid_0's auc: 0.665928
[370]	valid_0's auc: 0.666182
[380]	valid_0's auc: 0.66635
[390]	valid_0's auc: 0.666589
[400]	valid_0's auc: 0.667124
[410]	valid_0's auc: 0.667363
[420]	valid_0's auc: 0.667596
[430]	valid_0's auc: 0.667766
[440]	valid_0's auc: 0.667922
[450]	valid_0's auc: 0.667973
[460]	valid_0's auc: 0.667884
[470]	valid_0's auc: 0.668063
[480]	valid_0's auc: 0.668227
[490]	valid_0's auc: 0.668159
[500]	valid_0's auc: 0.668201
[510]	valid_0's auc: 0.668267
[520]	valid_0's auc: 0.668445
[530]	valid_0's auc: 0.668603
[540]	valid_0's auc: 0.668616
[550]	valid_0's auc: 0.668533
[560]	valid_0's auc: 0.668714
[570]	valid_0's auc: 0.668791
[580]	valid_0's auc: 0.668992
[590]	valid_0's auc: 0.669079
[600]	valid_0's auc: 0.669234
[610]	valid_0's auc: 0.669291
[620]	valid_0's auc: 0.669565
[630]	valid_0's auc: 0.669512
[640]	valid_0's auc: 0.669516
[650]	valid_0's auc: 0.669483
[660]	valid_0's auc: 0.669439
[670]	valid_0's auc: 0.669714
[680]	valid_0's auc: 0.669735
[690]	valid_0's auc: 0.669731
[700]	valid_0's auc: 0.669748
[710]	valid_0's auc: 0.669708
[720]	valid_0's auc: 0.66991
[730]	valid_0's auc: 0.669971
[740]	valid_0's auc: 0.670036
[750]	valid_0's auc: 0.67004
[760]	valid_0's auc: 0.670107
[770]	valid_0's auc: 0.670111
[780]	valid_0's auc: 0.670156
[790]	valid_0's auc: 0.670147
[800]	valid_0's auc: 0.670265
[810]	valid_0's auc: 0.670232
[820]	valid_0's auc: 0.670247
[830]	valid_0's auc: 0.670267
[840]	valid_0's auc: 0.670283
[850]	valid_0's auc: 0.670284
[860]	valid_0's auc: 0.670297
[870]	valid_0's auc: 0.670317
[880]	valid_0's auc: 0.670352
[890]	valid_0's auc: 0.670399
[900]	valid_0's auc: 0.670462
[910]	valid_0's auc: 0.67043
[920]	valid_0's auc: 0.670358
[930]	valid_0's auc: 0.670355
[940]	valid_0's auc: 0.670333
[950]	valid_0's auc: 0.670257
[960]	valid_0's auc: 0.670285
[970]	valid_0's auc: 0.670311
[980]	valid_0's auc: 0.670304
[990]	valid_0's auc: 0.670328
[1000]	valid_0's auc: 0.670376
Early stopping, best iteration is:
[900]	valid_0's auc: 0.670462
best score: 0.670462144001
best iteration: 900

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 612m 45s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

boosting             : dart
learning_rate        : 0.07
max_depth            : 10
num_leaves           : 300

Training until validation scores don't improve for 100 rounds.
[10]	valid_0's auc: 0.624121
[20]	valid_0's auc: 0.630338
[30]	valid_0's auc: 0.63448
[40]	valid_0's auc: 0.636148
[50]	valid_0's auc: 0.636872
[60]	valid_0's auc: 0.637848
[70]	valid_0's auc: 0.638635
[80]	valid_0's auc: 0.639194
[90]	valid_0's auc: 0.639438
[100]	valid_0's auc: 0.639962
[110]	valid_0's auc: 0.639883
[120]	valid_0's auc: 0.640652
[130]	valid_0's auc: 0.6411
[140]	valid_0's auc: 0.641765
[150]	valid_0's auc: 0.641878
[160]	valid_0's auc: 0.641784
[170]	valid_0's auc: 0.642517
[180]	valid_0's auc: 0.643214
[190]	valid_0's auc: 0.644138
[200]	valid_0's auc: 0.644611
[210]	valid_0's auc: 0.644551
[220]	valid_0's auc: 0.645067
[230]	valid_0's auc: 0.645125
[240]	valid_0's auc: 0.646007
[250]	valid_0's auc: 0.646466
[260]	valid_0's auc: 0.646952
[270]	valid_0's auc: 0.647487
[280]	valid_0's auc: 0.647805
[290]	valid_0's auc: 0.648205
[300]	valid_0's auc: 0.648398
[310]	valid_0's auc: 0.648451
[320]	valid_0's auc: 0.648947
[330]	valid_0's auc: 0.649154
[340]	valid_0's auc: 0.649167
[350]	valid_0's auc: 0.649528
[360]	valid_0's auc: 0.649869
[370]	valid_0's auc: 0.64999
[380]	valid_0's auc: 0.650267
[390]	valid_0's auc: 0.650575
[400]	valid_0's auc: 0.650937
[410]	valid_0's auc: 0.651188
[420]	valid_0's auc: 0.651319
[430]	valid_0's auc: 0.651846
[440]	valid_0's auc: 0.652057
[450]	valid_0's auc: 0.652161
[460]	valid_0's auc: 0.652206
[470]	valid_0's auc: 0.652429
[480]	valid_0's auc: 0.652586
[490]	valid_0's auc: 0.652683
[500]	valid_0's auc: 0.652709
[510]	valid_0's auc: 0.652836
[520]	valid_0's auc: 0.652995
[530]	valid_0's auc: 0.653172
[540]	valid_0's auc: 0.653113
[550]	valid_0's auc: 0.653133
[560]	valid_0's auc: 0.653326
[570]	valid_0's auc: 0.653456
[580]	valid_0's auc: 0.653723
[590]	valid_0's auc: 0.653801
[600]	valid_0's auc: 0.653975
[610]	valid_0's auc: 0.654326
[620]	valid_0's auc: 0.654689
[630]	valid_0's auc: 0.654588
[640]	valid_0's auc: 0.654695
[650]	valid_0's auc: 0.654822
[660]	valid_0's auc: 0.654834
[670]	valid_0's auc: 0.655042
[680]	valid_0's auc: 0.655117
[690]	valid_0's auc: 0.655086
[700]	valid_0's auc: 0.655169
[710]	valid_0's auc: 0.655115
[720]	valid_0's auc: 0.655278
[730]	valid_0's auc: 0.65535
[740]	valid_0's auc: 0.655509
[750]	valid_0's auc: 0.655519
[760]	valid_0's auc: 0.655641
[770]	valid_0's auc: 0.655768
[780]	valid_0's auc: 0.655799
[790]	valid_0's auc: 0.655888
[800]	valid_0's auc: 0.656016
[810]	valid_0's auc: 0.655941
[820]	valid_0's auc: 0.656022
[830]	valid_0's auc: 0.656025
[840]	valid_0's auc: 0.656013
[850]	valid_0's auc: 0.656177
[860]	valid_0's auc: 0.656242
[870]	valid_0's auc: 0.656315
[880]	valid_0's auc: 0.656369
[890]	valid_0's auc: 0.656486
[900]	valid_0's auc: 0.656478
[910]	valid_0's auc: 0.656487
[920]	valid_0's auc: 0.656535
[930]	valid_0's auc: 0.656563
[940]	valid_0's auc: 0.6565
[950]	valid_0's auc: 0.656407
[960]	valid_0's auc: 0.656541
[970]	valid_0's auc: 0.656674
[980]	valid_0's auc: 0.656661
[990]	valid_0's auc: 0.656617
[1000]	valid_0's auc: 0.656701
[1010]	valid_0's auc: 0.656715
[1020]	valid_0's auc: 0.656873
[1030]	valid_0's auc: 0.65687
[1040]	valid_0's auc: 0.656871
[1050]	valid_0's auc: 0.657028
[1060]	valid_0's auc: 0.657154
[1070]	valid_0's auc: 0.657137
[1080]	valid_0's auc: 0.65728
[1090]	valid_0's auc: 0.657278
[1100]	valid_0's auc: 0.657301
[1110]	valid_0's auc: 0.657512
[1120]	valid_0's auc: 0.657531
[1130]	valid_0's auc: 0.657515
[1140]	valid_0's auc: 0.657717
[1150]	valid_0's auc: 0.657694
[1160]	valid_0's auc: 0.657798
[1170]	valid_0's auc: 0.657875
[1180]	valid_0's auc: 0.657907
[1190]	valid_0's auc: 0.658054
[1200]	valid_0's auc: 0.658103
[1210]	valid_0's auc: 0.658228
[1220]	valid_0's auc: 0.658223
[1230]	valid_0's auc: 0.65827
[1240]	valid_0's auc: 0.658438
[1250]	valid_0's auc: 0.658518
[1260]	valid_0's auc: 0.658568
[1270]	valid_0's auc: 0.658567
[1280]	valid_0's auc: 0.658757
[1290]	valid_0's auc: 0.65887
[1300]	valid_0's auc: 0.658975
[1310]	valid_0's auc: 0.658993
[1320]	valid_0's auc: 0.659042
[1330]	valid_0's auc: 0.659088
[1340]	valid_0's auc: 0.659145
[1350]	valid_0's auc: 0.659173
[1360]	valid_0's auc: 0.65915
[1370]	valid_0's auc: 0.659262
[1380]	valid_0's auc: 0.659353
[1390]	valid_0's auc: 0.659461
[1400]	valid_0's auc: 0.659482
[1410]	valid_0's auc: 0.659532
[1420]	valid_0's auc: 0.659653
[1430]	valid_0's auc: 0.659701
[1440]	valid_0's auc: 0.659779
[1450]	valid_0's auc: 0.6598
[1460]	valid_0's auc: 0.659923
[1470]	valid_0's auc: 0.659965
[1480]	valid_0's auc: 0.660008
[1490]	valid_0's auc: 0.66009
[1500]	valid_0's auc: 0.660203
[1510]	valid_0's auc: 0.660277
[1520]	valid_0's auc: 0.660422
[1530]	valid_0's auc: 0.660475
[1540]	valid_0's auc: 0.660552
[1550]	valid_0's auc: 0.66061
[1560]	valid_0's auc: 0.660635
[1570]	valid_0's auc: 0.660671
[1580]	valid_0's auc: 0.660693
[1590]	valid_0's auc: 0.660736
[1600]	valid_0's auc: 0.66079
[1610]	valid_0's auc: 0.660824
[1620]	valid_0's auc: 0.660904
[1630]	valid_0's auc: 0.660879
[1640]	valid_0's auc: 0.660872
[1650]	valid_0's auc: 0.660869
[1660]	valid_0's auc: 0.660861
[1670]	valid_0's auc: 0.660945
[1680]	valid_0's auc: 0.660991
[1690]	valid_0's auc: 0.660981
[1700]	valid_0's auc: 0.661047
[1710]	valid_0's auc: 0.661075
[1720]	valid_0's auc: 0.661073
[1730]	valid_0's auc: 0.661153
[1740]	valid_0's auc: 0.661135
[1750]	valid_0's auc: 0.661202
[1760]	valid_0's auc: 0.661188
[1770]	valid_0's auc: 0.661265
[1780]	valid_0's auc: 0.661243
[1790]	valid_0's auc: 0.661243
[1800]	valid_0's auc: 0.66134
[1810]	valid_0's auc: 0.661342
[1820]	valid_0's auc: 0.661429
[1830]	valid_0's auc: 0.661474
[1840]	valid_0's auc: 0.661484
[1850]	valid_0's auc: 0.661589
[1860]	valid_0's auc: 0.661648
[1870]	valid_0's auc: 0.661724
[1880]	valid_0's auc: 0.661811
[1890]	valid_0's auc: 0.661812
[1900]	valid_0's auc: 0.661861
[1910]	valid_0's auc: 0.661881
[1920]	valid_0's auc: 0.661969
[1930]	valid_0's auc: 0.662058
[1940]	valid_0's auc: 0.662111
[1950]	valid_0's auc: 0.662227
[1960]	valid_0's auc: 0.662251
[1970]	valid_0's auc: 0.662293
[1980]	valid_0's auc: 0.662322
[1990]	valid_0's auc: 0.662334
[2000]	valid_0's auc: 0.662327
[2010]	valid_0's auc: 0.66234
[2020]	valid_0's auc: 0.662456
[2030]	valid_0's auc: 0.662451
[2040]	valid_0's auc: 0.662416
[2050]	valid_0's auc: 0.66244
[2060]	valid_0's auc: 0.662448
[2070]	valid_0's auc: 0.662482
[2080]	valid_0's auc: 0.662498
[2090]	valid_0's auc: 0.662549
[2100]	valid_0's auc: 0.662526
[2110]	valid_0's auc: 0.662495
[2120]	valid_0's auc: 0.662506
[2130]	valid_0's auc: 0.662505
[2140]	valid_0's auc: 0.662545
[2150]	valid_0's auc: 0.66253
[2160]	valid_0's auc: 0.662572
[2170]	valid_0's auc: 0.662583
[2180]	valid_0's auc: 0.662571
[2190]	valid_0's auc: 0.662665
[2200]	valid_0's auc: 0.662677
[2210]	valid_0's auc: 0.662695
[2220]	valid_0's auc: 0.662729
[2230]	valid_0's auc: 0.662695
[2240]	valid_0's auc: 0.662772
[2250]	valid_0's auc: 0.662877
[2260]	valid_0's auc: 0.662926
[2270]	valid_0's auc: 0.662886
[2280]	valid_0's auc: 0.662967
[2290]	valid_0's auc: 0.662979
[2300]	valid_0's auc: 0.66304
[2310]	valid_0's auc: 0.663014
[2320]	valid_0's auc: 0.663082
[2330]	valid_0's auc: 0.663183
[2340]	valid_0's auc: 0.663218
[2350]	valid_0's auc: 0.663251
[2360]	valid_0's auc: 0.663302
[2370]	valid_0's auc: 0.663313
[2380]	valid_0's auc: 0.66332
[2390]	valid_0's auc: 0.663297
[2400]	valid_0's auc: 0.663337
[2410]	valid_0's auc: 0.663333
[2420]	valid_0's auc: 0.663333
[2430]	valid_0's auc: 0.663307
[2440]	valid_0's auc: 0.663334
[2450]	valid_0's auc: 0.66334
[2460]	valid_0's auc: 0.663344
[2470]	valid_0's auc: 0.663378
[2480]	valid_0's auc: 0.663399
[2490]	valid_0's auc: 0.663383
[2500]	valid_0's auc: 0.663413
[2510]	valid_0's auc: 0.663414
[2520]	valid_0's auc: 0.66342
[2530]	valid_0's auc: 0.663515
[2540]	valid_0's auc: 0.663526
[2550]	valid_0's auc: 0.663558
[2560]	valid_0's auc: 0.663616
[2570]	valid_0's auc: 0.663624
[2580]	valid_0's auc: 0.663618
[2590]	valid_0's auc: 0.663662
[2600]	valid_0's auc: 0.66366
[2610]	valid_0's auc: 0.663678
[2620]	valid_0's auc: 0.66369
[2630]	valid_0's auc: 0.663775
[2640]	valid_0's auc: 0.663783
[2650]	valid_0's auc: 0.663817
[2660]	valid_0's auc: 0.663889
[2670]	valid_0's auc: 0.663871
[2680]	valid_0's auc: 0.663893
[2690]	valid_0's auc: 0.663934
[2700]	valid_0's auc: 0.663964
[2710]	valid_0's auc: 0.66397
[2720]	valid_0's auc: 0.66395
[2730]	valid_0's auc: 0.663946
[2740]	valid_0's auc: 0.663977
[2750]	valid_0's auc: 0.664026
[2760]	valid_0's auc: 0.664025
[2770]	valid_0's auc: 0.664043
[2780]	valid_0's auc: 0.664078
[2790]	valid_0's auc: 0.664117
[2800]	valid_0's auc: 0.664151
[2810]	valid_0's auc: 0.664178
[2820]	valid_0's auc: 0.664177
[2830]	valid_0's auc: 0.664197
[2840]	valid_0's auc: 0.664236
[2850]	valid_0's auc: 0.664251
[2860]	valid_0's auc: 0.664338
[2870]	valid_0's auc: 0.664412
[2880]	valid_0's auc: 0.664423
[2890]	valid_0's auc: 0.664421
[2900]	valid_0's auc: 0.664415
[2910]	valid_0's auc: 0.664422
[2920]	valid_0's auc: 0.664419
[2930]	valid_0's auc: 0.664452
[2940]	valid_0's auc: 0.664464
[2950]	valid_0's auc: 0.664453
[2960]	valid_0's auc: 0.664487
[2970]	valid_0's auc: 0.664484
[2980]	valid_0's auc: 0.664532
[2990]	valid_0's auc: 0.664525
[3000]	valid_0's auc: 0.66461
[3010]	valid_0's auc: 0.664579
[3020]	valid_0's auc: 0.664599
[3030]	valid_0's auc: 0.6646
[3040]	valid_0's auc: 0.664621
[3050]	valid_0's auc: 0.664599
[3060]	valid_0's auc: 0.664594
[3070]	valid_0's auc: 0.664606
[3080]	valid_0's auc: 0.664605
[3090]	valid_0's auc: 0.664604
[3100]	valid_0's auc: 0.664575
[3110]	valid_0's auc: 0.664584
[3120]	valid_0's auc: 0.664602
[3130]	valid_0's auc: 0.664606
[3140]	valid_0's auc: 0.664644
[3150]	valid_0's auc: 0.664655
[3160]	valid_0's auc: 0.664667
[3170]	valid_0's auc: 0.664706
[3180]	valid_0's auc: 0.664724
[3190]	valid_0's auc: 0.664714
[3200]	valid_0's auc: 0.664677
[3210]	valid_0's auc: 0.66463
[3220]	valid_0's auc: 0.664633
[3230]	valid_0's auc: 0.66464
[3240]	valid_0's auc: 0.664703
[3250]	valid_0's auc: 0.664688
[3260]	valid_0's auc: 0.664667
[3270]	valid_0's auc: 0.664665
Early stopping, best iteration is:
[3178]	valid_0's auc: 0.664728
best score: 0.664728259533
best iteration: 3178

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 2146m 3s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

boosting             : dart
learning_rate        : 0.02
max_depth            : 20
num_leaves           : 500

Training until validation scores don't improve for 100 rounds.
[10]	valid_0's auc: 0.629268
[20]	valid_0's auc: 0.632557
[30]	valid_0's auc: 0.636898
[40]	valid_0's auc: 0.637673
[50]	valid_0's auc: 0.638858
[60]	valid_0's auc: 0.639374
[70]	valid_0's auc: 0.639918
[80]	valid_0's auc: 0.640325
[90]	valid_0's auc: 0.640517
[100]	valid_0's auc: 0.640907
[110]	valid_0's auc: 0.640907
[120]	valid_0's auc: 0.641732
[130]	valid_0's auc: 0.641957
[140]	valid_0's auc: 0.64234
[150]	valid_0's auc: 0.642394
[160]	valid_0's auc: 0.642509
[170]	valid_0's auc: 0.643054
[180]	valid_0's auc: 0.643263
[190]	valid_0's auc: 0.64355
[200]	valid_0's auc: 0.643891
[210]	valid_0's auc: 0.64398
[220]	valid_0's auc: 0.644209
[230]	valid_0's auc: 0.644418
[240]	valid_0's auc: 0.644892
[250]	valid_0's auc: 0.6451
[260]	valid_0's auc: 0.645478
[270]	valid_0's auc: 0.645812
[280]	valid_0's auc: 0.646009
[290]	valid_0's auc: 0.6465
[300]	valid_0's auc: 0.646564
[310]	valid_0's auc: 0.646666
[320]	valid_0's auc: 0.64696
[330]	valid_0's auc: 0.647283
[340]	valid_0's auc: 0.647321
[350]	valid_0's auc: 0.647555
[360]	valid_0's auc: 0.647781
[370]	valid_0's auc: 0.647918
[380]	valid_0's auc: 0.648232
[390]	valid_0's auc: 0.648533
[400]	valid_0's auc: 0.648883
[410]	valid_0's auc: 0.649092
[420]	valid_0's auc: 0.649192
[430]	valid_0's auc: 0.649345
[440]	valid_0's auc: 0.64953
[450]	valid_0's auc: 0.649753
[460]	valid_0's auc: 0.649764
[470]	valid_0's auc: 0.64999
[480]	valid_0's auc: 0.650273
[490]	valid_0's auc: 0.650251
[500]	valid_0's auc: 0.650426
[510]	valid_0's auc: 0.650598
[520]	valid_0's auc: 0.650766
[530]	valid_0's auc: 0.650911
[540]	valid_0's auc: 0.650945
[550]	valid_0's auc: 0.651035
[560]	valid_0's auc: 0.651221
[570]	valid_0's auc: 0.651405
[580]	valid_0's auc: 0.651525
[590]	valid_0's auc: 0.651684
[600]	valid_0's auc: 0.651958
[610]	valid_0's auc: 0.652182
[620]	valid_0's auc: 0.652505
[630]	valid_0's auc: 0.652502
[640]	valid_0's auc: 0.65256
[650]	valid_0's auc: 0.652684
[660]	valid_0's auc: 0.652785
[670]	valid_0's auc: 0.653115
[680]	valid_0's auc: 0.6532
[690]	valid_0's auc: 0.653284
[700]	valid_0's auc: 0.653424
[710]	valid_0's auc: 0.65342
[720]	valid_0's auc: 0.653597
[730]	valid_0's auc: 0.653742
[740]	valid_0's auc: 0.653821
[750]	valid_0's auc: 0.65384
[760]	valid_0's auc: 0.653962
[770]	valid_0's auc: 0.654088
[780]	valid_0's auc: 0.654148
[790]	valid_0's auc: 0.654194
[800]	valid_0's auc: 0.654335
[810]	valid_0's auc: 0.65435
[820]	valid_0's auc: 0.654423
[830]	valid_0's auc: 0.654455
[840]	valid_0's auc: 0.654478
[850]	valid_0's auc: 0.65458
[860]	valid_0's auc: 0.654711
[870]	valid_0's auc: 0.654772
[880]	valid_0's auc: 0.654935
[890]	valid_0's auc: 0.654945
[900]	valid_0's auc: 0.655026
[910]	valid_0's auc: 0.65502
[920]	valid_0's auc: 0.655018
[930]	valid_0's auc: 0.654984
[940]	valid_0's auc: 0.654967
[950]	valid_0's auc: 0.654939
[960]	valid_0's auc: 0.655007
[970]	valid_0's auc: 0.655116
[980]	valid_0's auc: 0.655149
[990]	valid_0's auc: 0.655212
[1000]	valid_0's auc: 0.655326
[1010]	valid_0's auc: 0.655395
[1020]	valid_0's auc: 0.655499
[1030]	valid_0's auc: 0.655486
[1040]	valid_0's auc: 0.655466
[1050]	valid_0's auc: 0.65562
[1060]	valid_0's auc: 0.655761
[1070]	valid_0's auc: 0.655898
[1080]	valid_0's auc: 0.65596
[1090]	valid_0's auc: 0.655981
[1100]	valid_0's auc: 0.65597
[1110]	valid_0's auc: 0.656102
[1120]	valid_0's auc: 0.656172
[1130]	valid_0's auc: 0.656146
[1140]	valid_0's auc: 0.656239
[1150]	valid_0's auc: 0.656233
[1160]	valid_0's auc: 0.656443
[1170]	valid_0's auc: 0.656574
[1180]	valid_0's auc: 0.656704
[1190]	valid_0's auc: 0.656753
[1200]	valid_0's auc: 0.656915
[1210]	valid_0's auc: 0.6571
[1220]	valid_0's auc: 0.657231
[1230]	valid_0's auc: 0.657259
[1240]	valid_0's auc: 0.657287
[1250]	valid_0's auc: 0.657424
[1260]	valid_0's auc: 0.657478
[1270]	valid_0's auc: 0.657474
[1280]	valid_0's auc: 0.657699
[1290]	valid_0's auc: 0.657809
[1300]	valid_0's auc: 0.657954
[1310]	valid_0's auc: 0.658094
[1320]	valid_0's auc: 0.658111
[1330]	valid_0's auc: 0.658148
[1340]	valid_0's auc: 0.658344
[1350]	valid_0's auc: 0.658504
[1360]	valid_0's auc: 0.658545
[1370]	valid_0's auc: 0.65869
[1380]	valid_0's auc: 0.658747
[1390]	valid_0's auc: 0.65874
[1400]	valid_0's auc: 0.658854
[1410]	valid_0's auc: 0.658932
[1420]	valid_0's auc: 0.659003
[1430]	valid_0's auc: 0.659073
[1440]	valid_0's auc: 0.659165
[1450]	valid_0's auc: 0.659338
[1460]	valid_0's auc: 0.65968
[1470]	valid_0's auc: 0.65985
[1480]	valid_0's auc: 0.659936
[1490]	valid_0's auc: 0.660077
[1500]	valid_0's auc: 0.660182
[1510]	valid_0's auc: 0.660377
[1520]	valid_0's auc: 0.66049
[1530]	valid_0's auc: 0.660509
[1540]	valid_0's auc: 0.660642
[1550]	valid_0's auc: 0.660809
[1560]	valid_0's auc: 0.660831
[1570]	valid_0's auc: 0.660961
[1580]	valid_0's auc: 0.660974
[1590]	valid_0's auc: 0.661031
[1600]	valid_0's auc: 0.661092
[1610]	valid_0's auc: 0.661141
[1620]	valid_0's auc: 0.661214
[1630]	valid_0's auc: 0.661185
[1640]	valid_0's auc: 0.661263
[1650]	valid_0's auc: 0.661274
[1660]	valid_0's auc: 0.661349
[1670]	valid_0's auc: 0.661384
[1680]	valid_0's auc: 0.661438
[1690]	valid_0's auc: 0.661454
[1700]	valid_0's auc: 0.661559
[1710]	valid_0's auc: 0.661688
[1720]	valid_0's auc: 0.661706
[1730]	valid_0's auc: 0.661762
[1740]	valid_0's auc: 0.661789
[1750]	valid_0's auc: 0.66186
[1760]	valid_0's auc: 0.661904
[1770]	valid_0's auc: 0.662022
[1780]	valid_0's auc: 0.661982
[1790]	valid_0's auc: 0.662002
[1800]	valid_0's auc: 0.662088
[1810]	valid_0's auc: 0.662063
[1820]	valid_0's auc: 0.662158
[1830]	valid_0's auc: 0.662188
[1840]	valid_0's auc: 0.662297
[1850]	valid_0's auc: 0.662404
[1860]	valid_0's auc: 0.66244
[1870]	valid_0's auc: 0.662557
[1880]	valid_0's auc: 0.662594
[1890]	valid_0's auc: 0.662605
[1900]	valid_0's auc: 0.662628
[1910]	valid_0's auc: 0.662677
[1920]	valid_0's auc: 0.662752
[1930]	valid_0's auc: 0.662739
[1940]	valid_0's auc: 0.662772
[1950]	valid_0's auc: 0.662838
[1960]	valid_0's auc: 0.662872
[1970]	valid_0's auc: 0.662941
[1980]	valid_0's auc: 0.663051
[1990]	valid_0's auc: 0.66307
[2000]	valid_0's auc: 0.663056
[2010]	valid_0's auc: 0.663094
[2020]	valid_0's auc: 0.66312
[2030]	valid_0's auc: 0.66314
[2040]	valid_0's auc: 0.663144
[2050]	valid_0's auc: 0.663176
[2060]	valid_0's auc: 0.663195
[2070]	valid_0's auc: 0.66325
[2080]	valid_0's auc: 0.663334
[2090]	valid_0's auc: 0.663349
[2100]	valid_0's auc: 0.663386
[2110]	valid_0's auc: 0.663412
[2120]	valid_0's auc: 0.663444
[2130]	valid_0's auc: 0.663454
[2140]	valid_0's auc: 0.663475
[2150]	valid_0's auc: 0.663482
[2160]	valid_0's auc: 0.663536
[2170]	valid_0's auc: 0.663557
[2180]	valid_0's auc: 0.663566
[2190]	valid_0's auc: 0.663609
[2200]	valid_0's auc: 0.663596
[2210]	valid_0's auc: 0.663629
[2220]	valid_0's auc: 0.663661
[2230]	valid_0's auc: 0.663662
[2240]	valid_0's auc: 0.663746
[2250]	valid_0's auc: 0.663826
[2260]	valid_0's auc: 0.66385
[2270]	valid_0's auc: 0.663828
[2280]	valid_0's auc: 0.663837
[2290]	valid_0's auc: 0.663837
[2300]	valid_0's auc: 0.663898
[2310]	valid_0's auc: 0.663901
[2320]	valid_0's auc: 0.663951
[2330]	valid_0's auc: 0.663972
[2340]	valid_0's auc: 0.663949
[2350]	valid_0's auc: 0.663968
[2360]	valid_0's auc: 0.663985
[2370]	valid_0's auc: 0.664019
[2380]	valid_0's auc: 0.664015
[2390]	valid_0's auc: 0.663991
[2400]	valid_0's auc: 0.664032
[2410]	valid_0's auc: 0.664062
[2420]	valid_0's auc: 0.664083
[2430]	valid_0's auc: 0.664095
[2440]	valid_0's auc: 0.664141
[2450]	valid_0's auc: 0.664151
[2460]	valid_0's auc: 0.664184
[2470]	valid_0's auc: 0.664218
[2480]	valid_0's auc: 0.66426
[2490]	valid_0's auc: 0.664295
[2500]	valid_0's auc: 0.664354
[2510]	valid_0's auc: 0.664358

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''