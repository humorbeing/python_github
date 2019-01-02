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
# lr_s = [0.3, 10, 1000, 1000000, 100000000000, 0.001]
# lr_s = [20, 19, 18, 17, 16, 15]
nl_s = [25, 100, 200, 300]
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

train_set = lgb.Dataset(X_tr, Y_tr, free_raw_data=False)
val_set = lgb.Dataset(X_val, Y_val, free_raw_data=False)
del X_tr, Y_tr, X_val, Y_val

print('Training...')
print()

runs = 0


def ccc(x):
    # print(x)
    # print(lr_s[runs])
    return lr_s[runs]


while True:
    boosting = b_s[0]
    learning_rate = lr_s[np.random.randint(0, 4)]
    num_leaves = nl_s[np.random.randint(0, 4)]
    max_depth = md_s[np.random.randint(0, 4)]
    # num_leaves = nl_s[3]
    # max_depth = md_s[3]
    # lambda_l1 = np.random.random()
    # lambda_l2 = np.random.random()
    params = {
              # 'objective': 'binary',
              # 'metric': 'auc',
              'boosting': boosting,
              'learning_rate': learning_rate,
              # 'verbose': -1,
              'num_leaves': num_leaves,

              # 'bagging_fraction': bagging_fraction,
              # 'bagging_freq': bagging_freq,
              # 'bagging_seed': bagging_seed,
              # 'feature_fraction': feature_fraction,
              # 'feature_fraction_seed': feature_fraction_seed,
              # 'max_bin': 255,
              'max_depth': max_depth,
              # 'min_data': 500,
              # 'min_hessian': 0.05,
              # 'num_rounds': 500,
              # "min_data_in_leaf": 1,
              # 'min_data': 1,
              # 'min_data_in_bin': 1,
              # 'lambda_l2': lambda_l2,
              # 'lambda_l1': lambda_l1

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
    # params['learning_rate'] = lr_s[0]
    # print('learning rate:',lr_s[0])
    model = lgb.train(params,
                      train_set,
                      num_boost_round=50000,
                      early_stopping_rounds=50,
                      # learning_rates=ccc,
                      valid_sets=val_set,
                      verbose_eval=10,
                      )

    print(model.best_score['valid_0']['auc'])
    # print(type(model.best_iteration))
    print(model.best_iteration)
    print()
    print('<'*50)
# li = model.eval_valid()
# print('len list:', len(li))
# print('max list:', max(li))
del train_set, val_set
print()
# print('complete on:', w)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/playground_V1006/B_training_V1305.py
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

num_leaves           : 100
max_depth            : 30
learning_rate        : 0.07
boosting             : gbdt

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.641154
[20]	valid_0's auc: 0.645619
[30]	valid_0's auc: 0.649525
[40]	valid_0's auc: 0.652889
[50]	valid_0's auc: 0.655558
[60]	valid_0's auc: 0.657224
[70]	valid_0's auc: 0.65845
[80]	valid_0's auc: 0.659231
[90]	valid_0's auc: 0.660022
[100]	valid_0's auc: 0.66076
[110]	valid_0's auc: 0.661773
[120]	valid_0's auc: 0.662286
[130]	valid_0's auc: 0.662923
[140]	valid_0's auc: 0.663437
[150]	valid_0's auc: 0.663908
[160]	valid_0's auc: 0.664201
[170]	valid_0's auc: 0.664538
[180]	valid_0's auc: 0.664904
[190]	valid_0's auc: 0.665239
[200]	valid_0's auc: 0.665583
[210]	valid_0's auc: 0.66588
[220]	valid_0's auc: 0.666157
[230]	valid_0's auc: 0.666465
[240]	valid_0's auc: 0.666672
[250]	valid_0's auc: 0.666942
[260]	valid_0's auc: 0.6672
[270]	valid_0's auc: 0.667478
[280]	valid_0's auc: 0.667724
[290]	valid_0's auc: 0.667922
[300]	valid_0's auc: 0.668043
[310]	valid_0's auc: 0.668281
[320]	valid_0's auc: 0.668422
[330]	valid_0's auc: 0.668575
[340]	valid_0's auc: 0.668719
[350]	valid_0's auc: 0.668809
[360]	valid_0's auc: 0.668929
[370]	valid_0's auc: 0.66906
[380]	valid_0's auc: 0.669152
[390]	valid_0's auc: 0.669213
[400]	valid_0's auc: 0.669294
[410]	valid_0's auc: 0.669383
[420]	valid_0's auc: 0.669405
[430]	valid_0's auc: 0.669462
[440]	valid_0's auc: 0.669617
[450]	valid_0's auc: 0.66971
[460]	valid_0's auc: 0.66976
[470]	valid_0's auc: 0.669834
[480]	valid_0's auc: 0.669894
[490]	valid_0's auc: 0.669966
[500]	valid_0's auc: 0.66999
[510]	valid_0's auc: 0.669995
[520]	valid_0's auc: 0.670067
[530]	valid_0's auc: 0.670154
[540]	valid_0's auc: 0.670145
[550]	valid_0's auc: 0.670187
[560]	valid_0's auc: 0.670204
[570]	valid_0's auc: 0.670243
[580]	valid_0's auc: 0.670248
[590]	valid_0's auc: 0.670325
[600]	valid_0's auc: 0.670338
[610]	valid_0's auc: 0.67036
[620]	valid_0's auc: 0.670397
[630]	valid_0's auc: 0.6704
[640]	valid_0's auc: 0.670394
[650]	valid_0's auc: 0.670381
[660]	valid_0's auc: 0.6704
[670]	valid_0's auc: 0.67046
[680]	valid_0's auc: 0.670462
[690]	valid_0's auc: 0.670485
[700]	valid_0's auc: 0.670729
[710]	valid_0's auc: 0.670776
[720]	valid_0's auc: 0.670786
[730]	valid_0's auc: 0.670777
[740]	valid_0's auc: 0.670843
[750]	valid_0's auc: 0.670848
[760]	valid_0's auc: 0.670833
[770]	valid_0's auc: 0.670859
[780]	valid_0's auc: 0.670878
[790]	valid_0's auc: 0.670963
[800]	valid_0's auc: 0.670944
[810]	valid_0's auc: 0.670937
[820]	valid_0's auc: 0.670946
[830]	valid_0's auc: 0.670973
[840]	valid_0's auc: 0.671059
[850]	valid_0's auc: 0.671066
[860]	valid_0's auc: 0.671105
[870]	valid_0's auc: 0.671105
[880]	valid_0's auc: 0.671162
[890]	valid_0's auc: 0.671174
[900]	valid_0's auc: 0.671173
[910]	valid_0's auc: 0.671184
[920]	valid_0's auc: 0.671189
[930]	valid_0's auc: 0.671211
[940]	valid_0's auc: 0.671228
[950]	valid_0's auc: 0.671253
[960]	valid_0's auc: 0.671273
[970]	valid_0's auc: 0.671262
[980]	valid_0's auc: 0.671248
[990]	valid_0's auc: 0.671255
[1000]	valid_0's auc: 0.671279
Early stopping, best iteration is:
[954]	valid_0's auc: 0.671292
0.67129151292
954

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 25
max_depth            : 10
learning_rate        : 0.02
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.61998
[20]	valid_0's auc: 0.622021
[30]	valid_0's auc: 0.624149
[40]	valid_0's auc: 0.627666
[50]	valid_0's auc: 0.628384
[60]	valid_0's auc: 0.629244
[70]	valid_0's auc: 0.630522
[80]	valid_0's auc: 0.632098
[90]	valid_0's auc: 0.63304
[100]	valid_0's auc: 0.633891
[110]	valid_0's auc: 0.634494
[120]	valid_0's auc: 0.63522
[130]	valid_0's auc: 0.636082
[140]	valid_0's auc: 0.637035
[150]	valid_0's auc: 0.638018
[160]	valid_0's auc: 0.638632
[170]	valid_0's auc: 0.639035
[180]	valid_0's auc: 0.639466
[190]	valid_0's auc: 0.639841
[200]	valid_0's auc: 0.640289
[210]	valid_0's auc: 0.640591
[220]	valid_0's auc: 0.640826
[230]	valid_0's auc: 0.641137
[240]	valid_0's auc: 0.641478
[250]	valid_0's auc: 0.641841
[260]	valid_0's auc: 0.642208
[270]	valid_0's auc: 0.642481
[280]	valid_0's auc: 0.642733
[290]	valid_0's auc: 0.642943
[300]	valid_0's auc: 0.643166
[310]	valid_0's auc: 0.643353
[320]	valid_0's auc: 0.643507
[330]	valid_0's auc: 0.643723
[340]	valid_0's auc: 0.643963
[350]	valid_0's auc: 0.644135
[360]	valid_0's auc: 0.644293
[370]	valid_0's auc: 0.64446
[380]	valid_0's auc: 0.644614
[390]	valid_0's auc: 0.644816
[400]	valid_0's auc: 0.645038
[410]	valid_0's auc: 0.645258
[420]	valid_0's auc: 0.645431
[430]	valid_0's auc: 0.645652
[440]	valid_0's auc: 0.645909
[450]	valid_0's auc: 0.646142
[460]	valid_0's auc: 0.646356
[470]	valid_0's auc: 0.646479
[480]	valid_0's auc: 0.646688
[490]	valid_0's auc: 0.646875
[500]	valid_0's auc: 0.647029
[510]	valid_0's auc: 0.64725
[520]	valid_0's auc: 0.647503
[530]	valid_0's auc: 0.647687
[540]	valid_0's auc: 0.647842
[550]	valid_0's auc: 0.648007
[560]	valid_0's auc: 0.648254
[570]	valid_0's auc: 0.648394
[580]	valid_0's auc: 0.64849
[590]	valid_0's auc: 0.64861
[600]	valid_0's auc: 0.648733
[610]	valid_0's auc: 0.648844
[620]	valid_0's auc: 0.648948
[630]	valid_0's auc: 0.64907
[640]	valid_0's auc: 0.649213
[650]	valid_0's auc: 0.6493
[660]	valid_0's auc: 0.64941
[670]	valid_0's auc: 0.649517
[680]	valid_0's auc: 0.649674
[690]	valid_0's auc: 0.649801
[700]	valid_0's auc: 0.650045
[710]	valid_0's auc: 0.650184
[720]	valid_0's auc: 0.650331
[730]	valid_0's auc: 0.650439
[740]	valid_0's auc: 0.650518
[750]	valid_0's auc: 0.650622
[760]	valid_0's auc: 0.650743
[770]	valid_0's auc: 0.650844
[780]	valid_0's auc: 0.650939
[790]	valid_0's auc: 0.651046
[800]	valid_0's auc: 0.651127
[810]	valid_0's auc: 0.651228
[820]	valid_0's auc: 0.651328
[830]	valid_0's auc: 0.651434
[840]	valid_0's auc: 0.651515
[850]	valid_0's auc: 0.651611
[860]	valid_0's auc: 0.651731
[870]	valid_0's auc: 0.651841
[880]	valid_0's auc: 0.651983
[890]	valid_0's auc: 0.652105
[900]	valid_0's auc: 0.652217
[910]	valid_0's auc: 0.65231
[920]	valid_0's auc: 0.652428
[930]	valid_0's auc: 0.652521
[940]	valid_0's auc: 0.652617
[950]	valid_0's auc: 0.652745
[960]	valid_0's auc: 0.652843
[970]	valid_0's auc: 0.652957
[980]	valid_0's auc: 0.653048
[990]	valid_0's auc: 0.653188
[1000]	valid_0's auc: 0.653294
[1010]	valid_0's auc: 0.653383
[1020]	valid_0's auc: 0.653612
[1030]	valid_0's auc: 0.653817
[1040]	valid_0's auc: 0.65395
[1050]	valid_0's auc: 0.654162
[1060]	valid_0's auc: 0.654315
[1070]	valid_0's auc: 0.654454
[1080]	valid_0's auc: 0.654518
[1090]	valid_0's auc: 0.654599
[1100]	valid_0's auc: 0.654667
[1110]	valid_0's auc: 0.654761
[1120]	valid_0's auc: 0.654846
[1130]	valid_0's auc: 0.654896
[1140]	valid_0's auc: 0.654957
[1150]	valid_0's auc: 0.655028
[1160]	valid_0's auc: 0.655092
[1170]	valid_0's auc: 0.655147
[1180]	valid_0's auc: 0.655229
[1190]	valid_0's auc: 0.655287
[1200]	valid_0's auc: 0.65534
[1210]	valid_0's auc: 0.655388
[1220]	valid_0's auc: 0.655429
[1230]	valid_0's auc: 0.655496
[1240]	valid_0's auc: 0.655549
[1250]	valid_0's auc: 0.655596
[1260]	valid_0's auc: 0.655656
[1270]	valid_0's auc: 0.655717
[1280]	valid_0's auc: 0.655786
[1290]	valid_0's auc: 0.655858
[1300]	valid_0's auc: 0.655937
[1310]	valid_0's auc: 0.655994
[1320]	valid_0's auc: 0.656061
[1330]	valid_0's auc: 0.656169
[1340]	valid_0's auc: 0.656207
[1350]	valid_0's auc: 0.656255
[1360]	valid_0's auc: 0.656309
[1370]	valid_0's auc: 0.656352
[1380]	valid_0's auc: 0.656408
[1390]	valid_0's auc: 0.656457
[1400]	valid_0's auc: 0.656502
[1410]	valid_0's auc: 0.656548
[1420]	valid_0's auc: 0.656616
[1430]	valid_0's auc: 0.65666
[1440]	valid_0's auc: 0.656709
[1450]	valid_0's auc: 0.656756
[1460]	valid_0's auc: 0.656806
[1470]	valid_0's auc: 0.656863
[1480]	valid_0's auc: 0.656926
[1490]	valid_0's auc: 0.657007
[1500]	valid_0's auc: 0.657081
[1510]	valid_0's auc: 0.657119
[1520]	valid_0's auc: 0.657219
[1530]	valid_0's auc: 0.657292
[1540]	valid_0's auc: 0.657338
[1550]	valid_0's auc: 0.657401
[1560]	valid_0's auc: 0.657457
[1570]	valid_0's auc: 0.657523
[1580]	valid_0's auc: 0.657572
[1590]	valid_0's auc: 0.657661
[1600]	valid_0's auc: 0.657717
[1610]	valid_0's auc: 0.657793
[1620]	valid_0's auc: 0.657831
[1630]	valid_0's auc: 0.657871
[1640]	valid_0's auc: 0.657957
[1650]	valid_0's auc: 0.658015
[1660]	valid_0's auc: 0.658074
[1670]	valid_0's auc: 0.658133
[1680]	valid_0's auc: 0.65819
[1690]	valid_0's auc: 0.658236
[1700]	valid_0's auc: 0.658293
[1710]	valid_0's auc: 0.658343
[1720]	valid_0's auc: 0.658396
[1730]	valid_0's auc: 0.658437
[1740]	valid_0's auc: 0.658495
[1750]	valid_0's auc: 0.658546
[1760]	valid_0's auc: 0.658624
[1770]	valid_0's auc: 0.658686
[1780]	valid_0's auc: 0.658835
[1790]	valid_0's auc: 0.658874
[1800]	valid_0's auc: 0.658915
[1810]	valid_0's auc: 0.658989
[1820]	valid_0's auc: 0.659048
[1830]	valid_0's auc: 0.659093
[1840]	valid_0's auc: 0.659134
[1850]	valid_0's auc: 0.659189
[1860]	valid_0's auc: 0.659236
[1870]	valid_0's auc: 0.659281
[1880]	valid_0's auc: 0.659381
[1890]	valid_0's auc: 0.6594
[1900]	valid_0's auc: 0.659448
[1910]	valid_0's auc: 0.659478
[1920]	valid_0's auc: 0.65952
[1930]	valid_0's auc: 0.659569
[1940]	valid_0's auc: 0.659603
[1950]	valid_0's auc: 0.659649
[1960]	valid_0's auc: 0.659672
[1970]	valid_0's auc: 0.659716
[1980]	valid_0's auc: 0.65974
[1990]	valid_0's auc: 0.659789
[2000]	valid_0's auc: 0.659828
[2010]	valid_0's auc: 0.659888
[2020]	valid_0's auc: 0.659962
[2030]	valid_0's auc: 0.659996
[2040]	valid_0's auc: 0.660027
[2050]	valid_0's auc: 0.660054
[2060]	valid_0's auc: 0.660075
[2070]	valid_0's auc: 0.660113
[2080]	valid_0's auc: 0.660175
[2090]	valid_0's auc: 0.660236
[2100]	valid_0's auc: 0.660254
[2110]	valid_0's auc: 0.660287
[2120]	valid_0's auc: 0.660314
[2130]	valid_0's auc: 0.660345
[2140]	valid_0's auc: 0.6604
[2150]	valid_0's auc: 0.660462
[2160]	valid_0's auc: 0.660506
[2170]	valid_0's auc: 0.660523
[2180]	valid_0's auc: 0.660564
[2190]	valid_0's auc: 0.6606
[2200]	valid_0's auc: 0.660619
[2210]	valid_0's auc: 0.660642
[2220]	valid_0's auc: 0.660665
[2230]	valid_0's auc: 0.660714
[2240]	valid_0's auc: 0.660746
[2250]	valid_0's auc: 0.660767
[2260]	valid_0's auc: 0.660791
[2270]	valid_0's auc: 0.660841
[2280]	valid_0's auc: 0.660859
[2290]	valid_0's auc: 0.660896
[2300]	valid_0's auc: 0.660942
[2310]	valid_0's auc: 0.660983
[2320]	valid_0's auc: 0.661015
[2330]	valid_0's auc: 0.661049
[2340]	valid_0's auc: 0.661093
[2350]	valid_0's auc: 0.66113
[2360]	valid_0's auc: 0.661165
[2370]	valid_0's auc: 0.661203
[2380]	valid_0's auc: 0.661227
[2390]	valid_0's auc: 0.661264
[2400]	valid_0's auc: 0.661288
[2410]	valid_0's auc: 0.66132
[2420]	valid_0's auc: 0.66134
[2430]	valid_0's auc: 0.661358
[2440]	valid_0's auc: 0.661388
[2450]	valid_0's auc: 0.661418
[2460]	valid_0's auc: 0.661454
[2470]	valid_0's auc: 0.66148
[2480]	valid_0's auc: 0.661527
[2490]	valid_0's auc: 0.661545
[2500]	valid_0's auc: 0.66157
[2510]	valid_0's auc: 0.661594
[2520]	valid_0's auc: 0.661622
[2530]	valid_0's auc: 0.66164
[2540]	valid_0's auc: 0.661674
[2550]	valid_0's auc: 0.661693
[2560]	valid_0's auc: 0.661711
[2570]	valid_0's auc: 0.661747
[2580]	valid_0's auc: 0.661773
[2590]	valid_0's auc: 0.6618
[2600]	valid_0's auc: 0.661825
[2610]	valid_0's auc: 0.661874
[2620]	valid_0's auc: 0.661932
[2630]	valid_0's auc: 0.661965
[2640]	valid_0's auc: 0.661988
[2650]	valid_0's auc: 0.662017
[2660]	valid_0's auc: 0.662047
[2670]	valid_0's auc: 0.662076
[2680]	valid_0's auc: 0.662082
[2690]	valid_0's auc: 0.662107
[2700]	valid_0's auc: 0.662129
[2710]	valid_0's auc: 0.662166
[2720]	valid_0's auc: 0.662221
[2730]	valid_0's auc: 0.662237
[2740]	valid_0's auc: 0.662258
[2750]	valid_0's auc: 0.662279
[2760]	valid_0's auc: 0.66229
[2770]	valid_0's auc: 0.662314
[2780]	valid_0's auc: 0.66234
[2790]	valid_0's auc: 0.662351
[2800]	valid_0's auc: 0.662383
[2810]	valid_0's auc: 0.662408
[2820]	valid_0's auc: 0.662435
[2830]	valid_0's auc: 0.662444
[2840]	valid_0's auc: 0.662485
[2850]	valid_0's auc: 0.662498
[2860]	valid_0's auc: 0.662531
[2870]	valid_0's auc: 0.662586
[2880]	valid_0's auc: 0.662602
[2890]	valid_0's auc: 0.662644
[2900]	valid_0's auc: 0.66267
[2910]	valid_0's auc: 0.66271
[2920]	valid_0's auc: 0.662744
[2930]	valid_0's auc: 0.662771
[2940]	valid_0's auc: 0.662807
[2950]	valid_0's auc: 0.662837
[2960]	valid_0's auc: 0.662877
[2970]	valid_0's auc: 0.662892
[2980]	valid_0's auc: 0.662932
[2990]	valid_0's auc: 0.662942
[3000]	valid_0's auc: 0.662967
[3010]	valid_0's auc: 0.663003
[3020]	valid_0's auc: 0.663049
[3030]	valid_0's auc: 0.6631
[3040]	valid_0's auc: 0.663116
[3050]	valid_0's auc: 0.66313
[3060]	valid_0's auc: 0.663138
[3070]	valid_0's auc: 0.663146
[3080]	valid_0's auc: 0.663163
[3090]	valid_0's auc: 0.663227
[3100]	valid_0's auc: 0.663254
[3110]	valid_0's auc: 0.663272
[3120]	valid_0's auc: 0.663286
[3130]	valid_0's auc: 0.663317
[3140]	valid_0's auc: 0.663338
[3150]	valid_0's auc: 0.663351
[3160]	valid_0's auc: 0.663383
[3170]	valid_0's auc: 0.6634
[3180]	valid_0's auc: 0.663416
[3190]	valid_0's auc: 0.663443
[3200]	valid_0's auc: 0.663457
[3210]	valid_0's auc: 0.66347
[3220]	valid_0's auc: 0.663494
[3230]	valid_0's auc: 0.663494
[3240]	valid_0's auc: 0.663515
[3250]	valid_0's auc: 0.663531
[3260]	valid_0's auc: 0.663552
[3270]	valid_0's auc: 0.663576
[3280]	valid_0's auc: 0.663581
[3290]	valid_0's auc: 0.663614
[3300]	valid_0's auc: 0.663636
[3310]	valid_0's auc: 0.663645
[3320]	valid_0's auc: 0.663655
[3330]	valid_0's auc: 0.66368
[3340]	valid_0's auc: 0.663713
[3350]	valid_0's auc: 0.66374
[3360]	valid_0's auc: 0.663764
[3370]	valid_0's auc: 0.663817
[3380]	valid_0's auc: 0.663837
[3390]	valid_0's auc: 0.663834
[3400]	valid_0's auc: 0.663853
[3410]	valid_0's auc: 0.663878
[3420]	valid_0's auc: 0.663882
[3430]	valid_0's auc: 0.663916
[3440]	valid_0's auc: 0.66394
[3450]	valid_0's auc: 0.663957
[3460]	valid_0's auc: 0.663973
[3470]	valid_0's auc: 0.663989
[3480]	valid_0's auc: 0.664022
[3490]	valid_0's auc: 0.664032
[3500]	valid_0's auc: 0.664062
[3510]	valid_0's auc: 0.664076
[3520]	valid_0's auc: 0.664089
[3530]	valid_0's auc: 0.664097
[3540]	valid_0's auc: 0.664113
[3550]	valid_0's auc: 0.664131
[3560]	valid_0's auc: 0.664175
[3570]	valid_0's auc: 0.66419
[3580]	valid_0's auc: 0.664233
[3590]	valid_0's auc: 0.664247
[3600]	valid_0's auc: 0.664272
[3610]	valid_0's auc: 0.664279
[3620]	valid_0's auc: 0.664294
[3630]	valid_0's auc: 0.664336
[3640]	valid_0's auc: 0.664362
[3650]	valid_0's auc: 0.664385
[3660]	valid_0's auc: 0.664409
[3670]	valid_0's auc: 0.664427
[3680]	valid_0's auc: 0.664462
[3690]	valid_0's auc: 0.664474
[3700]	valid_0's auc: 0.664494
[3710]	valid_0's auc: 0.664515
[3720]	valid_0's auc: 0.664546
[3730]	valid_0's auc: 0.664568
[3740]	valid_0's auc: 0.664579
[3750]	valid_0's auc: 0.664593
[3760]	valid_0's auc: 0.664629
[3770]	valid_0's auc: 0.664645
[3780]	valid_0's auc: 0.664658
[3790]	valid_0's auc: 0.6647
[3800]	valid_0's auc: 0.66472
[3810]	valid_0's auc: 0.664743
[3820]	valid_0's auc: 0.664749
[3830]	valid_0's auc: 0.664766
[3840]	valid_0's auc: 0.664805
[3850]	valid_0's auc: 0.664818
[3860]	valid_0's auc: 0.664837
[3870]	valid_0's auc: 0.664849
[3880]	valid_0's auc: 0.664865
[3890]	valid_0's auc: 0.664874
[3900]	valid_0's auc: 0.664894
[3910]	valid_0's auc: 0.664907
[3920]	valid_0's auc: 0.664914
[3930]	valid_0's auc: 0.664932
[3940]	valid_0's auc: 0.664952
[3950]	valid_0's auc: 0.664974
[3960]	valid_0's auc: 0.664986
[3970]	valid_0's auc: 0.665032
[3980]	valid_0's auc: 0.665045
[3990]	valid_0's auc: 0.665065
[4000]	valid_0's auc: 0.665085
[4010]	valid_0's auc: 0.66509
[4020]	valid_0's auc: 0.665117
[4030]	valid_0's auc: 0.665135
[4040]	valid_0's auc: 0.665141
[4050]	valid_0's auc: 0.665147
[4060]	valid_0's auc: 0.665161
[4070]	valid_0's auc: 0.665185
[4080]	valid_0's auc: 0.665196
[4090]	valid_0's auc: 0.665216
[4100]	valid_0's auc: 0.665229
[4110]	valid_0's auc: 0.665248
[4120]	valid_0's auc: 0.66528
[4130]	valid_0's auc: 0.665291
[4140]	valid_0's auc: 0.665319
[4150]	valid_0's auc: 0.665331
[4160]	valid_0's auc: 0.665357
[4170]	valid_0's auc: 0.665375
[4180]	valid_0's auc: 0.665381
[4190]	valid_0's auc: 0.665396
[4200]	valid_0's auc: 0.665428
[4210]	valid_0's auc: 0.665443
[4220]	valid_0's auc: 0.665457
[4230]	valid_0's auc: 0.665486
[4240]	valid_0's auc: 0.665505
[4250]	valid_0's auc: 0.665508
[4260]	valid_0's auc: 0.665524
[4270]	valid_0's auc: 0.665537
[4280]	valid_0's auc: 0.665546
[4290]	valid_0's auc: 0.66556
[4300]	valid_0's auc: 0.66557
[4310]	valid_0's auc: 0.665596
[4320]	valid_0's auc: 0.665603
[4330]	valid_0's auc: 0.665618
[4340]	valid_0's auc: 0.665618
[4350]	valid_0's auc: 0.665642
[4360]	valid_0's auc: 0.665675
[4370]	valid_0's auc: 0.665673
[4380]	valid_0's auc: 0.665688
[4390]	valid_0's auc: 0.665703
[4400]	valid_0's auc: 0.665712
[4410]	valid_0's auc: 0.665729
[4420]	valid_0's auc: 0.665737
[4430]	valid_0's auc: 0.665765
[4440]	valid_0's auc: 0.66577
[4450]	valid_0's auc: 0.665792
[4460]	valid_0's auc: 0.665798
[4470]	valid_0's auc: 0.6658
[4480]	valid_0's auc: 0.665797
[4490]	valid_0's auc: 0.665797
[4500]	valid_0's auc: 0.665804
[4510]	valid_0's auc: 0.665812
[4520]	valid_0's auc: 0.66581
[4530]	valid_0's auc: 0.665821
[4540]	valid_0's auc: 0.665834
[4550]	valid_0's auc: 0.665858
[4560]	valid_0's auc: 0.665859
[4570]	valid_0's auc: 0.66586
[4580]	valid_0's auc: 0.665866
[4590]	valid_0's auc: 0.665877
[4600]	valid_0's auc: 0.665887
[4610]	valid_0's auc: 0.665889
[4620]	valid_0's auc: 0.665898
[4630]	valid_0's auc: 0.665895
[4640]	valid_0's auc: 0.665904
[4650]	valid_0's auc: 0.665923
[4660]	valid_0's auc: 0.665937
[4670]	valid_0's auc: 0.665952
[4680]	valid_0's auc: 0.665962
[4690]	valid_0's auc: 0.665974
[4700]	valid_0's auc: 0.665983
[4710]	valid_0's auc: 0.665993
[4720]	valid_0's auc: 0.665994
[4730]	valid_0's auc: 0.666002
[4740]	valid_0's auc: 0.666008
[4750]	valid_0's auc: 0.666019
[4760]	valid_0's auc: 0.666023
[4770]	valid_0's auc: 0.66604
[4780]	valid_0's auc: 0.666049
[4790]	valid_0's auc: 0.666053
[4800]	valid_0's auc: 0.666059
[4810]	valid_0's auc: 0.666066
[4820]	valid_0's auc: 0.666077
[4830]	valid_0's auc: 0.666086
[4840]	valid_0's auc: 0.666096
[4850]	valid_0's auc: 0.666101
[4860]	valid_0's auc: 0.666112
[4870]	valid_0's auc: 0.666121
[4880]	valid_0's auc: 0.666127
[4890]	valid_0's auc: 0.666141
[4900]	valid_0's auc: 0.666145
[4910]	valid_0's auc: 0.666162
[4920]	valid_0's auc: 0.666171
[4930]	valid_0's auc: 0.666181
[4940]	valid_0's auc: 0.666178
[4950]	valid_0's auc: 0.666181
[4960]	valid_0's auc: 0.666187
[4970]	valid_0's auc: 0.666189
[4980]	valid_0's auc: 0.666195
[4990]	valid_0's auc: 0.666205
[5000]	valid_0's auc: 0.66621
[5010]	valid_0's auc: 0.666217
[5020]	valid_0's auc: 0.666219
[5030]	valid_0's auc: 0.666223
[5040]	valid_0's auc: 0.666241
[5050]	valid_0's auc: 0.666246
[5060]	valid_0's auc: 0.666249
[5070]	valid_0's auc: 0.666254
[5080]	valid_0's auc: 0.666263
[5090]	valid_0's auc: 0.666268
[5100]	valid_0's auc: 0.666272
[5110]	valid_0's auc: 0.666279
[5120]	valid_0's auc: 0.666288
[5130]	valid_0's auc: 0.666296
[5140]	valid_0's auc: 0.666314
[5150]	valid_0's auc: 0.666341
[5160]	valid_0's auc: 0.666344
[5170]	valid_0's auc: 0.666354
[5180]	valid_0's auc: 0.66636
[5190]	valid_0's auc: 0.666369
[5200]	valid_0's auc: 0.666377
[5210]	valid_0's auc: 0.666377
[5220]	valid_0's auc: 0.666381
[5230]	valid_0's auc: 0.666389
[5240]	valid_0's auc: 0.666401
[5250]	valid_0's auc: 0.666411
[5260]	valid_0's auc: 0.666419
[5270]	valid_0's auc: 0.666427
[5280]	valid_0's auc: 0.666434
[5290]	valid_0's auc: 0.666436
[5300]	valid_0's auc: 0.666445
[5310]	valid_0's auc: 0.666458
[5320]	valid_0's auc: 0.666458
[5330]	valid_0's auc: 0.666466
[5340]	valid_0's auc: 0.666481
[5350]	valid_0's auc: 0.666491
[5360]	valid_0's auc: 0.666498
[5370]	valid_0's auc: 0.666502
[5380]	valid_0's auc: 0.666509
[5390]	valid_0's auc: 0.666515
[5400]	valid_0's auc: 0.666523
[5410]	valid_0's auc: 0.66653
[5420]	valid_0's auc: 0.666538
[5430]	valid_0's auc: 0.666539
[5440]	valid_0's auc: 0.66655
[5450]	valid_0's auc: 0.666558
[5460]	valid_0's auc: 0.666565
[5470]	valid_0's auc: 0.666571
[5480]	valid_0's auc: 0.66658
[5490]	valid_0's auc: 0.666578
[5500]	valid_0's auc: 0.66659
[5510]	valid_0's auc: 0.666598
[5520]	valid_0's auc: 0.666606
[5530]	valid_0's auc: 0.666613
[5540]	valid_0's auc: 0.666621
[5550]	valid_0's auc: 0.666627
[5560]	valid_0's auc: 0.666634
[5570]	valid_0's auc: 0.666654
[5580]	valid_0's auc: 0.666659
[5590]	valid_0's auc: 0.666661
[5600]	valid_0's auc: 0.666665
[5610]	valid_0's auc: 0.666668
[5620]	valid_0's auc: 0.666675
[5630]	valid_0's auc: 0.666684
[5640]	valid_0's auc: 0.66669
[5650]	valid_0's auc: 0.666699
[5660]	valid_0's auc: 0.666705
[5670]	valid_0's auc: 0.66671
[5680]	valid_0's auc: 0.666712
[5690]	valid_0's auc: 0.66672
[5700]	valid_0's auc: 0.666724
[5710]	valid_0's auc: 0.66673
[5720]	valid_0's auc: 0.66674
[5730]	valid_0's auc: 0.666741
[5740]	valid_0's auc: 0.666749
[5750]	valid_0's auc: 0.666762
[5760]	valid_0's auc: 0.666772
[5770]	valid_0's auc: 0.666778
[5780]	valid_0's auc: 0.66678
[5790]	valid_0's auc: 0.666783
[5800]	valid_0's auc: 0.666784
[5810]	valid_0's auc: 0.666791
[5820]	valid_0's auc: 0.666796
[5830]	valid_0's auc: 0.666802
[5840]	valid_0's auc: 0.666806
[5850]	valid_0's auc: 0.666812
[5860]	valid_0's auc: 0.666824
[5870]	valid_0's auc: 0.666834
[5880]	valid_0's auc: 0.666838
[5890]	valid_0's auc: 0.666842
[5900]	valid_0's auc: 0.666844
[5910]	valid_0's auc: 0.666851
[5920]	valid_0's auc: 0.666864
[5930]	valid_0's auc: 0.666869
[5940]	valid_0's auc: 0.666879
[5950]	valid_0's auc: 0.666885
[5960]	valid_0's auc: 0.666891
[5970]	valid_0's auc: 0.666893
[5980]	valid_0's auc: 0.666895
[5990]	valid_0's auc: 0.666904
[6000]	valid_0's auc: 0.666906
[6010]	valid_0's auc: 0.666908
[6020]	valid_0's auc: 0.666915
[6030]	valid_0's auc: 0.66692
[6040]	valid_0's auc: 0.666937
[6050]	valid_0's auc: 0.66694
[6060]	valid_0's auc: 0.666946
[6070]	valid_0's auc: 0.666954
[6080]	valid_0's auc: 0.666955
[6090]	valid_0's auc: 0.666956
[6100]	valid_0's auc: 0.666959
[6110]	valid_0's auc: 0.666972
[6120]	valid_0's auc: 0.666989
[6130]	valid_0's auc: 0.666994
[6140]	valid_0's auc: 0.667
[6150]	valid_0's auc: 0.667005
[6160]	valid_0's auc: 0.667009
[6170]	valid_0's auc: 0.667009
[6180]	valid_0's auc: 0.667017
[6190]	valid_0's auc: 0.667022
[6200]	valid_0's auc: 0.66702
[6210]	valid_0's auc: 0.667028
[6220]	valid_0's auc: 0.667029
[6230]	valid_0's auc: 0.667032
[6240]	valid_0's auc: 0.667037
[6250]	valid_0's auc: 0.667053
[6260]	valid_0's auc: 0.667055
[6270]	valid_0's auc: 0.667057
[6280]	valid_0's auc: 0.66706
[6290]	valid_0's auc: 0.667063
[6300]	valid_0's auc: 0.66707
[6310]	valid_0's auc: 0.667072
[6320]	valid_0's auc: 0.667075
[6330]	valid_0's auc: 0.667077
[6340]	valid_0's auc: 0.667083
[6350]	valid_0's auc: 0.66709
[6360]	valid_0's auc: 0.667106
[6370]	valid_0's auc: 0.667109
[6380]	valid_0's auc: 0.667115
[6390]	valid_0's auc: 0.667119
[6400]	valid_0's auc: 0.667128
[6410]	valid_0's auc: 0.667138
[6420]	valid_0's auc: 0.66714
[6430]	valid_0's auc: 0.667141
[6440]	valid_0's auc: 0.667148
[6450]	valid_0's auc: 0.667152
[6460]	valid_0's auc: 0.667154
[6470]	valid_0's auc: 0.667158
[6480]	valid_0's auc: 0.667164
[6490]	valid_0's auc: 0.667166
[6500]	valid_0's auc: 0.667177
[6510]	valid_0's auc: 0.667179
[6520]	valid_0's auc: 0.667187
[6530]	valid_0's auc: 0.667197
[6540]	valid_0's auc: 0.667202
[6550]	valid_0's auc: 0.667212
[6560]	valid_0's auc: 0.667229
[6570]	valid_0's auc: 0.667228
[6580]	valid_0's auc: 0.667229
[6590]	valid_0's auc: 0.667234
[6600]	valid_0's auc: 0.667236
[6610]	valid_0's auc: 0.667239
[6620]	valid_0's auc: 0.667243
[6630]	valid_0's auc: 0.667248
[6640]	valid_0's auc: 0.667251
[6650]	valid_0's auc: 0.667252
[6660]	valid_0's auc: 0.667264
[6670]	valid_0's auc: 0.667282
[6680]	valid_0's auc: 0.667283
[6690]	valid_0's auc: 0.667281
[6700]	valid_0's auc: 0.667289
[6710]	valid_0's auc: 0.667298
[6720]	valid_0's auc: 0.667301
[6730]	valid_0's auc: 0.66732
[6740]	valid_0's auc: 0.667323
[6750]	valid_0's auc: 0.667328
[6760]	valid_0's auc: 0.667333
[6770]	valid_0's auc: 0.667335
[6780]	valid_0's auc: 0.667337
[6790]	valid_0's auc: 0.667343
[6800]	valid_0's auc: 0.667348
[6810]	valid_0's auc: 0.667353
[6820]	valid_0's auc: 0.667352
[6830]	valid_0's auc: 0.667361
[6840]	valid_0's auc: 0.667365
[6850]	valid_0's auc: 0.66737
[6860]	valid_0's auc: 0.667377
[6870]	valid_0's auc: 0.667379
[6880]	valid_0's auc: 0.667379
[6890]	valid_0's auc: 0.667375
[6900]	valid_0's auc: 0.6674
[6910]	valid_0's auc: 0.667415
[6920]	valid_0's auc: 0.667421
[6930]	valid_0's auc: 0.667424
[6940]	valid_0's auc: 0.667429
[6950]	valid_0's auc: 0.66743
[6960]	valid_0's auc: 0.667433
[6970]	valid_0's auc: 0.667445
[6980]	valid_0's auc: 0.667448
[6990]	valid_0's auc: 0.667461
[7000]	valid_0's auc: 0.667467
[7010]	valid_0's auc: 0.66747
[7020]	valid_0's auc: 0.667472
[7030]	valid_0's auc: 0.667472
[7040]	valid_0's auc: 0.667476
[7050]	valid_0's auc: 0.66748
[7060]	valid_0's auc: 0.667479
[7070]	valid_0's auc: 0.667482
[7080]	valid_0's auc: 0.667495
[7090]	valid_0's auc: 0.667502
[7100]	valid_0's auc: 0.6675
[7110]	valid_0's auc: 0.667503
[7120]	valid_0's auc: 0.667516
[7130]	valid_0's auc: 0.667541
[7140]	valid_0's auc: 0.667545
[7150]	valid_0's auc: 0.667547
[7160]	valid_0's auc: 0.667553
[7170]	valid_0's auc: 0.66756
[7180]	valid_0's auc: 0.667561
[7190]	valid_0's auc: 0.66757
[7200]	valid_0's auc: 0.667573
[7210]	valid_0's auc: 0.667576
[7220]	valid_0's auc: 0.66758
[7230]	valid_0's auc: 0.66758
[7240]	valid_0's auc: 0.667583
[7250]	valid_0's auc: 0.667584
[7260]	valid_0's auc: 0.667588
[7270]	valid_0's auc: 0.66759
[7280]	valid_0's auc: 0.667591
[7290]	valid_0's auc: 0.6676
[7300]	valid_0's auc: 0.667604
[7310]	valid_0's auc: 0.667615
[7320]	valid_0's auc: 0.667613
[7330]	valid_0's auc: 0.667619
[7340]	valid_0's auc: 0.667622
[7350]	valid_0's auc: 0.667627
[7360]	valid_0's auc: 0.667635
[7370]	valid_0's auc: 0.667639
[7380]	valid_0's auc: 0.66765
[7390]	valid_0's auc: 0.667663
[7400]	valid_0's auc: 0.667666
[7410]	valid_0's auc: 0.667665
[7420]	valid_0's auc: 0.667663
[7430]	valid_0's auc: 0.667665
[7440]	valid_0's auc: 0.667674
[7450]	valid_0's auc: 0.66768
[7460]	valid_0's auc: 0.667682
[7470]	valid_0's auc: 0.66768
[7480]	valid_0's auc: 0.667685
[7490]	valid_0's auc: 0.667691
[7500]	valid_0's auc: 0.667691
[7510]	valid_0's auc: 0.667694
[7520]	valid_0's auc: 0.667716
[7530]	valid_0's auc: 0.667717
[7540]	valid_0's auc: 0.667719
[7550]	valid_0's auc: 0.667722
[7560]	valid_0's auc: 0.667724
[7570]	valid_0's auc: 0.667727
[7580]	valid_0's auc: 0.66773
[7590]	valid_0's auc: 0.667735
[7600]	valid_0's auc: 0.667742
[7610]	valid_0's auc: 0.667762
[7620]	valid_0's auc: 0.667764
[7630]	valid_0's auc: 0.667763
[7640]	valid_0's auc: 0.667764
[7650]	valid_0's auc: 0.667759
[7660]	valid_0's auc: 0.66776
[7670]	valid_0's auc: 0.66776
[7680]	valid_0's auc: 0.66776
Early stopping, best iteration is:
[7639]	valid_0's auc: 0.667765
0.667764647413
7639

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 300
max_depth            : -1
learning_rate        : 0.02
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650103
[20]	valid_0's auc: 0.652404
[30]	valid_0's auc: 0.654006
[40]	valid_0's auc: 0.655779
[50]	valid_0's auc: 0.657424
[60]	valid_0's auc: 0.658376
[70]	valid_0's auc: 0.659284
[80]	valid_0's auc: 0.660437
[90]	valid_0's auc: 0.66154
[100]	valid_0's auc: 0.662687
[110]	valid_0's auc: 0.663602
[120]	valid_0's auc: 0.664419
[130]	valid_0's auc: 0.665409
[140]	valid_0's auc: 0.666421
[150]	valid_0's auc: 0.667175
[160]	valid_0's auc: 0.66784
[170]	valid_0's auc: 0.668352
[180]	valid_0's auc: 0.668927
[190]	valid_0's auc: 0.669299
[200]	valid_0's auc: 0.669724
[210]	valid_0's auc: 0.670047
[220]	valid_0's auc: 0.670576
[230]	valid_0's auc: 0.670981
[240]	valid_0's auc: 0.67128
[250]	valid_0's auc: 0.671564
[260]	valid_0's auc: 0.67177
[270]	valid_0's auc: 0.671902
[280]	valid_0's auc: 0.672136
[290]	valid_0's auc: 0.672372
[300]	valid_0's auc: 0.67255
[310]	valid_0's auc: 0.672729
[320]	valid_0's auc: 0.67284
[330]	valid_0's auc: 0.673048
[340]	valid_0's auc: 0.673172
[350]	valid_0's auc: 0.673276
[360]	valid_0's auc: 0.673343
[370]	valid_0's auc: 0.673394
[380]	valid_0's auc: 0.673481
[390]	valid_0's auc: 0.673546
[400]	valid_0's auc: 0.67362
[410]	valid_0's auc: 0.673712
[420]	valid_0's auc: 0.673753
[430]	valid_0's auc: 0.673777
[440]	valid_0's auc: 0.673848
[450]	valid_0's auc: 0.67392
[460]	valid_0's auc: 0.673969
[470]	valid_0's auc: 0.67399
[480]	valid_0's auc: 0.674017
[490]	valid_0's auc: 0.674041
[500]	valid_0's auc: 0.674076
[510]	valid_0's auc: 0.6741
[520]	valid_0's auc: 0.674094
[530]	valid_0's auc: 0.674096
[540]	valid_0's auc: 0.67413
[550]	valid_0's auc: 0.674128
[560]	valid_0's auc: 0.674161
[570]	valid_0's auc: 0.674191
[580]	valid_0's auc: 0.674183
[590]	valid_0's auc: 0.674192
[600]	valid_0's auc: 0.674174
[610]	valid_0's auc: 0.674196
[620]	valid_0's auc: 0.674193
[630]	valid_0's auc: 0.674199
[640]	valid_0's auc: 0.674177
[650]	valid_0's auc: 0.674183
[660]	valid_0's auc: 0.674161
Early stopping, best iteration is:
[612]	valid_0's auc: 0.674206
0.674206404529
612

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 200
max_depth            : 30
learning_rate        : 0.3
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.654184
[20]	valid_0's auc: 0.659949
[30]	valid_0's auc: 0.663379
[40]	valid_0's auc: 0.66496
[50]	valid_0's auc: 0.666072
[60]	valid_0's auc: 0.667143
[70]	valid_0's auc: 0.667805
[80]	valid_0's auc: 0.668292
[90]	valid_0's auc: 0.668706
[100]	valid_0's auc: 0.668977
[110]	valid_0's auc: 0.669253
[120]	valid_0's auc: 0.669444
[130]	valid_0's auc: 0.669584
[140]	valid_0's auc: 0.669747
[150]	valid_0's auc: 0.669901
[160]	valid_0's auc: 0.67001
[170]	valid_0's auc: 0.669936
[180]	valid_0's auc: 0.670134
[190]	valid_0's auc: 0.67009
[200]	valid_0's auc: 0.670189
[210]	valid_0's auc: 0.670138
[220]	valid_0's auc: 0.670195
[230]	valid_0's auc: 0.670181
[240]	valid_0's auc: 0.670135
Early stopping, best iteration is:
[197]	valid_0's auc: 0.670234
0.670234144328
197

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 300
max_depth            : 30
learning_rate        : 0.02
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.635771
[20]	valid_0's auc: 0.641236
[30]	valid_0's auc: 0.644317
[40]	valid_0's auc: 0.646461
[50]	valid_0's auc: 0.647611
[60]	valid_0's auc: 0.648717
[70]	valid_0's auc: 0.649538
[80]	valid_0's auc: 0.650498
[90]	valid_0's auc: 0.651673
[100]	valid_0's auc: 0.65227
[110]	valid_0's auc: 0.653046
[120]	valid_0's auc: 0.653897
[130]	valid_0's auc: 0.65469
[140]	valid_0's auc: 0.655308
[150]	valid_0's auc: 0.655959
[160]	valid_0's auc: 0.656773
[170]	valid_0's auc: 0.657737
[180]	valid_0's auc: 0.658209
[190]	valid_0's auc: 0.658651
[200]	valid_0's auc: 0.658996
[210]	valid_0's auc: 0.659358
[220]	valid_0's auc: 0.659688
[230]	valid_0's auc: 0.659989
[240]	valid_0's auc: 0.660334
[250]	valid_0's auc: 0.660471
[260]	valid_0's auc: 0.660701
[270]	valid_0's auc: 0.660846
[280]	valid_0's auc: 0.661052
[290]	valid_0's auc: 0.661275
[300]	valid_0's auc: 0.66155
[310]	valid_0's auc: 0.661916
[320]	valid_0's auc: 0.66209
[330]	valid_0's auc: 0.662295
[340]	valid_0's auc: 0.662511
[350]	valid_0's auc: 0.662768
[360]	valid_0's auc: 0.662948
[370]	valid_0's auc: 0.663115
[380]	valid_0's auc: 0.663263
[390]	valid_0's auc: 0.663432
[400]	valid_0's auc: 0.663599
[410]	valid_0's auc: 0.663829
[420]	valid_0's auc: 0.664007
[430]	valid_0's auc: 0.66414
[440]	valid_0's auc: 0.664291
[450]	valid_0's auc: 0.664443
[460]	valid_0's auc: 0.664603
[470]	valid_0's auc: 0.664799
[480]	valid_0's auc: 0.66492
[490]	valid_0's auc: 0.665072
[500]	valid_0's auc: 0.665226
[510]	valid_0's auc: 0.665352
[520]	valid_0's auc: 0.665475
[530]	valid_0's auc: 0.665626
[540]	valid_0's auc: 0.665729
[550]	valid_0's auc: 0.665864
[560]	valid_0's auc: 0.665985
[570]	valid_0's auc: 0.666095
[580]	valid_0's auc: 0.666198
[590]	valid_0's auc: 0.666298
[600]	valid_0's auc: 0.666403
[610]	valid_0's auc: 0.666525
[620]	valid_0's auc: 0.666617
[630]	valid_0's auc: 0.666737
[640]	valid_0's auc: 0.666856
[650]	valid_0's auc: 0.666966
[660]	valid_0's auc: 0.667042
[670]	valid_0's auc: 0.667139
[680]	valid_0's auc: 0.667244
[690]	valid_0's auc: 0.667342
[700]	valid_0's auc: 0.667416
[710]	valid_0's auc: 0.667527
[720]	valid_0's auc: 0.667628
[730]	valid_0's auc: 0.667711
[740]	valid_0's auc: 0.667831
[750]	valid_0's auc: 0.667899
[760]	valid_0's auc: 0.667984
[770]	valid_0's auc: 0.668068
[780]	valid_0's auc: 0.668137
[790]	valid_0's auc: 0.668209
[800]	valid_0's auc: 0.668299
[810]	valid_0's auc: 0.668373
[820]	valid_0's auc: 0.668433
[830]	valid_0's auc: 0.668499
[840]	valid_0's auc: 0.668571
[850]	valid_0's auc: 0.668642
[860]	valid_0's auc: 0.668702
[870]	valid_0's auc: 0.668792
[880]	valid_0's auc: 0.668874
[890]	valid_0's auc: 0.668951
[900]	valid_0's auc: 0.669035
[910]	valid_0's auc: 0.669114
[920]	valid_0's auc: 0.66918
[930]	valid_0's auc: 0.669254
[940]	valid_0's auc: 0.669317
[950]	valid_0's auc: 0.669387
[960]	valid_0's auc: 0.669427
[970]	valid_0's auc: 0.669477
[980]	valid_0's auc: 0.669522
[990]	valid_0's auc: 0.669559
[1000]	valid_0's auc: 0.669619
[1010]	valid_0's auc: 0.669661
[1020]	valid_0's auc: 0.669696
[1030]	valid_0's auc: 0.669731
[1040]	valid_0's auc: 0.66977
[1050]	valid_0's auc: 0.669789
[1060]	valid_0's auc: 0.669852
[1070]	valid_0's auc: 0.669901
[1080]	valid_0's auc: 0.669946
[1090]	valid_0's auc: 0.669985
[1100]	valid_0's auc: 0.670061
[1110]	valid_0's auc: 0.670109
[1120]	valid_0's auc: 0.670129
[1130]	valid_0's auc: 0.670154
[1140]	valid_0's auc: 0.670189
[1150]	valid_0's auc: 0.670211
[1160]	valid_0's auc: 0.670235
[1170]	valid_0's auc: 0.670287
[1180]	valid_0's auc: 0.670367
[1190]	valid_0's auc: 0.6704
[1200]	valid_0's auc: 0.670439
[1210]	valid_0's auc: 0.670469
[1220]	valid_0's auc: 0.670483
[1230]	valid_0's auc: 0.670524
[1240]	valid_0's auc: 0.670555
[1250]	valid_0's auc: 0.670587
[1260]	valid_0's auc: 0.670628
[1270]	valid_0's auc: 0.670641
[1280]	valid_0's auc: 0.670677
[1290]	valid_0's auc: 0.670693
[1300]	valid_0's auc: 0.670738
[1310]	valid_0's auc: 0.670767
[1320]	valid_0's auc: 0.670781
[1330]	valid_0's auc: 0.670805
[1340]	valid_0's auc: 0.670842
[1350]	valid_0's auc: 0.670866
[1360]	valid_0's auc: 0.670923
[1370]	valid_0's auc: 0.67095
[1380]	valid_0's auc: 0.670969
[1390]	valid_0's auc: 0.670985
[1400]	valid_0's auc: 0.671024
[1410]	valid_0's auc: 0.671053
[1420]	valid_0's auc: 0.671083
[1430]	valid_0's auc: 0.671115
[1440]	valid_0's auc: 0.671139
[1450]	valid_0's auc: 0.671159
[1460]	valid_0's auc: 0.671194
[1470]	valid_0's auc: 0.671228
[1480]	valid_0's auc: 0.671254
[1490]	valid_0's auc: 0.671266
[1500]	valid_0's auc: 0.6713
[1510]	valid_0's auc: 0.671334
[1520]	valid_0's auc: 0.671346
[1530]	valid_0's auc: 0.671362
[1540]	valid_0's auc: 0.671378
[1550]	valid_0's auc: 0.671376
[1560]	valid_0's auc: 0.671376
[1570]	valid_0's auc: 0.671394
[1580]	valid_0's auc: 0.671403
[1590]	valid_0's auc: 0.671407
[1600]	valid_0's auc: 0.671432
[1610]	valid_0's auc: 0.671452
[1620]	valid_0's auc: 0.671468
[1630]	valid_0's auc: 0.671489
[1640]	valid_0's auc: 0.671499
[1650]	valid_0's auc: 0.671521
[1660]	valid_0's auc: 0.67155
[1670]	valid_0's auc: 0.671563
[1680]	valid_0's auc: 0.67159
[1690]	valid_0's auc: 0.671607
[1700]	valid_0's auc: 0.67162
[1710]	valid_0's auc: 0.671655
[1720]	valid_0's auc: 0.671683
[1730]	valid_0's auc: 0.671698
[1740]	valid_0's auc: 0.671739
[1750]	valid_0's auc: 0.671762
[1760]	valid_0's auc: 0.671778
[1770]	valid_0's auc: 0.671792
[1780]	valid_0's auc: 0.671792
[1790]	valid_0's auc: 0.671829
[1800]	valid_0's auc: 0.671846
[1810]	valid_0's auc: 0.671862
[1820]	valid_0's auc: 0.671867
[1830]	valid_0's auc: 0.67186
[1840]	valid_0's auc: 0.671954
[1850]	valid_0's auc: 0.672059
[1860]	valid_0's auc: 0.672071
[1870]	valid_0's auc: 0.672081
[1880]	valid_0's auc: 0.672107
[1890]	valid_0's auc: 0.672126
[1900]	valid_0's auc: 0.672141
[1910]	valid_0's auc: 0.672169
[1920]	valid_0's auc: 0.672175
[1930]	valid_0's auc: 0.672202
[1940]	valid_0's auc: 0.672217
[1950]	valid_0's auc: 0.672237
[1960]	valid_0's auc: 0.67225
[1970]	valid_0's auc: 0.67226
[1980]	valid_0's auc: 0.672263
[1990]	valid_0's auc: 0.672259
[2000]	valid_0's auc: 0.67228
[2010]	valid_0's auc: 0.672279
[2020]	valid_0's auc: 0.6723
[2030]	valid_0's auc: 0.672342
[2040]	valid_0's auc: 0.672367
[2050]	valid_0's auc: 0.672375
[2060]	valid_0's auc: 0.672403
[2070]	valid_0's auc: 0.672418
[2080]	valid_0's auc: 0.672436
[2090]	valid_0's auc: 0.672441
[2100]	valid_0's auc: 0.672452
[2110]	valid_0's auc: 0.672469
[2120]	valid_0's auc: 0.672487
[2130]	valid_0's auc: 0.672493
[2140]	valid_0's auc: 0.672511
[2150]	valid_0's auc: 0.672521
[2160]	valid_0's auc: 0.67253
[2170]	valid_0's auc: 0.67254
[2180]	valid_0's auc: 0.672555
[2190]	valid_0's auc: 0.672561
[2200]	valid_0's auc: 0.67258
[2210]	valid_0's auc: 0.6726
[2220]	valid_0's auc: 0.67261
[2230]	valid_0's auc: 0.672631
[2240]	valid_0's auc: 0.672636
[2250]	valid_0's auc: 0.672649
[2260]	valid_0's auc: 0.672655
[2270]	valid_0's auc: 0.672664
[2280]	valid_0's auc: 0.672675
[2290]	valid_0's auc: 0.672691
[2300]	valid_0's auc: 0.672711
[2310]	valid_0's auc: 0.672719
[2320]	valid_0's auc: 0.672725
[2330]	valid_0's auc: 0.672739
[2340]	valid_0's auc: 0.672736
[2350]	valid_0's auc: 0.67275
[2360]	valid_0's auc: 0.672757
[2370]	valid_0's auc: 0.67276
[2380]	valid_0's auc: 0.672767
[2390]	valid_0's auc: 0.67278
[2400]	valid_0's auc: 0.672788
[2410]	valid_0's auc: 0.672797
[2420]	valid_0's auc: 0.672801
[2430]	valid_0's auc: 0.672825
[2440]	valid_0's auc: 0.672826
[2450]	valid_0's auc: 0.672839
[2460]	valid_0's auc: 0.672858
[2470]	valid_0's auc: 0.672859
[2480]	valid_0's auc: 0.672862
[2490]	valid_0's auc: 0.672874
[2500]	valid_0's auc: 0.672878
[2510]	valid_0's auc: 0.672892
[2520]	valid_0's auc: 0.672903
[2530]	valid_0's auc: 0.672912
[2540]	valid_0's auc: 0.672909
[2550]	valid_0's auc: 0.672914
[2560]	valid_0's auc: 0.672914
[2570]	valid_0's auc: 0.672917
[2580]	valid_0's auc: 0.672919
[2590]	valid_0's auc: 0.672927
[2600]	valid_0's auc: 0.67293
[2610]	valid_0's auc: 0.672937
[2620]	valid_0's auc: 0.672945
[2630]	valid_0's auc: 0.67295
[2640]	valid_0's auc: 0.672954
[2650]	valid_0's auc: 0.672966
[2660]	valid_0's auc: 0.672978
[2670]	valid_0's auc: 0.672972
[2680]	valid_0's auc: 0.672975
[2690]	valid_0's auc: 0.672979
[2700]	valid_0's auc: 0.672982
[2710]	valid_0's auc: 0.672986
[2720]	valid_0's auc: 0.672994
[2730]	valid_0's auc: 0.673
[2740]	valid_0's auc: 0.673005
[2750]	valid_0's auc: 0.673018
[2760]	valid_0's auc: 0.673029
[2770]	valid_0's auc: 0.673035
[2780]	valid_0's auc: 0.673037
[2790]	valid_0's auc: 0.673033
[2800]	valid_0's auc: 0.673039
[2810]	valid_0's auc: 0.673049
[2820]	valid_0's auc: 0.673065
[2830]	valid_0's auc: 0.67307
[2840]	valid_0's auc: 0.673081
[2850]	valid_0's auc: 0.673085
[2860]	valid_0's auc: 0.6731
[2870]	valid_0's auc: 0.673099
[2880]	valid_0's auc: 0.673103
[2890]	valid_0's auc: 0.673113
[2900]	valid_0's auc: 0.673118
[2910]	valid_0's auc: 0.673125
[2920]	valid_0's auc: 0.673128
[2930]	valid_0's auc: 0.673127
[2940]	valid_0's auc: 0.673134
[2950]	valid_0's auc: 0.673132
[2960]	valid_0's auc: 0.673136
[2970]	valid_0's auc: 0.673141
[2980]	valid_0's auc: 0.673148
[2990]	valid_0's auc: 0.673153
[3000]	valid_0's auc: 0.673159
[3010]	valid_0's auc: 0.673166
[3020]	valid_0's auc: 0.673168
[3030]	valid_0's auc: 0.673172
[3040]	valid_0's auc: 0.673173
[3050]	valid_0's auc: 0.673179
[3060]	valid_0's auc: 0.673181
[3070]	valid_0's auc: 0.673179
[3080]	valid_0's auc: 0.67318
[3090]	valid_0's auc: 0.673184
[3100]	valid_0's auc: 0.673188
[3110]	valid_0's auc: 0.673197
[3120]	valid_0's auc: 0.673202
[3130]	valid_0's auc: 0.673195
[3140]	valid_0's auc: 0.673196
[3150]	valid_0's auc: 0.673206
[3160]	valid_0's auc: 0.673207
[3170]	valid_0's auc: 0.673209
[3180]	valid_0's auc: 0.673195
[3190]	valid_0's auc: 0.67319
[3200]	valid_0's auc: 0.673179
[3210]	valid_0's auc: 0.673188
Early stopping, best iteration is:
[3166]	valid_0's auc: 0.673213
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/playground_V1006/B_training_V1305.py", line 150, in <module>
    verbose_eval=10,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 130 (interrupted by signal 2: SIGINT)
'''
