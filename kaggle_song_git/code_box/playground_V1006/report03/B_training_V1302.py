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
lambda_l1 = 0.1

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
                      num_boost_round=5000,
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/playground_V1006/training_V1304.py
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

max_depth            : 20
boosting             : gbdt
learning_rate        : 0.3
num_leaves           : 100

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.648023
[20]	valid_0's auc: 0.65465
[30]	valid_0's auc: 0.657333
[40]	valid_0's auc: 0.659297
[50]	valid_0's auc: 0.660841
[60]	valid_0's auc: 0.662405
[70]	valid_0's auc: 0.663262
[80]	valid_0's auc: 0.664172
[90]	valid_0's auc: 0.664781
[100]	valid_0's auc: 0.665138
[110]	valid_0's auc: 0.665848
[120]	valid_0's auc: 0.666247
[130]	valid_0's auc: 0.6665
[140]	valid_0's auc: 0.666826
[150]	valid_0's auc: 0.667227
[160]	valid_0's auc: 0.667207
[170]	valid_0's auc: 0.667437
[180]	valid_0's auc: 0.667617
[190]	valid_0's auc: 0.667664
[200]	valid_0's auc: 0.667847
[210]	valid_0's auc: 0.667856
[220]	valid_0's auc: 0.668322
[230]	valid_0's auc: 0.668335
[240]	valid_0's auc: 0.668442
[250]	valid_0's auc: 0.668469
[260]	valid_0's auc: 0.668494
[270]	valid_0's auc: 0.668446
[280]	valid_0's auc: 0.668525
[290]	valid_0's auc: 0.668695
[300]	valid_0's auc: 0.668701
[310]	valid_0's auc: 0.668629
[320]	valid_0's auc: 0.668635
[330]	valid_0's auc: 0.668734
[340]	valid_0's auc: 0.668687
[350]	valid_0's auc: 0.668609
[360]	valid_0's auc: 0.668598
[370]	valid_0's auc: 0.66874
[380]	valid_0's auc: 0.66879
[390]	valid_0's auc: 0.668845
[400]	valid_0's auc: 0.668808
[410]	valid_0's auc: 0.668723
[420]	valid_0's auc: 0.668719
[430]	valid_0's auc: 0.668671
Early stopping, best iteration is:
[389]	valid_0's auc: 0.668865
0.668865090796
389

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 10
boosting             : gbdt
learning_rate        : 0.3
num_leaves           : 25

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.635712
[20]	valid_0's auc: 0.64143
[30]	valid_0's auc: 0.645577
[40]	valid_0's auc: 0.648214
[50]	valid_0's auc: 0.650108
[60]	valid_0's auc: 0.651741
[70]	valid_0's auc: 0.653107
[80]	valid_0's auc: 0.654106
[90]	valid_0's auc: 0.655058
[100]	valid_0's auc: 0.656031
[110]	valid_0's auc: 0.656863
[120]	valid_0's auc: 0.657644
[130]	valid_0's auc: 0.658213
[140]	valid_0's auc: 0.658644
[150]	valid_0's auc: 0.659074
[160]	valid_0's auc: 0.659446
[170]	valid_0's auc: 0.659856
[180]	valid_0's auc: 0.660115
[190]	valid_0's auc: 0.660433
[200]	valid_0's auc: 0.661164
[210]	valid_0's auc: 0.661711
[220]	valid_0's auc: 0.661831
[230]	valid_0's auc: 0.661973
[240]	valid_0's auc: 0.662256
[250]	valid_0's auc: 0.662408
[260]	valid_0's auc: 0.662574
[270]	valid_0's auc: 0.662832
[280]	valid_0's auc: 0.663115
[290]	valid_0's auc: 0.663272
[300]	valid_0's auc: 0.663312
[310]	valid_0's auc: 0.663442
[320]	valid_0's auc: 0.663563
[330]	valid_0's auc: 0.66359
[340]	valid_0's auc: 0.663674
[350]	valid_0's auc: 0.663849
[360]	valid_0's auc: 0.664107
[370]	valid_0's auc: 0.664075
[380]	valid_0's auc: 0.664167
[390]	valid_0's auc: 0.664338
[400]	valid_0's auc: 0.664331
[410]	valid_0's auc: 0.664413
[420]	valid_0's auc: 0.664548
[430]	valid_0's auc: 0.664592
[440]	valid_0's auc: 0.664611
[450]	valid_0's auc: 0.664723
[460]	valid_0's auc: 0.664842
[470]	valid_0's auc: 0.664831
[480]	valid_0's auc: 0.664915
[490]	valid_0's auc: 0.665007
[500]	valid_0's auc: 0.66501
[510]	valid_0's auc: 0.665058
[520]	valid_0's auc: 0.664974
[530]	valid_0's auc: 0.665005
[540]	valid_0's auc: 0.665034
[550]	valid_0's auc: 0.665086
[560]	valid_0's auc: 0.665189
[570]	valid_0's auc: 0.665235
[580]	valid_0's auc: 0.6653
[590]	valid_0's auc: 0.665346
[600]	valid_0's auc: 0.665396
[610]	valid_0's auc: 0.665403
[620]	valid_0's auc: 0.66537
[630]	valid_0's auc: 0.665405
[640]	valid_0's auc: 0.665428
[650]	valid_0's auc: 0.665446
[660]	valid_0's auc: 0.665625
[670]	valid_0's auc: 0.665634
[680]	valid_0's auc: 0.665696
[690]	valid_0's auc: 0.66565
[700]	valid_0's auc: 0.665625
[710]	valid_0's auc: 0.665774
[720]	valid_0's auc: 0.665873
[730]	valid_0's auc: 0.665862
[740]	valid_0's auc: 0.665874
[750]	valid_0's auc: 0.665896
[760]	valid_0's auc: 0.665928
[770]	valid_0's auc: 0.666076
[780]	valid_0's auc: 0.666131
[790]	valid_0's auc: 0.666143
[800]	valid_0's auc: 0.666146
[810]	valid_0's auc: 0.666171
[820]	valid_0's auc: 0.666233
[830]	valid_0's auc: 0.666254
[840]	valid_0's auc: 0.666276
[850]	valid_0's auc: 0.666242
[860]	valid_0's auc: 0.666389
[870]	valid_0's auc: 0.666407
[880]	valid_0's auc: 0.666442
[890]	valid_0's auc: 0.666452
[900]	valid_0's auc: 0.666387
[910]	valid_0's auc: 0.666392
[920]	valid_0's auc: 0.666345
[930]	valid_0's auc: 0.666437
Early stopping, best iteration is:
[885]	valid_0's auc: 0.666467
0.666466959078
885

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 30
boosting             : gbdt
learning_rate        : 0.07
num_leaves           : 25

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.623957
[20]	valid_0's auc: 0.630463
[30]	valid_0's auc: 0.634006
[40]	valid_0's auc: 0.637078
[50]	valid_0's auc: 0.640128
[60]	valid_0's auc: 0.642623
[70]	valid_0's auc: 0.64425
[80]	valid_0's auc: 0.645979
[90]	valid_0's auc: 0.647346
[100]	valid_0's auc: 0.64827
[110]	valid_0's auc: 0.649536
[120]	valid_0's auc: 0.65046
[130]	valid_0's auc: 0.65137
[140]	valid_0's auc: 0.652127
[150]	valid_0's auc: 0.652801
[160]	valid_0's auc: 0.653421
[170]	valid_0's auc: 0.653935
[180]	valid_0's auc: 0.654386
[190]	valid_0's auc: 0.654824
[200]	valid_0's auc: 0.655461
[210]	valid_0's auc: 0.656026
[220]	valid_0's auc: 0.656511
[230]	valid_0's auc: 0.656811
[240]	valid_0's auc: 0.657088
[250]	valid_0's auc: 0.657466
[260]	valid_0's auc: 0.65769
[270]	valid_0's auc: 0.658053
[280]	valid_0's auc: 0.658279
[290]	valid_0's auc: 0.658414
[300]	valid_0's auc: 0.658535
[310]	valid_0's auc: 0.6587
[320]	valid_0's auc: 0.658958
[330]	valid_0's auc: 0.659092
[340]	valid_0's auc: 0.659332
[350]	valid_0's auc: 0.659724
[360]	valid_0's auc: 0.660015
[370]	valid_0's auc: 0.66024
[380]	valid_0's auc: 0.66053
[390]	valid_0's auc: 0.660698
[400]	valid_0's auc: 0.660896
[410]	valid_0's auc: 0.660955
[420]	valid_0's auc: 0.6611
[430]	valid_0's auc: 0.661288
[440]	valid_0's auc: 0.661426
[450]	valid_0's auc: 0.661615
[460]	valid_0's auc: 0.661807
[470]	valid_0's auc: 0.661958
[480]	valid_0's auc: 0.662111
[490]	valid_0's auc: 0.662228
[500]	valid_0's auc: 0.662354
[510]	valid_0's auc: 0.662445
[520]	valid_0's auc: 0.662538
[530]	valid_0's auc: 0.662605
[540]	valid_0's auc: 0.662641
[550]	valid_0's auc: 0.6627
[560]	valid_0's auc: 0.662815
[570]	valid_0's auc: 0.66286
[580]	valid_0's auc: 0.662928
[590]	valid_0's auc: 0.662999
[600]	valid_0's auc: 0.663072
[610]	valid_0's auc: 0.663115
[620]	valid_0's auc: 0.663167
[630]	valid_0's auc: 0.663209
[640]	valid_0's auc: 0.663284
[650]	valid_0's auc: 0.663329
[660]	valid_0's auc: 0.663399
[670]	valid_0's auc: 0.663439
[680]	valid_0's auc: 0.663524
[690]	valid_0's auc: 0.663582
[700]	valid_0's auc: 0.663629
[710]	valid_0's auc: 0.66364
[720]	valid_0's auc: 0.66369
[730]	valid_0's auc: 0.663748
[740]	valid_0's auc: 0.663849
[750]	valid_0's auc: 0.663816
[760]	valid_0's auc: 0.663852
[770]	valid_0's auc: 0.663874
[780]	valid_0's auc: 0.663908
[790]	valid_0's auc: 0.663955
[800]	valid_0's auc: 0.664022
[810]	valid_0's auc: 0.664138
[820]	valid_0's auc: 0.664164
[830]	valid_0's auc: 0.664226
[840]	valid_0's auc: 0.664291
[850]	valid_0's auc: 0.664331
[860]	valid_0's auc: 0.66433
[870]	valid_0's auc: 0.664399
[880]	valid_0's auc: 0.664404
[890]	valid_0's auc: 0.664463
[900]	valid_0's auc: 0.664495
[910]	valid_0's auc: 0.664547
[920]	valid_0's auc: 0.664584
[930]	valid_0's auc: 0.664621
[940]	valid_0's auc: 0.664658
[950]	valid_0's auc: 0.664681
[960]	valid_0's auc: 0.664695
[970]	valid_0's auc: 0.664714
[980]	valid_0's auc: 0.664737
[990]	valid_0's auc: 0.664756
[1000]	valid_0's auc: 0.664782
[1010]	valid_0's auc: 0.664857
[1020]	valid_0's auc: 0.664894
[1030]	valid_0's auc: 0.664938
[1040]	valid_0's auc: 0.664951
[1050]	valid_0's auc: 0.66494
[1060]	valid_0's auc: 0.664955
[1070]	valid_0's auc: 0.664984
[1080]	valid_0's auc: 0.664994
[1090]	valid_0's auc: 0.665006
[1100]	valid_0's auc: 0.665039
[1110]	valid_0's auc: 0.665083
[1120]	valid_0's auc: 0.665111
[1130]	valid_0's auc: 0.66514
[1140]	valid_0's auc: 0.665126
[1150]	valid_0's auc: 0.665131
[1160]	valid_0's auc: 0.665157
[1170]	valid_0's auc: 0.665205
[1180]	valid_0's auc: 0.665252
[1190]	valid_0's auc: 0.665281
[1200]	valid_0's auc: 0.665328
[1210]	valid_0's auc: 0.665322
[1220]	valid_0's auc: 0.665325
[1230]	valid_0's auc: 0.66534
[1240]	valid_0's auc: 0.665369
[1250]	valid_0's auc: 0.665407
[1260]	valid_0's auc: 0.665433
[1270]	valid_0's auc: 0.665479
[1280]	valid_0's auc: 0.665487
[1290]	valid_0's auc: 0.665532
[1300]	valid_0's auc: 0.665589
[1310]	valid_0's auc: 0.665618
[1320]	valid_0's auc: 0.66563
[1330]	valid_0's auc: 0.665669
[1340]	valid_0's auc: 0.665685
[1350]	valid_0's auc: 0.665723
[1360]	valid_0's auc: 0.665716
[1370]	valid_0's auc: 0.665717
[1380]	valid_0's auc: 0.665722
[1390]	valid_0's auc: 0.665732
[1400]	valid_0's auc: 0.665726
[1410]	valid_0's auc: 0.665746
[1420]	valid_0's auc: 0.665801
[1430]	valid_0's auc: 0.665838
[1440]	valid_0's auc: 0.665893
[1450]	valid_0's auc: 0.665903
[1460]	valid_0's auc: 0.665927
[1470]	valid_0's auc: 0.665939
[1480]	valid_0's auc: 0.665959
[1490]	valid_0's auc: 0.665961
[1500]	valid_0's auc: 0.665993
[1510]	valid_0's auc: 0.666023
[1520]	valid_0's auc: 0.666062
[1530]	valid_0's auc: 0.666066
[1540]	valid_0's auc: 0.66606
[1550]	valid_0's auc: 0.666075
[1560]	valid_0's auc: 0.666088
[1570]	valid_0's auc: 0.666113
[1580]	valid_0's auc: 0.666122
[1590]	valid_0's auc: 0.666128
[1600]	valid_0's auc: 0.666141
[1610]	valid_0's auc: 0.666142
[1620]	valid_0's auc: 0.666151
[1630]	valid_0's auc: 0.666154
[1640]	valid_0's auc: 0.66617
[1650]	valid_0's auc: 0.666222
[1660]	valid_0's auc: 0.666239
[1670]	valid_0's auc: 0.66629
[1680]	valid_0's auc: 0.666299
[1690]	valid_0's auc: 0.666298
[1700]	valid_0's auc: 0.666308
[1710]	valid_0's auc: 0.666327
[1720]	valid_0's auc: 0.666333
[1730]	valid_0's auc: 0.666346
[1740]	valid_0's auc: 0.666356
[1750]	valid_0's auc: 0.666398
[1760]	valid_0's auc: 0.666453
[1770]	valid_0's auc: 0.666468
[1780]	valid_0's auc: 0.666474
[1790]	valid_0's auc: 0.66648
[1800]	valid_0's auc: 0.666474
[1810]	valid_0's auc: 0.666493
[1820]	valid_0's auc: 0.666492
[1830]	valid_0's auc: 0.666476
[1840]	valid_0's auc: 0.666488
[1850]	valid_0's auc: 0.666479
[1860]	valid_0's auc: 0.666499
[1870]	valid_0's auc: 0.666534
[1880]	valid_0's auc: 0.666542
[1890]	valid_0's auc: 0.666566
[1900]	valid_0's auc: 0.666594
[1910]	valid_0's auc: 0.666591
[1920]	valid_0's auc: 0.666558
[1930]	valid_0's auc: 0.666603
[1940]	valid_0's auc: 0.666626
[1950]	valid_0's auc: 0.666634
[1960]	valid_0's auc: 0.666643
[1970]	valid_0's auc: 0.666658
[1980]	valid_0's auc: 0.666676
[1990]	valid_0's auc: 0.666682
[2000]	valid_0's auc: 0.666692
[2010]	valid_0's auc: 0.666713
[2020]	valid_0's auc: 0.666711
[2030]	valid_0's auc: 0.666745
[2040]	valid_0's auc: 0.666754
[2050]	valid_0's auc: 0.666753
[2060]	valid_0's auc: 0.666751
[2070]	valid_0's auc: 0.666776
[2080]	valid_0's auc: 0.666772
[2090]	valid_0's auc: 0.666789
[2100]	valid_0's auc: 0.666798
[2110]	valid_0's auc: 0.666824
[2120]	valid_0's auc: 0.666846
[2130]	valid_0's auc: 0.666844
[2140]	valid_0's auc: 0.66686
[2150]	valid_0's auc: 0.666861
[2160]	valid_0's auc: 0.666871
[2170]	valid_0's auc: 0.666877
[2180]	valid_0's auc: 0.666896
[2190]	valid_0's auc: 0.666915
[2200]	valid_0's auc: 0.666947
[2210]	valid_0's auc: 0.666964
[2220]	valid_0's auc: 0.666985
[2230]	valid_0's auc: 0.666997
[2240]	valid_0's auc: 0.667009
[2250]	valid_0's auc: 0.667022
[2260]	valid_0's auc: 0.667015
[2270]	valid_0's auc: 0.66702
[2280]	valid_0's auc: 0.667027
[2290]	valid_0's auc: 0.667036
[2300]	valid_0's auc: 0.667043
[2310]	valid_0's auc: 0.667067
[2320]	valid_0's auc: 0.667069
[2330]	valid_0's auc: 0.667082
[2340]	valid_0's auc: 0.6671
[2350]	valid_0's auc: 0.667103
[2360]	valid_0's auc: 0.667111
[2370]	valid_0's auc: 0.667146
[2380]	valid_0's auc: 0.667158
[2390]	valid_0's auc: 0.667185
[2400]	valid_0's auc: 0.667202
[2410]	valid_0's auc: 0.667225
[2420]	valid_0's auc: 0.667236
[2430]	valid_0's auc: 0.667236
[2440]	valid_0's auc: 0.667227
[2450]	valid_0's auc: 0.667228
[2460]	valid_0's auc: 0.667264
[2470]	valid_0's auc: 0.667273
[2480]	valid_0's auc: 0.667302
[2490]	valid_0's auc: 0.667317
[2500]	valid_0's auc: 0.667331
[2510]	valid_0's auc: 0.667337
[2520]	valid_0's auc: 0.667337
[2530]	valid_0's auc: 0.667353
[2540]	valid_0's auc: 0.667358
[2550]	valid_0's auc: 0.667375
[2560]	valid_0's auc: 0.667391
[2570]	valid_0's auc: 0.667396
[2580]	valid_0's auc: 0.66741
[2590]	valid_0's auc: 0.667424
[2600]	valid_0's auc: 0.667426
[2610]	valid_0's auc: 0.667448
[2620]	valid_0's auc: 0.667447
[2630]	valid_0's auc: 0.667445
[2640]	valid_0's auc: 0.667441
[2650]	valid_0's auc: 0.667431
[2660]	valid_0's auc: 0.667431
Early stopping, best iteration is:
[2613]	valid_0's auc: 0.66745
0.667450178478
2613

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 30
boosting             : gbdt
learning_rate        : 0.07
num_leaves           : 300

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.645378
[20]	valid_0's auc: 0.649326
[30]	valid_0's auc: 0.652991
[40]	valid_0's auc: 0.655609
[50]	valid_0's auc: 0.658104
[60]	valid_0's auc: 0.659389
[70]	valid_0's auc: 0.660726
[80]	valid_0's auc: 0.661425
[90]	valid_0's auc: 0.661952
[100]	valid_0's auc: 0.662614
[110]	valid_0's auc: 0.6634
[120]	valid_0's auc: 0.664053
[130]	valid_0's auc: 0.664424
[140]	valid_0's auc: 0.664938
[150]	valid_0's auc: 0.66538
[160]	valid_0's auc: 0.665724
[170]	valid_0's auc: 0.666139
[180]	valid_0's auc: 0.666467
[190]	valid_0's auc: 0.666988
[200]	valid_0's auc: 0.667371
[210]	valid_0's auc: 0.667703
[220]	valid_0's auc: 0.66794
[230]	valid_0's auc: 0.668195
[240]	valid_0's auc: 0.66844
[250]	valid_0's auc: 0.668683
[260]	valid_0's auc: 0.668884
[270]	valid_0's auc: 0.669053
[280]	valid_0's auc: 0.669235
[290]	valid_0's auc: 0.669359
[300]	valid_0's auc: 0.669454
[310]	valid_0's auc: 0.669618
[320]	valid_0's auc: 0.669786
[330]	valid_0's auc: 0.669877
[340]	valid_0's auc: 0.669911
[350]	valid_0's auc: 0.67009
[360]	valid_0's auc: 0.670261
[370]	valid_0's auc: 0.670454
[380]	valid_0's auc: 0.67051
[390]	valid_0's auc: 0.670641
[400]	valid_0's auc: 0.670688
[410]	valid_0's auc: 0.670751
[420]	valid_0's auc: 0.670884
[430]	valid_0's auc: 0.670977
[440]	valid_0's auc: 0.671115
[450]	valid_0's auc: 0.671154
[460]	valid_0's auc: 0.671206
[470]	valid_0's auc: 0.67124
[480]	valid_0's auc: 0.67132
[490]	valid_0's auc: 0.671448
[500]	valid_0's auc: 0.671462
[510]	valid_0's auc: 0.671496
[520]	valid_0's auc: 0.671495
[530]	valid_0's auc: 0.671532
[540]	valid_0's auc: 0.671592
[550]	valid_0's auc: 0.671621
[560]	valid_0's auc: 0.671672
[570]	valid_0's auc: 0.671729
[580]	valid_0's auc: 0.671758
[590]	valid_0's auc: 0.671808
[600]	valid_0's auc: 0.671885
[610]	valid_0's auc: 0.671926
[620]	valid_0's auc: 0.671989
[630]	valid_0's auc: 0.671997
[640]	valid_0's auc: 0.672017
[650]	valid_0's auc: 0.672061
[660]	valid_0's auc: 0.672106
[670]	valid_0's auc: 0.672115
[680]	valid_0's auc: 0.672169
[690]	valid_0's auc: 0.672227
[700]	valid_0's auc: 0.672239
[710]	valid_0's auc: 0.672254
[720]	valid_0's auc: 0.672265
[730]	valid_0's auc: 0.672276
[740]	valid_0's auc: 0.672308
[750]	valid_0's auc: 0.672357
[760]	valid_0's auc: 0.672386
[770]	valid_0's auc: 0.672384
[780]	valid_0's auc: 0.6724
[790]	valid_0's auc: 0.672439
[800]	valid_0's auc: 0.672462
[810]	valid_0's auc: 0.672481
[820]	valid_0's auc: 0.672482
[830]	valid_0's auc: 0.672539
[840]	valid_0's auc: 0.672547
[850]	valid_0's auc: 0.672565
[860]	valid_0's auc: 0.67259
[870]	valid_0's auc: 0.67264
[880]	valid_0's auc: 0.672647
[890]	valid_0's auc: 0.672666
[900]	valid_0's auc: 0.672669
[910]	valid_0's auc: 0.672685
[920]	valid_0's auc: 0.672711
[930]	valid_0's auc: 0.672712
[940]	valid_0's auc: 0.672709
[950]	valid_0's auc: 0.672714
[960]	valid_0's auc: 0.672711
[970]	valid_0's auc: 0.672737
[980]	valid_0's auc: 0.672721
[990]	valid_0's auc: 0.672714
[1000]	valid_0's auc: 0.672725
[1010]	valid_0's auc: 0.672722
[1020]	valid_0's auc: 0.672738
Early stopping, best iteration is:
[972]	valid_0's auc: 0.672748
0.672748017471
972

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 20
boosting             : gbdt
learning_rate        : 0.1
num_leaves           : 200

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.640042
[20]	valid_0's auc: 0.646305
[30]	valid_0's auc: 0.65043
[40]	valid_0's auc: 0.65292
[50]	valid_0's auc: 0.654154
[60]	valid_0's auc: 0.655255
[70]	valid_0's auc: 0.656582
[80]	valid_0's auc: 0.657526
[90]	valid_0's auc: 0.658158
[100]	valid_0's auc: 0.659057
[110]	valid_0's auc: 0.659961
[120]	valid_0's auc: 0.660543
[130]	valid_0's auc: 0.661276
[140]	valid_0's auc: 0.661795
[150]	valid_0's auc: 0.662181
[160]	valid_0's auc: 0.662612
[170]	valid_0's auc: 0.662978
[180]	valid_0's auc: 0.663459
[190]	valid_0's auc: 0.663957
[200]	valid_0's auc: 0.664332
[210]	valid_0's auc: 0.664749
[220]	valid_0's auc: 0.665049
[230]	valid_0's auc: 0.665254
[240]	valid_0's auc: 0.665551
[250]	valid_0's auc: 0.665833
[260]	valid_0's auc: 0.666027
[270]	valid_0's auc: 0.666288
[280]	valid_0's auc: 0.666485
[290]	valid_0's auc: 0.666648
[300]	valid_0's auc: 0.666836
[310]	valid_0's auc: 0.667007
[320]	valid_0's auc: 0.667312
[330]	valid_0's auc: 0.667439
[340]	valid_0's auc: 0.667593
[350]	valid_0's auc: 0.667679
[360]	valid_0's auc: 0.667709
[370]	valid_0's auc: 0.667835
[380]	valid_0's auc: 0.668072
[390]	valid_0's auc: 0.668177
[400]	valid_0's auc: 0.668329
[410]	valid_0's auc: 0.668457
[420]	valid_0's auc: 0.668556
[430]	valid_0's auc: 0.668638
[440]	valid_0's auc: 0.668696
[450]	valid_0's auc: 0.668856
[460]	valid_0's auc: 0.668949
[470]	valid_0's auc: 0.669006
[480]	valid_0's auc: 0.669054
[490]	valid_0's auc: 0.669162
[500]	valid_0's auc: 0.669235
[510]	valid_0's auc: 0.669257
[520]	valid_0's auc: 0.669333
[530]	valid_0's auc: 0.669437
[540]	valid_0's auc: 0.669511
[550]	valid_0's auc: 0.669577
[560]	valid_0's auc: 0.669652
[570]	valid_0's auc: 0.6697
[580]	valid_0's auc: 0.669739
[590]	valid_0's auc: 0.669857
[600]	valid_0's auc: 0.669898
[610]	valid_0's auc: 0.66994
[620]	valid_0's auc: 0.669925
[630]	valid_0's auc: 0.670005
[640]	valid_0's auc: 0.670044
[650]	valid_0's auc: 0.670182
[660]	valid_0's auc: 0.670182
[670]	valid_0's auc: 0.670329
[680]	valid_0's auc: 0.670329
[690]	valid_0's auc: 0.670432
[700]	valid_0's auc: 0.670468
[710]	valid_0's auc: 0.670563
[720]	valid_0's auc: 0.670594
[730]	valid_0's auc: 0.67063
[740]	valid_0's auc: 0.670677
[750]	valid_0's auc: 0.670679
[760]	valid_0's auc: 0.670682
[770]	valid_0's auc: 0.67072
[780]	valid_0's auc: 0.670867
[790]	valid_0's auc: 0.670931
[800]	valid_0's auc: 0.670948
[810]	valid_0's auc: 0.670957
[820]	valid_0's auc: 0.670975
[830]	valid_0's auc: 0.670987
[840]	valid_0's auc: 0.670988
[850]	valid_0's auc: 0.671014
[860]	valid_0's auc: 0.671094
[870]	valid_0's auc: 0.671151
[880]	valid_0's auc: 0.671185
[890]	valid_0's auc: 0.67119
[900]	valid_0's auc: 0.6712
[910]	valid_0's auc: 0.671241
[920]	valid_0's auc: 0.671253
[930]	valid_0's auc: 0.671306
[940]	valid_0's auc: 0.671387
[950]	valid_0's auc: 0.671419
[960]	valid_0's auc: 0.671366
[970]	valid_0's auc: 0.671353
[980]	valid_0's auc: 0.671359
[990]	valid_0's auc: 0.671385
[1000]	valid_0's auc: 0.671381
Early stopping, best iteration is:
[950]	valid_0's auc: 0.671419
0.671418962746
950

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 20
boosting             : gbdt
learning_rate        : 0.1
num_leaves           : 200

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.640042
[20]	valid_0's auc: 0.646305
[30]	valid_0's auc: 0.65043
[40]	valid_0's auc: 0.65292
[50]	valid_0's auc: 0.654154
[60]	valid_0's auc: 0.655255
[70]	valid_0's auc: 0.656582
[80]	valid_0's auc: 0.657526
[90]	valid_0's auc: 0.658158
[100]	valid_0's auc: 0.659057
[110]	valid_0's auc: 0.659961
[120]	valid_0's auc: 0.660543
[130]	valid_0's auc: 0.661276
[140]	valid_0's auc: 0.661795
[150]	valid_0's auc: 0.662181
[160]	valid_0's auc: 0.662612
[170]	valid_0's auc: 0.662978
[180]	valid_0's auc: 0.663459
[190]	valid_0's auc: 0.663957
[200]	valid_0's auc: 0.664332
[210]	valid_0's auc: 0.664749
[220]	valid_0's auc: 0.665049
[230]	valid_0's auc: 0.665254
[240]	valid_0's auc: 0.665551
[250]	valid_0's auc: 0.665833
[260]	valid_0's auc: 0.666027
[270]	valid_0's auc: 0.666288
[280]	valid_0's auc: 0.666485
[290]	valid_0's auc: 0.666648
[300]	valid_0's auc: 0.666836
[310]	valid_0's auc: 0.667007
[320]	valid_0's auc: 0.667312
[330]	valid_0's auc: 0.667439
[340]	valid_0's auc: 0.667593
[350]	valid_0's auc: 0.667679
[360]	valid_0's auc: 0.667709
[370]	valid_0's auc: 0.667835
[380]	valid_0's auc: 0.668072
[390]	valid_0's auc: 0.668177
[400]	valid_0's auc: 0.668329
[410]	valid_0's auc: 0.668457
[420]	valid_0's auc: 0.668556
[430]	valid_0's auc: 0.668638
[440]	valid_0's auc: 0.668696
[450]	valid_0's auc: 0.668856
[460]	valid_0's auc: 0.668949
[470]	valid_0's auc: 0.669006
[480]	valid_0's auc: 0.669054
[490]	valid_0's auc: 0.669162
[500]	valid_0's auc: 0.669235
[510]	valid_0's auc: 0.669257
[520]	valid_0's auc: 0.669333
[530]	valid_0's auc: 0.669437
[540]	valid_0's auc: 0.669511
[550]	valid_0's auc: 0.669577
[560]	valid_0's auc: 0.669652
[570]	valid_0's auc: 0.6697
[580]	valid_0's auc: 0.669739
[590]	valid_0's auc: 0.669857
[600]	valid_0's auc: 0.669898
[610]	valid_0's auc: 0.66994
[620]	valid_0's auc: 0.669925
[630]	valid_0's auc: 0.670005
[640]	valid_0's auc: 0.670044
[650]	valid_0's auc: 0.670182
[660]	valid_0's auc: 0.670182
[670]	valid_0's auc: 0.670329
[680]	valid_0's auc: 0.670329
[690]	valid_0's auc: 0.670432
[700]	valid_0's auc: 0.670468
[710]	valid_0's auc: 0.670563
[720]	valid_0's auc: 0.670594
[730]	valid_0's auc: 0.67063
[740]	valid_0's auc: 0.670677
[750]	valid_0's auc: 0.670679
[760]	valid_0's auc: 0.670682
[770]	valid_0's auc: 0.67072
[780]	valid_0's auc: 0.670867
[790]	valid_0's auc: 0.670931
[800]	valid_0's auc: 0.670948
[810]	valid_0's auc: 0.670957
[820]	valid_0's auc: 0.670975
[830]	valid_0's auc: 0.670987
[840]	valid_0's auc: 0.670988
[850]	valid_0's auc: 0.671014
[860]	valid_0's auc: 0.671094
[870]	valid_0's auc: 0.671151
[880]	valid_0's auc: 0.671185
[890]	valid_0's auc: 0.67119
[900]	valid_0's auc: 0.6712
[910]	valid_0's auc: 0.671241
[920]	valid_0's auc: 0.671253
[930]	valid_0's auc: 0.671306
[940]	valid_0's auc: 0.671387
[950]	valid_0's auc: 0.671419
[960]	valid_0's auc: 0.671366
[970]	valid_0's auc: 0.671353
[980]	valid_0's auc: 0.671359
[990]	valid_0's auc: 0.671385
[1000]	valid_0's auc: 0.671381
Early stopping, best iteration is:
[950]	valid_0's auc: 0.671419
0.671418962746
950

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 30
boosting             : gbdt
learning_rate        : 0.02
num_leaves           : 100

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.633077
[20]	valid_0's auc: 0.638319
[30]	valid_0's auc: 0.640875
[40]	valid_0's auc: 0.642752
[50]	valid_0's auc: 0.644308
[60]	valid_0's auc: 0.645618
[70]	valid_0's auc: 0.646452
[80]	valid_0's auc: 0.647525
[90]	valid_0's auc: 0.648516
[100]	valid_0's auc: 0.649349
[110]	valid_0's auc: 0.650408
[120]	valid_0's auc: 0.65126
[130]	valid_0's auc: 0.652404
[140]	valid_0's auc: 0.653398
[150]	valid_0's auc: 0.654121
[160]	valid_0's auc: 0.654903
[170]	valid_0's auc: 0.655771
[180]	valid_0's auc: 0.656412
[190]	valid_0's auc: 0.656901
[200]	valid_0's auc: 0.657374
[210]	valid_0's auc: 0.65777
[220]	valid_0's auc: 0.65816
[230]	valid_0's auc: 0.65848
[240]	valid_0's auc: 0.658768
[250]	valid_0's auc: 0.659109
[260]	valid_0's auc: 0.659329
[270]	valid_0's auc: 0.659508
[280]	valid_0's auc: 0.659737
[290]	valid_0's auc: 0.659977
[300]	valid_0's auc: 0.660186
[310]	valid_0's auc: 0.660373
[320]	valid_0's auc: 0.660606
[330]	valid_0's auc: 0.660973
[340]	valid_0's auc: 0.661175
[350]	valid_0's auc: 0.661392
[360]	valid_0's auc: 0.661594
[370]	valid_0's auc: 0.661775
[380]	valid_0's auc: 0.661979
[390]	valid_0's auc: 0.662114
[400]	valid_0's auc: 0.662251
[410]	valid_0's auc: 0.662461
[420]	valid_0's auc: 0.662609
[430]	valid_0's auc: 0.662741
[440]	valid_0's auc: 0.663021
[450]	valid_0's auc: 0.663207
[460]	valid_0's auc: 0.663336
[470]	valid_0's auc: 0.663453
[480]	valid_0's auc: 0.663599
[490]	valid_0's auc: 0.663716
[500]	valid_0's auc: 0.663833
[510]	valid_0's auc: 0.66399
[520]	valid_0's auc: 0.664137
[530]	valid_0's auc: 0.664243
[540]	valid_0's auc: 0.664357
[550]	valid_0's auc: 0.664493
[560]	valid_0's auc: 0.664596
[570]	valid_0's auc: 0.664717
[580]	valid_0's auc: 0.6648
[590]	valid_0's auc: 0.664925
[600]	valid_0's auc: 0.665032
[610]	valid_0's auc: 0.665149
[620]	valid_0's auc: 0.665278
[630]	valid_0's auc: 0.665385
[640]	valid_0's auc: 0.66554
[650]	valid_0's auc: 0.665663
[660]	valid_0's auc: 0.665812
[670]	valid_0's auc: 0.66591
[680]	valid_0's auc: 0.666002
[690]	valid_0's auc: 0.666077
[700]	valid_0's auc: 0.666197
[710]	valid_0's auc: 0.666302
[720]	valid_0's auc: 0.666402
[730]	valid_0's auc: 0.666462
[740]	valid_0's auc: 0.666547
[750]	valid_0's auc: 0.666641
[760]	valid_0's auc: 0.666725
[770]	valid_0's auc: 0.666796
[780]	valid_0's auc: 0.666879
[790]	valid_0's auc: 0.666945
[800]	valid_0's auc: 0.667031
[810]	valid_0's auc: 0.6671
[820]	valid_0's auc: 0.667173
[830]	valid_0's auc: 0.667251
[840]	valid_0's auc: 0.667334
[850]	valid_0's auc: 0.667442
[860]	valid_0's auc: 0.667525
[870]	valid_0's auc: 0.667578
[880]	valid_0's auc: 0.667612
[890]	valid_0's auc: 0.667689
[900]	valid_0's auc: 0.667757
[910]	valid_0's auc: 0.667812
[920]	valid_0's auc: 0.66788
[930]	valid_0's auc: 0.66793
[940]	valid_0's auc: 0.668003
[950]	valid_0's auc: 0.668062
[960]	valid_0's auc: 0.668137
[970]	valid_0's auc: 0.668231
[980]	valid_0's auc: 0.66827
[990]	valid_0's auc: 0.668325
[1000]	valid_0's auc: 0.668378
[1010]	valid_0's auc: 0.668409
[1020]	valid_0's auc: 0.668455
[1030]	valid_0's auc: 0.668481
[1040]	valid_0's auc: 0.668525
[1050]	valid_0's auc: 0.668572
[1060]	valid_0's auc: 0.668615
[1070]	valid_0's auc: 0.668648
[1080]	valid_0's auc: 0.668687
[1090]	valid_0's auc: 0.668727
[1100]	valid_0's auc: 0.668752
[1110]	valid_0's auc: 0.668786
[1120]	valid_0's auc: 0.668825
[1130]	valid_0's auc: 0.668873
[1140]	valid_0's auc: 0.668897
[1150]	valid_0's auc: 0.668925
[1160]	valid_0's auc: 0.668964
[1170]	valid_0's auc: 0.669006
[1180]	valid_0's auc: 0.669027
[1190]	valid_0's auc: 0.669062
[1200]	valid_0's auc: 0.66909
[1210]	valid_0's auc: 0.669122
[1220]	valid_0's auc: 0.669158
[1230]	valid_0's auc: 0.669183
[1240]	valid_0's auc: 0.669218
[1250]	valid_0's auc: 0.669251
[1260]	valid_0's auc: 0.669274
[1270]	valid_0's auc: 0.669299
[1280]	valid_0's auc: 0.669346
[1290]	valid_0's auc: 0.669377
[1300]	valid_0's auc: 0.669413
[1310]	valid_0's auc: 0.669442
[1320]	valid_0's auc: 0.669484
[1330]	valid_0's auc: 0.669532
[1340]	valid_0's auc: 0.669554
[1350]	valid_0's auc: 0.669571
[1360]	valid_0's auc: 0.669583
[1370]	valid_0's auc: 0.669624
[1380]	valid_0's auc: 0.669658
[1390]	valid_0's auc: 0.669679
[1400]	valid_0's auc: 0.669711
[1410]	valid_0's auc: 0.669716
[1420]	valid_0's auc: 0.669741
[1430]	valid_0's auc: 0.669766
[1440]	valid_0's auc: 0.669789
[1450]	valid_0's auc: 0.66981
[1460]	valid_0's auc: 0.669837
[1470]	valid_0's auc: 0.669868
[1480]	valid_0's auc: 0.669898
[1490]	valid_0's auc: 0.669947
[1500]	valid_0's auc: 0.669969
[1510]	valid_0's auc: 0.669978
[1520]	valid_0's auc: 0.670008
[1530]	valid_0's auc: 0.670023
[1540]	valid_0's auc: 0.670047
[1550]	valid_0's auc: 0.670075
[1560]	valid_0's auc: 0.670091
[1570]	valid_0's auc: 0.6701
[1580]	valid_0's auc: 0.670111
[1590]	valid_0's auc: 0.67012
[1600]	valid_0's auc: 0.670145
[1610]	valid_0's auc: 0.670163
[1620]	valid_0's auc: 0.670167
[1630]	valid_0's auc: 0.670172
[1640]	valid_0's auc: 0.670182
[1650]	valid_0's auc: 0.670198
[1660]	valid_0's auc: 0.670242
[1670]	valid_0's auc: 0.670254
[1680]	valid_0's auc: 0.670272
[1690]	valid_0's auc: 0.670282
[1700]	valid_0's auc: 0.670289
[1710]	valid_0's auc: 0.670298
[1720]	valid_0's auc: 0.670308
[1730]	valid_0's auc: 0.670322
[1740]	valid_0's auc: 0.67031
[1750]	valid_0's auc: 0.670316
[1760]	valid_0's auc: 0.670326
[1770]	valid_0's auc: 0.670326
[1780]	valid_0's auc: 0.670329
[1790]	valid_0's auc: 0.670349
[1800]	valid_0's auc: 0.670362
[1810]	valid_0's auc: 0.670359
[1820]	valid_0's auc: 0.670376
[1830]	valid_0's auc: 0.670374
[1840]	valid_0's auc: 0.670384
[1850]	valid_0's auc: 0.670389
[1860]	valid_0's auc: 0.670402
[1870]	valid_0's auc: 0.670412
[1880]	valid_0's auc: 0.670432
[1890]	valid_0's auc: 0.67046
[1900]	valid_0's auc: 0.670472
[1910]	valid_0's auc: 0.67048
[1920]	valid_0's auc: 0.670493
[1930]	valid_0's auc: 0.670506
[1940]	valid_0's auc: 0.670514
[1950]	valid_0's auc: 0.670528
[1960]	valid_0's auc: 0.670528
[1970]	valid_0's auc: 0.670535
[1980]	valid_0's auc: 0.670537
[1990]	valid_0's auc: 0.670548
[2000]	valid_0's auc: 0.670546
[2010]	valid_0's auc: 0.670568
[2020]	valid_0's auc: 0.670584
[2030]	valid_0's auc: 0.670591
[2040]	valid_0's auc: 0.670614
[2050]	valid_0's auc: 0.670615
[2060]	valid_0's auc: 0.670618
[2070]	valid_0's auc: 0.670623
[2080]	valid_0's auc: 0.670618
[2090]	valid_0's auc: 0.670619
[2100]	valid_0's auc: 0.670617
[2110]	valid_0's auc: 0.670624
[2120]	valid_0's auc: 0.670628
[2130]	valid_0's auc: 0.670639
[2140]	valid_0's auc: 0.67064
[2150]	valid_0's auc: 0.670645
[2160]	valid_0's auc: 0.670649
[2170]	valid_0's auc: 0.67065
[2180]	valid_0's auc: 0.670656
[2190]	valid_0's auc: 0.670661
[2200]	valid_0's auc: 0.670676
[2210]	valid_0's auc: 0.670676
[2220]	valid_0's auc: 0.670677
[2230]	valid_0's auc: 0.670688
[2240]	valid_0's auc: 0.670684
[2250]	valid_0's auc: 0.670703
[2260]	valid_0's auc: 0.670716
[2270]	valid_0's auc: 0.670723
[2280]	valid_0's auc: 0.670738
[2290]	valid_0's auc: 0.670747
[2300]	valid_0's auc: 0.670754
[2310]	valid_0's auc: 0.670761
[2320]	valid_0's auc: 0.670778
[2330]	valid_0's auc: 0.670785
[2340]	valid_0's auc: 0.670791
[2350]	valid_0's auc: 0.670801
[2360]	valid_0's auc: 0.670813
[2370]	valid_0's auc: 0.670826
[2380]	valid_0's auc: 0.670829
[2390]	valid_0's auc: 0.670849
[2400]	valid_0's auc: 0.67086
[2410]	valid_0's auc: 0.670861
[2420]	valid_0's auc: 0.670869
[2430]	valid_0's auc: 0.670874
[2440]	valid_0's auc: 0.670887
[2450]	valid_0's auc: 0.670916
[2460]	valid_0's auc: 0.670931
[2470]	valid_0's auc: 0.670941
[2480]	valid_0's auc: 0.670941
[2490]	valid_0's auc: 0.670952
[2500]	valid_0's auc: 0.670963
[2510]	valid_0's auc: 0.670959
[2520]	valid_0's auc: 0.670964
[2530]	valid_0's auc: 0.670974
[2540]	valid_0's auc: 0.670983
[2550]	valid_0's auc: 0.670981
[2560]	valid_0's auc: 0.670991
[2570]	valid_0's auc: 0.671002
[2580]	valid_0's auc: 0.671
[2590]	valid_0's auc: 0.671014
[2600]	valid_0's auc: 0.671022
[2610]	valid_0's auc: 0.671022
[2620]	valid_0's auc: 0.671026
[2630]	valid_0's auc: 0.671036
[2640]	valid_0's auc: 0.671045
[2650]	valid_0's auc: 0.67105
[2660]	valid_0's auc: 0.671052
[2670]	valid_0's auc: 0.671055
[2680]	valid_0's auc: 0.671055
[2690]	valid_0's auc: 0.671058
[2700]	valid_0's auc: 0.671074
[2710]	valid_0's auc: 0.671076
[2720]	valid_0's auc: 0.67109
[2730]	valid_0's auc: 0.671093
[2740]	valid_0's auc: 0.671112
[2750]	valid_0's auc: 0.671143
[2760]	valid_0's auc: 0.671149
[2770]	valid_0's auc: 0.671187
[2780]	valid_0's auc: 0.671194
[2790]	valid_0's auc: 0.671198
[2800]	valid_0's auc: 0.671205
[2810]	valid_0's auc: 0.67122
[2820]	valid_0's auc: 0.671219
[2830]	valid_0's auc: 0.671218
[2840]	valid_0's auc: 0.671242
[2850]	valid_0's auc: 0.671244
[2860]	valid_0's auc: 0.671269
[2870]	valid_0's auc: 0.671278
[2880]	valid_0's auc: 0.671276
[2890]	valid_0's auc: 0.671285
[2900]	valid_0's auc: 0.671288
[2910]	valid_0's auc: 0.671291
[2920]	valid_0's auc: 0.671296
[2930]	valid_0's auc: 0.671288
[2940]	valid_0's auc: 0.671302
[2950]	valid_0's auc: 0.671303
[2960]	valid_0's auc: 0.671313
[2970]	valid_0's auc: 0.671316
[2980]	valid_0's auc: 0.671322
[2990]	valid_0's auc: 0.671334
[3000]	valid_0's auc: 0.671335
[3010]	valid_0's auc: 0.671339
[3020]	valid_0's auc: 0.67135
[3030]	valid_0's auc: 0.671363
[3040]	valid_0's auc: 0.67136
[3050]	valid_0's auc: 0.671363
[3060]	valid_0's auc: 0.671364
[3070]	valid_0's auc: 0.671361
[3080]	valid_0's auc: 0.671367
Early stopping, best iteration is:
[3034]	valid_0's auc: 0.671369
0.67136943719
3034

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_depth            : 20
boosting             : gbdt
learning_rate        : 0.02
num_leaves           : 200

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.63039
[20]	valid_0's auc: 0.636016
[30]	valid_0's auc: 0.638073
[40]	valid_0's auc: 0.639939
[50]	valid_0's auc: 0.641762
[60]	valid_0's auc: 0.642985
[70]	valid_0's auc: 0.644016
[80]	valid_0's auc: 0.645352
[90]	valid_0's auc: 0.64608
[100]	valid_0's auc: 0.646837
[110]	valid_0's auc: 0.647506
[120]	valid_0's auc: 0.648229
[130]	valid_0's auc: 0.648914
[140]	valid_0's auc: 0.649544
[150]	valid_0's auc: 0.650307
[160]	valid_0's auc: 0.651132
[170]	valid_0's auc: 0.651766
[180]	valid_0's auc: 0.652237
[190]	valid_0's auc: 0.652621
[200]	valid_0's auc: 0.653049
[210]	valid_0's auc: 0.653546
[220]	valid_0's auc: 0.653866
[230]	valid_0's auc: 0.654082
[240]	valid_0's auc: 0.654262
[250]	valid_0's auc: 0.65442
[260]	valid_0's auc: 0.654625
[270]	valid_0's auc: 0.654799
[280]	valid_0's auc: 0.654966
[290]	valid_0's auc: 0.655274
[300]	valid_0's auc: 0.655543
[310]	valid_0's auc: 0.655793
[320]	valid_0's auc: 0.656028
[330]	valid_0's auc: 0.656298
[340]	valid_0's auc: 0.656569
[350]	valid_0's auc: 0.656778
[360]	valid_0's auc: 0.656898
[370]	valid_0's auc: 0.657042
[380]	valid_0's auc: 0.657205
[390]	valid_0's auc: 0.657331
[400]	valid_0's auc: 0.65754
[410]	valid_0's auc: 0.657768
[420]	valid_0's auc: 0.657928
[430]	valid_0's auc: 0.65813
[440]	valid_0's auc: 0.658341
[450]	valid_0's auc: 0.658587
[460]	valid_0's auc: 0.658711
[470]	valid_0's auc: 0.658821
[480]	valid_0's auc: 0.658989
[490]	valid_0's auc: 0.659166
[500]	valid_0's auc: 0.65936
[510]	valid_0's auc: 0.65954
[520]	valid_0's auc: 0.659663
[530]	valid_0's auc: 0.659811
[540]	valid_0's auc: 0.659952
[550]	valid_0's auc: 0.660079
[560]	valid_0's auc: 0.660243
[570]	valid_0's auc: 0.66041
[580]	valid_0's auc: 0.660521
[590]	valid_0's auc: 0.660648
[600]	valid_0's auc: 0.660823
[610]	valid_0's auc: 0.660932
[620]	valid_0's auc: 0.661064
[630]	valid_0's auc: 0.661168
[640]	valid_0's auc: 0.66126
[650]	valid_0's auc: 0.661397
[660]	valid_0's auc: 0.661586
[670]	valid_0's auc: 0.661663
[680]	valid_0's auc: 0.661804
[690]	valid_0's auc: 0.661913
[700]	valid_0's auc: 0.661977
[710]	valid_0's auc: 0.662142
[720]	valid_0's auc: 0.662268
[730]	valid_0's auc: 0.662349
[740]	valid_0's auc: 0.662415
[750]	valid_0's auc: 0.662497
[760]	valid_0's auc: 0.662578
[770]	valid_0's auc: 0.662671
[780]	valid_0's auc: 0.662753
[790]	valid_0's auc: 0.662858
[800]	valid_0's auc: 0.662951
[810]	valid_0's auc: 0.663047
[820]	valid_0's auc: 0.663158
[830]	valid_0's auc: 0.663287
[840]	valid_0's auc: 0.663384
[850]	valid_0's auc: 0.663495
[860]	valid_0's auc: 0.66358
[870]	valid_0's auc: 0.663715
[880]	valid_0's auc: 0.663787
[890]	valid_0's auc: 0.663917
[900]	valid_0's auc: 0.663984
[910]	valid_0's auc: 0.66407
[920]	valid_0's auc: 0.664149
[930]	valid_0's auc: 0.664224
[940]	valid_0's auc: 0.664298
[950]	valid_0's auc: 0.664394
[960]	valid_0's auc: 0.664467
[970]	valid_0's auc: 0.664544
[980]	valid_0's auc: 0.664642
[990]	valid_0's auc: 0.664733
[1000]	valid_0's auc: 0.664816
[1010]	valid_0's auc: 0.664886
[1020]	valid_0's auc: 0.665002
[1030]	valid_0's auc: 0.665056
[1040]	valid_0's auc: 0.665163
[1050]	valid_0's auc: 0.665214
[1060]	valid_0's auc: 0.665282
[1070]	valid_0's auc: 0.66537
[1080]	valid_0's auc: 0.665479
[1090]	valid_0's auc: 0.665533
[1100]	valid_0's auc: 0.665586
[1110]	valid_0's auc: 0.665697
[1120]	valid_0's auc: 0.665792
[1130]	valid_0's auc: 0.665896
[1140]	valid_0's auc: 0.665985
[1150]	valid_0's auc: 0.666061
[1160]	valid_0's auc: 0.666124
[1170]	valid_0's auc: 0.666167
[1180]	valid_0's auc: 0.666242
[1190]	valid_0's auc: 0.666295
[1200]	valid_0's auc: 0.666333
[1210]	valid_0's auc: 0.666382
[1220]	valid_0's auc: 0.666414
[1230]	valid_0's auc: 0.666481
[1240]	valid_0's auc: 0.666511
[1250]	valid_0's auc: 0.666555
[1260]	valid_0's auc: 0.666592
[1270]	valid_0's auc: 0.66667
[1280]	valid_0's auc: 0.66673
[1290]	valid_0's auc: 0.666771
[1300]	valid_0's auc: 0.666826
[1310]	valid_0's auc: 0.666865
[1320]	valid_0's auc: 0.666933
[1330]	valid_0's auc: 0.666971
[1340]	valid_0's auc: 0.667023
[1350]	valid_0's auc: 0.667057
[1360]	valid_0's auc: 0.667089
[1370]	valid_0's auc: 0.667132
[1380]	valid_0's auc: 0.667159
[1390]	valid_0's auc: 0.667214
[1400]	valid_0's auc: 0.667275
[1410]	valid_0's auc: 0.667305
[1420]	valid_0's auc: 0.667334
[1430]	valid_0's auc: 0.667372
[1440]	valid_0's auc: 0.667393
[1450]	valid_0's auc: 0.667428
[1460]	valid_0's auc: 0.66746
[1470]	valid_0's auc: 0.667497
[1480]	valid_0's auc: 0.667527
[1490]	valid_0's auc: 0.667568
[1500]	valid_0's auc: 0.667608
[1510]	valid_0's auc: 0.667617
[1520]	valid_0's auc: 0.667634
[1530]	valid_0's auc: 0.66766
[1540]	valid_0's auc: 0.667681
[1550]	valid_0's auc: 0.667711
[1560]	valid_0's auc: 0.667762
[1570]	valid_0's auc: 0.667798
[1580]	valid_0's auc: 0.667848
[1590]	valid_0's auc: 0.667898
[1600]	valid_0's auc: 0.667916
[1610]	valid_0's auc: 0.667956
[1620]	valid_0's auc: 0.667979
[1630]	valid_0's auc: 0.667988
[1640]	valid_0's auc: 0.668027
[1650]	valid_0's auc: 0.668053
[1660]	valid_0's auc: 0.668089
[1670]	valid_0's auc: 0.668114
[1680]	valid_0's auc: 0.668135
[1690]	valid_0's auc: 0.668163
[1700]	valid_0's auc: 0.668183
[1710]	valid_0's auc: 0.668227
[1720]	valid_0's auc: 0.668271
[1730]	valid_0's auc: 0.668295
[1740]	valid_0's auc: 0.668335
[1750]	valid_0's auc: 0.668369
[1760]	valid_0's auc: 0.668407
[1770]	valid_0's auc: 0.668438
[1780]	valid_0's auc: 0.668476
[1790]	valid_0's auc: 0.668516
[1800]	valid_0's auc: 0.668549
[1810]	valid_0's auc: 0.66856
[1820]	valid_0's auc: 0.668587
[1830]	valid_0's auc: 0.668599
[1840]	valid_0's auc: 0.668615
[1850]	valid_0's auc: 0.668625
[1860]	valid_0's auc: 0.668654
[1870]	valid_0's auc: 0.66867
[1880]	valid_0's auc: 0.66871
[1890]	valid_0's auc: 0.668752
[1900]	valid_0's auc: 0.668782
[1910]	valid_0's auc: 0.668814
[1920]	valid_0's auc: 0.668837
[1930]	valid_0's auc: 0.668853
[1940]	valid_0's auc: 0.668894
[1950]	valid_0's auc: 0.668914
[1960]	valid_0's auc: 0.668925
[1970]	valid_0's auc: 0.668966
[1980]	valid_0's auc: 0.668995
[1990]	valid_0's auc: 0.669004
[2000]	valid_0's auc: 0.669022
[2010]	valid_0's auc: 0.669025
[2020]	valid_0's auc: 0.66905
[2030]	valid_0's auc: 0.669088
[2040]	valid_0's auc: 0.66911
[2050]	valid_0's auc: 0.669113
[2060]	valid_0's auc: 0.669144
[2070]	valid_0's auc: 0.669152
[2080]	valid_0's auc: 0.669166
[2090]	valid_0's auc: 0.669191
[2100]	valid_0's auc: 0.669205
[2110]	valid_0's auc: 0.669232
[2120]	valid_0's auc: 0.669254
[2130]	valid_0's auc: 0.669262
[2140]	valid_0's auc: 0.669337
[2150]	valid_0's auc: 0.669359
[2160]	valid_0's auc: 0.669377
[2170]	valid_0's auc: 0.669395
[2180]	valid_0's auc: 0.669428
[2190]	valid_0's auc: 0.669465
[2200]	valid_0's auc: 0.669487
[2210]	valid_0's auc: 0.669517
[2220]	valid_0's auc: 0.669538
[2230]	valid_0's auc: 0.669542
[2240]	valid_0's auc: 0.669554
[2250]	valid_0's auc: 0.669565
[2260]	valid_0's auc: 0.669576
[2270]	valid_0's auc: 0.669586
[2280]	valid_0's auc: 0.669591
[2290]	valid_0's auc: 0.669605
[2300]	valid_0's auc: 0.669625
[2310]	valid_0's auc: 0.669628
[2320]	valid_0's auc: 0.669635
[2330]	valid_0's auc: 0.669638
[2340]	valid_0's auc: 0.669655
[2350]	valid_0's auc: 0.669671
[2360]	valid_0's auc: 0.669697
[2370]	valid_0's auc: 0.669712
[2380]	valid_0's auc: 0.669746
[2390]	valid_0's auc: 0.669759
[2400]	valid_0's auc: 0.669768
[2410]	valid_0's auc: 0.669791
[2420]	valid_0's auc: 0.669789
[2430]	valid_0's auc: 0.669811
[2440]	valid_0's auc: 0.669823
[2450]	valid_0's auc: 0.669866
[2460]	valid_0's auc: 0.669873
[2470]	valid_0's auc: 0.669892
[2480]	valid_0's auc: 0.669946
[2490]	valid_0's auc: 0.669967
[2500]	valid_0's auc: 0.66997
[2510]	valid_0's auc: 0.669965
[2520]	valid_0's auc: 0.669968
[2530]	valid_0's auc: 0.669967
[2540]	valid_0's auc: 0.669976
[2550]	valid_0's auc: 0.669993
[2560]	valid_0's auc: 0.670013
[2570]	valid_0's auc: 0.670035
[2580]	valid_0's auc: 0.670048
[2590]	valid_0's auc: 0.670052
[2600]	valid_0's auc: 0.670056
[2610]	valid_0's auc: 0.670064
[2620]	valid_0's auc: 0.670077
[2630]	valid_0's auc: 0.670082
[2640]	valid_0's auc: 0.67009
[2650]	valid_0's auc: 0.670098
[2660]	valid_0's auc: 0.670128
[2670]	valid_0's auc: 0.670162
[2680]	valid_0's auc: 0.670169
[2690]	valid_0's auc: 0.670203
[2700]	valid_0's auc: 0.670227
[2710]	valid_0's auc: 0.670235
[2720]	valid_0's auc: 0.670239
[2730]	valid_0's auc: 0.670245
[2740]	valid_0's auc: 0.670245
[2750]	valid_0's auc: 0.670249
[2760]	valid_0's auc: 0.670259
[2770]	valid_0's auc: 0.670262
[2780]	valid_0's auc: 0.670272
[2790]	valid_0's auc: 0.670273
[2800]	valid_0's auc: 0.670278
[2810]	valid_0's auc: 0.670291
[2820]	valid_0's auc: 0.670336
[2830]	valid_0's auc: 0.670367
[2840]	valid_0's auc: 0.670387
[2850]	valid_0's auc: 0.670419
[2860]	valid_0's auc: 0.670451
[2870]	valid_0's auc: 0.670461
[2880]	valid_0's auc: 0.67047
[2890]	valid_0's auc: 0.670472
[2900]	valid_0's auc: 0.670484
[2910]	valid_0's auc: 0.670493
[2920]	valid_0's auc: 0.670511
[2930]	valid_0's auc: 0.670527
[2940]	valid_0's auc: 0.67056
[2950]	valid_0's auc: 0.670559
[2960]	valid_0's auc: 0.670567
[2970]	valid_0's auc: 0.670567
[2980]	valid_0's auc: 0.670588
[2990]	valid_0's auc: 0.670598
[3000]	valid_0's auc: 0.670605
[3010]	valid_0's auc: 0.670608
[3020]	valid_0's auc: 0.67061
[3030]	valid_0's auc: 0.670619
[3040]	valid_0's auc: 0.670653
[3050]	valid_0's auc: 0.670687
[3060]	valid_0's auc: 0.670723
[3070]	valid_0's auc: 0.670739
[3080]	valid_0's auc: 0.67075
[3090]	valid_0's auc: 0.670762
[3100]	valid_0's auc: 0.670759
[3110]	valid_0's auc: 0.670757
[3120]	valid_0's auc: 0.670763
[3130]	valid_0's auc: 0.670772
[3140]	valid_0's auc: 0.67078
[3150]	valid_0's auc: 0.670796
[3160]	valid_0's auc: 0.670813
[3170]	valid_0's auc: 0.670855
[3180]	valid_0's auc: 0.670897
[3190]	valid_0's auc: 0.670913
[3200]	valid_0's auc: 0.670927
[3210]	valid_0's auc: 0.67094
[3220]	valid_0's auc: 0.670946
[3230]	valid_0's auc: 0.670954
[3240]	valid_0's auc: 0.670962
[3250]	valid_0's auc: 0.670965
[3260]	valid_0's auc: 0.670967
[3270]	valid_0's auc: 0.670988
[3280]	valid_0's auc: 0.671015
[3290]	valid_0's auc: 0.671041
[3300]	valid_0's auc: 0.671048
[3310]	valid_0's auc: 0.671054
[3320]	valid_0's auc: 0.671076
[3330]	valid_0's auc: 0.671102
[3340]	valid_0's auc: 0.671108
[3350]	valid_0's auc: 0.671114
[3360]	valid_0's auc: 0.671119
[3370]	valid_0's auc: 0.671122
[3380]	valid_0's auc: 0.671122
[3390]	valid_0's auc: 0.671121
[3400]	valid_0's auc: 0.671124
[3410]	valid_0's auc: 0.671131
[3420]	valid_0's auc: 0.671156
[3430]	valid_0's auc: 0.671156
[3440]	valid_0's auc: 0.671158
[3450]	valid_0's auc: 0.671163
[3460]	valid_0's auc: 0.67116
[3470]	valid_0's auc: 0.671167
[3480]	valid_0's auc: 0.67118
[3490]	valid_0's auc: 0.671201
[3500]	valid_0's auc: 0.6712
[3510]	valid_0's auc: 0.671211
[3520]	valid_0's auc: 0.671246
[3530]	valid_0's auc: 0.671281
[3540]	valid_0's auc: 0.671301
[3550]	valid_0's auc: 0.67131
[3560]	valid_0's auc: 0.671321
[3570]	valid_0's auc: 0.671331
[3580]	valid_0's auc: 0.67134
[3590]	valid_0's auc: 0.671342
[3600]	valid_0's auc: 0.671351
[3610]	valid_0's auc: 0.671364
[3620]	valid_0's auc: 0.671367
[3630]	valid_0's auc: 0.671381
[3640]	valid_0's auc: 0.67139
[3650]	valid_0's auc: 0.671396
[3660]	valid_0's auc: 0.671399
[3670]	valid_0's auc: 0.671413
[3680]	valid_0's auc: 0.671419
[3690]	valid_0's auc: 0.671428
[3700]	valid_0's auc: 0.671442
[3710]	valid_0's auc: 0.671446
[3720]	valid_0's auc: 0.67145
[3730]	valid_0's auc: 0.671459
[3740]	valid_0's auc: 0.671462
[3750]	valid_0's auc: 0.671467
[3760]	valid_0's auc: 0.671479
[3770]	valid_0's auc: 0.671483
[3780]	valid_0's auc: 0.671486
[3790]	valid_0's auc: 0.671492
[3800]	valid_0's auc: 0.6715
[3810]	valid_0's auc: 0.671512
[3820]	valid_0's auc: 0.671514
[3830]	valid_0's auc: 0.671525
[3840]	valid_0's auc: 0.671532
[3850]	valid_0's auc: 0.671545
[3860]	valid_0's auc: 0.671555
[3870]	valid_0's auc: 0.671571
[3880]	valid_0's auc: 0.671579
[3890]	valid_0's auc: 0.671587
[3900]	valid_0's auc: 0.671589
[3910]	valid_0's auc: 0.671591
[3920]	valid_0's auc: 0.671599
[3930]	valid_0's auc: 0.6716
[3940]	valid_0's auc: 0.671608
[3950]	valid_0's auc: 0.671621
[3960]	valid_0's auc: 0.671623
[3970]	valid_0's auc: 0.671628
[3980]	valid_0's auc: 0.671632
[3990]	valid_0's auc: 0.671635
[4000]	valid_0's auc: 0.671658
[4010]	valid_0's auc: 0.671668
[4020]	valid_0's auc: 0.671672
[4030]	valid_0's auc: 0.671696
[4040]	valid_0's auc: 0.671718
[4050]	valid_0's auc: 0.671727
[4060]	valid_0's auc: 0.671727
[4070]	valid_0's auc: 0.671737
[4080]	valid_0's auc: 0.67174
[4090]	valid_0's auc: 0.671751
[4100]	valid_0's auc: 0.671756
[4110]	valid_0's auc: 0.671755
[4120]	valid_0's auc: 0.67176
[4130]	valid_0's auc: 0.671762
[4140]	valid_0's auc: 0.671767
[4150]	valid_0's auc: 0.671778
[4160]	valid_0's auc: 0.671779
[4170]	valid_0's auc: 0.671787
[4180]	valid_0's auc: 0.671792
[4190]	valid_0's auc: 0.671798
[4200]	valid_0's auc: 0.671803
[4210]	valid_0's auc: 0.671809
[4220]	valid_0's auc: 0.671816
[4230]	valid_0's auc: 0.671812
[4240]	valid_0's auc: 0.671819
[4250]	valid_0's auc: 0.67182
[4260]	valid_0's auc: 0.671831
[4270]	valid_0's auc: 0.671831
[4280]	valid_0's auc: 0.671833
[4290]	valid_0's auc: 0.671838
[4300]	valid_0's auc: 0.671846
[4310]	valid_0's auc: 0.671853
[4320]	valid_0's auc: 0.67186
[4330]	valid_0's auc: 0.671861
[4340]	valid_0's auc: 0.671864
[4350]	valid_0's auc: 0.671868
[4360]	valid_0's auc: 0.671869
[4370]	valid_0's auc: 0.671875
[4380]	valid_0's auc: 0.67189
[4390]	valid_0's auc: 0.671898
[4400]	valid_0's auc: 0.671903
[4410]	valid_0's auc: 0.671921
[4420]	valid_0's auc: 0.671922
[4430]	valid_0's auc: 0.671926
[4440]	valid_0's auc: 0.671926
[4450]	valid_0's auc: 0.671931
[4460]	valid_0's auc: 0.67194
[4470]	valid_0's auc: 0.671945
[4480]	valid_0's auc: 0.671944
[4490]	valid_0's auc: 0.671948
[4500]	valid_0's auc: 0.671958
[4510]	valid_0's auc: 0.671966
[4520]	valid_0's auc: 0.671972
[4530]	valid_0's auc: 0.67197
[4540]	valid_0's auc: 0.67198
[4550]	valid_0's auc: 0.671979
[4560]	valid_0's auc: 0.671976
[4570]	valid_0's auc: 0.671982
[4580]	valid_0's auc: 0.671989
[4590]	valid_0's auc: 0.671995
[4600]	valid_0's auc: 0.671999
[4610]	valid_0's auc: 0.672005
[4620]	valid_0's auc: 0.672008
[4630]	valid_0's auc: 0.672009
[4640]	valid_0's auc: 0.672014
[4650]	valid_0's auc: 0.672014
[4660]	valid_0's auc: 0.672026
[4670]	valid_0's auc: 0.672024
[4680]	valid_0's auc: 0.672026
[4690]	valid_0's auc: 0.672022
[4700]	valid_0's auc: 0.672029
[4710]	valid_0's auc: 0.672033
[4720]	valid_0's auc: 0.672034
[4730]	valid_0's auc: 0.672038
[4740]	valid_0's auc: 0.672038
[4750]	valid_0's auc: 0.672042
[4760]	valid_0's auc: 0.672045
[4770]	valid_0's auc: 0.672054
[4780]	valid_0's auc: 0.672067
[4790]	valid_0's auc: 0.672071
[4800]	valid_0's auc: 0.67208
[4810]	valid_0's auc: 0.67208
[4820]	valid_0's auc: 0.672084
[4830]	valid_0's auc: 0.67209
[4840]	valid_0's auc: 0.672088
[4850]	valid_0's auc: 0.67209
[4860]	valid_0's auc: 0.672099
[4870]	valid_0's auc: 0.672102
[4880]	valid_0's auc: 0.672104
[4890]	valid_0's auc: 0.672107
[4900]	valid_0's auc: 0.672109
[4910]	valid_0's auc: 0.672115
[4920]	valid_0's auc: 0.672124
[4930]	valid_0's auc: 0.672129
[4940]	valid_0's auc: 0.672132
[4950]	valid_0's auc: 0.672136
[4960]	valid_0's auc: 0.67214
[4970]	valid_0's auc: 0.672149
[4980]	valid_0's auc: 0.672147
[4990]	valid_0's auc: 0.672152
[5000]	valid_0's auc: 0.672158
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/playground_V1006/training_V1304.py", line 150, in <module>
    verbose_eval=10,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 1
'''