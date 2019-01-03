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

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1304.py
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
max_depth            : 20
learning_rate        : 0.02
boosting             : gbdt

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.62916
[20]	valid_0's auc: 0.635496
[30]	valid_0's auc: 0.637365
[40]	valid_0's auc: 0.639498
[50]	valid_0's auc: 0.640839
[60]	valid_0's auc: 0.641998
[70]	valid_0's auc: 0.64315
[80]	valid_0's auc: 0.64425
[90]	valid_0's auc: 0.645091
[100]	valid_0's auc: 0.645859
[110]	valid_0's auc: 0.646724
[120]	valid_0's auc: 0.64765
[130]	valid_0's auc: 0.648572
[140]	valid_0's auc: 0.649331
[150]	valid_0's auc: 0.650033
[160]	valid_0's auc: 0.650794
[170]	valid_0's auc: 0.651431
[180]	valid_0's auc: 0.651891
[190]	valid_0's auc: 0.652282
[200]	valid_0's auc: 0.652664
[210]	valid_0's auc: 0.653097
[220]	valid_0's auc: 0.653616
[230]	valid_0's auc: 0.653803
[240]	valid_0's auc: 0.653961
[250]	valid_0's auc: 0.654147
[260]	valid_0's auc: 0.654337
[270]	valid_0's auc: 0.654489
[280]	valid_0's auc: 0.654705
[290]	valid_0's auc: 0.654982
[300]	valid_0's auc: 0.655265
[310]	valid_0's auc: 0.655647
[320]	valid_0's auc: 0.655894
[330]	valid_0's auc: 0.656118
[340]	valid_0's auc: 0.656286
[350]	valid_0's auc: 0.656393
[360]	valid_0's auc: 0.656573
[370]	valid_0's auc: 0.656794
[380]	valid_0's auc: 0.656964
[390]	valid_0's auc: 0.657173
[400]	valid_0's auc: 0.657319
[410]	valid_0's auc: 0.657501
[420]	valid_0's auc: 0.657688
[430]	valid_0's auc: 0.65788
[440]	valid_0's auc: 0.658047
[450]	valid_0's auc: 0.658233
[460]	valid_0's auc: 0.65845
[470]	valid_0's auc: 0.658605
[480]	valid_0's auc: 0.658762
[490]	valid_0's auc: 0.6589
[500]	valid_0's auc: 0.659063
[510]	valid_0's auc: 0.659261
[520]	valid_0's auc: 0.659382
[530]	valid_0's auc: 0.659497
[540]	valid_0's auc: 0.659586
[550]	valid_0's auc: 0.659727
[560]	valid_0's auc: 0.659833
[570]	valid_0's auc: 0.659942
[580]	valid_0's auc: 0.660079
[590]	valid_0's auc: 0.660267
[600]	valid_0's auc: 0.660428
[610]	valid_0's auc: 0.660555
[620]	valid_0's auc: 0.660657
[630]	valid_0's auc: 0.660753
[640]	valid_0's auc: 0.660873
[650]	valid_0's auc: 0.660977
[660]	valid_0's auc: 0.661086
[670]	valid_0's auc: 0.6612
[680]	valid_0's auc: 0.661349
[690]	valid_0's auc: 0.661444
[700]	valid_0's auc: 0.661607
[710]	valid_0's auc: 0.6617
[720]	valid_0's auc: 0.661827
[730]	valid_0's auc: 0.661909
[740]	valid_0's auc: 0.662028
[750]	valid_0's auc: 0.662124
[760]	valid_0's auc: 0.662185
[770]	valid_0's auc: 0.662344
[780]	valid_0's auc: 0.662409
[790]	valid_0's auc: 0.662502
[800]	valid_0's auc: 0.662572
[810]	valid_0's auc: 0.662658
[820]	valid_0's auc: 0.662774
[830]	valid_0's auc: 0.662841
[840]	valid_0's auc: 0.662936
[850]	valid_0's auc: 0.663049
[860]	valid_0's auc: 0.663103
[870]	valid_0's auc: 0.6632
[880]	valid_0's auc: 0.663272
[890]	valid_0's auc: 0.663341
[900]	valid_0's auc: 0.663484
[910]	valid_0's auc: 0.663622
[920]	valid_0's auc: 0.663715
[930]	valid_0's auc: 0.663798
[940]	valid_0's auc: 0.663868
[950]	valid_0's auc: 0.663997
[960]	valid_0's auc: 0.664076
[970]	valid_0's auc: 0.664177
[980]	valid_0's auc: 0.664254
[990]	valid_0's auc: 0.664323
[1000]	valid_0's auc: 0.664373
[1010]	valid_0's auc: 0.664476
[1020]	valid_0's auc: 0.664577
[1030]	valid_0's auc: 0.66467
[1040]	valid_0's auc: 0.664776
[1050]	valid_0's auc: 0.664859
[1060]	valid_0's auc: 0.664923
[1070]	valid_0's auc: 0.66503
[1080]	valid_0's auc: 0.665106
[1090]	valid_0's auc: 0.665178
[1100]	valid_0's auc: 0.665265
[1110]	valid_0's auc: 0.665315
[1120]	valid_0's auc: 0.665387
[1130]	valid_0's auc: 0.66543
[1140]	valid_0's auc: 0.665478
[1150]	valid_0's auc: 0.665522
[1160]	valid_0's auc: 0.665586
[1170]	valid_0's auc: 0.665637
[1180]	valid_0's auc: 0.665676
[1190]	valid_0's auc: 0.665722
[1200]	valid_0's auc: 0.665752
[1210]	valid_0's auc: 0.66581
[1220]	valid_0's auc: 0.665859
[1230]	valid_0's auc: 0.665906
[1240]	valid_0's auc: 0.665947
[1250]	valid_0's auc: 0.665969
[1260]	valid_0's auc: 0.666022
[1270]	valid_0's auc: 0.666052
[1280]	valid_0's auc: 0.666104
[1290]	valid_0's auc: 0.666157
[1300]	valid_0's auc: 0.666205
[1310]	valid_0's auc: 0.666234
[1320]	valid_0's auc: 0.666264
[1330]	valid_0's auc: 0.666312
[1340]	valid_0's auc: 0.666378
[1350]	valid_0's auc: 0.666413
[1360]	valid_0's auc: 0.666468
[1370]	valid_0's auc: 0.666496
[1380]	valid_0's auc: 0.666541
[1390]	valid_0's auc: 0.666568
[1400]	valid_0's auc: 0.666599
[1410]	valid_0's auc: 0.666636
[1420]	valid_0's auc: 0.666669
[1430]	valid_0's auc: 0.666693
[1440]	valid_0's auc: 0.66672
[1450]	valid_0's auc: 0.666751
[1460]	valid_0's auc: 0.666783
[1470]	valid_0's auc: 0.666813
[1480]	valid_0's auc: 0.666839
[1490]	valid_0's auc: 0.666882
[1500]	valid_0's auc: 0.666928
[1510]	valid_0's auc: 0.666968
[1520]	valid_0's auc: 0.667028
[1530]	valid_0's auc: 0.667063
[1540]	valid_0's auc: 0.667115
[1550]	valid_0's auc: 0.66714
[1560]	valid_0's auc: 0.667161
[1570]	valid_0's auc: 0.667199
[1580]	valid_0's auc: 0.667224
[1590]	valid_0's auc: 0.66727
[1600]	valid_0's auc: 0.667303
[1610]	valid_0's auc: 0.667348
[1620]	valid_0's auc: 0.66739
[1630]	valid_0's auc: 0.667422
[1640]	valid_0's auc: 0.667457
[1650]	valid_0's auc: 0.667493
[1660]	valid_0's auc: 0.667547
[1670]	valid_0's auc: 0.667577
[1680]	valid_0's auc: 0.667605
[1690]	valid_0's auc: 0.667645
[1700]	valid_0's auc: 0.667671
[1710]	valid_0's auc: 0.667726
[1720]	valid_0's auc: 0.667758
[1730]	valid_0's auc: 0.667797
[1740]	valid_0's auc: 0.667817
[1750]	valid_0's auc: 0.667842
[1760]	valid_0's auc: 0.66786
[1770]	valid_0's auc: 0.667891
[1780]	valid_0's auc: 0.667923
[1790]	valid_0's auc: 0.667973
[1800]	valid_0's auc: 0.667993
[1810]	valid_0's auc: 0.668011
[1820]	valid_0's auc: 0.668042
[1830]	valid_0's auc: 0.668075
[1840]	valid_0's auc: 0.668097
[1850]	valid_0's auc: 0.668129
[1860]	valid_0's auc: 0.66816
[1870]	valid_0's auc: 0.668183
[1880]	valid_0's auc: 0.668204
[1890]	valid_0's auc: 0.668229
[1900]	valid_0's auc: 0.668243
[1910]	valid_0's auc: 0.668281
[1920]	valid_0's auc: 0.668302
[1930]	valid_0's auc: 0.668328
[1940]	valid_0's auc: 0.668361
[1950]	valid_0's auc: 0.668386
[1960]	valid_0's auc: 0.668396
[1970]	valid_0's auc: 0.668416
[1980]	valid_0's auc: 0.668441
[1990]	valid_0's auc: 0.668461
[2000]	valid_0's auc: 0.668481
[2010]	valid_0's auc: 0.668494
[2020]	valid_0's auc: 0.668499
[2030]	valid_0's auc: 0.668513
[2040]	valid_0's auc: 0.668527
[2050]	valid_0's auc: 0.668539
[2060]	valid_0's auc: 0.668529
[2070]	valid_0's auc: 0.668558
[2080]	valid_0's auc: 0.668578
[2090]	valid_0's auc: 0.668605
[2100]	valid_0's auc: 0.668625
[2110]	valid_0's auc: 0.668637
[2120]	valid_0's auc: 0.668655
[2130]	valid_0's auc: 0.668663
[2140]	valid_0's auc: 0.668681
[2150]	valid_0's auc: 0.668689
[2160]	valid_0's auc: 0.668711
[2170]	valid_0's auc: 0.668738
[2180]	valid_0's auc: 0.668757
[2190]	valid_0's auc: 0.668791
[2200]	valid_0's auc: 0.668817
[2210]	valid_0's auc: 0.668832
[2220]	valid_0's auc: 0.668839
[2230]	valid_0's auc: 0.668832
[2240]	valid_0's auc: 0.668872
[2250]	valid_0's auc: 0.668883
[2260]	valid_0's auc: 0.668882
[2270]	valid_0's auc: 0.668891
[2280]	valid_0's auc: 0.66892
[2290]	valid_0's auc: 0.668941
[2300]	valid_0's auc: 0.668953
[2310]	valid_0's auc: 0.668988
[2320]	valid_0's auc: 0.668998
[2330]	valid_0's auc: 0.669041
[2340]	valid_0's auc: 0.669055
[2350]	valid_0's auc: 0.669061
[2360]	valid_0's auc: 0.669082
[2370]	valid_0's auc: 0.669133
[2380]	valid_0's auc: 0.669146
[2390]	valid_0's auc: 0.669163
[2400]	valid_0's auc: 0.669173
[2410]	valid_0's auc: 0.669186
[2420]	valid_0's auc: 0.669206
[2430]	valid_0's auc: 0.669218
[2440]	valid_0's auc: 0.669236
[2450]	valid_0's auc: 0.66924
[2460]	valid_0's auc: 0.669273
[2470]	valid_0's auc: 0.669275
[2480]	valid_0's auc: 0.669281
[2490]	valid_0's auc: 0.669321
[2500]	valid_0's auc: 0.669368
[2510]	valid_0's auc: 0.669393
[2520]	valid_0's auc: 0.669431
[2530]	valid_0's auc: 0.669428
[2540]	valid_0's auc: 0.669435
[2550]	valid_0's auc: 0.66945
[2560]	valid_0's auc: 0.66947
[2570]	valid_0's auc: 0.669494
[2580]	valid_0's auc: 0.669543
[2590]	valid_0's auc: 0.669572
[2600]	valid_0's auc: 0.669608
[2610]	valid_0's auc: 0.669648
[2620]	valid_0's auc: 0.669678
[2630]	valid_0's auc: 0.669707
[2640]	valid_0's auc: 0.669727
[2650]	valid_0's auc: 0.66977
[2660]	valid_0's auc: 0.669817
[2670]	valid_0's auc: 0.669845
[2680]	valid_0's auc: 0.669866
[2690]	valid_0's auc: 0.669901
[2700]	valid_0's auc: 0.669963
[2710]	valid_0's auc: 0.669965
[2720]	valid_0's auc: 0.669991
[2730]	valid_0's auc: 0.670002
[2740]	valid_0's auc: 0.670006
[2750]	valid_0's auc: 0.670016
[2760]	valid_0's auc: 0.670016
[2770]	valid_0's auc: 0.670023
[2780]	valid_0's auc: 0.670028
[2790]	valid_0's auc: 0.670035
[2800]	valid_0's auc: 0.670032
[2810]	valid_0's auc: 0.670038
[2820]	valid_0's auc: 0.670045
[2830]	valid_0's auc: 0.670058
[2840]	valid_0's auc: 0.670077
[2850]	valid_0's auc: 0.670079
[2860]	valid_0's auc: 0.670085
[2870]	valid_0's auc: 0.670092
[2880]	valid_0's auc: 0.670104
[2890]	valid_0's auc: 0.670107
[2900]	valid_0's auc: 0.670108
[2910]	valid_0's auc: 0.670128
[2920]	valid_0's auc: 0.670141
[2930]	valid_0's auc: 0.670149
[2940]	valid_0's auc: 0.67016
[2950]	valid_0's auc: 0.670171
[2960]	valid_0's auc: 0.670188
[2970]	valid_0's auc: 0.670193
[2980]	valid_0's auc: 0.670216
[2990]	valid_0's auc: 0.670236
[3000]	valid_0's auc: 0.670257
[3010]	valid_0's auc: 0.670266
[3020]	valid_0's auc: 0.670271
[3030]	valid_0's auc: 0.670281
[3040]	valid_0's auc: 0.670286
[3050]	valid_0's auc: 0.670291
[3060]	valid_0's auc: 0.670301
[3070]	valid_0's auc: 0.670309
[3080]	valid_0's auc: 0.670333
[3090]	valid_0's auc: 0.670349
[3100]	valid_0's auc: 0.670356
[3110]	valid_0's auc: 0.670369
[3120]	valid_0's auc: 0.670381
[3130]	valid_0's auc: 0.670406
[3140]	valid_0's auc: 0.670417
[3150]	valid_0's auc: 0.670434
[3160]	valid_0's auc: 0.67044
[3170]	valid_0's auc: 0.670449
[3180]	valid_0's auc: 0.67046
[3190]	valid_0's auc: 0.670483
[3200]	valid_0's auc: 0.670488
[3210]	valid_0's auc: 0.670491
[3220]	valid_0's auc: 0.670498
[3230]	valid_0's auc: 0.670512
[3240]	valid_0's auc: 0.670525
[3250]	valid_0's auc: 0.670523
[3260]	valid_0's auc: 0.670525
[3270]	valid_0's auc: 0.67054
[3280]	valid_0's auc: 0.670544
[3290]	valid_0's auc: 0.670553
[3300]	valid_0's auc: 0.670567
[3310]	valid_0's auc: 0.670585
[3320]	valid_0's auc: 0.670594
[3330]	valid_0's auc: 0.670607
[3340]	valid_0's auc: 0.670628
[3350]	valid_0's auc: 0.670634
[3360]	valid_0's auc: 0.670644
[3370]	valid_0's auc: 0.67065
[3380]	valid_0's auc: 0.670662
[3390]	valid_0's auc: 0.670674
[3400]	valid_0's auc: 0.670677
[3410]	valid_0's auc: 0.670688
[3420]	valid_0's auc: 0.670692
[3430]	valid_0's auc: 0.670694
[3440]	valid_0's auc: 0.670704
[3450]	valid_0's auc: 0.670711
[3460]	valid_0's auc: 0.670721
[3470]	valid_0's auc: 0.670738
[3480]	valid_0's auc: 0.670741
[3490]	valid_0's auc: 0.670726
[3500]	valid_0's auc: 0.670744
[3510]	valid_0's auc: 0.670756
[3520]	valid_0's auc: 0.670761
[3530]	valid_0's auc: 0.670767
[3540]	valid_0's auc: 0.670772
[3550]	valid_0's auc: 0.670781
[3560]	valid_0's auc: 0.670784
[3570]	valid_0's auc: 0.670796
[3580]	valid_0's auc: 0.670803
[3590]	valid_0's auc: 0.670821
[3600]	valid_0's auc: 0.670823
[3610]	valid_0's auc: 0.67083
[3620]	valid_0's auc: 0.670832
[3630]	valid_0's auc: 0.670837
[3640]	valid_0's auc: 0.670843
[3650]	valid_0's auc: 0.670847
[3660]	valid_0's auc: 0.670858
[3670]	valid_0's auc: 0.670866
[3680]	valid_0's auc: 0.670871
[3690]	valid_0's auc: 0.670893
[3700]	valid_0's auc: 0.670898
[3710]	valid_0's auc: 0.670907
[3720]	valid_0's auc: 0.670912
[3730]	valid_0's auc: 0.670917
[3740]	valid_0's auc: 0.670922
[3750]	valid_0's auc: 0.67093
[3760]	valid_0's auc: 0.670937
[3770]	valid_0's auc: 0.670944
[3780]	valid_0's auc: 0.670951
[3790]	valid_0's auc: 0.670957
[3800]	valid_0's auc: 0.670993
[3810]	valid_0's auc: 0.671004
[3820]	valid_0's auc: 0.671021
[3830]	valid_0's auc: 0.671022
[3840]	valid_0's auc: 0.671051
[3850]	valid_0's auc: 0.671075
[3860]	valid_0's auc: 0.671088
[3870]	valid_0's auc: 0.671085
[3880]	valid_0's auc: 0.671088
[3890]	valid_0's auc: 0.6711
[3900]	valid_0's auc: 0.671099
[3910]	valid_0's auc: 0.671097
[3920]	valid_0's auc: 0.671098
[3930]	valid_0's auc: 0.671102
[3940]	valid_0's auc: 0.671105
[3950]	valid_0's auc: 0.671112
[3960]	valid_0's auc: 0.671116
[3970]	valid_0's auc: 0.671119
[3980]	valid_0's auc: 0.671126
[3990]	valid_0's auc: 0.67112
[4000]	valid_0's auc: 0.671128
[4010]	valid_0's auc: 0.671143
[4020]	valid_0's auc: 0.671154
[4030]	valid_0's auc: 0.671163
[4040]	valid_0's auc: 0.671167
[4050]	valid_0's auc: 0.671169
[4060]	valid_0's auc: 0.671178
[4070]	valid_0's auc: 0.671184
[4080]	valid_0's auc: 0.671191
[4090]	valid_0's auc: 0.671198
[4100]	valid_0's auc: 0.671208
[4110]	valid_0's auc: 0.67122
[4120]	valid_0's auc: 0.671225
[4130]	valid_0's auc: 0.671229
[4140]	valid_0's auc: 0.67123
[4150]	valid_0's auc: 0.671231
[4160]	valid_0's auc: 0.671231
[4170]	valid_0's auc: 0.671234
[4180]	valid_0's auc: 0.671233
[4190]	valid_0's auc: 0.671235
[4200]	valid_0's auc: 0.671234
[4210]	valid_0's auc: 0.67123
[4220]	valid_0's auc: 0.671243
[4230]	valid_0's auc: 0.671248
[4240]	valid_0's auc: 0.671264
[4250]	valid_0's auc: 0.67126
[4260]	valid_0's auc: 0.671258
[4270]	valid_0's auc: 0.671262
[4280]	valid_0's auc: 0.671276
[4290]	valid_0's auc: 0.671292
[4300]	valid_0's auc: 0.671297
[4310]	valid_0's auc: 0.671297
[4320]	valid_0's auc: 0.671304
[4330]	valid_0's auc: 0.671309
[4340]	valid_0's auc: 0.67131
[4350]	valid_0's auc: 0.671321
[4360]	valid_0's auc: 0.671317
[4370]	valid_0's auc: 0.671321
[4380]	valid_0's auc: 0.671319
[4390]	valid_0's auc: 0.671337
[4400]	valid_0's auc: 0.671352
[4410]	valid_0's auc: 0.671356
[4420]	valid_0's auc: 0.67136
[4430]	valid_0's auc: 0.671361
[4440]	valid_0's auc: 0.671361
[4450]	valid_0's auc: 0.671367
[4460]	valid_0's auc: 0.671371
[4470]	valid_0's auc: 0.671371
[4480]	valid_0's auc: 0.671375
[4490]	valid_0's auc: 0.671388
[4500]	valid_0's auc: 0.671393
[4510]	valid_0's auc: 0.671394
[4520]	valid_0's auc: 0.671404
[4530]	valid_0's auc: 0.671402
[4540]	valid_0's auc: 0.671414
[4550]	valid_0's auc: 0.671426
[4560]	valid_0's auc: 0.671435
[4570]	valid_0's auc: 0.671452
[4580]	valid_0's auc: 0.671451
[4590]	valid_0's auc: 0.67145
[4600]	valid_0's auc: 0.67145
[4610]	valid_0's auc: 0.671451
[4620]	valid_0's auc: 0.671452
[4630]	valid_0's auc: 0.67146
[4640]	valid_0's auc: 0.671466
[4650]	valid_0's auc: 0.671467
[4660]	valid_0's auc: 0.671466
[4670]	valid_0's auc: 0.671471
[4680]	valid_0's auc: 0.671472
[4690]	valid_0's auc: 0.671479
[4700]	valid_0's auc: 0.671488
[4710]	valid_0's auc: 0.671491
[4720]	valid_0's auc: 0.671509
[4730]	valid_0's auc: 0.671513
[4740]	valid_0's auc: 0.671516
[4750]	valid_0's auc: 0.671518
[4760]	valid_0's auc: 0.671525
[4770]	valid_0's auc: 0.671527
[4780]	valid_0's auc: 0.671533
[4790]	valid_0's auc: 0.671536
[4800]	valid_0's auc: 0.671546
[4810]	valid_0's auc: 0.671554
[4820]	valid_0's auc: 0.671557
[4830]	valid_0's auc: 0.671555
[4840]	valid_0's auc: 0.671563
[4850]	valid_0's auc: 0.671567
[4860]	valid_0's auc: 0.671569
[4870]	valid_0's auc: 0.671571
[4880]	valid_0's auc: 0.671571
[4890]	valid_0's auc: 0.671571
[4900]	valid_0's auc: 0.671579
[4910]	valid_0's auc: 0.671581
[4920]	valid_0's auc: 0.671579
[4930]	valid_0's auc: 0.671579
[4940]	valid_0's auc: 0.67158
[4950]	valid_0's auc: 0.671578
Early stopping, best iteration is:
[4907]	valid_0's auc: 0.671583
0.671583065364
4907

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 25
max_depth            : 30
learning_rate        : 0.1
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.627845
[20]	valid_0's auc: 0.633327
[30]	valid_0's auc: 0.637982
[40]	valid_0's auc: 0.64202
[50]	valid_0's auc: 0.644388
[60]	valid_0's auc: 0.646973
[70]	valid_0's auc: 0.648682
[80]	valid_0's auc: 0.649996
[90]	valid_0's auc: 0.651046
[100]	valid_0's auc: 0.652212
[110]	valid_0's auc: 0.653269
[120]	valid_0's auc: 0.654119
[130]	valid_0's auc: 0.654784
[140]	valid_0's auc: 0.655509
[150]	valid_0's auc: 0.656128
[160]	valid_0's auc: 0.656704
[170]	valid_0's auc: 0.657202
[180]	valid_0's auc: 0.657758
[190]	valid_0's auc: 0.658055
[200]	valid_0's auc: 0.658462
[210]	valid_0's auc: 0.658704
[220]	valid_0's auc: 0.659003
[230]	valid_0's auc: 0.659319
[240]	valid_0's auc: 0.659563
[250]	valid_0's auc: 0.659861
[260]	valid_0's auc: 0.66007
[270]	valid_0's auc: 0.66035
[280]	valid_0's auc: 0.660465
[290]	valid_0's auc: 0.660778
[300]	valid_0's auc: 0.66096
[310]	valid_0's auc: 0.661044
[320]	valid_0's auc: 0.661202
[330]	valid_0's auc: 0.661251
[340]	valid_0's auc: 0.661428
[350]	valid_0's auc: 0.661548
[360]	valid_0's auc: 0.661682
[370]	valid_0's auc: 0.661812
[380]	valid_0's auc: 0.661864
[390]	valid_0's auc: 0.661989
[400]	valid_0's auc: 0.662058
[410]	valid_0's auc: 0.662184
[420]	valid_0's auc: 0.662304
[430]	valid_0's auc: 0.662335
[440]	valid_0's auc: 0.662513
[450]	valid_0's auc: 0.662707
[460]	valid_0's auc: 0.662913
[470]	valid_0's auc: 0.663017
[480]	valid_0's auc: 0.663127
[490]	valid_0's auc: 0.66323
[500]	valid_0's auc: 0.663305
[510]	valid_0's auc: 0.663422
[520]	valid_0's auc: 0.663569
[530]	valid_0's auc: 0.663587
[540]	valid_0's auc: 0.663651
[550]	valid_0's auc: 0.663708
[560]	valid_0's auc: 0.663797
[570]	valid_0's auc: 0.663828
[580]	valid_0's auc: 0.663909
[590]	valid_0's auc: 0.663974
[600]	valid_0's auc: 0.664034
[610]	valid_0's auc: 0.664051
[620]	valid_0's auc: 0.66417
[630]	valid_0's auc: 0.664218
[640]	valid_0's auc: 0.664295
[650]	valid_0's auc: 0.664335
[660]	valid_0's auc: 0.664431
[670]	valid_0's auc: 0.664442
[680]	valid_0's auc: 0.664471
[690]	valid_0's auc: 0.664478
[700]	valid_0's auc: 0.664501
[710]	valid_0's auc: 0.664535
[720]	valid_0's auc: 0.664543
[730]	valid_0's auc: 0.664581
[740]	valid_0's auc: 0.66463
[750]	valid_0's auc: 0.664672
[760]	valid_0's auc: 0.664699
[770]	valid_0's auc: 0.664731
[780]	valid_0's auc: 0.664849
[790]	valid_0's auc: 0.664937
[800]	valid_0's auc: 0.665007
[810]	valid_0's auc: 0.665066
[820]	valid_0's auc: 0.665133
[830]	valid_0's auc: 0.66515
[840]	valid_0's auc: 0.665163
[850]	valid_0's auc: 0.665176
[860]	valid_0's auc: 0.665191
[870]	valid_0's auc: 0.665252
[880]	valid_0's auc: 0.665265
[890]	valid_0's auc: 0.665365
[900]	valid_0's auc: 0.665393
[910]	valid_0's auc: 0.665418
[920]	valid_0's auc: 0.665452
[930]	valid_0's auc: 0.665561
[940]	valid_0's auc: 0.665656
[950]	valid_0's auc: 0.665701
[960]	valid_0's auc: 0.665735
[970]	valid_0's auc: 0.66574
[980]	valid_0's auc: 0.665783
[990]	valid_0's auc: 0.665836
[1000]	valid_0's auc: 0.665849
[1010]	valid_0's auc: 0.665873
[1020]	valid_0's auc: 0.66589
[1030]	valid_0's auc: 0.665913
[1040]	valid_0's auc: 0.665935
[1050]	valid_0's auc: 0.665994
[1060]	valid_0's auc: 0.666016
[1070]	valid_0's auc: 0.666033
[1080]	valid_0's auc: 0.666078
[1090]	valid_0's auc: 0.666117
[1100]	valid_0's auc: 0.666151
[1110]	valid_0's auc: 0.666169
[1120]	valid_0's auc: 0.666186
[1130]	valid_0's auc: 0.666182
[1140]	valid_0's auc: 0.666199
[1150]	valid_0's auc: 0.666208
[1160]	valid_0's auc: 0.66622
[1170]	valid_0's auc: 0.666257
[1180]	valid_0's auc: 0.666278
[1190]	valid_0's auc: 0.666302
[1200]	valid_0's auc: 0.666296
[1210]	valid_0's auc: 0.666329
[1220]	valid_0's auc: 0.666376
[1230]	valid_0's auc: 0.666368
[1240]	valid_0's auc: 0.666387
[1250]	valid_0's auc: 0.666443
[1260]	valid_0's auc: 0.666449
[1270]	valid_0's auc: 0.666451
[1280]	valid_0's auc: 0.666471
[1290]	valid_0's auc: 0.666503
[1300]	valid_0's auc: 0.666466
[1310]	valid_0's auc: 0.666488
[1320]	valid_0's auc: 0.666502
[1330]	valid_0's auc: 0.666572
[1340]	valid_0's auc: 0.666595
[1350]	valid_0's auc: 0.666621
[1360]	valid_0's auc: 0.666619
[1370]	valid_0's auc: 0.66665
[1380]	valid_0's auc: 0.666682
[1390]	valid_0's auc: 0.666721
[1400]	valid_0's auc: 0.666715
[1410]	valid_0's auc: 0.666715
[1420]	valid_0's auc: 0.666741
[1430]	valid_0's auc: 0.666759
[1440]	valid_0's auc: 0.666776
[1450]	valid_0's auc: 0.6668
[1460]	valid_0's auc: 0.666799
[1470]	valid_0's auc: 0.666815
[1480]	valid_0's auc: 0.666811
[1490]	valid_0's auc: 0.666818
[1500]	valid_0's auc: 0.666819
[1510]	valid_0's auc: 0.666828
[1520]	valid_0's auc: 0.666827
[1530]	valid_0's auc: 0.666871
[1540]	valid_0's auc: 0.666886
[1550]	valid_0's auc: 0.666876
[1560]	valid_0's auc: 0.666906
[1570]	valid_0's auc: 0.666944
[1580]	valid_0's auc: 0.666969
[1590]	valid_0's auc: 0.666977
[1600]	valid_0's auc: 0.666977
[1610]	valid_0's auc: 0.666985
[1620]	valid_0's auc: 0.667028
[1630]	valid_0's auc: 0.667042
[1640]	valid_0's auc: 0.667062
[1650]	valid_0's auc: 0.667084
[1660]	valid_0's auc: 0.667091
[1670]	valid_0's auc: 0.667124
[1680]	valid_0's auc: 0.667148
[1690]	valid_0's auc: 0.667163
[1700]	valid_0's auc: 0.667206
[1710]	valid_0's auc: 0.667219
[1720]	valid_0's auc: 0.667233
[1730]	valid_0's auc: 0.667239
[1740]	valid_0's auc: 0.667255
[1750]	valid_0's auc: 0.667279
[1760]	valid_0's auc: 0.667306
[1770]	valid_0's auc: 0.667324
[1780]	valid_0's auc: 0.667356
[1790]	valid_0's auc: 0.667365
[1800]	valid_0's auc: 0.667366
[1810]	valid_0's auc: 0.667398
[1820]	valid_0's auc: 0.667456
[1830]	valid_0's auc: 0.667477
[1840]	valid_0's auc: 0.667483
[1850]	valid_0's auc: 0.667492
[1860]	valid_0's auc: 0.667514
[1870]	valid_0's auc: 0.667521
[1880]	valid_0's auc: 0.667541
[1890]	valid_0's auc: 0.667561
[1900]	valid_0's auc: 0.667607
[1910]	valid_0's auc: 0.66761
[1920]	valid_0's auc: 0.667607
[1930]	valid_0's auc: 0.667624
[1940]	valid_0's auc: 0.667622
[1950]	valid_0's auc: 0.667616
[1960]	valid_0's auc: 0.667629
[1970]	valid_0's auc: 0.667697
[1980]	valid_0's auc: 0.667748
[1990]	valid_0's auc: 0.667747
[2000]	valid_0's auc: 0.667747
[2010]	valid_0's auc: 0.667817
[2020]	valid_0's auc: 0.667813
[2030]	valid_0's auc: 0.667832
[2040]	valid_0's auc: 0.667836
[2050]	valid_0's auc: 0.667846
[2060]	valid_0's auc: 0.667853
[2070]	valid_0's auc: 0.667861
[2080]	valid_0's auc: 0.667878
[2090]	valid_0's auc: 0.667886
[2100]	valid_0's auc: 0.667861
[2110]	valid_0's auc: 0.667864
[2120]	valid_0's auc: 0.66786
[2130]	valid_0's auc: 0.667884
Early stopping, best iteration is:
[2087]	valid_0's auc: 0.667889
0.667889255572
2087

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 300
max_depth            : -1
learning_rate        : 0.02
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650767
[20]	valid_0's auc: 0.652648
[30]	valid_0's auc: 0.654441
[40]	valid_0's auc: 0.656059
[50]	valid_0's auc: 0.657655
[60]	valid_0's auc: 0.658689
[70]	valid_0's auc: 0.659781
[80]	valid_0's auc: 0.660699
[90]	valid_0's auc: 0.661738
[100]	valid_0's auc: 0.662831
[110]	valid_0's auc: 0.663737
[120]	valid_0's auc: 0.664713
[130]	valid_0's auc: 0.665698
[140]	valid_0's auc: 0.66666
[150]	valid_0's auc: 0.667407
[160]	valid_0's auc: 0.668037
[170]	valid_0's auc: 0.668496
[180]	valid_0's auc: 0.668851
[190]	valid_0's auc: 0.66924
[200]	valid_0's auc: 0.669509
[210]	valid_0's auc: 0.669973
[220]	valid_0's auc: 0.670434
[230]	valid_0's auc: 0.670839
[240]	valid_0's auc: 0.671163
[250]	valid_0's auc: 0.671459
[260]	valid_0's auc: 0.67168
[270]	valid_0's auc: 0.671841
[280]	valid_0's auc: 0.672025
[290]	valid_0's auc: 0.672288
[300]	valid_0's auc: 0.672435
[310]	valid_0's auc: 0.672678
[320]	valid_0's auc: 0.672828
[330]	valid_0's auc: 0.672946
[340]	valid_0's auc: 0.67309
[350]	valid_0's auc: 0.673209
[360]	valid_0's auc: 0.673273
[370]	valid_0's auc: 0.673373
[380]	valid_0's auc: 0.673452
[390]	valid_0's auc: 0.673534
[400]	valid_0's auc: 0.673588
[410]	valid_0's auc: 0.673663
[420]	valid_0's auc: 0.67373
[430]	valid_0's auc: 0.673793
[440]	valid_0's auc: 0.673858
[450]	valid_0's auc: 0.673891
[460]	valid_0's auc: 0.673941
[470]	valid_0's auc: 0.673973
[480]	valid_0's auc: 0.674002
[490]	valid_0's auc: 0.674042
[500]	valid_0's auc: 0.674069
[510]	valid_0's auc: 0.674087
[520]	valid_0's auc: 0.674114
[530]	valid_0's auc: 0.674101
[540]	valid_0's auc: 0.674102
[550]	valid_0's auc: 0.674105
[560]	valid_0's auc: 0.674111
[570]	valid_0's auc: 0.674123
[580]	valid_0's auc: 0.674144
[590]	valid_0's auc: 0.674148
[600]	valid_0's auc: 0.674118
[610]	valid_0's auc: 0.674128
[620]	valid_0's auc: 0.674127
[630]	valid_0's auc: 0.674127
Early stopping, best iteration is:
[584]	valid_0's auc: 0.674155
0.674154868711
584

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

num_leaves           : 200
max_depth            : 10
learning_rate        : 0.02
boosting             : gbdt

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.621199
[20]	valid_0's auc: 0.62347
[30]	valid_0's auc: 0.6257
[40]	valid_0's auc: 0.627434
[50]	valid_0's auc: 0.630681
[60]	valid_0's auc: 0.631693
[70]	valid_0's auc: 0.633099
[80]	valid_0's auc: 0.634801
[90]	valid_0's auc: 0.635592
[100]	valid_0's auc: 0.636441
[110]	valid_0's auc: 0.637178
[120]	valid_0's auc: 0.637658
[130]	valid_0's auc: 0.638612
[140]	valid_0's auc: 0.639467
[150]	valid_0's auc: 0.640161
[160]	valid_0's auc: 0.640784
[170]	valid_0's auc: 0.641228
[180]	valid_0's auc: 0.641706
[190]	valid_0's auc: 0.642107
[200]	valid_0's auc: 0.642436
[210]	valid_0's auc: 0.642811
[220]	valid_0's auc: 0.64304
[230]	valid_0's auc: 0.64324
[240]	valid_0's auc: 0.643426
[250]	valid_0's auc: 0.643721
[260]	valid_0's auc: 0.644049
[270]	valid_0's auc: 0.644409
[280]	valid_0's auc: 0.644695
[290]	valid_0's auc: 0.644928
[300]	valid_0's auc: 0.645113
[310]	valid_0's auc: 0.645269
[320]	valid_0's auc: 0.645457
[330]	valid_0's auc: 0.645643
[340]	valid_0's auc: 0.645748
[350]	valid_0's auc: 0.645965
[360]	valid_0's auc: 0.646129
[370]	valid_0's auc: 0.646279
[380]	valid_0's auc: 0.64647
[390]	valid_0's auc: 0.646629
[400]	valid_0's auc: 0.646756
[410]	valid_0's auc: 0.646983
[420]	valid_0's auc: 0.647213
[430]	valid_0's auc: 0.647413
[440]	valid_0's auc: 0.647654
[450]	valid_0's auc: 0.647864
[460]	valid_0's auc: 0.64814
[470]	valid_0's auc: 0.648267
[480]	valid_0's auc: 0.648408
[490]	valid_0's auc: 0.648533
[500]	valid_0's auc: 0.648665
[510]	valid_0's auc: 0.648799
[520]	valid_0's auc: 0.648915
[530]	valid_0's auc: 0.649045
[540]	valid_0's auc: 0.649141
[550]	valid_0's auc: 0.649244
[560]	valid_0's auc: 0.649337
[570]	valid_0's auc: 0.649494
[580]	valid_0's auc: 0.649638
[590]	valid_0's auc: 0.649799
[600]	valid_0's auc: 0.649956
[610]	valid_0's auc: 0.650107
[620]	valid_0's auc: 0.650267
[630]	valid_0's auc: 0.650412
[640]	valid_0's auc: 0.65056
[650]	valid_0's auc: 0.65069
[660]	valid_0's auc: 0.650828
[670]	valid_0's auc: 0.650977
[680]	valid_0's auc: 0.651112
[690]	valid_0's auc: 0.651236
[700]	valid_0's auc: 0.651356
[710]	valid_0's auc: 0.651463
[720]	valid_0's auc: 0.651587
[730]	valid_0's auc: 0.651691
[740]	valid_0's auc: 0.651779
[750]	valid_0's auc: 0.651902
[760]	valid_0's auc: 0.652005
[770]	valid_0's auc: 0.652113
[780]	valid_0's auc: 0.652217
[790]	valid_0's auc: 0.652339
[800]	valid_0's auc: 0.652438
[810]	valid_0's auc: 0.652583
[820]	valid_0's auc: 0.65273
[830]	valid_0's auc: 0.652845
[840]	valid_0's auc: 0.652938
[850]	valid_0's auc: 0.653034
[860]	valid_0's auc: 0.653131
[870]	valid_0's auc: 0.653244
[880]	valid_0's auc: 0.653316
[890]	valid_0's auc: 0.6534
[900]	valid_0's auc: 0.653486
[910]	valid_0's auc: 0.653573
[920]	valid_0's auc: 0.653661
[930]	valid_0's auc: 0.653722
[940]	valid_0's auc: 0.653809
[950]	valid_0's auc: 0.653888
[960]	valid_0's auc: 0.65402
[970]	valid_0's auc: 0.654087
[980]	valid_0's auc: 0.654152
[990]	valid_0's auc: 0.654266
[1000]	valid_0's auc: 0.654339
[1010]	valid_0's auc: 0.654449
[1020]	valid_0's auc: 0.654513
[1030]	valid_0's auc: 0.654611
[1040]	valid_0's auc: 0.654745
[1050]	valid_0's auc: 0.654808
[1060]	valid_0's auc: 0.654859
[1070]	valid_0's auc: 0.654951
[1080]	valid_0's auc: 0.655039
[1090]	valid_0's auc: 0.655156
[1100]	valid_0's auc: 0.655251
[1110]	valid_0's auc: 0.655317
[1120]	valid_0's auc: 0.655389
[1130]	valid_0's auc: 0.655471
[1140]	valid_0's auc: 0.655599
[1150]	valid_0's auc: 0.655669
[1160]	valid_0's auc: 0.655746
[1170]	valid_0's auc: 0.655819
[1180]	valid_0's auc: 0.655922
[1190]	valid_0's auc: 0.656079
[1200]	valid_0's auc: 0.656214
[1210]	valid_0's auc: 0.65634
[1220]	valid_0's auc: 0.656459
[1230]	valid_0's auc: 0.656545
[1240]	valid_0's auc: 0.656638
[1250]	valid_0's auc: 0.656689
[1260]	valid_0's auc: 0.656767
[1270]	valid_0's auc: 0.656828
[1280]	valid_0's auc: 0.656914
[1290]	valid_0's auc: 0.65696
[1300]	valid_0's auc: 0.657025
[1310]	valid_0's auc: 0.657097
[1320]	valid_0's auc: 0.657116
[1330]	valid_0's auc: 0.657184
[1340]	valid_0's auc: 0.657253
[1350]	valid_0's auc: 0.657323
[1360]	valid_0's auc: 0.657387
[1370]	valid_0's auc: 0.657419
[1380]	valid_0's auc: 0.657465
[1390]	valid_0's auc: 0.657528
[1400]	valid_0's auc: 0.657604
[1410]	valid_0's auc: 0.657662
[1420]	valid_0's auc: 0.657734
[1430]	valid_0's auc: 0.657798
[1440]	valid_0's auc: 0.657866
[1450]	valid_0's auc: 0.65793
[1460]	valid_0's auc: 0.657989
[1470]	valid_0's auc: 0.658087
[1480]	valid_0's auc: 0.658174
[1490]	valid_0's auc: 0.658222
[1500]	valid_0's auc: 0.65826
[1510]	valid_0's auc: 0.658308
[1520]	valid_0's auc: 0.658363
[1530]	valid_0's auc: 0.658411
[1540]	valid_0's auc: 0.658465
[1550]	valid_0's auc: 0.658518
[1560]	valid_0's auc: 0.658577
[1570]	valid_0's auc: 0.658641
[1580]	valid_0's auc: 0.65869
[1590]	valid_0's auc: 0.658752
[1600]	valid_0's auc: 0.658828
[1610]	valid_0's auc: 0.658889
[1620]	valid_0's auc: 0.658937
[1630]	valid_0's auc: 0.658987
[1640]	valid_0's auc: 0.659026
[1650]	valid_0's auc: 0.659117
[1660]	valid_0's auc: 0.659174
[1670]	valid_0's auc: 0.659256
[1680]	valid_0's auc: 0.659367
[1690]	valid_0's auc: 0.659448
[1700]	valid_0's auc: 0.659514
[1710]	valid_0's auc: 0.659584
[1720]	valid_0's auc: 0.659651
[1730]	valid_0's auc: 0.659704
[1740]	valid_0's auc: 0.659764
[1750]	valid_0's auc: 0.659842
[1760]	valid_0's auc: 0.659912
[1770]	valid_0's auc: 0.65997
[1780]	valid_0's auc: 0.660031
[1790]	valid_0's auc: 0.660104
[1800]	valid_0's auc: 0.660208
[1810]	valid_0's auc: 0.660249
[1820]	valid_0's auc: 0.660297
[1830]	valid_0's auc: 0.660319
[1840]	valid_0's auc: 0.660347
[1850]	valid_0's auc: 0.660385
[1860]	valid_0's auc: 0.660428
[1870]	valid_0's auc: 0.660492
[1880]	valid_0's auc: 0.660527
[1890]	valid_0's auc: 0.660572
[1900]	valid_0's auc: 0.660599
[1910]	valid_0's auc: 0.66063
[1920]	valid_0's auc: 0.660678
[1930]	valid_0's auc: 0.660708
[1940]	valid_0's auc: 0.660753
[1950]	valid_0's auc: 0.660799
[1960]	valid_0's auc: 0.660843
[1970]	valid_0's auc: 0.660884
[1980]	valid_0's auc: 0.660901
[1990]	valid_0's auc: 0.66093
[2000]	valid_0's auc: 0.660966
[2010]	valid_0's auc: 0.660997
[2020]	valid_0's auc: 0.661034
[2030]	valid_0's auc: 0.661083
[2040]	valid_0's auc: 0.661101
[2050]	valid_0's auc: 0.661146
[2060]	valid_0's auc: 0.661178
[2070]	valid_0's auc: 0.661209
[2080]	valid_0's auc: 0.661236
[2090]	valid_0's auc: 0.661271
[2100]	valid_0's auc: 0.661308
[2110]	valid_0's auc: 0.661348
[2120]	valid_0's auc: 0.661414
[2130]	valid_0's auc: 0.661443
[2140]	valid_0's auc: 0.661467
[2150]	valid_0's auc: 0.661508
[2160]	valid_0's auc: 0.661536
[2170]	valid_0's auc: 0.661572
[2180]	valid_0's auc: 0.661603
[2190]	valid_0's auc: 0.661651
[2200]	valid_0's auc: 0.661681
[2210]	valid_0's auc: 0.661722
[2220]	valid_0's auc: 0.661757
[2230]	valid_0's auc: 0.66181
[2240]	valid_0's auc: 0.661844
[2250]	valid_0's auc: 0.661893
[2260]	valid_0's auc: 0.661942
[2270]	valid_0's auc: 0.661974
[2280]	valid_0's auc: 0.661998
[2290]	valid_0's auc: 0.662046
[2300]	valid_0's auc: 0.66209
[2310]	valid_0's auc: 0.662129
[2320]	valid_0's auc: 0.662167
[2330]	valid_0's auc: 0.662193
[2340]	valid_0's auc: 0.662261
[2350]	valid_0's auc: 0.662312
[2360]	valid_0's auc: 0.662334
[2370]	valid_0's auc: 0.662364
[2380]	valid_0's auc: 0.662393
[2390]	valid_0's auc: 0.662433
[2400]	valid_0's auc: 0.662513
[2410]	valid_0's auc: 0.662553
[2420]	valid_0's auc: 0.662594
[2430]	valid_0's auc: 0.662621
[2440]	valid_0's auc: 0.662678
[2450]	valid_0's auc: 0.662703
[2460]	valid_0's auc: 0.662741
[2470]	valid_0's auc: 0.662821
[2480]	valid_0's auc: 0.662854
[2490]	valid_0's auc: 0.662903
[2500]	valid_0's auc: 0.662936
[2510]	valid_0's auc: 0.662968
[2520]	valid_0's auc: 0.663009
[2530]	valid_0's auc: 0.663044
[2540]	valid_0's auc: 0.663119
[2550]	valid_0's auc: 0.663139
[2560]	valid_0's auc: 0.663167
[2570]	valid_0's auc: 0.663196
[2580]	valid_0's auc: 0.663246
[2590]	valid_0's auc: 0.663259
[2600]	valid_0's auc: 0.663307
[2610]	valid_0's auc: 0.663334
[2620]	valid_0's auc: 0.663371
[2630]	valid_0's auc: 0.663398
[2640]	valid_0's auc: 0.663472
[2650]	valid_0's auc: 0.6635
[2660]	valid_0's auc: 0.663516
[2670]	valid_0's auc: 0.663562
[2680]	valid_0's auc: 0.663592
[2690]	valid_0's auc: 0.663628
[2700]	valid_0's auc: 0.663671
[2710]	valid_0's auc: 0.663699
[2720]	valid_0's auc: 0.663757
[2730]	valid_0's auc: 0.663772
[2740]	valid_0's auc: 0.663797
[2750]	valid_0's auc: 0.663837
[2760]	valid_0's auc: 0.663862
[2770]	valid_0's auc: 0.663895
[2780]	valid_0's auc: 0.663919
[2790]	valid_0's auc: 0.66394
[2800]	valid_0's auc: 0.663986
[2810]	valid_0's auc: 0.664022
[2820]	valid_0's auc: 0.664041
[2830]	valid_0's auc: 0.664095
[2840]	valid_0's auc: 0.664127
[2850]	valid_0's auc: 0.664167
[2860]	valid_0's auc: 0.664186
[2870]	valid_0's auc: 0.664215
[2880]	valid_0's auc: 0.664237
[2890]	valid_0's auc: 0.664257
[2900]	valid_0's auc: 0.664288
[2910]	valid_0's auc: 0.6643
[2920]	valid_0's auc: 0.664323
[2930]	valid_0's auc: 0.664347
[2940]	valid_0's auc: 0.664363
[2950]	valid_0's auc: 0.664405
[2960]	valid_0's auc: 0.664422
[2970]	valid_0's auc: 0.664435
[2980]	valid_0's auc: 0.664561
[2990]	valid_0's auc: 0.664585
[3000]	valid_0's auc: 0.664612
[3010]	valid_0's auc: 0.664626
[3020]	valid_0's auc: 0.664637
[3030]	valid_0's auc: 0.664675
[3040]	valid_0's auc: 0.664708
[3050]	valid_0's auc: 0.664752
[3060]	valid_0's auc: 0.664785
[3070]	valid_0's auc: 0.664821
[3080]	valid_0's auc: 0.664831
[3090]	valid_0's auc: 0.66485
[3100]	valid_0's auc: 0.664883
[3110]	valid_0's auc: 0.664901
[3120]	valid_0's auc: 0.664942
[3130]	valid_0's auc: 0.664957
[3140]	valid_0's auc: 0.664985
[3150]	valid_0's auc: 0.665015
[3160]	valid_0's auc: 0.665045
[3170]	valid_0's auc: 0.66506
[3180]	valid_0's auc: 0.665089
[3190]	valid_0's auc: 0.665114
[3200]	valid_0's auc: 0.665136
[3210]	valid_0's auc: 0.665162
[3220]	valid_0's auc: 0.665192
[3230]	valid_0's auc: 0.665211
[3240]	valid_0's auc: 0.665238
[3250]	valid_0's auc: 0.665265
[3260]	valid_0's auc: 0.665279
[3270]	valid_0's auc: 0.665299
[3280]	valid_0's auc: 0.665312
[3290]	valid_0's auc: 0.665355
[3300]	valid_0's auc: 0.665397
[3310]	valid_0's auc: 0.665416
[3320]	valid_0's auc: 0.665454
[3330]	valid_0's auc: 0.665475
[3340]	valid_0's auc: 0.665492
[3350]	valid_0's auc: 0.665515
[3360]	valid_0's auc: 0.665532
[3370]	valid_0's auc: 0.665551
[3380]	valid_0's auc: 0.665565
[3390]	valid_0's auc: 0.665575
[3400]	valid_0's auc: 0.665607
[3410]	valid_0's auc: 0.665629
[3420]	valid_0's auc: 0.665641
[3430]	valid_0's auc: 0.665652
[3440]	valid_0's auc: 0.665668
[3450]	valid_0's auc: 0.665676
[3460]	valid_0's auc: 0.665697
[3470]	valid_0's auc: 0.665703
[3480]	valid_0's auc: 0.66572
[3490]	valid_0's auc: 0.665736
[3500]	valid_0's auc: 0.66575
[3510]	valid_0's auc: 0.665765
[3520]	valid_0's auc: 0.665779
[3530]	valid_0's auc: 0.665791
[3540]	valid_0's auc: 0.665808
[3550]	valid_0's auc: 0.665821
[3560]	valid_0's auc: 0.665824
[3570]	valid_0's auc: 0.665844
[3580]	valid_0's auc: 0.665868
[3590]	valid_0's auc: 0.665877
[3600]	valid_0's auc: 0.665895
[3610]	valid_0's auc: 0.665906
[3620]	valid_0's auc: 0.66591
[3630]	valid_0's auc: 0.665927
[3640]	valid_0's auc: 0.665948
[3650]	valid_0's auc: 0.665966
[3660]	valid_0's auc: 0.665994
[3670]	valid_0's auc: 0.666011
[3680]	valid_0's auc: 0.666029
[3690]	valid_0's auc: 0.666047
[3700]	valid_0's auc: 0.666059
[3710]	valid_0's auc: 0.66607
[3720]	valid_0's auc: 0.666085
[3730]	valid_0's auc: 0.666093
[3740]	valid_0's auc: 0.666132
[3750]	valid_0's auc: 0.666149
[3760]	valid_0's auc: 0.666168
[3770]	valid_0's auc: 0.666182
[3780]	valid_0's auc: 0.666194
[3790]	valid_0's auc: 0.666212
[3800]	valid_0's auc: 0.666204
[3810]	valid_0's auc: 0.666214
[3820]	valid_0's auc: 0.666233
[3830]	valid_0's auc: 0.66627
[3840]	valid_0's auc: 0.666299
[3850]	valid_0's auc: 0.666305
[3860]	valid_0's auc: 0.666319
[3870]	valid_0's auc: 0.666332
[3880]	valid_0's auc: 0.666347
[3890]	valid_0's auc: 0.666355
[3900]	valid_0's auc: 0.666366
[3910]	valid_0's auc: 0.666377
[3920]	valid_0's auc: 0.666393
[3930]	valid_0's auc: 0.666408
[3940]	valid_0's auc: 0.666429
[3950]	valid_0's auc: 0.666444
[3960]	valid_0's auc: 0.666454
[3970]	valid_0's auc: 0.666467
[3980]	valid_0's auc: 0.666482
[3990]	valid_0's auc: 0.666497
[4000]	valid_0's auc: 0.666492
[4010]	valid_0's auc: 0.666513
[4020]	valid_0's auc: 0.666532
[4030]	valid_0's auc: 0.666544
[4040]	valid_0's auc: 0.666563
[4050]	valid_0's auc: 0.666579
[4060]	valid_0's auc: 0.66659
[4070]	valid_0's auc: 0.666586
[4080]	valid_0's auc: 0.666589
[4090]	valid_0's auc: 0.666593
[4100]	valid_0's auc: 0.666615
[4110]	valid_0's auc: 0.666618
[4120]	valid_0's auc: 0.666626
[4130]	valid_0's auc: 0.666643
[4140]	valid_0's auc: 0.666657
[4150]	valid_0's auc: 0.666665
[4160]	valid_0's auc: 0.66668
[4170]	valid_0's auc: 0.666692
[4180]	valid_0's auc: 0.666704
[4190]	valid_0's auc: 0.666717
[4200]	valid_0's auc: 0.666734
[4210]	valid_0's auc: 0.666742
[4220]	valid_0's auc: 0.666753
[4230]	valid_0's auc: 0.666767
[4240]	valid_0's auc: 0.666786
[4250]	valid_0's auc: 0.666804
[4260]	valid_0's auc: 0.666813
[4270]	valid_0's auc: 0.666819
[4280]	valid_0's auc: 0.666839
[4290]	valid_0's auc: 0.66685
[4300]	valid_0's auc: 0.66687
[4310]	valid_0's auc: 0.666874
[4320]	valid_0's auc: 0.666887
[4330]	valid_0's auc: 0.666894
[4340]	valid_0's auc: 0.666907
[4350]	valid_0's auc: 0.666918
[4360]	valid_0's auc: 0.666934
[4370]	valid_0's auc: 0.666946
[4380]	valid_0's auc: 0.666958
[4390]	valid_0's auc: 0.666972
[4400]	valid_0's auc: 0.666988
[4410]	valid_0's auc: 0.667007
[4420]	valid_0's auc: 0.667028
[4430]	valid_0's auc: 0.667032
[4440]	valid_0's auc: 0.667037
[4450]	valid_0's auc: 0.667038
[4460]	valid_0's auc: 0.667052
[4470]	valid_0's auc: 0.667066
[4480]	valid_0's auc: 0.667088
[4490]	valid_0's auc: 0.6671
[4500]	valid_0's auc: 0.667113
[4510]	valid_0's auc: 0.667121
[4520]	valid_0's auc: 0.667128
[4530]	valid_0's auc: 0.667135
[4540]	valid_0's auc: 0.66715
[4550]	valid_0's auc: 0.667162
[4560]	valid_0's auc: 0.667164
[4570]	valid_0's auc: 0.667178
[4580]	valid_0's auc: 0.667194
[4590]	valid_0's auc: 0.667204
[4600]	valid_0's auc: 0.667223
[4610]	valid_0's auc: 0.667241
[4620]	valid_0's auc: 0.667251
[4630]	valid_0's auc: 0.667256
[4640]	valid_0's auc: 0.667262
[4650]	valid_0's auc: 0.667282
[4660]	valid_0's auc: 0.667298
[4670]	valid_0's auc: 0.667307
[4680]	valid_0's auc: 0.66732
[4690]	valid_0's auc: 0.667336
[4700]	valid_0's auc: 0.667353
[4710]	valid_0's auc: 0.667363
[4720]	valid_0's auc: 0.667376
[4730]	valid_0's auc: 0.667396
[4740]	valid_0's auc: 0.66741
[4750]	valid_0's auc: 0.667421
[4760]	valid_0's auc: 0.667438
[4770]	valid_0's auc: 0.667442
[4780]	valid_0's auc: 0.667455
[4790]	valid_0's auc: 0.667467
[4800]	valid_0's auc: 0.667478
[4810]	valid_0's auc: 0.667494
[4820]	valid_0's auc: 0.667508
[4830]	valid_0's auc: 0.667511
[4840]	valid_0's auc: 0.667511
[4850]	valid_0's auc: 0.66752
[4860]	valid_0's auc: 0.667527
[4870]	valid_0's auc: 0.667525
[4880]	valid_0's auc: 0.66753
[4890]	valid_0's auc: 0.667543
[4900]	valid_0's auc: 0.667547
[4910]	valid_0's auc: 0.667558
[4920]	valid_0's auc: 0.667569
[4930]	valid_0's auc: 0.667578
[4940]	valid_0's auc: 0.66759
[4950]	valid_0's auc: 0.667606
[4960]	valid_0's auc: 0.667614
[4970]	valid_0's auc: 0.667631
[4980]	valid_0's auc: 0.667642
[4990]	valid_0's auc: 0.667641
[5000]	valid_0's auc: 0.667646
[5010]	valid_0's auc: 0.667651
[5020]	valid_0's auc: 0.667661
[5030]	valid_0's auc: 0.667665
[5040]	valid_0's auc: 0.667687
[5050]	valid_0's auc: 0.667709
[5060]	valid_0's auc: 0.667707
[5070]	valid_0's auc: 0.667712
[5080]	valid_0's auc: 0.667724
[5090]	valid_0's auc: 0.66773
[5100]	valid_0's auc: 0.667733
[5110]	valid_0's auc: 0.667737
[5120]	valid_0's auc: 0.667764
[5130]	valid_0's auc: 0.667772
[5140]	valid_0's auc: 0.667782
[5150]	valid_0's auc: 0.667821
[5160]	valid_0's auc: 0.667877
[5170]	valid_0's auc: 0.667896
[5180]	valid_0's auc: 0.667919
[5190]	valid_0's auc: 0.667925
[5200]	valid_0's auc: 0.66794
[5210]	valid_0's auc: 0.667945
[5220]	valid_0's auc: 0.66797
[5230]	valid_0's auc: 0.667988
[5240]	valid_0's auc: 0.668
[5250]	valid_0's auc: 0.668034
[5260]	valid_0's auc: 0.668045
[5270]	valid_0's auc: 0.668068
[5280]	valid_0's auc: 0.668095
[5290]	valid_0's auc: 0.668103
[5300]	valid_0's auc: 0.668114
[5310]	valid_0's auc: 0.668143
[5320]	valid_0's auc: 0.668175
[5330]	valid_0's auc: 0.668201
[5340]	valid_0's auc: 0.668211
[5350]	valid_0's auc: 0.668239
[5360]	valid_0's auc: 0.668261
[5370]	valid_0's auc: 0.668296
[5380]	valid_0's auc: 0.668339
[5390]	valid_0's auc: 0.668349
[5400]	valid_0's auc: 0.668359
[5410]	valid_0's auc: 0.668362
[5420]	valid_0's auc: 0.668359
[5430]	valid_0's auc: 0.668362
[5440]	valid_0's auc: 0.668372
[5450]	valid_0's auc: 0.668378
[5460]	valid_0's auc: 0.668412
[5470]	valid_0's auc: 0.668434
[5480]	valid_0's auc: 0.668452
[5490]	valid_0's auc: 0.668453
[5500]	valid_0's auc: 0.668464
[5510]	valid_0's auc: 0.668472
[5520]	valid_0's auc: 0.668477
[5530]	valid_0's auc: 0.668475
[5540]	valid_0's auc: 0.668482
[5550]	valid_0's auc: 0.668486
[5560]	valid_0's auc: 0.668506
[5570]	valid_0's auc: 0.668505
[5580]	valid_0's auc: 0.668511
[5590]	valid_0's auc: 0.668516
[5600]	valid_0's auc: 0.668516
[5610]	valid_0's auc: 0.66853
[5620]	valid_0's auc: 0.668532
[5630]	valid_0's auc: 0.668534
[5640]	valid_0's auc: 0.668539
[5650]	valid_0's auc: 0.668544
[5660]	valid_0's auc: 0.668547
[5670]	valid_0's auc: 0.66856
[5680]	valid_0's auc: 0.668565
[5690]	valid_0's auc: 0.668572
[5700]	valid_0's auc: 0.668575
[5710]	valid_0's auc: 0.66858
[5720]	valid_0's auc: 0.668584
[5730]	valid_0's auc: 0.668594
[5740]	valid_0's auc: 0.668601
[5750]	valid_0's auc: 0.668611
[5760]	valid_0's auc: 0.668613
[5770]	valid_0's auc: 0.668617
[5780]	valid_0's auc: 0.668618
[5790]	valid_0's auc: 0.668629
[5800]	valid_0's auc: 0.668633
[5810]	valid_0's auc: 0.668638
[5820]	valid_0's auc: 0.668646
[5830]	valid_0's auc: 0.668653
[5840]	valid_0's auc: 0.668654
[5850]	valid_0's auc: 0.668657
[5860]	valid_0's auc: 0.668659
[5870]	valid_0's auc: 0.668662
[5880]	valid_0's auc: 0.668677
[5890]	valid_0's auc: 0.66868
[5900]	valid_0's auc: 0.668693
[5910]	valid_0's auc: 0.668717
[5920]	valid_0's auc: 0.66872
[5930]	valid_0's auc: 0.66873
[5940]	valid_0's auc: 0.668738
[5950]	valid_0's auc: 0.668741
[5960]	valid_0's auc: 0.66875
[5970]	valid_0's auc: 0.668753
[5980]	valid_0's auc: 0.66876
[5990]	valid_0's auc: 0.668764
[6000]	valid_0's auc: 0.668767
[6010]	valid_0's auc: 0.668774
[6020]	valid_0's auc: 0.668785
[6030]	valid_0's auc: 0.668782
[6040]	valid_0's auc: 0.668784
[6050]	valid_0's auc: 0.668787
[6060]	valid_0's auc: 0.668788
[6070]	valid_0's auc: 0.66879
[6080]	valid_0's auc: 0.668801
[6090]	valid_0's auc: 0.668819
[6100]	valid_0's auc: 0.668825
[6110]	valid_0's auc: 0.668831
[6120]	valid_0's auc: 0.668833
[6130]	valid_0's auc: 0.668847
[6140]	valid_0's auc: 0.668864
[6150]	valid_0's auc: 0.668873
[6160]	valid_0's auc: 0.668879
[6170]	valid_0's auc: 0.668885
[6180]	valid_0's auc: 0.668886
[6190]	valid_0's auc: 0.668899
[6200]	valid_0's auc: 0.668907
[6210]	valid_0's auc: 0.668912
[6220]	valid_0's auc: 0.66892
[6230]	valid_0's auc: 0.668926
[6240]	valid_0's auc: 0.668928
[6250]	valid_0's auc: 0.668928
[6260]	valid_0's auc: 0.668932
[6270]	valid_0's auc: 0.66894
[6280]	valid_0's auc: 0.668941
[6290]	valid_0's auc: 0.668957
[6300]	valid_0's auc: 0.668961
[6310]	valid_0's auc: 0.668961
[6320]	valid_0's auc: 0.668965
[6330]	valid_0's auc: 0.668967
[6340]	valid_0's auc: 0.668977
[6350]	valid_0's auc: 0.668983
[6360]	valid_0's auc: 0.668994
[6370]	valid_0's auc: 0.669007
[6380]	valid_0's auc: 0.66901
[6390]	valid_0's auc: 0.669019
[6400]	valid_0's auc: 0.669023
[6410]	valid_0's auc: 0.669026
[6420]	valid_0's auc: 0.669026
[6430]	valid_0's auc: 0.669042
[6440]	valid_0's auc: 0.669048
[6450]	valid_0's auc: 0.669048
[6460]	valid_0's auc: 0.669054
[6470]	valid_0's auc: 0.669062
[6480]	valid_0's auc: 0.669067
[6490]	valid_0's auc: 0.669078
[6500]	valid_0's auc: 0.669096
[6510]	valid_0's auc: 0.669102
[6520]	valid_0's auc: 0.669112
[6530]	valid_0's auc: 0.669127
[6540]	valid_0's auc: 0.669135
[6550]	valid_0's auc: 0.669142
[6560]	valid_0's auc: 0.669149
[6570]	valid_0's auc: 0.669152
[6580]	valid_0's auc: 0.669158
[6590]	valid_0's auc: 0.669161
[6600]	valid_0's auc: 0.669165
[6610]	valid_0's auc: 0.669174
[6620]	valid_0's auc: 0.669181
[6630]	valid_0's auc: 0.669185
[6640]	valid_0's auc: 0.669185
[6650]	valid_0's auc: 0.6692
[6660]	valid_0's auc: 0.669215
[6670]	valid_0's auc: 0.669227
[6680]	valid_0's auc: 0.669232
[6690]	valid_0's auc: 0.66924
[6700]	valid_0's auc: 0.66926
[6710]	valid_0's auc: 0.669283
[6720]	valid_0's auc: 0.669299
[6730]	valid_0's auc: 0.669324
[6740]	valid_0's auc: 0.669325
[6750]	valid_0's auc: 0.669329
[6760]	valid_0's auc: 0.669331
[6770]	valid_0's auc: 0.669339
[6780]	valid_0's auc: 0.66935
[6790]	valid_0's auc: 0.669368
[6800]	valid_0's auc: 0.669369
[6810]	valid_0's auc: 0.669373
[6820]	valid_0's auc: 0.669382
[6830]	valid_0's auc: 0.669385
[6840]	valid_0's auc: 0.669392
[6850]	valid_0's auc: 0.669394
[6860]	valid_0's auc: 0.669411
[6870]	valid_0's auc: 0.669416
[6880]	valid_0's auc: 0.669418
[6890]	valid_0's auc: 0.669428
[6900]	valid_0's auc: 0.669433
[6910]	valid_0's auc: 0.669439
[6920]	valid_0's auc: 0.669451
[6930]	valid_0's auc: 0.669446
[6940]	valid_0's auc: 0.669449
[6950]	valid_0's auc: 0.669459
[6960]	valid_0's auc: 0.669472
[6970]	valid_0's auc: 0.6695
[6980]	valid_0's auc: 0.669529
[6990]	valid_0's auc: 0.669543
[7000]	valid_0's auc: 0.669549
[7010]	valid_0's auc: 0.669566
[7020]	valid_0's auc: 0.669574
[7030]	valid_0's auc: 0.6696
[7040]	valid_0's auc: 0.669603
[7050]	valid_0's auc: 0.669609
[7060]	valid_0's auc: 0.669617
[7070]	valid_0's auc: 0.669648
[7080]	valid_0's auc: 0.669668
[7090]	valid_0's auc: 0.669685
[7100]	valid_0's auc: 0.669692
[7110]	valid_0's auc: 0.669694
[7120]	valid_0's auc: 0.669704
[7130]	valid_0's auc: 0.669699
[7140]	valid_0's auc: 0.669701
[7150]	valid_0's auc: 0.669707
[7160]	valid_0's auc: 0.669712
[7170]	valid_0's auc: 0.669714
[7180]	valid_0's auc: 0.669717
[7190]	valid_0's auc: 0.66972
[7200]	valid_0's auc: 0.669724
[7210]	valid_0's auc: 0.66973
[7220]	valid_0's auc: 0.669735
[7230]	valid_0's auc: 0.669738
[7240]	valid_0's auc: 0.669748
[7250]	valid_0's auc: 0.669762
[7260]	valid_0's auc: 0.669763
[7270]	valid_0's auc: 0.669769
[7280]	valid_0's auc: 0.669777
[7290]	valid_0's auc: 0.669781
[7300]	valid_0's auc: 0.669794
[7310]	valid_0's auc: 0.669804
[7320]	valid_0's auc: 0.669812
[7330]	valid_0's auc: 0.669815
[7340]	valid_0's auc: 0.669821
[7350]	valid_0's auc: 0.669826
[7360]	valid_0's auc: 0.669838
[7370]	valid_0's auc: 0.669841
[7380]	valid_0's auc: 0.669849
[7390]	valid_0's auc: 0.669862
[7400]	valid_0's auc: 0.669868
[7410]	valid_0's auc: 0.669869
[7420]	valid_0's auc: 0.669871
[7430]	valid_0's auc: 0.669873
[7440]	valid_0's auc: 0.669877
[7450]	valid_0's auc: 0.669876
[7460]	valid_0's auc: 0.66989
[7470]	valid_0's auc: 0.669912
[7480]	valid_0's auc: 0.669932
[7490]	valid_0's auc: 0.669958
[7500]	valid_0's auc: 0.669977
[7510]	valid_0's auc: 0.669997
[7520]	valid_0's auc: 0.670026
[7530]	valid_0's auc: 0.670032
[7540]	valid_0's auc: 0.670031
[7550]	valid_0's auc: 0.670038
[7560]	valid_0's auc: 0.670039
[7570]	valid_0's auc: 0.670039
[7580]	valid_0's auc: 0.670046
[7590]	valid_0's auc: 0.670043
[7600]	valid_0's auc: 0.670048
[7610]	valid_0's auc: 0.670052
[7620]	valid_0's auc: 0.670054
[7630]	valid_0's auc: 0.670064
[7640]	valid_0's auc: 0.670066
[7650]	valid_0's auc: 0.670076
[7660]	valid_0's auc: 0.67008
[7670]	valid_0's auc: 0.67008
[7680]	valid_0's auc: 0.670086
[7690]	valid_0's auc: 0.670087
[7700]	valid_0's auc: 0.67009
[7710]	valid_0's auc: 0.670098
[7720]	valid_0's auc: 0.670106
[7730]	valid_0's auc: 0.670111
[7740]	valid_0's auc: 0.670111
[7750]	valid_0's auc: 0.670116
[7760]	valid_0's auc: 0.670116
[7770]	valid_0's auc: 0.670118
[7780]	valid_0's auc: 0.670123
[7790]	valid_0's auc: 0.670124
[7800]	valid_0's auc: 0.670128
[7810]	valid_0's auc: 0.670133
[7820]	valid_0's auc: 0.670136
[7830]	valid_0's auc: 0.670135
[7840]	valid_0's auc: 0.670138
[7850]	valid_0's auc: 0.670143
[7860]	valid_0's auc: 0.670147
[7870]	valid_0's auc: 0.670158
[7880]	valid_0's auc: 0.670178
[7890]	valid_0's auc: 0.670196
[7900]	valid_0's auc: 0.670218
[7910]	valid_0's auc: 0.670229
[7920]	valid_0's auc: 0.670236
[7930]	valid_0's auc: 0.670233
[7940]	valid_0's auc: 0.670233
[7950]	valid_0's auc: 0.670237
[7960]	valid_0's auc: 0.670243
[7970]	valid_0's auc: 0.670248
[7980]	valid_0's auc: 0.670255
[7990]	valid_0's auc: 0.670256
[8000]	valid_0's auc: 0.670264
[8010]	valid_0's auc: 0.67027
[8020]	valid_0's auc: 0.670276
[8030]	valid_0's auc: 0.670276
[8040]	valid_0's auc: 0.670276
[8050]	valid_0's auc: 0.670278
[8060]	valid_0's auc: 0.670283
[8070]	valid_0's auc: 0.670284
[8080]	valid_0's auc: 0.670289
[8090]	valid_0's auc: 0.670291
[8100]	valid_0's auc: 0.670297
[8110]	valid_0's auc: 0.670301
[8120]	valid_0's auc: 0.670307
[8130]	valid_0's auc: 0.670308
[8140]	valid_0's auc: 0.670315
[8150]	valid_0's auc: 0.670324
[8160]	valid_0's auc: 0.67034
[8170]	valid_0's auc: 0.67036
[8180]	valid_0's auc: 0.670373
[8190]	valid_0's auc: 0.670379
[8200]	valid_0's auc: 0.670383
[8210]	valid_0's auc: 0.670384
[8220]	valid_0's auc: 0.670387
[8230]	valid_0's auc: 0.670391
[8240]	valid_0's auc: 0.670397
[8250]	valid_0's auc: 0.670402
[8260]	valid_0's auc: 0.670408
[8270]	valid_0's auc: 0.670406
[8280]	valid_0's auc: 0.670401
[8290]	valid_0's auc: 0.670403
[8300]	valid_0's auc: 0.670403
[8310]	valid_0's auc: 0.670409
[8320]	valid_0's auc: 0.670413
[8330]	valid_0's auc: 0.670419
[8340]	valid_0's auc: 0.670422
[8350]	valid_0's auc: 0.670427
[8360]	valid_0's auc: 0.670433
[8370]	valid_0's auc: 0.670454
[8380]	valid_0's auc: 0.670466
[8390]	valid_0's auc: 0.670472
[8400]	valid_0's auc: 0.670477
[8410]	valid_0's auc: 0.670477
[8420]	valid_0's auc: 0.670481
[8430]	valid_0's auc: 0.670481
[8440]	valid_0's auc: 0.670484
[8450]	valid_0's auc: 0.670487
[8460]	valid_0's auc: 0.670488
[8470]	valid_0's auc: 0.670483
[8480]	valid_0's auc: 0.670483
[8490]	valid_0's auc: 0.670488
[8500]	valid_0's auc: 0.670493
[8510]	valid_0's auc: 0.670492
[8520]	valid_0's auc: 0.670495
[8530]	valid_0's auc: 0.670501
[8540]	valid_0's auc: 0.670502
[8550]	valid_0's auc: 0.670507
[8560]	valid_0's auc: 0.670509
[8570]	valid_0's auc: 0.670511
[8580]	valid_0's auc: 0.670514
[8590]	valid_0's auc: 0.670521
[8600]	valid_0's auc: 0.670527
[8610]	valid_0's auc: 0.67053
[8620]	valid_0's auc: 0.670535
[8630]	valid_0's auc: 0.670549
[8640]	valid_0's auc: 0.670548
[8650]	valid_0's auc: 0.670555
[8660]	valid_0's auc: 0.670555
[8670]	valid_0's auc: 0.670558
[8680]	valid_0's auc: 0.67056
[8690]	valid_0's auc: 0.67056
[8700]	valid_0's auc: 0.670562
[8710]	valid_0's auc: 0.670565
[8720]	valid_0's auc: 0.670573
[8730]	valid_0's auc: 0.670572
[8740]	valid_0's auc: 0.670578
[8750]	valid_0's auc: 0.670581
[8760]	valid_0's auc: 0.670582
[8770]	valid_0's auc: 0.670588
[8780]	valid_0's auc: 0.670588
[8790]	valid_0's auc: 0.670592
[8800]	valid_0's auc: 0.670597
[8810]	valid_0's auc: 0.670597
[8820]	valid_0's auc: 0.670596
[8830]	valid_0's auc: 0.670593
[8840]	valid_0's auc: 0.670594
[8850]	valid_0's auc: 0.670599
[8860]	valid_0's auc: 0.670596
[8870]	valid_0's auc: 0.670599
[8880]	valid_0's auc: 0.670602
[8890]	valid_0's auc: 0.670607
[8900]	valid_0's auc: 0.670611
[8910]	valid_0's auc: 0.670608
[8920]	valid_0's auc: 0.670612
[8930]	valid_0's auc: 0.670615
[8940]	valid_0's auc: 0.670632
[8950]	valid_0's auc: 0.670638
[8960]	valid_0's auc: 0.670639
[8970]	valid_0's auc: 0.670642
[8980]	valid_0's auc: 0.670644
[8990]	valid_0's auc: 0.670652
[9000]	valid_0's auc: 0.670666
[9010]	valid_0's auc: 0.670669
[9020]	valid_0's auc: 0.67067
[9030]	valid_0's auc: 0.67067
[9040]	valid_0's auc: 0.670682
[9050]	valid_0's auc: 0.670685
[9060]	valid_0's auc: 0.670694
[9070]	valid_0's auc: 0.670693
[9080]	valid_0's auc: 0.670698
[9090]	valid_0's auc: 0.670702
[9100]	valid_0's auc: 0.67071
[9110]	valid_0's auc: 0.670712
[9120]	valid_0's auc: 0.670718
[9130]	valid_0's auc: 0.67072
[9140]	valid_0's auc: 0.670722
[9150]	valid_0's auc: 0.670723
[9160]	valid_0's auc: 0.670725
[9170]	valid_0's auc: 0.670731
[9180]	valid_0's auc: 0.670734
[9190]	valid_0's auc: 0.670735
[9200]	valid_0's auc: 0.670739
[9210]	valid_0's auc: 0.670744
[9220]	valid_0's auc: 0.670745
[9230]	valid_0's auc: 0.670748
[9240]	valid_0's auc: 0.670748
[9250]	valid_0's auc: 0.670749
[9260]	valid_0's auc: 0.670754
[9270]	valid_0's auc: 0.670756
[9280]	valid_0's auc: 0.67076
[9290]	valid_0's auc: 0.670775
[9300]	valid_0's auc: 0.670779
[9310]	valid_0's auc: 0.670781
[9320]	valid_0's auc: 0.670786
[9330]	valid_0's auc: 0.670792
[9340]	valid_0's auc: 0.670796
[9350]	valid_0's auc: 0.670806
[9360]	valid_0's auc: 0.670809
[9370]	valid_0's auc: 0.67081
[9380]	valid_0's auc: 0.670814
[9390]	valid_0's auc: 0.670811
[9400]	valid_0's auc: 0.670812
[9410]	valid_0's auc: 0.67082
[9420]	valid_0's auc: 0.670825
[9430]	valid_0's auc: 0.670826
[9440]	valid_0's auc: 0.670831
[9450]	valid_0's auc: 0.670833
[9460]	valid_0's auc: 0.670837
[9470]	valid_0's auc: 0.670838
[9480]	valid_0's auc: 0.670839
[9490]	valid_0's auc: 0.67084
[9500]	valid_0's auc: 0.67084
[9510]	valid_0's auc: 0.670841
[9520]	valid_0's auc: 0.670843
[9530]	valid_0's auc: 0.670844
[9540]	valid_0's auc: 0.670849
[9550]	valid_0's auc: 0.670857
[9560]	valid_0's auc: 0.670865
[9570]	valid_0's auc: 0.670867
[9580]	valid_0's auc: 0.670868
[9590]	valid_0's auc: 0.670869
[9600]	valid_0's auc: 0.670873
[9610]	valid_0's auc: 0.670872
[9620]	valid_0's auc: 0.670872
[9630]	valid_0's auc: 0.670875
[9640]	valid_0's auc: 0.670878
[9650]	valid_0's auc: 0.670882
[9660]	valid_0's auc: 0.670883
[9670]	valid_0's auc: 0.670885
[9680]	valid_0's auc: 0.670885
[9690]	valid_0's auc: 0.670882
[9700]	valid_0's auc: 0.670885
[9710]	valid_0's auc: 0.670888
[9720]	valid_0's auc: 0.670889
[9730]	valid_0's auc: 0.670883
[9740]	valid_0's auc: 0.670884
[9750]	valid_0's auc: 0.670884
[9760]	valid_0's auc: 0.670885
Early stopping, best iteration is:
[9711]	valid_0's auc: 0.67089
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/playground_V1006/training_V1304.py", line 150, in <module>
    verbose_eval=10,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1682, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 1
'''
