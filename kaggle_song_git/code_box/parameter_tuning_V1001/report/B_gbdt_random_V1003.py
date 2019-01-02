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
lr_s = [0.05, 0.03, 0.02, 0.03, 0.1]
nl_s = [ 1023,  1023,  511, 511, 511]
md_s = [  -1,   10,   11, -1, 10]
l2_s = [   0,  0.7,  0.5, 2, 0.5]
l1_s = [   0,    0,    0, 0, 0.5]
mb_s = [ 511,  511,  255, 255, 127]
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1003.py
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

max_bin              : 511
feature_fraction_seed : 2
learning_rate        : 0.05
lambda_l1            : 0
bagging_freq         : 2
feature_fraction     : 0.9
bagging_seed         : 2
boosting             : gbdt
bagging_fraction     : 0.9
num_leaves           : 1023
max_depth            : -1
lambda_l2            : 0

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:648: LGBMDeprecationWarning: The `max_bin` parameter is deprecated and will be removed in 2.0.12 version. Please use `params` to pass this parameter.
  'Please use `params` to pass this parameter.', LGBMDeprecationWarning)
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.66394
[20]	valid_0's auc: 0.666535
[30]	valid_0's auc: 0.669272
[40]	valid_0's auc: 0.670955
[50]	valid_0's auc: 0.672684
[60]	valid_0's auc: 0.673705
[70]	valid_0's auc: 0.674608
[80]	valid_0's auc: 0.675169
[90]	valid_0's auc: 0.675718
[100]	valid_0's auc: 0.676012
[110]	valid_0's auc: 0.676258
[120]	valid_0's auc: 0.676424
[130]	valid_0's auc: 0.676559
[140]	valid_0's auc: 0.676621
[150]	valid_0's auc: 0.676713
[160]	valid_0's auc: 0.676716
[170]	valid_0's auc: 0.676704
[180]	valid_0's auc: 0.67672
[190]	valid_0's auc: 0.676708
[200]	valid_0's auc: 0.676771
[210]	valid_0's auc: 0.676757
[220]	valid_0's auc: 0.676749
[230]	valid_0's auc: 0.676717
[240]	valid_0's auc: 0.676695
[250]	valid_0's auc: 0.67672
[260]	valid_0's auc: 0.676669
[270]	valid_0's auc: 0.676645
[280]	valid_0's auc: 0.676688
[290]	valid_0's auc: 0.676594
[300]	valid_0's auc: 0.676578
[310]	valid_0's auc: 0.676557
[320]	valid_0's auc: 0.67652
[330]	valid_0's auc: 0.67646
[340]	valid_0's auc: 0.676442
[350]	valid_0's auc: 0.67639
[360]	valid_0's auc: 0.676343
[370]	valid_0's auc: 0.676326
[380]	valid_0's auc: 0.67631
[390]	valid_0's auc: 0.67632
[400]	valid_0's auc: 0.67626
Early stopping, best iteration is:
[203]	valid_0's auc: 0.676783
best score: 0.676782831198
best iteration: 203

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 73m 6s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_bin              : 511
feature_fraction_seed : 2
learning_rate        : 0.03
lambda_l1            : 0
bagging_freq         : 2
feature_fraction     : 0.9
bagging_seed         : 2
boosting             : gbdt
bagging_fraction     : 0.9
num_leaves           : 1023
max_depth            : 10
lambda_l2            : 0.7

Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.62854
[20]	valid_0's auc: 0.631924
[30]	valid_0's auc: 0.633555
[40]	valid_0's auc: 0.635127
[50]	valid_0's auc: 0.63695
[60]	valid_0's auc: 0.63853
[70]	valid_0's auc: 0.640173
[80]	valid_0's auc: 0.641428
[90]	valid_0's auc: 0.642452
[100]	valid_0's auc: 0.64343
[110]	valid_0's auc: 0.644161
[120]	valid_0's auc: 0.644764
[130]	valid_0's auc: 0.645307
[140]	valid_0's auc: 0.645886
[150]	valid_0's auc: 0.646449
[160]	valid_0's auc: 0.646901
[170]	valid_0's auc: 0.647275
[180]	valid_0's auc: 0.647694
[190]	valid_0's auc: 0.648099
[200]	valid_0's auc: 0.648456
[210]	valid_0's auc: 0.648796
[220]	valid_0's auc: 0.649139
[230]	valid_0's auc: 0.649414
[240]	valid_0's auc: 0.649697
[250]	valid_0's auc: 0.64998
[260]	valid_0's auc: 0.650294
[270]	valid_0's auc: 0.650587
[280]	valid_0's auc: 0.650824
[290]	valid_0's auc: 0.651013
[300]	valid_0's auc: 0.651275
[310]	valid_0's auc: 0.65157
[320]	valid_0's auc: 0.651837
[330]	valid_0's auc: 0.652008
[340]	valid_0's auc: 0.652283
[350]	valid_0's auc: 0.652518
[360]	valid_0's auc: 0.652731
[370]	valid_0's auc: 0.65286
[380]	valid_0's auc: 0.653116
[390]	valid_0's auc: 0.653291
[400]	valid_0's auc: 0.653513
[410]	valid_0's auc: 0.653643
[420]	valid_0's auc: 0.653808
[430]	valid_0's auc: 0.653995
[440]	valid_0's auc: 0.654208
[450]	valid_0's auc: 0.654424
[460]	valid_0's auc: 0.654593
[470]	valid_0's auc: 0.654739
[480]	valid_0's auc: 0.654951
[490]	valid_0's auc: 0.655131
[500]	valid_0's auc: 0.655266
[510]	valid_0's auc: 0.655417
[520]	valid_0's auc: 0.655578
[530]	valid_0's auc: 0.655759
[540]	valid_0's auc: 0.655951
[550]	valid_0's auc: 0.656168
[560]	valid_0's auc: 0.656319
[570]	valid_0's auc: 0.656487
[580]	valid_0's auc: 0.656653
[590]	valid_0's auc: 0.656797
[600]	valid_0's auc: 0.65695
[610]	valid_0's auc: 0.657041
[620]	valid_0's auc: 0.657184
[630]	valid_0's auc: 0.657302
[640]	valid_0's auc: 0.657444
[650]	valid_0's auc: 0.65756
[660]	valid_0's auc: 0.657658
[670]	valid_0's auc: 0.657778
[680]	valid_0's auc: 0.657898
[690]	valid_0's auc: 0.657998
[700]	valid_0's auc: 0.658142
[710]	valid_0's auc: 0.658242
[720]	valid_0's auc: 0.658388
[730]	valid_0's auc: 0.658447
[740]	valid_0's auc: 0.658551
[750]	valid_0's auc: 0.658679
[760]	valid_0's auc: 0.658776
[770]	valid_0's auc: 0.658898
[780]	valid_0's auc: 0.658992
[790]	valid_0's auc: 0.659081
[800]	valid_0's auc: 0.659175
[810]	valid_0's auc: 0.659265
[820]	valid_0's auc: 0.659367
[830]	valid_0's auc: 0.65948
[840]	valid_0's auc: 0.659555
[850]	valid_0's auc: 0.659662
[860]	valid_0's auc: 0.659727
[870]	valid_0's auc: 0.659824
[880]	valid_0's auc: 0.659948
[890]	valid_0's auc: 0.660021
[900]	valid_0's auc: 0.660107
[910]	valid_0's auc: 0.660185
[920]	valid_0's auc: 0.660272
[930]	valid_0's auc: 0.660365
[940]	valid_0's auc: 0.660447
[950]	valid_0's auc: 0.660533
[960]	valid_0's auc: 0.6606
[970]	valid_0's auc: 0.660661
[980]	valid_0's auc: 0.660733
[990]	valid_0's auc: 0.660779
[1000]	valid_0's auc: 0.660837
[1010]	valid_0's auc: 0.66091
[1020]	valid_0's auc: 0.660982
[1030]	valid_0's auc: 0.661068
[1040]	valid_0's auc: 0.661156
[1050]	valid_0's auc: 0.66124
[1060]	valid_0's auc: 0.661301
[1070]	valid_0's auc: 0.661374
[1080]	valid_0's auc: 0.661444
[1090]	valid_0's auc: 0.661527
[1100]	valid_0's auc: 0.661599
[1110]	valid_0's auc: 0.661644
[1120]	valid_0's auc: 0.661704
[1130]	valid_0's auc: 0.661797
[1140]	valid_0's auc: 0.661828
[1150]	valid_0's auc: 0.661872
[1160]	valid_0's auc: 0.661953
[1170]	valid_0's auc: 0.662018
[1180]	valid_0's auc: 0.662053
[1190]	valid_0's auc: 0.662107
[1200]	valid_0's auc: 0.662165
[1210]	valid_0's auc: 0.662224
[1220]	valid_0's auc: 0.662283
[1230]	valid_0's auc: 0.662328
[1240]	valid_0's auc: 0.662397
[1250]	valid_0's auc: 0.662465
[1260]	valid_0's auc: 0.662517
[1270]	valid_0's auc: 0.66257
[1280]	valid_0's auc: 0.66261
[1290]	valid_0's auc: 0.66267
[1300]	valid_0's auc: 0.662714
[1310]	valid_0's auc: 0.662782
[1320]	valid_0's auc: 0.662828
[1330]	valid_0's auc: 0.662896
[1340]	valid_0's auc: 0.662952
[1350]	valid_0's auc: 0.662996
[1360]	valid_0's auc: 0.663039
[1370]	valid_0's auc: 0.663104
[1380]	valid_0's auc: 0.663164
[1390]	valid_0's auc: 0.663189
[1400]	valid_0's auc: 0.663271
[1410]	valid_0's auc: 0.663304
[1420]	valid_0's auc: 0.663359
[1430]	valid_0's auc: 0.663423
[1440]	valid_0's auc: 0.663473
[1450]	valid_0's auc: 0.663528
[1460]	valid_0's auc: 0.663553
[1470]	valid_0's auc: 0.663596
[1480]	valid_0's auc: 0.663623
[1490]	valid_0's auc: 0.663659
[1500]	valid_0's auc: 0.663694
[1510]	valid_0's auc: 0.663725
[1520]	valid_0's auc: 0.663776
[1530]	valid_0's auc: 0.663797
[1540]	valid_0's auc: 0.663845
[1550]	valid_0's auc: 0.66388
[1560]	valid_0's auc: 0.663907
[1570]	valid_0's auc: 0.663968
[1580]	valid_0's auc: 0.664005
[1590]	valid_0's auc: 0.664033
[1600]	valid_0's auc: 0.664096
[1610]	valid_0's auc: 0.664121
[1620]	valid_0's auc: 0.664171
[1630]	valid_0's auc: 0.664254
[1640]	valid_0's auc: 0.664289
[1650]	valid_0's auc: 0.664315
[1660]	valid_0's auc: 0.664341
[1670]	valid_0's auc: 0.664392
[1680]	valid_0's auc: 0.664426
[1690]	valid_0's auc: 0.664468
[1700]	valid_0's auc: 0.664513
[1710]	valid_0's auc: 0.664548
[1720]	valid_0's auc: 0.664566
[1730]	valid_0's auc: 0.664604
[1740]	valid_0's auc: 0.664642
[1750]	valid_0's auc: 0.664687
[1760]	valid_0's auc: 0.664715
[1770]	valid_0's auc: 0.664747
[1780]	valid_0's auc: 0.664785
[1790]	valid_0's auc: 0.66482
[1800]	valid_0's auc: 0.664843
[1810]	valid_0's auc: 0.664869
[1820]	valid_0's auc: 0.664932
[1830]	valid_0's auc: 0.664965
[1840]	valid_0's auc: 0.665001
[1850]	valid_0's auc: 0.665042
[1860]	valid_0's auc: 0.665056
[1870]	valid_0's auc: 0.66508
[1880]	valid_0's auc: 0.665118
[1890]	valid_0's auc: 0.665157
[1900]	valid_0's auc: 0.66521
[1910]	valid_0's auc: 0.665235
[1920]	valid_0's auc: 0.665259
[1930]	valid_0's auc: 0.665329
[1940]	valid_0's auc: 0.665364
[1950]	valid_0's auc: 0.665422
[1960]	valid_0's auc: 0.665441
[1970]	valid_0's auc: 0.66546
[1980]	valid_0's auc: 0.665492
[1990]	valid_0's auc: 0.665526
[2000]	valid_0's auc: 0.665588
[2010]	valid_0's auc: 0.665624
[2020]	valid_0's auc: 0.665641
[2030]	valid_0's auc: 0.66566
[2040]	valid_0's auc: 0.66572
[2050]	valid_0's auc: 0.665753
[2060]	valid_0's auc: 0.66577
[2070]	valid_0's auc: 0.665829
[2080]	valid_0's auc: 0.665865
[2090]	valid_0's auc: 0.665922
[2100]	valid_0's auc: 0.665939
[2110]	valid_0's auc: 0.665954
[2120]	valid_0's auc: 0.665988
[2130]	valid_0's auc: 0.666025
[2140]	valid_0's auc: 0.666057
[2150]	valid_0's auc: 0.66609
[2160]	valid_0's auc: 0.666092
[2170]	valid_0's auc: 0.666137
[2180]	valid_0's auc: 0.666173
[2190]	valid_0's auc: 0.666196
[2200]	valid_0's auc: 0.666223
[2210]	valid_0's auc: 0.666243
[2220]	valid_0's auc: 0.666305
[2230]	valid_0's auc: 0.666335
[2240]	valid_0's auc: 0.666358
[2250]	valid_0's auc: 0.666382
[2260]	valid_0's auc: 0.66641
[2270]	valid_0's auc: 0.666433
[2280]	valid_0's auc: 0.666465
[2290]	valid_0's auc: 0.66648
[2300]	valid_0's auc: 0.666514
[2310]	valid_0's auc: 0.666552
[2320]	valid_0's auc: 0.666572
[2330]	valid_0's auc: 0.666591
[2340]	valid_0's auc: 0.666642
[2350]	valid_0's auc: 0.666682
[2360]	valid_0's auc: 0.66671
[2370]	valid_0's auc: 0.66673
[2380]	valid_0's auc: 0.666762
[2390]	valid_0's auc: 0.666785
[2400]	valid_0's auc: 0.666801
[2410]	valid_0's auc: 0.666835
[2420]	valid_0's auc: 0.666851
[2430]	valid_0's auc: 0.666869
[2440]	valid_0's auc: 0.666893
[2450]	valid_0's auc: 0.666923
[2460]	valid_0's auc: 0.66694
[2470]	valid_0's auc: 0.666992
[2480]	valid_0's auc: 0.667038
[2490]	valid_0's auc: 0.667098
[2500]	valid_0's auc: 0.667107
[2510]	valid_0's auc: 0.667128
[2520]	valid_0's auc: 0.667157
[2530]	valid_0's auc: 0.667183
[2540]	valid_0's auc: 0.667235
[2550]	valid_0's auc: 0.667284
[2560]	valid_0's auc: 0.667355
[2570]	valid_0's auc: 0.667376
[2580]	valid_0's auc: 0.667382
[2590]	valid_0's auc: 0.667404
[2600]	valid_0's auc: 0.667426
[2610]	valid_0's auc: 0.667445
[2620]	valid_0's auc: 0.667489
[2630]	valid_0's auc: 0.667515
[2640]	valid_0's auc: 0.667546
[2650]	valid_0's auc: 0.667557
[2660]	valid_0's auc: 0.667585
[2670]	valid_0's auc: 0.667617
[2680]	valid_0's auc: 0.667648
[2690]	valid_0's auc: 0.667678
[2700]	valid_0's auc: 0.667705
[2710]	valid_0's auc: 0.667727
[2720]	valid_0's auc: 0.667748
[2730]	valid_0's auc: 0.667757
[2740]	valid_0's auc: 0.66777
[2750]	valid_0's auc: 0.667782
[2760]	valid_0's auc: 0.667803
[2770]	valid_0's auc: 0.667835
[2780]	valid_0's auc: 0.667846
[2790]	valid_0's auc: 0.667865
[2800]	valid_0's auc: 0.667887
[2810]	valid_0's auc: 0.667907
[2820]	valid_0's auc: 0.667964
[2830]	valid_0's auc: 0.667976
[2840]	valid_0's auc: 0.668001
[2850]	valid_0's auc: 0.668053
[2860]	valid_0's auc: 0.66809
[2870]	valid_0's auc: 0.668123
[2880]	valid_0's auc: 0.66815
[2890]	valid_0's auc: 0.668171
[2900]	valid_0's auc: 0.668182
[2910]	valid_0's auc: 0.668186
[2920]	valid_0's auc: 0.668202
[2930]	valid_0's auc: 0.668222
[2940]	valid_0's auc: 0.668235
[2950]	valid_0's auc: 0.668251
[2960]	valid_0's auc: 0.668268
[2970]	valid_0's auc: 0.668271
[2980]	valid_0's auc: 0.668299
[2990]	valid_0's auc: 0.668322
[3000]	valid_0's auc: 0.668368
[3010]	valid_0's auc: 0.668409
[3020]	valid_0's auc: 0.668439
[3030]	valid_0's auc: 0.668439
[3040]	valid_0's auc: 0.668481
[3050]	valid_0's auc: 0.668513
[3060]	valid_0's auc: 0.66852
[3070]	valid_0's auc: 0.668533
[3080]	valid_0's auc: 0.668577
[3090]	valid_0's auc: 0.668584
[3100]	valid_0's auc: 0.668596
[3110]	valid_0's auc: 0.668606
[3120]	valid_0's auc: 0.668615
[3130]	valid_0's auc: 0.66863
[3140]	valid_0's auc: 0.668638
[3150]	valid_0's auc: 0.668649
[3160]	valid_0's auc: 0.66865
[3170]	valid_0's auc: 0.66869
[3180]	valid_0's auc: 0.668718
[3190]	valid_0's auc: 0.66875
[3200]	valid_0's auc: 0.668774
[3210]	valid_0's auc: 0.668813
[3220]	valid_0's auc: 0.668832
[3230]	valid_0's auc: 0.668822
[3240]	valid_0's auc: 0.668846
[3250]	valid_0's auc: 0.668863
[3260]	valid_0's auc: 0.668883
[3270]	valid_0's auc: 0.668899
[3280]	valid_0's auc: 0.668904
[3290]	valid_0's auc: 0.668909
[3300]	valid_0's auc: 0.668923
[3310]	valid_0's auc: 0.668941
[3320]	valid_0's auc: 0.668966
[3330]	valid_0's auc: 0.66898
[3340]	valid_0's auc: 0.669
[3350]	valid_0's auc: 0.669004
[3360]	valid_0's auc: 0.669028
[3370]	valid_0's auc: 0.669049
[3380]	valid_0's auc: 0.669057
[3390]	valid_0's auc: 0.669071
[3400]	valid_0's auc: 0.669078
[3410]	valid_0's auc: 0.669072
[3420]	valid_0's auc: 0.669097
[3430]	valid_0's auc: 0.669105
[3440]	valid_0's auc: 0.669115
[3450]	valid_0's auc: 0.669113
[3460]	valid_0's auc: 0.66913
[3470]	valid_0's auc: 0.669136
[3480]	valid_0's auc: 0.669138
[3490]	valid_0's auc: 0.669163
[3500]	valid_0's auc: 0.669183
[3510]	valid_0's auc: 0.669188
[3520]	valid_0's auc: 0.669207
[3530]	valid_0's auc: 0.669225
[3540]	valid_0's auc: 0.669242
[3550]	valid_0's auc: 0.669271
[3560]	valid_0's auc: 0.669279
[3570]	valid_0's auc: 0.669311
[3580]	valid_0's auc: 0.669332
[3590]	valid_0's auc: 0.669349
[3600]	valid_0's auc: 0.669369
[3610]	valid_0's auc: 0.669395
[3620]	valid_0's auc: 0.669416
[3630]	valid_0's auc: 0.66944
[3640]	valid_0's auc: 0.66945
[3650]	valid_0's auc: 0.66946
[3660]	valid_0's auc: 0.669473
[3670]	valid_0's auc: 0.669487
[3680]	valid_0's auc: 0.669493
[3690]	valid_0's auc: 0.669499
[3700]	valid_0's auc: 0.669507
[3710]	valid_0's auc: 0.66952
[3720]	valid_0's auc: 0.669551
[3730]	valid_0's auc: 0.66956
[3740]	valid_0's auc: 0.669566
[3750]	valid_0's auc: 0.669577
[3760]	valid_0's auc: 0.669582
[3770]	valid_0's auc: 0.669602
[3780]	valid_0's auc: 0.669601
[3790]	valid_0's auc: 0.669613
[3800]	valid_0's auc: 0.669625
[3810]	valid_0's auc: 0.669633
[3820]	valid_0's auc: 0.669641
[3830]	valid_0's auc: 0.669653
[3840]	valid_0's auc: 0.669661
[3850]	valid_0's auc: 0.669674
[3860]	valid_0's auc: 0.669688
[3870]	valid_0's auc: 0.669706
[3880]	valid_0's auc: 0.669721
[3890]	valid_0's auc: 0.669727
[3900]	valid_0's auc: 0.669736
[3910]	valid_0's auc: 0.669753
[3920]	valid_0's auc: 0.669759
[3930]	valid_0's auc: 0.66978
[3940]	valid_0's auc: 0.669788
[3950]	valid_0's auc: 0.669781
[3960]	valid_0's auc: 0.669791
[3970]	valid_0's auc: 0.669826
[3980]	valid_0's auc: 0.66983
[3990]	valid_0's auc: 0.669834
[4000]	valid_0's auc: 0.66985
[4010]	valid_0's auc: 0.669861
[4020]	valid_0's auc: 0.669873
[4030]	valid_0's auc: 0.669901
[4040]	valid_0's auc: 0.669916
[4050]	valid_0's auc: 0.669926
[4060]	valid_0's auc: 0.669936
[4070]	valid_0's auc: 0.669951
[4080]	valid_0's auc: 0.669958
[4090]	valid_0's auc: 0.669965
[4100]	valid_0's auc: 0.66997
[4110]	valid_0's auc: 0.669993
[4120]	valid_0's auc: 0.670001
[4130]	valid_0's auc: 0.670022
[4140]	valid_0's auc: 0.670027
[4150]	valid_0's auc: 0.67004
[4160]	valid_0's auc: 0.670051
[4170]	valid_0's auc: 0.670066
[4180]	valid_0's auc: 0.670067
[4190]	valid_0's auc: 0.670076
[4200]	valid_0's auc: 0.670084
[4210]	valid_0's auc: 0.670087
[4220]	valid_0's auc: 0.670096
[4230]	valid_0's auc: 0.670113
[4240]	valid_0's auc: 0.67017
[4250]	valid_0's auc: 0.670175
[4260]	valid_0's auc: 0.670178
[4270]	valid_0's auc: 0.670192
[4280]	valid_0's auc: 0.6702
[4290]	valid_0's auc: 0.670216
[4300]	valid_0's auc: 0.670223
[4310]	valid_0's auc: 0.670251
[4320]	valid_0's auc: 0.670259
[4330]	valid_0's auc: 0.670267
[4340]	valid_0's auc: 0.670272
[4350]	valid_0's auc: 0.670283
[4360]	valid_0's auc: 0.670292
[4370]	valid_0's auc: 0.670292
[4380]	valid_0's auc: 0.670297
[4390]	valid_0's auc: 0.670305
[4400]	valid_0's auc: 0.670309
[4410]	valid_0's auc: 0.670316
[4420]	valid_0's auc: 0.670323
[4430]	valid_0's auc: 0.670343
[4440]	valid_0's auc: 0.670361
[4450]	valid_0's auc: 0.670382
[4460]	valid_0's auc: 0.670392
[4470]	valid_0's auc: 0.670397
[4480]	valid_0's auc: 0.6704
[4490]	valid_0's auc: 0.67042
[4500]	valid_0's auc: 0.670424
[4510]	valid_0's auc: 0.670448
[4520]	valid_0's auc: 0.670454
[4530]	valid_0's auc: 0.67045
[4540]	valid_0's auc: 0.670474
[4550]	valid_0's auc: 0.670472
[4560]	valid_0's auc: 0.670503
[4570]	valid_0's auc: 0.670514
[4580]	valid_0's auc: 0.670516
[4590]	valid_0's auc: 0.670517
[4600]	valid_0's auc: 0.670542
[4610]	valid_0's auc: 0.670566
[4620]	valid_0's auc: 0.670569
[4630]	valid_0's auc: 0.670581
[4640]	valid_0's auc: 0.670585
[4650]	valid_0's auc: 0.670599
[4660]	valid_0's auc: 0.670605
[4670]	valid_0's auc: 0.670619
[4680]	valid_0's auc: 0.670627
[4690]	valid_0's auc: 0.670636
[4700]	valid_0's auc: 0.670657
[4710]	valid_0's auc: 0.670658
[4720]	valid_0's auc: 0.670662
[4730]	valid_0's auc: 0.670675
[4740]	valid_0's auc: 0.670688
[4750]	valid_0's auc: 0.670715
[4760]	valid_0's auc: 0.67072
[4770]	valid_0's auc: 0.670731
[4780]	valid_0's auc: 0.670746
[4790]	valid_0's auc: 0.670752
[4800]	valid_0's auc: 0.670758
[4810]	valid_0's auc: 0.670763
[4820]	valid_0's auc: 0.670772
[4830]	valid_0's auc: 0.670775
[4840]	valid_0's auc: 0.670783
[4850]	valid_0's auc: 0.670794
[4860]	valid_0's auc: 0.670809
[4870]	valid_0's auc: 0.67081
[4880]	valid_0's auc: 0.670811
[4890]	valid_0's auc: 0.670823
[4900]	valid_0's auc: 0.670834
[4910]	valid_0's auc: 0.670842
[4920]	valid_0's auc: 0.670842
[4930]	valid_0's auc: 0.67084
[4940]	valid_0's auc: 0.67085
[4950]	valid_0's auc: 0.670855
[4960]	valid_0's auc: 0.670864
[4970]	valid_0's auc: 0.670874
[4980]	valid_0's auc: 0.670885
[4990]	valid_0's auc: 0.670887
[5000]	valid_0's auc: 0.670889
[5010]	valid_0's auc: 0.67088
[5020]	valid_0's auc: 0.670892
[5030]	valid_0's auc: 0.670892
[5040]	valid_0's auc: 0.670902
[5050]	valid_0's auc: 0.670905
[5060]	valid_0's auc: 0.670915
[5070]	valid_0's auc: 0.67092
[5080]	valid_0's auc: 0.670917
[5090]	valid_0's auc: 0.670923
[5100]	valid_0's auc: 0.670939
[5110]	valid_0's auc: 0.670964
[5120]	valid_0's auc: 0.670969
[5130]	valid_0's auc: 0.670975
[5140]	valid_0's auc: 0.670984
[5150]	valid_0's auc: 0.670994
[5160]	valid_0's auc: 0.670998
[5170]	valid_0's auc: 0.671027
[5180]	valid_0's auc: 0.671035
[5190]	valid_0's auc: 0.671039
[5200]	valid_0's auc: 0.67105
[5210]	valid_0's auc: 0.671073
[5220]	valid_0's auc: 0.671077
[5230]	valid_0's auc: 0.671099
[5240]	valid_0's auc: 0.671116
[5250]	valid_0's auc: 0.671128
[5260]	valid_0's auc: 0.671138
[5270]	valid_0's auc: 0.671147
[5280]	valid_0's auc: 0.671157
[5290]	valid_0's auc: 0.671161
[5300]	valid_0's auc: 0.671164
[5310]	valid_0's auc: 0.671177
[5320]	valid_0's auc: 0.671176
[5330]	valid_0's auc: 0.671189
[5340]	valid_0's auc: 0.671191
[5350]	valid_0's auc: 0.6712
[5360]	valid_0's auc: 0.671205
[5370]	valid_0's auc: 0.671208
[5380]	valid_0's auc: 0.671215
[5390]	valid_0's auc: 0.67122
[5400]	valid_0's auc: 0.671221
[5410]	valid_0's auc: 0.671223
[5420]	valid_0's auc: 0.671232
[5430]	valid_0's auc: 0.671244
[5440]	valid_0's auc: 0.671247
[5450]	valid_0's auc: 0.671253
[5460]	valid_0's auc: 0.67127
[5470]	valid_0's auc: 0.671279
[5480]	valid_0's auc: 0.67128
[5490]	valid_0's auc: 0.671278
[5500]	valid_0's auc: 0.67128
[5510]	valid_0's auc: 0.671265
[5520]	valid_0's auc: 0.671277
[5530]	valid_0's auc: 0.671283
[5540]	valid_0's auc: 0.671289
[5550]	valid_0's auc: 0.671292
[5560]	valid_0's auc: 0.671295
[5570]	valid_0's auc: 0.671302
[5580]	valid_0's auc: 0.6713
[5590]	valid_0's auc: 0.671308
[5600]	valid_0's auc: 0.671309
[5610]	valid_0's auc: 0.67131
[5620]	valid_0's auc: 0.671311
[5630]	valid_0's auc: 0.671322
[5640]	valid_0's auc: 0.671325
[5650]	valid_0's auc: 0.671334
[5660]	valid_0's auc: 0.671341
[5670]	valid_0's auc: 0.671353
[5680]	valid_0's auc: 0.671368
[5690]	valid_0's auc: 0.671372
[5700]	valid_0's auc: 0.671375
[5710]	valid_0's auc: 0.671376
[5720]	valid_0's auc: 0.671375
[5730]	valid_0's auc: 0.671377
[5740]	valid_0's auc: 0.671378
[5750]	valid_0's auc: 0.671382
[5760]	valid_0's auc: 0.67139
[5770]	valid_0's auc: 0.671398
[5780]	valid_0's auc: 0.671395
[5790]	valid_0's auc: 0.671401
[5800]	valid_0's auc: 0.671413
[5810]	valid_0's auc: 0.671427
[5820]	valid_0's auc: 0.67143
[5830]	valid_0's auc: 0.671442
[5840]	valid_0's auc: 0.671463
[5850]	valid_0's auc: 0.671472
[5860]	valid_0's auc: 0.671479
[5870]	valid_0's auc: 0.671473
[5880]	valid_0's auc: 0.671478
[5890]	valid_0's auc: 0.671486
[5900]	valid_0's auc: 0.671483
[5910]	valid_0's auc: 0.671488
[5920]	valid_0's auc: 0.671488
[5930]	valid_0's auc: 0.671502
[5940]	valid_0's auc: 0.671507
[5950]	valid_0's auc: 0.671509
[5960]	valid_0's auc: 0.671512
[5970]	valid_0's auc: 0.67152
[5980]	valid_0's auc: 0.671535
[5990]	valid_0's auc: 0.671535
[6000]	valid_0's auc: 0.671533
[6010]	valid_0's auc: 0.67153
[6020]	valid_0's auc: 0.671536
[6030]	valid_0's auc: 0.671536
[6040]	valid_0's auc: 0.67154
[6050]	valid_0's auc: 0.671539
[6060]	valid_0's auc: 0.671545
[6070]	valid_0's auc: 0.671545
[6080]	valid_0's auc: 0.671547
[6090]	valid_0's auc: 0.671554
[6100]	valid_0's auc: 0.671562
[6110]	valid_0's auc: 0.67157
[6120]	valid_0's auc: 0.671575
[6130]	valid_0's auc: 0.671577
[6140]	valid_0's auc: 0.671582
[6150]	valid_0's auc: 0.671592
[6160]	valid_0's auc: 0.671588
[6170]	valid_0's auc: 0.671592
[6180]	valid_0's auc: 0.6716
[6190]	valid_0's auc: 0.671607
[6200]	valid_0's auc: 0.671611
[6210]	valid_0's auc: 0.67161
[6220]	valid_0's auc: 0.671614
[6230]	valid_0's auc: 0.671622
[6240]	valid_0's auc: 0.671641
[6250]	valid_0's auc: 0.671645
[6260]	valid_0's auc: 0.671655
[6270]	valid_0's auc: 0.671671
[6280]	valid_0's auc: 0.67167
[6290]	valid_0's auc: 0.671671
[6300]	valid_0's auc: 0.671673
[6310]	valid_0's auc: 0.671675
[6320]	valid_0's auc: 0.671681
[6330]	valid_0's auc: 0.671683
[6340]	valid_0's auc: 0.671686
[6350]	valid_0's auc: 0.671697
[6360]	valid_0's auc: 0.671693
[6370]	valid_0's auc: 0.671705
[6380]	valid_0's auc: 0.671709
[6390]	valid_0's auc: 0.671708
[6400]	valid_0's auc: 0.671741
[6410]	valid_0's auc: 0.671746
[6420]	valid_0's auc: 0.671756
[6430]	valid_0's auc: 0.671763
[6440]	valid_0's auc: 0.671771
[6450]	valid_0's auc: 0.671778
[6460]	valid_0's auc: 0.671792
[6470]	valid_0's auc: 0.671795
[6480]	valid_0's auc: 0.671818
[6490]	valid_0's auc: 0.671826
[6500]	valid_0's auc: 0.671833
[6510]	valid_0's auc: 0.671844
[6520]	valid_0's auc: 0.671846
[6530]	valid_0's auc: 0.67185
[6540]	valid_0's auc: 0.671847
[6550]	valid_0's auc: 0.671867
[6560]	valid_0's auc: 0.671874
[6570]	valid_0's auc: 0.671881
[6580]	valid_0's auc: 0.671885
[6590]	valid_0's auc: 0.671892
[6600]	valid_0's auc: 0.671894
[6610]	valid_0's auc: 0.671899
[6620]	valid_0's auc: 0.671908
[6630]	valid_0's auc: 0.671912
[6640]	valid_0's auc: 0.671916
[6650]	valid_0's auc: 0.671911
[6660]	valid_0's auc: 0.671913
[6670]	valid_0's auc: 0.671913
[6680]	valid_0's auc: 0.671915
[6690]	valid_0's auc: 0.671924
[6700]	valid_0's auc: 0.671923
[6710]	valid_0's auc: 0.671927
[6720]	valid_0's auc: 0.671934
[6730]	valid_0's auc: 0.671939
[6740]	valid_0's auc: 0.67194
[6750]	valid_0's auc: 0.671942
[6760]	valid_0's auc: 0.671943
[6770]	valid_0's auc: 0.671959
[6780]	valid_0's auc: 0.671973
[6790]	valid_0's auc: 0.671975
[6800]	valid_0's auc: 0.671976
[6810]	valid_0's auc: 0.671975
[6820]	valid_0's auc: 0.671981
[6830]	valid_0's auc: 0.671994
[6840]	valid_0's auc: 0.672
[6850]	valid_0's auc: 0.672008
[6860]	valid_0's auc: 0.672013
[6870]	valid_0's auc: 0.672023
[6880]	valid_0's auc: 0.672025
[6890]	valid_0's auc: 0.672031
[6900]	valid_0's auc: 0.672038
[6910]	valid_0's auc: 0.67204
[6920]	valid_0's auc: 0.672046
[6930]	valid_0's auc: 0.672049
[6940]	valid_0's auc: 0.672047
[6950]	valid_0's auc: 0.672049
[6960]	valid_0's auc: 0.672046
[6970]	valid_0's auc: 0.672046
[6980]	valid_0's auc: 0.672046
[6990]	valid_0's auc: 0.67205
[7000]	valid_0's auc: 0.672051
[7010]	valid_0's auc: 0.672052
[7020]	valid_0's auc: 0.672055
[7030]	valid_0's auc: 0.67206
[7040]	valid_0's auc: 0.67206
[7050]	valid_0's auc: 0.672062
[7060]	valid_0's auc: 0.672063
[7070]	valid_0's auc: 0.672071
[7080]	valid_0's auc: 0.672079
[7090]	valid_0's auc: 0.672083
[7100]	valid_0's auc: 0.672085
[7110]	valid_0's auc: 0.672103
[7120]	valid_0's auc: 0.672101
[7130]	valid_0's auc: 0.672107
[7140]	valid_0's auc: 0.672119
[7150]	valid_0's auc: 0.672123
[7160]	valid_0's auc: 0.672122
[7170]	valid_0's auc: 0.672124
[7180]	valid_0's auc: 0.672126
[7190]	valid_0's auc: 0.67213
[7200]	valid_0's auc: 0.67213
[7210]	valid_0's auc: 0.672128
[7220]	valid_0's auc: 0.672123
[7230]	valid_0's auc: 0.672127
[7240]	valid_0's auc: 0.672132
[7250]	valid_0's auc: 0.672137
[7260]	valid_0's auc: 0.672143
[7270]	valid_0's auc: 0.67214
[7280]	valid_0's auc: 0.672148
[7290]	valid_0's auc: 0.67215
[7300]	valid_0's auc: 0.67216
[7310]	valid_0's auc: 0.672162
[7320]	valid_0's auc: 0.672169
[7330]	valid_0's auc: 0.672169
[7340]	valid_0's auc: 0.672162
[7350]	valid_0's auc: 0.672169
[7360]	valid_0's auc: 0.672174
[7370]	valid_0's auc: 0.672181
[7380]	valid_0's auc: 0.672185
[7390]	valid_0's auc: 0.672188
[7400]	valid_0's auc: 0.672191
[7410]	valid_0's auc: 0.672203
[7420]	valid_0's auc: 0.672196
[7430]	valid_0's auc: 0.672198
[7440]	valid_0's auc: 0.672199
[7450]	valid_0's auc: 0.672205
[7460]	valid_0's auc: 0.672209
[7470]	valid_0's auc: 0.672217
[7480]	valid_0's auc: 0.672223
[7490]	valid_0's auc: 0.672217
[7500]	valid_0's auc: 0.67222
[7510]	valid_0's auc: 0.672218
[7520]	valid_0's auc: 0.672225
[7530]	valid_0's auc: 0.672229
[7540]	valid_0's auc: 0.672229
[7550]	valid_0's auc: 0.672231
[7560]	valid_0's auc: 0.672235
[7570]	valid_0's auc: 0.672242
[7580]	valid_0's auc: 0.672249
[7590]	valid_0's auc: 0.672247
[7600]	valid_0's auc: 0.672254
[7610]	valid_0's auc: 0.672257
[7620]	valid_0's auc: 0.672257
[7630]	valid_0's auc: 0.672259
[7640]	valid_0's auc: 0.672265
[7650]	valid_0's auc: 0.672265
[7660]	valid_0's auc: 0.672268
[7670]	valid_0's auc: 0.672268
[7680]	valid_0's auc: 0.672273
[7690]	valid_0's auc: 0.672277
[7700]	valid_0's auc: 0.672278
[7710]	valid_0's auc: 0.672283
[7720]	valid_0's auc: 0.672285
[7730]	valid_0's auc: 0.672291
[7740]	valid_0's auc: 0.672289
[7750]	valid_0's auc: 0.672296
[7760]	valid_0's auc: 0.672298
[7770]	valid_0's auc: 0.672295
[7780]	valid_0's auc: 0.672301
[7790]	valid_0's auc: 0.672304
[7800]	valid_0's auc: 0.672311
[7810]	valid_0's auc: 0.672319
[7820]	valid_0's auc: 0.67231
[7830]	valid_0's auc: 0.672314
[7840]	valid_0's auc: 0.672321
[7850]	valid_0's auc: 0.672326
[7860]	valid_0's auc: 0.672323
[7870]	valid_0's auc: 0.672318
[7880]	valid_0's auc: 0.672318
[7890]	valid_0's auc: 0.67232
[7900]	valid_0's auc: 0.672316
[7910]	valid_0's auc: 0.67232
[7920]	valid_0's auc: 0.67232
[7930]	valid_0's auc: 0.672321
[7940]	valid_0's auc: 0.672317
[7950]	valid_0's auc: 0.672317
[7960]	valid_0's auc: 0.672304
[7970]	valid_0's auc: 0.672303
[7980]	valid_0's auc: 0.672304
[7990]	valid_0's auc: 0.67231
[8000]	valid_0's auc: 0.672316
[8010]	valid_0's auc: 0.672324
[8020]	valid_0's auc: 0.672325
[8030]	valid_0's auc: 0.672328
[8040]	valid_0's auc: 0.672332
[8050]	valid_0's auc: 0.672334
[8060]	valid_0's auc: 0.67233
[8070]	valid_0's auc: 0.672332
[8080]	valid_0's auc: 0.672334
[8090]	valid_0's auc: 0.67233
[8100]	valid_0's auc: 0.672332
[8110]	valid_0's auc: 0.672338
[8120]	valid_0's auc: 0.672339
[8130]	valid_0's auc: 0.672345
[8140]	valid_0's auc: 0.672347
[8150]	valid_0's auc: 0.672343
[8160]	valid_0's auc: 0.672342
[8170]	valid_0's auc: 0.672347
[8180]	valid_0's auc: 0.672348
[8190]	valid_0's auc: 0.67235
[8200]	valid_0's auc: 0.672352
[8210]	valid_0's auc: 0.672361
[8220]	valid_0's auc: 0.672362
[8230]	valid_0's auc: 0.672362
[8240]	valid_0's auc: 0.672365
[8250]	valid_0's auc: 0.67237
[8260]	valid_0's auc: 0.672374
[8270]	valid_0's auc: 0.672374
[8280]	valid_0's auc: 0.672382
[8290]	valid_0's auc: 0.672382
[8300]	valid_0's auc: 0.672386
[8310]	valid_0's auc: 0.672393
[8320]	valid_0's auc: 0.672398
[8330]	valid_0's auc: 0.672403
[8340]	valid_0's auc: 0.672413
[8350]	valid_0's auc: 0.672416
[8360]	valid_0's auc: 0.672415
[8370]	valid_0's auc: 0.672413
[8380]	valid_0's auc: 0.672418
[8390]	valid_0's auc: 0.67242
[8400]	valid_0's auc: 0.67242
[8410]	valid_0's auc: 0.672423
[8420]	valid_0's auc: 0.672422
[8430]	valid_0's auc: 0.672425
[8440]	valid_0's auc: 0.67243
[8450]	valid_0's auc: 0.672428
[8460]	valid_0's auc: 0.672434
[8470]	valid_0's auc: 0.672433
[8480]	valid_0's auc: 0.672433
[8490]	valid_0's auc: 0.672446
[8500]	valid_0's auc: 0.672447
[8510]	valid_0's auc: 0.672448
[8520]	valid_0's auc: 0.67245
[8530]	valid_0's auc: 0.672453
[8540]	valid_0's auc: 0.672454
[8550]	valid_0's auc: 0.672454
[8560]	valid_0's auc: 0.672453
[8570]	valid_0's auc: 0.672457
[8580]	valid_0's auc: 0.672459
[8590]	valid_0's auc: 0.672466
[8600]	valid_0's auc: 0.672465
[8610]	valid_0's auc: 0.672467
[8620]	valid_0's auc: 0.672472
[8630]	valid_0's auc: 0.67248
[8640]	valid_0's auc: 0.672481
[8650]	valid_0's auc: 0.672481
[8660]	valid_0's auc: 0.67249
[8670]	valid_0's auc: 0.67249
[8680]	valid_0's auc: 0.672495
[8690]	valid_0's auc: 0.672494
[8700]	valid_0's auc: 0.672491
[8710]	valid_0's auc: 0.672494
[8720]	valid_0's auc: 0.672497
[8730]	valid_0's auc: 0.672495
[8740]	valid_0's auc: 0.672506
[8750]	valid_0's auc: 0.672507
[8760]	valid_0's auc: 0.672508
[8770]	valid_0's auc: 0.672505
[8780]	valid_0's auc: 0.672505
[8790]	valid_0's auc: 0.672503
[8800]	valid_0's auc: 0.672505
[8810]	valid_0's auc: 0.672508
[8820]	valid_0's auc: 0.67251
[8830]	valid_0's auc: 0.672516
[8840]	valid_0's auc: 0.672517
[8850]	valid_0's auc: 0.672517
[8860]	valid_0's auc: 0.672515
[8870]	valid_0's auc: 0.672523
[8880]	valid_0's auc: 0.672524
[8890]	valid_0's auc: 0.672527
[8900]	valid_0's auc: 0.672528
[8910]	valid_0's auc: 0.672526
[8920]	valid_0's auc: 0.672531
[8930]	valid_0's auc: 0.672531
[8940]	valid_0's auc: 0.672533
[8950]	valid_0's auc: 0.67254
[8960]	valid_0's auc: 0.672542
[8970]	valid_0's auc: 0.672541
[8980]	valid_0's auc: 0.672543
[8990]	valid_0's auc: 0.672547
[9000]	valid_0's auc: 0.672546
[9010]	valid_0's auc: 0.672543
[9020]	valid_0's auc: 0.672544
[9030]	valid_0's auc: 0.67255
[9040]	valid_0's auc: 0.672556
[9050]	valid_0's auc: 0.672556
[9060]	valid_0's auc: 0.672558
[9070]	valid_0's auc: 0.67256
[9080]	valid_0's auc: 0.672559
[9090]	valid_0's auc: 0.672571
[9100]	valid_0's auc: 0.672577
[9110]	valid_0's auc: 0.672581
[9120]	valid_0's auc: 0.672581
[9130]	valid_0's auc: 0.672584
[9140]	valid_0's auc: 0.672587
[9150]	valid_0's auc: 0.672591
[9160]	valid_0's auc: 0.67259
[9170]	valid_0's auc: 0.672588
[9180]	valid_0's auc: 0.672588
[9190]	valid_0's auc: 0.672575
[9200]	valid_0's auc: 0.672576
[9210]	valid_0's auc: 0.672579
[9220]	valid_0's auc: 0.672575
[9230]	valid_0's auc: 0.672576
[9240]	valid_0's auc: 0.672582
[9250]	valid_0's auc: 0.672582
[9260]	valid_0's auc: 0.672583
[9270]	valid_0's auc: 0.672587
[9280]	valid_0's auc: 0.67259
[9290]	valid_0's auc: 0.672588
[9300]	valid_0's auc: 0.672593
[9310]	valid_0's auc: 0.672591
[9320]	valid_0's auc: 0.672578
[9330]	valid_0's auc: 0.672579
[9340]	valid_0's auc: 0.672576
[9350]	valid_0's auc: 0.672582
[9360]	valid_0's auc: 0.672586
[9370]	valid_0's auc: 0.672585
[9380]	valid_0's auc: 0.67259
[9390]	valid_0's auc: 0.672589
[9400]	valid_0's auc: 0.672593
[9410]	valid_0's auc: 0.672593
[9420]	valid_0's auc: 0.672596
[9430]	valid_0's auc: 0.672596
[9440]	valid_0's auc: 0.672598
[9450]	valid_0's auc: 0.672601
[9460]	valid_0's auc: 0.672601
[9470]	valid_0's auc: 0.672602
[9480]	valid_0's auc: 0.672602
[9490]	valid_0's auc: 0.672603
[9500]	valid_0's auc: 0.672608
[9510]	valid_0's auc: 0.672606
[9520]	valid_0's auc: 0.67261
[9530]	valid_0's auc: 0.672612
[9540]	valid_0's auc: 0.672614
[9550]	valid_0's auc: 0.672615
[9560]	valid_0's auc: 0.672625
[9570]	valid_0's auc: 0.672631
[9580]	valid_0's auc: 0.672634
[9590]	valid_0's auc: 0.672631
[9600]	valid_0's auc: 0.672638
[9610]	valid_0's auc: 0.67264
[9620]	valid_0's auc: 0.672641
[9630]	valid_0's auc: 0.672641
[9640]	valid_0's auc: 0.672644
[9650]	valid_0's auc: 0.672647
[9660]	valid_0's auc: 0.672642
[9670]	valid_0's auc: 0.672649
[9680]	valid_0's auc: 0.672643
[9690]	valid_0's auc: 0.672647
[9700]	valid_0's auc: 0.672649
[9710]	valid_0's auc: 0.672654
[9720]	valid_0's auc: 0.672652
[9730]	valid_0's auc: 0.672656
[9740]	valid_0's auc: 0.672656
[9750]	valid_0's auc: 0.67266
[9760]	valid_0's auc: 0.672662
[9770]	valid_0's auc: 0.672666
[9780]	valid_0's auc: 0.672663
[9790]	valid_0's auc: 0.672665
[9800]	valid_0's auc: 0.672668
[9810]	valid_0's auc: 0.67267
[9820]	valid_0's auc: 0.672672
[9830]	valid_0's auc: 0.672675
[9840]	valid_0's auc: 0.672677
[9850]	valid_0's auc: 0.672675
[9860]	valid_0's auc: 0.672675
[9870]	valid_0's auc: 0.672674
[9880]	valid_0's auc: 0.672674
[9890]	valid_0's auc: 0.672682
[9900]	valid_0's auc: 0.672691
[9910]	valid_0's auc: 0.672693
[9920]	valid_0's auc: 0.672692
[9930]	valid_0's auc: 0.672691
[9940]	valid_0's auc: 0.672694
[9950]	valid_0's auc: 0.672692
[9960]	valid_0's auc: 0.672697
[9970]	valid_0's auc: 0.672698
[9980]	valid_0's auc: 0.672704
[9990]	valid_0's auc: 0.672706
[10000]	valid_0's auc: 0.672707
[10010]	valid_0's auc: 0.672714
[10020]	valid_0's auc: 0.672716
[10030]	valid_0's auc: 0.672721
[10040]	valid_0's auc: 0.672723
[10050]	valid_0's auc: 0.672724
[10060]	valid_0's auc: 0.672726
[10070]	valid_0's auc: 0.67273
[10080]	valid_0's auc: 0.672728
[10090]	valid_0's auc: 0.672725
[10100]	valid_0's auc: 0.672725
[10110]	valid_0's auc: 0.672727
[10120]	valid_0's auc: 0.672728
[10130]	valid_0's auc: 0.672727
[10140]	valid_0's auc: 0.672724
[10150]	valid_0's auc: 0.672725
[10160]	valid_0's auc: 0.672724
[10170]	valid_0's auc: 0.672725
[10180]	valid_0's auc: 0.67273
[10190]	valid_0's auc: 0.672733
[10200]	valid_0's auc: 0.672736
[10210]	valid_0's auc: 0.672743
[10220]	valid_0's auc: 0.672744
[10230]	valid_0's auc: 0.672745
[10240]	valid_0's auc: 0.672748
[10250]	valid_0's auc: 0.672749
[10260]	valid_0's auc: 0.672745
[10270]	valid_0's auc: 0.672752
[10280]	valid_0's auc: 0.672756
[10290]	valid_0's auc: 0.672759
[10300]	valid_0's auc: 0.672766
[10310]	valid_0's auc: 0.672767
[10320]	valid_0's auc: 0.672766
[10330]	valid_0's auc: 0.672771
[10340]	valid_0's auc: 0.67277
[10350]	valid_0's auc: 0.672766
[10360]	valid_0's auc: 0.672772
[10370]	valid_0's auc: 0.67277
[10380]	valid_0's auc: 0.672769
[10390]	valid_0's auc: 0.672766
[10400]	valid_0's auc: 0.672768
[10410]	valid_0's auc: 0.67277
[10420]	valid_0's auc: 0.672773
[10430]	valid_0's auc: 0.672769
[10440]	valid_0's auc: 0.67277
[10450]	valid_0's auc: 0.672772
[10460]	valid_0's auc: 0.672768
[10470]	valid_0's auc: 0.67277
[10480]	valid_0's auc: 0.672771
[10490]	valid_0's auc: 0.672775
[10500]	valid_0's auc: 0.672776
[10510]	valid_0's auc: 0.672781
[10520]	valid_0's auc: 0.672783
[10530]	valid_0's auc: 0.672786
[10540]	valid_0's auc: 0.67279
[10550]	valid_0's auc: 0.672793
[10560]	valid_0's auc: 0.67279
[10570]	valid_0's auc: 0.672798
[10580]	valid_0's auc: 0.672797
[10590]	valid_0's auc: 0.672791
[10600]	valid_0's auc: 0.672789
[10610]	valid_0's auc: 0.672793
[10620]	valid_0's auc: 0.672801
[10630]	valid_0's auc: 0.672799
[10640]	valid_0's auc: 0.672799
[10650]	valid_0's auc: 0.672807
[10660]	valid_0's auc: 0.672803
[10670]	valid_0's auc: 0.672803
[10680]	valid_0's auc: 0.672811
[10690]	valid_0's auc: 0.672812
[10700]	valid_0's auc: 0.672811
[10710]	valid_0's auc: 0.672807
[10720]	valid_0's auc: 0.672803
[10730]	valid_0's auc: 0.672792
[10740]	valid_0's auc: 0.672794
[10750]	valid_0's auc: 0.6728
[10760]	valid_0's auc: 0.6728
[10770]	valid_0's auc: 0.672799
[10780]	valid_0's auc: 0.672801
[10790]	valid_0's auc: 0.672803
[10800]	valid_0's auc: 0.672801
[10810]	valid_0's auc: 0.672801
[10820]	valid_0's auc: 0.672804
[10830]	valid_0's auc: 0.672803
[10840]	valid_0's auc: 0.672821
[10850]	valid_0's auc: 0.672825
[10860]	valid_0's auc: 0.672829
[10870]	valid_0's auc: 0.672826
[10880]	valid_0's auc: 0.672824
[10890]	valid_0's auc: 0.672825
[10900]	valid_0's auc: 0.672828
[10910]	valid_0's auc: 0.672832
[10920]	valid_0's auc: 0.672831
[10930]	valid_0's auc: 0.672839
[10940]	valid_0's auc: 0.672844
[10950]	valid_0's auc: 0.672842
[10960]	valid_0's auc: 0.672849
[10970]	valid_0's auc: 0.672843
[10980]	valid_0's auc: 0.67285
[10990]	valid_0's auc: 0.672848
[11000]	valid_0's auc: 0.672848
[11010]	valid_0's auc: 0.672848
[11020]	valid_0's auc: 0.67284
[11030]	valid_0's auc: 0.672843
[11040]	valid_0's auc: 0.672841
[11050]	valid_0's auc: 0.672843
[11060]	valid_0's auc: 0.672845
[11070]	valid_0's auc: 0.672847
[11080]	valid_0's auc: 0.672847
[11090]	valid_0's auc: 0.672842
[11100]	valid_0's auc: 0.67284
[11110]	valid_0's auc: 0.672844
[11120]	valid_0's auc: 0.672843
[11130]	valid_0's auc: 0.67285
[11140]	valid_0's auc: 0.672852
[11150]	valid_0's auc: 0.672852
[11160]	valid_0's auc: 0.672852
[11170]	valid_0's auc: 0.672851
[11180]	valid_0's auc: 0.672847
[11190]	valid_0's auc: 0.672848
[11200]	valid_0's auc: 0.672845
[11210]	valid_0's auc: 0.672842
[11220]	valid_0's auc: 0.672843
[11230]	valid_0's auc: 0.672841
[11240]	valid_0's auc: 0.672844
[11250]	valid_0's auc: 0.672848
[11260]	valid_0's auc: 0.672847
[11270]	valid_0's auc: 0.672841
[11280]	valid_0's auc: 0.672844
[11290]	valid_0's auc: 0.672846
[11300]	valid_0's auc: 0.672842
[11310]	valid_0's auc: 0.672842
[11320]	valid_0's auc: 0.672836
[11330]	valid_0's auc: 0.672843
[11340]	valid_0's auc: 0.672846
[11350]	valid_0's auc: 0.672846
[11360]	valid_0's auc: 0.672844
[11370]	valid_0's auc: 0.672849
Early stopping, best iteration is:
[11171]	valid_0's auc: 0.672854
best score: 0.672853732854
best iteration: 11171

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

[timer]: complete in 137m 5s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------

max_bin              : 255
feature_fraction_seed : 2
learning_rate        : 0.02
lambda_l1            : 0
bagging_freq         : 2
feature_fraction     : 0.9
bagging_seed         : 2
boosting             : gbdt
bagging_fraction     : 0.9
num_leaves           : 511
max_depth            : 11
lambda_l2            : 0.5

Training until validation scores don't improve for 200 rounds.
[10]	valid_0's auc: 0.628404
[20]	valid_0's auc: 0.630477
[30]	valid_0's auc: 0.632985
[40]	valid_0's auc: 0.63464
[50]	valid_0's auc: 0.635849
[60]	valid_0's auc: 0.636543
[70]	valid_0's auc: 0.637918
[80]	valid_0's auc: 0.639111
[90]	valid_0's auc: 0.640105
[100]	valid_0's auc: 0.641135
[110]	valid_0's auc: 0.64204
[120]	valid_0's auc: 0.642795
[130]	valid_0's auc: 0.643467
[140]	valid_0's auc: 0.644264
[150]	valid_0's auc: 0.645074
[160]	valid_0's auc: 0.645635
[170]	valid_0's auc: 0.646146
[180]	valid_0's auc: 0.646531
[190]	valid_0's auc: 0.646938
[200]	valid_0's auc: 0.647323
[210]	valid_0's auc: 0.647615
[220]	valid_0's auc: 0.64796
[230]	valid_0's auc: 0.648193
[240]	valid_0's auc: 0.648494
[250]	valid_0's auc: 0.648756
[260]	valid_0's auc: 0.649031
[270]	valid_0's auc: 0.64931
[280]	valid_0's auc: 0.649558
[290]	valid_0's auc: 0.649663
[300]	valid_0's auc: 0.649885
[310]	valid_0's auc: 0.650125
[320]	valid_0's auc: 0.650341
[330]	valid_0's auc: 0.650579
[340]	valid_0's auc: 0.650763
[350]	valid_0's auc: 0.650998
[360]	valid_0's auc: 0.651207
[370]	valid_0's auc: 0.651384
[380]	valid_0's auc: 0.651613
[390]	valid_0's auc: 0.651768
[400]	valid_0's auc: 0.651954
[410]	valid_0's auc: 0.652131
[420]	valid_0's auc: 0.65233
[430]	valid_0's auc: 0.652485
[440]	valid_0's auc: 0.652645
[450]	valid_0's auc: 0.6528
[460]	valid_0's auc: 0.652979
[470]	valid_0's auc: 0.653164
[480]	valid_0's auc: 0.653368
[490]	valid_0's auc: 0.653537
[500]	valid_0's auc: 0.653674
[510]	valid_0's auc: 0.653835
[520]	valid_0's auc: 0.653963
[530]	valid_0's auc: 0.654095
[540]	valid_0's auc: 0.654237
[550]	valid_0's auc: 0.654384
[560]	valid_0's auc: 0.654525
[570]	valid_0's auc: 0.654661
[580]	valid_0's auc: 0.654803
[590]	valid_0's auc: 0.654953
[600]	valid_0's auc: 0.65508
[610]	valid_0's auc: 0.655173
[620]	valid_0's auc: 0.655291
[630]	valid_0's auc: 0.655399
[640]	valid_0's auc: 0.655545
[650]	valid_0's auc: 0.655668
[660]	valid_0's auc: 0.655768
[670]	valid_0's auc: 0.6559
[680]	valid_0's auc: 0.65604
[690]	valid_0's auc: 0.656134
[700]	valid_0's auc: 0.656296
[710]	valid_0's auc: 0.656407
[720]	valid_0's auc: 0.65653
[730]	valid_0's auc: 0.656618
[740]	valid_0's auc: 0.656727
[750]	valid_0's auc: 0.656859
[760]	valid_0's auc: 0.656978
[770]	valid_0's auc: 0.657088
[780]	valid_0's auc: 0.65719
[790]	valid_0's auc: 0.657265
[800]	valid_0's auc: 0.65739
[810]	valid_0's auc: 0.657474
[820]	valid_0's auc: 0.657598
[830]	valid_0's auc: 0.65771
[840]	valid_0's auc: 0.657804
[850]	valid_0's auc: 0.657912
[860]	valid_0's auc: 0.657976
[870]	valid_0's auc: 0.658092
[880]	valid_0's auc: 0.65821
[890]	valid_0's auc: 0.658316
[900]	valid_0's auc: 0.658398
[910]	valid_0's auc: 0.658457
[920]	valid_0's auc: 0.658542
[930]	valid_0's auc: 0.658648
[940]	valid_0's auc: 0.658742
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/gbdt_random_V1003.py", line 135, in <module>
    verbose_eval=10,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 199, in train
    booster.update(fobj=fobj)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1507, in update
    ctypes.byref(is_finished)))
KeyboardInterrupt
'''