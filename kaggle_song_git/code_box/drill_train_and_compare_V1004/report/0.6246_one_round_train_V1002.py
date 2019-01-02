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

num_boost_round = 5000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.05
num_leaves = 511
max_depth = 10

# max_bin = 255
lambda_l1 = 0
lambda_l2 = 0.1


bagging_fraction = 0.8
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.8
feature_fraction_seed = 2

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

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print('Our guest selection:')
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

t = len(Y_tr)
t1 = sum(Y_tr)
t0 = t - t1
print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
t = len(Y_val)
t1 = sum(Y_val)
t0 = t - t1
print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
print()
print()

train_set = lgb.Dataset(X_tr, Y_tr)
val_set = lgb.Dataset(X_val, Y_val)
# train_set.max_bin = max_bin
# val_set.max_bin = max_bin

del X_tr, Y_tr, X_val, Y_val

params['metric'] = 'auc'
params['verbose'] = -1
params['objective'] = 'binary'

print('Training...')

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
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1004/0.6246_one_round_train_V1002.py
What we got:
target                          uint8
membership_days                 int64
bd_log10                      float64
expiration_month_log10        float64
IMC_expiration_month_log10    float64
bd_fixed_log10                float64
age_guess_log10               float64
bd_range_log10                float64
age_guess_range_log10         float64
bd_fixed_range_log10          float64
IMC_bd_log10                  float64
IMC_bd_fixed_log10            float64
IMC_age_guess_log10           float64
IMC_bd_range_log10            float64
IMC_bd_fixed_range_log10      float64
IMC_age_guess_range_log10     float64
IMC_membership_days_log10     float64
song_year                       int64
ISC_genre_ids                   int64
ISC_top1_in_song                int64
ISC_top2_in_song                int64
ISC_top3_in_song                int64
ISCZ_artist_name                int64
ISC_composer                    int64
ISCZ_lyricist                   int64
ISC_language                    int64
ISCZ_rc                         int64
ISCZ_isrc_rest                  int64
ISC_song_year                   int64
ISCZ_song_year                  int64
song_length_log10             float64
ISCZ_genre_ids_log10          float64
ISC_artist_name_log10         float64
ISCZ_composer_log10           float64
ISC_lyricist_log10            float64
ISC_name_log10                float64
ISCZ_name_ln                  float64
ISC_song_country_ln           float64
ISCZ_song_country_log10       float64
ISC_rc_ln                     float64
ISC_isrc_rest_log10           float64
dtype: object
number of rows: 7377418
number of columns: 41

Our guest selection:
target                          uint8
membership_days                 int64
bd_log10                      float64
expiration_month_log10        float64
IMC_expiration_month_log10    float64
bd_fixed_log10                float64
age_guess_log10               float64
bd_range_log10                float64
age_guess_range_log10         float64
bd_fixed_range_log10          float64
IMC_bd_log10                  float64
IMC_bd_fixed_log10            float64
IMC_age_guess_log10           float64
IMC_bd_range_log10            float64
IMC_bd_fixed_range_log10      float64
IMC_age_guess_range_log10     float64
IMC_membership_days_log10     float64
song_year                       int64
ISC_genre_ids                   int64
ISC_top1_in_song                int64
ISC_top2_in_song                int64
ISC_top3_in_song                int64
ISCZ_artist_name                int64
ISC_composer                    int64
ISCZ_lyricist                   int64
ISC_language                    int64
ISCZ_rc                         int64
ISCZ_isrc_rest                  int64
ISC_song_year                   int64
ISCZ_song_year                  int64
song_length_log10             float64
ISCZ_genre_ids_log10          float64
ISC_artist_name_log10         float64
ISCZ_composer_log10           float64
ISC_lyricist_log10            float64
ISC_name_log10                float64
ISCZ_name_ln                  float64
ISC_song_country_ln           float64
ISCZ_song_country_log10       float64
ISC_rc_ln                     float64
ISC_isrc_rest_log10           float64
dtype: object
number of columns: 41

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.571379
[20]	valid_0's auc: 0.573723
[30]	valid_0's auc: 0.577125
[40]	valid_0's auc: 0.579054
[50]	valid_0's auc: 0.582412
[60]	valid_0's auc: 0.584819
[70]	valid_0's auc: 0.586872
[80]	valid_0's auc: 0.588639
[90]	valid_0's auc: 0.590402
[100]	valid_0's auc: 0.591304
[110]	valid_0's auc: 0.592811
[120]	valid_0's auc: 0.593956
[130]	valid_0's auc: 0.595094
[140]	valid_0's auc: 0.596114
[150]	valid_0's auc: 0.597391
[160]	valid_0's auc: 0.598577
[170]	valid_0's auc: 0.599886
[180]	valid_0's auc: 0.600538
[190]	valid_0's auc: 0.600845
[200]	valid_0's auc: 0.601693
[210]	valid_0's auc: 0.602514
[220]	valid_0's auc: 0.603334
[230]	valid_0's auc: 0.603839
[240]	valid_0's auc: 0.604326
[250]	valid_0's auc: 0.604894
[260]	valid_0's auc: 0.605328
[270]	valid_0's auc: 0.6059
[280]	valid_0's auc: 0.606493
[290]	valid_0's auc: 0.606881
[300]	valid_0's auc: 0.607479
[310]	valid_0's auc: 0.607993
[320]	valid_0's auc: 0.60847
[330]	valid_0's auc: 0.6089
[340]	valid_0's auc: 0.609586
[350]	valid_0's auc: 0.610131
[360]	valid_0's auc: 0.610545
[370]	valid_0's auc: 0.611064
[380]	valid_0's auc: 0.61142
[390]	valid_0's auc: 0.611601
[400]	valid_0's auc: 0.611953
[410]	valid_0's auc: 0.612662
[420]	valid_0's auc: 0.612851
[430]	valid_0's auc: 0.613113
[440]	valid_0's auc: 0.613255
[450]	valid_0's auc: 0.613625
[460]	valid_0's auc: 0.613811
[470]	valid_0's auc: 0.614145
[480]	valid_0's auc: 0.61438
[490]	valid_0's auc: 0.614489
[500]	valid_0's auc: 0.61482
[510]	valid_0's auc: 0.61489
[520]	valid_0's auc: 0.615018
[530]	valid_0's auc: 0.615346
[540]	valid_0's auc: 0.615543
[550]	valid_0's auc: 0.615765
[560]	valid_0's auc: 0.615908
[570]	valid_0's auc: 0.615986
[580]	valid_0's auc: 0.616253
[590]	valid_0's auc: 0.616471
[600]	valid_0's auc: 0.616553
[610]	valid_0's auc: 0.616851
[620]	valid_0's auc: 0.616941
[630]	valid_0's auc: 0.617087
[640]	valid_0's auc: 0.617294
[650]	valid_0's auc: 0.617408
[660]	valid_0's auc: 0.617487
[670]	valid_0's auc: 0.617697
[680]	valid_0's auc: 0.617782
[690]	valid_0's auc: 0.617962
[700]	valid_0's auc: 0.618068
[710]	valid_0's auc: 0.618134
[720]	valid_0's auc: 0.618203
[730]	valid_0's auc: 0.618237
[740]	valid_0's auc: 0.618344
[750]	valid_0's auc: 0.618496
[760]	valid_0's auc: 0.618638
[770]	valid_0's auc: 0.618733
[780]	valid_0's auc: 0.618896
[790]	valid_0's auc: 0.6191
[800]	valid_0's auc: 0.619172
[810]	valid_0's auc: 0.619264
[820]	valid_0's auc: 0.61931
[830]	valid_0's auc: 0.619317
[840]	valid_0's auc: 0.619352
[850]	valid_0's auc: 0.619586
[860]	valid_0's auc: 0.619804
[870]	valid_0's auc: 0.619911
[880]	valid_0's auc: 0.620035
[890]	valid_0's auc: 0.620217
[900]	valid_0's auc: 0.620313
[910]	valid_0's auc: 0.620396
[920]	valid_0's auc: 0.620427
[930]	valid_0's auc: 0.620491
[940]	valid_0's auc: 0.62056
[950]	valid_0's auc: 0.620682
[960]	valid_0's auc: 0.620867
[970]	valid_0's auc: 0.620921
[980]	valid_0's auc: 0.620993
[990]	valid_0's auc: 0.621081
[1000]	valid_0's auc: 0.62109
[1010]	valid_0's auc: 0.621148
[1020]	valid_0's auc: 0.621258
[1030]	valid_0's auc: 0.621322
[1040]	valid_0's auc: 0.621354
[1050]	valid_0's auc: 0.621393
[1060]	valid_0's auc: 0.621459
[1070]	valid_0's auc: 0.621497
[1080]	valid_0's auc: 0.621567
[1090]	valid_0's auc: 0.621545
[1100]	valid_0's auc: 0.621646
[1110]	valid_0's auc: 0.621729
[1120]	valid_0's auc: 0.621823
[1130]	valid_0's auc: 0.621911
[1140]	valid_0's auc: 0.621939
[1150]	valid_0's auc: 0.62198
[1160]	valid_0's auc: 0.622016
[1170]	valid_0's auc: 0.622065
[1180]	valid_0's auc: 0.622095
[1190]	valid_0's auc: 0.62214
[1200]	valid_0's auc: 0.62217
[1210]	valid_0's auc: 0.622201
[1220]	valid_0's auc: 0.622204
[1230]	valid_0's auc: 0.622221
[1240]	valid_0's auc: 0.622313
[1250]	valid_0's auc: 0.622368
[1260]	valid_0's auc: 0.622382
[1270]	valid_0's auc: 0.622445
[1280]	valid_0's auc: 0.622483
[1290]	valid_0's auc: 0.622555
[1300]	valid_0's auc: 0.622558
[1310]	valid_0's auc: 0.622603
[1320]	valid_0's auc: 0.6227
[1330]	valid_0's auc: 0.622707
[1340]	valid_0's auc: 0.622732
[1350]	valid_0's auc: 0.622822
[1360]	valid_0's auc: 0.622829
[1370]	valid_0's auc: 0.622832
[1380]	valid_0's auc: 0.622884
[1390]	valid_0's auc: 0.622934
[1400]	valid_0's auc: 0.623003
[1410]	valid_0's auc: 0.622991
[1420]	valid_0's auc: 0.623032
[1430]	valid_0's auc: 0.623062
[1440]	valid_0's auc: 0.623068
[1450]	valid_0's auc: 0.623049
[1460]	valid_0's auc: 0.623078
[1470]	valid_0's auc: 0.62309
[1480]	valid_0's auc: 0.6231
[1490]	valid_0's auc: 0.623167
[1500]	valid_0's auc: 0.623205
[1510]	valid_0's auc: 0.623251
[1520]	valid_0's auc: 0.623307
[1530]	valid_0's auc: 0.623319
[1540]	valid_0's auc: 0.623347
[1550]	valid_0's auc: 0.623362
[1560]	valid_0's auc: 0.623345
[1570]	valid_0's auc: 0.623357
[1580]	valid_0's auc: 0.623398
[1590]	valid_0's auc: 0.623403
[1600]	valid_0's auc: 0.62339
[1610]	valid_0's auc: 0.623388
[1620]	valid_0's auc: 0.623463
[1630]	valid_0's auc: 0.623512
[1640]	valid_0's auc: 0.623552
[1650]	valid_0's auc: 0.623604
[1660]	valid_0's auc: 0.623691
[1670]	valid_0's auc: 0.62372
[1680]	valid_0's auc: 0.623822
[1690]	valid_0's auc: 0.623837
[1700]	valid_0's auc: 0.623891
[1710]	valid_0's auc: 0.623885
[1720]	valid_0's auc: 0.623957
[1730]	valid_0's auc: 0.62397
[1740]	valid_0's auc: 0.624015
[1750]	valid_0's auc: 0.62402
[1760]	valid_0's auc: 0.623987
[1770]	valid_0's auc: 0.624116
[1780]	valid_0's auc: 0.624102
[1790]	valid_0's auc: 0.624149
[1800]	valid_0's auc: 0.624173
[1810]	valid_0's auc: 0.624188
[1820]	valid_0's auc: 0.624211
[1830]	valid_0's auc: 0.624232
[1840]	valid_0's auc: 0.624243
[1850]	valid_0's auc: 0.624247
[1860]	valid_0's auc: 0.624274
[1870]	valid_0's auc: 0.624315
[1880]	valid_0's auc: 0.624343
[1890]	valid_0's auc: 0.624374
[1900]	valid_0's auc: 0.624378
[1910]	valid_0's auc: 0.624441
[1920]	valid_0's auc: 0.624445
[1930]	valid_0's auc: 0.624483
[1940]	valid_0's auc: 0.624507
[1950]	valid_0's auc: 0.624512
[1960]	valid_0's auc: 0.624553
[1970]	valid_0's auc: 0.624583
[1980]	valid_0's auc: 0.624589
[1990]	valid_0's auc: 0.62456
[2000]	valid_0's auc: 0.624605
[2010]	valid_0's auc: 0.624626
[2020]	valid_0's auc: 0.624555
[2030]	valid_0's auc: 0.624469
[2040]	valid_0's auc: 0.624485
[2050]	valid_0's auc: 0.624506
[2060]	valid_0's auc: 0.624537
Early stopping, best iteration is:
[2014]	valid_0's auc: 0.62464
best score: 0.62464028404
best iteration: 2014

[timer]: complete in 35m 30s

Process finished with exit code 0
'''
