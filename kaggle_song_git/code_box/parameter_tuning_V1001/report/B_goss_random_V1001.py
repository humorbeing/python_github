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
    boosting = b_s[3]
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/parameter_tuning_V1001/goss_random_V1001.py
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

max_depth            : -1
feature_fraction     : 0.8
boosting             : goss
lambda_l1            : 0
learning_rate        : 0.5
num_leaves           : 511
feature_fraction_seed : 2
lambda_l2            : 0

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.661875
[20]	valid_0's auc: 0.658524
[30]	valid_0's auc: 0.655582
[40]	valid_0's auc: 0.654137
[50]	valid_0's auc: 0.652881
Early stopping, best iteration is:
[9]	valid_0's auc: 0.661935
best score: 0.661935143681
best iteration: 9

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 0 complete in 11m 6s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 1

max_depth            : 10
feature_fraction     : 0.8
boosting             : goss
lambda_l1            : 0
learning_rate        : 0.1
num_leaves           : 1023
feature_fraction_seed : 2
lambda_l2            : 0

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.633014
[20]	valid_0's auc: 0.638179
[30]	valid_0's auc: 0.642027
[40]	valid_0's auc: 0.644344
[50]	valid_0's auc: 0.646263
[60]	valid_0's auc: 0.647978
[70]	valid_0's auc: 0.649315
[80]	valid_0's auc: 0.650195
[90]	valid_0's auc: 0.651032
[100]	valid_0's auc: 0.651741
[110]	valid_0's auc: 0.652512
[120]	valid_0's auc: 0.65326
[130]	valid_0's auc: 0.653915
[140]	valid_0's auc: 0.654493
[150]	valid_0's auc: 0.654983
[160]	valid_0's auc: 0.655507
[170]	valid_0's auc: 0.655842
[180]	valid_0's auc: 0.656047
[190]	valid_0's auc: 0.656348
[200]	valid_0's auc: 0.656518
[210]	valid_0's auc: 0.656784
[220]	valid_0's auc: 0.657016
[230]	valid_0's auc: 0.657266
[240]	valid_0's auc: 0.657382
[250]	valid_0's auc: 0.657788
[260]	valid_0's auc: 0.657857
[270]	valid_0's auc: 0.657974
[280]	valid_0's auc: 0.658062
[290]	valid_0's auc: 0.658196
[300]	valid_0's auc: 0.658336
[310]	valid_0's auc: 0.658517
[320]	valid_0's auc: 0.658603
[330]	valid_0's auc: 0.658694
[340]	valid_0's auc: 0.658749
[350]	valid_0's auc: 0.658811
[360]	valid_0's auc: 0.658878
[370]	valid_0's auc: 0.658941
[380]	valid_0's auc: 0.659103
[390]	valid_0's auc: 0.658962
[400]	valid_0's auc: 0.658949
[410]	valid_0's auc: 0.659057
[420]	valid_0's auc: 0.659017
[430]	valid_0's auc: 0.659167
[440]	valid_0's auc: 0.659182
[450]	valid_0's auc: 0.659246
[460]	valid_0's auc: 0.65932
[470]	valid_0's auc: 0.659414
[480]	valid_0's auc: 0.659446
[490]	valid_0's auc: 0.659462
[500]	valid_0's auc: 0.659451
[510]	valid_0's auc: 0.659577
[520]	valid_0's auc: 0.65962
[530]	valid_0's auc: 0.659671
[540]	valid_0's auc: 0.659712
[550]	valid_0's auc: 0.659779
[560]	valid_0's auc: 0.659818
[570]	valid_0's auc: 0.659825
[580]	valid_0's auc: 0.659831
[590]	valid_0's auc: 0.659817
[600]	valid_0's auc: 0.659961
[610]	valid_0's auc: 0.660033
[620]	valid_0's auc: 0.659937
[630]	valid_0's auc: 0.659846
[640]	valid_0's auc: 0.659936
[650]	valid_0's auc: 0.659948
Early stopping, best iteration is:
[607]	valid_0's auc: 0.660056
best score: 0.66005646867
best iteration: 607

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 1 complete in 7m 25s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 2

max_depth            : 11
feature_fraction     : 0.8
boosting             : goss
lambda_l1            : 0
learning_rate        : 0.02
num_leaves           : 511
feature_fraction_seed : 2
lambda_l2            : 0

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.628728
[20]	valid_0's auc: 0.631438
[30]	valid_0's auc: 0.632674
[40]	valid_0's auc: 0.634405
[50]	valid_0's auc: 0.635723
[60]	valid_0's auc: 0.63829
[70]	valid_0's auc: 0.640263
[80]	valid_0's auc: 0.641399
[90]	valid_0's auc: 0.642533
[100]	valid_0's auc: 0.64343
[110]	valid_0's auc: 0.644209
[120]	valid_0's auc: 0.64475
[130]	valid_0's auc: 0.645352
[140]	valid_0's auc: 0.64614
[150]	valid_0's auc: 0.646863
[160]	valid_0's auc: 0.647476
[170]	valid_0's auc: 0.648045
[180]	valid_0's auc: 0.648543
[190]	valid_0's auc: 0.648881
[200]	valid_0's auc: 0.649291
[210]	valid_0's auc: 0.649561
[220]	valid_0's auc: 0.649892
[230]	valid_0's auc: 0.650214
[240]	valid_0's auc: 0.650542
[250]	valid_0's auc: 0.650855
[260]	valid_0's auc: 0.651097
[270]	valid_0's auc: 0.651308
[280]	valid_0's auc: 0.651543
[290]	valid_0's auc: 0.651769
[300]	valid_0's auc: 0.65204
[310]	valid_0's auc: 0.652242
[320]	valid_0's auc: 0.652444
[330]	valid_0's auc: 0.652676
[340]	valid_0's auc: 0.652874
[350]	valid_0's auc: 0.653036
[360]	valid_0's auc: 0.653281
[370]	valid_0's auc: 0.653497
[380]	valid_0's auc: 0.653704
[390]	valid_0's auc: 0.653926
[400]	valid_0's auc: 0.654106
[410]	valid_0's auc: 0.654326
[420]	valid_0's auc: 0.654505
[430]	valid_0's auc: 0.654709
[440]	valid_0's auc: 0.654866
[450]	valid_0's auc: 0.655045
[460]	valid_0's auc: 0.655223
[470]	valid_0's auc: 0.655361
[480]	valid_0's auc: 0.6555
[490]	valid_0's auc: 0.655615
[500]	valid_0's auc: 0.655753
[510]	valid_0's auc: 0.655874
[520]	valid_0's auc: 0.655986
[530]	valid_0's auc: 0.656103
[540]	valid_0's auc: 0.656203
[550]	valid_0's auc: 0.656294
[560]	valid_0's auc: 0.656421
[570]	valid_0's auc: 0.656516
[580]	valid_0's auc: 0.656645
[590]	valid_0's auc: 0.656738
[600]	valid_0's auc: 0.656799
[610]	valid_0's auc: 0.656915
[620]	valid_0's auc: 0.657006
[630]	valid_0's auc: 0.657109
[640]	valid_0's auc: 0.6572
[650]	valid_0's auc: 0.657303
[660]	valid_0's auc: 0.657363
[670]	valid_0's auc: 0.657437
[680]	valid_0's auc: 0.657529
[690]	valid_0's auc: 0.657561
[700]	valid_0's auc: 0.657667
[710]	valid_0's auc: 0.657763
[720]	valid_0's auc: 0.657857
[730]	valid_0's auc: 0.657934
[740]	valid_0's auc: 0.658
[750]	valid_0's auc: 0.658092
[760]	valid_0's auc: 0.65816
[770]	valid_0's auc: 0.65826
[780]	valid_0's auc: 0.658345
[790]	valid_0's auc: 0.658402
[800]	valid_0's auc: 0.658476
[810]	valid_0's auc: 0.658553
[820]	valid_0's auc: 0.658636
[830]	valid_0's auc: 0.658702
[840]	valid_0's auc: 0.658769
[850]	valid_0's auc: 0.658827
[860]	valid_0's auc: 0.658861
[870]	valid_0's auc: 0.658921
[880]	valid_0's auc: 0.658973
[890]	valid_0's auc: 0.659008
[900]	valid_0's auc: 0.659079
[910]	valid_0's auc: 0.659107
[920]	valid_0's auc: 0.659186
[930]	valid_0's auc: 0.659208
[940]	valid_0's auc: 0.659241
[950]	valid_0's auc: 0.659291
[960]	valid_0's auc: 0.659347
[970]	valid_0's auc: 0.659373
[980]	valid_0's auc: 0.659409
[990]	valid_0's auc: 0.659429
[1000]	valid_0's auc: 0.65947
[1010]	valid_0's auc: 0.659518
[1020]	valid_0's auc: 0.659561
[1030]	valid_0's auc: 0.659581
[1040]	valid_0's auc: 0.6596
[1050]	valid_0's auc: 0.659631
[1060]	valid_0's auc: 0.659696
[1070]	valid_0's auc: 0.659762
[1080]	valid_0's auc: 0.659777
[1090]	valid_0's auc: 0.659822
[1100]	valid_0's auc: 0.659879
[1110]	valid_0's auc: 0.659916
[1120]	valid_0's auc: 0.659965
[1130]	valid_0's auc: 0.660016
[1140]	valid_0's auc: 0.660046
[1150]	valid_0's auc: 0.660056
[1160]	valid_0's auc: 0.660094
[1170]	valid_0's auc: 0.660131
[1180]	valid_0's auc: 0.660148
[1190]	valid_0's auc: 0.660216
[1200]	valid_0's auc: 0.660255
[1210]	valid_0's auc: 0.660275
[1220]	valid_0's auc: 0.66028
[1230]	valid_0's auc: 0.66034
[1240]	valid_0's auc: 0.660372
[1250]	valid_0's auc: 0.660426
[1260]	valid_0's auc: 0.660437
[1270]	valid_0's auc: 0.660464
[1280]	valid_0's auc: 0.660494
[1290]	valid_0's auc: 0.660525
[1300]	valid_0's auc: 0.660542
[1310]	valid_0's auc: 0.660585
[1320]	valid_0's auc: 0.660603
[1330]	valid_0's auc: 0.660633
[1340]	valid_0's auc: 0.660652
[1350]	valid_0's auc: 0.660658
[1360]	valid_0's auc: 0.660681
[1370]	valid_0's auc: 0.660734
[1380]	valid_0's auc: 0.660744
[1390]	valid_0's auc: 0.66075
[1400]	valid_0's auc: 0.660765
[1410]	valid_0's auc: 0.660795
[1420]	valid_0's auc: 0.660814
[1430]	valid_0's auc: 0.660822
[1440]	valid_0's auc: 0.660855
[1450]	valid_0's auc: 0.660851
[1460]	valid_0's auc: 0.660865
[1470]	valid_0's auc: 0.660868
[1480]	valid_0's auc: 0.660895
[1490]	valid_0's auc: 0.660905
[1500]	valid_0's auc: 0.660911
[1510]	valid_0's auc: 0.660913
[1520]	valid_0's auc: 0.660928
[1530]	valid_0's auc: 0.660958
[1540]	valid_0's auc: 0.660995
[1550]	valid_0's auc: 0.661013
[1560]	valid_0's auc: 0.661031
[1570]	valid_0's auc: 0.661045
[1580]	valid_0's auc: 0.661058
[1590]	valid_0's auc: 0.661073
[1600]	valid_0's auc: 0.66108
[1610]	valid_0's auc: 0.661104
[1620]	valid_0's auc: 0.661107
[1630]	valid_0's auc: 0.661092
[1640]	valid_0's auc: 0.6611
[1650]	valid_0's auc: 0.661097
[1660]	valid_0's auc: 0.661096
[1670]	valid_0's auc: 0.66111
Early stopping, best iteration is:
[1625]	valid_0's auc: 0.661113
best score: 0.661113473837
best iteration: 1625

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 2 complete in 20m 17s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 3

max_depth            : -1
feature_fraction     : 0.8
boosting             : goss
lambda_l1            : 0.3
learning_rate        : 0.3
num_leaves           : 511
feature_fraction_seed : 2
lambda_l2            : 0

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.666027
[20]	valid_0's auc: 0.668753
[30]	valid_0's auc: 0.667232
[40]	valid_0's auc: 0.667117
[50]	valid_0's auc: 0.666656
[60]	valid_0's auc: 0.666258
Early stopping, best iteration is:
[17]	valid_0's auc: 0.669246
best score: 0.669245611728
best iteration: 17

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 3 complete in 11m 18s

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
------------Parameters-----------
round: 4

max_depth            : 10
feature_fraction     : 0.8
boosting             : goss
lambda_l1            : 0
learning_rate        : 0.2
num_leaves           : 511
feature_fraction_seed : 2
lambda_l2            : 0.3

Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.637153
[20]	valid_0's auc: 0.643464
[30]	valid_0's auc: 0.647086
[40]	valid_0's auc: 0.649468
[50]	valid_0's auc: 0.651098
[60]	valid_0's auc: 0.65263
[70]	valid_0's auc: 0.653948
[80]	valid_0's auc: 0.654705
[90]	valid_0's auc: 0.655377
[100]	valid_0's auc: 0.655987
[110]	valid_0's auc: 0.656466
[120]	valid_0's auc: 0.656801
[130]	valid_0's auc: 0.657139
[140]	valid_0's auc: 0.657453
[150]	valid_0's auc: 0.657583
[160]	valid_0's auc: 0.657745
[170]	valid_0's auc: 0.658045
[180]	valid_0's auc: 0.658256
[190]	valid_0's auc: 0.658539
[200]	valid_0's auc: 0.658557
[210]	valid_0's auc: 0.658323
[220]	valid_0's auc: 0.658536
[230]	valid_0's auc: 0.658707
[240]	valid_0's auc: 0.658755
[250]	valid_0's auc: 0.658795
[260]	valid_0's auc: 0.658833
[270]	valid_0's auc: 0.659
[280]	valid_0's auc: 0.659278
[290]	valid_0's auc: 0.659312
[300]	valid_0's auc: 0.659462
[310]	valid_0's auc: 0.659445
[320]	valid_0's auc: 0.659519
[330]	valid_0's auc: 0.659537
[340]	valid_0's auc: 0.659525
[350]	valid_0's auc: 0.659422
[360]	valid_0's auc: 0.659618
[370]	valid_0's auc: 0.659518
[380]	valid_0's auc: 0.659203
[390]	valid_0's auc: 0.659271
[400]	valid_0's auc: 0.659221
Early stopping, best iteration is:
[358]	valid_0's auc: 0.659652
best score: 0.659652165711
best iteration: 358

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

round: 4 complete in 4m 29s

[timer]: complete in 55m 11s

Process finished with exit code 0
'''