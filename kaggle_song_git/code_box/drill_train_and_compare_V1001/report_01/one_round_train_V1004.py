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
num_boost_round = 500000
early_stopping_rounds = 1000
verbose_eval = 10
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'verbose': -1,
    'num_leaves': 2**10-1,

    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 2**10-1,
    'max_depth': -1,
}
df = df[[
         'msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'fake_song_count',
         'fake_member_count',
         ]]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')


print()
print('our guest:')
print()
print(df.dtypes)
print('number of columns:', len(df.columns))
print()
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
del X_tr, Y_tr, X_val, Y_val


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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1001/in_column_train_V1001.py
What we got:
msno                     object
song_id                  object
source_system_tab        object
source_screen_name       object
source_type              object
target                    uint8
fake_member_count         int64
member_count              int64
artist_name              object
language               category
song_count                int64
artist_count              int64
language_count            int64
fake_song_count           int64
fake_artist_count         int64
fake_language_count       int64
dtype: object
number of columns: 16

our guest:

msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
fake_member_count        int64
dtype: object
number of columns: 10


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 1000 rounds.
[10]	valid_0's auc: 0.670285
[20]	valid_0's auc: 0.671483
[30]	valid_0's auc: 0.672627
[40]	valid_0's auc: 0.673252
[50]	valid_0's auc: 0.673573
[60]	valid_0's auc: 0.674218
[70]	valid_0's auc: 0.674848
[80]	valid_0's auc: 0.675405
[90]	valid_0's auc: 0.67576
[100]	valid_0's auc: 0.676217
[110]	valid_0's auc: 0.676597
[120]	valid_0's auc: 0.6769
[130]	valid_0's auc: 0.677308
[140]	valid_0's auc: 0.677673
[150]	valid_0's auc: 0.678011
[160]	valid_0's auc: 0.67867
[170]	valid_0's auc: 0.679127
[180]	valid_0's auc: 0.679531
[190]	valid_0's auc: 0.67987
[200]	valid_0's auc: 0.68035
[210]	valid_0's auc: 0.680763
[220]	valid_0's auc: 0.681161
[230]	valid_0's auc: 0.681413
[240]	valid_0's auc: 0.681692
[250]	valid_0's auc: 0.681953
[260]	valid_0's auc: 0.682201
[270]	valid_0's auc: 0.682613
[280]	valid_0's auc: 0.682997
[290]	valid_0's auc: 0.683226
[300]	valid_0's auc: 0.683458
[310]	valid_0's auc: 0.68371
[320]	valid_0's auc: 0.683885
[330]	valid_0's auc: 0.684043
[340]	valid_0's auc: 0.68422
[350]	valid_0's auc: 0.684345
[360]	valid_0's auc: 0.684475
[370]	valid_0's auc: 0.684615
[380]	valid_0's auc: 0.684744
[390]	valid_0's auc: 0.684861
[400]	valid_0's auc: 0.684992
[410]	valid_0's auc: 0.685096
[420]	valid_0's auc: 0.685206
[430]	valid_0's auc: 0.685297
[440]	valid_0's auc: 0.685397
[450]	valid_0's auc: 0.685491
[460]	valid_0's auc: 0.68557
[470]	valid_0's auc: 0.685661
[480]	valid_0's auc: 0.685701
[490]	valid_0's auc: 0.685756
[500]	valid_0's auc: 0.685788
[510]	valid_0's auc: 0.685839
[520]	valid_0's auc: 0.685861
[530]	valid_0's auc: 0.685897
[540]	valid_0's auc: 0.685929
[550]	valid_0's auc: 0.685939
[560]	valid_0's auc: 0.685951
[570]	valid_0's auc: 0.685958
[580]	valid_0's auc: 0.685984
[590]	valid_0's auc: 0.685987
[600]	valid_0's auc: 0.686005
[610]	valid_0's auc: 0.686031
[620]	valid_0's auc: 0.686063
[630]	valid_0's auc: 0.686075
[640]	valid_0's auc: 0.686104
[650]	valid_0's auc: 0.686107
[660]	valid_0's auc: 0.686132
[670]	valid_0's auc: 0.686135
[680]	valid_0's auc: 0.686158
[690]	valid_0's auc: 0.686162
[700]	valid_0's auc: 0.686172
[710]	valid_0's auc: 0.686174
[720]	valid_0's auc: 0.686171
[730]	valid_0's auc: 0.686202
[740]	valid_0's auc: 0.686228
[750]	valid_0's auc: 0.686248
[760]	valid_0's auc: 0.686252
[770]	valid_0's auc: 0.686258
[780]	valid_0's auc: 0.686267
[790]	valid_0's auc: 0.686272
[800]	valid_0's auc: 0.686278
[810]	valid_0's auc: 0.686287
[820]	valid_0's auc: 0.68628
[830]	valid_0's auc: 0.686274
[840]	valid_0's auc: 0.686292
[850]	valid_0's auc: 0.686297
[860]	valid_0's auc: 0.686336
[870]	valid_0's auc: 0.68635
[880]	valid_0's auc: 0.686361
[890]	valid_0's auc: 0.686373
[900]	valid_0's auc: 0.686385
[910]	valid_0's auc: 0.686394
[920]	valid_0's auc: 0.686367
[930]	valid_0's auc: 0.686369
[940]	valid_0's auc: 0.686378
[950]	valid_0's auc: 0.686373
[960]	valid_0's auc: 0.686395
[970]	valid_0's auc: 0.686391
[980]	valid_0's auc: 0.686414
[990]	valid_0's auc: 0.686416
[1000]	valid_0's auc: 0.686418
[1010]	valid_0's auc: 0.686441
[1020]	valid_0's auc: 0.686467
[1030]	valid_0's auc: 0.68648
[1040]	valid_0's auc: 0.686481
[1050]	valid_0's auc: 0.686496
[1060]	valid_0's auc: 0.686493
[1070]	valid_0's auc: 0.686494
[1080]	valid_0's auc: 0.686507
[1090]	valid_0's auc: 0.68652
[1100]	valid_0's auc: 0.686526
[1110]	valid_0's auc: 0.686545
[1120]	valid_0's auc: 0.686521
[1130]	valid_0's auc: 0.686548
[1140]	valid_0's auc: 0.686563
[1150]	valid_0's auc: 0.686579
[1160]	valid_0's auc: 0.686586
[1170]	valid_0's auc: 0.686598
[1180]	valid_0's auc: 0.686613
[1190]	valid_0's auc: 0.686622
[1200]	valid_0's auc: 0.686634
[1210]	valid_0's auc: 0.686626
[1220]	valid_0's auc: 0.686631
[1230]	valid_0's auc: 0.686635
[1240]	valid_0's auc: 0.686643
[1250]	valid_0's auc: 0.686639
[1260]	valid_0's auc: 0.686644
[1270]	valid_0's auc: 0.686659
[1280]	valid_0's auc: 0.686672
[1290]	valid_0's auc: 0.686672
[1300]	valid_0's auc: 0.686678
[1310]	valid_0's auc: 0.686677
[1320]	valid_0's auc: 0.686691
[1330]	valid_0's auc: 0.686699
[1340]	valid_0's auc: 0.686706
[1350]	valid_0's auc: 0.686709
[1360]	valid_0's auc: 0.686725
[1370]	valid_0's auc: 0.686735
[1380]	valid_0's auc: 0.686745
[1390]	valid_0's auc: 0.686753
[1400]	valid_0's auc: 0.686757
[1410]	valid_0's auc: 0.686762
[1420]	valid_0's auc: 0.686776
[1430]	valid_0's auc: 0.686794
[1440]	valid_0's auc: 0.686795
[1450]	valid_0's auc: 0.686794
[1460]	valid_0's auc: 0.686797
[1470]	valid_0's auc: 0.6868
[1480]	valid_0's auc: 0.686809
[1490]	valid_0's auc: 0.686819
[1500]	valid_0's auc: 0.68681
[1510]	valid_0's auc: 0.686809
[1520]	valid_0's auc: 0.686814
[1530]	valid_0's auc: 0.686829
[1540]	valid_0's auc: 0.686841
[1550]	valid_0's auc: 0.686829

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''