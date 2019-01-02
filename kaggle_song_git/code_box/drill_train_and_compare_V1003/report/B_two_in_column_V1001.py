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
save_dir = '../saves01/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
df.drop('gender', axis=1, inplace=True)
print('What we got:')
print(df.dtypes)
print('number of columns:', len(df.columns))

num_boost_round = 500000
early_stopping_rounds = 50
verbose_eval = 10
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'verbose': -1,
    'num_leaves': 2**6-1,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 15,
    'max_depth': 5,
}
# df['song_year'] = df['song_year'].astype('category')
# on = ['msno',
#       'song_id',
#       'target',
#       'source_system_tab',
#       'source_screen_name',
#       'source_type',
#       'language',
#       'artist_name',
#       'fake_song_count',
#       'fake_member_count',
#       # candidate
#       # 'fake_artist_count',
#       # 'fake_source_screen_name_count',
#       # new members
#       # 'fake_genre_type_count',
#       # 'fake_top1',
#
#       'song_year', # int
#       # 'song_country',
#       'fake_song_year_count', #00
#       'fake_song_country_count', # 00
#       'fake_top1_count', # 00
#       ]
# df = df[on]

fixed = [
         # 'msno',
         # 'song_id',
         'target',
         # 'source_system_tab',
         # 'source_screen_name',
         # 'source_type',
         # 'language',
         # 'artist_name',
         # 'fake_song_count',
         # 'fake_member_count',
         ]
done = []
rounds = 1
for q in df.columns:
    for w in df.columns:
        if w in fixed:
            pass
        elif q in fixed:
            pass
        elif w == q:
            pass
        elif [w, q] in done or [q, w] in done:
            pass
        else:
            print('-'*20)
            print('this is round:', rounds)
            print(w, 'and', q, 'are not in [DONE]:')
            for iii in done:
                print(iii)
            print('-'*20)
            done.append([w, q])
            rounds += 1

            # print('working on:', w)
            toto = [i for i in fixed]
            toto.append(w)
            toto.append(q)
            df = df[toto]

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype('category')

            print()
            print()
            print('After selection:')
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
            del train_set, val_set
            print('complete on:', w)
            print()
            dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
            df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
            del dt
            df.drop('gender', axis=1, inplace=True)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/drill_train_and_compare_V1003/B_two_in_column_V1001.py
What we got:
msno                          object
song_id                       object
source_system_tab             object
source_screen_name            object
source_type                   object
target                         uint8
city                        category
registered_via              category
sex                         category
sex_guess1                  category
sex_guess2                  category
sex_guess3                  category
sex_guess4                  category
sex_guess5                  category
sex_freq_member             category
registration_year           category
registration_month          category
registration_date           category
expiration_year             category
expiration_month            category
expiration_date             category
genre_ids                     object
artist_name                   object
composer                      object
lyricist                      object
language                    category
name                          object
genre_ids_fre_song          category
song_year_fre_song          category
song_year                   category
song_country_fre_song       category
song_country                category
rc                          category
source_system_tab_guess       object
source_screen_name_guess      object
source_type_guess             object
dtype: object
number of columns: 36
--------------------
this is round: 1
song_id and msno are not in [DONE]:
--------------------


After selection:
target        uint8
song_id    category
msno       category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.538867
[20]	valid_0's auc: 0.545731
[30]	valid_0's auc: 0.551595
[40]	valid_0's auc: 0.556258
[50]	valid_0's auc: 0.560601
[60]	valid_0's auc: 0.564488
[70]	valid_0's auc: 0.567383
[80]	valid_0's auc: 0.569383
[90]	valid_0's auc: 0.572998
[100]	valid_0's auc: 0.576134
[110]	valid_0's auc: 0.578701
[120]	valid_0's auc: 0.580413
[130]	valid_0's auc: 0.581539
[140]	valid_0's auc: 0.583416
[150]	valid_0's auc: 0.584794
[160]	valid_0's auc: 0.58603
[170]	valid_0's auc: 0.587311
[180]	valid_0's auc: 0.5882
[190]	valid_0's auc: 0.589183
[200]	valid_0's auc: 0.590191
[210]	valid_0's auc: 0.591903
[220]	valid_0's auc: 0.592985
[230]	valid_0's auc: 0.594273
[240]	valid_0's auc: 0.595326
[250]	valid_0's auc: 0.596592
[260]	valid_0's auc: 0.597305
[270]	valid_0's auc: 0.598148
[280]	valid_0's auc: 0.598726
[290]	valid_0's auc: 0.599224
[300]	valid_0's auc: 0.59974
[310]	valid_0's auc: 0.600366
[320]	valid_0's auc: 0.601287
[330]	valid_0's auc: 0.601841
[340]	valid_0's auc: 0.602388
[350]	valid_0's auc: 0.602765
[360]	valid_0's auc: 0.603292
[370]	valid_0's auc: 0.603643
[380]	valid_0's auc: 0.604098
[390]	valid_0's auc: 0.604626
[400]	valid_0's auc: 0.605324
[410]	valid_0's auc: 0.605623
[420]	valid_0's auc: 0.606024
[430]	valid_0's auc: 0.606346
[440]	valid_0's auc: 0.6065
[450]	valid_0's auc: 0.606892
[460]	valid_0's auc: 0.607325
[470]	valid_0's auc: 0.60757
[480]	valid_0's auc: 0.607806
[490]	valid_0's auc: 0.608123
[500]	valid_0's auc: 0.608409
[510]	valid_0's auc: 0.608571
[520]	valid_0's auc: 0.608811
[530]	valid_0's auc: 0.609032
[540]	valid_0's auc: 0.609232
[550]	valid_0's auc: 0.609507
[560]	valid_0's auc: 0.609792
[570]	valid_0's auc: 0.610033
[580]	valid_0's auc: 0.610121
[590]	valid_0's auc: 0.610324
[600]	valid_0's auc: 0.610463
[610]	valid_0's auc: 0.610719
[620]	valid_0's auc: 0.610924
[630]	valid_0's auc: 0.611009
[640]	valid_0's auc: 0.61105
[650]	valid_0's auc: 0.611098
[660]	valid_0's auc: 0.611282
[670]	valid_0's auc: 0.611385
[680]	valid_0's auc: 0.611586
[690]	valid_0's auc: 0.611818
[700]	valid_0's auc: 0.612042
[710]	valid_0's auc: 0.61213
[720]	valid_0's auc: 0.612269
[730]	valid_0's auc: 0.612557
[740]	valid_0's auc: 0.612711
[750]	valid_0's auc: 0.61283
[760]	valid_0's auc: 0.613032
[770]	valid_0's auc: 0.613222
[780]	valid_0's auc: 0.613382
[790]	valid_0's auc: 0.613575
[800]	valid_0's auc: 0.613636
[810]	valid_0's auc: 0.613761
[820]	valid_0's auc: 0.613896
[830]	valid_0's auc: 0.614017
[840]	valid_0's auc: 0.614023
[850]	valid_0's auc: 0.614149
[860]	valid_0's auc: 0.614246
[870]	valid_0's auc: 0.614266
[880]	valid_0's auc: 0.614336
[890]	valid_0's auc: 0.614376
[900]	valid_0's auc: 0.614438
[910]	valid_0's auc: 0.614514
[920]	valid_0's auc: 0.614562
[930]	valid_0's auc: 0.614628
[940]	valid_0's auc: 0.614731
[950]	valid_0's auc: 0.614732
[960]	valid_0's auc: 0.614806
[970]	valid_0's auc: 0.614859
[980]	valid_0's auc: 0.614947
[990]	valid_0's auc: 0.615007
[1000]	valid_0's auc: 0.615136
[1010]	valid_0's auc: 0.615217
[1020]	valid_0's auc: 0.615285
[1030]	valid_0's auc: 0.615269
[1040]	valid_0's auc: 0.615333
[1050]	valid_0's auc: 0.615417
[1060]	valid_0's auc: 0.615453
[1070]	valid_0's auc: 0.615509
[1080]	valid_0's auc: 0.615575
[1090]	valid_0's auc: 0.615715
[1100]	valid_0's auc: 0.615776
[1110]	valid_0's auc: 0.615837
[1120]	valid_0's auc: 0.615987
[1130]	valid_0's auc: 0.616004
[1140]	valid_0's auc: 0.616087
[1150]	valid_0's auc: 0.616177
[1160]	valid_0's auc: 0.616286
[1170]	valid_0's auc: 0.616378
[1180]	valid_0's auc: 0.61642
[1190]	valid_0's auc: 0.616443
[1200]	valid_0's auc: 0.616528
[1210]	valid_0's auc: 0.616554
[1220]	valid_0's auc: 0.616535
[1230]	valid_0's auc: 0.616586
[1240]	valid_0's auc: 0.616583
[1250]	valid_0's auc: 0.616642
[1260]	valid_0's auc: 0.616699
[1270]	valid_0's auc: 0.616706
[1280]	valid_0's auc: 0.616745
[1290]	valid_0's auc: 0.616773
[1300]	valid_0's auc: 0.616815
[1310]	valid_0's auc: 0.616841
[1320]	valid_0's auc: 0.616896
[1330]	valid_0's auc: 0.616914
[1340]	valid_0's auc: 0.616926
[1350]	valid_0's auc: 0.616971
[1360]	valid_0's auc: 0.617022
[1370]	valid_0's auc: 0.617086
[1380]	valid_0's auc: 0.617099
[1390]	valid_0's auc: 0.617124
[1400]	valid_0's auc: 0.617147
[1410]	valid_0's auc: 0.617191
[1420]	valid_0's auc: 0.617217
[1430]	valid_0's auc: 0.617267
[1440]	valid_0's auc: 0.617313
[1450]	valid_0's auc: 0.617335
[1460]	valid_0's auc: 0.61737
[1470]	valid_0's auc: 0.617415
[1480]	valid_0's auc: 0.617414
[1490]	valid_0's auc: 0.617418
[1500]	valid_0's auc: 0.617439
[1510]	valid_0's auc: 0.617463
[1520]	valid_0's auc: 0.61746
[1530]	valid_0's auc: 0.617462
[1540]	valid_0's auc: 0.617469
[1550]	valid_0's auc: 0.617505
[1560]	valid_0's auc: 0.617536
[1570]	valid_0's auc: 0.617578
[1580]	valid_0's auc: 0.617584
[1590]	valid_0's auc: 0.617637
[1600]	valid_0's auc: 0.617645
[1610]	valid_0's auc: 0.617647
[1620]	valid_0's auc: 0.617674
[1630]	valid_0's auc: 0.617649
[1640]	valid_0's auc: 0.617624
[1650]	valid_0's auc: 0.617644
[1660]	valid_0's auc: 0.617668
[1670]	valid_0's auc: 0.617708
[1680]	valid_0's auc: 0.617744
[1690]	valid_0's auc: 0.61776
[1700]	valid_0's auc: 0.617772
[1710]	valid_0's auc: 0.617753
[1720]	valid_0's auc: 0.617791
[1730]	valid_0's auc: 0.617838
[1740]	valid_0's auc: 0.617868
[1750]	valid_0's auc: 0.617901
[1760]	valid_0's auc: 0.617931
[1770]	valid_0's auc: 0.617927
[1780]	valid_0's auc: 0.617926
[1790]	valid_0's auc: 0.617946
[1800]	valid_0's auc: 0.617968
[1810]	valid_0's auc: 0.617988
[1820]	valid_0's auc: 0.618028
[1830]	valid_0's auc: 0.618036
[1840]	valid_0's auc: 0.618058
[1850]	valid_0's auc: 0.618047
[1860]	valid_0's auc: 0.618025
[1870]	valid_0's auc: 0.618027
[1880]	valid_0's auc: 0.618034
[1890]	valid_0's auc: 0.618055
[1900]	valid_0's auc: 0.618076
[1910]	valid_0's auc: 0.618082
[1920]	valid_0's auc: 0.618076
[1930]	valid_0's auc: 0.618068
[1940]	valid_0's auc: 0.61807
[1950]	valid_0's auc: 0.618075
[1960]	valid_0's auc: 0.618109
[1970]	valid_0's auc: 0.618131
[1980]	valid_0's auc: 0.61814
[1990]	valid_0's auc: 0.618159
[2000]	valid_0's auc: 0.618168
[2010]	valid_0's auc: 0.618173
[2020]	valid_0's auc: 0.618177
[2030]	valid_0's auc: 0.618196
[2040]	valid_0's auc: 0.618192
[2050]	valid_0's auc: 0.618189
[2060]	valid_0's auc: 0.618188
[2070]	valid_0's auc: 0.618193
Early stopping, best iteration is:
[2027]	valid_0's auc: 0.618202
best score: 0.618201683505
best iteration: 2027
complete on: song_id

--------------------
this is round: 2
source_system_tab and msno are not in [DONE]:
['song_id', 'msno']
--------------------


After selection:
target                  uint8
source_system_tab    category
msno                 category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.593778
[20]	valid_0's auc: 0.597661
[30]	valid_0's auc: 0.600743
[40]	valid_0's auc: 0.602584
[50]	valid_0's auc: 0.604346
[60]	valid_0's auc: 0.60625
[70]	valid_0's auc: 0.607174
[80]	valid_0's auc: 0.608193
[90]	valid_0's auc: 0.609157
[100]	valid_0's auc: 0.609844
[110]	valid_0's auc: 0.61096
[120]	valid_0's auc: 0.612121
[130]	valid_0's auc: 0.613168
[140]	valid_0's auc: 0.613906
[150]	valid_0's auc: 0.614492
[160]	valid_0's auc: 0.615081
[170]	valid_0's auc: 0.615965
[180]	valid_0's auc: 0.616438
[190]	valid_0's auc: 0.616879
[200]	valid_0's auc: 0.617305
[210]	valid_0's auc: 0.617814
[220]	valid_0's auc: 0.617924
[230]	valid_0's auc: 0.618387
[240]	valid_0's auc: 0.618871
[250]	valid_0's auc: 0.619215
[260]	valid_0's auc: 0.619655
[270]	valid_0's auc: 0.620322
[280]	valid_0's auc: 0.620664
[290]	valid_0's auc: 0.621267
[300]	valid_0's auc: 0.62183
[310]	valid_0's auc: 0.622465
[320]	valid_0's auc: 0.622802
[330]	valid_0's auc: 0.623126
[340]	valid_0's auc: 0.623594
[350]	valid_0's auc: 0.623943
[360]	valid_0's auc: 0.624369
[370]	valid_0's auc: 0.62467
[380]	valid_0's auc: 0.625035
[390]	valid_0's auc: 0.625239
[400]	valid_0's auc: 0.625343
[410]	valid_0's auc: 0.625509
[420]	valid_0's auc: 0.6258
[430]	valid_0's auc: 0.62572
[440]	valid_0's auc: 0.625749
[450]	valid_0's auc: 0.62601
[460]	valid_0's auc: 0.626097
[470]	valid_0's auc: 0.62629
[480]	valid_0's auc: 0.626578
[490]	valid_0's auc: 0.626673
[500]	valid_0's auc: 0.626923
[510]	valid_0's auc: 0.627186
[520]	valid_0's auc: 0.627345
[530]	valid_0's auc: 0.62734
[540]	valid_0's auc: 0.627704
[550]	valid_0's auc: 0.627883
[560]	valid_0's auc: 0.627925
[570]	valid_0's auc: 0.627982
[580]	valid_0's auc: 0.627979
[590]	valid_0's auc: 0.628139
[600]	valid_0's auc: 0.628188
[610]	valid_0's auc: 0.628308
[620]	valid_0's auc: 0.628457
[630]	valid_0's auc: 0.628598
[640]	valid_0's auc: 0.62869
[650]	valid_0's auc: 0.628654
[660]	valid_0's auc: 0.628691
[670]	valid_0's auc: 0.628717
[680]	valid_0's auc: 0.628877
[690]	valid_0's auc: 0.628991
[700]	valid_0's auc: 0.62893
[710]	valid_0's auc: 0.629045
[720]	valid_0's auc: 0.629146
[730]	valid_0's auc: 0.629208
[740]	valid_0's auc: 0.629273
[750]	valid_0's auc: 0.629339
[760]	valid_0's auc: 0.629328
[770]	valid_0's auc: 0.629454
[780]	valid_0's auc: 0.629529
[790]	valid_0's auc: 0.629716
[800]	valid_0's auc: 0.629783
[810]	valid_0's auc: 0.629794
[820]	valid_0's auc: 0.629992
[830]	valid_0's auc: 0.630112
[840]	valid_0's auc: 0.630165
[850]	valid_0's auc: 0.630385
[860]	valid_0's auc: 0.630389
[870]	valid_0's auc: 0.630449
[880]	valid_0's auc: 0.630476
[890]	valid_0's auc: 0.630537
[900]	valid_0's auc: 0.630627
[910]	valid_0's auc: 0.630767
[920]	valid_0's auc: 0.630817
[930]	valid_0's auc: 0.630819
[940]	valid_0's auc: 0.630891
[950]	valid_0's auc: 0.630944
[960]	valid_0's auc: 0.630943
[970]	valid_0's auc: 0.630967
[980]	valid_0's auc: 0.631016
[990]	valid_0's auc: 0.631115
[1000]	valid_0's auc: 0.631187
[1010]	valid_0's auc: 0.631229
[1020]	valid_0's auc: 0.63126
[1030]	valid_0's auc: 0.631267
[1040]	valid_0's auc: 0.631296
[1050]	valid_0's auc: 0.631396
[1060]	valid_0's auc: 0.631469
[1070]	valid_0's auc: 0.631502
[1080]	valid_0's auc: 0.631523
[1090]	valid_0's auc: 0.631513
[1100]	valid_0's auc: 0.631576
[1110]	valid_0's auc: 0.631617
[1120]	valid_0's auc: 0.631617
[1130]	valid_0's auc: 0.631613
[1140]	valid_0's auc: 0.631631
[1150]	valid_0's auc: 0.63167
[1160]	valid_0's auc: 0.631716
[1170]	valid_0's auc: 0.631709
[1180]	valid_0's auc: 0.631682
[1190]	valid_0's auc: 0.631705
[1200]	valid_0's auc: 0.631682
Early stopping, best iteration is:
[1157]	valid_0's auc: 0.631729
best score: 0.631729273434
best iteration: 1157
complete on: source_system_tab

--------------------
this is round: 3
source_screen_name and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
--------------------


After selection:
target                   uint8
source_screen_name    category
msno                  category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.602235
[20]	valid_0's auc: 0.605567
[30]	valid_0's auc: 0.607827
[40]	valid_0's auc: 0.609921
[50]	valid_0's auc: 0.611035
[60]	valid_0's auc: 0.612464
[70]	valid_0's auc: 0.613714
[80]	valid_0's auc: 0.614797
[90]	valid_0's auc: 0.615613
[100]	valid_0's auc: 0.616372
[110]	valid_0's auc: 0.616922
[120]	valid_0's auc: 0.617193
[130]	valid_0's auc: 0.617979
[140]	valid_0's auc: 0.618956
[150]	valid_0's auc: 0.619599
[160]	valid_0's auc: 0.620336
[170]	valid_0's auc: 0.621117
[180]	valid_0's auc: 0.621963
[190]	valid_0's auc: 0.622355
[200]	valid_0's auc: 0.622885
[210]	valid_0's auc: 0.6234
[220]	valid_0's auc: 0.623738
[230]	valid_0's auc: 0.624104
[240]	valid_0's auc: 0.624392
[250]	valid_0's auc: 0.624695
[260]	valid_0's auc: 0.625329
[270]	valid_0's auc: 0.625583
[280]	valid_0's auc: 0.625738
[290]	valid_0's auc: 0.626073
[300]	valid_0's auc: 0.626343
[310]	valid_0's auc: 0.626638
[320]	valid_0's auc: 0.626893
[330]	valid_0's auc: 0.627052
[340]	valid_0's auc: 0.627261
[350]	valid_0's auc: 0.627583
[360]	valid_0's auc: 0.627955
[370]	valid_0's auc: 0.628306
[380]	valid_0's auc: 0.628674
[390]	valid_0's auc: 0.628997
[400]	valid_0's auc: 0.629317
[410]	valid_0's auc: 0.629615
[420]	valid_0's auc: 0.630008
[430]	valid_0's auc: 0.630258
[440]	valid_0's auc: 0.630454
[450]	valid_0's auc: 0.630649
[460]	valid_0's auc: 0.630769
[470]	valid_0's auc: 0.631056
[480]	valid_0's auc: 0.631248
[490]	valid_0's auc: 0.631462
[500]	valid_0's auc: 0.631607
[510]	valid_0's auc: 0.63181
[520]	valid_0's auc: 0.631953
[530]	valid_0's auc: 0.632092
[540]	valid_0's auc: 0.632083
[550]	valid_0's auc: 0.632288
[560]	valid_0's auc: 0.632313
[570]	valid_0's auc: 0.632433
[580]	valid_0's auc: 0.632576
[590]	valid_0's auc: 0.632673
[600]	valid_0's auc: 0.632824
[610]	valid_0's auc: 0.632962
[620]	valid_0's auc: 0.633097
[630]	valid_0's auc: 0.633144
[640]	valid_0's auc: 0.633191
[650]	valid_0's auc: 0.6333
[660]	valid_0's auc: 0.63349
[670]	valid_0's auc: 0.63355
[680]	valid_0's auc: 0.633679
[690]	valid_0's auc: 0.63372
[700]	valid_0's auc: 0.633842
[710]	valid_0's auc: 0.633911
[720]	valid_0's auc: 0.633958
[730]	valid_0's auc: 0.634082
[740]	valid_0's auc: 0.634088
[750]	valid_0's auc: 0.634141
[760]	valid_0's auc: 0.634195
[770]	valid_0's auc: 0.634225
[780]	valid_0's auc: 0.634347
[790]	valid_0's auc: 0.634432
[800]	valid_0's auc: 0.634535
[810]	valid_0's auc: 0.634676
[820]	valid_0's auc: 0.634721
[830]	valid_0's auc: 0.634844
[840]	valid_0's auc: 0.634874
[850]	valid_0's auc: 0.635016
[860]	valid_0's auc: 0.63511
[870]	valid_0's auc: 0.635149
[880]	valid_0's auc: 0.635286
[890]	valid_0's auc: 0.63545
[900]	valid_0's auc: 0.635593
[910]	valid_0's auc: 0.635672
[920]	valid_0's auc: 0.63581
[930]	valid_0's auc: 0.63584
[940]	valid_0's auc: 0.635936
[950]	valid_0's auc: 0.635963
[960]	valid_0's auc: 0.636047
[970]	valid_0's auc: 0.636108
[980]	valid_0's auc: 0.636178
[990]	valid_0's auc: 0.636221
[1000]	valid_0's auc: 0.636287
[1010]	valid_0's auc: 0.636346
[1020]	valid_0's auc: 0.636422
[1030]	valid_0's auc: 0.636491
[1040]	valid_0's auc: 0.636542
[1050]	valid_0's auc: 0.636632
[1060]	valid_0's auc: 0.63668
[1070]	valid_0's auc: 0.63672
[1080]	valid_0's auc: 0.63681
[1090]	valid_0's auc: 0.636876
[1100]	valid_0's auc: 0.636905
[1110]	valid_0's auc: 0.636924
[1120]	valid_0's auc: 0.636916
[1130]	valid_0's auc: 0.636981
[1140]	valid_0's auc: 0.637004
[1150]	valid_0's auc: 0.63706
[1160]	valid_0's auc: 0.637158
[1170]	valid_0's auc: 0.637163
[1180]	valid_0's auc: 0.637184
[1190]	valid_0's auc: 0.637206
[1200]	valid_0's auc: 0.637213
[1210]	valid_0's auc: 0.637238
[1220]	valid_0's auc: 0.637278
[1230]	valid_0's auc: 0.637301
[1240]	valid_0's auc: 0.637298
[1250]	valid_0's auc: 0.637342
[1260]	valid_0's auc: 0.637365
[1270]	valid_0's auc: 0.637401
[1280]	valid_0's auc: 0.637448
[1290]	valid_0's auc: 0.637419
[1300]	valid_0's auc: 0.637436
[1310]	valid_0's auc: 0.637421
[1320]	valid_0's auc: 0.637447
[1330]	valid_0's auc: 0.63755
[1340]	valid_0's auc: 0.637582
[1350]	valid_0's auc: 0.637589
[1360]	valid_0's auc: 0.637616
[1370]	valid_0's auc: 0.637653
[1380]	valid_0's auc: 0.637672
[1390]	valid_0's auc: 0.637653
[1400]	valid_0's auc: 0.637672
[1410]	valid_0's auc: 0.637717
[1420]	valid_0's auc: 0.637753
[1430]	valid_0's auc: 0.637769
[1440]	valid_0's auc: 0.637815
[1450]	valid_0's auc: 0.637939
[1460]	valid_0's auc: 0.637955
[1470]	valid_0's auc: 0.638015
[1480]	valid_0's auc: 0.638071
[1490]	valid_0's auc: 0.638082
[1500]	valid_0's auc: 0.638178
[1510]	valid_0's auc: 0.638206
[1520]	valid_0's auc: 0.638238
[1530]	valid_0's auc: 0.638244
[1540]	valid_0's auc: 0.638376
[1550]	valid_0's auc: 0.63837
[1560]	valid_0's auc: 0.638391
[1570]	valid_0's auc: 0.638401
[1580]	valid_0's auc: 0.638482
[1590]	valid_0's auc: 0.638503
[1600]	valid_0's auc: 0.638524
[1610]	valid_0's auc: 0.638544
[1620]	valid_0's auc: 0.638581
[1630]	valid_0's auc: 0.638594
[1640]	valid_0's auc: 0.638657
[1650]	valid_0's auc: 0.638682
[1660]	valid_0's auc: 0.63868
[1670]	valid_0's auc: 0.638723
[1680]	valid_0's auc: 0.63873
[1690]	valid_0's auc: 0.638734
[1700]	valid_0's auc: 0.638766
[1710]	valid_0's auc: 0.638804
[1720]	valid_0's auc: 0.63881
[1730]	valid_0's auc: 0.638864
[1740]	valid_0's auc: 0.638937
[1750]	valid_0's auc: 0.638953
[1760]	valid_0's auc: 0.638975
[1770]	valid_0's auc: 0.639063
[1780]	valid_0's auc: 0.639071
[1790]	valid_0's auc: 0.6391
[1800]	valid_0's auc: 0.639131
[1810]	valid_0's auc: 0.639166
[1820]	valid_0's auc: 0.639166
[1830]	valid_0's auc: 0.639188
[1840]	valid_0's auc: 0.639241
[1850]	valid_0's auc: 0.639251
[1860]	valid_0's auc: 0.639287
[1870]	valid_0's auc: 0.639281
[1880]	valid_0's auc: 0.63929
[1890]	valid_0's auc: 0.639316
[1900]	valid_0's auc: 0.639327
[1910]	valid_0's auc: 0.639331
[1920]	valid_0's auc: 0.639347
[1930]	valid_0's auc: 0.639461
[1940]	valid_0's auc: 0.639477
[1950]	valid_0's auc: 0.639491
[1960]	valid_0's auc: 0.639486
[1970]	valid_0's auc: 0.639528
[1980]	valid_0's auc: 0.639512
[1990]	valid_0's auc: 0.639526
[2000]	valid_0's auc: 0.639533
[2010]	valid_0's auc: 0.639564
[2020]	valid_0's auc: 0.639583
[2030]	valid_0's auc: 0.63959
[2040]	valid_0's auc: 0.639612
[2050]	valid_0's auc: 0.639623
[2060]	valid_0's auc: 0.639668
[2070]	valid_0's auc: 0.639673
[2080]	valid_0's auc: 0.639635
[2090]	valid_0's auc: 0.639647
[2100]	valid_0's auc: 0.639674
[2110]	valid_0's auc: 0.639678
[2120]	valid_0's auc: 0.639694
[2130]	valid_0's auc: 0.63972
[2140]	valid_0's auc: 0.639741
[2150]	valid_0's auc: 0.639738
[2160]	valid_0's auc: 0.639769
[2170]	valid_0's auc: 0.639807
[2180]	valid_0's auc: 0.639839
[2190]	valid_0's auc: 0.639877
[2200]	valid_0's auc: 0.639901
[2210]	valid_0's auc: 0.639947
[2220]	valid_0's auc: 0.639951
[2230]	valid_0's auc: 0.639975
[2240]	valid_0's auc: 0.639993
[2250]	valid_0's auc: 0.640019
[2260]	valid_0's auc: 0.640033
[2270]	valid_0's auc: 0.64004
[2280]	valid_0's auc: 0.640066
[2290]	valid_0's auc: 0.640099
[2300]	valid_0's auc: 0.640113
[2310]	valid_0's auc: 0.640138
[2320]	valid_0's auc: 0.640162
[2330]	valid_0's auc: 0.640199
[2340]	valid_0's auc: 0.640214
[2350]	valid_0's auc: 0.640239
[2360]	valid_0's auc: 0.640263
[2370]	valid_0's auc: 0.640331
[2380]	valid_0's auc: 0.640363
[2390]	valid_0's auc: 0.640403
[2400]	valid_0's auc: 0.640472
[2410]	valid_0's auc: 0.640501
[2420]	valid_0's auc: 0.64051
[2430]	valid_0's auc: 0.640527
[2440]	valid_0's auc: 0.640517
[2450]	valid_0's auc: 0.640512
[2460]	valid_0's auc: 0.640541
[2470]	valid_0's auc: 0.640629
[2480]	valid_0's auc: 0.640651
[2490]	valid_0's auc: 0.640667
[2500]	valid_0's auc: 0.640667
[2510]	valid_0's auc: 0.640668
[2520]	valid_0's auc: 0.640716
[2530]	valid_0's auc: 0.640744
[2540]	valid_0's auc: 0.64074
[2550]	valid_0's auc: 0.640747
[2560]	valid_0's auc: 0.640756
[2570]	valid_0's auc: 0.640729
[2580]	valid_0's auc: 0.64078
[2590]	valid_0's auc: 0.640884
[2600]	valid_0's auc: 0.640874
[2610]	valid_0's auc: 0.640889
[2620]	valid_0's auc: 0.640916
[2630]	valid_0's auc: 0.640967
[2640]	valid_0's auc: 0.641034
[2650]	valid_0's auc: 0.641037
[2660]	valid_0's auc: 0.641041
[2670]	valid_0's auc: 0.641171
[2680]	valid_0's auc: 0.641188
[2690]	valid_0's auc: 0.641215
[2700]	valid_0's auc: 0.64128
[2710]	valid_0's auc: 0.64128
[2720]	valid_0's auc: 0.64132
[2730]	valid_0's auc: 0.641312
[2740]	valid_0's auc: 0.641322
[2750]	valid_0's auc: 0.641342
[2760]	valid_0's auc: 0.641386
[2770]	valid_0's auc: 0.641404
[2780]	valid_0's auc: 0.641419
[2790]	valid_0's auc: 0.64143
[2800]	valid_0's auc: 0.641456
[2810]	valid_0's auc: 0.641457
[2820]	valid_0's auc: 0.641514
[2830]	valid_0's auc: 0.641533
[2840]	valid_0's auc: 0.64153
[2850]	valid_0's auc: 0.641519
[2860]	valid_0's auc: 0.641647
[2870]	valid_0's auc: 0.641632
[2880]	valid_0's auc: 0.64163
[2890]	valid_0's auc: 0.641656
[2900]	valid_0's auc: 0.641664
[2910]	valid_0's auc: 0.641681
[2920]	valid_0's auc: 0.641697
[2930]	valid_0's auc: 0.641697
[2940]	valid_0's auc: 0.641745
[2950]	valid_0's auc: 0.641774
[2960]	valid_0's auc: 0.641825
[2970]	valid_0's auc: 0.641852
[2980]	valid_0's auc: 0.641882
[2990]	valid_0's auc: 0.641897
[3000]	valid_0's auc: 0.641909
[3010]	valid_0's auc: 0.641915
[3020]	valid_0's auc: 0.641918
[3030]	valid_0's auc: 0.641963
[3040]	valid_0's auc: 0.641966
[3050]	valid_0's auc: 0.64196
[3060]	valid_0's auc: 0.641974
[3070]	valid_0's auc: 0.641999
[3080]	valid_0's auc: 0.642005
[3090]	valid_0's auc: 0.642024
[3100]	valid_0's auc: 0.642042
[3110]	valid_0's auc: 0.642045
[3120]	valid_0's auc: 0.642058
[3130]	valid_0's auc: 0.64208
[3140]	valid_0's auc: 0.64209
[3150]	valid_0's auc: 0.642117
[3160]	valid_0's auc: 0.642116
[3170]	valid_0's auc: 0.642085
[3180]	valid_0's auc: 0.642085
[3190]	valid_0's auc: 0.642092
[3200]	valid_0's auc: 0.642087
[3210]	valid_0's auc: 0.642133
[3220]	valid_0's auc: 0.642152
[3230]	valid_0's auc: 0.642148
[3240]	valid_0's auc: 0.64215
[3250]	valid_0's auc: 0.642155
[3260]	valid_0's auc: 0.642154
[3270]	valid_0's auc: 0.642152
[3280]	valid_0's auc: 0.642159
[3290]	valid_0's auc: 0.642173
[3300]	valid_0's auc: 0.642171
[3310]	valid_0's auc: 0.642198
[3320]	valid_0's auc: 0.64222
[3330]	valid_0's auc: 0.642229
[3340]	valid_0's auc: 0.642331
[3350]	valid_0's auc: 0.642411
[3360]	valid_0's auc: 0.642397
[3370]	valid_0's auc: 0.64244
[3380]	valid_0's auc: 0.642461
[3390]	valid_0's auc: 0.642465
[3400]	valid_0's auc: 0.642463
[3410]	valid_0's auc: 0.642481
[3420]	valid_0's auc: 0.642485
[3430]	valid_0's auc: 0.642472
[3440]	valid_0's auc: 0.642467
[3450]	valid_0's auc: 0.642545
[3460]	valid_0's auc: 0.642619
[3470]	valid_0's auc: 0.642633
[3480]	valid_0's auc: 0.642624
[3490]	valid_0's auc: 0.642639
[3500]	valid_0's auc: 0.642646
[3510]	valid_0's auc: 0.642651
[3520]	valid_0's auc: 0.642645
[3530]	valid_0's auc: 0.642671
[3540]	valid_0's auc: 0.642678
[3550]	valid_0's auc: 0.642681
[3560]	valid_0's auc: 0.642708
[3570]	valid_0's auc: 0.64274
[3580]	valid_0's auc: 0.642733
[3590]	valid_0's auc: 0.642735
[3600]	valid_0's auc: 0.642735
[3610]	valid_0's auc: 0.642781
[3620]	valid_0's auc: 0.642777
[3630]	valid_0's auc: 0.642774
[3640]	valid_0's auc: 0.64276
[3650]	valid_0's auc: 0.642777
[3660]	valid_0's auc: 0.642792
[3670]	valid_0's auc: 0.642811
[3680]	valid_0's auc: 0.642816
[3690]	valid_0's auc: 0.642829
[3700]	valid_0's auc: 0.642841
[3710]	valid_0's auc: 0.642857
[3720]	valid_0's auc: 0.642858
[3730]	valid_0's auc: 0.64286
[3740]	valid_0's auc: 0.642866
[3750]	valid_0's auc: 0.642909
[3760]	valid_0's auc: 0.642991
[3770]	valid_0's auc: 0.643001
[3780]	valid_0's auc: 0.643012
[3790]	valid_0's auc: 0.643008
[3800]	valid_0's auc: 0.643008
[3810]	valid_0's auc: 0.642997
[3820]	valid_0's auc: 0.643008
Early stopping, best iteration is:
[3776]	valid_0's auc: 0.643016
best score: 0.64301557249
best iteration: 3776
complete on: source_screen_name

--------------------
this is round: 4
source_type and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
--------------------


After selection:
target            uint8
source_type    category
msno           category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.599161
[20]	valid_0's auc: 0.603007
[30]	valid_0's auc: 0.609044
[40]	valid_0's auc: 0.610928
[50]	valid_0's auc: 0.612195
[60]	valid_0's auc: 0.613838
[70]	valid_0's auc: 0.61478
[80]	valid_0's auc: 0.615664
[90]	valid_0's auc: 0.616512
[100]	valid_0's auc: 0.617425
[110]	valid_0's auc: 0.618639
[120]	valid_0's auc: 0.619279
[130]	valid_0's auc: 0.620135
[140]	valid_0's auc: 0.620617
[150]	valid_0's auc: 0.621132
[160]	valid_0's auc: 0.621639
[170]	valid_0's auc: 0.622242
[180]	valid_0's auc: 0.622634
[190]	valid_0's auc: 0.623133
[200]	valid_0's auc: 0.623668
[210]	valid_0's auc: 0.623997
[220]	valid_0's auc: 0.624379
[230]	valid_0's auc: 0.624716
[240]	valid_0's auc: 0.625185
[250]	valid_0's auc: 0.62569
[260]	valid_0's auc: 0.626233
[270]	valid_0's auc: 0.626835
[280]	valid_0's auc: 0.627316
[290]	valid_0's auc: 0.627765
[300]	valid_0's auc: 0.628187
[310]	valid_0's auc: 0.628573
[320]	valid_0's auc: 0.62905
[330]	valid_0's auc: 0.629392
[340]	valid_0's auc: 0.629785
[350]	valid_0's auc: 0.630183
[360]	valid_0's auc: 0.630429
[370]	valid_0's auc: 0.630646
[380]	valid_0's auc: 0.630967
[390]	valid_0's auc: 0.631237
[400]	valid_0's auc: 0.631505
[410]	valid_0's auc: 0.631718
[420]	valid_0's auc: 0.631975
[430]	valid_0's auc: 0.632075
[440]	valid_0's auc: 0.63224
[450]	valid_0's auc: 0.632432
[460]	valid_0's auc: 0.632531
[470]	valid_0's auc: 0.632944
[480]	valid_0's auc: 0.6331
[490]	valid_0's auc: 0.633218
[500]	valid_0's auc: 0.63355
[510]	valid_0's auc: 0.633718
[520]	valid_0's auc: 0.633945
[530]	valid_0's auc: 0.634149
[540]	valid_0's auc: 0.634384
[550]	valid_0's auc: 0.634613
[560]	valid_0's auc: 0.634787
[570]	valid_0's auc: 0.634911
[580]	valid_0's auc: 0.63503
[590]	valid_0's auc: 0.635154
[600]	valid_0's auc: 0.63516
[610]	valid_0's auc: 0.635378
[620]	valid_0's auc: 0.635451
[630]	valid_0's auc: 0.635587
[640]	valid_0's auc: 0.635836
[650]	valid_0's auc: 0.635973
[660]	valid_0's auc: 0.636003
[670]	valid_0's auc: 0.636188
[680]	valid_0's auc: 0.636263
[690]	valid_0's auc: 0.636363
[700]	valid_0's auc: 0.636424
[710]	valid_0's auc: 0.636526
[720]	valid_0's auc: 0.636574
[730]	valid_0's auc: 0.636787
[740]	valid_0's auc: 0.63697
[750]	valid_0's auc: 0.637126
[760]	valid_0's auc: 0.637197
[770]	valid_0's auc: 0.637301
[780]	valid_0's auc: 0.637408
[790]	valid_0's auc: 0.637552
[800]	valid_0's auc: 0.637644
[810]	valid_0's auc: 0.63775
[820]	valid_0's auc: 0.637842
[830]	valid_0's auc: 0.637903
[840]	valid_0's auc: 0.638015
[850]	valid_0's auc: 0.638079
[860]	valid_0's auc: 0.638105
[870]	valid_0's auc: 0.638186
[880]	valid_0's auc: 0.63825
[890]	valid_0's auc: 0.638439
[900]	valid_0's auc: 0.638515
[910]	valid_0's auc: 0.638645
[920]	valid_0's auc: 0.638689
[930]	valid_0's auc: 0.638742
[940]	valid_0's auc: 0.638842
[950]	valid_0's auc: 0.638898
[960]	valid_0's auc: 0.638972
[970]	valid_0's auc: 0.639011
[980]	valid_0's auc: 0.639024
[990]	valid_0's auc: 0.639105
[1000]	valid_0's auc: 0.639177
[1010]	valid_0's auc: 0.639254
[1020]	valid_0's auc: 0.639288
[1030]	valid_0's auc: 0.639427
[1040]	valid_0's auc: 0.639429
[1050]	valid_0's auc: 0.639469
[1060]	valid_0's auc: 0.639529
[1070]	valid_0's auc: 0.639627
[1080]	valid_0's auc: 0.639691
[1090]	valid_0's auc: 0.639757
[1100]	valid_0's auc: 0.639792
[1110]	valid_0's auc: 0.63978
[1120]	valid_0's auc: 0.639817
[1130]	valid_0's auc: 0.639849
[1140]	valid_0's auc: 0.639918
[1150]	valid_0's auc: 0.639966
[1160]	valid_0's auc: 0.639978
[1170]	valid_0's auc: 0.640115
[1180]	valid_0's auc: 0.640119
[1190]	valid_0's auc: 0.640162
[1200]	valid_0's auc: 0.640211
[1210]	valid_0's auc: 0.640235
[1220]	valid_0's auc: 0.640266
[1230]	valid_0's auc: 0.640301
[1240]	valid_0's auc: 0.640351
[1250]	valid_0's auc: 0.640396
[1260]	valid_0's auc: 0.640477
[1270]	valid_0's auc: 0.640525
[1280]	valid_0's auc: 0.640536
[1290]	valid_0's auc: 0.640548
[1300]	valid_0's auc: 0.640651
[1310]	valid_0's auc: 0.640694
[1320]	valid_0's auc: 0.640709
[1330]	valid_0's auc: 0.640759
[1340]	valid_0's auc: 0.640748
[1350]	valid_0's auc: 0.640761
[1360]	valid_0's auc: 0.640742
[1370]	valid_0's auc: 0.640806
[1380]	valid_0's auc: 0.640882
[1390]	valid_0's auc: 0.640914
[1400]	valid_0's auc: 0.64091
[1410]	valid_0's auc: 0.641016
[1420]	valid_0's auc: 0.6411
[1430]	valid_0's auc: 0.641099
[1440]	valid_0's auc: 0.641166
[1450]	valid_0's auc: 0.641158
[1460]	valid_0's auc: 0.641164
[1470]	valid_0's auc: 0.641286
[1480]	valid_0's auc: 0.641328
[1490]	valid_0's auc: 0.641356
[1500]	valid_0's auc: 0.641362
[1510]	valid_0's auc: 0.641373
[1520]	valid_0's auc: 0.641414
[1530]	valid_0's auc: 0.641421
[1540]	valid_0's auc: 0.641477
[1550]	valid_0's auc: 0.641511
[1560]	valid_0's auc: 0.641505
[1570]	valid_0's auc: 0.641548
[1580]	valid_0's auc: 0.641573
[1590]	valid_0's auc: 0.641616
[1600]	valid_0's auc: 0.641634
[1610]	valid_0's auc: 0.641678
[1620]	valid_0's auc: 0.641713
[1630]	valid_0's auc: 0.641726
[1640]	valid_0's auc: 0.641764
[1650]	valid_0's auc: 0.641749
[1660]	valid_0's auc: 0.641799
[1670]	valid_0's auc: 0.641847
[1680]	valid_0's auc: 0.641888
[1690]	valid_0's auc: 0.641899
[1700]	valid_0's auc: 0.641939
[1710]	valid_0's auc: 0.642001
[1720]	valid_0's auc: 0.642028
[1730]	valid_0's auc: 0.642054
[1740]	valid_0's auc: 0.64207
[1750]	valid_0's auc: 0.64212
[1760]	valid_0's auc: 0.642145
[1770]	valid_0's auc: 0.642157
[1780]	valid_0's auc: 0.642173
[1790]	valid_0's auc: 0.642195
[1800]	valid_0's auc: 0.642275
[1810]	valid_0's auc: 0.642329
[1820]	valid_0's auc: 0.642326
[1830]	valid_0's auc: 0.642317
[1840]	valid_0's auc: 0.642341
[1850]	valid_0's auc: 0.642409
[1860]	valid_0's auc: 0.642424
[1870]	valid_0's auc: 0.642423
[1880]	valid_0's auc: 0.642436
[1890]	valid_0's auc: 0.642484
[1900]	valid_0's auc: 0.642464
[1910]	valid_0's auc: 0.642527
[1920]	valid_0's auc: 0.642529
[1930]	valid_0's auc: 0.64256
[1940]	valid_0's auc: 0.642563
[1950]	valid_0's auc: 0.642566
[1960]	valid_0's auc: 0.6427
[1970]	valid_0's auc: 0.642731
[1980]	valid_0's auc: 0.642741
[1990]	valid_0's auc: 0.642773
[2000]	valid_0's auc: 0.642819
[2010]	valid_0's auc: 0.642862
[2020]	valid_0's auc: 0.642902
[2030]	valid_0's auc: 0.642927
[2040]	valid_0's auc: 0.642986
[2050]	valid_0's auc: 0.643017
[2060]	valid_0's auc: 0.643012
[2070]	valid_0's auc: 0.643031
[2080]	valid_0's auc: 0.64305
[2090]	valid_0's auc: 0.643053
[2100]	valid_0's auc: 0.643093
[2110]	valid_0's auc: 0.643105
[2120]	valid_0's auc: 0.64313
[2130]	valid_0's auc: 0.643117
[2140]	valid_0's auc: 0.643254
[2150]	valid_0's auc: 0.643252
[2160]	valid_0's auc: 0.643302
[2170]	valid_0's auc: 0.643331
[2180]	valid_0's auc: 0.643357
[2190]	valid_0's auc: 0.64336
[2200]	valid_0's auc: 0.643381
[2210]	valid_0's auc: 0.643404
[2220]	valid_0's auc: 0.643419
[2230]	valid_0's auc: 0.64349
[2240]	valid_0's auc: 0.643483
[2250]	valid_0's auc: 0.643513
[2260]	valid_0's auc: 0.643548
[2270]	valid_0's auc: 0.643572
[2280]	valid_0's auc: 0.643552
[2290]	valid_0's auc: 0.64356
[2300]	valid_0's auc: 0.643578
[2310]	valid_0's auc: 0.643631
[2320]	valid_0's auc: 0.643652
[2330]	valid_0's auc: 0.643708
[2340]	valid_0's auc: 0.643688
[2350]	valid_0's auc: 0.643705
[2360]	valid_0's auc: 0.643721
[2370]	valid_0's auc: 0.643729
[2380]	valid_0's auc: 0.643739
[2390]	valid_0's auc: 0.643746
[2400]	valid_0's auc: 0.643816
[2410]	valid_0's auc: 0.643829
[2420]	valid_0's auc: 0.643841
[2430]	valid_0's auc: 0.643827
[2440]	valid_0's auc: 0.64385
[2450]	valid_0's auc: 0.643886
[2460]	valid_0's auc: 0.643925
[2470]	valid_0's auc: 0.643956
[2480]	valid_0's auc: 0.643972
[2490]	valid_0's auc: 0.643988
[2500]	valid_0's auc: 0.643988
[2510]	valid_0's auc: 0.644015
[2520]	valid_0's auc: 0.644029
[2530]	valid_0's auc: 0.644107
[2540]	valid_0's auc: 0.644112
[2550]	valid_0's auc: 0.644121
[2560]	valid_0's auc: 0.644139
[2570]	valid_0's auc: 0.644148
[2580]	valid_0's auc: 0.644153
[2590]	valid_0's auc: 0.644157
[2600]	valid_0's auc: 0.644169
[2610]	valid_0's auc: 0.644172
[2620]	valid_0's auc: 0.64421
[2630]	valid_0's auc: 0.644209
[2640]	valid_0's auc: 0.644212
[2650]	valid_0's auc: 0.644218
[2660]	valid_0's auc: 0.644219
[2670]	valid_0's auc: 0.644236
[2680]	valid_0's auc: 0.644283
[2690]	valid_0's auc: 0.644301
[2700]	valid_0's auc: 0.644352
[2710]	valid_0's auc: 0.644389
[2720]	valid_0's auc: 0.644387
[2730]	valid_0's auc: 0.644416
[2740]	valid_0's auc: 0.644394
[2750]	valid_0's auc: 0.644397
[2760]	valid_0's auc: 0.6444
[2770]	valid_0's auc: 0.644462
[2780]	valid_0's auc: 0.644475
[2790]	valid_0's auc: 0.644471
[2800]	valid_0's auc: 0.644511
[2810]	valid_0's auc: 0.644507
[2820]	valid_0's auc: 0.644521
[2830]	valid_0's auc: 0.644516
[2840]	valid_0's auc: 0.644541
[2850]	valid_0's auc: 0.644505
[2860]	valid_0's auc: 0.644506
[2870]	valid_0's auc: 0.644499
[2880]	valid_0's auc: 0.64453
[2890]	valid_0's auc: 0.64455
[2900]	valid_0's auc: 0.644577
[2910]	valid_0's auc: 0.644596
[2920]	valid_0's auc: 0.64463
[2930]	valid_0's auc: 0.644653
[2940]	valid_0's auc: 0.644666
[2950]	valid_0's auc: 0.644664
[2960]	valid_0's auc: 0.644674
[2970]	valid_0's auc: 0.644682
[2980]	valid_0's auc: 0.644686
[2990]	valid_0's auc: 0.644694
[3000]	valid_0's auc: 0.6447
[3010]	valid_0's auc: 0.644723
[3020]	valid_0's auc: 0.644736
[3030]	valid_0's auc: 0.644745
[3040]	valid_0's auc: 0.644763
[3050]	valid_0's auc: 0.644766
[3060]	valid_0's auc: 0.644778
[3070]	valid_0's auc: 0.64479
[3080]	valid_0's auc: 0.644806
[3090]	valid_0's auc: 0.644822
[3100]	valid_0's auc: 0.644846
[3110]	valid_0's auc: 0.644851
[3120]	valid_0's auc: 0.644892
[3130]	valid_0's auc: 0.64491
[3140]	valid_0's auc: 0.644905
[3150]	valid_0's auc: 0.644935
[3160]	valid_0's auc: 0.644938
[3170]	valid_0's auc: 0.644951
[3180]	valid_0's auc: 0.645026
[3190]	valid_0's auc: 0.645034
[3200]	valid_0's auc: 0.645053
[3210]	valid_0's auc: 0.645072
[3220]	valid_0's auc: 0.645061
[3230]	valid_0's auc: 0.645067
[3240]	valid_0's auc: 0.64508
[3250]	valid_0's auc: 0.645104
[3260]	valid_0's auc: 0.645171
[3270]	valid_0's auc: 0.645164
[3280]	valid_0's auc: 0.645168
[3290]	valid_0's auc: 0.645173
[3300]	valid_0's auc: 0.645196
[3310]	valid_0's auc: 0.645218
[3320]	valid_0's auc: 0.64522
[3330]	valid_0's auc: 0.645234
[3340]	valid_0's auc: 0.645236
[3350]	valid_0's auc: 0.645232
[3360]	valid_0's auc: 0.645251
[3370]	valid_0's auc: 0.645324
[3380]	valid_0's auc: 0.645451
[3390]	valid_0's auc: 0.64551
[3400]	valid_0's auc: 0.645553
[3410]	valid_0's auc: 0.645563
[3420]	valid_0's auc: 0.645603
[3430]	valid_0's auc: 0.645661
[3440]	valid_0's auc: 0.645697
[3450]	valid_0's auc: 0.645726
[3460]	valid_0's auc: 0.645723
[3470]	valid_0's auc: 0.645733
[3480]	valid_0's auc: 0.645753
[3490]	valid_0's auc: 0.645767
[3500]	valid_0's auc: 0.645771
[3510]	valid_0's auc: 0.645788
[3520]	valid_0's auc: 0.645806
[3530]	valid_0's auc: 0.645857
[3540]	valid_0's auc: 0.645857
[3550]	valid_0's auc: 0.645857
[3560]	valid_0's auc: 0.645869
[3570]	valid_0's auc: 0.645872
[3580]	valid_0's auc: 0.645858
[3590]	valid_0's auc: 0.645874
[3600]	valid_0's auc: 0.645884
[3610]	valid_0's auc: 0.645882
[3620]	valid_0's auc: 0.645919
[3630]	valid_0's auc: 0.645959
[3640]	valid_0's auc: 0.645971
[3650]	valid_0's auc: 0.646004
[3660]	valid_0's auc: 0.646015
[3670]	valid_0's auc: 0.646032
[3680]	valid_0's auc: 0.646057
[3690]	valid_0's auc: 0.646078
[3700]	valid_0's auc: 0.646108
[3710]	valid_0's auc: 0.64615
[3720]	valid_0's auc: 0.646268
[3730]	valid_0's auc: 0.64626
[3740]	valid_0's auc: 0.64632
[3750]	valid_0's auc: 0.646345
[3760]	valid_0's auc: 0.646354
[3770]	valid_0's auc: 0.646381
[3780]	valid_0's auc: 0.64638
[3790]	valid_0's auc: 0.646393
[3800]	valid_0's auc: 0.646384
[3810]	valid_0's auc: 0.646387
[3820]	valid_0's auc: 0.64643
[3830]	valid_0's auc: 0.646458
[3840]	valid_0's auc: 0.646509
[3850]	valid_0's auc: 0.64653
[3860]	valid_0's auc: 0.646542
[3870]	valid_0's auc: 0.646579
[3880]	valid_0's auc: 0.646615
[3890]	valid_0's auc: 0.646649
[3900]	valid_0's auc: 0.646668
[3910]	valid_0's auc: 0.646682
[3920]	valid_0's auc: 0.64668
[3930]	valid_0's auc: 0.646689
[3940]	valid_0's auc: 0.646685
[3950]	valid_0's auc: 0.646673
[3960]	valid_0's auc: 0.646698
[3970]	valid_0's auc: 0.646685
[3980]	valid_0's auc: 0.646689
[3990]	valid_0's auc: 0.646698
[4000]	valid_0's auc: 0.646711
[4010]	valid_0's auc: 0.646777
[4020]	valid_0's auc: 0.646833
[4030]	valid_0's auc: 0.646855
[4040]	valid_0's auc: 0.64686
[4050]	valid_0's auc: 0.646871
[4060]	valid_0's auc: 0.646867
[4070]	valid_0's auc: 0.646889
[4080]	valid_0's auc: 0.646906
[4090]	valid_0's auc: 0.646897
[4100]	valid_0's auc: 0.646906
[4110]	valid_0's auc: 0.64692
[4120]	valid_0's auc: 0.646921
[4130]	valid_0's auc: 0.646909
[4140]	valid_0's auc: 0.646913
[4150]	valid_0's auc: 0.646924
[4160]	valid_0's auc: 0.646933
[4170]	valid_0's auc: 0.646941
[4180]	valid_0's auc: 0.646956
[4190]	valid_0's auc: 0.646962
[4200]	valid_0's auc: 0.647005
[4210]	valid_0's auc: 0.647013
[4220]	valid_0's auc: 0.647026
[4230]	valid_0's auc: 0.647051
[4240]	valid_0's auc: 0.647077
[4250]	valid_0's auc: 0.647076
[4260]	valid_0's auc: 0.647089
[4270]	valid_0's auc: 0.647095
[4280]	valid_0's auc: 0.647093
[4290]	valid_0's auc: 0.647097
[4300]	valid_0's auc: 0.647087
[4310]	valid_0's auc: 0.647091
[4320]	valid_0's auc: 0.647102
[4330]	valid_0's auc: 0.647105
[4340]	valid_0's auc: 0.647038
[4350]	valid_0's auc: 0.647016
[4360]	valid_0's auc: 0.647083
[4370]	valid_0's auc: 0.647091
[4380]	valid_0's auc: 0.647105
[4390]	valid_0's auc: 0.647095
[4400]	valid_0's auc: 0.64711
[4410]	valid_0's auc: 0.64711
[4420]	valid_0's auc: 0.647114
[4430]	valid_0's auc: 0.647148
[4440]	valid_0's auc: 0.647132
[4450]	valid_0's auc: 0.647105
[4460]	valid_0's auc: 0.647088
[4470]	valid_0's auc: 0.647084
[4480]	valid_0's auc: 0.647093
Early stopping, best iteration is:
[4430]	valid_0's auc: 0.647148
best score: 0.647147867199
best iteration: 4430
complete on: source_type

--------------------
this is round: 5
city and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
--------------------


After selection:
target       uint8
city      category
msno      category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512901
[20]	valid_0's auc: 0.520766
[30]	valid_0's auc: 0.530283
[40]	valid_0's auc: 0.536522
[50]	valid_0's auc: 0.541237
[60]	valid_0's auc: 0.546524
[70]	valid_0's auc: 0.548044
[80]	valid_0's auc: 0.550761
[90]	valid_0's auc: 0.552566
[100]	valid_0's auc: 0.553913
[110]	valid_0's auc: 0.555739
[120]	valid_0's auc: 0.558807
[130]	valid_0's auc: 0.561101
[140]	valid_0's auc: 0.563936
[150]	valid_0's auc: 0.566154
[160]	valid_0's auc: 0.567574
[170]	valid_0's auc: 0.569187
[180]	valid_0's auc: 0.570523
[190]	valid_0's auc: 0.571747
[200]	valid_0's auc: 0.572331
[210]	valid_0's auc: 0.573219
[220]	valid_0's auc: 0.573934
[230]	valid_0's auc: 0.57455
[240]	valid_0's auc: 0.575106
[250]	valid_0's auc: 0.575781
[260]	valid_0's auc: 0.576115
[270]	valid_0's auc: 0.576623
[280]	valid_0's auc: 0.577276
[290]	valid_0's auc: 0.577687
[300]	valid_0's auc: 0.577603
[310]	valid_0's auc: 0.578142
[320]	valid_0's auc: 0.578705
[330]	valid_0's auc: 0.579098
[340]	valid_0's auc: 0.579607
[350]	valid_0's auc: 0.58007
[360]	valid_0's auc: 0.580371
[370]	valid_0's auc: 0.58048
[380]	valid_0's auc: 0.58065
[390]	valid_0's auc: 0.581215
[400]	valid_0's auc: 0.582283
[410]	valid_0's auc: 0.582729
[420]	valid_0's auc: 0.583352
[430]	valid_0's auc: 0.584004
[440]	valid_0's auc: 0.584476
[450]	valid_0's auc: 0.585022
[460]	valid_0's auc: 0.585699
[470]	valid_0's auc: 0.585945
[480]	valid_0's auc: 0.58645
[490]	valid_0's auc: 0.587067
[500]	valid_0's auc: 0.587536
[510]	valid_0's auc: 0.587836
[520]	valid_0's auc: 0.588179
[530]	valid_0's auc: 0.588146
[540]	valid_0's auc: 0.588375
[550]	valid_0's auc: 0.588586
[560]	valid_0's auc: 0.588793
[570]	valid_0's auc: 0.58898
[580]	valid_0's auc: 0.589234
[590]	valid_0's auc: 0.589365
[600]	valid_0's auc: 0.589441
[610]	valid_0's auc: 0.589596
[620]	valid_0's auc: 0.589824
[630]	valid_0's auc: 0.589933
[640]	valid_0's auc: 0.589974
[650]	valid_0's auc: 0.5902
[660]	valid_0's auc: 0.59055
[670]	valid_0's auc: 0.59076
[680]	valid_0's auc: 0.590859
[690]	valid_0's auc: 0.591125
[700]	valid_0's auc: 0.591366
[710]	valid_0's auc: 0.591485
[720]	valid_0's auc: 0.591588
[730]	valid_0's auc: 0.591737
[740]	valid_0's auc: 0.591869
[750]	valid_0's auc: 0.591971
[760]	valid_0's auc: 0.592042
[770]	valid_0's auc: 0.591917
[780]	valid_0's auc: 0.592095
[790]	valid_0's auc: 0.592205
[800]	valid_0's auc: 0.592331
[810]	valid_0's auc: 0.592396
[820]	valid_0's auc: 0.592458
[830]	valid_0's auc: 0.592605
[840]	valid_0's auc: 0.592704
[850]	valid_0's auc: 0.592683
[860]	valid_0's auc: 0.592729
[870]	valid_0's auc: 0.592792
[880]	valid_0's auc: 0.592826
[890]	valid_0's auc: 0.592914
[900]	valid_0's auc: 0.592987
[910]	valid_0's auc: 0.593004
[920]	valid_0's auc: 0.592989
[930]	valid_0's auc: 0.593013
[940]	valid_0's auc: 0.593052
[950]	valid_0's auc: 0.593124
[960]	valid_0's auc: 0.593097
[970]	valid_0's auc: 0.593136
[980]	valid_0's auc: 0.593269
[990]	valid_0's auc: 0.593338
[1000]	valid_0's auc: 0.593379
[1010]	valid_0's auc: 0.593377
[1020]	valid_0's auc: 0.593433
[1030]	valid_0's auc: 0.593493
[1040]	valid_0's auc: 0.593515
[1050]	valid_0's auc: 0.593595
[1060]	valid_0's auc: 0.593591
[1070]	valid_0's auc: 0.593602
[1080]	valid_0's auc: 0.593582
[1090]	valid_0's auc: 0.59362
[1100]	valid_0's auc: 0.593847
[1110]	valid_0's auc: 0.593908
[1120]	valid_0's auc: 0.593912
[1130]	valid_0's auc: 0.593962
[1140]	valid_0's auc: 0.593977
[1150]	valid_0's auc: 0.594018
[1160]	valid_0's auc: 0.594013
[1170]	valid_0's auc: 0.593973
[1180]	valid_0's auc: 0.594031
[1190]	valid_0's auc: 0.594062
[1200]	valid_0's auc: 0.594074
[1210]	valid_0's auc: 0.594081
[1220]	valid_0's auc: 0.594156
[1230]	valid_0's auc: 0.594143
[1240]	valid_0's auc: 0.594148
[1250]	valid_0's auc: 0.594144
[1260]	valid_0's auc: 0.594136
[1270]	valid_0's auc: 0.594127
[1280]	valid_0's auc: 0.594206
[1290]	valid_0's auc: 0.594214
[1300]	valid_0's auc: 0.594276
[1310]	valid_0's auc: 0.594293
[1320]	valid_0's auc: 0.594286
[1330]	valid_0's auc: 0.594387
[1340]	valid_0's auc: 0.59447
[1350]	valid_0's auc: 0.594492
[1360]	valid_0's auc: 0.594504
[1370]	valid_0's auc: 0.594496
[1380]	valid_0's auc: 0.594475
[1390]	valid_0's auc: 0.594511
[1400]	valid_0's auc: 0.594509
[1410]	valid_0's auc: 0.594507
[1420]	valid_0's auc: 0.594455
[1430]	valid_0's auc: 0.594524
[1440]	valid_0's auc: 0.594529
[1450]	valid_0's auc: 0.594541
[1460]	valid_0's auc: 0.594553
[1470]	valid_0's auc: 0.594529
[1480]	valid_0's auc: 0.594507
[1490]	valid_0's auc: 0.594549
[1500]	valid_0's auc: 0.59454
[1510]	valid_0's auc: 0.594568
[1520]	valid_0's auc: 0.594569
[1530]	valid_0's auc: 0.59455
[1540]	valid_0's auc: 0.59452
[1550]	valid_0's auc: 0.594539
[1560]	valid_0's auc: 0.59456
Early stopping, best iteration is:
[1517]	valid_0's auc: 0.594587
best score: 0.594587253355
best iteration: 1517
complete on: city

--------------------
this is round: 6
registered_via and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
--------------------


After selection:
target               uint8
registered_via    category
msno              category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.51985
[20]	valid_0's auc: 0.52627
[30]	valid_0's auc: 0.531627
[40]	valid_0's auc: 0.537163
[50]	valid_0's auc: 0.540888
[60]	valid_0's auc: 0.543741
[70]	valid_0's auc: 0.54731
[80]	valid_0's auc: 0.550761
[90]	valid_0's auc: 0.553225
[100]	valid_0's auc: 0.556146
[110]	valid_0's auc: 0.557475
[120]	valid_0's auc: 0.559142
[130]	valid_0's auc: 0.560457
[140]	valid_0's auc: 0.561559
[150]	valid_0's auc: 0.562952
[160]	valid_0's auc: 0.564372
[170]	valid_0's auc: 0.565722
[180]	valid_0's auc: 0.566509
[190]	valid_0's auc: 0.567212
[200]	valid_0's auc: 0.567736
[210]	valid_0's auc: 0.568634
[220]	valid_0's auc: 0.570245
[230]	valid_0's auc: 0.571791
[240]	valid_0's auc: 0.573453
[250]	valid_0's auc: 0.575029
[260]	valid_0's auc: 0.575559
[270]	valid_0's auc: 0.576396
[280]	valid_0's auc: 0.577817
[290]	valid_0's auc: 0.578538
[300]	valid_0's auc: 0.579544
[310]	valid_0's auc: 0.580633
[320]	valid_0's auc: 0.581091
[330]	valid_0's auc: 0.581696
[340]	valid_0's auc: 0.582499
[350]	valid_0's auc: 0.58285
[360]	valid_0's auc: 0.583417
[370]	valid_0's auc: 0.584104
[380]	valid_0's auc: 0.584236
[390]	valid_0's auc: 0.584609
[400]	valid_0's auc: 0.584832
[410]	valid_0's auc: 0.585071
[420]	valid_0's auc: 0.585487
[430]	valid_0's auc: 0.585856
[440]	valid_0's auc: 0.586502
[450]	valid_0's auc: 0.586692
[460]	valid_0's auc: 0.58691
[470]	valid_0's auc: 0.58708
[480]	valid_0's auc: 0.58724
[490]	valid_0's auc: 0.58746
[500]	valid_0's auc: 0.587672
[510]	valid_0's auc: 0.587996
[520]	valid_0's auc: 0.58822
[530]	valid_0's auc: 0.588543
[540]	valid_0's auc: 0.588825
[550]	valid_0's auc: 0.588887
[560]	valid_0's auc: 0.589206
[570]	valid_0's auc: 0.589497
[580]	valid_0's auc: 0.589545
[590]	valid_0's auc: 0.589544
[600]	valid_0's auc: 0.589829
[610]	valid_0's auc: 0.589866
[620]	valid_0's auc: 0.590319
[630]	valid_0's auc: 0.590644
[640]	valid_0's auc: 0.5907
[650]	valid_0's auc: 0.590998
[660]	valid_0's auc: 0.591315
[670]	valid_0's auc: 0.591434
[680]	valid_0's auc: 0.591637
[690]	valid_0's auc: 0.591644
[700]	valid_0's auc: 0.591357
[710]	valid_0's auc: 0.591419
[720]	valid_0's auc: 0.591512
[730]	valid_0's auc: 0.591641
[740]	valid_0's auc: 0.591718
Early stopping, best iteration is:
[698]	valid_0's auc: 0.591838
best score: 0.591837639097
best iteration: 698
complete on: registered_via

--------------------
this is round: 7
sex and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
--------------------


After selection:
target       uint8
sex       category
msno      category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512897
[20]	valid_0's auc: 0.521372
[30]	valid_0's auc: 0.526075
[40]	valid_0's auc: 0.530775
[50]	valid_0's auc: 0.534579
[60]	valid_0's auc: 0.542288
[70]	valid_0's auc: 0.547338
[80]	valid_0's auc: 0.550522
[90]	valid_0's auc: 0.552883
[100]	valid_0's auc: 0.55494
[110]	valid_0's auc: 0.556617
[120]	valid_0's auc: 0.559471
[130]	valid_0's auc: 0.562105
[140]	valid_0's auc: 0.564983
[150]	valid_0's auc: 0.566619
[160]	valid_0's auc: 0.568434
[170]	valid_0's auc: 0.569318
[180]	valid_0's auc: 0.570414
[190]	valid_0's auc: 0.571706
[200]	valid_0's auc: 0.572344
[210]	valid_0's auc: 0.573281
[220]	valid_0's auc: 0.574049
[230]	valid_0's auc: 0.575057
[240]	valid_0's auc: 0.575164
[250]	valid_0's auc: 0.576277
[260]	valid_0's auc: 0.576756
[270]	valid_0's auc: 0.577286
[280]	valid_0's auc: 0.577863
[290]	valid_0's auc: 0.578802
[300]	valid_0's auc: 0.579085
[310]	valid_0's auc: 0.579602
[320]	valid_0's auc: 0.579898
[330]	valid_0's auc: 0.580603
[340]	valid_0's auc: 0.580807
[350]	valid_0's auc: 0.580888
[360]	valid_0's auc: 0.5814
[370]	valid_0's auc: 0.58163
[380]	valid_0's auc: 0.582235
[390]	valid_0's auc: 0.583098
[400]	valid_0's auc: 0.584077
[410]	valid_0's auc: 0.584778
[420]	valid_0's auc: 0.585287
[430]	valid_0's auc: 0.585797
[440]	valid_0's auc: 0.585156
[450]	valid_0's auc: 0.585476
[460]	valid_0's auc: 0.586188
[470]	valid_0's auc: 0.586611
[480]	valid_0's auc: 0.58697
[490]	valid_0's auc: 0.587474
[500]	valid_0's auc: 0.588007
[510]	valid_0's auc: 0.588542
[520]	valid_0's auc: 0.588754
[530]	valid_0's auc: 0.588987
[540]	valid_0's auc: 0.589273
[550]	valid_0's auc: 0.589496
[560]	valid_0's auc: 0.589646
[570]	valid_0's auc: 0.589723
[580]	valid_0's auc: 0.58981
[590]	valid_0's auc: 0.589861
[600]	valid_0's auc: 0.589997
[610]	valid_0's auc: 0.590035
[620]	valid_0's auc: 0.590241
[630]	valid_0's auc: 0.590583
[640]	valid_0's auc: 0.59092
[650]	valid_0's auc: 0.591079
[660]	valid_0's auc: 0.591262
[670]	valid_0's auc: 0.591593
[680]	valid_0's auc: 0.59192
[690]	valid_0's auc: 0.591976
[700]	valid_0's auc: 0.592089
[710]	valid_0's auc: 0.592255
[720]	valid_0's auc: 0.592381
[730]	valid_0's auc: 0.592393
[740]	valid_0's auc: 0.592514
[750]	valid_0's auc: 0.592621
[760]	valid_0's auc: 0.592729
[770]	valid_0's auc: 0.592915
[780]	valid_0's auc: 0.592974
[790]	valid_0's auc: 0.592947
[800]	valid_0's auc: 0.593181
[810]	valid_0's auc: 0.593157
[820]	valid_0's auc: 0.593277
[830]	valid_0's auc: 0.593351
[840]	valid_0's auc: 0.593557
[850]	valid_0's auc: 0.593635
[860]	valid_0's auc: 0.593542
[870]	valid_0's auc: 0.5937
[880]	valid_0's auc: 0.59369
[890]	valid_0's auc: 0.593735
[900]	valid_0's auc: 0.593718
[910]	valid_0's auc: 0.593745
[920]	valid_0's auc: 0.593867
[930]	valid_0's auc: 0.59394
[940]	valid_0's auc: 0.593952
[950]	valid_0's auc: 0.594126
[960]	valid_0's auc: 0.594168
[970]	valid_0's auc: 0.594165
[980]	valid_0's auc: 0.594186
[990]	valid_0's auc: 0.594223
[1000]	valid_0's auc: 0.594282
[1010]	valid_0's auc: 0.59425
[1020]	valid_0's auc: 0.594271
[1030]	valid_0's auc: 0.594277
[1040]	valid_0's auc: 0.594271
[1050]	valid_0's auc: 0.594303
[1060]	valid_0's auc: 0.594311
[1070]	valid_0's auc: 0.594344
[1080]	valid_0's auc: 0.594344
[1090]	valid_0's auc: 0.594408
[1100]	valid_0's auc: 0.594458
[1110]	valid_0's auc: 0.594455
[1120]	valid_0's auc: 0.594538
[1130]	valid_0's auc: 0.59467
[1140]	valid_0's auc: 0.594668
[1150]	valid_0's auc: 0.594733
[1160]	valid_0's auc: 0.594656
[1170]	valid_0's auc: 0.594704
[1180]	valid_0's auc: 0.594783
[1190]	valid_0's auc: 0.594776
[1200]	valid_0's auc: 0.594839
[1210]	valid_0's auc: 0.594828
[1220]	valid_0's auc: 0.594884
[1230]	valid_0's auc: 0.594874
[1240]	valid_0's auc: 0.594896
[1250]	valid_0's auc: 0.594865
[1260]	valid_0's auc: 0.594911
[1270]	valid_0's auc: 0.594938
[1280]	valid_0's auc: 0.594905
[1290]	valid_0's auc: 0.594972
[1300]	valid_0's auc: 0.594984
[1310]	valid_0's auc: 0.594975
[1320]	valid_0's auc: 0.594979
[1330]	valid_0's auc: 0.594979
[1340]	valid_0's auc: 0.594996
[1350]	valid_0's auc: 0.595065
[1360]	valid_0's auc: 0.595017
[1370]	valid_0's auc: 0.595056
[1380]	valid_0's auc: 0.595107
[1390]	valid_0's auc: 0.595133
[1400]	valid_0's auc: 0.595257
[1410]	valid_0's auc: 0.59528
[1420]	valid_0's auc: 0.595285
[1430]	valid_0's auc: 0.595277
[1440]	valid_0's auc: 0.595296
[1450]	valid_0's auc: 0.595289
[1460]	valid_0's auc: 0.59529
[1470]	valid_0's auc: 0.595379
[1480]	valid_0's auc: 0.595356
[1490]	valid_0's auc: 0.595414
[1500]	valid_0's auc: 0.595336
[1510]	valid_0's auc: 0.595358
[1520]	valid_0's auc: 0.5954
[1530]	valid_0's auc: 0.595394
Early stopping, best iteration is:
[1489]	valid_0's auc: 0.595417
best score: 0.595416508905
best iteration: 1489
complete on: sex

--------------------
this is round: 8
sex_guess1 and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
--------------------


After selection:
target           uint8
sex_guess1    category
msno          category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512878
[20]	valid_0's auc: 0.521076
[30]	valid_0's auc: 0.525717
[40]	valid_0's auc: 0.530252
[50]	valid_0's auc: 0.534257
[60]	valid_0's auc: 0.538703
[70]	valid_0's auc: 0.543309
[80]	valid_0's auc: 0.546263
[90]	valid_0's auc: 0.548487
[100]	valid_0's auc: 0.55016
[110]	valid_0's auc: 0.552395
[120]	valid_0's auc: 0.553783
[130]	valid_0's auc: 0.555054
[140]	valid_0's auc: 0.556502
[150]	valid_0's auc: 0.557904
[160]	valid_0's auc: 0.558692
[170]	valid_0's auc: 0.559992
[180]	valid_0's auc: 0.560995
[190]	valid_0's auc: 0.561972
[200]	valid_0's auc: 0.563074
[210]	valid_0's auc: 0.565491
[220]	valid_0's auc: 0.567221
[230]	valid_0's auc: 0.568682
[240]	valid_0's auc: 0.570047
[250]	valid_0's auc: 0.571221
[260]	valid_0's auc: 0.572091
[270]	valid_0's auc: 0.573071
[280]	valid_0's auc: 0.574192
[290]	valid_0's auc: 0.574727
[300]	valid_0's auc: 0.575921
[310]	valid_0's auc: 0.576363
[320]	valid_0's auc: 0.576696
[330]	valid_0's auc: 0.577079
[340]	valid_0's auc: 0.577929
[350]	valid_0's auc: 0.57849
[360]	valid_0's auc: 0.578661
[370]	valid_0's auc: 0.579242
[380]	valid_0's auc: 0.579655
[390]	valid_0's auc: 0.579781
[400]	valid_0's auc: 0.580672
[410]	valid_0's auc: 0.581337
[420]	valid_0's auc: 0.581721
[430]	valid_0's auc: 0.581974
[440]	valid_0's auc: 0.582142
[450]	valid_0's auc: 0.582294
[460]	valid_0's auc: 0.582615
[470]	valid_0's auc: 0.583057
[480]	valid_0's auc: 0.583314
[490]	valid_0's auc: 0.583688
[500]	valid_0's auc: 0.584041
[510]	valid_0's auc: 0.584462
[520]	valid_0's auc: 0.584819
[530]	valid_0's auc: 0.585107
[540]	valid_0's auc: 0.585197
[550]	valid_0's auc: 0.585342
[560]	valid_0's auc: 0.585507
[570]	valid_0's auc: 0.585773
[580]	valid_0's auc: 0.586047
[590]	valid_0's auc: 0.586266
[600]	valid_0's auc: 0.58637
[610]	valid_0's auc: 0.586421
[620]	valid_0's auc: 0.586573
[630]	valid_0's auc: 0.587055
[640]	valid_0's auc: 0.587443
[650]	valid_0's auc: 0.587905
[660]	valid_0's auc: 0.588176
[670]	valid_0's auc: 0.588364
[680]	valid_0's auc: 0.588583
[690]	valid_0's auc: 0.588834
[700]	valid_0's auc: 0.589057
[710]	valid_0's auc: 0.589176
[720]	valid_0's auc: 0.589251
[730]	valid_0's auc: 0.589186
[740]	valid_0's auc: 0.589307
[750]	valid_0's auc: 0.589443
[760]	valid_0's auc: 0.589481
[770]	valid_0's auc: 0.589658
[780]	valid_0's auc: 0.589655
[790]	valid_0's auc: 0.589705
[800]	valid_0's auc: 0.589834
[810]	valid_0's auc: 0.58987
[820]	valid_0's auc: 0.590051
[830]	valid_0's auc: 0.590024
[840]	valid_0's auc: 0.590114
[850]	valid_0's auc: 0.590232
[860]	valid_0's auc: 0.590262
[870]	valid_0's auc: 0.590321
[880]	valid_0's auc: 0.590354
[890]	valid_0's auc: 0.590468
[900]	valid_0's auc: 0.590577
[910]	valid_0's auc: 0.590597
[920]	valid_0's auc: 0.59059
[930]	valid_0's auc: 0.590581
[940]	valid_0's auc: 0.590815
[950]	valid_0's auc: 0.590886
[960]	valid_0's auc: 0.590852
[970]	valid_0's auc: 0.590898
[980]	valid_0's auc: 0.590888
[990]	valid_0's auc: 0.590881
[1000]	valid_0's auc: 0.590964
[1010]	valid_0's auc: 0.590966
[1020]	valid_0's auc: 0.591121
[1030]	valid_0's auc: 0.591434
[1040]	valid_0's auc: 0.591453
[1050]	valid_0's auc: 0.591512
[1060]	valid_0's auc: 0.591574
[1070]	valid_0's auc: 0.591671
[1080]	valid_0's auc: 0.591579
[1090]	valid_0's auc: 0.591569
[1100]	valid_0's auc: 0.591625
[1110]	valid_0's auc: 0.591715
[1120]	valid_0's auc: 0.591753
[1130]	valid_0's auc: 0.591842
[1140]	valid_0's auc: 0.591864
[1150]	valid_0's auc: 0.59184
[1160]	valid_0's auc: 0.591811
[1170]	valid_0's auc: 0.591786
[1180]	valid_0's auc: 0.59183
[1190]	valid_0's auc: 0.591902
[1200]	valid_0's auc: 0.591954
[1210]	valid_0's auc: 0.591955
[1220]	valid_0's auc: 0.591949
[1230]	valid_0's auc: 0.591964
[1240]	valid_0's auc: 0.59201
[1250]	valid_0's auc: 0.591869
[1260]	valid_0's auc: 0.591935
[1270]	valid_0's auc: 0.591995
[1280]	valid_0's auc: 0.592017
[1290]	valid_0's auc: 0.59202
[1300]	valid_0's auc: 0.591988
[1310]	valid_0's auc: 0.592016
[1320]	valid_0's auc: 0.59207
[1330]	valid_0's auc: 0.592138
[1340]	valid_0's auc: 0.592235
[1350]	valid_0's auc: 0.592268
[1360]	valid_0's auc: 0.592268
[1370]	valid_0's auc: 0.592263
[1380]	valid_0's auc: 0.59229
[1390]	valid_0's auc: 0.592272
[1400]	valid_0's auc: 0.592273
[1410]	valid_0's auc: 0.592295
[1420]	valid_0's auc: 0.592289
[1430]	valid_0's auc: 0.592294
[1440]	valid_0's auc: 0.592344
[1450]	valid_0's auc: 0.592362
[1460]	valid_0's auc: 0.592501
[1470]	valid_0's auc: 0.592559
[1480]	valid_0's auc: 0.592611
[1490]	valid_0's auc: 0.592605
[1500]	valid_0's auc: 0.592645
[1510]	valid_0's auc: 0.592609
[1520]	valid_0's auc: 0.592695
[1530]	valid_0's auc: 0.592743
[1540]	valid_0's auc: 0.592755
[1550]	valid_0's auc: 0.592829
[1560]	valid_0's auc: 0.592862
[1570]	valid_0's auc: 0.592837
[1580]	valid_0's auc: 0.592813
[1590]	valid_0's auc: 0.592792
[1600]	valid_0's auc: 0.592819
Early stopping, best iteration is:
[1559]	valid_0's auc: 0.59287
best score: 0.592869533619
best iteration: 1559
complete on: sex_guess1

--------------------
this is round: 9
sex_guess2 and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
--------------------


After selection:
target           uint8
sex_guess2    category
msno          category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513789
[20]	valid_0's auc: 0.521417
[30]	valid_0's auc: 0.525444
[40]	valid_0's auc: 0.530118
[50]	valid_0's auc: 0.534637
[60]	valid_0's auc: 0.53989
[70]	valid_0's auc: 0.542449
[80]	valid_0's auc: 0.544054
[90]	valid_0's auc: 0.54657
[100]	valid_0's auc: 0.54864
[110]	valid_0's auc: 0.551505
[120]	valid_0's auc: 0.553867
[130]	valid_0's auc: 0.556976
[140]	valid_0's auc: 0.559149
[150]	valid_0's auc: 0.560926
[160]	valid_0's auc: 0.561811
[170]	valid_0's auc: 0.563096
[180]	valid_0's auc: 0.564481
[190]	valid_0's auc: 0.565544
[200]	valid_0's auc: 0.566742
[210]	valid_0's auc: 0.567467
[220]	valid_0's auc: 0.568317
[230]	valid_0's auc: 0.569143
[240]	valid_0's auc: 0.57
[250]	valid_0's auc: 0.571106
[260]	valid_0's auc: 0.571421
[270]	valid_0's auc: 0.572048
[280]	valid_0's auc: 0.572652
[290]	valid_0's auc: 0.573032
[300]	valid_0's auc: 0.574073
[310]	valid_0's auc: 0.574945
[320]	valid_0's auc: 0.575311
[330]	valid_0's auc: 0.575586
[340]	valid_0's auc: 0.575794
[350]	valid_0's auc: 0.576447
[360]	valid_0's auc: 0.57757
[370]	valid_0's auc: 0.578146
[380]	valid_0's auc: 0.57867
[390]	valid_0's auc: 0.579336
[400]	valid_0's auc: 0.580168
[410]	valid_0's auc: 0.580679
[420]	valid_0's auc: 0.58149
[430]	valid_0's auc: 0.582002
[440]	valid_0's auc: 0.582247
[450]	valid_0's auc: 0.582596
[460]	valid_0's auc: 0.583326
[470]	valid_0's auc: 0.583945
[480]	valid_0's auc: 0.584399
[490]	valid_0's auc: 0.584805
[500]	valid_0's auc: 0.585137
[510]	valid_0's auc: 0.585392
[520]	valid_0's auc: 0.585578
[530]	valid_0's auc: 0.585868
[540]	valid_0's auc: 0.585898
[550]	valid_0's auc: 0.585975
[560]	valid_0's auc: 0.586128
[570]	valid_0's auc: 0.586153
[580]	valid_0's auc: 0.586466
[590]	valid_0's auc: 0.586836
[600]	valid_0's auc: 0.58701
[610]	valid_0's auc: 0.587302
[620]	valid_0's auc: 0.587594
[630]	valid_0's auc: 0.587816
[640]	valid_0's auc: 0.588073
[650]	valid_0's auc: 0.588328
[660]	valid_0's auc: 0.588508
[670]	valid_0's auc: 0.588472
[680]	valid_0's auc: 0.588627
[690]	valid_0's auc: 0.588861
[700]	valid_0's auc: 0.589103
[710]	valid_0's auc: 0.589291
[720]	valid_0's auc: 0.589349
[730]	valid_0's auc: 0.589506
[740]	valid_0's auc: 0.589679
[750]	valid_0's auc: 0.589772
[760]	valid_0's auc: 0.589875
[770]	valid_0's auc: 0.589895
[780]	valid_0's auc: 0.590108
[790]	valid_0's auc: 0.590225
[800]	valid_0's auc: 0.590192
[810]	valid_0's auc: 0.590207
[820]	valid_0's auc: 0.59021
[830]	valid_0's auc: 0.590202
[840]	valid_0's auc: 0.590325
[850]	valid_0's auc: 0.590325
[860]	valid_0's auc: 0.590715
[870]	valid_0's auc: 0.590791
[880]	valid_0's auc: 0.59085
[890]	valid_0's auc: 0.590918
[900]	valid_0's auc: 0.590962
[910]	valid_0's auc: 0.590901
[920]	valid_0's auc: 0.59097
[930]	valid_0's auc: 0.590979
[940]	valid_0's auc: 0.590987
[950]	valid_0's auc: 0.591136
[960]	valid_0's auc: 0.59111
[970]	valid_0's auc: 0.591127
[980]	valid_0's auc: 0.591241
[990]	valid_0's auc: 0.591337
[1000]	valid_0's auc: 0.591302
[1010]	valid_0's auc: 0.591341
[1020]	valid_0's auc: 0.591523
[1030]	valid_0's auc: 0.59153
[1040]	valid_0's auc: 0.591539
[1050]	valid_0's auc: 0.591635
[1060]	valid_0's auc: 0.591627
[1070]	valid_0's auc: 0.591687
[1080]	valid_0's auc: 0.591795
[1090]	valid_0's auc: 0.591811
[1100]	valid_0's auc: 0.591852
[1110]	valid_0's auc: 0.591894
[1120]	valid_0's auc: 0.591973
[1130]	valid_0's auc: 0.592057
[1140]	valid_0's auc: 0.592124
[1150]	valid_0's auc: 0.592151
[1160]	valid_0's auc: 0.592176
[1170]	valid_0's auc: 0.59221
[1180]	valid_0's auc: 0.592285
[1190]	valid_0's auc: 0.592386
[1200]	valid_0's auc: 0.592452
[1210]	valid_0's auc: 0.592376
[1220]	valid_0's auc: 0.59238
[1230]	valid_0's auc: 0.592385
[1240]	valid_0's auc: 0.592477
[1250]	valid_0's auc: 0.592515
[1260]	valid_0's auc: 0.592562
[1270]	valid_0's auc: 0.592552
[1280]	valid_0's auc: 0.592565
[1290]	valid_0's auc: 0.59252
[1300]	valid_0's auc: 0.592516
[1310]	valid_0's auc: 0.592553
[1320]	valid_0's auc: 0.592636
[1330]	valid_0's auc: 0.59264
[1340]	valid_0's auc: 0.592678
[1350]	valid_0's auc: 0.59267
[1360]	valid_0's auc: 0.59267
[1370]	valid_0's auc: 0.592656
[1380]	valid_0's auc: 0.592688
[1390]	valid_0's auc: 0.592638
[1400]	valid_0's auc: 0.592658
[1410]	valid_0's auc: 0.592716
[1420]	valid_0's auc: 0.592736
[1430]	valid_0's auc: 0.59274
[1440]	valid_0's auc: 0.592717
[1450]	valid_0's auc: 0.592719
[1460]	valid_0's auc: 0.592769
[1470]	valid_0's auc: 0.592813
[1480]	valid_0's auc: 0.592802
[1490]	valid_0's auc: 0.592953
[1500]	valid_0's auc: 0.592962
[1510]	valid_0's auc: 0.592989
[1520]	valid_0's auc: 0.592975
[1530]	valid_0's auc: 0.592989
[1540]	valid_0's auc: 0.593017
[1550]	valid_0's auc: 0.592989
[1560]	valid_0's auc: 0.59302
[1570]	valid_0's auc: 0.593087
[1580]	valid_0's auc: 0.59311
[1590]	valid_0's auc: 0.593148
[1600]	valid_0's auc: 0.593153
[1610]	valid_0's auc: 0.593169
[1620]	valid_0's auc: 0.593178
[1630]	valid_0's auc: 0.593214
[1640]	valid_0's auc: 0.593234
[1650]	valid_0's auc: 0.593244
[1660]	valid_0's auc: 0.593222
[1670]	valid_0's auc: 0.593224
[1680]	valid_0's auc: 0.593212
[1690]	valid_0's auc: 0.593212
[1700]	valid_0's auc: 0.593215
Early stopping, best iteration is:
[1651]	valid_0's auc: 0.593245
best score: 0.593244822829
best iteration: 1651
complete on: sex_guess2

--------------------
this is round: 10
sex_guess3 and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
--------------------


After selection:
target           uint8
sex_guess3    category
msno          category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513678
[20]	valid_0's auc: 0.521133
[30]	valid_0's auc: 0.526008
[40]	valid_0's auc: 0.530094
[50]	valid_0's auc: 0.533961
[60]	valid_0's auc: 0.538471
[70]	valid_0's auc: 0.543117
[80]	valid_0's auc: 0.546221
[90]	valid_0's auc: 0.54795
[100]	valid_0's auc: 0.550008
[110]	valid_0's auc: 0.551743
[120]	valid_0's auc: 0.552716
[130]	valid_0's auc: 0.554349
[140]	valid_0's auc: 0.555752
[150]	valid_0's auc: 0.557193
[160]	valid_0's auc: 0.558306
[170]	valid_0's auc: 0.559529
[180]	valid_0's auc: 0.560111
[190]	valid_0's auc: 0.561404
[200]	valid_0's auc: 0.562622
[210]	valid_0's auc: 0.564011
[220]	valid_0's auc: 0.56621
[230]	valid_0's auc: 0.568071
[240]	valid_0's auc: 0.569539
[250]	valid_0's auc: 0.571077
[260]	valid_0's auc: 0.57232
[270]	valid_0's auc: 0.573127
[280]	valid_0's auc: 0.573748
[290]	valid_0's auc: 0.574803
[300]	valid_0's auc: 0.575429
[310]	valid_0's auc: 0.576086
[320]	valid_0's auc: 0.577075
[330]	valid_0's auc: 0.577748
[340]	valid_0's auc: 0.578068
[350]	valid_0's auc: 0.578701
[360]	valid_0's auc: 0.57931
[370]	valid_0's auc: 0.579751
[380]	valid_0's auc: 0.580019
[390]	valid_0's auc: 0.580587
[400]	valid_0's auc: 0.580921
[410]	valid_0's auc: 0.581265
[420]	valid_0's auc: 0.581524
[430]	valid_0's auc: 0.582289
[440]	valid_0's auc: 0.583065
[450]	valid_0's auc: 0.583173
[460]	valid_0's auc: 0.583457
[470]	valid_0's auc: 0.583655
[480]	valid_0's auc: 0.583743
[490]	valid_0's auc: 0.584074
[500]	valid_0's auc: 0.584514
[510]	valid_0's auc: 0.584713
[520]	valid_0's auc: 0.585242
[530]	valid_0's auc: 0.585469
[540]	valid_0's auc: 0.585827
[550]	valid_0's auc: 0.58611
[560]	valid_0's auc: 0.586416
[570]	valid_0's auc: 0.586511
[580]	valid_0's auc: 0.58661
[590]	valid_0's auc: 0.586846
[600]	valid_0's auc: 0.587005
[610]	valid_0's auc: 0.587324
[620]	valid_0's auc: 0.587456
[630]	valid_0's auc: 0.587609
[640]	valid_0's auc: 0.587836
[650]	valid_0's auc: 0.587966
[660]	valid_0's auc: 0.588096
[670]	valid_0's auc: 0.58815
[680]	valid_0's auc: 0.588302
[690]	valid_0's auc: 0.588364
[700]	valid_0's auc: 0.588487
[710]	valid_0's auc: 0.588464
[720]	valid_0's auc: 0.588664
[730]	valid_0's auc: 0.589045
[740]	valid_0's auc: 0.5893
[750]	valid_0's auc: 0.589482
[760]	valid_0's auc: 0.589808
[770]	valid_0's auc: 0.589915
[780]	valid_0's auc: 0.590089
[790]	valid_0's auc: 0.590195
[800]	valid_0's auc: 0.590301
[810]	valid_0's auc: 0.590239
[820]	valid_0's auc: 0.590361
[830]	valid_0's auc: 0.590357
[840]	valid_0's auc: 0.590389
[850]	valid_0's auc: 0.590558
[860]	valid_0's auc: 0.590653
[870]	valid_0's auc: 0.590618
[880]	valid_0's auc: 0.590708
[890]	valid_0's auc: 0.590815
[900]	valid_0's auc: 0.590802
[910]	valid_0's auc: 0.590944
[920]	valid_0's auc: 0.590929
[930]	valid_0's auc: 0.590934
[940]	valid_0's auc: 0.591099
[950]	valid_0's auc: 0.591137
[960]	valid_0's auc: 0.591148
[970]	valid_0's auc: 0.591219
[980]	valid_0's auc: 0.591419
[990]	valid_0's auc: 0.591455
[1000]	valid_0's auc: 0.591549
[1010]	valid_0's auc: 0.591565
[1020]	valid_0's auc: 0.591575
[1030]	valid_0's auc: 0.591556
[1040]	valid_0's auc: 0.591553
[1050]	valid_0's auc: 0.591639
[1060]	valid_0's auc: 0.591641
[1070]	valid_0's auc: 0.591761
[1080]	valid_0's auc: 0.592074
[1090]	valid_0's auc: 0.592131
[1100]	valid_0's auc: 0.592185
[1110]	valid_0's auc: 0.592231
[1120]	valid_0's auc: 0.592306
[1130]	valid_0's auc: 0.592225
[1140]	valid_0's auc: 0.592199
[1150]	valid_0's auc: 0.592246
[1160]	valid_0's auc: 0.592352
[1170]	valid_0's auc: 0.592357
[1180]	valid_0's auc: 0.592438
[1190]	valid_0's auc: 0.592467
[1200]	valid_0's auc: 0.592505
[1210]	valid_0's auc: 0.59244
[1220]	valid_0's auc: 0.592427
[1230]	valid_0's auc: 0.592407
[1240]	valid_0's auc: 0.592544
[1250]	valid_0's auc: 0.592593
[1260]	valid_0's auc: 0.592582
[1270]	valid_0's auc: 0.592583
[1280]	valid_0's auc: 0.592583
[1290]	valid_0's auc: 0.592604
[1300]	valid_0's auc: 0.592607
[1310]	valid_0's auc: 0.592509
[1320]	valid_0's auc: 0.592599
[1330]	valid_0's auc: 0.592677
[1340]	valid_0's auc: 0.592699
[1350]	valid_0's auc: 0.592704
[1360]	valid_0's auc: 0.592663
[1370]	valid_0's auc: 0.592687
[1380]	valid_0's auc: 0.592735
[1390]	valid_0's auc: 0.59278
[1400]	valid_0's auc: 0.592881
[1410]	valid_0's auc: 0.592902
[1420]	valid_0's auc: 0.592898
[1430]	valid_0's auc: 0.592897
[1440]	valid_0's auc: 0.592928
[1450]	valid_0's auc: 0.592892
[1460]	valid_0's auc: 0.592917
[1470]	valid_0's auc: 0.592922
[1480]	valid_0's auc: 0.592988
[1490]	valid_0's auc: 0.593025
[1500]	valid_0's auc: 0.593067
[1510]	valid_0's auc: 0.59307
[1520]	valid_0's auc: 0.593122
[1530]	valid_0's auc: 0.593131
[1540]	valid_0's auc: 0.593134
[1550]	valid_0's auc: 0.59316
[1560]	valid_0's auc: 0.593157
[1570]	valid_0's auc: 0.593174
[1580]	valid_0's auc: 0.593204
[1590]	valid_0's auc: 0.593188
[1600]	valid_0's auc: 0.593206
[1610]	valid_0's auc: 0.593232
[1620]	valid_0's auc: 0.593174
[1630]	valid_0's auc: 0.593244
[1640]	valid_0's auc: 0.593321
[1650]	valid_0's auc: 0.593291
[1660]	valid_0's auc: 0.593341
[1670]	valid_0's auc: 0.593372
[1680]	valid_0's auc: 0.593379
[1690]	valid_0's auc: 0.593375
[1700]	valid_0's auc: 0.593359
[1710]	valid_0's auc: 0.593343
[1720]	valid_0's auc: 0.593337
Early stopping, best iteration is:
[1675]	valid_0's auc: 0.593414
best score: 0.593413585181
best iteration: 1675
complete on: sex_guess3

--------------------
this is round: 11
sex_guess4 and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
--------------------


After selection:
target           uint8
sex_guess4    category
msno          category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512732
[20]	valid_0's auc: 0.519854
[30]	valid_0's auc: 0.525157
[40]	valid_0's auc: 0.530018
[50]	valid_0's auc: 0.533756
[60]	valid_0's auc: 0.536523
[70]	valid_0's auc: 0.540913
[80]	valid_0's auc: 0.544808
[90]	valid_0's auc: 0.547568
[100]	valid_0's auc: 0.549648
[110]	valid_0's auc: 0.551311
[120]	valid_0's auc: 0.5524
[130]	valid_0's auc: 0.553986
[140]	valid_0's auc: 0.555423
[150]	valid_0's auc: 0.556856
[160]	valid_0's auc: 0.558157
[170]	valid_0's auc: 0.558862
[180]	valid_0's auc: 0.559718
[190]	valid_0's auc: 0.561691
[200]	valid_0's auc: 0.564079
[210]	valid_0's auc: 0.566165
[220]	valid_0's auc: 0.567586
[230]	valid_0's auc: 0.569082
[240]	valid_0's auc: 0.57033
[250]	valid_0's auc: 0.571012
[260]	valid_0's auc: 0.57207
[270]	valid_0's auc: 0.573273
[280]	valid_0's auc: 0.573883
[290]	valid_0's auc: 0.575175
[300]	valid_0's auc: 0.575653
[310]	valid_0's auc: 0.575985
[320]	valid_0's auc: 0.576234
[330]	valid_0's auc: 0.577101
[340]	valid_0's auc: 0.577911
[350]	valid_0's auc: 0.57846
[360]	valid_0's auc: 0.57865
[370]	valid_0's auc: 0.57919
[380]	valid_0's auc: 0.579666
[390]	valid_0's auc: 0.579823
[400]	valid_0's auc: 0.580772
[410]	valid_0's auc: 0.581409
[420]	valid_0's auc: 0.581825
[430]	valid_0's auc: 0.582058
[440]	valid_0's auc: 0.582247
[450]	valid_0's auc: 0.582306
[460]	valid_0's auc: 0.582661
[470]	valid_0's auc: 0.583082
[480]	valid_0's auc: 0.583411
[490]	valid_0's auc: 0.583877
[500]	valid_0's auc: 0.584149
[510]	valid_0's auc: 0.584563
[520]	valid_0's auc: 0.584941
[530]	valid_0's auc: 0.585281
[540]	valid_0's auc: 0.585393
[550]	valid_0's auc: 0.585538
[560]	valid_0's auc: 0.585698
[570]	valid_0's auc: 0.585979
[580]	valid_0's auc: 0.586276
[590]	valid_0's auc: 0.586454
[600]	valid_0's auc: 0.586664
[610]	valid_0's auc: 0.586776
[620]	valid_0's auc: 0.586934
[630]	valid_0's auc: 0.586992
[640]	valid_0's auc: 0.587247
[650]	valid_0's auc: 0.587738
[660]	valid_0's auc: 0.588223
[670]	valid_0's auc: 0.588487
[680]	valid_0's auc: 0.588744
[690]	valid_0's auc: 0.588907
[700]	valid_0's auc: 0.589281
[710]	valid_0's auc: 0.5894
[720]	valid_0's auc: 0.589518
[730]	valid_0's auc: 0.589585
[740]	valid_0's auc: 0.58964
[750]	valid_0's auc: 0.589626
[760]	valid_0's auc: 0.589755
[770]	valid_0's auc: 0.589871
[780]	valid_0's auc: 0.589956
[790]	valid_0's auc: 0.590101
[800]	valid_0's auc: 0.590082
[810]	valid_0's auc: 0.590067
[820]	valid_0's auc: 0.59024
[830]	valid_0's auc: 0.590274
[840]	valid_0's auc: 0.590416
[850]	valid_0's auc: 0.590402
[860]	valid_0's auc: 0.590372
[870]	valid_0's auc: 0.590534
[880]	valid_0's auc: 0.590578
[890]	valid_0's auc: 0.59057
[900]	valid_0's auc: 0.590638
[910]	valid_0's auc: 0.590655
[920]	valid_0's auc: 0.590782
[930]	valid_0's auc: 0.590711
[940]	valid_0's auc: 0.590674
[950]	valid_0's auc: 0.590838
[960]	valid_0's auc: 0.591021
[970]	valid_0's auc: 0.591014
[980]	valid_0's auc: 0.591034
[990]	valid_0's auc: 0.590999
[1000]	valid_0's auc: 0.591026
[1010]	valid_0's auc: 0.591041
[1020]	valid_0's auc: 0.591086
[1030]	valid_0's auc: 0.591163
[1040]	valid_0's auc: 0.591493
[1050]	valid_0's auc: 0.591574
[1060]	valid_0's auc: 0.591591
[1070]	valid_0's auc: 0.591676
[1080]	valid_0's auc: 0.591788
[1090]	valid_0's auc: 0.591742
[1100]	valid_0's auc: 0.591696
[1110]	valid_0's auc: 0.59175
[1120]	valid_0's auc: 0.591833
[1130]	valid_0's auc: 0.591913
[1140]	valid_0's auc: 0.592011
[1150]	valid_0's auc: 0.592009
[1160]	valid_0's auc: 0.592048
[1170]	valid_0's auc: 0.591991
[1180]	valid_0's auc: 0.592003
[1190]	valid_0's auc: 0.592004
[1200]	valid_0's auc: 0.592104
[1210]	valid_0's auc: 0.592161
[1220]	valid_0's auc: 0.592172
[1230]	valid_0's auc: 0.592163
[1240]	valid_0's auc: 0.59218
[1250]	valid_0's auc: 0.592192
[1260]	valid_0's auc: 0.592102
[1270]	valid_0's auc: 0.592156
[1280]	valid_0's auc: 0.592218
[1290]	valid_0's auc: 0.592251
[1300]	valid_0's auc: 0.592241
[1310]	valid_0's auc: 0.592232
[1320]	valid_0's auc: 0.592211
[1330]	valid_0's auc: 0.592259
[1340]	valid_0's auc: 0.592317
[1350]	valid_0's auc: 0.592414
[1360]	valid_0's auc: 0.592445
[1370]	valid_0's auc: 0.59246
[1380]	valid_0's auc: 0.592476
[1390]	valid_0's auc: 0.592486
[1400]	valid_0's auc: 0.592477
[1410]	valid_0's auc: 0.59251
[1420]	valid_0's auc: 0.592519
[1430]	valid_0's auc: 0.592588
[1440]	valid_0's auc: 0.592604
[1450]	valid_0's auc: 0.592649
[1460]	valid_0's auc: 0.592679
[1470]	valid_0's auc: 0.592744
[1480]	valid_0's auc: 0.592753
[1490]	valid_0's auc: 0.592765
[1500]	valid_0's auc: 0.592743
[1510]	valid_0's auc: 0.592788
[1520]	valid_0's auc: 0.592804
[1530]	valid_0's auc: 0.592861
[1540]	valid_0's auc: 0.592923
[1550]	valid_0's auc: 0.592953
[1560]	valid_0's auc: 0.592978
[1570]	valid_0's auc: 0.593011
[1580]	valid_0's auc: 0.593067
[1590]	valid_0's auc: 0.593013
[1600]	valid_0's auc: 0.593102
[1610]	valid_0's auc: 0.593142
[1620]	valid_0's auc: 0.593128
[1630]	valid_0's auc: 0.593184
[1640]	valid_0's auc: 0.593246
[1650]	valid_0's auc: 0.593236
[1660]	valid_0's auc: 0.59317
[1670]	valid_0's auc: 0.593165
[1680]	valid_0's auc: 0.593168
[1690]	valid_0's auc: 0.593181
Early stopping, best iteration is:
[1640]	valid_0's auc: 0.593246
best score: 0.593246251579
best iteration: 1640
complete on: sex_guess4

--------------------
this is round: 12
sex_guess5 and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
--------------------


After selection:
target           uint8
sex_guess5    category
msno          category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513132
[20]	valid_0's auc: 0.521363
[30]	valid_0's auc: 0.525956
[40]	valid_0's auc: 0.531541
[50]	valid_0's auc: 0.535335
[60]	valid_0's auc: 0.538259
[70]	valid_0's auc: 0.540126
[80]	valid_0's auc: 0.5428
[90]	valid_0's auc: 0.546306
[100]	valid_0's auc: 0.549752
[110]	valid_0's auc: 0.552509
[120]	valid_0's auc: 0.554983
[130]	valid_0's auc: 0.557175
[140]	valid_0's auc: 0.558098
[150]	valid_0's auc: 0.559506
[160]	valid_0's auc: 0.560732
[170]	valid_0's auc: 0.562096
[180]	valid_0's auc: 0.563122
[190]	valid_0's auc: 0.564023
[200]	valid_0's auc: 0.564833
[210]	valid_0's auc: 0.565709
[220]	valid_0's auc: 0.566759
[230]	valid_0's auc: 0.567853
[240]	valid_0's auc: 0.568172
[250]	valid_0's auc: 0.568883
[260]	valid_0's auc: 0.569434
[270]	valid_0's auc: 0.571274
[280]	valid_0's auc: 0.572511
[290]	valid_0's auc: 0.57376
[300]	valid_0's auc: 0.574772
[310]	valid_0's auc: 0.575417
[320]	valid_0's auc: 0.576294
[330]	valid_0's auc: 0.577248
[340]	valid_0's auc: 0.577643
[350]	valid_0's auc: 0.578308
[360]	valid_0's auc: 0.579015
[370]	valid_0's auc: 0.579584
[380]	valid_0's auc: 0.57997
[390]	valid_0's auc: 0.580614
[400]	valid_0's auc: 0.58133
[410]	valid_0's auc: 0.582048
[420]	valid_0's auc: 0.582429
[430]	valid_0's auc: 0.582945
[440]	valid_0's auc: 0.583083
[450]	valid_0's auc: 0.583346
[460]	valid_0's auc: 0.583584
[470]	valid_0's auc: 0.583708
[480]	valid_0's auc: 0.583845
[490]	valid_0's auc: 0.584401
[500]	valid_0's auc: 0.58486
[510]	valid_0's auc: 0.585611
[520]	valid_0's auc: 0.585659
[530]	valid_0's auc: 0.585833
[540]	valid_0's auc: 0.585973
[550]	valid_0's auc: 0.586071
[560]	valid_0's auc: 0.586335
[570]	valid_0's auc: 0.586711
[580]	valid_0's auc: 0.586938
[590]	valid_0's auc: 0.587295
[600]	valid_0's auc: 0.587533
[610]	valid_0's auc: 0.587753
[620]	valid_0's auc: 0.588068
[630]	valid_0's auc: 0.588346
[640]	valid_0's auc: 0.588493
[650]	valid_0's auc: 0.588476
[660]	valid_0's auc: 0.588665
[670]	valid_0's auc: 0.588932
[680]	valid_0's auc: 0.589097
[690]	valid_0's auc: 0.589277
[700]	valid_0's auc: 0.589353
[710]	valid_0's auc: 0.589573
[720]	valid_0's auc: 0.589685
[730]	valid_0's auc: 0.589776
[740]	valid_0's auc: 0.589866
[750]	valid_0's auc: 0.589838
[760]	valid_0's auc: 0.590029
[770]	valid_0's auc: 0.590134
[780]	valid_0's auc: 0.590088
[790]	valid_0's auc: 0.590126
[800]	valid_0's auc: 0.590123
[810]	valid_0's auc: 0.590117
[820]	valid_0's auc: 0.590243
[830]	valid_0's auc: 0.590286
[840]	valid_0's auc: 0.590362
[850]	valid_0's auc: 0.590395
[860]	valid_0's auc: 0.590371
[870]	valid_0's auc: 0.590523
[880]	valid_0's auc: 0.590475
[890]	valid_0's auc: 0.590525
[900]	valid_0's auc: 0.590645
[910]	valid_0's auc: 0.59075
[920]	valid_0's auc: 0.590753
[930]	valid_0's auc: 0.590682
[940]	valid_0's auc: 0.590936
[950]	valid_0's auc: 0.590995
[960]	valid_0's auc: 0.591006
[970]	valid_0's auc: 0.591105
[980]	valid_0's auc: 0.591083
[990]	valid_0's auc: 0.591176
[1000]	valid_0's auc: 0.591266
[1010]	valid_0's auc: 0.591336
[1020]	valid_0's auc: 0.591329
[1030]	valid_0's auc: 0.591445
[1040]	valid_0's auc: 0.591598
[1050]	valid_0's auc: 0.591638
[1060]	valid_0's auc: 0.591642
[1070]	valid_0's auc: 0.591714
[1080]	valid_0's auc: 0.59174
[1090]	valid_0's auc: 0.591854
[1100]	valid_0's auc: 0.591921
[1110]	valid_0's auc: 0.591935
[1120]	valid_0's auc: 0.591959
[1130]	valid_0's auc: 0.592278
[1140]	valid_0's auc: 0.59231
[1150]	valid_0's auc: 0.592319
[1160]	valid_0's auc: 0.59238
[1170]	valid_0's auc: 0.592481
[1180]	valid_0's auc: 0.592438
[1190]	valid_0's auc: 0.592368
[1200]	valid_0's auc: 0.592429
[1210]	valid_0's auc: 0.592448
[1220]	valid_0's auc: 0.592549
[1230]	valid_0's auc: 0.592555
[1240]	valid_0's auc: 0.59262
[1250]	valid_0's auc: 0.592602
[1260]	valid_0's auc: 0.592611
[1270]	valid_0's auc: 0.592604
[1280]	valid_0's auc: 0.592576
[1290]	valid_0's auc: 0.592606
[1300]	valid_0's auc: 0.592677
[1310]	valid_0's auc: 0.59276
[1320]	valid_0's auc: 0.592733
[1330]	valid_0's auc: 0.59272
[1340]	valid_0's auc: 0.592709
[1350]	valid_0's auc: 0.592738
[1360]	valid_0's auc: 0.592757
Early stopping, best iteration is:
[1312]	valid_0's auc: 0.592765
best score: 0.592764916226
best iteration: 1312
complete on: sex_guess5

--------------------
this is round: 13
sex_freq_member and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
--------------------


After selection:
target                uint8
sex_freq_member    category
msno               category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513184
[20]	valid_0's auc: 0.520473
[30]	valid_0's auc: 0.525783
[40]	valid_0's auc: 0.530771
[50]	valid_0's auc: 0.534608
[60]	valid_0's auc: 0.537552
[70]	valid_0's auc: 0.542685
[80]	valid_0's auc: 0.546136
[90]	valid_0's auc: 0.548228
[100]	valid_0's auc: 0.550289
[110]	valid_0's auc: 0.55197
[120]	valid_0's auc: 0.55291
[130]	valid_0's auc: 0.554544
[140]	valid_0's auc: 0.555978
[150]	valid_0's auc: 0.557539
[160]	valid_0's auc: 0.558572
[170]	valid_0's auc: 0.55988
[180]	valid_0's auc: 0.56062
[190]	valid_0's auc: 0.561665
[200]	valid_0's auc: 0.563166
[210]	valid_0's auc: 0.564968
[220]	valid_0's auc: 0.567116
[230]	valid_0's auc: 0.568587
[240]	valid_0's auc: 0.569969
[250]	valid_0's auc: 0.571408
[260]	valid_0's auc: 0.572398
[270]	valid_0's auc: 0.572827
[280]	valid_0's auc: 0.574136
[290]	valid_0's auc: 0.5751
[300]	valid_0's auc: 0.575816
[310]	valid_0's auc: 0.576732
[320]	valid_0's auc: 0.577413
[330]	valid_0's auc: 0.577903
[340]	valid_0's auc: 0.578454
[350]	valid_0's auc: 0.578687
[360]	valid_0's auc: 0.579399
[370]	valid_0's auc: 0.579977
[380]	valid_0's auc: 0.580216
[390]	valid_0's auc: 0.580539
[400]	valid_0's auc: 0.58102
[410]	valid_0's auc: 0.581324
[420]	valid_0's auc: 0.582089
[430]	valid_0's auc: 0.582874
[440]	valid_0's auc: 0.583047
[450]	valid_0's auc: 0.583274
[460]	valid_0's auc: 0.583394
[470]	valid_0's auc: 0.583581
[480]	valid_0's auc: 0.583836
[490]	valid_0's auc: 0.584248
[500]	valid_0's auc: 0.584518
[510]	valid_0's auc: 0.584953
[520]	valid_0's auc: 0.585266
[530]	valid_0's auc: 0.585657
[540]	valid_0's auc: 0.585973
[550]	valid_0's auc: 0.586305
[560]	valid_0's auc: 0.586385
[570]	valid_0's auc: 0.586535
[580]	valid_0's auc: 0.586709
[590]	valid_0's auc: 0.586908
[600]	valid_0's auc: 0.58721
[610]	valid_0's auc: 0.587396
[620]	valid_0's auc: 0.587526
[630]	valid_0's auc: 0.587581
[640]	valid_0's auc: 0.587958
[650]	valid_0's auc: 0.588315
[660]	valid_0's auc: 0.588721
[670]	valid_0's auc: 0.588916
[680]	valid_0's auc: 0.589122
[690]	valid_0's auc: 0.589451
[700]	valid_0's auc: 0.589715
[710]	valid_0's auc: 0.589757
[720]	valid_0's auc: 0.589861
[730]	valid_0's auc: 0.589866
[740]	valid_0's auc: 0.589954
[750]	valid_0's auc: 0.590094
[760]	valid_0's auc: 0.590026
[770]	valid_0's auc: 0.590257
[780]	valid_0's auc: 0.590283
[790]	valid_0's auc: 0.590304
[800]	valid_0's auc: 0.590463
[810]	valid_0's auc: 0.590488
[820]	valid_0's auc: 0.590599
[830]	valid_0's auc: 0.590633
[840]	valid_0's auc: 0.590627
[850]	valid_0's auc: 0.590696
[860]	valid_0's auc: 0.590708
[870]	valid_0's auc: 0.590749
[880]	valid_0's auc: 0.590896
[890]	valid_0's auc: 0.591001
[900]	valid_0's auc: 0.591026
[910]	valid_0's auc: 0.591066
[920]	valid_0's auc: 0.59115
[930]	valid_0's auc: 0.591375
[940]	valid_0's auc: 0.591335
[950]	valid_0's auc: 0.591354
[960]	valid_0's auc: 0.591342
[970]	valid_0's auc: 0.591335
[980]	valid_0's auc: 0.592528
[990]	valid_0's auc: 0.592532
[1000]	valid_0's auc: 0.592588
[1010]	valid_0's auc: 0.592874
[1020]	valid_0's auc: 0.592959
[1030]	valid_0's auc: 0.592953
[1040]	valid_0's auc: 0.593027
[1050]	valid_0's auc: 0.59311
[1060]	valid_0's auc: 0.593078
[1070]	valid_0's auc: 0.593127
[1080]	valid_0's auc: 0.593155
[1090]	valid_0's auc: 0.593137
[1100]	valid_0's auc: 0.59314
[1110]	valid_0's auc: 0.59327
[1120]	valid_0's auc: 0.593373
[1130]	valid_0's auc: 0.593303
[1140]	valid_0's auc: 0.593308
[1150]	valid_0's auc: 0.593271
[1160]	valid_0's auc: 0.593393
[1170]	valid_0's auc: 0.593416
[1180]	valid_0's auc: 0.593412
[1190]	valid_0's auc: 0.59343
[1200]	valid_0's auc: 0.59345
[1210]	valid_0's auc: 0.593444
[1220]	valid_0's auc: 0.593457
[1230]	valid_0's auc: 0.593503
[1240]	valid_0's auc: 0.593507
[1250]	valid_0's auc: 0.593544
[1260]	valid_0's auc: 0.593535
[1270]	valid_0's auc: 0.593546
[1280]	valid_0's auc: 0.593531
[1290]	valid_0's auc: 0.593591
[1300]	valid_0's auc: 0.593686
[1310]	valid_0's auc: 0.593688
[1320]	valid_0's auc: 0.593712
[1330]	valid_0's auc: 0.593754
[1340]	valid_0's auc: 0.593719
[1350]	valid_0's auc: 0.593664
[1360]	valid_0's auc: 0.593694
[1370]	valid_0's auc: 0.593704
[1380]	valid_0's auc: 0.593781
[1390]	valid_0's auc: 0.593816
[1400]	valid_0's auc: 0.593843
[1410]	valid_0's auc: 0.593871
[1420]	valid_0's auc: 0.593858
[1430]	valid_0's auc: 0.593872
[1440]	valid_0's auc: 0.593896
[1450]	valid_0's auc: 0.593889
[1460]	valid_0's auc: 0.593856
[1470]	valid_0's auc: 0.593927
[1480]	valid_0's auc: 0.594055
[1490]	valid_0's auc: 0.593926
[1500]	valid_0's auc: 0.593984
[1510]	valid_0's auc: 0.594035
[1520]	valid_0's auc: 0.594104
[1530]	valid_0's auc: 0.594194
[1540]	valid_0's auc: 0.5942
[1550]	valid_0's auc: 0.594257
[1560]	valid_0's auc: 0.594286
[1570]	valid_0's auc: 0.594335
[1580]	valid_0's auc: 0.594345
[1590]	valid_0's auc: 0.594323
[1600]	valid_0's auc: 0.594386
[1610]	valid_0's auc: 0.594414
[1620]	valid_0's auc: 0.59447
[1630]	valid_0's auc: 0.594475
[1640]	valid_0's auc: 0.594506
[1650]	valid_0's auc: 0.594529
[1660]	valid_0's auc: 0.594552
[1670]	valid_0's auc: 0.59454
[1680]	valid_0's auc: 0.59457
[1690]	valid_0's auc: 0.594579
[1700]	valid_0's auc: 0.594538
[1710]	valid_0's auc: 0.594553
[1720]	valid_0's auc: 0.594575
[1730]	valid_0's auc: 0.594578
Early stopping, best iteration is:
[1687]	valid_0's auc: 0.594586
best score: 0.594585757499
best iteration: 1687
complete on: sex_freq_member

--------------------
this is round: 14
registration_year and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
--------------------


After selection:
target                  uint8
registration_year    category
msno                 category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512835
[20]	valid_0's auc: 0.52328
[30]	valid_0's auc: 0.5295
[40]	valid_0's auc: 0.53396
[50]	valid_0's auc: 0.537668
[60]	valid_0's auc: 0.540601
[70]	valid_0's auc: 0.545259
[80]	valid_0's auc: 0.548073
[90]	valid_0's auc: 0.550262
[100]	valid_0's auc: 0.551716
[110]	valid_0's auc: 0.553037
[120]	valid_0's auc: 0.554105
[130]	valid_0's auc: 0.5553
[140]	valid_0's auc: 0.556461
[150]	valid_0's auc: 0.557678
[160]	valid_0's auc: 0.558891
[170]	valid_0's auc: 0.561343
[180]	valid_0's auc: 0.564203
[190]	valid_0's auc: 0.565939
[200]	valid_0's auc: 0.567435
[210]	valid_0's auc: 0.569254
[220]	valid_0's auc: 0.570382
[230]	valid_0's auc: 0.571585
[240]	valid_0's auc: 0.572158
[250]	valid_0's auc: 0.572983
[260]	valid_0's auc: 0.573762
[270]	valid_0's auc: 0.574409
[280]	valid_0's auc: 0.575154
[290]	valid_0's auc: 0.57576
[300]	valid_0's auc: 0.576243
[310]	valid_0's auc: 0.577036
[320]	valid_0's auc: 0.577514
[330]	valid_0's auc: 0.57784
[340]	valid_0's auc: 0.578757
[350]	valid_0's auc: 0.579358
[360]	valid_0's auc: 0.579351
[370]	valid_0's auc: 0.57978
[380]	valid_0's auc: 0.580066
[390]	valid_0's auc: 0.58026
[400]	valid_0's auc: 0.58072
[410]	valid_0's auc: 0.580922
[420]	valid_0's auc: 0.581323
[430]	valid_0's auc: 0.58185
[440]	valid_0's auc: 0.582123
[450]	valid_0's auc: 0.582371
[460]	valid_0's auc: 0.582547
[470]	valid_0's auc: 0.582976
[480]	valid_0's auc: 0.583245
[490]	valid_0's auc: 0.583522
[500]	valid_0's auc: 0.583857
[510]	valid_0's auc: 0.584105
[520]	valid_0's auc: 0.58435
[530]	valid_0's auc: 0.58466
[540]	valid_0's auc: 0.584863
[550]	valid_0's auc: 0.584786
[560]	valid_0's auc: 0.584918
[570]	valid_0's auc: 0.585321
[580]	valid_0's auc: 0.585639
[590]	valid_0's auc: 0.586013
[600]	valid_0's auc: 0.586268
[610]	valid_0's auc: 0.586594
[620]	valid_0's auc: 0.586886
[630]	valid_0's auc: 0.587043
[640]	valid_0's auc: 0.587276
[650]	valid_0's auc: 0.587464
[660]	valid_0's auc: 0.587643
[670]	valid_0's auc: 0.587886
[680]	valid_0's auc: 0.587982
[690]	valid_0's auc: 0.588141
[700]	valid_0's auc: 0.58842
[710]	valid_0's auc: 0.588443
[720]	valid_0's auc: 0.588603
[730]	valid_0's auc: 0.58862
[740]	valid_0's auc: 0.588847
[750]	valid_0's auc: 0.58893
[760]	valid_0's auc: 0.58891
[770]	valid_0's auc: 0.58886
[780]	valid_0's auc: 0.588871
[790]	valid_0's auc: 0.588998
[800]	valid_0's auc: 0.589049
[810]	valid_0's auc: 0.589121
[820]	valid_0's auc: 0.58921
[830]	valid_0's auc: 0.589235
[840]	valid_0's auc: 0.589338
[850]	valid_0's auc: 0.589444
[860]	valid_0's auc: 0.589517
[870]	valid_0's auc: 0.589563
[880]	valid_0's auc: 0.589692
[890]	valid_0's auc: 0.589791
[900]	valid_0's auc: 0.589858
[910]	valid_0's auc: 0.589908
[920]	valid_0's auc: 0.590014
[930]	valid_0's auc: 0.590116
[940]	valid_0's auc: 0.590191
[950]	valid_0's auc: 0.590238
[960]	valid_0's auc: 0.590262
[970]	valid_0's auc: 0.590355
[980]	valid_0's auc: 0.5904
[990]	valid_0's auc: 0.590459
[1000]	valid_0's auc: 0.590531
[1010]	valid_0's auc: 0.590512
[1020]	valid_0's auc: 0.590593
[1030]	valid_0's auc: 0.590737
[1040]	valid_0's auc: 0.590623
[1050]	valid_0's auc: 0.590659
[1060]	valid_0's auc: 0.590705
[1070]	valid_0's auc: 0.590735
[1080]	valid_0's auc: 0.590814
[1090]	valid_0's auc: 0.590865
[1100]	valid_0's auc: 0.590872
[1110]	valid_0's auc: 0.591012
[1120]	valid_0's auc: 0.59114
[1130]	valid_0's auc: 0.591211
[1140]	valid_0's auc: 0.591297
[1150]	valid_0's auc: 0.591316
[1160]	valid_0's auc: 0.591326
[1170]	valid_0's auc: 0.591333
[1180]	valid_0's auc: 0.591391
[1190]	valid_0's auc: 0.591394
[1200]	valid_0's auc: 0.591352
[1210]	valid_0's auc: 0.591361
[1220]	valid_0's auc: 0.5914
[1230]	valid_0's auc: 0.591479
[1240]	valid_0's auc: 0.591554
[1250]	valid_0's auc: 0.591524
[1260]	valid_0's auc: 0.591555
[1270]	valid_0's auc: 0.591573
[1280]	valid_0's auc: 0.591599
[1290]	valid_0's auc: 0.591586
[1300]	valid_0's auc: 0.591654
[1310]	valid_0's auc: 0.591612
[1320]	valid_0's auc: 0.591626
[1330]	valid_0's auc: 0.591604
[1340]	valid_0's auc: 0.591668
[1350]	valid_0's auc: 0.591626
[1360]	valid_0's auc: 0.591659
[1370]	valid_0's auc: 0.591688
[1380]	valid_0's auc: 0.591655
[1390]	valid_0's auc: 0.591677
[1400]	valid_0's auc: 0.591655
[1410]	valid_0's auc: 0.591689
[1420]	valid_0's auc: 0.591735
[1430]	valid_0's auc: 0.59175
[1440]	valid_0's auc: 0.59171
[1450]	valid_0's auc: 0.591744
[1460]	valid_0's auc: 0.591795
[1470]	valid_0's auc: 0.59179
[1480]	valid_0's auc: 0.591793
[1490]	valid_0's auc: 0.591807
[1500]	valid_0's auc: 0.591803
[1510]	valid_0's auc: 0.591822
[1520]	valid_0's auc: 0.591832
[1530]	valid_0's auc: 0.591821
[1540]	valid_0's auc: 0.591856
[1550]	valid_0's auc: 0.591873
[1560]	valid_0's auc: 0.591862
[1570]	valid_0's auc: 0.591876
[1580]	valid_0's auc: 0.591869
[1590]	valid_0's auc: 0.591865
[1600]	valid_0's auc: 0.591876
[1610]	valid_0's auc: 0.591884
[1620]	valid_0's auc: 0.591901
[1630]	valid_0's auc: 0.59193
[1640]	valid_0's auc: 0.591927
[1650]	valid_0's auc: 0.591913
[1660]	valid_0's auc: 0.591936
[1670]	valid_0's auc: 0.591956
[1680]	valid_0's auc: 0.591979
[1690]	valid_0's auc: 0.591964
[1700]	valid_0's auc: 0.591934
[1710]	valid_0's auc: 0.591953
[1720]	valid_0's auc: 0.591956
[1730]	valid_0's auc: 0.591963
Early stopping, best iteration is:
[1682]	valid_0's auc: 0.591984
best score: 0.591984163499
best iteration: 1682
complete on: registration_year

--------------------
this is round: 15
registration_month and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
--------------------


After selection:
target                   uint8
registration_month    category
msno                  category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513589
[20]	valid_0's auc: 0.521279
[30]	valid_0's auc: 0.525318
[40]	valid_0's auc: 0.532301
[50]	valid_0's auc: 0.536077
[60]	valid_0's auc: 0.539148
[70]	valid_0's auc: 0.54202
[80]	valid_0's auc: 0.545796
[90]	valid_0's auc: 0.549027
[100]	valid_0's auc: 0.551939
[110]	valid_0's auc: 0.553523
[120]	valid_0's auc: 0.555124
[130]	valid_0's auc: 0.556195
[140]	valid_0's auc: 0.557598
[150]	valid_0's auc: 0.55886
[160]	valid_0's auc: 0.560346
[170]	valid_0's auc: 0.561199
[180]	valid_0's auc: 0.56221
[190]	valid_0's auc: 0.563407
[200]	valid_0's auc: 0.564156
[210]	valid_0's auc: 0.564923
[220]	valid_0's auc: 0.566953
[230]	valid_0's auc: 0.568703
[240]	valid_0's auc: 0.570255
[250]	valid_0's auc: 0.571565
[260]	valid_0's auc: 0.572907
[270]	valid_0's auc: 0.573432
[280]	valid_0's auc: 0.574565
[290]	valid_0's auc: 0.575188
[300]	valid_0's auc: 0.575802
[310]	valid_0's auc: 0.576784
[320]	valid_0's auc: 0.577468
[330]	valid_0's auc: 0.577912
[340]	valid_0's auc: 0.57855
[350]	valid_0's auc: 0.579501
[360]	valid_0's auc: 0.580103
[370]	valid_0's auc: 0.580344
[380]	valid_0's auc: 0.580686
[390]	valid_0's auc: 0.581347
[400]	valid_0's auc: 0.581504
[410]	valid_0's auc: 0.582001
[420]	valid_0's auc: 0.58212
[430]	valid_0's auc: 0.582477
[440]	valid_0's auc: 0.582979
[450]	valid_0's auc: 0.583683
[460]	valid_0's auc: 0.583917
[470]	valid_0's auc: 0.584057
[480]	valid_0's auc: 0.584229
[490]	valid_0's auc: 0.584388
[500]	valid_0's auc: 0.584643
[510]	valid_0's auc: 0.584953
[520]	valid_0's auc: 0.585279
[530]	valid_0's auc: 0.585684
[540]	valid_0's auc: 0.585911
[550]	valid_0's auc: 0.58612
[560]	valid_0's auc: 0.586426
[570]	valid_0's auc: 0.586794
[580]	valid_0's auc: 0.586977
[590]	valid_0's auc: 0.587059
[600]	valid_0's auc: 0.587201
[610]	valid_0's auc: 0.58739
[620]	valid_0's auc: 0.587756
[630]	valid_0's auc: 0.587825
[640]	valid_0's auc: 0.58794
[650]	valid_0's auc: 0.588053
[660]	valid_0's auc: 0.588245
[670]	valid_0's auc: 0.588412
[680]	valid_0's auc: 0.588478
[690]	valid_0's auc: 0.588389
[700]	valid_0's auc: 0.588469
[710]	valid_0's auc: 0.58856
[720]	valid_0's auc: 0.588737
[730]	valid_0's auc: 0.588777
[740]	valid_0's auc: 0.588791
[750]	valid_0's auc: 0.588815
[760]	valid_0's auc: 0.588845
[770]	valid_0's auc: 0.588937
[780]	valid_0's auc: 0.589088
[790]	valid_0's auc: 0.589461
[800]	valid_0's auc: 0.589557
[810]	valid_0's auc: 0.589819
[820]	valid_0's auc: 0.589951
[830]	valid_0's auc: 0.590007
[840]	valid_0's auc: 0.590011
[850]	valid_0's auc: 0.590134
[860]	valid_0's auc: 0.590113
[870]	valid_0's auc: 0.590295
[880]	valid_0's auc: 0.590678
[890]	valid_0's auc: 0.590771
[900]	valid_0's auc: 0.590736
[910]	valid_0's auc: 0.590851
[920]	valid_0's auc: 0.590875
[930]	valid_0's auc: 0.590895
[940]	valid_0's auc: 0.590977
[950]	valid_0's auc: 0.590913
[960]	valid_0's auc: 0.590947
[970]	valid_0's auc: 0.591011
[980]	valid_0's auc: 0.591352
[990]	valid_0's auc: 0.591383
[1000]	valid_0's auc: 0.591554
[1010]	valid_0's auc: 0.59168
[1020]	valid_0's auc: 0.591739
[1030]	valid_0's auc: 0.591824
[1040]	valid_0's auc: 0.591767
[1050]	valid_0's auc: 0.591744
[1060]	valid_0's auc: 0.591871
[1070]	valid_0's auc: 0.591866
[1080]	valid_0's auc: 0.591847
[1090]	valid_0's auc: 0.591882
[1100]	valid_0's auc: 0.591889
[1110]	valid_0's auc: 0.59216
[1120]	valid_0's auc: 0.592288
[1130]	valid_0's auc: 0.592298
[1140]	valid_0's auc: 0.592385
[1150]	valid_0's auc: 0.592477
[1160]	valid_0's auc: 0.592563
[1170]	valid_0's auc: 0.592619
[1180]	valid_0's auc: 0.592635
[1190]	valid_0's auc: 0.592621
[1200]	valid_0's auc: 0.59267
[1210]	valid_0's auc: 0.592708
[1220]	valid_0's auc: 0.59272
[1230]	valid_0's auc: 0.592754
[1240]	valid_0's auc: 0.59276
[1250]	valid_0's auc: 0.592853
[1260]	valid_0's auc: 0.592845
[1270]	valid_0's auc: 0.592834
[1280]	valid_0's auc: 0.592879
[1290]	valid_0's auc: 0.592859
[1300]	valid_0's auc: 0.592925
[1310]	valid_0's auc: 0.593023
[1320]	valid_0's auc: 0.593048
[1330]	valid_0's auc: 0.59313
[1340]	valid_0's auc: 0.593111
[1350]	valid_0's auc: 0.59306
[1360]	valid_0's auc: 0.593098
[1370]	valid_0's auc: 0.593106
Early stopping, best iteration is:
[1329]	valid_0's auc: 0.593132
best score: 0.593132052878
best iteration: 1329
complete on: registration_month

--------------------
this is round: 16
registration_date and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
--------------------


After selection:
target                  uint8
registration_date    category
msno                 category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512862
[20]	valid_0's auc: 0.521255
[30]	valid_0's auc: 0.525254
[40]	valid_0's auc: 0.531122
[50]	valid_0's auc: 0.534774
[60]	valid_0's auc: 0.537753
[70]	valid_0's auc: 0.539552
[80]	valid_0's auc: 0.542193
[90]	valid_0's auc: 0.546663
[100]	valid_0's auc: 0.549466
[110]	valid_0's auc: 0.551965
[120]	valid_0's auc: 0.554821
[130]	valid_0's auc: 0.556106
[140]	valid_0's auc: 0.557359
[150]	valid_0's auc: 0.558749
[160]	valid_0's auc: 0.559962
[170]	valid_0's auc: 0.561187
[180]	valid_0's auc: 0.562314
[190]	valid_0's auc: 0.563622
[200]	valid_0's auc: 0.564244
[210]	valid_0's auc: 0.565522
[220]	valid_0's auc: 0.567668
[230]	valid_0's auc: 0.568401
[240]	valid_0's auc: 0.569095
[250]	valid_0's auc: 0.569839
[260]	valid_0's auc: 0.570579
[270]	valid_0's auc: 0.572219
[280]	valid_0's auc: 0.573393
[290]	valid_0's auc: 0.574977
[300]	valid_0's auc: 0.576029
[310]	valid_0's auc: 0.576787
[320]	valid_0's auc: 0.577355
[330]	valid_0's auc: 0.577891
[340]	valid_0's auc: 0.578524
[350]	valid_0's auc: 0.579331
[360]	valid_0's auc: 0.579943
[370]	valid_0's auc: 0.580385
[380]	valid_0's auc: 0.580598
[390]	valid_0's auc: 0.581433
[400]	valid_0's auc: 0.582336
[410]	valid_0's auc: 0.582725
[420]	valid_0's auc: 0.583144
[430]	valid_0's auc: 0.583517
[440]	valid_0's auc: 0.583761
[450]	valid_0's auc: 0.583961
[460]	valid_0's auc: 0.584326
[470]	valid_0's auc: 0.584762
[480]	valid_0's auc: 0.585201
[490]	valid_0's auc: 0.585423
[500]	valid_0's auc: 0.585595
[510]	valid_0's auc: 0.58567
[520]	valid_0's auc: 0.585826
[530]	valid_0's auc: 0.586188
[540]	valid_0's auc: 0.586642
[550]	valid_0's auc: 0.586893
[560]	valid_0's auc: 0.587317
[570]	valid_0's auc: 0.587538
[580]	valid_0's auc: 0.587722
[590]	valid_0's auc: 0.587931
[600]	valid_0's auc: 0.588248
[610]	valid_0's auc: 0.588356
[620]	valid_0's auc: 0.588423
[630]	valid_0's auc: 0.58861
[640]	valid_0's auc: 0.588749
[650]	valid_0's auc: 0.58892
[660]	valid_0's auc: 0.58909
[670]	valid_0's auc: 0.589443
[680]	valid_0's auc: 0.589598
[690]	valid_0's auc: 0.589664
[700]	valid_0's auc: 0.589628
[710]	valid_0's auc: 0.58978
[720]	valid_0's auc: 0.590108
[730]	valid_0's auc: 0.590063
[740]	valid_0's auc: 0.590196
[750]	valid_0's auc: 0.590198
[760]	valid_0's auc: 0.59019
[770]	valid_0's auc: 0.590244
[780]	valid_0's auc: 0.590201
[790]	valid_0's auc: 0.590309
[800]	valid_0's auc: 0.590445
[810]	valid_0's auc: 0.590545
[820]	valid_0's auc: 0.590523
[830]	valid_0's auc: 0.590443
[840]	valid_0's auc: 0.590693
[850]	valid_0's auc: 0.590689
[860]	valid_0's auc: 0.590611
[870]	valid_0's auc: 0.590702
[880]	valid_0's auc: 0.59085
[890]	valid_0's auc: 0.590986
[900]	valid_0's auc: 0.591001
[910]	valid_0's auc: 0.590948
[920]	valid_0's auc: 0.590928
[930]	valid_0's auc: 0.591041
[940]	valid_0's auc: 0.591019
[950]	valid_0's auc: 0.591033
[960]	valid_0's auc: 0.591126
[970]	valid_0's auc: 0.591396
[980]	valid_0's auc: 0.591309
[990]	valid_0's auc: 0.591407
[1000]	valid_0's auc: 0.59153
[1010]	valid_0's auc: 0.591628
[1020]	valid_0's auc: 0.591692
[1030]	valid_0's auc: 0.591663
[1040]	valid_0's auc: 0.591735
[1050]	valid_0's auc: 0.591761
[1060]	valid_0's auc: 0.591862
[1070]	valid_0's auc: 0.591919
[1080]	valid_0's auc: 0.592224
[1090]	valid_0's auc: 0.592387
[1100]	valid_0's auc: 0.592352
[1110]	valid_0's auc: 0.592205
[1120]	valid_0's auc: 0.592186
[1130]	valid_0's auc: 0.592259
[1140]	valid_0's auc: 0.592296
Early stopping, best iteration is:
[1093]	valid_0's auc: 0.592401
best score: 0.592401161697
best iteration: 1093
complete on: registration_date

--------------------
this is round: 17
expiration_year and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
--------------------


After selection:
target                uint8
expiration_year    category
msno               category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.517482
[20]	valid_0's auc: 0.525296
[30]	valid_0's auc: 0.530368
[40]	valid_0's auc: 0.53585
[50]	valid_0's auc: 0.540141
[60]	valid_0's auc: 0.543261
[70]	valid_0's auc: 0.545556
[80]	valid_0's auc: 0.54986
[90]	valid_0's auc: 0.553414
[100]	valid_0's auc: 0.556495
[110]	valid_0's auc: 0.558055
[120]	valid_0's auc: 0.560059
[130]	valid_0's auc: 0.561317
[140]	valid_0's auc: 0.562342
[150]	valid_0's auc: 0.56355
[160]	valid_0's auc: 0.564814
[170]	valid_0's auc: 0.56563
[180]	valid_0's auc: 0.566722
[190]	valid_0's auc: 0.567819
[200]	valid_0's auc: 0.569942
[210]	valid_0's auc: 0.571651
[220]	valid_0's auc: 0.573119
[230]	valid_0's auc: 0.574169
[240]	valid_0's auc: 0.57517
[250]	valid_0's auc: 0.576597
[260]	valid_0's auc: 0.577158
[270]	valid_0's auc: 0.578366
[280]	valid_0's auc: 0.579056
[290]	valid_0's auc: 0.579697
[300]	valid_0's auc: 0.580578
[310]	valid_0's auc: 0.58169
[320]	valid_0's auc: 0.582103
[330]	valid_0's auc: 0.582624
[340]	valid_0's auc: 0.582917
[350]	valid_0's auc: 0.583546
[360]	valid_0's auc: 0.5842
[370]	valid_0's auc: 0.58472
[380]	valid_0's auc: 0.584823
[390]	valid_0's auc: 0.585311
[400]	valid_0's auc: 0.585579
[410]	valid_0's auc: 0.585762
[420]	valid_0's auc: 0.586231
[430]	valid_0's auc: 0.586736
[440]	valid_0's auc: 0.587365
[450]	valid_0's auc: 0.587519
[460]	valid_0's auc: 0.587806
[470]	valid_0's auc: 0.587875
[480]	valid_0's auc: 0.587977
[490]	valid_0's auc: 0.588187
[500]	valid_0's auc: 0.588465
[510]	valid_0's auc: 0.588814
[520]	valid_0's auc: 0.589749
[530]	valid_0's auc: 0.590023
[540]	valid_0's auc: 0.590238
[550]	valid_0's auc: 0.590374
[560]	valid_0's auc: 0.590605
[570]	valid_0's auc: 0.59083
[580]	valid_0's auc: 0.590904
[590]	valid_0's auc: 0.590939
[600]	valid_0's auc: 0.591025
[610]	valid_0's auc: 0.591175
[620]	valid_0's auc: 0.59129
[630]	valid_0's auc: 0.591534
[640]	valid_0's auc: 0.591957
[650]	valid_0's auc: 0.592225
[660]	valid_0's auc: 0.592382
[670]	valid_0's auc: 0.592464
[680]	valid_0's auc: 0.592509
[690]	valid_0's auc: 0.59265
[700]	valid_0's auc: 0.592824
[710]	valid_0's auc: 0.592945
[720]	valid_0's auc: 0.593025
[730]	valid_0's auc: 0.593007
[740]	valid_0's auc: 0.593146
[750]	valid_0's auc: 0.593303
[760]	valid_0's auc: 0.59346
[770]	valid_0's auc: 0.593531
[780]	valid_0's auc: 0.59355
[790]	valid_0's auc: 0.593578
[800]	valid_0's auc: 0.593662
[810]	valid_0's auc: 0.593882
[820]	valid_0's auc: 0.593977
[830]	valid_0's auc: 0.594076
[840]	valid_0's auc: 0.594067
[850]	valid_0's auc: 0.594254
[860]	valid_0's auc: 0.594325
[870]	valid_0's auc: 0.594313
[880]	valid_0's auc: 0.594324
[890]	valid_0's auc: 0.594457
[900]	valid_0's auc: 0.594599
[910]	valid_0's auc: 0.594706
[920]	valid_0's auc: 0.594715
[930]	valid_0's auc: 0.594708
[940]	valid_0's auc: 0.594649
[950]	valid_0's auc: 0.594874
[960]	valid_0's auc: 0.594945
[970]	valid_0's auc: 0.594938
[980]	valid_0's auc: 0.594881
[990]	valid_0's auc: 0.594865
[1000]	valid_0's auc: 0.594885
[1010]	valid_0's auc: 0.594848
Early stopping, best iteration is:
[964]	valid_0's auc: 0.594955
best score: 0.594955370499
best iteration: 964
complete on: expiration_year

--------------------
this is round: 18
expiration_month and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
--------------------


After selection:
target                 uint8
expiration_month    category
msno                category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.528033
[20]	valid_0's auc: 0.533355
[30]	valid_0's auc: 0.538399
[40]	valid_0's auc: 0.542513
[50]	valid_0's auc: 0.547469
[60]	valid_0's auc: 0.550169
[70]	valid_0's auc: 0.551799
[80]	valid_0's auc: 0.555475
[90]	valid_0's auc: 0.558437
[100]	valid_0's auc: 0.561235
[110]	valid_0's auc: 0.562986
[120]	valid_0's auc: 0.564386
[130]	valid_0's auc: 0.565462
[140]	valid_0's auc: 0.566394
[150]	valid_0's auc: 0.567441
[160]	valid_0's auc: 0.568408
[170]	valid_0's auc: 0.569305
[180]	valid_0's auc: 0.570381
[190]	valid_0's auc: 0.571129
[200]	valid_0's auc: 0.572011
[210]	valid_0's auc: 0.572625
[220]	valid_0's auc: 0.573921
[230]	valid_0's auc: 0.575633
[240]	valid_0's auc: 0.577022
[250]	valid_0's auc: 0.577997
[260]	valid_0's auc: 0.578665
[270]	valid_0's auc: 0.579324
[280]	valid_0's auc: 0.579994
[290]	valid_0's auc: 0.580915
[300]	valid_0's auc: 0.581459
[310]	valid_0's auc: 0.581926
[320]	valid_0's auc: 0.58258
[330]	valid_0's auc: 0.583351
[340]	valid_0's auc: 0.584017
[350]	valid_0's auc: 0.584587
[360]	valid_0's auc: 0.585085
[370]	valid_0's auc: 0.585677
[380]	valid_0's auc: 0.586098
[390]	valid_0's auc: 0.586512
[400]	valid_0's auc: 0.586763
[410]	valid_0's auc: 0.587056
[420]	valid_0's auc: 0.587407
[430]	valid_0's auc: 0.587734
[440]	valid_0's auc: 0.588048
[450]	valid_0's auc: 0.588562
[460]	valid_0's auc: 0.588818
[470]	valid_0's auc: 0.589163
[480]	valid_0's auc: 0.589387
[490]	valid_0's auc: 0.589339
[500]	valid_0's auc: 0.589404
[510]	valid_0's auc: 0.589562
[520]	valid_0's auc: 0.589865
[530]	valid_0's auc: 0.590036
[540]	valid_0's auc: 0.590069
[550]	valid_0's auc: 0.590411
[560]	valid_0's auc: 0.590732
[570]	valid_0's auc: 0.590808
[580]	valid_0's auc: 0.590961
[590]	valid_0's auc: 0.591287
[600]	valid_0's auc: 0.591441
[610]	valid_0's auc: 0.591631
[620]	valid_0's auc: 0.591711
[630]	valid_0's auc: 0.591707
[640]	valid_0's auc: 0.591849
[650]	valid_0's auc: 0.592011
[660]	valid_0's auc: 0.592073
[670]	valid_0's auc: 0.59231
[680]	valid_0's auc: 0.592399
[690]	valid_0's auc: 0.592502
[700]	valid_0's auc: 0.592625
[710]	valid_0's auc: 0.592706
[720]	valid_0's auc: 0.592745
[730]	valid_0's auc: 0.592783
[740]	valid_0's auc: 0.592842
[750]	valid_0's auc: 0.593011
[760]	valid_0's auc: 0.593198
[770]	valid_0's auc: 0.593245
[780]	valid_0's auc: 0.59332
[790]	valid_0's auc: 0.593491
[800]	valid_0's auc: 0.593591
[810]	valid_0's auc: 0.59378
[820]	valid_0's auc: 0.593972
[830]	valid_0's auc: 0.594026
[840]	valid_0's auc: 0.594212
[850]	valid_0's auc: 0.594286
[860]	valid_0's auc: 0.594405
[870]	valid_0's auc: 0.594388
[880]	valid_0's auc: 0.594461
[890]	valid_0's auc: 0.594556
[900]	valid_0's auc: 0.594613
[910]	valid_0's auc: 0.594752
[920]	valid_0's auc: 0.59482
[930]	valid_0's auc: 0.594833
[940]	valid_0's auc: 0.594894
[950]	valid_0's auc: 0.594896
[960]	valid_0's auc: 0.595005
[970]	valid_0's auc: 0.595033
[980]	valid_0's auc: 0.595118
[990]	valid_0's auc: 0.595103
[1000]	valid_0's auc: 0.595124
[1010]	valid_0's auc: 0.595052
[1020]	valid_0's auc: 0.595094
[1030]	valid_0's auc: 0.595156
[1040]	valid_0's auc: 0.595226
[1050]	valid_0's auc: 0.595215
[1060]	valid_0's auc: 0.595188
[1070]	valid_0's auc: 0.595216
[1080]	valid_0's auc: 0.595276
[1090]	valid_0's auc: 0.595252
[1100]	valid_0's auc: 0.595224
[1110]	valid_0's auc: 0.59529
[1120]	valid_0's auc: 0.595341
[1130]	valid_0's auc: 0.595329
[1140]	valid_0's auc: 0.595268
[1150]	valid_0's auc: 0.59531
[1160]	valid_0's auc: 0.595267
Early stopping, best iteration is:
[1117]	valid_0's auc: 0.595382
best score: 0.595381985592
best iteration: 1117
complete on: expiration_month

--------------------
this is round: 19
expiration_date and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
--------------------


After selection:
target                uint8
expiration_date    category
msno               category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.512937
[20]	valid_0's auc: 0.520339
[30]	valid_0's auc: 0.525488
[40]	valid_0's auc: 0.529855
[50]	valid_0's auc: 0.533218
[60]	valid_0's auc: 0.538763
[70]	valid_0's auc: 0.543519
[80]	valid_0's auc: 0.546692
[90]	valid_0's auc: 0.548531
[100]	valid_0's auc: 0.550356
[110]	valid_0's auc: 0.552225
[120]	valid_0's auc: 0.553328
[130]	valid_0's auc: 0.554967
[140]	valid_0's auc: 0.556397
[150]	valid_0's auc: 0.557864
[160]	valid_0's auc: 0.559016
[170]	valid_0's auc: 0.559892
[180]	valid_0's auc: 0.560734
[190]	valid_0's auc: 0.56278
[200]	valid_0's auc: 0.565197
[210]	valid_0's auc: 0.567116
[220]	valid_0's auc: 0.568583
[230]	valid_0's auc: 0.56996
[240]	valid_0's auc: 0.571349
[250]	valid_0's auc: 0.572107
[260]	valid_0's auc: 0.573957
[270]	valid_0's auc: 0.575015
[280]	valid_0's auc: 0.575423
[290]	valid_0's auc: 0.576476
[300]	valid_0's auc: 0.577211
[310]	valid_0's auc: 0.577634
[320]	valid_0's auc: 0.578104
[330]	valid_0's auc: 0.578383
[340]	valid_0's auc: 0.579263
[350]	valid_0's auc: 0.580166
[360]	valid_0's auc: 0.580569
[370]	valid_0's auc: 0.580721
[380]	valid_0's auc: 0.581425
[390]	valid_0's auc: 0.581908
[400]	valid_0's auc: 0.582019
[410]	valid_0's auc: 0.582706
[420]	valid_0's auc: 0.583175
[430]	valid_0's auc: 0.583556
[440]	valid_0's auc: 0.583667
[450]	valid_0's auc: 0.584056
[460]	valid_0's auc: 0.584247
[470]	valid_0's auc: 0.584505
[480]	valid_0's auc: 0.584834
[490]	valid_0's auc: 0.585119
[500]	valid_0's auc: 0.585543
[510]	valid_0's auc: 0.586221
[520]	valid_0's auc: 0.586382
[530]	valid_0's auc: 0.586687
[540]	valid_0's auc: 0.58697
[550]	valid_0's auc: 0.587086
[560]	valid_0's auc: 0.587193
[570]	valid_0's auc: 0.587375
[580]	valid_0's auc: 0.587763
[590]	valid_0's auc: 0.587895
[600]	valid_0's auc: 0.588175
[610]	valid_0's auc: 0.588214
[620]	valid_0's auc: 0.588443
[630]	valid_0's auc: 0.588587
[640]	valid_0's auc: 0.588726
[650]	valid_0's auc: 0.588759
[660]	valid_0's auc: 0.588906
[670]	valid_0's auc: 0.589262
[680]	valid_0's auc: 0.589745
[690]	valid_0's auc: 0.5902
[700]	valid_0's auc: 0.590227
[710]	valid_0's auc: 0.590592
[720]	valid_0's auc: 0.590979
[730]	valid_0's auc: 0.591035
[740]	valid_0's auc: 0.591131
[750]	valid_0's auc: 0.591259
[760]	valid_0's auc: 0.591337
[770]	valid_0's auc: 0.591346
[780]	valid_0's auc: 0.591522
[790]	valid_0's auc: 0.591476
[800]	valid_0's auc: 0.591555
[810]	valid_0's auc: 0.591669
[820]	valid_0's auc: 0.591663
[830]	valid_0's auc: 0.591683
[840]	valid_0's auc: 0.591856
[850]	valid_0's auc: 0.592024
[860]	valid_0's auc: 0.592107
[870]	valid_0's auc: 0.592073
[880]	valid_0's auc: 0.59216
[890]	valid_0's auc: 0.592173
[900]	valid_0's auc: 0.592099
[910]	valid_0's auc: 0.592083
[920]	valid_0's auc: 0.592168
[930]	valid_0's auc: 0.592292
[940]	valid_0's auc: 0.592356
[950]	valid_0's auc: 0.592372
[960]	valid_0's auc: 0.592457
[970]	valid_0's auc: 0.592478
[980]	valid_0's auc: 0.592498
[990]	valid_0's auc: 0.592777
[1000]	valid_0's auc: 0.592794
[1010]	valid_0's auc: 0.592629
[1020]	valid_0's auc: 0.592727
[1030]	valid_0's auc: 0.592745
[1040]	valid_0's auc: 0.592873
[1050]	valid_0's auc: 0.592928
[1060]	valid_0's auc: 0.592982
[1070]	valid_0's auc: 0.592991
[1080]	valid_0's auc: 0.593114
[1090]	valid_0's auc: 0.59312
[1100]	valid_0's auc: 0.593056
[1110]	valid_0's auc: 0.593027
[1120]	valid_0's auc: 0.593095
[1130]	valid_0's auc: 0.593144
[1140]	valid_0's auc: 0.593196
[1150]	valid_0's auc: 0.593255
[1160]	valid_0's auc: 0.593257
[1170]	valid_0's auc: 0.593299
[1180]	valid_0's auc: 0.593295
[1190]	valid_0's auc: 0.593375
[1200]	valid_0's auc: 0.593399
[1210]	valid_0's auc: 0.593607
[1220]	valid_0's auc: 0.593655
[1230]	valid_0's auc: 0.593697
[1240]	valid_0's auc: 0.593722
[1250]	valid_0's auc: 0.593656
[1260]	valid_0's auc: 0.593722
[1270]	valid_0's auc: 0.593713
[1280]	valid_0's auc: 0.593765
[1290]	valid_0's auc: 0.593749
[1300]	valid_0's auc: 0.593768
[1310]	valid_0's auc: 0.593816
[1320]	valid_0's auc: 0.593872
[1330]	valid_0's auc: 0.593949
[1340]	valid_0's auc: 0.594013
[1350]	valid_0's auc: 0.594008
[1360]	valid_0's auc: 0.594109
[1370]	valid_0's auc: 0.594185
[1380]	valid_0's auc: 0.594164
[1390]	valid_0's auc: 0.594212
[1400]	valid_0's auc: 0.594224
[1410]	valid_0's auc: 0.594235
[1420]	valid_0's auc: 0.594242
[1430]	valid_0's auc: 0.594268
[1440]	valid_0's auc: 0.594264
[1450]	valid_0's auc: 0.594289
[1460]	valid_0's auc: 0.594371
[1470]	valid_0's auc: 0.5944
[1480]	valid_0's auc: 0.594468
[1490]	valid_0's auc: 0.594476
[1500]	valid_0's auc: 0.594541
[1510]	valid_0's auc: 0.594523
[1520]	valid_0's auc: 0.594558
[1530]	valid_0's auc: 0.594722
[1540]	valid_0's auc: 0.594679
[1550]	valid_0's auc: 0.594685
[1560]	valid_0's auc: 0.594736
[1570]	valid_0's auc: 0.59479
[1580]	valid_0's auc: 0.59477
[1590]	valid_0's auc: 0.594771
[1600]	valid_0's auc: 0.594758
[1610]	valid_0's auc: 0.594794
[1620]	valid_0's auc: 0.594821
[1630]	valid_0's auc: 0.594794
[1640]	valid_0's auc: 0.594785
[1650]	valid_0's auc: 0.594811
[1660]	valid_0's auc: 0.594823
Early stopping, best iteration is:
[1617]	valid_0's auc: 0.594835
best score: 0.594835238539
best iteration: 1617
complete on: expiration_date

--------------------
this is round: 20
genre_ids and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
--------------------


After selection:
target          uint8
genre_ids    category
msno         category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537861
[20]	valid_0's auc: 0.544394
[30]	valid_0's auc: 0.548789
[40]	valid_0's auc: 0.552014
[50]	valid_0's auc: 0.554939
[60]	valid_0's auc: 0.558373
[70]	valid_0's auc: 0.560435
[80]	valid_0's auc: 0.562595
[90]	valid_0's auc: 0.563891
[100]	valid_0's auc: 0.565183
[110]	valid_0's auc: 0.566305
[120]	valid_0's auc: 0.567836
[130]	valid_0's auc: 0.569667
[140]	valid_0's auc: 0.571457
[150]	valid_0's auc: 0.573187
[160]	valid_0's auc: 0.574782
[170]	valid_0's auc: 0.576407
[180]	valid_0's auc: 0.577489
[190]	valid_0's auc: 0.578691
[200]	valid_0's auc: 0.579803
[210]	valid_0's auc: 0.580873
[220]	valid_0's auc: 0.581339
[230]	valid_0's auc: 0.58221
[240]	valid_0's auc: 0.582978
[250]	valid_0's auc: 0.583493
[260]	valid_0's auc: 0.584119
[270]	valid_0's auc: 0.584555
[280]	valid_0's auc: 0.585139
[290]	valid_0's auc: 0.585658
[300]	valid_0's auc: 0.586143
[310]	valid_0's auc: 0.586623
[320]	valid_0's auc: 0.587113
[330]	valid_0's auc: 0.587429
[340]	valid_0's auc: 0.588042
[350]	valid_0's auc: 0.588436
[360]	valid_0's auc: 0.588824
[370]	valid_0's auc: 0.589245
[380]	valid_0's auc: 0.589495
[390]	valid_0's auc: 0.589702
[400]	valid_0's auc: 0.589873
[410]	valid_0's auc: 0.590072
[420]	valid_0's auc: 0.590416
[430]	valid_0's auc: 0.590781
[440]	valid_0's auc: 0.591205
[450]	valid_0's auc: 0.591294
[460]	valid_0's auc: 0.591763
[470]	valid_0's auc: 0.592128
[480]	valid_0's auc: 0.592626
[490]	valid_0's auc: 0.593025
[500]	valid_0's auc: 0.593317
[510]	valid_0's auc: 0.593555
[520]	valid_0's auc: 0.593977
[530]	valid_0's auc: 0.59429
[540]	valid_0's auc: 0.594583
[550]	valid_0's auc: 0.594943
[560]	valid_0's auc: 0.595077
[570]	valid_0's auc: 0.595311
[580]	valid_0's auc: 0.595557
[590]	valid_0's auc: 0.595802
[600]	valid_0's auc: 0.595994
[610]	valid_0's auc: 0.59623
[620]	valid_0's auc: 0.596363
[630]	valid_0's auc: 0.596695
[640]	valid_0's auc: 0.596838
[650]	valid_0's auc: 0.596945
[660]	valid_0's auc: 0.597129
[670]	valid_0's auc: 0.597226
[680]	valid_0's auc: 0.597277
[690]	valid_0's auc: 0.597331
[700]	valid_0's auc: 0.597537
[710]	valid_0's auc: 0.597624
[720]	valid_0's auc: 0.59768
[730]	valid_0's auc: 0.597821
[740]	valid_0's auc: 0.598027
[750]	valid_0's auc: 0.598149
[760]	valid_0's auc: 0.598241
[770]	valid_0's auc: 0.598366
[780]	valid_0's auc: 0.598396
[790]	valid_0's auc: 0.598423
[800]	valid_0's auc: 0.598514
[810]	valid_0's auc: 0.598697
[820]	valid_0's auc: 0.598769
[830]	valid_0's auc: 0.598924
[840]	valid_0's auc: 0.599003
[850]	valid_0's auc: 0.599028
[860]	valid_0's auc: 0.599043
[870]	valid_0's auc: 0.599077
[880]	valid_0's auc: 0.599209
[890]	valid_0's auc: 0.599268
[900]	valid_0's auc: 0.599345
[910]	valid_0's auc: 0.599463
[920]	valid_0's auc: 0.599558
[930]	valid_0's auc: 0.599608
[940]	valid_0's auc: 0.59951
[950]	valid_0's auc: 0.59969
[960]	valid_0's auc: 0.599765
[970]	valid_0's auc: 0.599823
[980]	valid_0's auc: 0.599945
[990]	valid_0's auc: 0.599975
[1000]	valid_0's auc: 0.600026
[1010]	valid_0's auc: 0.600095
[1020]	valid_0's auc: 0.600174
[1030]	valid_0's auc: 0.60023
[1040]	valid_0's auc: 0.600239
[1050]	valid_0's auc: 0.600269
[1060]	valid_0's auc: 0.600343
[1070]	valid_0's auc: 0.600378
[1080]	valid_0's auc: 0.600456
[1090]	valid_0's auc: 0.600481
[1100]	valid_0's auc: 0.600488
[1110]	valid_0's auc: 0.600479
[1120]	valid_0's auc: 0.60058
[1130]	valid_0's auc: 0.600619
[1140]	valid_0's auc: 0.600603
[1150]	valid_0's auc: 0.600641
[1160]	valid_0's auc: 0.600598
[1170]	valid_0's auc: 0.600587
[1180]	valid_0's auc: 0.600573
[1190]	valid_0's auc: 0.60055
Early stopping, best iteration is:
[1145]	valid_0's auc: 0.600652
best score: 0.600651997587
best iteration: 1145
complete on: genre_ids

--------------------
this is round: 21
artist_name and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
--------------------


After selection:
target            uint8
artist_name    category
msno           category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.528699
[20]	valid_0's auc: 0.536525
[30]	valid_0's auc: 0.54045
[40]	valid_0's auc: 0.546019
[50]	valid_0's auc: 0.549121
[60]	valid_0's auc: 0.553025
[70]	valid_0's auc: 0.556529
[80]	valid_0's auc: 0.558339
[90]	valid_0's auc: 0.560848
[100]	valid_0's auc: 0.562791
[110]	valid_0's auc: 0.564312
[120]	valid_0's auc: 0.565983
[130]	valid_0's auc: 0.567512
[140]	valid_0's auc: 0.569851
[150]	valid_0's auc: 0.571877
[160]	valid_0's auc: 0.573594
[170]	valid_0's auc: 0.575822
[180]	valid_0's auc: 0.577313
[190]	valid_0's auc: 0.578656
[200]	valid_0's auc: 0.579625
[210]	valid_0's auc: 0.580667
[220]	valid_0's auc: 0.581305
[230]	valid_0's auc: 0.581923
[240]	valid_0's auc: 0.582516
[250]	valid_0's auc: 0.583484
[260]	valid_0's auc: 0.584262
[270]	valid_0's auc: 0.584973
[280]	valid_0's auc: 0.586724
[290]	valid_0's auc: 0.587224
[300]	valid_0's auc: 0.587885
[310]	valid_0's auc: 0.588933
[320]	valid_0's auc: 0.589459
[330]	valid_0's auc: 0.590039
[340]	valid_0's auc: 0.59063
[350]	valid_0's auc: 0.590985
[360]	valid_0's auc: 0.591105
[370]	valid_0's auc: 0.591784
[380]	valid_0's auc: 0.592404
[390]	valid_0's auc: 0.593133
[400]	valid_0's auc: 0.593712
[410]	valid_0's auc: 0.593957
[420]	valid_0's auc: 0.594635
[430]	valid_0's auc: 0.595388
[440]	valid_0's auc: 0.59604
[450]	valid_0's auc: 0.596627
[460]	valid_0's auc: 0.596951
[470]	valid_0's auc: 0.597363
[480]	valid_0's auc: 0.597658
[490]	valid_0's auc: 0.598588
[500]	valid_0's auc: 0.599027
[510]	valid_0's auc: 0.599314
[520]	valid_0's auc: 0.599541
[530]	valid_0's auc: 0.599918
[540]	valid_0's auc: 0.60019
[550]	valid_0's auc: 0.600508
[560]	valid_0's auc: 0.600633
[570]	valid_0's auc: 0.601952
[580]	valid_0's auc: 0.604788
[590]	valid_0's auc: 0.604894
[600]	valid_0's auc: 0.605137
[610]	valid_0's auc: 0.605275
[620]	valid_0's auc: 0.6054
[630]	valid_0's auc: 0.605513
[640]	valid_0's auc: 0.605611
[650]	valid_0's auc: 0.605864
[660]	valid_0's auc: 0.606048
[670]	valid_0's auc: 0.60613
[680]	valid_0's auc: 0.606352
[690]	valid_0's auc: 0.606459
[700]	valid_0's auc: 0.606596
[710]	valid_0's auc: 0.606648
[720]	valid_0's auc: 0.606773
[730]	valid_0's auc: 0.60686
[740]	valid_0's auc: 0.606987
[750]	valid_0's auc: 0.607073
[760]	valid_0's auc: 0.607244
[770]	valid_0's auc: 0.607317
[780]	valid_0's auc: 0.607312
[790]	valid_0's auc: 0.607294
[800]	valid_0's auc: 0.607416
[810]	valid_0's auc: 0.607492
[820]	valid_0's auc: 0.607618
[830]	valid_0's auc: 0.607738
[840]	valid_0's auc: 0.607909
[850]	valid_0's auc: 0.608023
[860]	valid_0's auc: 0.60803
[870]	valid_0's auc: 0.608046
[880]	valid_0's auc: 0.608173
[890]	valid_0's auc: 0.608283
[900]	valid_0's auc: 0.60835
[910]	valid_0's auc: 0.608796
[920]	valid_0's auc: 0.609383
[930]	valid_0's auc: 0.609492
[940]	valid_0's auc: 0.609595
[950]	valid_0's auc: 0.609632
[960]	valid_0's auc: 0.609639
[970]	valid_0's auc: 0.609725
[980]	valid_0's auc: 0.609774
[990]	valid_0's auc: 0.609855
[1000]	valid_0's auc: 0.609964
[1010]	valid_0's auc: 0.610067
[1020]	valid_0's auc: 0.61011
[1030]	valid_0's auc: 0.61016
[1040]	valid_0's auc: 0.610193
[1050]	valid_0's auc: 0.61024
[1060]	valid_0's auc: 0.610314
[1070]	valid_0's auc: 0.610338
[1080]	valid_0's auc: 0.610422
[1090]	valid_0's auc: 0.610408
[1100]	valid_0's auc: 0.610455
[1110]	valid_0's auc: 0.610524
[1120]	valid_0's auc: 0.610514
[1130]	valid_0's auc: 0.610547
[1140]	valid_0's auc: 0.610597
[1150]	valid_0's auc: 0.610703
[1160]	valid_0's auc: 0.610762
[1170]	valid_0's auc: 0.610776
[1180]	valid_0's auc: 0.610826
[1190]	valid_0's auc: 0.610898
[1200]	valid_0's auc: 0.610943
[1210]	valid_0's auc: 0.611041
[1220]	valid_0's auc: 0.611033
[1230]	valid_0's auc: 0.611115
[1240]	valid_0's auc: 0.611073
[1250]	valid_0's auc: 0.611091
[1260]	valid_0's auc: 0.611087
[1270]	valid_0's auc: 0.611103
[1280]	valid_0's auc: 0.611102
Early stopping, best iteration is:
[1237]	valid_0's auc: 0.611162
best score: 0.611162335123
best iteration: 1237
complete on: artist_name

--------------------
this is round: 22
composer and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
--------------------


After selection:
target         uint8
composer    category
msno        category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.534699
[20]	valid_0's auc: 0.541841
[30]	valid_0's auc: 0.547255
[40]	valid_0's auc: 0.552584
[50]	valid_0's auc: 0.556163
[60]	valid_0's auc: 0.564651
[70]	valid_0's auc: 0.566822
[80]	valid_0's auc: 0.56957
[90]	valid_0's auc: 0.573112
[100]	valid_0's auc: 0.575415
[110]	valid_0's auc: 0.576798
[120]	valid_0's auc: 0.577999
[130]	valid_0's auc: 0.57943
[140]	valid_0's auc: 0.580982
[150]	valid_0's auc: 0.582183
[160]	valid_0's auc: 0.583266
[170]	valid_0's auc: 0.584406
[180]	valid_0's auc: 0.585023
[190]	valid_0's auc: 0.58581
[200]	valid_0's auc: 0.586566
[210]	valid_0's auc: 0.587969
[220]	valid_0's auc: 0.589345
[230]	valid_0's auc: 0.590604
[240]	valid_0's auc: 0.591693
[250]	valid_0's auc: 0.592527
[260]	valid_0's auc: 0.593291
[270]	valid_0's auc: 0.594106
[280]	valid_0's auc: 0.594552
[290]	valid_0's auc: 0.595125
[300]	valid_0's auc: 0.595806
[310]	valid_0's auc: 0.596062
[320]	valid_0's auc: 0.596337
[330]	valid_0's auc: 0.596676
[340]	valid_0's auc: 0.597258
[350]	valid_0's auc: 0.597585
[360]	valid_0's auc: 0.598124
[370]	valid_0's auc: 0.598343
[380]	valid_0's auc: 0.598797
[390]	valid_0's auc: 0.599584
[400]	valid_0's auc: 0.599707
[410]	valid_0's auc: 0.600229
[420]	valid_0's auc: 0.600636
[430]	valid_0's auc: 0.602127
[440]	valid_0's auc: 0.602297
[450]	valid_0's auc: 0.602619
[460]	valid_0's auc: 0.602861
[470]	valid_0's auc: 0.603226
[480]	valid_0's auc: 0.603553
[490]	valid_0's auc: 0.60363
[500]	valid_0's auc: 0.604142
[510]	valid_0's auc: 0.604299
[520]	valid_0's auc: 0.604621
[530]	valid_0's auc: 0.6049
[540]	valid_0's auc: 0.605024
[550]	valid_0's auc: 0.605253
[560]	valid_0's auc: 0.60558
[570]	valid_0's auc: 0.605647
[580]	valid_0's auc: 0.605972
[590]	valid_0's auc: 0.606191
[600]	valid_0's auc: 0.606295
[610]	valid_0's auc: 0.60639
[620]	valid_0's auc: 0.60667
[630]	valid_0's auc: 0.606845
[640]	valid_0's auc: 0.607136
[650]	valid_0's auc: 0.607371
[660]	valid_0's auc: 0.607524
[670]	valid_0's auc: 0.607682
[680]	valid_0's auc: 0.60788
[690]	valid_0's auc: 0.608073
[700]	valid_0's auc: 0.608278
[710]	valid_0's auc: 0.6083
[720]	valid_0's auc: 0.608439
[730]	valid_0's auc: 0.608591
[740]	valid_0's auc: 0.6088
[750]	valid_0's auc: 0.608896
[760]	valid_0's auc: 0.608935
[770]	valid_0's auc: 0.608981
[780]	valid_0's auc: 0.609063
[790]	valid_0's auc: 0.609225
[800]	valid_0's auc: 0.609258
[810]	valid_0's auc: 0.609209
[820]	valid_0's auc: 0.609248
[830]	valid_0's auc: 0.609265
[840]	valid_0's auc: 0.609303
[850]	valid_0's auc: 0.609368
[860]	valid_0's auc: 0.60946
[870]	valid_0's auc: 0.609492
[880]	valid_0's auc: 0.609522
[890]	valid_0's auc: 0.609557
[900]	valid_0's auc: 0.609666
[910]	valid_0's auc: 0.609799
[920]	valid_0's auc: 0.609801
[930]	valid_0's auc: 0.609962
[940]	valid_0's auc: 0.610043
[950]	valid_0's auc: 0.610097
[960]	valid_0's auc: 0.610214
[970]	valid_0's auc: 0.610271
[980]	valid_0's auc: 0.610273
[990]	valid_0's auc: 0.610306
[1000]	valid_0's auc: 0.610321
[1010]	valid_0's auc: 0.610475
[1020]	valid_0's auc: 0.610585
[1030]	valid_0's auc: 0.610584
[1040]	valid_0's auc: 0.61066
[1050]	valid_0's auc: 0.610709
[1060]	valid_0's auc: 0.610787
[1070]	valid_0's auc: 0.610867
[1080]	valid_0's auc: 0.610905
[1090]	valid_0's auc: 0.610948
[1100]	valid_0's auc: 0.611003
[1110]	valid_0's auc: 0.611006
[1120]	valid_0's auc: 0.611109
[1130]	valid_0's auc: 0.611145
[1140]	valid_0's auc: 0.611182
[1150]	valid_0's auc: 0.611211
[1160]	valid_0's auc: 0.611277
[1170]	valid_0's auc: 0.611324
[1180]	valid_0's auc: 0.611366
[1190]	valid_0's auc: 0.611383
[1200]	valid_0's auc: 0.611443
[1210]	valid_0's auc: 0.611471
[1220]	valid_0's auc: 0.61153
[1230]	valid_0's auc: 0.611596
[1240]	valid_0's auc: 0.611576
[1250]	valid_0's auc: 0.611616
[1260]	valid_0's auc: 0.61166
[1270]	valid_0's auc: 0.61169
[1280]	valid_0's auc: 0.611749
[1290]	valid_0's auc: 0.611729
[1300]	valid_0's auc: 0.611762
[1310]	valid_0's auc: 0.611816
[1320]	valid_0's auc: 0.611811
[1330]	valid_0's auc: 0.611862
[1340]	valid_0's auc: 0.611881
[1350]	valid_0's auc: 0.611921
[1360]	valid_0's auc: 0.611933
[1370]	valid_0's auc: 0.611939
[1380]	valid_0's auc: 0.611982
[1390]	valid_0's auc: 0.611972
[1400]	valid_0's auc: 0.612033
[1410]	valid_0's auc: 0.612037
[1420]	valid_0's auc: 0.612027
[1430]	valid_0's auc: 0.612001
[1440]	valid_0's auc: 0.612037
[1450]	valid_0's auc: 0.612029
[1460]	valid_0's auc: 0.612028
[1470]	valid_0's auc: 0.612012
[1480]	valid_0's auc: 0.612049
[1490]	valid_0's auc: 0.612018
[1500]	valid_0's auc: 0.612
[1510]	valid_0's auc: 0.611999
[1520]	valid_0's auc: 0.612021
[1530]	valid_0's auc: 0.612076
[1540]	valid_0's auc: 0.612095
[1550]	valid_0's auc: 0.612112
[1560]	valid_0's auc: 0.612132
[1570]	valid_0's auc: 0.612162
[1580]	valid_0's auc: 0.612165
[1590]	valid_0's auc: 0.612121
[1600]	valid_0's auc: 0.612095
[1610]	valid_0's auc: 0.612126
[1620]	valid_0's auc: 0.612165
Early stopping, best iteration is:
[1573]	valid_0's auc: 0.612179
best score: 0.612179253222
best iteration: 1573
complete on: composer

--------------------
this is round: 23
lyricist and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
--------------------


After selection:
target         uint8
lyricist    category
msno        category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.534811
[20]	valid_0's auc: 0.542458
[30]	valid_0's auc: 0.547173
[40]	valid_0's auc: 0.552821
[50]	valid_0's auc: 0.557493
[60]	valid_0's auc: 0.560573
[70]	valid_0's auc: 0.562902
[80]	valid_0's auc: 0.564625
[90]	valid_0's auc: 0.567913
[100]	valid_0's auc: 0.570859
[110]	valid_0's auc: 0.572813
[120]	valid_0's auc: 0.575027
[130]	valid_0's auc: 0.57698
[140]	valid_0's auc: 0.577583
[150]	valid_0's auc: 0.578999
[160]	valid_0's auc: 0.580126
[170]	valid_0's auc: 0.581419
[180]	valid_0's auc: 0.5829
[190]	valid_0's auc: 0.583914
[200]	valid_0's auc: 0.585516
[210]	valid_0's auc: 0.586056
[220]	valid_0's auc: 0.586606
[230]	valid_0's auc: 0.587525
[240]	valid_0's auc: 0.588103
[250]	valid_0's auc: 0.588682
[260]	valid_0's auc: 0.589355
[270]	valid_0's auc: 0.589701
[280]	valid_0's auc: 0.590849
[290]	valid_0's auc: 0.591775
[300]	valid_0's auc: 0.592484
[310]	valid_0's auc: 0.593341
[320]	valid_0's auc: 0.594074
[330]	valid_0's auc: 0.594529
[340]	valid_0's auc: 0.595005
[350]	valid_0's auc: 0.595673
[360]	valid_0's auc: 0.596222
[370]	valid_0's auc: 0.59656
[380]	valid_0's auc: 0.596946
[390]	valid_0's auc: 0.597504
[400]	valid_0's auc: 0.59805
[410]	valid_0's auc: 0.598511
[420]	valid_0's auc: 0.598807
[430]	valid_0's auc: 0.599042
[440]	valid_0's auc: 0.599257
[450]	valid_0's auc: 0.600084
[460]	valid_0's auc: 0.600327
[470]	valid_0's auc: 0.600602
[480]	valid_0's auc: 0.600838
[490]	valid_0's auc: 0.601075
[500]	valid_0's auc: 0.601206
[510]	valid_0's auc: 0.601521
[520]	valid_0's auc: 0.601769
[530]	valid_0's auc: 0.602157
[540]	valid_0's auc: 0.602521
[550]	valid_0's auc: 0.602742
[560]	valid_0's auc: 0.603036
[570]	valid_0's auc: 0.603095
[580]	valid_0's auc: 0.603301
[590]	valid_0's auc: 0.603562
[600]	valid_0's auc: 0.603866
[610]	valid_0's auc: 0.603974
[620]	valid_0's auc: 0.604034
[630]	valid_0's auc: 0.604225
[640]	valid_0's auc: 0.604368
[650]	valid_0's auc: 0.604477
[660]	valid_0's auc: 0.604737
[670]	valid_0's auc: 0.60487
[680]	valid_0's auc: 0.605608
[690]	valid_0's auc: 0.605654
[700]	valid_0's auc: 0.605716
[710]	valid_0's auc: 0.605825
[720]	valid_0's auc: 0.605962
[730]	valid_0's auc: 0.605981
[740]	valid_0's auc: 0.606075
[750]	valid_0's auc: 0.606215
[760]	valid_0's auc: 0.606355
[770]	valid_0's auc: 0.606435
[780]	valid_0's auc: 0.606537
[790]	valid_0's auc: 0.606715
[800]	valid_0's auc: 0.606738
[810]	valid_0's auc: 0.60687
[820]	valid_0's auc: 0.606958
[830]	valid_0's auc: 0.60702
[840]	valid_0's auc: 0.607098
[850]	valid_0's auc: 0.607199
[860]	valid_0's auc: 0.607298
[870]	valid_0's auc: 0.607387
[880]	valid_0's auc: 0.607479
[890]	valid_0's auc: 0.607617
[900]	valid_0's auc: 0.60769
[910]	valid_0's auc: 0.607788
[920]	valid_0's auc: 0.607865
[930]	valid_0's auc: 0.607897
[940]	valid_0's auc: 0.607957
[950]	valid_0's auc: 0.608006
[960]	valid_0's auc: 0.608106
[970]	valid_0's auc: 0.608174
[980]	valid_0's auc: 0.608243
[990]	valid_0's auc: 0.60829
[1000]	valid_0's auc: 0.608264
[1010]	valid_0's auc: 0.608324
[1020]	valid_0's auc: 0.608373
[1030]	valid_0's auc: 0.608421
[1040]	valid_0's auc: 0.608459
[1050]	valid_0's auc: 0.608455
[1060]	valid_0's auc: 0.608431
[1070]	valid_0's auc: 0.608476
[1080]	valid_0's auc: 0.608527
[1090]	valid_0's auc: 0.608577
[1100]	valid_0's auc: 0.608618
[1110]	valid_0's auc: 0.608693
[1120]	valid_0's auc: 0.608759
[1130]	valid_0's auc: 0.608752
[1140]	valid_0's auc: 0.608851
[1150]	valid_0's auc: 0.60891
[1160]	valid_0's auc: 0.608927
[1170]	valid_0's auc: 0.60897
[1180]	valid_0's auc: 0.609018
[1190]	valid_0's auc: 0.609092
[1200]	valid_0's auc: 0.609093
[1210]	valid_0's auc: 0.609118
[1220]	valid_0's auc: 0.609163
[1230]	valid_0's auc: 0.609222
[1240]	valid_0's auc: 0.609232
[1250]	valid_0's auc: 0.609261
[1260]	valid_0's auc: 0.609236
[1270]	valid_0's auc: 0.60922
[1280]	valid_0's auc: 0.609263
[1290]	valid_0's auc: 0.609318
[1300]	valid_0's auc: 0.609416
[1310]	valid_0's auc: 0.609471
[1320]	valid_0's auc: 0.609451
[1330]	valid_0's auc: 0.609461
[1340]	valid_0's auc: 0.609478
[1350]	valid_0's auc: 0.609493
[1360]	valid_0's auc: 0.609515
[1370]	valid_0's auc: 0.609518
[1380]	valid_0's auc: 0.609552
[1390]	valid_0's auc: 0.609571
[1400]	valid_0's auc: 0.609559
[1410]	valid_0's auc: 0.6096
[1420]	valid_0's auc: 0.609613
[1430]	valid_0's auc: 0.609629
[1440]	valid_0's auc: 0.609657
[1450]	valid_0's auc: 0.609651
[1460]	valid_0's auc: 0.609647
[1470]	valid_0's auc: 0.609701
[1480]	valid_0's auc: 0.6097
[1490]	valid_0's auc: 0.609756
[1500]	valid_0's auc: 0.609742
[1510]	valid_0's auc: 0.609685
[1520]	valid_0's auc: 0.609713
[1530]	valid_0's auc: 0.609745
[1540]	valid_0's auc: 0.609766
[1550]	valid_0's auc: 0.609811
[1560]	valid_0's auc: 0.609807
[1570]	valid_0's auc: 0.609804
[1580]	valid_0's auc: 0.609851
[1590]	valid_0's auc: 0.609859
[1600]	valid_0's auc: 0.609914
[1610]	valid_0's auc: 0.609939
[1620]	valid_0's auc: 0.609905
[1630]	valid_0's auc: 0.609894
[1640]	valid_0's auc: 0.609893
[1650]	valid_0's auc: 0.609882
Early stopping, best iteration is:
[1609]	valid_0's auc: 0.609956
best score: 0.609956219002
best iteration: 1609
complete on: lyricist

--------------------
this is round: 24
language and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
--------------------


After selection:
target         uint8
language    category
msno        category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.538487
[20]	valid_0's auc: 0.544105
[30]	valid_0's auc: 0.547834
[40]	valid_0's auc: 0.551364
[50]	valid_0's auc: 0.553583
[60]	valid_0's auc: 0.556262
[70]	valid_0's auc: 0.559401
[80]	valid_0's auc: 0.56255
[90]	valid_0's auc: 0.564002
[100]	valid_0's auc: 0.565458
[110]	valid_0's auc: 0.566835
[120]	valid_0's auc: 0.568171
[130]	valid_0's auc: 0.569182
[140]	valid_0's auc: 0.56997
[150]	valid_0's auc: 0.571022
[160]	valid_0's auc: 0.571205
[170]	valid_0's auc: 0.572036
[180]	valid_0's auc: 0.57358
[190]	valid_0's auc: 0.575307
[200]	valid_0's auc: 0.576989
[210]	valid_0's auc: 0.578251
[220]	valid_0's auc: 0.579443
[230]	valid_0's auc: 0.580306
[240]	valid_0's auc: 0.581122
[250]	valid_0's auc: 0.581967
[260]	valid_0's auc: 0.582598
[270]	valid_0's auc: 0.583335
[280]	valid_0's auc: 0.583929
[290]	valid_0's auc: 0.584471
[300]	valid_0's auc: 0.584754
[310]	valid_0's auc: 0.585062
[320]	valid_0's auc: 0.585588
[330]	valid_0's auc: 0.586099
[340]	valid_0's auc: 0.586645
[350]	valid_0's auc: 0.587192
[360]	valid_0's auc: 0.587447
[370]	valid_0's auc: 0.587653
[380]	valid_0's auc: 0.588172
[390]	valid_0's auc: 0.588859
[400]	valid_0's auc: 0.589244
[410]	valid_0's auc: 0.589433
[420]	valid_0's auc: 0.589786
[430]	valid_0's auc: 0.590074
[440]	valid_0's auc: 0.59014
[450]	valid_0's auc: 0.590434
[460]	valid_0's auc: 0.590904
[470]	valid_0's auc: 0.591146
[480]	valid_0's auc: 0.591088
[490]	valid_0's auc: 0.591377
[500]	valid_0's auc: 0.591801
[510]	valid_0's auc: 0.592226
[520]	valid_0's auc: 0.592344
[530]	valid_0's auc: 0.592447
[540]	valid_0's auc: 0.592598
[550]	valid_0's auc: 0.592773
[560]	valid_0's auc: 0.59294
[570]	valid_0's auc: 0.593092
[580]	valid_0's auc: 0.593152
[590]	valid_0's auc: 0.593432
[600]	valid_0's auc: 0.593678
[610]	valid_0's auc: 0.593924
[620]	valid_0's auc: 0.59426
[630]	valid_0's auc: 0.594591
[640]	valid_0's auc: 0.594748
[650]	valid_0's auc: 0.595002
[660]	valid_0's auc: 0.59534
[670]	valid_0's auc: 0.595514
[680]	valid_0's auc: 0.595766
[690]	valid_0's auc: 0.595997
[700]	valid_0's auc: 0.596017
[710]	valid_0's auc: 0.596172
[720]	valid_0's auc: 0.596202
[730]	valid_0's auc: 0.596334
[740]	valid_0's auc: 0.59651
[750]	valid_0's auc: 0.596665
[760]	valid_0's auc: 0.596658
[770]	valid_0's auc: 0.596869
[780]	valid_0's auc: 0.596924
[790]	valid_0's auc: 0.597073
[800]	valid_0's auc: 0.597166
[810]	valid_0's auc: 0.597224
[820]	valid_0's auc: 0.59734
[830]	valid_0's auc: 0.597335
[840]	valid_0's auc: 0.597422
[850]	valid_0's auc: 0.597481
[860]	valid_0's auc: 0.597523
[870]	valid_0's auc: 0.597717
[880]	valid_0's auc: 0.597744
[890]	valid_0's auc: 0.597745
[900]	valid_0's auc: 0.597761
[910]	valid_0's auc: 0.597849
[920]	valid_0's auc: 0.597957
[930]	valid_0's auc: 0.597958
[940]	valid_0's auc: 0.598001
[950]	valid_0's auc: 0.598041
[960]	valid_0's auc: 0.5981
[970]	valid_0's auc: 0.598082
[980]	valid_0's auc: 0.598177
[990]	valid_0's auc: 0.598165
[1000]	valid_0's auc: 0.598168
[1010]	valid_0's auc: 0.598161
[1020]	valid_0's auc: 0.598169
[1030]	valid_0's auc: 0.598184
[1040]	valid_0's auc: 0.598238
[1050]	valid_0's auc: 0.598291
[1060]	valid_0's auc: 0.598333
[1070]	valid_0's auc: 0.598404
[1080]	valid_0's auc: 0.598377
[1090]	valid_0's auc: 0.598334
[1100]	valid_0's auc: 0.598406
[1110]	valid_0's auc: 0.598462
[1120]	valid_0's auc: 0.598554
[1130]	valid_0's auc: 0.598595
[1140]	valid_0's auc: 0.598551
[1150]	valid_0's auc: 0.59858
[1160]	valid_0's auc: 0.598608
[1170]	valid_0's auc: 0.598671
[1180]	valid_0's auc: 0.598705
[1190]	valid_0's auc: 0.598724
[1200]	valid_0's auc: 0.598707
[1210]	valid_0's auc: 0.59873
[1220]	valid_0's auc: 0.598744
[1230]	valid_0's auc: 0.598706
[1240]	valid_0's auc: 0.598699
[1250]	valid_0's auc: 0.598692
[1260]	valid_0's auc: 0.598726
Early stopping, best iteration is:
[1213]	valid_0's auc: 0.598762
best score: 0.598761590473
best iteration: 1213
complete on: language

--------------------
this is round: 25
name and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
--------------------


After selection:
target       uint8
name      category
msno      category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.539767
[20]	valid_0's auc: 0.546491
[30]	valid_0's auc: 0.551425
[40]	valid_0's auc: 0.556786
[50]	valid_0's auc: 0.560461
[60]	valid_0's auc: 0.564269
[70]	valid_0's auc: 0.567285
[80]	valid_0's auc: 0.569371
[90]	valid_0's auc: 0.571648
[100]	valid_0's auc: 0.574483
[110]	valid_0's auc: 0.577321
[120]	valid_0's auc: 0.579417
[130]	valid_0's auc: 0.581368
[140]	valid_0's auc: 0.583084
[150]	valid_0's auc: 0.584278
[160]	valid_0's auc: 0.586491
[170]	valid_0's auc: 0.588114
[180]	valid_0's auc: 0.589199
[190]	valid_0's auc: 0.590365
[200]	valid_0's auc: 0.592794
[210]	valid_0's auc: 0.593803
[220]	valid_0's auc: 0.594401
[230]	valid_0's auc: 0.595164
[240]	valid_0's auc: 0.59558
[250]	valid_0's auc: 0.596445
[260]	valid_0's auc: 0.59678
[270]	valid_0's auc: 0.597374
[280]	valid_0's auc: 0.598319
[290]	valid_0's auc: 0.59906
[300]	valid_0's auc: 0.599714
[310]	valid_0's auc: 0.600182
[320]	valid_0's auc: 0.601141
[330]	valid_0's auc: 0.601851
[340]	valid_0's auc: 0.602504
[350]	valid_0's auc: 0.602952
[360]	valid_0's auc: 0.603706
[370]	valid_0's auc: 0.60405
[380]	valid_0's auc: 0.604632
[390]	valid_0's auc: 0.604922
[400]	valid_0's auc: 0.605287
[410]	valid_0's auc: 0.605794
[420]	valid_0's auc: 0.606193
[430]	valid_0's auc: 0.606328
[440]	valid_0's auc: 0.607283
[450]	valid_0's auc: 0.607618
[460]	valid_0's auc: 0.607767
[470]	valid_0's auc: 0.608186
[480]	valid_0's auc: 0.608498
[490]	valid_0's auc: 0.608753
[500]	valid_0's auc: 0.609039
[510]	valid_0's auc: 0.609178
[520]	valid_0's auc: 0.60943
[530]	valid_0's auc: 0.60971
[540]	valid_0's auc: 0.60984
[550]	valid_0's auc: 0.61019
[560]	valid_0's auc: 0.610435
[570]	valid_0's auc: 0.610686
[580]	valid_0's auc: 0.610848
[590]	valid_0's auc: 0.611149
[600]	valid_0's auc: 0.611278
[610]	valid_0's auc: 0.611584
[620]	valid_0's auc: 0.611866
[630]	valid_0's auc: 0.611929
[640]	valid_0's auc: 0.612122
[650]	valid_0's auc: 0.612313
[660]	valid_0's auc: 0.612462
[670]	valid_0's auc: 0.61264
[680]	valid_0's auc: 0.612754
[690]	valid_0's auc: 0.612825
[700]	valid_0's auc: 0.613043
[710]	valid_0's auc: 0.613166
[720]	valid_0's auc: 0.613329
[730]	valid_0's auc: 0.613449
[740]	valid_0's auc: 0.613568
[750]	valid_0's auc: 0.613596
[760]	valid_0's auc: 0.61371
[770]	valid_0's auc: 0.61385
[780]	valid_0's auc: 0.613995
[790]	valid_0's auc: 0.614031
[800]	valid_0's auc: 0.614119
[810]	valid_0's auc: 0.6142
[820]	valid_0's auc: 0.614289
[830]	valid_0's auc: 0.614373
[840]	valid_0's auc: 0.614428
[850]	valid_0's auc: 0.614527
[860]	valid_0's auc: 0.614736
[870]	valid_0's auc: 0.614888
[880]	valid_0's auc: 0.614944
[890]	valid_0's auc: 0.614956
[900]	valid_0's auc: 0.614956
[910]	valid_0's auc: 0.615053
[920]	valid_0's auc: 0.615164
[930]	valid_0's auc: 0.615221
[940]	valid_0's auc: 0.615283
[950]	valid_0's auc: 0.615383
[960]	valid_0's auc: 0.615431
[970]	valid_0's auc: 0.615418
[980]	valid_0's auc: 0.615438
[990]	valid_0's auc: 0.615476
[1000]	valid_0's auc: 0.615548
[1010]	valid_0's auc: 0.615601
[1020]	valid_0's auc: 0.615621
[1030]	valid_0's auc: 0.615712
[1040]	valid_0's auc: 0.615741
[1050]	valid_0's auc: 0.615732
[1060]	valid_0's auc: 0.615775
[1070]	valid_0's auc: 0.615798
[1080]	valid_0's auc: 0.615796
[1090]	valid_0's auc: 0.615875
[1100]	valid_0's auc: 0.615902
[1110]	valid_0's auc: 0.615898
[1120]	valid_0's auc: 0.615946
[1130]	valid_0's auc: 0.615937
[1140]	valid_0's auc: 0.61596
[1150]	valid_0's auc: 0.615974
[1160]	valid_0's auc: 0.615981
[1170]	valid_0's auc: 0.615969
[1180]	valid_0's auc: 0.615972
[1190]	valid_0's auc: 0.616025
[1200]	valid_0's auc: 0.616009
[1210]	valid_0's auc: 0.616036
[1220]	valid_0's auc: 0.616012
[1230]	valid_0's auc: 0.616058
[1240]	valid_0's auc: 0.616107
[1250]	valid_0's auc: 0.616088
[1260]	valid_0's auc: 0.616143
[1270]	valid_0's auc: 0.616133
[1280]	valid_0's auc: 0.616184
[1290]	valid_0's auc: 0.616221
[1300]	valid_0's auc: 0.616368
[1310]	valid_0's auc: 0.61643
[1320]	valid_0's auc: 0.616495
[1330]	valid_0's auc: 0.616663
[1340]	valid_0's auc: 0.616723
[1350]	valid_0's auc: 0.616728
[1360]	valid_0's auc: 0.616806
[1370]	valid_0's auc: 0.616843
[1380]	valid_0's auc: 0.616883
[1390]	valid_0's auc: 0.617011
[1400]	valid_0's auc: 0.617062
[1410]	valid_0's auc: 0.617087
[1420]	valid_0's auc: 0.617102
[1430]	valid_0's auc: 0.617173
[1440]	valid_0's auc: 0.617232
[1450]	valid_0's auc: 0.61727
[1460]	valid_0's auc: 0.617309
[1470]	valid_0's auc: 0.617302
[1480]	valid_0's auc: 0.617322
[1490]	valid_0's auc: 0.617354
[1500]	valid_0's auc: 0.617373
[1510]	valid_0's auc: 0.617473
[1520]	valid_0's auc: 0.617536
[1530]	valid_0's auc: 0.617572
[1540]	valid_0's auc: 0.617605
[1550]	valid_0's auc: 0.617582
[1560]	valid_0's auc: 0.617652
[1570]	valid_0's auc: 0.617687
[1580]	valid_0's auc: 0.617708
[1590]	valid_0's auc: 0.617729
[1600]	valid_0's auc: 0.617742
[1610]	valid_0's auc: 0.617732
[1620]	valid_0's auc: 0.617745
[1630]	valid_0's auc: 0.617747
[1640]	valid_0's auc: 0.61784
[1650]	valid_0's auc: 0.617893
[1660]	valid_0's auc: 0.617895
[1670]	valid_0's auc: 0.617908
[1680]	valid_0's auc: 0.617941
[1690]	valid_0's auc: 0.617963
[1700]	valid_0's auc: 0.618
[1710]	valid_0's auc: 0.618002
[1720]	valid_0's auc: 0.617999
[1730]	valid_0's auc: 0.618004
[1740]	valid_0's auc: 0.617982
[1750]	valid_0's auc: 0.618026
[1760]	valid_0's auc: 0.618078
[1770]	valid_0's auc: 0.618094
[1780]	valid_0's auc: 0.618084
[1790]	valid_0's auc: 0.618081
[1800]	valid_0's auc: 0.618079
[1810]	valid_0's auc: 0.618082
[1820]	valid_0's auc: 0.618096
[1830]	valid_0's auc: 0.618103
[1840]	valid_0's auc: 0.618101
[1850]	valid_0's auc: 0.618088
[1860]	valid_0's auc: 0.618078
[1870]	valid_0's auc: 0.6181
Early stopping, best iteration is:
[1824]	valid_0's auc: 0.618113
best score: 0.61811297759
best iteration: 1824
complete on: name

--------------------
this is round: 26
genre_ids_fre_song and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
--------------------


After selection:
target                   uint8
genre_ids_fre_song    category
msno                  category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.543153
[20]	valid_0's auc: 0.544281
[30]	valid_0's auc: 0.548665
[40]	valid_0's auc: 0.552537
[50]	valid_0's auc: 0.555429
[60]	valid_0's auc: 0.559092
[70]	valid_0's auc: 0.561611
[80]	valid_0's auc: 0.563006
[90]	valid_0's auc: 0.564376
[100]	valid_0's auc: 0.565704
[110]	valid_0's auc: 0.567547
[120]	valid_0's auc: 0.569441
[130]	valid_0's auc: 0.571231
[140]	valid_0's auc: 0.573162
[150]	valid_0's auc: 0.574512
[160]	valid_0's auc: 0.575726
[170]	valid_0's auc: 0.577015
[180]	valid_0's auc: 0.577894
[190]	valid_0's auc: 0.578894
[200]	valid_0's auc: 0.579703
[210]	valid_0's auc: 0.580365
[220]	valid_0's auc: 0.581179
[230]	valid_0's auc: 0.581532
[240]	valid_0's auc: 0.582083
[250]	valid_0's auc: 0.582756
[260]	valid_0's auc: 0.583235
[270]	valid_0's auc: 0.583992
[280]	valid_0's auc: 0.58438
[290]	valid_0's auc: 0.585026
[300]	valid_0's auc: 0.585441
[310]	valid_0's auc: 0.585828
[320]	valid_0's auc: 0.586299
[330]	valid_0's auc: 0.586646
[340]	valid_0's auc: 0.586926
[350]	valid_0's auc: 0.58773
[360]	valid_0's auc: 0.588308
[370]	valid_0's auc: 0.58908
[380]	valid_0's auc: 0.589599
[390]	valid_0's auc: 0.590034
[400]	valid_0's auc: 0.590484
[410]	valid_0's auc: 0.591071
[420]	valid_0's auc: 0.591524
[430]	valid_0's auc: 0.591914
[440]	valid_0's auc: 0.592244
[450]	valid_0's auc: 0.592691
[460]	valid_0's auc: 0.593082
[470]	valid_0's auc: 0.593472
[480]	valid_0's auc: 0.593758
[490]	valid_0's auc: 0.594064
[500]	valid_0's auc: 0.594293
[510]	valid_0's auc: 0.594474
[520]	valid_0's auc: 0.594802
[530]	valid_0's auc: 0.594908
[540]	valid_0's auc: 0.595103
[550]	valid_0's auc: 0.595193
[560]	valid_0's auc: 0.595267
[570]	valid_0's auc: 0.595408
[580]	valid_0's auc: 0.595587
[590]	valid_0's auc: 0.595841
[600]	valid_0's auc: 0.595987
[610]	valid_0's auc: 0.59624
[620]	valid_0's auc: 0.596392
[630]	valid_0's auc: 0.596713
[640]	valid_0's auc: 0.59686
[650]	valid_0's auc: 0.597118
[660]	valid_0's auc: 0.597262
[670]	valid_0's auc: 0.597398
[680]	valid_0's auc: 0.597535
[690]	valid_0's auc: 0.597603
[700]	valid_0's auc: 0.597657
[710]	valid_0's auc: 0.597707
[720]	valid_0's auc: 0.597815
[730]	valid_0's auc: 0.598008
[740]	valid_0's auc: 0.598079
[750]	valid_0's auc: 0.598255
[760]	valid_0's auc: 0.598287
[770]	valid_0's auc: 0.598381
[780]	valid_0's auc: 0.598503
[790]	valid_0's auc: 0.59873
[800]	valid_0's auc: 0.598803
[810]	valid_0's auc: 0.598789
[820]	valid_0's auc: 0.598897
[830]	valid_0's auc: 0.599048
[840]	valid_0's auc: 0.599123
[850]	valid_0's auc: 0.5992
[860]	valid_0's auc: 0.59923
[870]	valid_0's auc: 0.59928
[880]	valid_0's auc: 0.599184
[890]	valid_0's auc: 0.599268
[900]	valid_0's auc: 0.59938
[910]	valid_0's auc: 0.599442
[920]	valid_0's auc: 0.599488
[930]	valid_0's auc: 0.599522
[940]	valid_0's auc: 0.599646
[950]	valid_0's auc: 0.599742
[960]	valid_0's auc: 0.599863
[970]	valid_0's auc: 0.599931
[980]	valid_0's auc: 0.600037
[990]	valid_0's auc: 0.600081
[1000]	valid_0's auc: 0.600144
[1010]	valid_0's auc: 0.600262
[1020]	valid_0's auc: 0.600263
[1030]	valid_0's auc: 0.600293
[1040]	valid_0's auc: 0.600363
[1050]	valid_0's auc: 0.600444
[1060]	valid_0's auc: 0.600522
[1070]	valid_0's auc: 0.60057
[1080]	valid_0's auc: 0.600617
[1090]	valid_0's auc: 0.60065
[1100]	valid_0's auc: 0.600716
[1110]	valid_0's auc: 0.600736
[1120]	valid_0's auc: 0.600761
[1130]	valid_0's auc: 0.600787
[1140]	valid_0's auc: 0.600811
[1150]	valid_0's auc: 0.600887
[1160]	valid_0's auc: 0.600911
[1170]	valid_0's auc: 0.600984
[1180]	valid_0's auc: 0.601029
[1190]	valid_0's auc: 0.601029
[1200]	valid_0's auc: 0.601052
[1210]	valid_0's auc: 0.601091
[1220]	valid_0's auc: 0.601129
[1230]	valid_0's auc: 0.601106
[1240]	valid_0's auc: 0.601127
[1250]	valid_0's auc: 0.601148
[1260]	valid_0's auc: 0.601182
[1270]	valid_0's auc: 0.601213
[1280]	valid_0's auc: 0.601206
[1290]	valid_0's auc: 0.601252
[1300]	valid_0's auc: 0.601273
[1310]	valid_0's auc: 0.60133
[1320]	valid_0's auc: 0.601363
[1330]	valid_0's auc: 0.60139
[1340]	valid_0's auc: 0.601444
[1350]	valid_0's auc: 0.601486
[1360]	valid_0's auc: 0.601471
[1370]	valid_0's auc: 0.601478
[1380]	valid_0's auc: 0.601514
[1390]	valid_0's auc: 0.601537
[1400]	valid_0's auc: 0.601576
[1410]	valid_0's auc: 0.601586
[1420]	valid_0's auc: 0.601599
[1430]	valid_0's auc: 0.601618
[1440]	valid_0's auc: 0.60171
[1450]	valid_0's auc: 0.601707
[1460]	valid_0's auc: 0.601747
[1470]	valid_0's auc: 0.601754
[1480]	valid_0's auc: 0.601817
[1490]	valid_0's auc: 0.601823
[1500]	valid_0's auc: 0.601844
[1510]	valid_0's auc: 0.601863
[1520]	valid_0's auc: 0.601912
[1530]	valid_0's auc: 0.601902
[1540]	valid_0's auc: 0.601885
[1550]	valid_0's auc: 0.601861
[1560]	valid_0's auc: 0.60187
[1570]	valid_0's auc: 0.601929
[1580]	valid_0's auc: 0.601959
[1590]	valid_0's auc: 0.601964
[1600]	valid_0's auc: 0.601969
[1610]	valid_0's auc: 0.601961
[1620]	valid_0's auc: 0.601957
[1630]	valid_0's auc: 0.601951
Early stopping, best iteration is:
[1588]	valid_0's auc: 0.601987
best score: 0.601987175709
best iteration: 1588
complete on: genre_ids_fre_song

--------------------
this is round: 27
song_year_fre_song and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
--------------------


After selection:
target                   uint8
song_year_fre_song    category
msno                  category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.544048
[20]	valid_0's auc: 0.549024
[30]	valid_0's auc: 0.553578
[40]	valid_0's auc: 0.556601
[50]	valid_0's auc: 0.559732
[60]	valid_0's auc: 0.561921
[70]	valid_0's auc: 0.563872
[80]	valid_0's auc: 0.566121
[90]	valid_0's auc: 0.569234
[100]	valid_0's auc: 0.57108
[110]	valid_0's auc: 0.572825
[120]	valid_0's auc: 0.574175
[130]	valid_0's auc: 0.575431
[140]	valid_0's auc: 0.576387
[150]	valid_0's auc: 0.57741
[160]	valid_0's auc: 0.578325
[170]	valid_0's auc: 0.579305
[180]	valid_0's auc: 0.579986
[190]	valid_0's auc: 0.580629
[200]	valid_0's auc: 0.58132
[210]	valid_0's auc: 0.582001
[220]	valid_0's auc: 0.582527
[230]	valid_0's auc: 0.583096
[240]	valid_0's auc: 0.583493
[250]	valid_0's auc: 0.584253
[260]	valid_0's auc: 0.584892
[270]	valid_0's auc: 0.585648
[280]	valid_0's auc: 0.586442
[290]	valid_0's auc: 0.58726
[300]	valid_0's auc: 0.587831
[310]	valid_0's auc: 0.588516
[320]	valid_0's auc: 0.588953
[330]	valid_0's auc: 0.589395
[340]	valid_0's auc: 0.589925
[350]	valid_0's auc: 0.590391
[360]	valid_0's auc: 0.590875
[370]	valid_0's auc: 0.591288
[380]	valid_0's auc: 0.591595
[390]	valid_0's auc: 0.591862
[400]	valid_0's auc: 0.592208
[410]	valid_0's auc: 0.592503
[420]	valid_0's auc: 0.59278
[430]	valid_0's auc: 0.593123
[440]	valid_0's auc: 0.593299
[450]	valid_0's auc: 0.593406
[460]	valid_0's auc: 0.593607
[470]	valid_0's auc: 0.593837
[480]	valid_0's auc: 0.594052
[490]	valid_0's auc: 0.59423
[500]	valid_0's auc: 0.594539
[510]	valid_0's auc: 0.594796
[520]	valid_0's auc: 0.594953
[530]	valid_0's auc: 0.595142
[540]	valid_0's auc: 0.595354
[550]	valid_0's auc: 0.595485
[560]	valid_0's auc: 0.595598
[570]	valid_0's auc: 0.595851
[580]	valid_0's auc: 0.596085
[590]	valid_0's auc: 0.596278
[600]	valid_0's auc: 0.596386
[610]	valid_0's auc: 0.596485
[620]	valid_0's auc: 0.596649
[630]	valid_0's auc: 0.596827
[640]	valid_0's auc: 0.597528
[650]	valid_0's auc: 0.597704
[660]	valid_0's auc: 0.597759
[670]	valid_0's auc: 0.597845
[680]	valid_0's auc: 0.597997
[690]	valid_0's auc: 0.598256
[700]	valid_0's auc: 0.598348
[710]	valid_0's auc: 0.598476
[720]	valid_0's auc: 0.598553
[730]	valid_0's auc: 0.598586
[740]	valid_0's auc: 0.598814
[750]	valid_0's auc: 0.59892
[760]	valid_0's auc: 0.599029
[770]	valid_0's auc: 0.599097
[780]	valid_0's auc: 0.599131
[790]	valid_0's auc: 0.599205
[800]	valid_0's auc: 0.599504
[810]	valid_0's auc: 0.599566
[820]	valid_0's auc: 0.599657
[830]	valid_0's auc: 0.599717
[840]	valid_0's auc: 0.599705
[850]	valid_0's auc: 0.599777
[860]	valid_0's auc: 0.599817
[870]	valid_0's auc: 0.599902
[880]	valid_0's auc: 0.599904
[890]	valid_0's auc: 0.599932
[900]	valid_0's auc: 0.600072
[910]	valid_0's auc: 0.600169
[920]	valid_0's auc: 0.600239
[930]	valid_0's auc: 0.600306
[940]	valid_0's auc: 0.600365
[950]	valid_0's auc: 0.600448
[960]	valid_0's auc: 0.600623
[970]	valid_0's auc: 0.600671
[980]	valid_0's auc: 0.600707
[990]	valid_0's auc: 0.6008
[1000]	valid_0's auc: 0.600906
[1010]	valid_0's auc: 0.600939
[1020]	valid_0's auc: 0.600966
[1030]	valid_0's auc: 0.601165
[1040]	valid_0's auc: 0.601223
[1050]	valid_0's auc: 0.601281
[1060]	valid_0's auc: 0.601289
[1070]	valid_0's auc: 0.601427
[1080]	valid_0's auc: 0.601454
[1090]	valid_0's auc: 0.601523
[1100]	valid_0's auc: 0.601521
[1110]	valid_0's auc: 0.601562
[1120]	valid_0's auc: 0.601566
[1130]	valid_0's auc: 0.6016
[1140]	valid_0's auc: 0.60161
[1150]	valid_0's auc: 0.601618
[1160]	valid_0's auc: 0.601587
[1170]	valid_0's auc: 0.601671
[1180]	valid_0's auc: 0.601644
[1190]	valid_0's auc: 0.601678
[1200]	valid_0's auc: 0.601674
[1210]	valid_0's auc: 0.601681
[1220]	valid_0's auc: 0.601702
[1230]	valid_0's auc: 0.601804
[1240]	valid_0's auc: 0.601844
[1250]	valid_0's auc: 0.601863
[1260]	valid_0's auc: 0.601929
[1270]	valid_0's auc: 0.601982
[1280]	valid_0's auc: 0.601983
[1290]	valid_0's auc: 0.601989
[1300]	valid_0's auc: 0.602102
[1310]	valid_0's auc: 0.602116
[1320]	valid_0's auc: 0.602173
[1330]	valid_0's auc: 0.602213
[1340]	valid_0's auc: 0.602175
[1350]	valid_0's auc: 0.602215
[1360]	valid_0's auc: 0.602206
[1370]	valid_0's auc: 0.602239
[1380]	valid_0's auc: 0.602282
[1390]	valid_0's auc: 0.602349
[1400]	valid_0's auc: 0.602341
[1410]	valid_0's auc: 0.60238
[1420]	valid_0's auc: 0.602367
[1430]	valid_0's auc: 0.602438
[1440]	valid_0's auc: 0.602459
[1450]	valid_0's auc: 0.602455
[1460]	valid_0's auc: 0.602434
[1470]	valid_0's auc: 0.602433
[1480]	valid_0's auc: 0.602409
Early stopping, best iteration is:
[1435]	valid_0's auc: 0.602468
best score: 0.602468415627
best iteration: 1435
complete on: song_year_fre_song

--------------------
this is round: 28
song_year and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
--------------------


After selection:
target          uint8
song_year    category
msno         category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.546161
[20]	valid_0's auc: 0.551662
[30]	valid_0's auc: 0.5555
[40]	valid_0's auc: 0.558788
[50]	valid_0's auc: 0.561164
[60]	valid_0's auc: 0.563414
[70]	valid_0's auc: 0.565224
[80]	valid_0's auc: 0.566819
[90]	valid_0's auc: 0.569171
[100]	valid_0's auc: 0.571387
[110]	valid_0's auc: 0.573648
[120]	valid_0's auc: 0.57545
[130]	valid_0's auc: 0.577004
[140]	valid_0's auc: 0.577976
[150]	valid_0's auc: 0.578909
[160]	valid_0's auc: 0.580033
[170]	valid_0's auc: 0.581044
[180]	valid_0's auc: 0.581646
[190]	valid_0's auc: 0.582258
[200]	valid_0's auc: 0.582942
[210]	valid_0's auc: 0.583426
[220]	valid_0's auc: 0.58405
[230]	valid_0's auc: 0.584701
[240]	valid_0's auc: 0.585232
[250]	valid_0's auc: 0.585544
[260]	valid_0's auc: 0.586254
[270]	valid_0's auc: 0.587012
[280]	valid_0's auc: 0.587466
[290]	valid_0's auc: 0.587909
[300]	valid_0's auc: 0.588253
[310]	valid_0's auc: 0.589134
[320]	valid_0's auc: 0.58964
[330]	valid_0's auc: 0.590242
[340]	valid_0's auc: 0.590768
[350]	valid_0's auc: 0.591174
[360]	valid_0's auc: 0.591778
[370]	valid_0's auc: 0.59221
[380]	valid_0's auc: 0.592533
[390]	valid_0's auc: 0.592999
[400]	valid_0's auc: 0.593327
[410]	valid_0's auc: 0.593631
[420]	valid_0's auc: 0.593946
[430]	valid_0's auc: 0.594272
[440]	valid_0's auc: 0.594468
[450]	valid_0's auc: 0.594746
[460]	valid_0's auc: 0.595045
[470]	valid_0's auc: 0.595227
[480]	valid_0's auc: 0.595418
[490]	valid_0's auc: 0.59564
[500]	valid_0's auc: 0.595716
[510]	valid_0's auc: 0.595863
[520]	valid_0's auc: 0.596175
[530]	valid_0's auc: 0.596273
[540]	valid_0's auc: 0.596515
[550]	valid_0's auc: 0.59664
[560]	valid_0's auc: 0.596834
[570]	valid_0's auc: 0.597008
[580]	valid_0's auc: 0.597138
[590]	valid_0's auc: 0.597265
[600]	valid_0's auc: 0.597471
[610]	valid_0's auc: 0.597653
[620]	valid_0's auc: 0.597805
[630]	valid_0's auc: 0.597975
[640]	valid_0's auc: 0.598209
[650]	valid_0's auc: 0.598966
[660]	valid_0's auc: 0.59903
[670]	valid_0's auc: 0.599103
[680]	valid_0's auc: 0.599318
[690]	valid_0's auc: 0.599705
[700]	valid_0's auc: 0.599814
[710]	valid_0's auc: 0.599932
[720]	valid_0's auc: 0.600003
[730]	valid_0's auc: 0.600086
[740]	valid_0's auc: 0.600189
[750]	valid_0's auc: 0.600774
[760]	valid_0's auc: 0.600969
[770]	valid_0's auc: 0.601061
[780]	valid_0's auc: 0.601143
[790]	valid_0's auc: 0.601223
[800]	valid_0's auc: 0.601242
[810]	valid_0's auc: 0.601479
[820]	valid_0's auc: 0.601541
[830]	valid_0's auc: 0.601592
[840]	valid_0's auc: 0.601617
[850]	valid_0's auc: 0.601686
[860]	valid_0's auc: 0.601781
[870]	valid_0's auc: 0.601793
[880]	valid_0's auc: 0.601819
[890]	valid_0's auc: 0.601875
[900]	valid_0's auc: 0.601917
[910]	valid_0's auc: 0.601938
[920]	valid_0's auc: 0.601965
[930]	valid_0's auc: 0.602073
[940]	valid_0's auc: 0.602159
[950]	valid_0's auc: 0.602192
[960]	valid_0's auc: 0.60223
[970]	valid_0's auc: 0.602248
[980]	valid_0's auc: 0.602348
[990]	valid_0's auc: 0.602531
[1000]	valid_0's auc: 0.602616
[1010]	valid_0's auc: 0.60266
[1020]	valid_0's auc: 0.602685
[1030]	valid_0's auc: 0.602736
[1040]	valid_0's auc: 0.602795
[1050]	valid_0's auc: 0.602826
[1060]	valid_0's auc: 0.602867
[1070]	valid_0's auc: 0.602918
[1080]	valid_0's auc: 0.602956
[1090]	valid_0's auc: 0.60299
[1100]	valid_0's auc: 0.603088
[1110]	valid_0's auc: 0.603126
[1120]	valid_0's auc: 0.603152
[1130]	valid_0's auc: 0.603156
[1140]	valid_0's auc: 0.603205
[1150]	valid_0's auc: 0.603189
[1160]	valid_0's auc: 0.603203
[1170]	valid_0's auc: 0.603223
[1180]	valid_0's auc: 0.603237
[1190]	valid_0's auc: 0.603258
[1200]	valid_0's auc: 0.603275
[1210]	valid_0's auc: 0.603281
[1220]	valid_0's auc: 0.603292
[1230]	valid_0's auc: 0.603293
[1240]	valid_0's auc: 0.603353
[1250]	valid_0's auc: 0.603347
[1260]	valid_0's auc: 0.603339
[1270]	valid_0's auc: 0.603368
[1280]	valid_0's auc: 0.603498
[1290]	valid_0's auc: 0.603542
[1300]	valid_0's auc: 0.603538
[1310]	valid_0's auc: 0.603625
[1320]	valid_0's auc: 0.603638
[1330]	valid_0's auc: 0.603634
[1340]	valid_0's auc: 0.603654
[1350]	valid_0's auc: 0.603703
[1360]	valid_0's auc: 0.603696
[1370]	valid_0's auc: 0.603701
[1380]	valid_0's auc: 0.603717
[1390]	valid_0's auc: 0.603724
[1400]	valid_0's auc: 0.60375
[1410]	valid_0's auc: 0.603771
[1420]	valid_0's auc: 0.603784
[1430]	valid_0's auc: 0.603874
[1440]	valid_0's auc: 0.603874
[1450]	valid_0's auc: 0.603907
[1460]	valid_0's auc: 0.603943
[1470]	valid_0's auc: 0.603962
[1480]	valid_0's auc: 0.60394
[1490]	valid_0's auc: 0.603942
[1500]	valid_0's auc: 0.604013
[1510]	valid_0's auc: 0.604079
[1520]	valid_0's auc: 0.604092
[1530]	valid_0's auc: 0.604101
[1540]	valid_0's auc: 0.604082
[1550]	valid_0's auc: 0.604092
[1560]	valid_0's auc: 0.604091
[1570]	valid_0's auc: 0.604091
[1580]	valid_0's auc: 0.6041
Early stopping, best iteration is:
[1534]	valid_0's auc: 0.604116
best score: 0.604116109102
best iteration: 1534
complete on: song_year

--------------------
this is round: 29
song_country_fre_song and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
['song_year', 'msno']
--------------------


After selection:
target                      uint8
song_country_fre_song    category
msno                     category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.53773
[20]	valid_0's auc: 0.544315
[30]	valid_0's auc: 0.548004
[40]	valid_0's auc: 0.550903
[50]	valid_0's auc: 0.553308
[60]	valid_0's auc: 0.556059
[70]	valid_0's auc: 0.558628
[80]	valid_0's auc: 0.561553
[90]	valid_0's auc: 0.563388
[100]	valid_0's auc: 0.564998
[110]	valid_0's auc: 0.566355
[120]	valid_0's auc: 0.567374
[130]	valid_0's auc: 0.568552
[140]	valid_0's auc: 0.56949
[150]	valid_0's auc: 0.570566
[160]	valid_0's auc: 0.571561
[170]	valid_0's auc: 0.572112
[180]	valid_0's auc: 0.572938
[190]	valid_0's auc: 0.574631
[200]	valid_0's auc: 0.576227
[210]	valid_0's auc: 0.57759
[220]	valid_0's auc: 0.578824
[230]	valid_0's auc: 0.58002
[240]	valid_0's auc: 0.580899
[250]	valid_0's auc: 0.581832
[260]	valid_0's auc: 0.582569
[270]	valid_0's auc: 0.583085
[280]	valid_0's auc: 0.583641
[290]	valid_0's auc: 0.584069
[300]	valid_0's auc: 0.584643
[310]	valid_0's auc: 0.585231
[320]	valid_0's auc: 0.585732
[330]	valid_0's auc: 0.585941
[340]	valid_0's auc: 0.586277
[350]	valid_0's auc: 0.58658
[360]	valid_0's auc: 0.586862
[370]	valid_0's auc: 0.587321
[380]	valid_0's auc: 0.587452
[390]	valid_0's auc: 0.587752
[400]	valid_0's auc: 0.588206
[410]	valid_0's auc: 0.588683
[420]	valid_0's auc: 0.589104
[430]	valid_0's auc: 0.589365
[440]	valid_0's auc: 0.589683
[450]	valid_0's auc: 0.590034
[460]	valid_0's auc: 0.590274
[470]	valid_0's auc: 0.590456
[480]	valid_0's auc: 0.590785
[490]	valid_0's auc: 0.590983
[500]	valid_0's auc: 0.5911
[510]	valid_0's auc: 0.591433
[520]	valid_0's auc: 0.591674
[530]	valid_0's auc: 0.591914
[540]	valid_0's auc: 0.592049
[550]	valid_0's auc: 0.592175
[560]	valid_0's auc: 0.592392
[570]	valid_0's auc: 0.592591
[580]	valid_0's auc: 0.592674
[590]	valid_0's auc: 0.592921
[600]	valid_0's auc: 0.593038
[610]	valid_0's auc: 0.593075
[620]	valid_0's auc: 0.5933
[630]	valid_0's auc: 0.593385
[640]	valid_0's auc: 0.59346
[650]	valid_0's auc: 0.59352
[660]	valid_0's auc: 0.593609
[670]	valid_0's auc: 0.59376
[680]	valid_0's auc: 0.593976
[690]	valid_0's auc: 0.594124
[700]	valid_0's auc: 0.594427
[710]	valid_0's auc: 0.594597
[720]	valid_0's auc: 0.594756
[730]	valid_0's auc: 0.594773
[740]	valid_0's auc: 0.594987
[750]	valid_0's auc: 0.595175
[760]	valid_0's auc: 0.595369
[770]	valid_0's auc: 0.595453
[780]	valid_0's auc: 0.595628
[790]	valid_0's auc: 0.595681
[800]	valid_0's auc: 0.595854
[810]	valid_0's auc: 0.595932
[820]	valid_0's auc: 0.596028
[830]	valid_0's auc: 0.596147
[840]	valid_0's auc: 0.59621
[850]	valid_0's auc: 0.596297
[860]	valid_0's auc: 0.596352
[870]	valid_0's auc: 0.59646
[880]	valid_0's auc: 0.59648
[890]	valid_0's auc: 0.596598
[900]	valid_0's auc: 0.596798
[910]	valid_0's auc: 0.596935
[920]	valid_0's auc: 0.59693
[930]	valid_0's auc: 0.596943
[940]	valid_0's auc: 0.596976
[950]	valid_0's auc: 0.597051
[960]	valid_0's auc: 0.59715
[970]	valid_0's auc: 0.597175
[980]	valid_0's auc: 0.597181
[990]	valid_0's auc: 0.59726
[1000]	valid_0's auc: 0.59733
[1010]	valid_0's auc: 0.597416
[1020]	valid_0's auc: 0.597344
[1030]	valid_0's auc: 0.597362
[1040]	valid_0's auc: 0.597352
[1050]	valid_0's auc: 0.597389
[1060]	valid_0's auc: 0.597441
[1070]	valid_0's auc: 0.597443
[1080]	valid_0's auc: 0.597483
[1090]	valid_0's auc: 0.59743
[1100]	valid_0's auc: 0.597489
[1110]	valid_0's auc: 0.597482
[1120]	valid_0's auc: 0.597453
[1130]	valid_0's auc: 0.597502
[1140]	valid_0's auc: 0.597489
[1150]	valid_0's auc: 0.597533
[1160]	valid_0's auc: 0.597569
[1170]	valid_0's auc: 0.597575
[1180]	valid_0's auc: 0.597629
[1190]	valid_0's auc: 0.597659
[1200]	valid_0's auc: 0.597735
[1210]	valid_0's auc: 0.597722
[1220]	valid_0's auc: 0.59774
[1230]	valid_0's auc: 0.597773
[1240]	valid_0's auc: 0.597769
[1250]	valid_0's auc: 0.597821
[1260]	valid_0's auc: 0.597848
[1270]	valid_0's auc: 0.597827
[1280]	valid_0's auc: 0.597797
[1290]	valid_0's auc: 0.597842
[1300]	valid_0's auc: 0.597831
[1310]	valid_0's auc: 0.59785
[1320]	valid_0's auc: 0.597889
[1330]	valid_0's auc: 0.59786
[1340]	valid_0's auc: 0.597901
[1350]	valid_0's auc: 0.597957
[1360]	valid_0's auc: 0.597944
[1370]	valid_0's auc: 0.597929
[1380]	valid_0's auc: 0.597932
[1390]	valid_0's auc: 0.597941
Early stopping, best iteration is:
[1347]	valid_0's auc: 0.597998
best score: 0.597997716877
best iteration: 1347
complete on: song_country_fre_song

--------------------
this is round: 30
song_country and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
['song_year', 'msno']
['song_country_fre_song', 'msno']
--------------------


After selection:
target             uint8
song_country    category
msno            category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537434
[20]	valid_0's auc: 0.543034
[30]	valid_0's auc: 0.547496
[40]	valid_0's auc: 0.551636
[50]	valid_0's auc: 0.553627
[60]	valid_0's auc: 0.555948
[70]	valid_0's auc: 0.559121
[80]	valid_0's auc: 0.561161
[90]	valid_0's auc: 0.562912
[100]	valid_0's auc: 0.564356
[110]	valid_0's auc: 0.565511
[120]	valid_0's auc: 0.566821
[130]	valid_0's auc: 0.568096
[140]	valid_0's auc: 0.569667
[150]	valid_0's auc: 0.571544
[160]	valid_0's auc: 0.57328
[170]	valid_0's auc: 0.574658
[180]	valid_0's auc: 0.576063
[190]	valid_0's auc: 0.577427
[200]	valid_0's auc: 0.577946
[210]	valid_0's auc: 0.578935
[220]	valid_0's auc: 0.579625
[230]	valid_0's auc: 0.58001
[240]	valid_0's auc: 0.580675
[250]	valid_0's auc: 0.581128
[260]	valid_0's auc: 0.581553
[270]	valid_0's auc: 0.582147
[280]	valid_0's auc: 0.582485
[290]	valid_0's auc: 0.582985
[300]	valid_0's auc: 0.583413
[310]	valid_0's auc: 0.583886
[320]	valid_0's auc: 0.584323
[330]	valid_0's auc: 0.584789
[340]	valid_0's auc: 0.585107
[350]	valid_0's auc: 0.585618
[360]	valid_0's auc: 0.585941
[370]	valid_0's auc: 0.586095
[380]	valid_0's auc: 0.586465
[390]	valid_0's auc: 0.586699
[400]	valid_0's auc: 0.587362
[410]	valid_0's auc: 0.587856
[420]	valid_0's auc: 0.588294
[430]	valid_0's auc: 0.588767
[440]	valid_0's auc: 0.589312
[450]	valid_0's auc: 0.589756
[460]	valid_0's auc: 0.590136
[470]	valid_0's auc: 0.590415
[480]	valid_0's auc: 0.590911
[490]	valid_0's auc: 0.591277
[500]	valid_0's auc: 0.59166
[510]	valid_0's auc: 0.591887
[520]	valid_0's auc: 0.59219
[530]	valid_0's auc: 0.592481
[540]	valid_0's auc: 0.59267
[550]	valid_0's auc: 0.592822
[560]	valid_0's auc: 0.593225
[570]	valid_0's auc: 0.59357
[580]	valid_0's auc: 0.593808
[590]	valid_0's auc: 0.593944
[600]	valid_0's auc: 0.594123
[610]	valid_0's auc: 0.594326
[620]	valid_0's auc: 0.594469
[630]	valid_0's auc: 0.594627
[640]	valid_0's auc: 0.594662
[650]	valid_0's auc: 0.59491
[660]	valid_0's auc: 0.595026
[670]	valid_0's auc: 0.595164
[680]	valid_0's auc: 0.595298
[690]	valid_0's auc: 0.595493
[700]	valid_0's auc: 0.59567
[710]	valid_0's auc: 0.595766
[720]	valid_0's auc: 0.59592
[730]	valid_0's auc: 0.596
[740]	valid_0's auc: 0.596111
[750]	valid_0's auc: 0.596127
[760]	valid_0's auc: 0.596203
[770]	valid_0's auc: 0.596315
[780]	valid_0's auc: 0.596422
[790]	valid_0's auc: 0.596442
[800]	valid_0's auc: 0.596579
[810]	valid_0's auc: 0.596695
[820]	valid_0's auc: 0.596745
[830]	valid_0's auc: 0.596788
[840]	valid_0's auc: 0.596837
[850]	valid_0's auc: 0.596963
[860]	valid_0's auc: 0.597096
[870]	valid_0's auc: 0.597211
[880]	valid_0's auc: 0.597283
[890]	valid_0's auc: 0.597287
[900]	valid_0's auc: 0.597331
[910]	valid_0's auc: 0.597374
[920]	valid_0's auc: 0.597389
[930]	valid_0's auc: 0.597367
[940]	valid_0's auc: 0.597419
[950]	valid_0's auc: 0.597436
[960]	valid_0's auc: 0.597483
[970]	valid_0's auc: 0.597503
[980]	valid_0's auc: 0.597554
[990]	valid_0's auc: 0.597602
[1000]	valid_0's auc: 0.59764
[1010]	valid_0's auc: 0.597659
[1020]	valid_0's auc: 0.597727
[1030]	valid_0's auc: 0.597727
[1040]	valid_0's auc: 0.597729
[1050]	valid_0's auc: 0.597737
[1060]	valid_0's auc: 0.597786
[1070]	valid_0's auc: 0.597846
[1080]	valid_0's auc: 0.597896
[1090]	valid_0's auc: 0.59788
[1100]	valid_0's auc: 0.597878
[1110]	valid_0's auc: 0.597883
[1120]	valid_0's auc: 0.597904
[1130]	valid_0's auc: 0.59793
[1140]	valid_0's auc: 0.59797
[1150]	valid_0's auc: 0.598037
[1160]	valid_0's auc: 0.598062
[1170]	valid_0's auc: 0.598081
[1180]	valid_0's auc: 0.598117
[1190]	valid_0's auc: 0.598147
[1200]	valid_0's auc: 0.598185
[1210]	valid_0's auc: 0.598249
[1220]	valid_0's auc: 0.598253
[1230]	valid_0's auc: 0.598267
[1240]	valid_0's auc: 0.598361
[1250]	valid_0's auc: 0.598393
[1260]	valid_0's auc: 0.598376
[1270]	valid_0's auc: 0.598434
[1280]	valid_0's auc: 0.598455
[1290]	valid_0's auc: 0.598462
[1300]	valid_0's auc: 0.598505
[1310]	valid_0's auc: 0.598495
[1320]	valid_0's auc: 0.598533
[1330]	valid_0's auc: 0.598562
[1340]	valid_0's auc: 0.598567
[1350]	valid_0's auc: 0.598744
[1360]	valid_0's auc: 0.598773
[1370]	valid_0's auc: 0.598771
[1380]	valid_0's auc: 0.598816
[1390]	valid_0's auc: 0.598831
[1400]	valid_0's auc: 0.598817
[1410]	valid_0's auc: 0.598811
[1420]	valid_0's auc: 0.598811
[1430]	valid_0's auc: 0.598838
[1440]	valid_0's auc: 0.598811
[1450]	valid_0's auc: 0.598834
[1460]	valid_0's auc: 0.598841
[1470]	valid_0's auc: 0.598874
[1480]	valid_0's auc: 0.59887
[1490]	valid_0's auc: 0.59886
[1500]	valid_0's auc: 0.59885
[1510]	valid_0's auc: 0.598839
[1520]	valid_0's auc: 0.598869
Early stopping, best iteration is:
[1471]	valid_0's auc: 0.59888
best score: 0.598879848554
best iteration: 1471
complete on: song_country

--------------------
this is round: 31
rc and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
['song_year', 'msno']
['song_country_fre_song', 'msno']
['song_country', 'msno']
--------------------


After selection:
target       uint8
rc        category
msno      category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537544
[20]	valid_0's auc: 0.545026
[30]	valid_0's auc: 0.549786
[40]	valid_0's auc: 0.553246
[50]	valid_0's auc: 0.556438
[60]	valid_0's auc: 0.56026
[70]	valid_0's auc: 0.56367
[80]	valid_0's auc: 0.565877
[90]	valid_0's auc: 0.567662
[100]	valid_0's auc: 0.569306
[110]	valid_0's auc: 0.571231
[120]	valid_0's auc: 0.572081
[130]	valid_0's auc: 0.573357
[140]	valid_0's auc: 0.574594
[150]	valid_0's auc: 0.581678
[160]	valid_0's auc: 0.582332
[170]	valid_0's auc: 0.582963
[180]	valid_0's auc: 0.584307
[190]	valid_0's auc: 0.58566
[200]	valid_0's auc: 0.586937
[210]	valid_0's auc: 0.588346
[220]	valid_0's auc: 0.588963
[230]	valid_0's auc: 0.589951
[240]	valid_0's auc: 0.590933
[250]	valid_0's auc: 0.59136
[260]	valid_0's auc: 0.5919
[270]	valid_0's auc: 0.592711
[280]	valid_0's auc: 0.593205
[290]	valid_0's auc: 0.593498
[300]	valid_0's auc: 0.593948
[310]	valid_0's auc: 0.594438
[320]	valid_0's auc: 0.594747
[330]	valid_0's auc: 0.595079
[340]	valid_0's auc: 0.595461
[350]	valid_0's auc: 0.595809
[360]	valid_0's auc: 0.596193
[370]	valid_0's auc: 0.596683
[380]	valid_0's auc: 0.596952
[390]	valid_0's auc: 0.597279
[400]	valid_0's auc: 0.597717
[410]	valid_0's auc: 0.598028
[420]	valid_0's auc: 0.598299
[430]	valid_0's auc: 0.598571
[440]	valid_0's auc: 0.598816
[450]	valid_0's auc: 0.598991
[460]	valid_0's auc: 0.599227
[470]	valid_0's auc: 0.599537
[480]	valid_0's auc: 0.599808
[490]	valid_0's auc: 0.600064
[500]	valid_0's auc: 0.600507
[510]	valid_0's auc: 0.600637
[520]	valid_0's auc: 0.601081
[530]	valid_0's auc: 0.601155
[540]	valid_0's auc: 0.601309
[550]	valid_0's auc: 0.601393
[560]	valid_0's auc: 0.601501
[570]	valid_0's auc: 0.601603
[580]	valid_0's auc: 0.601813
[590]	valid_0's auc: 0.601931
[600]	valid_0's auc: 0.602013
[610]	valid_0's auc: 0.602221
[620]	valid_0's auc: 0.602465
[630]	valid_0's auc: 0.602598
[640]	valid_0's auc: 0.602777
[650]	valid_0's auc: 0.60281
[660]	valid_0's auc: 0.602992
[670]	valid_0's auc: 0.603192
[680]	valid_0's auc: 0.603396
[690]	valid_0's auc: 0.603592
[700]	valid_0's auc: 0.603781
[710]	valid_0's auc: 0.603827
[720]	valid_0's auc: 0.603949
[730]	valid_0's auc: 0.604125
[740]	valid_0's auc: 0.60418
[750]	valid_0's auc: 0.60425
[760]	valid_0's auc: 0.604287
[770]	valid_0's auc: 0.604391
[780]	valid_0's auc: 0.604426
[790]	valid_0's auc: 0.604447
[800]	valid_0's auc: 0.604497
[810]	valid_0's auc: 0.604573
[820]	valid_0's auc: 0.604781
[830]	valid_0's auc: 0.604858
[840]	valid_0's auc: 0.604914
[850]	valid_0's auc: 0.604961
[860]	valid_0's auc: 0.60502
[870]	valid_0's auc: 0.605096
[880]	valid_0's auc: 0.605176
[890]	valid_0's auc: 0.605187
[900]	valid_0's auc: 0.605289
[910]	valid_0's auc: 0.605353
[920]	valid_0's auc: 0.605443
[930]	valid_0's auc: 0.605469
[940]	valid_0's auc: 0.605514
[950]	valid_0's auc: 0.605514
[960]	valid_0's auc: 0.605613
[970]	valid_0's auc: 0.605729
[980]	valid_0's auc: 0.605773
[990]	valid_0's auc: 0.605764
[1000]	valid_0's auc: 0.605802
[1010]	valid_0's auc: 0.605869
[1020]	valid_0's auc: 0.605889
[1030]	valid_0's auc: 0.605938
[1040]	valid_0's auc: 0.605971
[1050]	valid_0's auc: 0.606021
[1060]	valid_0's auc: 0.606073
[1070]	valid_0's auc: 0.6061
[1080]	valid_0's auc: 0.606125
[1090]	valid_0's auc: 0.606132
[1100]	valid_0's auc: 0.606172
[1110]	valid_0's auc: 0.606146
[1120]	valid_0's auc: 0.606162
[1130]	valid_0's auc: 0.606232
[1140]	valid_0's auc: 0.606244
[1150]	valid_0's auc: 0.606267
[1160]	valid_0's auc: 0.606278
[1170]	valid_0's auc: 0.606313
[1180]	valid_0's auc: 0.606334
[1190]	valid_0's auc: 0.606326
[1200]	valid_0's auc: 0.606397
[1210]	valid_0's auc: 0.606424
[1220]	valid_0's auc: 0.606388
[1230]	valid_0's auc: 0.60641
[1240]	valid_0's auc: 0.606463
[1250]	valid_0's auc: 0.606479
[1260]	valid_0's auc: 0.606515
[1270]	valid_0's auc: 0.606536
[1280]	valid_0's auc: 0.606534
[1290]	valid_0's auc: 0.606538
[1300]	valid_0's auc: 0.606573
[1310]	valid_0's auc: 0.606592
[1320]	valid_0's auc: 0.606596
[1330]	valid_0's auc: 0.606586
[1340]	valid_0's auc: 0.606602
[1350]	valid_0's auc: 0.606605
[1360]	valid_0's auc: 0.606671
[1370]	valid_0's auc: 0.606666
[1380]	valid_0's auc: 0.606671
[1390]	valid_0's auc: 0.606766
[1400]	valid_0's auc: 0.606878
[1410]	valid_0's auc: 0.60685
[1420]	valid_0's auc: 0.606901
[1430]	valid_0's auc: 0.606891
[1440]	valid_0's auc: 0.606888
[1450]	valid_0's auc: 0.606943
[1460]	valid_0's auc: 0.606948
[1470]	valid_0's auc: 0.606958
[1480]	valid_0's auc: 0.606977
[1490]	valid_0's auc: 0.606982
[1500]	valid_0's auc: 0.606996
[1510]	valid_0's auc: 0.607027
[1520]	valid_0's auc: 0.607059
[1530]	valid_0's auc: 0.607076
[1540]	valid_0's auc: 0.607092
[1550]	valid_0's auc: 0.607104
[1560]	valid_0's auc: 0.607116
[1570]	valid_0's auc: 0.607149
[1580]	valid_0's auc: 0.607165
[1590]	valid_0's auc: 0.607167
[1600]	valid_0's auc: 0.607176
[1610]	valid_0's auc: 0.607155
[1620]	valid_0's auc: 0.607204
[1630]	valid_0's auc: 0.607185
[1640]	valid_0's auc: 0.607186
[1650]	valid_0's auc: 0.60717
[1660]	valid_0's auc: 0.607168
[1670]	valid_0's auc: 0.607175
Early stopping, best iteration is:
[1623]	valid_0's auc: 0.607218
best score: 0.607217640383
best iteration: 1623
complete on: rc

--------------------
this is round: 32
source_system_tab_guess and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
['song_year', 'msno']
['song_country_fre_song', 'msno']
['song_country', 'msno']
['rc', 'msno']
--------------------


After selection:
target                        uint8
source_system_tab_guess    category
msno                       category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.593627
[20]	valid_0's auc: 0.597348
[30]	valid_0's auc: 0.60028
[40]	valid_0's auc: 0.602419
[50]	valid_0's auc: 0.604546
[60]	valid_0's auc: 0.605897
[70]	valid_0's auc: 0.606668
[80]	valid_0's auc: 0.60816
[90]	valid_0's auc: 0.609418
[100]	valid_0's auc: 0.610535
[110]	valid_0's auc: 0.611238
[120]	valid_0's auc: 0.611821
[130]	valid_0's auc: 0.612547
[140]	valid_0's auc: 0.613676
[150]	valid_0's auc: 0.614642
[160]	valid_0's auc: 0.61536
[170]	valid_0's auc: 0.616074
[180]	valid_0's auc: 0.616818
[190]	valid_0's auc: 0.617496
[200]	valid_0's auc: 0.618062
[210]	valid_0's auc: 0.61847
[220]	valid_0's auc: 0.619013
[230]	valid_0's auc: 0.619374
[240]	valid_0's auc: 0.619821
[250]	valid_0's auc: 0.620167
[260]	valid_0's auc: 0.620468
[270]	valid_0's auc: 0.620856
[280]	valid_0's auc: 0.621247
[290]	valid_0's auc: 0.621452
[300]	valid_0's auc: 0.621701
[310]	valid_0's auc: 0.621979
[320]	valid_0's auc: 0.622046
[330]	valid_0's auc: 0.622661
[340]	valid_0's auc: 0.622841
[350]	valid_0's auc: 0.623238
[360]	valid_0's auc: 0.62344
[370]	valid_0's auc: 0.623747
[380]	valid_0's auc: 0.624074
[390]	valid_0's auc: 0.624264
[400]	valid_0's auc: 0.624728
[410]	valid_0's auc: 0.625247
[420]	valid_0's auc: 0.625506
[430]	valid_0's auc: 0.625804
[440]	valid_0's auc: 0.626169
[450]	valid_0's auc: 0.626454
[460]	valid_0's auc: 0.626814
[470]	valid_0's auc: 0.627103
[480]	valid_0's auc: 0.627414
[490]	valid_0's auc: 0.627578
[500]	valid_0's auc: 0.627688
[510]	valid_0's auc: 0.627789
[520]	valid_0's auc: 0.627932
[530]	valid_0's auc: 0.628094
[540]	valid_0's auc: 0.62827
[550]	valid_0's auc: 0.628422
[560]	valid_0's auc: 0.628558
[570]	valid_0's auc: 0.62867
[580]	valid_0's auc: 0.628752
[590]	valid_0's auc: 0.628956
[600]	valid_0's auc: 0.629171
[610]	valid_0's auc: 0.629341
[620]	valid_0's auc: 0.629361
[630]	valid_0's auc: 0.629411
[640]	valid_0's auc: 0.629466
[650]	valid_0's auc: 0.629594
[660]	valid_0's auc: 0.629584
[670]	valid_0's auc: 0.629686
[680]	valid_0's auc: 0.629726
[690]	valid_0's auc: 0.629786
[700]	valid_0's auc: 0.629878
[710]	valid_0's auc: 0.629998
[720]	valid_0's auc: 0.630052
[730]	valid_0's auc: 0.630089
[740]	valid_0's auc: 0.630227
[750]	valid_0's auc: 0.630354
[760]	valid_0's auc: 0.630388
[770]	valid_0's auc: 0.630423
[780]	valid_0's auc: 0.6305
[790]	valid_0's auc: 0.630556
[800]	valid_0's auc: 0.630614
[810]	valid_0's auc: 0.630666
[820]	valid_0's auc: 0.63076
[830]	valid_0's auc: 0.630829
[840]	valid_0's auc: 0.630844
[850]	valid_0's auc: 0.630901
[860]	valid_0's auc: 0.631009
[870]	valid_0's auc: 0.63103
[880]	valid_0's auc: 0.631077
[890]	valid_0's auc: 0.631084
[900]	valid_0's auc: 0.631152
[910]	valid_0's auc: 0.63118
[920]	valid_0's auc: 0.63116
[930]	valid_0's auc: 0.631166
[940]	valid_0's auc: 0.63124
[950]	valid_0's auc: 0.631247
[960]	valid_0's auc: 0.63121
[970]	valid_0's auc: 0.631251
[980]	valid_0's auc: 0.631294
[990]	valid_0's auc: 0.631602
[1000]	valid_0's auc: 0.631643
[1010]	valid_0's auc: 0.631681
[1020]	valid_0's auc: 0.631747
[1030]	valid_0's auc: 0.631762
[1040]	valid_0's auc: 0.631779
[1050]	valid_0's auc: 0.631795
[1060]	valid_0's auc: 0.631861
[1070]	valid_0's auc: 0.631878
[1080]	valid_0's auc: 0.631925
[1090]	valid_0's auc: 0.632062
[1100]	valid_0's auc: 0.632145
[1110]	valid_0's auc: 0.632234
[1120]	valid_0's auc: 0.632266
[1130]	valid_0's auc: 0.632303
[1140]	valid_0's auc: 0.632362
[1150]	valid_0's auc: 0.632416
[1160]	valid_0's auc: 0.632433
[1170]	valid_0's auc: 0.632493
[1180]	valid_0's auc: 0.632525
[1190]	valid_0's auc: 0.632539
[1200]	valid_0's auc: 0.632552
[1210]	valid_0's auc: 0.632573
[1220]	valid_0's auc: 0.63261
[1230]	valid_0's auc: 0.632683
[1240]	valid_0's auc: 0.632719
[1250]	valid_0's auc: 0.63276
[1260]	valid_0's auc: 0.632842
[1270]	valid_0's auc: 0.632899
[1280]	valid_0's auc: 0.632944
[1290]	valid_0's auc: 0.632931
[1300]	valid_0's auc: 0.632947
[1310]	valid_0's auc: 0.632945
[1320]	valid_0's auc: 0.632997
[1330]	valid_0's auc: 0.633054
[1340]	valid_0's auc: 0.633089
[1350]	valid_0's auc: 0.633133
[1360]	valid_0's auc: 0.633202
[1370]	valid_0's auc: 0.633236
[1380]	valid_0's auc: 0.633267
[1390]	valid_0's auc: 0.633333
[1400]	valid_0's auc: 0.633352
[1410]	valid_0's auc: 0.633383
[1420]	valid_0's auc: 0.633451
[1430]	valid_0's auc: 0.633486
[1440]	valid_0's auc: 0.633624
[1450]	valid_0's auc: 0.633637
[1460]	valid_0's auc: 0.633633
[1470]	valid_0's auc: 0.633638
[1480]	valid_0's auc: 0.633618
[1490]	valid_0's auc: 0.633656
[1500]	valid_0's auc: 0.633701
[1510]	valid_0's auc: 0.633705
[1520]	valid_0's auc: 0.633729
[1530]	valid_0's auc: 0.633718
[1540]	valid_0's auc: 0.633735
[1550]	valid_0's auc: 0.633738
[1560]	valid_0's auc: 0.633746
[1570]	valid_0's auc: 0.63376
[1580]	valid_0's auc: 0.633769
[1590]	valid_0's auc: 0.633774
[1600]	valid_0's auc: 0.633782
[1610]	valid_0's auc: 0.633757
[1620]	valid_0's auc: 0.633803
[1630]	valid_0's auc: 0.633833
[1640]	valid_0's auc: 0.633988
[1650]	valid_0's auc: 0.634009
[1660]	valid_0's auc: 0.633994
[1670]	valid_0's auc: 0.634034
[1680]	valid_0's auc: 0.634026
[1690]	valid_0's auc: 0.634036
[1700]	valid_0's auc: 0.634029
[1710]	valid_0's auc: 0.63406
[1720]	valid_0's auc: 0.634119
[1730]	valid_0's auc: 0.634126
[1740]	valid_0's auc: 0.634152
[1750]	valid_0's auc: 0.634163
[1760]	valid_0's auc: 0.634183
[1770]	valid_0's auc: 0.634167
[1780]	valid_0's auc: 0.634252
[1790]	valid_0's auc: 0.634445
[1800]	valid_0's auc: 0.634446
[1810]	valid_0's auc: 0.634468
[1820]	valid_0's auc: 0.634471
[1830]	valid_0's auc: 0.634481
[1840]	valid_0's auc: 0.634616
[1850]	valid_0's auc: 0.634617
[1860]	valid_0's auc: 0.634627
[1870]	valid_0's auc: 0.634657
[1880]	valid_0's auc: 0.634755
[1890]	valid_0's auc: 0.634757
[1900]	valid_0's auc: 0.634767
[1910]	valid_0's auc: 0.634771
[1920]	valid_0's auc: 0.634851
[1930]	valid_0's auc: 0.634856
[1940]	valid_0's auc: 0.634862
[1950]	valid_0's auc: 0.634969
[1960]	valid_0's auc: 0.634954
[1970]	valid_0's auc: 0.634959
[1980]	valid_0's auc: 0.634956
[1990]	valid_0's auc: 0.634946
[2000]	valid_0's auc: 0.634959
[2010]	valid_0's auc: 0.634961
Early stopping, best iteration is:
[1967]	valid_0's auc: 0.634973
best score: 0.634972521906
best iteration: 1967
complete on: source_system_tab_guess

--------------------
this is round: 33
source_screen_name_guess and msno are not in [DONE]:
['song_id', 'msno']
['source_system_tab', 'msno']
['source_screen_name', 'msno']
['source_type', 'msno']
['city', 'msno']
['registered_via', 'msno']
['sex', 'msno']
['sex_guess1', 'msno']
['sex_guess2', 'msno']
['sex_guess3', 'msno']
['sex_guess4', 'msno']
['sex_guess5', 'msno']
['sex_freq_member', 'msno']
['registration_year', 'msno']
['registration_month', 'msno']
['registration_date', 'msno']
['expiration_year', 'msno']
['expiration_month', 'msno']
['expiration_date', 'msno']
['genre_ids', 'msno']
['artist_name', 'msno']
['composer', 'msno']
['lyricist', 'msno']
['language', 'msno']
['name', 'msno']
['genre_ids_fre_song', 'msno']
['song_year_fre_song', 'msno']
['song_year', 'msno']
['song_country_fre_song', 'msno']
['song_country', 'msno']
['rc', 'msno']
['source_system_tab_guess', 'msno']
--------------------


After selection:
target                         uint8
source_screen_name_guess    category
msno                        category
dtype: object
number of columns: 3


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.596079
[20]	valid_0's auc: 0.599175
[30]	valid_0's auc: 0.601617
[40]	valid_0's auc: 0.603474
[50]	valid_0's auc: 0.605338
[60]	valid_0's auc: 0.606363
[70]	valid_0's auc: 0.607221
[80]	valid_0's auc: 0.608129
[90]	valid_0's auc: 0.609098
[100]	valid_0's auc: 0.610202
[110]	valid_0's auc: 0.611626
[120]	valid_0's auc: 0.612495
[130]	valid_0's auc: 0.613468
[140]	valid_0's auc: 0.614379
[150]	valid_0's auc: 0.615003
[160]	valid_0's auc: 0.615628
[170]	valid_0's auc: 0.616272
[180]	valid_0's auc: 0.616841
[190]	valid_0's auc: 0.617142
[200]	valid_0's auc: 0.61752
[210]	valid_0's auc: 0.617916
[220]	valid_0's auc: 0.618262
[230]	valid_0's auc: 0.618537
[240]	valid_0's auc: 0.619032
[250]	valid_0's auc: 0.619336
[260]	valid_0's auc: 0.619596
[270]	valid_0's auc: 0.620016
[280]	valid_0's auc: 0.620691
[290]	valid_0's auc: 0.621237
[300]	valid_0's auc: 0.621731
[310]	valid_0's auc: 0.622075
[320]	valid_0's auc: 0.622397
[330]	valid_0's auc: 0.622785
[340]	valid_0's auc: 0.6233
[350]	valid_0's auc: 0.623713
[360]	valid_0's auc: 0.624069
[370]	valid_0's auc: 0.624349
[380]	valid_0's auc: 0.624571
[390]	valid_0's auc: 0.624913
[400]	valid_0's auc: 0.625148
[410]	valid_0's auc: 0.62547
[420]	valid_0's auc: 0.625723
[430]	valid_0's auc: 0.625933
[440]	valid_0's auc: 0.626059
[450]	valid_0's auc: 0.626108
[460]	valid_0's auc: 0.626139
[470]	valid_0's auc: 0.626223
[480]	valid_0's auc: 0.626418
[490]	valid_0's auc: 0.626616
[500]	valid_0's auc: 0.6267
[510]	valid_0's auc: 0.626892
[520]	valid_0's auc: 0.627018
[530]	valid_0's auc: 0.627244
[540]	valid_0's auc: 0.627303
[550]	valid_0's auc: 0.627411
[560]	valid_0's auc: 0.627529
[570]	valid_0's auc: 0.627595
[580]	valid_0's auc: 0.627667
[590]	valid_0's auc: 0.627731
[600]	valid_0's auc: 0.627899
[610]	valid_0's auc: 0.628058
[620]	valid_0's auc: 0.62814
[630]	valid_0's auc: 0.628248
[640]	valid_0's auc: 0.628345
[650]	valid_0's auc: 0.62857
[660]	valid_0's auc: 0.628728
[670]	valid_0's auc: 0.628879
[680]	valid_0's auc: 0.628971
[690]	valid_0's auc: 0.62913
[700]	valid_0's auc: 0.62925
[710]	valid_0's auc: 0.629395
[720]	valid_0's auc: 0.629489
[730]	valid_0's auc: 0.629644
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/drill_train_and_compare_V1003/B_two_in_column_V1001.py", line 159, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 199, in train
    booster.update(fobj=fobj)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1507, in update
    ctypes.byref(is_finished)))
KeyboardInterrupt

Process finished with exit code 1
'''