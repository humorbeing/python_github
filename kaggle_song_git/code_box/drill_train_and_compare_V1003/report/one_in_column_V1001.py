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

for w in df.columns:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/drill_train_and_compare_V1003/one_in_column_member_count_int.py
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
working on: msno


After selection:
target       uint8
msno      category
dtype: object
number of columns: 2


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
[10]	valid_0's auc: 0.513742
[20]	valid_0's auc: 0.521552
[30]	valid_0's auc: 0.526134
[40]	valid_0's auc: 0.53059
[50]	valid_0's auc: 0.534324
[60]	valid_0's auc: 0.537327
[70]	valid_0's auc: 0.54176
[80]	valid_0's auc: 0.54575
[90]	valid_0's auc: 0.548862
[100]	valid_0's auc: 0.551104
[110]	valid_0's auc: 0.553091
[120]	valid_0's auc: 0.554665
[130]	valid_0's auc: 0.555609
[140]	valid_0's auc: 0.557124
[150]	valid_0's auc: 0.558635
[160]	valid_0's auc: 0.559866
[170]	valid_0's auc: 0.561019
[180]	valid_0's auc: 0.561951
[190]	valid_0's auc: 0.562859
[200]	valid_0's auc: 0.563861
[210]	valid_0's auc: 0.564943
[220]	valid_0's auc: 0.565919
[230]	valid_0's auc: 0.568306
[240]	valid_0's auc: 0.569875
[250]	valid_0's auc: 0.571203
[260]	valid_0's auc: 0.572606
[270]	valid_0's auc: 0.573873
[280]	valid_0's auc: 0.574371
[290]	valid_0's auc: 0.575444
[300]	valid_0's auc: 0.576526
[310]	valid_0's auc: 0.577034
[320]	valid_0's auc: 0.578219
[330]	valid_0's auc: 0.578681
[340]	valid_0's auc: 0.579104
[350]	valid_0's auc: 0.579654
[360]	valid_0's auc: 0.580365
[370]	valid_0's auc: 0.581292
[380]	valid_0's auc: 0.581438
[390]	valid_0's auc: 0.581692
[400]	valid_0's auc: 0.582197
[410]	valid_0's auc: 0.582499
[420]	valid_0's auc: 0.582754
[430]	valid_0's auc: 0.58306
[440]	valid_0's auc: 0.583787
[450]	valid_0's auc: 0.584438
[460]	valid_0's auc: 0.584704
[470]	valid_0's auc: 0.584943
[480]	valid_0's auc: 0.58509
[490]	valid_0's auc: 0.585146
[500]	valid_0's auc: 0.585409
[510]	valid_0's auc: 0.585743
[520]	valid_0's auc: 0.586184
[530]	valid_0's auc: 0.586483
[540]	valid_0's auc: 0.586692
[550]	valid_0's auc: 0.587045
[560]	valid_0's auc: 0.587298
[570]	valid_0's auc: 0.587624
[580]	valid_0's auc: 0.587793
[590]	valid_0's auc: 0.58781
[600]	valid_0's auc: 0.587942
[610]	valid_0's auc: 0.588261
[620]	valid_0's auc: 0.588477
[630]	valid_0's auc: 0.588666
[640]	valid_0's auc: 0.588711
[650]	valid_0's auc: 0.589009
[660]	valid_0's auc: 0.589181
[670]	valid_0's auc: 0.589261
[680]	valid_0's auc: 0.589317
[690]	valid_0's auc: 0.589207
[700]	valid_0's auc: 0.589526
[710]	valid_0's auc: 0.589687
[720]	valid_0's auc: 0.589787
[730]	valid_0's auc: 0.590015
[740]	valid_0's auc: 0.59022
[750]	valid_0's auc: 0.590524
[760]	valid_0's auc: 0.590674
[770]	valid_0's auc: 0.590779
[780]	valid_0's auc: 0.590905
[790]	valid_0's auc: 0.590847
[800]	valid_0's auc: 0.590962
[810]	valid_0's auc: 0.590978
[820]	valid_0's auc: 0.591036
[830]	valid_0's auc: 0.591205
[840]	valid_0's auc: 0.591308
[850]	valid_0's auc: 0.591284
[860]	valid_0's auc: 0.591395
[870]	valid_0's auc: 0.59147
[880]	valid_0's auc: 0.591495
[890]	valid_0's auc: 0.591643
[900]	valid_0's auc: 0.591605
[910]	valid_0's auc: 0.591595
[920]	valid_0's auc: 0.591781
[930]	valid_0's auc: 0.59184
[940]	valid_0's auc: 0.591825
[950]	valid_0's auc: 0.591892
[960]	valid_0's auc: 0.592066
[970]	valid_0's auc: 0.592088
[980]	valid_0's auc: 0.592133
[990]	valid_0's auc: 0.592181
[1000]	valid_0's auc: 0.592161
[1010]	valid_0's auc: 0.592179
[1020]	valid_0's auc: 0.592185
[1030]	valid_0's auc: 0.592172
[1040]	valid_0's auc: 0.592266
[1050]	valid_0's auc: 0.592323
[1060]	valid_0's auc: 0.592602
[1070]	valid_0's auc: 0.592711
[1080]	valid_0's auc: 0.59271
[1090]	valid_0's auc: 0.592755
[1100]	valid_0's auc: 0.592867
[1110]	valid_0's auc: 0.592886
[1120]	valid_0's auc: 0.592794
[1130]	valid_0's auc: 0.592803
[1140]	valid_0's auc: 0.592848
[1150]	valid_0's auc: 0.592923
[1160]	valid_0's auc: 0.59294
[1170]	valid_0's auc: 0.593033
[1180]	valid_0's auc: 0.593032
[1190]	valid_0's auc: 0.593018
[1200]	valid_0's auc: 0.592994
[1210]	valid_0's auc: 0.592995
[1220]	valid_0's auc: 0.593014
[1230]	valid_0's auc: 0.593108
[1240]	valid_0's auc: 0.593177
[1250]	valid_0's auc: 0.593175
[1260]	valid_0's auc: 0.593161
[1270]	valid_0's auc: 0.593173
[1280]	valid_0's auc: 0.593224
[1290]	valid_0's auc: 0.593091
[1300]	valid_0's auc: 0.593141
[1310]	valid_0's auc: 0.593182
[1320]	valid_0's auc: 0.593252
[1330]	valid_0's auc: 0.593248
[1340]	valid_0's auc: 0.593209
[1350]	valid_0's auc: 0.593205
[1360]	valid_0's auc: 0.593192
Early stopping, best iteration is:
[1318]	valid_0's auc: 0.59326
best score: 0.59325952624
best iteration: 1318
complete on: msno

working on: song_id


After selection:
target        uint8
song_id    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537073
[20]	valid_0's auc: 0.542826
[30]	valid_0's auc: 0.544325
[40]	valid_0's auc: 0.545282
[50]	valid_0's auc: 0.546046
[60]	valid_0's auc: 0.546649
[70]	valid_0's auc: 0.547095
[80]	valid_0's auc: 0.550072
[90]	valid_0's auc: 0.553146
[100]	valid_0's auc: 0.555086
[110]	valid_0's auc: 0.556483
[120]	valid_0's auc: 0.557529
[130]	valid_0's auc: 0.558405
[140]	valid_0's auc: 0.558942
[150]	valid_0's auc: 0.559707
[160]	valid_0's auc: 0.560512
[170]	valid_0's auc: 0.56088
[180]	valid_0's auc: 0.561441
[190]	valid_0's auc: 0.561661
[200]	valid_0's auc: 0.562042
[210]	valid_0's auc: 0.56231
[220]	valid_0's auc: 0.562577
[230]	valid_0's auc: 0.562706
[240]	valid_0's auc: 0.562815
[250]	valid_0's auc: 0.562888
[260]	valid_0's auc: 0.562988
[270]	valid_0's auc: 0.563098
[280]	valid_0's auc: 0.563253
[290]	valid_0's auc: 0.5634
[300]	valid_0's auc: 0.563471
[310]	valid_0's auc: 0.563719
[320]	valid_0's auc: 0.564028
[330]	valid_0's auc: 0.564246
[340]	valid_0's auc: 0.564411
[350]	valid_0's auc: 0.564548
[360]	valid_0's auc: 0.564726
[370]	valid_0's auc: 0.564982
[380]	valid_0's auc: 0.565126
[390]	valid_0's auc: 0.565245
[400]	valid_0's auc: 0.565344
[410]	valid_0's auc: 0.565466
[420]	valid_0's auc: 0.565622
[430]	valid_0's auc: 0.565703
[440]	valid_0's auc: 0.565811
[450]	valid_0's auc: 0.565933
[460]	valid_0's auc: 0.566007
[470]	valid_0's auc: 0.566076
[480]	valid_0's auc: 0.566137
[490]	valid_0's auc: 0.566165
[500]	valid_0's auc: 0.566218
[510]	valid_0's auc: 0.566259
[520]	valid_0's auc: 0.566319
[530]	valid_0's auc: 0.566374
[540]	valid_0's auc: 0.566455
[550]	valid_0's auc: 0.566555
[560]	valid_0's auc: 0.566617
[570]	valid_0's auc: 0.566631
[580]	valid_0's auc: 0.566701
[590]	valid_0's auc: 0.566786
[600]	valid_0's auc: 0.566797
[610]	valid_0's auc: 0.566814
[620]	valid_0's auc: 0.56685
[630]	valid_0's auc: 0.566856
[640]	valid_0's auc: 0.566889
[650]	valid_0's auc: 0.566937
[660]	valid_0's auc: 0.567006
[670]	valid_0's auc: 0.567029
[680]	valid_0's auc: 0.567083
[690]	valid_0's auc: 0.567094
[700]	valid_0's auc: 0.567113
[710]	valid_0's auc: 0.56712
[720]	valid_0's auc: 0.567128
[730]	valid_0's auc: 0.567164
[740]	valid_0's auc: 0.567181
[750]	valid_0's auc: 0.567214
[760]	valid_0's auc: 0.56727
[770]	valid_0's auc: 0.567315
[780]	valid_0's auc: 0.567279
[790]	valid_0's auc: 0.567304
[800]	valid_0's auc: 0.567339
[810]	valid_0's auc: 0.567364
[820]	valid_0's auc: 0.567374
[830]	valid_0's auc: 0.567375
[840]	valid_0's auc: 0.567406
[850]	valid_0's auc: 0.567391
[860]	valid_0's auc: 0.567412
[870]	valid_0's auc: 0.567467
[880]	valid_0's auc: 0.567471
[890]	valid_0's auc: 0.56747
[900]	valid_0's auc: 0.567441
[910]	valid_0's auc: 0.567428
[920]	valid_0's auc: 0.567402
[930]	valid_0's auc: 0.567416
Early stopping, best iteration is:
[886]	valid_0's auc: 0.567479
best score: 0.567479069058
best iteration: 886
complete on: song_id

working on: source_system_tab


After selection:
target                  uint8
source_system_tab    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.585163
[20]	valid_0's auc: 0.58519
[30]	valid_0's auc: 0.58519
[40]	valid_0's auc: 0.585177
[50]	valid_0's auc: 0.585177
[60]	valid_0's auc: 0.585177
Early stopping, best iteration is:
[14]	valid_0's auc: 0.58519
best score: 0.585189781567
best iteration: 14
complete on: source_system_tab

working on: source_screen_name


After selection:
target                   uint8
source_screen_name    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.593036
[20]	valid_0's auc: 0.593036
[30]	valid_0's auc: 0.593043
[40]	valid_0's auc: 0.593167
[50]	valid_0's auc: 0.593167
[60]	valid_0's auc: 0.593167
[70]	valid_0's auc: 0.593167
[80]	valid_0's auc: 0.593167
Early stopping, best iteration is:
[35]	valid_0's auc: 0.593167
best score: 0.59316707849
best iteration: 35
complete on: source_screen_name

working on: source_type


After selection:
target            uint8
source_type    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.591709
[20]	valid_0's auc: 0.591709
[30]	valid_0's auc: 0.591709
[40]	valid_0's auc: 0.591709
[50]	valid_0's auc: 0.591709
Early stopping, best iteration is:
[9]	valid_0's auc: 0.591709
best score: 0.591708884203
best iteration: 9
complete on: source_type

working on: city


After selection:
target       uint8
city      category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505718
[20]	valid_0's auc: 0.505718
[30]	valid_0's auc: 0.505718
[40]	valid_0's auc: 0.505718
[50]	valid_0's auc: 0.505718
Early stopping, best iteration is:
[8]	valid_0's auc: 0.505718
best score: 0.505717833263
best iteration: 8
complete on: city

working on: registered_via


After selection:
target               uint8
registered_via    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.502389
[20]	valid_0's auc: 0.502389
[30]	valid_0's auc: 0.502389
[40]	valid_0's auc: 0.502389
[50]	valid_0's auc: 0.502389
Early stopping, best iteration is:
[1]	valid_0's auc: 0.502389
best score: 0.502388935196
best iteration: 1
complete on: registered_via

working on: sex


After selection:
target       uint8
sex       category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.504738
[20]	valid_0's auc: 0.504738
[30]	valid_0's auc: 0.504738
[40]	valid_0's auc: 0.504738
[50]	valid_0's auc: 0.504738
Early stopping, best iteration is:
[1]	valid_0's auc: 0.504738
best score: 0.504738069576
best iteration: 1
complete on: sex

working on: sex_guess1


After selection:
target           uint8
sex_guess1    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.500764
[20]	valid_0's auc: 0.500764
[30]	valid_0's auc: 0.500764
[40]	valid_0's auc: 0.500764
[50]	valid_0's auc: 0.500764
Early stopping, best iteration is:
[1]	valid_0's auc: 0.500764
best score: 0.500764121463
best iteration: 1
complete on: sex_guess1

working on: sex_guess2


After selection:
target           uint8
sex_guess2    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.500183
[20]	valid_0's auc: 0.500183
[30]	valid_0's auc: 0.500183
[40]	valid_0's auc: 0.500183
[50]	valid_0's auc: 0.500183
Early stopping, best iteration is:
[1]	valid_0's auc: 0.500183
best score: 0.500182849055
best iteration: 1
complete on: sex_guess2

working on: sex_guess3


After selection:
target           uint8
sex_guess3    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.499617
[20]	valid_0's auc: 0.499617
[30]	valid_0's auc: 0.499617
[40]	valid_0's auc: 0.499617
[50]	valid_0's auc: 0.499617
Early stopping, best iteration is:
[1]	valid_0's auc: 0.499617
best score: 0.499616693386
best iteration: 1
complete on: sex_guess3

working on: sex_guess4


After selection:
target           uint8
sex_guess4    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.501794
[20]	valid_0's auc: 0.501794
[30]	valid_0's auc: 0.501794
[40]	valid_0's auc: 0.501794
[50]	valid_0's auc: 0.501794
Early stopping, best iteration is:
[1]	valid_0's auc: 0.501794
best score: 0.501794350374
best iteration: 1
complete on: sex_guess4

working on: sex_guess5


After selection:
target           uint8
sex_guess5    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.498607
[20]	valid_0's auc: 0.498607
[30]	valid_0's auc: 0.498607
[40]	valid_0's auc: 0.498607
[50]	valid_0's auc: 0.498607
Early stopping, best iteration is:
[1]	valid_0's auc: 0.498607
best score: 0.498606717392
best iteration: 1
complete on: sex_guess5

working on: sex_freq_member


After selection:
target                uint8
sex_freq_member    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.502116
[20]	valid_0's auc: 0.502116
[30]	valid_0's auc: 0.502116
[40]	valid_0's auc: 0.502116
[50]	valid_0's auc: 0.502116
Early stopping, best iteration is:
[1]	valid_0's auc: 0.502116
best score: 0.502115527386
best iteration: 1
complete on: sex_freq_member

working on: registration_year


After selection:
target                  uint8
registration_year    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.50299
[20]	valid_0's auc: 0.50299
[30]	valid_0's auc: 0.50299
[40]	valid_0's auc: 0.50299
[50]	valid_0's auc: 0.50299
[60]	valid_0's auc: 0.50299
Early stopping, best iteration is:
[10]	valid_0's auc: 0.50299
best score: 0.502990207385
best iteration: 10
complete on: registration_year

working on: registration_month


After selection:
target                   uint8
registration_month    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505121
[20]	valid_0's auc: 0.5053
[30]	valid_0's auc: 0.505328
[40]	valid_0's auc: 0.505328
[50]	valid_0's auc: 0.505328
[60]	valid_0's auc: 0.505328
[70]	valid_0's auc: 0.505328
Early stopping, best iteration is:
[22]	valid_0's auc: 0.505328
best score: 0.505328020014
best iteration: 22
complete on: registration_month

working on: registration_date


After selection:
target                  uint8
registration_date    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.504817
[20]	valid_0's auc: 0.504898
[30]	valid_0's auc: 0.504898
[40]	valid_0's auc: 0.504948
[50]	valid_0's auc: 0.504948
[60]	valid_0's auc: 0.504948
[70]	valid_0's auc: 0.504948
[80]	valid_0's auc: 0.504948
[90]	valid_0's auc: 0.504948
Early stopping, best iteration is:
[40]	valid_0's auc: 0.504948
best score: 0.504947617225
best iteration: 40
complete on: registration_date

working on: expiration_year


After selection:
target                uint8
expiration_year    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.506713
[20]	valid_0's auc: 0.506713
[30]	valid_0's auc: 0.506713
[40]	valid_0's auc: 0.506713
[50]	valid_0's auc: 0.506713
[60]	valid_0's auc: 0.506713
[70]	valid_0's auc: 0.506713
[80]	valid_0's auc: 0.506713
[90]	valid_0's auc: 0.506713
Early stopping, best iteration is:
[40]	valid_0's auc: 0.506713
best score: 0.506713158376
best iteration: 40
complete on: expiration_year

working on: expiration_month


After selection:
target                 uint8
expiration_month    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.513423
[20]	valid_0's auc: 0.513447
[30]	valid_0's auc: 0.513447
[40]	valid_0's auc: 0.513447
[50]	valid_0's auc: 0.513447
Early stopping, best iteration is:
[1]	valid_0's auc: 0.513456
best score: 0.513455747223
best iteration: 1
complete on: expiration_month

working on: expiration_date


After selection:
target                uint8
expiration_date    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.505816
[20]	valid_0's auc: 0.505854
[30]	valid_0's auc: 0.505747
[40]	valid_0's auc: 0.50574
[50]	valid_0's auc: 0.50574
Early stopping, best iteration is:
[1]	valid_0's auc: 0.506128
best score: 0.506127944185
best iteration: 1
complete on: expiration_date

working on: genre_ids


After selection:
target          uint8
genre_ids    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.528322
[20]	valid_0's auc: 0.527725
[30]	valid_0's auc: 0.527727
[40]	valid_0's auc: 0.527727
[50]	valid_0's auc: 0.527727
Early stopping, best iteration is:
[3]	valid_0's auc: 0.528624
best score: 0.528624157069
best iteration: 3
complete on: genre_ids

working on: artist_name


After selection:
target            uint8
artist_name    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.529077
[20]	valid_0's auc: 0.53261
[30]	valid_0's auc: 0.53497
[40]	valid_0's auc: 0.536641
[50]	valid_0's auc: 0.538096
[60]	valid_0's auc: 0.548481
[70]	valid_0's auc: 0.549386
[80]	valid_0's auc: 0.549945
[90]	valid_0's auc: 0.550431
[100]	valid_0's auc: 0.551167
[110]	valid_0's auc: 0.550955
[120]	valid_0's auc: 0.551713
[130]	valid_0's auc: 0.55183
[140]	valid_0's auc: 0.552043
[150]	valid_0's auc: 0.55258
[160]	valid_0's auc: 0.552947
[170]	valid_0's auc: 0.552678
[180]	valid_0's auc: 0.553114
[190]	valid_0's auc: 0.55482
[200]	valid_0's auc: 0.555346
[210]	valid_0's auc: 0.556673
[220]	valid_0's auc: 0.556831
[230]	valid_0's auc: 0.557028
[240]	valid_0's auc: 0.557399
[250]	valid_0's auc: 0.557366
[260]	valid_0's auc: 0.557778
[270]	valid_0's auc: 0.557789
[280]	valid_0's auc: 0.557861
[290]	valid_0's auc: 0.557959
[300]	valid_0's auc: 0.557999
[310]	valid_0's auc: 0.55803
[320]	valid_0's auc: 0.558076
[330]	valid_0's auc: 0.558133
[340]	valid_0's auc: 0.558174
[350]	valid_0's auc: 0.558193
[360]	valid_0's auc: 0.55823
[370]	valid_0's auc: 0.558307
[380]	valid_0's auc: 0.558092
[390]	valid_0's auc: 0.558109
[400]	valid_0's auc: 0.558124
[410]	valid_0's auc: 0.559377
[420]	valid_0's auc: 0.559575
[430]	valid_0's auc: 0.559656
[440]	valid_0's auc: 0.559701
[450]	valid_0's auc: 0.559749
[460]	valid_0's auc: 0.560178
[470]	valid_0's auc: 0.5602
[480]	valid_0's auc: 0.560371
[490]	valid_0's auc: 0.560395
[500]	valid_0's auc: 0.56046
[510]	valid_0's auc: 0.560587
[520]	valid_0's auc: 0.560666
[530]	valid_0's auc: 0.560814
[540]	valid_0's auc: 0.560832
[550]	valid_0's auc: 0.561135
[560]	valid_0's auc: 0.561129
[570]	valid_0's auc: 0.561137
[580]	valid_0's auc: 0.561183
[590]	valid_0's auc: 0.56123
[600]	valid_0's auc: 0.561246
[610]	valid_0's auc: 0.561255
[620]	valid_0's auc: 0.56125
[630]	valid_0's auc: 0.561272
[640]	valid_0's auc: 0.561283
[650]	valid_0's auc: 0.56127
[660]	valid_0's auc: 0.561328
[670]	valid_0's auc: 0.56136
[680]	valid_0's auc: 0.561357
[690]	valid_0's auc: 0.56144
[700]	valid_0's auc: 0.561511
[710]	valid_0's auc: 0.561595
[720]	valid_0's auc: 0.561595
[730]	valid_0's auc: 0.561583
[740]	valid_0's auc: 0.561751
[750]	valid_0's auc: 0.561757
[760]	valid_0's auc: 0.561741
[770]	valid_0's auc: 0.561765
[780]	valid_0's auc: 0.561745
[790]	valid_0's auc: 0.561821
[800]	valid_0's auc: 0.561794
[810]	valid_0's auc: 0.561823
[820]	valid_0's auc: 0.561848
[830]	valid_0's auc: 0.561861
[840]	valid_0's auc: 0.561896
[850]	valid_0's auc: 0.561898
[860]	valid_0's auc: 0.561899
[870]	valid_0's auc: 0.561888
[880]	valid_0's auc: 0.56188
[890]	valid_0's auc: 0.5619
Early stopping, best iteration is:
[845]	valid_0's auc: 0.561907
best score: 0.561907132557
best iteration: 845
complete on: artist_name

working on: composer


After selection:
target         uint8
composer    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537629
[20]	valid_0's auc: 0.540667
[30]	valid_0's auc: 0.541717
[40]	valid_0's auc: 0.542444
[50]	valid_0's auc: 0.542382
[60]	valid_0's auc: 0.54275
[70]	valid_0's auc: 0.543362
[80]	valid_0's auc: 0.543884
[90]	valid_0's auc: 0.545836
[100]	valid_0's auc: 0.549924
[110]	valid_0's auc: 0.551753
[120]	valid_0's auc: 0.552346
[130]	valid_0's auc: 0.552826
[140]	valid_0's auc: 0.553309
[150]	valid_0's auc: 0.553591
[160]	valid_0's auc: 0.553915
[170]	valid_0's auc: 0.554143
[180]	valid_0's auc: 0.554325
[190]	valid_0's auc: 0.554454
[200]	valid_0's auc: 0.554556
[210]	valid_0's auc: 0.55464
[220]	valid_0's auc: 0.554706
[230]	valid_0's auc: 0.554834
[240]	valid_0's auc: 0.554916
[250]	valid_0's auc: 0.555363
[260]	valid_0's auc: 0.555544
[270]	valid_0's auc: 0.555857
[280]	valid_0's auc: 0.556067
[290]	valid_0's auc: 0.556126
[300]	valid_0's auc: 0.556271
[310]	valid_0's auc: 0.556347
[320]	valid_0's auc: 0.556513
[330]	valid_0's auc: 0.55672
[340]	valid_0's auc: 0.55674
[350]	valid_0's auc: 0.556808
[360]	valid_0's auc: 0.556876
[370]	valid_0's auc: 0.556899
[380]	valid_0's auc: 0.556913
[390]	valid_0's auc: 0.557058
[400]	valid_0's auc: 0.557132
[410]	valid_0's auc: 0.557233
[420]	valid_0's auc: 0.557272
[430]	valid_0's auc: 0.557279
[440]	valid_0's auc: 0.557304
[450]	valid_0's auc: 0.55744
[460]	valid_0's auc: 0.55743
[470]	valid_0's auc: 0.557425
[480]	valid_0's auc: 0.557447
[490]	valid_0's auc: 0.557464
[500]	valid_0's auc: 0.557473
[510]	valid_0's auc: 0.557484
[520]	valid_0's auc: 0.557508
[530]	valid_0's auc: 0.557496
[540]	valid_0's auc: 0.557499
[550]	valid_0's auc: 0.557514
[560]	valid_0's auc: 0.557546
[570]	valid_0's auc: 0.557515
[580]	valid_0's auc: 0.557518
[590]	valid_0's auc: 0.557534
[600]	valid_0's auc: 0.557525
Early stopping, best iteration is:
[556]	valid_0's auc: 0.557548
best score: 0.557548076976
best iteration: 556
complete on: composer

working on: lyricist


After selection:
target         uint8
lyricist    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.5319
[20]	valid_0's auc: 0.534829
[30]	valid_0's auc: 0.535909
[40]	valid_0's auc: 0.538449
[50]	valid_0's auc: 0.540342
[60]	valid_0's auc: 0.543387
[70]	valid_0's auc: 0.54387
[80]	valid_0's auc: 0.544373
[90]	valid_0's auc: 0.544897
[100]	valid_0's auc: 0.545224
[110]	valid_0's auc: 0.545811
[120]	valid_0's auc: 0.546291
[130]	valid_0's auc: 0.546612
[140]	valid_0's auc: 0.546887
[150]	valid_0's auc: 0.547008
[160]	valid_0's auc: 0.5471
[170]	valid_0's auc: 0.547147
[180]	valid_0's auc: 0.547267
[190]	valid_0's auc: 0.547393
[200]	valid_0's auc: 0.547469
[210]	valid_0's auc: 0.547682
[220]	valid_0's auc: 0.54781
[230]	valid_0's auc: 0.547768
[240]	valid_0's auc: 0.547849
[250]	valid_0's auc: 0.547968
[260]	valid_0's auc: 0.547974
[270]	valid_0's auc: 0.547969
[280]	valid_0's auc: 0.54803
[290]	valid_0's auc: 0.548027
[300]	valid_0's auc: 0.548027
[310]	valid_0's auc: 0.54801
[320]	valid_0's auc: 0.547999
[330]	valid_0's auc: 0.548034
[340]	valid_0's auc: 0.548075
[350]	valid_0's auc: 0.548105
[360]	valid_0's auc: 0.548066
[370]	valid_0's auc: 0.547931
[380]	valid_0's auc: 0.547847
[390]	valid_0's auc: 0.547812
[400]	valid_0's auc: 0.547794
Early stopping, best iteration is:
[357]	valid_0's auc: 0.548133
best score: 0.548133137289
best iteration: 357
complete on: lyricist

working on: language


After selection:
target         uint8
language    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.524767
[20]	valid_0's auc: 0.524768
[30]	valid_0's auc: 0.524768
[40]	valid_0's auc: 0.524768
[50]	valid_0's auc: 0.524768
[60]	valid_0's auc: 0.524768
Early stopping, best iteration is:
[12]	valid_0's auc: 0.524768
best score: 0.52476758398
best iteration: 12
complete on: language

working on: name


After selection:
target       uint8
name      category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.537324
[20]	valid_0's auc: 0.539136
[30]	valid_0's auc: 0.54049
[40]	valid_0's auc: 0.541597
[50]	valid_0's auc: 0.548528
[60]	valid_0's auc: 0.55239
[70]	valid_0's auc: 0.5544
[80]	valid_0's auc: 0.556427
[90]	valid_0's auc: 0.557044
[100]	valid_0's auc: 0.557484
[110]	valid_0's auc: 0.557849
[120]	valid_0's auc: 0.557993
[130]	valid_0's auc: 0.558368
[140]	valid_0's auc: 0.558577
[150]	valid_0's auc: 0.55874
[160]	valid_0's auc: 0.558917
[170]	valid_0's auc: 0.559145
[180]	valid_0's auc: 0.559502
[190]	valid_0's auc: 0.560409
[200]	valid_0's auc: 0.561146
[210]	valid_0's auc: 0.561794
[220]	valid_0's auc: 0.562532
[230]	valid_0's auc: 0.562959
[240]	valid_0's auc: 0.563472
[250]	valid_0's auc: 0.563905
[260]	valid_0's auc: 0.564182
[270]	valid_0's auc: 0.564562
[280]	valid_0's auc: 0.564809
[290]	valid_0's auc: 0.565021
[300]	valid_0's auc: 0.565239
[310]	valid_0's auc: 0.565483
[320]	valid_0's auc: 0.565559
[330]	valid_0's auc: 0.565604
[340]	valid_0's auc: 0.565692
[350]	valid_0's auc: 0.565754
[360]	valid_0's auc: 0.565813
[370]	valid_0's auc: 0.565823
[380]	valid_0's auc: 0.565897
[390]	valid_0's auc: 0.565998
[400]	valid_0's auc: 0.566293
[410]	valid_0's auc: 0.566446
[420]	valid_0's auc: 0.566561
[430]	valid_0's auc: 0.56667
[440]	valid_0's auc: 0.56678
[450]	valid_0's auc: 0.566965
[460]	valid_0's auc: 0.566978
[470]	valid_0's auc: 0.567034
[480]	valid_0's auc: 0.567221
[490]	valid_0's auc: 0.567243
[500]	valid_0's auc: 0.567293
[510]	valid_0's auc: 0.56744
[520]	valid_0's auc: 0.567509
[530]	valid_0's auc: 0.567622
[540]	valid_0's auc: 0.567683
[550]	valid_0's auc: 0.567746
[560]	valid_0's auc: 0.567794
[570]	valid_0's auc: 0.567854
[580]	valid_0's auc: 0.567853
[590]	valid_0's auc: 0.567896
[600]	valid_0's auc: 0.567925
[610]	valid_0's auc: 0.567963
[620]	valid_0's auc: 0.568245
[630]	valid_0's auc: 0.568264
[640]	valid_0's auc: 0.568358
[650]	valid_0's auc: 0.568387
[660]	valid_0's auc: 0.56841
[670]	valid_0's auc: 0.568426
[680]	valid_0's auc: 0.568451
[690]	valid_0's auc: 0.56849
[700]	valid_0's auc: 0.568491
[710]	valid_0's auc: 0.568511
[720]	valid_0's auc: 0.568563
[730]	valid_0's auc: 0.568584
[740]	valid_0's auc: 0.568581
[750]	valid_0's auc: 0.568598
[760]	valid_0's auc: 0.56867
[770]	valid_0's auc: 0.568692
[780]	valid_0's auc: 0.568691
[790]	valid_0's auc: 0.568714
[800]	valid_0's auc: 0.568779
[810]	valid_0's auc: 0.56876
[820]	valid_0's auc: 0.568715
[830]	valid_0's auc: 0.568747
[840]	valid_0's auc: 0.568763
[850]	valid_0's auc: 0.568759
Early stopping, best iteration is:
[801]	valid_0's auc: 0.56878
best score: 0.568779530968
best iteration: 801
complete on: name

working on: genre_ids_fre_song


After selection:
target                   uint8
genre_ids_fre_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.527951
[20]	valid_0's auc: 0.527298
[30]	valid_0's auc: 0.527308
[40]	valid_0's auc: 0.527308
[50]	valid_0's auc: 0.527308
Early stopping, best iteration is:
[3]	valid_0's auc: 0.528292
best score: 0.528291867212
best iteration: 3
complete on: genre_ids_fre_song

working on: song_year_fre_song


After selection:
target                   uint8
song_year_fre_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.535566
[20]	valid_0's auc: 0.535565
[30]	valid_0's auc: 0.535572
[40]	valid_0's auc: 0.535572
[50]	valid_0's auc: 0.535572
[60]	valid_0's auc: 0.535582
[70]	valid_0's auc: 0.535582
[80]	valid_0's auc: 0.535582
[90]	valid_0's auc: 0.535582
[100]	valid_0's auc: 0.535582
Early stopping, best iteration is:
[51]	valid_0's auc: 0.535582
best score: 0.535581858375
best iteration: 51
complete on: song_year_fre_song

working on: song_year


After selection:
target          uint8
song_year    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.538068
[20]	valid_0's auc: 0.538043
[30]	valid_0's auc: 0.538043
[40]	valid_0's auc: 0.538043
[50]	valid_0's auc: 0.538043
Early stopping, best iteration is:
[1]	valid_0's auc: 0.538073
best score: 0.538072665247
best iteration: 1
complete on: song_year

working on: song_country_fre_song


After selection:
target                      uint8
song_country_fre_song    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525668
[20]	valid_0's auc: 0.525669
[30]	valid_0's auc: 0.525669
[40]	valid_0's auc: 0.525666
[50]	valid_0's auc: 0.525666
[60]	valid_0's auc: 0.525672
[70]	valid_0's auc: 0.525672
[80]	valid_0's auc: 0.525672
[90]	valid_0's auc: 0.525672
[100]	valid_0's auc: 0.525672
Early stopping, best iteration is:
[59]	valid_0's auc: 0.525672
best score: 0.525672003334
best iteration: 59
complete on: song_country_fre_song

working on: song_country


After selection:
target             uint8
song_country    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.525884
[20]	valid_0's auc: 0.525893
[30]	valid_0's auc: 0.525893
[40]	valid_0's auc: 0.525893
[50]	valid_0's auc: 0.525893
Early stopping, best iteration is:
[5]	valid_0's auc: 0.525901
best score: 0.525901232455
best iteration: 5
complete on: song_country

working on: rc


After selection:
target       uint8
rc        category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.54317
[20]	valid_0's auc: 0.544177
[30]	valid_0's auc: 0.545472
[40]	valid_0's auc: 0.545879
[50]	valid_0's auc: 0.54659
[60]	valid_0's auc: 0.546844
[70]	valid_0's auc: 0.5467
[80]	valid_0's auc: 0.547126
[90]	valid_0's auc: 0.547179
[100]	valid_0's auc: 0.547257
[110]	valid_0's auc: 0.547298
[120]	valid_0's auc: 0.547381
Early stopping, best iteration is:
[74]	valid_0's auc: 0.547549
best score: 0.547548558221
best iteration: 74
complete on: rc

working on: source_system_tab_guess


After selection:
target                        uint8
source_system_tab_guess    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.584839
[20]	valid_0's auc: 0.584865
[30]	valid_0's auc: 0.584865
[40]	valid_0's auc: 0.584853
[50]	valid_0's auc: 0.584853
[60]	valid_0's auc: 0.584853
Early stopping, best iteration is:
[15]	valid_0's auc: 0.584865
best score: 0.584865451025
best iteration: 15
complete on: source_system_tab_guess

working on: source_screen_name_guess


After selection:
target                         uint8
source_screen_name_guess    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.587045
[20]	valid_0's auc: 0.587046
[30]	valid_0's auc: 0.587046
[40]	valid_0's auc: 0.587171
[50]	valid_0's auc: 0.587171
[60]	valid_0's auc: 0.587171
[70]	valid_0's auc: 0.587171
[80]	valid_0's auc: 0.587171
Early stopping, best iteration is:
[36]	valid_0's auc: 0.587171
best score: 0.587170532259
best iteration: 36
complete on: source_screen_name_guess

working on: source_type_guess


After selection:
target                  uint8
source_type_guess    category
dtype: object
number of columns: 2


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.591584
[20]	valid_0's auc: 0.591584
[30]	valid_0's auc: 0.591584
[40]	valid_0's auc: 0.591584
[50]	valid_0's auc: 0.591584
Early stopping, best iteration is:
[9]	valid_0's auc: 0.591584
best score: 0.591584080701
best iteration: 9
complete on: source_type_guess


[timer]: complete in 60m 10s

Process finished with exit code 0
'''