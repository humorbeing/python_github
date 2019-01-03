import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_play.csv'
df = read_df(load_name)
on = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'target',
    # 'genre_ids',
    'artist_name',
    # 'composer',
    # 'lyricist',
    # 'language',
    'song_year',
    # 'song_country',
    # 'rc',
    # 'top1_in_song',
    # 'top2_in_song',
    # 'top3_in_song',
    # 'membership_days',
    # 'song_year_int',
    # 'ISC_top1_in_song',
    'ISC_top2_in_song',
    # 'ISC_top3_in_song',
    # 'ISC_language',
    # 'ISCZ_rc',
    # 'ISCZ_isrc_rest',
    # 'ISC_song_year',
    # 'song_length_log10',
    # 'ISCZ_genre_ids_log10',
    # 'ISC_artist_name_log10',
    # 'ISCZ_composer_log10',
    # 'ISC_lyricist_log10',
    # 'ISC_song_country_ln',
    'ITC_song_id_log10_1',
    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_song_year_log10_1',
    # 'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    # 'ITC_top3_in_song_log10_1',
    'ITC_msno_log10_1',
    # 'OinC_msno',
    # 'ITC_language_log10_1',
    # 'OinC_language',
]
df = df[on]
show_df(df)

# save_me = True
save_me = False
if save_me:
    save_df(df)

dfs, val = fake_df(df)
del df
K = 2
dfs = divide_df(dfs, K)
dcs = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dcs.append(dc)

vc = pd.DataFrame()
vc['target'] = val['target']
v = np.zeros(shape=[len(val)])
save_name = ''

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
r = 'dart'
save_name += r+'_'

on = [

]
params = {
    'boosting': 'dart',

    'learning_rate': 0.5,
    'num_leaves': 15,
    'max_depth': 5,

    'lambda_l1': 0,
    'lambda_l2': 0,
    'max_bin': 15,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}

num_boost_round = 5
early_stopping_rounds = 50
verbose_eval = 1

for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dfs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = val_df(
        params, dt, val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    del dt
    dcs[i][r] = model.predict(dfs[i])
    v += model.predict(val)

vc[r] = v / K
v = np.zeros(shape=[len(val)])


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
r = 'goss'
save_name += r + '_'

on = [

]
params = {
    'boosting': 'goss',

    'learning_rate': 0.3,
    'num_leaves': 15,
    'max_depth': 6,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 15,


    'bagging_fraction': 1,
    'bagging_freq': 0,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}

num_boost_round = 5
early_stopping_rounds = 50
verbose_eval = 1

for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dfs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = val_df(
        params, dt, val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    del dt
    dcs[i][r] = model.predict(dfs[i])
    v += model.predict(val)

vc[r] = v / K
v = np.zeros(shape=[len(val)])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
r = 'rf'
save_name += r + '_'

on = [

]
params = {
    'boosting': 'rf',

    'learning_rate': 0.3,
    'num_leaves': 511,
    'max_depth': 10,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 63,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}

num_boost_round = 5
early_stopping_rounds = 50
verbose_eval = 1

for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dfs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = val_df(
        params, dt, val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    del dt
    dcs[i][r] = model.predict(dfs[i])
    v += model.predict(val)

vc[r] = v / K
v = np.zeros(shape=[len(val)])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
r = 'gbdt'
save_name += r + '_'

on = [

]
params = {
    'boosting': 'gbdt',

    'learning_rate': 0.032,
    'num_leaves': 750,
    'max_depth': 50,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 172,


    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 2,
}

num_boost_round = 5
early_stopping_rounds = 50
verbose_eval = 1

for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dfs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = val_df(
        params, dt, val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    del dt
    dcs[i][r] = model.predict(dfs[i])
    v += model.predict(val)

vc[r] = v / K
v = np.zeros(shape=[len(val)])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# new_t = pd.concat(dcs)
# save_df(new_t, 'TRAIN_'+save_name, '../fake/saves/feature/')
# save_df(vc, 'TEST_'+save_name, '../fake/saves/feature/')





v = np.zeros(shape=[len(val)])

for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dcs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = cat(
        dt, vc, 5, learning_rate=0.3,
        depth=6
    )
    del dt
    # dcs[i][r] = model.predict(dfs[i])
    p = model.predict_proba(vc.drop('target',axis=1))
    tt = np.array(p).T[1]
    v += tt

v = v / K

from sklearn.metrics import roc_auc_score
print(roc_auc_score(val['target'], v))


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('done')
'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/MODIFY_K-fold_V1001/dart_goss_rf_gbdt_V1001.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
  return f(*args, **kwds)

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
target                    uint8
artist_name            category
song_year              category
ISC_top2_in_song          int64
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
dtype: object
number of rows: 7377418
number of columns: 11

'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'artist_name',
'song_year',
'ISC_top2_in_song',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: dart  k-fold: 0

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
[1]	training's auc: 0.691383	valid_1's auc: 0.623376
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.704385	valid_1's auc: 0.629433
[3]	training's auc: 0.711163	valid_1's auc: 0.633169
[4]	training's auc: 0.718664	valid_1's auc: 0.639404
[5]	training's auc: 0.721069	valid_1's auc: 0.641856
[6]	training's auc: 0.724831	valid_1's auc: 0.642802
[7]	training's auc: 0.727367	valid_1's auc: 0.644212
[8]	training's auc: 0.727995	valid_1's auc: 0.644656
[9]	training's auc: 0.731451	valid_1's auc: 0.645257
[10]	training's auc: 0.73424	valid_1's auc: 0.647316
[11]	training's auc: 0.736488	valid_1's auc: 0.648157
[12]	training's auc: 0.73641	valid_1's auc: 0.647877
[13]	training's auc: 0.737248	valid_1's auc: 0.649281
[14]	training's auc: 0.738998	valid_1's auc: 0.649976
[15]	training's auc: 0.740277	valid_1's auc: 0.650544
[16]	training's auc: 0.741643	valid_1's auc: 0.651165
[17]	training's auc: 0.742657	valid_1's auc: 0.652864
[18]	training's auc: 0.744167	valid_1's auc: 0.653604
[19]	training's auc: 0.745378	valid_1's auc: 0.654026
[20]	training's auc: 0.746576	valid_1's auc: 0.654429
[21]	training's auc: 0.746847	valid_1's auc: 0.654362
[22]	training's auc: 0.747928	valid_1's auc: 0.654898
[23]	training's auc: 0.748984	valid_1's auc: 0.65535
[24]	training's auc: 0.749864	valid_1's auc: 0.655571
[25]	training's auc: 0.750915	valid_1's auc: 0.656102
[26]	training's auc: 0.751974	valid_1's auc: 0.656539
[27]	training's auc: 0.752971	valid_1's auc: 0.657089
[28]	training's auc: 0.752863	valid_1's auc: 0.657097
[29]	training's auc: 0.753354	valid_1's auc: 0.657537
[30]	training's auc: 0.754067	valid_1's auc: 0.657705
[31]	training's auc: 0.754082	valid_1's auc: 0.657731
[32]	training's auc: 0.754325	valid_1's auc: 0.65792
[33]	training's auc: 0.755314	valid_1's auc: 0.658324
[34]	training's auc: 0.755521	valid_1's auc: 0.658465
[35]	training's auc: 0.755273	valid_1's auc: 0.65825
[36]	training's auc: 0.754843	valid_1's auc: 0.65793
[37]	training's auc: 0.755531	valid_1's auc: 0.658159
[38]	training's auc: 0.756198	valid_1's auc: 0.658374
[39]	training's auc: 0.75691	valid_1's auc: 0.658547
[40]	training's auc: 0.757431	valid_1's auc: 0.658807
[41]	training's auc: 0.757184	valid_1's auc: 0.658719
[42]	training's auc: 0.757811	valid_1's auc: 0.658925
[43]	training's auc: 0.757568	valid_1's auc: 0.658816
[44]	training's auc: 0.758156	valid_1's auc: 0.658864
[45]	training's auc: 0.758782	valid_1's auc: 0.659106
[46]	training's auc: 0.75939	valid_1's auc: 0.659392
[47]	training's auc: 0.760806	valid_1's auc: 0.659727
[48]	training's auc: 0.760931	valid_1's auc: 0.659778
[49]	training's auc: 0.761108	valid_1's auc: 0.659792
[50]	training's auc: 0.760795	valid_1's auc: 0.65955

in model: dart  k-fold: 1

[1]	training's auc: 0.691369	valid_1's auc: 0.623472
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.704361	valid_1's auc: 0.629373
[3]	training's auc: 0.710705	valid_1's auc: 0.633387
[4]	training's auc: 0.71826	valid_1's auc: 0.639255
[5]	training's auc: 0.720743	valid_1's auc: 0.641995
[6]	training's auc: 0.724541	valid_1's auc: 0.643018
[7]	training's auc: 0.726852	valid_1's auc: 0.643897
[8]	training's auc: 0.727533	valid_1's auc: 0.644268
[9]	training's auc: 0.730902	valid_1's auc: 0.645581
[10]	training's auc: 0.733838	valid_1's auc: 0.64709
[11]	training's auc: 0.736498	valid_1's auc: 0.648431
[12]	training's auc: 0.736677	valid_1's auc: 0.648326
[13]	training's auc: 0.73781	valid_1's auc: 0.649715
[14]	training's auc: 0.739508	valid_1's auc: 0.650423
[15]	training's auc: 0.740817	valid_1's auc: 0.650848
[16]	training's auc: 0.742131	valid_1's auc: 0.651752
[17]	training's auc: 0.743049	valid_1's auc: 0.652874
[18]	training's auc: 0.744369	valid_1's auc: 0.653235
[19]	training's auc: 0.745547	valid_1's auc: 0.653852
[20]	training's auc: 0.746715	valid_1's auc: 0.654387
[21]	training's auc: 0.746956	valid_1's auc: 0.654231
[22]	training's auc: 0.747944	valid_1's auc: 0.654422
[23]	training's auc: 0.749128	valid_1's auc: 0.654972
[24]	training's auc: 0.75008	valid_1's auc: 0.655373
[25]	training's auc: 0.751062	valid_1's auc: 0.655826
[26]	training's auc: 0.751872	valid_1's auc: 0.65608
[27]	training's auc: 0.752815	valid_1's auc: 0.656535
[28]	training's auc: 0.752719	valid_1's auc: 0.656683
[29]	training's auc: 0.753079	valid_1's auc: 0.657071
[30]	training's auc: 0.753954	valid_1's auc: 0.657473
[31]	training's auc: 0.753979	valid_1's auc: 0.657486
[32]	training's auc: 0.75437	valid_1's auc: 0.658031
[33]	training's auc: 0.755297	valid_1's auc: 0.658404
[34]	training's auc: 0.755593	valid_1's auc: 0.658789
[35]	training's auc: 0.755343	valid_1's auc: 0.658627
[36]	training's auc: 0.754992	valid_1's auc: 0.658371
[37]	training's auc: 0.755746	valid_1's auc: 0.658511
[38]	training's auc: 0.756539	valid_1's auc: 0.659003
[39]	training's auc: 0.757186	valid_1's auc: 0.659196
[40]	training's auc: 0.757611	valid_1's auc: 0.659475
[41]	training's auc: 0.757387	valid_1's auc: 0.659254
[42]	training's auc: 0.758079	valid_1's auc: 0.659649
[43]	training's auc: 0.757854	valid_1's auc: 0.659593
[44]	training's auc: 0.758516	valid_1's auc: 0.659711
[45]	training's auc: 0.759118	valid_1's auc: 0.659785
[46]	training's auc: 0.759768	valid_1's auc: 0.660147
[47]	training's auc: 0.761149	valid_1's auc: 0.660252
[48]	training's auc: 0.761299	valid_1's auc: 0.660302
[49]	training's auc: 0.761153	valid_1's auc: 0.660109
[50]	training's auc: 0.760853	valid_1's auc: 0.659905

in model: dart  k-fold: 2

[1]	training's auc: 0.691719	valid_1's auc: 0.622482
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.704729	valid_1's auc: 0.629724
[3]	training's auc: 0.711331	valid_1's auc: 0.633409
[4]	training's auc: 0.718785	valid_1's auc: 0.63925
[5]	training's auc: 0.720861	valid_1's auc: 0.641514
[6]	training's auc: 0.724573	valid_1's auc: 0.642676
[7]	training's auc: 0.726835	valid_1's auc: 0.643686
[8]	training's auc: 0.727418	valid_1's auc: 0.64408
[9]	training's auc: 0.729404	valid_1's auc: 0.644511
[10]	training's auc: 0.732213	valid_1's auc: 0.645724
[11]	training's auc: 0.734174	valid_1's auc: 0.647158
[12]	training's auc: 0.735075	valid_1's auc: 0.647385
[13]	training's auc: 0.73621	valid_1's auc: 0.648855
[14]	training's auc: 0.737922	valid_1's auc: 0.649611
[15]	training's auc: 0.739446	valid_1's auc: 0.650154
[16]	training's auc: 0.740953	valid_1's auc: 0.65101
[17]	training's auc: 0.742095	valid_1's auc: 0.652299
[18]	training's auc: 0.743444	valid_1's auc: 0.652842
[19]	training's auc: 0.744644	valid_1's auc: 0.653217
[20]	training's auc: 0.745733	valid_1's auc: 0.653693
[21]	training's auc: 0.74659	valid_1's auc: 0.653816
[22]	training's auc: 0.747778	valid_1's auc: 0.654268
[23]	training's auc: 0.748912	valid_1's auc: 0.654844
[24]	training's auc: 0.749973	valid_1's auc: 0.655152
[25]	training's auc: 0.75092	valid_1's auc: 0.655484
[26]	training's auc: 0.751864	valid_1's auc: 0.656055
[27]	training's auc: 0.752771	valid_1's auc: 0.656244
[28]	training's auc: 0.752616	valid_1's auc: 0.656301
[29]	training's auc: 0.752857	valid_1's auc: 0.656338
[30]	training's auc: 0.75371	valid_1's auc: 0.656825
[31]	training's auc: 0.753738	valid_1's auc: 0.656836
[32]	training's auc: 0.754134	valid_1's auc: 0.657205
[33]	training's auc: 0.755033	valid_1's auc: 0.65762
[34]	training's auc: 0.755376	valid_1's auc: 0.658272
[35]	training's auc: 0.755132	valid_1's auc: 0.658127
[36]	training's auc: 0.754771	valid_1's auc: 0.657887
[37]	training's auc: 0.755546	valid_1's auc: 0.658098
[38]	training's auc: 0.756406	valid_1's auc: 0.658649
[39]	training's auc: 0.757087	valid_1's auc: 0.658774
[40]	training's auc: 0.757483	valid_1's auc: 0.658995
[41]	training's auc: 0.757171	valid_1's auc: 0.658792
[42]	training's auc: 0.757773	valid_1's auc: 0.65892
[43]	training's auc: 0.75752	valid_1's auc: 0.658793
[44]	training's auc: 0.758077	valid_1's auc: 0.65908
[45]	training's auc: 0.758695	valid_1's auc: 0.659217
[46]	training's auc: 0.759308	valid_1's auc: 0.659671
[47]	training's auc: 0.760691	valid_1's auc: 0.659734
[48]	training's auc: 0.760854	valid_1's auc: 0.659833
[49]	training's auc: 0.760969	valid_1's auc: 0.659886
[50]	training's auc: 0.760827	valid_1's auc: 0.659798

in model: dart  k-fold: 3

[1]	training's auc: 0.692031	valid_1's auc: 0.621045
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.705406	valid_1's auc: 0.630643
[3]	training's auc: 0.711155	valid_1's auc: 0.632802
[4]	training's auc: 0.719087	valid_1's auc: 0.639739
[5]	training's auc: 0.72128	valid_1's auc: 0.642325
[6]	training's auc: 0.725055	valid_1's auc: 0.643237
[7]	training's auc: 0.727288	valid_1's auc: 0.64418
[8]	training's auc: 0.727618	valid_1's auc: 0.644342
[9]	training's auc: 0.729795	valid_1's auc: 0.645298
[10]	training's auc: 0.732153	valid_1's auc: 0.646682
[11]	training's auc: 0.733999	valid_1's auc: 0.647365
[12]	training's auc: 0.734557	valid_1's auc: 0.647331
[13]	training's auc: 0.735555	valid_1's auc: 0.648757
[14]	training's auc: 0.737265	valid_1's auc: 0.649487
[15]	training's auc: 0.738527	valid_1's auc: 0.649927
[16]	training's auc: 0.739887	valid_1's auc: 0.650498
[17]	training's auc: 0.74085	valid_1's auc: 0.652102
[18]	training's auc: 0.742287	valid_1's auc: 0.652643
[19]	training's auc: 0.7435	valid_1's auc: 0.653164
[20]	training's auc: 0.744633	valid_1's auc: 0.65363
[21]	training's auc: 0.744976	valid_1's auc: 0.653634
[22]	training's auc: 0.746201	valid_1's auc: 0.654165
[23]	training's auc: 0.747382	valid_1's auc: 0.654555
[24]	training's auc: 0.748333	valid_1's auc: 0.654821
[25]	training's auc: 0.749291	valid_1's auc: 0.655254
[26]	training's auc: 0.750162	valid_1's auc: 0.655667
[27]	training's auc: 0.751054	valid_1's auc: 0.655909
[28]	training's auc: 0.75094	valid_1's auc: 0.655876
[29]	training's auc: 0.751744	valid_1's auc: 0.6566
[30]	training's auc: 0.752712	valid_1's auc: 0.65701
[31]	training's auc: 0.752717	valid_1's auc: 0.657059
[32]	training's auc: 0.753078	valid_1's auc: 0.65725
[33]	training's auc: 0.754018	valid_1's auc: 0.657585
[34]	training's auc: 0.754392	valid_1's auc: 0.657938
[35]	training's auc: 0.754149	valid_1's auc: 0.657794
[36]	training's auc: 0.753719	valid_1's auc: 0.657495
[37]	training's auc: 0.754477	valid_1's auc: 0.657838
[38]	training's auc: 0.755106	valid_1's auc: 0.658009
[39]	training's auc: 0.755821	valid_1's auc: 0.658352
[40]	training's auc: 0.756403	valid_1's auc: 0.658667
[41]	training's auc: 0.756151	valid_1's auc: 0.658553
[42]	training's auc: 0.756864	valid_1's auc: 0.658911
[43]	training's auc: 0.756629	valid_1's auc: 0.658822
[44]	training's auc: 0.757285	valid_1's auc: 0.659073
[45]	training's auc: 0.757919	valid_1's auc: 0.659461
[46]	training's auc: 0.758516	valid_1's auc: 0.659894
[47]	training's auc: 0.759971	valid_1's auc: 0.660092
[48]	training's auc: 0.760121	valid_1's auc: 0.660168
[49]	training's auc: 0.760336	valid_1's auc: 0.660051
[50]	training's auc: 0.760271	valid_1's auc: 0.659938

in model: dart  k-fold: 4

[1]	training's auc: 0.692363	valid_1's auc: 0.620787
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.704997	valid_1's auc: 0.631328
[3]	training's auc: 0.711563	valid_1's auc: 0.632727
[4]	training's auc: 0.719255	valid_1's auc: 0.639356
[5]	training's auc: 0.721639	valid_1's auc: 0.641604
[6]	training's auc: 0.725584	valid_1's auc: 0.643077
[7]	training's auc: 0.727933	valid_1's auc: 0.643966
[8]	training's auc: 0.728303	valid_1's auc: 0.643932
[9]	training's auc: 0.731944	valid_1's auc: 0.645405
[10]	training's auc: 0.735078	valid_1's auc: 0.647641
[11]	training's auc: 0.737015	valid_1's auc: 0.64879
[12]	training's auc: 0.73738	valid_1's auc: 0.648631
[13]	training's auc: 0.738322	valid_1's auc: 0.650078
[14]	training's auc: 0.739849	valid_1's auc: 0.650719
[15]	training's auc: 0.741075	valid_1's auc: 0.651274
[16]	training's auc: 0.74265	valid_1's auc: 0.651992
[17]	training's auc: 0.74366	valid_1's auc: 0.653272
[18]	training's auc: 0.744983	valid_1's auc: 0.653823
[19]	training's auc: 0.746172	valid_1's auc: 0.6543
[20]	training's auc: 0.747084	valid_1's auc: 0.654642
[21]	training's auc: 0.747264	valid_1's auc: 0.654567
[22]	training's auc: 0.748293	valid_1's auc: 0.654947
[23]	training's auc: 0.749274	valid_1's auc: 0.655425
[24]	training's auc: 0.75022	valid_1's auc: 0.655845
[25]	training's auc: 0.751272	valid_1's auc: 0.656268
[26]	training's auc: 0.752223	valid_1's auc: 0.656496
[27]	training's auc: 0.753029	valid_1's auc: 0.656827
[28]	training's auc: 0.752838	valid_1's auc: 0.656726
[29]	training's auc: 0.753186	valid_1's auc: 0.656627
[30]	training's auc: 0.753951	valid_1's auc: 0.656869
[31]	training's auc: 0.753969	valid_1's auc: 0.656855
[32]	training's auc: 0.754309	valid_1's auc: 0.657443
[33]	training's auc: 0.755323	valid_1's auc: 0.657916
[34]	training's auc: 0.755484	valid_1's auc: 0.65818
[35]	training's auc: 0.755239	valid_1's auc: 0.657985
[36]	training's auc: 0.754841	valid_1's auc: 0.657673
[37]	training's auc: 0.755573	valid_1's auc: 0.658115
[38]	training's auc: 0.756375	valid_1's auc: 0.6584
[39]	training's auc: 0.757077	valid_1's auc: 0.65894
[40]	training's auc: 0.757904	valid_1's auc: 0.659357
[41]	training's auc: 0.757599	valid_1's auc: 0.65916
[42]	training's auc: 0.758245	valid_1's auc: 0.659389
[43]	training's auc: 0.757993	valid_1's auc: 0.659274
[44]	training's auc: 0.758694	valid_1's auc: 0.659357
[45]	training's auc: 0.759367	valid_1's auc: 0.659597
[46]	training's auc: 0.7599	valid_1's auc: 0.659999
[47]	training's auc: 0.761031	valid_1's auc: 0.659849
[48]	training's auc: 0.76117	valid_1's auc: 0.659948
[49]	training's auc: 0.761358	valid_1's auc: 0.660042
[50]	training's auc: 0.761225	valid_1's auc: 0.659957

in model: goss  k-fold: 0

[1]	training's auc: 0.691392	valid_1's auc: 0.622862
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.700037	valid_1's auc: 0.629255
[3]	training's auc: 0.705484	valid_1's auc: 0.630247
[4]	training's auc: 0.71127	valid_1's auc: 0.63467
[5]	training's auc: 0.712955	valid_1's auc: 0.636556
[6]	training's auc: 0.716305	valid_1's auc: 0.637378
[7]	training's auc: 0.720568	valid_1's auc: 0.639949
[8]	training's auc: 0.72386	valid_1's auc: 0.642011
[9]	training's auc: 0.726282	valid_1's auc: 0.642739
[10]	training's auc: 0.728519	valid_1's auc: 0.64367
[11]	training's auc: 0.730984	valid_1's auc: 0.644853
[12]	training's auc: 0.732553	valid_1's auc: 0.645593
[13]	training's auc: 0.733031	valid_1's auc: 0.646085
[14]	training's auc: 0.734491	valid_1's auc: 0.646665
[15]	training's auc: 0.735817	valid_1's auc: 0.647031
[16]	training's auc: 0.737025	valid_1's auc: 0.6477
[17]	training's auc: 0.737602	valid_1's auc: 0.648189
[18]	training's auc: 0.738831	valid_1's auc: 0.648627
[19]	training's auc: 0.739986	valid_1's auc: 0.649165
[20]	training's auc: 0.741084	valid_1's auc: 0.649615
[21]	training's auc: 0.742172	valid_1's auc: 0.650081
[22]	training's auc: 0.743267	valid_1's auc: 0.650758
[23]	training's auc: 0.744257	valid_1's auc: 0.651192
[24]	training's auc: 0.745241	valid_1's auc: 0.651599
[25]	training's auc: 0.746225	valid_1's auc: 0.652115
[26]	training's auc: 0.747097	valid_1's auc: 0.652741
[27]	training's auc: 0.747929	valid_1's auc: 0.653034
[28]	training's auc: 0.748244	valid_1's auc: 0.653466
[29]	training's auc: 0.748601	valid_1's auc: 0.654001
[30]	training's auc: 0.749387	valid_1's auc: 0.654202
[31]	training's auc: 0.75022	valid_1's auc: 0.654639
[32]	training's auc: 0.750322	valid_1's auc: 0.654692
[33]	training's auc: 0.751188	valid_1's auc: 0.655102
[34]	training's auc: 0.75135	valid_1's auc: 0.655712
[35]	training's auc: 0.751441	valid_1's auc: 0.655738
[36]	training's auc: 0.751538	valid_1's auc: 0.655686
[37]	training's auc: 0.752309	valid_1's auc: 0.6559
[38]	training's auc: 0.753086	valid_1's auc: 0.656244
[39]	training's auc: 0.753797	valid_1's auc: 0.656567
[40]	training's auc: 0.754397	valid_1's auc: 0.656791
[41]	training's auc: 0.754436	valid_1's auc: 0.656732
[42]	training's auc: 0.755054	valid_1's auc: 0.65695
[43]	training's auc: 0.75566	valid_1's auc: 0.657235
[44]	training's auc: 0.756244	valid_1's auc: 0.657279
[45]	training's auc: 0.75685	valid_1's auc: 0.65753
[46]	training's auc: 0.757441	valid_1's auc: 0.657793
[47]	training's auc: 0.757944	valid_1's auc: 0.658031
[48]	training's auc: 0.758417	valid_1's auc: 0.658178
[49]	training's auc: 0.758893	valid_1's auc: 0.658481
[50]	training's auc: 0.759375	valid_1's auc: 0.658642

in model: goss  k-fold: 1

[1]	training's auc: 0.691964	valid_1's auc: 0.621156
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.699953	valid_1's auc: 0.629605
[3]	training's auc: 0.70723	valid_1's auc: 0.632347
[4]	training's auc: 0.711538	valid_1's auc: 0.63595
[5]	training's auc: 0.712844	valid_1's auc: 0.63731
[6]	training's auc: 0.716443	valid_1's auc: 0.638658
[7]	training's auc: 0.719793	valid_1's auc: 0.640244
[8]	training's auc: 0.722817	valid_1's auc: 0.641655
[9]	training's auc: 0.725767	valid_1's auc: 0.642985
[10]	training's auc: 0.728387	valid_1's auc: 0.644417
[11]	training's auc: 0.730532	valid_1's auc: 0.645642
[12]	training's auc: 0.732052	valid_1's auc: 0.646388
[13]	training's auc: 0.732854	valid_1's auc: 0.647613
[14]	training's auc: 0.734398	valid_1's auc: 0.648285
[15]	training's auc: 0.735901	valid_1's auc: 0.64886
[16]	training's auc: 0.737209	valid_1's auc: 0.64939
[17]	training's auc: 0.737651	valid_1's auc: 0.650153
[18]	training's auc: 0.73886	valid_1's auc: 0.650585
[19]	training's auc: 0.740114	valid_1's auc: 0.65106
[20]	training's auc: 0.741223	valid_1's auc: 0.651519
[21]	training's auc: 0.74235	valid_1's auc: 0.651978
[22]	training's auc: 0.743344	valid_1's auc: 0.65252
[23]	training's auc: 0.744335	valid_1's auc: 0.652881
[24]	training's auc: 0.745337	valid_1's auc: 0.653265
[25]	training's auc: 0.746177	valid_1's auc: 0.653525
[26]	training's auc: 0.747209	valid_1's auc: 0.653775
[27]	training's auc: 0.748013	valid_1's auc: 0.654145
[28]	training's auc: 0.748168	valid_1's auc: 0.654197
[29]	training's auc: 0.748766	valid_1's auc: 0.654691
[30]	training's auc: 0.74956	valid_1's auc: 0.655118
[31]	training's auc: 0.750274	valid_1's auc: 0.655469
[32]	training's auc: 0.750398	valid_1's auc: 0.655639
[33]	training's auc: 0.751142	valid_1's auc: 0.655954
[34]	training's auc: 0.751274	valid_1's auc: 0.656193
[35]	training's auc: 0.751452	valid_1's auc: 0.656302
[36]	training's auc: 0.751477	valid_1's auc: 0.656294
[37]	training's auc: 0.752212	valid_1's auc: 0.65659
[38]	training's auc: 0.752911	valid_1's auc: 0.656826
[39]	training's auc: 0.753605	valid_1's auc: 0.657082
[40]	training's auc: 0.754251	valid_1's auc: 0.657409
[41]	training's auc: 0.754382	valid_1's auc: 0.657371
[42]	training's auc: 0.755019	valid_1's auc: 0.657582
[43]	training's auc: 0.755592	valid_1's auc: 0.657824
[44]	training's auc: 0.756198	valid_1's auc: 0.658058
[45]	training's auc: 0.756776	valid_1's auc: 0.658334
[46]	training's auc: 0.757393	valid_1's auc: 0.658569
[47]	training's auc: 0.75784	valid_1's auc: 0.658525
[48]	training's auc: 0.758412	valid_1's auc: 0.658843
[49]	training's auc: 0.758919	valid_1's auc: 0.659003
[50]	training's auc: 0.759469	valid_1's auc: 0.659152

in model: goss  k-fold: 2

[1]	training's auc: 0.6922	valid_1's auc: 0.621222
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.699879	valid_1's auc: 0.629617
[3]	training's auc: 0.706586	valid_1's auc: 0.631368
[4]	training's auc: 0.711016	valid_1's auc: 0.634067
[5]	training's auc: 0.712742	valid_1's auc: 0.636411
[6]	training's auc: 0.716935	valid_1's auc: 0.63832
[7]	training's auc: 0.72116	valid_1's auc: 0.641008
[8]	training's auc: 0.724572	valid_1's auc: 0.642581
[9]	training's auc: 0.727176	valid_1's auc: 0.643522
[10]	training's auc: 0.729774	valid_1's auc: 0.645467
[11]	training's auc: 0.730699	valid_1's auc: 0.645668
[12]	training's auc: 0.732276	valid_1's auc: 0.646382
[13]	training's auc: 0.73299	valid_1's auc: 0.647427
[14]	training's auc: 0.734448	valid_1's auc: 0.648231
[15]	training's auc: 0.735911	valid_1's auc: 0.648721
[16]	training's auc: 0.737183	valid_1's auc: 0.649271
[17]	training's auc: 0.737708	valid_1's auc: 0.649972
[18]	training's auc: 0.739026	valid_1's auc: 0.650584
[19]	training's auc: 0.74018	valid_1's auc: 0.650996
[20]	training's auc: 0.741368	valid_1's auc: 0.651688
[21]	training's auc: 0.742413	valid_1's auc: 0.652198
[22]	training's auc: 0.743425	valid_1's auc: 0.652792
[23]	training's auc: 0.744444	valid_1's auc: 0.653158
[24]	training's auc: 0.745432	valid_1's auc: 0.653585
[25]	training's auc: 0.746296	valid_1's auc: 0.65391
[26]	training's auc: 0.74718	valid_1's auc: 0.654214
[27]	training's auc: 0.748048	valid_1's auc: 0.654553
[28]	training's auc: 0.748296	valid_1's auc: 0.654562
[29]	training's auc: 0.748481	valid_1's auc: 0.654611
[30]	training's auc: 0.749282	valid_1's auc: 0.65501
[31]	training's auc: 0.750054	valid_1's auc: 0.655196
[32]	training's auc: 0.750211	valid_1's auc: 0.655482
[33]	training's auc: 0.751054	valid_1's auc: 0.65584
[34]	training's auc: 0.751205	valid_1's auc: 0.65603
[35]	training's auc: 0.75117	valid_1's auc: 0.655891
[36]	training's auc: 0.751272	valid_1's auc: 0.655895
[37]	training's auc: 0.752011	valid_1's auc: 0.65613
[38]	training's auc: 0.752689	valid_1's auc: 0.656423
[39]	training's auc: 0.753443	valid_1's auc: 0.656646
[40]	training's auc: 0.754122	valid_1's auc: 0.656852
[41]	training's auc: 0.754215	valid_1's auc: 0.656764
[42]	training's auc: 0.754791	valid_1's auc: 0.65711
[43]	training's auc: 0.755407	valid_1's auc: 0.657205
[44]	training's auc: 0.755976	valid_1's auc: 0.657355
[45]	training's auc: 0.756604	valid_1's auc: 0.657626
[46]	training's auc: 0.757218	valid_1's auc: 0.657911
[47]	training's auc: 0.757721	valid_1's auc: 0.658061
[48]	training's auc: 0.758217	valid_1's auc: 0.658408
[49]	training's auc: 0.758663	valid_1's auc: 0.658542
[50]	training's auc: 0.759133	valid_1's auc: 0.658706

in model: goss  k-fold: 3

[1]	training's auc: 0.692079	valid_1's auc: 0.621322
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.700089	valid_1's auc: 0.629174
[3]	training's auc: 0.707075	valid_1's auc: 0.631727
[4]	training's auc: 0.712204	valid_1's auc: 0.635706
[5]	training's auc: 0.713285	valid_1's auc: 0.636803
[6]	training's auc: 0.71695	valid_1's auc: 0.637719
[7]	training's auc: 0.720849	valid_1's auc: 0.639854
[8]	training's auc: 0.724116	valid_1's auc: 0.641703
[9]	training's auc: 0.72667	valid_1's auc: 0.642544
[10]	training's auc: 0.729274	valid_1's auc: 0.644346
[11]	training's auc: 0.731118	valid_1's auc: 0.64505
[12]	training's auc: 0.73292	valid_1's auc: 0.646151
[13]	training's auc: 0.733622	valid_1's auc: 0.64662
[14]	training's auc: 0.735162	valid_1's auc: 0.647554
[15]	training's auc: 0.736471	valid_1's auc: 0.648158
[16]	training's auc: 0.737699	valid_1's auc: 0.648729
[17]	training's auc: 0.738441	valid_1's auc: 0.649768
[18]	training's auc: 0.739712	valid_1's auc: 0.650147
[19]	training's auc: 0.740895	valid_1's auc: 0.650805
[20]	training's auc: 0.742086	valid_1's auc: 0.651494
[21]	training's auc: 0.743137	valid_1's auc: 0.651909
[22]	training's auc: 0.744208	valid_1's auc: 0.652265
[23]	training's auc: 0.745157	valid_1's auc: 0.652558
[24]	training's auc: 0.746072	valid_1's auc: 0.652687
[25]	training's auc: 0.747071	valid_1's auc: 0.653116
[26]	training's auc: 0.7479	valid_1's auc: 0.653364
[27]	training's auc: 0.748751	valid_1's auc: 0.653738
[28]	training's auc: 0.749184	valid_1's auc: 0.653676
[29]	training's auc: 0.749335	valid_1's auc: 0.653568
[30]	training's auc: 0.750145	valid_1's auc: 0.653983
[31]	training's auc: 0.750923	valid_1's auc: 0.654333
[32]	training's auc: 0.750974	valid_1's auc: 0.654731
[33]	training's auc: 0.751758	valid_1's auc: 0.654927
[34]	training's auc: 0.752078	valid_1's auc: 0.655341
[35]	training's auc: 0.752192	valid_1's auc: 0.655302
[36]	training's auc: 0.752183	valid_1's auc: 0.655394
[37]	training's auc: 0.752881	valid_1's auc: 0.655594
[38]	training's auc: 0.753544	valid_1's auc: 0.655863
[39]	training's auc: 0.754235	valid_1's auc: 0.65612
[40]	training's auc: 0.754973	valid_1's auc: 0.656329
[41]	training's auc: 0.755054	valid_1's auc: 0.656287
[42]	training's auc: 0.755662	valid_1's auc: 0.656489
[43]	training's auc: 0.756298	valid_1's auc: 0.656736
[44]	training's auc: 0.756879	valid_1's auc: 0.656835
[45]	training's auc: 0.757547	valid_1's auc: 0.657268
[46]	training's auc: 0.758156	valid_1's auc: 0.657639
[47]	training's auc: 0.758745	valid_1's auc: 0.657797
[48]	training's auc: 0.759299	valid_1's auc: 0.658002
[49]	training's auc: 0.759794	valid_1's auc: 0.658131
[50]	training's auc: 0.760318	valid_1's auc: 0.658248

in model: goss  k-fold: 4

[1]	training's auc: 0.692267	valid_1's auc: 0.621328
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.70014	valid_1's auc: 0.629262
[3]	training's auc: 0.707336	valid_1's auc: 0.630842
[4]	training's auc: 0.711998	valid_1's auc: 0.635329
[5]	training's auc: 0.713779	valid_1's auc: 0.637084
[6]	training's auc: 0.717786	valid_1's auc: 0.638766
[7]	training's auc: 0.721974	valid_1's auc: 0.641495
[8]	training's auc: 0.725096	valid_1's auc: 0.643055
[9]	training's auc: 0.726924	valid_1's auc: 0.643556
[10]	training's auc: 0.72932	valid_1's auc: 0.644747
[11]	training's auc: 0.731123	valid_1's auc: 0.64595
[12]	training's auc: 0.732466	valid_1's auc: 0.64659
[13]	training's auc: 0.733199	valid_1's auc: 0.647286
[14]	training's auc: 0.73464	valid_1's auc: 0.647918
[15]	training's auc: 0.735999	valid_1's auc: 0.648392
[16]	training's auc: 0.737289	valid_1's auc: 0.648964
[17]	training's auc: 0.737793	valid_1's auc: 0.649447
[18]	training's auc: 0.73906	valid_1's auc: 0.649906
[19]	training's auc: 0.740339	valid_1's auc: 0.650527
[20]	training's auc: 0.741427	valid_1's auc: 0.650936
[21]	training's auc: 0.742484	valid_1's auc: 0.651488
[22]	training's auc: 0.743417	valid_1's auc: 0.651855
[23]	training's auc: 0.744454	valid_1's auc: 0.652272
[24]	training's auc: 0.745403	valid_1's auc: 0.652669
[25]	training's auc: 0.746231	valid_1's auc: 0.653042
[26]	training's auc: 0.747015	valid_1's auc: 0.653368
[27]	training's auc: 0.747899	valid_1's auc: 0.653856
[28]	training's auc: 0.748356	valid_1's auc: 0.654118
[29]	training's auc: 0.74861	valid_1's auc: 0.654249
[30]	training's auc: 0.749397	valid_1's auc: 0.654567
[31]	training's auc: 0.750171	valid_1's auc: 0.654879
[32]	training's auc: 0.750346	valid_1's auc: 0.655492
[33]	training's auc: 0.751024	valid_1's auc: 0.65573
[34]	training's auc: 0.751051	valid_1's auc: 0.655677
[35]	training's auc: 0.75118	valid_1's auc: 0.655972
[36]	training's auc: 0.75135	valid_1's auc: 0.656047
[37]	training's auc: 0.752103	valid_1's auc: 0.656278
[38]	training's auc: 0.75286	valid_1's auc: 0.65659
[39]	training's auc: 0.753521	valid_1's auc: 0.656786
[40]	training's auc: 0.754196	valid_1's auc: 0.65702
[41]	training's auc: 0.754418	valid_1's auc: 0.656877
[42]	training's auc: 0.755066	valid_1's auc: 0.657129
[43]	training's auc: 0.755661	valid_1's auc: 0.657331
[44]	training's auc: 0.756288	valid_1's auc: 0.657505
[45]	training's auc: 0.756879	valid_1's auc: 0.657715
[46]	training's auc: 0.757487	valid_1's auc: 0.657921
[47]	training's auc: 0.758065	valid_1's auc: 0.658123
[48]	training's auc: 0.758552	valid_1's auc: 0.658314
[49]	training's auc: 0.75908	valid_1's auc: 0.658431
[50]	training's auc: 0.759575	valid_1's auc: 0.658704

in model: rf  k-fold: 0

[1]	training's auc: 0.727727	valid_1's auc: 0.638815
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.727727	valid_1's auc: 0.638815
[3]	training's auc: 0.732386	valid_1's auc: 0.640081
[4]	training's auc: 0.735431	valid_1's auc: 0.64222
[5]	training's auc: 0.736421	valid_1's auc: 0.644856
[6]	training's auc: 0.735876	valid_1's auc: 0.64379
[7]	training's auc: 0.736825	valid_1's auc: 0.644507
[8]	training's auc: 0.737672	valid_1's auc: 0.645161
[9]	training's auc: 0.737339	valid_1's auc: 0.645056
[10]	training's auc: 0.737907	valid_1's auc: 0.645222
[11]	training's auc: 0.73826	valid_1's auc: 0.645343
[12]	training's auc: 0.738368	valid_1's auc: 0.645395
[13]	training's auc: 0.738578	valid_1's auc: 0.645805
[14]	training's auc: 0.738869	valid_1's auc: 0.645832
[15]	training's auc: 0.738961	valid_1's auc: 0.646079
[16]	training's auc: 0.738719	valid_1's auc: 0.645877
[17]	training's auc: 0.738327	valid_1's auc: 0.646129
[18]	training's auc: 0.738601	valid_1's auc: 0.646146
[19]	training's auc: 0.738605	valid_1's auc: 0.646187
[20]	training's auc: 0.738434	valid_1's auc: 0.646125
[21]	training's auc: 0.738288	valid_1's auc: 0.645972
[22]	training's auc: 0.738098	valid_1's auc: 0.645888
[23]	training's auc: 0.737818	valid_1's auc: 0.64576
[24]	training's auc: 0.737525	valid_1's auc: 0.645572
[25]	training's auc: 0.737333	valid_1's auc: 0.645393
[26]	training's auc: 0.737462	valid_1's auc: 0.645425
[27]	training's auc: 0.737691	valid_1's auc: 0.64549
[28]	training's auc: 0.737734	valid_1's auc: 0.645698
[29]	training's auc: 0.737641	valid_1's auc: 0.645762
[30]	training's auc: 0.737508	valid_1's auc: 0.645559
[31]	training's auc: 0.737691	valid_1's auc: 0.645615
[32]	training's auc: 0.737613	valid_1's auc: 0.645803
[33]	training's auc: 0.737764	valid_1's auc: 0.645856
[34]	training's auc: 0.737601	valid_1's auc: 0.646011
[35]	training's auc: 0.737502	valid_1's auc: 0.645989
[36]	training's auc: 0.737333	valid_1's auc: 0.645999
[37]	training's auc: 0.737624	valid_1's auc: 0.646203
[38]	training's auc: 0.737707	valid_1's auc: 0.646281
[39]	training's auc: 0.737972	valid_1's auc: 0.646464
[40]	training's auc: 0.738108	valid_1's auc: 0.646478
[41]	training's auc: 0.737838	valid_1's auc: 0.646351
[42]	training's auc: 0.737979	valid_1's auc: 0.646376
[43]	training's auc: 0.738071	valid_1's auc: 0.64645
[44]	training's auc: 0.738197	valid_1's auc: 0.646475
[45]	training's auc: 0.738238	valid_1's auc: 0.646562
[46]	training's auc: 0.738436	valid_1's auc: 0.646685
[47]	training's auc: 0.738613	valid_1's auc: 0.646813
[48]	training's auc: 0.738713	valid_1's auc: 0.646829
[49]	training's auc: 0.738925	valid_1's auc: 0.646964
[50]	training's auc: 0.738855	valid_1's auc: 0.6469

in model: rf  k-fold: 1

[1]	training's auc: 0.72786	valid_1's auc: 0.638725
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.72786	valid_1's auc: 0.638725
[3]	training's auc: 0.73244	valid_1's auc: 0.640016
[4]	training's auc: 0.73518	valid_1's auc: 0.642702
[5]	training's auc: 0.736113	valid_1's auc: 0.645034
[6]	training's auc: 0.73542	valid_1's auc: 0.643706
[7]	training's auc: 0.736607	valid_1's auc: 0.644481
[8]	training's auc: 0.737591	valid_1's auc: 0.64523
[9]	training's auc: 0.73715	valid_1's auc: 0.645021
[10]	training's auc: 0.737661	valid_1's auc: 0.64537
[11]	training's auc: 0.738223	valid_1's auc: 0.645691
[12]	training's auc: 0.738415	valid_1's auc: 0.645835
[13]	training's auc: 0.738497	valid_1's auc: 0.646117
[14]	training's auc: 0.738755	valid_1's auc: 0.646153
[15]	training's auc: 0.73886	valid_1's auc: 0.646428
[16]	training's auc: 0.73871	valid_1's auc: 0.646274
[17]	training's auc: 0.738413	valid_1's auc: 0.646627
[18]	training's auc: 0.738669	valid_1's auc: 0.646642
[19]	training's auc: 0.738646	valid_1's auc: 0.64671
[20]	training's auc: 0.738391	valid_1's auc: 0.64649
[21]	training's auc: 0.738095	valid_1's auc: 0.646256
[22]	training's auc: 0.737779	valid_1's auc: 0.645994
[23]	training's auc: 0.737479	valid_1's auc: 0.645785
[24]	training's auc: 0.737154	valid_1's auc: 0.645578
[25]	training's auc: 0.736895	valid_1's auc: 0.645388
[26]	training's auc: 0.737142	valid_1's auc: 0.645566
[27]	training's auc: 0.737364	valid_1's auc: 0.645627
[28]	training's auc: 0.737373	valid_1's auc: 0.645788
[29]	training's auc: 0.737386	valid_1's auc: 0.645961
[30]	training's auc: 0.737196	valid_1's auc: 0.645717
[31]	training's auc: 0.737369	valid_1's auc: 0.645798
[32]	training's auc: 0.737325	valid_1's auc: 0.645937
[33]	training's auc: 0.737488	valid_1's auc: 0.645993
[34]	training's auc: 0.737327	valid_1's auc: 0.646129
[35]	training's auc: 0.737241	valid_1's auc: 0.646176
[36]	training's auc: 0.73709	valid_1's auc: 0.646219
[37]	training's auc: 0.737357	valid_1's auc: 0.646417
[38]	training's auc: 0.737419	valid_1's auc: 0.646528
[39]	training's auc: 0.73768	valid_1's auc: 0.646722
[40]	training's auc: 0.737827	valid_1's auc: 0.646758
[41]	training's auc: 0.737514	valid_1's auc: 0.646551
[42]	training's auc: 0.737675	valid_1's auc: 0.646607
[43]	training's auc: 0.73778	valid_1's auc: 0.646676
[44]	training's auc: 0.737907	valid_1's auc: 0.646671
[45]	training's auc: 0.738053	valid_1's auc: 0.646845
[46]	training's auc: 0.738251	valid_1's auc: 0.646963
[47]	training's auc: 0.738453	valid_1's auc: 0.647105
[48]	training's auc: 0.738661	valid_1's auc: 0.647223
[49]	training's auc: 0.738925	valid_1's auc: 0.647384
[50]	training's auc: 0.738797	valid_1's auc: 0.647269

in model: rf  k-fold: 2

[1]	training's auc: 0.728528	valid_1's auc: 0.639825
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.728528	valid_1's auc: 0.639825
[3]	training's auc: 0.732914	valid_1's auc: 0.641933
[4]	training's auc: 0.736537	valid_1's auc: 0.644416
[5]	training's auc: 0.737308	valid_1's auc: 0.646552
[6]	training's auc: 0.736703	valid_1's auc: 0.645416
[7]	training's auc: 0.737769	valid_1's auc: 0.6461
[8]	training's auc: 0.738622	valid_1's auc: 0.646754
[9]	training's auc: 0.738069	valid_1's auc: 0.646216
[10]	training's auc: 0.738601	valid_1's auc: 0.646643
[11]	training's auc: 0.738995	valid_1's auc: 0.646704
[12]	training's auc: 0.739091	valid_1's auc: 0.64672
[13]	training's auc: 0.739215	valid_1's auc: 0.647158
[14]	training's auc: 0.739472	valid_1's auc: 0.647374
[15]	training's auc: 0.739537	valid_1's auc: 0.647628
[16]	training's auc: 0.739231	valid_1's auc: 0.647338
[17]	training's auc: 0.738778	valid_1's auc: 0.647568
[18]	training's auc: 0.739059	valid_1's auc: 0.647552
[19]	training's auc: 0.739045	valid_1's auc: 0.647529
[20]	training's auc: 0.738897	valid_1's auc: 0.647275
[21]	training's auc: 0.738638	valid_1's auc: 0.646975
[22]	training's auc: 0.738328	valid_1's auc: 0.646709
[23]	training's auc: 0.737994	valid_1's auc: 0.64647
[24]	training's auc: 0.737643	valid_1's auc: 0.64623
[25]	training's auc: 0.737502	valid_1's auc: 0.646124
[26]	training's auc: 0.737636	valid_1's auc: 0.646164
[27]	training's auc: 0.73783	valid_1's auc: 0.646187
[28]	training's auc: 0.737839	valid_1's auc: 0.646347
[29]	training's auc: 0.737791	valid_1's auc: 0.646446
[30]	training's auc: 0.737653	valid_1's auc: 0.646275
[31]	training's auc: 0.737804	valid_1's auc: 0.646289
[32]	training's auc: 0.737734	valid_1's auc: 0.646428
[33]	training's auc: 0.737871	valid_1's auc: 0.646484
[34]	training's auc: 0.73769	valid_1's auc: 0.646587
[35]	training's auc: 0.73763	valid_1's auc: 0.646618
[36]	training's auc: 0.737467	valid_1's auc: 0.646638
[37]	training's auc: 0.737734	valid_1's auc: 0.646845
[38]	training's auc: 0.737781	valid_1's auc: 0.646932
[39]	training's auc: 0.738004	valid_1's auc: 0.647095
[40]	training's auc: 0.738173	valid_1's auc: 0.647201
[41]	training's auc: 0.737886	valid_1's auc: 0.647002
[42]	training's auc: 0.738039	valid_1's auc: 0.647021
[43]	training's auc: 0.738123	valid_1's auc: 0.647107
[44]	training's auc: 0.738249	valid_1's auc: 0.647101
[45]	training's auc: 0.738272	valid_1's auc: 0.647177
[46]	training's auc: 0.73846	valid_1's auc: 0.647292
[47]	training's auc: 0.738643	valid_1's auc: 0.647405
[48]	training's auc: 0.738842	valid_1's auc: 0.647509
[49]	training's auc: 0.739117	valid_1's auc: 0.64766
[50]	training's auc: 0.73903	valid_1's auc: 0.647553

in model: rf  k-fold: 3

[1]	training's auc: 0.728062	valid_1's auc: 0.638239
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.728062	valid_1's auc: 0.638239
[3]	training's auc: 0.732424	valid_1's auc: 0.640188
[4]	training's auc: 0.7354	valid_1's auc: 0.642276
[5]	training's auc: 0.736271	valid_1's auc: 0.644734
[6]	training's auc: 0.735464	valid_1's auc: 0.643537
[7]	training's auc: 0.736555	valid_1's auc: 0.644247
[8]	training's auc: 0.737463	valid_1's auc: 0.645012
[9]	training's auc: 0.737068	valid_1's auc: 0.644649
[10]	training's auc: 0.737738	valid_1's auc: 0.645097
[11]	training's auc: 0.738327	valid_1's auc: 0.645632
[12]	training's auc: 0.738517	valid_1's auc: 0.645727
[13]	training's auc: 0.738554	valid_1's auc: 0.64607
[14]	training's auc: 0.738798	valid_1's auc: 0.646077
[15]	training's auc: 0.738906	valid_1's auc: 0.646333
[16]	training's auc: 0.73863	valid_1's auc: 0.646094
[17]	training's auc: 0.738158	valid_1's auc: 0.646268
[18]	training's auc: 0.738482	valid_1's auc: 0.646278
[19]	training's auc: 0.738537	valid_1's auc: 0.646322
[20]	training's auc: 0.738272	valid_1's auc: 0.646144
[21]	training's auc: 0.738023	valid_1's auc: 0.645924
[22]	training's auc: 0.737738	valid_1's auc: 0.645738
[23]	training's auc: 0.737413	valid_1's auc: 0.645549
[24]	training's auc: 0.737086	valid_1's auc: 0.645354
[25]	training's auc: 0.73682	valid_1's auc: 0.645143
[26]	training's auc: 0.73706	valid_1's auc: 0.645283
[27]	training's auc: 0.737324	valid_1's auc: 0.645415
[28]	training's auc: 0.737343	valid_1's auc: 0.645564
[29]	training's auc: 0.737326	valid_1's auc: 0.645653
[30]	training's auc: 0.737193	valid_1's auc: 0.645473
[31]	training's auc: 0.737366	valid_1's auc: 0.645569
[32]	training's auc: 0.73729	valid_1's auc: 0.6457
[33]	training's auc: 0.73748	valid_1's auc: 0.645806
[34]	training's auc: 0.737301	valid_1's auc: 0.645919
[35]	training's auc: 0.73721	valid_1's auc: 0.645924
[36]	training's auc: 0.737029	valid_1's auc: 0.645972
[37]	training's auc: 0.737289	valid_1's auc: 0.646176
[38]	training's auc: 0.737352	valid_1's auc: 0.646316
[39]	training's auc: 0.737609	valid_1's auc: 0.646491
[40]	training's auc: 0.737772	valid_1's auc: 0.646523
[41]	training's auc: 0.737472	valid_1's auc: 0.646336
[42]	training's auc: 0.737615	valid_1's auc: 0.646337
[43]	training's auc: 0.737722	valid_1's auc: 0.646464
[44]	training's auc: 0.737853	valid_1's auc: 0.646527
[45]	training's auc: 0.737909	valid_1's auc: 0.646583
[46]	training's auc: 0.738093	valid_1's auc: 0.646704
[47]	training's auc: 0.738259	valid_1's auc: 0.646833
[48]	training's auc: 0.738345	valid_1's auc: 0.646815
[49]	training's auc: 0.738609	valid_1's auc: 0.646945
[50]	training's auc: 0.738485	valid_1's auc: 0.646836

in model: rf  k-fold: 4

[1]	training's auc: 0.72755	valid_1's auc: 0.638788
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.72755	valid_1's auc: 0.638788
[3]	training's auc: 0.732495	valid_1's auc: 0.640532
[4]	training's auc: 0.735259	valid_1's auc: 0.642429
[5]	training's auc: 0.736408	valid_1's auc: 0.645379
[6]	training's auc: 0.735787	valid_1's auc: 0.644214
[7]	training's auc: 0.736914	valid_1's auc: 0.645044
[8]	training's auc: 0.73791	valid_1's auc: 0.645989
[9]	training's auc: 0.737652	valid_1's auc: 0.645663
[10]	training's auc: 0.738156	valid_1's auc: 0.645898
[11]	training's auc: 0.738565	valid_1's auc: 0.646041
[12]	training's auc: 0.738746	valid_1's auc: 0.646173
[13]	training's auc: 0.738844	valid_1's auc: 0.646513
[14]	training's auc: 0.739152	valid_1's auc: 0.646512
[15]	training's auc: 0.739232	valid_1's auc: 0.646708
[16]	training's auc: 0.739044	valid_1's auc: 0.64656
[17]	training's auc: 0.738624	valid_1's auc: 0.64677
[18]	training's auc: 0.738901	valid_1's auc: 0.646797
[19]	training's auc: 0.738864	valid_1's auc: 0.646724
[20]	training's auc: 0.738611	valid_1's auc: 0.646553
[21]	training's auc: 0.738344	valid_1's auc: 0.646316
[22]	training's auc: 0.738039	valid_1's auc: 0.646141
[23]	training's auc: 0.737736	valid_1's auc: 0.645976
[24]	training's auc: 0.737418	valid_1's auc: 0.645757
[25]	training's auc: 0.737171	valid_1's auc: 0.645522
[26]	training's auc: 0.737339	valid_1's auc: 0.645597
[27]	training's auc: 0.737546	valid_1's auc: 0.645697
[28]	training's auc: 0.737669	valid_1's auc: 0.645915
[29]	training's auc: 0.737658	valid_1's auc: 0.646045
[30]	training's auc: 0.73758	valid_1's auc: 0.645886
[31]	training's auc: 0.737767	valid_1's auc: 0.64595
[32]	training's auc: 0.737687	valid_1's auc: 0.646089
[33]	training's auc: 0.737833	valid_1's auc: 0.64613
[34]	training's auc: 0.737677	valid_1's auc: 0.64626
[35]	training's auc: 0.737601	valid_1's auc: 0.646274
[36]	training's auc: 0.737415	valid_1's auc: 0.646311
[37]	training's auc: 0.737689	valid_1's auc: 0.646522
[38]	training's auc: 0.737773	valid_1's auc: 0.646619
[39]	training's auc: 0.737978	valid_1's auc: 0.646791
[40]	training's auc: 0.738134	valid_1's auc: 0.646815
[41]	training's auc: 0.737867	valid_1's auc: 0.646655
[42]	training's auc: 0.738017	valid_1's auc: 0.646689
[43]	training's auc: 0.738119	valid_1's auc: 0.64674
[44]	training's auc: 0.738275	valid_1's auc: 0.646774
[45]	training's auc: 0.738313	valid_1's auc: 0.646835
[46]	training's auc: 0.73849	valid_1's auc: 0.64695
[47]	training's auc: 0.738659	valid_1's auc: 0.647082
[48]	training's auc: 0.738745	valid_1's auc: 0.647077
[49]	training's auc: 0.73898	valid_1's auc: 0.647209
[50]	training's auc: 0.738854	valid_1's auc: 0.6471

in model: gbdt  k-fold: 0

[1]	training's auc: 0.775026	valid_1's auc: 0.654673
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.783797	valid_1's auc: 0.659294
[3]	training's auc: 0.785997	valid_1's auc: 0.660352
[4]	training's auc: 0.788412	valid_1's auc: 0.661341
[5]	training's auc: 0.790177	valid_1's auc: 0.66228
[6]	training's auc: 0.792492	valid_1's auc: 0.66354
[7]	training's auc: 0.79375	valid_1's auc: 0.664067
[8]	training's auc: 0.794574	valid_1's auc: 0.664374
[9]	training's auc: 0.796182	valid_1's auc: 0.664981
[10]	training's auc: 0.797091	valid_1's auc: 0.665293
[11]	training's auc: 0.79793	valid_1's auc: 0.66548
[12]	training's auc: 0.79849	valid_1's auc: 0.665619
[13]	training's auc: 0.798823	valid_1's auc: 0.665959
[14]	training's auc: 0.799582	valid_1's auc: 0.666204
[15]	training's auc: 0.800391	valid_1's auc: 0.666413
[16]	training's auc: 0.801328	valid_1's auc: 0.666726
[17]	training's auc: 0.801895	valid_1's auc: 0.666991
[18]	training's auc: 0.802434	valid_1's auc: 0.667093
[19]	training's auc: 0.803334	valid_1's auc: 0.667356
[20]	training's auc: 0.804341	valid_1's auc: 0.667759
[21]	training's auc: 0.804924	valid_1's auc: 0.667956
[22]	training's auc: 0.805349	valid_1's auc: 0.668048
[23]	training's auc: 0.806081	valid_1's auc: 0.668384
[24]	training's auc: 0.806697	valid_1's auc: 0.668612
[25]	training's auc: 0.807207	valid_1's auc: 0.668744
[26]	training's auc: 0.807852	valid_1's auc: 0.668919
[27]	training's auc: 0.808437	valid_1's auc: 0.668998
[28]	training's auc: 0.808587	valid_1's auc: 0.669075
[29]	training's auc: 0.80866	valid_1's auc: 0.669107
[30]	training's auc: 0.809241	valid_1's auc: 0.669164
[31]	training's auc: 0.809867	valid_1's auc: 0.669313
[32]	training's auc: 0.810413	valid_1's auc: 0.669476
[33]	training's auc: 0.810953	valid_1's auc: 0.669624
[34]	training's auc: 0.811028	valid_1's auc: 0.669655
[35]	training's auc: 0.811091	valid_1's auc: 0.669661
[36]	training's auc: 0.811085	valid_1's auc: 0.669642
[37]	training's auc: 0.811775	valid_1's auc: 0.669851
[38]	training's auc: 0.812228	valid_1's auc: 0.669974
[39]	training's auc: 0.812816	valid_1's auc: 0.67014
[40]	training's auc: 0.813465	valid_1's auc: 0.670352
[41]	training's auc: 0.814186	valid_1's auc: 0.670677
[42]	training's auc: 0.814801	valid_1's auc: 0.670842
[43]	training's auc: 0.815469	valid_1's auc: 0.67106
[44]	training's auc: 0.816036	valid_1's auc: 0.671254
[45]	training's auc: 0.816574	valid_1's auc: 0.671425
[46]	training's auc: 0.817121	valid_1's auc: 0.671549
[47]	training's auc: 0.817672	valid_1's auc: 0.671707
[48]	training's auc: 0.818209	valid_1's auc: 0.671845
[49]	training's auc: 0.818633	valid_1's auc: 0.671947
[50]	training's auc: 0.819134	valid_1's auc: 0.672186

in model: gbdt  k-fold: 1

[1]	training's auc: 0.774176	valid_1's auc: 0.653667
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.7819	valid_1's auc: 0.657481
[3]	training's auc: 0.785971	valid_1's auc: 0.659562
[4]	training's auc: 0.788598	valid_1's auc: 0.660617
[5]	training's auc: 0.789843	valid_1's auc: 0.661318
[6]	training's auc: 0.793066	valid_1's auc: 0.662849
[7]	training's auc: 0.794312	valid_1's auc: 0.663503
[8]	training's auc: 0.79497	valid_1's auc: 0.66381
[9]	training's auc: 0.796161	valid_1's auc: 0.664268
[10]	training's auc: 0.796765	valid_1's auc: 0.664656
[11]	training's auc: 0.797573	valid_1's auc: 0.664798
[12]	training's auc: 0.798165	valid_1's auc: 0.664924
[13]	training's auc: 0.798483	valid_1's auc: 0.66522
[14]	training's auc: 0.799147	valid_1's auc: 0.665361
[15]	training's auc: 0.799935	valid_1's auc: 0.665714
[16]	training's auc: 0.801512	valid_1's auc: 0.666282
[17]	training's auc: 0.80208	valid_1's auc: 0.66644
[18]	training's auc: 0.802496	valid_1's auc: 0.666549
[19]	training's auc: 0.803123	valid_1's auc: 0.666788
[20]	training's auc: 0.804053	valid_1's auc: 0.667129
[21]	training's auc: 0.804554	valid_1's auc: 0.667265
[22]	training's auc: 0.805021	valid_1's auc: 0.667363
[23]	training's auc: 0.805915	valid_1's auc: 0.667789
[24]	training's auc: 0.806717	valid_1's auc: 0.668161
[25]	training's auc: 0.807446	valid_1's auc: 0.668312
[26]	training's auc: 0.807939	valid_1's auc: 0.668427
[27]	training's auc: 0.808551	valid_1's auc: 0.668537
[28]	training's auc: 0.808722	valid_1's auc: 0.668642
[29]	training's auc: 0.808809	valid_1's auc: 0.66868
[30]	training's auc: 0.809437	valid_1's auc: 0.668748
[31]	training's auc: 0.810172	valid_1's auc: 0.669041
[32]	training's auc: 0.810653	valid_1's auc: 0.669238
[33]	training's auc: 0.811118	valid_1's auc: 0.669383
[34]	training's auc: 0.811241	valid_1's auc: 0.669442
[35]	training's auc: 0.811307	valid_1's auc: 0.669478
[36]	training's auc: 0.811309	valid_1's auc: 0.669482
[37]	training's auc: 0.812081	valid_1's auc: 0.66973
[38]	training's auc: 0.812558	valid_1's auc: 0.669888
[39]	training's auc: 0.813149	valid_1's auc: 0.670063
[40]	training's auc: 0.813767	valid_1's auc: 0.670257
[41]	training's auc: 0.81443	valid_1's auc: 0.670552
[42]	training's auc: 0.814971	valid_1's auc: 0.670716
[43]	training's auc: 0.815525	valid_1's auc: 0.67092
[44]	training's auc: 0.816086	valid_1's auc: 0.67108
[45]	training's auc: 0.816614	valid_1's auc: 0.671241
[46]	training's auc: 0.817045	valid_1's auc: 0.67134
[47]	training's auc: 0.817621	valid_1's auc: 0.671524
[48]	training's auc: 0.818091	valid_1's auc: 0.671608
[49]	training's auc: 0.818592	valid_1's auc: 0.67174
[50]	training's auc: 0.819101	valid_1's auc: 0.671987

in model: gbdt  k-fold: 2

[1]	training's auc: 0.774614	valid_1's auc: 0.652876
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.781666	valid_1's auc: 0.656027
[3]	training's auc: 0.785891	valid_1's auc: 0.659051
[4]	training's auc: 0.788299	valid_1's auc: 0.660476
[5]	training's auc: 0.791012	valid_1's auc: 0.661923
[6]	training's auc: 0.793854	valid_1's auc: 0.663224
[7]	training's auc: 0.794376	valid_1's auc: 0.663447
[8]	training's auc: 0.795047	valid_1's auc: 0.663855
[9]	training's auc: 0.795963	valid_1's auc: 0.664222
[10]	training's auc: 0.796409	valid_1's auc: 0.664656
[11]	training's auc: 0.797195	valid_1's auc: 0.664807
[12]	training's auc: 0.798433	valid_1's auc: 0.665173
[13]	training's auc: 0.798817	valid_1's auc: 0.665519
[14]	training's auc: 0.799545	valid_1's auc: 0.665627
[15]	training's auc: 0.800152	valid_1's auc: 0.665998
[16]	training's auc: 0.80109	valid_1's auc: 0.666308
[17]	training's auc: 0.802087	valid_1's auc: 0.666622
[18]	training's auc: 0.80261	valid_1's auc: 0.666729
[19]	training's auc: 0.803157	valid_1's auc: 0.666936
[20]	training's auc: 0.804302	valid_1's auc: 0.667408
[21]	training's auc: 0.804901	valid_1's auc: 0.667578
[22]	training's auc: 0.805336	valid_1's auc: 0.667681
[23]	training's auc: 0.80614	valid_1's auc: 0.668005
[24]	training's auc: 0.80674	valid_1's auc: 0.668235
[25]	training's auc: 0.80738	valid_1's auc: 0.668393
[26]	training's auc: 0.808023	valid_1's auc: 0.668585
[27]	training's auc: 0.808649	valid_1's auc: 0.668672
[28]	training's auc: 0.808792	valid_1's auc: 0.668764
[29]	training's auc: 0.808855	valid_1's auc: 0.668827
[30]	training's auc: 0.809432	valid_1's auc: 0.668891
[31]	training's auc: 0.810023	valid_1's auc: 0.668987
[32]	training's auc: 0.810605	valid_1's auc: 0.669201
[33]	training's auc: 0.811151	valid_1's auc: 0.669327
[34]	training's auc: 0.811219	valid_1's auc: 0.669349
[35]	training's auc: 0.811273	valid_1's auc: 0.669376
[36]	training's auc: 0.811277	valid_1's auc: 0.669401
[37]	training's auc: 0.812031	valid_1's auc: 0.669649
[38]	training's auc: 0.812642	valid_1's auc: 0.669869
[39]	training's auc: 0.813321	valid_1's auc: 0.670069
[40]	training's auc: 0.813934	valid_1's auc: 0.670278
[41]	training's auc: 0.814492	valid_1's auc: 0.670561
[42]	training's auc: 0.815008	valid_1's auc: 0.670714
[43]	training's auc: 0.815643	valid_1's auc: 0.670937
[44]	training's auc: 0.816194	valid_1's auc: 0.67107
[45]	training's auc: 0.816752	valid_1's auc: 0.671284
[46]	training's auc: 0.817254	valid_1's auc: 0.671394
[47]	training's auc: 0.817715	valid_1's auc: 0.671503
[48]	training's auc: 0.818137	valid_1's auc: 0.671594
[49]	training's auc: 0.818594	valid_1's auc: 0.671734
[50]	training's auc: 0.81908	valid_1's auc: 0.671972

in model: gbdt  k-fold: 3

[1]	training's auc: 0.774722	valid_1's auc: 0.654966
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.78175	valid_1's auc: 0.658261
[3]	training's auc: 0.785915	valid_1's auc: 0.660523
[4]	training's auc: 0.789004	valid_1's auc: 0.661595
[5]	training's auc: 0.790886	valid_1's auc: 0.662614
[6]	training's auc: 0.793364	valid_1's auc: 0.663766
[7]	training's auc: 0.793892	valid_1's auc: 0.663994
[8]	training's auc: 0.794366	valid_1's auc: 0.66422
[9]	training's auc: 0.79581	valid_1's auc: 0.664722
[10]	training's auc: 0.79674	valid_1's auc: 0.665133
[11]	training's auc: 0.797795	valid_1's auc: 0.665403
[12]	training's auc: 0.798486	valid_1's auc: 0.665488
[13]	training's auc: 0.798838	valid_1's auc: 0.665741
[14]	training's auc: 0.799415	valid_1's auc: 0.665954
[15]	training's auc: 0.800551	valid_1's auc: 0.666367
[16]	training's auc: 0.801492	valid_1's auc: 0.666744
[17]	training's auc: 0.802233	valid_1's auc: 0.667057
[18]	training's auc: 0.80272	valid_1's auc: 0.667178
[19]	training's auc: 0.803515	valid_1's auc: 0.667326
[20]	training's auc: 0.804466	valid_1's auc: 0.667753
[21]	training's auc: 0.804937	valid_1's auc: 0.66788
[22]	training's auc: 0.80543	valid_1's auc: 0.667945
[23]	training's auc: 0.806312	valid_1's auc: 0.668298
[24]	training's auc: 0.807053	valid_1's auc: 0.668565
[25]	training's auc: 0.807689	valid_1's auc: 0.668726
[26]	training's auc: 0.808117	valid_1's auc: 0.668855
[27]	training's auc: 0.808716	valid_1's auc: 0.668964
[28]	training's auc: 0.808865	valid_1's auc: 0.669028
[29]	training's auc: 0.808912	valid_1's auc: 0.669082
[30]	training's auc: 0.809564	valid_1's auc: 0.669182
[31]	training's auc: 0.810212	valid_1's auc: 0.669291
[32]	training's auc: 0.810748	valid_1's auc: 0.669468
[33]	training's auc: 0.811146	valid_1's auc: 0.669534
[34]	training's auc: 0.811226	valid_1's auc: 0.66958
[35]	training's auc: 0.81129	valid_1's auc: 0.669645
[36]	training's auc: 0.811298	valid_1's auc: 0.669636
[37]	training's auc: 0.811978	valid_1's auc: 0.66983
[38]	training's auc: 0.812504	valid_1's auc: 0.669975
[39]	training's auc: 0.813166	valid_1's auc: 0.67017
[40]	training's auc: 0.81378	valid_1's auc: 0.670349
[41]	training's auc: 0.814464	valid_1's auc: 0.670608
[42]	training's auc: 0.815004	valid_1's auc: 0.670728
[43]	training's auc: 0.815668	valid_1's auc: 0.670939
[44]	training's auc: 0.816194	valid_1's auc: 0.671133
[45]	training's auc: 0.81665	valid_1's auc: 0.671253
[46]	training's auc: 0.817189	valid_1's auc: 0.671401
[47]	training's auc: 0.817684	valid_1's auc: 0.67152
[48]	training's auc: 0.818144	valid_1's auc: 0.671616
[49]	training's auc: 0.818677	valid_1's auc: 0.671776
[50]	training's auc: 0.819226	valid_1's auc: 0.672008

in model: gbdt  k-fold: 4

[1]	training's auc: 0.774522	valid_1's auc: 0.654341
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.781389	valid_1's auc: 0.657467
[3]	training's auc: 0.78451	valid_1's auc: 0.659221
[4]	training's auc: 0.786887	valid_1's auc: 0.660355
[5]	training's auc: 0.789416	valid_1's auc: 0.66213
[6]	training's auc: 0.791879	valid_1's auc: 0.663528
[7]	training's auc: 0.792633	valid_1's auc: 0.66374
[8]	training's auc: 0.793741	valid_1's auc: 0.664228
[9]	training's auc: 0.795372	valid_1's auc: 0.66474
[10]	training's auc: 0.795964	valid_1's auc: 0.665075
[11]	training's auc: 0.796802	valid_1's auc: 0.665168
[12]	training's auc: 0.797373	valid_1's auc: 0.665337
[13]	training's auc: 0.797789	valid_1's auc: 0.665602
[14]	training's auc: 0.798584	valid_1's auc: 0.66577
[15]	training's auc: 0.799825	valid_1's auc: 0.666196
[16]	training's auc: 0.800849	valid_1's auc: 0.66665
[17]	training's auc: 0.801794	valid_1's auc: 0.666982
[18]	training's auc: 0.802255	valid_1's auc: 0.667079
[19]	training's auc: 0.803118	valid_1's auc: 0.667385
[20]	training's auc: 0.803934	valid_1's auc: 0.667764
[21]	training's auc: 0.804535	valid_1's auc: 0.667982
[22]	training's auc: 0.805085	valid_1's auc: 0.668124
[23]	training's auc: 0.805721	valid_1's auc: 0.668484
[24]	training's auc: 0.806359	valid_1's auc: 0.668744
[25]	training's auc: 0.806856	valid_1's auc: 0.668836
[26]	training's auc: 0.807508	valid_1's auc: 0.669016
[27]	training's auc: 0.808114	valid_1's auc: 0.669101
[28]	training's auc: 0.808287	valid_1's auc: 0.669206
[29]	training's auc: 0.808349	valid_1's auc: 0.669251
[30]	training's auc: 0.808997	valid_1's auc: 0.669327
[31]	training's auc: 0.809429	valid_1's auc: 0.669446
[32]	training's auc: 0.810164	valid_1's auc: 0.669667
[33]	training's auc: 0.810798	valid_1's auc: 0.669892
[34]	training's auc: 0.810887	valid_1's auc: 0.669932
[35]	training's auc: 0.810907	valid_1's auc: 0.669968
[36]	training's auc: 0.810929	valid_1's auc: 0.669999
[37]	training's auc: 0.811585	valid_1's auc: 0.670186
[38]	training's auc: 0.812155	valid_1's auc: 0.670392
[39]	training's auc: 0.812847	valid_1's auc: 0.670563
[40]	training's auc: 0.813449	valid_1's auc: 0.670744
[41]	training's auc: 0.814199	valid_1's auc: 0.671077
[42]	training's auc: 0.814713	valid_1's auc: 0.671259
[43]	training's auc: 0.815274	valid_1's auc: 0.671458
[44]	training's auc: 0.815823	valid_1's auc: 0.671635
[45]	training's auc: 0.816479	valid_1's auc: 0.671823
[46]	training's auc: 0.816992	valid_1's auc: 0.671997
[47]	training's auc: 0.817502	valid_1's auc: 0.672148
[48]	training's auc: 0.818089	valid_1's auc: 0.672295
[49]	training's auc: 0.818542	valid_1's auc: 0.672418
[50]	training's auc: 0.819029	valid_1's auc: 0.67266

in model: gbdt  k-fold: 0

0: learn: 0.7272443	test: 0.6373735	bestTest: 0.6373735 (0)	total: 3.82s	remaining: 34.3s
1: learn: 0.7376864	test: 0.6454675	bestTest: 0.6454675 (1)	total: 7.54s	remaining: 30.1s
2: learn: 0.7388203	test: 0.646298	bestTest: 0.646298 (2)	total: 11.2s	remaining: 26.2s
3: learn: 0.7404818	test: 0.6473632	bestTest: 0.6473632 (3)	total: 14.9s	remaining: 22.3s
4: learn: 0.7420809	test: 0.6483209	bestTest: 0.6483209 (4)	total: 18.7s	remaining: 18.7s
5: learn: 0.7432716	test: 0.6486151	bestTest: 0.6486151 (5)	total: 22.5s	remaining: 15s
6: learn: 0.7440606	test: 0.6488273	bestTest: 0.6488273 (6)	total: 26.2s	remaining: 11.2s
7: learn: 0.744725	test: 0.6487505	bestTest: 0.6488273 (6)	total: 29.9s	remaining: 7.48s
8: learn: 0.7451502	test: 0.6489943	bestTest: 0.6489943 (8)	total: 33.6s	remaining: 3.73s
9: learn: 0.7453125	test: 0.6491628	bestTest: 0.6491628 (9)	total: 37.3s	remaining: 0us

bestTest = 0.649162818
bestIteration = 9


in model: gbdt  k-fold: 1

0: learn: 0.7247631	test: 0.6403272	bestTest: 0.6403272 (0)	total: 3.32s	remaining: 29.9s
1: learn: 0.7354186	test: 0.6451581	bestTest: 0.6451581 (1)	total: 6.78s	remaining: 27.1s
2: learn: 0.736915	test: 0.6463676	bestTest: 0.6463676 (2)	total: 10.3s	remaining: 24s
3: learn: 0.7387093	test: 0.6475144	bestTest: 0.6475144 (3)	total: 13.8s	remaining: 20.7s
4: learn: 0.7403844	test: 0.6478195	bestTest: 0.6478195 (4)	total: 17.3s	remaining: 17.3s
5: learn: 0.7413972	test: 0.6483495	bestTest: 0.6483495 (5)	total: 20.8s	remaining: 13.9s
6: learn: 0.7423832	test: 0.6486579	bestTest: 0.6486579 (6)	total: 24.4s	remaining: 10.5s
7: learn: 0.7427952	test: 0.6487613	bestTest: 0.6487613 (7)	total: 28s	remaining: 7s
8: learn: 0.7434174	test: 0.6490678	bestTest: 0.6490678 (8)	total: 31.6s	remaining: 3.51s
9: learn: 0.7439238	test: 0.6490086	bestTest: 0.6490678 (8)	total: 35.2s	remaining: 0us

bestTest = 0.6490678423
bestIteration = 8


in model: gbdt  k-fold: 2

0: learn: 0.7256705	test: 0.638606	bestTest: 0.638606 (0)	total: 3.28s	remaining: 29.5s
1: learn: 0.7362996	test: 0.6452091	bestTest: 0.6452091 (1)	total: 6.72s	remaining: 26.9s
2: learn: 0.7383093	test: 0.6467837	bestTest: 0.6467837 (2)	total: 10.1s	remaining: 23.6s
3: learn: 0.7400204	test: 0.647441	bestTest: 0.647441 (3)	total: 13.6s	remaining: 20.3s
4: learn: 0.7413857	test: 0.6475112	bestTest: 0.6475112 (4)	total: 17.1s	remaining: 17.1s
5: learn: 0.7420311	test: 0.6478313	bestTest: 0.6478313 (5)	total: 20.6s	remaining: 13.7s
6: learn: 0.7431634	test: 0.648646	bestTest: 0.648646 (6)	total: 24.1s	remaining: 10.3s
7: learn: 0.7437214	test: 0.6488279	bestTest: 0.6488279 (7)	total: 27.6s	remaining: 6.91s
8: learn: 0.7444034	test: 0.6491763	bestTest: 0.6491763 (8)	total: 31.2s	remaining: 3.46s
9: learn: 0.7447911	test: 0.6493905	bestTest: 0.6493905 (9)	total: 34.8s	remaining: 0us

bestTest = 0.6493904927
bestIteration = 9


in model: gbdt  k-fold: 3

0: learn: 0.725582	test: 0.635956	bestTest: 0.635956 (0)	total: 3.35s	remaining: 30.2s
1: learn: 0.7356949	test: 0.6439026	bestTest: 0.6439026 (1)	total: 6.71s	remaining: 26.8s
2: learn: 0.7379437	test: 0.645681	bestTest: 0.645681 (2)	total: 10.1s	remaining: 23.7s
3: learn: 0.7399527	test: 0.6466822	bestTest: 0.6466822 (3)	total: 13.6s	remaining: 20.4s
4: learn: 0.7413422	test: 0.6475173	bestTest: 0.6475173 (4)	total: 17.1s	remaining: 17.1s
5: learn: 0.7422954	test: 0.6476618	bestTest: 0.6476618 (5)	total: 20.7s	remaining: 13.8s
6: learn: 0.7431293	test: 0.6480226	bestTest: 0.6480226 (6)	total: 24.3s	remaining: 10.4s
7: learn: 0.7437648	test: 0.6482811	bestTest: 0.6482811 (7)	total: 27.7s	remaining: 6.93s
8: learn: 0.7441534	test: 0.6484609	bestTest: 0.6484609 (8)	total: 31.3s	remaining: 3.48s
9: learn: 0.7448425	test: 0.649126	bestTest: 0.649126 (9)	total: 34.9s	remaining: 0us

bestTest = 0.6491260254
bestIteration = 9


in model: gbdt  k-fold: 4

0: learn: 0.7241155	test: 0.64003	bestTest: 0.64003 (0)	total: 3.45s	remaining: 31s
1: learn: 0.7350893	test: 0.644986	bestTest: 0.644986 (1)	total: 6.97s	remaining: 27.9s
2: learn: 0.7384792	test: 0.6460612	bestTest: 0.6460612 (2)	total: 10.5s	remaining: 24.4s
3: learn: 0.7407098	test: 0.6473587	bestTest: 0.6473587 (3)	total: 14s	remaining: 21s
4: learn: 0.7415816	test: 0.6477075	bestTest: 0.6477075 (4)	total: 17.5s	remaining: 17.5s
5: learn: 0.7427819	test: 0.6485101	bestTest: 0.6485101 (5)	total: 21.1s	remaining: 14.1s
6: learn: 0.743249	test: 0.6486327	bestTest: 0.6486327 (6)	total: 24.7s	remaining: 10.6s
7: learn: 0.7439077	test: 0.6489517	bestTest: 0.6489517 (7)	total: 28.3s	remaining: 7.08s
8: learn: 0.7442932	test: 0.6491195	bestTest: 0.6491195 (8)	total: 31.9s	remaining: 3.54s
9: learn: 0.7449053	test: 0.6491254	bestTest: 0.6491254 (9)	total: 35.5s	remaining: 0us

bestTest = 0.6491253954
bestIteration = 9

0.999957811206

[timer]: complete in 39m 37s
done

Process finished with exit code 0
'''