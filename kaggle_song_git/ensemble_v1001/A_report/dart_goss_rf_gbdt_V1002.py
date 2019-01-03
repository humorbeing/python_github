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

num_boost_round = 30
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
        params, dt, dfs[i],
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

num_boost_round = 30
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
        params, dt, dfs[i],
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

num_boost_round = 30
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
        params, dt, dfs[i],
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

num_boost_round = 30
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
        params, dt, dfs[i],
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
r = 'cat'
for i in range(K):
    print()
    print('in model:', r, ' k-fold:', i)
    print()
    b = [i for i in range(K)]
    b.remove(i)
    c = [dcs[b[j]] for j in range(K - 1)]
    dt = pd.concat(c)
    model, cols = cat(
        dt, dcs[i], 10, learning_rate=0.3,
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

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/ensemble_v1001/dart_goss_rf_gbdt_V1002.py
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
[1]	training's auc: 0.692728	valid_1's auc: 0.692028
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.706221	valid_1's auc: 0.705446
[3]	training's auc: 0.712369	valid_1's auc: 0.711358
[4]	training's auc: 0.7201	valid_1's auc: 0.718788
[5]	training's auc: 0.722602	valid_1's auc: 0.721122
[6]	training's auc: 0.726569	valid_1's auc: 0.724927
[7]	training's auc: 0.728732	valid_1's auc: 0.727037
[8]	training's auc: 0.729006	valid_1's auc: 0.727307
[9]	training's auc: 0.732665	valid_1's auc: 0.730846
[10]	training's auc: 0.735098	valid_1's auc: 0.7332
[11]	training's auc: 0.736827	valid_1's auc: 0.734875
[12]	training's auc: 0.737261	valid_1's auc: 0.73532
[13]	training's auc: 0.738411	valid_1's auc: 0.736227
[14]	training's auc: 0.740155	valid_1's auc: 0.737857
[15]	training's auc: 0.741789	valid_1's auc: 0.739414
[16]	training's auc: 0.74303	valid_1's auc: 0.7406
[17]	training's auc: 0.743728	valid_1's auc: 0.741064
[18]	training's auc: 0.745135	valid_1's auc: 0.742404
[19]	training's auc: 0.74646	valid_1's auc: 0.743655
[20]	training's auc: 0.74761	valid_1's auc: 0.744743
[21]	training's auc: 0.747912	valid_1's auc: 0.745037
[22]	training's auc: 0.748877	valid_1's auc: 0.74594
[23]	training's auc: 0.749826	valid_1's auc: 0.746806
[24]	training's auc: 0.750742	valid_1's auc: 0.747674
[25]	training's auc: 0.75165	valid_1's auc: 0.74848
[26]	training's auc: 0.752495	valid_1's auc: 0.749253
[27]	training's auc: 0.753419	valid_1's auc: 0.750115
[28]	training's auc: 0.753256	valid_1's auc: 0.749973
[29]	training's auc: 0.753867	valid_1's auc: 0.750489
[30]	training's auc: 0.754758	valid_1's auc: 0.751358

in model: dart  k-fold: 1

[1]	training's auc: 0.691902	valid_1's auc: 0.692058
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.705093	valid_1's auc: 0.704955
[3]	training's auc: 0.711777	valid_1's auc: 0.711442
[4]	training's auc: 0.719629	valid_1's auc: 0.718999
[5]	training's auc: 0.722199	valid_1's auc: 0.721378
[6]	training's auc: 0.726025	valid_1's auc: 0.725033
[7]	training's auc: 0.728555	valid_1's auc: 0.727424
[8]	training's auc: 0.728901	valid_1's auc: 0.727775
[9]	training's auc: 0.730885	valid_1's auc: 0.729682
[10]	training's auc: 0.733433	valid_1's auc: 0.732087
[11]	training's auc: 0.736531	valid_1's auc: 0.735072
[12]	training's auc: 0.736369	valid_1's auc: 0.734955
[13]	training's auc: 0.737455	valid_1's auc: 0.735885
[14]	training's auc: 0.739219	valid_1's auc: 0.737575
[15]	training's auc: 0.740687	valid_1's auc: 0.738968
[16]	training's auc: 0.742188	valid_1's auc: 0.740369
[17]	training's auc: 0.74296	valid_1's auc: 0.740991
[18]	training's auc: 0.744483	valid_1's auc: 0.742459
[19]	training's auc: 0.745882	valid_1's auc: 0.743731
[20]	training's auc: 0.746952	valid_1's auc: 0.744771
[21]	training's auc: 0.747235	valid_1's auc: 0.745052
[22]	training's auc: 0.748516	valid_1's auc: 0.746197
[23]	training's auc: 0.749542	valid_1's auc: 0.747186
[24]	training's auc: 0.750458	valid_1's auc: 0.748052
[25]	training's auc: 0.751401	valid_1's auc: 0.748946
[26]	training's auc: 0.752168	valid_1's auc: 0.74964
[27]	training's auc: 0.753121	valid_1's auc: 0.750549
[28]	training's auc: 0.752972	valid_1's auc: 0.750385
[29]	training's auc: 0.75356	valid_1's auc: 0.750811
[30]	training's auc: 0.754529	valid_1's auc: 0.751725

in model: goss  k-fold: 0

[1]	training's auc: 0.692779	valid_1's auc: 0.692033
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.701187	valid_1's auc: 0.700395
[3]	training's auc: 0.707355	valid_1's auc: 0.706331
[4]	training's auc: 0.712582	valid_1's auc: 0.711555
[5]	training's auc: 0.713922	valid_1's auc: 0.712778
[6]	training's auc: 0.718391	valid_1's auc: 0.717111
[7]	training's auc: 0.722054	valid_1's auc: 0.720612
[8]	training's auc: 0.725191	valid_1's auc: 0.723676
[9]	training's auc: 0.72749	valid_1's auc: 0.725852
[10]	training's auc: 0.730199	valid_1's auc: 0.728363
[11]	training's auc: 0.732154	valid_1's auc: 0.730212
[12]	training's auc: 0.733836	valid_1's auc: 0.731813
[13]	training's auc: 0.734271	valid_1's auc: 0.732105
[14]	training's auc: 0.735799	valid_1's auc: 0.733569
[15]	training's auc: 0.737244	valid_1's auc: 0.734986
[16]	training's auc: 0.738512	valid_1's auc: 0.736152
[17]	training's auc: 0.738808	valid_1's auc: 0.736232
[18]	training's auc: 0.740063	valid_1's auc: 0.737462
[19]	training's auc: 0.741236	valid_1's auc: 0.738571
[20]	training's auc: 0.742469	valid_1's auc: 0.739696
[21]	training's auc: 0.743595	valid_1's auc: 0.740764
[22]	training's auc: 0.74474	valid_1's auc: 0.741833
[23]	training's auc: 0.745758	valid_1's auc: 0.742789
[24]	training's auc: 0.746705	valid_1's auc: 0.743661
[25]	training's auc: 0.747656	valid_1's auc: 0.74454
[26]	training's auc: 0.748533	valid_1's auc: 0.745349
[27]	training's auc: 0.749454	valid_1's auc: 0.746216
[28]	training's auc: 0.749962	valid_1's auc: 0.746599
[29]	training's auc: 0.750456	valid_1's auc: 0.746936
[30]	training's auc: 0.751241	valid_1's auc: 0.747667

in model: goss  k-fold: 1

[1]	training's auc: 0.691638	valid_1's auc: 0.691822
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.700787	valid_1's auc: 0.700922
[3]	training's auc: 0.707504	valid_1's auc: 0.707443
[4]	training's auc: 0.712064	valid_1's auc: 0.711816
[5]	training's auc: 0.713768	valid_1's auc: 0.713298
[6]	training's auc: 0.717376	valid_1's auc: 0.716707
[7]	training's auc: 0.721194	valid_1's auc: 0.720384
[8]	training's auc: 0.724182	valid_1's auc: 0.723287
[9]	training's auc: 0.726354	valid_1's auc: 0.72535
[10]	training's auc: 0.729033	valid_1's auc: 0.727846
[11]	training's auc: 0.730847	valid_1's auc: 0.729602
[12]	training's auc: 0.73259	valid_1's auc: 0.73127
[13]	training's auc: 0.733112	valid_1's auc: 0.731629
[14]	training's auc: 0.734745	valid_1's auc: 0.73321
[15]	training's auc: 0.73635	valid_1's auc: 0.734681
[16]	training's auc: 0.737877	valid_1's auc: 0.736138
[17]	training's auc: 0.738372	valid_1's auc: 0.736374
[18]	training's auc: 0.739676	valid_1's auc: 0.737637
[19]	training's auc: 0.740956	valid_1's auc: 0.738891
[20]	training's auc: 0.742279	valid_1's auc: 0.740124
[21]	training's auc: 0.743371	valid_1's auc: 0.741183
[22]	training's auc: 0.744438	valid_1's auc: 0.742185
[23]	training's auc: 0.745461	valid_1's auc: 0.743134
[24]	training's auc: 0.746394	valid_1's auc: 0.744023
[25]	training's auc: 0.747421	valid_1's auc: 0.74501
[26]	training's auc: 0.748336	valid_1's auc: 0.745875
[27]	training's auc: 0.749274	valid_1's auc: 0.746783
[28]	training's auc: 0.749802	valid_1's auc: 0.747209
[29]	training's auc: 0.75011	valid_1's auc: 0.74734
[30]	training's auc: 0.751017	valid_1's auc: 0.748159

in model: rf  k-fold: 0

[1]	training's auc: 0.726686	valid_1's auc: 0.722648
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.726686	valid_1's auc: 0.722648
[3]	training's auc: 0.731284	valid_1's auc: 0.727399
[4]	training's auc: 0.73558	valid_1's auc: 0.73154
[5]	training's auc: 0.736825	valid_1's auc: 0.732365
[6]	training's auc: 0.736304	valid_1's auc: 0.731997
[7]	training's auc: 0.73725	valid_1's auc: 0.732932
[8]	training's auc: 0.73787	valid_1's auc: 0.733562
[9]	training's auc: 0.737582	valid_1's auc: 0.733379
[10]	training's auc: 0.738167	valid_1's auc: 0.733884
[11]	training's auc: 0.738498	valid_1's auc: 0.734208
[12]	training's auc: 0.738768	valid_1's auc: 0.734447
[13]	training's auc: 0.739042	valid_1's auc: 0.734484
[14]	training's auc: 0.739282	valid_1's auc: 0.734669
[15]	training's auc: 0.739335	valid_1's auc: 0.734741
[16]	training's auc: 0.739031	valid_1's auc: 0.734545
[17]	training's auc: 0.738708	valid_1's auc: 0.734073
[18]	training's auc: 0.739019	valid_1's auc: 0.734354
[19]	training's auc: 0.739059	valid_1's auc: 0.734391
[20]	training's auc: 0.738999	valid_1's auc: 0.734403
[21]	training's auc: 0.738861	valid_1's auc: 0.734314
[22]	training's auc: 0.738686	valid_1's auc: 0.734184
[23]	training's auc: 0.738445	valid_1's auc: 0.734009
[24]	training's auc: 0.738268	valid_1's auc: 0.733874
[25]	training's auc: 0.738089	valid_1's auc: 0.733722
[26]	training's auc: 0.738192	valid_1's auc: 0.733832
[27]	training's auc: 0.738333	valid_1's auc: 0.733962
[28]	training's auc: 0.738407	valid_1's auc: 0.733981
[29]	training's auc: 0.738414	valid_1's auc: 0.733895
[30]	training's auc: 0.738251	valid_1's auc: 0.733772

in model: rf  k-fold: 1

[1]	training's auc: 0.728347	valid_1's auc: 0.724658
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.728347	valid_1's auc: 0.724658
[3]	training's auc: 0.732777	valid_1's auc: 0.729273
[4]	training's auc: 0.735427	valid_1's auc: 0.731951
[5]	training's auc: 0.736688	valid_1's auc: 0.732786
[6]	training's auc: 0.735861	valid_1's auc: 0.732066
[7]	training's auc: 0.737503	valid_1's auc: 0.733673
[8]	training's auc: 0.73826	valid_1's auc: 0.734458
[9]	training's auc: 0.737971	valid_1's auc: 0.734325
[10]	training's auc: 0.738635	valid_1's auc: 0.73494
[11]	training's auc: 0.739085	valid_1's auc: 0.735328
[12]	training's auc: 0.739478	valid_1's auc: 0.735717
[13]	training's auc: 0.739575	valid_1's auc: 0.735636
[14]	training's auc: 0.739841	valid_1's auc: 0.735882
[15]	training's auc: 0.739914	valid_1's auc: 0.735985
[16]	training's auc: 0.739637	valid_1's auc: 0.735806
[17]	training's auc: 0.739319	valid_1's auc: 0.735394
[18]	training's auc: 0.739592	valid_1's auc: 0.735658
[19]	training's auc: 0.739662	valid_1's auc: 0.735714
[20]	training's auc: 0.739486	valid_1's auc: 0.735612
[21]	training's auc: 0.73928	valid_1's auc: 0.735454
[22]	training's auc: 0.739005	valid_1's auc: 0.735233
[23]	training's auc: 0.738815	valid_1's auc: 0.735088
[24]	training's auc: 0.738623	valid_1's auc: 0.734938
[25]	training's auc: 0.738366	valid_1's auc: 0.734709
[26]	training's auc: 0.738507	valid_1's auc: 0.734818
[27]	training's auc: 0.738703	valid_1's auc: 0.734994
[28]	training's auc: 0.738768	valid_1's auc: 0.734986
[29]	training's auc: 0.738789	valid_1's auc: 0.734929
[30]	training's auc: 0.73865	valid_1's auc: 0.734815

in model: gbdt  k-fold: 0

[1]	training's auc: 0.77279	valid_1's auc: 0.759602
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.77919	valid_1's auc: 0.764677
[3]	training's auc: 0.782645	valid_1's auc: 0.768132
[4]	training's auc: 0.784625	valid_1's auc: 0.769701
[5]	training's auc: 0.786389	valid_1's auc: 0.771143
[6]	training's auc: 0.789562	valid_1's auc: 0.774838
[7]	training's auc: 0.790412	valid_1's auc: 0.775423
[8]	training's auc: 0.791568	valid_1's auc: 0.776327
[9]	training's auc: 0.792889	valid_1's auc: 0.777918
[10]	training's auc: 0.793505	valid_1's auc: 0.77839
[11]	training's auc: 0.794899	valid_1's auc: 0.779271
[12]	training's auc: 0.795788	valid_1's auc: 0.779829
[13]	training's auc: 0.796546	valid_1's auc: 0.77967
[14]	training's auc: 0.797336	valid_1's auc: 0.780239
[15]	training's auc: 0.798151	valid_1's auc: 0.7808
[16]	training's auc: 0.799236	valid_1's auc: 0.78199
[17]	training's auc: 0.799942	valid_1's auc: 0.782494
[18]	training's auc: 0.800525	valid_1's auc: 0.782966
[19]	training's auc: 0.801572	valid_1's auc: 0.78373
[20]	training's auc: 0.802863	valid_1's auc: 0.785161
[21]	training's auc: 0.803324	valid_1's auc: 0.785411
[22]	training's auc: 0.803903	valid_1's auc: 0.78579
[23]	training's auc: 0.804971	valid_1's auc: 0.786931
[24]	training's auc: 0.805616	valid_1's auc: 0.787602
[25]	training's auc: 0.806269	valid_1's auc: 0.788013
[26]	training's auc: 0.806834	valid_1's auc: 0.788309
[27]	training's auc: 0.807542	valid_1's auc: 0.788765
[28]	training's auc: 0.807744	valid_1's auc: 0.788564
[29]	training's auc: 0.807875	valid_1's auc: 0.788329
[30]	training's auc: 0.80852	valid_1's auc: 0.788749

in model: gbdt  k-fold: 1

[1]	training's auc: 0.772648	valid_1's auc: 0.760413
Training until validation scores don't improve for 50 rounds.
[2]	training's auc: 0.781327	valid_1's auc: 0.767747
[3]	training's auc: 0.784335	valid_1's auc: 0.770614
[4]	training's auc: 0.786893	valid_1's auc: 0.77279
[5]	training's auc: 0.78824	valid_1's auc: 0.773897
[6]	training's auc: 0.791186	valid_1's auc: 0.777265
[7]	training's auc: 0.791797	valid_1's auc: 0.777639
[8]	training's auc: 0.792848	valid_1's auc: 0.778402
[9]	training's auc: 0.794189	valid_1's auc: 0.780005
[10]	training's auc: 0.795169	valid_1's auc: 0.780704
[11]	training's auc: 0.796154	valid_1's auc: 0.78132
[12]	training's auc: 0.796854	valid_1's auc: 0.78173
[13]	training's auc: 0.797345	valid_1's auc: 0.781383
[14]	training's auc: 0.797889	valid_1's auc: 0.781687
[15]	training's auc: 0.799356	valid_1's auc: 0.782941
[16]	training's auc: 0.800324	valid_1's auc: 0.783986
[17]	training's auc: 0.800924	valid_1's auc: 0.784371
[18]	training's auc: 0.801495	valid_1's auc: 0.784826
[19]	training's auc: 0.802437	valid_1's auc: 0.785583
[20]	training's auc: 0.803478	valid_1's auc: 0.786733
[21]	training's auc: 0.803965	valid_1's auc: 0.787007
[22]	training's auc: 0.804506	valid_1's auc: 0.787324
[23]	training's auc: 0.805291	valid_1's auc: 0.788072
[24]	training's auc: 0.805864	valid_1's auc: 0.788616
[25]	training's auc: 0.806436	valid_1's auc: 0.788927
[26]	training's auc: 0.807048	valid_1's auc: 0.789302
[27]	training's auc: 0.807837	valid_1's auc: 0.789813
[28]	training's auc: 0.808008	valid_1's auc: 0.78963
[29]	training's auc: 0.808093	valid_1's auc: 0.789385
[30]	training's auc: 0.808816	valid_1's auc: 0.789872

in model: cat  k-fold: 0

0: learn: 0.728051	test: 0.7200886	bestTest: 0.7200886 (0)	total: 2.33s	remaining: 21s
1: learn: 0.733373	test: 0.7246267	bestTest: 0.7246267 (1)	total: 4.62s	remaining: 18.5s
2: learn: 0.7355984	test: 0.7277207	bestTest: 0.7277207 (2)	total: 6.92s	remaining: 16.1s
3: learn: 0.7372188	test: 0.7296587	bestTest: 0.7296587 (3)	total: 9.23s	remaining: 13.8s
4: learn: 0.7389711	test: 0.7323886	bestTest: 0.7323886 (4)	total: 11.6s	remaining: 11.6s
5: learn: 0.7405934	test: 0.7323197	bestTest: 0.7323886 (4)	total: 14s	remaining: 9.3s
6: learn: 0.7417629	test: 0.7329814	bestTest: 0.7329814 (6)	total: 16.3s	remaining: 7s
7: learn: 0.7422718	test: 0.7328908	bestTest: 0.7329814 (6)	total: 18.7s	remaining: 4.68s
8: learn: 0.7429881	test: 0.7326128	bestTest: 0.7329814 (6)	total: 21.2s	remaining: 2.35s
9: learn: 0.7434361	test: 0.7327996	bestTest: 0.7329814 (6)	total: 23.6s	remaining: 0us

bestTest = 0.7329813582
bestIteration = 6


in model: cat  k-fold: 1

0: learn: 0.7247233	test: 0.7099041	bestTest: 0.7099041 (0)	total: 2.23s	remaining: 20.1s
1: learn: 0.7335864	test: 0.7155616	bestTest: 0.7155616 (1)	total: 4.48s	remaining: 17.9s
2: learn: 0.7366442	test: 0.7197813	bestTest: 0.7197813 (2)	total: 6.76s	remaining: 15.8s
3: learn: 0.7377461	test: 0.7181224	bestTest: 0.7197813 (2)	total: 9.06s	remaining: 13.6s
4: learn: 0.739092	test: 0.7219474	bestTest: 0.7219474 (4)	total: 11.4s	remaining: 11.4s
5: learn: 0.7398916	test: 0.7224859	bestTest: 0.7224859 (5)	total: 13.8s	remaining: 9.18s
6: learn: 0.7404926	test: 0.7228219	bestTest: 0.7228219 (6)	total: 16.2s	remaining: 6.94s
7: learn: 0.7415526	test: 0.7235978	bestTest: 0.7235978 (7)	total: 18.6s	remaining: 4.64s
8: learn: 0.7428092	test: 0.7237539	bestTest: 0.7237539 (8)	total: 21s	remaining: 2.33s
9: learn: 0.743448	test: 0.7238338	bestTest: 0.7238338 (9)	total: 23.3s	remaining: 0us

bestTest = 0.7238338286
bestIteration = 9

0.647683629723

[timer]: complete in 9m 11s
done

Process finished with exit code 0
'''