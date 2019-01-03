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

num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

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

num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

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

num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

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

num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

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
        dt, vc, 30, learning_rate=0.3,
        depth=10
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/ensemble_v1001/dart_goss_rf_gbdt_V1001.py
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
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.735525	valid_1's auc: 0.647648
[20]	training's auc: 0.747764	valid_1's auc: 0.654294
[30]	training's auc: 0.755039	valid_1's auc: 0.656524
[40]	training's auc: 0.758548	valid_1's auc: 0.658606
[50]	training's auc: 0.761938	valid_1's auc: 0.659607
[60]	training's auc: 0.76568	valid_1's auc: 0.661352
[70]	training's auc: 0.768585	valid_1's auc: 0.662683
[80]	training's auc: 0.76944	valid_1's auc: 0.663516
[90]	training's auc: 0.769918	valid_1's auc: 0.663393
[100]	training's auc: 0.77252	valid_1's auc: 0.664566
[110]	training's auc: 0.772112	valid_1's auc: 0.663997
[120]	training's auc: 0.773951	valid_1's auc: 0.664527
[130]	training's auc: 0.775489	valid_1's auc: 0.665583
[140]	training's auc: 0.777395	valid_1's auc: 0.665922
[150]	training's auc: 0.77817	valid_1's auc: 0.666371
[160]	training's auc: 0.777876	valid_1's auc: 0.666022
[170]	training's auc: 0.780077	valid_1's auc: 0.666911
[180]	training's auc: 0.78099	valid_1's auc: 0.667128
[190]	training's auc: 0.782001	valid_1's auc: 0.667845
[200]	training's auc: 0.782993	valid_1's auc: 0.668353
[210]	training's auc: 0.783129	valid_1's auc: 0.668464
[220]	training's auc: 0.784504	valid_1's auc: 0.66868
[230]	training's auc: 0.784875	valid_1's auc: 0.668713
[240]	training's auc: 0.785922	valid_1's auc: 0.669339
[250]	training's auc: 0.786441	valid_1's auc: 0.669468
[260]	training's auc: 0.786997	valid_1's auc: 0.66977
[270]	training's auc: 0.788173	valid_1's auc: 0.670251
[280]	training's auc: 0.788827	valid_1's auc: 0.670393
[290]	training's auc: 0.789476	valid_1's auc: 0.670461
[300]	training's auc: 0.78983	valid_1's auc: 0.6705
[310]	training's auc: 0.790067	valid_1's auc: 0.670762
[320]	training's auc: 0.790617	valid_1's auc: 0.67116
[330]	training's auc: 0.791084	valid_1's auc: 0.671119
[340]	training's auc: 0.79104	valid_1's auc: 0.67142
[350]	training's auc: 0.791696	valid_1's auc: 0.67146
[360]	training's auc: 0.792341	valid_1's auc: 0.67169
[370]	training's auc: 0.792656	valid_1's auc: 0.671838
[380]	training's auc: 0.793203	valid_1's auc: 0.671633
[390]	training's auc: 0.794137	valid_1's auc: 0.672163
[400]	training's auc: 0.794839	valid_1's auc: 0.671941
[410]	training's auc: 0.795373	valid_1's auc: 0.671944
[420]	training's auc: 0.795658	valid_1's auc: 0.671896
[430]	training's auc: 0.796059	valid_1's auc: 0.672055
Early stopping, best iteration is:
[388]	training's auc: 0.793875	valid_1's auc: 0.672195

in model: dart  k-fold: 1

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.735944	valid_1's auc: 0.648658
[20]	training's auc: 0.748578	valid_1's auc: 0.654645
[30]	training's auc: 0.755794	valid_1's auc: 0.657406
[40]	training's auc: 0.759574	valid_1's auc: 0.659513
[50]	training's auc: 0.762691	valid_1's auc: 0.660656
[60]	training's auc: 0.766053	valid_1's auc: 0.661661
[70]	training's auc: 0.769191	valid_1's auc: 0.663062
[80]	training's auc: 0.769801	valid_1's auc: 0.663371
[90]	training's auc: 0.769989	valid_1's auc: 0.663433
[100]	training's auc: 0.771823	valid_1's auc: 0.664134
[110]	training's auc: 0.771304	valid_1's auc: 0.664
[120]	training's auc: 0.774386	valid_1's auc: 0.665504
[130]	training's auc: 0.775165	valid_1's auc: 0.666008
[140]	training's auc: 0.777496	valid_1's auc: 0.666781
[150]	training's auc: 0.778025	valid_1's auc: 0.666643
[160]	training's auc: 0.777558	valid_1's auc: 0.666647
[170]	training's auc: 0.779483	valid_1's auc: 0.66759
[180]	training's auc: 0.780122	valid_1's auc: 0.667989
[190]	training's auc: 0.781301	valid_1's auc: 0.66862
[200]	training's auc: 0.78235	valid_1's auc: 0.668806
[210]	training's auc: 0.782319	valid_1's auc: 0.668534
[220]	training's auc: 0.783711	valid_1's auc: 0.669136
[230]	training's auc: 0.784	valid_1's auc: 0.669624
[240]	training's auc: 0.785307	valid_1's auc: 0.669773
[250]	training's auc: 0.785719	valid_1's auc: 0.669715
[260]	training's auc: 0.786232	valid_1's auc: 0.669951
[270]	training's auc: 0.787346	valid_1's auc: 0.670012
[280]	training's auc: 0.788123	valid_1's auc: 0.670115
[290]	training's auc: 0.789158	valid_1's auc: 0.67065
[300]	training's auc: 0.789553	valid_1's auc: 0.670629
[310]	training's auc: 0.789732	valid_1's auc: 0.670407
[320]	training's auc: 0.790493	valid_1's auc: 0.67073
[330]	training's auc: 0.79071	valid_1's auc: 0.670873
[340]	training's auc: 0.790798	valid_1's auc: 0.670979
[350]	training's auc: 0.791532	valid_1's auc: 0.671246
[360]	training's auc: 0.792198	valid_1's auc: 0.671371
[370]	training's auc: 0.792362	valid_1's auc: 0.671703
[380]	training's auc: 0.792934	valid_1's auc: 0.671976
[390]	training's auc: 0.793182	valid_1's auc: 0.671689
[400]	training's auc: 0.793996	valid_1's auc: 0.671916
[410]	training's auc: 0.794648	valid_1's auc: 0.671975
[420]	training's auc: 0.794711	valid_1's auc: 0.671989
[430]	training's auc: 0.795182	valid_1's auc: 0.672132
[440]	training's auc: 0.795517	valid_1's auc: 0.672254
[450]	training's auc: 0.795834	valid_1's auc: 0.672262
[460]	training's auc: 0.795929	valid_1's auc: 0.672488
[470]	training's auc: 0.796295	valid_1's auc: 0.672845
[480]	training's auc: 0.796755	valid_1's auc: 0.672894
[490]	training's auc: 0.79656	valid_1's auc: 0.672914
[500]	training's auc: 0.796673	valid_1's auc: 0.673011

in model: goss  k-fold: 0

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.730554	valid_1's auc: 0.645297
[20]	training's auc: 0.743367	valid_1's auc: 0.651335
[30]	training's auc: 0.751855	valid_1's auc: 0.655114
[40]	training's auc: 0.756835	valid_1's auc: 0.657121
[50]	training's auc: 0.762014	valid_1's auc: 0.658642
[60]	training's auc: 0.766008	valid_1's auc: 0.660267
[70]	training's auc: 0.768796	valid_1's auc: 0.661361
[80]	training's auc: 0.770302	valid_1's auc: 0.661574
[90]	training's auc: 0.772122	valid_1's auc: 0.662242
[100]	training's auc: 0.773465	valid_1's auc: 0.662457
[110]	training's auc: 0.774717	valid_1's auc: 0.663227
[120]	training's auc: 0.775787	valid_1's auc: 0.663502
[130]	training's auc: 0.776441	valid_1's auc: 0.663768
[140]	training's auc: 0.777304	valid_1's auc: 0.663986
[150]	training's auc: 0.777848	valid_1's auc: 0.664254
[160]	training's auc: 0.778398	valid_1's auc: 0.664456
[170]	training's auc: 0.778825	valid_1's auc: 0.664281
[180]	training's auc: 0.779278	valid_1's auc: 0.664608
[190]	training's auc: 0.779593	valid_1's auc: 0.66468
[200]	training's auc: 0.779926	valid_1's auc: 0.664672
[210]	training's auc: 0.780288	valid_1's auc: 0.664723
[220]	training's auc: 0.780521	valid_1's auc: 0.664587
[230]	training's auc: 0.780656	valid_1's auc: 0.664707
[240]	training's auc: 0.780866	valid_1's auc: 0.664448
[250]	training's auc: 0.780984	valid_1's auc: 0.664444
[260]	training's auc: 0.781142	valid_1's auc: 0.664601
[270]	training's auc: 0.781388	valid_1's auc: 0.664393
Early stopping, best iteration is:
[228]	training's auc: 0.78069	valid_1's auc: 0.664852

in model: goss  k-fold: 1

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.730243	valid_1's auc: 0.645197
[20]	training's auc: 0.742822	valid_1's auc: 0.650914
[30]	training's auc: 0.751282	valid_1's auc: 0.654953
[40]	training's auc: 0.756302	valid_1's auc: 0.657114
[50]	training's auc: 0.761495	valid_1's auc: 0.659055
[60]	training's auc: 0.765372	valid_1's auc: 0.660905
[70]	training's auc: 0.768295	valid_1's auc: 0.661996
[80]	training's auc: 0.769451	valid_1's auc: 0.662434
[90]	training's auc: 0.771277	valid_1's auc: 0.662829
[100]	training's auc: 0.772804	valid_1's auc: 0.66354
[110]	training's auc: 0.773879	valid_1's auc: 0.66364
[120]	training's auc: 0.775093	valid_1's auc: 0.6638
[130]	training's auc: 0.775845	valid_1's auc: 0.663984
[140]	training's auc: 0.776694	valid_1's auc: 0.664121
[150]	training's auc: 0.777415	valid_1's auc: 0.664332
[160]	training's auc: 0.778012	valid_1's auc: 0.664954
[170]	training's auc: 0.778295	valid_1's auc: 0.664969
[180]	training's auc: 0.778737	valid_1's auc: 0.665137
[190]	training's auc: 0.779002	valid_1's auc: 0.664889
[200]	training's auc: 0.779295	valid_1's auc: 0.664898
[210]	training's auc: 0.779577	valid_1's auc: 0.665045
[220]	training's auc: 0.779857	valid_1's auc: 0.66496
Early stopping, best iteration is:
[175]	training's auc: 0.778499	valid_1's auc: 0.665299

in model: rf  k-fold: 0

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.737769	valid_1's auc: 0.645286
[20]	training's auc: 0.73882	valid_1's auc: 0.6461
[30]	training's auc: 0.73821	valid_1's auc: 0.645591
[40]	training's auc: 0.7388	valid_1's auc: 0.646367
[50]	training's auc: 0.739438	valid_1's auc: 0.646539
[60]	training's auc: 0.739756	valid_1's auc: 0.646647
[70]	training's auc: 0.739873	valid_1's auc: 0.646757
[80]	training's auc: 0.739538	valid_1's auc: 0.64684
[90]	training's auc: 0.739407	valid_1's auc: 0.646755
[100]	training's auc: 0.739401	valid_1's auc: 0.646707
[110]	training's auc: 0.739414	valid_1's auc: 0.646733
[120]	training's auc: 0.739503	valid_1's auc: 0.646768
Early stopping, best iteration is:
[75]	training's auc: 0.739933	valid_1's auc: 0.646798

in model: rf  k-fold: 1

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.73872	valid_1's auc: 0.647069
[20]	training's auc: 0.739658	valid_1's auc: 0.648195
[30]	training's auc: 0.738806	valid_1's auc: 0.647412
[40]	training's auc: 0.739357	valid_1's auc: 0.64819
[50]	training's auc: 0.740034	valid_1's auc: 0.648505
[60]	training's auc: 0.74048	valid_1's auc: 0.648644
[70]	training's auc: 0.74052	valid_1's auc: 0.648675
[80]	training's auc: 0.740026	valid_1's auc: 0.648726
[90]	training's auc: 0.739852	valid_1's auc: 0.64856
[100]	training's auc: 0.73984	valid_1's auc: 0.648449
[110]	training's auc: 0.739874	valid_1's auc: 0.648516
Early stopping, best iteration is:
[69]	training's auc: 0.740576	valid_1's auc: 0.648746

in model: gbdt  k-fold: 0

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.79379	valid_1's auc: 0.662714
[20]	training's auc: 0.80245	valid_1's auc: 0.665536
[30]	training's auc: 0.808503	valid_1's auc: 0.667266
[40]	training's auc: 0.81311	valid_1's auc: 0.668411
[50]	training's auc: 0.818968	valid_1's auc: 0.67028
[60]	training's auc: 0.82412	valid_1's auc: 0.671916
[70]	training's auc: 0.828317	valid_1's auc: 0.673267
[80]	training's auc: 0.831068	valid_1's auc: 0.674033
[90]	training's auc: 0.833762	valid_1's auc: 0.675164
[100]	training's auc: 0.836518	valid_1's auc: 0.676194
[110]	training's auc: 0.838735	valid_1's auc: 0.677057
[120]	training's auc: 0.840448	valid_1's auc: 0.677821
[130]	training's auc: 0.841812	valid_1's auc: 0.678444
[140]	training's auc: 0.843006	valid_1's auc: 0.678915
[150]	training's auc: 0.843977	valid_1's auc: 0.679392
[160]	training's auc: 0.844995	valid_1's auc: 0.679803
[170]	training's auc: 0.845966	valid_1's auc: 0.680122
[180]	training's auc: 0.846744	valid_1's auc: 0.68034
[190]	training's auc: 0.847666	valid_1's auc: 0.680625
[200]	training's auc: 0.848463	valid_1's auc: 0.680787
[210]	training's auc: 0.84931	valid_1's auc: 0.680979
[220]	training's auc: 0.849985	valid_1's auc: 0.681086
[230]	training's auc: 0.850771	valid_1's auc: 0.681253
[240]	training's auc: 0.851474	valid_1's auc: 0.681401
[250]	training's auc: 0.852229	valid_1's auc: 0.68154
[260]	training's auc: 0.852943	valid_1's auc: 0.681652
[270]	training's auc: 0.853677	valid_1's auc: 0.681804
[280]	training's auc: 0.854315	valid_1's auc: 0.681899
[290]	training's auc: 0.855011	valid_1's auc: 0.682025
[300]	training's auc: 0.855618	valid_1's auc: 0.682124
[310]	training's auc: 0.856197	valid_1's auc: 0.68218
[320]	training's auc: 0.85681	valid_1's auc: 0.682242
[330]	training's auc: 0.85745	valid_1's auc: 0.682338
[340]	training's auc: 0.858048	valid_1's auc: 0.682421
[350]	training's auc: 0.858673	valid_1's auc: 0.682521
[360]	training's auc: 0.8592	valid_1's auc: 0.682557
[370]	training's auc: 0.85974	valid_1's auc: 0.682564
[380]	training's auc: 0.86038	valid_1's auc: 0.682665
[390]	training's auc: 0.860987	valid_1's auc: 0.682769
[400]	training's auc: 0.86147	valid_1's auc: 0.682797
[410]	training's auc: 0.862103	valid_1's auc: 0.682862
[420]	training's auc: 0.862631	valid_1's auc: 0.68289
[430]	training's auc: 0.863097	valid_1's auc: 0.682905
[440]	training's auc: 0.863641	valid_1's auc: 0.682958
[450]	training's auc: 0.864115	valid_1's auc: 0.682992
[460]	training's auc: 0.864664	valid_1's auc: 0.683052
[470]	training's auc: 0.865169	valid_1's auc: 0.683129
[480]	training's auc: 0.865647	valid_1's auc: 0.683146
[490]	training's auc: 0.8661	valid_1's auc: 0.683237
[500]	training's auc: 0.86656	valid_1's auc: 0.683243

in model: gbdt  k-fold: 1

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794759	valid_1's auc: 0.663618
[20]	training's auc: 0.802706	valid_1's auc: 0.665991
[30]	training's auc: 0.80837	valid_1's auc: 0.667444
[40]	training's auc: 0.813347	valid_1's auc: 0.668731
[50]	training's auc: 0.819233	valid_1's auc: 0.670637
[60]	training's auc: 0.824058	valid_1's auc: 0.672242
[70]	training's auc: 0.828201	valid_1's auc: 0.673579
[80]	training's auc: 0.830947	valid_1's auc: 0.674268
[90]	training's auc: 0.83361	valid_1's auc: 0.675274
[100]	training's auc: 0.836366	valid_1's auc: 0.67638
[110]	training's auc: 0.838561	valid_1's auc: 0.677253
[120]	training's auc: 0.840364	valid_1's auc: 0.678028
[130]	training's auc: 0.841733	valid_1's auc: 0.678655
[140]	training's auc: 0.842902	valid_1's auc: 0.679128
[150]	training's auc: 0.843969	valid_1's auc: 0.679589
[160]	training's auc: 0.845081	valid_1's auc: 0.679993
[170]	training's auc: 0.845874	valid_1's auc: 0.680272
[180]	training's auc: 0.846754	valid_1's auc: 0.680556
[190]	training's auc: 0.847791	valid_1's auc: 0.680891
[200]	training's auc: 0.84856	valid_1's auc: 0.681096
[210]	training's auc: 0.849322	valid_1's auc: 0.681284
[220]	training's auc: 0.850034	valid_1's auc: 0.681421
[230]	training's auc: 0.850747	valid_1's auc: 0.68154
[240]	training's auc: 0.851455	valid_1's auc: 0.681669
[250]	training's auc: 0.852185	valid_1's auc: 0.681808
[260]	training's auc: 0.852831	valid_1's auc: 0.681919
[270]	training's auc: 0.853428	valid_1's auc: 0.682015
[280]	training's auc: 0.854087	valid_1's auc: 0.68211
[290]	training's auc: 0.854725	valid_1's auc: 0.68218
[300]	training's auc: 0.855339	valid_1's auc: 0.682264
[310]	training's auc: 0.856048	valid_1's auc: 0.682411
[320]	training's auc: 0.856712	valid_1's auc: 0.682515
[330]	training's auc: 0.857307	valid_1's auc: 0.68259
[340]	training's auc: 0.857882	valid_1's auc: 0.682647
[350]	training's auc: 0.858582	valid_1's auc: 0.682708
[360]	training's auc: 0.859143	valid_1's auc: 0.682767
[370]	training's auc: 0.85975	valid_1's auc: 0.682852
[380]	training's auc: 0.860282	valid_1's auc: 0.682905
[390]	training's auc: 0.860878	valid_1's auc: 0.682962
[400]	training's auc: 0.861502	valid_1's auc: 0.683038
[410]	training's auc: 0.86214	valid_1's auc: 0.683111
[420]	training's auc: 0.862692	valid_1's auc: 0.683162
[430]	training's auc: 0.863157	valid_1's auc: 0.683177
[440]	training's auc: 0.863644	valid_1's auc: 0.683217
[450]	training's auc: 0.864114	valid_1's auc: 0.683258
[460]	training's auc: 0.864633	valid_1's auc: 0.683282
[470]	training's auc: 0.865104	valid_1's auc: 0.683292
[480]	training's auc: 0.8656	valid_1's auc: 0.683366
[490]	training's auc: 0.866032	valid_1's auc: 0.683382
[500]	training's auc: 0.86653	valid_1's auc: 0.68341

in model: <function cat at 0x7fbe9b024048>  k-fold: 0

0: learn: 0.75941	test: 0.6514668	bestTest: 0.6514668 (0)	total: 2.52s	remaining: 1m 13s
1: learn: 0.7634589	test: 0.6540385	bestTest: 0.6540385 (1)	total: 5.04s	remaining: 1m 10s
2: learn: 0.7656636	test: 0.65456	bestTest: 0.65456 (2)	total: 7.67s	remaining: 1m 9s
3: learn: 0.7674821	test: 0.6555452	bestTest: 0.6555452 (3)	total: 10.2s	remaining: 1m 6s
4: learn: 0.7683282	test: 0.656605	bestTest: 0.656605 (4)	total: 12.9s	remaining: 1m 4s
5: learn: 0.7693361	test: 0.6575318	bestTest: 0.6575318 (5)	total: 15.5s	remaining: 1m 2s
6: learn: 0.7697075	test: 0.658012	bestTest: 0.658012 (6)	total: 18.2s	remaining: 60s
7: learn: 0.7703296	test: 0.6580409	bestTest: 0.6580409 (7)	total: 21s	remaining: 57.8s
8: learn: 0.7708215	test: 0.6587659	bestTest: 0.6587659 (8)	total: 23.8s	remaining: 55.5s
9: learn: 0.7712886	test: 0.65885	bestTest: 0.65885 (9)	total: 26.6s	remaining: 53.2s
10: learn: 0.771886	test: 0.6595083	bestTest: 0.6595083 (10)	total: 29.5s	remaining: 51s
11: learn: 0.7722375	test: 0.6598982	bestTest: 0.6598982 (11)	total: 32.4s	remaining: 48.6s
12: learn: 0.7724374	test: 0.6600005	bestTest: 0.6600005 (12)	total: 35.2s	remaining: 46s
13: learn: 0.772708	test: 0.6600725	bestTest: 0.6600725 (13)	total: 38.2s	remaining: 43.6s
14: learn: 0.7729715	test: 0.6602066	bestTest: 0.6602066 (14)	total: 41.1s	remaining: 41.1s
15: learn: 0.7731403	test: 0.6601716	bestTest: 0.6602066 (14)	total: 43.9s	remaining: 38.4s
16: learn: 0.7735664	test: 0.6599579	bestTest: 0.6602066 (14)	total: 46.8s	remaining: 35.8s
17: learn: 0.7737388	test: 0.6599238	bestTest: 0.6602066 (14)	total: 49.8s	remaining: 33.2s
18: learn: 0.7740126	test: 0.6601055	bestTest: 0.6602066 (14)	total: 52.7s	remaining: 30.5s
19: learn: 0.7742513	test: 0.6597893	bestTest: 0.6602066 (14)	total: 55.7s	remaining: 27.8s
20: learn: 0.7744939	test: 0.6598632	bestTest: 0.6602066 (14)	total: 58.7s	remaining: 25.1s
21: learn: 0.7746381	test: 0.6598291	bestTest: 0.6602066 (14)	total: 1m 1s	remaining: 22.4s
22: learn: 0.7747762	test: 0.6597342	bestTest: 0.6602066 (14)	total: 1m 4s	remaining: 19.6s
23: learn: 0.7749632	test: 0.6596656	bestTest: 0.6602066 (14)	total: 1m 7s	remaining: 16.9s
24: learn: 0.7750942	test: 0.6596901	bestTest: 0.6602066 (14)	total: 1m 10s	remaining: 14.1s
25: learn: 0.7752675	test: 0.6597287	bestTest: 0.6602066 (14)	total: 1m 13s	remaining: 11.3s
26: learn: 0.7753643	test: 0.6597197	bestTest: 0.6602066 (14)	total: 1m 16s	remaining: 8.48s
27: learn: 0.7754558	test: 0.6597249	bestTest: 0.6602066 (14)	total: 1m 19s	remaining: 5.66s
28: learn: 0.7755904	test: 0.6595577	bestTest: 0.6602066 (14)	total: 1m 22s	remaining: 2.84s
29: learn: 0.7757418	test: 0.659574	bestTest: 0.6602066 (14)	total: 1m 25s	remaining: 0us

bestTest = 0.6602066055
bestIteration = 14


in model: <function cat at 0x7fbe9b024048>  k-fold: 1

0: learn: 0.757675	test: 0.6535734	bestTest: 0.6535734 (0)	total: 2.5s	remaining: 1m 12s
1: learn: 0.762057	test: 0.6557566	bestTest: 0.6557566 (1)	total: 5.04s	remaining: 1m 10s
2: learn: 0.7640137	test: 0.6575534	bestTest: 0.6575534 (2)	total: 7.65s	remaining: 1m 8s
3: learn: 0.7654386	test: 0.6588114	bestTest: 0.6588114 (3)	total: 10.3s	remaining: 1m 7s
4: learn: 0.7668338	test: 0.6589516	bestTest: 0.6589516 (4)	total: 12.9s	remaining: 1m 4s
5: learn: 0.7676967	test: 0.6590478	bestTest: 0.6590478 (5)	total: 15.6s	remaining: 1m 2s
6: learn: 0.7685083	test: 0.659528	bestTest: 0.659528 (6)	total: 18.3s	remaining: 1m
7: learn: 0.7696371	test: 0.6598458	bestTest: 0.6598458 (7)	total: 21.1s	remaining: 58s
8: learn: 0.770196	test: 0.6602697	bestTest: 0.6602697 (8)	total: 23.9s	remaining: 55.8s
9: learn: 0.7707375	test: 0.6605197	bestTest: 0.6605197 (9)	total: 26.8s	remaining: 53.6s
10: learn: 0.7710124	test: 0.6605317	bestTest: 0.6605317 (10)	total: 29.6s	remaining: 51.1s
11: learn: 0.7715384	test: 0.6603086	bestTest: 0.6605317 (10)	total: 32.4s	remaining: 48.6s
12: learn: 0.7718354	test: 0.6604434	bestTest: 0.6605317 (10)	total: 35.3s	remaining: 46.1s
13: learn: 0.772101	test: 0.6603769	bestTest: 0.6605317 (10)	total: 38.1s	remaining: 43.6s
14: learn: 0.7723402	test: 0.660307	bestTest: 0.6605317 (10)	total: 41s	remaining: 41s
15: learn: 0.7725291	test: 0.6603113	bestTest: 0.6605317 (10)	total: 43.8s	remaining: 38.3s
16: learn: 0.7727778	test: 0.6601863	bestTest: 0.6605317 (10)	total: 46.6s	remaining: 35.7s
17: learn: 0.772968	test: 0.6601755	bestTest: 0.6605317 (10)	total: 49.6s	remaining: 33.1s
18: learn: 0.7731244	test: 0.6600496	bestTest: 0.6605317 (10)	total: 52.5s	remaining: 30.4s
19: learn: 0.7734059	test: 0.6598828	bestTest: 0.6605317 (10)	total: 55.3s	remaining: 27.7s
20: learn: 0.7735858	test: 0.6599572	bestTest: 0.6605317 (10)	total: 58.2s	remaining: 25s
21: learn: 0.7737354	test: 0.6598974	bestTest: 0.6605317 (10)	total: 1m 1s	remaining: 22.2s
22: learn: 0.7738405	test: 0.6598574	bestTest: 0.6605317 (10)	total: 1m 3s	remaining: 19.5s
23: learn: 0.774014	test: 0.6597517	bestTest: 0.6605317 (10)	total: 1m 6s	remaining: 16.7s
24: learn: 0.7741394	test: 0.6597641	bestTest: 0.6605317 (10)	total: 1m 9s	remaining: 14s
25: learn: 0.7742459	test: 0.6597336	bestTest: 0.6605317 (10)	total: 1m 12s	remaining: 11.2s
26: learn: 0.774344	test: 0.6596087	bestTest: 0.6605317 (10)	total: 1m 15s	remaining: 8.41s
27: learn: 0.7743979	test: 0.6595708	bestTest: 0.6605317 (10)	total: 1m 18s	remaining: 5.61s
28: learn: 0.7745141	test: 0.6596367	bestTest: 0.6605317 (10)	total: 1m 21s	remaining: 2.81s
29: learn: 0.7745865	test: 0.6596313	bestTest: 0.6605317 (10)	total: 1m 24s	remaining: 0us

bestTest = 0.6605316918
bestIteration = 10

0.660570217324

[timer]: complete in 90m 50s
done

Process finished with exit code 0
'''