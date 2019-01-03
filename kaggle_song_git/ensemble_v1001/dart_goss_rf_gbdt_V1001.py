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