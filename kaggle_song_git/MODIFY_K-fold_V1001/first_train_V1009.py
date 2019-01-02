import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split
import h2o
from sklearn.metrics import roc_auc_score
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_me_play.csv'
df = read_df(load_name)
on = [
    'msno',
    'song_id',
    # 'source_system_tab',
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
    'top2_in_song',
    # 'top3_in_song',
    # 'membership_days',
    # 'song_year_int',
    # 'ISC_top1_in_song',
    # 'ISC_top2_in_song',
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


num_boost_round = 500
early_stopping_rounds = 100
verbose_eval = 10

params = []

param1 = {
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
param2 = {
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
param3 = {
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
param4 = {
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
params.append(param1)
params.append(param2)
params.append(param3)
params.append(param4)


on1 = [
    'top3_in_song',
    'ITC_song_id_log10_1',
    'msno',
    'song_id',
    'target',
]
on2 = [
    'target',
    'artist_name',
    'song_year',
    'msno',
]
on3 = [
    'song_id',
    'target',
    'source_system_tab',
    'source_screen_name',
]
on4 = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    # 'language',
    'top3_in_song',
    'ITC_song_id_log10_1',
]
ons = []
ons.append(on1)
ons.append(on2)
ons.append(on3)
ons.append(on4)
fixed = [
    'target',
    'msno',
    'song_id',
    # 'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    # 'language',
    'top2_in_song',
    'ITC_song_id_log10_1',
    # 'ITC_msno_log10_1',
    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'FAKE_1512883008',
]

result = {}
for w in df.columns:
    print("'{}',".format(w))

work_on = [
    # 'ITC_msno',
    # 'CC11_msno',
    # 'ITC_name',
    # 'CC11_name',
    # 'ITC_song_id_log10',
    # 'ITC_song_id_log10_1',
    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    'ITC_msno_log10_1',
    # 'ITC_msno_x_1',
    # 'OinC_msno',
    # 'ITC_name_log10',
    # 'ITC_name_log10_1',
    # 'ITC_name_x_1',
    # 'OinC_name',
]


for w in work_on:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df_on = df[toto]

        # save_me = True
        save_me = False
        if save_me:
            save_df(df_on)

        dfs, val = fake_df(df_on)
        del df_on
        K = 3
        dfs = divide_df(dfs, K)
        dcs = []
        for i in range(K):
            dc = pd.DataFrame()
            dc['target'] = dfs[i]['target']
            dcs.append(dc)

        vc = pd.DataFrame()
        vc['target'] = val['target']
        v = np.zeros(shape=[len(val)])
        # print(dfs[0].head())
        for r in range(4):
            for i in range(K):
                print()
                print('in round:', r, ' block:', i)
                print()
                b = [i for i in range(K)]
                b.remove(i)
                # print(i)
                # print(b)
                c = [dfs[b[j]] for j in range(K-1)]
                # print(c)
                # for j in range(K):
                #     show_df(dfs[j])
                dt = pd.concat(c)
                # show_df(dt)
                model, cols = val_df(
                    params[r], dt, val,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
                dcs[i]['from_model'+str(r)] = model.predict(dfs[i])
                v += model.predict(val)

            vc['from_model'+str(r)] = v / K
            v = np.zeros(shape=[len(val)])

        del dfs, val, model
        for i in dcs:
            print(i.dtypes)
            print(i.head())
            print(i.describe())
        new_t = pd.concat(dcs)

        print(new_t.dtypes)
        print(new_t.head())
        print(new_t.describe())
        print(vc.dtypes)
        print(vc.head())
        print(vc.describe())
        model, cols = val_df(
            params[3], new_t, vc,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        # val = add_column(model, cols, val, 'from_model'+str(o))




print('done')