import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn import linear_model
import h2o

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

on_top2 = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_language = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_sst_c = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    # 'language',
    'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
on_num = [
    # 'msno',
    # 'song_id',
    'source_screen_name',
    'source_type',
    'target',
    # 'artist_name',
    # 'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    'ISC_song_year',
    'OinC_language',
]


def dart_1(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'dart_1'

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
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i+1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        model, cols = val_df(
            params, dt, test,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        del dt
        dfs_collector[i][r] = model.predict(dfs[i])
        print(dfs_collector[i].head())
        # dfs_collector[i][r+'a'] = model.predict(dfs[i][cols])
        # print(dfs_collector[i].head())
        v += model.predict(test[cols])

    test_collector[r] = v / K
    return dfs_collector, test_collector, r


def gbdt_optimal_on_top2(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'gbdt_optimal_on_top2'

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

    num_boost_round = 2000
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i+1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        model, cols = val_df(
            params, dt[on_top2], test,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        del dt
        dfs_collector[i][r] = model.predict(dfs[i])
        print(dfs_collector[i].head())
        # dfs_collector[i][r+'a'] = model.predict(dfs[i][cols])
        # print(dfs_collector[i].head())
        v += model.predict(test[cols])

    test_collector[r] = v / K
    return dfs_collector, test_collector, r


def cat_1(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'cat_1'

    on = [

    ]
    iterations = 3
    learning_rate = 0.3
    depth = 6
    early_stop = 40
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        model, cols = cat(
            dt, test, iterations=iterations,
            learning_rate=learning_rate, depth=depth,
            early_stop=early_stop,
        )
        del dt
        dfs_collector[i][r] = model.predict_proba(dfs[i])[:, 1]
        print(dfs_collector[i].head())
        # dfs_collector[i][r+'a'] = model.predict_proba(dfs[i][cols])[:, 1]
        # print(dfs_collector[i].head())
        v += model.predict_proba(test[cols])[:, 1]

    test_collector[r] = v / K
    return dfs_collector, test_collector, r


def logi_1(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'rogi_1'

    on = [

    ]

    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        X = dt.drop('target', axis=1)
        cols = [c for c in X.columns]
        Y = dt['target']
        del dt
        model = linear_model.LogisticRegression(C=1e5)
        model.fit(X, Y)
        dfs_collector[i][r] = model.predict_proba(dfs[i][cols])[:, 1]
        v += model.predict_proba(test[cols])[:, 1]

    test_collector[r] = v / K
    return dfs_collector, test_collector, r


def gbm_1_R(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'gbm_1'

    on = [

    ]
    val_hf = h2o.H2OFrame(test)
    ntrees = 100
    seed = 1155
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        train_hf = h2o.H2OFrame(dt)
        del dt
        dfs_i = h2o.H2OFrame(dfs[i])

        features = list(train_hf.columns)
        features.remove('target')
        model = H2OGradientBoostingEstimator(
            model_id='gbm_manual',
            seed=seed,
            ntrees=ntrees,
            sample_rate=0.9,
            col_sample_rate=0.9
        )

        model.train(x=features,
                         y='target',
                         training_frame=train_hf)
        del train_hf
        p = model.predict(dfs_i)
        dfs_collector[i][r] = h2o.as_list(p, use_pandas=True).values
        print(dfs_collector[i].head())
        print(dfs_collector[i].head().dtypes)
        q = model.predict(val_hf)

        dd = h2o.as_list(q, use_pandas=True)
        a = dd['predict']
        a = np.array(a, dtype=pd.Series).tolist()
        # print(type(a))
        # print(a.shape)
        v += a

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def glm_1_C(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'glm_1'

    on = [

    ]
    val_hf = h2o.H2OFrame(test)
    val_hf['target'] = val_hf['target'].asfactor()

    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
        print()
        b = [i for i in range(K)]
        b.remove(i)
        c = [dfs[b[j]] for j in range(K - 1)]
        dt = pd.concat(c)
        train_hf = h2o.H2OFrame(dt)
        train_hf['target'] = train_hf['target'].asfactor()
        del dt
        dfs_i = h2o.H2OFrame(dfs[i])

        features = list(train_hf.columns)
        features.remove('target')
        model = H2OGeneralizedLinearEstimator(
            family = 'binomial',
            model_id = 'glm_default'
        )

        model.train(x=features,
                         y='target',
                         training_frame=train_hf)
        del train_hf
        p = model.predict(dfs_i)
        dfs_collector[i][r] = h2o.as_list(p, use_pandas=True)['p1']
        print(dfs_collector[i].head())
        print(dfs_collector[i].head().dtypes)
        q = model.predict(val_hf)

        dd = h2o.as_list(q, use_pandas=True)
        a = dd['p1']
        a = np.array(a, dtype=pd.Series).tolist()
        # print(type(a))
        # print(a.shape)
        v += a

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r