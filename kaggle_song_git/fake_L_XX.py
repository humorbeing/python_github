import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from sklearn import linear_model

on_top2 = [
    'msno',
    'song_id',
    # 'source_screen_name',
    # 'source_type',
    'target',
    # 'artist_name',
    # 'song_year',
    # 'ITC_song_id_log10_1',
    # 'ITC_msno_log10_1',
    # ------------------
    # 'top2_in_song',
    'language',
    'top3_in_song',

    # ------------------
    # 'source_system_tab',
    'ITC_source_system_tab_log10_1',
    'ISC_song_country_ln',

    # ------------------
    'membership_days',
    'ISC_song_year',
    'OinC_language',
]

def Ldrt_top2_1(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Ldrt_XX_1'

    params = {
        'boosting': 'dart',

        'learning_rate': 0.3,
        'num_leaves': 100,
        'max_depth': 10,

        'lambda_l1': 0,
        'lambda_l2': 0,
        'max_bin': 255,

        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Ldrt_top2_2(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Ldrt_XX_2'

    params = {
        'boosting': 'dart',

        'learning_rate': 0.9,
        'num_leaves': 50,
        'max_depth': 5,

        'lambda_l1': 0.1,
        'lambda_l2': 0,
        'max_bin': 15,

        'bagging_fraction': 0.5,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lgos_top2_1(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lgos_XX_1'

    params = {
        'boosting': 'goss',

        'learning_rate': 0.3,
        'num_leaves': 31,
        'max_depth': 9,

        'lambda_l1': 0.2,
        'lambda_l2': 0,
        'max_bin': 255,

        'bagging_fraction': 1,
        'bagging_freq': 0,
        'bagging_seed': 2,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lgos_top2_2(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lgos_XX_2'

    params = {
        'boosting': 'goss',

        'learning_rate': 0.03,
        'num_leaves': 100,
        'max_depth': 9,

        'lambda_l1': 0,
        'lambda_l2': 0.1,
        'max_bin': 63,

        'bagging_fraction': 1,
        'bagging_freq': 0,
        'bagging_seed': 2,
        'feature_fraction': 0.7,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lrf_top2_1(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lrf_XX_1'

    params = {
        'boosting': 'rf',

        'learning_rate': 0.3,
        'num_leaves': 750,
        'max_depth': 12,

        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'max_bin': 63,

        'bagging_fraction': 0.5,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.5,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lrf_top2_2(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lrf_XX_2'

    params = {
        'boosting': 'rf',

        'learning_rate': 0.09,
        'num_leaves': 511,
        'max_depth': 30,

        'lambda_l1': 0.1,
        'lambda_l2': 0,
        'max_bin': 127,

        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lgbt_top2_1(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lgbt_XX_1'

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

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def Lgbt_top2_2(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'Lgbt_XX_2'

    params = {
        'boosting': 'gbdt',

        'learning_rate': 0.32,
        'num_leaves': 127,
        'max_depth': -1,

        'lambda_l1': 0,
        'lambda_l2': 0.2,
        'max_bin': 63,

        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'bagging_seed': 2,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 2,
    }

    num_boost_round = 1500
    early_stopping_rounds = 50
    verbose_eval = 10
    v = np.zeros(shape=[len(test)])
    for i in range(K):
        print()
        print('in model:', r, ' k-fold:', i + 1, '/', K)
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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


