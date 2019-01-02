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



def deep_1(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'deep_1'

    features = on_top2
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

        # features = list(train_hf.columns)
        features.remove('target')
        print('- ' * 10)
        for c in features:
            print("'{}',".format(c))
        print('- ' * 10)
        model = H2ODeepLearningEstimator(hidden=[200,200], epochs=500)
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
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def deep_2(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'deep_2'

    features = on_top2
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

        # features = list(train_hf.columns)
        features.remove('target')
        print('- ' * 10)
        for c in features:
            print("'{}',".format(c))
        print('- ' * 10)
        model = H2ODeepLearningEstimator(hidden=[128,128,128,128,128], epochs=500)
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
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def deep_3(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'deep_3'

    features = on_top2
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

        # features = list(train_hf.columns)
        features.remove('target')
        print('- ' * 10)
        for c in features:
            print("'{}',".format(c))
        print('- ' * 10)
        model = H2ODeepLearningEstimator(hidden=[128,256,256,32], epochs=500)
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
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r


def deep_4(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'deep_4'

    features = on_top2
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

        # features = list(train_hf.columns)
        features.remove('target')
        print('- ' * 10)
        for c in features:
            print("'{}',".format(c))
        print('- ' * 10)
        model = H2ODeepLearningEstimator(hidden=[32,32,32,32,32,32,32,32,32,32], epochs=500)
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
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)

    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r