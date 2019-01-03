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
    'target',
    'CatC_top2_1',
    'CatR_top2_1',
    # 'CatC_top2_2',
    # 'CatR_top2_2',
    # 'CatC_XX_1',
    # 'CatR_XX_1',
    # 'CatC_XX_2',
    # 'CatR_XX_2',
    # 'Lgos_all_1',
    # 'Ldrt_all_2',
    # 'Lrf_all_2',
    # 'Lgbt_all_2',
    # 'Lgos_top2_1',
    # 'Lrf_top2_1',
    # 'Ldrt_top2_2',
    # 'Lgos_top2_2',
    # 'Lrf_top2_2',
    # 'Lgbt_top2_2',
    # 'Lgos_XX_1',
    # 'Lrf_XX_1',
    # 'Ldrt_XX_2',
    # 'Lgos_XX_2',
    # 'Lrf_XX_2',
    # 'Lgbt_XX_2',
    # 'Ldrt_top2_1',
    # 'Lgbt_top2_1',
]

from sklearn import svm

def logi_3(
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
        dt = dt[on_top2]
        X = dt.drop('target', axis=1)
        cols = [c for c in X.columns]
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        Y = dt['target']
        del dt

        model = svm.SVC()
        model.fit(X, Y)
        dfs_collector[i][r] = model.predict_proba(dfs[i][cols])[:, 1]
        print(dfs_collector[i].head())
        print(dfs_collector[i].head().dtypes)
        v += model.predict_proba(test[cols])[:, 1]
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)
    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r
from sklearn.cross_decomposition import PLSRegression

from sklearn.kernel_ridge import KernelRidge
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
        dt = dt[on_top2]
        X = dt.drop('target', axis=1)
        cols = [c for c in X.columns]
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        Y = dt['target']
        del dt
        from sklearn.gaussian_process import GaussianProcessRegressor
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        model = KernelRidge(alpha=1.0)
        # model = linear_model.Lasso()
        # model = linear_model.SGDClassifier(loss='log')

        model.fit(X, Y)
        dfs_collector[i][r] = model.predict(dfs[i][cols])
        print(dfs_collector[i].head())
        print(dfs_collector[i].head().dtypes)
        v += model.predict(test[cols])
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)
    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r



from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def logi_2(
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
        dt = dt[on_top2]
        X = dt.drop('target', axis=1)
        cols = [c for c in X.columns]
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        Y = dt['target']
        del dt
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.Perceptron()
        # model = KNeighborsClassifier(n_neighbors=3)
        # kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
        model = PassiveAggressiveClassifier(random_state=0)
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #                       hidden_layer_sizes = (5, 2), random_state = 1)
        # model = linear_model.Lasso()

        # model = linear_model.SGDClassifier(loss='log')

        model.fit(X, Y)
        dfs_collector[i][r] = model.predict_proba(dfs[i][cols])[:, 1]
        print(dfs_collector[i].head())
        print(dfs_collector[i].head().dtypes)
        v += model.predict_proba(test[cols])[:, 1]
        print('# ' * 10)
        for show_v in range(5):
            print(v[show_v])
        print('# ' * 10)
    test_collector[r] = v / K
    print(test_collector.head())
    return dfs_collector, test_collector, r



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
        model = H2ODeepLearningEstimator(hidden=[200,200], epochs=2)
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
        model = H2ODeepLearningEstimator(hidden=[128,128,128,128,128], epochs=2)
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
        model = H2ODeepLearningEstimator(hidden=[128,256,256,32], epochs=2)
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
        model = H2ODeepLearningEstimator(hidden=[32,32,32,32,32,32,32,32,32,32], epochs=2)
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