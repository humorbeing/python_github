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
<<<<<<< HEAD
# import h2o
#
# from h2o.estimators.random_forest import H2ORandomForestEstimator
# from h2o.estimators.gbm import H2OGradientBoostingEstimator
# from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# from h2o.estimators.glm import H2OGeneralizedLinearEstimator
=======
import h2o

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cross_decomposition import PLSRegression

from sklearn.kernel_ridge import KernelRidge


>>>>>>> 99c3ac2fe98440ca40cfb3df8a0deea5e5bdee92

on_top2 = [
    'target',
    'CatC_top2_1',
    'CatR_top2_1',
    'CatC_top2_2',
    'CatR_top2_2',
    'CatC_XX_1',
    'CatR_XX_1',
    'CatC_XX_2',
    'CatR_XX_2',
    'Lgos_all_1',
    'Ldrt_all_2',
    'Lrf_all_2',
    'Lgbt_all_2',
    'Lgos_top2_1',
    'Lrf_top2_1',
    'Ldrt_top2_2',
    'Lgos_top2_2',
    'Lrf_top2_2',
    'Lgbt_top2_2',
    'Lgos_XX_1',
    'Lrf_XX_1',
    'Ldrt_XX_2',
    'Lgos_XX_2',
    'Lrf_XX_2',
    'Lgbt_XX_2',
    'Ldrt_top2_1',
    'Lgbt_top2_1',
]


def logi_1(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'rogi_lvl2_1'

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
        model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.SGDClassifier()

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


def sgd(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'sgd_lvl2'

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
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LogisticRegression(C=1e5)
        model = linear_model.SGDClassifier(loss='log')

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


from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV



def GaussianNB(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'GaussianNB_lvl2'

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
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
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

from sklearn.ensemble import RandomForestClassifier

def CV(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'CV_lvl2'

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
        from sklearn.naive_bayes import GaussianNB
        from sklearn.calibration import CalibratedClassifierCV
        clf = GaussianNB()
        clf.fit(X, Y)  # GaussianNB itself does not support sample-weights

        model = CalibratedClassifierCV(clf, cv=2, method='isotonic')
        model.fit(X, Y)

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




def RF(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'RF_lvl2'

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
        model = RandomForestClassifier(max_depth=2, random_state=0)
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





#  NN
from sklearn.neural_network import MLPClassifier


def nn_1(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'nn_lvl2_1'

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

        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(5, 2), random_state=1)
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

def nn_2(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'nn_lvl2_2'

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

        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(10, 20, 30), random_state=1)
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


def nn_3(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'nn_lvl2_3'

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

        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(100, 50, 20, 5, 20), random_state=1)
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