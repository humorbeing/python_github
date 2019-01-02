import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cross_decomposition import PLSRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier





def LogisticRegression_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'LogisticRegression'

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
        print('- ' * 10)
        for c in cols:
            print("'{}',".format(c))
        print('- ' * 10)
        Y = dt['target']
        del dt
        model = linear_model.LogisticRegression(C=1e5)

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
def SGDClassifier_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'SGDClassifier'

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
def GaussianNB_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'GaussianNB'

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
        # dt = dt[on_top2]
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
def CV_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'CV'

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
        # dt = dt[on_top2]
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
def RF_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'RandomForest'

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
        # dt = dt[on_top2]
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
def Neural_net_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):
    r = 'Neural_net'

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
        # dt = dt[on_top2]
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

        model = MLPClassifier(solver='lbfgs', alpha=0.01,
                              hidden_layer_sizes=(4, 4), random_state=1)
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
def Dart_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'DART'

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
    early_stopping_rounds = 20
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
            params, dt, dfs[i],
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
def GOSS_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'GOSS'

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
    early_stopping_rounds = 20
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
            params, dt, dfs[i],
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
def RF_LIGHT_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'LIGHT_RF'

    params = {
        'boosting': 'rf',

        'learning_rate': 0.3,
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
    early_stopping_rounds = 15
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
            params, dt, dfs[i],
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
def LGBT_NODE(
        K, dfs, dfs_collector, test,
        test_collector
):

    r = 'LIGHTgbm'

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
    early_stopping_rounds = 20
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
            params, dt, dfs[i],
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
