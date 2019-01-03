import pandas as pd
import seaborn as sns
import time
import numpy as np
import pickle
import lightgbm as lgb
import math
import gc

def train_light(
    parameters, train_set,
    num_boost_round=5000,
):


    params = parameters
    train_set = train_set.sample(frac=1)
    X_tr = train_set.drop(['target'], axis=1)
    cols = [i for i in X_tr.columns]
    Y_tr = train_set['target'].values
    del train_set
    train_set = lgb.Dataset(X_tr, Y_tr)
    del X_tr, Y_tr

    params['metric'] = 'auc'
    params['verbose'] = -1
    params['objective'] = 'binary'

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
    )
    del train_set
    return model, cols


def show_mo(model):
    print('model:')
    b = model.best_score['valid_1']['auc']
    print('best score:', model.best_score['valid_1']['auc'])
    print('best iteration:', model.best_iteration)
    print()
    ns = model.feature_name()
    ims = model.feature_importance()
    for i in range(len(ns)):
        print(ns[i].rjust(20), ':', ims[i])
    return b

def save_df(df, name='save_me', save_to='../saves/'):
    print(' SAVE ' * 5)
    print(' SAVE ' * 5)
    print(' SAVE ' * 5)

    print('saving df:')
    d = df.dtypes.to_dict()
    print('dtypes of df:')
    print('>' * 20)
    print(df.dtypes)
    print('number of columns:', len(df.columns))
    print('number of data:', len(df))
    print('<' * 20)
    df.to_csv(save_to + name + '.csv', index=False)
    pickle.dump(d, open(save_to + name + '_dict.save', "wb"))

    print('saving DONE.')
def val_df(parameters, train_set, val_set,
            num_boost_round=5000,
            early_stopping_rounds=50,
            verbose_eval=10,
           learning_rate=False
            ):


    # for col in cols:
    #     if train_set[col].dtype == object:
    #         train_set[col] = train_set[col].astype('category')
    # for col in cols:
    #     if val_set[col].dtype == object:
    #         val_set[col] = val_set[col].astype('category')

    params = parameters
    train_set = train_set.sample(frac=1)
    X_tr = train_set.drop(['target'], axis=1)
    cols = [i for i in X_tr.columns]
    Y_tr = train_set['target'].values

    X_val = val_set.drop(['target'], axis=1)
    X_val = X_val[cols]
    Y_val = val_set['target'].values

    del train_set, val_set

    train_set = lgb.Dataset(X_tr, Y_tr)
    val_set = lgb.Dataset(X_val, Y_val)
    del X_tr, Y_tr, X_val, Y_val

    params['metric'] = 'auc'
    params['verbose'] = -1
    params['objective'] = 'binary'

    def lll(x):
        print('set it yourself.')
        return 5

    if learning_rate:

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            valid_sets=[train_set, val_set],
            verbose_eval=verbose_eval,
            learning_rate=lll,
        )
    else:
        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            valid_sets=[train_set, val_set],
            verbose_eval=verbose_eval,
        )

    del train_set, val_set
    return model, cols
def fake_df(df, size=0.76):
    length = len(df)
    train_set = df.head(int(length * size))
    val = df.drop(train_set.index)

    return train_set, val
def divide_df(df, K):

    dfs = []
    if K > 1:
        for i in range(K):
            dfs.append(df[df.index % K == i])
        del df
    else:
        dfs = [df]
        del df

    return dfs
def read_df(load_name, read_from='../saves/'):
    load_name = load_name[:-4]
    dt = pickle.load(open(read_from + load_name + '_dict.save', "rb"))
    df = pd.read_csv(read_from + load_name + ".csv", dtype=dt)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')
    del dt

    return df

def show_df(df, detail=False):
    print()
    print('>' * 20)
    print('>' * 20)
    print('dtypes of df:')

    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))
    print()
    for w in df.columns:
        print("'{}',".format(w))
    print()
    if detail:
        for on in df.columns:
            print()
            print('inspecting:'.ljust(20), on)
            print('any null:'.ljust(15), df[on].isnull().values.any())
            print('null number:'.ljust(15), df[on].isnull().values.sum())
            print(on, 'dtype:', df[on].dtypes)
            print('-' * 20)
            print('describing', on, ':')
            print(df[on].describe())
            print('-' * 20)
            l = df[on]
            s = set(l)
            print('list len:'.ljust(20), len(l))
            print('set len:'.ljust(20), len(s))
            print()
    print('<' * 20)
    print('<' * 20)
    print('<' * 20)

def add_ITC(df, cols, real=False):
    # df = df
    def add_this_counter_column(on_in, df, real=False):
        # global df
        if real:
            read_from = '../saves/'
        else:
            read_from = '../fake/saves/'
        counter = pickle.load(open(read_from + 'counter/' + 'ITC_' + on_in + '_dict.save', "rb"))

        def get_count(x):
            try:
                return counter[x]
            except KeyError:
                return 0

        df['ITC_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
        # counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
        # df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
        # df.drop(on_in, axis=1, inplace=True)
        # df.drop('CC11_'+on_in, axis=1, inplace=True)
        return df


    def log10me(x):
        return np.log10(x)

    def log10me1(x):
        return np.round(np.log10(x + 1), 5)

    def xxx(x):
        d = x / (x + 1)
        return d

    for col in cols:
        df = add_this_counter_column(col, df, real=real)

    for col in cols:
        colc = 'ITC_' + col
        # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
        df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float32)
        # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
        # col1 = 'CC11_'+col
        # df['OinC_'+col] = df[col1]/df[colc]
        df.drop(colc, axis=1, inplace=True)

    return df

def add_11(df, cols, real=False):
    # df = df
    def add_this_counter_column(on_in, df, real=False):
        # global df
        if real:
            read_from = '../saves/'
        else:
            read_from = '../fake/saves/'
        counter = pickle.load(open(read_from + 'counter/' + 'ITC_' + on_in + '_dict.save', "rb"))

        def get_count(x):
            try:
                return counter[x]
            except KeyError:
                return 0

        df['ITC_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
        counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
        df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
        # df.drop(on_in, axis=1, inplace=True)
        # df.drop('CC11_'+on_in, axis=1, inplace=True)
        return df

    def log10me(x):
        return np.log10(x)

    def log10me1(x):
        return np.round(np.log10(x + 1), 5)

    def xxx(x):
        d = x / (x + 1)
        return d

    for col in cols:
        df = add_this_counter_column(col, df, real=real)

    for col in cols:
        colc = 'ITC_' + col
        # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
        df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float32)
        # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
        col1 = 'CC11_' + col
        df['OinC_' + col] = df[col1] / df[colc]
        df['OinC_' + col] = df['OinC_' + col].astype(np.float32)
        df.drop(colc, axis=1, inplace=True)
        df.drop(col1, axis=1, inplace=True)

    return df

def add_column(model, cols, df, column_name):
    output = model.predict(df[cols])
    df[column_name] = output
    return df

def cat(
        train, val,
        iterations=5000, learning_rate=0.3,
        depth=16, early_stop=40,

):
    from catboost import CatBoostClassifier

    X = train.drop('target', axis=1)
    cols = [i for i in X.columns]
    Y = train['target']
    vX = val.drop('target', axis=1)
    vX = vX[cols]
    vY = val['target']
    cat_feature = np.where(X.dtypes == 'category')[0]
    del train, val

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate,
        depth=depth, logging_level='Verbose',
        loss_function='Logloss',
        eval_metric='AUC',
        od_type='Iter',
        od_wait=early_stop,
    )
    model.fit(
        X, Y,
        cat_features=cat_feature,
        eval_set=(vX, vY)
    )
    return model, cols

def train_cat(
        train,
        iterations,
        learning_rate=0.3,
        depth=16,
):
    from catboost import CatBoostClassifier

    X = train.drop('target', axis=1)
    cols = [i for i in X.columns]
    Y = train['target']
    cat_feature = np.where(X.dtypes == 'category')[0]
    del train

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate,
        depth=depth, logging_level='Verbose',
        loss_function='Logloss',
        eval_metric='AUC',
    )
    model.fit(
        X, Y,
        cat_features=cat_feature,
    )
    return model, cols

def cat_predict(model, test):
    p = model.predict_proba(test)
    t = np.array(p).T[1]
    return t