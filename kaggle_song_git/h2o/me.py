import pandas as pd
import time
import numpy as np
import pickle
import h2o
from sklearn.metrics import roc_auc_score
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def auc(m, v, t):
    y_true = v[t]
    y_scores = m.predict(v)
    y_true = h2o.as_list(y_true, use_pandas=True).values
    y_scores = h2o.as_list(y_scores, use_pandas=True).values
    d = roc_auc_score(y_true, y_scores)
    # score.append(d)
    print('AUC:', d)
    return d
    del m
def show_df(df, showme=False):
    print()
    print('>' * 20)
    print('>' * 20)
    print('dtypes of df:')

    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))

    if showme:
        for on in df.columns:
            print()
            print('inspecting:'.ljust(20), on)
            print('any null:'.ljust(15), df[on].isnull().values.any())
            print('null number:'.ljust(15), df[on].isnull().values.sum())
            print(on, 'dtype:', df[on].dtypes)
            print('-' * 20)
            l = df[on]
            s = set(l)
            print('list len:'.ljust(20), len(l))
            print('set len:'.ljust(20), len(s))
            print()
    print('<' * 20)
    print('<' * 20)
    print('<' * 20)
def div_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')

    print()
    print('Our guest selection:')
    print(df.dtypes)
    print('number of columns:', len(df.columns))
    print()

    length = len(df)
    train_size = 0.76
    train_df = df.head(int(length * train_size))
    val_df = df.drop(train_df.index)
    del df
    return  train_df, val_df
def read_df(read_from, load_name):
    load_name = load_name[:-4]
    dt = pickle.load(open(read_from + load_name + '_dict.save', "rb"))
    df = pd.read_csv(read_from + load_name + ".csv", dtype=dt)
    del dt
    return df
