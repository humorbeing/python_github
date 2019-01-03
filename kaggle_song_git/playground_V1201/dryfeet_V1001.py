import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np

# clf = svm.SVC()
clf = SVC(kernel='linear', probability=True)
# clf = RandomForestClassifier(min_samples_leaf=20)
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

barebone = True
# barebone = False
if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('target')
    df.drop(ccc, axis=1, inplace=True)


inner = [
    '[0.67982]_0.6788_Light_gbdt_1512750240.csv',
    '[0.62259]_0.6246_Light_gbdt_1512859793.csv'
]
# inner = False


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    df1.rename(columns={'target': on}, inplace=True)
    # print(df1.head(10))
    df = df.join(df1)
    del df1


def insert_this_test(on_in):
    global df
    on_in = on_in[:-4]
    df1 = pd.read_csv('../saves/submission/'+on_in+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on_in = on_in[-10:]
    df1.rename(columns={'target': on_in}, inplace=True)
    df = df.join(df1)
    del df1


if inner:
    for i in inner:
        insert_this(i)


print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

X_tr = df.drop(['target'], axis=1)
Y_tr = df['target'].values
del df

clf.fit(X_tr, Y_tr)

load_name = 'test_set'
dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
del dt

if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('id')
    df.drop(ccc, axis=1, inplace=True)

if inner:
    for i in inner:
        insert_this_test(i)

print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))


X_test = df.drop(['id'], axis=1)
ids = df['id'].values
del df

print('Making predictions...')

p_test_1 = clf.predict(X_test)
del clf

print('prediction done.')
print('creating submission')
subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = p_test_1
del p_test_1
model_name = 'svm'
subm.to_csv(save_dir+'submission/'+model_name+'.csv',
            index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')
