import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

# barebone = True
barebone = False
if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('target')
    df.drop(ccc, axis=1, inplace=True)


# must be a fake feature
inner = [
    'FAKE_[]_0.6788_Light_gbdt_1512883008.csv'
]
# inner = False


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    df1.rename(columns={'target': 'FAKE_'+on}, inplace=True)
    # print(df1.head(10))
    df = df.join(df1)
    del df1


if inner:
    for i in inner:
        insert_this(i)


df = df[[
    'song_year',
    'ISCZ_song_year',
    # 'IMC_membership_days_log10',
    # 'membership_days',
    # 'expiration_month_log10',
    # 'IMC_expiration_month_log10',
    # 'ISC_language',
    # 'ISCZ_name_ln',
    # 'ISC_top2_in_song',
    'target',
    'FAKE_1512883008',
]]
print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 5000
early_stopping_rounds = 200
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.04
num_leaves = 63
max_depth = 10

lambda_l1 = 0
lambda_l2 = 0.3


bagging_fraction = 0.8
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.8
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}
# on = [
#     'msno',
#     'song_id',
#     'target',
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'language',
#     'artist_name',
#     'song_count',
#     'member_count',
#     'song_year',
# ]
# df = df[on]

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
train_set = df.head(int(length*train_size))
val_set = df.drop(train_set.index)
del df
train_set1 = train_set[train_set.index % 2 == 0]
train_set2 = train_set[train_set.index % 2 == 1]
train_set = train_set1.sample(frac=1)
del train_set1
X_tr = train_set.drop(['target'], axis=1)
Y_tr = train_set['target'].values

X_val = val_set.drop(['target'], axis=1)
Y_val = val_set['target'].values

del train_set, val_set

t = len(Y_tr)
t1 = sum(Y_tr)
t0 = t - t1
print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
t = len(Y_val)
t1 = sum(Y_val)
t0 = t - t1
print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
print()
print()

train_set = lgb.Dataset(
    X_tr, Y_tr,
    # weight=[0.1, 1]
)
val_set = lgb.Dataset(
    X_val, Y_val,
    # weight=[0.1, 1]
)
# train_set.max_bin = max_bin
# val_set.max_bin = max_bin

del X_tr, Y_tr, X_val, Y_val

params['metric'] = 'auc'
params['verbose'] = -1
params['objective'] = 'binary'

print('Training...')

model = lgb.train(params,
                  train_set,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping_rounds,
                  valid_sets=[train_set, val_set],
                  verbose_eval=verbose_eval,
                  )

print('best score:', model.best_score['valid_1']['auc'])
print('best iteration:', model.best_iteration)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


