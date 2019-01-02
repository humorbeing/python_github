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
inner = False


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    df1.rename(columns={'target': 'FAKE_'+on}, inplace=True)
    df = df.join(df1)
    del df1


cc = df.drop('target', axis=1)
# print(cc.dtypes)
cols = cc.columns
del cc

counter = {}


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 0


def add_this_counter_column(on_in):
    global counter, df
    read_from = '../fake/saves/'
    counter = pickle.load(open(read_from+'counter/'+'ITC_'+on_in+'_dict.save', "rb"))
    df['ITC_'+on_in] = df[on_in].apply(get_count).astype(np.int64)
    counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
    df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
    # df.drop(on_in, axis=1, inplace=True)
    # df.drop('CC11_'+on_in, axis=1, inplace=True)


for col in cols:
    print("'{}',".format(col))
    # add_this_counter_column(col)

cols = ['song_id', 'msno', 'name']
for col in cols:
    # print("'{}',".format(col))
    add_this_counter_column(col)

def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.log10(x+1)


def xxx(x):
    d = x / (x + 1)
    return d


for col in cols:
    colc = 'ITC_'+col
    df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
    col1 = 'CC11_'+col
    df['OinC_'+col] = df[col1]/df[colc]
    # df.drop(colc, axis=1, inplace=True)


# load_name = 'train_set'
# read_from = '../saves01/'
# dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
# train = pd.read_csv(read_from+load_name+".csv", dtype=dt)
# del dt
#
# train.drop(
#     [
#         'target',
#     ],
#     axis=1,
#     inplace=True
# )
#
# df = df.join(train)
# del train


if inner:
    for i in inner:
        insert_this(i)

print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 5000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 511
max_depth = -1

max_bin = 255
lambda_l1 = 0.2
lambda_l2 = 0


bagging_fraction = 0.9
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.9
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,
    'max_bin': max_bin,

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
fixed = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    # 'composer',
    # 'lyricist',
    'song_year',
    'language',
    # 'top3_in_song',
    # 'rc',
    'ITC_song_id_log10_1',
    # 'ITC_msno_log10_1',
    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'FAKE_1512883008',
]
result = {}
for w in df.columns:
    print("'{}',".format(w))

work_on = [
    # 'ITC_msno',
    # 'CC11_msno',
    # 'ITC_name',
    # 'CC11_name',
    # 'ITC_song_id_log10',
    # 'ITC_song_id_log10_1',
    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    # 'ITC_msno_log10_1',
    'ITC_msno_x_1',
    'OinC_msno',
    # 'ITC_name_log10',
    # 'ITC_name_log10_1',
    # 'ITC_name_x_1',
    # 'OinC_name',
]
for w in work_on:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df_on = df[toto]

        for col in df_on.columns:
            if df_on[col].dtype == object:
                df_on[col] = df_on[col].astype('category')


        print()
        print('Our guest selection:')
        print(df_on.dtypes)
        print('number of columns:', len(df_on.columns))
        print()

        # save_me = True
        save_me = False
        if save_me:
            print(' SAVE ' * 5)
            print(' SAVE ' * 5)
            print(' SAVE ' * 5)

            print('creating train set.')
            save_name = 'train'
            vers = '_me2'
            d = df_on.dtypes.to_dict()
            # print(d)
            print('dtypes of df:')
            print('>' * 20)
            print(df_on.dtypes)
            print('number of columns:', len(df_on.columns))
            print('number of data:', len(df_on))
            print('<' * 20)
            df_on.to_csv(save_dir + save_name + vers + '.csv', index=False)
            pickle.dump(d, open(save_dir + save_name + vers + '_dict.save', "wb"))

            print('done.')


        length = len(df_on)
        train_size = 0.76
        train_set = df_on.head(int(length*train_size))
        val_set = df_on.drop(train_set.index)
        del df_on

        train_set = train_set.sample(frac=1)
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
        # train_set.max_bin = max_bin
        val_set = lgb.Dataset(
            X_val, Y_val,
            # weight=[0.1, 1]
        )
        train_set.max_bin = max_bin
        val_set.max_bin = max_bin

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
        del train_set, val_set
        print('complete on:', w)
        result[w] = model.best_score['valid_1']['auc']
        print()
        ns = model.feature_name()
        ims = model.feature_importance()
        for i in range(len(ns)):
            print(ns[i].rjust(20), ':', ims[i])


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))
# reversed(sorted_x)
# print(sorted_x)
for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'gender',
'artist_name',
'composer',
'lyricist',
'language',
'name',
'song_year',
'song_country',
'rc',
'isrc_rest',
'top1_in_song',
'top2_in_song',
'top3_in_song',
What we got:
msno                     object
song_id                  object
source_system_tab        object
source_screen_name       object
source_type              object
target                    uint8
gender                   object
artist_name              object
composer                 object
lyricist                 object
language               category
name                     object
song_year              category
song_country           category
rc                     category
isrc_rest              category
top1_in_song           category
top2_in_song           category
top3_in_song           category
ITC_song_id               int64
CC11_song_id              int64
ITC_msno                  int64
CC11_msno                 int64
ITC_name                  int64
CC11_name                 int64
ITC_song_id_log10       float64
ITC_song_id_log10_1     float64
ITC_song_id_x_1         float64
OinC_song_id            float64
ITC_msno_log10          float64
ITC_msno_log10_1        float64
ITC_msno_x_1            float64
OinC_msno               float64
ITC_name_log10          float64
ITC_name_log10_1        float64
ITC_name_x_1            float64
OinC_name               float64
dtype: object
number of rows: 7377418
number of columns: 37
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'gender',
'artist_name',
'composer',
'lyricist',
'language',
'name',
'song_year',
'song_country',
'rc',
'isrc_rest',
'top1_in_song',
'top2_in_song',
'top3_in_song',
'ITC_song_id',
'CC11_song_id',
'ITC_msno',
'CC11_msno',
'ITC_name',
'CC11_name',
'ITC_song_id_log10',
'ITC_song_id_log10_1',
'ITC_song_id_x_1',
'OinC_song_id',
'ITC_msno_log10',
'ITC_msno_log10_1',
'ITC_msno_x_1',
'OinC_msno',
'ITC_name_log10',
'ITC_name_log10_1',
'ITC_name_x_1',
'OinC_name',
working on: ITC_msno_x_1
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py:237: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')

Our guest selection:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
language               category
ITC_song_id_log10_1     float64
ITC_msno_x_1            float64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:648: LGBMDeprecationWarning: The `max_bin` parameter is deprecated and will be removed in 2.0.12 version. Please use `params` to pass this parameter.
  'Please use `params` to pass this parameter.', LGBMDeprecationWarning)
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795487	valid_1's auc: 0.667227
[20]	training's auc: 0.799734	valid_1's auc: 0.669443
[30]	training's auc: 0.802657	valid_1's auc: 0.67083
[40]	training's auc: 0.804064	valid_1's auc: 0.671221
[50]	training's auc: 0.808138	valid_1's auc: 0.672922
[60]	training's auc: 0.810596	valid_1's auc: 0.67364
[70]	training's auc: 0.813536	valid_1's auc: 0.674897
[80]	training's auc: 0.81522	valid_1's auc: 0.675456
[90]	training's auc: 0.818273	valid_1's auc: 0.676777
[100]	training's auc: 0.821169	valid_1's auc: 0.677996
[110]	training's auc: 0.823521	valid_1's auc: 0.678879
[120]	training's auc: 0.825968	valid_1's auc: 0.679825
[130]	training's auc: 0.828361	valid_1's auc: 0.680704
[140]	training's auc: 0.830601	valid_1's auc: 0.681624
[150]	training's auc: 0.832724	valid_1's auc: 0.682432
[160]	training's auc: 0.834696	valid_1's auc: 0.683114
[170]	training's auc: 0.836593	valid_1's auc: 0.683738
[180]	training's auc: 0.838331	valid_1's auc: 0.684311
[190]	training's auc: 0.839986	valid_1's auc: 0.684847
[200]	training's auc: 0.841455	valid_1's auc: 0.685233
[210]	training's auc: 0.842892	valid_1's auc: 0.6857
[220]	training's auc: 0.844128	valid_1's auc: 0.686141
[230]	training's auc: 0.845366	valid_1's auc: 0.686456
[240]	training's auc: 0.846453	valid_1's auc: 0.686756
[250]	training's auc: 0.84751	valid_1's auc: 0.686994
[260]	training's auc: 0.848474	valid_1's auc: 0.687141
[270]	training's auc: 0.849363	valid_1's auc: 0.68724
[280]	training's auc: 0.850286	valid_1's auc: 0.687453
[290]	training's auc: 0.851049	valid_1's auc: 0.687518
[300]	training's auc: 0.851805	valid_1's auc: 0.687592
[310]	training's auc: 0.85254	valid_1's auc: 0.687698
[320]	training's auc: 0.853263	valid_1's auc: 0.687794
[330]	training's auc: 0.853938	valid_1's auc: 0.687847
[340]	training's auc: 0.854652	valid_1's auc: 0.687872
[350]	training's auc: 0.855351	valid_1's auc: 0.687903
[360]	training's auc: 0.855945	valid_1's auc: 0.687918
[370]	training's auc: 0.856497	valid_1's auc: 0.687926
[380]	training's auc: 0.857102	valid_1's auc: 0.68795
[390]	training's auc: 0.857741	valid_1's auc: 0.687967
[400]	training's auc: 0.858369	valid_1's auc: 0.687977
[410]	training's auc: 0.858936	valid_1's auc: 0.687967
[420]	training's auc: 0.859427	valid_1's auc: 0.687989
[430]	training's auc: 0.859893	valid_1's auc: 0.688025
[440]	training's auc: 0.860447	valid_1's auc: 0.687989
[450]	training's auc: 0.860908	valid_1's auc: 0.687997
[460]	training's auc: 0.861348	valid_1's auc: 0.687992
[470]	training's auc: 0.861844	valid_1's auc: 0.688029
[480]	training's auc: 0.862321	valid_1's auc: 0.688013
[490]	training's auc: 0.862821	valid_1's auc: 0.688015
[500]	training's auc: 0.863279	valid_1's auc: 0.688009
[510]	training's auc: 0.863748	valid_1's auc: 0.688008
[520]	training's auc: 0.864227	valid_1's auc: 0.688026
[530]	training's auc: 0.864646	valid_1's auc: 0.688006
[540]	training's auc: 0.865094	valid_1's auc: 0.688021
[550]	training's auc: 0.865551	valid_1's auc: 0.68803
[560]	training's auc: 0.866011	valid_1's auc: 0.688048
[570]	training's auc: 0.866476	valid_1's auc: 0.688059
[580]	training's auc: 0.86689	valid_1's auc: 0.688034
[590]	training's auc: 0.8673	valid_1's auc: 0.688001
[600]	training's auc: 0.867695	valid_1's auc: 0.688019
[610]	training's auc: 0.868073	valid_1's auc: 0.688018
Early stopping, best iteration is:
[568]	training's auc: 0.866396	valid_1's auc: 0.688062
best score: 0.688062494045
best iteration: 568
complete on: ITC_msno_x_1

                msno : 177850
             song_id : 38544
   source_system_tab : 598
  source_screen_name : 1661
         source_type : 1398
         artist_name : 62461
           song_year : 1685
            language : 389
 ITC_song_id_log10_1 : 2308
        ITC_msno_x_1 : 2786
working on: OinC_msno

Our guest selection:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
language               category
ITC_song_id_log10_1     float64
OinC_msno               float64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.810551	valid_1's auc: 0.638514
[20]	training's auc: 0.813911	valid_1's auc: 0.643098
[30]	training's auc: 0.816223	valid_1's auc: 0.641859
[40]	training's auc: 0.817825	valid_1's auc: 0.642008
[50]	training's auc: 0.820655	valid_1's auc: 0.642909
[60]	training's auc: 0.822837	valid_1's auc: 0.643381
[70]	training's auc: 0.825092	valid_1's auc: 0.644337
[80]	training's auc: 0.826653	valid_1's auc: 0.644942
[90]	training's auc: 0.828781	valid_1's auc: 0.645626
[100]	training's auc: 0.830788	valid_1's auc: 0.64615
[110]	training's auc: 0.832582	valid_1's auc: 0.6466
[120]	training's auc: 0.83448	valid_1's auc: 0.647081
[130]	training's auc: 0.836326	valid_1's auc: 0.647632
[140]	training's auc: 0.838038	valid_1's auc: 0.648067
[150]	training's auc: 0.839756	valid_1's auc: 0.648509
[160]	training's auc: 0.841377	valid_1's auc: 0.648866
[170]	training's auc: 0.843011	valid_1's auc: 0.649285
[180]	training's auc: 0.844565	valid_1's auc: 0.649635
[190]	training's auc: 0.846006	valid_1's auc: 0.649915
[200]	training's auc: 0.847378	valid_1's auc: 0.650166
[210]	training's auc: 0.848628	valid_1's auc: 0.650409
[220]	training's auc: 0.849963	valid_1's auc: 0.650693
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py", line 323, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 205, in train
    evaluation_result_list.extend(booster.eval_train(feval))
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1612, in eval_train
    return self.__inner_eval(self.__train_data_name, 0, feval)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1880, in __inner_eval
    result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
KeyboardInterrupt

Process finished with exit code 1
'''