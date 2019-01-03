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
    # counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
    # df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
    # df.drop(on_in, axis=1, inplace=True)


for col in cols:
    print("'{}',".format(col))
    # add_this_counter_column(col)

cols = ['song_id', 'msno']
for col in cols:
    # print("'{}',".format(col))
    add_this_counter_column(col)

def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.log10(x+1)


def xxx(x):
    d = x / (x + 1)
    return x


for col in cols:
    colc = 'ITC_'+col
    # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
    # col1 = 'CC11_'+col
    # df['OinC_'+col] = df[col1]/df[colc]
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

num_boost_round = 2000
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
    # 'rc',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
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
    'top1_in_song',
    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_language_log10_1',

    # 'ITC_song_year_log10_1',
    # 'ITC_song_country_log10_1',
    # 'ITC_rc_log10_1',
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/0.68791one_train_V1003.py
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
ITC_msno                  int64
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
dtype: object
number of rows: 7377418
number of columns: 23
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
'ITC_msno',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
working on: top1_in_song
/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/0.68791one_train_V1003.py:227: SettingWithCopyWarning: 
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
ITC_msno_log10_1        float64
top1_in_song           category
dtype: object
number of columns: 12

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
[10]	training's auc: 0.794614	valid_1's auc: 0.666612
[20]	training's auc: 0.798156	valid_1's auc: 0.668383
[30]	training's auc: 0.801508	valid_1's auc: 0.669835
[40]	training's auc: 0.803115	valid_1's auc: 0.670776
[50]	training's auc: 0.805595	valid_1's auc: 0.671727
[60]	training's auc: 0.809274	valid_1's auc: 0.673017
[70]	training's auc: 0.81145	valid_1's auc: 0.673848
[80]	training's auc: 0.814855	valid_1's auc: 0.67524
[90]	training's auc: 0.817378	valid_1's auc: 0.676192
[100]	training's auc: 0.820216	valid_1's auc: 0.677426
[110]	training's auc: 0.823021	valid_1's auc: 0.678369
[120]	training's auc: 0.825418	valid_1's auc: 0.679407
[130]	training's auc: 0.82771	valid_1's auc: 0.680228
[140]	training's auc: 0.830048	valid_1's auc: 0.681137
[150]	training's auc: 0.831847	valid_1's auc: 0.681776
[160]	training's auc: 0.833845	valid_1's auc: 0.682457
[170]	training's auc: 0.835694	valid_1's auc: 0.683079
[180]	training's auc: 0.837162	valid_1's auc: 0.683512
[190]	training's auc: 0.838608	valid_1's auc: 0.68394
[200]	training's auc: 0.840331	valid_1's auc: 0.684613
[210]	training's auc: 0.841753	valid_1's auc: 0.68503
[220]	training's auc: 0.842877	valid_1's auc: 0.685333
[230]	training's auc: 0.844071	valid_1's auc: 0.685598
[240]	training's auc: 0.845173	valid_1's auc: 0.685928
[250]	training's auc: 0.846359	valid_1's auc: 0.686231
[260]	training's auc: 0.847458	valid_1's auc: 0.686439
[270]	training's auc: 0.848321	valid_1's auc: 0.686579
[280]	training's auc: 0.849133	valid_1's auc: 0.686661
[290]	training's auc: 0.849897	valid_1's auc: 0.686756
[300]	training's auc: 0.850689	valid_1's auc: 0.686857
[310]	training's auc: 0.851516	valid_1's auc: 0.686999
[320]	training's auc: 0.852265	valid_1's auc: 0.687075
[330]	training's auc: 0.853013	valid_1's auc: 0.687167
[340]	training's auc: 0.853693	valid_1's auc: 0.687233
[350]	training's auc: 0.854378	valid_1's auc: 0.687297
[360]	training's auc: 0.854972	valid_1's auc: 0.687334
[370]	training's auc: 0.855559	valid_1's auc: 0.687357
[380]	training's auc: 0.856158	valid_1's auc: 0.687384
[390]	training's auc: 0.856752	valid_1's auc: 0.687396
[400]	training's auc: 0.857283	valid_1's auc: 0.6874
[410]	training's auc: 0.857854	valid_1's auc: 0.687424
[420]	training's auc: 0.858355	valid_1's auc: 0.687426
[430]	training's auc: 0.858898	valid_1's auc: 0.687453
[440]	training's auc: 0.859371	valid_1's auc: 0.68748
[450]	training's auc: 0.859908	valid_1's auc: 0.687471
[460]	training's auc: 0.860377	valid_1's auc: 0.687488
[470]	training's auc: 0.860808	valid_1's auc: 0.687491
[480]	training's auc: 0.861281	valid_1's auc: 0.687525
[490]	training's auc: 0.861742	valid_1's auc: 0.687519
[500]	training's auc: 0.862262	valid_1's auc: 0.687551
[510]	training's auc: 0.862756	valid_1's auc: 0.687573
[520]	training's auc: 0.863167	valid_1's auc: 0.687582
[530]	training's auc: 0.863516	valid_1's auc: 0.687562
[540]	training's auc: 0.863965	valid_1's auc: 0.687571
[550]	training's auc: 0.864368	valid_1's auc: 0.687527
[560]	training's auc: 0.864713	valid_1's auc: 0.687527
[570]	training's auc: 0.865107	valid_1's auc: 0.687521
Early stopping, best iteration is:
[520]	training's auc: 0.863167	valid_1's auc: 0.687582
'''