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
    # 'language',
    # 'top3_in_song',
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
    'top3_in_song',
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
        ns = model.feature_name()
        ims = model.feature_importance()
        for i in range(len(ns)):
            print(ns[i], ':', ims[i])


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
'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/one_train_V1003.py
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
working on: top3_in_song
/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/one_train_V1003.py:228: SettingWithCopyWarning: 
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
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
top3_in_song           category
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
[10]	training's auc: 0.789841	valid_1's auc: 0.665122
[20]	training's auc: 0.793647	valid_1's auc: 0.666553
[30]	training's auc: 0.800009	valid_1's auc: 0.669633
[40]	training's auc: 0.802774	valid_1's auc: 0.670727
[50]	training's auc: 0.806488	valid_1's auc: 0.672119
[60]	training's auc: 0.80926	valid_1's auc: 0.672994
[70]	training's auc: 0.812225	valid_1's auc: 0.674168
[80]	training's auc: 0.813986	valid_1's auc: 0.67483
[90]	training's auc: 0.816939	valid_1's auc: 0.676054
[100]	training's auc: 0.819838	valid_1's auc: 0.677226
[110]	training's auc: 0.822448	valid_1's auc: 0.678344
[120]	training's auc: 0.825016	valid_1's auc: 0.679317
[130]	training's auc: 0.827382	valid_1's auc: 0.680236
[140]	training's auc: 0.829725	valid_1's auc: 0.681161
[150]	training's auc: 0.832051	valid_1's auc: 0.682106
[160]	training's auc: 0.834073	valid_1's auc: 0.682799
[170]	training's auc: 0.836048	valid_1's auc: 0.683488
[180]	training's auc: 0.837806	valid_1's auc: 0.68409
[190]	training's auc: 0.839479	valid_1's auc: 0.68466
[200]	training's auc: 0.840967	valid_1's auc: 0.685133
[210]	training's auc: 0.842386	valid_1's auc: 0.685568
[220]	training's auc: 0.843712	valid_1's auc: 0.685937
[230]	training's auc: 0.844884	valid_1's auc: 0.686239
[240]	training's auc: 0.845908	valid_1's auc: 0.686496
[250]	training's auc: 0.846932	valid_1's auc: 0.686723
[260]	training's auc: 0.84788	valid_1's auc: 0.686907
[270]	training's auc: 0.848758	valid_1's auc: 0.687098
[280]	training's auc: 0.849666	valid_1's auc: 0.68728
[290]	training's auc: 0.850429	valid_1's auc: 0.687382
[300]	training's auc: 0.851238	valid_1's auc: 0.687489
[310]	training's auc: 0.851987	valid_1's auc: 0.687579
[320]	training's auc: 0.852791	valid_1's auc: 0.687685
[330]	training's auc: 0.85349	valid_1's auc: 0.687757
[340]	training's auc: 0.854166	valid_1's auc: 0.687817
[350]	training's auc: 0.854898	valid_1's auc: 0.687863
[360]	training's auc: 0.855558	valid_1's auc: 0.687915
[370]	training's auc: 0.85622	valid_1's auc: 0.687939
[380]	training's auc: 0.856874	valid_1's auc: 0.687963
[390]	training's auc: 0.857462	valid_1's auc: 0.688024
[400]	training's auc: 0.858061	valid_1's auc: 0.688075
[410]	training's auc: 0.858667	valid_1's auc: 0.688086
[420]	training's auc: 0.859176	valid_1's auc: 0.6881
[430]	training's auc: 0.859696	valid_1's auc: 0.688094
[440]	training's auc: 0.860312	valid_1's auc: 0.688126
[450]	training's auc: 0.860786	valid_1's auc: 0.688127
[460]	training's auc: 0.861267	valid_1's auc: 0.68811
[470]	training's auc: 0.86169	valid_1's auc: 0.688126
[480]	training's auc: 0.86223	valid_1's auc: 0.688146
[490]	training's auc: 0.862722	valid_1's auc: 0.688154
[500]	training's auc: 0.863191	valid_1's auc: 0.688182
[510]	training's auc: 0.863733	valid_1's auc: 0.688243
[520]	training's auc: 0.864196	valid_1's auc: 0.688248
[530]	training's auc: 0.864621	valid_1's auc: 0.688239
[540]	training's auc: 0.865083	valid_1's auc: 0.688241
[550]	training's auc: 0.865496	valid_1's auc: 0.688261
[560]	training's auc: 0.865917	valid_1's auc: 0.688244
[570]	training's auc: 0.866333	valid_1's auc: 0.688244
[580]	training's auc: 0.866727	valid_1's auc: 0.688236
[590]	training's auc: 0.867136	valid_1's auc: 0.688246
[600]	training's auc: 0.867548	valid_1's auc: 0.688266
[610]	training's auc: 0.867988	valid_1's auc: 0.688287
[620]	training's auc: 0.868422	valid_1's auc: 0.688315
[630]	training's auc: 0.86877	valid_1's auc: 0.688332
[640]	training's auc: 0.869188	valid_1's auc: 0.688332
[650]	training's auc: 0.86961	valid_1's auc: 0.688346
[660]	training's auc: 0.869982	valid_1's auc: 0.688347
[670]	training's auc: 0.870326	valid_1's auc: 0.688348
[680]	training's auc: 0.870674	valid_1's auc: 0.688359
[690]	training's auc: 0.871073	valid_1's auc: 0.688389
[700]	training's auc: 0.87144	valid_1's auc: 0.688398
[710]	training's auc: 0.871862	valid_1's auc: 0.688402
[720]	training's auc: 0.87222	valid_1's auc: 0.688412
[730]	training's auc: 0.872561	valid_1's auc: 0.688401
[740]	training's auc: 0.872893	valid_1's auc: 0.688417
[750]	training's auc: 0.873223	valid_1's auc: 0.688439
[760]	training's auc: 0.873613	valid_1's auc: 0.688483
[770]	training's auc: 0.87395	valid_1's auc: 0.688477
[780]	training's auc: 0.874308	valid_1's auc: 0.688499
[790]	training's auc: 0.874684	valid_1's auc: 0.688485
[800]	training's auc: 0.875045	valid_1's auc: 0.688498
[810]	training's auc: 0.875392	valid_1's auc: 0.688514
[820]	training's auc: 0.875673	valid_1's auc: 0.688508
[830]	training's auc: 0.87602	valid_1's auc: 0.688522
[840]	training's auc: 0.876333	valid_1's auc: 0.688536
[850]	training's auc: 0.876678	valid_1's auc: 0.688551
[860]	training's auc: 0.876967	valid_1's auc: 0.688566
[870]	training's auc: 0.877277	valid_1's auc: 0.688588
[880]	training's auc: 0.87754	valid_1's auc: 0.688611
[890]	training's auc: 0.877829	valid_1's auc: 0.68862
[900]	training's auc: 0.878171	valid_1's auc: 0.688623
[910]	training's auc: 0.878484	valid_1's auc: 0.68865
[920]	training's auc: 0.878766	valid_1's auc: 0.688633
[930]	training's auc: 0.879043	valid_1's auc: 0.688663
[940]	training's auc: 0.879309	valid_1's auc: 0.688648
[950]	training's auc: 0.879588	valid_1's auc: 0.688661
[960]	training's auc: 0.879849	valid_1's auc: 0.688675
[970]	training's auc: 0.880116	valid_1's auc: 0.688674
[980]	training's auc: 0.880373	valid_1's auc: 0.688679
[990]	training's auc: 0.88063	valid_1's auc: 0.688696
[1000]	training's auc: 0.880899	valid_1's auc: 0.688694
[1010]	training's auc: 0.881182	valid_1's auc: 0.68868
[1020]	training's auc: 0.881475	valid_1's auc: 0.688678
[1030]	training's auc: 0.881759	valid_1's auc: 0.688675
[1040]	training's auc: 0.882062	valid_1's auc: 0.688682
Early stopping, best iteration is:
[994]	training's auc: 0.880753	valid_1's auc: 0.688702
'''

