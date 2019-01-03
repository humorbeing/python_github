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
load_name = 'train_best'
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
    df.drop(on_in, axis=1, inplace=True)


# for col in cols:
#     add_this_counter_column(col)


def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.log10(x+1)


def xxx(x):
    d = x / (x + 1)
    return x


# for col in cols:
#     colc = 'ITC_'+col
#     # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
#     df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
#     # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
#     # col1 = 'CC11_'+col
#     # df['OinC_'+col] = df[col1]/df[colc]
#     df.drop(colc, axis=1, inplace=True)


load_name = 'train_set'
read_from = '../saves01/'
dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
train = pd.read_csv(read_from+load_name+".csv", dtype=dt)
del dt

train.drop(
    [
        'target',
        'msno',
        'song_id',
        'source_system_tab',
        'source_screen_name',
        'source_type',
        'artist_name',
        'song_year',
        'language',
    ],
    axis=1,
    inplace=True
)

df = df.join(train)
del train
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
    # 'language',
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

    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_language_log10_1',

    # 'ITC_song_year_log10_1',
    # 'ITC_song_country_log10_1',
    # 'ITC_rc_log10_1',
]
for w in df.columns:
# for w in work_on:
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1002BBBB.py
What we got:
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
expiration_month       category
composer                 object
lyricist                 object
name                     object
song_country           category
rc                     category
isrc_rest              category
top1_in_song           category
top2_in_song           category
top3_in_song           category
dtype: object
number of rows: 7377418
number of columns: 21
'target',
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'language',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'expiration_month',
'composer',
'lyricist',
'name',
'song_country',
'rc',
'isrc_rest',
'top1_in_song',
'top2_in_song',
'top3_in_song',
working on: language

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
language               category
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
[10]	training's auc: 0.789878	valid_1's auc: 0.664156
[20]	training's auc: 0.794119	valid_1's auc: 0.666007
[30]	training's auc: 0.800603	valid_1's auc: 0.66947
[40]	training's auc: 0.803199	valid_1's auc: 0.67062
[50]	training's auc: 0.806999	valid_1's auc: 0.67206
[60]	training's auc: 0.809616	valid_1's auc: 0.672844
[70]	training's auc: 0.81259	valid_1's auc: 0.674035
[80]	training's auc: 0.814311	valid_1's auc: 0.674753
[90]	training's auc: 0.817252	valid_1's auc: 0.675969
[100]	training's auc: 0.820197	valid_1's auc: 0.677147
[110]	training's auc: 0.822858	valid_1's auc: 0.678285
[120]	training's auc: 0.825417	valid_1's auc: 0.679269
[130]	training's auc: 0.827849	valid_1's auc: 0.680186
[140]	training's auc: 0.830154	valid_1's auc: 0.681108
[150]	training's auc: 0.832386	valid_1's auc: 0.681995
[160]	training's auc: 0.834397	valid_1's auc: 0.682715
[170]	training's auc: 0.836303	valid_1's auc: 0.683347
[180]	training's auc: 0.838097	valid_1's auc: 0.68399
[190]	training's auc: 0.839694	valid_1's auc: 0.684506
[200]	training's auc: 0.841259	valid_1's auc: 0.684966
[210]	training's auc: 0.842619	valid_1's auc: 0.685369
[220]	training's auc: 0.843891	valid_1's auc: 0.685668
[230]	training's auc: 0.845117	valid_1's auc: 0.685972
[240]	training's auc: 0.846173	valid_1's auc: 0.686211
[250]	training's auc: 0.847167	valid_1's auc: 0.686391
[260]	training's auc: 0.848156	valid_1's auc: 0.686554
[270]	training's auc: 0.849055	valid_1's auc: 0.686716
[280]	training's auc: 0.849968	valid_1's auc: 0.68686
[290]	training's auc: 0.850772	valid_1's auc: 0.686968
[300]	training's auc: 0.851731	valid_1's auc: 0.687175
[310]	training's auc: 0.852513	valid_1's auc: 0.68724
[320]	training's auc: 0.853209	valid_1's auc: 0.68733
[330]	training's auc: 0.853887	valid_1's auc: 0.687359
[340]	training's auc: 0.854599	valid_1's auc: 0.687444
[350]	training's auc: 0.855246	valid_1's auc: 0.687452
[360]	training's auc: 0.855915	valid_1's auc: 0.687488
[370]	training's auc: 0.856551	valid_1's auc: 0.687498
[380]	training's auc: 0.857161	valid_1's auc: 0.68757
[390]	training's auc: 0.857737	valid_1's auc: 0.68763
[400]	training's auc: 0.858307	valid_1's auc: 0.687605
[410]	training's auc: 0.858881	valid_1's auc: 0.687641
[420]	training's auc: 0.859376	valid_1's auc: 0.68767
[430]	training's auc: 0.859842	valid_1's auc: 0.687665
[440]	training's auc: 0.860414	valid_1's auc: 0.687633
[450]	training's auc: 0.860895	valid_1's auc: 0.687667
[460]	training's auc: 0.861392	valid_1's auc: 0.687699
[470]	training's auc: 0.861875	valid_1's auc: 0.687676
[480]	training's auc: 0.862425	valid_1's auc: 0.687743
[490]	training's auc: 0.862902	valid_1's auc: 0.687789
[500]	training's auc: 0.863393	valid_1's auc: 0.687768
[510]	training's auc: 0.863868	valid_1's auc: 0.687769
[520]	training's auc: 0.864341	valid_1's auc: 0.687744
[530]	training's auc: 0.864792	valid_1's auc: 0.68773
Early stopping, best iteration is:
[489]	training's auc: 0.862853	valid_1's auc: 0.687795
best score: 0.687795007625
best iteration: 489
complete on: language

working on: expiration_month

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
expiration_month       category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.789603	valid_1's auc: 0.664127
[20]	training's auc: 0.794039	valid_1's auc: 0.66605
[30]	training's auc: 0.800665	valid_1's auc: 0.669313
[40]	training's auc: 0.803297	valid_1's auc: 0.67043
[50]	training's auc: 0.806907	valid_1's auc: 0.671956
[60]	training's auc: 0.809603	valid_1's auc: 0.672804
[70]	training's auc: 0.812537	valid_1's auc: 0.673864
[80]	training's auc: 0.814359	valid_1's auc: 0.674528
[90]	training's auc: 0.817293	valid_1's auc: 0.67569
[100]	training's auc: 0.82021	valid_1's auc: 0.676861
[110]	training's auc: 0.82293	valid_1's auc: 0.67804
[120]	training's auc: 0.825487	valid_1's auc: 0.679019
[130]	training's auc: 0.827858	valid_1's auc: 0.679847
[140]	training's auc: 0.830088	valid_1's auc: 0.68072
[150]	training's auc: 0.83229	valid_1's auc: 0.681639
[160]	training's auc: 0.834261	valid_1's auc: 0.682263
[170]	training's auc: 0.836114	valid_1's auc: 0.682913
[180]	training's auc: 0.837818	valid_1's auc: 0.683453
[190]	training's auc: 0.839473	valid_1's auc: 0.68405
[200]	training's auc: 0.840953	valid_1's auc: 0.684529
[210]	training's auc: 0.842335	valid_1's auc: 0.684873
[220]	training's auc: 0.843593	valid_1's auc: 0.685204
[230]	training's auc: 0.844744	valid_1's auc: 0.685438
[240]	training's auc: 0.84589	valid_1's auc: 0.685705
[250]	training's auc: 0.846847	valid_1's auc: 0.685866
[260]	training's auc: 0.847795	valid_1's auc: 0.686001
[270]	training's auc: 0.848702	valid_1's auc: 0.686169
[280]	training's auc: 0.849564	valid_1's auc: 0.68628
[290]	training's auc: 0.850381	valid_1's auc: 0.686403
[300]	training's auc: 0.851178	valid_1's auc: 0.686499
[310]	training's auc: 0.851847	valid_1's auc: 0.68656
[320]	training's auc: 0.85263	valid_1's auc: 0.686691
[330]	training's auc: 0.85336	valid_1's auc: 0.686734
[340]	training's auc: 0.854012	valid_1's auc: 0.686796
[350]	training's auc: 0.854753	valid_1's auc: 0.686816
[360]	training's auc: 0.85542	valid_1's auc: 0.686815
[370]	training's auc: 0.856101	valid_1's auc: 0.68686
[380]	training's auc: 0.856659	valid_1's auc: 0.686855
[390]	training's auc: 0.857294	valid_1's auc: 0.686882
[400]	training's auc: 0.85788	valid_1's auc: 0.686861
[410]	training's auc: 0.858478	valid_1's auc: 0.686928
[420]	training's auc: 0.859014	valid_1's auc: 0.686939
[430]	training's auc: 0.859557	valid_1's auc: 0.686921
[440]	training's auc: 0.860111	valid_1's auc: 0.686965
[450]	training's auc: 0.860587	valid_1's auc: 0.686992
[460]	training's auc: 0.861108	valid_1's auc: 0.68698
[470]	training's auc: 0.861584	valid_1's auc: 0.686972
[480]	training's auc: 0.862096	valid_1's auc: 0.68697
[490]	training's auc: 0.862553	valid_1's auc: 0.686993
[500]	training's auc: 0.863004	valid_1's auc: 0.686991
[510]	training's auc: 0.863474	valid_1's auc: 0.686997
[520]	training's auc: 0.863965	valid_1's auc: 0.68702
[530]	training's auc: 0.86446	valid_1's auc: 0.686988
[540]	training's auc: 0.864889	valid_1's auc: 0.686992
[550]	training's auc: 0.865303	valid_1's auc: 0.687012
[560]	training's auc: 0.865718	valid_1's auc: 0.687021
[570]	training's auc: 0.866132	valid_1's auc: 0.687002
Early stopping, best iteration is:
[522]	training's auc: 0.864082	valid_1's auc: 0.687034
best score: 0.687033862794
best iteration: 522
complete on: expiration_month

working on: composer

/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1002BBBB.py:230: SettingWithCopyWarning: 
Our guest selection:
target                    uint8
A value is trying to be set on a copy of a slice from a DataFrame.
msno                   category
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
composer               category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790241	valid_1's auc: 0.664699
[20]	training's auc: 0.793959	valid_1's auc: 0.666266
[30]	training's auc: 0.80039	valid_1's auc: 0.669235
[40]	training's auc: 0.80301	valid_1's auc: 0.670264
[50]	training's auc: 0.806611	valid_1's auc: 0.67168
[60]	training's auc: 0.809353	valid_1's auc: 0.67262
[70]	training's auc: 0.812356	valid_1's auc: 0.673752
[80]	training's auc: 0.814102	valid_1's auc: 0.674389
[90]	training's auc: 0.817077	valid_1's auc: 0.675627
[100]	training's auc: 0.820118	valid_1's auc: 0.676907
[110]	training's auc: 0.822666	valid_1's auc: 0.677962
[120]	training's auc: 0.825291	valid_1's auc: 0.678993
[130]	training's auc: 0.827727	valid_1's auc: 0.679925
[140]	training's auc: 0.829873	valid_1's auc: 0.680707
[150]	training's auc: 0.832115	valid_1's auc: 0.681606
[160]	training's auc: 0.834069	valid_1's auc: 0.682291
[170]	training's auc: 0.835979	valid_1's auc: 0.682907
[180]	training's auc: 0.83775	valid_1's auc: 0.683467
[190]	training's auc: 0.839394	valid_1's auc: 0.683948
[200]	training's auc: 0.840944	valid_1's auc: 0.684422
[210]	training's auc: 0.842241	valid_1's auc: 0.684788
[220]	training's auc: 0.843515	valid_1's auc: 0.68511
[230]	training's auc: 0.844628	valid_1's auc: 0.685371
[240]	training's auc: 0.845671	valid_1's auc: 0.685631
[250]	training's auc: 0.846658	valid_1's auc: 0.685799
[260]	training's auc: 0.847563	valid_1's auc: 0.685924
[270]	training's auc: 0.848481	valid_1's auc: 0.686017
[280]	training's auc: 0.849345	valid_1's auc: 0.68615
[290]	training's auc: 0.850134	valid_1's auc: 0.686203
[300]	training's auc: 0.850902	valid_1's auc: 0.686292
[310]	training's auc: 0.851648	valid_1's auc: 0.686337
[320]	training's auc: 0.85232	valid_1's auc: 0.68638
[330]	training's auc: 0.852989	valid_1's auc: 0.686419
[340]	training's auc: 0.853662	valid_1's auc: 0.686458
[350]	training's auc: 0.854348	valid_1's auc: 0.68647
[360]	training's auc: 0.854929	valid_1's auc: 0.686475
[370]	training's auc: 0.85557	valid_1's auc: 0.686495
[380]	training's auc: 0.856194	valid_1's auc: 0.686507
[390]	training's auc: 0.856793	valid_1's auc: 0.686516
[400]	training's auc: 0.857414	valid_1's auc: 0.686537
[410]	training's auc: 0.857953	valid_1's auc: 0.686544
[420]	training's auc: 0.858492	valid_1's auc: 0.686577
[430]	training's auc: 0.858994	valid_1's auc: 0.686549
[440]	training's auc: 0.859541	valid_1's auc: 0.686536
[450]	training's auc: 0.86013	valid_1's auc: 0.686536
[460]	training's auc: 0.860655	valid_1's auc: 0.686544
[470]	training's auc: 0.861127	valid_1's auc: 0.686555
Early stopping, best iteration is:
[422]	training's auc: 0.858601	valid_1's auc: 0.686585
best score: 0.686584528334
best iteration: 422
complete on: composer

working on: lyricist

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
lyricist               category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790409	valid_1's auc: 0.664809
[20]	training's auc: 0.794467	valid_1's auc: 0.666364
[30]	training's auc: 0.800604	valid_1's auc: 0.669432
[40]	training's auc: 0.803251	valid_1's auc: 0.670503
[50]	training's auc: 0.806971	valid_1's auc: 0.671985
[60]	training's auc: 0.809681	valid_1's auc: 0.672845
[70]	training's auc: 0.812625	valid_1's auc: 0.673924
[80]	training's auc: 0.814437	valid_1's auc: 0.674606
[90]	training's auc: 0.817324	valid_1's auc: 0.675763
[100]	training's auc: 0.820262	valid_1's auc: 0.676937
[110]	training's auc: 0.822922	valid_1's auc: 0.678068
[120]	training's auc: 0.825515	valid_1's auc: 0.679089
[130]	training's auc: 0.82778	valid_1's auc: 0.67985
[140]	training's auc: 0.830024	valid_1's auc: 0.680742
[150]	training's auc: 0.832197	valid_1's auc: 0.681582
[160]	training's auc: 0.834165	valid_1's auc: 0.682215
[170]	training's auc: 0.836048	valid_1's auc: 0.682863
[180]	training's auc: 0.837786	valid_1's auc: 0.683416
[190]	training's auc: 0.839294	valid_1's auc: 0.683928
[200]	training's auc: 0.840781	valid_1's auc: 0.684359
[210]	training's auc: 0.842119	valid_1's auc: 0.684714
[220]	training's auc: 0.843403	valid_1's auc: 0.685082
[230]	training's auc: 0.844544	valid_1's auc: 0.685311
[240]	training's auc: 0.84556	valid_1's auc: 0.685595
[250]	training's auc: 0.846558	valid_1's auc: 0.685782
[260]	training's auc: 0.847479	valid_1's auc: 0.685923
[270]	training's auc: 0.84837	valid_1's auc: 0.686024
[280]	training's auc: 0.849251	valid_1's auc: 0.686128
[290]	training's auc: 0.850031	valid_1's auc: 0.686211
[300]	training's auc: 0.850747	valid_1's auc: 0.686294
[310]	training's auc: 0.851463	valid_1's auc: 0.686328
[320]	training's auc: 0.852144	valid_1's auc: 0.686399
[330]	training's auc: 0.852791	valid_1's auc: 0.686438
[340]	training's auc: 0.853473	valid_1's auc: 0.686474
[350]	training's auc: 0.854172	valid_1's auc: 0.686496
[360]	training's auc: 0.854836	valid_1's auc: 0.686553
[370]	training's auc: 0.855479	valid_1's auc: 0.686571
[380]	training's auc: 0.856059	valid_1's auc: 0.686578
[390]	training's auc: 0.856618	valid_1's auc: 0.686576
[400]	training's auc: 0.857201	valid_1's auc: 0.686559
[410]	training's auc: 0.85774	valid_1's auc: 0.686588
[420]	training's auc: 0.858255	valid_1's auc: 0.686605
[430]	training's auc: 0.858811	valid_1's auc: 0.686601
[440]	training's auc: 0.859356	valid_1's auc: 0.686638
[450]	training's auc: 0.859797	valid_1's auc: 0.686607
[460]	training's auc: 0.860285	valid_1's auc: 0.686633
[470]	training's auc: 0.860789	valid_1's auc: 0.686654
[480]	training's auc: 0.861277	valid_1's auc: 0.686635
[490]	training's auc: 0.861698	valid_1's auc: 0.686621
[500]	training's auc: 0.86215	valid_1's auc: 0.686625
[510]	training's auc: 0.862636	valid_1's auc: 0.68667
[520]	training's auc: 0.863136	valid_1's auc: 0.686676
[530]	training's auc: 0.863594	valid_1's auc: 0.686645
[540]	training's auc: 0.864047	valid_1's auc: 0.686653
[550]	training's auc: 0.864487	valid_1's auc: 0.686675
[560]	training's auc: 0.864925	valid_1's auc: 0.686698
[570]	training's auc: 0.865384	valid_1's auc: 0.686709
[580]	training's auc: 0.865842	valid_1's auc: 0.686746
[590]	training's auc: 0.866342	valid_1's auc: 0.686806
[600]	training's auc: 0.866743	valid_1's auc: 0.686829
[610]	training's auc: 0.867108	valid_1's auc: 0.686825
[620]	training's auc: 0.867519	valid_1's auc: 0.686806
[630]	training's auc: 0.867951	valid_1's auc: 0.686775
[640]	training's auc: 0.868365	valid_1's auc: 0.686797
Early stopping, best iteration is:
[599]	training's auc: 0.866703	valid_1's auc: 0.686835
best score: 0.686835043535
best iteration: 599
complete on: lyricist

working on: name

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
name                   category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790325	valid_1's auc: 0.664735
[20]	training's auc: 0.794199	valid_1's auc: 0.666584
[30]	training's auc: 0.800683	valid_1's auc: 0.669634
[40]	training's auc: 0.803395	valid_1's auc: 0.67069
[50]	training's auc: 0.806988	valid_1's auc: 0.672031
[60]	training's auc: 0.809723	valid_1's auc: 0.672977
[70]	training's auc: 0.812676	valid_1's auc: 0.674093
[80]	training's auc: 0.81444	valid_1's auc: 0.674726
[90]	training's auc: 0.817347	valid_1's auc: 0.675878
[100]	training's auc: 0.820302	valid_1's auc: 0.677177
[110]	training's auc: 0.822961	valid_1's auc: 0.678355
[120]	training's auc: 0.825531	valid_1's auc: 0.679408
[130]	training's auc: 0.82787	valid_1's auc: 0.680233
[140]	training's auc: 0.830128	valid_1's auc: 0.681073
[150]	training's auc: 0.832323	valid_1's auc: 0.681873
[160]	training's auc: 0.834237	valid_1's auc: 0.682505
[170]	training's auc: 0.836154	valid_1's auc: 0.683197
[180]	training's auc: 0.837841	valid_1's auc: 0.683696
[190]	training's auc: 0.839477	valid_1's auc: 0.684178
[200]	training's auc: 0.840937	valid_1's auc: 0.684663
[210]	training's auc: 0.842238	valid_1's auc: 0.684984
[220]	training's auc: 0.843472	valid_1's auc: 0.685377
[230]	training's auc: 0.844698	valid_1's auc: 0.68563
[240]	training's auc: 0.845772	valid_1's auc: 0.685877
[250]	training's auc: 0.846765	valid_1's auc: 0.68608
[260]	training's auc: 0.847721	valid_1's auc: 0.68626
[270]	training's auc: 0.848649	valid_1's auc: 0.68638
[280]	training's auc: 0.849541	valid_1's auc: 0.686495
[290]	training's auc: 0.85036	valid_1's auc: 0.686565
[300]	training's auc: 0.85114	valid_1's auc: 0.686667
[310]	training's auc: 0.851874	valid_1's auc: 0.686754
[320]	training's auc: 0.852617	valid_1's auc: 0.686778
[330]	training's auc: 0.853319	valid_1's auc: 0.686802
[340]	training's auc: 0.853963	valid_1's auc: 0.686805
[350]	training's auc: 0.854624	valid_1's auc: 0.686845
[360]	training's auc: 0.855244	valid_1's auc: 0.686866
[370]	training's auc: 0.855859	valid_1's auc: 0.686877
[380]	training's auc: 0.85641	valid_1's auc: 0.686857
[390]	training's auc: 0.857058	valid_1's auc: 0.686877
[400]	training's auc: 0.85762	valid_1's auc: 0.686862
[410]	training's auc: 0.858183	valid_1's auc: 0.686879
[420]	training's auc: 0.85871	valid_1's auc: 0.686895
[430]	training's auc: 0.859236	valid_1's auc: 0.686924
[440]	training's auc: 0.859792	valid_1's auc: 0.686963
[450]	training's auc: 0.860306	valid_1's auc: 0.686937
[460]	training's auc: 0.860813	valid_1's auc: 0.686974
[470]	training's auc: 0.861258	valid_1's auc: 0.686964
[480]	training's auc: 0.86176	valid_1's auc: 0.686937
[490]	training's auc: 0.862235	valid_1's auc: 0.686942
[500]	training's auc: 0.862715	valid_1's auc: 0.68695
[510]	training's auc: 0.863177	valid_1's auc: 0.686972
Early stopping, best iteration is:
[462]	training's auc: 0.860917	valid_1's auc: 0.68699
best score: 0.686990364608
best iteration: 462
complete on: name

working on: song_country

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
song_country           category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.789879	valid_1's auc: 0.664823
[20]	training's auc: 0.793976	valid_1's auc: 0.666518
[30]	training's auc: 0.800345	valid_1's auc: 0.669709
[40]	training's auc: 0.803058	valid_1's auc: 0.670817
[50]	training's auc: 0.806604	valid_1's auc: 0.672182
[60]	training's auc: 0.809414	valid_1's auc: 0.673152
[70]	training's auc: 0.812449	valid_1's auc: 0.674305
[80]	training's auc: 0.81415	valid_1's auc: 0.675017
[90]	training's auc: 0.817075	valid_1's auc: 0.676193
[100]	training's auc: 0.819995	valid_1's auc: 0.677428
[110]	training's auc: 0.822654	valid_1's auc: 0.678513
[120]	training's auc: 0.825244	valid_1's auc: 0.679521
[130]	training's auc: 0.827647	valid_1's auc: 0.680409
[140]	training's auc: 0.829848	valid_1's auc: 0.68122
[150]	training's auc: 0.832105	valid_1's auc: 0.682085
[160]	training's auc: 0.834132	valid_1's auc: 0.682819
[170]	training's auc: 0.836017	valid_1's auc: 0.683445
[180]	training's auc: 0.837802	valid_1's auc: 0.683969
[190]	training's auc: 0.839365	valid_1's auc: 0.684454
[200]	training's auc: 0.840908	valid_1's auc: 0.684888
[210]	training's auc: 0.842278	valid_1's auc: 0.685276
[220]	training's auc: 0.843639	valid_1's auc: 0.685646
[230]	training's auc: 0.844816	valid_1's auc: 0.685947
[240]	training's auc: 0.845904	valid_1's auc: 0.686191
[250]	training's auc: 0.846993	valid_1's auc: 0.686357
[260]	training's auc: 0.847925	valid_1's auc: 0.686486
[270]	training's auc: 0.848957	valid_1's auc: 0.686656
[280]	training's auc: 0.849832	valid_1's auc: 0.686803
[290]	training's auc: 0.850581	valid_1's auc: 0.686863
[300]	training's auc: 0.851414	valid_1's auc: 0.68694
[310]	training's auc: 0.852094	valid_1's auc: 0.687078
[320]	training's auc: 0.852777	valid_1's auc: 0.687155
[330]	training's auc: 0.853456	valid_1's auc: 0.687191
[340]	training's auc: 0.854184	valid_1's auc: 0.687254
[350]	training's auc: 0.854852	valid_1's auc: 0.687276
[360]	training's auc: 0.855517	valid_1's auc: 0.687304
[370]	training's auc: 0.85612	valid_1's auc: 0.687302
[380]	training's auc: 0.856763	valid_1's auc: 0.68734
[390]	training's auc: 0.857365	valid_1's auc: 0.687357
[400]	training's auc: 0.857939	valid_1's auc: 0.68737
[410]	training's auc: 0.858531	valid_1's auc: 0.687369
[420]	training's auc: 0.859098	valid_1's auc: 0.687398
[430]	training's auc: 0.859652	valid_1's auc: 0.687384
[440]	training's auc: 0.860174	valid_1's auc: 0.687425
[450]	training's auc: 0.860641	valid_1's auc: 0.687419
[460]	training's auc: 0.861133	valid_1's auc: 0.687452
[470]	training's auc: 0.861543	valid_1's auc: 0.687454
[480]	training's auc: 0.86202	valid_1's auc: 0.687458
[490]	training's auc: 0.862477	valid_1's auc: 0.687475
[500]	training's auc: 0.862967	valid_1's auc: 0.687479
[510]	training's auc: 0.863406	valid_1's auc: 0.687485
[520]	training's auc: 0.863882	valid_1's auc: 0.687516
[530]	training's auc: 0.864337	valid_1's auc: 0.687511
[540]	training's auc: 0.864759	valid_1's auc: 0.687484
[550]	training's auc: 0.865206	valid_1's auc: 0.687509
[560]	training's auc: 0.865614	valid_1's auc: 0.68752
[570]	training's auc: 0.86607	valid_1's auc: 0.687558
[580]	training's auc: 0.866504	valid_1's auc: 0.68752
[590]	training's auc: 0.866997	valid_1's auc: 0.687529
[600]	training's auc: 0.867365	valid_1's auc: 0.687539
[610]	training's auc: 0.867767	valid_1's auc: 0.687543
[620]	training's auc: 0.868192	valid_1's auc: 0.687536
Early stopping, best iteration is:
[571]	training's auc: 0.866111	valid_1's auc: 0.68756
best score: 0.687559582708
best iteration: 571
complete on: song_country

working on: rc

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
rc                     category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790143	valid_1's auc: 0.6647
[20]	training's auc: 0.794051	valid_1's auc: 0.666298
[30]	training's auc: 0.800398	valid_1's auc: 0.669342
[40]	training's auc: 0.803084	valid_1's auc: 0.670466
[50]	training's auc: 0.80674	valid_1's auc: 0.671888
[60]	training's auc: 0.809513	valid_1's auc: 0.67276
[70]	training's auc: 0.812563	valid_1's auc: 0.67397
[80]	training's auc: 0.814343	valid_1's auc: 0.674634
[90]	training's auc: 0.817221	valid_1's auc: 0.675804
[100]	training's auc: 0.820083	valid_1's auc: 0.677004
[110]	training's auc: 0.822771	valid_1's auc: 0.678201
[120]	training's auc: 0.825221	valid_1's auc: 0.679099
[130]	training's auc: 0.827712	valid_1's auc: 0.680114
[140]	training's auc: 0.829951	valid_1's auc: 0.680945
[150]	training's auc: 0.832128	valid_1's auc: 0.681789
[160]	training's auc: 0.834148	valid_1's auc: 0.682436
[170]	training's auc: 0.836086	valid_1's auc: 0.683111
[180]	training's auc: 0.837822	valid_1's auc: 0.68364
[190]	training's auc: 0.839352	valid_1's auc: 0.684181
[200]	training's auc: 0.840897	valid_1's auc: 0.684614
[210]	training's auc: 0.842222	valid_1's auc: 0.684973
[220]	training's auc: 0.843483	valid_1's auc: 0.685313
[230]	training's auc: 0.844645	valid_1's auc: 0.685616
[240]	training's auc: 0.845713	valid_1's auc: 0.685842
[250]	training's auc: 0.846729	valid_1's auc: 0.686087
[260]	training's auc: 0.847653	valid_1's auc: 0.686262
[270]	training's auc: 0.848565	valid_1's auc: 0.68641
[280]	training's auc: 0.84947	valid_1's auc: 0.686522
[290]	training's auc: 0.850233	valid_1's auc: 0.686644
[300]	training's auc: 0.851073	valid_1's auc: 0.686718
[310]	training's auc: 0.851769	valid_1's auc: 0.686812
[320]	training's auc: 0.852509	valid_1's auc: 0.686851
[330]	training's auc: 0.853227	valid_1's auc: 0.686854
[340]	training's auc: 0.853858	valid_1's auc: 0.686877
[350]	training's auc: 0.854506	valid_1's auc: 0.686906
[360]	training's auc: 0.855206	valid_1's auc: 0.686952
[370]	training's auc: 0.85581	valid_1's auc: 0.687002
[380]	training's auc: 0.856427	valid_1's auc: 0.687034
[390]	training's auc: 0.857108	valid_1's auc: 0.687022
[400]	training's auc: 0.857679	valid_1's auc: 0.687032
[410]	training's auc: 0.858236	valid_1's auc: 0.687026
[420]	training's auc: 0.85875	valid_1's auc: 0.687089
[430]	training's auc: 0.85928	valid_1's auc: 0.6871
[440]	training's auc: 0.859887	valid_1's auc: 0.687153
[450]	training's auc: 0.860405	valid_1's auc: 0.687156
[460]	training's auc: 0.860934	valid_1's auc: 0.687145
[470]	training's auc: 0.861425	valid_1's auc: 0.687142
[480]	training's auc: 0.861966	valid_1's auc: 0.687161
[490]	training's auc: 0.86238	valid_1's auc: 0.687151
Early stopping, best iteration is:
[443]	training's auc: 0.86005	valid_1's auc: 0.687165
best score: 0.687165135875
best iteration: 443
complete on: rc

working on: isrc_rest

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
isrc_rest              category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.789194	valid_1's auc: 0.664441
[20]	training's auc: 0.793749	valid_1's auc: 0.666309
[30]	training's auc: 0.800462	valid_1's auc: 0.669608
[40]	training's auc: 0.803104	valid_1's auc: 0.670642
[50]	training's auc: 0.806626	valid_1's auc: 0.672047
[60]	training's auc: 0.809402	valid_1's auc: 0.672886
[70]	training's auc: 0.81242	valid_1's auc: 0.674068
[80]	training's auc: 0.814215	valid_1's auc: 0.674722
[90]	training's auc: 0.817199	valid_1's auc: 0.676008
[100]	training's auc: 0.820046	valid_1's auc: 0.677169
[110]	training's auc: 0.822736	valid_1's auc: 0.678375
[120]	training's auc: 0.825378	valid_1's auc: 0.679501
[130]	training's auc: 0.827782	valid_1's auc: 0.680368
[140]	training's auc: 0.829986	valid_1's auc: 0.681165
[150]	training's auc: 0.832147	valid_1's auc: 0.682003
[160]	training's auc: 0.834109	valid_1's auc: 0.68269
[170]	training's auc: 0.836046	valid_1's auc: 0.683348
[180]	training's auc: 0.837802	valid_1's auc: 0.683947
[190]	training's auc: 0.839448	valid_1's auc: 0.684399
[200]	training's auc: 0.840866	valid_1's auc: 0.684842
[210]	training's auc: 0.842191	valid_1's auc: 0.685224
[220]	training's auc: 0.843442	valid_1's auc: 0.685516
[230]	training's auc: 0.844586	valid_1's auc: 0.685828
[240]	training's auc: 0.845665	valid_1's auc: 0.686116
[250]	training's auc: 0.846686	valid_1's auc: 0.686237
[260]	training's auc: 0.847637	valid_1's auc: 0.686436
[270]	training's auc: 0.848565	valid_1's auc: 0.686584
[280]	training's auc: 0.84947	valid_1's auc: 0.686745
[290]	training's auc: 0.850296	valid_1's auc: 0.686807
[300]	training's auc: 0.851064	valid_1's auc: 0.686898
[310]	training's auc: 0.851807	valid_1's auc: 0.68703
[320]	training's auc: 0.852471	valid_1's auc: 0.687091
[330]	training's auc: 0.853154	valid_1's auc: 0.687136
[340]	training's auc: 0.85387	valid_1's auc: 0.687206
[350]	training's auc: 0.854555	valid_1's auc: 0.687235
[360]	training's auc: 0.855182	valid_1's auc: 0.687291
[370]	training's auc: 0.855821	valid_1's auc: 0.687295
[380]	training's auc: 0.856454	valid_1's auc: 0.687289
[390]	training's auc: 0.857044	valid_1's auc: 0.687295
[400]	training's auc: 0.857637	valid_1's auc: 0.687327
[410]	training's auc: 0.85825	valid_1's auc: 0.687311
[420]	training's auc: 0.858835	valid_1's auc: 0.687267
[430]	training's auc: 0.859441	valid_1's auc: 0.687306
[440]	training's auc: 0.859977	valid_1's auc: 0.687306
[450]	training's auc: 0.860444	valid_1's auc: 0.687309
Early stopping, best iteration is:
[402]	training's auc: 0.857742	valid_1's auc: 0.687329
best score: 0.687329301932
best iteration: 402
complete on: isrc_rest

working on: top1_in_song

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
top1_in_song           category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790227	valid_1's auc: 0.664696
[20]	training's auc: 0.794223	valid_1's auc: 0.666335
[30]	training's auc: 0.800572	valid_1's auc: 0.66943
[40]	training's auc: 0.803256	valid_1's auc: 0.670548
[50]	training's auc: 0.806819	valid_1's auc: 0.671913
[60]	training's auc: 0.809436	valid_1's auc: 0.672689
[70]	training's auc: 0.812382	valid_1's auc: 0.673856
[80]	training's auc: 0.814208	valid_1's auc: 0.674596
[90]	training's auc: 0.817168	valid_1's auc: 0.675753
[100]	training's auc: 0.820005	valid_1's auc: 0.676833
[110]	training's auc: 0.822777	valid_1's auc: 0.678041
[120]	training's auc: 0.825331	valid_1's auc: 0.678972
[130]	training's auc: 0.827743	valid_1's auc: 0.679898
[140]	training's auc: 0.830037	valid_1's auc: 0.680811
[150]	training's auc: 0.832367	valid_1's auc: 0.681696
[160]	training's auc: 0.834377	valid_1's auc: 0.682406
[170]	training's auc: 0.836408	valid_1's auc: 0.683078
[180]	training's auc: 0.838161	valid_1's auc: 0.683607
[190]	training's auc: 0.839841	valid_1's auc: 0.684166
[200]	training's auc: 0.841365	valid_1's auc: 0.684693
[210]	training's auc: 0.842705	valid_1's auc: 0.685051
[220]	training's auc: 0.844005	valid_1's auc: 0.685423
[230]	training's auc: 0.845289	valid_1's auc: 0.685758
[240]	training's auc: 0.846302	valid_1's auc: 0.686044
[250]	training's auc: 0.847351	valid_1's auc: 0.686243
[260]	training's auc: 0.848321	valid_1's auc: 0.686452
[270]	training's auc: 0.849225	valid_1's auc: 0.686597
[280]	training's auc: 0.85015	valid_1's auc: 0.686783
[290]	training's auc: 0.850974	valid_1's auc: 0.686927
[300]	training's auc: 0.851837	valid_1's auc: 0.687031
[310]	training's auc: 0.852577	valid_1's auc: 0.687133
[320]	training's auc: 0.853278	valid_1's auc: 0.687225
[330]	training's auc: 0.854015	valid_1's auc: 0.68729
[340]	training's auc: 0.854623	valid_1's auc: 0.687358
[350]	training's auc: 0.855397	valid_1's auc: 0.687424
[360]	training's auc: 0.856002	valid_1's auc: 0.687479
[370]	training's auc: 0.856655	valid_1's auc: 0.687532
[380]	training's auc: 0.857294	valid_1's auc: 0.687566
[390]	training's auc: 0.85789	valid_1's auc: 0.68762
[400]	training's auc: 0.858485	valid_1's auc: 0.687673
[410]	training's auc: 0.859022	valid_1's auc: 0.687695
[420]	training's auc: 0.859533	valid_1's auc: 0.687701
[430]	training's auc: 0.86004	valid_1's auc: 0.687685
[440]	training's auc: 0.860575	valid_1's auc: 0.687681
[450]	training's auc: 0.861064	valid_1's auc: 0.687709
[460]	training's auc: 0.861563	valid_1's auc: 0.687723
[470]	training's auc: 0.86206	valid_1's auc: 0.687707
[480]	training's auc: 0.862661	valid_1's auc: 0.687762
[490]	training's auc: 0.863124	valid_1's auc: 0.687775
[500]	training's auc: 0.863605	valid_1's auc: 0.68776
[510]	training's auc: 0.864053	valid_1's auc: 0.687755
[520]	training's auc: 0.864496	valid_1's auc: 0.687753
[530]	training's auc: 0.864933	valid_1's auc: 0.68774
Early stopping, best iteration is:
[487]	training's auc: 0.862997	valid_1's auc: 0.687778
best score: 0.687778075323
best iteration: 487
complete on: top1_in_song

working on: top2_in_song

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
top2_in_song           category
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.789782	valid_1's auc: 0.664738
[20]	training's auc: 0.793882	valid_1's auc: 0.666403
[30]	training's auc: 0.800654	valid_1's auc: 0.669762
[40]	training's auc: 0.803165	valid_1's auc: 0.670754
[50]	training's auc: 0.806763	valid_1's auc: 0.672091
[60]	training's auc: 0.809593	valid_1's auc: 0.673026
[70]	training's auc: 0.812595	valid_1's auc: 0.674092
[80]	training's auc: 0.814329	valid_1's auc: 0.674752
[90]	training's auc: 0.817289	valid_1's auc: 0.675911
[100]	training's auc: 0.820206	valid_1's auc: 0.677192
[110]	training's auc: 0.82289	valid_1's auc: 0.678403
[120]	training's auc: 0.825477	valid_1's auc: 0.679343
[130]	training's auc: 0.827972	valid_1's auc: 0.680404
[140]	training's auc: 0.830183	valid_1's auc: 0.681182
[150]	training's auc: 0.832445	valid_1's auc: 0.682117
[160]	training's auc: 0.834457	valid_1's auc: 0.682792
[170]	training's auc: 0.836471	valid_1's auc: 0.683548
[180]	training's auc: 0.83832	valid_1's auc: 0.684158
[190]	training's auc: 0.840007	valid_1's auc: 0.684674
[200]	training's auc: 0.841544	valid_1's auc: 0.68514
[210]	training's auc: 0.842895	valid_1's auc: 0.685595
[220]	training's auc: 0.844175	valid_1's auc: 0.685969
[230]	training's auc: 0.845324	valid_1's auc: 0.686294
[240]	training's auc: 0.846464	valid_1's auc: 0.686583
[250]	training's auc: 0.847489	valid_1's auc: 0.686771
[260]	training's auc: 0.84849	valid_1's auc: 0.686975
[270]	training's auc: 0.849443	valid_1's auc: 0.687168
[280]	training's auc: 0.850357	valid_1's auc: 0.687287
[290]	training's auc: 0.851116	valid_1's auc: 0.6874
[300]	training's auc: 0.85192	valid_1's auc: 0.687532
[310]	training's auc: 0.852671	valid_1's auc: 0.68764
[320]	training's auc: 0.853458	valid_1's auc: 0.687748
[330]	training's auc: 0.854169	valid_1's auc: 0.687796
[340]	training's auc: 0.854804	valid_1's auc: 0.687815
[350]	training's auc: 0.855447	valid_1's auc: 0.687846
[360]	training's auc: 0.856066	valid_1's auc: 0.687901
[370]	training's auc: 0.8567	valid_1's auc: 0.687948
[380]	training's auc: 0.857331	valid_1's auc: 0.687989
[390]	training's auc: 0.857995	valid_1's auc: 0.688023
[400]	training's auc: 0.858594	valid_1's auc: 0.688059
[410]	training's auc: 0.85914	valid_1's auc: 0.688096
[420]	training's auc: 0.859665	valid_1's auc: 0.688129
[430]	training's auc: 0.86018	valid_1's auc: 0.688134
[440]	training's auc: 0.86074	valid_1's auc: 0.688162
[450]	training's auc: 0.861242	valid_1's auc: 0.688136
[460]	training's auc: 0.861741	valid_1's auc: 0.688171
[470]	training's auc: 0.862243	valid_1's auc: 0.688188
[480]	training's auc: 0.862741	valid_1's auc: 0.688194
[490]	training's auc: 0.863238	valid_1's auc: 0.688179
[500]	training's auc: 0.86371	valid_1's auc: 0.688162
[510]	training's auc: 0.864194	valid_1's auc: 0.688189
[520]	training's auc: 0.864641	valid_1's auc: 0.688191
[530]	training's auc: 0.865067	valid_1's auc: 0.688183
Early stopping, best iteration is:
[482]	training's auc: 0.862848	valid_1's auc: 0.688203
best score: 0.688203331335
best iteration: 482
complete on: top2_in_song

working on: top3_in_song

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
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.789517	valid_1's auc: 0.664327
[20]	training's auc: 0.793973	valid_1's auc: 0.666159
[30]	training's auc: 0.800465	valid_1's auc: 0.669496
[40]	training's auc: 0.803296	valid_1's auc: 0.670634
[50]	training's auc: 0.806854	valid_1's auc: 0.671918
[60]	training's auc: 0.809512	valid_1's auc: 0.672808
[70]	training's auc: 0.81257	valid_1's auc: 0.673936
[80]	training's auc: 0.814261	valid_1's auc: 0.674599
[90]	training's auc: 0.817303	valid_1's auc: 0.675844
[100]	training's auc: 0.820196	valid_1's auc: 0.677029
[110]	training's auc: 0.822828	valid_1's auc: 0.678193
[120]	training's auc: 0.825299	valid_1's auc: 0.679164
[130]	training's auc: 0.827693	valid_1's auc: 0.680084
[140]	training's auc: 0.829993	valid_1's auc: 0.681013
[150]	training's auc: 0.83232	valid_1's auc: 0.681936
[160]	training's auc: 0.834319	valid_1's auc: 0.682609
[170]	training's auc: 0.836226	valid_1's auc: 0.683337
[180]	training's auc: 0.838053	valid_1's auc: 0.683956
[190]	training's auc: 0.839735	valid_1's auc: 0.684495
[200]	training's auc: 0.841294	valid_1's auc: 0.68494
[210]	training's auc: 0.842536	valid_1's auc: 0.6853
[220]	training's auc: 0.843863	valid_1's auc: 0.685655
[230]	training's auc: 0.845076	valid_1's auc: 0.68597
[240]	training's auc: 0.846213	valid_1's auc: 0.686224
[250]	training's auc: 0.84737	valid_1's auc: 0.686436
[260]	training's auc: 0.848291	valid_1's auc: 0.686627
[270]	training's auc: 0.849204	valid_1's auc: 0.686776
[280]	training's auc: 0.850151	valid_1's auc: 0.686953
[290]	training's auc: 0.85095	valid_1's auc: 0.686986
[300]	training's auc: 0.851684	valid_1's auc: 0.687081
[310]	training's auc: 0.852386	valid_1's auc: 0.687127
[320]	training's auc: 0.853129	valid_1's auc: 0.687214
[330]	training's auc: 0.853818	valid_1's auc: 0.68726
[340]	training's auc: 0.854513	valid_1's auc: 0.687317
[350]	training's auc: 0.855178	valid_1's auc: 0.687423
[360]	training's auc: 0.855845	valid_1's auc: 0.687464
[370]	training's auc: 0.856419	valid_1's auc: 0.687451
[380]	training's auc: 0.857072	valid_1's auc: 0.687495
[390]	training's auc: 0.857738	valid_1's auc: 0.687516
[400]	training's auc: 0.858324	valid_1's auc: 0.687549
[410]	training's auc: 0.858859	valid_1's auc: 0.687558
[420]	training's auc: 0.859387	valid_1's auc: 0.687556
[430]	training's auc: 0.859914	valid_1's auc: 0.687553
[440]	training's auc: 0.860479	valid_1's auc: 0.687575
[450]	training's auc: 0.860954	valid_1's auc: 0.687587
[460]	training's auc: 0.861481	valid_1's auc: 0.687623
[470]	training's auc: 0.861952	valid_1's auc: 0.687646
[480]	training's auc: 0.862457	valid_1's auc: 0.687635
[490]	training's auc: 0.862914	valid_1's auc: 0.687611
[500]	training's auc: 0.863355	valid_1's auc: 0.68764
[510]	training's auc: 0.863829	valid_1's auc: 0.68766
[520]	training's auc: 0.864307	valid_1's auc: 0.687661
[530]	training's auc: 0.864756	valid_1's auc: 0.687668
[540]	training's auc: 0.865185	valid_1's auc: 0.687701
[550]	training's auc: 0.865602	valid_1's auc: 0.687716
[560]	training's auc: 0.866064	valid_1's auc: 0.687756
[570]	training's auc: 0.866481	valid_1's auc: 0.687767
[580]	training's auc: 0.866926	valid_1's auc: 0.687793
[590]	training's auc: 0.867341	valid_1's auc: 0.687776
[600]	training's auc: 0.867769	valid_1's auc: 0.68779
[610]	training's auc: 0.868182	valid_1's auc: 0.687787
[620]	training's auc: 0.868588	valid_1's auc: 0.687815
[630]	training's auc: 0.868986	valid_1's auc: 0.687819
[640]	training's auc: 0.86939	valid_1's auc: 0.687807
[650]	training's auc: 0.869761	valid_1's auc: 0.687795
[660]	training's auc: 0.870142	valid_1's auc: 0.68783
[670]	training's auc: 0.870565	valid_1's auc: 0.687829
[680]	training's auc: 0.870912	valid_1's auc: 0.687821
[690]	training's auc: 0.871276	valid_1's auc: 0.687859
[700]	training's auc: 0.871646	valid_1's auc: 0.687868
[710]	training's auc: 0.872025	valid_1's auc: 0.687855
[720]	training's auc: 0.872368	valid_1's auc: 0.687841
[730]	training's auc: 0.872709	valid_1's auc: 0.687866
[740]	training's auc: 0.873086	valid_1's auc: 0.687894
[750]	training's auc: 0.873478	valid_1's auc: 0.687888
[760]	training's auc: 0.873811	valid_1's auc: 0.687908
[770]	training's auc: 0.874143	valid_1's auc: 0.687909
[780]	training's auc: 0.874484	valid_1's auc: 0.687913
[790]	training's auc: 0.874832	valid_1's auc: 0.68793
[800]	training's auc: 0.875201	valid_1's auc: 0.687924
[810]	training's auc: 0.87558	valid_1's auc: 0.687952
[820]	training's auc: 0.875881	valid_1's auc: 0.687965
[830]	training's auc: 0.876202	valid_1's auc: 0.687949
[840]	training's auc: 0.87651	valid_1's auc: 0.687966
[850]	training's auc: 0.876811	valid_1's auc: 0.687968
[860]	training's auc: 0.877108	valid_1's auc: 0.687979
[870]	training's auc: 0.877392	valid_1's auc: 0.687984
[880]	training's auc: 0.87771	valid_1's auc: 0.688018
[890]	training's auc: 0.878044	valid_1's auc: 0.68802
[900]	training's auc: 0.878335	valid_1's auc: 0.688038
[910]	training's auc: 0.878643	valid_1's auc: 0.688069
[920]	training's auc: 0.878931	valid_1's auc: 0.688076
[930]	training's auc: 0.879183	valid_1's auc: 0.688093
[940]	training's auc: 0.879484	valid_1's auc: 0.688084
[950]	training's auc: 0.87975	valid_1's auc: 0.68809
[960]	training's auc: 0.88001	valid_1's auc: 0.688099
[970]	training's auc: 0.880282	valid_1's auc: 0.688106
[980]	training's auc: 0.880541	valid_1's auc: 0.688118
[990]	training's auc: 0.880773	valid_1's auc: 0.688115
[1000]	training's auc: 0.881046	valid_1's auc: 0.68812
[1010]	training's auc: 0.881338	valid_1's auc: 0.688144
[1020]	training's auc: 0.881609	valid_1's auc: 0.68814
[1030]	training's auc: 0.88189	valid_1's auc: 0.688145
[1040]	training's auc: 0.882158	valid_1's auc: 0.688163
[1050]	training's auc: 0.882421	valid_1's auc: 0.688178
[1060]	training's auc: 0.88267	valid_1's auc: 0.688167
[1070]	training's auc: 0.882938	valid_1's auc: 0.688173
[1080]	training's auc: 0.883186	valid_1's auc: 0.688176
[1090]	training's auc: 0.883445	valid_1's auc: 0.688179
[1100]	training's auc: 0.883664	valid_1's auc: 0.688186
[1110]	training's auc: 0.883891	valid_1's auc: 0.688184
[1120]	training's auc: 0.884162	valid_1's auc: 0.688177
[1130]	training's auc: 0.884427	valid_1's auc: 0.68817
[1140]	training's auc: 0.884685	valid_1's auc: 0.688192
[1150]	training's auc: 0.884919	valid_1's auc: 0.68818
[1160]	training's auc: 0.885128	valid_1's auc: 0.688191
[1170]	training's auc: 0.88534	valid_1's auc: 0.688197
[1180]	training's auc: 0.885568	valid_1's auc: 0.688199
[1190]	training's auc: 0.8858	valid_1's auc: 0.688198
[1200]	training's auc: 0.886071	valid_1's auc: 0.688214
[1210]	training's auc: 0.886315	valid_1's auc: 0.688222
[1220]	training's auc: 0.886516	valid_1's auc: 0.688215
[1230]	training's auc: 0.886785	valid_1's auc: 0.688213
[1240]	training's auc: 0.887029	valid_1's auc: 0.688206
[1250]	training's auc: 0.887308	valid_1's auc: 0.688205
[1260]	training's auc: 0.887536	valid_1's auc: 0.68821
Early stopping, best iteration is:
[1217]	training's auc: 0.886466	valid_1's auc: 0.688229
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1002BBBB.py", line 312, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 1
'''