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

learning_rate = 0.2
num_leaves = 63
max_depth = 10

max_bin = 15
lambda_l1 = 0
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
    # 'source_system_tab',
    # 'source_screen_name',
    # 'source_type',
    'artist_name',
    # 'composer',
    # 'lyricist',
    # 'song_year',
    # 'language',
    # 'top3_in_song',
    # 'rc',
    # 'ITC_song_id_log10_1',
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
    'source_system_tab',
    'source_screen_name',
    'source_type',
    # 'artist_name',
    # 'composer',
    # 'lyricist',
    'song_year',
    'language',
    'top3_in_song',
    # 'rc',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # 'top3_in_song',
    # 'artist_name',
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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1103BBBB.py
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
working on: source_system_tab
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1103BBBB.py:241: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')

Our guest selection:
target                  uint8
msno                 category
song_id              category
artist_name          category
source_system_tab    category
dtype: object
number of columns: 5

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
[10]	training's auc: 0.705014	valid_1's auc: 0.620909
[20]	training's auc: 0.718892	valid_1's auc: 0.627563
[30]	training's auc: 0.726678	valid_1's auc: 0.63041
[40]	training's auc: 0.733346	valid_1's auc: 0.633621
[50]	training's auc: 0.73941	valid_1's auc: 0.636195
[60]	training's auc: 0.744226	valid_1's auc: 0.638625
[70]	training's auc: 0.748268	valid_1's auc: 0.640116
[80]	training's auc: 0.751553	valid_1's auc: 0.64142
[90]	training's auc: 0.754442	valid_1's auc: 0.642702
[100]	training's auc: 0.757105	valid_1's auc: 0.643738
[110]	training's auc: 0.759458	valid_1's auc: 0.644551
[120]	training's auc: 0.761959	valid_1's auc: 0.645498
[130]	training's auc: 0.764166	valid_1's auc: 0.646332
[140]	training's auc: 0.766053	valid_1's auc: 0.646923
[150]	training's auc: 0.7678	valid_1's auc: 0.647575
[160]	training's auc: 0.769627	valid_1's auc: 0.648042
[170]	training's auc: 0.770797	valid_1's auc: 0.648462
[180]	training's auc: 0.772356	valid_1's auc: 0.648909
[190]	training's auc: 0.773365	valid_1's auc: 0.649226
[200]	training's auc: 0.774775	valid_1's auc: 0.64959
[210]	training's auc: 0.775729	valid_1's auc: 0.649887
[220]	training's auc: 0.776948	valid_1's auc: 0.650223
[230]	training's auc: 0.777885	valid_1's auc: 0.650379
[240]	training's auc: 0.77876	valid_1's auc: 0.650603
[250]	training's auc: 0.77978	valid_1's auc: 0.650826
[260]	training's auc: 0.780573	valid_1's auc: 0.651
[270]	training's auc: 0.781365	valid_1's auc: 0.651219
[280]	training's auc: 0.782206	valid_1's auc: 0.651416
[290]	training's auc: 0.782785	valid_1's auc: 0.651612
[300]	training's auc: 0.783394	valid_1's auc: 0.651805
[310]	training's auc: 0.78405	valid_1's auc: 0.651927
[320]	training's auc: 0.784631	valid_1's auc: 0.651995
[330]	training's auc: 0.785268	valid_1's auc: 0.652163
[340]	training's auc: 0.785849	valid_1's auc: 0.65229
[350]	training's auc: 0.786288	valid_1's auc: 0.652502
[360]	training's auc: 0.786794	valid_1's auc: 0.652554
[370]	training's auc: 0.787389	valid_1's auc: 0.652662
[380]	training's auc: 0.787825	valid_1's auc: 0.652672
[390]	training's auc: 0.788514	valid_1's auc: 0.652766
[400]	training's auc: 0.789435	valid_1's auc: 0.65306
[410]	training's auc: 0.790156	valid_1's auc: 0.653208
[420]	training's auc: 0.791572	valid_1's auc: 0.653558
[430]	training's auc: 0.792101	valid_1's auc: 0.653641
[440]	training's auc: 0.792658	valid_1's auc: 0.653699
[450]	training's auc: 0.793203	valid_1's auc: 0.653792
[460]	training's auc: 0.793658	valid_1's auc: 0.653883
[470]	training's auc: 0.794108	valid_1's auc: 0.653974
[480]	training's auc: 0.794663	valid_1's auc: 0.65402
[490]	training's auc: 0.795078	valid_1's auc: 0.654014
[500]	training's auc: 0.795602	valid_1's auc: 0.654064
[510]	training's auc: 0.796289	valid_1's auc: 0.654153
[520]	training's auc: 0.796781	valid_1's auc: 0.654262
[530]	training's auc: 0.797298	valid_1's auc: 0.654273
[540]	training's auc: 0.797597	valid_1's auc: 0.65427
[550]	training's auc: 0.798015	valid_1's auc: 0.654288
[560]	training's auc: 0.798368	valid_1's auc: 0.654329
[570]	training's auc: 0.79872	valid_1's auc: 0.6543
[580]	training's auc: 0.799062	valid_1's auc: 0.654302
[590]	training's auc: 0.799328	valid_1's auc: 0.654317
[600]	training's auc: 0.79983	valid_1's auc: 0.654339
[610]	training's auc: 0.800029	valid_1's auc: 0.654355
[620]	training's auc: 0.800373	valid_1's auc: 0.654394
[630]	training's auc: 0.800732	valid_1's auc: 0.654431
[640]	training's auc: 0.801038	valid_1's auc: 0.65444
[650]	training's auc: 0.801348	valid_1's auc: 0.65448
[660]	training's auc: 0.801673	valid_1's auc: 0.654514
[670]	training's auc: 0.80195	valid_1's auc: 0.654547
[680]	training's auc: 0.802282	valid_1's auc: 0.654543
[690]	training's auc: 0.802605	valid_1's auc: 0.654539
[700]	training's auc: 0.802841	valid_1's auc: 0.654556
[710]	training's auc: 0.803122	valid_1's auc: 0.654547
[720]	training's auc: 0.803399	valid_1's auc: 0.654526
[730]	training's auc: 0.80365	valid_1's auc: 0.654569
[740]	training's auc: 0.803914	valid_1's auc: 0.654563
[750]	training's auc: 0.804306	valid_1's auc: 0.654589
[760]	training's auc: 0.804819	valid_1's auc: 0.654712
[770]	training's auc: 0.805142	valid_1's auc: 0.654781
[780]	training's auc: 0.805411	valid_1's auc: 0.654793
[790]	training's auc: 0.805678	valid_1's auc: 0.654769
[800]	training's auc: 0.805921	valid_1's auc: 0.654765
[810]	training's auc: 0.806116	valid_1's auc: 0.654776
[820]	training's auc: 0.806311	valid_1's auc: 0.654793
[830]	training's auc: 0.806551	valid_1's auc: 0.65479
Early stopping, best iteration is:
[784]	training's auc: 0.805514	valid_1's auc: 0.654813
best score: 0.65481288231
best iteration: 784
complete on: source_system_tab

                msno : 6544
             song_id : 5245
         artist_name : 14905
   source_system_tab : 7301
working on: source_screen_name

Our guest selection:
target                   uint8
msno                  category
song_id               category
artist_name           category
source_screen_name    category
dtype: object
number of columns: 5

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.71261	valid_1's auc: 0.624832
[20]	training's auc: 0.725592	valid_1's auc: 0.629993
[30]	training's auc: 0.733129	valid_1's auc: 0.633321
[40]	training's auc: 0.739281	valid_1's auc: 0.636062
[50]	training's auc: 0.744282	valid_1's auc: 0.638302
[60]	training's auc: 0.748853	valid_1's auc: 0.640226
[70]	training's auc: 0.752263	valid_1's auc: 0.642029
[80]	training's auc: 0.755981	valid_1's auc: 0.643794
[90]	training's auc: 0.758932	valid_1's auc: 0.645005
[100]	training's auc: 0.761609	valid_1's auc: 0.64599
[110]	training's auc: 0.763972	valid_1's auc: 0.646815
[120]	training's auc: 0.766386	valid_1's auc: 0.647778
[130]	training's auc: 0.768581	valid_1's auc: 0.64853
[140]	training's auc: 0.770337	valid_1's auc: 0.649039
[150]	training's auc: 0.772069	valid_1's auc: 0.649561
[160]	training's auc: 0.77403	valid_1's auc: 0.650177
[170]	training's auc: 0.775422	valid_1's auc: 0.650777
[180]	training's auc: 0.776954	valid_1's auc: 0.651163
[190]	training's auc: 0.777982	valid_1's auc: 0.651698
[200]	training's auc: 0.779388	valid_1's auc: 0.652171
[210]	training's auc: 0.780436	valid_1's auc: 0.65254
[220]	training's auc: 0.781684	valid_1's auc: 0.652936
[230]	training's auc: 0.782577	valid_1's auc: 0.653135
[240]	training's auc: 0.783465	valid_1's auc: 0.653408
[250]	training's auc: 0.784528	valid_1's auc: 0.653662
[260]	training's auc: 0.785344	valid_1's auc: 0.65389
[270]	training's auc: 0.786078	valid_1's auc: 0.654104
[280]	training's auc: 0.786886	valid_1's auc: 0.654312
[290]	training's auc: 0.787557	valid_1's auc: 0.654586
[300]	training's auc: 0.788214	valid_1's auc: 0.654684
[310]	training's auc: 0.788915	valid_1's auc: 0.654861
[320]	training's auc: 0.789709	valid_1's auc: 0.654979
[330]	training's auc: 0.790346	valid_1's auc: 0.655067
[340]	training's auc: 0.791073	valid_1's auc: 0.655283
[350]	training's auc: 0.791612	valid_1's auc: 0.655428
[360]	training's auc: 0.792105	valid_1's auc: 0.655537
[370]	training's auc: 0.792815	valid_1's auc: 0.65563
[380]	training's auc: 0.793266	valid_1's auc: 0.655677
[390]	training's auc: 0.793909	valid_1's auc: 0.655819
[400]	training's auc: 0.795591	valid_1's auc: 0.656393
[410]	training's auc: 0.796601	valid_1's auc: 0.65652
[420]	training's auc: 0.797267	valid_1's auc: 0.656546
[430]	training's auc: 0.797825	valid_1's auc: 0.656704
[440]	training's auc: 0.798413	valid_1's auc: 0.656804
[450]	training's auc: 0.798882	valid_1's auc: 0.656878
[460]	training's auc: 0.799304	valid_1's auc: 0.657001
[470]	training's auc: 0.799701	valid_1's auc: 0.657021
[480]	training's auc: 0.800548	valid_1's auc: 0.657157
[490]	training's auc: 0.80099	valid_1's auc: 0.657199
[500]	training's auc: 0.801497	valid_1's auc: 0.657274
[510]	training's auc: 0.80215	valid_1's auc: 0.657381
[520]	training's auc: 0.802849	valid_1's auc: 0.657444
[530]	training's auc: 0.803383	valid_1's auc: 0.657467
[540]	training's auc: 0.803756	valid_1's auc: 0.657555
[550]	training's auc: 0.804168	valid_1's auc: 0.657578
[560]	training's auc: 0.804439	valid_1's auc: 0.657609
[570]	training's auc: 0.804846	valid_1's auc: 0.657625
[580]	training's auc: 0.805232	valid_1's auc: 0.657675
[590]	training's auc: 0.80557	valid_1's auc: 0.657678
[600]	training's auc: 0.806497	valid_1's auc: 0.657906
[610]	training's auc: 0.806793	valid_1's auc: 0.657942
[620]	training's auc: 0.80714	valid_1's auc: 0.657989
[630]	training's auc: 0.807517	valid_1's auc: 0.65805
[640]	training's auc: 0.807875	valid_1's auc: 0.658096
[650]	training's auc: 0.80836	valid_1's auc: 0.658204
[660]	training's auc: 0.808715	valid_1's auc: 0.658204
[670]	training's auc: 0.809197	valid_1's auc: 0.658262
[680]	training's auc: 0.809489	valid_1's auc: 0.658294
[690]	training's auc: 0.809832	valid_1's auc: 0.658319
[700]	training's auc: 0.810231	valid_1's auc: 0.658371
[710]	training's auc: 0.81055	valid_1's auc: 0.658425
[720]	training's auc: 0.810903	valid_1's auc: 0.658402
[730]	training's auc: 0.811169	valid_1's auc: 0.658389
[740]	training's auc: 0.811429	valid_1's auc: 0.658416
[750]	training's auc: 0.811764	valid_1's auc: 0.658392
[760]	training's auc: 0.812007	valid_1's auc: 0.658429
Early stopping, best iteration is:
[712]	training's auc: 0.810614	valid_1's auc: 0.658446
best score: 0.658445904208
best iteration: 712
complete on: source_screen_name

                msno : 5967
             song_id : 4392
         artist_name : 13185
  source_screen_name : 10837
working on: source_type

Our guest selection:
target            uint8
msno           category
song_id        category
artist_name    category
source_type    category
dtype: object
number of columns: 5

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.714822	valid_1's auc: 0.623945
[20]	training's auc: 0.729124	valid_1's auc: 0.630604
[30]	training's auc: 0.736587	valid_1's auc: 0.634015
[40]	training's auc: 0.743044	valid_1's auc: 0.637073
[50]	training's auc: 0.748022	valid_1's auc: 0.63913
[60]	training's auc: 0.75276	valid_1's auc: 0.641149
[70]	training's auc: 0.755871	valid_1's auc: 0.642488
[80]	training's auc: 0.758834	valid_1's auc: 0.643879
[90]	training's auc: 0.761473	valid_1's auc: 0.64533
[100]	training's auc: 0.764092	valid_1's auc: 0.646636
[110]	training's auc: 0.766688	valid_1's auc: 0.647612
[120]	training's auc: 0.76902	valid_1's auc: 0.648506
[130]	training's auc: 0.771191	valid_1's auc: 0.649311
[140]	training's auc: 0.772921	valid_1's auc: 0.650204
[150]	training's auc: 0.774729	valid_1's auc: 0.651075
[160]	training's auc: 0.776528	valid_1's auc: 0.651611
[170]	training's auc: 0.777889	valid_1's auc: 0.652062
[180]	training's auc: 0.779266	valid_1's auc: 0.652532
[190]	training's auc: 0.780431	valid_1's auc: 0.652783
[200]	training's auc: 0.781859	valid_1's auc: 0.653064
[210]	training's auc: 0.782766	valid_1's auc: 0.653398
[220]	training's auc: 0.783845	valid_1's auc: 0.653785
[230]	training's auc: 0.784805	valid_1's auc: 0.654062
[240]	training's auc: 0.78565	valid_1's auc: 0.654216
[250]	training's auc: 0.786714	valid_1's auc: 0.654678
[260]	training's auc: 0.787574	valid_1's auc: 0.654941
[270]	training's auc: 0.788343	valid_1's auc: 0.655131
[280]	training's auc: 0.789176	valid_1's auc: 0.655354
[290]	training's auc: 0.789803	valid_1's auc: 0.655494
[300]	training's auc: 0.790467	valid_1's auc: 0.655641
[310]	training's auc: 0.791118	valid_1's auc: 0.655871
[320]	training's auc: 0.791727	valid_1's auc: 0.656022
[330]	training's auc: 0.792314	valid_1's auc: 0.656234
[340]	training's auc: 0.792967	valid_1's auc: 0.656416
[350]	training's auc: 0.793823	valid_1's auc: 0.656731
[360]	training's auc: 0.794657	valid_1's auc: 0.656976
[370]	training's auc: 0.795305	valid_1's auc: 0.657127
[380]	training's auc: 0.795768	valid_1's auc: 0.657237
[390]	training's auc: 0.796817	valid_1's auc: 0.657433
[400]	training's auc: 0.797603	valid_1's auc: 0.657527
[410]	training's auc: 0.798503	valid_1's auc: 0.657701
[420]	training's auc: 0.79925	valid_1's auc: 0.6578
[430]	training's auc: 0.799755	valid_1's auc: 0.657873
[440]	training's auc: 0.800297	valid_1's auc: 0.658002
[450]	training's auc: 0.800751	valid_1's auc: 0.658036
[460]	training's auc: 0.801168	valid_1's auc: 0.658097
[470]	training's auc: 0.801653	valid_1's auc: 0.658135
[480]	training's auc: 0.802394	valid_1's auc: 0.658272
[490]	training's auc: 0.80281	valid_1's auc: 0.658352
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1103BBBB.py", line 327, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 199, in train
    booster.update(fobj=fobj)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1507, in update
    ctypes.byref(is_finished)))
KeyboardInterrupt

Process finished with exit code 1
'''
