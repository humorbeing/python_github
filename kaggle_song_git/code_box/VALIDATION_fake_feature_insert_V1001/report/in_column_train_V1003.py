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
    df.drop(on_in, axis=1, inplace=True)


for col in cols:
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
    df.drop(colc, axis=1, inplace=True)


load_name = 'train_set'
read_from = '../saves01/'
dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
train = pd.read_csv(read_from+load_name+".csv", dtype=dt)
del dt

train.drop(
    [
        'target',
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

learning_rate = 0.2
num_leaves = 127
max_depth = 10

max_bin = 225
lambda_l1 = 0
lambda_l2 = 0


bagging_fraction = 0.8
bagging_freq = 2
bagging_seed = 1
feature_fraction = 0.9
feature_fraction_seed = 1

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
    'composer',
    'lyricist',
    'song_year',
    'language',
    'rc',
    # 'FAKE_1512883008',
]
result = {}
for w in df.columns:
    print("'{}',".format(w))

work_on = [
    'ITC_msno_log10_1',
    'ITC_song_id_log10_1',
    'ITC_source_system_tab_log10_1',
    'ITC_source_screen_name_log10_1',
    'ITC_source_type_log10_1',
    'ITC_artist_name_log10_1',
    'ITC_composer_log10_1',
    'ITC_lyricist_log10_1',
    'ITC_language_log10_1',
    'ITC_song_year_log10_1',
    'ITC_song_country_log10_1',
    'ITC_rc_log10_1',
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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003.py
What we got:
target                               uint8
ITC_msno_log10_1                   float64
ITC_song_id_log10_1                float64
ITC_source_system_tab_log10_1      float64
ITC_source_screen_name_log10_1     float64
ITC_source_type_log10_1            float64
ITC_gender_log10_1                 float64
ITC_artist_name_log10_1            float64
ITC_composer_log10_1               float64
ITC_lyricist_log10_1               float64
ITC_language_log10_1               float64
ITC_name_log10_1                   float64
ITC_song_year_log10_1              float64
ITC_song_country_log10_1           float64
ITC_rc_log10_1                     float64
ITC_isrc_rest_log10_1              float64
ITC_top1_in_song_log10_1           float64
ITC_top2_in_song_log10_1           float64
ITC_top3_in_song_log10_1           float64
msno                                object
song_id                             object
source_system_tab                   object
source_screen_name                  object
source_type                         object
expiration_month                  category
genre_ids                           object
artist_name                         object
composer                            object
lyricist                            object
language                          category
name                                object
song_year                         category
song_country                      category
rc                                category
isrc_rest                         category
top1_in_song                      category
top2_in_song                      category
top3_in_song                      category
dtype: object
number of rows: 7377418
number of columns: 38
'target',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
'ITC_gender_log10_1',
'ITC_artist_name_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_language_log10_1',
'ITC_name_log10_1',
'ITC_song_year_log10_1',
'ITC_song_country_log10_1',
'ITC_rc_log10_1',
'ITC_isrc_rest_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'expiration_month',
'genre_ids',
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
working on: ITC_msno_log10_1
/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003.py:218: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')

Our guest selection:
target                   uint8
msno                  category
song_id               category
source_system_tab     category
source_screen_name    category
source_type           category
artist_name           category
composer              category
lyricist              category
song_year             category
language              category
rc                    category
ITC_msno_log10_1       float64
dtype: object
number of columns: 13

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
[10]	training's auc: 0.745212	valid_1's auc: 0.641753
[20]	training's auc: 0.760215	valid_1's auc: 0.648575
[30]	training's auc: 0.766634	valid_1's auc: 0.651935
[40]	training's auc: 0.771981	valid_1's auc: 0.654262
[50]	training's auc: 0.776228	valid_1's auc: 0.655967
[60]	training's auc: 0.779441	valid_1's auc: 0.657112
[70]	training's auc: 0.782472	valid_1's auc: 0.658347
[80]	training's auc: 0.786091	valid_1's auc: 0.659391
[90]	training's auc: 0.788465	valid_1's auc: 0.660361
[100]	training's auc: 0.790455	valid_1's auc: 0.661006
[110]	training's auc: 0.792264	valid_1's auc: 0.66182
[120]	training's auc: 0.79403	valid_1's auc: 0.662557
[130]	training's auc: 0.795774	valid_1's auc: 0.66312
[140]	training's auc: 0.797276	valid_1's auc: 0.663681
[150]	training's auc: 0.798683	valid_1's auc: 0.663984
[160]	training's auc: 0.800384	valid_1's auc: 0.664482
[170]	training's auc: 0.801645	valid_1's auc: 0.664842
[180]	training's auc: 0.802779	valid_1's auc: 0.665161
[190]	training's auc: 0.804033	valid_1's auc: 0.665463
[200]	training's auc: 0.805364	valid_1's auc: 0.665816
[210]	training's auc: 0.806619	valid_1's auc: 0.666039
[220]	training's auc: 0.80784	valid_1's auc: 0.666378
[230]	training's auc: 0.808641	valid_1's auc: 0.666517
[240]	training's auc: 0.80964	valid_1's auc: 0.666585
[250]	training's auc: 0.810545	valid_1's auc: 0.6669
[260]	training's auc: 0.811337	valid_1's auc: 0.667039
[270]	training's auc: 0.812683	valid_1's auc: 0.667479
[280]	training's auc: 0.813433	valid_1's auc: 0.667636
[290]	training's auc: 0.814258	valid_1's auc: 0.667798
[300]	training's auc: 0.815344	valid_1's auc: 0.668083
[310]	training's auc: 0.816034	valid_1's auc: 0.668334
[320]	training's auc: 0.816702	valid_1's auc: 0.66838
[330]	training's auc: 0.817315	valid_1's auc: 0.668407
[340]	training's auc: 0.817931	valid_1's auc: 0.668488
[350]	training's auc: 0.818617	valid_1's auc: 0.668624
[360]	training's auc: 0.819501	valid_1's auc: 0.668697
[370]	training's auc: 0.820213	valid_1's auc: 0.668742
[380]	training's auc: 0.820817	valid_1's auc: 0.668806
[390]	training's auc: 0.821521	valid_1's auc: 0.668972
[400]	training's auc: 0.822078	valid_1's auc: 0.669084
[410]	training's auc: 0.822577	valid_1's auc: 0.669131
[420]	training's auc: 0.823398	valid_1's auc: 0.669365
[430]	training's auc: 0.823952	valid_1's auc: 0.669367
[440]	training's auc: 0.824392	valid_1's auc: 0.669413
[450]	training's auc: 0.824957	valid_1's auc: 0.669518
[460]	training's auc: 0.825382	valid_1's auc: 0.669623
[470]	training's auc: 0.825858	valid_1's auc: 0.66966
[480]	training's auc: 0.826366	valid_1's auc: 0.669706
[490]	training's auc: 0.826762	valid_1's auc: 0.669759
[500]	training's auc: 0.82758	valid_1's auc: 0.66991
[510]	training's auc: 0.828204	valid_1's auc: 0.669977
[520]	training's auc: 0.828671	valid_1's auc: 0.670036
[530]	training's auc: 0.829191	valid_1's auc: 0.670078
[540]	training's auc: 0.829809	valid_1's auc: 0.670243
[550]	training's auc: 0.830203	valid_1's auc: 0.67027
[560]	training's auc: 0.830637	valid_1's auc: 0.670373
[570]	training's auc: 0.83098	valid_1's auc: 0.670321
[580]	training's auc: 0.831455	valid_1's auc: 0.670358
[590]	training's auc: 0.831961	valid_1's auc: 0.670365
[600]	training's auc: 0.832323	valid_1's auc: 0.670471
[610]	training's auc: 0.832656	valid_1's auc: 0.670531
[620]	training's auc: 0.833078	valid_1's auc: 0.670617
[630]	training's auc: 0.833383	valid_1's auc: 0.670629
[640]	training's auc: 0.833838	valid_1's auc: 0.670727
[650]	training's auc: 0.834213	valid_1's auc: 0.67074
[660]	training's auc: 0.834518	valid_1's auc: 0.670735
[670]	training's auc: 0.83482	valid_1's auc: 0.670759
[680]	training's auc: 0.835325	valid_1's auc: 0.6709
[690]	training's auc: 0.835623	valid_1's auc: 0.670907
[700]	training's auc: 0.835952	valid_1's auc: 0.670918
[710]	training's auc: 0.836189	valid_1's auc: 0.670906
[720]	training's auc: 0.836528	valid_1's auc: 0.670992
[730]	training's auc: 0.836776	valid_1's auc: 0.670983
[740]	training's auc: 0.837048	valid_1's auc: 0.670977
[750]	training's auc: 0.837301	valid_1's auc: 0.670997
[760]	training's auc: 0.837631	valid_1's auc: 0.671015
[770]	training's auc: 0.837864	valid_1's auc: 0.671056
[780]	training's auc: 0.838141	valid_1's auc: 0.671077
[790]	training's auc: 0.838333	valid_1's auc: 0.671117
[800]	training's auc: 0.838602	valid_1's auc: 0.671136
[810]	training's auc: 0.838992	valid_1's auc: 0.671174
[820]	training's auc: 0.839245	valid_1's auc: 0.671121
[830]	training's auc: 0.839485	valid_1's auc: 0.671148
[840]	training's auc: 0.83969	valid_1's auc: 0.671138
[850]	training's auc: 0.839906	valid_1's auc: 0.671129
[860]	training's auc: 0.840135	valid_1's auc: 0.671154
[870]	training's auc: 0.840369	valid_1's auc: 0.671211
[880]	training's auc: 0.840557	valid_1's auc: 0.671135
[890]	training's auc: 0.840805	valid_1's auc: 0.671146
[900]	training's auc: 0.841065	valid_1's auc: 0.671151
[910]	training's auc: 0.841291	valid_1's auc: 0.67117
[920]	training's auc: 0.841605	valid_1's auc: 0.671176
Early stopping, best iteration is:
[871]	training's auc: 0.840381	valid_1's auc: 0.671216
best score: 0.671215535927
best iteration: 871
complete on: ITC_msno_log10_1

working on: ITC_song_id_log10_1

Our guest selection:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
composer               category
lyricist               category
song_year              category
language               category
rc                     category
ITC_song_id_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.74717	valid_1's auc: 0.65031
[20]	training's auc: 0.759769	valid_1's auc: 0.655903
[30]	training's auc: 0.766165	valid_1's auc: 0.658848
[40]	training's auc: 0.770613	valid_1's auc: 0.6611
[50]	training's auc: 0.774886	valid_1's auc: 0.662815
[60]	training's auc: 0.778389	valid_1's auc: 0.66407
[70]	training's auc: 0.781567	valid_1's auc: 0.665319
[80]	training's auc: 0.784018	valid_1's auc: 0.666253
[90]	training's auc: 0.786514	valid_1's auc: 0.667067
[100]	training's auc: 0.789151	valid_1's auc: 0.66795
[110]	training's auc: 0.790922	valid_1's auc: 0.668556
[120]	training's auc: 0.792687	valid_1's auc: 0.669145
[130]	training's auc: 0.794576	valid_1's auc: 0.669598
[140]	training's auc: 0.795959	valid_1's auc: 0.670005
[150]	training's auc: 0.797513	valid_1's auc: 0.670534
[160]	training's auc: 0.799037	valid_1's auc: 0.670888
[170]	training's auc: 0.800258	valid_1's auc: 0.671159
[180]	training's auc: 0.801553	valid_1's auc: 0.671541
[190]	training's auc: 0.8028	valid_1's auc: 0.671888
[200]	training's auc: 0.804003	valid_1's auc: 0.672146
[210]	training's auc: 0.805409	valid_1's auc: 0.672446
[220]	training's auc: 0.806433	valid_1's auc: 0.672666
[230]	training's auc: 0.807392	valid_1's auc: 0.672852
[240]	training's auc: 0.808278	valid_1's auc: 0.67298
[250]	training's auc: 0.809131	valid_1's auc: 0.673104
[260]	training's auc: 0.810048	valid_1's auc: 0.673293
[270]	training's auc: 0.810825	valid_1's auc: 0.673422
[280]	training's auc: 0.811538	valid_1's auc: 0.673534
[290]	training's auc: 0.812902	valid_1's auc: 0.674032
[300]	training's auc: 0.813829	valid_1's auc: 0.674117
[310]	training's auc: 0.814482	valid_1's auc: 0.674263
[320]	training's auc: 0.815238	valid_1's auc: 0.674393
[330]	training's auc: 0.815867	valid_1's auc: 0.674499
[340]	training's auc: 0.816479	valid_1's auc: 0.674543
[350]	training's auc: 0.817582	valid_1's auc: 0.674962
[360]	training's auc: 0.818284	valid_1's auc: 0.675014
[370]	training's auc: 0.81913	valid_1's auc: 0.675137
[380]	training's auc: 0.819765	valid_1's auc: 0.675124
[390]	training's auc: 0.820342	valid_1's auc: 0.675175
[400]	training's auc: 0.820975	valid_1's auc: 0.675207
[410]	training's auc: 0.821533	valid_1's auc: 0.675186
[420]	training's auc: 0.82214	valid_1's auc: 0.675256
[430]	training's auc: 0.822603	valid_1's auc: 0.675171
[440]	training's auc: 0.82312	valid_1's auc: 0.675203
[450]	training's auc: 0.823631	valid_1's auc: 0.675284
[460]	training's auc: 0.82413	valid_1's auc: 0.675371
[470]	training's auc: 0.824563	valid_1's auc: 0.67536
[480]	training's auc: 0.824997	valid_1's auc: 0.675298
[490]	training's auc: 0.825497	valid_1's auc: 0.675355
[500]	training's auc: 0.826038	valid_1's auc: 0.675355
Early stopping, best iteration is:
[459]	training's auc: 0.824094	valid_1's auc: 0.675386
best score: 0.67538557413
best iteration: 459
complete on: ITC_song_id_log10_1

working on: ITC_source_system_tab_log10_1

Our guest selection:
target                              uint8
msno                             category
song_id                          category
source_system_tab                category
source_screen_name               category
source_type                      category
artist_name                      category
composer                         category
lyricist                         category
song_year                        category
language                         category
rc                               category
ITC_source_system_tab_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.734722	valid_1's auc: 0.638187
[20]	training's auc: 0.748792	valid_1's auc: 0.64411
[30]	training's auc: 0.756259	valid_1's auc: 0.647217
[40]	training's auc: 0.761144	valid_1's auc: 0.648949
[50]	training's auc: 0.765552	valid_1's auc: 0.650611
[60]	training's auc: 0.770235	valid_1's auc: 0.652299
[70]	training's auc: 0.773604	valid_1's auc: 0.653569
[80]	training's auc: 0.776764	valid_1's auc: 0.654489
[90]	training's auc: 0.77926	valid_1's auc: 0.655415
[100]	training's auc: 0.781419	valid_1's auc: 0.656237
[110]	training's auc: 0.783165	valid_1's auc: 0.656987
[120]	training's auc: 0.784956	valid_1's auc: 0.657625
[130]	training's auc: 0.78698	valid_1's auc: 0.658282
[140]	training's auc: 0.788759	valid_1's auc: 0.658827
[150]	training's auc: 0.790279	valid_1's auc: 0.659374
[160]	training's auc: 0.791853	valid_1's auc: 0.659866
[170]	training's auc: 0.793271	valid_1's auc: 0.660196
[180]	training's auc: 0.794562	valid_1's auc: 0.660497
[190]	training's auc: 0.796048	valid_1's auc: 0.661003
[200]	training's auc: 0.797335	valid_1's auc: 0.661215
[210]	training's auc: 0.798516	valid_1's auc: 0.661429
[220]	training's auc: 0.79963	valid_1's auc: 0.661869
[230]	training's auc: 0.800698	valid_1's auc: 0.662037
[240]	training's auc: 0.801728	valid_1's auc: 0.662189
[250]	training's auc: 0.80251	valid_1's auc: 0.662443
[260]	training's auc: 0.803406	valid_1's auc: 0.662536
[270]	training's auc: 0.804313	valid_1's auc: 0.662789
[280]	training's auc: 0.805134	valid_1's auc: 0.662898
[290]	training's auc: 0.806223	valid_1's auc: 0.663056
[300]	training's auc: 0.807225	valid_1's auc: 0.663248
[310]	training's auc: 0.808053	valid_1's auc: 0.663477
[320]	training's auc: 0.808853	valid_1's auc: 0.663567
[330]	training's auc: 0.809474	valid_1's auc: 0.6638
[340]	training's auc: 0.810215	valid_1's auc: 0.663968
[350]	training's auc: 0.811077	valid_1's auc: 0.664192
[360]	training's auc: 0.812055	valid_1's auc: 0.664433
[370]	training's auc: 0.812722	valid_1's auc: 0.664474
[380]	training's auc: 0.813632	valid_1's auc: 0.664682
[390]	training's auc: 0.814335	valid_1's auc: 0.664796
[400]	training's auc: 0.814894	valid_1's auc: 0.664965
[410]	training's auc: 0.815654	valid_1's auc: 0.665116
[420]	training's auc: 0.816168	valid_1's auc: 0.665219
[430]	training's auc: 0.816676	valid_1's auc: 0.665356
[440]	training's auc: 0.817193	valid_1's auc: 0.665485
[450]	training's auc: 0.81798	valid_1's auc: 0.665672
[460]	training's auc: 0.818444	valid_1's auc: 0.665761
[470]	training's auc: 0.818903	valid_1's auc: 0.66578
[480]	training's auc: 0.81933	valid_1's auc: 0.665832
[490]	training's auc: 0.820292	valid_1's auc: 0.666077
[500]	training's auc: 0.820853	valid_1's auc: 0.666161
[510]	training's auc: 0.821522	valid_1's auc: 0.666289
[520]	training's auc: 0.822024	valid_1's auc: 0.666375
[530]	training's auc: 0.822423	valid_1's auc: 0.666417
[540]	training's auc: 0.822876	valid_1's auc: 0.666479
[550]	training's auc: 0.823307	valid_1's auc: 0.66649
[560]	training's auc: 0.823836	valid_1's auc: 0.666593
[570]	training's auc: 0.824283	valid_1's auc: 0.666671
[580]	training's auc: 0.824718	valid_1's auc: 0.666679
[590]	training's auc: 0.825182	valid_1's auc: 0.666743
[600]	training's auc: 0.825504	valid_1's auc: 0.666806
[610]	training's auc: 0.825864	valid_1's auc: 0.666805
[620]	training's auc: 0.826181	valid_1's auc: 0.666857
[630]	training's auc: 0.82657	valid_1's auc: 0.666984
[640]	training's auc: 0.826886	valid_1's auc: 0.666995
[650]	training's auc: 0.827181	valid_1's auc: 0.666992
[660]	training's auc: 0.827531	valid_1's auc: 0.667051
[670]	training's auc: 0.827787	valid_1's auc: 0.667103
[680]	training's auc: 0.828109	valid_1's auc: 0.667065
[690]	training's auc: 0.828626	valid_1's auc: 0.667208
[700]	training's auc: 0.829129	valid_1's auc: 0.667309
[710]	training's auc: 0.829426	valid_1's auc: 0.667387
[720]	training's auc: 0.829639	valid_1's auc: 0.66726
[730]	training's auc: 0.829904	valid_1's auc: 0.667271
[740]	training's auc: 0.83018	valid_1's auc: 0.667274
[750]	training's auc: 0.83049	valid_1's auc: 0.667355
[760]	training's auc: 0.830808	valid_1's auc: 0.667384
Early stopping, best iteration is:
[713]	training's auc: 0.829486	valid_1's auc: 0.667406
best score: 0.667405504199
best iteration: 713
complete on: ITC_source_system_tab_log10_1

working on: ITC_source_screen_name_log10_1

Our guest selection:
target                               uint8
msno                              category
song_id                           category
source_system_tab                 category
source_screen_name                category
source_type                       category
artist_name                       category
composer                          category
lyricist                          category
song_year                         category
language                          category
rc                                category
ITC_source_screen_name_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.733926	valid_1's auc: 0.637419
[20]	training's auc: 0.749411	valid_1's auc: 0.644273
[30]	training's auc: 0.75673	valid_1's auc: 0.647414
[40]	training's auc: 0.761636	valid_1's auc: 0.649891
[50]	training's auc: 0.766015	valid_1's auc: 0.651759
[60]	training's auc: 0.769707	valid_1's auc: 0.65314
[70]	training's auc: 0.772911	valid_1's auc: 0.654485
[80]	training's auc: 0.776376	valid_1's auc: 0.655211
[90]	training's auc: 0.778978	valid_1's auc: 0.65613
[100]	training's auc: 0.781273	valid_1's auc: 0.656852
[110]	training's auc: 0.783026	valid_1's auc: 0.657514
[120]	training's auc: 0.785227	valid_1's auc: 0.658371
[130]	training's auc: 0.787062	valid_1's auc: 0.659025
[140]	training's auc: 0.788602	valid_1's auc: 0.65967
[150]	training's auc: 0.790124	valid_1's auc: 0.660139
[160]	training's auc: 0.791693	valid_1's auc: 0.660524
[170]	training's auc: 0.793091	valid_1's auc: 0.660845
[180]	training's auc: 0.794384	valid_1's auc: 0.661153
[190]	training's auc: 0.795587	valid_1's auc: 0.661358
[200]	training's auc: 0.796869	valid_1's auc: 0.661717
[210]	training's auc: 0.798298	valid_1's auc: 0.662303
[220]	training's auc: 0.799316	valid_1's auc: 0.662613
[230]	training's auc: 0.800285	valid_1's auc: 0.662872
[240]	training's auc: 0.801627	valid_1's auc: 0.66328
[250]	training's auc: 0.802558	valid_1's auc: 0.663531
[260]	training's auc: 0.803431	valid_1's auc: 0.663706
[270]	training's auc: 0.804366	valid_1's auc: 0.663878
[280]	training's auc: 0.805558	valid_1's auc: 0.664185
[290]	training's auc: 0.806354	valid_1's auc: 0.664313
[300]	training's auc: 0.807346	valid_1's auc: 0.664538
[310]	training's auc: 0.808153	valid_1's auc: 0.664612
[320]	training's auc: 0.809265	valid_1's auc: 0.664945
[330]	training's auc: 0.809959	valid_1's auc: 0.665167
[340]	training's auc: 0.810618	valid_1's auc: 0.665162
[350]	training's auc: 0.811285	valid_1's auc: 0.665303
[360]	training's auc: 0.811977	valid_1's auc: 0.665444
[370]	training's auc: 0.813099	valid_1's auc: 0.665805
[380]	training's auc: 0.813706	valid_1's auc: 0.665829
[390]	training's auc: 0.814428	valid_1's auc: 0.666013
[400]	training's auc: 0.815132	valid_1's auc: 0.666088
[410]	training's auc: 0.815695	valid_1's auc: 0.666131
[420]	training's auc: 0.81649	valid_1's auc: 0.666264
[430]	training's auc: 0.817073	valid_1's auc: 0.666353
[440]	training's auc: 0.817894	valid_1's auc: 0.666469
[450]	training's auc: 0.818703	valid_1's auc: 0.666606
[460]	training's auc: 0.819299	valid_1's auc: 0.666702
[470]	training's auc: 0.819755	valid_1's auc: 0.666677
[480]	training's auc: 0.820551	valid_1's auc: 0.66692
[490]	training's auc: 0.821052	valid_1's auc: 0.666978
[500]	training's auc: 0.821518	valid_1's auc: 0.666995
[510]	training's auc: 0.822167	valid_1's auc: 0.667116
[520]	training's auc: 0.822618	valid_1's auc: 0.667138
[530]	training's auc: 0.823261	valid_1's auc: 0.667238
[540]	training's auc: 0.823813	valid_1's auc: 0.667274
[550]	training's auc: 0.824474	valid_1's auc: 0.6674
[560]	training's auc: 0.825011	valid_1's auc: 0.66754
[570]	training's auc: 0.825378	valid_1's auc: 0.667619
[580]	training's auc: 0.825784	valid_1's auc: 0.667594
[590]	training's auc: 0.826283	valid_1's auc: 0.667606
[600]	training's auc: 0.8267	valid_1's auc: 0.667553
[610]	training's auc: 0.827049	valid_1's auc: 0.667505
[620]	training's auc: 0.827533	valid_1's auc: 0.66756
Early stopping, best iteration is:
[570]	training's auc: 0.825378	valid_1's auc: 0.667619
best score: 0.667619499267
best iteration: 570
complete on: ITC_source_screen_name_log10_1

working on: ITC_source_type_log10_1

Our guest selection:
target                        uint8
msno                       category
song_id                    category
source_system_tab          category
source_screen_name         category
source_type                category
artist_name                category
composer                   category
lyricist                   category
song_year                  category
language                   category
rc                         category
ITC_source_type_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.735314	valid_1's auc: 0.637089
[20]	training's auc: 0.748977	valid_1's auc: 0.643801
[30]	training's auc: 0.75645	valid_1's auc: 0.647253
[40]	training's auc: 0.761562	valid_1's auc: 0.649467
[50]	training's auc: 0.765917	valid_1's auc: 0.651102
[60]	training's auc: 0.769711	valid_1's auc: 0.652553
[70]	training's auc: 0.773095	valid_1's auc: 0.65403
[80]	training's auc: 0.776007	valid_1's auc: 0.654761
[90]	training's auc: 0.778614	valid_1's auc: 0.655592
[100]	training's auc: 0.780921	valid_1's auc: 0.656202
[110]	training's auc: 0.782799	valid_1's auc: 0.657012
[120]	training's auc: 0.784783	valid_1's auc: 0.657732
[130]	training's auc: 0.786674	valid_1's auc: 0.658376
[140]	training's auc: 0.788303	valid_1's auc: 0.658825
[150]	training's auc: 0.789927	valid_1's auc: 0.65949
[160]	training's auc: 0.791975	valid_1's auc: 0.660069
[170]	training's auc: 0.793161	valid_1's auc: 0.660439
[180]	training's auc: 0.794472	valid_1's auc: 0.66076
[190]	training's auc: 0.795752	valid_1's auc: 0.661145
[200]	training's auc: 0.797597	valid_1's auc: 0.66144
[210]	training's auc: 0.798874	valid_1's auc: 0.661755
[220]	training's auc: 0.79978	valid_1's auc: 0.66207
[230]	training's auc: 0.800825	valid_1's auc: 0.66232
[240]	training's auc: 0.801961	valid_1's auc: 0.662629
[250]	training's auc: 0.803034	valid_1's auc: 0.663078
[260]	training's auc: 0.803821	valid_1's auc: 0.66322
[270]	training's auc: 0.804678	valid_1's auc: 0.663481
[280]	training's auc: 0.80572	valid_1's auc: 0.663687
[290]	training's auc: 0.806797	valid_1's auc: 0.663971
[300]	training's auc: 0.807532	valid_1's auc: 0.664116
[310]	training's auc: 0.808197	valid_1's auc: 0.664298
[320]	training's auc: 0.809339	valid_1's auc: 0.664659
[330]	training's auc: 0.80997	valid_1's auc: 0.664834
[340]	training's auc: 0.810614	valid_1's auc: 0.665019
[350]	training's auc: 0.811411	valid_1's auc: 0.665231
[360]	training's auc: 0.812047	valid_1's auc: 0.665246
[370]	training's auc: 0.812987	valid_1's auc: 0.665473
[380]	training's auc: 0.814511	valid_1's auc: 0.665872
[390]	training's auc: 0.81523	valid_1's auc: 0.666046
[400]	training's auc: 0.815748	valid_1's auc: 0.666087
[410]	training's auc: 0.816264	valid_1's auc: 0.66618
[420]	training's auc: 0.816767	valid_1's auc: 0.666244
[430]	training's auc: 0.817703	valid_1's auc: 0.666432
[440]	training's auc: 0.818343	valid_1's auc: 0.666649
[450]	training's auc: 0.818987	valid_1's auc: 0.666785
[460]	training's auc: 0.819463	valid_1's auc: 0.666875
[470]	training's auc: 0.819929	valid_1's auc: 0.666984
[480]	training's auc: 0.82032	valid_1's auc: 0.666988
[490]	training's auc: 0.820786	valid_1's auc: 0.666997
[500]	training's auc: 0.82141	valid_1's auc: 0.6671
[510]	training's auc: 0.822055	valid_1's auc: 0.66714
[520]	training's auc: 0.822604	valid_1's auc: 0.667242
[530]	training's auc: 0.823295	valid_1's auc: 0.667335
[540]	training's auc: 0.823704	valid_1's auc: 0.667371
[550]	training's auc: 0.824264	valid_1's auc: 0.667498
[560]	training's auc: 0.824722	valid_1's auc: 0.6675
[570]	training's auc: 0.825086	valid_1's auc: 0.667529
[580]	training's auc: 0.825549	valid_1's auc: 0.667565
[590]	training's auc: 0.826077	valid_1's auc: 0.667607
[600]	training's auc: 0.826412	valid_1's auc: 0.667577
[610]	training's auc: 0.826781	valid_1's auc: 0.667582
[620]	training's auc: 0.827268	valid_1's auc: 0.667693
[630]	training's auc: 0.827666	valid_1's auc: 0.667778
[640]	training's auc: 0.827998	valid_1's auc: 0.667824
[650]	training's auc: 0.82842	valid_1's auc: 0.667948
[660]	training's auc: 0.828736	valid_1's auc: 0.668015
[670]	training's auc: 0.829054	valid_1's auc: 0.668037
[680]	training's auc: 0.829385	valid_1's auc: 0.668066
[690]	training's auc: 0.829779	valid_1's auc: 0.668119
[700]	training's auc: 0.830057	valid_1's auc: 0.668145
[710]	training's auc: 0.830285	valid_1's auc: 0.668124
[720]	training's auc: 0.830512	valid_1's auc: 0.668186
[730]	training's auc: 0.830772	valid_1's auc: 0.668224
[740]	training's auc: 0.83109	valid_1's auc: 0.668262
[750]	training's auc: 0.831353	valid_1's auc: 0.668299
[760]	training's auc: 0.83166	valid_1's auc: 0.668245
[770]	training's auc: 0.83195	valid_1's auc: 0.668284
[780]	training's auc: 0.832251	valid_1's auc: 0.668284
[790]	training's auc: 0.832455	valid_1's auc: 0.668321
[800]	training's auc: 0.832724	valid_1's auc: 0.66844
[810]	training's auc: 0.832956	valid_1's auc: 0.6684
[820]	training's auc: 0.833191	valid_1's auc: 0.668429
[830]	training's auc: 0.833624	valid_1's auc: 0.668499
[840]	training's auc: 0.833899	valid_1's auc: 0.668489
[850]	training's auc: 0.834154	valid_1's auc: 0.668513
[860]	training's auc: 0.834404	valid_1's auc: 0.668572
[870]	training's auc: 0.834649	valid_1's auc: 0.6686
[880]	training's auc: 0.834901	valid_1's auc: 0.668663
[890]	training's auc: 0.835151	valid_1's auc: 0.668706
[900]	training's auc: 0.835384	valid_1's auc: 0.668717
[910]	training's auc: 0.835613	valid_1's auc: 0.668715
[920]	training's auc: 0.835858	valid_1's auc: 0.668764
[930]	training's auc: 0.836129	valid_1's auc: 0.668756
[940]	training's auc: 0.836317	valid_1's auc: 0.668771
[950]	training's auc: 0.836573	valid_1's auc: 0.668784
[960]	training's auc: 0.836821	valid_1's auc: 0.668852
[970]	training's auc: 0.837059	valid_1's auc: 0.668847
[980]	training's auc: 0.837263	valid_1's auc: 0.6688
[990]	training's auc: 0.837437	valid_1's auc: 0.668847
[1000]	training's auc: 0.837655	valid_1's auc: 0.668868
[1010]	training's auc: 0.837871	valid_1's auc: 0.668865
[1020]	training's auc: 0.8381	valid_1's auc: 0.668857
[1030]	training's auc: 0.838345	valid_1's auc: 0.668833
[1040]	training's auc: 0.838573	valid_1's auc: 0.668819
[1050]	training's auc: 0.838772	valid_1's auc: 0.668868
Early stopping, best iteration is:
[1004]	training's auc: 0.837744	valid_1's auc: 0.66889
best score: 0.668890269839
best iteration: 1004
complete on: ITC_source_type_log10_1

working on: ITC_artist_name_log10_1

Our guest selection:
target                        uint8
msno                       category
song_id                    category
source_system_tab          category
source_screen_name         category
source_type                category
artist_name                category
composer                   category
lyricist                   category
song_year                  category
language                   category
rc                         category
ITC_artist_name_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.737873	valid_1's auc: 0.639382
[20]	training's auc: 0.75354	valid_1's auc: 0.645839
[30]	training's auc: 0.760618	valid_1's auc: 0.649311
[40]	training's auc: 0.765705	valid_1's auc: 0.65173
[50]	training's auc: 0.769958	valid_1's auc: 0.653391
[60]	training's auc: 0.773295	valid_1's auc: 0.654649
[70]	training's auc: 0.776437	valid_1's auc: 0.655831
[80]	training's auc: 0.77939	valid_1's auc: 0.656865
[90]	training's auc: 0.781855	valid_1's auc: 0.657802
[100]	training's auc: 0.784006	valid_1's auc: 0.658536
[110]	training's auc: 0.786002	valid_1's auc: 0.659327
[120]	training's auc: 0.787907	valid_1's auc: 0.660046
[130]	training's auc: 0.789543	valid_1's auc: 0.660442
[140]	training's auc: 0.791174	valid_1's auc: 0.660999
[150]	training's auc: 0.792691	valid_1's auc: 0.661424
[160]	training's auc: 0.794266	valid_1's auc: 0.661892
[170]	training's auc: 0.795646	valid_1's auc: 0.662193
[180]	training's auc: 0.796872	valid_1's auc: 0.662593
[190]	training's auc: 0.798137	valid_1's auc: 0.662856
[200]	training's auc: 0.79934	valid_1's auc: 0.663134
[210]	training's auc: 0.800549	valid_1's auc: 0.663434
[220]	training's auc: 0.801633	valid_1's auc: 0.663725
[230]	training's auc: 0.802716	valid_1's auc: 0.663952
[240]	training's auc: 0.803838	valid_1's auc: 0.66425
[250]	training's auc: 0.804645	valid_1's auc: 0.664468
[260]	training's auc: 0.805647	valid_1's auc: 0.664616
[270]	training's auc: 0.806519	valid_1's auc: 0.664726
[280]	training's auc: 0.807241	valid_1's auc: 0.665023
[290]	training's auc: 0.808036	valid_1's auc: 0.665135
[300]	training's auc: 0.808938	valid_1's auc: 0.665201
[310]	training's auc: 0.809665	valid_1's auc: 0.665401
[320]	training's auc: 0.810758	valid_1's auc: 0.665676
[330]	training's auc: 0.811668	valid_1's auc: 0.66599
[340]	training's auc: 0.812487	valid_1's auc: 0.666178
[350]	training's auc: 0.813668	valid_1's auc: 0.666429
[360]	training's auc: 0.814345	valid_1's auc: 0.666543
[370]	training's auc: 0.81537	valid_1's auc: 0.666842
[380]	training's auc: 0.816262	valid_1's auc: 0.666971
[390]	training's auc: 0.816912	valid_1's auc: 0.667053
[400]	training's auc: 0.817531	valid_1's auc: 0.6671
[410]	training's auc: 0.818111	valid_1's auc: 0.66703
[420]	training's auc: 0.818761	valid_1's auc: 0.667053
[430]	training's auc: 0.819234	valid_1's auc: 0.667116
[440]	training's auc: 0.81971	valid_1's auc: 0.667166
[450]	training's auc: 0.82051	valid_1's auc: 0.667265
[460]	training's auc: 0.820929	valid_1's auc: 0.667331
[470]	training's auc: 0.821422	valid_1's auc: 0.667334
[480]	training's auc: 0.821927	valid_1's auc: 0.667437
[490]	training's auc: 0.822468	valid_1's auc: 0.667479
[500]	training's auc: 0.822978	valid_1's auc: 0.667526
[510]	training's auc: 0.823426	valid_1's auc: 0.66757
[520]	training's auc: 0.823818	valid_1's auc: 0.667595
[530]	training's auc: 0.824291	valid_1's auc: 0.667605
[540]	training's auc: 0.824944	valid_1's auc: 0.667701
[550]	training's auc: 0.825398	valid_1's auc: 0.667765
[560]	training's auc: 0.825858	valid_1's auc: 0.667739
[570]	training's auc: 0.826234	valid_1's auc: 0.667778
[580]	training's auc: 0.826665	valid_1's auc: 0.667812
[590]	training's auc: 0.827284	valid_1's auc: 0.667953
[600]	training's auc: 0.827603	valid_1's auc: 0.667988
[610]	training's auc: 0.828222	valid_1's auc: 0.668068
[620]	training's auc: 0.828569	valid_1's auc: 0.668117
[630]	training's auc: 0.828903	valid_1's auc: 0.668076
[640]	training's auc: 0.829243	valid_1's auc: 0.66806
[650]	training's auc: 0.829672	valid_1's auc: 0.667952
[660]	training's auc: 0.829984	valid_1's auc: 0.667991
Early stopping, best iteration is:
[619]	training's auc: 0.828543	valid_1's auc: 0.668129
best score: 0.668129215836
best iteration: 619
complete on: ITC_artist_name_log10_1

working on: ITC_composer_log10_1

Our guest selection:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
composer                category
lyricist                category
song_year               category
language                category
rc                      category
ITC_composer_log10_1     float64
dtype: object
number of columns: 13

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003.py", line 279, in <module>
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