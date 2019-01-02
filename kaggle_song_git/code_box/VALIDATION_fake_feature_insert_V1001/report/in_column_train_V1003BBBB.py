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
    df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
    col1 = 'CC11_'+col
    df['OinC_'+col] = df[col1]/df[colc]
    # df.drop(colc, axis=1, inplace=True)


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

num_boost_round = 4000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.1
num_leaves = 127
max_depth = 10

max_bin = 225
lambda_l1 = 0.1
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
    # 'language',
    # 'rc',
    # 'FAKE_1512883008',
]
result = {}
for w in df.columns:
    print("'{}',".format(w))

# work_on = [
#     'ITC_msno_log10_1',
#     'ITC_song_id_log10_1',
#     'ITC_source_system_tab_log10_1',
#     'ITC_source_screen_name_log10_1',
#     'ITC_source_type_log10_1',
#     'ITC_artist_name_log10_1',
#     'ITC_composer_log10_1',
#     'ITC_lyricist_log10_1',
#     'ITC_language_log10_1',
#     'ITC_song_year_log10_1',
#     'ITC_song_country_log10_1',
#     'ITC_rc_log10_1',
# ]
for w in df.columns:
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003BBBB.py
What we got:
target                         uint8
ITC_msno                       int64
CC11_msno                      int64
ITC_song_id                    int64
CC11_song_id                   int64
ITC_source_system_tab          int64
CC11_source_system_tab         int64
ITC_source_screen_name         int64
CC11_source_screen_name        int64
ITC_source_type                int64
CC11_source_type               int64
ITC_gender                     int64
CC11_gender                    int64
ITC_artist_name                int64
CC11_artist_name               int64
ITC_composer                   int64
CC11_composer                  int64
ITC_lyricist                   int64
CC11_lyricist                  int64
ITC_language                   int64
CC11_language                  int64
ITC_name                       int64
CC11_name                      int64
ITC_song_year                  int64
CC11_song_year                 int64
ITC_song_country               int64
CC11_song_country              int64
ITC_rc                         int64
CC11_rc                        int64
ITC_isrc_rest                  int64
                              ...   
ITC_top1_in_song_log10       float64
ITC_top1_in_song_log10_1     float64
ITC_top1_in_song_x_1         float64
OinC_top1_in_song            float64
ITC_top2_in_song_log10       float64
ITC_top2_in_song_log10_1     float64
ITC_top2_in_song_x_1         float64
OinC_top2_in_song            float64
ITC_top3_in_song_log10       float64
ITC_top3_in_song_log10_1     float64
ITC_top3_in_song_x_1         float64
OinC_top3_in_song            float64
msno                          object
song_id                       object
source_system_tab             object
source_screen_name            object
source_type                   object
expiration_month            category
artist_name                   object
composer                      object
lyricist                      object
language                    category
name                          object
song_year                   category
song_country                category
rc                          category
isrc_rest                   category
top1_in_song                category
top2_in_song                category
top3_in_song                category
Length: 127, dtype: object
number of rows: 7377418
number of columns: 127
'target',
'ITC_msno',
'CC11_msno',
'ITC_song_id',
'CC11_song_id',
'ITC_source_system_tab',
'CC11_source_system_tab',
'ITC_source_screen_name',
'CC11_source_screen_name',
'ITC_source_type',
'CC11_source_type',
'ITC_gender',
'CC11_gender',
'ITC_artist_name',
'CC11_artist_name',
'ITC_composer',
'CC11_composer',
'ITC_lyricist',
'CC11_lyricist',
'ITC_language',
'CC11_language',
'ITC_name',
'CC11_name',
'ITC_song_year',
'CC11_song_year',
'ITC_song_country',
'CC11_song_country',
'ITC_rc',
'CC11_rc',
'ITC_isrc_rest',
'CC11_isrc_rest',
'ITC_top1_in_song',
'CC11_top1_in_song',
'ITC_top2_in_song',
'CC11_top2_in_song',
'ITC_top3_in_song',
'CC11_top3_in_song',
'ITC_msno_log10',
'ITC_msno_log10_1',
'ITC_msno_x_1',
'OinC_msno',
'ITC_song_id_log10',
'ITC_song_id_log10_1',
'ITC_song_id_x_1',
'OinC_song_id',
'ITC_source_system_tab_log10',
'ITC_source_system_tab_log10_1',
'ITC_source_system_tab_x_1',
'OinC_source_system_tab',
'ITC_source_screen_name_log10',
'ITC_source_screen_name_log10_1',
'ITC_source_screen_name_x_1',
'OinC_source_screen_name',
'ITC_source_type_log10',
'ITC_source_type_log10_1',
'ITC_source_type_x_1',
'OinC_source_type',
'ITC_gender_log10',
'ITC_gender_log10_1',
'ITC_gender_x_1',
'OinC_gender',
'ITC_artist_name_log10',
'ITC_artist_name_log10_1',
'ITC_artist_name_x_1',
'OinC_artist_name',
'ITC_composer_log10',
'ITC_composer_log10_1',
'ITC_composer_x_1',
'OinC_composer',
'ITC_lyricist_log10',
'ITC_lyricist_log10_1',
'ITC_lyricist_x_1',
'OinC_lyricist',
'ITC_language_log10',
'ITC_language_log10_1',
'ITC_language_x_1',
'OinC_language',
'ITC_name_log10',
'ITC_name_log10_1',
'ITC_name_x_1',
'OinC_name',
'ITC_song_year_log10',
'ITC_song_year_log10_1',
'ITC_song_year_x_1',
'OinC_song_year',
'ITC_song_country_log10',
'ITC_song_country_log10_1',
'ITC_song_country_x_1',
'OinC_song_country',
'ITC_rc_log10',
'ITC_rc_log10_1',
'ITC_rc_x_1',
'OinC_rc',
'ITC_isrc_rest_log10',
'ITC_isrc_rest_log10_1',
'ITC_isrc_rest_x_1',
'OinC_isrc_rest',
'ITC_top1_in_song_log10',
'ITC_top1_in_song_log10_1',
'ITC_top1_in_song_x_1',
'OinC_top1_in_song',
'ITC_top2_in_song_log10',
'ITC_top2_in_song_log10_1',
'ITC_top2_in_song_x_1',
'OinC_top2_in_song',
'ITC_top3_in_song_log10',
'ITC_top3_in_song_log10_1',
'ITC_top3_in_song_x_1',
'OinC_top3_in_song',
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'expiration_month',
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
working on: ITC_msno
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003BBBB.py:218: SettingWithCopyWarning: 
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
ITC_msno                 int64
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
[10]	training's auc: 0.730698	valid_1's auc: 0.633937
[20]	training's auc: 0.745438	valid_1's auc: 0.640264
[30]	training's auc: 0.755726	valid_1's auc: 0.645183
[40]	training's auc: 0.760384	valid_1's auc: 0.647595
[50]	training's auc: 0.763463	valid_1's auc: 0.649147
[60]	training's auc: 0.767109	valid_1's auc: 0.650638
[70]	training's auc: 0.769561	valid_1's auc: 0.651872
[80]	training's auc: 0.771667	valid_1's auc: 0.652784
[90]	training's auc: 0.773815	valid_1's auc: 0.653729
[100]	training's auc: 0.775564	valid_1's auc: 0.654459
[110]	training's auc: 0.777311	valid_1's auc: 0.655118
[120]	training's auc: 0.779033	valid_1's auc: 0.655723
[130]	training's auc: 0.780449	valid_1's auc: 0.656383
[140]	training's auc: 0.781808	valid_1's auc: 0.656907
[150]	training's auc: 0.783226	valid_1's auc: 0.657397
[160]	training's auc: 0.7846	valid_1's auc: 0.657865
[170]	training's auc: 0.785861	valid_1's auc: 0.658571
[180]	training's auc: 0.787208	valid_1's auc: 0.659079
[190]	training's auc: 0.788328	valid_1's auc: 0.659511
[200]	training's auc: 0.789369	valid_1's auc: 0.660059
[210]	training's auc: 0.790279	valid_1's auc: 0.660338
[220]	training's auc: 0.791131	valid_1's auc: 0.660744
[230]	training's auc: 0.791984	valid_1's auc: 0.661038
[240]	training's auc: 0.792916	valid_1's auc: 0.661427
[250]	training's auc: 0.793758	valid_1's auc: 0.661677
[260]	training's auc: 0.794608	valid_1's auc: 0.661971
[270]	training's auc: 0.795334	valid_1's auc: 0.662183
[280]	training's auc: 0.796138	valid_1's auc: 0.662468
[290]	training's auc: 0.796947	valid_1's auc: 0.66274
[300]	training's auc: 0.797677	valid_1's auc: 0.662988
[310]	training's auc: 0.798833	valid_1's auc: 0.663472
[320]	training's auc: 0.799718	valid_1's auc: 0.663728
[330]	training's auc: 0.800288	valid_1's auc: 0.663927
[340]	training's auc: 0.80089	valid_1's auc: 0.664125
[350]	training's auc: 0.801825	valid_1's auc: 0.664443
[360]	training's auc: 0.802413	valid_1's auc: 0.664556
[370]	training's auc: 0.803064	valid_1's auc: 0.664805
[380]	training's auc: 0.80356	valid_1's auc: 0.664942
[390]	training's auc: 0.804088	valid_1's auc: 0.665095
[400]	training's auc: 0.80465	valid_1's auc: 0.665364
[410]	training's auc: 0.805216	valid_1's auc: 0.665561
[420]	training's auc: 0.805833	valid_1's auc: 0.665726
[430]	training's auc: 0.806336	valid_1's auc: 0.665928
[440]	training's auc: 0.806871	valid_1's auc: 0.666051
[450]	training's auc: 0.80763	valid_1's auc: 0.666292
[460]	training's auc: 0.808086	valid_1's auc: 0.666462
[470]	training's auc: 0.80858	valid_1's auc: 0.66656
[480]	training's auc: 0.809095	valid_1's auc: 0.666668
[490]	training's auc: 0.809701	valid_1's auc: 0.666774
[500]	training's auc: 0.810266	valid_1's auc: 0.666916
[510]	training's auc: 0.810728	valid_1's auc: 0.666958
[520]	training's auc: 0.811159	valid_1's auc: 0.667074
[530]	training's auc: 0.811688	valid_1's auc: 0.667204
[540]	training's auc: 0.812277	valid_1's auc: 0.667395
[550]	training's auc: 0.812824	valid_1's auc: 0.667549
[560]	training's auc: 0.813306	valid_1's auc: 0.66767
[570]	training's auc: 0.813695	valid_1's auc: 0.667757
[580]	training's auc: 0.814126	valid_1's auc: 0.667914
[590]	training's auc: 0.814535	valid_1's auc: 0.66802
[600]	training's auc: 0.814873	valid_1's auc: 0.668052
[610]	training's auc: 0.815288	valid_1's auc: 0.66813
[620]	training's auc: 0.815698	valid_1's auc: 0.668259
[630]	training's auc: 0.816177	valid_1's auc: 0.668392
[640]	training's auc: 0.816501	valid_1's auc: 0.668516
[650]	training's auc: 0.816829	valid_1's auc: 0.668587
[660]	training's auc: 0.817137	valid_1's auc: 0.668643
[670]	training's auc: 0.817484	valid_1's auc: 0.668707
[680]	training's auc: 0.817938	valid_1's auc: 0.668851
[690]	training's auc: 0.818318	valid_1's auc: 0.66895
[700]	training's auc: 0.818812	valid_1's auc: 0.669064
[710]	training's auc: 0.819123	valid_1's auc: 0.669146
[720]	training's auc: 0.819493	valid_1's auc: 0.669244
[730]	training's auc: 0.819769	valid_1's auc: 0.669246
[740]	training's auc: 0.820045	valid_1's auc: 0.669279
[750]	training's auc: 0.820323	valid_1's auc: 0.669343
[760]	training's auc: 0.82063	valid_1's auc: 0.669362
[770]	training's auc: 0.820888	valid_1's auc: 0.669365
[780]	training's auc: 0.821307	valid_1's auc: 0.66948
[790]	training's auc: 0.821726	valid_1's auc: 0.669577
[800]	training's auc: 0.822009	valid_1's auc: 0.669612
[810]	training's auc: 0.822311	valid_1's auc: 0.66964
[820]	training's auc: 0.822577	valid_1's auc: 0.669675
[830]	training's auc: 0.822878	valid_1's auc: 0.669737
[840]	training's auc: 0.823164	valid_1's auc: 0.669789
[850]	training's auc: 0.823477	valid_1's auc: 0.669837
[860]	training's auc: 0.823745	valid_1's auc: 0.669897
[870]	training's auc: 0.824015	valid_1's auc: 0.669964
[880]	training's auc: 0.824331	valid_1's auc: 0.670014
[890]	training's auc: 0.824587	valid_1's auc: 0.670067
[900]	training's auc: 0.824806	valid_1's auc: 0.670124
[910]	training's auc: 0.825117	valid_1's auc: 0.670163
[920]	training's auc: 0.825371	valid_1's auc: 0.670221
[930]	training's auc: 0.825625	valid_1's auc: 0.670244
[940]	training's auc: 0.825987	valid_1's auc: 0.670319
[950]	training's auc: 0.826222	valid_1's auc: 0.670355
[960]	training's auc: 0.826515	valid_1's auc: 0.670413
[970]	training's auc: 0.826767	valid_1's auc: 0.670409
[980]	training's auc: 0.82706	valid_1's auc: 0.67048
[990]	training's auc: 0.827261	valid_1's auc: 0.670514
[1000]	training's auc: 0.827554	valid_1's auc: 0.670525
[1010]	training's auc: 0.827815	valid_1's auc: 0.67058
[1020]	training's auc: 0.828013	valid_1's auc: 0.670626
[1030]	training's auc: 0.828229	valid_1's auc: 0.670638
[1040]	training's auc: 0.828521	valid_1's auc: 0.670694
[1050]	training's auc: 0.828699	valid_1's auc: 0.670743
[1060]	training's auc: 0.828936	valid_1's auc: 0.670772
[1070]	training's auc: 0.829161	valid_1's auc: 0.670824
[1080]	training's auc: 0.829425	valid_1's auc: 0.670843
[1090]	training's auc: 0.829681	valid_1's auc: 0.670917
[1100]	training's auc: 0.829889	valid_1's auc: 0.670968
[1110]	training's auc: 0.830106	valid_1's auc: 0.670998
[1120]	training's auc: 0.830319	valid_1's auc: 0.671041
[1130]	training's auc: 0.830509	valid_1's auc: 0.671049
[1140]	training's auc: 0.830666	valid_1's auc: 0.671061
[1150]	training's auc: 0.830857	valid_1's auc: 0.671098
[1160]	training's auc: 0.83103	valid_1's auc: 0.671111
[1170]	training's auc: 0.831263	valid_1's auc: 0.671144
[1180]	training's auc: 0.831428	valid_1's auc: 0.671152
[1190]	training's auc: 0.831625	valid_1's auc: 0.671209
[1200]	training's auc: 0.831776	valid_1's auc: 0.671207
[1210]	training's auc: 0.831936	valid_1's auc: 0.671217
[1220]	training's auc: 0.832092	valid_1's auc: 0.671215
[1230]	training's auc: 0.832318	valid_1's auc: 0.671239
[1240]	training's auc: 0.832516	valid_1's auc: 0.671267
[1250]	training's auc: 0.832668	valid_1's auc: 0.671245
[1260]	training's auc: 0.832807	valid_1's auc: 0.671221
[1270]	training's auc: 0.832956	valid_1's auc: 0.671249
[1280]	training's auc: 0.833127	valid_1's auc: 0.671276
[1290]	training's auc: 0.833422	valid_1's auc: 0.671362
[1300]	training's auc: 0.833601	valid_1's auc: 0.671411
[1310]	training's auc: 0.833754	valid_1's auc: 0.671425
[1320]	training's auc: 0.833894	valid_1's auc: 0.671432
[1330]	training's auc: 0.834043	valid_1's auc: 0.6714
[1340]	training's auc: 0.834198	valid_1's auc: 0.671404
[1350]	training's auc: 0.834404	valid_1's auc: 0.671461
[1360]	training's auc: 0.834551	valid_1's auc: 0.671486
[1370]	training's auc: 0.834709	valid_1's auc: 0.671509
[1380]	training's auc: 0.834851	valid_1's auc: 0.671492
[1390]	training's auc: 0.835049	valid_1's auc: 0.671543
[1400]	training's auc: 0.835215	valid_1's auc: 0.671592
[1410]	training's auc: 0.835343	valid_1's auc: 0.671613
[1420]	training's auc: 0.835468	valid_1's auc: 0.671631
[1430]	training's auc: 0.835586	valid_1's auc: 0.671651
[1440]	training's auc: 0.835716	valid_1's auc: 0.671649
[1450]	training's auc: 0.835865	valid_1's auc: 0.671651
[1460]	training's auc: 0.835992	valid_1's auc: 0.671631
[1470]	training's auc: 0.83615	valid_1's auc: 0.671646
[1480]	training's auc: 0.836269	valid_1's auc: 0.671677
[1490]	training's auc: 0.836397	valid_1's auc: 0.671687
[1500]	training's auc: 0.836543	valid_1's auc: 0.67168
[1510]	training's auc: 0.836677	valid_1's auc: 0.671702
[1520]	training's auc: 0.836792	valid_1's auc: 0.67171
[1530]	training's auc: 0.836914	valid_1's auc: 0.671708
[1540]	training's auc: 0.837041	valid_1's auc: 0.671706
[1550]	training's auc: 0.837147	valid_1's auc: 0.671695
[1560]	training's auc: 0.83727	valid_1's auc: 0.671703
[1570]	training's auc: 0.837383	valid_1's auc: 0.671733
[1580]	training's auc: 0.837494	valid_1's auc: 0.671723
[1590]	training's auc: 0.837618	valid_1's auc: 0.671714
[1600]	training's auc: 0.837739	valid_1's auc: 0.671734
[1610]	training's auc: 0.837857	valid_1's auc: 0.671739
[1620]	training's auc: 0.837972	valid_1's auc: 0.671739
[1630]	training's auc: 0.83811	valid_1's auc: 0.671764
[1640]	training's auc: 0.838212	valid_1's auc: 0.671772
[1650]	training's auc: 0.838335	valid_1's auc: 0.671776
[1660]	training's auc: 0.838453	valid_1's auc: 0.671787
[1670]	training's auc: 0.838579	valid_1's auc: 0.671782
[1680]	training's auc: 0.838686	valid_1's auc: 0.671785
[1690]	training's auc: 0.838797	valid_1's auc: 0.671786
[1700]	training's auc: 0.838909	valid_1's auc: 0.671772
[1710]	training's auc: 0.839004	valid_1's auc: 0.671777
[1720]	training's auc: 0.839113	valid_1's auc: 0.671789
Early stopping, best iteration is:
[1675]	training's auc: 0.83863	valid_1's auc: 0.671795
best score: 0.671795447532
best iteration: 1675
complete on: ITC_msno

working on: CC11_msno

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
CC11_msno                int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.754529	valid_1's auc: 0.617182
[20]	training's auc: 0.76632	valid_1's auc: 0.621336
[30]	training's auc: 0.77302	valid_1's auc: 0.624312
[40]	training's auc: 0.777437	valid_1's auc: 0.626427
[50]	training's auc: 0.779903	valid_1's auc: 0.62768
[60]	training's auc: 0.781768	valid_1's auc: 0.62845
[70]	training's auc: 0.783577	valid_1's auc: 0.629185
[80]	training's auc: 0.785263	valid_1's auc: 0.629977
[90]	training's auc: 0.786559	valid_1's auc: 0.630409
[100]	training's auc: 0.787943	valid_1's auc: 0.630924
[110]	training's auc: 0.789044	valid_1's auc: 0.631358
[120]	training's auc: 0.790242	valid_1's auc: 0.631763
[130]	training's auc: 0.791213	valid_1's auc: 0.632026
[140]	training's auc: 0.792329	valid_1's auc: 0.632356
[150]	training's auc: 0.793283	valid_1's auc: 0.632683
[160]	training's auc: 0.794488	valid_1's auc: 0.63312
[170]	training's auc: 0.7954	valid_1's auc: 0.633492
[180]	training's auc: 0.796498	valid_1's auc: 0.633898
[190]	training's auc: 0.797335	valid_1's auc: 0.634139
[200]	training's auc: 0.798082	valid_1's auc: 0.634398
[210]	training's auc: 0.798846	valid_1's auc: 0.634643
[220]	training's auc: 0.799537	valid_1's auc: 0.634861
[230]	training's auc: 0.800307	valid_1's auc: 0.635103
[240]	training's auc: 0.801138	valid_1's auc: 0.635365
[250]	training's auc: 0.801943	valid_1's auc: 0.635651
[260]	training's auc: 0.802572	valid_1's auc: 0.635813
[270]	training's auc: 0.803285	valid_1's auc: 0.636129
[280]	training's auc: 0.803916	valid_1's auc: 0.636295
[290]	training's auc: 0.804696	valid_1's auc: 0.636558
[300]	training's auc: 0.805315	valid_1's auc: 0.636826
[310]	training's auc: 0.805896	valid_1's auc: 0.636982
[320]	training's auc: 0.806541	valid_1's auc: 0.637153
[330]	training's auc: 0.807275	valid_1's auc: 0.63742
[340]	training's auc: 0.807929	valid_1's auc: 0.6376
[350]	training's auc: 0.808474	valid_1's auc: 0.637759
[360]	training's auc: 0.808991	valid_1's auc: 0.637952
[370]	training's auc: 0.809547	valid_1's auc: 0.638115
[380]	training's auc: 0.810128	valid_1's auc: 0.638266
[390]	training's auc: 0.810744	valid_1's auc: 0.638426
[400]	training's auc: 0.811378	valid_1's auc: 0.638596
[410]	training's auc: 0.811997	valid_1's auc: 0.63887
[420]	training's auc: 0.812553	valid_1's auc: 0.639011
[430]	training's auc: 0.81298	valid_1's auc: 0.639127
[440]	training's auc: 0.813717	valid_1's auc: 0.63933
[450]	training's auc: 0.814153	valid_1's auc: 0.639416
[460]	training's auc: 0.814667	valid_1's auc: 0.639604
[470]	training's auc: 0.815017	valid_1's auc: 0.639672
[480]	training's auc: 0.815516	valid_1's auc: 0.639784
[490]	training's auc: 0.815966	valid_1's auc: 0.639879
[500]	training's auc: 0.816371	valid_1's auc: 0.639911
[510]	training's auc: 0.816779	valid_1's auc: 0.640028
[520]	training's auc: 0.817157	valid_1's auc: 0.640128
[530]	training's auc: 0.817573	valid_1's auc: 0.640267
[540]	training's auc: 0.817962	valid_1's auc: 0.640358
[550]	training's auc: 0.818396	valid_1's auc: 0.640461
[560]	training's auc: 0.818804	valid_1's auc: 0.640491
[570]	training's auc: 0.819181	valid_1's auc: 0.640598
[580]	training's auc: 0.819547	valid_1's auc: 0.640732
[590]	training's auc: 0.819932	valid_1's auc: 0.640838
[600]	training's auc: 0.820297	valid_1's auc: 0.640928
[610]	training's auc: 0.820623	valid_1's auc: 0.640984
[620]	training's auc: 0.82092	valid_1's auc: 0.641058
[630]	training's auc: 0.821274	valid_1's auc: 0.641119
[640]	training's auc: 0.821647	valid_1's auc: 0.641212
[650]	training's auc: 0.821949	valid_1's auc: 0.64124
[660]	training's auc: 0.822446	valid_1's auc: 0.641375
[670]	training's auc: 0.822758	valid_1's auc: 0.641435
[680]	training's auc: 0.82311	valid_1's auc: 0.641553
[690]	training's auc: 0.823465	valid_1's auc: 0.641595
[700]	training's auc: 0.823758	valid_1's auc: 0.641586
[710]	training's auc: 0.824154	valid_1's auc: 0.641652
[720]	training's auc: 0.824491	valid_1's auc: 0.641732
[730]	training's auc: 0.824836	valid_1's auc: 0.641814
[740]	training's auc: 0.825095	valid_1's auc: 0.641868
[750]	training's auc: 0.825371	valid_1's auc: 0.64189
[760]	training's auc: 0.825769	valid_1's auc: 0.642015
[770]	training's auc: 0.826044	valid_1's auc: 0.642046
[780]	training's auc: 0.826344	valid_1's auc: 0.642114
[790]	training's auc: 0.826647	valid_1's auc: 0.642162
[800]	training's auc: 0.826919	valid_1's auc: 0.642194
[810]	training's auc: 0.827246	valid_1's auc: 0.642267
[820]	training's auc: 0.827555	valid_1's auc: 0.642328
[830]	training's auc: 0.827817	valid_1's auc: 0.642356
[840]	training's auc: 0.828214	valid_1's auc: 0.642449
[850]	training's auc: 0.828453	valid_1's auc: 0.642456
[860]	training's auc: 0.828771	valid_1's auc: 0.642488
[870]	training's auc: 0.829062	valid_1's auc: 0.642548
[880]	training's auc: 0.829342	valid_1's auc: 0.642583
[890]	training's auc: 0.829585	valid_1's auc: 0.642601
[900]	training's auc: 0.829931	valid_1's auc: 0.642699
[910]	training's auc: 0.830159	valid_1's auc: 0.642705
[920]	training's auc: 0.830388	valid_1's auc: 0.642775
[930]	training's auc: 0.830616	valid_1's auc: 0.642785
[940]	training's auc: 0.830903	valid_1's auc: 0.642797
[950]	training's auc: 0.831116	valid_1's auc: 0.642808
[960]	training's auc: 0.831361	valid_1's auc: 0.642846
[970]	training's auc: 0.831569	valid_1's auc: 0.642885
[980]	training's auc: 0.831852	valid_1's auc: 0.642921
[990]	training's auc: 0.832044	valid_1's auc: 0.642933
[1000]	training's auc: 0.832232	valid_1's auc: 0.642953
[1010]	training's auc: 0.832438	valid_1's auc: 0.642956
[1020]	training's auc: 0.832666	valid_1's auc: 0.642975
[1030]	training's auc: 0.832897	valid_1's auc: 0.643006
[1040]	training's auc: 0.833095	valid_1's auc: 0.642982
[1050]	training's auc: 0.83335	valid_1's auc: 0.643022
[1060]	training's auc: 0.833539	valid_1's auc: 0.643018
[1070]	training's auc: 0.833754	valid_1's auc: 0.643075
[1080]	training's auc: 0.833968	valid_1's auc: 0.643087
[1090]	training's auc: 0.834117	valid_1's auc: 0.643094
[1100]	training's auc: 0.834294	valid_1's auc: 0.643116
[1110]	training's auc: 0.834564	valid_1's auc: 0.643196
[1120]	training's auc: 0.834784	valid_1's auc: 0.643261
[1130]	training's auc: 0.834962	valid_1's auc: 0.643248
[1140]	training's auc: 0.835124	valid_1's auc: 0.643286
[1150]	training's auc: 0.835332	valid_1's auc: 0.643294
[1160]	training's auc: 0.835522	valid_1's auc: 0.643338
[1170]	training's auc: 0.835734	valid_1's auc: 0.643362
[1180]	training's auc: 0.835911	valid_1's auc: 0.643384
[1190]	training's auc: 0.83605	valid_1's auc: 0.643394
[1200]	training's auc: 0.836307	valid_1's auc: 0.64345
[1210]	training's auc: 0.836478	valid_1's auc: 0.643474
[1220]	training's auc: 0.836649	valid_1's auc: 0.643502
[1230]	training's auc: 0.836836	valid_1's auc: 0.643525
[1240]	training's auc: 0.837015	valid_1's auc: 0.643508
[1250]	training's auc: 0.837162	valid_1's auc: 0.643505
[1260]	training's auc: 0.837286	valid_1's auc: 0.643505
[1270]	training's auc: 0.837444	valid_1's auc: 0.643525
[1280]	training's auc: 0.83763	valid_1's auc: 0.643504
Early stopping, best iteration is:
[1233]	training's auc: 0.836895	valid_1's auc: 0.643537
best score: 0.643537102025
best iteration: 1233
complete on: CC11_msno

working on: ITC_song_id

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
ITC_song_id              int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.740407	valid_1's auc: 0.6477
[20]	training's auc: 0.751804	valid_1's auc: 0.651726
[30]	training's auc: 0.758114	valid_1's auc: 0.654819
[40]	training's auc: 0.761721	valid_1's auc: 0.656962
[50]	training's auc: 0.764993	valid_1's auc: 0.658385
[60]	training's auc: 0.767629	valid_1's auc: 0.659679
[70]	training's auc: 0.770216	valid_1's auc: 0.660789
[80]	training's auc: 0.772366	valid_1's auc: 0.661852
[90]	training's auc: 0.774417	valid_1's auc: 0.662676
[100]	training's auc: 0.776263	valid_1's auc: 0.663342
[110]	training's auc: 0.777968	valid_1's auc: 0.664025
[120]	training's auc: 0.779567	valid_1's auc: 0.66477
[130]	training's auc: 0.781324	valid_1's auc: 0.665435
[140]	training's auc: 0.78276	valid_1's auc: 0.665922
[150]	training's auc: 0.784102	valid_1's auc: 0.666463
[160]	training's auc: 0.785382	valid_1's auc: 0.666982
[170]	training's auc: 0.786508	valid_1's auc: 0.667537
[180]	training's auc: 0.787649	valid_1's auc: 0.667883
[190]	training's auc: 0.788704	valid_1's auc: 0.668279
[200]	training's auc: 0.789734	valid_1's auc: 0.668636
[210]	training's auc: 0.790731	valid_1's auc: 0.668909
[220]	training's auc: 0.791699	valid_1's auc: 0.669227
[230]	training's auc: 0.792621	valid_1's auc: 0.669532
[240]	training's auc: 0.793575	valid_1's auc: 0.669824
[250]	training's auc: 0.794499	valid_1's auc: 0.670119
[260]	training's auc: 0.795379	valid_1's auc: 0.670317
[270]	training's auc: 0.796229	valid_1's auc: 0.67058
[280]	training's auc: 0.797054	valid_1's auc: 0.670799
[290]	training's auc: 0.797738	valid_1's auc: 0.671026
[300]	training's auc: 0.798528	valid_1's auc: 0.671157
[310]	training's auc: 0.799288	valid_1's auc: 0.671389
[320]	training's auc: 0.799998	valid_1's auc: 0.671539
[330]	training's auc: 0.80058	valid_1's auc: 0.671748
[340]	training's auc: 0.80122	valid_1's auc: 0.671937
[350]	training's auc: 0.801847	valid_1's auc: 0.672138
[360]	training's auc: 0.802516	valid_1's auc: 0.672234
[370]	training's auc: 0.803113	valid_1's auc: 0.672393
[380]	training's auc: 0.803687	valid_1's auc: 0.672542
[390]	training's auc: 0.804287	valid_1's auc: 0.672627
[400]	training's auc: 0.804841	valid_1's auc: 0.672805
[410]	training's auc: 0.805337	valid_1's auc: 0.672965
[420]	training's auc: 0.805876	valid_1's auc: 0.673125
[430]	training's auc: 0.806349	valid_1's auc: 0.673208
[440]	training's auc: 0.806834	valid_1's auc: 0.673299
[450]	training's auc: 0.807646	valid_1's auc: 0.673496
[460]	training's auc: 0.808276	valid_1's auc: 0.67364
[470]	training's auc: 0.808738	valid_1's auc: 0.673735
[480]	training's auc: 0.809192	valid_1's auc: 0.673846
[490]	training's auc: 0.809703	valid_1's auc: 0.673946
[500]	training's auc: 0.810142	valid_1's auc: 0.674044
[510]	training's auc: 0.810697	valid_1's auc: 0.674165
[520]	training's auc: 0.811348	valid_1's auc: 0.674304
[530]	training's auc: 0.811846	valid_1's auc: 0.674379
[540]	training's auc: 0.812244	valid_1's auc: 0.674406
[550]	training's auc: 0.812621	valid_1's auc: 0.674454
[560]	training's auc: 0.813002	valid_1's auc: 0.674522
[570]	training's auc: 0.813396	valid_1's auc: 0.674608
[580]	training's auc: 0.813728	valid_1's auc: 0.674678
[590]	training's auc: 0.814094	valid_1's auc: 0.674727
[600]	training's auc: 0.8145	valid_1's auc: 0.674822
[610]	training's auc: 0.814808	valid_1's auc: 0.674846
[620]	training's auc: 0.815258	valid_1's auc: 0.674912
[630]	training's auc: 0.815672	valid_1's auc: 0.674983
[640]	training's auc: 0.816001	valid_1's auc: 0.674999
[650]	training's auc: 0.816314	valid_1's auc: 0.675059
[660]	training's auc: 0.816625	valid_1's auc: 0.675055
[670]	training's auc: 0.817035	valid_1's auc: 0.675137
[680]	training's auc: 0.817446	valid_1's auc: 0.675217
[690]	training's auc: 0.818027	valid_1's auc: 0.675325
[700]	training's auc: 0.818334	valid_1's auc: 0.675388
[710]	training's auc: 0.81885	valid_1's auc: 0.675515
[720]	training's auc: 0.819154	valid_1's auc: 0.675569
[730]	training's auc: 0.819403	valid_1's auc: 0.675594
[740]	training's auc: 0.819721	valid_1's auc: 0.675644
[750]	training's auc: 0.820106	valid_1's auc: 0.675668
[760]	training's auc: 0.820424	valid_1's auc: 0.675681
[770]	training's auc: 0.820835	valid_1's auc: 0.675776
[780]	training's auc: 0.821221	valid_1's auc: 0.67586
[790]	training's auc: 0.821515	valid_1's auc: 0.675875
[800]	training's auc: 0.821782	valid_1's auc: 0.675913
[810]	training's auc: 0.822109	valid_1's auc: 0.675918
[820]	training's auc: 0.822464	valid_1's auc: 0.675948
[830]	training's auc: 0.822759	valid_1's auc: 0.675966
[840]	training's auc: 0.823082	valid_1's auc: 0.676015
[850]	training's auc: 0.823487	valid_1's auc: 0.676102
[860]	training's auc: 0.823788	valid_1's auc: 0.676102
[870]	training's auc: 0.824024	valid_1's auc: 0.676087
[880]	training's auc: 0.824288	valid_1's auc: 0.676089
[890]	training's auc: 0.824694	valid_1's auc: 0.676115
[900]	training's auc: 0.824926	valid_1's auc: 0.676131
[910]	training's auc: 0.825173	valid_1's auc: 0.67614
[920]	training's auc: 0.825404	valid_1's auc: 0.676155
[930]	training's auc: 0.825693	valid_1's auc: 0.676189
[940]	training's auc: 0.825993	valid_1's auc: 0.676233
[950]	training's auc: 0.826252	valid_1's auc: 0.676262
[960]	training's auc: 0.826465	valid_1's auc: 0.676274
[970]	training's auc: 0.826724	valid_1's auc: 0.67625
[980]	training's auc: 0.827013	valid_1's auc: 0.676279
[990]	training's auc: 0.827398	valid_1's auc: 0.676435
[1000]	training's auc: 0.827613	valid_1's auc: 0.676441
[1010]	training's auc: 0.827866	valid_1's auc: 0.676445
[1020]	training's auc: 0.828126	valid_1's auc: 0.676478
[1030]	training's auc: 0.828358	valid_1's auc: 0.676466
[1040]	training's auc: 0.828581	valid_1's auc: 0.676467
[1050]	training's auc: 0.828769	valid_1's auc: 0.676452
[1060]	training's auc: 0.828979	valid_1's auc: 0.67648
[1070]	training's auc: 0.829151	valid_1's auc: 0.676502
[1080]	training's auc: 0.829364	valid_1's auc: 0.676484
[1090]	training's auc: 0.829556	valid_1's auc: 0.676454
[1100]	training's auc: 0.829748	valid_1's auc: 0.676459
[1110]	training's auc: 0.829938	valid_1's auc: 0.676466
Early stopping, best iteration is:
[1069]	training's auc: 0.829142	valid_1's auc: 0.676511
best score: 0.676511283808
best iteration: 1069
complete on: ITC_song_id

working on: CC11_song_id

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
CC11_song_id             int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.747173	valid_1's auc: 0.60739
[20]	training's auc: 0.757916	valid_1's auc: 0.610554
[30]	training's auc: 0.76427	valid_1's auc: 0.613336
[40]	training's auc: 0.767876	valid_1's auc: 0.615014
[50]	training's auc: 0.771768	valid_1's auc: 0.616621
[60]	training's auc: 0.774344	valid_1's auc: 0.617612
[70]	training's auc: 0.776877	valid_1's auc: 0.618467
[80]	training's auc: 0.779089	valid_1's auc: 0.619675
[90]	training's auc: 0.781231	valid_1's auc: 0.620229
[100]	training's auc: 0.783155	valid_1's auc: 0.621053
[110]	training's auc: 0.785042	valid_1's auc: 0.621643
[120]	training's auc: 0.786748	valid_1's auc: 0.622381
[130]	training's auc: 0.788248	valid_1's auc: 0.622835
[140]	training's auc: 0.789622	valid_1's auc: 0.623284
[150]	training's auc: 0.79102	valid_1's auc: 0.623603
[160]	training's auc: 0.792696	valid_1's auc: 0.624052
[170]	training's auc: 0.793962	valid_1's auc: 0.624711
[180]	training's auc: 0.795134	valid_1's auc: 0.625108
[190]	training's auc: 0.796252	valid_1's auc: 0.625389
[200]	training's auc: 0.797226	valid_1's auc: 0.625719
[210]	training's auc: 0.798207	valid_1's auc: 0.626006
[220]	training's auc: 0.799146	valid_1's auc: 0.626313
[230]	training's auc: 0.800013	valid_1's auc: 0.626537
[240]	training's auc: 0.800945	valid_1's auc: 0.62679
[250]	training's auc: 0.801858	valid_1's auc: 0.62704
[260]	training's auc: 0.802751	valid_1's auc: 0.62724
[270]	training's auc: 0.803653	valid_1's auc: 0.627477
[280]	training's auc: 0.804479	valid_1's auc: 0.627627
[290]	training's auc: 0.805189	valid_1's auc: 0.627799
[300]	training's auc: 0.805957	valid_1's auc: 0.627902
[310]	training's auc: 0.806702	valid_1's auc: 0.628059
[320]	training's auc: 0.807435	valid_1's auc: 0.628258
[330]	training's auc: 0.808109	valid_1's auc: 0.628436
[340]	training's auc: 0.808746	valid_1's auc: 0.628604
[350]	training's auc: 0.809311	valid_1's auc: 0.628636
[360]	training's auc: 0.809989	valid_1's auc: 0.628744
[370]	training's auc: 0.810603	valid_1's auc: 0.628896
[380]	training's auc: 0.811171	valid_1's auc: 0.628995
[390]	training's auc: 0.811729	valid_1's auc: 0.629103
[400]	training's auc: 0.81216	valid_1's auc: 0.629254
[410]	training's auc: 0.812671	valid_1's auc: 0.629314
[420]	training's auc: 0.813188	valid_1's auc: 0.629416
[430]	training's auc: 0.813699	valid_1's auc: 0.629489
[440]	training's auc: 0.814232	valid_1's auc: 0.629575
[450]	training's auc: 0.814731	valid_1's auc: 0.629643
[460]	training's auc: 0.815341	valid_1's auc: 0.629763
[470]	training's auc: 0.816118	valid_1's auc: 0.629909
[480]	training's auc: 0.81684	valid_1's auc: 0.630089
[490]	training's auc: 0.817336	valid_1's auc: 0.630153
[500]	training's auc: 0.817798	valid_1's auc: 0.630223
[510]	training's auc: 0.818257	valid_1's auc: 0.630232
[520]	training's auc: 0.818739	valid_1's auc: 0.630316
[530]	training's auc: 0.819495	valid_1's auc: 0.630429
[540]	training's auc: 0.819941	valid_1's auc: 0.630494
[550]	training's auc: 0.820343	valid_1's auc: 0.630564
[560]	training's auc: 0.820948	valid_1's auc: 0.630659
[570]	training's auc: 0.821304	valid_1's auc: 0.630673
[580]	training's auc: 0.821738	valid_1's auc: 0.630795
[590]	training's auc: 0.822127	valid_1's auc: 0.630803
[600]	training's auc: 0.822501	valid_1's auc: 0.630863
[610]	training's auc: 0.822782	valid_1's auc: 0.63094
[620]	training's auc: 0.823145	valid_1's auc: 0.630958
[630]	training's auc: 0.823489	valid_1's auc: 0.630993
[640]	training's auc: 0.823901	valid_1's auc: 0.631031
[650]	training's auc: 0.824212	valid_1's auc: 0.631063
[660]	training's auc: 0.824689	valid_1's auc: 0.631152
[670]	training's auc: 0.825092	valid_1's auc: 0.631162
[680]	training's auc: 0.825409	valid_1's auc: 0.631184
[690]	training's auc: 0.825708	valid_1's auc: 0.631213
[700]	training's auc: 0.82623	valid_1's auc: 0.631237
[710]	training's auc: 0.826705	valid_1's auc: 0.631306
[720]	training's auc: 0.827011	valid_1's auc: 0.631252
[730]	training's auc: 0.827376	valid_1's auc: 0.631303
[740]	training's auc: 0.827655	valid_1's auc: 0.631317
[750]	training's auc: 0.827904	valid_1's auc: 0.631344
[760]	training's auc: 0.828233	valid_1's auc: 0.631373
[770]	training's auc: 0.828672	valid_1's auc: 0.631466
[780]	training's auc: 0.828977	valid_1's auc: 0.631513
[790]	training's auc: 0.829233	valid_1's auc: 0.631553
[800]	training's auc: 0.829518	valid_1's auc: 0.631575
[810]	training's auc: 0.829806	valid_1's auc: 0.631594
[820]	training's auc: 0.830089	valid_1's auc: 0.631593
[830]	training's auc: 0.830451	valid_1's auc: 0.631624
[840]	training's auc: 0.830708	valid_1's auc: 0.631627
[850]	training's auc: 0.830982	valid_1's auc: 0.631625
[860]	training's auc: 0.831294	valid_1's auc: 0.631627
[870]	training's auc: 0.831611	valid_1's auc: 0.631651
[880]	training's auc: 0.831928	valid_1's auc: 0.631712
[890]	training's auc: 0.832148	valid_1's auc: 0.631715
[900]	training's auc: 0.832371	valid_1's auc: 0.631718
[910]	training's auc: 0.832612	valid_1's auc: 0.631743
[920]	training's auc: 0.832865	valid_1's auc: 0.631782
[930]	training's auc: 0.83311	valid_1's auc: 0.631784
[940]	training's auc: 0.833499	valid_1's auc: 0.631812
[950]	training's auc: 0.833753	valid_1's auc: 0.631822
[960]	training's auc: 0.833982	valid_1's auc: 0.631817
[970]	training's auc: 0.834234	valid_1's auc: 0.631817
[980]	training's auc: 0.834579	valid_1's auc: 0.631839
[990]	training's auc: 0.834804	valid_1's auc: 0.631836
[1000]	training's auc: 0.834997	valid_1's auc: 0.631819
[1010]	training's auc: 0.835219	valid_1's auc: 0.631841
[1020]	training's auc: 0.835514	valid_1's auc: 0.631861
[1030]	training's auc: 0.835775	valid_1's auc: 0.631846
[1040]	training's auc: 0.836118	valid_1's auc: 0.631853
[1050]	training's auc: 0.836336	valid_1's auc: 0.631916
[1060]	training's auc: 0.836562	valid_1's auc: 0.631923
[1070]	training's auc: 0.836726	valid_1's auc: 0.631934
[1080]	training's auc: 0.837052	valid_1's auc: 0.631978
[1090]	training's auc: 0.837225	valid_1's auc: 0.63195
[1100]	training's auc: 0.837425	valid_1's auc: 0.631948
[1110]	training's auc: 0.83762	valid_1's auc: 0.631967
[1120]	training's auc: 0.837824	valid_1's auc: 0.631957
Early stopping, best iteration is:
[1074]	training's auc: 0.836926	valid_1's auc: 0.631993
best score: 0.63199304514
best iteration: 1074
complete on: CC11_song_id

working on: ITC_source_system_tab

Our guest selection:
target                      uint8
msno                     category
song_id                  category
source_system_tab        category
source_screen_name       category
source_type              category
artist_name              category
composer                 category
lyricist                 category
song_year                category
ITC_source_system_tab       int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721934	valid_1's auc: 0.629665
[20]	training's auc: 0.734435	valid_1's auc: 0.635193
[30]	training's auc: 0.744119	valid_1's auc: 0.640334
[40]	training's auc: 0.748648	valid_1's auc: 0.642502
[50]	training's auc: 0.751828	valid_1's auc: 0.643967
[60]	training's auc: 0.754818	valid_1's auc: 0.645302
[70]	training's auc: 0.757535	valid_1's auc: 0.646437
[80]	training's auc: 0.760082	valid_1's auc: 0.647177
[90]	training's auc: 0.762488	valid_1's auc: 0.648242
[100]	training's auc: 0.764359	valid_1's auc: 0.649151
[110]	training's auc: 0.76621	valid_1's auc: 0.64991
[120]	training's auc: 0.767947	valid_1's auc: 0.650573
[130]	training's auc: 0.769803	valid_1's auc: 0.651265
[140]	training's auc: 0.771332	valid_1's auc: 0.6521
[150]	training's auc: 0.772734	valid_1's auc: 0.652722
[160]	training's auc: 0.774229	valid_1's auc: 0.653331
[170]	training's auc: 0.775366	valid_1's auc: 0.653941
[180]	training's auc: 0.776576	valid_1's auc: 0.654424
[190]	training's auc: 0.777789	valid_1's auc: 0.654872
[200]	training's auc: 0.778982	valid_1's auc: 0.655302
[210]	training's auc: 0.780058	valid_1's auc: 0.655712
[220]	training's auc: 0.781087	valid_1's auc: 0.656097
[230]	training's auc: 0.782013	valid_1's auc: 0.656382
[240]	training's auc: 0.782947	valid_1's auc: 0.656685
[250]	training's auc: 0.784043	valid_1's auc: 0.656933
[260]	training's auc: 0.785138	valid_1's auc: 0.6572
[270]	training's auc: 0.786184	valid_1's auc: 0.657445
[280]	training's auc: 0.787046	valid_1's auc: 0.657768
[290]	training's auc: 0.7877	valid_1's auc: 0.657981
[300]	training's auc: 0.78879	valid_1's auc: 0.658349
[310]	training's auc: 0.78974	valid_1's auc: 0.658611
[320]	training's auc: 0.790476	valid_1's auc: 0.658754
[330]	training's auc: 0.791432	valid_1's auc: 0.6591
[340]	training's auc: 0.792144	valid_1's auc: 0.659312
[350]	training's auc: 0.792772	valid_1's auc: 0.659557
[360]	training's auc: 0.793353	valid_1's auc: 0.659779
[370]	training's auc: 0.794075	valid_1's auc: 0.66011
[380]	training's auc: 0.794835	valid_1's auc: 0.660373
[390]	training's auc: 0.795453	valid_1's auc: 0.660564
[400]	training's auc: 0.795959	valid_1's auc: 0.660762
[410]	training's auc: 0.796584	valid_1's auc: 0.660848
[420]	training's auc: 0.797338	valid_1's auc: 0.661078
[430]	training's auc: 0.797792	valid_1's auc: 0.661198
[440]	training's auc: 0.798394	valid_1's auc: 0.661331
[450]	training's auc: 0.798876	valid_1's auc: 0.661412
[460]	training's auc: 0.799649	valid_1's auc: 0.661678
[470]	training's auc: 0.800232	valid_1's auc: 0.66183
[480]	training's auc: 0.800704	valid_1's auc: 0.661938
[490]	training's auc: 0.801258	valid_1's auc: 0.662144
[500]	training's auc: 0.801754	valid_1's auc: 0.662303
[510]	training's auc: 0.802477	valid_1's auc: 0.662549
[520]	training's auc: 0.802918	valid_1's auc: 0.662663
[530]	training's auc: 0.803586	valid_1's auc: 0.662842
[540]	training's auc: 0.804013	valid_1's auc: 0.66297
[550]	training's auc: 0.804554	valid_1's auc: 0.663114
[560]	training's auc: 0.804981	valid_1's auc: 0.66321
[570]	training's auc: 0.80536	valid_1's auc: 0.663247
[580]	training's auc: 0.805812	valid_1's auc: 0.663438
[590]	training's auc: 0.806127	valid_1's auc: 0.663502
[600]	training's auc: 0.806595	valid_1's auc: 0.663578
[610]	training's auc: 0.806991	valid_1's auc: 0.663702
[620]	training's auc: 0.807495	valid_1's auc: 0.663817
[630]	training's auc: 0.808101	valid_1's auc: 0.663939
[640]	training's auc: 0.808447	valid_1's auc: 0.664012
[650]	training's auc: 0.808813	valid_1's auc: 0.664049
[660]	training's auc: 0.809173	valid_1's auc: 0.664129
[670]	training's auc: 0.809595	valid_1's auc: 0.664232
[680]	training's auc: 0.809924	valid_1's auc: 0.664308
[690]	training's auc: 0.810441	valid_1's auc: 0.664446
[700]	training's auc: 0.810746	valid_1's auc: 0.664497
[710]	training's auc: 0.81122	valid_1's auc: 0.664667
[720]	training's auc: 0.811605	valid_1's auc: 0.664827
[730]	training's auc: 0.811922	valid_1's auc: 0.664885
[740]	training's auc: 0.812194	valid_1's auc: 0.664948
[750]	training's auc: 0.812639	valid_1's auc: 0.664976
[760]	training's auc: 0.812954	valid_1's auc: 0.664927
[770]	training's auc: 0.81341	valid_1's auc: 0.665061
[780]	training's auc: 0.813734	valid_1's auc: 0.66517
[790]	training's auc: 0.814053	valid_1's auc: 0.665196
[800]	training's auc: 0.814352	valid_1's auc: 0.665265
[810]	training's auc: 0.814627	valid_1's auc: 0.665263
[820]	training's auc: 0.814952	valid_1's auc: 0.665301
[830]	training's auc: 0.815394	valid_1's auc: 0.665422
[840]	training's auc: 0.815796	valid_1's auc: 0.665538
[850]	training's auc: 0.81606	valid_1's auc: 0.665585
[860]	training's auc: 0.81634	valid_1's auc: 0.66562
[870]	training's auc: 0.81664	valid_1's auc: 0.665677
[880]	training's auc: 0.81694	valid_1's auc: 0.665746
[890]	training's auc: 0.817311	valid_1's auc: 0.665758
[900]	training's auc: 0.817656	valid_1's auc: 0.665813
[910]	training's auc: 0.817912	valid_1's auc: 0.665824
[920]	training's auc: 0.818211	valid_1's auc: 0.665854
[930]	training's auc: 0.818467	valid_1's auc: 0.665889
[940]	training's auc: 0.818756	valid_1's auc: 0.665917
[950]	training's auc: 0.818991	valid_1's auc: 0.665906
[960]	training's auc: 0.819275	valid_1's auc: 0.665927
[970]	training's auc: 0.819606	valid_1's auc: 0.666015
[980]	training's auc: 0.819826	valid_1's auc: 0.666027
[990]	training's auc: 0.820081	valid_1's auc: 0.666066
[1000]	training's auc: 0.820337	valid_1's auc: 0.666125
[1010]	training's auc: 0.820578	valid_1's auc: 0.666148
[1020]	training's auc: 0.820854	valid_1's auc: 0.666181
[1030]	training's auc: 0.82108	valid_1's auc: 0.666225
[1040]	training's auc: 0.821294	valid_1's auc: 0.666262
[1050]	training's auc: 0.821502	valid_1's auc: 0.666304
[1060]	training's auc: 0.821735	valid_1's auc: 0.66634
[1070]	training's auc: 0.821931	valid_1's auc: 0.666355
[1080]	training's auc: 0.822129	valid_1's auc: 0.666387
[1090]	training's auc: 0.822301	valid_1's auc: 0.666422
[1100]	training's auc: 0.822553	valid_1's auc: 0.666476
[1110]	training's auc: 0.822752	valid_1's auc: 0.666517
[1120]	training's auc: 0.82297	valid_1's auc: 0.666438
[1130]	training's auc: 0.823183	valid_1's auc: 0.666472
[1140]	training's auc: 0.823367	valid_1's auc: 0.666466
[1150]	training's auc: 0.82359	valid_1's auc: 0.666495
[1160]	training's auc: 0.823801	valid_1's auc: 0.666519
[1170]	training's auc: 0.824016	valid_1's auc: 0.666527
[1180]	training's auc: 0.824215	valid_1's auc: 0.666562
[1190]	training's auc: 0.824397	valid_1's auc: 0.666585
[1200]	training's auc: 0.824771	valid_1's auc: 0.666727
[1210]	training's auc: 0.824974	valid_1's auc: 0.666785
[1220]	training's auc: 0.82514	valid_1's auc: 0.666813
[1230]	training's auc: 0.825362	valid_1's auc: 0.66683
[1240]	training's auc: 0.825566	valid_1's auc: 0.666824
[1250]	training's auc: 0.825783	valid_1's auc: 0.666907
[1260]	training's auc: 0.825934	valid_1's auc: 0.666967
[1270]	training's auc: 0.826083	valid_1's auc: 0.666987
[1280]	training's auc: 0.826246	valid_1's auc: 0.666982
[1290]	training's auc: 0.826416	valid_1's auc: 0.667015
[1300]	training's auc: 0.826563	valid_1's auc: 0.667036
[1310]	training's auc: 0.826785	valid_1's auc: 0.667062
[1320]	training's auc: 0.826911	valid_1's auc: 0.667051
[1330]	training's auc: 0.82709	valid_1's auc: 0.667036
[1340]	training's auc: 0.827234	valid_1's auc: 0.667063
[1350]	training's auc: 0.827403	valid_1's auc: 0.667057
[1360]	training's auc: 0.827562	valid_1's auc: 0.667072
[1370]	training's auc: 0.827752	valid_1's auc: 0.667099
[1380]	training's auc: 0.827875	valid_1's auc: 0.667099
[1390]	training's auc: 0.828035	valid_1's auc: 0.66712
[1400]	training's auc: 0.828175	valid_1's auc: 0.667126
[1410]	training's auc: 0.828293	valid_1's auc: 0.667119
[1420]	training's auc: 0.828438	valid_1's auc: 0.667156
[1430]	training's auc: 0.828564	valid_1's auc: 0.667161
[1440]	training's auc: 0.828701	valid_1's auc: 0.667185
[1450]	training's auc: 0.828846	valid_1's auc: 0.66718
[1460]	training's auc: 0.828968	valid_1's auc: 0.667174
[1470]	training's auc: 0.829092	valid_1's auc: 0.667188
[1480]	training's auc: 0.829201	valid_1's auc: 0.667213
[1490]	training's auc: 0.82937	valid_1's auc: 0.667269
[1500]	training's auc: 0.829512	valid_1's auc: 0.667294
[1510]	training's auc: 0.829664	valid_1's auc: 0.667298
[1520]	training's auc: 0.829778	valid_1's auc: 0.667295
[1530]	training's auc: 0.829904	valid_1's auc: 0.66731
[1540]	training's auc: 0.830061	valid_1's auc: 0.667331
[1550]	training's auc: 0.830209	valid_1's auc: 0.667337
[1560]	training's auc: 0.8304	valid_1's auc: 0.667361
[1570]	training's auc: 0.830522	valid_1's auc: 0.667381
[1580]	training's auc: 0.830644	valid_1's auc: 0.667375
[1590]	training's auc: 0.830824	valid_1's auc: 0.66744
[1600]	training's auc: 0.831023	valid_1's auc: 0.667488
[1610]	training's auc: 0.831167	valid_1's auc: 0.667475
[1620]	training's auc: 0.831282	valid_1's auc: 0.667497
[1630]	training's auc: 0.831407	valid_1's auc: 0.667528
[1640]	training's auc: 0.831519	valid_1's auc: 0.667517
[1650]	training's auc: 0.831646	valid_1's auc: 0.667486
[1660]	training's auc: 0.8318	valid_1's auc: 0.667506
[1670]	training's auc: 0.831926	valid_1's auc: 0.667501
[1680]	training's auc: 0.832012	valid_1's auc: 0.667514
[1690]	training's auc: 0.832228	valid_1's auc: 0.667559
[1700]	training's auc: 0.832354	valid_1's auc: 0.667581
[1710]	training's auc: 0.83247	valid_1's auc: 0.667576
[1720]	training's auc: 0.832567	valid_1's auc: 0.667598
[1730]	training's auc: 0.832674	valid_1's auc: 0.667623
[1740]	training's auc: 0.832759	valid_1's auc: 0.667634
[1750]	training's auc: 0.83288	valid_1's auc: 0.667628
[1760]	training's auc: 0.832977	valid_1's auc: 0.667631
[1770]	training's auc: 0.833115	valid_1's auc: 0.667629
[1780]	training's auc: 0.833229	valid_1's auc: 0.667634
[1790]	training's auc: 0.833347	valid_1's auc: 0.667668
[1800]	training's auc: 0.833467	valid_1's auc: 0.667677
[1810]	training's auc: 0.833593	valid_1's auc: 0.667675
[1820]	training's auc: 0.833704	valid_1's auc: 0.667687
[1830]	training's auc: 0.833823	valid_1's auc: 0.667688
[1840]	training's auc: 0.833925	valid_1's auc: 0.66772
[1850]	training's auc: 0.834042	valid_1's auc: 0.667736
[1860]	training's auc: 0.834136	valid_1's auc: 0.667717
[1870]	training's auc: 0.834242	valid_1's auc: 0.667741
[1880]	training's auc: 0.834351	valid_1's auc: 0.667762
[1890]	training's auc: 0.83447	valid_1's auc: 0.66776
[1900]	training's auc: 0.834568	valid_1's auc: 0.667757
[1910]	training's auc: 0.834677	valid_1's auc: 0.667781
[1920]	training's auc: 0.834791	valid_1's auc: 0.667789
[1930]	training's auc: 0.835038	valid_1's auc: 0.667925
[1940]	training's auc: 0.835136	valid_1's auc: 0.66793
[1950]	training's auc: 0.835262	valid_1's auc: 0.667927
[1960]	training's auc: 0.835389	valid_1's auc: 0.667944
[1970]	training's auc: 0.835483	valid_1's auc: 0.667931
[1980]	training's auc: 0.83558	valid_1's auc: 0.667939
[1990]	training's auc: 0.835658	valid_1's auc: 0.667932
[2000]	training's auc: 0.835748	valid_1's auc: 0.667931
Early stopping, best iteration is:
[1956]	training's auc: 0.835347	valid_1's auc: 0.667953
best score: 0.667953198364
best iteration: 1956
complete on: ITC_source_system_tab

working on: CC11_source_system_tab

Our guest selection:
target                       uint8
msno                      category
song_id                   category
source_system_tab         category
source_screen_name        category
source_type               category
artist_name               category
composer                  category
lyricist                  category
song_year                 category
CC11_source_system_tab       int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721281	valid_1's auc: 0.629427
[20]	training's auc: 0.736274	valid_1's auc: 0.635741
[30]	training's auc: 0.743545	valid_1's auc: 0.639814
[40]	training's auc: 0.747833	valid_1's auc: 0.641952
[50]	training's auc: 0.75141	valid_1's auc: 0.643753
[60]	training's auc: 0.754354	valid_1's auc: 0.645178
[70]	training's auc: 0.757302	valid_1's auc: 0.646128
[80]	training's auc: 0.759945	valid_1's auc: 0.647398
[90]	training's auc: 0.762176	valid_1's auc: 0.648269
[100]	training's auc: 0.764204	valid_1's auc: 0.649066
[110]	training's auc: 0.76603	valid_1's auc: 0.649813
[120]	training's auc: 0.767734	valid_1's auc: 0.650467
[130]	training's auc: 0.769433	valid_1's auc: 0.651236
[140]	training's auc: 0.771052	valid_1's auc: 0.651925
[150]	training's auc: 0.772627	valid_1's auc: 0.652608
[160]	training's auc: 0.774003	valid_1's auc: 0.653189
[170]	training's auc: 0.775321	valid_1's auc: 0.653727
[180]	training's auc: 0.776529	valid_1's auc: 0.654229
[190]	training's auc: 0.777737	valid_1's auc: 0.654623
[200]	training's auc: 0.77882	valid_1's auc: 0.655035
[210]	training's auc: 0.779881	valid_1's auc: 0.655457
[220]	training's auc: 0.780889	valid_1's auc: 0.65594
[230]	training's auc: 0.781874	valid_1's auc: 0.656223
[240]	training's auc: 0.782846	valid_1's auc: 0.65658
[250]	training's auc: 0.783884	valid_1's auc: 0.656946
[260]	training's auc: 0.78479	valid_1's auc: 0.657148
[270]	training's auc: 0.785838	valid_1's auc: 0.657538
[280]	training's auc: 0.786936	valid_1's auc: 0.657915
[290]	training's auc: 0.787592	valid_1's auc: 0.658055
[300]	training's auc: 0.788346	valid_1's auc: 0.658258
[310]	training's auc: 0.78935	valid_1's auc: 0.658625
[320]	training's auc: 0.790226	valid_1's auc: 0.658889
[330]	training's auc: 0.79097	valid_1's auc: 0.659055
[340]	training's auc: 0.791703	valid_1's auc: 0.659306
[350]	training's auc: 0.792407	valid_1's auc: 0.659481
[360]	training's auc: 0.793001	valid_1's auc: 0.659682
[370]	training's auc: 0.793703	valid_1's auc: 0.659708
[380]	training's auc: 0.794561	valid_1's auc: 0.659951
[390]	training's auc: 0.795342	valid_1's auc: 0.66022
[400]	training's auc: 0.795816	valid_1's auc: 0.660376
[410]	training's auc: 0.796486	valid_1's auc: 0.660545
[420]	training's auc: 0.797089	valid_1's auc: 0.660752
[430]	training's auc: 0.797597	valid_1's auc: 0.660859
[440]	training's auc: 0.798304	valid_1's auc: 0.661181
[450]	training's auc: 0.798851	valid_1's auc: 0.661288
[460]	training's auc: 0.799409	valid_1's auc: 0.661423
[470]	training's auc: 0.800092	valid_1's auc: 0.66161
[480]	training's auc: 0.800537	valid_1's auc: 0.661699
[490]	training's auc: 0.801276	valid_1's auc: 0.661924
[500]	training's auc: 0.801716	valid_1's auc: 0.662041
[510]	training's auc: 0.802147	valid_1's auc: 0.662165
[520]	training's auc: 0.802946	valid_1's auc: 0.662471
[530]	training's auc: 0.803514	valid_1's auc: 0.662625
[540]	training's auc: 0.803941	valid_1's auc: 0.662757
[550]	training's auc: 0.804539	valid_1's auc: 0.662838
[560]	training's auc: 0.804977	valid_1's auc: 0.662937
[570]	training's auc: 0.805402	valid_1's auc: 0.663012
[580]	training's auc: 0.805778	valid_1's auc: 0.663112
[590]	training's auc: 0.806254	valid_1's auc: 0.663227
[600]	training's auc: 0.806622	valid_1's auc: 0.663343
[610]	training's auc: 0.806982	valid_1's auc: 0.663369
[620]	training's auc: 0.807434	valid_1's auc: 0.663473
[630]	training's auc: 0.807856	valid_1's auc: 0.663591
[640]	training's auc: 0.80835	valid_1's auc: 0.663712
[650]	training's auc: 0.80864	valid_1's auc: 0.663801
[660]	training's auc: 0.809173	valid_1's auc: 0.663944
[670]	training's auc: 0.809552	valid_1's auc: 0.664029
[680]	training's auc: 0.809865	valid_1's auc: 0.664113
[690]	training's auc: 0.810168	valid_1's auc: 0.664194
[700]	training's auc: 0.810568	valid_1's auc: 0.664351
[710]	training's auc: 0.810952	valid_1's auc: 0.664424
[720]	training's auc: 0.81126	valid_1's auc: 0.664527
[730]	training's auc: 0.811636	valid_1's auc: 0.664554
[740]	training's auc: 0.812003	valid_1's auc: 0.664599
[750]	training's auc: 0.812369	valid_1's auc: 0.664712
[760]	training's auc: 0.812913	valid_1's auc: 0.664873
[770]	training's auc: 0.813248	valid_1's auc: 0.664943
[780]	training's auc: 0.813604	valid_1's auc: 0.664992
[790]	training's auc: 0.813952	valid_1's auc: 0.665061
[800]	training's auc: 0.814248	valid_1's auc: 0.665083
[810]	training's auc: 0.814536	valid_1's auc: 0.665133
[820]	training's auc: 0.814857	valid_1's auc: 0.665173
[830]	training's auc: 0.815162	valid_1's auc: 0.665202
[840]	training's auc: 0.815494	valid_1's auc: 0.665284
[850]	training's auc: 0.815786	valid_1's auc: 0.665275
[860]	training's auc: 0.81629	valid_1's auc: 0.665386
[870]	training's auc: 0.816564	valid_1's auc: 0.665423
[880]	training's auc: 0.81708	valid_1's auc: 0.665565
[890]	training's auc: 0.817449	valid_1's auc: 0.665615
[900]	training's auc: 0.817703	valid_1's auc: 0.665642
[910]	training's auc: 0.817938	valid_1's auc: 0.665652
[920]	training's auc: 0.818167	valid_1's auc: 0.665689
[930]	training's auc: 0.818557	valid_1's auc: 0.665749
[940]	training's auc: 0.818813	valid_1's auc: 0.665785
[950]	training's auc: 0.819049	valid_1's auc: 0.665814
[960]	training's auc: 0.819422	valid_1's auc: 0.665877
[970]	training's auc: 0.819641	valid_1's auc: 0.665908
[980]	training's auc: 0.819866	valid_1's auc: 0.665927
[990]	training's auc: 0.820152	valid_1's auc: 0.665943
[1000]	training's auc: 0.820372	valid_1's auc: 0.665989
[1010]	training's auc: 0.820666	valid_1's auc: 0.666046
[1020]	training's auc: 0.820882	valid_1's auc: 0.66607
[1030]	training's auc: 0.821125	valid_1's auc: 0.666106
[1040]	training's auc: 0.821377	valid_1's auc: 0.666095
[1050]	training's auc: 0.821598	valid_1's auc: 0.666075
[1060]	training's auc: 0.821898	valid_1's auc: 0.666111
[1070]	training's auc: 0.822086	valid_1's auc: 0.66613
[1080]	training's auc: 0.822316	valid_1's auc: 0.666154
[1090]	training's auc: 0.822487	valid_1's auc: 0.666171
[1100]	training's auc: 0.822673	valid_1's auc: 0.666206
[1110]	training's auc: 0.822898	valid_1's auc: 0.666222
[1120]	training's auc: 0.823069	valid_1's auc: 0.66625
[1130]	training's auc: 0.823292	valid_1's auc: 0.666257
[1140]	training's auc: 0.823476	valid_1's auc: 0.666269
[1150]	training's auc: 0.823706	valid_1's auc: 0.666298
[1160]	training's auc: 0.823897	valid_1's auc: 0.666315
[1170]	training's auc: 0.824179	valid_1's auc: 0.666382
[1180]	training's auc: 0.824353	valid_1's auc: 0.666412
[1190]	training's auc: 0.824485	valid_1's auc: 0.6664
[1200]	training's auc: 0.824757	valid_1's auc: 0.666464
[1210]	training's auc: 0.824926	valid_1's auc: 0.666517
[1220]	training's auc: 0.825128	valid_1's auc: 0.666557
[1230]	training's auc: 0.825368	valid_1's auc: 0.666604
[1240]	training's auc: 0.825549	valid_1's auc: 0.666575
[1250]	training's auc: 0.82575	valid_1's auc: 0.666598
[1260]	training's auc: 0.825922	valid_1's auc: 0.666636
[1270]	training's auc: 0.826101	valid_1's auc: 0.666654
[1280]	training's auc: 0.82637	valid_1's auc: 0.666733
[1290]	training's auc: 0.826656	valid_1's auc: 0.666779
[1300]	training's auc: 0.826859	valid_1's auc: 0.666771
[1310]	training's auc: 0.827015	valid_1's auc: 0.666777
[1320]	training's auc: 0.827206	valid_1's auc: 0.666797
[1330]	training's auc: 0.827363	valid_1's auc: 0.66681
[1340]	training's auc: 0.827525	valid_1's auc: 0.666837
[1350]	training's auc: 0.827661	valid_1's auc: 0.666845
[1360]	training's auc: 0.82779	valid_1's auc: 0.666853
[1370]	training's auc: 0.828012	valid_1's auc: 0.666888
[1380]	training's auc: 0.828144	valid_1's auc: 0.666889
[1390]	training's auc: 0.828339	valid_1's auc: 0.666918
[1400]	training's auc: 0.828484	valid_1's auc: 0.666944
[1410]	training's auc: 0.828605	valid_1's auc: 0.66693
[1420]	training's auc: 0.82873	valid_1's auc: 0.666953
[1430]	training's auc: 0.82885	valid_1's auc: 0.666981
[1440]	training's auc: 0.829029	valid_1's auc: 0.667048
[1450]	training's auc: 0.829159	valid_1's auc: 0.66705
[1460]	training's auc: 0.829277	valid_1's auc: 0.667057
[1470]	training's auc: 0.829445	valid_1's auc: 0.667086
[1480]	training's auc: 0.829568	valid_1's auc: 0.667197
[1490]	training's auc: 0.829708	valid_1's auc: 0.667214
[1500]	training's auc: 0.829825	valid_1's auc: 0.667205
[1510]	training's auc: 0.829941	valid_1's auc: 0.667196
[1520]	training's auc: 0.83007	valid_1's auc: 0.667208
[1530]	training's auc: 0.830183	valid_1's auc: 0.667226
[1540]	training's auc: 0.830298	valid_1's auc: 0.667252
[1550]	training's auc: 0.830406	valid_1's auc: 0.667265
[1560]	training's auc: 0.830549	valid_1's auc: 0.667258
[1570]	training's auc: 0.830648	valid_1's auc: 0.667274
[1580]	training's auc: 0.830761	valid_1's auc: 0.667303
[1590]	training's auc: 0.830887	valid_1's auc: 0.6673
[1600]	training's auc: 0.831031	valid_1's auc: 0.667317
[1610]	training's auc: 0.831148	valid_1's auc: 0.667304
[1620]	training's auc: 0.831262	valid_1's auc: 0.667317
[1630]	training's auc: 0.831392	valid_1's auc: 0.667327
[1640]	training's auc: 0.8315	valid_1's auc: 0.667292
[1650]	training's auc: 0.831628	valid_1's auc: 0.667297
[1660]	training's auc: 0.831726	valid_1's auc: 0.667297
[1670]	training's auc: 0.83191	valid_1's auc: 0.667328
[1680]	training's auc: 0.832008	valid_1's auc: 0.667349
[1690]	training's auc: 0.832112	valid_1's auc: 0.667349
[1700]	training's auc: 0.832226	valid_1's auc: 0.667384
[1710]	training's auc: 0.832337	valid_1's auc: 0.667407
[1720]	training's auc: 0.832434	valid_1's auc: 0.667417
[1730]	training's auc: 0.832563	valid_1's auc: 0.667428
[1740]	training's auc: 0.832661	valid_1's auc: 0.667423
[1750]	training's auc: 0.832772	valid_1's auc: 0.667439
[1760]	training's auc: 0.832873	valid_1's auc: 0.667462
[1770]	training's auc: 0.832984	valid_1's auc: 0.667476
[1780]	training's auc: 0.833063	valid_1's auc: 0.667478
[1790]	training's auc: 0.833169	valid_1's auc: 0.667486
[1800]	training's auc: 0.833272	valid_1's auc: 0.667467
[1810]	training's auc: 0.833402	valid_1's auc: 0.667479
[1820]	training's auc: 0.833517	valid_1's auc: 0.667471
[1830]	training's auc: 0.833642	valid_1's auc: 0.667504
[1840]	training's auc: 0.833753	valid_1's auc: 0.667537
[1850]	training's auc: 0.833862	valid_1's auc: 0.667557
[1860]	training's auc: 0.833986	valid_1's auc: 0.667575
[1870]	training's auc: 0.834098	valid_1's auc: 0.66757
[1880]	training's auc: 0.834195	valid_1's auc: 0.667596
[1890]	training's auc: 0.834301	valid_1's auc: 0.667584
[1900]	training's auc: 0.834377	valid_1's auc: 0.667591
[1910]	training's auc: 0.834486	valid_1's auc: 0.667602
[1920]	training's auc: 0.834596	valid_1's auc: 0.66761
[1930]	training's auc: 0.834693	valid_1's auc: 0.66764
[1940]	training's auc: 0.834803	valid_1's auc: 0.667625
[1950]	training's auc: 0.834915	valid_1's auc: 0.667645
[1960]	training's auc: 0.835036	valid_1's auc: 0.667648
[1970]	training's auc: 0.835106	valid_1's auc: 0.667634
[1980]	training's auc: 0.835218	valid_1's auc: 0.667633
[1990]	training's auc: 0.835315	valid_1's auc: 0.667638
[2000]	training's auc: 0.835395	valid_1's auc: 0.667647
[2010]	training's auc: 0.835476	valid_1's auc: 0.667664
[2020]	training's auc: 0.835575	valid_1's auc: 0.667661
[2030]	training's auc: 0.835687	valid_1's auc: 0.667665
[2040]	training's auc: 0.83579	valid_1's auc: 0.667692
[2050]	training's auc: 0.835883	valid_1's auc: 0.667715
[2060]	training's auc: 0.835976	valid_1's auc: 0.667753
[2070]	training's auc: 0.836071	valid_1's auc: 0.667779
[2080]	training's auc: 0.836161	valid_1's auc: 0.667772
[2090]	training's auc: 0.836287	valid_1's auc: 0.667784
[2100]	training's auc: 0.836388	valid_1's auc: 0.667792
[2110]	training's auc: 0.836484	valid_1's auc: 0.667794
[2120]	training's auc: 0.836573	valid_1's auc: 0.667814
[2130]	training's auc: 0.836648	valid_1's auc: 0.667834
[2140]	training's auc: 0.836732	valid_1's auc: 0.667842
[2150]	training's auc: 0.836813	valid_1's auc: 0.667834
[2160]	training's auc: 0.8369	valid_1's auc: 0.66785
[2170]	training's auc: 0.836994	valid_1's auc: 0.667837
[2180]	training's auc: 0.837141	valid_1's auc: 0.667886
[2190]	training's auc: 0.837227	valid_1's auc: 0.667896
[2200]	training's auc: 0.837363	valid_1's auc: 0.667903
[2210]	training's auc: 0.837449	valid_1's auc: 0.667927
[2220]	training's auc: 0.837541	valid_1's auc: 0.667905
[2230]	training's auc: 0.837642	valid_1's auc: 0.667928
[2240]	training's auc: 0.837726	valid_1's auc: 0.667923
[2250]	training's auc: 0.837798	valid_1's auc: 0.667946
[2260]	training's auc: 0.837876	valid_1's auc: 0.667934
[2270]	training's auc: 0.837963	valid_1's auc: 0.667946
[2280]	training's auc: 0.838044	valid_1's auc: 0.667962
[2290]	training's auc: 0.838131	valid_1's auc: 0.667977
[2300]	training's auc: 0.838206	valid_1's auc: 0.667991
[2310]	training's auc: 0.838292	valid_1's auc: 0.667991
[2320]	training's auc: 0.838377	valid_1's auc: 0.66799
[2330]	training's auc: 0.838461	valid_1's auc: 0.668006
[2340]	training's auc: 0.838547	valid_1's auc: 0.668019
[2350]	training's auc: 0.838646	valid_1's auc: 0.668041
[2360]	training's auc: 0.838727	valid_1's auc: 0.66803
[2370]	training's auc: 0.838809	valid_1's auc: 0.668017
[2380]	training's auc: 0.838905	valid_1's auc: 0.668004
[2390]	training's auc: 0.838976	valid_1's auc: 0.667965
[2400]	training's auc: 0.839067	valid_1's auc: 0.667967
Early stopping, best iteration is:
[2354]	training's auc: 0.838678	valid_1's auc: 0.668048
best score: 0.668047645829
best iteration: 2354
complete on: CC11_source_system_tab

working on: ITC_source_screen_name

Our guest selection:
target                       uint8
msno                      category
song_id                   category
source_system_tab         category
source_screen_name        category
source_type               category
artist_name               category
composer                  category
lyricist                  category
song_year                 category
ITC_source_screen_name       int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.720063	valid_1's auc: 0.629034
[20]	training's auc: 0.734625	valid_1's auc: 0.634839
[30]	training's auc: 0.743341	valid_1's auc: 0.639565
[40]	training's auc: 0.747631	valid_1's auc: 0.641686
[50]	training's auc: 0.751235	valid_1's auc: 0.643263
[60]	training's auc: 0.754103	valid_1's auc: 0.644561
[70]	training's auc: 0.757213	valid_1's auc: 0.645855
[80]	training's auc: 0.759355	valid_1's auc: 0.646845
[90]	training's auc: 0.761725	valid_1's auc: 0.647707
[100]	training's auc: 0.763918	valid_1's auc: 0.648668
[110]	training's auc: 0.766056	valid_1's auc: 0.649472
[120]	training's auc: 0.767972	valid_1's auc: 0.650156
[130]	training's auc: 0.769585	valid_1's auc: 0.650819
[140]	training's auc: 0.771095	valid_1's auc: 0.651453
[150]	training's auc: 0.772682	valid_1's auc: 0.651757
[160]	training's auc: 0.773956	valid_1's auc: 0.652272
[170]	training's auc: 0.775287	valid_1's auc: 0.653063
[180]	training's auc: 0.776678	valid_1's auc: 0.653644
[190]	training's auc: 0.777826	valid_1's auc: 0.654011
[200]	training's auc: 0.778841	valid_1's auc: 0.654395
[210]	training's auc: 0.779946	valid_1's auc: 0.654836
[220]	training's auc: 0.781	valid_1's auc: 0.65515
[230]	training's auc: 0.78192	valid_1's auc: 0.655543
[240]	training's auc: 0.782839	valid_1's auc: 0.655797
[250]	training's auc: 0.783932	valid_1's auc: 0.65609
[260]	training's auc: 0.785245	valid_1's auc: 0.656507
[270]	training's auc: 0.786044	valid_1's auc: 0.656749
[280]	training's auc: 0.786968	valid_1's auc: 0.657071
[290]	training's auc: 0.787675	valid_1's auc: 0.657273
[300]	training's auc: 0.788721	valid_1's auc: 0.657598
[310]	training's auc: 0.789488	valid_1's auc: 0.65768
[320]	training's auc: 0.790216	valid_1's auc: 0.657985
[330]	training's auc: 0.79091	valid_1's auc: 0.658125
[340]	training's auc: 0.791647	valid_1's auc: 0.658351
[350]	training's auc: 0.792609	valid_1's auc: 0.658747
[360]	training's auc: 0.793208	valid_1's auc: 0.658881
[370]	training's auc: 0.793897	valid_1's auc: 0.659148
[380]	training's auc: 0.794668	valid_1's auc: 0.659439
[390]	training's auc: 0.795204	valid_1's auc: 0.659528
[400]	training's auc: 0.795665	valid_1's auc: 0.659701
[410]	training's auc: 0.796233	valid_1's auc: 0.659834
[420]	training's auc: 0.796911	valid_1's auc: 0.659996
[430]	training's auc: 0.79744	valid_1's auc: 0.660151
[440]	training's auc: 0.798114	valid_1's auc: 0.660266
[450]	training's auc: 0.798664	valid_1's auc: 0.660339
[460]	training's auc: 0.799371	valid_1's auc: 0.660667
[470]	training's auc: 0.799861	valid_1's auc: 0.660801
[480]	training's auc: 0.800402	valid_1's auc: 0.660973
[490]	training's auc: 0.80113	valid_1's auc: 0.661203
[500]	training's auc: 0.801615	valid_1's auc: 0.661296
[510]	training's auc: 0.80221	valid_1's auc: 0.661439
[520]	training's auc: 0.803114	valid_1's auc: 0.66174
[530]	training's auc: 0.803659	valid_1's auc: 0.66194
[540]	training's auc: 0.804109	valid_1's auc: 0.662048
[550]	training's auc: 0.804485	valid_1's auc: 0.662161
[560]	training's auc: 0.804956	valid_1's auc: 0.66229
[570]	training's auc: 0.805475	valid_1's auc: 0.662389
[580]	training's auc: 0.805903	valid_1's auc: 0.66257
[590]	training's auc: 0.80635	valid_1's auc: 0.662671
[600]	training's auc: 0.806937	valid_1's auc: 0.662773
[610]	training's auc: 0.807247	valid_1's auc: 0.662862
[620]	training's auc: 0.807654	valid_1's auc: 0.662913
[630]	training's auc: 0.808033	valid_1's auc: 0.663084
[640]	training's auc: 0.808416	valid_1's auc: 0.663193
[650]	training's auc: 0.808831	valid_1's auc: 0.663299
[660]	training's auc: 0.809169	valid_1's auc: 0.663349
[670]	training's auc: 0.809649	valid_1's auc: 0.663455
[680]	training's auc: 0.810111	valid_1's auc: 0.663589
[690]	training's auc: 0.810431	valid_1's auc: 0.663654
[700]	training's auc: 0.810818	valid_1's auc: 0.663789
[710]	training's auc: 0.811149	valid_1's auc: 0.663831
[720]	training's auc: 0.811489	valid_1's auc: 0.663887
[730]	training's auc: 0.811882	valid_1's auc: 0.663921
[740]	training's auc: 0.812318	valid_1's auc: 0.664022
[750]	training's auc: 0.81262	valid_1's auc: 0.664099
[760]	training's auc: 0.812967	valid_1's auc: 0.664129
[770]	training's auc: 0.813263	valid_1's auc: 0.664194
[780]	training's auc: 0.813719	valid_1's auc: 0.664303
[790]	training's auc: 0.814102	valid_1's auc: 0.664333
[800]	training's auc: 0.814418	valid_1's auc: 0.664417
[810]	training's auc: 0.814696	valid_1's auc: 0.664459
[820]	training's auc: 0.815094	valid_1's auc: 0.664574
[830]	training's auc: 0.815437	valid_1's auc: 0.664702
[840]	training's auc: 0.815772	valid_1's auc: 0.664765
[850]	training's auc: 0.816194	valid_1's auc: 0.664855
[860]	training's auc: 0.816476	valid_1's auc: 0.664844
[870]	training's auc: 0.816775	valid_1's auc: 0.664893
[880]	training's auc: 0.817089	valid_1's auc: 0.664957
[890]	training's auc: 0.817438	valid_1's auc: 0.665065
[900]	training's auc: 0.817678	valid_1's auc: 0.665071
[910]	training's auc: 0.817927	valid_1's auc: 0.665066
[920]	training's auc: 0.818222	valid_1's auc: 0.665146
[930]	training's auc: 0.818528	valid_1's auc: 0.665171
[940]	training's auc: 0.818825	valid_1's auc: 0.665171
[950]	training's auc: 0.819102	valid_1's auc: 0.665229
[960]	training's auc: 0.81939	valid_1's auc: 0.665293
[970]	training's auc: 0.819682	valid_1's auc: 0.665341
[980]	training's auc: 0.819925	valid_1's auc: 0.665382
[990]	training's auc: 0.820223	valid_1's auc: 0.665427
[1000]	training's auc: 0.820522	valid_1's auc: 0.665537
[1010]	training's auc: 0.820796	valid_1's auc: 0.665621
[1020]	training's auc: 0.82103	valid_1's auc: 0.665647
[1030]	training's auc: 0.821275	valid_1's auc: 0.665664
[1040]	training's auc: 0.821609	valid_1's auc: 0.665668
[1050]	training's auc: 0.821879	valid_1's auc: 0.665708
[1060]	training's auc: 0.822135	valid_1's auc: 0.665749
[1070]	training's auc: 0.822328	valid_1's auc: 0.66574
[1080]	training's auc: 0.822558	valid_1's auc: 0.66576
[1090]	training's auc: 0.822752	valid_1's auc: 0.665779
[1100]	training's auc: 0.822965	valid_1's auc: 0.665778
[1110]	training's auc: 0.823196	valid_1's auc: 0.665841
[1120]	training's auc: 0.823413	valid_1's auc: 0.665907
[1130]	training's auc: 0.823662	valid_1's auc: 0.665926
[1140]	training's auc: 0.82388	valid_1's auc: 0.665953
[1150]	training's auc: 0.824065	valid_1's auc: 0.665978
[1160]	training's auc: 0.824287	valid_1's auc: 0.666014
[1170]	training's auc: 0.824676	valid_1's auc: 0.666096
[1180]	training's auc: 0.824838	valid_1's auc: 0.66613
[1190]	training's auc: 0.825007	valid_1's auc: 0.666204
[1200]	training's auc: 0.825235	valid_1's auc: 0.666246
[1210]	training's auc: 0.825416	valid_1's auc: 0.666234
[1220]	training's auc: 0.82561	valid_1's auc: 0.666216
[1230]	training's auc: 0.825838	valid_1's auc: 0.666251
[1240]	training's auc: 0.82611	valid_1's auc: 0.666341
[1250]	training's auc: 0.826264	valid_1's auc: 0.666359
[1260]	training's auc: 0.826454	valid_1's auc: 0.66643
[1270]	training's auc: 0.826697	valid_1's auc: 0.66646
[1280]	training's auc: 0.826893	valid_1's auc: 0.666455
[1290]	training's auc: 0.827101	valid_1's auc: 0.666493
[1300]	training's auc: 0.827231	valid_1's auc: 0.666504
[1310]	training's auc: 0.827393	valid_1's auc: 0.666513
[1320]	training's auc: 0.827531	valid_1's auc: 0.66652
[1330]	training's auc: 0.827679	valid_1's auc: 0.666534
[1340]	training's auc: 0.827944	valid_1's auc: 0.666604
[1350]	training's auc: 0.828097	valid_1's auc: 0.666647
[1360]	training's auc: 0.828249	valid_1's auc: 0.666673
[1370]	training's auc: 0.828416	valid_1's auc: 0.666695
[1380]	training's auc: 0.828563	valid_1's auc: 0.666705
[1390]	training's auc: 0.828707	valid_1's auc: 0.666703
[1400]	training's auc: 0.828847	valid_1's auc: 0.666689
[1410]	training's auc: 0.828993	valid_1's auc: 0.666677
[1420]	training's auc: 0.829203	valid_1's auc: 0.666692
[1430]	training's auc: 0.829318	valid_1's auc: 0.666721
[1440]	training's auc: 0.829457	valid_1's auc: 0.666701
[1450]	training's auc: 0.82964	valid_1's auc: 0.666709
[1460]	training's auc: 0.829771	valid_1's auc: 0.666725
[1470]	training's auc: 0.829907	valid_1's auc: 0.666723
[1480]	training's auc: 0.830046	valid_1's auc: 0.666751
[1490]	training's auc: 0.830201	valid_1's auc: 0.666751
[1500]	training's auc: 0.830345	valid_1's auc: 0.666735
[1510]	training's auc: 0.830489	valid_1's auc: 0.666735
[1520]	training's auc: 0.830617	valid_1's auc: 0.666767
[1530]	training's auc: 0.83075	valid_1's auc: 0.666791
[1540]	training's auc: 0.83087	valid_1's auc: 0.666779
[1550]	training's auc: 0.831004	valid_1's auc: 0.666781
[1560]	training's auc: 0.831131	valid_1's auc: 0.666776
[1570]	training's auc: 0.831241	valid_1's auc: 0.666791
[1580]	training's auc: 0.831372	valid_1's auc: 0.666808
[1590]	training's auc: 0.831484	valid_1's auc: 0.666847
[1600]	training's auc: 0.831641	valid_1's auc: 0.666847
[1610]	training's auc: 0.831778	valid_1's auc: 0.666857
[1620]	training's auc: 0.831884	valid_1's auc: 0.666879
[1630]	training's auc: 0.832003	valid_1's auc: 0.666889
[1640]	training's auc: 0.832107	valid_1's auc: 0.666901
[1650]	training's auc: 0.832247	valid_1's auc: 0.666892
[1660]	training's auc: 0.832358	valid_1's auc: 0.6669
[1670]	training's auc: 0.832473	valid_1's auc: 0.666903
[1680]	training's auc: 0.832601	valid_1's auc: 0.666942
[1690]	training's auc: 0.832714	valid_1's auc: 0.666961
[1700]	training's auc: 0.832846	valid_1's auc: 0.666973
[1710]	training's auc: 0.832963	valid_1's auc: 0.666994
[1720]	training's auc: 0.833069	valid_1's auc: 0.667002
[1730]	training's auc: 0.833174	valid_1's auc: 0.667055
[1740]	training's auc: 0.833271	valid_1's auc: 0.66709
[1750]	training's auc: 0.833387	valid_1's auc: 0.667098
[1760]	training's auc: 0.833465	valid_1's auc: 0.667113
[1770]	training's auc: 0.833594	valid_1's auc: 0.66713
[1780]	training's auc: 0.833694	valid_1's auc: 0.667125
[1790]	training's auc: 0.83385	valid_1's auc: 0.667168
[1800]	training's auc: 0.833955	valid_1's auc: 0.667176
[1810]	training's auc: 0.834061	valid_1's auc: 0.667183
[1820]	training's auc: 0.834161	valid_1's auc: 0.667229
[1830]	training's auc: 0.834273	valid_1's auc: 0.667222
[1840]	training's auc: 0.834362	valid_1's auc: 0.667233
[1850]	training's auc: 0.834464	valid_1's auc: 0.667276
[1860]	training's auc: 0.834577	valid_1's auc: 0.66728
[1870]	training's auc: 0.834691	valid_1's auc: 0.667283
[1880]	training's auc: 0.834787	valid_1's auc: 0.667272
[1890]	training's auc: 0.834881	valid_1's auc: 0.667303
[1900]	training's auc: 0.834977	valid_1's auc: 0.667308
[1910]	training's auc: 0.835073	valid_1's auc: 0.667301
[1920]	training's auc: 0.835179	valid_1's auc: 0.667325
[1930]	training's auc: 0.835273	valid_1's auc: 0.667334
[1940]	training's auc: 0.83538	valid_1's auc: 0.667346
[1950]	training's auc: 0.835478	valid_1's auc: 0.667346
[1960]	training's auc: 0.83557	valid_1's auc: 0.667377
[1970]	training's auc: 0.835654	valid_1's auc: 0.667444
[1980]	training's auc: 0.835749	valid_1's auc: 0.667459
[1990]	training's auc: 0.835827	valid_1's auc: 0.667449
[2000]	training's auc: 0.8359	valid_1's auc: 0.667457
[2010]	training's auc: 0.836002	valid_1's auc: 0.667476
[2020]	training's auc: 0.836111	valid_1's auc: 0.667468
[2030]	training's auc: 0.836278	valid_1's auc: 0.667503
[2040]	training's auc: 0.836367	valid_1's auc: 0.667481
[2050]	training's auc: 0.836461	valid_1's auc: 0.667456
[2060]	training's auc: 0.83656	valid_1's auc: 0.66745
[2070]	training's auc: 0.83667	valid_1's auc: 0.667454
[2080]	training's auc: 0.836787	valid_1's auc: 0.667503
[2090]	training's auc: 0.836897	valid_1's auc: 0.667525
[2100]	training's auc: 0.836985	valid_1's auc: 0.667518
[2110]	training's auc: 0.837081	valid_1's auc: 0.66752
[2120]	training's auc: 0.837166	valid_1's auc: 0.667512
[2130]	training's auc: 0.837255	valid_1's auc: 0.667517
[2140]	training's auc: 0.837359	valid_1's auc: 0.667539
[2150]	training's auc: 0.837449	valid_1's auc: 0.667549
[2160]	training's auc: 0.837552	valid_1's auc: 0.667525
[2170]	training's auc: 0.837655	valid_1's auc: 0.667562
[2180]	training's auc: 0.837767	valid_1's auc: 0.66759
[2190]	training's auc: 0.837864	valid_1's auc: 0.667572
[2200]	training's auc: 0.837955	valid_1's auc: 0.667548
[2210]	training's auc: 0.838056	valid_1's auc: 0.667567
[2220]	training's auc: 0.838143	valid_1's auc: 0.667567
Early stopping, best iteration is:
[2177]	training's auc: 0.837733	valid_1's auc: 0.667597
best score: 0.667597465014
best iteration: 2177
complete on: ITC_source_screen_name

working on: CC11_source_screen_name

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
CC11_source_screen_name       int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.720793	valid_1's auc: 0.630396
[20]	training's auc: 0.736557	valid_1's auc: 0.636675
[30]	training's auc: 0.744109	valid_1's auc: 0.640357
[40]	training's auc: 0.748622	valid_1's auc: 0.6424
[50]	training's auc: 0.752222	valid_1's auc: 0.644379
[60]	training's auc: 0.754943	valid_1's auc: 0.645441
[70]	training's auc: 0.75761	valid_1's auc: 0.646477
[80]	training's auc: 0.759925	valid_1's auc: 0.647611
[90]	training's auc: 0.762283	valid_1's auc: 0.648495
[100]	training's auc: 0.764523	valid_1's auc: 0.648893
[110]	training's auc: 0.766417	valid_1's auc: 0.649712
[120]	training's auc: 0.768246	valid_1's auc: 0.650443
[130]	training's auc: 0.770135	valid_1's auc: 0.651267
[140]	training's auc: 0.771771	valid_1's auc: 0.65172
[150]	training's auc: 0.773352	valid_1's auc: 0.652324
[160]	training's auc: 0.774707	valid_1's auc: 0.652836
[170]	training's auc: 0.775964	valid_1's auc: 0.65333
[180]	training's auc: 0.777378	valid_1's auc: 0.653919
[190]	training's auc: 0.778385	valid_1's auc: 0.65438
[200]	training's auc: 0.779455	valid_1's auc: 0.654756
[210]	training's auc: 0.780502	valid_1's auc: 0.655174
[220]	training's auc: 0.781591	valid_1's auc: 0.65573
[230]	training's auc: 0.782551	valid_1's auc: 0.656088
[240]	training's auc: 0.783516	valid_1's auc: 0.656397
[250]	training's auc: 0.784511	valid_1's auc: 0.656699
[260]	training's auc: 0.785406	valid_1's auc: 0.656906
[270]	training's auc: 0.786209	valid_1's auc: 0.657138
[280]	training's auc: 0.787366	valid_1's auc: 0.657623
[290]	training's auc: 0.788249	valid_1's auc: 0.657961
[300]	training's auc: 0.788951	valid_1's auc: 0.658163
[310]	training's auc: 0.789783	valid_1's auc: 0.658424
[320]	training's auc: 0.790781	valid_1's auc: 0.658811
[330]	training's auc: 0.791404	valid_1's auc: 0.659023
[340]	training's auc: 0.792074	valid_1's auc: 0.659257
[350]	training's auc: 0.792741	valid_1's auc: 0.659479
[360]	training's auc: 0.793406	valid_1's auc: 0.659716
[370]	training's auc: 0.793965	valid_1's auc: 0.659854
[380]	training's auc: 0.794769	valid_1's auc: 0.660073
[390]	training's auc: 0.795309	valid_1's auc: 0.660245
[400]	training's auc: 0.795861	valid_1's auc: 0.660443
[410]	training's auc: 0.796481	valid_1's auc: 0.66059
[420]	training's auc: 0.797039	valid_1's auc: 0.66077
[430]	training's auc: 0.797589	valid_1's auc: 0.660952
[440]	training's auc: 0.798242	valid_1's auc: 0.661133
[450]	training's auc: 0.798784	valid_1's auc: 0.661191
[460]	training's auc: 0.79962	valid_1's auc: 0.661489
[470]	training's auc: 0.800105	valid_1's auc: 0.661598
[480]	training's auc: 0.800779	valid_1's auc: 0.661864
[490]	training's auc: 0.801323	valid_1's auc: 0.661918
[500]	training's auc: 0.801867	valid_1's auc: 0.662071
[510]	training's auc: 0.80229	valid_1's auc: 0.662189
[520]	training's auc: 0.802751	valid_1's auc: 0.662315
[530]	training's auc: 0.803344	valid_1's auc: 0.662542
[540]	training's auc: 0.803963	valid_1's auc: 0.662666
[550]	training's auc: 0.804576	valid_1's auc: 0.662772
[560]	training's auc: 0.805026	valid_1's auc: 0.662903
[570]	training's auc: 0.805585	valid_1's auc: 0.663012
[580]	training's auc: 0.80596	valid_1's auc: 0.663098
[590]	training's auc: 0.806301	valid_1's auc: 0.663172
[600]	training's auc: 0.806753	valid_1's auc: 0.663289
[610]	training's auc: 0.8071	valid_1's auc: 0.663386
[620]	training's auc: 0.807479	valid_1's auc: 0.663432
[630]	training's auc: 0.807936	valid_1's auc: 0.663566
[640]	training's auc: 0.808362	valid_1's auc: 0.663666
[650]	training's auc: 0.808689	valid_1's auc: 0.663666
[660]	training's auc: 0.80924	valid_1's auc: 0.663872
[670]	training's auc: 0.809578	valid_1's auc: 0.66393
[680]	training's auc: 0.809907	valid_1's auc: 0.664006
[690]	training's auc: 0.810254	valid_1's auc: 0.664082
[700]	training's auc: 0.810662	valid_1's auc: 0.664142
[710]	training's auc: 0.81107	valid_1's auc: 0.664236
[720]	training's auc: 0.81141	valid_1's auc: 0.664325
[730]	training's auc: 0.811885	valid_1's auc: 0.664384
[740]	training's auc: 0.812245	valid_1's auc: 0.664443
[750]	training's auc: 0.812527	valid_1's auc: 0.6645
[760]	training's auc: 0.812858	valid_1's auc: 0.664526
[770]	training's auc: 0.813265	valid_1's auc: 0.664616
[780]	training's auc: 0.813645	valid_1's auc: 0.66468
[790]	training's auc: 0.813955	valid_1's auc: 0.664724
[800]	training's auc: 0.814336	valid_1's auc: 0.664859
[810]	training's auc: 0.81462	valid_1's auc: 0.66494
[820]	training's auc: 0.814905	valid_1's auc: 0.664944
[830]	training's auc: 0.815284	valid_1's auc: 0.665025
[840]	training's auc: 0.815597	valid_1's auc: 0.665032
[850]	training's auc: 0.815971	valid_1's auc: 0.665041
[860]	training's auc: 0.816358	valid_1's auc: 0.665071
[870]	training's auc: 0.816651	valid_1's auc: 0.665166
[880]	training's auc: 0.816963	valid_1's auc: 0.66522
[890]	training's auc: 0.817214	valid_1's auc: 0.665263
[900]	training's auc: 0.817513	valid_1's auc: 0.665324
[910]	training's auc: 0.817768	valid_1's auc: 0.665348
[920]	training's auc: 0.818055	valid_1's auc: 0.665377
[930]	training's auc: 0.818407	valid_1's auc: 0.665425
[940]	training's auc: 0.818679	valid_1's auc: 0.66546
[950]	training's auc: 0.818929	valid_1's auc: 0.66549
[960]	training's auc: 0.81924	valid_1's auc: 0.665566
[970]	training's auc: 0.819546	valid_1's auc: 0.665578
[980]	training's auc: 0.819869	valid_1's auc: 0.665661
[990]	training's auc: 0.820105	valid_1's auc: 0.665708
[1000]	training's auc: 0.820388	valid_1's auc: 0.66575
[1010]	training's auc: 0.820596	valid_1's auc: 0.665745
[1020]	training's auc: 0.820827	valid_1's auc: 0.665796
[1030]	training's auc: 0.821224	valid_1's auc: 0.665944
[1040]	training's auc: 0.821479	valid_1's auc: 0.66596
[1050]	training's auc: 0.821709	valid_1's auc: 0.665982
[1060]	training's auc: 0.821905	valid_1's auc: 0.665969
[1070]	training's auc: 0.822101	valid_1's auc: 0.666043
[1080]	training's auc: 0.822357	valid_1's auc: 0.666092
[1090]	training's auc: 0.822537	valid_1's auc: 0.666096
[1100]	training's auc: 0.822746	valid_1's auc: 0.666105
[1110]	training's auc: 0.82294	valid_1's auc: 0.666086
[1120]	training's auc: 0.82315	valid_1's auc: 0.666115
[1130]	training's auc: 0.823354	valid_1's auc: 0.666148
[1140]	training's auc: 0.82352	valid_1's auc: 0.666192
[1150]	training's auc: 0.823722	valid_1's auc: 0.666232
[1160]	training's auc: 0.82391	valid_1's auc: 0.666231
[1170]	training's auc: 0.824227	valid_1's auc: 0.666246
[1180]	training's auc: 0.82442	valid_1's auc: 0.666289
[1190]	training's auc: 0.824586	valid_1's auc: 0.666317
[1200]	training's auc: 0.824954	valid_1's auc: 0.666394
[1210]	training's auc: 0.825137	valid_1's auc: 0.666403
[1220]	training's auc: 0.82531	valid_1's auc: 0.666418
[1230]	training's auc: 0.825531	valid_1's auc: 0.666437
[1240]	training's auc: 0.825726	valid_1's auc: 0.666443
[1250]	training's auc: 0.825972	valid_1's auc: 0.666454
[1260]	training's auc: 0.826153	valid_1's auc: 0.666468
[1270]	training's auc: 0.826375	valid_1's auc: 0.666512
[1280]	training's auc: 0.826553	valid_1's auc: 0.666523
[1290]	training's auc: 0.826775	valid_1's auc: 0.666549
[1300]	training's auc: 0.826933	valid_1's auc: 0.666584
[1310]	training's auc: 0.827113	valid_1's auc: 0.666582
[1320]	training's auc: 0.827266	valid_1's auc: 0.666555
[1330]	training's auc: 0.827461	valid_1's auc: 0.666577
[1340]	training's auc: 0.827708	valid_1's auc: 0.666572
[1350]	training's auc: 0.82787	valid_1's auc: 0.666595
[1360]	training's auc: 0.828012	valid_1's auc: 0.666605
[1370]	training's auc: 0.828172	valid_1's auc: 0.666605
[1380]	training's auc: 0.828299	valid_1's auc: 0.666603
[1390]	training's auc: 0.828468	valid_1's auc: 0.666634
[1400]	training's auc: 0.828606	valid_1's auc: 0.66664
[1410]	training's auc: 0.828773	valid_1's auc: 0.666682
[1420]	training's auc: 0.828934	valid_1's auc: 0.666729
[1430]	training's auc: 0.829057	valid_1's auc: 0.666743
[1440]	training's auc: 0.829234	valid_1's auc: 0.666755
[1450]	training's auc: 0.829369	valid_1's auc: 0.666765
[1460]	training's auc: 0.829481	valid_1's auc: 0.66676
[1470]	training's auc: 0.829637	valid_1's auc: 0.666765
[1480]	training's auc: 0.829767	valid_1's auc: 0.666767
[1490]	training's auc: 0.829898	valid_1's auc: 0.666769
[1500]	training's auc: 0.830079	valid_1's auc: 0.66683
[1510]	training's auc: 0.8302	valid_1's auc: 0.666837
[1520]	training's auc: 0.830304	valid_1's auc: 0.666848
[1530]	training's auc: 0.830453	valid_1's auc: 0.666877
[1540]	training's auc: 0.830575	valid_1's auc: 0.666886
[1550]	training's auc: 0.830692	valid_1's auc: 0.666903
[1560]	training's auc: 0.830832	valid_1's auc: 0.666907
[1570]	training's auc: 0.830925	valid_1's auc: 0.666902
[1580]	training's auc: 0.831068	valid_1's auc: 0.666918
[1590]	training's auc: 0.831193	valid_1's auc: 0.666911
[1600]	training's auc: 0.831323	valid_1's auc: 0.666914
[1610]	training's auc: 0.83146	valid_1's auc: 0.666929
[1620]	training's auc: 0.831681	valid_1's auc: 0.667025
[1630]	training's auc: 0.831826	valid_1's auc: 0.667043
[1640]	training's auc: 0.831942	valid_1's auc: 0.667068
[1650]	training's auc: 0.832076	valid_1's auc: 0.667091
[1660]	training's auc: 0.832226	valid_1's auc: 0.667102
[1670]	training's auc: 0.832365	valid_1's auc: 0.667109
[1680]	training's auc: 0.832486	valid_1's auc: 0.667159
[1690]	training's auc: 0.832598	valid_1's auc: 0.667164
[1700]	training's auc: 0.832718	valid_1's auc: 0.667181
[1710]	training's auc: 0.832833	valid_1's auc: 0.66717
[1720]	training's auc: 0.832932	valid_1's auc: 0.667158
[1730]	training's auc: 0.833052	valid_1's auc: 0.66715
[1740]	training's auc: 0.833151	valid_1's auc: 0.667161
[1750]	training's auc: 0.833267	valid_1's auc: 0.667169
Early stopping, best iteration is:
[1703]	training's auc: 0.832748	valid_1's auc: 0.667182
best score: 0.667181552659
best iteration: 1703
complete on: CC11_source_screen_name

working on: ITC_source_type

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
ITC_source_type          int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721252	valid_1's auc: 0.630243
[20]	training's auc: 0.735338	valid_1's auc: 0.63646
[30]	training's auc: 0.742998	valid_1's auc: 0.640209
[40]	training's auc: 0.747296	valid_1's auc: 0.642203
[50]	training's auc: 0.750704	valid_1's auc: 0.643775
[60]	training's auc: 0.753798	valid_1's auc: 0.645285
[70]	training's auc: 0.756505	valid_1's auc: 0.64647
[80]	training's auc: 0.759169	valid_1's auc: 0.647475
[90]	training's auc: 0.761592	valid_1's auc: 0.648565
[100]	training's auc: 0.763481	valid_1's auc: 0.649399
[110]	training's auc: 0.765312	valid_1's auc: 0.65014
[120]	training's auc: 0.767309	valid_1's auc: 0.650922
[130]	training's auc: 0.768943	valid_1's auc: 0.65157
[140]	training's auc: 0.770383	valid_1's auc: 0.65221
[150]	training's auc: 0.771893	valid_1's auc: 0.652729
[160]	training's auc: 0.77383	valid_1's auc: 0.653255
[170]	training's auc: 0.775377	valid_1's auc: 0.654008
[180]	training's auc: 0.776692	valid_1's auc: 0.654456
[190]	training's auc: 0.777857	valid_1's auc: 0.654893
[200]	training's auc: 0.778884	valid_1's auc: 0.655386
[210]	training's auc: 0.779949	valid_1's auc: 0.655713
[220]	training's auc: 0.780874	valid_1's auc: 0.656091
[230]	training's auc: 0.781829	valid_1's auc: 0.65639
[240]	training's auc: 0.78276	valid_1's auc: 0.656663
[250]	training's auc: 0.783724	valid_1's auc: 0.657013
[260]	training's auc: 0.784652	valid_1's auc: 0.657392
[270]	training's auc: 0.785697	valid_1's auc: 0.65769
[280]	training's auc: 0.786494	valid_1's auc: 0.657933
[290]	training's auc: 0.787394	valid_1's auc: 0.658227
[300]	training's auc: 0.788465	valid_1's auc: 0.658547
[310]	training's auc: 0.789244	valid_1's auc: 0.658733
[320]	training's auc: 0.790042	valid_1's auc: 0.659066
[330]	training's auc: 0.790847	valid_1's auc: 0.659379
[340]	training's auc: 0.791712	valid_1's auc: 0.659723
[350]	training's auc: 0.792623	valid_1's auc: 0.660013
[360]	training's auc: 0.793232	valid_1's auc: 0.66013
[370]	training's auc: 0.793959	valid_1's auc: 0.660366
[380]	training's auc: 0.794589	valid_1's auc: 0.660319
[390]	training's auc: 0.795111	valid_1's auc: 0.66049
[400]	training's auc: 0.795581	valid_1's auc: 0.660729
[410]	training's auc: 0.796229	valid_1's auc: 0.660898
[420]	training's auc: 0.796759	valid_1's auc: 0.661019
[430]	training's auc: 0.797307	valid_1's auc: 0.661182
[440]	training's auc: 0.797915	valid_1's auc: 0.661388
[450]	training's auc: 0.798588	valid_1's auc: 0.661611
[460]	training's auc: 0.799164	valid_1's auc: 0.661772
[470]	training's auc: 0.799649	valid_1's auc: 0.661888
[480]	training's auc: 0.800246	valid_1's auc: 0.662127
[490]	training's auc: 0.801087	valid_1's auc: 0.662444
[500]	training's auc: 0.801536	valid_1's auc: 0.662549
[510]	training's auc: 0.802136	valid_1's auc: 0.662741
[520]	training's auc: 0.802795	valid_1's auc: 0.662947
[530]	training's auc: 0.803237	valid_1's auc: 0.663042
[540]	training's auc: 0.803832	valid_1's auc: 0.663277
[550]	training's auc: 0.804244	valid_1's auc: 0.663376
[560]	training's auc: 0.8047	valid_1's auc: 0.663452
[570]	training's auc: 0.805116	valid_1's auc: 0.663596
[580]	training's auc: 0.805549	valid_1's auc: 0.663693
[590]	training's auc: 0.805927	valid_1's auc: 0.663763
[600]	training's auc: 0.806342	valid_1's auc: 0.663869
[610]	training's auc: 0.806673	valid_1's auc: 0.663928
[620]	training's auc: 0.807139	valid_1's auc: 0.664022
[630]	training's auc: 0.807537	valid_1's auc: 0.66412
[640]	training's auc: 0.807851	valid_1's auc: 0.664173
[650]	training's auc: 0.808304	valid_1's auc: 0.664286
[660]	training's auc: 0.808639	valid_1's auc: 0.664333
[670]	training's auc: 0.809003	valid_1's auc: 0.66433
[680]	training's auc: 0.809509	valid_1's auc: 0.664433
[690]	training's auc: 0.809958	valid_1's auc: 0.664583
[700]	training's auc: 0.810288	valid_1's auc: 0.664634
[710]	training's auc: 0.81084	valid_1's auc: 0.664784
[720]	training's auc: 0.811188	valid_1's auc: 0.664845
[730]	training's auc: 0.81172	valid_1's auc: 0.665064
[740]	training's auc: 0.81218	valid_1's auc: 0.665204
[750]	training's auc: 0.812473	valid_1's auc: 0.665254
[760]	training's auc: 0.812853	valid_1's auc: 0.665338
[770]	training's auc: 0.813263	valid_1's auc: 0.665492
[780]	training's auc: 0.81364	valid_1's auc: 0.665546
[790]	training's auc: 0.813975	valid_1's auc: 0.665583
[800]	training's auc: 0.814308	valid_1's auc: 0.665611
[810]	training's auc: 0.814598	valid_1's auc: 0.665612
[820]	training's auc: 0.815077	valid_1's auc: 0.665694
[830]	training's auc: 0.815336	valid_1's auc: 0.665714
[840]	training's auc: 0.815894	valid_1's auc: 0.665913
[850]	training's auc: 0.816182	valid_1's auc: 0.665936
[860]	training's auc: 0.816466	valid_1's auc: 0.665964
[870]	training's auc: 0.816846	valid_1's auc: 0.666082
[880]	training's auc: 0.817242	valid_1's auc: 0.666198
[890]	training's auc: 0.817557	valid_1's auc: 0.666204
[900]	training's auc: 0.817852	valid_1's auc: 0.666263
[910]	training's auc: 0.818227	valid_1's auc: 0.666355
[920]	training's auc: 0.818459	valid_1's auc: 0.666412
[930]	training's auc: 0.818703	valid_1's auc: 0.666452
[940]	training's auc: 0.818954	valid_1's auc: 0.666546
[950]	training's auc: 0.819311	valid_1's auc: 0.666659
[960]	training's auc: 0.8196	valid_1's auc: 0.666708
[970]	training's auc: 0.819836	valid_1's auc: 0.66675
[980]	training's auc: 0.820088	valid_1's auc: 0.666765
[990]	training's auc: 0.820444	valid_1's auc: 0.666826
[1000]	training's auc: 0.82064	valid_1's auc: 0.666881
[1010]	training's auc: 0.820921	valid_1's auc: 0.666944
[1020]	training's auc: 0.821153	valid_1's auc: 0.667013
[1030]	training's auc: 0.821372	valid_1's auc: 0.667033
[1040]	training's auc: 0.82169	valid_1's auc: 0.667013
[1050]	training's auc: 0.82191	valid_1's auc: 0.667009
[1060]	training's auc: 0.82211	valid_1's auc: 0.667024
[1070]	training's auc: 0.822269	valid_1's auc: 0.667018
[1080]	training's auc: 0.822504	valid_1's auc: 0.667049
[1090]	training's auc: 0.822754	valid_1's auc: 0.667072
[1100]	training's auc: 0.823029	valid_1's auc: 0.667108
[1110]	training's auc: 0.823277	valid_1's auc: 0.66714
[1120]	training's auc: 0.823467	valid_1's auc: 0.667013
[1130]	training's auc: 0.823671	valid_1's auc: 0.667018
[1140]	training's auc: 0.823854	valid_1's auc: 0.667039
[1150]	training's auc: 0.824067	valid_1's auc: 0.667064
[1160]	training's auc: 0.824257	valid_1's auc: 0.667064
Early stopping, best iteration is:
[1111]	training's auc: 0.8233	valid_1's auc: 0.66715
best score: 0.667150395288
best iteration: 1111
complete on: ITC_source_type

working on: CC11_source_type

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
CC11_source_type         int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721442	valid_1's auc: 0.629842
[20]	training's auc: 0.736649	valid_1's auc: 0.63661
[30]	training's auc: 0.743839	valid_1's auc: 0.640656
[40]	training's auc: 0.748477	valid_1's auc: 0.64289
[50]	training's auc: 0.75208	valid_1's auc: 0.644771
[60]	training's auc: 0.754906	valid_1's auc: 0.645971
[70]	training's auc: 0.757518	valid_1's auc: 0.646973
[80]	training's auc: 0.760041	valid_1's auc: 0.647854
[90]	training's auc: 0.762209	valid_1's auc: 0.648644
[100]	training's auc: 0.764756	valid_1's auc: 0.64956
[110]	training's auc: 0.766688	valid_1's auc: 0.650306
[120]	training's auc: 0.768516	valid_1's auc: 0.65087
[130]	training's auc: 0.770126	valid_1's auc: 0.65163
[140]	training's auc: 0.771475	valid_1's auc: 0.6523
[150]	training's auc: 0.773288	valid_1's auc: 0.652632
[160]	training's auc: 0.774964	valid_1's auc: 0.653209
[170]	training's auc: 0.776169	valid_1's auc: 0.653784
[180]	training's auc: 0.777434	valid_1's auc: 0.65417
[190]	training's auc: 0.778596	valid_1's auc: 0.654605
[200]	training's auc: 0.779644	valid_1's auc: 0.655027
[210]	training's auc: 0.780706	valid_1's auc: 0.655421
[220]	training's auc: 0.78159	valid_1's auc: 0.655807
[230]	training's auc: 0.78248	valid_1's auc: 0.656199
[240]	training's auc: 0.783483	valid_1's auc: 0.656544
[250]	training's auc: 0.784535	valid_1's auc: 0.656975
[260]	training's auc: 0.78575	valid_1's auc: 0.657289
[270]	training's auc: 0.786544	valid_1's auc: 0.657513
[280]	training's auc: 0.787546	valid_1's auc: 0.657829
[290]	training's auc: 0.788598	valid_1's auc: 0.658048
[300]	training's auc: 0.789337	valid_1's auc: 0.65827
[310]	training's auc: 0.790164	valid_1's auc: 0.658471
[320]	training's auc: 0.790874	valid_1's auc: 0.658556
[330]	training's auc: 0.791772	valid_1's auc: 0.658774
[340]	training's auc: 0.792438	valid_1's auc: 0.658955
[350]	training's auc: 0.793194	valid_1's auc: 0.659217
[360]	training's auc: 0.793746	valid_1's auc: 0.659371
[370]	training's auc: 0.794401	valid_1's auc: 0.659648
[380]	training's auc: 0.795116	valid_1's auc: 0.659979
[390]	training's auc: 0.796024	valid_1's auc: 0.660289
[400]	training's auc: 0.796438	valid_1's auc: 0.66043
[410]	training's auc: 0.796958	valid_1's auc: 0.660518
[420]	training's auc: 0.797505	valid_1's auc: 0.660684
[430]	training's auc: 0.798019	valid_1's auc: 0.660819
[440]	training's auc: 0.798677	valid_1's auc: 0.660924
[450]	training's auc: 0.799286	valid_1's auc: 0.661058
[460]	training's auc: 0.80006	valid_1's auc: 0.661283
[470]	training's auc: 0.800547	valid_1's auc: 0.661398
[480]	training's auc: 0.801061	valid_1's auc: 0.661487
[490]	training's auc: 0.801715	valid_1's auc: 0.661719
[500]	training's auc: 0.802196	valid_1's auc: 0.66183
[510]	training's auc: 0.802876	valid_1's auc: 0.662003
[520]	training's auc: 0.803289	valid_1's auc: 0.662104
[530]	training's auc: 0.803804	valid_1's auc: 0.662273
[540]	training's auc: 0.804582	valid_1's auc: 0.662446
[550]	training's auc: 0.80497	valid_1's auc: 0.662523
[560]	training's auc: 0.805404	valid_1's auc: 0.662627
[570]	training's auc: 0.80594	valid_1's auc: 0.66278
[580]	training's auc: 0.806313	valid_1's auc: 0.662904
[590]	training's auc: 0.806841	valid_1's auc: 0.663052
[600]	training's auc: 0.80722	valid_1's auc: 0.663149
[610]	training's auc: 0.807862	valid_1's auc: 0.663325
[620]	training's auc: 0.808261	valid_1's auc: 0.663388
[630]	training's auc: 0.8087	valid_1's auc: 0.663471
[640]	training's auc: 0.809069	valid_1's auc: 0.663577
[650]	training's auc: 0.809462	valid_1's auc: 0.663654
[660]	training's auc: 0.809844	valid_1's auc: 0.663733
[670]	training's auc: 0.810241	valid_1's auc: 0.663841
[680]	training's auc: 0.810561	valid_1's auc: 0.663929
[690]	training's auc: 0.811039	valid_1's auc: 0.66407
[700]	training's auc: 0.811544	valid_1's auc: 0.664251
[710]	training's auc: 0.811864	valid_1's auc: 0.664286
[720]	training's auc: 0.812252	valid_1's auc: 0.664388
[730]	training's auc: 0.812542	valid_1's auc: 0.664489
[740]	training's auc: 0.812955	valid_1's auc: 0.664574
[750]	training's auc: 0.813345	valid_1's auc: 0.664721
[760]	training's auc: 0.813752	valid_1's auc: 0.664824
[770]	training's auc: 0.814163	valid_1's auc: 0.664923
[780]	training's auc: 0.814468	valid_1's auc: 0.664924
[790]	training's auc: 0.814827	valid_1's auc: 0.664995
[800]	training's auc: 0.815166	valid_1's auc: 0.665056
[810]	training's auc: 0.81545	valid_1's auc: 0.665102
[820]	training's auc: 0.815828	valid_1's auc: 0.665136
[830]	training's auc: 0.816156	valid_1's auc: 0.665216
[840]	training's auc: 0.816434	valid_1's auc: 0.665258
[850]	training's auc: 0.816792	valid_1's auc: 0.665328
[860]	training's auc: 0.817054	valid_1's auc: 0.665377
[870]	training's auc: 0.817338	valid_1's auc: 0.665418
[880]	training's auc: 0.817669	valid_1's auc: 0.665454
[890]	training's auc: 0.81799	valid_1's auc: 0.665463
[900]	training's auc: 0.81822	valid_1's auc: 0.665491
[910]	training's auc: 0.818471	valid_1's auc: 0.665545
[920]	training's auc: 0.818785	valid_1's auc: 0.665605
[930]	training's auc: 0.819159	valid_1's auc: 0.665725
[940]	training's auc: 0.819406	valid_1's auc: 0.665745
[950]	training's auc: 0.819736	valid_1's auc: 0.665768
[960]	training's auc: 0.820014	valid_1's auc: 0.665812
[970]	training's auc: 0.820271	valid_1's auc: 0.665826
[980]	training's auc: 0.820509	valid_1's auc: 0.665851
[990]	training's auc: 0.820773	valid_1's auc: 0.665874
[1000]	training's auc: 0.820991	valid_1's auc: 0.665932
[1010]	training's auc: 0.821231	valid_1's auc: 0.666009
[1020]	training's auc: 0.821456	valid_1's auc: 0.665982
[1030]	training's auc: 0.821729	valid_1's auc: 0.666026
[1040]	training's auc: 0.821974	valid_1's auc: 0.666078
[1050]	training's auc: 0.822181	valid_1's auc: 0.666149
[1060]	training's auc: 0.822378	valid_1's auc: 0.666157
[1070]	training's auc: 0.822589	valid_1's auc: 0.666188
[1080]	training's auc: 0.822881	valid_1's auc: 0.666239
[1090]	training's auc: 0.823045	valid_1's auc: 0.666248
[1100]	training's auc: 0.823242	valid_1's auc: 0.666242
[1110]	training's auc: 0.823434	valid_1's auc: 0.66628
[1120]	training's auc: 0.823636	valid_1's auc: 0.666289
[1130]	training's auc: 0.82391	valid_1's auc: 0.666384
[1140]	training's auc: 0.824113	valid_1's auc: 0.666412
[1150]	training's auc: 0.824324	valid_1's auc: 0.666448
[1160]	training's auc: 0.824538	valid_1's auc: 0.666494
[1170]	training's auc: 0.824775	valid_1's auc: 0.666498
[1180]	training's auc: 0.824949	valid_1's auc: 0.666527
[1190]	training's auc: 0.825106	valid_1's auc: 0.666562
[1200]	training's auc: 0.825279	valid_1's auc: 0.666579
[1210]	training's auc: 0.825451	valid_1's auc: 0.666615
[1220]	training's auc: 0.825672	valid_1's auc: 0.666627
[1230]	training's auc: 0.825948	valid_1's auc: 0.666651
[1240]	training's auc: 0.826147	valid_1's auc: 0.666652
[1250]	training's auc: 0.826429	valid_1's auc: 0.666681
[1260]	training's auc: 0.826604	valid_1's auc: 0.666685
[1270]	training's auc: 0.82678	valid_1's auc: 0.66673
[1280]	training's auc: 0.826964	valid_1's auc: 0.666724
[1290]	training's auc: 0.827125	valid_1's auc: 0.666746
[1300]	training's auc: 0.827465	valid_1's auc: 0.666849
[1310]	training's auc: 0.827634	valid_1's auc: 0.666856
[1320]	training's auc: 0.82783	valid_1's auc: 0.666913
[1330]	training's auc: 0.827994	valid_1's auc: 0.666912
[1340]	training's auc: 0.82818	valid_1's auc: 0.666905
[1350]	training's auc: 0.828335	valid_1's auc: 0.666917
[1360]	training's auc: 0.828504	valid_1's auc: 0.666949
[1370]	training's auc: 0.828656	valid_1's auc: 0.666955
[1380]	training's auc: 0.828787	valid_1's auc: 0.666981
[1390]	training's auc: 0.82894	valid_1's auc: 0.666996
[1400]	training's auc: 0.829066	valid_1's auc: 0.667006
[1410]	training's auc: 0.829215	valid_1's auc: 0.666984
[1420]	training's auc: 0.829356	valid_1's auc: 0.666995
[1430]	training's auc: 0.829574	valid_1's auc: 0.667092
[1440]	training's auc: 0.829763	valid_1's auc: 0.66713
[1450]	training's auc: 0.829914	valid_1's auc: 0.667145
[1460]	training's auc: 0.830038	valid_1's auc: 0.667131
[1470]	training's auc: 0.830241	valid_1's auc: 0.667168
[1480]	training's auc: 0.830384	valid_1's auc: 0.667186
[1490]	training's auc: 0.830516	valid_1's auc: 0.667197
[1500]	training's auc: 0.830654	valid_1's auc: 0.667219
[1510]	training's auc: 0.830766	valid_1's auc: 0.667247
[1520]	training's auc: 0.83087	valid_1's auc: 0.667247
[1530]	training's auc: 0.831018	valid_1's auc: 0.667284
[1540]	training's auc: 0.831156	valid_1's auc: 0.667323
[1550]	training's auc: 0.831262	valid_1's auc: 0.667332
[1560]	training's auc: 0.831413	valid_1's auc: 0.66737
[1570]	training's auc: 0.831499	valid_1's auc: 0.667358
[1580]	training's auc: 0.831636	valid_1's auc: 0.667388
[1590]	training's auc: 0.83175	valid_1's auc: 0.6674
[1600]	training's auc: 0.831874	valid_1's auc: 0.667408
[1610]	training's auc: 0.832039	valid_1's auc: 0.667428
[1620]	training's auc: 0.832146	valid_1's auc: 0.667421
[1630]	training's auc: 0.832255	valid_1's auc: 0.667439
[1640]	training's auc: 0.832344	valid_1's auc: 0.667444
[1650]	training's auc: 0.832478	valid_1's auc: 0.667457
[1660]	training's auc: 0.832601	valid_1's auc: 0.667443
[1670]	training's auc: 0.832734	valid_1's auc: 0.667483
[1680]	training's auc: 0.83286	valid_1's auc: 0.667492
[1690]	training's auc: 0.832962	valid_1's auc: 0.667491
[1700]	training's auc: 0.833103	valid_1's auc: 0.667489
[1710]	training's auc: 0.833208	valid_1's auc: 0.667535
[1720]	training's auc: 0.833309	valid_1's auc: 0.667555
[1730]	training's auc: 0.83341	valid_1's auc: 0.667578
[1740]	training's auc: 0.833511	valid_1's auc: 0.667582
[1750]	training's auc: 0.83362	valid_1's auc: 0.667584
[1760]	training's auc: 0.833714	valid_1's auc: 0.667622
[1770]	training's auc: 0.83388	valid_1's auc: 0.667653
[1780]	training's auc: 0.833978	valid_1's auc: 0.667662
[1790]	training's auc: 0.834094	valid_1's auc: 0.667664
[1800]	training's auc: 0.834209	valid_1's auc: 0.667682
[1810]	training's auc: 0.83434	valid_1's auc: 0.667687
[1820]	training's auc: 0.834459	valid_1's auc: 0.667701
[1830]	training's auc: 0.834577	valid_1's auc: 0.66773
[1840]	training's auc: 0.83469	valid_1's auc: 0.667747
[1850]	training's auc: 0.834792	valid_1's auc: 0.667756
[1860]	training's auc: 0.83489	valid_1's auc: 0.667752
[1870]	training's auc: 0.835008	valid_1's auc: 0.66775
[1880]	training's auc: 0.835106	valid_1's auc: 0.667794
[1890]	training's auc: 0.835227	valid_1's auc: 0.667794
[1900]	training's auc: 0.835331	valid_1's auc: 0.66781
[1910]	training's auc: 0.835442	valid_1's auc: 0.667805
[1920]	training's auc: 0.835557	valid_1's auc: 0.667812
[1930]	training's auc: 0.835762	valid_1's auc: 0.667873
[1940]	training's auc: 0.83588	valid_1's auc: 0.667888
[1950]	training's auc: 0.836004	valid_1's auc: 0.667922
[1960]	training's auc: 0.8361	valid_1's auc: 0.667928
[1970]	training's auc: 0.836195	valid_1's auc: 0.66791
[1980]	training's auc: 0.836297	valid_1's auc: 0.667904
[1990]	training's auc: 0.836376	valid_1's auc: 0.667923
[2000]	training's auc: 0.836463	valid_1's auc: 0.667954
[2010]	training's auc: 0.83656	valid_1's auc: 0.667969
[2020]	training's auc: 0.83667	valid_1's auc: 0.667969
[2030]	training's auc: 0.836766	valid_1's auc: 0.66797
[2040]	training's auc: 0.836882	valid_1's auc: 0.667973
[2050]	training's auc: 0.836988	valid_1's auc: 0.667979
[2060]	training's auc: 0.837083	valid_1's auc: 0.667994
[2070]	training's auc: 0.837168	valid_1's auc: 0.667976
[2080]	training's auc: 0.837261	valid_1's auc: 0.667962
[2090]	training's auc: 0.837371	valid_1's auc: 0.66796
[2100]	training's auc: 0.837469	valid_1's auc: 0.667986
[2110]	training's auc: 0.837558	valid_1's auc: 0.667991
Early stopping, best iteration is:
[2061]	training's auc: 0.837091	valid_1's auc: 0.667994
best score: 0.667994296552
best iteration: 2061
complete on: CC11_source_type

working on: ITC_gender

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
ITC_gender               int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721819	valid_1's auc: 0.630759
[20]	training's auc: 0.734715	valid_1's auc: 0.635556
[30]	training's auc: 0.74378	valid_1's auc: 0.640366
[40]	training's auc: 0.747797	valid_1's auc: 0.642344
[50]	training's auc: 0.751163	valid_1's auc: 0.643932
[60]	training's auc: 0.754038	valid_1's auc: 0.645406
[70]	training's auc: 0.75674	valid_1's auc: 0.64654
[80]	training's auc: 0.759076	valid_1's auc: 0.647446
[90]	training's auc: 0.761452	valid_1's auc: 0.648379
[100]	training's auc: 0.763587	valid_1's auc: 0.64934
[110]	training's auc: 0.765412	valid_1's auc: 0.650105
[120]	training's auc: 0.767183	valid_1's auc: 0.650824
[130]	training's auc: 0.769164	valid_1's auc: 0.651644
[140]	training's auc: 0.770601	valid_1's auc: 0.652276
[150]	training's auc: 0.772508	valid_1's auc: 0.653001
[160]	training's auc: 0.773974	valid_1's auc: 0.653454
[170]	training's auc: 0.775299	valid_1's auc: 0.654052
[180]	training's auc: 0.77642	valid_1's auc: 0.654528
[190]	training's auc: 0.777525	valid_1's auc: 0.65498
[200]	training's auc: 0.778556	valid_1's auc: 0.655478
[210]	training's auc: 0.779583	valid_1's auc: 0.655805
[220]	training's auc: 0.780723	valid_1's auc: 0.656095
[230]	training's auc: 0.78169	valid_1's auc: 0.656525
[240]	training's auc: 0.782883	valid_1's auc: 0.656843
[250]	training's auc: 0.78389	valid_1's auc: 0.65724
[260]	training's auc: 0.784793	valid_1's auc: 0.657447
[270]	training's auc: 0.785875	valid_1's auc: 0.657854
[280]	training's auc: 0.786706	valid_1's auc: 0.658135
[290]	training's auc: 0.787506	valid_1's auc: 0.658454
[300]	training's auc: 0.788261	valid_1's auc: 0.658614
[310]	training's auc: 0.789242	valid_1's auc: 0.658965
[320]	training's auc: 0.789976	valid_1's auc: 0.659193
[330]	training's auc: 0.790732	valid_1's auc: 0.659335
[340]	training's auc: 0.791552	valid_1's auc: 0.659557
[350]	training's auc: 0.792192	valid_1's auc: 0.659809
[360]	training's auc: 0.793297	valid_1's auc: 0.66013
[370]	training's auc: 0.793938	valid_1's auc: 0.660309
[380]	training's auc: 0.794468	valid_1's auc: 0.6604
[390]	training's auc: 0.795023	valid_1's auc: 0.660604
[400]	training's auc: 0.795678	valid_1's auc: 0.660727
[410]	training's auc: 0.796217	valid_1's auc: 0.660804
[420]	training's auc: 0.796749	valid_1's auc: 0.660974
[430]	training's auc: 0.797245	valid_1's auc: 0.661116
[440]	training's auc: 0.797849	valid_1's auc: 0.661288
[450]	training's auc: 0.798407	valid_1's auc: 0.661446
[460]	training's auc: 0.79924	valid_1's auc: 0.661712
[470]	training's auc: 0.799837	valid_1's auc: 0.661861
[480]	training's auc: 0.800474	valid_1's auc: 0.662098
[490]	training's auc: 0.800929	valid_1's auc: 0.662163
[500]	training's auc: 0.801569	valid_1's auc: 0.662328
[510]	training's auc: 0.802176	valid_1's auc: 0.662526
[520]	training's auc: 0.802751	valid_1's auc: 0.66264
[530]	training's auc: 0.803251	valid_1's auc: 0.662821
[540]	training's auc: 0.803759	valid_1's auc: 0.662982
[550]	training's auc: 0.80423	valid_1's auc: 0.663196
[560]	training's auc: 0.804629	valid_1's auc: 0.663264
[570]	training's auc: 0.805213	valid_1's auc: 0.663393
[580]	training's auc: 0.805639	valid_1's auc: 0.663459
[590]	training's auc: 0.806088	valid_1's auc: 0.663579
[600]	training's auc: 0.806421	valid_1's auc: 0.663645
[610]	training's auc: 0.806878	valid_1's auc: 0.663809
[620]	training's auc: 0.807295	valid_1's auc: 0.663918
[630]	training's auc: 0.807661	valid_1's auc: 0.664
[640]	training's auc: 0.808136	valid_1's auc: 0.664115
[650]	training's auc: 0.808499	valid_1's auc: 0.664161
[660]	training's auc: 0.809057	valid_1's auc: 0.664319
[670]	training's auc: 0.809397	valid_1's auc: 0.664371
[680]	training's auc: 0.809834	valid_1's auc: 0.664446
[690]	training's auc: 0.810355	valid_1's auc: 0.664566
[700]	training's auc: 0.810684	valid_1's auc: 0.664595
[710]	training's auc: 0.811068	valid_1's auc: 0.664674
[720]	training's auc: 0.811366	valid_1's auc: 0.664779
[730]	training's auc: 0.811694	valid_1's auc: 0.664874
[740]	training's auc: 0.811997	valid_1's auc: 0.664912
[750]	training's auc: 0.812243	valid_1's auc: 0.664973
[760]	training's auc: 0.812583	valid_1's auc: 0.665066
[770]	training's auc: 0.812971	valid_1's auc: 0.665175
[780]	training's auc: 0.813262	valid_1's auc: 0.665221
[790]	training's auc: 0.813569	valid_1's auc: 0.665257
[800]	training's auc: 0.813912	valid_1's auc: 0.665291
[810]	training's auc: 0.814196	valid_1's auc: 0.665327
[820]	training's auc: 0.814605	valid_1's auc: 0.665354
[830]	training's auc: 0.814925	valid_1's auc: 0.665383
[840]	training's auc: 0.815222	valid_1's auc: 0.665399
[850]	training's auc: 0.815596	valid_1's auc: 0.665496
[860]	training's auc: 0.815868	valid_1's auc: 0.665519
[870]	training's auc: 0.816175	valid_1's auc: 0.665603
[880]	training's auc: 0.816591	valid_1's auc: 0.665695
[890]	training's auc: 0.81691	valid_1's auc: 0.665724
[900]	training's auc: 0.817184	valid_1's auc: 0.665756
[910]	training's auc: 0.817429	valid_1's auc: 0.665831
[920]	training's auc: 0.81769	valid_1's auc: 0.665894
[930]	training's auc: 0.817957	valid_1's auc: 0.665909
[940]	training's auc: 0.818261	valid_1's auc: 0.66595
[950]	training's auc: 0.818532	valid_1's auc: 0.66601
[960]	training's auc: 0.818796	valid_1's auc: 0.666015
[970]	training's auc: 0.819031	valid_1's auc: 0.666026
[980]	training's auc: 0.819358	valid_1's auc: 0.66615
[990]	training's auc: 0.819617	valid_1's auc: 0.666174
[1000]	training's auc: 0.819865	valid_1's auc: 0.666229
[1010]	training's auc: 0.82008	valid_1's auc: 0.666257
[1020]	training's auc: 0.820371	valid_1's auc: 0.666292
[1030]	training's auc: 0.820587	valid_1's auc: 0.666341
[1040]	training's auc: 0.8208	valid_1's auc: 0.666371
[1050]	training's auc: 0.821037	valid_1's auc: 0.666396
[1060]	training's auc: 0.821345	valid_1's auc: 0.666427
[1070]	training's auc: 0.821524	valid_1's auc: 0.666448
[1080]	training's auc: 0.821766	valid_1's auc: 0.666483
[1090]	training's auc: 0.822061	valid_1's auc: 0.666534
[1100]	training's auc: 0.822214	valid_1's auc: 0.666542
[1110]	training's auc: 0.822447	valid_1's auc: 0.666584
[1120]	training's auc: 0.822633	valid_1's auc: 0.666596
[1130]	training's auc: 0.822838	valid_1's auc: 0.666597
[1140]	training's auc: 0.823025	valid_1's auc: 0.666603
[1150]	training's auc: 0.823299	valid_1's auc: 0.666637
[1160]	training's auc: 0.823507	valid_1's auc: 0.666684
[1170]	training's auc: 0.823722	valid_1's auc: 0.666676
[1180]	training's auc: 0.823967	valid_1's auc: 0.666711
[1190]	training's auc: 0.824151	valid_1's auc: 0.666714
[1200]	training's auc: 0.824344	valid_1's auc: 0.666728
[1210]	training's auc: 0.82461	valid_1's auc: 0.666772
[1220]	training's auc: 0.824806	valid_1's auc: 0.666813
[1230]	training's auc: 0.825018	valid_1's auc: 0.666838
[1240]	training's auc: 0.825201	valid_1's auc: 0.666874
[1250]	training's auc: 0.825368	valid_1's auc: 0.666902
[1260]	training's auc: 0.825522	valid_1's auc: 0.66694
[1270]	training's auc: 0.825717	valid_1's auc: 0.666986
[1280]	training's auc: 0.825906	valid_1's auc: 0.66702
[1290]	training's auc: 0.826053	valid_1's auc: 0.667026
[1300]	training's auc: 0.826206	valid_1's auc: 0.667039
[1310]	training's auc: 0.826362	valid_1's auc: 0.667044
[1320]	training's auc: 0.826541	valid_1's auc: 0.667068
[1330]	training's auc: 0.826735	valid_1's auc: 0.667108
[1340]	training's auc: 0.826894	valid_1's auc: 0.667105
[1350]	training's auc: 0.827035	valid_1's auc: 0.667119
[1360]	training's auc: 0.82718	valid_1's auc: 0.667147
[1370]	training's auc: 0.827325	valid_1's auc: 0.667163
[1380]	training's auc: 0.82745	valid_1's auc: 0.667171
[1390]	training's auc: 0.827589	valid_1's auc: 0.667176
[1400]	training's auc: 0.827749	valid_1's auc: 0.667233
[1410]	training's auc: 0.828104	valid_1's auc: 0.667395
[1420]	training's auc: 0.828252	valid_1's auc: 0.667411
[1430]	training's auc: 0.828377	valid_1's auc: 0.667399
[1440]	training's auc: 0.828498	valid_1's auc: 0.667455
[1450]	training's auc: 0.82863	valid_1's auc: 0.667483
[1460]	training's auc: 0.828738	valid_1's auc: 0.66747
[1470]	training's auc: 0.828885	valid_1's auc: 0.667479
[1480]	training's auc: 0.829005	valid_1's auc: 0.667537
[1490]	training's auc: 0.829131	valid_1's auc: 0.667572
[1500]	training's auc: 0.82929	valid_1's auc: 0.667586
[1510]	training's auc: 0.829389	valid_1's auc: 0.66759
[1520]	training's auc: 0.829512	valid_1's auc: 0.66762
[1530]	training's auc: 0.829647	valid_1's auc: 0.66762
[1540]	training's auc: 0.829816	valid_1's auc: 0.667629
[1550]	training's auc: 0.829939	valid_1's auc: 0.667635
[1560]	training's auc: 0.830075	valid_1's auc: 0.667627
[1570]	training's auc: 0.830191	valid_1's auc: 0.66764
[1580]	training's auc: 0.830308	valid_1's auc: 0.667642
[1590]	training's auc: 0.83043	valid_1's auc: 0.667652
[1600]	training's auc: 0.830576	valid_1's auc: 0.667674
[1610]	training's auc: 0.830701	valid_1's auc: 0.667673
[1620]	training's auc: 0.830807	valid_1's auc: 0.667698
[1630]	training's auc: 0.830936	valid_1's auc: 0.667702
[1640]	training's auc: 0.831095	valid_1's auc: 0.667719
[1650]	training's auc: 0.831235	valid_1's auc: 0.667752
[1660]	training's auc: 0.831348	valid_1's auc: 0.667792
[1670]	training's auc: 0.831457	valid_1's auc: 0.66783
[1680]	training's auc: 0.831555	valid_1's auc: 0.667836
[1690]	training's auc: 0.831664	valid_1's auc: 0.667852
[1700]	training's auc: 0.831809	valid_1's auc: 0.667901
[1710]	training's auc: 0.831908	valid_1's auc: 0.667919
[1720]	training's auc: 0.832016	valid_1's auc: 0.667927
[1730]	training's auc: 0.832137	valid_1's auc: 0.667933
[1740]	training's auc: 0.832231	valid_1's auc: 0.667963
[1750]	training's auc: 0.832333	valid_1's auc: 0.667952
[1760]	training's auc: 0.832423	valid_1's auc: 0.667976
[1770]	training's auc: 0.832528	valid_1's auc: 0.667977
[1780]	training's auc: 0.832631	valid_1's auc: 0.668001
[1790]	training's auc: 0.832749	valid_1's auc: 0.668014
[1800]	training's auc: 0.832866	valid_1's auc: 0.668015
[1810]	training's auc: 0.832985	valid_1's auc: 0.668019
[1820]	training's auc: 0.833118	valid_1's auc: 0.668016
[1830]	training's auc: 0.83323	valid_1's auc: 0.668018
[1840]	training's auc: 0.833329	valid_1's auc: 0.668056
[1850]	training's auc: 0.833427	valid_1's auc: 0.668062
[1860]	training's auc: 0.833542	valid_1's auc: 0.668095
[1870]	training's auc: 0.833646	valid_1's auc: 0.668083
[1880]	training's auc: 0.83378	valid_1's auc: 0.668104
[1890]	training's auc: 0.833892	valid_1's auc: 0.668094
[1900]	training's auc: 0.834005	valid_1's auc: 0.668131
[1910]	training's auc: 0.834099	valid_1's auc: 0.668149
[1920]	training's auc: 0.834211	valid_1's auc: 0.66812
[1930]	training's auc: 0.83431	valid_1's auc: 0.668132
[1940]	training's auc: 0.834416	valid_1's auc: 0.66815
[1950]	training's auc: 0.834515	valid_1's auc: 0.668175
[1960]	training's auc: 0.83462	valid_1's auc: 0.668186
[1970]	training's auc: 0.834719	valid_1's auc: 0.668174
[1980]	training's auc: 0.834829	valid_1's auc: 0.668164
[1990]	training's auc: 0.834903	valid_1's auc: 0.66818
[2000]	training's auc: 0.834976	valid_1's auc: 0.6682
[2010]	training's auc: 0.835068	valid_1's auc: 0.668218
[2020]	training's auc: 0.835169	valid_1's auc: 0.6682
[2030]	training's auc: 0.835268	valid_1's auc: 0.668211
[2040]	training's auc: 0.835358	valid_1's auc: 0.668197
[2050]	training's auc: 0.835475	valid_1's auc: 0.668215
[2060]	training's auc: 0.835564	valid_1's auc: 0.668213
[2070]	training's auc: 0.835702	valid_1's auc: 0.668222
[2080]	training's auc: 0.835793	valid_1's auc: 0.668252
[2090]	training's auc: 0.83589	valid_1's auc: 0.66825
[2100]	training's auc: 0.835984	valid_1's auc: 0.668249
[2110]	training's auc: 0.83607	valid_1's auc: 0.668266
[2120]	training's auc: 0.836169	valid_1's auc: 0.66827
[2130]	training's auc: 0.83628	valid_1's auc: 0.668308
[2140]	training's auc: 0.836376	valid_1's auc: 0.668313
[2150]	training's auc: 0.836476	valid_1's auc: 0.668337
[2160]	training's auc: 0.836561	valid_1's auc: 0.668337
[2170]	training's auc: 0.836649	valid_1's auc: 0.668339
[2180]	training's auc: 0.83675	valid_1's auc: 0.668342
[2190]	training's auc: 0.836866	valid_1's auc: 0.66835
[2200]	training's auc: 0.836951	valid_1's auc: 0.668374
[2210]	training's auc: 0.837042	valid_1's auc: 0.668374
[2220]	training's auc: 0.837141	valid_1's auc: 0.668387
[2230]	training's auc: 0.837229	valid_1's auc: 0.668388
[2240]	training's auc: 0.837311	valid_1's auc: 0.668378
[2250]	training's auc: 0.837393	valid_1's auc: 0.66839
[2260]	training's auc: 0.837474	valid_1's auc: 0.668409
[2270]	training's auc: 0.837563	valid_1's auc: 0.668442
[2280]	training's auc: 0.83764	valid_1's auc: 0.668434
[2290]	training's auc: 0.837719	valid_1's auc: 0.668425
[2300]	training's auc: 0.837795	valid_1's auc: 0.668435
[2310]	training's auc: 0.837883	valid_1's auc: 0.668448
[2320]	training's auc: 0.837963	valid_1's auc: 0.668451
[2330]	training's auc: 0.838063	valid_1's auc: 0.668449
[2340]	training's auc: 0.838158	valid_1's auc: 0.668455
[2350]	training's auc: 0.838249	valid_1's auc: 0.668449
[2360]	training's auc: 0.838327	valid_1's auc: 0.668444
[2370]	training's auc: 0.838402	valid_1's auc: 0.668455
[2380]	training's auc: 0.838479	valid_1's auc: 0.66846
[2390]	training's auc: 0.838569	valid_1's auc: 0.668466
[2400]	training's auc: 0.838653	valid_1's auc: 0.668468
[2410]	training's auc: 0.838724	valid_1's auc: 0.668492
[2420]	training's auc: 0.838813	valid_1's auc: 0.668485
[2430]	training's auc: 0.83889	valid_1's auc: 0.668501
[2440]	training's auc: 0.838973	valid_1's auc: 0.668501
[2450]	training's auc: 0.839049	valid_1's auc: 0.668509
[2460]	training's auc: 0.839128	valid_1's auc: 0.668517
[2470]	training's auc: 0.839217	valid_1's auc: 0.668519
[2480]	training's auc: 0.839288	valid_1's auc: 0.668537
[2490]	training's auc: 0.839367	valid_1's auc: 0.668556
[2500]	training's auc: 0.839437	valid_1's auc: 0.668566
[2510]	training's auc: 0.839531	valid_1's auc: 0.668575
[2520]	training's auc: 0.839602	valid_1's auc: 0.668598
[2530]	training's auc: 0.839682	valid_1's auc: 0.668591
[2540]	training's auc: 0.839764	valid_1's auc: 0.668597
[2550]	training's auc: 0.839843	valid_1's auc: 0.668616
[2560]	training's auc: 0.839931	valid_1's auc: 0.668632
[2570]	training's auc: 0.840001	valid_1's auc: 0.668645
[2580]	training's auc: 0.840078	valid_1's auc: 0.668625
[2590]	training's auc: 0.840161	valid_1's auc: 0.668633
[2600]	training's auc: 0.84025	valid_1's auc: 0.668645
[2610]	training's auc: 0.840319	valid_1's auc: 0.668644
[2620]	training's auc: 0.840403	valid_1's auc: 0.668674
[2630]	training's auc: 0.840474	valid_1's auc: 0.668697
[2640]	training's auc: 0.840559	valid_1's auc: 0.668697
[2650]	training's auc: 0.840649	valid_1's auc: 0.668698
[2660]	training's auc: 0.840735	valid_1's auc: 0.668717
[2670]	training's auc: 0.840812	valid_1's auc: 0.668737
[2680]	training's auc: 0.840884	valid_1's auc: 0.668719
[2690]	training's auc: 0.840977	valid_1's auc: 0.66872
[2700]	training's auc: 0.841056	valid_1's auc: 0.668733
[2710]	training's auc: 0.841134	valid_1's auc: 0.668724
Early stopping, best iteration is:
[2669]	training's auc: 0.840807	valid_1's auc: 0.66874
best score: 0.668739635206
best iteration: 2669
complete on: ITC_gender

working on: CC11_gender

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
CC11_gender              int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.722281	valid_1's auc: 0.630249
[20]	training's auc: 0.736468	valid_1's auc: 0.636587
[30]	training's auc: 0.744102	valid_1's auc: 0.640363
[40]	training's auc: 0.748581	valid_1's auc: 0.642758
[50]	training's auc: 0.752113	valid_1's auc: 0.644661
[60]	training's auc: 0.75506	valid_1's auc: 0.645886
[70]	training's auc: 0.757546	valid_1's auc: 0.646895
[80]	training's auc: 0.760621	valid_1's auc: 0.648229
[90]	training's auc: 0.762947	valid_1's auc: 0.649092
[100]	training's auc: 0.765062	valid_1's auc: 0.649882
[110]	training's auc: 0.766801	valid_1's auc: 0.650669
[120]	training's auc: 0.768675	valid_1's auc: 0.651543
[130]	training's auc: 0.770211	valid_1's auc: 0.652225
[140]	training's auc: 0.7719	valid_1's auc: 0.652416
[150]	training's auc: 0.77333	valid_1's auc: 0.652994
[160]	training's auc: 0.77483	valid_1's auc: 0.653699
[170]	training's auc: 0.776059	valid_1's auc: 0.654274
[180]	training's auc: 0.777304	valid_1's auc: 0.654943
[190]	training's auc: 0.778362	valid_1's auc: 0.655271
[200]	training's auc: 0.7794	valid_1's auc: 0.655707
[210]	training's auc: 0.780509	valid_1's auc: 0.656173
[220]	training's auc: 0.781478	valid_1's auc: 0.656519
[230]	training's auc: 0.782423	valid_1's auc: 0.656849
[240]	training's auc: 0.783658	valid_1's auc: 0.657068
[250]	training's auc: 0.784644	valid_1's auc: 0.657406
[260]	training's auc: 0.785477	valid_1's auc: 0.657644
[270]	training's auc: 0.786569	valid_1's auc: 0.657921
[280]	training's auc: 0.787295	valid_1's auc: 0.658226
[290]	training's auc: 0.78802	valid_1's auc: 0.658473
[300]	training's auc: 0.788839	valid_1's auc: 0.658683
[310]	training's auc: 0.789682	valid_1's auc: 0.659
[320]	training's auc: 0.790495	valid_1's auc: 0.659285
[330]	training's auc: 0.79119	valid_1's auc: 0.659479
[340]	training's auc: 0.792253	valid_1's auc: 0.659708
[350]	training's auc: 0.793114	valid_1's auc: 0.660004
[360]	training's auc: 0.793701	valid_1's auc: 0.66012
[370]	training's auc: 0.794404	valid_1's auc: 0.660329
[380]	training's auc: 0.795001	valid_1's auc: 0.660621
[390]	training's auc: 0.795551	valid_1's auc: 0.660739
[400]	training's auc: 0.796115	valid_1's auc: 0.661004
[410]	training's auc: 0.7967	valid_1's auc: 0.661043
[420]	training's auc: 0.797294	valid_1's auc: 0.661259
[430]	training's auc: 0.797782	valid_1's auc: 0.66136
[440]	training's auc: 0.798298	valid_1's auc: 0.661495
[450]	training's auc: 0.799013	valid_1's auc: 0.661636
[460]	training's auc: 0.799637	valid_1's auc: 0.661823
[470]	training's auc: 0.800111	valid_1's auc: 0.661966
[480]	training's auc: 0.800844	valid_1's auc: 0.662146
[490]	training's auc: 0.801316	valid_1's auc: 0.662215
[500]	training's auc: 0.801969	valid_1's auc: 0.66244
[510]	training's auc: 0.802355	valid_1's auc: 0.662528
[520]	training's auc: 0.803137	valid_1's auc: 0.662796
[530]	training's auc: 0.803958	valid_1's auc: 0.663116
[540]	training's auc: 0.804344	valid_1's auc: 0.663246
[550]	training's auc: 0.80493	valid_1's auc: 0.663453
[560]	training's auc: 0.805347	valid_1's auc: 0.66355
[570]	training's auc: 0.805764	valid_1's auc: 0.663654
[580]	training's auc: 0.806192	valid_1's auc: 0.663815
[590]	training's auc: 0.806652	valid_1's auc: 0.663902
[600]	training's auc: 0.806996	valid_1's auc: 0.663995
[610]	training's auc: 0.80741	valid_1's auc: 0.664145
[620]	training's auc: 0.807805	valid_1's auc: 0.664272
[630]	training's auc: 0.808208	valid_1's auc: 0.66436
[640]	training's auc: 0.808703	valid_1's auc: 0.66447
[650]	training's auc: 0.809023	valid_1's auc: 0.664522
[660]	training's auc: 0.809426	valid_1's auc: 0.664637
[670]	training's auc: 0.809795	valid_1's auc: 0.664678
[680]	training's auc: 0.810192	valid_1's auc: 0.664739
[690]	training's auc: 0.810639	valid_1's auc: 0.664897
[700]	training's auc: 0.811046	valid_1's auc: 0.665002
[710]	training's auc: 0.811351	valid_1's auc: 0.665044
[720]	training's auc: 0.811761	valid_1's auc: 0.665165
[730]	training's auc: 0.812089	valid_1's auc: 0.665237
[740]	training's auc: 0.812361	valid_1's auc: 0.665251
[750]	training's auc: 0.812708	valid_1's auc: 0.665322
[760]	training's auc: 0.813121	valid_1's auc: 0.665396
[770]	training's auc: 0.813395	valid_1's auc: 0.665424
[780]	training's auc: 0.813821	valid_1's auc: 0.665491
[790]	training's auc: 0.814185	valid_1's auc: 0.665607
[800]	training's auc: 0.814502	valid_1's auc: 0.665649
[810]	training's auc: 0.814813	valid_1's auc: 0.665666
[820]	training's auc: 0.815095	valid_1's auc: 0.66571
[830]	training's auc: 0.815361	valid_1's auc: 0.665714
[840]	training's auc: 0.815705	valid_1's auc: 0.665781
[850]	training's auc: 0.815993	valid_1's auc: 0.665836
[860]	training's auc: 0.816297	valid_1's auc: 0.66592
[870]	training's auc: 0.81663	valid_1's auc: 0.666017
[880]	training's auc: 0.817046	valid_1's auc: 0.666109
[890]	training's auc: 0.817411	valid_1's auc: 0.666219
[900]	training's auc: 0.817763	valid_1's auc: 0.666263
[910]	training's auc: 0.818061	valid_1's auc: 0.666311
[920]	training's auc: 0.818342	valid_1's auc: 0.666366
[930]	training's auc: 0.818602	valid_1's auc: 0.666422
[940]	training's auc: 0.818855	valid_1's auc: 0.666469
[950]	training's auc: 0.819163	valid_1's auc: 0.66656
[960]	training's auc: 0.819381	valid_1's auc: 0.666609
[970]	training's auc: 0.819647	valid_1's auc: 0.666627
[980]	training's auc: 0.81986	valid_1's auc: 0.66666
[990]	training's auc: 0.820109	valid_1's auc: 0.666685
[1000]	training's auc: 0.820306	valid_1's auc: 0.666719
[1010]	training's auc: 0.820544	valid_1's auc: 0.66679
[1020]	training's auc: 0.820796	valid_1's auc: 0.666835
[1030]	training's auc: 0.821061	valid_1's auc: 0.666827
[1040]	training's auc: 0.821299	valid_1's auc: 0.66687
[1050]	training's auc: 0.821635	valid_1's auc: 0.666934
[1060]	training's auc: 0.821849	valid_1's auc: 0.666926
[1070]	training's auc: 0.82209	valid_1's auc: 0.666956
[1080]	training's auc: 0.822319	valid_1's auc: 0.667018
[1090]	training's auc: 0.822515	valid_1's auc: 0.667098
[1100]	training's auc: 0.822704	valid_1's auc: 0.6671
[1110]	training's auc: 0.822918	valid_1's auc: 0.667176
[1120]	training's auc: 0.823105	valid_1's auc: 0.667165
[1130]	training's auc: 0.82333	valid_1's auc: 0.667178
[1140]	training's auc: 0.823539	valid_1's auc: 0.667197
[1150]	training's auc: 0.823871	valid_1's auc: 0.667276
[1160]	training's auc: 0.824089	valid_1's auc: 0.667298
[1170]	training's auc: 0.824296	valid_1's auc: 0.667296
[1180]	training's auc: 0.824496	valid_1's auc: 0.667347
[1190]	training's auc: 0.824683	valid_1's auc: 0.667363
[1200]	training's auc: 0.824925	valid_1's auc: 0.667432
[1210]	training's auc: 0.825228	valid_1's auc: 0.667555
[1220]	training's auc: 0.825427	valid_1's auc: 0.667569
[1230]	training's auc: 0.82561	valid_1's auc: 0.667588
[1240]	training's auc: 0.825859	valid_1's auc: 0.667661
[1250]	training's auc: 0.826033	valid_1's auc: 0.667643
[1260]	training's auc: 0.826202	valid_1's auc: 0.667566
[1270]	training's auc: 0.826365	valid_1's auc: 0.667581
[1280]	training's auc: 0.82655	valid_1's auc: 0.66759
[1290]	training's auc: 0.82671	valid_1's auc: 0.667575
Early stopping, best iteration is:
[1240]	training's auc: 0.825859	valid_1's auc: 0.667661
best score: 0.667660778761
best iteration: 1240
complete on: CC11_gender

working on: ITC_artist_name

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
ITC_artist_name          int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.72952	valid_1's auc: 0.635201
[20]	training's auc: 0.744773	valid_1's auc: 0.640649
[30]	training's auc: 0.752093	valid_1's auc: 0.644192
[40]	training's auc: 0.755767	valid_1's auc: 0.646085
[50]	training's auc: 0.759111	valid_1's auc: 0.647775
[60]	training's auc: 0.761976	valid_1's auc: 0.649195
[70]	training's auc: 0.764761	valid_1's auc: 0.650534
[80]	training's auc: 0.767056	valid_1's auc: 0.651453
[90]	training's auc: 0.769235	valid_1's auc: 0.652269
[100]	training's auc: 0.771438	valid_1's auc: 0.653178
[110]	training's auc: 0.773126	valid_1's auc: 0.653875
[120]	training's auc: 0.774875	valid_1's auc: 0.65448
[130]	training's auc: 0.776656	valid_1's auc: 0.655115
[140]	training's auc: 0.77818	valid_1's auc: 0.655418
[150]	training's auc: 0.779508	valid_1's auc: 0.655841
[160]	training's auc: 0.780723	valid_1's auc: 0.656305
[170]	training's auc: 0.781991	valid_1's auc: 0.656895
[180]	training's auc: 0.783291	valid_1's auc: 0.657356
[190]	training's auc: 0.784427	valid_1's auc: 0.657782
[200]	training's auc: 0.785444	valid_1's auc: 0.658188
[210]	training's auc: 0.786582	valid_1's auc: 0.658488
[220]	training's auc: 0.787486	valid_1's auc: 0.658831
[230]	training's auc: 0.788396	valid_1's auc: 0.659208
[240]	training's auc: 0.789299	valid_1's auc: 0.659585
[250]	training's auc: 0.790152	valid_1's auc: 0.659807
[260]	training's auc: 0.791031	valid_1's auc: 0.660082
[270]	training's auc: 0.792146	valid_1's auc: 0.660386
[280]	training's auc: 0.792918	valid_1's auc: 0.660671
[290]	training's auc: 0.793705	valid_1's auc: 0.660905
[300]	training's auc: 0.794498	valid_1's auc: 0.6611
[310]	training's auc: 0.795313	valid_1's auc: 0.661321
[320]	training's auc: 0.79598	valid_1's auc: 0.661537
[330]	training's auc: 0.7966	valid_1's auc: 0.661669
[340]	training's auc: 0.797301	valid_1's auc: 0.661922
[350]	training's auc: 0.797924	valid_1's auc: 0.662127
[360]	training's auc: 0.798595	valid_1's auc: 0.662292
[370]	training's auc: 0.799224	valid_1's auc: 0.66246
[380]	training's auc: 0.799719	valid_1's auc: 0.662572
[390]	training's auc: 0.800553	valid_1's auc: 0.662816
[400]	training's auc: 0.80102	valid_1's auc: 0.663032
[410]	training's auc: 0.801581	valid_1's auc: 0.663144
[420]	training's auc: 0.802239	valid_1's auc: 0.663321
[430]	training's auc: 0.802908	valid_1's auc: 0.663519
[440]	training's auc: 0.803413	valid_1's auc: 0.663563
[450]	training's auc: 0.803961	valid_1's auc: 0.66365
[460]	training's auc: 0.804532	valid_1's auc: 0.663797
[470]	training's auc: 0.804938	valid_1's auc: 0.663952
[480]	training's auc: 0.805539	valid_1's auc: 0.664111
[490]	training's auc: 0.805987	valid_1's auc: 0.664212
[500]	training's auc: 0.806444	valid_1's auc: 0.664364
[510]	training's auc: 0.807059	valid_1's auc: 0.664517
[520]	training's auc: 0.807459	valid_1's auc: 0.664605
[530]	training's auc: 0.808095	valid_1's auc: 0.664767
[540]	training's auc: 0.808558	valid_1's auc: 0.6649
[550]	training's auc: 0.808947	valid_1's auc: 0.665024
[560]	training's auc: 0.809343	valid_1's auc: 0.665189
[570]	training's auc: 0.810031	valid_1's auc: 0.665415
[580]	training's auc: 0.810435	valid_1's auc: 0.665582
[590]	training's auc: 0.810835	valid_1's auc: 0.665632
[600]	training's auc: 0.811299	valid_1's auc: 0.665769
[610]	training's auc: 0.811678	valid_1's auc: 0.665838
[620]	training's auc: 0.812016	valid_1's auc: 0.665876
[630]	training's auc: 0.812527	valid_1's auc: 0.666022
[640]	training's auc: 0.812839	valid_1's auc: 0.666114
[650]	training's auc: 0.813156	valid_1's auc: 0.66614
[660]	training's auc: 0.813486	valid_1's auc: 0.66619
[670]	training's auc: 0.813863	valid_1's auc: 0.666287
[680]	training's auc: 0.814214	valid_1's auc: 0.666389
[690]	training's auc: 0.814585	valid_1's auc: 0.666489
[700]	training's auc: 0.815006	valid_1's auc: 0.666514
[710]	training's auc: 0.815321	valid_1's auc: 0.666568
[720]	training's auc: 0.815725	valid_1's auc: 0.666682
[730]	training's auc: 0.815999	valid_1's auc: 0.666732
[740]	training's auc: 0.816281	valid_1's auc: 0.666749
[750]	training's auc: 0.816613	valid_1's auc: 0.666789
[760]	training's auc: 0.816941	valid_1's auc: 0.666855
[770]	training's auc: 0.817395	valid_1's auc: 0.666954
[780]	training's auc: 0.817696	valid_1's auc: 0.666999
[790]	training's auc: 0.81797	valid_1's auc: 0.667044
[800]	training's auc: 0.818258	valid_1's auc: 0.667056
[810]	training's auc: 0.818651	valid_1's auc: 0.667135
[820]	training's auc: 0.818944	valid_1's auc: 0.667151
[830]	training's auc: 0.819189	valid_1's auc: 0.667152
[840]	training's auc: 0.819569	valid_1's auc: 0.667216
[850]	training's auc: 0.819837	valid_1's auc: 0.667223
[860]	training's auc: 0.820155	valid_1's auc: 0.667261
[870]	training's auc: 0.820426	valid_1's auc: 0.667316
[880]	training's auc: 0.820882	valid_1's auc: 0.66748
[890]	training's auc: 0.82123	valid_1's auc: 0.667517
[900]	training's auc: 0.821491	valid_1's auc: 0.66751
[910]	training's auc: 0.821753	valid_1's auc: 0.667516
[920]	training's auc: 0.821964	valid_1's auc: 0.667566
[930]	training's auc: 0.822266	valid_1's auc: 0.667622
[940]	training's auc: 0.822475	valid_1's auc: 0.667632
[950]	training's auc: 0.822696	valid_1's auc: 0.667675
[960]	training's auc: 0.822948	valid_1's auc: 0.667684
[970]	training's auc: 0.823362	valid_1's auc: 0.667769
[980]	training's auc: 0.823665	valid_1's auc: 0.667785
[990]	training's auc: 0.82393	valid_1's auc: 0.667838
[1000]	training's auc: 0.824149	valid_1's auc: 0.667764
[1010]	training's auc: 0.824445	valid_1's auc: 0.66782
[1020]	training's auc: 0.824693	valid_1's auc: 0.667804
[1030]	training's auc: 0.824946	valid_1's auc: 0.667856
[1040]	training's auc: 0.825208	valid_1's auc: 0.66788
[1050]	training's auc: 0.825373	valid_1's auc: 0.667959
[1060]	training's auc: 0.825586	valid_1's auc: 0.667984
[1070]	training's auc: 0.825801	valid_1's auc: 0.668048
[1080]	training's auc: 0.826086	valid_1's auc: 0.668104
[1090]	training's auc: 0.826307	valid_1's auc: 0.668091
[1100]	training's auc: 0.826599	valid_1's auc: 0.668176
[1110]	training's auc: 0.826841	valid_1's auc: 0.668176
[1120]	training's auc: 0.827046	valid_1's auc: 0.668221
[1130]	training's auc: 0.827248	valid_1's auc: 0.66824
[1140]	training's auc: 0.827445	valid_1's auc: 0.668246
[1150]	training's auc: 0.827792	valid_1's auc: 0.668314
[1160]	training's auc: 0.82801	valid_1's auc: 0.668331
[1170]	training's auc: 0.828223	valid_1's auc: 0.668353
[1180]	training's auc: 0.828421	valid_1's auc: 0.66837
[1190]	training's auc: 0.828599	valid_1's auc: 0.668395
[1200]	training's auc: 0.828815	valid_1's auc: 0.6684
[1210]	training's auc: 0.828991	valid_1's auc: 0.668412
[1220]	training's auc: 0.829179	valid_1's auc: 0.668438
[1230]	training's auc: 0.82938	valid_1's auc: 0.66845
[1240]	training's auc: 0.829556	valid_1's auc: 0.668457
[1250]	training's auc: 0.829762	valid_1's auc: 0.668452
[1260]	training's auc: 0.829912	valid_1's auc: 0.668433
[1270]	training's auc: 0.830153	valid_1's auc: 0.668443
[1280]	training's auc: 0.830327	valid_1's auc: 0.668441
[1290]	training's auc: 0.830522	valid_1's auc: 0.6685
[1300]	training's auc: 0.8307	valid_1's auc: 0.66851
[1310]	training's auc: 0.830869	valid_1's auc: 0.668527
[1320]	training's auc: 0.831061	valid_1's auc: 0.668529
[1330]	training's auc: 0.831238	valid_1's auc: 0.668539
[1340]	training's auc: 0.831383	valid_1's auc: 0.668562
[1350]	training's auc: 0.831639	valid_1's auc: 0.66864
[1360]	training's auc: 0.831797	valid_1's auc: 0.668637
[1370]	training's auc: 0.831981	valid_1's auc: 0.668679
[1380]	training's auc: 0.832156	valid_1's auc: 0.668697
[1390]	training's auc: 0.832319	valid_1's auc: 0.668715
[1400]	training's auc: 0.832547	valid_1's auc: 0.668818
[1410]	training's auc: 0.832668	valid_1's auc: 0.668815
[1420]	training's auc: 0.832818	valid_1's auc: 0.668822
[1430]	training's auc: 0.832966	valid_1's auc: 0.668885
[1440]	training's auc: 0.833137	valid_1's auc: 0.668901
[1450]	training's auc: 0.833303	valid_1's auc: 0.668928
[1460]	training's auc: 0.833437	valid_1's auc: 0.668927
[1470]	training's auc: 0.833583	valid_1's auc: 0.668939
[1480]	training's auc: 0.83373	valid_1's auc: 0.66896
[1490]	training's auc: 0.833905	valid_1's auc: 0.668991
[1500]	training's auc: 0.834047	valid_1's auc: 0.66902
[1510]	training's auc: 0.834156	valid_1's auc: 0.669
[1520]	training's auc: 0.834257	valid_1's auc: 0.669008
[1530]	training's auc: 0.834448	valid_1's auc: 0.669028
[1540]	training's auc: 0.834578	valid_1's auc: 0.669066
[1550]	training's auc: 0.834707	valid_1's auc: 0.669063
[1560]	training's auc: 0.834914	valid_1's auc: 0.669105
[1570]	training's auc: 0.835035	valid_1's auc: 0.669116
[1580]	training's auc: 0.835159	valid_1's auc: 0.669127
[1590]	training's auc: 0.835276	valid_1's auc: 0.669116
[1600]	training's auc: 0.835424	valid_1's auc: 0.669118
[1610]	training's auc: 0.8357	valid_1's auc: 0.669201
[1620]	training's auc: 0.835837	valid_1's auc: 0.669226
[1630]	training's auc: 0.835967	valid_1's auc: 0.669231
[1640]	training's auc: 0.836115	valid_1's auc: 0.669252
[1650]	training's auc: 0.836253	valid_1's auc: 0.669257
[1660]	training's auc: 0.836371	valid_1's auc: 0.669249
[1670]	training's auc: 0.8365	valid_1's auc: 0.669253
[1680]	training's auc: 0.83662	valid_1's auc: 0.669287
[1690]	training's auc: 0.836745	valid_1's auc: 0.669317
[1700]	training's auc: 0.836886	valid_1's auc: 0.669291
[1710]	training's auc: 0.837014	valid_1's auc: 0.669285
[1720]	training's auc: 0.837117	valid_1's auc: 0.669296
[1730]	training's auc: 0.837239	valid_1's auc: 0.66932
[1740]	training's auc: 0.837343	valid_1's auc: 0.669325
[1750]	training's auc: 0.837462	valid_1's auc: 0.669323
[1760]	training's auc: 0.837579	valid_1's auc: 0.669318
[1770]	training's auc: 0.837701	valid_1's auc: 0.669334
[1780]	training's auc: 0.837816	valid_1's auc: 0.669315
[1790]	training's auc: 0.837939	valid_1's auc: 0.669314
[1800]	training's auc: 0.838056	valid_1's auc: 0.669342
[1810]	training's auc: 0.838206	valid_1's auc: 0.669335
[1820]	training's auc: 0.838334	valid_1's auc: 0.66936
[1830]	training's auc: 0.838511	valid_1's auc: 0.669417
[1840]	training's auc: 0.838635	valid_1's auc: 0.66939
[1850]	training's auc: 0.838738	valid_1's auc: 0.669385
[1860]	training's auc: 0.838836	valid_1's auc: 0.669396
[1870]	training's auc: 0.838944	valid_1's auc: 0.669401
Early stopping, best iteration is:
[1826]	training's auc: 0.838463	valid_1's auc: 0.669419
best score: 0.669418682831
best iteration: 1826
complete on: ITC_artist_name

working on: CC11_artist_name

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
CC11_artist_name         int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.730448	valid_1's auc: 0.629478
[20]	training's auc: 0.743677	valid_1's auc: 0.633784
[30]	training's auc: 0.751924	valid_1's auc: 0.637597
[40]	training's auc: 0.75645	valid_1's auc: 0.639809
[50]	training's auc: 0.7598	valid_1's auc: 0.641396
[60]	training's auc: 0.762673	valid_1's auc: 0.64271
[70]	training's auc: 0.765116	valid_1's auc: 0.64361
[80]	training's auc: 0.767384	valid_1's auc: 0.6445
[90]	training's auc: 0.769424	valid_1's auc: 0.645313
[100]	training's auc: 0.771337	valid_1's auc: 0.646144
[110]	training's auc: 0.773133	valid_1's auc: 0.646875
[120]	training's auc: 0.77468	valid_1's auc: 0.647452
[130]	training's auc: 0.776315	valid_1's auc: 0.647974
[140]	training's auc: 0.777881	valid_1's auc: 0.648374
[150]	training's auc: 0.77947	valid_1's auc: 0.649107
[160]	training's auc: 0.78106	valid_1's auc: 0.649718
[170]	training's auc: 0.782179	valid_1's auc: 0.650244
[180]	training's auc: 0.783373	valid_1's auc: 0.650706
[190]	training's auc: 0.784531	valid_1's auc: 0.651074
[200]	training's auc: 0.785698	valid_1's auc: 0.651597
[210]	training's auc: 0.786658	valid_1's auc: 0.651931
[220]	training's auc: 0.787585	valid_1's auc: 0.652253
[230]	training's auc: 0.788488	valid_1's auc: 0.652449
[240]	training's auc: 0.789444	valid_1's auc: 0.652841
[250]	training's auc: 0.790283	valid_1's auc: 0.65308
[260]	training's auc: 0.791189	valid_1's auc: 0.653255
[270]	training's auc: 0.791997	valid_1's auc: 0.653507
[280]	training's auc: 0.793123	valid_1's auc: 0.653951
[290]	training's auc: 0.793899	valid_1's auc: 0.65415
[300]	training's auc: 0.794619	valid_1's auc: 0.654314
[310]	training's auc: 0.795333	valid_1's auc: 0.65455
[320]	training's auc: 0.796052	valid_1's auc: 0.654745
[330]	training's auc: 0.796876	valid_1's auc: 0.655134
[340]	training's auc: 0.79752	valid_1's auc: 0.655314
[350]	training's auc: 0.798203	valid_1's auc: 0.655598
[360]	training's auc: 0.798942	valid_1's auc: 0.65573
[370]	training's auc: 0.799511	valid_1's auc: 0.655849
[380]	training's auc: 0.800158	valid_1's auc: 0.656018
[390]	training's auc: 0.8007	valid_1's auc: 0.656148
[400]	training's auc: 0.801212	valid_1's auc: 0.656323
[410]	training's auc: 0.801801	valid_1's auc: 0.656452
[420]	training's auc: 0.802429	valid_1's auc: 0.656646
[430]	training's auc: 0.802912	valid_1's auc: 0.656766
[440]	training's auc: 0.803427	valid_1's auc: 0.656925
[450]	training's auc: 0.803934	valid_1's auc: 0.656991
[460]	training's auc: 0.804545	valid_1's auc: 0.657133
[470]	training's auc: 0.804957	valid_1's auc: 0.657225
[480]	training's auc: 0.805464	valid_1's auc: 0.657396
[490]	training's auc: 0.806112	valid_1's auc: 0.657592
[500]	training's auc: 0.806801	valid_1's auc: 0.657768
[510]	training's auc: 0.807213	valid_1's auc: 0.657811
[520]	training's auc: 0.807992	valid_1's auc: 0.658052
[530]	training's auc: 0.808384	valid_1's auc: 0.658094
[540]	training's auc: 0.808826	valid_1's auc: 0.658236
[550]	training's auc: 0.80918	valid_1's auc: 0.658326
[560]	training's auc: 0.809717	valid_1's auc: 0.658443
[570]	training's auc: 0.810143	valid_1's auc: 0.658546
[580]	training's auc: 0.81054	valid_1's auc: 0.658685
[590]	training's auc: 0.811047	valid_1's auc: 0.658796
[600]	training's auc: 0.811463	valid_1's auc: 0.658928
[610]	training's auc: 0.811792	valid_1's auc: 0.659015
[620]	training's auc: 0.812209	valid_1's auc: 0.659073
[630]	training's auc: 0.812726	valid_1's auc: 0.659234
[640]	training's auc: 0.813225	valid_1's auc: 0.659379
[650]	training's auc: 0.813545	valid_1's auc: 0.659466
[660]	training's auc: 0.81395	valid_1's auc: 0.659527
[670]	training's auc: 0.814385	valid_1's auc: 0.659622
[680]	training's auc: 0.814801	valid_1's auc: 0.659715
[690]	training's auc: 0.815252	valid_1's auc: 0.659808
[700]	training's auc: 0.815522	valid_1's auc: 0.659883
[710]	training's auc: 0.815797	valid_1's auc: 0.659912
[720]	training's auc: 0.816124	valid_1's auc: 0.659969
[730]	training's auc: 0.816536	valid_1's auc: 0.660165
[740]	training's auc: 0.816866	valid_1's auc: 0.660234
[750]	training's auc: 0.817228	valid_1's auc: 0.660232
[760]	training's auc: 0.817583	valid_1's auc: 0.660294
[770]	training's auc: 0.817956	valid_1's auc: 0.660428
[780]	training's auc: 0.818462	valid_1's auc: 0.660526
[790]	training's auc: 0.818729	valid_1's auc: 0.660567
[800]	training's auc: 0.819004	valid_1's auc: 0.6606
[810]	training's auc: 0.819331	valid_1's auc: 0.660644
[820]	training's auc: 0.819714	valid_1's auc: 0.660698
[830]	training's auc: 0.819996	valid_1's auc: 0.660737
[840]	training's auc: 0.820498	valid_1's auc: 0.660844
[850]	training's auc: 0.820839	valid_1's auc: 0.660887
[860]	training's auc: 0.821145	valid_1's auc: 0.660903
[870]	training's auc: 0.821452	valid_1's auc: 0.660906
[880]	training's auc: 0.821768	valid_1's auc: 0.660988
[890]	training's auc: 0.822119	valid_1's auc: 0.661028
[900]	training's auc: 0.82235	valid_1's auc: 0.661014
[910]	training's auc: 0.822676	valid_1's auc: 0.661079
[920]	training's auc: 0.82292	valid_1's auc: 0.661148
[930]	training's auc: 0.823224	valid_1's auc: 0.661137
[940]	training's auc: 0.823452	valid_1's auc: 0.661162
[950]	training's auc: 0.823732	valid_1's auc: 0.661162
[960]	training's auc: 0.823953	valid_1's auc: 0.661216
[970]	training's auc: 0.824307	valid_1's auc: 0.661263
[980]	training's auc: 0.824603	valid_1's auc: 0.661318
[990]	training's auc: 0.824923	valid_1's auc: 0.66139
[1000]	training's auc: 0.825262	valid_1's auc: 0.661489
[1010]	training's auc: 0.825491	valid_1's auc: 0.661537
[1020]	training's auc: 0.825696	valid_1's auc: 0.66157
[1030]	training's auc: 0.825946	valid_1's auc: 0.661617
[1040]	training's auc: 0.826296	valid_1's auc: 0.661647
[1050]	training's auc: 0.826537	valid_1's auc: 0.661677
[1060]	training's auc: 0.826728	valid_1's auc: 0.661688
[1070]	training's auc: 0.826927	valid_1's auc: 0.661722
[1080]	training's auc: 0.827194	valid_1's auc: 0.661799
[1090]	training's auc: 0.827444	valid_1's auc: 0.661859
[1100]	training's auc: 0.827645	valid_1's auc: 0.661838
[1110]	training's auc: 0.82787	valid_1's auc: 0.661839
[1120]	training's auc: 0.828061	valid_1's auc: 0.661876
[1130]	training's auc: 0.828242	valid_1's auc: 0.661909
[1140]	training's auc: 0.828428	valid_1's auc: 0.661916
[1150]	training's auc: 0.828643	valid_1's auc: 0.661949
[1160]	training's auc: 0.828827	valid_1's auc: 0.661954
[1170]	training's auc: 0.829119	valid_1's auc: 0.662016
[1180]	training's auc: 0.829345	valid_1's auc: 0.662078
[1190]	training's auc: 0.829569	valid_1's auc: 0.662118
[1200]	training's auc: 0.829777	valid_1's auc: 0.662147
[1210]	training's auc: 0.829978	valid_1's auc: 0.662163
[1220]	training's auc: 0.830213	valid_1's auc: 0.662188
[1230]	training's auc: 0.830394	valid_1's auc: 0.662208
[1240]	training's auc: 0.830575	valid_1's auc: 0.662223
[1250]	training's auc: 0.830773	valid_1's auc: 0.662266
[1260]	training's auc: 0.830936	valid_1's auc: 0.662263
[1270]	training's auc: 0.831129	valid_1's auc: 0.662283
[1280]	training's auc: 0.831372	valid_1's auc: 0.662339
[1290]	training's auc: 0.831633	valid_1's auc: 0.662352
[1300]	training's auc: 0.83178	valid_1's auc: 0.662363
[1310]	training's auc: 0.83198	valid_1's auc: 0.662381
[1320]	training's auc: 0.832211	valid_1's auc: 0.662403
[1330]	training's auc: 0.832516	valid_1's auc: 0.662456
[1340]	training's auc: 0.832688	valid_1's auc: 0.662458
[1350]	training's auc: 0.832875	valid_1's auc: 0.66242
[1360]	training's auc: 0.833009	valid_1's auc: 0.662432
[1370]	training's auc: 0.833324	valid_1's auc: 0.662521
[1380]	training's auc: 0.833485	valid_1's auc: 0.662528
[1390]	training's auc: 0.833666	valid_1's auc: 0.662578
[1400]	training's auc: 0.833923	valid_1's auc: 0.662642
[1410]	training's auc: 0.834147	valid_1's auc: 0.662675
[1420]	training's auc: 0.834305	valid_1's auc: 0.662681
[1430]	training's auc: 0.834416	valid_1's auc: 0.662685
[1440]	training's auc: 0.834571	valid_1's auc: 0.662683
[1450]	training's auc: 0.834741	valid_1's auc: 0.662699
[1460]	training's auc: 0.834889	valid_1's auc: 0.662685
[1470]	training's auc: 0.835042	valid_1's auc: 0.662733
[1480]	training's auc: 0.835181	valid_1's auc: 0.66268
[1490]	training's auc: 0.835316	valid_1's auc: 0.662655
[1500]	training's auc: 0.835477	valid_1's auc: 0.662654
[1510]	training's auc: 0.835595	valid_1's auc: 0.662685
[1520]	training's auc: 0.835706	valid_1's auc: 0.662693
Early stopping, best iteration is:
[1475]	training's auc: 0.835114	valid_1's auc: 0.662733
best score: 0.662733095592
best iteration: 1475
complete on: CC11_artist_name

working on: ITC_composer

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
ITC_composer             int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.728783	valid_1's auc: 0.637782
[20]	training's auc: 0.742601	valid_1's auc: 0.643246
[30]	training's auc: 0.750112	valid_1's auc: 0.64663
[40]	training's auc: 0.754382	valid_1's auc: 0.648737
[50]	training's auc: 0.757611	valid_1's auc: 0.65009
[60]	training's auc: 0.760568	valid_1's auc: 0.651479
[70]	training's auc: 0.763369	valid_1's auc: 0.652587
[80]	training's auc: 0.765693	valid_1's auc: 0.653461
[90]	training's auc: 0.767818	valid_1's auc: 0.654351
[100]	training's auc: 0.769812	valid_1's auc: 0.655101
[110]	training's auc: 0.771938	valid_1's auc: 0.655892
[120]	training's auc: 0.773835	valid_1's auc: 0.65655
[130]	training's auc: 0.775298	valid_1's auc: 0.65713
[140]	training's auc: 0.776715	valid_1's auc: 0.657775
[150]	training's auc: 0.778134	valid_1's auc: 0.65822
[160]	training's auc: 0.779394	valid_1's auc: 0.658676
[170]	training's auc: 0.780777	valid_1's auc: 0.659221
[180]	training's auc: 0.782078	valid_1's auc: 0.659689
[190]	training's auc: 0.783259	valid_1's auc: 0.660057
[200]	training's auc: 0.784348	valid_1's auc: 0.660444
[210]	training's auc: 0.785369	valid_1's auc: 0.660782
[220]	training's auc: 0.786315	valid_1's auc: 0.661173
[230]	training's auc: 0.78726	valid_1's auc: 0.661513
[240]	training's auc: 0.788156	valid_1's auc: 0.661746
[250]	training's auc: 0.789063	valid_1's auc: 0.661967
[260]	training's auc: 0.789902	valid_1's auc: 0.662249
[270]	training's auc: 0.79072	valid_1's auc: 0.662532
[280]	training's auc: 0.7915	valid_1's auc: 0.662719
[290]	training's auc: 0.792245	valid_1's auc: 0.662977
[300]	training's auc: 0.793277	valid_1's auc: 0.663338
[310]	training's auc: 0.794342	valid_1's auc: 0.663594
[320]	training's auc: 0.795004	valid_1's auc: 0.663792
[330]	training's auc: 0.795567	valid_1's auc: 0.663942
[340]	training's auc: 0.796193	valid_1's auc: 0.664106
[350]	training's auc: 0.796765	valid_1's auc: 0.664235
[360]	training's auc: 0.797576	valid_1's auc: 0.664436
[370]	training's auc: 0.798234	valid_1's auc: 0.664649
[380]	training's auc: 0.79878	valid_1's auc: 0.664826
[390]	training's auc: 0.799453	valid_1's auc: 0.664951
[400]	training's auc: 0.799943	valid_1's auc: 0.665135
[410]	training's auc: 0.800518	valid_1's auc: 0.665308
[420]	training's auc: 0.801372	valid_1's auc: 0.665575
[430]	training's auc: 0.801855	valid_1's auc: 0.665698
[440]	training's auc: 0.802456	valid_1's auc: 0.665775
[450]	training's auc: 0.802987	valid_1's auc: 0.665797
[460]	training's auc: 0.803582	valid_1's auc: 0.665951
[470]	training's auc: 0.804124	valid_1's auc: 0.666064
[480]	training's auc: 0.804866	valid_1's auc: 0.66628
[490]	training's auc: 0.805407	valid_1's auc: 0.666434
[500]	training's auc: 0.805855	valid_1's auc: 0.666516
[510]	training's auc: 0.806638	valid_1's auc: 0.666724
[520]	training's auc: 0.807023	valid_1's auc: 0.666768
[530]	training's auc: 0.80761	valid_1's auc: 0.666897
[540]	training's auc: 0.807997	valid_1's auc: 0.666894
[550]	training's auc: 0.80843	valid_1's auc: 0.667008
[560]	training's auc: 0.808948	valid_1's auc: 0.667154
[570]	training's auc: 0.809399	valid_1's auc: 0.667259
[580]	training's auc: 0.809793	valid_1's auc: 0.667418
[590]	training's auc: 0.810159	valid_1's auc: 0.667494
[600]	training's auc: 0.810572	valid_1's auc: 0.667552
[610]	training's auc: 0.811069	valid_1's auc: 0.667765
[620]	training's auc: 0.811401	valid_1's auc: 0.667789
[630]	training's auc: 0.811765	valid_1's auc: 0.667803
[640]	training's auc: 0.812159	valid_1's auc: 0.667928
[650]	training's auc: 0.812538	valid_1's auc: 0.668004
[660]	training's auc: 0.812974	valid_1's auc: 0.668115
[670]	training's auc: 0.813315	valid_1's auc: 0.668161
[680]	training's auc: 0.813605	valid_1's auc: 0.66813
[690]	training's auc: 0.81406	valid_1's auc: 0.668211
[700]	training's auc: 0.814339	valid_1's auc: 0.66822
[710]	training's auc: 0.814652	valid_1's auc: 0.66828
[720]	training's auc: 0.81497	valid_1's auc: 0.66837
[730]	training's auc: 0.815312	valid_1's auc: 0.668508
[740]	training's auc: 0.815611	valid_1's auc: 0.668554
[750]	training's auc: 0.815936	valid_1's auc: 0.668588
[760]	training's auc: 0.816257	valid_1's auc: 0.668616
[770]	training's auc: 0.816692	valid_1's auc: 0.668673
[780]	training's auc: 0.817024	valid_1's auc: 0.668718
[790]	training's auc: 0.817421	valid_1's auc: 0.668749
[800]	training's auc: 0.817703	valid_1's auc: 0.668829
[810]	training's auc: 0.81803	valid_1's auc: 0.668904
[820]	training's auc: 0.818399	valid_1's auc: 0.668984
[830]	training's auc: 0.818716	valid_1's auc: 0.669049
[840]	training's auc: 0.818989	valid_1's auc: 0.669076
[850]	training's auc: 0.819263	valid_1's auc: 0.669121
[860]	training's auc: 0.819597	valid_1's auc: 0.669135
[870]	training's auc: 0.819922	valid_1's auc: 0.66919
[880]	training's auc: 0.820177	valid_1's auc: 0.669122
[890]	training's auc: 0.820484	valid_1's auc: 0.669139
[900]	training's auc: 0.820762	valid_1's auc: 0.669204
[910]	training's auc: 0.821062	valid_1's auc: 0.669206
[920]	training's auc: 0.821294	valid_1's auc: 0.669252
[930]	training's auc: 0.821543	valid_1's auc: 0.669292
[940]	training's auc: 0.821856	valid_1's auc: 0.669356
[950]	training's auc: 0.822108	valid_1's auc: 0.669377
[960]	training's auc: 0.822428	valid_1's auc: 0.669444
[970]	training's auc: 0.822632	valid_1's auc: 0.669492
[980]	training's auc: 0.822866	valid_1's auc: 0.66951
[990]	training's auc: 0.823106	valid_1's auc: 0.669547
[1000]	training's auc: 0.82335	valid_1's auc: 0.669596
[1010]	training's auc: 0.823599	valid_1's auc: 0.66961
[1020]	training's auc: 0.823812	valid_1's auc: 0.669632
[1030]	training's auc: 0.82405	valid_1's auc: 0.669649
[1040]	training's auc: 0.824294	valid_1's auc: 0.669662
[1050]	training's auc: 0.82451	valid_1's auc: 0.66972
[1060]	training's auc: 0.824706	valid_1's auc: 0.66971
[1070]	training's auc: 0.824928	valid_1's auc: 0.66973
[1080]	training's auc: 0.825196	valid_1's auc: 0.669792
[1090]	training's auc: 0.825386	valid_1's auc: 0.669792
[1100]	training's auc: 0.825575	valid_1's auc: 0.669772
[1110]	training's auc: 0.82579	valid_1's auc: 0.669792
[1120]	training's auc: 0.826	valid_1's auc: 0.669837
[1130]	training's auc: 0.826485	valid_1's auc: 0.669986
[1140]	training's auc: 0.826711	valid_1's auc: 0.670023
[1150]	training's auc: 0.826906	valid_1's auc: 0.67002
[1160]	training's auc: 0.827099	valid_1's auc: 0.670029
[1170]	training's auc: 0.827302	valid_1's auc: 0.67002
[1180]	training's auc: 0.827468	valid_1's auc: 0.670022
[1190]	training's auc: 0.827654	valid_1's auc: 0.670041
[1200]	training's auc: 0.827841	valid_1's auc: 0.670041
[1210]	training's auc: 0.828065	valid_1's auc: 0.670038
[1220]	training's auc: 0.828238	valid_1's auc: 0.670039
[1230]	training's auc: 0.828426	valid_1's auc: 0.670034
[1240]	training's auc: 0.828598	valid_1's auc: 0.670047
[1250]	training's auc: 0.828763	valid_1's auc: 0.670073
[1260]	training's auc: 0.828937	valid_1's auc: 0.670041
[1270]	training's auc: 0.829113	valid_1's auc: 0.670055
[1280]	training's auc: 0.829296	valid_1's auc: 0.67009
[1290]	training's auc: 0.829477	valid_1's auc: 0.670119
[1300]	training's auc: 0.82963	valid_1's auc: 0.670142
[1310]	training's auc: 0.829808	valid_1's auc: 0.670119
[1320]	training's auc: 0.830035	valid_1's auc: 0.670147
[1330]	training's auc: 0.83021	valid_1's auc: 0.670161
[1340]	training's auc: 0.830375	valid_1's auc: 0.670167
[1350]	training's auc: 0.83055	valid_1's auc: 0.670175
[1360]	training's auc: 0.830732	valid_1's auc: 0.670227
[1370]	training's auc: 0.830953	valid_1's auc: 0.670251
[1380]	training's auc: 0.831109	valid_1's auc: 0.670275
[1390]	training's auc: 0.831305	valid_1's auc: 0.6703
[1400]	training's auc: 0.831444	valid_1's auc: 0.6703
[1410]	training's auc: 0.831575	valid_1's auc: 0.670291
[1420]	training's auc: 0.831723	valid_1's auc: 0.67028
[1430]	training's auc: 0.831846	valid_1's auc: 0.670282
[1440]	training's auc: 0.832003	valid_1's auc: 0.670273
Early stopping, best iteration is:
[1393]	training's auc: 0.831347	valid_1's auc: 0.670309
best score: 0.670308755653
best iteration: 1393
complete on: ITC_composer

working on: CC11_composer

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
CC11_composer            int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.729482	valid_1's auc: 0.621103
[20]	training's auc: 0.743675	valid_1's auc: 0.625222
[30]	training's auc: 0.750703	valid_1's auc: 0.62868
[40]	training's auc: 0.755484	valid_1's auc: 0.631089
[50]	training's auc: 0.759233	valid_1's auc: 0.63275
[60]	training's auc: 0.761803	valid_1's auc: 0.633935
[70]	training's auc: 0.7646	valid_1's auc: 0.63493
[80]	training's auc: 0.766854	valid_1's auc: 0.63592
[90]	training's auc: 0.768977	valid_1's auc: 0.636705
[100]	training's auc: 0.770993	valid_1's auc: 0.637453
[110]	training's auc: 0.772822	valid_1's auc: 0.638107
[120]	training's auc: 0.774456	valid_1's auc: 0.638667
[130]	training's auc: 0.776033	valid_1's auc: 0.639283
[140]	training's auc: 0.777481	valid_1's auc: 0.639823
[150]	training's auc: 0.778955	valid_1's auc: 0.640328
[160]	training's auc: 0.780263	valid_1's auc: 0.640767
[170]	training's auc: 0.781515	valid_1's auc: 0.641177
[180]	training's auc: 0.782902	valid_1's auc: 0.641752
[190]	training's auc: 0.783986	valid_1's auc: 0.642092
[200]	training's auc: 0.785038	valid_1's auc: 0.642461
[210]	training's auc: 0.786002	valid_1's auc: 0.642762
[220]	training's auc: 0.787045	valid_1's auc: 0.643192
[230]	training's auc: 0.788031	valid_1's auc: 0.643482
[240]	training's auc: 0.78903	valid_1's auc: 0.643801
[250]	training's auc: 0.789909	valid_1's auc: 0.644019
[260]	training's auc: 0.790809	valid_1's auc: 0.64416
[270]	training's auc: 0.791835	valid_1's auc: 0.644484
[280]	training's auc: 0.792589	valid_1's auc: 0.644634
[290]	training's auc: 0.79372	valid_1's auc: 0.644951
[300]	training's auc: 0.794532	valid_1's auc: 0.645056
[310]	training's auc: 0.795271	valid_1's auc: 0.645243
[320]	training's auc: 0.795962	valid_1's auc: 0.645447
[330]	training's auc: 0.796636	valid_1's auc: 0.645749
[340]	training's auc: 0.797318	valid_1's auc: 0.645891
[350]	training's auc: 0.797978	valid_1's auc: 0.646056
[360]	training's auc: 0.798717	valid_1's auc: 0.646182
[370]	training's auc: 0.799304	valid_1's auc: 0.646391
[380]	training's auc: 0.799949	valid_1's auc: 0.646525
[390]	training's auc: 0.80047	valid_1's auc: 0.646677
[400]	training's auc: 0.800915	valid_1's auc: 0.646804
[410]	training's auc: 0.801449	valid_1's auc: 0.646928
[420]	training's auc: 0.802018	valid_1's auc: 0.647103
[430]	training's auc: 0.802501	valid_1's auc: 0.647217
[440]	training's auc: 0.802992	valid_1's auc: 0.64729
[450]	training's auc: 0.803679	valid_1's auc: 0.647489
[460]	training's auc: 0.804166	valid_1's auc: 0.647619
[470]	training's auc: 0.804697	valid_1's auc: 0.647711
[480]	training's auc: 0.805155	valid_1's auc: 0.647831
[490]	training's auc: 0.805733	valid_1's auc: 0.64797
[500]	training's auc: 0.806434	valid_1's auc: 0.648107
[510]	training's auc: 0.806956	valid_1's auc: 0.648201
[520]	training's auc: 0.80742	valid_1's auc: 0.64831
[530]	training's auc: 0.807923	valid_1's auc: 0.648418
[540]	training's auc: 0.808366	valid_1's auc: 0.648463
[550]	training's auc: 0.808756	valid_1's auc: 0.64853
[560]	training's auc: 0.809116	valid_1's auc: 0.648551
[570]	training's auc: 0.809734	valid_1's auc: 0.648706
[580]	training's auc: 0.810036	valid_1's auc: 0.648825
[590]	training's auc: 0.810572	valid_1's auc: 0.648981
[600]	training's auc: 0.810908	valid_1's auc: 0.649049
[610]	training's auc: 0.811236	valid_1's auc: 0.649153
[620]	training's auc: 0.811645	valid_1's auc: 0.649228
[630]	training's auc: 0.812194	valid_1's auc: 0.649371
[640]	training's auc: 0.812564	valid_1's auc: 0.649399
[650]	training's auc: 0.812921	valid_1's auc: 0.649452
[660]	training's auc: 0.813273	valid_1's auc: 0.649505
[670]	training's auc: 0.813574	valid_1's auc: 0.649576
[680]	training's auc: 0.813906	valid_1's auc: 0.649632
[690]	training's auc: 0.814344	valid_1's auc: 0.649736
[700]	training's auc: 0.814945	valid_1's auc: 0.649904
[710]	training's auc: 0.815251	valid_1's auc: 0.649939
[720]	training's auc: 0.815605	valid_1's auc: 0.650006
[730]	training's auc: 0.815908	valid_1's auc: 0.650047
[740]	training's auc: 0.816278	valid_1's auc: 0.650087
[750]	training's auc: 0.816557	valid_1's auc: 0.650117
[760]	training's auc: 0.816975	valid_1's auc: 0.650183
[770]	training's auc: 0.81725	valid_1's auc: 0.65021
[780]	training's auc: 0.817839	valid_1's auc: 0.650321
[790]	training's auc: 0.818165	valid_1's auc: 0.650375
[800]	training's auc: 0.81848	valid_1's auc: 0.650418
[810]	training's auc: 0.818794	valid_1's auc: 0.650464
[820]	training's auc: 0.819218	valid_1's auc: 0.650511
[830]	training's auc: 0.819536	valid_1's auc: 0.650561
[840]	training's auc: 0.819908	valid_1's auc: 0.650613
[850]	training's auc: 0.820309	valid_1's auc: 0.650744
[860]	training's auc: 0.820722	valid_1's auc: 0.650843
[870]	training's auc: 0.82099	valid_1's auc: 0.650846
[880]	training's auc: 0.821293	valid_1's auc: 0.650916
[890]	training's auc: 0.821594	valid_1's auc: 0.650916
[900]	training's auc: 0.821829	valid_1's auc: 0.650961
[910]	training's auc: 0.822162	valid_1's auc: 0.651042
[920]	training's auc: 0.822386	valid_1's auc: 0.651081
[930]	training's auc: 0.822657	valid_1's auc: 0.651126
[940]	training's auc: 0.822971	valid_1's auc: 0.651127
[950]	training's auc: 0.823215	valid_1's auc: 0.65113
[960]	training's auc: 0.823616	valid_1's auc: 0.651189
[970]	training's auc: 0.823845	valid_1's auc: 0.651219
[980]	training's auc: 0.824105	valid_1's auc: 0.651244
[990]	training's auc: 0.824345	valid_1's auc: 0.651258
[1000]	training's auc: 0.824524	valid_1's auc: 0.651259
[1010]	training's auc: 0.824888	valid_1's auc: 0.651299
[1020]	training's auc: 0.825138	valid_1's auc: 0.651341
[1030]	training's auc: 0.825368	valid_1's auc: 0.651347
[1040]	training's auc: 0.825604	valid_1's auc: 0.65134
[1050]	training's auc: 0.82582	valid_1's auc: 0.651361
[1060]	training's auc: 0.82599	valid_1's auc: 0.651376
[1070]	training's auc: 0.826171	valid_1's auc: 0.651413
[1080]	training's auc: 0.82638	valid_1's auc: 0.651436
[1090]	training's auc: 0.826597	valid_1's auc: 0.651456
[1100]	training's auc: 0.826884	valid_1's auc: 0.651518
[1110]	training's auc: 0.827131	valid_1's auc: 0.651544
[1120]	training's auc: 0.827324	valid_1's auc: 0.651555
[1130]	training's auc: 0.827621	valid_1's auc: 0.651602
[1140]	training's auc: 0.827792	valid_1's auc: 0.651602
[1150]	training's auc: 0.828055	valid_1's auc: 0.651647
[1160]	training's auc: 0.828239	valid_1's auc: 0.651664
[1170]	training's auc: 0.828436	valid_1's auc: 0.65169
[1180]	training's auc: 0.828636	valid_1's auc: 0.651706
[1190]	training's auc: 0.828797	valid_1's auc: 0.651682
[1200]	training's auc: 0.829028	valid_1's auc: 0.651719
[1210]	training's auc: 0.829189	valid_1's auc: 0.65172
[1220]	training's auc: 0.829419	valid_1's auc: 0.651749
[1230]	training's auc: 0.829582	valid_1's auc: 0.65175
[1240]	training's auc: 0.82979	valid_1's auc: 0.651757
[1250]	training's auc: 0.82998	valid_1's auc: 0.651807
[1260]	training's auc: 0.830164	valid_1's auc: 0.651801
[1270]	training's auc: 0.830367	valid_1's auc: 0.651795
[1280]	training's auc: 0.830552	valid_1's auc: 0.651812
[1290]	training's auc: 0.830718	valid_1's auc: 0.651837
[1300]	training's auc: 0.830892	valid_1's auc: 0.651861
[1310]	training's auc: 0.831041	valid_1's auc: 0.651877
[1320]	training's auc: 0.831183	valid_1's auc: 0.651878
[1330]	training's auc: 0.831349	valid_1's auc: 0.651847
[1340]	training's auc: 0.831531	valid_1's auc: 0.65187
[1350]	training's auc: 0.8317	valid_1's auc: 0.651891
[1360]	training's auc: 0.831876	valid_1's auc: 0.651916
[1370]	training's auc: 0.832032	valid_1's auc: 0.651922
[1380]	training's auc: 0.832178	valid_1's auc: 0.651936
[1390]	training's auc: 0.832347	valid_1's auc: 0.651962
[1400]	training's auc: 0.83251	valid_1's auc: 0.651986
[1410]	training's auc: 0.832669	valid_1's auc: 0.651994
[1420]	training's auc: 0.832812	valid_1's auc: 0.652005
[1430]	training's auc: 0.832944	valid_1's auc: 0.652019
[1440]	training's auc: 0.833093	valid_1's auc: 0.65201
[1450]	training's auc: 0.833222	valid_1's auc: 0.652018
[1460]	training's auc: 0.833363	valid_1's auc: 0.652027
[1470]	training's auc: 0.833489	valid_1's auc: 0.652046
[1480]	training's auc: 0.833619	valid_1's auc: 0.652059
[1490]	training's auc: 0.833751	valid_1's auc: 0.652061
[1500]	training's auc: 0.833898	valid_1's auc: 0.652054
[1510]	training's auc: 0.834003	valid_1's auc: 0.652043
[1520]	training's auc: 0.834138	valid_1's auc: 0.652038
[1530]	training's auc: 0.834283	valid_1's auc: 0.652042
[1540]	training's auc: 0.834411	valid_1's auc: 0.652019
Early stopping, best iteration is:
[1492]	training's auc: 0.833778	valid_1's auc: 0.652065
best score: 0.652064504099
best iteration: 1492
complete on: CC11_composer

working on: ITC_lyricist

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
ITC_lyricist             int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.723862	valid_1's auc: 0.633519
[20]	training's auc: 0.740938	valid_1's auc: 0.640354
[30]	training's auc: 0.74898	valid_1's auc: 0.644478
[40]	training's auc: 0.752753	valid_1's auc: 0.646445
[50]	training's auc: 0.755953	valid_1's auc: 0.64812
[60]	training's auc: 0.758896	valid_1's auc: 0.649444
[70]	training's auc: 0.761784	valid_1's auc: 0.650569
[80]	training's auc: 0.764322	valid_1's auc: 0.651974
[90]	training's auc: 0.766601	valid_1's auc: 0.652876
[100]	training's auc: 0.768471	valid_1's auc: 0.653548
[110]	training's auc: 0.770282	valid_1's auc: 0.654325
[120]	training's auc: 0.772128	valid_1's auc: 0.655082
[130]	training's auc: 0.773617	valid_1's auc: 0.655678
[140]	training's auc: 0.775113	valid_1's auc: 0.656348
[150]	training's auc: 0.776463	valid_1's auc: 0.656999
[160]	training's auc: 0.778187	valid_1's auc: 0.65772
[170]	training's auc: 0.779325	valid_1's auc: 0.6582
[180]	training's auc: 0.780571	valid_1's auc: 0.658758
[190]	training's auc: 0.781643	valid_1's auc: 0.659143
[200]	training's auc: 0.782679	valid_1's auc: 0.659594
[210]	training's auc: 0.783752	valid_1's auc: 0.659884
[220]	training's auc: 0.784756	valid_1's auc: 0.660294
[230]	training's auc: 0.785675	valid_1's auc: 0.660629
[240]	training's auc: 0.786531	valid_1's auc: 0.660955
[250]	training's auc: 0.78755	valid_1's auc: 0.661304
[260]	training's auc: 0.788481	valid_1's auc: 0.661523
[270]	training's auc: 0.789354	valid_1's auc: 0.661709
[280]	training's auc: 0.790162	valid_1's auc: 0.661871
[290]	training's auc: 0.790897	valid_1's auc: 0.662088
[300]	training's auc: 0.791554	valid_1's auc: 0.662246
[310]	training's auc: 0.79236	valid_1's auc: 0.662562
[320]	training's auc: 0.79318	valid_1's auc: 0.662797
[330]	training's auc: 0.793806	valid_1's auc: 0.663047
[340]	training's auc: 0.794463	valid_1's auc: 0.663128
[350]	training's auc: 0.795021	valid_1's auc: 0.663282
[360]	training's auc: 0.795614	valid_1's auc: 0.663442
[370]	training's auc: 0.796235	valid_1's auc: 0.663561
[380]	training's auc: 0.796787	valid_1's auc: 0.66371
[390]	training's auc: 0.797726	valid_1's auc: 0.664022
[400]	training's auc: 0.798357	valid_1's auc: 0.664264
[410]	training's auc: 0.799165	valid_1's auc: 0.664425
[420]	training's auc: 0.799674	valid_1's auc: 0.664586
[430]	training's auc: 0.800222	valid_1's auc: 0.664695
[440]	training's auc: 0.800769	valid_1's auc: 0.664898
[450]	training's auc: 0.801323	valid_1's auc: 0.665056
[460]	training's auc: 0.801755	valid_1's auc: 0.665141
[470]	training's auc: 0.802222	valid_1's auc: 0.665225
[480]	training's auc: 0.802687	valid_1's auc: 0.665385
[490]	training's auc: 0.803554	valid_1's auc: 0.66563
[500]	training's auc: 0.804296	valid_1's auc: 0.665844
[510]	training's auc: 0.804711	valid_1's auc: 0.665923
[520]	training's auc: 0.805195	valid_1's auc: 0.665991
[530]	training's auc: 0.805683	valid_1's auc: 0.666086
[540]	training's auc: 0.806207	valid_1's auc: 0.666266
[550]	training's auc: 0.806763	valid_1's auc: 0.666407
[560]	training's auc: 0.807293	valid_1's auc: 0.666543
[570]	training's auc: 0.807799	valid_1's auc: 0.66668
[580]	training's auc: 0.808097	valid_1's auc: 0.666761
[590]	training's auc: 0.808417	valid_1's auc: 0.666781
[600]	training's auc: 0.808769	valid_1's auc: 0.666882
[610]	training's auc: 0.809071	valid_1's auc: 0.66697
[620]	training's auc: 0.809576	valid_1's auc: 0.667075
[630]	training's auc: 0.810001	valid_1's auc: 0.667265
[640]	training's auc: 0.810322	valid_1's auc: 0.667308
[650]	training's auc: 0.810811	valid_1's auc: 0.667405
[660]	training's auc: 0.811214	valid_1's auc: 0.667479
[670]	training's auc: 0.811712	valid_1's auc: 0.667537
[680]	training's auc: 0.812063	valid_1's auc: 0.66759
[690]	training's auc: 0.812481	valid_1's auc: 0.667666
[700]	training's auc: 0.812911	valid_1's auc: 0.66779
[710]	training's auc: 0.813233	valid_1's auc: 0.667839
[720]	training's auc: 0.813526	valid_1's auc: 0.667937
[730]	training's auc: 0.813779	valid_1's auc: 0.667968
[740]	training's auc: 0.81419	valid_1's auc: 0.668092
[750]	training's auc: 0.814506	valid_1's auc: 0.668168
[760]	training's auc: 0.814794	valid_1's auc: 0.668182
[770]	training's auc: 0.815262	valid_1's auc: 0.668327
[780]	training's auc: 0.815519	valid_1's auc: 0.668367
[790]	training's auc: 0.815923	valid_1's auc: 0.66842
[800]	training's auc: 0.816294	valid_1's auc: 0.668458
[810]	training's auc: 0.816718	valid_1's auc: 0.668527
[820]	training's auc: 0.816971	valid_1's auc: 0.668576
[830]	training's auc: 0.817425	valid_1's auc: 0.668622
[840]	training's auc: 0.817721	valid_1's auc: 0.668683
[850]	training's auc: 0.818021	valid_1's auc: 0.668689
[860]	training's auc: 0.818449	valid_1's auc: 0.66873
[870]	training's auc: 0.818754	valid_1's auc: 0.668812
[880]	training's auc: 0.819137	valid_1's auc: 0.668872
[890]	training's auc: 0.819513	valid_1's auc: 0.668957
[900]	training's auc: 0.819734	valid_1's auc: 0.668949
[910]	training's auc: 0.820039	valid_1's auc: 0.669031
[920]	training's auc: 0.820249	valid_1's auc: 0.66905
[930]	training's auc: 0.820587	valid_1's auc: 0.669107
[940]	training's auc: 0.82082	valid_1's auc: 0.669136
[950]	training's auc: 0.821093	valid_1's auc: 0.669181
[960]	training's auc: 0.821415	valid_1's auc: 0.669272
[970]	training's auc: 0.8217	valid_1's auc: 0.669321
[980]	training's auc: 0.822019	valid_1's auc: 0.669365
[990]	training's auc: 0.822288	valid_1's auc: 0.669352
[1000]	training's auc: 0.822499	valid_1's auc: 0.669363
[1010]	training's auc: 0.822812	valid_1's auc: 0.669478
[1020]	training's auc: 0.823117	valid_1's auc: 0.669503
[1030]	training's auc: 0.823413	valid_1's auc: 0.669593
[1040]	training's auc: 0.823653	valid_1's auc: 0.669649
[1050]	training's auc: 0.823865	valid_1's auc: 0.66972
[1060]	training's auc: 0.824143	valid_1's auc: 0.669802
[1070]	training's auc: 0.824297	valid_1's auc: 0.669815
[1080]	training's auc: 0.824541	valid_1's auc: 0.669841
[1090]	training's auc: 0.824718	valid_1's auc: 0.669866
[1100]	training's auc: 0.824911	valid_1's auc: 0.669868
[1110]	training's auc: 0.825181	valid_1's auc: 0.669911
[1120]	training's auc: 0.825384	valid_1's auc: 0.669959
[1130]	training's auc: 0.825599	valid_1's auc: 0.669988
[1140]	training's auc: 0.825863	valid_1's auc: 0.67008
[1150]	training's auc: 0.826096	valid_1's auc: 0.670106
[1160]	training's auc: 0.826312	valid_1's auc: 0.670134
[1170]	training's auc: 0.826534	valid_1's auc: 0.670155
[1180]	training's auc: 0.826687	valid_1's auc: 0.670164
[1190]	training's auc: 0.826875	valid_1's auc: 0.670213
[1200]	training's auc: 0.827115	valid_1's auc: 0.670262
[1210]	training's auc: 0.827298	valid_1's auc: 0.670286
[1220]	training's auc: 0.827492	valid_1's auc: 0.670287
[1230]	training's auc: 0.827693	valid_1's auc: 0.6703
[1240]	training's auc: 0.827892	valid_1's auc: 0.670304
[1250]	training's auc: 0.828054	valid_1's auc: 0.670305
[1260]	training's auc: 0.828332	valid_1's auc: 0.670344
[1270]	training's auc: 0.828764	valid_1's auc: 0.670487
[1280]	training's auc: 0.82895	valid_1's auc: 0.670533
[1290]	training's auc: 0.829118	valid_1's auc: 0.670544
[1300]	training's auc: 0.82929	valid_1's auc: 0.67057
[1310]	training's auc: 0.829495	valid_1's auc: 0.670553
[1320]	training's auc: 0.829655	valid_1's auc: 0.670563
[1330]	training's auc: 0.829823	valid_1's auc: 0.670567
[1340]	training's auc: 0.830022	valid_1's auc: 0.670601
[1350]	training's auc: 0.830173	valid_1's auc: 0.670621
[1360]	training's auc: 0.830345	valid_1's auc: 0.670593
[1370]	training's auc: 0.8305	valid_1's auc: 0.670608
[1380]	training's auc: 0.830628	valid_1's auc: 0.670596
[1390]	training's auc: 0.830779	valid_1's auc: 0.670636
[1400]	training's auc: 0.830924	valid_1's auc: 0.670646
[1410]	training's auc: 0.83108	valid_1's auc: 0.670655
[1420]	training's auc: 0.831229	valid_1's auc: 0.670654
[1430]	training's auc: 0.83139	valid_1's auc: 0.670689
[1440]	training's auc: 0.831538	valid_1's auc: 0.670708
[1450]	training's auc: 0.831701	valid_1's auc: 0.67072
[1460]	training's auc: 0.831847	valid_1's auc: 0.670706
[1470]	training's auc: 0.832031	valid_1's auc: 0.670734
[1480]	training's auc: 0.832135	valid_1's auc: 0.670717
[1490]	training's auc: 0.83227	valid_1's auc: 0.670721
[1500]	training's auc: 0.83243	valid_1's auc: 0.67073
[1510]	training's auc: 0.832538	valid_1's auc: 0.670735
Early stopping, best iteration is:
[1468]	training's auc: 0.831996	valid_1's auc: 0.670744
best score: 0.67074389491
best iteration: 1468
complete on: ITC_lyricist

working on: CC11_lyricist

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
CC11_lyricist            int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.724314	valid_1's auc: 0.622307
[20]	training's auc: 0.740169	valid_1's auc: 0.627748
[30]	training's auc: 0.748104	valid_1's auc: 0.630586
[40]	training's auc: 0.752516	valid_1's auc: 0.632681
[50]	training's auc: 0.75575	valid_1's auc: 0.634092
[60]	training's auc: 0.758738	valid_1's auc: 0.635488
[70]	training's auc: 0.761438	valid_1's auc: 0.636664
[80]	training's auc: 0.763759	valid_1's auc: 0.637506
[90]	training's auc: 0.765962	valid_1's auc: 0.638343
[100]	training's auc: 0.76795	valid_1's auc: 0.639207
[110]	training's auc: 0.769676	valid_1's auc: 0.639778
[120]	training's auc: 0.771425	valid_1's auc: 0.640439
[130]	training's auc: 0.772922	valid_1's auc: 0.640985
[140]	training's auc: 0.774704	valid_1's auc: 0.641787
[150]	training's auc: 0.776132	valid_1's auc: 0.642542
[160]	training's auc: 0.778005	valid_1's auc: 0.64309
[170]	training's auc: 0.779454	valid_1's auc: 0.64358
[180]	training's auc: 0.780622	valid_1's auc: 0.643973
[190]	training's auc: 0.781817	valid_1's auc: 0.644434
[200]	training's auc: 0.782845	valid_1's auc: 0.644834
[210]	training's auc: 0.783851	valid_1's auc: 0.645213
[220]	training's auc: 0.784864	valid_1's auc: 0.645612
[230]	training's auc: 0.785786	valid_1's auc: 0.64587
[240]	training's auc: 0.786811	valid_1's auc: 0.64613
[250]	training's auc: 0.787745	valid_1's auc: 0.646438
[260]	training's auc: 0.788623	valid_1's auc: 0.646615
[270]	training's auc: 0.789408	valid_1's auc: 0.646817
[280]	training's auc: 0.790194	valid_1's auc: 0.646988
[290]	training's auc: 0.790897	valid_1's auc: 0.647294
[300]	training's auc: 0.791658	valid_1's auc: 0.647445
[310]	training's auc: 0.792489	valid_1's auc: 0.647676
[320]	training's auc: 0.793203	valid_1's auc: 0.647949
[330]	training's auc: 0.793897	valid_1's auc: 0.648163
[340]	training's auc: 0.794829	valid_1's auc: 0.648468
[350]	training's auc: 0.795455	valid_1's auc: 0.648607
[360]	training's auc: 0.79626	valid_1's auc: 0.648766
[370]	training's auc: 0.796855	valid_1's auc: 0.648941
[380]	training's auc: 0.79748	valid_1's auc: 0.649049
[390]	training's auc: 0.798108	valid_1's auc: 0.649221
[400]	training's auc: 0.798655	valid_1's auc: 0.649424
[410]	training's auc: 0.799201	valid_1's auc: 0.649505
[420]	training's auc: 0.799712	valid_1's auc: 0.649734
[430]	training's auc: 0.800224	valid_1's auc: 0.649918
[440]	training's auc: 0.800802	valid_1's auc: 0.650048
[450]	training's auc: 0.801556	valid_1's auc: 0.650159
[460]	training's auc: 0.802032	valid_1's auc: 0.650245
[470]	training's auc: 0.80257	valid_1's auc: 0.650347
[480]	training's auc: 0.803356	valid_1's auc: 0.650526
[490]	training's auc: 0.804187	valid_1's auc: 0.650776
[500]	training's auc: 0.804668	valid_1's auc: 0.650903
[510]	training's auc: 0.805243	valid_1's auc: 0.651033
[520]	training's auc: 0.805681	valid_1's auc: 0.651098
[530]	training's auc: 0.806144	valid_1's auc: 0.651207
[540]	training's auc: 0.806534	valid_1's auc: 0.651311
[550]	training's auc: 0.806939	valid_1's auc: 0.651409
[560]	training's auc: 0.807372	valid_1's auc: 0.65149
[570]	training's auc: 0.807817	valid_1's auc: 0.651608
[580]	training's auc: 0.808226	valid_1's auc: 0.65171
[590]	training's auc: 0.808752	valid_1's auc: 0.651741
[600]	training's auc: 0.809509	valid_1's auc: 0.651966
[610]	training's auc: 0.809818	valid_1's auc: 0.65206
[620]	training's auc: 0.810248	valid_1's auc: 0.652167
[630]	training's auc: 0.810591	valid_1's auc: 0.652219
[640]	training's auc: 0.810982	valid_1's auc: 0.652287
[650]	training's auc: 0.811396	valid_1's auc: 0.652339
[660]	training's auc: 0.811821	valid_1's auc: 0.652449
[670]	training's auc: 0.812197	valid_1's auc: 0.65251
[680]	training's auc: 0.812548	valid_1's auc: 0.652525
[690]	training's auc: 0.812966	valid_1's auc: 0.652588
[700]	training's auc: 0.813342	valid_1's auc: 0.652689
[710]	training's auc: 0.813704	valid_1's auc: 0.652716
[720]	training's auc: 0.814017	valid_1's auc: 0.652762
[730]	training's auc: 0.814295	valid_1's auc: 0.652792
[740]	training's auc: 0.814608	valid_1's auc: 0.652814
[750]	training's auc: 0.814908	valid_1's auc: 0.652879
[760]	training's auc: 0.815291	valid_1's auc: 0.652924
[770]	training's auc: 0.815859	valid_1's auc: 0.653109
[780]	training's auc: 0.816376	valid_1's auc: 0.653231
[790]	training's auc: 0.816663	valid_1's auc: 0.653272
[800]	training's auc: 0.816939	valid_1's auc: 0.653308
[810]	training's auc: 0.817257	valid_1's auc: 0.653353
[820]	training's auc: 0.8176	valid_1's auc: 0.65337
[830]	training's auc: 0.817963	valid_1's auc: 0.653472
[840]	training's auc: 0.818255	valid_1's auc: 0.653522
[850]	training's auc: 0.818585	valid_1's auc: 0.653537
[860]	training's auc: 0.81905	valid_1's auc: 0.653618
[870]	training's auc: 0.819358	valid_1's auc: 0.653682
[880]	training's auc: 0.819671	valid_1's auc: 0.6537
[890]	training's auc: 0.81998	valid_1's auc: 0.653763
[900]	training's auc: 0.82022	valid_1's auc: 0.653818
[910]	training's auc: 0.82049	valid_1's auc: 0.653873
[920]	training's auc: 0.820716	valid_1's auc: 0.653898
[930]	training's auc: 0.820996	valid_1's auc: 0.653903
[940]	training's auc: 0.821234	valid_1's auc: 0.653902
[950]	training's auc: 0.821468	valid_1's auc: 0.653911
[960]	training's auc: 0.821751	valid_1's auc: 0.653959
[970]	training's auc: 0.822055	valid_1's auc: 0.654014
[980]	training's auc: 0.822285	valid_1's auc: 0.654029
[990]	training's auc: 0.822657	valid_1's auc: 0.654179
[1000]	training's auc: 0.822924	valid_1's auc: 0.654216
[1010]	training's auc: 0.823159	valid_1's auc: 0.654213
[1020]	training's auc: 0.823407	valid_1's auc: 0.654239
[1030]	training's auc: 0.823647	valid_1's auc: 0.654313
[1040]	training's auc: 0.823883	valid_1's auc: 0.654334
[1050]	training's auc: 0.824141	valid_1's auc: 0.654394
[1060]	training's auc: 0.824401	valid_1's auc: 0.654422
[1070]	training's auc: 0.824574	valid_1's auc: 0.654409
[1080]	training's auc: 0.824849	valid_1's auc: 0.654439
[1090]	training's auc: 0.825051	valid_1's auc: 0.654453
[1100]	training's auc: 0.825238	valid_1's auc: 0.654453
[1110]	training's auc: 0.825515	valid_1's auc: 0.654464
[1120]	training's auc: 0.825764	valid_1's auc: 0.654473
[1130]	training's auc: 0.825966	valid_1's auc: 0.654485
[1140]	training's auc: 0.826174	valid_1's auc: 0.654489
[1150]	training's auc: 0.826436	valid_1's auc: 0.654517
[1160]	training's auc: 0.826721	valid_1's auc: 0.654582
[1170]	training's auc: 0.826944	valid_1's auc: 0.654626
[1180]	training's auc: 0.82711	valid_1's auc: 0.65465
[1190]	training's auc: 0.82725	valid_1's auc: 0.654669
[1200]	training's auc: 0.827427	valid_1's auc: 0.65468
[1210]	training's auc: 0.827629	valid_1's auc: 0.654704
[1220]	training's auc: 0.827875	valid_1's auc: 0.654731
[1230]	training's auc: 0.828121	valid_1's auc: 0.654725
[1240]	training's auc: 0.828439	valid_1's auc: 0.654753
[1250]	training's auc: 0.828747	valid_1's auc: 0.654803
[1260]	training's auc: 0.828902	valid_1's auc: 0.654815
[1270]	training's auc: 0.829059	valid_1's auc: 0.654847
[1280]	training's auc: 0.82924	valid_1's auc: 0.654874
[1290]	training's auc: 0.829415	valid_1's auc: 0.654887
[1300]	training's auc: 0.829562	valid_1's auc: 0.654905
[1310]	training's auc: 0.829724	valid_1's auc: 0.654904
[1320]	training's auc: 0.829904	valid_1's auc: 0.65489
[1330]	training's auc: 0.830125	valid_1's auc: 0.654908
[1340]	training's auc: 0.830296	valid_1's auc: 0.654927
[1350]	training's auc: 0.830452	valid_1's auc: 0.654923
[1360]	training's auc: 0.830611	valid_1's auc: 0.654949
[1370]	training's auc: 0.83083	valid_1's auc: 0.654987
[1380]	training's auc: 0.83096	valid_1's auc: 0.655002
[1390]	training's auc: 0.831137	valid_1's auc: 0.655019
[1400]	training's auc: 0.83129	valid_1's auc: 0.655018
[1410]	training's auc: 0.831445	valid_1's auc: 0.655036
[1420]	training's auc: 0.831593	valid_1's auc: 0.655061
[1430]	training's auc: 0.831722	valid_1's auc: 0.655054
[1440]	training's auc: 0.831881	valid_1's auc: 0.655068
[1450]	training's auc: 0.832038	valid_1's auc: 0.655115
[1460]	training's auc: 0.832167	valid_1's auc: 0.655116
[1470]	training's auc: 0.832335	valid_1's auc: 0.655124
[1480]	training's auc: 0.832468	valid_1's auc: 0.655156
[1490]	training's auc: 0.832602	valid_1's auc: 0.655188
[1500]	training's auc: 0.832728	valid_1's auc: 0.655193
[1510]	training's auc: 0.832942	valid_1's auc: 0.655221
[1520]	training's auc: 0.833053	valid_1's auc: 0.655218
[1530]	training's auc: 0.833209	valid_1's auc: 0.655242
[1540]	training's auc: 0.833366	valid_1's auc: 0.655271
[1550]	training's auc: 0.833495	valid_1's auc: 0.655283
[1560]	training's auc: 0.833641	valid_1's auc: 0.655302
[1570]	training's auc: 0.833749	valid_1's auc: 0.655308
[1580]	training's auc: 0.833951	valid_1's auc: 0.655358
[1590]	training's auc: 0.834083	valid_1's auc: 0.655376
[1600]	training's auc: 0.834253	valid_1's auc: 0.655385
[1610]	training's auc: 0.834408	valid_1's auc: 0.655379
[1620]	training's auc: 0.834545	valid_1's auc: 0.655388
[1630]	training's auc: 0.83468	valid_1's auc: 0.655392
[1640]	training's auc: 0.834856	valid_1's auc: 0.655372
[1650]	training's auc: 0.835038	valid_1's auc: 0.655368
[1660]	training's auc: 0.835209	valid_1's auc: 0.655403
[1670]	training's auc: 0.835337	valid_1's auc: 0.655404
[1680]	training's auc: 0.835473	valid_1's auc: 0.65544
[1690]	training's auc: 0.835578	valid_1's auc: 0.655443
[1700]	training's auc: 0.835707	valid_1's auc: 0.655461
[1710]	training's auc: 0.835834	valid_1's auc: 0.655472
[1720]	training's auc: 0.83595	valid_1's auc: 0.655479
[1730]	training's auc: 0.836082	valid_1's auc: 0.655479
[1740]	training's auc: 0.83619	valid_1's auc: 0.655477
[1750]	training's auc: 0.836312	valid_1's auc: 0.655484
[1760]	training's auc: 0.83641	valid_1's auc: 0.655496
[1770]	training's auc: 0.836574	valid_1's auc: 0.655499
[1780]	training's auc: 0.836686	valid_1's auc: 0.655477
[1790]	training's auc: 0.83681	valid_1's auc: 0.655482
[1800]	training's auc: 0.836939	valid_1's auc: 0.655512
[1810]	training's auc: 0.837068	valid_1's auc: 0.655527
[1820]	training's auc: 0.83719	valid_1's auc: 0.655516
[1830]	training's auc: 0.837329	valid_1's auc: 0.655546
[1840]	training's auc: 0.837505	valid_1's auc: 0.655562
[1850]	training's auc: 0.837665	valid_1's auc: 0.655593
[1860]	training's auc: 0.837767	valid_1's auc: 0.655593
[1870]	training's auc: 0.837882	valid_1's auc: 0.655573
[1880]	training's auc: 0.837972	valid_1's auc: 0.655573
[1890]	training's auc: 0.838086	valid_1's auc: 0.655551
[1900]	training's auc: 0.838201	valid_1's auc: 0.655537
Early stopping, best iteration is:
[1854]	training's auc: 0.837707	valid_1's auc: 0.655596
best score: 0.655596430335
best iteration: 1854
complete on: CC11_lyricist

working on: ITC_language

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
ITC_language             int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.724357	valid_1's auc: 0.63154
[20]	training's auc: 0.739243	valid_1's auc: 0.638019
[30]	training's auc: 0.747526	valid_1's auc: 0.642302
[40]	training's auc: 0.751329	valid_1's auc: 0.64406
[50]	training's auc: 0.755156	valid_1's auc: 0.646126
[60]	training's auc: 0.758129	valid_1's auc: 0.647228
[70]	training's auc: 0.760689	valid_1's auc: 0.648284
[80]	training's auc: 0.763004	valid_1's auc: 0.649504
[90]	training's auc: 0.765128	valid_1's auc: 0.650375
[100]	training's auc: 0.767022	valid_1's auc: 0.651095
[110]	training's auc: 0.769319	valid_1's auc: 0.652018
[120]	training's auc: 0.771244	valid_1's auc: 0.652358
[130]	training's auc: 0.772918	valid_1's auc: 0.652976
[140]	training's auc: 0.774247	valid_1's auc: 0.653562
[150]	training's auc: 0.77578	valid_1's auc: 0.65419
[160]	training's auc: 0.777606	valid_1's auc: 0.654806
[170]	training's auc: 0.778777	valid_1's auc: 0.655363
[180]	training's auc: 0.779926	valid_1's auc: 0.655881
[190]	training's auc: 0.781022	valid_1's auc: 0.6563
[200]	training's auc: 0.781943	valid_1's auc: 0.65675
[210]	training's auc: 0.783056	valid_1's auc: 0.657103
[220]	training's auc: 0.783925	valid_1's auc: 0.65743
[230]	training's auc: 0.784799	valid_1's auc: 0.657756
[240]	training's auc: 0.785803	valid_1's auc: 0.658056
[250]	training's auc: 0.786695	valid_1's auc: 0.658344
[260]	training's auc: 0.787562	valid_1's auc: 0.658538
[270]	training's auc: 0.788378	valid_1's auc: 0.658769
[280]	training's auc: 0.789334	valid_1's auc: 0.659006
[290]	training's auc: 0.790078	valid_1's auc: 0.659255
[300]	training's auc: 0.790971	valid_1's auc: 0.659516
[310]	training's auc: 0.791691	valid_1's auc: 0.659717
[320]	training's auc: 0.79264	valid_1's auc: 0.659957
[330]	training's auc: 0.793167	valid_1's auc: 0.660205
[340]	training's auc: 0.794007	valid_1's auc: 0.660438
[350]	training's auc: 0.794848	valid_1's auc: 0.660728
[360]	training's auc: 0.795783	valid_1's auc: 0.661076
[370]	training's auc: 0.796373	valid_1's auc: 0.661258
[380]	training's auc: 0.796943	valid_1's auc: 0.661478
[390]	training's auc: 0.797505	valid_1's auc: 0.661561
[400]	training's auc: 0.797943	valid_1's auc: 0.661693
[410]	training's auc: 0.798556	valid_1's auc: 0.661948
[420]	training's auc: 0.799077	valid_1's auc: 0.662119
[430]	training's auc: 0.799536	valid_1's auc: 0.662217
[440]	training's auc: 0.800263	valid_1's auc: 0.662464
[450]	training's auc: 0.800798	valid_1's auc: 0.662566
[460]	training's auc: 0.801361	valid_1's auc: 0.662653
[470]	training's auc: 0.802013	valid_1's auc: 0.662908
[480]	training's auc: 0.802544	valid_1's auc: 0.66303
[490]	training's auc: 0.802968	valid_1's auc: 0.663135
[500]	training's auc: 0.803468	valid_1's auc: 0.663295
[510]	training's auc: 0.80404	valid_1's auc: 0.663423
[520]	training's auc: 0.804474	valid_1's auc: 0.663522
[530]	training's auc: 0.805241	valid_1's auc: 0.663771
[540]	training's auc: 0.805654	valid_1's auc: 0.663807
[550]	training's auc: 0.806181	valid_1's auc: 0.664038
[560]	training's auc: 0.806555	valid_1's auc: 0.664151
[570]	training's auc: 0.806926	valid_1's auc: 0.664231
[580]	training's auc: 0.807295	valid_1's auc: 0.664369
[590]	training's auc: 0.807813	valid_1's auc: 0.664544
[600]	training's auc: 0.808278	valid_1's auc: 0.664634
[610]	training's auc: 0.808654	valid_1's auc: 0.664756
[620]	training's auc: 0.809035	valid_1's auc: 0.664864
[630]	training's auc: 0.80941	valid_1's auc: 0.664897
[640]	training's auc: 0.809748	valid_1's auc: 0.664965
[650]	training's auc: 0.810036	valid_1's auc: 0.665049
[660]	training's auc: 0.810518	valid_1's auc: 0.665112
[670]	training's auc: 0.810863	valid_1's auc: 0.665207
[680]	training's auc: 0.811374	valid_1's auc: 0.665367
[690]	training's auc: 0.811802	valid_1's auc: 0.665406
[700]	training's auc: 0.812208	valid_1's auc: 0.665484
[710]	training's auc: 0.812645	valid_1's auc: 0.665521
[720]	training's auc: 0.813051	valid_1's auc: 0.665632
[730]	training's auc: 0.813304	valid_1's auc: 0.665641
[740]	training's auc: 0.813605	valid_1's auc: 0.665702
[750]	training's auc: 0.813867	valid_1's auc: 0.665738
[760]	training's auc: 0.814246	valid_1's auc: 0.665799
[770]	training's auc: 0.814777	valid_1's auc: 0.665943
[780]	training's auc: 0.815174	valid_1's auc: 0.666002
[790]	training's auc: 0.815473	valid_1's auc: 0.666046
[800]	training's auc: 0.815747	valid_1's auc: 0.66609
[810]	training's auc: 0.816033	valid_1's auc: 0.666124
[820]	training's auc: 0.816409	valid_1's auc: 0.666203
[830]	training's auc: 0.816731	valid_1's auc: 0.666322
[840]	training's auc: 0.817021	valid_1's auc: 0.666381
[850]	training's auc: 0.817328	valid_1's auc: 0.666468
[860]	training's auc: 0.817625	valid_1's auc: 0.666458
[870]	training's auc: 0.81801	valid_1's auc: 0.666554
[880]	training's auc: 0.818456	valid_1's auc: 0.666673
[890]	training's auc: 0.818724	valid_1's auc: 0.666717
[900]	training's auc: 0.818969	valid_1's auc: 0.666764
[910]	training's auc: 0.819287	valid_1's auc: 0.666811
[920]	training's auc: 0.819512	valid_1's auc: 0.666804
[930]	training's auc: 0.819786	valid_1's auc: 0.666822
[940]	training's auc: 0.820153	valid_1's auc: 0.666889
[950]	training's auc: 0.820492	valid_1's auc: 0.666963
[960]	training's auc: 0.82078	valid_1's auc: 0.667047
[970]	training's auc: 0.821072	valid_1's auc: 0.667092
[980]	training's auc: 0.821318	valid_1's auc: 0.667127
[990]	training's auc: 0.821575	valid_1's auc: 0.667185
[1000]	training's auc: 0.821791	valid_1's auc: 0.667209
[1010]	training's auc: 0.822044	valid_1's auc: 0.667261
[1020]	training's auc: 0.822336	valid_1's auc: 0.667305
[1030]	training's auc: 0.822591	valid_1's auc: 0.667337
[1040]	training's auc: 0.822842	valid_1's auc: 0.667349
[1050]	training's auc: 0.823084	valid_1's auc: 0.66735
[1060]	training's auc: 0.82336	valid_1's auc: 0.667391
[1070]	training's auc: 0.823551	valid_1's auc: 0.667421
[1080]	training's auc: 0.82373	valid_1's auc: 0.667461
[1090]	training's auc: 0.823955	valid_1's auc: 0.66752
[1100]	training's auc: 0.824176	valid_1's auc: 0.667547
[1110]	training's auc: 0.82439	valid_1's auc: 0.667552
[1120]	training's auc: 0.824653	valid_1's auc: 0.66762
[1130]	training's auc: 0.824873	valid_1's auc: 0.667638
[1140]	training's auc: 0.825054	valid_1's auc: 0.667671
[1150]	training's auc: 0.825269	valid_1's auc: 0.667671
[1160]	training's auc: 0.825444	valid_1's auc: 0.667702
[1170]	training's auc: 0.82564	valid_1's auc: 0.667728
[1180]	training's auc: 0.825838	valid_1's auc: 0.667723
[1190]	training's auc: 0.825982	valid_1's auc: 0.667719
[1200]	training's auc: 0.826182	valid_1's auc: 0.667792
[1210]	training's auc: 0.826371	valid_1's auc: 0.667814
[1220]	training's auc: 0.826579	valid_1's auc: 0.667888
[1230]	training's auc: 0.826777	valid_1's auc: 0.667909
[1240]	training's auc: 0.826955	valid_1's auc: 0.667913
[1250]	training's auc: 0.827081	valid_1's auc: 0.667928
[1260]	training's auc: 0.827256	valid_1's auc: 0.667936
[1270]	training's auc: 0.82761	valid_1's auc: 0.668017
[1280]	training's auc: 0.827794	valid_1's auc: 0.66806
[1290]	training's auc: 0.827963	valid_1's auc: 0.668057
[1300]	training's auc: 0.828121	valid_1's auc: 0.668078
[1310]	training's auc: 0.828279	valid_1's auc: 0.668073
[1320]	training's auc: 0.828425	valid_1's auc: 0.668067
[1330]	training's auc: 0.828583	valid_1's auc: 0.668066
[1340]	training's auc: 0.82883	valid_1's auc: 0.668138
[1350]	training's auc: 0.828984	valid_1's auc: 0.66816
[1360]	training's auc: 0.829132	valid_1's auc: 0.668159
[1370]	training's auc: 0.829262	valid_1's auc: 0.668142
[1380]	training's auc: 0.82949	valid_1's auc: 0.668172
[1390]	training's auc: 0.829623	valid_1's auc: 0.668159
[1400]	training's auc: 0.829782	valid_1's auc: 0.668189
[1410]	training's auc: 0.829929	valid_1's auc: 0.668196
[1420]	training's auc: 0.830083	valid_1's auc: 0.668195
[1430]	training's auc: 0.830194	valid_1's auc: 0.668206
[1440]	training's auc: 0.830453	valid_1's auc: 0.668301
[1450]	training's auc: 0.830623	valid_1's auc: 0.668329
[1460]	training's auc: 0.830741	valid_1's auc: 0.668321
[1470]	training's auc: 0.830857	valid_1's auc: 0.668301
[1480]	training's auc: 0.830995	valid_1's auc: 0.668338
[1490]	training's auc: 0.831125	valid_1's auc: 0.668366
[1500]	training's auc: 0.831276	valid_1's auc: 0.668403
[1510]	training's auc: 0.831393	valid_1's auc: 0.668401
[1520]	training's auc: 0.831503	valid_1's auc: 0.668385
[1530]	training's auc: 0.831695	valid_1's auc: 0.668411
[1540]	training's auc: 0.83183	valid_1's auc: 0.668382
[1550]	training's auc: 0.831958	valid_1's auc: 0.668405
[1560]	training's auc: 0.832123	valid_1's auc: 0.668426
[1570]	training's auc: 0.83224	valid_1's auc: 0.668442
[1580]	training's auc: 0.832339	valid_1's auc: 0.66843
[1590]	training's auc: 0.832463	valid_1's auc: 0.668455
[1600]	training's auc: 0.832693	valid_1's auc: 0.668484
[1610]	training's auc: 0.832836	valid_1's auc: 0.668472
[1620]	training's auc: 0.832958	valid_1's auc: 0.668487
[1630]	training's auc: 0.83308	valid_1's auc: 0.668517
[1640]	training's auc: 0.833196	valid_1's auc: 0.668524
[1650]	training's auc: 0.833303	valid_1's auc: 0.668508
[1660]	training's auc: 0.833429	valid_1's auc: 0.668532
[1670]	training's auc: 0.833586	valid_1's auc: 0.66854
[1680]	training's auc: 0.833728	valid_1's auc: 0.668581
[1690]	training's auc: 0.833845	valid_1's auc: 0.668583
[1700]	training's auc: 0.833957	valid_1's auc: 0.668603
[1710]	training's auc: 0.834077	valid_1's auc: 0.668635
[1720]	training's auc: 0.834179	valid_1's auc: 0.668639
[1730]	training's auc: 0.834297	valid_1's auc: 0.668634
[1740]	training's auc: 0.834393	valid_1's auc: 0.668646
[1750]	training's auc: 0.83452	valid_1's auc: 0.668668
[1760]	training's auc: 0.834617	valid_1's auc: 0.668673
[1770]	training's auc: 0.834727	valid_1's auc: 0.668716
[1780]	training's auc: 0.834847	valid_1's auc: 0.668708
[1790]	training's auc: 0.834958	valid_1's auc: 0.668699
[1800]	training's auc: 0.835148	valid_1's auc: 0.668717
[1810]	training's auc: 0.835265	valid_1's auc: 0.668748
[1820]	training's auc: 0.83539	valid_1's auc: 0.668729
[1830]	training's auc: 0.835532	valid_1's auc: 0.668736
[1840]	training's auc: 0.835635	valid_1's auc: 0.668726
[1850]	training's auc: 0.835723	valid_1's auc: 0.668719
[1860]	training's auc: 0.835816	valid_1's auc: 0.668729
Early stopping, best iteration is:
[1811]	training's auc: 0.835283	valid_1's auc: 0.668752
best score: 0.668751590622
best iteration: 1811
complete on: ITC_language

working on: CC11_language

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
CC11_language            int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.723386	valid_1's auc: 0.630935
[20]	training's auc: 0.73938	valid_1's auc: 0.638041
[30]	training's auc: 0.748011	valid_1's auc: 0.641817
[40]	training's auc: 0.752275	valid_1's auc: 0.644194
[50]	training's auc: 0.755718	valid_1's auc: 0.645908
[60]	training's auc: 0.759029	valid_1's auc: 0.647259
[70]	training's auc: 0.7614	valid_1's auc: 0.648172
[80]	training's auc: 0.763683	valid_1's auc: 0.649417
[90]	training's auc: 0.765863	valid_1's auc: 0.650211
[100]	training's auc: 0.767936	valid_1's auc: 0.651059
[110]	training's auc: 0.769892	valid_1's auc: 0.651921
[120]	training's auc: 0.771632	valid_1's auc: 0.652784
[130]	training's auc: 0.773038	valid_1's auc: 0.653361
[140]	training's auc: 0.774576	valid_1's auc: 0.653835
[150]	training's auc: 0.775998	valid_1's auc: 0.654387
[160]	training's auc: 0.777431	valid_1's auc: 0.654932
[170]	training's auc: 0.778542	valid_1's auc: 0.655609
[180]	training's auc: 0.779702	valid_1's auc: 0.656016
[190]	training's auc: 0.780746	valid_1's auc: 0.65637
[200]	training's auc: 0.781709	valid_1's auc: 0.656743
[210]	training's auc: 0.782836	valid_1's auc: 0.657013
[220]	training's auc: 0.783979	valid_1's auc: 0.657556
[230]	training's auc: 0.784892	valid_1's auc: 0.65789
[240]	training's auc: 0.785822	valid_1's auc: 0.658259
[250]	training's auc: 0.786742	valid_1's auc: 0.658482
[260]	training's auc: 0.787606	valid_1's auc: 0.658591
[270]	training's auc: 0.788544	valid_1's auc: 0.658913
[280]	training's auc: 0.789358	valid_1's auc: 0.659178
[290]	training's auc: 0.790193	valid_1's auc: 0.659508
[300]	training's auc: 0.791088	valid_1's auc: 0.659763
[310]	training's auc: 0.791814	valid_1's auc: 0.659908
[320]	training's auc: 0.792782	valid_1's auc: 0.660132
[330]	training's auc: 0.793515	valid_1's auc: 0.660371
[340]	training's auc: 0.794122	valid_1's auc: 0.660565
[350]	training's auc: 0.794746	valid_1's auc: 0.660805
[360]	training's auc: 0.795343	valid_1's auc: 0.66102
[370]	training's auc: 0.795948	valid_1's auc: 0.661184
[380]	training's auc: 0.796519	valid_1's auc: 0.661297
[390]	training's auc: 0.797149	valid_1's auc: 0.661429
[400]	training's auc: 0.797756	valid_1's auc: 0.661675
[410]	training's auc: 0.798234	valid_1's auc: 0.661798
[420]	training's auc: 0.799104	valid_1's auc: 0.662113
[430]	training's auc: 0.799555	valid_1's auc: 0.662233
[440]	training's auc: 0.800018	valid_1's auc: 0.662326
[450]	training's auc: 0.800482	valid_1's auc: 0.662397
[460]	training's auc: 0.8011	valid_1's auc: 0.662575
[470]	training's auc: 0.801936	valid_1's auc: 0.662889
[480]	training's auc: 0.802591	valid_1's auc: 0.663089
[490]	training's auc: 0.803163	valid_1's auc: 0.663252
[500]	training's auc: 0.803696	valid_1's auc: 0.663403
[510]	training's auc: 0.80412	valid_1's auc: 0.66349
[520]	training's auc: 0.804812	valid_1's auc: 0.663693
[530]	training's auc: 0.805479	valid_1's auc: 0.66388
[540]	training's auc: 0.806153	valid_1's auc: 0.664095
[550]	training's auc: 0.806614	valid_1's auc: 0.664216
[560]	training's auc: 0.80704	valid_1's auc: 0.664361
[570]	training's auc: 0.807524	valid_1's auc: 0.664455
[580]	training's auc: 0.807969	valid_1's auc: 0.664615
[590]	training's auc: 0.808389	valid_1's auc: 0.664741
[600]	training's auc: 0.808796	valid_1's auc: 0.664878
[610]	training's auc: 0.809104	valid_1's auc: 0.664982
[620]	training's auc: 0.809504	valid_1's auc: 0.665142
[630]	training's auc: 0.80999	valid_1's auc: 0.665212
[640]	training's auc: 0.810374	valid_1's auc: 0.66528
[650]	training's auc: 0.810809	valid_1's auc: 0.665381
[660]	training's auc: 0.811139	valid_1's auc: 0.665412
[670]	training's auc: 0.811679	valid_1's auc: 0.665572
[680]	training's auc: 0.812122	valid_1's auc: 0.665697
[690]	training's auc: 0.812554	valid_1's auc: 0.665843
[700]	training's auc: 0.812892	valid_1's auc: 0.665875
[710]	training's auc: 0.81338	valid_1's auc: 0.666017
[720]	training's auc: 0.81373	valid_1's auc: 0.666123
[730]	training's auc: 0.814004	valid_1's auc: 0.666184
[740]	training's auc: 0.81427	valid_1's auc: 0.666212
[750]	training's auc: 0.814684	valid_1's auc: 0.66632
[760]	training's auc: 0.815066	valid_1's auc: 0.666414
[770]	training's auc: 0.815344	valid_1's auc: 0.666473
[780]	training's auc: 0.815751	valid_1's auc: 0.66652
[790]	training's auc: 0.816064	valid_1's auc: 0.666577
[800]	training's auc: 0.816349	valid_1's auc: 0.666611
[810]	training's auc: 0.816698	valid_1's auc: 0.66673
[820]	training's auc: 0.817064	valid_1's auc: 0.666732
[830]	training's auc: 0.817411	valid_1's auc: 0.666801
[840]	training's auc: 0.817679	valid_1's auc: 0.666875
[850]	training's auc: 0.818038	valid_1's auc: 0.666962
[860]	training's auc: 0.818454	valid_1's auc: 0.66701
[870]	training's auc: 0.818697	valid_1's auc: 0.667039
[880]	training's auc: 0.818938	valid_1's auc: 0.667045
[890]	training's auc: 0.819278	valid_1's auc: 0.667099
[900]	training's auc: 0.819509	valid_1's auc: 0.667125
[910]	training's auc: 0.81981	valid_1's auc: 0.667139
[920]	training's auc: 0.82005	valid_1's auc: 0.667231
[930]	training's auc: 0.820437	valid_1's auc: 0.667322
[940]	training's auc: 0.820676	valid_1's auc: 0.667347
[950]	training's auc: 0.820985	valid_1's auc: 0.667411
[960]	training's auc: 0.821273	valid_1's auc: 0.667494
[970]	training's auc: 0.821519	valid_1's auc: 0.667533
[980]	training's auc: 0.821836	valid_1's auc: 0.667662
[990]	training's auc: 0.822074	valid_1's auc: 0.667663
[1000]	training's auc: 0.822333	valid_1's auc: 0.667703
[1010]	training's auc: 0.822533	valid_1's auc: 0.667703
[1020]	training's auc: 0.822776	valid_1's auc: 0.667718
[1030]	training's auc: 0.823068	valid_1's auc: 0.667766
[1040]	training's auc: 0.823415	valid_1's auc: 0.66787
[1050]	training's auc: 0.823701	valid_1's auc: 0.667948
[1060]	training's auc: 0.823938	valid_1's auc: 0.667996
[1070]	training's auc: 0.824105	valid_1's auc: 0.668031
[1080]	training's auc: 0.824442	valid_1's auc: 0.668075
[1090]	training's auc: 0.824639	valid_1's auc: 0.668094
[1100]	training's auc: 0.824865	valid_1's auc: 0.668106
[1110]	training's auc: 0.825069	valid_1's auc: 0.668116
[1120]	training's auc: 0.82528	valid_1's auc: 0.66815
[1130]	training's auc: 0.825479	valid_1's auc: 0.668184
[1140]	training's auc: 0.825683	valid_1's auc: 0.668232
[1150]	training's auc: 0.825864	valid_1's auc: 0.668227
[1160]	training's auc: 0.82604	valid_1's auc: 0.668223
[1170]	training's auc: 0.826288	valid_1's auc: 0.66825
[1180]	training's auc: 0.826584	valid_1's auc: 0.668387
[1190]	training's auc: 0.826721	valid_1's auc: 0.668411
[1200]	training's auc: 0.82697	valid_1's auc: 0.668438
[1210]	training's auc: 0.82715	valid_1's auc: 0.668436
[1220]	training's auc: 0.827348	valid_1's auc: 0.668458
[1230]	training's auc: 0.827649	valid_1's auc: 0.668538
[1240]	training's auc: 0.827846	valid_1's auc: 0.668554
[1250]	training's auc: 0.828026	valid_1's auc: 0.668569
[1260]	training's auc: 0.828194	valid_1's auc: 0.668629
[1270]	training's auc: 0.828367	valid_1's auc: 0.668632
[1280]	training's auc: 0.828535	valid_1's auc: 0.668643
[1290]	training's auc: 0.828718	valid_1's auc: 0.668666
[1300]	training's auc: 0.828893	valid_1's auc: 0.668663
[1310]	training's auc: 0.829011	valid_1's auc: 0.668678
[1320]	training's auc: 0.829185	valid_1's auc: 0.668721
[1330]	training's auc: 0.829328	valid_1's auc: 0.66874
[1340]	training's auc: 0.829483	valid_1's auc: 0.66877
[1350]	training's auc: 0.829615	valid_1's auc: 0.668789
[1360]	training's auc: 0.829765	valid_1's auc: 0.668801
[1370]	training's auc: 0.829983	valid_1's auc: 0.668849
[1380]	training's auc: 0.83019	valid_1's auc: 0.668889
[1390]	training's auc: 0.830338	valid_1's auc: 0.668903
[1400]	training's auc: 0.830488	valid_1's auc: 0.668945
[1410]	training's auc: 0.830612	valid_1's auc: 0.668957
[1420]	training's auc: 0.830752	valid_1's auc: 0.668971
[1430]	training's auc: 0.830901	valid_1's auc: 0.669034
[1440]	training's auc: 0.831045	valid_1's auc: 0.669044
[1450]	training's auc: 0.831185	valid_1's auc: 0.669065
[1460]	training's auc: 0.831369	valid_1's auc: 0.669109
[1470]	training's auc: 0.831517	valid_1's auc: 0.669118
[1480]	training's auc: 0.831661	valid_1's auc: 0.669128
[1490]	training's auc: 0.831795	valid_1's auc: 0.669125
[1500]	training's auc: 0.831954	valid_1's auc: 0.669146
[1510]	training's auc: 0.832077	valid_1's auc: 0.66917
[1520]	training's auc: 0.832184	valid_1's auc: 0.669189
[1530]	training's auc: 0.832312	valid_1's auc: 0.669214
[1540]	training's auc: 0.832447	valid_1's auc: 0.669227
[1550]	training's auc: 0.832572	valid_1's auc: 0.66923
[1560]	training's auc: 0.832686	valid_1's auc: 0.669252
[1570]	training's auc: 0.832778	valid_1's auc: 0.669261
[1580]	training's auc: 0.832904	valid_1's auc: 0.669252
[1590]	training's auc: 0.833014	valid_1's auc: 0.669261
[1600]	training's auc: 0.833147	valid_1's auc: 0.669273
[1610]	training's auc: 0.833406	valid_1's auc: 0.669399
[1620]	training's auc: 0.83353	valid_1's auc: 0.669404
[1630]	training's auc: 0.833663	valid_1's auc: 0.66944
[1640]	training's auc: 0.833782	valid_1's auc: 0.669449
[1650]	training's auc: 0.833915	valid_1's auc: 0.669459
[1660]	training's auc: 0.834025	valid_1's auc: 0.669464
[1670]	training's auc: 0.834148	valid_1's auc: 0.669463
[1680]	training's auc: 0.834263	valid_1's auc: 0.669467
[1690]	training's auc: 0.834382	valid_1's auc: 0.669484
[1700]	training's auc: 0.834508	valid_1's auc: 0.669482
[1710]	training's auc: 0.834621	valid_1's auc: 0.669517
[1720]	training's auc: 0.834781	valid_1's auc: 0.669547
[1730]	training's auc: 0.83489	valid_1's auc: 0.669579
[1740]	training's auc: 0.834975	valid_1's auc: 0.669594
[1750]	training's auc: 0.835101	valid_1's auc: 0.669564
[1760]	training's auc: 0.835214	valid_1's auc: 0.66955
[1770]	training's auc: 0.835319	valid_1's auc: 0.669552
[1780]	training's auc: 0.835431	valid_1's auc: 0.669547
Early stopping, best iteration is:
[1737]	training's auc: 0.834953	valid_1's auc: 0.669599
best score: 0.669598843171
best iteration: 1737
complete on: CC11_language

working on: ITC_name

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
ITC_name                 int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.736457	valid_1's auc: 0.643944
[20]	training's auc: 0.747487	valid_1's auc: 0.648303
[30]	training's auc: 0.754509	valid_1's auc: 0.651461
[40]	training's auc: 0.758396	valid_1's auc: 0.653212
[50]	training's auc: 0.761459	valid_1's auc: 0.654772
[60]	training's auc: 0.764226	valid_1's auc: 0.65599
[70]	training's auc: 0.767131	valid_1's auc: 0.657106
[80]	training's auc: 0.769306	valid_1's auc: 0.658026
[90]	training's auc: 0.771517	valid_1's auc: 0.658919
[100]	training's auc: 0.773593	valid_1's auc: 0.659669
[110]	training's auc: 0.775368	valid_1's auc: 0.660409
[120]	training's auc: 0.776984	valid_1's auc: 0.661103
[130]	training's auc: 0.778428	valid_1's auc: 0.661639
[140]	training's auc: 0.779932	valid_1's auc: 0.662166
[150]	training's auc: 0.781359	valid_1's auc: 0.662619
[160]	training's auc: 0.78264	valid_1's auc: 0.663061
[170]	training's auc: 0.783942	valid_1's auc: 0.663537
[180]	training's auc: 0.785158	valid_1's auc: 0.663945
[190]	training's auc: 0.786247	valid_1's auc: 0.664394
[200]	training's auc: 0.787281	valid_1's auc: 0.664649
[210]	training's auc: 0.788376	valid_1's auc: 0.664937
[220]	training's auc: 0.789275	valid_1's auc: 0.665171
[230]	training's auc: 0.790232	valid_1's auc: 0.66549
[240]	training's auc: 0.791206	valid_1's auc: 0.66577
[250]	training's auc: 0.792187	valid_1's auc: 0.666032
[260]	training's auc: 0.79301	valid_1's auc: 0.666229
[270]	training's auc: 0.793874	valid_1's auc: 0.66644
[280]	training's auc: 0.794765	valid_1's auc: 0.666772
[290]	training's auc: 0.795832	valid_1's auc: 0.667053
[300]	training's auc: 0.796606	valid_1's auc: 0.667207
[310]	training's auc: 0.797337	valid_1's auc: 0.667367
[320]	training's auc: 0.798073	valid_1's auc: 0.667495
[330]	training's auc: 0.798722	valid_1's auc: 0.667605
[340]	training's auc: 0.799367	valid_1's auc: 0.667716
[350]	training's auc: 0.799944	valid_1's auc: 0.667865
[360]	training's auc: 0.800795	valid_1's auc: 0.668135
[370]	training's auc: 0.801358	valid_1's auc: 0.668208
[380]	training's auc: 0.801953	valid_1's auc: 0.668398
[390]	training's auc: 0.802643	valid_1's auc: 0.668566
[400]	training's auc: 0.803159	valid_1's auc: 0.668786
[410]	training's auc: 0.803698	valid_1's auc: 0.668874
[420]	training's auc: 0.804204	valid_1's auc: 0.669009
[430]	training's auc: 0.804742	valid_1's auc: 0.669095
[440]	training's auc: 0.805225	valid_1's auc: 0.66917
[450]	training's auc: 0.805742	valid_1's auc: 0.669212
[460]	training's auc: 0.806227	valid_1's auc: 0.669248
[470]	training's auc: 0.806968	valid_1's auc: 0.669449
[480]	training's auc: 0.807442	valid_1's auc: 0.669541
[490]	training's auc: 0.80794	valid_1's auc: 0.669645
[500]	training's auc: 0.808711	valid_1's auc: 0.669881
[510]	training's auc: 0.809326	valid_1's auc: 0.670006
[520]	training's auc: 0.80972	valid_1's auc: 0.670104
[530]	training's auc: 0.810125	valid_1's auc: 0.670137
[540]	training's auc: 0.810554	valid_1's auc: 0.670275
[550]	training's auc: 0.810998	valid_1's auc: 0.670326
[560]	training's auc: 0.811518	valid_1's auc: 0.670451
[570]	training's auc: 0.811958	valid_1's auc: 0.67053
[580]	training's auc: 0.812278	valid_1's auc: 0.670556
[590]	training's auc: 0.812626	valid_1's auc: 0.670619
[600]	training's auc: 0.813091	valid_1's auc: 0.67076
[610]	training's auc: 0.8134	valid_1's auc: 0.670815
[620]	training's auc: 0.813765	valid_1's auc: 0.670827
[630]	training's auc: 0.814124	valid_1's auc: 0.670871
[640]	training's auc: 0.814475	valid_1's auc: 0.670889
[650]	training's auc: 0.814778	valid_1's auc: 0.670914
[660]	training's auc: 0.815238	valid_1's auc: 0.671001
[670]	training's auc: 0.815592	valid_1's auc: 0.671061
[680]	training's auc: 0.815919	valid_1's auc: 0.671114
[690]	training's auc: 0.816387	valid_1's auc: 0.671207
[700]	training's auc: 0.816724	valid_1's auc: 0.671235
[710]	training's auc: 0.817052	valid_1's auc: 0.671269
[720]	training's auc: 0.817329	valid_1's auc: 0.671337
[730]	training's auc: 0.817576	valid_1's auc: 0.671341
[740]	training's auc: 0.817896	valid_1's auc: 0.67138
[750]	training's auc: 0.818183	valid_1's auc: 0.671421
[760]	training's auc: 0.818528	valid_1's auc: 0.671483
[770]	training's auc: 0.818892	valid_1's auc: 0.671521
[780]	training's auc: 0.819212	valid_1's auc: 0.671521
[790]	training's auc: 0.819487	valid_1's auc: 0.671542
[800]	training's auc: 0.819761	valid_1's auc: 0.671594
[810]	training's auc: 0.820108	valid_1's auc: 0.671596
[820]	training's auc: 0.820587	valid_1's auc: 0.671762
[830]	training's auc: 0.820867	valid_1's auc: 0.671763
[840]	training's auc: 0.821224	valid_1's auc: 0.671779
[850]	training's auc: 0.821511	valid_1's auc: 0.671842
[860]	training's auc: 0.821988	valid_1's auc: 0.671898
[870]	training's auc: 0.822411	valid_1's auc: 0.671951
[880]	training's auc: 0.822743	valid_1's auc: 0.672012
[890]	training's auc: 0.823048	valid_1's auc: 0.67204
[900]	training's auc: 0.823335	valid_1's auc: 0.672128
[910]	training's auc: 0.823628	valid_1's auc: 0.672169
[920]	training's auc: 0.823828	valid_1's auc: 0.672157
[930]	training's auc: 0.824087	valid_1's auc: 0.672172
[940]	training's auc: 0.824351	valid_1's auc: 0.672233
[950]	training's auc: 0.824623	valid_1's auc: 0.672252
[960]	training's auc: 0.824859	valid_1's auc: 0.672253
[970]	training's auc: 0.825148	valid_1's auc: 0.672324
[980]	training's auc: 0.825643	valid_1's auc: 0.672392
[990]	training's auc: 0.825837	valid_1's auc: 0.67239
[1000]	training's auc: 0.826071	valid_1's auc: 0.672427
[1010]	training's auc: 0.826323	valid_1's auc: 0.672448
[1020]	training's auc: 0.826587	valid_1's auc: 0.672436
[1030]	training's auc: 0.826779	valid_1's auc: 0.672444
[1040]	training's auc: 0.826987	valid_1's auc: 0.672474
[1050]	training's auc: 0.82723	valid_1's auc: 0.672502
[1060]	training's auc: 0.827449	valid_1's auc: 0.672519
[1070]	training's auc: 0.827644	valid_1's auc: 0.672517
[1080]	training's auc: 0.82784	valid_1's auc: 0.672509
[1090]	training's auc: 0.828021	valid_1's auc: 0.672488
[1100]	training's auc: 0.828318	valid_1's auc: 0.672579
[1110]	training's auc: 0.828535	valid_1's auc: 0.672574
[1120]	training's auc: 0.828728	valid_1's auc: 0.672572
[1130]	training's auc: 0.828946	valid_1's auc: 0.672584
[1140]	training's auc: 0.829123	valid_1's auc: 0.672588
[1150]	training's auc: 0.829387	valid_1's auc: 0.672601
[1160]	training's auc: 0.829579	valid_1's auc: 0.67259
[1170]	training's auc: 0.82979	valid_1's auc: 0.672596
[1180]	training's auc: 0.829973	valid_1's auc: 0.67263
[1190]	training's auc: 0.830138	valid_1's auc: 0.672609
[1200]	training's auc: 0.830355	valid_1's auc: 0.672636
[1210]	training's auc: 0.830549	valid_1's auc: 0.672647
[1220]	training's auc: 0.830746	valid_1's auc: 0.672667
[1230]	training's auc: 0.830931	valid_1's auc: 0.672663
[1240]	training's auc: 0.831126	valid_1's auc: 0.672656
[1250]	training's auc: 0.83129	valid_1's auc: 0.672665
[1260]	training's auc: 0.831502	valid_1's auc: 0.67269
[1270]	training's auc: 0.831676	valid_1's auc: 0.67269
[1280]	training's auc: 0.831898	valid_1's auc: 0.672723
[1290]	training's auc: 0.832079	valid_1's auc: 0.672728
[1300]	training's auc: 0.832253	valid_1's auc: 0.672784
[1310]	training's auc: 0.832474	valid_1's auc: 0.672813
[1320]	training's auc: 0.832644	valid_1's auc: 0.672842
[1330]	training's auc: 0.832795	valid_1's auc: 0.672829
[1340]	training's auc: 0.832981	valid_1's auc: 0.672825
[1350]	training's auc: 0.833228	valid_1's auc: 0.672885
[1360]	training's auc: 0.83339	valid_1's auc: 0.672875
[1370]	training's auc: 0.833556	valid_1's auc: 0.672877
[1380]	training's auc: 0.833768	valid_1's auc: 0.672918
[1390]	training's auc: 0.83393	valid_1's auc: 0.672906
[1400]	training's auc: 0.834097	valid_1's auc: 0.672917
[1410]	training's auc: 0.834231	valid_1's auc: 0.672931
[1420]	training's auc: 0.8344	valid_1's auc: 0.672916
[1430]	training's auc: 0.834533	valid_1's auc: 0.67292
[1440]	training's auc: 0.8347	valid_1's auc: 0.67295
[1450]	training's auc: 0.834837	valid_1's auc: 0.672946
[1460]	training's auc: 0.834972	valid_1's auc: 0.672966
[1470]	training's auc: 0.83511	valid_1's auc: 0.672981
[1480]	training's auc: 0.835261	valid_1's auc: 0.673
[1490]	training's auc: 0.835419	valid_1's auc: 0.673026
[1500]	training's auc: 0.83556	valid_1's auc: 0.673026
[1510]	training's auc: 0.835675	valid_1's auc: 0.673024
[1520]	training's auc: 0.835774	valid_1's auc: 0.673055
[1530]	training's auc: 0.835934	valid_1's auc: 0.673059
[1540]	training's auc: 0.836057	valid_1's auc: 0.673066
[1550]	training's auc: 0.836185	valid_1's auc: 0.673048
[1560]	training's auc: 0.836327	valid_1's auc: 0.673055
[1570]	training's auc: 0.836441	valid_1's auc: 0.673055
[1580]	training's auc: 0.836563	valid_1's auc: 0.673055
Early stopping, best iteration is:
[1534]	training's auc: 0.835986	valid_1's auc: 0.673072
best score: 0.673071761185
best iteration: 1534
complete on: ITC_name

working on: CC11_name

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
CC11_name                int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.740231	valid_1's auc: 0.619514
[20]	training's auc: 0.750412	valid_1's auc: 0.622387
[30]	training's auc: 0.756832	valid_1's auc: 0.625179
[40]	training's auc: 0.76072	valid_1's auc: 0.626849
[50]	training's auc: 0.764013	valid_1's auc: 0.628363
[60]	training's auc: 0.767454	valid_1's auc: 0.629544
[70]	training's auc: 0.770104	valid_1's auc: 0.63065
[80]	training's auc: 0.772434	valid_1's auc: 0.631751
[90]	training's auc: 0.774582	valid_1's auc: 0.632531
[100]	training's auc: 0.776495	valid_1's auc: 0.633378
[110]	training's auc: 0.778358	valid_1's auc: 0.634096
[120]	training's auc: 0.779989	valid_1's auc: 0.634736
[130]	training's auc: 0.781795	valid_1's auc: 0.635291
[140]	training's auc: 0.783186	valid_1's auc: 0.635827
[150]	training's auc: 0.784698	valid_1's auc: 0.636491
[160]	training's auc: 0.786007	valid_1's auc: 0.636965
[170]	training's auc: 0.787201	valid_1's auc: 0.637487
[180]	training's auc: 0.788482	valid_1's auc: 0.637983
[190]	training's auc: 0.789632	valid_1's auc: 0.638331
[200]	training's auc: 0.790599	valid_1's auc: 0.638675
[210]	training's auc: 0.791664	valid_1's auc: 0.638995
[220]	training's auc: 0.792699	valid_1's auc: 0.639285
[230]	training's auc: 0.793591	valid_1's auc: 0.639518
[240]	training's auc: 0.794512	valid_1's auc: 0.639784
[250]	training's auc: 0.795508	valid_1's auc: 0.640006
[260]	training's auc: 0.796444	valid_1's auc: 0.640156
[270]	training's auc: 0.797343	valid_1's auc: 0.640388
[280]	training's auc: 0.798096	valid_1's auc: 0.640611
[290]	training's auc: 0.799037	valid_1's auc: 0.640962
[300]	training's auc: 0.799966	valid_1's auc: 0.641086
[310]	training's auc: 0.800769	valid_1's auc: 0.641268
[320]	training's auc: 0.80152	valid_1's auc: 0.641463
[330]	training's auc: 0.80222	valid_1's auc: 0.641714
[340]	training's auc: 0.802945	valid_1's auc: 0.641922
[350]	training's auc: 0.803578	valid_1's auc: 0.6421
[360]	training's auc: 0.804162	valid_1's auc: 0.642183
[370]	training's auc: 0.804675	valid_1's auc: 0.642317
[380]	training's auc: 0.805295	valid_1's auc: 0.642498
[390]	training's auc: 0.805875	valid_1's auc: 0.642629
[400]	training's auc: 0.806542	valid_1's auc: 0.642941
[410]	training's auc: 0.807257	valid_1's auc: 0.643005
[420]	training's auc: 0.807869	valid_1's auc: 0.643196
[430]	training's auc: 0.808361	valid_1's auc: 0.643248
[440]	training's auc: 0.808913	valid_1's auc: 0.64339
[450]	training's auc: 0.80954	valid_1's auc: 0.643558
[460]	training's auc: 0.81	valid_1's auc: 0.643599
[470]	training's auc: 0.810452	valid_1's auc: 0.643679
[480]	training's auc: 0.810922	valid_1's auc: 0.643762
[490]	training's auc: 0.811513	valid_1's auc: 0.643921
[500]	training's auc: 0.812021	valid_1's auc: 0.644029
[510]	training's auc: 0.812474	valid_1's auc: 0.644064
[520]	training's auc: 0.813133	valid_1's auc: 0.644245
[530]	training's auc: 0.813807	valid_1's auc: 0.644322
[540]	training's auc: 0.814199	valid_1's auc: 0.644367
[550]	training's auc: 0.814719	valid_1's auc: 0.64447
[560]	training's auc: 0.815099	valid_1's auc: 0.644537
[570]	training's auc: 0.815532	valid_1's auc: 0.644601
[580]	training's auc: 0.815898	valid_1's auc: 0.644922
[590]	training's auc: 0.816354	valid_1's auc: 0.645024
[600]	training's auc: 0.816919	valid_1's auc: 0.645129
[610]	training's auc: 0.817254	valid_1's auc: 0.645209
[620]	training's auc: 0.817581	valid_1's auc: 0.645216
[630]	training's auc: 0.817928	valid_1's auc: 0.645236
[640]	training's auc: 0.818363	valid_1's auc: 0.645372
[650]	training's auc: 0.818664	valid_1's auc: 0.645355
[660]	training's auc: 0.819052	valid_1's auc: 0.645413
[670]	training's auc: 0.819539	valid_1's auc: 0.645521
[680]	training's auc: 0.819842	valid_1's auc: 0.645585
[690]	training's auc: 0.820168	valid_1's auc: 0.645623
[700]	training's auc: 0.820626	valid_1's auc: 0.645709
[710]	training's auc: 0.821199	valid_1's auc: 0.645883
[720]	training's auc: 0.821549	valid_1's auc: 0.645948
[730]	training's auc: 0.821769	valid_1's auc: 0.645963
[740]	training's auc: 0.822077	valid_1's auc: 0.645949
[750]	training's auc: 0.822413	valid_1's auc: 0.646004
[760]	training's auc: 0.822828	valid_1's auc: 0.646045
[770]	training's auc: 0.823153	valid_1's auc: 0.646107
[780]	training's auc: 0.823562	valid_1's auc: 0.646178
[790]	training's auc: 0.823824	valid_1's auc: 0.64619
[800]	training's auc: 0.824145	valid_1's auc: 0.646193
[810]	training's auc: 0.824432	valid_1's auc: 0.646206
[820]	training's auc: 0.824677	valid_1's auc: 0.646222
[830]	training's auc: 0.824967	valid_1's auc: 0.646268
[840]	training's auc: 0.825344	valid_1's auc: 0.646327
[850]	training's auc: 0.825777	valid_1's auc: 0.646374
[860]	training's auc: 0.826075	valid_1's auc: 0.646409
[870]	training's auc: 0.826337	valid_1's auc: 0.646422
[880]	training's auc: 0.826601	valid_1's auc: 0.646438
[890]	training's auc: 0.826969	valid_1's auc: 0.646477
[900]	training's auc: 0.827207	valid_1's auc: 0.646514
[910]	training's auc: 0.827485	valid_1's auc: 0.646551
[920]	training's auc: 0.827728	valid_1's auc: 0.646544
[930]	training's auc: 0.828011	valid_1's auc: 0.646541
[940]	training's auc: 0.828301	valid_1's auc: 0.646566
[950]	training's auc: 0.828567	valid_1's auc: 0.646559
[960]	training's auc: 0.828794	valid_1's auc: 0.646595
[970]	training's auc: 0.829141	valid_1's auc: 0.646663
[980]	training's auc: 0.829369	valid_1's auc: 0.646654
[990]	training's auc: 0.829593	valid_1's auc: 0.646724
[1000]	training's auc: 0.829793	valid_1's auc: 0.646731
[1010]	training's auc: 0.830158	valid_1's auc: 0.646763
[1020]	training's auc: 0.830407	valid_1's auc: 0.646806
[1030]	training's auc: 0.830677	valid_1's auc: 0.646847
[1040]	training's auc: 0.830911	valid_1's auc: 0.646866
[1050]	training's auc: 0.831112	valid_1's auc: 0.646885
[1060]	training's auc: 0.831325	valid_1's auc: 0.646897
[1070]	training's auc: 0.831495	valid_1's auc: 0.646901
[1080]	training's auc: 0.831704	valid_1's auc: 0.646878
[1090]	training's auc: 0.831892	valid_1's auc: 0.646861
[1100]	training's auc: 0.832156	valid_1's auc: 0.646931
[1110]	training's auc: 0.832483	valid_1's auc: 0.646962
[1120]	training's auc: 0.832681	valid_1's auc: 0.646979
[1130]	training's auc: 0.832967	valid_1's auc: 0.647032
[1140]	training's auc: 0.833202	valid_1's auc: 0.647063
[1150]	training's auc: 0.83341	valid_1's auc: 0.647038
[1160]	training's auc: 0.833582	valid_1's auc: 0.647048
[1170]	training's auc: 0.833796	valid_1's auc: 0.647046
[1180]	training's auc: 0.833976	valid_1's auc: 0.647042
[1190]	training's auc: 0.834127	valid_1's auc: 0.647014
Early stopping, best iteration is:
[1147]	training's auc: 0.833346	valid_1's auc: 0.647069
best score: 0.647068616817
best iteration: 1147
complete on: CC11_name

working on: ITC_song_year

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
ITC_song_year            int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.720269	valid_1's auc: 0.629627
[20]	training's auc: 0.735624	valid_1's auc: 0.635812
[30]	training's auc: 0.743596	valid_1's auc: 0.639897
[40]	training's auc: 0.747933	valid_1's auc: 0.641948
[50]	training's auc: 0.751366	valid_1's auc: 0.64367
[60]	training's auc: 0.754485	valid_1's auc: 0.644748
[70]	training's auc: 0.757213	valid_1's auc: 0.645873
[80]	training's auc: 0.759451	valid_1's auc: 0.646878
[90]	training's auc: 0.761847	valid_1's auc: 0.647853
[100]	training's auc: 0.764107	valid_1's auc: 0.648313
[110]	training's auc: 0.766277	valid_1's auc: 0.649076
[120]	training's auc: 0.768042	valid_1's auc: 0.64984
[130]	training's auc: 0.770113	valid_1's auc: 0.650798
[140]	training's auc: 0.771674	valid_1's auc: 0.651452
[150]	training's auc: 0.773449	valid_1's auc: 0.65212
[160]	training's auc: 0.774748	valid_1's auc: 0.652563
[170]	training's auc: 0.776207	valid_1's auc: 0.653092
[180]	training's auc: 0.777339	valid_1's auc: 0.653665
[190]	training's auc: 0.778424	valid_1's auc: 0.654065
[200]	training's auc: 0.779446	valid_1's auc: 0.654503
[210]	training's auc: 0.780542	valid_1's auc: 0.654859
[220]	training's auc: 0.781585	valid_1's auc: 0.655264
[230]	training's auc: 0.78255	valid_1's auc: 0.655659
[240]	training's auc: 0.78348	valid_1's auc: 0.655973
[250]	training's auc: 0.784473	valid_1's auc: 0.65627
[260]	training's auc: 0.785428	valid_1's auc: 0.656532
[270]	training's auc: 0.786688	valid_1's auc: 0.656906
[280]	training's auc: 0.787453	valid_1's auc: 0.657125
[290]	training's auc: 0.788335	valid_1's auc: 0.657481
[300]	training's auc: 0.789171	valid_1's auc: 0.657744
[310]	training's auc: 0.789996	valid_1's auc: 0.658069
[320]	training's auc: 0.790771	valid_1's auc: 0.65832
[330]	training's auc: 0.791464	valid_1's auc: 0.65862
[340]	training's auc: 0.79207	valid_1's auc: 0.658826
[350]	training's auc: 0.792782	valid_1's auc: 0.659074
[360]	training's auc: 0.793342	valid_1's auc: 0.659222
[370]	training's auc: 0.79402	valid_1's auc: 0.659432
[380]	training's auc: 0.794786	valid_1's auc: 0.65974
[390]	training's auc: 0.795426	valid_1's auc: 0.659924
[400]	training's auc: 0.795883	valid_1's auc: 0.660096
[410]	training's auc: 0.796666	valid_1's auc: 0.660209
[420]	training's auc: 0.797199	valid_1's auc: 0.660379
[430]	training's auc: 0.79778	valid_1's auc: 0.660532
[440]	training's auc: 0.79824	valid_1's auc: 0.660647
[450]	training's auc: 0.798982	valid_1's auc: 0.660858
[460]	training's auc: 0.799462	valid_1's auc: 0.660993
[470]	training's auc: 0.799942	valid_1's auc: 0.661108
[480]	training's auc: 0.800446	valid_1's auc: 0.661306
[490]	training's auc: 0.800976	valid_1's auc: 0.661399
[500]	training's auc: 0.801515	valid_1's auc: 0.661505
[510]	training's auc: 0.801983	valid_1's auc: 0.661634
[520]	training's auc: 0.802552	valid_1's auc: 0.6617
[530]	training's auc: 0.803091	valid_1's auc: 0.661836
[540]	training's auc: 0.803945	valid_1's auc: 0.662184
[550]	training's auc: 0.804485	valid_1's auc: 0.662357
[560]	training's auc: 0.804901	valid_1's auc: 0.662473
[570]	training's auc: 0.805465	valid_1's auc: 0.662627
[580]	training's auc: 0.805888	valid_1's auc: 0.662722
[590]	training's auc: 0.806269	valid_1's auc: 0.662815
[600]	training's auc: 0.806706	valid_1's auc: 0.66296
[610]	training's auc: 0.807056	valid_1's auc: 0.663073
[620]	training's auc: 0.807607	valid_1's auc: 0.66315
[630]	training's auc: 0.80812	valid_1's auc: 0.663245
[640]	training's auc: 0.808541	valid_1's auc: 0.663295
[650]	training's auc: 0.808911	valid_1's auc: 0.663363
[660]	training's auc: 0.809323	valid_1's auc: 0.663509
[670]	training's auc: 0.809809	valid_1's auc: 0.663631
[680]	training's auc: 0.810159	valid_1's auc: 0.66371
[690]	training's auc: 0.810459	valid_1's auc: 0.663738
[700]	training's auc: 0.81091	valid_1's auc: 0.663812
[710]	training's auc: 0.811427	valid_1's auc: 0.663948
[720]	training's auc: 0.811729	valid_1's auc: 0.663989
[730]	training's auc: 0.812053	valid_1's auc: 0.664065
[740]	training's auc: 0.812442	valid_1's auc: 0.664164
[750]	training's auc: 0.812717	valid_1's auc: 0.664231
[760]	training's auc: 0.813063	valid_1's auc: 0.664297
[770]	training's auc: 0.813368	valid_1's auc: 0.664342
[780]	training's auc: 0.81367	valid_1's auc: 0.664395
[790]	training's auc: 0.814023	valid_1's auc: 0.664472
[800]	training's auc: 0.814348	valid_1's auc: 0.664503
[810]	training's auc: 0.81477	valid_1's auc: 0.664579
[820]	training's auc: 0.815016	valid_1's auc: 0.664582
[830]	training's auc: 0.81542	valid_1's auc: 0.664696
[840]	training's auc: 0.815818	valid_1's auc: 0.66478
[850]	training's auc: 0.816112	valid_1's auc: 0.664854
[860]	training's auc: 0.816522	valid_1's auc: 0.664881
[870]	training's auc: 0.816798	valid_1's auc: 0.664921
[880]	training's auc: 0.817124	valid_1's auc: 0.664991
[890]	training's auc: 0.817483	valid_1's auc: 0.665036
[900]	training's auc: 0.817756	valid_1's auc: 0.664988
[910]	training's auc: 0.818056	valid_1's auc: 0.665038
[920]	training's auc: 0.818343	valid_1's auc: 0.665078
[930]	training's auc: 0.818628	valid_1's auc: 0.665106
[940]	training's auc: 0.818887	valid_1's auc: 0.665146
[950]	training's auc: 0.819184	valid_1's auc: 0.665201
[960]	training's auc: 0.819439	valid_1's auc: 0.665203
[970]	training's auc: 0.819751	valid_1's auc: 0.665262
[980]	training's auc: 0.820061	valid_1's auc: 0.665303
[990]	training's auc: 0.82043	valid_1's auc: 0.665397
[1000]	training's auc: 0.820633	valid_1's auc: 0.665397
[1010]	training's auc: 0.82089	valid_1's auc: 0.665423
[1020]	training's auc: 0.821092	valid_1's auc: 0.66546
[1030]	training's auc: 0.821415	valid_1's auc: 0.665501
[1040]	training's auc: 0.821683	valid_1's auc: 0.665558
[1050]	training's auc: 0.821989	valid_1's auc: 0.665601
[1060]	training's auc: 0.822229	valid_1's auc: 0.665645
[1070]	training's auc: 0.822424	valid_1's auc: 0.665647
[1080]	training's auc: 0.822675	valid_1's auc: 0.665654
[1090]	training's auc: 0.822869	valid_1's auc: 0.665637
[1100]	training's auc: 0.823079	valid_1's auc: 0.66563
[1110]	training's auc: 0.823313	valid_1's auc: 0.665658
[1120]	training's auc: 0.823522	valid_1's auc: 0.665725
[1130]	training's auc: 0.823809	valid_1's auc: 0.665782
[1140]	training's auc: 0.82405	valid_1's auc: 0.66582
[1150]	training's auc: 0.824369	valid_1's auc: 0.665913
[1160]	training's auc: 0.824676	valid_1's auc: 0.665968
[1170]	training's auc: 0.824876	valid_1's auc: 0.665974
[1180]	training's auc: 0.825054	valid_1's auc: 0.665998
[1190]	training's auc: 0.825242	valid_1's auc: 0.666019
[1200]	training's auc: 0.825445	valid_1's auc: 0.66601
[1210]	training's auc: 0.825626	valid_1's auc: 0.66604
[1220]	training's auc: 0.825791	valid_1's auc: 0.666048
[1230]	training's auc: 0.82606	valid_1's auc: 0.666081
[1240]	training's auc: 0.826383	valid_1's auc: 0.666204
[1250]	training's auc: 0.826563	valid_1's auc: 0.66623
[1260]	training's auc: 0.826753	valid_1's auc: 0.666276
[1270]	training's auc: 0.826941	valid_1's auc: 0.666308
[1280]	training's auc: 0.827133	valid_1's auc: 0.666276
[1290]	training's auc: 0.827348	valid_1's auc: 0.6663
[1300]	training's auc: 0.827486	valid_1's auc: 0.666302
[1310]	training's auc: 0.827668	valid_1's auc: 0.666331
[1320]	training's auc: 0.827891	valid_1's auc: 0.666417
[1330]	training's auc: 0.828109	valid_1's auc: 0.666469
[1340]	training's auc: 0.828356	valid_1's auc: 0.666488
[1350]	training's auc: 0.828534	valid_1's auc: 0.666505
[1360]	training's auc: 0.828688	valid_1's auc: 0.666485
[1370]	training's auc: 0.82887	valid_1's auc: 0.666509
[1380]	training's auc: 0.829008	valid_1's auc: 0.666539
[1390]	training's auc: 0.829154	valid_1's auc: 0.666572
[1400]	training's auc: 0.829324	valid_1's auc: 0.666543
[1410]	training's auc: 0.829494	valid_1's auc: 0.66658
[1420]	training's auc: 0.829649	valid_1's auc: 0.666583
[1430]	training's auc: 0.829775	valid_1's auc: 0.666588
[1440]	training's auc: 0.829962	valid_1's auc: 0.666657
[1450]	training's auc: 0.830087	valid_1's auc: 0.666683
[1460]	training's auc: 0.830217	valid_1's auc: 0.666713
[1470]	training's auc: 0.830333	valid_1's auc: 0.666707
[1480]	training's auc: 0.830492	valid_1's auc: 0.666761
[1490]	training's auc: 0.83062	valid_1's auc: 0.666784
[1500]	training's auc: 0.830762	valid_1's auc: 0.66678
[1510]	training's auc: 0.83088	valid_1's auc: 0.666772
[1520]	training's auc: 0.831011	valid_1's auc: 0.666778
[1530]	training's auc: 0.83116	valid_1's auc: 0.666775
[1540]	training's auc: 0.831285	valid_1's auc: 0.666804
[1550]	training's auc: 0.831392	valid_1's auc: 0.666808
[1560]	training's auc: 0.831531	valid_1's auc: 0.666825
[1570]	training's auc: 0.831647	valid_1's auc: 0.666838
[1580]	training's auc: 0.831778	valid_1's auc: 0.666808
[1590]	training's auc: 0.831894	valid_1's auc: 0.66684
[1600]	training's auc: 0.832031	valid_1's auc: 0.666832
[1610]	training's auc: 0.832137	valid_1's auc: 0.666821
[1620]	training's auc: 0.832263	valid_1's auc: 0.666833
[1630]	training's auc: 0.832376	valid_1's auc: 0.666863
[1640]	training's auc: 0.832482	valid_1's auc: 0.666875
[1650]	training's auc: 0.832637	valid_1's auc: 0.666906
[1660]	training's auc: 0.832763	valid_1's auc: 0.666915
[1670]	training's auc: 0.832891	valid_1's auc: 0.666887
[1680]	training's auc: 0.832994	valid_1's auc: 0.66688
[1690]	training's auc: 0.833088	valid_1's auc: 0.666858
[1700]	training's auc: 0.833217	valid_1's auc: 0.666834
Early stopping, best iteration is:
[1653]	training's auc: 0.832669	valid_1's auc: 0.666918
best score: 0.666917635627
best iteration: 1653
complete on: ITC_song_year

working on: CC11_song_year

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
CC11_song_year           int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.721157	valid_1's auc: 0.629353
[20]	training's auc: 0.736036	valid_1's auc: 0.636669
[30]	training's auc: 0.743575	valid_1's auc: 0.640672
[40]	training's auc: 0.747619	valid_1's auc: 0.642729
[50]	training's auc: 0.751519	valid_1's auc: 0.644411
[60]	training's auc: 0.754728	valid_1's auc: 0.645776
[70]	training's auc: 0.757487	valid_1's auc: 0.646865
[80]	training's auc: 0.759783	valid_1's auc: 0.647868
[90]	training's auc: 0.76211	valid_1's auc: 0.648783
[100]	training's auc: 0.764205	valid_1's auc: 0.649748
[110]	training's auc: 0.76622	valid_1's auc: 0.650436
[120]	training's auc: 0.768109	valid_1's auc: 0.650897
[130]	training's auc: 0.770363	valid_1's auc: 0.651789
[140]	training's auc: 0.771812	valid_1's auc: 0.652477
[150]	training's auc: 0.773209	valid_1's auc: 0.653218
[160]	training's auc: 0.774978	valid_1's auc: 0.653918
[170]	training's auc: 0.776214	valid_1's auc: 0.654348
[180]	training's auc: 0.777391	valid_1's auc: 0.6549
[190]	training's auc: 0.778794	valid_1's auc: 0.65541
[200]	training's auc: 0.779848	valid_1's auc: 0.655811
[210]	training's auc: 0.780836	valid_1's auc: 0.656222
[220]	training's auc: 0.781967	valid_1's auc: 0.656614
[230]	training's auc: 0.782926	valid_1's auc: 0.656962
[240]	training's auc: 0.783818	valid_1's auc: 0.657298
[250]	training's auc: 0.784812	valid_1's auc: 0.657571
[260]	training's auc: 0.786083	valid_1's auc: 0.657864
[270]	training's auc: 0.786902	valid_1's auc: 0.658118
[280]	training's auc: 0.78776	valid_1's auc: 0.658357
[290]	training's auc: 0.788595	valid_1's auc: 0.658771
[300]	training's auc: 0.789369	valid_1's auc: 0.658969
[310]	training's auc: 0.790071	valid_1's auc: 0.659252
[320]	training's auc: 0.791333	valid_1's auc: 0.659767
[330]	training's auc: 0.791969	valid_1's auc: 0.65991
[340]	training's auc: 0.792559	valid_1's auc: 0.660072
[350]	training's auc: 0.793136	valid_1's auc: 0.660217
[360]	training's auc: 0.793836	valid_1's auc: 0.660301
[370]	training's auc: 0.79463	valid_1's auc: 0.660564
[380]	training's auc: 0.79525	valid_1's auc: 0.660792
[390]	training's auc: 0.795933	valid_1's auc: 0.661211
[400]	training's auc: 0.796455	valid_1's auc: 0.661435
[410]	training's auc: 0.797123	valid_1's auc: 0.661579
[420]	training's auc: 0.797642	valid_1's auc: 0.661691
[430]	training's auc: 0.798179	valid_1's auc: 0.661872
[440]	training's auc: 0.798759	valid_1's auc: 0.661893
[450]	training's auc: 0.799415	valid_1's auc: 0.662024
[460]	training's auc: 0.799918	valid_1's auc: 0.66213
[470]	training's auc: 0.800421	valid_1's auc: 0.662292
[480]	training's auc: 0.801175	valid_1's auc: 0.662453
[490]	training's auc: 0.801839	valid_1's auc: 0.662638
[500]	training's auc: 0.802306	valid_1's auc: 0.662739
[510]	training's auc: 0.802878	valid_1's auc: 0.662868
[520]	training's auc: 0.803302	valid_1's auc: 0.662973
[530]	training's auc: 0.803759	valid_1's auc: 0.663081
[540]	training's auc: 0.804182	valid_1's auc: 0.663233
[550]	training's auc: 0.804716	valid_1's auc: 0.663306
[560]	training's auc: 0.805143	valid_1's auc: 0.663405
[570]	training's auc: 0.805622	valid_1's auc: 0.663566
[580]	training's auc: 0.806043	valid_1's auc: 0.663741
[590]	training's auc: 0.806428	valid_1's auc: 0.663849
[600]	training's auc: 0.806844	valid_1's auc: 0.663924
[610]	training's auc: 0.807247	valid_1's auc: 0.663918
[620]	training's auc: 0.807944	valid_1's auc: 0.66417
[630]	training's auc: 0.808335	valid_1's auc: 0.664228
[640]	training's auc: 0.808695	valid_1's auc: 0.664267
[650]	training's auc: 0.809108	valid_1's auc: 0.664372
[660]	training's auc: 0.809457	valid_1's auc: 0.664434
[670]	training's auc: 0.809961	valid_1's auc: 0.664621
[680]	training's auc: 0.810325	valid_1's auc: 0.664741
[690]	training's auc: 0.810608	valid_1's auc: 0.664818
[700]	training's auc: 0.811105	valid_1's auc: 0.664917
[710]	training's auc: 0.811479	valid_1's auc: 0.665021
[720]	training's auc: 0.81182	valid_1's auc: 0.665114
[730]	training's auc: 0.812152	valid_1's auc: 0.665139
[740]	training's auc: 0.812599	valid_1's auc: 0.665229
[750]	training's auc: 0.812883	valid_1's auc: 0.665285
[760]	training's auc: 0.813303	valid_1's auc: 0.665314
[770]	training's auc: 0.813618	valid_1's auc: 0.665372
[780]	training's auc: 0.813975	valid_1's auc: 0.665437
[790]	training's auc: 0.81436	valid_1's auc: 0.665517
[800]	training's auc: 0.814701	valid_1's auc: 0.665584
[810]	training's auc: 0.815005	valid_1's auc: 0.665646
[820]	training's auc: 0.815459	valid_1's auc: 0.665695
[830]	training's auc: 0.815757	valid_1's auc: 0.665704
[840]	training's auc: 0.816142	valid_1's auc: 0.665759
[850]	training's auc: 0.816603	valid_1's auc: 0.665847
[860]	training's auc: 0.816934	valid_1's auc: 0.66586
[870]	training's auc: 0.817223	valid_1's auc: 0.665871
[880]	training's auc: 0.817644	valid_1's auc: 0.66599
[890]	training's auc: 0.817987	valid_1's auc: 0.666032
[900]	training's auc: 0.818251	valid_1's auc: 0.666039
[910]	training's auc: 0.818542	valid_1's auc: 0.666098
[920]	training's auc: 0.818794	valid_1's auc: 0.666144
[930]	training's auc: 0.81908	valid_1's auc: 0.66618
[940]	training's auc: 0.819604	valid_1's auc: 0.666309
[950]	training's auc: 0.819906	valid_1's auc: 0.666342
[960]	training's auc: 0.820174	valid_1's auc: 0.666385
[970]	training's auc: 0.820438	valid_1's auc: 0.666395
[980]	training's auc: 0.820856	valid_1's auc: 0.666497
[990]	training's auc: 0.821153	valid_1's auc: 0.666538
[1000]	training's auc: 0.821474	valid_1's auc: 0.666605
[1010]	training's auc: 0.821728	valid_1's auc: 0.666598
[1020]	training's auc: 0.82199	valid_1's auc: 0.666624
[1030]	training's auc: 0.822193	valid_1's auc: 0.666666
[1040]	training's auc: 0.822422	valid_1's auc: 0.666691
[1050]	training's auc: 0.822709	valid_1's auc: 0.666737
[1060]	training's auc: 0.822945	valid_1's auc: 0.666764
[1070]	training's auc: 0.823113	valid_1's auc: 0.666797
[1080]	training's auc: 0.823312	valid_1's auc: 0.666833
[1090]	training's auc: 0.823649	valid_1's auc: 0.666888
[1100]	training's auc: 0.823837	valid_1's auc: 0.666889
[1110]	training's auc: 0.824117	valid_1's auc: 0.666966
[1120]	training's auc: 0.824409	valid_1's auc: 0.667009
[1130]	training's auc: 0.824637	valid_1's auc: 0.667032
[1140]	training's auc: 0.824819	valid_1's auc: 0.667023
[1150]	training's auc: 0.825057	valid_1's auc: 0.667031
[1160]	training's auc: 0.825283	valid_1's auc: 0.667051
[1170]	training's auc: 0.825481	valid_1's auc: 0.667056
[1180]	training's auc: 0.825653	valid_1's auc: 0.667061
[1190]	training's auc: 0.825836	valid_1's auc: 0.66709
[1200]	training's auc: 0.826023	valid_1's auc: 0.66712
[1210]	training's auc: 0.826199	valid_1's auc: 0.667149
[1220]	training's auc: 0.82637	valid_1's auc: 0.667188
[1230]	training's auc: 0.826541	valid_1's auc: 0.66721
[1240]	training's auc: 0.826703	valid_1's auc: 0.667214
[1250]	training's auc: 0.826946	valid_1's auc: 0.667269
[1260]	training's auc: 0.827102	valid_1's auc: 0.66728
[1270]	training's auc: 0.827274	valid_1's auc: 0.667304
[1280]	training's auc: 0.827451	valid_1's auc: 0.667309
[1290]	training's auc: 0.827634	valid_1's auc: 0.667303
[1300]	training's auc: 0.827799	valid_1's auc: 0.667322
[1310]	training's auc: 0.828009	valid_1's auc: 0.667346
[1320]	training's auc: 0.828179	valid_1's auc: 0.667367
[1330]	training's auc: 0.828353	valid_1's auc: 0.667392
[1340]	training's auc: 0.828568	valid_1's auc: 0.667372
[1350]	training's auc: 0.828722	valid_1's auc: 0.667383
[1360]	training's auc: 0.828878	valid_1's auc: 0.667412
[1370]	training's auc: 0.829056	valid_1's auc: 0.667422
[1380]	training's auc: 0.829199	valid_1's auc: 0.667428
[1390]	training's auc: 0.829347	valid_1's auc: 0.667435
[1400]	training's auc: 0.829494	valid_1's auc: 0.667452
[1410]	training's auc: 0.829642	valid_1's auc: 0.667479
[1420]	training's auc: 0.829874	valid_1's auc: 0.667564
[1430]	training's auc: 0.829986	valid_1's auc: 0.667545
[1440]	training's auc: 0.830136	valid_1's auc: 0.667526
[1450]	training's auc: 0.83027	valid_1's auc: 0.667535
[1460]	training's auc: 0.830391	valid_1's auc: 0.667565
[1470]	training's auc: 0.830512	valid_1's auc: 0.667588
[1480]	training's auc: 0.830643	valid_1's auc: 0.667587
[1490]	training's auc: 0.830811	valid_1's auc: 0.667637
[1500]	training's auc: 0.830988	valid_1's auc: 0.667643
[1510]	training's auc: 0.831128	valid_1's auc: 0.66766
[1520]	training's auc: 0.831266	valid_1's auc: 0.667659
[1530]	training's auc: 0.831395	valid_1's auc: 0.667672
[1540]	training's auc: 0.831514	valid_1's auc: 0.667674
[1550]	training's auc: 0.831639	valid_1's auc: 0.66769
[1560]	training's auc: 0.831837	valid_1's auc: 0.667723
[1570]	training's auc: 0.831933	valid_1's auc: 0.667747
[1580]	training's auc: 0.832067	valid_1's auc: 0.66771
[1590]	training's auc: 0.832188	valid_1's auc: 0.667719
[1600]	training's auc: 0.83235	valid_1's auc: 0.667752
[1610]	training's auc: 0.832484	valid_1's auc: 0.667747
[1620]	training's auc: 0.832621	valid_1's auc: 0.667752
[1630]	training's auc: 0.832772	valid_1's auc: 0.667764
[1640]	training's auc: 0.832875	valid_1's auc: 0.667777
[1650]	training's auc: 0.833006	valid_1's auc: 0.667809
[1660]	training's auc: 0.833103	valid_1's auc: 0.667814
[1670]	training's auc: 0.833225	valid_1's auc: 0.667844
[1680]	training's auc: 0.833327	valid_1's auc: 0.667876
[1690]	training's auc: 0.833422	valid_1's auc: 0.667884
[1700]	training's auc: 0.833534	valid_1's auc: 0.66789
[1710]	training's auc: 0.833644	valid_1's auc: 0.667925
[1720]	training's auc: 0.83375	valid_1's auc: 0.667952
[1730]	training's auc: 0.833876	valid_1's auc: 0.667944
[1740]	training's auc: 0.833979	valid_1's auc: 0.667949
[1750]	training's auc: 0.83412	valid_1's auc: 0.667958
[1760]	training's auc: 0.834211	valid_1's auc: 0.667949
[1770]	training's auc: 0.834328	valid_1's auc: 0.667965
[1780]	training's auc: 0.834443	valid_1's auc: 0.667965
[1790]	training's auc: 0.834545	valid_1's auc: 0.667967
[1800]	training's auc: 0.834665	valid_1's auc: 0.667978
[1810]	training's auc: 0.834777	valid_1's auc: 0.667981
[1820]	training's auc: 0.834897	valid_1's auc: 0.667982
[1830]	training's auc: 0.83502	valid_1's auc: 0.66799
[1840]	training's auc: 0.835121	valid_1's auc: 0.667976
[1850]	training's auc: 0.835245	valid_1's auc: 0.668036
[1860]	training's auc: 0.835354	valid_1's auc: 0.66802
[1870]	training's auc: 0.835468	valid_1's auc: 0.668028
[1880]	training's auc: 0.835594	valid_1's auc: 0.668066
[1890]	training's auc: 0.835698	valid_1's auc: 0.668066
[1900]	training's auc: 0.835794	valid_1's auc: 0.668073
[1910]	training's auc: 0.835898	valid_1's auc: 0.668097
[1920]	training's auc: 0.836008	valid_1's auc: 0.668112
[1930]	training's auc: 0.836125	valid_1's auc: 0.668144
[1940]	training's auc: 0.836241	valid_1's auc: 0.668141
[1950]	training's auc: 0.836349	valid_1's auc: 0.668128
[1960]	training's auc: 0.83645	valid_1's auc: 0.668123
[1970]	training's auc: 0.836532	valid_1's auc: 0.668147
[1980]	training's auc: 0.836633	valid_1's auc: 0.668146
[1990]	training's auc: 0.836725	valid_1's auc: 0.668167
[2000]	training's auc: 0.836814	valid_1's auc: 0.668155
[2010]	training's auc: 0.83693	valid_1's auc: 0.66818
[2020]	training's auc: 0.837029	valid_1's auc: 0.668179
[2030]	training's auc: 0.83713	valid_1's auc: 0.668206
[2040]	training's auc: 0.83724	valid_1's auc: 0.668208
[2050]	training's auc: 0.837345	valid_1's auc: 0.66825
[2060]	training's auc: 0.83743	valid_1's auc: 0.668222
[2070]	training's auc: 0.837539	valid_1's auc: 0.668245
[2080]	training's auc: 0.837624	valid_1's auc: 0.66825
[2090]	training's auc: 0.837738	valid_1's auc: 0.668243
[2100]	training's auc: 0.837841	valid_1's auc: 0.66826
[2110]	training's auc: 0.837943	valid_1's auc: 0.668271
[2120]	training's auc: 0.838038	valid_1's auc: 0.668287
[2130]	training's auc: 0.838122	valid_1's auc: 0.668278
[2140]	training's auc: 0.838214	valid_1's auc: 0.66828
[2150]	training's auc: 0.838311	valid_1's auc: 0.668287
[2160]	training's auc: 0.838415	valid_1's auc: 0.668312
[2170]	training's auc: 0.838519	valid_1's auc: 0.668302
[2180]	training's auc: 0.838621	valid_1's auc: 0.668307
[2190]	training's auc: 0.838731	valid_1's auc: 0.668337
[2200]	training's auc: 0.838847	valid_1's auc: 0.66835
[2210]	training's auc: 0.838931	valid_1's auc: 0.668338
[2220]	training's auc: 0.839037	valid_1's auc: 0.66833
[2230]	training's auc: 0.839145	valid_1's auc: 0.668351
[2240]	training's auc: 0.839228	valid_1's auc: 0.668366
[2250]	training's auc: 0.839314	valid_1's auc: 0.668384
[2260]	training's auc: 0.839412	valid_1's auc: 0.668398
[2270]	training's auc: 0.839504	valid_1's auc: 0.668406
[2280]	training's auc: 0.839589	valid_1's auc: 0.668428
[2290]	training's auc: 0.839685	valid_1's auc: 0.668411
[2300]	training's auc: 0.839769	valid_1's auc: 0.668441
[2310]	training's auc: 0.839866	valid_1's auc: 0.668453
[2320]	training's auc: 0.839961	valid_1's auc: 0.668434
[2330]	training's auc: 0.840061	valid_1's auc: 0.668448
[2340]	training's auc: 0.840159	valid_1's auc: 0.668474
[2350]	training's auc: 0.840251	valid_1's auc: 0.668476
[2360]	training's auc: 0.840338	valid_1's auc: 0.668484
[2370]	training's auc: 0.840421	valid_1's auc: 0.66848
[2380]	training's auc: 0.840503	valid_1's auc: 0.668524
[2390]	training's auc: 0.840581	valid_1's auc: 0.668524
[2400]	training's auc: 0.840679	valid_1's auc: 0.668534
[2410]	training's auc: 0.84075	valid_1's auc: 0.668552
[2420]	training's auc: 0.840836	valid_1's auc: 0.668561
[2430]	training's auc: 0.840933	valid_1's auc: 0.668567
[2440]	training's auc: 0.841018	valid_1's auc: 0.668555
[2450]	training's auc: 0.841107	valid_1's auc: 0.668581
[2460]	training's auc: 0.841185	valid_1's auc: 0.668571
[2470]	training's auc: 0.84128	valid_1's auc: 0.668566
[2480]	training's auc: 0.841364	valid_1's auc: 0.668582
[2490]	training's auc: 0.841472	valid_1's auc: 0.668592
[2500]	training's auc: 0.841552	valid_1's auc: 0.668589
[2510]	training's auc: 0.841619	valid_1's auc: 0.668606
[2520]	training's auc: 0.841702	valid_1's auc: 0.668591
[2530]	training's auc: 0.841789	valid_1's auc: 0.668604
[2540]	training's auc: 0.841875	valid_1's auc: 0.668585
[2550]	training's auc: 0.841965	valid_1's auc: 0.668596
[2560]	training's auc: 0.842051	valid_1's auc: 0.668611
[2570]	training's auc: 0.842118	valid_1's auc: 0.668622
[2580]	training's auc: 0.842199	valid_1's auc: 0.668608
[2590]	training's auc: 0.842289	valid_1's auc: 0.668623
[2600]	training's auc: 0.842389	valid_1's auc: 0.668646
[2610]	training's auc: 0.842455	valid_1's auc: 0.668658
[2620]	training's auc: 0.842544	valid_1's auc: 0.668654
[2630]	training's auc: 0.842625	valid_1's auc: 0.668677
[2640]	training's auc: 0.842713	valid_1's auc: 0.668673
[2650]	training's auc: 0.842806	valid_1's auc: 0.668667
[2660]	training's auc: 0.842885	valid_1's auc: 0.668661
[2670]	training's auc: 0.842964	valid_1's auc: 0.668657
Early stopping, best iteration is:
[2629]	training's auc: 0.842619	valid_1's auc: 0.668678
best score: 0.6686782975
best iteration: 2629
complete on: CC11_song_year

working on: ITC_song_country

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
ITC_song_country         int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.722258	valid_1's auc: 0.629747
[20]	training's auc: 0.737369	valid_1's auc: 0.637377
[30]	training's auc: 0.745771	valid_1's auc: 0.641717
[40]	training's auc: 0.750361	valid_1's auc: 0.643815
[50]	training's auc: 0.754287	valid_1's auc: 0.645562
[60]	training's auc: 0.75734	valid_1's auc: 0.64705
[70]	training's auc: 0.760113	valid_1's auc: 0.648369
[80]	training's auc: 0.76262	valid_1's auc: 0.64941
[90]	training's auc: 0.765228	valid_1's auc: 0.650352
[100]	training's auc: 0.767293	valid_1's auc: 0.651116
[110]	training's auc: 0.768982	valid_1's auc: 0.651696
[120]	training's auc: 0.770681	valid_1's auc: 0.652605
[130]	training's auc: 0.772116	valid_1's auc: 0.653108
[140]	training's auc: 0.773889	valid_1's auc: 0.653541
[150]	training's auc: 0.775176	valid_1's auc: 0.654044
[160]	training's auc: 0.776877	valid_1's auc: 0.654595
[170]	training's auc: 0.778384	valid_1's auc: 0.655111
[180]	training's auc: 0.779603	valid_1's auc: 0.655611
[190]	training's auc: 0.78067	valid_1's auc: 0.656047
[200]	training's auc: 0.7817	valid_1's auc: 0.656407
[210]	training's auc: 0.782758	valid_1's auc: 0.656757
[220]	training's auc: 0.783763	valid_1's auc: 0.657134
[230]	training's auc: 0.784714	valid_1's auc: 0.657554
[240]	training's auc: 0.785662	valid_1's auc: 0.657866
[250]	training's auc: 0.786648	valid_1's auc: 0.658158
[260]	training's auc: 0.787495	valid_1's auc: 0.658487
[270]	training's auc: 0.788222	valid_1's auc: 0.658655
[280]	training's auc: 0.789008	valid_1's auc: 0.658897
[290]	training's auc: 0.789814	valid_1's auc: 0.659146
[300]	training's auc: 0.790539	valid_1's auc: 0.659323
[310]	training's auc: 0.791299	valid_1's auc: 0.659597
[320]	training's auc: 0.792096	valid_1's auc: 0.659865
[330]	training's auc: 0.792733	valid_1's auc: 0.660105
[340]	training's auc: 0.793359	valid_1's auc: 0.660259
[350]	training's auc: 0.793954	valid_1's auc: 0.660461
[360]	training's auc: 0.794574	valid_1's auc: 0.660637
[370]	training's auc: 0.795201	valid_1's auc: 0.660844
[380]	training's auc: 0.795998	valid_1's auc: 0.661081
[390]	training's auc: 0.796538	valid_1's auc: 0.661225
[400]	training's auc: 0.797101	valid_1's auc: 0.661425
[410]	training's auc: 0.797723	valid_1's auc: 0.661607
[420]	training's auc: 0.798411	valid_1's auc: 0.661794
[430]	training's auc: 0.79892	valid_1's auc: 0.661942
[440]	training's auc: 0.799393	valid_1's auc: 0.662133
[450]	training's auc: 0.800005	valid_1's auc: 0.662265
[460]	training's auc: 0.800486	valid_1's auc: 0.662417
[470]	training's auc: 0.801089	valid_1's auc: 0.662603
[480]	training's auc: 0.801535	valid_1's auc: 0.662642
[490]	training's auc: 0.802093	valid_1's auc: 0.662772
[500]	training's auc: 0.80285	valid_1's auc: 0.663013
[510]	training's auc: 0.803272	valid_1's auc: 0.663079
[520]	training's auc: 0.803874	valid_1's auc: 0.663231
[530]	training's auc: 0.804322	valid_1's auc: 0.663331
[540]	training's auc: 0.804821	valid_1's auc: 0.663518
[550]	training's auc: 0.805269	valid_1's auc: 0.66368
[560]	training's auc: 0.805776	valid_1's auc: 0.663802
[570]	training's auc: 0.806148	valid_1's auc: 0.663907
[580]	training's auc: 0.806548	valid_1's auc: 0.664067
[590]	training's auc: 0.806946	valid_1's auc: 0.664165
[600]	training's auc: 0.807337	valid_1's auc: 0.664281
[610]	training's auc: 0.807875	valid_1's auc: 0.66449
[620]	training's auc: 0.808231	valid_1's auc: 0.664554
[630]	training's auc: 0.808678	valid_1's auc: 0.664621
[640]	training's auc: 0.809128	valid_1's auc: 0.664663
[650]	training's auc: 0.809619	valid_1's auc: 0.664762
[660]	training's auc: 0.809995	valid_1's auc: 0.664812
[670]	training's auc: 0.81033	valid_1's auc: 0.664881
[680]	training's auc: 0.810794	valid_1's auc: 0.664926
[690]	training's auc: 0.81125	valid_1's auc: 0.66505
[700]	training's auc: 0.811632	valid_1's auc: 0.665166
[710]	training's auc: 0.812008	valid_1's auc: 0.665236
[720]	training's auc: 0.812338	valid_1's auc: 0.665323
[730]	training's auc: 0.812678	valid_1's auc: 0.665434
[740]	training's auc: 0.813098	valid_1's auc: 0.6656
[750]	training's auc: 0.813468	valid_1's auc: 0.665603
[760]	training's auc: 0.813779	valid_1's auc: 0.665627
[770]	training's auc: 0.814104	valid_1's auc: 0.665666
[780]	training's auc: 0.814431	valid_1's auc: 0.665774
[790]	training's auc: 0.814702	valid_1's auc: 0.665835
[800]	training's auc: 0.81502	valid_1's auc: 0.665918
[810]	training's auc: 0.815289	valid_1's auc: 0.665963
[820]	training's auc: 0.815695	valid_1's auc: 0.66608
[830]	training's auc: 0.81608	valid_1's auc: 0.666159
[840]	training's auc: 0.81653	valid_1's auc: 0.666273
[850]	training's auc: 0.816798	valid_1's auc: 0.666331
[860]	training's auc: 0.817108	valid_1's auc: 0.666386
[870]	training's auc: 0.817382	valid_1's auc: 0.666395
[880]	training's auc: 0.817674	valid_1's auc: 0.666436
[890]	training's auc: 0.818096	valid_1's auc: 0.666552
[900]	training's auc: 0.818344	valid_1's auc: 0.666595
[910]	training's auc: 0.818754	valid_1's auc: 0.666678
[920]	training's auc: 0.818968	valid_1's auc: 0.666704
[930]	training's auc: 0.819295	valid_1's auc: 0.666773
[940]	training's auc: 0.819785	valid_1's auc: 0.66689
[950]	training's auc: 0.82015	valid_1's auc: 0.666921
[960]	training's auc: 0.82037	valid_1's auc: 0.666918
[970]	training's auc: 0.820663	valid_1's auc: 0.666972
[980]	training's auc: 0.820948	valid_1's auc: 0.667009
[990]	training's auc: 0.821195	valid_1's auc: 0.667051
[1000]	training's auc: 0.821631	valid_1's auc: 0.66714
[1010]	training's auc: 0.821925	valid_1's auc: 0.667176
[1020]	training's auc: 0.822185	valid_1's auc: 0.667196
[1030]	training's auc: 0.822391	valid_1's auc: 0.667206
[1040]	training's auc: 0.822644	valid_1's auc: 0.667254
[1050]	training's auc: 0.822848	valid_1's auc: 0.667281
[1060]	training's auc: 0.823103	valid_1's auc: 0.667323
[1070]	training's auc: 0.823345	valid_1's auc: 0.667375
[1080]	training's auc: 0.82355	valid_1's auc: 0.667395
[1090]	training's auc: 0.823806	valid_1's auc: 0.667438
[1100]	training's auc: 0.823988	valid_1's auc: 0.667458
[1110]	training's auc: 0.824183	valid_1's auc: 0.667462
[1120]	training's auc: 0.824559	valid_1's auc: 0.667562
[1130]	training's auc: 0.824836	valid_1's auc: 0.667615
[1140]	training's auc: 0.825066	valid_1's auc: 0.667656
[1150]	training's auc: 0.825363	valid_1's auc: 0.667701
[1160]	training's auc: 0.825608	valid_1's auc: 0.667734
[1170]	training's auc: 0.825855	valid_1's auc: 0.667685
[1180]	training's auc: 0.826053	valid_1's auc: 0.667738
[1190]	training's auc: 0.826265	valid_1's auc: 0.667796
[1200]	training's auc: 0.826445	valid_1's auc: 0.667762
[1210]	training's auc: 0.826669	valid_1's auc: 0.667859
[1220]	training's auc: 0.826842	valid_1's auc: 0.667883
[1230]	training's auc: 0.827064	valid_1's auc: 0.667935
[1240]	training's auc: 0.827241	valid_1's auc: 0.667942
[1250]	training's auc: 0.827436	valid_1's auc: 0.667934
[1260]	training's auc: 0.827598	valid_1's auc: 0.667968
[1270]	training's auc: 0.827745	valid_1's auc: 0.667966
[1280]	training's auc: 0.827922	valid_1's auc: 0.667984
[1290]	training's auc: 0.828082	valid_1's auc: 0.667996
[1300]	training's auc: 0.828351	valid_1's auc: 0.668061
[1310]	training's auc: 0.828521	valid_1's auc: 0.668088
[1320]	training's auc: 0.82879	valid_1's auc: 0.668135
[1330]	training's auc: 0.82893	valid_1's auc: 0.668143
[1340]	training's auc: 0.829101	valid_1's auc: 0.66813
[1350]	training's auc: 0.82927	valid_1's auc: 0.668197
[1360]	training's auc: 0.829448	valid_1's auc: 0.668172
[1370]	training's auc: 0.829587	valid_1's auc: 0.668182
[1380]	training's auc: 0.829756	valid_1's auc: 0.66821
[1390]	training's auc: 0.829896	valid_1's auc: 0.668245
[1400]	training's auc: 0.830079	valid_1's auc: 0.668267
[1410]	training's auc: 0.830243	valid_1's auc: 0.66827
[1420]	training's auc: 0.830386	valid_1's auc: 0.66826
[1430]	training's auc: 0.830539	valid_1's auc: 0.668311
[1440]	training's auc: 0.830711	valid_1's auc: 0.668341
[1450]	training's auc: 0.830833	valid_1's auc: 0.66834
[1460]	training's auc: 0.830965	valid_1's auc: 0.668355
[1470]	training's auc: 0.831078	valid_1's auc: 0.668346
[1480]	training's auc: 0.831211	valid_1's auc: 0.668359
[1490]	training's auc: 0.831369	valid_1's auc: 0.668389
[1500]	training's auc: 0.831526	valid_1's auc: 0.668402
[1510]	training's auc: 0.831629	valid_1's auc: 0.668412
[1520]	training's auc: 0.831763	valid_1's auc: 0.66844
[1530]	training's auc: 0.831896	valid_1's auc: 0.668437
[1540]	training's auc: 0.832039	valid_1's auc: 0.668467
[1550]	training's auc: 0.832174	valid_1's auc: 0.668484
[1560]	training's auc: 0.832296	valid_1's auc: 0.668491
[1570]	training's auc: 0.832383	valid_1's auc: 0.66851
[1580]	training's auc: 0.832531	valid_1's auc: 0.668525
[1590]	training's auc: 0.832656	valid_1's auc: 0.668534
[1600]	training's auc: 0.832797	valid_1's auc: 0.668539
[1610]	training's auc: 0.832923	valid_1's auc: 0.668472
[1620]	training's auc: 0.833058	valid_1's auc: 0.668481
[1630]	training's auc: 0.833169	valid_1's auc: 0.668487
[1640]	training's auc: 0.833292	valid_1's auc: 0.668498
Early stopping, best iteration is:
[1594]	training's auc: 0.832714	valid_1's auc: 0.668541
best score: 0.66854093887
best iteration: 1594
complete on: ITC_song_country

working on: CC11_song_country

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
CC11_song_country        int64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.722353	valid_1's auc: 0.631631
[20]	training's auc: 0.736565	valid_1's auc: 0.63761
[30]	training's auc: 0.745808	valid_1's auc: 0.6423
[40]	training's auc: 0.750649	valid_1's auc: 0.644332
[50]	training's auc: 0.75446	valid_1's auc: 0.646062
[60]	training's auc: 0.757767	valid_1's auc: 0.647386
[70]	training's auc: 0.760435	valid_1's auc: 0.648389
[80]	training's auc: 0.762953	valid_1's auc: 0.649593
[90]	training's auc: 0.765155	valid_1's auc: 0.650341
[100]	training's auc: 0.767226	valid_1's auc: 0.651436
[110]	training's auc: 0.769005	valid_1's auc: 0.6521
[120]	training's auc: 0.770688	valid_1's auc: 0.652733
[130]	training's auc: 0.772049	valid_1's auc: 0.653334
[140]	training's auc: 0.774213	valid_1's auc: 0.654168
[150]	training's auc: 0.775523	valid_1's auc: 0.654813
[160]	training's auc: 0.777086	valid_1's auc: 0.65534
[170]	training's auc: 0.778212	valid_1's auc: 0.655934
[180]	training's auc: 0.779541	valid_1's auc: 0.656441
[190]	training's auc: 0.780591	valid_1's auc: 0.656756
[200]	training's auc: 0.781603	valid_1's auc: 0.657195
[210]	training's auc: 0.78272	valid_1's auc: 0.657625
[220]	training's auc: 0.783556	valid_1's auc: 0.657881
[230]	training's auc: 0.784505	valid_1's auc: 0.658258
[240]	training's auc: 0.785416	valid_1's auc: 0.658639
[250]	training's auc: 0.786444	valid_1's auc: 0.658977
[260]	training's auc: 0.78745	valid_1's auc: 0.659196
[270]	training's auc: 0.78823	valid_1's auc: 0.659435
[280]	training's auc: 0.789006	valid_1's auc: 0.6596
[290]	training's auc: 0.789705	valid_1's auc: 0.659802
[300]	training's auc: 0.790598	valid_1's auc: 0.659982
[310]	training's auc: 0.791368	valid_1's auc: 0.660243
[320]	training's auc: 0.792021	valid_1's auc: 0.660465
[330]	training's auc: 0.792803	valid_1's auc: 0.660767
[340]	training's auc: 0.793606	valid_1's auc: 0.660921
[350]	training's auc: 0.794278	valid_1's auc: 0.661093
[360]	training's auc: 0.794881	valid_1's auc: 0.661254
[370]	training's auc: 0.79576	valid_1's auc: 0.661612
[380]	training's auc: 0.796213	valid_1's auc: 0.661761
[390]	training's auc: 0.796738	valid_1's auc: 0.661984
[400]	training's auc: 0.797255	valid_1's auc: 0.662059
[410]	training's auc: 0.797938	valid_1's auc: 0.662237
[420]	training's auc: 0.798445	valid_1's auc: 0.662363
[430]	training's auc: 0.798974	valid_1's auc: 0.662553
[440]	training's auc: 0.799527	valid_1's auc: 0.66271
[450]	training's auc: 0.799993	valid_1's auc: 0.66281
[460]	training's auc: 0.800498	valid_1's auc: 0.662941
[470]	training's auc: 0.801159	valid_1's auc: 0.663065
[480]	training's auc: 0.80183	valid_1's auc: 0.663362
[490]	training's auc: 0.802518	valid_1's auc: 0.663519
[500]	training's auc: 0.802945	valid_1's auc: 0.66362
[510]	training's auc: 0.803571	valid_1's auc: 0.663751
[520]	training's auc: 0.804069	valid_1's auc: 0.663862
[530]	training's auc: 0.804496	valid_1's auc: 0.663926
[540]	training's auc: 0.804898	valid_1's auc: 0.664036
[550]	training's auc: 0.805402	valid_1's auc: 0.664229
[560]	training's auc: 0.806105	valid_1's auc: 0.664414
[570]	training's auc: 0.806645	valid_1's auc: 0.664599
[580]	training's auc: 0.807016	valid_1's auc: 0.664758
[590]	training's auc: 0.807496	valid_1's auc: 0.664918
[600]	training's auc: 0.807894	valid_1's auc: 0.665026
[610]	training's auc: 0.808224	valid_1's auc: 0.665103
[620]	training's auc: 0.808651	valid_1's auc: 0.665165
[630]	training's auc: 0.808997	valid_1's auc: 0.665243
[640]	training's auc: 0.809424	valid_1's auc: 0.665352
[650]	training's auc: 0.809787	valid_1's auc: 0.665487
[660]	training's auc: 0.810095	valid_1's auc: 0.665543
[670]	training's auc: 0.8105	valid_1's auc: 0.665607
[680]	training's auc: 0.810895	valid_1's auc: 0.665652
[690]	training's auc: 0.811272	valid_1's auc: 0.665705
[700]	training's auc: 0.811801	valid_1's auc: 0.665874
[710]	training's auc: 0.812127	valid_1's auc: 0.665921
[720]	training's auc: 0.812474	valid_1's auc: 0.666043
[730]	training's auc: 0.812849	valid_1's auc: 0.666128
[740]	training's auc: 0.813186	valid_1's auc: 0.66619
[750]	training's auc: 0.81359	valid_1's auc: 0.666336
[760]	training's auc: 0.813887	valid_1's auc: 0.666351
[770]	training's auc: 0.814255	valid_1's auc: 0.666417
[780]	training's auc: 0.814768	valid_1's auc: 0.666561
[790]	training's auc: 0.815076	valid_1's auc: 0.666626
[800]	training's auc: 0.815413	valid_1's auc: 0.666653
[810]	training's auc: 0.815727	valid_1's auc: 0.666686
[820]	training's auc: 0.816006	valid_1's auc: 0.666741
[830]	training's auc: 0.816328	valid_1's auc: 0.666773
[840]	training's auc: 0.816633	valid_1's auc: 0.666793
[850]	training's auc: 0.816988	valid_1's auc: 0.666869
[860]	training's auc: 0.817478	valid_1's auc: 0.666937
[870]	training's auc: 0.817722	valid_1's auc: 0.666969
[880]	training's auc: 0.81811	valid_1's auc: 0.667058
[890]	training's auc: 0.818411	valid_1's auc: 0.667102
[900]	training's auc: 0.818652	valid_1's auc: 0.667111
[910]	training's auc: 0.818908	valid_1's auc: 0.667144
[920]	training's auc: 0.819163	valid_1's auc: 0.667237
[930]	training's auc: 0.81954	valid_1's auc: 0.667318
[940]	training's auc: 0.819865	valid_1's auc: 0.66735
[950]	training's auc: 0.8201	valid_1's auc: 0.667356
[960]	training's auc: 0.820424	valid_1's auc: 0.667437
[970]	training's auc: 0.820752	valid_1's auc: 0.667511
[980]	training's auc: 0.821015	valid_1's auc: 0.667551
[990]	training's auc: 0.821278	valid_1's auc: 0.667615
[1000]	training's auc: 0.821537	valid_1's auc: 0.667606
[1010]	training's auc: 0.82186	valid_1's auc: 0.667674
[1020]	training's auc: 0.82219	valid_1's auc: 0.66776
[1030]	training's auc: 0.82247	valid_1's auc: 0.667792
[1040]	training's auc: 0.822883	valid_1's auc: 0.667923
[1050]	training's auc: 0.82315	valid_1's auc: 0.667988
[1060]	training's auc: 0.823333	valid_1's auc: 0.667992
[1070]	training's auc: 0.823524	valid_1's auc: 0.668053
[1080]	training's auc: 0.823743	valid_1's auc: 0.668049
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/in_column_train_V1003BBBB.py", line 279, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 206, in train
    evaluation_result_list.extend(booster.eval_valid(feval))
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1627, in eval_valid
    return [item for i in range_(1, self.__num_dataset)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1628, in <listcomp>
    for item in self.__inner_eval(self.name_valid_sets[i - 1], i, feval)]
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1880, in __inner_eval
    result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
KeyboardInterrupt

Process finished with exit code 1
'''