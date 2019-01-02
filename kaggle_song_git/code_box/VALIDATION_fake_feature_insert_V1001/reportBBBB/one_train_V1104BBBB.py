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
working on: ITC_msno
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
ITC_msno                  int64
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
[10]	training's auc: 0.795441	valid_1's auc: 0.666797
[20]	training's auc: 0.799761	valid_1's auc: 0.669144
[30]	training's auc: 0.802928	valid_1's auc: 0.670514
[40]	training's auc: 0.804174	valid_1's auc: 0.670952
[50]	training's auc: 0.808041	valid_1's auc: 0.672574
[60]	training's auc: 0.810707	valid_1's auc: 0.673454
[70]	training's auc: 0.81353	valid_1's auc: 0.67457
[80]	training's auc: 0.815165	valid_1's auc: 0.675192
[90]	training's auc: 0.818198	valid_1's auc: 0.676566
[100]	training's auc: 0.821174	valid_1's auc: 0.677845
[110]	training's auc: 0.823566	valid_1's auc: 0.678809
[120]	training's auc: 0.826003	valid_1's auc: 0.679676
[130]	training's auc: 0.828471	valid_1's auc: 0.680811
[140]	training's auc: 0.830604	valid_1's auc: 0.681596
[150]	training's auc: 0.832688	valid_1's auc: 0.682407
[160]	training's auc: 0.834668	valid_1's auc: 0.682938
[170]	training's auc: 0.836604	valid_1's auc: 0.683642
[180]	training's auc: 0.838283	valid_1's auc: 0.684133
[190]	training's auc: 0.839852	valid_1's auc: 0.684588
[200]	training's auc: 0.841442	valid_1's auc: 0.685096
[210]	training's auc: 0.842816	valid_1's auc: 0.685493
[220]	training's auc: 0.844093	valid_1's auc: 0.685866
[230]	training's auc: 0.845376	valid_1's auc: 0.686208
[240]	training's auc: 0.846411	valid_1's auc: 0.686429
[250]	training's auc: 0.847495	valid_1's auc: 0.686694
[260]	training's auc: 0.848493	valid_1's auc: 0.686915
[270]	training's auc: 0.849385	valid_1's auc: 0.687059
[280]	training's auc: 0.850333	valid_1's auc: 0.68721
[290]	training's auc: 0.8511	valid_1's auc: 0.687277
[300]	training's auc: 0.85187	valid_1's auc: 0.68739
[310]	training's auc: 0.852636	valid_1's auc: 0.687453
[320]	training's auc: 0.853354	valid_1's auc: 0.68752
[330]	training's auc: 0.854	valid_1's auc: 0.687595
[340]	training's auc: 0.854665	valid_1's auc: 0.687616
[350]	training's auc: 0.855388	valid_1's auc: 0.687702
[360]	training's auc: 0.856079	valid_1's auc: 0.68774
[370]	training's auc: 0.856637	valid_1's auc: 0.687737
[380]	training's auc: 0.857225	valid_1's auc: 0.687787
[390]	training's auc: 0.857807	valid_1's auc: 0.687784
[400]	training's auc: 0.858446	valid_1's auc: 0.687829
[410]	training's auc: 0.859034	valid_1's auc: 0.687876
[420]	training's auc: 0.859545	valid_1's auc: 0.68793
[430]	training's auc: 0.860047	valid_1's auc: 0.68791
[440]	training's auc: 0.860515	valid_1's auc: 0.687919
[450]	training's auc: 0.861001	valid_1's auc: 0.687925
[460]	training's auc: 0.861494	valid_1's auc: 0.687931
[470]	training's auc: 0.86193	valid_1's auc: 0.687921
[480]	training's auc: 0.862406	valid_1's auc: 0.687923
[490]	training's auc: 0.86287	valid_1's auc: 0.687931
[500]	training's auc: 0.863329	valid_1's auc: 0.687945
[510]	training's auc: 0.863794	valid_1's auc: 0.687935
[520]	training's auc: 0.864255	valid_1's auc: 0.687961
[530]	training's auc: 0.864756	valid_1's auc: 0.687945
[540]	training's auc: 0.865176	valid_1's auc: 0.687945
[550]	training's auc: 0.865608	valid_1's auc: 0.687957
[560]	training's auc: 0.866048	valid_1's auc: 0.687971
[570]	training's auc: 0.866478	valid_1's auc: 0.687975
[580]	training's auc: 0.866905	valid_1's auc: 0.688003
[590]	training's auc: 0.867305	valid_1's auc: 0.688001
[600]	training's auc: 0.86773	valid_1's auc: 0.687992
[610]	training's auc: 0.868178	valid_1's auc: 0.688014
[620]	training's auc: 0.868515	valid_1's auc: 0.688011
[630]	training's auc: 0.868895	valid_1's auc: 0.688033
[640]	training's auc: 0.869256	valid_1's auc: 0.688037
[650]	training's auc: 0.869651	valid_1's auc: 0.688016
[660]	training's auc: 0.870026	valid_1's auc: 0.688013
[670]	training's auc: 0.870383	valid_1's auc: 0.688021
[680]	training's auc: 0.870748	valid_1's auc: 0.688025
Early stopping, best iteration is:
[638]	training's auc: 0.86919	valid_1's auc: 0.688043
best score: 0.688042794995
best iteration: 638
complete on: ITC_msno

                msno : 191943
             song_id : 44703
   source_system_tab : 621
  source_screen_name : 1934
         source_type : 1541
         artist_name : 75248
           song_year : 3334
            language : 497
 ITC_song_id_log10_1 : 2619
            ITC_msno : 2940
working on: ITC_msno_log10

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
ITC_msno_log10          float64
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795407	valid_1's auc: 0.666583
[20]	training's auc: 0.800345	valid_1's auc: 0.669249
[30]	training's auc: 0.803197	valid_1's auc: 0.670708
[40]	training's auc: 0.804416	valid_1's auc: 0.671033
[50]	training's auc: 0.808102	valid_1's auc: 0.672621
[60]	training's auc: 0.810633	valid_1's auc: 0.673503
[70]	training's auc: 0.813456	valid_1's auc: 0.674596
[80]	training's auc: 0.815178	valid_1's auc: 0.675282
[90]	training's auc: 0.818262	valid_1's auc: 0.676698
[100]	training's auc: 0.821227	valid_1's auc: 0.677954
[110]	training's auc: 0.823601	valid_1's auc: 0.678818
[120]	training's auc: 0.826019	valid_1's auc: 0.679772
[130]	training's auc: 0.828301	valid_1's auc: 0.680611
[140]	training's auc: 0.830582	valid_1's auc: 0.681528
[150]	training's auc: 0.832733	valid_1's auc: 0.682346
[160]	training's auc: 0.834735	valid_1's auc: 0.682937
[170]	training's auc: 0.836618	valid_1's auc: 0.683584
[180]	training's auc: 0.838323	valid_1's auc: 0.684046
[190]	training's auc: 0.839897	valid_1's auc: 0.684481
[200]	training's auc: 0.84145	valid_1's auc: 0.685022
[210]	training's auc: 0.842827	valid_1's auc: 0.685393
[220]	training's auc: 0.844185	valid_1's auc: 0.685805
[230]	training's auc: 0.845396	valid_1's auc: 0.68619
[240]	training's auc: 0.846479	valid_1's auc: 0.68644
[250]	training's auc: 0.847478	valid_1's auc: 0.686641
[260]	training's auc: 0.848411	valid_1's auc: 0.686775
[270]	training's auc: 0.849347	valid_1's auc: 0.686949
[280]	training's auc: 0.850256	valid_1's auc: 0.68707
[290]	training's auc: 0.851045	valid_1's auc: 0.687175
[300]	training's auc: 0.851816	valid_1's auc: 0.687238
[310]	training's auc: 0.852554	valid_1's auc: 0.687274
[320]	training's auc: 0.853283	valid_1's auc: 0.687295
[330]	training's auc: 0.853946	valid_1's auc: 0.687382
[340]	training's auc: 0.854542	valid_1's auc: 0.687432
[350]	training's auc: 0.855231	valid_1's auc: 0.687487
[360]	training's auc: 0.855866	valid_1's auc: 0.687526
[370]	training's auc: 0.856506	valid_1's auc: 0.687568
[380]	training's auc: 0.857086	valid_1's auc: 0.687574
[390]	training's auc: 0.857767	valid_1's auc: 0.687607
[400]	training's auc: 0.85835	valid_1's auc: 0.687613
[410]	training's auc: 0.858933	valid_1's auc: 0.687673
[420]	training's auc: 0.85945	valid_1's auc: 0.687698
[430]	training's auc: 0.859993	valid_1's auc: 0.687681
[440]	training's auc: 0.860506	valid_1's auc: 0.687679
[450]	training's auc: 0.860959	valid_1's auc: 0.687651
[460]	training's auc: 0.861465	valid_1's auc: 0.687659
[470]	training's auc: 0.86188	valid_1's auc: 0.687625
Early stopping, best iteration is:
[421]	training's auc: 0.859509	valid_1's auc: 0.687703
best score: 0.687703432451
best iteration: 421
complete on: ITC_msno_log10

                msno : 143528
             song_id : 25748
   source_system_tab : 501
  source_screen_name : 1339
         source_type : 1156
         artist_name : 37407
           song_year : 540
            language : 195
 ITC_song_id_log10_1 : 1899
      ITC_msno_log10 : 2397
working on: ITC_msno_log10_1

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
dtype: object
number of columns: 11

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795365	valid_1's auc: 0.6672
[20]	training's auc: 0.799385	valid_1's auc: 0.669316
[30]	training's auc: 0.802696	valid_1's auc: 0.670755
[40]	training's auc: 0.804206	valid_1's auc: 0.671226
[50]	training's auc: 0.808142	valid_1's auc: 0.672835
[60]	training's auc: 0.810686	valid_1's auc: 0.67363
[70]	training's auc: 0.813584	valid_1's auc: 0.674813
[80]	training's auc: 0.815286	valid_1's auc: 0.675486
[90]	training's auc: 0.818333	valid_1's auc: 0.676883
[100]	training's auc: 0.821288	valid_1's auc: 0.678167
[110]	training's auc: 0.823657	valid_1's auc: 0.678976
[120]	training's auc: 0.826133	valid_1's auc: 0.67992
[130]	training's auc: 0.82856	valid_1's auc: 0.680915
[140]	training's auc: 0.830775	valid_1's auc: 0.681758
[150]	training's auc: 0.832948	valid_1's auc: 0.6826
[160]	training's auc: 0.834928	valid_1's auc: 0.683222
[170]	training's auc: 0.836792	valid_1's auc: 0.683856
[180]	training's auc: 0.83856	valid_1's auc: 0.684405
[190]	training's auc: 0.840165	valid_1's auc: 0.684879
[200]	training's auc: 0.841743	valid_1's auc: 0.685382
[210]	training's auc: 0.843131	valid_1's auc: 0.685725
[220]	training's auc: 0.844423	valid_1's auc: 0.686106
[230]	training's auc: 0.845671	valid_1's auc: 0.686449
[240]	training's auc: 0.846783	valid_1's auc: 0.686752
[250]	training's auc: 0.847878	valid_1's auc: 0.686968
[260]	training's auc: 0.848867	valid_1's auc: 0.687198
[270]	training's auc: 0.849797	valid_1's auc: 0.687346
[280]	training's auc: 0.850673	valid_1's auc: 0.687467
[290]	training's auc: 0.851455	valid_1's auc: 0.687546
[300]	training's auc: 0.852194	valid_1's auc: 0.687688
[310]	training's auc: 0.852938	valid_1's auc: 0.687772
[320]	training's auc: 0.853641	valid_1's auc: 0.687799
[330]	training's auc: 0.854328	valid_1's auc: 0.687831
[340]	training's auc: 0.85493	valid_1's auc: 0.687899
[350]	training's auc: 0.855619	valid_1's auc: 0.687889
[360]	training's auc: 0.85632	valid_1's auc: 0.687959
[370]	training's auc: 0.856907	valid_1's auc: 0.688
[380]	training's auc: 0.857517	valid_1's auc: 0.68803
[390]	training's auc: 0.858122	valid_1's auc: 0.688071
[400]	training's auc: 0.85873	valid_1's auc: 0.688086
[410]	training's auc: 0.859243	valid_1's auc: 0.688118
[420]	training's auc: 0.859795	valid_1's auc: 0.688127
[430]	training's auc: 0.860267	valid_1's auc: 0.688116
[440]	training's auc: 0.860816	valid_1's auc: 0.688128
[450]	training's auc: 0.861277	valid_1's auc: 0.68815
[460]	training's auc: 0.861769	valid_1's auc: 0.688168
[470]	training's auc: 0.862237	valid_1's auc: 0.688206
[480]	training's auc: 0.862782	valid_1's auc: 0.68824
[490]	training's auc: 0.863197	valid_1's auc: 0.68823
[500]	training's auc: 0.86367	valid_1's auc: 0.688247
[510]	training's auc: 0.864145	valid_1's auc: 0.688263
[520]	training's auc: 0.86461	valid_1's auc: 0.688242
[530]	training's auc: 0.865063	valid_1's auc: 0.688239
[540]	training's auc: 0.865514	valid_1's auc: 0.688226
[550]	training's auc: 0.865985	valid_1's auc: 0.688235
[560]	training's auc: 0.866406	valid_1's auc: 0.688265
[570]	training's auc: 0.86684	valid_1's auc: 0.68827
[580]	training's auc: 0.867235	valid_1's auc: 0.688279
[590]	training's auc: 0.867664	valid_1's auc: 0.688291
[600]	training's auc: 0.868054	valid_1's auc: 0.688285
[610]	training's auc: 0.868434	valid_1's auc: 0.688303
[620]	training's auc: 0.868777	valid_1's auc: 0.688324
[630]	training's auc: 0.869193	valid_1's auc: 0.68833
[640]	training's auc: 0.869562	valid_1's auc: 0.688314
[650]	training's auc: 0.87	valid_1's auc: 0.688336
[660]	training's auc: 0.870433	valid_1's auc: 0.688381
[670]	training's auc: 0.870846	valid_1's auc: 0.688413
[680]	training's auc: 0.871195	valid_1's auc: 0.688414
[690]	training's auc: 0.87152	valid_1's auc: 0.688407
[700]	training's auc: 0.871859	valid_1's auc: 0.688416
[710]	training's auc: 0.872235	valid_1's auc: 0.688433
[720]	training's auc: 0.872618	valid_1's auc: 0.688448
[730]	training's auc: 0.87294	valid_1's auc: 0.688448
[740]	training's auc: 0.873318	valid_1's auc: 0.688474
[750]	training's auc: 0.873725	valid_1's auc: 0.688503
[760]	training's auc: 0.874104	valid_1's auc: 0.688516
[770]	training's auc: 0.874403	valid_1's auc: 0.688519
[780]	training's auc: 0.8747	valid_1's auc: 0.688524
[790]	training's auc: 0.87506	valid_1's auc: 0.688526
[800]	training's auc: 0.875408	valid_1's auc: 0.688538
[810]	training's auc: 0.875737	valid_1's auc: 0.688537
[820]	training's auc: 0.876039	valid_1's auc: 0.68855
[830]	training's auc: 0.876371	valid_1's auc: 0.688557
[840]	training's auc: 0.8767	valid_1's auc: 0.688546
[850]	training's auc: 0.876996	valid_1's auc: 0.688544
[860]	training's auc: 0.877318	valid_1's auc: 0.688543
[870]	training's auc: 0.877601	valid_1's auc: 0.688552
[880]	training's auc: 0.877913	valid_1's auc: 0.68856
[890]	training's auc: 0.878218	valid_1's auc: 0.688584
[900]	training's auc: 0.878515	valid_1's auc: 0.688569
[910]	training's auc: 0.878822	valid_1's auc: 0.688574
[920]	training's auc: 0.879105	valid_1's auc: 0.68858
[930]	training's auc: 0.879419	valid_1's auc: 0.688603
[940]	training's auc: 0.879705	valid_1's auc: 0.688586
[950]	training's auc: 0.879992	valid_1's auc: 0.6886
[960]	training's auc: 0.880291	valid_1's auc: 0.688599
[970]	training's auc: 0.88057	valid_1's auc: 0.688606
[980]	training's auc: 0.880844	valid_1's auc: 0.688594
[990]	training's auc: 0.881116	valid_1's auc: 0.688609
[1000]	training's auc: 0.881394	valid_1's auc: 0.688612
[1010]	training's auc: 0.881696	valid_1's auc: 0.688646
[1020]	training's auc: 0.881971	valid_1's auc: 0.688627
[1030]	training's auc: 0.882265	valid_1's auc: 0.688647
[1040]	training's auc: 0.88253	valid_1's auc: 0.688657
[1050]	training's auc: 0.882803	valid_1's auc: 0.688671
[1060]	training's auc: 0.883033	valid_1's auc: 0.688689
[1070]	training's auc: 0.883288	valid_1's auc: 0.688693
[1080]	training's auc: 0.883532	valid_1's auc: 0.688691
[1090]	training's auc: 0.883779	valid_1's auc: 0.688683
[1100]	training's auc: 0.884021	valid_1's auc: 0.688672
[1110]	training's auc: 0.884256	valid_1's auc: 0.688678
[1120]	training's auc: 0.884526	valid_1's auc: 0.688698
[1130]	training's auc: 0.88476	valid_1's auc: 0.688716
[1140]	training's auc: 0.885011	valid_1's auc: 0.688713
[1150]	training's auc: 0.885229	valid_1's auc: 0.688708
[1160]	training's auc: 0.885456	valid_1's auc: 0.6887
[1170]	training's auc: 0.885678	valid_1's auc: 0.688715
[1180]	training's auc: 0.88589	valid_1's auc: 0.688726
[1190]	training's auc: 0.886122	valid_1's auc: 0.688719
[1200]	training's auc: 0.886369	valid_1's auc: 0.68871
[1210]	training's auc: 0.886632	valid_1's auc: 0.688732
[1220]	training's auc: 0.88687	valid_1's auc: 0.688742
[1230]	training's auc: 0.887105	valid_1's auc: 0.688765
[1240]	training's auc: 0.887338	valid_1's auc: 0.688771
[1250]	training's auc: 0.887574	valid_1's auc: 0.688776
[1260]	training's auc: 0.887792	valid_1's auc: 0.688772
[1270]	training's auc: 0.888063	valid_1's auc: 0.68877
[1280]	training's auc: 0.888303	valid_1's auc: 0.688775
[1290]	training's auc: 0.888528	valid_1's auc: 0.688762
[1300]	training's auc: 0.888756	valid_1's auc: 0.688775
Early stopping, best iteration is:
[1252]	training's auc: 0.887621	valid_1's auc: 0.688782
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py", line 323, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 1
'''