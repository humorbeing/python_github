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
    # 'language',
    'top3_in_song',
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
    'ITC_name_log10',
    'ITC_name_log10_1',
    'ITC_name',
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
working on: ITC_name_log10
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py:238: SettingWithCopyWarning: 
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
top3_in_song           category
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
ITC_name_log10          float64
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
[10]	training's auc: 0.792094	valid_1's auc: 0.664724
[20]	training's auc: 0.795897	valid_1's auc: 0.666571
[30]	training's auc: 0.799143	valid_1's auc: 0.668259
[40]	training's auc: 0.800663	valid_1's auc: 0.669182
[50]	training's auc: 0.803624	valid_1's auc: 0.670392
[60]	training's auc: 0.807231	valid_1's auc: 0.671627
[70]	training's auc: 0.80971	valid_1's auc: 0.672655
[80]	training's auc: 0.812753	valid_1's auc: 0.673759
[90]	training's auc: 0.81557	valid_1's auc: 0.674944
[100]	training's auc: 0.818385	valid_1's auc: 0.676139
[110]	training's auc: 0.821463	valid_1's auc: 0.677316
[120]	training's auc: 0.823986	valid_1's auc: 0.67832
[130]	training's auc: 0.826383	valid_1's auc: 0.679211
[140]	training's auc: 0.828722	valid_1's auc: 0.680127
[150]	training's auc: 0.830597	valid_1's auc: 0.680785
[160]	training's auc: 0.832689	valid_1's auc: 0.681565
[170]	training's auc: 0.834536	valid_1's auc: 0.68222
[180]	training's auc: 0.836023	valid_1's auc: 0.682742
[190]	training's auc: 0.837519	valid_1's auc: 0.683227
[200]	training's auc: 0.839207	valid_1's auc: 0.683909
[210]	training's auc: 0.840625	valid_1's auc: 0.684354
[220]	training's auc: 0.841855	valid_1's auc: 0.684759
[230]	training's auc: 0.84306	valid_1's auc: 0.685102
[240]	training's auc: 0.844115	valid_1's auc: 0.685433
[250]	training's auc: 0.845162	valid_1's auc: 0.68572
[260]	training's auc: 0.846186	valid_1's auc: 0.685971
[270]	training's auc: 0.847113	valid_1's auc: 0.686127
[280]	training's auc: 0.847862	valid_1's auc: 0.686264
[290]	training's auc: 0.848596	valid_1's auc: 0.686367
[300]	training's auc: 0.849396	valid_1's auc: 0.686449
[310]	training's auc: 0.850163	valid_1's auc: 0.686547
[320]	training's auc: 0.850929	valid_1's auc: 0.686632
[330]	training's auc: 0.851715	valid_1's auc: 0.686686
[340]	training's auc: 0.852395	valid_1's auc: 0.686727
[350]	training's auc: 0.85308	valid_1's auc: 0.686765
[360]	training's auc: 0.853689	valid_1's auc: 0.686799
[370]	training's auc: 0.854289	valid_1's auc: 0.686826
[380]	training's auc: 0.854949	valid_1's auc: 0.686937
[390]	training's auc: 0.855562	valid_1's auc: 0.686937
[400]	training's auc: 0.856127	valid_1's auc: 0.686985
[410]	training's auc: 0.856651	valid_1's auc: 0.687042
[420]	training's auc: 0.857165	valid_1's auc: 0.687062
[430]	training's auc: 0.857728	valid_1's auc: 0.687097
[440]	training's auc: 0.858214	valid_1's auc: 0.687152
[450]	training's auc: 0.858729	valid_1's auc: 0.687181
[460]	training's auc: 0.85919	valid_1's auc: 0.687186
[470]	training's auc: 0.859695	valid_1's auc: 0.687189
[480]	training's auc: 0.860131	valid_1's auc: 0.687215
[490]	training's auc: 0.860563	valid_1's auc: 0.687233
[500]	training's auc: 0.861129	valid_1's auc: 0.687249
[510]	training's auc: 0.861615	valid_1's auc: 0.687273
[520]	training's auc: 0.862116	valid_1's auc: 0.687306
[530]	training's auc: 0.862529	valid_1's auc: 0.687341
[540]	training's auc: 0.862943	valid_1's auc: 0.687333
[550]	training's auc: 0.863401	valid_1's auc: 0.687366
[560]	training's auc: 0.863766	valid_1's auc: 0.687373
[570]	training's auc: 0.86417	valid_1's auc: 0.687378
[580]	training's auc: 0.864598	valid_1's auc: 0.68737
[590]	training's auc: 0.86502	valid_1's auc: 0.687389
[600]	training's auc: 0.86538	valid_1's auc: 0.687424
[610]	training's auc: 0.865755	valid_1's auc: 0.687451
[620]	training's auc: 0.866091	valid_1's auc: 0.687484
[630]	training's auc: 0.86651	valid_1's auc: 0.687527
[640]	training's auc: 0.866932	valid_1's auc: 0.687561
[650]	training's auc: 0.867326	valid_1's auc: 0.687572
[660]	training's auc: 0.867705	valid_1's auc: 0.687577
[670]	training's auc: 0.868053	valid_1's auc: 0.687565
[680]	training's auc: 0.868398	valid_1's auc: 0.687568
[690]	training's auc: 0.868776	valid_1's auc: 0.687573
[700]	training's auc: 0.869176	valid_1's auc: 0.687585
[710]	training's auc: 0.869566	valid_1's auc: 0.687609
[720]	training's auc: 0.869867	valid_1's auc: 0.687615
[730]	training's auc: 0.870161	valid_1's auc: 0.687606
[740]	training's auc: 0.870452	valid_1's auc: 0.687637
[750]	training's auc: 0.870713	valid_1's auc: 0.687646
[760]	training's auc: 0.870984	valid_1's auc: 0.687666
[770]	training's auc: 0.871315	valid_1's auc: 0.687683
[780]	training's auc: 0.871653	valid_1's auc: 0.68767
[790]	training's auc: 0.871946	valid_1's auc: 0.687677
[800]	training's auc: 0.872221	valid_1's auc: 0.687687
[810]	training's auc: 0.872546	valid_1's auc: 0.687706
[820]	training's auc: 0.872831	valid_1's auc: 0.687705
[830]	training's auc: 0.873151	valid_1's auc: 0.687716
[840]	training's auc: 0.873459	valid_1's auc: 0.687723
[850]	training's auc: 0.873718	valid_1's auc: 0.687729
[860]	training's auc: 0.874011	valid_1's auc: 0.687738
[870]	training's auc: 0.874297	valid_1's auc: 0.687758
[880]	training's auc: 0.874602	valid_1's auc: 0.687763
[890]	training's auc: 0.874841	valid_1's auc: 0.687774
[900]	training's auc: 0.875125	valid_1's auc: 0.687765
[910]	training's auc: 0.875381	valid_1's auc: 0.687777
[920]	training's auc: 0.875657	valid_1's auc: 0.687781
[930]	training's auc: 0.875931	valid_1's auc: 0.687789
[940]	training's auc: 0.876173	valid_1's auc: 0.68782
[950]	training's auc: 0.876436	valid_1's auc: 0.687836
[960]	training's auc: 0.876702	valid_1's auc: 0.687841
[970]	training's auc: 0.876957	valid_1's auc: 0.687854
[980]	training's auc: 0.877228	valid_1's auc: 0.687873
[990]	training's auc: 0.877517	valid_1's auc: 0.687885
[1000]	training's auc: 0.877775	valid_1's auc: 0.687888
[1010]	training's auc: 0.878034	valid_1's auc: 0.687908
[1020]	training's auc: 0.878293	valid_1's auc: 0.687908
[1030]	training's auc: 0.878557	valid_1's auc: 0.68792
[1040]	training's auc: 0.878807	valid_1's auc: 0.687932
[1050]	training's auc: 0.879057	valid_1's auc: 0.687943
[1060]	training's auc: 0.879325	valid_1's auc: 0.687957
[1070]	training's auc: 0.879581	valid_1's auc: 0.687958
[1080]	training's auc: 0.879808	valid_1's auc: 0.687953
[1090]	training's auc: 0.880038	valid_1's auc: 0.687953
[1100]	training's auc: 0.880265	valid_1's auc: 0.687963
[1110]	training's auc: 0.88053	valid_1's auc: 0.68797
[1120]	training's auc: 0.880774	valid_1's auc: 0.687968
[1130]	training's auc: 0.881023	valid_1's auc: 0.687952
[1140]	training's auc: 0.881252	valid_1's auc: 0.687961
[1150]	training's auc: 0.881517	valid_1's auc: 0.687962
[1160]	training's auc: 0.881758	valid_1's auc: 0.687953
Early stopping, best iteration is:
[1113]	training's auc: 0.880605	valid_1's auc: 0.687972
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1104BBBB.py", line 324, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 130 (interrupted by signal 2: SIGINT)
'''