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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1105BBBB.py
0.778151250384
0.778151
0.77832
0.778151250384
0.778151250384
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
ITC_song_id_log10_1     float16
ITC_song_id_x_1         float64
OinC_song_id            float64
ITC_msno_log10          float64
ITC_msno_log10_1        float16
ITC_msno_x_1            float64
OinC_msno               float64
ITC_name_log10          float64
ITC_name_log10_1        float16
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
working on: ITC_name_log10_1
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1105BBBB.py:245: SettingWithCopyWarning: 
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
ITC_song_id_log10_1     float16
ITC_msno_log10_1        float16
ITC_name_log10_1        float16
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
[10]	training's auc: 0.792526	valid_1's auc: 0.665875
[20]	training's auc: 0.79598	valid_1's auc: 0.667414
[30]	training's auc: 0.799172	valid_1's auc: 0.668978
[40]	training's auc: 0.800732	valid_1's auc: 0.669803
[50]	training's auc: 0.803663	valid_1's auc: 0.670986
[60]	training's auc: 0.807295	valid_1's auc: 0.672173
[70]	training's auc: 0.809808	valid_1's auc: 0.673221
[80]	training's auc: 0.812756	valid_1's auc: 0.674309
[90]	training's auc: 0.815628	valid_1's auc: 0.675477
[100]	training's auc: 0.818402	valid_1's auc: 0.676665
[110]	training's auc: 0.821563	valid_1's auc: 0.677891
[120]	training's auc: 0.824106	valid_1's auc: 0.678879
[130]	training's auc: 0.826479	valid_1's auc: 0.679735
[140]	training's auc: 0.828866	valid_1's auc: 0.680769
[150]	training's auc: 0.83082	valid_1's auc: 0.681453
[160]	training's auc: 0.832887	valid_1's auc: 0.682177
[170]	training's auc: 0.834715	valid_1's auc: 0.682889
[180]	training's auc: 0.836175	valid_1's auc: 0.683388
[190]	training's auc: 0.837575	valid_1's auc: 0.68387
[200]	training's auc: 0.839232	valid_1's auc: 0.684511
[210]	training's auc: 0.840699	valid_1's auc: 0.684979
[220]	training's auc: 0.841973	valid_1's auc: 0.685317
[230]	training's auc: 0.843112	valid_1's auc: 0.685644
[240]	training's auc: 0.844272	valid_1's auc: 0.686023
[250]	training's auc: 0.84538	valid_1's auc: 0.68629
[260]	training's auc: 0.846515	valid_1's auc: 0.686508
[270]	training's auc: 0.847419	valid_1's auc: 0.686684
[280]	training's auc: 0.848195	valid_1's auc: 0.6868
[290]	training's auc: 0.848996	valid_1's auc: 0.686954
[300]	training's auc: 0.84975	valid_1's auc: 0.687046
[310]	training's auc: 0.850508	valid_1's auc: 0.687092
[320]	training's auc: 0.851149	valid_1's auc: 0.687102
[330]	training's auc: 0.851969	valid_1's auc: 0.687146
[340]	training's auc: 0.852603	valid_1's auc: 0.687204
[350]	training's auc: 0.853204	valid_1's auc: 0.687217
[360]	training's auc: 0.853867	valid_1's auc: 0.687306
[370]	training's auc: 0.8545	valid_1's auc: 0.687335
[380]	training's auc: 0.855119	valid_1's auc: 0.687379
[390]	training's auc: 0.855684	valid_1's auc: 0.68741
[400]	training's auc: 0.856263	valid_1's auc: 0.687411
[410]	training's auc: 0.856843	valid_1's auc: 0.68743
[420]	training's auc: 0.857341	valid_1's auc: 0.687437
[430]	training's auc: 0.857884	valid_1's auc: 0.687443
[440]	training's auc: 0.858395	valid_1's auc: 0.687445
[450]	training's auc: 0.858945	valid_1's auc: 0.687454
[460]	training's auc: 0.859417	valid_1's auc: 0.687449
[470]	training's auc: 0.859926	valid_1's auc: 0.687452
[480]	training's auc: 0.860409	valid_1's auc: 0.687462
[490]	training's auc: 0.86086	valid_1's auc: 0.687444
[500]	training's auc: 0.861355	valid_1's auc: 0.687473
[510]	training's auc: 0.86188	valid_1's auc: 0.687484
[520]	training's auc: 0.862311	valid_1's auc: 0.687511
[530]	training's auc: 0.862768	valid_1's auc: 0.68752
[540]	training's auc: 0.863195	valid_1's auc: 0.687507
[550]	training's auc: 0.863559	valid_1's auc: 0.68755
[560]	training's auc: 0.863959	valid_1's auc: 0.687533
[570]	training's auc: 0.864377	valid_1's auc: 0.687541
[580]	training's auc: 0.864741	valid_1's auc: 0.687564
[590]	training's auc: 0.865178	valid_1's auc: 0.687569
[600]	training's auc: 0.865509	valid_1's auc: 0.687559
[610]	training's auc: 0.865878	valid_1's auc: 0.687558
[620]	training's auc: 0.866229	valid_1's auc: 0.687557
[630]	training's auc: 0.86659	valid_1's auc: 0.687568
[640]	training's auc: 0.866977	valid_1's auc: 0.68759
[650]	training's auc: 0.867353	valid_1's auc: 0.687604
[660]	training's auc: 0.867734	valid_1's auc: 0.687619
[670]	training's auc: 0.868127	valid_1's auc: 0.687612
[680]	training's auc: 0.868509	valid_1's auc: 0.687639
[690]	training's auc: 0.868908	valid_1's auc: 0.687663
[700]	training's auc: 0.869249	valid_1's auc: 0.68768
[710]	training's auc: 0.869638	valid_1's auc: 0.687689
[720]	training's auc: 0.869907	valid_1's auc: 0.687689
[730]	training's auc: 0.870171	valid_1's auc: 0.687702
[740]	training's auc: 0.870454	valid_1's auc: 0.687706
[750]	training's auc: 0.87073	valid_1's auc: 0.687715
[760]	training's auc: 0.871044	valid_1's auc: 0.687703
[770]	training's auc: 0.871407	valid_1's auc: 0.687709
[780]	training's auc: 0.871729	valid_1's auc: 0.687731
[790]	training's auc: 0.872017	valid_1's auc: 0.687735
[800]	training's auc: 0.872346	valid_1's auc: 0.687735
[810]	training's auc: 0.872616	valid_1's auc: 0.687739
[820]	training's auc: 0.8729	valid_1's auc: 0.687746
[830]	training's auc: 0.873235	valid_1's auc: 0.687762
[840]	training's auc: 0.873517	valid_1's auc: 0.687752
[850]	training's auc: 0.873806	valid_1's auc: 0.687757
[860]	training's auc: 0.874066	valid_1's auc: 0.687772
[870]	training's auc: 0.87436	valid_1's auc: 0.68778
[880]	training's auc: 0.874655	valid_1's auc: 0.687785
[890]	training's auc: 0.874916	valid_1's auc: 0.687766
[900]	training's auc: 0.875241	valid_1's auc: 0.687792
[910]	training's auc: 0.875498	valid_1's auc: 0.687798
[920]	training's auc: 0.87578	valid_1's auc: 0.687826
[930]	training's auc: 0.876069	valid_1's auc: 0.687835
[940]	training's auc: 0.876327	valid_1's auc: 0.687846
[950]	training's auc: 0.876575	valid_1's auc: 0.687866
[960]	training's auc: 0.876843	valid_1's auc: 0.687884
[970]	training's auc: 0.877079	valid_1's auc: 0.68787
[980]	training's auc: 0.877352	valid_1's auc: 0.687883
[990]	training's auc: 0.877627	valid_1's auc: 0.68789
[1000]	training's auc: 0.877878	valid_1's auc: 0.687895
[1010]	training's auc: 0.878123	valid_1's auc: 0.687897
[1020]	training's auc: 0.878379	valid_1's auc: 0.687897
[1030]	training's auc: 0.878635	valid_1's auc: 0.687875
[1040]	training's auc: 0.878866	valid_1's auc: 0.687879
[1050]	training's auc: 0.879124	valid_1's auc: 0.687887
[1060]	training's auc: 0.879374	valid_1's auc: 0.68789
[1070]	training's auc: 0.879633	valid_1's auc: 0.687887
Early stopping, best iteration is:
[1022]	training's auc: 0.878419	valid_1's auc: 0.6879
Traceback (most recent call last):
  File "/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1105BBBB.py", line 331, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 223, in train
    booster._load_model_from_string(booster._save_model_to_string())
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1691, in _save_model_to_string
    return string_buffer.value.decode()
SystemError: Negative size passed to PyBytes_FromStringAndSize

Process finished with exit code 1
'''
