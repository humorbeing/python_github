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


print(np.log10(5+1).astype(np.float64))
print(np.log10(5+1).astype(np.float32))
print(np.log10(5+1).astype(np.float16))
print(np.log10(5+1).astype(np.float128))
print(np.log10(5+1).astype(np.float))
# print(np.log10(5+1).astype(np.float80))
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

cols = ['song_id', 'msno', 'artist_name', 'composer', 'lyricist']
for col in cols:
    # print("'{}',".format(col))
    add_this_counter_column(col)

def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.round(np.log10(x+1), 5)


def xxx(x):
    d = x / (x + 1)
    return d


for col in cols:
    colc = 'ITC_'+col
    df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float16)
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
    'ITC_artist_name_log10_1',
    'ITC_composer_log10_1',
    'ITC_lyricist_log10_1',
    # 'ITC_song_id_log10_1',

    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    # 'ITC_msno_log10_1',
    # 'ITC_name_log10',
    # 'ITC_name_log10_1',
    # 'ITC_name',
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
msno                         object
song_id                      object
source_system_tab            object
source_screen_name           object
source_type                  object
target                        uint8
gender                       object
artist_name                  object
composer                     object
lyricist                     object
language                   category
name                         object
song_year                  category
song_country               category
rc                         category
isrc_rest                  category
top1_in_song               category
top2_in_song               category
top3_in_song               category
ITC_song_id                   int64
CC11_song_id                  int64
ITC_msno                      int64
CC11_msno                     int64
ITC_artist_name               int64
CC11_artist_name              int64
ITC_composer                  int64
CC11_composer                 int64
ITC_lyricist                  int64
CC11_lyricist                 int64
ITC_song_id_log10           float64
ITC_song_id_log10_1         float16
ITC_song_id_x_1             float64
OinC_song_id                float64
ITC_msno_log10              float64
ITC_msno_log10_1            float16
ITC_msno_x_1                float64
OinC_msno                   float64
ITC_artist_name_log10       float64
ITC_artist_name_log10_1     float16
ITC_artist_name_x_1         float64
OinC_artist_name            float64
ITC_composer_log10          float64
ITC_composer_log10_1        float16
ITC_composer_x_1            float64
OinC_composer               float64
ITC_lyricist_log10          float64
ITC_lyricist_log10_1        float16
ITC_lyricist_x_1            float64
OinC_lyricist               float64
dtype: object
number of rows: 7377418
number of columns: 49
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
'ITC_artist_name',
'CC11_artist_name',
'ITC_composer',
'CC11_composer',
'ITC_lyricist',
'CC11_lyricist',
'ITC_song_id_log10',
'ITC_song_id_log10_1',
'ITC_song_id_x_1',
'OinC_song_id',
'ITC_msno_log10',
'ITC_msno_log10_1',
'ITC_msno_x_1',
'OinC_msno',
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
working on: ITC_artist_name_log10_1
/home/vb/workspace/python/kagglebigdata/VALIDATION_fake_feature_insert_V1001/one_train_V1105BBBB.py:250: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  df_on[col] = df_on[col].astype('category')

Our guest selection:
target                        uint8
msno                       category
song_id                    category
source_system_tab          category
source_screen_name         category
source_type                category
artist_name                category
song_year                  category
top3_in_song               category
ITC_song_id_log10_1         float16
ITC_msno_log10_1            float16
ITC_artist_name_log10_1     float16
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
[10]	training's auc: 0.794633	valid_1's auc: 0.666247
[20]	training's auc: 0.798061	valid_1's auc: 0.667829
[30]	training's auc: 0.801157	valid_1's auc: 0.669316
[40]	training's auc: 0.802906	valid_1's auc: 0.670216
[50]	training's auc: 0.805381	valid_1's auc: 0.671226
[60]	training's auc: 0.809094	valid_1's auc: 0.672594
[70]	training's auc: 0.81135	valid_1's auc: 0.673546
[80]	training's auc: 0.814699	valid_1's auc: 0.674873
[90]	training's auc: 0.817342	valid_1's auc: 0.675959
[100]	training's auc: 0.820108	valid_1's auc: 0.677097
[110]	training's auc: 0.822962	valid_1's auc: 0.678069
[120]	training's auc: 0.825383	valid_1's auc: 0.679101
[130]	training's auc: 0.827688	valid_1's auc: 0.679964
[140]	training's auc: 0.830022	valid_1's auc: 0.68095
[150]	training's auc: 0.831776	valid_1's auc: 0.68158
[160]	training's auc: 0.833821	valid_1's auc: 0.682221
[170]	training's auc: 0.835544	valid_1's auc: 0.682852
[180]	training's auc: 0.836977	valid_1's auc: 0.683337
[190]	training's auc: 0.838335	valid_1's auc: 0.683757
[200]	training's auc: 0.839995	valid_1's auc: 0.684463
[210]	training's auc: 0.841463	valid_1's auc: 0.684907
[220]	training's auc: 0.8426	valid_1's auc: 0.685246
[230]	training's auc: 0.843835	valid_1's auc: 0.685536
[240]	training's auc: 0.844922	valid_1's auc: 0.685854
[250]	training's auc: 0.846096	valid_1's auc: 0.686141
[260]	training's auc: 0.847111	valid_1's auc: 0.686337
[270]	training's auc: 0.848013	valid_1's auc: 0.686475
[280]	training's auc: 0.848906	valid_1's auc: 0.686645
[290]	training's auc: 0.849698	valid_1's auc: 0.686725
[300]	training's auc: 0.850446	valid_1's auc: 0.686841
[310]	training's auc: 0.851209	valid_1's auc: 0.686951
[320]	training's auc: 0.851973	valid_1's auc: 0.68707
[330]	training's auc: 0.852668	valid_1's auc: 0.687133
[340]	training's auc: 0.853404	valid_1's auc: 0.687263
[350]	training's auc: 0.854108	valid_1's auc: 0.687314
[360]	training's auc: 0.854696	valid_1's auc: 0.687367
[370]	training's auc: 0.85537	valid_1's auc: 0.687425
[380]	training's auc: 0.855978	valid_1's auc: 0.687478
[390]	training's auc: 0.856592	valid_1's auc: 0.687518
[400]	training's auc: 0.857199	valid_1's auc: 0.687547
[410]	training's auc: 0.857727	valid_1's auc: 0.687555
[420]	training's auc: 0.858268	valid_1's auc: 0.687571
[430]	training's auc: 0.858796	valid_1's auc: 0.687566
[440]	training's auc: 0.859252	valid_1's auc: 0.687578
[450]	training's auc: 0.859848	valid_1's auc: 0.687599
[460]	training's auc: 0.860316	valid_1's auc: 0.687601
[470]	training's auc: 0.860813	valid_1's auc: 0.687631
[480]	training's auc: 0.861307	valid_1's auc: 0.687654
[490]	training's auc: 0.861776	valid_1's auc: 0.687656
[500]	training's auc: 0.86226	valid_1's auc: 0.687657
[510]	training's auc: 0.86276	valid_1's auc: 0.687644
[520]	training's auc: 0.863245	valid_1's auc: 0.687645
[530]	training's auc: 0.863652	valid_1's auc: 0.687654
Early stopping, best iteration is:
[483]	training's auc: 0.861442	valid_1's auc: 0.687675
best score: 0.687675408413
best iteration: 483
complete on: ITC_artist_name_log10_1

                msno : 151049
             song_id : 34500
   source_system_tab : 598
  source_screen_name : 1635
         source_type : 1482
         artist_name : 48961
           song_year : 1212
        top3_in_song : 1187
 ITC_song_id_log10_1 : 2340
    ITC_msno_log10_1 : 2910
ITC_artist_name_log10_1 : 456
working on: ITC_composer_log10_1

Our guest selection:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
top3_in_song            category
ITC_song_id_log10_1      float16
ITC_msno_log10_1         float16
ITC_composer_log10_1     float16
dtype: object
number of columns: 12

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.793581	valid_1's auc: 0.666218
[20]	training's auc: 0.796982	valid_1's auc: 0.667711
[30]	training's auc: 0.800188	valid_1's auc: 0.669305
[40]	training's auc: 0.801841	valid_1's auc: 0.670226
[50]	training's auc: 0.804552	valid_1's auc: 0.671286
[60]	training's auc: 0.808413	valid_1's auc: 0.672695
[70]	training's auc: 0.810852	valid_1's auc: 0.67372
[80]	training's auc: 0.814266	valid_1's auc: 0.675146
[90]	training's auc: 0.816932	valid_1's auc: 0.676191
[100]	training's auc: 0.819762	valid_1's auc: 0.677372
[110]	training's auc: 0.822603	valid_1's auc: 0.67831
[120]	training's auc: 0.824975	valid_1's auc: 0.679292
[130]	training's auc: 0.827237	valid_1's auc: 0.680227
[140]	training's auc: 0.829565	valid_1's auc: 0.681172
[150]	training's auc: 0.831451	valid_1's auc: 0.681826
[160]	training's auc: 0.833431	valid_1's auc: 0.682579
[170]	training's auc: 0.835243	valid_1's auc: 0.683248
[180]	training's auc: 0.836613	valid_1's auc: 0.683685
[190]	training's auc: 0.837999	valid_1's auc: 0.684097
[200]	training's auc: 0.839724	valid_1's auc: 0.684865
[210]	training's auc: 0.841179	valid_1's auc: 0.685283
[220]	training's auc: 0.842296	valid_1's auc: 0.685607
[230]	training's auc: 0.843448	valid_1's auc: 0.685923
[240]	training's auc: 0.844526	valid_1's auc: 0.686242
[250]	training's auc: 0.845608	valid_1's auc: 0.686539
[260]	training's auc: 0.846666	valid_1's auc: 0.68671
[270]	training's auc: 0.847563	valid_1's auc: 0.686844
[280]	training's auc: 0.848352	valid_1's auc: 0.687001
[290]	training's auc: 0.849121	valid_1's auc: 0.687162
[300]	training's auc: 0.849912	valid_1's auc: 0.687272
[310]	training's auc: 0.85063	valid_1's auc: 0.687365
[320]	training's auc: 0.851372	valid_1's auc: 0.687473
[330]	training's auc: 0.852166	valid_1's auc: 0.687542
[340]	training's auc: 0.85282	valid_1's auc: 0.687635
[350]	training's auc: 0.853449	valid_1's auc: 0.687707
[360]	training's auc: 0.854054	valid_1's auc: 0.687746
[370]	training's auc: 0.854653	valid_1's auc: 0.687768
[380]	training's auc: 0.855317	valid_1's auc: 0.687815
[390]	training's auc: 0.855906	valid_1's auc: 0.687855
[400]	training's auc: 0.856518	valid_1's auc: 0.687872
[410]	training's auc: 0.857102	valid_1's auc: 0.687877
[420]	training's auc: 0.857578	valid_1's auc: 0.687903
[430]	training's auc: 0.858122	valid_1's auc: 0.687922
[440]	training's auc: 0.858646	valid_1's auc: 0.687958
[450]	training's auc: 0.859167	valid_1's auc: 0.687996
[460]	training's auc: 0.859624	valid_1's auc: 0.687989
[470]	training's auc: 0.860101	valid_1's auc: 0.688013
[480]	training's auc: 0.860559	valid_1's auc: 0.688012
[490]	training's auc: 0.861041	valid_1's auc: 0.688011
[500]	training's auc: 0.861543	valid_1's auc: 0.688023
[510]	training's auc: 0.862032	valid_1's auc: 0.688006
[520]	training's auc: 0.862483	valid_1's auc: 0.68798
[530]	training's auc: 0.862898	valid_1's auc: 0.688
[540]	training's auc: 0.863337	valid_1's auc: 0.687999
[550]	training's auc: 0.863762	valid_1's auc: 0.688014
Early stopping, best iteration is:
[502]	training's auc: 0.861669	valid_1's auc: 0.688037
best score: 0.688036935867
best iteration: 502
complete on: ITC_composer_log10_1

                msno : 154642
             song_id : 35734
   source_system_tab : 640
  source_screen_name : 1672
         source_type : 1557
         artist_name : 53496
           song_year : 1419
        top3_in_song : 1313
 ITC_song_id_log10_1 : 2366
    ITC_msno_log10_1 : 3050
ITC_composer_log10_1 : 131
working on: ITC_lyricist_log10_1

Our guest selection:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
top3_in_song            category
ITC_song_id_log10_1      float16
ITC_msno_log10_1         float16
ITC_lyricist_log10_1     float16
dtype: object
number of columns: 12

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794531	valid_1's auc: 0.666492
[20]	training's auc: 0.798192	valid_1's auc: 0.668259
[30]	training's auc: 0.801386	valid_1's auc: 0.669745
[40]	training's auc: 0.802827	valid_1's auc: 0.670591
[50]	training's auc: 0.805551	valid_1's auc: 0.671745
[60]	training's auc: 0.809184	valid_1's auc: 0.67299
[70]	training's auc: 0.811427	valid_1's auc: 0.673918
[80]	training's auc: 0.814688	valid_1's auc: 0.675279
[90]	training's auc: 0.817332	valid_1's auc: 0.676371
[100]	training's auc: 0.82013	valid_1's auc: 0.677475
[110]	training's auc: 0.822933	valid_1's auc: 0.67855
[120]	training's auc: 0.82524	valid_1's auc: 0.679555
[130]	training's auc: 0.827593	valid_1's auc: 0.680509
[140]	training's auc: 0.829835	valid_1's auc: 0.681346
[150]	training's auc: 0.831648	valid_1's auc: 0.682001
[160]	training's auc: 0.833579	valid_1's auc: 0.682739
[170]	training's auc: 0.835339	valid_1's auc: 0.683386
[180]	training's auc: 0.83678	valid_1's auc: 0.68386
[190]	training's auc: 0.838191	valid_1's auc: 0.684333
[200]	training's auc: 0.839788	valid_1's auc: 0.685047
[210]	training's auc: 0.84125	valid_1's auc: 0.685533
[220]	training's auc: 0.842428	valid_1's auc: 0.685843
[230]	training's auc: 0.843568	valid_1's auc: 0.686111
[240]	training's auc: 0.844603	valid_1's auc: 0.68641
[250]	training's auc: 0.845643	valid_1's auc: 0.686677
[260]	training's auc: 0.846736	valid_1's auc: 0.686875
[270]	training's auc: 0.847602	valid_1's auc: 0.687007
[280]	training's auc: 0.848403	valid_1's auc: 0.687116
[290]	training's auc: 0.849245	valid_1's auc: 0.687251
[300]	training's auc: 0.850021	valid_1's auc: 0.687363
[310]	training's auc: 0.850787	valid_1's auc: 0.687433
[320]	training's auc: 0.851481	valid_1's auc: 0.687497
[330]	training's auc: 0.852257	valid_1's auc: 0.687593
[340]	training's auc: 0.852928	valid_1's auc: 0.687675
[350]	training's auc: 0.853532	valid_1's auc: 0.687708
[360]	training's auc: 0.854223	valid_1's auc: 0.687774
[370]	training's auc: 0.854885	valid_1's auc: 0.687822
[380]	training's auc: 0.855461	valid_1's auc: 0.687843
[390]	training's auc: 0.85606	valid_1's auc: 0.687871
[400]	training's auc: 0.85665	valid_1's auc: 0.687883
[410]	training's auc: 0.857232	valid_1's auc: 0.687906
[420]	training's auc: 0.857697	valid_1's auc: 0.687899
[430]	training's auc: 0.858236	valid_1's auc: 0.687911
[440]	training's auc: 0.858779	valid_1's auc: 0.687944
[450]	training's auc: 0.859273	valid_1's auc: 0.687972
[460]	training's auc: 0.859758	valid_1's auc: 0.687992
[470]	training's auc: 0.860259	valid_1's auc: 0.687982
[480]	training's auc: 0.860707	valid_1's auc: 0.687983
[490]	training's auc: 0.861192	valid_1's auc: 0.687992
[500]	training's auc: 0.861695	valid_1's auc: 0.688024
[510]	training's auc: 0.862132	valid_1's auc: 0.688016
[520]	training's auc: 0.86255	valid_1's auc: 0.688014
[530]	training's auc: 0.862965	valid_1's auc: 0.68803
[540]	training's auc: 0.863422	valid_1's auc: 0.687998
[550]	training's auc: 0.863831	valid_1's auc: 0.68799
Early stopping, best iteration is:
[503]	training's auc: 0.861843	valid_1's auc: 0.688032
best score: 0.688031790231
best iteration: 503
complete on: ITC_lyricist_log10_1

                msno : 155168
             song_id : 36305
   source_system_tab : 597
  source_screen_name : 1642
         source_type : 1558
         artist_name : 52998
           song_year : 1443
        top3_in_song : 1329
 ITC_song_id_log10_1 : 2345
    ITC_msno_log10_1 : 3078
ITC_lyricist_log10_1 : 67
              ITC_artist_name_log10_1:  0.687675408413
                 ITC_lyricist_log10_1:  0.688031790231
                 ITC_composer_log10_1:  0.688036935867

[timer]: complete in 160m 5s

Process finished with exit code 0
'''