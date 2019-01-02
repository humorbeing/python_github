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
# inner = False


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


for col in cols:
    colc = 'ITC_'+col
    df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    col1 = 'CC11_'+col
    df['OinC_'+col] = df[col1]/df[colc]


if inner:
    for i in inner:
        insert_this(i)

print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 1000
early_stopping_rounds = 200
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.04
num_leaves = 63
max_depth = 10

lambda_l1 = 0
lambda_l2 = 0.3


bagging_fraction = 0.8
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.8
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,

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
    'FAKE_1512883008',
]
result = {}
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
        val_set = lgb.Dataset(
            X_val, Y_val,
            # weight=[0.1, 1]
        )
        # train_set.max_bin = max_bin
        # val_set.max_bin = max_bin

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

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1002.py
What we got:
target                        uint8
ITC_msno                      int64
CC11_msno                     int64
ITC_song_id                   int64
CC11_song_id                  int64
ITC_source_system_tab         int64
CC11_source_system_tab        int64
ITC_source_screen_name        int64
CC11_source_screen_name       int64
ITC_source_type               int64
CC11_source_type              int64
ITC_gender                    int64
CC11_gender                   int64
ITC_artist_name               int64
CC11_artist_name              int64
ITC_composer                  int64
CC11_composer                 int64
ITC_lyricist                  int64
CC11_lyricist                 int64
ITC_language                  int64
CC11_language                 int64
ITC_name                      int64
CC11_name                     int64
ITC_song_year                 int64
CC11_song_year                int64
ITC_song_country              int64
CC11_song_country             int64
ITC_rc                        int64
CC11_rc                       int64
ITC_isrc_rest                 int64
                             ...   
ITC_lyricist_log10_1        float64
OinC_lyricist               float64
ITC_language_log10          float64
ITC_language_log10_1        float64
OinC_language               float64
ITC_name_log10              float64
ITC_name_log10_1            float64
OinC_name                   float64
ITC_song_year_log10         float64
ITC_song_year_log10_1       float64
OinC_song_year              float64
ITC_song_country_log10      float64
ITC_song_country_log10_1    float64
OinC_song_country           float64
ITC_rc_log10                float64
ITC_rc_log10_1              float64
OinC_rc                     float64
ITC_isrc_rest_log10         float64
ITC_isrc_rest_log10_1       float64
OinC_isrc_rest              float64
ITC_top1_in_song_log10      float64
ITC_top1_in_song_log10_1    float64
OinC_top1_in_song           float64
ITC_top2_in_song_log10      float64
ITC_top2_in_song_log10_1    float64
OinC_top2_in_song           float64
ITC_top3_in_song_log10      float64
ITC_top3_in_song_log10_1    float64
OinC_top3_in_song           float64
FAKE_1512883008             float64
Length: 92, dtype: object
number of rows: 7377418
number of columns: 92
working on: ITC_msno

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_msno             int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.85671	valid_1's auc: 0.678083
[20]	training's auc: 0.857131	valid_1's auc: 0.678337
[30]	training's auc: 0.857469	valid_1's auc: 0.678546
[40]	training's auc: 0.857731	valid_1's auc: 0.678723
[50]	training's auc: 0.857936	valid_1's auc: 0.678853
[60]	training's auc: 0.858089	valid_1's auc: 0.678939
[70]	training's auc: 0.858209	valid_1's auc: 0.679004
[80]	training's auc: 0.858307	valid_1's auc: 0.679049
[90]	training's auc: 0.858385	valid_1's auc: 0.679081
[100]	training's auc: 0.858446	valid_1's auc: 0.679102
[110]	training's auc: 0.858495	valid_1's auc: 0.679118
[120]	training's auc: 0.858533	valid_1's auc: 0.679126
[130]	training's auc: 0.858563	valid_1's auc: 0.679128
[140]	training's auc: 0.858589	valid_1's auc: 0.679126
[150]	training's auc: 0.858609	valid_1's auc: 0.679125
[160]	training's auc: 0.858625	valid_1's auc: 0.679125
[170]	training's auc: 0.858638	valid_1's auc: 0.679121
[180]	training's auc: 0.858649	valid_1's auc: 0.679116
[190]	training's auc: 0.858658	valid_1's auc: 0.679112
[200]	training's auc: 0.858664	valid_1's auc: 0.67911
[210]	training's auc: 0.85867	valid_1's auc: 0.679108
[220]	training's auc: 0.858674	valid_1's auc: 0.679103
[230]	training's auc: 0.858677	valid_1's auc: 0.6791
[240]	training's auc: 0.85868	valid_1's auc: 0.679098
[250]	training's auc: 0.858682	valid_1's auc: 0.679095
[260]	training's auc: 0.858683	valid_1's auc: 0.679092
[270]	training's auc: 0.858685	valid_1's auc: 0.679088
[280]	training's auc: 0.858686	valid_1's auc: 0.679087
[290]	training's auc: 0.858687	valid_1's auc: 0.679084
[300]	training's auc: 0.858688	valid_1's auc: 0.679082
[310]	training's auc: 0.858689	valid_1's auc: 0.679081
[320]	training's auc: 0.858689	valid_1's auc: 0.679079
Early stopping, best iteration is:
[127]	training's auc: 0.858556	valid_1's auc: 0.679129
best score: 0.679129248405
best iteration: 127
complete on: ITC_msno

working on: CC11_msno

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_msno            int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.850605	valid_1's auc: 0.65618
[20]	training's auc: 0.852334	valid_1's auc: 0.657056
[30]	training's auc: 0.853875	valid_1's auc: 0.657538
[40]	training's auc: 0.855178	valid_1's auc: 0.657722
[50]	training's auc: 0.856279	valid_1's auc: 0.657684
[60]	training's auc: 0.857183	valid_1's auc: 0.657479
[70]	training's auc: 0.857922	valid_1's auc: 0.657172
[80]	training's auc: 0.858522	valid_1's auc: 0.6568
[90]	training's auc: 0.859007	valid_1's auc: 0.656388
[100]	training's auc: 0.859395	valid_1's auc: 0.655974
[110]	training's auc: 0.859703	valid_1's auc: 0.655572
[120]	training's auc: 0.859946	valid_1's auc: 0.655193
[130]	training's auc: 0.860138	valid_1's auc: 0.654841
[140]	training's auc: 0.860288	valid_1's auc: 0.654526
[150]	training's auc: 0.860412	valid_1's auc: 0.654248
[160]	training's auc: 0.860509	valid_1's auc: 0.653999
[170]	training's auc: 0.860583	valid_1's auc: 0.653788
[180]	training's auc: 0.860641	valid_1's auc: 0.65361
[190]	training's auc: 0.860685	valid_1's auc: 0.653454
[200]	training's auc: 0.860718	valid_1's auc: 0.653329
[210]	training's auc: 0.860743	valid_1's auc: 0.653219
[220]	training's auc: 0.860762	valid_1's auc: 0.653132
[230]	training's auc: 0.860777	valid_1's auc: 0.653059
[240]	training's auc: 0.860787	valid_1's auc: 0.652999
Early stopping, best iteration is:
[42]	training's auc: 0.855415	valid_1's auc: 0.657735
best score: 0.657734817727
best iteration: 42
complete on: CC11_msno

working on: ITC_song_id

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_song_id          int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.847726	valid_1's auc: 0.679386
[20]	training's auc: 0.849465	valid_1's auc: 0.679753
[30]	training's auc: 0.851032	valid_1's auc: 0.680004
[40]	training's auc: 0.852395	valid_1's auc: 0.680173
[50]	training's auc: 0.853549	valid_1's auc: 0.68026
[60]	training's auc: 0.854495	valid_1's auc: 0.680272
[70]	training's auc: 0.85526	valid_1's auc: 0.680235
[80]	training's auc: 0.855869	valid_1's auc: 0.680158
[90]	training's auc: 0.856355	valid_1's auc: 0.680054
[100]	training's auc: 0.85674	valid_1's auc: 0.679934
[110]	training's auc: 0.857048	valid_1's auc: 0.679803
[120]	training's auc: 0.857292	valid_1's auc: 0.679676
[130]	training's auc: 0.857488	valid_1's auc: 0.679547
[140]	training's auc: 0.857644	valid_1's auc: 0.679425
[150]	training's auc: 0.857769	valid_1's auc: 0.679309
[160]	training's auc: 0.85787	valid_1's auc: 0.679205
[170]	training's auc: 0.857949	valid_1's auc: 0.679107
[180]	training's auc: 0.858015	valid_1's auc: 0.679012
[190]	training's auc: 0.858065	valid_1's auc: 0.67892
[200]	training's auc: 0.858107	valid_1's auc: 0.678834
[210]	training's auc: 0.85814	valid_1's auc: 0.678755
[220]	training's auc: 0.858165	valid_1's auc: 0.678687
[230]	training's auc: 0.858186	valid_1's auc: 0.678621
[240]	training's auc: 0.858202	valid_1's auc: 0.678564
[250]	training's auc: 0.858215	valid_1's auc: 0.678517
Early stopping, best iteration is:
[59]	training's auc: 0.854348	valid_1's auc: 0.680278
best score: 0.680277535195
best iteration: 59
complete on: ITC_song_id

working on: CC11_song_id

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_song_id         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.853352	valid_1's auc: 0.648729
[20]	training's auc: 0.855032	valid_1's auc: 0.649227
[30]	training's auc: 0.856556	valid_1's auc: 0.649335
[40]	training's auc: 0.857883	valid_1's auc: 0.649183
[50]	training's auc: 0.859005	valid_1's auc: 0.648856
[60]	training's auc: 0.859941	valid_1's auc: 0.648397
[70]	training's auc: 0.860701	valid_1's auc: 0.647858
[80]	training's auc: 0.861318	valid_1's auc: 0.647275
[90]	training's auc: 0.861816	valid_1's auc: 0.646674
[100]	training's auc: 0.862217	valid_1's auc: 0.646076
[110]	training's auc: 0.86254	valid_1's auc: 0.645491
[120]	training's auc: 0.862802	valid_1's auc: 0.644935
[130]	training's auc: 0.863013	valid_1's auc: 0.644409
[140]	training's auc: 0.863183	valid_1's auc: 0.643927
[150]	training's auc: 0.863322	valid_1's auc: 0.643479
[160]	training's auc: 0.863431	valid_1's auc: 0.643068
[170]	training's auc: 0.863523	valid_1's auc: 0.6427
[180]	training's auc: 0.863599	valid_1's auc: 0.642365
[190]	training's auc: 0.863656	valid_1's auc: 0.642064
[200]	training's auc: 0.863701	valid_1's auc: 0.6418
[210]	training's auc: 0.863737	valid_1's auc: 0.641564
[220]	training's auc: 0.863766	valid_1's auc: 0.64136
Early stopping, best iteration is:
[28]	training's auc: 0.856262	valid_1's auc: 0.649349
best score: 0.649348603769
best iteration: 28
complete on: CC11_song_id

working on: ITC_source_system_tab

Our guest selection:
target                     uint8
FAKE_1512883008          float64
ITC_source_system_tab      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.837857	valid_1's auc: 0.671892
[20]	training's auc: 0.840559	valid_1's auc: 0.673096
[30]	training's auc: 0.84315	valid_1's auc: 0.674221
[40]	training's auc: 0.845556	valid_1's auc: 0.675228
[50]	training's auc: 0.847715	valid_1's auc: 0.676095
[60]	training's auc: 0.849562	valid_1's auc: 0.67678
[70]	training's auc: 0.851203	valid_1's auc: 0.677346
[80]	training's auc: 0.852614	valid_1's auc: 0.677773
[90]	training's auc: 0.853788	valid_1's auc: 0.678079
[100]	training's auc: 0.854763	valid_1's auc: 0.678292
[110]	training's auc: 0.855586	valid_1's auc: 0.67842
[120]	training's auc: 0.856271	valid_1's auc: 0.678487
[130]	training's auc: 0.856817	valid_1's auc: 0.678504
[140]	training's auc: 0.857292	valid_1's auc: 0.678481
[150]	training's auc: 0.857646	valid_1's auc: 0.678437
[160]	training's auc: 0.857927	valid_1's auc: 0.678392
[170]	training's auc: 0.858232	valid_1's auc: 0.678276
[180]	training's auc: 0.858416	valid_1's auc: 0.678184
[190]	training's auc: 0.858562	valid_1's auc: 0.6781
[200]	training's auc: 0.858689	valid_1's auc: 0.678
[210]	training's auc: 0.858793	valid_1's auc: 0.677913
[220]	training's auc: 0.858878	valid_1's auc: 0.67782
[230]	training's auc: 0.85894	valid_1's auc: 0.677735
[240]	training's auc: 0.858995	valid_1's auc: 0.677653
[250]	training's auc: 0.859033	valid_1's auc: 0.677573
[260]	training's auc: 0.859063	valid_1's auc: 0.677508
[270]	training's auc: 0.859086	valid_1's auc: 0.67745
[280]	training's auc: 0.859106	valid_1's auc: 0.677391
[290]	training's auc: 0.859124	valid_1's auc: 0.677331
[300]	training's auc: 0.859137	valid_1's auc: 0.677281
[310]	training's auc: 0.859146	valid_1's auc: 0.677232
[320]	training's auc: 0.859154	valid_1's auc: 0.67719
Early stopping, best iteration is:
[127]	training's auc: 0.856715	valid_1's auc: 0.678508
best score: 0.67850835602
best iteration: 127
complete on: ITC_source_system_tab

working on: CC11_source_system_tab

Our guest selection:
target                      uint8
FAKE_1512883008           float64
CC11_source_system_tab      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.837829	valid_1's auc: 0.67188
[20]	training's auc: 0.840592	valid_1's auc: 0.673109
[30]	training's auc: 0.84315	valid_1's auc: 0.674219
[40]	training's auc: 0.845566	valid_1's auc: 0.675229
[50]	training's auc: 0.847706	valid_1's auc: 0.676087
[60]	training's auc: 0.849591	valid_1's auc: 0.676795
[70]	training's auc: 0.851226	valid_1's auc: 0.677355
[80]	training's auc: 0.852592	valid_1's auc: 0.677767
[90]	training's auc: 0.853786	valid_1's auc: 0.678076
[100]	training's auc: 0.854765	valid_1's auc: 0.678291
[110]	training's auc: 0.855584	valid_1's auc: 0.678417
[120]	training's auc: 0.856271	valid_1's auc: 0.678486
[130]	training's auc: 0.856827	valid_1's auc: 0.678508
[140]	training's auc: 0.857291	valid_1's auc: 0.678486
[150]	training's auc: 0.85764	valid_1's auc: 0.678435
[160]	training's auc: 0.857922	valid_1's auc: 0.678394
[170]	training's auc: 0.858234	valid_1's auc: 0.678281
[180]	training's auc: 0.858411	valid_1's auc: 0.678191
[190]	training's auc: 0.858563	valid_1's auc: 0.678105
[200]	training's auc: 0.858691	valid_1's auc: 0.678
[210]	training's auc: 0.858792	valid_1's auc: 0.677915
[220]	training's auc: 0.85888	valid_1's auc: 0.677825
[230]	training's auc: 0.858939	valid_1's auc: 0.677738
[240]	training's auc: 0.858988	valid_1's auc: 0.677651
[250]	training's auc: 0.859029	valid_1's auc: 0.677575
[260]	training's auc: 0.859063	valid_1's auc: 0.677508
[270]	training's auc: 0.859087	valid_1's auc: 0.677446
[280]	training's auc: 0.859108	valid_1's auc: 0.677382
[290]	training's auc: 0.859124	valid_1's auc: 0.67733
[300]	training's auc: 0.859135	valid_1's auc: 0.677273
[310]	training's auc: 0.859145	valid_1's auc: 0.677233
[320]	training's auc: 0.859154	valid_1's auc: 0.677194
Early stopping, best iteration is:
[129]	training's auc: 0.85682	valid_1's auc: 0.678509
best score: 0.678508617324
best iteration: 129
complete on: CC11_source_system_tab

working on: ITC_source_screen_name

Our guest selection:
target                      uint8
FAKE_1512883008           float64
ITC_source_screen_name      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.835963	valid_1's auc: 0.670804
[20]	training's auc: 0.83882	valid_1's auc: 0.672082
[30]	training's auc: 0.841565	valid_1's auc: 0.673281
[40]	training's auc: 0.844132	valid_1's auc: 0.674393
[50]	training's auc: 0.846417	valid_1's auc: 0.675322
[60]	training's auc: 0.848466	valid_1's auc: 0.676115
[70]	training's auc: 0.850236	valid_1's auc: 0.676764
[80]	training's auc: 0.851791	valid_1's auc: 0.677285
[90]	training's auc: 0.853103	valid_1's auc: 0.677676
[100]	training's auc: 0.854217	valid_1's auc: 0.677967
[110]	training's auc: 0.855155	valid_1's auc: 0.678174
[120]	training's auc: 0.855941	valid_1's auc: 0.67831
[130]	training's auc: 0.856596	valid_1's auc: 0.678398
[140]	training's auc: 0.857148	valid_1's auc: 0.678431
[150]	training's auc: 0.85757	valid_1's auc: 0.678422
[160]	training's auc: 0.857915	valid_1's auc: 0.678421
[170]	training's auc: 0.858275	valid_1's auc: 0.678365
[180]	training's auc: 0.858501	valid_1's auc: 0.678295
[190]	training's auc: 0.858688	valid_1's auc: 0.678239
[200]	training's auc: 0.85886	valid_1's auc: 0.678164
[210]	training's auc: 0.858993	valid_1's auc: 0.678088
[220]	training's auc: 0.859098	valid_1's auc: 0.678016
[230]	training's auc: 0.859178	valid_1's auc: 0.677943
[240]	training's auc: 0.859245	valid_1's auc: 0.677867
[250]	training's auc: 0.859301	valid_1's auc: 0.677798
[260]	training's auc: 0.859345	valid_1's auc: 0.677737
[270]	training's auc: 0.859379	valid_1's auc: 0.677678
[280]	training's auc: 0.859408	valid_1's auc: 0.677622
[290]	training's auc: 0.859432	valid_1's auc: 0.67757
[300]	training's auc: 0.859451	valid_1's auc: 0.67752
[310]	training's auc: 0.859467	valid_1's auc: 0.677472
[320]	training's auc: 0.85948	valid_1's auc: 0.677432
[330]	training's auc: 0.859491	valid_1's auc: 0.677395
[340]	training's auc: 0.859498	valid_1's auc: 0.677363
Early stopping, best iteration is:
[141]	training's auc: 0.857236	valid_1's auc: 0.678436
best score: 0.678436068342
best iteration: 141
complete on: ITC_source_screen_name

working on: CC11_source_screen_name

Our guest selection:
target                       uint8
FAKE_1512883008            float64
CC11_source_screen_name      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.835959	valid_1's auc: 0.6708
[20]	training's auc: 0.838827	valid_1's auc: 0.672084
[30]	training's auc: 0.841576	valid_1's auc: 0.673294
[40]	training's auc: 0.844139	valid_1's auc: 0.674387
[50]	training's auc: 0.846421	valid_1's auc: 0.675318
[60]	training's auc: 0.848469	valid_1's auc: 0.676114
[70]	training's auc: 0.850251	valid_1's auc: 0.676765
[80]	training's auc: 0.851793	valid_1's auc: 0.677282
[90]	training's auc: 0.853111	valid_1's auc: 0.677678
[100]	training's auc: 0.854226	valid_1's auc: 0.677973
[110]	training's auc: 0.855171	valid_1's auc: 0.678176
[120]	training's auc: 0.855957	valid_1's auc: 0.678316
[130]	training's auc: 0.856604	valid_1's auc: 0.678393
[140]	training's auc: 0.857142	valid_1's auc: 0.678426
[150]	training's auc: 0.857571	valid_1's auc: 0.678424
[160]	training's auc: 0.857931	valid_1's auc: 0.678421
[170]	training's auc: 0.858278	valid_1's auc: 0.678366
[180]	training's auc: 0.858504	valid_1's auc: 0.678297
[190]	training's auc: 0.858696	valid_1's auc: 0.678239
[200]	training's auc: 0.858863	valid_1's auc: 0.678165
[210]	training's auc: 0.858993	valid_1's auc: 0.678085
[220]	training's auc: 0.859096	valid_1's auc: 0.678012
[230]	training's auc: 0.859181	valid_1's auc: 0.677934
[240]	training's auc: 0.859249	valid_1's auc: 0.677863
[250]	training's auc: 0.859305	valid_1's auc: 0.677796
[260]	training's auc: 0.859348	valid_1's auc: 0.677735
[270]	training's auc: 0.859383	valid_1's auc: 0.677674
[280]	training's auc: 0.859412	valid_1's auc: 0.677623
[290]	training's auc: 0.859436	valid_1's auc: 0.677571
[300]	training's auc: 0.859456	valid_1's auc: 0.677518
[310]	training's auc: 0.859471	valid_1's auc: 0.677474
[320]	training's auc: 0.859482	valid_1's auc: 0.677437
[330]	training's auc: 0.859492	valid_1's auc: 0.677401
[340]	training's auc: 0.8595	valid_1's auc: 0.677362
Early stopping, best iteration is:
[146]	training's auc: 0.85742	valid_1's auc: 0.67843
best score: 0.678430430108
best iteration: 146
complete on: CC11_source_screen_name

working on: ITC_source_type

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_source_type      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.835878	valid_1's auc: 0.670884
[20]	training's auc: 0.838744	valid_1's auc: 0.672143
[30]	training's auc: 0.841467	valid_1's auc: 0.67332
[40]	training's auc: 0.844006	valid_1's auc: 0.67439
[50]	training's auc: 0.846297	valid_1's auc: 0.675319
[60]	training's auc: 0.848376	valid_1's auc: 0.67612
[70]	training's auc: 0.850154	valid_1's auc: 0.676751
[80]	training's auc: 0.851699	valid_1's auc: 0.677262
[90]	training's auc: 0.853025	valid_1's auc: 0.677659
[100]	training's auc: 0.854152	valid_1's auc: 0.677949
[110]	training's auc: 0.855093	valid_1's auc: 0.678152
[120]	training's auc: 0.855889	valid_1's auc: 0.678296
[130]	training's auc: 0.856534	valid_1's auc: 0.678374
[140]	training's auc: 0.857088	valid_1's auc: 0.678416
[150]	training's auc: 0.85753	valid_1's auc: 0.67841
[160]	training's auc: 0.857868	valid_1's auc: 0.678402
[170]	training's auc: 0.858238	valid_1's auc: 0.678346
[180]	training's auc: 0.858459	valid_1's auc: 0.678286
[190]	training's auc: 0.858644	valid_1's auc: 0.678236
[200]	training's auc: 0.858813	valid_1's auc: 0.678155
[210]	training's auc: 0.858942	valid_1's auc: 0.678086
[220]	training's auc: 0.859047	valid_1's auc: 0.678015
[230]	training's auc: 0.85913	valid_1's auc: 0.677945
[240]	training's auc: 0.859196	valid_1's auc: 0.677873
[250]	training's auc: 0.859251	valid_1's auc: 0.677805
[260]	training's auc: 0.859297	valid_1's auc: 0.677748
[270]	training's auc: 0.859331	valid_1's auc: 0.677695
[280]	training's auc: 0.85936	valid_1's auc: 0.677638
[290]	training's auc: 0.859383	valid_1's auc: 0.677586
[300]	training's auc: 0.859403	valid_1's auc: 0.677541
[310]	training's auc: 0.859417	valid_1's auc: 0.677498
[320]	training's auc: 0.859429	valid_1's auc: 0.67746
[330]	training's auc: 0.85944	valid_1's auc: 0.677426
[340]	training's auc: 0.859447	valid_1's auc: 0.677392
Early stopping, best iteration is:
[141]	training's auc: 0.857187	valid_1's auc: 0.67842
best score: 0.678419627417
best iteration: 141
complete on: ITC_source_type

working on: CC11_source_type

Our guest selection:
target                uint8
FAKE_1512883008     float64
CC11_source_type      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.835842	valid_1's auc: 0.670864
[20]	training's auc: 0.838747	valid_1's auc: 0.672138
[30]	training's auc: 0.841471	valid_1's auc: 0.673313
[40]	training's auc: 0.844011	valid_1's auc: 0.674385
[50]	training's auc: 0.846297	valid_1's auc: 0.675316
[60]	training's auc: 0.848353	valid_1's auc: 0.676109
[70]	training's auc: 0.850152	valid_1's auc: 0.67675
[80]	training's auc: 0.851691	valid_1's auc: 0.677261
[90]	training's auc: 0.853022	valid_1's auc: 0.677656
[100]	training's auc: 0.854144	valid_1's auc: 0.677946
[110]	training's auc: 0.8551	valid_1's auc: 0.678153
[120]	training's auc: 0.855881	valid_1's auc: 0.678288
[130]	training's auc: 0.856532	valid_1's auc: 0.678373
[140]	training's auc: 0.857096	valid_1's auc: 0.678413
[150]	training's auc: 0.857528	valid_1's auc: 0.67841
[160]	training's auc: 0.857871	valid_1's auc: 0.678393
[170]	training's auc: 0.858238	valid_1's auc: 0.67834
[180]	training's auc: 0.858459	valid_1's auc: 0.678286
[190]	training's auc: 0.858649	valid_1's auc: 0.678224
[200]	training's auc: 0.858811	valid_1's auc: 0.678158
[210]	training's auc: 0.85894	valid_1's auc: 0.678083
[220]	training's auc: 0.859048	valid_1's auc: 0.678014
[230]	training's auc: 0.859134	valid_1's auc: 0.677938
[240]	training's auc: 0.859197	valid_1's auc: 0.677867
[250]	training's auc: 0.859253	valid_1's auc: 0.677798
[260]	training's auc: 0.859299	valid_1's auc: 0.677738
[270]	training's auc: 0.859332	valid_1's auc: 0.677688
[280]	training's auc: 0.85936	valid_1's auc: 0.67763
[290]	training's auc: 0.859383	valid_1's auc: 0.677577
[300]	training's auc: 0.859401	valid_1's auc: 0.677532
[310]	training's auc: 0.859418	valid_1's auc: 0.677488
[320]	training's auc: 0.859431	valid_1's auc: 0.67745
[330]	training's auc: 0.85944	valid_1's auc: 0.677415
[340]	training's auc: 0.859449	valid_1's auc: 0.677383
[350]	training's auc: 0.859455	valid_1's auc: 0.677354
[360]	training's auc: 0.859461	valid_1's auc: 0.677327
Early stopping, best iteration is:
[163]	training's auc: 0.858041	valid_1's auc: 0.678416
best score: 0.678416179311
best iteration: 163
complete on: CC11_source_type

working on: ITC_gender

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_gender           int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857835	valid_1's auc: 0.6782
[20]	training's auc: 0.857881	valid_1's auc: 0.678225
[30]	training's auc: 0.857921	valid_1's auc: 0.678251
[40]	training's auc: 0.857948	valid_1's auc: 0.678266
[50]	training's auc: 0.857968	valid_1's auc: 0.678274
[60]	training's auc: 0.857981	valid_1's auc: 0.678281
[70]	training's auc: 0.857993	valid_1's auc: 0.678288
[80]	training's auc: 0.858001	valid_1's auc: 0.678291
[90]	training's auc: 0.858008	valid_1's auc: 0.678298
[100]	training's auc: 0.858012	valid_1's auc: 0.6783
[110]	training's auc: 0.858016	valid_1's auc: 0.678303
[120]	training's auc: 0.858019	valid_1's auc: 0.678303
[130]	training's auc: 0.858021	valid_1's auc: 0.678305
[140]	training's auc: 0.858022	valid_1's auc: 0.678305
[150]	training's auc: 0.858023	valid_1's auc: 0.678306
[160]	training's auc: 0.858024	valid_1's auc: 0.678306
[170]	training's auc: 0.858025	valid_1's auc: 0.678307
[180]	training's auc: 0.858026	valid_1's auc: 0.678306
[190]	training's auc: 0.858026	valid_1's auc: 0.678305
[200]	training's auc: 0.858029	valid_1's auc: 0.678304
[210]	training's auc: 0.858033	valid_1's auc: 0.678303
[220]	training's auc: 0.858033	valid_1's auc: 0.678304
[230]	training's auc: 0.858034	valid_1's auc: 0.678306
[240]	training's auc: 0.858034	valid_1's auc: 0.678306
[250]	training's auc: 0.858034	valid_1's auc: 0.678306
[260]	training's auc: 0.858034	valid_1's auc: 0.678306
[270]	training's auc: 0.858034	valid_1's auc: 0.678306
[280]	training's auc: 0.858034	valid_1's auc: 0.678306
[290]	training's auc: 0.858034	valid_1's auc: 0.678306
[300]	training's auc: 0.858034	valid_1's auc: 0.678305
[310]	training's auc: 0.858034	valid_1's auc: 0.678305
[320]	training's auc: 0.858034	valid_1's auc: 0.678305
[330]	training's auc: 0.858035	valid_1's auc: 0.678304
[340]	training's auc: 0.858035	valid_1's auc: 0.678305
[350]	training's auc: 0.858035	valid_1's auc: 0.678305
[360]	training's auc: 0.858035	valid_1's auc: 0.678305
[370]	training's auc: 0.858035	valid_1's auc: 0.678305
Early stopping, best iteration is:
[173]	training's auc: 0.858025	valid_1's auc: 0.678308
best score: 0.678307541751
best iteration: 173
complete on: ITC_gender

working on: CC11_gender

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_gender          int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857834	valid_1's auc: 0.678196
[20]	training's auc: 0.857884	valid_1's auc: 0.678227
[30]	training's auc: 0.857918	valid_1's auc: 0.678247
[40]	training's auc: 0.857949	valid_1's auc: 0.678262
[50]	training's auc: 0.857969	valid_1's auc: 0.678276
[60]	training's auc: 0.857982	valid_1's auc: 0.678284
[70]	training's auc: 0.857991	valid_1's auc: 0.678289
[80]	training's auc: 0.858	valid_1's auc: 0.678292
[90]	training's auc: 0.858007	valid_1's auc: 0.678296
[100]	training's auc: 0.858011	valid_1's auc: 0.678299
[110]	training's auc: 0.858015	valid_1's auc: 0.678303
[120]	training's auc: 0.858019	valid_1's auc: 0.678306
[130]	training's auc: 0.858021	valid_1's auc: 0.678308
[140]	training's auc: 0.858022	valid_1's auc: 0.678309
[150]	training's auc: 0.858024	valid_1's auc: 0.678309
[160]	training's auc: 0.858025	valid_1's auc: 0.678308
[170]	training's auc: 0.858026	valid_1's auc: 0.678308
[180]	training's auc: 0.858027	valid_1's auc: 0.678308
[190]	training's auc: 0.858027	valid_1's auc: 0.678308
[200]	training's auc: 0.858033	valid_1's auc: 0.678306
[210]	training's auc: 0.858033	valid_1's auc: 0.678306
[220]	training's auc: 0.858034	valid_1's auc: 0.678307
[230]	training's auc: 0.858035	valid_1's auc: 0.678309
[240]	training's auc: 0.858035	valid_1's auc: 0.678309
[250]	training's auc: 0.858035	valid_1's auc: 0.678309
[260]	training's auc: 0.858035	valid_1's auc: 0.678309
[270]	training's auc: 0.858035	valid_1's auc: 0.67831
[280]	training's auc: 0.858035	valid_1's auc: 0.67831
[290]	training's auc: 0.858035	valid_1's auc: 0.678311
[300]	training's auc: 0.858035	valid_1's auc: 0.67831
[310]	training's auc: 0.858035	valid_1's auc: 0.67831
[320]	training's auc: 0.858035	valid_1's auc: 0.67831
[330]	training's auc: 0.858035	valid_1's auc: 0.67831
[340]	training's auc: 0.858035	valid_1's auc: 0.678309
[350]	training's auc: 0.858035	valid_1's auc: 0.678309
[360]	training's auc: 0.858035	valid_1's auc: 0.678309
[370]	training's auc: 0.858035	valid_1's auc: 0.67831
[380]	training's auc: 0.858036	valid_1's auc: 0.67831
[390]	training's auc: 0.858035	valid_1's auc: 0.67831
[400]	training's auc: 0.858036	valid_1's auc: 0.678309
[410]	training's auc: 0.858035	valid_1's auc: 0.678309
[420]	training's auc: 0.858035	valid_1's auc: 0.678309
[430]	training's auc: 0.858035	valid_1's auc: 0.678309
[440]	training's auc: 0.858035	valid_1's auc: 0.678309
[450]	training's auc: 0.858035	valid_1's auc: 0.678309
[460]	training's auc: 0.858035	valid_1's auc: 0.678309
[470]	training's auc: 0.858035	valid_1's auc: 0.678308
[480]	training's auc: 0.858035	valid_1's auc: 0.678309
[490]	training's auc: 0.858035	valid_1's auc: 0.678309
Early stopping, best iteration is:
[294]	training's auc: 0.858035	valid_1's auc: 0.678311
best score: 0.678310819559
best iteration: 294
complete on: CC11_gender

working on: ITC_artist_name

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_artist_name      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.850001	valid_1's auc: 0.676338
[20]	training's auc: 0.851496	valid_1's auc: 0.676963
[30]	training's auc: 0.852831	valid_1's auc: 0.677502
[40]	training's auc: 0.853926	valid_1's auc: 0.677901
[50]	training's auc: 0.854812	valid_1's auc: 0.678197
[60]	training's auc: 0.855503	valid_1's auc: 0.678405
[70]	training's auc: 0.856055	valid_1's auc: 0.678547
[80]	training's auc: 0.8565	valid_1's auc: 0.678637
[90]	training's auc: 0.856855	valid_1's auc: 0.678686
[100]	training's auc: 0.857136	valid_1's auc: 0.67871
[110]	training's auc: 0.857358	valid_1's auc: 0.678709
[120]	training's auc: 0.857536	valid_1's auc: 0.678695
[130]	training's auc: 0.857678	valid_1's auc: 0.678669
[140]	training's auc: 0.857789	valid_1's auc: 0.678636
[150]	training's auc: 0.857878	valid_1's auc: 0.678601
[160]	training's auc: 0.85795	valid_1's auc: 0.678571
[170]	training's auc: 0.858015	valid_1's auc: 0.678536
[180]	training's auc: 0.858067	valid_1's auc: 0.678498
[190]	training's auc: 0.858105	valid_1's auc: 0.678471
[200]	training's auc: 0.858135	valid_1's auc: 0.678434
[210]	training's auc: 0.858159	valid_1's auc: 0.6784
[220]	training's auc: 0.858179	valid_1's auc: 0.678373
[230]	training's auc: 0.858194	valid_1's auc: 0.678348
[240]	training's auc: 0.858207	valid_1's auc: 0.678325
[250]	training's auc: 0.858216	valid_1's auc: 0.678304
[260]	training's auc: 0.858225	valid_1's auc: 0.678286
[270]	training's auc: 0.858232	valid_1's auc: 0.678266
[280]	training's auc: 0.858237	valid_1's auc: 0.678253
[290]	training's auc: 0.858242	valid_1's auc: 0.67824
[300]	training's auc: 0.858246	valid_1's auc: 0.678226
Early stopping, best iteration is:
[105]	training's auc: 0.857266	valid_1's auc: 0.678713
best score: 0.67871301447
best iteration: 105
complete on: ITC_artist_name

working on: CC11_artist_name

Our guest selection:
target                uint8
FAKE_1512883008     float64
CC11_artist_name      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.850445	valid_1's auc: 0.671489
[20]	training's auc: 0.851925	valid_1's auc: 0.672115
[30]	training's auc: 0.85326	valid_1's auc: 0.672589
[40]	training's auc: 0.854389	valid_1's auc: 0.672915
[50]	training's auc: 0.855301	valid_1's auc: 0.673114
[60]	training's auc: 0.856008	valid_1's auc: 0.673192
[70]	training's auc: 0.856579	valid_1's auc: 0.673208
[80]	training's auc: 0.857038	valid_1's auc: 0.673165
[90]	training's auc: 0.857406	valid_1's auc: 0.673087
[100]	training's auc: 0.857702	valid_1's auc: 0.672976
[110]	training's auc: 0.85794	valid_1's auc: 0.672855
[120]	training's auc: 0.85813	valid_1's auc: 0.672725
[130]	training's auc: 0.858281	valid_1's auc: 0.672596
[140]	training's auc: 0.858402	valid_1's auc: 0.67246
[150]	training's auc: 0.858499	valid_1's auc: 0.672332
[160]	training's auc: 0.858579	valid_1's auc: 0.672215
[170]	training's auc: 0.858648	valid_1's auc: 0.672097
[180]	training's auc: 0.858704	valid_1's auc: 0.671997
[190]	training's auc: 0.858746	valid_1's auc: 0.671908
[200]	training's auc: 0.858777	valid_1's auc: 0.671825
[210]	training's auc: 0.858802	valid_1's auc: 0.671748
[220]	training's auc: 0.858822	valid_1's auc: 0.671684
[230]	training's auc: 0.858839	valid_1's auc: 0.671632
[240]	training's auc: 0.858852	valid_1's auc: 0.671585
[250]	training's auc: 0.858863	valid_1's auc: 0.67154
[260]	training's auc: 0.858871	valid_1's auc: 0.671501
Early stopping, best iteration is:
[68]	training's auc: 0.856474	valid_1's auc: 0.673208
best score: 0.673208140087
best iteration: 68
complete on: CC11_artist_name

working on: ITC_composer

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_composer         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.850697	valid_1's auc: 0.677095
[20]	training's auc: 0.852102	valid_1's auc: 0.677609
[30]	training's auc: 0.853324	valid_1's auc: 0.67801
[40]	training's auc: 0.854335	valid_1's auc: 0.678321
[50]	training's auc: 0.855134	valid_1's auc: 0.678525
[60]	training's auc: 0.855758	valid_1's auc: 0.678668
[70]	training's auc: 0.856255	valid_1's auc: 0.678751
[80]	training's auc: 0.85665	valid_1's auc: 0.6788
[90]	training's auc: 0.856964	valid_1's auc: 0.678817
[100]	training's auc: 0.857213	valid_1's auc: 0.678813
[110]	training's auc: 0.857411	valid_1's auc: 0.678795
[120]	training's auc: 0.857568	valid_1's auc: 0.678773
[130]	training's auc: 0.857694	valid_1's auc: 0.678746
[140]	training's auc: 0.857794	valid_1's auc: 0.678709
[150]	training's auc: 0.857873	valid_1's auc: 0.678675
[160]	training's auc: 0.857937	valid_1's auc: 0.678644
[170]	training's auc: 0.85799	valid_1's auc: 0.678617
[180]	training's auc: 0.858034	valid_1's auc: 0.67858
[190]	training's auc: 0.858069	valid_1's auc: 0.678549
[200]	training's auc: 0.858097	valid_1's auc: 0.678512
[210]	training's auc: 0.858118	valid_1's auc: 0.678485
[220]	training's auc: 0.858136	valid_1's auc: 0.678461
[230]	training's auc: 0.85815	valid_1's auc: 0.678442
[240]	training's auc: 0.858161	valid_1's auc: 0.678424
[250]	training's auc: 0.85817	valid_1's auc: 0.678401
[260]	training's auc: 0.858178	valid_1's auc: 0.678376
[270]	training's auc: 0.858184	valid_1's auc: 0.678361
[280]	training's auc: 0.85819	valid_1's auc: 0.678349
[290]	training's auc: 0.858194	valid_1's auc: 0.678335
Early stopping, best iteration is:
[95]	training's auc: 0.857106	valid_1's auc: 0.678821
best score: 0.678820907539
best iteration: 95
complete on: ITC_composer

working on: CC11_composer

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_composer        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.851509	valid_1's auc: 0.664165
[20]	training's auc: 0.852944	valid_1's auc: 0.664801
[30]	training's auc: 0.854185	valid_1's auc: 0.665229
[40]	training's auc: 0.855189	valid_1's auc: 0.665465
[50]	training's auc: 0.856008	valid_1's auc: 0.66555
[60]	training's auc: 0.856655	valid_1's auc: 0.665526
[70]	training's auc: 0.857174	valid_1's auc: 0.665427
[80]	training's auc: 0.857583	valid_1's auc: 0.665279
[90]	training's auc: 0.857911	valid_1's auc: 0.665094
[100]	training's auc: 0.858172	valid_1's auc: 0.664892
[110]	training's auc: 0.85838	valid_1's auc: 0.664674
[120]	training's auc: 0.858548	valid_1's auc: 0.664461
[130]	training's auc: 0.858684	valid_1's auc: 0.664251
[140]	training's auc: 0.858791	valid_1's auc: 0.664048
[150]	training's auc: 0.858878	valid_1's auc: 0.66386
[160]	training's auc: 0.858947	valid_1's auc: 0.66369
[170]	training's auc: 0.859009	valid_1's auc: 0.663527
[180]	training's auc: 0.859057	valid_1's auc: 0.663378
[190]	training's auc: 0.859097	valid_1's auc: 0.663242
[200]	training's auc: 0.859127	valid_1's auc: 0.663117
[210]	training's auc: 0.859152	valid_1's auc: 0.663016
[220]	training's auc: 0.85917	valid_1's auc: 0.662919
[230]	training's auc: 0.859185	valid_1's auc: 0.662838
[240]	training's auc: 0.859198	valid_1's auc: 0.662761
[250]	training's auc: 0.859209	valid_1's auc: 0.6627
Early stopping, best iteration is:
[52]	training's auc: 0.856148	valid_1's auc: 0.665551
best score: 0.665550732003
best iteration: 52
complete on: CC11_composer

working on: ITC_lyricist

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_lyricist         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.852298	valid_1's auc: 0.677075
[20]	training's auc: 0.853426	valid_1's auc: 0.677538
[30]	training's auc: 0.854371	valid_1's auc: 0.677904
[40]	training's auc: 0.855142	valid_1's auc: 0.678194
[50]	training's auc: 0.855769	valid_1's auc: 0.67838
[60]	training's auc: 0.856251	valid_1's auc: 0.678499
[70]	training's auc: 0.85664	valid_1's auc: 0.678588
[80]	training's auc: 0.856947	valid_1's auc: 0.678633
[90]	training's auc: 0.857189	valid_1's auc: 0.678654
[100]	training's auc: 0.857384	valid_1's auc: 0.678663
[110]	training's auc: 0.857537	valid_1's auc: 0.678654
[120]	training's auc: 0.857659	valid_1's auc: 0.678638
[130]	training's auc: 0.857756	valid_1's auc: 0.678604
[140]	training's auc: 0.857834	valid_1's auc: 0.678575
[150]	training's auc: 0.857894	valid_1's auc: 0.678552
[160]	training's auc: 0.857943	valid_1's auc: 0.678533
[170]	training's auc: 0.857982	valid_1's auc: 0.678506
[180]	training's auc: 0.858022	valid_1's auc: 0.678476
[190]	training's auc: 0.858052	valid_1's auc: 0.678458
[200]	training's auc: 0.858075	valid_1's auc: 0.67841
[210]	training's auc: 0.858093	valid_1's auc: 0.67839
[220]	training's auc: 0.858109	valid_1's auc: 0.678356
[230]	training's auc: 0.858119	valid_1's auc: 0.678335
[240]	training's auc: 0.85813	valid_1's auc: 0.67832
[250]	training's auc: 0.858138	valid_1's auc: 0.678303
[260]	training's auc: 0.858146	valid_1's auc: 0.67829
[270]	training's auc: 0.858152	valid_1's auc: 0.678276
[280]	training's auc: 0.858157	valid_1's auc: 0.678255
[290]	training's auc: 0.858162	valid_1's auc: 0.678244
Early stopping, best iteration is:
[99]	training's auc: 0.857374	valid_1's auc: 0.678666
best score: 0.678665785936
best iteration: 99
complete on: ITC_lyricist

working on: CC11_lyricist

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_lyricist        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.852316	valid_1's auc: 0.665723
[20]	training's auc: 0.853499	valid_1's auc: 0.666266
[30]	training's auc: 0.85452	valid_1's auc: 0.666649
[40]	training's auc: 0.85536	valid_1's auc: 0.666887
[50]	training's auc: 0.856018	valid_1's auc: 0.667005
[60]	training's auc: 0.856543	valid_1's auc: 0.667028
[70]	training's auc: 0.856958	valid_1's auc: 0.666982
[80]	training's auc: 0.857286	valid_1's auc: 0.666906
[90]	training's auc: 0.857548	valid_1's auc: 0.666793
[100]	training's auc: 0.857754	valid_1's auc: 0.666665
[110]	training's auc: 0.857921	valid_1's auc: 0.666525
[120]	training's auc: 0.858053	valid_1's auc: 0.666385
[130]	training's auc: 0.858162	valid_1's auc: 0.666244
[140]	training's auc: 0.858247	valid_1's auc: 0.666112
[150]	training's auc: 0.858315	valid_1's auc: 0.665987
[160]	training's auc: 0.85837	valid_1's auc: 0.665871
[170]	training's auc: 0.858415	valid_1's auc: 0.665766
[180]	training's auc: 0.858455	valid_1's auc: 0.665675
[190]	training's auc: 0.858491	valid_1's auc: 0.665584
[200]	training's auc: 0.858517	valid_1's auc: 0.665496
[210]	training's auc: 0.858537	valid_1's auc: 0.665422
[220]	training's auc: 0.858553	valid_1's auc: 0.66536
[230]	training's auc: 0.858566	valid_1's auc: 0.665305
[240]	training's auc: 0.858575	valid_1's auc: 0.665258
[250]	training's auc: 0.858584	valid_1's auc: 0.665212
Early stopping, best iteration is:
[54]	training's auc: 0.856245	valid_1's auc: 0.66703
best score: 0.667029626265
best iteration: 54
complete on: CC11_lyricist

working on: ITC_language

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_language         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855485	valid_1's auc: 0.677536
[20]	training's auc: 0.855981	valid_1's auc: 0.677776
[30]	training's auc: 0.856433	valid_1's auc: 0.677977
[40]	training's auc: 0.856792	valid_1's auc: 0.678133
[50]	training's auc: 0.857075	valid_1's auc: 0.67824
[60]	training's auc: 0.857302	valid_1's auc: 0.678316
[70]	training's auc: 0.857473	valid_1's auc: 0.678364
[80]	training's auc: 0.857608	valid_1's auc: 0.678388
[90]	training's auc: 0.857711	valid_1's auc: 0.678395
[100]	training's auc: 0.857792	valid_1's auc: 0.678394
[110]	training's auc: 0.857857	valid_1's auc: 0.678383
[120]	training's auc: 0.857909	valid_1's auc: 0.678372
[130]	training's auc: 0.857948	valid_1's auc: 0.678353
[140]	training's auc: 0.857974	valid_1's auc: 0.678341
[150]	training's auc: 0.857995	valid_1's auc: 0.678326
[160]	training's auc: 0.858011	valid_1's auc: 0.678308
[170]	training's auc: 0.85806	valid_1's auc: 0.678273
[180]	training's auc: 0.858067	valid_1's auc: 0.678265
[190]	training's auc: 0.858075	valid_1's auc: 0.678253
[200]	training's auc: 0.858081	valid_1's auc: 0.678243
[210]	training's auc: 0.858085	valid_1's auc: 0.678229
[220]	training's auc: 0.85809	valid_1's auc: 0.678218
[230]	training's auc: 0.858093	valid_1's auc: 0.678208
[240]	training's auc: 0.858096	valid_1's auc: 0.678199
[250]	training's auc: 0.858098	valid_1's auc: 0.678191
[260]	training's auc: 0.858099	valid_1's auc: 0.678184
[270]	training's auc: 0.858101	valid_1's auc: 0.678179
[280]	training's auc: 0.858102	valid_1's auc: 0.678172
[290]	training's auc: 0.858102	valid_1's auc: 0.678169
Early stopping, best iteration is:
[92]	training's auc: 0.857728	valid_1's auc: 0.678397
best score: 0.678396797018
best iteration: 92
complete on: ITC_language

working on: CC11_language

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_language        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855491	valid_1's auc: 0.677543
[20]	training's auc: 0.856004	valid_1's auc: 0.677787
[30]	training's auc: 0.856444	valid_1's auc: 0.677986
[40]	training's auc: 0.856802	valid_1's auc: 0.678133
[50]	training's auc: 0.857073	valid_1's auc: 0.678241
[60]	training's auc: 0.857304	valid_1's auc: 0.67832
[70]	training's auc: 0.857472	valid_1's auc: 0.678363
[80]	training's auc: 0.857612	valid_1's auc: 0.678387
[90]	training's auc: 0.857716	valid_1's auc: 0.678395
[100]	training's auc: 0.857794	valid_1's auc: 0.678396
[110]	training's auc: 0.85786	valid_1's auc: 0.678385
[120]	training's auc: 0.857907	valid_1's auc: 0.678374
[130]	training's auc: 0.857948	valid_1's auc: 0.678362
[140]	training's auc: 0.857976	valid_1's auc: 0.678344
[150]	training's auc: 0.857996	valid_1's auc: 0.678333
[160]	training's auc: 0.858011	valid_1's auc: 0.678314
[170]	training's auc: 0.858058	valid_1's auc: 0.678281
[180]	training's auc: 0.858067	valid_1's auc: 0.678269
[190]	training's auc: 0.858076	valid_1's auc: 0.678254
[200]	training's auc: 0.858082	valid_1's auc: 0.678242
[210]	training's auc: 0.858086	valid_1's auc: 0.67823
[220]	training's auc: 0.85809	valid_1's auc: 0.678219
[230]	training's auc: 0.858093	valid_1's auc: 0.678209
[240]	training's auc: 0.858096	valid_1's auc: 0.678201
[250]	training's auc: 0.858099	valid_1's auc: 0.678193
[260]	training's auc: 0.858101	valid_1's auc: 0.678187
[270]	training's auc: 0.858101	valid_1's auc: 0.678182
[280]	training's auc: 0.858102	valid_1's auc: 0.678176
[290]	training's auc: 0.858103	valid_1's auc: 0.678171
Early stopping, best iteration is:
[94]	training's auc: 0.857749	valid_1's auc: 0.6784
best score: 0.678399539644
best iteration: 94
complete on: CC11_language

working on: ITC_name

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_name             int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.848164	valid_1's auc: 0.6788
[20]	training's auc: 0.849871	valid_1's auc: 0.679196
[30]	training's auc: 0.851389	valid_1's auc: 0.679489
[40]	training's auc: 0.852699	valid_1's auc: 0.679696
[50]	training's auc: 0.853797	valid_1's auc: 0.679818
[60]	training's auc: 0.854695	valid_1's auc: 0.679866
[70]	training's auc: 0.855416	valid_1's auc: 0.67986
[80]	training's auc: 0.855993	valid_1's auc: 0.679815
[90]	training's auc: 0.856452	valid_1's auc: 0.679744
[100]	training's auc: 0.856817	valid_1's auc: 0.679657
[110]	training's auc: 0.857107	valid_1's auc: 0.679561
[120]	training's auc: 0.857337	valid_1's auc: 0.679463
[130]	training's auc: 0.857521	valid_1's auc: 0.679361
[140]	training's auc: 0.857666	valid_1's auc: 0.679263
[150]	training's auc: 0.857782	valid_1's auc: 0.679169
[160]	training's auc: 0.857876	valid_1's auc: 0.679082
[170]	training's auc: 0.857949	valid_1's auc: 0.679006
[180]	training's auc: 0.858011	valid_1's auc: 0.678933
[190]	training's auc: 0.858059	valid_1's auc: 0.678853
[200]	training's auc: 0.858097	valid_1's auc: 0.678773
[210]	training's auc: 0.858128	valid_1's auc: 0.678708
[220]	training's auc: 0.858153	valid_1's auc: 0.678652
[230]	training's auc: 0.858172	valid_1's auc: 0.678602
[240]	training's auc: 0.858187	valid_1's auc: 0.678559
[250]	training's auc: 0.858198	valid_1's auc: 0.678519
[260]	training's auc: 0.858208	valid_1's auc: 0.678486
Early stopping, best iteration is:
[63]	training's auc: 0.854889	valid_1's auc: 0.679872
best score: 0.679872423395
best iteration: 63
complete on: ITC_name

working on: CC11_name

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_name            int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.851446	valid_1's auc: 0.662031
[20]	training's auc: 0.853119	valid_1's auc: 0.662601
[30]	training's auc: 0.854621	valid_1's auc: 0.66293
[40]	training's auc: 0.855932	valid_1's auc: 0.66309
[50]	training's auc: 0.857037	valid_1's auc: 0.663099
[60]	training's auc: 0.857945	valid_1's auc: 0.663
[70]	training's auc: 0.858682	valid_1's auc: 0.66282
[80]	training's auc: 0.859274	valid_1's auc: 0.662585
[90]	training's auc: 0.85975	valid_1's auc: 0.662316
[100]	training's auc: 0.860133	valid_1's auc: 0.662033
[110]	training's auc: 0.860441	valid_1's auc: 0.661739
[120]	training's auc: 0.860686	valid_1's auc: 0.661449
[130]	training's auc: 0.860886	valid_1's auc: 0.661171
[140]	training's auc: 0.861048	valid_1's auc: 0.660904
[150]	training's auc: 0.861177	valid_1's auc: 0.660657
[160]	training's auc: 0.861277	valid_1's auc: 0.660431
[170]	training's auc: 0.861362	valid_1's auc: 0.660212
[180]	training's auc: 0.861433	valid_1's auc: 0.660012
[190]	training's auc: 0.861489	valid_1's auc: 0.659826
[200]	training's auc: 0.861531	valid_1's auc: 0.659669
[210]	training's auc: 0.861565	valid_1's auc: 0.659521
[220]	training's auc: 0.861592	valid_1's auc: 0.65939
[230]	training's auc: 0.861613	valid_1's auc: 0.659274
[240]	training's auc: 0.86163	valid_1's auc: 0.659173
Early stopping, best iteration is:
[44]	training's auc: 0.8564	valid_1's auc: 0.66311
best score: 0.663110437157
best iteration: 44
complete on: CC11_name

working on: ITC_song_year

Our guest selection:
target               uint8
FAKE_1512883008    float64
ITC_song_year        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855454	valid_1's auc: 0.679165
[20]	training's auc: 0.85598	valid_1's auc: 0.679264
[30]	training's auc: 0.856415	valid_1's auc: 0.679316
[40]	training's auc: 0.856772	valid_1's auc: 0.67932
[50]	training's auc: 0.857057	valid_1's auc: 0.679293
[60]	training's auc: 0.85728	valid_1's auc: 0.679233
[70]	training's auc: 0.857453	valid_1's auc: 0.679164
[80]	training's auc: 0.85759	valid_1's auc: 0.679079
[90]	training's auc: 0.857698	valid_1's auc: 0.678981
[100]	training's auc: 0.857783	valid_1's auc: 0.678907
[110]	training's auc: 0.857849	valid_1's auc: 0.678823
[120]	training's auc: 0.857901	valid_1's auc: 0.678738
[130]	training's auc: 0.857941	valid_1's auc: 0.678656
[140]	training's auc: 0.857971	valid_1's auc: 0.678595
[150]	training's auc: 0.858001	valid_1's auc: 0.678551
[160]	training's auc: 0.858041	valid_1's auc: 0.678475
[170]	training's auc: 0.858063	valid_1's auc: 0.678427
[180]	training's auc: 0.858075	valid_1's auc: 0.678391
[190]	training's auc: 0.858084	valid_1's auc: 0.678322
[200]	training's auc: 0.858092	valid_1's auc: 0.678265
[210]	training's auc: 0.858099	valid_1's auc: 0.678242
[220]	training's auc: 0.858104	valid_1's auc: 0.678194
[230]	training's auc: 0.858108	valid_1's auc: 0.678179
[240]	training's auc: 0.858112	valid_1's auc: 0.678153
Early stopping, best iteration is:
[41]	training's auc: 0.856759	valid_1's auc: 0.67933
best score: 0.679329703696
best iteration: 41
complete on: ITC_song_year

working on: CC11_song_year

Our guest selection:
target               uint8
FAKE_1512883008    float64
CC11_song_year       int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855468	valid_1's auc: 0.67917
[20]	training's auc: 0.855986	valid_1's auc: 0.67927
[30]	training's auc: 0.856426	valid_1's auc: 0.679312
[40]	training's auc: 0.856778	valid_1's auc: 0.67931
[50]	training's auc: 0.857062	valid_1's auc: 0.679288
[60]	training's auc: 0.857285	valid_1's auc: 0.679241
[70]	training's auc: 0.857459	valid_1's auc: 0.679159
[80]	training's auc: 0.857595	valid_1's auc: 0.679073
[90]	training's auc: 0.857701	valid_1's auc: 0.678988
[100]	training's auc: 0.857785	valid_1's auc: 0.678903
[110]	training's auc: 0.857852	valid_1's auc: 0.678821
[120]	training's auc: 0.857904	valid_1's auc: 0.678734
[130]	training's auc: 0.857944	valid_1's auc: 0.678662
[140]	training's auc: 0.857972	valid_1's auc: 0.678598
[150]	training's auc: 0.858004	valid_1's auc: 0.678548
[160]	training's auc: 0.858043	valid_1's auc: 0.678478
[170]	training's auc: 0.858065	valid_1's auc: 0.678425
[180]	training's auc: 0.858077	valid_1's auc: 0.678384
[190]	training's auc: 0.858087	valid_1's auc: 0.67832
[200]	training's auc: 0.858094	valid_1's auc: 0.678261
[210]	training's auc: 0.858101	valid_1's auc: 0.67824
[220]	training's auc: 0.858106	valid_1's auc: 0.678194
[230]	training's auc: 0.858111	valid_1's auc: 0.678175
Early stopping, best iteration is:
[35]	training's auc: 0.856538	valid_1's auc: 0.679318
best score: 0.67931828406
best iteration: 35
complete on: CC11_song_year

working on: ITC_song_country

Our guest selection:
target                uint8
FAKE_1512883008     float64
ITC_song_country      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855318	valid_1's auc: 0.677232
[20]	training's auc: 0.855858	valid_1's auc: 0.677505
[30]	training's auc: 0.856315	valid_1's auc: 0.677753
[40]	training's auc: 0.856703	valid_1's auc: 0.677947
[50]	training's auc: 0.857005	valid_1's auc: 0.678083
[60]	training's auc: 0.857246	valid_1's auc: 0.678185
[70]	training's auc: 0.85743	valid_1's auc: 0.678252
[80]	training's auc: 0.857578	valid_1's auc: 0.678297
[90]	training's auc: 0.857688	valid_1's auc: 0.678325
[100]	training's auc: 0.857777	valid_1's auc: 0.678347
[110]	training's auc: 0.857842	valid_1's auc: 0.678351
[120]	training's auc: 0.857898	valid_1's auc: 0.67835
[130]	training's auc: 0.85794	valid_1's auc: 0.67835
[140]	training's auc: 0.857971	valid_1's auc: 0.678348
[150]	training's auc: 0.857995	valid_1's auc: 0.67834
[160]	training's auc: 0.858033	valid_1's auc: 0.678323
[170]	training's auc: 0.858057	valid_1's auc: 0.678308
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1002.py", line 225, in <module>
    verbose_eval=verbose_eval,
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/engine.py", line 199, in train
    booster.update(fobj=fobj)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1507, in update
    ctypes.byref(is_finished)))
KeyboardInterrupt

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''
