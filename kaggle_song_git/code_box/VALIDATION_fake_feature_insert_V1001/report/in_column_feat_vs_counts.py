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
dff = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

# barebone = True
barebone = False
if barebone:
    ccc = [i for i in dff.columns]
    ccc.remove('target')
    dff.drop(ccc, axis=1, inplace=True)

# dff = dff[['target','membership_days', 'bd_log10']]
# must be a fake feature
inner = [
    'FAKE_[]_0.6788_Light_gbdt_1512883008.csv'
]
# inner = False


def insert_this(on):
    global dff
    on = on[:-4]
    dff1 = pd.read_csv('../saves/feature/'+on+'.csv')
    dff1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    dff1.rename(columns={'target': 'FAKE_'+on}, inplace=True)
    # print(dff1.head(10))
    dff = dff.join(dff1)
    del dff1


if inner:
    for i in inner:
        insert_this(i)

print('What we got:')
print(dff.dtypes)
print('number of rows:', len(dff))
print('number of columns:', len(dff.columns))

num_boost_round = 5000
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
for w in dff.columns:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df = dff[toto]

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype('category')

        print()
        print('Our guest selection:')
        print(df.dtypes)
        print('number of columns:', len(df.columns))
        print()

        length = len(df)
        train_size = 0.76
        train_set = df.head(int(length*train_size))
        val_set = df.drop(train_set.index)
        del df

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


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/in_column_train_V1001.py
What we got:
target                          uint8
membership_days                 int64
bd_log10                      float64
expiration_month_log10        float64
IMC_expiration_month_log10    float64
bd_fixed_log10                float64
age_guess_log10               float64
bd_range_log10                float64
age_guess_range_log10         float64
bd_fixed_range_log10          float64
IMC_bd_log10                  float64
IMC_bd_fixed_log10            float64
IMC_age_guess_log10           float64
IMC_bd_range_log10            float64
IMC_bd_fixed_range_log10      float64
IMC_age_guess_range_log10     float64
IMC_membership_days_log10     float64
song_year                       int64
ISC_genre_ids                   int64
ISC_top1_in_song                int64
ISC_top2_in_song                int64
ISC_top3_in_song                int64
ISCZ_artist_name                int64
ISC_composer                    int64
ISCZ_lyricist                   int64
ISC_language                    int64
ISCZ_rc                         int64
ISCZ_isrc_rest                  int64
ISC_song_year                   int64
ISCZ_song_year                  int64
song_length_log10             float64
ISCZ_genre_ids_log10          float64
ISC_artist_name_log10         float64
ISCZ_composer_log10           float64
ISC_lyricist_log10            float64
ISC_name_log10                float64
ISCZ_name_ln                  float64
ISC_song_country_ln           float64
ISCZ_song_country_log10       float64
ISC_rc_ln                     float64
ISC_isrc_rest_log10           float64
FAKE_1512883008               float64
dtype: object
number of rows: 7377418
number of columns: 42
working on: membership_days

Our guest selection:
target               uint8
FAKE_1512883008    float64
membership_days      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855524	valid_1's auc: 0.676868
[20]	training's auc: 0.856011	valid_1's auc: 0.677247
[30]	training's auc: 0.856432	valid_1's auc: 0.677561
[40]	training's auc: 0.85678	valid_1's auc: 0.677819
[50]	training's auc: 0.857058	valid_1's auc: 0.678022
[60]	training's auc: 0.857273	valid_1's auc: 0.67816
[70]	training's auc: 0.857447	valid_1's auc: 0.678276
[80]	training's auc: 0.857588	valid_1's auc: 0.678357
[90]	training's auc: 0.8577	valid_1's auc: 0.678412
[100]	training's auc: 0.857786	valid_1's auc: 0.678455
[110]	training's auc: 0.857856	valid_1's auc: 0.678482
[120]	training's auc: 0.857911	valid_1's auc: 0.678501
[130]	training's auc: 0.857953	valid_1's auc: 0.678511
[140]	training's auc: 0.857986	valid_1's auc: 0.678513
[150]	training's auc: 0.858013	valid_1's auc: 0.678517
[160]	training's auc: 0.858035	valid_1's auc: 0.678513
[170]	training's auc: 0.858053	valid_1's auc: 0.678506
[180]	training's auc: 0.858067	valid_1's auc: 0.678504
[190]	training's auc: 0.858078	valid_1's auc: 0.678501
[200]	training's auc: 0.858086	valid_1's auc: 0.678494
[210]	training's auc: 0.858093	valid_1's auc: 0.678489
[220]	training's auc: 0.858099	valid_1's auc: 0.678488
[230]	training's auc: 0.858103	valid_1's auc: 0.678484
[240]	training's auc: 0.858106	valid_1's auc: 0.678479
[250]	training's auc: 0.858109	valid_1's auc: 0.678477
[260]	training's auc: 0.858111	valid_1's auc: 0.678475
[270]	training's auc: 0.858113	valid_1's auc: 0.678473
[280]	training's auc: 0.858115	valid_1's auc: 0.678471
[290]	training's auc: 0.858116	valid_1's auc: 0.67847
[300]	training's auc: 0.858117	valid_1's auc: 0.678467
[310]	training's auc: 0.858118	valid_1's auc: 0.678464
[320]	training's auc: 0.858118	valid_1's auc: 0.678461
[330]	training's auc: 0.858119	valid_1's auc: 0.678461
[340]	training's auc: 0.85812	valid_1's auc: 0.678458
[350]	training's auc: 0.85812	valid_1's auc: 0.678459
Early stopping, best iteration is:
[153]	training's auc: 0.858022	valid_1's auc: 0.678518
best score: 0.678518360679
best iteration: 153
complete on: membership_days

working on: bd_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
bd_log10           float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856485	valid_1's auc: 0.677753
[20]	training's auc: 0.856811	valid_1's auc: 0.677919
[30]	training's auc: 0.857078	valid_1's auc: 0.678037
[40]	training's auc: 0.857302	valid_1's auc: 0.678138
[50]	training's auc: 0.857477	valid_1's auc: 0.678207
[60]	training's auc: 0.857612	valid_1's auc: 0.678258
[70]	training's auc: 0.857719	valid_1's auc: 0.678294
[80]	training's auc: 0.857799	valid_1's auc: 0.678313
[90]	training's auc: 0.857863	valid_1's auc: 0.678322
[100]	training's auc: 0.857911	valid_1's auc: 0.678326
[110]	training's auc: 0.857948	valid_1's auc: 0.678331
[120]	training's auc: 0.857976	valid_1's auc: 0.678332
[130]	training's auc: 0.857998	valid_1's auc: 0.67833
[140]	training's auc: 0.858014	valid_1's auc: 0.678325
[150]	training's auc: 0.858029	valid_1's auc: 0.678319
[160]	training's auc: 0.858045	valid_1's auc: 0.678314
[170]	training's auc: 0.858055	valid_1's auc: 0.678312
[180]	training's auc: 0.858062	valid_1's auc: 0.678307
[190]	training's auc: 0.858069	valid_1's auc: 0.678296
[200]	training's auc: 0.858072	valid_1's auc: 0.678293
[210]	training's auc: 0.858074	valid_1's auc: 0.678288
[220]	training's auc: 0.858077	valid_1's auc: 0.678283
[230]	training's auc: 0.858078	valid_1's auc: 0.678279
[240]	training's auc: 0.858079	valid_1's auc: 0.678275
[250]	training's auc: 0.85808	valid_1's auc: 0.678271
[260]	training's auc: 0.85808	valid_1's auc: 0.678267
[270]	training's auc: 0.858081	valid_1's auc: 0.678262
[280]	training's auc: 0.858082	valid_1's auc: 0.678258
[290]	training's auc: 0.858082	valid_1's auc: 0.678256
[300]	training's auc: 0.858082	valid_1's auc: 0.678255
[310]	training's auc: 0.858082	valid_1's auc: 0.678254
Early stopping, best iteration is:
[119]	training's auc: 0.857975	valid_1's auc: 0.678332
best score: 0.678331922299
best iteration: 119
complete on: bd_log10

working on: expiration_month_log10

Our guest selection:
target                      uint8
FAKE_1512883008           float64
expiration_month_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856827	valid_1's auc: 0.678237
[20]	training's auc: 0.857082	valid_1's auc: 0.678313
[30]	training's auc: 0.857295	valid_1's auc: 0.678369
[40]	training's auc: 0.857467	valid_1's auc: 0.678402
[50]	training's auc: 0.857599	valid_1's auc: 0.678419
[60]	training's auc: 0.857703	valid_1's auc: 0.678426
[70]	training's auc: 0.857782	valid_1's auc: 0.678422
[80]	training's auc: 0.857845	valid_1's auc: 0.678415
[90]	training's auc: 0.857894	valid_1's auc: 0.6784
[100]	training's auc: 0.85793	valid_1's auc: 0.678386
[110]	training's auc: 0.857958	valid_1's auc: 0.678375
[120]	training's auc: 0.85798	valid_1's auc: 0.678365
[130]	training's auc: 0.857995	valid_1's auc: 0.67835
[140]	training's auc: 0.858008	valid_1's auc: 0.67834
[150]	training's auc: 0.858017	valid_1's auc: 0.678331
[160]	training's auc: 0.858028	valid_1's auc: 0.678311
[170]	training's auc: 0.858041	valid_1's auc: 0.678289
[180]	training's auc: 0.858047	valid_1's auc: 0.678273
[190]	training's auc: 0.85805	valid_1's auc: 0.678266
[200]	training's auc: 0.858053	valid_1's auc: 0.67826
[210]	training's auc: 0.858055	valid_1's auc: 0.678259
[220]	training's auc: 0.858057	valid_1's auc: 0.678256
[230]	training's auc: 0.858058	valid_1's auc: 0.67825
[240]	training's auc: 0.858059	valid_1's auc: 0.678246
[250]	training's auc: 0.85806	valid_1's auc: 0.678242
[260]	training's auc: 0.85806	valid_1's auc: 0.678238
Early stopping, best iteration is:
[61]	training's auc: 0.857708	valid_1's auc: 0.678426
best score: 0.678426182425
best iteration: 61
complete on: expiration_month_log10

working on: IMC_expiration_month_log10

Our guest selection:
target                          uint8
FAKE_1512883008               float64
IMC_expiration_month_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856828	valid_1's auc: 0.678238
[20]	training's auc: 0.857077	valid_1's auc: 0.678314
[30]	training's auc: 0.857295	valid_1's auc: 0.67837
[40]	training's auc: 0.857456	valid_1's auc: 0.678399
[50]	training's auc: 0.857597	valid_1's auc: 0.67842
[60]	training's auc: 0.857698	valid_1's auc: 0.678424
[70]	training's auc: 0.857779	valid_1's auc: 0.678418
[80]	training's auc: 0.857846	valid_1's auc: 0.678412
[90]	training's auc: 0.85789	valid_1's auc: 0.678401
[100]	training's auc: 0.857926	valid_1's auc: 0.678388
[110]	training's auc: 0.857956	valid_1's auc: 0.678373
[120]	training's auc: 0.857978	valid_1's auc: 0.678364
[130]	training's auc: 0.857994	valid_1's auc: 0.678351
[140]	training's auc: 0.858007	valid_1's auc: 0.678342
[150]	training's auc: 0.858017	valid_1's auc: 0.678328
[160]	training's auc: 0.858026	valid_1's auc: 0.678312
[170]	training's auc: 0.858038	valid_1's auc: 0.678289
[180]	training's auc: 0.858044	valid_1's auc: 0.678272
[190]	training's auc: 0.858047	valid_1's auc: 0.678266
[200]	training's auc: 0.85805	valid_1's auc: 0.678259
[210]	training's auc: 0.858053	valid_1's auc: 0.678259
[220]	training's auc: 0.858055	valid_1's auc: 0.678253
[230]	training's auc: 0.858056	valid_1's auc: 0.678249
[240]	training's auc: 0.858057	valid_1's auc: 0.678244
[250]	training's auc: 0.858058	valid_1's auc: 0.678239
[260]	training's auc: 0.858058	valid_1's auc: 0.678235
Early stopping, best iteration is:
[60]	training's auc: 0.857698	valid_1's auc: 0.678424
best score: 0.678424316512
best iteration: 60
complete on: IMC_expiration_month_log10

working on: bd_fixed_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
bd_fixed_log10     float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856619	valid_1's auc: 0.677808
[20]	training's auc: 0.856916	valid_1's auc: 0.677952
[30]	training's auc: 0.857173	valid_1's auc: 0.678073
[40]	training's auc: 0.857375	valid_1's auc: 0.678164
[50]	training's auc: 0.857531	valid_1's auc: 0.678222
[60]	training's auc: 0.857657	valid_1's auc: 0.678265
[70]	training's auc: 0.857752	valid_1's auc: 0.678294
[80]	training's auc: 0.857824	valid_1's auc: 0.678314
[90]	training's auc: 0.85788	valid_1's auc: 0.678322
[100]	training's auc: 0.857922	valid_1's auc: 0.678322
[110]	training's auc: 0.857955	valid_1's auc: 0.678323
[120]	training's auc: 0.857979	valid_1's auc: 0.678322
[130]	training's auc: 0.857997	valid_1's auc: 0.67832
[140]	training's auc: 0.858011	valid_1's auc: 0.678315
[150]	training's auc: 0.858026	valid_1's auc: 0.678309
[160]	training's auc: 0.858039	valid_1's auc: 0.678302
[170]	training's auc: 0.858049	valid_1's auc: 0.678303
[180]	training's auc: 0.858055	valid_1's auc: 0.678298
[190]	training's auc: 0.85806	valid_1's auc: 0.678286
[200]	training's auc: 0.858063	valid_1's auc: 0.678282
[210]	training's auc: 0.858064	valid_1's auc: 0.678277
[220]	training's auc: 0.858066	valid_1's auc: 0.678274
[230]	training's auc: 0.858067	valid_1's auc: 0.678269
[240]	training's auc: 0.858068	valid_1's auc: 0.678265
[250]	training's auc: 0.858068	valid_1's auc: 0.678261
[260]	training's auc: 0.858069	valid_1's auc: 0.678255
[270]	training's auc: 0.858069	valid_1's auc: 0.678252
[280]	training's auc: 0.85807	valid_1's auc: 0.67825
[290]	training's auc: 0.85807	valid_1's auc: 0.678248
Early stopping, best iteration is:
[91]	training's auc: 0.857886	valid_1's auc: 0.678323
best score: 0.678323457558
best iteration: 91
complete on: bd_fixed_log10

working on: age_guess_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
age_guess_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856842	valid_1's auc: 0.677828
[20]	training's auc: 0.8571	valid_1's auc: 0.677958
[30]	training's auc: 0.857317	valid_1's auc: 0.678065
[40]	training's auc: 0.857488	valid_1's auc: 0.678149
[50]	training's auc: 0.857621	valid_1's auc: 0.67821
[60]	training's auc: 0.857723	valid_1's auc: 0.678254
[70]	training's auc: 0.857802	valid_1's auc: 0.678285
[80]	training's auc: 0.85786	valid_1's auc: 0.678304
[90]	training's auc: 0.857906	valid_1's auc: 0.678314
[100]	training's auc: 0.857941	valid_1's auc: 0.678321
[110]	training's auc: 0.857967	valid_1's auc: 0.678325
[120]	training's auc: 0.857988	valid_1's auc: 0.678325
[130]	training's auc: 0.858004	valid_1's auc: 0.678326
[140]	training's auc: 0.858018	valid_1's auc: 0.678326
[150]	training's auc: 0.858029	valid_1's auc: 0.678318
[160]	training's auc: 0.85804	valid_1's auc: 0.678315
[170]	training's auc: 0.858049	valid_1's auc: 0.678315
[180]	training's auc: 0.858054	valid_1's auc: 0.678312
[190]	training's auc: 0.858057	valid_1's auc: 0.678308
[200]	training's auc: 0.85806	valid_1's auc: 0.678304
[210]	training's auc: 0.858061	valid_1's auc: 0.678302
[220]	training's auc: 0.858062	valid_1's auc: 0.678297
[230]	training's auc: 0.858064	valid_1's auc: 0.678294
[240]	training's auc: 0.858064	valid_1's auc: 0.678292
[250]	training's auc: 0.858064	valid_1's auc: 0.67829
[260]	training's auc: 0.858065	valid_1's auc: 0.678289
[270]	training's auc: 0.858065	valid_1's auc: 0.678288
[280]	training's auc: 0.858065	valid_1's auc: 0.678286
[290]	training's auc: 0.858066	valid_1's auc: 0.678285
[300]	training's auc: 0.858066	valid_1's auc: 0.678283
[310]	training's auc: 0.858066	valid_1's auc: 0.678283
[320]	training's auc: 0.858066	valid_1's auc: 0.678283
[330]	training's auc: 0.858066	valid_1's auc: 0.678282
Early stopping, best iteration is:
[132]	training's auc: 0.858008	valid_1's auc: 0.678326
best score: 0.678326130207
best iteration: 132
complete on: age_guess_log10

working on: bd_range_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
bd_range_log10     float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857041	valid_1's auc: 0.677975
[20]	training's auc: 0.857259	valid_1's auc: 0.678085
[30]	training's auc: 0.857444	valid_1's auc: 0.678161
[40]	training's auc: 0.857586	valid_1's auc: 0.678222
[50]	training's auc: 0.857693	valid_1's auc: 0.678263
[60]	training's auc: 0.857776	valid_1's auc: 0.678284
[70]	training's auc: 0.857841	valid_1's auc: 0.678302
[80]	training's auc: 0.857887	valid_1's auc: 0.678314
[90]	training's auc: 0.857922	valid_1's auc: 0.678315
[100]	training's auc: 0.857952	valid_1's auc: 0.678318
[110]	training's auc: 0.857972	valid_1's auc: 0.678318
[120]	training's auc: 0.857987	valid_1's auc: 0.678317
[130]	training's auc: 0.857999	valid_1's auc: 0.678315
[140]	training's auc: 0.858009	valid_1's auc: 0.678314
[150]	training's auc: 0.858017	valid_1's auc: 0.678308
[160]	training's auc: 0.858029	valid_1's auc: 0.678303
[170]	training's auc: 0.858039	valid_1's auc: 0.678294
[180]	training's auc: 0.858047	valid_1's auc: 0.67829
[190]	training's auc: 0.858049	valid_1's auc: 0.678288
[200]	training's auc: 0.85805	valid_1's auc: 0.678284
[210]	training's auc: 0.858051	valid_1's auc: 0.67828
[220]	training's auc: 0.858052	valid_1's auc: 0.678278
[230]	training's auc: 0.858054	valid_1's auc: 0.678274
[240]	training's auc: 0.858054	valid_1's auc: 0.678272
[250]	training's auc: 0.858055	valid_1's auc: 0.67827
[260]	training's auc: 0.858055	valid_1's auc: 0.678269
[270]	training's auc: 0.858055	valid_1's auc: 0.678267
[280]	training's auc: 0.858055	valid_1's auc: 0.678266
[290]	training's auc: 0.858055	valid_1's auc: 0.678265
[300]	training's auc: 0.858055	valid_1's auc: 0.678264
[310]	training's auc: 0.858055	valid_1's auc: 0.678263
Early stopping, best iteration is:
[113]	training's auc: 0.857979	valid_1's auc: 0.678318
best score: 0.678318367582
best iteration: 113
complete on: bd_range_log10

working on: age_guess_range_log10

Our guest selection:
target                     uint8
FAKE_1512883008          float64
age_guess_range_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857278	valid_1's auc: 0.67805
[20]	training's auc: 0.857438	valid_1's auc: 0.678132
[30]	training's auc: 0.857579	valid_1's auc: 0.678196
[40]	training's auc: 0.857686	valid_1's auc: 0.678239
[50]	training's auc: 0.857771	valid_1's auc: 0.678274
[60]	training's auc: 0.857838	valid_1's auc: 0.678298
[70]	training's auc: 0.857886	valid_1's auc: 0.678313
[80]	training's auc: 0.857921	valid_1's auc: 0.678322
[90]	training's auc: 0.85795	valid_1's auc: 0.678322
[100]	training's auc: 0.85797	valid_1's auc: 0.678321
[110]	training's auc: 0.857984	valid_1's auc: 0.678319
[120]	training's auc: 0.857998	valid_1's auc: 0.678313
[130]	training's auc: 0.858006	valid_1's auc: 0.678314
[140]	training's auc: 0.858012	valid_1's auc: 0.678313
[150]	training's auc: 0.858016	valid_1's auc: 0.678313
[160]	training's auc: 0.858033	valid_1's auc: 0.67831
[170]	training's auc: 0.858038	valid_1's auc: 0.678304
[180]	training's auc: 0.858044	valid_1's auc: 0.678294
[190]	training's auc: 0.858045	valid_1's auc: 0.678293
[200]	training's auc: 0.858046	valid_1's auc: 0.678291
[210]	training's auc: 0.858046	valid_1's auc: 0.678287
[220]	training's auc: 0.858047	valid_1's auc: 0.678286
[230]	training's auc: 0.858048	valid_1's auc: 0.678282
[240]	training's auc: 0.858048	valid_1's auc: 0.678279
[250]	training's auc: 0.858049	valid_1's auc: 0.678276
[260]	training's auc: 0.858049	valid_1's auc: 0.678274
[270]	training's auc: 0.858049	valid_1's auc: 0.678273
[280]	training's auc: 0.858049	valid_1's auc: 0.678273
[290]	training's auc: 0.858049	valid_1's auc: 0.678274
Early stopping, best iteration is:
[96]	training's auc: 0.857962	valid_1's auc: 0.678323
best score: 0.678322704119
best iteration: 96
complete on: age_guess_range_log10

working on: bd_fixed_range_log10

Our guest selection:
target                    uint8
FAKE_1512883008         float64
bd_fixed_range_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857051	valid_1's auc: 0.677984
[20]	training's auc: 0.857255	valid_1's auc: 0.678082
[30]	training's auc: 0.857434	valid_1's auc: 0.67816
[40]	training's auc: 0.857576	valid_1's auc: 0.678213
[50]	training's auc: 0.857688	valid_1's auc: 0.678254
[60]	training's auc: 0.857775	valid_1's auc: 0.678278
[70]	training's auc: 0.857839	valid_1's auc: 0.678299
[80]	training's auc: 0.857888	valid_1's auc: 0.678309
[90]	training's auc: 0.857924	valid_1's auc: 0.678317
[100]	training's auc: 0.857951	valid_1's auc: 0.678318
[110]	training's auc: 0.857973	valid_1's auc: 0.678316
[120]	training's auc: 0.857988	valid_1's auc: 0.678318
[130]	training's auc: 0.857999	valid_1's auc: 0.678316
[140]	training's auc: 0.858011	valid_1's auc: 0.678313
[150]	training's auc: 0.858017	valid_1's auc: 0.67831
[160]	training's auc: 0.85803	valid_1's auc: 0.678304
[170]	training's auc: 0.858038	valid_1's auc: 0.678294
[180]	training's auc: 0.858048	valid_1's auc: 0.678294
[190]	training's auc: 0.85805	valid_1's auc: 0.678292
[200]	training's auc: 0.858051	valid_1's auc: 0.678289
[210]	training's auc: 0.858053	valid_1's auc: 0.678286
[220]	training's auc: 0.858054	valid_1's auc: 0.678285
[230]	training's auc: 0.858054	valid_1's auc: 0.678283
[240]	training's auc: 0.858055	valid_1's auc: 0.67828
[250]	training's auc: 0.858055	valid_1's auc: 0.678277
[260]	training's auc: 0.858056	valid_1's auc: 0.678277
[270]	training's auc: 0.858056	valid_1's auc: 0.678275
[280]	training's auc: 0.858056	valid_1's auc: 0.678274
[290]	training's auc: 0.858056	valid_1's auc: 0.678272
Early stopping, best iteration is:
[99]	training's auc: 0.85795	valid_1's auc: 0.678319
best score: 0.678318531259
best iteration: 99
complete on: bd_fixed_range_log10

working on: IMC_bd_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
IMC_bd_log10       float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856761	valid_1's auc: 0.677848
[20]	training's auc: 0.857035	valid_1's auc: 0.677993
[30]	training's auc: 0.857265	valid_1's auc: 0.678103
[40]	training's auc: 0.857447	valid_1's auc: 0.678181
[50]	training's auc: 0.857585	valid_1's auc: 0.678238
[60]	training's auc: 0.857693	valid_1's auc: 0.678275
[70]	training's auc: 0.857777	valid_1's auc: 0.678298
[80]	training's auc: 0.857841	valid_1's auc: 0.678313
[90]	training's auc: 0.857893	valid_1's auc: 0.678322
[100]	training's auc: 0.857931	valid_1's auc: 0.678321
[110]	training's auc: 0.857959	valid_1's auc: 0.678322
[120]	training's auc: 0.857981	valid_1's auc: 0.678322
[130]	training's auc: 0.857998	valid_1's auc: 0.67832
[140]	training's auc: 0.858012	valid_1's auc: 0.678314
[150]	training's auc: 0.858026	valid_1's auc: 0.67831
[160]	training's auc: 0.858039	valid_1's auc: 0.678304
[170]	training's auc: 0.858049	valid_1's auc: 0.678303
[180]	training's auc: 0.858055	valid_1's auc: 0.678299
[190]	training's auc: 0.85806	valid_1's auc: 0.67829
[200]	training's auc: 0.858063	valid_1's auc: 0.678286
[210]	training's auc: 0.858065	valid_1's auc: 0.678281
[220]	training's auc: 0.858066	valid_1's auc: 0.678279
[230]	training's auc: 0.858068	valid_1's auc: 0.678276
[240]	training's auc: 0.858068	valid_1's auc: 0.678274
[250]	training's auc: 0.858069	valid_1's auc: 0.67827
[260]	training's auc: 0.858069	valid_1's auc: 0.678264
[270]	training's auc: 0.85807	valid_1's auc: 0.678261
[280]	training's auc: 0.85807	valid_1's auc: 0.678261
[290]	training's auc: 0.85807	valid_1's auc: 0.678258
Early stopping, best iteration is:
[91]	training's auc: 0.857898	valid_1's auc: 0.678323
best score: 0.678323298962
best iteration: 91
complete on: IMC_bd_log10

working on: IMC_bd_fixed_log10

Our guest selection:
target                  uint8
FAKE_1512883008       float64
IMC_bd_fixed_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856807	valid_1's auc: 0.677851
[20]	training's auc: 0.857077	valid_1's auc: 0.678001
[30]	training's auc: 0.857299	valid_1's auc: 0.678106
[40]	training's auc: 0.857466	valid_1's auc: 0.678182
[50]	training's auc: 0.857605	valid_1's auc: 0.678241
[60]	training's auc: 0.85771	valid_1's auc: 0.678278
[70]	training's auc: 0.857789	valid_1's auc: 0.678304
[80]	training's auc: 0.857851	valid_1's auc: 0.678319
[90]	training's auc: 0.857897	valid_1's auc: 0.678323
[100]	training's auc: 0.857935	valid_1's auc: 0.678323
[110]	training's auc: 0.857963	valid_1's auc: 0.678323
[120]	training's auc: 0.857984	valid_1's auc: 0.678323
[130]	training's auc: 0.858	valid_1's auc: 0.678319
[140]	training's auc: 0.858013	valid_1's auc: 0.678314
[150]	training's auc: 0.858026	valid_1's auc: 0.678309
[160]	training's auc: 0.858037	valid_1's auc: 0.678303
[170]	training's auc: 0.858047	valid_1's auc: 0.678302
[180]	training's auc: 0.858053	valid_1's auc: 0.678296
[190]	training's auc: 0.858058	valid_1's auc: 0.678285
[200]	training's auc: 0.858061	valid_1's auc: 0.678279
[210]	training's auc: 0.858063	valid_1's auc: 0.678275
[220]	training's auc: 0.858064	valid_1's auc: 0.678272
[230]	training's auc: 0.858065	valid_1's auc: 0.678268
[240]	training's auc: 0.858066	valid_1's auc: 0.678266
[250]	training's auc: 0.858066	valid_1's auc: 0.678264
[260]	training's auc: 0.858067	valid_1's auc: 0.678259
[270]	training's auc: 0.858067	valid_1's auc: 0.678256
[280]	training's auc: 0.858067	valid_1's auc: 0.678253
[290]	training's auc: 0.858068	valid_1's auc: 0.67825
[300]	training's auc: 0.858068	valid_1's auc: 0.678248
[310]	training's auc: 0.858068	valid_1's auc: 0.678245
Early stopping, best iteration is:
[116]	training's auc: 0.857977	valid_1's auc: 0.678325
best score: 0.678325074431
best iteration: 116
complete on: IMC_bd_fixed_log10

working on: IMC_age_guess_log10

Our guest selection:
target                   uint8
FAKE_1512883008        float64
IMC_age_guess_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856865	valid_1's auc: 0.677796
[20]	training's auc: 0.857118	valid_1's auc: 0.677932
[30]	training's auc: 0.85733	valid_1's auc: 0.678047
[40]	training's auc: 0.857496	valid_1's auc: 0.678132
[50]	training's auc: 0.857628	valid_1's auc: 0.678197
[60]	training's auc: 0.857726	valid_1's auc: 0.678243
[70]	training's auc: 0.857803	valid_1's auc: 0.678274
[80]	training's auc: 0.857862	valid_1's auc: 0.678293
[90]	training's auc: 0.857907	valid_1's auc: 0.678304
[100]	training's auc: 0.857942	valid_1's auc: 0.678313
[110]	training's auc: 0.857968	valid_1's auc: 0.678316
[120]	training's auc: 0.857989	valid_1's auc: 0.678317
[130]	training's auc: 0.858005	valid_1's auc: 0.678316
[140]	training's auc: 0.858018	valid_1's auc: 0.678318
[150]	training's auc: 0.85803	valid_1's auc: 0.678314
[160]	training's auc: 0.858041	valid_1's auc: 0.678318
[170]	training's auc: 0.858051	valid_1's auc: 0.678313
[180]	training's auc: 0.858055	valid_1's auc: 0.678308
[190]	training's auc: 0.858059	valid_1's auc: 0.678304
[200]	training's auc: 0.858061	valid_1's auc: 0.678303
[210]	training's auc: 0.858063	valid_1's auc: 0.6783
[220]	training's auc: 0.858065	valid_1's auc: 0.678296
[230]	training's auc: 0.858066	valid_1's auc: 0.678293
[240]	training's auc: 0.858067	valid_1's auc: 0.67829
[250]	training's auc: 0.858067	valid_1's auc: 0.678286
[260]	training's auc: 0.858067	valid_1's auc: 0.678283
[270]	training's auc: 0.858068	valid_1's auc: 0.678282
[280]	training's auc: 0.858068	valid_1's auc: 0.678282
[290]	training's auc: 0.858068	valid_1's auc: 0.67828
[300]	training's auc: 0.858068	valid_1's auc: 0.678279
[310]	training's auc: 0.858068	valid_1's auc: 0.678277
[320]	training's auc: 0.858068	valid_1's auc: 0.678276
[330]	training's auc: 0.858068	valid_1's auc: 0.678275
[340]	training's auc: 0.858068	valid_1's auc: 0.678274
[350]	training's auc: 0.858068	valid_1's auc: 0.678272
[360]	training's auc: 0.858069	valid_1's auc: 0.678272
Early stopping, best iteration is:
[163]	training's auc: 0.858045	valid_1's auc: 0.678319
best score: 0.678319300627
best iteration: 163
complete on: IMC_age_guess_log10

working on: IMC_bd_range_log10

Our guest selection:
target                  uint8
FAKE_1512883008       float64
IMC_bd_range_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857044	valid_1's auc: 0.677975
[20]	training's auc: 0.85726	valid_1's auc: 0.67808
[30]	training's auc: 0.85744	valid_1's auc: 0.678158
[40]	training's auc: 0.857585	valid_1's auc: 0.678216
[50]	training's auc: 0.857692	valid_1's auc: 0.678254
[60]	training's auc: 0.857776	valid_1's auc: 0.678278
[70]	training's auc: 0.857837	valid_1's auc: 0.678299
[80]	training's auc: 0.857889	valid_1's auc: 0.678307
[90]	training's auc: 0.857923	valid_1's auc: 0.678312
[100]	training's auc: 0.857953	valid_1's auc: 0.678313
[110]	training's auc: 0.857974	valid_1's auc: 0.678314
[120]	training's auc: 0.857989	valid_1's auc: 0.678312
[130]	training's auc: 0.858	valid_1's auc: 0.678312
[140]	training's auc: 0.858011	valid_1's auc: 0.678313
[150]	training's auc: 0.858018	valid_1's auc: 0.678308
[160]	training's auc: 0.85803	valid_1's auc: 0.678304
[170]	training's auc: 0.85804	valid_1's auc: 0.678294
[180]	training's auc: 0.858049	valid_1's auc: 0.678291
[190]	training's auc: 0.85805	valid_1's auc: 0.678288
[200]	training's auc: 0.858052	valid_1's auc: 0.678287
[210]	training's auc: 0.858053	valid_1's auc: 0.678285
[220]	training's auc: 0.858054	valid_1's auc: 0.678281
[230]	training's auc: 0.858055	valid_1's auc: 0.678277
[240]	training's auc: 0.858055	valid_1's auc: 0.678274
[250]	training's auc: 0.858056	valid_1's auc: 0.678272
[260]	training's auc: 0.858056	valid_1's auc: 0.678269
[270]	training's auc: 0.858056	valid_1's auc: 0.678268
[280]	training's auc: 0.858056	valid_1's auc: 0.678266
[290]	training's auc: 0.858056	valid_1's auc: 0.678264
Early stopping, best iteration is:
[98]	training's auc: 0.857948	valid_1's auc: 0.678315
best score: 0.678315104346
best iteration: 98
complete on: IMC_bd_range_log10

working on: IMC_bd_fixed_range_log10

Our guest selection:
target                        uint8
FAKE_1512883008             float64
IMC_bd_fixed_range_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.85705	valid_1's auc: 0.67797
[20]	training's auc: 0.857262	valid_1's auc: 0.678076
[30]	training's auc: 0.857443	valid_1's auc: 0.678156
[40]	training's auc: 0.857584	valid_1's auc: 0.678208
[50]	training's auc: 0.857689	valid_1's auc: 0.67825
[60]	training's auc: 0.857775	valid_1's auc: 0.678275
[70]	training's auc: 0.857839	valid_1's auc: 0.678294
[80]	training's auc: 0.857886	valid_1's auc: 0.678307
[90]	training's auc: 0.857923	valid_1's auc: 0.678313
[100]	training's auc: 0.857952	valid_1's auc: 0.678311
[110]	training's auc: 0.857974	valid_1's auc: 0.678313
[120]	training's auc: 0.857988	valid_1's auc: 0.678313
[130]	training's auc: 0.857999	valid_1's auc: 0.678314
[140]	training's auc: 0.858011	valid_1's auc: 0.678312
[150]	training's auc: 0.858017	valid_1's auc: 0.678309
[160]	training's auc: 0.85803	valid_1's auc: 0.678305
[170]	training's auc: 0.858038	valid_1's auc: 0.678294
[180]	training's auc: 0.858047	valid_1's auc: 0.678291
[190]	training's auc: 0.858049	valid_1's auc: 0.67829
[200]	training's auc: 0.858051	valid_1's auc: 0.678287
[210]	training's auc: 0.858052	valid_1's auc: 0.678284
[220]	training's auc: 0.858052	valid_1's auc: 0.67828
[230]	training's auc: 0.858053	valid_1's auc: 0.678277
[240]	training's auc: 0.858054	valid_1's auc: 0.678274
[250]	training's auc: 0.858054	valid_1's auc: 0.678272
[260]	training's auc: 0.858055	valid_1's auc: 0.67827
[270]	training's auc: 0.858055	valid_1's auc: 0.67827
[280]	training's auc: 0.858055	valid_1's auc: 0.678268
[290]	training's auc: 0.858055	valid_1's auc: 0.678266
[300]	training's auc: 0.858055	valid_1's auc: 0.678264
[310]	training's auc: 0.858054	valid_1's auc: 0.678261
[320]	training's auc: 0.858054	valid_1's auc: 0.67826
[330]	training's auc: 0.858054	valid_1's auc: 0.678259
Early stopping, best iteration is:
[135]	training's auc: 0.858007	valid_1's auc: 0.678314
best score: 0.678314446116
best iteration: 135
complete on: IMC_bd_fixed_range_log10

working on: IMC_age_guess_range_log10

Our guest selection:
target                         uint8
FAKE_1512883008              float64
IMC_age_guess_range_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.857278	valid_1's auc: 0.678039
[20]	training's auc: 0.857442	valid_1's auc: 0.678134
[30]	training's auc: 0.857581	valid_1's auc: 0.678199
[40]	training's auc: 0.857687	valid_1's auc: 0.678241
[50]	training's auc: 0.857772	valid_1's auc: 0.678275
[60]	training's auc: 0.857835	valid_1's auc: 0.678294
[70]	training's auc: 0.857885	valid_1's auc: 0.678305
[80]	training's auc: 0.857922	valid_1's auc: 0.678315
[90]	training's auc: 0.85795	valid_1's auc: 0.678318
[100]	training's auc: 0.857971	valid_1's auc: 0.678322
[110]	training's auc: 0.857986	valid_1's auc: 0.67832
[120]	training's auc: 0.857998	valid_1's auc: 0.678313
[130]	training's auc: 0.858006	valid_1's auc: 0.67831
[140]	training's auc: 0.858012	valid_1's auc: 0.678309
[150]	training's auc: 0.858016	valid_1's auc: 0.678308
[160]	training's auc: 0.858033	valid_1's auc: 0.678309
[170]	training's auc: 0.858039	valid_1's auc: 0.678306
[180]	training's auc: 0.858044	valid_1's auc: 0.678294
[190]	training's auc: 0.858045	valid_1's auc: 0.678291
[200]	training's auc: 0.858046	valid_1's auc: 0.678289
[210]	training's auc: 0.858047	valid_1's auc: 0.678288
[220]	training's auc: 0.858048	valid_1's auc: 0.678285
[230]	training's auc: 0.858048	valid_1's auc: 0.678283
[240]	training's auc: 0.858049	valid_1's auc: 0.678278
[250]	training's auc: 0.858049	valid_1's auc: 0.678276
[260]	training's auc: 0.858049	valid_1's auc: 0.678276
[270]	training's auc: 0.858049	valid_1's auc: 0.678274
[280]	training's auc: 0.85805	valid_1's auc: 0.678274
[290]	training's auc: 0.85805	valid_1's auc: 0.678272
Early stopping, best iteration is:
[98]	training's auc: 0.857967	valid_1's auc: 0.678322
best score: 0.678322408093
best iteration: 98
complete on: IMC_age_guess_range_log10

working on: IMC_membership_days_log10

Our guest selection:
target                         uint8
FAKE_1512883008              float64
IMC_membership_days_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856471	valid_1's auc: 0.677907
[20]	training's auc: 0.856755	valid_1's auc: 0.678085
[30]	training's auc: 0.857003	valid_1's auc: 0.678245
[40]	training's auc: 0.85722	valid_1's auc: 0.678376
[50]	training's auc: 0.8574	valid_1's auc: 0.678476
[60]	training's auc: 0.857547	valid_1's auc: 0.67854
[70]	training's auc: 0.857668	valid_1's auc: 0.678589
[80]	training's auc: 0.857763	valid_1's auc: 0.678613
[90]	training's auc: 0.857837	valid_1's auc: 0.67862
[100]	training's auc: 0.857894	valid_1's auc: 0.678617
[110]	training's auc: 0.85794	valid_1's auc: 0.678609
[120]	training's auc: 0.857977	valid_1's auc: 0.678597
[130]	training's auc: 0.858004	valid_1's auc: 0.678583
[140]	training's auc: 0.858024	valid_1's auc: 0.678566
[150]	training's auc: 0.858041	valid_1's auc: 0.678551
[160]	training's auc: 0.858055	valid_1's auc: 0.678535
[170]	training's auc: 0.858065	valid_1's auc: 0.678521
[180]	training's auc: 0.858071	valid_1's auc: 0.67851
[190]	training's auc: 0.858076	valid_1's auc: 0.678493
[200]	training's auc: 0.85808	valid_1's auc: 0.678481
[210]	training's auc: 0.858083	valid_1's auc: 0.678472
[220]	training's auc: 0.858085	valid_1's auc: 0.678462
[230]	training's auc: 0.858086	valid_1's auc: 0.678453
[240]	training's auc: 0.858087	valid_1's auc: 0.678446
[250]	training's auc: 0.858088	valid_1's auc: 0.678434
[260]	training's auc: 0.858089	valid_1's auc: 0.678429
[270]	training's auc: 0.858089	valid_1's auc: 0.678426
[280]	training's auc: 0.858089	valid_1's auc: 0.678421
[290]	training's auc: 0.858089	valid_1's auc: 0.678418
Early stopping, best iteration is:
[92]	training's auc: 0.85785	valid_1's auc: 0.678624
best score: 0.678624369088
best iteration: 92
complete on: IMC_membership_days_log10

working on: song_year

Our guest selection:
target               uint8
FAKE_1512883008    float64
song_year            int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855458	valid_1's auc: 0.679186
[20]	training's auc: 0.855976	valid_1's auc: 0.679261
[30]	training's auc: 0.856419	valid_1's auc: 0.679305
[40]	training's auc: 0.856776	valid_1's auc: 0.679313
[50]	training's auc: 0.857063	valid_1's auc: 0.679281
[60]	training's auc: 0.857283	valid_1's auc: 0.679214
[70]	training's auc: 0.857456	valid_1's auc: 0.679152
[80]	training's auc: 0.857594	valid_1's auc: 0.679069
[90]	training's auc: 0.8577	valid_1's auc: 0.67899
[100]	training's auc: 0.857785	valid_1's auc: 0.678887
[110]	training's auc: 0.857851	valid_1's auc: 0.678815
[120]	training's auc: 0.857903	valid_1's auc: 0.678724
[130]	training's auc: 0.857943	valid_1's auc: 0.678644
[140]	training's auc: 0.857973	valid_1's auc: 0.678584
[150]	training's auc: 0.858003	valid_1's auc: 0.678545
[160]	training's auc: 0.858044	valid_1's auc: 0.67847
[170]	training's auc: 0.858065	valid_1's auc: 0.678425
[180]	training's auc: 0.858077	valid_1's auc: 0.678389
[190]	training's auc: 0.858086	valid_1's auc: 0.678334
[200]	training's auc: 0.858094	valid_1's auc: 0.67826
[210]	training's auc: 0.8581	valid_1's auc: 0.678236
[220]	training's auc: 0.858105	valid_1's auc: 0.678182
[230]	training's auc: 0.85811	valid_1's auc: 0.678157
Early stopping, best iteration is:
[35]	training's auc: 0.856532	valid_1's auc: 0.679322
best score: 0.679322358459
best iteration: 35
complete on: song_year

working on: ISC_genre_ids

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_genre_ids        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854669	valid_1's auc: 0.676993
[20]	training's auc: 0.855311	valid_1's auc: 0.677313
[30]	training's auc: 0.85587	valid_1's auc: 0.677608
[40]	training's auc: 0.856333	valid_1's auc: 0.677834
[50]	training's auc: 0.856713	valid_1's auc: 0.678018
[60]	training's auc: 0.857008	valid_1's auc: 0.678134
[70]	training's auc: 0.857242	valid_1's auc: 0.678219
[80]	training's auc: 0.857431	valid_1's auc: 0.678282
[90]	training's auc: 0.85758	valid_1's auc: 0.678315
[100]	training's auc: 0.857696	valid_1's auc: 0.678338
[110]	training's auc: 0.857788	valid_1's auc: 0.678344
[120]	training's auc: 0.857861	valid_1's auc: 0.678346
[130]	training's auc: 0.857917	valid_1's auc: 0.678344
[140]	training's auc: 0.85796	valid_1's auc: 0.678331
[150]	training's auc: 0.857995	valid_1's auc: 0.67832
[160]	training's auc: 0.858036	valid_1's auc: 0.678249
[170]	training's auc: 0.858067	valid_1's auc: 0.678245
[180]	training's auc: 0.858086	valid_1's auc: 0.678233
[190]	training's auc: 0.858099	valid_1's auc: 0.678219
[200]	training's auc: 0.85811	valid_1's auc: 0.678205
[210]	training's auc: 0.858119	valid_1's auc: 0.678191
[220]	training's auc: 0.858126	valid_1's auc: 0.678176
[230]	training's auc: 0.858132	valid_1's auc: 0.678187
[240]	training's auc: 0.858137	valid_1's auc: 0.678178
[250]	training's auc: 0.85814	valid_1's auc: 0.678172
[260]	training's auc: 0.858143	valid_1's auc: 0.678163
[270]	training's auc: 0.858145	valid_1's auc: 0.678157
[280]	training's auc: 0.858147	valid_1's auc: 0.678152
[290]	training's auc: 0.858148	valid_1's auc: 0.678146
[300]	training's auc: 0.858149	valid_1's auc: 0.678141
[310]	training's auc: 0.85815	valid_1's auc: 0.678134
Early stopping, best iteration is:
[112]	training's auc: 0.857804	valid_1's auc: 0.678349
best score: 0.678348873864
best iteration: 112
complete on: ISC_genre_ids

working on: ISC_top1_in_song

Our guest selection:
target                uint8
FAKE_1512883008     float64
ISC_top1_in_song      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855119	valid_1's auc: 0.677154
[20]	training's auc: 0.855675	valid_1's auc: 0.677446
[30]	training's auc: 0.856149	valid_1's auc: 0.677698
[40]	training's auc: 0.856546	valid_1's auc: 0.677903
[50]	training's auc: 0.856876	valid_1's auc: 0.678071
[60]	training's auc: 0.857142	valid_1's auc: 0.678175
[70]	training's auc: 0.857346	valid_1's auc: 0.678251
[80]	training's auc: 0.85751	valid_1's auc: 0.678308
[90]	training's auc: 0.857637	valid_1's auc: 0.678333
[100]	training's auc: 0.857738	valid_1's auc: 0.678351
[110]	training's auc: 0.857819	valid_1's auc: 0.678356
[120]	training's auc: 0.857884	valid_1's auc: 0.678339
[130]	training's auc: 0.857933	valid_1's auc: 0.678335
[140]	training's auc: 0.857971	valid_1's auc: 0.678326
[150]	training's auc: 0.858001	valid_1's auc: 0.678308
[160]	training's auc: 0.858039	valid_1's auc: 0.678234
[170]	training's auc: 0.858066	valid_1's auc: 0.67823
[180]	training's auc: 0.858081	valid_1's auc: 0.678224
[190]	training's auc: 0.858093	valid_1's auc: 0.678211
[200]	training's auc: 0.858103	valid_1's auc: 0.678198
[210]	training's auc: 0.85811	valid_1's auc: 0.678185
[220]	training's auc: 0.858116	valid_1's auc: 0.678176
[230]	training's auc: 0.85812	valid_1's auc: 0.678164
[240]	training's auc: 0.858124	valid_1's auc: 0.678155
[250]	training's auc: 0.858127	valid_1's auc: 0.678141
[260]	training's auc: 0.85813	valid_1's auc: 0.678158
[270]	training's auc: 0.858132	valid_1's auc: 0.678149
[280]	training's auc: 0.858133	valid_1's auc: 0.678143
[290]	training's auc: 0.858134	valid_1's auc: 0.678135
[300]	training's auc: 0.858135	valid_1's auc: 0.678129
Early stopping, best iteration is:
[107]	training's auc: 0.857801	valid_1's auc: 0.678357
best score: 0.678357038556
best iteration: 107
complete on: ISC_top1_in_song

working on: ISC_top2_in_song

Our guest selection:
target                uint8
FAKE_1512883008     float64
ISC_top2_in_song      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854855	valid_1's auc: 0.677195
[20]	training's auc: 0.855469	valid_1's auc: 0.677488
[30]	training's auc: 0.855988	valid_1's auc: 0.677738
[40]	training's auc: 0.85643	valid_1's auc: 0.677955
[50]	training's auc: 0.856786	valid_1's auc: 0.678109
[60]	training's auc: 0.857071	valid_1's auc: 0.678206
[70]	training's auc: 0.857291	valid_1's auc: 0.678282
[80]	training's auc: 0.857468	valid_1's auc: 0.678332
[90]	training's auc: 0.85761	valid_1's auc: 0.678357
[100]	training's auc: 0.857719	valid_1's auc: 0.678367
[110]	training's auc: 0.857807	valid_1's auc: 0.678372
[120]	training's auc: 0.857875	valid_1's auc: 0.678359
[130]	training's auc: 0.857928	valid_1's auc: 0.678348
[140]	training's auc: 0.857968	valid_1's auc: 0.678338
[150]	training's auc: 0.857999	valid_1's auc: 0.678321
[160]	training's auc: 0.858044	valid_1's auc: 0.678235
[170]	training's auc: 0.858069	valid_1's auc: 0.678226
[180]	training's auc: 0.858087	valid_1's auc: 0.678211
[190]	training's auc: 0.8581	valid_1's auc: 0.678204
[200]	training's auc: 0.85811	valid_1's auc: 0.678195
[210]	training's auc: 0.858118	valid_1's auc: 0.678187
[220]	training's auc: 0.858125	valid_1's auc: 0.678172
[230]	training's auc: 0.85813	valid_1's auc: 0.678158
[240]	training's auc: 0.858135	valid_1's auc: 0.678171
[250]	training's auc: 0.858138	valid_1's auc: 0.678164
[260]	training's auc: 0.858141	valid_1's auc: 0.678153
[270]	training's auc: 0.858142	valid_1's auc: 0.678144
[280]	training's auc: 0.858144	valid_1's auc: 0.678137
[290]	training's auc: 0.858145	valid_1's auc: 0.67813
[300]	training's auc: 0.858145	valid_1's auc: 0.678125
Early stopping, best iteration is:
[109]	training's auc: 0.857805	valid_1's auc: 0.678373
best score: 0.678372764534
best iteration: 109
complete on: ISC_top2_in_song

working on: ISC_top3_in_song

Our guest selection:
target                uint8
FAKE_1512883008     float64
ISC_top3_in_song      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855184	valid_1's auc: 0.677129
[20]	training's auc: 0.855741	valid_1's auc: 0.677425
[30]	training's auc: 0.856207	valid_1's auc: 0.677678
[40]	training's auc: 0.856601	valid_1's auc: 0.677881
[50]	training's auc: 0.856912	valid_1's auc: 0.678041
[60]	training's auc: 0.857167	valid_1's auc: 0.678143
[70]	training's auc: 0.85737	valid_1's auc: 0.678213
[80]	training's auc: 0.857527	valid_1's auc: 0.678269
[90]	training's auc: 0.85765	valid_1's auc: 0.678302
[100]	training's auc: 0.857748	valid_1's auc: 0.678322
[110]	training's auc: 0.857828	valid_1's auc: 0.678328
[120]	training's auc: 0.857889	valid_1's auc: 0.678335
[130]	training's auc: 0.857938	valid_1's auc: 0.678335
[140]	training's auc: 0.857973	valid_1's auc: 0.678326
[150]	training's auc: 0.858002	valid_1's auc: 0.678316
[160]	training's auc: 0.85804	valid_1's auc: 0.678245
[170]	training's auc: 0.858064	valid_1's auc: 0.678242
[180]	training's auc: 0.858078	valid_1's auc: 0.67823
[190]	training's auc: 0.858089	valid_1's auc: 0.678221
[200]	training's auc: 0.8581	valid_1's auc: 0.678214
[210]	training's auc: 0.858107	valid_1's auc: 0.678203
[220]	training's auc: 0.858112	valid_1's auc: 0.678193
[230]	training's auc: 0.858118	valid_1's auc: 0.678202
[240]	training's auc: 0.858121	valid_1's auc: 0.678192
[250]	training's auc: 0.858124	valid_1's auc: 0.678183
[260]	training's auc: 0.858126	valid_1's auc: 0.678178
[270]	training's auc: 0.858128	valid_1's auc: 0.678171
[280]	training's auc: 0.858129	valid_1's auc: 0.678169
[290]	training's auc: 0.85813	valid_1's auc: 0.678162
[300]	training's auc: 0.85813	valid_1's auc: 0.678158
[310]	training's auc: 0.858131	valid_1's auc: 0.678155
[320]	training's auc: 0.858132	valid_1's auc: 0.678151
Early stopping, best iteration is:
[125]	training's auc: 0.857919	valid_1's auc: 0.678336
best score: 0.678336423949
best iteration: 125
complete on: ISC_top3_in_song

working on: ISCZ_artist_name

Our guest selection:
target                uint8
FAKE_1512883008     float64
ISCZ_artist_name      int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.853647	valid_1's auc: 0.676171
[20]	training's auc: 0.854504	valid_1's auc: 0.676651
[30]	training's auc: 0.855247	valid_1's auc: 0.67708
[40]	training's auc: 0.855862	valid_1's auc: 0.677412
[50]	training's auc: 0.856343	valid_1's auc: 0.677684
[60]	training's auc: 0.856687	valid_1's auc: 0.677879
[70]	training's auc: 0.856967	valid_1's auc: 0.678019
[80]	training's auc: 0.857197	valid_1's auc: 0.678125
[90]	training's auc: 0.857382	valid_1's auc: 0.678203
[100]	training's auc: 0.857528	valid_1's auc: 0.67826
[110]	training's auc: 0.857648	valid_1's auc: 0.678292
[120]	training's auc: 0.857739	valid_1's auc: 0.678321
[130]	training's auc: 0.857814	valid_1's auc: 0.678332
[140]	training's auc: 0.857873	valid_1's auc: 0.678334
[150]	training's auc: 0.85792	valid_1's auc: 0.678338
[160]	training's auc: 0.857959	valid_1's auc: 0.678339
[170]	training's auc: 0.857994	valid_1's auc: 0.678331
[180]	training's auc: 0.858025	valid_1's auc: 0.67832
[190]	training's auc: 0.85805	valid_1's auc: 0.678314
[200]	training's auc: 0.858069	valid_1's auc: 0.678309
[210]	training's auc: 0.858083	valid_1's auc: 0.678302
[220]	training's auc: 0.858097	valid_1's auc: 0.678295
[230]	training's auc: 0.858107	valid_1's auc: 0.678289
[240]	training's auc: 0.858117	valid_1's auc: 0.678278
[250]	training's auc: 0.858125	valid_1's auc: 0.678275
[260]	training's auc: 0.858132	valid_1's auc: 0.678264
[270]	training's auc: 0.858138	valid_1's auc: 0.678255
[280]	training's auc: 0.858143	valid_1's auc: 0.678249
[290]	training's auc: 0.858147	valid_1's auc: 0.678242
[300]	training's auc: 0.858151	valid_1's auc: 0.678235
[310]	training's auc: 0.858154	valid_1's auc: 0.67823
[320]	training's auc: 0.858157	valid_1's auc: 0.678225
[330]	training's auc: 0.85816	valid_1's auc: 0.67822
[340]	training's auc: 0.858163	valid_1's auc: 0.678214
[350]	training's auc: 0.858165	valid_1's auc: 0.678206
Early stopping, best iteration is:
[157]	training's auc: 0.857951	valid_1's auc: 0.67834
best score: 0.678340120589
best iteration: 157
complete on: ISCZ_artist_name

working on: ISC_composer

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_composer         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854813	valid_1's auc: 0.677055
[20]	training's auc: 0.855472	valid_1's auc: 0.677462
[30]	training's auc: 0.855991	valid_1's auc: 0.677719
[40]	training's auc: 0.85643	valid_1's auc: 0.677921
[50]	training's auc: 0.856775	valid_1's auc: 0.67806
[60]	training's auc: 0.857039	valid_1's auc: 0.67816
[70]	training's auc: 0.857249	valid_1's auc: 0.678228
[80]	training's auc: 0.857419	valid_1's auc: 0.678271
[90]	training's auc: 0.857554	valid_1's auc: 0.678298
[100]	training's auc: 0.857661	valid_1's auc: 0.678316
[110]	training's auc: 0.857747	valid_1's auc: 0.678324
[120]	training's auc: 0.857816	valid_1's auc: 0.678329
[130]	training's auc: 0.85787	valid_1's auc: 0.678335
[140]	training's auc: 0.857912	valid_1's auc: 0.678333
[150]	training's auc: 0.857945	valid_1's auc: 0.678335
[160]	training's auc: 0.857972	valid_1's auc: 0.678329
[170]	training's auc: 0.857996	valid_1's auc: 0.678312
[180]	training's auc: 0.858018	valid_1's auc: 0.678306
[190]	training's auc: 0.85804	valid_1's auc: 0.67829
[200]	training's auc: 0.858058	valid_1's auc: 0.678272
[210]	training's auc: 0.85807	valid_1's auc: 0.678262
[220]	training's auc: 0.85808	valid_1's auc: 0.678251
[230]	training's auc: 0.858088	valid_1's auc: 0.678246
[240]	training's auc: 0.858094	valid_1's auc: 0.678241
[250]	training's auc: 0.858098	valid_1's auc: 0.678237
[260]	training's auc: 0.858103	valid_1's auc: 0.678235
[270]	training's auc: 0.858108	valid_1's auc: 0.678234
[280]	training's auc: 0.858111	valid_1's auc: 0.678233
[290]	training's auc: 0.858114	valid_1's auc: 0.678225
[300]	training's auc: 0.858117	valid_1's auc: 0.67822
[310]	training's auc: 0.858119	valid_1's auc: 0.678217
[320]	training's auc: 0.858121	valid_1's auc: 0.678215
[330]	training's auc: 0.858123	valid_1's auc: 0.678213
[340]	training's auc: 0.858124	valid_1's auc: 0.678211
Early stopping, best iteration is:
[149]	training's auc: 0.857945	valid_1's auc: 0.678335
best score: 0.678335017117
best iteration: 149
complete on: ISC_composer

working on: ISCZ_lyricist

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISCZ_lyricist        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854712	valid_1's auc: 0.677003
[20]	training's auc: 0.855382	valid_1's auc: 0.677304
[30]	training's auc: 0.855958	valid_1's auc: 0.677579
[40]	training's auc: 0.85643	valid_1's auc: 0.677795
[50]	training's auc: 0.856797	valid_1's auc: 0.677952
[60]	training's auc: 0.85708	valid_1's auc: 0.678063
[70]	training's auc: 0.857293	valid_1's auc: 0.678148
[80]	training's auc: 0.857466	valid_1's auc: 0.67821
[90]	training's auc: 0.857601	valid_1's auc: 0.678252
[100]	training's auc: 0.857708	valid_1's auc: 0.678269
[110]	training's auc: 0.857789	valid_1's auc: 0.67829
[120]	training's auc: 0.857855	valid_1's auc: 0.678297
[130]	training's auc: 0.857906	valid_1's auc: 0.678302
[140]	training's auc: 0.857947	valid_1's auc: 0.678307
[150]	training's auc: 0.857976	valid_1's auc: 0.678313
[160]	training's auc: 0.858	valid_1's auc: 0.678308
[170]	training's auc: 0.858026	valid_1's auc: 0.678277
[180]	training's auc: 0.858049	valid_1's auc: 0.678277
[190]	training's auc: 0.858064	valid_1's auc: 0.678256
[200]	training's auc: 0.858074	valid_1's auc: 0.678253
[210]	training's auc: 0.858081	valid_1's auc: 0.67825
[220]	training's auc: 0.858087	valid_1's auc: 0.678246
[230]	training's auc: 0.858092	valid_1's auc: 0.678242
[240]	training's auc: 0.858097	valid_1's auc: 0.678235
[250]	training's auc: 0.858101	valid_1's auc: 0.678234
[260]	training's auc: 0.858103	valid_1's auc: 0.678232
[270]	training's auc: 0.858105	valid_1's auc: 0.678229
[280]	training's auc: 0.858107	valid_1's auc: 0.678227
[290]	training's auc: 0.858108	valid_1's auc: 0.678226
[300]	training's auc: 0.85811	valid_1's auc: 0.678221
[310]	training's auc: 0.858111	valid_1's auc: 0.678219
[320]	training's auc: 0.858111	valid_1's auc: 0.678218
[330]	training's auc: 0.858112	valid_1's auc: 0.678218
[340]	training's auc: 0.858112	valid_1's auc: 0.678215
[350]	training's auc: 0.858113	valid_1's auc: 0.678213
Early stopping, best iteration is:
[152]	training's auc: 0.857981	valid_1's auc: 0.678314
best score: 0.678314469513
best iteration: 152
complete on: ISCZ_lyricist

working on: ISC_language

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_language         int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855483	valid_1's auc: 0.677555
[20]	training's auc: 0.855992	valid_1's auc: 0.677782
[30]	training's auc: 0.856444	valid_1's auc: 0.67799
[40]	training's auc: 0.856788	valid_1's auc: 0.678133
[50]	training's auc: 0.857071	valid_1's auc: 0.678238
[60]	training's auc: 0.857297	valid_1's auc: 0.678314
[70]	training's auc: 0.857467	valid_1's auc: 0.678363
[80]	training's auc: 0.857605	valid_1's auc: 0.678386
[90]	training's auc: 0.857712	valid_1's auc: 0.678397
[100]	training's auc: 0.857789	valid_1's auc: 0.678395
[110]	training's auc: 0.857855	valid_1's auc: 0.678383
[120]	training's auc: 0.857907	valid_1's auc: 0.678373
[130]	training's auc: 0.857945	valid_1's auc: 0.67836
[140]	training's auc: 0.857973	valid_1's auc: 0.678344
[150]	training's auc: 0.857994	valid_1's auc: 0.678333
[160]	training's auc: 0.858011	valid_1's auc: 0.678313
[170]	training's auc: 0.858056	valid_1's auc: 0.678281
[180]	training's auc: 0.858064	valid_1's auc: 0.678267
[190]	training's auc: 0.858071	valid_1's auc: 0.678255
[200]	training's auc: 0.858078	valid_1's auc: 0.678244
[210]	training's auc: 0.858083	valid_1's auc: 0.678233
[220]	training's auc: 0.858087	valid_1's auc: 0.67822
[230]	training's auc: 0.85809	valid_1's auc: 0.678209
[240]	training's auc: 0.858093	valid_1's auc: 0.678201
[250]	training's auc: 0.858095	valid_1's auc: 0.678193
[260]	training's auc: 0.858097	valid_1's auc: 0.678186
[270]	training's auc: 0.858098	valid_1's auc: 0.678179
[280]	training's auc: 0.858099	valid_1's auc: 0.678171
[290]	training's auc: 0.8581	valid_1's auc: 0.678168
Early stopping, best iteration is:
[91]	training's auc: 0.857725	valid_1's auc: 0.678399
best score: 0.678398508255
best iteration: 91
complete on: ISC_language

working on: ISCZ_rc

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISCZ_rc              int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.85296	valid_1's auc: 0.676288
[20]	training's auc: 0.853967	valid_1's auc: 0.676806
[30]	training's auc: 0.854842	valid_1's auc: 0.677198
[40]	training's auc: 0.855592	valid_1's auc: 0.677547
[50]	training's auc: 0.856141	valid_1's auc: 0.677803
[60]	training's auc: 0.856533	valid_1's auc: 0.677968
[70]	training's auc: 0.856854	valid_1's auc: 0.678099
[80]	training's auc: 0.857113	valid_1's auc: 0.678195
[90]	training's auc: 0.857323	valid_1's auc: 0.678264
[100]	training's auc: 0.85749	valid_1's auc: 0.678306
[110]	training's auc: 0.857623	valid_1's auc: 0.678332
[120]	training's auc: 0.857729	valid_1's auc: 0.678346
[130]	training's auc: 0.857814	valid_1's auc: 0.678351
[140]	training's auc: 0.857882	valid_1's auc: 0.67835
[150]	training's auc: 0.857937	valid_1's auc: 0.678352
[160]	training's auc: 0.857983	valid_1's auc: 0.67835
[170]	training's auc: 0.858027	valid_1's auc: 0.678325
[180]	training's auc: 0.858062	valid_1's auc: 0.678302
[190]	training's auc: 0.858087	valid_1's auc: 0.678292
[200]	training's auc: 0.858107	valid_1's auc: 0.678275
[210]	training's auc: 0.858122	valid_1's auc: 0.678262
[220]	training's auc: 0.858135	valid_1's auc: 0.678249
[230]	training's auc: 0.858146	valid_1's auc: 0.678236
[240]	training's auc: 0.858155	valid_1's auc: 0.678225
[250]	training's auc: 0.858164	valid_1's auc: 0.67822
[260]	training's auc: 0.858171	valid_1's auc: 0.678212
[270]	training's auc: 0.858175	valid_1's auc: 0.678207
[280]	training's auc: 0.858181	valid_1's auc: 0.678197
[290]	training's auc: 0.858186	valid_1's auc: 0.678192
[300]	training's auc: 0.858189	valid_1's auc: 0.678184
[310]	training's auc: 0.858192	valid_1's auc: 0.678179
[320]	training's auc: 0.858195	valid_1's auc: 0.678173
[330]	training's auc: 0.858197	valid_1's auc: 0.678171
[340]	training's auc: 0.858199	valid_1's auc: 0.678166
[350]	training's auc: 0.858201	valid_1's auc: 0.678161
Early stopping, best iteration is:
[151]	training's auc: 0.857946	valid_1's auc: 0.678357
best score: 0.678356605986
best iteration: 151
complete on: ISCZ_rc

working on: ISCZ_isrc_rest

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISCZ_isrc_rest       int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856649	valid_1's auc: 0.677518
[20]	training's auc: 0.856954	valid_1's auc: 0.67773
[30]	training's auc: 0.857195	valid_1's auc: 0.677894
[40]	training's auc: 0.85737	valid_1's auc: 0.678001
[50]	training's auc: 0.857501	valid_1's auc: 0.678065
[60]	training's auc: 0.857603	valid_1's auc: 0.678114
[70]	training's auc: 0.857683	valid_1's auc: 0.678141
[80]	training's auc: 0.857751	valid_1's auc: 0.678172
[90]	training's auc: 0.857806	valid_1's auc: 0.67819
[100]	training's auc: 0.857851	valid_1's auc: 0.678206
[110]	training's auc: 0.857887	valid_1's auc: 0.678218
[120]	training's auc: 0.857917	valid_1's auc: 0.678228
[130]	training's auc: 0.857941	valid_1's auc: 0.678235
[140]	training's auc: 0.857959	valid_1's auc: 0.678238
[150]	training's auc: 0.857976	valid_1's auc: 0.678239
[160]	training's auc: 0.857989	valid_1's auc: 0.678232
[170]	training's auc: 0.858	valid_1's auc: 0.678231
[180]	training's auc: 0.858011	valid_1's auc: 0.67823
[190]	training's auc: 0.858022	valid_1's auc: 0.678226
[200]	training's auc: 0.858031	valid_1's auc: 0.678229
[210]	training's auc: 0.858037	valid_1's auc: 0.678229
[220]	training's auc: 0.858043	valid_1's auc: 0.678229
[230]	training's auc: 0.85805	valid_1's auc: 0.678231
[240]	training's auc: 0.858054	valid_1's auc: 0.678231
[250]	training's auc: 0.858059	valid_1's auc: 0.678231
[260]	training's auc: 0.858063	valid_1's auc: 0.678232
[270]	training's auc: 0.858066	valid_1's auc: 0.678234
[280]	training's auc: 0.858069	valid_1's auc: 0.678234
[290]	training's auc: 0.858072	valid_1's auc: 0.678237
[300]	training's auc: 0.858074	valid_1's auc: 0.678238
[310]	training's auc: 0.858075	valid_1's auc: 0.678238
[320]	training's auc: 0.858077	valid_1's auc: 0.678237
[330]	training's auc: 0.858079	valid_1's auc: 0.678236
[340]	training's auc: 0.85808	valid_1's auc: 0.678237
Early stopping, best iteration is:
[146]	training's auc: 0.857969	valid_1's auc: 0.678241
best score: 0.678240753625
best iteration: 146
complete on: ISCZ_isrc_rest

working on: ISC_song_year

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_song_year        int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855464	valid_1's auc: 0.679125
[20]	training's auc: 0.855983	valid_1's auc: 0.67922
[30]	training's auc: 0.856421	valid_1's auc: 0.679265
[40]	training's auc: 0.856782	valid_1's auc: 0.67928
[50]	training's auc: 0.857062	valid_1's auc: 0.679249
[60]	training's auc: 0.857285	valid_1's auc: 0.67919
[70]	training's auc: 0.857457	valid_1's auc: 0.679117
[80]	training's auc: 0.857594	valid_1's auc: 0.679042
[90]	training's auc: 0.857703	valid_1's auc: 0.678965
[100]	training's auc: 0.857787	valid_1's auc: 0.67888
[110]	training's auc: 0.857852	valid_1's auc: 0.678807
[120]	training's auc: 0.857904	valid_1's auc: 0.678729
[130]	training's auc: 0.857943	valid_1's auc: 0.678651
[140]	training's auc: 0.857971	valid_1's auc: 0.67858
[150]	training's auc: 0.858002	valid_1's auc: 0.67854
[160]	training's auc: 0.858044	valid_1's auc: 0.678473
[170]	training's auc: 0.858065	valid_1's auc: 0.678425
[180]	training's auc: 0.858079	valid_1's auc: 0.678377
[190]	training's auc: 0.858088	valid_1's auc: 0.678312
[200]	training's auc: 0.858095	valid_1's auc: 0.678256
[210]	training's auc: 0.858101	valid_1's auc: 0.678237
[220]	training's auc: 0.858106	valid_1's auc: 0.678185
[230]	training's auc: 0.85811	valid_1's auc: 0.678167
[240]	training's auc: 0.858114	valid_1's auc: 0.678149
Early stopping, best iteration is:
[41]	training's auc: 0.856767	valid_1's auc: 0.67929
best score: 0.679289509667
best iteration: 41
complete on: ISC_song_year

working on: ISCZ_song_year

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISCZ_song_year       int64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855474	valid_1's auc: 0.679185
[20]	training's auc: 0.85599	valid_1's auc: 0.679276
[30]	training's auc: 0.856422	valid_1's auc: 0.679315
[40]	training's auc: 0.856779	valid_1's auc: 0.679315
[50]	training's auc: 0.857062	valid_1's auc: 0.679288
[60]	training's auc: 0.857285	valid_1's auc: 0.679228
[70]	training's auc: 0.857458	valid_1's auc: 0.679149
[80]	training's auc: 0.857594	valid_1's auc: 0.679062
[90]	training's auc: 0.857703	valid_1's auc: 0.678973
[100]	training's auc: 0.857786	valid_1's auc: 0.678888
[110]	training's auc: 0.857854	valid_1's auc: 0.678804
[120]	training's auc: 0.857906	valid_1's auc: 0.678729
[130]	training's auc: 0.857944	valid_1's auc: 0.678667
[140]	training's auc: 0.857975	valid_1's auc: 0.678602
[150]	training's auc: 0.858005	valid_1's auc: 0.678541
[160]	training's auc: 0.858045	valid_1's auc: 0.678467
[170]	training's auc: 0.858065	valid_1's auc: 0.678414
[180]	training's auc: 0.858077	valid_1's auc: 0.678378
[190]	training's auc: 0.858086	valid_1's auc: 0.678312
[200]	training's auc: 0.858094	valid_1's auc: 0.67826
[210]	training's auc: 0.8581	valid_1's auc: 0.678237
[220]	training's auc: 0.858106	valid_1's auc: 0.678192
[230]	training's auc: 0.85811	valid_1's auc: 0.678177
Early stopping, best iteration is:
[39]	training's auc: 0.856694	valid_1's auc: 0.679322
best score: 0.679322087335
best iteration: 39
complete on: ISCZ_song_year

working on: song_length_log10

Our guest selection:
target                 uint8
FAKE_1512883008      float64
song_length_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855752	valid_1's auc: 0.677562
[20]	training's auc: 0.856244	valid_1's auc: 0.677752
[30]	training's auc: 0.856635	valid_1's auc: 0.677938
[40]	training's auc: 0.856942	valid_1's auc: 0.678084
[50]	training's auc: 0.85718	valid_1's auc: 0.678191
[60]	training's auc: 0.857353	valid_1's auc: 0.678253
[70]	training's auc: 0.857491	valid_1's auc: 0.678298
[80]	training's auc: 0.8576	valid_1's auc: 0.678325
[90]	training's auc: 0.857687	valid_1's auc: 0.678342
[100]	training's auc: 0.857757	valid_1's auc: 0.678353
[110]	training's auc: 0.857813	valid_1's auc: 0.678352
[120]	training's auc: 0.857859	valid_1's auc: 0.67835
[130]	training's auc: 0.857895	valid_1's auc: 0.678343
[140]	training's auc: 0.857926	valid_1's auc: 0.678335
[150]	training's auc: 0.85795	valid_1's auc: 0.678329
[160]	training's auc: 0.857971	valid_1's auc: 0.678322
[170]	training's auc: 0.857986	valid_1's auc: 0.678312
[180]	training's auc: 0.857999	valid_1's auc: 0.678305
[190]	training's auc: 0.858012	valid_1's auc: 0.678293
[200]	training's auc: 0.858022	valid_1's auc: 0.678287
[210]	training's auc: 0.858031	valid_1's auc: 0.678274
[220]	training's auc: 0.858041	valid_1's auc: 0.678265
[230]	training's auc: 0.858049	valid_1's auc: 0.678253
[240]	training's auc: 0.858054	valid_1's auc: 0.678246
[250]	training's auc: 0.85806	valid_1's auc: 0.678242
[260]	training's auc: 0.858064	valid_1's auc: 0.678236
[270]	training's auc: 0.858069	valid_1's auc: 0.67823
[280]	training's auc: 0.858072	valid_1's auc: 0.678229
[290]	training's auc: 0.858075	valid_1's auc: 0.678223
[300]	training's auc: 0.858078	valid_1's auc: 0.678224
Early stopping, best iteration is:
[100]	training's auc: 0.857757	valid_1's auc: 0.678353
best score: 0.678352732936
best iteration: 100
complete on: song_length_log10

working on: ISCZ_genre_ids_log10

Our guest selection:
target                    uint8
FAKE_1512883008         float64
ISCZ_genre_ids_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854659	valid_1's auc: 0.676961
[20]	training's auc: 0.855312	valid_1's auc: 0.677294
[30]	training's auc: 0.855866	valid_1's auc: 0.677581
[40]	training's auc: 0.85633	valid_1's auc: 0.677811
[50]	training's auc: 0.856711	valid_1's auc: 0.677998
[60]	training's auc: 0.857009	valid_1's auc: 0.678129
[70]	training's auc: 0.857243	valid_1's auc: 0.678205
[80]	training's auc: 0.85743	valid_1's auc: 0.678272
[90]	training's auc: 0.857577	valid_1's auc: 0.678317
[100]	training's auc: 0.857694	valid_1's auc: 0.678336
[110]	training's auc: 0.857788	valid_1's auc: 0.678349
[120]	training's auc: 0.857862	valid_1's auc: 0.678346
[130]	training's auc: 0.857919	valid_1's auc: 0.678345
[140]	training's auc: 0.857962	valid_1's auc: 0.678333
[150]	training's auc: 0.857996	valid_1's auc: 0.678322
[160]	training's auc: 0.858038	valid_1's auc: 0.678251
[170]	training's auc: 0.858069	valid_1's auc: 0.678239
[180]	training's auc: 0.858086	valid_1's auc: 0.67823
[190]	training's auc: 0.8581	valid_1's auc: 0.678218
[200]	training's auc: 0.858111	valid_1's auc: 0.678202
[210]	training's auc: 0.85812	valid_1's auc: 0.678192
[220]	training's auc: 0.858127	valid_1's auc: 0.678182
[230]	training's auc: 0.858132	valid_1's auc: 0.678194
[240]	training's auc: 0.858136	valid_1's auc: 0.678187
[250]	training's auc: 0.85814	valid_1's auc: 0.678175
[260]	training's auc: 0.858143	valid_1's auc: 0.678168
[270]	training's auc: 0.858146	valid_1's auc: 0.678163
[280]	training's auc: 0.858147	valid_1's auc: 0.678156
[290]	training's auc: 0.858149	valid_1's auc: 0.678154
[300]	training's auc: 0.858151	valid_1's auc: 0.678148
[310]	training's auc: 0.858152	valid_1's auc: 0.67814
Early stopping, best iteration is:
[112]	training's auc: 0.857803	valid_1's auc: 0.678353
best score: 0.678352632886
best iteration: 112
complete on: ISCZ_genre_ids_log10

working on: ISC_artist_name_log10

Our guest selection:
target                     uint8
FAKE_1512883008          float64
ISC_artist_name_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.853701	valid_1's auc: 0.676165
[20]	training's auc: 0.854527	valid_1's auc: 0.676619
[30]	training's auc: 0.855253	valid_1's auc: 0.677035
[40]	training's auc: 0.855868	valid_1's auc: 0.677391
[50]	training's auc: 0.856337	valid_1's auc: 0.677672
[60]	training's auc: 0.856679	valid_1's auc: 0.677866
[70]	training's auc: 0.856961	valid_1's auc: 0.67801
[80]	training's auc: 0.857193	valid_1's auc: 0.678119
[90]	training's auc: 0.857375	valid_1's auc: 0.678198
[100]	training's auc: 0.857522	valid_1's auc: 0.678252
[110]	training's auc: 0.85764	valid_1's auc: 0.678292
[120]	training's auc: 0.857733	valid_1's auc: 0.678315
[130]	training's auc: 0.857808	valid_1's auc: 0.678327
[140]	training's auc: 0.85787	valid_1's auc: 0.67833
[150]	training's auc: 0.857918	valid_1's auc: 0.678336
[160]	training's auc: 0.857956	valid_1's auc: 0.678332
[170]	training's auc: 0.857991	valid_1's auc: 0.678323
[180]	training's auc: 0.85802	valid_1's auc: 0.67831
[190]	training's auc: 0.858044	valid_1's auc: 0.678309
[200]	training's auc: 0.858064	valid_1's auc: 0.678303
[210]	training's auc: 0.858079	valid_1's auc: 0.678295
[220]	training's auc: 0.858092	valid_1's auc: 0.678291
[230]	training's auc: 0.858104	valid_1's auc: 0.678277
[240]	training's auc: 0.858113	valid_1's auc: 0.678271
[250]	training's auc: 0.858121	valid_1's auc: 0.678262
[260]	training's auc: 0.85813	valid_1's auc: 0.678251
[270]	training's auc: 0.858136	valid_1's auc: 0.678241
[280]	training's auc: 0.858142	valid_1's auc: 0.678235
[290]	training's auc: 0.858147	valid_1's auc: 0.678227
[300]	training's auc: 0.858151	valid_1's auc: 0.678213
[310]	training's auc: 0.858155	valid_1's auc: 0.678207
[320]	training's auc: 0.858159	valid_1's auc: 0.678201
[330]	training's auc: 0.858161	valid_1's auc: 0.678196
[340]	training's auc: 0.858163	valid_1's auc: 0.678194
Early stopping, best iteration is:
[147]	training's auc: 0.857908	valid_1's auc: 0.678337
best score: 0.678337026509
best iteration: 147
complete on: ISC_artist_name_log10

working on: ISCZ_composer_log10

Our guest selection:
target                   uint8
FAKE_1512883008        float64
ISCZ_composer_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.855049	valid_1's auc: 0.677243
[20]	training's auc: 0.855613	valid_1's auc: 0.677583
[30]	training's auc: 0.85609	valid_1's auc: 0.677824
[40]	training's auc: 0.856495	valid_1's auc: 0.677988
[50]	training's auc: 0.856812	valid_1's auc: 0.678111
[60]	training's auc: 0.85707	valid_1's auc: 0.6782
[70]	training's auc: 0.857273	valid_1's auc: 0.678265
[80]	training's auc: 0.857436	valid_1's auc: 0.678298
[90]	training's auc: 0.857566	valid_1's auc: 0.678316
[100]	training's auc: 0.857668	valid_1's auc: 0.67833
[110]	training's auc: 0.857749	valid_1's auc: 0.678341
[120]	training's auc: 0.857814	valid_1's auc: 0.678339
[130]	training's auc: 0.857864	valid_1's auc: 0.678341
[140]	training's auc: 0.857905	valid_1's auc: 0.678333
[150]	training's auc: 0.857939	valid_1's auc: 0.678334
[160]	training's auc: 0.857964	valid_1's auc: 0.678333
[170]	training's auc: 0.85799	valid_1's auc: 0.678317
[180]	training's auc: 0.85801	valid_1's auc: 0.678309
[190]	training's auc: 0.858031	valid_1's auc: 0.678298
[200]	training's auc: 0.858048	valid_1's auc: 0.678286
[210]	training's auc: 0.858061	valid_1's auc: 0.678269
[220]	training's auc: 0.858073	valid_1's auc: 0.678264
[230]	training's auc: 0.858081	valid_1's auc: 0.678253
[240]	training's auc: 0.858088	valid_1's auc: 0.678245
[250]	training's auc: 0.858096	valid_1's auc: 0.678241
[260]	training's auc: 0.858101	valid_1's auc: 0.678241
[270]	training's auc: 0.858106	valid_1's auc: 0.678233
[280]	training's auc: 0.858109	valid_1's auc: 0.678227
[290]	training's auc: 0.858113	valid_1's auc: 0.678231
[300]	training's auc: 0.858116	valid_1's auc: 0.678231
[310]	training's auc: 0.858119	valid_1's auc: 0.678226
[320]	training's auc: 0.858122	valid_1's auc: 0.678224
Early stopping, best iteration is:
[123]	training's auc: 0.857834	valid_1's auc: 0.678341
best score: 0.678340835132
best iteration: 123
complete on: ISCZ_composer_log10

working on: ISC_lyricist_log10

Our guest selection:
target                  uint8
FAKE_1512883008       float64
ISC_lyricist_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.854691	valid_1's auc: 0.676993
[20]	training's auc: 0.855378	valid_1's auc: 0.67733
[30]	training's auc: 0.855957	valid_1's auc: 0.677595
[40]	training's auc: 0.856428	valid_1's auc: 0.677808
[50]	training's auc: 0.856795	valid_1's auc: 0.677961
[60]	training's auc: 0.857076	valid_1's auc: 0.678069
[70]	training's auc: 0.857292	valid_1's auc: 0.678149
[80]	training's auc: 0.857468	valid_1's auc: 0.678211
[90]	training's auc: 0.857603	valid_1's auc: 0.678252
[100]	training's auc: 0.857707	valid_1's auc: 0.678271
[110]	training's auc: 0.857791	valid_1's auc: 0.678288
[120]	training's auc: 0.857855	valid_1's auc: 0.678302
[130]	training's auc: 0.857908	valid_1's auc: 0.678309
[140]	training's auc: 0.857947	valid_1's auc: 0.678312
[150]	training's auc: 0.857977	valid_1's auc: 0.678314
[160]	training's auc: 0.857999	valid_1's auc: 0.678312
[170]	training's auc: 0.858025	valid_1's auc: 0.678276
[180]	training's auc: 0.858048	valid_1's auc: 0.678276
[190]	training's auc: 0.858064	valid_1's auc: 0.678262
[200]	training's auc: 0.858075	valid_1's auc: 0.678256
[210]	training's auc: 0.858083	valid_1's auc: 0.678252
[220]	training's auc: 0.858089	valid_1's auc: 0.678249
[230]	training's auc: 0.858094	valid_1's auc: 0.678244
[240]	training's auc: 0.858099	valid_1's auc: 0.67824
[250]	training's auc: 0.858102	valid_1's auc: 0.678236
[260]	training's auc: 0.858104	valid_1's auc: 0.678234
[270]	training's auc: 0.858107	valid_1's auc: 0.678229
[280]	training's auc: 0.858109	valid_1's auc: 0.678227
[290]	training's auc: 0.85811	valid_1's auc: 0.678226
[300]	training's auc: 0.858111	valid_1's auc: 0.678223
[310]	training's auc: 0.858112	valid_1's auc: 0.67822
[320]	training's auc: 0.858112	valid_1's auc: 0.678218
[330]	training's auc: 0.858113	valid_1's auc: 0.678215
[340]	training's auc: 0.858114	valid_1's auc: 0.678213
Early stopping, best iteration is:
[147]	training's auc: 0.857971	valid_1's auc: 0.678316
best score: 0.67831551792
best iteration: 147
complete on: ISC_lyricist_log10

working on: ISC_name_log10

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_name_log10     float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856962	valid_1's auc: 0.677807
[20]	training's auc: 0.85719	valid_1's auc: 0.677931
[30]	training's auc: 0.857377	valid_1's auc: 0.678016
[40]	training's auc: 0.857527	valid_1's auc: 0.678084
[50]	training's auc: 0.857634	valid_1's auc: 0.678145
[60]	training's auc: 0.857716	valid_1's auc: 0.67819
[70]	training's auc: 0.857784	valid_1's auc: 0.678221
[80]	training's auc: 0.857842	valid_1's auc: 0.678246
[90]	training's auc: 0.857885	valid_1's auc: 0.678266
[100]	training's auc: 0.857919	valid_1's auc: 0.678298
[110]	training's auc: 0.857951	valid_1's auc: 0.678314
[120]	training's auc: 0.857974	valid_1's auc: 0.67833
[130]	training's auc: 0.857993	valid_1's auc: 0.678337
[140]	training's auc: 0.858009	valid_1's auc: 0.678344
[150]	training's auc: 0.858022	valid_1's auc: 0.678353
[160]	training's auc: 0.858033	valid_1's auc: 0.678358
[170]	training's auc: 0.858042	valid_1's auc: 0.678362
[180]	training's auc: 0.858049	valid_1's auc: 0.678369
[190]	training's auc: 0.858055	valid_1's auc: 0.678369
[200]	training's auc: 0.85806	valid_1's auc: 0.678369
[210]	training's auc: 0.858065	valid_1's auc: 0.678367
[220]	training's auc: 0.858068	valid_1's auc: 0.678366
[230]	training's auc: 0.858071	valid_1's auc: 0.678365
[240]	training's auc: 0.858073	valid_1's auc: 0.67836
[250]	training's auc: 0.858075	valid_1's auc: 0.678352
[260]	training's auc: 0.858077	valid_1's auc: 0.67835
[270]	training's auc: 0.858078	valid_1's auc: 0.678342
[280]	training's auc: 0.85808	valid_1's auc: 0.67834
[290]	training's auc: 0.858081	valid_1's auc: 0.678339
[300]	training's auc: 0.858081	valid_1's auc: 0.67833
[310]	training's auc: 0.858082	valid_1's auc: 0.678325
[320]	training's auc: 0.858083	valid_1's auc: 0.678324
[330]	training's auc: 0.858083	valid_1's auc: 0.678324
[340]	training's auc: 0.858083	valid_1's auc: 0.678322
[350]	training's auc: 0.858084	valid_1's auc: 0.678322
[360]	training's auc: 0.858084	valid_1's auc: 0.678318
[370]	training's auc: 0.858084	valid_1's auc: 0.678317
[380]	training's auc: 0.858085	valid_1's auc: 0.678315
[390]	training's auc: 0.858085	valid_1's auc: 0.678315
Early stopping, best iteration is:
[193]	training's auc: 0.858057	valid_1's auc: 0.678371
best score: 0.678370827447
best iteration: 193
complete on: ISC_name_log10

working on: ISCZ_name_ln

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISCZ_name_ln       float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856926	valid_1's auc: 0.677821
[20]	training's auc: 0.85717	valid_1's auc: 0.677933
[30]	training's auc: 0.857378	valid_1's auc: 0.678028
[40]	training's auc: 0.857522	valid_1's auc: 0.678092
[50]	training's auc: 0.857633	valid_1's auc: 0.67817
[60]	training's auc: 0.857716	valid_1's auc: 0.678221
[70]	training's auc: 0.857783	valid_1's auc: 0.678254
[80]	training's auc: 0.85784	valid_1's auc: 0.678282
[90]	training's auc: 0.857886	valid_1's auc: 0.678298
[100]	training's auc: 0.857922	valid_1's auc: 0.678331
[110]	training's auc: 0.857949	valid_1's auc: 0.678341
[120]	training's auc: 0.857973	valid_1's auc: 0.678355
[130]	training's auc: 0.857992	valid_1's auc: 0.678359
[140]	training's auc: 0.858009	valid_1's auc: 0.678362
[150]	training's auc: 0.858023	valid_1's auc: 0.678364
[160]	training's auc: 0.858034	valid_1's auc: 0.678368
[170]	training's auc: 0.858043	valid_1's auc: 0.678374
[180]	training's auc: 0.858049	valid_1's auc: 0.678379
[190]	training's auc: 0.858056	valid_1's auc: 0.678376
[200]	training's auc: 0.858061	valid_1's auc: 0.678373
[210]	training's auc: 0.858064	valid_1's auc: 0.678369
[220]	training's auc: 0.858068	valid_1's auc: 0.678365
[230]	training's auc: 0.858071	valid_1's auc: 0.678364
[240]	training's auc: 0.858073	valid_1's auc: 0.678358
[250]	training's auc: 0.858075	valid_1's auc: 0.678352
[260]	training's auc: 0.858077	valid_1's auc: 0.67835
[270]	training's auc: 0.858079	valid_1's auc: 0.678345
[280]	training's auc: 0.85808	valid_1's auc: 0.678339
[290]	training's auc: 0.858081	valid_1's auc: 0.678327
[300]	training's auc: 0.858082	valid_1's auc: 0.678324
[310]	training's auc: 0.858082	valid_1's auc: 0.67832
[320]	training's auc: 0.858083	valid_1's auc: 0.678319
[330]	training's auc: 0.858084	valid_1's auc: 0.678317
[340]	training's auc: 0.858084	valid_1's auc: 0.678315
[350]	training's auc: 0.858085	valid_1's auc: 0.678314
[360]	training's auc: 0.858085	valid_1's auc: 0.678311
[370]	training's auc: 0.858085	valid_1's auc: 0.678311
[380]	training's auc: 0.858086	valid_1's auc: 0.67831
Early stopping, best iteration is:
[180]	training's auc: 0.858049	valid_1's auc: 0.678379
best score: 0.678379418961
best iteration: 180
complete on: ISCZ_name_ln

working on: ISC_song_country_ln

Our guest selection:
target                   uint8
FAKE_1512883008        float64
ISC_song_country_ln    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.85532	valid_1's auc: 0.677262
[20]	training's auc: 0.85585	valid_1's auc: 0.677535
[30]	training's auc: 0.856321	valid_1's auc: 0.677763
[40]	training's auc: 0.856693	valid_1's auc: 0.677939
[50]	training's auc: 0.857	valid_1's auc: 0.678082
[60]	training's auc: 0.857242	valid_1's auc: 0.67818
[70]	training's auc: 0.857428	valid_1's auc: 0.678249
[80]	training's auc: 0.857578	valid_1's auc: 0.678295
[90]	training's auc: 0.857689	valid_1's auc: 0.678323
[100]	training's auc: 0.857777	valid_1's auc: 0.678344
[110]	training's auc: 0.857845	valid_1's auc: 0.678346
[120]	training's auc: 0.8579	valid_1's auc: 0.678349
[130]	training's auc: 0.857941	valid_1's auc: 0.678346
[140]	training's auc: 0.857971	valid_1's auc: 0.678338
[150]	training's auc: 0.857996	valid_1's auc: 0.678332
[160]	training's auc: 0.858035	valid_1's auc: 0.678317
[170]	training's auc: 0.858059	valid_1's auc: 0.678298
[180]	training's auc: 0.858071	valid_1's auc: 0.678291
[190]	training's auc: 0.85808	valid_1's auc: 0.678278
[200]	training's auc: 0.858087	valid_1's auc: 0.678266
[210]	training's auc: 0.858093	valid_1's auc: 0.678259
[220]	training's auc: 0.858098	valid_1's auc: 0.678248
[230]	training's auc: 0.858102	valid_1's auc: 0.67824
[240]	training's auc: 0.858105	valid_1's auc: 0.678235
[250]	training's auc: 0.858107	valid_1's auc: 0.678229
[260]	training's auc: 0.858109	valid_1's auc: 0.678223
[270]	training's auc: 0.858111	valid_1's auc: 0.678217
[280]	training's auc: 0.858112	valid_1's auc: 0.678211
[290]	training's auc: 0.858113	valid_1's auc: 0.678207
[300]	training's auc: 0.858114	valid_1's auc: 0.678205
[310]	training's auc: 0.858114	valid_1's auc: 0.678201
[320]	training's auc: 0.858115	valid_1's auc: 0.678198
Early stopping, best iteration is:
[120]	training's auc: 0.8579	valid_1's auc: 0.678349
best score: 0.678349104814
best iteration: 120
complete on: ISC_song_country_ln

working on: ISCZ_song_country_log10

Our guest selection:
target                       uint8
FAKE_1512883008            float64
ISCZ_song_country_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.85534	valid_1's auc: 0.677347
[20]	training's auc: 0.855868	valid_1's auc: 0.677586
[30]	training's auc: 0.85633	valid_1's auc: 0.677785
[40]	training's auc: 0.856703	valid_1's auc: 0.677964
[50]	training's auc: 0.857011	valid_1's auc: 0.678095
[60]	training's auc: 0.857251	valid_1's auc: 0.67819
[70]	training's auc: 0.857434	valid_1's auc: 0.678251
[80]	training's auc: 0.857577	valid_1's auc: 0.678296
[90]	training's auc: 0.85769	valid_1's auc: 0.678322
[100]	training's auc: 0.857777	valid_1's auc: 0.678342
[110]	training's auc: 0.857846	valid_1's auc: 0.678346
[120]	training's auc: 0.857902	valid_1's auc: 0.678346
[130]	training's auc: 0.857942	valid_1's auc: 0.678345
[140]	training's auc: 0.857974	valid_1's auc: 0.67834
[150]	training's auc: 0.857999	valid_1's auc: 0.678332
[160]	training's auc: 0.858036	valid_1's auc: 0.678315
[170]	training's auc: 0.85806	valid_1's auc: 0.678299
[180]	training's auc: 0.858071	valid_1's auc: 0.678287
[190]	training's auc: 0.858081	valid_1's auc: 0.678275
[200]	training's auc: 0.858088	valid_1's auc: 0.678264
[210]	training's auc: 0.858094	valid_1's auc: 0.678256
[220]	training's auc: 0.858098	valid_1's auc: 0.678245
[230]	training's auc: 0.858102	valid_1's auc: 0.678237
[240]	training's auc: 0.858106	valid_1's auc: 0.678231
[250]	training's auc: 0.858108	valid_1's auc: 0.678225
[260]	training's auc: 0.85811	valid_1's auc: 0.678221
[270]	training's auc: 0.858112	valid_1's auc: 0.678216
[280]	training's auc: 0.858113	valid_1's auc: 0.678212
[290]	training's auc: 0.858114	valid_1's auc: 0.678208
[300]	training's auc: 0.858115	valid_1's auc: 0.678204
[310]	training's auc: 0.858116	valid_1's auc: 0.6782
Early stopping, best iteration is:
[115]	training's auc: 0.85788	valid_1's auc: 0.678349
best score: 0.678348875478
best iteration: 115
complete on: ISCZ_song_country_log10

working on: ISC_rc_ln

Our guest selection:
target               uint8
FAKE_1512883008    float64
ISC_rc_ln          float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.852856	valid_1's auc: 0.676118
[20]	training's auc: 0.853918	valid_1's auc: 0.676689
[30]	training's auc: 0.854778	valid_1's auc: 0.677083
[40]	training's auc: 0.855513	valid_1's auc: 0.677428
[50]	training's auc: 0.856073	valid_1's auc: 0.677692
[60]	training's auc: 0.85649	valid_1's auc: 0.677895
[70]	training's auc: 0.856826	valid_1's auc: 0.678041
[80]	training's auc: 0.857093	valid_1's auc: 0.678151
[90]	training's auc: 0.85731	valid_1's auc: 0.678222
[100]	training's auc: 0.857481	valid_1's auc: 0.678277
[110]	training's auc: 0.857618	valid_1's auc: 0.67831
[120]	training's auc: 0.857727	valid_1's auc: 0.678329
[130]	training's auc: 0.857814	valid_1's auc: 0.678339
[140]	training's auc: 0.857884	valid_1's auc: 0.678344
[150]	training's auc: 0.857938	valid_1's auc: 0.678344
[160]	training's auc: 0.857984	valid_1's auc: 0.678343
[170]	training's auc: 0.858027	valid_1's auc: 0.67832
[180]	training's auc: 0.858062	valid_1's auc: 0.678311
[190]	training's auc: 0.858087	valid_1's auc: 0.678298
[200]	training's auc: 0.858108	valid_1's auc: 0.678286
[210]	training's auc: 0.858125	valid_1's auc: 0.678271
[220]	training's auc: 0.858138	valid_1's auc: 0.678259
[230]	training's auc: 0.858149	valid_1's auc: 0.678251
[240]	training's auc: 0.858158	valid_1's auc: 0.67824
[250]	training's auc: 0.858166	valid_1's auc: 0.678236
[260]	training's auc: 0.858172	valid_1's auc: 0.678225
[270]	training's auc: 0.858178	valid_1's auc: 0.678218
[280]	training's auc: 0.858182	valid_1's auc: 0.678209
[290]	training's auc: 0.858186	valid_1's auc: 0.6782
[300]	training's auc: 0.85819	valid_1's auc: 0.678192
[310]	training's auc: 0.858193	valid_1's auc: 0.678187
[320]	training's auc: 0.858195	valid_1's auc: 0.678181
[330]	training's auc: 0.858197	valid_1's auc: 0.678175
[340]	training's auc: 0.858199	valid_1's auc: 0.678169
[350]	training's auc: 0.8582	valid_1's auc: 0.678164
Early stopping, best iteration is:
[155]	training's auc: 0.857967	valid_1's auc: 0.678347
best score: 0.678346758606
best iteration: 155
complete on: ISC_rc_ln

working on: ISC_isrc_rest_log10

Our guest selection:
target                   uint8
FAKE_1512883008        float64
ISC_isrc_rest_log10    float64
dtype: object
number of columns: 3

train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 200 rounds.
[10]	training's auc: 0.856722	valid_1's auc: 0.67765
[20]	training's auc: 0.857019	valid_1's auc: 0.677845
[30]	training's auc: 0.857241	valid_1's auc: 0.677976
[40]	training's auc: 0.857408	valid_1's auc: 0.678044
[50]	training's auc: 0.857535	valid_1's auc: 0.6781
[60]	training's auc: 0.857626	valid_1's auc: 0.678139
[70]	training's auc: 0.857702	valid_1's auc: 0.678169
[80]	training's auc: 0.857765	valid_1's auc: 0.678193
[90]	training's auc: 0.857815	valid_1's auc: 0.678209
[100]	training's auc: 0.857855	valid_1's auc: 0.678222
[110]	training's auc: 0.857889	valid_1's auc: 0.678231
[120]	training's auc: 0.857916	valid_1's auc: 0.678235
[130]	training's auc: 0.857939	valid_1's auc: 0.678244
[140]	training's auc: 0.857958	valid_1's auc: 0.678246
[150]	training's auc: 0.857974	valid_1's auc: 0.678246
[160]	training's auc: 0.857986	valid_1's auc: 0.678249
[170]	training's auc: 0.857998	valid_1's auc: 0.678246
[180]	training's auc: 0.858008	valid_1's auc: 0.678246
[190]	training's auc: 0.858016	valid_1's auc: 0.678245
[200]	training's auc: 0.858026	valid_1's auc: 0.678242
[210]	training's auc: 0.858032	valid_1's auc: 0.67824
[220]	training's auc: 0.858039	valid_1's auc: 0.678243
[230]	training's auc: 0.858046	valid_1's auc: 0.678244
[240]	training's auc: 0.85805	valid_1's auc: 0.678247
[250]	training's auc: 0.858054	valid_1's auc: 0.678249
[260]	training's auc: 0.858059	valid_1's auc: 0.67825
[270]	training's auc: 0.858062	valid_1's auc: 0.678251
[280]	training's auc: 0.858065	valid_1's auc: 0.678254
[290]	training's auc: 0.858068	valid_1's auc: 0.678253
[300]	training's auc: 0.85807	valid_1's auc: 0.678254
[310]	training's auc: 0.858073	valid_1's auc: 0.678258
[320]	training's auc: 0.858074	valid_1's auc: 0.678257
[330]	training's auc: 0.858076	valid_1's auc: 0.678259
[340]	training's auc: 0.858077	valid_1's auc: 0.678259
[350]	training's auc: 0.858078	valid_1's auc: 0.678262
[360]	training's auc: 0.85808	valid_1's auc: 0.678261
[370]	training's auc: 0.858081	valid_1's auc: 0.678262
[380]	training's auc: 0.858082	valid_1's auc: 0.678261
[390]	training's auc: 0.858082	valid_1's auc: 0.678259
[400]	training's auc: 0.858083	valid_1's auc: 0.67826
[410]	training's auc: 0.858084	valid_1's auc: 0.678263
[420]	training's auc: 0.858084	valid_1's auc: 0.67826
[430]	training's auc: 0.858085	valid_1's auc: 0.678259
[440]	training's auc: 0.858085	valid_1's auc: 0.67826
[450]	training's auc: 0.858086	valid_1's auc: 0.678261
[460]	training's auc: 0.858086	valid_1's auc: 0.678261
[470]	training's auc: 0.858086	valid_1's auc: 0.678259
[480]	training's auc: 0.858087	valid_1's auc: 0.678259
[490]	training's auc: 0.858087	valid_1's auc: 0.678261
[500]	training's auc: 0.858087	valid_1's auc: 0.678261
[510]	training's auc: 0.858088	valid_1's auc: 0.678262
[520]	training's auc: 0.858088	valid_1's auc: 0.678262
[530]	training's auc: 0.858088	valid_1's auc: 0.678264
[540]	training's auc: 0.858088	valid_1's auc: 0.678265
[550]	training's auc: 0.858088	valid_1's auc: 0.678265
[560]	training's auc: 0.858088	valid_1's auc: 0.678262
[570]	training's auc: 0.858088	valid_1's auc: 0.678263
[580]	training's auc: 0.858088	valid_1's auc: 0.678264
[590]	training's auc: 0.858089	valid_1's auc: 0.678265
[600]	training's auc: 0.858089	valid_1's auc: 0.678265
[610]	training's auc: 0.858089	valid_1's auc: 0.678265
[620]	training's auc: 0.858089	valid_1's auc: 0.678265
[630]	training's auc: 0.858089	valid_1's auc: 0.678265
[640]	training's auc: 0.858089	valid_1's auc: 0.678267
[650]	training's auc: 0.858089	valid_1's auc: 0.678267
[660]	training's auc: 0.858089	valid_1's auc: 0.678266
[670]	training's auc: 0.858089	valid_1's auc: 0.678266
[680]	training's auc: 0.858089	valid_1's auc: 0.678266
[690]	training's auc: 0.858089	valid_1's auc: 0.678267
[700]	training's auc: 0.858089	valid_1's auc: 0.67827
[710]	training's auc: 0.858089	valid_1's auc: 0.678269
[720]	training's auc: 0.858089	valid_1's auc: 0.678269
[730]	training's auc: 0.858089	valid_1's auc: 0.678271
[740]	training's auc: 0.858089	valid_1's auc: 0.678268
[750]	training's auc: 0.858089	valid_1's auc: 0.678271
[760]	training's auc: 0.858089	valid_1's auc: 0.678272
[770]	training's auc: 0.858089	valid_1's auc: 0.678272
[780]	training's auc: 0.85809	valid_1's auc: 0.678273
[790]	training's auc: 0.85809	valid_1's auc: 0.678272
[800]	training's auc: 0.85809	valid_1's auc: 0.678271
[810]	training's auc: 0.85809	valid_1's auc: 0.67827
[820]	training's auc: 0.858089	valid_1's auc: 0.67827
[830]	training's auc: 0.85809	valid_1's auc: 0.678271
[840]	training's auc: 0.85809	valid_1's auc: 0.678269
[850]	training's auc: 0.85809	valid_1's auc: 0.678269
[860]	training's auc: 0.85809	valid_1's auc: 0.678269
[870]	training's auc: 0.85809	valid_1's auc: 0.678269
[880]	training's auc: 0.85809	valid_1's auc: 0.678269
[890]	training's auc: 0.85809	valid_1's auc: 0.678272
[900]	training's auc: 0.85809	valid_1's auc: 0.678272
[910]	training's auc: 0.85809	valid_1's auc: 0.67827
[920]	training's auc: 0.85809	valid_1's auc: 0.678269
[930]	training's auc: 0.85809	valid_1's auc: 0.678269
[940]	training's auc: 0.85809	valid_1's auc: 0.678271
[950]	training's auc: 0.85809	valid_1's auc: 0.678271
[960]	training's auc: 0.85809	valid_1's auc: 0.67827
[970]	training's auc: 0.85809	valid_1's auc: 0.678269
Early stopping, best iteration is:
[776]	training's auc: 0.85809	valid_1's auc: 0.678273
best score: 0.678272787597
best iteration: 776
complete on: ISC_isrc_rest_log10

                       ISCZ_isrc_rest:  0.678240753625
                  ISC_isrc_rest_log10:  0.678272787597
             IMC_bd_fixed_range_log10:  0.678314446116
                        ISCZ_lyricist:  0.678314469513
                   IMC_bd_range_log10:  0.678315104346
                   ISC_lyricist_log10:  0.67831551792
                       bd_range_log10:  0.678318367582
                 bd_fixed_range_log10:  0.678318531259
                  IMC_age_guess_log10:  0.678319300627
            IMC_age_guess_range_log10:  0.678322408093
                age_guess_range_log10:  0.678322704119
                         IMC_bd_log10:  0.678323298962
                       bd_fixed_log10:  0.678323457558
                   IMC_bd_fixed_log10:  0.678325074431
                      age_guess_log10:  0.678326130207
                             bd_log10:  0.678331922299
                         ISC_composer:  0.678335017117
                     ISC_top3_in_song:  0.678336423949
                ISC_artist_name_log10:  0.678337026509
                     ISCZ_artist_name:  0.678340120589
                  ISCZ_composer_log10:  0.678340835132
                            ISC_rc_ln:  0.678346758606
                        ISC_genre_ids:  0.678348873864
              ISCZ_song_country_log10:  0.678348875478
                  ISC_song_country_ln:  0.678349104814
                 ISCZ_genre_ids_log10:  0.678352632886
                    song_length_log10:  0.678352732936
                              ISCZ_rc:  0.678356605986
                     ISC_top1_in_song:  0.678357038556
                       ISC_name_log10:  0.678370827447
                     ISC_top2_in_song:  0.678372764534
                         ISCZ_name_ln:  0.678379418961
                         ISC_language:  0.678398508255
           IMC_expiration_month_log10:  0.678424316512
               expiration_month_log10:  0.678426182425
                      membership_days:  0.678518360679
            IMC_membership_days_log10:  0.678624369088
                        ISC_song_year:  0.679289509667
                       ISCZ_song_year:  0.679322087335
                            song_year:  0.679322358459

[timer]: complete in 201m 47s

Process finished with exit code 0
'''