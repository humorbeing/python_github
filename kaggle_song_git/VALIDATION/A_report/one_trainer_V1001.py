import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()
result = {}
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set.csv'

df = read_df(load_name)
# cols = [
#     'msno',
#     'song_id',
#     'artist_name',
#     'top1_in_song',
#     'top2_in_song',
#     'top3_in_song',
#     'language',
#     'song_year',
# ]
# df = add_ITC(df, cols)
show_df(df)

num_boost_round = 2000
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.022
num_leaves = 511
max_depth = 31

max_bin = 63
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
fixed = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',

    # 'top3_in_song',
    # 'rc',

    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'FAKE_1512883008',
]

work_on = [
    # 'ITC_msno',
    # 'CC11_msno',
    # 'ITC_name',
    # 'language',
    'top2_in_song',
    # 'CC11_name',
    # 'ITC_song_id_log10',
    # 'ITC_song_id_log10_1',
    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    # 'ITC_msno_log10_1',
    # 'ITC_msno_x_1',
    # 'OinC_msno',
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
        show_df(df_on)
        save_me = True
        # save_me = False
        if save_me:
            save_df(df_on)

        train, val = fake_df(df_on)
        del df_on
        model, cols = val_df(
            params, train, val,
            num_boost_round,
            early_stopping_rounds,
            verbose_eval,
            learning_rate=False
        )
        del train, val
        print('complete on:', w)
        result[w] = show_mo(model)


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))
for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION_fake_feature_insert_V1001/one_trainer_V1001.py

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
target                    uint8
gender                 category
artist_name            category
composer               category
lyricist               category
language               category
name                   category
song_year              category
song_country           category
rc                     category
isrc_rest              category
top1_in_song           category
top2_in_song           category
top3_in_song           category
ITC_msno                  int64
ITC_song_id               int64
ITC_msno_log10_1        float64
ITC_song_id_log10_1     float64
dtype: object
number of rows: 7377418
number of columns: 23
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
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
'ITC_msno',
'ITC_song_id',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
working on: top2_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
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
number of rows: 7377418
number of columns: 11
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.790377	valid_1's auc: 0.664638
[20]	training's auc: 0.794031	valid_1's auc: 0.666311
[30]	training's auc: 0.800408	valid_1's auc: 0.669519
[40]	training's auc: 0.803111	valid_1's auc: 0.670627
[50]	training's auc: 0.806707	valid_1's auc: 0.671939
[60]	training's auc: 0.809536	valid_1's auc: 0.672858
[70]	training's auc: 0.812457	valid_1's auc: 0.673949
[80]	training's auc: 0.814236	valid_1's auc: 0.674637
[90]	training's auc: 0.817227	valid_1's auc: 0.67585
[100]	training's auc: 0.820226	valid_1's auc: 0.677057
[110]	training's auc: 0.822822	valid_1's auc: 0.678169
[120]	training's auc: 0.825386	valid_1's auc: 0.679137
[130]	training's auc: 0.827806	valid_1's auc: 0.68009
[140]	training's auc: 0.830009	valid_1's auc: 0.680855
[150]	training's auc: 0.832325	valid_1's auc: 0.681781
[160]	training's auc: 0.834378	valid_1's auc: 0.682506
[170]	training's auc: 0.836436	valid_1's auc: 0.683224
[180]	training's auc: 0.838213	valid_1's auc: 0.683819
[190]	training's auc: 0.839897	valid_1's auc: 0.684367
[200]	training's auc: 0.841473	valid_1's auc: 0.684895
[210]	training's auc: 0.842875	valid_1's auc: 0.685291
[220]	training's auc: 0.844104	valid_1's auc: 0.68563
[230]	training's auc: 0.845321	valid_1's auc: 0.685914
[240]	training's auc: 0.846348	valid_1's auc: 0.686171
[250]	training's auc: 0.847451	valid_1's auc: 0.686402
[260]	training's auc: 0.848348	valid_1's auc: 0.686563
[270]	training's auc: 0.849362	valid_1's auc: 0.686722
[280]	training's auc: 0.850326	valid_1's auc: 0.686844
[290]	training's auc: 0.851061	valid_1's auc: 0.686932
[300]	training's auc: 0.851841	valid_1's auc: 0.687044
[310]	training's auc: 0.852686	valid_1's auc: 0.687197
[320]	training's auc: 0.853456	valid_1's auc: 0.687263
[330]	training's auc: 0.854233	valid_1's auc: 0.687257
[340]	training's auc: 0.854866	valid_1's auc: 0.687304
[350]	training's auc: 0.855547	valid_1's auc: 0.687313
[360]	training's auc: 0.856225	valid_1's auc: 0.687351
[370]	training's auc: 0.856869	valid_1's auc: 0.687387
[380]	training's auc: 0.857529	valid_1's auc: 0.687424
[390]	training's auc: 0.858168	valid_1's auc: 0.687464
[400]	training's auc: 0.858764	valid_1's auc: 0.687516
[410]	training's auc: 0.8593	valid_1's auc: 0.687535
[420]	training's auc: 0.859803	valid_1's auc: 0.687562
[430]	training's auc: 0.860381	valid_1's auc: 0.687575
[440]	training's auc: 0.860894	valid_1's auc: 0.687589
[450]	training's auc: 0.861363	valid_1's auc: 0.687623
[460]	training's auc: 0.861858	valid_1's auc: 0.687625
[470]	training's auc: 0.862344	valid_1's auc: 0.687632
[480]	training's auc: 0.862818	valid_1's auc: 0.687631
[490]	training's auc: 0.863324	valid_1's auc: 0.687638
[500]	training's auc: 0.863767	valid_1's auc: 0.687641
[510]	training's auc: 0.864251	valid_1's auc: 0.687692
[520]	training's auc: 0.864715	valid_1's auc: 0.687688
[530]	training's auc: 0.865149	valid_1's auc: 0.687684
[540]	training's auc: 0.86563	valid_1's auc: 0.68769
[550]	training's auc: 0.866096	valid_1's auc: 0.687701
[560]	training's auc: 0.86651	valid_1's auc: 0.687711
[570]	training's auc: 0.866943	valid_1's auc: 0.687734
[580]	training's auc: 0.867401	valid_1's auc: 0.687769
[590]	training's auc: 0.867819	valid_1's auc: 0.687773
[600]	training's auc: 0.868184	valid_1's auc: 0.687775
[610]	training's auc: 0.868586	valid_1's auc: 0.687776
[620]	training's auc: 0.869008	valid_1's auc: 0.687802
[630]	training's auc: 0.869373	valid_1's auc: 0.687803
[640]	training's auc: 0.869783	valid_1's auc: 0.687822
[650]	training's auc: 0.870185	valid_1's auc: 0.687809
[660]	training's auc: 0.870584	valid_1's auc: 0.687833
[670]	training's auc: 0.871006	valid_1's auc: 0.68786
[680]	training's auc: 0.871354	valid_1's auc: 0.687862
[690]	training's auc: 0.871684	valid_1's auc: 0.68785
[700]	training's auc: 0.872021	valid_1's auc: 0.687851
[710]	training's auc: 0.872384	valid_1's auc: 0.687863
[720]	training's auc: 0.872718	valid_1's auc: 0.687864
Early stopping, best iteration is:
[678]	training's auc: 0.871292	valid_1's auc: 0.687873
complete on: top2_in_song
model:
best score: 0.687873241605
best iteration: 678

                msno : 203332
             song_id : 46414
   source_system_tab : 630
  source_screen_name : 2083
         source_type : 1584
         artist_name : 79720
           song_year : 3648
 ITC_song_id_log10_1 : 2740
    ITC_msno_log10_1 : 3121
        top2_in_song : 2508
                         top2_in_song:  0.687873241605

[timer]: complete in 60m 14s

Process finished with exit code 0
'''