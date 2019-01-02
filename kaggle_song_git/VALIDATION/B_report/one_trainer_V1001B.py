import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set.csv'

def intme(x):
    return int(x)

df = read_df(load_name)
df['song_year'] = df['song_year'].astype(object)
df['song_year_int'] = df['song_year'].apply(intme).astype(np.int64)
df['song_year'] = df['song_year'].astype('category')

# show_df(df)
cols = [
    'msno',
    'song_id',
    'artist_name',
    'top1_in_song',
    'top2_in_song',
    'top3_in_song',
    'language',
    'song_year',
    'composer',
    'lyricist',
    'source_screen_name',
    'source_type',
]
df = add_ITC(df, cols)

show_df(df)


num_boost_round = 800
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
    'top2_in_song',
    # 'top3_in_song',
    # 'rc',

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
    # 'language',

    # 'CC11_name',
    'song_year_int',
    'ITC_song_year_log10_1',
    'ITC_source_screen_name_log10_1',
    'ITC_source_type_log10_1',
    'ITC_language_log10_1',
    'ITC_top1_in_song_log10_1',
    'ITC_top2_in_song_log10_1',
    'ITC_top3_in_song_log10_1',
    'ITC_composer_log10_1',
    'ITC_lyricist_log10_1',
    'ITC_artist_name_log10_1',

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
        # save_me = True
        save_me = False
        if save_me:
            save_df(df_on)

        train, val = fake_df(df_on)
        del df_on
        model, cols = val_df(
            params, train, val,
            num_boost_round,
            early_stopping_rounds,
            verbose_eval
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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION/one_trainer_V1001B.py

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                              category
song_id                           category
source_system_tab                 category
source_screen_name                category
source_type                       category
target                               uint8
gender                            category
artist_name                       category
composer                          category
lyricist                          category
language                          category
name                              category
song_year                         category
song_country                      category
rc                                category
isrc_rest                         category
top1_in_song                      category
top2_in_song                      category
top3_in_song                      category
song_year_int                        int64
ITC_msno                             int64
ITC_song_id                          int64
ITC_artist_name                      int64
ITC_top1_in_song                     int64
ITC_top2_in_song                     int64
ITC_top3_in_song                     int64
ITC_language                         int64
ITC_song_year                        int64
ITC_composer                         int64
ITC_lyricist                         int64
ITC_source_screen_name               int64
ITC_source_type                      int64
ITC_msno_log10_1                   float16
ITC_song_id_log10_1                float16
ITC_artist_name_log10_1            float16
ITC_top1_in_song_log10_1           float16
ITC_top2_in_song_log10_1           float16
ITC_top3_in_song_log10_1           float16
ITC_language_log10_1               float16
ITC_song_year_log10_1              float16
ITC_composer_log10_1               float16
ITC_lyricist_log10_1               float16
ITC_source_screen_name_log10_1     float16
ITC_source_type_log10_1            float16
dtype: object
number of rows: 7377418
number of columns: 44

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
'song_year_int',
'ITC_msno',
'ITC_song_id',
'ITC_artist_name',
'ITC_top1_in_song',
'ITC_top2_in_song',
'ITC_top3_in_song',
'ITC_language',
'ITC_song_year',
'ITC_composer',
'ITC_lyricist',
'ITC_source_screen_name',
'ITC_source_type',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_artist_name_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_language_log10_1',
'ITC_song_year_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',

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
'song_year_int',
'ITC_msno',
'ITC_song_id',
'ITC_artist_name',
'ITC_top1_in_song',
'ITC_top2_in_song',
'ITC_top3_in_song',
'ITC_language',
'ITC_song_year',
'ITC_composer',
'ITC_lyricist',
'ITC_source_screen_name',
'ITC_source_type',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_artist_name_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_language_log10_1',
'ITC_song_year_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
working on: song_year_int

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
ITC_song_id_log10_1     float16
ITC_msno_log10_1        float16
top2_in_song           category
song_year_int             int64
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'song_year_int',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.778141	valid_1's auc: 0.65957
[20]	training's auc: 0.784221	valid_1's auc: 0.662024
[30]	training's auc: 0.788005	valid_1's auc: 0.663463
[40]	training's auc: 0.790212	valid_1's auc: 0.66451
[50]	training's auc: 0.793695	valid_1's auc: 0.665748
[60]	training's auc: 0.797591	valid_1's auc: 0.6669
[70]	training's auc: 0.80039	valid_1's auc: 0.667926
[80]	training's auc: 0.803688	valid_1's auc: 0.669115
[90]	training's auc: 0.806194	valid_1's auc: 0.670027
[100]	training's auc: 0.80833	valid_1's auc: 0.670923
[110]	training's auc: 0.810601	valid_1's auc: 0.671724
[120]	training's auc: 0.812596	valid_1's auc: 0.672473
[130]	training's auc: 0.814507	valid_1's auc: 0.673292
[140]	training's auc: 0.816281	valid_1's auc: 0.674033
[150]	training's auc: 0.817843	valid_1's auc: 0.674635
[160]	training's auc: 0.819311	valid_1's auc: 0.67533
[170]	training's auc: 0.820417	valid_1's auc: 0.67584
[180]	training's auc: 0.821345	valid_1's auc: 0.676297
[190]	training's auc: 0.822106	valid_1's auc: 0.676699
[200]	training's auc: 0.822759	valid_1's auc: 0.677148
[210]	training's auc: 0.823479	valid_1's auc: 0.677483
[220]	training's auc: 0.824122	valid_1's auc: 0.677736
[230]	training's auc: 0.824722	valid_1's auc: 0.67802
[240]	training's auc: 0.825263	valid_1's auc: 0.678295
[250]	training's auc: 0.825782	valid_1's auc: 0.678527
[260]	training's auc: 0.826328	valid_1's auc: 0.678762
[270]	training's auc: 0.826884	valid_1's auc: 0.678985
[280]	training's auc: 0.827454	valid_1's auc: 0.679222
[290]	training's auc: 0.828022	valid_1's auc: 0.679451
[300]	training's auc: 0.828569	valid_1's auc: 0.679649
[310]	training's auc: 0.829113	valid_1's auc: 0.679876
[320]	training's auc: 0.829627	valid_1's auc: 0.680079
[330]	training's auc: 0.830062	valid_1's auc: 0.680227
[340]	training's auc: 0.830557	valid_1's auc: 0.680432
[350]	training's auc: 0.831022	valid_1's auc: 0.680599
[360]	training's auc: 0.831442	valid_1's auc: 0.680745
[370]	training's auc: 0.831835	valid_1's auc: 0.680906
[380]	training's auc: 0.832285	valid_1's auc: 0.681076
[390]	training's auc: 0.832713	valid_1's auc: 0.681243
[400]	training's auc: 0.833198	valid_1's auc: 0.681354
[410]	training's auc: 0.833571	valid_1's auc: 0.681491
[420]	training's auc: 0.83395	valid_1's auc: 0.681708
[430]	training's auc: 0.834377	valid_1's auc: 0.681878
[440]	training's auc: 0.834751	valid_1's auc: 0.681982
[450]	training's auc: 0.835155	valid_1's auc: 0.682088
[460]	training's auc: 0.835541	valid_1's auc: 0.682257
[470]	training's auc: 0.835934	valid_1's auc: 0.682369
[480]	training's auc: 0.836348	valid_1's auc: 0.682485
[490]	training's auc: 0.83668	valid_1's auc: 0.682587
[500]	training's auc: 0.837017	valid_1's auc: 0.682786
[510]	training's auc: 0.837427	valid_1's auc: 0.682878
[520]	training's auc: 0.837768	valid_1's auc: 0.682978
[530]	training's auc: 0.838146	valid_1's auc: 0.68308
[540]	training's auc: 0.838482	valid_1's auc: 0.683177
[550]	training's auc: 0.838847	valid_1's auc: 0.683356
[560]	training's auc: 0.839113	valid_1's auc: 0.683463
[570]	training's auc: 0.839456	valid_1's auc: 0.683559
[580]	training's auc: 0.839786	valid_1's auc: 0.683706
[590]	training's auc: 0.840129	valid_1's auc: 0.683786
[600]	training's auc: 0.840406	valid_1's auc: 0.68386
[610]	training's auc: 0.840682	valid_1's auc: 0.683911
[620]	training's auc: 0.840969	valid_1's auc: 0.683979
[630]	training's auc: 0.841282	valid_1's auc: 0.684058
[640]	training's auc: 0.841644	valid_1's auc: 0.684148
[650]	training's auc: 0.841953	valid_1's auc: 0.684208
[660]	training's auc: 0.842277	valid_1's auc: 0.684285
[670]	training's auc: 0.842565	valid_1's auc: 0.684345
[680]	training's auc: 0.842848	valid_1's auc: 0.684407
[690]	training's auc: 0.843182	valid_1's auc: 0.68454
[700]	training's auc: 0.843501	valid_1's auc: 0.684745
[710]	training's auc: 0.8438	valid_1's auc: 0.684791
[720]	training's auc: 0.844029	valid_1's auc: 0.684834
[730]	training's auc: 0.844254	valid_1's auc: 0.684862
[740]	training's auc: 0.844529	valid_1's auc: 0.685004
[750]	training's auc: 0.844775	valid_1's auc: 0.685057
[760]	training's auc: 0.845029	valid_1's auc: 0.685108
[770]	training's auc: 0.845318	valid_1's auc: 0.685155
[780]	training's auc: 0.845619	valid_1's auc: 0.685221
[790]	training's auc: 0.845899	valid_1's auc: 0.68528
[800]	training's auc: 0.84615	valid_1's auc: 0.685325
complete on: song_year_int
model:
best score: 0.685324837361
best iteration: 0

                msno : 55470
             song_id : 20889
   source_system_tab : 5148
  source_screen_name : 19777
         source_type : 14530
         artist_name : 65166
           song_year : 21692
 ITC_song_id_log10_1 : 71925
    ITC_msno_log10_1 : 78465
        top2_in_song : 15671
       song_year_int : 39267
working on: ITC_song_year_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                      uint8
msno                     category
song_id                  category
source_system_tab        category
source_screen_name       category
source_type              category
artist_name              category
song_year                category
ITC_song_id_log10_1       float16
ITC_msno_log10_1          float16
top2_in_song             category
ITC_song_year_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_song_year_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.776982	valid_1's auc: 0.659279
[20]	training's auc: 0.783998	valid_1's auc: 0.661847
[30]	training's auc: 0.787916	valid_1's auc: 0.663523
[40]	training's auc: 0.790188	valid_1's auc: 0.664552
[50]	training's auc: 0.794112	valid_1's auc: 0.665855
[60]	training's auc: 0.798092	valid_1's auc: 0.667075
[70]	training's auc: 0.800739	valid_1's auc: 0.668034
[80]	training's auc: 0.804026	valid_1's auc: 0.669292
[90]	training's auc: 0.80659	valid_1's auc: 0.670185
[100]	training's auc: 0.808591	valid_1's auc: 0.671032
[110]	training's auc: 0.810781	valid_1's auc: 0.671877
[120]	training's auc: 0.812769	valid_1's auc: 0.67267
[130]	training's auc: 0.814663	valid_1's auc: 0.673565
[140]	training's auc: 0.816416	valid_1's auc: 0.674248
[150]	training's auc: 0.818056	valid_1's auc: 0.674937
[160]	training's auc: 0.819436	valid_1's auc: 0.675586
[170]	training's auc: 0.820633	valid_1's auc: 0.67617
[180]	training's auc: 0.821469	valid_1's auc: 0.676572
[190]	training's auc: 0.822292	valid_1's auc: 0.677021
[200]	training's auc: 0.822889	valid_1's auc: 0.677384
[210]	training's auc: 0.823504	valid_1's auc: 0.677742
[220]	training's auc: 0.82408	valid_1's auc: 0.67798
[230]	training's auc: 0.824732	valid_1's auc: 0.678286
[240]	training's auc: 0.825245	valid_1's auc: 0.678537
[250]	training's auc: 0.825801	valid_1's auc: 0.678794
[260]	training's auc: 0.826305	valid_1's auc: 0.679019
[270]	training's auc: 0.826922	valid_1's auc: 0.679263
[280]	training's auc: 0.827446	valid_1's auc: 0.679518
[290]	training's auc: 0.828007	valid_1's auc: 0.679778
[300]	training's auc: 0.828564	valid_1's auc: 0.68
[310]	training's auc: 0.829142	valid_1's auc: 0.680244
[320]	training's auc: 0.829593	valid_1's auc: 0.680422
[330]	training's auc: 0.830105	valid_1's auc: 0.680657
[340]	training's auc: 0.830561	valid_1's auc: 0.680808
[350]	training's auc: 0.831013	valid_1's auc: 0.68098
[360]	training's auc: 0.831441	valid_1's auc: 0.681175
[370]	training's auc: 0.83182	valid_1's auc: 0.681315
[380]	training's auc: 0.832235	valid_1's auc: 0.681474
[390]	training's auc: 0.832692	valid_1's auc: 0.681616
[400]	training's auc: 0.833118	valid_1's auc: 0.68173
[410]	training's auc: 0.833524	valid_1's auc: 0.681873
[420]	training's auc: 0.833902	valid_1's auc: 0.68197
[430]	training's auc: 0.834304	valid_1's auc: 0.682094
[440]	training's auc: 0.834679	valid_1's auc: 0.682264
[450]	training's auc: 0.835084	valid_1's auc: 0.6824
[460]	training's auc: 0.835488	valid_1's auc: 0.682514
[470]	training's auc: 0.835885	valid_1's auc: 0.682622
[480]	training's auc: 0.836291	valid_1's auc: 0.682741
[490]	training's auc: 0.836643	valid_1's auc: 0.682833
[500]	training's auc: 0.837003	valid_1's auc: 0.682933
[510]	training's auc: 0.837371	valid_1's auc: 0.683029
[520]	training's auc: 0.837712	valid_1's auc: 0.683113
[530]	training's auc: 0.838055	valid_1's auc: 0.683256
[540]	training's auc: 0.838405	valid_1's auc: 0.683413
[550]	training's auc: 0.838739	valid_1's auc: 0.683511
[560]	training's auc: 0.839025	valid_1's auc: 0.683628
[570]	training's auc: 0.83937	valid_1's auc: 0.683712
[580]	training's auc: 0.839716	valid_1's auc: 0.683857
[590]	training's auc: 0.840093	valid_1's auc: 0.683952
[600]	training's auc: 0.840427	valid_1's auc: 0.684106
[610]	training's auc: 0.840729	valid_1's auc: 0.684186
[620]	training's auc: 0.841008	valid_1's auc: 0.68425
[630]	training's auc: 0.841329	valid_1's auc: 0.68432
[640]	training's auc: 0.841671	valid_1's auc: 0.684378
[650]	training's auc: 0.842014	valid_1's auc: 0.684442
[660]	training's auc: 0.842341	valid_1's auc: 0.684537
[670]	training's auc: 0.842615	valid_1's auc: 0.684605
[680]	training's auc: 0.842928	valid_1's auc: 0.684692
[690]	training's auc: 0.843242	valid_1's auc: 0.684771
[700]	training's auc: 0.843541	valid_1's auc: 0.684821
[710]	training's auc: 0.843841	valid_1's auc: 0.684884
[720]	training's auc: 0.844066	valid_1's auc: 0.684915
[730]	training's auc: 0.844288	valid_1's auc: 0.684946
[740]	training's auc: 0.844554	valid_1's auc: 0.684992
[750]	training's auc: 0.844811	valid_1's auc: 0.685031
[760]	training's auc: 0.845087	valid_1's auc: 0.685097
[770]	training's auc: 0.845375	valid_1's auc: 0.685141
[780]	training's auc: 0.84569	valid_1's auc: 0.685208
[790]	training's auc: 0.84596	valid_1's auc: 0.685307
[800]	training's auc: 0.846227	valid_1's auc: 0.685369
complete on: ITC_song_year_log10_1
model:
best score: 0.685369186765
best iteration: 0

                msno : 55307
             song_id : 20842
   source_system_tab : 5211
  source_screen_name : 19723
         source_type : 14481
         artist_name : 65211
           song_year : 22018
 ITC_song_id_log10_1 : 71413
    ITC_msno_log10_1 : 77930
        top2_in_song : 15941
ITC_song_year_log10_1 : 39923
working on: ITC_source_screen_name_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                               uint8
msno                              category
song_id                           category
source_system_tab                 category
source_screen_name                category
source_type                       category
artist_name                       category
song_year                         category
ITC_song_id_log10_1                float16
ITC_msno_log10_1                   float16
top2_in_song                      category
ITC_source_screen_name_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_source_screen_name_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.778091	valid_1's auc: 0.660314
[20]	training's auc: 0.784172	valid_1's auc: 0.662491
[30]	training's auc: 0.788053	valid_1's auc: 0.664004
[40]	training's auc: 0.790071	valid_1's auc: 0.664879
[50]	training's auc: 0.793701	valid_1's auc: 0.66601
[60]	training's auc: 0.797824	valid_1's auc: 0.667282
[70]	training's auc: 0.80052	valid_1's auc: 0.668252
[80]	training's auc: 0.803738	valid_1's auc: 0.669585
[90]	training's auc: 0.806201	valid_1's auc: 0.670452
[100]	training's auc: 0.808132	valid_1's auc: 0.671241
[110]	training's auc: 0.810412	valid_1's auc: 0.672129
[120]	training's auc: 0.81254	valid_1's auc: 0.672929
[130]	training's auc: 0.814328	valid_1's auc: 0.673734
[140]	training's auc: 0.816095	valid_1's auc: 0.674489
[150]	training's auc: 0.817649	valid_1's auc: 0.675144
[160]	training's auc: 0.819125	valid_1's auc: 0.675812
[170]	training's auc: 0.820338	valid_1's auc: 0.676365
[180]	training's auc: 0.821169	valid_1's auc: 0.676784
[190]	training's auc: 0.82209	valid_1's auc: 0.677225
[200]	training's auc: 0.822624	valid_1's auc: 0.677557
[210]	training's auc: 0.823246	valid_1's auc: 0.678018
[220]	training's auc: 0.82389	valid_1's auc: 0.678292
[230]	training's auc: 0.82451	valid_1's auc: 0.678578
[240]	training's auc: 0.825095	valid_1's auc: 0.678889
[250]	training's auc: 0.825655	valid_1's auc: 0.679134
[260]	training's auc: 0.826151	valid_1's auc: 0.679374
[270]	training's auc: 0.826693	valid_1's auc: 0.679585
[280]	training's auc: 0.827256	valid_1's auc: 0.679823
[290]	training's auc: 0.827792	valid_1's auc: 0.680038
[300]	training's auc: 0.828305	valid_1's auc: 0.68022
[310]	training's auc: 0.828815	valid_1's auc: 0.680432
[320]	training's auc: 0.82931	valid_1's auc: 0.680626
[330]	training's auc: 0.829792	valid_1's auc: 0.680802
[340]	training's auc: 0.830226	valid_1's auc: 0.680955
[350]	training's auc: 0.830735	valid_1's auc: 0.681166
[360]	training's auc: 0.8312	valid_1's auc: 0.681332
[370]	training's auc: 0.831582	valid_1's auc: 0.681451
[380]	training's auc: 0.832071	valid_1's auc: 0.681613
[390]	training's auc: 0.832573	valid_1's auc: 0.681807
[400]	training's auc: 0.833005	valid_1's auc: 0.681918
[410]	training's auc: 0.833391	valid_1's auc: 0.682034
[420]	training's auc: 0.833728	valid_1's auc: 0.682113
[430]	training's auc: 0.834138	valid_1's auc: 0.682218
[440]	training's auc: 0.834535	valid_1's auc: 0.682346
[450]	training's auc: 0.834935	valid_1's auc: 0.682471
[460]	training's auc: 0.835319	valid_1's auc: 0.682579
[470]	training's auc: 0.83569	valid_1's auc: 0.682676
[480]	training's auc: 0.836095	valid_1's auc: 0.682803
[490]	training's auc: 0.836432	valid_1's auc: 0.682892
[500]	training's auc: 0.836792	valid_1's auc: 0.682976
[510]	training's auc: 0.837198	valid_1's auc: 0.683088
[520]	training's auc: 0.837529	valid_1's auc: 0.68318
[530]	training's auc: 0.837857	valid_1's auc: 0.683258
[540]	training's auc: 0.838207	valid_1's auc: 0.683387
[550]	training's auc: 0.838533	valid_1's auc: 0.683471
[560]	training's auc: 0.83883	valid_1's auc: 0.683508
[570]	training's auc: 0.839204	valid_1's auc: 0.683599
[580]	training's auc: 0.839517	valid_1's auc: 0.683692
[590]	training's auc: 0.839894	valid_1's auc: 0.683796
[600]	training's auc: 0.840193	valid_1's auc: 0.683875
[610]	training's auc: 0.840472	valid_1's auc: 0.683929
[620]	training's auc: 0.840826	valid_1's auc: 0.684014
[630]	training's auc: 0.841139	valid_1's auc: 0.684068
[640]	training's auc: 0.841479	valid_1's auc: 0.684129
[650]	training's auc: 0.841817	valid_1's auc: 0.684188
[660]	training's auc: 0.842155	valid_1's auc: 0.68424
[670]	training's auc: 0.842447	valid_1's auc: 0.684329
[680]	training's auc: 0.842744	valid_1's auc: 0.684407
[690]	training's auc: 0.843064	valid_1's auc: 0.684478
[700]	training's auc: 0.843352	valid_1's auc: 0.684532
[710]	training's auc: 0.843659	valid_1's auc: 0.684605
[720]	training's auc: 0.84388	valid_1's auc: 0.684645
[730]	training's auc: 0.844116	valid_1's auc: 0.684724
[740]	training's auc: 0.844355	valid_1's auc: 0.684768
[750]	training's auc: 0.844618	valid_1's auc: 0.684824
[760]	training's auc: 0.844837	valid_1's auc: 0.684866
[770]	training's auc: 0.845132	valid_1's auc: 0.684894
[780]	training's auc: 0.845433	valid_1's auc: 0.684952
[790]	training's auc: 0.84573	valid_1's auc: 0.68501
[800]	training's auc: 0.846039	valid_1's auc: 0.685091
complete on: ITC_source_screen_name_log10_1
model:
best score: 0.685091007661
best iteration: 0

                msno : 55207
             song_id : 20465
   source_system_tab : 4692
  source_screen_name : 16170
         source_type : 13010
         artist_name : 66061
           song_year : 23930
 ITC_song_id_log10_1 : 74440
    ITC_msno_log10_1 : 80157
        top2_in_song : 16202
ITC_source_screen_name_log10_1 : 37529
working on: ITC_source_type_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                        uint8
msno                       category
song_id                    category
source_system_tab          category
source_screen_name         category
source_type                category
artist_name                category
song_year                  category
ITC_song_id_log10_1         float16
ITC_msno_log10_1            float16
top2_in_song               category
ITC_source_type_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_source_type_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.777655	valid_1's auc: 0.659503
[20]	training's auc: 0.784276	valid_1's auc: 0.662024
[30]	training's auc: 0.788046	valid_1's auc: 0.66368
[40]	training's auc: 0.789969	valid_1's auc: 0.664623
[50]	training's auc: 0.793946	valid_1's auc: 0.665951
[60]	training's auc: 0.797911	valid_1's auc: 0.667135
[70]	training's auc: 0.800609	valid_1's auc: 0.668046
[80]	training's auc: 0.80399	valid_1's auc: 0.669361
[90]	training's auc: 0.806343	valid_1's auc: 0.670163
[100]	training's auc: 0.80841	valid_1's auc: 0.671038
[110]	training's auc: 0.8107	valid_1's auc: 0.671846
[120]	training's auc: 0.812828	valid_1's auc: 0.672706
[130]	training's auc: 0.814743	valid_1's auc: 0.673468
[140]	training's auc: 0.816417	valid_1's auc: 0.67416
[150]	training's auc: 0.817936	valid_1's auc: 0.674804
[160]	training's auc: 0.819372	valid_1's auc: 0.675489
[170]	training's auc: 0.820497	valid_1's auc: 0.676046
[180]	training's auc: 0.821339	valid_1's auc: 0.67643
[190]	training's auc: 0.822086	valid_1's auc: 0.676753
[200]	training's auc: 0.822702	valid_1's auc: 0.677119
[210]	training's auc: 0.823407	valid_1's auc: 0.677519
[220]	training's auc: 0.824051	valid_1's auc: 0.6778
[230]	training's auc: 0.824644	valid_1's auc: 0.678065
[240]	training's auc: 0.825182	valid_1's auc: 0.678306
[250]	training's auc: 0.825696	valid_1's auc: 0.678563
[260]	training's auc: 0.826237	valid_1's auc: 0.678785
[270]	training's auc: 0.826857	valid_1's auc: 0.679028
[280]	training's auc: 0.827347	valid_1's auc: 0.679224
[290]	training's auc: 0.827868	valid_1's auc: 0.679427
[300]	training's auc: 0.828456	valid_1's auc: 0.679672
[310]	training's auc: 0.828959	valid_1's auc: 0.679836
[320]	training's auc: 0.829472	valid_1's auc: 0.680072
[330]	training's auc: 0.829964	valid_1's auc: 0.680266
[340]	training's auc: 0.830416	valid_1's auc: 0.680455
[350]	training's auc: 0.830873	valid_1's auc: 0.680632
[360]	training's auc: 0.831311	valid_1's auc: 0.680799
[370]	training's auc: 0.831685	valid_1's auc: 0.68096
[380]	training's auc: 0.832122	valid_1's auc: 0.681121
[390]	training's auc: 0.832555	valid_1's auc: 0.681276
[400]	training's auc: 0.832982	valid_1's auc: 0.6814
[410]	training's auc: 0.83338	valid_1's auc: 0.681515
[420]	training's auc: 0.833774	valid_1's auc: 0.681629
[430]	training's auc: 0.834158	valid_1's auc: 0.681745
[440]	training's auc: 0.834556	valid_1's auc: 0.681885
[450]	training's auc: 0.834986	valid_1's auc: 0.682021
[460]	training's auc: 0.835358	valid_1's auc: 0.682119
[470]	training's auc: 0.835726	valid_1's auc: 0.682213
[480]	training's auc: 0.836128	valid_1's auc: 0.682327
[490]	training's auc: 0.836481	valid_1's auc: 0.682439
[500]	training's auc: 0.83685	valid_1's auc: 0.682604
[510]	training's auc: 0.837204	valid_1's auc: 0.682684
[520]	training's auc: 0.837554	valid_1's auc: 0.682774
[530]	training's auc: 0.837898	valid_1's auc: 0.682882
[540]	training's auc: 0.838244	valid_1's auc: 0.682982
[550]	training's auc: 0.83857	valid_1's auc: 0.683074
[560]	training's auc: 0.838871	valid_1's auc: 0.683211
[570]	training's auc: 0.839205	valid_1's auc: 0.68331
[580]	training's auc: 0.839544	valid_1's auc: 0.683408
[590]	training's auc: 0.839908	valid_1's auc: 0.683499
[600]	training's auc: 0.840212	valid_1's auc: 0.683574
[610]	training's auc: 0.840497	valid_1's auc: 0.683656
[620]	training's auc: 0.840794	valid_1's auc: 0.683715
[630]	training's auc: 0.84111	valid_1's auc: 0.683779
[640]	training's auc: 0.841459	valid_1's auc: 0.683864
[650]	training's auc: 0.8418	valid_1's auc: 0.683938
[660]	training's auc: 0.842129	valid_1's auc: 0.684014
[670]	training's auc: 0.842397	valid_1's auc: 0.684123
[680]	training's auc: 0.842671	valid_1's auc: 0.684173
[690]	training's auc: 0.842966	valid_1's auc: 0.684232
[700]	training's auc: 0.843241	valid_1's auc: 0.684281
[710]	training's auc: 0.843536	valid_1's auc: 0.684314
[720]	training's auc: 0.843759	valid_1's auc: 0.684353
[730]	training's auc: 0.843997	valid_1's auc: 0.684404
[740]	training's auc: 0.844266	valid_1's auc: 0.684519
[750]	training's auc: 0.844515	valid_1's auc: 0.684561
[760]	training's auc: 0.844755	valid_1's auc: 0.684605
[770]	training's auc: 0.845041	valid_1's auc: 0.684657
[780]	training's auc: 0.845383	valid_1's auc: 0.684727
[790]	training's auc: 0.845659	valid_1's auc: 0.684787
[800]	training's auc: 0.845928	valid_1's auc: 0.684833
complete on: ITC_source_type_log10_1
model:
best score: 0.684832560552
best iteration: 0

                msno : 55597
             song_id : 20722
   source_system_tab : 4663
  source_screen_name : 18442
         source_type : 10838
         artist_name : 65869
           song_year : 24111
 ITC_song_id_log10_1 : 75084
    ITC_msno_log10_1 : 79268
        top2_in_song : 16055
ITC_source_type_log10_1 : 37192
working on: ITC_language_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float16
ITC_msno_log10_1         float16
top2_in_song            category
ITC_language_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_language_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.777579	valid_1's auc: 0.660092
[20]	training's auc: 0.784346	valid_1's auc: 0.662738
[30]	training's auc: 0.788221	valid_1's auc: 0.664193
[40]	training's auc: 0.790301	valid_1's auc: 0.665121
[50]	training's auc: 0.793703	valid_1's auc: 0.666175
[60]	training's auc: 0.797712	valid_1's auc: 0.667441
[70]	training's auc: 0.800541	valid_1's auc: 0.668482
[80]	training's auc: 0.803766	valid_1's auc: 0.66967
[90]	training's auc: 0.806329	valid_1's auc: 0.670509
[100]	training's auc: 0.808532	valid_1's auc: 0.671424
[110]	training's auc: 0.810602	valid_1's auc: 0.67227
[120]	training's auc: 0.812706	valid_1's auc: 0.673026
[130]	training's auc: 0.814542	valid_1's auc: 0.673788
[140]	training's auc: 0.816263	valid_1's auc: 0.674481
[150]	training's auc: 0.817792	valid_1's auc: 0.675095
[160]	training's auc: 0.819151	valid_1's auc: 0.675729
[170]	training's auc: 0.820373	valid_1's auc: 0.676294
[180]	training's auc: 0.821255	valid_1's auc: 0.676746
[190]	training's auc: 0.822019	valid_1's auc: 0.677079
[200]	training's auc: 0.822644	valid_1's auc: 0.677445
[210]	training's auc: 0.823339	valid_1's auc: 0.677794
[220]	training's auc: 0.823988	valid_1's auc: 0.678058
[230]	training's auc: 0.824578	valid_1's auc: 0.67833
[240]	training's auc: 0.82517	valid_1's auc: 0.678613
[250]	training's auc: 0.825727	valid_1's auc: 0.678891
[260]	training's auc: 0.82625	valid_1's auc: 0.679118
[270]	training's auc: 0.826888	valid_1's auc: 0.679379
[280]	training's auc: 0.827403	valid_1's auc: 0.679566
[290]	training's auc: 0.827962	valid_1's auc: 0.679816
[300]	training's auc: 0.828512	valid_1's auc: 0.680009
[310]	training's auc: 0.829018	valid_1's auc: 0.680197
[320]	training's auc: 0.829547	valid_1's auc: 0.680408
[330]	training's auc: 0.830039	valid_1's auc: 0.680596
[340]	training's auc: 0.830513	valid_1's auc: 0.680771
[350]	training's auc: 0.830967	valid_1's auc: 0.68094
[360]	training's auc: 0.83143	valid_1's auc: 0.681137
[370]	training's auc: 0.831842	valid_1's auc: 0.681288
[380]	training's auc: 0.832358	valid_1's auc: 0.681457
[390]	training's auc: 0.832786	valid_1's auc: 0.6816
[400]	training's auc: 0.833176	valid_1's auc: 0.681682
[410]	training's auc: 0.833572	valid_1's auc: 0.681817
[420]	training's auc: 0.833979	valid_1's auc: 0.681924
[430]	training's auc: 0.83438	valid_1's auc: 0.682047
[440]	training's auc: 0.834782	valid_1's auc: 0.682161
[450]	training's auc: 0.835199	valid_1's auc: 0.6823
[460]	training's auc: 0.835583	valid_1's auc: 0.682406
[470]	training's auc: 0.835959	valid_1's auc: 0.682533
[480]	training's auc: 0.836351	valid_1's auc: 0.682663
[490]	training's auc: 0.836692	valid_1's auc: 0.682752
[500]	training's auc: 0.837073	valid_1's auc: 0.682866
[510]	training's auc: 0.837425	valid_1's auc: 0.682944
[520]	training's auc: 0.837767	valid_1's auc: 0.683021
[530]	training's auc: 0.838117	valid_1's auc: 0.683125
[540]	training's auc: 0.838473	valid_1's auc: 0.683198
[550]	training's auc: 0.838784	valid_1's auc: 0.683268
[560]	training's auc: 0.839072	valid_1's auc: 0.683316
[570]	training's auc: 0.839429	valid_1's auc: 0.683392
[580]	training's auc: 0.839747	valid_1's auc: 0.683467
[590]	training's auc: 0.840116	valid_1's auc: 0.68354
[600]	training's auc: 0.840393	valid_1's auc: 0.683598
[610]	training's auc: 0.840695	valid_1's auc: 0.683654
[620]	training's auc: 0.841001	valid_1's auc: 0.683709
[630]	training's auc: 0.841327	valid_1's auc: 0.683783
[640]	training's auc: 0.841682	valid_1's auc: 0.683842
[650]	training's auc: 0.841991	valid_1's auc: 0.683908
[660]	training's auc: 0.842315	valid_1's auc: 0.683984
[670]	training's auc: 0.842594	valid_1's auc: 0.68403
[680]	training's auc: 0.842877	valid_1's auc: 0.684105
[690]	training's auc: 0.843183	valid_1's auc: 0.684188
[700]	training's auc: 0.84351	valid_1's auc: 0.684274
[710]	training's auc: 0.843813	valid_1's auc: 0.684329
[720]	training's auc: 0.844063	valid_1's auc: 0.684384
[730]	training's auc: 0.844353	valid_1's auc: 0.684442
[740]	training's auc: 0.844601	valid_1's auc: 0.6845
[750]	training's auc: 0.844856	valid_1's auc: 0.684558
[760]	training's auc: 0.84506	valid_1's auc: 0.684596
[770]	training's auc: 0.845356	valid_1's auc: 0.684639
[780]	training's auc: 0.845643	valid_1's auc: 0.684676
[790]	training's auc: 0.845903	valid_1's auc: 0.684719
[800]	training's auc: 0.846177	valid_1's auc: 0.684763
complete on: ITC_language_log10_1
model:
best score: 0.684762665487
best iteration: 0

                msno : 55154
             song_id : 20879
   source_system_tab : 5181
  source_screen_name : 20037
         source_type : 14811
         artist_name : 65966
           song_year : 24102
 ITC_song_id_log10_1 : 75492
    ITC_msno_log10_1 : 82125
        top2_in_song : 15182
ITC_language_log10_1 : 28980
working on: ITC_top1_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_system_tab           category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float16
ITC_msno_log10_1             float16
top2_in_song                category
ITC_top1_in_song_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_top1_in_song_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.778087	valid_1's auc: 0.660504
[20]	training's auc: 0.784426	valid_1's auc: 0.662779
[30]	training's auc: 0.788495	valid_1's auc: 0.664442
[40]	training's auc: 0.790207	valid_1's auc: 0.665233
[50]	training's auc: 0.793657	valid_1's auc: 0.666404
[60]	training's auc: 0.797724	valid_1's auc: 0.66766
[70]	training's auc: 0.800514	valid_1's auc: 0.668684
[80]	training's auc: 0.803896	valid_1's auc: 0.669934
[90]	training's auc: 0.806408	valid_1's auc: 0.670775
[100]	training's auc: 0.808459	valid_1's auc: 0.671615
[110]	training's auc: 0.810522	valid_1's auc: 0.672395
[120]	training's auc: 0.81271	valid_1's auc: 0.673221
[130]	training's auc: 0.814623	valid_1's auc: 0.674056
[140]	training's auc: 0.816281	valid_1's auc: 0.674725
[150]	training's auc: 0.817833	valid_1's auc: 0.675345
[160]	training's auc: 0.819239	valid_1's auc: 0.676013
[170]	training's auc: 0.820521	valid_1's auc: 0.676615
[180]	training's auc: 0.82134	valid_1's auc: 0.677045
[190]	training's auc: 0.822144	valid_1's auc: 0.677427
[200]	training's auc: 0.822725	valid_1's auc: 0.677796
[210]	training's auc: 0.823302	valid_1's auc: 0.678142
[220]	training's auc: 0.823992	valid_1's auc: 0.678446
[230]	training's auc: 0.824617	valid_1's auc: 0.678753
[240]	training's auc: 0.825185	valid_1's auc: 0.679026
[250]	training's auc: 0.825767	valid_1's auc: 0.679324
[260]	training's auc: 0.826285	valid_1's auc: 0.679552
[270]	training's auc: 0.826787	valid_1's auc: 0.679724
[280]	training's auc: 0.827371	valid_1's auc: 0.679969
[290]	training's auc: 0.827935	valid_1's auc: 0.680204
[300]	training's auc: 0.82845	valid_1's auc: 0.680393
[310]	training's auc: 0.828973	valid_1's auc: 0.680602
[320]	training's auc: 0.829492	valid_1's auc: 0.680817
[330]	training's auc: 0.829917	valid_1's auc: 0.680983
[340]	training's auc: 0.830403	valid_1's auc: 0.681178
[350]	training's auc: 0.830832	valid_1's auc: 0.681331
[360]	training's auc: 0.831243	valid_1's auc: 0.681475
[370]	training's auc: 0.83162	valid_1's auc: 0.681614
[380]	training's auc: 0.832074	valid_1's auc: 0.68176
[390]	training's auc: 0.832576	valid_1's auc: 0.681934
[400]	training's auc: 0.833042	valid_1's auc: 0.682075
[410]	training's auc: 0.833422	valid_1's auc: 0.682196
[420]	training's auc: 0.83382	valid_1's auc: 0.682334
[430]	training's auc: 0.834233	valid_1's auc: 0.682486
[440]	training's auc: 0.834599	valid_1's auc: 0.682591
[450]	training's auc: 0.835013	valid_1's auc: 0.682727
[460]	training's auc: 0.835382	valid_1's auc: 0.682826
[470]	training's auc: 0.835775	valid_1's auc: 0.682923
[480]	training's auc: 0.836168	valid_1's auc: 0.683045
[490]	training's auc: 0.836506	valid_1's auc: 0.683131
[500]	training's auc: 0.836895	valid_1's auc: 0.683234
[510]	training's auc: 0.837283	valid_1's auc: 0.683343
[520]	training's auc: 0.837624	valid_1's auc: 0.68343
[530]	training's auc: 0.837985	valid_1's auc: 0.683512
[540]	training's auc: 0.838342	valid_1's auc: 0.683607
[550]	training's auc: 0.838685	valid_1's auc: 0.683688
[560]	training's auc: 0.838961	valid_1's auc: 0.683762
[570]	training's auc: 0.839289	valid_1's auc: 0.683836
[580]	training's auc: 0.839601	valid_1's auc: 0.683908
[590]	training's auc: 0.839934	valid_1's auc: 0.683976
[600]	training's auc: 0.840239	valid_1's auc: 0.684055
[610]	training's auc: 0.840568	valid_1's auc: 0.684172
[620]	training's auc: 0.840877	valid_1's auc: 0.68424
[630]	training's auc: 0.841193	valid_1's auc: 0.684308
[640]	training's auc: 0.841555	valid_1's auc: 0.684377
[650]	training's auc: 0.841889	valid_1's auc: 0.684453
[660]	training's auc: 0.842187	valid_1's auc: 0.6845
[670]	training's auc: 0.842469	valid_1's auc: 0.684573
[680]	training's auc: 0.842784	valid_1's auc: 0.684663
[690]	training's auc: 0.84308	valid_1's auc: 0.684725
[700]	training's auc: 0.843386	valid_1's auc: 0.684787
[710]	training's auc: 0.843679	valid_1's auc: 0.684853
[720]	training's auc: 0.843888	valid_1's auc: 0.684885
[730]	training's auc: 0.84414	valid_1's auc: 0.684941
[740]	training's auc: 0.844399	valid_1's auc: 0.684972
[750]	training's auc: 0.844668	valid_1's auc: 0.68503
[760]	training's auc: 0.844877	valid_1's auc: 0.685081
[770]	training's auc: 0.845162	valid_1's auc: 0.685129
[780]	training's auc: 0.845463	valid_1's auc: 0.685185
[790]	training's auc: 0.845744	valid_1's auc: 0.685234
[800]	training's auc: 0.846	valid_1's auc: 0.68526
complete on: ITC_top1_in_song_log10_1
model:
best score: 0.68525979095
best iteration: 0

                msno : 55375
             song_id : 20199
   source_system_tab : 5352
  source_screen_name : 20119
         source_type : 14858
         artist_name : 65662
           song_year : 24344
 ITC_song_id_log10_1 : 74397
    ITC_msno_log10_1 : 80423
        top2_in_song : 14848
ITC_top1_in_song_log10_1 : 32423
working on: ITC_top2_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_system_tab           category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float16
ITC_msno_log10_1             float16
top2_in_song                category
ITC_top2_in_song_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_top2_in_song_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.7771	valid_1's auc: 0.659391
[20]	training's auc: 0.783935	valid_1's auc: 0.662169
[30]	training's auc: 0.787862	valid_1's auc: 0.663761
[40]	training's auc: 0.789986	valid_1's auc: 0.664712
[50]	training's auc: 0.793745	valid_1's auc: 0.665925
[60]	training's auc: 0.797811	valid_1's auc: 0.667125
[70]	training's auc: 0.800596	valid_1's auc: 0.668148
[80]	training's auc: 0.803575	valid_1's auc: 0.669303
[90]	training's auc: 0.806091	valid_1's auc: 0.67018
[100]	training's auc: 0.808133	valid_1's auc: 0.671028
[110]	training's auc: 0.810484	valid_1's auc: 0.671901
[120]	training's auc: 0.81268	valid_1's auc: 0.672798
[130]	training's auc: 0.814576	valid_1's auc: 0.673541
[140]	training's auc: 0.81637	valid_1's auc: 0.674282
[150]	training's auc: 0.817902	valid_1's auc: 0.674902
[160]	training's auc: 0.819274	valid_1's auc: 0.675507
[170]	training's auc: 0.820603	valid_1's auc: 0.676113
[180]	training's auc: 0.821423	valid_1's auc: 0.676538
[190]	training's auc: 0.822188	valid_1's auc: 0.676891
[200]	training's auc: 0.822709	valid_1's auc: 0.677202
[210]	training's auc: 0.823356	valid_1's auc: 0.677556
[220]	training's auc: 0.824054	valid_1's auc: 0.677858
[230]	training's auc: 0.824618	valid_1's auc: 0.678099
[240]	training's auc: 0.825159	valid_1's auc: 0.678367
[250]	training's auc: 0.825699	valid_1's auc: 0.678624
[260]	training's auc: 0.826276	valid_1's auc: 0.678902
[270]	training's auc: 0.826821	valid_1's auc: 0.679094
[280]	training's auc: 0.827353	valid_1's auc: 0.679316
[290]	training's auc: 0.827919	valid_1's auc: 0.679548
[300]	training's auc: 0.828452	valid_1's auc: 0.679753
[310]	training's auc: 0.82899	valid_1's auc: 0.679969
[320]	training's auc: 0.829475	valid_1's auc: 0.680141
[330]	training's auc: 0.829976	valid_1's auc: 0.680331
[340]	training's auc: 0.83045	valid_1's auc: 0.680492
[350]	training's auc: 0.830857	valid_1's auc: 0.680615
[360]	training's auc: 0.831332	valid_1's auc: 0.680793
[370]	training's auc: 0.831715	valid_1's auc: 0.68093
[380]	training's auc: 0.832221	valid_1's auc: 0.681109
[390]	training's auc: 0.832652	valid_1's auc: 0.681234
[400]	training's auc: 0.833109	valid_1's auc: 0.681386
[410]	training's auc: 0.833496	valid_1's auc: 0.681521
[420]	training's auc: 0.833868	valid_1's auc: 0.681624
[430]	training's auc: 0.834282	valid_1's auc: 0.681768
[440]	training's auc: 0.834699	valid_1's auc: 0.6819
[450]	training's auc: 0.835117	valid_1's auc: 0.682028
[460]	training's auc: 0.835491	valid_1's auc: 0.68212
[470]	training's auc: 0.835871	valid_1's auc: 0.682214
[480]	training's auc: 0.836306	valid_1's auc: 0.682342
[490]	training's auc: 0.836641	valid_1's auc: 0.682429
[500]	training's auc: 0.837007	valid_1's auc: 0.682529
[510]	training's auc: 0.837387	valid_1's auc: 0.682591
[520]	training's auc: 0.837756	valid_1's auc: 0.682697
[530]	training's auc: 0.838076	valid_1's auc: 0.682782
[540]	training's auc: 0.838438	valid_1's auc: 0.682889
[550]	training's auc: 0.838769	valid_1's auc: 0.682988
[560]	training's auc: 0.839071	valid_1's auc: 0.683047
[570]	training's auc: 0.8394	valid_1's auc: 0.683124
[580]	training's auc: 0.839743	valid_1's auc: 0.683216
[590]	training's auc: 0.840115	valid_1's auc: 0.683304
[600]	training's auc: 0.840387	valid_1's auc: 0.683367
[610]	training's auc: 0.840669	valid_1's auc: 0.683441
[620]	training's auc: 0.841012	valid_1's auc: 0.683537
[630]	training's auc: 0.841343	valid_1's auc: 0.683616
[640]	training's auc: 0.841699	valid_1's auc: 0.683702
[650]	training's auc: 0.842046	valid_1's auc: 0.683791
[660]	training's auc: 0.842368	valid_1's auc: 0.683885
[670]	training's auc: 0.842646	valid_1's auc: 0.683934
[680]	training's auc: 0.842931	valid_1's auc: 0.683988
[690]	training's auc: 0.843239	valid_1's auc: 0.684065
[700]	training's auc: 0.843533	valid_1's auc: 0.684118
[710]	training's auc: 0.843833	valid_1's auc: 0.684188
[720]	training's auc: 0.844048	valid_1's auc: 0.684225
[730]	training's auc: 0.844292	valid_1's auc: 0.684261
[740]	training's auc: 0.844544	valid_1's auc: 0.684305
[750]	training's auc: 0.844838	valid_1's auc: 0.684374
[760]	training's auc: 0.845045	valid_1's auc: 0.684399
[770]	training's auc: 0.845321	valid_1's auc: 0.684433
[780]	training's auc: 0.845618	valid_1's auc: 0.684479
[790]	training's auc: 0.845882	valid_1's auc: 0.684501
[800]	training's auc: 0.846135	valid_1's auc: 0.68454
complete on: ITC_top2_in_song_log10_1
model:
best score: 0.684539677674
best iteration: 0

                msno : 55379
             song_id : 20268
   source_system_tab : 5281
  source_screen_name : 19911
         source_type : 14832
         artist_name : 66233
           song_year : 24401
 ITC_song_id_log10_1 : 73742
    ITC_msno_log10_1 : 79357
        top2_in_song : 14613
ITC_top2_in_song_log10_1 : 33983
working on: ITC_top3_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_system_tab           category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float16
ITC_msno_log10_1             float16
top2_in_song                category
ITC_top3_in_song_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_top3_in_song_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.778823	valid_1's auc: 0.660416
[20]	training's auc: 0.784805	valid_1's auc: 0.662622
[30]	training's auc: 0.788453	valid_1's auc: 0.664165
[40]	training's auc: 0.79047	valid_1's auc: 0.665062
[50]	training's auc: 0.794032	valid_1's auc: 0.666357
[60]	training's auc: 0.797992	valid_1's auc: 0.667536
[70]	training's auc: 0.800867	valid_1's auc: 0.668542
[80]	training's auc: 0.804173	valid_1's auc: 0.669879
[90]	training's auc: 0.806768	valid_1's auc: 0.670848
[100]	training's auc: 0.808856	valid_1's auc: 0.671672
[110]	training's auc: 0.810805	valid_1's auc: 0.672376
[120]	training's auc: 0.812985	valid_1's auc: 0.673185
[130]	training's auc: 0.814791	valid_1's auc: 0.673996
[140]	training's auc: 0.816555	valid_1's auc: 0.674778
[150]	training's auc: 0.818055	valid_1's auc: 0.675439
[160]	training's auc: 0.819478	valid_1's auc: 0.676114
[170]	training's auc: 0.820731	valid_1's auc: 0.676709
[180]	training's auc: 0.821616	valid_1's auc: 0.677158
[190]	training's auc: 0.822375	valid_1's auc: 0.677536
[200]	training's auc: 0.82298	valid_1's auc: 0.677916
[210]	training's auc: 0.823738	valid_1's auc: 0.678328
[220]	training's auc: 0.824379	valid_1's auc: 0.678619
[230]	training's auc: 0.824983	valid_1's auc: 0.678889
[240]	training's auc: 0.825555	valid_1's auc: 0.679179
[250]	training's auc: 0.826081	valid_1's auc: 0.679439
[260]	training's auc: 0.826601	valid_1's auc: 0.679704
[270]	training's auc: 0.827278	valid_1's auc: 0.679971
[280]	training's auc: 0.827807	valid_1's auc: 0.680198
[290]	training's auc: 0.828329	valid_1's auc: 0.680429
[300]	training's auc: 0.828843	valid_1's auc: 0.680612
[310]	training's auc: 0.829378	valid_1's auc: 0.680852
[320]	training's auc: 0.829853	valid_1's auc: 0.681036
[330]	training's auc: 0.830319	valid_1's auc: 0.681222
[340]	training's auc: 0.830817	valid_1's auc: 0.681415
[350]	training's auc: 0.831256	valid_1's auc: 0.68156
[360]	training's auc: 0.831728	valid_1's auc: 0.68174
[370]	training's auc: 0.832108	valid_1's auc: 0.681863
[380]	training's auc: 0.832547	valid_1's auc: 0.682018
[390]	training's auc: 0.832982	valid_1's auc: 0.68217
[400]	training's auc: 0.833385	valid_1's auc: 0.68224
[410]	training's auc: 0.833799	valid_1's auc: 0.682412
[420]	training's auc: 0.834238	valid_1's auc: 0.682549
[430]	training's auc: 0.834637	valid_1's auc: 0.682676
[440]	training's auc: 0.835044	valid_1's auc: 0.682829
[450]	training's auc: 0.835442	valid_1's auc: 0.682965
[460]	training's auc: 0.835805	valid_1's auc: 0.683064
[470]	training's auc: 0.836157	valid_1's auc: 0.68316
[480]	training's auc: 0.836566	valid_1's auc: 0.683278
[490]	training's auc: 0.836905	valid_1's auc: 0.683372
[500]	training's auc: 0.837271	valid_1's auc: 0.683474
[510]	training's auc: 0.837628	valid_1's auc: 0.683545
[520]	training's auc: 0.838007	valid_1's auc: 0.683648
[530]	training's auc: 0.838351	valid_1's auc: 0.683742
[540]	training's auc: 0.838685	valid_1's auc: 0.683842
[550]	training's auc: 0.839021	valid_1's auc: 0.683938
[560]	training's auc: 0.839312	valid_1's auc: 0.684016
[570]	training's auc: 0.83965	valid_1's auc: 0.684109
[580]	training's auc: 0.840017	valid_1's auc: 0.684215
[590]	training's auc: 0.840376	valid_1's auc: 0.68429
[600]	training's auc: 0.840653	valid_1's auc: 0.684342
[610]	training's auc: 0.840948	valid_1's auc: 0.684425
[620]	training's auc: 0.841228	valid_1's auc: 0.684479
[630]	training's auc: 0.841541	valid_1's auc: 0.684571
[640]	training's auc: 0.841914	valid_1's auc: 0.684653
[650]	training's auc: 0.842233	valid_1's auc: 0.684711
[660]	training's auc: 0.842525	valid_1's auc: 0.684767
[670]	training's auc: 0.842808	valid_1's auc: 0.684841
[680]	training's auc: 0.843084	valid_1's auc: 0.684877
[690]	training's auc: 0.843386	valid_1's auc: 0.684936
[700]	training's auc: 0.843698	valid_1's auc: 0.685
[710]	training's auc: 0.844003	valid_1's auc: 0.685058
[720]	training's auc: 0.844251	valid_1's auc: 0.685081
[730]	training's auc: 0.844475	valid_1's auc: 0.68513
[740]	training's auc: 0.844746	valid_1's auc: 0.685179
[750]	training's auc: 0.844988	valid_1's auc: 0.685225
[760]	training's auc: 0.84521	valid_1's auc: 0.685267
[770]	training's auc: 0.845495	valid_1's auc: 0.685322
[780]	training's auc: 0.845789	valid_1's auc: 0.685362
[790]	training's auc: 0.846056	valid_1's auc: 0.685403
[800]	training's auc: 0.846318	valid_1's auc: 0.685464
complete on: ITC_top3_in_song_log10_1
model:
best score: 0.685463888941
best iteration: 0

                msno : 55617
             song_id : 20601
   source_system_tab : 5199
  source_screen_name : 19763
         source_type : 14578
         artist_name : 66120
           song_year : 24182
 ITC_song_id_log10_1 : 73964
    ITC_msno_log10_1 : 79585
        top2_in_song : 14750
ITC_top3_in_song_log10_1 : 33641
working on: ITC_composer_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float16
ITC_msno_log10_1         float16
top2_in_song            category
ITC_composer_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_composer_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.779112	valid_1's auc: 0.659517
[20]	training's auc: 0.784845	valid_1's auc: 0.661947
[30]	training's auc: 0.788673	valid_1's auc: 0.663497
[40]	training's auc: 0.790683	valid_1's auc: 0.664519
[50]	training's auc: 0.794285	valid_1's auc: 0.665707
[60]	training's auc: 0.798471	valid_1's auc: 0.66691
[70]	training's auc: 0.801037	valid_1's auc: 0.667763
[80]	training's auc: 0.804411	valid_1's auc: 0.669068
[90]	training's auc: 0.806707	valid_1's auc: 0.669919
[100]	training's auc: 0.808852	valid_1's auc: 0.670774
[110]	training's auc: 0.810985	valid_1's auc: 0.671602
[120]	training's auc: 0.813152	valid_1's auc: 0.67244
[130]	training's auc: 0.814948	valid_1's auc: 0.673177
[140]	training's auc: 0.816603	valid_1's auc: 0.673863
[150]	training's auc: 0.8182	valid_1's auc: 0.67453
[160]	training's auc: 0.819692	valid_1's auc: 0.675198
[170]	training's auc: 0.820879	valid_1's auc: 0.675739
[180]	training's auc: 0.82175	valid_1's auc: 0.676157
[190]	training's auc: 0.822473	valid_1's auc: 0.676495
[200]	training's auc: 0.823153	valid_1's auc: 0.676874
[210]	training's auc: 0.82368	valid_1's auc: 0.677149
[220]	training's auc: 0.824297	valid_1's auc: 0.677409
[230]	training's auc: 0.824994	valid_1's auc: 0.677722
[240]	training's auc: 0.825564	valid_1's auc: 0.677999
[250]	training's auc: 0.826134	valid_1's auc: 0.678285
[260]	training's auc: 0.826689	valid_1's auc: 0.67854
[270]	training's auc: 0.827303	valid_1's auc: 0.678776
[280]	training's auc: 0.827846	valid_1's auc: 0.678993
[290]	training's auc: 0.828381	valid_1's auc: 0.679184
[300]	training's auc: 0.828947	valid_1's auc: 0.679392
[310]	training's auc: 0.829455	valid_1's auc: 0.679582
[320]	training's auc: 0.829943	valid_1's auc: 0.679766
[330]	training's auc: 0.830457	valid_1's auc: 0.67998
[340]	training's auc: 0.830947	valid_1's auc: 0.680138
[350]	training's auc: 0.831415	valid_1's auc: 0.680318
[360]	training's auc: 0.831831	valid_1's auc: 0.680593
[370]	training's auc: 0.83223	valid_1's auc: 0.680743
[380]	training's auc: 0.832661	valid_1's auc: 0.680877
[390]	training's auc: 0.83308	valid_1's auc: 0.681
[400]	training's auc: 0.833576	valid_1's auc: 0.681146
[410]	training's auc: 0.833996	valid_1's auc: 0.681277
[420]	training's auc: 0.834374	valid_1's auc: 0.681393
[430]	training's auc: 0.834785	valid_1's auc: 0.681531
[440]	training's auc: 0.83517	valid_1's auc: 0.681658
[450]	training's auc: 0.835554	valid_1's auc: 0.681752
[460]	training's auc: 0.835945	valid_1's auc: 0.681904
[470]	training's auc: 0.836342	valid_1's auc: 0.68202
[480]	training's auc: 0.836718	valid_1's auc: 0.682112
[490]	training's auc: 0.837062	valid_1's auc: 0.682236
[500]	training's auc: 0.837423	valid_1's auc: 0.682326
[510]	training's auc: 0.837785	valid_1's auc: 0.682391
[520]	training's auc: 0.838169	valid_1's auc: 0.682502
[530]	training's auc: 0.838462	valid_1's auc: 0.682554
[540]	training's auc: 0.838841	valid_1's auc: 0.682647
[550]	training's auc: 0.839152	valid_1's auc: 0.682717
[560]	training's auc: 0.83945	valid_1's auc: 0.682797
[570]	training's auc: 0.83979	valid_1's auc: 0.682876
[580]	training's auc: 0.840158	valid_1's auc: 0.682963
[590]	training's auc: 0.840497	valid_1's auc: 0.683043
[600]	training's auc: 0.840785	valid_1's auc: 0.683103
[610]	training's auc: 0.841092	valid_1's auc: 0.683182
[620]	training's auc: 0.841381	valid_1's auc: 0.683233
[630]	training's auc: 0.841709	valid_1's auc: 0.683304
[640]	training's auc: 0.842051	valid_1's auc: 0.683379
[650]	training's auc: 0.842393	valid_1's auc: 0.683451
[660]	training's auc: 0.842674	valid_1's auc: 0.683505
[670]	training's auc: 0.84296	valid_1's auc: 0.683571
[680]	training's auc: 0.843252	valid_1's auc: 0.683635
[690]	training's auc: 0.843541	valid_1's auc: 0.683674
[700]	training's auc: 0.843834	valid_1's auc: 0.683721
[710]	training's auc: 0.844118	valid_1's auc: 0.68376
[720]	training's auc: 0.844335	valid_1's auc: 0.683791
[730]	training's auc: 0.844573	valid_1's auc: 0.68383
[740]	training's auc: 0.844836	valid_1's auc: 0.683868
[750]	training's auc: 0.845121	valid_1's auc: 0.683956
[760]	training's auc: 0.845323	valid_1's auc: 0.683992
[770]	training's auc: 0.84563	valid_1's auc: 0.684049
[780]	training's auc: 0.845914	valid_1's auc: 0.684094
[790]	training's auc: 0.846193	valid_1's auc: 0.684152
[800]	training's auc: 0.846507	valid_1's auc: 0.684214
complete on: ITC_composer_log10_1
model:
best score: 0.68421373503
best iteration: 0

                msno : 55998
             song_id : 20460
   source_system_tab : 5093
  source_screen_name : 19351
         source_type : 14051
         artist_name : 64396
           song_year : 23191
 ITC_song_id_log10_1 : 68215
    ITC_msno_log10_1 : 75418
        top2_in_song : 15579
ITC_composer_log10_1 : 46248
working on: ITC_lyricist_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_system_tab       category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float16
ITC_msno_log10_1         float16
top2_in_song            category
ITC_lyricist_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_lyricist_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.777162	valid_1's auc: 0.659168
[20]	training's auc: 0.784241	valid_1's auc: 0.662058
[30]	training's auc: 0.787793	valid_1's auc: 0.663457
[40]	training's auc: 0.78981	valid_1's auc: 0.664423
[50]	training's auc: 0.793942	valid_1's auc: 0.665791
[60]	training's auc: 0.797902	valid_1's auc: 0.667006
[70]	training's auc: 0.800642	valid_1's auc: 0.668044
[80]	training's auc: 0.80402	valid_1's auc: 0.669224
[90]	training's auc: 0.80657	valid_1's auc: 0.67014
[100]	training's auc: 0.80867	valid_1's auc: 0.670968
[110]	training's auc: 0.810845	valid_1's auc: 0.671764
[120]	training's auc: 0.812852	valid_1's auc: 0.672518
[130]	training's auc: 0.814711	valid_1's auc: 0.673282
[140]	training's auc: 0.816384	valid_1's auc: 0.673991
[150]	training's auc: 0.818021	valid_1's auc: 0.674681
[160]	training's auc: 0.819488	valid_1's auc: 0.675362
[170]	training's auc: 0.820818	valid_1's auc: 0.675975
[180]	training's auc: 0.821611	valid_1's auc: 0.67639
[190]	training's auc: 0.822423	valid_1's auc: 0.676738
[200]	training's auc: 0.82316	valid_1's auc: 0.677174
[210]	training's auc: 0.823796	valid_1's auc: 0.677516
[220]	training's auc: 0.824442	valid_1's auc: 0.677832
[230]	training's auc: 0.825042	valid_1's auc: 0.6781
[240]	training's auc: 0.825625	valid_1's auc: 0.678383
[250]	training's auc: 0.826177	valid_1's auc: 0.67866
[260]	training's auc: 0.826697	valid_1's auc: 0.678903
[270]	training's auc: 0.827282	valid_1's auc: 0.679138
[280]	training's auc: 0.82778	valid_1's auc: 0.679329
[290]	training's auc: 0.828342	valid_1's auc: 0.679566
[300]	training's auc: 0.828914	valid_1's auc: 0.679787
[310]	training's auc: 0.829422	valid_1's auc: 0.679979
[320]	training's auc: 0.829903	valid_1's auc: 0.680179
[330]	training's auc: 0.830352	valid_1's auc: 0.680355
[340]	training's auc: 0.830841	valid_1's auc: 0.680524
[350]	training's auc: 0.83134	valid_1's auc: 0.680719
[360]	training's auc: 0.831764	valid_1's auc: 0.680836
[370]	training's auc: 0.832161	valid_1's auc: 0.680975
[380]	training's auc: 0.832641	valid_1's auc: 0.681127
[390]	training's auc: 0.833045	valid_1's auc: 0.681259
[400]	training's auc: 0.83349	valid_1's auc: 0.681376
[410]	training's auc: 0.833877	valid_1's auc: 0.681497
[420]	training's auc: 0.834264	valid_1's auc: 0.68165
[430]	training's auc: 0.834638	valid_1's auc: 0.681775
[440]	training's auc: 0.835041	valid_1's auc: 0.681918
[450]	training's auc: 0.83543	valid_1's auc: 0.682041
[460]	training's auc: 0.83582	valid_1's auc: 0.682141
[470]	training's auc: 0.836232	valid_1's auc: 0.682256
[480]	training's auc: 0.836648	valid_1's auc: 0.68237
[490]	training's auc: 0.836986	valid_1's auc: 0.682478
[500]	training's auc: 0.837367	valid_1's auc: 0.682576
[510]	training's auc: 0.837774	valid_1's auc: 0.682689
[520]	training's auc: 0.838119	valid_1's auc: 0.682768
[530]	training's auc: 0.838424	valid_1's auc: 0.682831
[540]	training's auc: 0.838773	valid_1's auc: 0.682959
[550]	training's auc: 0.839109	valid_1's auc: 0.68307
[560]	training's auc: 0.839365	valid_1's auc: 0.683104
[570]	training's auc: 0.839708	valid_1's auc: 0.683195
[580]	training's auc: 0.840051	valid_1's auc: 0.683276
[590]	training's auc: 0.840427	valid_1's auc: 0.683367
[600]	training's auc: 0.840691	valid_1's auc: 0.683422
[610]	training's auc: 0.840991	valid_1's auc: 0.683511
[620]	training's auc: 0.841327	valid_1's auc: 0.68358
[630]	training's auc: 0.841655	valid_1's auc: 0.683659
[640]	training's auc: 0.842008	valid_1's auc: 0.683728
[650]	training's auc: 0.842309	valid_1's auc: 0.683789
[660]	training's auc: 0.842605	valid_1's auc: 0.683852
[670]	training's auc: 0.842927	valid_1's auc: 0.683944
[680]	training's auc: 0.843208	valid_1's auc: 0.684009
[690]	training's auc: 0.843491	valid_1's auc: 0.684043
[700]	training's auc: 0.84379	valid_1's auc: 0.684103
[710]	training's auc: 0.844087	valid_1's auc: 0.684164
[720]	training's auc: 0.844329	valid_1's auc: 0.684225
[730]	training's auc: 0.844566	valid_1's auc: 0.684278
[740]	training's auc: 0.844836	valid_1's auc: 0.684338
[750]	training's auc: 0.845083	valid_1's auc: 0.684392
[760]	training's auc: 0.845304	valid_1's auc: 0.684483
[770]	training's auc: 0.845616	valid_1's auc: 0.684561
[780]	training's auc: 0.84591	valid_1's auc: 0.68461
[790]	training's auc: 0.846178	valid_1's auc: 0.684662
[800]	training's auc: 0.846453	valid_1's auc: 0.684725
complete on: ITC_lyricist_log10_1
model:
best score: 0.684724640425
best iteration: 0

                msno : 55970
             song_id : 20950
   source_system_tab : 5009
  source_screen_name : 19535
         source_type : 14288
         artist_name : 65203
           song_year : 23854
 ITC_song_id_log10_1 : 70611
    ITC_msno_log10_1 : 77046
        top2_in_song : 15988
ITC_lyricist_log10_1 : 39546
working on: ITC_artist_name_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                        uint8
msno                       category
song_id                    category
source_system_tab          category
source_screen_name         category
source_type                category
artist_name                category
song_year                  category
ITC_song_id_log10_1         float16
ITC_msno_log10_1            float16
top2_in_song               category
ITC_artist_name_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 12

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
'top2_in_song',
'ITC_artist_name_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.77936	valid_1's auc: 0.659717
[20]	training's auc: 0.784602	valid_1's auc: 0.661841
[30]	training's auc: 0.78879	valid_1's auc: 0.663473
[40]	training's auc: 0.790568	valid_1's auc: 0.664393
[50]	training's auc: 0.794157	valid_1's auc: 0.665593
[60]	training's auc: 0.798161	valid_1's auc: 0.666823
[70]	training's auc: 0.800921	valid_1's auc: 0.667811
[80]	training's auc: 0.804053	valid_1's auc: 0.668925
[90]	training's auc: 0.806544	valid_1's auc: 0.669841
[100]	training's auc: 0.808823	valid_1's auc: 0.670834
[110]	training's auc: 0.81116	valid_1's auc: 0.671712
[120]	training's auc: 0.81342	valid_1's auc: 0.672611
[130]	training's auc: 0.815246	valid_1's auc: 0.673414
[140]	training's auc: 0.816765	valid_1's auc: 0.674108
[150]	training's auc: 0.818341	valid_1's auc: 0.674793
[160]	training's auc: 0.819791	valid_1's auc: 0.675488
[170]	training's auc: 0.82104	valid_1's auc: 0.676056
[180]	training's auc: 0.821869	valid_1's auc: 0.676468
[190]	training's auc: 0.82264	valid_1's auc: 0.676803
[200]	training's auc: 0.823234	valid_1's auc: 0.677155
[210]	training's auc: 0.823849	valid_1's auc: 0.677503
[220]	training's auc: 0.824563	valid_1's auc: 0.677785
[230]	training's auc: 0.825217	valid_1's auc: 0.6781
[240]	training's auc: 0.825765	valid_1's auc: 0.678345
[250]	training's auc: 0.826342	valid_1's auc: 0.678638
[260]	training's auc: 0.826836	valid_1's auc: 0.678858
[270]	training's auc: 0.827524	valid_1's auc: 0.679136
[280]	training's auc: 0.828039	valid_1's auc: 0.679345
[290]	training's auc: 0.828561	valid_1's auc: 0.679536
[300]	training's auc: 0.829084	valid_1's auc: 0.679724
[310]	training's auc: 0.829637	valid_1's auc: 0.679951
[320]	training's auc: 0.830141	valid_1's auc: 0.68015
[330]	training's auc: 0.830593	valid_1's auc: 0.680299
[340]	training's auc: 0.831109	valid_1's auc: 0.680499
[350]	training's auc: 0.831612	valid_1's auc: 0.680701
[360]	training's auc: 0.832065	valid_1's auc: 0.680888
[370]	training's auc: 0.83251	valid_1's auc: 0.681028
[380]	training's auc: 0.832934	valid_1's auc: 0.681163
[390]	training's auc: 0.833388	valid_1's auc: 0.681313
[400]	training's auc: 0.833867	valid_1's auc: 0.681447
[410]	training's auc: 0.834274	valid_1's auc: 0.681601
[420]	training's auc: 0.834683	valid_1's auc: 0.681696
[430]	training's auc: 0.835142	valid_1's auc: 0.681837
[440]	training's auc: 0.835531	valid_1's auc: 0.681962
[450]	training's auc: 0.835924	valid_1's auc: 0.682098
[460]	training's auc: 0.836332	valid_1's auc: 0.682207
[470]	training's auc: 0.836698	valid_1's auc: 0.682284
[480]	training's auc: 0.837089	valid_1's auc: 0.682391
[490]	training's auc: 0.837448	valid_1's auc: 0.682489
[500]	training's auc: 0.837825	valid_1's auc: 0.682588
[510]	training's auc: 0.838207	valid_1's auc: 0.682688
[520]	training's auc: 0.838559	valid_1's auc: 0.682787
[530]	training's auc: 0.838935	valid_1's auc: 0.682886
[540]	training's auc: 0.839275	valid_1's auc: 0.682984
[550]	training's auc: 0.839617	valid_1's auc: 0.683076
[560]	training's auc: 0.839901	valid_1's auc: 0.683126
[570]	training's auc: 0.840238	valid_1's auc: 0.683203
[580]	training's auc: 0.840597	valid_1's auc: 0.68332
[590]	training's auc: 0.840939	valid_1's auc: 0.683407
[600]	training's auc: 0.841276	valid_1's auc: 0.683495
[610]	training's auc: 0.841579	valid_1's auc: 0.683569
[620]	training's auc: 0.841875	valid_1's auc: 0.6836
[630]	training's auc: 0.842232	valid_1's auc: 0.683693
[640]	training's auc: 0.842572	valid_1's auc: 0.683754
[650]	training's auc: 0.84291	valid_1's auc: 0.683823
[660]	training's auc: 0.843211	valid_1's auc: 0.683888
[670]	training's auc: 0.843475	valid_1's auc: 0.683942
[680]	training's auc: 0.843808	valid_1's auc: 0.684003
[690]	training's auc: 0.844134	valid_1's auc: 0.684081
[700]	training's auc: 0.844441	valid_1's auc: 0.684145
[710]	training's auc: 0.844756	valid_1's auc: 0.684203
[720]	training's auc: 0.844986	valid_1's auc: 0.684246
[730]	training's auc: 0.845245	valid_1's auc: 0.684324
[740]	training's auc: 0.845502	valid_1's auc: 0.68435
[750]	training's auc: 0.845766	valid_1's auc: 0.684382
[760]	training's auc: 0.846021	valid_1's auc: 0.684414
[770]	training's auc: 0.846312	valid_1's auc: 0.684482
[780]	training's auc: 0.846623	valid_1's auc: 0.684539
[790]	training's auc: 0.846888	valid_1's auc: 0.684604
[800]	training's auc: 0.847162	valid_1's auc: 0.684658
complete on: ITC_artist_name_log10_1
model:
best score: 0.684657757613
best iteration: 0

                msno : 56276
             song_id : 22401
   source_system_tab : 4830
  source_screen_name : 18845
         source_type : 13931
         artist_name : 64466
           song_year : 23430
 ITC_song_id_log10_1 : 61703
    ITC_msno_log10_1 : 68981
        top2_in_song : 15363
ITC_artist_name_log10_1 : 57774
                 ITC_composer_log10_1:  0.68421373503
             ITC_top2_in_song_log10_1:  0.684539677674
              ITC_artist_name_log10_1:  0.684657757613
                 ITC_lyricist_log10_1:  0.684724640425
                 ITC_language_log10_1:  0.684762665487
              ITC_source_type_log10_1:  0.684832560552
       ITC_source_screen_name_log10_1:  0.685091007661
             ITC_top1_in_song_log10_1:  0.68525979095
                        song_year_int:  0.685324837361
                ITC_song_year_log10_1:  0.685369186765
             ITC_top3_in_song_log10_1:  0.685463888941

[timer]: complete in 343m 35s

Process finished with exit code 0
'''