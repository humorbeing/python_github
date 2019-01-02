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
load_name = 'train_me_play.csv'
# df = pd.read_csv('../saves/train_me_play.csv')
# def intme(x):
#     return int(x)
#
df = read_df(load_name)
# df['song_year'] = df['song_year'].astype(object)
# df['song_year_int'] = df['song_year'].apply(intme).astype(np.int64)
# df['song_year'] = df['song_year'].astype('category')
#
# # show_df(df)
# cols = [
#     'msno',
#     'song_id',
#     # 'artist_name',
#     'top1_in_song',
#     # 'top2_in_song',
#     'top3_in_song',
#     # 'language',
#     'song_year',
#     # 'composer',
#     # 'lyricist',
#     'source_screen_name',
#     'source_type',
# ]
# df = add_ITC(df, cols)

show_df(df)


num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.032
num_leaves = 750
max_depth = 50

max_bin = 172
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
    # 'source_system_tab',
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
    # 'ITC_language_log10_1',
    'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    'ITC_top3_in_song_log10_1',
    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_artist_name_log10_1',

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
for w in df.columns:
# for w in work_on:
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


'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION/one_trainer_V1003B.py

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                              category
song_id                           category
source_system_tab                 category
source_screen_name                category
source_type                       category
target                               uint8
genre_ids                         category
artist_name                       category
composer                          category
lyricist                          category
language                          category
song_year                         category
song_country                      category
rc                                category
top1_in_song                      category
top2_in_song                      category
top3_in_song                      category
membership_days                      int64
song_year_int                        int64
ISC_top1_in_song                     int64
ISC_top2_in_song                     int64
ISC_top3_in_song                     int64
ISC_language                         int64
ISCZ_rc                              int64
ISCZ_isrc_rest                       int64
ISC_song_year                        int64
song_length_log10                  float64
ISCZ_genre_ids_log10               float64
ISC_artist_name_log10              float64
ISCZ_composer_log10                float64
ISC_lyricist_log10                 float64
ISC_song_country_ln                float64
ITC_song_id_log10_1                float32
ITC_source_system_tab_log10_1      float32
ITC_source_screen_name_log10_1     float32
ITC_source_type_log10_1            float32
ITC_artist_name_log10_1            float32
ITC_composer_log10_1               float32
ITC_lyricist_log10_1               float32
ITC_song_year_log10_1              float32
ITC_top1_in_song_log10_1           float32
ITC_top2_in_song_log10_1           float32
ITC_top3_in_song_log10_1           float32
ITC_msno_log10_1                   float32
OinC_msno                          float32
ITC_language_log10_1               float32
OinC_language                      float32
dtype: object
number of rows: 7377418
number of columns: 47

'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'genre_ids',
'artist_name',
'composer',
'lyricist',
'language',
'song_year',
'song_country',
'rc',
'top1_in_song',
'top2_in_song',
'top3_in_song',
'membership_days',
'song_year_int',
'ISC_top1_in_song',
'ISC_top2_in_song',
'ISC_top3_in_song',
'ISC_language',
'ISCZ_rc',
'ISCZ_isrc_rest',
'ISC_song_year',
'song_length_log10',
'ISCZ_genre_ids_log10',
'ISC_artist_name_log10',
'ISCZ_composer_log10',
'ISC_lyricist_log10',
'ISC_song_country_ln',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
'ITC_artist_name_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_song_year_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_msno_log10_1',
'OinC_msno',
'ITC_language_log10_1',
'OinC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'genre_ids',
'artist_name',
'composer',
'lyricist',
'language',
'song_year',
'song_country',
'rc',
'top1_in_song',
'top2_in_song',
'top3_in_song',
'membership_days',
'song_year_int',
'ISC_top1_in_song',
'ISC_top2_in_song',
'ISC_top3_in_song',
'ISC_language',
'ISCZ_rc',
'ISCZ_isrc_rest',
'ISC_song_year',
'song_length_log10',
'ISCZ_genre_ids_log10',
'ISC_artist_name_log10',
'ISCZ_composer_log10',
'ISC_lyricist_log10',
'ISC_song_country_ln',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
'ITC_artist_name_log10_1',
'ITC_composer_log10_1',
'ITC_lyricist_log10_1',
'ITC_song_year_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top2_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_msno_log10_1',
'OinC_msno',
'ITC_language_log10_1',
'OinC_language',
working on: source_system_tab

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
source_system_tab      category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'source_system_tab',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795778	valid_1's auc: 0.663952
[20]	training's auc: 0.803129	valid_1's auc: 0.666824
[30]	training's auc: 0.808384	valid_1's auc: 0.66878
[40]	training's auc: 0.813176	valid_1's auc: 0.67051
[50]	training's auc: 0.818406	valid_1's auc: 0.672385
[60]	training's auc: 0.823183	valid_1's auc: 0.673881
[70]	training's auc: 0.827261	valid_1's auc: 0.675314
[80]	training's auc: 0.829751	valid_1's auc: 0.676142
[90]	training's auc: 0.832311	valid_1's auc: 0.677099
[100]	training's auc: 0.834737	valid_1's auc: 0.678074
[110]	training's auc: 0.836487	valid_1's auc: 0.678773
[120]	training's auc: 0.838161	valid_1's auc: 0.67962
[130]	training's auc: 0.839553	valid_1's auc: 0.680252
[140]	training's auc: 0.840733	valid_1's auc: 0.68081
[150]	training's auc: 0.841731	valid_1's auc: 0.681298
[160]	training's auc: 0.842716	valid_1's auc: 0.681714
[170]	training's auc: 0.843746	valid_1's auc: 0.682121
[180]	training's auc: 0.844578	valid_1's auc: 0.682459
[190]	training's auc: 0.845486	valid_1's auc: 0.682806
[200]	training's auc: 0.846396	valid_1's auc: 0.683172
[210]	training's auc: 0.847204	valid_1's auc: 0.683379
[220]	training's auc: 0.847958	valid_1's auc: 0.683624
[230]	training's auc: 0.848648	valid_1's auc: 0.683871
[240]	training's auc: 0.84942	valid_1's auc: 0.684111
[250]	training's auc: 0.850053	valid_1's auc: 0.684287
[260]	training's auc: 0.850729	valid_1's auc: 0.684423
[270]	training's auc: 0.851434	valid_1's auc: 0.68465
[280]	training's auc: 0.851989	valid_1's auc: 0.684762
[290]	training's auc: 0.852598	valid_1's auc: 0.684886
[300]	training's auc: 0.853238	valid_1's auc: 0.68507
[310]	training's auc: 0.85381	valid_1's auc: 0.68518
[320]	training's auc: 0.854369	valid_1's auc: 0.685292
[330]	training's auc: 0.8549	valid_1's auc: 0.685375
[340]	training's auc: 0.855442	valid_1's auc: 0.685436
[350]	training's auc: 0.856005	valid_1's auc: 0.685554
[360]	training's auc: 0.856603	valid_1's auc: 0.68563
[370]	training's auc: 0.857111	valid_1's auc: 0.685752
[380]	training's auc: 0.857644	valid_1's auc: 0.68582
[390]	training's auc: 0.858119	valid_1's auc: 0.685878
[400]	training's auc: 0.858674	valid_1's auc: 0.685991
[410]	training's auc: 0.859167	valid_1's auc: 0.686063
[420]	training's auc: 0.859574	valid_1's auc: 0.686166
[430]	training's auc: 0.860051	valid_1's auc: 0.686228
[440]	training's auc: 0.86049	valid_1's auc: 0.686271
[450]	training's auc: 0.860905	valid_1's auc: 0.686329
[460]	training's auc: 0.861328	valid_1's auc: 0.686417
[470]	training's auc: 0.861785	valid_1's auc: 0.686474
[480]	training's auc: 0.862176	valid_1's auc: 0.686511
[490]	training's auc: 0.862579	valid_1's auc: 0.686573
[500]	training's auc: 0.862996	valid_1's auc: 0.686617
complete on: source_system_tab
model:
best score: 0.686617418696
best iteration: 0

                msno : 66511
             song_id : 18810
  source_screen_name : 16399
         source_type : 10860
         artist_name : 62643
           song_year : 27623
 ITC_song_id_log10_1 : 80540
    ITC_msno_log10_1 : 72444
        top2_in_song : 15565
   source_system_tab : 3105
working on: genre_ids

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
genre_ids              category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'genre_ids',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.79461	valid_1's auc: 0.663483
[20]	training's auc: 0.802366	valid_1's auc: 0.666558
[30]	training's auc: 0.806954	valid_1's auc: 0.66835
[40]	training's auc: 0.811338	valid_1's auc: 0.669948
[50]	training's auc: 0.816639	valid_1's auc: 0.671761
[60]	training's auc: 0.821204	valid_1's auc: 0.673351
[70]	training's auc: 0.824978	valid_1's auc: 0.674659
[80]	training's auc: 0.827337	valid_1's auc: 0.675431
[90]	training's auc: 0.830034	valid_1's auc: 0.676487
[100]	training's auc: 0.832292	valid_1's auc: 0.677436
[110]	training's auc: 0.834345	valid_1's auc: 0.678301
[120]	training's auc: 0.836006	valid_1's auc: 0.679102
[130]	training's auc: 0.837307	valid_1's auc: 0.67976
[140]	training's auc: 0.838458	valid_1's auc: 0.680285
[150]	training's auc: 0.839509	valid_1's auc: 0.680747
[160]	training's auc: 0.840469	valid_1's auc: 0.681137
[170]	training's auc: 0.841589	valid_1's auc: 0.681628
[180]	training's auc: 0.842459	valid_1's auc: 0.681954
[190]	training's auc: 0.843246	valid_1's auc: 0.682224
[200]	training's auc: 0.844051	valid_1's auc: 0.68248
[210]	training's auc: 0.844888	valid_1's auc: 0.682719
[220]	training's auc: 0.845679	valid_1's auc: 0.682958
[230]	training's auc: 0.846423	valid_1's auc: 0.683221
[240]	training's auc: 0.847217	valid_1's auc: 0.683471
[250]	training's auc: 0.847956	valid_1's auc: 0.683673
[260]	training's auc: 0.848734	valid_1's auc: 0.683836
[270]	training's auc: 0.849443	valid_1's auc: 0.684014
[280]	training's auc: 0.850005	valid_1's auc: 0.684146
[290]	training's auc: 0.850639	valid_1's auc: 0.68426
[300]	training's auc: 0.851307	valid_1's auc: 0.684457
[310]	training's auc: 0.851873	valid_1's auc: 0.684557
[320]	training's auc: 0.852448	valid_1's auc: 0.684679
[330]	training's auc: 0.853019	valid_1's auc: 0.68479
[340]	training's auc: 0.853573	valid_1's auc: 0.684888
[350]	training's auc: 0.854146	valid_1's auc: 0.684983
[360]	training's auc: 0.854709	valid_1's auc: 0.685052
[370]	training's auc: 0.855294	valid_1's auc: 0.685128
[380]	training's auc: 0.855828	valid_1's auc: 0.685202
[390]	training's auc: 0.856443	valid_1's auc: 0.685325
[400]	training's auc: 0.85699	valid_1's auc: 0.685408
[410]	training's auc: 0.85752	valid_1's auc: 0.685467
[420]	training's auc: 0.857962	valid_1's auc: 0.685531
[430]	training's auc: 0.858435	valid_1's auc: 0.685541
[440]	training's auc: 0.858948	valid_1's auc: 0.68561
[450]	training's auc: 0.859426	valid_1's auc: 0.685702
[460]	training's auc: 0.85987	valid_1's auc: 0.685734
[470]	training's auc: 0.860341	valid_1's auc: 0.685784
[480]	training's auc: 0.860773	valid_1's auc: 0.685855
[490]	training's auc: 0.861209	valid_1's auc: 0.68592
[500]	training's auc: 0.861651	valid_1's auc: 0.685973
complete on: genre_ids
model:
best score: 0.685972899984
best iteration: 0

                msno : 66269
             song_id : 18409
  source_screen_name : 17006
         source_type : 11442
         artist_name : 62497
           song_year : 27857
 ITC_song_id_log10_1 : 78990
    ITC_msno_log10_1 : 72686
        top2_in_song : 8314
           genre_ids : 11030
working on: composer

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
composer               category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'composer',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794633	valid_1's auc: 0.662518
[20]	training's auc: 0.802537	valid_1's auc: 0.665895
[30]	training's auc: 0.807363	valid_1's auc: 0.667875
[40]	training's auc: 0.811697	valid_1's auc: 0.66936
[50]	training's auc: 0.816853	valid_1's auc: 0.67102
[60]	training's auc: 0.821345	valid_1's auc: 0.672568
[70]	training's auc: 0.825101	valid_1's auc: 0.673749
[80]	training's auc: 0.827505	valid_1's auc: 0.674531
[90]	training's auc: 0.830145	valid_1's auc: 0.675563
[100]	training's auc: 0.832455	valid_1's auc: 0.676592
[110]	training's auc: 0.834773	valid_1's auc: 0.677557
[120]	training's auc: 0.836419	valid_1's auc: 0.678348
[130]	training's auc: 0.837771	valid_1's auc: 0.678951
[140]	training's auc: 0.838879	valid_1's auc: 0.679453
[150]	training's auc: 0.839785	valid_1's auc: 0.679902
[160]	training's auc: 0.84073	valid_1's auc: 0.680308
[170]	training's auc: 0.84168	valid_1's auc: 0.680661
[180]	training's auc: 0.842572	valid_1's auc: 0.681002
[190]	training's auc: 0.843471	valid_1's auc: 0.681342
[200]	training's auc: 0.844249	valid_1's auc: 0.681583
[210]	training's auc: 0.845144	valid_1's auc: 0.681833
[220]	training's auc: 0.845992	valid_1's auc: 0.682107
[230]	training's auc: 0.846682	valid_1's auc: 0.682257
[240]	training's auc: 0.84753	valid_1's auc: 0.68249
[250]	training's auc: 0.848319	valid_1's auc: 0.682719
[260]	training's auc: 0.849015	valid_1's auc: 0.682872
[270]	training's auc: 0.849698	valid_1's auc: 0.683037
[280]	training's auc: 0.850388	valid_1's auc: 0.683191
[290]	training's auc: 0.851017	valid_1's auc: 0.683322
[300]	training's auc: 0.851638	valid_1's auc: 0.683501
[310]	training's auc: 0.852266	valid_1's auc: 0.683619
[320]	training's auc: 0.852853	valid_1's auc: 0.683746
[330]	training's auc: 0.853396	valid_1's auc: 0.683804
[340]	training's auc: 0.853975	valid_1's auc: 0.683907
[350]	training's auc: 0.854583	valid_1's auc: 0.683986
[360]	training's auc: 0.855176	valid_1's auc: 0.684085
[370]	training's auc: 0.855719	valid_1's auc: 0.684148
[380]	training's auc: 0.856232	valid_1's auc: 0.684201
[390]	training's auc: 0.856726	valid_1's auc: 0.684276
[400]	training's auc: 0.857276	valid_1's auc: 0.684377
[410]	training's auc: 0.857731	valid_1's auc: 0.68443
[420]	training's auc: 0.85821	valid_1's auc: 0.684477
[430]	training's auc: 0.858724	valid_1's auc: 0.684535
[440]	training's auc: 0.859194	valid_1's auc: 0.684596
[450]	training's auc: 0.859633	valid_1's auc: 0.684633
[460]	training's auc: 0.860074	valid_1's auc: 0.684684
[470]	training's auc: 0.860515	valid_1's auc: 0.684705
[480]	training's auc: 0.860981	valid_1's auc: 0.684772
[490]	training's auc: 0.861422	valid_1's auc: 0.684807
[500]	training's auc: 0.861867	valid_1's auc: 0.684823
complete on: composer
model:
best score: 0.684823173065
best iteration: 0

                msno : 65501
             song_id : 15101
  source_screen_name : 16114
         source_type : 11052
         artist_name : 58755
           song_year : 26492
 ITC_song_id_log10_1 : 79221
    ITC_msno_log10_1 : 73648
        top2_in_song : 14568
            composer : 14048
working on: lyricist

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
lyricist               category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'lyricist',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794396	valid_1's auc: 0.663182
[20]	training's auc: 0.801986	valid_1's auc: 0.66608
[30]	training's auc: 0.807014	valid_1's auc: 0.668209
[40]	training's auc: 0.811223	valid_1's auc: 0.669661
[50]	training's auc: 0.816498	valid_1's auc: 0.671454
[60]	training's auc: 0.821105	valid_1's auc: 0.673052
[70]	training's auc: 0.824798	valid_1's auc: 0.674225
[80]	training's auc: 0.827117	valid_1's auc: 0.67499
[90]	training's auc: 0.829675	valid_1's auc: 0.675979
[100]	training's auc: 0.832122	valid_1's auc: 0.677063
[110]	training's auc: 0.834253	valid_1's auc: 0.677992
[120]	training's auc: 0.835881	valid_1's auc: 0.678794
[130]	training's auc: 0.837257	valid_1's auc: 0.679456
[140]	training's auc: 0.838474	valid_1's auc: 0.680017
[150]	training's auc: 0.839578	valid_1's auc: 0.680517
[160]	training's auc: 0.840573	valid_1's auc: 0.680915
[170]	training's auc: 0.841514	valid_1's auc: 0.681314
[180]	training's auc: 0.842395	valid_1's auc: 0.681649
[190]	training's auc: 0.843144	valid_1's auc: 0.681913
[200]	training's auc: 0.843952	valid_1's auc: 0.682187
[210]	training's auc: 0.844883	valid_1's auc: 0.682465
[220]	training's auc: 0.845623	valid_1's auc: 0.682692
[230]	training's auc: 0.846321	valid_1's auc: 0.682898
[240]	training's auc: 0.847042	valid_1's auc: 0.68308
[250]	training's auc: 0.84781	valid_1's auc: 0.683313
[260]	training's auc: 0.84849	valid_1's auc: 0.683467
[270]	training's auc: 0.849196	valid_1's auc: 0.683627
[280]	training's auc: 0.849838	valid_1's auc: 0.683782
[290]	training's auc: 0.850472	valid_1's auc: 0.683918
[300]	training's auc: 0.851043	valid_1's auc: 0.684042
[310]	training's auc: 0.85162	valid_1's auc: 0.684148
[320]	training's auc: 0.852237	valid_1's auc: 0.684265
[330]	training's auc: 0.852842	valid_1's auc: 0.684388
[340]	training's auc: 0.853419	valid_1's auc: 0.684481
[350]	training's auc: 0.854008	valid_1's auc: 0.684573
[360]	training's auc: 0.85459	valid_1's auc: 0.684703
[370]	training's auc: 0.855127	valid_1's auc: 0.684794
[380]	training's auc: 0.855699	valid_1's auc: 0.68488
[390]	training's auc: 0.856252	valid_1's auc: 0.684955
[400]	training's auc: 0.856791	valid_1's auc: 0.685015
[410]	training's auc: 0.857293	valid_1's auc: 0.685098
[420]	training's auc: 0.857733	valid_1's auc: 0.685128
[430]	training's auc: 0.85823	valid_1's auc: 0.685209
[440]	training's auc: 0.858687	valid_1's auc: 0.685273
[450]	training's auc: 0.85919	valid_1's auc: 0.685395
[460]	training's auc: 0.859682	valid_1's auc: 0.68545
[470]	training's auc: 0.860122	valid_1's auc: 0.685525
[480]	training's auc: 0.860573	valid_1's auc: 0.685572
[490]	training's auc: 0.861005	valid_1's auc: 0.6856
[500]	training's auc: 0.861437	valid_1's auc: 0.685635
complete on: lyricist
model:
best score: 0.685634683698
best iteration: 0

                msno : 65475
             song_id : 15908
  source_screen_name : 16438
         source_type : 11284
         artist_name : 60196
           song_year : 27253
 ITC_song_id_log10_1 : 78177
    ITC_msno_log10_1 : 73261
        top2_in_song : 15322
            lyricist : 11186
working on: language

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
language               category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794687	valid_1's auc: 0.663382
[20]	training's auc: 0.801988	valid_1's auc: 0.66622
[30]	training's auc: 0.807361	valid_1's auc: 0.668414
[40]	training's auc: 0.811409	valid_1's auc: 0.669866
[50]	training's auc: 0.816627	valid_1's auc: 0.671598
[60]	training's auc: 0.821	valid_1's auc: 0.673106
[70]	training's auc: 0.824801	valid_1's auc: 0.674387
[80]	training's auc: 0.827153	valid_1's auc: 0.675221
[90]	training's auc: 0.829728	valid_1's auc: 0.676147
[100]	training's auc: 0.83236	valid_1's auc: 0.677276
[110]	training's auc: 0.834418	valid_1's auc: 0.67822
[120]	training's auc: 0.836022	valid_1's auc: 0.679052
[130]	training's auc: 0.837574	valid_1's auc: 0.679753
[140]	training's auc: 0.838633	valid_1's auc: 0.680263
[150]	training's auc: 0.839622	valid_1's auc: 0.680738
[160]	training's auc: 0.840642	valid_1's auc: 0.681185
[170]	training's auc: 0.8416	valid_1's auc: 0.681572
[180]	training's auc: 0.842479	valid_1's auc: 0.681896
[190]	training's auc: 0.843343	valid_1's auc: 0.682243
[200]	training's auc: 0.844179	valid_1's auc: 0.682515
[210]	training's auc: 0.845059	valid_1's auc: 0.682829
[220]	training's auc: 0.845834	valid_1's auc: 0.68309
[230]	training's auc: 0.846549	valid_1's auc: 0.683333
[240]	training's auc: 0.847366	valid_1's auc: 0.683606
[250]	training's auc: 0.848109	valid_1's auc: 0.683824
[260]	training's auc: 0.848888	valid_1's auc: 0.684052
[270]	training's auc: 0.849577	valid_1's auc: 0.684229
[280]	training's auc: 0.85022	valid_1's auc: 0.684389
[290]	training's auc: 0.850847	valid_1's auc: 0.684561
[300]	training's auc: 0.851444	valid_1's auc: 0.684718
[310]	training's auc: 0.852019	valid_1's auc: 0.684877
[320]	training's auc: 0.85258	valid_1's auc: 0.684966
[330]	training's auc: 0.853228	valid_1's auc: 0.685111
[340]	training's auc: 0.853823	valid_1's auc: 0.685224
[350]	training's auc: 0.854407	valid_1's auc: 0.685325
[360]	training's auc: 0.854945	valid_1's auc: 0.685432
[370]	training's auc: 0.855464	valid_1's auc: 0.685493
[380]	training's auc: 0.856019	valid_1's auc: 0.685557
[390]	training's auc: 0.856552	valid_1's auc: 0.685606
[400]	training's auc: 0.857108	valid_1's auc: 0.6857
[410]	training's auc: 0.857616	valid_1's auc: 0.685781
[420]	training's auc: 0.858055	valid_1's auc: 0.685838
[430]	training's auc: 0.858542	valid_1's auc: 0.685906
[440]	training's auc: 0.858998	valid_1's auc: 0.685971
[450]	training's auc: 0.85949	valid_1's auc: 0.68603
[460]	training's auc: 0.859919	valid_1's auc: 0.686084
[470]	training's auc: 0.860384	valid_1's auc: 0.686145
[480]	training's auc: 0.860823	valid_1's auc: 0.686191
[490]	training's auc: 0.861292	valid_1's auc: 0.686256
[500]	training's auc: 0.861697	valid_1's auc: 0.686351
complete on: language
model:
best score: 0.686351090659
best iteration: 0

                msno : 66107
             song_id : 18840
  source_screen_name : 17101
         source_type : 11783
         artist_name : 63063
           song_year : 28066
 ITC_song_id_log10_1 : 77048
    ITC_msno_log10_1 : 72835
        top2_in_song : 14907
            language : 4750
working on: song_country

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
song_country           category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'song_country',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.793998	valid_1's auc: 0.663355
[20]	training's auc: 0.801638	valid_1's auc: 0.666472
[30]	training's auc: 0.806855	valid_1's auc: 0.668619
[40]	training's auc: 0.811479	valid_1's auc: 0.6702
[50]	training's auc: 0.816583	valid_1's auc: 0.671898
[60]	training's auc: 0.821001	valid_1's auc: 0.673371
[70]	training's auc: 0.824745	valid_1's auc: 0.674565
[80]	training's auc: 0.827059	valid_1's auc: 0.67532
[90]	training's auc: 0.829585	valid_1's auc: 0.676218
[100]	training's auc: 0.832241	valid_1's auc: 0.677396
[110]	training's auc: 0.834244	valid_1's auc: 0.678294
[120]	training's auc: 0.835975	valid_1's auc: 0.679111
[130]	training's auc: 0.837324	valid_1's auc: 0.679785
[140]	training's auc: 0.838461	valid_1's auc: 0.680358
[150]	training's auc: 0.839507	valid_1's auc: 0.680785
[160]	training's auc: 0.840376	valid_1's auc: 0.681238
[170]	training's auc: 0.841277	valid_1's auc: 0.681599
[180]	training's auc: 0.842295	valid_1's auc: 0.681955
[190]	training's auc: 0.843203	valid_1's auc: 0.68228
[200]	training's auc: 0.844122	valid_1's auc: 0.682628
[210]	training's auc: 0.844977	valid_1's auc: 0.682882
[220]	training's auc: 0.845836	valid_1's auc: 0.683145
[230]	training's auc: 0.846529	valid_1's auc: 0.683347
[240]	training's auc: 0.84729	valid_1's auc: 0.683555
[250]	training's auc: 0.848006	valid_1's auc: 0.683752
[260]	training's auc: 0.848784	valid_1's auc: 0.683953
[270]	training's auc: 0.849524	valid_1's auc: 0.684115
[280]	training's auc: 0.850232	valid_1's auc: 0.684291
[290]	training's auc: 0.850858	valid_1's auc: 0.684389
[300]	training's auc: 0.851445	valid_1's auc: 0.684511
[310]	training's auc: 0.852027	valid_1's auc: 0.684696
[320]	training's auc: 0.85266	valid_1's auc: 0.684828
[330]	training's auc: 0.853239	valid_1's auc: 0.684895
[340]	training's auc: 0.853776	valid_1's auc: 0.684977
[350]	training's auc: 0.854364	valid_1's auc: 0.685081
[360]	training's auc: 0.85494	valid_1's auc: 0.685173
[370]	training's auc: 0.855516	valid_1's auc: 0.685275
[380]	training's auc: 0.856049	valid_1's auc: 0.685357
[390]	training's auc: 0.85664	valid_1's auc: 0.685436
[400]	training's auc: 0.857193	valid_1's auc: 0.685504
[410]	training's auc: 0.857718	valid_1's auc: 0.685605
[420]	training's auc: 0.858233	valid_1's auc: 0.685704
[430]	training's auc: 0.8587	valid_1's auc: 0.685763
[440]	training's auc: 0.859189	valid_1's auc: 0.685839
[450]	training's auc: 0.859663	valid_1's auc: 0.685924
[460]	training's auc: 0.860151	valid_1's auc: 0.685972
[470]	training's auc: 0.860598	valid_1's auc: 0.68606
[480]	training's auc: 0.861028	valid_1's auc: 0.686093
[490]	training's auc: 0.861443	valid_1's auc: 0.686167
[500]	training's auc: 0.861871	valid_1's auc: 0.686191
complete on: song_country
model:
best score: 0.686191389739
best iteration: 0

                msno : 66104
             song_id : 19185
  source_screen_name : 16646
         source_type : 11268
         artist_name : 62556
           song_year : 27284
 ITC_song_id_log10_1 : 77211
    ITC_msno_log10_1 : 70901
        top2_in_song : 14328
        song_country : 9017
working on: rc

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
rc                     category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'rc',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.793963	valid_1's auc: 0.662915
[20]	training's auc: 0.802262	valid_1's auc: 0.666261
[30]	training's auc: 0.807581	valid_1's auc: 0.66827
[40]	training's auc: 0.811878	valid_1's auc: 0.669806
[50]	training's auc: 0.817253	valid_1's auc: 0.671603
[60]	training's auc: 0.821609	valid_1's auc: 0.67307
[70]	training's auc: 0.825179	valid_1's auc: 0.67417
[80]	training's auc: 0.8278	valid_1's auc: 0.675055
[90]	training's auc: 0.83046	valid_1's auc: 0.676029
[100]	training's auc: 0.8331	valid_1's auc: 0.677129
[110]	training's auc: 0.835224	valid_1's auc: 0.678049
[120]	training's auc: 0.836834	valid_1's auc: 0.678874
[130]	training's auc: 0.838197	valid_1's auc: 0.679549
[140]	training's auc: 0.839275	valid_1's auc: 0.680075
[150]	training's auc: 0.840382	valid_1's auc: 0.680527
[160]	training's auc: 0.841356	valid_1's auc: 0.68099
[170]	training's auc: 0.842327	valid_1's auc: 0.681403
[180]	training's auc: 0.843215	valid_1's auc: 0.681758
[190]	training's auc: 0.844127	valid_1's auc: 0.682077
[200]	training's auc: 0.845042	valid_1's auc: 0.68239
[210]	training's auc: 0.845927	valid_1's auc: 0.682642
[220]	training's auc: 0.846817	valid_1's auc: 0.682955
[230]	training's auc: 0.847551	valid_1's auc: 0.683169
[240]	training's auc: 0.848306	valid_1's auc: 0.683365
[250]	training's auc: 0.849061	valid_1's auc: 0.683552
[260]	training's auc: 0.849806	valid_1's auc: 0.683748
[270]	training's auc: 0.850559	valid_1's auc: 0.683913
[280]	training's auc: 0.851215	valid_1's auc: 0.684051
[290]	training's auc: 0.851883	valid_1's auc: 0.684202
[300]	training's auc: 0.852478	valid_1's auc: 0.684317
[310]	training's auc: 0.853058	valid_1's auc: 0.684414
[320]	training's auc: 0.853638	valid_1's auc: 0.684499
[330]	training's auc: 0.854271	valid_1's auc: 0.68458
[340]	training's auc: 0.854836	valid_1's auc: 0.684676
[350]	training's auc: 0.85542	valid_1's auc: 0.684782
[360]	training's auc: 0.855989	valid_1's auc: 0.684848
[370]	training's auc: 0.856566	valid_1's auc: 0.684926
[380]	training's auc: 0.857142	valid_1's auc: 0.685026
[390]	training's auc: 0.857689	valid_1's auc: 0.685099
[400]	training's auc: 0.85824	valid_1's auc: 0.68514
[410]	training's auc: 0.858787	valid_1's auc: 0.685292
[420]	training's auc: 0.859252	valid_1's auc: 0.68535
[430]	training's auc: 0.859783	valid_1's auc: 0.685421
[440]	training's auc: 0.860293	valid_1's auc: 0.685497
[450]	training's auc: 0.860755	valid_1's auc: 0.685572
[460]	training's auc: 0.861239	valid_1's auc: 0.685632
[470]	training's auc: 0.861747	valid_1's auc: 0.685676
[480]	training's auc: 0.862222	valid_1's auc: 0.685747
[490]	training's auc: 0.862703	valid_1's auc: 0.685827
[500]	training's auc: 0.863157	valid_1's auc: 0.685875
complete on: rc
model:
best score: 0.685874608088
best iteration: 0

                msno : 65972
             song_id : 17201
  source_screen_name : 15825
         source_type : 10678
         artist_name : 55293
           song_year : 24640
 ITC_song_id_log10_1 : 72581
    ITC_msno_log10_1 : 68043
        top2_in_song : 11222
                  rc : 33045
working on: top1_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
top1_in_song           category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'top1_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794044	valid_1's auc: 0.663057
[20]	training's auc: 0.801275	valid_1's auc: 0.665968
[30]	training's auc: 0.806967	valid_1's auc: 0.668134
[40]	training's auc: 0.811385	valid_1's auc: 0.669697
[50]	training's auc: 0.816754	valid_1's auc: 0.671425
[60]	training's auc: 0.82112	valid_1's auc: 0.672911
[70]	training's auc: 0.824924	valid_1's auc: 0.674213
[80]	training's auc: 0.827354	valid_1's auc: 0.67499
[90]	training's auc: 0.829887	valid_1's auc: 0.675945
[100]	training's auc: 0.832501	valid_1's auc: 0.67704
[110]	training's auc: 0.83473	valid_1's auc: 0.677949
[120]	training's auc: 0.836227	valid_1's auc: 0.678736
[130]	training's auc: 0.837592	valid_1's auc: 0.679363
[140]	training's auc: 0.838576	valid_1's auc: 0.679803
[150]	training's auc: 0.839507	valid_1's auc: 0.680266
[160]	training's auc: 0.840517	valid_1's auc: 0.680703
[170]	training's auc: 0.841467	valid_1's auc: 0.681089
[180]	training's auc: 0.842354	valid_1's auc: 0.681434
[190]	training's auc: 0.843167	valid_1's auc: 0.681724
[200]	training's auc: 0.844006	valid_1's auc: 0.682013
[210]	training's auc: 0.844827	valid_1's auc: 0.682273
[220]	training's auc: 0.845569	valid_1's auc: 0.682524
[230]	training's auc: 0.846395	valid_1's auc: 0.682752
[240]	training's auc: 0.847194	valid_1's auc: 0.683024
[250]	training's auc: 0.847866	valid_1's auc: 0.683149
[260]	training's auc: 0.848609	valid_1's auc: 0.683395
[270]	training's auc: 0.849274	valid_1's auc: 0.683532
[280]	training's auc: 0.849889	valid_1's auc: 0.683643
[290]	training's auc: 0.8505	valid_1's auc: 0.683801
[300]	training's auc: 0.851109	valid_1's auc: 0.683976
[310]	training's auc: 0.851673	valid_1's auc: 0.684054
[320]	training's auc: 0.852329	valid_1's auc: 0.684239
[330]	training's auc: 0.852889	valid_1's auc: 0.684344
[340]	training's auc: 0.853472	valid_1's auc: 0.684422
[350]	training's auc: 0.854078	valid_1's auc: 0.684555
[360]	training's auc: 0.854633	valid_1's auc: 0.684684
[370]	training's auc: 0.85519	valid_1's auc: 0.684793
[380]	training's auc: 0.855747	valid_1's auc: 0.684899
[390]	training's auc: 0.856294	valid_1's auc: 0.685015
[400]	training's auc: 0.856878	valid_1's auc: 0.68516
[410]	training's auc: 0.857333	valid_1's auc: 0.685311
[420]	training's auc: 0.857799	valid_1's auc: 0.685434
[430]	training's auc: 0.858301	valid_1's auc: 0.685504
[440]	training's auc: 0.858764	valid_1's auc: 0.685606
[450]	training's auc: 0.859232	valid_1's auc: 0.685669
[460]	training's auc: 0.859668	valid_1's auc: 0.685703
[470]	training's auc: 0.860147	valid_1's auc: 0.685785
[480]	training's auc: 0.860596	valid_1's auc: 0.685873
[490]	training's auc: 0.861035	valid_1's auc: 0.685927
[500]	training's auc: 0.861447	valid_1's auc: 0.685981
complete on: top1_in_song
model:
best score: 0.685981388019
best iteration: 0

                msno : 65591
             song_id : 18243
  source_screen_name : 17136
         source_type : 11517
         artist_name : 62716
           song_year : 27828
 ITC_song_id_log10_1 : 79357
    ITC_msno_log10_1 : 72973
        top2_in_song : 11107
        top1_in_song : 8032
working on: top3_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
top3_in_song           category
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'top3_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794034	valid_1's auc: 0.662828
[20]	training's auc: 0.802241	valid_1's auc: 0.666391
[30]	training's auc: 0.807229	valid_1's auc: 0.668262
[40]	training's auc: 0.811403	valid_1's auc: 0.669791
[50]	training's auc: 0.816708	valid_1's auc: 0.671512
[60]	training's auc: 0.821127	valid_1's auc: 0.672964
[70]	training's auc: 0.82477	valid_1's auc: 0.674129
[80]	training's auc: 0.827119	valid_1's auc: 0.674973
[90]	training's auc: 0.82956	valid_1's auc: 0.675938
[100]	training's auc: 0.832384	valid_1's auc: 0.677214
[110]	training's auc: 0.834536	valid_1's auc: 0.678149
[120]	training's auc: 0.83611	valid_1's auc: 0.6789
[130]	training's auc: 0.8375	valid_1's auc: 0.679528
[140]	training's auc: 0.838715	valid_1's auc: 0.68014
[150]	training's auc: 0.839787	valid_1's auc: 0.680619
[160]	training's auc: 0.840717	valid_1's auc: 0.68106
[170]	training's auc: 0.841676	valid_1's auc: 0.681423
[180]	training's auc: 0.842551	valid_1's auc: 0.681771
[190]	training's auc: 0.843439	valid_1's auc: 0.682165
[200]	training's auc: 0.84432	valid_1's auc: 0.682438
[210]	training's auc: 0.845131	valid_1's auc: 0.682698
[220]	training's auc: 0.845877	valid_1's auc: 0.682925
[230]	training's auc: 0.846627	valid_1's auc: 0.683178
[240]	training's auc: 0.847325	valid_1's auc: 0.683354
[250]	training's auc: 0.847986	valid_1's auc: 0.683497
[260]	training's auc: 0.848692	valid_1's auc: 0.683658
[270]	training's auc: 0.849376	valid_1's auc: 0.683801
[280]	training's auc: 0.849993	valid_1's auc: 0.683952
[290]	training's auc: 0.850609	valid_1's auc: 0.684107
[300]	training's auc: 0.851184	valid_1's auc: 0.684222
[310]	training's auc: 0.851752	valid_1's auc: 0.68434
[320]	training's auc: 0.852365	valid_1's auc: 0.684501
[330]	training's auc: 0.852929	valid_1's auc: 0.684578
[340]	training's auc: 0.853582	valid_1's auc: 0.684757
[350]	training's auc: 0.854159	valid_1's auc: 0.684903
[360]	training's auc: 0.854731	valid_1's auc: 0.684998
[370]	training's auc: 0.855341	valid_1's auc: 0.68516
[380]	training's auc: 0.855886	valid_1's auc: 0.685245
[390]	training's auc: 0.856474	valid_1's auc: 0.68532
[400]	training's auc: 0.857001	valid_1's auc: 0.685397
[410]	training's auc: 0.857549	valid_1's auc: 0.685482
[420]	training's auc: 0.857974	valid_1's auc: 0.685579
[430]	training's auc: 0.858467	valid_1's auc: 0.685638
[440]	training's auc: 0.858901	valid_1's auc: 0.685687
[450]	training's auc: 0.859334	valid_1's auc: 0.685733
[460]	training's auc: 0.859782	valid_1's auc: 0.685786
[470]	training's auc: 0.86021	valid_1's auc: 0.685888
[480]	training's auc: 0.860648	valid_1's auc: 0.685943
[490]	training's auc: 0.861095	valid_1's auc: 0.685997
[500]	training's auc: 0.861488	valid_1's auc: 0.686024
complete on: top3_in_song
model:
best score: 0.686024160132
best iteration: 0

                msno : 66286
             song_id : 18146
  source_screen_name : 17088
         source_type : 11433
         artist_name : 62741
           song_year : 28429
 ITC_song_id_log10_1 : 78751
    ITC_msno_log10_1 : 72450
        top2_in_song : 11640
        top3_in_song : 7536
working on: membership_days

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
membership_days           int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'membership_days',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.796283	valid_1's auc: 0.664624
[20]	training's auc: 0.803899	valid_1's auc: 0.667505
[30]	training's auc: 0.808771	valid_1's auc: 0.669372
[40]	training's auc: 0.812942	valid_1's auc: 0.67078
[50]	training's auc: 0.817875	valid_1's auc: 0.672469
[60]	training's auc: 0.822188	valid_1's auc: 0.673827
[70]	training's auc: 0.825952	valid_1's auc: 0.674995
[80]	training's auc: 0.828383	valid_1's auc: 0.675789
[90]	training's auc: 0.831068	valid_1's auc: 0.676853
[100]	training's auc: 0.833575	valid_1's auc: 0.677878
[110]	training's auc: 0.835692	valid_1's auc: 0.678725
[120]	training's auc: 0.837458	valid_1's auc: 0.679557
[130]	training's auc: 0.83882	valid_1's auc: 0.680178
[140]	training's auc: 0.839944	valid_1's auc: 0.680728
[150]	training's auc: 0.840947	valid_1's auc: 0.68118
[160]	training's auc: 0.84186	valid_1's auc: 0.68156
[170]	training's auc: 0.842812	valid_1's auc: 0.681953
[180]	training's auc: 0.843633	valid_1's auc: 0.68227
[190]	training's auc: 0.844465	valid_1's auc: 0.68257
[200]	training's auc: 0.845297	valid_1's auc: 0.68285
[210]	training's auc: 0.846211	valid_1's auc: 0.683149
[220]	training's auc: 0.846957	valid_1's auc: 0.683334
[230]	training's auc: 0.847701	valid_1's auc: 0.683587
[240]	training's auc: 0.848384	valid_1's auc: 0.683747
[250]	training's auc: 0.849124	valid_1's auc: 0.683971
[260]	training's auc: 0.849867	valid_1's auc: 0.68421
[270]	training's auc: 0.850519	valid_1's auc: 0.684359
[280]	training's auc: 0.851145	valid_1's auc: 0.684526
[290]	training's auc: 0.851869	valid_1's auc: 0.684709
[300]	training's auc: 0.852444	valid_1's auc: 0.684806
[310]	training's auc: 0.853021	valid_1's auc: 0.684923
[320]	training's auc: 0.853646	valid_1's auc: 0.685034
[330]	training's auc: 0.854181	valid_1's auc: 0.685121
[340]	training's auc: 0.854763	valid_1's auc: 0.685254
[350]	training's auc: 0.855307	valid_1's auc: 0.685346
[360]	training's auc: 0.855915	valid_1's auc: 0.685478
[370]	training's auc: 0.85642	valid_1's auc: 0.685571
[380]	training's auc: 0.856926	valid_1's auc: 0.685622
[390]	training's auc: 0.857497	valid_1's auc: 0.685736
[400]	training's auc: 0.858018	valid_1's auc: 0.685802
[410]	training's auc: 0.858525	valid_1's auc: 0.685931
[420]	training's auc: 0.858944	valid_1's auc: 0.686065
[430]	training's auc: 0.859423	valid_1's auc: 0.686126
[440]	training's auc: 0.85988	valid_1's auc: 0.686181
[450]	training's auc: 0.860331	valid_1's auc: 0.686276
[460]	training's auc: 0.860772	valid_1's auc: 0.686369
[470]	training's auc: 0.861208	valid_1's auc: 0.6864
[480]	training's auc: 0.86168	valid_1's auc: 0.686464
[490]	training's auc: 0.862136	valid_1's auc: 0.686505
[500]	training's auc: 0.862532	valid_1's auc: 0.686529
complete on: membership_days
model:
best score: 0.686529285402
best iteration: 0

                msno : 66288
             song_id : 19121
  source_screen_name : 15643
         source_type : 10233
         artist_name : 61367
           song_year : 25757
 ITC_song_id_log10_1 : 66481
    ITC_msno_log10_1 : 52714
        top2_in_song : 14629
     membership_days : 42267
working on: song_year_int

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
song_year_int             int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794585	valid_1's auc: 0.663747
[20]	training's auc: 0.801961	valid_1's auc: 0.666489
[30]	training's auc: 0.807254	valid_1's auc: 0.668341
[40]	training's auc: 0.81149	valid_1's auc: 0.669815
[50]	training's auc: 0.81702	valid_1's auc: 0.671705
[60]	training's auc: 0.821393	valid_1's auc: 0.673194
[70]	training's auc: 0.825146	valid_1's auc: 0.674413
[80]	training's auc: 0.827411	valid_1's auc: 0.675171
[90]	training's auc: 0.829955	valid_1's auc: 0.67615
[100]	training's auc: 0.83269	valid_1's auc: 0.677315
[110]	training's auc: 0.834712	valid_1's auc: 0.678147
[120]	training's auc: 0.836341	valid_1's auc: 0.678961
[130]	training's auc: 0.837643	valid_1's auc: 0.679628
[140]	training's auc: 0.838836	valid_1's auc: 0.680189
[150]	training's auc: 0.839824	valid_1's auc: 0.680641
[160]	training's auc: 0.840785	valid_1's auc: 0.681082
[170]	training's auc: 0.84176	valid_1's auc: 0.681507
[180]	training's auc: 0.842878	valid_1's auc: 0.681882
[190]	training's auc: 0.843624	valid_1's auc: 0.682123
[200]	training's auc: 0.844452	valid_1's auc: 0.682395
[210]	training's auc: 0.845294	valid_1's auc: 0.682674
[220]	training's auc: 0.846071	valid_1's auc: 0.682931
[230]	training's auc: 0.846749	valid_1's auc: 0.68314
[240]	training's auc: 0.847526	valid_1's auc: 0.68336
[250]	training's auc: 0.848211	valid_1's auc: 0.683538
[260]	training's auc: 0.849034	valid_1's auc: 0.683717
[270]	training's auc: 0.849656	valid_1's auc: 0.683857
[280]	training's auc: 0.850306	valid_1's auc: 0.684041
[290]	training's auc: 0.850902	valid_1's auc: 0.684195
[300]	training's auc: 0.851532	valid_1's auc: 0.684352
[310]	training's auc: 0.852109	valid_1's auc: 0.684497
[320]	training's auc: 0.852694	valid_1's auc: 0.684628
[330]	training's auc: 0.85325	valid_1's auc: 0.684712
[340]	training's auc: 0.853836	valid_1's auc: 0.684818
[350]	training's auc: 0.854387	valid_1's auc: 0.684925
[360]	training's auc: 0.854954	valid_1's auc: 0.68503
[370]	training's auc: 0.855453	valid_1's auc: 0.6851
[380]	training's auc: 0.855995	valid_1's auc: 0.685174
[390]	training's auc: 0.85655	valid_1's auc: 0.685303
[400]	training's auc: 0.8571	valid_1's auc: 0.685384
[410]	training's auc: 0.857592	valid_1's auc: 0.685498
[420]	training's auc: 0.858051	valid_1's auc: 0.68567
[430]	training's auc: 0.858533	valid_1's auc: 0.685714
[440]	training's auc: 0.858983	valid_1's auc: 0.685759
[450]	training's auc: 0.859425	valid_1's auc: 0.68589
[460]	training's auc: 0.859901	valid_1's auc: 0.685919
[470]	training's auc: 0.86037	valid_1's auc: 0.685966
[480]	training's auc: 0.860817	valid_1's auc: 0.68603
[490]	training's auc: 0.861236	valid_1's auc: 0.686151
[500]	training's auc: 0.86165	valid_1's auc: 0.68622
complete on: song_year_int
model:
best score: 0.686220067189
best iteration: 0

                msno : 66284
             song_id : 18268
  source_screen_name : 16658
         source_type : 11020
         artist_name : 61768
           song_year : 24788
 ITC_song_id_log10_1 : 70141
    ITC_msno_log10_1 : 65199
        top2_in_song : 15052
       song_year_int : 25322
working on: ISC_top1_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_top1_in_song          int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_top1_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.793878	valid_1's auc: 0.662695
[20]	training's auc: 0.801857	valid_1's auc: 0.665702
[30]	training's auc: 0.807202	valid_1's auc: 0.667805
[40]	training's auc: 0.811357	valid_1's auc: 0.669207
[50]	training's auc: 0.816855	valid_1's auc: 0.67121
[60]	training's auc: 0.821351	valid_1's auc: 0.672672
[70]	training's auc: 0.824987	valid_1's auc: 0.673905
[80]	training's auc: 0.82742	valid_1's auc: 0.674799
[90]	training's auc: 0.829918	valid_1's auc: 0.675813
[100]	training's auc: 0.832341	valid_1's auc: 0.676836
[110]	training's auc: 0.834581	valid_1's auc: 0.677822
[120]	training's auc: 0.836236	valid_1's auc: 0.678647
[130]	training's auc: 0.837655	valid_1's auc: 0.679295
[140]	training's auc: 0.838791	valid_1's auc: 0.679841
[150]	training's auc: 0.839862	valid_1's auc: 0.680368
[160]	training's auc: 0.840766	valid_1's auc: 0.680754
[170]	training's auc: 0.841718	valid_1's auc: 0.681139
[180]	training's auc: 0.842587	valid_1's auc: 0.681504
[190]	training's auc: 0.84341	valid_1's auc: 0.681807
[200]	training's auc: 0.84424	valid_1's auc: 0.68212
[210]	training's auc: 0.845164	valid_1's auc: 0.682397
[220]	training's auc: 0.845885	valid_1's auc: 0.682624
[230]	training's auc: 0.846581	valid_1's auc: 0.682842
[240]	training's auc: 0.847325	valid_1's auc: 0.683052
[250]	training's auc: 0.848015	valid_1's auc: 0.683265
[260]	training's auc: 0.848731	valid_1's auc: 0.683416
[270]	training's auc: 0.84947	valid_1's auc: 0.683607
[280]	training's auc: 0.850192	valid_1's auc: 0.683784
[290]	training's auc: 0.850787	valid_1's auc: 0.683919
[300]	training's auc: 0.851403	valid_1's auc: 0.68407
[310]	training's auc: 0.852021	valid_1's auc: 0.684207
[320]	training's auc: 0.852577	valid_1's auc: 0.684329
[330]	training's auc: 0.853125	valid_1's auc: 0.684435
[340]	training's auc: 0.853641	valid_1's auc: 0.684519
[350]	training's auc: 0.854218	valid_1's auc: 0.684636
[360]	training's auc: 0.854746	valid_1's auc: 0.684692
[370]	training's auc: 0.855335	valid_1's auc: 0.684796
[380]	training's auc: 0.855894	valid_1's auc: 0.684875
[390]	training's auc: 0.856462	valid_1's auc: 0.684962
[400]	training's auc: 0.856995	valid_1's auc: 0.685032
[410]	training's auc: 0.857546	valid_1's auc: 0.685145
[420]	training's auc: 0.858051	valid_1's auc: 0.685218
[430]	training's auc: 0.858509	valid_1's auc: 0.685257
[440]	training's auc: 0.858979	valid_1's auc: 0.685284
[450]	training's auc: 0.859388	valid_1's auc: 0.685396
[460]	training's auc: 0.859833	valid_1's auc: 0.68548
[470]	training's auc: 0.860248	valid_1's auc: 0.685523
[480]	training's auc: 0.860684	valid_1's auc: 0.685563
[490]	training's auc: 0.861118	valid_1's auc: 0.685599
[500]	training's auc: 0.861537	valid_1's auc: 0.685672
complete on: ISC_top1_in_song
model:
best score: 0.685672321673
best iteration: 0

                msno : 66113
             song_id : 18489
  source_screen_name : 16645
         source_type : 11013
         artist_name : 62183
           song_year : 26988
 ITC_song_id_log10_1 : 72328
    ITC_msno_log10_1 : 66530
        top2_in_song : 14064
    ISC_top1_in_song : 20147
working on: ISC_top2_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_top2_in_song          int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_top2_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794302	valid_1's auc: 0.663266
[20]	training's auc: 0.801731	valid_1's auc: 0.66616
[30]	training's auc: 0.807143	valid_1's auc: 0.668245
[40]	training's auc: 0.811573	valid_1's auc: 0.66978
[50]	training's auc: 0.81694	valid_1's auc: 0.671532
[60]	training's auc: 0.821486	valid_1's auc: 0.672983
[70]	training's auc: 0.82504	valid_1's auc: 0.674216
[80]	training's auc: 0.827589	valid_1's auc: 0.67506
[90]	training's auc: 0.83011	valid_1's auc: 0.676012
[100]	training's auc: 0.832524	valid_1's auc: 0.677047
[110]	training's auc: 0.834669	valid_1's auc: 0.677944
[120]	training's auc: 0.836308	valid_1's auc: 0.678749
[130]	training's auc: 0.837631	valid_1's auc: 0.679387
[140]	training's auc: 0.838778	valid_1's auc: 0.679954
[150]	training's auc: 0.839797	valid_1's auc: 0.680357
[160]	training's auc: 0.840741	valid_1's auc: 0.680754
[170]	training's auc: 0.841609	valid_1's auc: 0.681127
[180]	training's auc: 0.842559	valid_1's auc: 0.681496
[190]	training's auc: 0.84346	valid_1's auc: 0.681839
[200]	training's auc: 0.844339	valid_1's auc: 0.682138
[210]	training's auc: 0.845288	valid_1's auc: 0.682441
[220]	training's auc: 0.846121	valid_1's auc: 0.682754
[230]	training's auc: 0.846857	valid_1's auc: 0.682972
[240]	training's auc: 0.847606	valid_1's auc: 0.683179
[250]	training's auc: 0.848338	valid_1's auc: 0.683356
[260]	training's auc: 0.849031	valid_1's auc: 0.683526
[270]	training's auc: 0.849707	valid_1's auc: 0.683681
[280]	training's auc: 0.850315	valid_1's auc: 0.683818
[290]	training's auc: 0.850966	valid_1's auc: 0.683925
[300]	training's auc: 0.851601	valid_1's auc: 0.684064
[310]	training's auc: 0.852168	valid_1's auc: 0.684178
[320]	training's auc: 0.852721	valid_1's auc: 0.68426
[330]	training's auc: 0.853314	valid_1's auc: 0.684379
[340]	training's auc: 0.853872	valid_1's auc: 0.684472
[350]	training's auc: 0.854477	valid_1's auc: 0.684545
[360]	training's auc: 0.85502	valid_1's auc: 0.684636
[370]	training's auc: 0.855584	valid_1's auc: 0.684741
[380]	training's auc: 0.856157	valid_1's auc: 0.68483
[390]	training's auc: 0.85671	valid_1's auc: 0.684903
[400]	training's auc: 0.857265	valid_1's auc: 0.685025
[410]	training's auc: 0.857805	valid_1's auc: 0.685124
[420]	training's auc: 0.85824	valid_1's auc: 0.685174
[430]	training's auc: 0.858702	valid_1's auc: 0.68524
[440]	training's auc: 0.859166	valid_1's auc: 0.685327
[450]	training's auc: 0.859635	valid_1's auc: 0.685362
[460]	training's auc: 0.860053	valid_1's auc: 0.685412
[470]	training's auc: 0.860484	valid_1's auc: 0.685444
[480]	training's auc: 0.860963	valid_1's auc: 0.685501
[490]	training's auc: 0.861424	valid_1's auc: 0.685565
[500]	training's auc: 0.861864	valid_1's auc: 0.685604
complete on: ISC_top2_in_song
model:
best score: 0.685603515281
best iteration: 0

                msno : 66447
             song_id : 18334
  source_screen_name : 16337
         source_type : 11113
         artist_name : 61466
           song_year : 26565
 ITC_song_id_log10_1 : 72934
    ITC_msno_log10_1 : 67136
        top2_in_song : 13646
    ISC_top2_in_song : 20522
working on: ISC_top3_in_song

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_top3_in_song          int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_top3_in_song',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794312	valid_1's auc: 0.663677
[20]	training's auc: 0.801755	valid_1's auc: 0.666417
[30]	training's auc: 0.806896	valid_1's auc: 0.668426
[40]	training's auc: 0.811244	valid_1's auc: 0.669863
[50]	training's auc: 0.81652	valid_1's auc: 0.671722
[60]	training's auc: 0.821126	valid_1's auc: 0.673253
[70]	training's auc: 0.82487	valid_1's auc: 0.674433
[80]	training's auc: 0.827404	valid_1's auc: 0.675272
[90]	training's auc: 0.829898	valid_1's auc: 0.67623
[100]	training's auc: 0.832376	valid_1's auc: 0.677329
[110]	training's auc: 0.834418	valid_1's auc: 0.678151
[120]	training's auc: 0.835983	valid_1's auc: 0.678969
[130]	training's auc: 0.837416	valid_1's auc: 0.679652
[140]	training's auc: 0.838582	valid_1's auc: 0.680196
[150]	training's auc: 0.839688	valid_1's auc: 0.680755
[160]	training's auc: 0.840625	valid_1's auc: 0.681191
[170]	training's auc: 0.841528	valid_1's auc: 0.681563
[180]	training's auc: 0.842393	valid_1's auc: 0.681927
[190]	training's auc: 0.843307	valid_1's auc: 0.682236
[200]	training's auc: 0.844125	valid_1's auc: 0.682501
[210]	training's auc: 0.84499	valid_1's auc: 0.68279
[220]	training's auc: 0.845736	valid_1's auc: 0.683057
[230]	training's auc: 0.846447	valid_1's auc: 0.68327
[240]	training's auc: 0.847173	valid_1's auc: 0.683468
[250]	training's auc: 0.84787	valid_1's auc: 0.683669
[260]	training's auc: 0.848604	valid_1's auc: 0.683862
[270]	training's auc: 0.84934	valid_1's auc: 0.684054
[280]	training's auc: 0.850106	valid_1's auc: 0.684248
[290]	training's auc: 0.85076	valid_1's auc: 0.684355
[300]	training's auc: 0.851348	valid_1's auc: 0.684465
[310]	training's auc: 0.851878	valid_1's auc: 0.68458
[320]	training's auc: 0.852436	valid_1's auc: 0.684683
[330]	training's auc: 0.852954	valid_1's auc: 0.68478
[340]	training's auc: 0.853528	valid_1's auc: 0.684896
[350]	training's auc: 0.854104	valid_1's auc: 0.684992
[360]	training's auc: 0.854633	valid_1's auc: 0.685097
[370]	training's auc: 0.855159	valid_1's auc: 0.685162
[380]	training's auc: 0.855715	valid_1's auc: 0.685228
[390]	training's auc: 0.856329	valid_1's auc: 0.685372
[400]	training's auc: 0.85684	valid_1's auc: 0.685442
[410]	training's auc: 0.857403	valid_1's auc: 0.685542
[420]	training's auc: 0.857873	valid_1's auc: 0.685607
[430]	training's auc: 0.858363	valid_1's auc: 0.685656
[440]	training's auc: 0.858819	valid_1's auc: 0.685719
[450]	training's auc: 0.859273	valid_1's auc: 0.685805
[460]	training's auc: 0.859795	valid_1's auc: 0.685885
[470]	training's auc: 0.860237	valid_1's auc: 0.685922
[480]	training's auc: 0.860712	valid_1's auc: 0.68601
[490]	training's auc: 0.861169	valid_1's auc: 0.68607
[500]	training's auc: 0.861596	valid_1's auc: 0.68611
complete on: ISC_top3_in_song
model:
best score: 0.686109699775
best iteration: 0

                msno : 66243
             song_id : 18437
  source_screen_name : 16672
         source_type : 11331
         artist_name : 62541
           song_year : 27299
 ITC_song_id_log10_1 : 71796
    ITC_msno_log10_1 : 66099
        top2_in_song : 14286
    ISC_top3_in_song : 19796
working on: ISC_language

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_language              int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.79431	valid_1's auc: 0.663297
[20]	training's auc: 0.80206	valid_1's auc: 0.666161
[30]	training's auc: 0.806625	valid_1's auc: 0.668045
[40]	training's auc: 0.8112	valid_1's auc: 0.669728
[50]	training's auc: 0.816379	valid_1's auc: 0.671426
[60]	training's auc: 0.821037	valid_1's auc: 0.673021
[70]	training's auc: 0.824861	valid_1's auc: 0.674229
[80]	training's auc: 0.827116	valid_1's auc: 0.674889
[90]	training's auc: 0.829594	valid_1's auc: 0.675876
[100]	training's auc: 0.832244	valid_1's auc: 0.677021
[110]	training's auc: 0.834365	valid_1's auc: 0.677961
[120]	training's auc: 0.835972	valid_1's auc: 0.678763
[130]	training's auc: 0.837459	valid_1's auc: 0.679521
[140]	training's auc: 0.838542	valid_1's auc: 0.680048
[150]	training's auc: 0.839563	valid_1's auc: 0.680571
[160]	training's auc: 0.840573	valid_1's auc: 0.681003
[170]	training's auc: 0.841519	valid_1's auc: 0.681384
[180]	training's auc: 0.842537	valid_1's auc: 0.681798
[190]	training's auc: 0.843417	valid_1's auc: 0.682141
[200]	training's auc: 0.844258	valid_1's auc: 0.682417
[210]	training's auc: 0.845088	valid_1's auc: 0.682688
[220]	training's auc: 0.845908	valid_1's auc: 0.682964
[230]	training's auc: 0.846631	valid_1's auc: 0.683171
[240]	training's auc: 0.84738	valid_1's auc: 0.683352
[250]	training's auc: 0.848085	valid_1's auc: 0.683546
[260]	training's auc: 0.848751	valid_1's auc: 0.683693
[270]	training's auc: 0.849443	valid_1's auc: 0.683873
[280]	training's auc: 0.850207	valid_1's auc: 0.684107
[290]	training's auc: 0.850916	valid_1's auc: 0.684302
[300]	training's auc: 0.851473	valid_1's auc: 0.684422
[310]	training's auc: 0.852025	valid_1's auc: 0.684516
[320]	training's auc: 0.852649	valid_1's auc: 0.684649
[330]	training's auc: 0.853197	valid_1's auc: 0.684737
[340]	training's auc: 0.85377	valid_1's auc: 0.68483
[350]	training's auc: 0.854338	valid_1's auc: 0.684931
[360]	training's auc: 0.854886	valid_1's auc: 0.685042
[370]	training's auc: 0.8554	valid_1's auc: 0.685122
[380]	training's auc: 0.855956	valid_1's auc: 0.685207
[390]	training's auc: 0.856533	valid_1's auc: 0.685284
[400]	training's auc: 0.857076	valid_1's auc: 0.685385
[410]	training's auc: 0.857569	valid_1's auc: 0.685503
[420]	training's auc: 0.858029	valid_1's auc: 0.685638
[430]	training's auc: 0.858539	valid_1's auc: 0.685691
[440]	training's auc: 0.858996	valid_1's auc: 0.685772
[450]	training's auc: 0.859455	valid_1's auc: 0.685833
[460]	training's auc: 0.859929	valid_1's auc: 0.685884
[470]	training's auc: 0.860378	valid_1's auc: 0.68594
[480]	training's auc: 0.860831	valid_1's auc: 0.685993
[490]	training's auc: 0.861275	valid_1's auc: 0.686034
[500]	training's auc: 0.861723	valid_1's auc: 0.686103
complete on: ISC_language
model:
best score: 0.68610324517
best iteration: 0

                msno : 66153
             song_id : 18576
  source_screen_name : 16819
         source_type : 11250
         artist_name : 62212
           song_year : 26933
 ITC_song_id_log10_1 : 71913
    ITC_msno_log10_1 : 67069
        top2_in_song : 14584
        ISC_language : 18991
working on: ISCZ_rc

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISCZ_rc                   int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISCZ_rc',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794623	valid_1's auc: 0.663271
[20]	training's auc: 0.802046	valid_1's auc: 0.666154
[30]	training's auc: 0.806854	valid_1's auc: 0.668124
[40]	training's auc: 0.81125	valid_1's auc: 0.669723
[50]	training's auc: 0.816749	valid_1's auc: 0.671588
[60]	training's auc: 0.821177	valid_1's auc: 0.673105
[70]	training's auc: 0.824874	valid_1's auc: 0.674297
[80]	training's auc: 0.827366	valid_1's auc: 0.675132
[90]	training's auc: 0.830114	valid_1's auc: 0.676202
[100]	training's auc: 0.83261	valid_1's auc: 0.677247
[110]	training's auc: 0.834773	valid_1's auc: 0.6782
[120]	training's auc: 0.836449	valid_1's auc: 0.678973
[130]	training's auc: 0.837762	valid_1's auc: 0.679604
[140]	training's auc: 0.838971	valid_1's auc: 0.680131
[150]	training's auc: 0.840006	valid_1's auc: 0.680579
[160]	training's auc: 0.840945	valid_1's auc: 0.681026
[170]	training's auc: 0.841929	valid_1's auc: 0.681426
[180]	training's auc: 0.842869	valid_1's auc: 0.681769
[190]	training's auc: 0.843797	valid_1's auc: 0.682105
[200]	training's auc: 0.84463	valid_1's auc: 0.682367
[210]	training's auc: 0.845431	valid_1's auc: 0.682605
[220]	training's auc: 0.846237	valid_1's auc: 0.682903
[230]	training's auc: 0.846927	valid_1's auc: 0.683128
[240]	training's auc: 0.847704	valid_1's auc: 0.683353
[250]	training's auc: 0.848389	valid_1's auc: 0.683559
[260]	training's auc: 0.849088	valid_1's auc: 0.683739
[270]	training's auc: 0.849761	valid_1's auc: 0.683907
[280]	training's auc: 0.850364	valid_1's auc: 0.684047
[290]	training's auc: 0.850957	valid_1's auc: 0.68416
[300]	training's auc: 0.851548	valid_1's auc: 0.684261
[310]	training's auc: 0.852096	valid_1's auc: 0.684394
[320]	training's auc: 0.852692	valid_1's auc: 0.68457
[330]	training's auc: 0.853264	valid_1's auc: 0.684723
[340]	training's auc: 0.853833	valid_1's auc: 0.684822
[350]	training's auc: 0.854396	valid_1's auc: 0.684951
[360]	training's auc: 0.85501	valid_1's auc: 0.68506
[370]	training's auc: 0.855532	valid_1's auc: 0.685159
[380]	training's auc: 0.856087	valid_1's auc: 0.685258
[390]	training's auc: 0.856628	valid_1's auc: 0.685415
[400]	training's auc: 0.857158	valid_1's auc: 0.685569
[410]	training's auc: 0.857748	valid_1's auc: 0.685718
[420]	training's auc: 0.858204	valid_1's auc: 0.685901
[430]	training's auc: 0.858705	valid_1's auc: 0.685967
[440]	training's auc: 0.859162	valid_1's auc: 0.686026
[450]	training's auc: 0.859642	valid_1's auc: 0.686067
[460]	training's auc: 0.860075	valid_1's auc: 0.686113
[470]	training's auc: 0.860524	valid_1's auc: 0.686145
[480]	training's auc: 0.860966	valid_1's auc: 0.6862
[490]	training's auc: 0.861402	valid_1's auc: 0.686235
[500]	training's auc: 0.861828	valid_1's auc: 0.686291
complete on: ISCZ_rc
model:
best score: 0.686290768709
best iteration: 0

                msno : 66326
             song_id : 18391
  source_screen_name : 15803
         source_type : 10680
         artist_name : 60818
           song_year : 25486
 ITC_song_id_log10_1 : 64364
    ITC_msno_log10_1 : 59994
        top2_in_song : 14423
             ISCZ_rc : 38215
working on: ISCZ_isrc_rest

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISCZ_isrc_rest            int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISCZ_isrc_rest',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794156	valid_1's auc: 0.663324
[20]	training's auc: 0.80167	valid_1's auc: 0.666382
[30]	training's auc: 0.806751	valid_1's auc: 0.668403
[40]	training's auc: 0.811038	valid_1's auc: 0.669869
[50]	training's auc: 0.816347	valid_1's auc: 0.671622
[60]	training's auc: 0.821186	valid_1's auc: 0.673215
[70]	training's auc: 0.824935	valid_1's auc: 0.674517
[80]	training's auc: 0.827309	valid_1's auc: 0.67534
[90]	training's auc: 0.829974	valid_1's auc: 0.676422
[100]	training's auc: 0.832335	valid_1's auc: 0.677377
[110]	training's auc: 0.834463	valid_1's auc: 0.678336
[120]	training's auc: 0.835956	valid_1's auc: 0.67913
[130]	training's auc: 0.837237	valid_1's auc: 0.679706
[140]	training's auc: 0.838291	valid_1's auc: 0.680256
[150]	training's auc: 0.839385	valid_1's auc: 0.68072
[160]	training's auc: 0.840343	valid_1's auc: 0.681137
[170]	training's auc: 0.841402	valid_1's auc: 0.681556
[180]	training's auc: 0.842375	valid_1's auc: 0.68194
[190]	training's auc: 0.843211	valid_1's auc: 0.682274
[200]	training's auc: 0.844074	valid_1's auc: 0.682542
[210]	training's auc: 0.84495	valid_1's auc: 0.682799
[220]	training's auc: 0.845695	valid_1's auc: 0.683056
[230]	training's auc: 0.846396	valid_1's auc: 0.683274
[240]	training's auc: 0.847208	valid_1's auc: 0.683497
[250]	training's auc: 0.848026	valid_1's auc: 0.683682
[260]	training's auc: 0.848708	valid_1's auc: 0.683819
[270]	training's auc: 0.849426	valid_1's auc: 0.683977
[280]	training's auc: 0.850077	valid_1's auc: 0.68413
[290]	training's auc: 0.850774	valid_1's auc: 0.684362
[300]	training's auc: 0.85133	valid_1's auc: 0.684477
[310]	training's auc: 0.851852	valid_1's auc: 0.684586
[320]	training's auc: 0.85242	valid_1's auc: 0.684704
[330]	training's auc: 0.853097	valid_1's auc: 0.684858
[340]	training's auc: 0.853648	valid_1's auc: 0.684954
[350]	training's auc: 0.854217	valid_1's auc: 0.68506
[360]	training's auc: 0.854772	valid_1's auc: 0.685113
[370]	training's auc: 0.855272	valid_1's auc: 0.685227
[380]	training's auc: 0.855862	valid_1's auc: 0.685328
[390]	training's auc: 0.856423	valid_1's auc: 0.685439
[400]	training's auc: 0.856958	valid_1's auc: 0.685504
[410]	training's auc: 0.857499	valid_1's auc: 0.685587
[420]	training's auc: 0.857947	valid_1's auc: 0.685656
[430]	training's auc: 0.858457	valid_1's auc: 0.685723
[440]	training's auc: 0.858931	valid_1's auc: 0.685812
[450]	training's auc: 0.859413	valid_1's auc: 0.685871
[460]	training's auc: 0.859875	valid_1's auc: 0.685907
[470]	training's auc: 0.860393	valid_1's auc: 0.685978
[480]	training's auc: 0.860828	valid_1's auc: 0.68604
[490]	training's auc: 0.861269	valid_1's auc: 0.68609
[500]	training's auc: 0.861684	valid_1's auc: 0.686135
complete on: ISCZ_isrc_rest
model:
best score: 0.686135360636
best iteration: 0

                msno : 66370
             song_id : 18055
  source_screen_name : 15654
         source_type : 10509
         artist_name : 60846
           song_year : 24915
 ITC_song_id_log10_1 : 62477
    ITC_msno_log10_1 : 58307
        top2_in_song : 14539
      ISCZ_isrc_rest : 42828
working on: ISC_song_year

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_song_year             int64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_song_year',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794573	valid_1's auc: 0.664001
[20]	training's auc: 0.802093	valid_1's auc: 0.666857
[30]	training's auc: 0.807114	valid_1's auc: 0.66882
[40]	training's auc: 0.811519	valid_1's auc: 0.670264
[50]	training's auc: 0.816534	valid_1's auc: 0.671889
[60]	training's auc: 0.821102	valid_1's auc: 0.673366
[70]	training's auc: 0.824942	valid_1's auc: 0.674568
[80]	training's auc: 0.827233	valid_1's auc: 0.675392
[90]	training's auc: 0.829977	valid_1's auc: 0.676453
[100]	training's auc: 0.832328	valid_1's auc: 0.677457
[110]	training's auc: 0.83446	valid_1's auc: 0.678404
[120]	training's auc: 0.836021	valid_1's auc: 0.679203
[130]	training's auc: 0.837422	valid_1's auc: 0.679762
[140]	training's auc: 0.838653	valid_1's auc: 0.680334
[150]	training's auc: 0.839588	valid_1's auc: 0.680771
[160]	training's auc: 0.840521	valid_1's auc: 0.681204
[170]	training's auc: 0.841471	valid_1's auc: 0.681564
[180]	training's auc: 0.842487	valid_1's auc: 0.681929
[190]	training's auc: 0.84328	valid_1's auc: 0.682265
[200]	training's auc: 0.844112	valid_1's auc: 0.682529
[210]	training's auc: 0.844976	valid_1's auc: 0.6828
[220]	training's auc: 0.845694	valid_1's auc: 0.683038
[230]	training's auc: 0.846405	valid_1's auc: 0.683238
[240]	training's auc: 0.847202	valid_1's auc: 0.683451
[250]	training's auc: 0.847957	valid_1's auc: 0.683667
[260]	training's auc: 0.848659	valid_1's auc: 0.683855
[270]	training's auc: 0.849343	valid_1's auc: 0.68401
[280]	training's auc: 0.849997	valid_1's auc: 0.68417
[290]	training's auc: 0.850628	valid_1's auc: 0.684368
[300]	training's auc: 0.851258	valid_1's auc: 0.684486
[310]	training's auc: 0.851845	valid_1's auc: 0.684644
[320]	training's auc: 0.852454	valid_1's auc: 0.684802
[330]	training's auc: 0.853032	valid_1's auc: 0.684917
[340]	training's auc: 0.853622	valid_1's auc: 0.685039
[350]	training's auc: 0.854153	valid_1's auc: 0.685129
[360]	training's auc: 0.85471	valid_1's auc: 0.6852
[370]	training's auc: 0.855223	valid_1's auc: 0.685297
[380]	training's auc: 0.855756	valid_1's auc: 0.685396
[390]	training's auc: 0.856285	valid_1's auc: 0.685471
[400]	training's auc: 0.856817	valid_1's auc: 0.685519
[410]	training's auc: 0.857339	valid_1's auc: 0.685637
[420]	training's auc: 0.85781	valid_1's auc: 0.68598
[430]	training's auc: 0.858331	valid_1's auc: 0.686053
[440]	training's auc: 0.858805	valid_1's auc: 0.686127
[450]	training's auc: 0.859344	valid_1's auc: 0.686177
[460]	training's auc: 0.85979	valid_1's auc: 0.686263
[470]	training's auc: 0.86025	valid_1's auc: 0.686309
[480]	training's auc: 0.860676	valid_1's auc: 0.686355
[490]	training's auc: 0.86108	valid_1's auc: 0.686382
[500]	training's auc: 0.861474	valid_1's auc: 0.686452
complete on: ISC_song_year
model:
best score: 0.686452117745
best iteration: 0

                msno : 65510
             song_id : 18864
  source_screen_name : 16736
         source_type : 11121
         artist_name : 62276
           song_year : 25078
 ITC_song_id_log10_1 : 69279
    ITC_msno_log10_1 : 64392
        top2_in_song : 15461
       ISC_song_year : 25783
working on: song_length_log10

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
song_length_log10       float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'song_length_log10',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794972	valid_1's auc: 0.663914
[20]	training's auc: 0.801787	valid_1's auc: 0.666517
[30]	training's auc: 0.807272	valid_1's auc: 0.668677
[40]	training's auc: 0.811771	valid_1's auc: 0.670084
[50]	training's auc: 0.817207	valid_1's auc: 0.671716
[60]	training's auc: 0.821806	valid_1's auc: 0.67324
[70]	training's auc: 0.825336	valid_1's auc: 0.674388
[80]	training's auc: 0.827667	valid_1's auc: 0.675121
[90]	training's auc: 0.830347	valid_1's auc: 0.676233
[100]	training's auc: 0.83261	valid_1's auc: 0.677111
[110]	training's auc: 0.83477	valid_1's auc: 0.678062
[120]	training's auc: 0.836502	valid_1's auc: 0.678886
[130]	training's auc: 0.837855	valid_1's auc: 0.679481
[140]	training's auc: 0.83898	valid_1's auc: 0.679977
[150]	training's auc: 0.840011	valid_1's auc: 0.68046
[160]	training's auc: 0.840998	valid_1's auc: 0.680886
[170]	training's auc: 0.841909	valid_1's auc: 0.681236
[180]	training's auc: 0.842793	valid_1's auc: 0.681516
[190]	training's auc: 0.843711	valid_1's auc: 0.681881
[200]	training's auc: 0.844541	valid_1's auc: 0.682171
[210]	training's auc: 0.845442	valid_1's auc: 0.682434
[220]	training's auc: 0.846148	valid_1's auc: 0.68265
[230]	training's auc: 0.846899	valid_1's auc: 0.682904
[240]	training's auc: 0.847622	valid_1's auc: 0.683085
[250]	training's auc: 0.848383	valid_1's auc: 0.683301
[260]	training's auc: 0.849118	valid_1's auc: 0.683467
[270]	training's auc: 0.849812	valid_1's auc: 0.68363
[280]	training's auc: 0.850408	valid_1's auc: 0.683757
[290]	training's auc: 0.851095	valid_1's auc: 0.683916
[300]	training's auc: 0.851717	valid_1's auc: 0.684039
[310]	training's auc: 0.852275	valid_1's auc: 0.684137
[320]	training's auc: 0.852826	valid_1's auc: 0.684221
[330]	training's auc: 0.853395	valid_1's auc: 0.684314
[340]	training's auc: 0.853969	valid_1's auc: 0.684406
[350]	training's auc: 0.854614	valid_1's auc: 0.684541
[360]	training's auc: 0.855199	valid_1's auc: 0.684618
[370]	training's auc: 0.85572	valid_1's auc: 0.684688
[380]	training's auc: 0.856257	valid_1's auc: 0.684781
[390]	training's auc: 0.85679	valid_1's auc: 0.684854
[400]	training's auc: 0.857356	valid_1's auc: 0.685
[410]	training's auc: 0.857865	valid_1's auc: 0.685044
[420]	training's auc: 0.85831	valid_1's auc: 0.685218
[430]	training's auc: 0.858822	valid_1's auc: 0.685283
[440]	training's auc: 0.859269	valid_1's auc: 0.685336
[450]	training's auc: 0.859709	valid_1's auc: 0.685393
[460]	training's auc: 0.86021	valid_1's auc: 0.685451
[470]	training's auc: 0.860651	valid_1's auc: 0.685516
[480]	training's auc: 0.86107	valid_1's auc: 0.685555
[490]	training's auc: 0.861527	valid_1's auc: 0.685601
[500]	training's auc: 0.861936	valid_1's auc: 0.685633
complete on: song_length_log10
model:
best score: 0.685632537696
best iteration: 0

                msno : 66222
             song_id : 18371
  source_screen_name : 15826
         source_type : 10484
         artist_name : 61298
           song_year : 25295
 ITC_song_id_log10_1 : 62032
    ITC_msno_log10_1 : 58561
        top2_in_song : 14297
   song_length_log10 : 42114
working on: ISCZ_genre_ids_log10

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float32
ITC_msno_log10_1         float32
top2_in_song            category
ISCZ_genre_ids_log10     float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISCZ_genre_ids_log10',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794077	valid_1's auc: 0.663066
[20]	training's auc: 0.801852	valid_1's auc: 0.666009
[30]	training's auc: 0.807144	valid_1's auc: 0.668196
[40]	training's auc: 0.811544	valid_1's auc: 0.669726
[50]	training's auc: 0.816756	valid_1's auc: 0.671526
[60]	training's auc: 0.821324	valid_1's auc: 0.673097
[70]	training's auc: 0.82522	valid_1's auc: 0.674433
[80]	training's auc: 0.827593	valid_1's auc: 0.675225
[90]	training's auc: 0.830237	valid_1's auc: 0.676246
[100]	training's auc: 0.832637	valid_1's auc: 0.677255
[110]	training's auc: 0.834636	valid_1's auc: 0.678173
[120]	training's auc: 0.836292	valid_1's auc: 0.679036
[130]	training's auc: 0.837706	valid_1's auc: 0.679682
[140]	training's auc: 0.838828	valid_1's auc: 0.680289
[150]	training's auc: 0.839764	valid_1's auc: 0.680723
[160]	training's auc: 0.840738	valid_1's auc: 0.681158
[170]	training's auc: 0.841667	valid_1's auc: 0.681552
[180]	training's auc: 0.842528	valid_1's auc: 0.681894
[190]	training's auc: 0.843332	valid_1's auc: 0.682209
[200]	training's auc: 0.844163	valid_1's auc: 0.682492
[210]	training's auc: 0.845023	valid_1's auc: 0.682804
[220]	training's auc: 0.845751	valid_1's auc: 0.683061
[230]	training's auc: 0.846509	valid_1's auc: 0.683296
[240]	training's auc: 0.847252	valid_1's auc: 0.683487
[250]	training's auc: 0.847975	valid_1's auc: 0.683699
[260]	training's auc: 0.848687	valid_1's auc: 0.683878
[270]	training's auc: 0.849382	valid_1's auc: 0.684068
[280]	training's auc: 0.850062	valid_1's auc: 0.68424
[290]	training's auc: 0.850672	valid_1's auc: 0.684362
[300]	training's auc: 0.851359	valid_1's auc: 0.684508
[310]	training's auc: 0.851914	valid_1's auc: 0.684623
[320]	training's auc: 0.852498	valid_1's auc: 0.684728
[330]	training's auc: 0.853056	valid_1's auc: 0.684832
[340]	training's auc: 0.853589	valid_1's auc: 0.684904
[350]	training's auc: 0.854218	valid_1's auc: 0.685043
[360]	training's auc: 0.854795	valid_1's auc: 0.685123
[370]	training's auc: 0.855341	valid_1's auc: 0.685223
[380]	training's auc: 0.855881	valid_1's auc: 0.685317
[390]	training's auc: 0.856397	valid_1's auc: 0.685371
[400]	training's auc: 0.856905	valid_1's auc: 0.685444
[410]	training's auc: 0.857413	valid_1's auc: 0.685495
[420]	training's auc: 0.857855	valid_1's auc: 0.685551
[430]	training's auc: 0.858348	valid_1's auc: 0.685624
[440]	training's auc: 0.858797	valid_1's auc: 0.685691
[450]	training's auc: 0.859294	valid_1's auc: 0.685818
[460]	training's auc: 0.859718	valid_1's auc: 0.685884
[470]	training's auc: 0.860152	valid_1's auc: 0.685927
[480]	training's auc: 0.86058	valid_1's auc: 0.686006
[490]	training's auc: 0.861026	valid_1's auc: 0.686042
[500]	training's auc: 0.861421	valid_1's auc: 0.686114
complete on: ISCZ_genre_ids_log10
model:
best score: 0.686114428266
best iteration: 0

                msno : 65931
             song_id : 18275
  source_screen_name : 16400
         source_type : 11319
         artist_name : 62146
           song_year : 27210
 ITC_song_id_log10_1 : 71437
    ITC_msno_log10_1 : 65510
        top2_in_song : 14194
ISCZ_genre_ids_log10 : 22078
working on: ISC_artist_name_log10

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                      uint8
msno                     category
song_id                  category
source_screen_name       category
source_type              category
artist_name              category
song_year                category
ITC_song_id_log10_1       float32
ITC_msno_log10_1          float32
top2_in_song             category
ISC_artist_name_log10     float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_artist_name_log10',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.79446	valid_1's auc: 0.663656
[20]	training's auc: 0.802062	valid_1's auc: 0.666315
[30]	training's auc: 0.807157	valid_1's auc: 0.668477
[40]	training's auc: 0.811486	valid_1's auc: 0.669942
[50]	training's auc: 0.816777	valid_1's auc: 0.671531
[60]	training's auc: 0.821264	valid_1's auc: 0.673009
[70]	training's auc: 0.825108	valid_1's auc: 0.674247
[80]	training's auc: 0.827427	valid_1's auc: 0.675033
[90]	training's auc: 0.830016	valid_1's auc: 0.676058
[100]	training's auc: 0.832395	valid_1's auc: 0.677056
[110]	training's auc: 0.834467	valid_1's auc: 0.67795
[120]	training's auc: 0.836177	valid_1's auc: 0.678819
[130]	training's auc: 0.837544	valid_1's auc: 0.67949
[140]	training's auc: 0.838775	valid_1's auc: 0.680079
[150]	training's auc: 0.839718	valid_1's auc: 0.680485
[160]	training's auc: 0.840766	valid_1's auc: 0.680904
[170]	training's auc: 0.841625	valid_1's auc: 0.68127
[180]	training's auc: 0.842585	valid_1's auc: 0.681684
[190]	training's auc: 0.843477	valid_1's auc: 0.682065
[200]	training's auc: 0.844399	valid_1's auc: 0.682405
[210]	training's auc: 0.845243	valid_1's auc: 0.682678
[220]	training's auc: 0.845988	valid_1's auc: 0.682897
[230]	training's auc: 0.846741	valid_1's auc: 0.683157
[240]	training's auc: 0.847495	valid_1's auc: 0.68336
[250]	training's auc: 0.848211	valid_1's auc: 0.683533
[260]	training's auc: 0.848935	valid_1's auc: 0.683721
[270]	training's auc: 0.849584	valid_1's auc: 0.683879
[280]	training's auc: 0.850342	valid_1's auc: 0.684112
[290]	training's auc: 0.850976	valid_1's auc: 0.684254
[300]	training's auc: 0.85154	valid_1's auc: 0.684376
[310]	training's auc: 0.852188	valid_1's auc: 0.684504
[320]	training's auc: 0.852811	valid_1's auc: 0.684621
[330]	training's auc: 0.853426	valid_1's auc: 0.684748
[340]	training's auc: 0.853977	valid_1's auc: 0.684877
[350]	training's auc: 0.854548	valid_1's auc: 0.684953
[360]	training's auc: 0.855152	valid_1's auc: 0.685057
[370]	training's auc: 0.855658	valid_1's auc: 0.685157
[380]	training's auc: 0.856233	valid_1's auc: 0.685254
[390]	training's auc: 0.856791	valid_1's auc: 0.685339
[400]	training's auc: 0.857389	valid_1's auc: 0.685436
[410]	training's auc: 0.857926	valid_1's auc: 0.685506
[420]	training's auc: 0.858378	valid_1's auc: 0.685571
[430]	training's auc: 0.858867	valid_1's auc: 0.685611
[440]	training's auc: 0.859347	valid_1's auc: 0.685663
[450]	training's auc: 0.859775	valid_1's auc: 0.685709
[460]	training's auc: 0.860247	valid_1's auc: 0.685757
[470]	training's auc: 0.860734	valid_1's auc: 0.685816
[480]	training's auc: 0.86122	valid_1's auc: 0.685896
[490]	training's auc: 0.861689	valid_1's auc: 0.685975
[500]	training's auc: 0.862114	valid_1's auc: 0.686019
complete on: ISC_artist_name_log10
model:
best score: 0.686019127851
best iteration: 0

                msno : 65933
             song_id : 18644
  source_screen_name : 15416
         source_type : 10455
         artist_name : 60345
           song_year : 25357
 ITC_song_id_log10_1 : 62474
    ITC_msno_log10_1 : 57983
        top2_in_song : 13896
ISC_artist_name_log10 : 43997
working on: ISCZ_composer_log10

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISCZ_composer_log10     float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISCZ_composer_log10',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794484	valid_1's auc: 0.663699
[20]	training's auc: 0.802173	valid_1's auc: 0.666469
[30]	training's auc: 0.807203	valid_1's auc: 0.668502
[40]	training's auc: 0.811758	valid_1's auc: 0.6701
[50]	training's auc: 0.816881	valid_1's auc: 0.671707
[60]	training's auc: 0.821342	valid_1's auc: 0.673246
[70]	training's auc: 0.824953	valid_1's auc: 0.674405
[80]	training's auc: 0.827224	valid_1's auc: 0.675219
[90]	training's auc: 0.829825	valid_1's auc: 0.676171
[100]	training's auc: 0.83242	valid_1's auc: 0.67725
[110]	training's auc: 0.834649	valid_1's auc: 0.678163
[120]	training's auc: 0.836317	valid_1's auc: 0.678872
[130]	training's auc: 0.837719	valid_1's auc: 0.679518
[140]	training's auc: 0.838816	valid_1's auc: 0.680025
[150]	training's auc: 0.839752	valid_1's auc: 0.680455
[160]	training's auc: 0.840747	valid_1's auc: 0.680896
[170]	training's auc: 0.841669	valid_1's auc: 0.681286
[180]	training's auc: 0.842561	valid_1's auc: 0.6816
[190]	training's auc: 0.843436	valid_1's auc: 0.681876
[200]	training's auc: 0.844278	valid_1's auc: 0.682174
[210]	training's auc: 0.845126	valid_1's auc: 0.68243
[220]	training's auc: 0.845898	valid_1's auc: 0.68269
[230]	training's auc: 0.846584	valid_1's auc: 0.682894
[240]	training's auc: 0.847374	valid_1's auc: 0.683137
[250]	training's auc: 0.848095	valid_1's auc: 0.683352
[260]	training's auc: 0.848829	valid_1's auc: 0.683543
[270]	training's auc: 0.849519	valid_1's auc: 0.68372
[280]	training's auc: 0.850139	valid_1's auc: 0.683851
[290]	training's auc: 0.85078	valid_1's auc: 0.68395
[300]	training's auc: 0.851379	valid_1's auc: 0.684076
[310]	training's auc: 0.851999	valid_1's auc: 0.684189
[320]	training's auc: 0.852572	valid_1's auc: 0.684297
[330]	training's auc: 0.85318	valid_1's auc: 0.684411
[340]	training's auc: 0.853752	valid_1's auc: 0.684512
[350]	training's auc: 0.854396	valid_1's auc: 0.684597
[360]	training's auc: 0.854968	valid_1's auc: 0.684692
[370]	training's auc: 0.855557	valid_1's auc: 0.684791
[380]	training's auc: 0.856081	valid_1's auc: 0.684869
[390]	training's auc: 0.856621	valid_1's auc: 0.684915
[400]	training's auc: 0.857144	valid_1's auc: 0.684985
[410]	training's auc: 0.857647	valid_1's auc: 0.685095
[420]	training's auc: 0.858116	valid_1's auc: 0.685163
[430]	training's auc: 0.858634	valid_1's auc: 0.685221
[440]	training's auc: 0.859086	valid_1's auc: 0.685261
[450]	training's auc: 0.859579	valid_1's auc: 0.685338
[460]	training's auc: 0.860025	valid_1's auc: 0.685381
[470]	training's auc: 0.860498	valid_1's auc: 0.685437
[480]	training's auc: 0.860919	valid_1's auc: 0.685479
[490]	training's auc: 0.86137	valid_1's auc: 0.685521
[500]	training's auc: 0.861811	valid_1's auc: 0.685566
complete on: ISCZ_composer_log10
model:
best score: 0.685565696392
best iteration: 0

                msno : 66168
             song_id : 18393
  source_screen_name : 16098
         source_type : 10799
         artist_name : 61453
           song_year : 26286
 ITC_song_id_log10_1 : 65352
    ITC_msno_log10_1 : 61282
        top2_in_song : 14568
 ISCZ_composer_log10 : 34101
working on: ISC_lyricist_log10

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_lyricist_log10      float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_lyricist_log10',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794183	valid_1's auc: 0.663063
[20]	training's auc: 0.802294	valid_1's auc: 0.666277
[30]	training's auc: 0.807382	valid_1's auc: 0.668202
[40]	training's auc: 0.811574	valid_1's auc: 0.669626
[50]	training's auc: 0.816825	valid_1's auc: 0.67136
[60]	training's auc: 0.821373	valid_1's auc: 0.672841
[70]	training's auc: 0.825101	valid_1's auc: 0.674085
[80]	training's auc: 0.827525	valid_1's auc: 0.674909
[90]	training's auc: 0.830166	valid_1's auc: 0.675946
[100]	training's auc: 0.832415	valid_1's auc: 0.676869
[110]	training's auc: 0.834692	valid_1's auc: 0.67782
[120]	training's auc: 0.836273	valid_1's auc: 0.678609
[130]	training's auc: 0.837556	valid_1's auc: 0.679223
[140]	training's auc: 0.838808	valid_1's auc: 0.679839
[150]	training's auc: 0.839772	valid_1's auc: 0.680299
[160]	training's auc: 0.840716	valid_1's auc: 0.680691
[170]	training's auc: 0.841726	valid_1's auc: 0.681073
[180]	training's auc: 0.842611	valid_1's auc: 0.681418
[190]	training's auc: 0.843573	valid_1's auc: 0.681776
[200]	training's auc: 0.844426	valid_1's auc: 0.682048
[210]	training's auc: 0.845276	valid_1's auc: 0.682342
[220]	training's auc: 0.846043	valid_1's auc: 0.682639
[230]	training's auc: 0.846707	valid_1's auc: 0.682818
[240]	training's auc: 0.847475	valid_1's auc: 0.683028
[250]	training's auc: 0.848179	valid_1's auc: 0.6832
[260]	training's auc: 0.848923	valid_1's auc: 0.683366
[270]	training's auc: 0.849596	valid_1's auc: 0.683529
[280]	training's auc: 0.850199	valid_1's auc: 0.683682
[290]	training's auc: 0.850885	valid_1's auc: 0.683832
[300]	training's auc: 0.851431	valid_1's auc: 0.683927
[310]	training's auc: 0.851989	valid_1's auc: 0.684053
[320]	training's auc: 0.852562	valid_1's auc: 0.684182
[330]	training's auc: 0.85314	valid_1's auc: 0.684292
[340]	training's auc: 0.853693	valid_1's auc: 0.68436
[350]	training's auc: 0.854307	valid_1's auc: 0.684486
[360]	training's auc: 0.854923	valid_1's auc: 0.684623
[370]	training's auc: 0.855447	valid_1's auc: 0.684741
[380]	training's auc: 0.855996	valid_1's auc: 0.684818
[390]	training's auc: 0.856559	valid_1's auc: 0.684892
[400]	training's auc: 0.857125	valid_1's auc: 0.684966
[410]	training's auc: 0.857607	valid_1's auc: 0.685078
[420]	training's auc: 0.858054	valid_1's auc: 0.685181
[430]	training's auc: 0.858547	valid_1's auc: 0.685246
[440]	training's auc: 0.859016	valid_1's auc: 0.685315
[450]	training's auc: 0.859533	valid_1's auc: 0.685429
[460]	training's auc: 0.859983	valid_1's auc: 0.685471
[470]	training's auc: 0.860451	valid_1's auc: 0.685511
[480]	training's auc: 0.860886	valid_1's auc: 0.685581
[490]	training's auc: 0.861351	valid_1's auc: 0.685646
[500]	training's auc: 0.861736	valid_1's auc: 0.685715
complete on: ISC_lyricist_log10
model:
best score: 0.685714531114
best iteration: 0

                msno : 66156
             song_id : 18404
  source_screen_name : 16305
         source_type : 11102
         artist_name : 61971
           song_year : 26723
 ITC_song_id_log10_1 : 68929
    ITC_msno_log10_1 : 64283
        top2_in_song : 15114
  ISC_lyricist_log10 : 25513
working on: ISC_song_country_ln

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
ISC_song_country_ln     float64
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ISC_song_country_ln',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794787	valid_1's auc: 0.66378
[20]	training's auc: 0.801865	valid_1's auc: 0.666399
[30]	training's auc: 0.807162	valid_1's auc: 0.668572
[40]	training's auc: 0.81154	valid_1's auc: 0.670109
[50]	training's auc: 0.816667	valid_1's auc: 0.671787
[60]	training's auc: 0.820912	valid_1's auc: 0.673114
[70]	training's auc: 0.824636	valid_1's auc: 0.674376
[80]	training's auc: 0.826903	valid_1's auc: 0.67515
[90]	training's auc: 0.82957	valid_1's auc: 0.676149
[100]	training's auc: 0.831947	valid_1's auc: 0.677237
[110]	training's auc: 0.834375	valid_1's auc: 0.678282
[120]	training's auc: 0.836041	valid_1's auc: 0.679119
[130]	training's auc: 0.837505	valid_1's auc: 0.679768
[140]	training's auc: 0.838729	valid_1's auc: 0.680395
[150]	training's auc: 0.839654	valid_1's auc: 0.680848
[160]	training's auc: 0.840677	valid_1's auc: 0.681321
[170]	training's auc: 0.841662	valid_1's auc: 0.681749
[180]	training's auc: 0.842517	valid_1's auc: 0.682078
[190]	training's auc: 0.843285	valid_1's auc: 0.682338
[200]	training's auc: 0.844237	valid_1's auc: 0.682732
[210]	training's auc: 0.845139	valid_1's auc: 0.682997
[220]	training's auc: 0.845936	valid_1's auc: 0.683262
[230]	training's auc: 0.846691	valid_1's auc: 0.683516
[240]	training's auc: 0.847427	valid_1's auc: 0.683738
[250]	training's auc: 0.848088	valid_1's auc: 0.683907
[260]	training's auc: 0.848784	valid_1's auc: 0.684079
[270]	training's auc: 0.849448	valid_1's auc: 0.684241
[280]	training's auc: 0.850132	valid_1's auc: 0.684434
[290]	training's auc: 0.85072	valid_1's auc: 0.684582
[300]	training's auc: 0.851295	valid_1's auc: 0.68472
[310]	training's auc: 0.851862	valid_1's auc: 0.684822
[320]	training's auc: 0.852535	valid_1's auc: 0.684975
[330]	training's auc: 0.853067	valid_1's auc: 0.685104
[340]	training's auc: 0.853623	valid_1's auc: 0.685218
[350]	training's auc: 0.854286	valid_1's auc: 0.685349
[360]	training's auc: 0.854857	valid_1's auc: 0.685441
[370]	training's auc: 0.855388	valid_1's auc: 0.685539
[380]	training's auc: 0.855965	valid_1's auc: 0.685642
[390]	training's auc: 0.856529	valid_1's auc: 0.685729
[400]	training's auc: 0.857043	valid_1's auc: 0.686011
[410]	training's auc: 0.857552	valid_1's auc: 0.686123
[420]	training's auc: 0.857986	valid_1's auc: 0.686318
[430]	training's auc: 0.858512	valid_1's auc: 0.686405
[440]	training's auc: 0.858952	valid_1's auc: 0.68647
[450]	training's auc: 0.859421	valid_1's auc: 0.686524
[460]	training's auc: 0.859896	valid_1's auc: 0.686583
[470]	training's auc: 0.860342	valid_1's auc: 0.686628
[480]	training's auc: 0.860808	valid_1's auc: 0.686674
[490]	training's auc: 0.861258	valid_1's auc: 0.686775
[500]	training's auc: 0.861681	valid_1's auc: 0.686822
complete on: ISC_song_country_ln
model:
best score: 0.686821705221
best iteration: 0

                msno : 66488
             song_id : 17977
  source_screen_name : 16358
         source_type : 11065
         artist_name : 61447
           song_year : 26382
 ITC_song_id_log10_1 : 70854
    ITC_msno_log10_1 : 66192
        top2_in_song : 14539
 ISC_song_country_ln : 23198
working on: ITC_source_system_tab_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                              uint8
msno                             category
song_id                          category
source_screen_name               category
source_type                      category
artist_name                      category
song_year                        category
ITC_song_id_log10_1               float32
ITC_msno_log10_1                  float32
top2_in_song                     category
ITC_source_system_tab_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'ITC_source_system_tab_log10_1',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.79518	valid_1's auc: 0.664378
[20]	training's auc: 0.802937	valid_1's auc: 0.66724
[30]	training's auc: 0.807879	valid_1's auc: 0.669255
[40]	training's auc: 0.812673	valid_1's auc: 0.670906
[50]	training's auc: 0.81801	valid_1's auc: 0.67274
[60]	training's auc: 0.822661	valid_1's auc: 0.674156
[70]	training's auc: 0.827113	valid_1's auc: 0.675611
[80]	training's auc: 0.829468	valid_1's auc: 0.676457
[90]	training's auc: 0.831888	valid_1's auc: 0.677339
[100]	training's auc: 0.834267	valid_1's auc: 0.678328
[110]	training's auc: 0.836212	valid_1's auc: 0.67911
[120]	training's auc: 0.837999	valid_1's auc: 0.680005
[130]	training's auc: 0.839306	valid_1's auc: 0.680629
[140]	training's auc: 0.840463	valid_1's auc: 0.681183
[150]	training's auc: 0.841469	valid_1's auc: 0.681637
[160]	training's auc: 0.842366	valid_1's auc: 0.682073
[170]	training's auc: 0.843302	valid_1's auc: 0.682448
[180]	training's auc: 0.844099	valid_1's auc: 0.682738
[190]	training's auc: 0.844945	valid_1's auc: 0.683032
[200]	training's auc: 0.845804	valid_1's auc: 0.683345
[210]	training's auc: 0.846614	valid_1's auc: 0.683611
[220]	training's auc: 0.847411	valid_1's auc: 0.683858
[230]	training's auc: 0.848084	valid_1's auc: 0.684051
[240]	training's auc: 0.848793	valid_1's auc: 0.684257
[250]	training's auc: 0.849549	valid_1's auc: 0.684526
[260]	training's auc: 0.850233	valid_1's auc: 0.68467
[270]	training's auc: 0.850858	valid_1's auc: 0.684814
[280]	training's auc: 0.851425	valid_1's auc: 0.684953
[290]	training's auc: 0.852106	valid_1's auc: 0.685115
[300]	training's auc: 0.852702	valid_1's auc: 0.685265
[310]	training's auc: 0.853304	valid_1's auc: 0.685385
[320]	training's auc: 0.853901	valid_1's auc: 0.685498
[330]	training's auc: 0.854487	valid_1's auc: 0.685573
[340]	training's auc: 0.855032	valid_1's auc: 0.685738
[350]	training's auc: 0.855616	valid_1's auc: 0.685829
[360]	training's auc: 0.856156	valid_1's auc: 0.685908
[370]	training's auc: 0.856751	valid_1's auc: 0.686014
[380]	training's auc: 0.857318	valid_1's auc: 0.6861
[390]	training's auc: 0.857881	valid_1's auc: 0.686189
[400]	training's auc: 0.858421	valid_1's auc: 0.686235
[410]	training's auc: 0.858961	valid_1's auc: 0.686305
[420]	training's auc: 0.859411	valid_1's auc: 0.686362
[430]	training's auc: 0.859895	valid_1's auc: 0.686414
[440]	training's auc: 0.860365	valid_1's auc: 0.686496
[450]	training's auc: 0.860829	valid_1's auc: 0.686576
[460]	training's auc: 0.861304	valid_1's auc: 0.686626
[470]	training's auc: 0.861741	valid_1's auc: 0.686672
[480]	training's auc: 0.862214	valid_1's auc: 0.68671
[490]	training's auc: 0.862633	valid_1's auc: 0.68674
[500]	training's auc: 0.863065	valid_1's auc: 0.686838
complete on: ITC_source_system_tab_log10_1
model:
best score: 0.686837933944
best iteration: 0

                msno : 66085
             song_id : 18662
  source_screen_name : 15380
         source_type : 10122
         artist_name : 62428
           song_year : 27263
 ITC_song_id_log10_1 : 74496
    ITC_msno_log10_1 : 66849
        top2_in_song : 15395
ITC_source_system_tab_log10_1 : 17820
working on: ITC_source_screen_name_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                               uint8
msno                              category
song_id                           category
source_screen_name                category
source_type                       category
artist_name                       category
song_year                         category
ITC_song_id_log10_1                float32
ITC_msno_log10_1                   float32
top2_in_song                      category
ITC_source_screen_name_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794112	valid_1's auc: 0.662885
[20]	training's auc: 0.801778	valid_1's auc: 0.666104
[30]	training's auc: 0.807418	valid_1's auc: 0.668177
[40]	training's auc: 0.811547	valid_1's auc: 0.669685
[50]	training's auc: 0.816779	valid_1's auc: 0.671386
[60]	training's auc: 0.821435	valid_1's auc: 0.672977
[70]	training's auc: 0.825189	valid_1's auc: 0.674194
[80]	training's auc: 0.827498	valid_1's auc: 0.674965
[90]	training's auc: 0.830133	valid_1's auc: 0.675984
[100]	training's auc: 0.832539	valid_1's auc: 0.677047
[110]	training's auc: 0.834866	valid_1's auc: 0.678016
[120]	training's auc: 0.836422	valid_1's auc: 0.678804
[130]	training's auc: 0.837883	valid_1's auc: 0.679445
[140]	training's auc: 0.839016	valid_1's auc: 0.679976
[150]	training's auc: 0.839897	valid_1's auc: 0.680413
[160]	training's auc: 0.84084	valid_1's auc: 0.680869
[170]	training's auc: 0.841836	valid_1's auc: 0.681269
[180]	training's auc: 0.84279	valid_1's auc: 0.681642
[190]	training's auc: 0.843583	valid_1's auc: 0.681967
[200]	training's auc: 0.844417	valid_1's auc: 0.682231
[210]	training's auc: 0.845343	valid_1's auc: 0.682507
[220]	training's auc: 0.846068	valid_1's auc: 0.682719
[230]	training's auc: 0.846843	valid_1's auc: 0.682969
[240]	training's auc: 0.847555	valid_1's auc: 0.683189
[250]	training's auc: 0.848209	valid_1's auc: 0.683343
[260]	training's auc: 0.84892	valid_1's auc: 0.683531
[270]	training's auc: 0.84955	valid_1's auc: 0.683667
[280]	training's auc: 0.850231	valid_1's auc: 0.683884
[290]	training's auc: 0.850922	valid_1's auc: 0.684044
[300]	training's auc: 0.851513	valid_1's auc: 0.684174
[310]	training's auc: 0.852083	valid_1's auc: 0.684294
[320]	training's auc: 0.852688	valid_1's auc: 0.684434
[330]	training's auc: 0.85324	valid_1's auc: 0.684578
[340]	training's auc: 0.853818	valid_1's auc: 0.684664
[350]	training's auc: 0.854425	valid_1's auc: 0.684823
[360]	training's auc: 0.854989	valid_1's auc: 0.684924
[370]	training's auc: 0.85551	valid_1's auc: 0.684996
[380]	training's auc: 0.856111	valid_1's auc: 0.685087
[390]	training's auc: 0.856636	valid_1's auc: 0.685153
[400]	training's auc: 0.857132	valid_1's auc: 0.685249
[410]	training's auc: 0.857641	valid_1's auc: 0.68538
[420]	training's auc: 0.858067	valid_1's auc: 0.685481
[430]	training's auc: 0.858545	valid_1's auc: 0.685522
[440]	training's auc: 0.859002	valid_1's auc: 0.685587
[450]	training's auc: 0.859469	valid_1's auc: 0.68566
[460]	training's auc: 0.859913	valid_1's auc: 0.685741
[470]	training's auc: 0.860344	valid_1's auc: 0.685796
[480]	training's auc: 0.860784	valid_1's auc: 0.685818
[490]	training's auc: 0.86123	valid_1's auc: 0.685864
[500]	training's auc: 0.861641	valid_1's auc: 0.685904
complete on: ITC_source_screen_name_log10_1
model:
best score: 0.685903865632
best iteration: 0

                msno : 66024
             song_id : 18105
  source_screen_name : 13789
         source_type : 10082
         artist_name : 62150
           song_year : 27024
 ITC_song_id_log10_1 : 71606
    ITC_msno_log10_1 : 65291
        top2_in_song : 15187
ITC_source_screen_name_log10_1 : 25242
working on: ITC_source_type_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                        uint8
msno                       category
song_id                    category
source_screen_name         category
source_type                category
artist_name                category
song_year                  category
ITC_song_id_log10_1         float32
ITC_msno_log10_1            float32
top2_in_song               category
ITC_source_type_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.793913	valid_1's auc: 0.663089
[20]	training's auc: 0.80186	valid_1's auc: 0.666024
[30]	training's auc: 0.806816	valid_1's auc: 0.668062
[40]	training's auc: 0.811366	valid_1's auc: 0.66961
[50]	training's auc: 0.816696	valid_1's auc: 0.671437
[60]	training's auc: 0.820951	valid_1's auc: 0.672839
[70]	training's auc: 0.824583	valid_1's auc: 0.673908
[80]	training's auc: 0.827071	valid_1's auc: 0.674731
[90]	training's auc: 0.829595	valid_1's auc: 0.675665
[100]	training's auc: 0.832043	valid_1's auc: 0.676688
[110]	training's auc: 0.834087	valid_1's auc: 0.677553
[120]	training's auc: 0.835754	valid_1's auc: 0.678368
[130]	training's auc: 0.83714	valid_1's auc: 0.679088
[140]	training's auc: 0.838267	valid_1's auc: 0.679593
[150]	training's auc: 0.8394	valid_1's auc: 0.680093
[160]	training's auc: 0.840337	valid_1's auc: 0.680517
[170]	training's auc: 0.841268	valid_1's auc: 0.680863
[180]	training's auc: 0.842221	valid_1's auc: 0.681242
[190]	training's auc: 0.843072	valid_1's auc: 0.681548
[200]	training's auc: 0.843969	valid_1's auc: 0.68183
[210]	training's auc: 0.844767	valid_1's auc: 0.682024
[220]	training's auc: 0.845493	valid_1's auc: 0.682248
[230]	training's auc: 0.846301	valid_1's auc: 0.682485
[240]	training's auc: 0.847104	valid_1's auc: 0.682723
[250]	training's auc: 0.847776	valid_1's auc: 0.682882
[260]	training's auc: 0.848599	valid_1's auc: 0.683134
[270]	training's auc: 0.849318	valid_1's auc: 0.683299
[280]	training's auc: 0.849953	valid_1's auc: 0.683474
[290]	training's auc: 0.850632	valid_1's auc: 0.683608
[300]	training's auc: 0.851168	valid_1's auc: 0.683723
[310]	training's auc: 0.851743	valid_1's auc: 0.683845
[320]	training's auc: 0.852342	valid_1's auc: 0.683967
[330]	training's auc: 0.852932	valid_1's auc: 0.684101
[340]	training's auc: 0.85349	valid_1's auc: 0.684187
[350]	training's auc: 0.854075	valid_1's auc: 0.684286
[360]	training's auc: 0.85469	valid_1's auc: 0.684497
[370]	training's auc: 0.85521	valid_1's auc: 0.68455
[380]	training's auc: 0.855738	valid_1's auc: 0.6846
[390]	training's auc: 0.856256	valid_1's auc: 0.684673
[400]	training's auc: 0.856785	valid_1's auc: 0.684786
[410]	training's auc: 0.857262	valid_1's auc: 0.684875
[420]	training's auc: 0.857725	valid_1's auc: 0.684934
[430]	training's auc: 0.858209	valid_1's auc: 0.684994
[440]	training's auc: 0.858667	valid_1's auc: 0.68504
[450]	training's auc: 0.859126	valid_1's auc: 0.68509
[460]	training's auc: 0.859579	valid_1's auc: 0.685132
[470]	training's auc: 0.860038	valid_1's auc: 0.68517
[480]	training's auc: 0.860498	valid_1's auc: 0.685233
[490]	training's auc: 0.86091	valid_1's auc: 0.685315
[500]	training's auc: 0.861339	valid_1's auc: 0.685366
complete on: ITC_source_type_log10_1
model:
best score: 0.685365536858
best iteration: 0

                msno : 66194
             song_id : 18152
  source_screen_name : 15314
         source_type : 8085
         artist_name : 61909
           song_year : 26933
 ITC_song_id_log10_1 : 72272
    ITC_msno_log10_1 : 65028
        top2_in_song : 15224
ITC_source_type_log10_1 : 25389
working on: ITC_artist_name_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                        uint8
msno                       category
song_id                    category
source_screen_name         category
source_type                category
artist_name                category
song_year                  category
ITC_song_id_log10_1         float32
ITC_msno_log10_1            float32
top2_in_song               category
ITC_artist_name_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794271	valid_1's auc: 0.662871
[20]	training's auc: 0.801935	valid_1's auc: 0.665668
[30]	training's auc: 0.80742	valid_1's auc: 0.667991
[40]	training's auc: 0.811692	valid_1's auc: 0.669598
[50]	training's auc: 0.817097	valid_1's auc: 0.671385
[60]	training's auc: 0.82151	valid_1's auc: 0.672897
[70]	training's auc: 0.825188	valid_1's auc: 0.674143
[80]	training's auc: 0.827527	valid_1's auc: 0.67496
[90]	training's auc: 0.830183	valid_1's auc: 0.676001
[100]	training's auc: 0.832411	valid_1's auc: 0.676966
[110]	training's auc: 0.834585	valid_1's auc: 0.677927
[120]	training's auc: 0.836262	valid_1's auc: 0.67877
[130]	training's auc: 0.83759	valid_1's auc: 0.679415
[140]	training's auc: 0.838725	valid_1's auc: 0.67994
[150]	training's auc: 0.839776	valid_1's auc: 0.6804
[160]	training's auc: 0.840864	valid_1's auc: 0.680867
[170]	training's auc: 0.841968	valid_1's auc: 0.681266
[180]	training's auc: 0.842828	valid_1's auc: 0.681577
[190]	training's auc: 0.843725	valid_1's auc: 0.681895
[200]	training's auc: 0.844619	valid_1's auc: 0.682254
[210]	training's auc: 0.845487	valid_1's auc: 0.682494
[220]	training's auc: 0.846236	valid_1's auc: 0.682718
[230]	training's auc: 0.847012	valid_1's auc: 0.682963
[240]	training's auc: 0.847843	valid_1's auc: 0.683216
[250]	training's auc: 0.848532	valid_1's auc: 0.683403
[260]	training's auc: 0.849373	valid_1's auc: 0.683651
[270]	training's auc: 0.850056	valid_1's auc: 0.6838
[280]	training's auc: 0.850746	valid_1's auc: 0.683992
[290]	training's auc: 0.851381	valid_1's auc: 0.684114
[300]	training's auc: 0.851946	valid_1's auc: 0.684221
[310]	training's auc: 0.852487	valid_1's auc: 0.684356
[320]	training's auc: 0.853118	valid_1's auc: 0.684483
[330]	training's auc: 0.853692	valid_1's auc: 0.684572
[340]	training's auc: 0.854272	valid_1's auc: 0.684662
[350]	training's auc: 0.854845	valid_1's auc: 0.684747
[360]	training's auc: 0.855445	valid_1's auc: 0.684842
[370]	training's auc: 0.855999	valid_1's auc: 0.68494
[380]	training's auc: 0.856532	valid_1's auc: 0.685004
[390]	training's auc: 0.857134	valid_1's auc: 0.685055
[400]	training's auc: 0.857685	valid_1's auc: 0.685129
[410]	training's auc: 0.858257	valid_1's auc: 0.685212
[420]	training's auc: 0.85872	valid_1's auc: 0.685313
[430]	training's auc: 0.859206	valid_1's auc: 0.685373
[440]	training's auc: 0.859664	valid_1's auc: 0.685415
[450]	training's auc: 0.86023	valid_1's auc: 0.685503
[460]	training's auc: 0.860748	valid_1's auc: 0.685558
[470]	training's auc: 0.86123	valid_1's auc: 0.685615
[480]	training's auc: 0.861718	valid_1's auc: 0.685712
[490]	training's auc: 0.862178	valid_1's auc: 0.685777
[500]	training's auc: 0.86259	valid_1's auc: 0.685907
complete on: ITC_artist_name_log10_1
model:
best score: 0.68590666245
best iteration: 0

                msno : 66405
             song_id : 19567
  source_screen_name : 15481
         source_type : 10644
         artist_name : 60860
           song_year : 25799
 ITC_song_id_log10_1 : 56593
    ITC_msno_log10_1 : 54658
        top2_in_song : 14196
ITC_artist_name_log10_1 : 50297
working on: ITC_composer_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float32
ITC_msno_log10_1         float32
top2_in_song            category
ITC_composer_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794408	valid_1's auc: 0.663289
[20]	training's auc: 0.802199	valid_1's auc: 0.666183
[30]	training's auc: 0.807331	valid_1's auc: 0.668162
[40]	training's auc: 0.811569	valid_1's auc: 0.669671
[50]	training's auc: 0.816746	valid_1's auc: 0.671321
[60]	training's auc: 0.821183	valid_1's auc: 0.672811
[70]	training's auc: 0.824868	valid_1's auc: 0.674022
[80]	training's auc: 0.827309	valid_1's auc: 0.674866
[90]	training's auc: 0.829811	valid_1's auc: 0.675849
[100]	training's auc: 0.832354	valid_1's auc: 0.676931
[110]	training's auc: 0.834319	valid_1's auc: 0.677811
[120]	training's auc: 0.835929	valid_1's auc: 0.67858
[130]	training's auc: 0.837294	valid_1's auc: 0.679271
[140]	training's auc: 0.83844	valid_1's auc: 0.679775
[150]	training's auc: 0.839503	valid_1's auc: 0.680285
[160]	training's auc: 0.840516	valid_1's auc: 0.680715
[170]	training's auc: 0.841475	valid_1's auc: 0.681136
[180]	training's auc: 0.842396	valid_1's auc: 0.681466
[190]	training's auc: 0.843297	valid_1's auc: 0.681816
[200]	training's auc: 0.844255	valid_1's auc: 0.682192
[210]	training's auc: 0.84509	valid_1's auc: 0.682447
[220]	training's auc: 0.845793	valid_1's auc: 0.682688
[230]	training's auc: 0.846583	valid_1's auc: 0.68291
[240]	training's auc: 0.847344	valid_1's auc: 0.683121
[250]	training's auc: 0.848016	valid_1's auc: 0.683325
[260]	training's auc: 0.848725	valid_1's auc: 0.683486
[270]	training's auc: 0.849413	valid_1's auc: 0.683632
[280]	training's auc: 0.850074	valid_1's auc: 0.683791
[290]	training's auc: 0.850714	valid_1's auc: 0.683964
[300]	training's auc: 0.851296	valid_1's auc: 0.684095
[310]	training's auc: 0.851833	valid_1's auc: 0.684174
[320]	training's auc: 0.852403	valid_1's auc: 0.684294
[330]	training's auc: 0.852981	valid_1's auc: 0.684367
[340]	training's auc: 0.853585	valid_1's auc: 0.684478
[350]	training's auc: 0.854217	valid_1's auc: 0.684607
[360]	training's auc: 0.854827	valid_1's auc: 0.684727
[370]	training's auc: 0.855362	valid_1's auc: 0.684847
[380]	training's auc: 0.85592	valid_1's auc: 0.684914
[390]	training's auc: 0.8565	valid_1's auc: 0.685008
[400]	training's auc: 0.857007	valid_1's auc: 0.685076
[410]	training's auc: 0.857554	valid_1's auc: 0.685261
[420]	training's auc: 0.858016	valid_1's auc: 0.685359
[430]	training's auc: 0.858485	valid_1's auc: 0.685419
[440]	training's auc: 0.858947	valid_1's auc: 0.685453
[450]	training's auc: 0.859394	valid_1's auc: 0.685512
[460]	training's auc: 0.859845	valid_1's auc: 0.685576
[470]	training's auc: 0.860345	valid_1's auc: 0.685651
[480]	training's auc: 0.860818	valid_1's auc: 0.68571
[490]	training's auc: 0.861242	valid_1's auc: 0.685742
[500]	training's auc: 0.861712	valid_1's auc: 0.685838
complete on: ITC_composer_log10_1
model:
best score: 0.685838446176
best iteration: 0

                msno : 65858
             song_id : 17835
  source_screen_name : 16073
         source_type : 10816
         artist_name : 60951
           song_year : 25717
 ITC_song_id_log10_1 : 63747
    ITC_msno_log10_1 : 60868
        top2_in_song : 14718
ITC_composer_log10_1 : 37917
working on: ITC_lyricist_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float32
ITC_msno_log10_1         float32
top2_in_song            category
ITC_lyricist_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.79425	valid_1's auc: 0.663598
[20]	training's auc: 0.802283	valid_1's auc: 0.666501
[30]	training's auc: 0.80723	valid_1's auc: 0.668357
[40]	training's auc: 0.811516	valid_1's auc: 0.669766
[50]	training's auc: 0.816789	valid_1's auc: 0.67144
[60]	training's auc: 0.82112	valid_1's auc: 0.67287
[70]	training's auc: 0.824734	valid_1's auc: 0.674079
[80]	training's auc: 0.827053	valid_1's auc: 0.674871
[90]	training's auc: 0.829502	valid_1's auc: 0.675799
[100]	training's auc: 0.832026	valid_1's auc: 0.676889
[110]	training's auc: 0.834163	valid_1's auc: 0.677799
[120]	training's auc: 0.835913	valid_1's auc: 0.678654
[130]	training's auc: 0.837297	valid_1's auc: 0.679302
[140]	training's auc: 0.838377	valid_1's auc: 0.67982
[150]	training's auc: 0.839517	valid_1's auc: 0.680397
[160]	training's auc: 0.840572	valid_1's auc: 0.680857
[170]	training's auc: 0.841524	valid_1's auc: 0.681228
[180]	training's auc: 0.842472	valid_1's auc: 0.681615
[190]	training's auc: 0.84328	valid_1's auc: 0.681891
[200]	training's auc: 0.844113	valid_1's auc: 0.682173
[210]	training's auc: 0.844982	valid_1's auc: 0.682461
[220]	training's auc: 0.84575	valid_1's auc: 0.682711
[230]	training's auc: 0.846572	valid_1's auc: 0.682999
[240]	training's auc: 0.847335	valid_1's auc: 0.683185
[250]	training's auc: 0.848023	valid_1's auc: 0.683337
[260]	training's auc: 0.848747	valid_1's auc: 0.683499
[270]	training's auc: 0.849445	valid_1's auc: 0.683682
[280]	training's auc: 0.85008	valid_1's auc: 0.683835
[290]	training's auc: 0.850765	valid_1's auc: 0.684049
[300]	training's auc: 0.851502	valid_1's auc: 0.684237
[310]	training's auc: 0.852081	valid_1's auc: 0.684354
[320]	training's auc: 0.85271	valid_1's auc: 0.684498
[330]	training's auc: 0.853305	valid_1's auc: 0.684613
[340]	training's auc: 0.853873	valid_1's auc: 0.684739
[350]	training's auc: 0.854454	valid_1's auc: 0.684856
[360]	training's auc: 0.855042	valid_1's auc: 0.684965
[370]	training's auc: 0.85563	valid_1's auc: 0.685116
[380]	training's auc: 0.856198	valid_1's auc: 0.6852
[390]	training's auc: 0.856752	valid_1's auc: 0.685318
[400]	training's auc: 0.857303	valid_1's auc: 0.685413
[410]	training's auc: 0.857802	valid_1's auc: 0.685483
[420]	training's auc: 0.858284	valid_1's auc: 0.685559
[430]	training's auc: 0.85879	valid_1's auc: 0.685637
[440]	training's auc: 0.859282	valid_1's auc: 0.685704
[450]	training's auc: 0.859742	valid_1's auc: 0.685753
[460]	training's auc: 0.860225	valid_1's auc: 0.685831
[470]	training's auc: 0.860701	valid_1's auc: 0.68588
[480]	training's auc: 0.861128	valid_1's auc: 0.685949
[490]	training's auc: 0.861596	valid_1's auc: 0.685997
[500]	training's auc: 0.862018	valid_1's auc: 0.686057
complete on: ITC_lyricist_log10_1
model:
best score: 0.686057425849
best iteration: 0

                msno : 66231
             song_id : 18000
  source_screen_name : 15843
         source_type : 10801
         artist_name : 61495
           song_year : 26369
 ITC_song_id_log10_1 : 66948
    ITC_msno_log10_1 : 62398
        top2_in_song : 14790
ITC_lyricist_log10_1 : 31625
working on: ITC_song_year_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                      uint8
msno                     category
song_id                  category
source_screen_name       category
source_type              category
artist_name              category
song_year                category
ITC_song_id_log10_1       float32
ITC_msno_log10_1          float32
top2_in_song             category
ITC_song_year_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794258	valid_1's auc: 0.663324
[20]	training's auc: 0.801942	valid_1's auc: 0.666226
[30]	training's auc: 0.807052	valid_1's auc: 0.66837
[40]	training's auc: 0.811406	valid_1's auc: 0.670005
[50]	training's auc: 0.816681	valid_1's auc: 0.671769
[60]	training's auc: 0.820946	valid_1's auc: 0.67322
[70]	training's auc: 0.824723	valid_1's auc: 0.674469
[80]	training's auc: 0.827093	valid_1's auc: 0.675186
[90]	training's auc: 0.829651	valid_1's auc: 0.676159
[100]	training's auc: 0.832199	valid_1's auc: 0.677257
[110]	training's auc: 0.83443	valid_1's auc: 0.678247
[120]	training's auc: 0.836109	valid_1's auc: 0.679099
[130]	training's auc: 0.837462	valid_1's auc: 0.679809
[140]	training's auc: 0.838612	valid_1's auc: 0.68034
[150]	training's auc: 0.839678	valid_1's auc: 0.680841
[160]	training's auc: 0.840639	valid_1's auc: 0.681272
[170]	training's auc: 0.841547	valid_1's auc: 0.681621
[180]	training's auc: 0.842608	valid_1's auc: 0.682057
[190]	training's auc: 0.84341	valid_1's auc: 0.682362
[200]	training's auc: 0.844327	valid_1's auc: 0.682671
[210]	training's auc: 0.845164	valid_1's auc: 0.682913
[220]	training's auc: 0.845872	valid_1's auc: 0.683155
[230]	training's auc: 0.846577	valid_1's auc: 0.683406
[240]	training's auc: 0.847281	valid_1's auc: 0.68357
[250]	training's auc: 0.848004	valid_1's auc: 0.683789
[260]	training's auc: 0.848678	valid_1's auc: 0.683925
[270]	training's auc: 0.84937	valid_1's auc: 0.684084
[280]	training's auc: 0.84998	valid_1's auc: 0.684228
[290]	training's auc: 0.850609	valid_1's auc: 0.68435
[300]	training's auc: 0.851218	valid_1's auc: 0.684506
[310]	training's auc: 0.851753	valid_1's auc: 0.684607
[320]	training's auc: 0.852389	valid_1's auc: 0.684717
[330]	training's auc: 0.853005	valid_1's auc: 0.684844
[340]	training's auc: 0.853578	valid_1's auc: 0.684953
[350]	training's auc: 0.854124	valid_1's auc: 0.68502
[360]	training's auc: 0.854711	valid_1's auc: 0.685124
[370]	training's auc: 0.855257	valid_1's auc: 0.68522
[380]	training's auc: 0.855807	valid_1's auc: 0.685294
[390]	training's auc: 0.85642	valid_1's auc: 0.685408
[400]	training's auc: 0.856934	valid_1's auc: 0.685478
[410]	training's auc: 0.857449	valid_1's auc: 0.68556
[420]	training's auc: 0.857879	valid_1's auc: 0.685628
[430]	training's auc: 0.858368	valid_1's auc: 0.685701
[440]	training's auc: 0.858816	valid_1's auc: 0.685731
[450]	training's auc: 0.8593	valid_1's auc: 0.685757
[460]	training's auc: 0.859722	valid_1's auc: 0.685803
[470]	training's auc: 0.86018	valid_1's auc: 0.685892
[480]	training's auc: 0.860608	valid_1's auc: 0.685968
[490]	training's auc: 0.861041	valid_1's auc: 0.68601
[500]	training's auc: 0.861472	valid_1's auc: 0.686071
complete on: ITC_song_year_log10_1
model:
best score: 0.68607086352
best iteration: 0

                msno : 65904
             song_id : 18375
  source_screen_name : 16443
         source_type : 10948
         artist_name : 62078
           song_year : 24809
 ITC_song_id_log10_1 : 69501
    ITC_msno_log10_1 : 65057
        top2_in_song : 15005
ITC_song_year_log10_1 : 26380
working on: ITC_top1_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float32
ITC_msno_log10_1             float32
top2_in_song                category
ITC_top1_in_song_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794052	valid_1's auc: 0.663446
[20]	training's auc: 0.801941	valid_1's auc: 0.666297
[30]	training's auc: 0.806928	valid_1's auc: 0.668295
[40]	training's auc: 0.811601	valid_1's auc: 0.669913
[50]	training's auc: 0.816812	valid_1's auc: 0.671647
[60]	training's auc: 0.821446	valid_1's auc: 0.673203
[70]	training's auc: 0.825051	valid_1's auc: 0.674475
[80]	training's auc: 0.8274	valid_1's auc: 0.675338
[90]	training's auc: 0.829896	valid_1's auc: 0.676328
[100]	training's auc: 0.832407	valid_1's auc: 0.67741
[110]	training's auc: 0.834489	valid_1's auc: 0.678308
[120]	training's auc: 0.835947	valid_1's auc: 0.679078
[130]	training's auc: 0.837301	valid_1's auc: 0.679703
[140]	training's auc: 0.838354	valid_1's auc: 0.68024
[150]	training's auc: 0.839447	valid_1's auc: 0.680748
[160]	training's auc: 0.840352	valid_1's auc: 0.68119
[170]	training's auc: 0.8413	valid_1's auc: 0.681517
[180]	training's auc: 0.842198	valid_1's auc: 0.681887
[190]	training's auc: 0.843099	valid_1's auc: 0.682194
[200]	training's auc: 0.843956	valid_1's auc: 0.68249
[210]	training's auc: 0.844833	valid_1's auc: 0.682744
[220]	training's auc: 0.845582	valid_1's auc: 0.682959
[230]	training's auc: 0.84628	valid_1's auc: 0.683183
[240]	training's auc: 0.847114	valid_1's auc: 0.683437
[250]	training's auc: 0.847858	valid_1's auc: 0.683587
[260]	training's auc: 0.848525	valid_1's auc: 0.683707
[270]	training's auc: 0.849267	valid_1's auc: 0.683914
[280]	training's auc: 0.849893	valid_1's auc: 0.684046
[290]	training's auc: 0.850545	valid_1's auc: 0.684184
[300]	training's auc: 0.851111	valid_1's auc: 0.684276
[310]	training's auc: 0.851723	valid_1's auc: 0.684368
[320]	training's auc: 0.852295	valid_1's auc: 0.68448
[330]	training's auc: 0.85291	valid_1's auc: 0.68463
[340]	training's auc: 0.853464	valid_1's auc: 0.684711
[350]	training's auc: 0.854018	valid_1's auc: 0.684795
[360]	training's auc: 0.854638	valid_1's auc: 0.684866
[370]	training's auc: 0.855156	valid_1's auc: 0.684985
[380]	training's auc: 0.855672	valid_1's auc: 0.685048
[390]	training's auc: 0.856245	valid_1's auc: 0.685154
[400]	training's auc: 0.856796	valid_1's auc: 0.685241
[410]	training's auc: 0.857266	valid_1's auc: 0.685295
[420]	training's auc: 0.857723	valid_1's auc: 0.685377
[430]	training's auc: 0.858231	valid_1's auc: 0.685443
[440]	training's auc: 0.858731	valid_1's auc: 0.685547
[450]	training's auc: 0.859214	valid_1's auc: 0.685602
[460]	training's auc: 0.859662	valid_1's auc: 0.685651
[470]	training's auc: 0.860113	valid_1's auc: 0.685695
[480]	training's auc: 0.860561	valid_1's auc: 0.685752
[490]	training's auc: 0.861002	valid_1's auc: 0.685777
[500]	training's auc: 0.861403	valid_1's auc: 0.685829
complete on: ITC_top1_in_song_log10_1
model:
best score: 0.685829250815
best iteration: 0

                msno : 66049
             song_id : 18099
  source_screen_name : 16394
         source_type : 11102
         artist_name : 62347
           song_year : 27093
 ITC_song_id_log10_1 : 71561
    ITC_msno_log10_1 : 65992
        top2_in_song : 14002
ITC_top1_in_song_log10_1 : 21861
working on: ITC_top2_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float32
ITC_msno_log10_1             float32
top2_in_song                category
ITC_top2_in_song_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.793651	valid_1's auc: 0.663194
[20]	training's auc: 0.801832	valid_1's auc: 0.666093
[30]	training's auc: 0.806894	valid_1's auc: 0.668208
[40]	training's auc: 0.811458	valid_1's auc: 0.66982
[50]	training's auc: 0.816972	valid_1's auc: 0.671684
[60]	training's auc: 0.821364	valid_1's auc: 0.673238
[70]	training's auc: 0.824957	valid_1's auc: 0.67444
[80]	training's auc: 0.827553	valid_1's auc: 0.67526
[90]	training's auc: 0.830138	valid_1's auc: 0.676281
[100]	training's auc: 0.832579	valid_1's auc: 0.677379
[110]	training's auc: 0.834565	valid_1's auc: 0.678264
[120]	training's auc: 0.836222	valid_1's auc: 0.679096
[130]	training's auc: 0.837627	valid_1's auc: 0.679758
[140]	training's auc: 0.838657	valid_1's auc: 0.680325
[150]	training's auc: 0.839677	valid_1's auc: 0.680839
[160]	training's auc: 0.840689	valid_1's auc: 0.681325
[170]	training's auc: 0.841754	valid_1's auc: 0.681724
[180]	training's auc: 0.842597	valid_1's auc: 0.682044
[190]	training's auc: 0.843432	valid_1's auc: 0.682367
[200]	training's auc: 0.844293	valid_1's auc: 0.682694
[210]	training's auc: 0.84517	valid_1's auc: 0.68297
[220]	training's auc: 0.845937	valid_1's auc: 0.683277
[230]	training's auc: 0.846643	valid_1's auc: 0.683518
[240]	training's auc: 0.847361	valid_1's auc: 0.68376
[250]	training's auc: 0.848023	valid_1's auc: 0.683929
[260]	training's auc: 0.84882	valid_1's auc: 0.6841
[270]	training's auc: 0.849495	valid_1's auc: 0.684291
[280]	training's auc: 0.850234	valid_1's auc: 0.684474
[290]	training's auc: 0.850833	valid_1's auc: 0.684628
[300]	training's auc: 0.851369	valid_1's auc: 0.684745
[310]	training's auc: 0.851951	valid_1's auc: 0.684911
[320]	training's auc: 0.852547	valid_1's auc: 0.685035
[330]	training's auc: 0.853112	valid_1's auc: 0.685134
[340]	training's auc: 0.853653	valid_1's auc: 0.685283
[350]	training's auc: 0.854208	valid_1's auc: 0.685371
[360]	training's auc: 0.85478	valid_1's auc: 0.685533
[370]	training's auc: 0.855291	valid_1's auc: 0.685606
[380]	training's auc: 0.855841	valid_1's auc: 0.685679
[390]	training's auc: 0.856366	valid_1's auc: 0.685752
[400]	training's auc: 0.856916	valid_1's auc: 0.685824
[410]	training's auc: 0.857411	valid_1's auc: 0.685939
[420]	training's auc: 0.857851	valid_1's auc: 0.686111
[430]	training's auc: 0.858335	valid_1's auc: 0.686172
[440]	training's auc: 0.858774	valid_1's auc: 0.686249
[450]	training's auc: 0.859225	valid_1's auc: 0.686373
[460]	training's auc: 0.859683	valid_1's auc: 0.686538
[470]	training's auc: 0.860161	valid_1's auc: 0.68659
[480]	training's auc: 0.860623	valid_1's auc: 0.686643
[490]	training's auc: 0.861054	valid_1's auc: 0.686716
[500]	training's auc: 0.861506	valid_1's auc: 0.686794
complete on: ITC_top2_in_song_log10_1
model:
best score: 0.686794075221
best iteration: 0

                msno : 66495
             song_id : 18667
  source_screen_name : 16426
         source_type : 11106
         artist_name : 62000
           song_year : 26946
 ITC_song_id_log10_1 : 71572
    ITC_msno_log10_1 : 65659
        top2_in_song : 13714
ITC_top2_in_song_log10_1 : 21915
working on: ITC_top3_in_song_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                         uint8
msno                        category
song_id                     category
source_screen_name          category
source_type                 category
artist_name                 category
song_year                   category
ITC_song_id_log10_1          float32
ITC_msno_log10_1             float32
top2_in_song                category
ITC_top3_in_song_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.794522	valid_1's auc: 0.663485
[20]	training's auc: 0.802007	valid_1's auc: 0.666144
[30]	training's auc: 0.807044	valid_1's auc: 0.668168
[40]	training's auc: 0.811385	valid_1's auc: 0.669754
[50]	training's auc: 0.816598	valid_1's auc: 0.671545
[60]	training's auc: 0.821065	valid_1's auc: 0.673027
[70]	training's auc: 0.824673	valid_1's auc: 0.674227
[80]	training's auc: 0.827141	valid_1's auc: 0.675028
[90]	training's auc: 0.829618	valid_1's auc: 0.675995
[100]	training's auc: 0.832094	valid_1's auc: 0.677008
[110]	training's auc: 0.834125	valid_1's auc: 0.677868
[120]	training's auc: 0.835842	valid_1's auc: 0.678711
[130]	training's auc: 0.837292	valid_1's auc: 0.679415
[140]	training's auc: 0.838378	valid_1's auc: 0.679913
[150]	training's auc: 0.839423	valid_1's auc: 0.680416
[160]	training's auc: 0.840487	valid_1's auc: 0.680875
[170]	training's auc: 0.841418	valid_1's auc: 0.681286
[180]	training's auc: 0.842203	valid_1's auc: 0.681638
[190]	training's auc: 0.843054	valid_1's auc: 0.681964
[200]	training's auc: 0.843954	valid_1's auc: 0.682257
[210]	training's auc: 0.844805	valid_1's auc: 0.682545
[220]	training's auc: 0.845556	valid_1's auc: 0.682754
[230]	training's auc: 0.846247	valid_1's auc: 0.682997
[240]	training's auc: 0.847017	valid_1's auc: 0.683223
[250]	training's auc: 0.847728	valid_1's auc: 0.683403
[260]	training's auc: 0.848431	valid_1's auc: 0.683566
[270]	training's auc: 0.84916	valid_1's auc: 0.683754
[280]	training's auc: 0.849815	valid_1's auc: 0.68395
[290]	training's auc: 0.850431	valid_1's auc: 0.684094
[300]	training's auc: 0.851012	valid_1's auc: 0.684199
[310]	training's auc: 0.85158	valid_1's auc: 0.684286
[320]	training's auc: 0.852167	valid_1's auc: 0.684431
[330]	training's auc: 0.852749	valid_1's auc: 0.684577
[340]	training's auc: 0.853343	valid_1's auc: 0.684666
[350]	training's auc: 0.853938	valid_1's auc: 0.684759
[360]	training's auc: 0.854555	valid_1's auc: 0.684865
[370]	training's auc: 0.855091	valid_1's auc: 0.684939
[380]	training's auc: 0.85565	valid_1's auc: 0.685033
[390]	training's auc: 0.856205	valid_1's auc: 0.685125
[400]	training's auc: 0.856727	valid_1's auc: 0.685199
[410]	training's auc: 0.85723	valid_1's auc: 0.68529
[420]	training's auc: 0.85768	valid_1's auc: 0.685358
[430]	training's auc: 0.858197	valid_1's auc: 0.685428
[440]	training's auc: 0.858654	valid_1's auc: 0.685489
[450]	training's auc: 0.859118	valid_1's auc: 0.685553
[460]	training's auc: 0.859551	valid_1's auc: 0.685579
[470]	training's auc: 0.860008	valid_1's auc: 0.685699
[480]	training's auc: 0.860484	valid_1's auc: 0.685779
[490]	training's auc: 0.86094	valid_1's auc: 0.68581
[500]	training's auc: 0.861386	valid_1's auc: 0.685908
complete on: ITC_top3_in_song_log10_1
model:
best score: 0.685907998485
best iteration: 0

                msno : 65874
             song_id : 17952
  source_screen_name : 16250
         source_type : 11040
         artist_name : 61936
           song_year : 26870
 ITC_song_id_log10_1 : 72147
    ITC_msno_log10_1 : 66339
        top2_in_song : 13831
ITC_top3_in_song_log10_1 : 22261
working on: OinC_msno

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
OinC_msno               float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'OinC_msno',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.820004	valid_1's auc: 0.641132
[20]	training's auc: 0.825129	valid_1's auc: 0.645715
[30]	training's auc: 0.828037	valid_1's auc: 0.644626
[40]	training's auc: 0.83121	valid_1's auc: 0.645062
[50]	training's auc: 0.83511	valid_1's auc: 0.646198
[60]	training's auc: 0.83846	valid_1's auc: 0.647159
[70]	training's auc: 0.841589	valid_1's auc: 0.648103
[80]	training's auc: 0.844061	valid_1's auc: 0.648831
[90]	training's auc: 0.846803	valid_1's auc: 0.649571
[100]	training's auc: 0.849091	valid_1's auc: 0.650222
[110]	training's auc: 0.851227	valid_1's auc: 0.650779
[120]	training's auc: 0.853157	valid_1's auc: 0.651235
[130]	training's auc: 0.854754	valid_1's auc: 0.651645
[140]	training's auc: 0.856092	valid_1's auc: 0.651882
[150]	training's auc: 0.857209	valid_1's auc: 0.65208
[160]	training's auc: 0.858129	valid_1's auc: 0.652235
[170]	training's auc: 0.858941	valid_1's auc: 0.652364
[180]	training's auc: 0.859627	valid_1's auc: 0.652481
[190]	training's auc: 0.8603	valid_1's auc: 0.652603
[200]	training's auc: 0.860866	valid_1's auc: 0.652721
[210]	training's auc: 0.861401	valid_1's auc: 0.652825
[220]	training's auc: 0.861906	valid_1's auc: 0.652871
[230]	training's auc: 0.862373	valid_1's auc: 0.652936
[240]	training's auc: 0.86289	valid_1's auc: 0.653
[250]	training's auc: 0.863342	valid_1's auc: 0.653095
[260]	training's auc: 0.863828	valid_1's auc: 0.653179
[270]	training's auc: 0.864264	valid_1's auc: 0.653219
[280]	training's auc: 0.86468	valid_1's auc: 0.653279
[290]	training's auc: 0.865102	valid_1's auc: 0.653348
[300]	training's auc: 0.865531	valid_1's auc: 0.653387
[310]	training's auc: 0.865942	valid_1's auc: 0.653466
[320]	training's auc: 0.866366	valid_1's auc: 0.653521
[330]	training's auc: 0.866761	valid_1's auc: 0.65355
[340]	training's auc: 0.86717	valid_1's auc: 0.653571
[350]	training's auc: 0.867612	valid_1's auc: 0.653631
[360]	training's auc: 0.868032	valid_1's auc: 0.653676
[370]	training's auc: 0.868428	valid_1's auc: 0.653726
[380]	training's auc: 0.868858	valid_1's auc: 0.653762
[390]	training's auc: 0.869324	valid_1's auc: 0.653799
[400]	training's auc: 0.869699	valid_1's auc: 0.653871
[410]	training's auc: 0.870152	valid_1's auc: 0.653928
[420]	training's auc: 0.870517	valid_1's auc: 0.653959
[430]	training's auc: 0.8709	valid_1's auc: 0.653985
[440]	training's auc: 0.871284	valid_1's auc: 0.654024
[450]	training's auc: 0.871634	valid_1's auc: 0.654053
[460]	training's auc: 0.871986	valid_1's auc: 0.65408
[470]	training's auc: 0.872357	valid_1's auc: 0.654113
[480]	training's auc: 0.872801	valid_1's auc: 0.654185
[490]	training's auc: 0.87315	valid_1's auc: 0.65421
[500]	training's auc: 0.873464	valid_1's auc: 0.654245
complete on: OinC_msno
model:
best score: 0.654245332705
best iteration: 0

                msno : 78234
             song_id : 27408
  source_screen_name : 10820
         source_type : 7357
         artist_name : 55880
           song_year : 20370
 ITC_song_id_log10_1 : 65545
    ITC_msno_log10_1 : 50593
        top2_in_song : 11067
           OinC_msno : 47226
working on: ITC_language_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                     uint8
msno                    category
song_id                 category
source_screen_name      category
source_type             category
artist_name             category
song_year               category
ITC_song_id_log10_1      float32
ITC_msno_log10_1         float32
top2_in_song            category
ITC_language_log10_1     float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
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
[10]	training's auc: 0.793945	valid_1's auc: 0.66296
[20]	training's auc: 0.801408	valid_1's auc: 0.665869
[30]	training's auc: 0.806974	valid_1's auc: 0.668072
[40]	training's auc: 0.811514	valid_1's auc: 0.669632
[50]	training's auc: 0.816725	valid_1's auc: 0.671396
[60]	training's auc: 0.821165	valid_1's auc: 0.672914
[70]	training's auc: 0.824844	valid_1's auc: 0.674144
[80]	training's auc: 0.827189	valid_1's auc: 0.674906
[90]	training's auc: 0.829585	valid_1's auc: 0.67589
[100]	training's auc: 0.832265	valid_1's auc: 0.67706
[110]	training's auc: 0.834573	valid_1's auc: 0.678006
[120]	training's auc: 0.836256	valid_1's auc: 0.678815
[130]	training's auc: 0.837669	valid_1's auc: 0.67951
[140]	training's auc: 0.838821	valid_1's auc: 0.680073
[150]	training's auc: 0.839822	valid_1's auc: 0.680551
[160]	training's auc: 0.840724	valid_1's auc: 0.680964
[170]	training's auc: 0.84163	valid_1's auc: 0.681317
[180]	training's auc: 0.842558	valid_1's auc: 0.68169
[190]	training's auc: 0.843396	valid_1's auc: 0.68204
[200]	training's auc: 0.844261	valid_1's auc: 0.682383
[210]	training's auc: 0.845068	valid_1's auc: 0.682596
[220]	training's auc: 0.845851	valid_1's auc: 0.682868
[230]	training's auc: 0.846586	valid_1's auc: 0.683117
[240]	training's auc: 0.84744	valid_1's auc: 0.683365
[250]	training's auc: 0.848092	valid_1's auc: 0.68352
[260]	training's auc: 0.848897	valid_1's auc: 0.68373
[270]	training's auc: 0.849603	valid_1's auc: 0.683915
[280]	training's auc: 0.850325	valid_1's auc: 0.684108
[290]	training's auc: 0.850938	valid_1's auc: 0.684259
[300]	training's auc: 0.851619	valid_1's auc: 0.684417
[310]	training's auc: 0.852194	valid_1's auc: 0.684518
[320]	training's auc: 0.852789	valid_1's auc: 0.684623
[330]	training's auc: 0.853303	valid_1's auc: 0.684765
[340]	training's auc: 0.853861	valid_1's auc: 0.684881
[350]	training's auc: 0.854441	valid_1's auc: 0.685006
[360]	training's auc: 0.85499	valid_1's auc: 0.685085
[370]	training's auc: 0.855566	valid_1's auc: 0.685196
[380]	training's auc: 0.856091	valid_1's auc: 0.685269
[390]	training's auc: 0.856676	valid_1's auc: 0.685381
[400]	training's auc: 0.857215	valid_1's auc: 0.685436
[410]	training's auc: 0.857668	valid_1's auc: 0.68551
[420]	training's auc: 0.858117	valid_1's auc: 0.685578
[430]	training's auc: 0.858668	valid_1's auc: 0.685728
[440]	training's auc: 0.85913	valid_1's auc: 0.685789
[450]	training's auc: 0.859572	valid_1's auc: 0.685848
[460]	training's auc: 0.860006	valid_1's auc: 0.685903
[470]	training's auc: 0.860426	valid_1's auc: 0.685948
[480]	training's auc: 0.860858	valid_1's auc: 0.686038
[490]	training's auc: 0.861292	valid_1's auc: 0.686093
[500]	training's auc: 0.86174	valid_1's auc: 0.686178
complete on: ITC_language_log10_1
model:
best score: 0.686177846472
best iteration: 0

                msno : 66524
             song_id : 18562
  source_screen_name : 16581
         source_type : 11102
         artist_name : 62321
           song_year : 27183
 ITC_song_id_log10_1 : 71407
    ITC_msno_log10_1 : 67300
        top2_in_song : 14597
ITC_language_log10_1 : 18923
working on: OinC_language

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
ITC_song_id_log10_1     float32
ITC_msno_log10_1        float32
top2_in_song           category
OinC_language           float32
dtype: object
number of rows: 7377418
number of columns: 11

'target',
'msno',
'song_id',
'source_screen_name',
'source_type',
'artist_name',
'song_year',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
'top2_in_song',
'OinC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.794897	valid_1's auc: 0.663633
[20]	training's auc: 0.802293	valid_1's auc: 0.666714
[30]	training's auc: 0.807387	valid_1's auc: 0.668607
[40]	training's auc: 0.811663	valid_1's auc: 0.670085
[50]	training's auc: 0.816832	valid_1's auc: 0.671785
[60]	training's auc: 0.821046	valid_1's auc: 0.673154
[70]	training's auc: 0.82496	valid_1's auc: 0.674482
[80]	training's auc: 0.827416	valid_1's auc: 0.675358
[90]	training's auc: 0.830075	valid_1's auc: 0.676366
[100]	training's auc: 0.832426	valid_1's auc: 0.677424
[110]	training's auc: 0.834346	valid_1's auc: 0.678273
[120]	training's auc: 0.836071	valid_1's auc: 0.679133
[130]	training's auc: 0.837452	valid_1's auc: 0.679847
[140]	training's auc: 0.838501	valid_1's auc: 0.680383
[150]	training's auc: 0.839488	valid_1's auc: 0.680836
[160]	training's auc: 0.840516	valid_1's auc: 0.681336
[170]	training's auc: 0.841518	valid_1's auc: 0.68173
[180]	training's auc: 0.842383	valid_1's auc: 0.682074
[190]	training's auc: 0.843287	valid_1's auc: 0.682446
[200]	training's auc: 0.844253	valid_1's auc: 0.682805
[210]	training's auc: 0.845096	valid_1's auc: 0.683089
[220]	training's auc: 0.845859	valid_1's auc: 0.683301
[230]	training's auc: 0.846597	valid_1's auc: 0.683552
[240]	training's auc: 0.847372	valid_1's auc: 0.683777
[250]	training's auc: 0.848091	valid_1's auc: 0.684003
[260]	training's auc: 0.848812	valid_1's auc: 0.684189
[270]	training's auc: 0.849603	valid_1's auc: 0.684395
[280]	training's auc: 0.850208	valid_1's auc: 0.684535
[290]	training's auc: 0.850808	valid_1's auc: 0.684663
[300]	training's auc: 0.851409	valid_1's auc: 0.684811
[310]	training's auc: 0.851979	valid_1's auc: 0.684925
[320]	training's auc: 0.852534	valid_1's auc: 0.684999
[330]	training's auc: 0.853097	valid_1's auc: 0.685113
[340]	training's auc: 0.853641	valid_1's auc: 0.685186
[350]	training's auc: 0.854268	valid_1's auc: 0.68529
[360]	training's auc: 0.854818	valid_1's auc: 0.685379
[370]	training's auc: 0.855356	valid_1's auc: 0.685458
[380]	training's auc: 0.855903	valid_1's auc: 0.685549
[390]	training's auc: 0.856489	valid_1's auc: 0.685668
[400]	training's auc: 0.856993	valid_1's auc: 0.685744
[410]	training's auc: 0.857592	valid_1's auc: 0.685844
[420]	training's auc: 0.858025	valid_1's auc: 0.685923
[430]	training's auc: 0.858498	valid_1's auc: 0.686005
[440]	training's auc: 0.858946	valid_1's auc: 0.686057
[450]	training's auc: 0.859407	valid_1's auc: 0.686131
[460]	training's auc: 0.859905	valid_1's auc: 0.686216
[470]	training's auc: 0.860371	valid_1's auc: 0.686267
[480]	training's auc: 0.860825	valid_1's auc: 0.686325
[490]	training's auc: 0.861246	valid_1's auc: 0.686372
[500]	training's auc: 0.861671	valid_1's auc: 0.686424
complete on: OinC_language
model:
best score: 0.686424495738
best iteration: 0

                msno : 66017
             song_id : 17930
  source_screen_name : 16691
         source_type : 11096
         artist_name : 62340
           song_year : 27384
 ITC_song_id_log10_1 : 72110
    ITC_msno_log10_1 : 67331
        top2_in_song : 14497
       OinC_language : 19104
                            OinC_msno:  0.654245332705
                             composer:  0.684823173065
              ITC_source_type_log10_1:  0.685365536858
                  ISCZ_composer_log10:  0.685565696392
                     ISC_top2_in_song:  0.685603515281
                    song_length_log10:  0.685632537696
                             lyricist:  0.685634683698
                     ISC_top1_in_song:  0.685672321673
                   ISC_lyricist_log10:  0.685714531114
             ITC_top1_in_song_log10_1:  0.685829250815
                 ITC_composer_log10_1:  0.685838446176
                                   rc:  0.685874608088
       ITC_source_screen_name_log10_1:  0.685903865632
              ITC_artist_name_log10_1:  0.68590666245
             ITC_top3_in_song_log10_1:  0.685907998485
                            genre_ids:  0.685972899984
                         top1_in_song:  0.685981388019
                ISC_artist_name_log10:  0.686019127851
                         top3_in_song:  0.686024160132
                 ITC_lyricist_log10_1:  0.686057425849
                ITC_song_year_log10_1:  0.68607086352
                         ISC_language:  0.68610324517
                     ISC_top3_in_song:  0.686109699775
                 ISCZ_genre_ids_log10:  0.686114428266
                       ISCZ_isrc_rest:  0.686135360636
                 ITC_language_log10_1:  0.686177846472
                         song_country:  0.686191389739
                        song_year_int:  0.686220067189
                              ISCZ_rc:  0.686290768709
                             language:  0.686351090659
                        OinC_language:  0.686424495738
                        ISC_song_year:  0.686452117745
                      membership_days:  0.686529285402
                    source_system_tab:  0.686617418696
             ITC_top2_in_song_log10_1:  0.686794075221
                  ISC_song_country_ln:  0.686821705221
        ITC_source_system_tab_log10_1:  0.686837933944

[timer]: complete in 972m 45s

Process finished with exit code 0
'''