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
    # 'artist_name',
    'top1_in_song',
    # 'top2_in_song',
    'top3_in_song',
    # 'language',
    'song_year',
    # 'composer',
    # 'lyricist',
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

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/VALIDATION/one_trainer_V1002B.py

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
ITC_top1_in_song                     int64
ITC_top3_in_song                     int64
ITC_song_year                        int64
ITC_source_screen_name               int64
ITC_source_type                      int64
ITC_msno_log10_1                   float16
ITC_song_id_log10_1                float16
ITC_top1_in_song_log10_1           float16
ITC_top3_in_song_log10_1           float16
ITC_song_year_log10_1              float16
ITC_source_screen_name_log10_1     float16
ITC_source_type_log10_1            float16
dtype: object
number of rows: 7377418
number of columns: 34

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
'ITC_top1_in_song',
'ITC_top3_in_song',
'ITC_song_year',
'ITC_source_screen_name',
'ITC_source_type',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_song_year_log10_1',
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
'ITC_top1_in_song',
'ITC_top3_in_song',
'ITC_song_year',
'ITC_source_screen_name',
'ITC_source_type',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_top1_in_song_log10_1',
'ITC_top3_in_song_log10_1',
'ITC_song_year_log10_1',
'ITC_source_screen_name_log10_1',
'ITC_source_type_log10_1',
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
ITC_song_id_log10_1     float16
ITC_msno_log10_1        float16
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
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.776682	valid_1's auc: 0.658345
[20]	training's auc: 0.782503	valid_1's auc: 0.660643
[30]	training's auc: 0.786413	valid_1's auc: 0.662401
[40]	training's auc: 0.78981	valid_1's auc: 0.663664
[50]	training's auc: 0.793892	valid_1's auc: 0.66494
[60]	training's auc: 0.797809	valid_1's auc: 0.666127
[70]	training's auc: 0.800931	valid_1's auc: 0.667199
[80]	training's auc: 0.802861	valid_1's auc: 0.667944
[90]	training's auc: 0.80537	valid_1's auc: 0.668889
[100]	training's auc: 0.807601	valid_1's auc: 0.669759
[110]	training's auc: 0.809478	valid_1's auc: 0.670519
[120]	training's auc: 0.811896	valid_1's auc: 0.67154
[130]	training's auc: 0.813737	valid_1's auc: 0.672245
[140]	training's auc: 0.815257	valid_1's auc: 0.673036
[150]	training's auc: 0.816957	valid_1's auc: 0.673842
[160]	training's auc: 0.818557	valid_1's auc: 0.674566
[170]	training's auc: 0.819766	valid_1's auc: 0.675157
[180]	training's auc: 0.820725	valid_1's auc: 0.675653
[190]	training's auc: 0.821543	valid_1's auc: 0.676082
[200]	training's auc: 0.822312	valid_1's auc: 0.676479
[210]	training's auc: 0.822982	valid_1's auc: 0.676781
[220]	training's auc: 0.82361	valid_1's auc: 0.6771
[230]	training's auc: 0.824166	valid_1's auc: 0.677333
[240]	training's auc: 0.82476	valid_1's auc: 0.67758
[250]	training's auc: 0.825389	valid_1's auc: 0.677853
[260]	training's auc: 0.825963	valid_1's auc: 0.678089
[270]	training's auc: 0.82656	valid_1's auc: 0.678328
[280]	training's auc: 0.827111	valid_1's auc: 0.678556
[290]	training's auc: 0.827738	valid_1's auc: 0.678799
[300]	training's auc: 0.828179	valid_1's auc: 0.678995
[310]	training's auc: 0.828713	valid_1's auc: 0.679195
[320]	training's auc: 0.829262	valid_1's auc: 0.679438
[330]	training's auc: 0.829845	valid_1's auc: 0.679658
[340]	training's auc: 0.830357	valid_1's auc: 0.67989
[350]	training's auc: 0.830828	valid_1's auc: 0.680071
[360]	training's auc: 0.831286	valid_1's auc: 0.680237
[370]	training's auc: 0.831764	valid_1's auc: 0.680387
[380]	training's auc: 0.832248	valid_1's auc: 0.680549
[390]	training's auc: 0.832711	valid_1's auc: 0.680736
[400]	training's auc: 0.833159	valid_1's auc: 0.680867
[410]	training's auc: 0.833627	valid_1's auc: 0.681035
[420]	training's auc: 0.834015	valid_1's auc: 0.681164
[430]	training's auc: 0.834422	valid_1's auc: 0.681277
[440]	training's auc: 0.834832	valid_1's auc: 0.681384
[450]	training's auc: 0.83523	valid_1's auc: 0.681518
[460]	training's auc: 0.835597	valid_1's auc: 0.681641
[470]	training's auc: 0.835981	valid_1's auc: 0.68175
[480]	training's auc: 0.836364	valid_1's auc: 0.681868
[490]	training's auc: 0.836756	valid_1's auc: 0.681974
[500]	training's auc: 0.837108	valid_1's auc: 0.682063
[510]	training's auc: 0.83747	valid_1's auc: 0.682152
[520]	training's auc: 0.837874	valid_1's auc: 0.682261
[530]	training's auc: 0.838245	valid_1's auc: 0.682346
[540]	training's auc: 0.8386	valid_1's auc: 0.682439
[550]	training's auc: 0.838958	valid_1's auc: 0.682609
[560]	training's auc: 0.839338	valid_1's auc: 0.682679
[570]	training's auc: 0.839722	valid_1's auc: 0.68276
[580]	training's auc: 0.840044	valid_1's auc: 0.682815
[590]	training's auc: 0.840343	valid_1's auc: 0.682887
[600]	training's auc: 0.840651	valid_1's auc: 0.682955
[610]	training's auc: 0.840961	valid_1's auc: 0.683008
[620]	training's auc: 0.841257	valid_1's auc: 0.683084
[630]	training's auc: 0.841645	valid_1's auc: 0.683197
[640]	training's auc: 0.841985	valid_1's auc: 0.683258
[650]	training's auc: 0.842331	valid_1's auc: 0.68335
[660]	training's auc: 0.842631	valid_1's auc: 0.683404
[670]	training's auc: 0.842941	valid_1's auc: 0.683473
[680]	training's auc: 0.843278	valid_1's auc: 0.683548
[690]	training's auc: 0.843584	valid_1's auc: 0.683597
[700]	training's auc: 0.843874	valid_1's auc: 0.683642
[710]	training's auc: 0.844195	valid_1's auc: 0.683687
[720]	training's auc: 0.844458	valid_1's auc: 0.683744
[730]	training's auc: 0.84472	valid_1's auc: 0.683881
[740]	training's auc: 0.844977	valid_1's auc: 0.68392
[750]	training's auc: 0.845264	valid_1's auc: 0.683981
[760]	training's auc: 0.84557	valid_1's auc: 0.684025
[770]	training's auc: 0.84583	valid_1's auc: 0.684066
[780]	training's auc: 0.846109	valid_1's auc: 0.684117
[790]	training's auc: 0.846382	valid_1's auc: 0.684153
[800]	training's auc: 0.846683	valid_1's auc: 0.684208
complete on: song_year_int
model:
best score: 0.684208441166
best iteration: 0

                msno : 60836
             song_id : 14630
  source_screen_name : 18976
         source_type : 14275
         artist_name : 65332
           song_year : 23676
 ITC_song_id_log10_1 : 77820
    ITC_msno_log10_1 : 80276
        top2_in_song : 15297
       song_year_int : 36882
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
ITC_song_id_log10_1       float16
ITC_msno_log10_1          float16
top2_in_song             category
ITC_song_year_log10_1     float16
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
[10]	training's auc: 0.776264	valid_1's auc: 0.658306
[20]	training's auc: 0.782366	valid_1's auc: 0.660712
[30]	training's auc: 0.786535	valid_1's auc: 0.662757
[40]	training's auc: 0.789748	valid_1's auc: 0.663861
[50]	training's auc: 0.794005	valid_1's auc: 0.665198
[60]	training's auc: 0.79757	valid_1's auc: 0.666422
[70]	training's auc: 0.800638	valid_1's auc: 0.667405
[80]	training's auc: 0.802645	valid_1's auc: 0.66809
[90]	training's auc: 0.804991	valid_1's auc: 0.668985
[100]	training's auc: 0.807249	valid_1's auc: 0.669941
[110]	training's auc: 0.80927	valid_1's auc: 0.670685
[120]	training's auc: 0.811395	valid_1's auc: 0.671611
[130]	training's auc: 0.813196	valid_1's auc: 0.672372
[140]	training's auc: 0.814782	valid_1's auc: 0.673063
[150]	training's auc: 0.816548	valid_1's auc: 0.673868
[160]	training's auc: 0.818161	valid_1's auc: 0.674578
[170]	training's auc: 0.819426	valid_1's auc: 0.675196
[180]	training's auc: 0.820464	valid_1's auc: 0.67574
[190]	training's auc: 0.82126	valid_1's auc: 0.676134
[200]	training's auc: 0.822049	valid_1's auc: 0.676538
[210]	training's auc: 0.822734	valid_1's auc: 0.676846
[220]	training's auc: 0.823323	valid_1's auc: 0.677114
[230]	training's auc: 0.823897	valid_1's auc: 0.677365
[240]	training's auc: 0.824553	valid_1's auc: 0.67767
[250]	training's auc: 0.825091	valid_1's auc: 0.677908
[260]	training's auc: 0.825632	valid_1's auc: 0.67813
[270]	training's auc: 0.826305	valid_1's auc: 0.678415
[280]	training's auc: 0.826887	valid_1's auc: 0.678677
[290]	training's auc: 0.827458	valid_1's auc: 0.678916
[300]	training's auc: 0.827964	valid_1's auc: 0.679143
[310]	training's auc: 0.828492	valid_1's auc: 0.679346
[320]	training's auc: 0.829033	valid_1's auc: 0.679564
[330]	training's auc: 0.829572	valid_1's auc: 0.679752
[340]	training's auc: 0.830065	valid_1's auc: 0.679899
[350]	training's auc: 0.83059	valid_1's auc: 0.68008
[360]	training's auc: 0.831068	valid_1's auc: 0.680242
[370]	training's auc: 0.831528	valid_1's auc: 0.680383
[380]	training's auc: 0.83204	valid_1's auc: 0.680552
[390]	training's auc: 0.832493	valid_1's auc: 0.680728
[400]	training's auc: 0.832943	valid_1's auc: 0.680874
[410]	training's auc: 0.833357	valid_1's auc: 0.681004
[420]	training's auc: 0.833798	valid_1's auc: 0.681127
[430]	training's auc: 0.834263	valid_1's auc: 0.681271
[440]	training's auc: 0.834658	valid_1's auc: 0.681414
[450]	training's auc: 0.835082	valid_1's auc: 0.681538
[460]	training's auc: 0.835491	valid_1's auc: 0.681644
[470]	training's auc: 0.8359	valid_1's auc: 0.681763
[480]	training's auc: 0.836243	valid_1's auc: 0.681857
[490]	training's auc: 0.836655	valid_1's auc: 0.681958
[500]	training's auc: 0.837006	valid_1's auc: 0.682052
[510]	training's auc: 0.83738	valid_1's auc: 0.682153
[520]	training's auc: 0.837757	valid_1's auc: 0.68225
[530]	training's auc: 0.838113	valid_1's auc: 0.682322
[540]	training's auc: 0.838423	valid_1's auc: 0.682391
[550]	training's auc: 0.838814	valid_1's auc: 0.682507
[560]	training's auc: 0.839206	valid_1's auc: 0.682584
[570]	training's auc: 0.839557	valid_1's auc: 0.68265
[580]	training's auc: 0.839901	valid_1's auc: 0.682728
[590]	training's auc: 0.840249	valid_1's auc: 0.682786
[600]	training's auc: 0.840583	valid_1's auc: 0.682831
[610]	training's auc: 0.840906	valid_1's auc: 0.682884
[620]	training's auc: 0.8412	valid_1's auc: 0.682929
[630]	training's auc: 0.841537	valid_1's auc: 0.682991
[640]	training's auc: 0.841839	valid_1's auc: 0.683045
[650]	training's auc: 0.842182	valid_1's auc: 0.683129
[660]	training's auc: 0.842482	valid_1's auc: 0.683193
[670]	training's auc: 0.842786	valid_1's auc: 0.683255
[680]	training's auc: 0.843088	valid_1's auc: 0.683291
[690]	training's auc: 0.843397	valid_1's auc: 0.683335
[700]	training's auc: 0.843721	valid_1's auc: 0.6834
[710]	training's auc: 0.844007	valid_1's auc: 0.68343
[720]	training's auc: 0.84427	valid_1's auc: 0.683482
[730]	training's auc: 0.844564	valid_1's auc: 0.683542
[740]	training's auc: 0.844864	valid_1's auc: 0.683583
[750]	training's auc: 0.845148	valid_1's auc: 0.683632
[760]	training's auc: 0.845465	valid_1's auc: 0.683695
[770]	training's auc: 0.845757	valid_1's auc: 0.683749
[780]	training's auc: 0.846018	valid_1's auc: 0.683792
[790]	training's auc: 0.846325	valid_1's auc: 0.683841
[800]	training's auc: 0.846586	valid_1's auc: 0.683882
complete on: ITC_song_year_log10_1
model:
best score: 0.683882408722
best iteration: 0

                msno : 60674
             song_id : 14638
  source_screen_name : 19022
         source_type : 14411
         artist_name : 65583
           song_year : 23737
 ITC_song_id_log10_1 : 77037
    ITC_msno_log10_1 : 80227
        top2_in_song : 15222
ITC_song_year_log10_1 : 37449
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
ITC_song_id_log10_1                float16
ITC_msno_log10_1                   float16
top2_in_song                      category
ITC_source_screen_name_log10_1     float16
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
[10]	training's auc: 0.776977	valid_1's auc: 0.658207
[20]	training's auc: 0.782757	valid_1's auc: 0.66079
[30]	training's auc: 0.786741	valid_1's auc: 0.662666
[40]	training's auc: 0.790026	valid_1's auc: 0.663878
[50]	training's auc: 0.794296	valid_1's auc: 0.665275
[60]	training's auc: 0.7977	valid_1's auc: 0.666377
[70]	training's auc: 0.800813	valid_1's auc: 0.667319
[80]	training's auc: 0.802828	valid_1's auc: 0.668058
[90]	training's auc: 0.805401	valid_1's auc: 0.668972
[100]	training's auc: 0.807523	valid_1's auc: 0.669766
[110]	training's auc: 0.809501	valid_1's auc: 0.670473
[120]	training's auc: 0.811632	valid_1's auc: 0.671453
[130]	training's auc: 0.813447	valid_1's auc: 0.67216
[140]	training's auc: 0.814973	valid_1's auc: 0.672863
[150]	training's auc: 0.81689	valid_1's auc: 0.673769
[160]	training's auc: 0.818341	valid_1's auc: 0.674422
[170]	training's auc: 0.819544	valid_1's auc: 0.674996
[180]	training's auc: 0.820446	valid_1's auc: 0.675496
[190]	training's auc: 0.821333	valid_1's auc: 0.675944
[200]	training's auc: 0.822047	valid_1's auc: 0.676334
[210]	training's auc: 0.822685	valid_1's auc: 0.676635
[220]	training's auc: 0.823315	valid_1's auc: 0.676954
[230]	training's auc: 0.823949	valid_1's auc: 0.677237
[240]	training's auc: 0.8246	valid_1's auc: 0.677557
[250]	training's auc: 0.825189	valid_1's auc: 0.677842
[260]	training's auc: 0.82581	valid_1's auc: 0.678108
[270]	training's auc: 0.826373	valid_1's auc: 0.678364
[280]	training's auc: 0.826937	valid_1's auc: 0.678609
[290]	training's auc: 0.827555	valid_1's auc: 0.678855
[300]	training's auc: 0.828095	valid_1's auc: 0.67909
[310]	training's auc: 0.828635	valid_1's auc: 0.67933
[320]	training's auc: 0.82915	valid_1's auc: 0.679552
[330]	training's auc: 0.829638	valid_1's auc: 0.679744
[340]	training's auc: 0.830151	valid_1's auc: 0.679933
[350]	training's auc: 0.830643	valid_1's auc: 0.680122
[360]	training's auc: 0.831095	valid_1's auc: 0.680281
[370]	training's auc: 0.831538	valid_1's auc: 0.68045
[380]	training's auc: 0.832009	valid_1's auc: 0.680606
[390]	training's auc: 0.83247	valid_1's auc: 0.680736
[400]	training's auc: 0.832924	valid_1's auc: 0.680862
[410]	training's auc: 0.833365	valid_1's auc: 0.681048
[420]	training's auc: 0.833738	valid_1's auc: 0.681171
[430]	training's auc: 0.834136	valid_1's auc: 0.681289
[440]	training's auc: 0.834539	valid_1's auc: 0.681401
[450]	training's auc: 0.834943	valid_1's auc: 0.68156
[460]	training's auc: 0.835337	valid_1's auc: 0.681669
[470]	training's auc: 0.835734	valid_1's auc: 0.681821
[480]	training's auc: 0.836074	valid_1's auc: 0.681915
[490]	training's auc: 0.836445	valid_1's auc: 0.682036
[500]	training's auc: 0.836798	valid_1's auc: 0.682134
[510]	training's auc: 0.837147	valid_1's auc: 0.68224
[520]	training's auc: 0.837527	valid_1's auc: 0.682322
[530]	training's auc: 0.837915	valid_1's auc: 0.682413
[540]	training's auc: 0.838283	valid_1's auc: 0.6825
[550]	training's auc: 0.838629	valid_1's auc: 0.682585
[560]	training's auc: 0.838979	valid_1's auc: 0.682644
[570]	training's auc: 0.839369	valid_1's auc: 0.682724
[580]	training's auc: 0.839739	valid_1's auc: 0.682815
[590]	training's auc: 0.840055	valid_1's auc: 0.68291
[600]	training's auc: 0.84043	valid_1's auc: 0.682985
[610]	training's auc: 0.840731	valid_1's auc: 0.683039
[620]	training's auc: 0.841018	valid_1's auc: 0.68309
[630]	training's auc: 0.84135	valid_1's auc: 0.683159
[640]	training's auc: 0.84165	valid_1's auc: 0.683211
[650]	training's auc: 0.84195	valid_1's auc: 0.683267
[660]	training's auc: 0.842274	valid_1's auc: 0.68333
[670]	training's auc: 0.842641	valid_1's auc: 0.683427
[680]	training's auc: 0.842936	valid_1's auc: 0.683482
[690]	training's auc: 0.843287	valid_1's auc: 0.683551
[700]	training's auc: 0.843609	valid_1's auc: 0.683611
[710]	training's auc: 0.843959	valid_1's auc: 0.683711
[720]	training's auc: 0.844219	valid_1's auc: 0.683748
[730]	training's auc: 0.844474	valid_1's auc: 0.683778
[740]	training's auc: 0.844758	valid_1's auc: 0.683806
[750]	training's auc: 0.845054	valid_1's auc: 0.68387
[760]	training's auc: 0.845364	valid_1's auc: 0.683927
[770]	training's auc: 0.845643	valid_1's auc: 0.683974
[780]	training's auc: 0.845922	valid_1's auc: 0.684021
[790]	training's auc: 0.846212	valid_1's auc: 0.684075
[800]	training's auc: 0.846505	valid_1's auc: 0.684128
complete on: ITC_source_screen_name_log10_1
model:
best score: 0.684127757883
best iteration: 0

                msno : 61023
             song_id : 14323
  source_screen_name : 15328
         source_type : 12571
         artist_name : 65802
           song_year : 26808
 ITC_song_id_log10_1 : 79477
    ITC_msno_log10_1 : 80628
        top2_in_song : 15828
ITC_source_screen_name_log10_1 : 36212
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
ITC_song_id_log10_1         float16
ITC_msno_log10_1            float16
top2_in_song               category
ITC_source_type_log10_1     float16
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
[10]	training's auc: 0.776557	valid_1's auc: 0.658288
[20]	training's auc: 0.782833	valid_1's auc: 0.660757
[30]	training's auc: 0.786597	valid_1's auc: 0.662483
[40]	training's auc: 0.789841	valid_1's auc: 0.663702
[50]	training's auc: 0.794248	valid_1's auc: 0.66508
[60]	training's auc: 0.797797	valid_1's auc: 0.666248
[70]	training's auc: 0.801025	valid_1's auc: 0.66724
[80]	training's auc: 0.802985	valid_1's auc: 0.66803
[90]	training's auc: 0.805269	valid_1's auc: 0.668978
[100]	training's auc: 0.807539	valid_1's auc: 0.669836
[110]	training's auc: 0.809628	valid_1's auc: 0.670623
[120]	training's auc: 0.811822	valid_1's auc: 0.671555
[130]	training's auc: 0.813596	valid_1's auc: 0.672348
[140]	training's auc: 0.815117	valid_1's auc: 0.673004
[150]	training's auc: 0.816915	valid_1's auc: 0.673816
[160]	training's auc: 0.81842	valid_1's auc: 0.674488
[170]	training's auc: 0.819641	valid_1's auc: 0.67509
[180]	training's auc: 0.820602	valid_1's auc: 0.675558
[190]	training's auc: 0.82144	valid_1's auc: 0.676003
[200]	training's auc: 0.822151	valid_1's auc: 0.676402
[210]	training's auc: 0.822899	valid_1's auc: 0.67675
[220]	training's auc: 0.823454	valid_1's auc: 0.677026
[230]	training's auc: 0.824099	valid_1's auc: 0.677353
[240]	training's auc: 0.824664	valid_1's auc: 0.677608
[250]	training's auc: 0.825261	valid_1's auc: 0.677875
[260]	training's auc: 0.825827	valid_1's auc: 0.678092
[270]	training's auc: 0.826464	valid_1's auc: 0.678368
[280]	training's auc: 0.826961	valid_1's auc: 0.678575
[290]	training's auc: 0.827588	valid_1's auc: 0.678845
[300]	training's auc: 0.828148	valid_1's auc: 0.679088
[310]	training's auc: 0.828654	valid_1's auc: 0.679281
[320]	training's auc: 0.829168	valid_1's auc: 0.679489
[330]	training's auc: 0.82971	valid_1's auc: 0.679679
[340]	training's auc: 0.830204	valid_1's auc: 0.679853
[350]	training's auc: 0.830672	valid_1's auc: 0.680012
[360]	training's auc: 0.831162	valid_1's auc: 0.680201
[370]	training's auc: 0.8316	valid_1's auc: 0.680351
[380]	training's auc: 0.832058	valid_1's auc: 0.680501
[390]	training's auc: 0.832558	valid_1's auc: 0.680699
[400]	training's auc: 0.832977	valid_1's auc: 0.680811
[410]	training's auc: 0.833412	valid_1's auc: 0.680947
[420]	training's auc: 0.833787	valid_1's auc: 0.681076
[430]	training's auc: 0.834262	valid_1's auc: 0.68123
[440]	training's auc: 0.834639	valid_1's auc: 0.681353
[450]	training's auc: 0.835051	valid_1's auc: 0.681472
[460]	training's auc: 0.835443	valid_1's auc: 0.681578
[470]	training's auc: 0.835831	valid_1's auc: 0.681675
[480]	training's auc: 0.836199	valid_1's auc: 0.681796
[490]	training's auc: 0.836586	valid_1's auc: 0.681908
[500]	training's auc: 0.836963	valid_1's auc: 0.682024
[510]	training's auc: 0.837358	valid_1's auc: 0.682139
[520]	training's auc: 0.837719	valid_1's auc: 0.682225
[530]	training's auc: 0.838135	valid_1's auc: 0.682346
[540]	training's auc: 0.838465	valid_1's auc: 0.682447
[550]	training's auc: 0.838803	valid_1's auc: 0.682525
[560]	training's auc: 0.839154	valid_1's auc: 0.682597
[570]	training's auc: 0.839499	valid_1's auc: 0.682637
[580]	training's auc: 0.839848	valid_1's auc: 0.682718
[590]	training's auc: 0.840161	valid_1's auc: 0.68282
[600]	training's auc: 0.840513	valid_1's auc: 0.682879
[610]	training's auc: 0.840838	valid_1's auc: 0.682949
[620]	training's auc: 0.841145	valid_1's auc: 0.683029
[630]	training's auc: 0.841481	valid_1's auc: 0.683105
[640]	training's auc: 0.841804	valid_1's auc: 0.683156
[650]	training's auc: 0.842102	valid_1's auc: 0.683233
[660]	training's auc: 0.842394	valid_1's auc: 0.683288
[670]	training's auc: 0.842825	valid_1's auc: 0.683383
[680]	training's auc: 0.843092	valid_1's auc: 0.683419
[690]	training's auc: 0.8434	valid_1's auc: 0.68345
[700]	training's auc: 0.843685	valid_1's auc: 0.683531
[710]	training's auc: 0.843997	valid_1's auc: 0.683597
[720]	training's auc: 0.844264	valid_1's auc: 0.683667
[730]	training's auc: 0.844535	valid_1's auc: 0.683715
[740]	training's auc: 0.844859	valid_1's auc: 0.683787
[750]	training's auc: 0.845131	valid_1's auc: 0.683817
[760]	training's auc: 0.845409	valid_1's auc: 0.683847
[770]	training's auc: 0.845692	valid_1's auc: 0.683898
[780]	training's auc: 0.84596	valid_1's auc: 0.683949
[790]	training's auc: 0.846257	valid_1's auc: 0.683998
[800]	training's auc: 0.846531	valid_1's auc: 0.684023
complete on: ITC_source_type_log10_1
model:
best score: 0.684022869229
best iteration: 0

                msno : 61116
             song_id : 14803
  source_screen_name : 17859
         source_type : 9939
         artist_name : 66363
           song_year : 26653
 ITC_song_id_log10_1 : 79091
    ITC_msno_log10_1 : 80216
        top2_in_song : 15763
ITC_source_type_log10_1 : 36197
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
ITC_song_id_log10_1          float16
ITC_msno_log10_1             float16
top2_in_song                category
ITC_top1_in_song_log10_1     float16
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
[10]	training's auc: 0.777033	valid_1's auc: 0.658513
[20]	training's auc: 0.782688	valid_1's auc: 0.661023
[30]	training's auc: 0.786428	valid_1's auc: 0.662743
[40]	training's auc: 0.790102	valid_1's auc: 0.664057
[50]	training's auc: 0.794267	valid_1's auc: 0.66536
[60]	training's auc: 0.798014	valid_1's auc: 0.666563
[70]	training's auc: 0.801039	valid_1's auc: 0.667556
[80]	training's auc: 0.803042	valid_1's auc: 0.668328
[90]	training's auc: 0.805486	valid_1's auc: 0.669251
[100]	training's auc: 0.807676	valid_1's auc: 0.670052
[110]	training's auc: 0.809588	valid_1's auc: 0.670789
[120]	training's auc: 0.812012	valid_1's auc: 0.67182
[130]	training's auc: 0.813721	valid_1's auc: 0.672514
[140]	training's auc: 0.815148	valid_1's auc: 0.67318
[150]	training's auc: 0.816977	valid_1's auc: 0.674023
[160]	training's auc: 0.818348	valid_1's auc: 0.674649
[170]	training's auc: 0.819629	valid_1's auc: 0.675245
[180]	training's auc: 0.820406	valid_1's auc: 0.675691
[190]	training's auc: 0.821159	valid_1's auc: 0.67612
[200]	training's auc: 0.821905	valid_1's auc: 0.676492
[210]	training's auc: 0.822557	valid_1's auc: 0.676812
[220]	training's auc: 0.823229	valid_1's auc: 0.677121
[230]	training's auc: 0.823778	valid_1's auc: 0.677392
[240]	training's auc: 0.824406	valid_1's auc: 0.677717
[250]	training's auc: 0.82501	valid_1's auc: 0.677949
[260]	training's auc: 0.825647	valid_1's auc: 0.678188
[270]	training's auc: 0.826212	valid_1's auc: 0.678409
[280]	training's auc: 0.826888	valid_1's auc: 0.678718
[290]	training's auc: 0.827486	valid_1's auc: 0.678944
[300]	training's auc: 0.827986	valid_1's auc: 0.679159
[310]	training's auc: 0.828553	valid_1's auc: 0.679389
[320]	training's auc: 0.829059	valid_1's auc: 0.679603
[330]	training's auc: 0.82952	valid_1's auc: 0.679784
[340]	training's auc: 0.830034	valid_1's auc: 0.679968
[350]	training's auc: 0.830573	valid_1's auc: 0.680161
[360]	training's auc: 0.831097	valid_1's auc: 0.680323
[370]	training's auc: 0.831626	valid_1's auc: 0.680511
[380]	training's auc: 0.832056	valid_1's auc: 0.68067
[390]	training's auc: 0.832529	valid_1's auc: 0.680854
[400]	training's auc: 0.833012	valid_1's auc: 0.681002
[410]	training's auc: 0.833412	valid_1's auc: 0.681156
[420]	training's auc: 0.833796	valid_1's auc: 0.681254
[430]	training's auc: 0.834255	valid_1's auc: 0.681402
[440]	training's auc: 0.834641	valid_1's auc: 0.681523
[450]	training's auc: 0.835072	valid_1's auc: 0.681647
[460]	training's auc: 0.835518	valid_1's auc: 0.681793
[470]	training's auc: 0.83591	valid_1's auc: 0.681909
[480]	training's auc: 0.836255	valid_1's auc: 0.682002
[490]	training's auc: 0.836646	valid_1's auc: 0.682098
[500]	training's auc: 0.836969	valid_1's auc: 0.682184
[510]	training's auc: 0.837294	valid_1's auc: 0.682272
[520]	training's auc: 0.837672	valid_1's auc: 0.68235
[530]	training's auc: 0.838072	valid_1's auc: 0.682478
[540]	training's auc: 0.838408	valid_1's auc: 0.68255
[550]	training's auc: 0.838774	valid_1's auc: 0.68265
[560]	training's auc: 0.839145	valid_1's auc: 0.682722
[570]	training's auc: 0.839515	valid_1's auc: 0.682815
[580]	training's auc: 0.839859	valid_1's auc: 0.682911
[590]	training's auc: 0.840167	valid_1's auc: 0.682996
[600]	training's auc: 0.840488	valid_1's auc: 0.683054
[610]	training's auc: 0.840814	valid_1's auc: 0.683125
[620]	training's auc: 0.841095	valid_1's auc: 0.683182
[630]	training's auc: 0.841437	valid_1's auc: 0.683245
[640]	training's auc: 0.841766	valid_1's auc: 0.683318
[650]	training's auc: 0.842061	valid_1's auc: 0.683378
[660]	training's auc: 0.84239	valid_1's auc: 0.683441
[670]	training's auc: 0.842736	valid_1's auc: 0.6835
[680]	training's auc: 0.843074	valid_1's auc: 0.683565
[690]	training's auc: 0.843379	valid_1's auc: 0.683616
[700]	training's auc: 0.843659	valid_1's auc: 0.683658
[710]	training's auc: 0.84396	valid_1's auc: 0.683709
[720]	training's auc: 0.844201	valid_1's auc: 0.683752
[730]	training's auc: 0.844481	valid_1's auc: 0.68381
[740]	training's auc: 0.844769	valid_1's auc: 0.683864
[750]	training's auc: 0.845068	valid_1's auc: 0.683922
[760]	training's auc: 0.845395	valid_1's auc: 0.683999
[770]	training's auc: 0.845675	valid_1's auc: 0.684063
[780]	training's auc: 0.845928	valid_1's auc: 0.684095
[790]	training's auc: 0.846252	valid_1's auc: 0.684151
[800]	training's auc: 0.846519	valid_1's auc: 0.684171
complete on: ITC_top1_in_song_log10_1
model:
best score: 0.684171255532
best iteration: 0

                msno : 60713
             song_id : 14693
  source_screen_name : 19541
         source_type : 14461
         artist_name : 65768
           song_year : 26370
 ITC_song_id_log10_1 : 80344
    ITC_msno_log10_1 : 81997
        top2_in_song : 14217
ITC_top1_in_song_log10_1 : 29896
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
ITC_song_id_log10_1          float16
ITC_msno_log10_1             float16
top2_in_song                category
ITC_top3_in_song_log10_1     float16
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
[10]	training's auc: 0.776516	valid_1's auc: 0.658395
[20]	training's auc: 0.782613	valid_1's auc: 0.661073
[30]	training's auc: 0.786409	valid_1's auc: 0.662806
[40]	training's auc: 0.789864	valid_1's auc: 0.66402
[50]	training's auc: 0.793979	valid_1's auc: 0.665316
[60]	training's auc: 0.797735	valid_1's auc: 0.666601
[70]	training's auc: 0.800853	valid_1's auc: 0.667511
[80]	training's auc: 0.802852	valid_1's auc: 0.668261
[90]	training's auc: 0.805258	valid_1's auc: 0.669212
[100]	training's auc: 0.807549	valid_1's auc: 0.670097
[110]	training's auc: 0.809424	valid_1's auc: 0.670775
[120]	training's auc: 0.811717	valid_1's auc: 0.671739
[130]	training's auc: 0.813438	valid_1's auc: 0.672456
[140]	training's auc: 0.815026	valid_1's auc: 0.673153
[150]	training's auc: 0.816756	valid_1's auc: 0.673988
[160]	training's auc: 0.818114	valid_1's auc: 0.674632
[170]	training's auc: 0.819427	valid_1's auc: 0.675249
[180]	training's auc: 0.820385	valid_1's auc: 0.675763
[190]	training's auc: 0.821217	valid_1's auc: 0.676207
[200]	training's auc: 0.821902	valid_1's auc: 0.676561
[210]	training's auc: 0.822587	valid_1's auc: 0.676907
[220]	training's auc: 0.823201	valid_1's auc: 0.677226
[230]	training's auc: 0.823791	valid_1's auc: 0.677532
[240]	training's auc: 0.824333	valid_1's auc: 0.677767
[250]	training's auc: 0.824995	valid_1's auc: 0.67808
[260]	training's auc: 0.825672	valid_1's auc: 0.678358
[270]	training's auc: 0.826218	valid_1's auc: 0.678599
[280]	training's auc: 0.82677	valid_1's auc: 0.678837
[290]	training's auc: 0.827374	valid_1's auc: 0.679102
[300]	training's auc: 0.827812	valid_1's auc: 0.679276
[310]	training's auc: 0.828373	valid_1's auc: 0.679534
[320]	training's auc: 0.82891	valid_1's auc: 0.679732
[330]	training's auc: 0.829418	valid_1's auc: 0.679924
[340]	training's auc: 0.82986	valid_1's auc: 0.680095
[350]	training's auc: 0.830345	valid_1's auc: 0.680269
[360]	training's auc: 0.830851	valid_1's auc: 0.680439
[370]	training's auc: 0.831319	valid_1's auc: 0.680584
[380]	training's auc: 0.831791	valid_1's auc: 0.680746
[390]	training's auc: 0.832201	valid_1's auc: 0.68088
[400]	training's auc: 0.832667	valid_1's auc: 0.681045
[410]	training's auc: 0.833168	valid_1's auc: 0.681243
[420]	training's auc: 0.833599	valid_1's auc: 0.681419
[430]	training's auc: 0.834019	valid_1's auc: 0.681542
[440]	training's auc: 0.834377	valid_1's auc: 0.681652
[450]	training's auc: 0.834745	valid_1's auc: 0.681765
[460]	training's auc: 0.83514	valid_1's auc: 0.681879
[470]	training's auc: 0.835534	valid_1's auc: 0.681979
[480]	training's auc: 0.83595	valid_1's auc: 0.682115
[490]	training's auc: 0.836349	valid_1's auc: 0.68223
[500]	training's auc: 0.836714	valid_1's auc: 0.682339
[510]	training's auc: 0.837088	valid_1's auc: 0.682418
[520]	training's auc: 0.837503	valid_1's auc: 0.682522
[530]	training's auc: 0.837867	valid_1's auc: 0.682624
[540]	training's auc: 0.8382	valid_1's auc: 0.682692
[550]	training's auc: 0.838596	valid_1's auc: 0.682807
[560]	training's auc: 0.838928	valid_1's auc: 0.682866
[570]	training's auc: 0.839307	valid_1's auc: 0.682951
[580]	training's auc: 0.839634	valid_1's auc: 0.683018
[590]	training's auc: 0.83997	valid_1's auc: 0.683096
[600]	training's auc: 0.840288	valid_1's auc: 0.683166
[610]	training's auc: 0.840597	valid_1's auc: 0.683223
[620]	training's auc: 0.840909	valid_1's auc: 0.683322
[630]	training's auc: 0.84127	valid_1's auc: 0.683426
[640]	training's auc: 0.841604	valid_1's auc: 0.683492
[650]	training's auc: 0.841902	valid_1's auc: 0.68355
[660]	training's auc: 0.842264	valid_1's auc: 0.683635
[670]	training's auc: 0.842602	valid_1's auc: 0.683692
[680]	training's auc: 0.84293	valid_1's auc: 0.683759
[690]	training's auc: 0.843213	valid_1's auc: 0.683802
[700]	training's auc: 0.843519	valid_1's auc: 0.683866
[710]	training's auc: 0.843847	valid_1's auc: 0.683921
[720]	training's auc: 0.844169	valid_1's auc: 0.683988
[730]	training's auc: 0.844457	valid_1's auc: 0.684057
[740]	training's auc: 0.844723	valid_1's auc: 0.684101
[750]	training's auc: 0.845035	valid_1's auc: 0.684167
[760]	training's auc: 0.845368	valid_1's auc: 0.684226
[770]	training's auc: 0.845638	valid_1's auc: 0.684276
[780]	training's auc: 0.845905	valid_1's auc: 0.684322
[790]	training's auc: 0.846188	valid_1's auc: 0.684371
[800]	training's auc: 0.846472	valid_1's auc: 0.684416
complete on: ITC_top3_in_song_log10_1
model:
best score: 0.684415770822
best iteration: 0

                msno : 60932
             song_id : 14643
  source_screen_name : 19422
         source_type : 14435
         artist_name : 66091
           song_year : 26471
 ITC_song_id_log10_1 : 79506
    ITC_msno_log10_1 : 81397
        top2_in_song : 14066
ITC_top3_in_song_log10_1 : 31037
                ITC_song_year_log10_1:  0.683882408722
              ITC_source_type_log10_1:  0.684022869229
       ITC_source_screen_name_log10_1:  0.684127757883
             ITC_top1_in_song_log10_1:  0.684171255532
                        song_year_int:  0.684208441166
             ITC_top3_in_song_log10_1:  0.684415770822

[timer]: complete in 186m 36s

Process finished with exit code 0
'''