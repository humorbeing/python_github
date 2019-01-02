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
load_name = 'train_me_play.csv'

df = read_df(load_name)

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


train, val = fake_df(df)
del df
model, cols = val_df(
    params, train, val,
    num_boost_round,
    early_stopping_rounds,
    verbose_eval,
    learning_rate=False
)
del train, val
show_mo(model)



print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION/me_trainer_V1001A.py

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
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.808884	valid_1's auc: 0.634578
[20]	training's auc: 0.813274	valid_1's auc: 0.637404
[30]	training's auc: 0.816354	valid_1's auc: 0.639114
[40]	training's auc: 0.819633	valid_1's auc: 0.641573
[50]	training's auc: 0.822256	valid_1's auc: 0.64239
[60]	training's auc: 0.824739	valid_1's auc: 0.644037
[70]	training's auc: 0.827175	valid_1's auc: 0.644965
[80]	training's auc: 0.829579	valid_1's auc: 0.6459
[90]	training's auc: 0.831782	valid_1's auc: 0.646857
[100]	training's auc: 0.833785	valid_1's auc: 0.647669
[110]	training's auc: 0.835835	valid_1's auc: 0.648407
[120]	training's auc: 0.837618	valid_1's auc: 0.649104
[130]	training's auc: 0.839312	valid_1's auc: 0.649726
[140]	training's auc: 0.840788	valid_1's auc: 0.650187
[150]	training's auc: 0.842191	valid_1's auc: 0.650636
[160]	training's auc: 0.84344	valid_1's auc: 0.650994
[170]	training's auc: 0.844647	valid_1's auc: 0.651356
[180]	training's auc: 0.845818	valid_1's auc: 0.651693
[190]	training's auc: 0.846806	valid_1's auc: 0.651981
[200]	training's auc: 0.847604	valid_1's auc: 0.652231
[210]	training's auc: 0.848448	valid_1's auc: 0.652393
[220]	training's auc: 0.849162	valid_1's auc: 0.652564
[230]	training's auc: 0.849775	valid_1's auc: 0.652662
[240]	training's auc: 0.850355	valid_1's auc: 0.652805
[250]	training's auc: 0.850809	valid_1's auc: 0.652892
[260]	training's auc: 0.851241	valid_1's auc: 0.652993
[270]	training's auc: 0.851627	valid_1's auc: 0.653062
[280]	training's auc: 0.852048	valid_1's auc: 0.653134
[290]	training's auc: 0.852418	valid_1's auc: 0.653207
[300]	training's auc: 0.852716	valid_1's auc: 0.653264
[310]	training's auc: 0.85303	valid_1's auc: 0.653331
[320]	training's auc: 0.853394	valid_1's auc: 0.653459
[330]	training's auc: 0.853736	valid_1's auc: 0.653516
[340]	training's auc: 0.854075	valid_1's auc: 0.653569
[350]	training's auc: 0.854384	valid_1's auc: 0.653676
[360]	training's auc: 0.854666	valid_1's auc: 0.653709
[370]	training's auc: 0.854922	valid_1's auc: 0.653744
[380]	training's auc: 0.85521	valid_1's auc: 0.653795
[390]	training's auc: 0.855484	valid_1's auc: 0.653846
[400]	training's auc: 0.855749	valid_1's auc: 0.653885
[410]	training's auc: 0.85601	valid_1's auc: 0.653924
[420]	training's auc: 0.856296	valid_1's auc: 0.653975
[430]	training's auc: 0.85654	valid_1's auc: 0.654028
[440]	training's auc: 0.856803	valid_1's auc: 0.654064
[450]	training's auc: 0.857053	valid_1's auc: 0.654089
[460]	training's auc: 0.857308	valid_1's auc: 0.654132
[470]	training's auc: 0.857566	valid_1's auc: 0.654162
[480]	training's auc: 0.85781	valid_1's auc: 0.654188
[490]	training's auc: 0.858082	valid_1's auc: 0.654211
[500]	training's auc: 0.858337	valid_1's auc: 0.654259
[510]	training's auc: 0.858564	valid_1's auc: 0.654282
[520]	training's auc: 0.858801	valid_1's auc: 0.654309
[530]	training's auc: 0.859052	valid_1's auc: 0.654329
[540]	training's auc: 0.859284	valid_1's auc: 0.654354
[550]	training's auc: 0.85952	valid_1's auc: 0.654371
[560]	training's auc: 0.859762	valid_1's auc: 0.654401
[570]	training's auc: 0.860008	valid_1's auc: 0.654437
[580]	training's auc: 0.860262	valid_1's auc: 0.654468
[590]	training's auc: 0.860501	valid_1's auc: 0.654506
[600]	training's auc: 0.86074	valid_1's auc: 0.654536
[610]	training's auc: 0.860991	valid_1's auc: 0.654567
[620]	training's auc: 0.861248	valid_1's auc: 0.654587
[630]	training's auc: 0.861493	valid_1's auc: 0.654605
[640]	training's auc: 0.861731	valid_1's auc: 0.65462
[650]	training's auc: 0.861957	valid_1's auc: 0.654647
[660]	training's auc: 0.862196	valid_1's auc: 0.654673
[670]	training's auc: 0.86242	valid_1's auc: 0.65469
[680]	training's auc: 0.862648	valid_1's auc: 0.654722
[690]	training's auc: 0.86288	valid_1's auc: 0.654747
[700]	training's auc: 0.863119	valid_1's auc: 0.654799
[710]	training's auc: 0.863339	valid_1's auc: 0.654818
[720]	training's auc: 0.863564	valid_1's auc: 0.654841
[730]	training's auc: 0.863797	valid_1's auc: 0.654865
[740]	training's auc: 0.86403	valid_1's auc: 0.654884
[750]	training's auc: 0.864239	valid_1's auc: 0.654894
[760]	training's auc: 0.86445	valid_1's auc: 0.654905
[770]	training's auc: 0.864669	valid_1's auc: 0.654924
[780]	training's auc: 0.864889	valid_1's auc: 0.654958
[790]	training's auc: 0.86514	valid_1's auc: 0.654988
[800]	training's auc: 0.865353	valid_1's auc: 0.655012
[810]	training's auc: 0.865562	valid_1's auc: 0.655026
[820]	training's auc: 0.865789	valid_1's auc: 0.655038
[830]	training's auc: 0.866025	valid_1's auc: 0.655061
[840]	training's auc: 0.866249	valid_1's auc: 0.655075
[850]	training's auc: 0.866475	valid_1's auc: 0.655094
[860]	training's auc: 0.866689	valid_1's auc: 0.655172
[870]	training's auc: 0.866913	valid_1's auc: 0.655184
[880]	training's auc: 0.867118	valid_1's auc: 0.655208
[890]	training's auc: 0.867322	valid_1's auc: 0.65523
[900]	training's auc: 0.867531	valid_1's auc: 0.655248
[910]	training's auc: 0.867742	valid_1's auc: 0.655276
[920]	training's auc: 0.86796	valid_1's auc: 0.655304
[930]	training's auc: 0.868172	valid_1's auc: 0.655335
[940]	training's auc: 0.86839	valid_1's auc: 0.655371
[950]	training's auc: 0.868586	valid_1's auc: 0.655385
[960]	training's auc: 0.868782	valid_1's auc: 0.655401
[970]	training's auc: 0.868983	valid_1's auc: 0.655413
[980]	training's auc: 0.869193	valid_1's auc: 0.655431
[990]	training's auc: 0.869425	valid_1's auc: 0.655474
[1000]	training's auc: 0.86962	valid_1's auc: 0.655492
[1010]	training's auc: 0.869837	valid_1's auc: 0.65551
[1020]	training's auc: 0.870036	valid_1's auc: 0.65552
[1030]	training's auc: 0.870249	valid_1's auc: 0.65554
[1040]	training's auc: 0.870446	valid_1's auc: 0.655568
[1050]	training's auc: 0.87065	valid_1's auc: 0.655589
[1060]	training's auc: 0.870841	valid_1's auc: 0.65561
[1070]	training's auc: 0.871044	valid_1's auc: 0.655676
[1080]	training's auc: 0.871232	valid_1's auc: 0.655689
[1090]	training's auc: 0.871421	valid_1's auc: 0.655705
[1100]	training's auc: 0.871612	valid_1's auc: 0.65572
[1110]	training's auc: 0.871804	valid_1's auc: 0.655728
[1120]	training's auc: 0.871988	valid_1's auc: 0.655742
[1130]	training's auc: 0.872166	valid_1's auc: 0.655748
[1140]	training's auc: 0.872366	valid_1's auc: 0.655757
[1150]	training's auc: 0.872548	valid_1's auc: 0.655773
[1160]	training's auc: 0.872733	valid_1's auc: 0.655776
[1170]	training's auc: 0.872936	valid_1's auc: 0.655796
[1180]	training's auc: 0.87313	valid_1's auc: 0.655815
[1190]	training's auc: 0.873314	valid_1's auc: 0.65585
[1200]	training's auc: 0.87348	valid_1's auc: 0.655867
[1210]	training's auc: 0.873676	valid_1's auc: 0.655897
[1220]	training's auc: 0.873847	valid_1's auc: 0.655907
[1230]	training's auc: 0.874014	valid_1's auc: 0.655914
[1240]	training's auc: 0.874193	valid_1's auc: 0.655939
[1250]	training's auc: 0.874388	valid_1's auc: 0.655958
[1260]	training's auc: 0.874579	valid_1's auc: 0.655979
[1270]	training's auc: 0.87477	valid_1's auc: 0.655993
[1280]	training's auc: 0.874961	valid_1's auc: 0.656013
[1290]	training's auc: 0.875139	valid_1's auc: 0.656024
[1300]	training's auc: 0.875336	valid_1's auc: 0.656049
[1310]	training's auc: 0.875539	valid_1's auc: 0.65606
[1320]	training's auc: 0.875713	valid_1's auc: 0.656082
[1330]	training's auc: 0.875887	valid_1's auc: 0.656087
[1340]	training's auc: 0.876048	valid_1's auc: 0.656105
[1350]	training's auc: 0.876217	valid_1's auc: 0.65611
[1360]	training's auc: 0.8764	valid_1's auc: 0.656133
[1370]	training's auc: 0.876567	valid_1's auc: 0.656142
[1380]	training's auc: 0.876748	valid_1's auc: 0.656148
[1390]	training's auc: 0.876923	valid_1's auc: 0.65616
[1400]	training's auc: 0.877081	valid_1's auc: 0.656171
[1410]	training's auc: 0.877279	valid_1's auc: 0.656183
[1420]	training's auc: 0.877443	valid_1's auc: 0.656191
[1430]	training's auc: 0.877601	valid_1's auc: 0.656197
[1440]	training's auc: 0.877764	valid_1's auc: 0.656196
[1450]	training's auc: 0.877929	valid_1's auc: 0.656216
[1460]	training's auc: 0.87809	valid_1's auc: 0.656221
[1470]	training's auc: 0.878251	valid_1's auc: 0.656226
[1480]	training's auc: 0.878416	valid_1's auc: 0.656232
[1490]	training's auc: 0.878583	valid_1's auc: 0.656241
[1500]	training's auc: 0.878742	valid_1's auc: 0.656255
[1510]	training's auc: 0.878896	valid_1's auc: 0.656267
[1520]	training's auc: 0.879061	valid_1's auc: 0.656265
[1530]	training's auc: 0.879214	valid_1's auc: 0.65627
[1540]	training's auc: 0.879375	valid_1's auc: 0.656289
[1550]	training's auc: 0.879537	valid_1's auc: 0.65629
[1560]	training's auc: 0.879687	valid_1's auc: 0.656295
[1570]	training's auc: 0.879833	valid_1's auc: 0.656303
[1580]	training's auc: 0.879995	valid_1's auc: 0.656318
[1590]	training's auc: 0.880148	valid_1's auc: 0.65632
[1600]	training's auc: 0.880312	valid_1's auc: 0.65633
[1610]	training's auc: 0.88047	valid_1's auc: 0.656346
[1620]	training's auc: 0.88063	valid_1's auc: 0.656369
[1630]	training's auc: 0.880798	valid_1's auc: 0.656389
[1640]	training's auc: 0.880957	valid_1's auc: 0.656397
[1650]	training's auc: 0.881108	valid_1's auc: 0.656418
[1660]	training's auc: 0.881244	valid_1's auc: 0.656424
[1670]	training's auc: 0.88139	valid_1's auc: 0.656428
[1680]	training's auc: 0.88154	valid_1's auc: 0.656441
[1690]	training's auc: 0.881692	valid_1's auc: 0.656453
[1700]	training's auc: 0.881834	valid_1's auc: 0.656464
[1710]	training's auc: 0.881979	valid_1's auc: 0.656473
[1720]	training's auc: 0.882122	valid_1's auc: 0.656481
[1730]	training's auc: 0.882257	valid_1's auc: 0.656488
[1740]	training's auc: 0.8824	valid_1's auc: 0.656495
[1750]	training's auc: 0.882541	valid_1's auc: 0.656499
[1760]	training's auc: 0.882684	valid_1's auc: 0.656516
[1770]	training's auc: 0.882837	valid_1's auc: 0.656528
[1780]	training's auc: 0.882978	valid_1's auc: 0.656533
[1790]	training's auc: 0.883111	valid_1's auc: 0.656544
[1800]	training's auc: 0.883255	valid_1's auc: 0.65656
[1810]	training's auc: 0.883404	valid_1's auc: 0.656571
[1820]	training's auc: 0.883541	valid_1's auc: 0.656591
[1830]	training's auc: 0.883683	valid_1's auc: 0.656607
[1840]	training's auc: 0.883825	valid_1's auc: 0.656614
[1850]	training's auc: 0.883965	valid_1's auc: 0.656631
[1860]	training's auc: 0.884093	valid_1's auc: 0.656641
[1870]	training's auc: 0.884232	valid_1's auc: 0.656649
[1880]	training's auc: 0.88436	valid_1's auc: 0.656655
[1890]	training's auc: 0.884496	valid_1's auc: 0.656663
[1900]	training's auc: 0.884625	valid_1's auc: 0.656671
[1910]	training's auc: 0.88476	valid_1's auc: 0.656676
[1920]	training's auc: 0.884891	valid_1's auc: 0.656673
[1930]	training's auc: 0.885025	valid_1's auc: 0.656682
[1940]	training's auc: 0.885152	valid_1's auc: 0.656686
[1950]	training's auc: 0.885283	valid_1's auc: 0.656689
[1960]	training's auc: 0.885409	valid_1's auc: 0.656698
[1970]	training's auc: 0.885543	valid_1's auc: 0.6567
[1980]	training's auc: 0.885668	valid_1's auc: 0.6567
[1990]	training's auc: 0.885789	valid_1's auc: 0.656708
[2000]	training's auc: 0.88592	valid_1's auc: 0.656719
model:
best score: 0.656718968103
best iteration: 0

                msno : 99035 1
             song_id : 31945
   source_system_tab : 2127
  source_screen_name : 18248
         source_type : 9926
           genre_ids : 5950
         artist_name : 89284 2
            composer : 18705
            lyricist : 11161
            language : 563
           song_year : 31452
        song_country : 3443
                  rc : 44158 7
        top1_in_song : 2671
        top2_in_song : 4385
        top3_in_song : 1598
     membership_days : 51948 6
       song_year_int : 15981
    ISC_top1_in_song : 6374
    ISC_top2_in_song : 5216
    ISC_top3_in_song : 3781
        ISC_language : 6645
             ISCZ_rc : 20254
      ISCZ_isrc_rest : 35440
       ISC_song_year : 12440
   song_length_log10 : 41825 8
ISCZ_genre_ids_log10 : 5845
ISC_artist_name_log10 : 30331
 ISCZ_composer_log10 : 25172
  ISC_lyricist_log10 : 15389
 ISC_song_country_ln : 8131
 ITC_song_id_log10_1 : 59745 4
ITC_source_system_tab_log10_1 : 13909
ITC_source_screen_name_log10_1 : 24552
ITC_source_type_log10_1 : 24428
ITC_artist_name_log10_1 : 42444 8
ITC_composer_log10_1 : 28302
ITC_lyricist_log10_1 : 18197
ITC_song_year_log10_1 : 4883
ITC_top1_in_song_log10_1 : 5372
ITC_top2_in_song_log10_1 : 4868
ITC_top3_in_song_log10_1 : 4568
    ITC_msno_log10_1 : 58848 5
           OinC_msno : 60798 3
ITC_language_log10_1 : 5800
       OinC_language : 3863

[timer]: complete in 86m 29s

Process finished with exit code 0
'''