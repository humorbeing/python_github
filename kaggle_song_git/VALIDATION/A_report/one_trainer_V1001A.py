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
cols = [
    'msno',
    'song_id',
    # 'artist_name',
    # 'top1_in_song',
    # 'top2_in_song',
    # 'top3_in_song',
    # 'language',
    # 'song_year',
]
df = add_ITC(df, cols)
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
def intme(x):
    return int(x)
df['song_year'] = df['song_year'].astype(object)
df['song_year'] = df['song_year'].apply(intme).astype(np.int16)
print(set(df['song_year']))
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
    'top2_in_song',
    # 'ITC_msno',
    # 'CC11_msno',
    # 'ITC_name',
    # 'language',
    # 'language',
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
'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/VALIDATION/one_trainer_V1001A.py

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
ITC_msno_log10_1        float16
ITC_song_id_log10_1     float16
dtype: object
number of rows: 7377418
number of columns: 23

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

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
{0, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017}
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
song_year                 int16
ITC_song_id_log10_1     float16
ITC_msno_log10_1        float16
top2_in_song           category
dtype: object
number of rows: 7377418
number of columns: 11

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

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.777494	valid_1's auc: 0.659904
[20]	training's auc: 0.783772	valid_1's auc: 0.662106
[30]	training's auc: 0.788919	valid_1's auc: 0.664197
[40]	training's auc: 0.792246	valid_1's auc: 0.665322
[50]	training's auc: 0.796292	valid_1's auc: 0.666632
[60]	training's auc: 0.799744	valid_1's auc: 0.667715
[70]	training's auc: 0.802967	valid_1's auc: 0.668736
[80]	training's auc: 0.805055	valid_1's auc: 0.669435
[90]	training's auc: 0.807619	valid_1's auc: 0.670413
[100]	training's auc: 0.809886	valid_1's auc: 0.671277
[110]	training's auc: 0.811851	valid_1's auc: 0.672013
[120]	training's auc: 0.813949	valid_1's auc: 0.672879
[130]	training's auc: 0.815856	valid_1's auc: 0.67375
[140]	training's auc: 0.817325	valid_1's auc: 0.674408
[150]	training's auc: 0.818996	valid_1's auc: 0.675232
[160]	training's auc: 0.820593	valid_1's auc: 0.675895
[170]	training's auc: 0.821937	valid_1's auc: 0.676506
[180]	training's auc: 0.822931	valid_1's auc: 0.676998
[190]	training's auc: 0.823721	valid_1's auc: 0.677453
[200]	training's auc: 0.824522	valid_1's auc: 0.677845
[210]	training's auc: 0.825129	valid_1's auc: 0.678142
[220]	training's auc: 0.825742	valid_1's auc: 0.678451
[230]	training's auc: 0.82634	valid_1's auc: 0.678742
[240]	training's auc: 0.826938	valid_1's auc: 0.679027
[250]	training's auc: 0.827517	valid_1's auc: 0.679255
[260]	training's auc: 0.828118	valid_1's auc: 0.679538
[270]	training's auc: 0.828622	valid_1's auc: 0.679751
[280]	training's auc: 0.829176	valid_1's auc: 0.68
[290]	training's auc: 0.829755	valid_1's auc: 0.680242
[300]	training's auc: 0.830311	valid_1's auc: 0.680503
[310]	training's auc: 0.83086	valid_1's auc: 0.680704
[320]	training's auc: 0.831315	valid_1's auc: 0.680863
[330]	training's auc: 0.83182	valid_1's auc: 0.681074
[340]	training's auc: 0.832325	valid_1's auc: 0.681293
[350]	training's auc: 0.832795	valid_1's auc: 0.681463
[360]	training's auc: 0.833217	valid_1's auc: 0.68164
[370]	training's auc: 0.833637	valid_1's auc: 0.681765
[380]	training's auc: 0.834083	valid_1's auc: 0.681924
[390]	training's auc: 0.834505	valid_1's auc: 0.682051
[400]	training's auc: 0.834937	valid_1's auc: 0.682187
[410]	training's auc: 0.83542	valid_1's auc: 0.682352
[420]	training's auc: 0.83582	valid_1's auc: 0.682488
[430]	training's auc: 0.836246	valid_1's auc: 0.682633
[440]	training's auc: 0.836648	valid_1's auc: 0.682755
[450]	training's auc: 0.836996	valid_1's auc: 0.682877
[460]	training's auc: 0.837353	valid_1's auc: 0.682987
[470]	training's auc: 0.837719	valid_1's auc: 0.683096
[480]	training's auc: 0.838095	valid_1's auc: 0.68319
[490]	training's auc: 0.838405	valid_1's auc: 0.683286
[500]	training's auc: 0.838775	valid_1's auc: 0.683392
[510]	training's auc: 0.839108	valid_1's auc: 0.683486
[520]	training's auc: 0.839451	valid_1's auc: 0.683562
[530]	training's auc: 0.839802	valid_1's auc: 0.683643
[540]	training's auc: 0.840177	valid_1's auc: 0.683758
[550]	training's auc: 0.840493	valid_1's auc: 0.683823
[560]	training's auc: 0.840816	valid_1's auc: 0.683887
[570]	training's auc: 0.841142	valid_1's auc: 0.683953
[580]	training's auc: 0.841461	valid_1's auc: 0.684033
[590]	training's auc: 0.841816	valid_1's auc: 0.684103
[600]	training's auc: 0.842159	valid_1's auc: 0.684208
[610]	training's auc: 0.842447	valid_1's auc: 0.684277
[620]	training's auc: 0.842801	valid_1's auc: 0.68437
[630]	training's auc: 0.843093	valid_1's auc: 0.684438
[640]	training's auc: 0.84337	valid_1's auc: 0.684493
[650]	training's auc: 0.84367	valid_1's auc: 0.684563
[660]	training's auc: 0.843937	valid_1's auc: 0.684604
[670]	training's auc: 0.844236	valid_1's auc: 0.684663
[680]	training's auc: 0.844523	valid_1's auc: 0.684728
[690]	training's auc: 0.844753	valid_1's auc: 0.684765
[700]	training's auc: 0.845012	valid_1's auc: 0.684829
[710]	training's auc: 0.845278	valid_1's auc: 0.684886
[720]	training's auc: 0.84555	valid_1's auc: 0.684933
[730]	training's auc: 0.845805	valid_1's auc: 0.68497
[740]	training's auc: 0.846077	valid_1's auc: 0.685017
[750]	training's auc: 0.846338	valid_1's auc: 0.685044
[760]	training's auc: 0.846604	valid_1's auc: 0.685084
[770]	training's auc: 0.84688	valid_1's auc: 0.685141
[780]	training's auc: 0.847148	valid_1's auc: 0.685183
[790]	training's auc: 0.847438	valid_1's auc: 0.685225
[800]	training's auc: 0.847733	valid_1's auc: 0.685284
[810]	training's auc: 0.847989	valid_1's auc: 0.685312
[820]	training's auc: 0.848215	valid_1's auc: 0.685339
[830]	training's auc: 0.848497	valid_1's auc: 0.685384
[840]	training's auc: 0.848783	valid_1's auc: 0.685434
[850]	training's auc: 0.849017	valid_1's auc: 0.685476
[860]	training's auc: 0.8493	valid_1's auc: 0.685544
[870]	training's auc: 0.849543	valid_1's auc: 0.685585
[880]	training's auc: 0.849763	valid_1's auc: 0.685609
[890]	training's auc: 0.850018	valid_1's auc: 0.685634
[900]	training's auc: 0.850274	valid_1's auc: 0.685682
[910]	training's auc: 0.850498	valid_1's auc: 0.685706
[920]	training's auc: 0.850701	valid_1's auc: 0.685746
[930]	training's auc: 0.850923	valid_1's auc: 0.685804
[940]	training's auc: 0.851112	valid_1's auc: 0.685829
[950]	training's auc: 0.851369	valid_1's auc: 0.685874
[960]	training's auc: 0.851599	valid_1's auc: 0.685894
[970]	training's auc: 0.851823	valid_1's auc: 0.685932
[980]	training's auc: 0.852046	valid_1's auc: 0.685965
[990]	training's auc: 0.852253	valid_1's auc: 0.686004
[1000]	training's auc: 0.852469	valid_1's auc: 0.686047
[1010]	training's auc: 0.85269	valid_1's auc: 0.686081
[1020]	training's auc: 0.85292	valid_1's auc: 0.686102
[1030]	training's auc: 0.853143	valid_1's auc: 0.686137
[1040]	training's auc: 0.853372	valid_1's auc: 0.686159
[1050]	training's auc: 0.853601	valid_1's auc: 0.686201
[1060]	training's auc: 0.853818	valid_1's auc: 0.686375
[1070]	training's auc: 0.854002	valid_1's auc: 0.686403
[1080]	training's auc: 0.854207	valid_1's auc: 0.686429
[1090]	training's auc: 0.854413	valid_1's auc: 0.68645
[1100]	training's auc: 0.854609	valid_1's auc: 0.686484
[1110]	training's auc: 0.854796	valid_1's auc: 0.686506
[1120]	training's auc: 0.855022	valid_1's auc: 0.686535
[1130]	training's auc: 0.855251	valid_1's auc: 0.686592
[1140]	training's auc: 0.855449	valid_1's auc: 0.68661
[1150]	training's auc: 0.855613	valid_1's auc: 0.686624
[1160]	training's auc: 0.855825	valid_1's auc: 0.686676
[1170]	training's auc: 0.856007	valid_1's auc: 0.686704
[1180]	training's auc: 0.856187	valid_1's auc: 0.686725
[1190]	training's auc: 0.856359	valid_1's auc: 0.68675
[1200]	training's auc: 0.856559	valid_1's auc: 0.686777
[1210]	training's auc: 0.856779	valid_1's auc: 0.686807
[1220]	training's auc: 0.856979	valid_1's auc: 0.686844
[1230]	training's auc: 0.857222	valid_1's auc: 0.686903
[1240]	training's auc: 0.857443	valid_1's auc: 0.686929
[1250]	training's auc: 0.857662	valid_1's auc: 0.68695
[1260]	training's auc: 0.857839	valid_1's auc: 0.686977
[1270]	training's auc: 0.858073	valid_1's auc: 0.687016
[1280]	training's auc: 0.858317	valid_1's auc: 0.687051
[1290]	training's auc: 0.858516	valid_1's auc: 0.687079
[1300]	training's auc: 0.858767	valid_1's auc: 0.687114
[1310]	training's auc: 0.858945	valid_1's auc: 0.68719
[1320]	training's auc: 0.85911	valid_1's auc: 0.687193
[1330]	training's auc: 0.859298	valid_1's auc: 0.687214
[1340]	training's auc: 0.859487	valid_1's auc: 0.687238
[1350]	training's auc: 0.859676	valid_1's auc: 0.687263
[1360]	training's auc: 0.859858	valid_1's auc: 0.687275
[1370]	training's auc: 0.860036	valid_1's auc: 0.687293
[1380]	training's auc: 0.860204	valid_1's auc: 0.687306
[1390]	training's auc: 0.860391	valid_1's auc: 0.68733
[1400]	training's auc: 0.860564	valid_1's auc: 0.687338
[1410]	training's auc: 0.860762	valid_1's auc: 0.68736
[1420]	training's auc: 0.860922	valid_1's auc: 0.687369
[1430]	training's auc: 0.861095	valid_1's auc: 0.687381
[1440]	training's auc: 0.861288	valid_1's auc: 0.687405
[1450]	training's auc: 0.861453	valid_1's auc: 0.687409
[1460]	training's auc: 0.86162	valid_1's auc: 0.68742
[1470]	training's auc: 0.861824	valid_1's auc: 0.687459
[1480]	training's auc: 0.861992	valid_1's auc: 0.687471
[1490]	training's auc: 0.862165	valid_1's auc: 0.687482
[1500]	training's auc: 0.86233	valid_1's auc: 0.687504
[1510]	training's auc: 0.862486	valid_1's auc: 0.687507
[1520]	training's auc: 0.862632	valid_1's auc: 0.68752
[1530]	training's auc: 0.862842	valid_1's auc: 0.687567
[1540]	training's auc: 0.86301	valid_1's auc: 0.687577
[1550]	training's auc: 0.863171	valid_1's auc: 0.687587
[1560]	training's auc: 0.863358	valid_1's auc: 0.6876
[1570]	training's auc: 0.863522	valid_1's auc: 0.687616
[1580]	training's auc: 0.863682	valid_1's auc: 0.687635
[1590]	training's auc: 0.863829	valid_1's auc: 0.687664
[1600]	training's auc: 0.863987	valid_1's auc: 0.687686
[1610]	training's auc: 0.864161	valid_1's auc: 0.687693
[1620]	training's auc: 0.864309	valid_1's auc: 0.687697
[1630]	training's auc: 0.864445	valid_1's auc: 0.687712
[1640]	training's auc: 0.864579	valid_1's auc: 0.687715
[1650]	training's auc: 0.864737	valid_1's auc: 0.687734
[1660]	training's auc: 0.864892	valid_1's auc: 0.687741
[1670]	training's auc: 0.865045	valid_1's auc: 0.687748
[1680]	training's auc: 0.865183	valid_1's auc: 0.687759
[1690]	training's auc: 0.865326	valid_1's auc: 0.687775
[1700]	training's auc: 0.865479	valid_1's auc: 0.687782
[1710]	training's auc: 0.865604	valid_1's auc: 0.687793
[1720]	training's auc: 0.86574	valid_1's auc: 0.687814
[1730]	training's auc: 0.865887	valid_1's auc: 0.687819
[1740]	training's auc: 0.866009	valid_1's auc: 0.687833
[1750]	training's auc: 0.866128	valid_1's auc: 0.687836
[1760]	training's auc: 0.866247	valid_1's auc: 0.687841
[1770]	training's auc: 0.866391	valid_1's auc: 0.687855
[1780]	training's auc: 0.866542	valid_1's auc: 0.687898
[1790]	training's auc: 0.866669	valid_1's auc: 0.687899
[1800]	training's auc: 0.866817	valid_1's auc: 0.687903

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''