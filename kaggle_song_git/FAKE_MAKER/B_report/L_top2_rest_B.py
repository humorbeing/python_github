import sys
sys.path.insert(0, '../')
from me import *
from fake_L_top2 import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier



since = time.time()
print()
print('This is [no drill] training.')
print()
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'final_train_play.csv'
train = read_df(load_name)
show_df(train)

train, test = fake_df(train)


K = 3
dfs = divide_df(train, K)
del train
dfs_collector = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dfs_collector.append(dc)

test_collector = pd.DataFrame()
test_collector['target'] = test['target']


# !!!!!!!!!!!!!!!!!!!!!!!!!

# dfs_collector, test_collector, r = Ldrt_top2_1(
#     K, dfs, dfs_collector, test, test_collector
# )
#
dfs_collector, test_collector, r = Lgos_top2_1(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = Lrf_top2_1(
    K, dfs, dfs_collector, test, test_collector
)
#
# dfs_collector, test_collector, r = Lgbt_top2_1(
#     K, dfs, dfs_collector, test, test_collector
# )
#
# #-----------------------------
#
dfs_collector, test_collector, r = Ldrt_top2_2(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = Lgos_top2_2(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = Lrf_top2_2(
    K, dfs, dfs_collector, test, test_collector
)

dfs_collector, test_collector, r = Lgbt_top2_2(
    K, dfs, dfs_collector, test, test_collector
)



# !!!!!!!!!!!!!!!!!!!!!!!!!

print(test_collector.head())
print(test_collector.tail())
save_name = 'L_rest'
save_here = '../fake/saves/feature/level1/'
for i in range(K):
    save_train = save_here + 'train' + str(i+1) + '/'
    save_df(dfs_collector[i], name=save_name,
            save_to=save_train)

save_df(dfs_collector[i], name=save_name,
            save_to=save_here+'test/')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/FAKE_MAKER/L_top2_rest.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5

This is [no drill] training.
  return f(*args, **kwds)


>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                             category
song_id                          category
source_system_tab                category
source_screen_name               category
source_type                      category
target                              uint8
artist_name                      category
language                         category
song_year                        category
top2_in_song                     category
top3_in_song                     category
membership_days                     int64
ISC_song_year                       int64
ISC_song_country_ln               float64
ITC_msno_log10_1                  float32
ITC_song_id_log10_1               float32
ITC_source_system_tab_log10_1     float32
OinC_language                     float32
dtype: object
number of rows: 7377418
number of columns: 18

'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'artist_name',
'language',
'song_year',
'top2_in_song',
'top3_in_song',
'membership_days',
'ISC_song_year',
'ISC_song_country_ln',
'ITC_msno_log10_1',
'ITC_song_id_log10_1',
'ITC_source_system_tab_log10_1',
'OinC_language',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: Lgos_top2_1  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.745269	valid_1's auc: 0.650945
[20]	training's auc: 0.758375	valid_1's auc: 0.657114
[30]	training's auc: 0.766899	valid_1's auc: 0.660485
[40]	training's auc: 0.771373	valid_1's auc: 0.662623
[50]	training's auc: 0.775719	valid_1's auc: 0.664307
[60]	training's auc: 0.778898	valid_1's auc: 0.66559
[70]	training's auc: 0.781187	valid_1's auc: 0.666239
[80]	training's auc: 0.782159	valid_1's auc: 0.666474
[90]	training's auc: 0.783505	valid_1's auc: 0.66668
[100]	training's auc: 0.784412	valid_1's auc: 0.666993
[110]	training's auc: 0.785247	valid_1's auc: 0.667613
[120]	training's auc: 0.786013	valid_1's auc: 0.667583
[130]	training's auc: 0.786461	valid_1's auc: 0.667543
[140]	training's auc: 0.787039	valid_1's auc: 0.667444
[150]	training's auc: 0.787398	valid_1's auc: 0.667547
[160]	training's auc: 0.787968	valid_1's auc: 0.667879
[170]	training's auc: 0.788238	valid_1's auc: 0.667749
[180]	training's auc: 0.788555	valid_1's auc: 0.667792
[190]	training's auc: 0.788896	valid_1's auc: 0.668325
[200]	training's auc: 0.789099	valid_1's auc: 0.668214
[210]	training's auc: 0.789314	valid_1's auc: 0.668561
[220]	training's auc: 0.789493	valid_1's auc: 0.668475
[230]	training's auc: 0.789707	valid_1's auc: 0.668395
[240]	training's auc: 0.789811	valid_1's auc: 0.668111
[250]	training's auc: 0.789878	valid_1's auc: 0.668023
Early stopping, best iteration is:
[204]	training's auc: 0.789194	valid_1's auc: 0.668615
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1
0        1     0.464216
3        1     0.652435
6        1     0.854179
9        1     0.952163
12       0     0.786491
# # # # # # # # # # 
0.559258080371
0.392132187594
0.566807562915
0.31841855331
0.508495736546
# # # # # # # # # # 

in model: Lgos_top2_1  k-fold: 2 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.746676	valid_1's auc: 0.650855
[20]	training's auc: 0.759431	valid_1's auc: 0.658014
[30]	training's auc: 0.766539	valid_1's auc: 0.661305
[40]	training's auc: 0.770866	valid_1's auc: 0.663199
[50]	training's auc: 0.775103	valid_1's auc: 0.665131
[60]	training's auc: 0.778113	valid_1's auc: 0.665836
[70]	training's auc: 0.78104	valid_1's auc: 0.66656
[80]	training's auc: 0.781957	valid_1's auc: 0.667186
[90]	training's auc: 0.783212	valid_1's auc: 0.667406
[100]	training's auc: 0.784116	valid_1's auc: 0.66713
[110]	training's auc: 0.784837	valid_1's auc: 0.66747
[120]	training's auc: 0.785564	valid_1's auc: 0.66734
[130]	training's auc: 0.785964	valid_1's auc: 0.667539
[140]	training's auc: 0.786504	valid_1's auc: 0.667398
[150]	training's auc: 0.786857	valid_1's auc: 0.667591
[160]	training's auc: 0.787327	valid_1's auc: 0.667986
[170]	training's auc: 0.7876	valid_1's auc: 0.667822
[180]	training's auc: 0.787874	valid_1's auc: 0.667685
[190]	training's auc: 0.788213	valid_1's auc: 0.667897
[200]	training's auc: 0.788479	valid_1's auc: 0.667907
Early stopping, best iteration is:
[158]	training's auc: 0.787247	valid_1's auc: 0.668021
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1
1        1     0.640505
4        1     0.561655
7        1     0.708383
10       1     0.845610
13       1     0.767405
# # # # # # # # # # 
1.25622570587
0.830213078201
1.15680355209
0.894516878734
1.00968878873
# # # # # # # # # # 

in model: Lgos_top2_1  k-fold: 3 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.745841	valid_1's auc: 0.650806
[20]	training's auc: 0.758595	valid_1's auc: 0.657911
[30]	training's auc: 0.766055	valid_1's auc: 0.66116
[40]	training's auc: 0.770539	valid_1's auc: 0.663121
[50]	training's auc: 0.774945	valid_1's auc: 0.664881
[60]	training's auc: 0.778547	valid_1's auc: 0.666399
[70]	training's auc: 0.780591	valid_1's auc: 0.667177
[80]	training's auc: 0.781873	valid_1's auc: 0.667546
[90]	training's auc: 0.783264	valid_1's auc: 0.66802
[100]	training's auc: 0.784289	valid_1's auc: 0.668502
[110]	training's auc: 0.785085	valid_1's auc: 0.668761
[120]	training's auc: 0.785761	valid_1's auc: 0.668855
[130]	training's auc: 0.786175	valid_1's auc: 0.668673
[140]	training's auc: 0.786653	valid_1's auc: 0.668857
[150]	training's auc: 0.786902	valid_1's auc: 0.668816
[160]	training's auc: 0.787529	valid_1's auc: 0.669125
[170]	training's auc: 0.787813	valid_1's auc: 0.66937
[180]	training's auc: 0.788066	valid_1's auc: 0.66941
[190]	training's auc: 0.788327	valid_1's auc: 0.669441
[200]	training's auc: 0.788574	valid_1's auc: 0.669326
[210]	training's auc: 0.788727	valid_1's auc: 0.669152
[220]	training's auc: 0.788931	valid_1's auc: 0.669176
[230]	training's auc: 0.789	valid_1's auc: 0.669221
Early stopping, best iteration is:
[184]	training's auc: 0.788152	valid_1's auc: 0.669504
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1
2        1     0.709626
5        1     0.745778
8        1     0.926473
11       1     0.937283
14       1     0.928082
# # # # # # # # # # 
1.85048500555
1.34989910233
1.72569356595
1.33261053499
1.47533594081
# # # # # # # # # # 
         target  Lgos_top2_1
5606837       0     0.616828
5606838       0     0.449966
5606839       1     0.575231
5606840       1     0.444204
5606841       0     0.491779

in model: Lrf_top2_1  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.737107	valid_1's auc: 0.644517
[20]	training's auc: 0.736753	valid_1's auc: 0.646768
[30]	training's auc: 0.736454	valid_1's auc: 0.646841
[40]	training's auc: 0.736649	valid_1's auc: 0.647175
[50]	training's auc: 0.738817	valid_1's auc: 0.647818
[60]	training's auc: 0.738988	valid_1's auc: 0.647865
[70]	training's auc: 0.739145	valid_1's auc: 0.647781
[80]	training's auc: 0.738638	valid_1's auc: 0.647781
[90]	training's auc: 0.738056	valid_1's auc: 0.647397
[100]	training's auc: 0.737201	valid_1's auc: 0.646849
[110]	training's auc: 0.737053	valid_1's auc: 0.646694
Early stopping, best iteration is:
[64]	training's auc: 0.739953	valid_1's auc: 0.648303
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1
0        1     0.464216    0.481056
3        1     0.652435    0.579828
6        1     0.854179    0.707334
9        1     0.952163    0.668980
12       0     0.786491    0.598958
# # # # # # # # # # 
0.71488047975
0.608645479805
0.408528317973
0.386541836632
0.650591159228
# # # # # # # # # # 

in model: Lrf_top2_1  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.736731	valid_1's auc: 0.643919
[20]	training's auc: 0.736223	valid_1's auc: 0.646485
[30]	training's auc: 0.735971	valid_1's auc: 0.646673
[40]	training's auc: 0.736294	valid_1's auc: 0.647017
[50]	training's auc: 0.738435	valid_1's auc: 0.647689
[60]	training's auc: 0.738726	valid_1's auc: 0.647823
[70]	training's auc: 0.738867	valid_1's auc: 0.647745
[80]	training's auc: 0.738384	valid_1's auc: 0.647665
[90]	training's auc: 0.737856	valid_1's auc: 0.647258
[100]	training's auc: 0.736996	valid_1's auc: 0.646697
[110]	training's auc: 0.736837	valid_1's auc: 0.646531
Early stopping, best iteration is:
[64]	training's auc: 0.73971	valid_1's auc: 0.648261
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1
1        1     0.640505    0.590727
4        1     0.561655    0.519061
7        1     0.708383    0.523621
10       1     0.845610    0.639616
13       1     0.767405    0.683715
# # # # # # # # # # 
1.42765919448
1.24017161831
0.824347548888
0.777833510561
1.30313584546
# # # # # # # # # # 

in model: Lrf_top2_1  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.737345	valid_1's auc: 0.644714
[20]	training's auc: 0.73675	valid_1's auc: 0.646736
[30]	training's auc: 0.736305	valid_1's auc: 0.646771
[40]	training's auc: 0.736555	valid_1's auc: 0.647071
[50]	training's auc: 0.738832	valid_1's auc: 0.64792
[60]	training's auc: 0.738973	valid_1's auc: 0.647988
[70]	training's auc: 0.739081	valid_1's auc: 0.647909
[80]	training's auc: 0.73855	valid_1's auc: 0.647907
[90]	training's auc: 0.738003	valid_1's auc: 0.647528
[100]	training's auc: 0.737155	valid_1's auc: 0.646982
[110]	training's auc: 0.737008	valid_1's auc: 0.646813
Early stopping, best iteration is:
[64]	training's auc: 0.739887	valid_1's auc: 0.648434
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1
2        1     0.709626    0.590464
5        1     0.745778    0.548480
8        1     0.926473    0.681419
11       1     0.937283    0.669170
14       1     0.928082    0.720894
# # # # # # # # # # 
2.14567092368
1.87021057998
1.23375105337
1.16387895509
1.95420479498
# # # # # # # # # # 
         target  Lgos_top2_1  Lrf_top2_1
5606837       0     0.616828    0.715224
5606838       0     0.449966    0.623404
5606839       1     0.575231    0.411250
5606840       1     0.444204    0.387960
5606841       0     0.491779    0.651402

in model: Ldrt_top2_2  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.738667	valid_1's auc: 0.648806
[20]	training's auc: 0.752192	valid_1's auc: 0.655346
[30]	training's auc: 0.760511	valid_1's auc: 0.657298
[40]	training's auc: 0.765561	valid_1's auc: 0.659286
[50]	training's auc: 0.769074	valid_1's auc: 0.660963
[60]	training's auc: 0.772283	valid_1's auc: 0.662555
[70]	training's auc: 0.774353	valid_1's auc: 0.663235
[80]	training's auc: 0.775644	valid_1's auc: 0.664686
[90]	training's auc: 0.77617	valid_1's auc: 0.664679
[100]	training's auc: 0.777837	valid_1's auc: 0.665253
[110]	training's auc: 0.778184	valid_1's auc: 0.66598
[120]	training's auc: 0.779392	valid_1's auc: 0.667011
[130]	training's auc: 0.780065	valid_1's auc: 0.667238
[140]	training's auc: 0.781398	valid_1's auc: 0.667264
[150]	training's auc: 0.782257	valid_1's auc: 0.667435
[160]	training's auc: 0.78233	valid_1's auc: 0.667924
[170]	training's auc: 0.783369	valid_1's auc: 0.667961
[180]	training's auc: 0.784092	valid_1's auc: 0.668056
[190]	training's auc: 0.784822	valid_1's auc: 0.667797
[200]	training's auc: 0.785471	valid_1's auc: 0.667999
[210]	training's auc: 0.786107	valid_1's auc: 0.668344
[220]	training's auc: 0.786727	valid_1's auc: 0.668381
[230]	training's auc: 0.787096	valid_1's auc: 0.668906
[240]	training's auc: 0.787563	valid_1's auc: 0.669131
[250]	training's auc: 0.788053	valid_1's auc: 0.669403
[260]	training's auc: 0.788577	valid_1's auc: 0.669498
[270]	training's auc: 0.789014	valid_1's auc: 0.669354
[280]	training's auc: 0.789486	valid_1's auc: 0.669482
[290]	training's auc: 0.789767	valid_1's auc: 0.669167
[300]	training's auc: 0.790139	valid_1's auc: 0.669402
[310]	training's auc: 0.790577	valid_1's auc: 0.669438
Early stopping, best iteration is:
[263]	training's auc: 0.788747	valid_1's auc: 0.669662
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2
0        1     0.464216    0.481056     0.646621
3        1     0.652435    0.579828     0.670813
6        1     0.854179    0.707334     0.867603
9        1     0.952163    0.668980     0.913361
12       0     0.786491    0.598958     0.756578
# # # # # # # # # # 
0.637556450527
0.413968313885
0.524381046936
0.349283595332
0.579879334355
# # # # # # # # # # 

in model: Ldrt_top2_2  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.738506	valid_1's auc: 0.648718
[20]	training's auc: 0.752209	valid_1's auc: 0.655494
[30]	training's auc: 0.760094	valid_1's auc: 0.657286
[40]	training's auc: 0.765184	valid_1's auc: 0.659745
[50]	training's auc: 0.768024	valid_1's auc: 0.661888
[60]	training's auc: 0.771285	valid_1's auc: 0.662755
[70]	training's auc: 0.775	valid_1's auc: 0.664179
[80]	training's auc: 0.775692	valid_1's auc: 0.665001
[90]	training's auc: 0.776247	valid_1's auc: 0.665343
[100]	training's auc: 0.777944	valid_1's auc: 0.665908
[110]	training's auc: 0.77849	valid_1's auc: 0.666483
[120]	training's auc: 0.779569	valid_1's auc: 0.666883
[130]	training's auc: 0.78023	valid_1's auc: 0.667348
[140]	training's auc: 0.781674	valid_1's auc: 0.668099
[150]	training's auc: 0.782451	valid_1's auc: 0.668401
[160]	training's auc: 0.782856	valid_1's auc: 0.668366
[170]	training's auc: 0.783893	valid_1's auc: 0.668477
[180]	training's auc: 0.784534	valid_1's auc: 0.668327
[190]	training's auc: 0.785215	valid_1's auc: 0.668707
[200]	training's auc: 0.785818	valid_1's auc: 0.668666
[210]	training's auc: 0.786476	valid_1's auc: 0.669177
[220]	training's auc: 0.786971	valid_1's auc: 0.668876
[230]	training's auc: 0.787486	valid_1's auc: 0.669128
[240]	training's auc: 0.787968	valid_1's auc: 0.669247
[250]	training's auc: 0.788192	valid_1's auc: 0.669566
[260]	training's auc: 0.788672	valid_1's auc: 0.669407
[270]	training's auc: 0.789163	valid_1's auc: 0.669568
[280]	training's auc: 0.789602	valid_1's auc: 0.669608
[290]	training's auc: 0.78999	valid_1's auc: 0.669683
[300]	training's auc: 0.790451	valid_1's auc: 0.669777
[310]	training's auc: 0.790761	valid_1's auc: 0.669783
[320]	training's auc: 0.790889	valid_1's auc: 0.669955
[330]	training's auc: 0.791165	valid_1's auc: 0.670135
[340]	training's auc: 0.791434	valid_1's auc: 0.670369
[350]	training's auc: 0.791584	valid_1's auc: 0.669981
[360]	training's auc: 0.791909	valid_1's auc: 0.670141
[370]	training's auc: 0.792185	valid_1's auc: 0.670314
[380]	training's auc: 0.792486	valid_1's auc: 0.670347
Early stopping, best iteration is:
[336]	training's auc: 0.791352	valid_1's auc: 0.670568
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2
1        1     0.640505    0.590727     0.620106
4        1     0.561655    0.519061     0.593088
7        1     0.708383    0.523621     0.586171
10       1     0.845610    0.639616     0.872397
13       1     0.767405    0.683715     0.831088
# # # # # # # # # # 
1.27084665854
0.812019125514
1.09267087448
0.794177292226
0.933773283562
# # # # # # # # # # 

in model: Ldrt_top2_2  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.738062	valid_1's auc: 0.64789
[20]	training's auc: 0.752392	valid_1's auc: 0.655174
[30]	training's auc: 0.760503	valid_1's auc: 0.65771
[40]	training's auc: 0.764986	valid_1's auc: 0.659494
[50]	training's auc: 0.768432	valid_1's auc: 0.661934
[60]	training's auc: 0.771719	valid_1's auc: 0.663172
[70]	training's auc: 0.774167	valid_1's auc: 0.663762
[80]	training's auc: 0.774922	valid_1's auc: 0.664608
[90]	training's auc: 0.775501	valid_1's auc: 0.664922
[100]	training's auc: 0.777449	valid_1's auc: 0.66566
[110]	training's auc: 0.77801	valid_1's auc: 0.666555
[120]	training's auc: 0.779503	valid_1's auc: 0.667009
[130]	training's auc: 0.780125	valid_1's auc: 0.667226
[140]	training's auc: 0.781415	valid_1's auc: 0.667558
[150]	training's auc: 0.782274	valid_1's auc: 0.668006
[160]	training's auc: 0.782589	valid_1's auc: 0.668817
[170]	training's auc: 0.783621	valid_1's auc: 0.668819
[180]	training's auc: 0.784237	valid_1's auc: 0.668689
[190]	training's auc: 0.784981	valid_1's auc: 0.669046
[200]	training's auc: 0.78568	valid_1's auc: 0.669188
[210]	training's auc: 0.786094	valid_1's auc: 0.669841
[220]	training's auc: 0.786778	valid_1's auc: 0.669792
[230]	training's auc: 0.787171	valid_1's auc: 0.670176
[240]	training's auc: 0.787537	valid_1's auc: 0.669997
[250]	training's auc: 0.788097	valid_1's auc: 0.670757
[260]	training's auc: 0.788656	valid_1's auc: 0.670702
[270]	training's auc: 0.789123	valid_1's auc: 0.670399
[280]	training's auc: 0.789904	valid_1's auc: 0.67036
[290]	training's auc: 0.790375	valid_1's auc: 0.670139
Early stopping, best iteration is:
[249]	training's auc: 0.788105	valid_1's auc: 0.670865
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2
2        1     0.709626    0.590464     0.707518
5        1     0.745778    0.548480     0.695514
8        1     0.926473    0.681419     0.856057
11       1     0.937283    0.669170     0.914655
14       1     0.928082    0.720894     0.890572
# # # # # # # # # # 
1.86197091699
1.27343194404
1.6601695748
1.24670638625
1.46650145661
# # # # # # # # # # 
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2
5606837       0     0.616828    0.715224     0.620657
5606838       0     0.449966    0.623404     0.424477
5606839       1     0.575231    0.411250     0.553390
5606840       1     0.444204    0.387960     0.415569
5606841       0     0.491779    0.651402     0.488834

in model: Lgos_top2_2  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.734824	valid_1's auc: 0.644137
[20]	training's auc: 0.740506	valid_1's auc: 0.64847
[30]	training's auc: 0.74343	valid_1's auc: 0.650214
[40]	training's auc: 0.745649	valid_1's auc: 0.651297
[50]	training's auc: 0.749552	valid_1's auc: 0.653039
[60]	training's auc: 0.751899	valid_1's auc: 0.654132
[70]	training's auc: 0.754325	valid_1's auc: 0.655349
[80]	training's auc: 0.755265	valid_1's auc: 0.656163
[90]	training's auc: 0.756729	valid_1's auc: 0.656891
[100]	training's auc: 0.757756	valid_1's auc: 0.657466
[110]	training's auc: 0.758991	valid_1's auc: 0.658159
[120]	training's auc: 0.759778	valid_1's auc: 0.658525
[130]	training's auc: 0.760622	valid_1's auc: 0.65898
[140]	training's auc: 0.761694	valid_1's auc: 0.659518
[150]	training's auc: 0.762905	valid_1's auc: 0.660123
[160]	training's auc: 0.764117	valid_1's auc: 0.660633
[170]	training's auc: 0.765102	valid_1's auc: 0.661159
[180]	training's auc: 0.766162	valid_1's auc: 0.661647
[190]	training's auc: 0.766974	valid_1's auc: 0.66202
[200]	training's auc: 0.767945	valid_1's auc: 0.662409
[210]	training's auc: 0.768656	valid_1's auc: 0.662797
[220]	training's auc: 0.769429	valid_1's auc: 0.663156
[230]	training's auc: 0.770107	valid_1's auc: 0.663438
[240]	training's auc: 0.770784	valid_1's auc: 0.663765
[250]	training's auc: 0.771353	valid_1's auc: 0.664069
[260]	training's auc: 0.772045	valid_1's auc: 0.664325
[270]	training's auc: 0.772717	valid_1's auc: 0.664638
[280]	training's auc: 0.773548	valid_1's auc: 0.665011
[290]	training's auc: 0.774131	valid_1's auc: 0.665285
[300]	training's auc: 0.774874	valid_1's auc: 0.665608
[310]	training's auc: 0.775426	valid_1's auc: 0.665762
[320]	training's auc: 0.775956	valid_1's auc: 0.665982
[330]	training's auc: 0.776603	valid_1's auc: 0.666233
[340]	training's auc: 0.777072	valid_1's auc: 0.666425
[350]	training's auc: 0.777663	valid_1's auc: 0.66663
[360]	training's auc: 0.77808	valid_1's auc: 0.666805
[370]	training's auc: 0.7786	valid_1's auc: 0.666983
[380]	training's auc: 0.779168	valid_1's auc: 0.667213
[390]	training's auc: 0.779629	valid_1's auc: 0.667413
[400]	training's auc: 0.779944	valid_1's auc: 0.667563
[410]	training's auc: 0.780315	valid_1's auc: 0.667745
[420]	training's auc: 0.780731	valid_1's auc: 0.667935
[430]	training's auc: 0.781106	valid_1's auc: 0.668088
[440]	training's auc: 0.781537	valid_1's auc: 0.668251
[450]	training's auc: 0.781942	valid_1's auc: 0.668442
[460]	training's auc: 0.78238	valid_1's auc: 0.668659
[470]	training's auc: 0.782744	valid_1's auc: 0.668784
[480]	training's auc: 0.783073	valid_1's auc: 0.668866
[490]	training's auc: 0.783362	valid_1's auc: 0.669007
[500]	training's auc: 0.78372	valid_1's auc: 0.669137
[510]	training's auc: 0.783993	valid_1's auc: 0.669206
[520]	training's auc: 0.784406	valid_1's auc: 0.669354
[530]	training's auc: 0.78465	valid_1's auc: 0.669493
[540]	training's auc: 0.784916	valid_1's auc: 0.66958
[550]	training's auc: 0.785151	valid_1's auc: 0.669723
[560]	training's auc: 0.785537	valid_1's auc: 0.66989
[570]	training's auc: 0.785854	valid_1's auc: 0.670013
[580]	training's auc: 0.786169	valid_1's auc: 0.670126
[590]	training's auc: 0.786524	valid_1's auc: 0.670262
[600]	training's auc: 0.786786	valid_1's auc: 0.67035
[610]	training's auc: 0.787065	valid_1's auc: 0.670415
[620]	training's auc: 0.787374	valid_1's auc: 0.67054
[630]	training's auc: 0.787669	valid_1's auc: 0.670671
[640]	training's auc: 0.787908	valid_1's auc: 0.670745
[650]	training's auc: 0.788212	valid_1's auc: 0.670859
[660]	training's auc: 0.788437	valid_1's auc: 0.670896
[670]	training's auc: 0.788619	valid_1's auc: 0.670974
[680]	training's auc: 0.788854	valid_1's auc: 0.671021
[690]	training's auc: 0.789122	valid_1's auc: 0.671117
[700]	training's auc: 0.789332	valid_1's auc: 0.671211
[710]	training's auc: 0.789504	valid_1's auc: 0.671289
[720]	training's auc: 0.789699	valid_1's auc: 0.67134
[730]	training's auc: 0.789924	valid_1's auc: 0.671418
[740]	training's auc: 0.790072	valid_1's auc: 0.67144
[750]	training's auc: 0.790216	valid_1's auc: 0.67149
[760]	training's auc: 0.790437	valid_1's auc: 0.671562
[770]	training's auc: 0.790651	valid_1's auc: 0.671634
[780]	training's auc: 0.790815	valid_1's auc: 0.671682
[790]	training's auc: 0.790981	valid_1's auc: 0.671747
[800]	training's auc: 0.791161	valid_1's auc: 0.6718
[810]	training's auc: 0.791312	valid_1's auc: 0.671814
[820]	training's auc: 0.791493	valid_1's auc: 0.671861
[830]	training's auc: 0.791678	valid_1's auc: 0.671927
[840]	training's auc: 0.791829	valid_1's auc: 0.671921
[850]	training's auc: 0.791949	valid_1's auc: 0.671916
[860]	training's auc: 0.792109	valid_1's auc: 0.672022
[870]	training's auc: 0.792237	valid_1's auc: 0.672043
[880]	training's auc: 0.792348	valid_1's auc: 0.672067
[890]	training's auc: 0.79252	valid_1's auc: 0.672138
[900]	training's auc: 0.792669	valid_1's auc: 0.672172
[910]	training's auc: 0.792815	valid_1's auc: 0.672185
[920]	training's auc: 0.792967	valid_1's auc: 0.672231
[930]	training's auc: 0.793055	valid_1's auc: 0.672236
[940]	training's auc: 0.793156	valid_1's auc: 0.672232
[950]	training's auc: 0.793257	valid_1's auc: 0.672264
[960]	training's auc: 0.793384	valid_1's auc: 0.672357
[970]	training's auc: 0.793494	valid_1's auc: 0.672378
[980]	training's auc: 0.793601	valid_1's auc: 0.672404
[990]	training's auc: 0.793732	valid_1's auc: 0.672435
[1000]	training's auc: 0.79386	valid_1's auc: 0.672477
[1010]	training's auc: 0.793998	valid_1's auc: 0.672559
[1020]	training's auc: 0.794107	valid_1's auc: 0.672568
[1030]	training's auc: 0.794249	valid_1's auc: 0.672621
[1040]	training's auc: 0.794372	valid_1's auc: 0.672677
[1050]	training's auc: 0.794476	valid_1's auc: 0.672685
[1060]	training's auc: 0.794593	valid_1's auc: 0.67269
[1070]	training's auc: 0.794637	valid_1's auc: 0.672663
[1080]	training's auc: 0.79474	valid_1's auc: 0.672724
[1090]	training's auc: 0.794847	valid_1's auc: 0.672754
[1100]	training's auc: 0.794922	valid_1's auc: 0.672797
[1110]	training's auc: 0.795028	valid_1's auc: 0.672833
[1120]	training's auc: 0.795102	valid_1's auc: 0.672856
[1130]	training's auc: 0.795171	valid_1's auc: 0.672838
[1140]	training's auc: 0.79524	valid_1's auc: 0.672872
[1150]	training's auc: 0.795298	valid_1's auc: 0.672891
[1160]	training's auc: 0.795367	valid_1's auc: 0.672916
[1170]	training's auc: 0.795438	valid_1's auc: 0.672915
[1180]	training's auc: 0.795548	valid_1's auc: 0.672938
[1190]	training's auc: 0.795616	valid_1's auc: 0.67298
[1200]	training's auc: 0.795715	valid_1's auc: 0.673003
[1210]	training's auc: 0.795803	valid_1's auc: 0.673026
[1220]	training's auc: 0.795856	valid_1's auc: 0.673045
[1230]	training's auc: 0.795958	valid_1's auc: 0.673048
[1240]	training's auc: 0.796024	valid_1's auc: 0.673047
[1250]	training's auc: 0.79606	valid_1's auc: 0.673033
[1260]	training's auc: 0.796138	valid_1's auc: 0.673099
[1270]	training's auc: 0.796176	valid_1's auc: 0.673127
[1280]	training's auc: 0.796243	valid_1's auc: 0.67314
[1290]	training's auc: 0.796322	valid_1's auc: 0.673172
[1300]	training's auc: 0.796411	valid_1's auc: 0.673207
[1310]	training's auc: 0.796478	valid_1's auc: 0.673215
[1320]	training's auc: 0.796547	valid_1's auc: 0.673226
[1330]	training's auc: 0.796635	valid_1's auc: 0.673262
[1340]	training's auc: 0.796701	valid_1's auc: 0.673276
[1350]	training's auc: 0.79675	valid_1's auc: 0.673261
[1360]	training's auc: 0.796805	valid_1's auc: 0.673239
[1370]	training's auc: 0.796855	valid_1's auc: 0.673232
[1380]	training's auc: 0.796905	valid_1's auc: 0.673265
Early stopping, best iteration is:
[1335]	training's auc: 0.796672	valid_1's auc: 0.673289
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2
0        1     0.464216    0.481056     0.646621     0.459288
3        1     0.652435    0.579828     0.670813     0.620659
6        1     0.854179    0.707334     0.867603     0.825738
9        1     0.952163    0.668980     0.913361     0.911859
12       0     0.786491    0.598958     0.756578     0.779798
# # # # # # # # # # 
0.543899174081
0.473496582951
0.500314002265
0.388076395673
0.488038599934
# # # # # # # # # # 

in model: Lgos_top2_2  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.734779	valid_1's auc: 0.644264
[20]	training's auc: 0.740401	valid_1's auc: 0.648155
[30]	training's auc: 0.743532	valid_1's auc: 0.650054
[40]	training's auc: 0.745783	valid_1's auc: 0.651058
[50]	training's auc: 0.749628	valid_1's auc: 0.652593
[60]	training's auc: 0.751803	valid_1's auc: 0.653747
[70]	training's auc: 0.754192	valid_1's auc: 0.654897
[80]	training's auc: 0.755201	valid_1's auc: 0.655511
[90]	training's auc: 0.756716	valid_1's auc: 0.656386
[100]	training's auc: 0.757826	valid_1's auc: 0.656999
[110]	training's auc: 0.758951	valid_1's auc: 0.657617
[120]	training's auc: 0.759753	valid_1's auc: 0.65801
[130]	training's auc: 0.760571	valid_1's auc: 0.65843
[140]	training's auc: 0.761582	valid_1's auc: 0.659019
[150]	training's auc: 0.762784	valid_1's auc: 0.659621
[160]	training's auc: 0.763976	valid_1's auc: 0.660188
[170]	training's auc: 0.764956	valid_1's auc: 0.66067
[180]	training's auc: 0.766025	valid_1's auc: 0.661231
[190]	training's auc: 0.766782	valid_1's auc: 0.661651
[200]	training's auc: 0.767737	valid_1's auc: 0.662161
[210]	training's auc: 0.76846	valid_1's auc: 0.662568
[220]	training's auc: 0.769242	valid_1's auc: 0.662942
[230]	training's auc: 0.77006	valid_1's auc: 0.663277
[240]	training's auc: 0.770742	valid_1's auc: 0.663592
[250]	training's auc: 0.77136	valid_1's auc: 0.663919
[260]	training's auc: 0.772041	valid_1's auc: 0.664229
[270]	training's auc: 0.772751	valid_1's auc: 0.664592
[280]	training's auc: 0.773511	valid_1's auc: 0.66497
[290]	training's auc: 0.774115	valid_1's auc: 0.665256
[300]	training's auc: 0.774854	valid_1's auc: 0.665636
[310]	training's auc: 0.775429	valid_1's auc: 0.665885
[320]	training's auc: 0.775945	valid_1's auc: 0.666106
[330]	training's auc: 0.776612	valid_1's auc: 0.666404
[340]	training's auc: 0.777144	valid_1's auc: 0.666624
[350]	training's auc: 0.777708	valid_1's auc: 0.666849
[360]	training's auc: 0.778058	valid_1's auc: 0.666965
[370]	training's auc: 0.778597	valid_1's auc: 0.667195
[380]	training's auc: 0.779159	valid_1's auc: 0.667421
[390]	training's auc: 0.779606	valid_1's auc: 0.667513
[400]	training's auc: 0.779898	valid_1's auc: 0.667652
[410]	training's auc: 0.780262	valid_1's auc: 0.667788
[420]	training's auc: 0.780681	valid_1's auc: 0.667958
[430]	training's auc: 0.781049	valid_1's auc: 0.668114
[440]	training's auc: 0.781486	valid_1's auc: 0.668281
[450]	training's auc: 0.781913	valid_1's auc: 0.66846
[460]	training's auc: 0.782363	valid_1's auc: 0.668633
[470]	training's auc: 0.782719	valid_1's auc: 0.668741
[480]	training's auc: 0.783055	valid_1's auc: 0.668877
[490]	training's auc: 0.783381	valid_1's auc: 0.668953
[500]	training's auc: 0.783779	valid_1's auc: 0.669115
[510]	training's auc: 0.784039	valid_1's auc: 0.669208
[520]	training's auc: 0.784432	valid_1's auc: 0.669378
[530]	training's auc: 0.784713	valid_1's auc: 0.669455
[540]	training's auc: 0.784914	valid_1's auc: 0.669564
[550]	training's auc: 0.785194	valid_1's auc: 0.669653
[560]	training's auc: 0.785554	valid_1's auc: 0.669799
[570]	training's auc: 0.785837	valid_1's auc: 0.669861
[580]	training's auc: 0.786166	valid_1's auc: 0.669962
[590]	training's auc: 0.786514	valid_1's auc: 0.670152
[600]	training's auc: 0.786764	valid_1's auc: 0.670242
[610]	training's auc: 0.787042	valid_1's auc: 0.670346
[620]	training's auc: 0.787347	valid_1's auc: 0.670436
[630]	training's auc: 0.787667	valid_1's auc: 0.670545
[640]	training's auc: 0.787915	valid_1's auc: 0.67063
[650]	training's auc: 0.788199	valid_1's auc: 0.670758
[660]	training's auc: 0.788436	valid_1's auc: 0.670814
[670]	training's auc: 0.788603	valid_1's auc: 0.670858
[680]	training's auc: 0.78882	valid_1's auc: 0.670926
[690]	training's auc: 0.789103	valid_1's auc: 0.67099
[700]	training's auc: 0.789316	valid_1's auc: 0.671096
[710]	training's auc: 0.789479	valid_1's auc: 0.671184
[720]	training's auc: 0.789681	valid_1's auc: 0.671257
[730]	training's auc: 0.789891	valid_1's auc: 0.671326
[740]	training's auc: 0.790051	valid_1's auc: 0.671376
[750]	training's auc: 0.790261	valid_1's auc: 0.671423
[760]	training's auc: 0.790485	valid_1's auc: 0.671488
[770]	training's auc: 0.790688	valid_1's auc: 0.671582
[780]	training's auc: 0.790867	valid_1's auc: 0.671643
[790]	training's auc: 0.791016	valid_1's auc: 0.671751
[800]	training's auc: 0.791194	valid_1's auc: 0.671812
[810]	training's auc: 0.791341	valid_1's auc: 0.671812
[820]	training's auc: 0.791504	valid_1's auc: 0.671844
[830]	training's auc: 0.791697	valid_1's auc: 0.671895
[840]	training's auc: 0.791842	valid_1's auc: 0.671913
[850]	training's auc: 0.791998	valid_1's auc: 0.671936
[860]	training's auc: 0.792164	valid_1's auc: 0.672021
[870]	training's auc: 0.792316	valid_1's auc: 0.672067
[880]	training's auc: 0.792428	valid_1's auc: 0.672066
[890]	training's auc: 0.792619	valid_1's auc: 0.672107
[900]	training's auc: 0.792747	valid_1's auc: 0.672144
[910]	training's auc: 0.792885	valid_1's auc: 0.672211
[920]	training's auc: 0.793027	valid_1's auc: 0.672239
[930]	training's auc: 0.793132	valid_1's auc: 0.672309
[940]	training's auc: 0.793234	valid_1's auc: 0.672324
[950]	training's auc: 0.793357	valid_1's auc: 0.672326
[960]	training's auc: 0.793487	valid_1's auc: 0.672422
[970]	training's auc: 0.793601	valid_1's auc: 0.672443
[980]	training's auc: 0.793723	valid_1's auc: 0.672487
[990]	training's auc: 0.793862	valid_1's auc: 0.672487
[1000]	training's auc: 0.793976	valid_1's auc: 0.672521
[1010]	training's auc: 0.794104	valid_1's auc: 0.67257
[1020]	training's auc: 0.794216	valid_1's auc: 0.672586
[1030]	training's auc: 0.794362	valid_1's auc: 0.672631
[1040]	training's auc: 0.794464	valid_1's auc: 0.672656
[1050]	training's auc: 0.794572	valid_1's auc: 0.672694
[1060]	training's auc: 0.794664	valid_1's auc: 0.672699
[1070]	training's auc: 0.794708	valid_1's auc: 0.672713
[1080]	training's auc: 0.794821	valid_1's auc: 0.672744
[1090]	training's auc: 0.7949	valid_1's auc: 0.672765
[1100]	training's auc: 0.79498	valid_1's auc: 0.672776
[1110]	training's auc: 0.795052	valid_1's auc: 0.672813
[1120]	training's auc: 0.795168	valid_1's auc: 0.672848
[1130]	training's auc: 0.79526	valid_1's auc: 0.672873
[1140]	training's auc: 0.79532	valid_1's auc: 0.672842
[1150]	training's auc: 0.795375	valid_1's auc: 0.672859
[1160]	training's auc: 0.795453	valid_1's auc: 0.672871
[1170]	training's auc: 0.795515	valid_1's auc: 0.672894
[1180]	training's auc: 0.795629	valid_1's auc: 0.672922
[1190]	training's auc: 0.795693	valid_1's auc: 0.67294
[1200]	training's auc: 0.795783	valid_1's auc: 0.672959
[1210]	training's auc: 0.795872	valid_1's auc: 0.672987
[1220]	training's auc: 0.79595	valid_1's auc: 0.67299
[1230]	training's auc: 0.796039	valid_1's auc: 0.673013
[1240]	training's auc: 0.796094	valid_1's auc: 0.673004
[1250]	training's auc: 0.79615	valid_1's auc: 0.673023
[1260]	training's auc: 0.796231	valid_1's auc: 0.673071
[1270]	training's auc: 0.796272	valid_1's auc: 0.673085
[1280]	training's auc: 0.79634	valid_1's auc: 0.673128
[1290]	training's auc: 0.796405	valid_1's auc: 0.673133
[1300]	training's auc: 0.796473	valid_1's auc: 0.673128
[1310]	training's auc: 0.796535	valid_1's auc: 0.673127
[1320]	training's auc: 0.796591	valid_1's auc: 0.673123
[1330]	training's auc: 0.796669	valid_1's auc: 0.673159
[1340]	training's auc: 0.796734	valid_1's auc: 0.673162
[1350]	training's auc: 0.79676	valid_1's auc: 0.67317
[1360]	training's auc: 0.796825	valid_1's auc: 0.673204
[1370]	training's auc: 0.796871	valid_1's auc: 0.673202
[1380]	training's auc: 0.796949	valid_1's auc: 0.673207
[1390]	training's auc: 0.797019	valid_1's auc: 0.67321
[1400]	training's auc: 0.797072	valid_1's auc: 0.673241
[1410]	training's auc: 0.797143	valid_1's auc: 0.673239
[1420]	training's auc: 0.7972	valid_1's auc: 0.673228
[1430]	training's auc: 0.797255	valid_1's auc: 0.673232
[1440]	training's auc: 0.797322	valid_1's auc: 0.673253
[1450]	training's auc: 0.797373	valid_1's auc: 0.673241
[1460]	training's auc: 0.797429	valid_1's auc: 0.673267
[1470]	training's auc: 0.797464	valid_1's auc: 0.67329
[1480]	training's auc: 0.79753	valid_1's auc: 0.673294
[1490]	training's auc: 0.797573	valid_1's auc: 0.673267
[1500]	training's auc: 0.797587	valid_1's auc: 0.673275
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2
1        1     0.640505    0.590727     0.620106     0.648402
4        1     0.561655    0.519061     0.593088     0.602723
7        1     0.708383    0.523621     0.586171     0.624141
10       1     0.845610    0.639616     0.872397     0.857114
13       1     0.767405    0.683715     0.831088     0.780171
# # # # # # # # # # 
1.13184490674
0.90479026769
0.980038965761
0.809927966543
1.0014676272
# # # # # # # # # # 

in model: Lgos_top2_2  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.734726	valid_1's auc: 0.643987
[20]	training's auc: 0.740613	valid_1's auc: 0.648671
[30]	training's auc: 0.74348	valid_1's auc: 0.650472
[40]	training's auc: 0.745782	valid_1's auc: 0.651496
[50]	training's auc: 0.749704	valid_1's auc: 0.65321
[60]	training's auc: 0.751886	valid_1's auc: 0.654217
[70]	training's auc: 0.754368	valid_1's auc: 0.655431
[80]	training's auc: 0.755304	valid_1's auc: 0.656015
[90]	training's auc: 0.756813	valid_1's auc: 0.656927
[100]	training's auc: 0.757629	valid_1's auc: 0.657451
[110]	training's auc: 0.758868	valid_1's auc: 0.658082
[120]	training's auc: 0.759704	valid_1's auc: 0.658566
[130]	training's auc: 0.760504	valid_1's auc: 0.658944
[140]	training's auc: 0.761577	valid_1's auc: 0.659507
[150]	training's auc: 0.762778	valid_1's auc: 0.660134
[160]	training's auc: 0.763923	valid_1's auc: 0.660669
[170]	training's auc: 0.764905	valid_1's auc: 0.661117
[180]	training's auc: 0.765977	valid_1's auc: 0.661618
[190]	training's auc: 0.766744	valid_1's auc: 0.662054
[200]	training's auc: 0.767699	valid_1's auc: 0.662516
[210]	training's auc: 0.768345	valid_1's auc: 0.662903
[220]	training's auc: 0.769092	valid_1's auc: 0.663217
[230]	training's auc: 0.76986	valid_1's auc: 0.663501
[240]	training's auc: 0.770602	valid_1's auc: 0.663881
[250]	training's auc: 0.771213	valid_1's auc: 0.664258
[260]	training's auc: 0.771898	valid_1's auc: 0.664511
[270]	training's auc: 0.772544	valid_1's auc: 0.664805
[280]	training's auc: 0.773362	valid_1's auc: 0.665172
[290]	training's auc: 0.773998	valid_1's auc: 0.665409
[300]	training's auc: 0.774739	valid_1's auc: 0.665746
[310]	training's auc: 0.775282	valid_1's auc: 0.665993
[320]	training's auc: 0.775838	valid_1's auc: 0.666232
[330]	training's auc: 0.776486	valid_1's auc: 0.666534
[340]	training's auc: 0.776982	valid_1's auc: 0.666732
[350]	training's auc: 0.777581	valid_1's auc: 0.666981
[360]	training's auc: 0.777981	valid_1's auc: 0.667092
[370]	training's auc: 0.778493	valid_1's auc: 0.667311
[380]	training's auc: 0.779034	valid_1's auc: 0.667506
[390]	training's auc: 0.779509	valid_1's auc: 0.667737
[400]	training's auc: 0.779836	valid_1's auc: 0.667824
[410]	training's auc: 0.78021	valid_1's auc: 0.667948
[420]	training's auc: 0.780616	valid_1's auc: 0.668129
[430]	training's auc: 0.780983	valid_1's auc: 0.668286
[440]	training's auc: 0.781414	valid_1's auc: 0.668473
[450]	training's auc: 0.781858	valid_1's auc: 0.668701
[460]	training's auc: 0.782336	valid_1's auc: 0.668902
[470]	training's auc: 0.782661	valid_1's auc: 0.669031
[480]	training's auc: 0.783008	valid_1's auc: 0.669168
[490]	training's auc: 0.783334	valid_1's auc: 0.669274
[500]	training's auc: 0.783719	valid_1's auc: 0.669415
[510]	training's auc: 0.783978	valid_1's auc: 0.669488
[520]	training's auc: 0.784366	valid_1's auc: 0.669642
[530]	training's auc: 0.784594	valid_1's auc: 0.66972
[540]	training's auc: 0.784843	valid_1's auc: 0.669781
[550]	training's auc: 0.785114	valid_1's auc: 0.669898
[560]	training's auc: 0.785475	valid_1's auc: 0.670028
[570]	training's auc: 0.785809	valid_1's auc: 0.670146
[580]	training's auc: 0.786153	valid_1's auc: 0.67027
[590]	training's auc: 0.786506	valid_1's auc: 0.67038
[600]	training's auc: 0.786763	valid_1's auc: 0.670481
[610]	training's auc: 0.787027	valid_1's auc: 0.670568
[620]	training's auc: 0.787336	valid_1's auc: 0.670685
[630]	training's auc: 0.787652	valid_1's auc: 0.670803
[640]	training's auc: 0.78784	valid_1's auc: 0.67087
[650]	training's auc: 0.788127	valid_1's auc: 0.670969
[660]	training's auc: 0.788367	valid_1's auc: 0.671039
[670]	training's auc: 0.788528	valid_1's auc: 0.671104
[680]	training's auc: 0.788738	valid_1's auc: 0.671143
[690]	training's auc: 0.789008	valid_1's auc: 0.671233
[700]	training's auc: 0.789209	valid_1's auc: 0.67133
[710]	training's auc: 0.789357	valid_1's auc: 0.67136
[720]	training's auc: 0.789564	valid_1's auc: 0.671438
[730]	training's auc: 0.789791	valid_1's auc: 0.671533
[740]	training's auc: 0.789943	valid_1's auc: 0.671563
[750]	training's auc: 0.79012	valid_1's auc: 0.671595
[760]	training's auc: 0.790355	valid_1's auc: 0.671683
[770]	training's auc: 0.790546	valid_1's auc: 0.671735
[780]	training's auc: 0.79072	valid_1's auc: 0.671784
[790]	training's auc: 0.790859	valid_1's auc: 0.671852
[800]	training's auc: 0.791046	valid_1's auc: 0.671879
[810]	training's auc: 0.791189	valid_1's auc: 0.671919
[820]	training's auc: 0.791378	valid_1's auc: 0.671972
[830]	training's auc: 0.791577	valid_1's auc: 0.672043
[840]	training's auc: 0.791711	valid_1's auc: 0.672033
[850]	training's auc: 0.791843	valid_1's auc: 0.672075
[860]	training's auc: 0.791989	valid_1's auc: 0.672126
[870]	training's auc: 0.792129	valid_1's auc: 0.67211
[880]	training's auc: 0.792262	valid_1's auc: 0.672146
[890]	training's auc: 0.792445	valid_1's auc: 0.672198
[900]	training's auc: 0.792587	valid_1's auc: 0.672215
[910]	training's auc: 0.792719	valid_1's auc: 0.672262
[920]	training's auc: 0.792845	valid_1's auc: 0.672298
[930]	training's auc: 0.792965	valid_1's auc: 0.672336
[940]	training's auc: 0.793078	valid_1's auc: 0.67235
[950]	training's auc: 0.793176	valid_1's auc: 0.672366
[960]	training's auc: 0.793298	valid_1's auc: 0.67245
[970]	training's auc: 0.793407	valid_1's auc: 0.672486
[980]	training's auc: 0.793514	valid_1's auc: 0.672489
[990]	training's auc: 0.79364	valid_1's auc: 0.672506
[1000]	training's auc: 0.793758	valid_1's auc: 0.67254
[1010]	training's auc: 0.793881	valid_1's auc: 0.672614
[1020]	training's auc: 0.794002	valid_1's auc: 0.672602
[1030]	training's auc: 0.794138	valid_1's auc: 0.672657
[1040]	training's auc: 0.79426	valid_1's auc: 0.67268
[1050]	training's auc: 0.79434	valid_1's auc: 0.672704
[1060]	training's auc: 0.794454	valid_1's auc: 0.672692
[1070]	training's auc: 0.79451	valid_1's auc: 0.672698
[1080]	training's auc: 0.794621	valid_1's auc: 0.672686
[1090]	training's auc: 0.794732	valid_1's auc: 0.672688
[1100]	training's auc: 0.794804	valid_1's auc: 0.672699
[1110]	training's auc: 0.794897	valid_1's auc: 0.672716
[1120]	training's auc: 0.795	valid_1's auc: 0.672739
[1130]	training's auc: 0.795099	valid_1's auc: 0.672787
[1140]	training's auc: 0.795169	valid_1's auc: 0.672799
[1150]	training's auc: 0.795223	valid_1's auc: 0.672805
[1160]	training's auc: 0.795292	valid_1's auc: 0.672829
[1170]	training's auc: 0.795357	valid_1's auc: 0.672846
[1180]	training's auc: 0.79547	valid_1's auc: 0.672897
[1190]	training's auc: 0.79554	valid_1's auc: 0.672892
[1200]	training's auc: 0.79562	valid_1's auc: 0.672902
[1210]	training's auc: 0.795714	valid_1's auc: 0.672916
[1220]	training's auc: 0.795786	valid_1's auc: 0.672924
[1230]	training's auc: 0.795867	valid_1's auc: 0.672942
[1240]	training's auc: 0.795903	valid_1's auc: 0.672944
[1250]	training's auc: 0.79597	valid_1's auc: 0.672961
[1260]	training's auc: 0.796053	valid_1's auc: 0.673016
[1270]	training's auc: 0.796115	valid_1's auc: 0.673042
[1280]	training's auc: 0.796169	valid_1's auc: 0.673041
[1290]	training's auc: 0.796228	valid_1's auc: 0.673023
[1300]	training's auc: 0.796317	valid_1's auc: 0.673045
[1310]	training's auc: 0.796389	valid_1's auc: 0.673064
[1320]	training's auc: 0.796473	valid_1's auc: 0.673057
[1330]	training's auc: 0.796563	valid_1's auc: 0.67307
[1340]	training's auc: 0.796632	valid_1's auc: 0.673054
[1350]	training's auc: 0.796681	valid_1's auc: 0.673035
[1360]	training's auc: 0.796739	valid_1's auc: 0.673051
[1370]	training's auc: 0.796787	valid_1's auc: 0.673067
[1380]	training's auc: 0.796842	valid_1's auc: 0.67309
[1390]	training's auc: 0.796921	valid_1's auc: 0.673119
[1400]	training's auc: 0.796957	valid_1's auc: 0.673105
[1410]	training's auc: 0.79702	valid_1's auc: 0.67312
[1420]	training's auc: 0.797085	valid_1's auc: 0.673121
[1430]	training's auc: 0.797115	valid_1's auc: 0.673148
[1440]	training's auc: 0.797179	valid_1's auc: 0.673161
[1450]	training's auc: 0.797217	valid_1's auc: 0.673156
[1460]	training's auc: 0.79727	valid_1's auc: 0.673202
[1470]	training's auc: 0.79731	valid_1's auc: 0.673193
[1480]	training's auc: 0.797369	valid_1's auc: 0.673203
[1490]	training's auc: 0.797423	valid_1's auc: 0.673211
[1500]	training's auc: 0.797453	valid_1's auc: 0.673232
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2
2        1     0.709626    0.590464     0.707518     0.595401
5        1     0.745778    0.548480     0.695514     0.673461
8        1     0.926473    0.681419     0.856057     0.925611
11       1     0.937283    0.669170     0.914655     0.922491
14       1     0.928082    0.720894     0.890572     0.902568
# # # # # # # # # # 
1.7272044831
1.30531972847
1.46758690466
1.18869202728
1.48059998947
# # # # # # # # # # 
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2
5606837       0     0.616828    0.715224     0.620657     0.575735
5606838       0     0.449966    0.623404     0.424477     0.435107
5606839       1     0.575231    0.411250     0.553390     0.489196
5606840       1     0.444204    0.387960     0.415569     0.396231
5606841       0     0.491779    0.651402     0.488834     0.493533

in model: Lrf_top2_2  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.774349	valid_1's auc: 0.658907
[20]	training's auc: 0.775151	valid_1's auc: 0.659405
[30]	training's auc: 0.775245	valid_1's auc: 0.659607
[40]	training's auc: 0.774257	valid_1's auc: 0.659212
[50]	training's auc: 0.775108	valid_1's auc: 0.659402
[60]	training's auc: 0.775394	valid_1's auc: 0.65946
[70]	training's auc: 0.775742	valid_1's auc: 0.65953
Early stopping, best iteration is:
[27]	training's auc: 0.775854	valid_1's auc: 0.659756
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2
0        1     0.464216    0.481056     0.646621     0.459288    0.622414
3        1     0.652435    0.579828     0.670813     0.620659    0.604378
6        1     0.854179    0.707334     0.867603     0.825738    0.720088
9        1     0.952163    0.668980     0.913361     0.911859    0.685419
12       0     0.786491    0.598958     0.756578     0.779798    0.535406
# # # # # # # # # # 
0.722608288961
0.542650581177
0.375681667058
0.317374726712
0.688173456545
# # # # # # # # # # 

in model: Lrf_top2_2  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.7741	valid_1's auc: 0.658235
[20]	training's auc: 0.774751	valid_1's auc: 0.658916
[30]	training's auc: 0.774991	valid_1's auc: 0.659174
[40]	training's auc: 0.774055	valid_1's auc: 0.658797
[50]	training's auc: 0.774924	valid_1's auc: 0.659004
[60]	training's auc: 0.775277	valid_1's auc: 0.659093
[70]	training's auc: 0.775594	valid_1's auc: 0.659099
Early stopping, best iteration is:
[27]	training's auc: 0.775535	valid_1's auc: 0.659256
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2
1        1     0.640505    0.590727     0.620106     0.648402    0.601203
4        1     0.561655    0.519061     0.593088     0.602723    0.707695
7        1     0.708383    0.523621     0.586171     0.624141    0.698430
10       1     0.845610    0.639616     0.872397     0.857114    0.764084
13       1     0.767405    0.683715     0.831088     0.780171    0.695359
# # # # # # # # # # 
1.46842204609
1.13026513367
0.793035469934
0.65113000449
1.37270225783
# # # # # # # # # # 

in model: Lrf_top2_2  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.774197	valid_1's auc: 0.659587
[20]	training's auc: 0.774831	valid_1's auc: 0.660025
[30]	training's auc: 0.775002	valid_1's auc: 0.660354
[40]	training's auc: 0.773954	valid_1's auc: 0.659884
[50]	training's auc: 0.77495	valid_1's auc: 0.660124
[60]	training's auc: 0.775252	valid_1's auc: 0.660227
[70]	training's auc: 0.775682	valid_1's auc: 0.660349
Early stopping, best iteration is:
[27]	training's auc: 0.775644	valid_1's auc: 0.660465
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2
2        1     0.709626    0.590464     0.707518     0.595401    0.603777
5        1     0.745778    0.548480     0.695514     0.673461    0.735620
8        1     0.926473    0.681419     0.856057     0.925611    0.775460
11       1     0.937283    0.669170     0.914655     0.922491    0.693642
14       1     0.928082    0.720894     0.890572     0.902568    0.760462
# # # # # # # # # # 
2.22488724653
1.78624072471
1.15405512593
0.971044683699
2.05795581554
# # # # # # # # # # 
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2
5606837       0     0.616828    0.715224     0.620657     0.575735    0.741629
5606838       0     0.449966    0.623404     0.424477     0.435107    0.595414
5606839       1     0.575231    0.411250     0.553390     0.489196    0.384685
5606840       1     0.444204    0.387960     0.415569     0.396231    0.323682
5606841       0     0.491779    0.651402     0.488834     0.493533    0.685985

in model: Lgbt_top2_2  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.796604	valid_1's auc: 0.671173
[20]	training's auc: 0.815718	valid_1's auc: 0.679059
[30]	training's auc: 0.824143	valid_1's auc: 0.680457
[40]	training's auc: 0.828373	valid_1's auc: 0.68046
[50]	training's auc: 0.833017	valid_1's auc: 0.680168
[60]	training's auc: 0.838086	valid_1's auc: 0.680624
[70]	training's auc: 0.842039	valid_1's auc: 0.680821
[80]	training's auc: 0.844502	valid_1's auc: 0.68058
[90]	training's auc: 0.846491	valid_1's auc: 0.680484
[100]	training's auc: 0.848535	valid_1's auc: 0.680405
[110]	training's auc: 0.851076	valid_1's auc: 0.68059
[120]	training's auc: 0.852747	valid_1's auc: 0.680628
Early stopping, best iteration is:
[70]	training's auc: 0.842039	valid_1's auc: 0.680821
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2  \
0        1     0.464216    0.481056     0.646621     0.459288    0.622414   
3        1     0.652435    0.579828     0.670813     0.620659    0.604378   
6        1     0.854179    0.707334     0.867603     0.825738    0.720088   
9        1     0.952163    0.668980     0.913361     0.911859    0.685419   
12       0     0.786491    0.598958     0.756578     0.779798    0.535406   

    Lgbt_top2_2  
0      0.613949  
3      0.633014  
6      0.766519  
9      0.909227  
12     0.844650  
# # # # # # # # # # 
0.564884637958
0.338279456818
0.622305394552
0.490382524793
0.440013786194
# # # # # # # # # # 

in model: Lgbt_top2_2  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.796814	valid_1's auc: 0.67185
[20]	training's auc: 0.816813	valid_1's auc: 0.679376
[30]	training's auc: 0.826659	valid_1's auc: 0.681374
[40]	training's auc: 0.830571	valid_1's auc: 0.681389
[50]	training's auc: 0.835643	valid_1's auc: 0.681619
[60]	training's auc: 0.839746	valid_1's auc: 0.681566
[70]	training's auc: 0.843636	valid_1's auc: 0.681663
[80]	training's auc: 0.846386	valid_1's auc: 0.681308
[90]	training's auc: 0.848591	valid_1's auc: 0.681166
Early stopping, best iteration is:
[48]	training's auc: 0.834992	valid_1's auc: 0.681792
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2  \
1        1     0.640505    0.590727     0.620106     0.648402    0.601203   
4        1     0.561655    0.519061     0.593088     0.602723    0.707695   
7        1     0.708383    0.523621     0.586171     0.624141    0.698430   
10       1     0.845610    0.639616     0.872397     0.857114    0.764084   
13       1     0.767405    0.683715     0.831088     0.780171    0.695359   

    Lgbt_top2_2  
1      0.626855  
4      0.793086  
7      0.843540  
10     0.910000  
13     0.831979  
# # # # # # # # # # 
1.02702484149
0.721796010893
1.08523308373
0.813469157558
0.99123945245
# # # # # # # # # # 

in model: Lgbt_top2_2  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.796173	valid_1's auc: 0.670883
[20]	training's auc: 0.81583	valid_1's auc: 0.679046
[30]	training's auc: 0.825473	valid_1's auc: 0.68119
[40]	training's auc: 0.829748	valid_1's auc: 0.68127
[50]	training's auc: 0.835636	valid_1's auc: 0.681701
[60]	training's auc: 0.839444	valid_1's auc: 0.681276
[70]	training's auc: 0.842598	valid_1's auc: 0.681297
[80]	training's auc: 0.844865	valid_1's auc: 0.681399
[90]	training's auc: 0.846926	valid_1's auc: 0.681372
Early stopping, best iteration is:
[47]	training's auc: 0.833889	valid_1's auc: 0.68183
- - - - - - - - - - 
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
- - - - - - - - - - 
    target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  Lrf_top2_2  \
2        1     0.709626    0.590464     0.707518     0.595401    0.603777   
5        1     0.745778    0.548480     0.695514     0.673461    0.735620   
8        1     0.926473    0.681419     0.856057     0.925611    0.775460   
11       1     0.937283    0.669170     0.914655     0.922491    0.693642   
14       1     0.928082    0.720894     0.890572     0.902568    0.760462   

    Lgbt_top2_2  
2      0.766016  
5      0.804247  
8      0.922511  
11     0.917865  
14     0.927770  
# # # # # # # # # # 
1.55568735695
1.1519199278
1.61638285352
1.2105126055
1.49883349148
# # # # # # # # # # 
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  \
5606837       0     0.616828    0.715224     0.620657     0.575735   
5606838       0     0.449966    0.623404     0.424477     0.435107   
5606839       1     0.575231    0.411250     0.553390     0.489196   
5606840       1     0.444204    0.387960     0.415569     0.396231   
5606841       0     0.491779    0.651402     0.488834     0.493533   

         Lrf_top2_2  Lgbt_top2_2  
5606837    0.741629     0.518562  
5606838    0.595414     0.383973  
5606839    0.384685     0.538794  
5606840    0.323682     0.403504  
5606841    0.685985     0.499611  
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  \
5606837       0     0.616828    0.715224     0.620657     0.575735   
5606838       0     0.449966    0.623404     0.424477     0.435107   
5606839       1     0.575231    0.411250     0.553390     0.489196   
5606840       1     0.444204    0.387960     0.415569     0.396231   
5606841       0     0.491779    0.651402     0.488834     0.493533   

         Lrf_top2_2  Lgbt_top2_2  
5606837    0.741629     0.518562  
5606838    0.595414     0.383973  
5606839    0.384685     0.538794  
5606840    0.323682     0.403504  
5606841    0.685985     0.499611  
         target  Lgos_top2_1  Lrf_top2_1  Ldrt_top2_2  Lgos_top2_2  \
7377413       1     0.628210    0.625319     0.632481     0.601188   
7377414       0     0.654625    0.515857     0.483335     0.641492   
7377415       1     0.896764    0.726818     0.801889     0.870341   
7377416       1     0.542560    0.472766     0.538165     0.559190   
7377417       1     0.544264    0.471812     0.534216     0.525836   

         Lrf_top2_2  Lgbt_top2_2  
7377413    0.654915     0.710206  
7377414    0.584998     0.627111  
7377415    0.763713     0.856788  
7377416    0.664872     0.787356  
7377417    0.657095     0.763936  
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
Lgos_top2_1    float64
Lrf_top2_1     float64
Ldrt_top2_2    float64
Lgos_top2_2    float64
Lrf_top2_2     float64
Lgbt_top2_2    float64
dtype: object
number of columns: 7
number of data: 1868946
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
Lgos_top2_1    float64
Lrf_top2_1     float64
Ldrt_top2_2    float64
Lgos_top2_2    float64
Lrf_top2_2     float64
Lgbt_top2_2    float64
dtype: object
number of columns: 7
number of data: 1868946
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
Lgos_top2_1    float64
Lrf_top2_1     float64
Ldrt_top2_2    float64
Lgos_top2_2    float64
Lrf_top2_2     float64
Lgbt_top2_2    float64
dtype: object
number of columns: 7
number of data: 1868945
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target           uint8
Lgos_top2_1    float64
Lrf_top2_1     float64
Ldrt_top2_2    float64
Lgos_top2_2    float64
Lrf_top2_2     float64
Lgbt_top2_2    float64
dtype: object
number of columns: 7
number of data: 1868945
<<<<<<<<<<<<<<<<<<<<
saving DONE.

[timer]: complete in 158m 58s

Process finished with exit code 0
'''
