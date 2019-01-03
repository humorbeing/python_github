import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from models import *
# import h2o
# from sklearn.metrics import roc_auc_score
# from h2o.estimators.random_forest import H2ORandomForestEstimator
# from h2o.estimators.gbm import H2OGradientBoostingEstimator
# from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# from h2o.estimators.glm import H2OGeneralizedLinearEstimator


since = time.time()
since = time.time()
# h2o.init(nthreads=-1)
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'final_train_play.csv'
df = read_df(load_name)

show_df(df)

# save_me = True
save_me = False
if save_me:
    save_df(df)

dfs, val = fake_df(df)
del df
K = 3
dfs = divide_df(dfs, K)
dcs = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dcs.append(dc)

vc = pd.DataFrame()
vc['target'] = val['target']


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


dcs, vc, r = gbdt_optimal_on_top2(K, dfs, dcs, val, vc)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(val['target'], vc[r]))


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('done')
'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/FAKE_MAKER/try_gbdt_optimal_on_top2.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
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

in model: gbdt_optimal_on_top2  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795781	valid_1's auc: 0.664135
[20]	training's auc: 0.80379	valid_1's auc: 0.666821
[30]	training's auc: 0.808765	valid_1's auc: 0.668567
[40]	training's auc: 0.813677	valid_1's auc: 0.669928
[50]	training's auc: 0.819262	valid_1's auc: 0.671719
[60]	training's auc: 0.823845	valid_1's auc: 0.673002
[70]	training's auc: 0.828028	valid_1's auc: 0.674372
[80]	training's auc: 0.830778	valid_1's auc: 0.675212
[90]	training's auc: 0.833453	valid_1's auc: 0.676162
[100]	training's auc: 0.836044	valid_1's auc: 0.677235
[110]	training's auc: 0.838119	valid_1's auc: 0.677984
[120]	training's auc: 0.839823	valid_1's auc: 0.678841
[130]	training's auc: 0.841125	valid_1's auc: 0.679419
[140]	training's auc: 0.842265	valid_1's auc: 0.679933
[150]	training's auc: 0.84329	valid_1's auc: 0.68039
[160]	training's auc: 0.844426	valid_1's auc: 0.680844
[170]	training's auc: 0.845387	valid_1's auc: 0.681208
[180]	training's auc: 0.846312	valid_1's auc: 0.681525
[190]	training's auc: 0.847199	valid_1's auc: 0.681801
[200]	training's auc: 0.847954	valid_1's auc: 0.682054
[210]	training's auc: 0.848892	valid_1's auc: 0.682288
[220]	training's auc: 0.84959	valid_1's auc: 0.682479
[230]	training's auc: 0.850285	valid_1's auc: 0.682679
[240]	training's auc: 0.851002	valid_1's auc: 0.682814
[250]	training's auc: 0.851636	valid_1's auc: 0.682926
[260]	training's auc: 0.852335	valid_1's auc: 0.6831
[270]	training's auc: 0.852998	valid_1's auc: 0.68326
[280]	training's auc: 0.8536	valid_1's auc: 0.683381
[290]	training's auc: 0.854233	valid_1's auc: 0.683505
[300]	training's auc: 0.854821	valid_1's auc: 0.683649
[310]	training's auc: 0.855366	valid_1's auc: 0.683743
[320]	training's auc: 0.856053	valid_1's auc: 0.683897
[330]	training's auc: 0.856689	valid_1's auc: 0.683966
[340]	training's auc: 0.857265	valid_1's auc: 0.68407
[350]	training's auc: 0.857855	valid_1's auc: 0.684141
[360]	training's auc: 0.85844	valid_1's auc: 0.684229
[370]	training's auc: 0.858989	valid_1's auc: 0.684306
[380]	training's auc: 0.859525	valid_1's auc: 0.684388
[390]	training's auc: 0.86008	valid_1's auc: 0.684479
[400]	training's auc: 0.860671	valid_1's auc: 0.684592
[410]	training's auc: 0.861221	valid_1's auc: 0.684683
[420]	training's auc: 0.861715	valid_1's auc: 0.684801
[430]	training's auc: 0.862187	valid_1's auc: 0.68484
[440]	training's auc: 0.862645	valid_1's auc: 0.684893
[450]	training's auc: 0.863123	valid_1's auc: 0.684972
[460]	training's auc: 0.86356	valid_1's auc: 0.685002
[470]	training's auc: 0.864012	valid_1's auc: 0.685042
[480]	training's auc: 0.864502	valid_1's auc: 0.685097
[490]	training's auc: 0.864951	valid_1's auc: 0.685123
[500]	training's auc: 0.865371	valid_1's auc: 0.685133
[510]	training's auc: 0.865812	valid_1's auc: 0.685182
[520]	training's auc: 0.866294	valid_1's auc: 0.685203
[530]	training's auc: 0.866765	valid_1's auc: 0.685226
[540]	training's auc: 0.867264	valid_1's auc: 0.685319
[550]	training's auc: 0.867717	valid_1's auc: 0.685327
[560]	training's auc: 0.868139	valid_1's auc: 0.685388
[570]	training's auc: 0.868564	valid_1's auc: 0.68542
[580]	training's auc: 0.869024	valid_1's auc: 0.685438
[590]	training's auc: 0.869395	valid_1's auc: 0.685455
[600]	training's auc: 0.869811	valid_1's auc: 0.685532
[610]	training's auc: 0.870223	valid_1's auc: 0.685566
[620]	training's auc: 0.870568	valid_1's auc: 0.685597
[630]	training's auc: 0.87106	valid_1's auc: 0.685627
[640]	training's auc: 0.871455	valid_1's auc: 0.685662
[650]	training's auc: 0.871824	valid_1's auc: 0.685676
[660]	training's auc: 0.87221	valid_1's auc: 0.685675
[670]	training's auc: 0.872658	valid_1's auc: 0.685698
[680]	training's auc: 0.873076	valid_1's auc: 0.685735
[690]	training's auc: 0.873436	valid_1's auc: 0.685739
[700]	training's auc: 0.873811	valid_1's auc: 0.685748
[710]	training's auc: 0.874173	valid_1's auc: 0.685774
[720]	training's auc: 0.874524	valid_1's auc: 0.685769
[730]	training's auc: 0.87487	valid_1's auc: 0.685784
[740]	training's auc: 0.875267	valid_1's auc: 0.685802
[750]	training's auc: 0.875595	valid_1's auc: 0.685809
[760]	training's auc: 0.875933	valid_1's auc: 0.685817
[770]	training's auc: 0.87625	valid_1's auc: 0.685813
[780]	training's auc: 0.876596	valid_1's auc: 0.685831
[790]	training's auc: 0.876986	valid_1's auc: 0.685859
[800]	training's auc: 0.8773	valid_1's auc: 0.685876
[810]	training's auc: 0.877645	valid_1's auc: 0.685884
[820]	training's auc: 0.877956	valid_1's auc: 0.685895
[830]	training's auc: 0.878307	valid_1's auc: 0.685918
[840]	training's auc: 0.878667	valid_1's auc: 0.68596
[850]	training's auc: 0.878975	valid_1's auc: 0.68598
[860]	training's auc: 0.879262	valid_1's auc: 0.685991
[870]	training's auc: 0.879574	valid_1's auc: 0.685998
[880]	training's auc: 0.879868	valid_1's auc: 0.686004
[890]	training's auc: 0.880156	valid_1's auc: 0.686
[900]	training's auc: 0.880457	valid_1's auc: 0.686006
[910]	training's auc: 0.880763	valid_1's auc: 0.686016
[920]	training's auc: 0.881065	valid_1's auc: 0.686007
[930]	training's auc: 0.88136	valid_1's auc: 0.686011
[940]	training's auc: 0.881669	valid_1's auc: 0.68601
[950]	training's auc: 0.881941	valid_1's auc: 0.686016
[960]	training's auc: 0.882248	valid_1's auc: 0.686006
Early stopping, best iteration is:
[911]	training's auc: 0.880792	valid_1's auc: 0.686019
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
    target  gbdt_optimal_on_top2
0        1              0.604521
3        1              0.588392
6        1              0.815846
9        1              0.887332
12       0              0.736547

in model: gbdt_optimal_on_top2  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795182	valid_1's auc: 0.663113
[20]	training's auc: 0.80359	valid_1's auc: 0.666159
[30]	training's auc: 0.80906	valid_1's auc: 0.66816
[40]	training's auc: 0.813745	valid_1's auc: 0.669574
[50]	training's auc: 0.819186	valid_1's auc: 0.671254
[60]	training's auc: 0.823935	valid_1's auc: 0.672727
[70]	training's auc: 0.828038	valid_1's auc: 0.674072
[80]	training's auc: 0.830644	valid_1's auc: 0.674922
[90]	training's auc: 0.833067	valid_1's auc: 0.675798
[100]	training's auc: 0.835822	valid_1's auc: 0.67693
[110]	training's auc: 0.837956	valid_1's auc: 0.677863
[120]	training's auc: 0.839623	valid_1's auc: 0.678708
[130]	training's auc: 0.840985	valid_1's auc: 0.679432
[140]	training's auc: 0.842002	valid_1's auc: 0.679962
[150]	training's auc: 0.843068	valid_1's auc: 0.680454
[160]	training's auc: 0.844072	valid_1's auc: 0.680867
[170]	training's auc: 0.845021	valid_1's auc: 0.68122
[180]	training's auc: 0.845999	valid_1's auc: 0.681553
[190]	training's auc: 0.846836	valid_1's auc: 0.681846
[200]	training's auc: 0.847593	valid_1's auc: 0.682072
[210]	training's auc: 0.848359	valid_1's auc: 0.682263
[220]	training's auc: 0.84914	valid_1's auc: 0.682524
[230]	training's auc: 0.849875	valid_1's auc: 0.68272
[240]	training's auc: 0.850601	valid_1's auc: 0.68287
[250]	training's auc: 0.851248	valid_1's auc: 0.683044
[260]	training's auc: 0.85197	valid_1's auc: 0.68321
[270]	training's auc: 0.852622	valid_1's auc: 0.683336
[280]	training's auc: 0.853254	valid_1's auc: 0.683472
[290]	training's auc: 0.853926	valid_1's auc: 0.683649
[300]	training's auc: 0.854529	valid_1's auc: 0.683758
[310]	training's auc: 0.855072	valid_1's auc: 0.683854
[320]	training's auc: 0.855749	valid_1's auc: 0.684042
[330]	training's auc: 0.85632	valid_1's auc: 0.684135
[340]	training's auc: 0.856897	valid_1's auc: 0.684207
[350]	training's auc: 0.857448	valid_1's auc: 0.684332
[360]	training's auc: 0.858004	valid_1's auc: 0.684398
[370]	training's auc: 0.858635	valid_1's auc: 0.68452
[380]	training's auc: 0.859225	valid_1's auc: 0.68459
[390]	training's auc: 0.859764	valid_1's auc: 0.684687
[400]	training's auc: 0.860327	valid_1's auc: 0.684771
[410]	training's auc: 0.860846	valid_1's auc: 0.684834
[420]	training's auc: 0.861323	valid_1's auc: 0.684837
[430]	training's auc: 0.861845	valid_1's auc: 0.68489
[440]	training's auc: 0.862286	valid_1's auc: 0.684915
[450]	training's auc: 0.862763	valid_1's auc: 0.684995
[460]	training's auc: 0.863222	valid_1's auc: 0.685028
[470]	training's auc: 0.863703	valid_1's auc: 0.685064
[480]	training's auc: 0.864205	valid_1's auc: 0.685147
[490]	training's auc: 0.864684	valid_1's auc: 0.685194
[500]	training's auc: 0.86512	valid_1's auc: 0.685239
[510]	training's auc: 0.865562	valid_1's auc: 0.685296
[520]	training's auc: 0.866064	valid_1's auc: 0.685324
[530]	training's auc: 0.866532	valid_1's auc: 0.685415
[540]	training's auc: 0.86702	valid_1's auc: 0.685455
[550]	training's auc: 0.867482	valid_1's auc: 0.685491
[560]	training's auc: 0.867912	valid_1's auc: 0.685531
[570]	training's auc: 0.868395	valid_1's auc: 0.685566
[580]	training's auc: 0.868885	valid_1's auc: 0.685617
[590]	training's auc: 0.869305	valid_1's auc: 0.68567
[600]	training's auc: 0.869722	valid_1's auc: 0.685709
[610]	training's auc: 0.87011	valid_1's auc: 0.685741
[620]	training's auc: 0.87047	valid_1's auc: 0.685774
[630]	training's auc: 0.870898	valid_1's auc: 0.685799
[640]	training's auc: 0.871327	valid_1's auc: 0.685844
[650]	training's auc: 0.871774	valid_1's auc: 0.68589
[660]	training's auc: 0.872184	valid_1's auc: 0.685971
[670]	training's auc: 0.872661	valid_1's auc: 0.686025
[680]	training's auc: 0.873089	valid_1's auc: 0.686025
[690]	training's auc: 0.873462	valid_1's auc: 0.686027
[700]	training's auc: 0.873836	valid_1's auc: 0.686057
[710]	training's auc: 0.874209	valid_1's auc: 0.686073
[720]	training's auc: 0.874561	valid_1's auc: 0.686104
[730]	training's auc: 0.874894	valid_1's auc: 0.686126
[740]	training's auc: 0.875284	valid_1's auc: 0.686139
[750]	training's auc: 0.875606	valid_1's auc: 0.686148
[760]	training's auc: 0.875992	valid_1's auc: 0.686161
[770]	training's auc: 0.876312	valid_1's auc: 0.686163
[780]	training's auc: 0.876643	valid_1's auc: 0.686171
[790]	training's auc: 0.876983	valid_1's auc: 0.686168
[800]	training's auc: 0.877326	valid_1's auc: 0.68619
[810]	training's auc: 0.877688	valid_1's auc: 0.686203
[820]	training's auc: 0.878005	valid_1's auc: 0.68621
[830]	training's auc: 0.878351	valid_1's auc: 0.686336
[840]	training's auc: 0.878675	valid_1's auc: 0.686333
[850]	training's auc: 0.878978	valid_1's auc: 0.686343
[860]	training's auc: 0.879271	valid_1's auc: 0.686349
[870]	training's auc: 0.879591	valid_1's auc: 0.686372
[880]	training's auc: 0.879892	valid_1's auc: 0.686378
[890]	training's auc: 0.880194	valid_1's auc: 0.686381
[900]	training's auc: 0.880502	valid_1's auc: 0.68639
[910]	training's auc: 0.880842	valid_1's auc: 0.686408
[920]	training's auc: 0.881135	valid_1's auc: 0.686432
[930]	training's auc: 0.881428	valid_1's auc: 0.686451
[940]	training's auc: 0.881713	valid_1's auc: 0.686449
[950]	training's auc: 0.881979	valid_1's auc: 0.686444
[960]	training's auc: 0.882325	valid_1's auc: 0.686463
[970]	training's auc: 0.88263	valid_1's auc: 0.686467
[980]	training's auc: 0.88291	valid_1's auc: 0.686471
[990]	training's auc: 0.883186	valid_1's auc: 0.686472
[1000]	training's auc: 0.883487	valid_1's auc: 0.686498
[1010]	training's auc: 0.883784	valid_1's auc: 0.686507
[1020]	training's auc: 0.884064	valid_1's auc: 0.686503
[1030]	training's auc: 0.884348	valid_1's auc: 0.686525
[1040]	training's auc: 0.884627	valid_1's auc: 0.686517
[1050]	training's auc: 0.884911	valid_1's auc: 0.686522
[1060]	training's auc: 0.885183	valid_1's auc: 0.686536
[1070]	training's auc: 0.885463	valid_1's auc: 0.686545
[1080]	training's auc: 0.885757	valid_1's auc: 0.686543
[1090]	training's auc: 0.886052	valid_1's auc: 0.68654
[1100]	training's auc: 0.886321	valid_1's auc: 0.686538
[1110]	training's auc: 0.886611	valid_1's auc: 0.686547
[1120]	training's auc: 0.886896	valid_1's auc: 0.686564
[1130]	training's auc: 0.887169	valid_1's auc: 0.68657
[1140]	training's auc: 0.887444	valid_1's auc: 0.686568
[1150]	training's auc: 0.887692	valid_1's auc: 0.686562
[1160]	training's auc: 0.887983	valid_1's auc: 0.686578
[1170]	training's auc: 0.888251	valid_1's auc: 0.686579
[1180]	training's auc: 0.888526	valid_1's auc: 0.686585
[1190]	training's auc: 0.88877	valid_1's auc: 0.686591
[1200]	training's auc: 0.889008	valid_1's auc: 0.686604
[1210]	training's auc: 0.889263	valid_1's auc: 0.686602
[1220]	training's auc: 0.889515	valid_1's auc: 0.686611
[1230]	training's auc: 0.88977	valid_1's auc: 0.686622
[1240]	training's auc: 0.890033	valid_1's auc: 0.686619
[1250]	training's auc: 0.89028	valid_1's auc: 0.686619
[1260]	training's auc: 0.890531	valid_1's auc: 0.686626
[1270]	training's auc: 0.890791	valid_1's auc: 0.686636
[1280]	training's auc: 0.891053	valid_1's auc: 0.686635
[1290]	training's auc: 0.891299	valid_1's auc: 0.686644
[1300]	training's auc: 0.891579	valid_1's auc: 0.686648
[1310]	training's auc: 0.891848	valid_1's auc: 0.686659
[1320]	training's auc: 0.892084	valid_1's auc: 0.686667
[1330]	training's auc: 0.892338	valid_1's auc: 0.686677
[1340]	training's auc: 0.892576	valid_1's auc: 0.686695
[1350]	training's auc: 0.892826	valid_1's auc: 0.686707
[1360]	training's auc: 0.893069	valid_1's auc: 0.686685
[1370]	training's auc: 0.893311	valid_1's auc: 0.686678
[1380]	training's auc: 0.893565	valid_1's auc: 0.686674
[1390]	training's auc: 0.89381	valid_1's auc: 0.686672
[1400]	training's auc: 0.894042	valid_1's auc: 0.686705
[1410]	training's auc: 0.894301	valid_1's auc: 0.686704
[1420]	training's auc: 0.894542	valid_1's auc: 0.686709
[1430]	training's auc: 0.894778	valid_1's auc: 0.686711
[1440]	training's auc: 0.895005	valid_1's auc: 0.686717
Early stopping, best iteration is:
[1394]	training's auc: 0.893904	valid_1's auc: 0.686722
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
    target  gbdt_optimal_on_top2
1        1              0.724402
4        1              0.757583
7        1              0.782424
10       1              0.956520
13       1              0.881529

in model: gbdt_optimal_on_top2  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.795598	valid_1's auc: 0.663787
[20]	training's auc: 0.803768	valid_1's auc: 0.666662
[30]	training's auc: 0.809002	valid_1's auc: 0.668559
[40]	training's auc: 0.813828	valid_1's auc: 0.670122
[50]	training's auc: 0.819389	valid_1's auc: 0.671872
[60]	training's auc: 0.824191	valid_1's auc: 0.673239
[70]	training's auc: 0.828309	valid_1's auc: 0.674515
[80]	training's auc: 0.830954	valid_1's auc: 0.675231
[90]	training's auc: 0.833613	valid_1's auc: 0.676329
[100]	training's auc: 0.835957	valid_1's auc: 0.677308
[110]	training's auc: 0.838117	valid_1's auc: 0.678172
[120]	training's auc: 0.839801	valid_1's auc: 0.678948
[130]	training's auc: 0.84122	valid_1's auc: 0.679592
[140]	training's auc: 0.842325	valid_1's auc: 0.680085
[150]	training's auc: 0.843387	valid_1's auc: 0.680491
[160]	training's auc: 0.844281	valid_1's auc: 0.680879
[170]	training's auc: 0.845228	valid_1's auc: 0.681225
[180]	training's auc: 0.846003	valid_1's auc: 0.681511
[190]	training's auc: 0.846841	valid_1's auc: 0.681793
[200]	training's auc: 0.84765	valid_1's auc: 0.682006
[210]	training's auc: 0.848533	valid_1's auc: 0.68223
[220]	training's auc: 0.849256	valid_1's auc: 0.682443
[230]	training's auc: 0.850017	valid_1's auc: 0.682664
[240]	training's auc: 0.850744	valid_1's auc: 0.682821
[250]	training's auc: 0.851416	valid_1's auc: 0.682963
[260]	training's auc: 0.852094	valid_1's auc: 0.683068
[270]	training's auc: 0.852754	valid_1's auc: 0.683173
[280]	training's auc: 0.853465	valid_1's auc: 0.683306
[290]	training's auc: 0.854108	valid_1's auc: 0.683403
[300]	training's auc: 0.854773	valid_1's auc: 0.683514
[310]	training's auc: 0.855318	valid_1's auc: 0.683605
[320]	training's auc: 0.855939	valid_1's auc: 0.683715
[330]	training's auc: 0.856489	valid_1's auc: 0.683779
[340]	training's auc: 0.85709	valid_1's auc: 0.683882
[350]	training's auc: 0.85768	valid_1's auc: 0.683987
[360]	training's auc: 0.858253	valid_1's auc: 0.684049
[370]	training's auc: 0.858763	valid_1's auc: 0.684121
[380]	training's auc: 0.859319	valid_1's auc: 0.684172
[390]	training's auc: 0.859928	valid_1's auc: 0.684252
[400]	training's auc: 0.860469	valid_1's auc: 0.684321
[410]	training's auc: 0.860988	valid_1's auc: 0.684431
[420]	training's auc: 0.861448	valid_1's auc: 0.684463
[430]	training's auc: 0.861946	valid_1's auc: 0.6845
[440]	training's auc: 0.862409	valid_1's auc: 0.684561
[450]	training's auc: 0.862921	valid_1's auc: 0.684597
[460]	training's auc: 0.863367	valid_1's auc: 0.684615
[470]	training's auc: 0.863862	valid_1's auc: 0.684676
[480]	training's auc: 0.864279	valid_1's auc: 0.68469
[490]	training's auc: 0.864751	valid_1's auc: 0.684728
[500]	training's auc: 0.865222	valid_1's auc: 0.684744
[510]	training's auc: 0.865665	valid_1's auc: 0.684792
[520]	training's auc: 0.866091	valid_1's auc: 0.68479
[530]	training's auc: 0.866591	valid_1's auc: 0.68481
[540]	training's auc: 0.867028	valid_1's auc: 0.68485
[550]	training's auc: 0.867448	valid_1's auc: 0.684869
[560]	training's auc: 0.867903	valid_1's auc: 0.684927
[570]	training's auc: 0.868349	valid_1's auc: 0.684939
[580]	training's auc: 0.868794	valid_1's auc: 0.684955
[590]	training's auc: 0.869175	valid_1's auc: 0.684976
[600]	training's auc: 0.86965	valid_1's auc: 0.685054
[610]	training's auc: 0.870034	valid_1's auc: 0.685102
[620]	training's auc: 0.87042	valid_1's auc: 0.685133
[630]	training's auc: 0.870817	valid_1's auc: 0.685141
[640]	training's auc: 0.871238	valid_1's auc: 0.685166
[650]	training's auc: 0.871588	valid_1's auc: 0.685166
[660]	training's auc: 0.871965	valid_1's auc: 0.685204
[670]	training's auc: 0.872447	valid_1's auc: 0.685248
[680]	training's auc: 0.872819	valid_1's auc: 0.685251
[690]	training's auc: 0.873189	valid_1's auc: 0.685266
[700]	training's auc: 0.873601	valid_1's auc: 0.685282
[710]	training's auc: 0.873994	valid_1's auc: 0.685306
[720]	training's auc: 0.874344	valid_1's auc: 0.685316
[730]	training's auc: 0.874697	valid_1's auc: 0.685316
[740]	training's auc: 0.875039	valid_1's auc: 0.685328
[750]	training's auc: 0.87538	valid_1's auc: 0.685337
[760]	training's auc: 0.875724	valid_1's auc: 0.685349
[770]	training's auc: 0.876048	valid_1's auc: 0.685365
[780]	training's auc: 0.876374	valid_1's auc: 0.685368
[790]	training's auc: 0.876724	valid_1's auc: 0.685378
[800]	training's auc: 0.877076	valid_1's auc: 0.685389
[810]	training's auc: 0.877411	valid_1's auc: 0.685371
[820]	training's auc: 0.87774	valid_1's auc: 0.68541
[830]	training's auc: 0.878035	valid_1's auc: 0.685417
[840]	training's auc: 0.878375	valid_1's auc: 0.685422
[850]	training's auc: 0.878681	valid_1's auc: 0.685435
[860]	training's auc: 0.878983	valid_1's auc: 0.685446
[870]	training's auc: 0.879298	valid_1's auc: 0.685441
[880]	training's auc: 0.879594	valid_1's auc: 0.685447
[890]	training's auc: 0.879897	valid_1's auc: 0.685481
[900]	training's auc: 0.88021	valid_1's auc: 0.685484
[910]	training's auc: 0.880523	valid_1's auc: 0.685488
[920]	training's auc: 0.880825	valid_1's auc: 0.685518
[930]	training's auc: 0.881122	valid_1's auc: 0.685526
[940]	training's auc: 0.881412	valid_1's auc: 0.685525
[950]	training's auc: 0.881701	valid_1's auc: 0.685546
[960]	training's auc: 0.88202	valid_1's auc: 0.685552
[970]	training's auc: 0.882342	valid_1's auc: 0.685542
[980]	training's auc: 0.882652	valid_1's auc: 0.685548
[990]	training's auc: 0.882941	valid_1's auc: 0.685555
[1000]	training's auc: 0.883243	valid_1's auc: 0.685567
[1010]	training's auc: 0.883538	valid_1's auc: 0.68558
[1020]	training's auc: 0.883841	valid_1's auc: 0.685596
[1030]	training's auc: 0.884116	valid_1's auc: 0.685616
[1040]	training's auc: 0.884392	valid_1's auc: 0.685619
[1050]	training's auc: 0.88467	valid_1's auc: 0.685616
[1060]	training's auc: 0.884941	valid_1's auc: 0.68561
[1070]	training's auc: 0.885204	valid_1's auc: 0.685604
[1080]	training's auc: 0.885507	valid_1's auc: 0.685632
[1090]	training's auc: 0.88579	valid_1's auc: 0.685628
[1100]	training's auc: 0.886058	valid_1's auc: 0.68563
[1110]	training's auc: 0.886321	valid_1's auc: 0.68564
[1120]	training's auc: 0.886603	valid_1's auc: 0.685654
[1130]	training's auc: 0.886879	valid_1's auc: 0.68565
[1140]	training's auc: 0.887141	valid_1's auc: 0.685644
[1150]	training's auc: 0.887391	valid_1's auc: 0.685724
[1160]	training's auc: 0.887678	valid_1's auc: 0.685729
[1170]	training's auc: 0.887937	valid_1's auc: 0.68573
[1180]	training's auc: 0.888209	valid_1's auc: 0.685734
[1190]	training's auc: 0.888463	valid_1's auc: 0.685743
[1200]	training's auc: 0.888706	valid_1's auc: 0.685743
[1210]	training's auc: 0.888961	valid_1's auc: 0.685743
[1220]	training's auc: 0.889223	valid_1's auc: 0.685747
[1230]	training's auc: 0.889493	valid_1's auc: 0.685755
[1240]	training's auc: 0.88975	valid_1's auc: 0.685751
[1250]	training's auc: 0.889999	valid_1's auc: 0.685747
[1260]	training's auc: 0.890249	valid_1's auc: 0.685736
[1270]	training's auc: 0.890513	valid_1's auc: 0.685744
Early stopping, best iteration is:
[1229]	training's auc: 0.889468	valid_1's auc: 0.685757
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
    target  gbdt_optimal_on_top2
2        1              0.805142
5        1              0.788254
8        1              0.839732
11       1              0.943092
14       1              0.937435
0.690107781227

[timer]: complete in 247m 1s
done

Process finished with exit code 0
'''