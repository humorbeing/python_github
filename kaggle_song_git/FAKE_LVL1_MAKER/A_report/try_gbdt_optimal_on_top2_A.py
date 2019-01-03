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
h2o.init(nthreads=-1)
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
'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/MAKER/try_v1001.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
  return f(*args, **kwds)
Checking whether there is an H2O instance running at http://localhost:54321..... not found.
Attempting to start a local H2O server...
  Java Version: java version "1.8.0_151"; Java(TM) SE Runtime Environment (build 1.8.0_151-b12); Java HotSpot(TM) 64-Bit Server VM (build 25.151-b12, mixed mode)
  Starting server from /usr/local/lib/python3.5/dist-packages/h2o/backend/bin/h2o.jar
  Ice root: /tmp/tmpzu4i2uga
  JVM stdout: /tmp/tmpzu4i2uga/h2o_ray_started_from_python.out
  JVM stderr: /tmp/tmpzu4i2uga/h2o_ray_started_from_python.err
  Server is running at http://127.0.0.1:54321
Connecting to H2O server at http://127.0.0.1:54321... successful.
--------------------------  ----------------------------------------
H2O cluster uptime:         02 secs
H2O cluster version:        3.16.0.2
H2O cluster version age:    14 days, 22 hours and 58 minutes
H2O cluster name:           H2O_from_python_ray_fzfuww
H2O cluster total nodes:    1
H2O cluster free memory:    2.596 Gb
H2O cluster total cores:    4
H2O cluster allowed cores:  4
H2O cluster status:         accepting new members, healthy
H2O connection url:         http://127.0.0.1:54321
H2O connection proxy:
H2O internal security:      False
H2O API Extensions:         XGBoost, Algos, AutoML, Core V3, Core V4
Python version:             3.5.2 final
--------------------------  ----------------------------------------

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
[10]	training's auc: 0.795636	valid_1's auc: 0.663637
[20]	training's auc: 0.803214	valid_1's auc: 0.666225
[30]	training's auc: 0.808924	valid_1's auc: 0.668309
[40]	training's auc: 0.813626	valid_1's auc: 0.669764
[50]	training's auc: 0.819342	valid_1's auc: 0.671667
[60]	training's auc: 0.824183	valid_1's auc: 0.673081
[70]	training's auc: 0.82827	valid_1's auc: 0.67436
[80]	training's auc: 0.830719	valid_1's auc: 0.675153
[90]	training's auc: 0.833309	valid_1's auc: 0.676045
[100]	training's auc: 0.83591	valid_1's auc: 0.677157
[110]	training's auc: 0.838022	valid_1's auc: 0.677957
[120]	training's auc: 0.839739	valid_1's auc: 0.678707
[130]	training's auc: 0.841	valid_1's auc: 0.679254
[140]	training's auc: 0.842205	valid_1's auc: 0.679784
[150]	training's auc: 0.843275	valid_1's auc: 0.680241
[160]	training's auc: 0.844292	valid_1's auc: 0.680642
[170]	training's auc: 0.84529	valid_1's auc: 0.681014
[180]	training's auc: 0.846168	valid_1's auc: 0.681337
[190]	training's auc: 0.846978	valid_1's auc: 0.681594
[200]	training's auc: 0.847815	valid_1's auc: 0.681815
[210]	training's auc: 0.848619	valid_1's auc: 0.682003
[220]	training's auc: 0.849311	valid_1's auc: 0.682141
[230]	training's auc: 0.849978	valid_1's auc: 0.682305
[240]	training's auc: 0.85066	valid_1's auc: 0.682445
[250]	training's auc: 0.851381	valid_1's auc: 0.68262
[260]	training's auc: 0.852132	valid_1's auc: 0.682796
[270]	training's auc: 0.852794	valid_1's auc: 0.682961
[280]	training's auc: 0.853382	valid_1's auc: 0.683086
[290]	training's auc: 0.854011	valid_1's auc: 0.683166
[300]	training's auc: 0.854587	valid_1's auc: 0.683249
[310]	training's auc: 0.855115	valid_1's auc: 0.683303
[320]	training's auc: 0.855725	valid_1's auc: 0.683441
[330]	training's auc: 0.856271	valid_1's auc: 0.683558
[340]	training's auc: 0.856796	valid_1's auc: 0.683627
[350]	training's auc: 0.857375	valid_1's auc: 0.683694
[360]	training's auc: 0.857939	valid_1's auc: 0.683789
[370]	training's auc: 0.858472	valid_1's auc: 0.683831
[380]	training's auc: 0.859037	valid_1's auc: 0.683917
[390]	training's auc: 0.859652	valid_1's auc: 0.684072
[400]	training's auc: 0.860197	valid_1's auc: 0.684155
[410]	training's auc: 0.860795	valid_1's auc: 0.684223
[420]	training's auc: 0.861301	valid_1's auc: 0.684328
[430]	training's auc: 0.861764	valid_1's auc: 0.684356
[440]	training's auc: 0.862257	valid_1's auc: 0.684422
[450]	training's auc: 0.862768	valid_1's auc: 0.68447
[460]	training's auc: 0.863233	valid_1's auc: 0.684483
[470]	training's auc: 0.863714	valid_1's auc: 0.684532
[480]	training's auc: 0.864188	valid_1's auc: 0.684585
[490]	training's auc: 0.864667	valid_1's auc: 0.684642
[500]	training's auc: 0.865107	valid_1's auc: 0.684671
[510]	training's auc: 0.86554	valid_1's auc: 0.684721
[520]	training's auc: 0.866004	valid_1's auc: 0.684751
[530]	training's auc: 0.866482	valid_1's auc: 0.684784
[540]	training's auc: 0.866892	valid_1's auc: 0.684843
[550]	training's auc: 0.86733	valid_1's auc: 0.684851
[560]	training's auc: 0.867775	valid_1's auc: 0.684879
[570]	training's auc: 0.868222	valid_1's auc: 0.684916
[580]	training's auc: 0.868712	valid_1's auc: 0.684961
[590]	training's auc: 0.869092	valid_1's auc: 0.685014
[600]	training's auc: 0.869546	valid_1's auc: 0.685033
[610]	training's auc: 0.869919	valid_1's auc: 0.685072
[620]	training's auc: 0.870255	valid_1's auc: 0.685131
[630]	training's auc: 0.870652	valid_1's auc: 0.685147
[640]	training's auc: 0.871031	valid_1's auc: 0.685132
[650]	training's auc: 0.871436	valid_1's auc: 0.685163
[660]	training's auc: 0.871797	valid_1's auc: 0.68523
[670]	training's auc: 0.872265	valid_1's auc: 0.685247
[680]	training's auc: 0.872643	valid_1's auc: 0.685251
[690]	training's auc: 0.873028	valid_1's auc: 0.685268
[700]	training's auc: 0.873402	valid_1's auc: 0.685278
[710]	training's auc: 0.873784	valid_1's auc: 0.685299
[720]	training's auc: 0.87419	valid_1's auc: 0.68537
[730]	training's auc: 0.874553	valid_1's auc: 0.685378
[740]	training's auc: 0.874939	valid_1's auc: 0.68538
[750]	training's auc: 0.87527	valid_1's auc: 0.685379
[760]	training's auc: 0.875655	valid_1's auc: 0.6854
[770]	training's auc: 0.875972	valid_1's auc: 0.685413
[780]	training's auc: 0.876295	valid_1's auc: 0.68542
[790]	training's auc: 0.876656	valid_1's auc: 0.68544
[800]	training's auc: 0.876979	valid_1's auc: 0.685441
[810]	training's auc: 0.877356	valid_1's auc: 0.68546
[820]	training's auc: 0.877664	valid_1's auc: 0.685472
[830]	training's auc: 0.877988	valid_1's auc: 0.685498
[840]	training's auc: 0.878368	valid_1's auc: 0.685522
[850]	training's auc: 0.87869	valid_1's auc: 0.685533
[860]	training's auc: 0.878994	valid_1's auc: 0.685544
[870]	training's auc: 0.879344	valid_1's auc: 0.685585
[880]	training's auc: 0.879665	valid_1's auc: 0.685589
[890]	training's auc: 0.879967	valid_1's auc: 0.685595
[900]	training's auc: 0.880278	valid_1's auc: 0.685596
[910]	training's auc: 0.880598	valid_1's auc: 0.685613
[920]	training's auc: 0.880924	valid_1's auc: 0.685625
[930]	training's auc: 0.881209	valid_1's auc: 0.685636
[940]	training's auc: 0.881498	valid_1's auc: 0.685652
[950]	training's auc: 0.881767	valid_1's auc: 0.685669
[960]	training's auc: 0.882078	valid_1's auc: 0.685677
[970]	training's auc: 0.882371	valid_1's auc: 0.68569
[980]	training's auc: 0.882654	valid_1's auc: 0.685696
[990]	training's auc: 0.882928	valid_1's auc: 0.685704
[1000]	training's auc: 0.88322	valid_1's auc: 0.685736
[1010]	training's auc: 0.883528	valid_1's auc: 0.685736
[1020]	training's auc: 0.883809	valid_1's auc: 0.685731
[1030]	training's auc: 0.884079	valid_1's auc: 0.685738
[1040]	training's auc: 0.884358	valid_1's auc: 0.685744
[1050]	training's auc: 0.884631	valid_1's auc: 0.685747
[1060]	training's auc: 0.884902	valid_1's auc: 0.685752
[1070]	training's auc: 0.885174	valid_1's auc: 0.685759
[1080]	training's auc: 0.885468	valid_1's auc: 0.685784
[1090]	training's auc: 0.885739	valid_1's auc: 0.685796
[1100]	training's auc: 0.886027	valid_1's auc: 0.685811
[1110]	training's auc: 0.886301	valid_1's auc: 0.685827
[1120]	training's auc: 0.886585	valid_1's auc: 0.685832
[1130]	training's auc: 0.88687	valid_1's auc: 0.685819
[1140]	training's auc: 0.887144	valid_1's auc: 0.685823
[1150]	training's auc: 0.887395	valid_1's auc: 0.685851
[1160]	training's auc: 0.88766	valid_1's auc: 0.685851
[1170]	training's auc: 0.887923	valid_1's auc: 0.685856
[1180]	training's auc: 0.888196	valid_1's auc: 0.685876
[1190]	training's auc: 0.888439	valid_1's auc: 0.685887
[1200]	training's auc: 0.888686	valid_1's auc: 0.685892
[1210]	training's auc: 0.888945	valid_1's auc: 0.685894
[1220]	training's auc: 0.88921	valid_1's auc: 0.685899
[1230]	training's auc: 0.889469	valid_1's auc: 0.685899
[1240]	training's auc: 0.889735	valid_1's auc: 0.685911
[1250]	training's auc: 0.889986	valid_1's auc: 0.685912
[1260]	training's auc: 0.890236	valid_1's auc: 0.685911
[1270]	training's auc: 0.890505	valid_1's auc: 0.685916
[1280]	training's auc: 0.89077	valid_1's auc: 0.685927
[1290]	training's auc: 0.891031	valid_1's auc: 0.685935
[1300]	training's auc: 0.891297	valid_1's auc: 0.685936
[1310]	training's auc: 0.89155	valid_1's auc: 0.685937
[1320]	training's auc: 0.891793	valid_1's auc: 0.685942
[1330]	training's auc: 0.892053	valid_1's auc: 0.685938
[1340]	training's auc: 0.892298	valid_1's auc: 0.685935
[1350]	training's auc: 0.89255	valid_1's auc: 0.685945
[1360]	training's auc: 0.892803	valid_1's auc: 0.68595
[1370]	training's auc: 0.893044	valid_1's auc: 0.686001
[1380]	training's auc: 0.893305	valid_1's auc: 0.686008
[1390]	training's auc: 0.893557	valid_1's auc: 0.686002
[1400]	training's auc: 0.89379	valid_1's auc: 0.686001
[1410]	training's auc: 0.894046	valid_1's auc: 0.686
[1420]	training's auc: 0.894284	valid_1's auc: 0.68601
[1430]	training's auc: 0.894524	valid_1's auc: 0.68601
[1440]	training's auc: 0.894749	valid_1's auc: 0.686015
[1450]	training's auc: 0.894998	valid_1's auc: 0.68602
[1460]	training's auc: 0.895229	valid_1's auc: 0.686019
[1470]	training's auc: 0.895462	valid_1's auc: 0.686018
[1480]	training's auc: 0.895692	valid_1's auc: 0.68602
[1490]	training's auc: 0.895921	valid_1's auc: 0.686029
[1500]	training's auc: 0.896167	valid_1's auc: 0.686028
[1510]	training's auc: 0.896382	valid_1's auc: 0.686015
[1520]	training's auc: 0.896606	valid_1's auc: 0.686016
[1530]	training's auc: 0.896841	valid_1's auc: 0.686019
Early stopping, best iteration is:
[1488]	training's auc: 0.895875	valid_1's auc: 0.686033
Traceback (most recent call last):
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/MAKER/try_v1001.py", line 77, in <module>
  File "/media/ray/SSD/workspace/python/projects/kaggle_song_git/MAKER/G_models.py", line 221, in gbdt_optimal_on_top2
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 1766, in predict
    return predictor.predict(data, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, is_reshape)
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 387, in predict
    data = _data_from_pandas(data, None, None, self.pandas_categorical)[0]
  File "/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py", line 235, in _data_from_pandas
    raise ValueError('train and valid dataset categorical_feature do not match.')
ValueError: train and valid dataset categorical_feature do not match.
H2O session _sid_aaa2 closed.

Process finished with exit code 1
'''