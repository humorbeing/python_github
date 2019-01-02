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
import h2o
from sklearn.metrics import roc_auc_score
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# print(v.shape)
since = time.time()
since = time.time()
h2o.init(nthreads=-1)
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'final_train_play.csv'
df = read_df(load_name)
on = [
    'msno',
    'song_id',
    # 'source_system_tab',
    # 'source_screen_name',
    # 'source_type',
    'target',
    # 'genre_ids',
    # 'artist_name',
    # # 'composer',
    # # 'lyricist',
    # # 'language',
    # 'song_year',
    # # 'song_country',
    # # 'rc',
    # # 'top1_in_song',
    # # 'top2_in_song',
    # # 'top3_in_song',
    # # 'membership_days',
    # # 'song_year_int',
    # # 'ISC_top1_in_song',
    # 'ISC_top2_in_song',
    # # 'ISC_top3_in_song',
    # # 'ISC_language',
    # # 'ISCZ_rc',
    # # 'ISCZ_isrc_rest',
    # # 'ISC_song_year',
    # # 'song_length_log10',
    # # 'ISCZ_genre_ids_log10',
    # # 'ISC_artist_name_log10',
    # # 'ISCZ_composer_log10',
    # # 'ISC_lyricist_log10',
    # # 'ISC_song_country_ln',
    # 'ITC_song_id_log10_1',
    # # 'ITC_source_system_tab_log10_1',
    # # 'ITC_source_screen_name_log10_1',
    # # 'ITC_source_type_log10_1',
    # # 'ITC_artist_name_log10_1',
    # # 'ITC_composer_log10_1',
    # # 'ITC_lyricist_log10_1',
    # # 'ITC_song_year_log10_1',
    # # 'ITC_top1_in_song_log10_1',
    # # 'ITC_top2_in_song_log10_1',
    # # 'ITC_top3_in_song_log10_1',
    # 'ITC_msno_log10_1',
    # # 'OinC_msno',
    # 'ITC_language_log10_1',
    # 'OinC_language',
]
df = df[on]
show_df(df)

# save_me = True
save_me = False
if save_me:
    save_df(df)

dfs, val = fake_df(df)
del df
K = 2
dfs = divide_df(dfs, K)
dcs = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dcs.append(dc)

vc = pd.DataFrame()
vc['target'] = val['target']
v = np.zeros(shape=[len(val)])
print(v.shape)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# show_df(vc)
# dcs, vc, r = gbm_1(K, dfs, dcs, val, vc)
dcs, vc, r = dart_1(K, dfs, dcs, val, vc)
dcs, vc, r = cat_1(K, dfs, dcs, val, vc)

dfs = dcs
val = vc
dcs = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dcs.append(dc)

vc = pd.DataFrame()
vc['target'] = val['target']
dcs, vc, r = glm_1(K, dfs, dcs, val, vc)
show_df(vc, detail=True)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(val['target'], vc[r]))


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('done')