import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split
import h2o
from sklearn.metrics import roc_auc_score
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from numpy import inf




since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_me_play.csv'
df = read_df(load_name)
on = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'target',
    # 'genre_ids',
    'artist_name',
    # 'composer',
    # 'lyricist',
    # 'language',
    'song_year',
    # 'song_country',
    # 'rc',
    # 'top1_in_song',
    # 'top2_in_song',
    # 'top3_in_song',
    # 'membership_days',
    # 'song_year_int',
    # 'ISC_top1_in_song',
    'ISC_top2_in_song',
    # 'ISC_top3_in_song',
    # 'ISC_language',
    # 'ISCZ_rc',
    # 'ISCZ_isrc_rest',
    # 'ISC_song_year',
    # 'song_length_log10',
    # 'ISCZ_genre_ids_log10',
    # 'ISC_artist_name_log10',
    # 'ISCZ_composer_log10',
    # 'ISC_lyricist_log10',
    # 'ISC_song_country_ln',
    'ITC_song_id_log10_1',
    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_song_year_log10_1',
    # 'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    # 'ITC_top3_in_song_log10_1',
    'ITC_msno_log10_1',
    # 'OinC_msno',
    # 'ITC_language_log10_1',
    # 'OinC_language',
]
df = df[on]

show_df(df)


# Initialize data
# cat_features = [0,1,2]
train_data = df.drop('target', axis=1)
train_labels = df['target']
# test_data = [["a","b",2,4,6,8],["a","d",1,4,50,60]]
# Initialize CatBoostClassifier

# Get predicted RawFormulaVal
# preds_raw = model.predict(train_data, prediction_type='RawFormulaVal')
num_boost_round = 3000
early_stopping_rounds = 100
verbose_eval = 10

params = []

param1 = {
    'boosting': 'dart',

    'learning_rate': 0.5,
    'num_leaves': 15,
    'max_depth': 5,

    'lambda_l1': 0,
    'lambda_l2': 0,
    'max_bin': 15,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param2 = {
    'boosting': 'goss',

    'learning_rate': 0.3,
    'num_leaves': 15,
    'max_depth': 6,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 15,


    'bagging_fraction': 1,
    'bagging_freq': 0,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param3 = {
    'boosting': 'rf',

    'learning_rate': 0.3,
    'num_leaves': 511,
    'max_depth': 10,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 63,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param4 = {
    'boosting': 'gbdt',

    'learning_rate': 0.032,
    'num_leaves': 750,
    'max_depth': 50,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 172,


    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 2,
}
params.append(param1)
params.append(param2)
params.append(param3)
params.append(param4)


on1 = [
    'target',
    'song_length_log10',
    # 'ISCZ_genre_ids_log10',
    'ISC_artist_name_log10',
    # 'ISCZ_composer_log10',
    # 'ISC_lyricist_log10',
    # 'ISC_song_country_ln',
    'ITC_song_id_log10_1',
    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    'ITC_artist_name_log10_1',
    'ITC_composer_log10_1',
    'ITC_lyricist_log10_1',
    # 'ITC_song_year_log10_1',
    # 'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    # 'ITC_top3_in_song_log10_1',
    'ITC_msno_log10_1',
]
on2 = [
    'target',
    'artist_name',
    'song_year',
    'msno',
]
on3 = [
    'song_id',
    'target',
    'source_system_tab',
    'source_screen_name',
]
on4 = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    # 'language',
    'top3_in_song',
    'ITC_song_id_log10_1',
]
ons = []
ons.append(on1)
ons.append(on2)
ons.append(on3)
ons.append(on4)


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




train_data = dfs[0].drop('target', axis=1)
print(dfs[0].dtypes)
print(dfs[0].dtypes == 'category')
print(np.where(dfs[0].dtypes == 'category'))
print(np.where(dfs[0].dtypes == 'category')[0])
cat_feature = np.where(train_data.dtypes == 'category')[0]
train_labels = dfs[0]['target']
model = CatBoostClassifier(
    iterations=5, learning_rate=0.3,
    depth=16, logging_level='Verbose',
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',
    od_wait=40,
)
val_data = val.drop('target', axis=1)
val_y = val['target']
# Fit model
model.fit(
    train_data, train_labels,
    cat_features=cat_feature,
    eval_set=(val_data, val_y)
)
# Get predicted classes
preds_class = model.predict(val.drop('target', axis=1))
# Get predicted probabilities for each class
preds_proba = model.predict_proba(val.drop('target', axis=1))

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#
clfs = [
      RandomForestClassifier(n_estimators=100, verbose=10, n_jobs=-1, criterion='gini'),
    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
    GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
]
# for clf in clfs:
#     since = time.time()
#     print(clf)
#     t = dfs[0][on1].drop('target', axis=1)
#     y = dfs[0]['target']
#     clf.fit(t, y)
# print(preds_class)
# print(preds_proba)
tt = np.array(preds_proba).T

# print(roc_auc_score(val['target'], tt[0]))
print(roc_auc_score(val['target'], tt[1]))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('done')