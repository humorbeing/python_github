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


dcs, vc, r = goss_on_top2_2(K, dfs, dcs, val, vc)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(val['target'], vc[r]))


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('done')