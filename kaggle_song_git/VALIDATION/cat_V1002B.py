import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle
from catboost import CatBoostClassifier


since = time.time()
result = {}
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_top2.csv'

df = read_df(load_name)
show_df(df)

train, val = fake_df(df)
del df

X = train.drop('target', axis=1)
Y = train['target']
vX = val.drop('target', axis=1)
vY = val['target']
cat_feature = np.where(X.dtypes == 'category')[0]
del train, val

model = CatBoostClassifier(
    iterations=200, learning_rate=0.3,
    depth=12, logging_level='Verbose',
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',
    od_wait=100,
)
model.fit(
    X, Y,
    cat_features=cat_feature,
    eval_set=(vX, vY)
)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))