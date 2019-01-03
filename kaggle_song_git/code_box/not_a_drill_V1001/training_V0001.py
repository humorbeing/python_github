import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
print('What we got:')
print(df.dtypes)
print('number of columns:', len(df.columns))


num_boost_round = 300
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'verbose': -1,
    'num_leaves': 127,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.9,
    # 'feature_fraction_seed': 1,
    'max_bin': 15,
    'max_depth': -1,
}
df = df[['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         'song_count',
         'member_count',
         'song_year'
         ]]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print()
print('After selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()
print()


X_tr = df.drop(['target'], axis=1)
Y_tr = df['target'].values
del df

train_set = lgb.Dataset(X_tr, Y_tr)
del X_tr, Y_tr

print('Training...')
print()
model = lgb.train(params,
                  train_set,
                  num_boost_round=num_boost_round,
                  )
model_name = 'model_V1001'
pickle.dump(model, open(save_dir+model_name+'.save', "wb"))
print('model saved as: ', save_dir, model_name)
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


