import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier


print()
print('This is [no drill] training.')
print()
since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set.csv'
df = read_df(load_name)
c = ['song_id', 'msno']
df = add_ITC(df, c, real=True)

on = [
    'msno',
    'song_id',
    'target',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    'song_year',
    'top3_in_song',
]
df = df[on]

show_df(df)

num_boost_round = 995
estimate = 0.6887  # make sure put in something here

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 511
max_depth = -1

max_bin = 255
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

    'max_bin': max_bin,
    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}

# !!!!!!!!!!!!!!!!!!!!!!!!!


model, cols = train_light(params, df)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
iterations = 5
learning_rate = 0.3
depth = 16
estimate = 0.6887

model, cols = train_cat(df, iterations,
                        learning_rate=learning_rate,
                        depth=depth)

# !!!!!!!!!!!!!!!!!!!!!!!!!

del df
print('training complete.')
print('Making prediction.')

load_name = 'test_set.csv'
df = read_df(load_name)
df = add_ITC(df, c, real=True)
cols.remove('target')
cols.append('id')
df = df[cols]
X_test = df.drop(['id'], axis=1)
ids = df['id'].values
p = model.predict_proba(X_test)
tt = np.array(p).T[1]
print('prediction done.')
print('creating submission')
subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = tt
del p, tt


model_time = str(int(time.time()))
model_name = '_Light_'+boosting
model_name = '[]_'+str(estimate)+model_name
model_name = model_name + '_' + model_time
subm.to_csv(save_dir+'submission/'+model_name+'.csv',
            index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')

pickle.dump(model, open(save_dir+'model/'+model_name+'.model', "wb"))
print('model saved as: ', save_dir+'model/'+model_name+'.model')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


