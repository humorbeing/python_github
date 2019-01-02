import sys
sys.path.insert(0, '../')
from me import *
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
load_name = 'final_train_real.csv'
df = read_df(load_name)

on = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'target',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    # ------------------
    'top2_in_song',
    # 'language',
    # 'top3_in_song',

    # ------------------
    'source_system_tab',
    # 'ITC_source_system_tab_log10_1',
    # 'ISC_song_country_ln',

    # ------------------
    # 'membership_days',
    # 'ISC_song_year',
    # 'OinC_language',
]
df = df[on]
show_df(df)

# !!!!!!!!!!!!!!!!!!!!!!!!!

iterations = 300
learning_rate = 0.3
depth = 6
estimate = 0.6925

model, cols = train_cat(df, iterations,
                        learning_rate=learning_rate,
                        depth=depth)
del df

# !!!!!!!!!!!!!!!!!!!!!!!!!


print('training complete.')
print('Making prediction')

load_name = 'final_test_real.csv'
df = read_df(load_name)

cols.remove('target')
cols.append('id')
df = df[cols]

test = df.drop(['id'], axis=1)
ids = df['id'].values
del df

p = cat_predict(model, test)
del model

print('prediction done.')
print('creating submission')
subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = p
del p


model_time = str(int(time.time()))
model_name = '_cat_'
model_name = '[]_'+str(estimate)+model_name
model_name = model_name + '_' + model_time
subm.to_csv(save_dir+'submission/'+model_name+'.csv',
            index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


