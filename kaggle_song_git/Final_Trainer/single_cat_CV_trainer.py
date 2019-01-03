import sys
sys.path.insert(0, '../')
from me import *
from model_cat_level1 import *
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
dfs = read_df(load_name)

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
dfs = dfs[on]
show_df(dfs)


load_name = 'final_test_real.csv'
val = read_df(load_name)

on = [
    'msno',
    'song_id',
    'source_screen_name',
    'source_type',
    'id',
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
val = val[on]
show_df(val)


K = 3
dfs = divide_df(dfs, K)
dcs = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dcs.append(dc)

vc = pd.DataFrame()
vc['id'] = val['id']


# !!!!!!!!!!!!!!!!!!!!!!!!!

dcs, vc, r = cat_on_top2_real(K, dfs, dcs, val, vc)

# !!!!!!!!!!!!!!!!!!!!!!!!!

print(vc.head())
print(vc.tail())
ids = vc['id'].values
del val

print('prediction done.')
print('creating submission')
subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = vc[r]

print(subm.head())
print(subm.tail())
estimate = '6360'
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


