import sys
sys.path.insert(0, '../')
from me import *
from real_cat_lvl2 import *
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
read_from = '../saves/feature/level1/'
file_name = 'Cat.csv'

dfs, test = read_lvl1(file_name)
print(test.head())
subm = pd.DataFrame()

subm['id'] = test['id']
a = test['CatR_top2_2']
a_max = np.max(a)
a_min = np.min(a)
b = (a-a_min)/(a_max-a_min)
subm['target'] = b

print(subm.head())
print(subm.tail())
estimate = 'CatR_top2_2'
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


