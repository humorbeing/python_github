import sys
sys.path.insert(0, '../')
from me import *
from fake_cat_lvl2 import *
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
read_from = '../fake/saves/feature/level1/'
file_name = 'L_rest.csv'
K = 3
dfs, test = merge_fake()

print(test.head())
show_df(test)


K = 3
# dfs = divide_df(train, K)
# del train
dfs_collector = []
for i in range(K):
    dc = pd.DataFrame()
    dc['target'] = dfs[i]['target']
    dfs_collector.append(dc)

test_collector = pd.DataFrame()
test_collector['target'] = test['target']


# !!!!!!!!!!!!!!!!!!!!!!!!!

dfs_collector, test_collector, r = CatC_top2_1(
    K, dfs, dfs_collector, test, test_collector
)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(test['target'], test_collector[r]))

dfs_collector, test_collector, r = CatR_top2_1(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))


#-----------------------------

dfs_collector, test_collector, r = CatC_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))
dfs_collector, test_collector, r = CatR_top2_2(
    K, dfs, dfs_collector, test, test_collector
)
print(roc_auc_score(test['target'], test_collector[r]))


# !!!!!!!!!!!!!!!!!!!!!!!!!

print(test_collector.head())
print(test_collector.tail())
save_name = 'Cat'
save_here = '../fake/saves/feature/level2/'
for i in range(K):
    save_train = save_here + 'train' + str(i+1) + '/'
    save_df(dfs_collector[i], name=save_name,
            save_to=save_train)

save_df(dfs_collector[i], name=save_name,
            save_to=save_here+'test/')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


