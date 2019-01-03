import sys
sys.path.insert(0, '../')
from me import *
# from fake_cat_lvl2 import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier



since = time.time()
print()
K = 3
print('This is [no drill] training.')
print()
data_dir = '../data/'
save_dir = '../saves/'

read_from = '../saves/feature/level1/'

load_name = 'Cat.csv'
dfs1, test1 = read_lvl1(load_name)

load_name = 'Cat_XX.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)



load_name = 'L_all.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)

load_name = 'L_rest.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)



load_name = 'L_XX_rest.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)

load_name = 'Ldrt_top2_1.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)

load_name = 'Lgbt_top2_1.csv'
dfs2, test2 = read_lvl1(load_name)

print(test1.head())
print('-'*20)
print(test2.head())
test1 = merge_id(test1, test2)
print('-'*20)
print(test1.head())


print(dfs1[0].head())
print('-'*20)
print(dfs2[0].head())
print('-'*20)
dfs_temp = []
for i in range(K):
    df = merge_target(dfs1[i], dfs2[i])
    dfs_temp.append(df)
dfs1 = dfs_temp
print(dfs1[0].head())
print('-'*20)


# read_from = '../fake/saves/feature/level1/'
# file_name = 'L_rest.csv'
#
# dfs1, test1 = read_fake_lvl1(file_name)
#
# file_name = 'L_XX_rest.csv'
# dfs2, test2 = read_fake_lvl1(file_name)
#
# t1 = merge_target(test1, test2)
# del test1, test2
#
#
# d1 = []
# for i in range(K):
#     df = merge_target(dfs1[i], dfs2[i])
#     d1.append(df)
# del dfs1, dfs2
#
#
# file_name = 'Cat.csv'
# dfs3, test3 = read_fake_lvl1(file_name)
#
# file_name = 'Lgbt_top2_1.csv'
# dfs4, test4 = read_fake_lvl1(file_name)
#
#
#
# t2 = merge_target(test3, test4)
# del test3, test4
#
#
# d2 = []
# for i in range(K):
#     df = merge_target(dfs3[i], dfs4[i])
#     d2.append(df)
# del dfs3, dfs4
#
#
# test = merge_target(t1, t2)
# del t1, t2
#
#
# dfs = []
# for i in range(K):
#     df = merge_target(d1[i], d2[i])
#     dfs.append(df)
# del d1, d2


# test2.drop('target', axis=1, inplace=True)
# test = pd.merge(test1, test2, left_index=True, right_index=True)
# print(test.head())
#
# file_name = 'L_XX_rest.csv'
# dfs2, test2 = read_fake_lvl1(file_name)
# test = merge_target(test1, test2)
print(test.head())
print('<>'*10)
print('<>'*10)
#
#
#
# dfs = []
# for i in range(K):
#     dfs2[i].drop('target', axis=1, inplace=True)
#     df = pd.merge(dfs1[i], dfs2[i], left_index=True, right_index=True)
#     dfs.append(df)
#
# print(dfs[0].head())
# file_name = 'L_XX_rest.csv'
# dfs2, test2 = read_fake_lvl1(file_name)
# dfs = []
# for i in range(K):
#     df = merge_target(dfs1[i], dfs2[i])
#     dfs.append(df)
#
print(dfs[0].head())




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


