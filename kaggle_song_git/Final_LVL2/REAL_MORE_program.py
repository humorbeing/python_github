import sys
sys.path.insert(0, '../')
from me import *
from NODE import *
import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


since = time.time()
K = 3
print()
print('This is [no drill] training.')
print()
for level in range(3, 100):
    print('ON LEVEL:', level)
    load_name = 'lvl'+str(level)+'.csv'
    dfs, test = read_lvl(load_name, level)
    show_df(dfs[0])
    show_df(test)

    dfs_collector = []
    for i in range(K):
        dc = pd.DataFrame()
        dc['target'] = dfs[i]['target']
        dfs_collector.append(dc)

    test_collector = pd.DataFrame()
    test_collector['id'] = test['id']

    fake = False
    # !!!!!!!!!!!!!!!!!!!!!!!!!

    dfs_collector, test_collector, r = LogisticRegression_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = SGDClassifier_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = GaussianNB_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = CV_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = RF_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))


    # ------------------------------------------------------------------------------------------


    dfs_collector, test_collector, r = Neural_net_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = Dart_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = GOSS_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = RF_LIGHT_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))

    dfs_collector, test_collector, r = LGBT_NODE(
        K, dfs, dfs_collector, test, test_collector
    )
    for j in range(K):
        print('AUC train', roc_auc_score(dfs_collector[j]['target'], dfs_collector[j][r]))
    if fake:
        print('AUC  test', roc_auc_score(test_collector['target'], test_collector[r]))



    # !!!!!!!!!!!!!!!!!!!!!!!!!



    show_df(test_collector)

    # print(test_collector)
    # print(test_collector.tail())
    save_name = 'lvl'+str(level+1)
    save_here = '../saves/feature/level'+str(level+1)+'/'
    for i in range(K):
        save_train = save_here + 'train' + str(i+1) + '/'
        save_df(dfs_collector[i], name=save_name,
                save_to=save_train)

    save_df(test_collector, name=save_name,
                save_to=save_here+'test/')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


